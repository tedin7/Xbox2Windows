"""Run the shared C++ direct-DXIL frontend on a source-built scalar/LDS request."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request


def scalar(op, dst, a, b):
    return 0x80000000 | op << 23 | dst << 16 | b << 8 | a


def sop1(op, dst, src):
    return 0xbe800000 | dst << 16 | op << 8 | src


def vmove(dst, src):
    return 0x7e000200 | dst << 17 | src


def ds(op, addr, data0=0, data1=0, dst=0, offset=0):
    return [0xd8000000 | op << 18 | offset, dst << 24 | data1 << 16 | data0 << 8 | addr]


def program(mutant):
    # Root table s4, group.x in s5, 64 bytes per group. Keep lane zero active.
    words = [sop1(3, 20, 4), sop1(3, 21, 128), 0xf4080000 | 32 << 6 | 10, 0xfa000020,
             scalar(0x1e, 7, 5, 134), vmove(8, 7), sop1(3, 126, 129)]
    # src = 0x81234567 ^ group.x. High signed/unsigned multiplication and bitset
    # must preserve zero SCC. Make both the bit position and destination s11.
    words += [scalar(0x12, 16, 5, 255), 0x81234567, 0xbf068180,
              scalar(0x36 if mutant else 0x35, 10, 16, 193), vmove(1, 10),
              scalar(0x36, 10, 16, 193), vmove(2, 10),
              sop1(3, 11, 255), 0x8000001f, sop1(0x1b, 11, 11), vmove(3, 11),
              sop1(0x3b, 16, 16), sop1(0x1d, 16, 134), sop1(0x1c, 16, 191),
              sop1(0x1e, 16, 190), vmove(4, 16), vmove(5, 17),
              # A nonempty mask (7 ones shifted by 3) must preserve SCC=0.
              scalar(0x24, 24, 135, 131), vmove(16, 24),
              scalar(0x0a, 10, 129, 128), vmove(6, 10),
              sop1(0x34, 10, 255), 0x80000000, vmove(7, 10),
              scalar(0x0a, 10, 129, 128), vmove(9, 10)]
    # Use distinct b64 tuples, reverse offset order on read and alias the
    # first result with the read's address. Returning subtract aliases data0.
    words += [vmove(10, 128), vmove(11, 129), vmove(12, 130), vmove(13, 131), vmove(14, 132),
              *ds(0x4e, 10, 11, 13, offset=0x0301), *ds(0x77, 10, dst=10, offset=0x0103),
              vmove(14, 128), vmove(15, 129), *ds(0x21, 14, 15, dst=15, offset=8), 0xbfa3ffe3,
              *ds(0x36, 14, dst=14, offset=8)]
    for index, vgpr in enumerate([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 14, 16]):
        words += [0xe0701000 | 4 * index, 0x80080008 | vgpr << 8]
    return words + [0xbbfd0000, 0xbf8a0000, 0xbf810000]  # drain stores, then group barrier


def reference(group):
    a = 0x81234567 ^ group
    replicated = sum(3 << (2 * i) for i in range(32) if a & (1 << i))
    replicated = (replicated | 1 << 6 | 1 << 62) & ~(1 << 63)
    return [a * 0xffffffff >> 32, 0, 31, replicated & 0xffffffff, replicated >> 32,
            0, 0x80000000, 1, 3, 4, 1, 2, 1, 0, 127 << 3]


def built_root_readback(work, gpu, sdk):
    source, root = work / "built.hlsl", work / "built.root"
    source.write_text('#define SIG "SRV(t0),DescriptorTable(UAV(u0,numDescriptors=8))"')
    subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, source], check=True)
    sa = struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1, 0, 0x80 | 7 << 1)
    for offset in (96, 92):
        # Build the byte-address V# in s16:19 from root SRV s4:5. Each
        # group selects an aligned index. Group 1 ends at the view limit.
        words = [sop1(4, 16, 4), sop1(3, 18, 255), 132, sop1(3, 19, 255), 0x31016fac,
                 sop1(3, 20, 6), sop1(3, 21, 128), 0xf4080000 | 40 << 6 | 10, 0xfa000020,
                 scalar(0x1e, 11, 7, 130), scalar(0xe, 11, 11, 132),
                 0xf42c0808, 11 << 25 | offset, scalar(0x1e, 12, 7, 133), vmove(8, 12),
                 sop1(3, 126, 129)]
        for j in range(8):
            words += [vmove(9, 32 + j), 0xe0701000 | 4 * j, 0x800a0908]
        marker = write_request(work, words + [0xbf810000], root.read_bytes(), sa)
        dest = work / "built-readback.bin"
        subprocess.run([gpu, sdk, marker, work / "cache", "0", dest, "--direct"], check=True)
        expected = [0x69696969] * 2064
        for group in range(2):
            for j in range(8):
                expected[8 + 8 * group + j] = 0x51aa0000 + offset // 4 + group + j
        actual = struct.unpack("<2064I", dest.read_bytes())
        assert list(actual) == expected, [(i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b][:10]
    print("Root-built scalar x8 matched 16 words and all guards at each offset; shifting by one dword changed all 16.")


    # The byte address in v8 overlaps the first destination of the x3 load.
    # Select a different source window in each group and keep all guard words.
    words = [sop1(3, 16, 4), scalar(0x10, 17, 5, 255), 0x10000,
             sop1(3, 18, 193), sop1(3, 19, 255), 0x11014fac,
             sop1(3, 20, 6), sop1(3, 21, 128), 0xf4080000 | 40 << 6 | 10, 0xfa000020,
             scalar(0x1e, 12, 7, 132), vmove(8, 12), vmove(11, 12),
             # The replay fixture creates the smallest raw SRV requested by the
             # program. A scalar load at byte 24 provisions the seven-word view.
             0xf4200000 | (25 << 6) | 8, 0xfa000018,
             0xe03c1000, 0x80040808, sop1(3, 126, 129)]
    for j in range(3):
        words.extend([0xe0701000 | 4*j, 0x800a000b | (8+j) << 8])
    marker = write_request(work, words + [0xbf810000], root.read_bytes(), sa)
    subprocess.run([gpu, sdk, marker, work / "cache", "0", dest, "--direct"], check=True)
    expected = [0x69696969] * 2064
    for group in range(2):
        for j in range(3):
            expected[8 + group * 4 + j] = 0x51aa0000 + group * 4 + j
    actual = struct.unpack("<2064I", dest.read_bytes())
    assert list(actual) == expected, [(i,a,b) for i,(a,b) in enumerate(zip(actual,expected)) if a!=b][:10]
    print("Root descriptor with byte stride one: x3 load matches both source windows and all guards.")


def vector_carry_readback(work, gpu, sdk):
    def vop2(op, dst, a, b):
        return op << 25 | dst << 17 | b << 9 | a

    forms = [(op, wide) for op in (0x128, 0x129, 0x12a) for wide in (False, True)]
    forms += [(op, True) for op in (0x30f, 0x310, 0x319)]
    full = (1 << 32) - 1
    active = 0xdedebdbd
    cin = 0x5a3c96e1
    # Each lane owns 128 bytes. Generate (a,b) of (0,0), (~0,0),
    # (0,1), (~0,1) repeating, covering carry on either addition step.
    words = [sop1(3, 20, 4), sop1(3, 21, 128), 0xf408080a, 0xfa000020,
             scalar(0x1e, 7, 5, 140), vop2(0x1a, 8, 135, 0), vop2(0x25, 8, 7, 8),
             vop2(0x1a, 1, 159, 0), vop2(0x18, 1, 159, 1), vop2(0x16, 2, 129, 0),
             vop2(0x1b, 2, 129, 2)]
    expected = [0x69696969] * 2064
    for idx, (op, wide) in enumerate(forms):
        ci = op in (0x128, 0x129, 0x12a)
        add = op in (0x128, 0x30f)
        reverse = op in (0x12a, 0x319)
        mask = 30 if wide else 106
        words += [vmove(10, 257), sop1(3, mask, 255), cin, sop1(3, 126, 255), active]
        if wide:
            words += [0xd4000000 | op << 16 | mask << 8 | 10, 266 | 258 << 9 | (mask if ci else 0) << 18]
        else:
            words.append(vop2(op - 0x100, 10, 266, 2))
        words += [sop1(3, 126, 193), vmove(11, mask), 0xe0701000 | 8 * idx, 0x80080a08,
                  0xe0701004 | 8 * idx, 0x80080b08]
        ballot = 0
        for lane in range(32):
            a, b = full * (lane & 1), (lane >> 1) & 1
            bit = int(ci) * ((cin >> lane) & 1)
            exact = a + b + bit if add else (b - a if reverse else a - b) - bit
            enabled = (active >> lane) & 1
            ballot |= int(enabled and (exact < 0 or exact > full)) << lane
            for group in range(2):
                pos = 8 + 32 * (group * 32 + lane) + 2 * idx
                expected[pos] = exact & full if enabled else a
        for lane in range(64):
            expected[8 + 32 * lane + 2 * idx + 1] = ballot
    sa = struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1, 0, 0x80 | 5 << 1)
    marker = write_request(work, words + [0xbf810000], (work / "root.bin").read_bytes(), sa)
    dest = work / "carry-readback.bin"
    subprocess.run([gpu, sdk, marker, work / "cache", "0", dest, "--direct"], check=True)
    actual = struct.unpack("<2064I", dest.read_bytes())
    assert list(actual) == expected, [(i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b][:10]
    print("Nine carry/borrow encodings matched 1152 result and ballot words plus all guards.")


def half_mad_readback(work, gpu, sdk):
    words = [sop1(3, 20, 4), sop1(3, 21, 128), 0xf408080a, 0xfa000020,
             scalar(0x1e, 7, 5, 136), vmove(8, 7), sop1(3, 126, 129)]
    a, b = 0x80007fff, 0xfffe0003
    words += [vmove(1, 255), a, vmove(2, 255), b]
    values = []
    for signed in (False, True):
        for addend in (0x80000010, 0x7ffffff0, 0xfffffffe):
            for clamp in (False, True):
                for halves in range(4):
                    # Alias the dword addend and the result VGPR.
                    inst = 0xd4000000 | (0x375 if signed else 0x373) << 16 | clamp << 15 | halves << 11 | 3
                    offset = 4 * len(values)
                    words += [vmove(3, 255), addend, inst, 257 | 258 << 9 | 259 << 18,
                              0xe0701000 | offset, 0x80080308]
                    x = (a >> (16 * (halves & 1))) & 65535
                    y = (b >> (16 * (halves >> 1))) & 65535
                    z = addend
                    if signed:
                        x -= (x >> 15) << 16
                        y -= (y >> 15) << 16
                        z -= (z >> 31) << 32
                    exact = x * y + z
                    if clamp:
                        low, high = (-2**31, 2**31-1) if signed else (0, 2**32-1)
                        exact = min(high, max(low, exact))
                    values.append(exact & 0xffffffff)
    sa = struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1, 0, 0x80 | 5 << 1)
    marker = write_request(work, words + [0xbf810000], (work / "root.bin").read_bytes(), sa)
    dest = work / "half-mad-readback.bin"
    subprocess.run([gpu, sdk, marker, work / "cache", "0", dest, "--direct"], check=True)
    expected = [0x69696969] * 2064
    for group in range(2):
        start = 8 + group * 64
        expected[start:start + len(values)] = values
    actual = struct.unpack("<2064I", dest.read_bytes())
    assert list(actual) == expected, [(i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b][:10]
    print("Halfword MAD matched 96 signed/unsigned results across half-selects, wrap, saturation and aliasing.")


def descriptor_scc_readback(work, gpu, sdk):
    # Use the table only as provenance for s[16:19]. Its guest address bits
    # are unavailable in the shader. OOB_SELECT=3 always sets SCC, whereas
    # the masked address's SCC must be dead at SAVEEXEC.
    words = [sop1(3, 20, 4), sop1(3, 21, 128),
             0xf408040a, 0xfa000000, 0xf408080a, 0xfa000020,
             scalar(0x1e, 7, 5, 132), vmove(8, 7),
             scalar(0xe, 23, 19, 255), 0xcfffffff, 0xbf068180,
             scalar(0x10, 23, 23, 255), 0x30000000, sop1(3, 25, 253),
             scalar(0xe, 17, 17, 255), 0xc000ffff, sop1(0x3c, 24, 5),
             sop1(3, 26, 253), sop1(3, 126, 129),
             vmove(9, 25), vmove(10, 26), 0xe0701000, 0x80080908, 0xe0701004, 0x80080a08,
             # -0 has class bit 5. BYTE_3 selects this bit only in group 1,
             # BYTE_0 selects it in both. Only lane zero contributes.
             scalar(0x1e, 11, 5, 157), scalar(0x10, 11, 11, 160), vmove(5, 11),
             0x7d100af9, 0x03968e80, vmove(9, 14), 0xe0701008, 0x80080908,
             0x7d100af9, 0x00968e80, vmove(9, 14), 0xe070100c, 0x80080908,
             0xbf810000]
    marker = write_request(work, words, (work / "root.bin").read_bytes(),
                           struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1, 0, 0x80 | 5 << 1))
    dest = work / "descriptor-scc.bin"
    subprocess.run([gpu, sdk, marker, work / "cache", "0", dest, "--direct"], check=True)
    expected = [0x69696969] * 2064
    expected[8:12] = [1, 0, 0, 1]
    expected[12:16] = [1, 1, 1, 1]
    actual = struct.unpack("<2064I", dest.read_bytes())
    assert list(actual) == expected, [(i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b][:10]
    print("Descriptor OOB OR / SAVEEXEC SCC and both -0 SDWA class-mask selections matched eight words and all guards.")


def main(gpu, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        src, root = work / "root.hlsl", work / "root.bin"
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, src], check=True)
        sa = struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1, 0, 0x808a)
        expected = [0x69696969] * 2064
        for group in range(2):
            start = 8 + 16 * group
            expected[start:start + 15] = reference(group)
        for mutant in (False, True):
            marker = write_request(work, program(mutant), root.read_bytes(), sa)
            dest = work / "result.bin"
            result = subprocess.run([gpu, sdk, marker, work / "cache", "0", dest, "--direct"])
            if result.returncode == 77:
                sys.exit(77)
            result.check_returncode()
            actual = struct.unpack("<2064I", dest.read_bytes())
            mismatches = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
            assert mismatches == ([8, 24] if mutant else []), [(i, actual[i], expected[i]) for i in mismatches[:10]]
        print("Two direct-DXIL workgroups matched 30 result words and all guards; the signed-multiply control differed twice.")
        built_root_readback(work, gpu, sdk)
        vector_carry_readback(work, gpu, sdk)
        half_mad_readback(work, gpu, sdk)
        descriptor_scc_readback(work, gpu, sdk)


if __name__ == "__main__":
    main(*sys.argv[1:])
