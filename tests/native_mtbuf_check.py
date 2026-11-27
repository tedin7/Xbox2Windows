"""Decode explicit MTBUF formats through the runtime's direct C++ compute backend."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from native_compute_clear_check import write_request
from native_scalar_lds_check import scalar, sop1, vmove


# format, channels actually requested, immediate byte displacement (plus group.x)
CASES = [(24, 2, 0), (24, 2, 4), (56, 4, 3), (60, 4, 7), (13, 1, 12),
         (71, 4, 12), (75, 4, 4), (2, 1, 7), (12, 1, 7), (25, 2, 3), (24, 2, 8188)]
CASES += [(fmt, 4, 1) for fmt in range(44, 56)]
PACKED = struct.pack("<6I", 0x7fff8000, 0x40008001, 0x007fff81, 0xfc003c00, 0x0001bc00, 0x42987ec5)


# The byte-address rewrite of the SRV descriptor s[32:35] in place (LLVM 15): s_waitcnt lgkmcnt(0);
# s_bfe_u32 s40, s33, 0xe0010 (stride); s_max_u32 s40, s40, 1; s_mul_i32 s34, s34, s40 (byte count);
# s_and_b32 s33, s33, 0xc000ffff (stride cleared); s_and_b32 s35, s35, 0xcfffffff; s_or_b32 s35, s35, 0x30000000.
BYTE_WINDOW = [0xbf8cc07f, 0x93a8ff21, 0x000e0010, 0x84a88128, 0x93222822, 0x8721ff21, 0xc000ffff,
               0x8723ff23, 0xcfffffff, 0x8823ff23, 0x30000000]


def program(unsigned, window=False):
    # Load a raw SRV and UAV from separate table parameters. One writer per group.
    # With window, every MTBUF load reads the byte window of the SRV (same raw view, same expected values).
    words = [sop1(3, 20, 4), sop1(3, 21, 128), sop1(3, 22, 5), sop1(3, 23, 128),
             0xf4080000 | 32 << 6 | 10, 0xfa000020, 0xf4080000 | 36 << 6 | 11, 0xfa000020,
             *(BYTE_WINDOW if window else ()),
             scalar(0x1e, 7, 6, 137), vmove(8, 7), sop1(3, 126, 129)]
    pos = 0
    for index, (fmt, channels, disp) in enumerate(CASES):
        if unsigned and index == 0:
            fmt = 23
        if disp > 4095:
            words += [scalar(0, 9, 6, 255), disp, vmove(10, 9)]
            disp = 0
        else:
            words += [vmove(10, 6)]
        words += [0xe8001000 | fmt << 19 | (channels - 1) << 16 | disp,
                  0x80080a0a]  # v10 is both byte offset and first output
        for component in range(channels):
            words += [0xe0701000 | 4 * pos, 0x80090008 | (10 + component) << 8]
            pos += 1
    return words + [0xbf810000]


def fbits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def expected(group):
    result = []
    data = (PACKED * 342)[:8192]
    for fmt, count, disp in CASES:
        if 44 <= fmt <= 55:
            fields = (2, 10, 10, 10) if fmt < 50 else (10, 10, 10, 2)
            number = (fmt - 44) % 6
            bits = int.from_bytes(data[disp + group:disp + group + 4], "little")
            for width in fields[:count]:
                value = bits & ((1 << width) - 1)
                bits >>= width
                if number in (1, 3, 5) and value >> (width - 1):
                    value -= 1 << width
                if number == 0:
                    value = fbits(value / ((1 << width) - 1))
                elif number == 1:
                    value = fbits(max(-1, value / ((1 << (width - 1)) - 1)))
                elif number in (2, 3):
                    value = fbits(value)
                result.append(value & 0xffffffff)
            continue
        width = 4 if fmt == 75 else 1 if fmt in (2, 56, 60) else 2
        if disp + group + width * count > len(data):
            result.extend([0] * count)
            continue
        for c in range(count):
            at = disp + group + c * width
            v = int.from_bytes(data[at:at + width], "little", signed=fmt in (2, 12, 24))
            if fmt in (13, 71):
                v = fbits(struct.unpack_from("<e", data, at)[0])
            elif fmt in (2, 24):
                v = fbits(max(-1, v / ((1 << (width * 8 - 1)) - 1)))
            elif fmt == 56:
                v = fbits(v / 255)
            elif fmt == 25:
                v = fbits(v)
            result.append(v & 0xffffffff)
    return result


def main(cli, gpu, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        src, root = work / "root.hlsl", work / "root.bin"
        src.write_text('#define SIG "DescriptorTable(SRV(t0,numDescriptors=8)),DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk)/"dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, src], check=True)
        sa = struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1, 0, 0x8c)
        reference = [0x69696969] * 2064
        for group in range(2):
            start = 8 + 128 * group
            row = expected(group)
            reference[start:start + len(row)] = row
        for unsigned, window in ((False, False), (True, False), (False, True)):
            words = program(unsigned, window)
            marker = write_request(work, words, root.read_bytes(), sa)
            dest = work / "result.bin"
            run = subprocess.run([gpu, sdk, marker, work/"cache", "0", dest, "--direct"])
            if run.returncode == 77:
                sys.exit(77)
            run.check_returncode()
            actual = struct.unpack("<2064I", dest.read_bytes())
            differences = [i for i, (a, b) in enumerate(zip(actual, reference)) if a != b]
            if unsigned:
                assert differences and set(differences) <= {8, 9, 136, 137}, differences
            else:
                assert not differences, [(i, hex(actual[i]), hex(reference[i])) for i in differences]
        words = program(False)  # the guards below change the plain (non-window) program
        insn = next(i for i, w in enumerate(words) if w >> 26 == 0x3a)
        for op, error in ((words[insn] | (1 << 13), "without index"),
                          (0xe8001000 | 24 << 19 | 3 << 16, "absent component")):
            bad = words.copy()
            bad[insn] = op
            marker = write_request(work, bad, root.read_bytes(), sa)
            reject = subprocess.run([cli, sdk, work/"cache", "gfx1013-direct-cs@0", marker, work/"bad.cso"],
                                    capture_output=True, text=True)
            assert reject.returncode and error in reject.stderr and not (work/"bad.cso").exists(), reject.stderr
        print(f"MTBUF matched {2 * len(row)} result words, also through the SRV byte window, and all guards; "
              "changed format, index and component controls passed.")


if __name__ == "__main__":
    main(*sys.argv[1:])
