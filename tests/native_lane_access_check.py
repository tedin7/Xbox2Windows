"""Compare source-built direct DXIL lane accesses with a CPU wave model."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_scalar_lds_check import scalar, sop1, vmove


def vop2(op, dst, src0, src1):
    return op << 25 | dst << 17 | src1 << 9 | src0


def readlane(dst, vgpr, selector):
    return [0xd7600000 | dst, (256 + vgpr) | selector << 9]


def writelane(dst, source, selector):
    return [0xd7610000 | dst, source | selector << 9]


def readfirstlane(dst, vgpr):
    return 0x7e000400 | dst << 17 | (256 + vgpr)


def launch(threads):
    return struct.pack("<7I", 0, 0, 0, threads | 1 << 16, 1 | (1 << 20), 0, 0x8a)


def program():
    # v8 = byte address of a 16-word record per lane in each workgroup.
    words = [sop1(3, 20, 4), sop1(3, 21, 128), 0xf408080a, 0xfa000020,
             scalar(0x1e, 7, 5, 139), vop2(0x1a, 8, 134, 0), vop2(0x25, 8, 7, 8),
             vmove(1, 256)]
    for slot, (lane, value) in enumerate(((0, 11), (5, 22), (31, 33))):
        words += [sop1(3, 10 + slot, 128 + value), *writelane(1, 10 + slot, 128 + lane)]
    for slot, lane in enumerate((0, 5, 63)):
        words += readlane(14 + slot, 1, 128 + lane)

    # The group ID changes the SGPR selector. Bit 5 must be ignored on wave32.
    words += [scalar(1, 13, 169, 5), *writelane(1, 170, 13), *readlane(17, 1, 13)]
    # A manual mutant that lets v_writelane honour EXEC must lose the value 59.
    # A manual mutant that reads lane ^ 1 must return the neighbouring lane's value.
    words += [sop1(3, 126, 129), vmove(1, 175), *writelane(1, 187, 151),
              *readlane(18, 1, 151), sop1(3, 126, 193)]
    words += [sop1(3, 19, 255), 0xa5a5a5a5, *writelane(1, 19, 142), *readlane(106, 1, 142),
              vmove(16, 137), vop2(1, 9, 133, 16),
              readfirstlane(106, 1), vmove(10, 106)]
    # A manual mutant that treats vcc_hi as constant 0 in wave32 must fail, and so must one that
    # reads the high word of a wave32 VCC pair as 0: the pair is s106:s107 on hardware.
    words += [*readlane(107, 1, 133), vmove(12, 107),
              readfirstlane(107, 1), vmove(13, 107),
              sop1(3, 107, 255), 0x89abcdef, vmove(14, 107),
              sop1(4, 24, 106), vmove(15, 25)]
    for slot, (kind, reg) in enumerate((("s", 14), ("s", 15), ("s", 16), ("s", 17),
                                        ("s", 18), ("v", 9), ("v", 1), ("v", 10),
                                        ("v", 12), ("v", 13), ("v", 14), ("v", 15))):
        if kind == "s":
            words.append(vmove(11, reg))
            reg = 11
        words += [0xe0701000 | 4 * slot, 0x80080008 | reg << 8]
    return words + [0xbf810000]


def reference():
    expected = [0x69696969] * 2064
    for group in range(2):
        lanes = list(range(32))
        for lane, value in ((0, 11), (5, 22), (31, 33)):
            lanes[lane] = value
        refill = [lanes[lane] for lane in (0, 5, 31)]
        selected = (41 - group) % len(lanes)
        lanes[selected] = 42
        dynamic = lanes[selected]
        exec_mask = 1
        for lane in range(len(lanes)):
            if exec_mask & (1 << lane):
                lanes[lane] = 47
        lanes[23] = 59  # Lane write ignores EXEC.
        narrowed = lanes[23]
        lanes[14] = 0xa5a5a5a5
        vcc = lanes[14]
        exec_mask = 0xffffffff
        first = next(lane for lane in range(len(lanes)) if exec_mask & (1 << lane))
        for lane, value in enumerate(lanes):
            base = 8 + 16 * (32 * group + lane)
            expected[base:base + 12] = [*refill, dynamic, narrowed, 9 if (vcc >> lane) & 1 else 5,
                                        value, lanes[first], lanes[5], lanes[first], 0x89abcdef, 0x89abcdef]
    return expected


def check_refusals(cli, sdk, work, root):
    direct = "gfx1013-direct-cs@0"
    hlsl = "gfx1013-compute-wave32@0"
    partial = "lane read/write needs full waves"
    invalid = "invalid scalar lane read destination"
    cases = [
        (direct, 30, readlane(14, 1, 13), partial),
        (direct, 30, writelane(1, 129, 159), partial),
        (direct, 62, readlane(14, 1, 13), partial),
        (direct, 32, readlane(14, 1, 257), "lane read/write needs a constant or SGPR lane"),
        (direct, 32, [0xd7610811, 129 | 128 << 9], "lane read/write encoding or operand modifiers"),
        (direct, 32, readlane(108, 1, 128), invalid),
        (direct, 32, [readfirstlane(108, 1)], invalid),
        (hlsl, 32, readlane(106, 1, 128), invalid),
        (hlsl, 32, [readfirstlane(106, 1)], invalid),
        (hlsl, 32, readlane(107, 1, 128), invalid),
        (hlsl, 32, [readfirstlane(107, 1)], invalid),
    ]
    for profile, threads, operation, diagnostic in cases:
        words = [vmove(1, 128), sop1(3, 13, 129), *operation, 0xbf810000]
        marker = write_request(work, words, root, launch(threads))
        dest = work / "refused.cso"
        result = subprocess.run([cli, sdk, work / "cache", profile, marker, dest],
                                capture_output=True, text=True)
        assert result.returncode != 0 and diagnostic in result.stdout + result.stderr, (
            profile, threads, operation, result.stdout, result.stderr)
        assert not dest.exists()


def main(cli, replay, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        source, root_file = work / "root.hlsl", work / "root.bin"
        source.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root_file, source], check=True)
        root = root_file.read_bytes()
        check_refusals(cli, sdk, work, root)
        marker = write_request(work, program(), root, launch(32))
        dest = work / "result.bin"
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", dest, "--direct"])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        actual = struct.unpack("<2064I", dest.read_bytes())
        expected = reference()
        mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
        assert not mismatches, mismatches[:12]
        print("Two workgroups matched 768 lane-access words and all guards.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
