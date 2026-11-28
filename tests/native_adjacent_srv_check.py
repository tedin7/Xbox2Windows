"""Check dynamic SRVs spanning adjacent ranges in a serialized guest root signature."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_scalar_lds_check import scalar, sop1, vmove


def main(gpu, sdk):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # s4 and s5 are the two root tables. s6 holds group.x. The
        # candidate span starts at slot 1, group 1 selects slot 2.
        words = [sop1(3, 20, 4), sop1(3, 21, 128), sop1(3, 22, 5), sop1(3, 23, 128),
                 scalar(0xe, 11, 6, 129), scalar(0x1e, 11, 11, 133),
                 0xf408040a, 0x16000020, 0xf408080b, 0xfa000000,
                 scalar(0x1e, 12, 6, 130), vmove(8, 12), sop1(3, 126, 129), vmove(3, 130),
                 0xe0002000, 0x80040103, 0xe0701000, 0x80080108, 0xbf810000]
        sa = struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1 | (1 << 20), 0, 0x80 | 6 << 1)
        for name, second, reason in (
                ("append", "SRV(t18, space=2)", ""),
                ("null", "SRV(t18, space=2)", ""),
                ("reversed", "SRV(t18, space=2, offset=2)", ""),
                ("hole", "SRV(t18, space=2, offset=3)", "exceeds its declared root range"),
                ("uav", "UAV(u18, space=2, offset=2)", "overlaps a non-SRV root range")):
            first = "SRV(t7, offset=1)"
            ranges = second + "," + first if name == "reversed" else first + "," + second
            src, root = work / "root.hlsl", work / "root.bin"
            src.write_text(f'#define SIG "DescriptorTable({ranges}),DescriptorTable(UAV(u0))"')
            subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, src], check=True)
            marker = write_request(work, words, root.read_bytes(), sa)
            dest = work / (name + ".bin")
            mode = "--direct-null-srv" if name == "null" else "--direct"
            result = subprocess.run([gpu, sdk, marker, work / "cache", "0", dest, mode],
                                    capture_output=True, text=True)
            if result.returncode == 77:
                sys.exit(77)
            if reason:
                assert result.returncode and reason in result.stderr and not dest.exists(), result.stderr
                continue
            assert not result.returncode, (result.stdout, result.stderr)
            expected = [0x69696969] * 2064
            expected[8:10] = [0x3f800000 + 256 * slot + 8 for slot in (1, 2)]
            if name == "null":
                expected[9] = 0
            actual = struct.unpack("<2064I", dest.read_bytes())
            assert list(actual) == expected, [(i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b][:10]
    print("Both layouts selected distinct SRVs, a recorded null returned zero, and hole/UAV controls were refused.")


if __name__ == "__main__":
    main(*sys.argv[1:])
