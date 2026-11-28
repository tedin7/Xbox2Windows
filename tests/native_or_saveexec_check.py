"""Direct wave32 s_or_saveexec_b32: saved EXEC, the OR result, SCC and a restore, in full and partial waves."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_scalar_lds_check import scalar, sop1, vmove


def or_saveexec_readback(work, gpu, sdk):
    full = 0xffffffff
    root = (work / "root.bin").read_bytes()
    for width in (8, 32):
        entry = (1 << width) - 1
        sa = struct.pack("<7I", 0, 0, 0, width | 1 << 16, 1, 0, 0x80 | 5 << 1)
        for name, old, src, mask in (
            ("all", 0x25, 193, full),
            ("sgpr", 0x25, 10, 0x80000092),  # Source aliases the saved EXEC destination.
            ("literal", 0x25, 255, 0x80000018),
            ("vcc", 0x25, 106, 0x80000042),
            ("empty", 0, 128, 0),
            ("past-end", 0, 255, 0x80000000),
        ):
            updated = old | mask  # EXEC is stored as computed, including bits past the group
            words = [sop1(3, 20, 4), sop1(3, 21, 128), 0xf408080a, 0xfa000020,
                     scalar(0x1e, 7, 5, 137), 0x34000000 | 8 << 17 | 132,
                     0x4a000000 | 8 << 17 | 8 << 9 | 7,
                     sop1(0x3c, 24, 255), old]
            if src in (10, 106):
                words += [sop1(3, src, 255), mask]
            if name == "all":
                words.append(sop1(3, 10, 32))  # Replace symbolic descriptor provenance with numeric EXEC.
            if name == "vcc":
                # Derive the same partial EXEC through S_ANDN2_WREXEC_B32.
                words += [sop1(3, 126, 255), entry ^ old, sop1(0x47, 25, 24)]
            # Set the incoming SCC opposite to the expected new SCC.
            words += [0xbf068180 if updated else 0xbf068080, sop1(0x3d, 10, src)]
            if src == 255:
                words.append(mask)
            words += [scalar(0xa, 11, 129, 128), vmove(9, 10), vmove(10, 126),
                      0xe0701000, 0x80080908, 0xe0701004, 0x80080a08,
                      sop1(3, 126, 10), vmove(9, 177), 0xe0701008, 0x80080908,
                      sop1(3, 126, 24), vmove(9, 11), 0xe070100c, 0x80080908,
                      0xbf810000]
            expected = [0x69696969] * 2064
            for group in range(2):
                for lane in range(width):
                    pos = 8 + group * 128 + lane * 4
                    if updated & (1 << lane):
                        expected[pos:pos + 2] = [old, updated]
                    if old & (1 << lane):
                        expected[pos + 2] = 49
                    expected[pos + 3] = int(bool(updated))
            marker = write_request(work, words, root, sa)
            dest = work / "or-saveexec.bin"
            result = subprocess.run([gpu, sdk, marker, work / "cache", "0", dest, "--direct"])
            if result.returncode == 77:
                sys.exit(77)
            result.check_returncode()
            actual = struct.unpack("<2064I", dest.read_bytes())
            assert list(actual) == expected, (width, name, [
                (i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b
            ][:10])


def main(gpu, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        src, root = work / "root.hlsl", work / "root.bin"
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, src], check=True)
        or_saveexec_readback(work, gpu, sdk)
        print("s_or_saveexec_b32 matched six sources in 8- and 32-thread groups, with SCC, restore and guards.")


if __name__ == "__main__":
    main(*sys.argv[1:])
