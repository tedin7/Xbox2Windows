"""Exercise symbolic descriptor spills and numeric spills in the same VGPR."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_lane_access_check import readlane, vop2, writelane
from native_scalar_lds_check import scalar, sop1, vmove


LAUNCH = struct.pack("<7I", 0, 0, 0, 32 | 1 << 16, 1, 0, 0x8c)
SPILL = 1
ROOT = "DescriptorTable(SRV(t0,numDescriptors=8)),DescriptorTable(UAV(u0,numDescriptors=8))"


def load_descriptor(dst, table_pair, byte_offset):
    return [0xf4080000 | dst << 6 | table_pair // 2, 0xfa000000 | byte_offset]


def spill_descriptor(first):
    return [word for lane in range(4) for word in writelane(SPILL, first + lane, 128 + lane)]


def program(control=""):
    # s20:21 and s22:23 are the table pointers. s6 is the workgroup ID.
    words = [sop1(3, 20, 4), sop1(3, 21, 128), sop1(3, 22, 5), sop1(3, 23, 128),
             *load_descriptor(32, 20, 32), *load_descriptor(36, 22, 32), 0xbf8cc07f,
             vmove(SPILL, 128), scalar(0x1e, 7, 6, 137),
             vop2(0x1a, 8, 132, 0), vop2(0x25, 8, 7, 8), vop2(0x1a, 2, 130, 0)]
    if control == "join":
        left, right = spill_descriptor(32), spill_descriptor(44)
        words += [*load_descriptor(44, 20, 64), 0xbf8cc07f,
                  0xbf840000 | (len(left) + 1), *left, 0xbf820000 | len(right), *right]
    else:
        words += spill_descriptor(32)
    words += [*writelane(SPILL, 139, 132), *writelane(SPILL, 150, 133)]
    words += [sop1(3, reg, 128) for reg in range(32, 36)]
    if control == "overwrite":
        words += [vmove(SPILL, 128)]
    if control == "valu":
        words += [vmove(3, 256 + SPILL)]
    if control == "sgpr_selector":
        words += [sop1(3, 10, 128), *readlane(32, SPILL, 10)]
    for lane in range(4):
        selector = 256 + 2 if control == "vgpr_selector" and lane == 0 else 128 + lane
        words += readlane(32 + lane, SPILL, selector)
    words += [*readlane(10, SPILL, 132), *readlane(11, SPILL, 133),
              0xe0301000, 0x80080902, 0xbf8c0f70, vmove(10, 10), vmove(11, 11)]
    for slot, vgpr in enumerate((9, 10, 11)):
        words += [0xe0701000 | 4 * slot, 0x80090008 | vgpr << 8]
    return words + [0xbf810000]


def reference():
    source = [0x7fff8000, 0x40008001, 0x007fff81, 0xfc003c00, 0x0001bc00, 0x42987ec5]
    expected = [0x69696969] * 2064
    for group in range(2):
        for lane in range(32):
            base = 8 + 4 * (32 * group + lane)
            expected[base:base + 3] = [source[lane % len(source)], 11, 22]
    return expected


def main(cli, replay, sdk, adapter):
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        source, root_file = work / "root.hlsl", work / "root.bin"
        source.write_text(f'#define SIG "{ROOT}"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root_file, source], check=True)
        root = root_file.read_bytes()
        cache = work / "cache"
        controls = {
            "vgpr_selector": "lane read/write needs a constant or SGPR lane",
            "sgpr_selector": "needs a constant lane",
            "overwrite": "write to a VGPR holding symbolic lanes",
            "join": "symbolic VGPR lane read has conflicting incoming values",
            "valu": "symbolic VGPR lane read of v1",
        }
        for control, diagnostic in controls.items():
            marker = write_request(work, program(control), root, LAUNCH)
            dest = work / f"{control}.cso"
            run = subprocess.run([cli, sdk, cache, "gfx1013-direct-cs@0", marker, dest],
                                 capture_output=True, text=True)
            assert run.returncode and diagnostic in run.stdout + run.stderr, (control, run.stdout, run.stderr)
            assert not dest.exists(), control

        marker = write_request(work, program(), root, LAUNCH)
        dest = work / "readback.bin"
        mode = "--direct-warp" if adapter == "warp" else "--direct"
        run = subprocess.run([replay, sdk, marker, cache, "0", dest, mode])
        if run.returncode == 77:
            return 77
        run.check_returncode()
        actual = struct.unpack("<2064I", dest.read_bytes())
        mismatches = [(i, a, e) for i, (a, e) in enumerate(zip(actual, reference())) if a != e]
        assert not mismatches, mismatches[:12]
        print("Two workgroups matched 192 descriptor-spill and numeric-spill words and all guards.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
