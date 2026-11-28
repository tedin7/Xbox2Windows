"""Compile and replay direct wave64 scalar loads that use VCC as scalar data."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1, vmove


THREADS = 64
LANE_WORDS = 16
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
SENTINEL = 0x69696969
INPUT_BASE = 0x51AA0000
EXEC_LO, EXEC_HI = 0x5A5AA5A5, 0xA5A55A5A


def program(first_dst=104):
    # Use the same root SRV, synthesized byte descriptor and UAV slot as the wave32 fixture.
    words = [
        sop1(4, 126, 193),  # s_mov_b64 exec, -1
        sop1(4, 16, 4), sop1(3, 18, 255), 132, sop1(3, 19, 255), 0x31016FAC,
        sop1(3, 20, 6), sop1(3, 21, 128), 0xF4080000 | 40 << 6 | 10, 0xFA000020,
        0xF4280008 | first_dst << 6, 0xFA000000,  # s_buffer_load_dwordx4 s[104:107], s[16:19], 0
        0xBF8CC07F,
        scalar(0x1E, 12, 7, 140),  # group.x * 4096 bytes
        vop2(0x1A, 8, 134, 0), vop2(0x25, 8, 12, 8),  # group byte base + local.x * 64
        sop1(3, 30, 106),  # s_mov_b32 s30, vcc_lo
    ]

    def store(reg, slot):
        return [vmove(9, reg), 0xE0701000 | slot * 4, 0x800A0908]

    for slot, reg in enumerate((104, 105, 30, 107)):
        words += store(reg, slot)
    words += [
        0xF40C0002 | 100 << 6, 0xFA000050,  # s_load_dwordx8 s[100:107], s[4:5], 80
        0xBF8CC07F, sop1(3, 30, 107),  # s_mov_b32 s30, vcc_hi
    ]
    for slot, reg in enumerate((*range(100, 107), 30), start=4):
        words += store(reg, slot)
    words += [
        sop1(3, 106, 132), sop1(3, 107, 136),  # scalar byte offsets 4 and 8
        0xF4200008 | 30 << 6, 106 << 25 | 24,  # s_buffer_load_dword s30, s[16:19], vcc_lo + 24
        0xF4000002 | 31 << 6, 107 << 25 | 32,  # s_load_dword s31, s[4:5], vcc_hi + 32
        0xBF8CC07F,
        *store(30, 12), *store(31, 13),
        sop1(3, 126, 255), EXEC_LO, sop1(3, 127, 255), EXEC_HI,
        0xF4040002 | 106 << 6, 0xFA000014,  # s_load_dwordx2 vcc, s[4:5], 20
        0xBF8CC07F,
        vmove(1, 129), vop2(1, 9, 128, 1),  # v_cndmask_b32 v9, 0, v1, vcc
        0xE0701000 | 14 * 4, 0x800A0908,  # EXEC-inactive lanes retain the sentinel
        0xBF810000,
    ]
    return words


def request(work, root, first_dst):
    # .sa dword 4 bit 20 is clear, so each 64-thread group is one guest wave64.
    sa = struct.pack("<7I", 0, 0, 0, THREADS | 1 << 16, 1, 0, 0x80 | 7 << 1)
    return write_request(work, program(first_dst), root, sa)


def reference():
    expected = [SENTINEL] * OUTPUT_WORDS
    vcc = (INPUT_BASE + 5) | ((INPUT_BASE + 6) << 32)
    exec_mask = EXEC_LO | (EXEC_HI << 32)
    loaded = [INPUT_BASE + index for index in (*range(4), *range(20, 28), 7, 10)]
    for group in range(2):
        for lane in range(THREADS):
            pos = GUARD_WORDS + (group * THREADS + lane) * LANE_WORDS
            expected[pos:pos + len(loaded)] = loaded
            if exec_mask & (1 << lane):
                expected[pos + 14] = (vcc >> lane) & 1
    return expected


def controls(tool, sdk, work, root):
    for dst, error in ((104, ""), (105, "unsupported scalar memory destination range")):
        marker = request(work, root, dst)
        output = marker.with_suffix(".cso")
        result = subprocess.run([tool, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                                capture_output=True, text=True)
        if error:
            assert result.returncode != 0 and error in result.stderr, (result.stdout, result.stderr)
            assert not output.exists(), output
        else:
            assert result.returncode == 0 and output.exists(), (result.stdout, result.stderr)
    return 0


def hardware(tool, sdk, work, root):
    marker = request(work, root, 104)
    output = work / "out.bin"
    result = subprocess.run([tool, sdk, marker, work / "cache", "0", output, "--direct"])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
    for word, (got, want) in enumerate(zip(actual, reference())):
        assert got == want, f"word {word}: got {got:#010x}, expected {want:#010x}"
    print("Wave64 SMEM scalar data, both VCC mask halves, and all guards matched.")
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        source, root = work / "built.hlsl", work / "built.root"
        source.write_text('#define SIG "SRV(t0),DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root, source], check=True)
        return {"controls": controls, "hardware": hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
