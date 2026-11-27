"""Direct compute wave32 EXEC bitreplicate readback."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1, vmove


BITREPLICATE = 0x3B
MOV_B32 = 3
EXEC_LO, EXEC_HI = 126, 127
LITERAL = 255
THREADS = 32
LANES = 2 * THREADS
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
SENTINEL = 0x69696969
CASES = (
    (0xFF00FF00, LITERAL),
    (0x0000FFFF, LITERAL),
    (5, 128 + 5),
    (0xF0F0F0F0, LITERAL),
)


def program(value, source, dst=EXEC_LO):
    # Table pointer s4, group.x s5. v8 addresses one dword per dispatch lane.
    words = [
        sop1(MOV_B32, 20, 4), sop1(MOV_B32, 21, 128),
        0xF408080A, 0xFA000020,  # s_load_dwordx4 s[8:11], s[20:21], 32
        scalar(0x1E, 7, 5, 135),  # group.x * 128
        vop2(0x1A, 8, 130, 0), vop2(0x25, 8, 7, 8),  # v8 = group base + local.x * 4
        0xBF8CC07F,
        0xBF068180,  # s_cmp_eq_u32 0, 1 (SCC = 0)
        sop1(BITREPLICATE, dst, source),
    ]
    if source == LITERAL:
        words.append(value)
    words += [
        0xE0701000, 0x80080008,  # store local.x from v0 under replicated EXEC
        sop1(MOV_B32, 6, EXEC_LO),
        sop1(MOV_B32, 12, EXEC_HI),
        sop1(MOV_B32, EXEC_LO, 193),  # restore all 32 lanes
    ]
    for plane, reg in enumerate((6, 12, 253), start=1):
        # Store the saved low word, saved high word and incoming SCC on every lane.
        offset = plane * LANES * 4
        words += [vmove(4, reg), 0xE0701000 | offset, 0x80080408]
    return words + [0xBF810000]


def request(work, root, words):
    sa = struct.pack("<7I", 0, 0, 0, THREADS | (1 << 16), 1 | (1 << 20), 0, 0x8A)
    return write_request(work, words, root, sa)


def reference(value):
    lo, hi = (sum(3 << (2 * bit) for bit in range(16) if value & (1 << (bit + start)))
              for start in (0, 16))
    masked = tuple(lane if lo & (1 << lane) else SENTINEL for lane in range(THREADS))
    return ((SENTINEL,) * GUARD_WORDS + masked * 2 + (lo,) * LANES + (hi,) * LANES
            + (0,) * LANES + (SENTINEL,) * (OUTPUT_WORDS - GUARD_WORDS - 4 * LANES))


def controls(cli, sdk, work, root):
    value, source = CASES[0]
    marker = request(work, root, program(value, source, dst=9))
    output = marker.with_suffix(".cso")
    result = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "bitreplicate requires an aligned general scalar pair" in result.stderr, (
        result.stdout, result.stderr)
    assert not output.exists(), output
    return 0


def hardware(replay, sdk, work, root):
    for value, source in CASES:
        marker = request(work, root, program(value, source))
        output = work / "out.bin"
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
        expected = reference(value)
        for word, (got, want) in enumerate(zip(actual, expected)):
            assert got == want, f"{value:#010x}, word {word}: got {got:#010x}, expected {want:#010x}"
        print(f"{value:#010x}: lane stores, saved EXEC, SCC and guards matched")
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / "root.hlsl"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / "root.bin"
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root, shader], check=True)
        return {"controls": controls, "hardware": hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
