"""Source-built wave64 spill/reload fixture for the bounded mesh interpreter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mesh_entry_aot import Wave


PREFIX = [
    "s_mov_b64 s[22:23], exec",
    "s_mov_b64 exec, 0",
    "s_mov_b32 s8, 17",
    "s_mov_b32 s9, 29",
    "v_writelane_b32 v10, s8, 3",
    "v_writelane_b32 v10, s9, 63",
    "s_mov_b32 s8, 0",
    "s_mov_b32 s9, 0",
]
READS = [
    "v_readlane_b32 s12, v10, 3",
    "v_readlane_b32 s13, v10, 63",
]
SUFFIX = [
    # Reuse one of the two slots for the saved high EXEC word.
    "v_writelane_b32 v10, s23, 63",
    "s_mov_b32 s23, 0",
    "v_readlane_b32 s23, v10, 63",
    "s_mov_b32 s22, 0",
    "s_mov_b64 exec, s[22:23]",
    "v_mov_b32_e32 v1, s12",
    "v_mov_b32_e32 v2, s13",
    "exp pos0, v1, v2, v1, v2 done",
    "s_endpgm",
]


def run(lines):
    wave = Wave(0, [])
    wave.run([(4 * i, line) for i, line in enumerate(lines)], {})
    return wave


def check_values(reads):
    wave = run(PREFIX + reads + SUFFIX)
    assert (wave.s[12], wave.s[13], wave.s[23]) == (17, 29, 0xFFFFFFFF), "reload values"
    assert wave.positions == {lane: [17, 29, 17, 29] for lane in range(32, 64)}


def refuses(lines, message):
    try:
        run(lines + ["s_endpgm"])
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Expected refusal: {message}")


def main():
    check_values(READS)
    try:
        check_values([
            "v_readlane_b32 s12, v10, 63",
            "v_readlane_b32 s13, v10, 3",
        ])
    except AssertionError as error:
        assert str(error) == "reload values"
    else:
        raise AssertionError("Swapped spill slots passed the value check")

    refuses(PREFIX + ["v_readlane_b32 s12, v10, s8"], "constant integer lane")
    refuses(PREFIX + ["v_writelane_b32 v10, s8, 64"], "out of range for wave64")
    refuses(PREFIX + ["v_readlane_b32 s12, v10, 4"], "unwritten lane spill slot v10[4]")
    refuses(
        PREFIX + ["s_mov_b64 exec, 8", "v_add_nc_u32_e32 v10, 1, v10"] + READS,
        "Mixed lane spill use of v10",
    )
    refuses([
        "s_cbranch_execz 1",
        "s_branch 2",
        "v_writelane_b32 v10, s17, 3",
        "s_nop 0",
        "v_readlane_b32 s12, v10, 3",
    ], "unwritten lane spill slot v10[3]")

    # EXEC words as writelane sources: a non-zero mask with bits in both halves.
    wave = run([
        "s_mov_b32 s22, 0x80000005",
        "s_mov_b32 s23, 0xc0000003",
        "s_mov_b64 exec, s[22:23]",
        "v_writelane_b32 v12, exec_lo, 5",
        "v_writelane_b32 v12, exec_hi, 6",
        "s_mov_b64 exec, 0",
        "v_readlane_b32 s30, v12, 5",
        "v_readlane_b32 s31, v12, 6",
        "s_endpgm",
    ])
    assert (wave.s[30], wave.s[31]) == (0x80000005, 0xC0000003), "exec word spill"

    # A register-range operand that covers the spill VGPR also counts as ordinary use.
    wave = Wave(0, [])
    try:
        wave.run([(4 * i, line) for i, line in enumerate(
            PREFIX + ["buffer_load_dwordx3 v[9:11], v0, s[16:19], 0 offen"] + READS + ["s_endpgm"])], {})
    except (ValueError, KeyError):
        pass
    assert 10 in wave.spill_vector_uses, "register range over the spill VGPR not seen"

    # The runtime geometry translator refuses lane spills even when no lane is live.
    from mesh_dynamic_aot import Geometry, RuntimeWave
    runtime = RuntimeWave(0, Geometry(1, ()))
    try:
        runtime.run([(0, "s_mov_b64 exec, 0"), (4, "v_writelane_b32 v10, s17, 3"), (8, "s_endpgm")], {})
    except ValueError as error:
        assert "Unsupported runtime mesh instruction v_writelane_b32" in str(error), str(error)
    else:
        raise AssertionError("RuntimeWave accepted a lane spill")

    wave = run([
        "s_mov_b32 s10, s17",
        "s_mov_b32 s11, 0",
        "s_load_dwordx4 s[16:19], s[10:11], 0x60",
        "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dword s30, s[16:19], 0x4",
        "s_waitcnt lgkmcnt(0)",
        "v_writelane_b32 v10, s30, 63",
        "v_readlane_b32 s12, v10, 63",
        "s_mov_b32 s22, 0",
        "s_mov_b32 s23, 0x80000000",
        "s_mov_b64 exec, s[22:23]",
        "v_mov_b32_e32 v1, v10",
        "s_endpgm",
    ])
    assert wave.s[12] == "pointData[0].y"
    assert wave.v[63][1] == "pointData[0].y"


if __name__ == "__main__":
    main()
