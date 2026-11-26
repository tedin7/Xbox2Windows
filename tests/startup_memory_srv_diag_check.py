"""Census lines for refused slot 70 calls and "token" texture nulls (src/startup_stubs/memory_srv_census.h).

The consumer (tests/startup_memory_srv_diag.cpp) spends the slot 70 line limit with eight zero-address
refusals, has a 2048x2048 BC1 10-mip texture refused, draws column 0 from the untouched (zero) table
entry, then publishes a supported 1-mip R8 texture and draws column 1 from it.

Expected: the BC1 refusal is counted and named as the only refused shape, and its line is logged past
the limit; the column 0 null is counted for the pipeline and its entry words are all zero; the R8
texture produces no refusal and its texels reach column 1. With --control the refused call is
replaced by the supported R8 publish: the control returns 0 only when the check then stops at exactly
CONTROL_MISMATCH (column 0 samples the R8 texel instead of a null view), 77 on a capability skip, and
1 for any other outcome, including a pass.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from startup_address_srv_check import prepare_draw_cache

HLSL = ("Texture2D<float> t : register(t0);\n"
        "float4 main(float4 p : SV_Position) : SV_Target {\n"
        "  return float4(t.Load(int3(uint2(p.xy) % 2, 0)), 0, 0, 1);\n}\n")
REFUSALS = (r"x2w-d3d12 memory SRV refusals \(slot 70\): total 9; null_destination 0, unreadable_resource 0, "
            r"unreadable_view 0, null_address 8, unsupported_description 1, destination_outside_view_heaps 0, "
            r"gpu_written_resource 0, as_unaligned 0, as_not_placed 0, as_wrong_state 0, buffer_unavailable 0, texture_unavailable 0, texture_no_resource 0, texture_unreadable 0, texture_over_64mib 0, partial_tile_xg_rows 0, partial_tile_beyond_xg_size 0, "
            r"unsupported_desktop_format 0, console_layout_xg_rows 0, console_layout_without_footprints 0, "
            r"console_layout_beyond_xg_size 0, overlapping_resource 0, other 0; "
            r"distinct shapes 1 \(calls past the shape table 0\)")
SHAPE = (r"x2w-d3d12 memory SRV refused shape 1: reason=unsupported_description dim=3 fmt=71 size=2048x2048x1 "
         r"mips=10 layout=0x0 flags=0x0 calls=1")
TOKENS = r"x2w-draw token texture nulls: total 1 in 1 pipelines; ([0-9a-f]{16})=1 other=0"
WRITES = r"x2w-d3d12 memory SRV writes \(slot 70\): calls 1, distinct entries 1 \(untracked 0\)"
WRITTEN = (r"x2w-d3d12 memory SRV written 1: dest=0x[0-9a-f]+ address=0x[0-9a-f]+ "
           r"res\{dim=3 fmt=61 size=2x2x1 mips=1 layout=0x0 flags=0x0 align=0 samples=1/0\} "
           r"view\{dim=4 fmt=61 map=0x1688 mostmip=0 mips=1 plane=0 clamp=0\}")
CONTROL_MISMATCH = "column 0 row 0: 100000ff, expected a null view (000000ff)"
ENTRY = r"x2w-draw token texture null: pipeline ([0-9a-f]{16}) srv 0 entry 0x[0-9a-f]+ words" + " 00000000" * 8


def census_block(stdout):
    blocks = re.findall(r"census report begin\n(.*?)census report end", stdout, re.S)
    if not blocks:
        raise AssertionError("no census report in the consumer output")
    return [line.rstrip() for line in blocks[-1].splitlines() if line.strip()]


def check(directory, consumer, backend, control=False):
    modes = {"X2W_TEST_MEMORY_SRV_DIAG": "1"}
    if control:
        modes["X2W_TEST_MEMORY_SRV_DIAG_CONTROL"] = "1"
    env, args, output = prepare_draw_cache(directory, consumer, HLSL, modes, "srv 0 tex2d table 0 0")
    trap_log = directory / "trap.log"
    env = dict(env, X2W_TRAP_LOG=str(trap_log))
    result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode == 77:
        return 77
    if result.returncode:
        raise RuntimeError(f"consumer failed ({result.returncode})\n{result.stderr[-5000:]}")

    # Pixels: column 0 sampled the zero entry (null view), column 1 the R8 texels.
    pixels = output.read_bytes()
    for y in range(4):
        left = pixels[(y * 8) * 4:(y * 8) * 4 + 4]
        right = pixels[(y * 8 + 1) * 4:(y * 8 + 1) * 4 + 4]
        if left != bytes((0, 0, 0, 255)):
            raise AssertionError(f"column 0 row {y}: {left.hex()}, expected a null view (000000ff)")
        if right != bytes((23 + 32 * (y % 2), 0, 0, 255)):
            raise AssertionError(f"column 1 row {y}: {right.hex()}, expected the R8 texel")

    lines = census_block(result.stdout)
    if len(lines) != 4:
        raise AssertionError("census report is not four lines:\n" + "\n".join(lines))
    if not re.fullmatch(REFUSALS, lines[0]):
        raise AssertionError(f"refusal counts\n  line     {lines[0]}\n  expected {REFUSALS}")
    if not re.fullmatch(SHAPE, lines[1]):
        raise AssertionError(f"refused shape\n  line     {lines[1]}\n  expected {SHAPE}")
    tokens = re.fullmatch(TOKENS, lines[2])
    if not tokens:
        raise AssertionError(f"token nulls\n  line     {lines[2]}\n  expected {TOKENS}")
    if not re.fullmatch(WRITES, lines[3]):
        raise AssertionError(f"slot 70 writes\n  line     {lines[3]}\n  expected {WRITES}")

    log = trap_log.read_text(errors="replace").splitlines()
    slot70 = [line for line in log if "Device::slot70" in line]
    reasons = [re.search(r"reason=(\S+)", line).group(1) for line in slot70]
    if reasons != ["null_address"] * 8 + ["unsupported_description"]:
        raise AssertionError("slot 70 lines (the ninth is past the line limit):\n" + "\n".join(slot70))
    if "fmt=71 size=2048x2048x1 mips=10" not in slot70[-1]:
        raise AssertionError(f"the ninth slot 70 line does not name the BC1 shape:\n{slot70[-1]}")
    written = [line.rstrip() for line in log if "memory SRV written" in line]
    if len(written) != 1 or not re.fullmatch(WRITTEN, written[0]):
        raise AssertionError("slot 70 write lines (only the R8 publish writes):\n" + "\n".join(written))
    entries = [m for m in (re.fullmatch(ENTRY, line.rstrip()) for line in log) if m]
    if len(entries) != 1 or entries[0].group(1) != tokens.group(1):
        raise AssertionError("token entry lines:\n" + "\n".join(line for line in log if "token texture null:" in line))
    print(f"{backend}: BC1 refusal counted and named past the line limit; zero entry counted for pipeline "
          f"{tokens.group(1)} with zero words; R8 accepted and sampled")
    return 0


def check_control(directory, consumer, backend):
    try:
        result = check(directory, consumer, backend, control=True)
    except AssertionError as mismatch:
        if str(mismatch) == CONTROL_MISMATCH:
            print(f"{backend} control: stopped at the expected mismatch: {mismatch}")
            return 0
        print(f"{backend} control: unexpected mismatch: {mismatch}")
        return 1
    if result == 77:
        return 77
    print(f"{backend} control: the check passed with the supported texture in place of the refused one")
    return 1


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if a != "--control"]
    with tempfile.TemporaryDirectory() as temporary:
        if "--control" in sys.argv[1:]:
            sys.exit(check_control(Path(temporary), *arguments))
        sys.exit(check(Path(temporary), *arguments))
