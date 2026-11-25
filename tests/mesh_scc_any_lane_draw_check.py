"""Draw the group-uniform "any lane active" mesh branch and read the pixels back.

Usage: mesh_scc_any_lane_draw_check.py <dxc> <offscreen_check> <warp|hardware>

Each threshold T selects the live lanes (thread id >= T) of one 128-thread group. The
group draws the full quad when any wave has a live lane and the half-square triangle
otherwise. The expected pixels come from group_reference, not from the lowering. Three
mutated programs must draw something other than the reference at T = 100.
"""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mesh_dynamic_aot_draw_check import compile_shader, fixture
from mesh_dynamic_geometry_check import SECOND
from mesh_entry_aot import lower
from mesh_scc_any_lane_check import CODE, LIMITS, decode, group_reference, patched

THRESHOLDS = (0, 64, 100, 127, 128, 200)
MUTANT_THRESHOLD = 100
QUAD = ((0, 1, 2), (2, 1, 3))
TRIANGLE = ((0, 1, 2),)


def mutants():
    return {
        "own flag only": patched(CODE, "ds_read_b32 v16", [0xD8D80000, 0x1000000E]),
        "flipped branch": patched(CODE, "s_cbranch_scc0", [0xBF850002]),
        "32-bit mask compare": patched(CODE, "s_cmp_lg_u64", [0xBF07806A]),
    }


def draw(runner, backend, mesh, pixel, inputs, threshold):
    expected = QUAD if group_reference(threshold) == (4, 2) else TRIANGLE
    inputs.write_bytes(fixture(0, 0, expected))
    return subprocess.run([runner, backend, "--mesh-indexed", mesh, pixel, inputs],
                          capture_output=True, text=True)


def check(dxc, runner, backend, target):
    pixel = compile_shader(dxc, target, "solid", "float4 main() : SV_Target { return float4(1,0,1,1); }", "ps_6_0")
    inputs = target / "inputs.bin"

    def shader(stem, code, threshold):
        return compile_shader(dxc, target, f"{stem}-{threshold}",
                              lower(decode(code), [threshold, SECOND], LIMITS), "ms_6_5")

    for threshold in THRESHOLDS:
        print(f"group any-lane branch T={threshold}", flush=True)
        result = draw(runner, backend, shader("scc", CODE, threshold), pixel, inputs, threshold)
        sys.stdout.write(result.stdout)
        if result.returncode:
            sys.stderr.write(result.stderr)
            return result.returncode
    for name, code in mutants().items():
        print(f"mutant: {name}", flush=True)
        result = draw(runner, backend, shader("mutant", code, MUTANT_THRESHOLD), pixel, inputs, MUTANT_THRESHOLD)
        if result.returncode == 77:
            return 77
        if result.returncode != 1 or "mismatches=0" in result.stdout:
            sys.stderr.write(f"The {name} program drew the reference pixels\n{result.stdout}{result.stderr}")
            return 1
    return 0


if __name__ == "__main__":
    output = Path("out").resolve()
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mesh-scc-any-lane-", dir=output) as target:
        sys.exit(check(*sys.argv[1:4], Path(target)))
