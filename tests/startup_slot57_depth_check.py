"""Draw two overlapping triangles of unequal depth through device slot 57 pipelines whose state block A carries
depth-stencil register words, with X2W_DEPTH_STATE on and off, and compare every pixel with a known reference.

usage: startup_slot57_depth_check.py <startup_slot57_depth> <warp|hardware>

The fixture (startup_slot57_depth.cpp) draws the same triangle twice, at z = 0.75 ("near", red) and z = 0.25
("far", green), into an 8x4 target whose left half is covered. The depth-stencil texture is D24S8, cleared on the
desktop device. Expected winners (B = the blue background, nothing drawn):

  case               words                           clear        draw order   switch on   switch off
  rz_greater         reversed-Z, GREATER, write      depth 0      near, far    red         green (last wins)
  rz_less_wrong      LESS (direction intentionally   depth 0      near, far    blue        green
                     wrong for a depth cleared to 0)
  fz_less            LESS                            depth 1      far, near    green       red (last wins)
  rz_greater_no_dsv  GREATER, no depth view bound    -            near, far    green       green  (state not applied)
  rz_refused         GREATER + colour-write tie      depth 0      near, far    green       green  (decoder refuses: desktop state)
  rz_none            all words zero                  depth 0      near, far    green       green
  st_equal_hit       EQUAL, reference 0x5A           stencil 0x5A far          green       green
  st_equal_miss      EQUAL, reference 0x5B           stencil 0x5A far          blue        green  (reference reached the list)
  st_replace_hit     REPLACE 7, then EQUAL 7         stencil 0    near, far    green       green
  st_replace_miss    REPLACE 7, then EQUAL 8         stencil 0    near, far    red         green  (REPLACE wrote 7, not the list default)
  st_refused         ONES op (no D3D12 form)         stencil 0x5A far          green       green  (decoder refuses: desktop state)

The depth cases show the winner with reversed-Z, the other result with the switch off and with the compare
direction wrong; the stencil cases are separate and need the decoded reference (the list's own is 0). The same
executable renders on WARP and on hardware. Not covered: raster bias, depth bounds, back-face-only state, and the
title's clears (#1806).
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from startup_stream_check import container, gfx2hlsl
from startup_stream_depth_check import vertex
from startup_stream_draw_check import pixel

BLUE, GREEN, RED = (0, 0, 255, 255), (0, 255, 0, 255), (255, 0, 0, 255)
# case -> (switch on, switch off)
EXPECTED = {
    "rz_greater": (RED, GREEN),
    "rz_less_wrong": (BLUE, GREEN),
    "fz_less": (GREEN, RED),
    "rz_greater_no_dsv": (GREEN, GREEN),
    "rz_refused": (GREEN, GREEN),
    "rz_none": (GREEN, GREEN),
    "st_equal_hit": (GREEN, GREEN),
    "st_equal_miss": (BLUE, GREEN),
    "st_replace_hit": (GREEN, GREEN),
    "st_replace_miss": (RED, GREEN),
    "st_ref_restore": (GREEN, GREEN),
    "st_ref_reset": (GREEN, GREEN),
    "st_refused": (GREEN, GREEN),
}
# Pipelines whose state block is taken (switch on), per case: log lines "depth-stencil from state block A".
REFUSED_PIPELINES = 3  # rz_refused: two pipelines, st_refused: one


def picture(path, name):
    """The drawn colour of an 8x4 RGBA target; the right half must still be the background."""
    pixels = path.read_bytes()
    assert len(pixels) == 8 * 4 * 4, f"{name}: {len(pixels)} bytes"
    drawn = None
    for y in range(4):
        for x in range(8):
            texel = tuple(pixels[(y * 8 + x) * 4:(y * 8 + x) * 4 + 4])
            if x >= 4:
                assert texel == BLUE, f"{name}: background at ({x}, {y}) is {texel}"
            elif drawn is None:
                drawn = texel
            else:
                assert texel == drawn, f"{name}: ({x}, {y}) is {texel}, (0, 0) is {drawn}"
    return drawn


def check(directory, exe, backend):
    cache = directory / "cache"
    cache.mkdir()
    shaders = {name: directory / name for name in ("vs_far", "vs_near", "ps_green", "ps_red", "root")}
    shaders["vs_far"].write_bytes(vertex(0x3e800000))   # z = 0.25f
    shaders["vs_near"].write_bytes(vertex(0x3f400000))  # z = 0.75f
    shaders["ps_green"].write_bytes(pixel(True))
    shaders["ps_red"].write_bytes(pixel(False))
    base_env = {k: v for k, v in os.environ.items() if k != "X2W_DEPTH_STATE"}
    env = dict(base_env, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1", X2W_D3D12_DEBUG="1",
               X2W_SHADER_CACHE=str(cache))
    args = [exe, *(shaders[n] for n in ("vs_far", "vs_near", "ps_green", "ps_red", "root")), backend]

    request = subprocess.run(args, env=env, capture_output=True, text=True)
    if backend == "hardware" and request.returncode == 77:
        print("No D3D12 feature-level 12 hardware adapter")
        return 77
    if request.returncode:
        raise RuntimeError(f"request run failed on {backend}: {request.returncode}\n{request.stdout}{request.stderr}")
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)

    results = {}
    for label, switch in (("off", None), ("on", "1")):
        out = directory / label
        out.mkdir()
        run_env = dict(env)
        if switch:
            run_env["X2W_DEPTH_STATE"] = switch
        run = subprocess.run(args + [str(out)], env=run_env, capture_output=True, text=True)
        if run.returncode or "-> drawn;" not in run.stderr:
            raise RuntimeError(f"switch {label}: draw failed on {backend}: {run.returncode}\n{run.stdout}{run.stderr}")
        assert not re.search(r"x2w-d3d12-debug.*sev=[012]\b", run.stderr), run.stderr
        if label == "off":
            print(run.stdout.splitlines()[0])
        results[label] = ({case: picture(out / f"{case}.bin", f"{label} {case}") for case in EXPECTED}, run.stderr)

    names = {BLUE: "blue", GREEN: "green", RED: "red"}
    for index, label in ((0, "on"), (1, "off")):
        pictures, _ = results[label]
        for case, expected in EXPECTED.items():
            assert pictures[case] == expected[index], (
                f"switch {label}, {case}: drew {names.get(pictures[case], pictures[case])}, expected {names[expected[index]]}")
    # The two results differ wherever the table says they do, and nowhere else.
    differing = sorted(case for case, (on, off) in EXPECTED.items() if on != off)
    assert differing == ["fz_less", "rz_greater", "rz_less_wrong", "st_equal_miss", "st_replace_miss"], differing

    # Default behaviour: with the switch off the stand-in logs nothing about depth-stencil registers.
    assert "depth-stencil from state block A" not in results["off"][1] and "registers not used" not in results["off"][1]
    # Switch on: every decodable pipeline logs its state; every refusal is logged with the reason and keeps the desktop state.
    log_on = results["on"][1]
    refused = re.findall(r"depth-stencil registers not used \((.*?)\); keeping the desktop state", log_on)
    assert len(refused) == REFUSED_PIPELINES, refused
    assert sum("colour writes" in r for r in refused) == 2 and sum("stencil op" in r for r in refused) == 1, refused
    assert "depth-stencil from state block A: depth 1 func 5 write 1" in log_on, "reversed-Z state not logged"  # GREATER = 5

    print(f"startup_slot57_depth: PASS on {backend}: switch on {len(EXPECTED)} cases as tabulated, "
          f"{len(differing)} differ with the switch off ({', '.join(differing)}), {REFUSED_PIPELINES} refusals fall back to the desktop state")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
