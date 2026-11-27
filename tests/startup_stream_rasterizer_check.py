"""Check slot-86 culling and constant depth bias through the 8x4 readback."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from hardware_adapter_filter_check import check as check_adapter_filter
from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex


def check(directory, consumer, backend, mode):
    cache = directory / "cache"
    cache.mkdir()
    vs, ps, root = (directory / name for name in ("vs", "ps", "root"))
    cases = ("back", "front", "dropped", "r1_back", "r1_front", "bias", "mixed_10_27", "mixed_27_10")
    if mode != "ignore":
        cases += ("ccw_back", "ccw_front", "r1_ccw_back", "r1_ccw_front")
        cases += ("d16s8_64_below", "d16s8_64_equal", "d24_64_below", "d24_64_equal",
                  "d16s8_half_below", "d16s8_half_equal")
    outs = {name: directory / name for name in cases}
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(True))

    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_D3D12_DEBUG="1", X2W_SHADER_CACHE=str(cache))
    base_args = [consumer, str(vs), str(ps), str(root), backend]
    if mode == "filter":
        return check_adapter_filter(base_args, env, allow_missing_factory=True)
    req = subprocess.run(base_args, env=env, capture_output=True, text=True)
    if backend == "hardware" and req.returncode == 77:
        print("No D3D12 feature-level 12 hardware adapter")
        return 77
    if req.returncode:
        print(req.stdout, end="")
        raise RuntimeError(f"request run failed on {backend}: {req.returncode}\n{req.stderr}")
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)

    run = subprocess.run(base_args + [str(outs[name]) for name in cases],
                         env=env, capture_output=True, text=True)
    print(run.stdout, end="")
    for line in run.stderr.splitlines():
        if "debug layer:" in line or "x2w-d3d12-debug " in line:
            print(line)
    if backend == "hardware" and run.returncode == 77:
        print("No D3D12 feature-level 12 hardware adapter")
        return 77
    if run.returncode or "-> drawn;" not in run.stderr:
        if mode == "ignore":
            print(f"ignore control did not reach a readback on {backend}: {run.returncode}")
            return 0
        raise RuntimeError(f"rasterizer draw failed on {backend}: {run.returncode}\n{run.stderr}")
    assert not re.search(r"x2w-d3d12-debug.*sev=[012]\b", run.stderr), run.stderr

    images = {name: outs[name].read_bytes() for name in outs}
    green = bytes((0, 255, 0, 255))
    clear = bytes((0, 0, 255, 255))
    drawn = (green * 4 + clear * 4) * 4
    blank = clear * 32

    # Dropping the subobject keeps today's solid / no-cull default. Ignoring a present
    # rasterizer makes the back and front images match that default, and this control fails.
    if mode == "ignore":
        if images["back"] == images["dropped"] and images["front"] == images["dropped"]:
            print(f"{backend}: ignore control saw the no-cull readback")
            return 0
        print(f"{backend}: ignore control failed because the stream rasterizer changed the 8x4 readback")
        return 1

    assert images["dropped"] == drawn, f"dropped: {images['dropped'].hex()}"
    # vertex() emits (-1,-1), (3,-1), (-1,3): counterclockwise as seen on the target.
    # With the default clockwise front face it survives front culling. Flipping the
    # winding bit makes it survive back culling for either stream subobject.
    for name in ("front", "r1_front", "ccw_back", "r1_ccw_back"):
        assert images[name] == drawn, f"{name}: expected the scissored triangle, got {images[name].hex()}"
    for name in ("back", "r1_back", "ccw_front", "r1_ccw_front"):
        assert images[name] == blank, f"{name}: expected the clear target, got {images[name].hex()}"
    assert images["bias"] == blank, f"fractional depth bias drew: {images['bias'].hex()}"
    mixed_draws = [name for name in ("mixed_10_27", "mixed_27_10") if images[name] != blank]
    assert not mixed_draws, f"duplicate rasterizer state drew pixels: {mixed_draws}"
    print(f"{backend}: eight cull/winding images matched; fractional bias and both mixed rasterizer orders stayed clear")
    for name, units in (("d16s8_64", 16384), ("d24_64", 64), ("d16s8_half", 128)):
        assert images[name + "_below"] == drawn, f"{name}: LESS failed one D24 unit below {units}"
        assert images[name + "_equal"] == blank, f"{name}: LESS passed at D24 depth {units}"
    print(f"{backend}: depth bias matched 16384, 64, and 128 D24 units")
    print("Mutant: without rescaling, format 118 stores a 256x smaller offset (64 D24 units); the below probe fails")
    return 0


if __name__ == "__main__":
    mode = sys.argv[3] if len(sys.argv) > 3 else "apply"
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), sys.argv[1], sys.argv[2], mode))
