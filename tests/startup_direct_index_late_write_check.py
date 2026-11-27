"""A translated direct indexed draw uses the aliased view's pre-submit contents."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from startup_address_srv_check import prepare_draw_cache

VERTEX = """
float4 main(uint vertex : SV_VertexID) : SV_Position {
    uint corner = vertex % 3;
    float2 xy = float2(corner == 1 ? 3.0 : -1.0, corner == 2 ? -3.0 : 1.0);
    return float4(xy.x + (vertex >= 3 ? 8.0 : 0.0), xy.y, 0.5, 1.0);
}
"""
VERTEX_VB = """cbuffer RootConstants : register(b0) {
    uint rect_flag;
    int base_vertex;
    uint start_instance;
};
ByteAddressBuffer vertices : register(t0);
float4 main(uint vertex : SV_VertexID) : SV_Position {
    int vindex = int(vertex) + base_vertex;
    float x_offset = asfloat(vertices.Load(vindex * 16));
    uint corner = vertex % 3;
    float2 xy = float2(corner == 1 ? 3.0 : -1.0, corner == 2 ? -3.0 : 1.0);
    return float4(xy.x + x_offset, xy.y, 0.5, 1.0);
}
"""
PIXEL = "float4 main() : SV_Target { return float4(1, 0, 0, 1); }"
VISIBLE = bytes((255, 0, 0, 255))
CLIPPED = bytes((0, 0, 255, 255))
CASES = (
    ("visible", VISIBLE),
    ("clipped", CLIPPED),
    ("late-visible", VISIBLE),
    ("late-clipped", CLIPPED),
    ("late-visible-r32", VISIBLE),
)
VB_CASES = (
    ("visible-vb", VISIBLE),
    ("clipped-vb", CLIPPED),
    ("late-visible-vb", VISIBLE),
    ("late-clipped-vb", CLIPPED),
    ("late-visible-vb-unplaced", CLIPPED),
)


def run_cases(cache_dir, consumer, backend, vertex_src, binding, cases, is_vb=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    env, args, output = prepare_draw_cache(cache_dir, consumer, PIXEL, {}, binding, vertex_hlsl=vertex_src)
    env["X2W_D3D12_DEBUG"] = "1"
    for mode, color in cases:
        output.unlink(missing_ok=True)
        result = subprocess.run([*args, backend, output], env={**env, "X2W_TEST_INDEX_PROBE": mode},
                                capture_output=True, text=True, errors="replace", timeout=20)
        if result.returncode == 77:
            print(result.stdout, result.stderr)
            return 77
        if result.returncode or re.search(r"x2w-d3d12-debug [^\n]*sev=[01]\b", result.stderr):
            raise RuntimeError(f"{mode} failed ({result.returncode})\n{result.stdout}\n{result.stderr[-4000:]}")
        expected_aliases = 2 if is_vb and mode != "late-visible-vb-unplaced" else 1
        aliases = len(re.findall(r"slot133[^\n]*-> 0x00000000 aliased gpuva=", result.stderr))
        if aliases < expected_aliases:
            raise AssertionError(f"{mode}: expected at least {expected_aliases} aliased buffers, saw {aliases}\n{result.stderr[-4000:]}")
        if not re.search(r"indexed\(3, 1, start [01], base 0\)[^\n]*-> drawn;", result.stderr):
            raise AssertionError(f"{mode}: translated indexed draw was not recorded\n{result.stderr[-4000:]}")
        pixels = output.read_bytes()
        if pixels != color * 32:
            raise AssertionError(f"{mode}: got {pixels.hex()}, expected 32 pixels of {color.hex()}")
        adapter = next(line for line in result.stdout.splitlines() if line.startswith("adapter="))
        debug_layer = "enabled" if "debug layer: 0x00000000 enabled" in result.stderr else "unavailable"
        print(f"{mode}: 32 exact pixels, {adapter}, debug layer {debug_layer}")
    return 0


def check(directory, consumer, backend):
    rc = run_cases(directory / "direct", consumer, backend, VERTEX, "", CASES)
    if rc == 77:
        return 77
    rc = run_cases(directory / "vb", consumer, backend, VERTEX_VB, "srv 0 buf vb 0", VB_CASES, is_vb=True)
    if rc == 77:
        return 77
    return 0


def check_read_cache(directory, consumer, backend, xmem, mode):
    vertex = """
    float4 main(uint index : SV_VertexID) : SV_Position {
        if (index > 2) return float4(4, 4, 0, 1);
        return float4(index == 1 ? 3 : -1, index == 2 ? -3 : 1, 0, 1);
    }
    """
    pixel = "float4 main() : SV_Target { return float4(0, 1, 0, 1); }"
    env, args, output = prepare_draw_cache(directory, consumer, pixel, {}, "", vertex_hlsl=vertex)
    env.update(X2W_D3D12_DEBUG="1", X2W_TEST_INDEX_PROBE=mode, X2W_READ_CACHE="1",
               X2W_TEST_READ_CACHE_XMEM=xmem)
    result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True,
                            errors="replace", timeout=45)
    measured = re.search(r"read cache: (\d+) queries / (\d+) indexed draws", result.stdout)
    if (result.returncode or not measured or int(measured[1]) >= int(measured[2])
            or re.search(r"x2w-d3d12-debug [^\n]*sev=[01]\b", result.stderr)):
        raise AssertionError(f"indexed cache ({result.returncode})\n{result.stdout}\n{result.stderr[-4000:]}")
    expected = bytes((0, 255, 0, 255)) * 32  # Full-screen triangle over the 8 x 4 CPU reference.
    actual = output.read_bytes()
    if actual != expected:
        if mode == "read-cache-clipped" and actual == CLIPPED * 32:
            print("FAIL: clipped index control differs from the full-screen CPU reference", flush=True)
            return 1
        raise AssertionError(f"indexed cache readback differs from the CPU reference: {actual.hex()}")
    print(result.stdout)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        if len(sys.argv) == 5 and sys.argv[3] in ("read-cache", "read-cache-clipped"):
            sys.exit(check_read_cache(Path(temporary), sys.argv[1], sys.argv[2], sys.argv[4], sys.argv[3]))
        else:
            sys.exit(check(Path(temporary), *sys.argv[1:]))
