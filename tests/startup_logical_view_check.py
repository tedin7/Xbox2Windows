"""Exercise guest slots 14, 17, 32 and 36 and compare desktop draw pixels for two oversized heaps."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_stream_draw_check import gfx2hlsl, pixel, vertex


def check(directory, consumer, backend):
    cache = directory / "cache"
    cache.mkdir()
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(False))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache), X2W_TEST_LOGICAL_VIEW="1")
    args = [consumer, vs, ps, root]
    subprocess.run(args, env=env, check=True)
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
    # Isolate table publication from instruction lowering. Keep the fixture's generated
    # screen triangle and replace its constant-colour PS with a single bound buffer read.
    manifest, = (path for path in cache.glob("*.txt") if path.name not in ("revision.txt", "x2w_key_seed.txt"))
    with manifest.open("a", encoding="ascii") as file:
        file.write("\nsrv 0 buffer table 0 0\n")
    shader = manifest.with_suffix(".ps.hlsl")
    shader.write_text("ByteAddressBuffer src : register(t0);\n"
                      "float4 main() : SV_Target { return float4(src.Load(0)/255.0,0,0,1); }\n")
    subprocess.run([env["X2W_FXC"], "/nologo", "/T", "ps_5_0", "/E", "main", "/Fo",
                    manifest.with_suffix(".ps.cso"), shader], check=True)
    result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode == 77:
        return 77
    if result.returncode:
        raise RuntimeError(f"consumer failed ({result.returncode}): {result.stderr[-4500:]}")
    red = [16, 32, 48, 0, 128, 0, 0, 32]
    expected = bytes(component for value in red for component in (value, 0, 0, 255)) * 4
    actual = output.read_bytes()
    if actual != expected:
        raise AssertionError(f"actual {actual.hex()} expected {expected.hex()}\n{result.stderr[-4500:]}")
    print(f"{backend}: both heaps, native boundary, logical end, invalidated and wrong-heap tables; all 32 pixels exact")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
