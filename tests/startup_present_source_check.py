"""The `x2w-render present source` line names what wrote the presented surface: the key of the translated draw that drew into it,
or "no translated draw" for a target that only holds its clear colour (startup_stream_consumer, X2W_TEST_PRESENT_SOURCE=draw|clear;
the surface is read back at present 6)."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex


def check(directory, consumer, backend):
    cache = directory / "cache"
    cache.mkdir()
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(False))
    args = [consumer, vs, ps, root]
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1", X2W_SHADER_CACHE=str(cache))
    env.setdefault("X2W_D3D12_DEBUG", "0")
    subprocess.run(args, env=env, check=True)  # publishes the request
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
    drawn_key = f"{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes()):016x}"
    lines = {}
    for mode in ("draw", "clear"):
        result = subprocess.run(args + [backend, output], env=dict(env, X2W_TEST_PRESENT_SOURCE=mode), capture_output=True,
                                text=True, timeout=60)
        if backend == "hardware" and result.returncode == 77:
            print("No D3D12 feature-level 12 hardware adapter")
            return 77
        if result.returncode:
            raise RuntimeError(f"present source {mode} failed on {backend}: {result.returncode}\n{result.stderr}")
        lines[mode] = [l for l in result.stderr.splitlines() if "x2w-render present source" in l]
        seen = [l[:160] for l in result.stderr.splitlines() if "slot32" in l or "x2w-present" in l or "x2w-render" in l or "heartbeat" in l]
        assert lines[mode], f"{mode}: no present source line; present/render lines ({len(result.stderr.splitlines())} stderr lines): {seen[-12:]}"
    named = re.search(r"present source at [\d.]+ s: last translated draw into the presented surface key ([0-9a-f]{16}) "
                      r"\(target 0, format \d+, (\d+) draws into it\)", lines["draw"][0])
    assert named and named.group(1) == drawn_key and int(named.group(2)) >= 1, (drawn_key, lines["draw"])
    assert "no translated draw wrote the presented surface" in lines["clear"][0] and "last translated draw" not in lines["clear"][0], lines["clear"]
    print(f"{backend}: present source names the drawn key {drawn_key}; a clear-only target says no translated draw")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
