"""Draw a small texture given the 4 KiB allocation answer and read the pixels back (src/startup_stubs/small_texture_4k.h).

A 32x32 R8G8B8A8 texture with two mips at Alignment 0x1000 is written through XG slot 8 into memory of exactly the
size XG slot 2 reports (0x2000), followed by a no-access guard page, placed through device slot 133 and sampled by a
translated draw into a 48x32 target: mip 0 at x < 32, mip 1 at x >= 32 and y < 16, black elsewhere. Every texel
holds its own x and y and a per-mip sentinel, so a missing, zero-filled or shifted row, or a mip read from the
wrong place, changes pixels.

usage: startup_small_texture_4k_draw_check.py <startup_stream_consumer> <dxc> warp|hardware
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_stream_check import container, gfx2hlsl

WIDTH, HEIGHT = 48, 32


def texel(x, y, mip):
    return bytes((x, y, (0x11, 0xEE)[mip], 0x5C))


def fixture():
    mip0 = b"".join(texel(x, y, 0) for y in range(32) for x in range(32))
    mip1 = b"".join(texel(x, y, 1) for y in range(16) for x in range(16))
    reference = bytearray(WIDTH * HEIGHT * 4)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if x < 32:
                value = texel(x, y, 0)
            elif y < 16:
                value = texel(x - 32, y, 1)
            else:
                value = bytes(4)
            reference[(y * WIDTH + x) * 4:(y * WIDTH + x + 1) * 4] = value
    return mip0 + mip1, bytes(reference)


def check(work, consumer, dxc, backend):
    cache = work / "cache"
    cache.mkdir()
    vs, ps, root, image, upload = (work / name for name in ("vs", "ps", "root", "pixels", "texels"))
    vs.write_bytes(container(1))
    ps.write_bytes(container(0))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache))
    texels, reference = fixture()
    upload.write_bytes(texels)
    base = [consumer, vs, ps, root, backend, image, upload, "32", "32", "small4k"]
    bootstrap = subprocess.run(base, env=dict(env, X2W_TEST_TRUNCATED_ONLY="1"))
    if bootstrap.returncode == 77:
        print(f"{backend}: D3D12 feature level 12 unavailable")
        return 77
    bootstrap.check_returncode()
    key = f"{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes()):016x}"
    (cache / f"{key}.txt").write_text("udsrc 0 rect\nsrv 0 tex2d table 0 0\n", encoding="ascii")
    sources = {
        "vs": """float4 main(uint id : SV_VertexID) : SV_Position {
            return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
        }""",
        "ps": """Texture2D<float4> tex : register(t0);
        float4 main(float4 position : SV_Position) : SV_Target {
            int2 p = int2(position.xy);
            if (p.x < 32) return tex.Load(int3(p, 0));
            if (p.y < 16) return tex.Load(int3(p.x - 32, p.y, 1));
            return float4(0.0, 0.0, 0.0, 0.0);
        }""",
    }
    for stage, source in sources.items():
        shader = work / f"{stage}.hlsl"
        shader.write_text(source, encoding="ascii")
        subprocess.run([dxc, "-T", f"{stage}_6_0", "-E", "main", "-Fo", cache / f"{key}.{stage}.cso", shader], check=True)

    result = subprocess.run(base, env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    for line in result.stderr.splitlines():
        if "debug layer:" in line or "x2w-d3d12-debug " in line or "compact plan" in line:
            print(line)
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"{backend}: small4k draw failed ({result.returncode})\n{result.stderr[-4000:]}")
    actual = image.read_bytes()
    assert len(actual) == len(reference), f"read {len(actual)} bytes, expected {len(reference)}"
    if actual != reference:
        first = next(i for i in range(0, len(reference), 4) if actual[i:i + 4] != reference[i:i + 4])
        x, y = first // 4 % WIDTH, first // 4 // WIDTH
        bad = sum(actual[i:i + 4] != reference[i:i + 4] for i in range(0, len(reference), 4))
        raise AssertionError(f"{backend}: {bad} pixels differ; first ({x}, {y}) read {actual[first:first + 4].hex()}, "
                             f"expected {reference[first:first + 4].hex()}")
    print(f"{backend}: small4k draw matched all {WIDTH * HEIGHT} reference pixels (32 + 16 distinct rows, two mips)")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
