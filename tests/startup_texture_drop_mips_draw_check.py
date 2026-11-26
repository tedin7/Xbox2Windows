"""A translated draw reads a big texture that lost its top mip level (src/startup_stubs/texture_drop_mips.h).

A 256x256 R8G8B8A8 texture with 9 mips, every mip one solid colour (mip k: 0x10 + 0x20 k, 0x40 + k, 0x80, 0xff) in the
title's row layout, is placed through device slot 133 (X2W_D3D12_LAZY_TEXTURES=1) and read by a translated pixel draw
into a 64x64 target. The manifest is written by hand: `srv 0 tex2d table 0 0`, an all-zero static sampler, and the
`srvuse` line under test.

  sampling only, drop on     the draw reads level 0 of the smaller texture: mip 1 colour; the runtime logs the drop
  sampling only, drop off    mip 0 colour (the opt-in is off by default)
  texel addressed (Load)     mip 0 colour at every texel, nothing dropped (reason texel_addressed)
  no `srvuse` line           mip 0 colour, nothing dropped (a draw that says nothing is unknown)
  threshold above the size   mip 0 colour, nothing dropped
  two levels                 the draw reads mip 2 colour

usage: startup_texture_drop_mips_draw_check.py <startup_stream_consumer> <dxc> warp|hardware
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_stream_check import container, gfx2hlsl

SIZE = 64


def mip_color(mip):
    return bytes((0x10 + 0x20 * mip, 0x40 + mip, 0x80, 0xFF))


SAMPLE_PS = """Texture2D<float4> tex : register(t0);
SamplerState smp : register(s0);
float4 main(float4 position : SV_Position) : SV_Target {
    return tex.SampleLevel(smp, float2(0.5, 0.5), 0);
}"""

TEXEL_PS = """Texture2D<float4> tex : register(t0);
float4 main(float4 position : SV_Position) : SV_Target {
    return tex.Load(int3(int2(position.xy), 0));
}"""

VS = """float4 main(uint id : SV_VertexID) : SV_Position {
    return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
}"""

# name, pixel shader, manifest tail, environment, expected mip, must log a drop
CASES = (
    ("sampling only, drop on", SAMPLE_PS, "srvuse 0 sample\nsmp 0 0 0 0 0\n",
     {"X2W_D3D12_TEXTURE_DROP_MIPS": "1"}, 1, True),
    ("sampling only, drop off", SAMPLE_PS, "srvuse 0 sample\nsmp 0 0 0 0 0\n", {}, 0, False),
    ("texel addressed, drop on", TEXEL_PS, "srvuse 0 texel\n", {"X2W_D3D12_TEXTURE_DROP_MIPS": "1"}, 0, False),
    ("no srvuse line, drop on", SAMPLE_PS, "smp 0 0 0 0 0\n", {"X2W_D3D12_TEXTURE_DROP_MIPS": "1"}, 0, False),
    ("threshold above the size", SAMPLE_PS, "srvuse 0 sample\nsmp 0 0 0 0 0\n",
     {"X2W_D3D12_TEXTURE_DROP_MIPS": "1", "X2W_D3D12_TEXTURE_DROP_MIPS_MIN_BYTES": "16777216"}, 0, False),
    ("two levels", SAMPLE_PS, "srvuse 0 sample\nsmp 0 0 0 0 0\n", {"X2W_D3D12_TEXTURE_DROP_MIPS": "2"}, 2, True),
    # The same draws with the resolved-view cache on (X2W_VIEW_CACHE): the view is built into the cache slot and copied,
    # so the pixels and the dropped levels must not change.
    ("view cache, drop on", SAMPLE_PS, "srvuse 0 sample\nsmp 0 0 0 0 0\n",
     {"X2W_D3D12_TEXTURE_DROP_MIPS": "1", "X2W_VIEW_CACHE": "1"}, 1, True),
    ("view cache, drop off", SAMPLE_PS, "srvuse 0 sample\nsmp 0 0 0 0 0\n", {"X2W_VIEW_CACHE": "1"}, 0, False),
    ("view cache, two levels", SAMPLE_PS, "srvuse 0 sample\nsmp 0 0 0 0 0\n",
     {"X2W_D3D12_TEXTURE_DROP_MIPS": "2", "X2W_VIEW_CACHE": "1"}, 2, True),
)


def run_case(work, consumer, dxc, backend, case):
    name, ps_source, tail, extra, mip, dropped = case
    cache = work / "cache"
    cache.mkdir()
    vs, ps, root, image, upload = (work / n for n in ("vs", "ps", "root", "pixels", "texels"))
    vs.write_bytes(container(1))
    ps.write_bytes(container(0))
    upload.write_bytes(bytes(4))  # the scenario builds its own texels; the file only marks a textured draw
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache), X2W_D3D12_LAZY_TEXTURES="1",
               X2W_D3D12_TEXTURE_DROP_MIPS_MIN_BYTES="65536")
    env.update(extra)
    base = [consumer, vs, ps, root, backend, image, upload, str(SIZE), str(SIZE), "dropmips"]
    bootstrap = subprocess.run(base, env=dict(env, X2W_TEST_TRUNCATED_ONLY="1"))
    if bootstrap.returncode == 77:
        print(f"{backend}: D3D12 feature level 12 unavailable")
        return 77
    bootstrap.check_returncode()
    key = f"{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes()):016x}"
    (cache / f"{key}.txt").write_text("udsrc 0 rect\nsrv 0 tex2d table 0 0\n" + tail, encoding="ascii")
    for stage, source in (("vs", VS), ("ps", ps_source)):
        shader = work / f"{stage}.hlsl"
        shader.write_text(source, encoding="ascii")
        subprocess.run([dxc, "-T", f"{stage}_6_0", "-E", "main", "-Fo", cache / f"{key}.{stage}.cso", shader], check=True)
    result = subprocess.run(base, env=env, capture_output=True, text=True)
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"{backend}: {name}: draw failed ({result.returncode})\n{result.stderr[-3000:]}")
    actual = image.read_bytes()
    reference = mip_color(mip) * (SIZE * SIZE)
    if actual != reference:
        first = next(i for i in range(0, len(reference), 4) if actual[i:i + 4] != reference[i:i + 4])
        raise AssertionError(f"{backend}: {name}: pixel {first // 4} read {actual[first:first + 4].hex()}, "
                             f"expected {reference[first:first + 4].hex()} (mip {mip})")
    logged = "texture drop-mips" in result.stderr
    if logged != dropped:
        raise AssertionError(f"{backend}: {name}: drop logged = {logged}, expected {dropped}\n{result.stderr[-2000:]}")
    print(f"{backend}: {name}: all {SIZE * SIZE} pixels are the mip {mip} colour")
    return 0


def check(work, consumer, dxc, backend):
    for number, case in enumerate(CASES):
        directory = work / f"case{number}"
        directory.mkdir()
        code = run_case(directory, consumer, dxc, backend, case)
        if code:
            return code
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
