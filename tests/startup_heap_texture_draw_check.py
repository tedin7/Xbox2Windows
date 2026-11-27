"""Replay two directly indexed texture loads and a raw load through the draw shim.

With no arguments, check the translations and the readback comparator on the CPU.
GPU usage: startup_heap_texture_draw_check.py <startup_stream_consumer> <dxc> warp|hardware
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_gfx2hlsl_dynamic_heap_check import G, IMAGE_PS, PS, VS_WORDS, root_blob
from startup_stream_check import container


def fixture():
    textures = [b"".join(bytes((red + x, green + y, blue, 0xFF)) for y in range(4) for x in range(4))
                for red, green, blue in [(0x19, 0x38, 0x65), (0xAC, 0xD1, 0x4E)]]
    return b"".join(textures), b"".join(texture[:4] for texture in textures)


def translate(code, root):
    pipe = G.Pipeline(VS_WORDS, code, root)
    vs, ps = pipe.hlsl_vs(), pipe.hlsl_ps()
    manifest = pipe.manifest()
    assert "dynamic_heap\n" in manifest and pipe.ps_profile() == "ps_6_6", manifest
    lines = manifest.splitlines()
    if code == IMAGE_PS:
        assert lines[lines.index("dynamic_heap") + 1] == "dynamic_heap_images", manifest
    else:
        assert "dynamic_heap_images" not in lines, manifest
    assert "udsrc 1 rootconst 0 0\n" in manifest, manifest
    assert "(ud1 + 0u) >> 5;" in ps and "ResourceDescriptorHeap[" in ps, ps
    assert ("Texture2D<float4> heap_tex_" in ps) == (code == IMAGE_PS), ps
    return vs, "vs_6_6", ps, pipe.ps_profile(), manifest


def mismatches(actual, expected):
    assert len(actual) == len(expected) == 8, (len(actual), len(expected))
    return [f"pixel {i // 4} read {actual[i:i + 4].hex()}, expected {expected[i:i + 4].hex()}"
            for i in range(0, len(expected), 4) if actual[i:i + 4] != expected[i:i + 4]]


def check_cpu():
    translate(IMAGE_PS, root_blob())
    translate(PS, root_blob())
    upload, expected = fixture()
    assert len(upload) == 128 and expected == upload[:4] + upload[64:68]
    assert expected[:4] != expected[4:] and not mismatches(expected, expected)
    assert len(mismatches(bytes(8), expected)) == 2
    print("heap texture translations, uploaded texel reference and mismatch comparator match")


def write_container(path, stage, code):
    path.write_bytes(container(stage, struct.unpack(f"<{len(code) // 4}I", code)))


def check_gpu(work, consumer, dxc, backend):
    cache = work / "cache"
    cache.mkdir()
    vs, ps, root, pixels, upload = (work / name for name in ("vs", "ps", "root", "pixels", "texels"))
    write_container(vs, 1, VS_WORDS)
    write_container(ps, 0, IMAGE_PS)
    data, expected = fixture()
    upload.write_bytes(data)
    args = [consumer, vs, ps, root, backend, pixels, upload]
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache), X2W_TEST_HEAP_TEXTURE_DRAW="image", X2W_TEST_SKIP_HEAP_TEXTURE_VIEWS="0")
    bootstrap = subprocess.run(args, env=dict(env, X2W_TEST_TRUNCATED_ONLY="1"))
    if bootstrap.returncode == 77:
        return 77
    bootstrap.check_returncode()
    root_bytes = root.read_bytes()

    def publish(code):
        write_container(ps, 0, code)
        key = f"{G.pipeline_key(vs.read_bytes(), ps.read_bytes(), root_bytes):016x}"
        vs_source, vs_profile, ps_source, ps_profile, manifest = translate(code, root_bytes)
        for stage, source, profile in [("vs", vs_source, vs_profile), ("ps", ps_source, ps_profile)]:
            hlsl = work / f"{stage}.hlsl"
            hlsl.write_text(source, encoding="ascii")
            subprocess.run([dxc, "-Gis", "-T", profile, "-E", "main", "-Fo", cache / f"{key}.{stage}.cso", hlsl], check=True)
        (cache / f"{key}.txt").write_text(manifest, encoding="ascii")

    def draw(mode, expected, count, *, mutant=False, allow_skip=False):
        pixels.unlink(missing_ok=True)
        draw_env = dict(env, X2W_TEST_HEAP_TEXTURE_DRAW=mode, X2W_TEST_SKIP_HEAP_TEXTURE_VIEWS=str(int(mutant)))
        result = subprocess.run(args, env=draw_env, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode == 77 and allow_skip:
            return 77
        if result.returncode or result.stderr.count("-> drawn;") != count:
            raise RuntimeError(f"{backend} {mode} draw failed ({result.returncode})\n{result.stderr[-4000:]}")
        for line in result.stderr.splitlines():
            if "debug layer:" in line or "heap textures:" in line:
                print(line)
        errors = mismatches(pixels.read_bytes(), expected)
        if mutant:
            assert len(errors) == count, f"expected both old buffer descriptors to fail readback, got {errors}"
            print(f"{backend} buffer-only mutant mismatch: {', '.join(errors)}")
        else:
            assert not errors, f"{backend} {mode} mismatch: {', '.join(errors)}"
            print(f"{backend} {mode} readback matched {expected.hex()}")
        return 0

    publish(IMAGE_PS)
    if draw("image", expected, 2, allow_skip=True) == 77:
        return 77
    if backend == "warp":
        # The old heap binds buffer descriptors at the texture indices. The descriptor mismatch has
        # undefined results, so exercise it only on WARP and require both readbacks to differ.
        draw("image", expected, 2, mutant=True)
    publish(PS)
    return draw("buffer", bytes([255]) * 8, 1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        check_cpu()
    elif len(sys.argv) == 4 and sys.argv[3] in ("warp", "hardware"):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            sys.exit(check_gpu(Path(temporary), *sys.argv[1:]))
    else:
        sys.exit(__doc__)
