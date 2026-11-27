"""Draw two source-built gfx1013 pixel shaders through a slot-86 wrapper and check every RGBA byte."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_stream_check import container, gfx2hlsl


def vop1(op, dst, src):
    return 0x7e000000 | dst << 17 | op << 9 | src


def vop2(op, dst, src0, vsrc1):
    return op << 25 | dst << 17 | vsrc1 << 9 | src0


def vertex():
    # Position from vertex ID i: x = 2 * ((2*i) & 2) - 1; y = 2 * (i & 2) - 1.
    words = [vop2(0x1a, 1, 129, 0), vop2(0x1b, 1, 130, 1), vop2(0x1b, 2, 130, 0)]
    for v in (1, 2):
        words.extend([vop1(6, v, 256 + v), vop2(8, v, 244, v), vop2(3, v, 243, v)])
    words.extend([vop1(1, 3, 128), vop1(1, 4, 242), 0xf80008cf, 0x04030201, 0xbf810000])
    return container(1, words)


def pixel(green):
    # v0 = 0.0, v1 = 1.0. Export opaque red or opaque green.
    channels = 0x01000100 if green else 0x01000001
    return container(0, [vop1(1, 0, 128), vop1(1, 1, 242), 0xf800180f, channels, 0xbf810000])


def check(directory, consumer, backend):
    cache = directory / "cache"
    cache.mkdir()
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex())
    args = [consumer, vs, ps, root]
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache))
    env.setdefault("X2W_D3D12_DEBUG", "0")
    observations = []
    for green in (False, True):
        ps.write_bytes(pixel(green))
        subprocess.run(args, env=env, check=True)
        subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
        result = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
        print(result.stdout, end="")
        for line in result.stderr.splitlines():
            if "debug layer:" in line or "x2w-d3d12-debug " in line:
                print(line)
        if backend == "hardware" and result.returncode == 77:
            print("No D3D12 feature-level 12 hardware adapter")
            return 77
        if result.returncode or "-> drawn;" not in result.stderr:
            raise RuntimeError(f"draw failed on {backend}: {result.returncode}\n{result.stderr}")

        pixels = output.read_bytes()
        color = bytes((0, 255, 0, 255) if green else (255, 0, 0, 255))
        clear = bytes((0, 0, 255, 255))
        expected = (color * 4 + clear * 4) * 4
        if pixels != expected:
            raise AssertionError(f"actual {pixels.hex()} expected {expected.hex()}")
        observations.append(pixels)
    # Hold back the last-published marker while keeping the generated shader bytecode in place.
    key = gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes())
    manifest = cache / f"{key:016x}.txt"
    manifest.rename(str(manifest) + ".pending")
    run = subprocess.run(args + [backend, output], env=dict(env, X2W_TEST_PUBLISH_MANIFEST=str(manifest)),
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert Path(str(output) + ".first").read_bytes() == clear * 32
    assert output.read_bytes() == observations[1], "same guest PSO remained clear after manifest publication"
    wrong = sum(observations[0][i:i+4] != observations[1][i:i+4] for i in range(0, 8 * 4 * 4, 4))
    assert wrong == 16, f"swapping the pixel shader produced {wrong} changed pixels"
    print(f"{backend}: red/green draws exact, 32 RGBA pixels each; swapped shader differs at {wrong} of 32")
    # Green has red=0 and alpha=1. Red makes the R8 target nonzero as a separate control.
    # A cut-down record, wrong ancestor or red/alpha swap changes at least one observation.
    for mode, green, value in (("parent-a8", True, 0), ("child-a8", True, 255),
                               ("child-r8", True, 0), ("child-r8", False, 255)):
        ps.write_bytes(pixel(green))
        result = subprocess.run(args + [backend, output, mode], env=env, capture_output=True, text=True)
        if result.returncode or "-> drawn;" not in result.stderr:
            raise RuntimeError(f"{mode} failed on {backend}: {result.returncode}\n{result.stderr}")
        expected = (bytes([value]) * 4 + bytes(4)) * 4
        if output.read_bytes() != expected:
            raise AssertionError(f"{mode}: got {output.read_bytes().hex()} expected {expected.hex()}")
        print(f"{backend} {mode} green={green}: 32 pixels exact, drawn byte {value}")
    return 0


def check_position_exports(directory, consumer, backend):
    from startup_image_3d_check import compile_hlsl
    from startup_packed_position_check import CLIP, CLIP_BOTH, CULL, LAYER, MISC, VIEWPORT, vertex_position_words

    cache = directory / "cache"
    cache.mkdir()
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(container(1, vertex_position_words()))
    ps.write_bytes(pixel(False))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache), X2W_TEST_POSITION_EXPORTS="1")
    args = [consumer, vs, ps, root, backend, output, "slot57"]
    fxc = gfx2hlsl.find_fxc()
    assert fxc, "FXC is required for the WARP position-export check"

    red, clear = bytes((255, 0, 0, 255)), bytes((0, 0, 255, 255))

    def reference(layer, coverage):
        centers = [(2 * (x + 0.5) / 8) - 1 for x in range(8)]
        return b"".join(red if z == layer and coverage(x) else clear
                        for z in range(2) for y in range(4) for x in centers)

    full = reference(0, lambda x: True)
    half = reference(0, lambda x: x >= 0)
    layer_one = reference(1, lambda x: True)
    empty = reference(0, lambda x: False)

    clip_bytecode = None
    for label, control, expected, mutation in (
        ("point size", MISC | (1 << 16), full, None),
        ("clip", CLIP, half, ("raw.posx2.x", "raw.posx1.x", full)),
        ("clip in both vectors", CLIP_BOTH, empty, ("raw.posx3.x", "raw.posx1.x", half)),
        ("cull", CULL, empty, ("raw.posx3.x", "raw.posx1.x", full)),
        ("layer", LAYER, layer_one, ("SV_RenderTargetArrayIndex", "SV_ViewportArrayIndex", half)),
        ("viewport", VIEWPORT, half, ("SV_ViewportArrayIndex", "SV_RenderTargetArrayIndex", layer_one)),
        ("disabled", 0, full, None),
    ):
        command = args + [f"vsout={control:x}"]
        warm = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
        assert warm.returncode == 0, warm.stderr
        key = gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes(), vs_out_cntl=control)
        assert (cache / f"req_{key:016x}.vsout").read_bytes() == control.to_bytes(4, "little")
        result = subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env,
                                check=True, capture_output=True, text=True, timeout=25)
        manifest = cache / f"{key:016x}.txt"
        assert manifest.is_file(), (result.stdout, result.stderr)
        assert "vsposx 1 2 3" in manifest.read_text().splitlines()

        def draw():
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            assert result.returncode == 0 and "-> drawn;" in result.stderr, result.stderr
            return output.read_bytes()

        actual = draw()
        assert actual == expected, f"{label}: {actual.hex()} != {expected.hex()}"
        bytecode = cache / f"{key:016x}.vs.cso"
        original = bytecode.read_bytes()
        if control == CLIP:
            clip_bytecode = original
        if mutation:
            old, new, bad_reference = mutation
            source = (cache / f"{key:016x}.vs.hlsl").read_text()
            assert source.count(old) == 1, (label, old)
            mutated = compile_hlsl(fxc, "fxc", source.replace(old, new), "mutant", "vs_5_1", directory)
            mutant = mutated.read_bytes()
        elif control == 0:
            mutant, bad_reference = clip_bytecode, half
        else:
            mutant = None
        if mutant is not None:
            bytecode.write_bytes(mutant)
            try:
                actual = draw()
            finally:
                bytecode.write_bytes(original)
            assert actual == bad_reference and actual != expected, f"{label} mutant: {actual.hex()}"
        print(f"{backend} {label}: 64 RGBA texels matched the CPU reference"
              + (", mutant rejected by the same reference" if mutant is not None else ""))
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        if len(sys.argv) == 4 and sys.argv[3] == "positions":
            sys.exit(check_position_exports(Path(temp), *sys.argv[1:3]))
        sys.exit(check(Path(temp), *sys.argv[1:]))
