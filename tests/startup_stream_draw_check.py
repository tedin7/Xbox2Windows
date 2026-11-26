"""Draw two source-built gfx1013 pixel shaders through a slot-86 wrapper and check every RGBA byte."""
import os
import re
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
    # The present-source line names what wrote the presented surface: the translated draw's key, or "no translated draw" when the
    # target only holds its clear colour (the line is printed at present 6).
    ps.write_bytes(pixel(False))
    drawn_key = f"{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes()):016x}"
    sources = {}
    for mode in ("draw", "clear"):
        result = subprocess.run(args + [backend, output], env=dict(env, X2W_TEST_PRESENT_SOURCE=mode), capture_output=True, text=True, timeout=30)
        if result.returncode or "x2w-render present source" not in result.stderr:
            raise RuntimeError(f"present source {mode} failed on {backend}: {result.returncode}\n{result.stderr}")
        sources[mode] = [l for l in result.stderr.splitlines() if "x2w-render present source" in l]
    named = re.search(r"present source at [\d.]+ s: last translated draw into the presented surface key ([0-9a-f]{16}) \(target 0, format \d+, (\d+) draws into it\)", sources["draw"][0])
    assert named and named.group(1) == drawn_key and int(named.group(2)) >= 1, (drawn_key, sources["draw"])
    assert "no translated draw wrote the presented surface" in sources["clear"][0] and "last translated draw" not in sources["clear"][0], sources["clear"]
    print(f"{backend}: present source names the drawn key {drawn_key}; a clear-only target says no translated draw")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
