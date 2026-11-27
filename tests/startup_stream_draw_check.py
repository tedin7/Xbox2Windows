"""Draw two source-built gfx1013 pixel shaders through a slot-86 wrapper and check every RGBA byte."""
import os
from pathlib import Path
import re
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


def census_rows(log, slot, *, device=False):
    prefix = f"x2w-noop census{' device' if device else ''} slot {slot}"
    headers = re.findall(rf"^{prefix}: calls (\d+), shapes (\d+), overflow (\d+)$", log, re.M)
    assert len(headers) == 1, f"{prefix} expected one exit report, got {len(headers)}"
    calls, shapes, overflow = map(int, headers[0])
    rows = re.findall(
        rf"^{prefix} shape \d+: calls (\d+), before draw (\d+), before indexed (\d+), "
        r"before dispatch (\d+), before indirect (\d+), after pipeline (\d+), args (.+)$",
        log, re.M,
    )
    assert len(rows) == shapes and sum(int(row[0]) for row in rows) + overflow == calls, prefix
    return calls, overflow, [(tuple(map(int, row[:6])), row[6].split()) for row in rows]


def check_census(log, *, mutant):
    calls, overflow, rows = census_rows(log, 110)
    assert (calls, overflow, len(rows)) == (9, 0, 3)
    by_count = {counts[0]: (counts, args) for counts, args in rows}
    assert set(by_count) == {5, 3, 1}
    sample = bytes(71 + 3 * i for i in range(32)).hex()
    assert by_count[5] == ((5, 5, 0, 0, 0, 1),
                           ["wrapper:CommandList", "wrapper:Resource/buffer", "small:11",
                            "zero", f"readable:{sample}", "zero"])
    assert by_count[1] == ((1, 0, 0, 0, 0, 0), ["wrapper:CommandList", "zero", "zero", "zero", "zero", "other"])
    counts, args = by_count[3]
    assert counts == (3, 0, 0, 3, 0, 0)
    assert args[0] == "wrapper:CommandList" and args[2:] == ["upload", "zero", "zero", "small:29"]
    calls, overflow, rows = census_rows(log, 111)
    assert (calls, overflow, len(rows)) == (13, 1, 12) and all(counts[0] == 1 for counts, _ in rows)
    assert census_rows(log, 61, device=True) == (
        1, 0, [((1, 0, 0, 0, 0, 0), ["wrapper:Device", "small:23", "zero", "zero", "zero", "zero"])]
    )
    if mutant:
        assert args[1].startswith("readable:"), "mutant did not route the token to the readable-pointer classifier"
    if args[1] != "view:slot18/SRV":
        raise ViewShapeMismatch("SRV token did not retain its view shape")


class ViewShapeMismatch(AssertionError):
    pass


def run_census(args, env, output, ps, cache):
    ps.write_bytes(pixel(False))
    subprocess.run(args, env=env, check=True)
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
    env = dict(env, X2W_TEST_NOOP_CENSUS="1", X2W_D3D12_DEVICE_NOOP="1")
    env.pop("X2W_TEST_NOOP_CENSUS_MUTANT", None)
    expected = (bytes((255, 0, 0, 255)) * 4 + bytes((0, 0, 255, 255)) * 4) * 4
    for mutant in (False, True):
        run_env = dict(env, X2W_TEST_NOOP_CENSUS_MUTANT="1") if mutant else env
        output.unlink(missing_ok=True)
        result = subprocess.run(args + ["warp", output], env=run_env, capture_output=True, text=True, timeout=30)
        if result.returncode or "-> drawn;" not in result.stderr:
            raise RuntimeError(f"census draw failed ({result.returncode}): {result.stderr}")
        assert output.read_bytes() == expected, "translated WARP pixels differ from the CPU reference"
        try:
            check_census(result.stderr, mutant=mutant)
        except ViewShapeMismatch:
            if not mutant:
                raise
        else:
            assert not mutant, "view-as-pointer mutant passed the census oracle"
    print("WARP census: 32 pixels exact in both runs, 3 shapes and 1 overflow verified, view mutant rejected")
    return 0


def check(directory, consumer, backend, mode=None):
    cache = directory / "cache"
    cache.mkdir()
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex())
    args = [consumer, vs, ps, root]
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache))
    env.setdefault("X2W_D3D12_DEBUG", "0")
    if mode == "noop-census":
        return run_census(args, env, output, ps, cache)
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


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
