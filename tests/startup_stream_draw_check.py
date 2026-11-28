"""Draw two source-built gfx1013 pixel shaders through a slot-86 wrapper and check every RGBA byte."""
import ctypes
from ctypes import wintypes
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


def check(directory, consumer, backend):
    cache = directory / "cache"
    cache.mkdir()
    recipes = directory / "recipes"
    recipes.mkdir()
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex())
    args = [consumer, vs, ps, root]
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache), X2W_PSO_RECIPE_DIR=str(recipes))
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
        # The translated stream draw creates its desktop PSO through pipeline_for. A second
        # process with the same recipe directory must replay it instead of reconstructing the
        # state from scratch.
        assert list(recipes.glob("pso_*.txt")), "the successful draw wrote no PSO recipe"
        replay = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
        assert replay.returncode == 0, replay.stderr
        assert "pso recipes" in replay.stderr and "precreated" in replay.stderr, replay.stderr
        assert output.read_bytes() == pixels, "the recipe replay changed the drawn pixels"
        # A second writer cannot enter while this process owns the recipe file exclusively. The
        # draw must still render through the PSO it creates; only the cache append is skipped.
        recipe_key = gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes())
        recipe_file = recipes / f"pso_{recipe_key:016x}.txt"
        assert recipe_file.is_file(), f"missing recipe {recipe_file}"
        size_before = recipe_file.stat().st_size
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                         wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.CreateFileW(str(recipe_file), 0x80000000, 0, None, 3, 0, None)
        assert handle != wintypes.HANDLE(-1).value, ctypes.get_last_error()
        try:
            locked = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
            assert locked.returncode == 0, locked.stderr
            assert output.read_bytes() == pixels, "the locked recipe path changed the drawn pixels"
            assert recipe_file.stat().st_size == size_before, "a contending draw appended to the locked recipe"
        finally:
            kernel32.CloseHandle(handle)
        print(f"{backend}: exclusive recipe ownership keeps the draw on the fallback path")
    # Hold back the last-published marker while keeping the generated shader bytecode in place.
    key = gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes())
    manifest = cache / f"{key:016x}.txt"
    green_recipe = recipes / f"pso_{key:016x}.txt"
    assert green_recipe.is_file(), f"missing green recipe {green_recipe}"
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

    # Seed the file with valid records that belong to another translation key. They parse but
    # never replay, so the next successful draw appends the matching record as number 64 and the
    # process after it precreates that record. At 64 records the writer must refuse the 65th.
    ps.write_bytes(pixel(True))
    original = green_recipe.read_text(encoding="ascii")
    records = original.split("x2w-pso-recipe 1\n")
    assert len(records) >= 2, original
    original = "x2w-pso-recipe 1\n" + records[1]
    foreign = re.sub(r"(?m)^translation 0x[0-9a-f]{16}$", "translation 0xdeadbeefcafef00d", original)
    assert foreign != original
    green_recipe.write_text(foreign * 63, encoding="ascii")
    run = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert output.read_bytes() == observations[1]
    assert green_recipe.read_text(encoding="ascii").count("x2w-pso-recipe 1") == 64
    replay = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
    assert replay.returncode == 0, replay.stderr
    assert "precreated 1" in replay.stderr, replay.stderr
    assert output.read_bytes() == observations[1]

    green_recipe.write_text(foreign * 64, encoding="ascii")
    run = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert output.read_bytes() == observations[1]
    assert green_recipe.read_text(encoding="ascii").count("x2w-pso-recipe 1") == 64, \
        "a full recipe file accepted a 65th record"
    print(f"{backend}: 63+1 records precreate; 64 records refuse the 65th")

    # A record without its final LF still parses, but appending would join `end` and the next
    # header on one line. The draw must render and leave the parseable file byte-for-byte intact.
    no_final_lf = green_recipe.read_bytes()
    assert no_final_lf.rstrip(b"\r\n").endswith(b"end"), no_final_lf[-40:]
    green_recipe.write_bytes(no_final_lf[:-1])
    before = green_recipe.read_bytes()
    run = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert output.read_bytes() == observations[1]
    assert green_recipe.read_bytes() == before, "a no-LF recipe file was modified by the refused append"
    replay = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
    assert replay.returncode == 0, replay.stderr
    assert output.read_bytes() == observations[1]
    print(f"{backend}: a recipe without a final LF stays readable and refuses the append")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
