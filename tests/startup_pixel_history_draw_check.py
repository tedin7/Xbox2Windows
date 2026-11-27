"""Exercise the runtime pixel-history hook through the real translated draw path and queue_present (#2843)."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import pixel, vertex


def run_case(consumer, backend, vs, ps, root, pixels, env, dump_dir):
    run_env = dict(env, X2W_DRAW_DUMP_FRAME="1", X2W_DRAW_DUMP_DIR=str(dump_dir))
    result = subprocess.run([consumer, vs, ps, root, backend, pixels], env=run_env,
                            capture_output=True, text=True, timeout=60)
    print(result.stdout, end="")
    for line in result.stderr.splitlines():
        if "debug layer:" in line or "x2w-d3d12-debug " in line or "x2w-draw pixel history:" in line:
            print(line)
    if backend == "hardware" and result.returncode == 77:
        return result
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"draw failed on {backend}: {result.returncode}\n{result.stderr}")
    return result


def parse_history(path):
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines and lines[0].startswith("x2w-pixel-history 1 "), lines
    return lines


def check(directory, consumer, backend):
    cache = directory / "cache"
    cache.mkdir()
    vs, ps, root, pixels = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(True))

    base_env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
                    X2W_SHADER_CACHE=str(cache), X2W_TEST_PIXEL_HISTORY="1")
    for name in ("X2W_DRAW_DUMP_PIXEL", "X2W_DRAW_DUMP_FRAME", "X2W_DRAW_DUMP_DIR"):
        base_env.pop(name, None)
    base_env.setdefault("X2W_D3D12_DEBUG", "0")

    # Publish and translate the green triangle pipeline once, before any draw runs.
    subprocess.run([consumer, vs, ps, root], env=base_env, check=True)
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=base_env, check=True)
    expected_key = f"{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes()):016x}"
    assert int(expected_key, 16) != 0

    # 1. Watched pixel (3, 2) inside scissor [2, 6) x [1, 3): green (0, 255, 0, 255).
    covered_dir = directory / "dump_covered"
    covered_dir.mkdir()
    res = run_case(consumer, backend, vs, ps, root, pixels,
                   dict(base_env, X2W_DRAW_DUMP_PIXEL="3,2"), covered_dir)
    if backend == "hardware" and res.returncode == 77:
        print("No D3D12 feature-level 12 hardware adapter")
        return 77

    # Check the target readback itself has distinct origin (0,0)=blue vs (3,2)=green.
    raw_pixels = pixels.read_bytes()
    assert len(raw_pixels) == 8 * 4 * 4, len(raw_pixels)
    assert raw_pixels[0:4] == bytes((0, 0, 255, 255)), raw_pixels[0:4].hex()
    texel_3_2 = (2 * 8 + 3) * 4
    assert raw_pixels[texel_3_2:texel_3_2 + 4] == bytes((0, 255, 0, 255)), raw_pixels[texel_3_2:texel_3_2 + 4].hex()

    history_path = covered_dir / "pixel_history.txt"
    assert history_path.is_file(), f"missing {history_path}"
    lines = parse_history(history_path)
    assert lines[0] == "x2w-pixel-history 1 samples=1", lines
    assert len(lines) == 2, lines
    line = lines[1]
    assert re.search(rf"\bdraw=1\b.*\bframe=1\b.*\bx=3 y=2\b.*\bkey={expected_key}\b.*\brgba=0,255,0,255\b", line), line
    print(f"{backend} covered (3,2): {line}")

    # 2. Control: target origin (0, 0) lies outside scissor, so its texel stays clear blue (0, 0, 255, 255);
    # an origin-copy mutant would report blue at (3, 2) instead of green.
    origin_dir = directory / "dump_origin"
    origin_dir.mkdir()
    run_case(consumer, backend, vs, ps, root, pixels,
             dict(base_env, X2W_DRAW_DUMP_PIXEL="0,0"), origin_dir)
    origin_lines = parse_history(origin_dir / "pixel_history.txt")
    assert len(origin_lines) == 2, origin_lines
    assert re.search(rf"\bdraw=1\b.*\bframe=1\b.*\bx=0 y=0\b.*\bkey={expected_key}\b.*\brgba=0,0,255,255\b",
                     origin_lines[1]), origin_lines[1]
    print(f"{backend} origin control (0,0): {origin_lines[1]}")

    # 3. Control: off-target pixel outside the 8x4 target produces no pixel_history.txt sample.
    off_dir = directory / "dump_off_target"
    off_dir.mkdir()
    run_case(consumer, backend, vs, ps, root, pixels,
             dict(base_env, X2W_DRAW_DUMP_PIXEL="8,4"), off_dir)
    off_history = off_dir / "pixel_history.txt"
    assert not off_history.exists() or parse_history(off_history)[0] == "x2w-pixel-history 1 samples=0", off_history.read_text()
    print(f"{backend} off-target control (8,4): no sample")

    # 4. Control: without X2W_DRAW_DUMP_PIXEL, no pixel_history.txt is created and full-dump BMP is written.
    full_dir = directory / "dump_full"
    full_dir.mkdir()
    run_case(consumer, backend, vs, ps, root, pixels, base_env, full_dir)
    assert not (full_dir / "pixel_history.txt").exists(), "pixel_history.txt written without X2W_DRAW_DUMP_PIXEL"
    bmps = sorted(full_dir.glob(f"draw_*_{expected_key}_3.bmp"))
    assert len(bmps) == 1 and bmps[0].stat().st_size >= 54 + 8 * 4 * 4, bmps
    print(f"{backend} full-dump control: {bmps[0].name} written, no pixel_history.txt")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
