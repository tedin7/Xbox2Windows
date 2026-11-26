"""The gray-world log lines of the console present (render mode): picture mean and spread, and the per-interval counters.

usage: startup_present_gray_check.py <startup_stream_consumer>

The consumer presents a 64x64 RGBA8 render target 600 times through queue slot 32 with X2W_D3D12_RENDER=1 on WARP. The stand-in
reads the picture back and logs `x2w-render presented surface 64x64 format 28, readback N, mean level M/255, spread S`, and
on every 300th present `x2w-gray interval (frame F): ...` with the change of nine counters since the last interval.

  present-flat     cleared to (128, 128, 128, 255): mean level 128.00, spread 0.00 (a flat fill or fade);
  present-split  the left half cleared to white (255) over black (0): mean level 127.50, spread 127.50 (a picture full of detail).
                   The same mean, so only the spread tells the two pictures apart.

Both runs must log the interval line exactly twice (frames 300 and 600) with all eight named counters, none negative.
Exit 77 when the adapter is not available.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import startup_stream_check as S

PRESENTED = re.compile(r'x2w-render presented surface 64x64 format \d+, readback (\d+), mean level ([0-9.]+)/255, spread ([0-9.]+)')
INTERVAL = re.compile(r'x2w-gray interval \(frame (\d+)\): ring refusals \+(-?\d+), null bindings \+(-?\d+) \(ring_full \+(-?\d+)\), '
                      r'draws not drawn \+(-?\d+), translated draws \+(-?\d+), compute translated \+(-?\d+), compute refused \+(-?\d+), '
                      r'textures unreadable \+(-?\d+), list copies refused \+(-?\d+)')


def present(consumer, directory, mode):
    cache = directory / 'cache'
    cache.mkdir(exist_ok=True)
    for name, data in (('vertex', S.container(1)), ('pixel', S.container(0))):
        (directory / name).write_bytes(data)
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_RENDER='1', X2W_D3D12_LIST_NOOP='1',
               X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache))
    result = subprocess.run([consumer, directory / 'vertex', directory / 'pixel', directory / 'root', mode], env=env,
                            capture_output=True, text=True)
    if result.returncode == 77:
        print(result.stdout, end='')
        sys.exit(77)
    assert result.returncode == 0, f'{mode}: consumer exit {result.returncode}\n{result.stderr[-3000:]}'
    return result.stderr


def check(consumer, mode, mean, spread):
    with tempfile.TemporaryDirectory() as temporary:
        log = present(consumer, Path(temporary), mode)
    pictures = [PRESENTED.search(line) for line in log.splitlines()]
    pictures = [m for m in pictures if m]
    assert pictures, f'{mode}: no "x2w-render presented" line\n{log[-3000:]}'
    for m in pictures:
        assert float(m.group(2)) == mean and float(m.group(3)) == spread, \
            f'{mode}: mean {m.group(2)} spread {m.group(3)}, expected {mean:.2f} and {spread:.2f}: {m.group(0)}'
    intervals = [m for m in map(INTERVAL.search, log.splitlines()) if m]
    frames = [int(m.group(1)) for m in intervals]
    assert frames == [300, 600], f'{mode}: interval lines at frames {frames}, expected [300, 600]\n{log[-3000:]}'
    assert all(int(v) >= 0 for m in intervals for v in m.groups()[1:]), [m.group(0) for m in intervals]
    print(f'{mode}: {len(pictures)} presented lines at mean {mean:.2f} spread {spread:.2f}, interval lines at {frames}')


def main(consumer):
    check(consumer, 'present-flat', 128.0, 0.0)
    check(consumer, 'present-split', 127.5, 127.5)
    print('PASS')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
