"""The gray-world log lines of the console present (render mode): picture mean and spread, and the per-interval counters.

usage: startup_present_gray_check.py <startup_stream_consumer>

The consumer presents a 64x64 RGBA8 render target 600 times through queue slot 32 with X2W_D3D12_RENDER=1 on WARP. The stand-in
reads the picture back and logs `x2w-render presented surface 64x64 format 28, readback N, mean level M/255, spread S`, and
on every 300th present `x2w-gray interval (frame F): ...` with the change of nine counters since the last interval.

  present-flat     cleared to (128, 128, 128, 255): mean level 128.00, spread 0.00 (a flat fill or fade);
  present-split  the left half cleared to white (255) over black (0): mean level 127.50, spread 127.50 (a picture full of detail).
                   The same mean, so only the spread tells the two pictures apart.

Each of those two runs must log the interval line exactly twice (frames 300 and 600) with all eight named counters, none negative.
present-onset   two-tone for about 6 s, then flat gray for 6 s: exactly one `x2w-gray onset` line, flat from a time between
                   6.0 s and the run time, mean 128.00, spread 0.00, and at least four picture lines in 12 s of a constant
                   picture (the time rule; the count rules log three). present-flat logs one onset line, present-split none.
The slow run also checks the totals, indirect and upload-ring lines at the first ten-second deadline.
Exit 77 when the adapter is not available.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import startup_stream_check as S

PRESENTED = re.compile(r'x2w-render presented surface 64x64 format \d+, readback (\d+), mean level ([0-9.]+)/255, spread ([0-9.]+)')
INTERVAL = re.compile(r'x2w-gray interval \(frame (\d+)\): ring refusals \+(-?\d+), null bindings \+(-?\d+) \(ring_full \+(-?\d+)\), '
                      r'draws not drawn \+(-?\d+), translated draws \+(-?\d+), compute translated \+(-?\d+), compute refused \+(-?\d+), '
                      r'textures unreadable \+(-?\d+), list copies refused \+(-?\d+)')


ONSET = re.compile(r'x2w-gray onset (\d+) \(frame (\d+)\): flat picture from ([0-9.]+) s \(mean ([0-9.]+), spread ([0-9.]+)\); '
                   r'first upload ring refusal at (NONE yet, )?([0-9.]+) s; first draw not drawn at (NONE yet, )?([0-9.]+) s')


def present(consumer, directory, mode):
    cache = directory / 'cache'
    cache.mkdir(exist_ok=True)
    for name, data in (('vertex', S.container(1)), ('pixel', S.container(0))):
        (directory / name).write_bytes(data)
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_RENDER='1', X2W_D3D12_LIST_NOOP='1',
               X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache))
    started = time.monotonic()
    result = subprocess.run([consumer, directory / 'vertex', directory / 'pixel', directory / 'root', mode], env=env,
                            capture_output=True, text=True)
    present.elapsed = time.monotonic() - started
    if result.returncode == 77:
        print(result.stdout, end='')
        sys.exit(77)
    assert result.returncode == 0, f'{mode}: consumer exit {result.returncode}\n{result.stderr[-3000:]}'
    if mode == 'present-onset':
        print(result.stdout, end='')
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
    onsets = [m for m in map(ONSET.search, log.splitlines()) if m]
    if spread < 4 and mean > 100:  # a flat picture from the first readback: one onset line, at the start of the run
        assert len(onsets) == 1 and (onsets[0].group(4), onsets[0].group(5)) == ('%.2f' % mean, '%.2f' % spread), \
            f'{mode}: expected one onset line with mean {mean:.2f} spread {spread:.2f}, got {[m.group(0) for m in onsets]}'
    else:  # a picture with detail is never an onset (negative control)
        assert not onsets, f'{mode}: onset line for a picture with detail: {onsets[0].group(0)}'
    print(f'{mode}: {len(pictures)} presented lines at mean {mean:.2f} spread {spread:.2f}, interval lines at {frames}, '
          f'{len(onsets)} onset lines')


TOTALS_PREFIXES = ("x2w-render totals:", "x2w-render indirect:", "x2w-render upload ring:")
LOCK_PREFIX = "x2w-render draw_lock since the last report"


def check_slow_totals(log):
    seen = set()
    reports = []
    last_ms = 0
    for line in log.splitlines():
        seen.update(prefix for prefix in (*TOTALS_PREFIXES, LOCK_PREFIX) if line.startswith(prefix))
        end = re.fullmatch(r"fixture present (\d+) ended at (\d+) ms", line)
        if end:
            frame, elapsed_ms = map(int, end.groups())
            if seen.intersection(TOTALS_PREFIXES):
                reports.append((frame, last_ms, elapsed_ms, seen))
            seen = set()
            last_ms = elapsed_ms
    assert len(reports) == 1, f"expected one slow totals report, found {reports}"
    frame, previous_ms, elapsed_ms, lines = reports[0]
    assert lines == {*TOTALS_PREFIXES, LOCK_PREFIX}, reports
    # The deadline is sampled inside Present. Allow one 200 ms frame and scheduling/readback slack.
    assert frame < 300 and previous_ms < 10000 and 9900 <= elapsed_ms <= 10750, reports


def check_onset(consumer):
    """Two-tone for about 6 s, then flat gray: one onset line at about the second the picture turned flat, and a picture
    line at least every 5 s (the run shows a constant picture for 12 s, which the count/40th rules alone would not log
    more than three times)."""
    with tempfile.TemporaryDirectory() as temporary:
        log = present(consumer, Path(temporary), 'present-onset')
    check_slow_totals(log)
    # Reproduce the old frame-only policy: this run never reaches its first report.
    frame_only = "\n".join(line for line in log.splitlines()
                           if not line.startswith(TOTALS_PREFIXES))
    try:
        check_slow_totals(frame_only)
    except AssertionError as exc:
        assert str(exc) == "expected one slow totals report, found []", exc
    else:
        raise AssertionError("frame-only control passed")
    elapsed = present.elapsed
    onsets = [m for m in map(ONSET.search, log.splitlines()) if m]
    assert len(onsets) == 1, f'present-onset: {len(onsets)} onset lines\n{log[-3000:]}'
    m = onsets[0]
    at = float(m.group(3))
    assert (m.group(4), m.group(5)) == ('128.00', '0.00'), m.group(0)
    assert 6.0 <= at <= elapsed, f'present-onset: flat from {at} s, expected between 6.0 and the run time {elapsed:.1f} s: {m.group(0)}'
    pictures = [m for m in map(PRESENTED.search, log.splitlines()) if m]
    assert len(pictures) >= 4, f'present-onset: {len(pictures)} picture lines, a timed line was expected\n{log[-3000:]}'
    expected = {(127.5, 127.5), (128.0, 0.0)}
    assert all((float(p.group(2)), float(p.group(3))) in expected for p in pictures), [p.group(0) for p in pictures]
    print(f'present-onset: one onset line, flat from {at} s of a {elapsed:.1f} s run; {len(pictures)} picture lines')


def main(consumer):
    check(consumer, 'present-flat', 128.0, 0.0)
    check(consumer, 'present-split', 127.5, 127.5)
    check_onset(consumer)
    print('PASS')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
