"""The gray-world log lines of the console present (render mode): picture mean and spread, and the per-interval counters.

usage: startup_present_gray_check.py <startup_stream_consumer> [--recent-fps]

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


def run_present(consumer, directory, mode, **overrides):
    cache = directory / 'cache'
    cache.mkdir(exist_ok=True)
    for name, data in (('vertex', S.container(1)), ('pixel', S.container(0))):
        (directory / name).write_bytes(data)
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_RENDER='1', X2W_D3D12_LIST_NOOP='1',
               X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache))
    env.update(overrides)
    started = time.monotonic()
    result = subprocess.run([consumer, directory / 'vertex', directory / 'pixel', directory / 'root', mode], env=env,
                            capture_output=True, text=True)
    elapsed = time.monotonic() - started
    if result.returncode == 77:
        print(result.stdout, end='')
        sys.exit(77)
    assert result.returncode == 0, f'{mode}: consumer exit {result.returncode}\n{result.stderr[-3000:]}'
    return result, elapsed


def check(consumer, mode, mean, spread):
    with tempfile.TemporaryDirectory() as temporary:
        result, _ = run_present(consumer, Path(temporary), mode)
    log = result.stderr
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


def check_onset(consumer):
    """Two-tone for about 6 s, then flat gray: one onset line at about the second the picture turned flat, and a picture
    line at least every 5 s (the run shows a constant picture for 12 s, which the count/40th rules alone would not log
    more than three times)."""
    with tempfile.TemporaryDirectory() as temporary:
        result, elapsed = run_present(consumer, Path(temporary), 'present-onset')
    log = result.stderr
    onsets = [m for m in map(ONSET.search, log.splitlines()) if m]
    assert len(onsets) == 1, f'present-onset: {len(onsets)} onset lines\n{log[-3000:]}'
    m = onsets[0]
    at = float(m.group(3))
    assert (m.group(4), m.group(5)) == ('128.00', '0.00'), m.group(0)
    assert 6.0 <= at <= elapsed, f'present-onset: flat from {at} s, expected between 6.0 and the run time {elapsed:.1f} s: {m.group(0)}'
    pictures = [m for m in map(PRESENTED.search, log.splitlines()) if m]
    assert len(pictures) >= 4, f'present-onset: {len(pictures)} picture lines, a timed line was expected\n{log[-3000:]}'
    print(f'present-onset: one onset line, flat from {at} s of a {elapsed:.1f} s run; {len(pictures)} picture lines')


RECENT_CAPTION = re.compile(r'^caption: .* - frame 60, ([0-9]+[.][0-9]) fps [(]last 30[)], '
                            r'([0-9]+[.][0-9]) fps [(]run average[)],', re.MULTILINE)
LEGACY_CAPTION = re.compile(r'^caption: .* - frame 60, [0-9]+[.][0-9] fps, [0-9]+ translated draws', re.MULTILINE)


def recent_rates(stdout):
    match = RECENT_CAPTION.search(stdout)
    if match:
        recent, average = map(float, match.groups())
        if 0 < recent <= 4.1 and recent < average:
            return recent, average
    return None


def check_recent_fps(consumer):
    reference = bytes((37, 109, 223, 255)) * (64 * 64)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for enabled in ('1', '0'):
            readback = directory / f'frame60-{enabled}.rgba'
            result, _ = run_present(consumer, directory, 'present-recent-fps',
                                    X2W_PRESENT_RECENT_FPS=enabled, X2W_TEST_PRESENT_OUT=str(readback))
            assert readback.read_bytes() == reference, f'frame 60 RGBA8 readback mismatch (switch={enabled})'
            rates = recent_rates(result.stdout)
            if enabled == '1':
                assert rates is not None, (f'expected slow rate first and higher run average: {result.stdout}\n'
                                           f'stderr tail: {result.stderr[-1500:]}')
                recent, average = rates
                match = RECENT_CAPTION.search(result.stdout)
                swapped = (result.stdout[:match.start(1)] + match[2] +
                           result.stdout[match.end(1):match.start(2)] + match[1] + result.stdout[match.end(2):])
                assert recent_rates(swapped) is None, f'swapped-rate control passed: {swapped}'
                print(f'frame 60: {recent:.1f} fps (last 30), {average:.1f} fps (run average), pixels match')
            else:
                # The old caption only appears when the picture reached the window; without it the new text must be absent.
                assert "(last 30)" not in result.stdout, f"switch off still shows the recent rate: {result.stdout}"
                legacy = LEGACY_CAPTION.search(result.stdout)
                assert legacy or "frame 60" not in result.stdout, f"unexpected caption with the switch off: {result.stdout}"
                assert rates is None, f'switch-off control unexpectedly passed the recent-caption check: {result.stdout}'
                print('switch-off control: legacy caption rejected by recent-caption check, pixels match')


def main(consumer):
    check(consumer, 'present-flat', 128.0, 0.0)
    check(consumer, 'present-split', 127.5, 127.5)
    check_onset(consumer)
    print('PASS')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[2] == "--recent-fps":
        check_recent_fps(sys.argv[1])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise SystemExit(__doc__)
