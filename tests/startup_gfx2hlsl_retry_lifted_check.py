#!/usr/bin/env python3
"""Check gfx2hlsl.py --retry-lifted, the pre-run retry tester_run.ps1 makes before the title starts.

In the run the --watch pool leaves old refusals alone and the pass after the run retries them, so a translator lift
would reach only the next run. Before the title starts tester_run translates again the old refusals (a .fail an older
build wrote) whose reason belongs to a family this build lifts (LIFTED_REASONS), on every core, for at most 180 s:

  - only old refusals of a lifted family are retried: a refusal this build wrote is final, another reason waits;
  - the time cap stops new requests; what is left keeps its .fail for the pass after the run;
  - the log line says how many keys, how many translate now, how many are still refused, and the seconds;
  - the command line form rewrites the .fail of a lifted family and leaves the others untouched.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def expect(label, got, want):
    if got != want:
        failures.append('%s: got %r, want %r' % (label, got, want))


def touch(path, data, mtime):
    path.write_bytes(data)
    os.utime(path, (mtime, mtime))


# One refusal text per family, as an older build wrote it, and reasons outside every family.
LIFTED = {
    'rw-buffer': "unsupported: descriptor (('table', 3), 0) used for a load and a store",
    'cube': 'unsupported: image dimension SQ_RSRC_IMG_CUBE',
    'tg-size': 'unsupported: COMPUTE_PGM_RSRC2 TG_SIZE_EN requires at most 64 threads per group',
    'dmask': 'unsupported: image_store with dmask 0x7: a typed UAV store writes all four channels',
    'dpp-pixel': 'unsupported: v_min_u32_dpp requires a straight-line pixel shader with entry EXEC',
    'dpp-compute': 'unsupported: v_or_b32_dpp requires a straight-line compute program with entry EXEC',
}
OTHER = ('unsupported: v_readfirstlane_b32 requires a proven uniform source',
         'unsupported: s_cbranch_execz skips code with effects outside active lanes')
for family, reason in LIFTED.items():
    expect('family of %r' % reason, G.lifted_family(reason), family)
for reason in OTHER:
    expect('family of %r' % reason, G.lifted_family(reason), None)


def fixture(d, built):
    """Keys 1-3: old refusals of lifted families; 4: a lifted family refused by this build; 5, 6: old, other reasons."""
    reasons = {1: LIFTED['cube'], 2: LIFTED['tg-size'], 3: LIFTED['dmask'], 4: LIFTED['rw-buffer'], 5: OTHER[0],
               6: OTHER[1]}
    for key, reason in reasons.items():
        touch(d / ('req_%016x.rtv' % key), b'not an rtv request', built - 200)
        touch(d / ('%016x.fail' % key), (reason + '\n').encode(), built + 50 if key == 4 else built - 100)
    return reasons


built = G.translator_stamp()
real = G.translate_one
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    fixture(d, built)
    expect('the old refusals of lifted families', G.lifted_failures(str(d), built), [1, 2, 3])
    called = []

    def fake(out_dir, key):
        called.append(key)
        return 'ok' if key == 1 else 'unsupported: still'
    G.translate_one = fake
    try:
        report = G.retry_lifted(str(d), jobs=1)
        expect('retried keys', called, [1, 2, 3])
        expect('counts', (report.keys, report.ok, report.refused, report.left), (3, 1, 2, 0))
        line = report.line()
        expect('log line', line.startswith('pre-run retry: 3 keys, ok 1, still refused 2, ') and line.endswith(' s'), True)
        # The time cap: no new request after it, the rest is left for the pass after the run.
        called.clear()
        now = [0.0]

        def clock():
            now[0] += 100.0
            return now[0]
        capped = G.retry_lifted(str(d), seconds=150.0, jobs=1, clock=clock)
        expect('time cap: requests started', called, [1])
        expect('time cap: counts', (capped.keys, capped.ok, capped.refused, capped.left), (3, 1, 0, 2))
        expect('time cap: log line names what is left', '2 left for the pass after the run (time cap)' in capped.line(),
               True)
        # Nothing to retry: no pool, an empty line still says so.
        empty = G.retry_lifted(str(d), jobs=1, families=())
        expect('no lifted family', (empty.keys, empty.left), (0, 0))
    finally:
        G.translate_one = real

# The command line form, with real requests that refuse (an invalid RTV request) and every core.
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    reasons = fixture(d, built)
    before = {key: (d / ('%016x.fail' % key)).stat().st_mtime for key in reasons}
    started = time.time()
    run = subprocess.run([sys.executable, '-B', str(ROOT / 'tools' / 'startup' / 'gfx2hlsl.py'), '--retry-lifted', str(d),
                          '180'], capture_output=True, text=True, timeout=300)
    expect('cli exit code', run.returncode, 0)
    lines = [line for line in run.stdout.splitlines() if line.strip()]
    expect('cli prints one line', len(lines), 1)
    expect('cli line', bool(lines) and lines[0].startswith('pre-run retry: 3 keys, ok 0, still refused 3, '), True)
    rewritten = {key: (d / ('%016x.fail' % key)).stat().st_mtime != before[key] for key in reasons}
    expect('cli rewrites only the old lifted refusals', rewritten,
           {1: True, 2: True, 3: True, 4: False, 5: False, 6: False})
    expect('the other old reasons are kept for the pass after the run', [
        (d / ('%016x.fail' % key)).read_text().strip() for key in (5, 6)], list(OTHER))
    expect('cli well under the cap', time.time() - started < 120, True)

if failures:
    print('\n'.join('FAIL ' + f for f in failures))
    sys.exit(1)
print('startup_gfx2hlsl_retry_lifted_check passed: old refusals of lifted families retried before the run, others '
      'kept, time cap, log line, command line form')
