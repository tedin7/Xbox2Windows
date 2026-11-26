#!/usr/bin/env python3
"""Check gfx2hlsl.py --retry-lifted, the pre-run retry tester_run.ps1 makes before the title starts.

In the run the --watch pool leaves old refusals alone and the pass after the run retries them, so a translator lift
would reach only the next run. Before the title starts tester_run translates again the old refusals (a .fail an older
build wrote), on every core, for at most 180 s, in this order: the families recent builds lift (LIFTED_REASONS),
then graphics and mesh requests before compute:

  - every old refusal is retried, lifted families first; a refusal newer than this build is final unless its reason is
    in a lifted family (a stale .fail of a lifted family, as the MRTZ request Run P kept refused);
  - the time cap stops new requests; what is left keeps its .fail for the pass after the run;
  - the log line says how many keys, how many translate now, how many are still refused, and the seconds;
  - the command line form rewrites every .fail it retried;
  - with real requests (a translatable pixel shader with a stale MRTZ .fail, one this build still refuses, one with a
    newer refusal of another family, one finished): the stale one is translated and loses its .fail, the refused one
    gets a fresh reason, the other keeps its .fail, the finished one is untouched, and a second run retries nothing.
"""
import os
import struct
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
    'mrtz': 'unsupported: MRTZ requires one uncompressed depth export in a straight-line pixel shader with entry EXEC',
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
    """Keys 1-3: old refusals of lifted families; 4: a lifted family, .fail newer than this build; 5: an old compute refusal
    and 6: an old graphics refusal, other reasons."""
    reasons = {1: LIFTED['cube'], 2: LIFTED['tg-size'], 3: LIFTED['dmask'], 4: LIFTED['rw-buffer'], 5: OTHER[0],
               6: OTHER[1]}
    for key, reason in reasons.items():
        touch(d / ('req_%016x.%s' % (key, 'c67' if key == 5 else 'rtv')), b'not a request', built - 200)
        touch(d / ('%016x.fail' % key), (reason + '\n').encode(), built + 50 if key == 4 else built - 100)
    return reasons


built = G.translator_stamp()
real = G.translate_one
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    fixture(d, built)
    expect('the refusals of lifted families, newer or not', G.lifted_failures(str(d), built), [1, 2, 3, 4])
    expect('retry order: lifted, then graphics, then compute', G.old_failures(str(d), built),
           [(1, True), (2, True), (3, True), (4, True), (6, False), (5, False)])
    called = []

    def fake(out_dir, key):
        called.append(key)
        return 'ok' if key == 1 else 'unsupported: still'
    G.translate_one = fake
    try:
        report = G.retry_lifted(str(d), jobs=1)
        expect('retried keys', called, [1, 2, 3, 4, 6, 5])
        expect('a translated key loses its stale .fail', (d / ('%016x.fail' % 1)).exists(), False)
        expect('counts', (report.keys, report.lifted, report.ok, report.refused, report.left), (6, 4, 1, 5, 0))
        line = report.line()
        expect('log line', line.startswith('pre-run retry: 6 keys (4 in lifted families), ok 1, still refused 5, ')
               and line.endswith(' s'), True)
        # The time cap: no new request after it, the rest is left for the pass after the run.
        called.clear()
        now = [0.0]

        def clock():
            now[0] += 100.0
            return now[0]
        fixture(d, built)  # the fake 'ok' removed key 1's .fail: back to the starting cache
        capped = G.retry_lifted(str(d), seconds=150.0, jobs=1, clock=clock)
        expect('time cap: requests started', called, [1])
        expect('time cap: counts', (capped.keys, capped.ok, capped.refused, capped.left), (6, 1, 0, 5))
        expect('time cap: log line names what is left', '5 left for the pass after the run (time cap)' in capped.line(),
               True)
        # No family table: every old refusal still goes, graphics before compute; the newer one is final.
        called.clear()
        fixture(d, built)
        plain = G.retry_lifted(str(d), jobs=1, families=())
        expect('no lifted family', (plain.keys, plain.lifted, called), (5, 0, [1, 2, 3, 6, 5]))
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
    expect('cli line', bool(lines) and lines[0].startswith('pre-run retry: 6 keys (4 in lifted families), ok 0, '
                                                           'still refused 6, '), True)
    rewritten = {key: (d / ('%016x.fail' % key)).stat().st_mtime != before[key] for key in reasons}
    expect('cli rewrites every refusal it retried', rewritten, {1: True, 2: True, 3: True, 4: True, 5: True, 6: True})
    expect('cli well under the cap', time.time() - started < 120, True)

# --jobs N (tester_run: 4 for the pre-run retry and the pass after the run) caps the workers, never above the cores.
expect('--jobs 4', G.jobs_argument(['180', '--jobs', '4']), min(4, G.translation_jobs()))
expect('--jobs 0 is one worker', G.jobs_argument(['--jobs', '0']), 1)
expect('no --jobs: the caller decides', G.jobs_argument(['180']), None)
try:
    G.jobs_argument(['--jobs'])
    expect('--jobs without a number is refused', False, True)
except SystemExit:
    pass

# Real requests (the cases of the stale MRTZ .fail): translatable pixel shaders from the provenance check's helpers.
sys.path.insert(0, str(ROOT / 'tests'))
import startup_packed_position_provenance_check as P  # noqa: E402

OLD_MRTZ = LIFTED['mrtz'] + '\n'
saved_fxc = os.environ.get('X2W_FXC')
os.environ['X2W_FXC'] = ''  # manifest only: no compiler needed
try:
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        stale, other, done = (P.put_request(cache, P.ps_words(r)) for r in (3, 4, 5))
        still = P.put_request(cache, struct.pack('<III', 0xffffffff, 0xffffffff, 0xbf810000))
        expect('four distinct requests', len({stale, other, done, still}), 4)
        P.translated_ps(cache, done)
        finished = {p.name: p.read_bytes() for p in cache.iterdir() if p.name.startswith('%016x.' % done)}
        expect('the finished key has a manifest', '%016x.txt' % done in finished, True)
        past = time.time() - 3600
        for key in (stale, other, still):
            for p in cache.glob('req_%016x.*' % key):
                os.utime(p, (past, past))
        # Written now: newer than the requests and than this build, as the .fail Run P kept.
        for key in (stale, still):
            (cache / ('%016x.fail' % key)).write_text(OLD_MRTZ, encoding='ascii')
        (cache / ('%016x.fail' % other)).write_text(OTHER[1] + '\n', encoding='ascii')
        expect('a watch pass (skip_failed) leaves the stale failures alone', sorted(G.pending_keys(str(cache), True)), [])
        report = G.retry_lifted(str(cache), jobs=1)
        expect('real requests: 2 retried, 1 translated', (report.keys, report.lifted, report.ok, report.refused),
               (2, 2, 1, 1))
        expect('the stale MRTZ key is translated and loses its .fail',
               ((cache / ('%016x.txt' % stale)).is_file(), (cache / ('%016x.fail' % stale)).exists()), (True, False))
        expect('the still-refused key has a fresh reason', (cache / ('%016x.fail' % still)).read_text() != OLD_MRTZ, True)
        expect('the newer refusal of another family keeps its .fail and has no result',
               ((cache / ('%016x.fail' % other)).read_text(), (cache / ('%016x.txt' % other)).exists()),
               (OTHER[1] + '\n', False))
        expect('the finished key is untouched byte for byte',
               {p.name: p.read_bytes() for p in cache.iterdir() if p.name.startswith('%016x.' % done)}, finished)
        # A key with no request is not a request: ignored.
        (cache / ('%016x.fail' % 0x1234567890abcdef)).write_text(OLD_MRTZ, encoding='ascii')
        again = G.retry_lifted(str(cache), jobs=1)
        expect('a second run retries nothing (the fresh refusal is final)', (again.keys, again.ok), (0, 0))
finally:
    if saved_fxc is None:
        os.environ.pop('X2W_FXC', None)
    else:
        os.environ['X2W_FXC'] = saved_fxc

if failures:
    print('\n'.join('FAIL ' + f for f in failures))
    sys.exit(1)
print('startup_gfx2hlsl_retry_lifted_check passed: every old refusal retried before the run, lifted families '
      'first, then graphics, then compute; time cap, log line, command line form')
