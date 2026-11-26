#!/usr/bin/env python3
"""Check the hard stop of gfx2hlsl.py --retry-lifted's time cap: a request still running when the cap and its grace are over
must not hold the game start. Three old refusals, two workers: the first two requests hang (a huge shader), so past the cap
(1 s) and the grace (2 s) their workers are ended, the report counts them as left, their .fail files stay, and the call
returns long before the hanging requests would have ended (60 s)."""
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402


def hang(out_dir, key):  # module level: the pool's workers import it by name
    time.sleep(60)
    return 'ok'


def main():
    built = G.translator_stamp()
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for key in (1, 2, 3):
            for name, data in (('req_%016x.rtv' % key, b'request'), ('%016x.fail' % key, b'unsupported: image dimension SQ_RSRC_IMG_CUBE\n')):
                (d / name).write_bytes(data)
                os.utime(d / name, (built - 100, built - 100))
        G.translate_one = hang
        G.PRE_RUN_RETRY_GRACE_SECONDS = 2.0
        started = time.monotonic()
        report = G.retry_lifted(str(d), seconds=1.0, jobs=2)
        took = time.monotonic() - started
        if took > 20:
            failures.append('the retry waited %.1f s for hanging requests (cap 1 s + grace 2 s)' % took)
        if (report.keys, report.ok) != (3, 0) or report.left != 3:
            failures.append('report: keys %d ok %d left %d, want 3 0 3' % (report.keys, report.ok, report.left))
        if not all((d / ('%016x.fail' % key)).is_file() for key in (1, 2, 3)):
            failures.append('an ended request lost its .fail')
    if failures:
        print('\n'.join('FAIL ' + f for f in failures))
        return 1
    print('startup_gfx2hlsl_retry_cap_check passed: hanging requests are ended after the cap and the grace (%.1f s)' % took)
    return 0


if __name__ == '__main__':
    sys.exit(main())
