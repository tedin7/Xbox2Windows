"""tools/startup/shader_cache_retry_keys.py translates chosen keys again, whatever an older build wrote for them.

usage: startup_shader_cache_retry_keys_check.py

One cache holds four graphics requests:
  stale     translatable, with a <key>.fail an older build wrote (newer than the request: a watch pass leaves it alone)
  named     translatable, also with a stale .fail, but not named to the tool
  still     a request this build still refuses (the stale .fail is replaced by a fresh one)
  done      already translated (its result must stay byte for byte)
plus a key with no request. `stale` and `still` are named (one as a KEY argument, one through a report's "not drawn" lines);
only `stale` ends up translated, `named` keeps its .fail, `done` is untouched, and a second run translates nothing new.
"""
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402
import startup_packed_position_provenance_check as P  # noqa: E402

TOOL = HERE.parent / 'tools' / 'startup' / 'shader_cache_retry_keys.py'
failures = []
OLD = 'unsupported: MRTZ requires one uncompressed depth export in a straight-line pixel shader with entry EXEC\n'


def check(label, ok):
    print(('ok   ' if ok else 'FAIL ') + label)
    if not ok:
        failures.append(label)


def run_tool(cache, *args):
    run = subprocess.run([sys.executable, '-B', str(TOOL), str(cache), *args], capture_output=True, text=True)
    return run.returncode, run.stdout.strip()


def main():
    os.environ['X2W_FXC'] = ''
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        stale, named, done = (P.put_request(cache, P.ps_words(r)) for r in (3, 4, 5))
        refused_words = struct.pack('<III', 0xffffffff, 0xffffffff, 0xbf810000)
        still = P.put_request(cache, refused_words)
        check('four distinct requests', len({stale, named, done, still}) == 4)
        P.translated_ps(cache, done)
        before = {p.name: p.read_bytes() for p in cache.iterdir() if p.name.startswith('%016x.' % done)}
        check('the finished key has a manifest', '%016x.txt' % done in before)
        past = time.time() - 3600
        for key in (stale, named, still):
            for p in cache.glob('req_%016x.*' % key):
                os.utime(p, (past, past))
        for key in (stale, named, still):
            (cache / ('%016x.fail' % key)).write_text(OLD, encoding='ascii')
        check('a watch pass (skip_failed) leaves the stale failures alone',
              sorted(G.pending_keys(str(cache), True)) == [])
        report = cache / 'REPORT.txt'
        report.write_text('  %016x   9 not drawn (x)\n  %016x   7 not drawn (y)\n  %016x   5 not drawn (z)\n' % (
            stale, still, 0x1234567890abcdef), encoding='ascii')
        code, out = run_tool(cache, '--report', str(report), '--top', '2')
        check('report keys: 2 asked, 2 retried, 1 translated', code == 0 and
              '2 keys asked, 2 retried' in out and '1 translated now' in out)
        check('the stale key is translated and loses its .fail',
              (cache / ('%016x.txt' % stale)).is_file() and not (cache / ('%016x.fail' % stale)).exists())
        check('the still-refused key has a new failure',
              (cache / ('%016x.fail' % still)).is_file() and (cache / ('%016x.fail' % still)).read_text() != OLD)
        check('the key not named keeps its old failure and has no result',
              (cache / ('%016x.fail' % named)).read_text() == OLD and not (cache / ('%016x.txt' % named)).exists())
        check('the finished key is untouched byte for byte',
              {p.name: p.read_bytes() for p in cache.iterdir() if p.name.startswith('%016x.' % done)} == before)
        code, out = run_tool(cache, '%016x' % stale, '%016x' % done, '%016x' % 0x1234567890abcdef)
        check('a second run with KEY arguments retries nothing finished', code == 0 and '3 keys asked, 0 retried' in out)
        code, out = run_tool(cache, '%016x' % named)
        check('a KEY argument retries the key left behind', code == 0 and '1 retried' in out and '1 translated now' in out)
        check('usage errors exit 2', run_tool(cache)[0] == 2 and run_tool(cache / 'missing', '%016x' % stale)[0] == 2
              and run_tool(cache, 'nonsense')[0] == 2)
    if failures:
        print('%d failure(s)' % len(failures))
        return 1
    print('ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
