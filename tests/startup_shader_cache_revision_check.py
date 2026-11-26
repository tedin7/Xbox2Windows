#!/usr/bin/env python3
"""Check tools/startup/shader_cache_revision.py, the launchers' shader cache revision warning (#2815).

The command line runs on synthetic cache folders and a synthetic translator file:
- same revisions in revision.txt: no output, exit 0;
- a different revision: one line naming both revisions and the fixes, exit 1;
- requests (req_*) but no revision.txt: the same kind of line, exit 1; no requests: silent;
- --fresh moves a cache that is not empty aside and leaves an empty or missing one alone.
The real gfx2hlsl.py must yield all three revisions, or the launchers could not compare anything.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / 'tools' / 'startup' / 'shader_cache_revision.py'
sys.path.insert(0, str(TOOL.parent))
import shader_cache_revision as R  # noqa: E402

# Synthetic translator source: the three literals as gfx2hlsl.py spells them.
TRANSLATOR = (b"def pipeline_key():\n    return fnv(b'x2w-startup-shader-90')\n"
              b"def compute_key():\n    return fnv(b'x2w-startup-shader-90cs')\n"
              b"def mesh57_key():\n    return fnv(b'x2w-startup-mesh-57-9')\n")
SAME = 'graphics x2w-startup-shader-90\ncompute x2w-startup-shader-90cs\nmesh57 x2w-startup-mesh-57-9\n'
OLDER = 'graphics x2w-startup-shader-89\ncompute x2w-startup-shader-89cs\nmesh57 x2w-startup-mesh-57-9\n'
failures = []


def check(name, ok, detail=''):
    print('%s: %s' % (name, 'ok' if ok else 'FAIL %s' % detail))
    if not ok:
        failures.append(name)


def run(*args):
    result = subprocess.run([sys.executable, '-B', str(TOOL), *map(str, args)], capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def cache_with(root, name, revision=None, request=True):
    cache = root / name
    cache.mkdir()
    if revision is not None:
        (cache / 'revision.txt').write_text(revision, encoding='ascii')
    if request:
        (cache / 'req_0123456789abcdef.a').write_bytes(b'\0' * 8)
    return cache


def main():
    real = R.translator_revisions(TOOL.parent / 'gfx2hlsl.py')
    check('the real gfx2hlsl.py yields graphics, compute and mesh57 revisions', sorted(real) == sorted(R.KINDS), real)
    check('the compute revision is the graphics one with cs',
          real.get('compute') == real.get('graphics', '') + 'cs', real)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        translator = root / 'gfx2hlsl.py'
        translator.write_bytes(TRANSLATOR)

        code, out, err = run(cache_with(root, 'same', SAME), '--translator', translator)
        check('same revisions: silent', (code, out, err) == (0, '', ''), (code, out, err))

        stale = cache_with(root, 'older', OLDER)
        code, out, err = run(stale, '--translator', translator)
        lines = out.splitlines()
        check('different revision: exit 1, one line', code == 1 and len(lines) == 1 and not err, (code, out, err))
        check('different revision: both revisions named',
              'x2w-startup-shader-89' in out and 'x2w-startup-shader-90' in out and str(stale) in out, out)
        check('different revision: both fixes named', 'rebuild the stand-ins' in out and 'empty shader cache' in out, out)

        code, out, err = run(cache_with(root, 'compute-only', 'compute x2w-startup-shader-89cs\n'),
                             '--translator', translator)
        check('one differing kind is enough', code == 1 and 'x2w-startup-shader-89cs' in out, (code, out))

        code, out, err = run(cache_with(root, 'garbled', 'not a revision\n'), '--translator', translator)
        check('a revision.txt with no known line warns', code == 1 and 'none readable' in out, (code, out))

        legacy = cache_with(root, 'legacy')
        code, out, err = run(legacy, '--translator', translator)
        check('requests without revision.txt warn',
              code == 1 and 'revision.txt' in out and 'x2w-startup-shader-90' in out and 'rebuild' in out, (code, out))

        code, out, err = run(cache_with(root, 'fresh', request=False), '--translator', translator)
        check('no requests, no revision.txt: silent', (code, out) == (0, ''), (code, out))
        code, out, err = run(root / 'missing', '--translator', translator)
        check('missing cache folder: silent', (code, out) == (0, ''), (code, out))

        code, out, err = run(legacy, '--fresh')
        moved = [p for p in root.iterdir() if p.name.startswith('legacy.old-')]
        check('--fresh moves a cache with files aside',
              code == 0 and not legacy.exists() and len(moved) == 1 and (moved[0] / 'req_0123456789abcdef.a').exists()
              and str(moved[0]) in out, (code, out, moved))
        empty = root / 'empty'
        empty.mkdir()
        code, out, err = run(empty, '--fresh')
        check('--fresh leaves an empty cache in place', (code, out) == (0, '') and empty.is_dir(), (code, out))
        code, out, err = run(root / 'missing', '--fresh')
        check('--fresh with no cache folder does nothing', (code, out) == (0, ''), (code, out))

        code, out, err = run()
        check('no cache argument: usage, exit 2', code == 2 and 'usage' in out, (code, out))
        code, out, err = run(stale, '--fresh', '--translator', translator)
        check('--fresh with --translator: usage, exit 2', code == 2, (code, out))
    print('FAIL: %s' % ', '.join(failures) if failures else 'PASS')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
