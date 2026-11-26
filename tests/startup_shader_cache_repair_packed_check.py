"""tools/startup/shader_cache_repair_packed.py withdraws exactly the cached results the packed-position derivation fixes.

usage: startup_shader_cache_repair_packed_check.py

One cache holds five graphics requests, all translated before the repair:
  stale   a bare request whose pixel stage splits v4 into its 16-bit halves, translated WITHOUT the derivation (the old
          translator: v4 never filled), and its RTV variant translated the same way
  fresh   a bare request that splits v3, translated with the derivation (its result has the v3 line)
  masked  the stale request's stage with a stream description (masks win; the result is right and has no derivation)
  plain   a bare request with no half-split evidence (nothing to derive)
Only the stale request and its variant are withdrawn (manifest first); the other three results stay byte for byte, the
request files stay, and `--pending` then translates exactly the two again with the line. A second run withdraws nothing.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402
import startup_packed_position_provenance_check as P  # noqa: E402

TOOL = HERE.parent / 'tools' / 'startup' / 'shader_cache_repair_packed.py'
failures = []


def check(label, ok):
    print(('ok   ' if ok else 'FAIL ') + label)
    if not ok:
        failures.append(label)


def snapshot(cache, key):
    return {p.name: p.read_bytes() for p in sorted(cache.iterdir()) if p.name.startswith('%016x.' % key)}


def translate_old(cache, key):
    """translate_request with the derivation switched off: what a cache translated before this change holds."""
    saved = G.Stage.derive_packed_position
    G.Stage.derive_packed_position = lambda self: None
    try:
        text, result = P.translated_ps(cache, key)
    finally:
        G.Stage.derive_packed_position = saved
    assert text is not None, result
    return text


def run_tool(cache, *flags):
    run = subprocess.run([sys.executable, '-B', str(TOOL), str(cache), *flags], capture_output=True, text=True)
    return run.returncode, run.stdout.strip()


def main():
    os.environ['X2W_FXC'] = ''
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        stale = P.put_request(cache, P.split_words(4))
        variant_text = 'rtv1 %d 28' % stale
        variant = G.fnv(variant_text.encode('ascii'))
        (cache / ('req_%016x.rtv' % variant)).write_text(variant_text, encoding='ascii')
        fresh = P.put_request(cache, P.split_words(3))
        masked = P.put_request(cache, P.split_words(4), P.TARGET)
        plain = P.put_request(cache, P.ps_words(4))
        keys = [stale, variant, fresh, masked, plain]
        check('five distinct requests', len(set(keys)) == 5)
        old = translate_old(cache, stale), translate_old(cache, variant)
        check('the old translation of the stale request has no packed line', not P.packed_line(old[0]) and not P.packed_line(old[1]))
        for key in (fresh, masked, plain):
            P.translated_ps(cache, key)
        check('fresh result carries its derived line (v3)', P.packed_line((cache / ('%016x.ps.hlsl' % fresh)).read_text()) == [3])
        check('masked result has the line of its masks (v4)', P.packed_line((cache / ('%016x.ps.hlsl' % masked)).read_text()) == [4])
        check('plain result has none', not P.packed_line((cache / ('%016x.ps.hlsl' % plain)).read_text()))
        check('nothing is pending before the repair', G.pending_keys(str(cache), False) == [])
        before = {key: snapshot(cache, key) for key in keys}
        requests = {p.name: p.read_bytes() for p in cache.iterdir() if p.name.startswith('req_')}

        code, out = run_tool(cache, '--dry-run')
        check('dry run counts 5 requests, 3 derivable, 2 stale and removes nothing',
              code == 0 and '5 graphics requests, 3 without a description' in out and '2 stale results found' in out
              and all(snapshot(cache, key) == before[key] for key in keys))
        code, out = run_tool(cache)
        check('repair withdraws 2 results', code == 0 and '2 stale results withdrawn' in out)
        for label, key in (('the stale request', stale), ('its RTV variant', variant)):
            left = snapshot(cache, key)
            check(label + ' loses its manifest and generated files', not left)
        for label, key in (('fresh', fresh), ('masked', masked), ('plain', plain)):
            check(label + ' result is untouched byte for byte', snapshot(cache, key) == before[key])
        check('request files are untouched', {p.name: p.read_bytes() for p in cache.iterdir() if p.name.startswith('req_')} == requests)
        check('exactly the stale request and its variant are pending again',
              sorted(G.pending_keys(str(cache), False)) == sorted([stale, variant]))
        G.translate_pending(str(cache), skip_failed=True, jobs=1)
        text = (cache / ('%016x.ps.hlsl' % stale)).read_text()
        check('re-translated: the stale request now fills v4 from SV_Position', P.packed_line(text) == [4])
        check('re-translated: its variant too', P.packed_line((cache / ('%016x.ps.hlsl' % variant)).read_text()) == [4])
        again = {key: snapshot(cache, key) for key in keys}
        code, out = run_tool(cache)
        check('a second run withdraws nothing', code == 0 and '0 stale results withdrawn' in out
              and all(snapshot(cache, key) == again[key] for key in keys))
        check('usage error exits 2', run_tool(cache / 'missing')[0] == 2)
    if failures:
        print('%d failure(s)' % len(failures))
        return 1
    print('ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
