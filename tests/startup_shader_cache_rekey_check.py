"""tools/startup/shader_cache_rekey.py on synthetic shader requests, pure Python (no Windows, D3D12, Wine or DXC).

The requests are fabricated byte strings, not shaders: the tool only hashes them. "Older" and "newer" cache
revisions come from a copy of gfx2hlsl.py's source with the revision number in its two key strings changed by
one, so the check follows the next revision bump without an edit.

- An old cache holds requests of the older revision (two graphics requests, one with an input layout, one
  compute request, an RTV variant of the first graphics request), one current-revision graphics request, the same
  pipeline as the first at the current revision, an old and a current slot 57 mesh request, a slot 86 marker,
  a result and a .fail. Old request names differ from the current pipeline_key, compute_key or mesh57_key.
- The tool, run as a script with two hashing processes, writes exactly the current names with the same bytes
  (the RTV variant re-keyed to its new base), no results or .fail, and gfx2hlsl.pending_keys lists them.
- With the newer graphics and compute revision those requests and the RTV variant get other names again.
- A second run writes nothing; re-keying a current cache keeps every name.
- A missing request part, a truncated RTV variant or an RTV variant whose base request is missing exits with 1
  and creates nothing.
- The old cache is unchanged: names, bytes and modification times.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STARTUP = ROOT / 'tools' / 'startup'
TOOL = STARTUP / 'shader_cache_rekey.py'
sys.path.insert(0, str(STARTUP))
import gfx2hlsl as G  # noqa: E402
import shader_cache_rekey as R  # noqa: E402

failures = []


def check(name, ok, detail=''):
    print('%s: %s%s' % ('ok' if ok else 'FAIL', name, (' (%s)' % detail) if detail and not ok else ''))
    if not ok:
        failures.append(name)


def revision_module(delta):
    """gfx2hlsl with the cache revision number changed by delta, and that number."""
    text = (STARTUP / 'gfx2hlsl.py').read_text(encoding='utf-8')
    literal = re.compile(r"b'x2w-startup-shader-([0-9]+)(cs)?'")
    found = literal.findall(text)
    assert sorted(suffix for _, suffix in found) == ['', 'cs'] and len({n for n, _ in found}) == 1, (
        "gfx2hlsl.py no longer spells the cache revision as one b'x2w-startup-shader-N' and one "
        "b'x2w-startup-shader-Ncs' literal; update revision_module (found %r)" % (found,))
    number = int(found[0][0]) + delta
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader('gfx2hlsl_%d' % number, loader=None))
    module.__file__ = str(STARTUP / 'gfx2hlsl.py')
    exec(compile(literal.sub(lambda m: "b'x2w-startup-shader-%d%s'" % (number, m.group(2) or ''), text),
                 module.__file__, 'exec'), module.__dict__)
    return module, number


def graphics(index, layout=b''):
    vs = struct.pack('<4I', 0x7e000280 + index, 0x01020304, index, 0xbf810000)
    ps = struct.pack('<3I', 0x7e020281, index * 7, 0xbf810000)
    parts = {'.b': ps, '.root': b'RTS0 synthetic root %d' % index, '.a': vs}
    if layout:
        parts['.ia'] = layout
    return parts


def gkey(module, parts):
    return module.pipeline_key(parts['.a'], parts['.b'], parts['.root'], parts.get('.ia', b''))


COMPUTE = {'.desc': struct.pack('<12Q', *range(12)), '.sa': struct.pack('<7I', *range(7)), '.sb': b'\x05' * 20,
           '.data': struct.pack('<2Q', 0x1122334455667788, 0x99), '.root': b'RTS0 synthetic compute root',
           '.c67': struct.pack('<2I', 0x7e020280, 0xbf810000)}
MESH_OLD = {'.ps57': b'', '.root': b'RTS0 synthetic mesh root', '.m57': b'synthetic mesh'}
MESH_CURRENT = {'.ps57': b'synthetic pixel stage', '.root': b'RTS0 synthetic mesh root',
                '.m57': b'another synthetic mesh'}


def ckey(module, parts):
    return module.compute_key(parts['.c67'], parts['.data'], parts['.root'], parts['.desc'], parts['.sa'], parts['.sb'])


def mkey(module, parts):
    return module.mesh57_key(parts['.m57'], parts['.ps57'], parts['.root'])


def rtv_text(base):
    return 'rtv1 %d 28 87' % base


def write_request(directory, key, parts):
    for ext, data in parts.items():
        (directory / ('req_%016x%s' % (key, ext))).write_bytes(data)


def write_rtv(directory, base):
    text = rtv_text(base).encode('ascii')
    key = G.fnv(text)
    (directory / ('req_%016x.rtv' % key)).write_bytes(text)
    return key


def snapshot(directory):
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(directory.iterdir())}


def run(src, dst, jobs='2'):
    env = dict(os.environ, X2W_TRANSLATE_JOBS=jobs)
    return subprocess.run([sys.executable, '-B', str(TOOL), str(src), str(dst)], capture_output=True, text=True,
                          env=env, timeout=120)


def expected_files(module, requests, computes, rtv_bases, meshes):
    """{name: bytes} of the requests named under module's revision."""
    files = {}
    for parts in requests:
        for ext, data in parts.items():
            files['req_%016x%s' % (gkey(module, parts), ext)] = data
    for parts in computes:
        for ext, data in parts.items():
            files['req_%016x%s' % (ckey(module, parts), ext)] = data
    for parts in rtv_bases:
        text = rtv_text(gkey(module, parts)).encode('ascii')
        files['req_%016x.rtv' % G.fnv(text)] = text
    for parts in meshes:
        for ext, data in parts.items():
            files['req_%016x%s' % (mkey(module, parts), ext)] = data
    return files


def contents(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def main():
    older, old_number = revision_module(-1)
    newer, new_number = revision_module(+1)
    g1, g2, g3 = graphics(1), graphics(2, b'synthetic layout POSITION 0 R32G32B32_FLOAT'), graphics(3)
    print('revisions: older %d, newer %d' % (old_number, new_number))

    with tempfile.TemporaryDirectory(prefix='x2w-rekey-') as tmp:
        tmp = Path(tmp)
        src = tmp / 'old_cache'
        src.mkdir()
        for parts in (g1, g2):
            write_request(src, gkey(older, parts), parts)
        write_request(src, ckey(older, COMPUTE), COMPUTE)
        old_rtv = write_rtv(src, gkey(older, g1))
        write_request(src, gkey(G, g1), g1)  # the same pipeline at the current revision
        write_request(src, gkey(G, g3), g3)  # current only
        meshes = (MESH_OLD, MESH_CURRENT)
        old_mesh = mkey(G, MESH_OLD) ^ 1
        mesh_keys = {mkey(G, parts) for parts in meshes}
        write_request(src, old_mesh, MESH_OLD)
        write_request(src, mkey(G, MESH_CURRENT), MESH_CURRENT)
        write_request(src, 0x123456789abcdef0, {'.m86': b'synthetic slot 86 marker'})
        (src / ('%016x.txt' % gkey(older, g1))).write_text('synthetic result\n')
        (src / ('%016x.fail' % gkey(older, g2))).write_text('synthetic refusal\n')
        before = snapshot(src)

        old_names = {gkey(older, g1), gkey(older, g2), ckey(older, COMPUTE)}
        current_names = {gkey(G, g1), gkey(G, g2), ckey(G, COMPUTE)}
        check('older-revision names differ from the current keys', not old_names & current_names)
        check('seed preflight counts graphics, compute and mesh but skips RTV and slot 86',
              G.key_mismatch_count(src) == (4, 7))

        # (a) re-keyed names use the current key functions, with the same bytes.
        dst = tmp / 'rekeyed'
        first = run(src, dst)
        check('re-key exits 0', first.returncode == 0, first.stderr.strip())
        want = expected_files(G, (g1, g2, g3), (COMPUTE,), (g1,), meshes)
        got = contents(dst) if dst.is_dir() else {}
        check('re-key stamps the translator seed on a single line',
              got.pop(G.KEY_SEED_NAME, b'').splitlines() == [G.key_seed_line().encode('ascii')])
        check('re-keyed files are exactly the current requests', got == want,
              'extra %s missing %s' % (sorted(set(got) - set(want)), sorted(set(want) - set(got))))
        check('no results or .fail written', all(name.startswith('req_') for name in got))
        check('graphics summary counts the duplicate and the current request',
              'graphics: 4 requests, 2 already current -> 3 keys' in first.stdout, first.stdout)
        check('mesh summary counts the old and current slot 57 requests',
              'm57: 2 requests, 1 already current -> 2 keys' in first.stdout, first.stdout)
        check('slot 86 marker is only counted', 'not re-keyed (own key strings): m86 1' in first.stdout,
              first.stdout)
        pending = G.pending_keys(str(dst), False)
        rtv_new = G.fnv(rtv_text(gkey(G, g1)).encode('ascii'))
        check('gfx2hlsl.pending_keys lists the re-keyed requests',
              pending == sorted({gkey(G, g1), gkey(G, g2), gkey(G, g3), ckey(G, COMPUTE), rtv_new} | mesh_keys),
              pending)

        # (e) the RTV variant names the new base and is named by the hash of its new text.
        rtv_file = dst / ('req_%016x.rtv' % rtv_new)
        check('RTV variant re-keyed to its new base', rtv_file.is_file() and rtv_new != old_rtv
              and rtv_file.read_bytes() == rtv_text(gkey(G, g1)).encode('ascii'))

        # (c) a second run writes nothing and changes nothing.
        after_first = snapshot(dst)
        second = run(src, dst)
        check('second run exits 0 and writes nothing', second.returncode == 0
              and 'files: 0 written, %d already present' % len(want) in second.stdout, second.stdout + second.stderr)
        check('second run leaves the re-keyed files unchanged', snapshot(dst) == after_first)

        wrong_seed = tmp / 'wrong_seed'
        wrong_seed.mkdir()
        (wrong_seed / G.KEY_SEED_NAME).write_text(older.key_seed_line() + '\n', encoding='ascii')
        stamped_before = snapshot(wrong_seed)
        rejected = run(src, wrong_seed)
        check('destination with another seed is refused before copying', rejected.returncode == 1
              and 'destination key seed mismatch' in rejected.stderr and older.key_seed_line() in rejected.stderr
              and G.key_seed_line() in rejected.stderr and snapshot(wrong_seed) == stamped_before, rejected.stderr)

        # (d) re-keying a current cache keeps every name and byte.
        again = tmp / 'again'
        third = run(dst, again, jobs='1')
        check('current requests keep their names', third.returncode == 0 and contents(again) == contents(dst)
              and 'graphics: 3 requests, 3 already current -> 3 keys' in third.stdout
              and 'compute: 1 requests, 1 already current -> 1 keys' in third.stdout
              and 'm57: 2 requests, 2 already current -> 2 keys' in third.stdout, third.stdout + third.stderr)

        # (b) under a newer revision the same requests get other names (in process: one job keeps the patch).
        R.g, saved = newer, R.g
        newest = tmp / 'newer'
        try:
            with open(os.devnull, 'w') as sink:
                code = R.rekey(str(src), str(newest), jobs=1, out=sink)
        finally:
            R.g = saved
        got_newer = contents(newest) if newest.is_dir() else {}
        check('re-key uses the new translator revision in the stamp',
              got_newer.pop(G.KEY_SEED_NAME, b'').splitlines() == [newer.key_seed_line().encode('ascii')])
        check('newer revision re-keys to its own names', code == 0
              and got_newer == expected_files(newer, (g1, g2, g3), (COMPUTE,), (g1,), meshes))
        check('only mesh names are shared between the graphics and compute revisions',
              set(got_newer) & set(want) == set(expected_files(G, (), (), (), meshes)))

        # (f) negative controls: an incomplete request, a truncated RTV variant or an RTV variant without its base
        # request writes nothing. Without the base's ready marker (.a), the request count drops from 8 to 7.
        for label, total, reason, damage in (
                ('missing .root', 8, 'unreadable req_%016x.root' % gkey(older, g2),
                 lambda d: (d / ('req_%016x.root' % gkey(older, g2))).unlink()),
                ('missing slot 57 .root', 8, 'unreadable req_%016x.root' % old_mesh,
                 lambda d: (d / ('req_%016x.root' % old_mesh)).unlink()),
                ('missing slot 57 .ps57', 8, 'unreadable req_%016x.ps57' % old_mesh,
                 lambda d: (d / ('req_%016x.ps57' % old_mesh)).unlink()),
                ('truncated RTV variant', 8, '.rtv text does not hash to its name',
                 lambda d: (d / ('req_%016x.rtv' % old_rtv)).write_bytes(
                     rtv_text(gkey(older, g1)).encode('ascii')[:-3])),
                ('missing RTV base', 7, '.rtv base request %016x is not a complete graphics request' % gkey(older, g1),
                 lambda d: (d / ('req_%016x.a' % gkey(older, g1))).unlink())):
            bad = tmp / ('bad_' + label.replace(' ', '_'))
            shutil.copytree(src, bad)
            damage(bad)
            if label == 'missing .root':
                check('unreadable requests count toward the total only', G.key_mismatch_count(bad) == (3, 7))
            out = tmp / ('out_' + label.replace(' ', '_'))
            result = run(bad, out)
            check('%s exits 1' % label, result.returncode == 1, '%d %s' % (result.returncode, result.stderr))
            check('%s is counted' % label, 'error: 1 of %d requests are incomplete or unreadable; nothing written'
                  % total in result.stderr and reason in result.stderr, result.stderr)
            check('%s writes nothing' % label, not out.exists())

        refused = run(src, src)
        check('the old cache is refused as the destination', refused.returncode == 2, refused.stderr)

        # (g) the old cache is unchanged.
        check('old cache unchanged (names, bytes, mtimes)', snapshot(src) == before)

    print('%d failed' % len(failures) if failures else 'all checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
