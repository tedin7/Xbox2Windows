"""Tests for `compile_dxil --rekey` parity with `tools/startup/shader_cache_rekey.py`."""
import hashlib
import importlib.util
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STARTUP = ROOT / 'tools' / 'startup'
REKEY_PY = STARTUP / 'shader_cache_rekey.py'
sys.path.insert(0, str(STARTUP))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(name, ok, detail=''):
    print('%s: %s%s' % ('ok' if ok else 'FAIL', name, (' (%s)' % detail) if detail and not ok else ''))
    if not ok:
        failures.append(name)


def older_revision_module():
    text = (STARTUP / 'gfx2hlsl.py').read_text(encoding='utf-8')
    literal = re.compile(r"b'x2w-startup-shader-([0-9]+)(cs)?'")
    found = literal.findall(text)
    assert sorted(suffix for _, suffix in found) == ['', 'cs'] and len({n for n, _ in found}) == 1
    number = int(found[0][0]) - 1
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader('gfx2hlsl_%d' % number, loader=None))
    module.__file__ = str(STARTUP / 'gfx2hlsl.py')
    exec(compile(literal.sub(lambda m: "b'x2w-startup-shader-%d%s'" % (number, m.group(2) or ''), text),
                 module.__file__, 'exec'), module.__dict__)
    return module


def graphics(index, layout=b''):
    vs = struct.pack('<4I', 0x7e000280 + index, 0x01020304, index, 0xbf810000)
    ps = struct.pack('<3I', 0x7e020281, index * 7, 0xbf810000)
    parts = {'.b': ps, '.root': b'RTS0 synthetic root %d' % index, '.a': vs}
    if layout:
        parts['.ia'] = layout
    return parts


def gkey(module, parts, ps_inputs=None):
    return module.pipeline_key(parts['.a'], parts['.b'], parts['.root'], parts.get('.ia', b''), ps_inputs)


COMPUTE = {'.desc': struct.pack('<12Q', *range(12)), '.sa': struct.pack('<7I', *range(7)), '.sb': b'\x05' * 20,
           '.data': struct.pack('<2Q', 0x1122334455667788, 0x99), '.root': b'RTS0 synthetic compute root',
           '.c67': struct.pack('<2I', 0x7e020280, 0xbf810000)}
MESH_OLD = {'.ps57': b'', '.root': b'RTS0 synthetic mesh root', '.m57': b'synthetic mesh'}
MESH_CURRENT = {'.ps57': b'synthetic pixel stage', '.root': b'RTS0 synthetic mesh root',
                '.m57': b'another synthetic mesh'}


def write_request(directory, key, parts):
    for ext, data in parts.items():
        (directory / ('req_%016x%s' % (key, ext))).write_bytes(data)


def write_rtv(directory, base):
    text = ('rtv1 %d 28 87' % base).encode('ascii')
    key = G.fnv(text)
    (directory / ('req_%016x.rtv' % key)).write_bytes(text)
    return key


def snapshot(directory):
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(directory.iterdir())}


def contents(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def run_cpp(exe, *args):
    return subprocess.run([str(exe), '--rekey', *[str(a) for a in args]],
                          capture_output=True, text=True, timeout=60)


def check_revision_strings():
    cpp_req = (ROOT / 'src' / 'shader_compute_request.cpp').read_text(encoding='utf-8')
    draw_inl = (ROOT / 'src' / 'startup_stubs' / 'd3d12_draw.inl').read_text(encoding='utf-8')
    py_text = (STARTUP / 'gfx2hlsl.py').read_text(encoding='utf-8')
    for token in ('x2w-startup-shader-29', 'x2w-startup-shader-29cs', 'x2w-startup-mesh-57-3'):
        ok = (token in cpp_req) and (token in draw_inl) and (token in py_text)
        check('revision string %s matches across C++ and Python' % token, ok)


def main():
    if len(sys.argv) != 2:
        print('usage: native_compile_dxil_rekey_check.py <path-to-compile_dxil>', file=sys.stderr)
        return 2
    exe = Path(sys.argv[1])
    check('compile_dxil executable exists', exe.is_file(), str(exe))
    check_revision_strings()

    older = older_revision_module()
    g1 = graphics(1)
    g2 = graphics(2, b'synthetic layout POSITION 0 R32G32B32_FLOAT')
    g3 = graphics(3)
    psin_masks = (0x0003, 0x000f)

    with tempfile.TemporaryDirectory(prefix='x2w-native-rekey-') as tmp:
        tmp = Path(tmp)
        src = tmp / 'old_cache'
        src.mkdir()

        k1_old = gkey(older, g1)
        k2_old = gkey(older, g2)
        k3_old = gkey(older, g3, psin_masks)
        k1_cur = gkey(G, g1)
        k2_cur = gkey(G, g2)
        k3_cur = gkey(G, g3, psin_masks)
        c_old = older.compute_key(COMPUTE['.c67'], COMPUTE['.data'], COMPUTE['.root'],
                                  COMPUTE['.desc'], COMPUTE['.sa'], COMPUTE['.sb'])
        c_cur = G.compute_key(COMPUTE['.c67'], COMPUTE['.data'], COMPUTE['.root'],
                              COMPUTE['.desc'], COMPUTE['.sa'], COMPUTE['.sb'])
        m_old_key = G.mesh57_key(MESH_OLD['.m57'], MESH_OLD['.ps57'], MESH_OLD['.root']) ^ 1
        m_old_cur = G.mesh57_key(MESH_OLD['.m57'], MESH_OLD['.ps57'], MESH_OLD['.root'])
        m_cur_key = G.mesh57_key(MESH_CURRENT['.m57'], MESH_CURRENT['.ps57'], MESH_CURRENT['.root'])

        write_request(src, k1_old, g1)
        write_request(src, k2_old, g2)
        g3_old_parts = dict(g3)
        g3_old_parts['.psin'] = ('psin1 %016x ena=0x%04x addr=0x%04x\n' % (k3_old, *psin_masks)).encode('ascii')
        write_request(src, k3_old, g3_old_parts)
        write_request(src, k1_cur, g1)  # duplicate current copy of g1
        write_request(src, c_old, COMPUTE)
        write_request(src, m_old_key, MESH_OLD)
        write_request(src, m_cur_key, MESH_CURRENT)
        old_rtv = write_rtv(src, k1_old)
        write_request(src, 0x123456789abcdef0, {'.m86': b'synthetic slot 86'})

        # Non-request outputs and diagnostics that must never be copied.
        (src / ('%016x.txt' % k1_old)).write_bytes(b'synthetic manifest\n')
        (src / ('%016x.fail' % k2_old)).write_bytes(b'synthetic refusal\n')
        (src / ('%016x.vs.cso' % k1_old)).write_bytes(b'DXBC...')
        (src / ('%016x.psdep' % k3_old)).write_bytes(b'psin1 ...\n')

        before_src = snapshot(src)

        dst_cpp = tmp / 'dst_cpp'
        dst_py = tmp / 'dst_py'
        res_cpp = run_cpp(exe, src, dst_cpp, '--jobs', '2')
        res_py = subprocess.run([sys.executable, '-B', str(REKEY_PY), str(src), str(dst_py)],
                                capture_output=True, text=True, timeout=60)

        check('compile_dxil --rekey exits 0', res_cpp.returncode == 0, res_cpp.stdout + res_cpp.stderr)
        check('shader_cache_rekey.py exits 0', res_py.returncode == 0, res_py.stdout + res_py.stderr)
        check('byte-for-byte parity with shader_cache_rekey.py',
              dst_cpp.is_dir() and contents(dst_cpp) == contents(dst_py))
        check('only req_* files copied (no .txt/.fail/.cso/.psdep)',
              all(name.startswith('req_') for name in contents(dst_cpp)))

        rtv_cur = G.fnv(('rtv1 %d 28 87' % k1_cur).encode('ascii'))
        check('rewritten .psin names new key with preserved masks',
              (dst_cpp / ('req_%016x.psin' % k3_cur)).read_bytes() ==
              ('psin1 %016x ena=0x%04x addr=0x%04x\n' % (k3_cur, *psin_masks)).encode('ascii'))
        check('rewritten .rtv names new base key',
              rtv_cur != old_rtv and
              (dst_cpp / ('req_%016x.rtv' % rtv_cur)).read_bytes() == ('rtv1 %d 28 87' % k1_cur).encode('ascii'))

        check('summary graphics line',
              'graphics: 4 requests, 1 already current -> 3 keys' in res_cpp.stdout, res_cpp.stdout)
        check('summary compute line',
              'compute: 1 requests, 0 already current -> 1 keys' in res_cpp.stdout, res_cpp.stdout)
        check('summary m57 line',
              'm57: 2 requests, 1 already current -> 2 keys' in res_cpp.stdout, res_cpp.stdout)
        check('summary rtv line',
              'rtv variants: 1 requests -> 1 keys' in res_cpp.stdout, res_cpp.stdout)
        check('summary m86 line',
              'not re-keyed (own key strings): m86 1' in res_cpp.stdout, res_cpp.stdout)

        expected_pending = sorted({k1_cur, k2_cur, k3_cur, c_cur, m_old_cur, m_cur_key, rtv_cur})
        check('gfx2hlsl.pending_keys recognizes re-keyed cache',
              G.pending_keys(str(dst_cpp), False) == expected_pending,
              str(G.pending_keys(str(dst_cpp), False)))

        # --jobs 1 matches --jobs 2
        dst_j1 = tmp / 'dst_j1'
        res_j1 = run_cpp(exe, src, dst_j1, '--jobs', '1')
        check('--jobs 1 produces identical contents to --jobs 2',
              res_j1.returncode == 0 and contents(dst_j1) == contents(dst_cpp), res_j1.stderr)

        # Idempotency (second run writes 0 files and touches nothing)
        snap_dst = snapshot(dst_cpp)
        second = run_cpp(exe, src, dst_cpp)
        total_files = len(snap_dst)
        check('second run is idempotent (0 written)',
              second.returncode == 0 and ('files: 0 written, %d already present' % total_files) in second.stdout,
              second.stdout + second.stderr)
        check('second run leaves mtimes and contents unchanged', snapshot(dst_cpp) == snap_dst)

        # Re-keying an already current cache preserves every name and reports all current
        dst_again = tmp / 'dst_again'
        third = run_cpp(exe, dst_cpp, dst_again)
        check('re-keying current cache preserves names and reports all current',
              third.returncode == 0 and contents(dst_again) == contents(dst_cpp)
              and 'graphics: 3 requests, 3 already current -> 3 keys' in third.stdout
              and 'compute: 1 requests, 1 already current -> 1 keys' in third.stdout
              and 'm57: 2 requests, 2 already current -> 2 keys' in third.stdout,
              third.stdout + third.stderr)

        # Incomplete/malformed negative controls (exit 1, write nothing, dst not created)
        bad_cases = [
            ('graphics missing .root', 8, lambda d: (d / ('req_%016x.root' % k2_old)).unlink()),
            ('graphics psin wrong key', 8,
             lambda d: (d / ('req_%016x.psin' % k3_old)).write_bytes(
                 ('psin1 %016x ena=0x0003 addr=0x000f\n' % (k3_old ^ 1)).encode('ascii'))),
            ('graphics psin contradictory masks', 8,
             lambda d: (d / ('req_%016x.psin' % k3_old)).write_bytes(
                 ('psin1 %016x ena=0x0010 addr=0x0001\n' % k3_old).encode('ascii'))),
            ('compute missing .sa', 8, lambda d: (d / ('req_%016x.sa' % c_old)).unlink()),
            ('mesh57 missing .ps57', 8, lambda d: (d / ('req_%016x.ps57' % m_old_key)).unlink()),
            ('truncated .rtv', 8,
             lambda d: (d / ('req_%016x.rtv' % old_rtv)).write_bytes(b'rtv1 123')),
            ('missing .rtv base', 7, lambda d: (d / ('req_%016x.a' % k1_old)).unlink()),
        ]
        for label, expected_total, damage in bad_cases:
            bad_src = tmp / ('bad_' + label.replace(' ', '_').replace('.', ''))
            bad_dst = tmp / ('out_' + label.replace(' ', '_').replace('.', ''))
            shutil.copytree(src, bad_src)
            damage(bad_src)
            res = run_cpp(exe, bad_src, bad_dst)
            check('%s exits 1' % label, res.returncode == 1, '%d %s' % (res.returncode, res.stderr))
            check('%s reports failure count without writing' % label,
                  ('error: 1 of %d requests are incomplete or unreadable; nothing written' % expected_total) in res.stderr
                  and not bad_dst.exists(),
                  res.stderr)

        # CLI negative controls (exit 2)
        reg_file = tmp / 'regular_file.txt'
        reg_file.write_bytes(b'not a dir')
        cli_cases = [
            ('src == dst', [src, src]),
            ('missing args', [src]),
            ('extra args', [src, tmp / 'x', tmp / 'y']),
            ('unknown flag', [src, tmp / 'x', '--bogus']),
            ('missing src dir', [tmp / 'does_not_exist', tmp / 'x']),
            ('dst is regular file', [src, reg_file]),
            ('jobs 0', [src, tmp / 'x', '--jobs', '0']),
            ('jobs 65', [src, tmp / 'x', '--jobs', '65']),
        ]
        for label, cli_args in cli_cases:
            res = run_cpp(exe, *cli_args)
            check('CLI %s exits 2' % label, res.returncode == 2, '%d %s' % (res.returncode, res.stderr))

        check('old cache completely unchanged', snapshot(src) == before_src)

    print('%d failed' % len(failures) if failures else 'all checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
