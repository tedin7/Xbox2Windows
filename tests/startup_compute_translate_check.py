"""Translation of device slot 67 compute requests by gfx2hlsl.py --pending (the compute stage of #2182).

usage: startup_compute_translate_check.py                     request checks, pure Python
       startup_compute_translate_check.py e2e <startup_stream_consumer> <dxc> <offscreen_check> <warp|hardware> [wrong-group]

Request checks: cache files are written here with gfx2hlsl.compute_key. Case A of
startup_gfx2hlsl_compute_check.py, with a 28-byte state block A carrying its group size and COMPUTE_PGM_RSRC2,
translates (manifest only, X2W_FXC empty) to a `stage cs` manifest. Each malformed request is refused into
<key>.fail with no manifest: code of 0 bytes or not whole dwords, code without s_endpgm, a part whose length
disagrees with the size words in .desc, files that do not hash to the key in their name, a state block A that
is not 28 bytes, and a missing part (the .fail text names no path).

e2e: the stand-in publishes case A, gfx2hlsl.py --pending translates it with DXC, and offscreen_check dispatches
the published <key>.cs.cso and compares all 320 words with that test's expectation. This proves the plumbing
is self-consistent (request files, translation, compiled bytecode, dispatch); it does not prove the state-block
mapping, because the fixture builds the block at the offsets the translator reads. "wrong-group" swaps the
group size in the state block (3x2x4 for 2x3x4); the readback must then mismatch.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_request_check import DATA, STATE_B, prepare, run
from startup_stream_check import gfx2hlsl as G


def state_a(group, rsrc2):
    # The observed offsets: dword 3 = X | Y << 16, dword 4 = Z, dword 6 = COMPUTE_PGM_RSRC2.
    return struct.pack('<7I', 0, 0, 0, group[0] | group[1] << 16, group[2], 0, rsrc2)


def write_request(cache, code, state, data=DATA, root=None, desc_sizes=None):
    root = C.root_blob() if root is None else root
    sizes = desc_sizes or (len(state), len(STATE_B), len(code), len(data))
    desc = struct.pack('<12Q', 0, 0, sizes[0], 0, sizes[1], 0, sizes[2], 0, sizes[3], 0, 0, 11)
    key = G.compute_key(code, data, root, desc, state, STATE_B)
    for ext, part in (('.desc', desc), ('.sa', state), ('.sb', STATE_B), ('.data', data), ('.root', root), ('.c67', code)):
        (cache / ('req_%016x%s' % (key, ext))).write_bytes(part)
    return key


def check_requests(directory):
    _, words, rsrc2, group, _ = C.CASES['A']
    good = state_a(group, rsrc2)
    cache = directory / 'cache'
    cache.mkdir()
    env = dict(os.environ, X2W_FXC='')
    key = write_request(cache, words, good)
    assert G.translate_request(str(cache), key) == 'ok' if os.environ.update(env) is None else False
    manifest = (cache / ('%016x.txt' % key)).read_text()
    assert manifest.startswith('x2w-manifest 2\nstage cs\ngroup 2 3 4\n'), manifest
    assert '[numthreads(2, 3, 4)]' in (cache / ('%016x.cs.hlsl' % key)).read_text()

    end = words.rindex(struct.pack('<I', G.S_ENDPGM))
    cases = {
        'code of 0 bytes': (lambda: write_request(cache, b'', good), 'compute code of 0 bytes is not whole dwords'),
        'code of 6 bytes': (lambda: write_request(cache, words[:6], good), 'compute code of 6 bytes is not whole dwords'),
        'no s_endpgm': (lambda: write_request(cache, words[:end], good), 'compute code without s_endpgm'),
        'data size word': (lambda: write_request(cache, words, good, desc_sizes=(28, 20, len(words), len(DATA) + 8)),
                           'compute request .data is 16 bytes, the description says 24'),
        'state block A size': (lambda: write_request(cache, words, good[:24], desc_sizes=(24, 20, len(words), 16)),
                               'compute state block A is 24 bytes, not 28'),
    }
    for name, (make, reason) in cases.items():
        bad = make()
        result = G.translate_request(str(cache), bad)
        assert result == 'unsupported: ' + reason, (name, result)
        assert (cache / ('%016x.fail' % bad)).read_text() == reason + '\n', name
        assert not (cache / ('%016x.txt' % bad)).exists(), name
    # Files renamed to another key, and a missing part.
    moved = key ^ 1
    for ext in ('.desc', '.sa', '.sb', '.data', '.root', '.c67'):
        (cache / ('req_%016x%s' % (moved, ext))).write_bytes((cache / ('req_%016x%s' % (key, ext))).read_bytes())
    assert G.translate_request(str(cache), moved) == 'unsupported: compute request files do not match the key in their name'
    (cache / ('req_%016x.root' % moved)).unlink()
    assert G.translate_request(str(cache), moved) == 'unsupported: unreadable shader request'
    assert str(cache) not in (cache / ('%016x.fail' % moved)).read_text()
    # The pending scan treats the translated request as done (manifest-only mode) and retries the others.
    pending = G.pending_keys(str(cache), skip_failed=False)
    assert key not in pending and moved in pending, pending
    # A retry of a previously published request must withdraw it if a required part vanishes.
    root = cache / ('req_%016x.root' % key)
    saved_root = root.read_bytes()
    root.unlink()
    assert G.translate_request(str(cache), key) == 'unsupported: unreadable shader request'
    assert not (cache / ('%016x.txt' % key)).exists(), 'a manifest was left next to a refused request'
    assert key in G.pending_keys(str(cache), skip_failed=False)
    root.write_bytes(saved_root)
    assert G.translate_request(str(cache), key) == 'ok'
    # A failed compile withdraws the manifest an earlier pass left. The Python interpreter stands in for DXC and
    # exits non-zero on DXC's options.
    os.environ.pop('X2W_FXC')
    os.environ['X2W_DXC'] = sys.executable
    assert (cache / ('%016x.txt' % key)).exists()
    result = G.translate_request(str(cache), key)
    assert result == 'dxc cs compile failed', result
    report = (cache / ('%016x.fail' % key)).read_text()
    assert report == 'dxc cs compile failed\n', report
    assert not (cache / ('%016x.txt' % key)).exists(), 'a manifest was left next to the failure'
    assert key in G.pending_keys(str(cache), skip_failed=False)
    # With compilation on (X2W_FXC unset), a compute request is done only with its manifest and non-empty
    # <key>.cs.cso; graphics bytecode names do not count.
    (cache / ('%016x.fail' % key)).unlink()
    (cache / ('%016x.txt' % key)).write_text('x2w-manifest 2\nstage cs\n')
    assert key in G.pending_keys(str(cache), skip_failed=False), 'a manifest alone made the request done'
    for stage in ('vs', 'ps'):
        (cache / ('%016x.%s.cso' % (key, stage))).write_bytes(b'DXBC')
    assert key in G.pending_keys(str(cache), skip_failed=False), 'graphics bytecode made a compute request done'
    (cache / ('%016x.cs.cso' % key)).write_bytes(b'')
    assert key in G.pending_keys(str(cache), skip_failed=False), 'an empty .cs.cso made the request done'
    (cache / ('%016x.cs.cso' % key)).write_bytes(b'DXBC')
    assert key not in G.pending_keys(str(cache), skip_failed=False), 'manifest and .cs.cso did not make it done'
    print('compute request translation: case A manifest, %d malformed requests refused with no manifest, '
          'renamed and incomplete requests refused, done test, failed compile withdraws the manifest, '
          'compiled done test' % len(cases))


def check_e2e(directory, consumer, dxc, runner, adapter, wrong_group):
    cache, env = prepare(directory)
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))  # the consumer creates it on a device
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    _, words, rsrc2, group, groups = C.CASES['A']
    if wrong_group:
        group = (group[1], group[0], group[2])
    lines = run(directory, consumer, env, data=DATA, state=state_a(group, rsrc2), program=words)
    assert lines and ' published ' in lines[0], lines
    key = int(lines[0].split()[3], 16)
    out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                         text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
    assert f'{key:016x}: ok' in out, out
    manifest = (cache / ('%016x.txt' % key)).read_text()
    assert manifest.startswith('x2w-manifest 2\nstage cs\ngroup %d %d %d\n' % group), manifest
    case = directory / 'case'
    case.write_bytes(struct.pack('<I', groups[0] | groups[1] << 8 | groups[2] << 16)
                     + struct.pack('<%dI' % C.WORDS, *C.expected('A')) + struct.pack('<16I', C.KEY, C.STRIDE, *[0] * 14))
    result = subprocess.run([str(Path(runner).resolve()), adapter, '--compute-store-case',
                             str(cache / ('%016x.cs.cso' % key)), str(case)], capture_output=True, text=True)
    print(result.stdout, end='')
    if result.returncode == 77:
        return 77
    if wrong_group:
        if result.returncode != 1 or 'expected=' not in result.stderr:
            raise AssertionError(f'wrong group size not rejected (exit {result.returncode}) {result.stderr}')
        print('control wrong-group: rejected at ' + result.stderr.strip().splitlines()[0])
        return 0
    if result.returncode or result.stderr:
        raise AssertionError(result.stderr or f'exit {result.returncode}')
    print(f'startup_compute_translate_e2e: PASS on {adapter} (request {key:016x} published by the stand-in, translated '
          f'and compiled by gfx2hlsl.py --pending, all {C.WORDS} words match)')
    return 0


if __name__ == '__main__':
    args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temporary:
        if not args:
            check_requests(Path(temporary))
            sys.exit(0)
        if len(args) in (5, 6) and args[0] == 'e2e' and args[5:] in ([], ['wrong-group']):
            sys.exit(check_e2e(Path(temporary), args[1], args[2], args[3], args[4], args[5:] == ['wrong-group']))
        sys.exit(__doc__)
