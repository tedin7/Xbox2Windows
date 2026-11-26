"""GPU-written Dispatch arguments consumed by a translated slot 67 pipeline on the same queue (#2492).

`mutation` runs the device-free half: it builds the same 320-word expectation as `check()`, verifies that the
prefilled sentinel readback matches zero written words, that one flipped consumer word and one flipped sentinel are
both caught, and that the 2-group and 5-group references differ only at indices 2, 3 and 4.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines
from startup_compute_dispatch_texture_check import SAMPLED, XOR
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_flat_param_check import assemble


# One thread per group. s7 is GroupID.x, s4 is a root constant, s6 is the buffer table.
SOURCE = """s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0
s_waitcnt lgkmcnt(0)
v_mov_b32 v3, s7
v_lshlrev_b32 v4, 2, v3
buffer_load_dword v5, v4, s[12:15], 0 offen
s_waitcnt vmcnt(0)
v_xor_b32 v5, s4, v5
buffer_store_dword v5, v4, s[16:19], 0 offen
s_endpgm
"""
PROGRAM = bytes.fromhex('06038abe80038bbe05030cf4000000fa7fc08cbf0702067e82060834001030e004050380'
                        '703f8cbf040a0a3a001070e004050480000081bf')


def expected_words(written):
    return [(((0x8b02a7f1 + i * 0x413b1) & C.MASK) ^ C.KEY) if i < written else C.SENTINEL
            for i in range(C.WORDS)]


def compare_words(actual, expected):
    """Return the differing indices, or one length entry when the readback size is wrong."""
    if len(actual) != len(expected):
        return [('length', len(actual), len(expected))]
    return [i for i, (word, want) in enumerate(zip(actual, expected)) if word != want]


def setup(directory, adapter):
    cache, env = prepare(directory)
    env.pop('X2W_INDIRECT_DISPATCH_TABLES', None)
    env.update(X2W_TEST_COMPUTE_ROOT=str(directory / 'compute_root'), X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
    if adapter == 'hardware':
        env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
    else:
        env.pop('X2W_TEST_DISPATCH_HARDWARE', None)
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
    (directory / 'program').write_bytes(PROGRAM)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a((1, 1, 1), C.USER7 | C.TGID_X))
    return cache, env


def check(directory, consumer, env, mode, groups, written, direct=False, reference=None):
    actual, result = dispatch(directory, consumer, env, ('dispatch-reference-' if direct else 'dispatch-indirect-') + mode)
    print(result.stdout, end='')
    expected = expected_words(written) if reference is None else reference
    if direct:
        assert len(lines(result, f'Dispatch({groups}, 1, 1) translated')) == 1, result.stderr[-2500:]
    else:
        assert f'[{groups},{int(bool(groups))},{int(bool(groups))}]' in result.stdout, result.stdout
    wrong = compare_words(actual, expected)
    if wrong:
        first = wrong[0]
        if isinstance(first, tuple):
            raise AssertionError(f'{mode}: readback has {first[1]} words, expected {first[2]}\n'
                                 f'{result.stderr[-2500:]}')
        raise AssertionError(f'{mode}: word {first} got {actual[first]:08x}, expected {expected[first]:08x}, '
                             f'{len(wrong)} differences\n{result.stderr[-2500:]}')
    for line in lines(result, 'x2w-d3d12-debug '):
        assert ' sev=0 ' not in line and ' sev=1 ' not in line, line
    label = 'direct' if direct else 'indirect'
    print(f'{label} {mode}: all {C.WORDS} words match ({written} written, {C.WORDS-written} sentinel)')
    return result


def check_mutation():
    sentinel = [C.SENTINEL] * C.WORDS
    assert len(sentinel) == 320 and C.WORDS == 320, (len(sentinel), C.WORDS)
    assert compare_words(sentinel, expected_words(0)) == [], 'a fully sentinel readback must match zero writes'
    for groups in (2, 5):
        reference = expected_words(groups)
        assert compare_words(list(reference), reference) == [], f'{groups}: reference must match itself'
        assert all(word != C.SENTINEL for word in reference[:groups]), f'{groups}: consumer words use the sentinel'
        for label, index in (('first consumer', 0), ('last consumer', groups - 1),
                             ('first sentinel', groups), ('last sentinel', C.WORDS - 1)):
            mutated = list(reference)
            mutated[index] ^= 1
            assert compare_words(mutated, reference) == [index], f'{groups}: {label} mutation was not isolated'
        assert compare_words(reference[:-1], reference), f'{groups}: a short readback passed'
        assert compare_words(reference + [C.SENTINEL], reference), f'{groups}: a long readback passed'
    two, five = expected_words(2), expected_words(5)
    differing = [i for i in range(C.WORDS) if two[i] != five[i]]
    assert differing == [2, 3, 4], f'2/5 group references differ at {differing}'
    assert compare_words(two, five) == [2, 3, 4]
    print(f'mutation oracle: {C.WORDS} words, sentinel 0x{C.SENTINEL:08x}; consumer and sentinel flips detected; '
          '2/5 references differ at [2, 3, 4]')


def run_mutation():
    try:
        check_mutation()
        root = C.G.ComputePipeline(ROOT_PROGRAM, root_srv_blob(), ROOT_RSRC2, (1, 1, 1))
        assert 'r0.Load(0)' in root.hlsl_cs()
        assert root.manifest().endswith('srv 0 buf root 2\nuav 0 table 1 16\n')
    except AssertionError as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    print('slot67 indirect mutation oracle and root SRV translation: PASS (device-free)')
    return 0


FIXED_TABLE_REFUSAL = 'translated indirect Dispatch requires fixed-table raw buffers'


def check_refusal(directory, consumer, env, mode, reason):
    result = check(directory, consumer, env, mode, 5, 0)
    refusals = lines(result, 'ExecuteIndirect refused')
    assert len(refusals) == 1 and refusals[0].endswith(reason), result.stderr[-2500:]


def publish(directory, consumer, dxc, cache, env, mode):
    check_refusal(directory, consumer, env, mode, 'no translation')
    req, = cache.glob('req_*.c67')
    key = req.stem.removeprefix('req_')
    result = subprocess.run([sys.executable, '-B', C.G.__file__, '--pending', cache], check=True,
                            capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc))
    assert f'{key}: ok' in result.stdout, result.stdout
    return cache / f'{key}.txt'


def with_binding(manifest, replacement):
    text = manifest.read_text()
    original = 'srv 0 buf table 1 0'
    assert text.count(original) == 1
    manifest.write_text(text.replace(original, replacement))


ROOT_SOURCE = """s_mov_b32 s12, s7
s_mov_b32 s13, 0
s_load_dwordx4 s[16:19], s[12:13], 16
s_load_dword s20, s[8:9], 0
s_waitcnt lgkmcnt(0)
v_mov_b32 v3, s10
v_lshlrev_b32 v4, 2, v3
v_mov_b32 v5, s20
v_xor_b32 v5, s4, v5
buffer_store_dword v5, v4, s[16:19], 0 offen
s_endpgm
"""
ROOT_PROGRAM = bytes.fromhex('07038cbe80038dbe060408f4100000fa040500f4000000fa7fc08cbf0a02067e'
                             '8206083414020a7e040a0a3a001070e004050480000081bf')
ROOT_RSRC2 = (10 << 1) | C.TGID_X


def root_srv_blob():
    # RTS0 v1.1: constants at parameter 0, a table at parameter 1, and a root SRV at parameter 2.
    words = (2, 3, 24, 0, 116, 0,
             1, 0, 60, 0, 0, 72, 3, 0, 104,
             0, 0, 3,
             1, 80, 0, 1, 0, 0, 0, 0xffffffff,
             1, 0, 0)
    part = b'RTS0' + struct.pack('<I', 4 * len(words)) + struct.pack('<29I', *words)
    return b'DXBC' + bytes(16) + struct.pack('<4I', 1, 36 + len(part), 1, 36) + part


def check_bindings(directory, consumer, dxc, adapter):
    enabled = {'X2W_INDIRECT_DISPATCH_TABLES': '1'}

    heap = directory / 'heap'
    cache, env = setup(heap, adapter)
    manifest = publish(heap, consumer, dxc, cache, env, 'table-token')
    with_binding(manifest, 'srv 0 buf heap rootconst 0 1')
    check_refusal(heap, consumer, env, 'table-token', FIXED_TABLE_REFUSAL)
    result = check(heap, consumer, dict(env, **enabled), 'table-token', 5, 5)
    assert not lines(result, 'ExecuteIndirect refused'), result.stderr[-2500:]

    texture = directory / 'texture'
    cache, env = setup(texture, adapter)
    C.SPLICE[XOR] = '05190a3a'
    (texture / 'program').write_bytes(C.splice(*SAMPLED))
    _, _, rsrc2, group, _ = C.CASES['A']
    (texture / 'state').write_bytes(state_a(group, rsrc2))
    manifest = publish(texture, consumer, dxc, cache, env, 'sampled')
    assert 'srv 1 tex2d table 1 64' in manifest.read_text()
    check_refusal(texture, consumer, env, 'sampled', FIXED_TABLE_REFUSAL)
    written = 5 * 24
    reference = [word ^ 0x3f800000 for word in C.expected('A')[:written]] + [C.SENTINEL] * (C.WORDS - written)
    result = check(texture, consumer, dict(env, **enabled), 'sampled', 5, written, reference=reference)
    assert not lines(result, 'ExecuteIndirect refused'), result.stderr[-2500:]

    root = directory / 'root-srv'
    cache, env = setup(root, adapter)
    (root / 'compute_root').write_bytes(root_srv_blob())
    (root / 'program').write_bytes(ROOT_PROGRAM)
    (root / 'state').write_bytes(state_a((1, 1, 1), ROOT_RSRC2))
    manifest = publish(root, consumer, dxc, cache, env, 'root-srv')
    assert 'srv 0 buf root 2' in manifest.read_text()
    check_refusal(root, consumer, env, 'root-srv',
                  'a resource that is not a raw table buffer, a 2D texture or a cube map')
    reference = [C.input_word(0) ^ C.KEY] * 5 + [C.SENTINEL] * (C.WORDS - 5)
    for mode in ('root-srv', 'root-srv-upload'):
        result = check(root, consumer, dict(env, **enabled), mode, 5, 5, reference=reference)
        assert not lines(result, 'ExecuteIndirect refused'), result.stderr[-2500:]
    check_refusal(root, consumer, dict(env, **enabled), 'root-srv-unbound',
                  'a compute root SRV address is missing, misaligned or out of range')


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        cache, env = setup(directory, adapter)
        missing = check(directory, consumer, env, '5', 5, 0)
        refused = lines(missing, 'refused')
        assert any('no translation' in line for line in refused), missing.stderr[-2500:]
        print(next(line for line in refused if 'no translation' in line))
        req, = cache.glob('req_*.c67')
        key = req.stem.removeprefix('req_')
        translated = subprocess.run([sys.executable, '-B', C.G.__file__, '--pending', cache], check=True,
                                    capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc))
        assert f'{key}: ok' in translated.stdout, translated.stdout
        for groups in (2, 5, 0):
            run = check(directory, consumer, env, str(groups) if groups else 'zero', groups, groups)
            assert not lines(run, 'refused:'), run.stderr[-2500:]
            if groups:
                check(directory, consumer, env, str(groups), groups, groups, direct=True)
        for mode, reason in (('multi', 'one rootless command'), ('counted', 'no count buffer'),
                             ('outside', 'outside a unique placed buffer'), ('output-alias', 'also a bound shader resource')):
            run = check(directory, consumer, env, mode, 5, 0)
            refusals = lines(run, 'ExecuteIndirect refused')
            assert len(refusals) == 1 and reason in refusals[0], run.stderr[-2500:]
            print(refusals[0])
        manifest = cache / f'{key}.txt'
        text = manifest.read_text()
        buf = 'srv 0 buf table 1 0'
        assert text.count(buf) == 1
        for other in ('srv 0 tex2d table 1 0', 'srv 0 buf heap rootconst 0 0'):
            manifest.write_text(text.replace(buf, other))
            run = check(directory, consumer, env, '5', 5, 0)
            refusals = lines(run, 'ExecuteIndirect refused')
            assert len(refusals) == 1 and 'requires fixed-table raw buffers' in refusals[0], run.stderr[-2500:]
            print(refusals[0])
        layer = lines(run, 'x2w-d3d12 debug layer: ')
        print('debug layer:', 'enabled' if layer and '0x00000000' in layer[0] else 'unavailable')


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'mutation':
        sys.exit(run_mutation())
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        assert assemble(sys.argv[2], SOURCE) == PROGRAM
        assert assemble(sys.argv[2], ROOT_SOURCE) == ROOT_PROGRAM
        print('slot 67 indirect consumers match LLVM gfx1013 assembly')
    elif len(sys.argv) == 5 and sys.argv[3] in ('hardware', 'warp') and sys.argv[4] == 'bindings':
        with tempfile.TemporaryDirectory() as tmp:
            check_bindings(Path(tmp), *sys.argv[1:4])
    elif len(sys.argv) == 4 and sys.argv[3] in ('hardware', 'warp'):
        main(*sys.argv[1:])
    else:
        sys.exit('usage: startup_slot67_indirect_dispatch_check.py <consumer> <dxc> <warp|hardware> [bindings] | '
                 'binary <llvm-mc> | mutation')
