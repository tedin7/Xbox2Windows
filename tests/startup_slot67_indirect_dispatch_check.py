"""GPU-written Dispatch arguments consumed by a translated slot 67 pipeline on the same queue (#2492).

`mutation` runs the device-free half: it builds the same 320-word expectation as `check()`, verifies that the
prefilled sentinel readback matches zero written words, that one flipped consumer word and one flipped sentinel are
both caught, and that the 2-group and 5-group references differ only at indices 2, 3 and 4.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines
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


def check(directory, consumer, env, mode, groups, written, direct=False):
    actual, result = dispatch(directory, consumer, env, ('dispatch-reference-' if direct else 'dispatch-indirect-') + mode)
    print(result.stdout, end='')
    expected = expected_words(written)
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


# Install a source-built desktop program under the synthetic request key to isolate indirect dispatch
# and binding from ISA translation. Two threads per group append records in unspecified order.
APPEND = """RWByteAddressBuffer output : register(u0);
[numthreads(2, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint index;
    output.InterlockedAdd(0, 1, index);
    output.Store(4 + 4 * index, 0xa700 + 2 * group.x + lane);
}
"""
BATCH_GROUPS = (2, 3, 1)
BATCH_SWITCH = 'X2W_TRANSLATED_INDIRECT_BATCH'
SPLIT_SWITCH = 'X2W_SPLIT_INDIRECT_DISPATCH'
OLD_REFUSAL = 'translated Dispatch requires one rootless command and no count buffer'


def batch_reference(commands):
    return sorted(0xa700 + 2 * x + lane
                  for groups in BATCH_GROUPS[:commands] for x in range(groups) for lane in range(2))


def batch_mismatch(actual, commands):
    expected = batch_reference(commands)
    count = len(expected)
    if actual[0] != count:
        return f'append counter {actual[0]}, expected {count}'
    records = sorted(actual[1:count + 1])
    if records != expected:
        return f'records {records}, expected {expected}'
    wrong = compare_words(actual[count + 1:], [C.SENTINEL] * (C.WORDS - count - 1))
    if wrong:
        return f'writes past append records: {wrong}'
    return None


def check_batches(directory, consumer, dxc, cache, key, env):
    source = directory / 'append.hlsl'
    source.write_text(APPEND, encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cache / f'{key}.cs.cso'), str(source)], check=True)
    (cache / f'{key}.txt').write_text('x2w-manifest 2\nstage cs\ngroup 2 1 1\nud 0\nuav 0 table 1 16\n', encoding='ascii')

    def check_case(mode, commands, enabled='1', split='0', reason=None):
        actual, result = dispatch(directory, consumer, dict(env, **{BATCH_SWITCH: enabled, SPLIT_SWITCH: split}),
                                  f'dispatch-indirect-batch{mode}')
        mismatch = batch_mismatch(actual, commands)
        assert mismatch is None, f'{mode or "uncounted"} (split={split}): {mismatch}\n{result.stderr[-2500:]}'
        refusals = lines(result, 'ExecuteIndirect refused')
        if reason:
            assert len(refusals) == 1 and reason in refusals[0], result.stderr[-2500:]
        else:
            assert not refusals, result.stderr[-2500:]
            census = lines(result, f'compute census key {key}: 1 dispatches, 1 translated')
            assert census, result.stderr[-2500:]
        for line in lines(result, 'x2w-d3d12-debug '):
            assert ' sev=0 ' not in line and ' sev=1 ' not in line, line
        print(f'batch{mode} (split={split}): {actual[0]} records match the CPU reference')
        return actual

    check_case('', 0, enabled='0', reason=OLD_REFUSAL)
    check_case('-counted', 0, enabled='0', reason=OLD_REFUSAL)
    for split in ('0', '1'):
        check_case('', 3, split=split)
        check_case('-counted', 2, split=split)
        check_case('-zero', 0, split=split)
    check_case('-short', 0, reason='argument record is outside a unique placed buffer')
    check_case('-argument-output', 0, reason='the indirect argument buffer is also a dispatch output')
    check_case('-count-output', 0, reason='the indirect count buffer is also a dispatch output')
    # A live first-command control must disagree with the entire-batch expectation.
    first_only = check_case('-first-only', 1)
    assert batch_mismatch(first_only, 3) == 'append counter 4, expected 12'


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
    for commands, count in ((0, 0), (1, 4), (2, 10), (3, 12)):
        records = batch_reference(commands)
        assert len(records) == count
        readback = [count] + list(reversed(records)) + [C.SENTINEL] * (C.WORDS - count - 1)
        assert batch_mismatch(readback, commands) is None
        if commands == 1:
            assert batch_mismatch(readback, 3) == 'append counter 4, expected 12'
        if commands:
            readback[1] ^= 1
            assert batch_mismatch(readback, commands).startswith('records ')
    print(f'mutation oracle: {C.WORDS} words, sentinel 0x{C.SENTINEL:08x}; consumer and sentinel flips detected; '
          '2/5 references differ at [2, 3, 4]')


def run_mutation():
    try:
        check_mutation()
    except AssertionError as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    print('slot67 indirect mutation oracle: PASS (device-free)')
    return 0


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        cache, env = setup(directory, adapter)
        env.update({BATCH_SWITCH: '0', SPLIT_SWITCH: '0'})
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
        if adapter == 'warp':
            check_batches(directory, consumer, dxc, cache, key, env)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'mutation':
        sys.exit(run_mutation())
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        assert assemble(sys.argv[2], SOURCE) == PROGRAM
        print('slot 67 indirect consumer matches LLVM gfx1013 assembly')
    elif len(sys.argv) == 4 and sys.argv[3] in ('hardware', 'warp'):
        main(*sys.argv[1:])
    else:
        sys.exit('usage: startup_slot67_indirect_dispatch_check.py <consumer> <dxc> <warp|hardware> | '
                 'binary <llvm-mc> | mutation')
