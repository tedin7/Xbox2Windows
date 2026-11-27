"""Named reasons for a translated compute input that is not a complete GPU-backed buffer range (list slot 14).

usage: startup_compute_dispatch_input_check.py <startup_stream_consumer> <dxc> <warp|hardware>
       startup_compute_dispatch_input_check.py --offline <llvm-mc>

The consumer records case A of startup_gfx2hlsl_compute_check.py as startup_compute_dispatch_check.py does. After
gfx2hlsl.py --pending with DXC, the unchanged recording still matches all 320 words. Then one input descriptor at a
time is replaced, and each dispatch is refused with its own reason, logs which input and descriptor once, writes
no output word and leaves the input as written:

- dispatch-input-empty: an unfilled heap slot, submitted twice to check one diagnostic;
- dispatch-input-past: 16 bytes past the end of the placed input buffer;
- dispatch-input-unplaced: memory between the two placed buffers;
- dispatch-input-copied: a buffer placed at an offset the desktop cannot alias, so it is a separate copy;
- dispatch-input-token: the token of a texture view;
- dispatch-input-null-token: a buffer SRV token with no resource. It runs only with
  X2W_NULL_TOKEN_ZERO=1, when all output words must match the CPU reference for zero input.

Last, dispatch-input-memory-view replaces the input descriptor with the token of a buffer view (device slot 70) of
memory no buffer is placed over, 16 bytes in, and writes the input words there after the view exists: all 320
words match. The stand-in places that buffer in the consumer's own heap, outside its placement index.
On WARP, a second program copies all records from title snapshots to a readback buffer.
Exit 77 when the adapter is not available.
"""
from math import prod
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a, write_request
from startup_stream_check import gfx2hlsl as G

CONTROLS = (
    ('dispatch-input-empty', 'an input buffer descriptor has zero records'),
    ('dispatch-input-past', 'an input range crosses the end of a placed buffer'),
    ('dispatch-input-unplaced', 'an input is in title memory outside any placed buffer'),
    ('dispatch-input-copied', 'an input is in a placed buffer that does not alias title memory'),
    ('dispatch-input-token', 'an input view token is not a buffer view of a desktop resource'),
    ('dispatch-input-null-token', 'an input view token is not a buffer view of a desktop resource'),
)


# gfx1013: load the SRV/UAV descriptors from the table, then copy record 64 * group.x + lane.
SNAPSHOT_SOURCE = """\
s_mov_b32 s22, s6
s_mov_b32 s23, 0
s_load_dwordx8 s[24:31], s[22:23], 0
s_lshl_b32 s19, s7, 6
s_waitcnt lgkmcnt(0)
v_add_nc_u32 v7, s19, v0
v_lshlrev_b32 v8, 2, v7
buffer_load_dword v9, v8, s[24:27], 0 offen
s_waitcnt vmcnt(0)
buffer_store_dword v9, v8, s[28:31], 0 offen
s_endpgm
"""
SNAPSHOT_CODE = bytes.fromhex(
    '060396be 800397be 0b060cf4000000fa 0786138f 7fc08cbf 13000e4a '
    '820e1034 001030e008090680 703f8cbf 001070e008090780 000081bf'
)
SNAPSHOT_STATE = state_a((64, 1, 1), C.USER7 | C.TGID_X | C.tidig(1))
SNAPSHOT_WORDS = 40 * 1024 // 4
SNAPSHOT_ROUNDS = 200


def snapshot_reference(words):
    before = [(0x23456789 + i * 0x12345) & 0xffffffff for i in range(words)]
    after = before.copy()
    after[words // 2] ^= 0x10203040
    return before, after


def check_snapshot_reuse(directory, consumer, dxc, cache, env):
    (directory / 'program').write_bytes(SNAPSHOT_CODE)
    (directory / 'state').write_bytes(SNAPSHOT_STATE)
    env = dict(env, X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT='1')
    count = SNAPSHOT_ROUNDS
    words = SNAPSHOT_WORDS
    before, after = snapshot_reference(words)
    expected = before * count

    def run(case, dispatches, *, range_words=words, **options):
        mode = f'dispatch-snapshot-{case}'
        return dispatch(directory, consumer, dict(env, **options), mode, word_count=dispatches * range_words)

    # Publish a new request through the runtime and translate the synthetic machine code with DXC.
    _, result = run('same', count, X2W_COMPUTE_SNAPSHOT_REUSE='0')
    published = lines(result, 'x2w-draw compute request')
    assert len(published) == 1 and ' published ' in published[0], result.stderr[-3000:]
    key = int(published[0].split()[3], 16)
    translated = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True,
                                capture_output=True, text=True,
                                env=dict(os.environ, X2W_DXC=dxc)).stdout
    assert f'{key:016x}: ok' in translated, translated

    def check(case, dispatches, reference, counters, *, reused='1', skip_hash='0', ring_kib=None, range_words=words, verify='0'):
        options = dict(X2W_COMPUTE_SNAPSHOT_REUSE=reused, X2W_COMPUTE_SNAPSHOT_VERIFY=verify,
                       X2W_TEST_COMPUTE_SNAPSHOT_SKIP_HASH=skip_hash)
        if ring_kib is not None:
            options['X2W_TEST_UPLOAD_RING_MAX_KIB'] = str(ring_kib)
        got, result = run(case, dispatches, range_words=range_words, **options)
        assert not lines(result, 'refused:'), result.stderr[-3000:]
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        stats, = lines(result, 'compute snapshots:')
        match = re.search(r'made (\d+), reused (\d+), replaced (\d+), MiB saved ([\d.]+)', stats)
        assert match and tuple(map(int, match.groups()[:3])) == counters, stats
        assert abs(float(match[4]) - counters[1] * range_words * 4 / (1 << 20)) < 0.001, stats
        ring, = lines(result, 'x2w-test snapshot ring:')
        usage = re.search(r'reserved (\d+) bytes, recycled (\d+)', ring)
        assert usage, ring
        if got != reference:
            index = next(i for i, (gpu, cpu) in enumerate(zip(got, reference)) if gpu != cpu)
            raise SnapshotMismatch(index, got[index], reference[index])
        return tuple(map(int, usage.groups()))

    off, _ = check('same', count, expected, (count, 0, 0), reused='0')
    on, _ = check('same', count, expected, (1, count - 1, 0))
    assert off > 0 and on * 20 < off, f'ring bytes on={on}, off={off}'
    changed = before * (count // 2) + after * (count // 2)
    check('rewrite', count, changed, (2, count - 2, 1))
    check('frame', count + 1, expected + before, (2, count - 1, 0))
    check('threads', count, expected, (1, count - 1, 0))
    _, recycled = check('recycle', 2, before * 2, (2, 0, 0), ring_kib=160)
    assert recycled == 1, recycled
    large_words = ((1 << 20) + 4096) // 4
    large, _ = snapshot_reference(large_words)
    check('large', 2, large * 2, (2, 0, 0), range_words=large_words)
    check('large', 2, large * 2, (1, 1, 0), range_words=large_words, verify='1')
    try:
        check('rewrite', count, changed, (1, count - 1, 0), skip_hash='1')
    except SnapshotMismatch as mismatch:
        assert mismatch.args == (count // 2 * words + words // 2, before[words // 2], after[words // 2]), mismatch
    else:
        raise AssertionError('the no-hash mutant passed the rewrite readback')
    print(f'snapshot reuse: same, rewrite, frame, threads and recycle match CPU; ring {on}/{off} bytes; hash mutant rejected')


class SnapshotMismatch(AssertionError):
    """First differing word, GPU value and CPU value."""


def check_snapshot_translation(assembler):
    assert C.assemble(assembler, SNAPSHOT_SOURCE) == SNAPSHOT_CODE
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary)
        key = write_request(cache, SNAPSHOT_CODE, SNAPSHOT_STATE, root=C.root_blob(table_range=True))
        args = [sys.executable, '-B', G.__file__, '--pending', cache]
        result = subprocess.run(args, env=dict(os.environ, X2W_FXC=''), check=True, capture_output=True, text=True)
        assert f'{key:016x}: ok' in result.stdout, result.stdout + result.stderr
        manifest = (cache / f'{key:016x}.txt').read_text()
        assert manifest.splitlines() == [
            'x2w-manifest 2', 'stage cs', 'group 64 1 1', 'ud 2',
            'udsrc 0 rootconst 0 0', 'udsrc 1 rootconst 0 1', 'srv 0 buf table 1 0', 'uav 0 table 1 16',
        ], manifest
    print('snapshot fixture assembly and manifest: PASS')


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare(directory)
        for name in ('X2W_NULL_TOKEN_ZERO', 'X2W_COMPUTE_SNAPSHOT_REUSE', 'X2W_COMPUTE_SNAPSHOT_VERIFY',
                     'X2W_TEST_COMPUTE_SNAPSHOT_SKIP_HASH', 'X2W_TEST_UPLOAD_RING_MAX_KIB',
                     'X2W_D3D12_LINEAR_BUFFERS', 'X2W_D3D12_CAP_EXISTING_HEAP', 'X2W_TEST_DISPATCH_HARDWARE',
                     'X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT'):
            env.pop(name, None)
        env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')  # guest lists; debug-layer messages go to the log
        if adapter == 'hardware':
            env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
        (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
        env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
        _, words, rsrc2, group, groups = C.CASES['A']
        assert groups == (3, 2, 2) and (C.KEY, C.STRIDE) == (0x01234567, 0x9e3779b9), 'the consumer hard-codes case A'
        (directory / 'program').write_bytes(words)
        (directory / 'data').write_bytes(DATA)
        (directory / 'state').write_bytes(state_a(group, rsrc2))
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')

        got, result = dispatch(directory, consumer, env, 'dispatch')
        print(result.stdout, end='')
        assert got == untouched and len(lines(result, 'refused: no translation')) == 1, result.stderr[-3000:]
        published = lines(result, 'x2w-draw compute request')
        assert published and ' published ' in published[0], result.stderr[-3000:]
        key = int(published[0].split()[3], 16)
        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in out, out

        got, result = dispatch(directory, consumer, env, 'dispatch')
        assert got == expected and not lines(result, 'refused'), result.stderr[-3000:]
        used = lines(result, f'x2w-render key use compute {key:016x} first ')
        assert len(used) == 1 and used[0].endswith(' uses 1') and ' last ' in used[0], (used, result.stderr[-3000:])
        layer = lines(result, 'x2w-d3d12 debug layer: ')
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        print('debug layer: ' + ('enabled, no error message' if layer and '0x00000000' in layer[0] else 'unavailable'))

        for mode, reason in CONTROLS:
            got, result = dispatch(directory, consumer, env, mode)
            assert got == untouched, '%s: a refused Dispatch wrote the output' % mode
            refused = lines(result, 'refused: ')
            census = lines(result, 'x2w-render compute census')
            where = lines(result, f'x2w-draw compute {key:016x} first refused input 0 (table ')
            calls = 2 if mode == 'dispatch-input-empty' else 1
            assert len(refused) == calls and all(line.endswith('refused: ' + reason) for line in refused), (mode, refused)
            assert len(census) == 3 and census[1].endswith(f'reason "{reason}": 1 keys, {calls} dispatches'), (mode, census)
            assert len(where) == 1, (mode, result.stderr[-3000:])
            slot_lines = lines(result, 'unfilled heap slot ')
            assert slot_lines == ([f'x2w-draw unfilled heap slot {0x32c - 1} key {key:016x}']
                                  if mode == 'dispatch-input-empty' else []), (mode, slot_lines)

        null_env = dict(env, X2W_NULL_TOKEN_ZERO='1')
        got, result = dispatch(directory, consumer, null_env, 'dispatch-input-empty')
        assert got == untouched and not lines(result, 'Dispatch(3, 2, 2) translated'), result.stderr[-3000:]
        assert lines(result, 'unfilled heap slot ') == [f'x2w-draw unfilled heap slot {0x32c - 1} key {key:016x}'], result.stderr[-3000:]

        got, result = dispatch(directory, consumer, null_env, 'dispatch-input-null-token')
        written = prod(group) * prod(groups)
        zero_expected = [(C.KEY + i * C.STRIDE) & C.MASK for i in range(written)]
        zero_expected += untouched[written:]
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused: '), result.stderr[-3000:]
        assert got == zero_expected, 'null-token output differs from the CPU reference for zero input'
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        got, result = dispatch(directory, consumer, env, 'dispatch-input-memory-view')
        assert got == expected and len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and             not lines(result, 'refused: '), result.stderr[-3000:]
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        # Opt-in X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT: an input in a placed buffer that does not alias title memory holds
        # its bytes only in title memory (the desktop copy is zeros). Off, it is refused; on, the bytes are snapshotted
        # into the upload ring and all 320 words match; with title memory zero (dispatch-input-copied) the dispatch runs
        # on zeros: not refused, not the expected words, not untouched.
        snapshot_env = dict(env, X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT='1')
        got, result = dispatch(directory, consumer, env, 'dispatch-input-snapshot')
        assert got == untouched and lines(result, 'refused: ')[0].endswith(CONTROLS[3][1]), result.stderr[-3000:]
        got, result = dispatch(directory, consumer, snapshot_env, 'dispatch-input-snapshot')
        assert got == expected and len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused: '), result.stderr[-3000:]
        assert len(lines(result, 'comes from a snapshot of title memory (1280 bytes)')) == 1, result.stderr[-3000:]
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        got, result = dispatch(directory, consumer, snapshot_env, 'dispatch-input-copied')
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused: '), result.stderr[-3000:]
        assert got != expected and got != untouched, 'a snapshot of zero title memory must not give the expected words'
        # An all-zero descriptor is an unbound buffer: the dispatch runs on zeros instead of being refused (the empty
        # control above has a nonzero base and stays refused).
        got, result = dispatch(directory, consumer, env, 'dispatch-input-null')
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused: '), result.stderr[-3000:]
        assert len(lines(result, 'comes from zeros for a null descriptor (16 bytes)')) == 1, result.stderr[-3000:]
        assert got == zero_expected, 'all-zero descriptor output differs from the CPU reference for zero input'
        # A manifest whose first input is out of order is not bound, and the log names the line: the pipeline key, the
        # line number, the reason and the line itself.
        manifest = cache / f'{key:016x}.txt'
        original = manifest.read_text()
        assert 'srv 0 ' in original, original
        manifest.write_text(original.replace('srv 0 ', 'srv 3 ', 1))
        got, result = dispatch(directory, consumer, env, 'dispatch')
        manifest.write_text(original)
        first = next(l for l in original.splitlines() if l.startswith('srv 0 ')).replace('srv 0 ', 'srv 3 ', 1)
        rejected = lines(result, f'x2w-draw compute pipeline {key:016x} manifest line ')
        assert got == untouched and len(rejected) == 1 and ' rejected (a resource that is not a raw table buffer' in rejected[0] \
            and rejected[0].endswith(': ' + first[:120]), result.stderr[-3000:]
        if adapter == 'warp':
            check_snapshot_reuse(directory, consumer, dxc, cache, env)
        print(f'{adapter}: title-input snapshot: off refused, on all {C.WORDS} words match, zero title memory runs on zeros')
        print(f'startup_compute_dispatch_input: PASS on {adapter} (all {C.WORDS} words match, also with the input behind '
              f'a memory view token; {len(CONTROLS)} inputs refused, each with its own reason and an untouched output)')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--offline':
        check_snapshot_translation(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
