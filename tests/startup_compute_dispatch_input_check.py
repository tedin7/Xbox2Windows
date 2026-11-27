"""Named reasons for a translated compute input that is not a complete GPU-backed buffer range (list slot 14).

usage: startup_compute_dispatch_input_check.py <startup_stream_consumer> <dxc> <warp|hardware>

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
Exit 77 when the adapter is not available.
"""
from math import prod
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import check_input_form, dispatch, input_form_fixture, lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G

CONTROLS = (
    ('dispatch-input-empty', 'an input buffer descriptor has zero records'),
    ('dispatch-input-past', 'an input range crosses the end of a placed buffer'),
    ('dispatch-input-unplaced', 'an input is in title memory outside any placed buffer'),
    ('dispatch-input-copied', 'an input is in a placed buffer that does not alias title memory'),
    ('dispatch-input-token', 'an input view token is not a buffer view of a desktop resource'),
    ('dispatch-input-null-token', 'an input view token is not a buffer view of a desktop resource'),
)


COPY_RAW = """
ByteAddressBuffer input_buffer : register(t0);
RWByteAddressBuffer output_buffer : register(u0);

[numthreads(1, 1, 1)]
void main()
{
    [unroll]
    for (uint offset = 0; offset < 20; offset += 4)
        output_buffer.Store(offset, input_buffer.Load(offset));
}
"""


def raw_tail(directory, consumer, dxc):
    mode = 'dispatch-form-tail'
    refusal = 'an input buffer range is not a whole number of dwords'
    env, _ = input_form_fixture(directory, consumer, dxc, mode, 'srv 0 buf table 1 0', COPY_RAW)
    reference = struct.unpack('<5I', bytes(3 * i + 7 for i in range(20)))
    check_input_form(directory, consumer, dict(env, X2W_COMPUTE_INPUT_FORMS='0'), mode, refusal=refusal)
    expected = check_input_form(directory, consumer, env, mode, reference,
                                totals='target views 0, title heap words 0, rounded raw ranges 1')
    check_input_form(directory, consumer, env, mode + '-upload', reference,
                     totals='target views 0, title heap words 0, rounded raw ranges 1')
    down = check_input_form(directory, consumer, env, mode + '-down', reference[:4] + (0,))
    assert down != expected, 'the rounded-down mutant passed the 20-byte comparison'
    check_input_form(directory, consumer, env, mode + '-end', refusal=refusal,
                     totals='target views 0, title heap words 0, rounded raw ranges 0')
    print('WARP raw tail: all 20 bytes matched, the rounded-down dword differed, and the allocation end was refused')


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare(directory)
        env.pop('X2W_NULL_TOKEN_ZERO', None)
        env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1', X2W_COMPUTE_INPUT_FORMS='0')  # guest lists; debug-layer messages go to the log
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
            raw_tail(directory / 'raw_tail', consumer, dxc)
        print(f'{adapter}: title-input snapshot: off refused, on all {C.WORDS} words match, zero title memory runs on zeros')
        print(f'startup_compute_dispatch_input: PASS on {adapter} (all {C.WORDS} words match, also with the input behind '
              f'a memory view token; {len(CONTROLS)} inputs refused, each with its own reason and an untouched output)')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
