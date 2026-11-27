"""A translated compute program whose input is the resource heap record a word of a table buffer indexes (slot 14).

usage: startup_compute_dispatch_heap_word_check.py <startup_stream_consumer> <dxc> <warp|hardware>

The consumer creates the program of startup_gfx2hlsl_heap_word_check.py as a device slot 67 pipeline and records it
as startup_compute_dispatch_heap_check.py does, with root constant word 2 set to 1 and a buffer in the recording's
upload memory (list slot 148) at table byte 96. Word 288 + 336 * 1 of that buffer is 2, the heap record of the
input. Heap records 0 and 1 hold the output's descriptor and every other word of the buffer names one of them, so
a dispatch that ignores the word, the root constant, its scale or the byte offset is refused instead of passing.
Runs sharing one cache:

1. before translation: the request is published and the Dispatch is refused with "no translation";
2. after gfx2hlsl.py --pending with DXC: the manifest names the word and all 320 words equal the expectation of
   case A of startup_gfx2hlsl_compute_check.py;
3. dispatch-heap-word-outside (the word is 0x07ffffff, a record past the heap), dispatch-heap-word-past (the
   buffer ends at the word) and dispatch-heap-word-placed (the buffer is the placed input, not upload memory):
   each refused with one logged reason, every output word keeps the sentinel and the input is unchanged.
Exit 77 when the adapter is not available.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
import startup_gfx2hlsl_heap_record_check as H
import startup_gfx2hlsl_heap_word_check as W
from startup_compute_dispatch_check import COPY_TEXTURE, check_input_form, dispatch, input_form_fixture, lines, texture_word
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G

CONTROLS = (
    ('dispatch-heap-word-outside', "a heap record is outside the list's view heap"),
    ('dispatch-heap-word-past', 'a heap index word is outside its table buffer'),
    ('dispatch-heap-word-placed', "a heap index word is not in the list's upload memory"),
)


def placed_index(directory, consumer, dxc):
    mode = 'dispatch-form-index'
    env, _ = input_form_fixture(directory, consumer, dxc, mode,
                                'srv 0 tex2d heap word table 1 96 at 0', COPY_TEXTURE)
    expected = check_input_form(directory, consumer, env, mode, (texture_word(7.5),),
                                totals='target views 0, title heap words 1, rounded raw ranges 0')
    check_input_form(directory, consumer, env, mode + '-plain', (texture_word(7.5),),
                     totals='target views 0, title heap words 1, rounded raw ranges 0')
    zero = check_input_form(directory, consumer, env, mode + '-zero', (texture_word(2.25),))
    assert zero != expected, 'the zero-index mutant passed the second-texture comparison'
    check_input_form(directory, consumer, env, mode + '-gpu', refusal='a heap index word is produced on the GPU',
                     totals='target views 0, title heap words 0, rounded raw ranges 0')
    print('WARP heap word: CPU index selected the second texture, zero differed, and GPU-written index was refused')


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare(directory)
        env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1', X2W_COMPUTE_INPUT_FORMS='0')  # guest lists; debug-layer messages go to the log
        if adapter == 'hardware':
            env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
        (directory / 'compute_root').write_bytes(H.root_blob(table_range=True))
        env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
        assert H.GROUPS == (3, 2, 2) and W.FORM == '288 rootconst 0 2 times 336', 'the consumer hard-codes these'
        (directory / 'program').write_bytes(W.WORDS)
        (directory / 'data').write_bytes(DATA)
        (directory / 'state').write_bytes(state_a(H.GROUP, H.RSRC2))
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')
        assert expected != untouched

        got, result = dispatch(directory, consumer, env, 'dispatch-heap-word')
        print(result.stdout, end='')
        assert got == untouched, 'a Dispatch without a translation wrote the output'
        assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
        published = lines(result, 'x2w-draw compute request')
        assert published and ' published ' in published[0], result.stderr[-3000:]
        key = int(published[0].split()[3], 16)

        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in out, out
        assert (cache / f'{key:016x}.txt').read_text() == W.MANIFEST % W.FORM

        got, result = dispatch(directory, consumer, env, 'dispatch-heap-word')
        wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
        assert not wrong, 'word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
            wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused'), result.stderr[-3000:]
        layer = lines(result, 'x2w-d3d12 debug layer: ')
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        print('debug layer: ' + ('enabled, no error message' if layer and '0x00000000' in layer[0] else 'unavailable'))

        for mode, reason in CONTROLS:
            got, result = dispatch(directory, consumer, env, mode)
            assert got == untouched, '%s: a refused Dispatch wrote the output' % mode
            refused = lines(result, 'refused: ')
            census = lines(result, 'x2w-render compute census')
            assert len(refused) == 1 and reason in refused[0], '%s:\n%s' % (mode, result.stderr[-3000:])
            assert len(census) == 3 and census[1].endswith(f'reason "{reason}": 1 keys, 1 dispatches'), (mode, census)
        # A null descriptor for the index buffer (all words zero) reads as 0: the dispatch runs and reads heap record 0,
        # which this run makes the input, so every word equals the expectation. The Run S refusal was this case.
        got, result = dispatch(directory, consumer, env, 'dispatch-heap-word-null')
        wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
        assert not wrong, 'null index buffer: word=%d got=%08x expected=%08x\n%s' % (
            wrong[0], got[wrong[0]], expected[wrong[0]], result.stderr[-3000:])
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused'), result.stderr[-3000:]
        if adapter == 'warp':
            placed_index(directory / 'placed_index', consumer, dxc)
        print(f'startup_compute_dispatch_heap_word: PASS on {adapter} (all {C.WORDS} words match through the index word; '
              f'untouched before translation and for {len(CONTROLS)} refused controls)')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
