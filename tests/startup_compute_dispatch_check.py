"""A translated compute program dispatched through the stand-in's command list (list slot 14).

usage: startup_compute_dispatch_check.py <startup_stream_consumer> <dxc> <warp|hardware>

The consumer creates case A of startup_gfx2hlsl_compute_check.py as a device slot 67 pipeline and records, on a
stand-in list, the descriptor heap, the pipeline, its root signature, two root constants (word 0 by slot 37, word 1
by slot 39, with a slot 40 call of other words in between), one table of two
hardware buffer descriptors and Dispatch(3, 2, 2). Input and output are placed buffers that alias the consumer's
memory; the output words are read from that memory after the queue's fence. Four runs share one cache:

1. before translation: the request is published, the Dispatch is refused with "no translation" and every output
   word keeps the sentinel (the check fails when nothing translated is dispatched);
2. after gfx2hlsl.py --pending with DXC: all 320 words equal the case's expectation;
3. dispatch-none (no Dispatch recorded): every word keeps the sentinel;
4. dispatch-swapped (the two descriptors exchanged, so the output names the buffer without unordered access):
   refused with one logged reason, every word keeps the sentinel and the input is unchanged;
5. dispatch-same-root (the same compute root set again after the constants): all 320 words still match;
6. five more refusals, each with its one logged reason and an untouched output: dispatch-missing-word (no slot 37
   call), dispatch-root-changed (another compute root and back after the constants), dispatch-reset (list closed
   and reset after the constants), dispatch-wrong-root (recorded under a root the pipeline was not created with)
   and dispatch-same-buffer (the output buffer is also the input).
Exit 77 when the adapter is not available.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G


def dispatch(directory, consumer, env, mode):
    out = directory / 'out.bin'
    out.unlink(missing_ok=True)
    args = [consumer, directory / 'vertex', directory / 'pixel', directory / 'root', 'slot67',
            directory / 'program', directory / 'data', directory / 'state', mode]
    result = subprocess.run(args, env=dict(env, X2W_TEST_DISPATCH_OUT=str(out)), capture_output=True, text=True)
    if result.returncode == 77:
        print(result.stdout, end='')
        sys.exit(77)
    assert result.returncode == 0, f'{mode}: consumer exit {result.returncode}\n{result.stderr[-3000:]}'
    return list(struct.unpack('<%dI' % C.WORDS, out.read_bytes())), result


def lines(result, text):
    return [line for line in result.stderr.splitlines() if text in line]


def prepare_case(directory, adapter):
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
    if adapter == 'hardware':
        env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
    else:
        env.pop('X2W_TEST_DISPATCH_HARDWARE', None)
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    _, words, rsrc2, group, groups = C.CASES['A']
    assert groups == (3, 2, 2) and (C.KEY, C.STRIDE) == (0x01234567, 0x9e3779b9), 'the consumer hard-codes case A'
    (directory / 'program').write_bytes(words)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(group, rsrc2))
    return cache, env


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, adapter)
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')
        assert expected != untouched

        got, result = dispatch(directory, consumer, env, 'dispatch')
        print(result.stdout, end='')
        assert got == untouched, 'a Dispatch without a translation wrote the output'
        assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
        published = lines(result, 'x2w-draw compute request')
        assert published and ' published ' in published[0], result.stderr[-3000:]
        key = int(published[0].split()[3], 16)
        census = lines(result, 'x2w-render compute census')
        assert census == ['x2w-render compute census: 1 keys dispatched',
                          'x2w-render compute census reason "no translation": 1 keys, 1 dispatches',
                          f'x2w-render compute census key {key:016x}: 1 dispatches, 0 translated, first refusal: '
                          'no translation [00000000 00000000 00000000 00000000]'], census

        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in out, out

        got, result = dispatch(directory, consumer, env, 'dispatch')
        wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
        assert not wrong, 'word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
            wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused'), result.stderr[-3000:]
        census = lines(result, 'x2w-render compute census')
        assert len(census) == 2 and census[1].endswith(': 1 dispatches, 1 translated, first refusal: none '
                                                       '[00000000 00000000 00000000 00000000]'), census
        layer = lines(result, 'x2w-d3d12 debug layer: ')
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        print('debug layer: ' + ('enabled, no error message' if layer and '0x00000000' in layer[0] else 'unavailable'))

        got, result = dispatch(directory, consumer, env, 'dispatch-none')
        assert got == untouched and not lines(result, 'Dispatch('), 'the output changed without a Dispatch'
        assert not lines(result, 'x2w-render compute census'), 'a census without a Dispatch'

        got, result = dispatch(directory, consumer, env, 'dispatch-swapped')
        assert got == untouched, 'a refused Dispatch wrote the output'
        refused = lines(result, 'refused: ')
        census = lines(result, 'x2w-render compute census')
        assert len(census) == 3 and census[1].endswith(
            'reason "an output buffer was created without unordered access": 1 keys, 1 dispatches') \
            and '[00000000 00000000 ' not in census[2], census
        assert len(refused) == 1 and 'an output buffer was created without unordered access' in refused[0], \
            result.stderr[-3000:]
        # The output descriptor is the token of a view the title made over the placed output buffer: the dispatch makes the
        # desktop UAV on that buffer, so every word still matches and the log says where the output came from.
        got, result = dispatch(directory, consumer, env, 'dispatch-output-token')
        wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
        assert not wrong, 'output token: word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
            wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
        assert not lines(result, 'refused') and len(lines(result, 'from a view token')) == 1, result.stderr[-3000:]
        got, result = dispatch(directory, consumer, env, 'dispatch-same-root')
        assert got == expected and not lines(result, 'refused'), 'setting the same compute root again lost the constants'

        unset = 'an entry value was not set since the compute root'
        for mode, reason in (('dispatch-missing-word', unset), ('dispatch-root-changed', unset), ('dispatch-reset', unset),
                             ('dispatch-wrong-root', 'the compute root set is not the one the pipeline was created with'),
                             ('dispatch-same-buffer', 'an output buffer is also an input')):
            got, result = dispatch(directory, consumer, env, mode)
            assert got == untouched, f'{mode}: a refused Dispatch wrote the output'
            refused = lines(result, 'refused: ')
            assert len(refused) == 1 and refused[0].endswith('refused: ' + reason), f'{mode}: {refused}\n{result.stderr[-3000:]}'
        print(f'startup_compute_dispatch: PASS on {adapter} (all {C.WORDS} words match after translation and with the '
              'same root set again; untouched before translation, without a Dispatch and for 6 refusals)')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
