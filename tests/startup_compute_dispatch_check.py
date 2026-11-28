"""A translated compute program dispatched through the stand-in's command list (list slot 14).

usage: startup_compute_dispatch_check.py <startup_stream_consumer> <dxc> <warp|hardware>

The consumer creates case A of startup_gfx2hlsl_compute_check.py as a device slot 67 pipeline and records, on a
stand-in list, the descriptor heap, the pipeline, its root signature, two root constants (word 0 by slot 37, word 1
by slot 39, with a slot 40 call of other words in between), one table of two
hardware buffer descriptors and Dispatch(3, 2, 2). Input and output are placed buffers that alias the consumer's
memory; the output words are read from that memory after the queue's fence. The runs share one cache:

1. before translation: the request is published, the Dispatch is refused with "no translation" and every output
   word keeps the sentinel (the check fails when nothing translated is dispatched);
2. after gfx2hlsl.py --pending with DXC: all 320 words equal the case's expectation;
3. dispatch-none (no Dispatch recorded): every word keeps the sentinel;
4. dispatch-swapped (the two descriptors exchanged, so the output names the buffer without unordered access):
   refused with "an output buffer has no unordered access", every word keeps the sentinel and the input is
   unchanged;
5. dispatch-same-root (the same compute root set again after the constants): all 320 words still match;
6. five more refusals, each with its one logged reason and an untouched output: dispatch-missing-word (no slot 37
   call), dispatch-root-changed (another compute root and back after the constants), dispatch-reset (list closed
   and reset after the constants), dispatch-wrong-root (recorded under a root the pipeline was not created with)
   and dispatch-same-buffer (the output buffer is also the input).
7. slot 35 selects the correct table at descriptor index 2 instead of the swapped pair at the heap start;
   setting the same heap again preserves the table, but changing the compute root or view heap after binding
   must refuse at the table lookup, not an earlier check.
8. published output compatibility: the same bytecode is refused with either stale native manifest
   and runs with the current native marker or with the Python sidecar.
Exit 77 only if the initial probe cannot use the adapter or required heaps.
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


def dispatch(directory, consumer, env, mode, *, probe=False):
    out = directory / 'out.bin'
    out.unlink(missing_ok=True)
    args = [consumer, directory / 'vertex', directory / 'pixel', directory / 'root', 'slot67',
            directory / 'program', directory / 'data', directory / 'state', mode]
    result = subprocess.run(args, env=dict(env, X2W_TEST_DISPATCH_OUT=str(out)), capture_output=True, text=True)
    if 'device removed' in result.stderr.lower():
        raise RuntimeError(f'{mode}: device removed\n{result.stderr[-3000:]}')
    if result.returncode == 77 and probe:
        print(result.stdout, end='')
        sys.exit(77)
    if result.returncode:
        raise RuntimeError(f'{mode}: consumer exit {result.returncode}\n{result.stdout[-1000:]}\n{result.stderr[-3000:]}')
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
        env.pop('X2W_NATIVE_SHADERS', None)  # Exercise published translations.
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')
        assert expected != untouched

        got, result = dispatch(directory, consumer, env, 'dispatch', probe=True)
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

        manifest = cache / f'{key:016x}.txt'
        hlsl = cache / f'{key:016x}.cs.hlsl'
        python_manifest = manifest.read_bytes()
        python_hlsl = hlsl.read_bytes()
        assert not any(line.startswith(b'native ') for line in python_manifest.splitlines())
        hlsl.unlink()
        for marker, reason in (
            (b'', 'native translation without a revision; rebuild it'),
            (b'native 0\n', 'native translation from another compiler revision; rebuild it'),
        ):
            manifest.write_bytes(python_manifest + marker)
            got, result = dispatch(directory, consumer, env, 'dispatch')
            assert got == untouched, 'a stale native output wrote the output buffer'
            assert len(lines(result, 'Dispatch(3, 2, 2) refused: ' + reason)) == 1, result.stderr[-3000:]

        # Reuse working bytecode to isolate the manifest gate from native adapter requirements.
        manifest.write_bytes(python_manifest + b'native 1\n')
        got, result = dispatch(directory, consumer, env, 'dispatch')
        assert got == expected and not lines(result, 'refused:'), result.stderr[-3000:]
        hlsl.write_bytes(python_hlsl)
        manifest.write_bytes(python_manifest)

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
            'reason "an output buffer has no unordered access": 1 keys, 1 dispatches') \
            and '[00000000 00000000 ' not in census[2], census
        assert len(refused) == 1 and 'an output buffer has no unordered access' in refused[0], \
            result.stderr[-3000:]
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
        debug_enabled = bool(layer and '0x00000000' in layer[0])
        for mode in ('dispatch-table-root-changed', 'dispatch-table-heap-changed', 'dispatch-table-offset'):
            got, result = dispatch(directory, consumer, env, mode)
            if debug_enabled:
                initialized = lines(result, 'x2w-d3d12 debug layer: ')
                assert len(initialized) == 1 and '0x00000000' in initialized[0], f'{mode}: debug layer not initialized'
            refused = lines(result, 'refused: ')
            translated = lines(result, 'Dispatch(3, 2, 2) translated')
            if mode == 'dispatch-table-offset':
                wrong = sum(a != b for a, b in zip(got, expected))
                assert not wrong, f'{mode}: {wrong} words differ from the CPU reference\n{result.stderr[-3000:]}'
                assert len(translated) == 1 and not refused, result.stderr[-3000:]
                print(f'{mode}: all {C.WORDS} words match the CPU reference')
            else:
                written = sum(word != C.SENTINEL for word in got)
                assert not written, f'{mode}: {written} output words changed after table invalidation\n{result.stderr[-3000:]}'
                assert not translated and len(refused) == 1 and refused[0].endswith(
                    'Dispatch(3, 2, 2) refused: a table descriptor is not bound'), result.stderr[-3000:]
                print(f'{mode}: exact table refusal, all {C.WORDS} sentinel words unchanged')
            errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
            assert not errors, errors[0]
        print(f'startup_compute_dispatch: PASS on {adapter}')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
