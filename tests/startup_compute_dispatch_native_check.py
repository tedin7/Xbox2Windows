"""A slot 67 compute program translated in-process by the C++ compiler and dispatched through the stand-in (#2934).

usage: startup_compute_dispatch_native_check.py <startup_stream_consumer> <DXC directory> <warp|hardware>

X2W_NATIVE_SHADERS names the directory of dxcompiler.dll and dxil.dll. The cache starts empty and no offline
translator runs before step 3. The program is case A of startup_gfx2hlsl_compute_check.py with its two buffer
descriptors loaded from 32-byte table records (two s_load_dwordx4, at table bytes 0 and 32), the layout the C++
compiler reads a table in. The consumer's dispatch-records32 mode writes the table that way. The reference is
that case's CPU expectation.

1. Native: the first Dispatch translates the request in the stand-in and is recorded. All 320 words match, the log
   names the native translation, and the cache holds request files and compiler cache entries only (no <key>.txt,
   <key>.cs.cso or <key>.fail). The same program over the plain table layout, which has no descriptor at byte 32,
   is then refused when it is bound and every word keeps the sentinel: the record at byte 32 is what was bound.
2. Unsupported workload: case A itself reads its second descriptor from table byte 16. The C++ compiler refuses
   it, the reason is logged, the Dispatch is refused with "no translation" and every word keeps the sentinel.
3. Fallback: gfx2hlsl.py --pending then publishes both requests, and with native translation still on the next run
   dispatches case A through the published translation. All 320 words match.
4. Order: the first program runs again, now with a published translation in the cache. The native translation is
   still the one used, and all 320 words match.

A direct compute module declares wave size 32. On an adapter without 32-lane waves, step 1 instead checks that the
native translation is refused by that name with the output untouched, steps 2 and 3 run as written, step 4 runs the
first program through its published translation, and the check exits 77. Exit 77 also when the adapter is not
available.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_stream_check import gfx2hlsl as G

LOAD_X8 = bytes.fromhex('05030cf4' '000000fa')
LOADS_X4 = bytes.fromhex('050308f4' '000000fa' '050408f4' '200000fa')
RECORD_LOADS = ['s_load_dwordx4 s[12:15], s[10:11], 0x0', 's_load_dwordx4 s[16:19], s[10:11], 0x20']
NO_WAVES = 'not translated natively: the adapter has no 32-lane waves'


def mismatch(got, expected, result):
    wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
    return '' if not wrong else 'word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
        wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])


def main(consumer, dxc_dir, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, adapter)
        env['X2W_NATIVE_SHADERS'] = dxc_dir
        case_a = (directory / 'program').read_bytes()
        records32 = case_a.replace(LOAD_X8, LOADS_X4)
        assert case_a.count(LOAD_X8) == 1 and [i[3] for i in G.decode(records32)[2:4]] == RECORD_LOADS
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')

        (directory / 'program').write_bytes(records32)
        got, result = dispatch(directory, consumer, env, 'dispatch-records32')
        print(result.stdout, end='')
        key32 = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
        no_waves = bool(lines(result, NO_WAVES))
        if no_waves:
            assert got == untouched, 'a Dispatch without 32-lane waves wrote the output'
            assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
        else:
            assert not mismatch(got, expected, result), mismatch(got, expected, result)
            assert len(lines(result, 'translated natively: 2 entry values, 1 inputs, 1 outputs')) == 1, result.stderr[-3000:]
            assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused'), result.stderr[-3000:]
            layer = lines(result, 'x2w-d3d12 debug layer: ')
            errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
            assert not errors, errors[0]
            print('debug layer: ' + ('enabled, no error message' if layer and '0x00000000' in layer[0] else 'unavailable'))
            got, result = dispatch(directory, consumer, env, 'dispatch')
            refused = lines(result, 'refused: ')
            assert got == untouched, 'a Dispatch with no descriptor at table byte 32 wrote the output'
            assert len(refused) == 1 and 'no translation' not in refused[0], result.stderr[-3000:]
        published = sorted(p.name for p in cache.iterdir() if not p.name.startswith('req_') and p.name != 'native')
        assert not published, f'the native route left {published} in the cache'

        (directory / 'program').write_bytes(case_a)
        got, result = dispatch(directory, consumer, env, 'dispatch')
        reason = lines(result, 'not translated natively: ')
        assert got == untouched, 'a Dispatch of a program the C++ compiler refused wrote the output'
        assert len(reason) == 1 and 'unaligned descriptor table entry' in reason[0], result.stderr[-3000:]
        assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
        key = int(reason[0].split('compute pipeline ')[1].split()[0], 16)

        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=str(Path(dxc_dir) / 'dxc.exe'))).stdout
        assert f'{key:016x}: ok' in out, out
        got, result = dispatch(directory, consumer, env, 'dispatch')
        assert not mismatch(got, expected, result), mismatch(got, expected, result)
        assert lines(result, 'not translated natively: ') and len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, \
            result.stderr[-3000:]

        assert f'{key32:016x}: ok' in out, out
        (directory / 'program').write_bytes(records32)
        got, result = dispatch(directory, consumer, env, 'dispatch-records32')
        assert not mismatch(got, expected, result), mismatch(got, expected, result)
        assert bool(lines(result, f'{key32:016x} translated natively')) != no_waves, result.stderr[-3000:]
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]

        controls = ('the C++ compiler refuses case A with every word untouched, and a published translation then runs '
                    'it')
        if no_waves:
            print(f'startup_compute_dispatch_native: SKIP on {adapter}: the adapter has no 32-lane waves. Checked: the '
                  f'native translation is refused by that name with every word untouched; {controls}; the first '
                  f'program matches all {C.WORDS} words through its published translation')
            sys.exit(77)
        shared_loads = LOADS_X4[:4] + bytes.fromhex('200000fa') + LOADS_X4[8:]
        (directory / 'program').write_bytes(case_a.replace(LOAD_X8, shared_loads))
        got, result = dispatch(directory, consumer, dict(env, X2W_NATIVE_SHADERS_ONLY='1'), 'dispatch-read-write-uav')
        expected[288:] = [C.input_word(i) for i in range(288, C.WORDS)]
        assert not mismatch(got, expected, result), mismatch(got, expected, result)
        assert lines(result, '0 inputs, 1 outputs'), result.stderr[-3000:]
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        print('shared guest UAV: in-place reads and writes matched with untouched tail')
        print(f'startup_compute_dispatch_native: PASS on {adapter} (all {C.WORDS} words match after the in-process '
              f'translation from an empty cache, and again with a published translation in the cache; the plain table '
              f'layout is refused when bound; {controls})')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
