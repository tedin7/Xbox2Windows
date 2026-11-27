"""Translated compute programs whose entry values are words of a bound buffer descriptor (list slot 14).

usage: startup_compute_dispatch_entry_check.py <startup_stream_consumer> <dxc> <warp|hardware>

The consumer records each program as startup_compute_dispatch_check.py does, with the input words and the stride,
record count and word 3 of both table descriptors set by this script (X2W_TEST_DISPATCH_IN and _FIELDS). Each
program writes the same lanes in every group, so the consumer's Dispatch(3, 2, 2) gives the result of one group.

1. Case L-uint of startup_gfx2hlsl_compute_check.py reads its input with buffer_load_format_xyzw; the manifest
   takes `stride 0` and `descword3 0` from the input's descriptor. Stride 6 and the word 3 of L-uint, then of
   L-unorm on the same translation, must each give that case's 320 words. A FORMAT the program does not decode is
   refused with one reason and keeps every output word at its sentinel.
2. Case F of startup_compute_store_d16_x_check.py stores with buffer_store_format_d16_x; the manifest takes
   `uav_descword3 0`, `uav_stride 0` and `uav_numrecords 0` from the output's descriptor. Three descriptors give
   three different expectations: 32_FLOAT with stride 8 and 12 records, 16_FLOAT with the same, and 32_FLOAT with
   stride 12 and 7 records. A fourth sets SWIZZLE_ENABLE (word 1 bit 31): the dispatch is translated and the
   program stores nothing, which holds only while the entry value keeps the upper bits of word 1.
Before translation each program is refused with "no translation". Exit 77 when the adapter is not available.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import startup_compute_store_d16_x_check as D
import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G

RAW = (4, C.WORDS, 0)  # the consumer's default stride, record count and word 3
UNDECODED = 'a formatted input has no word 3 the program decodes'


def store_f32(halves, stride, records):
    out = bytearray(struct.pack('<%dI' % C.WORDS, *[C.SENTINEL] * C.WORDS))
    for lane in range(records):
        struct.pack_into('<I', out, lane * stride + 4, D.f16_to_f32_bits(halves[lane]))
    return list(struct.unpack('<%dI' % C.WORDS, out))


def run_program(root, name, consumer, dxc, adapter, program, manifest_tail, runs):
    directory = root / name
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')  # guest lists; debug-layer messages go to the log
    if adapter == 'hardware':
        env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
    (directory / 'compute_root').write_bytes(C.root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
    words, rsrc2, group = program
    (directory / 'program').write_bytes(words)
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(group, rsrc2))
    untouched = [C.SENTINEL] * C.WORDS

    def run(source, fields):
        (directory / 'in.bin').write_bytes(source)
        (directory / 'fields.bin').write_bytes(struct.pack('<6I', *fields[0], *fields[1]))
        return dispatch(directory, consumer, dict(env, X2W_TEST_DISPATCH_IN=str(directory / 'in.bin'),
                                                  X2W_TEST_DISPATCH_FIELDS=str(directory / 'fields.bin')), 'dispatch')

    source, fields, _ = runs[0][1:]
    got, result = run(source, fields)
    print(result.stdout, end='')
    assert got == untouched and len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
    key = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
    out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                         text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
    assert f'{key:016x}: ok' in out, out
    manifest = (cache / f'{key:016x}.txt').read_text()
    assert manifest_tail in manifest, manifest

    for label, source, fields, expected in runs:
        got, result = run(source, fields)
        refused = lines(result, 'refused: ')
        if isinstance(expected, str):
            assert got == untouched, f'{label}: a refused Dispatch wrote the output'
            census = lines(result, 'x2w-render compute census')
            assert len(refused) == 1 and refused[0].endswith('refused: ' + expected), f'{label}: {refused}\n{result.stderr[-3000:]}'
            assert len(census) == 3 and census[1].endswith(f'reason "{expected}": 1 keys, 1 dispatches'), (label, census)
            print(f'{label}: refused ({expected}), output untouched')
            continue
        assert (expected == untouched) == label.endswith('-swizzle')
        wrong =[i for i in range(C.WORDS) if got[i] != expected[i]]
        assert not wrong, '%s: word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
            label, wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not refused, result.stderr[-3000:]
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        layer = lines(result, 'x2w-d3d12 debug layer: ')
        print(f'{label}: all {C.WORDS} words match (debug layer: '
              + ('enabled, no error message)' if layer and '0x00000000' in layer[0] else 'unavailable)'))


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _, words_l, rsrc2_l, group_l, _ = C.CASES['L-uint']
        formatted = bytes(C.formatted_input())
        undecoded = (C.L_WORD3["L-uint"] & ~(127 << 12)) | 3 << 12  # BUF_FMT_8_USCALED, in no format table
        assert C.L_STRIDE == 6 and not (undecoded >> 12) & 127 in (1, 5, 14, 22, 23, 24, 27, 29, 56, 60, 64, 71, 74, 77)
        # 20 records of 6 bytes cover the last lane's 4-byte read at byte 6 * 15 + 3.
        run_program(root, 'formatted', consumer, dxc, adapter, (words_l, rsrc2_l, group_l),
                    'udsrc 2 stride 0\nudsrc 3 descword3 0\nsrv 0 buf table 1 0\nuav 0 table 1 16\n', [
                        ('dispatch-formatted-uint', formatted, ((6, 20, C.L_WORD3['L-uint']), RAW), C.expected('L-uint')),
                        ('dispatch-formatted-unorm', formatted, ((6, 20, C.L_WORD3['L-unorm']), RAW), C.expected('L-unorm')),
                        ('dispatch-formatted-undecoded', formatted, ((6, 20, undecoded), RAW), UNDECODED),
                    ])

        _, words_f, rsrc2_f, group_f, _ = D.PROGRAMS['F']
        runs = []
        for variant, label in (('F-f32', 'f32'), ('F-f16', 'f16'), ('F-swizzle-en', 'f32-swizzle')):
            _, source, constants, expected, _ = D.case_inputs(variant)
            runs.append(('dispatch-store-' + label, source, (RAW, tuple(constants[3:5]) + (constants[2],)), expected))
        word3_f32, halves = D.F_VARIANTS['F-f32'][0], D.FP16_VALUES
        runs.append(('dispatch-store-f32-stride12', D.poisoned_source(halves), (RAW, (12, 7, word3_f32)), store_f32(halves, 12, 7)))
        assert runs[0][3] != runs[3][3] != runs[1][3]
        run_program(root, 'store', consumer, dxc, adapter, (words_f, rsrc2_f, group_f),
                    'udsrc 2 uav_descword3 0\nudsrc 3 uav_stride 0\nudsrc 4 uav_numrecords 0\n', runs)
        print(f'startup_compute_dispatch_entry: PASS on {adapter} (6 descriptor-word dispatches match, '
              '1 refused; untouched before translation)')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
