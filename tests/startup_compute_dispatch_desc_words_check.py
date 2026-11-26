"""Run synthetic descriptor-word, mask and wide-record probes through the WARP stand-in."""

import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_compute_dispatch_check import dispatch, lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_gfx2hlsl_compute_check import G, SENTINEL, WORDS, root_blob
from startup_gfx2hlsl_heap_record_check import root_blob as heap_root
from startup_gfx2hlsl_heap_word_check import MIXED_WORDS, PAIR_WORDS, RAW_WORDS
from startup_gfx2hlsl_mask_data_check import MASK_WORDS


def replace_once(path, before, after):
    text = path.read_text()
    assert text.count(before) == 1, before
    path.write_text(text.replace(before, after))


def run_case(directory, consumer, dxc, case):
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')
    env.pop('X2W_TEST_DISPATCH_HARDWARE', None)
    root = directory / 'compute_root'
    heap = case in ('pair', 'mixed')
    root.write_bytes(heap_root(table_range=True) if heap else root_blob(table_range=True))
    env['X2W_TEST_COMPUTE_ROOT'] = str(root)
    fields = directory / 'fields'
    env['X2W_TEST_DISPATCH_FIELDS'] = str(fields)
    (directory / 'program').write_bytes(
        {'word': RAW_WORDS, 'pair': PAIR_WORDS, 'mask': MASK_WORDS, 'mixed': MIXED_WORDS}[case])
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a((8, 1, 1), (9 if heap else 7) << 1))

    def bind(records, mode):
        fields.write_bytes(struct.pack('<6I', 4, records, 53, 4, WORDS, 0))
        return dispatch(directory, consumer, env, 'dispatch-descriptor-' + mode)

    def translated(records, mode, payload, *, mutant=False):
        got, result = bind(records, mode)
        assert lines(result, 'Dispatch(1, 1, 1) translated'), result.stderr[-3000:]
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors
        expected = payload + [SENTINEL] * (WORDS - len(payload))
        mismatch = [i for i, (actual, wanted) in enumerate(zip(got, expected)) if actual != wanted]
        assert all(i < len(payload) for i in mismatch), mismatch
        if mutant:
            assert mismatch, 'the GPU mutant matched the CPU reference'
            print(f'{case} mutant differs at {len(mismatch)} words, first at {mismatch[0]}')
        else:
            assert not mismatch, [(i, got[i], expected[i]) for i in mismatch[:8]]

    mode = case if heap else 'bound'
    got, result = bind(141, mode)
    assert got == [SENTINEL] * WORDS
    request, = [line for line in lines(result, 'x2w-draw compute request') if ' published ' in line]
    key = request.split()[3]
    result = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache],
                            env=dict(env, X2W_DXC=dxc), check=True, capture_output=True, text=True)
    assert f'{key}: ok' in result.stdout, result.stdout + result.stderr
    manifest, hlsl, shader = (cache / (key + suffix) for suffix in ('.txt', '.cs.hlsl', '.cs.cso'))

    def compile_mutant(before, after):
        replace_once(hlsl, before, after)
        subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(shader), str(hlsl)], check=True)

    if case == 'word':
        for records in (141, 98):
            translated(records, mode, [records * 9, 0] * 8)
        translated(0, 'null', [0, 1] * 8)
        translated(87, 'buffer-token', [87 * 9, 0] * 8)
        width = 6
        image_word = (1 << 31) | ((width - 1) >> 2)
        translated(141, 'texture-token', [(image_word * 9) & 0xffffffff, 0] * 8)
        replace_once(manifest, 'desc_word table 1 0 2', 'desc_word table 1 0 3')
        translated(141, mode, [141 * 9, 0] * 8, mutant=True)
        replace_once(manifest, 'desc_word table 1 0 3', 'desc_word table 1 0 2')
        compile_mutant('scc = ud3 != 0u;', 'scc = ud3 == 0u;')
        translated(0, 'null', [0, 1] * 8, mutant=True)
    elif case == 'mask':
        records = 141
        payload = [23 if records & (1 << lane) else 17 for lane in range(8)]
        translated(records, mode, payload)
        compile_mutant('ud2 >> (tid.x', 'ud2 >> (16u + tid.x')
        translated(records, mode, payload, mutant=True)
    else:
        payload = [0x3f800000, 0] * 8
        translated(141, mode, payload)
        wrong_offset = 48 if case == 'mixed' else 0
        replace_once(manifest, '0 2 -1 0 0 0 32', f'0 2 -1 0 0 0 {wrong_offset}')
        translated(141, mode, payload, mutant=True)


if __name__ == '__main__':
    consumer, dxc, case = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temporary:
        run_case(Path(temporary), consumer, dxc, case)
