"""Synthetic gfx1013 offset proofs, with optional D3D12 WARP dispatch and fenced readback."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

import startup_gfx2hlsl_heap_record_check as H
from startup_compute_dispatch_check import dispatch, lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_gfx2hlsl_compute_check import SENTINEL, WORDS, assemble, refused

G = H.G
GROUP = (1, 1, 1)
REFUSAL = 'whose offset is not a root constant shifted left by 5 since the last branch target'
BRANCH = (
    's_lshl_b32 s30, s6, 5',
    's_mov_b32 s26, 53',
    's_cmp_eq_u32 s9, 1',
    's_cbranch_scc1 1',
    's_mov_b32 s26, 61',
)
AFFINE = ('s_add_i32 s29, s6, 2', 's_mul_i32 s30, 32, s29', 's_mov_b32 s26, 53')
LOOP = BRANCH[:2] + (
    's_mov_b32 s31, 2',
    's_sub_u32 s31, s31, 1',
    's_cmp_lg_u32 s31, 0',
    's_cbranch_scc1 -3',
)
TAIL = (
    's_mul_i32 s27, s11, 6',
    's_mul_i32 s28, s10, 3',
    's_add_u32 s27, s27, s28',
    's_add_u32 s27, s27, s9',
    's_mov_b32 s9, 0',
    's_load_dwordx4 s[32:35], s[8:9], s30',
    's_mov_b32 s15, 0',
    's_mov_b32 s14, s7',
    's_load_dwordx4 s[36:39], s[14:15], 16',
    's_waitcnt lgkmcnt(0)',
    's_lshl_b32 s27, s27, 2',
    'v_mov_b32 v9, s27',
    'buffer_load_dword v10, v9, s[32:35], 0 offen',
    's_waitcnt vmcnt(0)',
    'v_xor_b32 v10, s26, v10',
    'buffer_store_dword v10, v9, s[36:39], 0 offen',
    's_endpgm',
)
CONTROLS = (
    's_lshl_b32 s30, s6, 5',  # independently computed on the fallthrough edge
    's_lshl_b32 s30, s5, 5',  # a different root word on the fallthrough edge
    's_add_u32 s30, s30, 1',  # record alignment is lost
    's_mul_i32 s30, 31, s29',
)

ENCODING = {
    's_lshl_b32 s30, s6, 5': '06851e8f',
    's_mov_b32 s26, 53': 'b5039abe',
    's_cmp_eq_u32 s9, 1': '098106bf',
    's_cbranch_scc1 1': '010085bf',
    's_mov_b32 s26, 61': 'bd039abe',
    's_add_i32 s29, s6, 2': '06821d81',
    's_mul_i32 s30, 32, s29': 'a01d1e93',
    's_mov_b32 s31, 2': '82039fbe',
    's_sub_u32 s31, s31, 1': '1f819f80',
    's_cmp_lg_u32 s31, 0': '1f8007bf',
    's_cbranch_scc1 -3': 'fdff85bf',
    's_mul_i32 s27, s11, 6': '0b861b93',
    's_mul_i32 s28, s10, 3': '0a831c93',
    's_add_u32 s27, s27, s28': '1b1c1b80',
    's_add_u32 s27, s27, s9': '1b091b80',
    's_mov_b32 s9, 0': '800389be',
    's_load_dwordx4 s[32:35], s[8:9], s30': '040808f40000003c',
    's_mov_b32 s15, 0': '80038fbe',
    's_mov_b32 s14, s7': '07038ebe',
    's_load_dwordx4 s[36:39], s[14:15], 16': '070908f4100000fa',
    's_waitcnt lgkmcnt(0)': '7fc08cbf',
    's_lshl_b32 s27, s27, 2': '1b821b8f',
    'v_mov_b32 v9, s27': '1b02127e',
    'buffer_load_dword v10, v9, s[32:35], 0 offen': '001030e0090a0880',
    's_waitcnt vmcnt(0)': '703f8cbf',
    'v_xor_b32 v10, s26, v10': '1a14143a',
    'buffer_store_dword v10, v9, s[36:39], 0 offen': '001070e0090a0980',
    's_endpgm': '000081bf',
    's_lshl_b32 s30, s5, 5': '05851e8f',
    's_add_u32 s30, s30, 1': '1e811e80',
    's_mul_i32 s30, 31, s29': '9f1d1e93',
}


def program(prefix):
    return b''.join(bytes.fromhex(ENCODING[line]) for line in prefix + TAIL)


def pipeline(prefix):
    return G.ComputePipeline(program(prefix), H.root_blob(), H.RSRC2, GROUP)


def check_proof():
    for prefix, add in ((BRANCH, 0), (AFFINE, 64), (BRANCH[:-1] + CONTROLS[:1], 0), (LOOP, 0)):
        pipe = pipeline(prefix)
        hlsl = pipe.hlsl_cs()
        assert f'srv 0 buf heap affine rootconst 0 2 scale 32 add {add}\n' in pipe.manifest()
        assert f'(ud2 * 32u + {add}u) / 32u' in hlsl
        assert 'if (heapIndex0 >=' in hlsl and 'ResourceDescriptorHeap[heapIndex0]' in hlsl
        assert 'register(t0)' not in hlsl
    for prefix in (BRANCH[:-1] + CONTROLS[1:2], AFFINE + CONTROLS[2:3], AFFINE[:1] + CONTROLS[3:]):
        refused(REFUSAL, lambda: pipeline(prefix))
    with patch.object(G.Stage, 'heap_affine_offsets', return_value={}):
        for prefix in BRANCH, AFFINE:
            refused(REFUSAL, lambda: pipeline(prefix))
    print('Heap offset proofs and negative controls passed')


def expect_words(got, expected):
    for i, (actual, want) in enumerate(zip(got, expected, strict=True)):
        assert actual == want, f'word {i}: {actual:#010x}, expected {want:#010x}'


def check_warp(consumer, dxc):
    input_words = [((i + 11) * 0x15983) ^ 0xf351a298 for i in range(WORDS)]
    untouched = [SENTINEL] * WORDS
    for prefix, mode in ((BRANCH, 'dispatch-heap-affine-branch'), (AFFINE, 'dispatch-heap-affine-add')):
        expected = [value ^ (61 if prefix == BRANCH and i % 3 != 1 else 53)
                    for i, value in enumerate(input_words[:12])] + untouched[12:]
        assert all(word != SENTINEL for word in expected[:12])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache, env = prepare(directory)
            (directory / 'compute_root').write_bytes(H.root_blob(table_range=True))
            (directory / 'program').write_bytes(program(prefix))
            (directory / 'data').write_bytes(DATA)
            (directory / 'state').write_bytes(state_a(GROUP, H.RSRC2))
            (directory / 'input').write_bytes(struct.pack(f'<{WORDS}I', *input_words))
            env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1',
                       X2W_TEST_COMPUTE_ROOT=str(directory / 'compute_root'),
                       X2W_TEST_DISPATCH_IN=str(directory / 'input'))

            got, result = dispatch(directory, consumer, env, mode)
            expect_words(got, untouched)
            assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
            published = lines(result, 'x2w-draw compute request')
            assert published and ' published ' in published[0], result.stderr[-3000:]
            key = published[0].split()[3]
            out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True,
                                 capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
            assert f'{key}: ok' in out, out

            got, result = dispatch(directory, consumer, env, mode)
            expect_words(got, expected)
            check_dispatch(result)
            print(result.stdout, end='')
            layer = lines(result, 'x2w-d3d12 debug layer: ')
            print(f'Windows build {sys.getwindowsversion().build}, debug layer: ' +
                  ('enabled' if layer and '0x00000000' in layer[0] else 'unavailable'))

            got, result = dispatch(directory, consumer, env, mode + '-outside')
            expect_words(got, untouched)
            assert len(lines(result, "refused: a heap record is outside the list's view heap")) == 1, result.stderr[-3000:]

            source = (cache / f'{key}.cs.hlsl').read_text()
            term = '/ 32u +'
            assert source.count(term) == 1
            # An incorrect byte-to-record calculation drives the checked desktop index out of bounds.
            mutant = directory / 'mutant.hlsl'
            mutant.write_text(source.replace(term, '/ 32u + 0x40000000u +'))
            subprocess.run([dxc, '-Gis', '-T', 'cs_6_6', '-E', 'main', '-Fo', cache / f'{key}.cs.cso', mutant],
                           check=True)
            got, result = dispatch(directory, consumer, env, mode)
            check_dispatch(result)
            expect_words(got, untouched)
            assert got != expected, 'GPU readback accepted the mutant'
            print(f'{mode}: WARP CPU reference, out-of-range record and GPU mutant passed')


def check_dispatch(result):
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused'), result.stderr[-3000:]
    errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
    assert not errors, errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    parser.add_argument('--warp', nargs=2, metavar=('CONSUMER', 'DXC'))
    args = parser.parse_args()
    if args.llvm_mc:
        for line, encoded in ENCODING.items():
            assert assemble(args.llvm_mc, line + '\n').hex() == encoded, line
    check_proof()
    if args.warp:
        check_warp(*args.warp)
