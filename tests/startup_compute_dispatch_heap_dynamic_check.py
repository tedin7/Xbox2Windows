"""Read a synthetic compute shader's dynamic raw-buffer heap records on WARP."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import startup_gfx2hlsl_compute_check as C
import startup_gfx2hlsl_heap_record_check as H
from startup_compute_dispatch_check import dispatch, lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a

SWITCH = 'X2W_DYNAMIC_RESOURCE_HEAP'
MODE = 'dispatch-heap-dynamic'
REFUSAL = 's_load of a resource heap record whose offset is not a root constant shifted left by 5'
MANIFEST = H.MANIFEST.replace('srv 0 buf heap rootconst 0 2', 'dynamic_resource_heap')


def program(shift):
    # s_lshl_b32 s6, s14, <shift>, with s14 holding GroupID.x. The offset is computed on the GPU.
    return H.WORDS.replace(bytes.fromhex(H.ENCODED[H.SHIFT]), bytes((0x0e, 0x80 + shift, 0x06, 0x8f)), 1)


def check_lowering():
    def pipeline(code):
        return H.G.ComputePipeline(code, H.root_blob(), H.RSRC2, H.GROUP)

    with patch.dict(os.environ, {SWITCH: '0'}):
        H.refused(REFUSAL, lambda: pipeline(program(6)))
        old = pipeline(H.WORDS).hlsl_cs()
    with patch.dict(os.environ, {SWITCH: '1'}):
        assert pipeline(H.WORDS).hlsl_cs() == old
        p = pipeline(program(6))
        hlsl = p.hlsl_cs()
        assert p.manifest() == MANIFEST
        assert 'uint index = offset / 32u;' in hlsl
        assert 'if ((offset & 31u) != 0u || index >= ud3) return 0u;' in hlsl
        assert 'ByteAddressBuffer buffer = ResourceDescriptorHeap[index];' in hlsl
        assert 'v5 = load_heap_buffer(heap_offset_' in hlsl
        for load, reason in (
            ('s_load_dwordx8 s[16:23], s[8:9], s6', REFUSAL),
            ('s_load_dwordx4 s[16:19], s[8:9], s6 offset:0x20', REFUSAL),
            ('s_load_dwordx4 s[16:19], s[8:9], m0', REFUSAL),
        ):
            code = program(6).replace(bytes.fromhex(H.ENCODED[H.LOAD]), bytes.fromhex(H.ENCODED[load]))
            H.refused(reason, lambda: pipeline(code))
    print('dynamic resource heap lowering and opt-in controls: PASS')


def expected(shift):
    out = [C.SENTINEL] * C.WORDS
    for group in range(12):
        record = ((group % 3) << shift) // 32
        for i in range(group * 24, (group + 1) * 24):
            word = {0: C.input_word(i), 2: C.input_word(i + 4)}.get(record, 0)
            out[i] = ((word * 33) & C.MASK) ^ ((C.KEY + i * C.STRIDE) & C.MASK)
    return out


def readback(directory, consumer, dxc, shift):
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1', X2W_TEST_DISPATCH_HARDWARE='',
               X2W_TEST_COMPUTE_ROOT=str(directory / 'compute_root'))
    env[SWITCH] = '1'
    (directory / 'compute_root').write_bytes(H.root_blob(table_range=True))
    (directory / 'program').write_bytes(program(shift))
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state_a(H.GROUP, H.RSRC2))
    untouched = [C.SENTINEL] * C.WORDS

    got, result = dispatch(directory, consumer, env, MODE)
    print(result.stdout, end='')
    assert got == untouched and lines(result, 'Dispatch(3, 2, 2) refused: no translation'), result.stderr[-3000:]
    published = lines(result, 'x2w-draw compute request')
    assert published and ' published ' in published[0], result.stderr[-3000:]
    key = int(published[0].split()[3], 16)
    translate_env = dict(os.environ, X2W_DXC=dxc)
    translate_env.pop('X2W_FXC', None)
    translate_env[SWITCH] = '1'
    result = subprocess.run([sys.executable, '-B', H.G.__file__, '--pending', cache],
                            check=True, capture_output=True, text=True, env=translate_env)
    assert f'{key:016x}: ok' in result.stdout, result.stdout + result.stderr
    assert (cache / f'{key:016x}.txt').read_text() == MANIFEST

    got, result = dispatch(directory, consumer, dict(env, **{SWITCH: '0'}), MODE)
    assert got == untouched and lines(result, 'the dynamic resource heap is disabled'), result.stderr[-3000:]
    got, result = dispatch(directory, consumer, env, MODE)
    if lines(result, 'the dynamic resource heap requires SM 6.6 and resource binding tier 3'):
        print('skip: WARP lacks SM 6.6 or resource binding tier 3')
        sys.exit(77)
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]
    errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
    assert not errors, errors[0]
    return got


def compare(got, want):
    for i, (actual, expected_word) in enumerate(zip(got, want, strict=True)):
        assert actual == expected_word, f'word={i} got={actual:08x} expected={expected_word:08x}'


def main(consumer, dxc):
    check_lowering()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        want = expected(6)
        compare(readback(directory / 'bounds', consumer, dxc, 6), want)
        mutant = readback(directory / 'wrong-stride', consumer, dxc, 5)
        compare(mutant, expected(5))
        try:
            compare(mutant, want)
        except AssertionError as error:
            assert str(error).startswith('word=24 '), error
            print(f'control wrong heap offset: rejected at {error}')
        else:
            raise AssertionError('the wrong heap offset passed the original CPU reference')
    print(f'dynamic resource heap: PASS on WARP ({C.WORDS} words, null record and out-of-range controls)')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_lowering()
    elif len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        sys.exit(__doc__)
