"""Dispatch shaders whose heap index comes from a raw root SRV or a fixed heap buffer."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_compute_dispatch_check import dispatch, lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_gfx2hlsl_dynamic_heap_check import G, compile_shader, mutate, root_blob

GROUP = (1, 1, 1)
RSRC2 = 11 << 1
SENTINEL = 0xc5c5c5c5
BUFFER_WORD = 0x428a6000
ROOT_LOAD = bytes.fromhex('840100f4040000fa')  # s_load_dword s6, s[8:9], 4
CS = bytes.fromhex(
    '07038cbe'          # s_mov_b32 s12, s7
    '80038dbe'          # s_mov_b32 s13, 0
    '060508f4100000fa'  # s_load_dwordx4 s[20:23], s[12:13], 0x10
) + ROOT_LOAD + bytes.fromhex(
    '7fc08cbf'          # s_waitcnt lgkmcnt(0)
    '0685068f'          # s_lshl_b32 s6, s6, 5
    '80038bbe'          # s_mov_b32 s11, 0
    '050408f40000000c'  # s_load_dwordx4 s[16:19], s[10:11], s6
    '7fc08cbf'          # s_waitcnt lgkmcnt(0)
    '8002087e'          # v_mov_b32 v4, 0
    '001030e004050480'  # buffer_load_dword v5, v4, s[16:19], 0 offen
    '703f8cbf'          # s_waitcnt vmcnt(0)
    '001070e004050580'  # buffer_store_dword v5, v4, s[20:23], 0 offen
    '000081bf')

HEAP_LOAD = bytes.fromhex(
    '80038bbe'          # s_mov_b32 s11, 0
    '050608f4000000fa'  # s_load_dwordx4 s[24:27], s[10:11], 0
    '7fc08cbf'          # s_waitcnt lgkmcnt(0)
    '8c0120f4040000fa') # s_buffer_load_dword s6, s[24:27], 4
CASES = (
    (CS.replace(ROOT_LOAD, HEAP_LOAD), 'srv 0 buf heap fixed 0', '0x00000004u + 0', '0x00000000u + 0', False),
    (CS, 'srv 0 buf root 2', '4', '0', False),
    (CS, 'srv 0 buf root 2', '4', '0', True),
)


def translate(code, binding, address, cbv):
    pipe = G.ComputePipeline(code, root_blob(compute=True, cbv=cbv), RSRC2, GROUP)
    source = pipe.hlsl_cs()
    assert 'dynamic_heap\n' in pipe.manifest() and binding + '\n' in pipe.manifest()
    assert f'r0.Load({address})' in source and '>> 5;' in source and 'ResourceDescriptorHeap[' in source
    return source


def check_dispatch(directory, consumer, env, mode, first, *, mismatch=None, swizzled=False):
    got, result = dispatch(directory, consumer, env, mode)
    assert len(lines(result, 'Dispatch(1, 1, 1) translated')) == 1, result.stderr[-3000:]
    refusals = lines(result, 'refused: ')
    if swizzled:
        diagnostic, = refusals
        assert diagnostic.startswith('x2w-draw input V# '), diagnostic
        assert diagnostic.endswith('an input buffer descriptor is swizzled (V# swizzle_enable); '
                                   'swizzled addressing is not implemented'), diagnostic
        assert int(diagnostic.split()[4], 16) & (1 << 31), diagnostic
    else:
        assert not refusals, result.stderr[-3000:]
    assert not [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line], result.stderr[-3000:]
    expected = [first] + [SENTINEL] * 319
    if mismatch is not None:
        assert got != expected and got == [mismatch] + expected[1:], got[:4]
    else:
        assert got == expected, got[:4]


def gpu_case(consumer, dxc, code, binding, address, wrong_address, cbv):
    source = translate(code, binding, address, cbv)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        cache, env = prepare(directory)
        env.pop('X2W_FXC', None)
        env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1', X2W_TEST_COMPUTE_ROOT=str(directory / 'compute_root'))
        (directory / 'compute_root').write_bytes(root_blob(compute=True, cbv=cbv))
        (directory / 'program').write_bytes(code)
        (directory / 'data').write_bytes(DATA)
        (directory / 'state').write_bytes(state_a(GROUP, RSRC2))
        mode = 'dispatch-heap-dynamic-cbv' if cbv else 'dispatch-heap-dynamic'
        got, result = dispatch(directory, consumer, env, mode)
        assert got == [SENTINEL] * 320
        published, = lines(result, 'x2w-draw compute request')
        key = int(published.split()[3], 16)
        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True,
                             capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in out, out
        check_dispatch(directory, consumer, env, mode, BUFFER_WORD)
        check_dispatch(directory, consumer, env, mode + '-outside', 0)
        # Both select slot 2. Rejecting the entire heap breaks the unrelated-slot case, and
        # mirroring the flagged slot as a linear view gives a nonzero word in the selected case.
        check_dispatch(directory, consumer, env, mode + '-unrelated-swizzled', BUFFER_WORD, swizzled=True)
        check_dispatch(directory, consumer, env, mode + '-selected-swizzled', 0, swizzled=True)

        cso = cache / f'{key:016x}.cs.cso'
        wrong_word = mutate(source, f'r0.Load({address})', f'r0.Load({wrong_address})')
        cso.write_bytes(compile_shader(dxc, directory, 'cs_6_6', wrong_word).read_bytes())
        check_dispatch(directory, consumer, env, mode, BUFFER_WORD, mismatch=0x42aa6000)
        unbounded = mutate(source, 'index >= heap_count', 'index > heap_count + 1u')  # the tables start after the format view
        cso.write_bytes(compile_shader(dxc, directory, 'cs_6_6', unbounded).read_bytes())
        check_dispatch(directory, consumer, env, mode + '-outside', 0, mismatch=3)


if __name__ == '__main__':
    for code, binding, address, wrong_address, cbv in CASES:
        if len(sys.argv) == 1:
            translate(code, binding, address, cbv)
        else:
            gpu_case(*sys.argv[1:], code, binding, address, wrong_address, cbv)
    print('dynamic compute heap: all three index sources matched the reference and controls' if len(sys.argv) > 1
          else 'dynamic compute heap: all three index sources translated')
