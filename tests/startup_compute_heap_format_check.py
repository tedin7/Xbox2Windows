"""Compute translation and WARP dispatch for dynamic heap formatted views."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_compute_dynamic_heap_check import CS, G, GROUP, RSRC2, SENTINEL, root_blob
from startup_compute_dispatch_check import dispatch, lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_gfx2hlsl_resheap_branch_check import mubuf
from startup_formatted_view_draw_check import cpu_decode, pack_element


load = bytearray(mubuf(3, 5, 4, 16))
load[1] = (load[1] & ~0x10) | 0x20
CODE = CS.replace(bytes.fromhex('8002087e'), bytes.fromhex('8102087e'))
CODE = CODE.replace(bytes.fromhex('001030e004050480'), bytes(load))
CODE = CODE.replace(bytes.fromhex('001070e004050580'), bytes.fromhex('8002087e') + mubuf(0x1e, 5, 4, 20))


def translate():
    pipe = G.ComputePipeline(CODE, root_blob(compute=True), RSRC2, GROUP)
    pipe.hlsl_cs()
    assert 'dynamic_heap\n' in pipe.manifest()
    assert 'dynamic_heap_formats\n' in pipe.manifest()


def warp(consumer, dxc):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        cache, env = prepare(directory)
        env.pop('X2W_FXC', None)
        env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1', X2W_TEST_COMPUTE_ROOT=str(directory / 'compute_root'))
        (directory / 'compute_root').write_bytes(root_blob(compute=True))
        (directory / 'program').write_bytes(CODE)
        (directory / 'data').write_bytes(DATA)
        (directory / 'state').write_bytes(state_a(GROUP, RSRC2))
        mode = 'dispatch-heap-dynamic-format'
        got, result = dispatch(directory, consumer, env, mode)
        assert got == [SENTINEL] * 320
        published, = lines(result, 'x2w-draw compute request')
        key = int(published.split()[3], 16)
        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True,
                             capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in out, out
        for mode, widths, fields, order in (
            ('dispatch-heap-dynamic-format', [16] * 4, [0x3c00, 0x4000, 0x4200, 0x4400], (3, 0, 2, 1)),
            ('dispatch-heap-dynamic-format-view', [16, 16], [0x3c00, 0x4000], (0, 1, 2, 3)),
        ):
            decoded = cpu_decode(pack_element(fields, widths), widths, 'FLOAT')
            expected = [decoded[i] for i in order] + [SENTINEL] * 316
            got, result = dispatch(directory, consumer, env, mode)
            assert len(lines(result, 'Dispatch(1, 1, 1) translated')) == 1, result.stderr[-3000:]
            assert not lines(result, 'refused'), result.stderr[-3000:]
            assert got == expected, (mode, got[:4], expected[:4])
    print('compute dynamic heap formats: WARP readback matched')


if __name__ == '__main__':
    translate()
    if len(sys.argv) > 1:
        warp(*sys.argv[1:])
    else:
        print('compute dynamic heap formats: translation controls matched')
