"""Offline slot-57 serial fallback, refusal and artifact compatibility checks."""

import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from startup_mesh57_loop_check import DATA_DEP
from startup_mesh57_skeleton_check import G, GROUP_MESH, HEAP_SETUP, MESH, PIXEL, heap_root, request, refusal_head


# Separate allocation block for wave 0, followed by a partial-EXEC lane read.
# The fixture has no proof-path exchanges and uses no local shader assets.
SERIAL_MESH = (
    '8706ff03 0f000000'  # s_and_b32 s6, s3, 0xf000000
    ' bf850003'          # s_cbranch_scc1 over allocation
    ' befc03ff 00001003'  # s_mov_b32 m0, 0x1003
    ' bf900009'           # s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
    ' befe0481'           # s_mov_b64 exec, 1
    ' 7e000280'           # v_mov_b32 v0, 0
    ' 7e280500'           # v_readfirstlane_b32 s20, v0
    ' f80008cf 00000000'  # exp pos0 v0, v0, v0, v0
    ' f8000941 00000000'  # exp target_20 v0, off, off, off
    ' bf810000'           # s_endpgm
).split()


def run(mesh, mode=None, pixel=(), *, artifacts=False, root_blob=None, stubs=False):
    with patch.dict(os.environ, {'X2W_FXC': ''}):
        for variable in ('X2W_MESH_WAVELOOP', 'X2W_MESH_WAVELOOP_STUBS', 'X2W_PLACEHOLDER_PS'):
            os.environ.pop(variable, None)
        if stubs:
            os.environ['X2W_MESH_WAVELOOP_STUBS'] = '1'
        if mode is not None:
            os.environ['X2W_MESH_WAVELOOP'] = mode
        with tempfile.TemporaryDirectory() as tmp:
            try:
                key, result = request(tmp, mesh, pixel, root_blob=root_blob)
            except G.Unsupported as reason:
                return refusal_head(str(reason)), {}
            output = {}
            if artifacts:
                for path in Path(tmp).glob('%016x.*' % key):
                    output[path.name.split('.', 1)[1]] = path.read_bytes()
            return result, output


def check_exec_zero_edges():
    # On the dead edge v0 and s20 retain their old storage. Only the active arm defines them.
    prefix = SERIAL_MESH[:6]
    writes = ['7e000280', 'be940380']  # v_mov_b32 v0, 0; s_mov_b32 s20, 0
    suffix = ['7e020214', *SERIAL_MESH[8:]]  # v_mov_b32 v1, s20; export v0
    for branch in (['bf880002', *writes], ['bf890001', 'bf820002', *writes]):
        status, _ = run(prefix + branch + suffix, 'force')
        assert status == 'ok', status
    status, _ = run(prefix + ['bf880001', writes[1]] + suffix, 'force')
    assert 'v0 may be read before it is written' in status, status
    status, _ = run(prefix + ['bf880001', writes[0]] + suffix, 'force')
    assert 's20 may be read before it is written' in status, status


def check_scalar_uniformity():
    # A scalar copied from m0 still has one value per guest wave.
    program = ['befc0380', 'be98037c', 'f408070d', '30000000']
    status, _ = run(SERIAL_MESH[:6] + HEAP_SETUP + program + SERIAL_MESH[6:],
                    'force', root_blob=heap_root())
    assert status == 'ok', status


def main():
    refused, _ = run(SERIAL_MESH, '0')
    assert refused.startswith('unsupported: slot 57 mesh'), refused
    translated, output = run(SERIAL_MESH, artifacts=True)
    assert translated == 'ok', translated
    assert output['ms57.hlsl'].startswith(b'// x2w_serial_wave 1\n')
    assert set(output) == {'ms57.hlsl', 'ms57.txt'}, output.keys()

    # The existing fixture carries a scalar counter around a data-dependent back edge.
    refused, _ = run(DATA_DEP, '0', PIXEL)
    assert refused == 'unsupported: ' + G.MESH_LOOP_REFUSAL, refused
    translated, output = run(DATA_DEP, '1', PIXEL, artifacts=True)
    assert translated == 'ok', translated
    assert output['ms57.hlsl'].startswith(b'// x2w_serial_wave 1\n')
    assert set(output) == {'ms57.hlsl', 'ps57.hlsl', 'ms57.txt'}, output.keys()

    # Break the descriptor base of the proof fixture while preserving the exchanges.
    broken = ['f408020f' if word == 'f4080209' else word for word in MESH]
    disabled, _ = run(broken, '0')
    enabled, _ = run(broken)
    assert 's_load through' in disabled and enabled == disabled, (disabled, enabled)

    for mesh in (MESH, GROUP_MESH):
        old, before = run(mesh, '0', PIXEL, artifacts=True)
        new, after = run(mesh, None, PIXEL, artifacts=True)
        assert old == new == 'ok' and before == after, (old, new)
    status, forced = run(MESH, 'force', PIXEL, artifacts=True)
    assert status == 'ok', status
    assert forced['ms57.hlsl'].startswith(b'// x2w_serial_wave 1\n')
    assert forced['ms57.txt'] == run(MESH, '0', PIXEL, artifacts=True)[1]['ms57.txt']

    undefined = [word for word in SERIAL_MESH if word != '7e000280']
    reason, _ = run(undefined)
    assert 'wave loop: v0 may be read before it is written' in reason, reason
    check_exec_zero_edges()
    check_scalar_uniformity()
    print('ok')


if __name__ == '__main__':
    main()
