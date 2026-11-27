"""Offline slot-57 serial fallback, refusal and artifact compatibility checks."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from startup_mesh57_loop_check import DATA_DEP
from startup_mesh57_skeleton_check import G, GROUP_MESH, HEAP_MESH, HEAP_SETUP, MESH, PIXEL, heap_root, request, refusal_head


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


def check_exec_zero_symbols():
    # The table descriptor reaches the join from the active arm only.
    load = ['f4081909', 'fa000000']
    use = ['e0301000', '80190600']  # buffer_load_dword v6, v0, s[100:103], 0 offen
    prefix, suffix = SERIAL_MESH[:8], use + SERIAL_MESH[9:]
    status, _ = run(prefix + ['bf880002'] + load + suffix, 'force')
    assert status == 'ok', status
    # An ordinary scalar branch provides no EXEC-zero guarantee.
    status, _ = run(prefix + ['bf840002'] + load + suffix, 'force')
    assert 's100 may be read before it is written' in status, status


def check_scalar_uniformity():
    # A scalar copied from m0 still has one value per guest wave.
    program = ['befc0380', 'be98037c', 'f408070d', '30000000']
    status, _ = run(SERIAL_MESH[:6] + HEAP_SETUP + program + SERIAL_MESH[6:],
                    'force', root_blob=heap_root())
    assert status == 'ok', status


def check_vcc_descriptor_alias():
    load = ['f4081a09', 'fa000000']  # s_load_dwordx4 s[104:107], s[18:19], 0
    use = ['e0301000', '801a0600']  # buffer_load_dword v6, v0, s[104:107], 0 offen
    prefix, suffix = SERIAL_MESH[:8] + load, use + SERIAL_MESH[9:]
    status, _ = run(prefix + suffix, 'force')
    assert status == 'ok', status
    for overwrite in ('7d840080', 'beea0481', 'beeb0380'):  # v_cmp_eq; s_mov_b64 vcc; s_mov_b32 vcc_hi
        status, _ = run(prefix + [overwrite] + suffix, 'force')
        assert 's[104:107] needs four consecutive words' in status, status


def check_mesh_heap_image_record():
    # Replace the fixture's dynamic buffer read with an image record and a 2D load.
    at = HEAP_MESH.index('f408070d')
    image = ['f40c080d', '30000000',  # s_load_dwordx8 s[32:39], s[26:27], s24
             'f0000108', '00081414']  # image_load v20, v[20:21], s[32:39]
    program = HEAP_MESH[:at] + image + HEAP_MESH[at + 4:]
    for mode in ('0', 'force'):
        status, output = run(program, mode, root_blob=heap_root(), artifacts=True)
        assert status == 'ok', status
        assert b'dynamic_heap' in output['ms57.txt'].splitlines()


def check_stub_cascade():
    import mesh_waveloop

    program = SERIAL_MESH[:9] + [
        'be941f00',  # s_getpc_b64 s[20:21]
        'be9e0314',  # s_mov_b32 s30, s20
    ] + SERIAL_MESH[9:]
    status, _ = run(program, 'force')
    assert 'unsupported s_getpc_b64' in status, status
    render = mesh_waveloop.render
    errors = {}

    def capture(pipe, gfx):
        errors.update({pipe.vs.real[i][2]: str(error) for i, error in pipe.vs.errors.items()})
        return render(pipe, gfx)

    with patch.object(mesh_waveloop, 'render', side_effect=capture):
        status, _ = run(program, 'force', stubs=True)
    assert status == 'ok', status
    assert 'unsupported s_getpc_b64' in errors['s_getpc_b64'], errors
    assert 's20 holds a stubbed instruction result' in errors['s_mov_b32'], errors


def check_carry_masks_use_partial_exec():
    carry = [
        'beea04c1',              # s_mov_b64 vcc, -1
        'be9e04c1',              # s_mov_b64 s[30:31], -1
        '50040081',              # v_add_co_ci_u32_e32 v2, vcc_lo, 1, v0, vcc_lo
        'd7101e03', '00020080',  # v_sub_co_u32 v3, s30, 0, v0
    ]
    calls = []
    carry_out, carry_in_out = G.Stage.carry_out_op, G.Stage.carry_in_out_op

    def track(op):
        def lower(stage, operands, modifiers, kind):
            calls.append((operands[1], stage.exec_known))
            return op(stage, operands, modifiers, kind)
        return lower

    with patch.object(G.Stage, 'carry_out_op', track(carry_out)), \
            patch.object(G.Stage, 'carry_in_out_op', track(carry_in_out)):
        status, _ = run(SERIAL_MESH[:8] + carry + SERIAL_MESH[9:], 'force')
    assert status == 'ok', status
    # The shared emitters use EXEC to clear inactive bits only in this mode.
    assert calls and set(calls) == {('vcc_lo', False), ('s30', False)}, calls


def check_level_zero_mesh_gather():
    program = [
        'f40c0809', 'fa000000',  # s_load_dwordx8 s[32:39], s[18:19], 0
        'bea80392', 'bea90380', 'beaa03ff', '04500000', 'beab0380',  # static sampler s[40:43]
        '7e020280',              # v_mov_b32 v1, 0
        'f11c0108', '01480600',  # image_gather4_lz v[6:9], v[0:1], s[32:39], s[40:43]
    ]
    status, output = run(SERIAL_MESH[:8] + program + SERIAL_MESH[9:], 'force', artifacts=True)
    assert status == 'ok', status
    manifest = output['ms57.txt'].splitlines()
    assert b'srv 0 tex2d table 1 0' in manifest and b'smp 0 0x12 0x0 0x4500000 0x0' in manifest


def check_export_rewrite(dxc):
    # Model an export expression using a temporary whose name ends with "o".
    original = G.MeshPipeline.mesh_output_line

    def with_temporary(pipe, line):
        return 'float4 foo = 0; ' + original(pipe, line) + ' o.pos.x += foo.x;'

    with patch.object(G.MeshPipeline, 'mesh_output_line', with_temporary):
        status, output = run(SERIAL_MESH, 'force', artifacts=True)
    assert status == 'ok', status
    with tempfile.TemporaryDirectory() as tmp:
        shader = Path(tmp) / 'export.hlsl'
        shader.write_bytes(output['ms57.hlsl'])
        subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', str(shader.with_suffix('.cso')),
                        str(shader)], check=True)


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
    check_exec_zero_symbols()
    check_scalar_uniformity()
    check_vcc_descriptor_alias()
    check_mesh_heap_image_record()
    check_stub_cascade()
    check_carry_masks_use_partial_exec()
    check_level_zero_mesh_gather()
    if len(sys.argv) == 2:
        check_export_rewrite(sys.argv[1])
    elif len(sys.argv) > 2:
        raise SystemExit('usage: startup_mesh57_waveloop_check.py [dxc]')
    print('ok')


if __name__ == '__main__':
    main()
