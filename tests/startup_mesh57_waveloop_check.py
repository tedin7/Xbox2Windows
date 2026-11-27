"""Slot-57 serial fallback checks and D3D12 readback against CPU references."""

import os
import re
import struct
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from startup_mesh57_loop_check import DATA_DEP, rasterise as loop_rasterise
from startup_mesh57_skeleton_check import (
    G, GROUP_MESH, HEAP_MESH, HEAP_SETUP, MESH, PIXEL, SIDE, assemble, buffer, code,
    geometry, heap_root, rasterise, refusal_head, request, root,
)


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


# Keep the two LDS exchanges and the global thread IDs from the skeleton.
# After the vertex count comparison only lanes 32..43 may export positions.
# Read their first active thread and use its lane number to choose vertex red.
UPPER_EXEC_ASM = """s_mov_b32 exec_lo, 0
s_and_b32 exec_hi, exec_hi, 0xfff
v_readfirstlane_b32 s25, v0
s_and_b32 s25, s25, 63
s_cmp_eq_i32 s25, 32
s_cselect_b32 s25, 1.0, 0
"""
_mask_at = MESH.index('7d280016') + 1
_red_at = MESH.index('f80008cf')
UPPER_EXEC_MESH = (
    MESH[:_mask_at] + 'befe0380 877fff7f 00000fff 7e320500 8719bf19 bf00a019 851980f2'.split()
    + MESH[_mask_at:_red_at] + ['7e280219'] + MESH[_red_at:]
)


def binary(mc):
    source = Path(__file__).with_name('startup_mesh57_skeleton.s').read_text()
    source = replace_once(source, 'v_cmpx_gt_i32 s22, v0\n', 'v_cmpx_gt_i32 s22, v0\n' + UPPER_EXEC_ASM)
    source = replace_once(source, 'exp pos0 ', 'v_mov_b32 v20, s25\nexp pos0 ')
    words = assemble(mc, source)
    assert words == UPPER_EXEC_MESH, words
    print('upper EXEC fixture matches llvm-mc')


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
    compile_mesh(dxc, output['ms57.hlsl'])


def check_vcc_numeric_alias_compiles(dxc):
    program = ['7d840080', 'bf00806a']  # v_cmp_eq_u32 into VCC, then s_cmp_eq_i32 reading its low word
    status, output = run(SERIAL_MESH[:8] + program + SERIAL_MESH[9:], 'force', artifacts=True)
    assert status == 'ok', status
    compile_mesh(dxc, output['ms57.hlsl'])


def compile_mesh(dxc, hlsl):
    with tempfile.TemporaryDirectory() as tmp:
        shader = Path(tmp) / 'mesh.hlsl'
        shader.write_bytes(hlsl)
        subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', str(shader.with_suffix('.cso')),
                        str(shader)], check=True)


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def check_upper_exec():
    from mesh_waveloop import proof_family

    status, _ = run(UPPER_EXEC_MESH, '0', PIXEL)
    assert status.startswith('unsupported: ') and proof_family(G, status.removeprefix('unsupported: ')), status
    status, output = run(UPPER_EXEC_MESH, '1', PIXEL, artifacts=True)
    assert status == 'ok', status
    source = output['ms57.hlsl']
    assert source.startswith(b'// x2w_serial_wave 1\n')
    assert source.count(b'SetMeshOutputCounts(') == 1
    return source


def upper_exec_case():
    points, colours, indices = geometry()
    # The second LDS exchange publishes these group counts to both waves.
    counts = (len(indices), 108)
    prims, verts = counts
    threads, red = [], []
    for wave in range(2):
        wave_threads = [wave * 64 + lane for lane in range(64)]
        exec_mask = sum(1 << lane for lane in range(32, 44) if wave_threads[lane] < verts)
        first_lane = (exec_mask & -exec_mask).bit_length() - 1
        threads.append([thread for lane, thread in enumerate(wave_threads) if exec_mask & (1 << lane)])
        red.append(255 if first_lane == 32 else 0)

    # Alternate wave ownership, preserving each triangle's original buffer slots.
    triangles = [tuple(threads[t % 2][slot] for slot in triangle) for t, triangle in enumerate(indices)]
    expected_colours = [(red[0], *colour[1:]) for colour in colours]
    assert red[0] == red[1] == 255
    image = rasterise(points, expected_colours, triangles, verts, prims)
    for wave in range(2):
        without_wave = [triangle if t % 2 != wave else (0, 0, 0) for t, triangle in enumerate(triangles)]
        assert image != rasterise(points, expected_colours, without_wave, verts, prims)
    assert image != rasterise(points, [(0, *c[1:]) for c in colours], triangles, verts, prims)
    return counts, triangles, image


def serial_shaders(directory, mesh, dxc, *, proof_accepted=False):
    mode = 'force' if proof_accepted else '1'
    if proof_accepted:
        status, _ = run(mesh, '0', PIXEL)
        assert status == 'ok', status
    elif mesh == DATA_DEP:
        status, _ = run(mesh, '0', PIXEL)
        assert status == 'unsupported: ' + G.MESH_LOOP_REFUSAL, status
    # Case B's proof-family refusal is checked by check_upper_exec.
    with patch.dict(os.environ, {'X2W_DXC': dxc, 'X2W_MESH_WAVELOOP': mode}):
        for key in ('X2W_FXC', 'X2W_MESH_WAVELOOP_STUBS', 'X2W_PLACEHOLDER_PS'):
            os.environ.pop(key, None)
        key, status = request(directory, mesh)
    assert status == 'ok', status
    base = directory / ('%016x' % key)
    source = base.with_suffix('.ms57.hlsl').read_text()
    assert source.startswith('// x2w_serial_wave 1\n')
    return base.with_suffix('.ms57.cso'), base.with_suffix('.ps57.cso'), source


def compile_variant(directory, dxc, label, source):
    shader = directory / (label + '.hlsl')
    bytecode = shader.with_suffix('.cso')
    shader.write_text(source)
    subprocess.run([dxc, '-Gis', '-T', 'ms_6_5', '-E', 'main', '-Fo', bytecode, shader], check=True)
    return bytecode


def draw_case(directory, renderer, adapter, label, shaders, counts, triangles, image, *,
              groups=1, mismatch=False):
    points, colours, _ = geometry()
    words = [a | b << 10 | c << 20 for a, b, c in triangles]
    # This root signature places the count constants at the same offsets for all cases.
    uds = G.MeshPipeline(code(MESH), code(PIXEL), root()).uds
    constants = bytearray(1024)
    for slot, count in enumerate(counts):
        struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, slot)), count)
    case = directory / (label + '.bin')
    case.write_bytes(bytes(constants) + buffer(points, colours, words) + b'\xcd' * 80
                     + b''.join(map(bytes, image)))

    command = [renderer, adapter, '--mesh-indexed' if groups == 1 else '--mesh-indexed-groups', *shaders, case]
    if groups != 1:
        command.append(str(groups))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 77:
        print(result.stdout + result.stderr)
        raise SystemExit(77)
    readback = re.search(rf'^pixels={SIDE * SIDE} mismatches=(\d+)$', result.stdout, re.MULTILINE)
    diagnostic = f'{label}: exit {result.returncode}\n{result.stdout}{result.stderr}'
    assert readback, diagnostic
    errors = int(readback[1])
    assert 0 <= errors <= SIDE * SIDE and bool(errors) == mismatch, diagnostic
    assert result.returncode == int(mismatch), diagnostic
    if mismatch:
        assert 'Draw reference mismatch' in result.stderr, diagnostic
    print(f'{label}: {readback[0]}')


def draw(dxc, renderer, adapter):
    check_upper_exec()
    points, colours, indices = geometry()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)

        # DATA_DEP uses s16 both as the trip bound and as the primitive count.
        trip_colours = [(255, *colour[1:]) for colour in colours]
        image = loop_rasterise(points, trip_colours, indices, 12, 4)
        ms, ps, source = serial_shaders(directory, DATA_DEP, dxc)
        draw_case(directory, renderer, adapter, 'loop-four', (ms, ps), (4, 12), indices, image)
        draw_case(directory, renderer, adapter, 'loop-three', (ms, ps), (3, 12), indices, image, mismatch=True)
        # The fixture writes zero to red when the equality fails.
        short_image = loop_rasterise(points, [(0, *colour[1:]) for colour in colours], indices, 12, 3)
        draw_case(directory, renderer, adapter, 'loop-three-reference', (ms, ps), (3, 12), indices, short_image)
        fuel = replace_once(source, 'fuel = 65536u;', 'fuel = 1u;')
        mutant = compile_variant(directory, dxc, 'loop-low-fuel', fuel)
        draw_case(directory, renderer, adapter, 'loop-low-fuel', (mutant, ps), (4, 12), indices, image, mismatch=True)

        counts, triangles, image = upper_exec_case()
        ms, ps, source = serial_shaders(directory, UPPER_EXEC_MESH, dxc)
        draw_case(directory, renderer, adapter, 'upper-exec', (ms, ps), counts, triangles, image)
        order = 'for (uint wave_id = 0u; wave_id < 2u && !error; ++wave_id) {'
        reverse = replace_once(source, order, 'for (uint turn = 0u; turn < 2u && !error; ++turn) {\n'
                                             '            uint wave_id = 1u - turn;')
        reversed_ms = compile_variant(directory, dxc, 'reverse-order', reverse)
        draw_case(directory, renderer, adapter, 'reverse-order', (reversed_ms, ps), counts, triangles, image)
        # With wave 1 first, losing its yield at the count barrier lets it read
        # LDS before wave 0 publishes. Reversing a correct schedule is harmless.
        bypass = replace_once(reverse, 'block_pc = 42u; yielded = true;', 'block_pc = 42u;')
        mutant = compile_variant(directory, dxc, 'early-count-read', bypass)
        draw_case(directory, renderer, adapter, 'early-count-read', (mutant, ps), counts, triangles, image,
                  mismatch=True)
        bad_mask = replace_once(source, '((l < 32u ? lo : hi) >> (l & 31u)) & 1u',
                                '(lo >> (l & 31u)) & 1u')
        mutant = compile_variant(directory, dxc, 'low-mask-only', bad_mask)
        draw_case(directory, renderer, adapter, 'low-mask-only', (mutant, ps), counts, triangles, image,
                  mismatch=True)

        image = rasterise(points, colours, indices, len(points), len(indices))
        for label, mesh, prims, groups in (('group-force', GROUP_MESH, 2, 2), ('mesh-force', MESH, 4, 1)):
            ms, ps, _ = serial_shaders(directory, mesh, dxc, proof_accepted=True)
            draw_case(directory, renderer, adapter, label, (ms, ps), (prims, len(points)), indices, image,
                      groups=groups)


def offline(dxc=None):
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
    source = check_upper_exec()
    upper_exec_case()
    if dxc:
        check_export_rewrite(dxc)
        check_vcc_numeric_alias_compiles(dxc)
        compile_mesh(dxc, source)
    print('ok')


def main():
    args = sys.argv[1:]
    if len(args) == 4 and args[0] == 'draw':
        draw(*args[1:])
    elif len(args) == 2 and args[0] == 'binary':
        binary(args[1])
    elif len(args) <= 1 and args not in (['draw'], ['binary']):
        offline(*args)
    else:
        raise SystemExit('usage: startup_mesh57_waveloop_check.py [dxc | binary <llvm-mc> | '
                         'draw <dxc> <renderer> <adapter>]')


if __name__ == '__main__':
    main()
