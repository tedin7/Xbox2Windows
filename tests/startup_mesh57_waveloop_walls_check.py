"""Offline slot 57 checks for descriptor arithmetic and serial control flow."""

from startup_mesh57_waveloop_check import SERIAL_MESH, run


def check_descriptor_shift_add():
    program = SERIAL_MESH[:8] + [
        'f4081909', 'fa000000',  # s_load_dwordx4 s[100:103], s[18:19], 0
        '97948166',              # s_lshl2_add_u32 s20, s102, 1
        '7e000214',              # v_mov_b32 v0, s20
    ] + SERIAL_MESH[8:]
    status, output = run(program, 'force', artifacts=True)
    assert status == 'ok', status
    manifest = output['ms57.txt'].decode().splitlines()
    assert manifest[0] == 'x2w-manifest 3', manifest
    word, = [line.split()[1] for line in manifest if line.endswith(' descword 1 8')]
    hlsl = output['ms57.hlsl'].decode()
    assert 'scc = (ud%s >> 30) != 0 ||' % word in hlsl, hlsl
    assert '(ud%s << 2) + 0x00000001u' % word in hlsl, hlsl


def check_descriptor_compare():
    program = SERIAL_MESH[:8] + [
        'f4081909', 'fa000000',  # s_load_dwordx4 s[100:103], s[18:19], 0
        'bf008066',              # s_cmp_eq_i32 s102, 0
        'bf850001',              # s_cbranch_scc1 over the second write to v0
        '7e000281',              # v_mov_b32 v0, 1
    ] + SERIAL_MESH[8:]
    status, output = run(program, 'force', artifacts=True)
    assert status == 'ok', status
    manifest = output['ms57.txt'].decode().splitlines()
    word, = [line.split()[1] for line in manifest if line.endswith(' descword 1 8')]
    hlsl = output['ms57.hlsl'].decode()
    assert 'scc = int(ud%s) == int(0x00000000u);' % word in hlsl, hlsl
    assert 'block_pc = (scc) ?' in hlsl, hlsl


def check_exec_zero_non_rejoining_exit():
    prefix = SERIAL_MESH[:8]
    suffix = ['7e020214'] + SERIAL_MESH[8:]  # v_mov_b32 v1, s20
    write = 'be940380'  # s_mov_b32 s20, 0
    for skip in (['bf880002'], ['bf890001', 'bf820002']):
        # EXEC zero skips the write and lands at the use. The active arm
        # either branches there as well or terminates at its own s_endpgm.
        status, _ = run(prefix + skip + [write, 'bf820000'] + suffix, 'force')
        assert status == 'ok', status
        status, _ = run(prefix + skip + [write, 'bf810000'] + suffix, 'force')
        assert 's20 may be read before it is written' in status, status


def check_descriptor_pair_moves_through_vcc():
    prefix = SERIAL_MESH[:8] + [
        'f4081909', 'fa000000',  # s_load_dwordx4 s[100:103], s[18:19], 0
        'bee80464',              # s_mov_b64 s[104:105], s[100:101]
        'beea0466',              # s_mov_b64 vcc, s[102:103]
    ]
    for move, descriptor in [
        ([], '801a0600'),  # use s[104:107] directly
        (['bee60480', 'bee6046a', 'beea0480'], '80190600'),  # restore s[102:103] from VCC
    ]:
        program = prefix + move + ['e0301000', descriptor] + SERIAL_MESH[8:]
        status, output = run(program, 'force', artifacts=True)
        assert status == 'ok', status
        assert b'descword' not in output['ms57.txt'], output['ms57.txt']
    status, _ = run(prefix + ['bf860000'] + SERIAL_MESH[8:], 'force')  # s_cbranch_vccz
    assert 'mask word s106 holds desc' in status, status


def check_runtime_scalar_join():
    prefix = SERIAL_MESH[:8] + [
        'f4081909', 'fa000000',  # s_load_dwordx4 s[100:103], s[18:19], 0
        'bf850003',              # s_cbranch_scc1 to the other assignment
        'f4200532', 'fa000000',  # s_buffer_load_dword s20, s[100:103], 0
        'bf820001',              # s_branch to the use
    ]
    suffix = ['7e000214'] + SERIAL_MESH[8:]  # v_mov_b32 v0, s20
    status, output = run(prefix + ['be940381'] + suffix, 'force', artifacts=True)  # s_mov_b32 s20, 1
    assert status == 'ok', status
    hlsl = output['ms57.hlsl'].decode()
    assert any('s20 = ' in line and '.Load(' in line for line in hlsl.splitlines()), hlsl
    assert 's20 = 0x00000001u;' in hlsl, hlsl
    assert 'v0[((wb + l) & 127u)] = s20;' in hlsl, hlsl
    status, _ = run(prefix + ['be940366'] + suffix, 'force')  # s_mov_b32 s20, s102
    assert 'numeric use of s20 holding a value joined across wave-loop blocks' in status, status


def check_sdwa_float_source_modifiers():
    prefix, suffix = SERIAL_MESH[:8], SERIAL_MESH[8:]
    for control, negative in [('00261500', False), ('00361500', True)]:
        # v_cvt_f16_f32_sdwa v0, |v0| or -|v0|, writing the high half.
        status, output = run(prefix + ['7e0014f9', control] + suffix, 'force', artifacts=True)
        assert status == 'ok', status
        value = '(v0[((wb + l) & 127u)] & 0x7fffffffu)'
        if negative:
            value = '(%s ^ 0x80000000u)' % value
        line, = [line for line in output['ms57.hlsl'].decode().splitlines() if 'f32tof16(F(' in line]
        assert 'f32tof16(F(%s))' % value in line, line
        assert 'asuint(v0[((wb + l) & 127u)]) & 0xffffu' in line, line
    status, _ = run(prefix + ['7e0014f9', '00241500'] + suffix, 'force')  # src0_sel:WORD_0
    assert "operand '|v0|'" in status, status


if __name__ == '__main__':
    check_descriptor_shift_add()
    check_descriptor_compare()
    check_exec_zero_non_rejoining_exit()
    check_descriptor_pair_moves_through_vcc()
    check_runtime_scalar_join()
    check_sdwa_float_source_modifiers()
