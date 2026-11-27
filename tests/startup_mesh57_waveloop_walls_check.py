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


if __name__ == '__main__':
    check_descriptor_shift_add()
    check_exec_zero_non_rejoining_exit()
