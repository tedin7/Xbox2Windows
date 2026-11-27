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


if __name__ == '__main__':
    check_descriptor_shift_add()
