"""Check the DXIL shader flag for more than 8 UAV slots in direct compute.

DXC sets flag 0x8000 when a module declares more than 8 UAV slots, and the validator
refuses a module whose declared flags differ ("Flags must match usage"). Each program
stores to N buffer descriptors read from one table, so the module declares N UAVs.
8 UAVs must compile without the flag (the negative control); 9 must compile with it.
"""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from native_direct_container_check import compute, words


def program(uavs):
    code = [0xBEFE03C1, 0xBE810380]  # s_mov_b32 exec_lo, -1; s_mov_b32 s1, 0
    for k in range(uavs):
        # s_load_dwordx4 s[4+4k:7+4k], s[0:1], 32*k (table entries are 32-byte aligned)
        code += [0xF4080000 | (4 + 4 * k) << 6, 0xFA000000 | 32 * k]
    code += [
        0xBF8CC07F,                      # s_waitcnt lgkmcnt(0)
        (26 << 25) | (5 << 17) | 130,    # v_lshlrev_b32 v5, 2, v0
    ]
    for k in range(uavs):
        # buffer_store_dword v0, v5, s[4+4k:7+4k], 0 offen
        code += [0xE0701000, 0x80000005 | (1 + k) << 16]
    code.append(0xBF810000)              # s_endpgm
    return compute(words(*code), table=True)


def main(cli, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        for uavs in (8, 9):
            source = work / f'uav{uavs}.c67'
            source.write_bytes(program(uavs))
            shader = source.with_suffix('.cso')
            run = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-cs', source, shader],
                                 capture_output=True, text=True, timeout=60)
            declared = len(re.findall(r'^u\d+ ', run.stdout, re.M))
            assert run.returncode == 0 and shader.is_file(), (uavs, run.stdout, run.stderr)
            assert declared == uavs, (uavs, declared, run.stdout)
            print(f'{uavs} UAVs: compiled and validated')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('usage: native_uav_slots_check.py COMPILE_DXIL DXC_BIN_DIR')
    main(*sys.argv[1:])
