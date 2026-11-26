"""Check whole-word vector data refusals for the startup translator's lane masks."""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402


class Pipe:
    sampler_heap = False
    root = {'params': [], 'flags': 0}


COMPARE = 'v_cmp_eq_u32_e64 s[30:31], 0, v0'
SDWA = 'dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD'


def stage():
    s = G.Stage(Pipe(), 'ps', struct.pack('<I', 0xbf810000))
    s.lines = []
    s.pc = 0
    s.one(COMPARE.split()[0], COMPARE)
    return s


def main():
    refused = (
        ('v_cndmask_b32_e64 v1, 0, s30, s[30:31]', 's30'),
        ('v_cndmask_b32_e64 v1, -s30, v0, s[30:31]', '-s30'),
        ('v_cndmask_b32_e64 v1, 0, |s30|, s[30:31]', '|s30|'),
        ('v_cndmask_b32_e64 v1, neg(abs(s30)), v0, s[30:31]', 'neg(abs(s30))'),
        ('v_cndmask_b32_e32 v1, vcc_lo, v0, vcc_lo', 'vcc_lo'),
        ('v_mov_b32 v1, s30', 's30'),
        ('v_mov_b32_e32 v1, exec_lo', 'exec_lo'),
        ('v_cmp_o_f16_e64 s20, exec_lo, v4', 'exec_lo'),
        ('v_cmp_eq_f16_e64 s20, v4, vcc_lo', 'vcc_lo'),
        ('v_cmp_lt_f16_e64 s20, -s30, v4', '-s30'),
        ('v_mov_b32_e64 v1, vcc_lo', 'vcc_lo'),
        ('v_mov_b32_sdwa v1, s30 ' + SDWA, 's30'),
        ('v_mov_b32_sdwa v1, sext(s30) ' + SDWA, 'sext(s30)'),
    )
    for asm, source in refused:
        s = stage()
        name = asm.split()[0]
        op = name.removesuffix('_e32').removesuffix('_e64').removesuffix('_sdwa')
        try:
            s.one(name, asm)
        except G.Unsupported as error:
            assert str(error) == f'{op} data operand {source} is a lane mask', (asm, error)
        else:
            raise AssertionError(f'Accepted a lane mask as data: {asm}')

    accepted = (
        ('v_cndmask_b32_e64 v1, 0, v0, s[30:31]', 'v1 = (s30 & 1) ? v0 : 0x00000000u;'),
        ('v_cndmask_b32_e32 v1, s29, v0, vcc_lo', 'v1 = (vcc & 1) ? v0 : 0x12345678u;'),
        ('v_mov_b32_e64 v1, s29', 'v1 = 0x12345678u;'),
    )
    for asm, line in accepted:
        s = stage()
        s.one('s_mov_b32', 's_mov_b32 s29, 0x12345678')
        s.one(asm.split()[0], asm)
        assert s.lines[-1] == line, (asm, s.lines)
    print(f'mask data: {len(refused)} refusals and {len(accepted)} accepted controls')


if __name__ == '__main__':
    main()
