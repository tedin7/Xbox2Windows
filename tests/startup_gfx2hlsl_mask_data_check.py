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


MASK_WORDS = bytes.fromhex(
    '800387be '  # s_mov_b32 s7, 0
    '030308f4000000fa '  # s_load_dwordx4 s[12:15], s[6:7], 0
    '030608f4100000fa '  # s_load_dwordx4 s[24:27], s[6:7], 16
    '7fc08cbf '  # s_waitcnt lgkmcnt(0)
    '0e03eabe '  # s_mov_b32 vcc_lo, s14
    '9702147e '  # v_mov_b32 v10, 23
    '91141202 '  # v_cndmask_b32_e32 v9, 17, v10, vcc_lo
    '82000c34 '  # v_lshlrev_b32 v6, 2, v0
    '001070e006090680 '  # buffer_store_dword v9, v6, s[24:27], 0 offen
    '000081bf '  # s_endpgm
)


def main():
    from startup_gfx2hlsl_compute_check import root_blob

    pipe = G.ComputePipeline(MASK_WORDS, root_blob(), 7 << 1, (8, 1, 1))
    assert 'udsrc 2 desc_word table 1 0 2' in pipe.manifest()
    assert 'ud2 >> (tid.x' in pipe.hlsl_cs()

    refused = (
        ('v_cndmask_b32_e64 v1, 0, s30, s[30:31]', 's30'),
        ('v_cndmask_b32_e64 v1, -s30, v0, s[30:31]', '-s30'),
        ('v_cndmask_b32_e64 v1, 0, |s30|, s[30:31]', '|s30|'),
        ('v_cndmask_b32_e64 v1, neg(abs(s30)), v0, s[30:31]', 'neg(abs(s30))'),
        ('v_cndmask_b32_e32 v1, vcc_lo, v0, vcc_lo', 'vcc_lo'),
        ('v_mov_b32 v1, s30', 's30'),
        ('v_mov_b32_e32 v1, exec_lo', 'exec_lo'),
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
