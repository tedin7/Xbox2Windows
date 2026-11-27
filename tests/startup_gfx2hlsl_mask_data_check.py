"""Lane-mask data controls and compute shift-and-add WARP readback."""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402
from startup_compute_tg_size_check import sop1, sopp, vop1, vop2, vop3  # noqa: E402
from startup_gfx2hlsl_compute_check import KEY, SENTINEL, STRIDE, USER7, WORDS, root_blob  # noqa: E402
from startup_gfx2hlsl_s_lshl_add_check import NAMES, reference, sop2  # noqa: E402
from startup_image_3d_check import compile_hlsl  # noqa: E402


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


GROUP = (32, 1, 1)
MASK_REG, ADDEND = 22, 7


def compute_program(n, selected_lane, mask_update=b''):
    return b''.join((
        sop1(3, 10, 6),  # user-data table pointer, low word
        sop1(3, 11, 128),
        struct.pack('<II', 0xf40c0305, 0xfa000000),  # s_load_dwordx8 s[12:19], s[10:11], 0
        vop3(0xc2, MASK_REG, 128 + selected_lane, 256),  # v_cmp_eq_u32 s22, selected_lane, v0
        mask_update,
        sop2(NAMES[n], MASK_REG, MASK_REG, 128 + ADDEND),
        sop2('s_cselect_b32', 24, 129, 128),  # snapshot carry
        vop1(1, 8, MASK_REG),  # must read the result as a full numeric word
        vop1(1, 9, 24),
        vop2(0x25, 10, MASK_REG, 0),  # add the per-thread input v0 to the scalar sum
        vop1(1, 11, 256),
        vop2(0x1a, 4, 132, 0),  # lane * 16, the byte offset for a four-word record
        sopp(0x0c, 0xc07f),  # wait for the descriptor
        struct.pack('<II', 0xe0781000, 0x80040804),  # buffer_store_dwordx4 v[8:11], v4, s[16:19], 0 offen
        sopp(1),
    ))


def compute_source(n, lane, group=GROUP, mask_update=b''):
    return G.ComputePipeline(compute_program(n, lane, mask_update), root_blob(), USER7, group).hlsl_cs()


def expected_compute(n, selected_lane):
    wave_word = sum(1 << lane for lane in range(GROUP[0]) if lane == selected_lane)
    value, carry = reference(n, wave_word, ADDEND)
    out = [SENTINEL] * WORDS
    for lane in range(GROUP[0]):
        out[4 * lane:4 * lane + 4] = (value, carry, (value + lane) & 0xffffffff, lane)
    return out


def refuse(reason, run):
    try:
        run()
    except G.Unsupported as error:
        assert str(error) == reason, (reason, error)
    else:
        raise AssertionError(f'Expected refusal: {reason}')


def shift_add_controls():
    propagate = G.Stage.propagate_mask_tags

    def leave_result_tagged(self, base, ops, tags):
        propagate(self, base, ops, tags)
        if base in NAMES.values():
            self.masks.add(MASK_REG)

    for n, name in NAMES.items():
        compute_source(n, 5)
        unshifted_carry(compute_source(n, 31))
        assert expected_compute(n, 5)[1] == 0 and expected_compute(n, 31)[1] == 1
        with patch.object(G.Stage, 'propagate_mask_tags', leave_result_tagged):
            refuse('v_mov_b32 data operand s22 is a lane mask', lambda: compute_source(n, 31))
        reason = f'{name} on a lane mask: wave word not available in this stage'
        refuse(reason, lambda: compute_source(n, 31, (64, 1, 1)))
        mixed = sop2('s_or_b32', MASK_REG, MASK_REG, 130)  # bit 1 set by a numeric literal
        refuse(reason, lambda: compute_source(n, 31, mask_update=mixed))
        ps = stage()
        asm = f'{name} s30, s30, 1'
        refuse(reason, lambda: ps.one(name, asm))

    # Keep the existing per-lane complement forms and their SCC behavior byte-for-byte.
    for name, operator in (('s_nand_b32', '&'), ('s_nor_b32', '|'), ('s_xnor_b32', '^')):
        ps = stage()
        ps.lines.clear()
        ps.one(name, f'{name} s34, s30, s30')
        assert ps.lines == [f's34 = (~(s30 {operator} s30)) & 1;'], (name, ps.lines)
        assert 34 in ps.masks and not ps.scc_verified
    print('shift-add mask controls: refusals, stale destination tag mutant, and unchanged complements')


def unshifted_carry(source):
    text, replaced = re.subn(r'^( +scc = ).*(shift_mask_\w+) >> .*;$',
                             rf'\g<1>(\2 + {ADDEND}u) < {ADDEND}u;', source, flags=re.M)
    assert replaced == 1, source
    return text


def dispatch(dxc, renderer):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        case = directory / 'shift_add.case'
        for n in NAMES:
            for lane in (5, 31):
                text = compute_source(n, lane)
                case.write_bytes(struct.pack(f'<{1 + WORDS + 16}I', 0x010101, *expected_compute(n, lane),
                                             KEY, STRIDE, *([0] * 14)))
                modes = (False, True) if lane == 31 else (False,)
                for mutant in modes:
                    source = unshifted_carry(text) if mutant else text
                    cso = compile_hlsl(dxc, 'dxc', source, 'shift_add', 'cs_6_0', directory)
                    proc = subprocess.run([renderer, 'warp', '--compute-store-case', str(cso), str(case)],
                                          capture_output=True, text=True)
                    print(proc.stdout, end='')
                    if proc.returncode == 77:
                        raise SystemExit(77)
                    assert 'debug_layer=enabled' not in proc.stdout or 'debug_errors=0 debug_warnings=0' in proc.stdout
                    if mutant:
                        assert proc.returncode == 1 and 'word=1 expected=00000001 actual=00000000' in proc.stderr, proc.stderr
                    else:
                        assert proc.returncode == 0, proc.stdout + proc.stderr
                    print(f'compute shift {n}, compare lane {lane}, mutant={mutant}: expected result')


if __name__ == '__main__':
    main()
    shift_add_controls()
    if len(sys.argv) > 1:
        mode, *args = sys.argv[1:]
        if mode != 'dispatch' or len(args) != 2:
            raise SystemExit('usage: startup_gfx2hlsl_mask_data_check.py [dispatch <dxc> <offscreen_check>]')
        dispatch(*args)
