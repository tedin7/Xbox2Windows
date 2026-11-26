#!/usr/bin/env python3
"""wave32 vcc_hi as an ordinary scalar register in a compute program of tools/startup/gfx2hlsl.py (sibling of
startup_gfx2hlsl_vcc_hi_check.py, which covers the pixel stage).

  (no arguments)                         translation controls and refusals, pure Python
  dispatch <dxc> <offscreen_check> <warp|hardware>
                                         translate, compile with DXC cs_6_0, dispatch and compare all 320 words
  fixture <early-out|vcc-pair> <dxc> <offscreen_check> <warp|hardware>
                                         dispatch synthetic fixtures and check predicted mutant readbacks
  control <vcc-hi-lost> <dxc> <offscreen_check> <warp|hardware>
                                         mutated translation must mismatch on readback at the word the reference predicts

In wave32 VCC is vcc_lo alone and compilers allocate vcc_hi (SGPR 107) like any other scalar register (AMD RDNA 2 ISA,
"VCC"). The decoder prints wave32 syntax, so the translator renames vcc_hi to an SGPR the program never names only when that
is exact. A compute program whose exec writes are all 32-bit forms (s_and_saveexec_b32, s_or_b32 exec_lo) is wave32, so the
rename holds even though a VALU compare writes vcc_lo, and a block skipped by s_cbranch_execz that holds such a compare no
longer "defines vcc_hi" for the data read after the join.

Program (32 threads, 4 x 4 x 2): vcc_hi = 3.0; lanes below 8 enter an execz block that compares lane < 4 into vcc_lo and
selects 1000 for those lanes (v_cndmask_b32); after the join every lane adds vcc_hi to float(lane). The reference is written
from that description with integers and float32, not from the translated text.
"""
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))

import gfx2hlsl as G  # noqa: E402
import disasm_rdna2 as D  # noqa: E402
from startup_compute_tg_size_check import S_ENDPGM, S_MOV_B32, V_MOV_B32, sop1, sop2, sopp, vop1  # noqa: E402
import startup_gfx2hlsl_dpp_quad_swap_cs_check as BASE  # noqa: E402
from startup_gfx2hlsl_compute_check import SENTINEL, WORDS, refused  # noqa: E402

GROUP = BASE.GROUP
VCC_HI = 107
VCC_LO = 106
EXEC_LO = 126
S_AND_SAVEEXEC_B32 = 0x3c   # SOP1
S_AND_B32 = 0x0e
S_BRANCH = 0x02
S_OR_B32 = 0x10             # SOP2
S_CBRANCH_EXECZ = 0x08      # SOPP
V_CMP_GT_I32 = 0x84         # VOPC
V_CMP_GT_U32 = 0xc4         # VOPC
V_CNDMASK_B32 = 0x01        # VOP2
V_ADD_F32 = 0x03            # VOP2
LITERAL = 255
THREE = 0x40400000          # 3.0f
SELECTED = 1000


def vopc(op, src0, vsrc1):
    return struct.pack('<I', 0x7c000000 | op << 17 | vsrc1 << 9 | src0)


def vop2(op, vdst, src0, vsrc1):
    return struct.pack('<I', op << 25 | vdst << 17 | vsrc1 << 9 | src0)


HEAD = BASE.PROGRAM[:BASE.PROGRAM.index(BASE.vop1_dpp(0x01, 8, 5, BASE.SWAP_X))]   # v5 = flat * 7 + 1, v6 = float(flat), v11 = flat
BLOCK = b''.join([
    vopc(V_CMP_GT_U32, 128 + 4, 11),                       # vcc_lo = flat < 4
    vop2(V_CNDMASK_B32, 8, 256 + 5, 10),                   # v8 = vcc_lo ? v10 : v5
])
PROGRAM = b''.join([
    HEAD,
    vop1(V_MOV_B32, 8, 256 + 5),                           # v8 = v5
    vop1(V_MOV_B32, 10, LITERAL, SELECTED),                # v10 = 1000
    sop1(S_MOV_B32, VCC_HI, LITERAL, THREE),               # vcc_hi = 3.0 (a scalar, not a mask)
    vopc(V_CMP_GT_U32, 128 + 8, 11),                       # vcc_lo = flat < 8
    sop1(S_AND_SAVEEXEC_B32, 6, VCC_LO),                   # s6 = exec_lo; exec_lo &= vcc_lo
    sopp(S_CBRANCH_EXECZ, len(BLOCK) // 4),                # skip the block when no lane is left
    BLOCK,
    sop2(S_OR_B32, EXEC_LO, EXEC_LO, 6),                   # join: restore exec_lo
    vop2(V_ADD_F32, 9, VCC_HI, 6),                         # v9 = vcc_hi + float(flat)
    BASE.S_WAITCNT_LGKM0,
    BASE.PROGRAM_TAIL_STORE,
])


# The c67 words are synthetic. VCC's low word is overwritten with full EXEC before it can be read.
HI_INIT = sop1(S_MOV_B32, VCC_HI, LITERAL, THREE)
PAIR_INIT = b''.join([
    sop1(S_MOV_B32, 94, 128 + 13),
    sop1(S_MOV_B32, 95, LITERAL, THREE),
    sop1(0x04, VCC_LO, 94),                              # s_mov_b64 vcc, s[94:95]
])
PAIR_PROGRAM = PROGRAM.replace(HI_INIT, PAIR_INIT, 1).replace(
    vopc(V_CMP_GT_U32, 128 + 4, 11), vopc(V_CMP_GT_I32, 128 + 4, 11), 1)


def early_out_program(limit, stop=128):
    return b''.join([
        HEAD,
        vop1(V_MOV_B32, 8, 256 + 5),                       # v8 = flat * 7 + 1
        vop1(V_MOV_B32, 9, 128),                           # v9 = 0
        vop1(V_MOV_B32, 10, LITERAL, SELECTED),             # v10 = 1000
        sop1(S_MOV_B32, 20, 193),                          # still-running mask = -1
        vopc(V_CMP_GT_U32, 128 + limit, 11),                # vcc_lo = flat < limit
        sop1(S_AND_SAVEEXEC_B32, 6, VCC_LO),
        sopp(S_CBRANCH_EXECZ, 2),
        sop1(S_MOV_B32, 20, stop),                         # stop every lane when the wave enters (0)
        sopp(S_BRANCH, 1),
        sopp(0x00),                                       # empty-mask path
        sop2(S_OR_B32, EXEC_LO, EXEC_LO, 6),
        sop2(S_AND_B32, EXEC_LO, EXEC_LO, 20),
        BASE.S_WAITCNT_LGKM0,
        BASE.PROGRAM_TAIL_STORE,
    ])


def early_out_words(limit, per_lane=False):
    words = [SENTINEL] * WORDS
    for lane in range(32):
        if limit == 0 or (per_lane and lane >= limit):
            words[4 * lane:4 * lane + 4] = [lane * 7 + 1, 0, SELECTED, lane]
    return words


def float_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def expected_words(vcc_hi=3.0):
    out = [SENTINEL] * WORDS
    for lane in range(32):
        out[4 * lane:4 * lane + 4] = [SELECTED if lane < 4 else lane * 7 + 1, float_bits(lane + vcc_hi), SELECTED, lane]
    return out


def translate(code=None, group=GROUP, wave_size=None):
    return G.ComputePipeline(PROGRAM if code is None else code, BASE.root_blob(), BASE.RSRC2, group, wave_size=wave_size)


def check_controls():
    asm = [a for _, _, _, a in G.decode(PROGRAM)]
    for line in ('s_mov_b32 vcc_hi, 0x40400000', 's_and_saveexec_b32 s6, vcc_lo', 's_or_b32 exec_lo, exec_lo, s6',
                 'v_add_f32_e32 v9, vcc_hi, v6'):
        assert line in asm, (line, asm)
    hlsl = translate().hlsl_cs()
    assert 'vcc_hi' not in hlsl, hlsl
    assert 'F(0x40400000u)' in hlsl, hlsl      # the scalar's value reaches the add
    # the caller's statement that the program is wave32 gives the same translation
    assert translate(wave_size=32).hlsl_cs() == hlsl
    # a program that names the 64-bit pair vcc keeps vcc_hi as the high half of the mask: refused as before
    pair = sop1(0x04, VCC_LO, 128)         # s_mov_b64 vcc, 0
    assert G.decode(pair)[0][3] == 's_mov_b64 vcc, 0', G.decode(pair)
    refused('vcc_hi', lambda: translate(code=PROGRAM[:len(HEAD)] + pair + PROGRAM[len(HEAD):]).hlsl_cs())
    pair_asm = [a for _, _, _, a in G.decode(PAIR_PROGRAM)]
    assert 's_mov_b64 vcc, s[94:95]' in pair_asm
    assert 'v_cmp_gt_i32_e32 vcc_lo, 4, v11' in pair_asm
    translate(PAIR_PROGRAM).hlsl_cs()
    refused('definition of vcc_hi, live where the block joins',
            lambda: translate(PAIR_PROGRAM, wave_size=64).hlsl_cs())
    low_read = vop1(V_MOV_B32, 8, VCC_LO)
    low_live = PAIR_PROGRAM.replace(PAIR_INIT, PAIR_INIT + low_read, 1)
    refused('data operand vcc_lo is a lane mask', lambda: translate(low_live).hlsl_cs())
    translate(early_out_program(8)).hlsl_cs()
    refused('definition of s20, live where the block joins',
            lambda: translate(early_out_program(8), group=(8, 4, 2)).hlsl_cs())
    shader = G.Stage.__new__(G.Stage)
    shader.ins = G.decode(early_out_program(8))
    shader.pipe = SimpleNamespace(group=(8, 4, 2), wave_size=32)
    assert any('branch out of the block' in reason for reason in shader.execz_refusals().values())
    # The group vote takes only a stop value of zero: any other value does not stop the lanes.
    for stop, taken in ((128, True), (193, False)):
        voter = G.Stage.__new__(G.Stage)
        voter.ins = G.decode(early_out_program(8, stop=stop))
        voter.pipe = SimpleNamespace(group=(8, 4, 1), wave_size=32)
        index = {row[0]: i for i, row in enumerate(voter.ins)}
        jumps = [(i, int(G.parse(row[3])[1][0].split('_')[1], 16)) for i, row in enumerate(voter.ins)
                 if row[2].startswith(('s_cbranch', 's_branch'))]
        assert (voter.single_wave_early_out(jumps, index) is not None) == taken, stop
    print('compute EXEC early-out and vcc_hi: translation controls and refusals pass')


CONTROLS = {
    # vcc_hi read as zero: every lane's v9 is float(lane) and lane 0 shows 0 instead of 3.0
    'vcc-hi-lost': (lambda h: h.replace('F(0x40400000u)', 'F(0x00000000u)', 1), dict(vcc_hi=0.0)),
}


def first_difference(expected, model):
    for index, (want, got) in enumerate(zip(expected, model)):
        if want != got:
            return 'word=%d expected=%08x actual=%08x' % (index, want, got)
    raise AssertionError('the mutant model equals the reference')


def check_dispatch(dxc, runner, adapter, control=None, fixture='vcc-hi'):
    if fixture == 'early-out':
        for limit in (0, 8):
            status = run_readback(dxc, runner, adapter, translate(early_out_program(limit)).hlsl_cs(),
                                  early_out_words(limit))
            if status:
                return status
        hlsl = translate(early_out_program(8)).hlsl_cs()
        assert hlsl.count('x2w_any_exec == 0u') == 1
        return run_readback(dxc, runner, adapter,
                            hlsl.replace('x2w_any_exec == 0u', '(exec & 1u) == 0u'),
                            early_out_words(8), early_out_words(8, per_lane=True), 'per-lane-execz')
    hlsl = translate(PAIR_PROGRAM if fixture == 'vcc-pair' else PROGRAM).hlsl_cs()
    mutant_words = None
    if control:
        mutate, model = CONTROLS[control]
        if fixture == 'vcc-pair':
            # The original scalar assignment and the pair's high-half copy both materialize the constant.
            before, literal, after = hlsl.rpartition('0x40400000u')
            assert literal
            mutated = before + '0x00000000u' + after
        else:
            mutated = mutate(hlsl)
        assert mutated != hlsl, (control, hlsl)
        hlsl = mutated
        mutant_words = expected_words(**model)
    return run_readback(dxc, runner, adapter, hlsl, expected_words(), mutant_words, control)


def run_readback(dxc, runner, adapter, hlsl, expected, mutant_words=None, control=None):
    runner = str(Path(runner).resolve())
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        d = Path(tmp)
        src, cso, case = d / 'vcchi.hlsl', d / 'vcchi.cso', d / 'vcchi.case'
        src.write_text(hlsl, encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
        case.write_bytes(struct.pack('<I', 1 | (1 << 8) | (1 << 16)) + struct.pack('<%dI' % WORDS, *expected)
                         + struct.pack('<16I', *([0] * 16)))
        run = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)
        print(run.stdout, end='')
        if run.returncode == 77:
            return 77
        diagnostics = [line for line in run.stderr.splitlines()
                       if not (adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '))]
        if control:
            want = first_difference(expected, mutant_words)
            if run.returncode != 1 or diagnostics[:1] != [want]:
                raise AssertionError('control %s: unexpected result (exit %d), wanted %s: %s' % (
                    control, run.returncode, want, run.stderr))
            print('control %s: rejected at %s' % (control, want))
        elif run.returncode or diagnostics:
            raise AssertionError('vcc_hi dispatch: %s' % (run.stderr or 'exit %d' % run.returncode))
        else:
            print('vcc_hi dispatch: all %d words match (1 group x %dx%dx%d threads)' % (WORDS, *GROUP))
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_controls()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(check_dispatch(*sys.argv[2:]))
    elif len(sys.argv) == 6 and sys.argv[1] == 'fixture':
        fixture = sys.argv[2]
        if fixture == 'vcc-pair':
            status = check_dispatch(*sys.argv[3:], fixture=fixture)
            sys.exit(status or check_dispatch(*sys.argv[3:], fixture=fixture, control='vcc-hi-lost'))
        if fixture == 'early-out':
            sys.exit(check_dispatch(*sys.argv[3:], fixture=fixture))
        sys.exit(__doc__)
    elif len(sys.argv) == 6 and sys.argv[1] == 'control':
        sys.exit(check_dispatch(*sys.argv[3:], control=sys.argv[2]))
    else:
        sys.exit(__doc__)
