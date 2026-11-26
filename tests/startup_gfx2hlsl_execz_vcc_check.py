"""Synthetic wave32 EXECZ VCC joins. Run with no arguments for translation checks,
or with <dxc> <offscreen_check> <warp> for dispatch and readback controls.
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import startup_gfx2hlsl_compute_vcc_hi_check as B


S_CBRANCH_VCCZ = 0x06
MOVE_MASK = 0xa5000081
CASES = {
    'signed': (B.vopc(0x84, 128 + 4, 11), lambda lane: lane < 4, 'v_cmp_gt_i32_e32'),
    'class': (B.vopc(0x88, 256 + 6, 10), lambda lane: lane != 0, 'v_cmp_class_f32_e32'),
    'float': (B.vopc(0x01, 256 + 6, 10), lambda lane: False, 'v_cmp_lt_f32_e32'),
    'move': (B.sop1(B.S_MOV_B32, B.VCC_LO, B.LITERAL, MOVE_MASK),
             lambda lane: bool(MOVE_MASK & (1 << lane)), 's_mov_b32'),
    'move-exec': (B.sop1(B.S_MOV_B32, B.VCC_LO, B.EXEC_LO), lambda lane: lane < 8, 's_mov_b32'),
}


def mov(dst, src, *literal):
    return B.vop1(B.V_MOV_B32, dst, src, *literal)


def program(name, empty):
    block, _, _ = CASES[name]
    return b''.join([
        B.HEAD,
        mov(10, B.LITERAL, 0x100 if name == 'class' else 0),
        B.vopc(B.V_CMP_GT_U32, 128 + (0 if empty else 8), 11),
        B.sop1(B.S_AND_SAVEEXEC_B32, 6, B.VCC_LO),
        B.sop1(B.S_MOV_B32, B.VCC_LO, 193),      # stale low bits in all lanes
        B.sop1(B.S_MOV_B32, B.VCC_HI, B.LITERAL, B.THREE),
        B.sopp(B.S_CBRANCH_EXECZ, len(block) // 4),
        block,
        B.sop2(B.S_OR_B32, B.EXEC_LO, B.EXEC_LO, 6),
        mov(8, B.LITERAL, B.SELECTED),
        B.vop2(B.V_CNDMASK_B32, 8, 256 + 5, 8),
        B.vop2(B.V_ADD_F32, 9, B.VCC_HI, 6),
        mov(10, 128),
        B.sopp(S_CBRANCH_VCCZ, 1),
        mov(10, 129),                           # check that all lanes take the same VCC branch
        B.BASE.S_WAITCNT_LGKM0,
        B.BASE.PROGRAM_TAIL_STORE,
    ])


def expected_words(name, empty, threads, mutant=False):
    _, predicate, _ = CASES[name]
    out = []
    for wave_start in range(0, threads, 32):
        # The second wave in a 64-thread group has no active lane at EXECZ.
        skip = empty or wave_start != 0
        bits = [skip or ((name.startswith('move') or lane < 8) and predicate(lane)) for lane in range(32)]
        high = 3 if skip or name.startswith('move') else 0
        stale = [mutant and not skip and lane >= 8 for lane in range(32)]
        nonzero = int(any(bits) or any(stale))
        for lane, bit in enumerate(bits):
            flat = wave_start + lane
            out.extend((B.SELECTED if bit or stale[lane] else flat * 7 + 1,
                        B.float_bits(flat + (3 if stale[lane] else high)), nonzero, flat))
    return out + [B.SENTINEL] * (B.WORDS - len(out))


def translate(name, empty, threads=32, **kwargs):
    group = (4, 4, threads // 16)
    return B.G.ComputePipeline(program(name, empty), B.BASE.root_blob(), B.BASE.RSRC2, group, **kwargs)


def local_branch(hlsl):
    """Restore the lane-local EXECZ decision while keeping the same lowered body."""
    old = 'pc == 0 ? exec : 0u);'
    assert hlsl.count(old) == 1, hlsl
    start = hlsl.index('    uint x2w_vote_0 = x2w_mask_ballot(')
    end = hlsl.index('\n', start)
    assert old in hlsl[start:end], hlsl[start:end]
    return hlsl[:start] + '    uint x2w_vote_0 = exec & 1u;' + hlsl[end:]


def checks():
    for name, (_, _, mnemonic) in CASES.items():
        for threads, empty in ((32, False), (32, True), (64, False)):
            hlsl = translate(name, empty, threads).hlsl_cs()
            assert 'pc == 0 ? exec : 0u);' in hlsl and 'pc == 2 ? vcc : 0u);' in hlsl, name
            refusals = []
            assert translate(name, empty, threads, refusals=refusals).hlsl_cs() == hlsl
            assert not refusals, refusals
            if not empty:
                assert local_branch(hlsl) != hlsl
                assert expected_words(name, False, threads, True) != expected_words(name, False, threads)
        with patch.object(B.G.Stage, 'can_vote_execz', return_value=False):
            B.refused(mnemonic + ' inside a block s_cbranch_execz skips per lane: definition of vcc',
                      lambda: translate(name, False).hlsl_cs())
    print('EXECZ VCC joins: 15 translations and 5 legacy refusal controls pass')


def dispatch(dxc, runner, adapter):
    runner = str(Path(runner).resolve())
    out = B.ROOT / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        src, binary, case = [Path(tmp) / file for file in ('join.hlsl', 'join.cso', 'join.case')]
        for name in CASES:
            for threads, empty, mutant in ((32, False, False), (32, True, False), (64, False, False),
                                           (32, False, True), (64, False, True)):
                hlsl = translate(name, empty, threads).hlsl_cs()
                src.write_text(local_branch(hlsl) if mutant else hlsl, encoding='ascii')
                subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(binary), str(src)], check=True)
                expected = expected_words(name, empty, threads)
                case.write_bytes(struct.pack('<%dI' % (1 + B.WORDS + 16),
                                             1 | (1 << 8) | (1 << 16), *expected, *([0] * 16)))
                run = subprocess.run([runner, adapter, '--compute-store-case', str(binary), str(case)],
                                     capture_output=True, text=True)
                if run.returncode == 77:
                    return 77
                if mutant:
                    diagnostic = B.first_difference(expected, expected_words(name, empty, threads, mutant=True))
                    assert run.returncode == 1 and run.stderr.splitlines()[:1] == [diagnostic], (
                        name, diagnostic, run.stdout, run.stderr)
                    print(f'{name} lane-local control rejected at {diagnostic}')
                else:
                    assert run.returncode == 0, (name, empty, run.stdout, run.stderr)
                    print(f'{name} threads={threads} empty={empty} matches all {B.WORDS} words')
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        checks()
    else:
        sys.exit(dispatch(*sys.argv[1:]))
