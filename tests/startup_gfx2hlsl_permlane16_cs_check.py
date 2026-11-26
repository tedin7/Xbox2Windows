#!/usr/bin/env python3
"""v_permlane16_b32 in a compute program of tools/startup/gfx2hlsl.py (sibling of startup_gfx2hlsl_dpp_quad_swap_cs_check.py).

  (no arguments)                         translation controls and refusals, pure Python
  dispatch <dxc> <offscreen_check> <warp|hardware>
                                         translate, compile with DXC cs_6_0, dispatch and compare all 320 words
  control <wrong-row|swapped-selects> <dxc> <offscreen_check> <warp|hardware>
                                         mutated translation must mismatch on readback at the word the oracle predicts

v_permlane16_b32 vdst, vsrc0, ssrc1, ssrc2 (AMD RDNA 2 ISA reference, VOP3 lane permutes; LLVM 15 gfx10): inside each row of
16 lanes, lane i reads vsrc0 of lane sel[i] of the same row; sel[i] is 4-bit field i of ssrc1 for lanes 0..7 and field i - 8
of ssrc2 for lanes 8..15. The reference below is a scalar lane oracle written from that rule, not from the translated text.
A selector can be an appended literal. The exchange goes through groupshared memory in a straight-line compute program with entry EXEC
in a group of whole rows (so the op_sel fetch-inactive and bound_ctrl flags, which only decide what a disabled lane reads,
decide nothing).
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))

import gfx2hlsl as G  # noqa: E402
import disasm_rdna2 as D  # noqa: E402
from startup_compute_tg_size_check import (  # noqa: E402
    S_ENDPGM, S_MOV_B32, S_CMP_EQ_U32, V_MOV_B32, sop1, sopc, sopp, vop1)
import startup_gfx2hlsl_dpp_quad_swap_cs_check as BASE  # noqa: E402
from startup_gfx2hlsl_compute_check import SENTINEL, WORDS, refused  # noqa: E402

GROUP = (4, 4, 4)             # 64 threads: four rows of 16
SEL_LOW, SEL_HIGH = 0x01234567, 0x89abcdef   # lanes 0..7 read 7 - i, lanes 8..15 read 23 - i
S_LOW, S_HIGH = 24, 25        # SGPRs the program fills with the selects (the prefix leaves them free)
LITERAL = 255
V_PERMLANE16_B32 = 0x377


def permlane(vdst, vsrc, s1, s2, op_sel=0, literal=None):
    """VOP3 v_permlane16_b32: op_sel[0] (fetch inactive) and op_sel[1] (bound_ctrl) are w0 bits 11 and 12."""
    w0 = 0xd4000000 | V_PERMLANE16_B32 << 16 | op_sel << 11 | vdst
    word = struct.pack('<2I', w0, (256 + vsrc) | s1 << 9 | s2 << 18)
    return word if literal is None else word + struct.pack('<I', literal)


# v5 = flat * 7 + 1; v11 = flat (both from BASE.PROGRAM's prefix); v8 = permlane(v5, s24, s25); v9 = permlane(v5, s25, s24)
# (selects exchanged); v10 = permlane(v11, s24, s25) with both op_sel flags set. v8 and v10 encode one select as a literal.
HEAD = BASE.PROGRAM[:BASE.PROGRAM.index(BASE.vop1_dpp(0x01, 8, 5, BASE.SWAP_X))]
PROGRAM = b''.join([
    HEAD,
    sop1(S_MOV_B32, S_LOW, LITERAL, SEL_LOW),
    sop1(S_MOV_B32, S_HIGH, LITERAL, SEL_HIGH),
    permlane(8, 5, LITERAL, S_HIGH, literal=SEL_LOW),
    permlane(9, 5, S_HIGH, S_LOW),
    permlane(10, 11, S_LOW, LITERAL, op_sel=3, literal=SEL_HIGH),
    BASE.S_WAITCNT_LGKM0,
    BASE.PROGRAM_TAIL_STORE,
])


def source_lane(lane, low, high):
    """The lane of its own row that `lane` reads: the scalar oracle."""
    field = (low if lane % 16 < 8 else high) >> (4 * (lane % 8)) & 15
    return lane // 16 * 16 + field


def expected_words(low=SEL_LOW, high=SEL_HIGH, whole_row=True, swap=False):
    out = [SENTINEL] * WORDS
    for lane in range(GROUP[0] * GROUP[1] * GROUP[2]):
        a = source_lane(lane, low, high)
        b = source_lane(lane, high, low)
        if not whole_row:                      # mutant: every lane reads row 0
            a, b = a % 16, b % 16
        if swap:                               # mutant: the two selects exchanged
            a = source_lane(lane, high, low)
        out[4 * lane:4 * lane + 4] = [a * 7 + 1, b * 7 + 1, a, lane]
    return out


def translate(code=None, group=GROUP):
    return G.ComputePipeline(PROGRAM if code is None else code, BASE.root_blob(), BASE.RSRC2, group)


def check_controls():
    permutes = [(length, asm) for _, length, op, asm in G.decode(PROGRAM) if op == 'v_permlane16_b32']
    assert permutes == [
        (12, 'v_permlane16_b32 v8, v5, 0x1234567, s25'),
        (8, 'v_permlane16_b32 v9, v5, s25, s24'),
        (12, 'v_permlane16_b32 v10, v11, s24, 0x89abcdef op_sel:[1,1]'),
    ], permutes
    missing_literal = permlane(8, 5, LITERAL, S_HIGH)
    assert D.decode_instruction(missing_literal, 0)[:2] == (8, 'vop3_truncated')
    double_literal = permlane(8, 5, LITERAL, LITERAL, literal=SEL_LOW)
    assert D.decode_instruction(double_literal, 0)[:2] == (12, 'v_permlane16_b32_unsupported')
    hlsl = translate().hlsl_cs()
    flat = '(tid.x + 4u * (tid.y + 4u * tid.z))'
    assert 'groupshared uint x2w_quad[64];' in hlsl, hlsl
    assert hlsl.count('& ~15u') == 3 and 'x2w_quad[%s] = v5;' % flat in hlsl, hlsl
    assert hlsl.count('GroupMemoryBarrierWithGroupSync();') == 6, hlsl
    one = permlane(8, 5, S_LOW, S_HIGH) + S_ENDPGM
    # whole rows only
    for bad in ((4, 1, 1), (12, 1, 1), (8, 1, 1)):
        refused('requires a group of whole rows', lambda g=bad: translate(code=one, group=g).hlsl_cs())
    # straight-line code with entry EXEC only
    branchy = b''.join([sopc(S_CMP_EQ_U32, 7, BASE.INLINE_0), sopp(0x04, 1), vop1(V_MOV_B32, 3, 256), one])
    refused('requires a straight-line compute program with entry EXEC', lambda: translate(code=branchy).hlsl_cs())
    # an SGPR source is a word the decoder reports as unsupported, so it is refused too
    sgpr_source = struct.pack('<2I', 0xd4000000 | V_PERMLANE16_B32 << 16 | 8, 6 | S_LOW << 9 | S_HIGH << 18)
    refused('unsupported', lambda: translate(code=sgpr_source + S_ENDPGM).hlsl_cs())
    # the decoder's unsupported encodings stay refused (abs bit set on the word)
    word = bytearray(permlane(8, 5, S_LOW, S_HIGH))
    word[1] |= 0x01
    assert D.decode_instruction(bytes(word), 0)[1].endswith('_unsupported')
    refused('unsupported', lambda: translate(code=bytes(word) + S_ENDPGM).hlsl_cs())
    print('v_permlane16_b32 in compute: translation controls and refusals pass')


CONTROLS = {
    # every lane reads row 0: lane 16 gets lane 7 instead of lane 23
    'wrong-row': (lambda h: h.replace('& ~15u', '& 0u'), dict(whole_row=False)),
    # the two selects exchanged in the first permute: lane 0 reads lane 8 + ... instead of lane 7
    'swapped-selects': (lambda h: h.replace('< 8u ? 0x01234567u : 0x89abcdefu', '< 8u ? 0x89abcdefu : 0x01234567u', 1),
                        dict(swap=True)),
}


def first_difference(expected, actual_model):
    for index, (want, got) in enumerate(zip(expected, actual_model)):
        if want != got:
            return 'word=%d expected=%08x actual=%08x' % (index, want, got)
    raise AssertionError('the mutant model equals the oracle')


def check_dispatch(dxc, runner, adapter, control=None):
    runner = str(Path(runner).resolve())
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    hlsl = translate().hlsl_cs()
    if control:
        if control not in CONTROLS:
            raise SystemExit('unknown control %r' % control)
        mutate, model = CONTROLS[control]
        mutated = mutate(hlsl)
        assert mutated != hlsl, (control, hlsl)
        hlsl = mutated
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        d = Path(tmp)
        src, cso, case = d / 'permlane.hlsl', d / 'permlane.cso', d / 'permlane.case'
        src.write_text(hlsl, encoding='ascii')
        subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
        case.write_bytes(struct.pack('<I', 1 | (1 << 8) | (1 << 16)) + struct.pack('<%dI' % WORDS, *expected_words())
                         + struct.pack('<16I', *([0] * 16)))
        run = subprocess.run([runner, adapter, '--compute-store-case', str(cso), str(case)], capture_output=True, text=True)
        print(run.stdout, end='')
        if run.returncode == 77:
            return 77
        diagnostics = [line for line in run.stderr.splitlines()
                       if not (adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '))]
        if control:
            want = first_difference(expected_words(), expected_words(**CONTROLS[control][1]))
            if run.returncode != 1 or diagnostics[:1] != [want]:
                raise AssertionError('control %s: unexpected result (exit %d), wanted %s: %s' % (
                    control, run.returncode, want, run.stderr))
            print('control %s: rejected at %s' % (control, want))
        elif run.returncode or diagnostics:
            raise AssertionError('permlane16 dispatch: %s' % (run.stderr or 'exit %d' % run.returncode))
        else:
            print('permlane16 dispatch: all %d words match (1 group x %dx%dx%d threads)' % (WORDS, *GROUP))
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_controls()
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(check_dispatch(*sys.argv[2:]))
    elif len(sys.argv) == 6 and sys.argv[1] == 'control':
        sys.exit(check_dispatch(*sys.argv[3:], control=sys.argv[2]))
    else:
        sys.exit(__doc__)
