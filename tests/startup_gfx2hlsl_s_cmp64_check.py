"""64-bit scalar equality comparisons (s_cmp_eq_u64, s_cmp_lg_u64) in tools/startup/gfx2hlsl.py.

  (no arguments)                       SOPC decode, value evaluation, uniformity, refusal and mutation
                                       controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per 64-bit pair and compare the four result words

RDNA 2 ISA, SOPC:
  S_CMP_EQ_U64 S0, S1: SCC = (S0.u64 == S1.u64)
  S_CMP_LG_U64 S0, S1: SCC = (S0.u64 != S1.u64)

The pixel fixture decodes synthetic gfx1013 bytes via G.decode() ONLY (no game data). It loads two 64-bit
values into s[4:5] and s[6:7] with SOP1 literal moves, executes s_cmp_eq_u64 / s_cmp_lg_u64 (pair vs pair,
pair vs sign-extended inline -1, and pair vs zero-extended 32-bit hex literal 0x80000001), materializes each
SCC via s_cselect_b32, XORs with 0x3f800000 for normal IEEE-754 transport into v12..v15, and exports mrt0.
"""
import re
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    MASK,
    POST_WORDS,
    PRE_WORDS,
    Probe,
    check,
    vs_words,
    words_text,
)

SOPC_OPCODES = {
    's_cmp_eq_u64': 0x12,
    's_cmp_lg_u64': 0x13,
}

# XOR 0/1 with 0x3f800000 so exported lanes are normal IEEE-754 floats (1.0 / 1.0 + 1 ULP), never subnormals.
DRAW_TRANSPORT_XOR = 0x3f800000


def sopc(op, ssrc0, ssrc1, lit=None):
    """Encode one synthetic gfx1013 SOPC 32-bit (or 64-bit with literal) instruction."""
    w0 = 0xbf000000 | (op << 16) | (ssrc1 << 8) | ssrc0
    out = struct.pack('<I', w0)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


def sop1_mov_lit(sdst, val):
    """Encode SOP1 s_mov_b32 s<sdst>, <32-bit literal> (op 0x03, ssrc0=255)."""
    return struct.pack('<2I', (0b101111101 << 23) | (sdst << 16) | (0x03 << 8) | 255, val & MASK)


def s_cselect_b32(sdst, ssrc0, ssrc1):
    """Encode SOP2 s_cselect_b32 s<sdst>, ssrc0, ssrc1 (op 0x0a)."""
    return struct.pack('<I', 0x80000000 | (0x0a << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0)


def s_xor_lit(sdst, ssrc0, lit):
    """Encode SOP2 s_xor_b32 s<sdst>, s<ssrc0>, <32-bit literal> (op 0x12, ssrc1=255)."""
    return struct.pack('<2I', 0x80000000 | (0x12 << 23) | (sdst << 16) | (255 << 8) | ssrc0, lit & MASK)


def v_mov_b32_from_sgpr(vdst, ssrc):
    """Encode VOP1 v_mov_b32_e32 v<vdst>, s<ssrc> (op 0x01)."""
    return struct.pack('<I', 0x7e000000 | (vdst << 17) | (0x01 << 9) | ssrc)


def ps_words(a0, a1, b0, b1):
    """Synthetic gfx1013 pixel stage exercising s_cmp_eq_u64 and s_cmp_lg_u64."""
    # Inline integer encodings in RDNA 2: 0 -> 128, 1 -> 129, -1 -> 193, literal -> 255.
    body = b''.join([
        sop1_mov_lit(4, a0),                      # s4 = a0
        sop1_mov_lit(5, a1),                      # s5 = a1
        sop1_mov_lit(6, b0),                      # s6 = b0
        sop1_mov_lit(7, b1),                      # s7 = b1
        sopc(0x12, 4, 6),                         # s_cmp_eq_u64 s[4:5], s[6:7]
        s_cselect_b32(8, 129, 128),               # s8 = scc ? 1 : 0
        s_xor_lit(8, 8, DRAW_TRANSPORT_XOR),
        v_mov_b32_from_sgpr(12, 8),               # v12 = (a == b) ^ 0x3f800000
        sopc(0x13, 4, 6),                         # s_cmp_lg_u64 s[4:5], s[6:7]
        s_cselect_b32(9, 129, 128),               # s9 = scc ? 1 : 0
        s_xor_lit(9, 9, DRAW_TRANSPORT_XOR),
        v_mov_b32_from_sgpr(13, 9),               # v13 = (a != b) ^ 0x3f800000
        sopc(0x12, 4, 193),                       # s_cmp_eq_u64 s[4:5], -1
        s_cselect_b32(10, 129, 128),              # s10 = scc ? 1 : 0
        s_xor_lit(10, 10, DRAW_TRANSPORT_XOR),
        v_mov_b32_from_sgpr(14, 10),              # v14 = (a == 0xffffffff_ffffffff) ^ 0x3f800000
        sopc(0x12, 4, 255, lit=0x80000001),       # s_cmp_eq_u64 s[4:5], 0x80000001
        s_cselect_b32(11, 129, 128),              # s11 = scc ? 1 : 0
        s_xor_lit(11, 11, DRAW_TRANSPORT_XOR),
        v_mov_b32_from_sgpr(15, 11),              # v15 = (a == 0x00000000_80000001) ^ 0x3f800000
    ])
    return PRE_WORDS + body + POST_WORDS


def translate_ps(a0=0x00000001, a1=0x00000002, b0=0x00000001, b1=0x00000002):
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(a0, a1, b0, b1), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def ref_cmp64(eq, a0, a1, b0, b1):
    """Independent 64-bit integer reference for S_CMP_EQ_U64 / S_CMP_LG_U64."""
    a = ((a1 & MASK) << 32) | (a0 & MASK)
    b = ((b1 & MASK) << 32) | (b0 & MASK)
    return (a == b) if eq else (a != b)


def eval_scc_expr(expr, env):
    """Evaluate a translated HLSL boolean SCC expression in Python."""
    text = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', expr)
    text = text.replace('&&', ' and ').replace('||', ' or ')
    return bool(eval(text, {'__builtins__': {}}, env))


def run_one(line, env, *, sym=None):
    stage = Probe()
    if sym:
        stage.sym.update(sym)
    stage.one(line.split(' ', 1)[0], line)
    assert len(stage.lines) == 1, stage.lines
    target, _, expr = stage.lines[0].rstrip(';').partition(' = ')
    assert target == 'scc', stage.lines
    return eval_scc_expr(expr, env), stage


def stage_refused(stage, mnemonic, line, needle=''):
    try:
        stage.one(mnemonic, line)
    except G.Unsupported as exc:
        return (not needle or needle in str(exc)) and not stage.lines
    return False


WORDS64 = (
    (0x00000000, 0x00000000),
    (0x00000001, 0x00000000),
    (0x00000000, 0x00000001),
    (0xffffffff, 0x00000000),
    (0x00000000, 0xffffffff),
    (0xffffffff, 0xffffffff),
    (0x80000001, 0x00000000),
    (0x80000001, 0xffffffff),
    (0x12345678, 0x9abcdef0),
    (0x12345678, 0x9abcdef1),
    (0x12345679, 0x9abcdef0),
)


def run_controls():
    # 1. Decode verification from synthetic gfx1013 SOPC bytes via G.decode().
    decoded = [
        G.decode(sopc(0x12, 4, 6))[0][3],
        G.decode(sopc(0x13, 4, 6))[0][3],
        G.decode(sopc(0x12, 4, 193))[0][3],
        G.decode(sopc(0x13, 255, 6, lit=0x80000001))[0][3],
        G.decode(sopc(0x12, 125, 128))[0][3],
    ]
    check('decode s_cmp_eq_u64 pair/pair', decoded[0] == 's_cmp_eq_u64 s[4:5], s[6:7]')
    check('decode s_cmp_lg_u64 pair/pair', decoded[1] == 's_cmp_lg_u64 s[4:5], s[6:7]')
    check('decode s_cmp_eq_u64 inline -1', decoded[2] == 's_cmp_eq_u64 s[4:5], -1')
    check('decode s_cmp_lg_u64 literal', decoded[3] == 's_cmp_lg_u64 0x80000001, s[6:7]')
    check('decode s_cmp_eq_u64 null/0', decoded[4] == 's_cmp_eq_u64 null, 0')

    # 2. Exhaustive SGPR pair x SGPR pair grid (including low-only and high-only mismatches).
    grid_ok = True
    for mnemonic, eq in (('s_cmp_eq_u64', True), ('s_cmp_lg_u64', False)):
        for a0, a1 in WORDS64:
            for b0, b1 in WORDS64:
                expected = ref_cmp64(eq, a0, a1, b0, b1)
                got, _ = run_one('%s s[4:5], s[6:7]' % mnemonic,
                                 {'s4': a0, 's5': a1, 's6': b0, 's7': b1})
                if got != expected:
                    grid_ok = False
                    print('MISMATCH %s a=(%#x,%#x) b=(%#x,%#x) got=%s want=%s' % (
                        mnemonic, a0, a1, b0, b1, got, expected))
    check('SGPR pair grid (%d cases)' % (2 * len(WORDS64) * len(WORDS64)), grid_ok)

    # 3. Inline integer 64-bit operands (negative values sign-extend to 64 bits).
    inline_ok = True
    for mnemonic, eq in (('s_cmp_eq_u64', True), ('s_cmp_lg_u64', False)):
        for val in (0, 1, 64, -1, -16):
            v0, v1 = val & MASK, (val >> 32) & MASK
            for a0, a1 in ((v0, v1), (v0, 0), (v0, 0xffffffff), (0, v1)):
                got_r, _ = run_one('%s s[4:5], %d' % (mnemonic, val), {'s4': a0, 's5': a1})
                got_l, _ = run_one('%s %d, s[4:5]' % (mnemonic, val), {'s4': a0, 's5': a1})
                want = ref_cmp64(eq, a0, a1, v0, v1)
                if got_r != want or got_l != want:
                    inline_ok = False
    check('inline integers sign-extend negative values to 64 bits', inline_ok)

    # 4. 32-bit hex literal zero-extends to 64 bits (hi == 0 even when bit 31 is set).
    lit_ok = True
    for mnemonic, eq in (('s_cmp_eq_u64', True), ('s_cmp_lg_u64', False)):
        for a0, a1 in ((0x80000001, 0), (0x80000001, 0xffffffff), (0x80000000, 0)):
            got_r, _ = run_one('%s s[4:5], 0x80000001' % mnemonic, {'s4': a0, 's5': a1})
            got_l, _ = run_one('%s 0x80000001, s[4:5]' % mnemonic, {'s4': a0, 's5': a1})
            want = ref_cmp64(eq, a0, a1, 0x80000001, 0)
            if got_r != want or got_l != want:
                lit_ok = False
    check('32-bit hex literal zero-extends to 64 bits', lit_ok)

    # 5. null operand and const-tracked SGPR pair.
    for mnemonic, eq in (('s_cmp_eq_u64', True), ('s_cmp_lg_u64', False)):
        got_zero, _ = run_one('%s s[4:5], null' % mnemonic, {'s4': 0, 's5': 0})
        got_hi, _ = run_one('%s null, s[4:5]' % mnemonic, {'s4': 0, 's5': 1})
        check('%s null == 0' % mnemonic, got_zero == eq)
        check('%s null != (0, 1)' % mnemonic, got_hi == (not eq))
        got_const, _ = run_one('%s s[4:5], s[6:7]' % mnemonic, {'s5': 0x99, 's6': 0x42, 's7': 0x99},
                               sym={4: ('const', 0x42)})
        check('%s const-tracked lo + runtime hi' % mnemonic, got_const == eq)

    # 6. Uniformity tracking across all four registers s4, s5, s6, s7.
    for mnemonic in ('s_cmp_eq_u64', 's_cmp_lg_u64'):
        s = Probe()
        s.varying_s = set()
        s.scc_uniform = False
        s.one(mnemonic, '%s s[4:5], s[6:7]' % mnemonic)
        check('%s uniform inputs -> scc_uniform=True' % mnemonic, s.scc_uniform is True)
        for bad_reg in (4, 5, 6, 7):
            s = Probe()
            s.varying_s = {bad_reg}
            s.scc_uniform = True
            s.one(mnemonic, '%s s[4:5], s[6:7]' % mnemonic)
            check('%s varying s%d -> scc_uniform=False' % (mnemonic, bad_reg), s.scc_uniform is False)

    # 7. Refusals (arity, modifiers, lane masks, mask_high, invalid register classes/alignment).
    for mnemonic in ('s_cmp_eq_u64', 's_cmp_lg_u64'):
        check('%s 1 operand refused' % mnemonic,
              stage_refused(Probe(), mnemonic, '%s s[4:5]' % mnemonic, 'two unmodified operands'))
        check('%s 3 operands refused' % mnemonic,
              stage_refused(Probe(), mnemonic, '%s s[4:5], s[6:7], s[8:9]' % mnemonic, 'two unmodified operands'))
        check('%s modifier refused' % mnemonic,
              stage_refused(Probe(), mnemonic, '%s s[4:5], s[6:7] clamp' % mnemonic, 'two unmodified operands'))
        for operands in (
            'exec, s[6:7]', 'exec_lo, s[6:7]', 'vcc, s[6:7]', 'vcc_lo, s[6:7]',
            's[4:5], exec', 's[4:5], exec_lo', 's[4:5], vcc', 's[4:5], vcc_lo',
        ):
            check('%s %s lane mask refused' % (mnemonic, operands),
                  stage_refused(Probe(), mnemonic, '%s %s' % (mnemonic, operands), 'lane mask'))
        for mask_reg, high_reg, operands in (
            (12, None, 's[12:13], s[6:7]'),
            (13, None, 's[12:13], s[6:7]'),
            (None, 13, 's[12:13], s[6:7]'),
            (12, None, 's[4:5], s[12:13]'),
            (13, None, 's[4:5], s[12:13]'),
            (None, 13, 's[4:5], s[12:13]'),
        ):
            p = Probe()
            if mask_reg is not None:
                p.masks = {mask_reg}
            if high_reg is not None:
                p.mask_high = {high_reg}
            check('%s %s tracked mask refused' % (mnemonic, operands),
                  stage_refused(p, mnemonic, '%s %s' % (mnemonic, operands), 'lane mask'))
        for bad_op in ('v[0:1]', 's4', 's[5:6]', 's[106:107]', '1.0', 'm0', 'src_scc'):
            check('%s bad S0 %s refused' % (mnemonic, bad_op),
                  stage_refused(Probe(), mnemonic, '%s %s, s[6:7]' % (mnemonic, bad_op)))
            check('%s bad S1 %s refused' % (mnemonic, bad_op),
                  stage_refused(Probe(), mnemonic, '%s s[4:5], %s' % (mnemonic, bad_op)))

    # 8. Full synthetic pixel pipeline translates cleanly through G.Pipeline().
    _, ps_hlsl = translate_ps()
    check('synthetic pixel pipeline translates', '&&' in ps_hlsl and '||' in ps_hlsl)

    # 9. Unlowered control: removing handlers from Stage restores refusals.
    saved = {name: getattr(G.Stage, name) for name in ('op_s_cmp_eq_u64', 'op_s_cmp_lg_u64')}
    try:
        for name in saved:
            delattr(G.Stage, name)
        for mnemonic in ('s_cmp_eq_u64', 's_cmp_lg_u64'):
            check('unlowered %s refused' % mnemonic,
                  stage_refused(Probe(), mnemonic, '%s s[4:5], s[6:7]' % mnemonic))
        try:
            translate_ps()
        except G.Unsupported:
            check('unlowered synthetic pipeline refused', True)
        else:
            check('unlowered synthetic pipeline refused', False)
    finally:
        for name, fn in saved.items():
            setattr(G.Stage, name, fn)

    # 10. Mutation / negative controls: prove our inputs distinguish common bugs.
    check('mutation: ignoring high word diverges on (1,0) vs (1,1)',
          ref_cmp64(True, 1, 0, 1, 1) != (1 == 1))
    check('mutation: ignoring low word diverges on (0,1) vs (1,1)',
          ref_cmp64(True, 0, 1, 1, 1) != (1 == 1))
    check('mutation: using && instead of || for lg diverges on high-only mismatch',
          ref_cmp64(False, 1, 0, 1, 1) != ((1 != 1) and (0 != 1)))
    check('mutation: zero-extending -1 diverges on (0xffffffff, 0xffffffff)',
          ref_cmp64(True, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff)
          != ref_cmp64(True, 0xffffffff, 0xffffffff, 0xffffffff, 0))
    check('mutation: sign-extending 0x80000001 diverges on (0x80000001, 0)',
          ref_cmp64(True, 0x80000001, 0, 0x80000001, 0)
          != ref_cmp64(True, 0x80000001, 0, 0x80000001, 0xffffffff))


GPU_VECTORS = (
    # (label, a0, a1, b0, b1)
    ('equal_pair',       0x12345678, 0x9abcdef0, 0x12345678, 0x9abcdef0),
    ('diff_lo_only',     0x12345678, 0x9abcdef0, 0x12345679, 0x9abcdef0),
    ('diff_hi_only',     0x12345678, 0x9abcdef0, 0x12345678, 0x9abcdef1),
    ('match_inline_m1',  0xffffffff, 0xffffffff, 0xffffffff, 0x00000000),
    ('m1_hi_zero_only',  0xffffffff, 0x00000000, 0xffffffff, 0x00000000),
    ('match_lit_zext',   0x80000001, 0x00000000, 0x00000000, 0x00000000),
    ('lit_hi_all_ones',  0x80000001, 0xffffffff, 0x80000001, 0xffffffff),
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory(prefix='x2w_s_cmp64_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            first_vs = None
            first_ps = None
            first_exp = None
            for label, a0, a1, b0, b1 in GPU_VECTORS:
                vs_hlsl, ps_hlsl = translate_ps(a0, a1, b0, b1)
                vs_bin = compile_hlsl(compiler, kind, vs_hlsl, 'vs_%s_%s' % (label, kind), vs_profile, td)
                ps_bin = compile_hlsl(compiler, kind, ps_hlsl, 'ps_%s_%s' % (label, kind), ps_profile, td)
                raw_exp = (
                    (1 if ref_cmp64(True, a0, a1, b0, b1) else 0) ^ DRAW_TRANSPORT_XOR,
                    (1 if ref_cmp64(False, a0, a1, b0, b1) else 0) ^ DRAW_TRANSPORT_XOR,
                    (1 if ref_cmp64(True, a0, a1, 0xffffffff, 0xffffffff) else 0) ^ DRAW_TRANSPORT_XOR,
                    (1 if ref_cmp64(True, a0, a1, 0x80000001, 0x00000000) else 0) ^ DRAW_TRANSPORT_XOR,
                )
                expected = words_text(raw_exp)
                res = draw(renderer, adapter, vs_bin, ps_bin, expected)
                if res.returncode == 77:
                    print('SKIP (%s): %s' % (adapter, res.stdout.strip() or res.stderr.strip()))
                    sys.exit(77)
                if res.returncode != 0:
                    raise AssertionError('%s %s %s expected=%s\n%s\n%s' % (
                        kind, adapter, label, expected, res.stdout, res.stderr))
                print('%s %s %s: %s' % (kind, adapter, label, expected), flush=True)
                if first_vs is None:
                    first_vs, first_ps, first_exp = vs_bin, ps_bin, raw_exp

            # Negative control: deliberately wrong expected word must fail draw().
            bad = words_text((first_exp[0] ^ 1, first_exp[1], first_exp[2], first_exp[3]))
            neg = draw(renderer, adapter, first_vs, first_ps, bad)
            check('draw %s %s negative control rejects wrong expectation' % (kind, adapter),
                  neg.returncode != 0)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run_controls()
        print('s_cmp64 controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('s_cmp64 draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_s_cmp64_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
