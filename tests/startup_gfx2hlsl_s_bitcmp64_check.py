"""64-bit scalar bit-compare (s_bitcmp0_b64, s_bitcmp1_b64) in tools/startup/gfx2hlsl.py.

  (no arguments)                       SOPC decode, value evaluation, uniformity, refusal and mutation
                                       controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per (lo, hi, index) triple and compare the four result words

RDNA 2 ISA, SOPC:
  S_BITCMP0_B64 S0, S1: SCC = (S0.u64[S1.u[5:0]] == 0)
  S_BITCMP1_B64 S0, S1: SCC = (S0.u64[S1.u[5:0]] == 1)

The pixel fixture decodes synthetic gfx1013 bytes via G.decode() ONLY (no game data). It loads (lo, hi, index)
into s4..s6 with SOP1 literal moves, executes s_bitcmp0_b64 / s_bitcmp1_b64 (register index, immediate index 37,
and sign-extended inline -1 source), materializes each SCC via s_cselect_b32, XORs with 0x3f800000 for normal
IEEE-754 transport into v12..v15, and exports mrt0.
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
    's_bitcmp0_b64': 0x0e,
    's_bitcmp1_b64': 0x0f,
}


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


# XOR 0/1 with 0x3f800000 so exported lanes are normal IEEE-754 floats (1.0 / 1.0 + 1 ULP), never subnormals.
DRAW_TRANSPORT_XOR = 0x3f800000


def ps_words(lo, hi, idx):
    """Synthetic gfx1013 pixel stage exercising s_bitcmp0_b64 and s_bitcmp1_b64."""
    # Inline integer encodings in RDNA 2: 0 -> 128, 1 -> 129, 37 -> 128+37=165, -1 -> 193.
    body = b''.join([
        sop1_mov_lit(4, lo),                  # s4 = lo
        sop1_mov_lit(5, hi),                  # s5 = hi
        sop1_mov_lit(6, idx),                 # s6 = bit index
        sopc(0x0e, 4, 6),                     # s_bitcmp0_b64 s[4:5], s6
        s_cselect_b32(8, 129, 128),           # s8 = scc ? 1 : 0
        s_xor_lit(8, 8, DRAW_TRANSPORT_XOR),
        v_mov_b32_from_sgpr(12, 8),           # v12 = s8 ^ 0x3f800000
        sopc(0x0f, 4, 6),                     # s_bitcmp1_b64 s[4:5], s6
        s_cselect_b32(9, 129, 128),           # s9 = scc ? 1 : 0
        s_xor_lit(9, 9, DRAW_TRANSPORT_XOR),
        v_mov_b32_from_sgpr(13, 9),           # v13 = s9 ^ 0x3f800000
        sopc(0x0f, 4, 128 + 37),              # s_bitcmp1_b64 s[4:5], 37
        s_cselect_b32(10, 129, 128),          # s10 = scc ? 1 : 0
        s_xor_lit(10, 10, DRAW_TRANSPORT_XOR),
        v_mov_b32_from_sgpr(14, 10),          # v14 = s10 ^ 0x3f800000
        sopc(0x0f, 193, 6),                   # s_bitcmp1_b64 -1, s6
        s_cselect_b32(11, 129, 128),          # s11 = scc ? 1 : 0
        s_xor_lit(11, 11, DRAW_TRANSPORT_XOR),
        v_mov_b32_from_sgpr(15, 11),          # v15 = s11 ^ 0x3f800000
    ])
    return PRE_WORDS + body + POST_WORDS


def translate_ps(lo=0x00000001, hi=0x00000020, idx=37):
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(lo, hi, idx), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def ref_bitcmp64(want, lo, hi, idx):
    """Independent bit-string reference for S_BITCMP{0,1}_B64."""
    bits = format((hi & MASK) << 32 | (lo & MASK), '064b')[::-1]
    return int(bits[idx & 63]) == want


def eval_scc_expr(expr, env):
    """Evaluate a translated HLSL boolean SCC expression in Python."""
    text = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', expr)
    # Convert C ternary (cond ? a : b) handling arbitrary balanced parens inside cond/a/b.
    while '?' in text:
        q = text.index('?')
        depth = 0
        start = q - 1
        while start >= 0:
            if text[start] == ')':
                depth += 1
            elif text[start] == '(':
                if depth == 0:
                    break
                depth -= 1
            start -= 1
        depth = 0
        colon = -1
        end = q + 1
        while end < len(text):
            if text[end] == '(':
                depth += 1
            elif text[end] == ')':
                if depth == 0:
                    break
                depth -= 1
            elif text[end] == ':' and depth == 0 and colon == -1:
                colon = end
            end += 1
        cond = text[start + 1:q].strip()
        yes = text[q + 1:colon].strip()
        no = text[colon + 1:end].strip()
        text = text[:start] + ('((%s) if (%s) else (%s))' % (yes, cond, no)) + text[end + 1:]
    return bool(eval(text, {'__builtins__': {}}, env))


def run_one(line, env, *, sym=None, scc_verified=False):
    stage = Probe()
    if sym:
        stage.sym.update(sym)
    stage.scc_verified = scc_verified
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


PAIRS = (
    (0x00000000, 0x00000000),
    (0xffffffff, 0xffffffff),
    (0x00000001, 0x80000000),
    (0x80000000, 0x00000001),
    (0xffffffff, 0x00000000),
    (0x00000000, 0xffffffff),
    (0x55555555, 0xaaaaaaaa),
    (0xaaaaaaaa, 0x55555555),
    (0x00010020, 0x00000020),
)
INDEXES = (0, 1, 5, 31, 32, 33, 37, 63, 64, 65, 96, 127, 0xffffffe5)


def run_controls():
    # 1. Decode verification from synthetic gfx1013 SOPC bytes via G.decode().
    decoded = [
        G.decode(sopc(0x0e, 4, 6))[0][3],
        G.decode(sopc(0x0f, 4, 128 + 37))[0][3],
        G.decode(sopc(0x0f, 193, 6))[0][3],
        G.decode(sopc(0x0e, 255, 6, lit=0x80000001))[0][3],
        G.decode(sopc(0x0f, 125, 128))[0][3],
    ]
    check('decode s_bitcmp0_b64 reg/reg', decoded[0] == 's_bitcmp0_b64 s[4:5], s6')
    check('decode s_bitcmp1_b64 reg/imm37', decoded[1] == 's_bitcmp1_b64 s[4:5], 37')
    check('decode s_bitcmp1_b64 inline -1', decoded[2] == 's_bitcmp1_b64 -1, s6')
    check('decode s_bitcmp0_b64 literal', decoded[3] == 's_bitcmp0_b64 0x80000001, s6')
    check('decode s_bitcmp1_b64 null', decoded[4] == 's_bitcmp1_b64 null, 0')

    # 2. Exhaustive SGPR pair + register/immediate index grid.
    grid_ok = True
    for mnemonic, want in (('s_bitcmp0_b64', 0), ('s_bitcmp1_b64', 1)):
        for lo, hi in PAIRS:
            for idx in INDEXES:
                expected = ref_bitcmp64(want, lo, hi, idx)
                for line, env in (
                    ('%s s[4:5], s6' % mnemonic, {'s4': lo, 's5': hi, 's6': idx}),
                    ('%s s[4:5], %d' % (mnemonic, idx), {'s4': lo, 's5': hi}),
                ):
                    got, _ = run_one(line, env)
                    if got != expected:
                        grid_ok = False
                        print('MISMATCH %s lo=%#x hi=%#x idx=%#x got=%s want=%s' % (
                            line, lo, hi, idx, got, expected))
    check('SGPR pair grid (%d cases)' % (2 * len(PAIRS) * len(INDEXES) * 2), grid_ok)

    # 3. Inline integer 64-bit S0 (including negative sign-extension to 64 bits).
    inline_ok = True
    for mnemonic, want in (('s_bitcmp0_b64', 0), ('s_bitcmp1_b64', 1)):
        for val in (0, 1, 64, -1, -16):
            lo, hi = val & MASK, (val >> 32) & MASK
            for idx in (0, 3, 4, 6, 31, 32, 37, 63, 64):
                got, _ = run_one('%s %d, s6' % (mnemonic, val), {'s6': idx})
                if got != ref_bitcmp64(want, lo, hi, idx):
                    inline_ok = False
    check('inline integer S0 sign-extends negative values to 64 bits', inline_ok)

    # 4. 32-bit hex literal S0 zero-extends to 64 bits (hi == 0 even when bit 31 is set).
    lit_ok = True
    for mnemonic, want in (('s_bitcmp0_b64', 0), ('s_bitcmp1_b64', 1)):
        for idx in (0, 1, 31, 32, 33, 63):
            got, _ = run_one('%s 0x80000001, s6' % mnemonic, {'s6': idx})
            if got != ref_bitcmp64(want, 0x80000001, 0, idx):
                lit_ok = False
    check('32-bit hex literal S0 zero-extends to 64 bits', lit_ok)

    # 5. null as S0 and S1, src_scc as S1, and const-tracked SGPR pair.
    for mnemonic, want in (('s_bitcmp0_b64', 0), ('s_bitcmp1_b64', 1)):
        for idx in (0, 31, 32, 63):
            got, _ = run_one('%s null, s6' % mnemonic, {'s6': idx})
            check('%s null, s6 (idx=%d)' % (mnemonic, idx), got == (want == 0))
        got_null_idx, _ = run_one('%s s[4:5], null' % mnemonic, {'s4': 1, 's5': 0})
        check('%s s[4:5], null reads bit 0' % mnemonic, got_null_idx == (want == 1))
        for scc_val in (False, True):
            got_scc, _ = run_one('%s s[4:5], src_scc' % mnemonic,
                                 {'s4': 0x2, 's5': 0x0, 'scc': scc_val}, scc_verified=True)
            check('%s src_scc=%s' % (mnemonic, scc_val),
                  got_scc == ref_bitcmp64(want, 0x2, 0x0, 1 if scc_val else 0))
        got_const, _ = run_one('%s s[4:5], s6' % mnemonic, {'s5': 0x20, 's6': 37},
                               sym={4: ('const', 0)})
        check('%s const-tracked lo + runtime hi' % mnemonic, got_const == (want == 1))

    # 6. Uniformity tracking on lo, hi, and bit index.
    for mnemonic in ('s_bitcmp0_b64', 's_bitcmp1_b64'):
        s = Probe()
        s.varying_s = set()
        s.scc_uniform = False
        s.one(mnemonic, '%s s[4:5], s6' % mnemonic)
        check('%s uniform inputs -> scc_uniform=True' % mnemonic, s.scc_uniform is True)
        for bad_reg in (4, 5, 6):
            s = Probe()
            s.varying_s = {bad_reg}
            s.scc_uniform = True
            s.one(mnemonic, '%s s[4:5], s6' % mnemonic)
            check('%s varying s%d -> scc_uniform=False' % (mnemonic, bad_reg), s.scc_uniform is False)

    # 7. Refusals (arity, modifiers, lane masks, mask_high, invalid register classes/alignment).
    for mnemonic in ('s_bitcmp0_b64', 's_bitcmp1_b64'):
        check('%s 1 operand refused' % mnemonic,
              stage_refused(Probe(), mnemonic, '%s s[4:5]' % mnemonic, 'two unmodified operands'))
        check('%s 3 operands refused' % mnemonic,
              stage_refused(Probe(), mnemonic, '%s s[4:5], s6, s7' % mnemonic, 'two unmodified operands'))
        check('%s modifier refused' % mnemonic,
              stage_refused(Probe(), mnemonic, '%s s[4:5], s6 clamp' % mnemonic, 'two unmodified operands'))
        for operands in (
            'exec, 1', 'exec_lo, 1', 'vcc, 1', 'vcc_lo, 1',
            's[4:5], exec_lo', 's[4:5], vcc_lo', 's[4:5], exec', 's[4:5], vcc',
        ):
            check('%s %s lane mask refused' % (mnemonic, operands),
                  stage_refused(Probe(), mnemonic, '%s %s' % (mnemonic, operands), 'lane mask'))
        for mask_reg, high_reg, operands in (
            (12, None, 's[12:13], 1'),
            (13, None, 's[12:13], 1'),
            (None, 13, 's[12:13], 1'),
            (12, None, 's[4:5], s12'),
            (None, 13, 's[4:5], s13'),
        ):
            p = Probe()
            if mask_reg is not None:
                p.masks = {mask_reg}
            if high_reg is not None:
                p.mask_high = {high_reg}
            check('%s %s tracked mask refused' % (mnemonic, operands),
                  stage_refused(p, mnemonic, '%s %s' % (mnemonic, operands), 'lane mask'))
        for bad_s0 in ('v[0:1]', 's4', 's[5:6]', 's[106:107]', '1.0', 'm0', 'src_scc'):
            check('%s bad S0 %s refused' % (mnemonic, bad_s0),
                  stage_refused(Probe(), mnemonic, '%s %s, s6' % (mnemonic, bad_s0)))
        for bad_s1 in ('v0', 's[6:7]', 's106', '1.0'):
            check('%s bad S1 %s refused' % (mnemonic, bad_s1),
                  stage_refused(Probe(), mnemonic, '%s s[4:5], %s' % (mnemonic, bad_s1), 'bit index'))
        p_unv = Probe()
        p_unv.scc_verified = False
        check('%s unverified src_scc refused' % mnemonic,
              stage_refused(p_unv, mnemonic, '%s s[4:5], src_scc' % mnemonic, 'unverified SCC'))

    # 8. Full synthetic pixel pipeline translates cleanly through G.Pipeline().
    _, ps_hlsl = translate_ps()
    check('synthetic pixel pipeline translates', '& 32u' in ps_hlsl and '& 31u' in ps_hlsl)

    # 9. Unlowered control: removing handlers from Stage restores refusals.
    saved = {name: getattr(G.Stage, name) for name in ('op_s_bitcmp0_b64', 'op_s_bitcmp1_b64')}
    try:
        for name in saved:
            delattr(G.Stage, name)
        for mnemonic in ('s_bitcmp0_b64', 's_bitcmp1_b64'):
            check('unlowered %s refused' % mnemonic,
                  stage_refused(Probe(), mnemonic, '%s s[4:5], s6' % mnemonic))
        try:
            translate_ps()
        except G.Unsupported:
            check('unlowered synthetic pipeline refused', True)
        else:
            check('unlowered synthetic pipeline refused', False)
    finally:
        for name, fn in saved.items():
            setattr(G.Stage, name, fn)

    # 10. Mutation / negative controls: prove our test inputs distinguish common bugs.
    check('mutation: dropping bit 5 diverges on bit 37',
          ref_bitcmp64(1, 0x0, 0x20, 37) != ref_bitcmp64(1, 0x0, 0x20, 37 & 31))
    check('mutation: swapping lo/hi diverges on bit 0',
          ref_bitcmp64(1, 0x1, 0x0, 0) != ref_bitcmp64(1, 0x0, 0x1, 0))
    check('mutation: inverting bit0/bit1 diverges',
          ref_bitcmp64(0, 0x0, 0x20, 37) != ref_bitcmp64(1, 0x0, 0x20, 37))
    check('mutation: zero-extending -1 diverges on bit 37',
          ref_bitcmp64(1, 0xffffffff, 0xffffffff, 37) != ref_bitcmp64(1, 0xffffffff, 0x0, 37))
    check('mutation: sign-extending 0x80000001 diverges on bit 63',
          ref_bitcmp64(1, 0x80000001, 0x0, 63) != ref_bitcmp64(1, 0x80000001, 0xffffffff, 63))


GPU_VECTORS = (
    # (label, lo, hi, idx)
    ('lo_bit0_set',    0x00000001, 0x00000000, 0),
    ('lo_bit31_clear', 0x7fffffff, 0xffffffff, 31),
    ('hi_bit32_set',   0x00000000, 0x00000001, 32),
    ('hi_bit37_set',   0x00000000, 0x00000020, 37),
    ('hi_bit63_set',   0x00000000, 0x80000000, 63),
    ('wrapped_101',    0x00000000, 0x00000020, 101),  # 101 & 63 == 37
    ('wrapped_64',     0x00000001, 0x00000000, 64),   # 64 & 63 == 0
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory(prefix='x2w_s_bitcmp64_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            first_vs = None
            first_ps = None
            first_exp = None
            for label, lo, hi, idx in GPU_VECTORS:
                vs_hlsl, ps_hlsl = translate_ps(lo, hi, idx)
                vs_bin = compile_hlsl(compiler, kind, vs_hlsl, 'vs_%s_%s' % (label, kind), vs_profile, td)
                ps_bin = compile_hlsl(compiler, kind, ps_hlsl, 'ps_%s_%s' % (label, kind), ps_profile, td)
                raw_exp = (
                    (1 if ref_bitcmp64(0, lo, hi, idx) else 0) ^ DRAW_TRANSPORT_XOR,
                    (1 if ref_bitcmp64(1, lo, hi, idx) else 0) ^ DRAW_TRANSPORT_XOR,
                    (1 if ref_bitcmp64(1, lo, hi, 37) else 0) ^ DRAW_TRANSPORT_XOR,
                    (1 if ref_bitcmp64(1, 0xffffffff, 0xffffffff, idx) else 0) ^ DRAW_TRANSPORT_XOR,
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
        print('s_bitcmp64 controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('s_bitcmp64 draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_s_bitcmp64_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
