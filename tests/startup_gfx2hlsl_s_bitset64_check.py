"""64-bit scalar bit set/clear (s_bitset0_b64, s_bitset1_b64) in tools/startup/gfx2hlsl.py.

  (no arguments)                       SOP1 decode, value evaluation, aliasing, constant folding, uniformity,
                                       refusal and mutation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per (lo, hi, index) triple and compare the four result words

RDNA 2 ISA, SOP1:
  S_BITSET0_B64 s[D:D+1], S0: D.u64[S0.u[5:0]] = 0; SCC unchanged.
  S_BITSET1_B64 s[D:D+1], S0: D.u64[S0.u[5:0]] = 1; SCC unchanged.

The pixel fixture decodes synthetic gfx1013 bytes via G.decode() ONLY (no game data). It loads (lo, hi, index)
into s[4:5]/s6 and s[8:9]/s6 with SOP1 literal moves, runs s_bitset0_b64 on s[4:5] and s_bitset1_b64 on s[8:9],
XORs the resulting four 32-bit words with 0x3f800000 for normal IEEE-754 transport into v12..v15, and exports mrt0.
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

SOP1_OPCODES = {
    's_bitset0_b64': 0x1c,
    's_bitset1_b64': 0x1e,
}

# Keep exported lanes away from NaN/subnormal bit patterns on float render targets.
DRAW_TRANSPORT_XOR = 0x3f800000


def sop1(op, sdst, ssrc0, lit=None):
    """Encode one synthetic gfx1013 SOP1 32-bit (or 64-bit with literal) instruction."""
    w0 = (0b101111101 << 23) | (sdst << 16) | (op << 8) | ssrc0
    out = struct.pack('<I', w0)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


def sop1_mov_lit(sdst, val):
    return sop1(0x03, sdst, 255, lit=val)


def s_xor_lit(sdst, ssrc0, lit):
    return struct.pack('<2I', 0x80000000 | (0x12 << 23) | (sdst << 16) | (255 << 8) | ssrc0, lit & MASK)


def v_mov_b32_from_sgpr(vdst, ssrc):
    return struct.pack('<I', 0x7e000000 | (vdst << 17) | (0x01 << 9) | ssrc)


def ps_words(lo, hi, idx):
    """Synthetic gfx1013 pixel stage running s_bitset0_b64 and s_bitset1_b64 on dynamic SGPRs."""
    # Use a non-const SGPR for idx by xoring two literals into s6 so the translator emits dynamic HLSL.
    body = b''.join([
        sop1_mov_lit(4, lo),
        sop1_mov_lit(5, hi),
        sop1_mov_lit(8, lo),
        sop1_mov_lit(9, hi),
        sop1_mov_lit(6, idx),
        s_xor_lit(6, 6, 0),                   # keeps s6 runtime (clears const sym)
        sop1(0x1c, 4, 6),                     # s_bitset0_b64 s[4:5], s6
        sop1(0x1e, 8, 6),                     # s_bitset1_b64 s[8:9], s6
        s_xor_lit(4, 4, DRAW_TRANSPORT_XOR),
        s_xor_lit(5, 5, DRAW_TRANSPORT_XOR),
        s_xor_lit(8, 8, DRAW_TRANSPORT_XOR),
        s_xor_lit(9, 9, DRAW_TRANSPORT_XOR),
        v_mov_b32_from_sgpr(12, 4),
        v_mov_b32_from_sgpr(13, 5),
        v_mov_b32_from_sgpr(14, 8),
        v_mov_b32_from_sgpr(15, 9),
    ])
    return PRE_WORDS + body + POST_WORDS


def translate_ps(lo=0x00000001, hi=0x00000020, idx=37):
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(lo, hi, idx), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def ref_bitset64(clear, lo, hi, idx):
    """Independent 64-bit integer reference returning (new_lo, new_hi)."""
    val = ((hi & MASK) << 32) | (lo & MASK)
    bit = 1 << (idx & 63)
    res = (val & ~bit) if clear else (val | bit)
    return res & MASK, (res >> 32) & MASK


def eval_u32_expr(expr, env):
    """Evaluate a translated HLSL uint expression (with optional ternaries) in Python."""
    text = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', expr)
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
    return eval(text, {'__builtins__': {}}, env) & MASK


def run_lines(line, env, *, sym=None):
    stage = Probe()
    if sym:
        stage.sym.update(sym)
    stage.one(line.split(' ', 1)[0], line)
    out = dict(env)
    for stmt in stage.lines:
        m = re.fullmatch(r'(\w+) = (.*);', stmt)
        assert m, stmt
        assert m.group(1) != 'scc', 'SCC must remain unchanged: %s' % stmt
        out[m.group(1)] = eval_u32_expr(m.group(2), out)
    return out, stage


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
    (0x12345678, 0x9abcdef0),
)
INDEXES = (0, 1, 5, 31, 32, 33, 37, 63, 64, 65, 96, 127, 0xffffffe5)


def run_controls():
    # 1. Decode verification from synthetic gfx1013 SOP1 bytes via G.decode().
    decoded = [
        G.decode(sop1(0x1c, 4, 6))[0][3],
        G.decode(sop1(0x1e, 4, 128 + 37))[0][3],
        G.decode(sop1(0x1c, 4, 255, lit=0x25))[0][3],
        G.decode(sop1(0x1e, 4, 125))[0][3],
    ]
    check('decode s_bitset0_b64 reg/reg', decoded[0] == 's_bitset0_b64 s[4:5], s6')
    check('decode s_bitset1_b64 reg/imm37', decoded[1] == 's_bitset1_b64 s[4:5], 37')
    check('decode s_bitset0_b64 literal', decoded[2] == 's_bitset0_b64 s[4:5], 0x25')
    check('decode s_bitset1_b64 null', decoded[3] == 's_bitset1_b64 s[4:5], null')

    # 2. Exhaustive SGPR pair + register/immediate index grid (SCC untouched).
    grid_ok = True
    for mnemonic, clear in (('s_bitset0_b64', True), ('s_bitset1_b64', False)):
        for lo, hi in PAIRS:
            for idx in INDEXES:
                want_lo, want_hi = ref_bitset64(clear, lo, hi, idx)
                for line, env in (
                    ('%s s[4:5], s6' % mnemonic, {'s4': lo, 's5': hi, 's6': idx}),
                    ('%s s[4:5], %d' % (mnemonic, idx), {'s4': lo, 's5': hi}),
                ):
                    out, _ = run_lines(line, env)
                    if (out['s4'], out['s5']) != (want_lo, want_hi):
                        grid_ok = False
                        print('MISMATCH %s lo=%#x hi=%#x idx=%#x got=(%#x,%#x) want=(%#x,%#x)' % (
                            line, lo, hi, idx, out['s4'], out['s5'], want_lo, want_hi))
    check('SGPR pair grid (%d cases)' % (2 * len(PAIRS) * len(INDEXES) * 2), grid_ok)

    # 3. Aliasing: S0 == s4 (low word of destination pair) and S0 == s5 (high word of destination pair).
    alias_ok = True
    for mnemonic, clear in (('s_bitset0_b64', True), ('s_bitset1_b64', False)):
        for lo, hi in ((0, 0), (35, 0), (35, 0xffffffff), (5, 0x12345678), (63, 0x80000000), (0, 37), (0xffffffff, 5)):
            out_lo, _ = run_lines('%s s[4:5], s4' % mnemonic, {'s4': lo, 's5': hi})
            if (out_lo['s4'], out_lo['s5']) != ref_bitset64(clear, lo, hi, lo):
                alias_ok = False
            out_hi, _ = run_lines('%s s[4:5], s5' % mnemonic, {'s4': lo, 's5': hi})
            if (out_hi['s4'], out_hi['s5']) != ref_bitset64(clear, lo, hi, hi):
                alias_ok = False
    check('aliasing s[4:5], s4 and s[4:5], s5 preserve original index', alias_ok)

    # 4. Constant folding (imm, null, and const-tracked SGPR index) leaves the untouched half's sym intact.
    for mnemonic, clear in (('s_bitset0_b64', True), ('s_bitset1_b64', False)):
        out_lo, st_lo = run_lines('%s s[4:5], 5' % mnemonic, {'s4': 0x20, 's5': 0x77},
                                  sym={4: ('const', 0x20), 5: ('const', 0x77)})
        want_lo, want_hi = ref_bitset64(clear, 0x20, 0x77, 5)
        check('%s folds const low word (bit 5)' % mnemonic,
              st_lo.sym.get(4) == ('const', want_lo) and st_lo.sym.get(5) == ('const', 0x77)
              and out_lo['s4'] == want_lo and len(st_lo.lines) == 1)
        out_hi, st_hi = run_lines('%s s[4:5], 37' % mnemonic, {'s4': 0x77, 's5': 0x20},
                                  sym={4: ('const', 0x77), 5: ('const', 0x20)})
        want_lo, want_hi = ref_bitset64(clear, 0x77, 0x20, 37)
        check('%s folds const high word (bit 37)' % mnemonic,
              st_hi.sym.get(4) == ('const', 0x77) and st_hi.sym.get(5) == ('const', want_hi)
              and out_hi['s5'] == want_hi and len(st_hi.lines) == 1)
        out_null, _ = run_lines('%s s[4:5], null' % mnemonic, {'s4': 1, 's5': 0x10})
        check('%s null targets bit 0' % mnemonic, (out_null['s4'], out_null['s5']) == ref_bitset64(clear, 1, 0x10, 0))
        out_cs, st_cs = run_lines('%s s[4:5], s6' % mnemonic, {'s4': 0, 's5': 0},
                                  sym={4: ('const', 0), 5: ('const', 0), 6: ('const', 63)})
        want_lo, want_hi = ref_bitset64(clear, 0, 0, 63)
        check('%s folds const SGPR index 63' % mnemonic,
              st_cs.sym.get(5) == ('const', want_hi) and out_cs['s5'] == want_hi)

    # 5. Uniformity tracking (varying index taints both halves; varying half stays varying).
    for mnemonic in ('s_bitset0_b64', 's_bitset1_b64'):
        s = Probe()
        s.varying_s = {6}
        s.scc_uniform = True
        s.one(mnemonic, '%s s[4:5], s6' % mnemonic)
        check('%s varying index taints s4 and s5' % mnemonic, 4 in s.varying_s and 5 in s.varying_s)
        s = Probe()
        s.varying_s = {4}
        s.scc_uniform = True
        s.one(mnemonic, '%s s[4:5], 37' % mnemonic)
        check('%s immediate bit 37 keeps s5 uniform when only s4 varies' % mnemonic,
              4 in s.varying_s and 5 not in s.varying_s)

    # 6. Refusals (arity, modifiers, lane masks, mask_high, descriptors, invalid register classes/alignment).
    for mnemonic in ('s_bitset0_b64', 's_bitset1_b64'):
        check('%s 1 operand refused' % mnemonic,
              stage_refused(Probe(), mnemonic, '%s s[4:5]' % mnemonic, 'expected 2 unmodified'))
        check('%s 3 operands refused' % mnemonic,
              stage_refused(Probe(), mnemonic, '%s s[4:5], s6, s7' % mnemonic, 'expected 2 unmodified'))
        check('%s modifier refused' % mnemonic,
              stage_refused(Probe(), mnemonic, '%s s[4:5], s6 clamp' % mnemonic, 'expected 2 unmodified'))
        for operands in ('exec, 1', 'vcc, 1', 's[4:5], exec_lo', 's[4:5], vcc_lo'):
            check('%s %s lane mask refused' % (mnemonic, operands),
                  stage_refused(Probe(), mnemonic, '%s %s' % (mnemonic, operands)))
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
        for bad_dst in ('v[0:1]', 's4', 's[5:6]', 's[106:107]', 'm0'):
            check('%s bad dst %s refused' % (mnemonic, bad_dst),
                  stage_refused(Probe(), mnemonic, '%s %s, s6' % (mnemonic, bad_dst), 'destination'))
        for bad_src in ('v0', 's[6:7]', 's106', '1.0', 'm0', 'src_scc'):
            check('%s bad src %s refused' % (mnemonic, bad_src),
                  stage_refused(Probe(), mnemonic, '%s s[4:5], %s' % (mnemonic, bad_src), 'source'))
        for sym_reg, operands in ((4, 's[4:5], 1'), (5, 's[4:5], 1'), (6, 's[4:5], s6')):
            p = Probe()
            p.sym[sym_reg] = ('descriptor', 0)
            check('%s descriptor sym on s%d refused' % (mnemonic, sym_reg),
                  stage_refused(p, mnemonic, '%s %s' % (mnemonic, operands), 'descriptor data'))

    # 7. Full synthetic pixel pipeline translates cleanly through G.Pipeline().
    _, ps_hlsl = translate_ps()
    check('synthetic pixel pipeline translates', '& 32u' in ps_hlsl and '& 31u' in ps_hlsl)

    # 8. Unlowered control: removing handlers from Stage restores refusals.
    saved = {name: getattr(G.Stage, name) for name in ('op_s_bitset0_b64', 'op_s_bitset1_b64')}
    try:
        for name in saved:
            delattr(G.Stage, name)
        for mnemonic in ('s_bitset0_b64', 's_bitset1_b64'):
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

    # 9. Mutation / negative controls.
    check('mutation: dropping bit 5 diverges on bit 37',
          ref_bitset64(False, 0, 0, 37) != ref_bitset64(False, 0, 0, 37 & 31))
    check('mutation: swapping clear/set diverges',
          ref_bitset64(True, 0, 0x20, 37) != ref_bitset64(False, 0, 0x20, 37))


GPU_VECTORS = (
    # Use small low/high patterns whose XOR with 0x3f800000 stays a normal IEEE-754 float (exponent 0x7f).
    # (label, lo, hi, idx)
    ('lo_bit0',    0x00000001, 0x00000000, 0),
    ('lo_bit20',   0x00100000, 0x00000000, 20),
    ('hi_bit32',   0x00000000, 0x00000001, 32),
    ('hi_bit37',   0x00000000, 0x00000020, 37),
    ('hi_bit52',   0x00000000, 0x00100000, 52),
    ('wrap_101',   0x00000000, 0x00000020, 101),  # 101 & 63 == 37
    ('wrap_64',    0x00000001, 0x00000000, 64),   # 64 & 63 == 0
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory(prefix='x2w_s_bitset64_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            first_vs = None
            first_ps = None
            first_exp = None
            for label, lo, hi, idx in GPU_VECTORS:
                vs_hlsl, ps_hlsl = translate_ps(lo, hi, idx)
                vs_bin = compile_hlsl(compiler, kind, vs_hlsl, 'vs_%s_%s' % (label, kind), vs_profile, td)
                ps_bin = compile_hlsl(compiler, kind, ps_hlsl, 'ps_%s_%s' % (label, kind), ps_profile, td)
                c0, c1 = ref_bitset64(True, lo, hi, idx)
                s0, s1 = ref_bitset64(False, lo, hi, idx)
                raw_exp = (
                    c0 ^ DRAW_TRANSPORT_XOR,
                    c1 ^ DRAW_TRANSPORT_XOR,
                    s0 ^ DRAW_TRANSPORT_XOR,
                    s1 ^ DRAW_TRANSPORT_XOR,
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
        print('s_bitset64 controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('s_bitset64 draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_s_bitset64_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
