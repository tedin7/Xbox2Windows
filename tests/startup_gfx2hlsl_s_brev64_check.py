"""64-bit scalar bit-reverse (s_brev_b64) in tools/startup/gfx2hlsl.py.

  (no arguments)                       SOP1 decode, value evaluation, in-place aliasing, constant folding,
                                       uniformity, refusal and mutation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per 64-bit input and compare the four result words

RDNA 2 ISA, SOP1 (0x0c):
  S_BREV_B64 s[D:D+1], S0: D.u64[63:0] = S0.u64[0:63]; SCC unchanged.

The pixel fixture decodes synthetic gfx1013 bytes via G.decode() ONLY (no game data). It loads a 64-bit value
into runtime SGPR pairs s[4:5] and s[6:7], runs disjoint s_brev_b64 s[8:9], s[4:5] and in-place
s_brev_b64 s[6:7], s[6:7], XORs the four output words with 0x3f800000 for normal IEEE-754 transport into
v12..v15, and exports mrt0.
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

# Keep exported lanes in normal IEEE-754 float range on float render targets.
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


def ps_words(lo, hi):
    """Synthetic gfx1013 pixel stage exercising disjoint and in-place s_brev_b64."""
    body = b''.join([
        sop1_mov_lit(4, lo),
        sop1_mov_lit(5, hi),
        s_xor_lit(4, 4, 0),                   # clear const sym so s_brev_b64 emits runtime reversebits
        s_xor_lit(5, 5, 0),
        sop1_mov_lit(6, lo),
        sop1_mov_lit(7, hi),
        s_xor_lit(6, 6, 0),
        s_xor_lit(7, 7, 0),
        sop1(0x0c, 8, 4),                     # disjoint: s_brev_b64 s[8:9], s[4:5]
        sop1(0x0c, 6, 6),                     # in-place: s_brev_b64 s[6:7], s[6:7]
        s_xor_lit(8, 8, DRAW_TRANSPORT_XOR),
        s_xor_lit(9, 9, DRAW_TRANSPORT_XOR),
        s_xor_lit(6, 6, DRAW_TRANSPORT_XOR),
        s_xor_lit(7, 7, DRAW_TRANSPORT_XOR),
        v_mov_b32_from_sgpr(12, 8),
        v_mov_b32_from_sgpr(13, 9),
        v_mov_b32_from_sgpr(14, 6),
        v_mov_b32_from_sgpr(15, 7),
    ])
    return PRE_WORDS + body + POST_WORDS


def translate_ps(lo=0x80000001, hi=0x00000001):
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(lo, hi), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def ref_brev64(lo, hi):
    """Independent 64-bit bit-string reference returning (rev_lo, rev_hi)."""
    bits = format(((hi & MASK) << 32) | (lo & MASK), '064b')[::-1]
    val = int(bits, 2)
    return val & MASK, (val >> 32) & MASK


def _reversebits(v):
    return int('{:032b}'.format(v & MASK)[::-1], 2)


def eval_u32(expr, env):
    text = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', expr)
    return eval(text, {'__builtins__': {}, 'reversebits': _reversebits}, env) & MASK


def run_lines(line, env, *, sym=None):
    stage = Probe()
    if sym:
        stage.sym.update(sym)
    stage.one('s_brev_b64', line)
    out = dict(env)
    for stmt in stage.lines:
        body = stmt.strip()
        if body.startswith('{') and body.endswith('}'):
            parts = [p.strip() for p in body[1:-1].split(';') if p.strip()]
        else:
            parts = [body.rstrip(';')]
        for p in parts:
            m = re.fullmatch(r'(?:uint\s+)?(\w+)\s*=\s*(.*)', p)
            assert m, p
            assert m.group(1) != 'scc', 'SCC must remain unchanged: %s' % p
            out[m.group(1)] = eval_u32(m.group(2), out)
    return out, stage


def stage_refused(stage, line, needle=''):
    try:
        stage.one('s_brev_b64', line)
    except G.Unsupported as exc:
        return (not needle or needle in str(exc)) and not stage.lines
    return False


PAIRS = (
    (0x00000000, 0x00000000),
    (0xffffffff, 0xffffffff),
    (0x00000001, 0x00000000),
    (0x00000000, 0x80000000),
    (0x80000001, 0x00000001),
    (0x12345678, 0x9abcdef0),
    (0x55555555, 0xaaaaaaaa),
    (0x00ff00ff, 0xff00ff00),
)


def run_controls():
    # 1. Decode verification from synthetic gfx1013 SOP1 bytes via G.decode().
    decoded = [
        G.decode(sop1(0x0c, 8, 4))[0][3],
        G.decode(sop1(0x0c, 4, 4))[0][3],
        G.decode(sop1(0x0c, 4, 193))[0][3],
        G.decode(sop1(0x0c, 4, 255, lit=0x80000001))[0][3],
        G.decode(sop1(0x0c, 4, 125))[0][3],
    ]
    check('decode s_brev_b64 disjoint pair', decoded[0] == 's_brev_b64 s[8:9], s[4:5]')
    check('decode s_brev_b64 in-place pair', decoded[1] == 's_brev_b64 s[4:5], s[4:5]')
    check('decode s_brev_b64 inline -1', decoded[2] == 's_brev_b64 s[4:5], -1')
    check('decode s_brev_b64 literal', decoded[3] == 's_brev_b64 s[4:5], 0x80000001')
    check('decode s_brev_b64 null', decoded[4] == 's_brev_b64 s[4:5], null')

    # 2. Disjoint and in-place SGPR pair evaluation (SCC untouched).
    disjoint_ok = True
    inplace_ok = True
    for lo, hi in PAIRS:
        want = ref_brev64(lo, hi)
        out_d, _ = run_lines('s_brev_b64 s[8:9], s[4:5]', {'s4': lo, 's5': hi})
        if (out_d['s8'], out_d['s9']) != want:
            disjoint_ok = False
        out_i, _ = run_lines('s_brev_b64 s[4:5], s[4:5]', {'s4': lo, 's5': hi})
        if (out_i['s4'], out_i['s5']) != want:
            inplace_ok = False
    check('disjoint SGPR pair (%d cases)' % len(PAIRS), disjoint_ok)
    check('in-place SGPR pair aliasing (%d cases)' % len(PAIRS), inplace_ok)

    # 3. Inline integers (negative sign-extends to 64 bits), 32-bit hex literal (zero-extends), null.
    for val in (0, 1, 64, -1, -16):
        lo, hi = val & MASK, (val >> 32) & MASK
        want = ref_brev64(lo, hi)
        out, st = run_lines('s_brev_b64 s[4:5], %d' % val, {})
        check('inline %d folds to %#x, %#x' % (val, want[0], want[1]),
              (out['s4'], out['s5']) == want and st.sym.get(4) == ('const', want[0])
              and st.sym.get(5) == ('const', want[1]))
    want_lit = ref_brev64(0x80000001, 0)
    out_lit, st_lit = run_lines('s_brev_b64 s[4:5], 0x80000001', {})
    check('32-bit hex literal zero-extends to 64 bits',
          (out_lit['s4'], out_lit['s5']) == want_lit and st_lit.sym.get(4) == ('const', 0))
    out_null, st_null = run_lines('s_brev_b64 s[4:5], null', {})
    check('null folds to (0, 0)',
          (out_null['s4'], out_null['s5']) == (0, 0) and st_null.sym.get(4) == ('const', 0))

    # 4. Constant-tracked SGPR pair (both halves const, and mixed const + runtime).
    want_c = ref_brev64(0x12345678, 0x9abcdef0)
    out_c, st_c = run_lines('s_brev_b64 s[4:5], s[4:5]', {},
                            sym={4: ('const', 0x12345678), 5: ('const', 0x9abcdef0)})
    check('const-tracked in-place pair folds',
          (out_c['s4'], out_c['s5']) == want_c and st_c.sym.get(4) == ('const', want_c[0]))
    out_m, st_m = run_lines('s_brev_b64 s[8:9], s[4:5]', {'s5': 0x80000001}, sym={4: ('const', 0x1)})
    check('mixed const lo + runtime hi folds hi output',
          (out_m['s8'], out_m['s9']) == ref_brev64(0x1, 0x80000001) and st_m.sym.get(9) == ('const', 0x80000000))

    # 5. Uniformity tracking (swapping halves swaps uniformity accurately, both disjoint and in-place).
    s = Probe()
    s.varying_s = {4}
    s.scc_uniform = True
    s.one('s_brev_b64', 's_brev_b64 s[8:9], s[4:5]')
    check('disjoint: varying s4 -> s9 varying, s8 uniform', 8 not in s.varying_s and 9 in s.varying_s)

    s = Probe()
    s.varying_s = {4}
    s.scc_uniform = True
    s.one('s_brev_b64', 's_brev_b64 s[4:5], s[4:5]')
    check('in-place: varying s4 -> s5 varying, s4 uniform', 4 not in s.varying_s and 5 in s.varying_s)

    # 6. Refusals (arity, modifiers, lane masks, mask_high, descriptors, invalid register classes/alignment).
    check('1 operand refused', stage_refused(Probe(), 's_brev_b64 s[4:5]', 'expected 2 unmodified'))
    check('3 operands refused', stage_refused(Probe(), 's_brev_b64 s[4:5], s[6:7], s[8:9]', 'expected 2 unmodified'))
    check('modifier refused', stage_refused(Probe(), 's_brev_b64 s[4:5], s[6:7] clamp', 'expected 2 unmodified'))
    for operands in ('exec, s[6:7]', 'vcc, s[6:7]', 's[4:5], exec', 's[4:5], vcc'):
        check('%s lane mask refused' % operands, stage_refused(Probe(), 's_brev_b64 %s' % operands))
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
        check('%s tracked mask refused' % operands, stage_refused(p, 's_brev_b64 %s' % operands, 'lane mask'))
    for bad_dst in ('v[0:1]', 's4', 's[5:6]', 's[106:107]', 'm0'):
        check('bad dst %s refused' % bad_dst, stage_refused(Probe(), 's_brev_b64 %s, s[6:7]' % bad_dst, 'destination'))
    for bad_src in ('v[0:1]', 's6', 's[7:8]', 's[106:107]', '1.0', 'm0', 'src_scc'):
        check('bad src %s refused' % bad_src, stage_refused(Probe(), 's_brev_b64 s[4:5], %s' % bad_src))
    for sym_reg in (6, 7):
        p = Probe()
        p.sym[sym_reg] = ('descriptor', 0)
        check('descriptor sym on s%d refused' % sym_reg,
              stage_refused(p, 's_brev_b64 s[4:5], s[6:7]', 'descriptor data'))

    # 7. Full synthetic pixel pipeline translates cleanly through G.Pipeline().
    _, ps_hlsl = translate_ps()
    check('synthetic pixel pipeline translates', 'reversebits(' in ps_hlsl and '_t0' in ps_hlsl)

    # 8. Unlowered control: removing op_s_brev_b64 from Stage restores refusals.
    saved = G.Stage.op_s_brev_b64
    try:
        delattr(G.Stage, 'op_s_brev_b64')
        check('unlowered s_brev_b64 refused', stage_refused(Probe(), 's_brev_b64 s[8:9], s[4:5]'))
        try:
            translate_ps()
        except G.Unsupported:
            check('unlowered synthetic pipeline refused', True)
        else:
            check('unlowered synthetic pipeline refused', False)
    finally:
        G.Stage.op_s_brev_b64 = saved

    # 9. Mutation / negative controls.
    check('mutation: not swapping halves diverges', ref_brev64(0x1, 0x0) != (0x80000000, 0x0))
    check('mutation: zero-extending -1 diverges', ref_brev64(0xffffffff, 0xffffffff) != ref_brev64(0xffffffff, 0))


# Choose inputs with bits [8:1] clear so reversebits(word) has bits [30:23] == 0, keeping XOR 0x3f800000 normal.
GPU_VECTORS = (
    ('bit0_to_bit63',   0x00000001, 0x00000000),
    ('bit63_to_bit0',   0x00000000, 0x80000000),
    ('mixed_halves',    0x80000001, 0x12345001),
    ('high_bits_only',  0x00000000, 0xf0a00001),
    ('both_halves',     0x13579001, 0x2468a000),
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory(prefix='x2w_s_brev64_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            first_vs = None
            first_ps = None
            first_exp = None
            for label, lo, hi in GPU_VECTORS:
                vs_hlsl, ps_hlsl = translate_ps(lo, hi)
                vs_bin = compile_hlsl(compiler, kind, vs_hlsl, 'vs_%s_%s' % (label, kind), vs_profile, td)
                ps_bin = compile_hlsl(compiler, kind, ps_hlsl, 'ps_%s_%s' % (label, kind), ps_profile, td)
                r0, r1 = ref_brev64(lo, hi)
                raw_exp = (
                    r0 ^ DRAW_TRANSPORT_XOR,
                    r1 ^ DRAW_TRANSPORT_XOR,
                    r0 ^ DRAW_TRANSPORT_XOR,
                    r1 ^ DRAW_TRANSPORT_XOR,
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

            bad = words_text((first_exp[0] ^ 1, first_exp[1], first_exp[2], first_exp[3]))
            neg = draw(renderer, adapter, first_vs, first_ps, bad)
            check('draw %s %s negative control rejects wrong expectation' % (kind, adapter),
                  neg.returncode != 0)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run_controls()
        print('s_brev64 controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('s_brev64 draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_s_brev64_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
