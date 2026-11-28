"""VOP3P 2-halfword integer dot products (v_dot2_i32_i16, v_dot2_u32_u16) in tools/startup/gfx2hlsl.py.

  (no arguments)                       VOP3P decode, on-demand helpers, value evaluation (wrapping and clamp,
                                       including double-overflow cases), EXEC masking, refusal and mutation
                                       controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input triples and compare the four result words

RDNA 2 ISA, VOP3P (0x14 / 0x15):
  V_DOT2_I32_I16 vD, S0, S1, S2 [clamp]: D = int16(S0.lo)*int16(S1.lo) + int16(S0.hi)*int16(S1.hi) + int32(S2)
  V_DOT2_U32_U16 vD, S0, S1, S2 [clamp]: D = uint16(S0.lo)*uint16(S1.lo) + uint16(S0.hi)*uint16(S1.hi) + uint32(S2)
"""
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

OPCODES = {
    'v_dot2_i32_i16': 0x14,
    'v_dot2_u32_u16': 0x15,
}


def vop3p(op, vdst, src0, src1, src2, *, clamp=0, op_sel=0, op_sel_hi=0b111, neg_lo=0, neg_hi=0, lit=None):
    w0 = (0xcc000000
          | (op << 16)
          | (clamp << 15)
          | (((op_sel_hi >> 2) & 1) << 14)
          | ((op_sel & 0x7) << 11)
          | ((neg_hi & 0x7) << 8)
          | (vdst & 0xff))
    w1 = (((neg_lo & 0x7) << 29)
          | ((op_sel_hi & 0x3) << 27)
          | ((src2 & 0x1ff) << 18)
          | ((src1 & 0x1ff) << 9)
          | (src0 & 0x1ff))
    out = struct.pack('<2I', w0, w1)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


def ps_words(op, clamp=0):
    code = OPCODES[op]
    body = b''.join(
        vop3p(code, 12 + lane, 256 + lane, 256 + 4 + lane, 256 + 8 + lane, clamp=clamp)
        for lane in range(4)
    )
    return PRE_WORDS + body + POST_WORDS


def translate(op, clamp=0):
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(op, clamp), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def sext16(v):
    v &= 0xffff
    return v - 0x10000 if v >= 0x8000 else v


def sext32(v):
    v &= MASK
    return v - (1 << 32) if v >= (1 << 31) else v


def ref_dot2_u32_u16(a, b, c, clamp=False):
    total = (c & MASK) + ((a & 0xffff) * (b & 0xffff)) + (((a >> 16) & 0xffff) * ((b >> 16) & 0xffff))
    if clamp and total > MASK:
        return MASK
    return total & MASK


def ref_dot2_i32_i16(a, b, c, clamp=False):
    total = sext32(c) + (sext16(a) * sext16(b)) + (sext16(a >> 16) * sext16(b >> 16))
    if clamp:
        if total > 0x7fffffff:
            return 0x7fffffff
        if total < -0x80000000:
            return 0x80000000
    return total & MASK


def eval_dot2_u32_u16_hlsl(a, b, c, sat):
    a, b, c = a & MASK, b & MASK, c & MASK
    p0 = ((a & 0xffff) * (b & 0xffff)) & MASK
    p1 = (((a >> 16) & 0xffff) * ((b >> 16) & 0xffff)) & MASK
    r0 = (c + p0) & MASK
    carry = 1 if r0 < c else 0
    r1 = (r0 + p1) & MASK
    carry += 1 if r1 < r0 else 0
    return 0xffffffff if (sat and carry != 0) else r1


def eval_dot2_i32_i16_hlsl(a, b, c, sat):
    a, b, c = a & MASK, b & MASK, c & MASK
    p0 = sext16(a) * sext16(b)
    p1 = sext16(a >> 16) * sext16(b >> 16)
    lo = c
    hi = -1 if sext32(c) < 0 else 0
    t0 = (lo + (p0 & MASK)) & MASK
    hi += (-1 if p0 < 0 else 0) + (1 if t0 < lo else 0)
    lo = t0
    t1 = (lo + (p1 & MASK)) & MASK
    hi += (-1 if p1 < 0 else 0) + (1 if t1 < lo else 0)
    lo = t1
    if sat:
        if hi > 0 or (hi == 0 and (lo & 0x80000000) != 0):
            return 0x7fffffff
        if hi < -1 or (hi == -1 and (lo & 0x80000000) == 0):
            return 0x80000000
    return lo


def stage_refused(stage, op, line, needle=''):
    try:
        stage.one(op, line)
    except G.Unsupported as exc:
        return (not needle or needle in str(exc)) and not stage.lines
    return False


WORDS_A = (0x00000000, 0xffffffff, 0x80008000, 0x7fff7fff, 0x80007fff, 0x00010002, 0xffff0001)
WORDS_B = (0x00000000, 0xffffffff, 0x80008000, 0x7fff7fff, 0x7fff8000, 0x00030004)
ACCUMS = (0x00000000, 0x00000001, 0x40000000, 0x7fffffff, 0x80000000, 0xc0000000, 0xffffffff)


def run_controls():
    # 1. Decode verification from synthetic gfx1013 VOP3P bytes via G.decode().
    check('decode v_dot2_i32_i16 plain',
          G.decode(vop3p(0x14, 12, 256, 260, 264))[0][3] == 'v_dot2_i32_i16 v12, v0, v4, v8')
    check('decode v_dot2_i32_i16 clamp',
          G.decode(vop3p(0x14, 12, 256, 260, 264, clamp=1))[0][3] == 'v_dot2_i32_i16 v12, v0, v4, v8 clamp')
    check('decode v_dot2_u32_u16 plain',
          G.decode(vop3p(0x15, 12, 256, 260, 264))[0][3] == 'v_dot2_u32_u16 v12, v0, v4, v8')
    check('decode v_dot2_u32_u16 clamp',
          G.decode(vop3p(0x15, 12, 256, 260, 264, clamp=1))[0][3] == 'v_dot2_u32_u16 v12, v0, v4, v8 clamp')

    # 2. Exhaustive grid for both wrapping and saturating modes (including 2*2^30 = 2^31 signed & 2*0xfffe0001 unsigned).
    u_ok = True
    i_ok = True
    for a in WORDS_A:
        for b in WORDS_B:
            for c in ACCUMS:
                for sat in (False, True):
                    if eval_dot2_u32_u16_hlsl(a, b, c, sat) != ref_dot2_u32_u16(a, b, c, sat):
                        u_ok = False
                    if eval_dot2_i32_i16_hlsl(a, b, c, sat) != ref_dot2_i32_i16(a, b, c, sat):
                        i_ok = False
    n_cases = len(WORDS_A) * len(WORDS_B) * len(ACCUMS) * 2
    check('v_dot2_u32_u16 grid (%d cases, wrap + clamp)' % n_cases, u_ok)
    check('v_dot2_i32_i16 grid (%d cases, wrap + clamp)' % n_cases, i_ok)

    # 3. On-demand helper emission: each helper emitted only when its instruction is used.
    _, ps_u = translate('v_dot2_u32_u16', 0)
    _, ps_i = translate('v_dot2_i32_i16', 0)
    check('dot2_u32_u16 emitted on demand and dot2_i32_i16 omitted',
          'uint dot2_u32_u16(' in ps_u and 'dot2_i32_i16' not in ps_u)
    check('dot2_i32_i16 emitted on demand and dot2_u32_u16 omitted',
          'uint dot2_i32_i16(' in ps_i and 'dot2_u32_u16' not in ps_i)
    check('dot2_i32_i16 helper uses strict unsigned carry compares',
          '((t0 < lo) ? 1 : 0)' in ps_i and '((t1 < lo) ? 1 : 0)' in ps_i)
    zero = [(0, 0, 0)] * 4
    unused_ps = G.Pipeline(vs_words(zero), PRE_WORDS + POST_WORDS, root_blob()).hlsl_ps()
    check('both dot2 helpers omitted when unused', 'dot2_' not in unused_ps)

    # 4. EXEC masking & clamp flag propagation.
    for op, helper in (('v_dot2_u32_u16', 'dot2_u32_u16'), ('v_dot2_i32_i16', 'dot2_i32_i16')):
        p = Probe()
        p.exec_known = False
        p.one(op, '%s v12, v0, v4, v8 clamp' % op)
        check('%s exec-masked + clamp' % op,
              p.lines == ['if (exec & 1) v12 = %s(v0, v4, v8, true);' % helper])

    # 5. Refusals: arity, non-clamp VOP3P modifiers, bad dst/src, source modifiers.
    for op, code in OPCODES.items():
        check('%s 3 operands refused' % op, stage_refused(Probe(), op, '%s v1, v2, v3' % op, 'expected 4'))
        check('%s 5 operands refused' % op, stage_refused(Probe(), op, '%s v1, v2, v3, v4, v5' % op, 'expected 4'))
        for label, bad_word in (
            ('op_sel', vop3p(code, 12, 256, 260, 264, op_sel=1)),
            ('op_sel_hi', vop3p(code, 12, 256, 260, 264, op_sel_hi=0)),
            ('neg_lo', vop3p(code, 12, 256, 260, 264, neg_lo=1)),
            ('neg_hi', vop3p(code, 12, 256, 260, 264, neg_hi=1)),
        ):
            asm = G.decode(bad_word)[0][3]
            check('%s VOP3P %s refused' % (op, label),
                  stage_refused(Probe(), op, asm, 'unsupported modifiers'))
        for bad_dst in ('s4', 'v[0:1]', 'v256'):
            check('%s bad dst %s refused' % (op, bad_dst),
                  stage_refused(Probe(), op, '%s %s, v0, v4, v8' % (op, bad_dst), 'destination'))
        for bad_src in ('|v0|', '-v0', 'neg(1)', 'v[0:1]', 's[0:1]', 'v256', 's106'):
            check('%s bad src %s refused' % (op, bad_src),
                  stage_refused(Probe(), op, '%s v12, %s, v4, v8' % (op, bad_src)))

    # 6. Unlowered control: removing handlers from Stage restores refusals.
    saved = {name: getattr(G.Stage, name) for name in ('op_v_dot2_i32_i16', 'op_v_dot2_u32_u16')}
    try:
        for name in saved:
            delattr(G.Stage, name)
        for op in OPCODES:
            check('unlowered %s refused' % op, stage_refused(Probe(), op, '%s v12, v0, v4, v8' % op))
            try:
                translate(op)
            except G.Unsupported:
                check('unlowered %s pipeline refused' % op, True)
            else:
                check('unlowered %s pipeline refused' % op, False)
    finally:
        for name, fn in saved.items():
            setattr(G.Stage, name, fn)


DRAW_CASES = (
    ('v_dot2_u32_u16', 0, [
        (0x00020003, 0x00040005, 0x3f800000),  # 3*5 + 2*4 = +23
        (0x00100020, 0x01000200, 0x3f800000),  # 32*512 + 16*256 = +20480
        (0x00010001, 0x00010001, 0xfffffffe),  # wraps cleanly to 0x00000000
        (0x00000000, 0xffffffff, 0x40000000),  # zero sum
    ]),
    ('v_dot2_u32_u16', 1, [
        (0x00020003, 0x00040005, 0x3f800000),  # no saturation
        (0xffffffff, 0xffffffff, 0x3f800000),  # double overflow -> saturates to 0xffffffff
        (0x80008000, 0x80008000, 0x80000000),  # overflow -> saturates to 0xffffffff
        (0x00000000, 0xffffffff, 0x3f800000),  # zero sum
    ]),
    ('v_dot2_i32_i16', 0, [
        (0x00020003, 0x00040005, 0x3f800000),  # +23
        (0xfffefffd, 0x00040005, 0x3f800000),  # (-3)*5 + (-2)*4 = -23
        (0x80008000, 0x80008000, 0x80000000),  # 2^30 + 2^30 - 2^31 = 0x00000000
        (0x00010000, 0xffff0000, 0x3f800000),  # -1
    ]),
    ('v_dot2_i32_i16', 1, [
        (0x80008000, 0x80008000, 0x00000000),  # 2^30 + 2^30 = +2^31 -> saturates to 0x7fffffff
        (0x80008000, 0x7fff7fff, 0x80000000),  # negative overflow -> saturates to 0x80000000
        (0x80008000, 0x80008000, 0x80000000),  # exact cancellation: 2^31 - 2^31 = 0x00000000 (NO sat!)
        (0x00020003, 0x00040005, 0x3f800000),  # no overflow
    ]),
    ('v_dot2_i32_i16', 1, [
        (0x00000003, 0x00010005, 0x3f800000),  # p1 == 0: no carry, no saturation
        (0x00030000, 0x00050001, 0xffffffff),  # p0 == 0, c < 0: +14
        (0x00000000, 0x12345678, 0x7fffffff),  # both products 0, c = INT32_MAX
        (0x00000000, 0x00000000, 0x80000000),  # both products 0, c = INT32_MIN
    ]),
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory(prefix='x2w_v_dot2_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            first_vs = None
            first_ps = None
            first_exp = None
            for op, clamp, triples in DRAW_CASES:
                ref_fn = ref_dot2_i32_i16 if op == 'v_dot2_i32_i16' else ref_dot2_u32_u16
                _, ps_hlsl = translate(op, clamp)
                ps_bin = compile_hlsl(compiler, kind, ps_hlsl, 'ps_%s_%d_%s' % (op, clamp, kind), ps_profile, td)
                vs_hlsl = G.Pipeline(vs_words(triples), ps_words(op, clamp), root_blob()).hlsl_vs()
                vs_bin = compile_hlsl(compiler, kind, vs_hlsl, 'vs_%s_%d_%s' % (op, clamp, kind), vs_profile, td)
                raw_exp = tuple(ref_fn(a, b, c, bool(clamp)) for a, b, c in triples)
                expected = words_text(raw_exp)
                res = draw(renderer, adapter, vs_bin, ps_bin, expected)
                if res.returncode == 77:
                    print('SKIP (%s): %s' % (adapter, res.stdout.strip() or res.stderr.strip()))
                    sys.exit(77)
                if res.returncode != 0:
                    raise AssertionError('%s %s %s clamp=%d expected=%s\n%s\n%s' % (
                        kind, adapter, op, clamp, expected, res.stdout, res.stderr))
                print('%s %s %s clamp=%d: %s' % (kind, adapter, op, clamp, expected), flush=True)
                if first_vs is None:
                    first_vs, first_ps, first_exp = vs_bin, ps_bin, raw_exp

            bad = words_text((first_exp[0] ^ 1, first_exp[1], first_exp[2], first_exp[3]))
            neg = draw(renderer, adapter, first_vs, first_ps, bad)
            check('draw %s %s negative control rejects wrong expectation' % (kind, adapter),
                  neg.returncode != 0)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run_controls()
        print('v_dot2 controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('v_dot2 draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_v_dot2_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
