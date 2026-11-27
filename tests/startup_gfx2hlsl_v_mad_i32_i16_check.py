"""Signed 16x16+32 multiply-add (v_mad_i32_i16) in tools/startup/gfx2hlsl.py.

  (no arguments)                       VOP3A decode, value evaluation, low16_only_v compatibility, EXEC masking,
                                       encoding & Stage refusals, unlowered and mutation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input triples and compare the four result words

RDNA 2 ISA, VOP3A (0x375):
  V_MAD_I32_I16 vD, S0, S1, S2: D = int32(int16(S0[15:0]) * int16(S1[15:0])) + int32(S2) (mod 2^32).
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


def vop3a(op, vdst, src0, src1, src2, *, abs_bits=0, op_sel=0, clamp=0, omod=0, neg=0, lit=None):
    w0 = 0xd4000000 | ((op & 0x3ff) << 16) | (clamp << 15) | (op_sel << 11) | (abs_bits << 8) | (vdst & 0xff)
    w1 = (neg << 29) | (omod << 27) | ((src2 & 0x1ff) << 18) | ((src1 & 0x1ff) << 9) | (src0 & 0x1ff)
    out = struct.pack('<2I', w0, w1)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


def ps_words():
    body = b''.join(vop3a(0x375, 12 + lane, 256 + lane, 256 + 4 + lane, 256 + 8 + lane) for lane in range(4))
    return PRE_WORDS + body + POST_WORDS


def translate():
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def sext16(v):
    v &= 0xffff
    return v - 0x10000 if v >= 0x8000 else v


def ref_mad_i32_i16(a, b, c):
    return (sext16(a) * sext16(b) + (c & MASK)) & MASK


def stage_refused(stage, line, needle=''):
    try:
        stage.one('v_mad_i32_i16', line)
    except G.Unsupported as exc:
        return (not needle or needle in str(exc)) and not stage.lines
    return False


WORDS_A = (0x00000000, 0xdead0001, 0xbeef7fff, 0x12348000, 0x5678ffff, 0x0000fffe)
WORDS_B = (0x00000000, 0xcafe0001, 0xbabe7fff, 0x9abc8000, 0xdef0ffff, 0x00000002)
ACCUMS = (0x00000000, 0x00000001, 0x3f800000, 0x7fffffff, 0x80000000, 0xffffffff)


def eval_emitted_u32(expr, env):
    """Evaluate a 32-bit HLSL uint/int expression (with asint/asuint, shifts, +, *) in Python."""
    import ast

    def to_i32(x):
        x &= MASK
        return x - 0x100000000 if x >= 0x80000000 else x

    def to_u32(x):
        return x & MASK

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            arg = ev(node.args[0])
            if node.func.id == 'asint':
                return to_i32(arg)
            if node.func.id == 'asuint':
                return to_u32(arg)
            raise ValueError(f'unsupported call {node.func.id}')
        if isinstance(node, ast.BinOp):
            l, r = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.LShift):
                return to_u32(l << (r & 31))
            if isinstance(node.op, ast.RShift):
                return to_i32(l) >> (r & 31) if l < 0 or isinstance(node.left, ast.Call) and getattr(node.left.func, 'id', '') == 'asint' else to_u32(l) >> (r & 31)
            if isinstance(node.op, ast.Mult):
                return to_u32(l * r)
            if isinstance(node.op, ast.Add):
                return to_u32(l + r)
            if isinstance(node.op, ast.BitAnd):
                return to_u32(l & r)
            raise ValueError(f'unsupported op {node.op}')
        raise ValueError(f'unsupported node {ast.dump(node)}')

    import re
    cleaned = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', expr)
    return to_u32(ev(ast.parse(cleaned, mode='eval')))


def run_controls():
    # 1. Decode verification from synthetic gfx1013 VOP3A bytes via G.decode().
    check('decode v_mad_i32_i16',
          G.decode(vop3a(0x375, 12, 256, 260, 264))[0][3] == 'v_mad_i32_i16 v12, v0, v4, v8')

    # 2. Exhaustive grid against the emitted HLSL expression (upper 16 bits of S0/S1 ignored; signed 16-bit).
    p_expr = Probe()
    p_expr.one('v_mad_i32_i16', 'v_mad_i32_i16 v12, v0, v4, v8')
    check('emitted v_mad_i32_i16 assignment',
          len(p_expr.lines) == 1 and p_expr.lines[0].startswith('v12 = ') and p_expr.lines[0].endswith(';'))
    emitted_rhs = p_expr.lines[0][len('v12 = '):-1]
    grid_ok = True
    for a in WORDS_A:
        for b in WORDS_B:
            for c in ACCUMS:
                got = eval_emitted_u32(emitted_rhs, {'v0': a & MASK, 'v4': b & MASK, 'v8': c & MASK})
                if got != ref_mad_i32_i16(a, b, c):
                    grid_ok = False
    check('v_mad_i32_i16 grid (%d cases)' % (len(WORDS_A) * len(WORDS_B) * len(ACCUMS)), grid_ok)

    # 3. Pipeline translates; EXEC masking works; low16_only_v accepted for S0/S1 and refused for S2.
    _, ps_hlsl = translate()
    check('pipeline translates', '<< 16) >> 16' in ps_hlsl)
    p = Probe()
    p.exec_known = False
    p.one('v_mad_i32_i16', 'v_mad_i32_i16 v12, v0, v4, v8')
    check('exec-masked v_mad_i32_i16 guards assignment',
          len(p.lines) == 1 and p.lines[0].startswith('if (exec & 1) v12 = '))

    p_lo = Probe()
    p_lo.low16_only_v = {0, 4}
    p_lo.one('v_mad_i32_i16', 'v_mad_i32_i16 v12, v0, v4, v8')
    check('low16_only_v accepted on S0 and S1', len(p_lo.lines) == 1)
    p_s2 = Probe()
    p_s2.low16_only_v = {8}
    check('low16_only_v refused on S2 (32-bit addend)',
          stage_refused(p_s2, 'v_mad_i32_i16 v12, v0, v4, v8', 'whole-word read'))

    # 4. VOP3A encoding guard refuses neg/abs.
    for label, bad_word in (
        ('abs0', vop3a(0x375, 12, 256, 260, 264, abs_bits=1)),
        ('neg2', vop3a(0x375, 12, 256, 260, 264, neg=4)),
        ('neg_inline0', vop3a(0x375, 12, 129, 260, 264, neg=1)),
    ):
        reasons = []
        G.decode(bad_word, reasons)
        check('encoding guard refuses %s' % label,
              len(reasons) == 1 and 'with neg/abs' in reasons[0][0])

    # 5. Stage refusals: modifiers, abs/neg, wrong arity.
    for asm in (
        'v_mad_i32_i16 v12, v0, v4, v8 clamp',
        'v_mad_i32_i16 v12, v0, v4, v8 mul:2',
        'v_mad_i32_i16 v12, v0, v4, v8 op_sel:[1,0,0]',
        'v_mad_i32_i16 v12, |v0|, v4, v8',
        'v_mad_i32_i16 v12, v0, -v4, v8',
        'v_mad_i32_i16 v12, v0, v4',
        'v_mad_i32_i16 v12, v0, v4, v8, v9',
    ):
        check('Stage refuses %s' % asm, stage_refused(Probe(), asm))

    # 6. Unlowered control: removing v_mad_i32_i16 from VEC_U restores refusals.
    saved = G.VEC_U.pop('v_mad_i32_i16')
    try:
        check('unlowered v_mad_i32_i16 refused', stage_refused(Probe(), 'v_mad_i32_i16 v12, v0, v4, v8'))
        try:
            translate()
        except G.Unsupported:
            check('unlowered pipeline refused', True)
        else:
            check('unlowered pipeline refused', False)
    finally:
        G.VEC_U['v_mad_i32_i16'] = saved


DRAW_BATCHES = (
    [
        (0xdead0003, 0xbeef0005, 0x3f800000),  # (+3)*(+5) + 0x3f800000 (upper 16 bits ignored)
        (0x1234ffff, 0x56780007, 0x3f800010),  # (-1)*(+7) + 0x3f800010 = 0x3f800009
        (0xabcd8000, 0xef018000, 0x3f800000),  # (-32768)*(-32768) = +0x40000000 -> 0x7f800000 (+inf, exact)
        (0x00008000, 0x00007fff, 0x80000000),  # (-32768)*(+32767) + 0x80000000 = 0x40008000
    ],
    [
        (0x0000fffe, 0x0000fffd, 0x3f800000),  # (-2)*(-3) + 0x3f800000 = 0x3f800006
        (0xffff0000, 0xffff7fff, 0x40000000),  # 0*(+32767) + 0x40000000 = 0x40000000
        (0x00000001, 0x00000001, 0xffffffff),  # wraps cleanly to 0x00000000
        (0x99990010, 0x8888fff0, 0x3f800100),  # (+16)*(-16) + 0x3f800100 = 0x3f800000
    ],
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    _, ps_hlsl = translate()
    with tempfile.TemporaryDirectory(prefix='x2w_v_mad_i32_i16_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            ps_bin = compile_hlsl(compiler, kind, ps_hlsl, 'ps_%s' % kind, ps_profile, td)
            first_vs = None
            first_exp = None
            for b_idx, triples in enumerate(DRAW_BATCHES):
                vs_hlsl = G.Pipeline(vs_words(triples), ps_words(), root_blob()).hlsl_vs()
                vs_bin = compile_hlsl(compiler, kind, vs_hlsl, 'vs_%d_%s' % (b_idx, kind), vs_profile, td)
                raw_exp = tuple(ref_mad_i32_i16(a, b, c) for a, b, c in triples)
                expected = words_text(raw_exp)
                res = draw(renderer, adapter, vs_bin, ps_bin, expected)
                if res.returncode == 77:
                    print('SKIP (%s): %s' % (adapter, res.stdout.strip() or res.stderr.strip()))
                    sys.exit(77)
                if res.returncode != 0:
                    raise AssertionError('%s %s batch=%d expected=%s\n%s\n%s' % (
                        kind, adapter, b_idx, expected, res.stdout, res.stderr))
                print('%s %s batch=%d: %s' % (kind, adapter, b_idx, expected), flush=True)
                if first_vs is None:
                    first_vs, first_exp = vs_bin, raw_exp

            bad = words_text((first_exp[0] ^ 1, first_exp[1], first_exp[2], first_exp[3]))
            neg = draw(renderer, adapter, first_vs, ps_bin, bad)
            check('draw %s %s negative control rejects wrong expectation' % (kind, adapter),
                  neg.returncode != 0)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run_controls()
        print('v_mad_i32_i16 controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('v_mad_i32_i16 draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_v_mad_i32_i16_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
