"""Packed 16-bit integer multiply/mad (v_pk_mul_lo_u16, v_pk_mad_u16, v_pk_mad_i16) in tools/startup/gfx2hlsl.py.

  (no arguments)                       VOP3P decode, value evaluation against emitted HLSL across op_sel/op_sel_hi
                                       and clamp, EXEC masking, low16_only_v rules, refusals, unlowered controls
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input triples and compare the four result words

RDNA 2 ISA, VOP3P:
  V_PK_MUL_LO_U16 (0x01): D_i = (S0_i * S1_i) & 0xffff (no clamp).
  V_PK_MAD_U16    (0x09): D_i = S0_i * S1_i + S2_i (mod 2^16, or saturated to [0, 65535] with clamp).
  V_PK_MAD_I16    (0x00): D_i = int16(S0_i) * int16(S1_i) + int16(S2_i) (mod 2^16, or saturated to [-32768, 32767]).
"""
import ast
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

OP_MAD_I16 = 0x00
OP_MUL_LO_U16 = 0x01
OP_MAD_U16 = 0x09
XOR_MASK = 0x3f800000


def vop3p(op, vdst, src0, src1, src2=0, *, op_sel=0, op_sel_hi=7, neg_lo=0, neg_hi=0, clamp=0, lit=None):
    w0 = (0xcc000000 | ((op & 0x7f) << 16) | (clamp << 15) | (((op_sel_hi >> 2) & 1) << 14)
          | ((op_sel & 7) << 11) | ((neg_hi & 7) << 8) | (vdst & 0xff))
    w1 = (((neg_lo & 7) << 29) | ((op_sel_hi & 3) << 27)
          | ((src2 & 0x1ff) << 18) | ((src1 & 0x1ff) << 9) | (src0 & 0x1ff))
    out = struct.pack('<2I', w0, w1)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


def v_xor_lit(vdst, vsrc, lit):
    w = (0x1d << 25) | ((vdst & 0xff) << 17) | ((vsrc & 0xff) << 9) | 255
    return struct.pack('<2I', w, lit & MASK)


def ps_words(op, op_sel=0, op_sel_hi=7, clamp=0):
    is_3src = op in (OP_MAD_I16, OP_MAD_U16)
    body = b''.join(
        vop3p(op, 12 + lane, 256 + lane, 256 + 4 + lane, (256 + 8 + lane) if is_3src else 0,
              op_sel=op_sel, op_sel_hi=op_sel_hi if is_3src else (op_sel_hi & 3), clamp=clamp)
        + v_xor_lit(12 + lane, 12 + lane, XOR_MASK)
        for lane in range(4)
    )
    return PRE_WORDS + body + POST_WORDS


def translate(op, op_sel=0, op_sel_hi=7, clamp=0):
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(op, op_sel=op_sel, op_sel_hi=op_sel_hi, clamp=clamp), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def sext16(v):
    v &= 0xffff
    return v - 0x10000 if v >= 0x8000 else v


def ref_op(op, a, b, c, clamp, op_sel=0, op_sel_hi=7):
    def pick(word, hi):
        return ((word >> 16) if hi else word) & 0xffff

    def lane(sel_mask):
        x = pick(a, sel_mask & 1)
        y = pick(b, (sel_mask >> 1) & 1)
        if op == OP_MUL_LO_U16:
            return (x * y) & 0xffff
        z = pick(c, (sel_mask >> 2) & 1)
        if op == OP_MAD_I16:
            r = sext16(x) * sext16(y) + sext16(z)
            if clamp:
                r = max(-32768, min(32767, r))
            return r & 0xffff
        r = x * y + z
        if clamp:
            r = min(0xffff, r)
        return r & 0xffff

    return (lane(op_sel) | (lane(op_sel_hi) << 16)) & MASK


def eval_hlsl_u32(expr, env):
    """Evaluate the emitted 32-bit HLSL expression in Python preserving signed/unsigned types."""
    cleaned = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', expr)

    class I32(int):
        pass

    def to_i32(x):
        x &= MASK
        return I32(x - 0x100000000 if x >= 0x80000000 else x)

    def to_u32(x):
        return int(x) & MASK

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return I32(-ev(node.operand))
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            args = [ev(a) for a in node.args]
            if fn == 'asint':
                return to_i32(args[0])
            if fn == 'asuint':
                return to_u32(args[0])
            if fn == 'min':
                return min(args[0], args[1])
            if fn == 'clamp':
                res = max(args[1], min(args[2], args[0]))
                return I32(res) if isinstance(args[0], I32) else to_u32(res)
            raise ValueError('unsupported call %s' % fn)
        if isinstance(node, ast.BinOp):
            l, r = ev(node.left), ev(node.right)
            signed_bin = isinstance(l, I32) or isinstance(r, I32)
            if isinstance(node.op, ast.Add):
                return to_i32(l + r) if signed_bin else to_u32(l + r)
            if isinstance(node.op, ast.Mult):
                return to_i32(l * r) if signed_bin else to_u32(l * r)
            if isinstance(node.op, ast.LShift):
                return to_u32(l << (r & 31))
            if isinstance(node.op, ast.RShift):
                return I32(to_i32(l) >> (r & 31)) if isinstance(l, I32) else to_u32(l) >> (r & 31)
            if isinstance(node.op, ast.BitAnd):
                return to_u32(l & r)
            if isinstance(node.op, ast.BitOr):
                return to_u32(l | r)
            raise ValueError('unsupported binop %s' % node.op)
        raise ValueError('unsupported node %s' % ast.dump(node))

    return to_u32(ev(ast.parse(cleaned, mode='eval')))


def stage_refused(stage, name, line, needle=''):
    try:
        stage.one(name, line)
    except G.Unsupported as exc:
        return (not needle or needle in str(exc)) and not stage.lines
    return False


EDGE_A = (0x00000000, 0x0002ffff, 0x7fff8000, 0x80007fff, 0xffff0003)
EDGE_B = (0x00000000, 0x0003fffe, 0x7fff8000, 0x80000002, 0xffff0005)
EDGE_C = (0x00000000, 0x0001ffff, 0x7fff8000, 0xffff0001)


def run_controls():
    # 1. Decode verification via G.decode().
    check('decode v_pk_mul_lo_u16',
          G.decode(vop3p(OP_MUL_LO_U16, 12, 256, 260, op_sel_hi=3))[0][3] == 'v_pk_mul_lo_u16 v12, v0, v4')
    check('decode v_pk_mul_lo_u16 rejects reserved src2',
          G.decode(vop3p(OP_MUL_LO_U16, 12, 256, 260, 1, op_sel_hi=3))[0][2] == 'v_pk_mul_lo_u16_unsupported')
    for op, name in ((OP_MAD_U16, 'v_pk_mad_u16'), (OP_MAD_I16, 'v_pk_mad_i16')):
        check('decode %s default' % name,
              G.decode(vop3p(op, 12, 256, 260, 264))[0][3] == '%s v12, v0, v4, v8' % name)
        check('decode %s clamp' % name,
              G.decode(vop3p(op, 12, 256, 260, 264, clamp=1))[0][3] == '%s v12, v0, v4, v8 clamp' % name)
        check('decode %s op_sel/op_sel_hi/clamp' % name,
              G.decode(vop3p(op, 12, 256, 260, 264, op_sel=5, op_sel_hi=2, clamp=1))[0][3]
              == '%s v12, v0, v4, v8 op_sel:[1,0,1] op_sel_hi:[0,1,0] clamp' % name)

    # 2. Exhaustive grid evaluated directly against emitted HLSL.
    for op_sel, op_sel_hi, mod_str in ((0, 3, ''), (1, 2, ' op_sel:[1,0] op_sel_hi:[0,1]')):
        p = Probe()
        asm = 'v_pk_mul_lo_u16 v12, v0, v4%s' % mod_str
        p.one('v_pk_mul_lo_u16', asm)
        check('emitted %s' % asm, len(p.lines) == 1 and p.lines[0].startswith('v12 = '))
        rhs = p.lines[0][len('v12 = '):-1]
        ok = all(
            eval_hlsl_u32(rhs, {'v0': a, 'v4': b}) == ref_op(OP_MUL_LO_U16, a, b, 0, 0, op_sel, op_sel_hi)
            for a in EDGE_A for b in EDGE_B
        )
        check('v_pk_mul_lo_u16%s grid (%d cases)' % (mod_str, len(EDGE_A) * len(EDGE_B)), ok)

    for op, name in ((OP_MAD_U16, 'v_pk_mad_u16'), (OP_MAD_I16, 'v_pk_mad_i16')):
        for clamp in (0, 1):
            for op_sel, op_sel_hi, mod_str in ((0, 7, ''), (5, 2, ' op_sel:[1,0,1] op_sel_hi:[0,1,0]')):
                p = Probe()
                asm = '%s v12, v0, v4, v8%s%s' % (name, mod_str, ' clamp' if clamp else '')
                p.one(name, asm)
                check('emitted %s' % asm, len(p.lines) == 1 and p.lines[0].startswith('v12 = '))
                rhs = p.lines[0][len('v12 = '):-1]
                ok = all(
                    eval_hlsl_u32(rhs, {'v0': a, 'v4': b, 'v8': c}) == ref_op(op, a, b, c, clamp, op_sel, op_sel_hi)
                    for a in EDGE_A for b in EDGE_B for c in EDGE_C
                )
                check('%s%s%s grid (%d cases)' % (
                    name, mod_str, ' clamp' if clamp else '', len(EDGE_A) * len(EDGE_B) * len(EDGE_C)), ok)

    # 3. EXEC masking & low16_only_v tracking.
    p = Probe()
    p.exec_known = False
    p.one('v_pk_mul_lo_u16', 'v_pk_mul_lo_u16 v12, v0, v4')
    check('exec-masked v_pk_mul_lo_u16 guards assignment',
          len(p.lines) == 1 and p.lines[0].startswith('if (exec & 1) v12 = '))

    p_lo = Probe()
    p_lo.low16_only_v = {0, 4}
    p_lo.one('v_pk_mul_lo_u16', 'v_pk_mul_lo_u16 v12, v0, v4 op_sel:[0,0] op_sel_hi:[0,0]')
    check('v_pk_mul_lo_u16 accepts low16_only_v when only low halves are read', len(p_lo.lines) == 1)

    for name in ('v_pk_mad_u16', 'v_pk_mad_i16'):
        for suffix in ('', ' clamp'):
            pm = Probe()
            pm.exec_known = False
            pm.one(name, '%s v12, v0, v4, v8%s' % (name, suffix))
            check('exec-masked %s%s guards assignment' % (name, suffix),
                  len(pm.lines) == 1 and pm.lines[0].startswith('if (exec & 1) v12 = '))
        pl = Probe()
        pl.low16_only_v = {0, 4, 8}
        pl.one(name, '%s v12, v0, v4, v8 op_sel:[0,0,0] op_sel_hi:[0,0,0]' % name)
        check('%s accepts low16_only_v when only low halves are read' % name, len(pl.lines) == 1)
        ph = Probe()
        ph.low16_only_v = {8}
        check('%s refuses low16_only_v when high half is read' % name,
              stage_refused(ph, name, '%s v12, v0, v4, v8' % name, 'whole-word read'))

    # 4. Stage refusals.
    for asm in (
        'v_pk_mul_lo_u16 v12, v0, v4 clamp',
        'v_pk_mul_lo_u16 v12, v0, v4 neg_lo:[1,0]',
        'v_pk_mul_lo_u16 v12, v0, v4 neg_hi:[0,1]',
        'v_pk_mul_lo_u16 v12, v0, v4 op_sel:[1,0,0]',
        'v_pk_mul_lo_u16 v12, v0, v4, v8',
        'v_pk_mul_lo_u16 v12, vcc_lo, v4',
        'v_pk_mul_lo_u16 s0, v0, v4',
    ):
        check('Stage refuses %s' % asm, stage_refused(Probe(), 'v_pk_mul_lo_u16', asm))
    for name in ('v_pk_mad_u16', 'v_pk_mad_i16'):
        for asm in (
            '%s v12, v0, v4, v8 clamp clamp' % name,
            '%s v12, v0, v4, v8 neg_lo:[1,0,0]' % name,
            '%s v12, v0, v4, v8 neg_hi:[0,1,0]' % name,
            '%s v12, v0, v4, v8 op_sel:[1,0]' % name,
            '%s v12, v0, v4' % name,
            '%s v12, v0, v4, v8, v9' % name,
            '%s v12, vcc_lo, v4, v8' % name,
            '%s s0, v0, v4, v8' % name,
        ):
            check('Stage refuses %s' % asm, stage_refused(Probe(), name, asm))

    # 5. Unlowered controls.
    for op, name, sample in (
        (OP_MUL_LO_U16, 'v_pk_mul_lo_u16', 'v_pk_mul_lo_u16 v12, v0, v4'),
        (OP_MAD_U16, 'v_pk_mad_u16', 'v_pk_mad_u16 v12, v0, v4, v8'),
        (OP_MAD_I16, 'v_pk_mad_i16', 'v_pk_mad_i16 v12, v0, v4, v8'),
    ):
        attr = 'op_' + name
        saved = getattr(G.Stage, attr)
        delattr(G.Stage, attr)
        try:
            check('unlowered %s refused' % name, stage_refused(Probe(), name, sample))
            try:
                translate(op)
            except G.Unsupported:
                check('unlowered %s pipeline refused' % name, True)
            else:
                check('unlowered %s pipeline refused' % name, False)
        finally:
            setattr(G.Stage, attr, saved)


DRAW_CASES = (
    (OP_MUL_LO_U16, 0, 0, 3, 'mul_lo_u16', [
        (0x00030005, 0x0007000b, 0),  # hi: 21, lo: 55
        (0xffff8000, 0xffff0002, 0),  # hi: (0xffff^2)&0xffff=1, lo: (0x8000*2)&0xffff=0 (no bleed!)
        (0x12345678, 0x00010000, 0),  # hi: 0x1234, lo: 0
        (0xabcd0100, 0x00040100, 0),  # hi: (0xabcd*4)&0xffff=0xaf34, lo: (0x100*0x100)&0xffff=0
    ]),
    (OP_MAD_U16, 1, 0, 7, 'mad_u16_clamp', [
        (0x01000003, 0x01000005, 0x00010007),  # hi: 0x10001 -> 0xffff, lo: 3*5+7 = 22
        (0xffff8000, 0xffff0002, 0xffff0000),  # hi: 0xffff, lo: 0xffff
        (0x00ff0000, 0x0100ffff, 0x00ff0001),  # hi: 255*256+255=0xffff (exact), lo: 1
        (0x00020004, 0x00030005, 0x00010002),  # hi: 7, lo: 22
    ]),
    (OP_MAD_I16, 0, 0, 7, 'mad_i16_wrap', [
        (0x7fffffff, 0x00020003, 0x00010005),  # hi: (0x7fff*2+1)&0xffff=0xffff, lo: (-1*3+5)=2
        (0x80008000, 0x80000002, 0x00010001),  # hi: 1, lo: 1
        (0x0003fffe, 0x0005fffd, 0x00010001),  # hi: 16, lo: (-2)*(-3)+1 = 7
        (0x00000000, 0xffff7fff, 0xdeadbeef),  # hi: 0xdead, lo: 0xbeef
    ]),
    (OP_MAD_I16, 1, 5, 2, 'mad_i16_clamp_swizzle', [
        # op_sel=[1,0,1] (a.hi * b.lo + c.hi), op_sel_hi=[0,1,0] (a.lo * b.hi + c.lo)
        (0x80007fff, 0x00028000, 0x7fff0001),  # lo: (-32768)*(-32768)+32767 -> 0x7fff; hi: (+32767)*2+1 -> 0x7fff
        (0x80008000, 0x7fff7fff, 0x80008000),  # lo: (-32768)*32767-32768 -> 0x8000; hi: (-32768)*32767-32768 -> 0x8000
        (0x0003fffd, 0x00050002, 0x0001fffe),  # lo: 3*2+1=7; hi: (-3)*5+(-2)=-17 (0xffef)
        (0xffff0001, 0x0001ffff, 0x00000000),  # lo: (-1)*(-1)=1; hi: 1*1=1
    ]),
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory(prefix='x2w_v_pk_mul_mad_16_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            first_vs = first_ps = first_exp = None
            for op, clamp, op_sel, op_sel_hi, label, triples in DRAW_CASES:
                pipe = G.Pipeline(vs_words(triples), ps_words(op, op_sel=op_sel, op_sel_hi=op_sel_hi, clamp=clamp),
                                  root_blob())
                vs_bin = compile_hlsl(compiler, kind, pipe.hlsl_vs(), 'vs_%s_%s' % (label, kind), vs_profile, td)
                ps_bin = compile_hlsl(compiler, kind, pipe.hlsl_ps(), 'ps_%s_%s' % (label, kind), ps_profile, td)
                raw_exp = tuple(ref_op(op, a, b, c, clamp, op_sel, op_sel_hi) ^ XOR_MASK for a, b, c in triples)
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
        print('v_pk_mul_lo_u16 / v_pk_mad_16 controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('v_pk_mul_lo_u16 / v_pk_mad_16 draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_v_pk_mul_mad_16_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
