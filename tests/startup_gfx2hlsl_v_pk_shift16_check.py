"""Packed 16-bit integer shifts (v_pk_lshlrev_b16, v_pk_lshrrev_b16, v_pk_ashrrev_i16) in tools/startup/gfx2hlsl.py.

  (no arguments)                       VOP3P decode, value evaluation against emitted HLSL across op_sel/op_sel_hi,
                                       EXEC masking, low16_only_v rules, refusals, unlowered controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input pairs and compare the four result words

RDNA 2 ISA, VOP3P:
  V_PK_LSHLREV_B16 (0x04): D_i = (S1_i << (S0_i & 15)) & 0xffff.
  V_PK_LSHRREV_B16 (0x05): D_i = S1_i >> (S0_i & 15) (unsigned 16-bit).
  V_PK_ASHRREV_I16 (0x06): D_i = (int16(S1_i) >> (S0_i & 15)) & 0xffff (signed 16-bit).
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

OP_LSHL = 0x04
OP_LSHR = 0x05
OP_ASHR = 0x06
ALL_OPS = (
    (OP_LSHL, 'v_pk_lshlrev_b16'),
    (OP_LSHR, 'v_pk_lshrrev_b16'),
    (OP_ASHR, 'v_pk_ashrrev_i16'),
)
XOR_MASK = 0x3f800000


def vop3p(op, vdst, src0, src1, src2=0, *, op_sel=0, op_sel_hi=3, neg_lo=0, neg_hi=0, clamp=0, lit=None):
    w0 = (0xcc000000 | ((op & 0x7f) << 16) | (clamp << 15) | ((op_sel & 7) << 11)
          | ((neg_hi & 7) << 8) | (vdst & 0xff))
    w1 = (((neg_lo & 7) << 29) | ((op_sel_hi & 3) << 27)
          | ((src2 & 0x1ff) << 18) | ((src1 & 0x1ff) << 9) | (src0 & 0x1ff))
    out = struct.pack('<2I', w0, w1)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


def v_xor_lit(vdst, vsrc, lit):
    w = (0x1d << 25) | ((vdst & 0xff) << 17) | ((vsrc & 0xff) << 9) | 255
    return struct.pack('<2I', w, lit & MASK)


def ps_words(op, op_sel=0, op_sel_hi=3):
    # XOR with 0x3f800000 so all 16-bit packed outputs survive R32G32B32A32_FLOAT mrt0 export bit-for-bit.
    body = b''.join(
        vop3p(op, 12 + lane, 256 + lane, 256 + 4 + lane, op_sel=op_sel, op_sel_hi=op_sel_hi)
        + v_xor_lit(12 + lane, 12 + lane, XOR_MASK)
        for lane in range(4)
    )
    return PRE_WORDS + body + POST_WORDS


def translate(op, op_sel=0, op_sel_hi=3):
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(op, op_sel=op_sel, op_sel_hi=op_sel_hi), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def sext16(v):
    v &= 0xffff
    return v - 0x10000 if v >= 0x8000 else v


def ref_pk_shift(op, a, b, op_sel=0, op_sel_hi=3):
    def pick(word, hi):
        return ((word >> 16) if hi else word) & 0xffff

    def lane(sel_mask):
        amt = pick(a, sel_mask & 1) & 15
        val = pick(b, (sel_mask >> 1) & 1)
        if op == OP_LSHL:
            return (val << amt) & 0xffff
        if op == OP_LSHR:
            return (val >> amt) & 0xffff
        return (sext16(val) >> amt) & 0xffff

    return (lane(op_sel) | (lane(op_sel_hi) << 16)) & MASK


def eval_hlsl_u32(expr, env):
    """Evaluate the emitted 32-bit HLSL expression directly in Python."""
    cleaned = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', expr)

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
            raise ValueError('unsupported call %s' % node.func.id)
        if isinstance(node, ast.BinOp):
            l, r = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return to_u32(l + r)
            if isinstance(node.op, ast.LShift):
                return to_u32(l << (r & 31))
            if isinstance(node.op, ast.RShift):
                is_signed = isinstance(node.left, ast.Call) and getattr(node.left.func, 'id', '') == 'asint'
                return (to_i32(l) >> (r & 31)) if is_signed else (to_u32(l) >> (r & 31))
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


SHIFT_WORDS = (
    0x00000000, 0x00010004, 0x000f0008, 0x0010001f,  # includes >15 counts (must mask & 15)
    0x00070002, 0xfff0000f, 0x000c0001, 0xabcd0000,
)
VAL_WORDS = (
    0x00000000, 0x0001ffff, 0x7fff8000, 0x80007fff,
    0xffff0001, 0x1234abcd, 0x80017ffe, 0xdeadbeef,
)


def run_controls():
    # 1. Decode verification via G.decode().
    for op, name in ALL_OPS:
        check('decode %s default' % name,
              G.decode(vop3p(op, 12, 256, 260))[0][3] == '%s v12, v0, v4' % name)
        check('decode %s op_sel/op_sel_hi' % name,
              G.decode(vop3p(op, 12, 256, 260, op_sel=1, op_sel_hi=2))[0][3]
              == '%s v12, v0, v4 op_sel:[1,0] op_sel_hi:[0,1]' % name)
        check('decode %s rejects reserved src2' % name,
              G.decode(vop3p(op, 12, 256, 260, src2=1))[0][2] == '%s_unsupported' % name)

    # 2. Exhaustive grid evaluated directly against emitted HLSL across 3 op_sel configs (576 cases total).
    for op, name in ALL_OPS:
        for op_sel, op_sel_hi, mod_str in (
            (0, 3, ''),
            (1, 2, ' op_sel:[1,0] op_sel_hi:[0,1]'),
            (0, 0, ' op_sel:[0,0] op_sel_hi:[0,0]'),
        ):
            p = Probe()
            asm = '%s v12, v0, v4%s' % (name, mod_str)
            p.one(name, asm)
            check('emitted %s' % asm,
                  len(p.lines) == 1 and p.lines[0].startswith('v12 = ') and p.lines[0].endswith(';'))
            rhs = p.lines[0][len('v12 = '):-1]
            ok = True
            for a in SHIFT_WORDS:
                for b in VAL_WORDS:
                    if eval_hlsl_u32(rhs, {'v0': a, 'v4': b}) != ref_pk_shift(op, a, b, op_sel, op_sel_hi):
                        ok = False
            check('%s%s grid (%d cases)' % (name, mod_str, len(SHIFT_WORDS) * len(VAL_WORDS)), ok)

    # 3. EXEC masking and low16_only_v tracking.
    for _, name in ALL_OPS:
        p = Probe()
        p.exec_known = False
        p.one(name, '%s v12, v0, v4' % name)
        check('exec-masked %s guards assignment' % name,
              len(p.lines) == 1 and p.lines[0].startswith('if (exec & 1) v12 = '))

        p_lo = Probe()
        p_lo.low16_only_v = {0, 4}
        p_lo.one(name, '%s v12, v0, v4 op_sel:[0,0] op_sel_hi:[0,0]' % name)
        check('%s accepts low16_only_v when only low halves are read' % name, len(p_lo.lines) == 1)

        p_hi = Probe()
        p_hi.low16_only_v = {4}
        check('%s refuses low16_only_v when high half is read' % name,
              stage_refused(p_hi, name, '%s v12, v0, v4' % name, 'whole-word read'))

    # 4. Stage refusals: clamp, neg_lo, neg_hi, bad op_sel, wrong arity, lane mask, bad dst.
    for _, name in ALL_OPS:
        for asm in (
            '%s v12, v0, v4 clamp' % name,
            '%s v12, v0, v4 neg_lo:[1,0]' % name,
            '%s v12, v0, v4 neg_hi:[0,1]' % name,
            '%s v12, v0, v4 op_sel:[1,0,0]' % name,
            '%s v12, v0, v4 op_sel:[2,0]' % name,
            '%s v12, v0' % name,
            '%s v12, v0, v4, v8' % name,
            '%s v12, vcc_lo, v4' % name,
            '%s s0, v0, v4' % name,
        ):
            check('Stage refuses %s' % asm, stage_refused(Probe(), name, asm))

    # 5. Unlowered control: removing handlers from Stage restores refusals.
    for op, name in ALL_OPS:
        attr = 'op_' + name
        saved = getattr(G.Stage, attr)
        delattr(G.Stage, attr)
        try:
            check('unlowered %s refused' % name, stage_refused(Probe(), name, '%s v12, v0, v4' % name))
            try:
                translate(op)
            except G.Unsupported:
                check('unlowered %s pipeline refused' % name, True)
            else:
                check('unlowered %s pipeline refused' % name, False)
        finally:
            setattr(G.Stage, attr, saved)


DRAW_CASES = (
    (OP_LSHL, 0, 3, 'lshlrev_b16', [
        (0x00040001, 0x00018001, 0),  # hi: 0x0001<<4=0x0010, lo: (0x8001<<1)&0xffff=0x0002 (no bleed!)
        (0x000f0010, 0x0001abcd, 0),  # hi: 0x0001<<15=0x8000, lo: 0xabcd<<(16&15=0)=0xabcd
        (0x001f0008, 0xffff1234, 0),  # hi: 0xffff<<15=0x8000, lo: 0x1234<<8=0x3400
        (0x00000000, 0x7fff8000, 0),  # identity
    ]),
    (OP_LSHR, 1, 2, 'lshrrev_b16_swizzle', [
        # op_sel=[1,0] (a.hi, b.lo), op_sel_hi=[0,1] (a.lo, b.hi)
        (0x00040001, 0x8000ffff, 0),  # lo: 0xffff>>4=0x0fff; hi: 0x8000>>1=0x4000
        (0x000f0010, 0xdeadbeef, 0),  # lo: 0xbeef>>15=0x0001; hi: 0xdead>>0=0xdead
        (0x0008000c, 0xfedc1234, 0),  # lo: 0x1234>>8=0x0012; hi: 0xfedc>>12=0x000f
        (0x0000000f, 0x80007fff, 0),  # lo: 0x7fff>>0=0x7fff; hi: 0x8000>>15=0x0001
    ]),
    (OP_ASHR, 0, 3, 'ashrrev_i16', [
        (0x00010001, 0x80007fff, 0),  # hi: -32768>>1=0xc000, lo: +32767>>1=0x3fff
        (0x000f000f, 0x80007fff, 0),  # hi: -32768>>15=0xffff, lo: +32767>>15=0x0000
        (0x00140004, 0xff0000f0, 0),  # hi: (-256)>>4=0xfff0, lo: 0x00f0>>4=0x000f
        (0x00000008, 0xffff8000, 0),  # hi: -1>>0=0xffff, lo: -32768>>8=0xff80
    ]),
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory(prefix='x2w_v_pk_shift16_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            first_vs = first_ps = first_exp = None
            for op, op_sel, op_sel_hi, label, triples in DRAW_CASES:
                pipe = G.Pipeline(vs_words(triples), ps_words(op, op_sel=op_sel, op_sel_hi=op_sel_hi), root_blob())
                vs_bin = compile_hlsl(compiler, kind, pipe.hlsl_vs(), 'vs_%s_%s' % (label, kind), vs_profile, td)
                ps_bin = compile_hlsl(compiler, kind, pipe.hlsl_ps(), 'ps_%s_%s' % (label, kind), ps_profile, td)
                raw_exp = tuple(ref_pk_shift(op, a, b, op_sel, op_sel_hi) ^ XOR_MASK for a, b, _ in triples)
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
        print('v_pk_shift16 controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('v_pk_shift16 draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_v_pk_shift16_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
