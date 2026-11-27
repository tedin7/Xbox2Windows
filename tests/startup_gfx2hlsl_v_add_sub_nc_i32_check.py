"""Signed 32-bit add/sub (v_add_nc_i32, v_sub_nc_i32, with and without clamp) in tools/startup/gfx2hlsl.py.

  (no arguments)                       VOP3A decode, value evaluation against emitted HLSL, EXEC masking,
                                       align4/uniformity rules, encoding & Stage refusals, unlowered controls
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input pairs and compare the four result words

RDNA 2 ISA, VOP3A:
  V_ADD_NC_I32 (0x37f): D.i32 = S0.i32 + S1.i32; with clamp, saturates to [-2^31, 2^31 - 1].
  V_SUB_NC_I32 (0x376): D.i32 = S0.i32 - S1.i32; with clamp, saturates to [-2^31, 2^31 - 1].
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

OP_ADD = 0x37f
OP_SUB = 0x376
XOR_MASK = 0x3f800000
INT32_MIN = -0x80000000
INT32_MAX = 0x7fffffff


def vop3a(op, vdst, src0, src1, src2=0, *, abs_bits=0, op_sel=0, clamp=0, omod=0, neg=0, lit=None):
    w0 = 0xd4000000 | ((op & 0x3ff) << 16) | (clamp << 15) | (op_sel << 11) | (abs_bits << 8) | (vdst & 0xff)
    w1 = (neg << 29) | (omod << 27) | ((src2 & 0x1ff) << 18) | ((src1 & 0x1ff) << 9) | (src0 & 0x1ff)
    out = struct.pack('<2I', w0, w1)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


def v_xor_lit(vdst, vsrc, lit):
    # VOP2 v_xor_b32_e32 (op 0x1d) with src0 = 255 (literal) and vsrc1 = vsrc.
    w = (0x1d << 25) | ((vdst & 0xff) << 17) | ((vsrc & 0xff) << 9) | 255
    return struct.pack('<2I', w, lit & MASK)


def ps_words(op, clamp=0):
    # Compute 4 lanes and XOR each output with 0x3f800000 so INT32_MAX (0x7fffffff, NaN bits)
    # and small integers remain normal float32 values through the R32G32B32A32_FLOAT mrt0 export.
    body = b''.join(
        vop3a(op, 12 + lane, 256 + lane, 256 + 4 + lane, clamp=clamp) + v_xor_lit(12 + lane, 12 + lane, XOR_MASK)
        for lane in range(4)
    )
    return PRE_WORDS + body + POST_WORDS


def translate(op, clamp=0):
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(op, clamp=clamp), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def s32(v):
    v &= MASK
    return v - 0x100000000 if v >= 0x80000000 else v


def ref_op(op, a, b, clamp):
    res = s32(a) + s32(b) if op == OP_ADD else s32(a) - s32(b)
    if clamp:
        res = max(INT32_MIN, min(INT32_MAX, res))
    return res & MASK


def eval_hlsl_u32(expr, env, helpers=None):
    """Evaluate a 32-bit unsigned HLSL expression (supporting helper calls, ternaries, bitwise/arith ops)."""
    cleaned = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', expr)
    # Convert C/HLSL ternary `cond ? a : b` at top level of helper return expressions.
    m = re.fullmatch(r'\s*(.+?)\s*\?\s*(.+?)\s*:\s*(.+?)\s*', cleaned)
    if m:
        cond, t_br, f_br = m.groups()
        return eval_hlsl_u32(t_br if eval_hlsl_u32(cond, env, helpers) else f_br, env, helpers)

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return (~ev(node.operand)) & MASK
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq):
            return int(ev(node.left) != ev(node.comparators[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if helpers and fn in helpers:
                args = [ev(a) & MASK for a in node.args]
                return helpers[fn](*args)
            raise ValueError('unsupported call %s' % fn)
        if isinstance(node, ast.BinOp):
            l, r = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return (l + r) & MASK
            if isinstance(node.op, ast.Sub):
                return (l - r) & MASK
            if isinstance(node.op, ast.BitXor):
                return (l ^ r) & MASK
            if isinstance(node.op, ast.BitAnd):
                return (l & r) & MASK
            if isinstance(node.op, ast.RShift):
                return (l & MASK) >> (r & 31)
            raise ValueError('unsupported binop %s' % node.op)
        raise ValueError('unsupported node %s' % ast.dump(node))

    return ev(ast.parse(cleaned, mode='eval')) & MASK


def extract_helper(hlsl, fn_name):
    m = re.search(
        r'uint\s+%s\(uint a, uint b\)\s*\{\s*uint\s+(\w+)\s*=\s*([^;]+);\s*return\s+([^;]+);\s*\}' % fn_name,
        hlsl,
    )
    if not m:
        return None
    tmp_var, tmp_expr, ret_expr = m.groups()

    def fn(a, b):
        env = {'a': a & MASK, 'b': b & MASK}
        env[tmp_var] = eval_hlsl_u32(tmp_expr, env)
        return eval_hlsl_u32(ret_expr, env)

    return fn


def stage_refused(stage, name, line, needle=''):
    try:
        stage.one(name, line)
    except G.Unsupported as exc:
        return (not needle or needle in str(exc)) and not stage.lines
    return False


EDGE_WORDS = (
    0x00000000, 0x00000001, 0x00000002, 0x3fffffff, 0x7ffffffe,
    0x7fffffff, 0x80000000, 0x80000001, 0xc0000000, 0xffffffff,
)


def run_controls():
    # 1. Decode verification from synthetic gfx1013 VOP3A bytes via G.decode().
    for op, name in ((OP_ADD, 'v_add_nc_i32'), (OP_SUB, 'v_sub_nc_i32')):
        check('decode %s' % name,
              G.decode(vop3a(op, 12, 256, 260))[0][3] == '%s v12, v0, v4' % name)
        check('decode %s clamp' % name,
              G.decode(vop3a(op, 12, 256, 260, clamp=1))[0][3] == '%s v12, v0, v4 clamp' % name)

    # 2. Selective helper emission and direct evaluation of emitted HLSL across 400 edge cases.
    _, ps_add = translate(OP_ADD, clamp=0)
    _, ps_sub = translate(OP_SUB, clamp=0)
    _, ps_add_c = translate(OP_ADD, clamp=1)
    _, ps_sub_c = translate(OP_SUB, clamp=1)
    check('unclamped add emits no sat helpers',
          'add_sat_i32' not in ps_add and 'sub_sat_i32' not in ps_add)
    check('unclamped sub emits no sat helpers',
          'add_sat_i32' not in ps_sub and 'sub_sat_i32' not in ps_sub)
    check('clamped add emits only add_sat_i32',
          'add_sat_i32(' in ps_add_c and 'sub_sat_i32' not in ps_add_c)
    check('clamped sub emits only sub_sat_i32',
          'sub_sat_i32(' in ps_sub_c and 'add_sat_i32' not in ps_sub_c)

    helpers = {
        'add_sat_i32': extract_helper(ps_add_c, 'add_sat_i32'),
        'sub_sat_i32': extract_helper(ps_sub_c, 'sub_sat_i32'),
    }
    check('extracted sat helpers from HLSL', all(helpers.values()))

    for op, name in ((OP_ADD, 'v_add_nc_i32'), (OP_SUB, 'v_sub_nc_i32')):
        for clamp in (0, 1):
            p = Probe()
            asm = '%s v12, v0, v4%s' % (name, ' clamp' if clamp else '')
            p.one(name, asm)
            check('emitted %s' % asm,
                  len(p.lines) == 1 and p.lines[0].startswith('v12 = ') and p.lines[0].endswith(';'))
            rhs = p.lines[0][len('v12 = '):-1]
            ok = True
            for a in EDGE_WORDS:
                for b in EDGE_WORDS:
                    got = eval_hlsl_u32(rhs, {'v0': a, 'v4': b}, helpers)
                    if got != ref_op(op, a, b, clamp):
                        ok = False
            check('%s%s grid (%d cases)' % (name, ' clamp' if clamp else '', len(EDGE_WORDS) ** 2), ok)

    # 3. EXEC masking, align4 propagation, and uniformity tracking.
    for name in ('v_add_nc_i32', 'v_sub_nc_i32'):
        for suffix in ('', ' clamp'):
            p = Probe()
            p.exec_known = False
            p.one(name, '%s v12, v0, v4%s' % (name, suffix))
            check('exec-masked %s%s guards assignment' % (name, suffix),
                  len(p.lines) == 1 and p.lines[0].startswith('if (exec & 1) v12 = '))

        p_a = Probe()
        p_a.align4_v = {0, 4}
        p_a.one(name, '%s v12, v0, v4' % name)
        check('unclamped %s preserves align4' % name, 12 in p_a.align4_v)

        p_ac = Probe()
        p_ac.align4_v = {0, 4}
        p_ac.one(name, '%s v12, v0, v4 clamp' % name)
        check('clamped %s clears align4' % name, 12 not in p_ac.align4_v)

        for suffix in ('', ' clamp'):
            p_u = Probe()
            p_u.exec_known = True
            p_u.past_entry = False
            p_u.uniform_v = {0, 4}
            p_u.varying_s = set()
            p_u.scc_uniform = True
            p_u.one(name, '%s v12, v0, v4%s' % (name, suffix))
            check('%s%s preserves uniformity' % (name, suffix), 12 in p_u.uniform_v)

    # 4. VOP3A encoding guard refuses neg/abs and nonzero unused src2.
    for op, name in ((OP_ADD, 'v_add_nc_i32'), (OP_SUB, 'v_sub_nc_i32')):
        for label, bad_word, needle in (
            ('abs0', vop3a(op, 12, 256, 260, abs_bits=1), 'with neg/abs'),
            ('neg1', vop3a(op, 12, 256, 260, neg=2), 'with neg/abs'),
            ('neg_inline0', vop3a(op, 12, 129, 260, neg=1), 'with neg/abs'),
            ('unused_src2', vop3a(op, 12, 256, 260, src2=264), 'unused src2'),
        ):
            reasons = []
            G.decode(bad_word, reasons)
            check('encoding guard refuses %s %s' % (name, label),
                  len(reasons) == 1 and needle in reasons[0][0])

    # 5. Stage refusals: _e32 spelling, bad modifiers, abs/neg, wrong arity, lane mask, bad dst.
    for name in ('v_add_nc_i32', 'v_sub_nc_i32'):
        check('Stage refuses %s_e32' % name,
              stage_refused(Probe(), '%s_e32' % name, '%s_e32 v12, v0, v4' % name))
        for asm in (
            '%s v12, v0, v4 mul:2' % name,
            '%s v12, v0, v4 clamp mul:2' % name,
            '%s v12, v0, v4 op_sel:[1,0,0]' % name,
            '%s v12, |v0|, v4' % name,
            '%s v12, v0, -v4' % name,
            '%s v12, v0' % name,
            '%s v12, v0, v4, v8' % name,
            '%s v12, vcc_lo, v4' % name,
            '%s s0, v0, v4' % name,
        ):
            check('Stage refuses %s' % asm, stage_refused(Probe(), name, asm))

    # 6. Unlowered control: removing handlers from Stage restores refusals.
    for op, name in ((OP_ADD, 'v_add_nc_i32'), (OP_SUB, 'v_sub_nc_i32')):
        attr = 'op_' + name
        saved = getattr(G.Stage, attr)
        delattr(G.Stage, attr)
        try:
            check('unlowered %s refused' % name,
                  stage_refused(Probe(), name, '%s v12, v0, v4' % name))
            try:
                translate(op, clamp=0)
            except G.Unsupported:
                check('unlowered %s pipeline refused' % name, True)
            else:
                check('unlowered %s pipeline refused' % name, False)
        finally:
            setattr(G.Stage, attr, saved)


DRAW_CASES = (
    (OP_ADD, 0, 'add_wrap', [
        (0x00000003, 0x00000005, 0),  # 3 + 5 = 8
        (0x7fffffff, 0x00000001, 0),  # INT32_MAX + 1 wraps to 0x80000000
        (0x80000000, 0xffffffff, 0),  # INT32_MIN + (-1) wraps to 0x7fffffff
        (0x80000000, 0x80000000, 0),  # INT32_MIN + INT32_MIN wraps to 0
    ]),
    (OP_ADD, 1, 'add_clamp', [
        (0x00000003, 0x00000005, 0),  # 3 + 5 = 8 (no saturation)
        (0x7fffffff, 0x00000001, 0),  # saturates to INT32_MAX (0x7fffffff)
        (0x80000000, 0xffffffff, 0),  # saturates to INT32_MIN (0x80000000)
        (0x7fffffff, 0x80000000, 0),  # opposite signs: -1 (0xffffffff, no saturation)
    ]),
    (OP_SUB, 0, 'sub_wrap', [
        (0x0000000a, 0x00000003, 0),  # 10 - 3 = 7
        (0x7fffffff, 0xffffffff, 0),  # INT32_MAX - (-1) wraps to 0x80000000
        (0x80000000, 0x00000001, 0),  # INT32_MIN - 1 wraps to 0x7fffffff
        (0x00000000, 0x80000000, 0),  # 0 - INT32_MIN wraps to 0x80000000
    ]),
    (OP_SUB, 1, 'sub_clamp', [
        (0x0000000a, 0x00000003, 0),  # 10 - 3 = 7 (no saturation)
        (0x7fffffff, 0xffffffff, 0),  # saturates to INT32_MAX (0x7fffffff)
        (0x80000000, 0x00000001, 0),  # saturates to INT32_MIN (0x80000000)
        (0x00000000, 0x80000000, 0),  # 0 - INT32_MIN saturates to INT32_MAX (0x7fffffff)
    ]),
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory(prefix='x2w_v_add_sub_nc_i32_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            first_vs = first_ps = first_exp = None
            for op, clamp, label, triples in DRAW_CASES:
                pipe = G.Pipeline(vs_words(triples), ps_words(op, clamp=clamp), root_blob())
                vs_bin = compile_hlsl(compiler, kind, pipe.hlsl_vs(), 'vs_%s_%s' % (label, kind), vs_profile, td)
                ps_bin = compile_hlsl(compiler, kind, pipe.hlsl_ps(), 'ps_%s_%s' % (label, kind), ps_profile, td)
                raw_exp = tuple(ref_op(op, a, b, clamp) ^ XOR_MASK for a, b, _ in triples)
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
        print('v_add_nc_i32 / v_sub_nc_i32 controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('v_add_nc_i32 / v_sub_nc_i32 draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_v_add_sub_nc_i32_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
