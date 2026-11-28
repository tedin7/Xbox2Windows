"""64-bit scalar bit-replicate (s_bitreplicate_b64_b32) in tools/startup/gfx2hlsl.py.

  (no arguments)                       SOP1 decode, helper emission, value evaluation, aliasing, constant
                                       folding, uniformity, refusal and mutation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per 32-bit input and compare the four result words

RDNA 2 ISA, SOP1 (0x3b):
  S_BITREPLICATE_B64_B32 s[D:D+1], S0: D[2*i] = D[2*i+1] = S0[i] for i in 0..31; SCC unchanged.

The pixel fixture decodes synthetic gfx1013 bytes via G.decode() ONLY (no game data). It loads a 32-bit
value into runtime SGPRs s6 and s4, runs disjoint s_bitreplicate_b64_b32 s[8:9], s6 and aliased
s_bitreplicate_b64_b32 s[4:5], s4, XORs the four output words with 0x3f800000 for normal IEEE-754 transport
into v12..v15, and exports mrt0.
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


def ps_words(val):
    """Synthetic gfx1013 pixel stage exercising disjoint and aliased s_bitreplicate_b64_b32."""
    body = b''.join([
        sop1_mov_lit(6, val),
        s_xor_lit(6, 6, 0),                   # clear const sym so runtime helper is emitted
        sop1_mov_lit(4, val),
        s_xor_lit(4, 4, 0),
        sop1(0x3b, 8, 6),                     # disjoint: s_bitreplicate_b64_b32 s[8:9], s6
        sop1(0x3b, 4, 4),                     # aliased:  s_bitreplicate_b64_b32 s[4:5], s4
        s_xor_lit(8, 8, DRAW_TRANSPORT_XOR),
        s_xor_lit(9, 9, DRAW_TRANSPORT_XOR),
        s_xor_lit(4, 4, DRAW_TRANSPORT_XOR),
        s_xor_lit(5, 5, DRAW_TRANSPORT_XOR),
        v_mov_b32_from_sgpr(12, 8),
        v_mov_b32_from_sgpr(13, 9),
        v_mov_b32_from_sgpr(14, 4),
        v_mov_b32_from_sgpr(15, 5),
    ])
    return PRE_WORDS + body + POST_WORDS


def translate_ps(val=0x00010001):
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(val), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def ref_bitreplicate(val):
    """Independent bit-string reference returning (lo, hi)."""
    bits = ''.join(ch * 2 for ch in format(val & MASK, '032b'))
    out = int(bits, 2)
    return out & MASK, (out >> 32) & MASK


def _bitreplicate16(x):
    x &= 0xffff
    x = (x | (x << 8)) & 0x00ff00ff
    x = (x | (x << 4)) & 0x0f0f0f0f
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return (x | (x << 1)) & MASK


def eval_u32(expr, env):
    text = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', expr)
    return eval(text, {'__builtins__': {}, 'bitreplicate16': _bitreplicate16}, env) & MASK


def run_lines(line, env, *, sym=None):
    stage = Probe()
    if sym:
        stage.sym.update(sym)
    stage.one('s_bitreplicate_b64_b32', line)
    out = dict(env)
    for stmt in stage.lines:
        m = re.fullmatch(r'(\w+) = (.*);', stmt)
        assert m, stmt
        assert m.group(1) != 'scc', 'SCC must remain unchanged: %s' % stmt
        out[m.group(1)] = eval_u32(m.group(2), out)
    return out, stage


def stage_refused(stage, line, needle=''):
    try:
        stage.one('s_bitreplicate_b64_b32', line)
    except G.Unsupported as exc:
        return (not needle or needle in str(exc)) and not stage.lines
    return False


VALUES = (
    0x00000000, 0xffffffff, 0x00000001, 0x80000000, 0x00008000, 0x00010000,
    0x55555555, 0xaaaaaaaa, 0x12345678, 0x9abcdef0, 0x00ff00ff, 0xff00ff00,
)


def run_controls():
    # 1. Decode verification from synthetic gfx1013 SOP1 bytes via G.decode().
    decoded = [
        G.decode(sop1(0x3b, 8, 6))[0][3],
        G.decode(sop1(0x3b, 4, 4))[0][3],
        G.decode(sop1(0x3b, 4, 193))[0][3],
        G.decode(sop1(0x3b, 4, 255, lit=0x12345678))[0][3],
        G.decode(sop1(0x3b, 4, 125))[0][3],
    ]
    check('decode s_bitreplicate_b64_b32 disjoint', decoded[0] == 's_bitreplicate_b64_b32 s[8:9], s6')
    check('decode s_bitreplicate_b64_b32 aliased', decoded[1] == 's_bitreplicate_b64_b32 s[4:5], s4')
    check('decode s_bitreplicate_b64_b32 inline -1', decoded[2] == 's_bitreplicate_b64_b32 s[4:5], -1')
    check('decode s_bitreplicate_b64_b32 literal', decoded[3] == 's_bitreplicate_b64_b32 s[4:5], 0x12345678')
    check('decode s_bitreplicate_b64_b32 null', decoded[4] == 's_bitreplicate_b64_b32 s[4:5], null')

    # 2. Disjoint and aliased (s[4:5], s4 and s[4:5], s5) evaluation (SCC untouched).
    disjoint_ok = True
    alias_lo_ok = True
    alias_hi_ok = True
    for val in VALUES:
        want = ref_bitreplicate(val)
        out_d, _ = run_lines('s_bitreplicate_b64_b32 s[8:9], s6', {'s6': val})
        if (out_d['s8'], out_d['s9']) != want:
            disjoint_ok = False
        out_l, _ = run_lines('s_bitreplicate_b64_b32 s[4:5], s4', {'s4': val, 's5': 0xdeadbeef})
        if (out_l['s4'], out_l['s5']) != want:
            alias_lo_ok = False
        out_h, _ = run_lines('s_bitreplicate_b64_b32 s[4:5], s5', {'s4': 0xdeadbeef, 's5': val})
        if (out_h['s4'], out_h['s5']) != want:
            alias_hi_ok = False
    check('disjoint SGPR (%d cases)' % len(VALUES), disjoint_ok)
    check('aliased s[4:5], s4 (%d cases)' % len(VALUES), alias_lo_ok)
    check('aliased s[4:5], s5 (%d cases)' % len(VALUES), alias_hi_ok)

    # 3. Constant folding (inline, literal, null, const-tracked SGPR).
    for val in (0, 1, 64, -1, -16, 0x12345678):
        want = ref_bitreplicate(val)
        tok = hex(val) if val > 64 else str(val)
        out, st = run_lines('s_bitreplicate_b64_b32 s[4:5], %s' % tok, {})
        check('const %s folds' % tok,
              (out['s4'], out['s5']) == want and st.sym.get(4) == ('const', want[0])
              and st.sym.get(5) == ('const', want[1]))
    out_n, st_n = run_lines('s_bitreplicate_b64_b32 s[4:5], null', {})
    check('null folds to (0, 0)', (out_n['s4'], out_n['s5']) == (0, 0) and st_n.sym.get(4) == ('const', 0))
    out_cs, st_cs = run_lines('s_bitreplicate_b64_b32 s[4:5], s6', {}, sym={6: ('const', 0x00018001)})
    want_cs = ref_bitreplicate(0x00018001)
    check('const-tracked SGPR folds', (out_cs['s4'], out_cs['s5']) == want_cs and st_cs.sym.get(5) == ('const', want_cs[1]))

    # 4. Uniformity tracking.
    s = Probe()
    s.varying_s = {6}
    s.scc_uniform = True
    s.one('s_bitreplicate_b64_b32', 's_bitreplicate_b64_b32 s[4:5], s6')
    check('varying source taints s4 and s5', 4 in s.varying_s and 5 in s.varying_s)
    s = Probe()
    s.varying_s = {4, 5}
    s.scc_uniform = True
    s.one('s_bitreplicate_b64_b32', 's_bitreplicate_b64_b32 s[4:5], s6')
    check('uniform source clears varying on s4 and s5', 4 not in s.varying_s and 5 not in s.varying_s)

    # 5. Refusals (arity, modifiers, lane masks, mask_high, descriptors, invalid register classes/alignment).
    check('1 operand refused', stage_refused(Probe(), 's_bitreplicate_b64_b32 s[4:5]', 'expected 2 unmodified'))
    check('3 operands refused', stage_refused(Probe(), 's_bitreplicate_b64_b32 s[4:5], s6, s7', 'expected 2 unmodified'))
    check('modifier refused', stage_refused(Probe(), 's_bitreplicate_b64_b32 s[4:5], s6 clamp', 'expected 2 unmodified'))
    for operands in ('exec, s6', 'vcc, s6', 's[4:5], exec_lo', 's[4:5], vcc_lo'):
        check('%s lane mask refused' % operands, stage_refused(Probe(), 's_bitreplicate_b64_b32 %s' % operands))
    for mask_reg, high_reg, operands in (
        (12, None, 's[12:13], s6'),
        (13, None, 's[12:13], s6'),
        (None, 13, 's[12:13], s6'),
        (12, None, 's[4:5], s12'),
        (None, 13, 's[4:5], s13'),
    ):
        p = Probe()
        if mask_reg is not None:
            p.masks = {mask_reg}
        if high_reg is not None:
            p.mask_high = {high_reg}
        check('%s tracked mask refused' % operands,
              stage_refused(p, 's_bitreplicate_b64_b32 %s' % operands, 'lane mask'))
    for bad_dst in ('v[0:1]', 's4', 's[5:6]', 's[106:107]', 'm0'):
        check('bad dst %s refused' % bad_dst,
              stage_refused(Probe(), 's_bitreplicate_b64_b32 %s, s6' % bad_dst, 'destination'))
    for bad_src in ('v0', 's[6:7]', 's106', '1.0', 'm0', 'src_scc'):
        check('bad src %s refused' % bad_src,
              stage_refused(Probe(), 's_bitreplicate_b64_b32 s[4:5], %s' % bad_src, 'source'))
    p_desc = Probe()
    p_desc.sym[6] = ('descriptor', 0)
    check('descriptor sym on s6 refused',
          stage_refused(p_desc, 's_bitreplicate_b64_b32 s[4:5], s6', 'descriptor data'))

    # 6. On-demand helper emission: emitted when used dynamically, omitted when unused.
    _, ps_hlsl = translate_ps()
    check('bitreplicate16 helper emitted on demand', 'uint bitreplicate16(uint x)' in ps_hlsl)
    zero = [(0, 0, 0)] * 4
    unused_ps = G.Pipeline(vs_words(zero), PRE_WORDS + POST_WORDS, root_blob()).hlsl_ps()
    check('bitreplicate16 helper omitted when unused', 'bitreplicate16' not in unused_ps)

    # 7. Unlowered control: removing op_s_bitreplicate_b64_b32 from Stage restores refusals.
    saved = G.Stage.op_s_bitreplicate_b64_b32
    try:
        delattr(G.Stage, 'op_s_bitreplicate_b64_b32')
        check('unlowered s_bitreplicate_b64_b32 refused',
              stage_refused(Probe(), 's_bitreplicate_b64_b32 s[8:9], s6'))
        try:
            translate_ps()
        except G.Unsupported:
            check('unlowered synthetic pipeline refused', True)
        else:
            check('unlowered synthetic pipeline refused', False)
    finally:
        G.Stage.op_s_bitreplicate_b64_b32 = saved


# Keep bits [14:11] and [30:27] clear so replicated bits [29:22] are 0 and XOR 0x3f800000 stays normal IEEE-754.
GPU_VECTORS = (
    ('bit0_and_bit16', 0x00010001),
    ('bit15_and_31',   0x80008000),
    ('low_pattern',    0x000005a3),
    ('high_pattern',   0x05a30000),
    ('mixed_pattern',  0x01230456),
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory(prefix='x2w_s_bitreplicate_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            first_vs = None
            first_ps = None
            first_exp = None
            for label, val in GPU_VECTORS:
                vs_hlsl, ps_hlsl = translate_ps(val)
                vs_bin = compile_hlsl(compiler, kind, vs_hlsl, 'vs_%s_%s' % (label, kind), vs_profile, td)
                ps_bin = compile_hlsl(compiler, kind, ps_hlsl, 'ps_%s_%s' % (label, kind), ps_profile, td)
                r0, r1 = ref_bitreplicate(val)
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
        print('s_bitreplicate controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('s_bitreplicate draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_s_bitreplicate_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
