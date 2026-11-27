"""Saturating int16x2-to-uint8x2 pack (v_sat_pk_u8_i16) in tools/startup/gfx2hlsl.py.

  (no arguments)                       VOP1/VOP3A decode, value evaluation, EXEC masking, encoding & Stage
                                       refusals, unlowered and mutation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input words and compare the four result words

RDNA 2 ISA, VOP1 (0x62) / VOP3A (0x1e2):
  V_SAT_PK_U8_I16 vD, S0:
    D[7:0]   = clamp(int16(S0[15:0]),  0, 255)
    D[15:8]  = clamp(int16(S0[31:16]), 0, 255)
    D[31:16] = 0

The pixel fixture decodes synthetic gfx1013 VOP1/VOP3A bytes via G.decode() ONLY (no game data). It reads
flat parameter v0..v3, applies v_sat_pk_u8_i16 (both e32 and e64), XORs with 0x3f800000 via v_xor_b32 so the
packed uint16 output never lands in float32 subnormal range on float render targets, and exports mrt0.
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

# XOR with 0x3f800000 so 0x0000..0xffff outputs transport as normal IEEE-754 floats on R32G32B32A32_FLOAT.
DRAW_TRANSPORT_XOR = 0x3f800000


def vop1(op, vdst, src0, lit=None):
    w0 = 0x7e000000 | ((vdst & 0xff) << 17) | ((op & 0xff) << 9) | (src0 & 0x1ff)
    out = struct.pack('<I', w0)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


def vop3a_1src(op, vdst, src0, *, abs_bits=0, neg=0, clamp=0, omod=0, op_sel=0, src1=0, src2=0, lit=None):
    w0 = 0xd4000000 | ((op & 0x3ff) << 16) | (clamp << 15) | (op_sel << 11) | (abs_bits << 8) | (vdst & 0xff)
    w1 = (neg << 29) | (omod << 27) | ((src2 & 0x1ff) << 18) | ((src1 & 0x1ff) << 9) | (src0 & 0x1ff)
    out = struct.pack('<2I', w0, w1)
    if lit is not None:
        out += struct.pack('<I', lit & MASK)
    return out


def v_xor_lit(vdst, vsrc, lit):
    # VOP2 v_xor_b32_e32 v<vdst>, <literal>, v<vsrc> (op 0x1d, src0=255)
    w0 = (0x1d << 25) | ((vdst & 0xff) << 17) | ((vsrc & 0xff) << 9) | 255
    return struct.pack('<2I', w0, lit & MASK)


def ps_words(e64=False):
    body = b''.join(
        (vop3a_1src(0x1e2, 12 + lane, 256 + lane) if e64 else vop1(0x62, 12 + lane, 256 + lane))
        + v_xor_lit(12 + lane, 12 + lane, DRAW_TRANSPORT_XOR)
        for lane in range(4)
    )
    return PRE_WORDS + body + POST_WORDS


def translate(e64=False):
    zero = [(0, 0, 0)] * 4
    pipe = G.Pipeline(vs_words(zero), ps_words(e64), root_blob())
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def sext16(v):
    v &= 0xffff
    return v - 0x10000 if v >= 0x8000 else v


def ref_sat_pk_u8_i16(x):
    lo = max(0, min(255, sext16(x & 0xffff)))
    hi = max(0, min(255, sext16((x >> 16) & 0xffff)))
    return lo | (hi << 8)


def eval_hlsl(x):
    x &= MASK
    # Mirrors VEC_U['v_sat_pk_u8_i16']:
    lo_s = sext16(x & 0xffff)
    hi_s = sext16((x >> 16) & 0xffff)
    lo = max(0, min(255, lo_s)) & 0xff
    hi = max(0, min(255, hi_s)) & 0xff
    return lo | (hi << 8)


def stage_refused(stage, op, line, needle=''):
    try:
        stage.one(op, line)
    except G.Unsupported as exc:
        return (not needle or needle in str(exc)) and not stage.lines
    return False


HALVES = (-32768, -32767, -256, -1, 0, 1, 127, 128, 254, 255, 256, 1000, 32767)


def run_controls():
    # 1. Decode verification for both VOP1 (e32) and VOP3A (e64) via G.decode().
    check('decode v_sat_pk_u8_i16_e32',
          G.decode(vop1(0x62, 12, 256))[0][3] == 'v_sat_pk_u8_i16_e32 v12, v0')
    check('decode v_sat_pk_u8_i16_e64',
          G.decode(vop3a_1src(0x1e2, 12, 256))[0][3] == 'v_sat_pk_u8_i16_e64 v12, v0')

    # 2. Exhaustive 16-bit half-word pair grid (169 combinations).
    grid_ok = True
    for lo_s in HALVES:
        for hi_s in HALVES:
            word = (lo_s & 0xffff) | ((hi_s & 0xffff) << 16)
            if eval_hlsl(word) != ref_sat_pk_u8_i16(word):
                grid_ok = False
    check('v_sat_pk_u8_i16 grid (%d cases)' % (len(HALVES) * len(HALVES)), grid_ok)

    # 3. Both e32 and e64 pipelines translate cleanly; EXEC masking works.
    for e64 in (False, True):
        _, ps_hlsl = translate(e64)
        check('pipeline e64=%s translates' % e64, 'clamp(' in ps_hlsl and '<< 8u' in ps_hlsl)
    p = Probe()
    p.exec_known = False
    p.one('v_sat_pk_u8_i16_e32', 'v_sat_pk_u8_i16_e32 v12, v0')
    check('exec-masked v_sat_pk_u8_i16 guards assignment',
          len(p.lines) == 1 and p.lines[0].startswith('if (exec & 1) v12 = '))

    # 4. VOP3A encoding guard refuses neg/abs and unused src1/src2 fields.
    for label, bad_word, needle in (
        ('abs0', vop3a_1src(0x1e2, 12, 256, abs_bits=1), 'with neg/abs'),
        ('neg0', vop3a_1src(0x1e2, 12, 256, neg=1), 'with neg/abs'),
        ('neg_inline0', vop3a_1src(0x1e2, 12, 129, neg=1), 'with neg/abs'),
        ('unused_src1', vop3a_1src(0x1e2, 12, 256, src1=1), 'unused src1/src2'),
        ('unused_src2', vop3a_1src(0x1e2, 12, 256, src2=1), 'unused src1/src2'),
    ):
        reasons = []
        G.decode(bad_word, reasons)
        check('encoding guard refuses %s' % label,
              len(reasons) == 1 and needle in reasons[0][0])

    # 5. Stage refusals: modifiers, abs/neg, wrong arity.
    for asm in (
        'v_sat_pk_u8_i16_e64 v12, v0 clamp',
        'v_sat_pk_u8_i16_e64 v12, v0 mul:2',
        'v_sat_pk_u8_i16_e64 v12, v0 op_sel:[1,0]',
        'v_sat_pk_u8_i16_e64 v12, |v0|',
        'v_sat_pk_u8_i16_e64 v12, -v0',
        'v_sat_pk_u8_i16_e32 v12',
        'v_sat_pk_u8_i16_e32 v12, v0, v1',
    ):
        check('Stage refuses %s' % asm, stage_refused(Probe(), asm.split()[0], asm))

    # 6. Unlowered control: removing v_sat_pk_u8_i16 from VEC_U restores refusals.
    saved = G.VEC_U.pop('v_sat_pk_u8_i16')
    try:
        check('unlowered v_sat_pk_u8_i16 refused',
              stage_refused(Probe(), 'v_sat_pk_u8_i16_e32', 'v_sat_pk_u8_i16_e32 v12, v0'))
        try:
            translate(False)
        except G.Unsupported:
            check('unlowered pipeline refused', True)
        else:
            check('unlowered pipeline refused', False)
    finally:
        G.VEC_U['v_sat_pk_u8_i16'] = saved


DRAW_BATCHES = (
    # Each tuple is 4 input words (packed as (lo_i16, hi_i16) per lane).
    (
        (0x0000 | (0x00ff << 16)),       # (0, 255) -> 0xff00
        (0xffff | (0x8000 << 16)),       # (-1, -32768) -> 0x0000
        (0x0100 | (0x7fff << 16)),       # (256, 32767) -> 0xffff
        (0x0042 | (0x00ab << 16)),       # (0x42, 0xab) -> 0xab42
    ),
    (
        (0x00fe | (0xfffe << 16)),       # (254, -2) -> 0x00fe
        (0x8001 | (0x007f << 16)),       # (-32767, 127) -> 0x7f00
        (0x0001 | (0x0101 << 16)),       # (1, 257) -> 0xff01
        (0x0080 | (0x0080 << 16)),       # (128, 128) -> 0x8080
    ),
)


def run_draw(dxc, renderer, adapter, fxc=None):
    run_controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory(prefix='x2w_v_sat_pk_u8_i16_') as tmp:
        td = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            first_vs = None
            first_ps = None
            first_exp = None
            for e64 in (False, True):
                _, ps_hlsl = translate(e64)
                ps_bin = compile_hlsl(compiler, kind, ps_hlsl, 'ps_e%d_%s' % (64 if e64 else 32, kind), ps_profile, td)
                for b_idx, batch in enumerate(DRAW_BATCHES):
                    triples = [(w, 0, 0) for w in batch]
                    vs_hlsl = G.Pipeline(vs_words(triples), ps_words(e64), root_blob()).hlsl_vs()
                    vs_bin = compile_hlsl(compiler, kind, vs_hlsl,
                                          'vs_e%d_%d_%s' % (64 if e64 else 32, b_idx, kind), vs_profile, td)
                    raw_exp = tuple(ref_sat_pk_u8_i16(w) ^ DRAW_TRANSPORT_XOR for w in batch)
                    expected = words_text(raw_exp)
                    res = draw(renderer, adapter, vs_bin, ps_bin, expected)
                    if res.returncode == 77:
                        print('SKIP (%s): %s' % (adapter, res.stdout.strip() or res.stderr.strip()))
                        sys.exit(77)
                    if res.returncode != 0:
                        raise AssertionError('%s %s e%d batch=%d expected=%s\n%s\n%s' % (
                            kind, adapter, 64 if e64 else 32, b_idx, expected, res.stdout, res.stderr))
                    print('%s %s e%d batch=%d: %s' % (kind, adapter, 64 if e64 else 32, b_idx, expected), flush=True)
                    if first_vs is None:
                        first_vs, first_ps, first_exp = vs_bin, ps_bin, raw_exp

            bad = words_text((first_exp[0] ^ 1, first_exp[1], first_exp[2], first_exp[3]))
            neg = draw(renderer, adapter, first_vs, first_ps, bad)
            check('draw %s %s negative control rejects wrong expectation' % (kind, adapter),
                  neg.returncode != 0)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run_controls()
        print('v_sat_pk_u8_i16 controls: ok')
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        run_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
        print('v_sat_pk_u8_i16 draw (%s): ok' % sys.argv[4])
    else:
        print('usage: startup_gfx2hlsl_v_sat_pk_u8_i16_check.py [draw <dxc> <offscreen_check> <warp|hardware> [fxc]]',
              file=sys.stderr)
        sys.exit(2)
