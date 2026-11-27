"""Verify that v_ldexp_f32 lowers to ldexp32 with valid float/integer operands and refuses invalid modifiers.

Refs #2618.
"""
from pathlib import Path
import re
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))

import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_compute_check as C  # noqa: E402


def vop3a(op, vdst, src0, src1, src2=0, *, abs_bits=0, clamp=0, omod=0, neg=0, lit=None):
    """Encode one synthetic gfx1013 VOP3A instruction."""
    w0 = 0xd4000000 | (op << 16) | (clamp << 15) | (abs_bits << 8) | vdst
    w1 = (neg << 29) | (omod << 27) | (src2 << 18) | (src1 << 9) | src0
    out = struct.pack('<2I', w0, w1)
    if lit is not None:
        out += struct.pack('<I', lit & 0xffffffff)
    return out


ENDPGM = struct.pack('<I', 0xbf800000)


def translate_cs(code):
    return G.ComputePipeline(code + ENDPGM, C.root_blob(), C.CASES['A'][2], (2, 3, 4)).hlsl_cs()


def check_translations():
    # 1. Preamble contains ldexp32
    hlsl_base = translate_cs(vop3a(0x362, 5, 240, 130))
    assert 'float ldexp32(float value, int delta) {' in hlsl_base, 'preamble missing ldexp32'
    assert 'v5 = U(ldexp32(0.5, asint(0x00000002u)));' in hlsl_base, f'unexpected hlsl: {hlsl_base}'

    # 2. Float input modifiers on src0:
    # Negation on inline float
    hlsl_neg = translate_cs(vop3a(0x362, 5, 240, 130, neg=1))
    assert 'v5 = U(ldexp32(-0.5, asint(0x00000002u)));' in hlsl_neg, f'unexpected neg: {hlsl_neg}'

    # Negation on negative inline float -0.5 (encoding 241):
    # Must format as (-(-0.5)) to avoid HLSL pre-decrement syntax (--0.5)
    hlsl_neg_minus_half = translate_cs(vop3a(0x362, 5, 241, 130, neg=1))
    assert 'v5 = U(ldexp32((-(-0.5)), asint(0x00000002u)));' in hlsl_neg_minus_half, f'unexpected neg -0.5: {hlsl_neg_minus_half}'
    assert '--0.5' not in hlsl_neg_minus_half, f'decrement operator found in hlsl: {hlsl_neg_minus_half}'

    # Negation on VGPR src0
    hlsl_vneg = translate_cs(vop3a(0x362, 5, 257, 130, neg=1))
    assert 'v5 = U(ldexp32((-F(v1)), asint(0x00000002u)));' in hlsl_vneg, f'unexpected vneg: {hlsl_vneg}'

    # Absolute on VGPR src0
    hlsl_vabs = translate_cs(vop3a(0x362, 5, 257, 130, abs_bits=1))
    assert 'v5 = U(ldexp32(abs(F(v1)), asint(0x00000002u)));' in hlsl_vabs, f'unexpected vabs: {hlsl_vabs}'

    # Combined -|src0| on VGPR
    hlsl_neg_abs = translate_cs(vop3a(0x362, 5, 257, 130, neg=1, abs_bits=1))
    assert 'v5 = U(ldexp32((-abs(F(v1))), asint(0x00000002u)));' in hlsl_neg_abs, f'unexpected neg abs: {hlsl_neg_abs}'

    # 3. Exponent operand types:
    # Negative inline integer constant -2 (encoding 194)
    hlsl_neg_exp = translate_cs(vop3a(0x362, 5, 240, 194))
    assert 'v5 = U(ldexp32(0.5, asint(0xfffffffeu)));' in hlsl_neg_exp, f'unexpected neg exp: {hlsl_neg_exp}'

    # Exponent from SGPR s4 (user data ud0, encoding 4)
    hlsl_sreg = translate_cs(vop3a(0x362, 5, 240, 4))
    assert 'v5 = U(ldexp32(0.5, asint(ud0)));' in hlsl_sreg, f'unexpected sreg: {hlsl_sreg}'

    # Exponent from VGPR v2 (encoding 258)
    hlsl_vreg = translate_cs(vop3a(0x362, 5, 240, 258))
    assert 'v5 = U(ldexp32(0.5, asint(v2)));' in hlsl_vreg, f'unexpected vreg: {hlsl_vreg}'

    # 4. Output modifiers:
    # clamp (generates saturate)
    hlsl_clamp = translate_cs(vop3a(0x362, 5, 240, 130, clamp=1))
    assert 'v5 = U(saturate(ldexp32(0.5, asint(0x00000002u))));' in hlsl_clamp, f'unexpected clamp: {hlsl_clamp}'

    # omod mul:2
    hlsl_mul2 = translate_cs(vop3a(0x362, 5, 240, 130, omod=1))
    assert 'v5 = U((ldexp32(0.5, asint(0x00000002u))) * 2.0);' in hlsl_mul2, f'unexpected mul2: {hlsl_mul2}'


def make_stage(kind='cs'):
    st = G.Stage.__new__(G.Stage)
    st.kind = kind
    st.lines = []
    st.sym = {}
    st.used_s, st.used_v, st.uniform_v = set(), set(), set()
    st.varying_s = set()
    st.scc_uniform = True
    st.uniform_expr = {}
    st.past_entry = False
    st.has_branches = False
    st.exec_known = True
    st.masks = set()
    st.exports = {}
    st.low16_only_v = set()
    st.pinned_low16_v = set()
    st.whole_read_v = set()
    st.byte_float_v = set()
    st.bad_sgprs = set()
    st.defined_uniform_s = None
    st.wave_reductions = []
    st.emitted_reductions = []
    st.wave_reduce_used_exec = False
    return st


def check_refusals():
    # Helper to test Stage.one directly
    def lower_inst(asm_line):
        name = asm_line.split()[0]
        st = make_stage('cs')
        return st.one(name, asm_line)

    # 1. Float modifier on integer exponent operand:
    bad_modifiers = (
        ('v_ldexp_f32 v0, v1, -s4', 'v_ldexp_f32 exponent -s4 has float modifier'),
        ('v_ldexp_f32 v0, v1, -v2', 'v_ldexp_f32 exponent -v2 has float modifier'),
        ('v_ldexp_f32 v0, v1, |s4|', 'v_ldexp_f32 exponent |s4| has float modifier'),
        ('v_ldexp_f32 v0, v1, abs(s4)', 'v_ldexp_f32 exponent abs(s4) has float modifier'),
    )
    for asm, expected in bad_modifiers:
        try:
            lower_inst(asm)
            raise AssertionError(f'expected refusal for {asm}, but succeeded')
        except G.Unsupported as e:
            assert expected in str(e), f'unexpected message for {asm}: {e}'

    # 2. Exponent out of 32-bit range:
    try:
        lower_inst('v_ldexp_f32 v0, v1, 0x100000000')
        raise AssertionError('expected refusal for 0x100000000, but succeeded')
    except G.Unsupported as e:
        assert 'v_ldexp_f32 source 0x100000000' in str(e), f'unexpected message: {e}'

    # 3. Operand counts:
    for asm in ('v_ldexp_f32 v0, v1', 'v_ldexp_f32 v0, v1, v2, v3'):
        try:
            lower_inst(asm)
            raise AssertionError(f'expected refusal for {asm}, but succeeded')
        except G.Unsupported as e:
            assert 'v_ldexp_f32 with' in str(e), f'unexpected message: {e}'

    # 4. Destination not a VGPR:
    try:
        lower_inst('v_ldexp_f32 s0, v1, v2')
        raise AssertionError('expected refusal for destination s0, but succeeded')
    except G.Unsupported as e:
        assert 'v_ldexp_f32 destination s0' in str(e), f'unexpected message: {e}'

    # 5. Disallowed instruction forms:
    for asm, expected in (
        ('v_ldexp_f32_e32 v0, v1, v2', 'v_ldexp_f32_e32'),
        ('v_ldexp_f32_sdwa v0, v1, v2', 'SDWA form of v_ldexp_f32'),
        ('v_ldexp_f32_dpp v0, v1, v2', 'DPP form of v_ldexp_f32'),
        ('v_ldexp_f32_dpp v0, v1, v2 quad_perm:[1,1,3,3] row_mask:0xf bank_mask:0xf', 'DPP form of v_ldexp_f32'),
    ):
        try:
            lower_inst(asm)
            raise AssertionError(f'expected refusal for {asm}, but succeeded')
        except G.Unsupported as e:
            assert expected in str(e), f'unexpected message: {e}'

    # 6. Machine-code VOP3 check_encoding refusals:
    # NEG bit on src1 exponent (w1[30])
    try:
        G.decode(vop3a(0x362, 0, 240, 130, neg=2))
        raise AssertionError('expected decode refusal for NEG on exponent')
    except G.Unsupported as e:
        assert 'v_ldexp_f32 with neg/abs on exponent or unused source' in str(e), f'unexpected: {e}'

    # ABS bit on src1 exponent (w0[9])
    try:
        G.decode(vop3a(0x362, 0, 240, 130, abs_bits=2))
        raise AssertionError('expected decode refusal for ABS on exponent')
    except G.Unsupported as e:
        assert 'v_ldexp_f32 with neg/abs on exponent or unused source' in str(e), f'unexpected: {e}'

    # Unused src2 field set (w1[26:18])
    try:
        G.decode(vop3a(0x362, 0, 240, 130, src2=1))
        raise AssertionError('expected decode refusal for src2 set')
    except G.Unsupported as e:
        assert 'v_ldexp_f32 with unused src2 field set' in str(e), f'unexpected: {e}'


def check_negative_controls():
    # Negative control 1: pre-fix mutant where op_v_ldexp_f32 is missing
    real_handler = getattr(G.Stage, 'op_v_ldexp_f32', None)
    try:
        delattr(G.Stage, 'op_v_ldexp_f32')
        try:
            translate_cs(vop3a(0x362, 5, 240, 130))
            raise AssertionError('expected translation failure under mutant')
        except G.Unsupported as e:
            assert 'v_ldexp_f32' in str(e), f'unexpected mutant error: {e}'
    finally:
        if real_handler:
            setattr(G.Stage, 'op_v_ldexp_f32', real_handler)

    # Negative control 2: mutant that treats exponent as float (accepting float modifiers)
    real_op = G.Stage.op_v_ldexp_f32

    def mutant_op(self, ops, mods):
        dst = G.reg_range(ops[0])
        src0 = self.f(ops[1])
        src1 = self.f(ops[2])
        self.set_vf(dst[1], 'ldexp32(%s, %s)' % (src0, src1), mods)

    try:
        G.Stage.op_v_ldexp_f32 = mutant_op
        # Under this mutant, float modifier on exponent would escape refusal
        st = make_stage('cs')
        st.lower('v_ldexp_f32', 'v_ldexp_f32 v0, v1, -s4', ['v0', 'v1', '-s4'], [])
        assert any('(-s4)' in line or 'F(s4)' in line for line in st.lines), 'mutant failed to reproduce bypass'
    finally:
        G.Stage.op_v_ldexp_f32 = real_op


def check_decode_and_lower():
    # Verify a synthetic VOP3A v_ldexp_f32 word decodes and lowers through Stage
    code = vop3a(0x362, 5, 240, 130)
    ins = G.decode(code)
    assert len(ins) == 1 and ins[0][2] == 'v_ldexp_f32', f'unexpected decode: {ins}'
    st = make_stage('vs')
    name, ops, mods = G.parse(ins[0][3])
    st.lower(ins[0][2], ins[0][3], ops, mods)
    assert any('ldexp32' in line for line in st.lines), 'ldexp32 missing from lowered stage'


def check_helper_parity():
    # Verify exact parity between the ldexp32 definition in Pipeline.common() and tools/pixel_aot.py
    pixel_aot_path = HERE.parent / 'tools' / 'pixel_aot.py'
    pixel_aot_text = pixel_aot_path.read_text(encoding='utf-8')
    m_aot = re.search(r'float ldexp32\(float value, int delta\) \{.*?^\}', pixel_aot_text, re.M | re.S)
    assert m_aot, 'ldexp32 helper missing from tools/pixel_aot.py'
    aot_ldexp = m_aot.group(0)

    pipe = G.ComputePipeline(b'', C.root_blob(), C.CASES['A'][2], (1, 1, 1))
    common_text = '\n'.join(pipe.common())
    m_gfx = re.search(r'float ldexp32\(float value, int delta\) \{.*?^\}', common_text, re.M | re.S)
    assert m_gfx, 'ldexp32 helper missing from Pipeline.common()'
    gfx_ldexp = m_gfx.group(0)

    def normalize(code):
        return [line.strip() for line in code.splitlines() if line.strip()]

    assert normalize(gfx_ldexp) == normalize(aot_ldexp), (
        'ldexp32 in gfx2hlsl Pipeline.common() does not match tools/pixel_aot.py'
    )

    # Negative control: an altered helper fails parity
    mutant = gfx_ldexp.replace('clamp(delta, -512, 512)', 'delta')
    assert normalize(mutant) != normalize(aot_ldexp), 'mutant failed to differ from pixel_aot.py'

    # Pure-Python simulation of ldexp32 logic on edge cases
    def ldexp32_sim(bits, delta):
        sign = bits & 0x80000000
        exp = (bits >> 23) & 0xff
        mant = bits & 0x7fffff
        if exp == 0xff or (exp == 0 and mant == 0):
            return bits
        e = exp
        if exp == 0:
            shift = 23 - (mant.bit_length() - 1)
            mant = (mant << shift) & 0x7fffff
            e = 1 - shift
        e += max(-512, min(512, delta))
        if e >= 255:
            return sign | 0x7f800000
        if e <= 0:
            if e < -23:
                return sign
            full = mant | 0x800000
            shift = 1 - e
            half_bit = 1 << (shift - 1)
            mask = (1 << shift) - 1
            rem = full & mask
            q = full >> shift
            if rem > half_bit or (rem == half_bit and (q & 1) != 0):
                q += 1
            return sign | q
        return sign | (e << 23) | mant

    def f2u(f): return struct.unpack('<I', struct.pack('<f', f))[0]
    def u2f(u): return struct.unpack('<f', struct.pack('<I', u))[0]

    # Normal scaling
    assert u2f(ldexp32_sim(f2u(1.0), 1)) == 2.0
    assert u2f(ldexp32_sim(f2u(1.0), -1)) == 0.5
    # Signed zero preserved
    assert ldexp32_sim(f2u(0.0), 10) == f2u(0.0)
    assert ldexp32_sim(f2u(-0.0), 10) == f2u(-0.0)
    # Infinity and NaN preserved
    assert ldexp32_sim(0x7f800000, 10) == 0x7f800000
    assert ldexp32_sim(0xff800000, -10) == 0xff800000
    assert ldexp32_sim(0x7fc00001, 5) == 0x7fc00001
    # Clamping overflow and underflow
    assert ldexp32_sim(f2u(1.0), 1000) == 0x7f800000
    assert ldexp32_sim(f2u(1.0), -1000) == 0
    # Subnormal 2^-150 tie to even (q=0 is even -> rounds to 0)
    assert ldexp32_sim(f2u(1.0), -150) == 0
    # Subnormal 1.5 * 2^-150 rounds up to 1 ULP
    assert ldexp32_sim(f2u(1.5), -150) == 1


def main():
    check_helper_parity()
    check_translations()
    check_refusals()
    check_negative_controls()
    check_decode_and_lower()
    print('gfx2hlsl v_ldexp_f32 lowering and refusal checks passed')


if __name__ == '__main__':
    main()
