"""Direct compute binary16 FMA (v_fma_f16, v_fmac_f16, v_fmamk_f16, v_fmaak_f16)."""
from fractions import Fraction
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request
import native_f16_alu_check as F
from native_scalar_lds_check import scalar, sop1


HI_DST = 0xABCD0000
HI_A = 0xBEEF0000
HI_B = 0x789A0000
HI_C = 0x13570000

OP_FMAC, OP_FMAMK, OP_FMAAK = 0x36, 0x37, 0x38
OP_FMA_VOP3 = 0x34B
OP_FMAC_VOP3 = 0x136


def vop3_tern(op, vdst, s0, s1, s2, *, abs_mask=0, neg=0, clamp=0, omod=0, op_sel=0):
    w0 = (0xD4000000 | ((op & 0x3FF) << 16) | ((clamp & 1) << 15) |
          ((op_sel & 0xF) << 11) | ((abs_mask & 7) << 8) | (vdst & 0xFF))
    w1 = ((neg & 7) << 29) | ((omod & 3) << 27) | ((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) | (s0 & 0x1FF)
    return [w0, w1]


def half_to_fraction(bits):
    bits &= 0xFFFF
    sign = -1 if (bits & 0x8000) else 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0x1F:
        return None
    if exp == 0:
        return Fraction(sign * mant, 1 << 24)
    return Fraction(sign * (1024 + mant), 1 << (25 - exp)) if exp < 25 else Fraction(sign * (1024 + mant) * (1 << (exp - 25)), 1)


def round_fraction_f16(q, neg_zero=False):
    if q == 0:
        return 0x8000 if neg_zero else 0x0000
    sign = 0x8000 if q < 0 else 0x0000
    aq = abs(q)
    # Overflow boundary: 65520 = 65504 + 16 -> +inf
    if aq >= Fraction(65520, 1):
        return sign | 0x7C00
    # Find largest e >= -24 such that 2^e <= aq, or subnormal grid 2^-24.
    if aq < Fraction(1, 1 << 14):
        scaled = aq * (1 << 24)
        n = scaled.numerator // scaled.denominator
        rem = scaled - n
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
            n += 1
        return sign | n
    # Normal range: 2^e <= aq < 2^(e+1)
    num, den = aq.numerator, aq.denominator
    e = num.bit_length() - den.bit_length()
    if aq < (Fraction(1 << e, 1) if e >= 0 else Fraction(1, 1 << (-e))):
        e -= 1
    elif aq >= (Fraction(1 << (e + 1), 1) if e + 1 >= 0 else Fraction(1, 1 << (-(e + 1)))):
        e += 1
    shift = e - 10
    step = Fraction(1 << shift, 1) if shift >= 0 else Fraction(1, 1 << (-shift))
    scaled = aq / step
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
        n += 1
    if n == 2048:
        n = 1024
        e += 1
    if e > 15:
        return sign | 0x7C00
    return sign | ((e + 15) << 10) | (n & 0x3FF)


def ref_fma16(a_bits, b_bits, c_bits, *, abs_mask=0, neg=0, clamp=False):
    a = F.modified(a_bits, abs_mask, neg, 0)
    b = F.modified(b_bits, abs_mask, neg, 1)
    c = F.modified(c_bits, abs_mask, neg, 2)

    def cls(h):
        e, f = (h >> 10) & 0x1F, h & 0x3FF
        return e == 0x1F and f != 0, e == 0x1F and f == 0, (h & 0x7FFF) == 0, (h >> 15) & 1

    an, ai, az, sa = cls(a)
    bn, bi, bz, sb = cls(b)
    cn, ci, cz, sc = cls(c)
    sp = sa ^ sb
    if an or bn or cn or (ai and bz) or (bi and az) or ((ai or bi) and ci and sp != sc):
        out = 0x7E00
    elif ai or bi:
        out = (sp << 15) | 0x7C00
    elif ci:
        out = (sc << 15) | 0x7C00
    else:
        exact = half_to_fraction(a) * half_to_fraction(b) + half_to_fraction(c)
        out = round_fraction_f16(exact, neg_zero=(az or bz) and cz and sp == 1 and sc == 1)
    if clamp:
        if (out & 0x7FFF) > 0x7C00 or (out & 0x8000):
            out = 0
        elif out > 0x3BFF:
            out = 0x3C00
    return out


# #2073 double-rounding tie witnesses: f32 a*b+c rounded to f16 disagrees with exact single-rounded FMA.
TIE_WITNESSES_2073 = (
    (0x4A54, 0x1EF4, 0x4E2D, 0x4E33),
    (0x51EA, 0xBF3E, 0x1D5F, 0xD55B),
    (0x2BD5, 0xBE83, 0x4528, 0x450F),
    (0x3915, 0x387A, 0x4B71, 0x4B9F),
)


def cases():
    items = []

    def add(name, code, want, dst=3):
        items.append((name, code, want, dst))

    # 1. All four #2073 single-rounding witnesses across v_fma_f16, v_fmac_f16, v_fmamk_f16, v_fmaak_f16.
    for a, b, c, want in TIE_WITNESSES_2073:
        assert ref_fma16(a, b, c) == want
        assert F.packed_half_rne(
            struct.unpack('<I', struct.pack('<f', F.M.h2f(a) * F.M.h2f(b) + F.M.h2f(c)))[0]
        ) != want
        add(f'fma2073 {hex(a)}',
            F.mov(1, HI_A | a) + F.mov(2, HI_B | b) + F.mov(4, HI_C | c) + F.mov(3, HI_DST | 0x1111) +
            vop3_tern(OP_FMA_VOP3, 3, 257, 258, 260), want)
        add(f'fmac2073 {hex(a)}',
            F.mov(1, HI_A | a) + F.mov(2, HI_B | b) + F.mov(3, HI_DST | c) + F.vop2(OP_FMAC, 3, 257, 2), want)
        add(f'fmamk2073 {hex(a)}',
            F.mov(1, HI_A | a) + F.mov(2, HI_B | c) + F.mov(3, HI_DST | 0x2222) + F.vop2(OP_FMAMK, 3, 257, 2) + [0xDEAD0000 | b], want)
        add(f'fmaak2073 {hex(a)}',
            F.mov(1, HI_A | a) + F.mov(2, HI_B | b) + F.mov(3, HI_DST | 0x3333) + F.vop2(OP_FMAAK, 3, 257, 2) + [0xCAFE0000 | c], want)

    # 2. v_fmac_f16 source kinds & VOP3 modifiers + destination overlap.
    add('fmac sgpr', F.smov(10, HI_A | 0x4000) + F.mov(2, HI_B | 0x4200) + F.mov(3, HI_DST | 0x3C00) + F.vop2(OP_FMAC, 3, 10, 2), ref_fma16(0x4000, 0x4200, 0x3C00))
    add('fmac inline_float', F.mov(2, HI_B | 0x4000) + F.mov(3, HI_DST | 0x3C00) + F.vop2(OP_FMAC, 3, 242, 2), ref_fma16(0x3C00, 0x4000, 0x3C00))
    add('fmac inv2pi', F.mov(2, HI_B | 0x4400) + F.mov(3, HI_DST | 0x3800) + F.vop2(OP_FMAC, 3, 248, 2), ref_fma16(0x3118, 0x4400, 0x3800))
    add('fmac inline_int', F.mov(2, HI_B | 0x4000) + F.mov(3, HI_DST | 0x3C00) + F.vop2(OP_FMAC, 3, 193, 2), ref_fma16(0xFFFF, 0x4000, 0x3C00))
    add('fmac e64 neg abs', F.mov(1, HI_A | 0xC000) + F.mov(2, HI_B | 0xC200) + F.mov(3, HI_DST | 0x3C00) + vop3_tern(OP_FMAC_VOP3, 3, 257, 258, 0, abs_mask=1, neg=2), ref_fma16(0xC000, 0xC200, 0x3C00, abs_mask=1, neg=2))
    add('fmac overlap dst=src0', F.mov(1, HI_DST | 0x4000) + F.mov(2, HI_B | 0x4200) + F.vop2(OP_FMAC, 1, 257, 2), ref_fma16(0x4000, 0x4200, 0x4000), dst=1)

    # 3. v_fma_f16 edge cases: abs/neg on all 3 sources, clamp, signed zero, cancellation, subnormals, overflow, Inf/NaN.
    edges = [
        ('neg0*-0 + -0', 0x8000, 0x0000, 0x8000, 0, 0, False),
        ('-0*-0 + -0', 0x8000, 0x8000, 0x8000, 0, 0, False),
        ('exact cancel 2*3-6', 0x4000, 0x4200, 0xC600, 0, 0, False),
        ('subnormal prod+subnormal', 0x0400, 0x0400, 0x0001, 0, 0, False),
        ('sticky diff>=30 midpoint', 0x4200, 0x5156, 0x0001, 0, 0, False),  # 128.0625 + 2^-24 -> 0x5801 (drop sticky -> 0x5800)
        ('subnormal tie to even', 0x0001, 0x3800, 0x0000, 0, 0, False),  # 2^-24 * 0.5 = 2^-25 -> 0
        ('subnormal above half', 0x0003, 0x3800, 0x0000, 0, 0, False),   # 3 * 2^-25 -> 0x0001
        ('overflow +inf', 0x7BFF, 0x4000, 0x3C00, 0, 0, False),
        ('overflow -inf', 0xFBFF, 0x4000, 0x3C00, 0, 0, False),
        ('clamp negative', 0xC000, 0x4000, 0x3C00, 0, 0, True),
        ('clamp >1.0', 0x4000, 0x4000, 0x3C00, 0, 0, True),
        ('clamp 0.5', 0x3800, 0x3C00, 0x0000, 0, 0, True),
        ('clamp +inf', 0x7C00, 0x3C00, 0x0000, 0, 0, True),
        ('clamp qNaN', 0x7E00, 0x3C00, 0x0000, 0, 0, True),
        ('0 * inf -> qNaN', 0x0000, 0x7C00, 0x3C00, 0, 0, False),
        ('+inf + -inf -> qNaN', 0x7C00, 0x3C00, 0xFC00, 0, 0, False),
        ('+inf + 1 -> +inf', 0x7C00, 0x3C00, 0x3C00, 0, 0, False),
        ('-inf + 1 -> -inf', 0xFC00, 0x3C00, 0x3C00, 0, 0, False),
        ('abs/neg all 3', 0xC000, 0xC200, 0x3C00, 5, 6, False),
    ]
    for label, a, b, c, absm, negm, cl in edges:
        add(label,
            F.mov(1, HI_A | a) + F.mov(2, HI_B | b) + F.mov(4, HI_C | c) + F.mov(3, HI_DST | 0x4444) +
            vop3_tern(OP_FMA_VOP3, 3, 257, 258, 260, abs_mask=absm, neg=negm, clamp=int(cl)),
            ref_fma16(a, b, c, abs_mask=absm, neg=negm, clamp=cl))

    assert len(items) < F.STRIDE
    return items


def emit_program(items):
    code = [sop1(3, 20, 4), sop1(3, 21, 128), 0xF408080A, 0xFA000020,
            scalar(0x1E, 7, 5, 128 + F.SHIFT), F.vmove(8, 7), sop1(3, 126, 129),
            0xBFA40000, 0xBFA5000F, 0xBA800881, 0]
    want = []
    for index, (_, snippet, expected, dst) in enumerate(items):
        code += snippet + F.store(4 * index, dst)
        want.append(HI_DST | expected)
    code.append(0xBF810000)
    return code, want


def same_word(got, want):
    if (got >> 16) != (HI_DST >> 16):
        return False
    return F.same_binary16(got & 0xFFFF, want & 0xFFFF)


def refusal_programs():
    fma = vop3_tern(OP_FMA_VOP3, 3, 257, 258, 260)
    fmac = F.vop2(OP_FMAC, 3, 257, 2)
    return [
        (F.vop2(OP_FMAC, 3, 250, 2) + [0xFF00B101, 0xBF810000], 'unsupported DPP arithmetic'),
        (F.vop2(OP_FMAC, 3, 249, 2) + [0x06060601, 0xBF810000], 'unimplemented direct SDWA operand profile'),
        (vop3_tern(OP_FMA_VOP3, 3, 257, 258, 260, omod=1) + [0xBF810000], 'unsupported v_fma_f16 omod'),
        (vop3_tern(OP_FMAC_VOP3, 3, 257, 258, 0, omod=1) + [0xBF810000], 'unsupported v_fmac_f16 omod'),
        (vop3_tern(OP_FMA_VOP3, 3, 257, 258, 260, op_sel=1) + [0xBF810000], 'unsupported v_fma_f16 high-half op_sel'),
        (vop3_tern(OP_FMAC_VOP3, 3, 257, 258, 0, op_sel=8) + [0xBF810000], 'unsupported v_fmac_f16 high-half op_sel'),
        (vop3_tern(OP_FMAC_VOP3, 3, 257, 258, 0, neg=4) + [0xBF810000], 'unsupported v_fmac_f16 third-source modifier'),
        (vop3_tern(OP_FMAC_VOP3, 3, 257, 258, 260) + [0xBF810000], 'unsupported v_fmac_f16 third-source modifier'),
        ([0xBFA4000C] + fma + [0xBF810000], 'unsupported binary16 rounding or denormal mode'),
        ([0xBFA50000] + fmac + [0xBF810000], 'unsupported binary16 rounding or denormal mode'),
        ([0xBA8005C1, 1] + fma + [0xBF810000], 'unsupported binary16 rounding or denormal mode'),
        ([0xB98005C1] + fmac + [0xBF810000], 'unsupported binary16 rounding or denormal mode'),
    ]


def ir(dumper):
    items = cases()
    code, want = emit_program(items)
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'fma-f16')
        assert run.returncode == 0, run.stderr
        stores = sorted(F.interpret(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, items[i][0], hex(got), hex(exp))
               for i, ((_, got), exp) in enumerate(zip(stores, want)) if not same_word(got, exp)]
        assert not bad, bad[:8]
        for idx, (refused, why) in enumerate(refusals):
            r = F.dump(dumper, work, refused, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (why, r.returncode, r.stderr)
    print(f'native_fma_f16 IR: {len(want)} single-rounded conversions match Fraction reference; {len(refusals)} refusals matched')
    return 0


def controls(cli, sdk, work, root):
    items = cases()
    code, want = emit_program(items)
    marker = request(work, root, code)
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    assert 'legacyF32ToF16' not in dump, dump

    refusals = refusal_programs()
    for n, (prog_words, why) in enumerate(refusals):
        m = request(work, root, prog_words)
        o = m.with_suffix(f'.refused{n}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_fma_f16 controls: compiled {len(want)} FMA cases cleanly to DXIL; verified {len(refusals)} refusals')
    return 0


def hardware(replay, sdk, work, root):
    items = cases()
    code, want = emit_program(items)
    marker = request(work, root, code)
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    words_out = list(struct.unpack(f'<{F.WORDS}I', output.read_bytes()))
    for group in range(F.GROUPS):
        head = F.BASE + group * F.STRIDE
        got = words_out[head:head + len(want)]
        bad = [(i, items[i][0], hex(a), hex(b)) for i, (a, b) in enumerate(zip(got, want)) if not same_word(a, b)]
        assert not bad, (group, bad[:8])
    words_out[F.BASE:F.BASE + len(want)] = [0] * len(want)
    words_out[F.BASE + F.STRIDE:F.BASE + F.STRIDE + len(want)] = [0] * len(want)
    assert all(w == F.SENTINEL for w in words_out if w != 0), 'guard overwritten'
    print(f'native_fma_f16 hardware: {len(want)} FMA cases matched across {F.GROUPS} groups')
    return 0


def main(mode, tool, sdk=None):
    if mode == 'ir':
        return ir(tool)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
