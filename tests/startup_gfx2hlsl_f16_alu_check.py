"""16-bit vector ALU lowering in tools/startup/gfx2hlsl.py (Part of #1723).

usage: startup_gfx2hlsl_f16_alu_check.py

v_add/sub/subrev/mul/min/max/fmac/fmamk/fmaak_f16 and v_cvt_f16_i16/u16, v_cvt_i16/u16_f16 read the low 16 bits of
their sources and write the low 16 bits of the destination (RDNA 2 zeroes the upper half). The check lowers each
instruction through Stage.one() and runs the emitted statement with a small Python model of the HLSL helpers, so the
result is compared as bits, not as text.
"""
import math
from fractions import Fraction
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402
sys.path.insert(0, here)
import f16_packed_model as M  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


def h2f(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 0xffff))[0]


def f2h(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def fma16o(a, b, c):
    """Model of the HLSL fma16o contract: finite binary16 operands give a*b+c rounded once, ties to even (the HLSL
    does it in 32-bit integer arithmetic, #2073); non-finite operands use binary32 arithmetic."""
    if not all(math.isfinite(v) for v in (a, b, c)):
        with_nan = a * b + c if not (math.isinf(a * b) and math.isinf(c) and (a * b > 0) != (c > 0)) else math.nan
        return with_nan
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        # IEEE round-to-nearest: -0 only when the product and the addend are both negative zeros, else +0.
        both_neg = math.copysign(1, a) * math.copysign(1, b) < 0 and math.copysign(1, c) < 0 and a * b == 0 and c == 0
        return -0.0 if both_neg else 0.0
    return h2f(f2h_exact(exact))


def f2h_exact(x):
    """Binary16 encoding of a rational, rounded to nearest even, overflow to infinity."""
    sign = 0x8000 if x < 0 else 0
    x = abs(x)
    if x == 0:
        return sign
    e = max(x.numerator.bit_length() - x.denominator.bit_length() - 1, -14)
    if Fraction(2) ** (e + 1) <= x:
        e += 1
    while Fraction(2) ** e > x and e > -14:
        e -= 1
    q = x / Fraction(2) ** (e - 10)
    n = int(q)
    if q - n > Fraction(1, 2) or (q - n == Fraction(1, 2) and n & 1):
        n += 1
    if n == 0x800:
        n, e = 0x400, e + 1
    if e > 15:
        return sign | 0x7c00
    return sign | (n if e == -14 and n < 0x400 else ((e + 15) << 10) | (n & 0x3ff))


def stage():
    s = G.Stage.__new__(G.Stage)
    s.lines, s.sym, s.used_s, s.used_v, s.masks = [], {}, set(), set(), set()
    s.exec_known = True
    return s


def minmax16_bits(a, b, maximum):
    """The HLSL helper of the same name: binary16 minimum or maximum on encodings (-0 below +0, a NaN loses)."""
    a_nan, b_nan = (a & 0x7fff) > 0x7c00, (b & 0x7fff) > 0x7c00
    if a_nan:
        return 0x7e00 if b_nan else b
    if b_nan:
        return a
    a_key = a ^ (0xffff if a & 0x8000 else 0x8000)
    b_key = b ^ (0xffff if b & 0x8000 else 0x8000)
    return a if (a_key > b_key if maximum else a_key < b_key) else b


def run(asm, regs):
    """Lower one instruction and evaluate its single statement over the register values in regs (v0, v1, ...)."""
    s = stage()
    name = asm.split()[0]
    s.one(name, asm)
    assert len(s.lines) == 1, s.lines
    dst, expr = re.fullmatch(r'(v\d+) = (.*);', s.lines[0]).groups()
    env = {f'v{i}': v for i, v in enumerate(regs)}
    env.update(
        mul16=lambda a, b: a * b,
        f16tof32=lambda x: h2f(x), fma16o=fma16o, f16rn=lambda x: 0x7e00 if math.isnan(x) else (f2h(x) if abs(x) < 65520.0 else (0xfc00 if x < 0 else 0x7c00)), asuint=lambda x: x, saturate=lambda x: min(max(x, 0.0), 1.0),
        minmax_zero32=lambda a, b, mx: (max if mx else min)(a, b),
        minmax16_bits=minmax16_bits,
        exp2=lambda x: 2.0 ** x, abs=abs, min=min, max=max, clamp=lambda x, lo, hi: min(max(x, lo), hi),
        trunc=lambda x: float(int(x)), isnan=math.isnan,
        asint=lambda x: x - (1 << 32) if x >> 31 else x, float=float, uint=lambda x: int(x) & 0xffffffff, int=int)
    py = expr.replace('0u', '0').replace('&&', 'and').replace('false', 'False').replace('true', 'True')
    py = re.sub(r'(0x[0-9a-f]+)u', r'\1', py)
    py = re.sub(r'\b(\d+)u\b', r'\1', py)
    py = re.sub(r'^isnan\((.*?)\) \? 0 : (.*)$', r'(0 if isnan(\1) else \2)', py)
    return dst, eval(py, {}, env)


ONE, TWO, HALF, THREE = f2h(1.0), f2h(2.0), f2h(0.5), f2h(3.0)
HIGH = 0xabcd0000
CASES = [
    ('v_add_f16 v3, v0, v1', [ONE | HIGH, TWO | HIGH], THREE),
    ('v_sub_f16 v3, v0, v1', [THREE, ONE], TWO),
    ('v_subrev_f16 v3, v0, v1', [ONE, THREE], TWO),
    ('v_mul_f16 v3, v0, v1', [THREE | HIGH, TWO], f2h(6.0)),
    ('v_min_f16 v3, v0, v1', [THREE, TWO], TWO),
    ('v_max_f16 v3, v0, v1', [THREE, TWO], THREE),
    ('v_fmac_f16 v2, v0, v1', [TWO, THREE, ONE], f2h(7.0)),
    ('v_fmamk_f16 v3, v0, 0x4000, v1', [THREE, ONE], f2h(7.0)),
    ('v_fmaak_f16 v3, v0, v1, 0x3c00', [THREE, TWO], f2h(7.0)),
    # Binary32 sum then binary16 would be one unit low on each (#2073); one rounding gives these words.
    ('v_fmac_f16 v2, v0, v1', [0x4a54, 0x1ef4, 0x4e2d], 0x4e33),
    ('v_fmac_f16 v2, v0, v1', [0x51ea, 0xbf3e, 0x1d5f], 0xd55b),
    ('v_fmac_f16 v2, v0, v1', [0x2bd5, 0xbe83, 0x4528], 0x450f),
    ('v_fmac_f16 v2, v0, v1', [0x3915, 0x387a, 0x4b71], 0x4b9f),
    ('v_add_f16 v3, 1.0, v1', [0, TWO], THREE),
    ('v_mul_f16 v3, 0.15915494, v1', [0, f2h(100.0)], f2h(h2f(f2h(0.15915494)) * 100.0)),
    ('v_mul_f16 v3, 0.15915494, v1', [0, 0x3002], 0x251b),
    ('v_cvt_f16_i16 v3, 1.0', [], f2h(15360.0)),
    ('v_cvt_f16_u16 v3, -1.0', [], f2h(48128.0)),
    ('v_add_f16 v3, -0x3c01, v1', [0, 0], 0xbc01),
    ('v_add_f16 v3, -v0, v1', [ONE, THREE], TWO),
    ('v_add_f16 v3, |v0|, v1', [f2h(-1.0), TWO], THREE),
    ('v_cvt_f16_i16 v3, v0', [0xfffd | HIGH], f2h(-3.0)),
    ('v_cvt_f16_u16 v3, v0', [0x8403 | HIGH], f2h(33795.0)),
    ('v_cvt_i16_f16 v3, v0', [f2h(-2.75) | HIGH], 0xfffe),
    ('v_cvt_i16_f16 v3, v0', [f2h(60000.0)], 0x7fff),
    ('v_cvt_u16_f16 v3, v0', [f2h(2.75) | HIGH], 2),
    ('v_cvt_u16_f16 v3, v0', [f2h(-1.5)], 0),
    # SDWA binary16: a selected half of each source, the result in one half of the destination. v3 holds the old
    # value 0x5a5a1234 where the other half is preserved.
    ('v_add_f16_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0',
     [ONE | HIGH, TWO | HIGH, 0, 0x5a5a1234], THREE),
    ('v_add_f16_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0',
     [0x3c00 << 16 | 0xabcd, TWO, 0, 0x5a5a1234], THREE),
    ('v_add_f16_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1',
     [ONE | HIGH, 0x4000 << 16 | 0x1234, 0, 0x5a5a1234], THREE),
    ('v_add_f16_sdwa v3, v0, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0',
     [ONE | HIGH, TWO | HIGH, 0, 0x5a5a1234], THREE << 16),
    ('v_add_f16_sdwa v3, v0, v1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0',
     [ONE | HIGH, TWO | HIGH, 0, 0x5a5a1234], THREE << 16 | 0x1234),
    ('v_add_f16_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0',
     [ONE | HIGH, TWO | HIGH, 0, 0x5a5a1234], 0x5a5a0000 | THREE),
    ('v_sub_f16_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0',
     [THREE << 16, ONE, 0, 0], TWO),
    ('v_subrev_f16_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_1',
     [ONE, THREE << 16, 0, 0], TWO),
    ('v_mul_f16_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1',
     [THREE << 16, TWO << 16, 0, 0], f2h(6.0)),
    ('v_add_f16_sdwa v3, -v0, |v1| dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0',
     [ONE, f2h(-3.0), 0, 0], TWO),
    # inf - inf is NaN: the lane keeps a NaN class in the selected half.
    ('v_sub_f16_sdwa v3, v0, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0',
     [0x7c00, 0x7c00, 0, 0], 0x7e00 << 16),
]
for asm, regs, expected in CASES:
    dst, got = run(asm, regs + [0] * (4 - len(regs)))
    check(asm + ' ' + str(regs), got == expected, 'got %#x want %#x' % (got, expected))
# The statement the lowering emits, evaluated over every lane of the SDWA draw fixtures (edge values, NaN, infinity,
# signed zero, denormals, both halves, PAD and PRESERVE) against the exact-rational reference of the draw check. A NaN
# has no portable sign or payload, so both sides compare the NaN class (the draw canonicalises the same way).
sys.path.insert(0, here)
import startup_gfx2hlsl_f16_alu_draw_check as D  # noqa: E402
for cfg, lanes in D.SDWA_DRAW.items():
    op, s0, s1, dsel, unused, m0, m1 = cfg[:7]
    asm = '%s_sdwa v3, %s, %s dst_sel:%s dst_unused:%s src0_sel:%s src1_sel:%s' % (
        op, D.sdwa_operand(m0, 0), D.sdwa_operand(m1, 1), dsel, unused, s0, s1)
    bad = [(x, D.sdwa_reference(cfg, x), run(asm, [x[0], x[1], 0, x[2]])[1]) for x in lanes]
    bad = [(x, want, got) for x, want, got in bad if D.canon_word(got) != want]
    check('sdwa %s over %d lanes' % (D.sdwa_name(cfg), len(lanes)), not bad,
          ' '.join('%08x,%08x,%08x want %08x got %08x' % (*x, want, got) for x, want, got in bad[:2]))


def low16_stage():
    s = stage()
    s.low16_only_v, s.pinned_low16_v = {3}, set()  # v3 holds a d16 load: its upper half is undefined
    return s


SDWA = 'v_add_f16_sdwa v3, v0, v1 dst_sel:%s dst_unused:%s src0_sel:%s src1_sel:WORD_0'
for sel, unused, src0, still_low16, label in (
        ('WORD_0', 'UNUSED_PRESERVE', 'WORD_0', True, 'keeping the undefined upper half leaves it undefined'),
        ('WORD_1', 'UNUSED_PRESERVE', 'WORD_0', False, 'a WORD_1 write over a defined low half is fully defined'),
        ('WORD_0', 'UNUSED_PAD', 'WORD_0', False, 'UNUSED_PAD zeroes the upper half')):
    s = low16_stage()
    s.one('v_add_f16_sdwa', SDWA % (sel, unused, src0))
    check('SDWA binary16 ' + label, (3 in s.low16_only_v) == still_low16)
try:
    low16_stage().one('v_add_f16_sdwa', 'v_add_f16_sdwa v4, v3, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD '
                      'src0_sel:WORD_1 src1_sel:WORD_0')
    check('SDWA binary16 refuses the undefined upper half of a d16 load as a source', False)
except G.Unsupported:
    check('SDWA binary16 refuses the undefined upper half of a d16 load as a source', True)
check('clamp saturates the binary16 result', run('v_add_f16 v3, v0, v1 clamp', [THREE, TWO, 0, 0])[1] == ONE)

for op, low_result, high_result in (('v_exp_f16', TWO, f2h(8.0)),
                                    ('v_rcp_f16', ONE, f2h(1.0 / 3.0))):
    regs = [(THREE << 16) | ONE, 0, 0, 0xabcd1234]
    check(op + ' plain low half', run(op + '_e32 v3, v0', regs)[1] == low_result)
    for src, result in (('DWORD', low_result), ('WORD_0', low_result), ('WORD_1', high_result)):
        for dst, shift, kept in (('WORD_0', 0, 0xabcd0000), ('WORD_1', 16, 0x1234)):
            for unused, preserved in (('UNUSED_PAD', 0), ('UNUSED_PRESERVE', kept)):
                asm = f'{op}_sdwa v3, v0 dst_sel:{dst} dst_unused:{unused} src0_sel:{src}'
                check(asm, run(asm, regs)[1] == ((result << shift) | preserved))
    for suffix in ('src0_sel:BYTE_1', 'src0_sel:WORD_0 src1_sel:WORD_0',
                   'src0_sel:WORD_0 clamp'):
        asm = f'{op}_sdwa v3, v0 dst_sel:WORD_0 dst_unused:UNUSED_PAD {suffix}'
        try:
            stage().one(op + '_sdwa', asm)
            check('refuses ' + asm, False)
        except G.Unsupported as reason:
            check('refuses ' + asm, op in str(reason))


for asm in ('v_add_f16 v3, v0, v1 op_sel:[1,0]', 'v_cvt_i16_f16 v3, v0 clamp', 'v_add_f16 v3, v0',
            'v_fmaak_f16 v3, v0, v1', 'v_mul_f16 v3, v0, v1 mul:2',
            # SDWA binary16: whole-word destination, byte fields, clamp, omod, integer sext and a constant
            # source stay refused.
            'v_add_f16_sdwa v3, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0',
            'v_add_f16_sdwa v3, v0, v1 dst_sel:BYTE_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0',
            'v_add_f16_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:WORD_0',
            'v_add_f16_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:WORD_0 src1_sel:WORD_0',
            'v_add_f16_sdwa v3, v0, v1 clamp dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0',
            'v_add_f16_sdwa v3, v0, v1 mul:2 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0',
            'v_add_f16_sdwa v3, sext(v0), v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0',
            'v_add_f16_sdwa v3, 1.0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0',
            'v_add_f16_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0',
            'v_min_f16_sdwa v3, v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0'):
    try:
        stage().one(asm.split()[0], asm)
        check('refuses ' + asm, False)
    except G.Unsupported:
        check('refuses ' + asm, True)

# v_pk_{mul,add,fma}_f16: the lowering is evaluated over every selector pair and a value set built from NaN, inf, -0,
# denormals, the binary16 limits and random words, and compared with the independent model in f16_packed_model.py.
SEL = {2: ('v_pk_mul_f16', 'mul'), 3: ('v_pk_fma_f16', 'fma')}
for name, op, arity in (('v_pk_mul_f16', 'mul', 2), ('v_pk_add_f16', 'add', 2), ('v_pk_fma_f16', 'fma', 3)):
    samples = M.sample(arity, 150, arity)
    bad = []
    seen = {'swap': 0, 'zero_hi': 0, 'default_hi': 0}
    for sel, hi in M.all_combos(arity):
        mods = ' op_sel:[%s] op_sel_hi:[%s]' % (','.join(map(str, sel)), ','.join(map(str, hi)))
        asm = '%s v3, %s%s' % (name, ', '.join('v%d' % i for i in range(arity)), mods)
        for words in samples:
            _, got = run(asm, list(words) + [0] * (4 - arity))
            want = M.packed(op, words, sel, hi)
            if not M.same(got, want):
                bad.append((asm, [hex(w) for w in words], hex(got), hex(want)))
            # The model must tell a wrong lowering from the right one: swapped lanes, zeroed high lane, default selectors.
            swapped = (want & 0xffff) << 16 | want >> 16
            seen['swap'] += not M.same(got, swapped)
            seen['zero_hi'] += not M.same(got, want & 0xffff)
            seen['default_hi'] += not M.same(got, M.packed(op, words, sel, [0] * arity))
    check('%s matches the independent model on every selector pair (%d pairs x %d value sets)'
          % (name, len(M.all_combos(arity)), len(samples)), not bad, str(bad[:2]))
    check('%s: the value sets tell a swapped-lane, zeroed-high-lane or wrong-op_sel_hi lowering from the right one' % name,
          all(seen.values()), str(seen))
    # Default selectors (no modifier printed) are op_sel [0..] and op_sel_hi [1..].
    plain = '%s v3, %s' % (name, ', '.join('v%d' % i for i in range(arity)))
    check('%s with no modifier uses the default selectors' % name,
          all(M.same(run(plain, list(w) + [0] * (4 - arity))[1], M.packed(op, w, [0] * arity, [1] * arity))
              for w in samples))
# Hand-checked words: 2*3 + 1 per lane, a high lane that reads the low half, inf*0 is NaN, overflow, -0.
PK = [
    ('v_pk_mul_f16 v3, v0, v1', [0x4000 | THREE << 16, 0x4200 | TWO << 16], 0x46004600),
    ('v_pk_add_f16 v3, v0, v1 op_sel_hi:[0,1]', [ONE | THREE << 16, ONE | TWO << 16], f2h(2.0) | f2h(3.0) << 16),
    ('v_pk_fma_f16 v3, v0, v1, v2', [TWO | TWO << 16, THREE | HALF << 16, ONE | ONE << 16], f2h(7.0) | f2h(2.0) << 16),
    ('v_pk_mul_f16 v3, v0, v1', [0x7bff | 0x8000 << 16, 0x4000 | 0x3c00 << 16], 0x7c00 | 0x8000 << 16),
    ('v_pk_mul_f16 v3, v0, v1', [0x8000 | 0x0000 << 16, 0x3c00 | 0xbc00 << 16], 0x80008000),
]
for asm, regs, expected in PK:
    got = run(asm, regs + [0] * (4 - len(regs)))[1]
    check(asm + ' ' + str([hex(r) for r in regs]), got == expected, 'got %#x want %#x' % (got, expected))
check('inf * 0 is NaN in both lanes', M.is_nan(run('v_pk_mul_f16 v3, v0, v1', [0x7c00 | 0xfc00 << 16, 0 | 0x8000 << 16, 0, 0])[1] & 0xffff)
      and M.is_nan(run('v_pk_mul_f16 v3, v0, v1', [0x7c00 | 0xfc00 << 16, 0 | 0x8000 << 16, 0, 0])[1] >> 16))
# A literal source read through its low half in both lanes (what the title's clamp uses: 0x7bef is just under the binary16 maximum).
PKLIT = [
    ('v_pk_max_f16 v3, 0xc900, v0 op_sel_hi:[0,1]', [0xbc00 | 0x4000 << 16], 0x4000bc00),
    ('v_pk_add_f16 v3, 1.0, v0 op_sel_hi:[0,1]', [ONE | TWO << 16], f2h(2.0) | f2h(3.0) << 16),
    ('v_pk_mul_f16 v3, -2.0, v0 op_sel_hi:[0,1]', [ONE | THREE << 16], f2h(-2.0) | f2h(-6.0) << 16),
    ('v_pk_min_f16 v3, v0, 0x7bef op_sel_hi:[1,0]', [0x7c00 | 0x4000 << 16], 0x40007bef),
    ('v_pk_min_f16 v3, v0, 0x7bef op_sel_hi:[1,0]', [0x3c00 | 0xfc00 << 16], 0xfc003c00),
    ('v_pk_max_f16 v3, v0, 0x7bef op_sel_hi:[1,0]', [0x3c00 | 0xfc00 << 16], 0x7bef7bef),
    ('v_pk_max_f16 v3, v0, 0x7bef op_sel_hi:[1,0]', [0x7c00 | 0x4000 << 16], 0x7bef7c00),
]
for asm, regs, expected in PKLIT:
    got = run(asm, regs + [0] * (4 - len(regs)))[1]
    check(asm + ' ' + str([hex(r) for r in regs]), got == expected, 'got %#x want %#x' % (got, expected))
for asm in ('v_pk_add_f16 v3, v0, v1 clamp', 'v_pk_add_f16 v3, v0, v1 neg_lo:[1]',
            'v_pk_mul_f16 v3, 2.0, v1', 'v_pk_mul_f16 v3, 2.0, v1 op_sel:[1,0] op_sel_hi:[0,1]', 'v_pk_add_f16 v3, 3, v1 op_sel_hi:[0,1]', 'v_pk_mul_f16 v3, v0, 0x4000', 'v_pk_min_f16 v3, v0, 1.0', 'v_pk_add_f16 v3, s0, v1',
            'v_pk_mul_f16 v3, v0, v1 op_sel:[1]', 'v_pk_fma_f16 v3, v0, v1 op_sel:[0,0]', 'v_pk_mul_f16 v3, v0, v1 op_sel:[2,0]',
            'v_pk_mul_f16 v3, -v0, v1', 'v_pk_fma_f16 v3, v0, v1, v2 mul:2', 'v_pk_mul_f16 v[3:4], v0, v1'):
    try:
        stage().one(asm.split()[0], asm)
        check('refuses ' + asm, False)
    except G.Unsupported:
        check('refuses ' + asm, True)

PACKED_INPUTS = (
    (0x40003c00, 0xc4004200),  # unequal magnitudes and signs
    (0x80000000, 0x00008000),  # both signs of zero
    (0x00017bff, 0x0400fbff),  # largest finite, smallest normal and subnormal
    (0x7e157c00, 0x3c000000),  # quiet NaN and infinity
    (0xfc008001, 0x7c007e75),  # negative infinity and NaN in the other source
)


def packed_negation_fixture(wrong_half=False):
    from startup_compute_lds_dual_atomics_check import (
        PREAMBLE_LINES, SENTINEL, USER7, WORDS, mubuf_store, root_blob, sopp, v_mov_imm, vop2, vopc,
    )

    code = bytearray(b''.join(raw for _, raw in PREAMBLE_LINES))
    code += vop2(0x1a, 4, 0x82, 0)  # per-thread output byte address
    for reg, value in enumerate(PACKED_INPUTS[0], 1):
        code += v_mov_imm(reg, value)
    for thread, pair in enumerate(PACKED_INPUTS[1:], 1):
        code += vopc(0xc2, 0x80 + thread, 0)
        for reg, value in enumerate(pair, 1):
            code += v_mov_imm(5, value)
            code += vop2(0x01, reg, 256 + reg, 5)  # select the table row for this thread

    def result(op, pair, half, mask):
        x = M.pick(pair[0], half) ^ (0x8000 if mask & 1 else 0)
        y = M.pick(pair[1], half) ^ (0x8000 if mask & 2 else 0)
        return D.canon_half(M.half_op(op, x, y))

    expected = []
    for op, opcode in (('add', 0x0f), ('sub', 0x0f), ('mul', 0x10), ('min', 0x11), ('max', 0x12)):
        # The ISA expresses packed subtraction as addition with SRC1 negated in both halves.
        sub = 2 if op == 'sub' else 0
        for source in range(2):
            for half in range(2):
                lo = sub ^ ((1 << source) if half == 0 else 0)
                hi = sub ^ ((1 << source) if half == 1 else 0)
                if wrong_half:
                    lo, hi = hi, lo
                code += struct.pack('<2I', 0xcc000003 | (opcode << 16) | (hi << 8),
                                    257 | (258 << 9) | (3 << 27) | (lo << 29))
                code += mubuf_store(0x1c, 3, 4, offset=4 * len(expected))
                alu = 'add' if op == 'sub' else op
                expected.extend(result(alu, pair, 0, lo) | (result(alu, pair, 1, hi) << 16)
                                for pair in PACKED_INPUTS)
    code += sopp(0x01)
    hlsl = G.ComputePipeline(bytes(code), root_blob(), USER7, (len(PACKED_INPUTS), 1, 1)).hlsl_cs()
    # GPU arithmetic can return any quiet NaN encoding. Compare all other bits raw.
    hlsl, stores = re.subn(r'\bStore\(a, v3\);', 'Store(a, nanc(v3));', hlsl)
    assert stores * len(PACKED_INPUTS) == len(expected), (stores, len(expected))
    return D.NAN_HLSL + hlsl, expected + [SENTINEL] * (WORDS - len(expected))


def packed_negation_gpu(dxc, runner, adapter, fixtures):
    from startup_compute_lds_dual_atomics_check import check_debug_layer, renderer_diagnostics, run_shader

    good, mutant = fixtures
    expected = good[1]
    mismatch = next(i for i, (a, b) in enumerate(zip(expected, mutant[1])) if a != b)
    wanted = 'word=%d expected=%08x actual=%08x' % (mismatch, expected[mismatch], mutant[1][mismatch])
    out = Path(here).parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for name, (source, _) in zip(('packed', 'wrong-half'), fixtures):
            run = run_shader(name, dxc, runner, adapter, Path(tmp), source, expected)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            check_debug_layer(name, run.stdout, run.stderr)
            diagnostics = renderer_diagnostics(run.stderr, adapter)
            if name == 'wrong-half':
                assert run.returncode == 1 and diagnostics and diagnostics[0] == wanted, run.stderr
                print('wrong-half control rejected at ' + diagnostics[0])
            else:
                assert not run.returncode and not diagnostics, (run.returncode, run.stderr)
    return 0


packed_fixtures = (packed_negation_fixture(), packed_negation_fixture(wrong_half=True))
check('packed negation wrong-half reference differs', packed_fixtures[0][1] != packed_fixtures[1][1])

if failures:
    print('FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('f16 alu ok')

if sys.argv[1:2] == ['dispatch']:
    sys.exit(packed_negation_gpu(*sys.argv[2:], packed_fixtures))
