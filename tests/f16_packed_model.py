"""Independent CPU model of v_pk_mul_f16 / v_pk_add_f16 / v_pk_fma_f16 / v_pk_min_f16 / v_pk_max_f16 (Part of #1723). Pure Python, no GPU.

Per RDNA 2 VOP3P: result low half = op(source i half op_sel[i], ...), result high half = op(source i half
op_sel_hi[i], ...); absent modifiers mean op_sel [0,..] and op_sel_hi [1,..]. Each half is an IEEE binary16
operation rounded once to nearest even. Finite operands use exact rational arithmetic; infinities and NaNs use
float64 (exact for those cases). Any NaN result is reported as the canonical quiet NaN 0x7e00 (its class is what
is compared: the payload of a generated NaN is not fixed by the format).
"""
import math
import random
import struct
from fractions import Fraction

# 0, -0, smallest denormal, largest denormal, 1, -1, 65504, -65504, inf, -inf, quiet NaN, signalling NaN, NaN with payload
GRID = [0x0000, 0x8000, 0x0001, 0x03ff, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x7d00, 0x7e01]
# Extra finite words: 1/3, a tie between two binary16 values (1 + 2^-11 + 1 ulp pattern), 0.1, -2.5, 65519.9 region, min normal
EXTRA = [0x3555, 0x3c01, 0x2e66, 0xc100, 0x7bfe, 0x0400, 0x8400, 0x1400, 0x4900, 0x5640]


def h2f(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 0xffff))[0]


def is_nan(bits):
    return (bits & 0x7c00) == 0x7c00 and (bits & 0x3ff) != 0


def is_inf(bits):
    return (bits & 0x7fff) == 0x7c00


def sign(bits):
    return bits >> 15 & 1


def round_exact(x):
    """Binary16 encoding of a nonzero rational, nearest even, overflow to infinity (exact, no intermediate float)."""
    neg = 0x8000 if x < 0 else 0
    x = abs(x)
    e = max(x.numerator.bit_length() - x.denominator.bit_length() - 1, -14)
    while Fraction(2) ** (e + 1) <= x:
        e += 1
    while Fraction(2) ** e > x and e > -14:
        e -= 1
    q = x / Fraction(2) ** (e - 10)
    n = int(q)
    if q - n > Fraction(1, 2) or (q - n == Fraction(1, 2) and n & 1):
        n += 1
    if n == 0:
        return neg
    if n == 0x800:
        n, e = 0x400, e + 1
    if e > 15:
        return neg | 0x7c00
    return neg | (n if e == -14 and n < 0x400 else ((e + 15) << 10) | (n & 0x3ff))


def minmax(op, a, b):
    """v_pk_min_f16 / v_pk_max_f16 lane: IEEE minNum / maxNum on binary16 words. A quiet NaN operand yields the other operand, two NaNs
    a quiet NaN, and the zeros are ordered (-0 below +0). Signalling NaN operands are not modelled (the callers leave them out)."""
    if is_nan(a) and is_nan(b):
        return 0x7e00
    if is_nan(a):
        return b
    if is_nan(b):
        return a
    x, y = h2f(a), h2f(b)
    if x == y:  # equal values: only the two zeros differ by their encoding
        return (a | b) if op == 'min' else (a & b)
    return a if (x < y) == (op == 'min') else b


def half_op(op, a, b, c=0):
    """One binary16 lane: op is 'mul', 'add', 'fma' (a * b + c), 'min' or 'max'."""
    if op in ('min', 'max'):
        return minmax(op, a, b)
    words = (a, b, c) if op == 'fma' else (a, b)
    if any(is_nan(w) for w in words):
        return 0x7e00
    if any(is_inf(w) for w in words):
        x, y = h2f(a), h2f(b)
        with_nan = (op == 'mul' and (math.isinf(x) or math.isinf(y)) and (x == 0 or y == 0))
        v = x * y if op != 'add' else x + y
        if op == 'fma':
            if (x == 0 and math.isinf(y)) or (y == 0 and math.isinf(x)):
                with_nan = True
            v = x * y + h2f(c) if not with_nan else math.nan
        if with_nan or math.isnan(v):
            return 0x7e00
        return 0x7c00 if v > 0 else 0xfc00
    fa, fb, fc = Fraction(h2f(a)), Fraction(h2f(b)), Fraction(h2f(c))
    exact = fa * fb if op == 'mul' else fa + fb if op == 'add' else fa * fb + fc
    if exact != 0:
        return round_exact(exact)
    # An exact zero: the sign rules of IEEE 754 round-to-nearest.
    if op == 'mul':
        return (sign(a) ^ sign(b)) << 15
    if op == 'add':
        return 0x8000 if (a == 0x8000 and b == 0x8000) else 0
    product_zero = fa * fb == 0
    psign = sign(a) ^ sign(b)
    return 0x8000 if product_zero and fc == 0 and psign and sign(c) else 0


def pick(word, high):
    return (word >> 16 if high else word) & 0xffff


def packed(op, words, op_sel, op_sel_hi):
    """Result word of v_pk_<op>_f16 over 32-bit source words."""
    n = len(words)
    lo = half_op(op, *[pick(words[i], op_sel[i]) for i in range(n)])
    hi = half_op(op, *[pick(words[i], op_sel_hi[i]) for i in range(n)])
    return (hi << 16) | lo


def same(a, b):
    """Word equality where every NaN half of one equals every NaN half of the other."""
    return all(x == y or (is_nan(x) and is_nan(y)) for x, y in ((a & 0xffff, b & 0xffff), (a >> 16, b >> 16)))


def combos(arity):
    """(op_sel, op_sel_hi) selector pairs: all of them for two sources, a covering sample of 16 of the 64 for three."""
    def bits(v):
        return [v >> i & 1 for i in range(arity)]
    if arity == 2:
        return [(bits(s), bits(h)) for s in range(4) for h in range(4)]
    return [(bits(s), bits(h)) for s, h in
            [(s, (5 * s + 3) % 8) for s in range(8)] + [((3 * s + 1) % 8, s) for s in range(8)]]


def all_combos(arity):
    def bits(v):
        return [v >> i & 1 for i in range(arity)]
    return [(bits(s), bits(h)) for s in range(1 << arity) for h in range(1 << arity)]


def sample(seed, count, arity):
    """count tuples of arity 32-bit words whose halves come from the grid, the finite extras and random words."""
    rng = random.Random(seed)
    pool = GRID + EXTRA
    def half():
        return rng.choice(pool) if rng.random() < 0.8 else rng.randrange(0x10000)
    return [tuple(half() | half() << 16 for _ in range(arity)) for _ in range(count)]
