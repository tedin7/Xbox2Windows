"""v_ldexp_f32 of tools/startup/gfx2hlsl.py (VOP3: D = S0.f * 2^S1.i, S1 a signed 32-bit exponent of any size).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel body and compare it with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

The reference scales the exact binary64 value with math.ldexp and rounds once to binary32, so it is the product the
ISA describes, whatever the exponent: signed zero, infinities, overflow to infinity, underflow to zero, and an
exponent beyond one scaling step (2^120 * 2^-240), denormal results rounded once, and a denormal source scaled back
to exactly 1.0 (a plain exp2(n) multiply turns it into infinity). The lowering works on the binary32 fields, so
compilers that flush float denormals (WARP under FXC) cannot change a result. The mutation takes the exponent clamped
to +-127 instead.
The fixture and runner are startup_gfx2hlsl_lane_alu.py.
"""
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_gfx2hlsl_lane_alu import main, signed, words  # noqa: E402


def f(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def v(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def scaled(bits, exponent, limit=None):
    """x * 2^exponent rounded once to binary32 (round to nearest even), as the ISA's product."""
    n = signed(exponent)
    if limit is not None:
        n = max(-limit, min(limit, n))
    x = v(bits)
    if math.isinf(x) or x == 0.0:
        return f(x)
    try:
        wide = math.ldexp(x, max(-5000, min(5000, n)))
    except OverflowError:
        return f(math.copysign(math.inf, x))
    try:
        return f(wide)
    except OverflowError:
        return f(math.copysign(math.inf, wide))


OPS = {'v_ldexp_f32': (2, lambda a, n: scaled(a, n), False)}
EXPONENT = lambda n: n & 0xffffffff  # noqa: E731
DRAW = {'v_ldexp_f32': [
    # Chunk 0 separates the mutation: an exponent past +-127 and a sum of two steps.
    (f(1.0), 3, 0), (f(1.0), 200, 0), (f(-1.5), EXPONENT(-2), 0), (f(2.0 ** -100), 130, 0),
    # Overflow, the largest exponent, a negative one and signed zeros.
    (f(1.0), 128, 0), (f(-1.0), 0x7fffffff, 0), (f(1.0), 0x80000000, 0), (f(-1.0), 0x80000000, 0),
    (f(0.0), 100, 0), (f(-0.0), EXPONENT(-100), 0), (0x7f800000, EXPONENT(-5), 0), (0xff800000, 7, 0),
    # Exponents past one scaling step with a normal result: both steps of the split, up and down, and the last bit.
    (f(2.0 ** 120), EXPONENT(-240), 0), (f(2.0 ** -120), 240, 0), (f(1.5 * 2.0 ** 100), EXPONENT(-200), 0),
    (f(3.0 * 2.0 ** 60), EXPONENT(-130), 0),
    (f(1.0), EXPONENT(-126), 0), (f(1.99999988), EXPONENT(-1), 0), (f(1.0), 127, 0), (f(-3.0), 0, 0),
    # Denormal results rounded once (ties to even), and a denormal source scaled back to exactly 1.0.
    (f(1.0), EXPONENT(-127), 0), (f(1.0), EXPONENT(-149), 0), (f(1.0), EXPONENT(-150), 0), (f(1.5), EXPONENT(-149), 0),
    (0x00000400, 140, 0), (f(3.0), EXPONENT(-130), 0), (f(1.00000012), EXPONENT(-149), 0), (0x00800000 | 1, EXPONENT(-1), 0),
]}
MUTATIONS = {'v_ldexp_f32': (r'x2w_ldexp\(asuint\((F\(v\d+\))\), asint\((v\d+)\)\)',
                             r'x2w_ldexp(asuint(\1), clamp(asint(\2), -127, 127))',
                             lambda a, n: scaled(a, n, 127))}
BODY = {'v_ldexp_f32': words('0c0062d7000902000d0062d7010b02000e0062d7020d02000f0062d7030f0200')}
FAMILY = {
    'ops': OPS, 'draw': DRAW, 'mutations': MUTATIONS, 'body': BODY,
    'accepted': ['v_ldexp_f32 v1, v2, v3', 'v_ldexp_f32_e64 v1, -v2, v3', 'v_ldexp_f32_e64 v1, |v2|, v3',
                 'v_ldexp_f32 v1, v2, -3', 'v_ldexp_f32_e64 v1, v2, v3 mul:2', 'v_ldexp_f32_e64 v1, v2, v3 clamp'],
    'refused': ['v_ldexp_f32_e64 v1, v2, -v3', 'v_ldexp_f32_e64 v1, v2, |v3|', 'v_ldexp_f32 v1, v2',
                'v_ldexp_f32_e64 v1, v2, v3 op_sel:[1,0]', 'v_ldexp_f32_e64 v1, v2, v3 neg_lo:[1]'],
}

if __name__ == '__main__':
    main(FAMILY, __doc__)
