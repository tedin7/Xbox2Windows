"""Generate the non-tie binary16 exp2 corrections for an 8 f32 ULP window."""
import re
import struct
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cache
from pathlib import Path


def floor_log2(value):
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    return exponent - (value < Fraction(2) ** exponent)


@cache
def exp_f16_midpoints():
    entries = {}
    with localcontext() as context:
        context.prec = 90
        ln2 = Decimal(2).ln()
        for bits in range(0x10000):
            if bits & 0x7fff >= 0x7c00:
                continue
            operand = struct.unpack('<e', struct.pack('<H', bits))[0]
            # Beyond these limits the half result is fixed with ample error margin.
            if operand < -25 or operand > 16:
                continue
            power = Fraction(operand)
            integer = power.numerator // power.denominator
            fractional = power - integer
            exponent = Decimal(fractional.numerator) / Decimal(fractional.denominator)
            value = Fraction(2) ** integer * Fraction((exponent * ln2).exp())
            binade = floor_log2(value)
            half_ulp = Fraction(2) ** (max(binade, -14) - 10)
            lower = int(value / half_ulp) * half_ulp
            midpoint = lower + half_ulp / 2
            distance = abs(value - midpoint)
            if distance == 0:
                assert bits == 0xce40  # The emitter handles exp2(-25) explicitly.
            elif distance < 8 * Fraction(2) ** (binade - 23):
                rounded = lower if value < midpoint else lower + half_ulp
                result = struct.unpack('<H', struct.pack('<e', float(rounded)))[0]
                entries[bits] = result
    return entries


def check_cpp_table():
    source = Path(__file__).resolve().parents[1] / 'src' / 'gfx1013_direct_compute.inl'
    array = re.search(r'exp_f16_midpoints\[\]\s*\{(.*?)\};',
                      source.read_text(encoding='utf-8'), re.DOTALL)
    assert array, 'missing C++ exp_f16_midpoints array'
    pairs = re.findall(r'\{\s*(0x[0-9a-f]+),\s*(0x[0-9a-f]+)\s*\}', array[1])
    parsed = [(int(source, 16), int(result, 16)) for source, result in pairs]
    assert len(parsed) == 70 and parsed == list(exp_f16_midpoints().items()), parsed


if __name__ == '__main__':
    check_cpp_table()
