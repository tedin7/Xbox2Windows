"""Conservative per-lane ranges for vertex program-image load proofs.

This module only models individual instructions; a caller must merge or discard
facts at control-flow joins and invalidate every destination it cannot model.
"""
import math
import struct
from fractions import Fraction


class VertexImageRanges:
    """Unsigned integer and non-negative binary32 intervals for active lanes."""

    LIMIT = 0xffffffff

    def __init__(self):
        self.facts = {}

    def forget(self, *registers):
        for reg in registers:
            self.facts.pop(reg, None)

    def exec_changed(self):
        # Previously inactive lanes can become active and have different values.
        self.facts.clear()

    def unorm(self, *registers):
        # AMD RDNA 2 ISA guide, section 8.1, buffer numeric formats, pp. 86-87.
        for reg in registers:
            self.facts[reg] = ('float', Fraction(0), Fraction(1))

    @staticmethod
    def _upper_float32(value):
        """Round a non-negative rational outward to a finite binary32 value."""
        try:
            packed = struct.pack('<f', float(value))
        except OverflowError:
            return None
        bits, = struct.unpack('<I', packed)
        result, = struct.unpack('<f', packed)
        if not math.isfinite(result):
            return None
        if Fraction.from_float(result) < value:
            bits += 1
            result, = struct.unpack('<f', struct.pack('<I', bits))
            if not math.isfinite(result):
                return None
        return Fraction.from_float(result)

    def fma(self, dst, src, multiplier, addend):
        fact = self.facts.get(src)
        self.forget(dst)
        if not fact or fact[0] != 'float':
            return
        try:
            mul = Fraction.from_float(multiplier)
            add = Fraction.from_float(addend)
        except (ValueError, OverflowError, TypeError):
            return
        if mul < 0 or add < 0:
            return
        # Zero is a safe lower bound after rounding. The upper bound is rounded
        # outward, including when the binary32 operation rounds upward.
        upper = self._upper_float32(fact[2] * mul + add)
        if upper is not None:
            self.facts[dst] = ('float', Fraction(0), upper)

    def cvt_u32(self, dst, src):
        fact = self.facts.get(src)
        self.forget(dst)
        if fact and fact[0] == 'float':
            self.facts[dst] = ('uint', min(int(fact[1]), self.LIMIT),
                               min(int(fact[2]), self.LIMIT))

    def integer(self, dst, lo, hi):
        self.forget(dst)
        if 0 <= lo <= hi <= self.LIMIT:
            self.facts[dst] = ('uint', lo, hi)

    def add_scaled(self, dst, left, scale, right):
        """Track scale * left + right, with no unsigned wrap."""
        a, b = self.facts.get(left), self.facts.get(right)
        self.forget(dst)
        if a and b and a[0] == b[0] == 'uint' and 0 <= scale <= self.LIMIT:
            self.integer(dst, scale * a[1] + b[1], scale * a[2] + b[2])

    def cmpx_ne(self, reg, value):
        # RDNA 2 ISA guide, VOPC comparisons, p. 260: CMPX narrows EXEC.
        fact = self.facts.get(reg)
        if fact and fact[0] == 'uint' and fact[2] == value and fact[1] < value:
            self.facts[reg] = ('uint', fact[1], value - 1)
        elif fact and fact[0] == 'uint' and fact[1] == fact[2] == value:
            self.forget(reg)  # No active lanes; do not invent a nonempty range.

    def in_table(self, reg, words):
        fact = self.facts.get(reg)
        return bool(fact and fact[0] == 'uint' and 0 <= fact[1] <= fact[2] < words)
