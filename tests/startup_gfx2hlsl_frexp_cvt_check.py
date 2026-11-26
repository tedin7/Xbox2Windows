"""v_frexp_mant_f32, v_frexp_exp_i32_f32, v_cvt_flr_i32_f32, v_cvt_rpi_i32_f32 and v_ldexp_f32 in
tools/startup/gfx2hlsl.py (FRONTIER_NEXT_BATCH_3 item 7).

usage: startup_gfx2hlsl_frexp_cvt_check.py   (pure Python, no compiler)

Each operation is lowered by the translator, then the emitted HLSL line is run on the CPU by a small evaluator
(binary32 rounding after every float multiply). The references follow the AMD RDNA 2 ISA reference:
  frexp_mant: sign and mantissa in [0.5, 1) of a normal input; zero and denormal give a signed zero; inf and NaN
              pass through.
  frexp_exp:  the matching exponent; 0 for zero, denormal, inf and NaN.
  cvt_flr / cvt_rpi: floor(x) / floor(x + 0.5) in exact arithmetic, saturated to int32; a NaN reads 0x7fffffff
              (sign clear) or 0x80000000 (sign set), as measured on RDNA 2 (CVT_FLR_RPI_EXCEPTIONAL).
  ldexp:      S0 * 2^S1 with S1 a signed integer.
Not covered: denormal inputs and results of v_ldexp_f32 (the denormal mode decides them) and signaling-NaN payloads.

Profile this oracle states: the NORMAL domain only (normal inputs, normal results, zero, inf, NaN). The binary32 denormal
selection (MODE[5:4]) is not modelled by the startup translator: a write that selects it is accepted and changes no emitted
line, so the emitted code follows the desktop compiler's own denormal behaviour (flushing, as measured on WARP and one
desktop GPU). ref_cvt_flr and ref_ldexp compute denormal words exactly, ref_frexp_mant answers a signed zero; none of
these denormal rows is a guest guarantee, and a desktop result on them says nothing about the console. A guest profile
for denormals needs a source-built native reference that records the selected mode, the input and the output words.
"""
import math
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

MASK = 0xffffffff
TOKEN = re.compile(r'0x[0-9a-fA-F]+u?|\d+\.\d+|\d+u?|[A-Za-z_]\w*|\|\||==|!=|<=|>=|>>|<<|[-+*/&|?:(),<>!~]')
BINARY = [['||'], ['|'], ['&'], ['==', '!='], ['<', '>', '<=', '>='], ['<<', '>>'], ['+', '-'], ['*', '/']]


def f32(x):
    try:
        return struct.unpack('<f', struct.pack('<f', x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


class F32(float):
    """binary32 value that keeps its source word: struct.unpack('<f') sets the quiet bit of a signaling NaN on
    some Python builds (3.11, 3.12), so F() -> U() would change the word."""
    word = None


def bits(x):
    if isinstance(x, F32) and x.word is not None:
        return x.word
    return struct.unpack('<I', struct.pack('<f', x))[0]


def value(word):
    r = F32(struct.unpack('<f', struct.pack('<I', word & MASK))[0])
    r.word = word & MASK
    return r


def to_int(x):
    if isinstance(x, float):
        return int(x)  # trunc toward zero; the lowering clamps and filters NaN first
    return x - (1 << 32) if x >= 1 << 31 else x


FUNCS = {
    'U': bits, 'F': value, 'uint': lambda x: int(x) & MASK, 'int': to_int, 'float': float,
    'floor': lambda x: x if math.isinf(x) else float(math.floor(x)), 'max': max, 'exp2': lambda x: f32(2.0 ** x),
    'clamp': lambda x, lo, hi: min(max(x, lo), hi), 'saturate': lambda x: min(max(x, 0.0), 1.0),
}


def idiv(a, b):
    return int(a / b)


def add(a, b):
    return f32(a + b) if isinstance(a, float) or isinstance(b, float) else (a + b) & MASK


def sub(a, b):
    return f32(a - b) if isinstance(a, float) or isinstance(b, float) else a - b


def mul(a, b):
    return f32(a * b) if isinstance(a, float) or isinstance(b, float) else (a * b) & MASK


class Parser:
    """C-precedence parser for the emitted subset; produces a Python expression string."""

    def __init__(self, text):
        self.t = TOKEN.findall(text)
        assert ''.join(self.t) == re.sub(r'\s+', '', text), text
        self.i = 0
        self.out = self.ternary()
        assert self.i == len(self.t), text

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        self.i += 1
        return self.t[self.i - 1]

    def ternary(self):
        cond = self.binary(0)
        if self.peek() != '?':
            return cond
        self.take()
        yes = self.ternary()
        assert self.take() == ':'
        return '(%s if %s else %s)' % (yes, cond, self.ternary())

    def binary(self, level):
        if level == len(BINARY):
            return self.unary()
        node = self.binary(level + 1)
        while self.peek() in BINARY[level]:
            op = self.take()
            right = self.binary(level + 1)
            if op == '||':
                node = '(%s or %s)' % (node, right)
            elif op == '/':
                node = 'idiv(%s, %s)' % (node, right)
            elif op == '*':
                node = 'mul(%s, %s)' % (node, right)
            elif op == '+':
                node = 'add(%s, %s)' % (node, right)
            elif op == '-':
                node = 'sub(%s, %s)' % (node, right)
            else:
                node = '(%s %s %s)' % (node, op, right)
        return node

    def unary(self):
        token = self.take()
        if token == '-':
            return '(-%s)' % self.unary()
        if token == '(':
            node = self.ternary()
            assert self.take() == ')'
            return node
        if token[0].isdigit():
            return repr(float(token)) if '.' in token else str(int(token.rstrip('u'), 0))
        if self.peek() != '(':
            return token
        self.take()
        args = [self.ternary()]
        while self.peek() == ',':
            self.take()
            args.append(self.ternary())
        assert self.take() == ')'
        return '%s(%s)' % (token, ', '.join(args))


def lowered(base, asm):
    """The right-hand side of the one register write the translator emits for `asm`."""
    stage = G.Stage.__new__(G.Stage)
    stage.sym, stage.used_s, stage.used_v, stage.uniform_v = {}, set(), set(), set()
    stage.exec_known, stage.has_branches, stage.lines = True, False, []
    stage.one(base, asm)
    assert len(stage.lines) == 1, stage.lines
    match = re.fullmatch(r'v3 = (.*);', stage.lines[0])
    assert match, stage.lines
    return match.group(1)


def run(text, **regs):
    env = dict(FUNCS, idiv=idiv, mul=mul, add=add, sub=sub, **regs)
    return eval(Parser(text).out, {'__builtins__': {}}, env)


# ---- references ----------------------------------------------------------------------------------------
def ref_frexp_mant(w):
    e = w >> 23 & 255
    if e == 255:
        return w
    if e == 0:
        return w & 0x80000000
    mant, _ = math.frexp(value(w))
    return bits(mant)


def ref_frexp_exp(w):
    e = w >> 23 & 255
    return 0 if e in (0, 255) else math.frexp(value(w))[1] & MASK


def saturate_int(x, word):
    if x != x:
        return 0x7fffffff + (word >> 31)
    return max(-2 ** 31, min(2 ** 31 - 1, math.floor(x))) & MASK if not math.isinf(x) else (
        MASK >> 1 if x > 0 else 0x80000000)


def ref_cvt_flr(w):
    return saturate_int(value(w), w)


def ref_cvt_rpi(w):
    x = value(w)
    return saturate_int(x if math.isinf(x) or x != x else math.floor(x + 0.5), w)


def ref_ldexp(a, n):
    n = n - (1 << 32) if n >= 1 << 31 else n
    x = value(a)
    if x != x or math.isinf(x) or x == 0.0:
        return a
    try:
        return bits(math.ldexp(x, max(-400, min(400, n))))
    except OverflowError:
        return bits(math.copysign(math.inf, x))


NAN, INF = 0x7fc00000, 0x7f800000
FLOATS = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 0.75, 1.5, -2.5, 3.0, 0.1, -0.1, 0.4999, 0.5001, -0.5001, 1e-5,
          123456.789, -98765.4, 1.0e20, -1.0e20, 2.0 ** -126, 2.0 ** 127, 1.7014118e38, 8388608.5, 2147483520.0,
          -2147483648.0, 2.0 ** 31, -2.0 ** 33, 3.4028235e38, -3.4028235e38]
WORDS = [bits(x) for x in FLOATS] + [NAN, INF, INF | 0x80000000, 0x00000001, 0x80400000, 0x007fffff,
                                      0xffc00000, 0x7f800001, 0xff800001, 0x7fffffff, 0xffffffff,
                                      0x3effffff, 0xbf000001, 0x4b000001, 0xcb000001, 0x4b7fffff, 0x4effffff]
SHIFTS = [0, 1, -1, 5, -5, 24, -24, 100, -100, 126, -126, 127, -127, 200, -200, 299, 300, 301, -300, 1000, -1000,
          2 ** 31 - 1, -2 ** 31]

failures = []


def check(label, got, want, case):
    if got != want:
        failures.append(label)
        print('%s: MISMATCH %s got 0x%08x want 0x%08x' % (label, case, got, want))


def unary(label, name, ref):
    for w in WORDS:
        for suffix in ('e32', 'e64'):
            got = run(lowered('%s_%s' % (name, suffix), '%s_%s v3, v1' % (name, suffix)), v1=w)
            check('%s_%s' % (label, suffix), got & MASK, ref(w) & MASK, hex(w))
    print('%s: %d inputs' % (label, len(WORDS)))


def main():
    unary('v_frexp_mant_f32', 'v_frexp_mant_f32', ref_frexp_mant)
    unary('v_frexp_exp_i32_f32', 'v_frexp_exp_i32_f32', ref_frexp_exp)
    unary('v_cvt_flr_i32_f32', 'v_cvt_flr_i32_f32', ref_cvt_flr)
    unary('v_cvt_rpi_i32_f32', 'v_cvt_rpi_i32_f32', ref_cvt_rpi)
    count = 0
    for a in [bits(x) for x in FLOATS if x != 2.0 ** -126] + [NAN, INF, INF | 0x80000000]:
        for n in SHIFTS:
            got = run(lowered('v_ldexp_f32', 'v_ldexp_f32 v3, v1, v2'), v1=a, v2=n & MASK)
            want = ref_ldexp(a, n & MASK)
            if (got & MASK) != want and not (got != got and want != want):
                check('v_ldexp_f32', got & MASK, want, '%s << %d' % (hex(a), n))
            count += 1
    print('v_ldexp_f32: %d inputs' % count)
    # Source and output modifiers: negation reaches the operand, mul:2 scales the result, and the operations
    # that cannot carry a modifier refuse it.
    check('ldexp neg', run(lowered('v_ldexp_f32', 'v_ldexp_f32 v3, -v1, v2'), v1=bits(1.5), v2=2) & MASK,
          bits(-6.0), 'neg')
    check('ldexp mul:2', run(lowered('v_ldexp_f32', 'v_ldexp_f32 v3, v1, v2 mul:2'), v1=bits(1.5), v2=2) & MASK,
          bits(12.0), 'mul:2')
    for asm in ('v_cvt_flr_i32_f32_e64 v3, v1 clamp', 'v_cvt_rpi_i32_f32_e64 v3, v1 mul:2',
                'v_frexp_exp_i32_f32_e64 v3, v1 clamp'):
        try:
            lowered(asm.split()[0], asm)
        except G.Unsupported:
            continue
        failures.append(asm)
        print('accepted a modifier: ' + asm)
    if failures:
        print('FAILED: %d' % len(failures))
        return 1
    print('gfx2hlsl frexp/cvt/ldexp check passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
