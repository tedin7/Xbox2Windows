"""v_cvt_pknorm_u16_f32 of tools/startup/gfx2hlsl.py.

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel body and compare it with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

The instruction is VOP3 only (opcode 0x369, no e32 form): bits 15:0 of the result hold clamp(S0, 0, 1) * 65535 and
bits 31:16 hold clamp(S1, 0, 1) * 65535, each rounded to nearest even, with a NaN reading 0. Source neg and abs apply
to the float inputs. An output modifier or op_sel changes the packed words and is refused. The fixture and runner are
startup_gfx2hlsl_lane_alu.py.

The reference rounds the binary32 product clamp(x) * 65535, which is what the translated shader computes. For 1.5/65535
that product is exactly 1.5 and ties to 2, while the exact rational value of the nearest binary32 input is just below
1.5; the reference follows the binary32 product. No hardware run of this instruction was available when this check was
written, so the draw mode (WARP and GPU) is what confirms it.

A result with a high half of 0 is a binary32 denormal when the fixture exports it to the float target, and FXC ps_5_1
reads such a word as 0 (see startup_gfx2hlsl_cvt_alu_check.py), so the draw goes through DXC ps_6_0 only; an optional FXC
argument is accepted and ignored.

Three mutated lowerings are checked, one per rule: the rounding of the low half, the upper clamp of the high half, and
the order of the two halves.
"""
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import startup_gfx2hlsl_lane_alu as lane  # noqa: E402
from startup_gfx2hlsl_lane_alu import check_draw, words  # noqa: E402

OP = 'v_cvt_pknorm_u16_f32'


def f32(value):
    """The binary32 word of a Python number, rounded to nearest even."""
    return struct.unpack('<I', struct.pack('<f', value))[0]


def value(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def to_f32(number):
    return value(f32(number))


def unorm16(word):
    x = value(word)
    if x != x:
        return 0
    x = min(max(x, 0.0), 1.0)
    return round(to_f32(x * 65535.0))  # Python's round() is round half to even


def reference(a, b):
    return unorm16(a) | unorm16(b) << 16


def swapped(a, b):
    return reference(b, a)


def floor_low(a, b):
    x = value(a)
    low = 0 if x != x else int(to_f32(min(max(x, 0.0), 1.0) * 65535.0))
    return low | unorm16(b) << 16


def no_upper_clamp(a, b):
    x = value(b)
    high = 0 if x != x or x in (float('inf'), float('-inf')) else round(to_f32(max(x, 0.0) * 65535.0))
    return unorm16(a) | (high << 16 & 0xffffffff)


def pair(low, high):
    return (f32(low), f32(high), 0)


NAN, INF = 0x7fc00000, float('inf')
# Eight sets, two draws. The first covers the clamp ends, the ties (0.5, 1.5 and 2.5 over 65535, and 0.25 and 0.5
# exactly) and a negative and an above-one value; the second a NaN, both infinities, -0 and a low half that differs
# from the high half in every lane, so a swapped pair cannot pass.
DRAW = {OP: [pair(0.0, 1.0), pair(-0.5, 1.5), pair(1.5 / 65535, 0.5 / 65535), pair(2.5 / 65535, 0.25),
             (NAN, f32(INF), 0), (0x80000000, f32(-INF), 0), (f32(INF), NAN, 0), pair(3.5 / 65535, 0.5)]}
OPS = {OP: (2, reference, False)}
BODY = {OP: words('0c0069d7000902000d0069d7010b02000e0069d7020d02000f0069d7030f0200')}
MUTATIONS = {
    'rounding': (r'uint\(round\((saturate\(F\(v\d+\)\) \* 65535\.0)\)\) \|', r'uint(floor(\1)) |', floor_low),
    'clamp': (r'\| \(uint\(round\(saturate\((F\(v\d+\))\)', r'| (uint(round(max(\1, 0.0)', no_upper_clamp),
    'halves': (r'= (uint\(round\(saturate\(F\(v\d+\)\) \* 65535\.0\)\)) \| \((uint\(round\(saturate\(F\(v\d+\)\) \* '
               r'65535\.0\)\)) << 16\);', r'= \2 | (\1 << 16);', swapped),
}
SHARED = {
    'ops': OPS, 'draw': DRAW, 'body': BODY,
    'accepted': [OP + ' v1, v2, v3', OP + '_e64 v1, -v2, |v3|', OP + ' v1, 0.5, s2'],
    'refused': [OP + '_e64 v1, v2, v3 clamp', OP + '_e64 v1, v2, v3 mul:2', OP + '_e64 v1, v2, v3 div:2',
                OP + ' v1, v2', OP + ' v1, v2, v3, v4'],
}
FAMILIES = {name: dict(SHARED, mutations={OP: mutation}) for name, mutation in MUTATIONS.items()}


class F32(float):
    """A binary32 value for the evaluation of translated text: a product rounds to binary32, as the shader's does."""

    def __mul__(self, other):
        return F32(to_f32(float(self) * float(other)))


def evaluate(text, sets):
    """Run the four translated lane lines of one draw on its input sets, with HLSL's saturate (NaN reads 0), round
    (half to even) and uint, and return the four words."""
    env = {'F': lambda word: F32(value(word)), 'uint': lambda x: int(x) & 0xffffffff, 'round': round,
           'saturate': lambda x: F32(0.0 if x != x else min(max(x, 0.0), 1.0)), 'abs': abs,
           'floor': lambda x: F32(float(int(x))), 'max': lambda a, b: F32(max(a, b)) if a == a else a}
    for lane_no, inputs in enumerate(sets):
        env['v%d' % lane_no] = inputs[0]
        env['v%d' % (4 + lane_no)] = inputs[1]
    found = re.findall(r'^\s*v(1[2-5]) = (.*);$', text, re.M)
    return [eval(expression, env) & 0xffffffff for _, expression in found]  # noqa: S307 - translator output


def lowering_matches(family):
    """The translated text gives the reference words on every draw set; the mutated text gives its own words and
    not the reference on the first draw."""
    sets = family['draw'][OP]
    good = lane.translate(family, OP)[1]
    for index in range(0, len(sets), 4):
        chunk = sets[index:index + 4]
        lane.check('lowering words, sets %d..%d' % (index, index + 3),
                   evaluate(good, chunk) == [reference(a, b) for a, b, _ in chunk])
    mutated = lane.translate(family, OP, True)[1]
    chunk = sets[:4]
    got, wrong = evaluate(mutated, chunk), [family['mutations'][OP][2](a, b) for a, b, _ in chunk]
    lane.check('mutated lowering reads its own words', got == wrong and got != [reference(a, b) for a, b, _ in chunk])


def source_modifiers():
    stage = lane.G.Stage.__new__(lane.G.Stage)
    stage.lines, stage.sym = [], {}
    stage.used_s, stage.used_v, stage.uniform_v, stage.exec_known = set(), set(), set(), True
    stage.one(OP + '_e64', OP + '_e64 v1, -v2, |v3|')
    line = stage.lines[-1]
    lane.check('neg and abs reach the sources', 'saturate((-F(v2)))' in line and 'saturate(abs(F(v3)))' in line)


def main():
    if len(sys.argv) == 1:
        for family in FAMILIES.values():
            lane.controls(family)
            lowering_matches(family)
        source_modifiers()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        lane.check_binary(next(iter(FAMILIES.values())), *args)
    elif mode == 'draw':
        for family in FAMILIES.values():
            check_draw(family, *args[:3])  # DXC only: see the note on denormal result words above
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
