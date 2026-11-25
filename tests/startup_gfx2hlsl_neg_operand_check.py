"""The `neg(x)` operand spelling in tools/startup/gfx2hlsl.py (part of #1723).

usage: startup_gfx2hlsl_neg_operand_check.py

LLVM prints a lone negation of a numeric operand as `neg(x)` so that it is not read as a sign
(`-1` would be the integer, `neg(1)` is 1 with its sign bit flipped). The translator only knew the
`-x` spelling, so every instruction carrying `neg(x)` was refused with `Unsupported: operand 'neg(...)'`.
The instruction text is fed to the translator directly, as the disassembler prints it.
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified, self.sampler_heap = {}, set(), set(), False


class Stage(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()


def translate(asm):
    stage = Stage()
    stage.one(asm.split()[0], asm)
    return ' '.join(stage.lines)


def refused(asm):
    try:
        translate(asm)
    except G.Unsupported:
        return True
    return False


# Integer select: neg(x) is the sign bit flipped, on a register and on a literal word.
check('neg(register) on the select flips the sign bit',
      translate('v_cndmask_b32_e64 v31, v30, neg(v30), s0') == 'v31 = (s0 & 1) ? (v30 ^ 0x80000000u) : v30;')
check('neg(literal) on the select flips the sign bit, not the integer sign',
      translate('v_cndmask_b32_e64 v31, neg(0x12345678), v30, s0')
      == 'v31 = (s0 & 1) ? v30 : (0x12345678u ^ 0x80000000u);')
check('neg(inline constant) on the select',
      translate('v_cndmask_b32_e64 v31, v30, neg(1), s0')
      == 'v31 = (s0 & 1) ? (0x00000001u ^ 0x80000000u) : v30;')

# The old spelling keeps its meaning.
check('-register still flips the sign bit',
      translate('v_cndmask_b32_e64 v31, v30, -v30, s0') == 'v31 = (s0 & 1) ? (v30 ^ 0x80000000u) : v30;')
check('plain -1 stays the integer',
      translate('v_cndmask_b32_e64 v31, v30, -1, s0') == 'v31 = (s0 & 1) ? 0xffffffffu : v30;')

# Float sources go through the same modifier.
check('neg(register) on a float add',
      '(-F(v2))' in translate('v_add_f32_e64 v1, neg(v2), v3'), translate('v_add_f32_e64 v1, neg(v2), v3'))
check('neg(literal) on a float add negates the float',
      '(-F(0x3f800001u))' in translate('v_add_f32_e64 v1, neg(0x3f800001), v3'))
check('neg(abs) on a float add',
      '(-abs(F(v2)))' in translate('v_add_f32_e64 v1, neg(|v2|), v3'))

# Float compare reads the word under the modifier for its NaN test.
check('neg(register) on an ordered compare',
      '(v2 & 0x7fffffffu)' in translate('v_cmp_lg_f32_e64 vcc, neg(v2), v3'))

# A negation on an integer op that does not model it stays refused.
check('neg(register) on an integer op that does not model it is refused',
      refused('v_and_b32_e64 v1, neg(v2), v3'))
check('malformed neg( is refused', refused('v_cndmask_b32_e64 v31, v30, neg(v30, s0'))

if failures:
    print('FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('PASS')
