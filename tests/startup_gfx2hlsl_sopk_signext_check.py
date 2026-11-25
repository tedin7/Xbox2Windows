"""SOPK 16-bit immediates in tools/startup/gfx2hlsl.py: s_movk_i32, s_cmpk_*_i32 and s_cmpk_*_u32.

usage: startup_gfx2hlsl_sopk_signext_check.py

RDNA 2 ISA, SOPK: S_MOVK_I32 is D.i32 = signext(SIMM16[15:0]); S_CMPK_{EQ,LG,GT,GE,LT,LE}_I32 compare S0.i32 with
signext(SIMM16[15:0]); the _U32 forms compare S0.u32 with SIMM16[15:0] zero-extended. LLVM 15 SOPInstructions.td agrees:
the _I32 forms take s16imm (isSignExt = 1), the _U32 forms u16imm (isSignExt = 0), and s_movk_i32 takes s16imm. The
disassembler prints the field as unsigned hex (0x8000, 0xffff).
The reference is plain Python arithmetic; the emitted text is evaluated on the CPU.
"""
import operator
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
import gfx2hlsl as G  # noqa: E402

failures = []


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for single instructions to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.masks = set()


def signed32(value):
    value &= 0xffffffff
    return value - (1 << 32) if value & 0x80000000 else value


def run(program, regs):
    """Translate each line, then evaluate the emitted assignments over regs; returns the final registers."""
    stage = Probe()
    for line in program:
        stage.one(line.split(' ', 1)[0], line)
    env = dict(regs)
    for text in stage.lines:
        target, _, expression = text.rstrip(';').partition(' = ')
        expression = re.sub(r'\b(0x[0-9a-f]+|\d+)u\b', r'\1', expression)
        env[target] = int(eval(expression, {'int': signed32, 'uint': lambda x: x & 0xffffffff}, env)) & 0xffffffff
    return env


def check(label, got, want):
    if got != want:
        failures.append(label)
        print('MISMATCH %s got=%r want=%r' % (label, got, want))


def signext16(word):
    return ((word & 0xffff) ^ 0x8000) - 0x8000


IMMEDIATES = (0x0, 0x1, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff)

# s_movk_i32: constant-tracking path (s0) and run-time path (m0), seen through a consumer comparison.
for imm in IMMEDIATES:
    want = signext16(imm) & 0xffffffff
    for dest in ('s0', 'm0'):
        env = run(['s_movk_i32 %s, 0x%x' % (dest, imm)], {})
        check('s_movk_i32 %s, %#x' % (dest, imm), env.get(dest), want)
        for expected, label in ((want, 'equal'), (imm, 'zero-extended')):
            env = run(['s_movk_i32 %s, 0x%x' % (dest, imm), 's_cmp_eq_u32 s3, %s' % dest], {'s3': expected})
            check('s_movk_i32 %s, %#x consumer vs %s' % (dest, imm, label),
                  bool(env['scc']), expected == want)
# a negative decimal spelling is the same field
check('s_movk_i32 -1', run(['s_movk_i32 s0, -1'], {}).get('s0'), 0xffffffff)
check('s_movk_i32 -32768', run(['s_movk_i32 s0, -32768'], {}).get('s0'), 0xffff8000)

COMPARE = {'eq': operator.eq, 'lg': operator.ne, 'gt': operator.gt, 'ge': operator.ge, 'lt': operator.lt, 'le': operator.le}
WORDS = (0x0, 0x1, 0x7fff, 0x8000, 0xffff, 0x10000, 0x7fffffff, 0x80000000, 0xffff7fff, 0xffff8000, 0xffff8001, 0xffffffff)
cases = 0
for name, relation in COMPARE.items():
    for imm in IMMEDIATES:
        for word in WORDS:
            for kind, rhs, lhs in (('i32', signext16(imm), signed32(word)), ('u32', imm, word)):
                line = 's_cmpk_%s_%s s5, 0x%x' % (name, kind, imm)
                check('%s s5=%#x' % (line, word), bool(run([line], {'s5': word})['scc']), relation(lhs, rhs))
                cases += 1
# the issue case: 0 < -1 is false
check('s_cmpk_lt_i32 s0=0, 0xffff', bool(run(['s_cmpk_lt_i32 s5, 0xffff'], {'s5': 0})['scc']), False)
check('s_cmpk_lt_i32 s0=0, -1', bool(run(['s_cmpk_lt_i32 s5, -1'], {'s5': 0})['scc']), False)

# negative controls: the translator must refuse what the encoding cannot hold, and what has no lowering
refused = 0
for program in (['s_movk_i32'], ['s_movk_i32 s0, 0x10000'], ['s_movk_i32 s0, -0x8001'], ['s_movk_i32 s0, -32769'], ['s_movk_i32 s0, s1'],
                ['s_cmpk_lt_i32 s5, 0x10000'], ['s_cmpk_lt_i32 s5, -32769'], ['s_cmpk_lt_i32 s5, s6'],
                ['s_cmpk_lt_u32 s5, 0x10000'], ['s_cmpk_lt_u32 s5, -1']):
    try:
        run(program, {'s5': 0, 's1': 0, 's6': 0})
    except G.Unsupported:
        refused += 1
    else:
        failures.append(program[0])
        print('ACCEPTED %s' % program[0])

print('sopk_signext: %s' % ('ok (%d compare cases, 2 movk paths x %d immediates, %d refusals)' % (cases, len(IMMEDIATES), refused)
                           if not failures else '%d mismatches' % len(failures)))
sys.exit(1 if failures else 0)
