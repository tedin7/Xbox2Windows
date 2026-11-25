"""s_*_saveexec whose destination is its own source reads the source before writing the destination.

usage: startup_gfx2hlsl_saveexec_same_reg_check.py

RDNA 2 ISA, SOP1 S_AND_SAVEEXEC_B64: D = EXEC; EXEC = S0 & EXEC; SCC = (EXEC != 0), where S0 is the value read at
issue (S_OR_SAVEEXEC and S_ANDN2_SAVEEXEC likewise with | and S0 & ~EXEC). Compiled code uses the form with
D == S0, for example `s_and_saveexec_b64 s[6:7], s[6:7]` after a compare into s[6:7]. tools/startup/gfx2hlsl.py
wrote `s6 = exec;` first and then took `exec & s6`, which is exec & exec: the lane entered the branch whatever the
compare gave. S0 now goes through the scratch word `w` when it is the destination.

Synthetic instruction text and register states only. Each case runs the emitted lines with the restricted
evaluator of startup_gfx2hlsl_bit_alu_check for a lane with exec = 1 and S0 = 0 or 1, and checks the new exec, SCC
and D (the old exec) against the ISA rule. Controls: with S0 a different register (s[20:21]) no scratch copy is
made and the result is the same; the old line order (D written before S0 is read) is evaluated too and must give
a different exec for S0 = 0.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

failures = 0


def check(what, ok, detail=''):
    global failures
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', what, '' if ok or not detail else ': %s' % (detail,)))
    if not ok:
        failures += 1


class Pipe:
    sampler_heap = False
    root = {'params': [{'type': 2}], 'flags': 0}

    def resource(self, key, kind):
        return 'r0'

    def ud(self, source):
        return 0


def stage():
    s = G.Stage(Pipe(), 'ps', struct.pack('<I', 0xbf810000))
    s.lines = []
    s.pc = 0
    s.masks.add(6)
    s.masks.add(20)
    return s


def statements(lines):
    for line in lines:
        for stmt in line.split(';'):
            stmt = stmt.strip()
            if stmt:
                name, expr = stmt.split(' = ', 1)
                yield name.strip(), expr


def evaluate(lines, env):
    env = dict(env)
    for name, expr in statements(lines):
        env[name] = Expr(expr).run(env)[0]
    return env


# (instruction, ISA result for the lane's exec bit given exec = 1 and S0)
FORMS = [('s_and_saveexec_b64', lambda s0: s0 & 1), ('s_or_saveexec_b64', lambda s0: (s0 | 1) & 1),
         ('s_andn2_saveexec_b64', lambda s0: s0 & ~1 & 1), ('s_and_saveexec_b32', lambda s0: s0 & 1)]


def main():
    for op, rule in FORMS:
        same = 's[6:7], s[6:7]' if op.endswith('b64') else 's6, s6'
        other = 's[6:7], s[20:21]' if op.endswith('b64') else 's6, s20'
        for operands, src, label in ((same, 's6', 'D == S0'), (other, 's20', 'distinct S0')):
            s = stage()
            s.one(op, '%s %s' % (op, operands))
            for s0 in (0, 1):
                env = evaluate(s.lines, {'exec': 1, 'scc': 0, 'w': 0, 's6': s0 if src == 's6' else 7, 's7': 0, 's20': s0, 's21': 0})
                want = rule(s0)
                check('%s %s, S0 = %d: exec %d' % (op, label, s0, want),
                      env['exec'] == want and int(bool(env['scc'])) == int(want != 0) and env['s6'] == 1, s.lines)
            uses_w = any(name == 'w' for name, _ in statements(s.lines))
            check('%s %s: %s scratch copy' % (op, label, 'a' if src == 's6' else 'no'), uses_w == (src == 's6'), s.lines)
    # Negative control: the old order, D written before S0 is read, admits the lane for S0 = 0.
    old = ['s6 = exec;', 'exec = (exec & s6) & 1; scc = exec != 0;']
    check('negative control: the old order gives exec 1 for S0 = 0', evaluate(old, {'exec': 1, 's6': 0})['exec'] == 1)
    print('saveexec same-register checks: %d failed' % failures)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
