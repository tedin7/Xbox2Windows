"""v_add_co_u32, v_sub_co_u32 and v_subrev_co_u32 in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_add_sub_co_check.py

Synthetic VOP3b words only; the spellings are asserted from the translator's decoder. AMD RDNA 2 ISA: V_ADD_CO_U32
D = S0 + S1 with the carry out of bit 31 written to the lane's bit of SDST; V_SUB_CO_U32 D = S0 - S1 with a borrow
when S1 > S0; V_SUBREV_CO_U32 swaps S0 and S1. These forms are VOP3b only and take no carry-in.

The emitted lines are evaluated with the restricted evaluator of startup_gfx2hlsl_bit_alu_check on a grid of edge
values and compared with Python references written from the ISA description, for an SGPR and a VCC carry-out. Also
checked: the carry-out SGPR becomes a lane mask that is not a uniform source, a readfirstlane of a copy of it is
refused, inactive lanes keep D and get a 0 carry when exec is not known, a destination that overlaps a source reads
the source first, and clamp, a wrong operand count and an EXEC carry-out are refused. Negative control: a borrow
written as S0 > S1 differs from the reference on the grid.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
GRID = [0, 1, 2, 0x7fffffff, 0x80000000, 0xfffffffe, MASK, 0x12345678]
failures = 0


def check(what, ok, detail=''):
    global failures
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', what, '' if ok or not detail else ': %s' % (detail,)))
    if not ok:
        failures += 1


class Pipe:
    root = {'params': [], 'flags': 0}

    def ud(self, source):
        return 0


def vop3b(op, vdst, sdst, src0, src1, clamp=False):
    return [0xd4000000 | op << 16 | (1 << 15 if clamp else 0) | sdst << 8 | vdst, src0 | src1 << 9]


V3, V4, VCC_LO, EXEC_LO = 256 + 3, 256 + 4, 106, 126


def stage(exec_known=True):
    s = G.Stage(Pipe(), 'ps', struct.pack('<I', 0xbf810000))
    s.lines = []
    s.pc = 0
    s.exec_known = exec_known
    return s


def translate(words, exec_known=True):
    s = stage(exec_known)
    _, _, name, text = G.decode(struct.pack('<%dI' % len(words), *words))[0]
    s.lines = []
    s.one(name, text)
    return s, text, s.lines


def evaluate(lines, env):
    """Run the emitted assignments in order on env (register name -> uint)."""
    env = dict(env)
    for line in lines:
        guard = None
        if line.startswith('if (exec & 1) '):
            guard, line = 'exec', line[len('if (exec & 1) '):]
        target, expr = line.rstrip(';').split(' = ', 1)
        value = Expr(expr).run(env)[0] & MASK
        if guard is None or env.get('exec', 1) & 1:
            env[target] = value
    return env


def ref(kind, a, b):
    if kind == 'subrev':
        a, b = b, a
    if kind == 'add':
        total = a + b
        return total & MASK, total >> 32
    return (a - b) & MASK, int(b > a)


FORMS = (('add', 0x30f), ('sub', 0x310), ('subrev', 0x319))


def main():
    count = 0
    for kind, op in FORMS:
        for sdst, carry_reg, spelling in ((18, 's18', 's18'), (VCC_LO, 'vcc', 'vcc_lo')):
            expected = 'v_%s_co_u32 v2, %s, v4, v3' % (kind, spelling)
            try:
                s, text, lines = translate(vop3b(op, 2, sdst, V4, V3))
            except G.Unsupported as error:
                check('%s translates' % expected, False, error)
                continue
            check('decodes as %s' % expected, text == expected, text)
            if carry_reg == 's18':
                check('%s: the carry-out SGPR is a lane mask' % text, 18 in s.masks, s.masks)
                if s.tracks_uniformity():
                    check('%s: the carry-out SGPR is not a uniform source' % text,
                          18 in s.varying_s and not s.uniform_source('s18'), (s.varying_s, s.masks))
            bad = None
            for a in GRID:
                for b in GRID:
                    env = evaluate(lines, {'v4': a, 'v3': b, 'vcc': 0xdeadbeef, 'v2': 0xdeadbeef, 's18': 0xdeadbeef})
                    got = env['v2'], env[carry_reg]
                    if got != ref(kind, a, b) and bad is None:
                        bad = (a, b, got, ref(kind, a, b), lines)
                    count += 1
            check('%s matches the ISA reference on the grid' % text, bad is None, bad)

    # The carry-out is per lane: a readfirstlane of a VGPR copied from it must not be proven uniform.
    s, text, _ = translate(vop3b(0x30f, 2, 18, V4, V3))
    try:
        s.one('v_mov_b32', 'v_mov_b32 v5, s18')
        s.one('v_readfirstlane_b32', 'v_readfirstlane_b32 s20, v5')
    except G.Unsupported as error:
        check('%s: readfirstlane of a copy of the carry-out is refused' % text, 'uniform' in str(error), str(error))
    else:
        check('%s: readfirstlane of a copy of the carry-out is refused' % text, False, s.lines)
    # D of varying sources is not proven uniform either.
    s, text, _ = translate(vop3b(0x30f, 2, 18, V4, V3))
    try:
        s.one('v_readfirstlane_b32', 'v_readfirstlane_b32 s20, v2')
    except G.Unsupported as error:
        check('%s: readfirstlane of D is refused' % text, 2 not in s.uniform_v and 'uniform' in str(error), str(error))
    else:
        check('%s: readfirstlane of D is refused' % text, False, s.lines)

    # exec not known: an inactive lane keeps D and writes a 0 carry; an active lane computes.
    s, text, lines = translate(vop3b(0x30f, 2, 18, V4, V3), exec_known=False)
    off = evaluate(lines, {'v4': MASK, 'v3': 1, 'v2': 7, 's18': 1, 'exec': 0})
    on = evaluate(lines, {'v4': MASK, 'v3': 1, 'v2': 7, 's18': 0, 'exec': 1})
    check('exec not known: an inactive lane keeps D and gets carry 0', (off['v2'], off['s18']) == (7, 0), lines)
    check('exec not known: an active lane computes', (on['v2'], on['s18']) == ref('add', MASK, 1), lines)

    # D overlapping a source.
    s, text, lines = translate(vop3b(0x310, 4, 18, V4, V3))
    env = evaluate(lines, {'v4': 0, 'v3': 1})
    check('%s: D = S0 reads S0 first' % text, (env['v4'], env['s18']) == ref('sub', 0, 1), lines)

    for words, why, what in ((vop3b(0x30f, 2, 18, V4, V3, clamp=True), 'clamp', 'clamp'),
                             (vop3b(0x30f, 2, EXEC_LO, V4, V3), 'carry-out', 'an EXEC carry-out')):
        try:
            _, text, lines = translate(words)
        except G.Unsupported as error:
            check('%s refused' % what, why in str(error), str(error))
        else:
            check('%s refused' % what, False, (text, lines))
    s = stage()
    try:
        s.one('v_add_co_u32', 'v_add_co_u32 v2, vcc_lo, v4, v3, vcc_lo')
    except G.Unsupported as error:
        check('a fifth operand refused', 'operands' in str(error), str(error))
    else:
        check('a fifth operand refused', False, s.lines)

    # Negative control: a borrow with the comparison reversed disagrees with the reference.
    _, _, lines = translate(vop3b(0x310, 2, 18, V4, V3))
    mutated = [l if not l.startswith('w = ') else 'w = (v4 > v3) ? 1u : 0u;' for l in lines]
    differs = any(evaluate(mutated, {'v4': a, 'v3': b})['s18'] != ref('sub', a, b)[1] for a in GRID for b in GRID)
    check('control: a reversed borrow differs from the reference', mutated != lines and differs, mutated)

    if failures:
        print('FAILED: %d check(s)' % failures)
        return 1
    print('add/sub co forms: 6 forms match the ISA references on %d inputs; exec, overlap and refusals checked' % count)
    return 0


if __name__ == '__main__':
    sys.exit(main())
