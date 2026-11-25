"""Scalar constants across branch joins in tools/startup/gfx2hlsl.py, without shader machine code.

  (no arguments)   translation controls, pure Python, no device

The translator folds a scalar register to its value while that value is known (`('const', n)` in Stage.sym). At a
branch target another path arrives, so the value is only known there if no other path writes the register:
  - a loop head is reached again over the back edge, after the loop body ran: a counter set to 0 before the loop
    and incremented in it must be read as the register, not as 0;
  - a forward branch skips instructions: a register written only on the fall-through path must be read as the
    register after the join.
A constant that no other path writes stays folded (control). Folding the counter froze it at 1, so the exit test
fired only for a trip count of 1 and every other loop ran until the dispatcher guard (65,536 iterations).

Mutations (each must turn the battery red): constants kept at every join (the behavior before this check), and
the loop body range left out (only forward branches handled).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH  ' + detail}")
    if not condition:
        failures.append(label)


class Pipe:
    def resource(self, key, kind):
        return 'r0'


def run(*instructions):
    """Translate disassembly text with the same dispatch and branch scan as a real stage -> emitted lines."""
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.lines = Pipe(), []
    s.attrs = {}
    s.sym = {}
    s.used_s, s.used_v, s.masks = set(), set(), set()
    s.exec_known = True
    s.ins = [(4 * i, 4, asm.split()[0], asm) for i, asm in enumerate(instructions)]
    s.translate()
    return [line for _, lines, _ in s.blocks for line in lines]


CASES = [
    # The shape of the slow loading-screen loops: counter = 0, then per pass counter += 1 and a compare.
    ('loop counter', ('s_mov_b32 s14, 0', 's_add_u32 s14, s14, 1', 'v_cvt_f32_i32_e32 v13, s14',
                      's_cmp_eq_u32 s14, s2', 's_cbranch_scc0 label_0004', 's_endpgm'),
     ['s14 = s14 + 0x00000001u;', 'v13 = U(float(int(s14)));'],
     ['s14 = 0x00000000u + 0x00000001u;', 'v13 = U(float(int(0x00000000u)));']),
    ('loop offset from counter', ('s_mov_b32 s9, 0', 's_lshl_b32 s10, s9, 3', 's_add_u32 s9, s9, 1',
                                  's_cmp_eq_u32 s9, s18', 's_cbranch_scc0 label_0004', 's_endpgm'),
     ['s9 = s9 + 0x00000001u;'],
     ['s9 = 0x00000000u + 0x00000001u;']),
    ('forward join', ('s_mov_b32 s5, 0', 's_cmp_eq_u32 s2, 0', 's_cbranch_scc1 label_0010', 's_mov_b32 s5, 7',
                      's_add_u32 s6, s5, 1', 's_endpgm'),
     ['s6 = s5 + 0x00000001u;'],
     ['s6 = 0x00000000u + 0x00000001u;', 's6 = 0x00000007u + 0x00000001u;']),
    # Control: s7 is set before the loop and never written in it, so it stays folded.
    ('constant not written in the loop', ('s_mov_b32 s7, 3', 's_mov_b32 s14, 0', 's_add_u32 s8, s7, 1',
                                          's_add_u32 s14, s14, 1', 's_cmp_eq_u32 s14, s2', 's_cbranch_scc0 label_0008',
                                          's_endpgm'),
     ['s8 = 0x00000003u + 0x00000001u;', 's14 = s14 + 0x00000001u;'],
     ['s8 = s7 + 0x00000001u;']),
]

for label, program, wanted, unwanted in CASES:
    lines = run(*program)
    missing = [w for w in wanted if w not in lines]
    present = [u for u in unwanted if u in lines]
    check(label, not missing and not present, f'missing {missing} present {present} in {lines}')

# The loop offset case: the shift reads the counter, not 0. The shift may be emitted in either operand order.
lines = run('s_mov_b32 s9, 0', 's_lshl_b32 s10, s9, 3', 's_add_u32 s9, s9, 1', 's_cmp_eq_u32 s9, s18',
            's_cbranch_scc0 label_0004', 's_endpgm')
shift = [line for line in lines if line.startswith('s10 = ')]
check('loop offset reads the counter', len(shift) == 1 and 's9' in shift[0] and '0x00000000u <<' not in shift[0],
      f'{shift}')

if failures:
    print('FAIL:', ', '.join(failures))
    sys.exit(1)
print('PASS')
