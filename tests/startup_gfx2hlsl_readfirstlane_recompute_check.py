"""Check v_readfirstlane_b32 past the entry block of a program with branches, without shader machine code.

There the read recomputes the uniform expression that defined its VGPR. It is accepted only when no branch
target (a join or a loop head), EXEC write, overwrite of the VGPR, or write of a scalar the expression reads
lies between the definition and the read, and every scalar source is either written since the last join, a
one-path scalar preserved across an untouched join, or a uniform loop counter whose updates stay uniform.
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
    """Translate disassembly text through the same dispatch and block split as a real stage."""
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.lines, s.attrs = Pipe(), [], {}
    s.sym = {4 + i: ('desc', ('table', 0), 0, i) for i in range(4)}
    s.used_s, s.used_v, s.masks = set(), set(), set()
    s.exec_known = True
    s.ins = [(4 * i, 4, asm.split()[0], asm) for i, asm in enumerate(instructions)]
    try:
        s.translate()
    except G.Unsupported as error:
        return str(error)
    return [line for _, lines, _ in s.blocks for line in lines]


REFUSAL = 'v_readfirstlane_b32 requires a proven uniform source'
EARLIER = {'lane mask source': 'v_mul_f32 reads lane mask s3 as numeric data'}
READ = 'v_readfirstlane_b32 s9, v7'
# Offset 0x8 is a join: the branch at 0x0 skips the instruction at 0x4.
JOIN = ('s_cbranch_scc0 label_0008', 's_nop 0')
LOAD = 's_buffer_load_dword s3, s[4:7], 0x10'      # a uniform scalar written after the join
DEFINE = 'v_mul_f32_e64 v7, s3, 2.0'
RECOMPUTED = 's9 = U(F(s3) * 2.0);'


def read_line(lines):
    return lines[-1] if isinstance(lines, list) and lines else lines


ACCEPTED = [
    ('definition and read after a join', JOIN + (LOAD, DEFINE, READ)),
    # The read does not copy this lane's VGPR, so an EXEC write before the definition does not matter.
    ('EXEC narrowed before the definition', JOIN + ('s_and_saveexec_b32 s8, s2', LOAD, DEFINE, READ)),
    ('EXEC restored before the definition', JOIN + ('s_mov_b32 exec_lo, s8', LOAD, DEFINE, READ)),
    # A conditional branch between them is a fall-through, not a join: one path reaches the read.
    ('fall-through between definition and read', JOIN + (LOAD, DEFINE, 's_cbranch_scc1 label_0020', READ,
                                                          's_endpgm')),
    ('chain through a second VGPR', JOIN + (LOAD, 'v_mov_b32_e32 v6, s3', 'v_mul_f32_e64 v7, v6, 2.0', READ)),
    ('unrelated scalar write in between', JOIN + (LOAD, DEFINE, 's_mov_b32 s2, 0', READ)),
    ('median of three after a join', JOIN + (LOAD, 'v_med3_f32 v7, s3, 0, 0.5', READ)),
    # A one-path scalar written before the join stays proven across a join whose other paths do not write it.
    ('one-path scalar written before the join', (LOAD,) + JOIN + (DEFINE, READ)),
    ('one-path scalar across a divergent branch when untouched',
     (LOAD, 's_cbranch_vccz label_000C', 's_nop 0', DEFINE, READ)),
    # A uniform loop counter initialized before the loop and incremented with uniform scalar ops stays proven.
    ('uniform loop counter read inside the loop',
     ('s_mov_b32 s3, 0', DEFINE, READ, 's_add_u32 s3, s3, 1', 's_cmp_lt_u32 s3, 4', 's_cbranch_scc1 label_0004',
      's_endpgm')),
    ('uniform loop counter read after the loop',
     ('s_mov_b32 s3, 0', 's_add_u32 s3, s3, 1', 's_cmp_lt_u32 s3, 4', 's_cbranch_scc1 label_0004', DEFINE, READ,
      's_endpgm')),
    ('uniform loop counter after an unrelated divergent else write',
     ('v_cmp_lt_f32_e32 vcc_lo, v0, v1', 's_cbranch_vccz label_0010', 's_nop 0', 's_branch label_0014',
      's_mov_b32 s3, 2', 's_mov_b32 s3, 0', 's_add_u32 s3, s3, 1', 's_cmp_lt_u32 s3, 4',
      's_cbranch_scc1 label_0018', DEFINE, READ, 's_endpgm')),
]
EXPECTED = {'chain through a second VGPR': 's9 = U(F((s3)) * 2.0);',
            'median of three after a join': 's9 = U(med3_32(F(s3), F(0x00000000u), 0.5));'}
for label, instructions in ACCEPTED:
    got = run(*instructions)
    want = EXPECTED.get(label, RECOMPUTED)
    ok = isinstance(got, list) and (RECOMPUTED in got if label == 'uniform loop counter read inside the loop' else read_line(got) == want)
    check(f'{label} accepted', ok, repr(got))

REFUSED = [
    # The first active lane at the read may not have run the definition.
    ('EXEC widened between definition and read', JOIN + (LOAD, 's_and_saveexec_b32 s8, s2', DEFINE,
                                                          's_mov_b32 exec_lo, s8', READ)),
    ('EXEC narrowed between definition and read', JOIN + (LOAD, DEFINE, 's_and_saveexec_b32 s8, s2', READ)),
    ('compare writes EXEC between definition and read', JOIN + (LOAD, DEFINE, 'v_cmpx_lt_f32_e32 v0, v1', READ)),
    ('WQM between definition and read', JOIN + (LOAD, DEFINE, 's_wqm_b32 exec_lo, exec_lo', READ)),
    # The recomputed expression would read a newer scalar.
    ('scalar input rewritten in between', JOIN + (LOAD, DEFINE, 's_mov_b32 s3, 0', READ)),
    ('scalar input reloaded in between', JOIN + (LOAD, DEFINE, 's_buffer_load_dword s3, s[4:7], 0x20', READ)),
    ('scalar input pair rewritten in between', JOIN + (LOAD, DEFINE, 's_mov_b64 s[2:3], 0', READ)),
    # A loop head between them: a later iteration reaches the read without the definition.
    ('loop back edge between definition and read', JOIN + (LOAD, DEFINE, 's_nop 0', READ,
                                                            's_cbranch_scc0 label_0014', 's_endpgm')),
    # A join where two definitions reach.
    ('join of two definitions', ('s_cbranch_scc0 label_0014', LOAD, DEFINE, 's_branch label_0014',
                                 'v_mov_b32_e32 v7, 0', READ, 's_endpgm')),
    # A second join (here a loop head at the definition) resets the since-join marks again: on the next iteration
    # s3 holds a lane mask.
    ('scalar written before the second join', JOIN + (LOAD, DEFINE, READ, 's_mov_b32 s3, vcc_lo',
                                                       's_cbranch_scc0 label_000C', 's_endpgm')),
    # A divergent branch or back-edge must still refuse scalars written inside its arms or loop body.
    ('divergent else arm writes scalar before join',
     ('s_mov_b32 s3, 1', 'v_cmp_lt_f32_e32 vcc_lo, v0, v1', 's_cbranch_vccz label_0014', 's_nop 0',
      's_branch label_0018', 's_mov_b32 s3, 2', DEFINE, READ, 's_endpgm')),
    ('loop counter with non-uniform back-edge',
     ('s_mov_b32 s3, 0', DEFINE, READ, 's_add_u32 s3, s3, 1', 'v_cmp_lt_f32_e32 vcc_lo, v0, v1',
      's_cbranch_vccnz label_0004', 's_endpgm')),
    ('lane-varying source', JOIN + ('v_mul_f32_e64 v7, v0, 2.0', READ)),
    ('lane mask source', JOIN + ('v_cmp_lt_f32_e64 s3, v0, v1', DEFINE, READ)),
    ('VGPR overwritten in between', JOIN + (LOAD, DEFINE, 'v_add_f32_e64 v7, v0, 1.0', READ)),
    ('chain across an EXEC write', JOIN + (LOAD, 'v_mov_b32_e32 v6, s3', 's_and_saveexec_b32 s8, s2',
                                           'v_mul_f32_e64 v7, v6, 2.0', READ)),
    ('median of three across an EXEC write', JOIN + (LOAD, 'v_med3_f32 v7, s3, 0, 0.5', 's_and_saveexec_b32 s8, s2',
                                                     READ)),
    ('select on an SCC from before the join', JOIN + ('s_cselect_b32 s3, 1, 0', DEFINE, READ)),
]
for label, instructions in REFUSED:
    got = run(*instructions)
    check(f'{label} refused', got == EARLIER.get(label, REFUSAL), repr(got))

# Writes the translator does not route through the first operand.
probe = G.Stage.__new__(G.Stage)
for name, ops, regs in (('v_add_co_u32_e64', ['v1', 's3', 'v2', 'v3'], {3}),
                        ('v_div_scale_f32', ['v1', 's3', 'v2', 'v3', 'v4'], {3}),
                        ('s_movreld_b32', ['s0', 's2'], {3, 2}),
                        ('s_cmp_eq_u32', ['s3', '0'], set())):
    probe.uniform_expr, probe.since_join = {7: 'U(F(s3) * 2.0)', 8: 'U(F(s2) * 2.0)'}, {2, 3}
    probe.forget_writes(name, ops)
    check(f'{name} forgets s{sorted(regs)}', probe.since_join == {2, 3} - regs and
          set(probe.uniform_expr) == {7, 8} - {7 if 3 in regs else 0, 8 if 2 in regs else 0},
          repr((probe.since_join, probe.uniform_expr)))
for name, ops, expected in (('s_mov_b64', ['exec', 's[2:3]'], True), ('s_mov_b32', ['exec_hi', 's2'], True),
                            ('s_load_dwordx2', ['s[126:127]', 's[2:3]', '0x0'], True),
                            ('s_mov_b32', ['s125', 's2'], False), ('s_mov_b32', ['s126', 's2'], True),
                            ('s_mov_b64', ['s[124:125]', 's[2:3]'], False),
                            ('s_or_saveexec_b64', ['s[2:3]', 's[4:5]'], True),
                            ('s_andn2_wrexec_b32', ['s2', 's4'], True), ('v_cmpx_lt_f32_e32', ['v0', 'v1'], True),
                            ('s_movreld_b32', ['s0', 's2'], True), ('s_mov_b32', ['s2', 'exec_lo'], False),
                            ('v_add_co_u32_e64', ['v1', 'exec_lo', 'v2', 'v3'], True),
                            ('v_mul_f32_e64', ['v1', 'exec_lo', 'v2'], False), ('s_cmp_eq_u32', ['exec_lo', '0'], False)):
    check(f'{name} {ops[0]} writes EXEC: {expected}', G.writes_exec(name, ops) == expected)

print('Readfirstlane recompute checks: %d failed' % len(failures))
sys.exit(1 if failures else 0)
