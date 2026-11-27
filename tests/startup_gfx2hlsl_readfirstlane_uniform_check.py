"""Check v_readfirstlane_b32 lowering in entry blocks and forward pixel flow."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402
from startup_firstlane_pixel_fixture import check_translation, check_whole_quad_translation  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH  ' + detail}")
    if not condition:
        failures.append(label)


class Pipe:
    """The minimal pipeline resources needed by the load probes."""

    resources = {(('table', 0), 0): {'index': 0}}  # the formatted load reads the stride and FORMAT words by index

    def __init__(self):
        self.modified, self.other_modified, self.uavs = set(), set(), {}

    def resource(self, key, kind):
        return 'r0'

    def uav(self, key, kind='raw'):
        return 'w0'

    def sampler(self, words, lod_limits=False):
        return 'smp0'

    def stored_images(self):
        return self.uavs

    def ud(self, key):
        return 0


REFUSAL = 'v_readfirstlane_b32 requires a proven uniform source'
READ = 'v_readfirstlane_b32 s9, v7'
MOVE = 'v_mov_b32_e32 v7, s2'
COMPARE = 'v_cmp_lt_f32_e64 s2, v0, v1'
ALIAS_LOAD = 's_buffer_load_dwordx2 s[2:3], s[4:7], s2'


def run(*instructions, exec_known=True, kind=None, group=None):
    """Translate disassembly text using the same dispatch and branch scan as a real stage."""
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.lines = Pipe(), []
    if group:
        s.pipe.group = group
    s.attrs = {}
    s.sym = {4 + i: ('desc', ('table', 0), 0, i) for i in range(4)}
    s.sym.update({16 + i: ('desc', ('table', 0), 32, i) for i in range(8)})
    s.sym.update({24 + i: ('desc', ('table', 0), 64, i) for i in range(8)})
    s.sym[3] = ('fixed_buffer_word',)
    s.used_s, s.used_v, s.masks = set(), set(), set()
    s.exec_known = exec_known
    if kind:
        s.kind = kind
    s.ins = [(4 * i, 4, asm.split()[0], asm) for i, asm in enumerate(instructions)]
    try:
        s.translate()
    except G.Unsupported as error:
        return str(error)
    return [line for _, lines, _ in s.blocks for line in lines]


# Cover unary scalar inputs, binary and ternary immediate inputs, both encoding suffixes,
# and the fixed buffer word that was already handled before uniformity propagation.
ACCEPTED = [
    ('unsigned conversion', ('v_cvt_u32_f32_e32 v7, s2',),
     ['v7 = uint(max(F(s2), 0.0));']),
    ('signed conversion', ('v_cvt_i32_f32_e32 v7, s2',),
     ['v7 = uint(int(F(s2)));']),
    ('fixed buffer word', ('v_cvt_i32_f32_e32 v7, s3',),
     ['v7 = uint(int(F(s3)));']),
    ('unsigned to float conversion', ('v_cvt_f32_u32_e32 v7, s2',),
     ['v7 = U(float(s2));']),
    # The unsigned-division reciprocal estimate, the shape that reaches the read in compiled shaders.
    ('division reciprocal chain', ('v_cvt_f32_u32_e32 v7, s2', 'v_rcp_f32_e32 v7, v7',
                                   'v_mul_f32_e32 v7, 0x4f7ffffe, v7', 'v_cvt_u32_f32_e32 v7, v7'),
     ['v7 = U(float(s2));', 'v7 = U(1.0 / F(v7));', 'v7 = U(F(0x4f7ffffeu) * F(v7));',
      'v7 = uint(max(F(v7), 0.0));']),
    ('byte conversion', ('v_cvt_f32_ubyte0_e32 v7, s2',),
     ['v7 = U(float(s2 & 255));']),
    ('multiply', ('v_mul_f32_e64 v7, s2, 2.0',),
     ['v7 = U(F(s2) * 2.0);']),
    ('clamped multiply', ('v_mul_f32_e64 v7, s2, 2.0 clamp',),
     ['v7 = U(saturate(F(s2) * 2.0));']),
    ('scaled multiply', ('v_mul_f32_e64 v7, s2, 2.0 mul:2',),
     ['v7 = U((F(s2) * 2.0) * 2.0);']),
    ('negated scalar source', ('v_mul_f32_e64 v7, -s2, 2.0',),
     ['v7 = U((-F(s2)) * 2.0);']),
    ('absolute scalar source', ('v_mul_f32_e64 v7, |s2|, 2.0',),
     ['v7 = U(abs(F(s2)) * 2.0);']),
    ('fused multiply-add', ('v_fma_f32 v7, s2, 2.0, 1.0',),
     ['v7 = U(F(s2) * 2.0 + 1.0);']),
    ('literal multiply-add', ('v_fmaak_f32 v7, s2, 2.0, 0x3f800000',),
     ['v7 = U(F(s2) * 2.0 + F(0x3f800000u));']),
    ('reciprocal', ('v_rcp_f32_e32 v7, s2',),
     ['v7 = U(1.0 / F(s2));']),
    ('base-2 logarithm', ('v_log_f32_e32 v7, s2',),
     ['v7 = U(log2(F(s2)));']),
    ('clamped log2 of abs scalar', ('v_log_f32_e64 v7, |s2| clamp',),
     ['v7 = U(saturate(log2(abs(F(s2)))));']),
    ('ceiling', ('v_ceil_f32_e32 v7, s2',),
     ['v7 = U(ceil(F(s2)));']),
    ('scaled ceiling of negated scalar', ('v_ceil_f32_e64 v7, -s2 mul:2',),
     ['v7 = U((ceil((-F(s2)))) * 2.0);']),
    ('log2 then ceil chain', ('v_log_f32_e32 v7, s2', 'v_ceil_f32_e32 v7, v7'),
     ['v7 = U(log2(F(s2)));', 'v7 = U(ceil(F(v7)));']),
    # The clamp to [0, 0.5] of a scalar, as compiled shaders write it.
    ('median of three', ('v_med3_f32 v7, s2, 0, 0.5',),
     ['v7 = U(med3_32(F(s2), F(0x00000000u), 0.5));']),
    ('median of three scalar sources', ('v_med3_f32 v7, s2, -s3, |s2| clamp',),
     ['v7 = U(saturate(med3_32(F(s2), (-F(s3)), abs(F(s2)))));']),
    ('move', ('v_mov_b32_e32 v7, s2',), ['v7 = s2;']),
    ('literal move', ('v_mov_b32_e32 v7, 0x1234',), ['v7 = 0x00001234u;']),
    ('chain with an in-place write', ('v_mul_f32_e64 v7, s2, 2.0', 'v_cvt_u32_f32_e32 v7, v7'),
     ['v7 = U(F(s2) * 2.0);', 'v7 = uint(max(F(v7), 0.0));']),
    ('cross-register chain', (MOVE, 'v_mul_f32_e64 v8, v7, 2.0'),
     ['v7 = s2;', 'v8 = U(F(v7) * 2.0);'], 8),
]
for label, instructions, lines, *source in ACCEPTED:
    reg = source[0] if source else 7
    got = run(*instructions, 'v_readfirstlane_b32 s9, v%d' % reg)
    check(label, got == lines + ['s9 = v%d;' % reg], repr(got))

REFUSED = [
    ('restored exec copy after WQM',
     ('s_mov_b64 s[10:11], exec', 's_wqm_b64 exec, exec', 's_mov_b64 s[12:13], exec', 's_mov_b64 exec, s[10:11]',
      'v_mov_b32_e32 v7, s2', 's_mov_b64 exec, s[12:13]', READ)),
    ('overlapping 64-bit move high word',
     ('v_cmp_lt_f32_e64 s2, v0, v1', 's_mov_b64 s[3:4], s[2:3]', 'v_mov_b32_e32 v7, s4', READ)),
    ('descriptor move high word',
     ('v_cmp_lt_f32_e64 s11, v0, v1', 's_buffer_load_dword s8, s[4:7], 0x0', 's_mov_b64 s[10:11], s[7:8]',
      'v_mov_b32_e32 v7, s11', READ)),
    ('mask from uniform compare', ('v_cmp_eq_u32_e64 s2, 0, 0', 'v_mov_b32_e32 v7, s2', READ)),
    ('entry VGPR', (MOVE, 'v_mov_b32_e32 v7, v0', READ)),
    ('vcc', (MOVE, 'v_mov_b32_e32 v7, vcc_lo', READ)),
    ('exec', (MOVE, 'v_mov_b32_e32 v7, exec_lo', READ)),
    ('WQM', (MOVE, 's_wqm_b64 exec, exec', READ)),
    ('compare writes exec', (MOVE, 'v_cmpx_lt_f32_e32 v0, v1', READ)),
    ('saveexec writes exec', (MOVE, 's_and_saveexec_b32 s8, s2', READ)),
    ('high word of saved exec', ('s_mov_b64 s[2:3], exec', 'v_mov_b32_e32 v7, s3', READ)),
    ('compare into SGPR', (COMPARE, MOVE, READ)),
    ('aliased load first word', (COMPARE, ALIAS_LOAD, MOVE, READ)),
    ('aliased load second word', (COMPARE, ALIAS_LOAD, 'v_mov_b32_e32 v7, s3', READ)),
    ('varying scalar buffer offset',
     (COMPARE, 's_buffer_load_dword s3, s[4:7], s2', 'v_mov_b32_e32 v7, s3', READ)),
    ('listed op with varying source', (MOVE, 'v_mul_f32_e64 v7, v0, 2.0', READ)),
    ('log2 of a varying source', ('v_log_f32_e32 v7, v0', READ)),
    ('ceil of a varying source', ('v_ceil_f32_e32 v7, v0', READ)),
    ('log2 of a lane mask', (COMPARE, 'v_log_f32_e32 v7, s2', READ)),
    ('ceil of a lane mask', (COMPARE, 'v_ceil_f32_e32 v7, s2', READ)),
    ('median of three with a varying source', (MOVE, 'v_med3_f32 v7, s2, v0, 0.5', READ)),
    ('median of three of a lane mask', (COMPARE, 'v_med3_f32 v7, s2, 0, 0.5', READ)),
    ('op_sel overwrite', (MOVE, 'v_mul_f32_e64 v7, s2, 2.0 op_sel:[0,0]', READ)),
    ('modified move overwrite', (MOVE, 'v_mov_b32_e32 v7, s2 clamp', READ)),
    ('multi-register read destination', (MOVE, 'v_readfirstlane_b32 s[8:9], v7')),
    ('multi-register read source',
     (MOVE, 'v_mov_b32_e32 v6, s2', 'v_readfirstlane_b32 s9, v[6:7]')),
    ('saved exec', ('s_mov_b32 s2, exec_lo', 'v_mov_b32_e32 v7, s2', READ)),
    ('mask through scalar arithmetic',
     ('s_mov_b32 s2, vcc_lo', 's_add_u32 s3, s2, 1', 'v_mov_b32_e32 v7, s3', READ)),
    ('mask through scalar condition',
     ('s_cmp_eq_u32 vcc_lo, 0', 's_cselect_b32 s2, 0, 1', 'v_mov_b32_e32 v7, s2', READ)),
    ('non-listed overwrite', ('v_mov_b32_e32 v7, s2', 'v_sin_f32_e32 v7, s2', READ)),
    ('SDWA overwrite', ('v_mov_b32_e32 v7, s2',
                        'v_cvt_f32_ubyte0_sdwa v7, s2 src0_sel:DWORD dst_sel:DWORD', READ)),
    ('interpolant overwrite', ('v_mov_b32_e32 v7, s2', 'v_interp_mov_f32 v7, p0, attr0.x', READ)),
    ('interpolation p1 overwrite', (MOVE, 'v_interp_p1_f32_e32 v7, v0, attr0.x', READ)),
    ('VGPR buffer address', (MOVE, 'buffer_load_dword v7, v0, s[4:7], 0 offen', READ)),
    # The unsigned-to-float conversion proves nothing from a lane-varying source, across an exec
    # write or an overwrite, or from a definition in another block.
    ('unsigned to float conversion of an entry VGPR', ('v_cvt_f32_u32_e32 v7, v0', READ)),
    ('unsigned to float conversion of a mask', (COMPARE, 'v_cvt_f32_u32_e32 v7, s2', READ)),
    ('exec write inside the chain', ('v_cvt_f32_u32_e32 v7, s3', 's_and_saveexec_b32 s8, s2',
                                     'v_rcp_f32_e32 v7, v7', READ)),
    ('overwrite inside the chain', ('v_cvt_f32_u32_e32 v7, s3', 'v_add_f32_e64 v7, v0, 1.0',
                                    'v_rcp_f32_e32 v7, v7', READ)),
    ('definition in another block', ('v_cvt_f32_u32_e32 v7, s3', 's_cbranch_scc0 label_0008',
                                     'v_rcp_f32_e32 v7, v7', READ, 's_endpgm')),
    # A program with branches proves only before its first branch target or terminator (the entry block).
    ('read at the first branch target', (MOVE, 's_cbranch_scc0 label_000C', 's_nop 0', READ, 's_endpgm')),
    ('entry block re-entered by a loop', (MOVE, READ, 's_mov_b32 s2, vcc_lo', 's_cbranch_scc0 label_0000',
                                          's_endpgm')),
    ('exec write in the entry block', (MOVE, 's_and_saveexec_b32 s8, s2', READ, 's_branch label_0010',
                                       's_endpgm')),
    ('mask in the entry block', (COMPARE, MOVE, READ, 's_branch label_0010', 's_endpgm')),
    # A branch target ends the entry block even before any terminator: a later branch can come back to it.
    ('read in a loop whose header is in the entry block',
     (MOVE, READ, 'v_mov_b32_e32 v7, v0', 's_cbranch_scc0 label_0004', 's_endpgm')),
    ('read at a backward branch target', (MOVE, 's_nop 0', READ, 'v_mov_b32_e32 v7, v0', 's_branch label_0008',
                                          's_endpgm')),
    ('read after s_endpgm in a branch-free program', (MOVE, 's_endpgm', READ)),
]
# The entry block of a program with branches: every path reaches it through the same instructions, so a read
# there is proven like straight-line code, up to the instruction just before the first branch target.
for label, instructions in (('branch after read', (MOVE, READ, 's_branch label_000C', 's_endpgm')),
                            ('read before the first branch target',
                             (MOVE, READ, 's_nop 0', 's_cbranch_scc0 label_0008', 's_endpgm'))):
    got = run(*instructions)
    check(f'{label} accepted', got[:2] == ['v7 = s2;', 's9 = v7;'], repr(got))
got = run('v_cvt_f32_u32_e32 v7, s3', 'v_rcp_f32_e32 v7, v7', READ, 's_cbranch_scc0 label_0010', 's_endpgm')
check('chain in the entry block accepted', got[:3] == ['v7 = U(float(s3));', 'v7 = U(1.0 / F(v7));', 's9 = v7;'],
      repr(got))
# Past the entry block the read recomputes the definition instead of copying the VGPR
# (tests/startup_gfx2hlsl_readfirstlane_recompute_check.py covers that rule).
got = run(MOVE, 's_cbranch_scc0 label_0010', READ, 's_endpgm', 's_endpgm')
check('read after the first terminator recomputed', got[:2] == ['v7 = s2;', 's9 = s2;'], repr(got))
# An EXEC write that only narrows the mask (a compare into EXEC) leaves every lane that is still active a lane that
# ran the definition, so the read past it recomputes from the scalars. A write that can widen the mask still refuses.
got = run(MOVE, 's_cbranch_scc0 label_0014', 'v_cmpx_lt_f32_e32 v0, v1', READ, 's_endpgm', 's_endpgm')
check('read after a compare into EXEC recomputed', isinstance(got, list) and 's9 = s2;' in got, repr(got))
got = run(MOVE, 's_cbranch_scc0 label_0014', 's_or_b32 exec_lo, exec_lo, s3', READ, 's_endpgm', 's_endpgm')
check('read after an EXEC write that can widen refused', got == REFUSAL, repr(got))
original_narrows = G.Stage.__dict__['exec_write_only_narrows']
G.Stage.exec_write_only_narrows = staticmethod(lambda name, ops: False)
got = run(MOVE, 's_cbranch_scc0 label_0014', 'v_cmpx_lt_f32_e32 v0, v1', READ, 's_endpgm', 's_endpgm')
G.Stage.exec_write_only_narrows = original_narrows
check('mutant: without the narrowing rule the read after a compare into EXEC is refused', got == REFUSAL, repr(got))
# EXEC returned to a copy taken earlier: what was defined before the copy was run by every lane that comes back, what
# was defined under the narrowed mask was not.
RESTORE = ('v_mov_b32_e32 v7, s2', 's_mov_b32 s8, exec_lo', 's_cbranch_scc0 label_0020', 'v_cmpx_lt_f32_e32 v0, v1',
           'v_mov_b32_e32 v6, s3', 's_mov_b32 exec_lo, s8')
got = run(*RESTORE, 'v_readfirstlane_b32 s9, v7', 's_endpgm', 's_endpgm')
check('read of a value defined before the saved copy accepted after the restore',
      isinstance(got, list) and 's9 = s2;' in got, repr(got))
got = run(*RESTORE, 'v_readfirstlane_b32 s9, v6', 's_endpgm', 's_endpgm')
check('read of a value defined under the narrowed mask refused after the restore', got == REFUSAL, repr(got))
got = run(RESTORE[0], 's_mov_b32 s8, exec_lo', 's_mov_b32 s8, 0', 's_cbranch_scc0 label_001c', 'v_cmpx_lt_f32_e32 v0, v1',
          'v_mov_b32_e32 v6, s3', 's_mov_b32 exec_lo, s8', 'v_readfirstlane_b32 s9, v7', 's_endpgm', 's_endpgm')
check('restore from an overwritten copy refused', got == REFUSAL, repr(got))
got = run('v_mov_b32_e32 v7, s2', 's_cbranch_scc0 label_0014', 's_mov_b32 s8, exec_lo', 'v_cmpx_lt_f32_e32 v0, v1',
          's_mov_b32 exec_lo, s8', 'v_readfirstlane_b32 s9, v7', 's_endpgm', 's_endpgm')
check('restore from a copy that a branch can skip refused', got == REFUSAL, repr(got))
original_restore = G.Stage.exec_restore_copy
G.Stage.exec_restore_copy = lambda self, name, ops: None
got = run(*RESTORE, 'v_readfirstlane_b32 s9, v7', 's_endpgm', 's_endpgm')
G.Stage.exec_restore_copy = original_restore
check('mutant: without the restore rule the read after the restore is refused', got == REFUSAL, repr(got))
# These instructions refuse before v_readfirstlane_b32 is reached.
EARLIER = {
    'op_sel overwrite': 'modifier op_sel:[0,0]',
    # The high word of a copied wave64 mask carries the mask tag: its numeric read is refused where it is lowered.
    'overlapping 64-bit move high word': 'v_mov_b32 reads lane mask s4 as numeric data',
    'high word of saved exec': 'v_mov_b32 reads lane mask s3 as numeric data',
    'modified move overwrite': "v_mov_b32 with ['clamp']",
    'mask from uniform compare': 'v_mov_b32 data operand s2 is a lane mask',
    'vcc': 'v_mov_b32 data operand vcc_lo is a lane mask',
    'exec': 'v_mov_b32 data operand exec_lo is a lane mask',
    'compare into SGPR': 'v_mov_b32 data operand s2 is a lane mask',
    'saved exec': 'v_mov_b32 data operand s2 is a lane mask',
    'mask in the entry block': 'v_mov_b32 data operand s2 is a lane mask',
    'median of three of a lane mask': 'v_med3_f32 reads lane mask s2 as numeric data',
    'log2 of a lane mask': 'v_log_f32 reads lane mask s2 as numeric data',
    'ceil of a lane mask': 'v_ceil_f32 reads lane mask s2 as numeric data',
    'unsigned to float conversion of a mask': 'v_cvt_f32_u32 reads lane mask s2 as numeric data',
    # A scalar op that reads a lane mask as a number is refused where it is lowered (before the read).
    'aliased load first word': 's_buffer_load_dwordx2 reads lane mask s2 as numeric data',
    'aliased load second word': 's_buffer_load_dwordx2 reads lane mask s2 as numeric data',
    'varying scalar buffer offset': 's_buffer_load_dword reads lane mask s2 as numeric data',
    'mask through scalar arithmetic': 's_add_u32 reads lane mask s2 as numeric data',
    'mask through scalar condition': 's_cmp_eq_u32 reads lane mask vcc_lo as numeric data',
}
for label, instructions in REFUSED:
    got = run(*instructions)
    check(f'{label} refused', got == EARLIER.get(label, REFUSAL), repr(got))
# SCC written through m0 (#1931 review): a compare mask carried into m0 makes SCC varying, so the
# select that reads it, and a VGPR copied from that select, stay unproven; EXEC is left unchanged.
got = run('v_cmp_lt_f32_e32 vcc_lo, v0, v1', 's_and_b32 m0, vcc_lo, -1', 's_cselect_b32 s2, 1, 0',
          'v_mov_b32_e32 v7, s2', READ)
check('compare mask through m0 into SCC refused', got == REFUSAL, repr(got))
# The same chain from a uniform scalar still proves the copy uniform.
got = run('s_and_b32 m0, s2, 0xff', 's_cselect_b32 s3, 1, 0', 'v_mov_b32_e32 v7, s3', READ)
check('uniform m0 write into SCC accepted',
      got == ['m0 = s2 & 0x000000ffu;', 'scc = m0 != 0;', 's3 = scc ? 0x00000001u : 0x00000000u;',
              'v7 = s3;', 's9 = v7;'], repr(got))
# s_add_u32 and its siblings emit their SCC (carry, borrow, overflow, selected source) from the sources, so the select that
# reads it is uniform when the sources are.
got = run('s_add_u32 s3, s2, 1', 's_cselect_b32 s3, 1, 0', 'v_mov_b32_e32 v7, s3', READ)
check('select after a modelled carry from a uniform source accepted',
      got == ['scc = (s2 + 0x00000001u) < (s2);', 's3 = s2 + 0x00000001u;', 's3 = scc ? 0x00000001u : 0x00000000u;',
              'v7 = s3;', 's9 = v7;'], repr(got))
# Every other SCC write either updates the proof or invalidates it: s_wqm and the symbolic descriptor path, which emits nothing.
for label, first in (('whole quad mode', 's_wqm_b64 exec, exec'),
                     ('symbolic descriptor word', 's_and_b32 s10, s4, 0xff')):
    got = run(first, 's_cselect_b32 s3, 1, 0', 'v_mov_b32_e32 v7, s3', READ)
    check(f'select after {label} refused', got == REFUSAL, repr(got))
got = run('v_mov_b32_e32 v7, s2', READ, exec_known=False)
check('unknown exec refused', got == REFUSAL, repr(got))

# A compute load through a read-only descriptor with a uniform address is uniform; the value has no recomputable
# expression, so only the entry block reads it.
LOAD_OFF = 'buffer_load_dword v7, off, s[4:7], s2'
LOAD_OFFEN = 'buffer_load_dword v8, v7, s[4:7], 0 offen'
got = run(LOAD_OFF, READ, kind='cs')
check('uniform scalar offset load accepted', got[-2:] == ['v7 = r0.Load(a + 0);', 's9 = v7;'], repr(got))
got = run(MOVE, LOAD_OFFEN, 'v_readfirstlane_b32 s9, v8', kind='cs')
check('uniform vgpr offset load accepted', got[-1] == 's9 = v8;' and len(got) == 4, repr(got))
got = run(MOVE, 'buffer_load_dword v7, v7, s[4:7], 0 offen', READ, kind='cs')
check('load overwriting its uniform address accepted', got[-1] == 's9 = v7;', repr(got))
for label, instructions, kind, known in (
        ('load with a varying vgpr offset', ('buffer_load_dword v8, v0, s[4:7], 0 offen', 'v_readfirstlane_b32 s9, v8'), 'cs', True),
        ('load with a varying scalar offset', (COMPARE, 'buffer_load_dword v7, off, s[4:7], s2', READ), 'cs', True),
        ('load in a graphics stage', (LOAD_OFF, READ), 'ps', True),
        ('load without a stage kind', (LOAD_OFF, READ), None, True),
        ('load with unknown exec', (LOAD_OFF, READ), 'cs', False),
        ('load after the entry block', (LOAD_OFF, 's_cbranch_scc0 label_0010', READ, 's_endpgm', 's_endpgm'), 'cs', True),
        ('load then a varying overwrite', (LOAD_OFF, 'v_mov_b32_e32 v7, v0', READ), 'cs', True)):
    got = run(*instructions, kind=kind, exec_known=known)
    check(f'{label} refused', got == REFUSAL, repr(got))
original = G.UNIFORM_LOADS
G.UNIFORM_LOADS = frozenset()
try:
    got = run(LOAD_OFF, READ, kind='cs')
    check('control: load refused without tracking', got == REFUSAL, repr(got))
finally:
    G.UNIFORM_LOADS = original

# Every destination register of a multi-register load is uniform, and only the registers the opcode writes.
for label, instructions, read in (
        ('two-dword load, first word', ('buffer_load_dwordx2 v[7:8], off, s[4:7], s2',), 7),
        ('two-dword load, second word', ('buffer_load_dwordx2 v[7:8], off, s[4:7], s2',), 8),
        ('four-dword load, last word', ('buffer_load_dwordx4 v[6:9], off, s[4:7], s2',), 9),
        ('load over a varying register', ('v_mov_b32_e32 v8, v0', 'buffer_load_dwordx2 v[7:8], off, s[4:7], s2'), 8),
        ('formatted two-component load', ('buffer_load_format_xy v[7:8], off, s[4:7], s2',), 8)):
    got = run(*instructions, 'v_readfirstlane_b32 s9, v%d' % read, kind='cs')
    check(f'{label} accepted', got[-1] == 's9 = v%d;' % read, repr(got))
for label, instructions, read in (
        ('two-dword load into three registers', ('v_mov_b32_e32 v9, v0', 'buffer_load_dwordx2 v[7:9], off, s[4:7], s2'), 9),
        ('two-dword load into one register', ('v_mov_b32_e32 v8, v0', 'buffer_load_dwordx2 v7, off, s[4:7], s2'), 8),
        ('two-dword load with a varying offset', ('buffer_load_dwordx2 v[7:8], v0, s[4:7], 0 offen',), 8),
        ('other register after a load', ('v_mov_b32_e32 v6, v0', 'buffer_load_dwordx2 v[7:8], off, s[4:7], s2'), 6)):
    got = run(*instructions, 'v_readfirstlane_b32 s9, v%d' % read, kind='cs')
    check(f'{label} refused', got == REFUSAL, repr(got))

# A compute image_load through an unmodified fixed table descriptor at uniform coordinates is uniform when
# no image write exists in the dispatch; the entry-block read has no recomputable expression.
IMG_LOAD = 'image_load v7, [v2, v3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D'
IMG_COORDS = ('v_mov_b32_e32 v2, s2', 'v_mov_b32_e32 v3, 1')
for label, instructions, read in (
        ('2D single-channel image_load', IMG_COORDS + (IMG_LOAD,), 7),
        ('2D multi-channel image_load, second component',
         IMG_COORDS + ('image_load v[7:8], v[2:3], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D',), 8),
        ('2D packed a16 image_load',
         ('v_mov_b32_e32 v2, s2', 'image_load v7, v2, s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D a16'), 7),
        ('2D-array image_load',
         ('v_mov_b32_e32 v2, 1', 'v_mov_b32_e32 v3, 2', 'v_mov_b32_e32 v4, s2',
          'image_load v7, v[2:4], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY'), 7),
        ('image_load overwriting its uniform coordinate',
         IMG_COORDS + ('image_load v2, v[2:3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D',), 2)):
    got = run(*instructions, 'v_readfirstlane_b32 s9, v%d' % read, kind='cs')
    check(f'{label} accepted', isinstance(got, list) and got[-1] == 's9 = v%d;' % read, repr(got))

for label, instructions, read, kind, known in (
        ('image_load with varying x coordinate',
         ('v_mov_b32_e32 v3, 1', 'image_load v7, [v0, v3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D'), 7, 'cs', True),
        ('image_load with varying y coordinate',
         ('v_mov_b32_e32 v2, 1', 'image_load v7, [v2, v0], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D'), 7, 'cs', True),
        ('image_load with modified descriptor',
         ('s_or_b32 s17, s17, 0x1000',) + IMG_COORDS + (IMG_LOAD,), 7, 'cs', True),
        ('image_load with overwritten descriptor word',
         ('s_mov_b32 s18, 0',) + IMG_COORDS + (IMG_LOAD,), 7, 'cs', True),
        ('image_load in a dispatch with image_store',
         IMG_COORDS + (
             'v_mov_b32_e32 v10, 0', 'v_mov_b32_e32 v11, 0',
             'v_mov_b32_e32 v12, 0', 'v_mov_b32_e32 v13, 0',
             'v_mov_b32_e32 v14, 0', 'v_mov_b32_e32 v15, 0',
             IMG_LOAD, 'image_store v[12:15], v[10:11], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D'),
         7, 'cs', True),
        ('image_load_mip with uniform coordinates',
         IMG_COORDS + ('v_mov_b32_e32 v4, 0', 'image_load_mip v7, v[2:4], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D'),
         7, 'cs', True),
        ('image_load in a stage kind with no proof', IMG_COORDS + (IMG_LOAD,), 7, 'gs', True),
        ('image_load without a stage kind', IMG_COORDS + (IMG_LOAD,), 7, None, True),
        ('image_load with unknown exec', IMG_COORDS + (IMG_LOAD,), 7, 'cs', False),
        ('image_load read after a branch target',
         IMG_COORDS + (IMG_LOAD, 's_cbranch_scc0 label_0018', 's_endpgm', 's_endpgm'), 7, 'cs', True),
        ('other register after two-channel image_load',
         ('v_mov_b32_e32 v9, v0',) + IMG_COORDS +
         ('image_load v[7:8], v[2:3], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D',), 9, 'cs', True)):
    got = run(*instructions, 'v_readfirstlane_b32 s9, v%d' % read, kind=kind, exec_known=known)
    check(f'{label} refused', got == REFUSAL, repr(got))

original_img = G.UNIFORM_IMAGE_LOADS
G.UNIFORM_IMAGE_LOADS = frozenset()
try:
    got = run(*(IMG_COORDS + (IMG_LOAD, READ)), kind='cs')
    check('control: image_load refused without tracking', got == REFUSAL, repr(got))
finally:
    G.UNIFORM_IMAGE_LOADS = original_img

# tid.x >> n is 0 in every lane when the group's X size is at most 1 << n (the thread id VGPR is untouched).
WAVE = 'v_lshrrev_b32_e32 v1, 6, v0'
READ1 = 'v_readfirstlane_b32 s9, v1'
for size in (1, 33, 64):
    got = run(WAVE, READ1, kind='cs', group=(size, 1, 1))
    check(f'wave index of a group of {size} accepted', got[-1] == 's9 = v1;', repr(got))
got = run(WAVE, 's_cbranch_scc0 label_0010', READ1, 's_endpgm', 's_endpgm', kind='cs', group=(64, 1, 1))
check('wave index read past the entry block recomputed as zero', got[-1] == 's9 = 0u;', repr(got))
# v0 overwritten by a constant is no thread id: the shift of a uniform value is itself uniform (200 >> 6).
got = run('v_mov_b32_e32 v0, 200', WAVE, READ1, kind='cs', group=(64, 1, 1))
check('shift of an overwritten v0 is a uniform integer chain', got[-1] == 's9 = v1;' and 'v1 = v0 >> (0x00000006u & 31);' in got, repr(got))
for label, instructions, kind, group in (
        ('wave index of a group of 65', (WAVE, READ1), 'cs', (65, 1, 1)),
        ('wave index of a group of 128', (WAVE, READ1), 'cs', (128, 1, 1)),
        ('wave index of a group with no recorded size', (WAVE, READ1), 'cs', None),
        ('shift by 5 of a group of 64', ('v_lshrrev_b32_e32 v1, 5, v0', READ1), 'cs', (64, 1, 1)),
        ('wave index in a graphics stage', (WAVE, READ1), 'ps', (64, 1, 1)),
        ('wave index without a stage kind', (WAVE, READ1), None, (64, 1, 1)),
        ('wave index of v1', ('v_lshrrev_b32_e32 v1, 6, v1', READ1), 'cs', (64, 1, 1)),
        ('wave index after v0 is loaded', ('buffer_load_dword v[0:1], off, s[4:7], s2', WAVE, READ1), 'cs', (64, 1, 1)),
        ('wave index after v0 is swapped', ('v_swap_b32 v2, v0', WAVE, READ1), 'cs', (64, 1, 1)),
        ('wave index with an output modifier', ('v_lshrrev_b32_e64 v1, 6, v0 clamp', READ1), 'cs', (64, 1, 1)),
        ('wave index of a register shift', ('v_lshrrev_b32_e32 v1, v2, v0', READ1), 'cs', (64, 1, 1))):
    got = run(*instructions, kind=kind, group=group)
    check(f'{label} refused', got == ("v_lshrrev_b32 with ['clamp']" if 'modifier' in label else REFUSAL), repr(got))

# The global thread id of a 64-wide group, (uniform << 6) + tid.x, shifted back by 6 is the uniform part.
GID = 'v_lshl_add_u32 v0, s2, 6, v0'
got = run(GID, WAVE, READ1, kind='cs', group=(64, 1, 1))
check('global thread id shifted back accepted', got[-1] == 's9 = v1;', repr(got))
got = run(GID, WAVE, 's_cbranch_scc0 label_0014', READ1, 's_endpgm', 's_endpgm', kind='cs', group=(64, 1, 1))
check('thread id base shifted back recomputed past the entry block', got[-1] == 's9 = ((s2 << 6u) >> 6u);', repr(got))
got = run(GID, WAVE, 's_mov_b32 s2, 5', 's_cbranch_scc0 label_0018', READ1, 's_endpgm', 's_endpgm', kind='cs', group=(64, 1, 1))
check('thread id base shifted back refused past the entry block after the scalar changed', got == REFUSAL, repr(got))
for label, instructions, group in (
        ('thread id base of a group of 128', (GID, WAVE, READ1), (128, 1, 1)),
        ('thread id base shifted by more than the scale', ('v_lshl_add_u32 v0, s2, 6, v0', 'v_lshrrev_b32_e32 v1, 7, v0', READ1), (64, 1, 1)),
        ('thread id base from a varying scalar', (COMPARE, 'v_lshl_add_u32 v0, s2, 6, v0', WAVE, READ1), (64, 1, 1)),
        ('thread id base with the scalar rewritten', (GID, 's_mov_b32 s2, vcc_lo', WAVE, READ1), (64, 1, 1)),
        ('thread id base from a vector', ('v_lshl_add_u32 v0, v5, 6, v0', WAVE, READ1), (64, 1, 1)),
        ('thread id base added to another vector', ('v_lshl_add_u32 v0, s2, 6, v4', WAVE, READ1), (64, 1, 1)),
        ('thread id base added to a rewritten v0', ('v_mov_b32_e32 v0, 3', GID, WAVE, READ1), (64, 1, 1)),
        ('thread id base rewritten after', (GID, 'v_add_nc_u32_e32 v0, s3, v0', WAVE, READ1), (64, 1, 1)),
        ('thread id base scaled by 32', ('v_lshl_add_u32 v0, s2, 5, v0', WAVE, READ1), (64, 1, 1))):
    got = run(*instructions, kind='cs', group=group)
    # The compare's mask in s2 is refused where v_lshl_add_u32 reads it as a number, before the read.
    expected = ('v_lshl_add_u32 reads lane mask s2 as numeric data'
                if label == 'thread id base from a varying scalar' else REFUSAL)
    check(f'{label} refused', got == expected, repr(got))

# Only bits that survive both the field operation and the final shift may vary across threads.
AND_OR = 'v_and_or_b32 v2, v0, 3, s2'
EXTRACT = 'v_bfe_u32 v2, v0, 2, 3'
SHIFT = 'v_lshrrev_b32_e32 v1, 2, v2'
SHIFT1 = 'v_lshrrev_b32_e32 v1, 1, v2'
for label, op, shift, size, expression in (
        ('masked low bits of a large group', AND_OR, SHIFT, 128, '(s2 >> 2u)'),
        ('mask above a non-power-of-two group', 'v_and_or_b32 v2, v0, 8, s2', SHIFT1, 7, '(s2 >> 1u)'),
        ('extracted bit cleared by shift', EXTRACT, SHIFT1, 8, '0u'),
        ('empty extracted field', 'v_bfe_u32 v2, v0, 0, 0', SHIFT1, 64, '0u'),
        ('narrow field of a large group', 'v_bfe_u32 v2, v0, 1, 2', SHIFT, 128, '0u')):
    got = run(op, shift, READ1, kind='cs', group=(size, 1, 1))
    check(f'{label} accepted at entry', got[-1] == 's9 = v1;', repr(got))
    got = run(op, shift, 's_cbranch_scc0 label_0014', READ1, 's_endpgm', 's_endpgm', kind='cs', group=(size, 1, 1))
    check(f'{label} recomputed past entry', got[-1] == f's9 = {expression};', repr(got))
for label, instructions, size in (
        ('surviving masked bit', (AND_OR, SHIFT1, READ1), 4),
        ('surviving extracted bit', (EXTRACT, SHIFT1, READ1), 9),
        ('variable mask', ('v_and_or_b32 v2, v0, s2, s2', SHIFT, READ1), 4),
        ('vector OR source', ('v_and_or_b32 v2, v0, 3, v1', SHIFT, READ1), 4),
        ('predicate as OR source', (COMPARE, AND_OR, SHIFT, READ1), 4),
        ('scalar overwritten before shift', (AND_OR, 's_mov_b32 s2, 5', SHIFT, READ1), 4),
        ('scalar overwritten before recompute', (AND_OR, SHIFT, 's_mov_b32 s2, 5', 's_cbranch_scc0 label_0018',
                                                 READ1, 's_endpgm', 's_endpgm'), 4),
        ('overwritten thread id', ('v_swap_b32 v1, v0', EXTRACT, SHIFT1, READ1), 8),
        ('overwritten field', (EXTRACT, 'v_mov_b32_e32 v2, v0', SHIFT1, READ1), 8),
        ('register field offset', ('v_bfe_u32 v2, v0, s2, 3', SHIFT1, READ1), 8),
        ('register field width', ('v_bfe_u32 v2, v0, 2, s2', SHIFT1, READ1), 8),
        ('signed field', ('v_bfe_i32 v2, v0, 2, 3', SHIFT1, READ1), 8),
        ('unknown group for a masked field', (AND_OR, SHIFT, READ1), None)):
    got = run(*instructions, kind='cs', group=(size, 1, 1) if size else None)
    # a lane mask as the OR source meets the mask rule (#2702) before the uniformity rule: either refusal keeps it refused
    mask_first = label == 'predicate as OR source' and got == 'v_and_or_b32 reads lane mask s2 as numeric data'
    check(f'{label} refused', got == REFUSAL or mask_first, repr(got))

# Older probes may provide either no tracker or only part of the tracking state.
for present in ((), ('uniform_v',), ('uniform_v', 'varying_s')):
    s = G.Stage.__new__(G.Stage)
    s.sym, s.attrs, s.lines = {}, {}, []
    s.used_s, s.used_v = set(), set()
    s.has_branches, s.exec_known = False, True
    for name in present:
        setattr(s, name, set())
    s.one('v_mov_b32_e32', MOVE)
    check('partial tracker %r proves nothing' % (present,), not getattr(s, 'uniform_v', ()))
    s.one('v_interp_p1_f32_e32', 'v_interp_p1_f32_e32 v7, v0, attr0.x')
    try:
        s.one('v_readfirstlane_b32', READ)
    except G.Unsupported as error:
        got = str(error)
    else:
        got = 'accepted'
    check('partial tracker %r refuses read' % (present,), got == REFUSAL, got)

# Remove the allowlist and require the same successful instruction sequences to refuse.
original = G.UNIFORM_VALU
G.UNIFORM_VALU = frozenset()
try:
    for label, instructions, _, *source in ACCEPTED:
        reg = source[0] if source else 7
        got = run(*instructions, 'v_readfirstlane_b32 s9, v%d' % reg)
        check(f'control: {label} refused without tracking', got == REFUSAL, repr(got))
finally:
    G.UNIFORM_VALU = original

# A uniform VGPR used as both operands doubles its recomputation text each time: the text stays bounded (a chain of
# 22 doublings would otherwise grow to megabytes per instruction). The entry-block read copies the VGPR, so it needs no
# expression; only a read past the entry block recomputes, and past the limit that chain has none.
DOUBLE = 'v_add_nc_u32_e32 v7, v7, v7'
got = run(MOVE, *[DOUBLE] * 3, READ, kind='cs', group=(64, 1, 1))
check('short doubling chain stays readable', isinstance(got, list) and got[-1] == 's9 = v7;', repr(got)[:200])
stage = G.Stage.__new__(G.Stage)
stage.pipe, stage.lines, stage.attrs = Pipe(), [], {}
stage.pipe.group = (64, 1, 1)
stage.sym = {4 + i: ('desc', ('table', 0), 0, i) for i in range(4)}
stage.sym[3] = ('fixed_buffer_word',)
stage.used_s, stage.used_v, stage.masks, stage.exec_known, stage.kind = set(), set(), set(), True, 'cs'
stage.ins = [(4 * i, 4, a.split()[0], a) for i, a in enumerate([MOVE] + [DOUBLE] * 22)]
stage.translate()
longest = max(map(len, stage.uniform_expr.values()), default=0)
check('long doubling chain keeps a bounded recompute expression', longest <= 2 * G.UNIFORM_EXPR_LIMIT, str(longest))
got = run(MOVE, *[DOUBLE] * 22, READ, kind='cs', group=(64, 1, 1))
check('entry-block read of a long doubling chain copies the VGPR', isinstance(got, list) and got[-1] == 's9 = v7;', repr(got)[:200])

# A pixel stage reading the first lane of a level-zero sample of a fixed table texture: coordinates built only from
# scalars, a sampler held in scalars that the program set, no image store. Every lane then reads the same texel.
SAMPLER = ('s_mov_b32 s8, 0x92', 's_mov_b32 s9, 0', 's_mov_b32 s10, 0', 's_mov_b32 s11, 0x14500000')
SAMPLE = 'image_sample_lz v7, v7, s[16:23], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D'
TEXEL = 'image_load v8, v6, s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm a16'
for label, instructions in (
        ('level-zero sample of a uniform coordinate', (MOVE, *SAMPLER, SAMPLE, 'v_med3_f32 v7, v7, s2, s3')),
        ('arithmetic on a uniform texel feeding a sample',
         ('v_mov_b32_e32 v6, 0', TEXEL, 'v_max_f32_e32 v8, 0x358637bd, v8', 'v_subrev_f32_e32 v7, s2, v8', *SAMPLER,
          SAMPLE, 'v_min_f32_e32 v7, s3, v7')),
        ('float add of two uniform values', (MOVE, 'v_add_f32_e32 v7, s3, v7'))):
    for kind in ('ps',):
        got = run(*instructions, READ, kind=kind)
        check(f'{label} accepted in a {kind} stage', isinstance(got, list) and got[-1] == 's9 = v7;', repr(got)[:200])
for label, instructions in (
        ('sample of a per-lane coordinate', (*SAMPLER, 'image_sample_lz v7, v0, s[16:23], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D')),
        ('sample with a sampler loaded as a descriptor', (MOVE, 'image_sample_lz v7, v7, s[16:23], s[4:7] dmask:0x1 dim:SQ_RSRC_IMG_1D')),
        ('implicit-level sample', (MOVE, *SAMPLER, 'image_sample v7, v7, s[16:23], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D')),
        ('sample with a bias', (MOVE, 'v_mov_b32_e32 v8, s2', *SAMPLER,
                                'image_sample_b v7, [v8, v7], s[16:23], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D')),
        ('texel at a per-lane coordinate', ('image_load v7, v0, s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm a16',))):
    got = run(*instructions, READ, kind='ps')
    check(f'{label} refused in a pixel stage', got == REFUSAL or (isinstance(got, str) and got != ''), repr(got)[:200])
got = run(MOVE, *SAMPLER, SAMPLE, READ, kind='ps')
check('control: a read of a sample result is accepted only with the proof', isinstance(got, list), repr(got)[:200])
original_sample = G.Stage.uniform_image_sample_regs
G.Stage.uniform_image_sample_regs = lambda self, base, ops, mods: ()
got = run(MOVE, *SAMPLER, SAMPLE, READ, kind='ps')
G.Stage.uniform_image_sample_regs = original_sample
check('mutant: without the sample proof the first-lane read is refused', got == REFUSAL, repr(got)[:200])

check_translation()
check_whole_quad_translation()
print('Readfirstlane uniformity checks: %d failed' % len(failures))
sys.exit(1 if failures else 0)
