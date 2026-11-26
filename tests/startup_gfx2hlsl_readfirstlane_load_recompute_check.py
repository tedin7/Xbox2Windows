"""Check v_readfirstlane_b32 of a uniform compute buffer load past the entry block, without shader machine code.

A lane the load skipped keeps an older VGPR, so every lane loads the words again into a temporary from an address
every lane computes alike (the uniform expression of the address VGPR, or its copy while EXEC is still the entry
mask), and the read copies that temporary. A branch target, an EXEC write or an overwrite between the load and
the read still refuses, as do a per-lane address, a graphics stage and a destination of the wrong width.
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
    resources = {(('table', 0), 0): {'index': 0}}  # the formatted load reads the stride and FORMAT words by index

    def __init__(self):
        self.modified, self.other_modified, self.uavs = set(), set(), {}

    def resource(self, key, kind):
        return 'r0'

    def uav(self, key, kind='raw'):
        return 'w0'

    def stored_images(self):
        return self.uavs

    def ud(self, key):
        return 0


def run(*instructions, kind='cs', descriptor_proven=False):
    """Translate disassembly text through the same dispatch and block split as a real stage."""
    s = G.Stage.__new__(G.Stage)
    if descriptor_proven:
        # The past-entry descriptor proof of a formatted load is checked in startup_gfx2hlsl_compute_check.py.
        s.formatted_descriptor_from_block = lambda reg, first: True
    s.pipe, s.lines, s.attrs = Pipe(), [], {}
    s.sym = {4 + i: ('desc', ('table', 0), 0, i) for i in range(4)}
    s.sym.update({16 + i: ('desc', ('table', 0), 32, i) for i in range(8)})
    s.sym.update({24 + i: ('desc', ('table', 0), 64, i) for i in range(8)})
    s.used_s, s.used_v, s.masks = set(), set(), set()
    s.exec_known, s.kind = True, kind
    s.ins = [(4 * i, 4, asm.split()[0], asm) for i, asm in enumerate(instructions)]
    try:
        s.translate()
    except G.Unsupported as error:
        return str(error)
    return [line for _, lines, _ in s.blocks for line in lines]


REFUSAL = 'v_readfirstlane_b32 requires a proven uniform source'
READ = 'v_readfirstlane_b32 s9, v7'
# Offset 0x8 is a join: the branch at 0x0 skips the instruction at 0x4, so everything after it is past the entry.
JOIN = ('s_cbranch_scc0 label_0008', 's_nop 0')
SCALAR = 's_buffer_load_dword s3, s[4:7], 0x10'  # a uniform scalar written after the join
LOAD = 'buffer_load_dword v7, off, s[4:7], s3'    # at 0xc after JOIN + (SCALAR,)
NARROW = 's_and_saveexec_b32 s8, s2'

# (label, instructions, the lines from the load on)
ACCEPTED = [
    ('scalar offset load', JOIN + (SCALAR, LOAD, READ),
     ['a = s3 + 0;', 'v7 = r0.Load(a + 0);', 'ulc_7 = r0.Load(a + 0);', 's9 = ulc_7;']),
    # The temporary already holds the words, so a later write of the offset scalar does not matter.
    ('offset scalar rewritten after the load', JOIN + (SCALAR, LOAD, 's_mov_b32 s3, 0', READ),
     ['a = s3 + 0;', 'v7 = r0.Load(a + 0);', 'ulc_7 = r0.Load(a + 0);', 's3 = 0x00000000u;', 's9 = ulc_7;']),
    # While EXEC is the entry mask every lane ran the address definition, so its copy serves even after the
    # scalar it read is rewritten.
    ('address copy under the entry exec',
     JOIN + (SCALAR, 'v_mov_b32_e32 v6, s3', 's_mov_b32 s3, 0', 'buffer_load_dword v7, v6, s[4:7], 0 offen', READ),
     ['a = v6 + 0;', 'v7 = r0.Load(a + 0);', 'ul14_7 = r0.Load(a + 0);', 's9 = ul14_7;']),
    ('second word of a two-dword load',
     JOIN + (SCALAR, 'buffer_load_dwordx2 v[7:8], off, s[4:7], s3', 'v_readfirstlane_b32 s9, v8'),
     ['a = s3 + 0;', 'v7 = r0.Load(a + 0);', 'ulc_7 = r0.Load(a + 0);', 'v8 = r0.Load(a + 4);',
      'ulc_8 = r0.Load(a + 4);', 's9 = ulc_8;']),
    ('conversion of the loaded word',
     JOIN + (SCALAR, LOAD, 'v_cvt_f32_u32_e32 v8, v7', 'v_readfirstlane_b32 s9, v8'),
     ['a = s3 + 0;', 'v7 = r0.Load(a + 0);', 'ulc_7 = r0.Load(a + 0);', 'v8 = U(float(v7));',
      's9 = U(float((ulc_7)));']),
]
for label, instructions, lines in ACCEPTED:
    got = run(*instructions)
    check(f'{label} accepted', isinstance(got, list) and got[-len(lines):] == lines, repr(got))

# A formatted load keeps the selected component; the address and the selection are the ones the guarded write uses.
got = run(*JOIN, SCALAR, 'buffer_load_format_x v7, off, s[4:7], s3', READ, descriptor_proven=True)
pick = '((ud0 >> 0) & 7u) == 4u ? c0 : ((ud0 >> 0) & 7u) == 5u ? c1 : ((ud0 >> 0) & 7u) == 6u ? c2 : ' \
       '((ud0 >> 0) & 7u) == 7u ? c3 : ((ud0 >> 0) & 7u) == 1u ? one : 0u'
check('formatted load accepted', isinstance(got, list) and 'a = s3 + 0;' in got and
      got[-3:] == ['ulc_7 = %s;' % pick, '}', 's9 = ulc_7;'] and got[-4] == 'v7 = %s;' % pick, repr(got))
# The temporary is declared once at function scope, next to the registers.
s = G.Stage.__new__(G.Stage)
s.uniform_load_temps, s.used_s, s.used_v, s.kind, s.pipe = {'ulc_7'}, set(), set(), 'cs', Pipe()
s.pipe.packed_position_vgpr = None
check('temporary declared', '    uint ulc_7 = 0;' in s.declarations(), repr(s.declarations()))

REFUSED = [
    ('loop carries a narrowed EXEC into the load', JOIN + (SCALAR, LOAD, READ, 's_add_u32 s3, s3, 4',
     'v_cmp_lt_f32_e32 vcc_lo, v0, v1', 's_and_b32 exec_lo, exec_lo, vcc_lo', 's_cmp_lt_u32 s3, 64',
     's_cbranch_scc1 label_000c', 's_endpgm')),
    ('EXEC write outside the textual back edge', JOIN + (SCALAR, LOAD, READ, 's_branch label_0020',
     's_cbranch_scc1 label_000c', 's_endpgm', NARROW, 's_branch label_0018')),
    # Two definitions reach the read at a join.
    ('join of a load and another definition', ('s_cbranch_scc0 label_0014', SCALAR, LOAD, 's_branch label_0014',
                                               'v_mov_b32_e32 v7, 0', READ, 's_endpgm')),
    ('per-lane address', JOIN + ('buffer_load_dword v7, v0, s[4:7], 0 offen', READ)),
    ('per-lane offset scalar', JOIN + ('v_cmp_lt_f32_e64 s3, v0, v1', LOAD, READ)),
    # A loop head between the load and the read: a later iteration reaches the read without the load.
    ('back-edge target between load and read', JOIN + (SCALAR, LOAD, 's_nop 0', READ, 's_cbranch_scc0 label_0014',
                                                       's_endpgm')),
    # The load runs inside a block lanes with clear EXEC skip, and the read is past the block's join.
    ('load inside a block one lane skips', JOIN + (SCALAR, 's_and_saveexec_b32 s8, vcc_lo', 's_cbranch_execz label_0018',
                                                   LOAD, 's_mov_b32 exec_lo, s8', READ, 's_endpgm')),
    ('EXEC narrowed between load and read', JOIN + (SCALAR, LOAD, NARROW, READ)),
    # On empty EXEC, RDNA 2 reads the old lane-0 VGPR, not an unguarded reload.
    ('empty EXEC keeps the old lane-zero VGPR', JOIN + ('v_mov_b32_e32 v7, 17', 's_mov_b32 exec_lo, 0',
                                                        SCALAR, LOAD, READ)),
    ('address expression with potentially empty EXEC',
     JOIN + (NARROW, SCALAR, 'v_mov_b32_e32 v7, s3', 'buffer_load_dword v7, v7, s[4:7], 0 offen', READ)),
    # Under a narrowed EXEC a lane that skipped the address definition holds an older copy, and the expression
    # is gone once its scalar is rewritten.
    ('stale address copy under a narrowed exec',
     JOIN + (NARROW, SCALAR, 'v_mov_b32_e32 v6, s3', 's_mov_b32 s3, 0', 'buffer_load_dword v7, v6, s[4:7], 0 offen',
             READ)),
    ('loaded register overwritten before the read', JOIN + (SCALAR, LOAD, 'v_mov_b32_e32 v7, v0', READ)),
    ('two-dword load into three registers', JOIN + (SCALAR, 'v_mov_b32_e32 v9, v0',
                                                    'buffer_load_dwordx2 v[7:9], off, s[4:7], s3',
                                                    'v_readfirstlane_b32 s9, v9')),
]
for label, instructions in REFUSED:
    got = run(*instructions)
    check(f'{label} refused', got == REFUSAL, repr(got))
got = run(*JOIN, SCALAR, LOAD, READ, 's_add_u32 s3, s3, 4', 's_cmp_lt_u32 s3, 64',
          's_cbranch_scc1 label_000c', NARROW, 's_endpgm')
check('loop with unchanged EXEC accepts the load', isinstance(got, list) and 's9 = ulc_7;' in got, repr(got))
got = run(*JOIN, SCALAR, LOAD, READ, kind='ps')
check('graphics stage refused', got == REFUSAL, repr(got))

# Without the load allowlist the accepted programs refuse again.
original = G.UNIFORM_LOADS
G.UNIFORM_LOADS = frozenset()
try:
    for label, instructions, _ in ACCEPTED:
        got = run(*instructions)
        check(f'control: {label} refused without the load proof', got == REFUSAL, repr(got))
finally:
    G.UNIFORM_LOADS = original

IMG_LOAD = 'image_load v7, [v2, v3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D'
IMG_COORDS = ('v_mov_b32_e32 v2, s3', 'v_mov_b32_e32 v3, 1')
IMG_ACCEPTED = [
    ('image_load past entry block', JOIN + (SCALAR,) + IMG_COORDS + (IMG_LOAD, READ),
     ['t = r0.Load(int3(v2, v3, 0));', 'v7 = U(t.x);', 'ul14_7 = U(t.x);', 's9 = ul14_7;']),
    ('two-channel image_load past entry block converted',
     JOIN + (SCALAR,) + IMG_COORDS + (
         'image_load v[7:8], v[2:3], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D',
         'v_cvt_f32_u32_e32 v9, v8', 'v_readfirstlane_b32 s9, v9'),
     ['t = r0.Load(int3(v2, v3, 0));', 'v7 = U(t.x);', 'ul14_7 = U(t.x);',
      'v8 = U(t.y);', 'ul14_8 = U(t.y);', 'v9 = U(float(v8));', 's9 = U(float((ul14_8)));']),
]
for label, instructions, lines in IMG_ACCEPTED:
    got = run(*instructions)
    check(f'{label} accepted', isinstance(got, list) and got[-len(lines):] == lines, repr(got))

for label, instructions in (
        ('image_load past entry block with image_store in dispatch',
         JOIN + (SCALAR,) + IMG_COORDS + (
             'v_mov_b32_e32 v10, 0', 'v_mov_b32_e32 v11, 0',
             'v_mov_b32_e32 v12, 0', 'v_mov_b32_e32 v13, 0',
             'v_mov_b32_e32 v14, 0', 'v_mov_b32_e32 v15, 0',
             IMG_LOAD, 'image_store v[12:15], v[10:11], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D', READ)),
        ('image_load in EXEC-modifying loop',
         JOIN + (SCALAR,) + IMG_COORDS + (
             IMG_LOAD, READ, 's_add_u32 s3, s3, 4',
             'v_cmp_lt_f32_e32 vcc_lo, v0, v1', 's_and_b32 exec_lo, exec_lo, vcc_lo',
             's_cmp_lt_u32 s3, 64', 's_cbranch_scc1 label_000c', 's_endpgm')),
        ('image_load inside a skipped execz block',
         JOIN + (SCALAR,) + IMG_COORDS + (
             's_and_saveexec_b32 s8, vcc_lo', 's_cbranch_execz label_0020',
             IMG_LOAD, 's_mov_b32 exec_lo, s8', READ, 's_endpgm')),
        ('join of image_load and another definition',
         ('s_cbranch_scc0 label_001c', SCALAR) + IMG_COORDS + (
             IMG_LOAD, 's_branch label_001c', 'v_mov_b32_e32 v7, 0', READ, 's_endpgm'))):
    got = run(*instructions)
    check(f'{label} refused', got == REFUSAL, repr(got))

original_img = G.UNIFORM_IMAGE_LOADS
G.UNIFORM_IMAGE_LOADS = frozenset()
try:
    for label, instructions, _ in IMG_ACCEPTED:
        got = run(*instructions)
        check(f'control: {label} refused without image_load proof', got == REFUSAL, repr(got))
finally:
    G.UNIFORM_IMAGE_LOADS = original_img

print('Readfirstlane load recompute checks: %d failed' % len(failures))
sys.exit(1 if failures else 0)
