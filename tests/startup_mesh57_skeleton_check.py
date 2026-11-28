"""Slot 57 mesh requests of one group-memory shape translate in tools/startup/gfx2hlsl.py (#2413).

usage: startup_mesh57_skeleton_check.py
       startup_mesh57_skeleton_check.py binary <llvm-mc>
       startup_mesh57_skeleton_check.py draw <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py lanemask <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py wave_collectives <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py depth <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py killexit <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py formatted <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py shift <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py group <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py heap <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py exchange <dxc> <offscreen_check> <warp|hardware> <publisher>

The fixture (tests/startup_mesh57_skeleton.s, source-built) holds the accepted shape: lane 0 of wave 0 stores a
word and both waves pass a barrier; later the same lane stores the primitive and vertex counts, both waves pass
a barrier, wave 0 sends one allocation request and wave 1 reads the counts back. Each thread then exports the
position and one parameter of vertex `thread` and the packed indices of primitive `thread`, each under its count.
The data of a thread is the buffer slot of its low four bits.

Without arguments: the fixture translates through the slot 57 request path, and each mutation of the shape is
refused with its own message. `binary` compares the embedded words with llvm-mc. `draw` compiles the request
path's output (ms_6_5, ps_6_0), draws it and compares the read-back target with a CPU rasteriser written here;
a reference with one wrong pixel must fail.
`depth` compiles the pair with a pixel stage that exports only to the null target (a depth-only pass), draws it
twice into a depth buffer cleared to the sentinel 1.0 with depth test LESS, and compares colour (left cleared) and
depth with the CPU rasteriser; a reference without the depth writes, and the same mesh with a null-export stage
that does not discard, must each fail.
`killexit` draws the colour pair whose pixel stage kills lanes with red at or below 0.5 and leaves through the
early-exit null export (entered by s_cbranch_execz, and in the demote form by s_cbranch_scc0 after s_cselect),
and compares both with the CPU rasteriser without the killed triangles; the same mesh with the pixel stage that
does not kill must differ at exactly the killed pixels.
The heap mode compares a root-selected raw heap buffer with the CPU image and a slot-zero shader mutant.
The exchange mode uses the runtime-published request and the same pixel comparison.
"""
import inspect
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

MESH = (
    'bfa00003 befe0481 be84047e 8706ff03 0f000000 bf850007 2c0a0a87 7e0c0280 d8340008 00000506'
    ' bfa3ffe3 bf8a0000 bf820004 bf8a0000 7e0c0280 d8d80008 05000006 befe04c1 9386ff03 00040018'
    ' 8f078606 d7650000 00000ec1 d7660000 000200c1 bf8cc07f 7e280505 f4080209 fa000000 be950310'
    ' be960311 bf8a0000 bf008006 bf840010 befe0404 bf880007 7e020215 7e040280 7e060216 d8340000'
    ' 00000102 d8340004 00000302 befe04c1 bfa3ffe3 bf8a0000 8f048c15 887c1604 bf900009 bf82000d'
    ' bf8a0000 befe0404 bf880009 7e040280 d8d80000 01000002 d8d80004 03000002 bf8cc17f 7e2a0501'
    ' bf8cc07f 7e2c0503 befe04c1 7d280016 3608008f 34080884 e0381000 80021004 e03810c0 80021404'
    ' bf8c3f70 f80008cf 13121110 f800020f 17161514 befe04c1 7d280015 3608008f 34080882 e0301180'
    ' 80021804 bf8c3f70 f8000941 00000018 bf810000').split()
# Select one pair of index words per mesh group; the exchange leaves the group index in s20.
_group_at = MESH.index('e0301180')
GROUP_MESH = MESH[:_group_at] + [
    '8f1e8314',  # s_lshl_b32 s30, s20, 3
    'd70f6a04', '0002081e',  # v_add_co_u32 v4, vcc, s30, v4
] + MESH[_group_at:]
# Keep the draw's primitive count fixed while s16 supplies the independent bound.
GROUP_JOIN_BASE = [word if word != 'be950310' else 'be950382' for word in GROUP_MESH]
_group_join_at = GROUP_JOIN_BASE.index('8f1e8314')
GROUP_JOIN = GROUP_JOIN_BASE[:_group_join_at] + [
    'bf0a1014',  # s_cmp_lt_u32 s20, s16
    'bf840001',  # s_cbranch_scc0 over the multiply
    '9314fd14',  # s_mul_i32 s20, s20, src_scc
] + GROUP_JOIN_BASE[_group_join_at:]
GROUP_CLAMP = [word if word != 'bf840001' else 'bf850001' for word in GROUP_JOIN]
GROUP_CONTROL = [word if word != '7e280505' else '7e280507' for word in GROUP_MESH]

# A root-selected heap record is loaded and used inside a uniform branch arm.
HEAP_SETUP = [
    'be980314',  # s_mov_b32 s24, s20
    'be930380',  # s_mov_b32 s19, 0
    'be9a0315',  # s_mov_b32 s26, s21
    'be9b0380',  # s_mov_b32 s27, 0
    '8718ff18', '000003ff',  # s_and_b32 s24, s24, 0x3ff
    '8f188518',  # s_lshl_b32 s24, s24, 5
]
HEAP_ARM = [
    'bf008080',  # s_cmp_eq_i32 0, 0
    'bf840004',  # s_cbranch_scc0 past the two loads
    'f408070d', '30000000',  # s_load_dwordx4 s[28:31], s[26:27], s24
    'e03810c0', '80071404',  # buffer_load_dwordx4 v[20:23], v4, s[28:31], 0 offen offset:192
]
HEAP_MESH = MESH.copy()
_heap_at = HEAP_MESH.index('e03810c0')
del HEAP_MESH[_heap_at:_heap_at + 2]  # colour comes from the pre-exchange heap read
HEAP_MESH[1:1] = HEAP_SETUP + [
    '7e080280',  # v_mov_b32 v4, 0 (the first colour)
    '7e280280', '7e2a0280', '7e2c0280', '7e2e0280',  # initialise v20..v23 for the skipped arm
] + HEAP_ARM + ['bf800000']  # s_nop 0, the join before the exchange


def heap_root():
    params = [(1, 0, 60), (0, 0, 72), (1, 0, 80)]
    rts = (struct.pack('<6I', 2, 3, 24, 0, 0, 0x400)
           + b''.join(struct.pack('<3I', *p) for p in params)
           + struct.pack('<3I', 0, 0, 2) + struct.pack('<2I', 0, 0) + struct.pack('<3I', 0, 1, 2))
    return b'DXBC' + bytes(16) + struct.pack('<4I', 1, 44 + len(rts), 1, 36) + b'RTS0' + struct.pack('<I', len(rts)) + rts


PIXEL_ASM = ('v_interp_mov_f32 v0, p0, attr0.x\nv_interp_mov_f32 v1, p0, attr0.y\nv_interp_mov_f32 v2, p0, attr0.z\n'
             'v_interp_mov_f32 v3, p0, attr0.w\nexp mrt0 v0, v1, v2, v3 done vm\ns_endpgm\n')
PIXEL = 'c8020002 c8060102 c80a0202 c80e0302 f800180f 03020100 bf810000'.split()
# A depth-only pixel stage: no colour output, only the null-target export (done, vm, no channel). Lanes whose flat
# attribute 0 red is not above 0.5 leave EXEC, so the vm export kills them and they write no depth.
DEPTH_PIXEL_ASM = ('v_interp_mov_f32 v0, p0, attr0.x\nv_cmpx_lt_f32_e32 0.5, v0\n'
                   'exp null off, off, off, off done vm\ns_endpgm\n')
DEPTH_PIXEL = 'c8020002 7c2200f0 f8001890 00000000 bf810000'.split()
NULL_EXPORT = DEPTH_PIXEL.index('f8001890')
# The drawn stage with exp mrt0 v0, v0, v0, v0 inserted before its null export (after the v_cmpx), then the drawn
# stage reading attribute 1, which the mesh does not export. The other mixed forms (mrtz, param0, pos0, both orders,
# no EXEC narrowing) are covered by startup_mesh57_depth_only_check.py.
# A colour stage with the early exit of a kill (LLVM SILateBranchLowering, Mesa ACO): when no lane is left after
# v_cmpx, s_cbranch_execz jumps past the main s_endpgm to a block that is only the null export and s_endpgm.
# Lanes whose flat attribute 0 red is not above 0.5 are killed; the others write their colour.
KILL_EXIT_PIXEL_ASM = ('v_interp_mov_f32 v0, p0, attr0.x\nv_interp_mov_f32 v1, p0, attr0.y\n'
                       'v_interp_mov_f32 v2, p0, attr0.z\nv_interp_mov_f32 v3, p0, attr0.w\n'
                       'v_cmpx_lt_f32_e32 0.5, v0\ns_cbranch_execz .exit\nexp mrt0 v0, v1, v2, v3 done vm\ns_endpgm\n'
                       '.exit:\nexp null off, off, off, off done vm\ns_endpgm\n')
KILL_EXIT_PIXEL = ('c8020002 c8060102 c80a0202 c80e0302 7c2200f0 bf880003 f800180f 03020100 bf810000'
                   ' f8001890 00000000 bf810000').split()
# The demote form of the same exit (LLVM SILateBranchLowering): the live mask leaves SCC, s_cselect sets EXEC to 0
# when no lane is live, and s_cbranch_scc0 then enters the exit with EXEC zero. Same kill as KILL_EXIT_PIXEL.
DEMOTE_EXIT_PIXEL_ASM = ('v_interp_mov_f32 v0, p0, attr0.x\nv_interp_mov_f32 v1, p0, attr0.y\n'
                         'v_interp_mov_f32 v2, p0, attr0.z\nv_interp_mov_f32 v3, p0, attr0.w\n'
                         's_mov_b64 s[12:13], exec\nv_cmp_gt_f32_e32 vcc, 0.5, v0\ns_andn2_b64 s[12:13], s[12:13], vcc\n'
                         's_cselect_b64 exec, exec, 0\ns_cbranch_scc0 .exit\ns_mov_b64 exec, s[12:13]\n'
                         'exp mrt0 v0, v1, v2, v3 done vm\ns_endpgm\n'
                         '.exit:\nexp null off, off, off, off done vm\ns_endpgm\n')
DEMOTE_EXIT_PIXEL = ('c8020002 c8060102 c80a0202 c80e0302 be8c047e 7c0800f0 8a8c6a0c 85fe807e bf840004 befe040c'
                     ' f800180f 03020100 bf810000 f8001890 00000000 bf810000').split()
# A depth export (exp mrtz v1 done vm, v1 = 0.5) behind the same kill: lanes whose flat attribute 0 red is not above 0.5 leave
# EXEC, s_cbranch_execz jumps to the early-exit null export (which discards), the others export depth 0.5.
MRTZ_KILL_EXIT_PIXEL_ASM = ('v_interp_mov_f32 v0, p0, attr0.x\nv_cmpx_lt_f32_e32 0.5, v0\ns_cbranch_execz .exit\n'
                            'v_mov_b32_e32 v1, 0.5\nexp mrtz v1, off, off, off done vm\ns_endpgm\n'
                            '.exit:\nexp null off, off, off, off done vm\ns_endpgm\n')
MRTZ_KILL_EXIT_PIXEL = 'c8020002 7c2200f0 bf880004 7e0202f0 f8001881 00000001 bf810000 f8001890 00000000 bf810000'.split()
# The same depth export with no kill (straight-line, accepted before): every covered pixel gets depth 0.5.
MRTZ_PLAIN_PIXEL = '7e0202f0 f8001881 00000001 bf810000'.split()
DEPTH_PIXEL_MUTANTS = (
    ('a colour write next to the drawn null export', DEPTH_PIXEL[:NULL_EXPORT] + ['f800000f', '00000000'] + DEPTH_PIXEL[NULL_EXPORT:],
     'slot 57 pixel stage mixes the null export with another export'),
    ('a depth-only pixel stage reading an attribute the mesh does not export', ['c8020402'] + DEPTH_PIXEL[1:],
     'pixel stage reads attributes [1] the vertex stage does not export'),
)

# Overwrite red after the raw colour load; green, blue and alpha still come from its other three words.
COLOUR_LOAD = 'buffer_load_dwordx4 v[20:23], v4, s[8:11], 0 offen offset:192'
FORMAT_LOAD = 'buffer_load_format_x v20, v4, s[8:11], 0 offen offset:192'
AFTER_COLOUR_LOAD = MESH.index('e03810c0') + 2
FORMATTED_MESH = MESH[:AFTER_COLOUR_LOAD] + ['e00010c0', '80021404'] + MESH[AFTER_COLOUR_LOAD:]
REWRITTEN_FORMAT = FORMATTED_MESH[:AFTER_COLOUR_LOAD] + ['be8b0380'] + FORMATTED_MESH[AFTER_COLOUR_LOAD:]  # s_mov_b32 s11, 0


def code(words):
    return b''.join(struct.pack('<I', int(word, 16)) for word in words)


def root():
    # Parameter 0: two root constants (primitive count, vertex count). Parameter 1: a descriptor table.
    rts0 = (struct.pack('<6I', 2, 2, 24, 0, 0, 0) + struct.pack('<3I', 1, 0, 48) + struct.pack('<3I', 0, 0, 60)
            + struct.pack('<3I', 0, 0, 2) + struct.pack('<2I', 0, 0))
    return b'DXBC' + bytes(16) + struct.pack('<4I', 1, 44 + len(rts0), 1, 36) + b'RTS0' + struct.pack('<I', len(rts0)) + rts0


def mutate(old, new, nth=0, *, mesh=MESH):
    """The fixture with the nth word equal to old replaced."""
    at = [i for i, word in enumerate(mesh) if word == old][nth]
    return mesh[:at] + [new] + mesh[at + 1:]


def branch(op, source, target):
    """Encode a SOPP branch using dword indices, relative to the next instruction."""
    return '%08x' % (op << 16 | ((target - source - 1) & 0xffff))


def vertex_red(words):
    """Run words after the count exchange and use the resulting v7 as vertex red."""
    wait = MESH.index('bf8c3f70')
    mesh = MESH[:wait + 1] + ['7e280307'] + MESH[wait + 1:]  # v_mov_b32 v20, v7
    at = mesh.index('7d280016')
    return mesh[:at] + words.split() + mesh[at:]


MASKED_IF_ELSE = '7e0e02f0 7d880086 be8c246a bf880002 7e0e02f2 bf820001 bf800000 8afe7e0c 7e0e0280 befe04c1'
# The same if/else with s_add_u32 s50, s21, 1 in the then arm. No path reads s50 after the join.
MASKED_DEAD_SCALAR = '7e0e02f0 7d880086 be8c246a bf880003 7e0e02f2 80328115 bf820001 bf800000 8afe7e0c 7e0e0280 befe04c1'


def masked_vertex_red(words=MASKED_IF_ELSE):
    mesh = vertex_red(words)
    at = mesh.index('f80008cf')
    return mesh[:at] + ['bf880004'] + mesh[at:]  # s_cbranch_execz skips both vertex exports


# The first six threads select the shift for their entire guest wave. The second wave keeps 64/255.
SHIFT_ASM = """s_mov_b32 s40, 0x3e808081
v_cmp_gt_u32_e32 vcc, 6, v0
s_and_saveexec_b64 s[12:13], vcc
s_cbranch_execz 2
s_lshl_b32 s40, 0x1fc00000, 1
s_mov_b64 exec, -1
v_mov_b32 v7, s40"""
SHIFT_WORDS = 'bea803ff 3e808081 7d880086 be8c246a bf880002 8f2881ff 1fc00000 befe04c1 7e0e0228'
SHIFT_MESH = vertex_red(SHIFT_WORDS)


# s_mov_b32 s40, 0.5; v_cmp_ge_f32_e64 vcc_lo, s40, 1.0
VCC_COMPARE = 'bea803f0 d406006a 0001e428'


BRANCH_CASES = (
    # v_mov v7, 0; s_cmp_eq_i32 0, <constant>; s_cbranch_scc1 .join; v_mov v7, 1.0
    ('SCC branch taken', vertex_red('7e0e0280 bf008080 bf850001 7e0e02f2'), lambda lane: 0),
    ('SCC branch not taken', vertex_red('7e0e0280 bf008180 bf850001 7e0e02f2'), lambda lane: 255),
    # v_mov v7, 0; set s40 and VCC as above; s_cbranch_vccz .join; v_mov v7, 1.0
    ('VCC-zero branch taken', vertex_red('7e0e0280 ' + VCC_COMPARE + ' bf860001 7e0e02f2'), lambda lane: 0),
    # s_mov_b32 s40, 1.0 changes the compare result to true.
    ('VCC-zero branch not taken', vertex_red('7e0e0280 bea803f2 d406006a 0001e428 bf860001 7e0e02f2'),
     lambda lane: 255),
    # The same false compare falls through for s_cbranch_vccnz.
    ('VCC-nonzero branch not taken', vertex_red('7e0e0280 ' + VCC_COMPARE + ' bf870001 7e0e02f2'),
     lambda lane: 255),
    # s_mov_b32 s40, 1.0 makes s_cbranch_vccnz take .join.
    ('VCC-nonzero branch taken', vertex_red('7e0e0280 bea803f2 d406006a 0001e428 bf870001 7e0e02f2'),
     lambda lane: 0),
    # s_cmp_eq_i32 0, 0; s_cbranch_scc1 .join skips a second uniform compare.
    # .join: s_cbranch_vccz .done skips v_mov v7, 1.0 using the earlier compare.
    ('VCC-zero branch after a join',
     vertex_red('7e0e0280 ' + VCC_COMPARE + ' bf008080 bf850002 d406006a 0001e028 bf860001 7e0e02f2'),
     lambda lane: 0),
    # v7 starts at 0.5. The then arm writes 1.0 for tid < 6, the else arm writes 0.0 for the remaining lanes.
    # v_cmp_gt_u32 vcc, 6, v0; s_and_saveexec_b64 s[12:13], vcc; s_cbranch_execz .empty
    # v_mov v7, 1.0; s_branch .else; .empty: s_nop; .else: s_andn2_b64 exec, s[12:13], exec
    # v_mov v7, 0; s_mov_b64 exec, -1
    ('lane-masked if/else', masked_vertex_red(), lambda lane: 255 if lane < 6 else 0),
    ('lane-masked if/else with a dead scalar write', masked_vertex_red(MASKED_DEAD_SCALAR),
     lambda lane: 255 if lane < 6 else 0),
)

COUNT = [i for i, word in enumerate(MESH) if word == 'bf8a0000'][2]
POST_COUNT = MESH.index('7d280016')
GROUP_END = MESH.index('05000006') + 1
PRIMITIVE_EXPORT = MESH.index('f8000941')

WAVE_LOAD_ASM = """s_mov_b32 s40, 0x3e808081
s_lshl_b32 s41, s6, 8
s_mov_b32 s42, 6
v_mov_b32 v7, 0
s_mov_b64 exec, -1
v_cmpx_gt_u32_e32 s42, v0
s_mov_b32 s42, 0
s_cbranch_execz 4
s_buffer_load_dword s40, s[8:11], s41 offset:0x1e0
v_mov_b32 v7, 1.0
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, -1"""
WAVE_LOAD_PREFIX = 'bea803ff 3e808081 8f298806 beaa0386 7e0e0280 befe04c1 7da8002a beaa0380'
WAVE_LOAD_BODY = 'f4200a04 520001e0 7e0e02f2 bf8cc07f'


def wave_load_mesh(prefix=WAVE_LOAD_PREFIX, body=WAVE_LOAD_BODY, *, exports='7e280228 7e2a0307'):
    """Insert the scalar load before the count exchange and export its value as vertex red."""
    body = body.split()
    before = MESH.index('be950310')
    wait = MESH.index('bf8c3f70')
    return (MESH[:before] + prefix.split() + [branch(0xbf88, 0, len(body) + 1)] + body + ['befe04c1']
            + MESH[before:wait + 1] + exports.split() + MESH[wait + 1:])


# Assembled in binary() from the synthetic instructions below. Branch distances
# are counted in dwords, including VOP3 and literal words.
MASK_PREFIX = WAVE_LOAD_PREFIX.replace('8f298806', '8f298206').replace('beaa0386', 'beaa03ff 00000080')
MASK_FRONT = 'bf068006'                       # s_cmp_eq_u32 s6, 0
MASK_RIGHT = 'd4c6002e 00010100'              # v_cmp_ge_u32_e64 s[46:47], v0, 0
MASK_LEFT = '7d860080 d4c6002e 00010d00 87ae2e6a'
MASK_SAVE = 'beae082e beae242e'               # not and aliased and_saveexec
MASK_LOAD = 'f4080c09 fa000000 bf8cc07f f4200a18 520001e0'  # table dwordx4, then scalar buffer load
MASK_ELSE = '8afe7e2e 7e0e02ff 3e800000'      # complement, v_mov_b32 v7, 0.25


def lane_mask_mesh(*, prefix=MASK_PREFIX, front=MASK_FRONT, left=MASK_LEFT, right=MASK_RIGHT,
                   setup='', load=MASK_LOAD, value='7e0e0228', otherwise=MASK_ELSE, exports='7e280228 7e2a0307'):
    left, right = left.split(), right.split()
    then = (load + ' bf8cc07f ' + value).split()
    body = (front.split() + [branch(0xbf84, 0, len(left) + 2)] + left
            + [branch(0xbf82, 0, len(right) + 1)] + right + setup.split() + MASK_SAVE.split()
            + [branch(0xbf88, 0, len(then) + 1)] + then + otherwise.split())
    return wave_load_mesh(prefix, ' '.join(body), exports=exports)


MASK_CASES = (
    ('S1', lane_mask_mesh()),
    ('S1b', lane_mask_mesh(front=MASK_FRONT + ' bf0681fd',
                          left='beb4047e ' + MASK_LEFT + ' 87ae342e')),
    ('S3', lane_mask_mesh(right='d4c1002e 00010100',
                          setup='bf068006 85b4c180 88ae342e')),
    ('S3 mask arms', lane_mask_mesh(setup='bf068006 85b480c1 85b62e34 87ae362e',
                                    value='d5010007 00d9e480')),
    ('S4 EXEC moves', lane_mask_mesh(left='beb4047e 7da80084 befe0434 87fe7e34 ' + MASK_LEFT
                                    + ' 87b67e6a 88ea366a 87ae2e6a')),
    ('S4 CMPX', lane_mask_mesh(setup='7da80088')),
    ('S4 VCC vote', lane_mask_mesh(left=MASK_LEFT + ' 7d880081 87ea7e6a bf870002 '
                                      'd4c6002e 00010100')),
    ('S4 closure', lane_mask_mesh(otherwise=MASK_ELSE + ' 88fe2e7e')),
)

COHERENT_THEN = 'f4200a04 520001e0 bf8cc07f 7e0e0228'  # s40 = buffer[480 + 4*wave], v7 = s40
COHERENT_ELSE = 'f4200f04 520001e8 bf8cc07f 7e0e023c'  # s60 = buffer[488 + 4*wave], v7 = s60
COHERENT_OFFSET_SETUP = 'f4200984 520001f0 bf8cc07f 8726c426'
COHERENT_OFFSET_ADD = '802eff26 000001e0'
COHERENT_OFFSET_LOAD = 'f4200b84 5c000000 bf8cc07f bea8032e 7e0e0228'
COHERENT_OFFSET_MASK = 'd4c4002e 000200ff 00000080'  # 128 > tid; true for all active lanes here
COHERENT_OFFSET_USE = '87fe2e7e'  # exec &= s[46:47]
COHERENT_DESCRIPTOR_LOAD = 'f4080e09 fa000000 bf8cc07f'  # s_load_dwordx4 s[56:59], s[18:19], 0
COHERENT_DESCRIPTOR_LOW = 'bee80438'  # s_mov_b64 s[104:105], s[56:57]
COHERENT_DESCRIPTOR_HIGH = 'beea043a'  # s_mov_b64 vcc, s[58:59]
COHERENT_DESCRIPTOR_MASK = 'd4c4003a 000200ff 00000080'  # v_cmp_gt_u32_e64 s58, 128, v0
# Base field 52 names s[104:107]. The upper pair is VCC.
COHERENT_DESCRIPTOR_READ = 'f4200bb4 5c000000 bf8cc07f bea8032e 7e0e0228'


def coherent_words(parts):
    """Resolve SOPP displacements in the synthetic wave64 words."""
    words, labels, fixups = [], {}, {}
    for part in parts:
        if isinstance(part, tuple):
            fixups[len(words)] = part
            words.append(None)
        elif part.endswith(':'):
            labels[part[:-1]] = len(words)
        else:
            words.extend(part.split())
    for at, (opcode, label) in fixups.items():
        words[at] = branch(opcode, at, labels[label])
    return ' '.join(words)


COHERENT_VCC_READ = 'd501003c 01a90280'  # v_cndmask_b32_e64 v60, 0, 1, vcc
COHERENT_EXEC_WRITE = 'bf068006 857ec180'  # s_cmp_eq_u32 s6, 0; s_cselect_b32 exec_lo, 0, -1
COHERENT_VCC_DRAW = 'S5 VCC read before compare'


SCC_COMPARE = '7d840081'  # v_cmp_eq_u32_e32 vcc, 1, v0
SCC_AND = {'SCC move': '87967e6a', 'SCC branch': '878c7e6a', 'SCC delayed select': '87927e6a'}
SCC_MOVE = 'be9403fd 9396ff14 00010000'  # s_mov_b32 s20, src_scc; s_bfe_u32 s22, s20, 0x10000
SCC_VALU = '7e9c0280'  # v_mov_b32 v78, 0
SCC_SELECT = '85148081'  # s_cselect_b32 s20, 1, 0
SCC_RED = '7e280c14'  # v_cvt_f32_u32_e32 v20, s20


def coherent_mesh(shape, *, full=False, control=None, positions=False, nonzero=False, live_mask=False,
                  numeric_pair=False, vcc_before_compare=False):
    # Wave 0 has six then lanes. Wave 1 has zero or 64. The enclosing prefix covers both waves.
    full = full or control in ('conditional address', 'mixed VCC source')  # exercise both SCC paths
    threshold = 'bf068006 ' + ('8532ff86 00000080' if full else '8532c086')  # s50 = SCC ? 6 : 128/64
    before = ['beb4047e bf8cc07f', 'head:', '7da80032']  # save EXEC, wait, CMPX s50 > v0
    then = [COHERENT_THEN]
    at_else = ['8afe7e34', (0xbf88, 'join')]  # exec = s[52:53] & ~exec; skip empty else
    otherwise = [COHERENT_ELSE]
    join = ['80288028', 'f4200bc4 520001e0 bf8cc07f', 'befe0434']
    if shape in SCC_AND:
        before.append(SCC_COMPARE)
        if control == 'SCC writer skipped':
            before.append((0xbf88, 'SCC reader'))
        before.append(SCC_AND[shape])
        if control == 'SCC overwritten on one path':
            before.extend(((0xbf88, 'SCC reader'), '803a8106'))  # s_add_u32 s58, s6, 1
        before.append('SCC reader:')
        if shape == 'SCC move':
            before.append(SCC_MOVE)
        elif shape == 'SCC branch':
            before.extend(((0xbf84, 'SCC zero'), '80148114', 'SCC zero:'))  # s_add_u32 s20, s20, 1
        else:
            # Both edges carry the AND's SCC to the select, including the empty-wave skip.
            before.extend(((0xbf88, 'SCC select'), SCC_VALU, 'SCC select:', SCC_SELECT))
    if shape == 'S9':
        then = [COHERENT_OFFSET_MASK, COHERENT_OFFSET_SETUP]
        if control == 'conditional address':
            then += ['bf068006', (0xbf84, 'offset join')]
        then.append(COHERENT_OFFSET_ADD)
        if control == 'conditional address':
            then.append('offset join:')
        if control == 'partial mask':
            then.append(COHERENT_OFFSET_USE)
        if control in ('conditional address', 'partial mask'):
            then.append(COHERENT_OFFSET_LOAD)
        else:
            then.append(COHERENT_DESCRIPTOR_LOAD)
            if control == 'mixed VCC source':
                then += ['bf068006', (0xbf84, 'descriptor join'), COHERENT_DESCRIPTOR_MASK,
                         'descriptor join:']
            elif control == 'compare VCC source':
                then.append(COHERENT_DESCRIPTOR_MASK)
            then += ['beea047e' if control == 'EXEC VCC source' else COHERENT_DESCRIPTOR_HIGH,
                     COHERENT_DESCRIPTOR_LOW, COHERENT_DESCRIPTOR_READ]
        then += [COHERENT_OFFSET_MASK, COHERENT_OFFSET_USE]
    if control == 'backward':
        then.append((0xbf82, 'head'))
    elif control == 'constant-trip':
        # s58 = 3, then add 2 while the counter is below 9.
        then.extend(('beba0383', 'loop:', '803a823a bf0a893a', (0xbf85, 'loop')))
    elif control == 'EXEC xor':
        then.insert(0, '89fe347e')
    elif control == 'EXEC half':
        then.insert(0, 'befe0380')
    elif control == 'mask SCC':
        before.extend(('7d880032 beb6246a', (0xbf84, 'join')))  # SCC from AND-saveexec
    elif control == 'varying address':
        before.append('bebe047e')  # save the narrowed EXEC in s[62:63]
        then[0] = COHERENT_THEN.replace('520001e0', '7c0001e0')  # use its low word as a numeric offset
    elif control == 'varying SCC':
        then[:0] = ['7d880032 bf06806a', (0xbf84, 'join')]  # numeric compare reads VCC_LO
    if positions:
        before.insert(1, '87fe7e34 87fe347e')  # both operand orders of EXEC &= saved EXEC
        then.append(GUARDED_POSITION_LOAD)
        if control != 'missing position':
            otherwise.append(GUARDED_POSITION_LOAD)

    if shape == 'S6':
        # Narrow three times to one shared restore, then run a second chain on the opposite lanes.
        body = ['8533ff88 00000080 beb4047e 7d880033 beae046a 87fe7e2e', (0xbf88, 'restore'),
                '7d880032 beb6246a', (0xbf88, 'restore'), 'beb8246a', (0xbf88, 'restore'),
                *then, 'restore:', 'befe0434 7d860032 beb6246a', (0xbf88, 'done'),
                'beb8246a', (0xbf88, 'done'), COHERENT_ELSE, 'done:', 'befe0434']
    elif shape == 'S7':
        # Keep VCC from before the last narrowing so its vote must also count inactive lanes.
        numeric = '7e78026a' if control == 'numeric VCC' else ''
        body = ['beb4047e 7d880032 beb6246a', (0xbf88, 'outer restore'),
                'beb8246a', (0xbf89 if nonzero else 0xbf88, 'inner compare'), '7e0e02f2', 'inner compare:',
                '7da80080', (0xbf88, 'inner restore'), '7e0e02f2',
                'inner restore:', 'befe0438 87ea7e38',  # VCC = saved s[56:57] & EXEC
                '7da80080', numeric, (0xbf86, 'outer restore'), (0xbf87 if nonzero else 0xbf86, 'load'),
                '7e0e02f2', 'load:', COHERENT_THEN,
                'outer restore:', 'befe0436', 'befe0434']
    else:
        if shape == 'S8':
            before = ['7d880032 beb4246a']
            # The uniform switch can enter either inner if/else. Their saves have separate lifetimes.
            then = ['bf068006', (0xbf84, 'other case'),
                    'beb6047e 7da80083', (0xbf88, 'then complement'), COHERENT_THEN,
                    'then complement:', '8afe7e36', (0xbf88, 'then restore'), COHERENT_ELSE,
                    'then restore:', 'befe0436', (0xbf82, 'switch join'),
                    'other case:', 'beb8047e 7da80084', (0xbf88, 'case complement'), COHERENT_THEN,
                    'case complement:', '8afe7e38', (0xbf88, 'case restore'), COHERENT_ELSE,
                    'case restore:', 'befe0438', 'switch join:']
            otherwise = ['beba047e 7da8008c', (0xbf88, 'else complement'), COHERENT_THEN,
                         'else complement:', '8afe7e3a', (0xbf88, 'else restore'), COHERENT_ELSE,
                         'else restore:', 'befe043a']
        body = [*before, (0xbf88, 'else'), *then, 'else:', *at_else, *otherwise, 'join:', *join]
    prefix = MASK_PREFIX
    if vcc_before_compare:
        before_compare = COHERENT_VCC_READ
        if control == 'EXEC write before compare':
            before_compare += ' ' + COHERENT_EXEC_WRITE
        if control == 'entry after reset':
            prefix = prefix.replace('befe04c1 ', 'bf840001 befe04c1 ')  # s_cbranch_scc0 skips the reset
        prefix = prefix.replace('befe04c1 ', 'befe04c1 ' + before_compare + ' ')
    if positions:
        prefix = prefix.replace('beaa03ff 00000080', 'be960311').replace(' beaa0380', '')
        prefix = prefix.replace('befe04c1 7da8002a', 'befe04c1 3608008f 34080884 7da80016')
    exports = '7e280228 7e2a0307'
    if shape in SCC_AND:
        prefix += ' 87148006'  # s_and_b32 s20, s6, 0 (including on the outer skip path)
        exports = SCC_RED + ' 7e2a0307'
    if live_mask:
        prefix += ' bebe0480'  # s[62:63] = 0 on the outer skip
        body.insert(-1, 'bebe047e')  # save the else EXEC, including zero when the else was skipped
        exports = '7e280228 d5010015 00fa0e80'  # v_cndmask_b32_e64 v21, 0, v7, s[62:63]
    setup = []
    if numeric_pair:
        # A 64-bit zero written as scalar data can feed a wave-uniform SCC branch.
        setup = ['bebe0480 bf06803e', (0xbf84, 'threshold'), 'bf800000', 'threshold:']
    mesh = wave_load_mesh(prefix, coherent_words([*setup, threshold, *body]), exports=exports)
    if positions:
        mesh.pop(mesh.index('be960311', mesh.index('be950310')))  # count is already set by the prefix
        mesh = mutate('13121110', '35343332', mesh=mesh)
    if control == 'outside entry':
        at = mesh.index('befe04c1', mesh.index('beaa03ff'))  # full reset immediately before the prefix
        target = mesh.index('beb4047e', at)
        mesh.insert(at, branch(0xbf84, at, target + 1))
    return mesh


COHERENT_DRAWS = {'S5': ('S5', {}), 'S6': ('S6', {}),
                  COHERENT_VCC_DRAW: ('S5', {'vcc_before_compare': True}),
                  'S9': ('S9', {}),
                  **{name: (name, {}) for name in SCC_AND}}
COHERENT_CONTROLS = (
    ('S5', 'backward', 'a backward branch inside the block'),
    ('S5', 'constant-trip', 'a backward branch inside the block'),
    ('S5', 'outside entry', 'a branch from outside lands inside the block'),
    ('S5', 'EXEC xor', 'an unsupported EXEC write or half access (s_xor_b64)'),
    ('S5', 'EXEC half', 'an unsupported EXEC write or half access (s_mov_b32)'),
    ('S7', 'numeric VCC', 'VCC vote does not read a pure lane mask'),
    ('S5', 'mask SCC', 'SCC of a mask operation is read'),
    ('SCC move', 'SCC overwritten on one path', 'SCC of a mask operation is read'),
    ('SCC move', 'SCC writer skipped', 'SCC of a mask operation is read'),
    ('S5', 'varying address', 'a scalar load address is not wave-uniform'),
    ('S9', 'conditional address', 'a scalar load address is not wave-uniform'),
    ('S9', 'partial mask', 'an unsupported vector instruction (v_cmp_gt_u32_e64)'),
    ('S9', 'compare VCC source', 'a scalar load address is not wave-uniform'),
    ('S9', 'EXEC VCC source', 'a scalar load address is not wave-uniform'),
    ('S9', 'mixed VCC source', 'a scalar load address is not wave-uniform'),
    ('S5', 'varying SCC', 'SCC is not proven wave-uniform'),
)

MASK_MEMORY = (
    ('tbuffer', wave_load_mesh(body='f4200a04 520001e0 e8b001e8 80020700 bf8c007f')),
    ('image', wave_load_mesh(body='f4200a04 520001e0 f40c0e09 fa000020 bf8cc07f '
                            '7e640280 7e660280 f0000108 000e0732 bf8c0f70')),
)

# Post-count, scalar-load-free skips use the same proof as the compute phi.
MASK_GENERIC = (
    ('generic live', vertex_red('beae0480 7d880086 be8c246a bf880002 d4c4002e 00020086 '
                                'befe04c1 d5010007 00b9e480')),
    ('generic dead', vertex_red('7e0e0280 7d880086 be8c246a bf880003 d4c4002e 00020086 '
                                '7e0e02f2 befe04c1')),
)

MASK_CONTROLS = (
    ('c1 numeric read', lane_mask_mesh(setup='beb6102e'), 'a lane mask is also read as a number'),
    ('c2 unproven base', lane_mask_mesh(load=MASK_LOAD.replace('f4080c09', 'f4080c1b')),
     'a masked arm reads a scalar that is not wave-uniform'),
    ('c3 undefined skip value', lane_mask_mesh(prefix=' '.join(MASK_PREFIX.split()[2:])),
     'a scalar live after the arm is undefined or non-uniform on the skip path'),
    ('c4 saved mask live', lane_mask_mesh(exports='beb6042e 7e280228 7e2a0307'),
     'a saved EXEC mask is live after the if/else'),
    ('c5 mask SCC', lane_mask_mesh(setup='87ae2e7e 85b4c180'),
     'SCC of a mask operation is read'),
    ('undefined VALU on restored lanes',
     lane_mask_mesh(left='beb4047e 7da80084 7e6402f2 befe0434 7e0e0332 ' + MASK_LEFT),
     'a vector input is undefined on the active lanes'),
)

COHERENT_LEGACY_CASES = (
    ('c6 enclosing lane skip', lane_mask_mesh(front='bf880001 bf800000 ' + MASK_FRONT), 2),
    ('c7 scalar else', lane_mask_mesh(otherwise=MASK_ELSE + ' f4200d84 fa0001e0'), 1),
)


# This pair holds ordinary data before the compare and again after the saved
# EXEC's last use. Both numeric reads have their own reaching definitions.
WEB_REUSE = lane_mask_mesh(prefix=MASK_PREFIX + ' beae0484 7e66022e',
                           exports='beae0488 7e66022f 7e280228 7e2a0307')
WEB_NUMERIC_EDGE = lane_mask_mesh(right='beae0389')
WEB_NUMERIC_READ = lane_mask_mesh(setup='7e66022f')
STANDALONE_SELECT = 'bf068006 85ea80c1 d5010007 01a9e480'
STANDALONE_MESH = vertex_red(STANDALONE_SELECT)
STANDALONE_WAVE_MESH = wave_load_mesh(body=WAVE_LOAD_BODY + ' ' + STANDALONE_SELECT)


def mask_region(head, then, tail):
    """Encode an execz to the complement after the then-arm words."""
    body = then.split()
    return ' '.join(head.split() + [branch(0xbf88, 0, len(body) + 1)] + body + tail.split())


def multiple_regions(nested, renarrow=False):
    first_head = 'd4c40034 00020086 beb62434'
    second = mask_region('d4c40038 00020083 beba2438', '7e0e02f2', '8afe7e3a 7e0e02ff 3e800000 88fe3a7e')
    first_then = MASK_LOAD + ' bf8cc07f 7e0e02f2'
    if nested:
        first_then += ' ' + second
    close = '87fe3c36' if renarrow else '88fe367e'  # from saved block EXEC, optionally narrowed by s[60:61]
    first = mask_region(first_head, first_then, '8afe7e36 7e0e02ff 3e800000 ' + close)
    if nested:
        return wave_load_mesh(MASK_PREFIX, first)
    before = 'd4c4003c 000200c0 ' if renarrow else ''  # s[60:61] = thread < 64
    return wave_load_mesh(MASK_PREFIX, before + first + ' ' + second.rsplit(' ', 1)[0])


ROUND2_ACCEPTED = (
    ('reused numeric pair', WEB_REUSE),
    ('standalone wave-load select', STANDALONE_WAVE_MESH),
    ('sequential mask regions', multiple_regions(False)),
    ('nested mask regions', multiple_regions(True)),
    ('re-narrowed mask regions', multiple_regions(False, renarrow=True)),
)
ROUND2_DRAWS = ROUND2_ACCEPTED + (
    ('standalone select', STANDALONE_MESH),
    ('standalone carry select', vertex_red('bf068006 85ea80c1 7e660280 d5283c32 01aa66c1 d5010007 00f1e480')),
)
ROUND2_REFUSED = (
    ('numeric definition reaches the joined mask', WEB_NUMERIC_EDGE, 'a lane mask is also read as a number'),
    ('live mask high word read as data', WEB_NUMERIC_READ, 'a lane mask is also read as a number'),
    ('standalone VCC read as a number', vertex_red(STANDALONE_SELECT + ' 7e66026b'),
     's_cselect_b64 on a wave mask is not covered'),
    ('standalone select with unverified SCC', vertex_red(STANDALONE_SELECT.split(' ', 1)[1]),
     's_cselect_b64 on a wave mask is not covered'),
)


# Both arms load each lane's position, including the high lanes of the first
# guest wave. These VGPRs have no definition outside the prefix guard.
GUARDED_POSITION_SETUP = (
    ('select', 'beaa0311 bf068010 85162a80'),
    ('same', 'be960311'),
    ('unsigned export', 'beaa0311 bf068010 85162a80'),
    ('plus one', 'beaa0311 8016812a'),
    ('different SGPR', 'beaa0310 be960311'),
    ('rewritten bound', 'beaa0311 bf068010 85162a80'),
    ('one arm', 'beaa0311 bf068010 85162a80'),
    ('wave index', 'beaa0306 be96032a'),
)
GUARDED_POSITION_LOAD = 'e0381000 80023204 bf8c3f70'


def guarded_position_mesh(case, setup):
    prefix = MASK_PREFIX.replace('beaa03ff 00000080', setup).replace(' beaa0380', '')
    prefix = prefix.replace('befe04c1', 'befe04c1 3608008f 34080884')  # v4 = (thread & 15) * 16
    if case == 'same':
        prefix = prefix.replace('7da8002a', '7da80016')
    otherwise = MASK_ELSE if case == 'one arm' else MASK_ELSE + ' ' + GUARDED_POSITION_LOAD
    mesh = lane_mask_mesh(prefix=prefix, value=GUARDED_POSITION_LOAD, otherwise=otherwise, exports='')
    count_copy = mesh.index('be960311', mesh.index('be950310'))
    mesh.pop(count_copy)
    if case == 'rewritten bound':
        mesh[count_copy:count_copy] = ['beaa0384', 'be96032a']  # V reads the replacement B, s42 = 4
    if case == 'unsigned export':
        mesh = mutate('7d280016', '7da80016', mesh=mesh)
    return mutate('13121110', '35343332', mesh=mesh)  # position uses v[50:53]


GUARDED_POSITIONS = {case: guarded_position_mesh(case, setup) for case, setup in GUARDED_POSITION_SETUP}
GUARDED_REFUSAL = ('unsupported: slot 57 mesh: v50 is read where it is not defined for every lane '
                   '(an entry value, or left different per wave by an exchange)')


def count_guard_mesh(case='select'):
    """Two paths to a scalar buffer offset, then separately guarded positions and primitive indices."""
    before = MESH.index('be950310')
    prefix = 'bf0c8010'.split()                    # s_bitcmp0_b32 s16, 0
    scalar = ['8f298411']                         # s_lshl_b32 s41, s17, 4 (s17 is the supplied group index)
    vector = '7e020211 e00021c0 80020101 bf8c3f70 7e520501'.split()
    # v_mov v1, s17; buffer_load_format_x v1, v1, s[8:11], 0 idxen offset:448;
    # s_waitcnt vmcnt(0); v_readfirstlane s41, v1
    if case == 'narrowed first lane':
        # Initialize v1 for all lanes, then overwrite it under a lane predicate.
        # The reset restores full EXEC, but that does not prove v1 uniform.
        vector = '7e020280 7d860086 bea4246a 7e020211 befe04c1'.split() + vector[1:]
    if case == 'different VCC arms':
        prefix = ['7d860086']                     # v_cmp_gt_u32 vcc, 6, v0
    branch_op = 0xbf86 if case == 'different VCC arms' else 0xbf84
    prefix += [branch(branch_op, 0, len(scalar) + 2)] + scalar + [branch(0xbf82, 0, len(vector) + 1)] + vector
    # Descriptor reload and scalar buffer read. P is the next byte, B is the low six bits.
    table = 'f4080209 fa000000 bf8cc07f'.split()
    bounds = 'f4240a84 520001d0 bf8cc07f 93abff2a 00080008 872abf2a 802a812a 802b812b'.split()
    if case == 'wave address':
        bounds[1] = '0c0001d0'                   # use s6 (wave index) as the offset
    if case == 'wave bound':
        bounds += ['beaa0306']                    # s_mov_b32 s42, s6
    if case == 'system bound':
        bounds += ['beaa0302']                    # s_mov_b32 s42, s2
    if case == 'memory store':
        bounds += 'e0701000 80020000'.split()    # buffer_store_dword v0, v0, s[8:11], 0 offen
    # Address calculation for the position block; s42 = B and s43 = P.
    # As in the guest, the output counts s21 and s22 are selected after the narrowing.
    limit = '3608008f 34080884 befe04c1 7da8002a 83ac2b2a bf00802c 85152b80 85162a80'.split()
    if case == 'same':
        limit[-1] = 'be96032a'                    # s_mov_b32 s22, s42
    if case == 'plus one':
        limit[-1] = '8016812a'                    # s_add_u32 s22, s42, 1
    if case == 'different SGPR':
        limit[-1] = 'be960311'                    # s_mov_b32 s22, s17
    # An inner scalar if/else with an EXEC-predicated position load in each arm.
    left = GUARDED_POSITION_LOAD.split()
    right = ['bf800000'] if case == 'one arm' else GUARDED_POSITION_LOAD.split()
    positions = ('f4200a04 fa0001e0 bf8cc07f bf0c8010'.split()
                 + [branch(0xbf84, 0, len(left) + 2)] + left
                 + [branch(0xbf82, 0, len(right) + 1)] + right)
    if case in ('bound overwritten in block', 'count overwritten in block'):
        positions += ['beaa0384' if case == 'bound overwritten in block' else 'be960384']
    if case == 'bound overwritten in block':
        # Make the count observe the post-join ID of B.
        limit[-1] = 'be960380'
    position_block = limit + [branch(0xbf88, 0, len(positions) + 1)] + positions + ['befe04c1']
    if case == 'bound overwritten in block':
        position_block += ['be96032a']
    # Load three separate indices per primitive. P is at most two in the draw.
    prim_load = 'f4200a04 fa0001e0 bf8cc07f 34080084 e03c1180 80023604 bf8c3f70'.split()
    if case == 'different guard rewrites position':
        prim_load += GUARDED_POSITION_LOAD.split()
    primitive_block = ['7da8002b', branch(0xbf88, 0, len(prim_load) + 1)] + prim_load + ['befe04c1']
    exchange = MESH[before + 2:POST_COUNT]
    pos = ('7da80016' if case == 'unsigned export' else '7d280016').split() + (
        '3608008f 34080884 e03810c0 80021404 bf8c3f70 f80008cf 35343332 f800020f 17161514 befe04c1'.split())
    # After narrowing by P, replace V by V - 1, clamp each index, then pack the triplet.
    clamp = ('7d280015 8016c116 266c6c16 266e6e16 26707016 346e6e8a 34707094'
             ' 386c6d37 386c6d38 f8000941 00000036 bf810000').split()
    if case == 'compare overwritten count':
        pos.insert(0, '8016c116')
    if case == 'rewrite under the same mask':
        pos.insert(1, '8016c116')
    if case == 'rewrite primitive count':
        clamp.insert(1, '8015c115')               # s_add_u32 s21, s21, -1
    if case == 'keep count':
        clamp[1] = '802cc116'                     # write V - 1 to s44 instead
        clamp[2:5] = [word[:-2] + '2c' for word in clamp[2:5]]
    return MESH[:before] + prefix + table + bounds + position_block + primitive_block + exchange + pos + clamp


def count_entry_guard_mesh(case='scalar bound'):
    """Write primitive indices only on the lanes that fall through to the count exchange."""
    mesh = count_guard_mesh()
    at = mesh.index('7da8002b')
    exchange = mesh.index('bf8a0000', at)
    # v_lshlrev_b32 v4, 4, v0; buffer_load_dwordx3 v[54:56], v4, s[8:11], 0 offen offset:384.
    body = '34080084 e03c1180 80023604 bf8c3f70'.split()
    compare = '7da27d00' if case == 'vector bound' else '7da8002b'
    mesh[at:exchange] = [compare, branch(0xbf88, 0, len(body) + 1)] + body
    end_bounds = mesh.index('802b812b') + 1
    if case == 'vector bound':
        # v_mov_b32 v62, s43; v_readfirstlane_b32 s43, v62 under full EXEC.
        # The count-entry compare is v_cmpx_lt_u32_e32 v0, v62.
        mesh[end_bounds:end_bounds] = '7e7c022b 7e56053e'.split()
    elif case == 'wave bound':
        mesh.insert(end_bounds, '80ab062b')  # s_sub_u32 s43, s43, s6
    elif case == 'different count':
        mesh = mutate('85152b80', '85151180', mesh=mesh)  # s_cselect_b32 s21, 0, s17
    elif case == 'exchange clobber':
        mesh = mutate('7e020215', '7e6c0215', mesh=mesh)  # v_mov_b32 v54, s21
        mesh = mutate('00000102', '00003602', mesh=mesh)  # ds_write_b32 v2, v54
    return mesh


def vector_count_guard_mesh(case='select mask'):
    """Derive the two counts in full-EXEC VGPRs and narrow positions by the VGPR bound."""
    mesh = count_guard_mesh()
    low = mesh.index('d7650000')
    high = mesh.index('d7660000')
    # s_bitcmp0_b32 s16, 0; mbcnt_lo; s_cmp_eq_u32 src_scc, 0;
    # s_cselect_b64 vcc, -1, 0; s_waitcnt lgkmcnt(0); mbcnt_hi.
    gap = 'bf0680fd 85ea80c1 bf8cc07f'.split()
    clear = {'thread clobber': '7e000280', 'EXEC clobber': 'befe04c1', 'base clobber': 'be870380'}
    if case in clear:
        gap.insert(0, clear[case])
    if case == 'branch in gap':
        gap.insert(0, 'bf840001')  # s_cbranch_scc0 to the VCC select
    if case != 'adjacent pair':
        mesh[low:high] = ['bf0c8010'] + mesh[low:high] + gap
    first = mesh.index('f4240a84')
    end = mesh.index('85162a80') + 1
    # v9 holds the joined scalar offset. The clear-bit arm adds four bytes.
    mask = 'bf0c8010 85ba80c1'
    if case == 'uniform compare mask':
        mask = 'd4c4003a 00010081'  # v_cmp_gt_u32_e64 s[58:59], 1, 0
    if case == 'per-lane compare mask':
        mask = 'd4c4003a 00020086'  # v_cmp_gt_u32_e64 s[58:59], 6, v0
    if case == 'constant mask':
        mask = 'beba04c1'           # s_mov_b64 s[58:59], -1
    address = ('7e120229 4e1412c4 ' + mask + ' d5010009 00ea1509').split()
    # Load B-1 and P-1 in one read, extract seven bits of B-1, and add one to each.
    loads = 'e03411d0 80023009 bf8c3f70 367860ff 0000007f 4a787881 4a7a6281'.split()
    # Scale B on the set-bit arm, forming a VGPR phi at the join.
    adjust = 'bf0c8010 bf850002 d569003c 00027811'.split()
    if case == 'thread bound':
        adjust += ['7e780300']      # v_mov_b32_e32 v60, v0
    if case == 'narrowed bound':
        adjust += '7d860086 bea4246a 4a787881 befe04c1'.split()
    # v_readfirstlane s22, v60; v_readfirstlane s21, v61; preserve raw P in s43.
    # After narrowing by thread < v60, choose either zero or the first-lane counts.
    narrow = ('7e2c053c 7e2a053d beab0315 3608008f 34080884 befe04c1 7da27900 '
              '83ac1516 bf00802c 85151580 85161680').split()
    mesh[first:end] = address + loads + adjust + narrow
    return mesh


def rewritten_guarded_position():
    """Overwriting v50 under the primitive mask invalidates its older position definition."""
    mesh = GUARDED_POSITIONS['select']
    at = mesh.index('7d280016')
    return mesh[:at] + '7d280015 7e640280 befe04c1'.split() + mesh[at:]


WAVE_LOAD_MESH = wave_load_mesh()
WAVE_LOAD_REFUSAL = 'slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes'
WAVE_LOAD_REFUSALS = (
    ('different EXEC compare', wave_load_mesh(WAVE_LOAD_PREFIX.replace('7da8002a', '7d28002a'))),
    ('different compare index', wave_load_mesh(WAVE_LOAD_PREFIX.replace('7da8002a', '7da80e2a'))),
    ('EXEC write between compare and branch', wave_load_mesh(WAVE_LOAD_PREFIX.replace('beaa0380', 'befe04c1'))),
    ('branch into load block', wave_load_mesh('bf840009 ' + WAVE_LOAD_PREFIX)),
    ('branch to the EXEC-zero instruction', wave_load_mesh('bf840008 ' + WAVE_LOAD_PREFIX)),
    # The block writes s43, which the join reads, but s43 has no value before the block for an empty wave to keep.
    ('second live scalar write without an old value',
     wave_load_mesh(body=WAVE_LOAD_BODY + ' beab0386', exports='7e280228 7e2a022b')),
    ('buffer store', wave_load_mesh(body=WAVE_LOAD_BODY + ' f4600a04 520001e0')),
    ('buffer atomic', wave_load_mesh(body=WAVE_LOAD_BODY + ' e0c81000 80020700')),
)
# Blocks that #2660 refused and that the wave block accepts: every scalar instruction of the block runs once for a
# wave with an active lane, so further scalar loads and writes, live or dead after the join, follow the guest.
WAVE_LOAD_ACCEPTED = (
    ('wider SMEM load', wave_load_mesh(body=WAVE_LOAD_BODY.replace('f4200a04', 'f4240a04'))),
    ('dead scalar result', wave_load_mesh(exports='7e2a0307')),
    # With the adjacent reset replaced by s_nop, the reset that closes the group-index exchange still reaches the
    # compare: nothing between them reads or writes EXEC or VCC, and no branch enters that range.
    ('full EXEC reset earlier in the region', wave_load_mesh(WAVE_LOAD_PREFIX.replace('befe04c1', 'bf800000'))),
)
# A table load is accepted by the wave block; this one then stops at the register-offset rule of s_load.
WAVE_LOAD_OTHER_REFUSALS = (
    ('unmodelled EXEC write before the compare', wave_load_mesh(WAVE_LOAD_PREFIX.replace('befe04c1', 'befe0404')),
     'slot 57 mesh: s_mov_b64 on a wave mask is not covered'),
    ('ordinary SMEM load with a register offset', wave_load_mesh(body=WAVE_LOAD_BODY.replace('f4200a04', 'f4000a09')),
     "s_load with a register offset through ('table', 1)"),
)

# Inner forward SCC branch inside the wave-load block (the shape of the real slot 57 bodies, lane-exact EXEC slice):
# the loaded word decides per wave whether the block's scalar (s40, the red) takes the loaded value or keeps the old one.
NEST_SCC_ASM = """s_mov_b32 s40, 0x3e808081
s_lshl_b32 s41, s6, 8
s_mov_b32 s42, 6
v_mov_b32 v7, 0
s_mov_b64 exec, -1
v_cmpx_gt_u32_e32 s42, v0
s_mov_b32 s42, 0
s_cbranch_execz 7
s_buffer_load_dword s43, s[8:11], s41 offset:0x1e0
s_waitcnt lgkmcnt(0)
s_cmp_eq_u32 s43, 0
s_cbranch_scc1 1
s_mov_b32 s40, s43
v_mov_b32 v7, 1.0
s_mov_b64 exec, -1"""
NEST_SCC_MESH = ['bfa00003', 'befe0481', 'be84047e', '8706ff03', '0f000000', 'bf850007', '2c0a0a87', '7e0c0280', 'd8340008', '00000506', 'bfa3ffe3', 'bf8a0000', 'bf820004', 'bf8a0000', '7e0c0280', 'd8d80008', '05000006', 'befe04c1', '9386ff03', '00040018', '8f078606', 'd7650000', '00000ec1', 'd7660000', '000200c1', 'bf8cc07f', '7e280505', 'f4080209', 'fa000000', 'bea803ff', '3e808081', '8f298806', 'beaa0386', '7e0e0280', 'befe04c1', '7da8002a', 'beaa0380', 'bf880007', 'f4200ac4', '520001e0', 'bf8cc07f', 'bf06802b', 'bf850001', 'bea8032b', '7e0e02f2', 'befe04c1', 'be950310', 'be960311', 'bf8a0000', 'bf008006', 'bf840010', 'befe0404', 'bf880007', '7e020215', '7e040280', '7e060216', 'd8340000', '00000102', 'd8340004', '00000302', 'befe04c1', 'bfa3ffe3', 'bf8a0000', '8f048c15', '887c1604', 'bf900009', 'bf82000d', 'bf8a0000', 'befe0404', 'bf880009', '7e040280', 'd8d80000', '01000002', 'd8d80004', '03000002', 'bf8cc17f', '7e2a0501', 'bf8cc07f', '7e2c0503', 'befe04c1', '7d280016', '3608008f', '34080884', 'e0381000', '80021004', 'e03810c0', '80021404', 'bf8c3f70', '7e280228', '7e2a0307', 'f80008cf', '13121110', 'f800020f', '17161514', 'befe04c1', '7d280015', '3608008f', '34080882', 'e0301180', '80021804', 'bf8c3f70', 'f8000941', '00000018', 'bf810000']
NEST_ACCEPTED = (('inner forward SCC branch in the wave block', NEST_SCC_MESH),)
# The inner SCC comes from a lane mask (s_and_b64 of a per-lane compare's VCC): one lane's bit, not the wave's, so refused.
NEST_MASK_SCC_MESH = ['bfa00003', 'befe0481', 'be84047e', '8706ff03', '0f000000', 'bf850007', '2c0a0a87', '7e0c0280', 'd8340008', '00000506', 'bfa3ffe3', 'bf8a0000', 'bf820004', 'bf8a0000', '7e0c0280', 'd8d80008', '05000006', 'befe04c1', '9386ff03', '00040018', '8f078606', 'd7650000', '00000ec1', 'd7660000', '000200c1', 'bf8cc07f', '7e280505', 'f4080209', 'fa000000', 'bea803ff', '3e808081', '8f298806', 'beaa0386', '7e0e0280', 'befe04c1', '7da8002a', 'beaa0380', 'bf880008', 'f4200ac4', '520001e0', 'bf8cc07f', '7d840083', '87ac6a6a', 'bf850001', 'bea80380', '7e0e02f2', 'befe04c1', 'be950310', 'be960311', 'bf8a0000', 'bf008006', 'bf840010', 'befe0404', 'bf880007', '7e020215', '7e040280', '7e060216', 'd8340000', '00000102', 'd8340004', '00000302', 'befe04c1', 'bfa3ffe3', 'bf8a0000', '8f048c15', '887c1604', 'bf900009', 'bf82000d', 'bf8a0000', 'befe0404', 'bf880009', '7e040280', 'd8d80000', '01000002', 'd8d80004', '03000002', 'bf8cc17f', '7e2a0501', 'bf8cc07f', '7e2c0503', 'befe04c1', '7d280016', '3608008f', '34080884', 'e0381000', '80021004', 'e03810c0', '80021404', 'bf8c3f70', '7e280228', '7e2a0307', 'f80008cf', '13121110', 'f800020f', '17161514', 'befe04c1', '7d280015', '3608008f', '34080882', 'e0301180', '80021804', 'bf8c3f70', 'f8000941', '00000018', 'bf810000']
NEST_MASK = ('inner SCC from a lane mask', NEST_MASK_SCC_MESH, 'slot 57 mesh: s_and_b64 on a wave mask is not covered')
# The inner branch goes backwards (to the s_cmp): a loop, refused with the loop text.
NEST_LOOP = ('inner backward branch', [w if w != 'bf850001' else 'bf85fffe' for w in NEST_SCC_MESH],
             'slot 57 mesh: a loop outside the group-memory exchanges is not covered')
# Same block, one word changed: every one of these is refused with the wave-load text.
NEST_REFUSALS = (
    # s_cbranch_vccz instead of s_cbranch_scc1: not an inner branch the wave block takes
    ('inner VCC branch', [w if w != 'bf850001' else 'bf860001' for w in NEST_SCC_MESH]),
    # the inner branch leaves the block (past the execz target)
    ('inner branch out of the block', [w if w != 'bf850001' else 'bf850003' for w in NEST_SCC_MESH]),
)

# A nested if inside the wave-load block, the shape of most real slot 57 bodies: a per-lane compare into VCC, s_and_saveexec_b64,
# s_cbranch_execz over a body of lane writes, then the restore s_or_b64 exec, exec, <saved>. Lanes with a thread id below 3 take
# the body (v7 = 1.0), the others keep v7 = 0; every thread holds its own EXEC bit and the saved mask is one bit per thread.
NEST_EXECZ_ASM = """s_mov_b32 s40, 0x3e808081
s_lshl_b32 s41, s6, 8
s_mov_b32 s42, 6
v_mov_b32 v7, 0
s_mov_b64 exec, -1
v_cmpx_gt_u32_e32 s42, v0
s_mov_b32 s42, 0
s_cbranch_execz 8
s_buffer_load_dword s43, s[8:11], s41 offset:0x1e0
s_waitcnt lgkmcnt(0)
v_cmp_gt_u32_e32 vcc, 3, v0
s_and_saveexec_b64 s[44:45], vcc
s_cbranch_execz 1
v_mov_b32 v7, 1.0
s_or_b64 exec, exec, s[44:45]
s_mov_b64 exec, -1"""
NEST_EXECZ_MESH = ['bfa00003', 'befe0481', 'be84047e', '8706ff03', '0f000000', 'bf850007', '2c0a0a87', '7e0c0280', 'd8340008', '00000506', 'bfa3ffe3', 'bf8a0000', 'bf820004', 'bf8a0000', '7e0c0280', 'd8d80008', '05000006', 'befe04c1', '9386ff03', '00040018', '8f078606', 'd7650000', '00000ec1', 'd7660000', '000200c1', 'bf8cc07f', '7e280505', 'f4080209', 'fa000000', 'bea803ff', '3e808081', '8f298806', 'beaa0386', '7e0e0280', 'befe04c1', '7da8002a', 'beaa0380', 'bf880008', 'f4200ac4', '520001e0', 'bf8cc07f', '7d880083', 'beac246a', 'bf880001', '7e0e02f2', '88fe2c7e', 'befe04c1', 'be950310', 'be960311', 'bf8a0000', 'bf008006', 'bf840010', 'befe0404', 'bf880007', '7e020215', '7e040280', '7e060216', 'd8340000', '00000102', 'd8340004', '00000302', 'befe04c1', 'bfa3ffe3', 'bf8a0000', '8f048c15', '887c1604', 'bf900009', 'bf82000d', 'bf8a0000', 'befe0404', 'bf880009', '7e040280', 'd8d80000', '01000002', 'd8d80004', '03000002', 'bf8cc17f', '7e2a0501', 'bf8cc07f', '7e2c0503', 'befe04c1', '7d280016', '3608008f', '34080884', 'e0381000', '80021004', 'e03810c0', '80021404', 'bf8c3f70', '7e280228', '7e2a0307', 'f80008cf', '13121110', 'f800020f', '17161514', 'befe04c1', '7d280015', '3608008f', '34080882', 'e0301180', '80021804', 'bf8c3f70', 'f8000941', '00000018', 'bf810000']
NEST_EXECZ_ACCEPTED = (('nested if (compare, saveexec, execz, restore) in the wave block', NEST_EXECZ_MESH),)
# Same block with one thing wrong each; every one is refused (assembled with llvm-mc from the same text with the change).
NEST_EXECZ_REFUSALS = (
    ('nested restore of another register pair', ['bfa00003', 'befe0481', 'be84047e', '8706ff03', '0f000000', 'bf850007', '2c0a0a87', '7e0c0280', 'd8340008', '00000506', 'bfa3ffe3', 'bf8a0000', 'bf820004', 'bf8a0000', '7e0c0280', 'd8d80008', '05000006', 'befe04c1', '9386ff03', '00040018', '8f078606', 'd7650000', '00000ec1', 'd7660000', '000200c1', 'bf8cc07f', '7e280505', 'f4080209', 'fa000000', 'bea803ff', '3e808081', '8f298806', 'beaa0386', '7e0e0280', 'befe04c1', '7da8002a', 'beaa0380', 'bf880008', 'f4200ac4', '520001e0', 'bf8cc07f', '7d880083', 'beac246a', 'bf880001', '7e0e02f2', '88fe2e7e', 'befe04c1', 'be950310', 'be960311', 'bf8a0000', 'bf008006', 'bf840010', 'befe0404', 'bf880007', '7e020215', '7e040280', '7e060216', 'd8340000', '00000102', 'd8340004', '00000302', 'befe04c1', 'bfa3ffe3', 'bf8a0000', '8f048c15', '887c1604', 'bf900009', 'bf82000d', 'bf8a0000', 'befe0404', 'bf880009', '7e040280', 'd8d80000', '01000002', 'd8d80004', '03000002', 'bf8cc17f', '7e2a0501', 'bf8cc07f', '7e2c0503', 'befe04c1', '7d280016', '3608008f', '34080884', 'e0381000', '80021004', 'e03810c0', '80021404', 'bf8c3f70', '7e280228', '7e2a0307', 'f80008cf', '13121110', 'f800020f', '17161514', 'befe04c1', '7d280015', '3608008f', '34080882', 'e0301180', '80021804', 'bf8c3f70', 'f8000941', '00000018', 'bf810000']),
    ('a scalar write in the nested body', ['bfa00003', 'befe0481', 'be84047e', '8706ff03', '0f000000', 'bf850007', '2c0a0a87', '7e0c0280', 'd8340008', '00000506', 'bfa3ffe3', 'bf8a0000', 'bf820004', 'bf8a0000', '7e0c0280', 'd8d80008', '05000006', 'befe04c1', '9386ff03', '00040018', '8f078606', 'd7650000', '00000ec1', 'd7660000', '000200c1', 'bf8cc07f', '7e280505', 'f4080209', 'fa000000', 'bea803ff', '3e808081', '8f298806', 'beaa0386', '7e0e0280', 'befe04c1', '7da8002a', 'beaa0380', 'bf880009', 'f4200ac4', '520001e0', 'bf8cc07f', '7d880083', 'beac246a', 'bf880002', 'bea8032b', '7e0e02f2', '88fe2c7e', 'befe04c1', 'be950310', 'be960311', 'bf8a0000', 'bf008006', 'bf840010', 'befe0404', 'bf880007', '7e020215', '7e040280', '7e060216', 'd8340000', '00000102', 'd8340004', '00000302', 'befe04c1', 'bfa3ffe3', 'bf8a0000', '8f048c15', '887c1604', 'bf900009', 'bf82000d', 'bf8a0000', 'befe0404', 'bf880009', '7e040280', 'd8d80000', '01000002', 'd8d80004', '03000002', 'bf8cc17f', '7e2a0501', 'bf8cc07f', '7e2c0503', 'befe04c1', '7d280016', '3608008f', '34080884', 'e0381000', '80021004', 'e03810c0', '80021404', 'bf8c3f70', '7e280228', '7e2a0307', 'f80008cf', '13121110', 'f800020f', '17161514', 'befe04c1', '7d280015', '3608008f', '34080882', 'e0301180', '80021804', 'bf8c3f70', 'f8000941', '00000018', 'bf810000']),
    ('a nested saveexec without its compare', ['bfa00003', 'befe0481', 'be84047e', '8706ff03', '0f000000', 'bf850007', '2c0a0a87', '7e0c0280', 'd8340008', '00000506', 'bfa3ffe3', 'bf8a0000', 'bf820004', 'bf8a0000', '7e0c0280', 'd8d80008', '05000006', 'befe04c1', '9386ff03', '00040018', '8f078606', 'd7650000', '00000ec1', 'd7660000', '000200c1', 'bf8cc07f', '7e280505', 'f4080209', 'fa000000', 'bea803ff', '3e808081', '8f298806', 'beaa0386', '7e0e0280', 'befe04c1', '7da8002a', 'beaa0380', 'bf880008', 'f4200ac4', '520001e0', 'bf8cc07f', 'bf800000', 'beac246a', 'bf880001', '7e0e02f2', '88fe2c7e', 'befe04c1', 'be950310', 'be960311', 'bf8a0000', 'bf008006', 'bf840010', 'befe0404', 'bf880007', '7e020215', '7e040280', '7e060216', 'd8340000', '00000102', 'd8340004', '00000302', 'befe04c1', 'bfa3ffe3', 'bf8a0000', '8f048c15', '887c1604', 'bf900009', 'bf82000d', 'bf8a0000', 'befe0404', 'bf880009', '7e040280', 'd8d80000', '01000002', 'd8d80004', '03000002', 'bf8cc17f', '7e2a0501', 'bf8cc07f', '7e2c0503', 'befe04c1', '7d280016', '3608008f', '34080884', 'e0381000', '80021004', 'e03810c0', '80021404', 'bf8c3f70', '7e280228', '7e2a0307', 'f80008cf', '13121110', 'f800020f', '17161514', 'befe04c1', '7d280015', '3608008f', '34080882', 'e0301180', '80021804', 'bf8c3f70', 'f8000941', '00000018', 'bf810000']),
    ('EXEC written in the nested body', ['bfa00003', 'befe0481', 'be84047e', '8706ff03', '0f000000', 'bf850007', '2c0a0a87', '7e0c0280', 'd8340008', '00000506', 'bfa3ffe3', 'bf8a0000', 'bf820004', 'bf8a0000', '7e0c0280', 'd8d80008', '05000006', 'befe04c1', '9386ff03', '00040018', '8f078606', 'd7650000', '00000ec1', 'd7660000', '000200c1', 'bf8cc07f', '7e280505', 'f4080209', 'fa000000', 'bea803ff', '3e808081', '8f298806', 'beaa0386', '7e0e0280', 'befe04c1', '7da8002a', 'beaa0380', 'bf880009', 'f4200ac4', '520001e0', 'bf8cc07f', '7d880083', 'beac246a', 'bf880002', 'befe04c1', '7e0e02f2', '88fe2c7e', 'befe04c1', 'be950310', 'be960311', 'bf8a0000', 'bf008006', 'bf840010', 'befe0404', 'bf880007', '7e020215', '7e040280', '7e060216', 'd8340000', '00000102', 'd8340004', '00000302', 'befe04c1', 'bfa3ffe3', 'bf8a0000', '8f048c15', '887c1604', 'bf900009', 'bf82000d', 'bf8a0000', 'befe0404', 'bf880009', '7e040280', 'd8d80000', '01000002', 'd8d80004', '03000002', 'bf8cc17f', '7e2a0501', 'bf8cc07f', '7e2c0503', 'befe04c1', '7d280016', '3608008f', '34080884', 'e0381000', '80021004', 'e03810c0', '80021404', 'bf8c3f70', '7e280228', '7e2a0307', 'f80008cf', '13121110', 'f800020f', '17161514', 'befe04c1', '7d280015', '3608008f', '34080882', 'e0301180', '80021804', 'bf8c3f70', 'f8000941', '00000018', 'bf810000']),
)

# Two scalar buffer loads (the second at an offset computed in the block), a dead table load, a dead per-lane VCC compare and a scalar
# write between the EXEC reset and the compare. The second load reads the same word as WAVE_LOAD_ASM
# (s45 + 0x1dc = s41 + 0x1e0); the first reads the next word and is dead after the join.
WAVE_BLOCK_ASM = """s_mov_b32 s40, 0x3e808081
s_lshl_b32 s41, s6, 8
s_mov_b32 s42, 6
v_mov_b32 v7, 0
s_mov_b64 exec, -1
s_mov_b32 s43, 0
v_cmpx_gt_u32_e32 s42, v0
s_mov_b32 s42, 0
s_cbranch_execz 11
s_buffer_load_dword s44, s[8:11], s41 offset:0x1e4
s_load_dwordx4 s[48:51], s[18:19], 0x0
s_add_u32 s45, s41, 4
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s40, s[8:11], s45 offset:0x1dc
v_cmp_class_f32_e32 vcc, -1, v7
v_mov_b32 v7, 1.0
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, -1"""
WAVE_BLOCK_PREFIX = WAVE_LOAD_PREFIX.replace('befe04c1 7da8002a', 'befe04c1 beab0380 7da8002a')
WAVE_BLOCK_BODY = 'f4200b04 520001e4 f4080c09 fa000000 802d8429 bf8cc07f f4200a04 5a0001dc 7d100ec1 7e0e02f2 bf8cc07f'
WAVE_BLOCK_MESH = wave_load_mesh(WAVE_BLOCK_PREFIX, WAVE_BLOCK_BODY)
WAVE_LOAD_OTHER_REFUSALS += (
    ('unmodelled EXEC write within the reset range',
     wave_load_mesh(WAVE_BLOCK_PREFIX.replace('beab0380', 'befe0404'), WAVE_BLOCK_BODY),
     'slot 57 mesh: s_mov_b64 on a wave mask is not covered'),
)
WAVE_BLOCK_REFUSALS = (
    ('VCC live at the join', wave_load_mesh(WAVE_BLOCK_PREFIX, WAVE_BLOCK_BODY, exports='7e280228 022a0e80')),
    ('EXEC copied in the block', wave_load_mesh(WAVE_BLOCK_PREFIX, WAVE_BLOCK_BODY + ' beb0047e')),
    ('VCC branch inside the block', wave_load_mesh(WAVE_BLOCK_PREFIX, WAVE_BLOCK_BODY.replace('bf8cc07f', 'bf860000', 1))),
    ('branch from outside between the reset and the compare', wave_load_mesh('bf840006 ' + WAVE_BLOCK_PREFIX,
                                                                              WAVE_BLOCK_BODY)),
    # s_cmp_eq_u32 s43, 0; s_cbranch_scc1 to the execz branch: that path never runs the compare.
    ('branch from between the reset and the compare past the compare',
     wave_load_mesh(WAVE_BLOCK_PREFIX.replace('beab0380', 'beab0380 bf06802b bf850002'), WAVE_BLOCK_BODY)),
)


# The loaded word decides whether any active lane sets VCC. The inner scalar write
# must reach a provoking thread outside EXEC when the wave takes the fall-through.
INNER_VCC_ASM = NEST_SCC_ASM.replace('s_cmp_eq_u32 s43, 0\ns_cbranch_scc1 1',
                                     'v_cmp_gt_u32_e32 vcc, s43, v0\ns_cbranch_vccz 1')
INNER_VCC_MESH = wave_load_mesh(body='f4200ac4 520001e0 bf8cc07f 7d88002b bf860001 bea8032b 7e0e02f2')

# The second bound alone would let wave 1 load from beyond the SRV. Keep the
# earlier EXEC narrowing to threads 0..2 when applying this bound.
NO_RESET_ASM = WAVE_LOAD_ASM.replace('s_mov_b32 s42, 6', 's_mov_b32 s42, 0x46').replace(
    's_mov_b64 exec, -1\nv_cmpx', 'v_cmp_gt_u32_e32 vcc, 3, v0\ns_and_saveexec_b64 s[44:45], vcc\nv_cmpx')
NO_RESET_MESH = wave_load_mesh(WAVE_LOAD_PREFIX.replace('beaa0386', 'beaa03ff 00000046').replace(
    'befe04c1', '7d880083 beac246a'))

# Select lane 1, which holds red 1.0. Lane 0 and the inactive provoking threads
# hold 0 in v7, and the scalar load gives s40 a different value.
FIRST_LANE_ASM = WAVE_LOAD_ASM.replace(
    's_mov_b64 exec, -1\nv_cmpx', 'v_cmp_lt_u32_e32 vcc, 0, v0\ns_and_saveexec_b64 s[44:45], vcc\nv_cmpx').replace(
    's_cbranch_execz 4', 's_cbranch_execz 5').replace(
    's_waitcnt lgkmcnt(0)\ns_mov_b64', 's_waitcnt lgkmcnt(0)\nv_readfirstlane_b32 s40, v7\ns_mov_b64')
FIRST_LANE_MESH = wave_load_mesh(WAVE_LOAD_PREFIX.replace('befe04c1', '7d820080 beac246a'),
                                 WAVE_LOAD_BODY + ' 7e500507')

WAVE_COLLECTIVES = (
    ('inner VCC branch', INNER_VCC_ASM, INNER_VCC_MESH),
    ('no full EXEC reset', NO_RESET_ASM, NO_RESET_MESH),
    ('readfirstlane broadcast', FIRST_LANE_ASM, FIRST_LANE_MESH),
)


def count_entry_red(*extra_words):
    wait = MESH.index('bf8c3f70')
    return MESH[:COUNT] + [
        '7e0e0280',  # v_mov_b32_e32 v7, 0
        'befe04c1',  # s_mov_b64 exec, -1
        '7da80015',  # v_cmpx_gt_u32_e32 s21, v0 (the proven thread index)
        branch(0xbf88, COUNT + 3, COUNT + 5 + len(extra_words)),  # s_cbranch_execz to the first count barrier
        '7e0e02f2',  # v_mov_b32_e32 v7, 1.0
    ] + list(extra_words) + MESH[COUNT:wait + 1] + [
        '7e280307',  # v_mov_b32_e32 v20, v7
    ] + MESH[wait + 1:]


COUNT_ENTRY = count_entry_red()
COUNT_ENTRY_LT = mutate('7da80015', '7da20015', mesh=COUNT_ENTRY)
# v_mov_b32_e32 v8, 4 before the reset, then v_cmpx_lt_u32_e32 v8, v0.
COUNT_ENTRY_VGPRS = mutate('7da80015', '7da20108', mesh=COUNT_ENTRY)
COUNT_ENTRY_VGPRS = COUNT_ENTRY_VGPRS[:COUNT + 1] + ['7e100284'] + COUNT_ENTRY_VGPRS[COUNT + 1:]
DEAD_SCALAR = '87288080'  # s_and_b32 s40, 0, 0
COUNT_ENTRY_SCALAR = count_entry_red(DEAD_SCALAR)
COUNT_ENTRY_CASES = (
    ('EXEC-zero branch to the count exchange after GT', COUNT_ENTRY, lambda lane: 255 if lane < 4 else 0),
    ('EXEC-zero branch to the count exchange after LT', COUNT_ENTRY_LT, lambda lane: 255 if lane > 4 else 0),
    ('EXEC-zero branch to the count exchange after LT with two VGPRs', COUNT_ENTRY_VGPRS,
     lambda lane: 255 if lane > 4 else 0),
    ('EXEC compare with a defined VGPR other than the thread index',
     mutate('7da80015', '7da80e15', mesh=COUNT_ENTRY), lambda lane: 255),  # s21 > v7 (4 > 0)
    ('EXEC-zero branch to the count exchange over dead scalar ALU', COUNT_ENTRY_SCALAR,
     lambda lane: 255 if lane < 4 else 0),
)
EXCHANGE_BRANCH_REFUSAL = 'slot 57 mesh: a branch enters or crosses a group-memory exchange'


def joined_vertex_masks():
    pos = MESH.index('f80008cf')
    # Move the vertex narrowing after the loads so the join cannot make any export operands stale.
    return MESH[:POST_COUNT] + MESH[POST_COUNT + 1:pos] + [
        '7d280016',  # v_cmpx_gt_i32 s22, v0
        'bf008080',  # s_cmp_eq_i32 0, 0
        'bf850001',  # s_cbranch_scc1 .pos (first incoming mask is the vertex count)
        'befe04c1',  # s_mov_b64 exec, -1 (fallthrough reaches .pos with all lanes)
    ] + MESH[pos:]


# Fixture body encodings are gfx1013 wave64. The existing synthetic shell supplies the two exchanges
# and the count-guarded exports. Every variant adds a loaded displacement to the exported X coordinate.
def scalar_position_mesh(shape, control=None):
    body = ['f4240b84', 'fa0001e0',  # s_buffer_load_dwordx2 s[46:47], s[8:11], 0x1e0
            'bf8cc07f']             # s_waitcnt lgkmcnt(0)
    step = ['bf07802e',             # s_cmp_lg_u32 s46, 0
            '8f30fd2e',             # s_lshl_b32 s48, s46, src_scc
            '7e0e0230']             # v_mov_b32 v7, s48
    if control == 'nonuniform input':
        step[0:1] = ['bf800000']     # SCC of saveexec is a lane bit in the model
    if control == 'undefined scalar input':
        step[0] = 'bf078036'        # s_cmp_lg_u32 s54, 0 (s54 has no definition)
    if control == 'mask select':
        step.insert(1, '8531806a')   # s_cselect_b32 s49, vcc_lo, 0
    if control == 'unguarded export':
        step.extend(('f80008cf', '13121110'))
    if control == 'unguarded store':
        step.extend(('e0701000', '80020700'))
    if shape == 2:
        # Compare only the narrowed lanes against the second loaded dword.
        step[0:0] = ['7d84002f', 'bf86%04x' % len(step)]
    if shape:
        skip = {'backward': -2, 'outside': len(step) + 2, 'exchange': len(step) + 5}
        body += ['7d880088', 'beb4246a',  # v_cmp_gt_u32 vcc, 8, v0; s_and_saveexec_b64 s[52:53], vcc
                 'bf88%04x' % (skip.get(control, len(step)) & 0xffff), *step,
                 '88fe347e']             # s_or_b64 exec, exec, s[52:53]
    else:
        body += step
    prefix = ['7e0e0280', 'beb20393', 'befe04c1', '7da80032']  # v7 = 0; full EXEC; thread < 19
    if control == 'enters block':
        prefix.insert(0, 'bf840005')  # jumps past the outer branch from outside
    prologue = MESH[:COUNT]
    suffix = MESH[COUNT:]
    pos = suffix.index('f80008cf')
    suffix[pos:pos] = ['06202107']  # v_add_f32 v16, v7, v16
    finish = ['befe04c1']
    if control == 'undefined result':
        finish += ['7e0e0230']
    if control in ('undefined inner result', 'defined inner result'):
        body += ['7e0e0230']  # every lane can read s48 after the inner restore
    if control == 'defined inner result':
        prefix.insert(0, 'beb00380')  # s_mov_b32 s48, 0 is the skipped arm's definition
    return prologue[:-2] + prefix + ['bf88%04x' % len(body)] + body + finish + prologue[-2:] + suffix


SCALAR_POSITION_SHAPES = ('scalar-only', 'inner EXEC skip', 'inner VCC and EXEC skips')


def request(directory, mesh, pixel=PIXEL, root_blob=None):
    ms, ps, blob = code(mesh), code(pixel), root_blob or root()
    key = G.mesh57_key(ms, ps, blob)
    for ext, data in (('.root', blob), ('.ps57', ps), ('.m57', ms)):
        (Path(directory) / ('req_%016x%s' % (key, ext))).write_bytes(data)
    return key, G.translate_request(str(directory), key)


FAILED = []


def refusal_head(text):
    """The refusal of a wave-load block names the failing conditions after the fixed text; the checks compare the fixed part."""
    return text.split(' (wave-load block:')[0]


def expect(what, got, want):
    if got != want:
        FAILED.append(what)
        print('FAIL %s: got %r, want %r' % (what, got, want))


REFUSALS = (
    ('formatted load with descriptor word 3 overwritten', REWRITTEN_FORMAT,
     'formatted buffer load through a descriptor whose words were rewritten'),
    # The request must carry the two stored counts, in this order.
    ('a request that publishes another count', mutate('887c1604', '887c1504'),
     'slot 57 mesh count exchange: instruction 15 of the accepted shape (s_or_b32) is absent'),
    ('no barrier between the count store and the reads', mutate('bf8a0000', 'bf800000', 3),
     'slot 57 mesh count exchange: instruction 13 of the accepted shape (s_barrier) is absent'),
    ('a count store by every lane', mutate('befe0404', 'befe04c1'),
     'slot 57 mesh count exchange: instruction 4 of the accepted shape (s_mov_b64) is absent'),
    ('no wait before the first barrier', mutate('bfa3ffe3', 'bf800000'),
     'slot 57 mesh group-index exchange: instruction 8 of the accepted shape (s_waitcnt_depctr) is absent'),
    ('a branch that leaves the count exchange', mutate('bf82000d', 'bf82000e'),
     'slot 57 mesh count exchange: a branch leaves the accepted shape'),
    ('a barrier after the count exchange', mutate('bf8c3f70', 'bf8a0000'),
     'slot 57 mesh: s_barrier outside the group-memory exchanges'),
    ('a loop after the count exchange',
     mutate('bf8c3f70', branch(0xbf82, MESH.index('bf8c3f70'), POST_COUNT)),
     'slot 57 mesh: a loop outside the group-memory exchanges is not covered'),
    ('a branch into the count exchange',
     MESH[:COUNT] + [branch(0xbf82, COUNT, COUNT + 2)] + MESH[COUNT:],
     'slot 57 mesh: a branch enters or crosses a group-memory exchange'),
    ('an EXEC compare with an undefined first VGPR operand',
     mutate('7da20108', '7da20101', mesh=COUNT_ENTRY_VGPRS), EXCHANGE_BRANCH_REFUSAL),
    ('an EXEC compare with an unmodelled entry VGPR',
     mutate('7da80015', '7da80215', mesh=COUNT_ENTRY), EXCHANGE_BRANCH_REFUSAL),
    ('no EXEC reset before the compare',
     mutate('befe04c1', 'bf800000', 1, mesh=COUNT_ENTRY), EXCHANGE_BRANCH_REFUSAL),
    ('an EXEC-nonzero branch to the count exchange',
     mutate('bf880001', 'bf890001', mesh=COUNT_ENTRY), EXCHANGE_BRANCH_REFUSAL),
    ('an EXEC write in the skipped block',
     mutate('7e0e02f2', 'befe04c1', mesh=COUNT_ENTRY), EXCHANGE_BRANCH_REFUSAL),
    # Replace v_mov_b32 v20, v7 after the count exchange with v_mov_b32 v20, s40.
    ('scalar ALU in the count-entry block with a live SGPR',
     mutate('7e280307', '7e280228', mesh=COUNT_ENTRY_SCALAR), EXCHANGE_BRANCH_REFUSAL),
    ('scalar ALU writing VCC in the count-entry block',
     count_entry_red('876a8080'), EXCHANGE_BRANCH_REFUSAL),  # s_and_b32 vcc_lo, 0, 0
    ('a scalar load in the count-entry block',
     count_entry_red('f4000a09', 'fa000000'), EXCHANGE_BRANCH_REFUSAL),  # s_load_dword s40, s[18:19], 0
    ('a v_cmpx in the skipped block',
     mutate('7e0e02f2', '7da80015', mesh=COUNT_ENTRY), EXCHANGE_BRANCH_REFUSAL),
    ('an EXEC-zero branch over the count exchange',
     mutate('bf880001', branch(0xbf88, COUNT + 3, POST_COUNT + 5), mesh=COUNT_ENTRY), EXCHANGE_BRANCH_REFUSAL),
    ('an EXEC-zero branch past the start of the count exchange',
     mutate('bf880001', 'bf880002', mesh=COUNT_ENTRY), EXCHANGE_BRANCH_REFUSAL),
    ('a branch that bypasses the EXEC reset',
     COUNT_ENTRY[:COUNT] + ['bf850002'] + COUNT_ENTRY[COUNT:], EXCHANGE_BRANCH_REFUSAL),
    ('a branch into the skipped count-entry block',
     COUNT_ENTRY[:COUNT] + ['bf850004'] + COUNT_ENTRY[COUNT:], EXCHANGE_BRANCH_REFUSAL),
    ('a branch over the group-index exchange', MESH[:1] + [branch(0xbf82, 1, GROUP_END + 1)] + MESH[1:],
     'slot 57 mesh: a branch enters or crosses a group-memory exchange'),
    # s_branch .post_count replaces s_waitcnt lgkmcnt(0) between the exchanges.
    ('a branch over the count exchange',
     mutate('bf8cc07f', branch(0xbf82, MESH.index('bf8cc07f'), POST_COUNT)),
     'slot 57 mesh: a branch enters or crosses a group-memory exchange'),
    ('an indirect branch', mutate('bf8c3f70', 'be80200c'),
     'slot 57 mesh: indirect control flow (s_setpc_b64) is not covered'),
    # s_call_b64 s[12:13], 2 replaces s_waitcnt vmcnt(0) and targets the parameter export.
    ('a call after the count exchange', mutate('bf8c3f70', 'bb0c0002'),
     'slot 57 mesh: indirect control flow (s_call_b64) is not covered'),
    # s_swappc_b64 s[12:13], s[14:15] replaces s_waitcnt vmcnt(0).
    ('a PC swap after the count exchange', mutate('bf8c3f70', 'be8c210e'),
     'slot 57 mesh: indirect control flow (s_swappc_b64) is not covered'),
    # s_cbranch_execnz .join replaces s_cbranch_scc1 .join and keeps its forward target.
    ('an EXEC-nonzero branch after the count exchange', mutate('bf850001', 'bf890001', mesh=BRANCH_CASES[0][1]),
     'slot 57 mesh: s_cbranch_execnz on a wave-wide condition is not covered'),
    # v_cmp_ge_f32_e64 vcc_lo, v0, 1.0; s_cbranch_vccz .join; v_mov v7, 1.0
    ('a VCC-zero branch after a VGPR compare',
     vertex_red('7e0e0280 d406006a 0001e500 bf860001 7e0e02f2'),
     'slot 57 mesh: s_cbranch_vccz on a wave-wide condition is not covered'),
    # v_cmp_gt_u32_e32 vcc_lo, 6, v0 replaces VCC before s_cbranch_vccnz .join.
    ('a VCC-nonzero branch after an intervening VCC write',
     vertex_red('7e0e0280 ' + VCC_COMPARE + ' 7d880086 bf870001 7e0e02f2'),
     'slot 57 mesh: s_cbranch_vccnz on a wave-wide condition is not covered'),
    # s_mov_b32 s40, 0.5; s_cmp_eq_i32 0, 0; s_cbranch_scc1 .join skips s_mov_b32 s40, 1.0; .join: the compare
    # reads s40, written before the join, which Stage's uniform_source does not prove; s_cbranch_vccz skips s_nop.
    ('a VCC compare on an SGPR written before a join',
     MESH[:POST_COUNT] + ['bea803f0', 'bf008080', 'bf850001', 'bea803f2'] + VCC_COMPARE.split()[1:]
     + ['bf860001', 'bf800000'] + MESH[POST_COUNT:],
     'slot 57 mesh: s_cbranch_vccz on a wave-wide condition is not covered'),
    # v_cmp_gt_u32 vcc, 6, v0; s_and_saveexec_b64 s[12:13], vcc narrows EXEC; the uniform compare then writes VCC
    # only for the narrowed lanes; s_mov_b64 exec, -1 restores the full mask before s_cbranch_vccz skips s_nop.
    ('a VCC compare under a narrowed EXEC before the branch',
     MESH[:POST_COUNT] + ['7d880086', 'be8c246a'] + VCC_COMPARE.split() + ['befe04c1', 'bf860001', 'bf800000']
     + MESH[POST_COUNT:],
     'slot 57 mesh: s_cbranch_vccz on a wave-wide condition is not covered'),
    # Set uniform VCC, then keep the v_cmpx_gt_i32 that narrows EXEC before s_cbranch_vccz skips s_nop.
    ('a VCC-zero branch under narrowed EXEC',
     MESH[:POST_COUNT] + VCC_COMPARE.split() + [MESH[POST_COUNT], 'bf860001', 'bf800000'] + MESH[POST_COUNT + 1:],
     'slot 57 mesh: s_cbranch_vccz on a wave-wide condition is not covered'),
    # s_cbranch_vccnz .join replaces the SCC branch without defining VCC.
    ('a VCC-nonzero branch without a compare', mutate('bf850001', 'bf870001', mesh=BRANCH_CASES[0][1]),
     'slot 57 mesh: s_cbranch_vccnz on a wave-wide condition is not covered'),
    # s_cbranch_scc1 .join can skip all three dwords that set s40 and VCC.
    ('a VCC-zero branch after a join missing a VCC definition on one path',
     vertex_red('7e0e0280 bf008080 bf850003 ' + VCC_COMPARE + ' bf860001 7e0e02f2'),
     'slot 57 mesh: s_cbranch_vccz on a wave-wide condition is not covered'),
    # s_endpgm before the vertex count compare, with the original final s_endpgm still present.
    ('an early program end', MESH[:POST_COUNT] + ['bf810000'] + MESH[POST_COUNT:],
     'slot 57 mesh: an early s_endpgm is not covered'),
    # s_or_b64 exec, exec, s[12:13] replaces the full EXEC reset after the else arm.
    ('a saved EXEC read after the masked if/else', mutate('befe04c1', '89fe7e0c', 3, mesh=masked_vertex_red()),
     'slot 57 mesh: an unproven use of a saved EXEC mask is not covered'),
    # s_branch .join leaves the fallthrough v_mov without an incoming edge.
    ('untargeted code after an unconditional forward branch',
     mutate('bf850001', 'bf820001', mesh=BRANCH_CASES[0][1]),
     'slot 57 mesh: unreachable code after a branch is not covered'),
    ('a VGPR defined on only one incoming path', vertex_red('bf008080 bf850001 7e0e02f2'),
     'slot 57 mesh: v7 is read where it is not defined for every lane (an entry value, or left different per wave '
     'by an exchange)'),
    ('a branch that skips the position EXEC narrowing',
     MESH[:POST_COUNT] + 'bf008080 bf850001'.split() + MESH[POST_COUNT:],
     'slot 57 mesh: export pos0 is not under exec = -1 narrowed by its output count > thread index'),
    ('a position export after joining different EXEC masks', joined_vertex_masks(),
     'slot 57 mesh: export pos0 is not under exec = -1 narrowed by its output count > thread index'),
    ('EXEC-zero skips a scalar memory load', vertex_red(SHIFT_WORDS.replace('8f2881ff 1fc00000', 'f4000a00 fa000000')),
     'slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes'),
    ('a shift with a mask from an unproven thread register',
     vertex_red(SHIFT_WORDS.replace('7d880086', '7e100300 7d881086')),  # v_mov v8,v0; compare against v8
     'slot 57 mesh: scalar shift needs a proven wave EXEC predicate'),
    ('a path enters the scalar shift without the compare or saveexec',
     vertex_red(SHIFT_WORDS.replace('7d880086', 'bf008080 bf850003 7d880086')),
     'slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes'),
    ('a shift without an immediate full EXEC restore',
     vertex_red(SHIFT_WORDS.replace('befe04c1', 'bf800000 befe04c1')),  # branch to s_nop before the restore
     'slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes'),
    # s_cmp_eq_i32 0, 0; s_cbranch_scc1 .end (over the two-dword primitive export).
    ('a uniform branch that skips the primitive export',
     MESH[:PRIMITIVE_EXPORT] + ['bf008080', 'bf850002'] + MESH[PRIMITIVE_EXPORT:],
     'slot 57 mesh: a branch bypasses an export'),
    # .empty: s_mov_b32 s40, 1 replaces the s_nop between the EXEC-zero target and .else.
    ('a branch in an EXEC-zero range that jumps past a scalar write at the target',
     mutate('bf800000', 'bea80381', mesh=masked_vertex_red()),
     'slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes'),
    ('a position export under the primitive count', mutate('7d280016', '7d280015'),
     'slot 57 mesh: export pos0 is not under exec = -1 narrowed by its output count > thread index'),
    # v_mov_b32 v7, v1: v1 holds the stored count in wave 0 and the read one elsewhere.
    ('a read of a register the exchange leaves different per wave', mutate('bf8c3f70', '7e0e0301'),
     'slot 57 mesh: v1 is read where it is not defined for every lane (an entry value, or left different per wave '
     'by an exchange)'),
    # v_mov_b32 v7, v10; s_mov_b32 s12, s25; s_mov_b32 s12, s3: entry values the model does not define.
    ('a read of an entry VGPR', mutate('bf8c3f70', '7e0e030a'),
     'slot 57 mesh: v10 is read where it is not defined for every lane (an entry value, or left different per wave '
     'by an exchange)'),
    ('a read of an entry SGPR', mutate('bf8c3f70', 'be8c0319'),
     'numeric use of s25 holding an entry value the mesh model does not define'),
    ('a read of s3 other than the wave index', mutate('bf8c3f70', 'be8c0303'),
     'slot 57 mesh: s3 is read other than as the wave index'),
    # s_and_b32 s4, s3, 0xf000000: wave 0 would restore a zero mask and store no count.
    ('a wave test that overwrites the saved lane-0 mask', mutate('8706ff03', '8704ff03'),
     'slot 57 mesh count exchange: registers of the accepted shape overlap'),
    # s_mov_b32 s6, s16 after the thread index is derived: the count compare no longer tells the waves apart.
    ('a count compare of a register rewritten after the wave index', mutate('bf8cc07f', 'be860310'),
     'slot 57 mesh count exchange: the compared register is not the wave index'),
)


def depth_only_controls():
    """A pixel stage that exports only to the null target is published as a depth-only pair."""
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH, DEPTH_PIXEL)
        expect('the depth-only pair translates', result, 'ok')
        if result == 'ok':
            names = sorted(p.name[16:] for p in Path(tmp).iterdir() if not p.name.startswith('req_'))
            expect('depth-only pair source and manifest', names, ['.ms57.hlsl', '.ms57.txt', '.ps57.hlsl'])
            manifest = (Path(tmp) / ('%016x.ms57.txt' % key)).read_text().splitlines()
            # The pixel stage is bound but names no render target: an empty targets line, never 'ps none'.
            expect('depth-only manifest stage information',
                   [line for line in manifest if line.startswith(('params ', 'targets', 'wmask ', 'ps '))],
                   ['params 1', 'targets '])
            pixel = (Path(tmp) / ('%016x.ps57.hlsl' % key)).read_text()
            expect('depth-only pixel stage has no output', ('void main(PSIn i)' in pixel, 'SV_Target' in pixel,
                                                            'SV_Depth' in pixel), (True, False, False))
            expect('depth-only pixel stage keeps the vm discard', pixel.count('discard;'), 1)
            # The existing manifest reader (pending_keys): the pair is complete only with both bytecodes.
            expect('depth-only manifest is enough in source-only mode', G.pending_keys(tmp, False), [])
            part = struct.pack('<4sI4s', b'DXIL', 4, b'test')
            bytecode = struct.pack('<4s16sHHIII', b'DXBC', bytes(16), 1, 0, 36 + len(part), 1, 36) + part
            os.environ.pop('X2W_FXC')
            (Path(tmp) / ('%016x.ms57.cso' % key)).write_bytes(bytecode)
            expect('a depth-only pair without its pixel bytecode is pending (unlike ps none)', G.pending_keys(tmp, False), [key])
            pixel_bytecode = Path(tmp) / ('%016x.ps57.cso' % key)
            pixel_bytecode.write_bytes(bytecode)
            expect('a depth-only pair with both bytecodes is done', G.pending_keys(tmp, False), [])
            pixel_bytecode.write_bytes(bytecode[:-1])
            expect('a depth-only pair with truncated pixel bytecode is pending', G.pending_keys(tmp, False), [key])
            os.environ['X2W_FXC'] = ''
    for what, pixel, reason in DEPTH_PIXEL_MUTANTS:
        with tempfile.TemporaryDirectory() as tmp:
            key, result = request(tmp, MESH, pixel)
            expect(what, result, 'unsupported: ' + reason)
            expect(what + ': only the refusal is published',
                   sorted(p.name for p in Path(tmp).iterdir() if not p.name.startswith('req_')), ['%016x.fail' % key])
    # A pair whose pixel stage is not the one the key was computed over, in both directions.
    for what, keyed, stored in (('a colour stage stored under a depth-only key', DEPTH_PIXEL, PIXEL),
                                ('a depth-only stage stored under a colour key', PIXEL, DEPTH_PIXEL)):
        with tempfile.TemporaryDirectory() as tmp:
            key = G.mesh57_key(code(MESH), code(keyed), root())
            for ext, data in (('.root', root()), ('.ps57', code(stored)), ('.m57', code(MESH))):
                (Path(tmp) / ('req_%016x%s' % (key, ext))).write_bytes(data)
            expect(what, G.translate_request(tmp, key), 'unsupported: slot 57 mesh request key mismatch')
            expect(what + ': only the refusal is published',
                   sorted(p.name for p in Path(tmp).iterdir() if not p.name.startswith('req_')), ['%016x.fail' % key])
    # The mesh shapes the paired path refuses stay refused, with the same message, next to a depth-only pixel stage.
    for what, mesh, reason in REFUSALS:
        with tempfile.TemporaryDirectory() as tmp:
            expect(what + ' (depth-only pair)', request(tmp, mesh, DEPTH_PIXEL)[1], 'unsupported: ' + reason)


COUNT_GUARD_ACCEPTED = ('select', 'same', 'unsigned export', 'keep count', 'rewrite under the same mask',
                        'rewrite primitive count')
COUNT_GUARD_REFUSED = ('wave bound', 'system bound', 'wave address', 'memory store', 'bound overwritten in block',
                       'count overwritten in block', 'narrowed first lane', 'different VCC arms', 'plus one',
                       'different SGPR', 'one arm', 'different guard rewrites position')
COUNT_COMPARE_REFUSAL = ('unsupported: slot 57 mesh: an EXEC compare other than output count > thread index under '
                         'exec = -1 is not covered')


def count_guard_mutants(tmp):
    """Make the wave-bound and later-write controls translate by breaking their respective proof."""
    number = G.mesh_value_numbers

    def ignore_wave_variance(*args):
        at, uniform, selects = number(*args)
        instructions = args[0]
        compare = next(i for i, ins in enumerate(instructions) if ins[3] == 'v_cmpx_gt_u32_e32 s42, v0')
        return at, uniform | {at[compare][0]['s42']}, selects

    with patch.object(G, 'mesh_value_numbers', ignore_wave_variance):
        expect('m1 ignoring uniformity fails the wave-index control', request(tmp, count_guard_mesh('wave bound'))[1], 'ok')

    writes = G.mesh_vector_writes

    def miss_later_write(name, asm):
        return set() if asm == 'v_mov_b32_e32 v50, 0' else writes(name, asm)

    with patch.object(G, 'mesh_vector_writes', miss_later_write):
        expect('m2 keeping stale tags fails the later-write control', request(tmp, rewritten_guarded_position())[1], 'ok')


def vector_count_guard_controls(tmp):
    for label in ('select mask', 'uniform compare mask', 'constant mask', 'adjacent pair'):
        key, result = request(tmp, vector_count_guard_mesh(label))
        expect('VGPR guard with ' + label, result, 'ok')
        if result == 'ok':
            source = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            expect('position guard compares the thread and VGPR bound', '(v0 < v60)' in source, True)
            expect('VGPR guard uses the whole-wave EXEC vote', 'uint mesh_vote_' in source, True)
            expect('VGPR guard uses the published vertex count', 'max(mesh_vertex_count, 1u) - 1u' in source, True)
    for label in ('thread bound', 'narrowed bound', 'per-lane compare mask'):
        expect('VGPR guard with ' + label, request(tmp, vector_count_guard_mesh(label))[1], GUARDED_REFUSAL)
    for label in ('thread clobber', 'EXEC clobber', 'base clobber', 'branch in gap'):
        expect('mbcnt gap with ' + label, request(tmp, vector_count_guard_mesh(label))[1],
               'unsupported: ' + WAVE_LOAD_REFUSAL)
    for bit in (0, 1):
        _, guest, wrong = count_guard_reference(bit, vector_bound=True)
        expect('VGPR bound clamp mutant changes prefix arm %d' % bit, guest != wrong, True)


def count_entry_guard_controls(tmp):
    for label in ('scalar bound', 'vector bound'):
        key, result = request(tmp, count_entry_guard_mesh(label))
        expect('count-entry primitive guard with ' + label, result, 'ok')
        if result == 'ok':
            source = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            steps = ('if (exec & 1) v54 = r0.Load(', 'SetMeshOutputCounts(',
                     'o.pos = ', 'int(s21) > int(v0)', 'if (exec & 1) v54 = min(s22, v54);')
            offsets = [source.find(step) for step in steps]
            expect('primitive load before exchange and count-masked clamp after the position export',
                   offsets == sorted(set(offsets)) and offsets[0] >= 0, True)
    for label in ('wave bound', 'different count', 'exchange clobber'):
        expect('count-entry primitive guard with ' + label, request(tmp, count_entry_guard_mesh(label))[1],
               GUARDED_REFUSAL.replace('v50', 'v54'))


def count_guard_controls(tmp):
    for label in COUNT_GUARD_ACCEPTED:
        key, result = request(tmp, count_guard_mesh(label))
        expect('count guard ' + label, result, 'ok')
        if result != 'ok':
            continue
        source = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
        vert = 's22' if label == 'keep count' else 'mesh_vertex_count'
        prim = 'mesh_primitive_count' if label == 'rewrite primitive count' else 's21'
        expect(label + ' allocates using the published counts', f'SetMeshOutputCounts({vert}, {prim});' in source, True)
        expect(label + ' clamps using the published vertex count', f'max({vert}, 1u) - 1u' in source, True)
        if label == 'select':
            expect('m3 rewritten count changes primitive-clamp HLSL', clamp_with_rewritten_count(source) != source, True)
    for label in COUNT_GUARD_REFUSED:
        expect('count guard ' + label, request(tmp, count_guard_mesh(label))[1], GUARDED_REFUSAL)
    expect('write under a later primitive mask', request(tmp, rewritten_guarded_position())[1], GUARDED_REFUSAL)
    expect('compare the rewritten count', request(tmp, count_guard_mesh('compare overwritten count'))[1],
           COUNT_COMPARE_REFUSAL)

    # Isolate the count-write rule with the old fully-defined output shape.
    at = MESH.index('7d280015') + 1
    key, result = request(tmp, MESH[:at] + ['8016c116'] + MESH[at:])
    expect('count rewrite in the old output shape', result, 'ok')
    if result == 'ok':
        source = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
        expect('legacy shape clamps with the saved count', 'max(mesh_vertex_count, 1u) - 1u' in source, True)

    pixel = PIXEL[:-3] + 'e0701000 80040000'.split() + PIXEL[-3:]  # memory store in the paired stage
    expect('paired-stage store blocks the mesh load proof', request(tmp, count_guard_mesh(), pixel)[1], GUARDED_REFUSAL)
    for bit in (0, 1):
        _, guest, wrong = count_guard_reference(bit)
        expect('m3 rewritten clamp changes the CPU image, prefix arm %d' % bit, guest != wrong, True)
    count_guard_mutants(tmp)


def lane_mask_controls(tmp):
    for label, mesh in MASK_CASES + MASK_MEMORY + MASK_GENERIC + ROUND2_DRAWS:
        key, result = request(tmp, mesh)
        expect(label + ' translates', result, 'ok')
        if result == 'ok':
            source = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            for mutant, mutated in lane_mask_mutants(label, source):
                expect(label + ' mutant changes HLSL', mutated != source, True)
                expect(label + ' mutant changes a CPU image', any(
                    lane_mask_reference(label, full, mutant)[-1] != lane_mask_reference(label, full)[-1]
                    for full in (False, True)), True)
    for label, mesh in GUARDED_POSITIONS.items():
        key, result = request(tmp, mesh)
        expect('guarded positions with ' + label, result, 'ok' if label in ('select', 'same', 'unsigned export') else GUARDED_REFUSAL)
        if label == 'select' and result == 'ok':
            source = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            expect('export count mutant changes its source', widen_position_count(source) != source, True)
            _, good, bad = guarded_position_reference()
            expect('export count mutant changes the CPU image', good != bad, True)
    count_guard_controls(tmp)
    vector_count_guard_controls(tmp)
    count_entry_guard_controls(tmp)
    for label, mesh, reason in ROUND2_REFUSED:
        expect(label, reason in request(tmp, mesh)[1], True)
    for label, mesh, reason in MASK_CONTROLS:
        expect(label, request(tmp, mesh)[1], f'unsupported: {WAVE_LOAD_REFUSAL} (wave-load block: {reason})')

    mesh = MASK_GENERIC[0][1]
    bad = mutate('d5010007', '7e0e022f', mesh=mesh)  # v_mov_b32 v7, s47
    bad.remove('00b9e480')
    expect('generic skip reads the high half as a number', request(tmp, bad)[1], 'unsupported: ' + WAVE_LOAD_REFUSAL)

    # m3 ignores a numeric use before running the purity proof. The classifier's
    # c1 assertion would fail under this mutant, independently of the emitter.
    def compare_web(mesh):
        ins = G.decode(code(mesh))
        i = next(i for i, item in enumerate(ins) if item[2] == 's_not_b64')
        return 's[46:47]' in G.lane_mask_webs(ins).get(i, ())
    expect('c1 checks pair purity', compare_web(MASK_CONTROLS[0][1]), False)
    expect('m3 ignoring the whole-word read fails c1', compare_web(MASK_CASES[0][1]), True)


def coherent_controls(tmp):
    for label, shape, options, votes in (
            ('S5', 'S5', {}, (2, 0)), ('S5 full', 'S5', {'full': True}, (2, 0)),
            ('S6', 'S6', {}, (5, 0)), ('S6 full', 'S6', {'full': True}, (5, 0)),
            ('S9 scalar offset', 'S9', {}, (2, 0)), ('S9 full', 'S9', {'full': True}, (2, 0)),
            ('S7', 'S7', {}, (5, 0)), ('S7 nonzero votes', 'S7', {'nonzero': True}, (3, 2)),
            ('S8', 'S8', {}, (8, 0)), ('S5 position exports', 'S5', {'positions': True}, (2, 0)),
            ('S5 live mask', 'S5', {'live_mask': True}, (2, 0)),
            (COHERENT_VCC_DRAW, 'S5', {'vcc_before_compare': True}, (3, 0)),
            (COHERENT_VCC_DRAW + ' full', 'S5', {'vcc_before_compare': True, 'full': True}, (3, 0)),
            ('S5 numeric scalar pair', 'S5', {'numeric_pair': True}, (2, 0)),
            ('SCC move', 'SCC move', {}, (2, 1)),
            ('SCC branch', 'SCC branch', {}, (3, 0)),
            ('SCC delayed select', 'SCC delayed select', {}, (3, 1)),
            ('SCC move full', 'SCC move', {'full': True}, (2, 1)),
            ('SCC branch full', 'SCC branch', {'full': True}, (3, 0)),
            ('SCC delayed select full', 'SCC delayed select', {'full': True}, (3, 1))):
        key, result = request(tmp, coherent_mesh(shape, **options))
        expect(label, result, 'ok')
        if result != 'ok':
            continue
        if label == COHERENT_VCC_DRAW:
            print(label + ': translated')
        source = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
        expect(label + ' votes', (source.count('mesh_vote == 0u'), source.count('mesh_vote != 0u')), votes)
        if label.startswith('S7'):
            expect(label + ' counts VCC even under zero EXEC',
                   bool(re.search(r'mesh_first_active\(tid, pc == \d+ && \(vcc & 1u\) != 0u', source)), True)
        if shape == 'S9':
            expect(label + ' register-offset load reaches the colour export',
                   bool(re.search(r's46 = r0\.Load\(s46 \+ 0\);\s+s40 = s46;'
                                  r'.*?v20 = s40;.*?o\.param0 = float4\(F\(v20\),',
                                  source, re.S)), True)
        draw_label = label.removesuffix(' full')
        if draw_label in COHERENT_DRAWS:
            full = options.get('full', False)
            _, guest = coherent_reference(shape, full)
            for mutant, changed in lane_mask_mutants(draw_label, source):
                _, wrong = coherent_reference(shape, full, mutant)
                expect(label + ' ' + mutant + ' changes source and pixels', (changed != source, wrong != guest),
                       (True, True))
        if shape in SCC_AND:
            _, wrong = coherent_reference(shape, full, 'always true')
            expect(label + ' false wave changes pixels', guest != wrong, True)
            print(f'{label}: {result}')
    for label, mesh, votes in COHERENT_LEGACY_CASES:
        key, result = request(tmp, mesh)
        expect(label, result, 'ok')
        if result == 'ok':
            source = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            expect(label + ' votes', source.count('mesh_vote == 0u'), votes)
    for shape, control, reason in COHERENT_CONTROLS:
        mesh = coherent_mesh(shape, control=control)
        if control == 'constant-trip':
            ins = G.decode(code(mesh))
            offsets = {item[0]: i for i, item in enumerate(ins)}
            scc_edges = ((i, offsets[int(G.parse(item[3])[1][0].split('_')[1], 16)])
                         for i, item in enumerate(ins) if item[2] == 's_cbranch_scc1')
            latch, header = next((i, j) for i, j in scc_edges if j < i)
            expect('the control has a proven three-trip loop', G.mesh_constant_trip(ins, header, latch, offsets), (3,))
        expect(control, request(tmp, mesh)[1],
               f'unsupported: {WAVE_LOAD_REFUSAL} (wave-load block: {reason})')
    expect('one missing position arm',
           request(tmp, coherent_mesh('S5', positions=True, control='missing position'))[1], GUARDED_REFUSAL)

    reset_refusal = (
        f'unsupported: {WAVE_LOAD_REFUSAL} (wave-load block: an unsupported branch inside the block '
        '(s_cbranch_execz forward in the block); a scalar instruction the wave block cannot run once per wave)')
    exec_write = coherent_mesh('S5', vcc_before_compare=True, control='EXEC write before compare')
    outside_entry = coherent_mesh('S5', vcc_before_compare=True, control='entry after reset')
    expect('EXEC write before compare', request(tmp, exec_write)[1], reset_refusal)
    expect('entry after reset', request(tmp, outside_entry)[1], reset_refusal)

    # Disable the EXEC guard only in the reset walk, leaving the emitter and branch check intact.
    mutant_source, edits = re.subn(r'(?m)^        if writes_exec[^\n]+:', '        if False:',
                                   inspect.getsource(G.mesh_full_exec_reset))
    expect('the mutant removes one EXEC guard', edits, 1)
    mutant_scope = vars(G).copy()
    exec(mutant_source, mutant_scope)
    with patch.object(G, 'mesh_full_exec_reset', mutant_scope['mesh_full_exec_reset']):
        expect('ignoring EXEC writes makes the refusal control translate', request(tmp, exec_write)[1], 'ok')
        expect('the entry control still refuses', request(tmp, outside_entry)[1], reset_refusal)

    # m3 removes the numeric use from the proof's view of the program. The same
    # VCC purity assertion used by the block must change from refusal to acceptance.
    ins = G.decode(code(coherent_mesh('S7', control='numeric VCC')))
    vote = next(i for i, item in enumerate(ins) if item[2] == 's_cbranch_vccz')
    numeric_read = next(i for i, item in enumerate(ins) if item[3] == 'v_mov_b32_e32 v60, vcc_lo')
    mutant = ins.copy()
    mutant[numeric_read] = (*ins[numeric_read][:2], 's_nop', 's_nop 0')
    expect('m3 control sees impure VCC', 'vcc' in G.lane_mask_webs(ins).get(vote, ()), False)
    expect('m3 accepting the numeric VCC fails the purity control',
           'vcc' in G.lane_mask_webs(mutant).get(vote, ()), True)

    coherent_offset_proofs(tmp)
    coherent_descriptor_proofs(tmp)


def coherent_descriptor_proofs(tmp):
    for control in (None, 'compare VCC source', 'EXEC VCC source', 'mixed VCC source'):
        ins = G.decode(code(coherent_mesh('S9', control=control)))
        load = next(i for i, item in enumerate(ins) if item[3] == 's_buffer_load_dword s46, s[104:107], s46')
        reads = set()
        G.lane_mask_webs(ins, numeric_reads=reads)
        expect(f'{control} descriptor is a numeric mask read', load in reads, control is not None)

    source = inspect.getsource(G.lane_mask_webs)
    seed = "(pair == 'vcc' and i not in vcc_copies)"
    expect('VCC seed mutant site is unique', source.count(seed), 1)
    namespace = dict(vars(G))
    exec(compile(source.replace(seed, "pair == 'vcc'"), '<VCC seed mutant>', 'exec'), namespace)
    with patch.object(G, 'lane_mask_webs', namespace['lane_mask_webs']):
        _, result = request(tmp, coherent_mesh('S9'))
    expect('unconditional VCC seed breaks the descriptor copy fixture', result,
           f'unsupported: {WAVE_LOAD_REFUSAL} (wave-load block: a scalar load address is not wave-uniform)')


def coherent_offset_proofs(tmp):
    """Check the words at the load and the mask use, including the bypass edge."""
    for control in ('conditional address', 'partial mask'):
        ins = G.decode(code(coherent_mesh('S9', control=control)))
        add = next(i for i, item in enumerate(ins) if item[3] == 's_add_u32 s46, s38, 0x1e0')
        load = next(i for i, item in enumerate(ins) if item[3] == 's_buffer_load_dword s46, s[8:11], s46')
        _, writes, possible_writes, candidates = G.lane_mask_accesses(ins)
        first_mask = next(i for i in candidates if i < add and candidates[i] == 's[46:47]')
        cfg = G.lane_mask_cfg(ins)
        incoming = G.reaching_mask_words(cfg, writes, possible_writes, {'s46', 's47'})
        low = {first_mask, add} if control == 'conditional address' else {add}
        expect(f'{control} offset reaches the load', incoming[load], {'s46': low, 's47': {first_mask}})
        reads = set()
        webs = G.lane_mask_webs(ins, numeric_reads=reads)
        expect(f'{control} offset is a numeric mask read', load in reads, control == 'conditional address')
        if control == 'partial mask':
            expect('the partial pair has no mask web', 's[46:47]' in webs.get(add + 1, ()), False)

    # This mutant loses definitions that exist on just one incoming path.
    source = inspect.getsource(G.reaching_mask_words)
    merge = 'before[w] | after[w]'
    expect('reaching definitions mutant site is unique', source.count(merge), 1)
    namespace = dict(vars(G))
    exec(compile(source.replace(merge, 'before[w] & after[w]'), '<reaching definitions mutant>', 'exec'),
         namespace)
    prove = G.mesh_coherent_block
    accepted = []

    def record_proof(*args):
        proof = prove(*args)
        accepted.append(proof)
        return proof

    with patch.object(G, 'reaching_mask_words', namespace['reaching_mask_words']), \
            patch.object(G, 'mesh_coherent_block', record_proof):
        _, result = request(tmp, coherent_mesh('S9', control='conditional address'))
    expect('intersection mutant accepts the conditional mask offset in the block proof', bool(accepted), True)
    expect('emitter still refuses the conditional mask offset', result,
           f'unsupported: {WAVE_LOAD_REFUSAL} (wave-load block: a masked arm reads a scalar that is not wave-uniform)')


def mesh_root_layout_controls():
    # Read platform user data before the descriptor load overwrites s8.
    platform = MESH.copy()
    platform.insert(platform.index('7e280505'), '80188108')  # s_add_u32 s24, s8, 1
    try:
        G.MeshPipeline(code(platform), code(PIXEL), root())
    except G.Unsupported as refusal:
        expect('platform user data is undefined', str(refusal),
               'numeric use of s8 holding an entry value the mesh model does not define')
    else:
        expect('platform user data is undefined', 'accepted', 'refused')

    def layout_root(two_tables):
        params = [(0, 0, 0)] * (2 if two_tables else 1) + [(3, 0, 0), (1, 0, 0)]
        params[-1] = (1, 0, 24 + 12 * len(params))
        rts0 = (struct.pack('<6I', 2, len(params), 24, 0, 0, 0x400)
                + b''.join(struct.pack('<3I', *p) for p in params) + struct.pack('<3I', 0, 0, 1))
        return (b'DXBC' + bytes(16) + struct.pack('<4I', 1, 44 + len(rts0), 1, 36)
                + b'RTS0' + struct.pack('<I', len(rts0)) + rts0)

    mesh = MESH.copy()
    at = mesh.index('7e280505')
    mesh[at:at + 1] = [
        '7e3c0505',  # v_readfirstlane_b32 s30, v5
        'be920311',  # s_mov_b32 s18, s17 (second table pointer)
        'be930380',  # s_mov_b32 s19, 0 (pointer high word)
    ]
    mesh = [{'be950310': 'be950314',  # s_mov_b32 s21, s20 (root constant)
             'be960311': 'be960314'}.get(word, word) for word in mesh]  # s_mov_b32 s22, s20
    pipe = G.MeshPipeline(code(mesh), code(PIXEL), layout_root(True))
    expect('packed root constant', ('rootconst', 3, 0) in pipe.uds, True)
    expect('second table binding', 'srv 0 buf table 1 0' in pipe.manifest().splitlines(), True)
    try:
        G.MeshPipeline(code(mesh), code(PIXEL), layout_root(False))
    except G.Unsupported as refusal:
        expect('shorter root changes binding', str(refusal), 'numeric use of s21 holding ptr')
    else:
        expect('shorter root changes binding', 'accepted', 'refused')


def controls():
    os.environ['X2W_FXC'] = ''   # sources and manifest only; the draw mode compiles
    mesh_root_layout_controls()
    group_reference_controls()
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, HEAP_MESH, root_blob=heap_root())
        expect('heap branch translates', result, 'ok')
        if result == 'ok':
            pipe = G.MeshPipeline(code(HEAP_MESH), code(PIXEL), heap_root())
            text = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            manifest = (Path(tmp) / ('%016x.ms57.txt' % key)).read_text()
            expect('heap helper emitted', 'ResourceDescriptorHeap' in text and 'x2w_heap_load' in text, True)
            expect('heap manifest', 'dynamic_heap\n' in manifest, True)
            expect('heap profile', pipe.mesh_profile(), 'ms_6_6')
        profiles = []
        def fake_dxc(args, **kwargs):
            profiles.append(args[args.index('-T') + 1])
            Path(args[args.index('-Fo') + 1]).write_bytes(b'synthetic bytecode')
            return subprocess.CompletedProcess(args, 0, '', '')
        with patch.dict(os.environ, {'X2W_FXC': 'enabled'}), \
             patch.object(G, 'find_dxc', return_value='dxc'), \
             patch.object(G, 'has_complete_mesh57_bytecode', return_value=True), \
             patch('subprocess.run', side_effect=fake_dxc):
            expect('heap compile dispatch', request(tmp, HEAP_MESH, root_blob=heap_root())[1], 'ok')
        expect('heap compiler profiles', profiles, ['ms_6_6', 'ps_6_6'])
        lane_mask = HEAP_MESH.copy()
        lane_mask.insert(lane_mask.index('bf008080'), 'be98036a')  # s_mov_b32 s24, vcc_lo
        expect('lane-mask index refused', request(tmp, lane_mask, root_blob=heap_root())[1],
               'unsupported: slot 57 mesh: s_mov_b32 on a wave mask is not covered')
        # Isolate the existing uniformity guard: the fixture supplies the load, while the
        # proof result is made negative for its offset register.
        original_uniform = G.Stage.uniform_source
        with patch.object(G.Stage, 'uniform_source',
                          lambda stage, token: False if token == 's24' else original_uniform(stage, token)):
            expect('nonuniform heap index refused', request(tmp, HEAP_MESH, root_blob=heap_root())[1],
                   "unsupported: s_load with a register offset through ('resheap',)")
        high_word = HEAP_MESH.copy()
        high_word[high_word.index('be9b0380')] = 'be9b0381'  # s_mov_b32 s27, 1
        expect('nonzero heap high word refused', request(tmp, high_word, root_blob=heap_root())[1],
               "unsupported: s_load with a register offset through ('resheap',)")
        advanced = HEAP_MESH.copy()
        advanced.insert(advanced.index('f408070d'), '801aa01a')  # s_add_u32 s26, s26, 32
        # Keep the branch target after both loads after adding one instruction.
        advanced[advanced.index('bf840004')] = 'bf840005'
        expect('advanced heap refused', request(tmp, advanced, root_blob=heap_root())[1],
               "unsupported: s_load through s26: the pointer was advanced by an untracked offset")
        formatted = HEAP_MESH.copy()
        formatted[formatted.index('e03810c0', formatted.index('f408070d'))] = 'e00c10c0'
        # A formatted buffer load through a heap record translates since the heap format table exists.
        expect('formatted heap load translates', request(tmp, formatted, root_blob=heap_root())[1], 'ok')
    with tempfile.TemporaryDirectory() as group_tmp:
        key, group_result = request(group_tmp, GROUP_MESH)
        expect('numeric group index translates', group_result, 'ok')
        if group_result == 'ok':
            group_source = (Path(group_tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            expect('group index uses SV_GroupID', 'uint3 gid : SV_GroupID' in group_source and 'gid.x' in group_source, True)
        for label, variant in (('join', GROUP_JOIN), ('clamp', GROUP_CLAMP)):
            key, status = request(group_tmp, variant)
            expect('group %s translates' % label, status, 'ok')
            if status == 'ok':
                source = (Path(group_tmp) / ('%016x.ms57.hlsl' % key)).read_text()
                expect('group %s reads gid once into an SGPR' % label,
                       bool(re.search(r'\bs20 = gid\.x;', source)) and source.count('gid.x') == 1
                       and 'uint3 gid : SV_GroupID' in source, True)
        expect('non-exchanged first lane refused', request(group_tmp, GROUP_CONTROL)[1],
               'unsupported: slot 57 mesh: v7 is read where it is not defined for every lane (an entry value, or left different per wave by an exchange)')
    pipe = G.MeshPipeline(code(MESH), code(PIXEL), root())
    pipe.vs.entry.append('uint entry_group = gid.x;')
    expect('entry group read adds SV_GroupID', 'uint3 gid : SV_GroupID' in pipe.hlsl_ms(), True)
    pipe.vs.entry.pop()
    declarations = pipe.vs.declarations
    with patch.object(pipe.vs, 'declarations', side_effect=lambda: declarations() + ['    uint decl_group = gid.x;']):
        expect('declaration group read adds SV_GroupID', 'uint3 gid : SV_GroupID' in pipe.hlsl_ms(), True)
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH)
        expect('the fixture translates', result, 'ok')
        names = sorted(p.name[16:] for p in Path(tmp).iterdir() if not p.name.startswith('req_'))
        expect('published files', names, ['.ms57.hlsl', '.ms57.txt', '.ps57.hlsl'])
        expect('a translated request is not pending', G.pending_keys(tmp, False), [])
        manifest = (Path(tmp) / ('%016x.ms57.txt' % key)).read_text().splitlines()
        expect('mesh manifest identifies request key', [line for line in manifest if line.startswith('key ')],
               ['key %016x' % key])
        expect('paired manifest stage information',
               [line for line in manifest if line.startswith(('params ', 'targets ', 'ps '))],
               ['params 1', 'targets 0'])
        text = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
        expect('unused group index has no SV_GroupID', 'SV_GroupID' in text, False)
        body = [line.strip() for line in text[text.index('void main('):].splitlines()]
        want = ['if (tid == 0u) { lds[0] = s21; lds[1] = s22; }', 'GroupMemoryBarrierWithGroupSync();',
                's21 = min(lds[0], 128u);', 's22 = min(lds[1], 128u);', 'if (s22 == 0u) s21 = 0u;',
                'SetMeshOutputCounts(s22, s21);']
        at = body.index(want[0]) if want[0] in body else 0
        expect('one store, one barrier, clamped counts, then the allocation', body[at:at + len(want)], want)
        expect('one barrier', text.count('GroupMemoryBarrier'), 1)
        expect('one allocation', text.count('SetMeshOutputCounts('), 1)
        expect('group memory is the two count words', re.findall(r'groupshared[^;]*;', text), ['groupshared uint lds[2];'])
        expect('outputs follow the allocation',
               all(body.index(line) > body.index(want[-1]) for line in body if 'verts[tid]' in line or 'tris[tid]' in line), True)
    prefix = MESH[:1] + 'bf008080 bf850001 bf800000 bf800000'.split() + MESH[1:]
    for what, mesh, _ in BRANCH_CASES + COUNT_ENTRY_CASES + (('SCC branch before the group-index exchange', prefix, None),):
        with tempfile.TemporaryDirectory() as tmp:
            key, result = request(tmp, mesh)
            expect(what + ' translates', result, 'ok')
            if result == 'ok':
                text = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
                expect(what + ' uses forward block dispatch', ('if (pc ==' in text, '[loop]' in text), (True, False))
                expect(what + ' guards the position export', bool(re.search(
                    r'if \(exec & 1\) \{[^\n]*o\.pos = [^\n]*mesh_vertex = true;', text)), True)
    with tempfile.TemporaryDirectory() as tmp:
        lane_mask_controls(tmp)
        coherent_controls(tmp)
        key, result = request(tmp, WAVE_LOAD_MESH)
        expect('pre-exchange wave load translates', result, 'ok')
        if result == 'ok':
            text = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            expect('the branch uses the wave predicate captured at the compare',
                   ('wave_any = (tid & ~63u) < ' in text, 'pc = (!wave_any) ?' in text), (True, True))
            expect('the scalar buffer read is inside a branched fall-through',
                   bool(re.search(r'\[branch\] if \(pc == \d+\) \{[^}]*s40 = [^\n]*\.Load\(', text)), True)
        # #2643 kept its dead-scalar rule to the count-entry block. A scalar result that no path reads past the join
        # is invisible in every other execz block too: lanes that skip never read it, lanes that run it compute the
        # wave-uniform value. These two shapes were refusal controls of #2643 and translate now.
        for what, mesh in (
                ('dead scalar ALU in a different EXEC-zero block', vertex_red('7e0e02f2 bf880001 ' + DEAD_SCALAR)),
                ('an EXEC-zero branch that skips a dead scalar write',
                 mutate('7e0e02f2', 'bea80381', mesh=masked_vertex_red()))):
            expect(what + ' translates', request(tmp, mesh)[1], 'ok')
        # v_mov_b32 v8, s50 after the full EXEC restore reads the scalar the skipped arm wrote.
        live = masked_vertex_red(MASKED_DEAD_SCALAR.replace('befe04c1', 'befe04c1 7e100232'))
        expect('scalar write in a skipped block read after the join', refusal_head(request(tmp, live)[1]),
               'unsupported: ' + WAVE_LOAD_REFUSAL)
        for what, mesh in WAVE_LOAD_REFUSALS + WAVE_BLOCK_REFUSALS + NEST_REFUSALS + NEST_EXECZ_REFUSALS:
            expect(what, refusal_head(request(tmp, mesh)[1]), 'unsupported: ' + WAVE_LOAD_REFUSAL)
        for what, mesh in WAVE_LOAD_ACCEPTED + NEST_ACCEPTED + NEST_EXECZ_ACCEPTED:
            expect(what + ' translates', request(tmp, mesh)[1], 'ok')
        for what, mesh, reason in WAVE_LOAD_OTHER_REFUSALS + (NEST_LOOP, NEST_MASK):
            expect(what, request(tmp, mesh)[1], 'unsupported: ' + reason)
        expect('wave-uniform explicit SCC in the block',
               request(tmp, wave_load_mesh(WAVE_BLOCK_PREFIX, WAVE_BLOCK_BODY + ' beaf03fd'))[1], 'ok')
        for shape, label in enumerate(SCALAR_POSITION_SHAPES):
            key, result = request(tmp, scalar_position_mesh(shape))
            expect(label + ' translates', result, 'ok')
            if result == 'ok':
                source = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
                expect(label + ' has a mutant', scalar_position_mutant(shape, source) is not None, True)
                _, correct, wrong = scalar_position_images(shape, 10 if shape == 2 else 5)
                expect(label + ' mutant changes CPU coverage', correct != wrong, True)
        expect('scalar defined on both paths to an inner join translates',
               request(tmp, scalar_position_mesh(1, 'defined inner result'))[1], 'ok')
        for control, expected in (
            ('backward', 's_cbranch_execz skips code with effects outside active lanes '
                         '(wave-load block: a backward branch inside the block)'),
            ('outside', 's_cbranch_execz skips code with effects outside active lanes'),
            ('exchange', 'a branch enters or crosses a group-memory exchange'),
            ('enters block', 's_cbranch_execz skips code with effects outside active lanes'),
            ('nonuniform input', 'SCC of the count exchange is read'),
            ('undefined scalar input', 's_cbranch_execz skips code with effects outside active lanes'),
            ('undefined result', 's_cbranch_execz skips code with effects outside active lanes'),
            ('undefined inner result', 's_cbranch_execz skips code with effects outside active lanes'),
            ('unguarded export', 's_cbranch_execz skips code with effects outside active lanes'),
            ('unguarded store', 's_cbranch_execz skips code with effects outside active lanes'),
            ('mask select', 's_cbranch_execz skips code with effects outside active lanes'),
        ):
            result = request(tmp, scalar_position_mesh(1, control))[1]
            expect(control, result.startswith('unsupported: slot 57 mesh: ' + expected), True)
            if not result.startswith('unsupported: slot 57 mesh: ' + expected):
                print(result)
        key, result = request(tmp, WAVE_BLOCK_MESH)
        expect('pre-exchange wave block translates', result, 'ok')
        if result == 'ok':
            text = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            block = re.search(r'\[branch\] if \(pc == \d+\) \{(.*?)\n    \}', text, re.S)
            expect('the wave block is one branched fall-through', bool(block) and 'pc = (!wave_any) ?' in text, True)
            if block:
                expect('both scalar loads and the offset sum run inside it',
                       (block[1].count('.Load('), bool(re.search(r'\bs45 = ', block[1]))), (2, True))
        # The accepted mesh opcodes cannot produce a lane-varying numeric SGPR here.
        # Inject that analysis fact at the Stage boundary to exercise the uniformity refusal.
        uniform_source = G.Stage.uniform_source
        with patch.object(G.Stage, 'uniform_source', lambda stage, token: token != 's42' and uniform_source(stage, token)):
            expect('non-uniform wave load bound', refusal_head(request(tmp, WAVE_LOAD_MESH)[1]), 'unsupported: ' + WAVE_LOAD_REFUSAL)
        with patch.object(G.Stage, 'uniform_source', lambda stage, token: token != 's41' and uniform_source(stage, token)):
            expect('unproven wave load address', refusal_head(request(tmp, WAVE_LOAD_MESH)[1]), 'unsupported: ' + WAVE_LOAD_REFUSAL)
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, FORMATTED_MESH)
        expect('formatted mesh translates', result, 'ok')
        if result == 'ok':
            manifest = (Path(tmp) / ('%016x.ms57.txt' % key)).read_text().splitlines()
            expect('formatted mesh reads word 3 from its bound SRV',
                   [s.split()[2:] for s in manifest if s.startswith('udsrc ') and 'descword3' in s], [['descword3', '0']])
            expect('formatted mesh SRV', [s for s in manifest if s.startswith('srv ')], ['srv 0 buf table 1 0'])
    for what, mesh, reason in REFUSALS:
        with tempfile.TemporaryDirectory() as tmp:
            key, result = request(tmp, mesh)
            expect(what, result, 'unsupported: ' + reason)
            expect(what + ': only the refusal is published',
                   sorted(p.name for p in Path(tmp).iterdir() if not p.name.startswith('req_')), ['%016x.fail' % key])
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH, [])
        expect('the fixture translates without a pixel stage', result, 'ok')
        if result == 'ok':
            names = sorted(p.name[16:] for p in Path(tmp).iterdir() if not p.name.startswith('req_'))
            expect('mesh-only source and manifest', names, ['.ms57.hlsl', '.ms57.txt'])
            manifest = (Path(tmp) / ('%016x.ms57.txt' % key)).read_text().splitlines()
            expect('mesh-only manifest stage information',
                   [line for line in manifest if line.startswith(('params ', 'targets ', 'ps '))],
                   ['params 0', 'ps none'])
            text = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
            expect('mesh-only vertex fields', re.findall(r'struct VSOut \{([^}]*)\}', text), ['\n    float4 pos : SV_Position;\n'])
            expect('mesh-only shader omits parameter writes', re.findall(r'\bo\.[fh]?param\d+', text), [])
            expect('mesh-only shader writes position and indices', 'o.pos =' in text and 'tris[tid] =' in text, True)
            expect('mesh-only manifest is enough in source-only mode', G.pending_keys(tmp, False), [])
            paired, result = request(tmp, MESH)
            expect('paired request for polling translates', result, 'ok')
            paired_text = (Path(tmp) / ('%016x.ms57.hlsl' % paired)).read_text()
            expect('mesh-only shader keeps every mesh_vertex mark of the pair',
                   text.count('mesh_vertex = true'), paired_text.count('mesh_vertex = true'))
            # The poller needs a complete container envelope, without compiling a shader here.
            part = struct.pack('<4sI4s', b'DXIL', 4, b'test')
            bytecode = struct.pack('<4s16sHHIII', b'DXBC', bytes(16), 1, 0, 36 + len(part), 1, 36) + part
            os.environ.pop('X2W_FXC')
            mesh_bytecode = Path(tmp) / ('%016x.ms57.cso' % key)
            mesh_bytecode.write_bytes(bytecode)
            (Path(tmp) / ('%016x.ms57.cso' % paired)).write_bytes(bytecode)
            expect('only the pair missing its pixel bytecode is pending', G.pending_keys(tmp, False), [paired])
            mesh_bytecode.write_bytes(bytecode[:-1])
            expect('mesh-only result with truncated bytecode is pending', G.pending_keys(tmp, False), sorted([key, paired]))
            mesh_bytecode.unlink()
            expect('mesh-only result missing mesh bytecode is pending', G.pending_keys(tmp, False), sorted([key, paired]))
            os.environ['X2W_FXC'] = ''
    # exp null off, off, off, off done vm; s_endpgm: a pixel stage without a colour output.
    with tempfile.TemporaryDirectory() as tmp:
        expect('a depth-only pixel stage', request(tmp, MESH, 'f8001890 00000000 bf810000'.split())[1], 'ok')
    depth_only_controls()
    # The request path refuses a missing s_endpgm earlier; the walk takes the last instruction as the terminator
    # unread, so the pipeline refuses it too (s_nop in its place).
    try:
        G.MeshPipeline(code(MESH[:-1] + ['bf800000']), code(PIXEL), root())
        expect('a mesh program without s_endpgm', 'ok', 'refused')
    except G.Unsupported as refusal:
        expect('a mesh program without s_endpgm', str(refusal), 'slot 57 mesh program does not end with s_endpgm')


def assemble(mc, text):
    r = subprocess.run([mc, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=+wavefrontsize64,-wavefrontsize32', '-show-encoding'],
                       input=text, capture_output=True, text=True, check=True)
    data = b''.join(bytes(int(b, 16) for b in m.split(',')) for m in re.findall(r'encoding: \[([^\]]*)\]', r.stdout))
    return ['%08x' % word for word in struct.unpack('<%dI' % (len(data) // 4), data)]


def lane_mask_binary(mc):
    """Verify the opcodes and independently assemble the mixed/empty S1 draw."""
    expect('VCC select and EXEC write words',
           assemble(mc, 'v_cndmask_b32_e64 v60, 0, 1, vcc\ns_cmp_eq_u32 s6, 0\ns_cselect_b32 exec_lo, 0, -1'),
           (COHERENT_VCC_READ + ' ' + COHERENT_EXEC_WRITE).split())
    expect('coherent scalar offset words', assemble(mc, """
s_buffer_load_dword s38, s[8:11], s41 offset:0x1f0
s_waitcnt lgkmcnt(0)
s_and_b32 s38, s38, -4
s_add_u32 s46, s38, 0x1e0
s_buffer_load_dword s46, s[8:11], s46
s_waitcnt lgkmcnt(0)
s_mov_b32 s40, s46
v_mov_b32 v7, s40
v_cmp_gt_u32_e64 s[46:47], 128, v0
s_and_b64 exec, exec, s[46:47]
"""), ' '.join((COHERENT_OFFSET_SETUP, COHERENT_OFFSET_ADD, COHERENT_OFFSET_LOAD,
                 COHERENT_OFFSET_MASK, COHERENT_OFFSET_USE)).split())
    expect('descriptor copy words', assemble(mc, """
s_load_dwordx4 s[56:59], s[18:19], 0
s_waitcnt lgkmcnt(0)
s_mov_b64 vcc, s[58:59]
s_mov_b64 s[104:105], s[56:57]
v_cmp_gt_u32_e64 s[58:59], 128, v0
s_mov_b64 vcc, exec
"""), ' '.join((COHERENT_DESCRIPTOR_LOAD, COHERENT_DESCRIPTOR_HIGH, COHERENT_DESCRIPTOR_LOW,
                 COHERENT_DESCRIPTOR_MASK, 'beea047e')).split())
    # llvm-mc rejects the VCC alias in the tuple. Verify the other fields
    # against the adjacent aligned base and check the full operand in Python.
    load = COHERENT_DESCRIPTOR_READ.split()[:2]
    load[0] = '%08x' % ((int(load[0], 16) & ~0x3f) | (100 >> 1))
    expect('descriptor register-offset load words',
           assemble(mc, 's_buffer_load_dword s46, s[100:103], s46'), load)
    expect('standalone select words', assemble(mc, """
s_cmp_eq_u32 s6, 0
s_cselect_b64 vcc, -1, 0
v_cndmask_b32_e64 v7, 0, 1.0, vcc
"""), STANDALONE_SELECT.split())
    expect('mask SCC fixture words', assemble(mc, """
v_cmp_eq_u32_e32 vcc, 1, v0
s_and_b64 s[22:23], vcc, exec
s_and_b64 s[12:13], vcc, exec
s_and_b64 s[18:19], vcc, exec
s_mov_b32 s20, src_scc
s_bfe_u32 s22, s20, 0x10000
v_mov_b32 v78, 0
s_cselect_b32 s20, 1, 0
v_cvt_f32_u32_e32 v20, s20
"""), ' '.join((SCC_COMPARE, *SCC_AND.values(), SCC_MOVE, SCC_VALU, SCC_SELECT, SCC_RED)).split())
    expect('re-narrowed EXEC words',
           assemble(mc, 'v_cmp_gt_u32_e64 s[60:61], 64, v0\ns_and_b64 exec, s[54:55], s[60:61]'),
           'd4c4003c 000200c0 87fe3c36'.split())
    expect('guarded position load words', assemble(mc, """
buffer_load_dwordx4 v[50:53], v4, s[8:11], 0 offen
s_waitcnt vmcnt(0)
"""), GUARDED_POSITION_LOAD.split())
    expect('guarded position select words', assemble(mc, """
s_mov_b32 s42, s17
s_cmp_eq_u32 s16, 0
s_cselect_b32 s22, 0, s42
"""), dict(GUARDED_POSITION_SETUP)['select'].split())
    source = """
s_mov_b32 s40, 0x3e808081
s_lshl_b32 s41, s6, 2
s_mov_b32 s42, 128
v_mov_b32 v7, 0
s_mov_b64 exec, -1
v_cmpx_gt_u32_e32 s42, v0
s_mov_b32 s42, 0
s_cmp_eq_u32 s6, 0
s_cbranch_scc0 5
v_cmp_le_u32_e32 vcc, 0, v0
v_cmp_ge_u32_e64 s[46:47], v0, 6
s_and_b64 s[46:47], vcc, s[46:47]
s_branch 2
v_cmp_ge_u32_e64 s[46:47], v0, 0
s_not_b64 s[46:47], s[46:47]
s_and_saveexec_b64 s[46:47], s[46:47]
s_cbranch_execz 7
s_load_dwordx4 s[48:51], s[18:19], 0
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s40, s[48:51], s41 offset:0x1e0
s_waitcnt lgkmcnt(0)
v_mov_b32 v7, s40
s_andn2_b64 exec, s[46:47], exec
v_mov_b32 v7, 0.25
"""
    prefix, body = source.split('s_cmp_eq_u32', 1)
    expect('S1 assembled words', wave_load_mesh(' '.join(assemble(mc, prefix)),
           ' '.join(assemble(mc, 's_cmp_eq_u32' + body))), MASK_CASES[0][1])
    for label, asm, expected in (
            ('S5 CMPX and complement', 's_mov_b64 s[52:53], exec\nv_cmpx_gt_u32_e32 s50, v0\n'
             's_andn2_b64 exec, s[52:53], exec', 'beb4047e 7da80032 8afe7e34'),
            ('S6 narrowing and restore', 's_and_b64 exec, s[46:47], exec\ns_and_saveexec_b64 s[54:55], vcc\n'
             's_and_saveexec_b64 s[56:57], vcc\ns_mov_b64 exec, s[52:53]', '87fe7e2e beb6246a beb8246a befe0434'),
            ('tbuffer', 'tbuffer_load_format_x v7, off, s[8:11], 0 format:[BUF_FMT_32_FLOAT] offset:488',
             'e8b001e8 80020700'),
            ('image', 'image_load v7, v[50:51], s[56:63] dmask:0x1 dim:SQ_RSRC_IMG_2D',
             'f0000108 000e0732'),
            ('S1b SCC', 's_cmp_eq_u32 src_scc, 1', 'bf0681fd'),
            ('S3 masks', 's_cselect_b64 s[54:55], s[52:53], s[46:47]', '85b62e34'),
            ('S4 VCC branch', 's_and_b64 vcc, vcc, exec\ns_cbranch_vccnz 2', '87ea7e6a bf870002')):
        expect(label + ' assembled words', assemble(mc, asm), expected.split())


def binary(mc):
    expect('root layout control words', assemble(mc, 's_add_u32 s24, s8, 1\n'
           'v_readfirstlane_b32 s30, v5\ns_mov_b32 s18, s17\ns_mov_b32 s19, 0\n'
           's_mov_b32 s21, s20\ns_mov_b32 s22, s20'),
           '80188108 7e3c0505 be920311 be930380 be950314 be960314'.split())
    expect('heap setup and branch words', assemble(mc, """
s_mov_b32 s24, s20
s_mov_b32 s19, 0
s_mov_b32 s26, s21
s_mov_b32 s27, 0
s_and_b32 s24, s24, 0x3ff
s_lshl_b32 s24, s24, 5
s_cmp_eq_i32 0, 0
s_cbranch_scc0 4
s_load_dwordx4 s[28:31], s[26:27], s24
buffer_load_dwordx4 v[20:23], v4, s[28:31], 0 offen offset:192
"""), HEAP_SETUP + HEAP_ARM)
    lane_mask_binary(mc)
    mesh = (HERE / 'startup_mesh57_skeleton.s').read_text()
    expect('mesh words match llvm-mc', assemble(mc, mesh), MESH)
    expect('group words match llvm-mc', assemble(mc, mesh.replace(
        'buffer_load_dword v24, v4', 's_lshl_b32 s30, s20, 3\nv_add_co_u32 v4, vcc, s30, v4\nbuffer_load_dword v24, v4')),
        GROUP_MESH)
    group_asm = mesh.replace('s_mov_b32 s21, s16', 's_mov_b32 s21, 2').replace(
        'buffer_load_dword v24, v4',
        's_cmp_lt_u32 s20, s16\ns_cbranch_scc0 1\ns_mul_i32 s20, s20, src_scc\n'
        's_lshl_b32 s30, s20, 3\nv_add_co_u32 v4, vcc, s30, v4\nbuffer_load_dword v24, v4')
    expect('group join words match llvm-mc', assemble(mc, group_asm), GROUP_JOIN)
    expect('group clamp words match llvm-mc', assemble(mc, group_asm.replace('s_cbranch_scc0 1', 's_cbranch_scc1 1')),
           GROUP_CLAMP)
    expect('formatted mesh words match llvm-mc', assemble(mc, mesh.replace(COLOUR_LOAD, COLOUR_LOAD + '\n' + FORMAT_LOAD)),
           FORMATTED_MESH)
    expect('rewritten descriptor words match llvm-mc',
           assemble(mc, mesh.replace(COLOUR_LOAD, COLOUR_LOAD + '\ns_mov_b32 s11, 0\n' + FORMAT_LOAD)), REWRITTEN_FORMAT)
    expect('pixel words match llvm-mc', assemble(mc, PIXEL_ASM), PIXEL)
    shift = mesh.replace('v_cmpx_gt_i32 s22, v0', SHIFT_ASM + '\nv_cmpx_gt_i32 s22, v0')
    shift = shift.replace('exp pos0', 'v_mov_b32 v20, v7\nexp pos0')
    expect('scalar shift mesh words match llvm-mc', assemble(mc, shift), SHIFT_MESH)
    wave_load = mesh.replace('s_mov_b32 s21, s16', WAVE_LOAD_ASM + '\ns_mov_b32 s21, s16')
    wave_load = wave_load.replace('exp pos0', 'v_mov_b32 v20, s40\nv_mov_b32 v21, v7\nexp pos0')
    expect('pre-exchange wave load words match llvm-mc', assemble(mc, wave_load), WAVE_LOAD_MESH)
    wave_block = mesh.replace('s_mov_b32 s21, s16', WAVE_BLOCK_ASM + '\ns_mov_b32 s21, s16')
    wave_block = wave_block.replace('exp pos0', 'v_mov_b32 v20, s40\nv_mov_b32 v21, v7\nexp pos0')
    expect('pre-exchange wave block words match llvm-mc', assemble(mc, wave_block), WAVE_BLOCK_MESH)
    nest = mesh.replace('s_mov_b32 s21, s16', NEST_SCC_ASM + '\ns_mov_b32 s21, s16')
    nest = nest.replace('exp pos0', 'v_mov_b32 v20, s40\nv_mov_b32 v21, v7\nexp pos0')
    expect('nested wave block words match llvm-mc', assemble(mc, nest), NEST_SCC_MESH)
    nest_execz = mesh.replace('s_mov_b32 s21, s16', NEST_EXECZ_ASM + '\ns_mov_b32 s21, s16')
    nest_execz = nest_execz.replace('exp pos0', 'v_mov_b32 v20, s40\nv_mov_b32 v21, v7\nexp pos0')
    expect('nested if wave block words match llvm-mc', assemble(mc, nest_execz), NEST_EXECZ_MESH)
    for label, assembly, words in WAVE_COLLECTIVES:
        fixture = mesh.replace('s_mov_b32 s21, s16', assembly + '\ns_mov_b32 s21, s16')
        fixture = fixture.replace('exp pos0', 'v_mov_b32 v20, s40\nv_mov_b32 v21, v7\nexp pos0')
        expect(label + ' words match llvm-mc', assemble(mc, fixture), words)


# ---- draw: four triangles in a 32x32 target, vertices stored out of order ----------------------
SIDE = 32
TRIANGLES = (((2, 2), (14, 2), (2, 14)), ((18, 3), (30, 5), (21, 14)), ((3, 18), (13, 20), (6, 30)), ((18, 18), (30, 18), (30, 30)))
ORDER = (7, 2, 10, 5, 0, 8, 3, 11, 1, 6, 9, 4)   # ORDER[3 * triangle + corner] = vertex slot


def geometry():
    points, colours, indices = [None] * 12, [None] * 12, []
    for t, triangle in enumerate(TRIANGLES):
        for c, point in enumerate(triangle):
            slot = ORDER[3 * t + c]
            points[slot] = point
            colours[slot] = (40 + 17 * slot, 250 - 19 * slot, (slot * 53) % 256, 255)
        indices.append(tuple(ORDER[3 * t:3 * t + 3]))
    return points, colours, indices


def buffer(points, colours, words):
    data = bytearray(512)
    for slot, (x, y) in enumerate(points):
        struct.pack_into('<4f', data, 16 * slot, x / 16 - 1, 1 - y / 16, 0.5, 1)
        struct.pack_into('<4f', data, 192 + 16 * slot, *(channel / 255 for channel in colours[slot]))
    for t, word in enumerate(words):
        struct.pack_into('<I', data, 384 + 4 * t, word)
    return bytes(data)


def rasterise(points, colours, indices, verts, prims):
    """D3D rules: sample at the pixel centre, top-left fill, flat colour of the first vertex, later primitives on top.

    Counts above the 128 threads of a group are clamped and an index at or past the vertex count names the last
    vertex, as the lowering documents. Vertex `thread` holds slot `thread & 15`, as the fixture reads it.
    Coordinates are doubled so centres are integers."""
    verts, prims = min(verts, 128), min(prims, 128)
    image = [(0, 0, 0, 255)] * (SIDE * SIDE)
    if not verts:
        return image
    for a, b, c in indices[:prims]:
        a, b, c = (min(i, verts - 1) for i in (a, b, c))
        colour = colours[a % 16]
        tri = [(2 * points[i % 16][0], 2 * points[i % 16][1]) for i in (a, b, c)]
        area = (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) - (tri[1][1] - tri[0][1]) * (tri[2][0] - tri[0][0])
        if not area:
            continue
        if area < 0:
            tri[1], tri[2] = tri[2], tri[1]
        for y in range(SIDE):
            for x in range(SIDE):
                inside = True
                for k in range(3):
                    (ax, ay), (bx, by) = tri[k], tri[(k + 1) % 3]
                    dx, dy = bx - ax, by - ay
                    edge = dx * (2 * y + 1 - ay) - dy * (2 * x + 1 - ax)
                    inside &= edge > 0 or (edge == 0 and (dy < 0 or (dy == 0 and dx > 0)))
                if inside:
                    image[y * SIDE + x] = colour
    return image


def group_reference_controls():
    """Check that the runtime fixture's negative controls have distinct images."""
    points, colours, indices = geometry()
    full = rasterise(points, colours, indices, len(points), len(indices))
    group_zero = rasterise(points, colours, indices[:2], len(points), 2)
    no_constants = rasterise(points, colours, indices, 0, 0)
    clear = [(0, 0, 0, 255)] * (SIDE * SIDE)
    expect('one group differs from two groups', group_zero != full, True)
    expect('zero entry counts leave the target clear', no_constants, clear)
    expect('zero entry counts differ from two groups', no_constants != full, True)


def lane_mask_reference(label, full, mutant=None):
    """Simulate scalar state per guest wave and VALU under that lane's EXEC."""
    if label in COHERENT_DRAWS:
        return coherent_reference(COHERENT_DRAWS[label][0], full, mutant)
    points, colours, indices = geometry()
    triangles = [tuple(slot + first for slot in tri) for tri, first in zip(indices, (32, 0, 64, 64))]
    data = bytearray(buffer(points, colours, [a | b << 10 | c << 20 for a, b, c in triangles]))
    loaded = (255, 192)
    for i, red in enumerate(loaded):
        struct.pack_into('<f', data, 480 + 4 * i, red / 255)
    struct.pack_into('<f', data, 488, 1.0)

    memory = label in ('tbuffer', 'image')
    for first in (0, 64):
        limit = 6 if memory else 128
        if label == 'standalone wave-load select':
            limit = 6
        if label == 'S4 CMPX' and mutant != 'dropped CMPX':
            limit = 8
        outer = [tid < limit for tid in range(first, first + 64)]
        then = [active and (lane < 6 if first == 0 else full)
                for lane, active in enumerate(outer)]
        if mutant == 'dropped EXEC restore' and first == 0:
            outer = [lane < 4 for lane in range(64)]
            then = outer
        if mutant == 'wrong constant select' and first == 64:
            then = outer
        if mutant == 'wrong mask select':
            # s54 = 0 clears s46 & s54 in every wave, so its complement opens the then-arm to every outer lane
            # of both waves (wave 1 too, whose correct then-arm is empty), and v7 there becomes 0.0.
            then = outer
        if mutant == 'per-lane VCC branch' and first == 0:
            then = [active and lane == 0 for lane, active in enumerate(outer)]
        for tri in triangles:
            for tid in tri:
                if tid & ~63 != first:
                    continue
                lane, slot = tid & 63, tid & 15
                runs = then[lane] if mutant == 'per-lane EXEC branch' else any(then)
                red = loaded[first // 64] if runs else 64
                if label == 'standalone wave-load select':
                    green = 255 if tid < 6 and mutant != 'opposite SCC select' else 0
                    colours[slot] = (red, green, *colours[slot][2:])
                    continue
                if label.endswith('mask regions'):
                    green = 255 if tid < 3 else 64
                    colours[slot] = (red, green, *colours[slot][2:])
                    continue
                if label in ('standalone select', 'standalone carry select'):
                    red = 255 if (tid < 64) != (mutant == 'opposite SCC select') else 0
                    colours[slot] = (red, *colours[slot][1:])
                    continue
                if label.startswith('generic'):
                    red = 255 if tid < 6 or mutant == 'wrong generic mask' else 0
                    colours[slot] = (red, *colours[slot][1:])
                    continue
                if memory:
                    green = 255 if then[lane] or (mutant == 'unpredicated memory' and any(outer)) else 0
                else:
                    otherwise = (then[lane] if mutant == 'wrong complement' else outer[lane] and not then[lane])
                    green = 64 if otherwise else 0
                    if then[lane] and not otherwise and mutant != 'wrong mask select':
                        green = 255 if label == 'S3 mask arms' else loaded[first // 64]
                colours[slot] = (red, green, *colours[slot][2:])
    return data, rasterise(points, colours, triangles, 128, len(triangles))


def coherent_reference(shape, full, mutant=None):
    """Interpret the voted arms for two guest wave64s and rasterize their outputs."""
    points, colours, indices = geometry()
    triangles = [tuple(slot + base for slot in tri) for tri, base in zip(indices, (32, 0, 64, 64))]
    data = bytearray(buffer(points, colours, [a | b << 10 | c << 20 for a, b, c in triangles]))
    then_red, else_green = (255, 192), (64, 96)
    for wave in (0, 1):
        struct.pack_into('<f', data, 480 + 4 * wave, then_red[wave] / 255)
        struct.pack_into('<f', data, 488 + 4 * wave, else_green[wave] / 255)
        if shape == 'S9':
            struct.pack_into('<I', data, 496 + 4 * wave, 4 * wave)
        saved = [True] * 64
        threshold = 6 if wave == 0 else 128 if full else 64
        cut = (8 if wave == 0 else 128) if shape == 'S6' else threshold
        active = [e and 64 * wave + lane < cut for lane, e in enumerate(saved)]
        first_vote = active
        if shape == 'S6':
            # The second compare can empty the wave even when the first vote passed.
            vcc = [e and 64 * wave + lane < threshold for lane, e in enumerate(active)]
            for _ in range(2):
                if not any(active):
                    break
                active = [e and bit for e, bit in zip(active, vcc)]
        then_runs = any(active)
        loaded = then_red[1 if mutant == 'wrong scalar offset' else wave]
        red = loaded if then_runs else 64
        if shape in SCC_AND:
            mask_result = [e and 64 * wave + lane == 1 for lane, e in enumerate(active)]
            vote = mask_result[0] if mutant == 'lane 0 only' else any(mask_result)
            red = 255 * (vote or mutant == 'always true')
        green = [loaded if e else 0 for e in active]
        if mutant == 'wrong complement':
            other = active
        else:
            other = [s and not e for s, e in zip(saved, active)]
        if any(other):
            green = [else_green[wave] if e else old for e, old in zip(other, green)]
        # Restoring EXEC makes the wave scalar observable on every exported lane.
        for tid in {i for tri in triangles for i in tri if i // 64 == wave}:
            lane, slot = tid & 63, tid & 15
            result = 64 if mutant == 'per-lane EXEC branch' and not first_vote[lane] else red
            colours[slot] = (result, green[lane], *colours[slot][2:])
    return data, rasterise(points, colours, triangles, 128, len(triangles))


def lane_mask_mutants(label, source):
    """m1 makes the branch lane-local, m2 drops the saved complement."""
    if label in SCC_AND:
        word = {'SCC move': 's22', 'SCC branch': 's12', 'SCC delayed select': 's18'}[label]
        pattern = r'(mesh_first_active\(tid, pc == \d+ && )(\(' + word + r' & 1u\) != 0u)'
        changed, sites = re.subn(pattern, r'\1(tid & 63u) == 0u && \2', source)
        expect(label + ' mask SCC vote is unique', sites, 1)
        return [('lane 0 only', changed)]
    if label in COHERENT_DRAWS:
        vote = 'mesh_vote == 0u'
        start = source.find(vote) + len(vote) if label == COHERENT_VCC_DRAW else 0
        expect(label + ' inner EXEC vote is present', vote in source[start:], True)
        mutants = [('per-lane EXEC branch', source[:start] + source[start:].replace(vote, 'exec == 0u', 1))]
        if COHERENT_DRAWS[label][0] == 'S5':
            complement = 'exec = (s52 & ~exec) & 1;'
            expect(label + ' complement is unique', source.count(complement), 1)
            mutants.append(('wrong complement', source.replace(complement, 'exec = exec & 1;')))
        if label == 'S9':
            address = 's46 = r0.Load(s46 + 0);'
            expect(label + ' computed scalar load is unique', source.count(address), 1)
            mutants.append(('wrong scalar offset', source.replace(address, 's46 = r0.Load(484);')))
        return mutants
    if label.endswith('mask regions'):
        expect(label + ' has two votes', source.count('mesh_vote == 0u'), 2)
        return [('per-lane EXEC branch', source.replace('mesh_vote == 0u', 'exec == 0u', 1))]
    if label.startswith('standalone'):
        old, new = 'vcc = (scc ?', 'vcc = (!scc ?'
        expect(label + ' mutant site is unique', source.count(old), 1)
        return [('opposite SCC select', source.replace(old, new))]
    if label.startswith('generic'):
        old, new = (('s46 = 0x00000000u;', 's46 = 0xffffffffu;') if label == 'generic live' else
                    ('if (exec & 1) v7 = 0x00000000u;', 'if (exec & 1) v7 = 0x3f800000u;'))
        expect(label + ' mutant site is unique', source.count(old), 1)
        return [('wrong generic mask', source.replace(old, new))]
    if label in ('tbuffer', 'image'):
        start = source.index('if (exec & 1) {', source.index('s40 = r0.Load('))
        end = source.index('pc =', start)
        return [('unpredicated memory', source[:start] + source[start:end].replace('if (exec & 1)', 'if (true)')
                 + source[end:])]
    edits = [('per-lane EXEC branch', 'mesh_vote == 0u', 'exec == 0u'),
             ('wrong complement', 'exec = (s46 & ~exec) & 1;', 'exec = exec & 1;')]
    if label == 'S3':
        edits.append(('wrong constant select', 's52 = (scc ? 0x00000000u : 0xffffffffu) & 1;',
                      's52 = 0u;'))
    if label == 'S3 mask arms':
        edits.append(('wrong mask select', 's54 = (scc ? s52 : s46) & 1;', 's54 = 0u;'))
    if label == 'S4 EXEC moves':
        edits.append(('dropped EXEC restore', 'exec = (s52) & 1;', 'exec = exec & 1;'))
    if label == 'S4 CMPX':
        edits.append(('dropped CMPX', 'exec = (exec & ((0x00000008u > v0) ? 1u : 0u));',
                      'exec = exec & 1;'))
    if label == 'S4 VCC vote':
        edits.append(('per-lane VCC branch', 'mesh_vote != 0u', 'vcc != 0u'))
    for _, old, _ in edits:
        expect(label + ' mutant site is unique', source.count(old), 1)
    return [(name, source.replace(old, new)) for name, old, new in edits]


def draw_lane_masks(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    for label, mesh in MASK_CASES + MASK_MEMORY + MASK_GENERIC + ROUND2_DRAWS:
        # A group has two guest waves. A second S1 draw changes wave 1 from
        # empty to full so the reference exercises all three kinds of wave.
        programs = [(False, mesh)]
        if label == 'S1':
            programs.append((True, lane_mask_mesh(right='d4c1002e 00010100')))
        for full, program in programs:
            with tempfile.TemporaryDirectory() as tmp:
                status = draw_lane_mask_case(tmp, dxc, renderer, adapter, label, program, full)
                if status:
                    return status
    for label, (shape, options) in COHERENT_DRAWS.items():
        for full in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                status = draw_lane_mask_case(
                    tmp, dxc, renderer, adapter, label, coherent_mesh(shape, full=full, **options), full)
                if status:
                    return status
    with tempfile.TemporaryDirectory() as tmp:
        status = draw_guarded_positions(Path(tmp), dxc, renderer, adapter)
        if status:
            return status
    for vector_bound in (False, True):
        with tempfile.TemporaryDirectory() as tmp:
            status = draw_count_guard(Path(tmp), dxc, renderer, adapter, vector_bound=vector_bound)
            if status:
                return status
    return 0


def guarded_position_reference():
    """Extra vertex 10 is inactive in the defining region and has zero clip position in the mutant."""
    points, colours, _ = geometry()
    triangles = [(7, 2, 10), (5, 0, 8), (3, 4, 1), (6, 9, 4)]
    data = buffer(points, colours, [a | b << 10 | c << 20 for a, b, c in triangles])
    correct = rasterise(points, colours, triangles, 10, 4)
    # The guest clamps index 10 to 9. When 10 is exported as (0,0,0,0),
    # all nonzero homogeneous combinations in its triangle project onto the other edge.
    mutant = rasterise(points, colours, triangles[1:], 11, 3)
    return data, correct, mutant


def widen_position_count(source):
    """Widen the export predicate and its allocation together by one vertex."""
    old = 's22 = min(lds[1], 128u);'
    assert source.count(old) == 1
    return source.replace(old, 's22 = min(lds[1] + 1u, 128u);')


def draw_guarded_positions(tmp, dxc, renderer, adapter):
    mesh = GUARDED_POSITIONS['select']
    key, result = request(tmp, mesh)
    expect('guarded positions compile', result, 'ok')
    if result != 'ok':
        return 1
    base = tmp / ('%016x' % key)
    constants = bytearray(1024)
    uds = G.MeshPipeline(code(mesh), code(PIXEL), root()).uds
    for slot, count in enumerate((4, 10)):
        struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, slot)), count)
    data, correct, wrong = guarded_position_reference()
    count = sum(a != b for a, b in zip(correct, wrong))
    expect('widened export changes the CPU reference', count > 0, True)
    path = tmp / 'guarded.bin'
    path.write_bytes(constants + data + b'\xcd' * 80 + b''.join(map(bytes, correct)))
    mutant, mutant_text = tmp / 'wider.cso', tmp / 'wider.hlsl'
    mutant_text.write_text(widen_position_count(base.with_suffix('.ms57.hlsl').read_text()), encoding='ascii')
    subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', mutant, mutant_text], check=True)
    for name, shader in (('guarded positions', base.with_suffix('.ms57.cso')), ('widened export', mutant)):
        run = subprocess.run([renderer, adapter, '--mesh-indexed', shader, base.with_suffix('.ps57.cso'), path],
                             capture_output=True, text=True)
        if run.returncode == 77:
            return 77
        matches = re.search(r'pixels=1024 mismatches=(\d+)', run.stdout)
        got = int(matches[1]) if matches else None
        if shader == mutant:
            expect(name + ' differs from the guest image', run.returncode == 1 and got is not None and got > 0, True)
        else:
            expect(name + ' matches the guest image', (run.returncode, got), (0, 0))
    return 0


def count_guard_reference(bit, *, vector_bound=False):
    """Build the two synthetic prefix inputs and model the guest index clamp."""
    points, colours, _ = geometry()
    triangles = [(7, 2, 15)]
    data = bytearray(buffer(points, colours, []))
    struct.pack_into('<3I', data, 384, *triangles[0])
    # The set-bit arm obtains byte offset 0 from the indexed R32_FLOAT load.
    if vector_bound:
        # s17 = 2. The clear-bit arm selects (2 << 4) + 4 as the offset.
        # The set-bit arm loads B = 5 at offset 0, then scales it by s17.
        struct.pack_into('<I', data, 456, 0)
        struct.pack_into('<2I', data, 464, 4, 0)
        struct.pack_into('<2I', data, 500, 8, 0)
        raw_v, raw_p = struct.unpack_from('<2I', data, 464 if bit else 500)
        verts, prims = ((raw_v & 127) + 1) * (2 if bit else 1), raw_p + 1
    else:
        # s17 = 1. The clear-bit arm computes 16, and both fields come from one word.
        struct.pack_into('<I', data, 452, 0)
        struct.pack_into('<2I', data, 464, 9, 0)
        struct.pack_into('<2I', data, 480, 8, 0)
        packed = struct.unpack_from('<I', data, 464 if bit else 480)[0]
        verts, prims = (packed & 63) + 1, ((packed >> 8) & 255) + 1
    guest = [tuple(min(i, verts - 1) for i in tri) for tri in triangles]
    wrong = [tuple(min(i, verts - 2) for i in tri) for tri in guest]
    return data, rasterise(points, colours, guest, verts, prims), rasterise(points, colours, wrong, verts, prims)


def clamp_with_rewritten_count(source):
    """m3 substitutes the decremented SGPR for the published vertex count in the exporter."""
    old, new = 'max(mesh_vertex_count, 1u) - 1u', 'max(s22, 1u) - 1u'
    assert source.count(old) == 1
    return source.replace(old, new)


def draw_count_guard(tmp, dxc, renderer, adapter, *, vector_bound=False):
    mesh = vector_count_guard_mesh() if vector_bound else count_guard_mesh()
    bound = 'VGPR' if vector_bound else 'SGPR'
    key, result = request(tmp, mesh)
    expect(f'two wave-load guards with a {bound} position bound compile', result, 'ok')
    if result != 'ok':
        return 1
    base = tmp / ('%016x' % key)
    constants = bytearray(1024)
    uds = G.MeshPipeline(code(mesh), code(PIXEL), root()).uds
    for key, value in ((('rootconst', 0, 1), 2 if vector_bound else 1),
                       (('stride', 0), 4), (('descword3', 0), (22 << 12) | 4)):
        struct.pack_into('<I', constants, 4 * uds.index(key), value)
    source = base.with_suffix('.ms57.hlsl').read_text()
    mutant = tmp / 'rewritten-clamp.cso'
    mutant_text = mutant.with_suffix('.hlsl')
    mutant_text.write_text(clamp_with_rewritten_count(source), encoding='ascii')
    subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', mutant, mutant_text], check=True)
    path = tmp / 'count-guard.bin'
    for bit in (0, 1):
        struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, 0)), bit)
        data, guest, wrong = count_guard_reference(bit, vector_bound=vector_bound)
        differences = sum(a != b for a, b in zip(guest, wrong))
        expect('the index clamp affects the image', differences > 0, True)
        cases = (('guest', base.with_suffix('.ms57.cso'), guest, 0),
                 ('m3 against guest', mutant, guest, differences), ('m3 against mutant', mutant, wrong, 0))
        for label, shader, image, mismatches in cases:
            path.write_bytes(constants + data + b'\xcd' * 80 + b''.join(map(bytes, image)))
            run = subprocess.run([renderer, adapter, '--mesh-indexed', shader, base.with_suffix('.ps57.cso'), path],
                                 capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                return 77
            expect(f'{bound} bound, prefix arm {bit}, {label}',
                   (run.returncode, f'pixels=1024 mismatches={mismatches}' in run.stdout),
                   (int(mismatches != 0), True))
    return 0


def draw_lane_mask_case(tmp, dxc, renderer, adapter, label, mesh, full):
    key, result = request(tmp, mesh)
    expect(label + ' compiles', result, 'ok')
    if result != 'ok':
        return 1
    base = Path(tmp) / ('%016x' % key)
    source = base.with_suffix('.ms57.hlsl').read_text()
    data, correct = lane_mask_reference(label, full)
    cases = [(label, base.with_suffix('.ms57.cso'), correct, 0)]
    for i, (name, mutated) in enumerate(lane_mask_mutants(label, source)):
        expect(name + ' changes HLSL', mutated != source, True)
        _, wrong = lane_mask_reference(label, full, name)
        differences = sum(a != b for a, b in zip(correct, wrong))
        expect(name + ' differs from guest pixels', differences > 0, True)
        hlsl, shader = Path(tmp, f'mutant{i}.hlsl'), Path(tmp, f'mutant{i}.cso')
        hlsl.write_text(mutated)
        subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', shader, hlsl], check=True)
        cases += [(name, shader, correct, differences), (name + ' reference', shader, wrong, 0)]
    constants = bytearray(1024)
    uds = G.MeshPipeline(code(mesh), code(PIXEL), root()).uds
    for slot, count in enumerate((4, 128)):
        struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, slot)), count)
    path = Path(tmp, 'lane-mask.bin')
    command = '--mesh-texture-load' if label == 'image' else '--mesh-indexed'
    texture = bytes((255, 0, 0, 255)) * 4 if label == 'image' else b''
    for name, shader, image, differences in cases:
        path.write_bytes(constants + data + b'\xcd' * 80 + b''.join(map(bytes, image)) + texture)
        run = subprocess.run([renderer, adapter, command, shader, base.with_suffix('.ps57.cso'), path],
                             capture_output=True, text=True)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            return 77
        expect(name, (run.returncode, f'pixels=1024 mismatches={differences}' in run.stdout),
               (int(differences != 0), True))
    return 0


def draw_wave_load(directory, renderer, adapter, mesh=WAVE_LOAD_MESH, label='wave load', dxc=None, word=1.0, rule=None,
                   inner_mutant=None, green=None, nest_mutant=None, wave_mutants=True, bound=6):
    """Draw the mesh and compare. With dxc, also draw a mutant whose block branch skips per lane instead of per
    wave: it must match the per-lane image and differ from the correct one."""
    key, result = request(directory, mesh)
    expect('pre-exchange %s compiles' % label, result, 'ok')
    if result != 'ok':
        return 0
    base = Path(directory) / ('%016x' % key)
    points, colours, indices = geometry()
    # Provoking threads 39 and 5 share the mixed wave; threads 67 and 70 are
    # in the empty wave. Every vertex keeps its original low-four-bit slot.
    triangles = [tuple(slot + wave for slot in tri) for tri, wave in zip(indices, (32, 0, 64, 64))]
    words = [a | b << 10 | c << 20 for a, b, c in triangles]
    data = bytearray(buffer(points, colours, words))
    struct.pack_into('<f', data, 480, word)
    rule = rule or (lambda value: round(value * 255))   # the red a wave that runs the block ends with
    constants = bytearray(1024)
    uds = G.MeshPipeline(code(mesh), code(PIXEL), root()).uds
    for slot, count in enumerate((len(triangles), 128)):
        struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, slot)), count)

    old = 64
    green = green or (lambda tid: 255 if tid < bound else 0)   # the green a thread ends with (v7 exported)
    wave_red = {}
    reads = []
    for first in range(0, 128, 64):
        wave_red[first] = old
        if first < bound:
            address = 480 + 4 * first
            reads.append(address)
            wave_red[first] = rule(struct.unpack_from('<f', data, address)[0])
    expect('only the mixed wave reads; the empty wave address is past the SRV',
           (reads, len(data) + 80 < 480 + 4 * 64, wave_red), ([480], True, {0: rule(word), 64: 64}))

    def reference(swap):
        rendered = list(colours)
        for tri in triangles:
            for tid in tri:
                first, slot = tid & ~63, tid & 15
                red = wave_red[first ^ (64 if swap else 0)]
                rendered[slot] = (red, green(tid), *colours[slot][2:])
        return rasterise(points, rendered, triangles, 128, len(triangles))

    correct, swapped = reference(False), reference(True)
    different = sum(a != b for a, b in zip(correct, swapped))
    if rule(word) != old:   # a word that leaves the old red in place cannot tell the two waves apart
        expect('swapping old and loaded red changes every covered pixel', different,
               sum(pixel != (0, 0, 0, 255) for pixel in correct))
        expect('the wave load covers pixels', different > 0, True)
    path = Path(directory) / 'wave-load.bin'
    shader = base.with_suffix('.ms57.cso')
    cases = [(label, shader, correct, 0, 0)]
    if different:
        cases.append(('swapped %s selection' % label, shader, swapped, 1, different))
    if dxc:
        # Lanes outside EXEC in the mixed wave keep the old red when the block is skipped per lane.
        source = base.with_suffix('.ms57.hlsl').read_text()
        expect('one wave predicate on the block branch', source.count('pc = (!wave_any) ?'), 1)
        if wave_mutants:
            per_lane = reference_per_lane(points, colours, triangles, bound, old, wave_red)
            lanes = sum(a != b for a, b in zip(correct, per_lane))
            mutated = Path(directory) / 'per-lane.hlsl'
            mutated.write_text(source.replace('pc = (!wave_any) ?', 'pc = ((exec & 1) == 0) ?'))
            mutant = Path(directory) / 'per-lane.cso'
            subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', mutant, mutated], check=True, capture_output=True)
            if lanes:   # a word that leaves the old red in place cannot tell the per-lane block from the wave block
                cases += [('per-lane %s against the wave image' % label, mutant, correct, 1, lanes),
                          ('per-lane %s against its own image' % label, mutant, per_lane, 0, 0)]
        if nest_mutant:
            # The nested narrowing is dropped: every lane of the block takes the body.
            drop, keep, wrong_green = nest_mutant
            expect('one nested narrowing', source.count(drop), 1)
            nested_hlsl = Path(directory) / 'nested.hlsl'
            nested_hlsl.write_text(source.replace(drop, keep))
            nested_cso = Path(directory) / 'nested.cso'
            subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', nested_cso, nested_hlsl], check=True, capture_output=True)
            rendered = list(colours)
            for tri in triangles:
                for tid in tri:
                    first, slot = tid & ~63, tid & 15
                    rendered[slot] = (wave_red[first], wrong_green(tid), *colours[slot][2:])
            wrong_image = rasterise(points, rendered, triangles, 128, len(triangles))
            dropped = sum(a != b for a, b in zip(correct, wrong_image))
            expect('the dropped narrowing changes covered pixels', dropped > 0, True)
            cases += [('dropped nested narrowing %s against the correct image' % label, nested_cso, correct, 1, dropped),
                      ('dropped nested narrowing %s against its own image' % label, nested_cso, wrong_image, 0, 0)]
        if inner_mutant:
            # The inner branch is dropped: every wave that runs the block takes the loaded word.
            expect('one inner SCC branch', source.count('pc = (scc) ? 3 : 2;'), 1)
            inner_hlsl = Path(directory) / 'inner.hlsl'
            inner_hlsl.write_text(source.replace('pc = (scc) ? 3 : 2;', 'pc = 2;'))
            inner_cso = Path(directory) / 'inner.cso'
            subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', inner_cso, inner_hlsl], check=True, capture_output=True)
            wrong = {first: (inner_mutant(struct.unpack_from('<f', data, 480 + 4 * first)[0]) if first < bound else old)
                     for first in range(0, 128, 64)}
            rendered = list(colours)
            for tri in triangles:
                for tid in tri:
                    first, slot = tid & ~63, tid & 15
                    rendered[slot] = (wrong[first], 255 if tid < bound else 0, *colours[slot][2:])
            wrong_image = rasterise(points, rendered, triangles, 128, len(triangles))
            dropped = sum(a != b for a, b in zip(correct, wrong_image))
            cases += [('dropped inner branch %s against the correct image' % label, inner_cso, correct, 1 if dropped else 0, dropped)]
    for what, shader, image, status, mismatches in cases:
        path.write_bytes(bytes(constants) + bytes(data) + b'\xcd' * 80 + b''.join(map(bytes, image)))
        run = subprocess.run([renderer, adapter, '--mesh-indexed', shader,
                              base.with_suffix('.ps57.cso'), path], capture_output=True, text=True)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            return 77
        expect(what, (run.returncode, f'pixels=1024 mismatches={mismatches}' in run.stdout), (status, True))
        print('%s: %d pixels differ from the reference' % (what, mismatches))
    return 0


def scalar_position_images(shape, probe):
    points, colours, indices = geometry()
    # The first two triangles use mixed/active lanes of wave 0. The other two
    # use wave 1, which takes the outer skip.
    triangles = [tuple(slot + wave for slot in tri) for tri, wave in zip(indices, (0, 0, 64, 64))]
    words = [a | b << 10 | c << 20 for a, b, c in triangles]
    data = bytearray(buffer(points, colours, words))
    struct.pack_into('<II', data, 480, 0x1f000000, probe)  # shifting the first word gives float 1/8, or two pixels

    def image(limit):
        displaced = points.copy()
        for tri in triangles:
            for tid in tri:
                if tid < limit:
                    x, y = points[tid]
                    displaced[tid] = (x + 2, y)
        return rasterise(displaced, colours, triangles, 128, len(triangles))

    limit = 19 if shape == 0 else 8
    correct = image(limit if shape < 2 or probe < 8 else 0)
    wrong = image(0 if shape == 0 else 19 if shape == 1 else 8)
    return data, correct, wrong


def scalar_position_mutant(shape, source):
    if shape < 2:
        original, replacement = (('v7 = s48;', 'v7 = 0u;') if shape == 0 else
                                 ('exec = (exec & vcc) & 1;', 'exec = exec & 1;'))
        count = source.count(original)
        expect('one position mutation site', count, 1)
        return source.replace(original, replacement) if count == 1 else None
    # The EXEC vote comes first. Force the VCC branch to fall through even
    # when the compare mask is empty.
    votes = list(re.finditer(r'pc = \(mesh_vote == 0u\) \? \d+ : (\d+);', source))
    expect('two inner votes', len(votes), 2)
    if len(votes) != 2:
        return None
    vcc_vote = votes[1]
    return source[:vcc_vote.start()] + 'pc = %s;' % vcc_vote[1] + source[vcc_vote.end():]


def draw_scalar_positions(directory, dxc, renderer, adapter):
    for shape, probe in ((0, 5), (1, 5), (2, 5), (2, 10)):
        label = '%s (compare index %d)' % (SCALAR_POSITION_SHAPES[shape], probe)
        mesh = scalar_position_mesh(shape)
        key, result = request(directory, mesh)
        expect(label + ' compiles', result, 'ok')
        if result != 'ok':
            return 1
        base = Path(directory) / ('%016x' % key)
        data, correct, wrong = scalar_position_images(shape, probe)
        cases = [(base.with_suffix('.ms57.cso'), correct, 0, 0)]
        if shape < 2 or probe == 10:
            source = base.with_suffix('.ms57.hlsl').read_text()
            mutated = scalar_position_mutant(shape, source)
            if mutated is None:
                return 1
            different = sum(a != b for a, b in zip(correct, wrong))
            expect(label + ' mutant changes covered pixels', different > 0, True)
            mutant_hlsl = base.with_suffix('.mutant.hlsl')
            mutant_hlsl.write_text(mutated)
            mutant_cso = base.with_suffix('.mutant.cso')
            subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', mutant_cso, mutant_hlsl],
                           check=True, capture_output=True)
            cases += [(mutant_cso, correct, 1, different), (mutant_cso, wrong, 0, 0)]
        constants = bytearray(1024)
        uds = G.MeshPipeline(code(mesh), code(PIXEL), root()).uds
        for slot, count in enumerate((4, 128)):
            struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, slot)), count)
        fixture = base.with_suffix('.bin')
        for shader, image, status, mismatches in cases:
            fixture.write_bytes(constants + data + b'\xcd' * 80 + b''.join(map(bytes, image)))
            run = subprocess.run([renderer, adapter, '--mesh-indexed', shader, base.with_suffix('.ps57.cso'),
                                  fixture], capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                return 77
            expect(label + ' ' + shader.suffix + ' readback',
                   (run.returncode, f'pixels=1024 mismatches={mismatches}' in run.stdout), (status, True))
    return 0


def draw_wave_collectives(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    cases = (
        (INNER_VCC_MESH, 'inner VCC branch falls through', {}),
        (INNER_VCC_MESH, 'inner VCC branch taken', {'word': 0.0, 'rule': lambda _: 64}),
        (NO_RESET_MESH, 'no full EXEC reset', {'bound': 3}),
        (FIRST_LANE_MESH, 'readfirstlane broadcast',
         {'word': 0.25, 'rule': lambda _: 255, 'green': lambda tid: 255 if 0 < tid < 6 else 0}),
    )
    with tempfile.TemporaryDirectory() as tmp:
        for mesh, label, options in cases:
            status = draw_wave_load(tmp, renderer, adapter, mesh, label, **options)
            if status:
                return status
    return 0


def reference_per_lane(points, colours, triangles, bound, old, wave_red):
    """The image when each lane skips the block by its own EXEC bit: only threads below the bound load."""
    rendered = list(colours)
    for tri in triangles:
        for tid in tri:
            slot = tid & 15
            red = wave_red[tid & ~63] if tid < bound else old
            rendered[slot] = (red, 255 if tid < bound else 0, *colours[slot][2:])
    return rasterise(points, rendered, triangles, 128, len(triangles))


def publish_request(directory, publisher, adapter, mesh=MESH):
    ms, ps = code(mesh), code(PIXEL)
    (directory / 'ms.bin').write_bytes(ms)
    (directory / 'ps.bin').write_bytes(ps)
    env = dict(os.environ, X2W_MESH_REQUEST_FIXTURE='1', X2W_TEST_MESH57_EXCHANGE=str(directory))
    result = subprocess.run([publisher, adapter, 'mesh'], env=env, capture_output=True, text=True)
    print(result.stdout, end='')
    if result.returncode:
        if result.returncode != 77:
            print(result.stderr, end='')
        return result.returncode, None
    cache = directory / 'requests'
    pair = [path for path in cache.glob('req_*.ps57') if path.stat().st_size]
    expect('one published mesh and pixel request', len(pair), 1)
    if len(pair) != 1:
        return 1, None
    root_blob = pair[0].with_suffix('.root').read_bytes()
    key = G.mesh57_key(ms, ps, root_blob)
    expect('published key matches the offline reader', pair[0].stem, f'req_{key:016x}')
    expect('published mesh matches the fixture', pair[0].with_suffix('.m57').read_bytes(), ms)
    expect('published pixel matches the fixture', pair[0].read_bytes(), ps)
    if FAILED:
        return 1, None
    return 0, (cache, key, root_blob)


def runtime(dxc, renderer, adapter, publisher):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = geometry()
    words = [a | b << 10 | c << 20 for a, b, c in indices]
    correct = b''.join(map(bytes, rasterise(points, colours, indices, len(points), len(indices))))
    group_zero = b''.join(map(bytes, rasterise(points, colours, indices[:2], len(points), 2)))
    clear = bytes((0, 0, 0, 255)) * (SIDE * SIDE)
    assert correct != group_zero and correct != clear
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        status, published = publish_request(directory, publisher, adapter, GROUP_MESH)
        if status:
            return status
        cache, key, _ = published
        data = directory / 'data.bin'
        arguments = directory / 'args.bin'
        pixels = directory / 'pixels.bin'
        log = directory / 'runtime.log'
        data.write_bytes(buffer(points, colours, words))
        env = dict(os.environ, X2W_MESH_REQUEST_FIXTURE='1', X2W_TEST_MESH57_EXCHANGE=str(directory),
                   X2W_TEST_MESH57_RUNTIME='1', X2W_TEST_MESH57_DATA=str(data),
                   X2W_TEST_MESH57_ARGS=str(arguments), X2W_TEST_MESH57_PIXELS=str(pixels),
                   X2W_TRAP_LOG=str(log))
        def execute(groups, expected, no_constants=False, reason=None):
            arguments.write_bytes(struct.pack('<3I', *groups))
            pixels.unlink(missing_ok=True)
            log.unlink(missing_ok=True)
            case_env = dict(env)
            if no_constants:
                case_env['X2W_TEST_MESH57_NO_CONSTANTS'] = '1'
            result = subprocess.run([publisher, adapter, 'mesh'], env=case_env, capture_output=True, text=True)
            if result.returncode == 77:
                return 77
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            actual = pixels.read_bytes()
            if actual != expected:
                raise AssertionError(f'mesh runtime {groups}: image differs at '
                                     f'{sum(a != b for a, b in zip(actual, expected))} bytes')
            if reason and (not log.exists() or reason not in log.read_text(errors='replace')):
                raise AssertionError(f'mesh runtime did not log {reason}')
            return 0
        if execute((2, 1, 1), clear, reason='slot57 mesh translation pending') == 77:
            return 77
        if G.translate_request(str(cache), key) != 'ok':
            raise AssertionError('mesh request translation failed')
        manifest = cache / f'{key:016x}.ms57.txt'
        original = manifest.read_text()
        manifest.write_text(original.replace(f'key {key:016x}', 'key 0000000000000000'))
        if execute((2, 1, 1), clear, reason='manifest key mismatch') == 77:
            return 77
        manifest.write_text(original)
        for groups, expected, missing in (((2, 1, 1), correct, False), ((1, 2, 1), correct, False),
                                          ((1, 1, 1), group_zero, False), ((2, 1, 1), clear, True)):
            if execute(groups, expected, missing) == 77:
                return 77
    return 0


def draw(dxc, renderer, adapter, publisher=None):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH, [])
        expect('the fixture compiles without a pixel stage', result, 'ok')
        names = sorted(p.name[16:] for p in Path(tmp).iterdir() if not p.name.startswith('req_'))
        expect('mesh-only compiled files', names, ['.ms57.cso', '.ms57.hlsl', '.ms57.txt'])
        expect('complete mesh-only bytecode was published',
               G.has_complete_mesh57_bytecode(Path(tmp) / ('%016x.ms57.cso' % key)), True)
    points, colours, indices = geometry()
    with tempfile.TemporaryDirectory() as tmp:
        if publisher:
            status, published = publish_request(Path(tmp), publisher, adapter)
            if status:
                return status
            cache, key, root_blob = published
            result = G.translate_request(str(cache), key)
        else:
            cache, root_blob = Path(tmp), root()
            key, result = request(tmp, MESH)
        expect('the fixture translates and compiles', result, 'ok')
        if result != 'ok':
            return 0
        name = str(cache / ('%016x' % key))
        uds = G.MeshPipeline(code(MESH), code(PIXEL), root_blob).uds
        # (vertex count, primitive count, thread of vertex slot 0, flags of the first primitive words): all, fewer
        # primitives, fewer vertices (indices clamp), no vertex; every thread, with the triangles on vertices
        # 112..123 (wave 1, lanes 48..59), each named by eight primitives and the other 96 primitives on vertex 0
        # alone, the first triangle null (bit 31) and the second with its three edge flags (bits 9, 19, 29).
        # Each case is one process and one device, which takes seconds on a hardware adapter.
        cases = ((12, 4, 0, ()), (12, 1, 0, ()), (7, 4, 0, ()), (0, 4, 0, ()),
                 (128, 128, 112, (1 << 31, 1 << 9 | 1 << 19 | 1 << 29)))
        if publisher:
            cases = cases[:1]
        for case in cases:
            verts, prims, first, flags = case
            named = [tuple(first + i for i in triangle) for triangle in indices]
            words = [a | b << 10 | c << 20 | flag for (a, b, c), flag in zip(named, flags + (0,) * 4)]
            drawn = [(0, 0, 0) if word >> 31 else triangle for triangle, word in zip(named, words)]
            constants = bytearray(1024)
            struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, 0)), prims)
            struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, 1)), verts)
            image = rasterise(points, colours, drawn, verts, prims)
            head = bytes(constants) + buffer(points, colours, words) + b'\xcd' * 80
            path = Path(tmp) / 'draw.bin'
            path.write_bytes(head + b''.join(bytes(pixel) for pixel in image))
            args = [renderer, adapter, '--mesh-indexed', name + '.ms57.cso', name + '.ps57.cso', str(path)]
            r = subprocess.run(args, capture_output=True, text=True)
            if r.returncode == 77:
                print(r.stdout + r.stderr)
                return 77
            covered = sum(pixel != (0, 0, 0, 255) for pixel in image)
            what = 'vertices=%d primitives=%d first=%d flags=%s' % (verts, prims, first, ['%08x' % flag for flag in flags])
            print('%s covered=%d: %s' % (what, covered, r.stdout.strip().splitlines()[-1:]))
            expect('draw of ' + what,
                   (r.returncode, 'pixels=1024 mismatches=0' in r.stdout), (0, True))
            if case is cases[0]:
                expect('the full case covers pixels', covered > 200, True)
                wrong = list(image)
                at = next(i for i, pixel in enumerate(image) if pixel != (0, 0, 0, 255))
                wrong[at] = (0, 0, 0, 255)
                path.write_bytes(head + b''.join(bytes(pixel) for pixel in wrong))
                r = subprocess.run(args, capture_output=True, text=True)
                expect('a reference with one wrong pixel fails',
                       (r.returncode, 'pixels=1024 mismatches=1' in r.stdout, 'Draw reference mismatch' in r.stderr),
                       (1, True, True))
        for what, mesh, red in BRANCH_CASES + COUNT_ENTRY_CASES:
            key, result = request(tmp, mesh)
            expect(what + ' translates and compiles', result, 'ok')
            if result != 'ok':
                continue
            name = str(Path(tmp) / ('%016x' % key))
            words = [a | b << 10 | c << 20 for a, b, c in indices]
            struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, 0)), 4)
            struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, 1)), 12)
            recoloured = [(red(lane), *colour[1:]) for lane, colour in enumerate(colours)]
            image = rasterise(points, recoloured, indices, 12, 4)
            path.write_bytes(bytes(constants) + buffer(points, colours, words) + b'\xcd' * 80
                             + b''.join(bytes(pixel) for pixel in image))
            r = subprocess.run([renderer, adapter, '--mesh-indexed', name + '.ms57.cso', name + '.ps57.cso', str(path)],
                               capture_output=True, text=True)
            if r.returncode == 77:
                print(r.stdout + r.stderr)
                return 77
            expect('draw of ' + what, (r.returncode, 'pixels=1024 mismatches=0' in r.stdout), (0, True))
        if publisher is None:
            status = draw_wave_load(tmp, renderer, adapter)
            if status:
                return status
            status = draw_wave_load(tmp, renderer, adapter, WAVE_BLOCK_MESH, 'wave block', dxc)
            if status:
                return status
            nest = lambda value: 64 if value == 0 else round(value * 255)   # the inner branch keeps the old red on a zero word
            for word in (1.0, 0.0):
                status = draw_wave_load(tmp, renderer, adapter, NEST_SCC_MESH, 'nested scc block (word %g)' % word, dxc,
                                        word=word, rule=nest, inner_mutant=lambda value: round(value * 255))
                if status:
                    return status
            # The nested if: threads below 3 run the body (v7 = 1.0), threads 3..5 of the same block keep v7 = 0.
            nested_green = lambda tid: 255 if tid < 3 else 0
            status = draw_wave_load(tmp, renderer, adapter, NEST_EXECZ_MESH, 'nested if block', dxc, green=nested_green,
                                    rule=lambda value: 64, wave_mutants=False,   # s40 keeps its old red in every wave
                                    nest_mutant=('exec = (exec & vcc) & 1;', 'exec = exec & 1;', lambda tid: 255 if tid < 6 else 0))
            if status:
                return status
            return draw_scalar_positions(tmp, dxc, renderer, adapter)
    return 0


# ---- depth: the depth-only pair drawn twice into a D32 target cleared to 1.0, depth test LESS ------------------
# Per triangle and draw: (depth, kept). A kept triangle has attribute 0 red 1.0 at its three vertices, a killed one
# 0.25, so the depth-only pixel stage keeps or discards all of its pixels. The four triangles cover: the second draw
# behind the first, the second in front, a first draw killed, both killed.
DEPTH_DRAWS = (((0.25, True), (0.75, True), (0.125, False), (0.5, False)),
               ((0.75, True), (0.375, True), (0.625, True), (0.875, False)))


def depth_scene(points, indices, draws, honour_kill=True):
    """The upload of each draw and the expected colour and depth images (cleared to black and 1.0)."""
    uploads, depth = [], [1.0] * (SIDE * SIDE)
    words = [a | b << 10 | c << 20 for a, b, c in indices]
    for draw in draws:
        data = bytearray(buffer(points, [(0, 0, 0, 0)] * 12, words))
        for t, (z, kept) in enumerate(draw):
            for slot in indices[t]:
                struct.pack_into('<f', data, 16 * slot + 8, z)
                struct.pack_into('<f', data, 192 + 16 * slot, 1.0 if kept else 0.25)
            if not (kept or not honour_kill):
                continue
            covered = rasterise(points, [(1, 1, 1, 1)] * 12, [indices[t]], 12, 1)
            for i, pixel in enumerate(covered):
                if pixel == (1, 1, 1, 1) and z < depth[i]:
                    depth[i] = z
        uploads.append(bytes(data))
    colour = b''.join(bytes((0, 0, 0, 255)) for _ in range(SIDE * SIDE))
    return uploads, colour, b''.join(struct.pack('<f', z) for z in depth)


def draw_depth(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, _, indices = geometry()
    uploads, colour, depth = depth_scene(points, indices, DEPTH_DRAWS)
    _, _, unkilled = depth_scene(points, indices, DEPTH_DRAWS, honour_kill=False)
    cleared = struct.pack('<f', 1.0) * (SIDE * SIDE)
    pixels = lambda a, b: sum(a[i:i + 4] != b[i:i + 4] for i in range(0, len(a), 4))
    written, killed = pixels(depth, cleared), pixels(depth, unkilled)
    assert written > 100 and killed > 50, (written, killed)
    # Depth is cleared to the sentinel 1.0 and every drawn depth is below it, so a discarded pixel keeps the sentinel.
    covered = pixels(unkilled, cleared)
    print('covered %d pixels: %d keep a written depth, %d keep the sentinel only because of the discard, '
          '%d differ from a draw without the discard' % (covered, written, covered - written, killed))
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH, DEPTH_PIXEL)
        expect('the depth-only pair translates and compiles', result, 'ok')
        # The same mesh with a null export and no discard: what a pass without the pixel stage would draw.
        bare, bare_result = request(tmp, MESH, 'f8001890 00000000 bf810000'.split())
        expect('the pixel stage without discard translates and compiles', bare_result, 'ok')
        if result != 'ok' or bare_result != 'ok':
            return 0
        mrtz, mrtz_result = request(tmp, MESH, MRTZ_KILL_EXIT_PIXEL)
        expect('the depth-export pair behind a kill translates and compiles', mrtz_result, 'ok')
        plain_z, plain_z_result = request(tmp, MESH, MRTZ_PLAIN_PIXEL)
        expect('the depth-export pair without a kill translates and compiles', plain_z_result, 'ok')
        if mrtz_result != 'ok' or plain_z_result != 'ok':
            return 0
        names = sorted(p.name[16:] for p in Path(tmp).iterdir() if p.name.startswith('%016x' % key))
        expect('depth-only compiled files', names, ['.ms57.cso', '.ms57.hlsl', '.ms57.txt', '.ps57.cso', '.ps57.hlsl'])
        manifest = (Path(tmp) / ('%016x.ms57.txt' % key)).read_text().splitlines()
        uds = {tuple(fields[2:]): int(fields[1]) for fields in map(str.split, manifest) if fields and fields[0] == 'udsrc'}
        constants = bytearray(1024)
        struct.pack_into('<I', constants, 4 * uds['rootconst', '0', '0'], len(indices))
        struct.pack_into('<I', constants, 4 * uds['rootconst', '0', '1'], len(points))
        head = b''.join(bytes(constants) + upload for upload in uploads) + b'\xcd' * 160
        path = Path(tmp) / 'depth.bin'
        stems = {'depth-only pair': key, 'pixel stage without discard': bare, 'depth export behind a kill': mrtz,
                 'depth export without a kill': plain_z}
        half = struct.pack('<f', 0.5)
        # The depth export writes the constant 0.5 where the pair draws and keeps the sentinel where the kill discards.
        exported = b''.join(half if depth[i:i + 4] != cleared[i:i + 4] else cleared[i:i + 4] for i in range(0, len(depth), 4))
        # (label, pair, reference depth, depth mismatches). The colour target must stay cleared in every case.
        cases = (('baseline', 'depth-only pair', depth, 0),
                 ('a reference without the depth writes', 'depth-only pair', cleared, written),
                 ('the discard dropped', 'pixel stage without discard', depth, killed),
                 ('depth export behind a kill', 'depth export behind a kill', exported, 0),
                 ('depth export behind a kill, reference without the depth writes', 'depth export behind a kill', cleared, written),
                 ('depth export without the kill', 'depth export without a kill', exported, covered - written))
        for label, pair, ref_depth, depth_errors in cases:
            name = str(Path(tmp) / ('%016x' % stems[pair]))
            path.write_bytes(head + colour + ref_depth)
            run = subprocess.run([renderer, adapter, '--mesh-depth-pair', name + '.ms57.cso', name + '.ps57.cso', str(path)],
                                 capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                return 77
            lines = set(run.stdout.splitlines())
            message = 'Depth reference mismatch' if depth_errors else None
            want = {'mesh_exports bytes=160 matched=1', 'pixels=1024 mismatches=0',
                    'depth_pixels=1024 depth_mismatches=%d' % depth_errors}
            clean = ('backend=%s debug_layer=unavailable' % adapter in lines or 'debug_errors=0 debug_warnings=0' in lines)
            ok = (run.returncode == int(message is not None) and want <= lines and clean
                  and (message is None or message in run.stderr))
            print('%s (%s): depth mismatches want %d, exit=%d -> %s' % (label, pair, depth_errors, run.returncode,
                                                                       'as expected' if ok else 'UNEXPECTED'))
            if not ok or label == 'baseline':
                print(run.stdout + run.stderr)
            expect('depth draw: ' + label, ok, True)
    return 0


def draw_kill_exit(dxc, renderer, adapter):
    """The colour pair whose pixel stage leaves through the early exit of a kill, against a CPU reference without
    the killed triangles; the same mesh with the pixel stage that does not kill must fail against it."""
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = geometry()
    kept = [triangle for triangle in indices if colours[triangle[0]][0] / 255 > 0.5]
    image, unkilled = rasterise(points, colours, kept, 12, len(kept)), rasterise(points, colours, indices, 12, 4)
    killed = sum(a != b for a, b in zip(image, unkilled))
    assert 0 < len(kept) < len(indices) and killed > 50, (len(kept), killed)
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH, KILL_EXIT_PIXEL)
        expect('the kill-exit pair translates and compiles', result, 'ok')
        demote, demote_result = request(tmp, MESH, DEMOTE_EXIT_PIXEL)
        expect('the demote-exit pair translates and compiles', demote_result, 'ok')
        plain, plain_result = request(tmp, MESH, PIXEL)
        expect('the pair without the kill translates and compiles', plain_result, 'ok')
        if result != 'ok' or demote_result != 'ok' or plain_result != 'ok':
            return 0
        uds = G.MeshPipeline(code(MESH), code(KILL_EXIT_PIXEL), root()).uds
        constants = bytearray(1024)
        struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, 0)), len(indices))
        struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, 1)), len(points))
        words = [a | b << 10 | c << 20 for a, b, c in indices]
        path = Path(tmp) / 'kill.bin'
        path.write_bytes(bytes(constants) + buffer(points, colours, words) + b'\xcd' * 80
                         + b''.join(bytes(pixel) for pixel in image))
        print('kept %d of %d triangles; %d pixels differ from a draw without the kill' % (len(kept), len(indices), killed))
        for label, stem, mismatches in (('kill-exit pair', key, 0), ('demote-exit pair', demote, 0),
                                        ('the pair without the kill', plain, killed)):
            name = str(Path(tmp) / ('%016x' % stem))
            run = subprocess.run([renderer, adapter, '--mesh-indexed', name + '.ms57.cso', name + '.ps57.cso', str(path)],
                                 capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                return 77
            ok = (run.returncode == int(mismatches > 0)
                  and 'pixels=1024 mismatches=%d' % mismatches in run.stdout)
            print('%s: mismatches want %d, exit=%d -> %s' % (label, mismatches, run.returncode,
                                                          'as expected' if ok else 'UNEXPECTED'))
            if not ok:
                print(run.stdout + run.stderr)
            expect('kill-exit draw: ' + label, ok, True)
    return 0


def draw_formatted(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = geometry()
    words = [a | b << 10 | c << 20 for a, b, c in indices]
    data = buffer(points, colours, words)
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, FORMATTED_MESH)
        if result != 'ok':
            raise AssertionError(result)
        name = str(Path(tmp) / ('%016x' % key))
        manifest = Path(name + '.ms57.txt').read_text().splitlines()
        uds = {tuple(fields[2:]): int(fields[1]) for fields in map(str.split, manifest) if fields and fields[0] == 'udsrc'}
        constants = bytearray(1024)
        struct.pack_into('<I', constants, 4 * uds['rootconst', '0', '0'], len(indices))
        struct.pack_into('<I', constants, 4 * uds['rootconst', '0', '1'], len(points))
        # gfx10 descriptor word 3: FORMAT in bits 12..18, DST_SEL_X in bits 0..2. Decode the same
        # four bytes at each vertex as either its original float or an individual UNORM8 component.
        float_x, unorm_x, unorm_z = (22 << 12) | 4, (56 << 12) | 4, (56 << 12) | 6
        images = {}
        for word3, byte in ((float_x, None), (unorm_x, 0), (unorm_z, 2)):
            shaded = [(r if byte is None else data[192 + 16 * slot + byte], g, b, a)
                      for slot, (r, g, b, a) in enumerate(colours)]
            images[word3] = rasterise(points, shaded, indices, len(points), len(indices))
        path = Path(tmp) / 'formatted.bin'
        args = [renderer, adapter, '--mesh-indexed', name + '.ms57.cso', name + '.ps57.cso', str(path)]
        for word3, reference in ((float_x, float_x), (unorm_x, unorm_x), (unorm_z, unorm_z),
                                 (float_x, unorm_x), (unorm_x, unorm_z)):
            struct.pack_into('<I', constants, 4 * uds['descword3', '0'], word3)
            path.write_bytes(bytes(constants) + data + b'\xcd' * 80 + b''.join(map(bytes, images[reference])))
            mismatches = sum(a != b for a, b in zip(images[word3], images[reference]))
            assert (word3 == reference) == (mismatches == 0)
            run = subprocess.run(args, capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                return 77
            if (run.returncode != int(bool(mismatches))
                    or not re.search(rf'\bpixels={SIDE * SIDE} mismatches={mismatches}\b', run.stdout)):
                raise AssertionError(f'descriptor {word3:#x}, reference {reference:#x}: {run.stdout}{run.stderr}')
            print(f'descriptor {word3:#x}, reference {reference:#x}: {mismatches} differing pixels')
    return 0


def draw_shift(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = geometry()
    # The first provoking vertex (thread 39) shares a guest wave with active thread 5 across the
    # host's possible 32-lane boundary. The last two triangles are in the all-empty guest wave.
    triangles = [tuple(slot + offset for slot in tri) for tri, offset in zip(indices, (32, 0, 64, 64))]
    words = [a | b << 10 | c << 20 for a, b, c in triangles]
    with tempfile.TemporaryDirectory() as tmp:
        key, status = request(tmp, SHIFT_MESH)
        if status != 'ok':
            raise AssertionError(status)
        base = Path(tmp) / ('%016x' % key)
        pipe = G.MeshPipeline(code(SHIFT_MESH), code(PIXEL), root())
        constants = bytearray(1024)
        for slot, count in enumerate((len(triangles), 128)):
            struct.pack_into('<I', constants, 4 * pipe.uds.index(('rootconst', 0, slot)), count)
        head = bytes(constants) + buffer(points, colours, words) + b'\xcd' * 80

        def reference(per_lane):
            red = list(colours)
            for tri in triangles:
                for tid in tri:
                    active = tid < 6 if per_lane else tid // 64 == 0
                    slot = tid % 16
                    red[slot] = (255 if active else 64, *colours[slot][1:])
            return rasterise(points, red, triangles, 128, len(triangles))

        correct, per_lane = reference(False), reference(True)
        different = sum(a != b for a, b in zip(correct, per_lane))
        assert different > 0
        source = base.with_suffix('.ms57.hlsl').read_text()
        predicate = '(tid & ~63u) >= 6u'
        assert source.count(predicate) == 1
        mutated = Path(tmp) / 'per-lane.hlsl'
        mutated.write_text(source.replace(predicate, '(exec & 1) == 0'))
        mutant = Path(tmp) / 'per-lane.cso'
        subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', mutant, mutated], check=True)
        case = Path(tmp) / 'shift.bin'
        for shader, image, mismatch in ((base.with_suffix('.ms57.cso'), correct, 0),
                                        (mutant, correct, different), (mutant, per_lane, 0)):
            case.write_bytes(head + b''.join(map(bytes, image)))
            run = subprocess.run([renderer, adapter, '--mesh-indexed', shader, base.with_suffix('.ps57.cso'), case],
                                 capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                return 77
            if (run.returncode != int(bool(mismatch))
                    or not re.search(rf'\bpixels={SIDE * SIDE} mismatches={mismatch}\b', run.stdout)):
                raise AssertionError(run.stdout + run.stderr)
            print(f'{shader.name}: {mismatch} pixels differ from the reference')
    return 0


def draw_group(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = geometry()
    words = [a | b << 10 | c << 20 for a, b, c in indices]
    correct = rasterise(points, colours, indices, len(points), len(indices))
    zero = rasterise(points, colours, indices[:2], len(points), 2)
    different = sum(a != b for a, b in zip(correct, zero))
    assert different > 0
    with tempfile.TemporaryDirectory() as tmp:
        key, status = request(tmp, GROUP_MESH)
        if status != 'ok':
            raise AssertionError(status)
        base = Path(tmp) / ('%016x' % key)
        pipe = G.MeshPipeline(code(GROUP_MESH), code(PIXEL), root())
        constants = bytearray(1024)
        for slot, count in enumerate((2, len(points))):
            struct.pack_into('<I', constants, 4 * pipe.uds.index(('rootconst', 0, slot)), count)
        head = bytes(constants) + buffer(points, colours, words) + b'\xcd' * 80
        source = base.with_suffix('.ms57.hlsl').read_text()
        assert source.count('gid.x') == 1
        mutated = Path(tmp) / 'group-zero.hlsl'
        mutated.write_text(source.replace('gid.x', '0u'))
        mutant = Path(tmp) / 'group-zero.cso'
        subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', mutant, mutated], check=True)
        case = Path(tmp) / 'group.bin'
        for shader, image, mismatch in ((base.with_suffix('.ms57.cso'), correct, 0),
                                        (mutant, correct, different), (mutant, zero, 0)):
            case.write_bytes(head + b''.join(map(bytes, image)))
            run = subprocess.run([renderer, adapter, '--mesh-indexed-groups', shader,
                                  base.with_suffix('.ps57.cso'), case, '2'], capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                return 77
            if (run.returncode != int(bool(mismatch)) or
                    not re.search(rf'\bpixels={SIDE * SIDE} mismatches={mismatch}\b', run.stdout)):
                raise AssertionError(run.stdout + run.stderr)
            print(f'{shader.name}: {mismatch} pixels differ from the reference')
        clamp_key, status = request(tmp, GROUP_CLAMP)
        if status != 'ok':
            raise AssertionError(status)
        clamp = Path(tmp) / ('%016x.ms57.cso' % clamp_key)
        clamp_pipe = G.MeshPipeline(code(GROUP_CLAMP), code(PIXEL), root())
        for bound, image in ((1, zero), (2, correct)):
            clamp_constants = bytearray(1024)
            for slot, value in enumerate((bound, len(points))):
                struct.pack_into('<I', clamp_constants, 4 * clamp_pipe.uds.index(('rootconst', 0, slot)), value)
            case.write_bytes(bytes(clamp_constants) + buffer(points, colours, words) + b'\xcd' * 80
                             + b''.join(map(bytes, image)))
            run = subprocess.run([renderer, adapter, '--mesh-indexed-groups', clamp,
                                  base.with_suffix('.ps57.cso'), case, '2'], capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                return 77
            if run.returncode or not re.search(rf'\bpixels={SIDE * SIDE} mismatches=0\b', run.stdout):
                raise AssertionError(run.stdout + run.stderr)
            print(f'group clamp bound {bound}: matches reference')
    return 0


def draw_heap(dxc, renderer, adapter):
    os.environ['X2W_DXC'] = dxc
    os.environ.pop('X2W_FXC', None)
    points, colours, indices = geometry()
    words = [a | b << 10 | c << 20 for a, b, c in indices]
    colours = [colours[0]] * len(colours)
    alternate = [(b, g, r, a) for r, g, b, a in colours]
    correct = rasterise(points, colours, indices, len(points), len(indices))
    zero = rasterise(points, alternate, indices, len(points), len(indices))
    different = sum(a != b for a, b in zip(correct, zero))
    assert different > 0
    with tempfile.TemporaryDirectory() as tmp:
        key, status = request(tmp, HEAP_MESH, root_blob=heap_root())
        if status != 'ok':
            raise AssertionError(status)
        base = Path(tmp) / ('%016x' % key)
        pipe = G.MeshPipeline(code(HEAP_MESH), code(PIXEL), heap_root())
        constants = bytearray(1024)
        for source, value in ((('rootconst', 0, 0), len(indices)),
                              (('rootconst', 0, 1), len(points)), (('rootconst', 2, 1), 1)):
            struct.pack_into('<I', constants, 4 * pipe.uds.index(source), value)
        struct.pack_into('<I', constants, 4 * len(pipe.uds), 2)  # heap_count follows user data
        head = bytes(constants) + buffer(points, colours, words) + b'\xcd' * 80
        source = base.with_suffix('.ms57.hlsl').read_text()
        mutant_source, replacements = re.subn(r'(heap_index_[0-9a-f]+ = )\(s24 \+ 0u\) >> 5;', r'\g<1>0u;', source)
        assert replacements == 1
        mutant_hlsl = Path(tmp) / 'heap-zero.hlsl'
        mutant_hlsl.write_text(mutant_source)
        mutant = Path(tmp) / 'heap-zero.cso'
        subprocess.run([dxc, '-T', 'ms_6_6', '-E', 'main', '-Fo', mutant, mutant_hlsl], check=True)
        case = Path(tmp) / 'heap.bin'
        for shader, image, mismatch in ((base.with_suffix('.ms57.cso'), correct, 0),
                                        (mutant, correct, different), (mutant, zero, 0)):
            case.write_bytes(head + b''.join(map(bytes, image)))
            run = subprocess.run([renderer, adapter, '--mesh-indexed-heap', shader,
                                  base.with_suffix('.ps57.cso'), case], capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                return 77
            if (run.returncode != int(bool(mismatch)) or
                    not re.search(rf'\bpixels={SIDE * SIDE} mismatches={mismatch}\b', run.stdout)):
                raise AssertionError(run.stdout + run.stderr)
            print(f'{shader.name}: {mismatch} pixels differ from the reference')
    return 0


def main(argv):
    status = 0
    if argv[1:2] == ['binary'] and len(argv) == 3:
        binary(argv[2])
    elif argv[1:2] == ['draw'] and len(argv) == 5:
        status = draw(*argv[2:])
    elif argv[1:2] == ['lanemask'] and len(argv) == 5:
        status = draw_lane_masks(*argv[2:])
    elif argv[1:2] == ['wave_collectives'] and len(argv) == 5:
        status = draw_wave_collectives(*argv[2:])
    elif argv[1:2] == ['depth'] and len(argv) == 5:
        status = draw_depth(*argv[2:])
    elif argv[1:2] == ['killexit'] and len(argv) == 5:
        status = draw_kill_exit(*argv[2:])
    elif argv[1:2] == ['formatted'] and len(argv) == 5:
        status = draw_formatted(*argv[2:])
    elif argv[1:2] == ['heap'] and len(argv) == 5:
        status = draw_heap(*argv[2:])
    elif argv[1:2] == ['group'] and len(argv) == 5:
        status = draw_group(*argv[2:])
    elif argv[1:2] == ['shift'] and len(argv) == 5:
        status = draw_shift(*argv[2:])
    elif argv[1:2] == ['exchange'] and len(argv) == 6:
        status = draw(*argv[2:])
    elif argv[1:2] == ['runtime'] and len(argv) == 6:
        status = runtime(*argv[2:])
    elif len(argv) == 1:
        controls()
    else:
        print(__doc__)
        return 2
    if FAILED:
        return 1
    print('skip: no mesh shader tier' if status == 77 else 'ok')
    return status


if __name__ == '__main__':
    sys.exit(main(sys.argv))
