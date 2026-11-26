"""Slot 57 mesh requests of one group-memory shape translate in tools/startup/gfx2hlsl.py (#2413).

usage: startup_mesh57_skeleton_check.py
       startup_mesh57_skeleton_check.py binary <llvm-mc>
       startup_mesh57_skeleton_check.py draw <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py depth <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py killexit <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py formatted <dxc> <offscreen_check> <warp|hardware>
       startup_mesh57_skeleton_check.py shift <dxc> <offscreen_check> <warp|hardware>
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
The exchange mode uses the runtime-published request and the same pixel comparison.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

MESH = (
    'bfa00003 befe0481 be84047e 8706ff03 0f000000 bf850007 2c0a0a87 7e0c0280 d8340008 00000506'
    ' bfa3ffe3 bf8a0000 bf820004 bf8a0000 7e0c0280 d8d80008 05000006 befe04c1 9386ff03 00040018'
    ' 8f078606 d7650000 00000ec1 d7660000 000200c1 bf8cc07f 7e180505 f4080405 fa000000 be8d0308'
    ' be8e0309 bf8a0000 bf008006 bf840010 befe0404 bf880007 7e02020d 7e040280 7e06020e d8340000'
    ' 00000102 d8340004 00000302 befe04c1 bfa3ffe3 bf8a0000 8f048c0d 887c0e04 bf900009 bf82000d'
    ' bf8a0000 befe0404 bf880009 7e040280 d8d80000 01000002 d8d80004 03000002 bf8cc17f 7e1a0501'
    ' bf8cc07f 7e1c0503 befe04c1 7d28000e 3608008f 34080884 e0381000 80041004 e03810c0 80041404'
    ' bf8c3f70 f80008cf 13121110 f800020f 17161514 befe04c1 7d28000d 3608008f 34080882 e0301180'
    ' 80041804 bf8c3f70 f8000941 00000018 bf810000').split()
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
DEPTH_PIXEL_MUTANTS = (
    ('a colour write next to the drawn null export', DEPTH_PIXEL[:NULL_EXPORT] + ['f800000f', '00000000'] + DEPTH_PIXEL[NULL_EXPORT:],
     'slot 57 pixel stage mixes the null export with another export'),
    ('a depth-only pixel stage reading an attribute the mesh does not export', ['c8020402'] + DEPTH_PIXEL[1:],
     'pixel stage reads attributes [1] the vertex stage does not export'),
)

# Overwrite red after the raw colour load; green, blue and alpha still come from its other three words.
COLOUR_LOAD = 'buffer_load_dwordx4 v[20:23], v4, s[16:19], 0 offen offset:192'
FORMAT_LOAD = 'buffer_load_format_x v20, v4, s[16:19], 0 offen offset:192'
AFTER_COLOUR_LOAD = MESH.index('e03810c0') + 2
FORMATTED_MESH = MESH[:AFTER_COLOUR_LOAD] + ['e00010c0', '80041404'] + MESH[AFTER_COLOUR_LOAD:]
REWRITTEN_FORMAT = FORMATTED_MESH[:AFTER_COLOUR_LOAD] + ['be930380'] + FORMATTED_MESH[AFTER_COLOUR_LOAD:]  # s_mov_b32 s19, 0


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
    at = mesh.index('7d28000e')
    return mesh[:at] + words.split() + mesh[at:]


MASKED_IF_ELSE = '7e0e02f0 7d880086 be94246a bf880002 7e0e02f2 bf820001 bf800000 8afe7e14 7e0e0280 befe04c1'
# The same if/else with s_add_u32 s50, s13, 1 in the then arm. No path reads s50 after the join.
MASKED_DEAD_SCALAR = '7e0e02f0 7d880086 be94246a bf880003 7e0e02f2 8032810d bf820001 bf800000 8afe7e14 7e0e0280 befe04c1'


def masked_vertex_red(words=MASKED_IF_ELSE):
    mesh = vertex_red(words)
    at = mesh.index('f80008cf')
    return mesh[:at] + ['bf880004'] + mesh[at:]  # s_cbranch_execz skips both vertex exports


# The first six threads select the shift for their entire guest wave. The second wave keeps 64/255.
SHIFT_ASM = """s_mov_b32 s40, 0x3e808081
v_cmp_gt_u32_e32 vcc, 6, v0
s_and_saveexec_b64 s[20:21], vcc
s_cbranch_execz 2
s_lshl_b32 s40, 0x1fc00000, 1
s_mov_b64 exec, -1
v_mov_b32 v7, s40"""
SHIFT_WORDS = 'bea803ff 3e808081 7d880086 be94246a bf880002 8f2881ff 1fc00000 befe04c1 7e0e0228'
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
    # v_cmp_gt_u32 vcc, 6, v0; s_and_saveexec_b64 s[20:21], vcc; s_cbranch_execz .empty
    # v_mov v7, 1.0; s_branch .else; .empty: s_nop; .else: s_andn2_b64 exec, s[20:21], exec
    # v_mov v7, 0; s_mov_b64 exec, -1
    ('lane-masked if/else', masked_vertex_red(), lambda lane: 255 if lane < 6 else 0),
    ('lane-masked if/else with a dead scalar write', masked_vertex_red(MASKED_DEAD_SCALAR),
     lambda lane: 255 if lane < 6 else 0),
)

COUNT = [i for i, word in enumerate(MESH) if word == 'bf8a0000'][2]
POST_COUNT = MESH.index('7d28000e')
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
s_buffer_load_dword s40, s[16:19], s41 offset:0x1e0
v_mov_b32 v7, 1.0
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, -1"""
WAVE_LOAD_PREFIX = 'bea803ff 3e808081 8f298806 beaa0386 7e0e0280 befe04c1 7da8002a beaa0380'
WAVE_LOAD_BODY = 'f4200a08 520001e0 7e0e02f2 bf8cc07f'


def wave_load_mesh(prefix=WAVE_LOAD_PREFIX, body=WAVE_LOAD_BODY, *, exports='7e280228 7e2a0307'):
    """Insert the scalar load before the count exchange and export its value as vertex red."""
    body = body.split()
    before = MESH.index('be8d0308')
    wait = MESH.index('bf8c3f70')
    return (MESH[:before] + prefix.split() + [branch(0xbf88, 0, len(body) + 1)] + body + ['befe04c1']
            + MESH[before:wait + 1] + exports.split() + MESH[wait + 1:])


WAVE_LOAD_MESH = wave_load_mesh()
WAVE_LOAD_REFUSAL = 'slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes'
WAVE_LOAD_REFUSALS = (
    ('different EXEC compare', wave_load_mesh(WAVE_LOAD_PREFIX.replace('7da8002a', '7d28002a'))),
    ('missing full EXEC reset', wave_load_mesh(WAVE_LOAD_PREFIX.replace('befe04c1', 'befe0404'))),
    ('different compare index', wave_load_mesh(WAVE_LOAD_PREFIX.replace('7da8002a', '7da80e2a'))),
    ('EXEC write between compare and branch', wave_load_mesh(WAVE_LOAD_PREFIX.replace('beaa0380', 'befe04c1'))),
    ('branch into load block', wave_load_mesh('bf840009 ' + WAVE_LOAD_PREFIX)),
    ('branch to the EXEC-zero instruction', wave_load_mesh('bf840008 ' + WAVE_LOAD_PREFIX)),
    # The block writes s43, which the join reads, but s43 has no value before the block for an empty wave to keep.
    ('second live scalar write without an old value',
     wave_load_mesh(body=WAVE_LOAD_BODY + ' beab0386', exports='7e280228 7e2a022b')),
    ('buffer store', wave_load_mesh(body=WAVE_LOAD_BODY + ' f4600a08 520001e0')),
    ('buffer atomic', wave_load_mesh(body=WAVE_LOAD_BODY + ' e0c81000 80040700')),
)
# Blocks that #2660 refused and that the wave block accepts: every scalar instruction of the block runs once for a
# wave with an active lane, so further scalar loads and writes, live or dead after the join, follow the guest.
WAVE_LOAD_ACCEPTED = (
    ('wider SMEM load', wave_load_mesh(body=WAVE_LOAD_BODY.replace('f4200a08', 'f4240a08'))),
    ('dead scalar result', wave_load_mesh(exports='7e2a0307')),
    # With the adjacent reset replaced by s_nop, the reset that closes the group-index exchange still reaches the
    # compare: nothing between them reads or writes EXEC or VCC, and no branch enters that range.
    ('full EXEC reset earlier in the region', wave_load_mesh(WAVE_LOAD_PREFIX.replace('befe04c1', 'bf800000'))),
)
# A table load is accepted by the wave block; this one then stops at the register-offset rule of s_load.
WAVE_LOAD_OTHER_REFUSALS = (
    ('ordinary SMEM load with a register offset', wave_load_mesh(body=WAVE_LOAD_BODY.replace('f4200a08', 'f4000a05')),
     "s_load with a register offset through ('table', 1)"),
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
s_buffer_load_dword s44, s[16:19], s41 offset:0x1e4
s_load_dwordx4 s[48:51], s[10:11], 0x0
s_add_u32 s45, s41, 4
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s40, s[16:19], s45 offset:0x1dc
v_cmp_class_f32_e32 vcc, -1, v7
v_mov_b32 v7, 1.0
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, -1"""
WAVE_BLOCK_PREFIX = WAVE_LOAD_PREFIX.replace('befe04c1 7da8002a', 'befe04c1 beab0380 7da8002a')
WAVE_BLOCK_BODY = 'f4200b08 520001e4 f4080c05 fa000000 802d8429 bf8cc07f f4200a08 5a0001dc 7d100ec1 7e0e02f2 bf8cc07f'
WAVE_BLOCK_MESH = wave_load_mesh(WAVE_BLOCK_PREFIX, WAVE_BLOCK_BODY)
WAVE_BLOCK_REFUSALS = (
    ('VCC live at the join', wave_load_mesh(WAVE_BLOCK_PREFIX, WAVE_BLOCK_BODY, exports='7e280228 022a0e80')),
    ('src_scc in the block', wave_load_mesh(WAVE_BLOCK_PREFIX, WAVE_BLOCK_BODY + ' beaf03fd')),
    ('EXEC copied in the block', wave_load_mesh(WAVE_BLOCK_PREFIX, WAVE_BLOCK_BODY + ' beb0047e')),
    ('branch inside the block', wave_load_mesh(WAVE_BLOCK_PREFIX, WAVE_BLOCK_BODY.replace('bf8cc07f', 'bf840000', 1))),
    ('branch from outside between the reset and the compare', wave_load_mesh('bf840006 ' + WAVE_BLOCK_PREFIX,
                                                                              WAVE_BLOCK_BODY)),
    # s_cmp_eq_u32 s43, 0; s_cbranch_scc1 to the execz branch: that path never runs the compare.
    ('branch from between the reset and the compare past the compare',
     wave_load_mesh(WAVE_BLOCK_PREFIX.replace('beab0380', 'beab0380 bf06802b bf850002'), WAVE_BLOCK_BODY)),
    ('EXEC write between the reset and the compare',
     wave_load_mesh(WAVE_BLOCK_PREFIX.replace('beab0380', 'befe0404'), WAVE_BLOCK_BODY)),
)


def count_entry_red(*extra_words):
    wait = MESH.index('bf8c3f70')
    return MESH[:COUNT] + [
        '7e0e0280',  # v_mov_b32_e32 v7, 0
        'befe04c1',  # s_mov_b64 exec, -1
        '7da8000d',  # v_cmpx_gt_u32_e32 s13, v0 (the proven thread index)
        branch(0xbf88, COUNT + 3, COUNT + 5 + len(extra_words)),  # s_cbranch_execz to the first count barrier
        '7e0e02f2',  # v_mov_b32_e32 v7, 1.0
    ] + list(extra_words) + MESH[COUNT:wait + 1] + [
        '7e280307',  # v_mov_b32_e32 v20, v7
    ] + MESH[wait + 1:]


COUNT_ENTRY = count_entry_red()
COUNT_ENTRY_LT = mutate('7da8000d', '7da2000d', mesh=COUNT_ENTRY)
# v_mov_b32_e32 v8, 4 before the reset, then v_cmpx_lt_u32_e32 v8, v0.
COUNT_ENTRY_VGPRS = mutate('7da8000d', '7da20108', mesh=COUNT_ENTRY)
COUNT_ENTRY_VGPRS = COUNT_ENTRY_VGPRS[:COUNT + 1] + ['7e100284'] + COUNT_ENTRY_VGPRS[COUNT + 1:]
DEAD_SCALAR = '87288080'  # s_and_b32 s40, 0, 0
COUNT_ENTRY_SCALAR = count_entry_red(DEAD_SCALAR)
COUNT_ENTRY_CASES = (
    ('EXEC-zero branch to the count exchange after GT', COUNT_ENTRY, lambda lane: 255 if lane < 4 else 0),
    ('EXEC-zero branch to the count exchange after LT', COUNT_ENTRY_LT, lambda lane: 255 if lane > 4 else 0),
    ('EXEC-zero branch to the count exchange after LT with two VGPRs', COUNT_ENTRY_VGPRS,
     lambda lane: 255 if lane > 4 else 0),
    ('EXEC compare with a defined VGPR other than the thread index',
     mutate('7da8000d', '7da80e0d', mesh=COUNT_ENTRY), lambda lane: 255),  # s13 > v7 (4 > 0)
    ('EXEC-zero branch to the count exchange over dead scalar ALU', COUNT_ENTRY_SCALAR,
     lambda lane: 255 if lane < 4 else 0),
)
EXCHANGE_BRANCH_REFUSAL = 'slot 57 mesh: a branch enters or crosses a group-memory exchange'


def joined_vertex_masks():
    pos = MESH.index('f80008cf')
    # Move the vertex narrowing after the loads so the join cannot make any export operands stale.
    return MESH[:POST_COUNT] + MESH[POST_COUNT + 1:pos] + [
        '7d28000e',  # v_cmpx_gt_i32 s14, v0
        'bf008080',  # s_cmp_eq_i32 0, 0
        'bf850001',  # s_cbranch_scc1 .pos (first incoming mask is the vertex count)
        'befe04c1',  # s_mov_b64 exec, -1 (fallthrough reaches .pos with all lanes)
    ] + MESH[pos:]


def request(directory, mesh, pixel=PIXEL):
    ms, ps, blob = code(mesh), code(pixel), root()
    key = G.mesh57_key(ms, ps, blob)
    for ext, data in (('.root', blob), ('.ps57', ps), ('.m57', ms)):
        (Path(directory) / ('req_%016x%s' % (key, ext))).write_bytes(data)
    return key, G.translate_request(str(directory), key)


FAILED = []


def expect(what, got, want):
    if got != want:
        FAILED.append(what)
        print('FAIL %s: got %r, want %r' % (what, got, want))


REFUSALS = (
    ('formatted load with descriptor word 3 overwritten', REWRITTEN_FORMAT,
     'formatted buffer load through a descriptor whose words were rewritten'),
    # The request must carry the two stored counts, in this order.
    ('a request that publishes another count', mutate('887c0e04', '887c0d04'),
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
     mutate('7da8000d', '7da8020d', mesh=COUNT_ENTRY), EXCHANGE_BRANCH_REFUSAL),
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
     count_entry_red('f4000a05', 'fa000000'), EXCHANGE_BRANCH_REFUSAL),  # s_load_dword s40, s[10:11], 0
    ('a v_cmpx in the skipped block',
     mutate('7e0e02f2', '7da8000d', mesh=COUNT_ENTRY), EXCHANGE_BRANCH_REFUSAL),
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
    ('an indirect branch', mutate('bf8c3f70', 'be802014'),
     'slot 57 mesh: indirect control flow (s_setpc_b64) is not covered'),
    # s_call_b64 s[20:21], 2 replaces s_waitcnt vmcnt(0) and targets the parameter export.
    ('a call after the count exchange', mutate('bf8c3f70', 'bb140002'),
     'slot 57 mesh: indirect control flow (s_call_b64) is not covered'),
    # s_swappc_b64 s[20:21], s[22:23] replaces s_waitcnt vmcnt(0).
    ('a PC swap after the count exchange', mutate('bf8c3f70', 'be942116'),
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
    # v_cmp_gt_u32 vcc, 6, v0; s_and_saveexec_b64 s[20:21], vcc narrows EXEC; the uniform compare then writes VCC
    # only for the narrowed lanes; s_mov_b64 exec, -1 restores the full mask before s_cbranch_vccz skips s_nop.
    ('a VCC compare under a narrowed EXEC before the branch',
     MESH[:POST_COUNT] + ['7d880086', 'be94246a'] + VCC_COMPARE.split() + ['befe04c1', 'bf860001', 'bf800000']
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
    # s_or_b64 exec, exec, s[20:21] replaces the full EXEC reset after the else arm.
    ('a saved EXEC read after the masked if/else', mutate('befe04c1', '89fe7e14', 3, mesh=masked_vertex_red()),
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
    ('a position export under the primitive count', mutate('7d28000e', '7d28000d'),
     'slot 57 mesh: export pos0 is not under exec = -1 narrowed by its output count > thread index'),
    # v_mov_b32 v7, v1: v1 holds the stored count in wave 0 and the read one elsewhere.
    ('a read of a register the exchange leaves different per wave', mutate('bf8c3f70', '7e0e0301'),
     'slot 57 mesh: v1 is read where it is not defined for every lane (an entry value, or left different per wave '
     'by an exchange)'),
    # v_mov_b32 v7, v10; s_mov_b32 s20, s25; s_mov_b32 s20, s3: entry values the model does not define.
    ('a read of an entry VGPR', mutate('bf8c3f70', '7e0e030a'),
     'slot 57 mesh: v10 is read where it is not defined for every lane (an entry value, or left different per wave '
     'by an exchange)'),
    ('a read of an entry SGPR', mutate('bf8c3f70', 'be940319'),
     'numeric use of s25 holding an entry value the mesh model does not define'),
    ('a read of s3 other than the wave index', mutate('bf8c3f70', 'be940303'),
     'slot 57 mesh: s3 is read other than as the wave index'),
    # s_and_b32 s4, s3, 0xf000000: wave 0 would restore a zero mask and store no count.
    ('a wave test that overwrites the saved lane-0 mask', mutate('8706ff03', '8704ff03'),
     'slot 57 mesh count exchange: registers of the accepted shape overlap'),
    # s_mov_b32 s6, s8 after the thread index is derived: the count compare no longer tells the waves apart.
    ('a count compare of a register rewritten after the wave index', mutate('bf8cc07f', 'be860308'),
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


def controls():
    os.environ['X2W_FXC'] = ''   # sources and manifest only; the draw mode compiles
    with tempfile.TemporaryDirectory() as tmp:
        key, result = request(tmp, MESH)
        expect('the fixture translates', result, 'ok')
        names = sorted(p.name[16:] for p in Path(tmp).iterdir() if not p.name.startswith('req_'))
        expect('published files', names, ['.ms57.hlsl', '.ms57.txt', '.ps57.hlsl'])
        expect('a translated request is not pending', G.pending_keys(tmp, False), [])
        manifest = (Path(tmp) / ('%016x.ms57.txt' % key)).read_text().splitlines()
        expect('paired manifest stage information',
               [line for line in manifest if line.startswith(('params ', 'targets ', 'ps '))],
               ['params 1', 'targets 0'])
        text = (Path(tmp) / ('%016x.ms57.hlsl' % key)).read_text()
        body = [line.strip() for line in text[text.index('void main('):].splitlines()]
        want = ['if (tid == 0u) { lds[0] = s13; lds[1] = s14; }', 'GroupMemoryBarrierWithGroupSync();',
                's13 = min(lds[0], 128u);', 's14 = min(lds[1], 128u);', 'if (s14 == 0u) s13 = 0u;',
                'SetMeshOutputCounts(s14, s13);']
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
        expect('scalar write in a skipped block read after the join', request(tmp, live)[1],
               'unsupported: ' + WAVE_LOAD_REFUSAL)
        for what, mesh in WAVE_LOAD_REFUSALS + WAVE_BLOCK_REFUSALS:
            expect(what, request(tmp, mesh)[1], 'unsupported: ' + WAVE_LOAD_REFUSAL)
        for what, mesh in WAVE_LOAD_ACCEPTED:
            expect(what + ' translates', request(tmp, mesh)[1], 'ok')
        for what, mesh, reason in WAVE_LOAD_OTHER_REFUSALS:
            expect(what, request(tmp, mesh)[1], 'unsupported: ' + reason)
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
        from unittest.mock import patch

        uniform_source = G.Stage.uniform_source
        with patch.object(G.Stage, 'uniform_source', lambda stage, token: token != 's42' and uniform_source(stage, token)):
            expect('non-uniform wave load bound', request(tmp, WAVE_LOAD_MESH)[1], 'unsupported: ' + WAVE_LOAD_REFUSAL)
        with patch.object(G.Stage, 'uniform_source', lambda stage, token: token != 's41' and uniform_source(stage, token)):
            expect('unproven wave load address', request(tmp, WAVE_LOAD_MESH)[1], 'unsupported: ' + WAVE_LOAD_REFUSAL)
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


def binary(mc):
    mesh = (HERE / 'startup_mesh57_skeleton.s').read_text()
    expect('mesh words match llvm-mc', assemble(mc, mesh), MESH)
    expect('formatted mesh words match llvm-mc', assemble(mc, mesh.replace(COLOUR_LOAD, COLOUR_LOAD + '\n' + FORMAT_LOAD)),
           FORMATTED_MESH)
    expect('rewritten descriptor words match llvm-mc',
           assemble(mc, mesh.replace(COLOUR_LOAD, COLOUR_LOAD + '\ns_mov_b32 s19, 0\n' + FORMAT_LOAD)), REWRITTEN_FORMAT)
    expect('pixel words match llvm-mc', assemble(mc, PIXEL_ASM), PIXEL)
    shift = mesh.replace('v_cmpx_gt_i32 s14, v0', SHIFT_ASM + '\nv_cmpx_gt_i32 s14, v0')
    shift = shift.replace('exp pos0', 'v_mov_b32 v20, v7\nexp pos0')
    expect('scalar shift mesh words match llvm-mc', assemble(mc, shift), SHIFT_MESH)
    wave_load = mesh.replace('s_mov_b32 s13, s8', WAVE_LOAD_ASM + '\ns_mov_b32 s13, s8')
    wave_load = wave_load.replace('exp pos0', 'v_mov_b32 v20, s40\nv_mov_b32 v21, v7\nexp pos0')
    expect('pre-exchange wave load words match llvm-mc', assemble(mc, wave_load), WAVE_LOAD_MESH)
    wave_block = mesh.replace('s_mov_b32 s13, s8', WAVE_BLOCK_ASM + '\ns_mov_b32 s13, s8')
    wave_block = wave_block.replace('exp pos0', 'v_mov_b32 v20, s40\nv_mov_b32 v21, v7\nexp pos0')
    expect('pre-exchange wave block words match llvm-mc', assemble(mc, wave_block), WAVE_BLOCK_MESH)


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


def draw_wave_load(directory, renderer, adapter, mesh=WAVE_LOAD_MESH, label='wave load', dxc=None):
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
    struct.pack_into('<f', data, 480, 1.0)
    constants = bytearray(1024)
    uds = G.MeshPipeline(code(mesh), code(PIXEL), root()).uds
    for slot, count in enumerate((len(triangles), 128)):
        struct.pack_into('<I', constants, 4 * uds.index(('rootconst', 0, slot)), count)

    bound, old = 6, 64
    wave_red = {}
    reads = []
    for first in range(0, 128, 64):
        wave_red[first] = old
        if first < bound:
            address = 480 + 4 * first
            reads.append(address)
            wave_red[first] = round(struct.unpack_from('<f', data, address)[0] * 255)
    expect('only the mixed wave reads; the empty wave address is past the SRV',
           (reads, len(data) + 80 < 480 + 4 * 64, wave_red), ([480], True, {0: 255, 64: 64}))

    def reference(swap):
        rendered = list(colours)
        for tri in triangles:
            for tid in tri:
                first, slot = tid & ~63, tid & 15
                red = wave_red[first ^ (64 if swap else 0)]
                rendered[slot] = (red, 255 if tid < bound else 0, *colours[slot][2:])
        return rasterise(points, rendered, triangles, 128, len(triangles))

    correct, swapped = reference(False), reference(True)
    different = sum(a != b for a, b in zip(correct, swapped))
    expect('swapping old and loaded red changes every covered pixel', different,
           sum(pixel != (0, 0, 0, 255) for pixel in correct))
    expect('the wave load covers pixels', different > 0, True)
    path = Path(directory) / 'wave-load.bin'
    shader = base.with_suffix('.ms57.cso')
    cases = [(label, shader, correct, 0, 0), ('swapped %s selection' % label, shader, swapped, 1, different)]
    if dxc:
        # Lanes outside EXEC in the mixed wave keep the old red when the block is skipped per lane.
        per_lane = reference_per_lane(points, colours, triangles, bound, old, wave_red)
        lanes = sum(a != b for a, b in zip(correct, per_lane))
        expect('the per-lane %s mutant changes pixels' % label, lanes > 0, True)
        source = base.with_suffix('.ms57.hlsl').read_text()
        expect('one wave predicate on the block branch', source.count('pc = (!wave_any) ?'), 1)
        mutated = Path(directory) / 'per-lane.hlsl'
        mutated.write_text(source.replace('pc = (!wave_any) ?', 'pc = ((exec & 1) == 0) ?'))
        mutant = Path(directory) / 'per-lane.cso'
        subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'main', '-Fo', mutant, mutated], check=True, capture_output=True)
        cases += [('per-lane %s against the wave image' % label, mutant, correct, 1, lanes),
                  ('per-lane %s against its own image' % label, mutant, per_lane, 0, 0)]
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


def reference_per_lane(points, colours, triangles, bound, old, wave_red):
    """The image when each lane skips the block by its own EXEC bit: only threads below the bound load."""
    rendered = list(colours)
    for tri in triangles:
        for tid in tri:
            slot = tid & 15
            red = wave_red[tid & ~63] if tid < bound else old
            rendered[slot] = (red, 255 if tid < bound else 0, *colours[slot][2:])
    return rasterise(points, rendered, triangles, 128, len(triangles))


def publish_request(directory, publisher, adapter):
    ms, ps = code(MESH), code(PIXEL)
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
            return draw_wave_load(tmp, renderer, adapter, WAVE_BLOCK_MESH, 'wave block', dxc)
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
        names = sorted(p.name[16:] for p in Path(tmp).iterdir() if p.name.startswith('%016x' % key))
        expect('depth-only compiled files', names, ['.ms57.cso', '.ms57.hlsl', '.ms57.txt', '.ps57.cso', '.ps57.hlsl'])
        manifest = (Path(tmp) / ('%016x.ms57.txt' % key)).read_text().splitlines()
        uds = {tuple(fields[2:]): int(fields[1]) for fields in map(str.split, manifest) if fields and fields[0] == 'udsrc'}
        constants = bytearray(1024)
        struct.pack_into('<I', constants, 4 * uds['rootconst', '0', '0'], len(indices))
        struct.pack_into('<I', constants, 4 * uds['rootconst', '0', '1'], len(points))
        head = b''.join(bytes(constants) + upload for upload in uploads) + b'\xcd' * 160
        path = Path(tmp) / 'depth.bin'
        stems = {'depth-only pair': key, 'pixel stage without discard': bare}
        # (label, pair, reference depth, depth mismatches). The colour target must stay cleared in every case.
        cases = (('baseline', 'depth-only pair', depth, 0),
                 ('a reference without the depth writes', 'depth-only pair', cleared, written),
                 ('the discard dropped', 'pixel stage without discard', depth, killed))
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


def main(argv):
    status = 0
    if argv[1:2] == ['binary'] and len(argv) == 3:
        binary(argv[2])
    elif argv[1:2] == ['draw'] and len(argv) == 5:
        status = draw(*argv[2:])
    elif argv[1:2] == ['depth'] and len(argv) == 5:
        status = draw_depth(*argv[2:])
    elif argv[1:2] == ['killexit'] and len(argv) == 5:
        status = draw_kill_exit(*argv[2:])
    elif argv[1:2] == ['formatted'] and len(argv) == 5:
        status = draw_formatted(*argv[2:])
    elif argv[1:2] == ['shift'] and len(argv) == 5:
        status = draw_shift(*argv[2:])
    elif argv[1:2] == ['exchange'] and len(argv) == 6:
        status = draw(*argv[2:])
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
