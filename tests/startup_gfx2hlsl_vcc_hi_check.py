"""Wave32 VCC words in the graphics translator, with optional D3D12 readback.

  (no arguments)                       translation controls and refusals, pure Python
  binary <llvm-mc>                     assemble the fixture sources with LLVM (gfx1013) and compare with the words here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pixel stage (DXC, and FXC when given), draw 16 x 16
                                       pixels and compare every word with the CPU reference below

In wave32 VCC is vcc_lo alone, and compilers allocate vcc_hi (SGPR 107) like any other scalar register: a compare or
carry-out destination, a v_cndmask mask, a scalar operand. The decoder prints wave32 syntax, so the text does not
give the wave size; the translator renames vcc_hi to an SGPR the program never names only when that is exact in both
wave sizes: no instruction names the 64-bit pair `vcc`, and either a VALU compare/carry writes vcc_hi (or an odd
SGPR) or a VOP3 mask source reads one (a misaligned pair in wave64, so the program is wave32), or no VALU instruction
and no vcc branch uses vcc_lo as a lane mask (in wave64 those also write or read vcc_hi for lanes 32..63). Otherwise vcc_hi is refused by name, as before, and a
scalar write into vcc_hi or exec_hi is now a refusal instead of a TypeError.

The draw fixture PS (packed screen position v0 = x | y << 16):
  v_cmp_gt_u32_e64 vcc_hi, v0, 0x50005         ; v_cndmask_b32_e64 v2, 1.0, 2.0, vcc_hi -> 2.0 where v0 > 0x50005
  v_add_co_u32 v3, vcc_hi, v0, 0xfff7fff8      ; v_add_co_ci_u32_e64 v4, s3, v0, 2.0, vcc_hi -> + the carry
  s_mov_b32 vcc_hi, 0x1234 ; s_add_u32 s0, vcc_hi, 16 ; v5 = s0 + v0
The reference below computes the same words with Python integers. The selected words are 1.0 and 2.0, not the
integers 1 and 2, and the carry rides on v0 + 0x40000000: with FXC a word that can only be 0 or 1 came back as
0 on WARP where it was 1 (DXC kept it), a denormal export artifact, not a translation error. Mutations (the compare bit not read, the carry
lost, vcc_hi read as zero) must fail the draw.

PS_DATA also exports a hardwired low-word comparison and two scalar arithmetic results. A first-lane read at the
join of a uniform branch exercises both retries. The exports use normal float bit patterns for both compilers.
PS_DATA2 keeps a pair write in place and reads uniform scalar selects as VOP3 masks.
"""
import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

MASK = 0xffffffff
WIDTH = HEIGHT = 16
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
failures = []

# name -> (source, words assembled by llvm-mc 19 for gfx1013, expected refusal or None)
FIXTURES = {
    'PS': ('''v_cmp_gt_u32_e64 vcc_hi, v0, 0x50005
v_cndmask_b32_e64 v2, 1.0, 2.0, vcc_hi
v_add_co_u32 v3, vcc_hi, v0, 0xfff7fff8
v_add_co_ci_u32_e64 v4, s3, v0, 0x40000000, vcc_hi
s_mov_b32 vcc_hi, 0x1234
s_add_u32 s0, vcc_hi, 16
v_add_nc_u32_e32 v5, s0, v0
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
''', '6b00c4d400ff010005000500020001d5f2e8ad01036b0fd700ff0100f8fff7ff040328d500e9ad01ff03ebbe341200006b900080'
         '00000a4a0f1800f802030405000081bf', None),
    # Only scalar writes of vcc_hi and no VALU VCC write: exact in either wave size.
    'SCALAR_ONLY': ('''s_mov_b32 vcc_hi, 0x1234
s_lshl_b32 vcc_hi, vcc_hi, 4
s_add_u32 s0, vcc_hi, 16
v_add_nc_u32_e32 v5, s0, v0
exp mrt0 v5, v0, v0, v0 done vm
s_endpgm
''', 'ff03ebbe341200006b846b8f6b90008000000a4a0f1800f805000000000081bf', None),
    # A VOPC writes vcc_lo and nothing proves wave32: in wave64 the compare would also write vcc_hi.
    'SCALAR_AND_VOPC': ('''s_mov_b32 vcc_hi, 0x1234
s_lshl_b32 vcc_hi, vcc_hi, 4
s_add_u32 s0, vcc_hi, 16
v_cmp_gt_u32_e32 vcc_lo, 0x50005, v0
v_cndmask_b32_e64 v2, 1, 2, vcc_lo
v_add_nc_u32_e32 v5, s0, v0
exp mrt0 v2, v5, v0, v0 done vm
s_endpgm
''', 'ff03ebbe341200006b846b8f6b900080ff00887d05000500020001d58104a90100000a4a0f1800f802050000000081bf',
        's_mov_b32 destination vcc_hi'),
    # A VALU mask read of vcc_lo and nothing proves wave32: in wave64 lanes 32..63 would read vcc_hi as their mask
    # bit, which the renamed scalar register does not model (found by an independent review).
    'MASK_READ': ('''s_mov_b32 vcc_lo, 0
s_mov_b32 vcc_hi, -1
v_mov_b32_e32 v2, 2.0
v_cndmask_b32_e32 v1, 1.0, v2, vcc_lo
exp mrt0 v1, v1, v1, v1 done vm
s_endpgm
''', '8003eabec103ebbef402047ef20402020f1800f801010101000081bf', 's_mov_b32 destination vcc_hi'),
    # A whole scalar word is not a lane mask: the lowering would test bit 0 of 0x55555555 for every lane.
    'SCALAR_MASK': ('''s_mov_b32 vcc_hi, 0x55555555
v_cndmask_b32_e64 v2, 1.0, 2.0, vcc_hi
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''', 'ff03ebbe55555555020001d5f2e8ad010f1800f802020202000081bf', 's_mov_b32 destination vcc_hi'),
    # The 64-bit pair `vcc` names vcc_hi as its high word.
    'PAIR': ('''v_cmp_gt_u32_e64 vcc_hi, v0, 0x50005
s_and_b64 s[2:3], vcc, exec
v_cndmask_b32_e64 v2, 1, 2, vcc_hi
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''', '6b00c4d400ff0100050005006a7e8287020001d58104ad010f1800f802020202000081bf', 'v_cmp_gt_u32 into vcc_hi'),
    # exec_hi is a mask half in every wave size: a scalar write is refused, not a crash.
    'EXEC_HI': ('''s_mov_b32 exec_hi, 0
v_mov_b32_e32 v2, 1
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''', '8003ffbe8102047e0f1800f802020202000081bf', 's_mov_b32 destination exec_hi'),
    # A v_readfirstlane_b32 into vcc_hi writes a whole word too: a straight-line VOP3 mask source read of it would
    # test bit 0 of that word for every lane, so the rename is refused (the old rule caught only s_ and v_readlane).
    'READFIRSTLANE_MASK': ('''v_mov_b32_e32 v2, s0
v_readfirstlane_b32 vcc_hi, v2
v_cndmask_b32_e64 v1, 1.0, 2.0, vcc_hi
exp mrt0 v1, v1, v1, v1 done vm
s_endpgm
''', '0002047e0205d67e010001d5f2e8ad010f1800f801010101000081bf', 'v_readfirstlane_b32 requires a proven uniform source'),
    # The same writer with the mask read across a branch boundary: the order is unknown, so it is refused as well.
    'READFIRSTLANE_BRANCH': ('''v_mov_b32_e32 v2, s0
v_readfirstlane_b32 vcc_hi, v2
s_cbranch_execz skip
v_cndmask_b32_e64 v1, 1.0, 2.0, vcc_hi
exp mrt0 v1, v1, v1, v1 done vm
s_endpgm
skip:
s_endpgm
''', '0002047e0205d67e0a0a88bf010001d5f2e8ad010f1800f801010101000081bf000081bf',
        'v_readfirstlane_b32 requires a proven uniform source'),
    # Branch-only: the VOP3 mask read comes before the whole-word writer in source order and a back edge reaches it,
    # so no sequential read_after applies: only the branchy writer check refuses. The old rule (s_ and v_readlane
    # writers only) renamed this one, so the old branchy check never saw a lane-read writer.
    'READFIRSTLANE_BACK_EDGE': ('''loop:
v_cndmask_b32_e64 v1, 1.0, 2.0, vcc_hi
v_add_nc_u32_e32 v3, v1, v0
exp mrt0 v3, v0, v0, v0 done vm
v_mov_b32_e32 v2, s0
v_readfirstlane_b32 vcc_hi, v2
s_branch loop
''', '010001d5f2e8ad010101064a0f1800f8030000000002047e0205d67e0a0a82bf',
        'operand \'vcc_hi\''),
    # The same shape with a v_readlane_b32 writer: refused by the pre-existing branchy writer check.
    'READLANE_BACK_EDGE': ('''loop:
v_cndmask_b32_e64 v1, 1.0, 2.0, vcc_hi
v_add_nc_u32_e32 v3, v1, v0
exp mrt0 v3, v0, v0, v0 done vm
v_readlane_b32 vcc_hi, v2, s0
s_branch loop
''', '010001d5f2e8ad010101064a0f1800f8030000006b0060d7020100000a0a82bf',
        'operand \'vcc_hi\''),
    # The same lane-read writer with only scalar reads of vcc_hi: exact in either wave size, so it translates.
    'READFIRSTLANE_SCALAR': ('''v_mov_b32_e32 v2, s0
v_readfirstlane_b32 vcc_hi, v2
s_add_u32 s1, vcc_hi, 16
v_add_nc_u32_e32 v5, s1, v0
exp mrt0 v5, v0, v0, v0 done vm
s_endpgm
''', '0002047e0205d67e6b90018001000a4a0f1800f805000000000081bf', None),
    'PS_DATA': ('''v_cmp_gt_u32_e64 vcc_hi, v0, 0x50005
v_cndmask_b32_e64 v2, 1.0, 2.0, vcc_hi
v_add_co_u32 v6, vcc_hi, v0, 0xfff7fff8
v_add_co_ci_u32_e64 v3, s3, v0, 0, vcc_hi
s_movk_i32 vcc_lo, 0x80
s_mul_i32 vcc_lo, vcc_lo, 2
s_add_u32 s20, vcc_lo, 0x3f800000
s_sub_i32 vcc_hi, 0, vcc_lo
s_mul_hi_u32 vcc_hi, 0x40000000, vcc_hi
s_mov_b32 s21, 0x10000000
s_add_u32 s21, s21, vcc_hi
v_add_nc_u32_e32 v3, s20, v3
v_add_nc_u32_e32 v4, s21, v0
v_mov_b32_e32 v7, 1.0
s_cmp_eq_u32 s20, 0
s_cbranch_scc1 1
v_mov_b32_e32 v7, 2.0
v_readfirstlane_b32 s22, v7
v_cmp_gt_u32_e32 vcc_lo, 0x90009, v0
v_mov_b32_e32 v6, s22
v_cndmask_b32_e32 v5, 1.0, v6, vcc_lo
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
''',
        '6b00c4d400ff010005000500020001d5f2e8ad01066b0fd700ff0100f8fff7ff030328d50001ad0180006ab06a826a93'
        '6af21480806aeb81f46beb9aff0395be00000010156b15801406064a1500084af2020e7e148006bf010085bff4020e7e'
        '07052c7eff00887d0900090016020c7ef20c0a020f1800f802030405000081bf', None),
    # The extra scalar ranges reserve s8 through s105 and force the retry to use the low SGPRs.
    'PS_DATA2': ('''s_mov_b64 vcc, 0x14500000
s_mov_b32 vcc_hi, 0x3f800000
s_add_u32 s20, vcc_hi, 16
v_add_nc_u32_e32 v5, s20, v0
s_cmp_eq_u32 s20, 0x3f800010
s_cselect_b32 vcc_hi, -1, 0
v_cndmask_b32_e64 v2, 1.0, 2.0, vcc_hi
v_add_co_ci_u32_e64 v3, s3, v0, 0, vcc_hi
v_add_nc_u32_e32 v3, 1.0, v3
s_cmp_eq_u32 s20, 0
s_cselect_b32 vcc_lo, -1, 0
v_cndmask_b32_e64 v4, 1.0, 2.0, vcc_lo
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
''',
        'ff04eabe00005014f203ebbe6b90148014000a4a14ff06bf1000803fc1806b85020001d5f2e8ad01030328d50001ad01'
        'f206064a148006bfc1806a85040001d5f2e8a9010f1800f802030405000081bf', None),
    'PAIR_REACH': ('''s_mov_b32 exec_lo, -1
s_mov_b32 vcc_hi, 0x40000000
s_mov_b64 vcc, 0x14500000
s_add_u32 s20, vcc_hi, 16
s_mov_b32 vcc_hi, 0x3f800000
s_add_u32 s21, vcc_hi, s20
v_mov_b32_e32 v2, s21
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''', 'c103febef403ebbeff04eabe000050146b901480f203ebbe6b1415801502047e0f1800f802020202000081bf', "operand 'vcc_hi'"),
    'UNIFORM_MOVES': ('''s_movk_i32 vcc_hi, -1
v_cndmask_b32_e64 v2, 1.0, 2.0, vcc_hi
s_mov_b32 vcc_lo, 0
v_cndmask_b32_e64 v3, 1.0, 2.0, vcc_lo
s_movk_i32 vcc_lo, 0
v_cndmask_b32_e32 v4, 1.0, v2, vcc_lo
s_mov_b32 vcc_hi, -1
v_cndmask_b32_e64 v5, 1.0, 2.0, vcc_hi
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
''',
        'ffff6bb0020001d5f2e8ad018003eabe030001d5f2e8a90100006ab0f2040802c103ebbe050001d5f2e8ad010f1800f8'
        '02030405000081bf', None),
    'PS_DESCRIPTOR': ('''s_load_dwordx8 s[100:107], s[8:9], 0x0
s_load_dwordx4 s[12:15], s[8:9], 0x20
s_mov_b32 s24, 0
s_mov_b32 s25, 0
s_mov_b32 s26, 0
s_mov_b32 s27, 0
v_mov_b32_e32 v1, 0.5
v_mov_b32_e32 v2, 0.5
s_waitcnt lgkmcnt(0)
image_sample_lz v[8:11], v[1:2], s[100:107], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D
s_load_dwordx16 s[16:31], s[8:9], 0x30
s_load_dwordx16 s[32:47], s[8:9], 0x70
s_load_dwordx16 s[48:63], s[8:9], 0xb0
s_load_dwordx16 s[64:79], s[8:9], 0xf0
s_load_dwordx16 s[80:95], s[8:9], 0x130
s_load_dwordx4 s[96:99], s[8:9], 0x170
s_mov_b32 s10, 0
s_mov_b32 s11, 0
s_mov_b32 exec_lo, -1
s_buffer_load_dword vcc_lo, s[12:15], 0x10
s_waitcnt lgkmcnt(0)
s_mul_i32 vcc_lo, vcc_lo, 3
s_sub_i32 vcc_hi, 0, vcc_lo
s_mul_hi_u32 vcc_hi, 0x40000000, vcc_hi
s_add_u32 s20, vcc_hi, vcc_lo
v_add_nc_u32_e32 v3, s20, v0
v_cmp_gt_u32_e32 vcc_lo, 0x90009, v0
v_mov_b32_e32 v6, 2.0
v_cndmask_b32_e32 v5, 1.0, v6, vcc_lo
s_waitcnt vmcnt(0)
exp mrt0 v8, v9, v3, v5 done vm
s_endpgm
''',
        '04190cf4000000fa040308f4200000fa800398be800399be80039abe80039bbef002027ef002047e7fc08cbf080f9cf0'
        '0108d900040410f4300000fa040810f4700000fa040c10f4b00000fa041010f4f00000fa041410f4300100fa041808f4'
        '700100fa80038abe80038bbec103febe861a20f4100000fa7fc08cbf6a836a93806aeb81f46beb9a6b6a14801400064a'
        'ff00887d09000900f4020c7ef20c0a02703f8cbf0f1800f808090305000081bf', None),
    'VS_DATA': ('''s_mov_b32 exec_lo, -1
v_cmp_gt_u32_e32 vcc_lo, 1, v0
v_mov_b32_e32 v1, 2.0
v_cndmask_b32_e32 v5, 1.0, v1, vcc_lo
s_mov_b32 vcc_hi, 0x40000000
s_mov_b32 vcc_lo, 4
s_mul_hi_u32 vcc_lo, vcc_hi, vcc_lo
s_add_u32 s20, vcc_lo, 0x3f800000
v_mov_b32_e32 v4, s20
exp pos0 v5, v5, v5, v4 done
s_endpgm
''',
        'c103febe8100887df402027ef2020a02f403ebbe8403eabe6b6aea9a6af214801402087ecf0800f805050504000081bf', None),
    'MIXED_JOIN': ('''s_cmp_eq_u32 s10, 0
s_cbranch_scc1 3
v_cmp_gt_u32_e64 vcc_hi, v0, 1
s_branch 2
s_mov_b32 vcc_hi, 0x55555555
v_cndmask_b32_e64 v2, 1.0, 2.0, vcc_hi
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''',
        '0a8006bf030085bf6b00c4d400030100020082bfff03ebbe55555555020001d5f2e8ad010f1800f802020202000081bf', 'v_cmp_gt_u32 into vcc_hi'),
    'HARDWIRED_MASK': ('''s_mov_b32 exec_lo, -1
s_load_dwordx4 s[12:15], s[8:9], 0x20
s_buffer_load_dword vcc_lo, s[12:15], 0x10
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 2.0
v_cndmask_b32_e32 v2, 1.0, v1, vcc_lo
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''',
        'c103febe040308f4200000fa861a20f4100000fa7fc08cbff402027ef20204020f1800f802020202000081bf', 's_buffer_load destination vcc_lo for 1 dwords'),
    'HARDWIRED_BRANCH': ('''s_mov_b32 exec_lo, -1
s_load_dwordx4 s[12:15], s[8:9], 0x20
s_buffer_load_dword vcc_lo, s[12:15], 0x10
s_waitcnt lgkmcnt(0)
s_cbranch_vccz 1
s_nop 0
v_mov_b32_e32 v2, 2.0
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''',
        'c103febe040308f4200000fa861a20f4100000fa7fc08cbf010086bf000080bff402047e0f1800f802020202000081bf', 's_buffer_load destination vcc_lo for 1 dwords'),
    'HARDWIRED_CARRY': ('''s_mov_b32 exec_lo, -1
s_load_dwordx4 s[12:15], s[8:9], 0x20
s_buffer_load_dword vcc_lo, s[12:15], 0x10
s_waitcnt lgkmcnt(0)
v_add_co_ci_u32_e32 v2, vcc_lo, 1, v0, vcc_lo
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''',
        'c103febe040308f4200000fa861a20f4100000fa7fc08cbf810004500f1800f802020202000081bf', 's_buffer_load destination vcc_lo for 1 dwords'),
    'HARDWIRED_JOIN': ('''s_mov_b32 exec_lo, -1
s_load_dwordx4 s[12:15], s[8:9], 0x20
s_mov_b32 vcc_lo, 0x55555555
s_cmp_eq_u32 s10, 0
s_cbranch_scc1 2
v_cmp_gt_u32_e32 vcc_lo, 1, v0
s_branch 1
s_addk_i32 vcc_lo, 0x100
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v1, 2.0
v_cndmask_b32_e32 v2, 1.0, v1, vcc_lo
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''',
        'c103febe040308f4200000faff03eabe555555550a8006bf020085bf8100887d010082bf0001eab77fc08cbff402027e'
        'f20204020f1800f802020202000081bf', 's_addk_i32 destination vcc_lo'),
    'ENTRY': ('''s_mov_b32 exec_lo, -1
s_add_u32 s20, vcc_hi, 1
v_cmp_gt_u32_e32 vcc_lo, 1, v0
v_add_nc_u32_e32 v2, s20, v0
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
''',
        'c103febe6b8114808100887d1400044a0f1800f802020202000081bf', "operand 'vcc_hi'"),
    'PARTIAL_MIXED': ('''s_mov_b32 exec_lo, -1
s_mov_b32 vcc_lo, 0x55555555
s_and_b32 vcc_lo, vcc_lo, -1
v_mov_b32_e32 v1, 2.0
v_cndmask_b32_e32 v2, 1.0, v1, vcc_lo
s_mov_b32 vcc_hi, 0x3f800000
s_add_u32 s20, vcc_hi, 1
v_mov_b32_e32 v3, s20
exp mrt0 v2, v3, v2, v3 done vm
s_endpgm
''',
        'c103febeff03eabe555555556ac16a87f402027ef2020402f203ebbe6b8114801402067e0f1800f802030203000081bf',
        's_mov_b32 destination vcc_hi'),
}


# SHA-256 of the exact HLSL emitted by 1aeb71894 for the original successful fixtures.
BASE_HLSL = {
    'PS': 'a2bd225589782462ee9eba5369cc0f225144de36dece7690401659234b9e35d1',
    'SCALAR_ONLY': '72f7187feab22d60cb106b1b8a9d83c6cb8cf4f0227ec202c1765e397e0235e2',
    'READFIRSTLANE_SCALAR': '5f495a754c51a8fb9a143b2520c1d7f11c4ef6f2a331b48548bb0c791d72e6b3',
}


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH  ' + str(detail)[:1500]))
    if not ok:
        failures.append(label)


def fixture_code(name):
    return bytes.fromhex(FIXTURES[name][1])


def translate(name, **kwargs):
    try:
        code = fixture_code(name)
        if name == 'VS_DATA':
            return G.Pipeline(code, None, root_blob(), **kwargs).hlsl_vs(), None
        return G.Pipeline(VS_WORDS, code, root_blob(), ps_hw_inputs=PACKED, **kwargs).hlsl_ps(), None
    except G.Unsupported as reason:
        return None, str(reason)


def image():
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            v0 = x | y << 16
            total = v0 + 0xfff7fff8
            out += [0x40000000 if v0 > 0x50005 else 0x3f800000, total & MASK, (v0 + 0x40000000 + (total >> 32)) & MASK,
                    (0x1244 + v0) & MASK]
    return out


def data_image():
    selected_word = {False: 0x3f800000, True: 0x40000000}
    lo = 0x80 * 2
    hi = (0x40000000 * (-lo & MASK)) >> 32
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pos = x | y << 16
            carry = (pos + 0xfff7fff8) >> 32
            out += [selected_word[pos > 0x50005], (0x3f800000 + lo + pos + carry) & MASK,
                    (0x10000000 + hi + pos) & MASK, selected_word[0x90009 > pos]]
    return out


def data2_image():
    scalar = 0x3f800000 + 16
    hi = MASK if scalar == 0x3f800010 else 0
    lo = MASK if scalar == 0 else 0
    selected_word = (0x3f800000, 0x40000000)
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pos = x | y << 16
            out += [selected_word[hi & 1], (0x3f800000 + pos + (hi & 1)) & MASK,
                    selected_word[lo & 1], (scalar + pos) & MASK]
    return out


def sub(text, pattern, replacement):
    mutated, count = re.subn(pattern, replacement, text)
    assert count == 1, pattern
    return mutated


# The renamed register is whichever SGPR the program never names; the patterns accept any number.
MUTATIONS = {
    'the compare bit is not read': (r'v2 = \(s(\d+) & 1\)', 'v2 = (0u & 1)'),
    'the carry-out is lost': (r'a = a \+ \(s(\d+) & 1u\);', 'a = a;'),
    'vcc_hi reads as zero': (r's0 = 0x00001234u \+', 's0 = 0x00000000u +'),
}

DATA_MUTATIONS = {
    'vcc_lo data reads as zero': (r's20 = s\d+ \+ 0x3f800000u;', 's20 = 0x3f800000u;'),
    'vcc_hi data is lost': (r's21 = 0x10000000u \+ s\d+;', 's21 = 0x10000000u;'),
    'the vcc_hi compare bit is ignored': (r'v2 = \(s\d+ & 1\)', 'v2 = (0u & 1)'),
    'the hardwired compare bit is ignored': (r'v5 = \(vcc & 1\)', 'v5 = (0u & 1)'),
}
DRAW_CASES = (
    ('PS', image, MUTATIONS),
    ('PS_DATA', data_image, DATA_MUTATIONS),
    ('PS_DATA2', data2_image, {'the uniform select is ignored': (r'v2 = \(s\d+ & 1\)', 'v2 = (0u & 1)')}),
)


def controls():
    for name, (_source, _words, refusal) in FIXTURES.items():
        try:
            text, reason = translate(name)
        except Exception as error:  # noqa: BLE001 - a crash is the failure this guards
            check('%s translates or refuses without a crash' % name, False, '%s: %s' % (type(error).__name__, error))
            continue
        if refusal is None:
            check('%s translates' % name, text is not None and 'vcc_hi' not in text, str(reason))
            if name in BASE_HLSL and text is not None:
                digest = hashlib.sha256(text.encode()).hexdigest()
                check('%s HLSL matches the base revision byte for byte' % name, digest == BASE_HLSL[name], digest)
        else:
            check('%s refused: %s' % (name, refusal), text is None and reason == refusal, str(reason))
    # A wide operand covering SGPR 107 (s_load_dwordx4 s[104:107]) writes vcc_hi: no free alias exists, so it stays refused.
    wide = [(0, 0, 's_mov_b32', 's_mov_b32 vcc_hi, 0x1234'),
            (4, 1, 's_load_dwordx4', 's_load_dwordx4 s[104:107], s[0:1], 0x0'),
            (12, 2, 'v_mov_b32_e32', 'v_mov_b32_e32 v2, vcc_hi')]
    check('a load into s[104:107] keeps vcc_hi refused', G.wave32_vcc_hi(wide) is wide)
    # Negative control for the whole-word writer rule: the readfirstlane mask-read fixtures must be stopped by
    # wave32_vcc_hi itself. The old rule (only s_ and v_readlane writers) renamed them, so this check fails there.
    for name in ('READFIRSTLANE_MASK', 'READFIRSTLANE_BRANCH', 'READFIRSTLANE_BACK_EDGE'):
        ins = G.decode(G.strip_header(bytes.fromhex(FIXTURES[name][1])))
        check('%s is refused before the vcc_hi rename' % name, G.wave32_vcc_hi(ins) is ins)
    text, _ = translate('PS')
    if text is not None:
        check('PS keeps the compare and carry in the renamed register',
              re.search(r's(\d+) = \(\(v0 > 0x00050005u\) \? 1u : 0u\);', text) is not None, text)
    for name, _, mutations in DRAW_CASES:
        text, _ = translate(name)
        if text is not None:
            for label, (pattern, replacement) in mutations.items():
                check('%s mutation "%s" changes the text' % (name, label), sub(text, pattern, replacement) != text)
    reference = image()
    check('reference covers both compare outcomes and both carries',
          {reference[i] for i in range(0, len(reference), 4)} == {0x3f800000, 0x40000000}
          and {(reference[i] - 0x40000000 - (i // 4) % WIDTH - ((i // 4 // WIDTH) << 16)) & MASK
               for i in range(2, len(reference), 4)} == {0, 1})
    reference = data_image()
    check('PS_DATA covers both outcomes of each comparison',
          all(set(reference[channel::4]) == {0x3f800000, 0x40000000} for channel in (0, 3)))
    check_split()


def check_split():
    with patch.object(G, 'wave32_vcc_split', return_value=None):
        for name, reason in (('PS_DATA', 'v_cmp_gt_u32 into vcc_hi'),
                             ('PS_DATA2', 's_mov_b32 destination vcc_hi'),
                             ('PS_DESCRIPTOR', 's_buffer_load destination vcc_lo for 1 dwords'),
                             ('VS_DATA', 's_mov_b32 destination vcc_hi')):
            _, refusal = translate(name)
            check('%s needs the retry' % name, refusal == reason, refusal)

    with patch.object(G, 'wave32_vcc_split', wraps=G.wave32_vcc_split) as split:
        pipe = G.Pipeline(VS_WORDS, fixture_code('PS_DATA'), root_blob(), ps_hw_inputs=PACKED)
        check('the first-lane retry keeps the split and builds it once',
              pipe.retry_firstlane and 's22 = v7;' in pipe.hlsl_ps() and split.call_count == 1)

    for name, kind in (('PS_DATA', 'ps'), ('VS_DATA', 'vs')):
        refusals = [('prior refusal', 'pipeline')]
        decoded = {kind: G.decode(fixture_code(name))}
        text, reason = translate(name, refusals=refusals, decoded=decoded)
        check('%s retry clears only the new refusals' % name,
              text is not None and text == translate(name)[0] and refusals == [('prior refusal', 'pipeline')],
              (reason, refusals))

    pipe = G.Pipeline(VS_WORDS, fixture_code('PS_DESCRIPTOR'), root_blob(), ps_hw_inputs=PACKED)
    hlsl = pipe.hlsl_ps()
    check('the texture stays in s[100:107] and the two data words use s7 and s6',
          [asm for _, _, _, asm in pipe.ps.ins if 's[100:107]' in asm]
          == [asm for _, _, _, asm in G.decode(fixture_code('PS_DESCRIPTOR')) if 's[100:107]' in asm]
          and 't = r0.SampleLevel(q0, float2(F(v1), F(v2)), 0);' in hlsl
          and 's7 = r1.Load(0x00000010u + 0);' in hlsl and 's6 = mul_hi_u32(0x40000000u, s6);' in hlsl
          and 'v5 = (vcc & 1) ? v6' in hlsl)

    for name in ('MIXED_JOIN', 'HARDWIRED_JOIN', 'HARDWIRED_MASK', 'HARDWIRED_BRANCH', 'HARDWIRED_CARRY',
                 'ENTRY', 'PAIR'):
        ins = G.decode(fixture_code(name))
        check('%s has a resolved CFG and stays pinned' % name, bool(G.lane_mask_cfg(ins)) and G.wave32_vcc_split(ins) is None)

    with patch.object(G, 'vcc_mixed_webs', return_value=set()):
        text, reason = translate('SCALAR_MASK')
        check('the mixed-web pin prevents the whole-word mask read', text is not None, reason)
    with patch.object(G, 'vcc_low_data_reaches', return_value=False):
        text, reason = translate('PARTIAL_MIXED')
        check('a separate rename cannot hide the low-word conflict', text is not None, reason)

    with patch.dict(os.environ, {'X2W_GFX_ASSUME_WAVE64': '1'}):
        text, reason = translate('PS_DATA')
        check('the wave64 assumption vetoes an odd mask field without a wave32 EXEC form',
              text is None and reason == 'v_cmp_gt_u32 into vcc_hi', reason)

    check_pair_and_uniform_masks()


def check_pair_and_uniform_masks():
    original = G.decode(fixture_code('PAIR_REACH'))
    split = G.wave32_vcc_split(original)
    check('a pair definition pins its high-word reader and leaves the surrounding writes free',
          split is not None and split[2:4] == original[2:4]
          and re.fullmatch(r's_mov_b32 s\d+, 2\.0', split[1][3]) is not None
          and re.fullmatch(r's_mov_b32 s\d+, 1\.0', split[4][3]) is not None
          and split[5][3] != original[5][3])

    pipe = G.Pipeline(VS_WORDS, fixture_code('PS_DATA2'), root_blob(), ps_hw_inputs=PACKED)
    check('PS_DATA2 preserves the encoded pair write',
          pipe.ps.ins[0] == G.decode(fixture_code('PS_DATA2'))[0])
    hlsl = pipe.hlsl_ps()
    select = re.search(r'(s\d+) = scc \? 0xffffffffu : 0x00000000u;\s+v2 = \(\1 & 1\)', hlsl)
    check('PS_DATA2 reads the scalar select bit for both cndmask and carry-in',
          select is not None and 'a = a + (%s & 1u);' % select[1] in hlsl)
    with patch.object(G, 'vcc_uniform_mask_def', return_value=False):
        text, reason = translate('PS_DATA2')
        check('PS_DATA2 needs the uniform-mask exemption', text is None and reason is not None, reason)


def draw(renderer, adapter, vs, ps, expected):
    run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT),
                          str(expected)], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(1 if failures else 77)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs_' + kind, vs_profile, directory)
            for name, expected, mutations in DRAW_CASES:
                text, reason = translate(name)
                if text is None:
                    check('%s translates for the draw' % name, False, reason)
                    continue
                stem = '%s_%s' % (name, kind)
                reference = directory / (stem + '.bin')
                reference.write_bytes(struct.pack('<%dI' % (WIDTH * HEIGHT * 4), *expected()))
                ps = compile_hlsl(compiler, kind, text, stem, ps_profile, directory)
                run = draw(renderer, adapter, vs, ps, reference)
                check('%s %s %s: all %d pixels match the reference' % (adapter, kind, name, WIDTH * HEIGHT),
                      run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                      run.stdout + run.stderr)
                for n, (label, (pattern, replacement)) in enumerate(mutations.items()):
                    bad = compile_hlsl(compiler, kind, sub(text, pattern, replacement), '%s_bad_%d' % (stem, n),
                                       ps_profile, directory)
                    run = draw(renderer, adapter, vs, bad, reference)
                    check('%s %s %s mutation "%s" fails' % (adapter, kind, name, label),
                          run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)


def check_binary(llvm_mc):
    for name, (source, words, _refusal) in FIXTURES.items():
        if name == 'PS_DESCRIPTOR':
            # LLVM 19 rejects the spelling s[100:107]. Assemble the adjacent legal range and advance
            # SMEM SDST by four and MIMG SRSRC by one four-word slot.
            binary = assemble(llvm_mc, source.replace('s[100:107]', 's[96:103]'))
            binary = binary.replace(bytes.fromhex('04180cf4'), bytes.fromhex('04190cf4'), 1)
            binary = binary.replace(bytes.fromhex('0108d800'), bytes.fromhex('0108d900'), 1)
        else:
            binary = assemble(llvm_mc, source)
        check('llvm-mc assembles the %s fixture to the same words' % name,
              binary == bytes.fromhex(words))


def main():
    if len(sys.argv) == 1:
        controls()
    else:
        mode, *args = sys.argv[1:]
        if mode == 'binary':
            check_binary(*args)
        elif mode == 'draw':
            controls()
            check_draw(*args)
        else:
            raise SystemExit(__doc__)
    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
