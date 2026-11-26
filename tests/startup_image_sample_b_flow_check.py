"""image_sample_b past a branch or an EXEC change in tools/startup/gfx2hlsl.py (implicit derivatives of the quad).

  (no arguments)                       translation controls and refusals, pure Python
  binary <llvm-mc>                     reassemble every fixture source and compare it with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given), draw each accepted
                                       fixture and compare the readback with the CPU reference; each mutant must differ

Semantics. image_sample_b adds its bias to the LOD the texture unit derives from the address VGPRs of the four lanes of
the pixel quad (RDNA 2 ISA section 8.2.4, "_B LOD BIAS: Add this BIAS to the LOD computed"). The lanes are read whatever
their EXEC bits (that is why compilers switch to whole quad mode before computing coordinates): a lane EXEC masked off
offers the value it last wrote. The translation runs one guest lane per D3D pixel or helper lane, writes a VGPR only
where the lane's EXEC bit is 1 and calls SampleBias unconditionally, so HLSL derives from the same four values when the
four lanes reach the call together. In a pixel program with such an op and no backward branch, s_cbranch_execz/execnz
are decided per quad (x2w_quad_any, fine derivatives of the EXEC bit): the guest's wave skips a
block only when no lane is active and runs it with the other lanes masked, so a quad with an active lane stays together.
A voted branch parts no quad, so it does not clear Stage.uniform_cf; the vote reads the quad, so it is only taken while
uniform_cf holds and no lane-exec break (a discard under an unknown EXEC, an s_wqm of other operands) came first.
s_wqm_b64 exec, exec on a narrowed EXEC is the same vote written back to EXEC (exec = x2w_quad_any(exec) ? 1u : 0u). Any other branch that can
differ between lanes (a VCC test, an SCC not proven uniform) clears uniform_cf, as in #2659.

Fixtures (synthetic gfx1013 wave64 programs, no game data). v0 is the packed pixel position (POS_FIXED_PT, the only
pixel input enabled), x = v0 & 0xffff. Every program starts with `s_wqm_b64 exec, exec`, sets u = 0.125 + 0.5 * x,
v = 0.25, the slice from root constant 1 (s9, drawn as 0), the bias -1, and samples the 4x4, two-slice, three-mip array
of tests/startup_image_3d.cpp (--startup-image-2d-array-mips: one covered pixel at (0, 0) and its three helper lanes)
with a point sampler, mip filter point, MAX_LOD 2. u depends on x only, so the LOD is log2(4 * |u(x=1) - u(x=0)|) - 1:
  join_pixel   lanes with x == 0 (the drawn pixel's column) branch into u -= 0.5, then EXEC is restored and the quad
               samples at the join: u = (-0.375, 0.625), LOD 2 - 1 = 1
  join_helper  the helper column x == 1 branches into u += 0.5: u = (0.125, 1.125), LOD 1; the drawn pixel did not branch
  cmpx         v_cmpx keeps x == 0, u -= 0.5 runs on those lanes only and the sample runs with that EXEC: the helper column
               offers its stale u = 0.625, LOD 2 - 1 = 1 (u = 0.125 there would give LOD 0)
  scc          a branch on root constant 1 (the same in every lane, not taken) before the sample: LOD 1 - 1 = 0
  inside       the sample inside the s_cbranch_execz block of x == 0, after u -= 0.5 there: the quad enters together,
               the helper column masked with its stale u: LOD 1
  kill         v_cmpx keeps x == 0, s_cbranch_execnz past an s_endpgm, u -= 0.5, sample: the quad goes on, LOD 1
  inside_y     u is the same in every lane and v = 0.25 + 0.5 * y: the block is entered by the row y == 0 only, the quad
               stays together and the LOD comes from the vertical derivative: LOD 1 - 1 = 0. The vote itself is drawn
               alone (VOTE_CASES): one lane of the quad set at each position, and each half of the helper dropped.
  uni_loop     a loop of two passes on a constant SGPR counter (SCC proven uniform): sample, then u -= 0.5 on every
               lane; the second pass reads u = (-0.375, 0.125), LOD 0
Refused: the sample inside or after a block a VCC branch skips (the guest takes a VCC branch once per wave, with EXEC
unchanged, so a lane skipping it on its own bit can hold another u), and inside a loop whose exit reads VCC.
"""
import math
import re
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_2d_array_check import mip_cell  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

PS_INPUTS = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA / ADDR: POS_FIXED_PT only, so v0 holds x | y << 16
PROLOGUE = """\
s_wqm_b64 exec, exec
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, 0x200000
s_mov_b32 s26, 0x4000000
s_mov_b32 s27, 0
v_and_b32_e32 v8, 0xffff, v0
v_cvt_f32_u32_e32 v9, v8
v_mul_f32_e32 v9, 0.5, v9
v_add_f32_e32 v2, 0x3e000000, v9
v_mov_b32_e32 v1, -1.0
v_mov_b32_e32 v3, 0x3e800000
v_mov_b32_e32 v4, s9
"""
SAMPLE = """\
s_waitcnt lgkmcnt(0)
image_sample_b v[10:13], v[1:4], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
"""
EXPORT = """\
s_waitcnt vmcnt(0)
exp mrt0 v10, v11, v12, v13 done vm
s_endpgm
"""


def branch_on_x(x, delta):
    return ('v_cmp_eq_u32_e32 vcc, %d, v8\ns_and_saveexec_b64 s[0:1], vcc\ns_cbranch_execz BB1\n'
            'v_add_f32_e32 v2, %s, v2\nBB1:\ns_or_b64 exec, exec, s[0:1]\n' % (x, delta))


# name: (source, u at x = 0 and x = 1 when the quad samples, bias) for the drawn fixtures
SUB = 'v_add_f32_e32 v2, -0.5, v2\n'
CMPX = 'v_cmpx_eq_u32_e32 0, v8\n'
FIXTURES = {
    'join_pixel': (PROLOGUE + branch_on_x(0, '-0.5') + SAMPLE + EXPORT, (Fraction(-3, 8), Fraction(5, 8)), -1),
    'join_helper': (PROLOGUE + branch_on_x(1, '0.5') + SAMPLE + EXPORT, (Fraction(1, 8), Fraction(9, 8)), -1),
    'cmpx': (PROLOGUE + CMPX + SUB + SAMPLE + EXPORT, (Fraction(-3, 8), Fraction(5, 8)), -1),
    'scc': (PROLOGUE + 's_cmp_lg_u32 s9, 0\ns_cbranch_scc1 BB1\n' + SAMPLE + 'BB1:\n' + EXPORT,
            (Fraction(1, 8), Fraction(5, 8)), -1),
    'inside': (PROLOGUE + 'v_cmp_eq_u32_e32 vcc, 0, v8\ns_and_saveexec_b64 s[0:1], vcc\ns_cbranch_execz BB1\n' + SUB
               + SAMPLE + 'BB1:\ns_or_b64 exec, exec, s[0:1]\n' + EXPORT, (Fraction(-3, 8), Fraction(5, 8)), -1),
    'kill': (PROLOGUE + CMPX + 's_cbranch_execnz BB1\ns_endpgm\nBB1:\n' + SUB + SAMPLE + EXPORT,
             (Fraction(-3, 8), Fraction(5, 8)), -1),
    # u is the same in every lane and v = 0.25 + 0.5 * y, so the LOD comes from the vertical derivative alone; only the row
    # y == 0 enters the block. (u0, u1) is (u of the drawn pixel, u + the footprint along y): 4 * 0.5 = 2, LOD 1 - 1 = 0.
    'inside_y': (PROLOGUE + 'v_mov_b32_e32 v2, 0x3e000000\nv_lshrrev_b32_e32 v7, 16, v0\nv_cvt_f32_u32_e32 v6, v7\nv_mul_f32_e32 v6, 0.5, v6\nv_add_f32_e32 v3, 0x3e800000, v6\nv_cmp_eq_u32_e32 vcc, 0, v7\ns_and_saveexec_b64 s[0:1], vcc\ns_cbranch_execz BB1\nv_mov_b32_e32 v14, 0\n'
                 + SAMPLE + 'BB1:\ns_or_b64 exec, exec, s[0:1]\n' + EXPORT, (Fraction(1, 8), Fraction(5, 8)), -1),
    'uni_loop': (PROLOGUE + 's_mov_b32 s30, 2\nBB1:\n' + SAMPLE + SUB + 's_add_i32 s30, s30, -1\ns_cmp_lg_u32 s30, 0\n'
                 's_cbranch_scc1 BB1\n' + EXPORT, (Fraction(-3, 8), Fraction(1, 8)), -1),
    # s_wqm exec, exec on a narrowed EXEC turns the whole quad back on: every lane takes the subtraction (u1 = 1/8, not 5/8).
    'wqm': (PROLOGUE + CMPX + 's_wqm_b64 exec, exec\n' + SUB + SAMPLE + EXPORT,
            (Fraction(-3, 8), Fraction(1, 8)), -1),
    # The bias differs per lane: the drawn lane (x = 0) takes -2, the helper column keeps -1. LOD 2 - 2 = 0 (mip 0); the
    # helper's -1 would give mip 1 and no bias mip 2.
    'bias_lane': (PROLOGUE + 'v_mov_b32_e32 v14, -2.0\nv_cmp_eq_u32_e32 vcc, 0, v8\nv_cndmask_b32_e32 v1, v1, v14, vcc\n'
                  + branch_on_x(0, '-0.5') + SAMPLE + EXPORT, (Fraction(-3, 8), Fraction(5, 8)), -2),
    # The bias is written INSIDE the divergent block. The drawn lane (x = 0) writes +1: LOD 1 + 1 = 2 (mip 2); without the
    # write its -1 gives mip 0.
    'bias_in_block': (PROLOGUE + 'v_cmp_eq_u32_e32 vcc, 0, v8\ns_and_saveexec_b64 s[0:1], vcc\ns_cbranch_execz BB1\nv_mov_b32_e32 v1, 1.0\n'
                          'BB1:\ns_or_b64 exec, exec, s[0:1]\n' + SAMPLE + EXPORT,
                      (Fraction(1, 8), Fraction(5, 8)), 1),
    # Only the helper column (x = 1) writes +1: the drawn lane keeps -1 (mip 0); a write by every lane gives mip 2.
    'bias_in_block_helper': (PROLOGUE + 'v_cmp_eq_u32_e32 vcc, 1, v8\ns_and_saveexec_b64 s[0:1], vcc\ns_cbranch_execz BB1\nv_mov_b32_e32 v1, 1.0\n'
                          'BB1:\ns_or_b64 exec, exec, s[0:1]\n' + SAMPLE + EXPORT,
                             (Fraction(1, 8), Fraction(5, 8)), -1),
}
# name: (source, text of the refusal)
REFUSED = {
    # The plain s_mov_b64 exec, s[N] restore of a mask saved before s_wqm is a lane-exec break too (main accepts the form).
    'restore_mov': ('s_mov_b64 s[12:13], exec\n' + PROLOGUE + 's_mov_b64 exec, s[12:13]\nv_add_f32_e32 v2, -0.5, v2\n' + SAMPLE + EXPORT,
                    'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # A constant 32-bit EXEC write is no more exact than a mask one: lanes 32..63 keep their old bits in wave64.
    'exec_lo_const': (PROLOGUE + 's_mov_b32 exec_lo, -1\nv_add_f32_e32 v2, -0.5, v2\n' + SAMPLE + EXPORT,
                      'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # A mask saved before s_wqm exec, exec has the guest's helper lanes off, while every thread here starts
    # with exec = 1: writing it back is a lane-exec break (review of ef786354).
    'restore': ('s_mov_b64 s[12:13], exec\n' + PROLOGUE + 's_and_b64 exec, exec, s[12:13]\n' + SAMPLE + EXPORT,
                   'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # The same restore, then u -= 0.5 on the lanes it leaves on: the guest's helper keeps its stale u, the
    # translation's helper takes the new one (measured by the reviewer on WARP and an RX 9070 XT).
    'restore_helper': ('s_mov_b64 s[12:13], exec\n' + PROLOGUE + 's_and_b64 exec, exec, s[12:13]\n'
                      'v_add_f32_e32 v2, -0.5, v2\n' + SAMPLE + EXPORT,
                   'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # A kill export under v_cmpx before the op: the discard is a lane-exec break (it fails without it).
    'discard_break': (PROLOGUE + 'v_cmpx_eq_u32_e32 0, v8\nexp null off, off, off, off vm\n' + SAMPLE + EXPORT,
                   'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # A 32-bit EXEC write keeps lanes 32..63 of the old mask in wave64: only the 64-bit forms are exact.
    'exec_lo_mask': (PROLOGUE + 'v_cmp_eq_u32_e32 vcc, 0, v8\ns_mov_b32 exec_lo, vcc_lo\n'
                         'v_add_f32_e32 v2, -0.5, v2\n' + SAMPLE + EXPORT,
                   'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # The wave takes a VCC branch once for all lanes and runs the block with EXEC unchanged, while a lane here follows
    # its own bit: it clears uniform_cf, so the sample after it is refused (#2659).
    'vcc_inside': (PROLOGUE + 'v_cmp_eq_u32_e32 vcc, 0, v8\ns_cbranch_vccz BB1\n' + SAMPLE + 'BB1:\n' + EXPORT,
                   'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    'vcc_join': (PROLOGUE + 'v_cmp_eq_u32_e32 vcc, 0, v8\ns_cbranch_vccz BB1\n' + SUB + 'BB1:\n' + SAMPLE + EXPORT,
                 'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    'loop': (PROLOGUE + 'BB1:\n' + SAMPLE + 's_waitcnt vmcnt(0)\nv_cmp_eq_u32_e32 vcc, 0, v8\ns_cbranch_vccnz BB1\n'
             + EXPORT, 'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    # A scalar write inside the block an EXEC branch can skip happens once for the wave in the guest; a quad that
    # votes to skip it would miss it (review of the first version): the branch is not voted, the op is refused.
    'scalar_in_block': (PROLOGUE + 's_mov_b32 s28, -1.0\nv_cmp_eq_u32_e32 vcc, 2, v8\ns_and_saveexec_b64 s[0:1], vcc\ns_cbranch_execz BB1\n'
                        's_mov_b32 s28, 0\nBB1:\ns_or_b64 exec, exec, s[0:1]\nv_mov_b32_e32 v1, s28\n'
                        + SAMPLE + EXPORT, 'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    'carry_in_block': (PROLOGUE + 's_mov_b64 s[28:29], exec\nv_cmp_eq_u32_e32 vcc, 2, v8\ns_and_saveexec_b64 s[0:1], vcc\ns_cbranch_execz BB1\nv_add_co_u32_e64 v14, s[28:29], 0, 0\nBB1:\ns_or_b64 exec, exec, s[0:1]\ns_or_b64 exec, exec, s[28:29]\ns_waitcnt lgkmcnt(0)\nimage_sample_b v[10:13], v[1:4], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY\ns_waitcnt vmcnt(0)\nexp mrt0 v10, v11, v12, v13 done vm\ns_endpgm\n', 'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    'exec_const_mask': (PROLOGUE + 's_mov_b64 s[12:13], exec\ns_mov_b64 exec, 1\nv_add_f32_e32 v2, 0.5, v2\ns_mov_b64 exec, s[12:13]\ns_waitcnt lgkmcnt(0)\nimage_sample_b v[10:13], v[1:4], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY\ns_waitcnt vmcnt(0)\nexp mrt0 v10, v11, v12, v13 done vm\ns_endpgm\n', 'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    'kill_then_restore': (PROLOGUE + 's_mov_b64 s[12:13], exec\nv_cmpx_eq_u32_e32 0, v8\ns_cbranch_execnz BB1\ns_endpgm\nBB1:\ns_mov_b64 exec, s[12:13]\nv_add_f32_e32 v2, -0.5, v2\ns_waitcnt lgkmcnt(0)\nimage_sample_b v[10:13], v[1:4], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY\ns_waitcnt vmcnt(0)\nexp mrt0 v10, v11, v12, v13 done vm\ns_endpgm\n', 'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
}
# llvm-mc-19 -arch=amdgcn -mcpu=gfx1013 -mattr=+wavefrontsize64,-wavefrontsize32 of the label-resolved sources
WORDS_HEX = {
    'restore_mov': (
        '7e048cbe7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff000008'
        '0d127ef0121210ff1204060000003ef302027eff02067e0000803e0902087e0c04febef10404067fc08cbf280f94f0010a'
        'c400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'inside_y': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0'
        '121210ff1204060000003ef302027eff02067e0000803e0902087eff02047e0000003e90000e2c070d0c7ef00c0c10ff0c'
        '06060000803e800e847d6a2480be040088bf80021c7e7fc08cbf280f94f0010ac4007e00fe88703f8cbf0f1800f80a0b0c'
        '0d000081bf'
    ),
    'exec_lo_const': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0'
        '121210ff1204060000003ef302027eff02067e0000803e0902087ec103febef10404067fc08cbf280f94f0010ac400703f'
        '8cbf0f1800f80a0b0c0d000081bf'
    ),
    'restore_helper': (
        '7e048cbe7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff000008'
        '0d127ef0121210ff1204060000003ef302027eff02067e0000803e0902087e7e0cfe87f10404067fc08cbf280f94f0010a'
        'c400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'discard_break': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0'
        '121210ff1204060000003ef302027eff02067e0000803e0902087e8010a47d901000f8000000007fc08cbf280f94f0010a'
        'c400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'exec_lo_mask': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0'
        '121210ff1204060000003ef302027eff02067e0000803e0902087e8010847d6a03febef10404067fc08cbf280f94f0010a'
        'c400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'join_pixel': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e8010847d6a2480be010088bff10404067e00fe88'
        '7fc08cbf280f94f0010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'join_helper': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e8110847d6a2480be010088bff00404067e00fe88'
        '7fc08cbf280f94f0010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'cmpx': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e8010a47df10404067fc08cbf280f94f0010ac400'
        '703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'scc': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e098007bf030085bf7fc08cbf280f94f0010ac400'
        '703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'inside': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e8010847d6a2480be040088bff10404067fc08cbf'
        '280f94f0010ac4007e00fe88703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'kill': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e8010a47d010089bf000081bff10404067fc08cbf'
        '280f94f0010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'uni_loop': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e82039ebe7fc08cbf280f94f0010ac400f1040406'
        '1ec11e811e8007bff9ff85bf703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'vcc_inside': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e8010847d030086bf7fc08cbf280f94f0010ac400'
        '703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'vcc_join': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e8010847d010086bff10404067fc08cbf280f94f0'
        '010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'loop': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e7fc08cbf280f94f0010ac400703f8cbf8010847d'
        'faff87bf703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'scalar_in_block': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0121210ff1204060000003ef302027eff02067e0000803e0902087ef3039cbe8210847d6a2480be010088bf80039cbe7e00fe881c02027e7fc08cbf280f94f0010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'restore': (
        '7e048cbe7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0121210ff1204060000003ef302027eff02067e0000803e0902087e7e0cfe877fc08cbf280f94f0010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'carry_in_block': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e7e049cbe8210847d6a2480be020088bf0e1c0fd7'
        '800001007e00fe887e1cfe887fc08cbf280f94f0010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'exec_const_mask': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e7e048cbe8104febef00404060c04febe7fc08cbf'
        '280f94f0010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'kill_then_restore': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e7e048cbe8010a47d010089bf000081bf0c04febe'
        'f10404067fc08cbf280f94f0010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'wqm': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127e'
        'f0121210ff1204060000003ef302027eff02067e0000803e0902087e8010a47d7e0afebef10404067fc08cbf280f94f0'
        '010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'bias_lane': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0121210ff1204060000003ef302027eff02067e0000803e0902087ef5021c7e8010847d011d02028010847d6a2480be010088bff10404067e00fe887fc08cbf280f94f0010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'bias_in_block': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0121210ff1204060000003ef302027eff02067e0000803e0902087e8010847d6a2480be010088bff202027e7e00fe887fc08cbf280f94f0010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
    'bias_in_block_helper': (
        '7e0afebe04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbeff001036ffff0000080d127ef0121210ff1204060000003ef302027eff02067e0000803e0902087e8110847d6a2480be010088bff202027e7e00fe887fc08cbf280f94f0010ac400703f8cbf0f1800f80a0b0c0d000081bf'
    ),
}
BRANCH = re.compile(r'^(s_c?branch\w*) (BB\d+)$')


def resolve(source, sizes):
    """The source with each label operand replaced by its SIMM16 (dwords after the next instruction)."""
    lines = [line for line in source.splitlines() if line]
    labels, at, off = {}, [], 0
    for line in lines:
        if line.endswith(':'):
            labels[line[:-1]] = off
        else:
            at.append(off)
            off += sizes[len(at) - 1]
    out = []
    for line in (line for line in lines if not line.endswith(':')):
        m = BRANCH.match(line)
        if m:
            line = '%s %d' % (m.group(1), (labels[m.group(2)] - at[len(out)] - 4) // 4)
        out.append(line)
    return '\n'.join(out) + '\n'


def assemble(llvm_mc, source):
    """Words of a labelled source: one pass with every branch at 0 sizes the instructions, the second encodes."""
    def encode(text):
        run = subprocess.run([llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=+wavefrontsize64,-wavefrontsize32',
                              '-show-encoding'], input=text, capture_output=True, text=True, check=True)
        return [bytes(int(b, 16) for b in m.group(1).split(','))
                for m in re.finditer(r'encoding: \[([^\]]*)\]', run.stdout)]
    plain = [BRANCH.sub(r'\1 0', line) for line in source.splitlines() if line and not line.endswith(':')]
    sizes = [len(w) for w in encode('\n'.join(plain) + '\n')]
    assert len(sizes) == len(plain), (len(sizes), len(plain))
    return b''.join(encode(resolve(source, sizes)))


def all_sources():
    return dict({k: v[0] for k, v in FIXTURES.items()}, **{k: v[0] for k, v in REFUSED.items()})


def words(name):
    return bytes.fromhex(WORDS_HEX[name])


def pipeline(name):
    return G.Pipeline(VS_WORDS, words(name), root_blob(), ps_hw_inputs=PS_INPUTS)


def refused(name, text):
    try:
        pipeline(name).hlsl_ps()
    except G.Unsupported as error:
        assert text in str(error), (name, text, str(error))
    else:
        raise AssertionError('%s accepted, expected %r' % (name, text))


def reference(name):
    """The texel the drawn pixel reads: LOD log2(4 |du/dx|) + bias (u is the same down a column), mip filter point."""
    _source, (u0, u1), bias = FIXTURES[name]
    footprint = 4 * abs(u1 - u0)
    lod = math.log2(footprint)
    assert lod == int(lod), (name, footprint)
    mip = min(max(int(lod) + bias, 0), 2)
    size = 4 >> mip
    wrapped = u0 - math.floor(u0)
    return mip_cell(int(wrapped * size), int(Fraction(1, 4) * size), 0, mip)


CALL = 't = r0.SampleBias(q0, float3(F(v2), F(v3), F(v4)), F(v1));'


def controls():
    for name in FIXTURES:
        text = pipeline(name).hlsl_ps()
        assert text.count(CALL) == 1, (name, text)
        assert 'SampleLevel' not in text, (name, text)
    for name, (_source, text) in REFUSED.items():
        refused(name, text)
    # The quad votes: the EXEC branches of inside, kill and join_pixel, one vote each; v_cmpx alone needs none.
    for name, votes in (('inside', 1), ('kill', 1), ('cmpx', 0), ('join_pixel', 1), ('join_helper', 1), ('scc', 0), ('inside_y', 1), ('wqm', 1), ('bias_lane', 1), ('bias_in_block', 1), ('bias_in_block_helper', 1)):
        text = pipeline(name).hlsl_ps()
        assert text.count('x2w_quad_any(exec)') == votes, (name, text)
    # The gate is the vote's: with the EXEC rule of #2659 (the entry mask only) the divergent-EXEC fixtures are refused.
    saved = G.Stage.derivative_exec_ok
    G.Stage.derivative_exec_ok = lambda self: self.exec_known
    try:
        for name in ('cmpx', 'inside', 'kill'):
            refused(name, 'image_sample_b needs implicit derivatives')
    finally:
        G.Stage.derivative_exec_ok = saved
    # The drawn vote helper is the text the translation emits.
    assert '\n'.join(G.QUAD_ANY_HLSL) in pipeline('inside').hlsl_ps()
    cells = {name: reference(name) for name in FIXTURES}
    print('image_sample_b after a join, under a v_cmpx EXEC, after a uniform branch, inside an EXEC branch and '
          'past a kill branch: SampleBias; references %s'
          % ', '.join('%s=%s' % kv for kv in cells.items()))
    print('refused: inside or after a block a VCC branch skips, in a VCC loop')


def check_binary(llvm_mc):
    for name, source in all_sources().items():
        built = assemble(llvm_mc, source)
        if built != words(name):
            raise AssertionError('%s: assembled %s, recorded %s' % (name, built.hex(), WORDS_HEX[name]))
    print('every fixture reassembles to its recorded words with %s' % llvm_mc)


def mutants(name, hlsl):
    """HLSL variants that must read another texel: the branch write dropped, the EXEC guard of the u write dropped,
    the bias dropped."""
    out = []
    write = re.search(r'\n\s*if \(exec & 1\) v2 = U\(-?0\.5 \+ F\(v2\)\);', hlsl)
    if name.startswith('join'):
        out.append(('without the branch write', hlsl.replace(write.group(0), '')))
    if name in ('cmpx', 'inside', 'kill'):
        out.append(('every lane writes u', hlsl.replace(write.group(0), write.group(0).replace('if (exec & 1) ', ''))))
    if name == 'wqm':
        out.append(('s_wqm as a no-op', hlsl.replace('exec = x2w_quad_any(exec) ? 1u : 0u;', '')))
    wr = re.search(r'\n\s*if \(exec & 1\) v1 = 0x3f800000u;', hlsl)
    if name == 'bias_in_block':
        out.append(('without the block write', hlsl.replace(wr.group(0), '')))
    if name == 'bias_in_block_helper':
        out.append(('every lane writes the bias', hlsl.replace(wr.group(0), wr.group(0).replace('if (exec & 1) ', ''))))
    if name == 'bias_lane':
        out.append(('the helper column bias', hlsl.replace(CALL, CALL.replace('F(v1))', '-1.0)'))))
    out.append(('bias ignored', hlsl.replace(CALL, CALL.replace('F(v1))', '0.0)'))))
    for label, text in out:
        assert text != hlsl, (name, label)
    return out


# The vote helper alone (QUAD_ANY_HLSL, the text the translation emits), drawn on the covered pixel (0, 0): a pixel shader
# whose bit is set on one lane of the quad only. The covered pixel answers 1 for every position of that lane, and each
# half of the helper is the only way to see one of them: the drawn lane is in row 0, column 0, so a lane at (1, 0) is
# seen through ddx_fine and a lane at (0, 1) through ddy_fine, and a lane at (1, 1) through both. Red is the answer.
VOTE_PS = """
%s
float4 main(float4 pos : SV_Position) : SV_Target {
    uint x = (uint)pos.x & 1u, y = (uint)pos.y & 1u;
    return float4(x2w_quad_any(%s) ? 1.0 : 0.0, 0.0, 0.0, 1.0);
}
"""
ONE, ZERO = '3f800000,00000000,00000000,3f800000', '00000000,00000000,00000000,3f800000'
# name: (bit of the lane, red answer of the covered pixel); the quad is x, y in {0, 1}^2
VOTE_CASES = {
    'lane_10': ('(x == 1u && y == 0u) ? 1u : 0u', ONE),
    'lane_01': ('(x == 0u && y == 1u) ? 1u : 0u', ONE),
    'lane_11': ('(x == 1u && y == 1u) ? 1u : 0u', ONE),
    'lane_00': ('(x == 0u && y == 0u) ? 1u : 0u', ONE),
    'none': ('0u', ZERO),
}
# what each case must stop answering when a half of the helper is dropped
VOTE_MUTANTS = {
    'lane_10': ('abs(ddx_fine(lane))', 'without the horizontal derivative'),
    'lane_01': ('abs(ddy_fine(row))', 'without the vertical derivative'),
}


def vote_hlsl(bit, mutant=None):
    helper = '\n'.join(G.QUAD_ANY_HLSL)
    if mutant:
        assert mutant in helper, mutant
        helper = helper.replace(mutant, '0.0')
    return VOTE_PS % (helper, bit)


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])

    def draw(vs, ps, expected, sampler):
        run = subprocess.run([renderer, adapter, '--startup-image-2d-array-mips', str(vs), str(ps), '0', expected]
                             + ([sampler] if sampler else []), capture_output=True, text=True)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            sys.exit(77)
        return run

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            for name in FIXTURES:
                pipe = pipeline(name)
                hlsl = pipe.hlsl_ps()
                sampler = [line for line in pipe.manifest().splitlines() if line.startswith(('smp ', 'smpl '))]
                assert len(sampler) == 1, pipe.manifest()
                want = reference(name)
                ps = compile_hlsl(compiler, kind, hlsl, '%s_%s' % (name, kind), ps_profile, directory)
                run = draw(vs, ps, want, sampler[0])
                if run.returncode:
                    raise AssertionError('%s %s %s: GPU differs from the CPU reference %s\n%s%s'
                                         % (adapter, kind, name, want, run.stdout, run.stderr))
                seen = []
                for i, (label, text) in enumerate(mutants(name, hlsl)):
                    ps = compile_hlsl(compiler, kind, text, '%s_m%d_%s' % (name, i, kind), ps_profile, directory)
                    run = draw(vs, ps, want, sampler[0])
                    got = re.search(r'actual=(\S+)', run.stderr)
                    if run.returncode == 0 or not got:
                        raise AssertionError('%s %s %s: mutant "%s" matched the reference\n%s%s'
                                             % (adapter, kind, name, label, run.stdout, run.stderr))
                    seen.append('%s -> %s' % (label, got.group(1)))
                print('%s %s %s: GPU == CPU reference %s; %s' % (adapter, kind, name, want, '; '.join(seen)),
                      flush=True)
            for name, (bit, want) in VOTE_CASES.items():
                ps = compile_hlsl(compiler, kind, vote_hlsl(bit), 'vote_%s_%s' % (name, kind), ps_profile, directory)
                run = draw(vs, ps, want, None)
                if run.returncode:
                    raise AssertionError('%s %s vote %s: GPU differs from the CPU reference %s\n%s%s'
                                         % (adapter, kind, name, want, run.stdout, run.stderr))
                seen = []
                if name in VOTE_MUTANTS:
                    cut, label = VOTE_MUTANTS[name]
                    ps = compile_hlsl(compiler, kind, vote_hlsl(bit, cut), 'vote_%s_m_%s' % (name, kind),
                                      ps_profile, directory)
                    run = draw(vs, ps, want, None)
                    got = re.search(r'actual=(\S+)', run.stderr)
                    if run.returncode == 0 or not got:
                        raise AssertionError('%s %s vote %s: the helper %s matched the reference\n%s%s'
                                             % (adapter, kind, name, label, run.stdout, run.stderr))
                    seen.append('%s -> %s' % (label, got.group(1)))
                print('%s %s vote %s: GPU == CPU reference %s; %s' % (adapter, kind, name, want, '; '.join(seen)),
                      flush=True)


def main():
    if len(sys.argv) == 1 or sys.argv[1] == 'controls':
        controls()
    elif sys.argv[1] == 'binary' and len(sys.argv) == 3:
        check_binary(sys.argv[2])
    elif sys.argv[1] == 'draw' and len(sys.argv) in (5, 6):
        controls()
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
