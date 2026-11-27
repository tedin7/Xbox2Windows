"""D16 image_load and image_load_mip lowering in tools/startup/gfx2hlsl.py (Part of #2023, item D).

  (no arguments)                       emitted-HLSL rounding, decoder, SRV format and odd-dmask controls
  binary <llvm-mc>                     reassemble every fixture source and compare it with the recorded words
  decode <libLLVM>                     disassemble the MIMG words with LLVM's C API and compare with disasm_rdna2.py
  odd-draw <dxc> <offscreen_check> warp
                                       check three odd-tail shaders on an RGBA16F input, with GPU readback controls
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given) and draw four
                                       pixel fixtures (level-zero 2D A/B, explicit-mip 2D C and explicit-mip
                                       2D-array D) against two-mip (8x2 + 4x1) R16_UNORM (including 0xffaf and
                                       0xffef) and R32G32B32A32_FLOAT 2D / two-slice 2D-array textures into an
                                       R32G32B32A32_UINT target, checking all 64 output words per draw, high-half
                                       odd-dmask isolation and fourteen mutations

On RDNA 2 (gfx1013), image_load and image_load_mip with d16 narrow the enabled dmask channels to IEEE binary16 (round to
nearest even)
and packs them pairwise into ceil(popcount(dmask) / 2) consecutive VGPRs (low half first, high half second; when
popcount(dmask) is odd, bits 31:16 of the final destination VGPR are hardware-undefined). HLSL f32tof16 truncates
toward zero on WARP (lowering 31,740 of the 65,536 R16_UNORM values by 1 ULP), while narrowing a binary32 intermediate
still double-rounds R16_UNORM at 0xffaf (0x3bfe vs 0x3bfd) and 0xffef (0x3c00 vs 0x3bff). gfx2hlsl.py therefore emits
both integer binary32-to-binary16 RNE (pack16_rne) and exact UNORM16 ratio-to-half (pack16_unorm16) selected per SRV
via 'udsrc <ud> d16unorm16 <index>', restricts 'd16srv <index> exact' to the 14 independently GPU-verified exact
formats, and refuses upper-half uses of odd-dmask tails. Plain VGPR copies retain the restriction until only defined bits are read.
"""
from fractions import Fraction
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, assemble, compile_hlsl, root_blob  # noqa: E402

IMG_LOAD = 0x00
IMG_LOAD_MIP = 0x01
IMG_SAMPLE = 0x20
DIM_1D = 0
DIM_2D = 1
DIM_3D = 2
DIM_CUBE = 3
DIM_2D_ARRAY = 5

WIDTH = 8
HEIGHT = 2
PACKED_INPUT = (0x8000, 0x8000)
UINT_RTV = (42,)  # DXGI_FORMAT_R32G32B32A32_UINT

# Includes both R16_UNORM double-rounding discriminators (0xffaf -> 0x3bfd, 0xffef -> 0x3bff).
DEPTH_TEXELS = (
    0x0000, 0x0001, 0x0003, 0x1234, 0x2aad, 0x4008, 0x5555, 0x8000,
    0x8010, 0xa018, 0xc018, 0xe00c, 0xffaf, 0xffef, 0xfffe, 0xffff,
)

DEPTH_MIP1 = (
    (0xffaf, 0xffef, 0x0003, 0x8010),
    (0xffef, 0xffaf, 0x5555, 0xa018),
)

FLOAT_TEXELS = (
    (0x3f801000, 0x3f803000, 0x3f801001, 0x3f7fffff),
    (0x387fe000, 0x387fdfff, 0x33800000, 0x33000000),
    (0x33000001, 0xb5c00000, 0x80000000, 0x00000000),
    (0xbf803000, 0xbf801000, 0x477fefff, 0x477ff000),
    (0xc77ff000, 0x7f800000, 0xff800000, 0x7fc00001),
    (0x40490fdb, 0x35555555, 0xbeaaaaab, 0xffc00123),
    (0x3c000000, 0xbc800000, 0x3d000000, 0xbd800000),
    (0x42000000, 0xc2200000, 0x42400000, 0xc2600000),
    (0x3e200000, 0x3ea00000, 0x3ef00000, 0x3f100000),
    (0xbe200000, 0xbea00000, 0xbef00000, 0xbf100000),
    (0x39001000, 0x39003000, 0xb9003000, 0x39001001),
    (0x44002000, 0x44006000, 0xc4006000, 0x44002001),
    (0x34000000, 0x34800000, 0x35000000, 0x35800000),
    (0xb4000000, 0xb4800000, 0xb5000000, 0xb5800000),
    (0x3fc00000, 0xc0000000, 0x40200000, 0xc0400000),
    (0x477fe000, 0xc77fe000, 0x38800000, 0xb8800000),
)

FLOAT_MIP1 = (
    (0x3f803000, 0x387fe000, 0xbf803000, 0x477fefff),
    (0x33800000, 0x33000001, 0x80000000, 0x7fc00001),
    (0x44006000, 0xc4006000, 0x39003000, 0xb9003000),
    (0x477ff000, 0xc77ff000, 0x3e200000, 0xbe200000),
)

# DXGI_FORMAT enum values referenced in src/startup_stubs/d16_image_format.h.
DXGI_FORMAT_VALUES = {
    'DXGI_FORMAT_R32G32B32A32_FLOAT': 2,
    'DXGI_FORMAT_R32G32B32_FLOAT': 6,
    'DXGI_FORMAT_R16G16B16A16_UNORM': 11,
    'DXGI_FORMAT_R32G32_FLOAT': 16,
    'DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS': 21,
    'DXGI_FORMAT_R8G8B8A8_UNORM': 28,
    'DXGI_FORMAT_R16G16_UNORM': 35,
    'DXGI_FORMAT_R32_FLOAT': 41,
    'DXGI_FORMAT_R8G8_UNORM': 49,
    'DXGI_FORMAT_R16_UNORM': 56,
    'DXGI_FORMAT_R8_UNORM': 61,
    'DXGI_FORMAT_A8_UNORM': 65,
    'DXGI_FORMAT_B8G8R8A8_UNORM': 87,
    'DXGI_FORMAT_B8G8R8X8_UNORM': 88,
}


def encode_mimg(op, *, dmask=0xf, dim=DIM_2D, vaddr=0, vdata=4, srsrc=16, ssamp=0, nsa=0, extra_addrs=(),
                dlc=0, unorm=0, glc=0, r128=0, tfe=0, lwe=0, slc=0, a16=0, d16=0, w0_reserved=0, w1_reserved=0):
    w0 = (0xf0000000 | ((op & 0x7f) << 18) | ((op >> 7) & 1) | ((nsa & 0x3) << 1)
          | ((dim & 0x7) << 3) | ((dlc & 1) << 7) | ((dmask & 0xf) << 8)
          | ((unorm & 1) << 12) | ((glc & 1) << 13) | ((r128 & 1) << 15)
          | ((tfe & 1) << 16) | ((lwe & 1) << 17) | ((slc & 1) << 25) | w0_reserved)
    w1 = ((vaddr & 0xff) | ((vdata & 0xff) << 8) | (((srsrc // 4) & 0x1f) << 16)
          | (((ssamp // 4) & 0x1f) << 21) | ((a16 & 1) << 30) | ((d16 & 1) << 31) | w1_reserved)
    ext = bytes(extra_addrs) + bytes(max(0, 4 * nsa - len(extra_addrs)))
    return struct.pack('<II', w0, w1) + ext


def encode_v_and_low16(reg):
    """v_and_b32_e32 v<reg>, 0xffff, v<reg>."""
    return struct.pack('<II', (0x1b << 25) | (reg << 17) | (reg << 9) | 255, 0x0000ffff)


# Entry v0 holds x | (y << 16). v1 and v3 hold out-of-bounds sentinels (63) so dropping a16 on v0
# (reading v[0:1]) or ignoring NSA on [v2, v9] (reading v[2:3]) misses every pixel of the 8x2 texture.
PRE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
v_mov_b32_e32 v1, 63
v_and_b32_e32 v2, 7, v0
v_mov_b32_e32 v3, 63
v_and_b32_e32 v8, 7, v0
v_lshrrev_b32_e32 v9, 16, v0
s_waitcnt lgkmcnt(0)
"""
PRE_WORDS = bytes.fromhex(
    '04040cf4000000fa'
    'bf02027e'
    '87000436'
    'bf02067e'
    '87001036'
    '9000122c'
    '7fc08cbf'
)

POST = """\
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
POST_WORDS = bytes.fromhex('703f8cbf' '0f1800f804050607' '000081bf')

# Fixture A consumes odd dmask:0x1 (v4) via v_cvt_f32_f16 and odd dmask:0x7 tail (v7) via SDWA WORD_0.
POST_A = """\
s_waitcnt vmcnt(0)
v_cvt_f32_f16_e32 v4, v4
v_and_b32_sdwa v7, v7, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
POST_WORDS_A = bytes.fromhex('703f8cbf' '0417087e' 'f90e0e3607060404' '0f1800f804050607' '000081bf')

# Fixture B consumes odd dmask:0x2 (v6) via v_and_b32 with constant 0xffff mask.
POST_B = """\
s_waitcnt vmcnt(0)
v_and_b32_e32 v6, 0xffff, v6
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
POST_WORDS_B = bytes.fromhex('703f8cbf' 'ff0c0c36ffff0000' '0f1800f804050607' '000081bf')

MIMG_LLVM_MC = {
    'd16_x': (
        'image_load v4, v[8:9], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D d16',
        encode_mimg(IMG_LOAD, dmask=0x1, vaddr=8, vdata=4, d16=1),
    ),
    'd16_xz_a16': (
        'image_load v5, v0, s[16:23] dmask:0x5 dim:SQ_RSRC_IMG_2D a16 d16',
        encode_mimg(IMG_LOAD, dmask=0x5, vaddr=0, vdata=5, a16=1, d16=1),
    ),
    'd16_xyz_nsa': (
        'image_load v[6:7], [v2, v9], s[16:23] dmask:0x7 dim:SQ_RSRC_IMG_2D glc slc dlc d16',
        encode_mimg(IMG_LOAD, dmask=0x7, vaddr=2, vdata=6, nsa=1, extra_addrs=(9,), glc=1, slc=1, dlc=1, d16=1),
    ),
    'd16_xyzw': (
        'image_load v[4:5], v[8:9], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D unorm d16',
        encode_mimg(IMG_LOAD, dmask=0xf, vaddr=8, vdata=4, unorm=1, d16=1),
    ),
    'd16_y': (
        'image_load v6, v[8:9], s[16:23] dmask:0x2 dim:SQ_RSRC_IMG_2D d16',
        encode_mimg(IMG_LOAD, dmask=0x2, vaddr=8, vdata=6, d16=1),
    ),
    'd16_zw': (
        'image_load v7, v[8:9], s[16:23] dmask:0xc dim:SQ_RSRC_IMG_2D d16',
        encode_mimg(IMG_LOAD, dmask=0xc, vaddr=8, vdata=7, d16=1),
    ),
    'd16_3d': (
        'image_load v[4:5], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_3D d16',
        encode_mimg(IMG_LOAD, dmask=0xf, dim=DIM_3D, vaddr=0, vdata=4, d16=1),
    ),
    'd16_2da': (
        'image_load v[4:5], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY d16',
        encode_mimg(IMG_LOAD, dmask=0xf, dim=DIM_2D_ARRAY, vaddr=0, vdata=4, d16=1),
    ),
    'mip_d16_2d': (
        'image_load_mip v[4:5], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D d16',
        encode_mimg(IMG_LOAD_MIP, dmask=0xf, vaddr=0, vdata=4, d16=1),
    ),
    'mip_d16_x': (
        'image_load_mip v4, v[8:10], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D d16',
        encode_mimg(IMG_LOAD_MIP, dmask=0x1, vaddr=8, vdata=4, d16=1),
    ),
    'mip_d16_xz': (
        'image_load_mip v5, v[8:10], s[16:23] dmask:0x5 dim:SQ_RSRC_IMG_2D unorm d16',
        encode_mimg(IMG_LOAD_MIP, dmask=0x5, vaddr=8, vdata=5, unorm=1, d16=1),
    ),
    'mip_d16_xyz_nsa': (
        'image_load_mip v[6:7], [v2, v9, v10], s[16:23] dmask:0x7 dim:SQ_RSRC_IMG_2D glc slc dlc d16',
        encode_mimg(IMG_LOAD_MIP, dmask=0x7, vaddr=2, vdata=6, nsa=1, extra_addrs=(9, 10), glc=1, slc=1, dlc=1, d16=1),
    ),
    'mip_d16_3d': (
        'image_load_mip v[4:5], v[0:3], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_3D d16',
        encode_mimg(IMG_LOAD_MIP, dmask=0xf, dim=DIM_3D, vaddr=0, vdata=4, d16=1),
    ),
    'mip_d16_2da_xyzw': (
        'image_load_mip v[4:5], v[8:11], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY d16',
        encode_mimg(IMG_LOAD_MIP, dmask=0xf, dim=DIM_2D_ARRAY, vaddr=8, vdata=4, d16=1),
    ),
    'mip_d16_2da_x_nsa': (
        'image_load_mip v6, [v12, v9, v10, v11], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY d16',
        encode_mimg(IMG_LOAD_MIP, dmask=0x1, dim=DIM_2D_ARRAY, vaddr=12, vdata=6, nsa=1, extra_addrs=(9, 10, 11), d16=1),
    ),
    'mip_d16_2da_zw_nsa': (
        'image_load_mip v7, [v12, v9, v10, v11], s[16:23] dmask:0xc dim:SQ_RSRC_IMG_2D_ARRAY d16',
        encode_mimg(IMG_LOAD_MIP, dmask=0xc, dim=DIM_2D_ARRAY, vaddr=12, vdata=7, nsa=1, extra_addrs=(9, 10, 11), d16=1),
    ),
    '1d_d16_refused': (
        'image_load v4, v0, s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_1D d16',
        encode_mimg(IMG_LOAD, dmask=0x1, dim=DIM_1D, vaddr=0, vdata=4, d16=1),
    ),
    'mip_1d_d16_refused': (
        'image_load_mip v4, v[0:1], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_1D d16',
        encode_mimg(IMG_LOAD_MIP, dmask=0x1, dim=DIM_1D, vaddr=0, vdata=4, d16=1),
    ),
    'cube_d16_refused': (
        'image_load v[4:5], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_CUBE d16',
        encode_mimg(IMG_LOAD, dmask=0xf, dim=DIM_CUBE, vaddr=0, vdata=4, d16=1),
    ),
    'mip_cube_d16_refused': (
        'image_load_mip v[4:5], v[0:3], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_CUBE d16',
        encode_mimg(IMG_LOAD_MIP, dmask=0xf, dim=DIM_CUBE, vaddr=0, vdata=4, d16=1),
    ),
}
DECODE_KEYS = tuple(MIMG_LLVM_MC)

# Prefix for 2D image_load_mip (Fixture C): v10 = y (mip 0 or 1), v8 = v2 = x >> y, v9 = 0, v3 = 63 (NSA sentinel).
PRE_C = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
v_and_b32_e32 v1, 7, v0
v_lshrrev_b32_e32 v10, 16, v0
v_lshrrev_b32_e32 v8, v10, v1
v_mov_b32_e32 v9, 0
v_lshrrev_b32_e32 v2, v10, v1
v_mov_b32_e32 v3, 63
s_waitcnt lgkmcnt(0)
"""
PRE_WORDS_C = bytes.fromhex('04040cf4000000fa' '87000236' '9000142c' '0a03102c' '8002127e' '0a03042c' 'bf02067e' '7fc08cbf')

# Prefix for 2D-array image_load_mip (Fixture D): v8 = v12 = x >> 1, v9 = 0, v10 = x & 1 (slice), v11 = y (mip), v13 = 63.
PRE_D = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
v_and_b32_e32 v1, 7, v0
v_lshrrev_b32_e32 v11, 16, v0
v_lshrrev_b32_e32 v8, 1, v1
v_mov_b32_e32 v9, 0
v_and_b32_e32 v10, 1, v1
v_lshrrev_b32_e32 v12, 1, v1
v_mov_b32_e32 v13, 63
s_waitcnt lgkmcnt(0)
"""
PRE_WORDS_D = bytes.fromhex('04040cf4000000fa' '87000236' '9000162c' '8102102c' '8002127e' '81021436' '8102182c' 'bf021a7e' '7fc08cbf')

# Fixture A exercises 1-channel depth d16 (dmask:0x1 -> v_cvt_f32_f16), 2-channel non-contiguous a16+d16 (dmask:0x5)
# and 3-channel NSA d16 (dmask:0x7 -> SDWA WORD_0 low-half consumer on v7).
PS_SOURCE_A = PRE + '\n'.join(MIMG_LLVM_MC[k][0] for k in ('d16_x', 'd16_xz_a16', 'd16_xyz_nsa')) + '\n' + POST_A
PS_WORDS_A = PRE_WORDS + b''.join(MIMG_LLVM_MC[k][1] for k in ('d16_x', 'd16_xz_a16', 'd16_xyz_nsa')) + POST_WORDS_A

# Fixture B exercises full 4-channel d16 (dmask:0xf), single y channel (dmask:0x2 -> v_and_b32 0xffff) and dmask:0xc.
PS_SOURCE_B = PRE + '\n'.join(MIMG_LLVM_MC[k][0] for k in ('d16_xyzw', 'd16_y', 'd16_zw')) + '\n' + POST_B
PS_WORDS_B = PRE_WORDS + b''.join(MIMG_LLVM_MC[k][1] for k in ('d16_xyzw', 'd16_y', 'd16_zw')) + POST_WORDS_B

# Fixture C exercises 2D image_load_mip with d16 across mip 0 (y=0) and mip 1 (y=1): odd dmask:0x1 -> v_cvt_f32_f16,
# 2-channel non-contiguous dmask:0x5 with unorm, and 3-channel NSA [v2, v9, v10] dmask:0x7 -> SDWA WORD_0 on v7.
PS_SOURCE_C = PRE_C + '\n'.join(MIMG_LLVM_MC[k][0] for k in ('mip_d16_x', 'mip_d16_xz', 'mip_d16_xyz_nsa')) + '\n' + POST_A
PS_WORDS_C = PRE_WORDS_C + b''.join(MIMG_LLVM_MC[k][1] for k in ('mip_d16_x', 'mip_d16_xz', 'mip_d16_xyz_nsa')) + POST_WORDS_A

# Fixture D exercises 2D-array image_load_mip with d16 across slices (x & 1) and mips (y): full dmask:0xf on v[8:11],
# NSA [v12, v9, v10, v11] odd dmask:0x1 -> v_and_b32 0xffff on v6, and NSA dmask:0xc on v7.
PS_SOURCE_D = PRE_D + '\n'.join(MIMG_LLVM_MC[k][0] for k in ('mip_d16_2da_xyzw', 'mip_d16_2da_x_nsa', 'mip_d16_2da_zw_nsa')) + '\n' + POST_B
PS_WORDS_D = PRE_WORDS_D + b''.join(MIMG_LLVM_MC[k][1] for k in ('mip_d16_2da_xyzw', 'mip_d16_2da_x_nsa', 'mip_d16_2da_zw_nsa')) + POST_WORDS_B


PACK16_RNE_HLSL = """\
uint pack16_rne(float value) {
    uint bits = asuint(value), sign = (bits >> 16) & 0x8000u, abs_bits = bits & 0x7fffffffu;
    if (abs_bits > 0x7f800000u) return sign | 0x7e00u;
    if (abs_bits >= 0x477ff000u) return sign | 0x7c00u;
    if (abs_bits >= 0x38800000u) {
        uint shifted = abs_bits - 0x38000000u;
        return sign | ((shifted + 0x00000fffu + ((shifted >> 13) & 1u)) >> 13);
    }
    if (abs_bits < 0x33000000u) return sign;
    uint sig = 0x00800000u | (abs_bits & 0x007fffffu), shift = 126u - (abs_bits >> 23);
    uint half_m = sig >> shift, rem = sig & ((1u << shift) - 1u), mid = 1u << (shift - 1u);
    if (rem > mid || (rem == mid && (half_m & 1u) != 0u)) half_m += 1u;
    return sign | half_m;
}"""

PACK16_UNORM16_HLSL = """\
uint pack16_unorm16(float value) {
    uint u = uint(saturate(value) * 65535.0 + 0.5);
    if (u <= 3u) return u << 8;
    if (u >= 65535u) return 0x3c00u;
    uint msb = firstbithigh(u);
    uint num = u << (26u - msb);
    uint q = num / 65535u;
    if (num - q * 65535u > 32767u) q += 1u;
    return ((msb - 1u) << 10) + (q - 1024u);
}"""

PACK16_D16_HLSL = """\
uint pack16_d16(float value, uint unorm16) {
    return unorm16 != 0u ? pack16_unorm16(value) : pack16_rne(value);
}"""

PACK16_HELPERS_HLSL = PACK16_RNE_HLSL + '\n' + PACK16_UNORM16_HLSL + '\n' + PACK16_D16_HLSL

PACK16_RTZ_HLSL = """\
uint pack16_rne(float value) {
    uint bits = asuint(value), sign = (bits >> 16) & 0x8000u, abs_bits = bits & 0x7fffffffu;
    if (abs_bits > 0x7f800000u) return sign | 0x7e00u;
    if (abs_bits >= 0x47800000u) return sign | 0x7c00u;
    if (abs_bits >= 0x477fe000u) return sign | 0x7bffu;
    if (abs_bits >= 0x38800000u) return sign | ((abs_bits - 0x38000000u) >> 13);
    if (abs_bits < 0x33800000u) return sign;
    uint sig = 0x00800000u | (abs_bits & 0x007fffffu), shift = 126u - (abs_bits >> 23);
    return sign | (sig >> shift);
}"""


def f32_bits_from_unorm16(u16):
    return struct.unpack('<I', struct.pack('<f', (u16 & 0xffff) / 65535.0))[0]


def f32_bits_from_f16_bits(h16):
    val = struct.unpack('<e', struct.pack('<H', h16 & 0xffff))[0]
    return struct.unpack('<I', struct.pack('<f', val))[0]


def exact_unorm16_f16_ref(u16):
    """Exact rational u16 / 65535 rounded directly to IEEE binary16 (round to nearest even)."""
    u16 &= 0xffff
    if u16 == 0:
        return 0
    x = Fraction(u16, 65535)
    if x < Fraction(1, 1 << 14):
        scaled = x * (1 << 24)
        q, r = divmod(scaled.numerator, scaled.denominator)
        if 2 * r > scaled.denominator or (2 * r == scaled.denominator and (q & 1)):
            q += 1
        return q
    e = x.numerator.bit_length() - x.denominator.bit_length() - 1
    while (x * (1 << (-e))) < 1:
        e -= 1
    while (x * (1 << (-e))) >= 2:
        e += 1
    scaled = x * (1 << (10 - e))
    q, r = divmod(scaled.numerator, scaled.denominator)
    if 2 * r > scaled.denominator or (2 * r == scaled.denominator and (q & 1)):
        q += 1
    if q == 2048:
        q, e = 1024, e + 1
    return ((e + 15) << 10) | (q - 1024)


def pack16_rne_ref(bits):
    """Independent IEEE-754 binary32-to-binary16 round-to-nearest-even reference (preserving NaN sign)."""
    bits &= 0xffffffff
    sign = (bits >> 16) & 0x8000
    abs_bits = bits & 0x7fffffff
    if abs_bits > 0x7f800000:
        return sign | 0x7e00
    if abs_bits >= 0x477ff000:
        return sign | 0x7c00
    val = struct.unpack('<f', struct.pack('<I', bits))[0]
    try:
        return struct.unpack('<H', struct.pack('<e', val))[0]
    except OverflowError:
        return sign | 0x7c00


def pack16_rtz_ref(bits):
    """HLSL f32tof16 (DXBC ftof16) round-toward-zero reference for the RTZ mutation control."""
    bits &= 0xffffffff
    sign = (bits >> 16) & 0x8000
    abs_bits = bits & 0x7fffffff
    if abs_bits > 0x7f800000:
        return sign | 0x7e00
    if abs_bits == 0x7f800000:
        return sign | 0x7c00
    if abs_bits >= 0x47800000:
        return sign | 0x7c00
    if abs_bits >= 0x477fe000:
        return sign | 0x7bff
    if abs_bits >= 0x38800000:
        return sign | ((abs_bits - 0x38000000) >> 13)
    if abs_bits < 0x33800000:
        return sign
    sig = 0x00800000 | (abs_bits & 0x007fffff)
    shift = 126 - (abs_bits >> 23)
    return sign | (sig >> shift)


def texel_f32_words(tex_mode, x, y, *, mip=0, slice_=0):
    is_depth = tex_mode.startswith('r16_unorm')
    oob = (0, 0, 0, 0x3f800000) if is_depth else (0, 0, 0, 0)
    max_slices = 2 if tex_mode.endswith('_array') else 1
    if not (0 <= mip < 2 and 0 <= slice_ < max_slices):
        return oob
    mw, mh = WIDTH >> mip, HEIGHT >> mip
    if not (0 <= x < mw and 0 <= y < mh):
        return oob
    if mip == 0:
        idx = (y * WIDTH + x) if slice_ == 0 else (WIDTH * HEIGHT - 1 - (y * WIDTH + x))
        if is_depth:
            return (f32_bits_from_unorm16(DEPTH_TEXELS[idx]), 0, 0, 0x3f800000)
        return FLOAT_TEXELS[idx]
    if is_depth:
        return (f32_bits_from_unorm16(DEPTH_MIP1[slice_][x]), 0, 0, 0x3f800000)
    idx = x if slice_ == 0 else ((WIDTH // 2 - 1) - x)
    return FLOAT_MIP1[idx]


def texel_halves(tex_mode, x, y, mutant=None, *, mip=0, slice_=0):
    is_depth = tex_mode.startswith('r16_unorm')
    oob = (0, 0, 0, 0x3c00) if is_depth else (0, 0, 0, 0)
    max_slices = 2 if tex_mode.endswith('_array') else 1
    if not (0 <= mip < 2 and 0 <= slice_ < max_slices):
        return oob
    mw, mh = WIDTH >> mip, HEIGHT >> mip
    if not (0 <= x < mw and 0 <= y < mh):
        return oob
    rgba = texel_f32_words(tex_mode, x, y, mip=mip, slice_=slice_)
    if is_depth:
        if mutant in ('rtz', 'mip_rtz'):
            return tuple(pack16_rtz_ref(c) for c in rgba)
        if mutant in ('double_round', 'mip_double_round'):
            return tuple(pack16_rne_ref(c) for c in rgba)
        u16 = (
            DEPTH_TEXELS[(y * WIDTH + x) if slice_ == 0 else (WIDTH * HEIGHT - 1 - (y * WIDTH + x))]
            if mip == 0
            else DEPTH_MIP1[slice_][x]
        )
        return (exact_unorm16_f16_ref(u16), 0, 0, 0x3c00)
    narrow = pack16_rtz_ref if mutant in ('rtz', 'mip_rtz') else pack16_rne_ref
    return tuple(narrow(c) for c in rgba)


def pack_pair(lo, hi):
    return (lo & 0xffff) | ((hi & 0xffff) << 16)


def expected_words(fixture, tex_mode, mutant=None):
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if fixture == 'A':
                rgba = texel_f32_words(tex_mode, x, y)
                hx, hy, hz, hw = texel_halves(tex_mode, x, y, mutant)
                w0 = rgba[0] if mutant == 'unpacked_x' else f32_bits_from_f16_bits(hx)
                ax, ay = (x | (y << 16), 63) if mutant == 'drop_a16' else (x, y)
                hax, _, haz, _ = texel_halves(tex_mode, ax, ay, mutant)
                if mutant == 'swap_halves':
                    w1 = pack_pair(haz, hax)
                elif mutant == 'drop_shift':
                    w1 = (hax & 0xffff) | (haz & 0xffff)
                else:
                    w1 = pack_pair(hax, haz)
                nx, ny = (x, 63) if mutant == 'drop_nsa' else (x, y)
                hnx, hny, hnz, _ = texel_halves(tex_mode, nx, ny, mutant)
                w2 = pack_pair(hnx, hny)
                w3 = (0xdead0000 | hnz) if mutant == 'unmasked_odd_hi' else hnz
                out.extend((w0, w1, w2, w3))
            elif fixture == 'B':
                hx, hy, hz, hw = texel_halves(tex_mode, x, y, mutant)
                out.extend((pack_pair(hx, hy), pack_pair(hz, hw), hy, pack_pair(hz, hw)))
            elif fixture == 'C':
                mip = 0 if mutant == 'mip_zero' else y
                tx = x >> y
                hx, _, hz, _ = texel_halves(tex_mode, tx, 0, mutant, mip=mip)
                w0 = f32_bits_from_f16_bits(hx)
                w1 = pack_pair(hx, hz)
                nty = 63 if mutant == 'mip_drop_nsa' else 0
                hnx, hny, hnz, _ = texel_halves(tex_mode, tx, nty, mutant, mip=mip)
                w2 = pack_pair(hnx, hny)
                w3 = (0xdead0000 | hnz) if mutant == 'mip_unmasked_odd_hi' else hnz
                out.extend((w0, w1, w2, w3))
            elif fixture == 'D':
                tx = x >> 1
                s, m = (y, x & 1) if mutant == 'swap_slice_mip' else (x & 1, y)
                hx, hy, hz, hw = texel_halves(tex_mode, tx, 0, mutant, mip=m, slice_=s)
                nty = 63 if mutant == 'array_drop_nsa' else 0
                hnx, _, hnz, hnw = texel_halves(tex_mode, tx, nty, mutant, mip=m, slice_=s)
                out.extend((pack_pair(hx, hy), pack_pair(hz, hw), hnx, pack_pair(hnz, hnw)))
            else:
                raise ValueError(fixture)
    return out


def expect_refused(fn, substring, label):
    try:
        fn()
    except G.Unsupported as exc:
        assert substring in str(exc), f'{label}: expected {substring!r} in {str(exc)!r}'
    else:
        raise AssertionError(f'{label}: unexpectedly accepted')


def transpile_emitted_helpers(hlsl_source):
    """Compile the live pack16_rne and pack16_unorm16 HLSL bodies emitted by gfx2hlsl.py into Python."""
    namespace = {
        'uint': lambda x: int(x) & 0xffffffff,
        'asuint': lambda bits: bits & 0xffffffff,
        'saturate': lambda bits: min(max(struct.unpack('<f', struct.pack('<I', bits & 0xffffffff))[0], 0.0), 1.0),
        'firstbithigh': lambda u: (u & 0xffffffff).bit_length() - 1,
    }
    for fn_name in ('pack16_rne', 'pack16_unorm16'):
        match = re.search(rf'uint {fn_name}\(float value\) \{{(.*?)\n\}}', hlsl_source, re.S)
        assert match, f'missing {fn_name} in emitted HLSL'
        py_lines = [f'def {fn_name}(value):']
        indent = 1
        for raw_line in match.group(1).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line == '}':
                indent -= 1
                continue
            line = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'\1', line)
            line = line.replace('||', ' or ').replace('&&', ' and ')
            if line.startswith('uint '):
                body = line[5:].rstrip(';').replace(' / ', ' // ')
                # Split comma-separated uint declarations at top level.
                for decl in G.split_top(body, ','):
                    py_lines.append('    ' * indent + decl)
                continue
            if line.startswith('if ') and line.endswith('{'):
                cond = line[3:-1].strip()
                assert cond.startswith('(') and cond.endswith(')'), cond
                py_lines.append('    ' * indent + f'if {cond[1:-1]}:')
                indent += 1
                continue
            if line.startswith('if '):
                m_if = re.fullmatch(r'if \((.*)\) (.*);', line)
                assert m_if, line
                stmt = m_if.group(2).replace(' / ', ' // ')
                py_lines.append('    ' * indent + f'if {m_if.group(1)}: {stmt}')
                continue
            stmt = line.rstrip(';').replace(' / ', ' // ')
            py_lines.append('    ' * indent + stmt)
        exec('\n'.join(py_lines), namespace)
    return namespace['pack16_rne'], namespace['pack16_unorm16']


def check_rounding_formula():
    """Verify the live HLSL helpers emitted by gfx2hlsl.py on Linux and Windows without needing a GPU."""
    emitted_ps = G.Pipeline(VS_WORDS, PS_WORDS_A, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV).hlsl_ps()
    assert emitted_ps.count(PACK16_HELPERS_HLSL) == 1, emitted_ps
    hlsl_pack16_rne, hlsl_pack16_unorm16 = transpile_emitted_helpers(emitted_ps)

    anchors = (
        0x00000000, 0x80000000, 0x00000001, 0x007fffff, 0x00800000,
        0x32ffffff, 0x33000000, 0x33000001, 0x337fffff, 0x33800000, 0x33800001,
        0x387fdfff, 0x387fe000, 0x387fe001, 0x387fffff, 0x38800000, 0x38801000,
        0x3f7fffff, 0x3f800000, 0x3f801000, 0x3f801001, 0x3f803000,
        0x477fdfff, 0x477fe000, 0x477fefff, 0x477ff000, 0x47800000,
        0x7f800000, 0x7f800001, 0x7fc00000, 0x7fc00123, 0x7fffffff,
    )
    tested = 0
    for base in anchors:
        for delta in (-2, -1, 0, 1, 2):
            word = (base + delta) & 0xffffffff
            for signed_word in (word, word ^ 0x80000000):
                got = hlsl_pack16_rne(signed_word)
                want = pack16_rne_ref(signed_word)
                assert got == want, f'0x{signed_word:08x}: got 0x{got:04x}, want 0x{want:04x}'
                tested += 1

    # Verify all 65,536 R16_UNORM codes through the transpiled pack16_unorm16 helper, and confirm
    # that double-rounding through binary32 differs exclusively at 0xffaf and 0xffef.
    double_round_codes = []
    for u16 in range(65536):
        w = f32_bits_from_unorm16(u16)
        got_exact = hlsl_pack16_unorm16(w)
        want_exact = exact_unorm16_f16_ref(u16)
        assert got_exact == want_exact, f'u16=0x{u16:04x}: got 0x{got_exact:04x}, want 0x{want_exact:04x}'
        via_f32 = hlsl_pack16_rne(w)
        assert via_f32 == pack16_rne_ref(w), f'u16=0x{u16:04x}'
        if via_f32 != want_exact:
            double_round_codes.append((u16, want_exact, via_f32))
    assert double_round_codes == [(0xffaf, 0x3bfd, 0x3bfe), (0xffef, 0x3bff, 0x3c00)], double_round_codes

    # Verify all 256 UNORM8 codes have zero double-rounding under pack16_rne.
    for u8 in range(256):
        f32_u8 = struct.unpack('<I', struct.pack('<f', u8 / 255.0))[0]
        want_u8 = struct.unpack('<H', struct.pack('<e', u8 / 255.0))[0]
        assert hlsl_pack16_rne(f32_u8) == want_u8, u8

    # Self-test: mutating the tie-to-even term in pack16_rne or the threshold in pack16_unorm16 must fail here.
    broken_tie, _ = transpile_emitted_helpers(emitted_ps.replace(' + ((shifted >> 13) & 1u)', '', 1))
    assert broken_tie(0x3f803000) != pack16_rne_ref(0x3f803000)
    _, broken_u16 = transpile_emitted_helpers(emitted_ps.replace('> 32767u', '>= 0u', 1))
    assert broken_u16(f32_bits_from_unorm16(0xffaf)) != 0x3bfd

    rtz_depth_diffs = sum(
        exact_unorm16_f16_ref(u) != pack16_rtz_ref(f32_bits_from_unorm16(u))
        for u in DEPTH_TEXELS
    )
    assert rtz_depth_diffs >= 6, f'expected >=6 RNE-vs-RTZ depth discriminators, got {rtz_depth_diffs}'
    print(f'emitted HLSL helpers: {tested} boundary floats, 256 UNORM8 and all 65,536 R16_UNORM codes exact')


def check_decoder_and_masks():
    for blob, expected_text in ((PRE_WORDS, PRE), (PRE_WORDS_C, PRE_C), (PRE_WORDS_D, PRE_D),
                                (POST_WORDS_A, POST_A), (POST_WORDS_B, POST_B)):
        off = 0
        decoded = []
        while off < len(blob):
            n, _, asm, _ = D.decode_instruction(blob, off)
            decoded.append(asm)
            off += n
        assert decoded == expected_text.strip().splitlines(), decoded

    for key in DECODE_KEYS:
        text, raw = MIMG_LLVM_MC[key]
        length, _, asm, _ = D.decode_instruction(raw, 0)
        assert length == len(raw) and asm == text, (key, length, asm, text)

    # Check all 15 non-zero dmasks on both image_load and image_load_mip for destination width,
    # pairwise component order, explicit mip Load coordinates and odd-dmask gating.
    for op_name, op_code, pre_words, want_load in (
        ('image_load', IMG_LOAD, PRE_WORDS, 'r0.Load(int3(v8, v9, 0))'),
        ('image_load_mip', IMG_LOAD_MIP, PRE_WORDS_C, 'r0.Load(int3(v8, v9, v10))'),
    ):
        for dmask in range(1, 16):
            comps = [c for bit, c in enumerate('xyzw') if dmask & (1 << bit)]
            expected_regs = (len(comps) + 1) // 2
            raw = encode_mimg(op_code, dmask=dmask, vaddr=8, vdata=4, d16=1)
            _, _, asm, _ = D.decode_instruction(raw, 0)
            expected_dst = 'v4' if expected_regs == 1 else 'v[4:5]'
            assert asm.startswith(f'{op_name} {expected_dst}, '), (op_name, dmask, asm)
            if len(comps) % 2 == 1:
                odd_reg = 4 + (len(comps) - 1) // 2
                expect_refused(
                    lambda r=raw, pw=pre_words: G.Pipeline(
                        VS_WORDS, pw + r + POST_WORDS, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV
                    ),
                    f'whole-word read of v{odd_reg} after odd d16 image_load',
                    f'{op_name} odd dmask 0x{dmask:x} direct export',
                )
                ps_blob = pre_words + raw + encode_v_and_low16(odd_reg) + POST_WORDS
            else:
                ps_blob = pre_words + raw + POST_WORDS
            pipe = G.Pipeline(VS_WORDS, ps_blob, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV)
            ps = pipe.hlsl_ps()
            manifest = pipe.manifest()
            assert PACK16_HELPERS_HLSL in ps
            assert want_load in ps, (op_name, dmask, ps)
            assert 'udsrc 4 d16unorm16 0' in manifest.splitlines(), manifest
            assert 'd16srv 0 exact' in manifest.splitlines(), manifest
            assert 'SamplerState' not in ps and 'smp ' not in manifest
            for i in range(0, len(comps), 2):
                reg = 4 + i // 2
                if i + 1 == len(comps):
                    want = f'v{reg} = pack16_d16(t.{comps[i]}, ud4);'
                else:
                    want = f'v{reg} = pack16_d16(t.{comps[i]}, ud4) | (pack16_d16(t.{comps[i + 1]}, ud4) << 16);'
                assert want in ps, (op_name, dmask, want, ps)

    # Verify 3D and 2D-array level-zero and explicit-mip D16 loads, plus vertex-stage emission.
    for key, hlsl_type, srv_kind, want_call, pre_words in (
        ('d16_3d', 'Texture3D<float4>', 'tex3d', 'r0.Load(int4(v0, v1, v2, 0))', PRE_WORDS),
        ('d16_2da', 'Texture2DArray<float4>', 'tex2da', 'r0.Load(int4(v0, v1, v2, 0))', PRE_WORDS),
        ('mip_d16_3d', 'Texture3D<float4>', 'tex3d', 'r0.Load(int4(v0, v1, v2, v3))', PRE_WORDS),
        ('mip_d16_2da_xyzw', 'Texture2DArray<float4>', 'tex2da', 'r0.Load(int4(v8, v9, v10, v11))', PRE_WORDS_D),
    ):
        ps_3 = pre_words + MIMG_LLVM_MC[key][1] + POST_WORDS
        pipe = G.Pipeline(VS_WORDS, ps_3, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV)
        hlsl = pipe.hlsl_ps()
        manifest = pipe.manifest().splitlines()
        assert hlsl_type in hlsl and want_call in hlsl, (key, hlsl)
        assert f'srv 0 {srv_kind} table 0 0' in manifest and 'd16srv 0 exact' in manifest

    # Vertex stage using image_load / image_load_mip ... d16 emits helpers and d16srv, while a pipeline without d16 omits both.
    ps_plain = bytes.fromhex('8002007e' '0f1800f800000000' '000081bf')
    for op_code, want_call in ((IMG_LOAD, 'r0.Load(int3(v0, v1, 0))'), (IMG_LOAD_MIP, 'r0.Load(int3(v0, v1, v2))')):
        vs_d16 = (
            bytes.fromhex('04040cf4000000fa' '8002027e' '8002047e' '7fc08cbf')
            + encode_mimg(op_code, dmask=0x3, vaddr=0, vdata=4, d16=1)
            + bytes.fromhex('703f8cbf' 'cf0800f804040404' '000081bf')
        )
        vs_pipe = G.Pipeline(vs_d16, ps_plain, root_blob())
        assert PACK16_HELPERS_HLSL in vs_pipe.hlsl_vs()
        assert want_call in vs_pipe.hlsl_vs(), vs_pipe.hlsl_vs()
        assert 'v4 = pack16_d16(t.x, ud4) | (pack16_d16(t.y, ud4) << 16);' in vs_pipe.hlsl_vs()
        assert 'd16srv 0 exact' in vs_pipe.manifest().splitlines()
    plain_pipe = G.Pipeline(VS_WORDS, ps_plain, root_blob())
    assert 'pack16_' not in plain_pipe.hlsl_vs() and 'pack16_' not in plain_pipe.hlsl_ps()
    assert 'd16srv' not in plain_pipe.manifest() and 'd16unorm16' not in plain_pipe.manifest()
    print('decoder & masks: all 15 non-zero dmasks (load & load_mip), 2D/a16/NSA/3D/2D_ARRAY, d16srv and VS/PS paths verified')


def check_srv_format_controls():
    header = (HERE.parent / 'src' / 'startup_stubs' / 'd16_image_format.h').read_text(encoding='utf-8')
    float_block, unorm16_block = header.split('return D16SrvMode::float_exact;', 1)
    float_names = re.findall(r'case\s+(DXGI_FORMAT_\w+)\s*:', float_block)
    unorm16_names = re.findall(r'case\s+(DXGI_FORMAT_\w+)\s*:', unorm16_block.split('return D16SrvMode::unorm16;', 1)[0])
    assert len(float_names) == 11 and len(unorm16_names) == 3, (float_names, unorm16_names)
    assert {DXGI_FORMAT_VALUES[n] for n in float_names} == G.D16_FLOAT_EXACT_SRV_FORMATS
    assert {DXGI_FORMAT_VALUES[n] for n in unorm16_names} == G.D16_UNORM16_SRV_FORMATS

    ps_even = PRE_WORDS + MIMG_LLVM_MC['d16_xyzw'][1] + POST_WORDS
    ps_mip_even = PRE_WORDS_C + MIMG_LLVM_MC['mip_d16_xz'][1] + POST_WORDS
    for fmt in sorted(G.D16_EXACT_SRV_FORMATS):
        for blob in (ps_even, ps_mip_even):
            pipe = G.Pipeline(VS_WORDS, blob, root_blob(), ps_hw_inputs=PACKED_INPUT,
                              rtv_formats=UINT_RTV, srv_formats=(fmt,))
            assert 'd16srv 0 exact' in pipe.manifest().splitlines()

    # Integer (UINT/SINT), typeless, depth-stencil-only, SNORM16, R24_UNORM, SRGB, BCn and console formats are refused.
    refused_srv_formats = (
        (0, 'UNKNOWN'),
        (3, 'R32G32B32A32_UINT'),
        (4, 'R32G32B32A32_SINT'),
        (13, 'R16G16B16A16_SNORM'),
        (20, 'D32_FLOAT_S8X24_UINT'),
        (22, 'X32_TYPELESS_G8X24_UINT'),
        (24, 'R10G10B10A2_UNORM'),
        (27, 'R8G8B8A8_TYPELESS'),
        (29, 'R8G8B8A8_UNORM_SRGB'),
        (30, 'R8G8B8A8_UINT'),
        (31, 'R8G8B8A8_SNORM'),
        (32, 'R8G8B8A8_SINT'),
        (37, 'R16G16_SNORM'),
        (39, 'R32_TYPELESS'),
        (40, 'D32_FLOAT'),
        (42, 'R32_UINT'),
        (43, 'R32_SINT'),
        (45, 'D24_UNORM_S8_UINT'),
        (46, 'R24_UNORM_X8_TYPELESS'),
        (47, 'X24_TYPELESS_G8_UINT'),
        (53, 'R16_TYPELESS'),
        (55, 'D16_UNORM'),
        (57, 'R16_UINT'),
        (58, 'R16_SNORM'),
        (59, 'R16_SINT'),
        (60, 'R8_TYPELESS'),
        (62, 'R8_UINT'),
        (64, 'R8_SINT'),
        (71, 'BC1_UNORM'),
        (85, 'B5G6R5_UNORM'),
        (98, 'BC7_UNORM'),
        (119, 'console R16_UNORM_X8_TYPELESS'),
    )
    for fmt, label in refused_srv_formats:
        for op_name, blob in (('image_load', ps_even), ('image_load_mip', ps_mip_even)):
            expect_refused(
                lambda f=fmt, b=blob: G.Pipeline(VS_WORDS, b, root_blob(), ps_hw_inputs=PACKED_INPUT,
                                                 rtv_formats=UINT_RTV, srv_formats={0: f}),
                f'{op_name} d16 unsupported SRV format {fmt}',
                f'{op_name} srv_formats {label}',
            )
    print(f'srv formats: 14 exact formats match d16_image_format.h; {len(refused_srv_formats)} non-exact formats refused on load & load_mip')


def check_odd_dmask_consumers():
    def dummy_pipe():
        pipe = G.Pipeline.__new__(G.Pipeline)
        pipe.resources = {}
        pipe.uavs = {}
        pipe.rtv_formats = ()
        pipe.srv_formats = {}
        pipe.uds = []
        return pipe

    def stage_with(lines, *, has_branches=False):
        stage = G.Stage.__new__(G.Stage)
        stage.pipe = dummy_pipe()
        stage.kind = 'ps'
        stage.lines = []
        stage.sym = {16 + i: ('desc', ('table', 0), 0, i) for i in range(8)}
        stage.sym[24] = ('const', 0x00ff)
        stage.sym[25] = ('const', 0x1ffff)
        stage.used_s, stage.used_v = set(), set()
        stage.exec_known = True
        stage.exports = {}
        stage.write_masks = {}
        stage.attrs = {}
        stage.masks = set()
        stage.fetched = False
        stage.uniform_v = set()
        stage.varying_s = set()
        stage.scc_uniform = True
        stage.low16_only_v = set()
        stage.pinned_low16_v = set()
        stage.whole_read_v = set()
        stage.has_branches = has_branches
        stage.blocks = []
        for asm in lines:
            stage.one(asm.split(' ', 1)[0], asm)
        return stage

    odd_1 = 'image_load v4, v[0:1], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D d16'
    odd_3 = 'image_load v[6:7], v[0:1], s[16:23] dmask:0x7 dim:SQ_RSRC_IMG_2D d16'
    odd_mip_1 = 'image_load_mip v4, v[0:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D d16'
    odd_mip_3 = 'image_load_mip v[6:7], v[0:2], s[16:23] dmask:0x7 dim:SQ_RSRC_IMG_2D d16'

    accepted_sequences = (
        (odd_1, 'v_mov_b32_e32 v12, v4'),
        (odd_1, 'v_mov_b32_e32 v12, v4', 'v_lshlrev_b32_e32 v13, 16, v12',
         'v_lshrrev_b32_e32 v13, 16, v13'),
        (odd_1, 'v_mov_b32_e32 v12, v4', 'v_and_b32_e32 v12, 0xffff, v12',
         'v_cmp_eq_u32_e32 vcc_lo, 0, v12', 'v_cndmask_b32_e32 v13, 0, v12, vcc_lo'),
        (odd_1, 'v_cvt_f32_f16_e32 v4, v4', 'v_add_f32_e32 v4, 1.0, v4'),
        (odd_1, 'v_cvt_f32_ubyte0_e32 v4, v4', 'v_mul_f32_e32 v4, 2.0, v4'),
        (odd_1, 'v_cvt_f32_ubyte1_e32 v4, v4', 'v_mul_f32_e32 v4, 2.0, v4'),
        (odd_1, 'v_and_b32_e32 v4, 0xffff, v4', 'v_add_nc_u32_e32 v4, 1, v4'),
        (odd_1, 'v_and_b32_e32 v4, v4, s24', 'v_add_nc_u32_e32 v4, 1, v4'),
        (odd_1, 'v_and_b32_sdwa v4, v4, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0',
         'v_add_nc_u32_e32 v4, 1, v4'),
        (odd_1, 'v_cvt_f32_u32_sdwa v4, sext(v4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1',
         'v_add_f32_e32 v4, 1.0, v4'),
        (odd_1, 'v_cmp_eq_u32_sdwa vcc_lo, v4, v4 src0_sel:WORD_0 src1_sel:BYTE_0'),
        (odd_1, 'v_cmp_class_f16_e32 vcc_lo, -|v4|, 0x3ff'),
        (odd_1, 'v_fma_mixlo_f16 v8, v4, 1.0, 0 op_sel_hi:[1,0,0]', 'v_add_nc_u32_e32 v8, 1, v8'),
        (odd_1, 'v_fma_mixhi_f16 v4, 1.0, 1.0, 0', 'v_add_nc_u32_e32 v4, 1, v4'),
        (odd_3, 'v_add_nc_u32_e32 v6, 1, v6', 'v_cvt_f32_f16_e32 v7, v7', 'v_add_f32_e32 v7, 1.0, v7'),
        (odd_mip_1, 'v_cvt_f32_f16_e32 v4, v4', 'v_add_f32_e32 v4, 1.0, v4'),
        (odd_mip_3, 'v_add_nc_u32_e32 v6, 1, v6', 'v_and_b32_e32 v7, 0xffff, v7', 'v_add_nc_u32_e32 v7, 1, v7'),
    )
    for seq in accepted_sequences:
        stage_with(seq)

    refused_sequences = (
        ('v4 add', (odd_1, 'v_add_nc_u32_e32 v5, 1, v4'), 'whole-word read of v4'),
        ('v4 shift16', (odd_1, 'v_lshrrev_b32_e32 v5, 16, v4'), 'whole-word read of v4'),
        ('v4 copy then add',
         (odd_1, 'v_mov_b32_e32 v5, v4', 'v_add_nc_u32_e32 v8, 3, v5'),
         'upper half used by v_add_nc_u32_e32 v8, 3, v5'),
        ('v4 swap', (odd_1, 'v_swap_b32_e32 v4, v5'), 'whole-word read of v4'),
        ('v4 float add', (odd_1, 'v_add_f32_e32 v5, 1.0, v4'), 'whole-word read of v4'),
        ('v4 ubyte2', (odd_1, 'v_cvt_f32_ubyte2_e32 v5, v4'), 'whole-word read of v4'),
        ('v4 ubyte3', (odd_1, 'v_cvt_f32_ubyte3_e32 v5, v4'), 'whole-word read of v4'),
        ('v4 vgpr mask lhs', (odd_1, 'v_and_b32_e32 v5, v2, v4'), 'whole-word read of v4'),
        ('v4 vgpr mask rhs', (odd_1, 'v_and_b32_e32 v5, v4, v2'), 'whole-word read of v4'),
        ('v4 wide mask imm lhs', (odd_1, 'v_and_b32_e32 v5, 0x1ffff, v4'), 'whole-word read of v4'),
        ('v4 wide mask imm rhs', (odd_1, 'v_and_b32_e64 v5, v4, -1'), 'whole-word read of v4'),
        ('v4 wide mask hex rhs', (odd_1, 'v_and_b32_e64 v5, v4, 0x1ffff'), 'whole-word read of v4'),
        ('v4 wide mask sgpr lhs', (odd_1, 'v_and_b32_e32 v5, s25, v4'), 'whole-word read of v4'),
        ('v4 wide mask sgpr rhs', (odd_1, 'v_and_b32_e64 v5, v4, s25'), 'whole-word read of v4'),
        ('v4 sdwa word1',
         (odd_1, 'v_and_b32_sdwa v5, v4, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0'),
         'whole-word read of v4'),
        ('v4 sdwa byte2',
         (odd_1, 'v_and_b32_sdwa v5, v4, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_0'),
         'whole-word read of v4'),
        ('v4 sdwa preserve low',
         (odd_1, 'v_and_b32_sdwa v4, v4, v4 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 src1_sel:WORD_0',
          'v_add_nc_u32_e32 v5, 1, v4'),
         'whole-word read of v4'),
        ('v4 mixlo preserve high',
         (odd_1, 'v_fma_mixlo_f16 v4, 1.0, 1.0, 0', 'v_add_nc_u32_e32 v5, 1, v4'),
         'whole-word read of v4'),
        ('v4 compr export',
         (odd_1, 'exp mrt0 v4, v4, off, off done compr'),
         'whole-word read of v4'),
        ('v4 predicated overwrite',
         (odd_1, 'v_cmpx_eq_u32_e32 0, v0', 'v_mov_b32_e32 v4, 0', 'v_add_nc_u32_e32 v5, 1, v4'),
         'whole-word read of v4'),
        ('v4 reload under narrower EXEC',
         (odd_1, 'v_cmpx_eq_u32_e32 0, v0', odd_1, 'v_mov_b32_e32 v4, 0', 'v_add_nc_u32_e32 v5, 1, v4'),
         'whole-word read of v4'),
        ('v7 dmask 0x7 tail',
         (odd_3, 'v_add_nc_u32_e32 v5, 1, v7'),
         'whole-word read of v7'),
        ('mip v4 add',
         (odd_mip_1, 'v_add_nc_u32_e32 v5, 1, v4'),
         'whole-word read of v4'),
        ('mip v7 dmask 0x7 tail',
         (odd_mip_3, 'v_add_nc_u32_e32 v5, 1, v7'),
         'whole-word read of v7'),
        ('mip coord odd v4',
         (odd_1, 'image_load_mip v[6:7], [v0, v1, v4], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D d16'),
         'whole-word read of v4'),
    )
    for label, seq, msg in refused_sequences:
        expect_refused(lambda s=seq: stage_with(s), msg, label)

    def translate_ins(ins):
        stage = G.Stage.__new__(G.Stage)
        stage.pipe = dummy_pipe()
        stage.kind = 'ps'
        stage.ins = ins
        stage.lines = []
        stage.sym = {16 + i: ('desc', ('table', 0), 0, i) for i in range(8)}
        stage.used_s, stage.used_v = set(), set()
        stage.exec_known = True
        stage.exports = {}
        stage.write_masks = {}
        stage.attrs = {}
        stage.masks = set()
        stage.fetched = False
        stage.translate()

    forward_ins = [
        (0, 8, 'image_load', odd_1),
        (8, 4, 's_cbranch_scc1', 's_cbranch_scc1 label_0010'),
        (12, 4, 'v_mov_b32_e32', 'v_mov_b32_e32 v4, 0'),
        (16, 4, 'v_add_nc_u32_e32', 'v_add_nc_u32_e32 v5, 1, v4'),
        (20, 4, 's_endpgm', 's_endpgm'),
    ]
    expect_refused(lambda: translate_ins(forward_ins), 'whole-word read of v4', 'forward branch skip overwrite')

    loop_ins = [
        (0, 4, 'v_add_nc_u32_e32', 'v_add_nc_u32_e32 v5, 1, v4'),
        (4, 8, 'image_load', odd_1),
        (12, 4, 's_cbranch_scc1', 's_cbranch_scc1 label_0000'),
        (16, 4, 's_endpgm', 's_endpgm'),
    ]
    expect_refused(lambda: translate_ins(loop_ins), 'whole-word read of v4', 'backward loop odd d16 leak')

    loop_copy_ins = [
        (0, 4, 'v_mov_b32_e32', 'v_mov_b32_e32 v10, v4'),
        (4, 4, 'v_add_nc_u32_e32', 'v_add_nc_u32_e32 v11, 3, v10'),
        (8, 8, 'image_load', odd_1),
        (16, 4, 's_cbranch_scc1', 's_cbranch_scc1 label_0000'),
        (20, 4, 's_endpgm', 's_endpgm'),
    ]
    expect_refused(lambda: translate_ins(loop_copy_ins), 'upper half used by v_add_nc_u32_e32 v11, 3, v10',
                   'backward loop with odd d16 copy')

    print(f'odd-dmask consumers: {len(accepted_sequences)} low-16-bit paths accepted; {len(refused_sequences) + 3} whole-word/control-flow leaks refused')


def check_refusals():
    def build_ps(mimg_bytes):
        return G.Pipeline(VS_WORDS, PRE_WORDS + mimg_bytes + POST_WORDS, root_blob(), ps_hw_inputs=PACKED_INPUT)

    expect_refused(lambda: build_ps(MIMG_LLVM_MC['1d_d16_refused'][1]), '1D', '1D image_load d16')
    expect_refused(lambda: build_ps(MIMG_LLVM_MC['mip_1d_d16_refused'][1]), '1D', '1D image_load_mip d16')
    expect_refused(lambda: build_ps(MIMG_LLVM_MC['cube_d16_refused'][1]), 'SQ_RSRC_IMG_CUBE', 'cube image_load d16')
    expect_refused(lambda: build_ps(MIMG_LLVM_MC['mip_cube_d16_refused'][1]), 'SQ_RSRC_IMG_CUBE', 'cube image_load_mip d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD, dim=DIM_3D, a16=1, d16=1)), 'image_load', '3D a16+d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD, dim=DIM_2D_ARRAY, a16=1, d16=1)), 'image_load', '2D_ARRAY a16+d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD_MIP, dim=DIM_2D, a16=1, d16=1)), 'image_load_mip a16', 'mip 2D a16+d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD_MIP, dim=DIM_3D, a16=1, d16=1)), 'image_load_mip a16', 'mip 3D a16+d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD_MIP, dim=DIM_2D_ARRAY, a16=1, d16=1)), 'image_load_mip a16', 'mip 2D_ARRAY a16+d16')
    for op_name, op_code in (('image_load', IMG_LOAD), ('image_load_mip', IMG_LOAD_MIP)):
        expect_refused(lambda o=op_code: build_ps(encode_mimg(o, dmask=0, d16=1)), 'dmask:0x0', f'{op_name} dmask 0 d16')
        expect_refused(lambda o=op_code: build_ps(encode_mimg(o, tfe=1, d16=1)), 'tfe', f'{op_name} tfe d16')
        expect_refused(lambda o=op_code: build_ps(encode_mimg(o, lwe=1, d16=1)), 'lwe', f'{op_name} lwe d16')
        expect_refused(lambda o=op_code: build_ps(encode_mimg(o, r128=1, d16=1)), 'r128', f'{op_name} r128 d16')
        expect_refused(lambda o=op_code: build_ps(encode_mimg(o, w0_reserved=0x40, d16=1)), 'reserved', f'{op_name} w0 reserved d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD, a16=1, d16=1, nsa=1, extra_addrs=(1,))), 'invalid_nsa', 'a16+d16 fake NSA')

    sample_pre = PRE_WORDS + bytes.fromhex('800398be800399be80039abe80039bbe')
    expect_refused(
        lambda: G.Pipeline(VS_WORDS, sample_pre + encode_mimg(IMG_SAMPLE, ssamp=24, d16=1) + POST_WORDS,
                           root_blob(), ps_hw_inputs=PACKED_INPUT),
        'image_sample modifier d16', 'implicit-LOD D16 sample',
    )

    pipe = G.Pipeline(VS_WORDS, PS_WORDS_A, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV)
    expect_refused(
        lambda: pipe.ps.image(['v[4:7]', 'v[8:9]', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0xf', 'd16'], 'load'),
        'image_load d16 destination v[4:7] for dmask 0xf',
        '4-VGPR dst on dmask 0xf d16',
    )
    expect_refused(
        lambda: pipe.ps.image(['v[4:5]', 'v[8:9]', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0x3', 'd16'], 'load'),
        'image_load d16 destination v[4:5] for dmask 0x3',
        '2-VGPR dst on dmask 0x3 d16',
    )
    expect_refused(
        lambda: pipe.ps.image(['v4', 'v8', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0x1', 'd16'], 'load'),
        'image_load address or modifiers',
        '1-VGPR addr without a16',
    )
    expect_refused(
        lambda: pipe.ps.image(['v[4:7]', 'v[8:10]', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0xf', 'd16'], 'load_mip'),
        'image_load_mip d16 destination v[4:7] for dmask 0xf',
        'mip 4-VGPR dst on dmask 0xf d16',
    )
    expect_refused(
        lambda: pipe.ps.image(['v[4:5]', 'v[8:10]', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0x3', 'd16'], 'load_mip'),
        'image_load_mip d16 destination v[4:5] for dmask 0x3',
        'mip 2-VGPR dst on dmask 0x3 d16',
    )
    expect_refused(
        lambda: pipe.ps.image(['v[4:5]', 'v[8:9]', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0xf', 'd16'], 'load_mip'),
        'image_load_mip address or modifiers',
        'mip 2-VGPR addr on 2D d16',
    )
    pipe_d = G.Pipeline(VS_WORDS, PS_WORDS_D, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV)
    expect_refused(
        lambda: pipe_d.ps.image(['v[4:5]', 'v[8:10]', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D_ARRAY', 'dmask:0xf', 'd16'], 'load_mip'),
        'image_load_mip address or modifiers',
        'mip 3-VGPR addr on 2D_ARRAY d16',
    )
    cs_stage = G.Stage.__new__(G.Stage)
    cs_stage.kind = 'cs'
    expect_refused(
        lambda: cs_stage.image(['v[4:5]', 'v[0:2]', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0xf', 'd16'], 'load_mip'),
        'compute d16 image load',
        'compute image_load_mip d16',
    )
    print('refusals: implicit-LOD sample d16, 1D/cube, mip/3D/array a16, compute d16, tfe/lwe/r128/reserved and bad tuple widths refused')


def controls():
    check_rounding_formula()
    check_decoder_and_masks()
    check_srv_format_controls()
    check_odd_dmask_consumers()
    check_odd_tail_programs()
    check_refusals()


def check_binary(llvm_mc):
    assert assemble(llvm_mc, PS_SOURCE_A) == PS_WORDS_A, 'PS_SOURCE_A encoding mismatch'
    assert assemble(llvm_mc, PS_SOURCE_B) == PS_WORDS_B, 'PS_SOURCE_B encoding mismatch'
    assert assemble(llvm_mc, PS_SOURCE_C) == PS_WORDS_C, 'PS_SOURCE_C encoding mismatch'
    assert assemble(llvm_mc, PS_SOURCE_D) == PS_WORDS_D, 'PS_SOURCE_D encoding mismatch'
    for key, (inst, expected) in sorted(MIMG_LLVM_MC.items()):
        built = assemble(llvm_mc, inst + '\n')
        if built != expected:
            raise AssertionError(f'{key}: assembled {built.hex()}, recorded {expected.hex()}')
    print(f'llvm-mc: {len(MIMG_LLVM_MC)} MIMG encodings and all four pixel fixtures match recorded words')


def check_decode(library):
    from pixel_aot import disassemble
    blob = b''.join(MIMG_LLVM_MC[k][1] for k in DECODE_KEYS)
    llvm_lines = disassemble(blob, library)
    ours = [D.decode_instruction(MIMG_LLVM_MC[k][1], 0)[2] for k in DECODE_KEYS]
    assert llvm_lines == ours, f'LLVM vs disasm_rdna2 mismatch: {list(zip(llvm_lines, ours))}'
    print(f'libLLVM: {len(DECODE_KEYS)} MIMG instructions match disasm_rdna2.py verbatim')


def mutate_hlsl_a(source, mutant):
    if mutant == 'rtz':
        assert source.count(PACK16_RNE_HLSL) == 1 and source.count(PACK16_D16_HLSL) == 1, source
        mutated = (source.replace(PACK16_RNE_HLSL, PACK16_RTZ_HLSL, 1)
                         .replace('return unorm16 != 0u ? pack16_unorm16(value) : pack16_rne(value);',
                                  'return pack16_rne(value);', 1))
    elif mutant == 'double_round':
        target = 'return unorm16 != 0u ? pack16_unorm16(value) : pack16_rne(value);'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 'return pack16_rne(value);', 1)
    elif mutant == 'unpacked_x':
        target_load = 'v4 = pack16_d16(t.x, ud4);'
        target_cvt = 'v4 = U(f16tof32(v4));'
        assert source.count(target_load) == 1 and source.count(target_cvt) == 1, source
        mutated = source.replace(target_load, 'v4 = U(t.x);', 1).replace(target_cvt, 'v4 = v4;', 1)
    elif mutant == 'swap_halves':
        target = 'v5 = pack16_d16(t.x, ud4) | (pack16_d16(t.z, ud4) << 16);'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 'v5 = pack16_d16(t.z, ud4) | (pack16_d16(t.x, ud4) << 16);', 1)
    elif mutant == 'drop_shift':
        target = 'v5 = pack16_d16(t.x, ud4) | (pack16_d16(t.z, ud4) << 16);'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 'v5 = pack16_d16(t.x, ud4) | pack16_d16(t.z, ud4);', 1)
    elif mutant == 'drop_a16':
        target = 't = r0.Load(int3(v0 & 0xffffu, v0 >> 16, 0));'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 't = r0.Load(int3(v0, v1, 0));', 1)
    elif mutant == 'drop_nsa':
        target = 't = r0.Load(int3(v2, v9, 0));'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 't = r0.Load(int3(v2, v3, 0));', 1)
    elif mutant == 'poison_odd_hi':
        t4 = 'v4 = pack16_d16(t.x, ud4);'
        t7 = 'v7 = pack16_d16(t.z, ud4);'
        assert source.count(t4) == 1 and source.count(t7) == 1, source
        mutated = (source.replace(t4, 'v4 = 0xdead0000u | pack16_d16(t.x, ud4);', 1)
                         .replace(t7, 'v7 = 0xdead0000u | pack16_d16(t.z, ud4);', 1))
    elif mutant == 'unmasked_odd_hi':
        t7 = 'v7 = pack16_d16(t.z, ud4);'
        sdwa7 = 'v7 = ((v7 >> 0u) & 0xffffu) & ((v7 >> 0u) & 0xffffu);'
        assert source.count(t7) == 1 and source.count(sdwa7) == 1, source
        mutated = (source.replace(t7, 'v7 = 0xdead0000u | pack16_d16(t.z, ud4);', 1)
                         .replace(sdwa7, 'v7 = v7;', 1))
    else:
        raise ValueError(mutant)
    assert mutated != source, mutant
    return mutated


def mutate_hlsl_c(source, mutant):
    if mutant == 'mip_zero':
        t_contig = 't = r0.Load(int3(v8, v9, v10));'
        t_nsa = 't = r0.Load(int3(v2, v9, v10));'
        assert source.count(t_contig) == 2 and source.count(t_nsa) == 1, source
        mutated = (source.replace(t_contig, 't = r0.Load(int3(v8, v9, 0));')
                         .replace(t_nsa, 't = r0.Load(int3(v2, v9, 0));'))
    elif mutant == 'mip_drop_nsa':
        target = 't = r0.Load(int3(v2, v9, v10));'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 't = r0.Load(int3(v2, v3, v10));', 1)
    elif mutant == 'mip_rtz':
        assert source.count(PACK16_RNE_HLSL) == 1 and source.count(PACK16_D16_HLSL) == 1, source
        mutated = (source.replace(PACK16_RNE_HLSL, PACK16_RTZ_HLSL, 1)
                         .replace('return unorm16 != 0u ? pack16_unorm16(value) : pack16_rne(value);',
                                  'return pack16_rne(value);', 1))
    elif mutant == 'mip_double_round':
        target = 'return unorm16 != 0u ? pack16_unorm16(value) : pack16_rne(value);'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 'return pack16_rne(value);', 1)
    elif mutant == 'poison_odd_hi':
        t4 = 'v4 = pack16_d16(t.x, ud4);'
        t7 = 'v7 = pack16_d16(t.z, ud4);'
        assert source.count(t4) == 1 and source.count(t7) == 1, source
        mutated = (source.replace(t4, 'v4 = 0xdead0000u | pack16_d16(t.x, ud4);', 1)
                         .replace(t7, 'v7 = 0xdead0000u | pack16_d16(t.z, ud4);', 1))
    else:
        raise ValueError(mutant)
    assert mutated != source, mutant
    return mutated


def mutate_hlsl_d(source, mutant):
    if mutant == 'swap_slice_mip':
        t_contig = 't = r0.Load(int4(v8, v9, v10, v11));'
        t_nsa = 't = r0.Load(int4(v12, v9, v10, v11));'
        assert source.count(t_contig) == 1 and source.count(t_nsa) == 2, source
        mutated = (source.replace(t_contig, 't = r0.Load(int4(v8, v9, v11, v10));')
                         .replace(t_nsa, 't = r0.Load(int4(v12, v9, v11, v10));'))
    elif mutant == 'array_drop_nsa':
        t_nsa = 't = r0.Load(int4(v12, v9, v10, v11));'
        assert source.count(t_nsa) == 2, source
        mutated = source.replace(t_nsa, 't = r0.Load(int4(v12, v13, v10, v11));')
    elif mutant == 'poison_odd_hi':
        t6 = 'v6 = pack16_d16(t.x, ud4);'
        assert source.count(t6) == 1, source
        mutated = source.replace(t6, 'v6 = 0xdead0000u | pack16_d16(t.x, ud4);', 1)
    else:
        raise ValueError(mutant)
    assert mutated != source, mutant
    return mutated



def odd_tail_program(components, *, unsafe=False, red_only=False):
    # A fullscreen pixel reads its own texel. VOP1 encodings follow the public ISA.
    def vop1(op, dst, src):
        return 0x7e000000 | (dst << 17) | (op << 9) | (256 + src)

    first = 20
    tail = first + components // 2
    load = encode_mimg(IMG_LOAD, dmask=(1 << components) - 1, vaddr=8, vdata=first, d16=1)
    instructions = [0xbf8c3f70, vop1(1, 31, tail)]  # s_waitcnt vmcnt(0); v_mov_b32
    copied = 31
    if components == 1:
        instructions.append(vop1(1, 27, copied))
        copied = 27
    if unsafe:
        instructions.append((0x25 << 25) | (29 << 17) | (copied << 9) | 131)  # v_add_nc_u32 with 3
    elif not red_only:
        instructions.append(vop1(0x0b, 29, copied))  # v_cvt_f32_f16
    output = 0xf8001c03 if red_only else 0xf800180f  # exp mrt0, RG compressed or RGBA
    # For the three-channel load, also put the copied tail above the first
    # component, then unpack that half as integer bits into output G.
    if components == 3 and not red_only and not unsafe:
        instructions.extend((
            *struct.unpack('<II', encode_v_and_low16(first)),
            (0x1a << 25) | (25 << 17) | (copied << 9) | 144,  # v_lshlrev_b32 v25, 16, v31
            (0x1c << 25) | (25 << 17) | (25 << 9) | (256 + first),  # v_or_b32 v25, v20, v25
            (0x16 << 25) | (25 << 17) | (25 << 9) | 144,  # v_lshrrev_b32 v25, 16, v25
        ))
    payload = copied if red_only else 29 * 0x01010101
    if components == 3 and not red_only and not unsafe:
        payload = (payload & ~0xff00) | (25 << 8)
    instructions.extend((output, payload, 0xbf810000))
    return PRE_WORDS + load + struct.pack(f'<{len(instructions)}I', *instructions)


def odd_tail_hlsl(components, **options):
    rtv = (41,) if options.get('red_only') else UINT_RTV
    return G.Pipeline(VS_WORDS, odd_tail_program(components, **options), root_blob(),
                      ps_hw_inputs=PACKED_INPUT, rtv_formats=rtv).hlsl_ps()


def masked_odd_tail_hlsl(*, intervening=(), wrong_overwrite=False):
    # Only the left four columns load and overwrite. Initialize the exported
    # registers before narrowing EXEC so the right four columns have a reference.
    def v_mov(dst, src):
        return 0x7e000200 | (dst << 17) | src

    load = encode_mimg(IMG_LOAD, dmask=7, vaddr=8, vdata=20, d16=1)
    before = (
        v_mov(29, 240),  # 0.5
        v_mov(30, 246),  # 4.0
        0x7da81084,  # v_cmpx_gt_u32_e32 4, v8
    )
    after = (
        0xbf8c3f70,  # s_waitcnt vmcnt(0)
        0x7e3a1715,  # v_cvt_f32_f16_e32 v29, v21
        *intervening,
        v_mov(21, 246 if wrong_overwrite else 242),  # 4.0 instead of 1.0 in the GPU control
        (0x03 << 25) | (30 << 17) | (21 << 9) | 242,  # v_add_f32_e32 v30, 1.0, v21
        0xbefe03c1,  # s_mov_b32 exec_lo, -1
        0xf800180f,  # exp mrt0 v29, v30, v29, v30
        0x1e1d1e1d,
        0xbf810000,  # s_endpgm
    )
    program = PRE_WORDS + struct.pack('<3I', *before) + load + struct.pack(f'<{len(after)}I', *after)
    return G.Pipeline(VS_WORDS, program, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV).hlsl_ps()


def masked_odd_tail_words():
    words = []
    for i in range(WIDTH * HEIGHT):
        if i % WIDTH < 4:
            low_half = f32_bits_from_f16_bits(0x4400 + i)
            result = 0x40000000  # 1.0 + 1.0
        else:
            low_half, result = 0x3f000000, 0x40800000
        words.extend((low_half, result, low_half, result))
    return words


def odd_tail_variants(components, *, red_only=False):
    hlsl = odd_tail_hlsl(components, red_only=red_only)
    tail = 20 + components // 2
    # Keep the valid half for poison, and replace the entire result with a neighbour for the failing mutant.
    channel = 'z' if components == 3 else 'x'
    pattern = rf'(v{tail} = )(pack16_d16\(([^;,]+)\.{channel}, (ud\d+)\))(;)'
    match = re.search(pattern, hlsl)
    assert match, hlsl
    tex, mode = match[3], match[4]
    neighbour = f'pack16_d16({tex}.y, {mode})'
    variants = {
        'reference': hlsl,
        'poison_upper': hlsl[:match.start(2)] + f'({match[2]} | ({neighbour} << 16))' + hlsl[match.end(2):],
        'wrong_component': hlsl[:match.start(2)] + neighbour + hlsl[match.end(2):],
    }
    return variants


def odd_tail_words(base, components, red_only):
    words = []
    for i in range(WIDTH * HEIGHT):
        half = base + i
        f32 = f32_bits_from_f16_bits(half)
        if red_only:
            words.extend((f32, 0, 0, 0))
        else:
            words.extend((f32, half if components == 3 else f32, f32, f32))
    return words


ODD_TAIL_CASES = ((3, False), (1, False), (3, True))


def check_odd_tail_programs():
    for components, red_only in ODD_TAIL_CASES:
        odd_tail_variants(components, red_only=red_only)
    expect_refused(lambda: odd_tail_hlsl(1, unsafe=True),
                   'upper half used by v_add_nc_u32_e32 v29, 3, v27', 'integer addition after two copies')
    expect_refused(
        lambda: G.Pipeline(VS_WORDS, odd_tail_program(3, red_only=True), root_blob(),
                           ps_hw_inputs=PACKED_INPUT, rtv_formats=(2,)).hlsl_ps(),
        'upper half used by exp mrt0', 'same compressed export to RGBA32_FLOAT')
    masked_odd_tail_hlsl()
    for label, word in (
        ('EXEC narrowed after odd load', 0x7da81082),  # v_cmpx_gt_u32_e32 2, v8
        ('EXEC rewritten after odd load', 0xbefe037e),  # s_mov_b32 exec_lo, exec_lo
        ('branch after odd load', 0xbf820000),  # s_branch to the following instruction
    ):
        expect_refused(lambda: masked_odd_tail_hlsl(intervening=(word,)),
                       'whole-word read of v21 after odd d16 image_load: upper half used by '
                       'v_add_f32_e32 v30, 1.0, v21', label)


def check_odd_tail_draw(dxc, renderer, adapter):
    check_odd_tail_programs()
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='image-load-d16-odd-', dir=out_dir) as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs', 'vs_6_0', directory)
        for components, red_only in ODD_TAIL_CASES:
            base = 0x4400 if components == 3 else 0x3800
            expected = odd_tail_words(base, components, red_only)
            mode = 'rgba16f_r32f' if red_only else 'rgba16f'
            for variant, source in odd_tail_variants(components, red_only=red_only).items():
                tag = f'odd_{components}_{red_only}_{variant}'
                ps = compile_hlsl(dxc, 'dxc', source, tag, 'ps_6_0', directory)
                run = run_d16_draw(renderer, adapter, mode, vs, ps, expected, directory, tag)
                if variant == 'wrong_component':
                    assert run.returncode == 1 and '16 of 16 pixels differ' in run.stderr, (run.stdout, run.stderr)
                    wrong = odd_tail_words(0xc000, components, red_only)
                    run = run_d16_draw(renderer, adapter, mode, vs, ps, wrong, directory, tag + '_mutated_model')
                assert run.returncode == 0, (run.stdout, run.stderr)
                print(f'{tag}: readback matched expected outcome', flush=True)

        expected = masked_odd_tail_words()
        for wrong_overwrite in (False, True):
            tag = 'masked_odd_tail_wrong_overwrite' if wrong_overwrite else 'masked_odd_tail'
            ps = compile_hlsl(dxc, 'dxc', masked_odd_tail_hlsl(wrong_overwrite=wrong_overwrite), tag,
                              'ps_6_0', directory)
            run = run_d16_draw(renderer, adapter, 'rgba16f', vs, ps, expected, directory, tag)
            if wrong_overwrite:
                assert run.returncode == 1 and '8 of 16 pixels differ' in run.stderr, (run.stdout, run.stderr)
            else:
                assert run.returncode == 0, (run.stdout, run.stderr)
            print(f'{tag}: readback matched expected outcome', flush=True)

def run_d16_draw(renderer, adapter, tex_mode, vs, ps, words, directory, tag):
    expected_path = directory / f'{tag}.bin'
    expected_path.write_bytes(struct.pack('<64I', *words))
    run = subprocess.run(
        [renderer, adapter, '--startup-image-load-d16', tex_mode, str(vs), str(ps), str(expected_path)],
        capture_output=True,
        text=True,
    )
    if run.returncode == 77:
        print(run.stdout + run.stderr, end='', flush=True)
        sys.exit(77)
    if 'debug_layer=enabled' in run.stdout:
        assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    hlsl_a = G.Pipeline(VS_WORDS, PS_WORDS_A, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV).hlsl_ps()
    hlsl_b = G.Pipeline(VS_WORDS, PS_WORDS_B, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV).hlsl_ps()
    hlsl_c = G.Pipeline(VS_WORDS, PS_WORDS_C, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV).hlsl_ps()
    hlsl_d = G.Pipeline(VS_WORDS, PS_WORDS_D, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV).hlsl_ps()
    mutants_a = ('rtz', 'double_round', 'unpacked_x', 'swap_halves', 'drop_shift', 'drop_a16', 'drop_nsa', 'unmasked_odd_hi')
    mutants_c = ('mip_zero', 'mip_drop_nsa', 'mip_rtz', 'mip_double_round')
    mutants_d = ('swap_slice_mip', 'array_drop_nsa')

    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='image-load-d16-', dir=out_dir) as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, f'vs_{kind}', vs_profile, directory)
            ps_a = compile_hlsl(compiler, kind, hlsl_a, f'ps_a_{kind}', ps_profile, directory)
            ps_b = compile_hlsl(compiler, kind, hlsl_b, f'ps_b_{kind}', ps_profile, directory)
            ps_c = compile_hlsl(compiler, kind, hlsl_c, f'ps_c_{kind}', ps_profile, directory)
            ps_d = compile_hlsl(compiler, kind, hlsl_d, f'ps_d_{kind}', ps_profile, directory)
            ps_poison_a = compile_hlsl(compiler, kind, mutate_hlsl_a(hlsl_a, 'poison_odd_hi'),
                                       f'ps_poison_a_{kind}', ps_profile, directory)
            ps_poison_c = compile_hlsl(compiler, kind, mutate_hlsl_c(hlsl_c, 'poison_odd_hi'),
                                       f'ps_poison_c_{kind}', ps_profile, directory)
            ps_poison_d = compile_hlsl(compiler, kind, mutate_hlsl_d(hlsl_d, 'poison_odd_hi'),
                                       f'ps_poison_d_{kind}', ps_profile, directory)

            for tex_mode in ('r16_unorm', 'rgba32f'):
                ref_a = expected_words('A', tex_mode)
                ref_b = expected_words('B', tex_mode)
                ref_c = expected_words('C', tex_mode)
                run_a = run_d16_draw(renderer, adapter, tex_mode, vs, ps_a, ref_a, directory, f'ref_a_{tex_mode}_{kind}')
                assert run_a.returncode == 0, f'{kind} A {tex_mode} failed:\n{run_a.stdout}{run_a.stderr}'
                run_b = run_d16_draw(renderer, adapter, tex_mode, vs, ps_b, ref_b, directory, f'ref_b_{tex_mode}_{kind}')
                assert run_b.returncode == 0, f'{kind} B {tex_mode} failed:\n{run_b.stdout}{run_b.stderr}'
                run_c = run_d16_draw(renderer, adapter, tex_mode, vs, ps_c, ref_c, directory, f'ref_c_{tex_mode}_{kind}')
                assert run_c.returncode == 0, f'{kind} C {tex_mode} failed:\n{run_c.stdout}{run_c.stderr}'
                run_pa = run_d16_draw(renderer, adapter, tex_mode, vs, ps_poison_a, ref_a,
                                      directory, f'poison_a_{tex_mode}_{kind}')
                assert run_pa.returncode == 0, f'{kind} poison_a {tex_mode} leaked:\n{run_pa.stdout}{run_pa.stderr}'
                run_pc = run_d16_draw(renderer, adapter, tex_mode, vs, ps_poison_c, ref_c,
                                      directory, f'poison_c_{tex_mode}_{kind}')
                assert run_pc.returncode == 0, f'{kind} poison_c {tex_mode} leaked:\n{run_pc.stdout}{run_pc.stderr}'
                arr_mode = f'{tex_mode}_array'
                ref_d = expected_words('D', arr_mode)
                run_d = run_d16_draw(renderer, adapter, arr_mode, vs, ps_d, ref_d, directory, f'ref_d_{arr_mode}_{kind}')
                assert run_d.returncode == 0, f'{kind} D {arr_mode} failed:\n{run_d.stdout}{run_d.stderr}'
                run_pd = run_d16_draw(renderer, adapter, arr_mode, vs, ps_poison_d, ref_d,
                                      directory, f'poison_d_{arr_mode}_{kind}')
                assert run_pd.returncode == 0, f'{kind} poison_d {arr_mode} leaked:\n{run_pd.stdout}{run_pd.stderr}'
                print(f'{kind} {tex_mode}[/_array]: fixtures A/B/C/D and odd-dmask poison checks matched all words', flush=True)

            # Every mutation must fail against the reference and match its mutated model across every
            # applicable texture mode (drop_shift uses rgba32f because t.z is 0.0 on R16_UNORM; double_round
            # and mip_double_round use r16_unorm to exercise 0xffaf -> 0x3bfe vs 0x3bfd and 0xffef -> 0x3c00 vs 0x3bff).
            mutant_cases = (
                [('A', mut, mutate_hlsl_a(hlsl_a, mut),
                  ('rgba32f',) if mut == 'drop_shift' else ('r16_unorm',) if mut == 'double_round' else ('r16_unorm', 'rgba32f'))
                 for mut in mutants_a]
                + [('C', mut, mutate_hlsl_c(hlsl_c, mut),
                    ('r16_unorm',) if mut == 'mip_double_round' else ('r16_unorm', 'rgba32f'))
                   for mut in mutants_c]
                + [('D', mut, mutate_hlsl_d(hlsl_d, mut), ('r16_unorm_array', 'rgba32f_array'))
                   for mut in mutants_d]
            )
            for fixture, mut, mutated_hlsl, modes in mutant_cases:
                ps_mut = compile_hlsl(compiler, kind, mutated_hlsl, f'mut_{mut}_{kind}', ps_profile, directory)
                for tex_mode in modes:
                    ref_words = expected_words(fixture, tex_mode)
                    fail_run = run_d16_draw(renderer, adapter, tex_mode, vs, ps_mut, ref_words,
                                            directory, f'fail_{mut}_{tex_mode}_{kind}')
                    assert fail_run.returncode == 1, (
                        f'{kind} mutant {mut} ({tex_mode}) unexpectedly passed:\n{fail_run.stdout}{fail_run.stderr}'
                    )
                    mut_words = expected_words(fixture, tex_mode, mutant=mut)
                    match_run = run_d16_draw(renderer, adapter, tex_mode, vs, ps_mut, mut_words,
                                             directory, f'model_{mut}_{tex_mode}_{kind}')
                    assert match_run.returncode == 0, (
                        f'{kind} mutant {mut} ({tex_mode}) did not match model:\n{match_run.stdout}{match_run.stderr}'
                    )
                print(f'{kind} negative control ({mut}): rejected against reference and matched mutated model', flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary' and len(args) == 1:
        check_binary(args[0])
    elif mode == 'decode' and len(args) == 1:
        check_decode(args[0])
    elif mode == 'odd-draw' and len(args) == 3:
        check_odd_tail_draw(*args)
    elif mode == 'draw' and len(args) in (3, 4):
        controls()
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
