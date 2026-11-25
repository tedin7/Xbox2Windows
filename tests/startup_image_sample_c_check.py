"""Depth-comparison sampling (image_sample_c, image_sample_c_lz) in tools/startup/gfx2hlsl.py.

  (no arguments)     translation controls, pure Python
  binary <llvm-mc>   reassemble every fixture source and compare it with the recorded words

Each fixture is a source-built gfx1013 pixel stage that sets v0 = 0.5, v1 = 0.25, v2 = 1.0, v3 = 2.0, builds a
constant comparison sampler in s[24:27] (point filtering, clamp, one SQ_TEX_DEPTH_COMPARE function) and runs one
image instruction on a texture from descriptor table 0. The address dwords of a comparison are the reference, then
u and v, so the contiguous form v[0:2] compares v0 against the texel at (v1, v2) and the NSA form [v3, v0, v1]
compares v3 at (v0, v1): the lowering must follow the position, not the register number. A 2D array (dim:SQ_RSRC_IMG_2D_ARRAY)
adds the slice as a fourth address register and becomes a Texture2DArray compared at float3(u, v, slice). The sampler is declared
as a SamplerComparisonState and listed as an smpc manifest line, even where its words equal an ordinary sampler's.

HLSL has no SampleCmpLevel before shader model 6.7, so image_sample_c_l is refused, as are the forms and
modifiers the lowering does not read. The comparison results themselves are checked on D3D12 for the same sampler
words by tests/pixel_compare_check.py; startup_sampler_comparison_check.cpp checks what the runtime binds for an smpc line.
"""
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, assemble, root_blob  # noqa: E402

PRE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, {w0:#x}
s_mov_b32 s25, {w1:#x}
s_mov_b32 s26, {w2:#x}
s_mov_b32 s27, 0
v_mov_b32 v0, 0.5
v_mov_b32 v1, 0x3e800000
v_mov_b32 v2, 1.0
v_mov_b32 v3, 2.0
s_waitcnt lgkmcnt(0)
"""
POST = """\
s_waitcnt vmcnt(0)
exp mrt0 v4, v4, v4, v4 done vm
s_endpgm
"""
# SQ_IMG_SAMP_WORD0: CLAMP_LAST_TEXEL (2) in X and Y, the comparison function in bits 14:12, and TRUNC_COORD in
# bit 27 for point filtering. Word 2 bits 20..23 are the magnification and minification filters.
POINT = 2 | 2 << 3 | 1 << 27
LESS = 1 << 12
BILINEAR_W2 = 1 << 20 | 1 << 22
IMAGE = {
    'lz': 'image_sample_c_lz v4, v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'lz_nsa': 'image_sample_c_lz v4, [v3, v0, v1], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'c': 'image_sample_c v4, v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'c_nsa': 'image_sample_c v4, [v3, v0, v1], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'c_l': 'image_sample_c_l v4, v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'c_l_nsa': 'image_sample_c_l v4, [v3, v0, v1, v2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'glc': 'image_sample_c_lz v4, v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D glc',
    'slc': 'image_sample_c_lz v4, v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D slc',
    'dlc': 'image_sample_c_lz v4, v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D dlc',
    'dmask3': 'image_sample_c_lz v[4:6], v[0:2], s[16:23], s[24:27] dmask:0x7 dim:SQ_RSRC_IMG_2D',
    'dmask2': 'image_sample_c_lz v4, v[0:2], s[16:23], s[24:27] dmask:0x2 dim:SQ_RSRC_IMG_2D',
    'array': 'image_sample_c_lz v4, v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY',
    'array_nsa': 'image_sample_c_lz v4, [v3, v0, v1, v2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY',
    'array_c': 'image_sample_c v4, v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY',
    'array_c_l': 'image_sample_c_l v4, v[0:4], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY',
    'array_dmask3': 'image_sample_c_lz v[4:6], v[0:3], s[16:23], s[24:27] dmask:0x7 dim:SQ_RSRC_IMG_2D_ARRAY',
    'array_offset': 'image_sample_c_lz_o v4, v[0:4], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY',
    'array_unorm': 'image_sample_c_lz v4, v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY unorm',
    'three': 'image_sample_c_lz v4, v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_3D',
    'cube': 'image_sample_c_lz v4, v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_CUBE',
    'dim1d': 'image_sample_c_lz v4, v[0:1], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_1D',
    'offset': 'image_sample_c_lz_o v4, v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'bias': 'image_sample_c_b v4, v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'clamp': 'image_sample_c_cl v4, v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'gather': 'image_gather4_c v[4:7], v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'a16': 'image_sample_c_lz v4, v[0:1], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D a16',
    'tfe': 'image_sample_c_lz v[4:5], v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D tfe',
    'unorm': 'image_sample_c_lz v4, v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm',
    'lwe': 'image_sample_c_lz v4, v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D lwe',
    'd16': 'image_sample_c_lz v4, v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D d16',
    'r128': 'image_sample_c_lz v4, v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D r128',
}
# SQ_IMG_SAMP_WORD1 bits 23:12 are MAX_LOD (4.8 fixed point). The translated static sampler never limits the
# LOD, so an implicit comparison is lowered only when the source limits it to level zero.
MAX_LOD_FULL = 0xfff << 12
W1 = {'c': MAX_LOD_FULL, 'c_nsa': MAX_LOD_FULL, 'c_max_partial': 1 << 12, 'array_c_full': MAX_LOD_FULL}
# name -> (image key, sampler word 0, sampler word 2)
VARIANTS = {name: (name, POINT | LESS, 0) for name in IMAGE}
VARIANTS.update({'func%d' % n: ('lz', POINT | n << 12, 0) for n in range(8) if n != 1})
VARIANTS['bilinear'] = ('lz', 2 | 2 << 3 | LESS, BILINEAR_W2)
VARIANTS['zero_sampler'] = ('lz', 0, 0)
VARIANTS['point_no_trunc'] = ('lz', 2 | 2 << 3 | LESS, 0)
VARIANTS['anisotropic'] = ('lz', POINT | LESS | 1 << 9, 0)
VARIANTS['wrap_x'] = ('lz', (POINT | LESS) & ~7, 0)
VARIANTS['mixed'] = ('mixed', POINT | LESS, 0)
VARIANTS['c_max0'] = ('c', POINT | LESS, 0)
VARIANTS['c_max_partial'] = ('c', POINT | LESS, 0)
VARIANTS['array_c_full'] = ('array_c', POINT | LESS, 0)
WORDS = {
    'lz': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'lz_nsa': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0a01bcf00304c40000010000'
        '703f8cbf0f1800f804040404000081bf'),
    'c': bytes.fromhex(
        '04040cf4000000faff0398be12100008ff0399be00f0ff0080039abe80039bbe'
        'f002007eff02027e0000803ef202047ef402067e7fc08cbf0801a0f00004c400'
        '703f8cbf0f1800f804040404000081bf'),
    'c_nsa': bytes.fromhex(
        '04040cf4000000faff0398be12100008ff0399be00f0ff0080039abe80039bbe'
        'f002007eff02027e0000803ef202047ef402067e7fc08cbf0a01a0f00304c400'
        '00010000703f8cbf0f1800f804040404000081bf'),
    'c_l': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801b0f00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'c_l_nsa': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0a01b0f00304c40000010200'
        '703f8cbf0f1800f804040404000081bf'),
    'glc': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0821bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'slc': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf20004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'dlc': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf8801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'dmask3': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0807bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'dmask2': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0802bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'array': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf2801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'array_nsa': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf2a01bcf00304c40000010200'
        '703f8cbf0f1800f804040404000081bf'),
    'array_c': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf2801a0f00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'array_c_l': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf2801b0f00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'array_dmask3': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf2807bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'array_offset': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf2801fcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'array_unorm': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf2811bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'array_c_full': bytes.fromhex(
        '04040cf4000000faff0398be12100008ff0399be00f0ff0080039abe80039bbe'
        'f002007eff02027e0000803ef202047ef402067e7fc08cbf2801a0f00004c400'
        '703f8cbf0f1800f804040404000081bf'),
    'three': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf1001bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'cube': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf1801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'dim1d': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0001bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'offset': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801fcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'bias': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801b4f00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'clamp': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801a4f00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'gather': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf080120f10004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'a16': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c440703f8cbf'
        '0f1800f804040404000081bf'),
    'tfe': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bdf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'unorm': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0811bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'lwe': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bef00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'd16': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c480703f8cbf'
        '0f1800f804040404000081bf'),
    'r128': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0881bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'func0': bytes.fromhex(
        '04040cf4000000faff0398be12000008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'func2': bytes.fromhex(
        '04040cf4000000faff0398be12200008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'func3': bytes.fromhex(
        '04040cf4000000faff0398be12300008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'func4': bytes.fromhex(
        '04040cf4000000faff0398be12400008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'func5': bytes.fromhex(
        '04040cf4000000faff0398be12500008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'func6': bytes.fromhex(
        '04040cf4000000faff0398be12600008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'func7': bytes.fromhex(
        '04040cf4000000faff0398be12700008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'bilinear': bytes.fromhex(
        '04040cf4000000faff0398be12100000800399beff039abe0000500080039bbe'
        'f002007eff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400'
        '703f8cbf0f1800f804040404000081bf'),
    'zero_sampler': bytes.fromhex(
        '04040cf4000000fa800398be800399be80039abe80039bbef002007eff02027e'
        '0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf0f1800f8'
        '04040404000081bf'),
    'point_no_trunc': bytes.fromhex(
        '04040cf4000000faff0398be12100000800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'anisotropic': bytes.fromhex(
        '04040cf4000000faff0398be12120008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'wrap_x': bytes.fromhex(
        '04040cf4000000faff0398be10100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801bcf00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'mixed': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf08019cf00005c4000801bcf0'
        '0004c400703f8cbf0f1800f804040404000081bf'),
    'c_max0': bytes.fromhex(
        '04040cf4000000faff0398be12100008800399be80039abe80039bbef002007e'
        'ff02027e0000803ef202047ef402067e7fc08cbf0801a0f00004c400703f8cbf'
        '0f1800f804040404000081bf'),
    'c_max_partial': bytes.fromhex(
        '04040cf4000000faff0398be12100008ff0399be0010000080039abe80039bbe'
        'f002007eff02027e0000803ef202047ef402067e7fc08cbf0801a0f00004c400'
        '703f8cbf0f1800f804040404000081bf'),
}


def source(name):
    key, w0, w2 = VARIANTS[name]
    if key == 'mixed':
        # One sampler used by an ordinary and a comparison sample: two bindings, never one SamplerState.
        body = ('image_sample_lz v5, v[0:1], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D\n'
                + IMAGE['lz'] + '\n')
    else:
        body = IMAGE[key] + '\n'
    return PRE.format(w0=w0, w1=W1.get(name, 0), w2=w2) + body + POST


# Reserved gfx10 MIMG bits: llvm-mc cannot assemble them, so each fixture sets one in the plain instruction.
RESERVED = {'reserved_w0_6': (1 << 6, 0), 'reserved_w0_14': (1 << 14, 0)}
RESERVED.update({'reserved_w1_%d' % bit: (0, 1 << bit) for bit in range(26, 30)})
PLAIN_INSTRUCTION = bytes.fromhex('0801bcf00004c400')


def reserved_words(name):
    w0, w1 = struct.unpack('<II', PLAIN_INSTRUCTION)
    patched = struct.pack('<II', w0 | RESERVED[name][0], w1 | RESERVED[name][1])
    assert WORDS['lz'].count(PLAIN_INSTRUCTION) == 1
    return WORDS['lz'].replace(PLAIN_INSTRUCTION, patched)


def pipeline(name):
    words = reserved_words(name) if name in RESERVED else WORDS[name]
    return G.Pipeline(VS_WORDS, words, root_blob(flags=0))


def refused(name, text):
    try:
        pipeline(name).hlsl_ps()
    except G.Unsupported as error:
        assert text in str(error), (name, str(error))
    else:
        raise AssertionError('%s was accepted' % name)


ARRAY_CALL = re.compile(r'v4 = U\((r\d+)\.(SampleCmp(?:LevelZero)?)\((q\d+), float3\(F\((v\d+)\), F\((v\d+)\), F\((v\d+)\)\), F\((v\d+)\)\)\);')
CALL = re.compile(r'v4 = U\((r\d+)\.(SampleCmp(?:LevelZero)?)\((q\d+), float2\(F\((v\d+)\), F\((v\d+)\)\), F\((v\d+)\)\)\);')


def lowered(ps):
    """(texture, method, sampler, u, v, reference) of the one comparison call in a pixel stage."""
    calls = CALL.findall(ps)
    assert len(calls) == 1, ps
    texture, method, sampler, u, v, reference = calls[0]
    return method, u, v, reference, sampler, texture


def check_call(ps, method, u, v, reference, declaration='SamplerComparisonState q0 : register(s0);'):
    """The exact comparison call and sampler type a lowering must produce."""
    got = lowered(ps)
    assert got[:4] == (method, u, v, reference), got
    assert declaration in ps, ps


def check_array_call(ps, u, v, slice_, reference, declaration='Texture2DArray<float4> r0 : register(t0);'):
    """The exact Texture2DArray comparison a 2D-array lowering must produce: the slice is the third location component."""
    calls = ARRAY_CALL.findall(ps)
    assert len(calls) == 1, ps
    _texture, method, _sampler, got_u, got_v, got_slice, got_reference = calls[0]
    assert (method, got_u, got_v, got_slice, got_reference) == ('SampleCmpLevelZero', u, v, slice_, reference), calls
    assert declaration in ps and 'SamplerComparisonState q0 : register(s0);' in ps, ps


def manifest_samplers(pipe):
    return [line for line in pipe.manifest().splitlines() if line.startswith(('smp ', 'smpc '))]


def controls():
    lz = pipeline('lz')
    check_call(lz.hlsl_ps(), 'SampleCmpLevelZero', 'v1', 'v2', 'v0')  # contiguous: reference first, then u, v
    check_call(pipeline('lz_nsa').hlsl_ps(), 'SampleCmpLevelZero', 'v0', 'v1', 'v3')  # NSA: v3 is the reference
    # Implicit mip selection is not lowered: the comparison sampler state ignores the mip filter, so any source that
    # does not limit the LOD to zero is refused. MAX_LOD 0 is exactly level zero.
    check_call(pipeline('c_max0').hlsl_ps(), 'SampleCmpLevelZero', 'v1', 'v2', 'v0')
    refused('c', 'implicit mip selection')
    refused('c_nsa', 'implicit mip selection')
    refused('c_max_partial', 'implicit mip selection')
    assert 'Texture2D<float4> r0' in lz.hlsl_ps() and 'srv 0 tex2d table 0 0' in lz.manifest()
    assert manifest_samplers(lz) == ['smpc 0 0x%x 0x0 0x0 0x0' % (POINT | LESS)], manifest_samplers(lz)
    # A 2D array adds the slice as the last address register and binds a Texture2DArray (manifest kind tex2da).
    array = pipeline('array')
    check_array_call(array.hlsl_ps(), 'v1', 'v2', 'v3', 'v0')
    assert 'srv 0 tex2da table 0 0' in array.manifest(), array.manifest()
    assert manifest_samplers(array) == ['smpc 0 0x%x 0x0 0x0 0x0' % (POINT | LESS)], manifest_samplers(array)
    check_array_call(pipeline('array_nsa').hlsl_ps(), 'v0', 'v1', 'v2', 'v3')  # NSA: reference v3, slice v2
    check_array_call(pipeline('array_c').hlsl_ps(), 'v1', 'v2', 'v3', 'v0')
    refused('array_c_full', 'implicit mip selection')
    refused('array_c_l', 'shader model 6.7')
    print('array: reference v0 at (v1, v2, slice v3); NSA reference v3, slice v2; Texture2DArray bound; implicit c needs MAX_LOD 0')
    print('lz: reference v0 at (v1, v2); NSA reference v3 at (v0, v1); implicit c refused; sampler smpc')

    # The comparison function travels in word 0 bits 14:12, one function per fixture.
    for n in range(8):
        name = 'lz' if n == 1 else 'func%d' % n
        assert manifest_samplers(pipeline(name)) == ['smpc 0 0x%x 0x0 0x0 0x0' % (POINT | n << 12)], name
        check_call(pipeline(name).hlsl_ps(), 'SampleCmpLevelZero', 'v1', 'v2', 'v0')
    assert manifest_samplers(pipeline('bilinear')) == ['smpc 0 0x%x 0x0 0x%x 0x0' % (2 | 2 << 3 | LESS, BILINEAR_W2)]
    print('all eight comparison functions and the bilinear filter reach the manifest')

    # The same words used by image_sample_lz and image_sample_c_lz are two bindings of different types.
    mixed = pipeline('mixed')
    text = mixed.hlsl_ps()
    assert 'SamplerState q0 : register(s0);' in text and 'SamplerComparisonState q1 : register(s1);' in text, text
    assert 'SampleLevel(q0,' in text and 'SampleCmpLevelZero(q1,' in text, text
    assert manifest_samplers(mixed) == ['smp 0 0x%x 0x0 0x0 0x0' % (POINT | LESS),
                                        'smpc 1 0x%x 0x0 0x0 0x0' % (POINT | LESS)], manifest_samplers(mixed)
    print('one sampler used both ways: smp 0 as SamplerState, smpc 1 as SamplerComparisonState')

    # Cache-policy bits do not change the compared value.
    for name in ('glc', 'slc', 'dlc'):
        assert pipeline(name).hlsl_ps() == lz.hlsl_ps(), name

    refused('c_l', 'image_sample_c_l')  # SampleCmpLevel is shader model 6.7
    refused('c_l_nsa', 'shader model 6.7')
    refused('dmask3', 'dmask 0x7')
    refused('dmask2', 'dmask 0x2')
    for name, dim in (('three', '3D'), ('cube', 'CUBE'), ('dim1d', '1D')):
        refused(name, 'image_sample_c_lz on SQ_RSRC_IMG_' + dim)
    refused('array_offset', 'image_sample_c_lz_o')
    refused('array_dmask3', 'dmask 0x7')
    refused('array_unorm', 'image_sample_c_lz modifier unorm')
    refused('offset', 'image_sample_c_lz_o')
    refused('bias', 'image_sample_c_b')
    refused('clamp', 'image_sample_c_cl')
    refused('gather', 'image_gather4_c')
    for name in ('a16', 'tfe', 'unorm', 'lwe', 'd16', 'r128'):
        refused(name, 'image_sample_c_lz modifier %s' % name)
    for name in RESERVED:
        refused(name, 'image_sample_c_lz modifier reserved')
    refused('zero_sampler', 'sampler: Comparison sampling requires CLAMP_LAST_TEXEL')
    refused('wrap_x', 'sampler: Comparison sampling requires CLAMP_LAST_TEXEL')
    refused('anisotropic', 'sampler: Comparison sampling with anisotropy')
    refused('point_no_trunc', 'sampler: Comparison sampling with point filtering requires TRUNC_COORD 1')
    print('refused: c_l, extra dmask channels, 3D/cube/1D, offset/bias/clamp/gather, '
          'a16/tfe/unorm/lwe/d16/r128, reserved bits, samplers outside the checked profile')


def negative_controls():
    """Wrong lowerings, made by editing the correct text, must fail the exact checks above.

    Every mutant keeps the registers the correct fixture reads, so only the edited call differs.
    """
    good = pipeline('lz').hlsl_ps()
    nsa = pipeline('lz_nsa').hlsl_ps()
    exact = ('SampleCmpLevelZero', 'v1', 'v2', 'v0')
    mutants = (
        ('reference read from the last address register', good.replace('F(v2)), F(v0))', 'F(v1)), F(v2))'), exact),
        ('reference swapped with u', good.replace('float2(F(v1), F(v2)), F(v0)', 'float2(F(v0), F(v2)), F(v1)'), exact),
        ('NSA reference taken as the first listed register', nsa.replace('F(v3))', 'F(v1))'),
         ('SampleCmpLevelZero', 'v0', 'v1', 'v3')),
        ('ordinary SamplerState instead of SamplerComparisonState',
         good.replace('SamplerComparisonState', 'SamplerState'), exact),
        ('SampleCmp instead of SampleCmpLevelZero', good.replace('SampleCmpLevelZero', 'SampleCmp'), exact),
        ('SampleLevel instead of a comparison',
         good.replace('SampleCmpLevelZero(q0, float2(F(v1), F(v2)), F(v0))', 'SampleLevel(q0, float2(F(v1), F(v2)), 0).x'),
         exact),
    )
    for label, text, want in mutants:
        try:
            check_call(text, *want)
        except AssertionError:
            print('mutant fails the exact call check:', label)
        else:
            raise AssertionError('mutant passed: ' + label)
    arr = pipeline('array').hlsl_ps()
    arr_want = ('v1', 'v2', 'v3', 'v0')
    array_mutants = (
        ('slice dropped (a 2D location)', arr.replace('float3(F(v1), F(v2), F(v3))', 'float2(F(v1), F(v2))')),
        ('slice read from the reference register', arr.replace('F(v2), F(v3))', 'F(v2), F(v0))')),
        ('slice read from v2 instead of the fourth register', arr.replace('F(v2), F(v3))', 'F(v2), F(v2))')),
        ('slice and v exchanged', arr.replace('F(v1), F(v2), F(v3)', 'F(v1), F(v3), F(v2)')),
        ('reference swapped with u', arr.replace('F(v1), F(v2), F(v3)), F(v0)', 'F(v0), F(v2), F(v3)), F(v1)')),
        ('plain Texture2D declared', arr.replace('Texture2DArray<float4>', 'Texture2D<float4>')),
        ('SampleCmp instead of SampleCmpLevelZero', arr.replace('SampleCmpLevelZero', 'SampleCmp')),
    )
    for label, text in array_mutants:
        assert text != arr, label
        try:
            check_array_call(text, *arr_want)
        except AssertionError:
            print('array mutant fails the exact call check:', label)
        else:
            raise AssertionError('array mutant passed: ' + label)
    # A wrong comparison function is a different manifest word: each function has its own smpc line.
    lines = [manifest_samplers(pipeline('lz' if n == 1 else 'func%d' % n)) for n in range(8)]
    assert all(lines[n] != lines[m] for n in range(8) for m in range(8) if n != m), lines
    print('each of the eight comparison functions has its own smpc line')


def check_binary(llvm_mc):
    for name in WORDS:
        built = assemble(llvm_mc, source(name))
        if built != WORDS[name]:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), WORDS[name].hex()))
    print('fixture words match llvm-mc')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
        negative_controls()
    elif sys.argv[1] == 'binary' and len(sys.argv) == 3:
        check_binary(sys.argv[2])
    else:
        sys.exit(__doc__)
