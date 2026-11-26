"""Biased implicit-LOD image_sample_b in tools/startup/gfx2hlsl.py (pixel stage only; M200-01 B).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble every fixture source and compare it with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given) and draw:
                                       the readback must name the mip level the bias selected

Semantics. The RDNA 2 ISA (section 8.2.4, Table 43) says "_B LOD BIAS: Add this BIAS to the LOD computed", with one
extra address dword, and section 8.2.5 orders the address as "{ offset } { bias } { z-compare } { derivative } { body }",
the bias being "One Dword float": the bias is the FIRST address register and the coordinates follow it. LLVM 15's
IntrinsicsAMDGPU.td lists the extra argument of AMDGPUSample_b as llvm_anyfloat_ty "bias" and MIMGInstructions.td gives
IMAGE_SAMPLE_B the gfx10 opcode 0x25. HLSL Texture.SampleBias(sampler, location, bias) adds the bias to the
implicit mip level, so the lowering is tex.SampleBias(smp, uv, bias). D3D documents the bias range as -16.0 to 15.99;
the ISA text states no hardware clamp, so values outside that range are an unverified difference. Accepted under the
entry EXEC mask and wave-uniform control flow even after a uniform (scc) join or later loop; an EXEC written per lane
(v_cmpx, saveexec, a cleared EXEC) is followed exactly by the one-lane model, so it is accepted in a program without a
backward branch (tests/startup_image_sample_b_flow_check.py draws it); divergent EXEC or a
vcc/exec-conditioned branch before the sample stays refused.

Each fixture is a source-built gfx1013 pixel stage: v0 = bias, v1 = u = 0.75, v2 = v = 0.25 and v3 = the slice (root
constant 1, s9). Constant coordinates have no derivatives (the implicit LOD would be log2(0) and the bias is lost), so
the draw rewrites only the u initialiser of the translated HLSL to 0.75 + 0.5 * (SV_Position.x - 0.5): the pixel
still samples u = 0.75, but u changes by 0.5 per pixel, a footprint of two texels on the 4x4 base level, an implicit
LOD of exactly 1. The call under test, SampleBias(..., bias), is not touched. The draw samples the 4x4 two-slice,
three-mip array of tests/startup_image_3d.cpp (--startup-image-2d-array-mips) with a point sampler whose descriptor
sets a mip filter of "point" (word 2 bits 27:26 = 1; a filter of 0 would disable mipmapping). Bias -1 then
reads mip 0, bias 0 mip 1 and bias +1 mip 2, and a sample without bias reads mip 1. Plain 2D, 3D and NSA forms are
checked on the generated HLSL only.

Sampler limits. The descriptor's MIN_LOD (word 1 bits 11:0) and MAX_LOD (bits 23:12) clamp the biased LOD and are
unsigned 4.8 (Mesa 24.0.0 src/amd/registers/gfx10-rsrc.json, SQ_IMG_SAMP_WORD1; radeonsi si_state.c:5169-5170). The
translated sampler carries them to MinLOD and MaxLOD (manifest line smpl). The fixtures that discriminate are the ones
whose limit is BELOW the mip the unclamped LOD reaches while that mip exists: MAX_LOD 1 with bias +1 must stay on mip 1
(unclamped mip 2 is visible, with its own colour), MIN_LOD 2 with bias -1 must read mip 2 (unclamped mip 0). The
MAX_LOD 2 cases cannot tell, since mip 2 is the last level. A descriptor LOD bias (word 2 bits 19:0), mip filter 0,
MIN_LOD above MAX_LOD and every sampler that is not a constant (sampler heap, root constant) are refused.
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_sampler_heap_check as H  # noqa: E402
from startup_image_2d_array_check import mip_cell  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, assemble, compile_hlsl, root_blob  # noqa: E402
import gpu_case_runner  # noqa: E402

# Sampler words: point filters, wrap, MAX_LOD 2.0 (u4.8 at word 1 bits 23:12), mip filter point (word 2 bits 27:26).
SAMPLER = ('0', '0x200000', '0x4000000')
PROGRAM = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, {w0}
s_mov_b32 s25, {w1}
s_mov_b32 s26, {w2}
s_mov_b32 s27, 0
v_mov_b32 v0, {bias}
v_mov_b32 v1, 0x3f400000
v_mov_b32 v2, 0x3e800000
v_mov_b32 v3, s9
s_waitcnt lgkmcnt(0)
{instruction}
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
# The sampler words come from a run-time load of the sampler heap at the byte offset in root constant 1 (s9).
HEAP_PROGRAM = """s_load_dwordx8 s[16:23], s[8:9], 0x0
s_load_dwordx4 s[24:27], s[10:11], s9
v_mov_b32 v0, 1.0
v_mov_b32 v1, 0x3f400000
v_mov_b32 v2, 0x3e800000
v_mov_b32 v3, 0
s_waitcnt lgkmcnt(0)
{instruction}
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
TAIL = 's[16:23], s[24:27] dmask:0xf dim:'
ARRAY = 'image_sample_b v[4:7], v[0:3], ' + TAIL + 'SQ_RSRC_IMG_2D_ARRAY'
# name: (instruction, bias literal, sampler words)
FIXTURES = {
    'array 0': (ARRAY, '0', SAMPLER),
    'array 1': (ARRAY, '1.0', SAMPLER),
    'array -1': (ARRAY, '-1.0', SAMPLER),
    # The same program without a bias: the address is u, v, slice.
    'array nobias': ('image_sample v[4:7], v[1:3], ' + TAIL + 'SQ_RSRC_IMG_2D_ARRAY', '1.0', SAMPLER),
    'plain': ('image_sample_b v[4:7], v[0:2], ' + TAIL + 'SQ_RSRC_IMG_2D', '1.0', SAMPLER),
    'nsa': ('image_sample_b v[4:7], [v3, v1, v2], ' + TAIL + 'SQ_RSRC_IMG_2D', '1.0', SAMPLER),
    'nsa array': ('image_sample_b v[4:7], [v3, v1, v2, v0], ' + TAIL + 'SQ_RSRC_IMG_2D_ARRAY', '1.0', SAMPLER),
    'three': ('image_sample_b v[4:7], v[0:3], ' + TAIL + 'SQ_RSRC_IMG_3D', '1.0', SAMPLER),
    'dmask3': ('image_sample_b v[4:6], v[0:2], s[16:23], s[24:27] dmask:0x7 dim:SQ_RSRC_IMG_2D', '1.0', SAMPLER),
    'glc': ('image_sample_b v[4:7], v[0:2], ' + TAIL + 'SQ_RSRC_IMG_2D glc', '1.0', SAMPLER),
    'slc': ('image_sample_b v[4:7], v[0:2], ' + TAIL + 'SQ_RSRC_IMG_2D slc', '1.0', SAMPLER),
    'dlc': ('image_sample_b v[4:7], v[0:2], ' + TAIL + 'SQ_RSRC_IMG_2D dlc', '1.0', SAMPLER),
    # Refused forms.
    'b_cl': ('image_sample_b_cl v[4:7], v[0:3], ' + TAIL + 'SQ_RSRC_IMG_2D', '1.0', SAMPLER),
    'c_b': ('image_sample_c_b v4, v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D', '1.0', SAMPLER),
    'offset': ('image_sample_b_o v[4:7], v[0:3], ' + TAIL + 'SQ_RSRC_IMG_2D', '1.0', SAMPLER),
    'gather': ('image_gather4_b v[4:7], v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D', '1.0', SAMPLER),
    'a16': ('image_sample_b v[4:7], v[0:1], ' + TAIL + 'SQ_RSRC_IMG_2D a16', '1.0', SAMPLER),
    'd16': ('image_sample_b v[4:5], v[0:2], ' + TAIL + 'SQ_RSRC_IMG_2D d16', '1.0', SAMPLER),
    'tfe': ('image_sample_b v[4:8], v[0:2], ' + TAIL + 'SQ_RSRC_IMG_2D tfe', '1.0', SAMPLER),
    'lwe': ('image_sample_b v[4:7], v[0:2], ' + TAIL + 'SQ_RSRC_IMG_2D lwe', '1.0', SAMPLER),
    'unorm': ('image_sample_b v[4:7], v[0:2], ' + TAIL + 'SQ_RSRC_IMG_2D unorm', '1.0', SAMPLER),
    'r128': ('image_sample_b v[4:7], v[0:2], ' + TAIL + 'SQ_RSRC_IMG_2D r128', '1.0', SAMPLER),
    'dmask0': ('image_sample_b v4, v[0:2], s[16:23], s[24:27] dmask:0x0 dim:SQ_RSRC_IMG_2D', '1.0', SAMPLER),
    'dim1d': ('image_sample_b v[4:7], v[0:1], ' + TAIL + 'SQ_RSRC_IMG_1D', '1.0', SAMPLER),
    'cube': ('image_sample_b v[4:7], v[0:3], ' + TAIL + 'SQ_RSRC_IMG_CUBE', '1.0', SAMPLER),
    'msaa': ('image_sample_b v[4:7], v[0:3], ' + TAIL + 'SQ_RSRC_IMG_2D_MSAA', '1.0', SAMPLER),
    # The sampler is loaded from the sampler heap at a run-time offset (no constant words): see HEAP_FIXTURES.
    'sampler heap': (ARRAY, None, None),
    # The guest LOD limits: MAX_LOD 1.0 (0x100 << 12) and MIN_LOD 2.0 (0x200 in bits 11:0), both with MAX_LOD <= 2.
    'array max1': (ARRAY, '1.0', ('0', '0x100000', '0x4000000')),
    'array max0': (ARRAY, '1.0', ('0', '0', '0x4000000')),
    'array min2': (ARRAY, '-1.0', ('0', '0x200200', '0x4000000')),
    'array min1 max1': (ARRAY, '-1.0', ('0', '0x100100', '0x4000000')),
    'array min3 max2': (ARRAY, '-1.0', ('0', '0x200300', '0x4000000')),
    'mip filter none': (ARRAY, '1.0', ('0', '0x200000', '0')),
    'sampler from root constant': (ARRAY, '1.0', ('s9', '0x200000', '0x4000000')),
    # A descriptor LOD bias (word 2 bits 13:0) or secondary LOD bias (bits 19:14) is not carried to the static sampler.
    'sampler bias': (ARRAY, '1.0', ('0', '0x200000', '0x4000001')),
    'sampler bias sec': (ARRAY, '1.0', ('0', '0x200000', '0x4004000')),
}
# Reserved gfx10 MIMG bits: llvm-mc cannot assemble them, so each fixture sets one in the plain instruction.
RESERVED = {'reserved_w0_6': (1 << 6, 0), 'reserved_w0_14': (1 << 14, 0)}
RESERVED.update({'reserved_w1_%d' % bit: (0, 1 << bit) for bit in range(26, 30)})

# BEGIN WORDS
WORDS = {
    'array 0': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        '8002007eff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'array 1': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'array -1': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f302007eff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'array nobias': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf280f80f0'
        '0104c400703f8cbf0f1800f804050607000081bf'
    ),
    'plain': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf080f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'nsa': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf0a0f94f0'
        '0304c40001020000703f8cbf0f1800f804050607000081bf'
    ),
    'nsa array': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf2a0f94f0'
        '0304c40001020000703f8cbf0f1800f804050607000081bf'
    ),
    'three': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf100f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'dmask3': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf080794f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'glc': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf082f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'slc': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf080f94f2'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'dlc': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf880f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'b_cl': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf080f98f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'c_b': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf0801b4f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'offset': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf080fd4f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'gather': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf080114f1'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'a16': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf080f94f0'
        '0004c440703f8cbf0f1800f804050607000081bf'
    ),
    'd16': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf080f94f0'
        '0004c480703f8cbf0f1800f804050607000081bf'
    ),
    'tfe': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf080f95f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'lwe': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf080f96f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'unorm': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf081f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'r128': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf088f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'dmask0': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf080094f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'dim1d': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf000f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'cube': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf180f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'msaa': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf300f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'sampler bias': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0100000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'sampler bias sec': (
        '04040cf4000000fa800398beff0399be00002000ff039abe0040000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'sampler heap': (
        '04040cf4000000fa050608f400000012f202007eff02027e0000403fff02047e'
        '0000803e8002067e7fc08cbf280f94f00004c400703f8cbf0f1800f804050607'
        '000081bf'
    ),
    'array max1': (
        '04040cf4000000fa800398beff0399be00001000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'array max0': (
        '04040cf4000000fa800398be800399beff039abe0000000480039bbef202007e'
        'ff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f00004c400'
        '703f8cbf0f1800f804050607000081bf'
    ),
    'array min2': (
        '04040cf4000000fa800398beff0399be00022000ff039abe0000000480039bbe'
        'f302007eff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'array min1 max1': (
        '04040cf4000000fa800398beff0399be00011000ff039abe0000000480039bbe'
        'f302007eff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'array min3 max2': (
        '04040cf4000000fa800398beff0399be00032000ff039abe0000000480039bbe'
        'f302007eff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'mip filter none': (
        '04040cf4000000fa800398beff0399be0000200080039abe80039bbef202007e'
        'ff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f00004c400'
        '703f8cbf0f1800f804050607000081bf'
    ),
    'sampler from root constant': (
        '04040cf4000000fa090398beff0399be00002000ff039abe0000000480039bbe'
        'f202007eff02027e0000403fff02047e0000803e0902067e7fc08cbf280f94f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
}
# END WORDS


def source(name):
    instruction, bias, sampler = FIXTURES[name]
    if sampler is None:
        return HEAP_PROGRAM.format(instruction=instruction)
    w0, w1, w2 = sampler
    return PROGRAM.format(instruction=instruction, bias=bias, w0=w0, w1=w1, w2=w2)


def words(name):
    if name in RESERVED:
        plain = bytes.fromhex(WORDS['plain'])
        instruction = struct.pack('<II', *(a | b for a, b in zip(struct.unpack('<II', plain[PLAIN_AT:PLAIN_AT + 8]),
                                                                   RESERVED[name])))
        return plain[:PLAIN_AT] + instruction + plain[PLAIN_AT + 8:]
    return bytes.fromhex(WORDS[name])


PLAIN_AT = None


def locate():
    """The 8-byte MIMG word pair of the plain fixture: it starts with the dim/dmask byte pair the opcode carries."""
    global PLAIN_AT
    data = bytes.fromhex(WORDS['plain'])
    # image_sample_b dim:2D dmask:0xf: 08 0f 94 f0 (word 0 bytes, little endian).
    PLAIN_AT = data.index(bytes.fromhex('080f94f0'))
    assert data.count(bytes.fromhex('080f94f0')) == 1


def stage(name, sampler_heap=None):
    if FIXTURES.get(name, (0, 0, 0))[2] is None:
        return G.Pipeline(H.VS_WORDS, words(name), H.root_blob(), sampler_heap=sampler_heap)
    return G.Pipeline(VS_WORDS, words(name), root_blob())


def refused(name, text):
    try:
        stage(name).hlsl_ps()
    except G.Unsupported as error:
        assert text in str(error), (name, str(error))
    else:
        raise AssertionError('%s was accepted' % name)


def bias_call(ps, kind):
    """(texture declaration, call arguments) of the one SampleBias call in the translated stage."""
    m = re.search(r'(\w+)\.SampleBias\((\w+), (float\d)\((.*?)\), (F\(v\d+\))\);', ps)
    assert m, ps
    assert m.group(3) == kind, (m.group(3), kind)
    coords = re.findall(r'F\((v\d+)\)', m.group(4))
    bias = re.fullmatch(r'F\((v\d+)\)', m.group(5))
    assert bias, m.group(5)
    return coords, bias.group(1)


BASE_LOD = 1  # the implicit LOD of the draw (see the module docstring)


def mip_for(bias, min_lod=0, max_lod=2):
    """The mip the point-mip sampler reads: the biased LOD, clamped to the descriptor's MIN_LOD and MAX_LOD, and to
    the last of the three levels."""
    return max(0, min(2, max(min_lod, min(max_lod, BASE_LOD + int(bias)))))


def texel_cell(mip, slice_index):
    """Cell the point sampler reads at (0.75, 0.25) on mip level `mip`."""
    size = 4 >> mip
    return mip_cell(int(0.75 * size), int(0.25 * size), slice_index, mip)


def check_guards():
    """The stage guard and the address-count guard are decided before any state is read, so a bare stage reaches them."""
    def call(coords, kind='ps', exec_known=True, blocks=(), uniform_cf=True):
        bare = G.Stage.__new__(G.Stage)
        bare.kind, bare.exec_known, bare.blocks = kind, exec_known, list(blocks)
        bare.uniform_cf = uniform_cf
        bare.image(['v5', coords, 's0'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0x1'], 'b')

    for coords, count in (('[v1, v2]', 2), ('[v1, v2, v3, v4]', 4)):
        try:
            call(coords)
        except G.Unsupported as error:
            assert str(error) == 'image_sample_b with %d address registers' % count, (coords, error)
        else:
            raise AssertionError('image_sample_b accepted %d address registers' % count)
    for label, kwargs in (('vertex stage', {'kind': 'vs'}), ('compute stage', {'kind': 'cs'}),
                          ('exec modified', {'exec_known': False}),
                          ('divergent CF', {'uniform_cf': False})):
        try:
            call('[v1, v2, v3]', **kwargs)
        except G.Unsupported as error:
            assert 'image_sample_b needs implicit derivatives' in str(error), (label, error)
        else:
            raise AssertionError('image_sample_b accepted: %s' % label)
    # A prior join (non-empty blocks) is fine while EXEC and wave-uniform CF hold (M200-01 B).
    try:
        call('[v1, v2, v3]', blocks=[object()], uniform_cf=True)
    except G.Unsupported as error:
        raise AssertionError('image_sample_b refused with entry EXEC after a uniform join: %s' % error)
    except AttributeError:
        pass
    else:
        raise AssertionError('image_sample_b after a join returned without needing a pipeline')
    # The legal case gets past both guards: the bare stage then stops where it needs a pipeline.
    try:
        call('[v1, v2, v3]')
    except G.Unsupported as error:
        raise AssertionError('legal image_sample_b refused: %s' % error)
    except AttributeError:
        pass
    else:
        raise AssertionError('legal image_sample_b returned without needing a pipeline')
    print('refused: wrong address count, vertex and compute stages, modified exec, divergent CF; '
          'join with entry EXEC and uniform CF accepted')


def sopp(op, simm):
    return struct.pack('<I', (simm & 0xffff) | op << 16 | 0x17f << 23)


def with_join_before_sample(data):
    """Uniform forward join (every lane takes the same path) immediately before the sample (sample_b or plain)."""
    off = next(o for o, _s, _n, a in G.decode(data) if a.startswith('image_sample'))
    join = (bytes.fromhex('800380be') + bytes.fromhex('008006bf') + sopp(5, 1) + bytes.fromhex('810381be'))
    return data[:off] + join + data[off:]


def with_divergent_exec_before_sample(data, word='8004febe'):
    """Clear EXEC after the loads so the entry-mask proof is gone before the sample (s_mov_b64 exec, 0 by default;
    8003febe is s_mov_b32 exec_lo, 0)."""
    off = next(o for o, _s, _n, a in G.decode(data) if 'image_sample_b' in a)
    return data[:off] + bytes.fromhex(word) + data[off:]


def with_branch_before_sample(data, sopp_op):
    """Insert s_mov; s_cbranch_<op> +1; s_mov immediately before the sample_b (review probe)."""
    off = next(o for o, _s, _n, a in G.decode(data) if 'image_sample_b' in a)
    insert = bytes.fromhex('800380be') + sopp(sopp_op, 1) + bytes.fromhex('810381be')
    return data[:off] + insert + data[off:]


def with_divergent_back_edge_after_sample(data, sopp_op):
    """Insert s_cbranch_<op> right after the sample_b, jumping back to it: the branch comes later in program order, but
    the sample runs again on the next pass, with lanes that left the loop at different passes."""
    off, size = next((o, s) for o, s, _n, a in G.decode(data) if 'image_sample_b' in a)
    return data[:off + size] + sopp(sopp_op, -(size // 4 + 1)) + data[off + size:]


def controls():
    assert WORDS, 'no recorded words'
    locate()
    stage('array 1').hlsl_ps()  # the plain translation first: a translator without the lowering stops here
    check_guards()
    plain = bytes.fromhex(WORDS['plain'])
    join_ps = G.Pipeline(VS_WORDS, with_join_before_sample(plain), root_blob()).hlsl_ps()
    assert 'SampleBias' in join_ps and bias_call(join_ps, 'float2') == (['v1', 'v2'], 'v0'), join_ps
    for label, mutate in (
            ('s_cbranch_vccz', lambda d: with_branch_before_sample(d, 6)),
            ('s_cbranch_vccnz', lambda d: with_branch_before_sample(d, 7)),
            ('s_cbranch_execz', lambda d: with_branch_before_sample(d, 8)),
            ('s_cbranch_execnz', lambda d: with_branch_before_sample(d, 9)),
            ('s_cbranch_vccnz back edge after the sample', lambda d: with_divergent_back_edge_after_sample(d, 7)),
            ('s_cbranch_execnz back edge after the sample', lambda d: with_divergent_back_edge_after_sample(d, 9))):
        try:
            G.Pipeline(VS_WORDS, mutate(plain), root_blob()).hlsl_ps()
        except G.Unsupported as error:
            assert 'image_sample_b needs implicit derivatives' in str(error), (label, error)
        else:
            raise AssertionError('image_sample_b accepted after %s' % label)
    # A 64-bit cleared EXEC is an exact per-lane write: the quad program (no backward branch) keeps the proof. The
    # 32-bit form keeps lanes 32..63 of the old mask in wave64, so it is refused.
    try:
        G.Pipeline(VS_WORDS, with_divergent_exec_before_sample(plain, '8003febe'), root_blob()).hlsl_ps()
    except G.Unsupported as error:
        assert 'image_sample_b needs implicit derivatives' in str(error), error
    else:
        raise AssertionError('image_sample_b accepted after s_mov_b32 exec_lo, 0')
    cleared = G.Pipeline(VS_WORDS, with_divergent_exec_before_sample(plain), root_blob()).hlsl_ps()
    assert 'SampleBias' in cleared and 'exec = (0x00000000u) & 1;' in cleared, cleared
    print('uniform join before sample_b accepted; divergent EXEC and vcc/exec branches before sample_b refused')
    # Address order: the bias register comes first and is the third SampleBias argument.
    expected = {
        'array 1': (['v1', 'v2', 'v3'], 'v0', 'float3', 'Texture2DArray<float4> r0'),
        'plain': (['v1', 'v2'], 'v0', 'float2', 'Texture2D<float4> r0'),
        'nsa': (['v1', 'v2'], 'v3', 'float2', 'Texture2D<float4> r0'),  # the bias is the first listed register
        'nsa array': (['v1', 'v2', 'v0'], 'v3', 'float3', 'Texture2DArray<float4> r0'),
        'three': (['v1', 'v2', 'v3'], 'v0', 'float3', 'Texture3D<float4> r0'),
    }
    for name, (coords, bias, kind, declaration) in expected.items():
        pipe = stage(name)
        ps = pipe.hlsl_ps()
        assert bias_call(ps, kind) == (coords, bias), (name, bias_call(ps, kind))
        assert declaration in ps and 'Sample(' not in ps and not pipe.notes, (name, ps)
    print('bias is the first address register: plain [v0 | v1 v2], NSA [v3 | v1 v2], array [v0 | v1 v2 v3], '
          'NSA array [v3 | v1 v2 v0], 3D [v0 | v1 v2 v3]')
    ps = stage('array 1').hlsl_ps()
    manifest = stage('array 1').manifest()
    assert 'srv 0 tex2da table 0 0' in manifest and 'smpl 0 0x0 0x200000 0x4000000 0x0' in manifest, manifest
    assert 'udsrc 1 rootconst 1 0' in manifest and re.search(r'\bv3 = ud1;', ps), (manifest, ps)

    # Negative control: without the bias the same program is a plain implicit sample on another address.
    nobias = stage('array nobias').hlsl_ps()
    assert 'SampleBias' not in nobias and 'r0.Sample(q0, float3(F(v1), F(v2), F(v3)))' in nobias, nobias
    assert nobias != ps
    print('image_sample (no bias) lowers to Sample, not SampleBias')

    refused('b_cl', 'image_sample_b_cl')      # the extra LOD clamp dword is not modelled
    refused('c_b', 'image_sample_c_b')
    refused('offset', 'image_sample_b_o')
    refused('gather', 'image_gather4_b')
    refused('cube', 'image dimension SQ_RSRC_IMG_CUBE')
    refused('msaa', 'image dimension SQ_RSRC_IMG_')
    refused('dim1d', 'unverified 1D image b address or modifiers')
    refused('unorm', 'image_sample_b modifier unorm')
    refused('tfe', 'image_sample_b modifier tfe')
    refused('a16', 'image_sample_b modifier a16')
    refused('lwe', 'image_sample_b modifier lwe')
    refused('d16', 'image_sample_b modifier d16')
    refused('r128', 'image_sample_b modifier r128')
    refused('dmask0', 'empty dmask')
    for name in RESERVED:
        refused(name, 'image_sample_b modifier reserved')
    refused('sampler bias', 'image_sample_b with a sampler LOD bias (word 2 0x4000001)')
    refused('sampler bias sec', 'image_sample_b with a sampler LOD bias (word 2 0x4004000)')
    print('refused: b_cl, c_b, offset, gather4_b, cube, MSAA, 1D, a16, d16, tfe, lwe, unorm, r128, dmask 0, '
          'reserved bits, sampler LOD bias')

    # Guest sampler limits: carried over for a constant descriptor, refused where they cannot be shown irrelevant.
    for name, line in (('array max1', 'smpl 0 0x0 0x100000 0x4000000 0x0'), ('array max0', 'smpl 0 0x0 0x0 0x4000000 0x0'),
                       ('array min2', 'smpl 0 0x0 0x200200 0x4000000 0x0'),
                       ('array min1 max1', 'smpl 0 0x0 0x100100 0x4000000 0x0')):
        manifest = stage(name).manifest()
        assert line in manifest.splitlines() and 'smp 0 ' not in manifest, (name, manifest)
    # Since #2177 an ordinary image_sample with a constant sampler carries the same limits (tests/startup_image_sample_lod_check.py
    # draws it). Control: the same words through image_sample are the same smpl line, and with mip filter 0 the
    # sampler stays an ordinary smp line (the ordinary forms leave that sampler as it was).
    other = G.Pipeline(VS_WORDS, bytes.fromhex(WORDS['array nobias']), root_blob())
    other.hlsl_ps()
    assert 'smpl 0 0x0 0x200000 0x4000000 0x0' in other.manifest().splitlines() and 'smp ' not in other.manifest()
    refused('array min3 max2', 'image_sample_b with sampler MIN_LOD 0x300 above MAX_LOD 0x200')
    refused('mip filter none', 'image_sample_b with mipmapping disabled')
    refused('sampler from root constant', 'image_sample_b with a sampler that is not constant')
    for heap, text in ((True, 'image_sample_b through a sampler heap (LOD limits and bias unknown)'),
                       (False, 'image_sample_b through a sampler heap (replaced by a default sampler)')):
        try:
            stage('sampler heap', sampler_heap=heap).hlsl_ps()
        except G.Unsupported as error:
            assert text == str(error), (heap, str(error))
        else:
            raise AssertionError('image_sample_b accepted a heap sampler (table %s)' % heap)
    # Control: the same heap program with a plain image_sample is still translated, so the refusal is the _b form's.
    plain_heap = H.PS_WORDS
    for heap in (True, False):
        G.Pipeline(H.VS_WORDS, plain_heap, H.root_blob(), sampler_heap=heap).hlsl_ps()
    print('limits: smpl line for MIN_LOD/MAX_LOD; refused: MIN_LOD > MAX_LOD, mip filter 0, root-constant sampler, '
          'sampler heap (table and default-sampler shortcut)')

    # Cache-policy bits do not change the sampled value; a partial dmask only changes the destination.
    for name in ('glc', 'slc', 'dlc'):
        assert stage(name).hlsl_ps() == stage('plain').hlsl_ps(), name
    assert bias_call(stage('dmask3').hlsl_ps(), 'float2') == (['v1', 'v2'], 'v0')
    print('glc, slc, dlc and a partial dmask are accepted')

    # Model: every bias case reads its own mip level and a no-bias sample reads level 0.
    cells = {(m, s): texel_cell(m, s) for m in range(3) for s in range(2)}
    assert len(set(cells.values())) == 6, cells
    assert [mip_for(b) for b in (-1, 0, 1)] == [0, 1, 2]
    print('draw model: 6 distinct (mip, slice) cells')


# (fixture, bias): the mip the bias selects.
DRAW_CASES = [('array -1', -1), ('array 0', 0), ('array 1', 1)]
# (fixture, bias, MIN_LOD, MAX_LOD, mip the guest reads, mip an ignored limit would read): the limits clamp the biased
# LOD below the mip the unclamped LOD reaches. Mip 2, the next level above MAX_LOD 1, is visible in the texture.
LIMIT_CASES = [('array max1', 1, 0, 1, 1, 2), ('array max0', 1, 0, 0, 0, 2),
               ('array min2', -1, 2, 2, 2, 0), ('array min1 max1', -1, 1, 1, 1, 0)]
CALL = 'r0.SampleBias(q0, float3(F(v1), F(v2), F(v3)), F(v0))'
U_INIT = 'v1 = 0x3f400000u;'
U_DERIVED = 'v1 = U(0.75 + 0.5 * (i.pos.x - 0.5));'
IGNORED = 'r0.SampleBias(q0, float3(F(v1), F(v2), F(v3)), 0.0)'
# Each mutant edits the translated call. (label, call, fixture, mip the mutant reads or None for any other cell)
MUTANTS = [
    ('bias ignored', IGNORED, 'array -1', 1),
    ('bias ignored', IGNORED, 'array 1', 1),
    ('bias negated', 'r0.SampleBias(q0, float3(F(v1), F(v2), F(v3)), -F(v0))', 'array -1', 2),
    ('bias negated', 'r0.SampleBias(q0, float3(F(v1), F(v2), F(v3)), -F(v0))', 'array 1', 0),
    ('bias and u exchanged', 'r0.SampleBias(q0, float3(F(v0), F(v2), F(v3)), F(v1))', 'array -1', None),
    ('bias and u exchanged', 'r0.SampleBias(q0, float3(F(v0), F(v2), F(v3)), F(v1))', 'array 1', None),
    ('plain Sample', 'r0.Sample(q0, float3(F(v1), F(v2), F(v3)))', 'array -1', 1),
    ('plain Sample', 'r0.Sample(q0, float3(F(v1), F(v2), F(v3)))', 'array 1', 1),
]


def drawable(hlsl):
    """The translated stage with u made to change by 0.5 per pixel (only the u initialiser is replaced)."""
    assert hlsl.count(U_INIT) == 1, hlsl
    return hlsl.replace(U_INIT, U_DERIVED)


def check_binary(llvm_mc):
    for name in FIXTURES:
        built = assemble(llvm_mc, source(name))
        if built != bytes.fromhex(WORDS[name]):
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), WORDS[name]))
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    operation = '--startup-image-2d-array-mips'
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    locate()

    def draw_once(vs, shader, slice_index, expected, sampler):
        run = gpu_case_runner.run(renderer, adapter, [operation, vs, shader, slice_index, expected, sampler])
        sys.stdout.write(run.stdout)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            sys.exit(77)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
        return run

    def sampler_line(pipe):
        lines = [line for line in pipe.manifest().splitlines() if line.startswith(('smp ', 'smpl '))]
        assert len(lines) == 1, pipe.manifest()
        return lines[0]

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'fullscreen_b_%s' % kind, vs_profile, directory)
            translated = {}
            for name, bias in DRAW_CASES + [('array nobias', None)]:
                pipe = stage(name)
                translated[name] = pipe.hlsl_ps()
                tag = name.replace(' ', '').replace('-', 'm')
                shader = compile_hlsl(compiler, kind, drawable(translated[name]), 'ps_%s_%s' % (tag, kind), ps_profile, directory)
                line = sampler_line(pipe)
                for slice_index in (0, 1):
                    if bias is None:
                        continue
                    expected = texel_cell(mip_for(bias), slice_index)
                    run = draw_once(vs, shader, slice_index, expected, line)
                    if run.returncode:
                        raise AssertionError('%s %s slice=%d\n%s\n%s' % (kind, name, slice_index, run.stdout, run.stderr))
                    print('%s sample_b %s slice=%d: %s' % (kind, name, slice_index, expected), flush=True)
            # M200-01 B: the same bias-1 array sample after a uniform join must still read mip 0.
            join_words = with_join_before_sample(bytes.fromhex(WORDS['array -1']))
            join_pipe = G.Pipeline(VS_WORDS, join_words, root_blob())
            join_shader = compile_hlsl(compiler, kind, drawable(join_pipe.hlsl_ps()), 'ps_join_m1_%s' % kind,
                                       ps_profile, directory)
            join_line = sampler_line(join_pipe)
            for slice_index in (0, 1):
                expected = texel_cell(mip_for(-1), slice_index)
                run = draw_once(vs, join_shader, slice_index, expected, join_line)
                if run.returncode:
                    raise AssertionError('%s join sample_b slice=%d\n%s\n%s' % (kind, slice_index, run.stdout, run.stderr))
                print('%s sample_b join-before bias=-1 slice=%d: %s' % (kind, slice_index, expected), flush=True)
            # A plain image_sample after a uniform join keeps the implicit LOD: it reads mip 1, not the mip 0 of
            # SampleLevel(.., 0). The mip-0 expectation is the negative control and must fail on the readback.
            plain_pipe = G.Pipeline(VS_WORDS, with_join_before_sample(bytes.fromhex(WORDS['array nobias'])), root_blob())
            plain_hlsl = plain_pipe.hlsl_ps()
            assert '.Sample(' in plain_hlsl and 'SampleLevel' not in plain_hlsl, plain_hlsl
            plain_shader = compile_hlsl(compiler, kind, drawable(plain_hlsl), 'ps_plain_join_%s' % kind, ps_profile, directory)
            plain_line = sampler_line(plain_pipe)
            for slice_index in (0, 1):
                expected = texel_cell(1, slice_index)
                run = draw_once(vs, plain_shader, slice_index, expected, plain_line)
                if run.returncode:
                    raise AssertionError('%s plain image_sample join slice=%d\n%s\n%s' % (kind, slice_index, run.stdout, run.stderr))
                level0 = draw_once(vs, plain_shader, slice_index, texel_cell(0, slice_index), plain_line)
                if level0.returncode == 0 or 'actual=%s' % expected not in level0.stderr:
                    raise AssertionError('%s plain image_sample join slice=%d: the level-0 expectation did not fail\n%s\n%s'
                                         % (kind, slice_index, level0.stdout, level0.stderr))
                print('%s image_sample join-before slice=%d: %s (level 0 fails as expected)' % (kind, slice_index, expected),
                      flush=True)
            # Negative control: without a bias the draw reads mip 1, so the bias -1 and bias +1 expectations must fail.
            line = sampler_line(stage('array nobias'))
            nobias = compile_hlsl(compiler, kind, drawable(translated['array nobias']), 'ps_nobias_%s' % kind, ps_profile, directory)
            for bias in (-1, 1):
                for slice_index in (0, 1):
                    actual = texel_cell(1, slice_index)
                    run = draw_once(vs, nobias, slice_index, texel_cell(mip_for(bias), slice_index), line)
                    if run.returncode == 0 or 'actual=%s' % actual not in run.stderr:
                        raise AssertionError('%s image_sample without bias, expecting bias %d: did not read mip 1\n%s\n%s'
                                             % (kind, bias, run.stdout, run.stderr))
                print('%s image_sample without bias, expecting bias %d: fails as expected, reads mip 1' % (kind, bias),
                      flush=True)
            # The descriptor limits are what the guest read, and mip 2 is visible: a bias that would reach it stays on the
            # limit. The control reads every case once through the smpl line.
            for name, bias, lo, hi, mip, unclamped in LIMIT_CASES:
                pipe = stage(name)
                assert mip == mip_for(bias, lo, hi) and unclamped == mip_for(bias) and mip != unclamped, name
                tag = name.replace(' ', '')
                shader = compile_hlsl(compiler, kind, drawable(pipe.hlsl_ps()), 'ps_limit_%s_%s' % (tag, kind), ps_profile, directory)
                line = sampler_line(pipe)
                assert line.startswith('smpl '), line
                for slice_index in (0, 1):
                    expected = texel_cell(mip, slice_index)
                    run = draw_once(vs, shader, slice_index, expected, line)
                    if run.returncode:
                        raise AssertionError('%s %s slice=%d\n%s\n%s' % (kind, name, slice_index, run.stdout, run.stderr))
                    print('%s sample_b %s slice=%d: %s' % (kind, name, slice_index, expected), flush=True)
                # Mutant: the limits are not carried (the ordinary smp line, MinLOD 0 and MaxLOD unbounded).
                dropped = 'smp ' + line[len('smpl '):]
                run = draw_once(vs, shader, 1, texel_cell(mip, 1), dropped)
                if run.returncode == 0 or 'actual=%s' % texel_cell(unclamped, 1) not in run.stderr:
                    raise AssertionError('%s %s: dropping the limits still read mip %d\n%s\n%s'
                                         % (kind, name, mip, run.stdout, run.stderr))
                print('%s sample_b %s, limits not carried: fails as expected, reads mip %d' % (kind, name, unclamped), flush=True)
            for label, call, name, mip in MUTANTS:
                bias = dict(DRAW_CASES)[name]
                assert CALL in translated[name], translated[name]
                source_text = drawable(translated[name].replace(CALL, call))
                assert source_text != drawable(translated[name]), (name, label)
                tag = '%s_%s' % (label.replace(' ', ''), name.replace(' ', '').replace('-', 'm'))
                mutant = compile_hlsl(compiler, kind, source_text, 'ps_mutant_%s_%s' % (tag, kind), ps_profile, directory)
                slice_index = 1
                expected = texel_cell(mip_for(bias), slice_index)
                run = draw_once(vs, mutant, slice_index, expected, sampler_line(stage(name)))
                if run.returncode == 0 or 'startup_image mismatch' not in run.stderr:
                    raise AssertionError('%s %s %s: the mutant still read %s\n%s\n%s'
                                         % (kind, name, label, expected, run.stdout, run.stderr))
                if mip is not None and 'actual=%s' % texel_cell(mip, slice_index) not in run.stderr:
                    raise AssertionError('%s %s %s: expected a read of mip %d\n%s' % (kind, name, label, mip, run.stderr))
                print('%s sample_b %s, %s: fails as expected' % (kind, name, label), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary' and len(args) == 1:
        check_binary(args[0])
    elif mode == 'draw' and len(args) in (3, 4):
        controls()
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
