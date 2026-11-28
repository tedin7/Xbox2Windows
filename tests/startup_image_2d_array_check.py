"""2D array images in tools/startup/gfx2hlsl.py (dim:SQ_RSRC_IMG_2D_ARRAY).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given) and
                                       draw: the readback must name the texel of the slice that was selected

Two source-built gfx1013 pixel stages. Both load a texture descriptor from table 0 and take the slice
from root constant 1 (s9). The sample stage builds an all-zero inline sampler and samples at
x = 0.75, y = 0.25 with image_sample. The load stage converts the slice to an integer and reads
texel (1, 0) with image_load, which needs no sampler. tests/startup_image_3d.cpp, run as
--startup-image-2d-array, binds a 2x2 R32G32B32A32_FLOAT texture with two slices whose texel
(x, y, slice) holds ((1 + x + 2y + 4 * slice) / 8, x, y, slice).

Two depth-comparison stages (image_sample_c_lz and image_sample_c with MAX_LOD 0) use the same
2x2 two-slice array as R32_FLOAT depths via an smpc manifest line (point clamp, LESS): at
x = 0.75, y = 0.25 (texel (1, 0)), slice 0 has depth 0.25 and slice 1 has depth 0.75. With a single
reference 0.5 between them, slice 0 fails LESS (0.0) and slice 1 passes (1.0); always-slice-zero and
swapped-slice mutants must each read the opposite comparison result.

image_load_mip on a 2D array takes four address registers, x, y, slice and mip, in that order, and is
translated to Load(int4(x, y, slice, mip)). Its fixtures are source-built stages that load texel (x, 0)
of a slice (root constant 1) at a constant mip level. They are drawn with --startup-image-2d-array-mips: a
4x4 texture of two slices with three mip levels whose cells all differ, texel (x, y) of slice s at mip m
holding ((1 + x + 4y + 16s + 32m) / 128, x, y, s + 2m). The expected colours come from mip_cell() below.
Cube loads stay refused (cube sampling: tests/startup_image_cube_check.py). The mip fixtures use spellings that llvm-mc 19 assembles; LLVM 15 is not known to
accept them, so 'binary <llvm-mc> mip' is opt-in and CI does not pass it.
"""
import re
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, assemble, compile_hlsl, root_blob, texel  # noqa: E402
import subprocess  # noqa: E402

SAMPLE_SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, 0
s_mov_b32 s26, 0
s_mov_b32 s27, 0
v_mov_b32 v0, 0x3f400000
v_mov_b32 v1, 0x3e800000
v_mov_b32 v2, s9
s_waitcnt lgkmcnt(0)
image_sample v[0:3], v[0:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
s_waitcnt vmcnt(0)
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
"""
SAMPLE_ARRAY = bytes.fromhex('280f80f00000c400')
SAMPLE_WORDS = bytes.fromhex(
    '04040cf4000000fa' '800398be' '800399be' '80039abe' '80039bbe' 'ff02007e0000403f' 'ff02027e0000803e'
    '0902047e' '7fc08cbf') + SAMPLE_ARRAY + bytes.fromhex('703f8cbf' '0f1800f800010203' '000081bf')
# The same program with only the dim field changed.
SOURCE_3D = SAMPLE_SOURCE.replace('dim:SQ_RSRC_IMG_2D_ARRAY', 'dim:SQ_RSRC_IMG_3D')
WORDS_3D = SAMPLE_WORDS.replace(SAMPLE_ARRAY, bytes.fromhex('100f80f00000c400'))
SOURCE_MSAA = SAMPLE_SOURCE.replace('dim:SQ_RSRC_IMG_2D_ARRAY', 'dim:SQ_RSRC_IMG_2D_MSAA')
WORDS_MSAA = SAMPLE_WORDS.replace(SAMPLE_ARRAY, bytes.fromhex('300f80f00000c400'))
SOURCE_LZ = SAMPLE_SOURCE.replace('image_sample ', 'image_sample_lz ')
WORDS_LZ = SAMPLE_WORDS.replace(SAMPLE_ARRAY, bytes.fromhex('280f9cf00000c400'))

LOAD_SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
v_mov_b32 v0, 1
v_mov_b32 v1, 0
v_mov_b32 v2, s9
v_cvt_i32_f32 v2, v2
s_waitcnt lgkmcnt(0)
image_load v[0:3], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
s_waitcnt vmcnt(0)
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
"""
LOAD_WORDS = bytes.fromhex(
    '04040cf4000000fa' '8102007e' '8002027e' '0902047e' '0211047e' '7fc08cbf' '280f00f000000400' '703f8cbf'
    '0f1800f800010203' '000081bf')

SAMPLE_CALL = 'r0.Sample(q0, float3(F(v0), F(v1), F(v2)))'
LOAD_CALL = 'r0.Load(int4(v0, v1, v2, 0))'

# Depth-comparison 2D-array stages: reference v0 = 0.5, u = 0.75 (v1), v = 0.25 (v2), slice = s9 (v3).
# Sampler word 0: CLAMP_LAST_TEXEL (2) in X and Y, LESS (1 << 12), TRUNC_COORD (1 << 27); word 1 = 0 (MAX_LOD 0).
CMP_SAMPLER_W0 = 2 | 2 << 3 | 1 << 12 | 1 << 27
CMP_SAMPLER_LINE = 'smpc 0 0x%x 0x0 0x0 0x0' % CMP_SAMPLER_W0
SAMPLE_C_LZ_SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0x8001012
s_mov_b32 s25, 0
s_mov_b32 s26, 0
s_mov_b32 s27, 0
v_mov_b32 v0, 0.5
v_mov_b32 v1, 0x3f400000
v_mov_b32 v2, 0x3e800000
v_mov_b32 v3, s9
s_waitcnt lgkmcnt(0)
image_sample_c_lz v4, v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY
s_waitcnt vmcnt(0)
exp mrt0 v4, v4, v4, v4 done vm
s_endpgm
"""
SAMPLE_C_LZ_WORDS = bytes.fromhex(
    '04040cf4000000fa' 'ff0398be12100008' '800399be' '80039abe' '80039bbe'
    'f002007e' 'ff02027e0000403f' 'ff02047e0000803e' '0902067e' '7fc08cbf'
    '2801bcf00004c400' '703f8cbf' '0f1800f804040404' '000081bf')
# The same stage with sampler word 2 = 0x70000000 (MIP_POINT_PRECLAMP, ANISO_OVERRIDE, BLEND_ZERO_PRT): the level-zero
# form ignores them, so the draw must match the CPU reference below exactly like the plain stage.
S_MOV_S26_ZERO = bytes.fromhex('80039abe')
SAMPLE_C_LZ_W2 = 0x70000000
SAMPLE_C_LZ_W2_WORDS = SAMPLE_C_LZ_WORDS.replace(
    S_MOV_S26_ZERO, bytes.fromhex('ff039abe') + struct.pack('<I', SAMPLE_C_LZ_W2))
assert SAMPLE_C_LZ_WORDS.count(S_MOV_S26_ZERO) == 1 and SAMPLE_C_LZ_W2_WORDS != SAMPLE_C_LZ_WORDS
SAMPLE_C_SOURCE = SAMPLE_C_LZ_SOURCE.replace('image_sample_c_lz ', 'image_sample_c ')
SAMPLE_C_WORDS = SAMPLE_C_LZ_WORDS.replace(bytes.fromhex('2801bcf00004c400'), bytes.fromhex('2801a0f00004c400'))
SAMPLE_C_CALL = 'r0.SampleCmpLevelZero(q0, float3(F(v1), F(v2), F(v3)), F(v0))'
CMP_ZERO = ','.join(['00000000'] * 4)
CMP_ONE = ','.join(['3f800000'] * 4)


def cmp_reference(slice_value):
    """CPU reference of the comparison draw: point-clamp read of texel (1, 0) of the rounded slice, LESS, ref 0.5."""
    depth = 0.25 + 0.5 * min(1, max(0, int(slice_value + 0.5)))
    return ','.join(['%08x' % struct.unpack('<I', struct.pack('<f', 1.0 if 0.5 < depth else 0.0))[0]] * 4)

# image_load_mip on a 2D array: x from a constant, y = 0, slice from root constant 1 (s9), mip a constant.
MIP_IMAGE = bytes.fromhex('280f04f000000400')
MIP_IMAGE_NSA = bytes.fromhex('2a0f04f00000040001030200')  # address [v0, v1, v3, v2]: slice is v3, mip is v2
MIP_X = {0: '8002007e', 1: '8102007e'}  # v_mov_b32 v0, 0 and 1
MIP_LEVEL = {0: '8002067e', 1: '8102067e', 2: '8202067e'}  # v_mov_b32 v3, 0, 1 and 2
MIP_CALL = 'r0.Load(int4(v0, v1, v2, v3))'
MIP_SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
v_mov_b32 v0, {x}
v_mov_b32 v1, 0
v_mov_b32 v2, s9
v_cvt_i32_f32 v2, v2
v_mov_b32 v3, {mip}
s_waitcnt lgkmcnt(0)
{instruction}
s_waitcnt vmcnt(0)
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
"""
MIP_INSTRUCTION = 'image_load_mip v[0:3], v[0:3], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY'
MIP_INSTRUCTION_NSA = 'image_load_mip v[0:3], [v0, v1, v3, v2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY'
# name: (x, mip). The 4x4 texture has mips of 4x4, 2x2 and 1x1, so x = 1 is valid for mips 0 and 1 only.
MIP_FIXTURES = {'mip 0': (1, 0), 'mip 1': (1, 1), 'mip 2': (0, 2)}


def check_compare_modifiers():
    """image_sample_c needs exactly one dim and one dmask; anything else is refused by name.

    No machine code and no device: the refusal is decided from the modifier list alone, so a bare
    stage is enough. The legal list is checked to reach a *different* failure, which proves the
    guard is not simply raising for every input.
    """
    base = 'image_sample_c'
    for mods in (['dim:SQ_RSRC_IMG_2D'], ['dmask:0x1'],
                 ['dim:SQ_RSRC_IMG_2D', 'dim:SQ_RSRC_IMG_2D', 'dmask:0x1']):
        stage = G.Stage.__new__(G.Stage)
        try:
            stage.image_compare(['v1', 'v2', 'v3', 's4'], mods, 'c')
        except G.Unsupported as error:
            assert str(error) == '%s without exactly one dim and dmask' % base, (mods, error)
        else:
            raise AssertionError('%s accepted modifiers %s' % (base, mods))
    stage = G.Stage.__new__(G.Stage)
    try:
        stage.image_compare(['v1', 'v2', 'v3', 's4'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0x1'], 'c')
    except G.Unsupported as error:
        assert 'without exactly one dim and dmask' not in str(error), error
    except AttributeError:
        pass  # past the guard: the bare stage has none of the state the lowering needs

def mip_words(x, mip, image=MIP_IMAGE):
    return (bytes.fromhex('04040cf4000000fa' + MIP_X[x] + '8002027e' '0902047e' '0211047e' + MIP_LEVEL[mip] + '7fc08cbf')
            + image + bytes.fromhex('703f8cbf' '0f1800f800010203' '000081bf'))


def mip_cell(x, y, s, m):
    """The mip fixture texture's value at texel (x, y) of slice s and mip level m, as four binary32 hex words."""
    assert 0 <= s < 2 and 0 <= m < 3 and 0 <= x < (4 >> m) and 0 <= y < (4 >> m), (x, y, s, m)
    return ','.join('%08x' % struct.unpack('<I', struct.pack('<f', c))[0]
                    for c in ((1 + x + 4 * y + 16 * s + 32 * m) / 128, x, y, s + 2 * m))


# Each mutant edits the translated address and must read another cell. (label, fixture, slice, address, cell read)
MIP_MUTANTS = [
    ('slice and mip exchanged', 'mip 0', 1, 'int4(v0, v1, v3, v2)', lambda x, y, s, m: (x, y, m, s)),
    ('slice and mip exchanged', 'mip 1', 0, 'int4(v0, v1, v3, v2)', lambda x, y, s, m: (x, y, m, s)),
    ('mip ignored', 'mip 1', 1, 'int4(v0, v1, v2, 0)', lambda x, y, s, m: (x, y, s, 0)),
    ('mip ignored', 'mip 2', 0, 'int4(v0, v1, v2, 0)', lambda x, y, s, m: (x, y, s, 0)),
    ('mip one too high', 'mip 0', 0, 'int4(v0, v1, v2, v3 + 1)', lambda x, y, s, m: (x, y, s, m + 1)),
    ('mip one too low', 'mip 2', 1, 'int4(v0, v1, v2, v3 - 1)', lambda x, y, s, m: (x, y, s, m - 1)),
    ('slice ignored', 'mip 1', 1, 'int4(v0, v1, 0, v3)', lambda x, y, s, m: (x, y, 0, m)),
]


def translate(ps_words):
    return G.Pipeline(VS_WORDS, ps_words, root_blob())


def mip_controls():
    # image_load_mip on a 2D array: the address registers in order, a Texture2DArray, mip 0 not assumed.
    for name, (x, mip) in MIP_FIXTURES.items():
        pipe = translate(mip_words(x, mip))
        ps, manifest = pipe.hlsl_ps(), pipe.manifest()
        assert 'Texture2DArray<float4> r0 : register(t0);' in ps and 'SamplerState' not in ps, ps
        assert 't = %s;' % MIP_CALL in ps and re.search(r'\bv2 = ud1;', ps), ps
        assert 'srv 0 tex2da table 0 0' in manifest and 'smp ' not in manifest and not pipe.notes, (manifest, pipe.notes)
    print('image_load_mip: Texture2DArray r0, %s with v2 = ud1; manifest srv 0 tex2da, no sampler' % MIP_CALL)

    # The same instruction with non-consecutive registers: x, y, slice, mip are taken in that order.
    ps = translate(mip_words(1, 0, MIP_IMAGE_NSA)).hlsl_ps()
    assert 't = r0.Load(int4(v0, v1, v3, v2));' in ps, ps
    print('image_load_mip, address [v0, v1, v3, v2]: Load(int4(v0, v1, v3, v2))')

    # The cache-policy bits and unorm do not change the address and are accepted.
    accepted = bytearray(MIP_IMAGE)
    accepted[1] |= 0x10  # unorm
    assert 't = %s;' % MIP_CALL in translate(mip_words(1, 0, bytes(accepted))).hlsl_ps()

    # Packed 16-bit data narrows the four channels into v[0:1] and marks SRV 0 for exact D16 validation.
    d16 = bytearray(MIP_IMAGE)
    d16[7] |= 0x80
    d16_pipe = translate(mip_words(1, 0, bytes(d16)))
    d16_ps = d16_pipe.hlsl_ps()
    assert 't = %s;' % MIP_CALL in d16_ps, d16_ps
    assert 'v0 = pack16_d16(t.x, ud4) | (pack16_d16(t.y, ud4) << 16);' in d16_ps, d16_ps
    assert 'v1 = pack16_d16(t.z, ud4) | (pack16_d16(t.w, ud4) << 16);' in d16_ps, d16_ps
    assert 'd16srv 0 exact' in d16_pipe.manifest().splitlines(), d16_pipe.manifest()

    # Refused: packed addresses and fault-status returns need their own lowering.
    for flag, byte, mask in (('a16', 7, 0x40), ('tfe', 2, 0x01)):
        refused = bytearray(MIP_IMAGE)
        refused[byte] |= mask
        try:
            translate(mip_words(1, 0, bytes(refused)))
        except G.Unsupported as error:
            assert 'image_load_mip' in str(error) and flag in str(error), error
        else:
            raise AssertionError('image_load_mip with %s accepted' % flag)

    # A register count other than x, y, slice, mip is refused. The decoder derives the count from the
    # dimension, so the same words are decoded as if the array took two address dwords.
    original = G.D.MIMG_DIMS[5]
    G.D.MIMG_DIMS[5] = (original[0], 2, original[2])
    try:
        translate(mip_words(1, 0))
    except G.Unsupported as error:
        assert 'image_load_mip address or modifiers' in str(error), error
    else:
        raise AssertionError('image_load_mip with three address registers accepted')
    finally:
        G.D.MIMG_DIMS[5] = original
    assert G.D.MIMG_DIMS[5] == original

    # Cube loads are refused as before. 2D and 3D keep their forms (same words, dim field changed).
    cube = bytearray(MIP_IMAGE)
    cube[0] = 0x18
    try:
        translate(mip_words(1, 0, bytes(cube)))
    except G.Unsupported as error:
        assert 'image dimension SQ_RSRC_IMG_CUBE' in str(error), error
    else:
        raise AssertionError('image_load_mip on a cube accepted')
    for dim_code, call, kind in ((0x08, 'int3(v0, v1, v2)', 'tex2d'), (0x10, 'int4(v0, v1, v2, v3)', 'tex3d')):
        other = bytearray(MIP_IMAGE)
        other[0] = dim_code
        pipe = translate(mip_words(1, 0, bytes(other)))
        assert 't = r0.Load(%s);' % call in pipe.hlsl_ps() and 'srv 0 %s table 0 0' % kind in pipe.manifest(), pipe.hlsl_ps()
    print('image_load_mip: d16, a16, tfe and a three-register address refused; cube refused; 2D and 3D unchanged')

    # The expected colours tell every swapped axis apart: all cells differ, and each mutant reads another one.
    cells = {mip_cell(x, y, s, m) for m in range(3) for s in range(2) for y in range(4 >> m) for x in range(4 >> m)}
    assert len(cells) == 2 * (16 + 4 + 1), len(cells)
    for label, fixture, slice_index, address, read in MIP_MUTANTS:
        x, mip = MIP_FIXTURES[fixture]
        assert mip_cell(*read(x, 0, slice_index, mip)) != mip_cell(x, 0, slice_index, mip), (label, fixture)
    print('mip model: %d distinct cells; %d mutants each read another cell' % (len(cells), len(MIP_MUTANTS)))


def controls():
    check_compare_modifiers()
    # image_sample: a Texture2DArray, with the slice as the third component of the coordinate.
    pipe = translate(SAMPLE_WORDS)
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'Texture2DArray<float4> r0 : register(t0);' in ps and 'Texture3D' not in ps and 'Texture2D<' not in ps, ps
    assert 't = %s;' % SAMPLE_CALL in ps, ps
    assert re.search(r'\bv2 = ud1;', ps), ps
    assert 'srv 0 tex2da table 0 0' in manifest and 'udsrc 1 rootconst 1 0' in manifest, manifest
    assert not pipe.notes, pipe.notes
    print('image_sample: Texture2DArray r0, %s with v2 = ud1 (root constant 1); manifest srv 0 tex2da' % SAMPLE_CALL)

    # image_sample_lz: the same coordinate at level 0.
    ps = translate(WORDS_LZ).hlsl_ps()
    assert 't = r0.SampleLevel(q0, float3(F(v0), F(v1), F(v2)), 0);' in ps and 'Texture2DArray<float4> r0' in ps, ps
    print('image_sample_lz: SampleLevel(q0, float3(F(v0), F(v1), F(v2)), 0)')

    # image_load: integer texel and slice, mip 0, and no sampler.
    pipe = translate(LOAD_WORDS)
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'Texture2DArray<float4> r0 : register(t0);' in ps, ps
    assert 't = %s;' % LOAD_CALL in ps and 'SamplerState' not in ps, ps
    assert 'srv 0 tex2da table 0 0' in manifest and 'smp ' not in manifest, manifest
    print('image_load: %s; manifest srv 0 tex2da, no sampler' % LOAD_CALL)

    # image_sample_c_lz and image_sample_c (MAX_LOD 0): Texture2DArray with SamplerComparisonState and smpc line.
    for name, words in (('image_sample_c_lz', SAMPLE_C_LZ_WORDS), ('image_sample_c', SAMPLE_C_WORDS)):
        pipe = translate(words)
        ps, manifest = pipe.hlsl_ps(), pipe.manifest()
        assert 'Texture2DArray<float4> r0 : register(t0);' in ps, ps
        assert 'SamplerComparisonState q0 : register(s0);' in ps, ps
        assert 'v4 = U(%s);' % SAMPLE_C_CALL in ps and re.search(r'\bv3 = ud1;', ps), ps
        assert 'srv 0 tex2da table 0 0' in manifest and 'udsrc 1 rootconst 1 0' in manifest, manifest
        assert CMP_SAMPLER_LINE in manifest and not pipe.notes, (manifest, pipe.notes)
    print('image_sample_c[_lz]: Texture2DArray r0, %s with v3 = ud1; manifest srv 0 tex2da, %s'
          % (SAMPLE_C_CALL, CMP_SAMPLER_LINE))

    # Only the dim field differs: the 3D form keeps its own type and manifest kind.
    pipe = translate(WORDS_3D)
    assert 'Texture3D<float4> r0 : register(t0);' in pipe.hlsl_ps() and 'srv 0 tex3d table 0 0' in pipe.manifest()
    print('dim 3D (same words, dim field changed): still Texture3D, manifest srv 0 tex3d')

    # A multisampled 2D image is still refused rather than sampled as an array.
    try:
        translate(WORDS_MSAA)
    except G.Unsupported as error:
        assert 'image dimension SQ_RSRC_IMG_2D_MSAA' in str(error), error
    else:
        raise AssertionError('dim:SQ_RSRC_IMG_2D_MSAA accepted')
    print('dim 2D MSAA: refused')
    mip_controls()


def check_binary(llvm_mc, with_mip=False):
    if with_mip:
        mip_fixtures = [('array load_mip %s' % name, MIP_SOURCE.format(x=x, mip=m, instruction=MIP_INSTRUCTION),
                         mip_words(x, m)) for name, (x, m) in MIP_FIXTURES.items()]
        mip_fixtures.append(('array load_mip NSA', MIP_SOURCE.format(x=1, mip=0, instruction=MIP_INSTRUCTION_NSA),
                             mip_words(1, 0, MIP_IMAGE_NSA)))
        for name, source, words in mip_fixtures:
            built = assemble(llvm_mc, source)
            if built != words:
                raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    for name, source, words in (('array sample', SAMPLE_SOURCE, SAMPLE_WORDS), ('array sample_lz', SOURCE_LZ, WORDS_LZ),
                                ('array load', LOAD_SOURCE, LOAD_WORDS),
                                ('array sample_c_lz', SAMPLE_C_LZ_SOURCE, SAMPLE_C_LZ_WORDS),
                                ('array sample_c', SAMPLE_C_SOURCE, SAMPLE_C_WORDS),
                                ('3D sample', SOURCE_3D, WORDS_3D),
                                ('MSAA sample', SOURCE_MSAA, WORDS_MSAA)):
        built = assemble(llvm_mc, source)
        if built != words:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    # (stage, words, call, cases, mutants, use_manifest_sampler). x = 0.75, y = 0.25 (sample/sample_c) and
    # texel (1, 0) (load) select column 1, row 0; the root constant selects the slice. A slice coordinate is
    # not normalized: 0.25 rounds to slice 0 and 0.75 to slice 1. For the comparison stages, slice 0 has depth
    # 0.25 and slice 1 has depth 0.75 with reference 0.5 and LESS, so slice 0 returns 0.0 and slice 1 returns 1.0.
    # Each mutant edits the translated HLSL and must read another texel or comparison result.
    cmp_cases = [('0', CMP_ZERO), ('1', CMP_ONE), ('0.25', CMP_ZERO), ('0.75', CMP_ONE)]
    cmp_mutants = [
        ('always slice 0', SAMPLE_C_CALL.replace('F(v3)', '0.0'), '1', CMP_ZERO),
        ('swapped slice 0->1', SAMPLE_C_CALL.replace('F(v3)', '1.0 - F(v3)'), '0', CMP_ONE),
        ('swapped slice 1->0', SAMPLE_C_CALL.replace('F(v3)', '1.0 - F(v3)'), '1', CMP_ZERO),
    ]
    stages = [
        ('sample', SAMPLE_WORDS, SAMPLE_CALL,
         [('0', texel(1, 0, 0)), ('1', texel(1, 0, 1)), ('0.25', texel(1, 0, 0)), ('0.75', texel(1, 0, 1))],
         [('slice fixed to 0', SAMPLE_CALL.replace('F(v2)', '0.0'), '1', texel(1, 0, 0)),
          ('x and y exchanged', 'r0.Sample(q0, float3(F(v1), F(v0), F(v2)))', '1', texel(0, 1, 1))],
         False),
        ('load', LOAD_WORDS, LOAD_CALL,
         [('0', texel(1, 0, 0)), ('1', texel(1, 0, 1))],
         [('slice fixed to 0', LOAD_CALL.replace('v2, 0)', '0, 0)'), '1', texel(1, 0, 0)),
          ('x and slice exchanged', 'r0.Load(int4(v2, v1, v0, 0))', '0', texel(0, 0, 1))],
         False),
        ('sample_c_lz', SAMPLE_C_LZ_WORDS, SAMPLE_C_CALL, cmp_cases, cmp_mutants, True),
        ('sample_c', SAMPLE_C_WORDS, SAMPLE_C_CALL, cmp_cases, cmp_mutants, True),
        ('sample_c_lz_word2_bits', SAMPLE_C_LZ_W2_WORDS, SAMPLE_C_CALL,
         [(z, cmp_reference(float(z))) for z in ('0', '1', '0.25', '0.75')], cmp_mutants, True),
    ]
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'fullscreen_%s' % kind, vs_profile, directory)
            for stage, words, call, cases, mutants, use_manifest_sampler in stages:
                pipe = translate(words)
                translated = pipe.hlsl_ps()
                sampler_args = []
                if use_manifest_sampler:
                    smpc = [line for line in pipe.manifest().splitlines() if line.startswith('smpc ')]
                    assert len(smpc) == 1, pipe.manifest()
                    sampler_args = [smpc[0]]
                shader = compile_hlsl(compiler, kind, translated, 'ps_%s_%s' % (stage, kind), ps_profile, directory)
                for z, expected in cases:
                    run = subprocess.run([renderer, adapter, '--startup-image-2d-array', str(vs), str(shader), z, expected,
                                          *sampler_args],
                                         capture_output=True, text=True)
                    sys.stdout.write(run.stdout)
                    if run.returncode == 77:
                        print(run.stdout + run.stderr)
                        sys.exit(77)
                    if run.returncode:
                        raise AssertionError('%s %s slice=%s\n%s\n%s' % (kind, stage, z, run.stdout, run.stderr))
                    if 'debug_layer=enabled' in run.stdout:
                        assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
                    print('%s %s slice=%s: %s' % (kind, stage, z, expected), flush=True)
                for label, mutated, z, actual in mutants:
                    source = translated.replace(call, mutated)
                    assert source != translated, (stage, label)
                    mutant = compile_hlsl(compiler, kind, source, 'ps_mutant_%s_%s' % (stage, kind), ps_profile, directory)
                    expected = dict(cases)[z]
                    run = subprocess.run([renderer, adapter, '--startup-image-2d-array', str(vs), str(mutant), z, expected,
                                          *sampler_args],
                                         capture_output=True, text=True)
                    sys.stdout.write(run.stdout)
                    if run.returncode == 0 or 'actual=%s' % actual not in run.stderr:
                        raise AssertionError('%s %s %s: control did not read texel %s\n%s\n%s'
                                             % (kind, stage, label, actual, run.stdout, run.stderr))
                    if 'debug_layer=enabled' in run.stdout:
                        assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
                    print('%s %s %s: fails as expected, reads %s' % (kind, stage, label, actual), flush=True)


def check_draw_mips(dxc, renderer, adapter, fxc=None):
    # Each fixture loads texel (x, 0) of the slice named by root constant 1 at one mip level; the readback must
    # be that cell. The mutants edit the translated address and must read the cell mip_cell() names for them.
    operation = '--startup-image-2d-array-mips'
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])

    def draw_once(vs, shader, slice_index, expected):
        run = subprocess.run([renderer, adapter, operation, str(vs), str(shader), str(slice_index), expected],
                             capture_output=True, text=True)
        sys.stdout.write(run.stdout)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            sys.exit(77)
        return run

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'fullscreen_mip_%s' % kind, vs_profile, directory)
            translated = {}
            for fixture, (x, mip) in MIP_FIXTURES.items():
                translated[fixture] = translate(mip_words(x, mip)).hlsl_ps()
                name = 'ps_%s_%s' % (fixture.replace(' ', ''), kind)
                shader = compile_hlsl(compiler, kind, translated[fixture], name, ps_profile, directory)
                for slice_index in (0, 1):
                    expected = mip_cell(x, 0, slice_index, mip)
                    run = draw_once(vs, shader, slice_index, expected)
                    if run.returncode:
                        raise AssertionError('%s %s slice=%d\n%s\n%s' % (kind, fixture, slice_index, run.stdout, run.stderr))
                    print('%s load_mip %s slice=%d: %s' % (kind, fixture, slice_index, expected), flush=True)
            for label, fixture, slice_index, address, read in MIP_MUTANTS:
                x, mip = MIP_FIXTURES[fixture]
                source = translated[fixture].replace('int4(v0, v1, v2, v3)', address)
                assert source != translated[fixture], (fixture, label)
                name = 'ps_mutant_%s_%s_%s' % (fixture.replace(' ', ''), label.replace(' ', ''), kind)
                mutant = compile_hlsl(compiler, kind, source, name, ps_profile, directory)
                actual = mip_cell(*read(x, 0, slice_index, mip))
                run = draw_once(vs, mutant, slice_index, mip_cell(x, 0, slice_index, mip))
                if run.returncode == 0 or 'actual=%s' % actual not in run.stderr:
                    raise AssertionError('%s %s %s: control did not read cell %s\n%s\n%s'
                                         % (kind, fixture, label, actual, run.stdout, run.stderr))
                print('%s load_mip %s, %s: fails as expected, reads %s' % (kind, fixture, label, actual), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(args[0], with_mip=args[1:2] == ['mip'])
    elif mode == 'draw':
        controls()
        check_draw(*args)
        check_draw_mips(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
