"""Constant-sampler LOD limits for image_sample, image_sample_l and image_sample_lz in tools/startup/gfx2hlsl.py (#2177).

  controls [form]                      translation controls, pure Python (the default); without a form, also the
                                       compute manifest sampler lines
  binary <llvm-mc>                     reassemble every fixture source and compare it with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given) and draw each form
                                       on its own: the readback must name the mip level the descriptor's limit selects

Semantics. SQ_IMG_SAMP_WORD1 bits 11:0 MIN_LOD and 23:12 MAX_LOD (u4.8, Mesa 24.0.0 src/amd/registers/gfx10-rsrc.json)
clamp the LOD of every sample form, not only of image_sample_b (#2051). A constant sampler of image_sample, _l and _lz now
reaches the static sampler through the same smpl manifest line. The 2D pixel sample and level-zero forms accept a
signed primary bias with linear mip filtering and zero secondary bias. MIN_LOD above MAX_LOD is refused.
A mip filter of 0 (word 2 bits 27:26) is NOT refused for unbiased sampling here
(only the bias form needs it: the bias would count): such a sampler keeps its ordinary smp line, as before. A sampler read
at run time (heap table or the default-sampler shortcut) keeps its ordinary static sampler too, is not validated, and is
logged once per translated module (#2221).

Each form is drawn alone against the 4x4, two-slice, three-mip 2D array of tests/startup_image_3d.cpp
(--startup-image-2d-array-mips) with a point sampler, mip filter "point" (word 2 bits 27:26 = 1), where every
(mip, slice) texel differs. The control of every form is the same program through the ordinary smp line (the
translation before this change), which must read a DIFFERENT mip:

  image_sample_l   explicit LOD 2.0 above MAX_LOD 1:     smpl reads mip 1, smp reads mip 2
  image_sample     MIN_LOD 2 above the implicit LOD 1:   smpl reads mip 2, smp reads mip 1
  image_sample_lz  MIN_LOD 1 above the zero LOD:         smpl reads mip 1, smp reads mip 0

A constant u has no derivatives, so for image_sample the draw replaces only the u initialiser of the translated HLSL by
0.75 + 0.5 * (SV_Position.x - 0.5) (as tests/startup_image_sample_b_check.py does): u changes by 0.5 per pixel, a
footprint of two texels on the 4x4 base level, an implicit LOD of exactly 1. The Sample call is not touched.
"""
import contextlib
import hashlib
import io
import math
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
import startup_image_sample_c_check as C  # noqa: E402
from startup_compute_gather_l_check import COMPUTE_TAIL, compute_prefix, cs_pipe, gather_insn  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, WORDS_2D, assemble, compile_hlsl, root_blob  # noqa: E402
from startup_image_sample_b_check import texel_cell  # noqa: E402

POINT_MIP = 0x4000000  # word 2 bits 27:26 = 1
LINEAR_MIP = 2 << 26
BIAS_LIMITS = 0x200000  # mip range [0, 2]
BIAS_VS = """\
v_mov_b32 v6, 1.0
exp param0 v6, v6, v6, v6
exp pos0 v6, v6, v6, v6 done
s_endpgm
"""
BIAS_PS = """\
s_load_dwordx8 s[32:39], s[8:9], 0x0
s_mov_b32 s52, 0x92
s_mov_b32 s53, 0x200000
s_mov_b32 s54, 0x8003f80
s_mov_b32 s55, 0
s_mov_b32 m0, s10
v_interp_p1_f32_e32 v19, v0, attr0.x
v_interp_p2_f32_e32 v19, v1, attr0.x
v_interp_p1_f32_e32 v20, v0, attr0.y
v_interp_p2_f32_e32 v20, v1, attr0.y
s_waitcnt lgkmcnt(0)
image_sample{suffix} v[24:27], v[19:20], s[32:39], s[52:55] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v24, v25, v26, v27 done vm
s_endpgm
"""
BIAS_VS_WORDS = bytes.fromhex('f2020c7e0f0200f806060606cf0800f806060606000081bf')
BIAS_PS_PREFIX = bytes.fromhex(
    '04080cf4000000faff03b4be92000000ff03b5be00002000ff03b6be803f00088003b7be'
    '0a03fcbe00004cc801004dc8000150c8010151c87fc08cbf'
)
BIAS_SAMPLE = {
    'sample': bytes.fromhex('080f80f01318a801703f8cbf'),
    'lz': bytes.fromhex('080f9cf01318a801703f8cbf'),
}
BIAS_PS_TAIL = bytes.fromhex('0f1800f818191a1b000081bf')


def vcc_sampler_program(shape='a', word0=0x92, word2=0x18500000, coordinate=0x3f000000, extra=b''):
    def sop1(op, dst, src):
        return struct.pack('<I', 0xbe800000 | dst << 16 | op << 8 | src)

    def sop2(op, dst, src0, src1):
        return struct.pack('<I', 0x80000000 | op << 23 | dst << 16 | src1 << 8 | src0)

    if shape == 'a':
        assert 0 <= word0 < 0x8000
        setup = struct.pack('<I', 0xb0000000 | 104 << 16 | word0) + sop2(0x24, 105, 130, 150)
        setup += sop1(4, 48, 106)  # save the incoming VCC mask
        setup += sop1(4, 106, 255) + struct.pack('<I', word2)
    else:
        setup = struct.pack('<IIII', 0xd7610047, 106 | 146 << 9, 0xd7610047, 107 | 147 << 9)
        setup += sop1(4, 106, 255) + struct.pack('<I', word2) + sop2(0x25, 104, 130, 182)
    setup += extra
    program = WORDS_2D[:8] + setup + WORDS_2D[24:]
    program = program.replace(bytes.fromhex('0000403f'), struct.pack('<I', coordinate))
    program = program.replace(bytes.fromhex('0000803e'), struct.pack('<I', 0x3f000000))
    old = bytes.fromhex('080f80f00000c400')
    sample = bytearray(bytes.fromhex('080f9cf00000c400'))
    word = struct.unpack_from('<I', sample, 4)[0]
    struct.pack_into('<I', sample, 4, (word & ~(31 << 21)) | (26 << 21))
    assert program.count(old) == 1
    return program.replace(old, sample)


def vcc_comparison_program():
    def sop1(op, dst, src):
        return struct.pack('<I', 0xbe800000 | dst << 16 | op << 8 | src)
    setup = sop1(3, 106, 255) + struct.pack('<I', 0x18500000)
    setup += sop1(11, 107, 129)  # s_brev_b32 vcc_hi, 1
    setup += struct.pack('<I', 0xb0000000 | 104 << 16 | 0x30b6)
    setup += struct.pack('<I', 0x80000000 | 0x24 << 23 | 105 << 16 | 150 << 8 | 130)
    program = C.WORDS['lz'][:8] + setup + C.WORDS['lz'][28:]
    sample = bytes.fromhex('0801bcf00004c400')
    word = struct.unpack_from('<I', sample, 4)[0]
    return program.replace(sample, sample[:4] + struct.pack('<I', (word & ~(31 << 21)) | 26 << 21))


def vcc_sampler_controls():
    unchanged = stage('lz limits')
    digest = hashlib.sha256((unchanged.hlsl_ps() + '\0' + unchanged.manifest()).encode()).hexdigest()
    assert digest == 'a98b36fe9b4cd302a35025630fc91ebb9088657c6bca923b63b07b75a93f8dbe', digest
    setup = struct.pack('<III', 0xbeea04ff, 0x18500000, 0x7c000000 | 0xc2 << 17 | 256)
    compare_after = WORDS_2D[:24] + setup + WORDS_2D[24:]
    unchanged = G.Pipeline(VS_WORDS, compare_after, root_blob())
    digest = hashlib.sha256((unchanged.hlsl_ps() + '\0' + unchanged.manifest()).encode()).hexdigest()
    assert digest == '5a866b8f985d63b9b846632e1508c978b9bf8db3c9030936b33cf542a0a6bd2b', digest
    for shape, first in (('a', 0x92), ('b', 0)):
        pipe = G.Pipeline(VS_WORDS, vcc_sampler_program(shape), root_blob())
        assert '.SampleLevel(' in pipe.hlsl_ps()
        assert sampler_line(pipe) == 'smpl 0 0x%x 0xc00000 0x18500000 0x0' % first
        assert pipe.ps.sym[106] == ('const', 0x18500000) and pipe.ps.sym[107] == ('const', 0)
    pipe = G.Pipeline(C.VS_WORDS, vcc_comparison_program(), C.root_blob(flags=0))
    assert '.SampleCmpLevelZero(' in pipe.hlsl_ps()
    assert 'smpc 0 0x30b6 0xc00000 0x18500000 0x80000000' in pipe.manifest().splitlines()
    # RDNA 2 ISA, 8.2.7: X/Y border, LESS_EQUAL, bilinear XY, opaque-white border.
    w0, w2, w3 = 0x30b6, 0x18500000, 0x80000000
    assert (w0 & 7, w0 >> 3 & 7, w0 >> 12 & 7, w2 >> 20 & 3,
            w2 >> 22 & 3, w3 >> 30) == (6, 6, 3, 1, 1, 2)
    brev = struct.pack('<I', 0xbe800000 | 107 << 16 | 11 << 8 | 129)
    bfm = struct.pack('<I', 0x80000000 | 0x24 << 23 | 107 << 16 | 159 << 8 | 129)
    mov = struct.pack('<II', 0xbe800000 | 107 << 16 | 3 << 8 | 255, 0x80000000)
    for replacement in (bfm, mov):
        code = vcc_comparison_program().replace(brev, replacement)
        pipe = G.Pipeline(C.VS_WORDS, code, C.root_blob(flags=0))
        assert 'smpc 0 0x30b6 0xc00000 0x18500000 0x80000000' in pipe.manifest().splitlines()
    mov64 = struct.pack('<II', 0xbeea04ff, 0x18500000)
    inline_float = vcc_sampler_program().replace(mov64, struct.pack('<I', 0xbeea04f2))
    probe = G.Stage.__new__(G.Stage)
    probe.ins = G.decode(inline_float)
    sample_pc = next(off for off, _, name, _ in probe.ins if name == 'image_sample_lz')
    assert probe.literal_vcc_sampler_words(sample_pc) == (0x92, 0xc00000, 0, 0x3ff00000)
    assert sampler_line(G.Pipeline(VS_WORDS, inline_float, root_blob())) == 'smp 0 0x92 0xc00000 0x0 0x3ff00000'
    literal_float = vcc_sampler_program().replace(mov64, struct.pack('<II', 0xbeea04ff, 0x3f800000))
    assert sampler_line(G.Pipeline(VS_WORDS, literal_float, root_blob())) == 'smpl 0 0x92 0xc00000 0x3f800000 0x0'
    inline_negative = vcc_sampler_program().replace(mov64, struct.pack('<I', 0xbeea04c1))
    probe.ins = G.decode(inline_negative)
    sample_pc = next(off for off, _, name, _ in probe.ins if name == 'image_sample_lz')
    assert probe.literal_vcc_sampler_words(sample_pc) == (0x92, 0xc00000, 0xffffffff, 0xffffffff)
    bfm64 = struct.pack('<I', 0x80000000 | 0x25 << 23 | 106 << 16 | 190 << 8 | 129)
    pipe = G.Pipeline(VS_WORDS, vcc_sampler_program().replace(mov64, bfm64), root_blob())
    assert sampler_line(pipe) == 'smp 0 0x92 0xc00000 0x0 0x40000000'
    movk_and_hi = struct.pack('<II', 0xb0000000 | 106 << 16, 0xbe800000 | 107 << 16 | 3 << 8 | 128)
    pipe = G.Pipeline(VS_WORDS, vcc_sampler_program().replace(mov64, movk_and_hi), root_blob())
    assert sampler_line(pipe) == 'smp 0 0x92 0xc00000 0x0 0x0'
    # Both edges reach the same VCC constant; the second fixture bypasses its definition.
    branch = struct.pack('<III', 0xbf008080, 0xbf840001, 0xbf800000)
    pipe = G.Pipeline(VS_WORDS, vcc_sampler_program(extra=branch), root_blob())
    assert sampler_line(pipe) == 'smpl 0 0x92 0xc00000 0x18500000 0x0'
    unjoined = vcc_sampler_program()
    position = unjoined.index(bytes.fromhex('ff04eabe'))
    unjoined = (unjoined[:position] + struct.pack('<III', 0x7c000000 | 0xc2 << 17 | 256,
                                                  0xbf008080, 0xbf840002) + unjoined[position:])
    code = vcc_sampler_program()
    end_sample = code.index(bytes.fromhex('080f9cf0')) + 8
    code = code[:end_sample] + struct.pack('<I', 0x7c000000 | 0xc2 << 17 | 256) + code[end_sample:]
    pipe = G.Pipeline(VS_WORDS, code, root_blob())
    assert 106 not in pipe.ps.sym and 107 not in pipe.ps.sym
    literal_minus_one = vcc_sampler_program().replace(mov64, struct.pack('<II', 0xbeea04ff, 0xffffffff))
    used_spill = vcc_sampler_program('b')
    position = used_spill.index(bytes.fromhex('080f9cf0')) + 8
    used_spill = used_spill[:position] + struct.pack('<I', 0x7e000200 | 327) + used_spill[position:]
    for code, reason in ((vcc_sampler_program(word2=0x98500000), 's_mov_b64 vcc literal bit 31'),
                         (literal_minus_one, 's_mov_b64 vcc literal bit 31'),
                         (vcc_sampler_program(extra=struct.pack('<I', 0x7c000000 | 0xc2 << 17 | 256)),
                          'descriptor operand s[104:107] holds'),
                         (unjoined, 'descriptor operand overlaps s106/s107 without a matching load on every path'),
                         (used_spill, 'v_writelane_b32 source vcc_lo is a lane mask or m0')):
        try:
            G.Pipeline(VS_WORDS, code, root_blob())
        except G.Unsupported as error:
            assert reason in str(error), str(error)
        else:
            raise AssertionError('unproved VCC sampler accepted')
    original = G.Stage.literal_vcc_sampler_words
    def wrong_filter(stage, at=None):
        words = original(stage, at)
        return words[:2] + (words[2] & ~0x500000, words[3]) if words else None
    G.Stage.literal_vcc_sampler_words = wrong_filter
    try:
        pipe = G.Pipeline(VS_WORDS, vcc_sampler_program(), root_blob())
        assert sampler_line(pipe) == 'smpl 0 0x92 0xc00000 0x18000000 0x0'
        assert vcc_sampler_reference(0x92, 0x18500000, 0.5, 0.5) != vcc_sampler_reference(0x92, 0x18000000, 0.5, 0.5)
    finally:
        G.Stage.literal_vcc_sampler_words = original


def vcc_sampler_reference(word0, word2, u, v):
    # RDNA 2 ISA, 8.2.7 Image Sampler: clamp modes and XY filter fields.
    modes = (word0 & 7, (word0 >> 3) & 7)
    linear = ((word2 >> 20) & 3) == 1 and ((word2 >> 22) & 3) == 1
    assert all(mode in (0, 2) for mode in modes)
    def address(index, mode):
        return index % 2 if mode == 0 else max(0, min(1, index))
    def weights(coord, mode):
        if not linear:
            return [(address(math.floor(coord * 2), mode), 1.0)]
        position = coord * 2 - 0.5
        low = math.floor(position)
        return [(address(low, mode), 1 - (position - low)), (address(low + 1, mode), position - low)]
    color = [0.0] * 4
    for x, wx in weights(u, modes[0]):
        for y, wy in weights(v, modes[1]):
            texel = ((1 + x + 2 * y) / 8, float(x), float(y), 0.0)
            for c in range(4):
                color[c] += texel[c] * wx * wy
    return ','.join('%08x' % struct.unpack('<I', struct.pack('<f', channel))[0] for channel in color)


def bias_code(forms, word2=LINEAR_MIP | 0x3f80, word1=BIAS_LIMITS):
    prefix = BIAS_PS_PREFIX.replace(bytes.fromhex('803f0008'), word2.to_bytes(4, 'little'))
    prefix = prefix[:20] + word1.to_bytes(4, 'little') + prefix[24:]
    return prefix + b''.join(BIAS_SAMPLE[form] for form in forms) + BIAS_PS_TAIL


def bias_stage(form, word2, word1=BIAS_LIMITS):
    return G.Pipeline(BIAS_VS_WORDS, bias_code((form,), word2, word1), root_blob())


def bias_controls():
    # RDNA 2 ISA, S_BFM_B64, section 7: a 64-bit mask can build two sampler words.
    instruction = struct.pack('<I', 0x80000000 | 0x25 << 23 | 52 << 16 | 182 << 8 | 130)
    program = bias_code(('lz',))
    program = program[:8] + instruction + program[24:]
    program = program.replace(bytes.fromhex('803f0008'), bytes.fromhex('00005018'))
    assert any(asm == 's_bfm_b64 s[52:53], 2, 54' for _, _, _, asm in G.decode(program))
    mask = ((1 << 2) - 1) << 54
    words = (mask & 0xffffffff, mask >> 32, 0x18500000, 0)
    assert words == (0, 0xc00000, 0x18500000, 0)
    assert ((1 << 2) - 1) << 22 != mask  # A 32-bit shift is a discriminating mutant.
    pipe = G.Pipeline(BIAS_VS_WORDS, program, root_blob())
    assert '.SampleLevel(' in pipe.hlsl_ps()
    assert sampler_line(pipe) == 'smpl 0 ' + ' '.join(hex(word) for word in words)

    for form, name, call in (('sample', 'image_sample', '.Sample'), ('lz', 'image_sample_lz', '.SampleLevel')):
        for bias, field in ((-128, 0x3f80), (128, 0x80), (0, 0), (-4096, 0x3000), (4093, 0xffd)):
            pipe = bias_stage(form, LINEAR_MIP | field)
            assert call + '(q0,' in pipe.hlsl_ps()
            expected_word = LINEAR_MIP | (field if form == 'sample' else 0)
            assert sampler_line(pipe) == 'smpl 0 0x92 0x200000 0x%x 0x0' % expected_word, (form, bias)
        for word2, word1, diagnostic in (
            (LINEAR_MIP | 0x7f80, BIAS_LIMITS, 'with a sampler LOD bias (word 2 0x8007f80)'),
            (LINEAR_MIP | 0xffe, BIAS_LIMITS, 'outside the D3D12 range'),
            (LINEAR_MIP | 0x2fff, BIAS_LIMITS, 'outside the D3D12 range'),
            (POINT_MIP | 0x3f80, BIAS_LIMITS, 'with a sampler LOD bias'),
            (LINEAR_MIP | 0x3f80, 0x100200, 'with sampler MIN_LOD 0x200 above MAX_LOD 0x100'),
        ):
            try:
                bias_stage(form, word2, word1)
            except G.Unsupported as error:
                assert str(error).startswith(name + ' ') and diagnostic in str(error), str(error)
            else:
                raise AssertionError((form, word1, word2, 'unexpectedly translated'))
    shared = G.Pipeline(BIAS_VS_WORDS, bias_code(('sample', 'lz')), root_blob())
    ps = shared.hlsl_ps()
    assert '.Sample(q0,' in ps and '.SampleLevel(q1,' in ps
    assert [line for line in shared.manifest().splitlines() if line.startswith('smpl ')] == [
        'smpl 0 0x92 0x200000 0x8003f80 0x0',
        'smpl 1 0x92 0x200000 0x8000000 0x0',
    ]
    print('2D inline sampler bias: separate bindings, signed limits, secondary bias, mip filter and LOD clamp controls pass')


PROGRAM = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, {w1}
s_mov_b32 s26, {w2}
s_mov_b32 s27, 0
v_mov_b32 v0, 0x3f400000
v_mov_b32 v1, 0x3e800000
v_mov_b32 v2, s9
v_mov_b32 v3, 0x40000000
s_waitcnt lgkmcnt(0)
{instruction}
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
# The sampler words come from a run-time load of the sampler heap at the byte offset in root constant 1 (s9).
HEAP_PROGRAM = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_load_dwordx4 s[24:27], s[10:11], s9
v_mov_b32 v0, 0x3f400000
v_mov_b32 v1, 0x3e800000
v_mov_b32 v2, 0
v_mov_b32 v3, 0x40000000
s_waitcnt lgkmcnt(0)
{instruction}
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
TAIL = 's[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY'
# form: (instruction, manifest/HLSL name, MIN_LOD and MAX_LOD word the draw uses, mip the limits read, mip an
# unlimited sampler reads)
FORMS = {
    'l': ('image_sample_l v[4:7], v[0:3], ' + TAIL, 'image_sample_l', 0x100000, 1, 2),
    'sample': ('image_sample v[4:7], v[0:2], ' + TAIL, 'image_sample', 0x200200, 2, 1),
    'lz': ('image_sample_lz v[4:7], v[0:2], ' + TAIL, 'image_sample_lz', 0x200100, 1, 0),
}
# variant: (word 1 as a function of the form's limit word, word 2)
VARIANTS = {
    'limits': (lambda w1: w1, POINT_MIP),
    'bias': (lambda w1: w1, POINT_MIP | 1),
    'bias sec': (lambda w1: w1, POINT_MIP | 0x4000),
    'range': (lambda w1: 0x200300, POINT_MIP),  # MIN_LOD 3 above MAX_LOD 2
    'mip filter none': (lambda w1: w1, 0),
}
CALLS = {  # the one texture call each form lowers to (kind 2D array, so the slice is the third component)
    'l': 'r0.SampleLevel(q0, float3(F(v0), F(v1), F(v2)), F(v3))',
    'sample': 'r0.Sample(q0, float3(F(v0), F(v1), F(v2)))',
    'lz': 'r0.SampleLevel(q0, float3(F(v0), F(v1), F(v2)), 0)',
}
U_INIT = 'v0 = 0x3f400000u;'
U_DERIVED = 'v0 = U(0.75 + 0.5 * (i.pos.x - 0.5));'


def fixtures():
    out = {}
    for form, (instruction, _, w1, _, _) in FORMS.items():
        for variant, (words1, w2) in VARIANTS.items():
            out['%s %s' % (form, variant)] = PROGRAM.format(instruction=instruction, w1='0x%x' % words1(w1), w2='0x%x' % w2)
        out['%s heap' % form] = HEAP_PROGRAM.format(instruction=instruction)
    return out


# BEGIN WORDS
WORDS = {
    'l limits': (
        '04040cf4000000fa800398beff0399be00001000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f90f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'l bias': (
        '04040cf4000000fa800398beff0399be00001000ff039abe0100000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f90f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'l bias sec': (
        '04040cf4000000fa800398beff0399be00001000ff039abe0040000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f90f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'l range': (
        '04040cf4000000fa800398beff0399be00032000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f90f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'l mip filter none': (
        '04040cf4000000fa800398beff0399be0000100080039abe80039bbeff02007e'
        '0000403fff02027e0000803e0902047ef402067e7fc08cbf280f90f00004c400'
        '703f8cbf0f1800f804050607000081bf'
    ),
    'l heap': (
        '04040cf4000000fa050608f400000012ff02007e0000403fff02027e0000803e'
        '8002047ef402067e7fc08cbf280f90f00004c400703f8cbf0f1800f804050607'
        '000081bf'
    ),
    'sample limits': (
        '04040cf4000000fa800398beff0399be00022000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f80f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'sample bias': (
        '04040cf4000000fa800398beff0399be00022000ff039abe0100000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f80f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'sample bias sec': (
        '04040cf4000000fa800398beff0399be00022000ff039abe0040000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f80f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'sample range': (
        '04040cf4000000fa800398beff0399be00032000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f80f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'sample mip filter none': (
        '04040cf4000000fa800398beff0399be0002200080039abe80039bbeff02007e'
        '0000403fff02027e0000803e0902047ef402067e7fc08cbf280f80f00004c400'
        '703f8cbf0f1800f804050607000081bf'
    ),
    'sample heap': (
        '04040cf4000000fa050608f400000012ff02007e0000403fff02027e0000803e'
        '8002047ef402067e7fc08cbf280f80f00004c400703f8cbf0f1800f804050607'
        '000081bf'
    ),
    'lz limits': (
        '04040cf4000000fa800398beff0399be00012000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f9cf0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'lz bias': (
        '04040cf4000000fa800398beff0399be00012000ff039abe0100000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f9cf0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'lz bias sec': (
        '04040cf4000000fa800398beff0399be00012000ff039abe0040000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f9cf0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'lz range': (
        '04040cf4000000fa800398beff0399be00032000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f9cf0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'lz mip filter none': (
        '04040cf4000000fa800398beff0399be0001200080039abe80039bbeff02007e'
        '0000403fff02027e0000803e0902047ef402067e7fc08cbf280f9cf00004c400'
        '703f8cbf0f1800f804050607000081bf'
    ),
    'lz heap': (
        '04040cf4000000fa050608f400000012ff02007e0000403fff02027e0000803e'
        '8002047ef402067e7fc08cbf280f9cf00004c400703f8cbf0f1800f804050607'
        '000081bf'
    ),
}
# END WORDS


def stage(name, sampler_heap=None):
    if name.endswith(' heap'):
        return G.Pipeline(H.VS_WORDS, bytes.fromhex(WORDS[name]), H.root_blob(), sampler_heap=sampler_heap)
    return G.Pipeline(VS_WORDS, bytes.fromhex(WORDS[name]), root_blob())


def sampler_line(pipe):
    lines = [line for line in pipe.manifest().splitlines() if line.startswith(('smp ', 'smpl '))]
    assert len(lines) == 1, pipe.manifest()
    return lines[0]


def refused(name, text):
    try:
        stage(name).hlsl_ps()
    except G.Unsupported as error:
        assert str(error) == text, (name, str(error))
    else:
        raise AssertionError('%s was accepted' % name)


def controls(only=None):
    assert WORDS, 'no recorded words'
    if only is None:
        bias_controls()
        vcc_sampler_controls()
        assert vcc_sampler_reference(0x92, 0x18500000, 0.5, 0.5) != vcc_sampler_reference(0x92, 0x18000000, 0.5, 0.5)
        assert vcc_sampler_reference(0x92, 0x18500000, 1.25, 0.5) != vcc_sampler_reference(0, 0x18500000, 1.25, 0.5)
    for form, (_, op, w1, mip, plain_mip) in FORMS.items():
        if only and form != only:
            continue
        name = '%s limits' % form
        pipe = stage(name)
        ps = pipe.hlsl_ps()
        # The call is the form's own and the sampler carries the limits (manifest line smpl).
        assert CALLS[form] in ps, (name, ps)
        line = 'smpl 0 0x0 0x%x 0x%x 0x0' % (w1, POINT_MIP)
        assert line in pipe.manifest().splitlines() and 'smp ' not in pipe.manifest(), (name, pipe.manifest())
        assert not pipe.lod_unvalidated and not pipe.notes, (name, pipe.lod_unvalidated)
        assert mip != plain_mip
        # Refusals use the validation shared with image_sample_b, under this form's name.
        refused('%s bias' % form, '%s with a sampler LOD bias (word 2 %#x)' % (op, POINT_MIP | 1))
        refused('%s bias sec' % form, '%s with a sampler LOD bias (word 2 %#x)' % (op, POINT_MIP | 0x4000))
        refused('%s range' % form, '%s with sampler MIN_LOD 0x300 above MAX_LOD 0x200' % op)
        # A mip filter of 0 keeps today's ordinary sampler: no smpl line, nothing refused.
        none = stage('%s mip filter none' % form)
        none.hlsl_ps()
        assert 'smp 0 0x0 0x%x 0x0 0x0' % w1 in none.manifest().splitlines() and 'smpl' not in none.manifest(), none.manifest()
        # A sampler from the heap (table or default-sampler shortcut) is still accepted, on the ordinary path, and
        # the module logs once that its LOD fields are not validated.
        for heap in (True, False):
            heaped = stage('%s heap' % form, sampler_heap=heap)
            heaped.hlsl_ps()
            manifest = heaped.manifest()
            assert 'smpl' not in manifest and heaped.lod_unvalidated == {op}, (form, heap, manifest, heaped.lod_unvalidated)
            assert ('smpheap' in manifest) == heap, (form, heap, manifest)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                G.log_unvalidated_lod(0x1234, heaped)
            text = stderr.getvalue()
            assert text.count('\n') == 1 and op in text and '#2221' in text and '0000000000001234' in text, text
        clean = io.StringIO()
        with contextlib.redirect_stderr(clean):
            G.log_unvalidated_lod(0x1234, pipe)
        assert not clean.getvalue(), clean.getvalue()
        print('%s: smpl line and call; refused: LOD bias, MIN_LOD > MAX_LOD; mip filter 0 stays smp; heap accepted and '
              'logged once' % op)
    # The log is written by the translation entry point, once per module, and only for a heap sampler.
    for name, logged in (('sample heap', True), ('sample limits', False)):
        vs, root = (H.VS_WORDS, H.root_blob()) if logged else (VS_WORDS, root_blob())
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = G.translate_to(tmp, 0x42, vs, bytes.fromhex(WORDS[name]), root, check=False)
        assert result == 'ok', (name, result)
        assert stderr.getvalue().count('image_sample through a sampler heap') == int(logged), (name, stderr.getvalue())
    print('translate_to logs the unvalidated heap sampler once, and nothing for a constant sampler')
    if only:
        return
    check_compute_manifest()
    # The three forms differ from each other and from the ordinary line, and the model has distinct mips per form.
    assert len({texel_cell(m, s) for m in range(3) for s in range(2)}) == 6
    assert len({tuple(FORMS[f][3:5]) for f in FORMS}) == 3
    print('draw model: every form reads its limit mip and its control reads another')


def check_compute_manifest():
    """The compute manifest names the same three sampler lines as the graphics one (#2463).

    The programs reuse the helpers of startup_compute_gather_l_check: its compute prefix (sampler words in s24-s27,
    s24 = 0x12), and gather_insn with a 2D sample opcode in place of the gather one. The binary mode of this check
    does not reassemble them with llvm-mc; each case asserts the sample call the translator decodes."""
    for form, opcode in (('lz', 0x27), ('l', 0x24)):
        for variant, w2, tag in (('limits', 0x4500000, 'smpl'), ('mip filter none', 0x500000, 'smp')):
            pipe = cs_pipe(compute_prefix(w2=w2) + gather_insn(0xf, 1, opcode) + COMPUTE_TAIL)
            assert '.SampleLevel(q0,' in pipe.hlsl_cs(), (form, variant)
            lines = [line for line in pipe.manifest().splitlines() if line.startswith('smp')]
            assert lines == ['%s 0 0x12 0xfff000 0x%x 0x0' % (tag, w2)], (form, variant, lines)
    pipe = cs_pipe(compute_prefix() + gather_insn(1, 1, 0x2f) + COMPUTE_TAIL)
    assert '.SampleCmpLevelZero(q0,' in pipe.hlsl_cs()
    lines = [line for line in pipe.manifest().splitlines() if line.startswith('smp')]
    assert lines == ['smpc 0 0x12 0xfff000 0x4500000 0x0'], lines
    print('compute manifest: smpl with LOD limits (lz, l); controls: smp with mip filter 0, smpc for comparison')


def check_binary(llvm_mc):
    assert assemble(llvm_mc, BIAS_VS) == BIAS_VS_WORDS
    for name, suffix in (('sample', ''), ('lz', '_lz')):
        assert assemble(llvm_mc, BIAS_PS.format(suffix=suffix)) == bias_code((name,)), name
    for name, source in fixtures().items():
        built = assemble(llvm_mc, source)
        if built != bytes.fromhex(WORDS[name]):
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), WORDS[name]))
    assert set(WORDS) == set(fixtures()), 'recorded and generated fixtures differ'
    print('fixture words match llvm-mc')


def drawable(form, hlsl):
    """The translated stage; for the implicit form, u made to change by 0.5 per pixel (only its initialiser)."""
    if form != 'sample':
        return hlsl
    assert hlsl.count(U_INIT) == 1, hlsl
    return hlsl.replace(U_INIT, U_DERIVED)


def check_draw(dxc, renderer, adapter, fxc=None):
    operation = '--startup-image-2d-array-mips'
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])

    def draw_once(vs, shader, slice_index, expected, sampler):
        run = subprocess.run([renderer, adapter, operation, str(vs), str(shader), str(slice_index), expected, sampler],
                             capture_output=True, text=True)
        sys.stdout.write(run.stdout)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            sys.exit(77)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
        return run

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'fullscreen_lod_%s' % kind, vs_profile, directory)
            for word0, word2, coordinate, bits in ((0x92, 0x18500000, 0.5, 0x3f000000),
                                                    (0x92, 0x18000000, 0.5, 0x3f000000),
                                                    (0x92, 0x18500000, 1.25, 0x3fa00000),
                                                    (0, 0x18500000, 1.25, 0x3fa00000)):
                pipe = G.Pipeline(VS_WORDS, vcc_sampler_program(word0=word0, word2=word2, coordinate=bits), root_blob())
                shader = compile_hlsl(compiler, kind, pipe.hlsl_ps(),
                                      'ps_vcc_%x_%x_%x_%s' % (word0, word2, bits, kind), ps_profile, directory)
                expected = vcc_sampler_reference(word0, word2, coordinate, 0.5)
                run = subprocess.run([renderer, adapter, '--startup-image-2d', str(vs), str(shader), '0', expected,
                                      sampler_line(pipe)], capture_output=True, text=True)
                sys.stdout.write(run.stdout)
                if run.returncode == 77:
                    sys.exit(77)
                if run.returncode:
                    raise AssertionError('VCC sampler draw: %s\n%s' % (run.stdout, run.stderr))
                if 'debug_layer=enabled' in run.stdout:
                    assert 'debug_errors=0 debug_warnings=0' in run.stdout
            for form, (_, op, w1, mip, plain_mip) in FORMS.items():
                pipe = stage('%s limits' % form)
                shader = compile_hlsl(compiler, kind, drawable(form, pipe.hlsl_ps()), 'ps_lod_%s_%s' % (form, kind),
                                      ps_profile, directory)
                line = sampler_line(pipe)
                assert line.startswith('smpl '), line
                for slice_index in (0, 1):
                    expected = texel_cell(mip, slice_index)
                    run = draw_once(vs, shader, slice_index, expected, line)
                    if run.returncode:
                        raise AssertionError('%s %s slice=%d\n%s\n%s' % (kind, op, slice_index, run.stdout, run.stderr))
                    print('%s %s slice=%d reads mip %d: %s' % (kind, op, slice_index, mip, expected), flush=True)
                # Control: the same words through the ordinary smp line (limits not carried) read another mip.
                plain = 'smp ' + line[len('smpl '):]
                for slice_index in (0, 1):
                    run = draw_once(vs, shader, slice_index, texel_cell(mip, slice_index), plain)
                    if run.returncode == 0 or 'actual=%s' % texel_cell(plain_mip, slice_index) not in run.stderr:
                        raise AssertionError('%s %s: smp control did not read mip %d\n%s\n%s'
                                             % (kind, op, plain_mip, run.stdout, run.stderr))
                print('%s %s: plain smp control fails as expected, reads mip %d' % (kind, op, plain_mip), flush=True)


def main():
    args = sys.argv[1:]
    if not args or args[0] == 'controls':
        controls(args[1] if len(args) > 1 else None)
    elif args[0] == 'binary' and len(args) == 2:
        check_binary(args[1])
    elif args[0] == 'draw' and len(args) in (4, 5):
        controls()
        check_draw(*args[1:])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
