"""Fixed-mip 2D comparison gathers (image_gather4_c_lz) in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation controls, pure Python
  --llvm-mc <llvm-mc>                  reassemble the fixture sources and compare them with the words below
  --draw <dxc> <renderer> <adapter> [fxc]
                                       translate, compile (DXC ps_6_0 and, when given, FXC ps_5_1) and draw
  --evidence <dir>                     keep the compiled shaders, cases, manifests and logs for the draw

Each fixture is a source-built gfx1013 pixel stage. It loads a texture descriptor from table 0, builds a
constant comparison sampler in s[24:27] (clamp to the last texel, bilinear XY, point mip, no bias) and runs
one image_gather4_c_lz on a texture from descriptor table 0. The address dwords of a comparison gather are
the reference, then u and v, so v[0:2] compares v0 at (v1, v2). The four taps are compared separately and
written to v4..v7 in the documented order (lower left, lower right, upper right, upper left).

The draw covers an 8x8 target and up to three view mips. The comparison function and filter come from the
guest sampler words, which tests/offscreen_check binds as a D3D12 comparison static sampler; the view is a
single-component R32_FLOAT texture, the format that advertises SHADER_GATHER_COMPARISON (RGBA32_FLOAT does
not). The CPU expectation is computed from the Direct3D gather4_c rule independently of the emitted HLSL.
"""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from pixel_compare_check import COMPARE, passes  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import mov_literal  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, CompileQueue, compile_hlsl  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, assemble, root_blob  # noqa: E402

SIDE = 8
CLAMP_LAST_TEXEL = 2
BILINEAR_POINT_MIP_W2 = (1 << 20) | (1 << 22) | (1 << 26)
LESS = COMPARE['LESS']
PROFILE = ('Unsupported gather sampler: requires normalized clamp, bilinear XY, point mip and no LOD '
           'modifiers')

SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, {w0:#x}
s_mov_b32 s25, {w1:#x}
s_mov_b32 s26, {w2:#x}
s_mov_b32 s27, 0
v_mov_b32 v0, 0.5
v_mov_b32 v1, 0x3e800000
v_mov_b32 v2, 0x3f800000
s_waitcnt lgkmcnt(0)
image_gather4_c_lz v[4:7], v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
PREFIX = bytes.fromhex(
    '04040cf4000000fa'
    'ff0398be12100000'
    '800399be'
    'ff039abe00005004'
    '80039bbe'
    'f002007e'
    'ff02027e0000803e'
    'f202047e'
    '7fc08cbf')
SUFFIX = bytes.fromhex('703f8cbf0f1800f804050607000081bf')
# The four s_mov_b32 instructions that build the sampler descriptor, with their default words.
# A zero word uses the four-byte SOP1 form with the inline constant; other words use a literal.
SAMPLER_MOVES = (
    bytes.fromhex('ff0398be12100000'),  # s24 = 0x1012: clamp X/Y, normalized, TRUNC 0, comparison LESS
    bytes.fromhex('800399be'),          # s25 = 0
    bytes.fromhex('ff039abe00005004'),  # s26 = 0x04500000: bilinear XY, point mip, no bias
    bytes.fromhex('80039bbe'),          # s27 = 0
)
DEFAULT_WORDS = (0x1012, 0, 0x04500000, 0)
GATHER_WORD = (0xF13C0108, 0x00E40400)    # image_gather4_c_lz at s[28:31]
ORDINARY_WORD = (0xF11C0108, 0x00C40400)  # image_gather4_lz at s[24:27]


def smov(sreg, word):
    """s_mov_b32 s<reg>, <word> as llvm-mc writes it: inline zero or a literal."""
    if word == 0:
        return struct.pack('<I', 0xBE800000 | (sreg << 16) | (3 << 8) | 0x80)
    return struct.pack('<II', 0xBE800000 | (sreg << 16) | (3 << 8) | 0xFF, word)


def with_words(word0=None, word1=None, word2=None, word3=None, prefix=PREFIX):
    values = list(DEFAULT_WORDS)
    for index, value in enumerate((word0, word1, word2, word3)):
        if value is not None:
            values[index] = value
    result = prefix
    for index, (marker, value) in enumerate(zip(SAMPLER_MOVES, values)):
        assert result.count(marker) == 1, marker.hex()
        result = result.replace(marker, smov(24 + index, value))
    return result


def shader(mask=1, dim=1, opcode=0x4F, flags=0, high=0, prefix=None, nsa=False, dst=4, addr=0,
           srsrc=16, ssamp=24):
    first = 0xF0000000 | (opcode << 18) | (mask << 8) | (dim << 3) | flags
    if nsa:
        first |= 2
    extension = bytes((addr + 1, addr + 2, 0, 0)) if nsa else b''
    second = addr | (dst << 8) | high | ((srsrc // 4) << 16) | ((ssamp // 4) << 21)
    return (PREFIX if prefix is None else prefix) + struct.pack('<II', first, second) + extension + SUFFIX


# The three input moves the draw rewrites: v0 is the comparison reference, v1 = u, v2 = v.
V0_MOVE = bytes.fromhex('f002007e')            # v_mov_b32 v0, 0.5 (inline)
V1_MOVE = bytes.fromhex('ff02027e0000803e')    # v_mov_b32 v1, 0x3e800000
V2_MOVE = bytes.fromhex('f202047e')            # v_mov_b32 v2, 1.0 (inline)


def fbits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def fixture(reference, u, v, word0):
    """The fixture with its reference and normalized coordinates encoded for the draw."""
    prefix = with_words(word0=word0)
    for marker, register, value in ((V0_MOVE, 0, reference), (V1_MOVE, 1, u), (V2_MOVE, 2, v)):
        assert prefix.count(marker) == 1, marker.hex()
        prefix = prefix.replace(marker, mov_literal(register, fbits(value)))
    return shader(prefix=prefix)


def pipeline(code):
    return G.Pipeline(VS_WORDS, code, root_blob(flags=0))


def refused(code, reason):
    try:
        pipeline(code)
    except G.Unsupported as error:
        assert reason in str(error), (reason, str(error))
    except Exception as error:  # noqa: BLE001 - a different exception is a failure of the control
        raise AssertionError('refusal raised %s: %s' % (type(error).__name__, error))
    else:
        raise AssertionError('comparison gather accepted a form expected to report %r' % reason)


class Probe(G.Stage):
    """A stage with no pipeline: enough state for the instruction handler to reject operands."""

    def __init__(self):
        self.kind = 'ps'
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.exec_known = set(), set(), True


def handler_refused(operands, reason):
    text = 'image_gather4_c_lz ' + operands
    try:
        Probe().one('image_gather4_c_lz', text)
    except G.Unsupported as error:
        assert reason in str(error), (reason, str(error))
    else:
        raise AssertionError('handler accepted %r' % text)


def load_mutant(mutations):
    """tools/startup/gfx2hlsl.py with the named guard text replaced, for a control."""
    source = (ROOT / 'tools' / 'startup' / 'gfx2hlsl.py').read_text(encoding='utf-8')
    for old, new in mutations:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    module = types.ModuleType('gfx2hlsl_mutant')
    module.__file__ = str(ROOT / 'tools' / 'startup' / 'gfx2hlsl.py')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def mutated_pipeline(module, code):
    return module.Pipeline(VS_WORDS, code, root_blob(flags=0))


def controls(llvm_mc):
    # The recorded bytes are what llvm-mc 15 assembles for the documented source.
    if llvm_mc:
        assert assemble(llvm_mc, SOURCE.format(w0=DEFAULT_WORDS[0], w1=0, w2=DEFAULT_WORDS[2])) == shader(), \
            'base fixture bytes differ from llvm-mc'
        for name, words in (('LESS', (0x1012, 0, BILINEAR_POINT_MIP_W2, 0)),
                            ('GREATER', (0x4012, 0, BILINEAR_POINT_MIP_W2, 0)),
                            ('MAX_LOD', (0x1012, 0xFFF000, BILINEAR_POINT_MIP_W2, 0))):
            built = assemble(llvm_mc, SOURCE.format(w0=words[0], w1=words[1], w2=words[2]))
            assert built == shader(prefix=with_words(*words)), '%s fixture bytes differ from llvm-mc' % name
        nsa_source = SOURCE.format(w0=0x1012, w1=0, w2=BILINEAR_POINT_MIP_W2).replace(
            'image_gather4_c_lz v[4:7], v[0:2],', 'image_gather4_c_lz v[4:7], [v0, v1, v2],')
        assert assemble(llvm_mc, nsa_source) == shader(nsa=True), 'NSA fixture bytes differ from llvm-mc'
        red_source = SOURCE.format(w0=0x1012, w1=0, w2=BILINEAR_POINT_MIP_W2).replace(
            'dmask:0x1', 'dmask:0x2')
        assert assemble(llvm_mc, red_source) == shader(mask=2), 'red-channel fixture bytes differ from llvm-mc'
        print('recorded words match llvm-mc: base, LESS, GREATER, MAX_LOD, NSA and green-dmask fixtures')

    # Decoding uses the real startup decoder on the recorded bytes: four destination VGPRs, three
    # address VGPRs (reference, u, v), eight descriptor SGPRs, four sampler SGPRs, red DMASK, 2D.
    assert fixture(Fraction(1, 2), Fraction(1, 4), Fraction(1), DEFAULT_WORDS[0]) == shader(), \
        'the default draw inputs must reproduce the recorded fixture'
    decoded = [row for row in G.decode(G.strip_header(shader())) if row[2].startswith('image_')]
    assert decoded == [(52, 8, 'image_gather4_c_lz',
                        'image_gather4_c_lz v[4:7], v[0:2], s[16:23], s[24:27] '
                        'dmask:0x1 dim:SQ_RSRC_IMG_2D')], decoded
    print('decode: ' + decoded[0][3])

    pipe = pipeline(shader())
    text, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'Texture2D<float4> r0 : register(t0);' in text, text
    assert 'SamplerComparisonState q0 : register(s0);' in text, text
    assert 't = r0.GatherCmp(q0, float2(F(v1), F(v2)), F(v0));' in text, text
    for register, lane in zip('4567', 'xyzw'):
        assert 'v%s = U(t.%s);' % (register, lane) in text, text
    assert 'srv 0 tex2d table 0 0\n' in manifest, manifest
    assert 'smpc 0 0x1012 0x0 0x4500000 0x0\n' in manifest, manifest
    assert 'smp 0 ' not in manifest, manifest
    print('base: GatherCmp(q0, (u, v), reference) to v4..v7; manifest srv tex2d + smpc 0x1012')

    # Every destination position needs its own result sequence over the positive matrix; otherwise a
    # swap between two positions produces the same image in every case. Before the middle-tap split the
    # lower-right and upper-right positions shared one sequence, so this requirement failed.
    signatures = lane_signatures(cases())
    assert len(set(signatures)) == 4, 'the case matrix must distinguish all four destinations: %r' % (signatures,)
    original = lane_signatures([case for case in cases() if case[0] != 'split middle LESS'])
    assert len(set(original)) == 3, 'the pre-hardening matrix is expected to merge taps 1 and 2: %r' % (original,)
    print('lane signatures: 4/4 destinations distinct (3/4 without the middle-tap split)')

    # All eight comparison functions travel in word 0 bits 14:12 and keep the same call.
    for function, code in COMPARE.items():
        word0 = CLAMP_LAST_TEXEL | (CLAMP_LAST_TEXEL << 3) | (code << 12)
        pipe = pipeline(shader(prefix=with_words(word0=word0)))
        assert 'smpc 0 0x%x 0x0 0x4500000 0x0' % word0 in pipe.manifest(), (function, pipe.manifest())
        assert 't = r0.GatherCmp(q0, float2(F(v1), F(v2)), F(v0));' in pipe.hlsl_ps(), function
    print('all eight comparison functions reach the call and their own smpc line')

    # Cache policy bits never change the compared value.
    plain = pipeline(shader()).hlsl_ps()
    for name, flags in (('glc', 1 << 13), ('slc', 1 << 25), ('dlc', 1 << 7)):
        assert pipeline(shader(flags=flags)).hlsl_ps() == plain, name
    print('glc, slc and dlc translate identically')

    # MAX_LOD is not carried into the static sampler; a fixed-mip gather never selects a level, so any
    # accepted value translates and stays in the manifest words.
    pipe = pipeline(shader(prefix=with_words(word1=0xFFF000)))
    assert 'smpc 0 0x1012 0xfff000 0x4500000 0x0' in pipe.manifest(), pipe.manifest()
    print('MAX_LOD bits are accepted and the guest words reach the manifest')

    # One ordinary and one comparison gather share the descriptor but bind different sampler types.
    moves = (smov(24, 0x12) + smov(25, 0) + smov(26, BILINEAR_POINT_MIP_W2) + smov(27, 0)
             + smov(28, 0x1012) + smov(29, 0) + smov(30, BILINEAR_POINT_MIP_W2) + smov(31, 0))
    head = PREFIX[:PREFIX.index(SAMPLER_MOVES[0])]
    tail = PREFIX[PREFIX.index(SAMPLER_MOVES[0]) + sum(len(m) for m in SAMPLER_MOVES):]
    mixed = (head + moves + tail + struct.pack('<II', *ORDINARY_WORD) + struct.pack('<II', *GATHER_WORD)
             + SUFFIX)
    pipe = pipeline(mixed)
    text, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'SamplerState q0 : register(s0);' in text and 'SamplerComparisonState q1 : register(s1);' in text, text
    assert 't = r0.GatherRed(q0, float2(F(v0), F(v1)));' in text, text
    assert 't = r0.GatherCmp(q1, float2(F(v1), F(v2)), F(v0));' in text, text
    assert 'smp 0 0x12 0x0 0x4500000 0x0' in manifest and 'smpc 1 0x1012 0x0 0x4500000 0x0' in manifest, manifest
    print('ordinary and comparison gathers bind q0 SamplerState and q1 SamplerComparisonState')

    # Destination boundary and a legal source/destination overlap.
    pipe = pipeline(shader(dst=252))
    text = pipe.hlsl_ps()
    assert 't = r0.GatherCmp(q0, float2(F(v1), F(v2)), F(v0));' in text, text
    assert 'v255 = U(t.w);' in text, text
    overlap = pipeline(shader(dst=0, addr=0)).hlsl_ps()
    call = overlap.index('t = r0.GatherCmp(q0, float2(F(v1), F(v2)), F(v0));')
    assert call < overlap.index('v0 = U(t.x);') < overlap.index('v3 = U(t.w);'), overlap
    print('destination v[252:255] and the v[0:3]/v[0:2] overlap keep the gather before the writes')

    # Encoded refusals: the malformed condition is carried by real MIMG words.
    for mask in (0, 3, 2, 4, 8):
        refused(shader(mask=mask), 'requires red dmask 0x1')
    for dim in (0, 2, 3, 5):
        refused(shader(dim=dim), 'requires a 2D texture')
    refused(shader(dst=0xFE), 'requires four destination registers')
    refused(shader(dst=0xFF), 'requires four destination registers')
    refused(shader(addr=0xFE), 'requires three contiguous address registers')
    refused(shader(srsrc=104), 'requires eight image descriptor registers')
    refused(shader(srsrc=108), 'requires eight image descriptor registers')
    refused(shader(ssamp=104), 'requires four sampler descriptor registers')
    refused(shader(nsa=True), 'requires three contiguous address registers')
    refused(shader(high=1 << 30), 'modifier a16')
    refused(shader(high=1 << 31), 'modifier d16')
    refused(shader(flags=1 << 16), 'modifier tfe')
    refused(shader(flags=1 << 17), 'modifier lwe')
    refused(shader(flags=1 << 15), 'modifier r128')
    refused(shader(flags=1 << 12), 'modifier unorm')
    for bit in (6, 14):
        refused(shader(flags=1 << bit), 'modifier reserved')
    for bit in (26, 27, 28, 29):
        refused(shader(high=1 << bit), 'modifier reserved')
    refused(shader(prefix=PREFIX.replace(SAMPLER_MOVES[0], b'')), 'requires a constant sampler')
    for opcode in (0x48, 0x4C, 0x58, 0x5F):
        refused(shader(opcode=opcode), 'image_gather4_c')
    refused(shader(opcode=0x50), 'image_gather4_o')
    try:
        G.Pipeline(shader(), VS_WORDS, root_blob(flags=0))
    except G.Unsupported as error:
        assert 'requires a pixel stage' in str(error), error
    else:
        raise AssertionError('comparison gather was accepted in a vertex stage')
    print('encoded refusals: dmask, dimension, register ranges, NSA, modifiers, dynamic sampler, variants')

    # The sampler profile is pixel_aot's comparison-gather profile, reused unchanged.
    bad = {
        'wrap x': with_words(word0=0x1010),
        'mirror y': with_words(word0=0x1012 | (1 << 3)),
        'unnormalized': with_words(word0=0x1012 | (1 << 15)),
        'clamp z': with_words(word0=0x1012 | (1 << 6)),
        'anisotropy': with_words(word0=0x1012 | (1 << 9)),
        'trunc coord': with_words(word0=0x1012 | (1 << 27)),
        'point xy': with_words(word2=1 << 26),
        'linear mip': with_words(word2=BILINEAR_POINT_MIP_W2 | (2 << 26)),
        'min lod': with_words(word1=1),
        'lod bias': with_words(word2=BILINEAR_POINT_MIP_W2 | 1),
        'word3': with_words(word3=1),
        'perf mip': with_words(word1=0x1000000),
        'word2 bit 28': with_words(word2=BILINEAR_POINT_MIP_W2 | (1 << 28)),
        'word2 bit 29': with_words(word2=BILINEAR_POINT_MIP_W2 | (1 << 29)),
        'word2 bit 30': with_words(word2=BILINEAR_POINT_MIP_W2 | (1 << 30)),
    }
    for prefix in bad.values():
        refused(shader(prefix=prefix), PROFILE)
    print('sampler profile: %d out-of-profile word sets refused with the shared gather diagnostic' % len(bad))

    # Operand widths that llvm-mc refuses to assemble are checked against the handler directly.
    for operands, reason in (
            ('v[4:6], v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
             'requires four destination registers'),
            ('v[4:8], v[0:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
             'requires four destination registers'),
            ('v[4:7], v[0:1], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
             'requires three contiguous address registers'),
            ('v[4:7], v[0:3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
             'requires three contiguous address registers'),
            ('v[4:7], v[0:2], s[16:22], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
             'requires eight image descriptor registers'),
            ('v[4:7], v[0:2], s[16:24], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
             'requires eight image descriptor registers'),
            ('v[4:7], v[0:2], s[16:23], s[24:26] dmask:0x1 dim:SQ_RSRC_IMG_2D',
             'requires four sampler descriptor registers'),
            ('v[4:7], v[0:2], s[16:23], s[24:28] dmask:0x1 dim:SQ_RSRC_IMG_2D',
             'requires four sampler descriptor registers'),
    ):
        handler_refused(operands, reason)
    print('direct handler validation: three/five destination, two/four address, seven/nine image, '
          'three/five sampler registers refused')

    # Negative controls that prove the guards are load-bearing: remove each one and require the
    # malformed form to pass, then restore the module.
    dst_guard = """        dst, addr, image, sampler = (reg_range(ops[i]) for i in (0, 1, 2, 3))
        if not dst or dst[0] != 'v' or dst[2] - dst[1] != 3 or dst[2] > 255:
            raise Unsupported('%s requires four destination registers' % name)
"""
    dmask_guard = """        if masks != ['dmask:0x1']:
            raise Unsupported('%s requires red dmask 0x1' % name)
"""
    sampler_guard = "            check_gather_sampler(words, 1, comparison=True)\n"
    for label, guard, replacement, code, reason in (
            ('destination width', dst_guard,
             "        dst, addr, image, sampler = (reg_range(ops[i]) for i in (0, 1, 2, 3))\n"
             "        pass\n", shader(dst=0xFE), 'four destination registers'),
            ('red dmask', dmask_guard, '        pass\n', shader(mask=2), 'red dmask 0x1'),
            ('sampler profile', sampler_guard, '            pass\n', shader(prefix=with_words(word0=0x1010)),
             'Unsupported gather sampler')):
        mutant = load_mutant([(guard, replacement)])
        mutated_pipeline(mutant, code)  # must not raise
        refused(code, reason)
        print('control: removing the %s guard makes the malformed form translate' % label)

    # Wrong lowerings made by editing the correct text must fail the exact-text checks above.
    good = pipeline(shader()).hlsl_ps()
    exact = 't = r0.GatherCmp(q0, float2(F(v1), F(v2)), F(v0));'
    mutants = {
        'reference read from u': good.replace(exact, 't = r0.GatherCmp(q0, float2(F(v0), F(v2)), F(v1));'),
        'u and v exchanged': good.replace(exact, 't = r0.GatherCmp(q0, float2(F(v2), F(v1)), F(v0));'),
        'ordinary gather substituted': good.replace(exact, 't = r0.GatherRed(q0, float2(F(v1), F(v2)));'),
        'first lane from t.w': good.replace('v4 = U(t.x);', 'v4 = U(t.w);'),
        'sampler declared ordinary': good.replace('SamplerComparisonState', 'SamplerState'),
    }
    for label, mutant in mutants.items():
        assert mutant != good, label
    assert exact not in mutants['reference read from u']
    assert exact not in mutants['ordinary gather substituted']
    assert 'v4 = U(t.x);' not in mutants['first lane from t.w']
    assert 'SamplerComparisonState' not in mutants['sampler declared ordinary']
    print('text controls: wrong reference, swapped coordinates, ordinary gather, lane and sampler edits differ')


def depth(resource_mip, x, y):
    """A distinct dyadic value in (0, 1] for every texel; exact in binary32."""
    return Fraction(((11 * x + 7 * y + 13 * resource_mip) % 32) + 1, 64)


def footprint(u, v, extent):
    x = math.floor(Fraction(u) * extent - Fraction(1, 2))
    y = math.floor(Fraction(v) * extent - Fraction(1, 2))
    return tuple((min(max(px, 0), extent - 1), min(max(py, 0), extent - 1))
                 for px, py in ((x, y + 1), (x + 1, y + 1), (x + 1, y), (x, y)))


def expected_case(function, reference, u, v, count, first):
    taps = footprint(u, v, SIDE)
    return tuple(float(passes(function, float(reference), float(depth(first, x, y)))) for x, y in taps)


def case_bytes(count, first, expected):
    data = bytearray(struct.pack('<II', first, count))
    for mip in range(count):
        extent = SIDE >> mip
        for y in range(extent):
            for x in range(extent):
                data += struct.pack('<f', float(depth(first + mip, x, y)))
    for _ in range(SIDE * SIDE):
        data += struct.pack('<4f', *expected)
    return bytes(data)


def cases():
    """(label, function, reference, u, v, count, first)."""
    base = (Fraction(1, 2), Fraction(1, 4))
    equality = (Fraction(1, 2), Fraction(1, 4))
    asymmetric = (Fraction(13, 16), Fraction(11, 16))
    edge = (Fraction(-1, 2), Fraction(5, 4))
    out = [(function, function, Fraction(1, 2), *base, 1, 0) for function in COMPARE]
    # Exact-equality boundary: the origin tap is exactly the reference at this footprint.
    eq_x, eq_y = footprint(*equality, SIDE)[3]
    boundary = depth(0, eq_x, eq_y)
    for function in ('EQUAL', 'LESS_EQUAL', 'GREATER_EQUAL', 'NOT_EQUAL', 'LESS', 'GREATER'):
        out.append(('equality ' + function, function, boundary, *equality, 1, 0))
    out.append(('asymmetric LESS', 'LESS', Fraction(1, 2), *asymmetric, 3, 0))
    out.append(('asymmetric GREATER_EQUAL', 'GREATER_EQUAL', Fraction(1, 2), *asymmetric, 3, 0))
    # These split references separate the low taps from the high taps, but they still give the
    # lower-right (tap 1) and upper-right (tap 2) taps the same result: at this footprint those texels
    # are 24/64 and 17/64, both above the 1/4 reference and both below the 1/2 reference.
    out.append(('split LESS', 'LESS', Fraction(1, 4), *asymmetric, 3, 0))
    out.append(('split GREATER', 'GREATER', Fraction(1, 4), *asymmetric, 3, 0))
    # A reference between those two values (5/16 = 20/64) separates tap 1 from tap 2 as well, so the
    # whole case matrix distinguishes all four destination positions. The coordinates differ from the
    # split cases but stay inside the same texel footprint, away from an integer footprint boundary.
    out.append(('split middle LESS', 'LESS', Fraction(5, 16),
                Fraction(53, 64), Fraction(45, 64), 3, 0))
    out.append(('second pattern NEQUAL', 'NOT_EQUAL', Fraction(1, 4), Fraction(7, 16), Fraction(13, 16), 3, 0))
    out.append(('view mip 1 LESS', 'LESS', Fraction(1, 2), *base, 3, 1))
    vx, vy = footprint(*base, SIDE)[0]
    out.append(('view mip 1 equality', 'EQUAL', depth(1, vx, vy), *base, 3, 1))
    out.append(('clamped uv', 'LESS', Fraction(1, 2), *edge, 1, 0))
    return out


def lane_signatures(matrix):
    """Each destination position's result sequence across a case matrix.

    A case matrix that leaves two positions with equal sequences cannot detect a swap between them,
    whatever the individual cases compare.
    """
    vectors = [expected_case(function, reference, u, v, count, first)
               for _, function, reference, u, v, count, first in matrix]
    return [tuple(vector[position] for vector in vectors) for position in range(4)]


def draw(directory, dxc, renderer=None, adapter=None, fxc=None, evidence=None):
    with CompileQueue() as queue:
        draw_cases(queue, directory, dxc, renderer, adapter, fxc, evidence)


def draw_cases(queue, directory, dxc, renderer, adapter, fxc, evidence):
    directory.mkdir(parents=True, exist_ok=True)
    if evidence:
        evidence.mkdir(parents=True, exist_ok=True)
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    summary = []

    identifiers = {}
    # The vertex stage is the same for every case: compile it once per compiler, all compilers together.
    for kind, compiler, vs_profile, _ps in compilers:
        queue.submit(compile_hlsl, compiler, kind, FULLSCREEN_VS, 'fullscreen_%s' % kind, vs_profile, directory)
    vs_paths = {kind: queue.call(compile_hlsl, compiler, kind, FULLSCREEN_VS, 'fullscreen_%s' % kind, vs_profile,
                                 directory) for kind, compiler, vs_profile, _ps in compilers}
    # Each case submits its pixel-stage compiles when it is declared. flush() then draws the declared cases in
    # order, so a failed compile or draw is still the first failure in declaration order.
    pending = []

    def flush():
        while pending:
            pending.pop(0)()

    def run(case_index, group, label, code, function, reference, u, v, count, first, expected, *,
            bad=False, hlsl_edits=()):
        pipe = pipeline(code)
        text = pipe.hlsl_ps()
        # A sensitivity control may mutate the generated pixel stage itself. The edited text is what is
        # compiled and drawn, and the artifact keeps the mutated source with the edit recorded.
        for old, new in hlsl_edits:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        smpc = [line for line in pipe.manifest().splitlines() if line.startswith('smpc ')]
        assert len(smpc) == 1, pipe.manifest()
        plan = []
        for kind, compiler, vs_profile, ps_profile in compilers:
            # Lowercase before sanitizing so labels that differ only in case cannot collide, and keep the
            # case index and compiler in the name so no case overwrites another's artifacts.
            slug = re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
            if group == 'mutant' and slug.startswith('mutant_'):
                slug = slug[len('mutant_'):]
            name = '%02d_%s_%s_%s' % (case_index, group, slug, kind)
            assert name not in identifiers, 'duplicate evidence identifier %r (%r and %r)' % (
                name, identifiers.get(name), (group, label))
            identifiers[name] = (group, label)
            queue.submit(compile_hlsl, compiler, kind, text, name, ps_profile, directory)
            plan.append((kind, compiler, ps_profile, name))

        def finish():
            for kind, compiler, ps_profile, name in plan:
                vs_path = vs_paths[kind]
                shader_path = queue.call(compile_hlsl, compiler, kind, text, name, ps_profile, directory)
                case = directory / (name + '.case')
                case.write_bytes(case_bytes(count, first, expected))
                if renderer is None:
                    continue
                command = [renderer, adapter, '--check-pixel-compare-gather', str(shader_path), str(case),
                           smpc[0], str(vs_path)]
                result = subprocess.run(command, capture_output=True, text=True)
                output = result.stdout + result.stderr
                print('%s %s\n%s' % (label, kind, output), end='', flush=True)
                if result.returncode == 77:
                    raise SystemExit(77)
                if bad:
                    assert result.returncode == 1 and 'pixel_mismatch' in output, (label, kind, output)
                else:
                    assert result.returncode == 0, (label, kind, output)
                    assert 'pixels=64 channels=256 max_absolute_error=0' in output, (label, kind, output)
                artifacts = [name + '.hlsl', name + '.manifest', name + '.case', shader_path.name, name + '.log']
                summary.append({'case_index': case_index, 'group': group, 'label': label, 'compiler': kind,
                                'profile': ps_profile, 'function': function, 'reference': str(reference),
                                'u': str(u), 'v': str(v), 'count': count, 'first': first,
                                'expected': list(expected), 'smpc': smpc[0],
                                'shader_edits': [list(edit) for edit in hlsl_edits],
                                'result': 'expected-mismatch' if bad else 'pass',
                                'exit': result.returncode, 'artifacts': artifacts, 'output': output})
                if evidence:
                    (evidence / (name + '.hlsl')).write_text(text, encoding='ascii')
                    (evidence / (name + '.manifest')).write_text(pipe.manifest(), encoding='ascii')
                    (evidence / (name + '.case')).write_bytes(case.read_bytes())
                    (evidence / shader_path.name).write_bytes(shader_path.read_bytes())
                    (evidence / (name + '.log')).write_text(output, encoding='ascii')

        pending.append(finish)

    for case_index, (label, function, reference, u, v, count, first) in enumerate(cases()):
        expected = expected_case(function, reference, u, v, count, first)
        word0 = CLAMP_LAST_TEXEL | (CLAMP_LAST_TEXEL << 3) | (COMPARE[function] << 12)
        run(case_index, 'positive', label, fixture(reference, u, v, word0), function, reference, u, v,
            count, first, expected)
    flush()
    if renderer is None:
        print('compiled %d comparison-gather cases; no renderer was given' % len(cases()))
        if evidence:
            (evidence / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='ascii')
        return

    # Output-sensitivity controls: each edited expectation or translation must complete a draw and
    # fail the pixel comparison at (0,0), not merely fail to compile or create a pipeline state.
    by_label = {case[0]: case for case in cases()}
    _, function, reference, u, v, count, first = by_label['split LESS']
    good = expected_case(function, reference, u, v, count, first)
    assert len(set(good)) == 2 and good[0] != good[1], 'the split case must distinguish a permutation: %r' % (good,)
    less_word0 = CLAMP_LAST_TEXEL | (CLAMP_LAST_TEXEL << 3) | (LESS << 12)
    less_code = fixture(reference, u, v, less_word0)
    mutant_base = len(cases())
    run(mutant_base, 'mutant', 'mutant swapped taps', less_code, function, reference, u, v, count, first,
        (good[1], good[0], good[2], good[3]), bad=True)
    run(mutant_base + 1, 'mutant', 'mutant changed last channel', less_code, function, reference, u, v,
        count, first, good[:3] + (1.0 - good[3],), bad=True)
    greater_code = fixture(reference, u, v,
                           CLAMP_LAST_TEXEL | (CLAMP_LAST_TEXEL << 3) | (COMPARE['GREATER'] << 12))
    run(mutant_base + 2, 'mutant', 'mutant reversed comparison', greater_code, function, reference, u, v,
        count, first, good, bad=True)
    _, _, _, eq_u, eq_v, eq_count, eq_first = by_label['equality EQUAL']
    eq_reference = by_label['equality EQUAL'][2]
    equal_word0 = CLAMP_LAST_TEXEL | (CLAMP_LAST_TEXEL << 3) | (COMPARE['EQUAL'] << 12)
    eq_expected = expected_case('EQUAL', eq_reference, eq_u, eq_v, eq_count, eq_first)
    strict = tuple(0.0 if depth(eq_first, x, y) == eq_reference else value
                   for value, (x, y) in zip(eq_expected, footprint(eq_u, eq_v, SIDE)))
    assert strict != eq_expected, 'the equality case must have an exactly equal tap'
    run(mutant_base + 3, 'mutant', 'mutant equality as strict less',
        fixture(eq_reference, eq_u, eq_v, equal_word0), 'EQUAL',
        eq_reference, eq_u, eq_v, eq_count, eq_first, strict, bad=True)
    _, _, mip_reference, mip_u, mip_v, mip_count, mip_first = by_label['view mip 1 equality']
    wrong_mip = expected_case('EQUAL', mip_reference, mip_u, mip_v, mip_count, 0)
    assert wrong_mip != expected_case('EQUAL', mip_reference, mip_u, mip_v, mip_count, mip_first)
    run(mutant_base + 4, 'mutant', 'mutant wrong visible mip', fixture(mip_reference, mip_u, mip_v,
        CLAMP_LAST_TEXEL | (CLAMP_LAST_TEXEL << 3) | (COMPARE['EQUAL'] << 12)),
        'EQUAL', mip_reference, mip_u, mip_v, mip_count, mip_first, wrong_mip, bad=True)
    run(mutant_base + 5, 'mutant', 'mutant wrong reference', fixture(Fraction(1, 2), u, v, less_word0),
        'LESS', reference, u, v, count, first, good, bad=True)
    run(mutant_base + 6, 'mutant', 'mutant raw gather values', less_code, function, reference, u, v, count,
        first, tuple(float(depth(first, x, y)) for x, y in footprint(u, v, SIDE)), bad=True)
    # A shader-output mutation, not an expectation edit: swap the lower-right and upper-right lanes in a
    # temporary copy of the translated stage. The correct expectation must fail after a real draw.
    _, mid_function, mid_reference, mid_u, mid_v, mid_count, mid_first = by_label['split middle LESS']
    mid_expected = expected_case(mid_function, mid_reference, mid_u, mid_v, mid_count, mid_first)
    assert mid_expected == (0.0, 1.0, 0.0, 0.0), mid_expected
    mid_code = fixture(mid_reference, mid_u, mid_v, less_word0)
    run(mutant_base + 7, 'mutant', 'mutant swapped middle lanes', mid_code, mid_function, mid_reference,
        mid_u, mid_v, mid_count, mid_first, mid_expected, bad=True,
        hlsl_edits=(('v5 = U(t.y);', 'v5 = U(t.z);'), ('v6 = U(t.z);', 'v6 = U(t.y);')))
    flush()
    if evidence:
        (evidence / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--llvm-mc')
    mode.add_argument('--draw', nargs='+', metavar=('DXC', 'RENDERER', 'ADAPTER'))
    parser.add_argument('--evidence')
    options = parser.parse_args()
    if options.draw:
        if len(options.draw) not in (3, 4):
            parser.error('--draw takes <dxc> <renderer> <adapter> [fxc]')
        directory = Path(tempfile.mkdtemp(prefix='comparison_gather_', dir=str(ROOT / 'out')))
        draw(directory, *options.draw[:3], fxc=options.draw[3] if len(options.draw) == 4 else None,
             evidence=Path(options.evidence) if options.evidence else None)
    else:
        controls(options.llvm_mc)


if __name__ == '__main__':
    main()
