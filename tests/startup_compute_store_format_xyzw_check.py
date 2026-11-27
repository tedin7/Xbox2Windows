#!/usr/bin/env python3
"""Verify compute buffer_store_format_xyzw lowering in tools/startup/gfx2hlsl.py (Refs #2274).

  (no arguments)                                     pure-Python unit, format, bounds and refusal checks
  binary <llvm-mc>                                   reassemble fixture sources and verify machine words
  dispatch <dxc> <offscreen_check> <warp|hardware>   translate, compile DXC cs_6_0, dispatch and verify readback
  control <name> <dxc> <offscreen_check> <adapter>   dispatch a mutated translation that must mismatch

The fixtures are source-built gfx1013 compute programs. Each lane reads four words from t0 and stores them with
`buffer_store_format_xyzw ... idxen`, the only address form the lowering models:
- Program X: `v[4:7], v0, s[16:19], 0 idxen` (16 lanes). Dispatched for the five 4-channel formats (32_32_32_32 FLOAT
  and UINT, 8_8_8_8_UNORM, 2_10_10_10_UNORM, 16_16_16_16_FLOAT; unified and legacy encodings) with structured
  OOB_SELECT 0 bounds, plus descriptors that must write nothing: 1- and 2-channel formats (channel count mismatch),
  OOB_SELECT 1/2, non-identity DST_SEL, stride 0 or nonzero against the OOB mode, DW1 swizzle bits and a stride
  smaller than the element.
- Program R: inline soffset 4 and `offset:12`, under raw OOB_SELECT 3 (stride 0; every lane writes the same element
  with the same data) and structured OOB_SELECT 0 (soffset added to the base, not checked against the stride).
- Program E: `s_and_saveexec_b32` narrows EXEC to lanes 0..5 before the store, so the store is lane guarded.
"""

import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))

import gfx2hlsl as G  # noqa: E402
import compute_store_aot as A  # noqa: E402
from startup_gfx2hlsl_compute_check import SENTINEL, USER7, WORDS, patch, refused, root_blob  # noqa: E402
from startup_compute_store_d16_x_check import PREAMBLE_X8, encode_mubuf  # noqa: E402

MUBUF_STORE_FORMAT_XYZW = 0x07
STORE_X = '00201ce000040480'  # buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 idxen

SOURCE_X = PREAMBLE_X8 + """\
v_lshlrev_b32 v4, 4, v0
buffer_load_dwordx4 v[4:7], v4, s[12:15], 0 offen
s_waitcnt vmcnt(0)
buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 idxen
s_endpgm
"""
LOAD_X = '06038abe' '80038bbe' '05030cf4000000fa' '7fc08cbf' '84000834' '001038e004040380' '703f8cbf'
WORDS_X = bytes.fromhex(LOAD_X + STORE_X + '000081bf')

SOURCE_R = SOURCE_X.replace('0 idxen\n', '4 idxen offset:12\n')
WORDS_R = bytes.fromhex(LOAD_X + '0c201ce000040484' '000081bf')

SOURCE_E = SOURCE_X.replace('buffer_store', 'v_cmp_gt_u32 vcc_lo, 6, v0\ns_and_saveexec_b32 s0, vcc_lo\nbuffer_store')
WORDS_E = bytes.fromhex(LOAD_X + '8600887d' '6a3c80be' + STORE_X + '000081bf')

# A second UAV through s[20:23] (table offset 0x20): its d16_x store and the xyzw store get separate entry constants.
SOURCE_D = SOURCE_X.replace('s_waitcnt lgkmcnt', 's_load_dwordx4 s[20:23], s[10:11], 0x20\ns_waitcnt lgkmcnt', 1).replace(
    'buffer_store', 'buffer_store_format_d16_x v4, v0, s[20:23], 0 idxen\nbuffer_store')
WORDS_D = bytes.fromhex('06038abe' '80038bbe' '05030cf4000000fa' '050508f4200000fa' '7fc08cbf' '84000834'
                        '001038e004040380' '703f8cbf' '002010e200040580' + STORE_X + '000081bf')

VERIFIED_SNIPPETS = {
    'buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 idxen': STORE_X,
    'buffer_store_format_xyzw v[4:7], v0, s[16:19], 4 idxen offset:12': '0c201ce000040484',
    'buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 offen': '00101ce000040480',
    'buffer_store_format_xyzw v[4:7], v[0:1], s[16:19], 0 idxen offen': '00301ce000040480',
    'buffer_store_format_xyzw v[4:7], off, s[16:19], 0': '00001ce000040480',
    'buffer_store_format_xyzw v[4:7], v0, s[16:19], s4 idxen': '00201ce000040404',
    'buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 idxen glc': '00601ce000040480',
    'buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 idxen slc': '00201ce000044480',
    'buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 idxen dlc': '00a01ce000040480',
    'buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 idxen tfe': '00201ce000048480',
}

PROGRAMS = {
    'X': (SOURCE_X, WORDS_X, USER7, (16, 1, 1), (1, 1, 1)),
    'R': (SOURCE_R, WORDS_R, USER7, (16, 1, 1), (1, 1, 1)),
    'E': (SOURCE_E, WORDS_E, USER7, (16, 1, 1), (1, 1, 1)),
    'D': (SOURCE_D, WORDS_D, USER7, (16, 1, 1), (1, 1, 1)),
}

EXPECTED_HELPER = """\
uint store_unorm(uint bits, float scale) {
    float x = asfloat(bits);
    if (isnan(x)) return 0u;
    precise float saturated = saturate(x);
    precise float scaled = saturated * scale;
    return (uint)floor(scaled + 0.5f);
}
void store_format_xyzw(RWByteAddressBuffer buf, uint word3, uint stride, uint num_records, uint idx, uint voff, \
uint soff, uint imm_off, uint4 v) {
    if ((word3 & 0xcef80000u) != 0u || (word3 & 0xfffu) != 0xfacu) return;
    uint oob_sel = (word3 >> 28u) & 3u;
    if ((oob_sel != 0u && oob_sel != 3u) || stride > 0x3fffu) return;
    uint field = (word3 >> 12u) & 127u;
    uint fmt = 0u;
    if (((word3 >> 24u) & 1u) != 0u) {
        fmt = field;
    } else {
        if (field == 72u) fmt = 50u;
        else if (field == 80u) fmt = 56u;
        else if (field == 103u) fmt = 71u;
        else if (field == 116u) fmt = 75u;
        else if (field == 119u) fmt = 77u;
    }
    uint elem_bytes = (fmt == 75u || fmt == 77u) ? 16u : fmt == 71u ? 8u : (fmt == 50u || fmt == 56u) ? 4u : 0u;
    if (elem_bytes == 0u) return;
BOUNDS    if ((addr & 3u) != 0u) return;
    uint uav_bytes = 0u;
    buf.GetDimensions(uav_bytes);
    if (uav_bytes < elem_bytes || addr > uav_bytes - elem_bytes) return;
    if (elem_bytes == 16u) {
        buf.Store4(addr, v);
    } else if (fmt == 71u) {
        buf.Store2(addr, uint2(pack16_rne(asfloat(v.x)) | (pack16_rne(asfloat(v.y)) << 16u), \
pack16_rne(asfloat(v.z)) | (pack16_rne(asfloat(v.w)) << 16u)));
    } else if (fmt == 56u) {
        buf.Store(addr, store_unorm(v.x, 255.0f) | (store_unorm(v.y, 255.0f) << 8u) \
| (store_unorm(v.z, 255.0f) << 16u) | (store_unorm(v.w, 255.0f) << 24u));
    } else {
        buf.Store(addr, store_unorm(v.x, 1023.0f) | (store_unorm(v.y, 1023.0f) << 10u) \
| (store_unorm(v.z, 1023.0f) << 20u) | (store_unorm(v.w, 3.0f) << 30u));
    }
}
"""
# Bounds and OOB_SELECT lines shared with buffer_store_format_d16_x: taken from its helper so the two cannot drift.
BOUNDS_FIRST, BOUNDS_LAST = '    if (voff > 0xffffffffu - imm_off) return;\n', '        addr = elem_addr + soff;\n    }\n'


def desc_word3(code, *, legacy=False, oob_select=0, dst_sel=0xfac):
    if legacy:
        field, level = next(k for k, v in G.STORE_FORMAT_XYZW_LEGACY_FORMATS.items() if v == code), 0
    else:
        field, level = code, 1
    return ((oob_select & 3) << 28) | (level << 24) | ((field & 0x7f) << 12) | (dst_sel & 0xfff)


# --- CPU reference of the format conversions ---------------------------------------------------------------------

def f32(bits):
    return struct.unpack('<f', struct.pack('<I', bits & 0xffffffff))[0]


def bits32(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def round_f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def unorm(bits, scale):
    """saturate, multiply and add 0.5 in binary32, then floor (Direct3D half away from zero); NaN gives 0."""
    x = f32(bits)
    if math.isnan(x):
        return 0
    return int(math.floor(round_f32(round_f32(min(1.0, max(0.0, x)) * scale) + 0.5)))


def f16_rne(bits):
    """binary32 to binary16, round to nearest even (CPython's binary16 pack), NaN to a quiet NaN keeping the sign."""
    sign = (bits >> 16) & 0x8000
    x = f32(bits)
    if math.isnan(x):
        return sign | 0x7e00
    try:
        return struct.unpack('<H', struct.pack('<e', x))[0]
    except OverflowError:
        return sign | 0x7c00


def pack_element(code, words):
    """Bytes the store writes for one element of a 4-channel format code."""
    if code in (75, 77):
        return struct.pack('<4I', *words)
    if code == 71:
        return struct.pack('<4H', *(f16_rne(w) for w in words))
    if code == 56:
        return struct.pack('<I', sum(unorm(w, 255.0) << (8 * i) for i, w in enumerate(words)))
    if code == 50:
        return struct.pack('<I', unorm(words[0], 1023.0) | (unorm(words[1], 1023.0) << 10)
                           | (unorm(words[2], 1023.0) << 20) | (unorm(words[3], 3.0) << 30))
    raise KeyError(code)


def check_reference():
    # Every binary16 value survives binary16 -> binary32 -> f16_rne (NaN becomes the quiet NaN of its sign).
    for h in range(0x10000):
        x = struct.unpack('<e', struct.pack('<H', h))[0]
        want = (h & 0x8000) | 0x7e00 if math.isnan(x) else h
        assert f16_rne(bits32(x)) == want, hex(h)
    # Every UNORM code n reads back from n / (2^k - 1) in binary32.
    for scale in (3.0, 255.0, 1023.0):
        for n in range(int(scale) + 1):
            assert unorm(bits32(n / scale), scale) == n, (scale, n)
    for bits, scale, want in ((0x3f000000, 255.0, 128), (0x3f000000, 1023.0, 512), (0x3f000000, 3.0, 2),
                              (0x7fc00000, 255.0, 0), (0xff800000, 255.0, 0), (0x7f800000, 255.0, 255),
                              (0xbf800000, 1023.0, 0), (0x00000001, 255.0, 0)):
        assert unorm(bits, scale) == want, (hex(bits), scale)
    for bits, want in ((0x3f801000, 0x3c00), (0x3f803000, 0x3c02), (0x3f801001, 0x3c01), (0x477fefff, 0x7bff),
                       (0x477ff000, 0x7c00), (0xc77ff000, 0xfc00), (0x33000000, 0x0000), (0x33000001, 0x0001),
                       (0x33c00000, 0x0002), (0x7f800001, 0x7e00), (0xffc00000, 0xfe00)):
        assert f16_rne(bits) == want, hex(bits)


# --- dispatch cases ----------------------------------------------------------------------------------------------

UNORM_EDGES = (0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 0x40000000, 0x7fc00000, 0xffc00000,
               0x7f800000, 0xff800000, 0x3b808081, 0x3e800000, 0x3f400000, 0x3eff7cee, 0x00000001, 0x3f7f7f7f)
F16_EDGES = (0x3f800000, 0x3f801000, 0x3f803000, 0x3f801001, 0x477fe000, 0x477fefff, 0x477ff000, 0xc77ff000,
             0x7fc00001, 0xffc00000, 0x7f800001, 0x7f800000, 0x33800000, 0x33000000, 0x33000001, 0x33c00000)
RAW_WORDS = tuple((0x9e3779b9 * (i + 1)) & 0xffffffff for i in range(48)) + (
    0x7fc00001, 0xff800001, 0x80000000, 0x00000001, 0xffffffff, 0x00000000, 0x7f7fffff, 0x0000ffff,
    0x80000001, 0x7f800001, 0xfffffffe, 0x12345678, 0x3f800000, 0xbf800000, 0x00800000, 0x807fffff)


def lane_words(kind):
    if kind == 'raw':
        flat = RAW_WORDS
    else:
        edges = UNORM_EDGES if kind == 'unorm' else F16_EDGES
        flat = tuple(edges[(lane * 5 + c * 3) % 16] if lane < 8 else edges[(lane * 4 + c) % 16]
                     for lane in range(16) for c in range(4))
    return [flat[4 * lane:4 * lane + 4] for lane in range(16)]


# name: (word3, uav_stride, num_records, kind, writes)
X_VARIANTS = {
    'X-f32': (desc_word3(77), 16, 12, 'raw', True),
    'X-u32-legacy': (desc_word3(75, legacy=True), 20, 10, 'raw', True),
    'X-unorm8': (desc_word3(56), 4, 16, 'unorm', True),
    'X-unorm10-legacy': (desc_word3(50, legacy=True), 8, 14, 'unorm', True),
    'X-f16': (desc_word3(71), 8, 16, 'f16', True),
    'X-f16-legacy': (desc_word3(71, legacy=True), 12, 15, 'f16', True),
    # Descriptors that must write nothing (all 320 sentinel words survive):
    'X-1ch': ((1 << 24) | (22 << 12) | 0xfac, 4, 16, 'raw', False),          # BUF_FMT_32_FLOAT: channel mismatch
    'X-2ch': ((1 << 24) | (64 << 12) | 0xfac, 8, 16, 'raw', False),          # BUF_FMT_32_32_FLOAT: channel mismatch
    'X-oob1': (desc_word3(77, oob_select=1), 16, 12, 'raw', False),
    'X-oob2': (desc_word3(77, oob_select=2), 16, 12, 'raw', False),
    'X-dst-sel': (desc_word3(77, dst_sel=0xfa5), 16, 12, 'raw', False),      # X and Y selects swapped
    'X-oob0-stride0': (desc_word3(77), 0, 256, 'raw', False),
    'X-oob3-stride16': (desc_word3(77, oob_select=3), 16, 256, 'raw', False),
    'X-swizzle-en': (desc_word3(77), 16 | 0x8000, 12, 'raw', False),
    'X-stride8': (desc_word3(77), 8, 12, 'raw', False),                      # element larger than the stride
}
DISPATCH_CASES = (*X_VARIANTS, 'R-raw', 'R-raw-oob', 'R-struct', 'E')


def case_inputs(name):
    """Return (program key, t0 bytes, 16 entry constants, expected 320 words)."""
    out = bytearray(struct.pack('<%dI' % WORDS, *([SENTINEL] * WORDS)))
    if name in X_VARIANTS:
        word3, stride, num_records, kind, writes = X_VARIANTS[name]
        lanes = lane_words(kind)
        if writes:
            code = G.validate_store_format_xyzw_word3(word3)[0]
            for lane in range(min(16, num_records)):
                element = pack_element(code, lanes[lane])
                out[lane * stride:lane * stride + len(element)] = element
        prog = 'X'
    elif name.startswith('R-'):
        lanes = lane_words('raw')
        if name == 'R-struct':
            word3, stride, num_records = desc_word3(77), 32, 8
            for lane in range(num_records):  # base = idx * stride + offset:12 + soffset 4
                out[lane * 32 + 16:lane * 32 + 32] = pack_element(77, lanes[lane])
        else:  # raw: stride 0, every lane writes bytes 16..31 with the same data
            lanes = [lanes[3]] * 16
            word3, stride, num_records = desc_word3(77, oob_select=3), 0, 32 if name == 'R-raw' else 31
            if name == 'R-raw':
                out[16:32] = pack_element(77, lanes[0])
        prog = 'R'
    elif name == 'E':
        lanes = lane_words('raw')
        word3, stride, num_records = desc_word3(75), 16, 16
        for lane in range(6):
            out[lane * 16:lane * 16 + 16] = pack_element(75, lanes[lane])
        prog = 'E'
    else:
        raise KeyError(name)
    source = [0] * WORDS
    for lane, words in enumerate(lanes):
        source[4 * lane:4 * lane + 4] = words
    constants = [0, 0, word3, stride, num_records] + [0] * 11
    return prog, struct.pack('<%dI' % WORDS, *source), constants, list(struct.unpack('<%dI' % WORDS, out))


def translate(prog_key, code=None):
    _, words, rsrc2, group, _ = PROGRAMS[prog_key]
    return G.ComputePipeline(words if code is None else code, root_blob(), rsrc2, group)


def emitted_text_problems():
    """Exact emitted text of the lowering; returns the mismatches (empty when the translator is as reviewed)."""
    problems = []
    pipe = translate('X')
    hlsl = pipe.hlsl_cs()
    d16 = translate('D').hlsl_cs()
    bounds = d16[d16.index(BOUNDS_FIRST):d16.index(BOUNDS_LAST, d16.index(BOUNDS_FIRST)) + len(BOUNDS_LAST)]
    helper = EXPECTED_HELPER.replace('BOUNDS', bounds)
    if helper not in hlsl:
        problems.append('helper text')
    if 'uint pack16_rne(float value) {' not in hlsl or hlsl.index('uint pack16_rne(') > hlsl.index('void store_format_xyzw('):
        problems.append('pack16_rne not defined before the helper')
    expected_manifest = ('x2w-manifest 2\nstage cs\ngroup 16 1 1\nud 5\n'
                         'udsrc 0 rootconst 0 0\nudsrc 1 rootconst 0 1\n'
                         'udsrc 2 uav_descword3 0\nudsrc 3 uav_stride 0\nudsrc 4 uav_numrecords 0\n'
                         'srv 0 buf table 1 0\nuav 0 table 1 16\n')
    if pipe.manifest() != expected_manifest:
        problems.append('manifest X')
    for prog, line in (('X', '\n    store_format_xyzw(u0, ud2, ud3, ud4, v0, 0u, 0u, 0u, uint4(v4, v5, v6, v7));\n'),
                       ('R', '\n    store_format_xyzw(u0, ud2, ud3, ud4, v0, 0u, 4u, 12u, uint4(v4, v5, v6, v7));\n'),
                       ('E', '\n    if (exec & 1) store_format_xyzw(u0, ud2, ud3, ud4, v0, 0u, 0u, 0u, '
                             'uint4(v4, v5, v6, v7));\n')):
        if line not in translate(prog).hlsl_cs():
            problems.append('call %s' % prog)
    two = translate('D')
    if ('udsrc 2 uav_descword3 0\nudsrc 3 uav_stride 0\nudsrc 4 uav_numrecords 0\n'
            'udsrc 5 uav_descword3 1\nudsrc 6 uav_stride 1\nudsrc 7 uav_numrecords 1\n') not in two.manifest() or \
            '\n    store_format_xyzw(u1, ud5, ud6, ud7, v0, 0u, 0u, 0u, uint4(v4, v5, v6, v7));\n' not in d16:
        problems.append('second UAV entry constants')
    return problems


def check_controls():
    check_reference()

    # 1. Decoder round trip of every recorded encoding.
    for line, hex_words in VERIFIED_SNIPPETS.items():
        (_, size, op, asm), = G.decode(bytes.fromhex(hex_words))
        assert (size, op, asm) == (8, 'buffer_store_format_xyzw', line), (line, size, op, asm)
    for line, fields in ((STORE_X, dict(idxen=True)), ('0c201ce000040484', dict(idxen=True, offset=12))):
        soffset = 0x84 if fields.get('offset') else 0x80
        assert encode_mubuf(MUBUF_STORE_FORMAT_XYZW, 4, 0, 16, soffset, **fields).hex() == line

    # 2. The format set is the 4-channel subset of compute_store_aot's reviewed tables, in both encodings.
    four = {code for code, (_, channels) in A.DESCRIPTOR_STORE_FORMATS.items() if channels == 4}
    assert set(G.STORE_FORMAT_XYZW_FORMATS) == four, (sorted(G.STORE_FORMAT_XYZW_FORMATS), sorted(four))
    assert G.STORE_FORMAT_XYZW_LEGACY_FORMATS == {f: c for f, c in A.LEGACY_DESCRIPTOR_FORMATS.items() if c in four}

    # 3. Word 3 validator: exhaustive over the FORMAT field in both encodings and both OOB modes.
    for level in (0, 1):
        for field in range(128):
            code = field if level else G.STORE_FORMAT_XYZW_LEGACY_FORMATS.get(field)
            for oob in (0, 3):
                w3 = (oob << 28) | (level << 24) | (field << 12) | 0xfac
                if code in G.STORE_FORMAT_XYZW_FORMATS:
                    assert G.validate_store_format_xyzw_word3(w3) == (code, *G.STORE_FORMAT_XYZW_FORMATS[code])
                else:
                    refused('is not a handled 4-channel format', lambda w=w3: G.validate_store_format_xyzw_word3(w))
    for bad, fragment in ((desc_word3(77, dst_sel=0xfa5), 'non-identity DST_SEL 0xfa5'),
                          (desc_word3(77, dst_sel=0xfa4), 'non-identity DST_SEL'),
                          (desc_word3(77, oob_select=1), 'OOB_SELECT 1'), (desc_word3(77, oob_select=2), 'OOB_SELECT 2'),
                          (desc_word3(77) | (1 << 23), 'flags'), (desc_word3(77) | (1 << 21), 'flags'),
                          (desc_word3(77) | (1 << 30), 'flags'), (desc_word3(77) | (1 << 25), 'flags'),
                          (-1, 'invalid descriptor word 3'), (True, 'invalid descriptor word 3'),
                          (1 << 32, 'invalid descriptor word 3')):
        refused(fragment, lambda w=bad: G.validate_store_format_xyzw_word3(w))

    # 4. Exact emitted text, entry constants, helper only when used.
    problems = emitted_text_problems()
    assert not problems, problems
    plain = G.ComputePipeline(bytes.fromhex('000081bf'), root_blob(), USER7, (1, 1, 1)).hlsl_cs()
    assert 'store_format_xyzw' not in plain and 'store_unorm' not in plain

    # 5. Refusals naming the unmodelled shape (negative controls).
    for hex_word, fragment in (
        (VERIFIED_SNIPPETS['buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 offen'], 'offen address form'),
        (VERIFIED_SNIPPETS['buffer_store_format_xyzw v[4:7], v[0:1], s[16:19], 0 idxen offen'], 'offen address form'),
        (VERIFIED_SNIPPETS['buffer_store_format_xyzw v[4:7], off, s[16:19], 0'], 'without idxen'),
        (VERIFIED_SNIPPETS['buffer_store_format_xyzw v[4:7], v0, s[16:19], s4 idxen'],
         'soffset s4 (only a non-negative inline constant is modelled)'),
        (VERIFIED_SNIPPETS['buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 idxen glc'], 'modifier glc'),
        (VERIFIED_SNIPPETS['buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 idxen slc'], 'modifier slc'),
        (VERIFIED_SNIPPETS['buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 idxen dlc'], 'modifier dlc'),
        (VERIFIED_SNIPPETS['buffer_store_format_xyzw v[4:7], v0, s[16:19], 0 idxen tfe'], 'modifier tfe'),
        (encode_mubuf(MUBUF_STORE_FORMAT_XYZW, 4, 0, 16, 0x80, idxen=True, lds=True).hex(), 'illegal MUBUF encoding'),
        (encode_mubuf(MUBUF_STORE_FORMAT_XYZW, 4, 0, 16, 0x80, idxen=True, w0_rsvd=True).hex(), 'illegal MUBUF'),
        (encode_mubuf(MUBUF_STORE_FORMAT_XYZW, 4, 0, 16, 0x80, idxen=True, w1_rsvd=True).hex(), 'illegal MUBUF'),
        (encode_mubuf(MUBUF_STORE_FORMAT_XYZW, 4, 0, 12, 0x80, idxen=True).hex(), 'used for a load and a store'),
    ):
        refused(fragment, lambda hw=hex_word: translate('X', code=patch(WORDS_X, STORE_X, hw)).hlsl_cs())
    for inserted, fragment in (('800392be', 'modified descriptor'),              # s_mov_b32 s18, 0
                               ('11ff1187ffff00c0', 'modified descriptor'),      # s_and_b32 s17, s17, 0xc000ffff
                               ('10841080', 'modified descriptor'),              # s_add_u32 s16, s16, 4
                               ('800392be000082bf', 'modified descriptor')):     # s_mov_b32 s18, 0; s_branch +0
        refused(fragment, lambda i=inserted: translate('X', code=patch(WORDS_X, STORE_X, i + STORE_X)).hlsl_cs())
    # A descriptor loaded above a branch target (a jump to the next instruction) is the one writer on every path: proven.
    translate('X', code=patch(WORDS_X, STORE_X, '000082bf' + STORE_X)).hlsl_cs()
    for kind in ('ps', 'vs'):  # outside compute: the unknown-op refusal, the bare mnemonic
        stage = G.Stage.__new__(G.Stage)
        stage.kind = kind
        try:
            stage.op_buffer_store_format_xyzw(['v[4:7]', 'v0', 's[16:19]', '0'], ['idxen'])
        except G.Unsupported as reason:
            assert str(reason) == 'buffer_store_format_xyzw', str(reason)
        else:
            raise AssertionError('%s stage accepted buffer_store_format_xyzw' % kind)
    cs = translate('X').cs
    cs.raw_code = b''  # synthetic operand lists below
    for ops, mods, fragment in (
        (['v[4:7]', 'v0', 's[16:19]'], ['idxen'], 'with 3 operands, expected 4'),
        (['v[4:7]', 'v0', 's[16:19]', '0'], ['idxen', 'idxen'], 'modifier idxen'),
        (['v[4:7]', 'v0', 's[16:19]', '0'], ['idxen', 'offset:4', 'offset:8'], 'modifier offset:8'),
        (['v[4:7]', 'v0', 's[16:19]', '0'], ['idxen', 'offset:4096'], 'modifier offset:4096'),
        (['v[4:7]', 'v0', 's[16:19]', '0'], ['idxen', 'lds'], 'modifier lds'),
        (['v[4:6]', 'v0', 's[16:19]', '0'], ['idxen'], 'data v[4:6]'),
        (['v[253:256]', 'v0', 's[16:19]', '0'], ['idxen'], 'data v[253:256]'),
        (['s[4:7]', 'v0', 's[16:19]', '0'], ['idxen'], 'data s[4:7]'),
        (['v[4:7]', 'v[0:1]', 's[16:19]', '0'], ['idxen'], 'vaddr v[0:1] for idxen'),
        (['v[4:7]', 's0', 's[16:19]', '0'], ['idxen'], 'vaddr s0 for idxen'),
        (['v[4:7]', 'v0', 's[16:19]', '-1'], ['idxen'], 'soffset -1'),
        (['v[4:7]', 'v0', 's[16:19]', 'v1'], ['idxen'], 'soffset v1'),
        (['v[4:7]', 'v0', 's[16:17]', '0'], ['idxen'], 'descriptor s[16:17]'),
        (['v[4:7]', 'v0', 's[104:107]', '0'], ['idxen'], 'descriptor s[104:107]'),
    ):
        refused(fragment, lambda o=ops, m=mods: cs.op_buffer_store_format_xyzw(o, m))

    # 6. Translator mutations turn the checks red.
    saved = G.Stage.op_buffer_store_format_xyzw
    try:
        delattr(G.Stage, 'op_buffer_store_format_xyzw')
        refused('buffer_store_format_xyzw', lambda: translate('X').hlsl_cs())
    finally:
        G.Stage.op_buffer_store_format_xyzw = saved
    legacy = dict(G.STORE_FORMAT_XYZW_LEGACY_FORMATS)
    for mutation in ({**legacy, (14 << 3) | 4: 77}, {k: v for k, v in legacy.items() if v != 50}):
        G.STORE_FORMAT_XYZW_LEGACY_FORMATS = mutation
        try:
            assert 'helper text' in emitted_text_problems(), mutation
        finally:
            G.STORE_FORMAT_XYZW_LEGACY_FORMATS = legacy
    saved_emit = G.Stage.emit
    try:  # dropping the EXEC guard
        G.Stage.emit = lambda self, line: saved_emit(self, line.replace('if (exec & 1) store_format_xyzw', 'store_format_xyzw'))
        assert 'call E' in emitted_text_problems()
    finally:
        G.Stage.emit = saved_emit

    # 7. Dispatch inputs are self-consistent: writing cases change words, no-op cases keep every sentinel.
    for name in DISPATCH_CASES:
        _, _, _, expected = case_inputs(name)
        writes = X_VARIANTS[name][4] if name in X_VARIANTS else name != 'R-raw-oob'
        assert (expected != [SENTINEL] * WORDS) == writes, name

    print('buffer_store_format_xyzw reference, format, bounds, provenance, refusal and mutation checks pass')


def check_binary(llvm_mc):
    from startup_flat_param_check import assemble
    for name, (source, words, *_rest) in PROGRAMS.items():
        built = assemble(llvm_mc, source)
        assert built == words, '%s: assembled %s, recorded %s' % (name, built.hex(), words.hex())
    for line, hex_words in VERIFIED_SNIPPETS.items():
        built = assemble(llvm_mc, line + '\n').hex()
        assert built == hex_words, '%s: assembled %s, recorded %s' % (line, built, hex_words)
    print('buffer_store_format_xyzw fixture words match llvm-mc (%d programs, %d snippets)'
          % (len(PROGRAMS), len(VERIFIED_SNIPPETS)))


CONTROLS = {
    'x-drop-store': ('X-f32', '    store_format_xyzw(u0, ud2, ud3, ud4, v0, 0u, 0u, 0u, uint4(v4, v5, v6, v7));\n', ''),
    'x-swap-xy': ('X-f32', 'buf.Store4(addr, v);', 'buf.Store4(addr, v.yxzw);'),
    'x-drop-index-oob': ('X-f32',
                         'if (idx >= num_records || elem_bytes > stride || rec_off > stride - elem_bytes) return;',
                         'if (elem_bytes > stride || rec_off > stride - elem_bytes) return;'),
    'unorm8-trunc': ('X-unorm8', 'return (uint)floor(scaled + 0.5f);', 'return (uint)floor(scaled);'),
    'unorm10-w-scale': ('X-unorm10-legacy', '(store_unorm(v.w, 3.0f) << 30u)', '(store_unorm(v.w, 1.0f) << 30u)'),
    'f16-drop-odd': ('X-f16', 'pack16_rne(asfloat(v.x)) |', '(pack16_rne(asfloat(v.x)) & 0xfffeu) |'),
    'mismatch-writes': ('X-1ch', '(fmt == 50u || fmt == 56u) ? 4u : 0u', '(fmt == 50u || fmt == 56u || fmt == 22u) ? 4u : 0u'),
    'r-drop-soffset': ('R-struct', 'v0, 0u, 4u, 12u, uint4', 'v0, 0u, 0u, 12u, uint4'),
    'e-drop-exec': ('E', 'if (exec & 1) store_format_xyzw(', 'store_format_xyzw('),
}


def run_case(name, dxc, runner, adapter, directory, control=None):
    prog, source, constants, expected = case_inputs(name)
    hlsl = translate(prog).hlsl_cs()
    if control:
        _, old, new = CONTROLS[control]
        assert hlsl.count(old) == 1, (control, old)
        hlsl = hlsl.replace(old, new)
    src, cso, case = (directory / ('%s.%s' % (name, ext)) for ext in ('hlsl', 'cso', 'case'))
    src.write_text(hlsl, encoding='ascii', newline='\n')
    subprocess.run([dxc, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    groups = PROGRAMS[prog][4]
    case.write_bytes(struct.pack('<I', groups[0] | (groups[1] << 8) | (groups[2] << 16))
                     + struct.pack('<%dI' % WORDS, *expected) + struct.pack('<16I', *constants) + source)
    # The helper's GetDimensions needs a descriptor-table view; root descriptors carry no size.
    return subprocess.run([runner, adapter, '--compute-store-sized-case', str(cso), str(case)],
                          capture_output=True, text=True)


def check_dispatch(dxc, runner, adapter, control=None):
    runner = str(Path(runner).resolve())
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        runs = [(CONTROLS[control][0], control)] if control else [(name, None) for name in DISPATCH_CASES]
        for name, mutation in runs:
            run = run_case(name, dxc, runner, adapter, Path(tmp), mutation)
            print(run.stdout, end='')
            if run.returncode == 77:
                return 77
            diagnostics = [line for line in run.stderr.splitlines()
                           if not (adapter == 'hardware' and line.startswith('XBOX2WINDOWS_ADAPTER: skipping '))]
            if mutation:
                if run.returncode != 1 or not any('expected=' in line for line in diagnostics):
                    raise AssertionError('control %s: not rejected (exit %d) %s' % (mutation, run.returncode, run.stderr))
                print('control %s: rejected at %s' % (mutation, diagnostics[0]))
            elif run.returncode or diagnostics:
                raise AssertionError('case %s: %s' % (name, run.stderr or 'exit %d' % run.returncode))
            else:
                print('case %s: all %d words match' % (name, WORDS))
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        check_controls()
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        check_binary(sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        sys.exit(check_dispatch(*sys.argv[2:]))
    elif len(sys.argv) == 6 and sys.argv[1] == 'control':
        sys.exit(check_dispatch(*sys.argv[3:], control=sys.argv[2]))
    else:
        sys.exit(__doc__)
