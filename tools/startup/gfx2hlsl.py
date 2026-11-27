"""gfx1013 machine code -> HLSL for a vertex+pixel pipeline (startup tools; see docs/startup-runtime.md).

Model: one lane. Registers are uint variables; exec/vcc/mask registers hold the lane's bit in bit 0.
Scalar registers are also tracked symbolically at translation time so that pointer and descriptor
values (tables, vertex-buffer table, root buffers) turn into HLSL resources instead of arithmetic.
The output is HLSL for both stages plus a manifest that tells the runtime what to bind where.
"""
import bisect
from collections import Counter
from contextlib import suppress
from copy import deepcopy
import functools
import hashlib
import math
import heapq
import os
from pathlib import Path
import re
import struct
import sys
import sysconfig
import tempfile
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import disasm_rdna2 as D  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_exec as LE  # noqa: E402
from shader_minmax import FLOAT_MINMAX  # noqa: E402


WAVE_REDUCTION_BRANCHES = ('wave reduction in a program with branches: only straight-line pixel stages are '
                           'lowered')
READFIRSTLANE_UNIFORM = 'v_readfirstlane_b32 requires a proven uniform source'


class Unsupported(Exception):
    pass


PREFETCH = bytes.fromhex('0300a0bf')

COMPUTE_DPP_FLOAT = frozenset(('v_min_f32', 'v_max_f32', 'v_add_f32'))

# DPP16 quad pair broadcasts (quad_perm of one lane of every horizontal or vertical pair into the other). The lanes of
# a quad are the pixels 0 1 / 2 3, row by row; lane i reads lane quad_perm[i].
DPP_PAIR = frozenset(((1, 1, 3, 3), (0, 0, 2, 2), (2, 3, 2, 3), (0, 1, 0, 1)))
# Exchanges projected onto one row (X difference) or one column (Y difference).
DPP_COARSE = frozenset(((1, 0, 1, 0), (3, 2, 3, 2), (2, 2, 0, 0), (3, 3, 1, 1)))
DPP_DERIVATIVE_OPS = frozenset(('v_mov_b32', 'v_sub_f32', 'v_subrev_f32'))
DPP_SCRATCH = 256

# SDWA (sub-dword addressing) selections, with the same meaning as tools/pixel_aot.py: a source
# selection is zero-extended, or sign-extended under sext(); a destination selection writes the low
# bits of the result into that field, UNUSED_PAD zeroes the others and UNUSED_PRESERVE keeps them.
SDWA_SEL = {'BYTE_0': (0, 8), 'BYTE_1': (8, 8), 'BYTE_2': (16, 8), 'BYTE_3': (24, 8),
            'WORD_0': (0, 16), 'WORD_1': (16, 16), 'DWORD': (0, 32)}


def sopk_imm(token, signed):
    """The 32-bit word a SOPK 16-bit immediate stands for.

    RDNA 2 ISA SOPK: s_movk_i32 and the s_cmpk_*_i32 forms use signext(SIMM16[15:0]) (LLVM s16imm); the s_cmpk_*_u32
    forms use SIMM16[15:0] zero-extended (LLVM u16imm). The disassembler prints the field as unsigned hex."""
    m = re.fullmatch(r'(-?)(?:0x([0-9a-f]+)|(\d+))', token)
    if not m:
        raise Unsupported('SOPK immediate %r' % token)
    word = int(m.group(2), 16) if m.group(2) else int(m.group(3))
    if m.group(1):
        word = -word
    if not (-0x8000 if signed else 0) <= word <= 0xffff:
        raise Unsupported('SOPK immediate %r outside 16 bits' % token)
    if signed:
        word = ((word & 0xffff) ^ 0x8000) - 0x8000
    return word & 0xffffffff


def sdwa_source(word, sel, sext):
    """The uint expression a source selection reads out of a 32-bit register."""
    offset, width = SDWA_SEL[sel]
    if width == 32:
        return word
    if sext:
        return 'asuint(asint(%s << %du) >> %du)' % (word, 32 - offset - width, 32 - width)
    return '((%s >> %du) & %#xu)' % (word, offset, (1 << width) - 1)


def sdwa_dest(result, sel, unused, old):
    """The uint expression a destination selection writes back into a 32-bit register."""
    offset, width = SDWA_SEL[sel]
    if width == 32:
        return result
    if unused == 'UNUSED_SEXT':
        raise Unsupported('SDWA UNUSED_SEXT')
    field = '((%s & %#xu) << %du)' % (result, (1 << width) - 1, offset)
    if unused == 'UNUSED_PAD':
        return field
    if unused != 'UNUSED_PRESERVE':
        raise Unsupported('SDWA dst_unused %s' % unused)
    keep = ~(((1 << width) - 1) << offset) & 0xffffffff
    return '((asuint(%s) & %#xu) | %s)' % (old, keep, field)


def gather_2d_profile_words(words):
    """The sampler words a 2D gather is checked with: fields a 2D gather cannot observe are set to the profile.

    SQ_IMG_SAMP_WORD0 bits 8:6 (CLAMP_Z) address the third coordinate, which a 2D view does not have; D3D11.3
    functional spec 22.4.2 (gather4): "Only the addressing modes of the sampler are used". For the same reason a
    point XY_MAG_FILTER or XY_MIN_FILTER (WORD2 bits 21:20 and 23:22, value 0) reads the same four texels as the
    bilinear one: gather4 "Gathers the four texels that would be used in a bi-linear filtering operation", whatever
    the sampler's filter, so a point filter is checked as bilinear. That point and bilinear read the same four
    texels is the D3D11.3 rule; it has not been measured on console hardware. Anisotropic filters (2, 3) keep
    their values and stay refused. WORD2 bit 28 (MIP_POINT_PRECLAMP) acts only on mip selection, and the _lz
    gather reads mip zero ("the top level of any mip pyramid is used", the same spec section), so it is cleared
    as well; the level-zero comparison path (pixel_aot.comparison_sampler_state) accepts it for the same reason.
    Bits 29-31 stay checked. The words bound as the static sampler stay the title's own.
    """
    w0, w1, w2, w3 = words
    for shift in (20, 22):
        if (w2 >> shift) & 3 == 0:
            w2 |= 1 << shift
    return (w0 & ~(7 << 6), w1, w2 & ~(1 << 28), w3)


def strip_header(code):
    at = code.find(PREFETCH, 0, 0x80)
    # A call (s_swappc_b64) just before the prefetch is the program's first instruction, not header: keep it.
    if at >= 4 and at % 4 == 0 and struct.unpack_from('<I', code, at - 4)[0] & 0xff80ff00 == 0xbe802100:
        at -= 4
    return code[at:] if at >= 0 and at % 4 == 0 else code


# A stream vertex shader whose inputs come from a fetch routine the driver builds from the input layout starts
# with this call; the routine's address arrives in s[4:5]. Its FSSM header lists, from +0x138, one 5-byte record
# per fetched input: kind (6), input-signature register, component mask, first VGPR, component count - 1.
# Checked offline against the input signatures of every readable vertex program of one title's shader archive
# that starts with the call: each record's register and mask match a signature input (the mask is the read
# mask; the count reaches its highest component), the first VGPR is v4 and each next record is 4 VGPRs on.
# Programs without the call have no records. The header's u32 at +0x1d8 holds the record count.
# Any other header size, record kind or count is refused.
FETCH_CALL = 0xbe842104  # s_swappc_b64 s[4:5], s[4:5]
FETCH_HEADER_BASE, FETCH_RECORDS, FETCH_KIND, FETCH_COUNT = 0x2f8, 0x138, 6, 0x1d8
FETCH_SLOTS = 8  # vertex-buffer slots the stand-in binds (D3D12_VERTEX_BUFFER_VIEW vb[8])
# DXGI format -> (BUF_FMT name in FETCH_FORMATS, bytes per element).
FETCH_DXGI = {2: ('BUF_FMT_32_32_32_32_FLOAT', 16), 6: ('BUF_FMT_32_32_32_FLOAT', 12), 16: ('BUF_FMT_32_32_FLOAT', 8),
              41: ('BUF_FMT_32_FLOAT', 4), 28: ('BUF_FMT_8_8_8_8_UNORM', 4), 49: ('BUF_FMT_8_8_UNORM', 2),
              61: ('BUF_FMT_8_UNORM', 1), 34: ('BUF_FMT_16_16_FLOAT', 4), 35: ('BUF_FMT_16_16_UNORM', 4),
              37: ('BUF_FMT_16_16_SNORM', 4), 30: ('BUF_FMT_8_8_8_8_UINT', 4),
              24: ('BUF_FMT_2_10_10_10_UNORM', 4)}
# Typed buffer loads: (word load or None, the four component expressions). Components a format lacks read as
# 0, 0, 0, 1 here, as in the existing tbuffer translation; that default is an assumption, not a measurement.
FETCH_FORMATS = {
    'BUF_FMT_8_8_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', 'U(((w >> 16) & 255) / 255.0)', 'U((w >> 24) / 255.0)']),
    'BUF_FMT_8_8_8_8_UINT': ('ldu({0}, a)', ['w & 255u', '(w >> 8) & 255u', '(w >> 16) & 255u', 'w >> 24']),
    'BUF_FMT_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', '0u', 'U(1.0)']),
    'BUF_FMT_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', '0u', '0u', 'U(1.0)']),
    'BUF_FMT_8_UINT': ('{0}.Load(a & ~3u) >> ((a & 3u) * 8u)', ['w & 255u', '0u', '0u', '1u']),
    # Three 10-bit fields and a 2-bit one in one dword, lowest bits first: x, y, z, then w.
    'BUF_FMT_2_10_10_10_UNORM': ('ldu({0}, a)', ['U((w & 1023) / 1023.0)', 'U(((w >> 10) & 1023) / 1023.0)', 'U(((w >> 20) & 1023) / 1023.0)', 'U((w >> 30) / 3.0)']),
    'BUF_FMT_16_16_UNORM': ('ldu({0}, a)', ['U((w & 65535) / 65535.0)', 'U((w >> 16) / 65535.0)', '0u', 'U(1.0)']),
    # Two unsigned 16-bit fields, x in the low half (LLVM 15: BUF_DATA_FORMAT_16_16 with BUF_NUM_FORMAT_UINT,
    # gfx10 unified format 27); a missing W of an integer format reads 1.
    'BUF_FMT_16_16_UINT': ('ldu({0}, a)', ['w & 65535u', 'w >> 16', '0u', '1u']),
    # SNORM: sign-extend each half, divide by 32767 and clamp to -1.0, so that -32768 and -32767 both give -1.0
    # (Direct3D / Vulkan SNORM rule; LLVM 15 and Mesa ac_formats name the format 16_16_SNORM, number 24).
    'BUF_FMT_16_16_SNORM': ('ldu({0}, a)', ['U(max(float(int(w << 16) >> 16) / 32767.0, -1.0))',
                                            'U(max(float(int(w) >> 16) / 32767.0, -1.0))', '0u', 'U(1.0)']),
    'BUF_FMT_16_16_FLOAT': ('ldu({0}, a)', ['U(f16tof32(w))', 'U(f16tof32(w >> 16))', '0u', 'U(1.0)']),
    # Four half channels in one 8-byte element: x/y in the first dword, z/w in the second.
    'BUF_FMT_16_16_16_16_FLOAT': (None, ['U(f16tof32(ldu({0}, a)))', 'U(f16tof32(ldu({0}, a) >> 16))',
                                         'U(f16tof32(ldu({0}, a + 4)))', 'U(f16tof32(ldu({0}, a + 4) >> 16))']),
    'BUF_FMT_32_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', '{0}.Load(a + 12)']),
    'BUF_FMT_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', 'U(1.0)']),
    'BUF_FMT_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '0u', 'U(1.0)']),
    'BUF_FMT_32_FLOAT': (None, ['{0}.Load(a)', '0u', '0u', 'U(1.0)'])}


# gfx10 buffer-descriptor FORMAT field (word 3, bits 12..18) for each FETCH_FORMATS entry: the same unified
# numbers as the tbuffer `format:` operand (tools/disasm_rdna2.py MTBUF_FORMATS; 1 is the omitted default).
# A MUBUF buffer_load_format_* has no format operand, so its translation switches on this field at run time.
# An invalid FORMAT reaches the switch default, which poisons the decoded components with this NaN pattern.
# Selectors for constants still return constants, so the caller must validate FORMAT before submission.
DESC_FORMAT_UNKNOWN = 0x7fc0f0f0
DESC_FORMAT_CODES = {
    'BUF_FMT_8_UNORM': 1, 'BUF_FMT_8_UINT': 5, 'BUF_FMT_8_8_UNORM': 14, 'BUF_FMT_32_FLOAT': 22,
    'BUF_FMT_2_10_10_10_UNORM': 50, 'BUF_FMT_16_16_UNORM': 23, 'BUF_FMT_16_16_SNORM': 24, 'BUF_FMT_16_16_UINT': 27, 'BUF_FMT_16_16_FLOAT': 29, 'BUF_FMT_8_8_8_8_UNORM': 56, 'BUF_FMT_8_8_8_8_UINT': 60,
    'BUF_FMT_32_32_FLOAT': 64, 'BUF_FMT_16_16_16_16_FLOAT': 71, 'BUF_FMT_32_32_32_FLOAT': 74,
    'BUF_FMT_32_32_32_32_FLOAT': 77}
DESC_FORMAT_SHIFT, DESC_FORMAT_MASK = 12, 127
# Integer formats read a DST_SEL constant 1 as the integer 1, every other format as 1.0.
DESC_ONE_INTEGER = ('BUF_FMT_8_UINT', 'BUF_FMT_16_16_UINT', 'BUF_FMT_8_8_8_8_UINT')

# Supported 1-channel formats for compute buffer_store_format_d16_x (RDNA 2 MUBUF 0x84):
# V# word 3 bits 18:12 carry the 7-bit GFX10 unified FORMAT when RESOURCE_LEVEL (bit 24) is 1,
# or legacy (DFMT << 3) | NFMT when bit 24 is 0. Bits checked zero: TYPE[31:30], reserved[27:25],
# ADD_TID_ENABLE[23], INDEX_STRIDE[22:21], reserved[20:19]. DST_SEL_X[2:0] must be 4 (SQ_SEL_X);
# OOB_SELECT[29:28] must be 0 (SQ_OOB_INDEX_AND_OFFSET, structured with stride != 0) or
# 3 (SQ_OOB_COMPLETE, raw byte bounds with stride == 0). Modes 1 (INDEX_ONLY) and 2
# (NUM_RECORDS_0 / disabled bounds check) are refused. Entry constant uav_stride carries the
# unmasked upper 16 bits of V# word 1 (DW1 >> 16), so stride > 0x3fffu rejects CACHE_SWIZZLE
# (bit 30) and SWIZZLE_ENABLE (bit 31).
STORE_D16_X_FORMATS = {
    7: ('BUF_FMT_16_UNORM', 2),
    8: ('BUF_FMT_16_SNORM', 2),
    11: ('BUF_FMT_16_UINT', 2),
    12: ('BUF_FMT_16_SINT', 2),
    13: ('BUF_FMT_16_FLOAT', 2),
    22: ('BUF_FMT_32_FLOAT', 4),
}
STORE_D16_X_LEGACY_FORMATS = {
    (2 << 3) | 0: 7,   # DFMT_16 + NFMT_UNORM
    (2 << 3) | 1: 8,   # DFMT_16 + NFMT_SNORM
    (2 << 3) | 4: 11,  # DFMT_16 + NFMT_UINT
    (2 << 3) | 5: 12,  # DFMT_16 + NFMT_SINT
    (2 << 3) | 7: 13,  # DFMT_16 + NFMT_FLOAT
    (4 << 3) | 7: 22,  # DFMT_32 + NFMT_FLOAT
}
STORE_D16_X_WORD3_ZERO_MASK = 0xcef80000  # bits 31:30 TYPE | 27:25 reserved | 23 ADD_TID_ENABLE | 22:21 INDEX_STRIDE | 20:19 reserved


def validate_store_d16_x_word3(word3):
    """Validate V# word 3 for compute buffer_store_format_d16_x and return (code, name, elem_bytes)."""
    if not isinstance(word3, int) or isinstance(word3, bool) or not 0 <= word3 <= 0xffffffff:
        raise Unsupported('buffer_store_format_d16_x invalid descriptor word 3 %r' % (word3,))
    if word3 & STORE_D16_X_WORD3_ZERO_MASK:
        raise Unsupported('buffer_store_format_d16_x unsupported descriptor word 3 flags 0x%08x' % word3)
    if (word3 & 7) != 4:
        raise Unsupported('buffer_store_format_d16_x non-identity DST_SEL_X %d' % (word3 & 7))
    oob_select = (word3 >> 28) & 3
    if oob_select not in (0, 3):
        raise Unsupported('buffer_store_format_d16_x unsupported OOB_SELECT %d' % oob_select)
    field = (word3 >> DESC_FORMAT_SHIFT) & DESC_FORMAT_MASK
    code = field if ((word3 >> 24) & 1) else STORE_D16_X_LEGACY_FORMATS.get(field)
    info = STORE_D16_X_FORMATS.get(code)
    if info is None:
        raise Unsupported('buffer_store_format_d16_x unsupported descriptor FORMAT %d' % field)
    return code, info[0], info[1]


# Compute buffer_store_format_xyzw (MUBUF 0x07) writes only formats with exactly four channels, the set and
# conversions of tools/compute_store_aot.py: a channel-count mismatch writes nothing, since what the format unit
# does with the extra components is not established here. Word 3 bounds and OOB_SELECT checks are those of
# buffer_store_format_d16_x above; DST_SEL must be the identity X, Y, Z, W (0xfac).
STORE_FORMAT_XYZW_FORMATS = {
    50: ('BUF_FMT_2_10_10_10_UNORM', 4),
    56: ('BUF_FMT_8_8_8_8_UNORM', 4),
    71: ('BUF_FMT_16_16_16_16_FLOAT', 8),
    75: ('BUF_FMT_32_32_32_32_UINT', 16),
    77: ('BUF_FMT_32_32_32_32_FLOAT', 16),
}
STORE_FORMAT_XYZW_LEGACY_FORMATS = {
    (9 << 3) | 0: 50,   # DFMT_2_10_10_10 + NFMT_UNORM
    (10 << 3) | 0: 56,  # DFMT_8_8_8_8 + NFMT_UNORM
    (12 << 3) | 7: 71,  # DFMT_16_16_16_16 + NFMT_FLOAT
    (14 << 3) | 4: 75,  # DFMT_32_32_32_32 + NFMT_UINT
    (14 << 3) | 7: 77,  # DFMT_32_32_32_32 + NFMT_FLOAT
}


def validate_store_format_xyzw_word3(word3):
    """Validate V# word 3 for compute buffer_store_format_xyzw and return (code, name, elem_bytes)."""
    name = 'buffer_store_format_xyzw'
    if not isinstance(word3, int) or isinstance(word3, bool) or not 0 <= word3 <= 0xffffffff:
        raise Unsupported('%s invalid descriptor word 3 %r' % (name, word3))
    if word3 & STORE_D16_X_WORD3_ZERO_MASK:
        raise Unsupported('%s unsupported descriptor word 3 flags 0x%08x' % (name, word3))
    if (word3 & 0xfff) != 0xfac:
        raise Unsupported('%s non-identity DST_SEL 0x%03x' % (name, word3 & 0xfff))
    oob_select = (word3 >> 28) & 3
    if oob_select not in (0, 3):
        raise Unsupported('%s unsupported OOB_SELECT %d' % (name, oob_select))
    field = (word3 >> DESC_FORMAT_SHIFT) & DESC_FORMAT_MASK
    code = field if ((word3 >> 24) & 1) else STORE_FORMAT_XYZW_LEGACY_FORMATS.get(field)
    info = STORE_FORMAT_XYZW_FORMATS.get(code)
    if info is None:
        raise Unsupported('%s descriptor FORMAT %d is not a handled 4-channel format' % (name, field))
    return code, info[0], info[1]

def starts_with_fetch_call(code):
    return len(code) >= 4 and struct.unpack_from('<I', code)[0] == FETCH_CALL


def parse_input_layout(text):
    """'ia1 <n>' then one '<semantic> <index> <format> <slot> <offset> <class> <step>' line per element."""
    lines = text.split('\n')
    # 'ia1 0' is a layout the pipeline declares with no elements: no fetched input has a source.
    if not re.fullmatch(r'ia1 (0|[1-9]|[12][0-9]|3[0-2])', lines[0]) or lines[-1] != '' or len(lines) != int(lines[0][4:]) + 2:
        raise Unsupported('invalid input layout request')
    elements, ends = [], {}
    for line in lines[1:-1]:
        m = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]{0,62}) (\d{1,10}) (\d{1,10}) (\d{1,2}) (\d{1,10}) (\d) (\d{1,10})', line)
        if not m:
            raise Unsupported('invalid input layout element')
        name, index, fmt, slot, offset, cls, step = m.group(1).upper(), *map(int, m.groups()[1:])
        if offset == 0xffffffff:  # D3D12_APPEND_ALIGNED_ELEMENT: right after the previous element of the slot
            # shortcut: unknown when an earlier size is unknown or ends off a dword (packing not measured).
            offset = ends.get(slot, 0)
            if offset is not None and offset % 4:
                offset = None
        size = FETCH_DXGI[fmt][1] if fmt in FETCH_DXGI else None
        ends[slot] = None if offset is None or size is None else offset + size
        # Format, class and slot are checked in stream_fetch, only for the elements a record fetches.
        elements.append(dict(name=name, index=index, format=fmt, slot=slot, offset=offset, cls=cls, step=step))
    return elements


def signature_inputs(isg1):
    """ISG1 part -> {register: (semantic name, semantic index, system value)}."""
    count, first = struct.unpack_from('<II', isg1, 0)
    out = {}
    for i in range(count):
        _, name_at, index, system, _, register, _ = struct.unpack_from('<6IB', isg1, first + 32 * i)
        out[register] = (isg1[name_at:isg1.index(b'\0', name_at)].decode('ascii').upper(), index, system)
    return out


def stream_fetch(vs_container, layout_text):
    """The fetch a stream vertex shader expects before its first instruction: [dict(vgpr, count, ...layout)]."""
    from pixel_aot import container_image
    from resource_inventory import parts
    if layout_text is None:
        raise Unsupported('vertex stage calls a fetch routine and the pipeline declares no input layout')
    sections = parts(vs_container)
    image = container_image(sections[b'DXIL'], stage=1)
    if len(image) < FETCH_HEADER_BASE or struct.unpack_from('<Q', image, 192)[0] != FETCH_HEADER_BASE:
        raise Unsupported('unverified FSSM header for a vertex fetch table')
    inputs, layout = signature_inputs(sections[b'ISG1']), parse_input_layout(layout_text)
    fetch, at = [], FETCH_RECORDS
    while image[at] != 0:
        kind, register, mask, vgpr, last = image[at:at + 5]
        count = last + 1
        # The count reaches the highest component the shader reads (mask 0x4 comes with count 3): load all of them.
        if (kind != FETCH_KIND or mask >> last != 1 or not 1 <= count <= 4 or vgpr < 1 or vgpr + count > 256
                or len(fetch) == 16):
            raise Unsupported('unverified vertex fetch record %s' % image[at:at + 5].hex())
        if register not in inputs:
            raise Unsupported('vertex fetch record for input register %d outside the signature' % register)
        name, index, _ = inputs[register]
        if not layout:
            # Declared layout without elements: the input has no vertex buffer; its registers read as zero
            # (shortcut: the zero default is an assumption, not a measurement).
            fetch.append(dict(vgpr=vgpr, count=count, empty=True))
            at += 5
            continue
        match = [e for e in layout if (e['name'], e['index']) == (name, index)]
        if len(match) != 1:
            raise Unsupported('input layout has %d elements for vertex input register %d' % (len(match), register))
        e = match[0]
        if e['format'] not in FETCH_DXGI:
            raise Unsupported('input layout format %d' % e['format'])
        if e['cls'] != 0 or e['step'] != 0:
            raise Unsupported('per-instance input layout element')
        if e['slot'] >= FETCH_SLOTS:
            raise Unsupported('input layout slot %d: the stand-in binds vertex-buffer slots 0 to %d' % (e['slot'], FETCH_SLOTS - 1))
        if e['offset'] is None:
            raise Unsupported('appended input layout element after an element of unknown or unaligned size')
        fetch.append(dict(e, vgpr=vgpr, count=count))
        at += 5
    if not fetch:
        raise Unsupported('vertex stage calls a fetch routine and its container lists no fetched inputs')
    stored = struct.unpack_from('<I', image, FETCH_COUNT)[0]
    if stored != len(fetch):
        raise Unsupported('vertex fetch header counts %d records, the table holds %d' % (stored, len(fetch)))
    taken = [r for e in fetch for r in range(e['vgpr'], e['vgpr'] + e['count'])]
    if len(taken) != len(set(taken)):
        raise Unsupported('vertex fetch records overlap')
    return fetch


# The interface between the two stages of a stream pipeline, as both containers state it in their native headers.
# Vertex: from +0x1dc one 6-byte record per exported parameter (kind 6, output-signature register, parameter index,
# 0, 0, 0), count at +0x29c. Pixel (header 0xef8 bytes): from +0x138 one 8-byte record per attribute (kind 6,
# input-signature register, component mask, attribute index, 0, 0, the register again, interpolation 1 or 2), count
# at +0x238. Records with other values pack two signature registers into one parameter or attribute; that form is
# refused here.
IFACE_VS_RECORDS, IFACE_VS_COUNT, IFACE_PS_HEADER_BASE, IFACE_PS_RECORDS, IFACE_PS_COUNT = 0x1dc, 0x29c, 0xef8, 0x138, 0x238


def signature_entries(part):
    """ISG1/OSG1 part -> [(register, semantic name, semantic index, system value, mask)]."""
    count, first = struct.unpack_from('<II', part, 0)
    out = []
    for i in range(count):
        _, name_at, index, system, _, register, mask = struct.unpack_from('<6IB', part, first + 32 * i)
        out.append((register, part[name_at:part.index(b'\0', name_at)].decode('ascii').upper(), index, system, mask))
    return out


def stream_interface(vs_container, ps_container):
    """{pixel attribute: vertex parameter}, matched by semantic through both containers' interface tables."""
    from pixel_aot import container_image
    from resource_inventory import parts
    vs, ps = parts(vs_container), parts(ps_container)
    inputs = [entry for entry in signature_entries(ps[b'ISG1']) if not entry[3]] if b'ISG1' in ps else []
    if not inputs:
        return {}  # nothing to link: a read of any attribute is refused in note_attr
    vs_image, ps_image = container_image(vs[b'DXIL'], stage=1), container_image(ps[b'DXIL'], stage=0)
    if (struct.unpack_from('<Q', vs_image, 192)[0] != FETCH_HEADER_BASE or
            struct.unpack_from('<Q', ps_image, 192)[0] != IFACE_PS_HEADER_BASE):
        raise Unsupported('unverified FSSM header for the stage interface')
    attributes = {}
    count = struct.unpack_from('<I', ps_image, IFACE_PS_COUNT)[0]
    if count > (IFACE_PS_COUNT - IFACE_PS_RECORDS) // 8:
        raise Unsupported('pixel input table exceeds its header range')
    for i in range(count):
        kind, register, mask, attr, b4, b5, again, interp = ps_image[IFACE_PS_RECORDS + 8 * i:IFACE_PS_RECORDS + 8 * i + 8]
        if (kind != 6 or attr != i or b4 or b5 or again != register or interp not in (1, 2) or not mask
                or mask & ~0xf or register in attributes):
            raise Unsupported('unverified pixel input record %s' % ps_image[IFACE_PS_RECORDS + 8 * i:IFACE_PS_RECORDS + 8 * i + 8].hex())
        declared = 0
        for entry in inputs:
            if entry[0] == register:
                declared |= entry[4]
        if mask != declared:
            raise Unsupported('pixel input record mask 0x%x differs from the signature mask 0x%x of register %d'
                              % (mask, declared, register))
        attributes[register] = mask
    if not attributes:
        return {}
    params = {}
    count = struct.unpack_from('<I', vs_image, IFACE_VS_COUNT)[0]
    if count > (IFACE_VS_COUNT - IFACE_VS_RECORDS) // 6:
        raise Unsupported('vertex output table exceeds its header range')
    for i in range(count):
        kind, register, param, *rest = vs_image[IFACE_VS_RECORDS + 6 * i:IFACE_VS_RECORDS + 6 * i + 6]
        if kind != 6 or param != i or any(rest) or register in params:
            raise Unsupported('unverified vertex output record %s' % vs_image[IFACE_VS_RECORDS + 6 * i:IFACE_VS_RECORDS + 6 * i + 6].hex())
        params[register] = param
    outputs = {}
    for register, name, index, system, mask in signature_entries(vs[b'OSG1']):
        if system:
            continue
        if (name, index) in outputs:
            raise Unsupported('duplicate vertex output semantic %s%d' % (name, index))
        outputs[name, index] = register, mask
    mapping = {}
    for attr, (register, mask) in enumerate(attributes.items()):
        sources = set()
        for _, name, index, _, semantic_mask in (e for e in inputs if e[0] == register):
            output = outputs.get((name, index))
            if not output or output[1] != semantic_mask:
                raise Unsupported('pixel input %s%d has no vertex output with the same components' % (name, index))
            sources.add(output[0])
        if len(sources) != 1 or next(iter(sources)) not in params:
            raise Unsupported('pixel attribute %d does not come from exactly one exported vertex register' % attr)
        mapping[attr] = (params[next(iter(sources))], mask)
    return mapping


# Binary32 words for inline source encodings 240 through 248, in encoding order.
CNDMASK_INLINE_FLOAT_BITS = (0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
                            0x40000000, 0xc0000000, 0x40800000, 0xc0800000,
                            0x3e22f983)  # 1/(2*pi)


INTERP_DST_HIGH = 'op_sel:[0,0,0,1]'


def check_encoding(code, off, n, name, asm):
    # These SOP1 forms have an unused field that the disassembler does not print.
    # Only the ordinary four-byte encodings (no literal or hidden operand) are used
    # by the calculated-branch proof.
    if name in ('s_getpc_b64', 's_setpc_b64'):
        word, = struct.unpack_from('<I', code, off)
        unused = word & (0xff if name == 's_getpc_b64' else 0x7f0000)
        if n != 4 or unused:
            raise Unsupported('s_setpc_b64: unsupported %s encoding at +0x%x' % (name, off))
    if name in ('v_interp_p1ll_f16', 'v_interp_p2_f16'):
        # The shared disassembler ignores op_sel (w0[14:11]) here, as LLVM 15 does (its VOP3Interp_gfx10 encoding
        # leaves bits 14:11 undefined). Only Mesa ACO states a meaning: v_interp_p2_hi_f16 is p2 with op_sel 0x8 (bit
        # 14, aco_assembler.cpp), which writes the high half of the destination; register allocation turns p2 into it
        # for a 16-bit value at byte 2 of a register that holds another 16-bit value. It is passed on as the printed
        # LLVM modifier `op_sel:[0,0,0,1]`. The attribute half `high` is a different bit (w1[8]) and is printed.
        sel = struct.unpack_from('<I', code, off)[0] >> 11 & 15
        if sel == 8 and name == 'v_interp_p2_f16':
            return asm + ' ' + INTERP_DST_HIGH
        if sel:
            raise Unsupported('%s with op_sel 0x%x: only the plain form and, on v_interp_p2_f16, op_sel 0x8 '
                              '(destination high half) are lowered' % (name, sel))
    if name == 'image_load_mip':
        # The shared disassembler does not expose these gfx10 MIMG flags.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & (1 << 16):
            raise Unsupported('image_load_mip tfe')
        if word1 & (1 << 30):
            raise Unsupported('image_load_mip a16')
    if name == 'global_load_dword':
        if n != 8 or off + 8 > len(code):
            raise Unsupported('global_load_dword encoding length %d at +0x%x' % (n, off))
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & (1 << 25):
            raise Unsupported('global_load_dword reserved bit 25 set at +0x%x' % off)
        if word1 & (1 << 23):
            raise Unsupported('global_load_dword nv')
        if word1 & 0x0000ff00:
            raise Unsupported('global_load_dword unused data field set at +0x%x' % off)
    if name in LDS_UNUSED_W1_MASK:
        if n != 8 or off + 8 > len(code):
            raise Unsupported('%s encoding length %d at +0x%x' % (name, n, off))
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & (1 << 16):
            raise Unsupported('%s reserved bit 16 set at +0x%x' % (name, off))
        if word1 & LDS_UNUSED_W1_MASK[name]:
            raise Unsupported('%s unused operand field set at +0x%x' % (name, off))
    if re.fullmatch(r'v_cmpx?_(lt|eq|le|gt|ne|ge)_[iu]16_e64', name):
        # The decoder prints NEG on an inline integer as a sign (inline 1 with NEG reads `-1`, the same text as
        # inline -1), and LLVM 15 rejects `-v1` and `|v1|` on these sources: refuse either bit (VOP3 w0[10:8] ABS,
        # w1[31:29] NEG) so a printed -N is always a genuine inline integer.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & 0x700 or word1 & 0xe0000000:
            raise Unsupported('%s with neg/abs: not a 16-bit integer source modifier' % name)
    if name in ('v_sad_u8', 'v_sad_hi_u8', 'v_sad_u16', 'v_sad_u32', 'v_msad_u8', *VOP3_ONLY_VEC_U):
        # VOP3A integer forms take no source abs/neg; refuse those bits here so neg on an inline
        # integer does not print as -N and get accepted as a negative constant.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & 0x700 or word1 & 0xe0000000:
            raise Unsupported('%s with neg/abs' % name)
        if name in VOP3_ONLY_VEC_U and (word1 >> 18) & 0x1ff:
            raise Unsupported('%s with unused src2 field set' % name)
    if name in ('v_cvt_pknorm_u16_f32', 'v_cvt_pknorm_i16_f32'):
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & 0xfc00 or word1 & 0x9ffc0000:
            raise Unsupported('%s with reserved operand or modifier bits' % name)
    if name == 'v_cvt_off_f32_i4_e64':
        # Integer-nibble conversion: the instruction has one source, so only src0's VOP3A ABS (w0[8]) and NEG
        # (w1[29]) can carry a modifier. Masking all three bit positions of each field (w0[10:8], w1[31:29])
        # also refuses the unused src1/src2 bits, which keeps an inline constant from masquerading as a
        # negative integer and keeps a hidden modifier out of the printed operand.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & 0x700 or word1 & 0xe0000000:
            raise Unsupported('%s with neg/abs: not a 4-bit integer source modifier' % name)
    if name == 'v_cndmask_b32_e64':
        # The printed -1 is ambiguous: it can be plain inline -1 or inline 1 with NEG.
        word0, word1 = struct.unpack_from('<2I', code, off)
        for i in range(2):
            source = (word1 >> (9 * i)) & 0x1ff
            if not (128 <= source <= 208 or 240 <= source <= 248 or source == 255):
                continue
            if word0 & (1 << (8 + i)):
                raise Unsupported('v_cndmask_b32_e64 ABS on an immediate')
            if not word1 & (1 << (29 + i)):
                continue
            if source <= 192:
                bits = source - 128
            elif source <= 208:
                bits = (192 - source) & 0xffffffff
            elif source <= 248:
                bits = CNDMASK_INLINE_FLOAT_BITS[source - 240]
            else:
                if n != 12 or off + n > len(code):
                    raise Unsupported('v_cndmask_b32_e64 missing literal dword')
                bits, = struct.unpack_from('<I', code, off + 8)
            # Preserve the other operand spellings and any modifiers on the final operand.
            fields = asm.split(',', 3)
            if len(fields) != 4:
                raise Unsupported('malformed v_cndmask_b32_e64 operands')
            operand = fields[i + 1]
            fields[i + 1] = operand[:len(operand) - len(operand.lstrip())] + '0x%08x' % (bits ^ 0x80000000)
            asm = ','.join(fields)
    if n >= 8 and struct.unpack_from('<I', code, off)[0] >> 26 == 0b110101:
        # Read the word from the encoding: NEG on inline 1 is printed as `-1`, indistinguishable
        # from the plain inline -1 selector.
        # LLVM AMDGPUAsmParser::applyInputFPModifiers folds NEG into an immediate by XOR with the
        # FP sign mask. RDNA 2 lists VOP3 NEG as a float input modifier. Hardware behavior for
        # these immediate sources has not been measured.
        word0, word1 = struct.unpack_from('<2I', code, off)
        base = name.removesuffix('_e64')
        float_sources = 0
        if base in VEC_F:
            float_sources = VEC_F[base][0]
        elif re.fullmatch(r'v_cmpx?_\w+_f32', base):
            float_sources = 1 if base in ('v_cmp_class_f32', 'v_cmpx_class_f32') else 2
        rewritten = {}
        for i in range(3):
            if name == 'v_cndmask_b32_e64' and i < 2 or not word1 & (1 << (29 + i)):
                continue
            source = (word1 >> (9 * i)) & 0x1ff
            if not (128 <= source <= 192 or source == 255):
                continue
            # The selected float forms are VOP3A, where w0[10:8] holds ABS. Keep the
            # refusal for NEG combined with ABS on this source.
            if i >= float_sources or word0 & (1 << (8 + i)):
                raise Unsupported('%s with NEG on an integer or literal source %d: the printed sign would read as inline -N'
                                  % (name, i))
            if source == 255:
                if n != 12 or off + n > len(code):
                    raise Unsupported('%s missing literal dword' % name)
                bits, = struct.unpack_from('<I', code, off + 8)
            else:
                bits = source - 128
            # f() bitcasts this hex word through F(), the asfloat macro.
            rewritten[i + 1] = '0x%08x' % (bits ^ 0x80000000)
        if rewritten:
            mnemonic, operands, mods = parse(asm)
            if len(operands) <= max(rewritten):
                raise Unsupported('malformed %s operands' % name)
            for index, token in rewritten.items():
                operands[index] = token
            asm = ' '.join([mnemonic, ', '.join(operands), *mods])
    return asm


def native_instructions(code):
    """[(byte offset, assembly text)] of every instruction of `code`, for pixel_aot's control-flow check.
    Unlike decode(), this always covers the entire span."""
    out, off = [], 0
    while off < len(code):
        try:
            n, _, asm, _ = D.decode_instruction(code, off)
        except Exception as error:
            raise ValueError('undecodable word at +0x%x: %s' % (off, error)) from error
        out.append((off, asm))
        off += n
    return out


def decode(code, refusals=None, *, embedded_vertex=False, exports_only=False):
    out, off = [], 0
    saw_getpc = False
    while off + 4 <= len(code):
        try:
            n, name, asm, _ = D.decode_instruction(code, off)
        except Exception as error:
            raise Unsupported('undecodable word at +0x%x: %s' % (off, error))
        if exports_only:
            if name == 'word_0x00000000' and out and out[-1][2] == 's_endpgm' and not any(code[off:]):
                break  # word-aligned zero fill after the program
            if off + n > len(code) or asm.startswith('.long') or '<truncated>' in asm:
                raise Unsupported('unreadable instruction in placeholder pixel program')
            if name != 'exp':
                out.append((off, n, name, asm))
                off += n
                continue
        try:
            asm = check_encoding(code, off, n, name, asm)
        except Unsupported as reason:
            if refusals is None:
                raise
            refusals.append((str(reason), 'decode'))
        else:
            out.append((off, n, name, asm))
            if embedded_vertex:
                saw_getpc |= name == 's_getpc_b64'
                # A proven table is data, even if its words resemble instructions.
                if name == 's_endpgm' and saw_getpc and embedded_buffer_proof(code, out):
                    return out
        off += n
    last = max((i for i, x in enumerate(out) if x[2] == 's_endpgm'), default=len(out) - 1)
    # A bounded stream can end in a back edge after the last exit. Keep the tail
    # when a preceding direct branch enters it; raw captures may have dead words there.
    if last + 1 < len(out) and out[-1][2] == 's_branch' and out[-1][0] + out[-1][1] == len(code):
        tail_start = out[last + 1][0]
        if any(name.startswith(('s_branch', 's_cbranch_'))
               and tail_start <= off + 4 + 4 * struct.unpack_from('<h', code, off)[0] <= out[-1][0]
               for off, _, name, _ in out[:last]):
            return out
    if exports_only and any(name == 'exp' for _, _, name, _ in out[last + 1:]):
        raise Unsupported('placeholder export after final program end')
    return out[:last + 1]


def export_operands(asm):
    # disasm_rdna2 prints a compressed EN that is not a whole channel pair (0x0, 0x3, 0xC or 0xF) as
    # an explicit en:0x.. flag. As in pixel_aot, such a mask is refused rather than guessed.
    m = re.fullmatch(r'exp (\w+) (.*?)((?: done| compr| vm| en:0x[0-9a-f]+)*)', asm)
    if not m:
        raise Unsupported('export form %r' % asm)
    target, vals, flags = m.group(1), [v.strip() for v in m.group(2).split(',')], m.group(3).split()
    return target, vals, flags


def split_top(text, sep=','):
    parts, depth, cur = [], 0, ''
    for ch in text:
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


@functools.lru_cache(maxsize=None)
def parse_cached(asm):
    name, _, rest = asm.partition(' ')
    ops = split_top(rest)
    mods = []
    if ops:
        tail = split_top(ops[-1], ' ')
        ops[-1], mods = tail[0], tail[1:]
    return name, tuple(ops), tuple(mods)


def parse(asm):
    """-> (mnemonic, [operands], [modifier tokens]); the analysis passes parse each instruction many times."""
    name, ops, mods = parse_cached(asm)
    return name, list(ops), list(mods)


FLOATS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0,
          '0.15915494': 0.15915494}


def fbits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def ptr_high_symbol(source, offset):
    """High word of a 64-bit pointer advanced by a literal offset: the original word, or the one a completed
    s_addc_u32 produced for exactly that accumulated offset (an advanced word never equals another offset's)."""
    return ('ptrhi', source) if not offset else ('ptrhi', source, offset)


# Sources whose pointer is a 64-bit register pair (the low word 'ptr', the high word 'ptrhi').
PTR64_SOURCES = ('root', 'vbtab')


@functools.lru_cache(maxsize=None)
def reg_range(tok):
    """'s[8:11]' -> ('s', 8, 11); 's8' -> ('s', 8, 8); 'v[1:3]' -> ('v', 1, 3); 'exec'/'vcc' pairs too."""
    m = re.fullmatch(r'([sv])\[(\d+):(\d+)\]', tok)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = re.fullmatch(r'([sv])(\d+)', tok)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(2))
    return None


# Scalar state for the compute execz-block check (Stage.execz_refusals): registers as 's<n>', 'vcc', 'exec', 'm0'
# and 'scc'. Reads may be over-approximated and must-writes under-approximated; both only add refusals.
# The low half of a mask (vcc_lo, exec_lo) is 'vcc' / 'exec', the high half 'vcc_hi' / 'exec_hi', and a 64-bit
# operand both, so a write of one half never counts as writing the other.
SCALAR_TOKEN = re.compile(r'\bs\[(\d+):(\d+)\]|\bs(\d+)\b|\b(vcc|exec)(_lo|_hi)?\b|\b(m0)\b|\bsrc_(scc)\b')
EXEC_KEYS = {'exec', 'exec_hi'}
NO_SCALAR_EFFECT = ('s_waitcnt', 's_nop', 's_inst_prefetch', 's_clause', 's_cbranch_', 's_branch')
# Forms that print no destination: every printed scalar operand is read. v_cmpx writes EXEC implicitly; its e32 and
# SDWA forms print only sources, and the repository decoder prints the e64 form's unused SDST field first.
# s_setpc reads its address; s_movrel* address a register through m0, so their printed operand is not a plain write.
NO_DESTINATION = ('v_cmpx', 's_setpc', 's_movrel', 'v_movrel')
IMPLICIT_VCC_READ = ('v_div_fmas',)
SCC_READERS = ('s_cselect', 's_cmov', 's_addc', 's_subb', 's_cbranch_scc')
READ_MODIFY_WRITE = ('s_addk_i32', 's_mulk_i32', 's_cmov', 's_bitset', 'v_mac', 'v_fmac')
TWO_DESTINATIONS = ('v_add_co', 'v_sub_co', 'v_subrev_co', 'v_div_scale', 'v_mad_u64_u32', 'v_mad_i64_i32')
# Numeric forms covered by the compute scalar join readback fixtures.
EXECZ_SCALAR_JOIN = frozenset({
    's_and_b32', 's_bfm_b32', 's_lshl_b32',
    's_buffer_load_dwordx2', 's_buffer_load_dwordx4', 's_load_dwordx2', 's_load_dwordx4',
})
# Effects outside scalar registers, or control flow the wave takes as a whole: refused inside such a block.
EXECZ_REFUSED = ('s_endpgm', 's_store', 's_atomic', 's_buffer_store', 's_buffer_atomic', 's_barrier', 's_sendmsg',
                 's_setreg', 's_setpc', 's_swappc', 's_dcache', 's_icache', 's_gl1', 's_memtime', 's_memrealtime',
                 's_trap', 's_sleep', 'v_writelane', 's_movrel', 'v_movrel')


def scalar_tokens(text):
    out = set()
    for lo, hi, single, mask, half, m0, scc in SCALAR_TOKEN.findall(text):
        if lo:
            out.update('s%d' % n for n in range(int(lo), int(hi) + 1))
        elif mask:
            out.update({'_lo': {mask}, '_hi': {mask + '_hi'}}.get(half, {mask, mask + '_hi'}))
        else:
            out.add('s' + single if single else m0 or scc)
    return out


def is_exec(regs):
    return regs in ({'exec'}, EXEC_KEYS)


def mask_words(tok, count):
    """The scalar state keys holding a mask operand's words, low word first ('0' for a zero literal), or None when
    the operand is not `count` whole words of scalar state."""
    if tok == '0':
        return ['0'] * count
    m = re.fullmatch(r's\[(\d+):(\d+)\]|s(\d+)|(vcc|exec)(_lo|_hi)?', tok)
    if not m:
        return None
    if m.group(4):
        words = {'_lo': [m.group(4)], '_hi': [m.group(4) + '_hi']}.get(m.group(5), [m.group(4), m.group(4) + '_hi'])
    else:
        words = ['s%d' % n for n in range(int(m.group(1) or m.group(3)), int(m.group(2) or m.group(3)) + 1)]
    return words if len(words) == count else None


@functools.lru_cache(maxsize=None)
def clear_shape(name, asm):
    """What clear_after needs from one instruction's text, which the fixed point would otherwise parse again on every
    visit: 'cmpx', None for a form that carries no fact, or (operation, saving suffix, word count, operand words)."""
    base = re.sub(r'_e(32|64)$', '', name)
    if base.startswith('v_cmpx'):
        return 'cmpx'
    m = re.fullmatch(r's_(mov|and|or|xor|andn2)(_saveexec|_wrexec)?_b(32|64)', base)
    if not m or (m.group(2) and m.group(1) == 'mov'):
        return None
    count = 2 if m.group(3) == '64' else 1
    words = [mask_words(op, count) for op in parse(asm)[1]]
    if None in words or len(words) != (2 if m.group(2) or m.group(1) == 'mov' else 3):
        return None
    return m.group(1), m.group(2), count, tuple(tuple(word) for word in words)


def clear_after(name, asm, may, clear):
    """Forward step of the proof Stage.execz_refusals runs over a block that s_cbranch_execz skips. A fact (key, h)
    says: for every lane of wave half h (0: lanes 0 to 31, 1: lanes 32 to 63) whose exec bit was clear at the branch,
    that lane's bit of the scalar state word `key` is clear. Returns the facts after the instruction, given the
    facts before it.

    Only bitwise forms carry a fact, per word: a zero literal holds it for both halves; D = A & B holds what either
    source holds; A | B and A ^ B what both hold; A & ~B and a move what A holds. A saveexec or wrexec form is its
    operation on exec plus the copy it saves, and v_cmpx ANDs its result into exec. Any other write, a compare
    result included, drops the facts of what it may write."""
    out = {fact for fact in clear if fact[0] not in may - EXEC_KEYS}
    shape = clear_shape(name, asm)
    if shape == 'cmpx':
        return out
    if shape is None:
        return {fact for fact in out if fact[0] not in may}
    op, saving, count, words = shape
    halves = lambda word: {0, 1} if word == '0' else {h for key, h in clear if key == word}
    dest, srcs = words[0], list(words[1:])
    if saving:  # RDNA 2 ISA SOP1: the operation is S0 op EXEC, so andn2 is S0 & ~EXEC
        srcs.append(('exec', 'exec_hi')[:count])
    combine = {'and': set.union, 'or': set.intersection, 'xor': set.intersection}.get(op, lambda a, *b: a)
    value = [combine(*(halves(src[j]) for src in srcs)) for j in range(count)]
    writes = list(zip(dest, value))
    if saving:  # saveexec keeps the old exec in D; wrexec writes the result to both
        writes = list(zip(('exec', 'exec_hi'), value)) + (writes if saving == '_wrexec' else
                                                          [(word, halves(old)) for word, old in zip(dest, srcs[-1])])
    out = {fact for fact in out if fact[0] not in dict(writes)}
    return out | {(word, h) for word, held in writes for h in held}


def mask_lanes(name, asm):
    """(operation, destination words, source word lists) of a bitwise scalar mask instruction, or None. Word p of
    every operand holds the same lanes: wave half p of a 64-bit form."""
    m = re.fullmatch(r's_(mov|and|andn2|or|orn2|nand|nor|xor|xnor)_b(32|64)', re.sub(r'_e(32|64)$', '', name))
    if not m:
        return None
    words = [mask_words(op, 2 if m.group(2) == '64' else 1) for op in parse(asm)[1]]
    if None in words or len(words) != (2 if m.group(1) == 'mov' else 3):
        return None
    return m.group(1), words[0], words[1:]


def high_half_of_b32(lanes, word):
    """True for a 32-bit form that names the high half of a mask (vcc_hi): its bit n is lane 32 + n, which the word
    position of a one-word operand (always the low half) does not say, so no lane-wise rule applies to it."""
    return word.endswith('_hi') and len(lanes[1]) == 1


def exec_absorbs(name, asm, word):
    """True when an instruction reads `word` only through an operation whose result bit is fixed while the exec bit
    of the same lanes is clear: W & exec and exec & W give 0, ~(W & exec) and W | ~exec give 1."""
    lanes = mask_lanes(name, asm)
    if not lanes or lanes[0] not in ('and', 'nand', 'orn2') or high_half_of_b32(lanes, word):
        return False
    a, c = lanes[2]
    pairs = [(a, c)] if lanes[0] == 'orn2' else [(a, c), (c, a)]
    return any(word in w and word not in e and all(e[p] == ('exec', 'exec_hi')[p] for p in range(len(w)) if w[p] == word)
               for w, e in pairs)


def kept_halves(name, asm, word, clear):
    """Wave halves whose lanes keep their bit of `word` through `s_andn2 R, R, X` when X is clear for them: R & ~X
    keeps bit n of R when bit n of X is clear. Empty for any other write of `word`."""
    lanes = mask_lanes(name, asm)
    if not lanes or lanes[0] != 'andn2' or lanes[1] != lanes[2][0]:
        return set()
    return {h for p, w in enumerate(lanes[1]) if w == word for key, h in clear if key == lanes[2][1][p]}


def lane_read(name, asm, word, halves):
    """True when an instruction reads `word` as a lane mask, bit n of each result from bit n of the sources, with
    `word` in a position whose lanes are in `halves`, any exec operand there the exec word of those lanes, and the
    result there written to that exec word or back to `word`, not copied to another register."""
    lanes = mask_lanes(name, asm)
    if not lanes or high_half_of_b32(lanes, word):
        return False
    _, dest, srcs = lanes
    at = {p for src in srcs for p, w in enumerate(src) if w == word}
    return bool(at) and at <= halves and all(dest[p] in (word, ('exec', 'exec_hi')[p]) for p in at) and all(
        w == ('exec', 'exec_hi')[p] for src in srcs for p, w in enumerate(src) if p in at and w in EXEC_KEYS)


def effect_destinations(base, ops):
    """Split the decoded operands into destination and source fields."""
    count = 2 if base.startswith(TWO_DESTINATIONS) or base == 'v_swap_b32' else 1
    return ops[:count], ops[count:]


def scalar_effects(name, asm):
    """(reads, must-writes, may-writes) of scalar state for one decoded instruction."""
    base = re.sub(r'_e(32|64)$', '', name)
    _, ops, _ = parse(asm)
    if base.startswith(NO_SCALAR_EFFECT) or base == 's_endpgm':
        reads = {'scc'} if base.startswith('s_cbranch_scc') else {'vcc'} if base.startswith('s_cbranch_vcc') else \
            {'exec'} if base.startswith('s_cbranch_exec') else set()
        return reads, set(), set()
    if base.startswith('v_cmpx'):
        return scalar_tokens(' '.join(ops)) | {'exec'}, {'exec'}, set(EXEC_KEYS)
    if '_store' in base or base.startswith(('ds_write', 'exp') + NO_DESTINATION) or not ops:
        return scalar_tokens(' '.join(ops)) | ({'m0'} if 'movrel' in base else set()), set(), set()
    if base.startswith(('s_cmp', 's_bitcmp')):
        return scalar_tokens(' '.join(ops)), {'scc'}, {'scc'}
    dests, srcs = effect_destinations(base, ops)
    writes = scalar_tokens(' '.join(dests))
    reads = scalar_tokens(' '.join(srcs))
    if base.startswith(READ_MODIFY_WRITE) or base == 'v_swap_b32':
        reads |= writes
    if base.startswith(SCC_READERS) or (base in SCA and 'scc' in SCA[base][0]):
        reads.add('scc')
    if base.startswith(IMPLICIT_VCC_READ):
        reads.add('vcc')
    may = set()
    if 'saveexec' in base or 'wrexec' in base:
        reads.add('exec')
        writes.add('exec')
        may |= EXEC_KEYS
    if (base in SCA and SCA[base][1]) or 'saveexec' in base or 'wrexec' in base:
        writes.add('scc')
    # A wave64 compare or carry-out prints its mask destination in the 32-bit spelling vcc_lo but writes vcc_hi too.
    if 'vcc' in writes and base.startswith(('v_cmp', 'v_add_co', 'v_sub_co', 'v_subrev_co', 'v_div_scale')):
        may.add('vcc_hi')
    return reads, writes, writes | may | ({'scc'} if base.startswith('s_') else set())

# Forms with no scalar destination: their printed operands are all sources (v_cmpx writes only EXEC, which the
# symbolic tracking does not hold).
NO_SCALAR_DESTINATION = ('s_cbranch', 's_branch', 's_waitcnt', 's_nop', 's_endpgm', 's_cmp', 's_bitcmp', 's_setpc',
                         's_barrier', 's_sendmsg', 's_inst_prefetch', 's_clause', 's_setprio', 's_sleep', 's_setreg',
                         's_code_end', 's_dcache', 's_icache', 's_ttrace', 'exp', 'ds_write', 'v_cmpx')
TWO_SCALAR_DESTINATIONS = ('v_add_co', 'v_sub_co', 'v_subrev_co', 'v_div_scale', 'v_mad_u64_u32', 'v_mad_i64_i32')
# Symbols whose register also holds the value at run time: constants are assigned when set, entry root constants are
# assigned at entry when a join needs it, and the others are assigned when tracked.
# Longest uniform HLSL expression a VGPR keeps for a first-lane read: a chain past it is no longer recomputable.
UNIFORM_EXPR_LIMIT = 4096

RUN_TIME_SYMBOLS = ('const', 'ud', 'execcopy', 'fixed_buffer_word')


def run_time_symbol(sym):
    """True when the register also holds the symbol's value at run time (a constant marked unassigned does not)."""
    return sym[0] in RUN_TIME_SYMBOLS and not (sym[0] == 'const' and len(sym) > 2)


def names_one_value(sym):
    """True when two equal symbols stand for the same value, so a join where every path brings it can keep it.
    A modified descriptor word, descnum, a fixed buffer word, a sampler-heap word and an unplaced entry register
    compare equal for different values (what was written, or where it came from, is not in the tuple). An exec copy
    is taken only while exec holds the entry mask, so equal copies hold the same mask."""
    if sym is None:
        return False
    if sym[0] == 'desc':
        # A compute descriptor can also carry an integer tag identifying the load or constant high word.
        return len(sym) == 4 or (len(sym) == 5 and isinstance(sym[4], int))
    # A table and a known byte offset identify one descriptor address. The legacy 'added'
    # marker carries no offset, so two occurrences of it can denote different addresses.
    if sym[0] == 'ptr' and len(sym) == 3 and sym[1][0] == 'table':
        return isinstance(sym[2], int)
    return sym[0] in ('const', 'ud', 'execcopy') or (sym[0] in ('ptr', 'ptrhi') and len(sym) == 2)


def scalar_destinations(name, asm):
    """The SGPRs (and 'm0') an instruction may write; 'any' for s_movrel*, whose register comes from m0."""
    if name.startswith(NO_SCALAR_DESTINATION) or '_store' in name:
        return set()
    if 'movrel' in name:
        return {'any'}
    ops = parse(asm)[1]
    out = set()
    for tok in ops[:2] if name.startswith(TWO_SCALAR_DESTINATIONS) else ops[:1]:
        if tok == 'm0':
            out.add('m0')
        r = reg_range(tok)
        if r and r[0] == 's':
            out.update(range(r[1], r[2] + 1))
    return out


def desc_key(sym):
    """Resource key of the descriptor a ('desc', source, load offset, dword) word belongs to: the key descriptor()
    gives its first word, so a write into a later descriptor of a wide load marks that descriptor."""
    return sym[1], None if sym[2] is None else sym[2] + 4 * (sym[3] & ~3)


def desc_word(sym, dword):
    return sym[:3] + (dword,) + sym[4:]


def raw_descriptor_alu(base, syms, constants):
    """Recognize only the word expressions needed for a structured-to-raw buffer store."""
    # gfx10 V# layout in docs/pixel_mubuf_contract.md: stride is DW1[29:16],
    # NUM_RECORDS is DW2 and OOB_SELECT is DW3[29:28].
    # The extracted stride may be clamped to at least 1 (s_max_u32 with 1), so that a descriptor that is
    # already raw (stride 0) keeps NUM_RECORDS bytes.
    if base == 's_max_u32':
        for i, word in enumerate(syms):
            if word and word[:2] == ('descnum', 'stride') and constants[1 - i] == 1:
                return ('descnum', 'stride_min1', word[2])
        return None
    for i, word in enumerate(syms):
        if not word or word[0] != 'desc' or len(word) < 4:
            continue
        other = 1 - i
        field = word[3] % 4
        loaded = 5 if len(word) > 4 and isinstance(word[4], int) else 4  # with the compute load tag
        plain = len(word) == loaded  # a loaded word of one s_load
        if plain and field == 1:
            if base == 's_and_b32' and constants[other] == 0xc000ffff:
                return word + ('stride0',)
            if base == 's_bfe_u32' and i == 0 and constants[1] == 0xe0010:
                return ('descnum', 'stride', word)
        if plain and field == 2 and base == 's_mul_i32':
            if syms[other] in (('descnum', kind, desc_word(word, word[3] - 1)) for kind in ('stride', 'stride_min1')):
                return word + ('byte_count',)
        if field == 3:
            if plain and base == 's_and_b32' and constants[other] == 0xcfffffff:
                return word + ('oob0',)
            if (plain or word[loaded:] == ('oob0',)) and base == 's_or_b32' and constants[other] == 0x30000000:
                return word[:loaded] + ('oob3',)
    return None


def smem_offset(tok):
    """The SMEM offset operand as a token: 'off' (no offset operand) is the literal 0, anything else unchanged."""
    return '0' if tok == 'off' else tok


MODIFIER_FREE_HANDLERS = frozenset((
    'op_v_mov_b32', 'op_v_cvt_pkrtz_f16_f32', 'op_v_interp_p1_f32',
    'op_v_interp_p2_f32', 'op_v_interp_mov_f32',
))


def smem_mods(mods):
    """Non-negative 21-bit signed immediate byte offset from SMEM modifiers, refusing unknown or negative forms."""
    imm = None
    for m in mods:
        if m in ('glc', 'dlc'):
            continue
        if m.startswith('offset:') and imm is None and re.fullmatch(r'-?(?:0[xX][0-9a-fA-F]+|\d+)', m[7:]):
            # Reject signed negative offsets and older unsigned spellings with bit 20 set.
            token = m[7:]
            value = int(token, 16 if '0x' in token.lower() else 10)
            if not 0 <= value <= 0xfffff:
                raise Unsupported('SMEM immediate offset %s' % m)
            imm = value
        else:
            raise Unsupported('SMEM modifier %s' % m)
    return imm


# Each conversion uses only its input bits (including byte extraction for ubyte0).
# Multiply, fused multiply-add (including the literal form), reciprocal, log2, ceil and median of three use
# only local arithmetic. Move copies its input bits. None reads a lane ID or another lane.
UNIFORM_VALU = frozenset((
    'v_cvt_u32_f32', 'v_cvt_i32_f32', 'v_cvt_f32_u32', 'v_cvt_f32_ubyte0', 'v_cvt_off_f32_i4',
    'v_cvt_pk_u16_u32', 'v_cvt_pk_i16_i32',
    'v_mul_f32', 'v_fma_f32', 'v_fmaak_f32', 'v_rcp_f32', 'v_log_f32', 'v_ceil_f32',
    'v_med3_f32', 'v_mov_b32',
    # Pure 32-bit integer ALU: each lane computes the result from its own sources only, so equal sources give
    # equal results in every lane (no carry-out, no lane ID, no cross-lane read).
    'v_add_nc_u32', 'v_sub_nc_u32', 'v_subrev_nc_u32', 'v_and_b32', 'v_or_b32', 'v_xor_b32', 'v_not_b32',
    'v_lshlrev_b32', 'v_lshrrev_b32', 'v_ashrrev_i32', 'v_mul_lo_u32', 'v_mul_u32_u24',
    'v_min_u32', 'v_max_u32', 'v_min_i32', 'v_max_i32',
))

UNIFORM_LOAD_WORDS = {'buffer_load_dword': 1, 'buffer_load_dwordx2': 2, 'buffer_load_dwordx3': 3, 'buffer_load_dwordx4': 4,
                      'buffer_load_format_x': 1, 'buffer_load_format_xy': 2, 'buffer_load_format_xyz': 3,
                      'buffer_load_format_xyzw': 4}
UNIFORM_LOADS = frozenset(UNIFORM_LOAD_WORDS)
UNIFORM_IMAGE_LOADS = frozenset(('image_load',))


# DPP quad_perm selections that exchange lanes of a 2x2 pixel quad (lane = 2 * y + x), with the SM 6.0 intrinsic.
# Image ops that take implicit derivatives of the pixel quad (see Stage.quad_vote).
DERIVATIVE_OPS = ('image_sample_b', 'image_get_lod')
# "Some lane of the 2x2 quad has bit 0 set": fine derivatives of a 0/1 value read the row neighbour (ddx_fine), then
# the other row (ddy_fine). Exact for 0/1 and defined only where the quad runs the statement together (SM 5.0 and 6.0).
# The derivatives are arithmetic operands, not operands of a short-circuit ||: a DXC build for some GPUs moved a derivative
# under the condition of an ||, where only some lanes ran it (measured on a hardware adapter).
QUAD_ANY_HLSL = [
    'bool x2w_quad_any(uint bit) {',
    '    float lane = (bit & 1u) != 0u ? 1.0 : 0.0;',
    '    float row = (lane + abs(ddx_fine(lane)) != 0.0) ? 1.0 : 0.0;',
    '    return row + abs(ddy_fine(row)) != 0.0;',
    '}',
]
ROW_ROR8 = 'RowRotate8'  # not an intrinsic: the compute exchange partner is lane ^ 8
QUAD_SWAPS = {(1, 0, 3, 2): 'QuadReadAcrossX', (2, 3, 0, 1): 'QuadReadAcrossY', (3, 2, 1, 0): 'QuadReadAcrossDiagonal'}


# Opcodes whose first operand is a source, not a destination (interp_f16_scan must not read it as a write).
INTERP_STORE_PREFIXES = ('buffer_store', 'tbuffer_store', 'global_store', 'flat_store', 'scratch_store', 'ds_write',
                         'ds_store', 'image_store')


# VALU forms whose second operand is an SGPR destination (VOP3b carry-out or scale flag), not a source.
VOP3B_SDST = re.compile(r'v_(add|sub|subrev)_co(_ci)?_u32|v_div_scale_f(32|64)|v_mad_[iu]64_[iu]32')


def destinations(name, ops):
    """The operands an instruction writes: the first, and the SGPR destination of the VOP3b forms. Compares write none."""
    if not ops or name.startswith(('s_cmp', 's_bitcmp')):
        return []
    return [ops[0]] + ([ops[1]] if len(ops) > 1 and VOP3B_SDST.match(name) else [])


def writes_exec(name, ops):
    """Conservatively: whether the instruction can write EXEC, explicitly or implicitly."""
    for tok in destinations(name, ops):
        r = reg_range(tok)
        if tok in ('exec', 'exec_lo', 'exec_hi') or r and r[0] == 's' and r[1] <= 127 and r[2] >= 126:
            return True
    return 'saveexec' in name or 'wrexec' in name or name.startswith(('v_cmpx', 's_movreld'))


# In wave32 VCC is vcc_lo alone and vcc_hi is a free scalar register that compilers allocate like any other (RDNA 2
# ISA, "VCC": in wave32 only VCC_LO is the condition mask). The translator models it as an ordinary SGPR the program
# never names, taken from the top of the 106 (beyond the user-data and system registers set at entry).
def vcc_hi_whole_word_writer(name, ops):
    """True when the instruction's first operand is vcc_hi and its name is a scalar instruction or an actual lane-read
    writer (v_readlane, v_readfirstlane): the register then holds one whole word, not one bit per lane."""
    return ops[:1] == ['vcc_hi'] and (name.startswith('s_') or name.startswith(('v_readlane', 'v_readfirstlane')))


def vcc_hi_scalar_mask_read(ins):
    """True when a VOP3 lane-mask source reads vcc_hi after a (whole-word) scalar or lane-read writer wrote it (or the
    program branches, so the order is unknown): the renamed register then holds a whole word, not the current lane's
    bit, and the lowering would test bit 0 of it for every lane. A compare or carry-out that wrote vcc_hi last keeps
    the per-lane form."""
    scalar, branchy, read_after = False, False, False
    reads = False
    for _o, _n, name, asm in ins:
        ops = parse(asm)[1]
        branchy |= name.startswith(('s_cbranch', 's_branch'))
        if (name.startswith('v_cmp_') and ops[:1] == ['vcc_hi']) or (VOP3B_SDST.match(name) and ops[1:2] == ['vcc_hi']):
            scalar = False
        elif vcc_hi_whole_word_writer(name, ops):
            scalar = True
        if name in ('v_cndmask_b32_e64', 'v_add_co_ci_u32_e64', 'v_sub_co_ci_u32_e64', 'v_subrev_co_ci_u32_e64') \
                and ops[-1:] == ['vcc_hi']:
            reads = True
            read_after |= scalar
    return read_after or (reads and branchy and any(
        vcc_hi_whole_word_writer(name, parse(asm)[1]) for _o, _n, name, asm in ins))


def wave32_vcc_hi(ins, wave32=False):
    """`ins` with every vcc_hi operand renamed to an unused SGPR when vcc_hi is provably an ordinary scalar register,
    else `ins`.

    The decoder prints wave32 syntax, so the program text alone does not give the wave size. vcc_hi holds its own
    word in both wave sizes as long as only scalar instructions write it; in wave64 a VALU write of VCC (a compare,
    a carry-out, v_div_scale, printed as vcc_lo) also writes vcc_hi with the high lanes, and a VALU mask read of VCC
    (v_cndmask_b32_e32, a carry-in, v_div_fmas) or s_cbranch_vccz/vccnz reads vcc_hi for them. So the rename is
    exact when
    - no instruction names the 64-bit pair `vcc` (whose high word would then be vcc_hi), and
    - a VALU compare or carry writes vcc_hi or an odd SGPR, or a VOP3 mask source reads one (a misaligned pair in
      wave64, so the program is wave32), or no VALU instruction and no vcc branch uses vcc_lo as a lane mask.
    Otherwise, or when M0-relative s_movrel* could reach the chosen register, vcc_hi is left alone and refused.
    wave32=True is the caller's proof that the program is wave32 (RDNA 2 ISA, wave32: VCC is vcc_lo alone, so no VALU
    instruction ever writes or reads vcc_hi as part of a mask): the two wave64 conditions above are then not needed."""
    if not any(re.search(r'\bvcc_hi\b', asm) for *_, asm in ins):
        return ins
    if any(re.search(r'\bvcc\b', asm) or name.startswith('s_movrel') for _o, _n, name, asm in ins):
        return ins
    masks = []  # lane-mask destinations of VALU compares and carry-outs (VOP3b SDST)
    for _o, _n, name, asm in ins:
        ops = parse(asm)[1]
        if name.startswith('v_cmp_') and ops:
            masks.append(ops[0])
        elif VOP3B_SDST.match(name) and len(ops) > 1:
            masks.append(ops[1])
        if name in ('v_cndmask_b32_e64', 'v_add_co_ci_u32_e64', 'v_sub_co_ci_u32_e64', 'v_subrev_co_ci_u32_e64') \
                and ops[-1] != 'vcc_lo':
            masks.append(ops[-1])  # a lane-mask source is a pair in wave64 too: vcc_hi or an odd SGPR proves wave32
    odd = any((reg_range(tok) or ('', 0, 1))[0] == 's' and reg_range(tok)[1] == reg_range(tok)[2]
              and reg_range(tok)[1] % 2 for tok in masks)
    # Without that proof the program may be wave64, where every VALU use of VCC as a lane mask (a write, an
    # implicit or explicit mask source such as v_cndmask_b32_e32, a carry-in, v_div_fmas) and s_cbranch_vccz/vccnz
    # see vcc_hi for lanes 32..63: none may be present.
    vcc_mask_use = any((name.startswith('v_') and re.search(r'\bvcc_lo\b', asm)) or name.startswith('v_div_fmas')
                       or name in ('s_cbranch_vccz', 's_cbranch_vccnz') for _o, _n, name, asm in ins)
    if not wave32 and 'vcc_hi' not in masks and not odd and vcc_mask_use:
        return ins
    if vcc_hi_scalar_mask_read(ins):
        return ins
    named = set()
    for *_, asm in ins:
        for lo, hi, single in re.findall(r'\bs\[(\d+):(\d+)\]|\bs(\d+)\b', asm):
            if lo and int(lo) <= 107 <= int(hi):
                return ins   # a wide operand such as s[104:107] covers SGPR 107, which is vcc_hi: no free alias
            named.update(range(int(lo), int(hi) + 1) if lo else (int(single),))
    free = next((n for n in range(105, 63, -1) if n not in named), None)
    if free is None:
        return ins
    return [(off, n, name, re.sub(r'\bvcc_hi\b', 's%d' % free, asm)) for off, n, name, asm in ins]


def pc_pair(token):
    """Only aligned pairs of ordinary SGPRs can carry a symbolic code address."""
    r = reg_range(token)
    if r and r[0] == 's' and r[2] == r[1] + 1 <= 105 and r[1] % 2 == 0:
        return r[1]
    return None


def pc_word(token):
    """A displacement word, not a register or an inline floating-point constant."""
    if re.fullmatch(r'-?(?:0x[0-9a-f]+|\d+)', token):
        value = int(token, 16 if '0x' in token else 10)
        if -0x80000000 <= value <= 0xffffffff:
            return ((value & 0xffffffff) ^ 0x80000000) - 0x80000000
    return None


def scalar_registers(text):
    """All explicit SGPR words in an operand, including modifiers and ranges."""
    out = set()
    for low, high, single in re.findall(r'\bs(?:\[(\d+):(\d+)\]|(\d+)\b)', text):
        if single:
            out.add(int(single))
        else:
            out.update(range(int(low), int(high) + 1))
    return out


def reads_scc(name, asm):
    # LLVM AMDGPU SOPInstructions.td lists implicit SCC uses. Cover each supported family here
    # (s_cmov includes s_cmovk_i32) so both the calculated-PC and descriptor checks see new readers.
    return 'src_scc' in asm or name.startswith(('s_cbranch_scc', 's_cmov', 's_cselect', 's_addc', 's_subb'))


def embedded_buffer_proof(code, ins):
    """Prove a vertex buffer descriptor whose only observable use is reading the shader image."""
    if not ins or ins[-1][2] != 's_endpgm' or classify_decoded(ins) != 'vs':
        return None
    pcs = {off for off, _, _, _ in ins}
    reads = [item for item in ins if item[2] == 's_getpc_b64']
    if len(reads) != 1:
        return None
    read, length, _, asm = reads[0]
    _, ops, mods = parse(asm)
    low = pc_pair(ops[0]) if len(ops) == 1 and not mods and length == 4 else None
    if low is None or low % 4 or low + 3 > 105:
        return None

    regs = set(range(low, low + 4))
    desc = ('s', low, low + 3)
    setup = {read: None}
    loads = set()
    offset = size = flags = None
    carry_live = False
    end = 0
    for pc, n, name, asm in ins:
        if pc != end:  # No ignored encoding failures or gaps in the executable prefix.
            return None
        end = pc + n
        if name.startswith(('s_setpc', 's_call', 's_rfe', 's_trap', 's_endpgm_')) or 'movrel' in name:
            return None
        _, ops, mods = parse(asm)
        if name.startswith('s_swappc') and (pc != 0 or ops != ['s[4:5]', 's[4:5]'] or mods):
            return None
        if name.startswith(('s_branch', 's_cbranch')):
            # Setup dominates all loads. Forward branches after setup cannot change
            # any of its words, since every other access to those words is checked below.
            label = re.fullmatch(r'label_0x([0-9a-f]+)', ops[0]) if len(ops) == 1 else None
            target = int(label[1], 16) if label else -1
            if None in (offset, size, flags) or target <= pc or target not in pcs:
                return None
            carry_live = False
        if pc == read:
            continue
        refs = scalar_registers(asm) & regs
        imm = pc_word(ops[-1]) if ops else None
        if (pc > read and refs and not mods and name in ('s_mov_b32', 's_movk_i32')
                and len(ops) == 2 and imm is not None):
            word = sopk_imm(ops[1], True) if name == 's_movk_i32' else imm & 0xffffffff
            if ops[0] == 's%d' % (low + 2) and size is None:
                size = word
            elif ops[0] == 's%d' % (low + 3) and flags is None:
                flags = word
            else:
                return None
        elif (pc > read and name == 's_add_u32' and not mods and offset is None and len(ops) == 3
              and ops[:2] == ['s%d' % low] * 2 and imm is not None and imm >= 0):
            offset = read + length + imm
            carry_live = True
        elif (carry_live and name == 's_addc_u32'
              and not mods and ops[:2] == ['s%d' % (low + 1)] * 2 and len(ops) == 3 and imm == 0):
            carry_live = False
        else:
            # An address-derived SCC cannot escape. Only SCC-preserving
            # instructions may intervene between the low add and its carry.
            if offset is not None and reads_scc(name, asm):
                return None
            if carry_live and not name.startswith(('v_', 's_mov_', 's_movk_', 's_waitcnt', 's_nop')):
                carry_live = False
            if refs:
                if (name != 'buffer_load_dword' or len(ops) != 4 or reg_range(ops[2]) != desc
                        or scalar_registers(' '.join(ops[:2] + ops[3:] + mods)) & regs
                        or 'idxen' in mods or None in (offset, size, flags)):
                    return None
                loads.add(pc)
            continue
        setup[pc] = None

    # RDNA 2 raw bounds (OOB_SELECT=3), resource level 1, a 32-bit format,
    # and no indexing, swizzling, thread ID, or non-buffer type flags.
    if (not loads or flags >> 12 not in (0x31014, 0x31015, 0x31016)
            or offset % 4 or offset < end or size == 0 or offset + size > len(code)):
        return None
    data = code[offset:offset + size]
    words = tuple(word for word, in struct.iter_unpack('<I', data + bytes(-size % 4)))
    return setup, loads, size, words


def calculated_pc_proof(ins, kind):
    """Return offset -> target for proven SETPCs, and offset -> None for their address arithmetic.

    RDNA 2 GETPC returns the address of its successor. A low-word ADD_U32 and a
    high-word ADDC_U32 with its matching sign extension add one signed byte
    displacement, including any carry across the (unknown) loaded PC's low word.
    A completed pair is therefore the shader's load address plus a known byte
    offset, not an integer value that can be emitted to HLSL.

    At a CFG join retain only pair facts that agree on every incoming edge;
    retain *all* potentially address-dependent words and SCCs. A recognized
    address instruction can be elided only if it is recognized on every path.
    No other operation can observe an elided word or SCC. This also checks loops:
    a newly discovered SETPC successor participates in the same worklist.
    Ordinary reads of those SGPRs also need a single reaching non-PC definition
    matching the definition seen in the linear lowering order: a path around an
    elided write may still hold a symbolic root binding, not its zero-initialized
    HLSL local.
    """
    if not any(name == 's_setpc_b64' for _, _, name, _ in ins):
        return {}
    by_off = {off: (n, name, asm) for off, n, name, asm in ins}
    end = ins[-1][0] + ins[-1][1]
    tracked = {reg for _, _, name, asm in ins if name == 's_getpc_b64' for reg in scalar_registers(asm)}
    entry_defs = dict.fromkeys(tracked, -1)

    def refuse(off, reason):
        raise Unsupported('s_setpc_b64: %s at +0x%x' % (reason, off))

    def boundary(off, target):
        if target % 4:
            refuse(off, 'unaligned calculated branch target %#x' % target)
        if not 0 <= target < end:
            refuse(off, 'target %#x outside decoded program' % target)
        if target not in by_off:
            refuse(off, 'target %#x in the interior of an instruction or undecoded bytes' % target)

    if 0 not in by_off:
        refuse(0, 'missing entry instruction')
    if not any(name == 's_endpgm' for _, _, name, _ in ins):
        refuse(0, 'decoded program has no s_endpgm')

    def ordinary_accesses(name, ops, mods, refs):
        # These scalar operations and loads have explicit unconditional destinations.
        # Otherwise conservatively treat a mentioned PC word as a read, and any
        # possible SGPR destination as an invalidation of its reaching definition.
        written = (scalar_registers(ops[0]) if ops and
                   (name in SCA or name.startswith(('s_load_dword', 's_buffer_load_dword'))
                    or name in ('s_cselect_b32', 's_cselect_b64')) else set())
        reads = scalar_registers(', '.join(ops[1:]) + ' ' + ' '.join(mods)) if written else refs
        maybe_written = {reg for dst in destinations(name, ops) for reg in scalar_registers(dst)} - written
        return written, reads, maybe_written

    # Each state is (possibly elided SGPR words, definitely proven pairs, SCC,
    # definitely reaching ordinary definitions of SGPRs the lowering can erase).
    # pairs[low] = (target, None) for a complete pair, or (target, high immediate)
    # for a partial low-word sum. SCC is None for ordinary state, the pair's low
    # register for the corresponding carry, or -1 for an elided/unusable flag.
    incoming = {0: (set(), {}, None, entry_defs)}
    pending = [0]
    lowered = {}

    def propagate(off, words, pairs, scc, definitions):
        state = incoming.get(off)
        if state is None:
            merged = (set(words), dict(pairs), scc, dict(definitions))
        else:
            old_words, old_pairs, old_scc, old_defs = state
            merged = (old_words | words, {p: v for p, v in old_pairs.items() if pairs.get(p) == v},
                      old_scc if old_scc == scc else -1,
                      {reg: d for reg, d in old_defs.items() if definitions.get(reg) == d})
        if state != merged:
            incoming[off] = merged
            pending.append(off)

    while pending:
        off = pending.pop()
        n, name, asm = by_off[off]
        _, ops, mods = parse(asm)
        words, pairs, scc, definitions = incoming[off]
        words, pairs, definitions = set(words), dict(pairs), dict(definitions)
        refs = scalar_registers(asm)
        successors = [off + n]

        def clobber(written):
            nonlocal scc
            words.difference_update(written)
            for reg in written:
                definitions.pop(reg, None)
            for p in list(pairs):
                if p in written or p + 1 in written:
                    del pairs[p]
                    if scc == p:
                        scc = -1

        if name == 's_getpc_b64':
            p = pc_pair(ops[0]) if len(ops) == 1 else None
            if p is None or mods or n != 4:
                refuse(off, 'unsupported GETPC destination/form')
            clobber({p, p + 1})
            words.update((p, p + 1))
            pairs[p] = (off + n, None)
            lowered[off] = None
        elif name in ('s_add_u32', 's_addc_u32') and words & refs:
            row = reg_range(ops[0]) if ops else None
            p = row[1] if row and row[0] == 's' and row[1] == row[2] else None
            imm = pc_word(ops[2]) if len(ops) == 3 else None
            if p is None or imm is None or mods or n not in (4, 8) or ops[0] != ops[1]:
                refuse(off, 'unsupported arithmetic on calculated PC words')
            if name == 's_add_u32':
                fact = pairs.get(p)
                if fact is None or fact[1] is not None:
                    refuse(off, 'unknown or ambiguous input pair for calculated PC low-word addition')
                pairs[p] = (fact[0] + imm, -1 if imm < 0 else 0)
                scc = p
            else:
                fact = pairs.get(p - 1)
                if fact is None or fact[1] is None or scc != p - 1:
                    refuse(off, 'missing matching low-word carry for calculated PC high-word addition')
                if imm != fact[1]:
                    refuse(off, 'calculated PC high word is not the signed displacement extension')
                pairs[p - 1] = (fact[0], None)
                scc = -1
            lowered[off] = None
        elif name == 's_setpc_b64':
            p = pc_pair(ops[0]) if len(ops) == 1 else None
            if p is None or mods or n != 4:
                refuse(off, 'unsupported SETPC source/form')
            fact = pairs.get(p)
            if fact is None:
                refuse(off, 'unknown or ambiguous calculated target from %s' % ops[0])
            target, high = fact
            if high is not None:
                refuse(off, 'incomplete calculated target (missing high-word addition with carry)')
            boundary(off, target)
            lowered[off] = target
            successors = [target]
        else:
            written, reads, maybe_written = ordinary_accesses(name, ops, mods, refs)
            if words & reads or words and name.startswith(('s_movrels', 's_movreld')):
                refuse(off, 'calculated PC word escapes into %s' % name)
            clobber(written)
            definitions.update((reg, off) for reg in written & tracked)
            for reg in maybe_written:
                definitions.pop(reg, None)
            if scc is not None:
                if reads_scc(name, asm):
                    refuse(off, 'calculated PC SCC escapes into %s' % name)
                # These comparisons always emit a replacement SCC (or refuse).
                # Other scalar writers may not model SCC, so they cannot make
                # a tainted flag usable; they also invalidate any pending carry.
                if re.fullmatch(r's_cmpk?_(eq|lg|gt|ge|lt|le)_[iu]32|s_bitcmp[01]_b32', name):
                    scc = None
                elif not name.startswith(('v_', 's_branch', 's_cbranch_', 's_mov_', 's_movk_', 's_load_',
                                          's_buffer_load_', 's_nop', 's_waitcnt', 's_inst_prefetch', 's_clause',
                                          'buffer_', 'tbuffer_', 'global_', 'image_', 'exp', 'ds_')):
                    scc = -1

            if name in ('s_swappc_b64', 's_call_b64', 's_rfe_b64'):
                # The only pre-existing exception is the vertex entry fetch call.
                if not (name == 's_swappc_b64' and kind == 'vs' and off == 0
                        and ops == ['s[4:5]', 's[4:5]'] and not mods):
                    refuse(off, 'unsupported control transfer %s' % name)
            if name == 's_endpgm':
                successors = []
            elif name == 's_branch' or name.startswith('s_cbranch_'):
                try:
                    target = int(ops[0].split('_')[1], 16)
                except (IndexError, ValueError):
                    refuse(off, 'unsupported direct branch operand')
                boundary(off, target)
                successors = [target] if name == 's_branch' else [target, off + n]

        for successor in successors:
            if successor not in by_off:
                refuse(off, 'missing decoded successor %#x' % successor)
            propagate(successor, words, pairs, scc, definitions)

    for off, _, name, _ in ins:
        if name == 's_setpc_b64' and off not in lowered:
            refuse(off, 'unreachable SETPC has no proven target')

    # Stage's existing resource/constant symbol table is intentionally linear.
    # Do not substitute a symbolic address into it or pretend an erased symbol
    # was a runtime word. Every later ordinary read must see the same *ordinary*
    # definition in both the listing and on all reachable paths. Include writes
    # in unreachable code when tracking the listing because Stage lowers those too.
    linear_defs = dict(entry_defs)
    for off, _, name, asm in ins:
        _, ops, mods = parse(asm)
        if off in lowered:
            if lowered[off] is None:
                for reg in scalar_registers(ops[0]):
                    linear_defs.pop(reg, None)
            continue
        written, reads, maybe_written = ordinary_accesses(name, ops, mods, scalar_registers(asm))
        if off in incoming:
            reaching = incoming[off][3]
            for reg in sorted(reads & tracked):
                if reg not in linear_defs or reaching.get(reg) != linear_defs[reg]:
                    refuse(off, 'elided PC register s%d has no matching ordinary definition on every incoming path' % reg)
        for reg in maybe_written:
            linear_defs.pop(reg, None)
        linear_defs.update((reg, off) for reg in written & tracked)
    return lowered


class ImageAccessRefusal(Unsupported):
    """A proved program-image use that cannot be lowered."""


class ImageAddress:
    """Image-relative byte address, independent of the loaded PC's low word."""

    def __init__(self, base, offsets=()):
        self.base, self.offsets = base, offsets

    def advance(self, constant, offset=None):
        return ImageAddress(self.base + constant, self.offsets + ((offset,) if offset else ()))


class ImageLoadProof:
    """Follow forward compute edges and erase address words only at proved uses.

    A low addition yields a pending pair and an identity for its carry. The
    high addition must consume that exact carry and the original high word.
    Captures retain bounded numeric offsets at their uses, even if the source
    register is subsequently overwritten.
    """

    MASK = 0xffffffff
    UNKNOWN = (0, MASK, 1)
    MOVES = {'s_mov_b32', 's_mov_b64', 'v_mov_b32'}
    LOW_ADDS = {'s_add_u32', 'v_add_co_u32'}
    HIGH_ADDS = {'s_addc_u32', 'v_add_co_ci_u32'}

    def __init__(self, image, thread_dimensions):
        self.image = image
        self.words, self.carries, self.definitions = {}, {}, {}
        self.numbers = {'v%d' % i: (0, size - 1, 1) for i, size in enumerate(thread_dimensions)}
        self.instructions, self.lowered, self.loads, self.captures = [], {}, {}, {}
        self.erased, self.scalar_reads = {}, {}
        self.reaching_definitions = {}

    @staticmethod
    def registers(op):
        row = reg_range(op)
        if row:
            kind, first, last = row
            limit = 105 if kind == 's' else 255
            if first <= last <= limit:
                return tuple('%s%d' % (kind, i) for i in range(first, last + 1))
        return ()

    @staticmethod
    def mentions(ops):
        return {reg for op in ops for word in re.findall(r'[sv](?:\[\d+:\d+\]|\d+\b)', op)
                for reg in ImageLoadProof.registers(word)}

    def refuse(self, reason, access=False):
        error = ImageAccessRefusal if access else Unsupported
        raise error('s_setpc_b64: %s at +0x%x' % (reason, self.pc))

    @classmethod
    def constant(cls, op):
        value = pc_word(op)
        return value & cls.MASK if value is not None else None

    def number(self, op):
        value = self.constant(op)
        if value is None:
            return self.numbers.get(op, self.UNKNOWN)
        return value, value, (value & -value) or (1 << 32)

    def exact(self, op):
        low, high, _ = self.number(op)
        return low if low == high else None

    def mask_pair(self, op):
        if op in ('vcc', 'vcc_lo'):
            return {'vcc', 'vcc_hi'}
        regs = self.registers(op)
        if len(regs) == 1 and op.startswith('s'):
            n = int(op[1:])
            if n < 105 and n % 2 == 0:
                return {op, 's%d' % (n + 1)}
        return set()

    def numeric_result(self, name, ops, mods):
        if mods:
            return self.UNKNOWN
        if name in self.MOVES and len(ops) == 2:
            return self.number(ops[1])
        if len(ops) == 3 and name in ('v_and_b32', 's_and_b32'):
            a, b = map(self.number, ops[1:])
            if a[0] == a[1] and b[0] == b[1]:
                value = a[0] & b[0]
                return value, value, (value & -value) or (1 << 32)
            return 0, min(a[1], b[1]), max(a[2], b[2])
        if len(ops) == 3 and name in ('v_lshlrev_b32', 's_lshl_b32'):
            src, shift = ops[1:] if name == 's_lshl_b32' else ops[1:][::-1]
            count = self.exact(shift)
            if count is not None:
                count &= 31
                low, high, align = self.number(src)
                if low == high:
                    value = (low << count) & self.MASK
                    return value, value, (value & -value) or (1 << 32)
                return 0, min(high << count, self.MASK), min(align << count, 1 << 32)
        return self.UNKNOWN

    def replace(self, dst, facts):
        for reg, fact in zip(dst, facts):
            self.numbers.pop(reg, None)
            self.carries.pop(reg, None)
            self.words.pop(reg, None)
            if fact is not None:
                self.words[reg] = fact

    def capture(self, token):
        _, limit, align = self.number(token)
        key = 'image_offset_%x' % self.pc
        self.captures[self.pc] = key, token
        self.scalar_reads.setdefault(self.pc, set()).update(scalar_tokens(token))
        return key, limit, align

    def pair(self, token, *, load=False):
        regs = self.registers(token)
        if len(regs) == 2 and (regs[0].startswith('v') or pc_pair(token) is not None):
            low, high = (self.words.get(reg) for reg in regs)
            if low and high and low[1] == 0 and high[:2] == (low[0], 1):
                return low[0].advance(high[2])
        if load:
            self.refuse('incomplete program image address (missing matching high-word carry)', access=True)
        self.refuse('unknown or incomplete calculated jump target from %s' % token)

    def add(self, name, ops, mods):
        vector = name.startswith('v_')
        high = name in self.HIGH_ADDS
        if mods or len(ops) != (4 + high if vector else 3):
            return False
        dst = self.registers(ops[0])
        sources = ops[2:4] if vector else ops[1:]
        if len(dst) != 1 or dst[0][0] != ('v' if vector else 's'):
            return False
        address_sources = [op for op in sources if op in self.words]
        if len(address_sources) != 1:
            return False
        src = address_sources[0]
        other = sources[1] if src == sources[0] else sources[0]
        carry_out = self.mask_pair(ops[1]) if vector else {'scc'}
        if not carry_out:
            return False
        address, part, *correction = self.words[src]
        constant = self.exact(other)
        if high:
            carry_in = self.mask_pair(ops[4]) if vector else {'scc'}
            pending = self.carries.get(next(iter(carry_in), None))
            if (part != 1 or pending is None or constant is None
                    or any(self.carries.get(reg) is not pending for reg in carry_in)
                    or pending[0] is not address
                    or not (constant == 0 or (constant == self.MASK and pending[2]))):
                self.refuse('missing matching high-word carry for program image address', access=True)
            # A pending low can have aliases. Only the high result completes the
            # new address; the matching low result is still usable unchanged.
            # In the relative sum the real machine carry cancels the low
            # word's wrap. A high immediate of -1 subtracts 2**32.
            self.replace(dst, [(pending[1], 1, correction[0] - ((1 << 32) if constant else 0))])
            pending = None  # carry out of the absolute high word is opaque
        else:
            if part != 0 or other in self.carries or (constant is None and len(self.registers(other)) != 1):
                return False
            # Constants may use a signed displacement (+0xffffffff in the
            # high word) or zero extension. The high instruction selects it.
            if constant is None:
                result = address.advance(0, self.capture(other))
            else:
                result = address.advance(constant)
            pending = (address, result, constant is not None)
            self.replace(dst, [(result, 0)])
        self.erased[self.pc] = set(dst) | carry_out
        for reg in carry_out:
            self.words.pop(reg, None)
            self.numbers.pop(reg, None)
            self.carries[reg] = pending
        self.lowered[self.pc] = None
        return True

    def memory(self, name, ops, mods, raw):
        if name.startswith(('flat_store_', 'global_store_', 'flat_atomic_', 'global_atomic_')):
            address = ops[:1] if '_store_' in name or 'glc' not in mods else ops[1:2]
            if name.startswith('global_'):
                address += ops[-1:]
            if any(self.words.keys() & set(self.registers(op)) for op in address):
                self.refuse('store to the program image', access=True)
            return False
        if name not in ('global_load_dword', 'flat_load_dword'):
            return False
        space = name.split('_', 1)[0]
        if len(ops) != (3 if space == 'global' else 2):
            return False
        address_op = ops[2] if space == 'global' and ops[2] != 'off' else ops[1]
        if not self.words.keys() & set(self.registers(address_op)):
            return False
        dst = self.registers(ops[0])
        if len(dst) != 1 or not ops[0].startswith('v'):
            return False
        word0, word1 = struct.unpack('<2I', raw)
        if word0 & ((1 << 25) | (1 << 13)) or word1 & 0x0080ff00:
            return False
        if space == 'flat' and (word1 >> 16) & 0x7f != 0x7d:
            return False
        addr = self.pair(address_op, load=True)
        if space == 'global' and ops[2] != 'off':
            if len(self.registers(ops[1])) != 1 or ops[1] in self.words or ops[1] in self.carries:
                return False
            addr = addr.advance(0, self.capture(ops[1]))
        for mod in mods:
            if mod.startswith('offset:'):
                addr = addr.advance(int(mod[7:]))
            elif mod not in ('glc', 'slc', 'dlc'):
                return False
        self.loads[self.pc] = int(dst[0][1:]), addr
        self.replace(dst, [None])
        return True

    def step(self, name, asm, raw):
        _, ops, mods = parse(asm)
        base = re.sub(r'_e(32|64)$', '', name)
        refs = self.mentions(ops)
        scalar_reads, scalar_writes, scalar_may = scalar_effects(name, asm)
        # Relative register addressing, lane exchanges and EXEC changes can
        # observe words outside the explicit, full-lane def/use chain.
        if (writes_exec(name, ops) or scalar_may & EXEC_KEYS or 'movrel' in name
                or name.startswith(('v_readlane', 'v_writelane', 'v_readfirstlane'))):
            self.refuse('unsupported lane or register access in program image proof')
        if name == 's_getpc_b64' and not mods and len(ops) == 1 and pc_pair(ops[0]) is not None:
            addr = ImageAddress(self.pc + len(raw))
            self.replace(self.registers(ops[0]), [(addr, 0), (addr, 1, 0)])
            self.erased[self.pc] = set(self.registers(ops[0]))
            self.lowered[self.pc] = None
        elif name == 's_setpc_b64' and not mods and len(ops) == 1 and pc_pair(ops[0]) is not None:
            addr = self.pair(ops[0])
            if addr.offsets:
                self.refuse('nonconstant calculated jump target')
            self.lowered[self.pc] = addr.base
            return (addr.base,)
        elif (base in self.MOVES and not mods and len(ops) == 2
              and self.words.keys() & set(self.registers(ops[1]))):
            dst, src = map(self.registers, ops)
            if (len(dst) != (2 if base == 's_mov_b64' else 1) or len(dst) != len(src)
                    or dst[0][0] != base[0] or (base[0] == 's' and src[0][0] != 's')
                    or (base == 's_mov_b64' and any(pc_pair(op) is None for op in ops))
                    or not all(reg in self.words for reg in src)):
                self.refuse('calculated PC word escapes into %s' % name)
            self.replace(dst, [self.words[reg] for reg in src])
            self.erased[self.pc] = set(dst)
            self.lowered[self.pc] = None
        elif (base in self.LOW_ADDS | self.HIGH_ADDS and refs & self.words.keys()
              and self.add(base, ops, mods)):
            pass
        elif refs & self.words.keys() and self.memory(name, ops, mods, raw):
            pass
        else:
            # Only exclude a register from reads when the instruction
            # definitely overwrites it without reading its old value.
            vector_write = (ops and name.startswith(('v_', 'flat_load_', 'global_load_', 'buffer_load_',
                                                     'tbuffer_load_', 'image_load_'))
                            and not name.startswith(('v_cmp', 'v_mac', 'v_fmac', 'v_swap')))
            dst = tuple(r for r in self.registers(ops[0]) if r.startswith('v')) if vector_write else ()
            vector_reads = self.mentions(ops[1:] if dst else ops)
            # Packed/partial writes may preserve old destination bits.
            if name.startswith('v_') and (mods or re.search(r'16|sdwa|dpp', name)):
                vector_reads.update(dst)
            touched = (scalar_reads | vector_reads) & (self.words.keys() | self.carries.keys())
            if touched:
                self.refuse('calculated PC word escapes into %s' % name)
            if name.startswith(('s_swappc', 's_call', 's_rfe', 's_setpc', 's_trap', 's_return')):
                self.refuse('unsupported control transfer in program image proof')
            bound = self.numeric_result(base, ops, mods)
            if not name.startswith('s_'):
                # Non-ALU instructions can also write vector data. Forget
                # their operands' numeric bounds when the access is ambiguous.
                for reg in refs:
                    if reg.startswith('v'):
                        self.numbers.pop(reg, None)
            elif name.startswith(('s_mov_', 's_movk_')):
                scalar_may.discard('scc')
            self.scalar_reads[self.pc] = scalar_reads
            self.replace(dst + tuple(scalar_writes), [None] * (len(dst) + len(scalar_writes)))
            for reg in scalar_may:
                if reg.startswith('s') and reg[1:].isdigit():
                    self.definitions[reg] = self.pc
            for reg in scalar_may - scalar_writes:
                self.numbers.pop(reg, None)
                if self.carries.get(reg) is not None:
                    self.carries[reg] = None
            if bound != self.UNKNOWN and ops:
                regs = self.registers(ops[0])
                if len(regs) == 1 and (regs[0] in dst or regs[0] in scalar_writes):
                    self.numbers[regs[0]] = bound
            if name == 's_branch':
                return (int(ops[0].split('_')[1], 16),)
            if name.startswith('s_cbranch_'):
                if name not in ('s_cbranch_scc0', 's_cbranch_scc1', 's_cbranch_vccz', 's_cbranch_vccnz',
                                's_cbranch_execz', 's_cbranch_execnz'):
                    self.refuse('unsupported conditional transfer in program image proof')
                return (self.pc + len(raw), int(ops[0].split('_')[1], 16))
        for reg in self.erased.get(self.pc, ()):
            if reg.startswith('s') and reg[1:].isdigit():
                self.definitions[reg] = None
        return (self.pc + len(raw),)

    def state(self):
        return self.words.copy(), self.carries.copy(), self.numbers.copy(), self.definitions.copy()

    def merge(self, target, states):
        words, carries, numbers, definitions = states[0]
        for ws, cs, ns, ds in states[1:]:
            if words != ws or carries != cs:
                self.refuse('ambiguous calculated PC words or carry at +0x%x' % target)
            numbers = {reg: bound for reg, bound in numbers.items() if ns.get(reg) == bound}
            definitions = {reg: definitions.get(reg, -1) if definitions.get(reg, -1) == ds.get(reg, -1) else None
                           for reg in definitions.keys() | ds.keys()}
        self.words, self.carries, self.numbers, self.definitions = words, carries, numbers, definitions

    def prove(self):
        pending, incoming, end = [0], {0: [self.state()]}, 0
        next_boundary = 0
        while pending:
            off = self.pc = heapq.heappop(pending)
            self.merge(off, incoming.pop(off))
            self.reaching_definitions[off] = self.definitions.copy()
            # Decode skipped bytes only up to the next target, to prevent a
            # branch into a literal or the second word of an instruction.
            while next_boundary < off:
                next_boundary += D.decode_instruction(self.image, next_boundary)[0]
            if next_boundary != off:
                self.refuse('target in the interior of a program image instruction')
            if off + 4 > len(self.image):
                self.refuse('missing program image instruction')
            size, name, asm, _ = D.decode_instruction(self.image, off)
            if off + size > len(self.image):
                self.refuse('truncated program image instruction')
            asm = check_encoding(self.image, off, size, name, asm)
            self.instructions.append((off, size, name, asm))
            next_boundary = off + size
            end = max(end, next_boundary)
            if name == 's_endpgm':
                continue
            for target in self.step(name, asm, self.image[off:next_boundary]):
                if not next_boundary <= target < len(self.image) or target % 4:
                    self.refuse('invalid forward target in program image proof')
                if target not in incoming:
                    incoming[target] = []
                    heapq.heappush(pending, target)
                incoming[target].append(self.state())

        # Stage's scalar resource/constant registry is also populated in
        # listing order. An erased write must not hide a binding on a bypass.
        erased_sgprs = {reg for erased in self.erased.values() for reg in erased
                        if reg.startswith('s') and reg[1:].isdigit()}
        linear = {}
        for off, _, name, asm in self.instructions:
            for reg in self.scalar_reads.get(off, set()) & erased_sgprs:
                definition = self.reaching_definitions[off].get(reg, -1)
                if definition is None or definition != linear.get(reg, -1):
                    self.pc = off
                    self.refuse('elided PC register %s has no matching ordinary definition' % reg)
            if off in self.lowered:
                linear.update(dict.fromkeys(self.erased.get(off, ())))
            elif off not in self.loads:
                linear.update((reg, off) for reg in scalar_effects(name, asm)[2] & erased_sgprs)

        if not self.loads:
            self.refuse('no load from the program image')
        tables = {}
        for pc, (_, address) in self.loads.items():
            self.pc = pc
            base, offsets = address.base, address.offsets
            limit = sum(bound - bound % align for _, bound, align in offsets)
            if any(bound == self.MASK for _, bound, _ in offsets):
                self.refuse('load from the program image with an unbounded offset', access=True)
            if base < end and base + limit + 4 > 0:
                self.refuse('load from the program image overlaps executed code', access=True)
            if base < 0 or limit > self.MASK or base + limit + 4 > len(self.image):
                self.refuse('load from the program image outside the image', access=True)
            if base % 4 or any(align < 4 for _, _, align in offsets):
                self.refuse('unaligned load from the program image', access=True)
            size = limit + 4
            tables[pc] = tuple(v for v, in struct.iter_unpack('<I', self.image[base:base + size]))
        return tables


def _lds_single(tok):
    """Single hardware VGPR vN (v0..v255) for bounded LDS exchange validation."""
    m = re.fullmatch(r'v(\d+)', tok)
    return int(m.group(1)) if m and int(m.group(1)) < 256 else None


def _lds_vgpr_span(tok, count):
    """Exact contiguous `count`-VGPR range v[n:n+count-1] (v0..v255) for bounded wide LDS exchanges."""
    m = re.fullmatch(r'v\[(\d+):(\d+)\]', tok)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return lo if hi == lo + count - 1 and hi < 256 else None


def _lds_pair(tok):
    """Exact contiguous 2-VGPR pair v[n:n+1] (v0..v255) for bounded 64-bit LDS exchanges."""
    return _lds_vgpr_span(tok, 2)


def _lds_quad(tok):
    """Exact contiguous 4-VGPR quad v[n:n+3] (v0..v255) for bounded 128-bit LDS exchanges."""
    return _lds_vgpr_span(tok, 4)


def lds_byte_offsets(name, mods):
    dual = LDS_DUAL_OPS.get(name)
    if name != 'ds_read2_b32' and dual is None:
        if not mods:
            return (0,)
        if len(mods) == 1 and re.fullmatch(r'offset:\d+', mods[0]):
            offset = int(mods[0][7:])
            if offset <= 0xffff:
                return (offset,)
        raise Unsupported('LDS exchange needs two VGPRs and an optional byte offset')
    # Dual forms encode two independent offsets in units of their opcode's stride.
    stride = dual[2] if dual else 4
    label = name if dual and dual[1] == 8 else 'LDS read2'
    offset_description = 'ordinary dword offsets' if label == 'LDS read2' else 'two ordinary 8-bit offsets'
    offsets = {}
    for mod in mods:
        field = re.fullmatch(r'offset([01]):(\d+)', mod)
        if not field or field[1] in offsets or int(field[2]) > 255:
            raise Unsupported('%s needs %s' % (label, offset_description))
        offsets[field[1]] = stride * int(field[2])
    first, second = offsets.get('0', 0), offsets.get('1', 0)
    if first == second:
        raise Unsupported('%s needs two distinct offsets' % label)
    return first, second


# (is_write, byte_width, fill_shift, fill_unit_name)
LDS_EXCHANGE_OPS = {
    'ds_write_b32': (True, 4, '2', 'dword'),
    'ds_read_b32': (False, 4, None, 'dword'),
    'ds_read2_b32': (False, 4, None, 'dword'),
    'ds_write_b64': (True, 8, '3', 'qword'),
    'ds_read_b64': (False, 8, None, 'qword'),
    'ds_write_b128': (True, 16, '4', 'quad-dword'),
    'ds_read_b128': (False, 16, None, 'quad-dword'),
}


def lds_vgpr_writes(base, ops):
    """VGPR may-writes, including register spans and both destinations of a swap."""
    if base.startswith(INTERP_STORE_PREFIXES + ('s_', 'v_cmp', 'v_readlane', 'v_readfirstlane')):
        return set()
    if base.startswith('v_movrel'):
        return set(range(256))
    dests, _ = effect_destinations(base, ops)
    return {reg for tok in dests if (r := reg_range(tok)) and r[0] == 'v'
            for reg in range(r[1], r[2] + 1)}


def lds_literal(tok):
    return int(tok, 0) if re.fullmatch(r'0x[0-9a-f]+|0|[1-9][0-9]*', tok) else None


def lds_affine(base, src, values, lanes):
    """Recover (byte stride, byte offset, local-ID definition) from plain integer address ALU."""
    if base == 'v_mov_b32' and len(src) == 1:
        return values.get(_lds_single(src[0]))
    if len(src) == 2:
        for literal, register in (src, src[::-1]):
            n = lds_literal(literal)
            value = values.get(_lds_single(register))
            if n is None or value is None or n > 0xffffffff:
                continue
            stride, offset, origin = value
            if base == 'v_lshlrev_b32' and literal == src[0] and n < 32:
                return stride << n, offset << n, origin
            if base == 'v_mul_lo_u32' or (
                    base == 'v_mul_u32_u24' and max(n, offset + (lanes - 1) * stride) < 1 << 24):
                return stride * n, offset * n, origin
            if base == 'v_add_nc_u32':
                return stride, offset + n, origin
    if base == 'v_lshl_add_u32' and len(src) == 3:
        value = values.get(_lds_single(src[0]))
        shift, add = map(lds_literal, src[1:])
        if value is not None and shift is not None and shift < 32 and add is not None:
            stride, offset, origin = value
            return stride << shift, (offset << shift) + add, origin
    return None


def lds_mask_addresses(mask):
    """All possible results of AND with a literal, without guessing the input VGPR."""
    address = mask
    while address:
        yield address
        address = (address - 1) & mask
    yield 0


def check_lds_exchange(ins, group, words):
    """Prove a race-free LDS fill, one uniform barrier, then reads of initialized dwords.

    Return the groupshared size in dwords. Strided qword fills use the extent of the verified slots.

    RDNA 2 ISA (Ch. 9 DS): ds_write_b32 / ds_read_b32 / ds_read2_b32 access 4-byte dwords,
    ds_write_b64 / ds_read_b64 access two contiguous little-endian dwords (v[n] at addr+offset, v[n+1] at addr+offset+4), and
    ds_write_b128 / ds_read_b128 access four contiguous little-endian dwords (v[n]..v[n+3] at addr+offset..addr+offset+12).
    All widths share the dword-indexed groupshared uint lds[N] array and require 4-byte alignment (natural 8/16-byte alignment is not required on GFX10/RDNA2 LDS).
    LLVM 15 AMDGPUUsage defines COMPUTE_PGM_RSRC2.GRANULATED_LDS_SIZE in 128-dword (512-byte) units on gfx10.
    """
    # One reason per failed condition, with the measured value, so a refusal count says which limit the title exceeds.
    if group[0] < 1 or group[1:] != (1, 1):
        raise Unsupported('LDS exchange needs a 1D group (the group is %dx%dx%d)' % tuple(group))
    if words == 0:
        raise Unsupported('LDS exchange needs a nonzero LDS allocation (COMPUTE_PGM_RSRC2 allocates 0 bytes)')
    if not 0 < words <= 8192:
        raise Unsupported('LDS exchange needs an allocation of at most 32 KiB (COMPUTE_PGM_RSRC2 allocates %d KiB)' % (words * 4 // 1024))
    fill_spans = []
    qword_slots = []
    qword_words = set()
    addresses = {0: (1, 0, -1)}
    saved_ids = {}
    first_x_write = None
    published = False
    read = False
    prev = (None, [], [])
    for i, (_, _, _, asm) in enumerate(ins):
        name, ops, mods = parse(asm)
        base = name.removesuffix('_e32').removesuffix('_e64')
        if (base.startswith(('s_branch', 's_cbranch', 's_setpc', 's_swappc', 's_call', 's_rfe', 's_trap'))
                or writes_exec(base, ops)
                or base == 's_endpgm' and i + 1 < len(ins)):
            raise Unsupported('LDS exchange needs straight-line code and entry EXEC')
        if base == 's_barrier':
            if (not (fill_spans or qword_slots) or published or prev != ('s_waitcnt', ['lgkmcnt(0)'], [])
                    or mods or ops not in ([], ['0'], ['0x0'])):
                raise Unsupported('LDS exchange needs one barrier after a completed fill')
            published = True
        elif base in LDS_EXCHANGE_OPS:
            write, width, shift, unit_name = LDS_EXCHANGE_OPS[base]
            dwords = width // 4
            addr_op = ops[0] if (write and len(ops) == 2) else (ops[1] if len(ops) == 2 else '')
            val_op = ops[1] if (write and len(ops) == 2) else (ops[0] if len(ops) == 2 else '')
            if base == 'ds_read2_b32':
                if len(ops) != 2 or _lds_pair(val_op) is None or _lds_single(addr_op) is None:
                    raise Unsupported('LDS read2 needs a VGPR pair and a byte address VGPR')
            elif (len(ops) != 2 or _lds_single(addr_op) is None
                  or (_lds_single(val_op) is None if dwords == 1 else _lds_vgpr_span(val_op, dwords) is None)):
                need_val = ('a matching data VGPR' if dwords == 1
                            else 'an aligned %d-VGPR %s' % (dwords, 'pair' if dwords == 2 else 'quad'))
                raise Unsupported('LDS exchange needs %s, %s and an optional byte offset'
                                  % ('two VGPRs' if dwords == 1 else 'a VGPR address', need_val))
            offsets = lds_byte_offsets(base, mods)
            producer, args, modifiers = prev
            if write:
                offset, = offsets
                reason = 'LDS exchange needs one bounded %s per local X ID' % unit_name
                address = addresses.get(_lds_single(addr_op))
                if first_x_write is not None and (address is None or address[2] not in saved_ids.values()):
                    raise Unsupported('LDS fill needs the unchanged entry local X ID: ' + first_x_write)
                if published or offset % 4:
                    raise Unsupported(reason)
                if base == 'ds_write_b64':
                    if fill_spans or address is None:
                        raise Unsupported(reason)
                    stride, start, _ = address
                    start += offset
                    end = start + (group[0] - 1) * stride + width
                    if (stride % 4 or start % 4 or end > words * 4 or stride < width * (len(qword_slots) + 1)
                            or qword_slots and stride != qword_slots[0][0]):
                        raise Unsupported(reason)
                    slot = {word for addr in range(start, end - width + 1, stride)
                            for word in range(addr, addr + width, 4)}
                    if slot & qword_words:
                        raise Unsupported(reason)
                    qword_words.update(slot)
                    qword_slots.append((stride, start, end))
                else:
                    # Preserve the existing narrow/quad-dword layouts and the two consecutive dword banks.
                    end = offset + group[0] * width
                    if (qword_slots or end > words * 4
                            or fill_spans and (base != 'ds_write_b32' or producer != 'ds_write_b32')):
                        raise Unsupported(reason)
                    if fill_spans:
                        if len(fill_spans) != 1:
                            raise Unsupported('LDS exchange allows at most two consecutive bounded dword fills')
                        if addr_op != args[0]:
                            raise Unsupported('LDS dword fills must use the same byte address VGPR')
                        start0, end0 = fill_spans[0]
                        if offset < end0 and start0 < end:
                            raise Unsupported('LDS dword fills must be disjoint')
                    elif (producer != 'v_lshlrev_b32' or args[:2] != [addr_op, shift]
                          or modifiers or address is None or address[:2] != (width, 0)):
                        raise Unsupported(reason)
                    fill_spans.append((offset, end))
            else:
                if not published or modifiers or not args or args[0] != addr_op:
                    raise Unsupported('LDS read needs a bounded address after the fill barrier')
                if producer == 'v_mov_b32' and len(args) == 2 and re.fullmatch(r'0x[0-9a-f]+|\d+', args[1]):
                    low = high = int(args[1], 0)
                elif (producer == 'v_and_b32' and len(args) == 3 and re.fullmatch(r'0x[0-9a-f]+|\d+', args[1])
                      and _lds_single(args[2]) is not None):
                    low, high = 0, int(args[1], 0)
                else:
                    raise Unsupported('LDS read needs a literal or literal-masked byte address')
                # Contiguous legacy fills keep their original read proof. Strided slots may
                # leave holes, so test every possible masked start against the initialized dwords.
                spans = fill_spans
                if len(qword_slots) == 1 and qword_slots[0][0] == 8:
                    spans = [qword_slots[0][1:]]
                if low % 4 or high % 4 or any(offset % 4 for offset in offsets):
                    raise Unsupported('LDS read outside the aligned, initialized span')
                if spans:
                    initialized = all(any(start <= low + offset and high + offset + width <= end
                                          for start, end in spans) for offset in offsets)
                else:
                    starts = (low,) if low == high else lds_mask_addresses(high)
                    initialized = high < words * 4 and all(
                        addr + offset + word in qword_words
                        for addr in starts for offset in offsets for word in range(0, width, 4))
                if not initialized:
                    raise Unsupported('LDS read outside the aligned, initialized span')
                if i + 1 >= len(ins) or ins[i + 1][3] != 's_waitcnt lgkmcnt(0)':
                    raise Unsupported('LDS read needs a completed LGKM wait')
                read = True
        elif base.startswith('ds_') or 'lds' in mods:
            raise Unsupported('LDS exchange does not cover ' + asm)
        elif not published:
            if not (fill_spans or qword_slots) and (
                    base.startswith(('v_readlane', 'v_readfirstlane', 'v_writelane', 'v_permlane', 'v_movrel',
                                     's_sendmsg', 's_sethalt')) or base.endswith(('_dpp', '_dpp8'))):
                raise Unsupported('LDS fill needs the unchanged entry local X ID: ' + name)
            written = lds_vgpr_writes(base, ops)
            dest = _lds_single(ops[0]) if ops else None
            value = lds_affine(base, ops[1:], addresses, group[0]) if dest in written and not mods else None
            if 0 in written and first_x_write is None:
                first_x_write = name
            for reg in written:
                addresses.pop(reg, None)
                saved_ids.pop(reg, None)
            if value is not None:
                stride, offset, origin = value
                if offset + (group[0] - 1) * stride <= 0xffffffff:
                    # A saved ID gets its own definition. Derivations keep that definition
                    # so a later overwrite of the saved register invalidates the proof.
                    if first_x_write is None and base == 'v_mov_b32' and value[:2] == (1, 0) and dest != 0:
                        origin = saved_ids[dest] = i
                    addresses[dest] = stride, offset, origin
        prev = base, ops, mods
    if not read:
        raise Unsupported('LDS exchange needs a fill, a barrier and a completed read')
    if qword_slots and (len(qword_slots) > 1 or qword_slots[0][0] != 8):
        return max(end for _, _, end in qword_slots) // 4
    return words


# ---- slot 57 mesh stage: the group-memory count exchange (#2413) --------------------------------
# A group is two wave64 waves and thread = wave * 64 + lane. 128 is also D3D12's limit for the numthreads of a mesh
# shader, and the declared size of both output arrays.
MESH_THREADS = 128
MESH_VREG = re.compile(r'\bv(?:\[(\d+):(\d+)\]|(\d+))')
MESH_EXCHANGE = [
    r's_mov_b64 exec, 1',
    r's_mov_b64 (?P<lane>s\[\d+:\d+\]), exec',
    r's_and_b32 (?P<a_bits>s\d+), s3, 0xf000000',
    r's_cbranch_scc1 label_(?P<a_other>\w+)',
    r'v_lshrrev_b32_e32 v5, 7, v5',
    r'v_mov_b32_e32 (?P<a_zero>v\d+), 0',
    r'ds_write_b32 {a_zero}, v5 offset:8',
    r's_waitcnt_depctr 0xffe3',
    r's_barrier 0x0',
    r's_branch label_(?P<a_join>\w+)',
    r's_barrier 0x0',
    r'v_mov_b32_e32 (?P<a_zero2>v\d+), 0',
    r'ds_read_b32 v5, {a_zero2} offset:8',
]
MESH_COUNTS = [
    r's_barrier 0x0',
    r's_cmp_eq_i32 (?P<wave>s\d+), 0',
    r's_cbranch_scc0 label_(?P<b_reader>\w+)',
    r's_mov_b64 exec, {lane}',
    r's_cbranch_execz label_(?P<b_wskip>\w+)',
    r'v_mov_b32_e32 (?P<b_p>v\d+), (?P<prims>s\d+)',
    r'v_mov_b32_e32 (?P<b_zero>v\d+), 0',
    r'v_mov_b32_e32 (?P<b_v>v\d+), (?P<verts>s\d+)',
    r'ds_write_b32 {b_zero}, {b_p}',
    r'ds_write_b32 {b_zero}, {b_v} offset:4',
    r's_mov_b64 exec, -1',
    r's_waitcnt_depctr 0xffe3',
    r's_barrier 0x0',
    r's_lshl_b32 (?P<b_tmp>s\d+), {prims}, 12',
    r's_or_b32 m0, {b_tmp}, {verts}',
    r's_sendmsg 0x9',
    r's_branch label_(?P<b_join>\w+)',
    r's_barrier 0x0',
    r's_mov_b64 exec, {lane}',
    r's_cbranch_execz label_(?P<b_rskip>\w+)',
    r'v_mov_b32_e32 (?P<b_zero2>v\d+), 0',
    r'ds_read_b32 (?P<b_r0>v\d+), {b_zero2}',
    r'ds_read_b32 (?P<b_r1>v\d+), {b_zero2} offset:4',
    r's_waitcnt lgkmcnt\(1\)',
    r'v_readfirstlane_b32 {prims}, {b_r0}',
    r's_waitcnt lgkmcnt\(0\)',
    r'v_readfirstlane_b32 {verts}, {b_r1}',
    r's_mov_b64 exec, -1',
]


def mesh_vregs(tokens):
    out = set()
    for m in MESH_VREG.finditer(' '.join(tokens)):
        out.update(range(int(m[1] or m[3]), int(m[2] or m[3]) + 1))
    return out


def mesh_run(ins, at, what, patterns, env):
    """Match consecutive instructions; {name} in a pattern is the text an earlier (?P<name>...) bound."""
    for k, pattern in enumerate(patterns):
        text = ins[at + k][3] if at + k < len(ins) else ''
        m = re.fullmatch(re.sub(r'\{(\w+)\}', lambda ref: re.escape(env[ref[1]]), pattern), text)
        if not m:
            raise Unsupported('slot 57 mesh %s: instruction %d of the accepted shape (%s) is absent'
                              % (what, k + 1, pattern.split()[0]))
        env.update(m.groupdict())
    return at + len(patterns)


MESH_VCC_BRANCHES = ('s_cbranch_vccz', 's_cbranch_vccnz')
# Vector ALU forms that take a lane mask as a per-lane select or carry, which the one-lane model handles.
MASK_AWARE_VALU = frozenset({'v_cndmask_b32', 'v_cndmask_b16', 'v_readlane_b32', 'v_writelane_b32', 'v_readfirstlane_b32'})
MASK_AWARE_VALU_RE = re.compile(r'_co(_ci)?_|addc|subb|_co$|^v_mad_[iu]64_[iu]32$|^v_(cndmask|readlane|writelane|readfirstlane)_')
# Scalar ALU forms that move or combine lane masks bit by bit, which the one-lane model handles (s_ff1_i32_b32 refuses
# anything but exec_lo at the entry mask itself). Any other scalar op that
# reads a lane mask (s_cmp*, s_cmpk*, s_bcnt*, s_lshr, s_add ...) reads the whole wave's word as a number.
MASK_AWARE_SALU_RE = re.compile(r'^s_(mov|and|or|xor|andn2|orn2|nand|nor|xnor|not|cselect|wqm|cbranch|and_saveexec|or_saveexec|'
                                r'xor_saveexec|andn2_saveexec|orn2_saveexec|nand_saveexec|nor_saveexec|xnor_saveexec|'
                                r'ff1|bitset[01]|setpc|getpc|swappc|branch|waitcnt|endpgm|nop)(_|$)')
# Scalar ops whose result is a lane mask when a source is one (a copy or a bitwise combination), unlike s_ff1 or s_bitset.
MASK_COPY_SALU_RE = re.compile(r'^s_(mov|and|or|xor|andn2|orn2|nand|nor|xnor|not|wqm)_b(32|64)$')
VCC_KEYS = {'vcc', 'vcc_hi'}


MESH_LOOP_REFUSAL = 'slot 57 mesh: a loop outside the group-memory exchanges is not covered'
# Scalar buffer loads a pre-exchange execz block may issue once per wave (#2660 accepted one s_buffer_load_dword).
MESH_WAVE_BLOCK_LOADS = ('s_buffer_load_dword', 's_buffer_load_dwordx2', 's_buffer_load_dwordx4',
                         's_buffer_load_dwordx8', 's_buffer_load_dwordx16', 's_load_dword', 's_load_dwordx2',
                         's_load_dwordx4', 's_load_dwordx8', 's_load_dwordx16')


def mesh_wave_block_scalar(name, asm):
    """True for an instruction of a pre-exchange execz block that the lowering may run for every lane of a wave with
    an active lane: a listed scalar buffer load or scalar ALU whose operands are SGPRs, SCC and constants only."""
    if name in MESH_WAVE_BLOCK_LOADS:
        pass
    elif name == 'v_readfirstlane_b32':
        return bool(re.fullmatch(r'v_readfirstlane_b32 s\d+, v\d+', asm))
    elif name.startswith('v_cmp_'):
        # A per-lane compare whose only scalar effect is its VCC destination; the caller refuses a VCC live at the join.
        return not re.search(r'\b(exec|m0|src_)', asm) and scalar_effects(name, asm)[2] <= VCC_KEYS
    elif not name.startswith('s_') or name.startswith(EXECZ_REFUSED + NO_SCALAR_DESTINATION + (
            's_load', 's_getpc', 's_setpc', 's_swappc', 's_call', 's_get', 's_set', 's_movrel', 's_rfe', 's_trap',
            's_memtime', 's_memrealtime', 's_buffer_load', 's_scratch', 's_dcache', 's_gl1', 's_wqm', 's_quadmask',
            's_denorm_mode', 's_round_mode')):
        return name.startswith(('s_cmp_', 's_bitcmp')) and not re.search(r'\b(exec|vcc|m0|src_)', asm)
    if re.search(r'\b(exec|m0|src_|ttmp|tba|tma|flat_scratch|xnack)', asm) or 'saveexec' in name:
        return False
    reads, must, may = scalar_effects(name, asm)
    return all(reg in ('scc', 'vcc', 'vcc_hi') or re.fullmatch(r's\d+', reg) for reg in reads | must | may)


MESH_TRIP_STEPS = ('s_add_u32', 's_add_i32', 's_sub_u32', 's_sub_i32')
MESH_TRIP_MAX = 64


def mesh_inline_u32(tok):
    """An inline integer or float-bit SOP operand as a u32, or None."""
    if tok in FLOATS:
        return fbits(FLOATS[tok]) & 0xffffffff
    m = re.fullmatch(r'-?(?:0x[0-9a-f]+|\d+)', tok)
    if not m:
        return None
    value = int(tok, 0)
    return value & 0xffffffff if value >= 0 else (value + (1 << 32)) & 0xffffffff


def mesh_trip_count(init, step_name, step_imm, cmp_name, limit, latch_name, cmp_counter_left):
    """Finite trip count for a scalar counted loop, or None when unbounded / unsupported."""
    counter = init & 0xffffffff
    step = step_imm & 0xffffffff
    signed = '_i32' in cmp_name  # compare opcode alone
    take_when_scc = latch_name == 's_cbranch_scc1'

    def as_signed(word):
        return word - (1 << 32) if word >= (1 << 31) else word

    for trips in range(1, MESH_TRIP_MAX + 1):
        if step_name.startswith('s_add'):
            counter = (counter + step) & 0xffffffff
        else:
            counter = (counter - step) & 0xffffffff
        left, right = (counter, limit) if cmp_counter_left else (limit, counter)
        if signed:
            left, right = as_signed(left), as_signed(right)
        if cmp_name.endswith(('_eq_i32', '_eq_u32')):
            scc = left == right
        elif cmp_name.endswith(('_lg_i32', '_lg_u32')):
            scc = left != right
        elif cmp_name.endswith(('_gt_i32', '_gt_u32')):
            scc = left > right
        elif cmp_name.endswith(('_ge_i32', '_ge_u32')):
            scc = left >= right
        elif cmp_name.endswith(('_lt_i32', '_lt_u32')):
            scc = left < right
        elif cmp_name.endswith(('_le_i32', '_le_u32')):
            scc = left <= right
        else:
            return None
        if take_when_scc != scc:
            return trips
    return None


def mesh_constant_trip(ins, header, latch, offsets):
    """Prove one constant-trip SCC back-edge: (trips,) or None.

    Shape (census "constant pattern", with an exact trip count): s_mov_b32 sN, C immediately before the
    header; the body changes sN only by s_add/s_sub of a constant; the last body instruction is s_cmp of sN
    against a constant; the latch is s_cbranch_scc0/scc1 back to the header. No other branch enters or leaves
    the body, and the body does not write EXEC, touch a barrier, LDS, sendmsg, or export.
    """
    if header <= 0 or latch <= header or latch >= len(ins):
        return None
    latch_name = ins[latch][2]
    if latch_name not in ('s_cbranch_scc0', 's_cbranch_scc1'):
        return None
    init_name, init_ops, _ = parse(ins[header - 1][3])
    if init_name != 's_mov_b32' or len(init_ops) != 2 or not re.fullmatch(r's\d+', init_ops[0]):
        return None
    init_imm = mesh_inline_u32(init_ops[1])
    if init_imm is None:
        return None
    counter = init_ops[0]
    counter_n = int(counter[1:])
    body = ins[header:latch]
    if not body or not body[-1][2].startswith('s_cmp_'):
        return None
    cmp_name, cmp_ops, _ = parse(body[-1][3])
    if len(cmp_ops) != 2:
        return None
    if cmp_ops[0] == counter:
        cmp_counter_left, limit_tok = True, cmp_ops[1]
    elif cmp_ops[1] == counter:
        cmp_counter_left, limit_tok = False, cmp_ops[0]
    else:
        return None
    limit = mesh_inline_u32(limit_tok)
    if limit is None:
        return None
    step = None
    for off, size, name, asm in body[:-1]:
        if name.startswith(('s_branch', 's_cbranch', 's_setpc', 's_call', 's_swappc')):
            return None
        if name.startswith(('ds_', 'exp')) or name in ('s_barrier', 's_sendmsg', 's_sendmsghalt'):
            return None
        _, ops, _ = parse(asm)
        if writes_exec(name, ops) or name.startswith('v_cmpx') or name == 's_and_saveexec_b64':
            return None
        if name.startswith('s_') and re.search(r'\bvcc(_lo|_hi)?\b', asm):
            return None
        writes = set()
        for tok in destinations(name, ops):
            r = reg_range(tok)
            if r and r[0] == 's':
                writes.update(range(r[1], r[2] + 1))
        if counter_n in writes:
            if name not in MESH_TRIP_STEPS or len(ops) != 3 or ops[0] != counter:
                return None
            if ops[1] == counter and mesh_inline_u32(ops[2]) is not None:
                imm = mesh_inline_u32(ops[2])
            elif (ops[2] == counter and mesh_inline_u32(ops[1]) is not None
                  and name.startswith('s_add')):
                # Add is commutative; s_sub with the counter as SRC1 is a different recurrence.
                imm = mesh_inline_u32(ops[1])
            else:
                return None
            if step is not None:
                return None
            step = (name, imm)
    if step is None:
        return None
    # No other branch may enter the body past the header or leave it before the latch.
    for i, (off, _, name, asm) in enumerate(ins):
        if not name.startswith(('s_cbranch', 's_branch')):
            continue
        target = int(parse(asm)[1][0].split('_')[1], 16)
        dest = offsets.get(target)
        if dest is None:
            continue
        if i == latch:
            continue
        if header <= i < latch and not (header <= dest <= latch):
            return None
        if header < dest <= latch and not (header <= i <= latch):
            return None
    trips = mesh_trip_count(init_imm, step[0], step[1], cmp_name, limit, latch_name, cmp_counter_left)
    return (trips,) if trips is not None else None


MESH_MASK_WORDS = {'exec_lo': ('exec', 0), 'exec_hi': ('exec', 32),
                   'vcc_lo': ('vcc', 0), 'vcc_hi': ('vcc', 32)}
MESH_MASK_SELECT = 's_cselect_b32_mesh'


def mesh_skeleton(ins, vcc_branches, wave_limits, wave_loads, wave_votes):
    """Prove the one accepted group-memory shape of a slot 57 mesh program and replace it by per-thread instructions.

    Accepted shape, in program order (anything else is refused with its own message):
    - Exchange: exec = 1 (lane 0 of each wave), the mask saved in an SGPR pair; wave 0 (s3 bits 24..27 zero) stores
      entry v5 >> 7 at byte 8, waits and passes a barrier; every other wave passes a barrier and reads byte 8 to v5.
    - Code up to the next barrier. Its only read of the exchanged v5 may be a v_readfirstlane_b32 under exec = -1.
      Other first-lane reads require a defined source and wave-coherent control flow. Wave-load blocks can use
      the original prefix test or vote on the current EXEC after an unsigned thread comparison.
      After a full EXEC reset, an unsigned GT or LT EXEC compare with defined operands may guard a forward
      execz branch to that barrier. The skipped instructions only affect active lanes or dead scalars, and
      both exchange arms replace EXEC, so a per-lane skip has the same visible effects as a per-wave skip.
    - Counts: one barrier; the wave whose index (s_bfe_u32 of s3, 4 bits at 24) is zero stores SGPR P at byte 0 and
      SGPR V at byte 4 under the saved lane-0 mask, waits, passes a barrier and sends one allocation request with
      m0 = P << 12 | V; every other wave passes a barrier and reads both words back into the same P and V.
    - Code to s_endpgm. Exports only here: pos0 and param under exec = -1 narrowed by one
      v_cmpx_gt_i32 V, thread; the primitive export likewise with P. thread is the v_mbcnt pair over wave << 6.
    Each path therefore passes three barriers, each store has one writer (one lane of one wave), and a wait and a
    barrier separate it from every read. The lowering keeps one writer and one barrier: thread 0 stores, all read.

    Registers the two arms leave different, that the lowering does not compute, or that hold an entry value the
    model does not define, are refused when read: VGPRs by the scan below, SGPRs by a symbol (setup_entry and the
    op_x2w_mesh_* handlers), SCC and m0 after the counts. Of s3 only the wave index is modelled.
    """
    env = {}
    first = 1 if ins and ins[0][2] == 's_inst_prefetch' else 0
    start = next((i for i in range(first, len(ins)) if ins[i][3] == 's_mov_b64 exec, 1'), first)
    a_end = mesh_run(ins, start, 'group-index exchange', MESH_EXCHANGE, env)
    count_at = next((i for i in range(a_end, len(ins)) if ins[i][2] == 's_barrier'), len(ins))
    b_end = mesh_run(ins, count_at, 'count exchange', MESH_COUNTS, env)

    def lands(key, index, what):
        if index >= len(ins) or int(env[key], 16) != ins[index][0]:
            raise Unsupported('slot 57 mesh %s: a branch leaves the accepted shape' % what)
    lands('a_other', start + 10, 'group-index exchange')
    lands('a_join', a_end, 'group-index exchange')
    lands('b_reader', count_at + 17, 'count exchange')
    lands('b_wskip', count_at + 10, 'count exchange')
    lands('b_join', b_end, 'count exchange')
    lands('b_rskip', count_at + 27, 'count exchange')
    lane = set(range(*(lambda r: (r[1], r[2] + 1))(reg_range(env['lane']))))
    a_bits = int(env['a_bits'][1:])
    sreg = {key: int(env[key][1:]) for key in ('wave', 'prims', 'verts', 'b_tmp')}
    vreg = {key: int(env[key][1:]) for key in ('a_zero', 'a_zero2', 'b_p', 'b_zero', 'b_v', 'b_zero2', 'b_r0', 'b_r1')}
    if (len({vreg['b_p'], vreg['b_zero'], vreg['b_v']}) != 3 or len({vreg['b_r0'], vreg['b_r1']}) != 2
            or vreg['b_r0'] == vreg['b_zero2'] or 5 in (vreg['a_zero'], vreg['a_zero2'])
            or len({sreg['prims'], sreg['verts'], sreg['b_tmp']}) != 3 or {sreg['prims'], sreg['verts']} & lane
            or sreg['wave'] in lane | {3, sreg['prims'], sreg['verts']} or a_bits in lane | {3} or 3 in lane):
        raise Unsupported('slot 57 mesh count exchange: registers of the accepted shape overlap')
    if not ins or ins[-1][2] != 's_endpgm':
        raise Unsupported('slot 57 mesh program does not end with s_endpgm')

    offsets = {item[0]: i for i, item in enumerate(ins)}
    edges = {}
    count_entry = count_entry_branch = None
    # header index -> (latch index, trip count). Constant-trip loops are unrolled below.
    constant_trips = {}
    no_effect = lambda i: ins[i][2].startswith(('s_nop', 's_waitcnt', 's_clause'))
    exchange_branch_refusal = 'slot 57 mesh: a branch enters or crosses a group-memory exchange'

    def lane_only(i):
        _, _, name, asm = ins[i]
        return (name.startswith(('v_', 'buffer_load')) and not name.startswith(EXECZ_REFUSED)
                and not scalar_effects(name, asm)[2])

    def dead_count_entry_scalar(i):
        _, _, name, asm = ins[i]
        if not re.fullmatch(
                r's_(and|or|xor|andn2|orn2|bfe|max|min|mul|add|sub|lshl|lshr|ashr|mov|movk)_[biu]\d+', name):
            return False
        _, must, may = scalar_effects(name, asm)
        written = must | may
        regs = written - {'scc'}
        if not written or any(not re.fullmatch(r's\d+', reg) for reg in regs):
            return False
        # Uses within the skipped block are fine. Every path out of it joins at the count exchange.
        if any(regs & scalar_effects(*ins[j][2:])[0] for j in range(count_at, len(ins))):
            return False
        if 'scc' in written:
            for j in range(count_at, len(ins)):
                reads, must, may = scalar_effects(*ins[j][2:])
                if ins[j][2].startswith(('s_cbranch', 's_branch', 's_setpc', 's_call', 's_swappc', 's_endpgm')):
                    return False
                if 'scc' in reads:
                    return False
                if 'scc' in must:  # a may-write alone (s_barrier) neither reads nor kills it
                    return True
            return False
        return True

    for lo, hi in ((0, start), (a_end, count_at), (b_end, len(ins))):
        for i in range(lo, hi):
            off, _, name, asm = ins[i]
            if name.startswith(('s_setpc', 's_call', 's_swappc')):
                raise Unsupported('slot 57 mesh: indirect control flow (%s) is not covered' % name)
            if name == 's_endpgm' and i != len(ins) - 1:
                raise Unsupported('slot 57 mesh: an early s_endpgm is not covered')
            if not name.startswith(('s_cbranch', 's_branch')):
                continue
            target = int(parse(asm)[1][0].split('_')[1], 16)
            dest = offsets.get(target)
            if dest is None:
                raise Unsupported('slot 57 mesh: a branch target is not an instruction boundary')
            if not lo <= dest < hi:
                # Both count exchange arms replace the incoming EXEC before lane-dependent work.
                if lo != a_end or dest != count_at or name != 's_cbranch_execz':
                    raise Unsupported(exchange_branch_refusal)
                compare = i - 1
                while compare >= lo and ins[compare][2] in ('s_nop', 's_waitcnt'):
                    compare -= 1
                if (compare <= lo or ins[compare - 1][3] != 's_mov_b64 exec, -1'
                        or ins[compare][2] not in ('v_cmpx_gt_u32_e32', 'v_cmpx_lt_u32_e32')
                        or any(not (no_effect(k) or lane_only(k) or dead_count_entry_scalar(k))
                               for k in range(i + 1, dest))):
                    raise Unsupported(exchange_branch_refusal)
                _, sources, _ = parse(ins[compare][3])
                if len(sources) != 2 or any(not (re.fullmatch(r'[sv]\d+', tok) or tok in FLOATS
                                                 or pc_word(tok) is not None) for tok in sources):
                    raise Unsupported(exchange_branch_refusal)
                count_entry, count_entry_branch = compare, i
            if target <= off:
                proven = mesh_constant_trip(ins, dest, i, offsets)
                if proven is None or dest in constant_trips:
                    raise Unsupported(MESH_LOOP_REFUSAL)
                constant_trips[dest] = (i, proven[0])
                continue
            if name not in ('s_branch', 's_cbranch_scc0', 's_cbranch_scc1', 's_cbranch_execz') + MESH_VCC_BRANCHES:
                raise Unsupported('slot 57 mesh: %s on a wave-wide condition is not covered' % name)
            edges[i] = dest

    targets = set(edges.values())
    if targets & set(constant_trips):
        raise Unsupported(MESH_LOOP_REFUSAL)
    if count_entry is not None and targets.intersection(range(count_entry, count_at)):
        raise Unsupported(exchange_branch_refusal)
    first_post_count_branch = next((i for i in edges if i >= b_end), None)
    shift_branch = None
    load_compares = {}  # compare index -> thread VGPR of each accepted pre-exchange wave block
    nested_saves, nested_restores = {}, {}  # nested if of a wave block: save index -> restore index, and back
    execz_effects = 'slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes'

    def live_at_join(reg, join):
        pending, seen = [join], set()
        while pending:
            i = pending.pop()
            if i >= len(ins) or i in seen:
                continue
            seen.add(i)
            if i == count_at:
                if reg in (env['prims'], env['verts']):
                    return True
                if reg != env['b_tmp']:
                    pending.append(b_end)
                continue
            name, asm = ins[i][2:]
            reads, writes, _ = scalar_effects(name, asm)
            if reg in reads:
                return True
            if reg not in writes:
                if i in edges:
                    pending.append(edges[i])
                if name not in ('s_branch', 's_endpgm'):
                    pending.append(i + 1)
        return False

    def dead_block_scalar(i, join):
        """Scalar ALU in a skipped block whose results no path reads past the join. Lanes that skip the block per lane
        never read them, and lanes that run it compute the wave-uniform values the guest computes once per wave."""
        name, asm = ins[i][2:]
        if name in MESH_WAVE_BLOCK_LOADS or not name.startswith('s_') or not mesh_wave_block_scalar(name, asm):
            return False
        _, must, may = scalar_effects(name, asm)
        return not any(live_at_join(reg, join) for reg in must | may)

    inner_vcc = set()
    load_branches = {}
    for source, target in edges.items():
        if ins[source][2] == 's_cbranch_execz':
            # A single scalar shift followed by a full EXEC restore has a separate wave-level proof below.
            if (source == first_post_count_branch and source >= b_end + 2 and target == source + 2
                    and ins[source + 1][2] == 's_lshl_b32' and ins[target][3] == 's_mov_b64 exec, -1'):
                shift_branch = source
                continue
            if a_end <= source < target < count_at:
                loads = [i for i in range(source + 1, target) if ins[i][2] in MESH_WAVE_BLOCK_LOADS]
                if loads:
                    compare = source - 1
                    while compare > a_end and not writes_exec(ins[compare][2], parse(ins[compare][3])[1]):
                        compare -= 1
                    prefix = re.fullmatch(r'v_cmpx_gt_u32_e32 (s\d+), v(\d+)', ins[compare][3])
                    vector_limit = re.fullmatch(r'v_cmpx_lt_u32_e32 v(\d+), v\d+', ins[compare][3])
                    # Inner forward scalar branches of the block (decided by SCC) stay inside it: the translation takes them
                    # per lane, which is the wave's decision when the SCC is wave-uniform (the emitter checks that).
                    vcc_edges = {
                        i for i in range(source + 2, target)
                        if ins[i][2] in MESH_VCC_BRANCHES and i < edges[i] <= target and i not in targets
                        and re.fullmatch(r'v_cmp_\w+_e(?:32|64) vcc(?:_lo)?, .+', ins[i - 1][3])
                    }
                    inner = [i for i in range(source + 1, target) if i in edges
                             and ins[i][2] in ('s_cbranch_scc0', 's_cbranch_scc1', 's_branch')
                             and i < edges[i] <= target]
                    # A nested if of the block: s_and_saveexec_b64 sK:sK+1, vcc / s_cbranch_execz over a body / the restore
                    # s_or_b64 exec, exec, sK:sK+1. Every thread holds its own EXEC bit and the saved mask is one bit per
                    # thread, so narrowing and restoring is per lane in the one-lane model; the per-lane skip of the
                    # branch is the generic execz rule, checked on its own edge below. The save, the branch and the
                    # restore are not scalar instructions the wave block would run once per wave.
                    nested = set()
                    for i in range(source + 2, target):
                        if ins[i][2] != 's_cbranch_execz' or i not in edges or not i < edges[i] < target:
                            continue
                        save = re.fullmatch(r's_and_saveexec_b64 s\[(\d+):(\d+)\], vcc', ins[i - 1][3])
                        done = ins[edges[i]][3]
                        if (not save or int(save[2]) != int(save[1]) + 1 or done != 's_or_b64 exec, exec, s[%s:%s]' % save.groups()
                                or not ins[i - 2][2].startswith('v_cmp_')
                                or any(writes_exec(ins[k][2], parse(ins[k][3])[1]) or re.search(r'\bs\[?%s\b|\bs%s\b' % (save[1], save[2]), ins[k][3])
                                       for k in range(i + 1, edges[i]))
                                or any(live_at_join(reg, edges[i] + 1) for reg in ('s' + save[1], 's' + save[2]))):
                            continue
                        nested |= {i - 1, i, edges[i]}
                        nested_saves[i - 1], nested_restores[edges[i]] = edges[i], i - 1
                    inner += sorted(vcc_edges | {i for i in nested if ins[i][2] == 's_cbranch_execz'})
                    scalar = [i for i in range(source + 1, target)
                              if i not in inner and i not in nested and not (no_effect(i) or lane_only(i))]
                    # The guest runs every scalar instruction of the block once per wave that has an active lane.
                    # The lowering runs them for all lanes of such a wave, so they may only read and write SGPRs,
                    # SCC and a VCC that is dead at the join: no EXEC, M0, branch, barrier, message, store or atomic.
                    # A per-lane v_cmp into VCC is allowed under the same dead-at-join rule.
                    written, masks = set(), set()
                    for i in scalar:
                        reads, must, may = scalar_effects(*ins[i][2:])
                        written |= {reg for reg in must | may if re.fullmatch(r's\d+', reg)}
                        masks |= ((must | may) & VCC_KEYS) | (must & {'scc'})  # every s_ op may-writes SCC
                    live = sorted((reg for reg in written if live_at_join(reg, target)), key=lambda r: int(r[1:]))
                    # Keep the prefix fast path for its original reset shape. Otherwise vote on the
                    # actual EXEC bits after the compare, including any earlier narrowing.
                    reset = compare - 1
                    while (reset > a_end and ins[reset][3] != 's_mov_b64 exec, -1'
                           and not writes_exec(ins[reset][2], parse(ins[reset][3])[1])
                           and not re.search(r'\b(exec|vcc)', ins[reset][3])):
                        reset -= 1
                    # A branch may land between the reset and the compare only from inside that range, none may
                    # land between the compare and the execz branch.
                    full_reset = ins[reset][3] == 's_mov_b64 exec, -1'
                    entries = [s for s, d in edges.items()
                               if (full_reset and reset < d <= compare and not reset <= s < d)
                               or compare < d <= source]
                    def branch_kinds():
                        """Mnemonic and direction of each inner branch the block cannot take (names only, for the triage)."""
                        kinds = set()
                        for k in range(source + 1, target):
                            if k in edges and k not in inner:
                                where = ('backward' if edges[k] <= k else 'past the block' if edges[k] > target
                                         else 'forward in the block')
                                kinds.add('%s %s' % (ins[k][2], where))
                        return ', '.join(sorted(kinds))
                    causes = [name for name, failed in (
                        ('no compare before the branch', compare <= a_end),
                        ('compare is not a supported unsigned thread comparison', not (prefix or vector_limit)),
                        ('a branch lands between the reset and the branch', bool(entries)),
                        ('a branch from outside lands inside the block',
                         any(source < d < target and not source < s < target for s, d in edges.items())),
                        ('an unsupported branch inside the block (%s)' % branch_kinds(),
                         any(s in edges and s not in inner for s in range(source + 1, target))),
                        ('VCC or SCC live at the join', any(live_at_join(reg, target) for reg in masks)),
                        ('a scalar instruction the wave block cannot run once per wave',
                         any(not mesh_wave_block_scalar(*ins[i][2:]) for i in scalar))) if failed]
                    if causes:
                        raise Unsupported(exchange_branch_refusal if target == count_at
                                          else execz_effects + ' (wave-load block: ' + '; '.join(causes) + ')')
                    inner_vcc.update(vcc_edges)
                    load_compares[compare] = int(prefix[2] if prefix else vector_limit[1])
                    load_branches.setdefault(compare, []).append(source)
                    wave_loads[ins[source][0]] = (ins[compare][0], prefix[1] if prefix and full_reset else None,
                                                  tuple(ins[i][0] for i in loads), tuple(live))
                    continue
            # The model skips per lane, while the guest skips per wave. The difference is invisible when
            # lanes with clear EXEC can only bypass predicated writes before the paths meet. A skipped
            # s_branch may land at the execz target or past it over instructions without effect (the
            # `s_branch .else` over `.empty: s_nop` of an if/else), so both paths meet with the same state.
            for i in range(source + 1, target):
                name = ins[i][2]
                if not (no_effect(i) or name == 'exp'
                        or (name == 's_branch' and edges[i] >= target
                            and all(no_effect(k) for k in range(target, edges[i])))
                        or lane_only(i) or (source == count_entry_branch and dead_count_entry_scalar(i))
                        or dead_block_scalar(i, target)):
                    raise Unsupported(execz_effects)
        elif any(ins[i][2] == 'exp' for i in range(source + 1, target)):
            raise Unsupported('slot 57 mesh: a branch bypasses an export')

    saved_masks = set().union(*(scalar_registers(parse(asm)[1][0]) for _, _, name, asm in ins
                              if name == 's_and_saveexec_b64'))
    out = list(ins[:first])
    span = lambda first, last: (ins[first][0], ins[last][0] + ins[last][1] - ins[first][0])
    # State of the walk over the code outside the two exchanges.
    stale = set(range(256))   # VGPRs the model does not hold for every lane; no entry VGPR is modelled
    pending = set()           # of those, the ones written only for the lanes of a narrowed exec
    mask = None           # None (all lanes), an output count, or another narrowed mask
    wave_known = False     # sreg['wave'] holds s_bfe_u32(s3, 4 bits at 24)
    thread = base = None   # VGPR of the thread index, SGPR holding wave << 6
    thread_high = None    # second instruction of a proven adjacent v_mbcnt pair
    scc_stale = False
    # Possible last full-EXEC compare offsets. An empty set means some path lacks such a definition.
    # The exchanges preserve VCC. Stage.translate checks the sources against its SGPR facts at each definition.
    vcc_compares = frozenset()
    exports = []
    incoming = {}
    falls = True
    def region_walk(lo, hi):
        """Instruction indices in [lo, hi), with each proven constant-trip body repeated and its latch dropped."""
        order, i = [], lo
        while i < hi:
            if i in constant_trips:
                latch, trips = constant_trips[i]
                for _ in range(trips):
                    order.extend(range(i, latch))
                i = latch + 1
                continue
            order.append(i)
            i += 1
        return order

    for index in (region_walk(first, start) + [start] + region_walk(a_end, count_at + 1)
                  + region_walk(b_end, len(ins))):
        if index in targets:
            paths = incoming[index]
            if falls:
                paths.append((stale, pending, mask, wave_known, thread, base, scc_stale, vcc_compares))
            stale = set().union(*(p[0] for p in paths))
            pending = set().union(*(p[1] for p in paths))
            mask = paths[0][2]
            if any(p[2] != mask for p in paths):
                stale |= pending
                pending.clear()
                mask = 'unknown'
            wave_known = all(p[3] for p in paths)
            thread, base = (paths[0][k] if all(p[k] == paths[0][k] for p in paths) else None for k in (4, 5))
            scc_stale = any(p[6] for p in paths)
            definitions = [p[7] for p in paths]
            vcc_compares = frozenset().union(*definitions) if all(definitions) else frozenset()
            falls = True
        if not falls:
            raise Unsupported('slot 57 mesh: unreachable code after a branch is not covered')
        if index == start:
            out.append(span(start, start + 1) + ('x2w_mesh_lane0', 'x2w_mesh_lane0 ' + env['lane']))
            out.append(ins[start + 2])
            out.append(span(start + 3, a_end - 1) + ('s_nop', 's_nop 0'))
            stale |= pending | {5, vreg['a_zero'], vreg['a_zero2']}
            pending.clear()
            mask, thread = 'lane0', None
            wave_known = wave_known and sreg['wave'] != a_bits
            if base == a_bits or base in lane:
                base = None
            continue
        if index == count_at:
            if not wave_known:
                raise Unsupported('slot 57 mesh count exchange: the compared register is not the wave index')
            out.append(span(count_at, b_end - 2) + (
                'x2w_mesh_counts', 'x2w_mesh_counts %s, %s, %s' % (env['prims'], env['verts'], env['b_tmp'])))
            out.append(ins[b_end - 1])
            stale |= pending | {vreg[k] for k in ('b_p', 'b_zero', 'b_v', 'b_zero2', 'b_r0', 'b_r1')}
            pending.clear()
            mask, scc_stale = None, True
            wave_known = wave_known and sreg['wave'] != sreg['b_tmp']
            if base in {sreg['prims'], sreg['verts'], sreg['b_tmp']}:
                base = None
            continue
        off, size, name, asm = ins[index]
        after = index >= b_end
        _, ops, _ = parse(asm)
        base_name = re.sub(r'_e(32|64)$', '', name)
        complement = (name == 's_andn2_b64' and type(mask) is int and len(ops) == 3
                      and ops == ['exec', parse(ins[mask][3])[1][0], 'exec'])
        mask_select = (name == 's_cselect_b32' and len(ops) == 3
                       and (any(tok in MESH_MASK_WORDS for tok in ops)
                            or bool(scalar_registers(', '.join(ops[1:])) & saved_masks))
                       and not scalar_registers(ops[0]) & (saved_masks | lane))
        if (name != 's_and_saveexec_b64' and not complement and index not in nested_restores and not mask_select
                and scalar_registers(asm) & saved_masks):
            raise Unsupported('slot 57 mesh: an unproven use of a saved EXEC mask is not covered')
        if index in edges:
            if scc_stale and reads_scc(name, asm):
                raise Unsupported('slot 57 mesh: an unmodelled SCC is read by a branch')
            if index == shift_branch:
                # Under full EXEC the preceding thread < limit comparison selects a contiguous prefix.
                # A guest wave64 has any active lane exactly when its first thread is below the limit.
                cmp = re.fullmatch(r'v_cmp_gt_u32_e32 vcc_lo, (0x[0-9a-f]+|[0-9]+), v(\d+)', ins[index - 2][3])
                if mask != index - 1 or not cmp or int(cmp[2]) != thread:
                    raise Unsupported('slot 57 mesh: scalar shift needs a proven wave EXEC predicate')
                wave_limits[off] = int(cmp[1], 0)
            if index in inner_vcc:
                wave_votes[off] = ('vcc', name.endswith('z'))
            elif name in MESH_VCC_BRANCHES:
                if mask is not None or not vcc_compares:
                    raise Unsupported('slot 57 mesh: %s on a wave-wide condition is not covered' % name)
                vcc_branches[off] = vcc_compares
            incoming.setdefault(edges[index], []).append(
                (stale.copy(), pending.copy(), mask, wave_known, thread, base, scc_stale, vcc_compares))
            out.append(ins[index])
            falls = name != 's_branch'
            continue
        if name.startswith('ds_') or name in ('s_barrier', 's_sendmsg', 's_sendmsghalt'):
            raise Unsupported('slot 57 mesh: %s outside the group-memory exchanges' % name)
        # Constant-lane v_writelane_b32 / v_readlane_b32 (#2413 T7, #2127): Stage lowers the idiom and
        # refuses ordinary VGPR use, unwritten slots and non-constant lanes. The spill VGPR is not a
        # per-lane value in this model, so it does not update the stale/pending sets below.
        lane_spill = base_name in ('v_writelane_b32', 'v_readlane_b32')
        if name.startswith(('v_readlane', 'v_writelane')) and not lane_spill:
            raise Unsupported('slot 57 mesh: lane spills are not covered')
        keep = ins[index]
        if base_name == 'v_readfirstlane_b32':
            if (a_end <= index < count_at and asm == 'v_readfirstlane_b32 %s, v5' % ops[0]
                    and mask is None and 5 in stale):
                keep = (off, size, 'x2w_mesh_group', 'x2w_mesh_group ' + ops[0])
                ops = ops[:1]
            elif re.fullmatch(r'v_readfirstlane_b32 s\d+, v\d+', asm):
                keep = (off, size, 'x2w_mesh_firstlane', 'x2w_mesh_firstlane ' + ', '.join(ops))
            else:
                raise Unsupported('slot 57 mesh: unsupported v_readfirstlane_b32 operands')
        writes_s = set()
        if not base_name.startswith(('v_cmpx', 'exp', 'ds_write', 'buffer_store')):
            for tok in destinations(name, ops):
                r = reg_range(tok)
                if r and r[0] == 's':
                    writes_s.update(range(r[1], r[2] + 1))
        if 3 in writes_s or writes_s & lane:
            raise Unsupported('slot 57 mesh: the wave word s3 or the saved lane-0 mask is written')
        if (writes_s and name.startswith('v_') and keep[2] not in ('x2w_mesh_group', 'x2w_mesh_firstlane')
                and base_name != 'v_readlane_b32'):
            raise Unsupported('slot 57 mesh: %s into an SGPR is not covered' % base_name)
        # Only the wave index of s3 is modelled.
        if not re.fullmatch(r's_bfe_u32 s\d+, s3, 0x40018', asm) and any(
                r and r[0] == 's' and r[1] <= 3 <= r[2] for r in map(reg_range, ops)):
            raise Unsupported('slot 57 mesh: s3 is read other than as the wave index')
        if after and writes_s & {sreg['prims'], sreg['verts']}:
            raise Unsupported('slot 57 mesh: an output count is written after the count exchange')
        if after and re.search(r'\bm0\b', asm):
            raise Unsupported('slot 57 mesh: m0 is used after the allocation request')
        if scc_stale and reads_scc(name, asm):
            raise Unsupported('slot 57 mesh: SCC of the count exchange is read')
        if name.startswith(('s_cmp', 's_bitcmp')):
            scc_stale = False
        if scalar_effects(name, asm)[2] & VCC_KEYS:
            vcc_compares = frozenset({off}) if (
                mask is None and base_name.startswith('v_cmp_') and name.endswith(('_e32', '_e64'))
                and len(ops) == 3 and ops[0] in ('vcc', 'vcc_lo')
            ) else frozenset()
        # Saveexec uses a fresh VCC from an adjacent compare. Only the matching complement can read the
        # saved mask. Hardware SCC (whether any bit is set) is not represented by a per-lane boolean.
        if asm == 's_mov_b64 exec, -1':
            mask = None
            stale |= pending
            pending.clear()
        elif index in nested_saves and mask == 'wave_load':
            mask, scc_stale = ('nest', index), True
        elif index in nested_restores and mask == ('nest', nested_restores[index]):
            mask = 'wave_load'
        elif (name == 's_and_saveexec_b64' and mask is None and index > 0 and index not in targets
              and re.fullmatch(r's_and_saveexec_b64 s\[\d+:\d+\], vcc', asm)
              and ins[index - 1][2].startswith('v_cmp_')
              and parse(ins[index - 1][3])[1][:1] in (['vcc'], ['vcc_lo'])):
            mask, scc_stale = index, True
        elif complement:
            stale |= pending
            pending.clear()
            mask, scc_stale = ('else', mask), True
        elif index == count_entry:
            if mask is not None or mesh_vregs(ops) & stale:
                raise Unsupported(exchange_branch_refusal)
            mask = 'count_entry'
        elif index in load_compares:
            if thread != load_compares[index] or thread in stale:
                raise Unsupported(execz_effects)
            for source in load_branches[index]:
                branch_off = ins[source][0]
                compare_off, bound, loads, live = wave_loads[branch_off]
                if mask is not None or bound is None:
                    wave_loads[branch_off] = (compare_off, None, loads, live)
                    wave_votes[branch_off] = ('exec', True)
            mask = 'wave_load'
        elif base_name.startswith('v_cmpx'):
            count = {env['verts']: 'verts', env['prims']: 'prims'}.get(ops[0] if ops else None)
            if (base_name != 'v_cmpx_gt_i32' or not after or mask is not None or count is None
                    or thread is None or ops[1:] != ['v%d' % thread]):
                raise Unsupported('slot 57 mesh: an EXEC compare other than output count > thread index under '
                                  'exec = -1 is not covered')
            mask = count
        elif mask_select:
            keep = (off, size, MESH_MASK_SELECT, MESH_MASK_SELECT + ' ' + ', '.join(ops))
            if ops[0] in ('exec_lo', 'exec_hi'):
                # No proof of full EXEC until the next recognized restore.
                stale |= pending
                pending.clear()
                mask = ('select', index)
        elif (not lane_spill and (
                writes_exec(name, ops) or re.search(r'\b(?:exec(?:_lo|_hi)?|src_execz|src_vccz)\b', asm) or (
                    name.startswith('s_') and re.search(r'\bvcc(_lo|_hi)?\b', asm)))):
            raise Unsupported('slot 57 mesh: %s on a wave mask is not covered' % base_name)
        # The thread index: v_mbcnt_lo(-1, wave << 6) then v_mbcnt_hi(-1, that) under exec = -1.
        if asm == 's_bfe_u32 %s, s3, 0x40018' % env['wave']:
            wave_known = True
        elif sreg['wave'] in writes_s:
            wave_known = False
        if wave_known and re.fullmatch(r's_lshl_b32 s\d+, %s, 6' % env['wave'], asm) and ops[0] != env['wave']:
            base = int(ops[0][1:])
        elif base in writes_s:
            base = None
        reads_v = mesh_vregs(ops if base_name.startswith(('v_cmp', 'exp', 'ds_write', 'buffer_store')) else ops[1:])
        dest = reg_range(ops[0]) if ops and not base_name.startswith(('v_cmp', 'exp', 'buffer_store')) else None
        writes_v = set(range(dest[1], dest[2] + 1)) if dest and dest[0] == 'v' else set()
        if base_name == 'v_mbcnt_lo_u32_b32' and base is not None and ops[1:] == ['-1', 's%d' % base] and mask is None \
                and index + 1 not in targets \
                and index + 1 < len(ins) and ins[index + 1][3] == 'v_mbcnt_hi_u32_b32 %s, -1, %s' % (ops[0], ops[0]):
            thread = dest[1]
            thread_high = index + 1
        elif thread in writes_v and index != thread_high:
            raise Unsupported('slot 57 mesh: the thread index register is rewritten')
        if lane_spill:
            reads_v, writes_v = set(), set()
        if keep[2] != 'x2w_mesh_group' and reads_v & stale:
            raise Unsupported('slot 57 mesh: v%d is read where it is not defined for every lane (an entry value, or '
                              'left different per wave by an exchange)' % min(reads_v & stale))
        if mask is not None:
            pending |= writes_v & stale
        else:
            pending -= writes_v
        stale -= writes_v
        if name == 'exp':
            target = asm.split()[1].rstrip(',')
            want = 'prims' if target == 'target_20' else 'verts' if re.fullmatch(r'pos0|param\d+', target) else None
            if want is None:
                raise Unsupported('slot 57 mesh: export target %s is not covered' % target)
            if not after or mask != want:
                raise Unsupported('slot 57 mesh: export %s is not under exec = -1 narrowed by its output count > '
                                  'thread index' % target)
            exports.append(target)
        out.append(keep)
    if exports.count('pos0') != 1 or exports.count('target_20') != 1:
        raise Unsupported('slot 57 mesh: one pos0 and one primitive export are required')
    return out


def adjacent_address_carries(ins, targets):
    """Map each candidate low add to its high-word carry read in the same basic block."""
    pairs = {}
    for index, (high_pc, _, name, asm) in enumerate(ins):
        if name != 's_add_u32' or 'src_scc' not in asm or not index or high_pc in targets:
            continue
        low_pc, _, low_name, low_asm = ins[index - 1]
        if low_name != 's_add_u32':
            continue
        low = parse(low_asm)[1]
        if len(low) == 3 and re.fullmatch(r's\d+', low[0]):
            reg = int(low[0][1:])
            if reg < 105 and parse(asm)[1] == ['s%d' % (reg + 1), '0', 'src_scc']:
                pairs[low_pc] = high_pc
    return pairs


def mrtz_followed_by_vm(ins, at):
    """Find the terminating valid-mask export in the same EXEC region and basic block."""
    targets = {parse(asm)[1][0] for _, _, name, asm in ins
               if name == 's_branch' or name.startswith('s_cbranch')}
    for off, _, name, asm in ins:
        if off <= at:
            continue
        if ('label_0x%x' % off in targets
                or name.startswith(('s_branch', 's_cbranch', 's_setpc', 's_swappc', 's_endpgm', 's_trap'))
                or writes_exec(name, parse(asm)[1])):
            break
        if name == 'exp' and 'done' in asm.split():
            return 'vm' in asm.split()
    return False


def mrtz_covers_every_path(ins, at, *, deferred_vm=False):
    """True when every path from the first instruction to an s_endpgm either runs the depth export at offset `at` or ends in a
    `exp null ... vm` export (the early exit of a kill: it discards a lane that left EXEC), and the export cannot run twice.

    Instruction-level control flow of a decoded pixel program (RDNA 2 ISA, SOPP branches). A taken s_cbranch_execz is the
    way out when no lane is live, which the null export's vm flag turns into a discard; every other conditional branch is
    followed both ways. An indirect jump or a branch that leaves the program makes the answer False."""
    index = {off: i for i, (off, _s, _n, _a) in enumerate(ins)}
    absorbing = {at}
    terminal_nulls = set()
    if deferred_vm:
        terminal_nulls = {off for i, (off, _, name, asm) in enumerate(ins[:-1])
                          if name == 'exp' and asm == 'exp null off, off, off, off done vm'
                          and ins[i + 1][2] == 's_endpgm'}
        # The canonical demote exit proves EXEC zero even on its SCC edge.
        if ins[-1][2] == 's_endpgm' and kill_exit_block(ins):
            absorbing.add(ins[-2][0])
    else:
        absorbing.update(off for off, _s, name, asm in ins
                         if name == 'exp' and re.match(r'exp null\b.*\bvm\b', asm) is not None)
    successors = []
    for i, (_off, _size, name, asm) in enumerate(ins):
        if name == 's_endpgm':
            successors.append([])
        elif name in ('s_setpc_b64', 's_swappc_b64', 's_trap'):
            return False
        elif name == 's_branch' or name.startswith('s_cbranch'):
            target = index.get(int(parse(asm)[1][0].split('_')[1], 16))
            if target is None:
                return False
            if deferred_vm and name != 's_branch' and i + 1 == len(ins):
                return False
            successors.append([target] if name == 's_branch' else [i + 1, target])
        else:
            successors.append([i + 1] if i + 1 < len(ins) else [])
    if at not in index:
        return False
    stack, seen = [(0, False)], set()
    while stack:
        i, zero_exec = stack.pop()
        if (i, zero_exec) in seen:
            continue
        seen.add((i, zero_exec))
        off, _, name, asm = ins[i]
        if off in absorbing or (zero_exec and off in terminal_nulls):
            continue
        if name == 's_endpgm' or (deferred_vm and not successors[i]):
            return False  # a path ends without the depth export and without the discarding exit
        if deferred_vm:
            zero_exec = zero_exec and not writes_exec(name, parse(asm)[1])
            if name in ('s_cbranch_execz', 's_cbranch_execnz'):
                fallthrough, taken = successors[i]
                stack.extend(((fallthrough, name == 's_cbranch_execnz'), (taken, name == 's_cbranch_execz')))
                continue
        stack.extend((n, zero_exec) for n in successors[i])
    # The export must not be inside a loop: it would run once per pass.
    stack, seen = list(successors[index[at]]), set()
    while stack:
        i = stack.pop()
        if i == index[at]:
            return False
        if i not in seen:
            seen.add(i)
            stack += successors[i]
    return True



# Inline float constants of a packed binary16 operation and their binary16 words.
PK_INLINE_F16 = {'0.5': 0x3800, '-0.5': 0xb800, '1.0': 0x3c00, '-1.0': 0xbc00, '2.0': 0x4000, '-2.0': 0xc000, '4.0': 0x4400, '-4.0': 0xc400}

class Stage:
    """Translates one stage. Shares the resource and entry-constant registries of its pipeline."""

    # Immutable fallbacks for unit probes that bypass __init__; __init__ and translate() assign fresh
    # per-instance sets, so nothing is shared across Stage instances.
    needs_sm6 = False         # the lowering uses a shader model 6.0 intrinsic, so the stage cannot compile as 5.1
    exec_known = False        # unit probes without __init__ customize this explicitly when needed
    # Every EXEC write so far was a v_cmpx, which keeps only lanes the mask already had: EXEC is a subset of the entry mask.
    exec_narrow_only = False
    has_image_writes = True   # proved False in translate() only when no image write exists in the dispatch
    loads_after_exec_write = ()  # compute loads reached from an EXEC writer through the block graph
    loads_after_wide_exec_write = ()  # the same for a writer that is not a v_cmpx
    low16_only_v = ()         # VGPRs whose upper 16 bits are undefined after odd-dmask d16 image_load
    pinned_low16_v = ()       # low16_only_v live across basic-block boundaries
    exec_generation = 0      # advances on every EXEC write and basic-block boundary
    low16_generation = None  # first outstanding low-half write, per VGPR
    whole_read_v = ()         # VGPRs read outside a proven low-16-bit consumer
    align4_v = ()             # VGPRs proven to hold a multiple of 4 in every lane
    align4_s = ()             # SGPRs proven to hold a multiple of 4
    used_atomic = ()          # atomic result types ('uint', 'int') needing declarations in main()
    # pixel: VGPR -> (attr, channel, high) of the v_interp_p1ll_f16 that wrote it. Created on first use, because
    # unit probes build a Stage without __init__.
    interp_p1 = None
    # The whole-program proof of interp_f16_scan depends only on the decoded program, so it runs once per Stage
    # (a class default keeps probes built without __init__ working).
    interp_scanned = False
    defined_uniform_s = None
    bad_sgprs = None
    _uniform_written = None
    raw_code = b''
    embedded_buffer = None
    address_carry_pairs = ()
    address_carry_reader = None

    def __init__(self, pipe, kind, code, refusals=None, decoded=None):
        self.pipe, self.kind = pipe, kind
        self.refusals = refusals
        self.raw_code = strip_header(code)
        self.image_proof = None
        self.image_tables = {}
        if kind == 'cs' and decoded is None:
            self.try_image_loads()
        if self.image_proof is None:
            self.ins = (decode(self.raw_code, refusals, embedded_vertex=kind == 'vs')
                        if refusals is None or decoded is None else decoded)
        if kind == 'vs' and any(name == 's_getpc_b64' for _, _, name, _ in self.ins):
            self.embedded_buffer = embedded_buffer_proof(self.raw_code, self.ins)
        if kind == 'ms':
            self.mesh_vcc_branches = {}
            self.mesh_wave_limits = {}
            self.mesh_wave_loads = {}
            self.mesh_wave_votes = {}
            # Keep the pre-skeleton program so spill-only proofs still see the exchange VGPRs
            # (mesh_skeleton rewrites those ops to x2w_mesh_* / s_nop).
            self.mesh_source_ins = list(self.ins)
            self.ins = mesh_skeleton(self.ins, self.mesh_vcc_branches, self.mesh_wave_limits,
                                     self.mesh_wave_loads, self.mesh_wave_votes)
            self.mesh_ballot_regs = set().union(*(scalar_registers(parse(asm)[1][0])
                                                  for _, _, name, asm in self.ins if name == MESH_MASK_SELECT))
        if kind in ('vs', 'ps'):
            self.ins = wave32_vcc_hi(self.ins)
        elif kind == 'cs' and getattr(getattr(self, 'pipe', None), 'wave_size', None) != 64:
            # A compute program also gets the rename, unless the caller says it is wave64. A program proven wave32 (the
            # caller says so, or its exec writes are all 32-bit forms) gets it even where a VALU instruction reads vcc_lo
            # as a lane mask.
            pipe = getattr(self, 'pipe', None)
            self.ins = wave32_vcc_hi(self.ins, wave32=getattr(pipe, 'wave_size', None) == 32 or wave32_exec_proof(self.ins))
        if kind == 'ps':
            self.derive_packed_position()
        # A loop header is translated before the back edges into it, so whether a back edge brings the symbol kept
        # there is known only afterwards: translate again with that register read at run time or refused there. A
        # register only moves from kept to 'run' to 'refuse', so this ends.
        self.retry_wave_reduction = False
        self.force_joined = {}    # branch target -> register -> 'run' | 'refuse'
        first_refusal = len(refusals) if refusals is not None else 0
        while True:
            self.lines = []           # body statements (per block)
            self.sym = {}             # scalar register -> symbolic value
            self.used_s, self.used_v = set(), set()
            self.used_atomic = set()
            self.exec_known = True    # exec still equals the entry mask (the lane is active)
            self.exec_narrow_only = True
            self.wqm_exec = False     # exec is proven whole-quad: an s_wqm of exec into itself ran while exec_known
            self.exports = {}         # 'pos0' / 'paramN' / 'mrtN' -> [4 expressions]
            self.write_masks = {}     # 'mrtN' -> channels the export writes (bit c = channel c), see export()
            self.attrs = {}           # pixel: attr index -> 'linear' | 'flat' | 'f16'
            self.compr_params = set() # vertex: param indices exported with the compr flag (packed binary16 pairs)
            self.plain_params = {}    # vertex: param index -> channels exported as `off` by its latest plain export
            self.f16_plain = {}       # pixel: attr index -> channels read as binary16 halves of a plain export
            self.uses_pos = False
            self.masks = set()        # scalar registers currently holding lane masks
            self.fetched = False      # vertex: the entry call into the fetch routine was translated as loads
            self.defined_uniform_s = set()
            self.bad_sgprs = set()
            self._uniform_written = set()
            self.setup_entry()
            try:
                self.translate()
            except Unsupported as reason:
                if kind != 'ps' or self.retry_wave_reduction or str(reason) != WAVE_REDUCTION_BRANCHES:
                    raise
                self.retry_wave_reduction = True
                self.force_joined.clear()
                continue
            if (kind == 'ps' and not self.retry_wave_reduction and refusals is not None
                    and (WAVE_REDUCTION_BRANCHES, kind) in refusals[first_refusal:]):
                self.retry_wave_reduction = True
                self.force_joined.clear()
                del refusals[first_refusal:]
                continue
            grown = False
            for t, regs in self.back_force.items():
                have = self.force_joined.setdefault(t, {})
                for reg, kind in regs.items():
                    if have.get(reg) not in (kind, 'refuse'):
                        have[reg], grown = kind, True
            if not grown:
                break
            if refusals is not None:
                del refusals[first_refusal:]

    # ---- entry state ---------------------------------------------------------------------
    def setup_entry(self):
        root = self.pipe.root
        # Graphics root parameters start at s8. For compute, s4 is an inference, not a documented layout: the only
        # in-repo evidence is a local compute-record scan whose table bindings used s4 or s5 (docs/experiments.md).
        # s0-s3 are not modelled.
        reg = 4 if self.kind == 'cs' else 8
        for index, p in enumerate(root['params']):
            if p['type'] == 0:
                self.sym[reg] = ('ptr', ('table', index))
                reg += 1
            elif p['type'] == 1:
                for i in range(p['count']):
                    self.sym[reg] = ('ud', self.pipe.ud(('rootconst', index, i)))
                    reg += 1
            else:
                self.sym[reg] = ('ptr', ('root', index))
                self.sym[reg + 1] = ('ptrhi', ('root', index))
                reg += 2
        if root['flags'] & 0x400:
            self.sym[reg] = ('ptr', ('resheap',))
            reg += 1
        if root['flags'] & 0x800:
            self.sym[reg] = ('ptr', ('smpheap',))
            reg += 1
        if self.kind == 'cs':
            self.setup_compute_entry(reg)
        elif self.kind == 'ms':
            # The wave index is bits 24..27 of s3 (mesh_skeleton requires the program to read it that way).
            self.used_s.add(3)
            self.entry = ['s3 = (tid >> 6) << 24;']
            for n in range(105):
                if n != 3 and n not in self.sym:
                    self.sym[n] = ('an entry value the mesh model does not define',)
        elif self.kind == 'vs':
            self.sym[0] = ('ptr', ('vbtab',))
            self.sym[1] = ('ptrhi', ('vbtab',))
            self.sym[2] = ('ud', self.pipe.ud(('basevertex',)))
            self.sym[3] = ('ud', self.pipe.ud(('startinstance',)))
        else:
            self.sym[reg] = ('const', 0)  # primitive mask: only ever moved to m0

    def derive_packed_position(self):
        """Without a published SPI_PS_INPUT_ENA / ADDR (a request with no .psin), take the packed pixel position from the
        program's own register use: the one live-in VGPR that is split into both 16-bit halves (tools/startup/
        ps_input_scan.py: read before any write, low half and high half used) is the POS_FIXED_PT input, which the entry
        code of the declarations fills from SV_Position. Several such registers, or none, leave the pipeline as it was."""
        pipe = getattr(self, 'pipe', None)
        if pipe is None or getattr(pipe, 'ps_inputs_known', True) or getattr(pipe, 'packed_position_vgpr', 0) is not None:
            return
        try:
            from ps_input_scan import contract_of_decoded
            halves = sorted(contract_of_decoded(self.ins).split16)
        except Exception:  # noqa: BLE001  (an undecodable stream keeps the old behaviour)
            return
        if len(halves) == 1:
            pipe.packed_position_vgpr = halves[0]
            pipe.packed_position_derived = True

    def setup_compute_entry(self, root_end):
        """Entry registers of a compute wave as compute_store_aot.entry_registers (#1608) gives them: the user SGPRs,
        then one SGPR per enabled group-ID axis (SV_GroupID), optional TG_SIZE when TG_SIZE_EN is set (only accepted
        for at most 32 threads per group, #2274), and v0..v2 holding SV_GroupThreadID. Where the root parameters sit
        inside the user SGPRs (from s4) is this translator's inference. User SGPRs the root does not place (s0-s3 and
        any past the root's span) have no known value, so any numeric read of them is refused."""
        user, groups, threads = self.pipe.entry
        if root_end > user:
            raise Unsupported('root signature needs s4-s%d, COMPUTE_PGM_RSRC2 gives %d user SGPRs' % (root_end - 1, user))
        for n in list(range(4)) + list(range(root_end, user)):
            self.sym[n] = ('unknown user sgpr',)
        self.entry = []
        self.entry_uniform_s = set()
        for axis, n in sorted(groups.items()):
            self.used_s.add(n)
            self.entry_uniform_s.add(n)
            self.entry.append('s%d = gid.%s;' % (n, 'xyz'[axis]))
        tg_size = getattr(self.pipe, 'tg_size_sgpr', None)
        if tg_size is not None:
            if self.pipe.group[0] * self.pipe.group[1] * self.pipe.group[2] > 32 and self.pipe.wave_size != 32 \
                    and not wave32_exec_proof(self.ins):
                raise Unsupported(TG_SIZE_WAVES_REFUSAL)
            self.sym[tg_size] = ('tg_size',)
        for i in range(threads):
            self.used_v.add(i)
            self.entry.append('v%d = tid.%s;' % (i, 'xyz'[i]))

    # ---- operand helpers ---------------------------------------------------------------------
    def S(self, n):
        """Runtime expression of scalar register n (uint)."""
        s = self.sym.get(n)
        if s:
            if s[0] == 'const':
                return '0x%08xu' % (s[1] & 0xffffffff)
            if s[0] == 'ud':
                return 'ud%d' % s[1]
            if s[0] in ('execcopy', 'fixed_buffer_word'):
                self.used_s.add(n)
                return 's%d' % n
            if s[0] == 'descnum' and len(s) == 3:
                return self.descriptor_number(self.descriptor_number_node(s))
            if s[0] == 'tg_size':
                raise Unsupported('TG_SIZE register s%d read outside s_and_b32 with 0xfc0' % n)
            raise Unsupported('numeric use of s%d holding %s' % (n, s[0]))
        self.used_s.add(n)
        return 's%d' % n

    @staticmethod
    def odd_d16_read_error(reg, asm):
        return Unsupported('whole-word read of v%d after odd d16 image_load: upper half used by %s' % (reg, asm))

    @staticmethod
    def descriptor_number_node(symbol):
        kind, node = symbol[1:]
        if kind in ('stride', 'stride_min1'):
            stride = ('alu', '({0} >> 16) & 0x3fffu', (node[:4],))
            return ('alu', 'max({0}, 1u)', (stride,)) if kind == 'stride_min1' else stride
        return node

    def descriptor_number(self, node, budget=None, *, fixed_only=False):
        """Materialize an immutable descriptor expression only when a numeric consumer needs it."""
        if budget is None:
            budget = [128]
        budget[0] -= 1
        if budget[0] < 0:
            raise Unsupported('descriptor numeric expression exceeds 128 operations and operands')
        kind, *args = node
        if kind == 'alu':
            template, operands = args
            return '(' + template.format(*(self.descriptor_number(word, budget, fixed_only=fixed_only) for word in operands)) + ')'
        if kind == 'value':
            return args[0]
        if kind == 'desc':
            source, offset, word = args
            if fixed_only and node in getattr(self, 'advanced_descriptor_words', ()):
                raise Unsupported('cannot materialize descriptor word %d from an advanced descriptor' % word)
            if (getattr(self, 'kind', None) in ('ps', 'vs') and source[0] == 'table' and 0 <= source[1] < 16
                    and offset is not None and offset % 4 == 0 and 0 <= offset + 4 * word <= 0x7ffffffc):
                return 'ud%d' % self.pipe.ud(('descword', source[1], offset + 4 * word))
            if getattr(self, 'kind', None) == 'cs' and source[0] == 'table' and offset is not None and word % 4 == 3:
                key = source, offset + 4 * (word - 3)
                self.pipe.resource(key, 'buf')
                return 'ud%d' % self.pipe.ud(('descword3', self.srv_index(key)))
            raise Unsupported('cannot materialize descriptor word %d from %s at %s' % (word, source, offset))
        raise Unsupported('cannot materialize descriptor number from %s' % args[0])

    def descriptor_operand(self, token):
        """Read a fixed descriptor word at an explicitly supported numeric use."""
        reg = reg_range(token)
        symbol = self.sym.get(reg[1]) if reg and reg[0] == 's' else None
        if symbol and symbol[0] == 'desc' and names_one_value(symbol):
            if getattr(self, 'kind', None) in ('ps', 'vs'):
                return self.descriptor_number(symbol[:4], fixed_only=True)
        if symbol and symbol[0] == 'descnum' and len(symbol) == 3:
            return self.descriptor_number(self.descriptor_number_node(symbol), fixed_only=True)
        return self.u(token)

    def mark_descriptor_scc(self):
        if hasattr(self, 'descriptor_scc_writes'):
            self.descriptor_scc_writes.add(self.pc)

    def descriptor_alu(self, template, sources, symbols):
        operands = []
        for token, symbol in zip(sources, symbols):
            if symbol and symbol[0] == 'desc' and names_one_value(symbol):
                node = symbol[:4]
            elif symbol and symbol[0] == 'descnum' and len(symbol) == 3:
                node = self.descriptor_number_node(symbol)
            elif (symbol and symbol[0] in ('const', 'ud')) or self.const_u32(token) is not None:
                node = ('value', self.u(token))
            else:
                node = ('unavailable', '%s holding %s' % (token, symbol))
            operands.append(node)
        return ('descnum', 'alu', ('alu', template, tuple(operands)))

    def note_whole_read(self, reg):
        asm = getattr(self, 'current_asm', '<unknown instruction>')
        if reg in self.low16_only_v:
            raise self.odd_d16_read_error(reg, asm)
        if isinstance(self.whole_read_v, set):
            self.whole_read_v.add(reg)
            sites = getattr(self, 'whole_read_sites', None)
            if sites is not None:
                sites.setdefault(reg, (self.pc, asm))

    def scc_refusal(self, text, path):
        """`text` plus the path that refused and the instruction's opcode and destination class, so a census of one SCC
        refusal says which emitting path and which instruction shape dominate. The refusal itself is unchanged."""
        parts = getattr(self, 'current_asm', '').split(None, 1)
        opcode = parts[0] if parts else 'unknown'
        first = parts[1].split(',')[0].strip() if len(parts) > 1 else ''
        reg = reg_range(first) if first else None
        if first in ('vcc', 'vcc_lo', 'vcc_hi', 'exec', 'exec_lo', 'exec_hi', 'm0', 'null'):
            dest = first
        elif reg and reg[0] in ('s', 'v'):
            dest = {'s': 'sgpr', 'v': 'vgpr'}[reg[0]]
        else:
            dest = 'none'
        return '%s [%s: %s, destination %s]' % (text, path, opcode, dest)

    def u(self, tok, *, low16_only=False):
        """uint expression for a 32-bit operand."""
        if tok == 'src_scc':
            # Source code 253 reads SCC as 0 or 1 (fits in 16 bits, so low16_only=True needs no mask).
            if not getattr(self, 'scc_verified', False):
                reason = self.scc_refusal('src_scc with unverified SCC', 'operand read')
                pending = getattr(self, 'scc_data_reads', None)
                if pending is None or self.kind not in ('ps', 'vs', 'cs'):
                    raise Unsupported(reason)
                pending[self.pc] = reason
            return '(scc ? 1u : 0u)'
        if tok in ('vcc_lo', 'vcc'):
            self.read_vcc()
            return 'vcc'
        if tok in ('exec_lo', 'exec'):
            return 'exec'
        if tok == 'm0':
            m = self.sym.get('m0')
            if m:
                raise Unsupported('numeric use of m0 holding %s' % m[0])
            return 'm0'
        if tok == 'null':
            return '0x00000000u'
        r = reg_range(tok)
        if r:
            if r[0] == 'v':
                for reg in range(r[1], r[2] + 1):
                    self.used_v.add(reg)
                    if not low16_only:
                        self.note_whole_read(reg)
                return 'v%d' % r[1]
            return self.S(r[1])
        if tok in FLOATS:
            return '0x%08xu' % fbits(FLOATS[tok])
        try:
            return '0x%08xu' % (int(tok, 0) & 0xffffffff)
        except ValueError:
            raise Unsupported('operand %r' % tok)

    def is_lane_mask(self, tok):
        reg = reg_range(tok)
        return (tok in ('exec', 'exec_lo', 'exec_hi', 'vcc', 'vcc_lo', 'vcc_hi')
                or bool(reg and reg[0] == 's' and reg[1] in getattr(self, 'masks', ())))

    @staticmethod
    def split_neg(tok):
        """(negated, operand) for the `-x` sign modifier and the `neg(x)` spelling of it."""
        # LLVM prints a lone negation of a numeric operand as neg(c), so that `-1` is not read as the
        # integer: it is the sign bit flipped, never an integer negation.
        m = re.fullmatch(r'neg\((.+)\)', tok)
        if m:
            return True, m.group(1)
        if tok.startswith('-') and not re.match(r'-[\d.]', tok):
            return True, tok[1:]
        return False, tok

    def reject_lane_mask_data(self, op, tok):
        """The lane bit cannot represent a mask's word when it is read as data."""
        source = tok
        while True:
            neg, operand = self.split_neg(source)
            if neg:
                source = operand
            elif source.startswith('|') and source.endswith('|') and len(source) >= 2:
                source = source[1:-1]
            elif source.startswith(('abs(', 'sext(')) and source.endswith(')'):
                source = source[source.index('(') + 1:-1]
            else:
                break
        if self.is_lane_mask(source):
            raise Unsupported('%s data operand %s is a lane mask' % (op, tok))

    def u_src(self, tok):
        """uint expression for an integer-context source that the disassembler may print with abs/neg."""
        # In RDNA2 VOP3, `abs` clears bit 31 first and `-` / `neg(...)` flips bit 31 second (so `-|x|`
        # forces bit 31 to 1, while `-x` flips bit 31). Only handlers that model them ask for this;
        # other integer sources keep using u(), so an unexpected modifier stays refused.
        neg, tok = self.split_neg(tok)
        absolute = tok.startswith('|') and tok.endswith('|')
        if absolute:
            tok = tok[1:-1]
        expr = self.u(tok)
        if absolute:
            expr = '(%s & 0x7fffffffu)' % expr
        if neg:
            expr = '(%s ^ 0x80000000u)' % expr
        return expr

    def f(self, tok):
        """float expression, with the inline neg/abs operand modifiers."""
        neg, tok = self.split_neg(tok)
        absolute = tok.startswith('|') and tok.endswith('|')
        if absolute:
            tok = tok[1:-1]
        if tok in FLOATS:
            text = repr(FLOATS[tok])
        elif tok == '-0':
            text = '0.0'
        else:
            text = 'F(%s)' % self.u(tok)
        if absolute:
            text = 'abs(%s)' % text
        return '(-%s)' % text if neg else text

    def is_const(self, n):
        s = self.sym.get(n)
        return s[1] if s and s[0] == 'const' else None

    def const_u32(self, tok):
        if tok == 'null':
            return 0
        r = reg_range(tok)
        if r:
            if r[0] == 's' and r[1] == r[2]:
                c = self.is_const(r[1])
                return None if c is None else (c & 0xffffffff)
            return None
        if tok in FLOATS:
            return fbits(FLOATS[tok])
        try:
            return int(tok, 0) & 0xffffffff
        except ValueError:
            return None

    def operand_align4(self, tok):
        """True when 32-bit operand tok is proven to be a multiple of 4 in every lane."""
        if self.is_lane_mask(tok):
            return False
        c = self.const_u32(tok)
        if c is not None:
            return (c & 3) == 0
        r = reg_range(tok)
        if not r or r[1] != r[2]:
            return False
        if r[0] == 'v':
            return r[1] in self.align4_v and r[1] not in self.low16_only_v
        if r[0] == 's':
            return r[1] in self.align4_s and self.sym.get(r[1]) is None
        return False

    def valu_align4(self, base, srcs):
        """True when a modifier-free VEC_U operation produces a multiple of 4 from srcs."""
        if base in ('v_add_nc_u32', 'v_sub_nc_u32', 'v_subrev_nc_u32', 'v_or_b32', 'v_xor_b32',
                    'v_min_u32', 'v_max_u32', 'v_min_i32', 'v_max_i32',
                    'v_add3_u32', 'v_or3_b32', 'v_xor3_b32', 'v_xad_u32',
                    'v_min3_u32', 'v_max3_u32', 'v_med3_u32', 'v_min3_i32', 'v_max3_i32', 'v_med3_i32'):
            return all(self.operand_align4(s) for s in srcs)
        if base in ('v_and_b32', 'v_mul_lo_u32', 'v_mul_u32_u24', 'v_mul_i32_i24'):
            return any(self.operand_align4(s) for s in srcs)
        if base == 'v_lshlrev_b32':
            return self.operand_align4(srcs[1]) or ((self.const_u32(srcs[0]) or 0) & 31) >= 2
        if base in ('v_lshl_add_u32', 'v_lshl_or_b32'):
            return (self.operand_align4(srcs[0]) or ((self.const_u32(srcs[1]) or 0) & 31) >= 2) and self.operand_align4(srcs[2])
        if base == 'v_add_lshl_u32':
            return ((self.const_u32(srcs[2]) or 0) & 31) >= 2 or (self.operand_align4(srcs[0]) and self.operand_align4(srcs[1]))
        if base in ('v_and_or_b32', 'v_mad_u32_u24', 'v_mad_i32_i24', 'v_mad_u32_u16'):
            return (self.operand_align4(srcs[0]) or self.operand_align4(srcs[1])) and self.operand_align4(srcs[2])
        return False

    def salu_align4(self, base, srcs):
        """True when a 32-bit non-mask Stage.scalar operation produces a multiple of 4 from srcs."""
        if base == 's_mov_b32':
            return self.operand_align4(srcs[0])
        if base == 's_lshl_b32':
            return self.operand_align4(srcs[0]) or ((self.const_u32(srcs[1]) or 0) & 31) >= 2
        if base in ('s_and_b32', 's_mul_i32'):
            return any(self.operand_align4(s) for s in srcs)
        if base == 's_andn2_b32':
            return self.operand_align4(srcs[0]) or ((self.const_u32(srcs[1]) or 0) & 3) == 3
        if base in ('s_add_u32', 's_sub_u32', 's_add_i32', 's_sub_i32', 's_or_b32', 's_xor_b32',
                    's_cselect_b32', 's_min_u32', 's_max_u32', 's_min_i32', 's_max_i32', 's_lshl1_add_u32'):
            return all(self.operand_align4(s) for s in srcs)
        if base in ('s_lshl2_add_u32', 's_lshl3_add_u32', 's_lshl4_add_u32'):
            return self.operand_align4(srcs[1])
        return False

    def read_vcc(self):
        if getattr(self, 'has_branches', False):
            self.vcc_events.setdefault(self.pc, []).append(('read_vcc', None))
        elif getattr(self, 'vcc_lost', False):
            raise Unsupported('vcc read after a descriptor load into s106/s107')

    def mark_vcc_descriptor(self, extended=False):
        self.vcc_lost = True
        self.vcc_constant_pair = False
        self.vcc_descriptor_extended = extended
        self.vcc_descriptor_overwritten = False
        if getattr(self, 'has_branches', False):
            value = tuple(self.sym[i] for i in (106, 107))
            self.vcc_events.setdefault(self.pc, []).append(('load_descriptor', value))

    def emit(self, text):
        if 'vcc' in text:
            assignment = re.match(r'vcc = (.*);$', text)
            rhs = assignment.group(1) if assignment else text
            if re.search(r'\bvcc\b', rhs):
                self.read_vcc()
            if assignment:
                if getattr(self, 'has_branches', False):
                    self.vcc_events.setdefault(self.pc, []).append(('write_vcc', None))
                if getattr(self, 'vcc_constant_pair', False):
                    self.vcc_constant_pair = False
                    if not getattr(self, 'has_branches', False):
                        self.sym.pop(106, None)
                        self.sym.pop(107, None)
                if getattr(self, 'vcc_lost', False):
                    # The scalar symbols are provisional across branches, checked after translation.
                    self.vcc_lost = False
                    self.vcc_descriptor_overwritten = True
                    if not getattr(self, 'has_branches', False):
                        self.sym.pop(106, None)
                        self.sym.pop(107, None)
        self.lines.append(text)

    def divergent_exit_loop(self, at):
        """True when the offset sits in a loop (a backward branch of any kind, also an unconditional one) that holds a
        wave-divergent branch: lanes of one quad can leave through the forward exit at different passes, so the next pass
        runs the sample on uneven quads."""
        for _o, _n, name, asm in self.ins:
            if name == 's_branch' or name.startswith('s_cbranch_'):
                target = int(parse(asm)[1][0].split('_')[1], 16)
                if target <= _o and target <= at <= _o and any(target <= d <= _o for d in self.divergent_branches):
                    return True
        return False

    def resolve_plain_samples(self):
        """Writes each plain image_sample past a join into the blocks: the implicit-LOD Sample unless a divergent back
        edge spans it (the same proof as image_sample_b / image_get_lod), else the level-0 call as before.
        Biased samples require the implicit path."""
        if not self.plain_samples:
            return
        choice = {}
        for at, (level0, implicit) in self.plain_samples.items():
            spanned = any(target <= at <= branch for target, branch in self.divergent_back_edges) or self.divergent_exit_loop(at)
            if spanned and level0 is None:
                raise Unsupported('image_sample with a sampler LOD bias and no implicit LOD')
            choice['X2W_PLAIN_SAMPLE<%x>' % at] = level0 if spanned else implicit

        def text(line):
            for token, call in choice.items():
                line = line.replace(token, call)
            return line
        self.blocks = [(start, [text(l) for l in lines], term) for start, lines, term in self.blocks]

    def check_vcc_paths(self):
        """An overlapping descriptor must reach its use on every path, and VCC must be numeric at a mask read."""
        if not self.has_branches or not any(event in ('load_descriptor', 'read_null_descriptor')
                                           for events in self.vcc_events.values()
                                           for event, _ in events):
            return
        offsets = [off for off, _, _ in self.blocks]
        index = {off: i for i, off in enumerate(offsets)}
        by_block = [[] for _ in offsets]
        block = 0
        for off, events in sorted(self.vcc_events.items()):
            while block + 1 < len(offsets) and offsets[block + 1] <= off:
                block += 1
            by_block[block].extend(events)
        todo = [(0, None)]
        seen = set()
        while todo:
            block, state = todo.pop()
            if (block, state) in seen:
                continue
            seen.add((block, state))
            for event, value in by_block[block]:
                if event == 'read_vcc' and state is not None and state[0] == 'load_descriptor':
                    raise Unsupported('vcc read after a descriptor load into s106/s107')
                if event in ('read_descriptor', 'read_null_descriptor'):
                    if state is None or any(state[1][reg] != symbol for reg, symbol in value):
                        raise Unsupported('descriptor operand overlaps s106/s107 without a matching load on every path')
                if event in ('load_descriptor', 'load_constant_descriptor'):
                    state = (event, value)
                elif event == 'write_vcc':
                    state = None
            term = self.blocks[block][2]
            if term and term[0] in ('jump', 'branch'):
                if term[-1] in index:
                    todo.append((index[term[-1]], state))
            if (not term or term[0] == 'branch') and block + 1 < len(offsets):
                todo.append((block + 1, state))

    def clear_low16(self, n):
        if not isinstance(self.low16_only_v, set) or n in self.pinned_low16_v:
            return
        generation = (self.low16_generation or {}).get(n)
        # The entry-mask rule still applies; an unchanged mask generation now also qualifies.
        if self.exec_known or generation == self.exec_generation:
            self.low16_only_v.discard(n)
            if generation is not None:
                del self.low16_generation[n]

    def set_v(self, n, expr, *, low16_only=False, align4=False, ignore_exec=False):
        if getattr(self, 'interp_p1', None):
            self.interp_p1.pop(n, None)
        if hasattr(self, 'uniform_v'):
            self.uniform_v.discard(n)
        if getattr(self, 'firstlane_cfg', False):
            self.firstlane_v.discard(n)
        if isinstance(self.align4_v, set):
            if align4 and not low16_only and (self.exec_known or n in self.align4_v):
                self.align4_v.add(n)
            else:
                self.align4_v.discard(n)
        self.last_v = (n, expr)
        if low16_only:
            if not isinstance(self.low16_only_v, set):
                self.low16_only_v = set()
            self.low16_only_v.add(n)
            if self.low16_generation is None:
                self.low16_generation = {}
            # A later predicated low-half write may leave earlier undefined lanes intact.
            self.low16_generation.setdefault(n, self.exec_generation)
        else:
            self.clear_low16(n)
        self.used_v.add(n)
        self.emit('v%d = %s;' % (n, expr) if self.exec_known or ignore_exec else 'if (exec & 1) v%d = %s;' % (n, expr))
        temps = getattr(self, 'uniform_temps', None)
        if temps is not None:  # a uniform load past the entry block (one()): every lane keeps the words
            temps[n] = 'ul%x_%d' % (self.pc, n)
            self.uniform_load_temps.add(temps[n])
            self.emit('%s = %s;' % (temps[n], expr))

    def f_mods(self, expr, mods=()):
        # The hardware applies the output modifier before the clamp, but the disassembler prints `clamp`
        # first, so the order of `mods` is not the order of the operations: scale first, saturate last.
        scale = 1.0
        clamp = False
        for m in mods:
            if m == 'clamp':
                clamp = True
            elif m in ('mul:2', 'mul:4'):
                scale *= float(m[-1])
            elif m == 'div:2':
                scale *= 0.5
            elif m.startswith('op_sel'):
                # op_sel selects which half of a packed source the operation reads. This path does not
                # model that, so dropping it would change the operand: refuse instead of ignoring it.
                raise Unsupported('modifier %s' % m)
            else:
                raise Unsupported('modifier %s' % m)
        if scale != 1.0:
            factor = ('%d.0' % int(scale)) if scale == int(scale) else ('%g' % scale)
            expr = '(%s) * %s' % (expr, factor)
        if clamp:
            expr = 'saturate(%s)' % expr
        return expr

    def set_vf(self, n, expr, mods=()):
        self.set_v(n, 'U(%s)' % self.f_mods(expr, mods))

    def scalar_expr_uniform(self, expr, *, assume_scc=False):
        # These identifiers are runtime inputs in the scalar expressions emitted below.
        # In particular, a compare can carry a lane mask into an otherwise scalar select.
        for token in re.findall(r'\b(?:s\d+|v\d+|vcc|exec|m0|scc)\b', expr):
            if token == 'scc':
                if not (assume_scc or self.scc_uniform):
                    return False
            elif not self.uniform_source(token):
                return False
        return True

    def set_scc(self, expr, proof=None):
        """Emit an SCC assignment and track its uniformity from `proof` (an expression over the
        sources) when the emitted text names a register, such as m0, that the proof cannot judge."""
        self.scc_effect = 'emitted'
        self.scc_verified = True
        if self.tracks_uniformity():
            source = expr if proof is None else proof
            self.scc_uniform = self.scalar_expr_uniform(source)
            self.scc_from_uniform_input = not self.scc_uniform and self.scalar_expr_uniform(source, assume_scc=True)
        self.emit('scc = %s;' % expr)

    def record_scalar_uniformity(self, n, uniform):
        if uniform:
            self.varying_s.discard(n)
            if self.defined_uniform_s is not None:
                self.defined_uniform_s.add(n)
            if self._uniform_written is not None:
                self._uniform_written.add(n)
        else:
            self.varying_s.add(n)
            if self.defined_uniform_s is not None:
                self.defined_uniform_s.discard(n)
            if self._uniform_written is not None:
                self._uniform_written.discard(n)
            if self.bad_sgprs is not None:
                self.bad_sgprs.add(n)
        # Only a proved assignment can add an SGPR to since_join. Unmodelled writes drop any stale mark.
        if getattr(self, 'since_join', None) is not None:
            (self.since_join.add if uniform else self.since_join.discard)(n)

    def set_s(self, n, expr, sym=None, scc=None, uniform=None, align4=False):
        """Assign scalar register n at run time and/or symbolically.

        Any write ends n's life as a lane mask; a caller that writes a mask marks n again after this call.
        The uniformity of expr is judged first, while a source that is n itself still has its old mark."""
        scc_dependency = False
        if self.tracks_uniformity() and (sym is None or sym[0] == 'const'):
            if uniform is None:
                uniform = self.scalar_expr_uniform(expr)
                if scc and not uniform:
                    scc_dependency = self.scalar_expr_uniform(expr, assume_scc=True)
            self.record_scalar_uniformity(n, uniform)
        getattr(self, 'masks', set()).discard(n)  # unit probes build a Stage without __init__
        getattr(self, 'mask_high', set()).discard(n)
        getattr(self, 'float_const_s', {}).pop(n, None)
        if isinstance(self.align4_s, set):
            if sym is not None and sym[0] == 'const':
                (self.align4_s.add if (sym[1] & 3) == 0 else self.align4_s.discard)(n)
            elif sym is None and align4:
                self.align4_s.add(n)
            else:
                self.align4_s.discard(n)
        if sym is not None:
            self.sym[n] = sym
            if sym[0] != 'const':
                return
        else:
            self.sym.pop(n, None)
        self.used_s.add(n)
        self.emit('s%d = %s;' % (n, expr))
        if scc:
            self.set_scc(scc.replace('$', 's%d' % n))
            if scc_dependency:
                self.scc_from_uniform_input = True

    def descriptor(self, tok, *, allow_null=True):
        """Return (source, byte offset) for a resource operand, or None for a proven null resource."""
        r = reg_range(tok)
        s = self.sym.get(r[1]) if r else None
        if r and r[0] == 's' and r[1] <= 107 and r[2] >= 106:
            if getattr(self, 'has_branches', False):
                expected = tuple((i - 106, self.sym.get(i)) for i in range(max(106, r[1]), min(107, r[2]) + 1))
                event = 'read_null_descriptor' if allow_null and self.null_descriptor(tok) else 'read_descriptor'
                self.vcc_events.setdefault(self.pc, []).append((event, expected))
            elif getattr(self, 'vcc_descriptor_overwritten', False):
                raise Unsupported('descriptor operand %s overlaps a redefined vcc' % tok)
        if r == ('s', 104, 107) and s and s[0] == 'desc' and (
                s[3] % 4 or
                any(self.sym.get(104 + i) != desc_word(s, s[3] + i) for i in range(1, 4))):
            raise Unsupported('descriptor operand %s needs four consecutive words of the same descriptor' % tok)
        if r == ('s', 100, 107) and getattr(self, 'vcc_descriptor_extended', False) and s and s[0] == 'desc':
            for i in (6, 7):
                word = self.sym.get(100 + i)
                if not word or (word[0] != 'const' and word != desc_word(s, s[3] + i)):
                    raise Unsupported('descriptor operand %s needs matching descriptor words in vcc' % tok)
        if s and s[0] == 'const' and r and r[0] == 's':
            if allow_null and self.null_descriptor(tok):
                return None
            value = next((self.is_const(i) for i in range(r[1], r[2] + 1) if self.is_const(i)), s[1])
            raise Unsupported("descriptor operand %s holds ('const', %#x)" % (tok, value & 0xffffffff))
        if s and s[0] == 'desc':
            if s[1][0] == 'dynamic_heap' and (r[0] != 's' or r[2] - r[1] != 3 or s[3] != 0 or
                    any(self.sym.get(r[1] + i) != desc_word(s, i) for i in range(1, 4))):
                raise Unsupported('dynamic resource heap descriptor requires four intact words')
            # A descriptor can start inside a wider load (s_load_dwordx16 fetches two image
            # descriptors): its offset is the load's offset plus the first register's dword index.
            return s[1], (None if s[2] is None else s[2] + 4 * s[3])
        if s and s[0] == 'ptr':
            if len(s) > 2 and s[2]:
                raise Unsupported('descriptor operand %s is a pointer with offset %s' % (tok, hex(s[2]) if isinstance(s[2], int) else s[2]))
            return s[1], None
        raise Unsupported('descriptor operand %s holds %s' % (tok, s))

    def null_descriptor(self, tok):
        r = reg_range(tok)
        return (r and r[0] == 's' and r[2] - r[1] in (3, 7)
                and all(self.is_const(i) == 0 for i in range(r[1], r[2] + 1)))

    def heap_sampler(self, tok):
        """'qh[...]' when the four sampler words of tok came from one run-time sampler heap load."""
        r = reg_range(tok)
        if not r or r[2] - r[1] != 3:
            return None
        first = self.sym.get(r[1])
        if not first or first[0] != 'smpheap':
            return None
        if any(self.sym.get(r[1] + i) != ('smpheap', first[1], i) for i in range(4)):
            raise Unsupported('sampler operand %s mixes sampler heap words' % tok)
        self.pipe.uses_sampler_heap = True
        # Sampler descriptors are 16 bytes; x2w_sampler_entry() maps the byte offset to its entry and
        # sends unaligned or out-of-range offsets to the fallback entry.
        return 'qh[x2w_sampler_entry(s%d)]' % first[1]

    def const_words(self, tok):
        r = reg_range(tok)
        fconst = getattr(self, 'float_const_s', {})
        words = [self.is_const(n) if self.is_const(n) is not None else fconst.get(n)
                 for n in range(r[1], r[2] + 1)]
        return None if any(w is None for w in words) else words

    def execz_counted_loop(self, guard, latch, into, effects, live_in):
        """Recognize a finite scalar loop entered at its head or, with EXEC empty, via its latch."""
        head, end = guard + 1, latch + 1
        if guard == 0 or latch < head + 4 or end >= len(self.ins):
            return None
        if self.ins[latch][2] != 's_branch' or into.get(head) != [latch] or into.get(latch) != [guard]:
            return None
        if any(i in into for i in range(head + 1, latch)):
            return None
        # The initialization must reach both paths. No instruction in the loop can change EXEC
        # or observe other lanes, so running it with empty EXEC affects only the counter and SCC.
        if guard in into or self.ins[guard - 1][2] != 's_mov_b32':
            return None
        _, init, mods = parse(self.ins[guard - 1][3])
        if len(init) != 2 or mods:
            return None
        reg, initial = init
        if not re.fullmatch(r's(?:[0-9]|[1-9][0-9]|10[0-5])', reg):
            return None
        if not re.fullmatch(r'0x[0-9a-f]+|[0-9]+', initial):
            return None
        _, _, cmp, asm = self.ins[head]
        _, args, mods = parse(asm)
        if (cmp != 's_cmp_lt_u32' or len(args) != 2 or mods or args[0] != reg
                or not re.fullmatch(r'0x[0-9a-f]+|[0-9]+', args[1])):
            return None
        _, _, branch, asm = self.ins[head + 1]
        if branch != 's_cbranch_scc0' or int(parse(asm)[1][0].split('_')[1], 16) != self.ins[end][0]:
            return None
        _, _, step, asm = self.ins[latch - 1]
        if step != 's_add_u32' or parse(asm)[1:] != ([reg, reg, '1'], []):
            return None
        if {reg, 'scc'} & live_in[end]:
            return None
        for i in range(head + 2, latch - 1):
            name = re.sub(r'_e(32|64)$', '', self.ins[i][2])
            if (name not in ('v_add_nc_u32', 'v_mov_b32', 'v_xor_b32') or effects[i][2]
                    or parse(self.ins[i][3])[2]):
                return None
        return head, end

    def execz_forward_join(self, branch, join, succ, jumps, index):
        """Find a common join for an execz region with an outward s_branch.

        Close the forward interval over its successors. Each path must reach the resulting
        boundary, and an outside edge may enter only at the opening branch or the boundary.
        The ordinary execz proof then checks the enlarged interval, including the path from
        the original join that an inactive invocation can run while the wave jumps over it.
        """
        if not any(branch < src < join and self.ins[src][2] == 's_branch'
                   and index.get(target, -1) > join for src, target in jumps):
            return join
        end = join
        for i in range(branch + 1, len(succ)):
            if i == end:
                break
            if not succ[i] or any(dst <= i for dst in succ[i]):
                return join
            end = max(end, *succ[i])
        if any(branch < index.get(target, -1) < end and not branch <= src < end
               for src, target in jumps):
            return join
        return end

    def execz_refusals(self):
        """{offset: reason} for compute blocks that s_cbranch_execz skips and the one-lane model cannot run.

        Also record which refusals the wave-uniform control-flow path can try to resolve.

        The wave runs such a block when any lane is active, so its scalar results are the same for every lane; the
        model skips the block for an inactive lane. That lane can differ from the wave only by reading, after the
        join, scalar state the block defines (no lane active means the wave skipped the block too). The block is
        therefore accepted unchanged when every scalar register, mask and SCC it may define is dead where it joins,
        and it has no effect outside scalar registers and no branch out of the proven region. An outward s_branch
        can extend the region to a common forward join. The block dispatcher uses each invocation's pc to
        guard the remaining blocks after a jump, including any code between the original join and the new one.
        An exec write is accepted when it
        cannot set the bit of a lane whose bit was clear at the branch: clear_after carries, over every path through
        the block (its branches and loops included), which masks have that lane's bit clear, and the write must
        leave exec among them. Such a lane's bit then stays clear at every instruction the wave runs in the block,
        so the wave changes nothing of that lane there and its exec at the join is the model's. The proof starts at
        the branch, so a block that a branch from outside enters keeps its exec writes refused. A definition live at
        the join is accepted when every read of it after the join is proven (execz_live_read), or when it has the
        shape below. Any other live definition, an unproven exec write, a refused effect, a branch leaving the block
        or a backward s_cbranch_execz is refused.

        The other accepted live definition is the i1-phi lane mask of the compiler's if-conversion, a 32-bit M (vcc_lo
        or one sN) zeroed before the branch (`s_mov_b32 M, 0`) and set in the block from exec only (`s_mov_b32 M,
        exec_lo`, `s_and_b32 M, X, exec_lo` in either operand order). Every write of M in the block yields a value
        whose bit is 0 for any lane whose exec bit is clear there: the clear-mask proof keeps exec clear for that lane, and a copy of exec, or X & exec, has that bit clear. M is 0
        at the branch on every path (nothing branches or lands between the zeroing and the branch). So for a lane
        the model skips, the wave's M has that lane's bit 0 whether the wave ran the block or not, which is the
        model's M (0); a lane that runs the block runs it in the model too. Writes that leave a skipped lane's bit
        as it was are accepted for any M: `s_andn2_b32 M, M, exec_lo` and `s_or_b32 M, M, T` where T is an exec-masked
        value computed in the block (`s_and_b32 T, X, exec_lo` or a v_cmp result), the blend the compiler emits for an
        i1 phi whose other input is a per-lane value. A v_cmp into M (not v_cmpx) needs M zeroed before the branch
        like the moves above, and `s_orn2_b32 M, X, exec_lo` needs M set to all ones (`s_mov_b32 M, -1`). Zeroing
        and all-ones writes do not mix. This holds only while M is read per
        lane. A whole-mask read (s_cbranch_vccz/vccnz, s_cmp, a plain copy, a data operand) sees the other lanes'
        bits, which the one-lane model does not have, so any such read of M reachable after the join, before M is
        rewritten, keeps the refusal. The reads accepted are the lane select of v_cndmask, the carry-in of v_addc /
        v_subb / v_add_co_ci style ops, and s_and_saveexec / s_and_b32 with exec."""
        offsets = [ins[0] for ins in self.ins]
        index = {off: i for i, off in enumerate(offsets)}
        effects = [scalar_effects(name, asm) for _, _, name, asm in self.ins]
        succ = []
        jumps = []  # (instruction, target) of every branch, the proven calculated s_setpc_b64 ones included
        for i, (off, n, name, asm) in enumerate(self.ins):
            to = int(parse(asm)[1][0].split('_')[1], 16) if name.startswith(('s_cbranch', 's_branch')) else None
            if name == 's_setpc_b64':  # a proven calculated jump reaches its target
                to = getattr(self, 'calculated_pc', {}).get(off)
            nxt = [i + 1] if i + 1 < len(self.ins) and name not in ('s_branch', 's_endpgm') else []
            if to is not None:
                jumps.append((i, to))
            succ.append(nxt + ([index[to]] if to in index else []))
        live_in = [set() for _ in self.ins]
        changed = True
        while changed:
            changed = False
            for i in reversed(range(len(self.ins))):
                out = set().union(*(live_in[s] for s in succ[i])) if succ[i] else set()
                reads, writes, _ = effects[i]
                value = reads | (out - writes)
                if value != live_in[i]:
                    live_in[i], changed = value, True
        self.waterfall_reads = {}
        self.waterfall_scalar_loads = {}
        for close, target in jumps:
            head = index.get(target)
            if (head is not None and head + 3 < close and self.ins[close][2] == 's_cbranch_scc1'
                    and self.ins[head][2] == 'v_readfirstlane_b32'
                    and self.ins[close - 1][2] == 's_andn2_wrexec_b32'):
                self.waterfall_reads[target] = self.waterfall_refusal(head, close, effects, live_in, jumps)
        self.execz_counted_loops = {}
        refused, into = {}, {}
        self.execz_scalar_votes = {}
        for k, target in jumps:
            into.setdefault(index.get(target), []).append(k)
        for b, (off, n, name, asm) in enumerate(self.ins):
            if name != 's_cbranch_execz':
                continue
            t = index.get(int(parse(asm)[1][0].split('_')[1], 16))
            if t is None or t <= b:
                refused.setdefault(off, 's_cbranch_execz to a block before it')
                continue
            if loop := self.execz_counted_loop(b, t, into, effects, live_in):
                head, end = loop
                self.execz_counted_loops[offsets[t]] = (offsets[head], offsets[head + 2], offsets[end])
                continue
            original_join = t
            t = self.execz_forward_join(b, t, succ, jumps, index)
            live = live_in[t] | {'m0'}  # m0 is also read implicitly; treat it as live
            # Facts before each instruction the wave can reach from the branch: the largest sets every path keeps.
            entry = frozenset({('exec', 0), ('exec_hi', 1)})
            before = {b + 1: entry} if b + 1 < t else {}
            if original_join < t:
                # The modeled inactive lane enters here directly. Facts from the body
                # cannot establish the values of any other masks on that edge.
                before[original_join] = entry
            entered = any(index.get(target) in range(b + 1, t) for k, target in jumps if not b < k < t)
            changed = not entered
            while changed:
                changed = False
                for i in range(b + 1, t):
                    if i in before:
                        after = frozenset(clear_after(*self.ins[i][2:], effects[i][2], before[i])) | entry
                        for s in succ[i]:
                            if b < s < t and before.get(s, after) & after != before.get(s):
                                before[s], changed = before.get(s, after) & after, True
            targets = [target for _, target in jumps]
            straight = not any(self.ins[k][2].startswith(('s_cbranch', 's_branch')) for k in range(b + 1, t)) and \
                not any(index.get(target) in range(b + 1, t) for target in targets)
            cand = {k for i in range(b + 1, t) for k in (effects[i][2] - EXEC_KEYS) & live if re.fullmatch(r'vcc|s\d+', k)}
            ok_masks = {k for k in cand if straight and self.phi_mask_ok(k, b, t, effects, succ, targets, index)}
            # A live definition is still accepted when every read of it from the join on is proven (execz_live_read):
            # the halves whose lanes keep their bit of the word through the block, and where exec is unchanged since
            # the join on every path from it.
            kept = {}
            for i in range(b + 1, t):
                for word in (effects[i][2] - EXEC_KEYS) & live:
                    if kept.get(word, True):  # an empty set stays empty
                        andn2 = i in before and not entered and self.ins[i][2].startswith('s_andn2')
                        halves = kept_halves(*self.ins[i][2:], word, before[i]) if andn2 else set()
                        kept[word] = kept.get(word, halves) & halves
            def quiet(j):  # exec unchanged since the join on every path to j: a straight run from t, entered at t only
                return t <= j and all(b <= k < t for k in into.get(t, ())) and not any(
                    effects[k][2] & EXEC_KEYS for k in range(t, j)) and not any(k in into for k in range(t + 1, j + 1))
            unread = {}
            for i in range(b + 1, t):
                at, _, inner, inner_asm = self.ins[i]
                may = effects[i][2]
                reason = None
                vote_eligible = False
                if inner.startswith(EXECZ_REFUSED):
                    reason = 'effect outside scalar registers'
                elif inner.startswith(('s_cbranch', 's_branch')) and not (
                        b < (exit_index := index.get(int(parse(inner_asm)[1][0].split('_')[1], 16), -1)) <= t):
                    reason = 'branch out of the block'
                    # A lane can leave while the wave takes the latch, or take the latch after the wave exits.
                    if inner == 's_cbranch_scc0' and any(
                            t <= latch < exit_index and self.ins[latch][2] == 's_cbranch_scc0'
                            and target in index and index[target] <= b for latch, target in jumps):
                        reason = 'loop exit crosses a backward s_cbranch_scc0'
                    elif inner == 's_cbranch_scc0' and any(
                            i < latch < exit_index and self.ins[latch][2] == 's_branch'
                            and target in index and index[target] <= i for latch, target in jumps):
                        reason = 'loop exit crosses a backward s_branch'
                elif may & EXEC_KEYS and (entered or i in before and
                                          not entry <= clear_after(inner, inner_asm, may, before[i])):
                    reason = 'exec write'
                elif words := [word for word in sorted((may - EXEC_KEYS - ok_masks) & live)
                               if self.execz_live_read(t, word, kept[word], effects, succ, quiet, unread) is not None]:
                    # Reserve the fallback for ordinary scalar writes. The vote keeps every lane on the same
                    # control-flow edge, so the existing runtime join selects every word (and SCC) together.
                    dst = parse(inner_asm)[1][:1]
                    vote_eligible = (straight and inner in EXECZ_SCALAR_JOIN and bool(dst)
                                     and re.fullmatch(r's(?:\d+|\[\d+:\d+\])', dst[0]) is not None
                                     and set(words) <= scalar_tokens(dst[0]) | (effects[i][1] & {'scc'}))
                    named = [unread[word] for word in words if unread[word] >= 0]
                    first = min(named) if named else None
                    word_of = next((w for w in words if unread[w] == first), None)
                    reason = 'definition of %s, live where the block joins' % ', '.join(words) + (
                        ': read by %s after the join' % self.ins[first][2] if named else '')
                    if named:
                        reason += ' (%s)' % self.execz_read_cause(first, word_of, b, t, effects, into)
                if reason:
                    message = '%s inside a block s_cbranch_execz skips per lane: %s' % (inner, reason)
                    refused.setdefault(at, message)
                    self.execz_scalar_votes[at] = self.execz_scalar_votes.get(at, True) and vote_eligible
        return refused

    def execz_read_cause(self, j, word, b, t, effects, into):
        """Why the read at instruction j of `word` after the join t of an execz block is not proven (execz_live_read),
        as a short phrase for the refusal text: the condition that failed first, in the order the proof checks them."""
        name, asm = self.ins[j][2:]
        if not exec_absorbs(name, asm, word):
            lanes = mask_lanes(name, asm)
            if not lanes:
                return 'cause: a data read or a copy, not a lane-mask operation'
            if lanes[0] not in ('and', 'nand', 'orn2'):
                return 'cause: lane-mask operation %s, not and/nand/orn2 with exec' % lanes[0]
            if high_half_of_b32(lanes, word):
                return 'cause: a 32-bit read of the high half'
            return 'cause: the word is not paired with the exec word of the same lanes'
        if j < t:
            return 'cause: read before the join, reached again through a loop'
        if not all(b <= k < t for k in into.get(t, ())):
            return 'cause: another path enters the join'
        if any(effects[k][2] & EXEC_KEYS for k in range(t, j)):
            return 'cause: exec is written between the join and the read (restored or changed)'
        if any(k in into for k in range(t + 1, j + 1)):
            return 'cause: a branch target (loop or join) lies between the join and the read'
        return 'cause: other'

    def execz_live_read(self, t, word, halves, effects, succ, quiet, unread):
        """None when every read of `word` from the join t of an execz block on, until a write of it ends the block's
        definition, gives a lane the model skipped the wave's result; else the first read that is not proven, or -1
        for m0 and SCC, which have implicit readers.

        Such a lane's exec bit is clear at the join, in the model and in the wave (execz_refusals proves the block's
        exec writes), and the two differ there at most in `word`. A read is proven when:
        - exec is unchanged on every path from the join to it (`quiet`: the read ends a straight run that starts at
          the join, which only the block enters, with no exec write in it) and it reads `word` only as W & exec,
          ~(W & exec) or W | ~exec, exec of the same lanes (exec_absorbs): the lane's bit of the result is fixed
          whatever its bit of `word` is, so the result agrees, and when it overwrites `word` the definition ends;
        - the block keeps the lane's bit of `word` for its half (kept_halves: every write of `word` in the block is
          s_andn2 R, R, X with X clear for the lane, checked over every path from the branch) and the read is a lane
          mask read of that half (lane_read): the lane's bit of `word` is the model's, so its bit of the result is
          too, and `word` written back that way keeps the property.
        Any other read, a data read or a copy into another register included, is not proven. The search stops at
        the first such read, nearest the join; the result is kept in `unread` for the block."""
        if word in unread:
            return unread[word]
        if not re.fullmatch(r'vcc(_hi)?|s\d+', word):
            unread[word] = -1
            return -1
        seen, work = {t}, [t]
        for j in work:  # breadth first: the list grows while it is read
            name, asm = self.ins[j][2:]
            kept_read = word in effects[j][0] and halves and lane_read(name, asm, word, halves)
            if word in effects[j][0] and not kept_read and not (exec_absorbs(name, asm, word) and quiet(j)):
                unread[word] = j
                return j
            if kept_read or word not in effects[j][1]:
                for s in succ[j]:
                    if s not in seen:
                        seen.add(s)
                        work.append(s)
        unread[word] = None
        return None

    def waterfall_refusal(self, head, close, effects, live_in, jumps):
        """Return None when the lane can pick its own key at the head of the waterfall.

        CMPX selects all remaining lanes with the first lane's key. WREXEC removes those lanes,
        so each incoming lane participates exactly once. A lane-local body computes its value on
        that visit. Other visits can only affect it through shared state or an escaping scalar.
        """
        ins = self.ins
        _, key_ops, mods = parse(ins[head][3])
        if (mods or len(key_ops) != 2 or not re.fullmatch(r's\d+', key_ops[0])
                or not re.fullmatch(r'v\d+', key_ops[1])):
            return 'unsupported key operands'
        key = key_ops[0]
        name, ops, mods = parse(ins[head + 1][3])
        if re.sub(r'_e(32|64)$', '', name) != 'v_cmpx_eq_i32' or mods or ops != key_ops:
            return 'the next instruction does not select lanes with the same key'
        _, ops, mods = parse(ins[close - 1][3])
        if mods or len(ops) != 2 or ops[0] != ops[1] or not re.fullmatch(r's\d+', ops[0]) or ops[0] == key:
            return 'the closing mask is not a distinct remaining-lanes register'
        remaining = ops[0]
        # The b32 close covers only the low 32 lanes when the hardware uses wave64.
        x, y, z = self.pipe.group
        if x * y * z > 32:
            return 'b32 remaining mask requires at most 32 threads per group'
        copy = None
        for i in reversed(range(head)):
            _, _, op, asm = ins[i]
            if op.startswith(('s_branch', 's_cbranch', 's_setpc', 's_endpgm', 's_movrel')):
                break
            if remaining in effects[i][2] or effects[i][2] & EXEC_KEYS:
                if op == 's_mov_b32' and parse(asm)[1] == [remaining, 'exec_lo']:
                    copy = i
                break
        if copy is None:
            return 'remaining mask is not an unchanged copy of EXEC'
        if any(ins[copy][0] < target <= ins[close][0] for branch, target in jumps if branch != close):
            return 'another branch enters the saved mask or loop'
        # Only ordinary lane-local operations and reads of a bound raw buffer. An instruction
        # that observes the other lanes sharing this key cannot use this scalarization (e.g. mbcnt of EXEC).
        body_writes = {key, 'scc'}  # WREXEC supplies SCC for the next iteration.
        numeric_writes = {key}
        descriptor_inputs, buffer_descriptors = set(), set()
        scalar_loads = set()
        reads_before_write = set()
        defined = {key}
        keyed = {key}  # scalars whose value depends on the key (so on the iteration) after the instructions so far
        raw_load_widths = {'buffer_load_dword': 1, 'buffer_load_dwordx2': 2,
                           'buffer_load_dwordx3': 3, 'buffer_load_dwordx4': 4}
        for j in range(head + 2, close - 1):
            base, ops, mods = parse(ins[j][3])
            base = re.sub(r'_e(32|64)$', '', base)
            width = raw_load_widths.get(base)
            # RDNA 2 SOP2 field extracts and integer min/max have the same scalar/SCC footprint
            # across signedness. VOP bitwise operations write only the executing lane's VGPR.
            if not width and base not in (
                    's_add_u32', 's_mov_b32', 's_and_b32', 's_or_b32', 's_lshl_b32', 's_waitcnt',
                    's_bfe_u32', 's_bfe_i32', 's_max_u32', 's_max_i32', 's_min_u32', 's_min_i32', 's_mul_i32',
                    'v_mov_b32', 'v_xor_b32', 'v_and_b32', 'v_or_b32', 'v_not_b32', 'v_xnor_b32', 'v_add_nc_u32',
                    's_load_dwordx4', 's_load_dwordx8', 's_buffer_load_dwordx2'):
                return 'body instruction %s is not proven lane-local' % base
            if mods and not (width and 'offen' in mods and all(
                    m == 'offen' or re.fullmatch(r'offset:(?:0x[0-9a-f]+|\d+)', m) for m in mods)):
                return 'body instruction %s has unproved modifiers' % base
            reads, writes, may_write = effects[j]
            if width:
                dst = reg_range(ops[0]) if ops else None
                if not dst or dst[0] != 'v' or dst[2] != dst[1] + width - 1 or dst[2] > 255:
                    return 'buffer load has an invalid vector destination'
                # Every destination belongs to this lane, including those read after EXEC is restored.
                # Keep them in the body write set but out of the escaping scalar dependency set.
                body_writes.update('v%d' % n for n in range(dst[1], dst[1] + width))
            if base == 's_bfe_i32':
                writes = writes | {'scc'}
            if (reads | may_write) & (EXEC_KEYS | {'vcc', 'vcc_hi', 'm0', remaining}):
                return 'body reads or changes a lane mask or special register'
            if base in ('s_load_dwordx8', 's_buffer_load_dwordx2'):
                scalar_loads.add(ins[j][0])
            if base in ('s_load_dwordx4', 's_load_dwordx8'):
                dst = reg_range(ops[0]) if len(ops) == 3 else None
                if not dst or dst[0] != 's' or dst[2] > 105:
                    return 'descriptor load overlaps a special register'
                descriptor_inputs.update(reads)
            else:
                numeric_writes.update(may_write)
            if width or base == 's_buffer_load_dwordx2':
                descriptor_index = 1 if base.startswith('s_buffer') else 2
                if len(ops) <= descriptor_index:
                    return 'buffer load has no descriptor'
                buffer_descriptors.update(scalar_tokens(ops[descriptor_index]))
            reads_before_write.update(reads - defined)
            defined.update(writes)
            body_writes.update(may_write)
            if (reads & keyed) or width or base == 's_buffer_load_dwordx2':
                keyed.update(may_write)
            else:
                keyed -= writes
        if reads_before_write & body_writes:
            return 'body reads loop-carried scalar state'
        # The data byte offset can use the key. Loading or modifying the resource descriptor
        # with the key needs a separate proof that the runtime binds the whole range.
        if descriptor_inputs & body_writes or buffer_descriptors & numeric_writes:
            return 'per-lane descriptor selection requires a proven bound heap range'
        # The close leaves EXEC and SCC zero. The last physical pass's other scalar
        # values can differ per lane in this model and must be dead on every successor.
        # A scalar written from loop-invariant inputs only (no read of the key or of a key-dependent scalar, no
        # load through a body address) holds the same value in every iteration, so the last pass's value is each
        # lane's own: it may be live. The key and anything computed from it may not.
        if close + 1 < len(ins) and (body_writes - {'scc'}) & live_in[close + 1] & (keyed | {key}):
            return 'body scalar is live at the loop exit'
        if any(name.startswith('s_movrel') for _, _, name, _ in ins):
            return 'relative scalar indexing can observe a loop register'
        self.waterfall_scalar_loads[ins[head][0]] = scalar_loads
        return None

    def phi_mask_ok(self, key, b, t, effects, succ, targets, index):
        """True when lane mask `key` ('vcc' for vcc_lo, or 'sN'), live at the join t of the s_cbranch_execz at b, leaves the
        bit of every lane the model skips as the wave has it (see execz_refusals), and after the join it is only read per
        lane. A lane the model skips has its exec bit clear for the whole block (the exec proof), and the wave runs the block
        iff some lane is active, so the model is exact iff the block never changes that lane's bit of M. Every write of M
        in the block is one of (mask_write_class), a 32-bit M written alone. Every mask it uses must hold lane bits
        (lane_bit_keys, or a block temp): the model keeps a lane's bit at bit 0 where the wave has it at bit `lane`:
          'keep'  s_andn2 M, M, exec_lo; s_or M, M, T with T written in the block by s_and with exec_lo or a v_cmp
                  (bit 0 for a lane whose exec bit is clear): the lane's bit is unchanged, whatever M held before.
          'zero'  s_mov M, exec_lo; s_and M, X, exec_lo; a v_cmp into M (VOPC writes 0 for an inactive lane): bit 0,
                  exact only when M was zero before the branch: the last write before it is `s_mov_b32 M, 0`.
          'ones'  s_orn2 M, X, exec_lo: bit 1, exact only when M was all ones before it: `s_mov_b32 M, -1`.
        Zeroing or setting M before the branch needs no branch and no branch target after it up to the branch; 'zero' and
        'ones' writes do not mix. Any other write (M reused as an offset) is accepted only if a later 'zero' or 'ones'
        write overwrites it, since a 'keep' write would carry its bits for a skipped lane. The caller has checked that the block is straight-line."""
        name = 'vcc_lo' if key == 'vcc' else key
        classes, temps, stale = set(), set(), False
        enc = self.lane_bit_keys(b, effects, targets, index)
        for i in range(b + 1, t):
            wrote = effects[i][2]
            _, ops, _ = parse(self.ins[i][3])
            base = re.sub(r'_e(32|64)$', '', self.ins[i][2])
            if key in wrote:
                if not ops or ops[0] != name:
                    return False
                kind = self.mask_write_class(base, ops, name, temps, enc)
                if kind is None:
                    stale = True   # data in M (an s_movk used as an offset): only a later full overwrite makes it exact
                else:
                    classes.add(kind)
                    stale = stale and kind == 'keep'
            for w in wrote - {key} - EXEC_KEYS:
                # T must also hold lane bits: an s_and of exec with a lane mask (not with a group id or other data)
                lane_src = base != 's_and_b32' or any(
                    ('vcc' if x == 'vcc_lo' else x) in temps | enc for x in ops[1:] if x != 'exec_lo')
                if self.exec_masked_def(base, ops, w) and lane_src:
                    temps.add(w)
                else:
                    temps.discard(w)
                enc.discard(w)
        if stale or ('zero' in classes and 'ones' in classes):
            return False
        if 'keep' in classes and key not in self.lane_bit_keys(b, effects, targets, index):
            return False   # M's bits before the block must already be lane bits (bit 0 = this lane), not data
        need = {'zero': '0', 'ones': '-1'}.get(next(iter(classes & {'zero', 'ones'}), None))
        if need is not None:
            # The setting: the last instruction before the branch that may write M.
            z = next((k for k in range(b - 1, -1, -1) if key in effects[k][2]), None)
            if z is None:
                return False
            _, ops, _ = parse(self.ins[z][3])
            if self.ins[z][2] not in ('s_mov_b32', 's_movk_i32') or ops != [name, need]:
                return False
            if any(self.ins[k][2].startswith(('s_cbranch', 's_branch', 's_setpc')) for k in range(z + 1, b)) or \
                    any(index.get(x) in range(z + 1, b + 1) for x in targets):
                return False
        # After the join, until M is rewritten, every read must be per lane.
        seen, todo = set(), [t]
        while todo:
            i = todo.pop()
            if i in seen:
                continue
            seen.add(i)
            reads, writes, _ = effects[i]
            if key in reads and not self.per_lane_read(i, key):
                return False
            if key not in writes:
                todo.extend(succ[i])
        return True

    def lane_bit_keys(self, b, effects, targets, index):
        """Keys ('vcc' or 'sN') whose value at the s_cbranch_execz at b holds the lane's own bit as bit 0, the encoding the
        one-lane model gives a lane mask: the last write before b is a v_cmp, an s_and with exec_lo, a copy of exec_lo, or
        0 / -1 (every bit alike), with no branch or branch target between it and b. A value from anywhere else (a group id,
        an offset, `s_mov_b32 M, 16`) has the wave's lane bit at bit `lane`, which the model reads from bit 0."""
        entries = {index.get(target) for target in targets}
        start = next((k + 1 for k in range(b - 1, -1, -1)
                      if self.ins[k][2].startswith(('s_cbranch', 's_branch', 's_setpc'))
                      or k + 1 in entries), 0)
        found = set()
        for k in range(start, b):
            _, ops, _ = parse(self.ins[k][3])
            base = re.sub(r'_e(32|64)$', '', self.ins[k][2])
            writes = effects[k][2] - EXEC_KEYS
            proven = set()
            for w in writes:
                name = 'vcc_lo' if w == 'vcc' else w
                masked = self.exec_masked_def(base, ops, w)
                if masked and base == 's_and_b32':
                    other = ops[2] if ops[1] == 'exec_lo' else ops[1]
                    key = 'vcc' if other == 'vcc_lo' else other
                    masked = key in found or other in ('0', '-1', 'exec_lo')
                if masked or (base in ('s_mov_b32', 's_movk_i32') and len(ops) == 2
                              and ops[0] == name and ops[1] in ('0', '-1', 'exec_lo')):
                    proven.add(w)
            found.difference_update(writes)
            found.update(proven)
        return found

    @staticmethod
    def exec_masked_def(base, ops, key):
        """True when this instruction writes `key` ('sN' or 'vcc') with a value whose bit is 0 for a lane whose exec bit is
        clear: s_and with exec_lo into it, or a v_cmp (not v_cmpx) into it."""
        name = 'vcc_lo' if key == 'vcc' else key
        if not ops or ops[0] != name:
            return False
        if base == 's_and_b32':
            return len(ops) == 3 and 'exec_lo' in ops[1:]
        return base.startswith('v_cmp') and not base.startswith('v_cmpx')

    @staticmethod
    def mask_write_class(base, ops, name, temps, enc=frozenset()):
        """'keep', 'zero' or 'ones' for a write of mask `name` inside an execz block (see phi_mask_ok), else None."""
        key = 'vcc' if name == 'vcc_lo' else name
        if base == 's_mov_b32' and ops[1:] == ['exec_lo']:
            return 'zero'
        lane = lambda x: ('vcc' if x == 'vcc_lo' else x) in temps | enc   # a lane mask: bit 0 is this lane's bit
        if base == 's_and_b32' and len(ops) == 3 and 'exec_lo' in ops[1:] and lane(ops[2] if ops[1] == 'exec_lo' else ops[1]):
            return 'zero'
        if base.startswith('v_cmp') and not base.startswith('v_cmpx'):
            return 'zero'
        if base == 's_orn2_b32' and len(ops) == 3 and ops[2] == 'exec_lo' and lane(ops[1]):
            return 'ones'
        if base == 's_andn2_b32' and ops == [name, name, 'exec_lo']:
            return 'keep'
        if base == 's_or_b32' and len(ops) == 3 and name in ops[1:]:
            other = ops[2] if ops[1] == name else ops[1]
            if other != name and ('vcc' if other == 'vcc_lo' else other) in temps:
                return 'keep'
        return None

    def per_lane_read(self, i, key):
        """True when instruction i reads mask `key` only as a lane select or carry, or into exec (s_and_saveexec,
        s_and with exec): the lane's own bit is all it uses."""
        _, _, name, asm = self.ins[i]
        base = re.sub(r'_e(32|64)$', '', name)
        _, ops, _ = parse(asm)
        if base.startswith(('v_cndmask', 'v_addc', 'v_subb', 'v_add_co_ci', 'v_sub_co_ci', 'v_subrev_co_ci')):
            # the mask is the last operand and appears nowhere else (not as a data word, not as the destination)
            return len(ops) >= 4 and key in scalar_tokens(ops[-1]) and \
                all(key not in scalar_tokens(op) for op in ops[:-1])
        if base == 's_and_saveexec_b32':
            return len(ops) == 2 and key in scalar_tokens(ops[1]) and key not in scalar_tokens(ops[0])
        if base == 's_and_b32':
            return len(ops) == 3 and key not in scalar_tokens(ops[0]) and any(
                is_exec(scalar_tokens(op)) for op in ops[1:])
        return False

    # ---- instruction translation -------------------------------------------------------------
    def joined_registers(self, with_cfg=False):
        """{branch target: registers whose program-order symbol may not hold on every path into it}.

        Program order carries the state after every instruction above the target into it. That state holds on every
        path only for a register whose writes above the target are all in blocks that dominate it (in dominance
        order), and which no block that reaches it from at or below it writes (a loop body, or a later block that
        branches back). Computed over the basic blocks, so a path that leaves a loop's textual range and comes back
        counts.

        With ``with_cfg=True``, returns ``(joined, forward_splits, div_targets)`` where ``forward_splits`` keeps only
        writes on non-dominating forward arms between a target's dominators and the target (excluding loop back edges)
        and ``div_targets`` maps each ``s_cbranch_*`` offset to ``(writes, affected_targets)`` inside its divergent
        CFG region (between the branch and its immediate post-dominator, or over the loop range for a backward edge).
        A fourth item, ``(bounds, block_of, succ, div_blocks)``, is the block graph and that region's blocks per
        ``s_cbranch_*`` offset, for the spill-slot proof (spill_reaching)."""
        ins = self.ins
        if not ins:
            return ({}, {}, {}, ([], {}, [], {})) if with_cfg else {}
        index = {off: i for i, (off, *_) in enumerate(ins)}
        calculated = getattr(self, 'calculated_pc', {})  # unit probes build a Stage without translate()
        branch = lambda name: name.startswith('s_cbranch') or name in ('s_branch', 's_setpc_b64')
        target_of = lambda off, name, asm: (calculated.get(off) if name == 's_setpc_b64'  # None: no proven target
                                            else int(parse(asm)[1][0].split('_')[1], 16))
        leaders = {0} | {target_of(off, name, asm) for off, _, name, asm in ins if branch(name)} - {None}
        for i, (off, n, name, asm) in enumerate(ins[:-1]):
            if branch(name) or name == 's_endpgm':
                leaders.add(ins[i + 1][0])
        starts = sorted(index[o] for o in leaders if o in index)
        bounds = list(zip(starts, starts[1:] + [len(ins)]))
        off_of = [ins[lo][0] for lo, _ in bounds]
        block_of = {}
        for b, (lo, hi) in enumerate(bounds):
            for i in range(lo, hi):
                block_of[i] = b
        succ = []
        for lo, hi in bounds:
            off, n, name, asm = ins[hi - 1]
            out = [] if name in ('s_branch', 's_setpc_b64', 's_endpgm') or hi == len(ins) else [block_of[hi]]
            if branch(name) and target_of(off, name, asm) in index:
                out.append(block_of[index[target_of(off, name, asm)]])
            succ.append(out)
        pred = [[a for a in range(len(bounds)) if b in succ[a]] for b in range(len(bounds))]
        writes = [set().union(*(scalar_destinations(*ins[i][2:]) for i in range(lo, hi))) for lo, hi in bounds]
        everything = set(range(106)) | {'m0'}
        writes = [everything if 'any' in w else w for w in writes]
        # Dominators of each block (iterative; the entry block is block 0).
        dom = [set(range(len(bounds))) for _ in bounds]
        dom[0] = {0}
        changed = True
        while changed:
            changed = False
            for b in range(1, len(bounds)):
                new = ({b} | set.intersection(*(dom[p] for p in pred[b]))) if pred[b] else {b}
                if new != dom[b]:
                    dom[b], changed = new, True
        all_targets = leaders - {0} | ({0} if any(0 in s for s in succ) else set())
        joined, forward_splits = {}, {}
        for target in all_targets:
            if target not in index:
                continue
            t = block_of[index[target]]
            reach, todo = set(), list(pred[t])  # blocks with a path of length >= 1 into the target
            while todo:
                b = todo.pop()
                if b not in reach:
                    reach.add(b)
                    todo += pred[b]
            regs = set()
            above_dominators = sorted(b for b in dom[t] - {t} if b < t)
            for b in range(len(bounds)):
                if b < t and b not in dom[t]:
                    regs |= writes[b]  # above the target but not on every path into it
                if b in reach and b >= t:
                    regs |= writes[b]  # a loop body or a later block that comes back
            # Writes in dominating blocks hold on every path only if their textual order is their execution order.
            non_chain = any(not (a in dom[c]) for a, c in zip(above_dominators, above_dominators[1:]))
            if non_chain:
                regs |= set().union(*(writes[b] for b in above_dominators))
            joined[target] = regs
            if with_cfg:
                above_set = dom[t] - {t}
                split_blks, stodo = set(), [p for p in pred[t] if p < t and p not in above_set]
                while stodo:
                    b = stodo.pop()
                    if b not in split_blks:
                        split_blks.add(b)
                        stodo += [p for p in pred[b] if p < t and p not in above_set]
                fwd = set().union(*(writes[b] for b in split_blks)) if split_blks else set()
                if non_chain:
                    fwd |= set().union(*(writes[b] for b in above_dominators))
                forward_splits[target] = fwd
        if not with_cfg:
            return joined
        pdom = [set(range(len(bounds))) if succ[b] else {b} for b in range(len(bounds))]
        changed = True
        while changed:
            changed = False
            for b in range(len(bounds) - 1, -1, -1):
                if not succ[b]:
                    continue
                new = {b} | set.intersection(*(pdom[s] for s in succ[b]))
                if new != pdom[b]:
                    pdom[b], changed = new, True
        div_targets, div_blocks = {}, {}
        for b, (lo, hi) in enumerate(bounds):
            off, _n, name, asm = ins[hi - 1]
            if not name.startswith('s_cbranch'):
                continue
            tgt = target_of(off, name, asm)
            if tgt is None:
                continue
            if tgt <= off:
                t_blk = block_of[index[tgt]] if tgt in index else b
                loop_blks = range(min(t_blk, b), b + 1)
                regs = set().union(*(writes[w] for w in loop_blks))
                affected = {off_of[w] for w in loop_blks} & all_targets
                div_blocks[off] = set(loop_blks)
            else:
                stop = pdom[b] - {b}
                ipd = max(stop, key=lambda x: len(pdom[x])) if stop else None
                div, dtodo = set(), list(succ[b])
                while dtodo:
                    w = dtodo.pop()
                    if w == ipd or w in div:
                        continue
                    div.add(w)
                    dtodo += succ[w]
                regs = set().union(*(writes[w] for w in div)) if div else set()
                affected = ({off_of[w] for w in div} | ({off_of[ipd]} if ipd is not None else set())) & all_targets
                div_blocks[off] = div
            div_targets[off] = (regs, affected)
        self.flow = (index, block_of, succ, dom, writes, {})
        return joined, forward_splits, div_targets, (bounds, block_of, succ, div_blocks)

    def unjoined(self, target, last_write, regs):
        """The registers of regs whose program-order state is not proven at a branch target. A lane-mask mark or a
        constant follows program order, so it holds on every path into the target only if the block of the register's
        last write dominates the target and no block on a path from that write to the target writes the register
        again."""
        index, block_of, succ, dom, writes, reached = self.flow
        def after(block):  # blocks reached from block over at least one edge
            if block not in reached:
                seen, todo = set(), list(succ[block])
                while todo:
                    b = todo.pop()
                    if b not in seen:
                        seen.add(b)
                        todo += succ[b]
                reached[block] = seen
            return reached[block]
        t = block_of[index[target]]
        into = {b for b in range(len(succ)) if t in after(b)}
        out = set()
        for reg in regs:
            w = block_of[index[last_write[reg]]] if reg in last_write else 0
            if w not in dom[t] or any(reg in writes[b] for b in (after(w) & into) - {w}):
                out.add(reg)
        return out

    def entry_exec_sites(self, cfg, reduction_restores=None, *, compute_dpp=False, pixel_full=False):
        """Prove entry EXEC and its saved copies on every incoming CFG edge.

        The pixel lane-zero retry also permits the whole-quad widening of a
        proven mask. Other callers still require exact entry EXEC.
        """
        bounds, _, succ, _ = cfg
        if not bounds or any(name.startswith(('s_swappc', 's_call', 's_rfe'))
                             or (not pixel_full and name.startswith('s_wqm'))
                             for _, _, name, _ in self.ins):
            return set()
        wave = getattr(self.pipe, 'wave_size', None)
        wave32 = wave == 32 or wave is None and wave32_exec_proof(self.ins)
        if reduction_restores is not None and not wave32:
            return set()
        width = 1 if wave32 else 2
        mask = ('exec_lo', 'exec') if wave32 else ('exec',)
        copy_op = 's_mov_b32' if wave32 else 's_mov_b64'

        suffix = '_b32' if wave32 else '_b64'
        operations = []
        for _, _, name, asm in self.ins:
            _, ops, mods = parse(asm)
            _, _, writes = scalar_effects(name, asm)
            if 'any' in scalar_destinations(name, asm):
                writes |= {'s%d' % i for i in range(106)}
            move = None
            if name == copy_op and len(ops) == 2 and not mods:
                dst, src = ops
                reg = reg_range(src if dst in mask else dst)
                if reg and reg[0] == 's' and reg[2] <= 105 and reg[2] - reg[1] + 1 == width:
                    words = ('s%d' % reg[1],) if wave32 else ('s%d' % reg[1], 's%d' % reg[2])
                    if src in mask:
                        move = ('save', words)
                    elif dst in mask:
                        move = ('restore', words)
            if (reduction_restores is not None and name in ('s_and_saveexec_b32', 's_or_saveexec_b32')
                    and len(ops) == 2 and not mods):
                reg = reg_range(ops[0])
                if reg and reg[0] == 's' and reg[1] == reg[2] <= 105:
                    move = ('save', ('s%d' % reg[1],))
            narrows = compute_dpp and name == 's_and_saveexec' + suffix and len(ops) == 2 and not mods
            if narrows:
                words = mask_words(ops[0], width)
                if words and all(re.fullmatch(r's\d+', word) for word in words):
                    move = ('save', tuple(words))
            elif compute_dpp and name == 's_or' + suffix and len(ops) == 3 and not mods and ops[0] in mask:
                src0, src1 = ops[1:]
                if src1 in mask:
                    src0, src1 = src1, src0
                words = mask_words(src1, width)
                if src0 in mask and words:
                    move = ('restore_or', tuple(words))
            widens = pixel_full and not mods and (
                name == 's_wqm_b64' and ops == ['exec', 'exec'] or
                wave32 and name == 's_wqm_b32' and ops in (['exec', 'exec'], ['exec_lo', 'exec_lo']))
            operations.append((writes_exec(name, ops) or bool(writes & EXEC_KEYS), writes, move, narrows, widens))

        # Missing blocks are unreachable so far. At a join or back edge keep only facts common to all paths.
        # () proves EXEC. The subset fact allows an OR restore after an AND narrowed it.
        # Other facts name the SGPRs of a saved mask copy.
        subset = ('subset',)
        before = {0: {(), subset} if compute_dpp else {()}}
        pending = {0}
        known = set()
        while pending:
            b = pending.pop()
            facts = before[b].copy()
            lo, hi = bounds[b]
            for i in range(lo, hi):
                off = self.ins[i][0]
                if () in facts:
                    known.add(off)
                else:
                    known.discard(off)
                exec_write, writes, move, narrows, widens = operations[i]
                entry = () in facts
                within_entry = subset in facts
                restore = (move is not None and move[1] in facts and
                           (move[0] == 'restore' or (move[0] == 'restore_or' and within_entry)))
                facts = {copy for copy in facts if writes.isdisjoint(copy)}
                if exec_write:
                    facts.discard(())
                    if not narrows:
                        facts.discard(subset)
                if reduction_restores is not None:
                    (reduction_restores.add if restore else reduction_restores.discard)(off)
                if restore or (entry and widens):
                    facts.add(())
                    if compute_dpp:
                        facts.add(subset)
                if entry and move is not None and move[0] == 'save':
                    facts.add(move[1])
            for target in succ[b]:
                merged = before.get(target, facts) & facts
                if before.get(target) != merged:
                    before[target] = merged
                    pending.add(target)
        return known

    def compute_dpp_sites(self, branches, names):
        """Map proven EXEC sites to blocks reached once by every group thread."""
        bounds, _, succ, _ = self.spill_cfg
        if not bounds or any(target <= pc or target not in names for pc, target in branches):
            return {}
        if any(name == 's_setpc_b64' and self.calculated_pc.get(pc) is None for pc, name in names.items()):
            return {}
        # Forward-only control flow. Intersect the blocks guaranteed on each successor path.
        # A final conditional also has a fall-through exit, even if its taken edge has a target.
        always = [set() for _ in bounds]
        for b in range(len(bounds) - 1, -1, -1):
            _, hi = bounds[b]
            always[b].add(b)
            if succ[b] and not (hi == len(self.ins) and self.ins[hi - 1][2].startswith('s_cbranch')):
                always[b].update(set.intersection(*(always[target] for target in succ[b])))
        entry = self.entry_exec_sites(self.spill_cfg, compute_dpp=True)
        return {self.ins[i][0]: self.ins[lo][0] for b in always[0] for lo, hi in (bounds[b],)
                for i in range(lo, hi) if self.ins[i][0] in entry}

    def whole_quad_exec_sites(self, cfg):
        """Must-analysis of lane-exact whole-quad EXEC before each instruction.

        The entry WQM no-op matches the guest on each host quad. From there, full-width saves, restores and
        narrowing by a whole-quad mask stay lane-exact. A write outside this model drops the
        proof. At joins and back edges intersect the facts from every incoming path.
        """
        bounds, _, succ, _ = cfg
        wave32 = wave32_exec_proof(self.ins)
        width = 'b32' if wave32 else 'b64'
        full_exec = {'exec', 'exec_lo'} if wave32 else {'exec'}
        constants = {'0', '-1'}
        move, narrow, save = ('s_%s_%s' % (op, width) for op in ('mov', 'and', 'and_saveexec'))
        operations = []
        for off, _, name, asm in self.ins:
            _, ops, mods = parse(asm)
            _, _, writes = scalar_effects(name, asm)
            operations.append((off, name, ops, mods, writes, writes_exec(name, ops) or bool(writes & EXEC_KEYS),
                               name.startswith('exp') and 'vm' in asm.split()))

        before = {0: {'entry'}}
        pending = {0}
        known = set()
        while pending:
            block = pending.pop()
            facts = before[block].copy()
            lo, hi = bounds[block]
            for off, name, ops, mods, writes, exec_write, discard in operations[lo:hi]:
                if 'exec' in facts:
                    known.add(off)
                else:
                    known.discard(off)
                if discard:
                    # An EXEC-guarded export can discard host lanes before a later restore.
                    facts.clear()
                    continue
                previous = facts
                facts = {fact for fact in previous if not isinstance(fact, frozenset) or writes.isdisjoint(fact)}
                if not exec_write:
                    continue
                facts.discard('entry')
                facts.discard('exec')
                if not mods and name == 's_wqm_' + width and ops[0] in full_exec and ops[1:] == ops[:1]:
                    if previous & {'entry', 'exec'}:
                        facts.add('exec')
                    continue
                if not mods and 'exec' in previous:
                    if name == narrow and ops[0] in full_exec and (
                            set(ops[1:]) <= full_exec | constants and set(ops[1:]) & full_exec):
                        facts.add('exec')
                        continue
                    if name == save and ops[1] in full_exec | constants:
                        dst = reg_range(ops[0])
                        if dst and dst[0] == 's' and dst[2] <= 105 and dst[2] - dst[1] == (0 if wave32 else 1):
                            facts.update(('exec', frozenset(scalar_tokens(ops[0]))))
                            continue
                    if name == move and ops[0] in full_exec and (
                            ops[1] in full_exec or frozenset(scalar_tokens(ops[1])) in previous):
                        facts.add('exec')
                        continue
                # A later restore cannot repair values produced by an inexact EXEC write.
                facts.clear()
            for target in succ[block]:
                merged = before.get(target, facts) & facts
                if before.get(target) != merged:
                    before[target] = merged
                    pending.add(target)
        return known

    def uniform_entry_exec(self):
        sites = getattr(self, 'entry_exec_proof', None)
        return self.exec_known if sites is None else self.pc in sites

    def try_image_loads(self):
        # The established translation wins even if the bytes also happen to
        # admit an image interpretation. Only GETPC programs need this retry.
        code = self.raw_code
        aligned = code[:len(code) & ~3]
        if not any(word & 0xff80ffff == 0xbe801f00 for word, in struct.iter_unpack('<I', aligned)):
            return
        if not any(word & 0xfc000000 == 0xdc000000 for word, in struct.iter_unpack('<I', aligned)):
            return
        try:
            ins = decode(code)
            calculated_pc_proof(ins, 'cs')
            if any(name == 's_setpc_b64' for _, _, name, _ in ins):
                return
        except Unsupported:
            pass
        proof = ImageLoadProof(code, self.pipe.group[:self.pipe.entry[2]])
        try:
            tables = proof.prove()
        except ImageAccessRefusal as reason:
            if self.refusals is None:
                raise
            self.refusals.append((str(reason), self.kind))
            return
        except Unsupported:
            return
        # If the legacy proof refused a different effect, let normal lowering
        # check it again with the reachable instruction list and image sites.
        self.image_proof, self.image_tables = proof, tables
        self.ins = proof.instructions

    def translate(self):
        refusals = getattr(self, 'refusals', None)  # unit probes build a Stage without __init__
        vcc_branches = getattr(self, 'mesh_vcc_branches', {})
        wave_limits = getattr(self, 'mesh_wave_limits', {})
        wave_loads = getattr(self, 'mesh_wave_loads', {})
        wave_load_compares = {compare: bound for compare, bound, _, _ in wave_loads.values() if bound is not None}
        mesh_votes = getattr(self, 'mesh_wave_votes', {})
        mesh_reads = {off for off, _, name, _ in self.ins if name == 'x2w_mesh_firstlane'}
        mesh_selects = {off for off, _, name, _ in self.ins if name == MESH_MASK_SELECT}
        wave_load_sites = {load for _, _, loads, _ in wave_loads.values() for load in loads}
        execz_effects = 'slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes'
        vcc_compares = set().union(*vcc_branches.values())
        try:
            image = getattr(self, 'image_proof', None)
            self.calculated_pc = image.lowered if image else calculated_pc_proof(self.ins, getattr(self, 'kind', None))
        except Unsupported as reason:
            if refusals is None:
                raise
            refusals.append((str(reason), self.kind))
            self.calculated_pc = {}
        if self.embedded_buffer:
            self.calculated_pc.update(self.embedded_buffer[0])
        targets = set()
        targets.update(target for target in self.calculated_pc.values() if target is not None)
        # (branch, target) of every branch, the proven s_setpc_b64 jumps included
        branches = [(pc, target) for pc, target in self.calculated_pc.items() if target is not None]
        for off, n, name, asm in self.ins:
            if name.startswith('s_cbranch') or name == 's_branch':
                target = int(parse(asm)[1][0].split('_')[1], 16)
                targets.add(target)
                branches.append((off, target))
        if getattr(self, 'kind', None) == 'cs':
            self.address_carry_pairs = adjacent_address_carries(self.ins, targets)
        # Forward edges reach each block in disassembly order. A back edge could
        # repeat a pointer advance before the next table load.
        self.forward_table_adds = (getattr(self, 'kind', None) in ('vs', 'ps') and bool(branches)
                                   and all(target > pc for pc, target in branches))
        # Targets that only forward s_cbranch_execz branches (and the fall-through) enter: on a taken edge the lane's
        # exec bit is 0 at the target. Compute stages have their own execz handling and are left out.
        names = {off: name for off, _n, name, _asm in self.ins}
        execz_targets = set() if getattr(self, 'kind', None) == 'cs' else {
            t for t in targets if all(pc < t and names[pc] == 's_cbranch_execz' for pc, to in branches if to == t)}
        back_targets = {to for pc, to in branches if to <= pc}
        self.mesh_allocation_block = None
        splits = targets | mesh_reads | mesh_selects
        if getattr(self, 'kind', None) == 'ms' and (targets or mesh_reads or mesh_selects):
            at = next(i for i, item in enumerate(self.ins) if item[2] == 'x2w_mesh_counts')
            self.mesh_allocation_block = self.ins[at][0]
            # The exchange and its EXEC restore always run. Keep the group barrier and allocation outside
            # the pc dispatch even when the preceding region took different paths.
            splits.update((self.mesh_allocation_block, self.ins[at + 2][0]))
        join_writes = self.join_writes(branches)
        refused = self.execz_refusals() if getattr(self, 'kind', None) == 'cs' else {}  # probes skip __init__
        # A pixel program with an implicit-derivative op and no backward branch decides its EXEC branches per quad
        # (quad vote): see Stage.quad_vote. Loops keep the rules of the back-edge proof below.
        self.quad_exec = getattr(self, 'kind', None) == 'ps' and not any(t <= pc for pc, t in branches) and any(
            name in DERIVATIVE_OPS for _off, _n, name, _asm in self.ins)
        self.uses_quad_vote = False
        # A pixel program with a wave-wide read (a whole-wave reduction) and only forward branches decides every EXEC / VCC
        # branch by a vote over the host wave, so that every thread of the wave reaches the read together (see
        # the active-lane model, rules C1..C5). Programs without such a read keep their per-thread branches.
        self.wave_exec = (getattr(self, 'kind', None) == 'ps' and bool(branches) and all(t > pc for pc, t in branches)
                          and (self.holds_wave_reduction() or (not self.quad_exec and self.skipped_scalar_read_later(branches))))
        # The new proof is a retry only. Keep the dispatch and HLSL of every legacy success.
        retry_wave = getattr(self, 'retry_wave_reduction', False)
        self.wave_exec |= retry_wave
        # Use the existing wave-uniform branch path only for a compute shader the lane proof rejected.
        # At most 4 threads always share one host wave (SM 6 minimum); a larger group could split across waves and vote per wave.
        scalar_vote = (bool(refused) and all(self.execz_scalar_votes.get(at) for at in refused)
                       and all(target > pc for pc, target in branches)
                       and math.prod(getattr(self.pipe, 'group', (5,))) <= 4)
        if scalar_vote:
            self.wave_exec = True
        joined, forward_splits, div_targets, self.spill_cfg = self.joined_registers(with_cfg=True)
        self.signed_pack_entry_exec = (self.entry_exec_sites(self.spill_cfg)
                                       if 'v_cvt_pknorm_i16_f32' in names.values() else None)
        if retry_wave:
            self.reduction_restores = set()
            self.reduction_entry_exec = self.entry_exec_sites(self.spill_cfg, self.reduction_restores)
        self.whole_quad_sites = set()
        if (getattr(self, 'kind', None) == 'ps' and not getattr(self, 'fetched', False) and
                any(name == 'v_add_f32_dpp' for _, _, name, _ in self.ins)):
            self.whole_quad_sites = self.whole_quad_exec_sites(self.spill_cfg)
        self.entry_wqm_sites = (self.entry_wqm_exec_sites()
                                if branches and getattr(self, 'kind', None) == 'ps' and
                                any(name == 'v_subrev_f32_dpp' for _, _, name, _ in self.ins)
                                else set())
        self.has_image_writes = any(name.startswith(('image_store', 'image_atomic')) for _, _, name, _ in self.ins)
        self.loads_after_exec_write = set()
        self.loads_after_wide_exec_write = set()
        all_uniform_loads = UNIFORM_LOADS | UNIFORM_IMAGE_LOADS
        if (back_targets and getattr(self, 'kind', None) == 'cs' and
                any(name in all_uniform_loads for _, _, name, _ in self.ins)):
            # A later block can reach an earlier load through one or more edges. exec_known only sees
            # instructions already translated. Follow the CFG, including paths outside a loop's textual range.
            bounds, _, succ, _ = self.spill_cfg
            for reached, wide_only in ((self.loads_after_exec_write, False), (self.loads_after_wide_exec_write, True)):
                pending = []
                for block, (lo, hi) in enumerate(bounds):
                    if any(writes_exec(name, parse(asm)[1]) and not (wide_only and name.startswith('v_cmpx'))
                           for _, _, name, asm in self.ins[lo:hi]):
                        pending.extend(succ[block])
                visited = set()
                while pending:
                    block = pending.pop()
                    if block not in visited:
                        visited.add(block)
                        lo, hi = bounds[block]
                        reached.update(pc for pc, _, name, _ in self.ins[lo:hi] if name in all_uniform_loads)
                        pending.extend(succ[block])
        self.compute_dpp_entry = {}
        if (getattr(self, 'kind', None) == 'cs' and
                any(name.endswith('_dpp') and name[:-4] in COMPUTE_DPP_FLOAT
                    for name in names.values())):
            self.compute_dpp_entry = self.compute_dpp_sites(branches, names)
        self.entry_exec_proof = (self.entry_exec_sites(self.spill_cfg)
                                 if getattr(getattr(self, 'pipe', None), 'retry_entry_exec', False) else None)
        self.spill_reach = self.spill_reaching(self.spill_cfg)
        init_sym = dict(self.sym)
        init_exec_known = getattr(self, 'exec_known', True)
        init_narrow_only = getattr(self, 'exec_narrow_only', True)
        init_wqm_exec = getattr(self, 'wqm_exec', False)
        init_masks = set(getattr(self, 'masks', ()))
        init_attrs = dict(getattr(self, 'attrs', {}))
        init_used_s = set(getattr(self, 'used_s', ()))
        init_used_v = set(getattr(self, 'used_v', ()))
        init_entry_uniform = set(getattr(self, 'entry_uniform_s', ())) | {
            k for k, v in init_sym.items()
            if isinstance(k, int) and v and (
                (v[0] == 'const' and len(v) == 2) or v[0] in ('ud', 'fixed_buffer_word')
            )
        }
        entry_ud = {reg: sym for reg, sym in init_sym.items() if sym[0] == 'ud'}
        forced = getattr(self, 'force_joined', {})
        force_unproven = set()
        force_at = {}
        first_refusal = len(refusals) if refusals is not None else 0
        every_sgpr = set(range(106)) | {'m0'}
        init_entry = [r for r in set(init_sym) | init_used_s if isinstance(r, int)]
        init_written = set(range(max(init_entry + [-1 if getattr(self, 'kind', None) == 'cs' else 15]) + 1)) | {'m0'}

        # Forward, wave-uniform branches can choose different values on each edge
        # while still selecting the same value for every lane at a join.
        self.firstlane_cfg = (getattr(getattr(self, 'pipe', None), 'retry_firstlane', False)
                              and not back_targets)
        self.firstlane_ps_exec = (self.entry_exec_sites(self.spill_cfg)
                                  if self.firstlane_cfg and self.kind == 'ps' else None)
        self.firstlane_ps_full_exec = (self.entry_exec_sites(self.spill_cfg, pixel_full=True)
                                       if self.firstlane_ps_exec is not None else set())
        for _ in range(max(4, len(self.ins) + 2)):
            scc_effects = {}
            self.scc_data_reads = {}
            self.advanced_descriptor_words = set()
            self.descriptor_scc_writes = set()
            self.descriptor_scc_selects = set()
            uniform_scc_writes, scc_dependent_writes = set(), set()
            self.address_carry_reader = None
            uniform_vcc_compares = set()
            self.lines = []
            self.blocks = []          # (offset, [lines], terminator)
            self.sym = dict(init_sym)
            self.exec_known = init_exec_known
            self.exec_narrow_only = init_narrow_only
            self.wqm_exec = init_wqm_exec
            self.wqm_count, self.execcopy_epoch = 0, {}
            self.helper_restore = False
            # True until a wave-divergent branch: SampleBias / CalculateLevelOfDetail need uniform CF.
            self.uniform_cf = True
            self.derivative_sites = []      # (offset, name) of image_sample_b / image_get_lod, for the back-edge proof
            self.divergent_back_edges = []  # (target, branch offset) of divergent backward branches
            self.lane_exec_breaks = []      # (offset, what): the lane's EXEC bit may stop following the guest's from here
            self.voted_branches = set()     # EXEC branches decided per quad (x2w_quad_any); they part no quad
            self.active_waterfall_loads = set()
            self.wave_unsound = None
            if retry_wave:
                self.reduce_plan, self.reduce_pending = {}, {}
                self.reduction_scratch = {}
            self.wave_voted_branches = set()  # EXEC / VCC branches decided by a vote over the host wave (wave_exec)
            self.plain_samples = {}         # offset -> (level-0 call, implicit-LOD call) of a plain image_sample past a join
            self.masks = set(init_masks)
            self.mask_high = set()
            self.uses_mask_ballot = False
            self.join_masks = set()   # marked SGPRs whose mark is not proven on every path (unjoined)
            last_write = {}           # SGPR -> offset of its last write in program order
            self.attrs = dict(init_attrs)
            self.used_s, self.used_v = set(init_used_s), set(init_used_v)
            self.exports = {}
            self.write_masks = {}
            self.compr_params = set()
            self.plain_params = {}
            self.f16_plain = {}
            self.uses_pos = False
            self.fetched = False
            self.uniform_v = set()
            self.firstlane_v = set()
            self.firstlane_ps_load_v = set()
            firstlane_edges = {}
            self.entry_written_v = set()  # uniform VGPRs written with entry EXEC, up to the next join or VGPR write
            self.varying_s = set()
            self.align4_v = set()
            self.align4_s = set()
            self.scc_uniform = True
            self.scc_verified = False
            # Offset of the emitted, uniform SCC write that reaches here (None: none, or not proven), and the one
            # each forward branch carries to its target.
            self.scc_def, scc_in = None, {}
            self.low16_only_v = set()   # VGPRs whose upper 16 bits are undefined after odd-dmask d16 image_load
            self.pinned_low16_v = set() # low16_only_v live across basic-block boundaries
            self.exec_generation = 0
            self.low16_generation = {}
            self.whole_read_v = set()   # VGPRs read outside a proven low-16-bit consumer
            self.whole_read_sites = {}  # first upper-half consumer per VGPR
            self.low16_copies = {}      # source VGPR -> copy destinations for the block-boundary guard
            self.float_const_s = {}     # SGPR -> 32-bit pattern from s_mov_b32 of an inline float constant
            self.has_branches = bool(targets)
            self.compute_dpp_blocks = set()
            self.whole_quad_reads = set()
            self.quad_flow_reads = []     # (offset, mnemonic, allow late WQM) for DPP reads in branching code
            self.divergent_branches = {}  # s_cbranch offset -> mnemonic, for branches not proven wave-uniform
            self.wqm_set_at = None        # offset of the s_wqm exec, exec that proved whole-quad EXEC first
            self.spill_slots = {}       # (vgpr, lane) -> (sym, uniform, proven, mask) for constant-lane SGPR spills
            self.vcc_events = {}
            self.vcc_lost = self.vcc_descriptor_overwritten = self.vcc_descriptor_extended = False
            self.vcc_constant_pair = False
            self.entry_table_pointers = {
                reg: sym[1] for reg, sym in init_sym.items()
                if isinstance(reg, int) and sym[0] == 'ptr' and sym[1][0] == 'table'
            }
            self.descriptor_block_first = 0
            self.uniform_expr = {}    # proven VGPR -> uniform HLSL expression over scalars that defined it
            self.uniform_load_temps = set()  # temporaries of uniform loads past the entry block (set_v)
            self.uniform_branches = set()  # s_cbranch_* offsets proven uniform in this pass (spill-slot proof)
            self.spill_writes = {}    # instruction index of a v_writelane_b32 translated in this pass -> (sym, uniform, proven, mask)
            self.heap_index = {}      # SGPR -> what its word is for a resource-heap s_load (see heap_record)
            self.heap_index_was = {}  # the marks the current instruction's destinations held before it
            self.since_join = None    # SGPRs proven uniform since the last branch target (None: none reached)
            self.defined_uniform_s = set(init_entry_uniform)
            self.bad_sgprs = set()
            bad_at_target = {}
            tentative_kept = set()
            materialize = {}
            incoming = {}
            kept = {}
            self.back_force = {}
            written = set(init_written)
            falls = True
            start = 0
            raised_unsupported = None
            self.mesh_collect_reads, self.mesh_collect_votes = {}, {}
            self.mesh_collect_masks = {}
            self.exec_keep = {}  # registers whose body-arm descriptor symbol is kept after an execz join (rule R1)

            for instruction_index, (off, n, name, asm) in enumerate(self.ins):
                if self.exec_keep:
                    self.end_exec_keep(instruction_index, off, name, asm, targets)
                if off in splits and off != start:
                    self.exec_generation += 1
                    self.pinned_low16_v.update(self.low16_only_v)
                    self.align4_v.clear()
                    self.align4_s.clear()
                    self.blocks.append((start, self.lines, None))
                    self.lines, start = [], off
                    self.descriptor_block_first = instruction_index
                # The relaxed pixel proof tracks a load only within its straight run.
                if self.firstlane_ps_load_v:
                    _, operands, _ = parse(asm)
                    if (instruction_index == self.descriptor_block_first or writes_exec(name, operands)
                            or scalar_effects(name, asm)[2] & EXEC_KEYS):
                        self.firstlane_ps_load_v.clear()
                    else:
                        base = re.sub(r'_e(32|64)$', '', name)
                        self.firstlane_ps_load_v.difference_update(lds_vgpr_writes(base, operands))
                self.pc = off
                if retry_wave:
                    self.exec_known = off in self.reduction_entry_exec
                self.descriptor_instruction_index = instruction_index
                self.scc_effect = None
                self.scc_from_uniform_input = False
                if off in targets:
                    # Every path in carries the same uniform emitted write, which the lane and the wave both ran
                    # last, whichever path each took (a per-lane s_cbranch_execz edge included); the SCC writes on a
                    # back edge are not known yet.
                    defs = ([self.scc_def] if falls else []) + scc_in.get(off, [])
                    one_def = off not in back_targets and len(set(defs)) == 1
                    self.scc_def = defs[0] if one_def else None
                elif not falls:
                    self.scc_def = None
                if not falls:
                    self.scc_verified = False
                    self.align4_v.clear()
                    self.align4_s.clear()
                # Before the first branch target or terminator every path has run exactly the instructions above, so
                # the program-order tracking is exact there even in a program with branches. Code after s_endpgm in a
                # branch-free program is past it too (dead code).
                self.past_entry = bool(self.blocks) or 0 in targets
                if off in targets:
                    self.heap_index.clear()  # another path into the join may hold a different word
                    self.align4_v.clear()
                    self.align4_s.clear()
                    pre_uniform = {
                        r for r in (self.defined_uniform_s if self.since_join is None else self.since_join)
                        if r not in self.varying_s and r not in self.masks
                    }
                    kept[off] = {}
                    here = joined.get(off, set())
                    arriving = ([({reg: self.sym.get(reg) for reg in here}, written & here)] if falls else []) + \
                        incoming.get(off, [])
                    written = (written - here) | set().union(*(w for _, w in arriving))
                    for reg in here:
                        paths = [s[reg] for s, w in arriving if reg in w]
                        force = forced.get(off, {}).get(reg)
                        if falls and off in execz_targets and self.exec_zero_keepable(off, reg, arriving, paths, branches):
                            # R1: a descriptor word the fall-through arm wrote and every other edge (a taken execz edge, exec == 0)
                            # skipped. Until exec is written again only exec-masked reads can see it: keep the arm's symbol.
                            self.sym[reg] = paths[0]
                            self.exec_keep[reg] = paths[0]
                            continue
                        if any(reg not in w for _, w in arriving) and any(p and not run_time_symbol(p) for p in paths):
                            self.sym[reg] = ('joined at a branch target',)
                            continue
                        if force is None and paths and names_one_value(paths[0]) and all(p == paths[0] for p in paths):
                            self.sym[reg] = kept[off][reg] = paths[0]
                            continue
                        if force == 'refuse' or any(p and not run_time_symbol(p) for p in paths):
                            self.sym[reg] = ('joined at a branch target',)
                            continue
                        if reg in entry_ud:
                            materialize[reg] = 'ud%d' % entry_ud[reg][1]
                        self.sym.pop(reg, None)
                        if reg != 'm0':
                            self.used_s.add(reg)
                    # A joined register the fall-through wrote as a plain run-time number: every other path into
                    # this target has exec bit 0, so the instruction at the target may AND it with exec (scalar()). A register that still holds a plain
                    # variable reads the same either way.
                    self.exec_zero_join = (off, {
                        reg for reg in here
                        if falls and off in execz_targets and reg in arriving[0][1] and arriving[0][0][reg] is None})
                    clobber = here if off in joined else every_sgpr
                    self.join_masks |= self.unjoined(off, last_write, self.masks)
                    fwd_clobber = forward_splits.get(off, clobber)
                    blocked = force_unproven | force_at.get(off, set())
                    kept_u = {
                        r for r in pre_uniform
                        if r not in clobber or (r not in fwd_clobber and r not in blocked)
                    }
                    tentative_kept.update((kept_u & clobber) - blocked)
                    self.uniform_v.clear()
                    self.entry_written_v.clear()
                    self.since_join = kept_u
                    self.scc_uniform = (self.firstlane_cfg and self.kind == 'ps' and self.uniform_cf
                                        and self.scc_def is not None)
                    self.scc_verified = self.scc_def is not None
                    # A constant is only known here if no other path into the join writes the register: a loop counter
                    # set before the loop and incremented in it reaches the loop head with a new value on every pass.
                    # set_s also assigns constants at run time, so the register reads its real value once it is dropped.
                    # clobber also holds the writes of every earlier block off the dominator chain; a constant written
                    # after those, by a block every path runs, with no later write on any path, is still known.
                    consts = {reg for reg in join_writes.get(off, set()) | (clobber & every_sgpr)
                              if (self.sym.get(reg) or ('',))[0] == 'const'}
                    for reg in self.unjoined(off, last_write, consts):
                        del self.sym[reg]
                    fconsts = (join_writes.get(off, set()) | (clobber & every_sgpr)) & set(self.float_const_s)
                    for reg in self.unjoined(off, last_write, fconsts):
                        self.float_const_s.pop(reg, None)
                    self.defined_uniform_s = set(kept_u) | {
                        k for k, v in self.sym.items()
                        if isinstance(k, int) and v and (
                            (v[0] == 'const' and len(v) == 2) or v[0] in ('ud', 'fixed_buffer_word')
                        )
                    }
                if self.firstlane_cfg and self.uniform_cf:
                    if off in targets:
                        paths = firstlane_edges.get(off, [])
                        if falls:
                            paths = paths + [(self.firstlane_v, pre_uniform)]
                        if paths:
                            self.firstlane_v = set.intersection(*(v for v, _ in paths))
                            scalar_phi = set.intersection(*(u for _, u in paths))
                            if self.kind == 'ps':
                                # Program order may still hold a write from a sibling arm.
                                # Only the incoming edges establish a definition at this join.
                                self.since_join = scalar_phi
                                self.defined_uniform_s = scalar_phi.copy()
                            else:
                                self.since_join.update(scalar_phi)
                                self.defined_uniform_s.update(scalar_phi)
                            self.varying_s.difference_update(scalar_phi)
                        else:
                            self.firstlane_v.clear()
                    elif not falls:
                        self.firstlane_v.clear()
                if off in targets and self.exec_keep and name.startswith('s_'):
                    # The join instruction itself runs with exec == 0 on the taken edge: a scalar read of a kept word ends it here.
                    self.end_exec_keep_scalar_read(name, asm)
                if off in targets and self.exec_keep and (
                        name.startswith('s_cbranch') or name in ('s_branch', 's_setpc_b64')):
                    # The join instruction is itself a branch: end_exec_keep only runs at the top of the next
                    # instruction, so retire the kept words before the branch records its incoming edge.
                    self.retire_exec_keep(set(self.exec_keep))
                if name.startswith('s_cbranch') or name == 's_branch' or (
                        name == 's_setpc_b64' and self.calculated_pc.get(off) is not None):
                    b_target = self.calculated_pc[off] if name == 's_setpc_b64' else int(parse(asm)[1][0].split('_')[1], 16)
                    there = joined.get(b_target, set())
                    if b_target > off:
                        incoming.setdefault(b_target, []).append(({reg: self.sym.get(reg) for reg in there}, written & there))
                        scc_in.setdefault(b_target, []).append(self.scc_def)
                        if self.firstlane_cfg and self.uniform_cf:
                            scalar_facts = self.defined_uniform_s if self.since_join is None else self.since_join
                            firstlane_edges.setdefault(b_target, []).append(
                                (self.firstlane_v.copy(), scalar_facts - self.varying_s - self.masks))
                    else:
                        for reg in there & written:
                            back, sym = self.sym.get(reg), kept.get(b_target, {}).get(reg)
                            if back and not run_time_symbol(back) and back != sym:
                                self.back_force.setdefault(b_target, {})[reg] = 'refuse'
                            elif sym is not None and back != sym:
                                self.back_force.setdefault(b_target, {}).setdefault(
                                    reg, 'run' if run_time_symbol(sym) else 'refuse')
                falls = name not in ('s_branch', 's_setpc_b64', 's_endpgm')
                dests = scalar_destinations(name, asm)
                written |= every_sgpr if 'any' in dests else dests
                last_write.update(dict.fromkeys(every_sgpr if 'any' in dests else dests, off))
                self._pre_fconst = dict(self.float_const_s)
                if 'any' in dests:
                    self.float_const_s.clear()
                else:
                    for reg in dests:
                        self.float_const_s.pop(reg, None)
                try:
                    if off in refused:
                        if not scalar_vote:
                            raise Unsupported(refused[off])
                        operands = parse(asm)[1]
                        sources = operands[1:] if name in SCA else [smem_offset(operands[2])]
                        if not all(self.uniform_source(src) for src in sources):
                            raise Unsupported(refused[off])
                        if name.startswith('s_buffer_load') and self.descriptor(operands[1]) in self.pipe.rw_keys:
                            raise Unsupported(refused[off])
                    base = re.sub(r'_e(32|64)$', '', name)
                    if off in wave_load_compares:
                        bound = wave_load_compares[off]
                        if not self.uniform_source(bound) or int(bound[1:]) not in self.defined_uniform_s:
                            raise Unsupported(execz_effects)
                        # Capture the bound at the compare, before any intervening SGPR write.
                        self.emit('wave_any = (tid & ~63u) < %s;' % self.u(bound))
                    if off in wave_load_sites:
                        offset = smem_offset(parse(asm)[1][2])
                        if not self.uniform_source(offset) or (
                                re.fullmatch(r's\d+', offset) and int(offset[1:]) not in self.defined_uniform_s):
                            raise Unsupported(execz_effects)
                    if off in vcc_compares and all(
                        not mesh_vregs([tok]) and self.uniform_source(tok)
                        for tok in parse(asm)[1][1:]
                    ):
                        uniform_vcc_compares.add(off)
                    if base.startswith('s_cbranch_'):
                        cond = base[10:]
                        if off in wave_loads:
                            # An empty wave keeps the previous values of the scalars the block writes and the
                            # join reads, so each must be wave-uniform and defined before the block.
                            for dst in wave_loads[off][3]:
                                if not self.uniform_source(dst) or int(dst[1:]) not in self.defined_uniform_s:
                                    raise Unsupported(execz_effects)
                                self.u(dst)  # declare the variable; refuses a symbol with no run-time value
                        cond_uniform = ((cond in ('scc0', 'scc1') and self.scc_uniform)
                                        or off in wave_limits or off in wave_loads or off in mesh_votes)
                        if off in vcc_branches:
                            if not vcc_branches[off] <= uniform_vcc_compares:
                                raise Unsupported('slot 57 mesh: %s on a wave-wide condition is not covered' % name)
                            cond_uniform = True
                        if cond_uniform:
                            self.uniform_branches.add(off)
                        elif self.wave_exec and cond in ('execz', 'execnz', 'vccz', 'vccnz'):
                            # Decided over the host wave (WaveActiveAnyTrue): every thread takes it together.
                            self.wave_voted_branches.add(off)
                            self.uniform_branches.add(off)
                            cond_uniform = True
                        elif (cond in ('execz', 'execnz') and self.quad_exec and not self.exec_known
                                and self.uniform_cf and not self.lane_exec_breaks
                                and self.skipped_block_vector_only(off, int(parse(asm)[1][0].split('_')[1], 16))):
                            # Decided per quad (x2w_quad_any): every lane of a quad takes it together, so it parts no
                            # quad and the derivative ops after it still see their four lanes. It reads the quad, so it
                            # is only taken while the control flow so far is quad-uniform and every lane's EXEC bit
                            # still follows the guest's.
                            self.voted_branches.add(off)
                            cond_uniform = True
                        if self.wave_exec and not cond_uniform and self.wave_unsound is None:
                            # A branch that is neither wave-uniform nor a vote parts the host wave, so no wave intrinsic
                            # can be trusted to see every lane. Refused at the end of the pass, so that an earlier
                            # refusal of the program keeps its own reason.
                            self.wave_unsound = Unsupported('wave-wide read in a program with a branch (%s) that is '
                                                            'neither wave-uniform nor an EXEC / VCC vote' % name)
                        if not cond_uniform:
                            # Divergent CF: later SampleBias / CalculateLevelOfDetail would see uneven quads.
                            self.uniform_cf = False
                            self.divergent_branches[off] = base
                            b_target = int(parse(asm)[1][0].split('_')[1], 16)
                            regs, affected = div_targets.get(off, (set(), set()))
                            regs = {r for r in regs if isinstance(r, int)}
                            if b_target <= off:
                                self.divergent_back_edges.append((b_target, off))
                                regs.update(r for r in joined.get(b_target, ()) if isinstance(r, int))
                            for tgt_off in affected:
                                bad_at_target.setdefault(tgt_off, set()).update(regs)
                    term = self.one(name, asm)
                    if off in wave_limits:
                        term = ('branch', '(tid & ~63u) >= %du' % wave_limits[off], term[2])
                    if off in mesh_votes:
                        word, zero = mesh_votes[off]
                        self.mesh_collect_votes[start] = word
                        term = ('branch', 'mesh_vote %s 0u' % ('==' if zero else '!='), term[2])
                    elif off in wave_loads:
                        term = ('branch', '!wave_any', term[2])
                except Unsupported as reason:
                    if refusals is None:
                        if self.has_branches and ('read_vcc', None) in self.vcc_events.get(off, ()):
                            # Some operand handlers refuse VCC before emitting code. Check whether
                            # a path to this instruction has already overwritten the mask.
                            self.blocks.append((start, self.lines, None))
                            try:
                                self.check_vcc_paths()
                            except Unsupported as vcc_reason:
                                reason = vcc_reason
                            finally:
                                self.blocks.pop()
                        if not hasattr(reason, 'instruction'):
                            reason.instruction = name  # for fail_reason(); the message itself is unchanged
                        raised_unsupported = reason
                        break
                    refusals.append((str(reason), self.kind))
                    continue
                except Exception as error:
                    if refusals is None:
                        raise
                    refusals.append((handler_error(name, error, refusals), self.kind))
                    continue
                # A write proves the mark again: every path from here on runs it. s_movreld can write any SGPR.
                self.join_masks = (self.join_masks | every_sgpr) if 'any' in dests else self.join_masks - dests
                if not self.scc_effect and ('scc' in scalar_effects(name, asm)[1] or name.startswith(('s_wqm', 's_quadmask'))):
                    # The guest wrote SCC and the lowering emitted nothing for it: a later reader would see an older
                    # value (descriptor_scc_reads follows every path from here to the reader).
                    self.scc_effect = 'an SCC write that is not modelled'
                if self.scc_effect:
                    scc_effects[off] = self.scc_effect
                    if self.scc_effect == 'emitted':
                        if self.scc_uniform:
                            uniform_scc_writes.add(off)
                        elif self.scc_from_uniform_input:
                            scc_dependent_writes.add(off)
                if self.scc_effect or not self.scc_verified:
                    self.scc_def = off if self.scc_effect == 'emitted' and self.scc_verified and self.scc_uniform else None
                if term:
                    self.exec_generation += 1
                    self.pinned_low16_v.update(self.low16_only_v)
                    self.blocks.append((start, self.lines, term))
                    self.lines, start = [], off + n
                    self.descriptor_block_first = instruction_index + 1

            new_bad = self.bad_sgprs - force_unproven
            new_at = {k: v - force_at.get(k, set()) for k, v in bad_at_target.items() if v - force_at.get(k, set())}
            if (new_bad or new_at) and tentative_kept:
                force_unproven.update(new_bad)
                for k, v in new_at.items():
                    force_at.setdefault(k, set()).update(v)
                if refusals is not None:
                    del refusals[first_refusal:]
                continue
            if retry_wave and raised_unsupported is None and self.wave_unsound is None:
                if self.lane_exec_breaks:
                    self.wave_unsound = Unsupported('wave reduction: lane participation may change at %s'
                                                    % self.lane_exec_breaks[0][1])
                else:
                    self.wave_unsound = self.wave_scratch_refusal()
            if raised_unsupported is None and self.wave_unsound is not None:
                if refusals is None:
                    raise self.wave_unsound
                refusals.append((str(self.wave_unsound), self.kind))
            if raised_unsupported is not None:
                raise raised_unsupported
            break
        else:
            reason = 'v_readfirstlane_b32 uniformity proof exceeded pass limit'
            if refusals is None:
                raise Unsupported(reason)
            refusals.append((reason, self.kind))
        if mesh_reads or mesh_votes or mesh_selects:
            self.check_mesh_collectives(mesh_reads | mesh_votes.keys() | mesh_selects, branches)
        if self.lines:
            self.blocks.append((start, self.lines, None))
        for at, reason in self.quad_flow_refusals(branches, targets):
            if refusals is None:
                error = Unsupported(reason)
                error.instruction = names[at]
                raise error
            refusals.append((reason, self.kind))
        self.materialized = materialize  # declarations() assigns these before the body (and before any loop)
        if self.scc_data_reads:
            reads = dict(self.scc_data_reads)
            for off, name in names.items():
                if name.startswith(('s_cselect', 's_cmov')):
                    reads.setdefault(off, 'unverified SCC data in ' + name)
            for off, detail in self.unverified_scc_data_paths(
                    dict(branches), scc_effects, uniform_scc_writes, scc_dependent_writes, reads):
                reason = '%s at +0x%x: %s' % (reads[off], off, detail)
                if refusals is None:
                    error = Unsupported(reason)
                    error.instruction = names[off]
                    raise error
                refusals.append((reason, self.kind))
        if self.descriptor_scc_writes:
            for off, reason in self.descriptor_flag_refusals(dict(branches), scc_effects, uniform_scc_writes):
                if refusals is None:
                    error = Unsupported(reason)
                    error.instruction = names[off]
                    raise error
                refusals.append((reason, self.kind))
        if set(scc_effects.values()) - {'emitted'}:
            for kind, name in self.descriptor_scc_reads(dict(branches), scc_effects):
                reason = 'scc from %s in %s' % ('descriptor data' if kind == 'descriptor' else kind, name)
                if refusals is None:
                    raise Unsupported(reason)
                refusals.append((reason, self.kind))
        for at, name in self.derivative_sites:
            # A divergent back edge that spans the call lets lanes of one quad leave the loop at different passes, so
            # the call runs again with uneven quads even though no divergent branch precedes it in program order.
            if any(target <= at <= branch for target, branch in self.divergent_back_edges):
                reason = ('%s needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform '
                          'control flow' % name)
                if refusals is None:
                    raise Unsupported(reason)
                refusals.append((reason, self.kind))
        self.resolve_plain_samples()
        # A tail live at a block boundary can reach copies earlier in the
        # instruction stream through a back edge. Keep the same conservative
        # guard for their destinations so a join cannot hide an upper-half use.
        pending = list(self.pinned_low16_v) if self.has_branches else []
        while pending:
            src = pending.pop()
            aliases = self.low16_copies.get(src, set()) - self.pinned_low16_v
            self.pinned_low16_v.update(aliases)
            pending.extend(aliases)
        try:
            self.check_vcc_paths()
        except Unsupported as reason:
            if refusals is None:
                raise
            refusals.append((str(reason), self.kind))
        leak = self.pinned_low16_v & self.whole_read_v
        if leak:
            reg = min(leak, key=lambda n: (self.whole_read_sites[n][0], n))
            reason = self.odd_d16_read_error(reg, self.whole_read_sites[reg][1])
            if refusals is None:
                raise reason
            refusals.append((str(reason), self.kind))

    def unverified_scc_data_paths(self, branches, effects, uniform_writes, dependent_writes, reads):
        """Check the paths to explicit reads that failed the legacy single-definition proof.

        The emitted scc local already acts as the phi: each executed block leaves its boolean
        in the same local. Keep that lowering and the existing uniformity proofs untouched.
        """
        instructions = {off: (size, name) for off, size, name, _ in self.ins}
        offsets = {off: i for i, off in enumerate(instructions)}
        unsafe = {}
        incoming = Counter(branches.values())
        for off, target in branches.items():
            counted = (target < off and target in offsets and incoming[target] == 1
                       and mesh_constant_trip(self.ins, offsets[target], offsets[off], offsets))
            if off in self.calculated_pc or target not in offsets or (target <= off and not counted):
                unsafe[off] = 'unsupported control-flow edge +0x%x -> +0x%x' % (off, target)
            elif instructions[off][1] != 's_branch' and off not in self.uniform_branches:
                unsafe[off] = 'branch at +0x%x is not proven wave-uniform for SCC data' % off

        pending = [(0, None, 'entry path has no SCC producer', None)]
        seen, reached_reads, failures = set(), set(), {}
        while pending:
            off, prev, missing, shape = pending.pop()
            # The strings are witnesses for diagnostics. Only their presence affects propagation.
            state = (off, missing is not None, shape is not None)
            if state in seen or off not in instructions:
                continue
            seen.add(state)
            size, name = instructions[off]
            if off in reads:
                reached_reads.add(off)
                if missing or shape:
                    edge = ' on edge +0x%x -> +0x%x' % (prev, off) if prev is not None else ''
                    if missing or off not in failures:
                        failures[off] = (missing or shape) + edge
            if off in effects:
                effect = effects[off]
                if name in SCC_LEGACY_MIN_MAX:
                    # The legacy expression chooses S0 on ties. The ISA uses strict comparisons.
                    missing = 'SCC producer %s at +0x%x has unverified tie semantics' % (name, off)
                elif off in uniform_writes:
                    missing = None
                elif off not in dependent_writes:
                    kind = {'emitted': 'a non-uniform value', 'descriptor': 'descriptor data'}.get(effect, effect)
                    missing = 'SCC producer %s at +0x%x yields %s' % (name, off, kind)
            if off in branches:
                shape = shape or unsafe.get(off)
                if missing and name.startswith('s_cbranch_scc'):
                    shape = shape or 'branch at +0x%x reads an unverified SCC predicate' % off
                pending.append((branches[off], off, missing, shape))
            if name not in ('s_branch', 's_setpc_b64', 's_endpgm'):
                pending.append((off + size, off, missing, shape))
        for off in reads:
            if off in failures or (off in self.scc_data_reads and off not in reached_reads):
                yield off, failures.get(off, 'no reachable SCC producer')

    def descriptor_flag_refusals(self, branches, effects, uniform_writes):
        """Check the actual reaching SCC writes, including paths through back edges."""
        instructions = {off: (size, name, asm) for off, size, name, asm in self.ins}
        pending = [(0, None)]
        seen, reaching = set(), {}
        while pending:
            off, producer = pending.pop()
            if (off, producer) in seen or off not in instructions:
                continue
            seen.add((off, producer))
            size, name, asm = instructions[off]
            if reads_scc(name, asm):
                reaching.setdefault(off, set()).add(producer)
            if off in effects:
                producer = off
            if off in branches:
                pending.append((branches[off], producer))
            if name not in ('s_branch', 's_setpc_b64', 's_endpgm'):
                pending.append((off + size, producer))
        for off, producers in sorted(reaching.items()):
            if not producers & self.descriptor_scc_writes:
                continue
            name = instructions[off][1]
            if off not in self.descriptor_scc_selects and name not in ('s_cbranch_scc0', 's_cbranch_scc1'):
                yield off, 'scc from descriptor data in %s outside a scalar select or branch' % name
            elif len(producers) != 1 or not producers <= uniform_writes:
                yield off, 'scc from descriptor data in %s without a single wave-uniform producer' % name

    def descriptor_scc_reads(self, branches, effects):
        """Find reads reachable with an elided SCC write, including through joins and back edges."""
        instructions = {off: (size, name, asm) for off, size, name, asm in self.ins}
        pending = [(0, None)]  # stale: None, or the scc_effect of the write that left the emitted scc behind
        seen = set()
        while pending:
            off, stale = pending.pop()
            if (off, stale) in seen or off not in instructions:
                continue
            seen.add((off, stale))
            size, name, asm = instructions[off]
            if (stale and off not in self.calculated_pc and off not in getattr(self, 'ptr_addc_pcs', ())
                    and reads_scc(name, asm)):
                yield stale, name
            if off in effects:
                stale = effects[off] if effects[off] != 'emitted' else None
            if off in branches:
                pending.append((branches[off], stale))
            if name not in ('s_branch', 's_setpc_b64', 's_endpgm'):
                pending.append((off + size, stale))

    def join_writes(self, branches):
        """Branch target -> the scalar registers another path can write before it reaches the target: the loop body
        [target, branch] for a backward branch, and the skipped instructions (branch, target) for a forward one."""
        offs, written = [], []  # per instruction, once: a program has hundreds of branches over thousands of instructions
        for off, _n, name, asm in self.ins:
            regs = set(range(128)) if name.startswith('s_movreld') else set()  # M0 picks the destination
            for tok in destinations(name, parse(asm)[1]):
                r = reg_range(tok)
                if r and r[0] == 's':
                    regs.update(range(r[1], r[2] + 1))
                elif tok == 'm0':
                    regs.add('m0')
            offs.append(off)
            written.append(regs)
        writes = {}
        for branch, target in branches:
            lo, hi = ((bisect.bisect_left(offs, target), bisect.bisect_right(offs, branch)) if target <= branch else
                      (bisect.bisect_right(offs, branch), bisect.bisect_left(offs, target)))
            writes.setdefault(target, set()).update(*written[lo:hi])
        return writes

    def tracks_uniformity(self):
        return all(hasattr(self, name) for name in ('uniform_v', 'varying_s', 'scc_uniform'))

    def vgpr_writer(self, reg, before, after=-1):
        """Index of the last instruction in self.ins[after + 1:before] that may write VGPR reg (the first operand,
        and the second of v_swap_b32), or None."""
        for i in range(before - 1, after, -1):
            _, _, name, asm = self.ins[i]
            ops = parse(asm)[1]
            for tok in (ops[:2] if name.startswith('v_swap') else ops[:1]):
                r = reg_range(tok)
                if r and r[0] == 'v' and r[1] <= reg <= r[2]:
                    return i
        return None

    def tid_shift(self, count, source):
        """The uniform expression of `v_lshrrev_b32 vD, count, source` in the entry block of a compute program, or None.

        The X thread id (v0, each local id has its own VGPR) is below the group's X size, so a right shift by n with
        size <= 1 << n clears every thread-id bit. That gives 0 for the thread id itself, and for a vector built as
        (uniform << K) + v0 with n <= K and size <= 1 << K the low K bits of the base are clear, the sum does not
        carry, and the result is the same in every lane. Program order is execution order here, so the nearest
        earlier writer is the definition; v0 must be unwritten at the definition and the scalar unwritten after it."""
        src, now = reg_range(source), self.pc_index()
        if (not re.fullmatch(r'0x[0-9a-f]+|\d+', count) or int(count, 0) >= 32 or not src or src[0] != 'v' or
                src[1] != src[2] or now is None):
            return None
        group = getattr(self.pipe, 'group', None)
        if not group or group[0] <= 0:
            return None
        n, size = int(count, 0), group[0]
        defined = self.vgpr_writer(src[1], now)
        if defined is None:
            return '0u' if src[1] == 0 and size <= 1 << n else None
        _, _, name, asm = self.ins[defined]
        _, d, mods = parse(asm)
        base = re.sub(r'_e(32|64)$', '', name)
        if (base not in ('v_lshl_add_u32', 'v_and_or_b32', 'v_bfe_u32') or mods or len(d) != 4 or
                self.vgpr_writer(0, defined) is not None or not re.fullmatch(r'0x[0-9a-f]+|\d+', d[2])):
            return None
        imm = int(d[2], 0)
        # The local id is in [0, size). These are all the bits that may vary before the shift.
        tid_bits = (1 << (size - 1).bit_length()) - 1
        if base == 'v_bfe_u32':
            if d[1] != 'v0' or imm >= 32 or not re.fullmatch(r'0x[0-9a-f]+|\d+', d[3]):
                return None
            width = int(d[3], 0)
            if width >= 32 or (((tid_bits >> imm) & ((1 << width) - 1)) >> n):
                return None
            return '0u'
        if base == 'v_and_or_b32':
            if d[1] != 'v0' or imm > 0xffffffff or ((tid_bits & imm) >> n):
                return None
            scalar_token = d[3]
        else:
            if d[3] != 'v0' or imm >= 32 or n > imm or size > 1 << n:
                return None
            scalar_token = d[1]
        scalar = reg_range(scalar_token)
        if (not scalar or scalar[0] != 's' or scalar[1] != scalar[2] or scalar[1] > 105 or
                not self.uniform_source(scalar_token)):
            return None
        for _, _, later, later_asm in self.ins[defined + 1:now]:
            writes = scalar_destinations(later, later_asm)
            if scalar[1] in writes or 'any' in writes:
                return None
        if base == 'v_and_or_b32':
            return '(%s >> %du)' % (self.S(scalar[1]), n)
        return '((s%d << %du) >> %du)' % (scalar[1], imm, n)

    def pc_index(self):
        return next((i for i, ins in enumerate(self.ins) if ins[0] == self.pc), None)

    def helper_off_copy(self, tok, sym):
        """Whether `tok` is a copy of EXEC saved before an `s_wqm_b64 exec, exec`: the guest's entry EXEC leaves the
        helper lanes off, while every thread here starts with exec = 1, so writing it back would turn the guest's helper
        lanes off (stale VGPRs) and keep the translation's on."""
        r = reg_range(tok)
        return bool(r and r[0] == 's' and sym == ('execcopy',)
                    and getattr(self, 'execcopy_epoch', {}).get(r[1], 0) < getattr(self, 'wqm_count', 0))

    def exec_write_exact(self, srcs, syms):
        """Whether a scalar EXEC write takes every source from a lane mask (vcc, exec, a saved EXEC or a tagged mask) or
        from the all-zero / all-ones constants: then each lane's bit is the guest's. Any other source (a mask built
        from a number such as 1) selects guest lanes the one-lane model cannot tell apart."""
        for tok, sym in zip(srcs, syms):
            r = reg_range(tok)
            if self.helper_off_copy(tok, sym):
                return False
            # The 64-bit forms only (see also the exec_lo destination at the call site): in wave64 a 32-bit write
            # (s_mov_b32 exec_lo, vcc_lo) keeps the old bits of lanes 32..63, while the model sets every lane.
            if tok in ('exec', 'vcc') or sym == ('execcopy',) or (
                    r and r[0] == 's' and r[1] in getattr(self, 'masks', ())):
                continue
            try:
                if int(tok, 0) in (0, -1, 0xffffffff, 0xffffffffffffffff):
                    continue
            except ValueError:
                pass
            return False
        return True

    def skipped_block_vector_only(self, branch, target):
        """Whether the instructions a forward EXEC branch can skip are per-lane vector work only.

        The guest takes the branch once for the whole wave, so a scalar write, a mask write or a branch inside the block
        happens once for the wave when any quad takes the block. A quad that votes to skip it would miss that effect while
        a later read in the same wave sees it. Only vector ALU/memory/export instructions and waits, which each lane
        applies to its own registers, are allowed between the branch and its target; s_endpgm too (a quad that ends
        reads nothing later)."""
        if target <= branch:
            return False
        for off, _n, name, asm in self.ins:
            if not branch < off < target:
                continue
            if name == 's_endpgm':
                # A quad that ends reads nothing later, unless later code can turn it back on: an EXEC write after the
                # branch (a restore of a saved mask) would reactivate every quad of the guest wave.
                if any(o > branch and writes_exec(n, parse(a)[1]) for o, _s, n, a in self.ins):
                    return False
                continue
            if not (self.vector_only_instruction(name, asm)
                    or name.startswith(('image_', 'buffer_', 'tbuffer_', 'global_', 'flat_', 'exp'))
                    or name in ('s_waitcnt', 's_nop', 's_waitcnt_vscnt')):
                return False
        return True

    @staticmethod
    def vector_only_instruction(name, asm):
        """A vector ALU instruction whose only destination is a VGPR: no SGPR, VCC or EXEC write (a carry-out, a compare,
        a readlane), which the guest performs once for the whole wave."""
        if not name.startswith('v_') or name.startswith(('v_cmp', 'v_readlane', 'v_readfirstlane', 'v_writelane',
                                                         'v_div_scale', 'v_mad_u64', 'v_mad_i64')):
            return False
        if re.match(r'v_(add|sub|subrev)_co|v_addc|v_subb', name):
            return False
        for tok in destinations(name, parse(asm)[1]):
            r = reg_range(tok)
            if tok in ('vcc', 'vcc_lo', 'vcc_hi', 'exec', 'exec_lo', 'exec_hi', 'm0') or (r and r[0] == 's'):
                return False
        return True

    def uniform_bias(self, tok):
        """Whether a VGPR holds one value in every lane at the op: its last earlier write is a v_mov_b32 of a constant
        that every lane ran (no EXEC write and no branch before it, so no lane skipped it and the other lanes keep an
        older value). No backward branch exists in a quad program, so later writes cannot reach the op."""
        r = reg_range(tok)
        if not r or r[0] != 'v' or r[1] != r[2]:
            return False
        last = None
        for off, _n, name, asm in self.ins:
            if off >= self.pc:
                break
            ops = parse(asm)[1]
            first = reg_range(ops[0]) if ops else None
            if first and first[0] == 'v' and first[1] <= r[1] <= first[2]:
                const = (re.sub(r'_e(32|64)$', '', name) == 'v_mov_b32' and len(ops) == 2 and not reg_range(ops[1]))
                last = off if const else None
        if last is None:
            return False
        narrowed = 0
        for off, _n, name, asm in self.ins:
            if off >= last:
                break
            base = re.sub(r'_e(32|64)$', '', name)
            ops = parse(asm)[1]
            if name.startswith('s_cbranch') or name == 's_branch':
                target = int(ops[0].split('_')[1], 16) if '_' in ops[0] else None
                if target is None or target > last:
                    return False  # the write can be skipped by some lanes
            elif base in ('s_or_b64', 's_or_b32') and ops[:2] in (['exec', 'exec'], ['exec_lo', 'exec_lo']):
                narrowed = max(narrowed - 1, 0)  # the restore that ends a saveexec region
            elif base.startswith('s_and_saveexec') or (base.startswith('s_wqm') and ops[:2] not in (['exec', 'exec'], ['exec_lo', 'exec_lo'])):
                narrowed += 1
            elif writes_exec(name, ops) and not (base.startswith('s_wqm') and ops[:2] in (['exec', 'exec'], ['exec_lo', 'exec_lo'])):
                return False
        return narrowed == 0

    def derivative_exec_ok(self):
        """Whether an implicit-derivative op may rely on the EXEC the lanes hold here.

        At the entry mask every lane runs the call. Past an EXEC write (v_cmpx, saveexec, a restore) the one-lane
        model still follows the guest's bit of each lane exactly (EXEC is written per lane, set_v is EXEC guarded),
        so a quad program (a pixel program with a derivative op and no backward branch) is proven as long as no
        lane-exec break happened: an s_wqm of a narrowed EXEC (a no-op here, while the guest turns on the lanes of
        every touched quad) or a discard under an unknown EXEC. The callers also need uniform_cf, which a branch
        that parts a quad clears; an EXEC branch decided per quad (quad_vote) does not."""
        # A restored pre-s_wqm mask (helper_restore) is the entry mask, but the guest's helper lanes are off from here.
        return ((self.exec_known and not getattr(self, 'helper_restore', False))
                or (getattr(self, 'quad_exec', False) and not getattr(self, 'lane_exec_breaks', None)))

    def uniform_source(self, tok):
        r = reg_range(tok)
        if r:
            kind, lo, hi = r
            if lo != hi:
                return False
            if kind == 'v':
                return lo in self.uniform_v
            if lo in getattr(self, 'masks', ()):
                return False
            if getattr(self, 'since_join', None) is not None and lo not in self.since_join:
                return False  # written before the last join: program-order tracking is not exact there
            sym = self.sym.get(lo)
            if sym:
                return sym[0] in ('const', 'ud', 'fixed_buffer_word')
            return lo not in self.varying_s
        if tok == 'null' or tok in FLOATS:
            return True
        try:
            int(tok, 0)
        except ValueError:
            return False
        return True

    def uniform_image_load_regs(self, base, ops, mods):
        """Return the range of destination VGPRs written uniformly by `image_load`, or () when unproved."""
        if (base not in UNIFORM_IMAGE_LOADS or len(ops) < 3 or not self.tracks_uniformity() or
                getattr(self, 'kind', None) != 'cs' or not getattr(self, 'exec_known', False) or
                getattr(self, 'has_image_writes', True)):
            return ()
        past_entry = getattr(self, 'past_entry', None)
        if past_entry is None or (past_entry and (getattr(self, 'uniform_expr', None) is None or
                                                self.pc in getattr(self, 'loads_after_exec_write', ()))):
            return ()
        rsrc = reg_range(ops[2])
        first = self.sym.get(rsrc[1]) if rsrc and rsrc[0] == 's' and rsrc[2] - rsrc[1] == 7 and rsrc[2] <= 105 else None
        if (not first or first[0] != 'desc' or first[1][0] != 'table' or first[2] is None or
                first[3] % 8 or not names_one_value(first)):
            return ()
        masks = getattr(self, 'masks', ())
        varying_s = getattr(self, 'varying_s', ())
        modified = getattr(self.pipe, 'modified', ())
        other_modified = getattr(self.pipe, 'other_modified', ())
        stored = self.pipe.stored_images() if hasattr(self.pipe, 'stored_images') else (getattr(self.pipe, 'uavs', None) or ())
        key = desc_key(first)
        if key in modified or key in other_modified or key in stored:
            return ()
        for i in range(8):
            reg = rsrc[1] + i
            word = desc_word(first, first[3] + i)
            if (reg in masks or reg in varying_s or self.sym.get(reg) != word or
                    desc_key(word) in modified):
                return ()
        dims = [m for m in mods if m.startswith('dim:')]
        dmasks = [m for m in mods if m.startswith('dmask:')]
        if len(dims) != 1 or len(dmasks) != 1:
            return ()
        dim = dims[0][4:]
        if dim not in ('SQ_RSRC_IMG_2D', 'SQ_RSRC_IMG_3D', 'SQ_RSRC_IMG_2D_ARRAY'):
            return ()
        try:
            dmask = int(dmasks[0][6:], 0)
        except ValueError:
            return ()
        if not dmask or (dmask & ~0xf):
            return ()
        packed_2d = dim == 'SQ_RSRC_IMG_2D' and 'a16' in mods
        allowed = ('unorm', 'glc', 'slc', 'dlc') + (('a16',) if packed_2d else ())
        if any(not m.startswith(('dim:', 'dmask:')) and m not in allowed for m in mods):
            return ()
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            if not r or r[0] != 'v':
                return ()
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)]
        expected_addr = 1 if packed_2d else (3 if dim in ('SQ_RSRC_IMG_3D', 'SQ_RSRC_IMG_2D_ARRAY') else 2)
        if len(cs) != expected_addr:
            return ()
        if not all(re.fullmatch(r'v\d+', c) and int(c[1:]) <= 255 and self.uniform_source(c) for c in cs):
            return ()
        candidate = reg_range(ops[0])
        count = bin(dmask).count('1')
        if not candidate or candidate[0] != 'v' or candidate[2] > 255 or candidate[2] - candidate[1] + 1 != count:
            return ()
        return range(candidate[1], candidate[2] + 1)

    def one(self, name, asm):
        _, ops, mods = parse(asm)
        if writes_exec(name, ops):
            self.exec_generation += 1
        image = getattr(self, 'image_proof', None)
        if image and (self.pc in image.lowered or self.pc in image.loads):
            return self.lower_image_instruction(image, name, asm, ops, mods)
        if (getattr(self, 'vcc_lost', False) or
                (getattr(self, 'has_branches', False) and
                 ('vcc' in asm or name.startswith(IMPLICIT_VCC_READ)))):
            # Read operands even in handlers that refuse a mask as numeric data before calling u().
            # VOPC's two-operand spelling has an implicit destination, and v_cmpx e64 prints an unused SDST.
            if name.startswith('v_cmpx') and name.endswith('_e64') and len(ops) == 3:
                reads = scalar_tokens(' '.join(ops[1:]))
            elif name.startswith('v_cmp') and len(ops) == 2:
                reads = scalar_tokens(' '.join(ops))
            else:
                reads, _, _ = scalar_effects(name, asm)
            if reads & {'vcc', 'vcc_hi'}:
                self.read_vcc()
        # v_bfm_b32 exists only as VOP3 (opcode 0x363); a VOP2 spelling is not an instruction.
        if name.endswith('_e32') and (name[:-4] in VOP3_ONLY_VEC_U or name[:-4] == 'v_bfm_b32'):
            raise Unsupported(name)
        base = re.sub(r'_e(32|64)$', '', name)
        # Compute the proof before lowering, since a destination can also be a source.
        dst = None
        self._uniform_written = set()
        if self.tracks_uniformity() and hasattr(self, 'uniform_expr'):
            wrote_s = {r for r in self.forget_writes(name, ops, drop_since=False) if isinstance(r, int)}
        else:
            wrote_s = set()
        # Unit probes build a Stage without __init__: without the tracking state, prove nothing.
        # Past the entry block of a program with branches the read recomputes the defining expression, so the
        # proof no longer needs a known exec (see op_v_readfirstlane_b32).
        if (base in UNIFORM_VALU and ops and self.tracks_uniformity() and
                (getattr(self, 'past_entry', False) and hasattr(self, 'uniform_expr') or
                 not getattr(self, 'past_entry', True) and self.uniform_entry_exec()) and
                (not mods or (base in VEC_F or base in VEC_U_TO_F) and
                 all(m in ('clamp', 'mul:2', 'mul:4', 'div:2') for m in mods))):
            candidate = reg_range(ops[0])
            if candidate and candidate[0] == 'v' and candidate[1] == candidate[2]:
                sources = ops[1:]
                if base in VEC_F:
                    sources = [re.sub(r'^-(?=[^\d.])', '', tok).strip('|') for tok in sources]
                if sources and all(self.uniform_source(tok) for tok in sources):
                    dst = candidate[1]
        # A compute load through a read-only descriptor (a descriptor is never both loaded and stored) with a uniform
        # address reads the same dword in every lane. In the entry block with known exec every lane ran it.
        load_form = (dst is None and base in UNIFORM_LOADS and len(ops) == 4 and self.tracks_uniformity() and
                     getattr(self, 'kind', None) == 'cs' and
                     all(m in ('idxen', 'offen', 'glc', 'slc', 'dlc') or m.startswith('offset:') for m in mods) and
                     not ('idxen' in mods and 'offen' in mods) and (ops[1] == 'off') == ('idxen' not in mods and 'offen' not in mods) and
                     (ops[1] == 'off' or self.uniform_source(ops[1])) and self.uniform_source(ops[3]))
        loaded = load_form and not getattr(self, 'past_entry', True) and self.uniform_entry_exec()
        # With empty EXEC, READFIRSTLANE uses the old VGPR in lane zero. An unguarded reload
        # cannot reproduce it. Require the entry mask until the path proves a nonempty wave.
        vaddr = reg_range(ops[1]) if load_form and ops[1] != 'off' else None
        exprs = getattr(self, 'uniform_expr', None)
        # A uniform address gives every active lane the same words, so the first active lane's value is the recomputed
        # temporary whatever lanes a v_cmpx removed (an empty wave observes nothing); any other EXEC write ends the proof.
        recomputed = (load_form and getattr(self, 'past_entry', False) and exprs is not None
                      and (getattr(self, 'entry_exec_proof', None) is not None and self.uniform_entry_exec()
                           or getattr(self, 'exec_known', False) and self.pc not in self.loads_after_exec_write
                           or getattr(self, 'exec_narrow_only', False) and self.pc not in self.loads_after_wide_exec_write))
        loaded_regs = ()
        if loaded or recomputed:
            candidate = reg_range(ops[0])
            # Exactly the registers the opcode writes, as many as its name says.
            count = UNIFORM_LOAD_WORDS[base]
            if candidate and candidate[0] == 'v' and candidate[2] <= 255 and candidate[2] - candidate[1] + 1 == count:
                dst, loaded_regs = candidate[1], range(candidate[1], candidate[2] + 1)
            else:
                loaded = recomputed = False
        if dst is None:
            img_regs = self.uniform_image_load_regs(base, ops, mods)
            if img_regs:
                dst, loaded_regs = img_regs.start, img_regs
                loaded = not getattr(self, 'past_entry', True)
                recomputed = getattr(self, 'past_entry', False)
        # A shift right of a value built from the X thread id that leaves no thread-id bit.
        shifted = None
        if (dst is None and base == 'v_lshrrev_b32' and not mods and len(ops) == 3 and
                self.tracks_uniformity() and hasattr(self, 'uniform_expr') and getattr(self, 'kind', None) == 'cs' and
                not getattr(self, 'past_entry', True) and getattr(self, 'exec_known', False)):
            candidate = reg_range(ops[0])
            shifted = self.tid_shift(ops[1], ops[2])
            if shifted is not None and candidate and candidate[0] == 'v' and candidate[1] == candidate[2]:
                dst = candidate[1]
            else:
                shifted = None
        extra_proof = getattr(self, 'firstlane_cfg', False) and self.uniform_cf
        extra_regs = ()
        ps_load_regs = ()
        entry_proof = (extra_proof and self.uniform_entry_exec()
                       and (self.firstlane_ps_exec is None or self.pc in self.firstlane_ps_exec))
        pixel_full = extra_proof and self.pc in getattr(self, 'firstlane_ps_full_exec', ())
        if entry_proof or pixel_full:
            # Table-driven VALU operations use only their explicit operands and, for the
            # accumulator forms, the old destination. No lane selectors or mask operands.
            if entry_proof and (base in VEC_U or base in VEC_F or base in VEC_U_TO_F or base in UNIFORM_VALU):
                dst_reg = reg_range(ops[0]) if ops else None
                inputs = ops if base in ('v_fmac_f32', 'v_mac_f32') else ops[1:]
                if (dst_reg and dst_reg[0] == 'v' and dst_reg[1] == dst_reg[2] and inputs
                        and all(m in ('clamp', 'mul:2', 'mul:4', 'div:2') for m in mods)
                        and all(self.firstlane_source(tok, base) for tok in inputs)):
                    extra_regs = (dst_reg[1],)
            elif (getattr(self, 'kind', None) == 'ps' and base in UNIFORM_LOADS
                  and base.startswith('buffer_load_format_') and len(ops) == 4):
                data, desc = reg_range(ops[0]), reg_range(ops[2])
                first = self.sym.get(desc[1]) if desc else None
                # MUBUF lowering checks the four descriptor words and, outside the entry block,
                # proves that the table pointer and descriptor load reach this use.
                if (data and data[0] == 'v' and first and first[0] == 'desc' and first[1][0] == 'table'
                        and ops[1] == 'off' and self.scalar_const(ops[3]) is not None
                        and self.firstlane_source(ops[3], base)):
                    regs = range(data[1], data[2] + 1)
                    if entry_proof:
                        extra_regs = regs
                    else:
                        ps_load_regs = regs
            elif entry_proof and getattr(self, 'kind', None) == 'cs' and base in UNIFORM_LOADS and len(ops) == 4:
                data = reg_range(ops[0])
                if (data and data[0] == 'v' and data[2] - data[1] + 1 == UNIFORM_LOAD_WORDS[base]
                        and (ops[1] == 'off' or self.firstlane_source(ops[1], base))
                        and self.firstlane_source(ops[3], base)
                        and not self.pipe.rw_keys):
                    extra_regs = range(data[1], data[2] + 1)
        if recomputed:
            # Read before the load can overwrite its own address register.
            self.uniform_vaddr = '(%s)' % exprs[vaddr[1]] if vaddr and vaddr[1] in exprs else None
            self.uniform_temps = {}
        self.last_v = None
        if hasattr(self, 'heap_index'):  # unit probes build a Stage without the tracking state
            # Any write ends a register's mark; the lowering below sets the new one, if any.
            wrote = scalar_destinations(name, asm)
            wrote = set(self.heap_index) if 'any' in wrote else wrote & set(self.heap_index)
            self.heap_index_was = {r: self.heap_index.pop(r) for r in wrote}
        try:
            term = self.lower(base, asm, ops, mods)
        finally:
            temps = getattr(self, 'uniform_temps', None)
            self.uniform_vaddr = self.uniform_temps = None
        if extra_proof:
            for token in ops[:2] if base == 'v_swap_b32' else destinations(base, ops):
                reg = reg_range(token)
                if reg and reg[0] == 'v':
                    self.firstlane_v.difference_update(range(reg[1], reg[2] + 1))
            self.firstlane_v.update(extra_regs)
            self.firstlane_ps_load_v.update(ps_load_regs)
        if getattr(self, 'entry_exec_proof', None) is not None:
            for token in (ops[:2] if base == 'v_swap_b32' else ops[:1]):
                reg = reg_range(token)
                if reg and reg[0] == 'v':
                    self.entry_written_v.difference_update(range(reg[1], reg[2] + 1))
            if dst is not None and self.uniform_entry_exec() and (base in UNIFORM_VALU or base in UNIFORM_LOADS):
                self.entry_written_v.update(loaded_regs or (dst,))
        uniform_written = self._uniform_written
        if self.tracks_uniformity() and getattr(self, 'since_join', None) is not None and wrote_s:
            self.since_join -= (wrote_s - uniform_written)
        if wrote_s and self.bad_sgprs is not None:
            for r in wrote_s - uniform_written:
                sym = self.sym.get(r)
                if (r in self.varying_s or r in getattr(self, 'masks', ()) or
                        (sym is not None and sym[0] not in ('const', 'ud', 'fixed_buffer_word')) or
                        (self.defined_uniform_s is not None and r not in self.defined_uniform_s)):
                    if self.defined_uniform_s is not None:
                        self.defined_uniform_s.discard(r)
                    self.bad_sgprs.add(r)
        if dst is not None:
            self.uniform_v.update(loaded_regs or (dst,))
            for reg in loaded_regs:
                getattr(self, 'uniform_expr', {}).pop(reg, None)
            if hasattr(self, 'uniform_expr'):
                # Proven sources are scalars, constants and VGPRs; each VGPR needs its own current expression.
                expr = self.last_v[1] if self.last_v and self.last_v[0] == dst else None
                vs = [int(n) for n in re.findall(r'\bv(\d+)\b', expr or '')]
                if shifted is not None:
                    self.uniform_expr[dst] = shifted
                elif expr is None or loaded_regs or any(n not in self.uniform_expr for n in vs):
                    self.uniform_expr.pop(dst, None)
                elif sum(len(self.uniform_expr[n]) for n in vs) > UNIFORM_EXPR_LIMIT:
                    # A VGPR used as both operands (v10 = v10 + v10) doubles the text each time: past the limit the
                    # chain has no recomputable expression, so a first-lane read of it is refused.
                    self.uniform_expr.pop(dst, None)
                else:
                    self.uniform_expr[dst] = re.sub(r'\bv(\d+)\b', lambda m: '(%s)' % self.uniform_expr[int(m[1])],
                                                    expr)
                if recomputed:
                    self.uniform_expr.update(temps)
        return term

    def quad_flow_straight(self):
        """Whether a DPP quad read sits in straight-line code (the quad intrinsics then run in uniform flow)."""
        return not getattr(self, 'has_branches', True) and not getattr(self, 'blocks', [])

    def quad_flow_allowed(self):
        """Whether a DPP quad read in a pixel program with branches or loops may be lowered now and proven over the
        whole program after translation (Stage.quad_flow_refusals). Unit probes without translate() cannot."""
        return getattr(self, 'kind', None) == 'ps' and getattr(self, 'quad_flow_reads', None) is not None

    def quad_exec_cause(self):
        """Why a DPP quad read meets an EXEC that is not the entry mask or a program that is not straight-line, for the
        refusal text (the refusal itself is unchanged): the stage when it is neither a pixel nor a compute stage, the last
        instruction before the read that can write EXEC (writes_exec), or, for a compute program, its first branch, so a
        tester report names the shape to lift. Empty when none applies."""
        kind = getattr(self, 'kind', None)
        if kind not in ('ps', 'cs'):
            return ': %s stage' % (kind or 'unknown')
        here = getattr(self, 'pc', None)
        before = [(off, name, asm) for off, _n, name, asm in getattr(self, 'ins', ()) if here is None or off < here]
        if not getattr(self, 'exec_known', False):
            writer = next(((off, name) for off, name, asm in reversed(before) if writes_exec(name, parse(asm)[1])), None)
            return ': EXEC written at +0x%x by %s' % writer if writer else ': EXEC not known at entry'
        if kind == 'cs' and not self.quad_flow_straight():
            branch = next(((off, name) for off, _n, name, _a in getattr(self, 'ins', ())
                           if name == 's_branch' or name.startswith('s_cbranch')), None)
            return ': branch at +0x%x (%s)' % branch if branch else ': not straight-line'
        return ''

    def note_quad_flow(self, base, allow_late_wqm=False):
        if not self.quad_flow_straight() and getattr(self, 'quad_flow_reads', None) is not None:
            self.quad_flow_reads.append((self.pc, base, allow_late_wqm))

    def entry_wqm_exec_sites(self):
        """Track the entry WQM mask through reachable blocks, intersecting facts at joins and back edges."""
        bounds, _, succ, _ = self.spill_cfg
        if not bounds:
            return set()
        # Only the entry block can establish WQM. Later s_wqm instructions cannot
        # recover an EXEC write that this translator does not model on helper lanes.
        entry_end = bounds[0][1]
        entry_wqm = None
        clobbers = set()
        wave32 = wave32_exec_proof(self.ins)
        for i, (_off, _n, name, asm) in enumerate(self.ins):
            _, ops, _ = parse(asm)
            full_wqm = (name == 's_wqm_b64' and ops == ['exec', 'exec'] or
                        wave32 and name == 's_wqm_b32' and ops in (['exec', 'exec'], ['exec_lo', 'exec_lo']))
            if full_wqm:
                if i < entry_end and entry_wqm is None and not clobbers:
                    entry_wqm = i
            elif writes_exec(name, ops) or name == 'exp' or name.startswith(('s_swappc', 's_call')):
                clobbers.add(i)
        if entry_wqm is None or any(0 in targets for targets in succ):
            return set()

        incoming, pending, sites = {0: False}, {0}, set()
        while pending:
            block = pending.pop()
            whole = incoming[block]
            lo, hi = bounds[block]
            for i in range(lo, hi):
                off = self.ins[i][0]
                if whole:
                    sites.add(off)
                else:
                    sites.discard(off)
                if i == entry_wqm:
                    whole = True
                elif i in clobbers:
                    whole = False
            for target in succ[block]:
                merged = incoming.get(target, True) and whole
                if incoming.get(target) != merged:
                    incoming[target] = merged
                    pending.add(target)
        return sites

    def quad_subrev_block_proof(self, base):
        return (base == 'v_subrev_f32' and self.quad_flow_allowed()
                and not getattr(self, 'fetched', False)
                and self.pc in getattr(self, 'entry_wqm_sites', ()))

    def quad_flow_refusals(self, branches, targets):
        """[(offset, reason)] for DPP quad reads lowered in a pixel program with branches or loops.

        The one-lane model runs each lane as its own HLSL thread through the pc dispatch, so the QuadReadLaneAt /
        QuadReadAcross* of a lane reads its neighbours only when all four threads of the quad reach that statement
        together (same block, same pass of the dispatch loop) with the word the guest lanes hold. The guest runs
        the wave as one: its scalar branches are wave-wide, and the quad is whole when EXEC is the s_wqm mask. A
        read is kept when, on every path to it:
        - the s_wqm exec, exec that proved whole-quad EXEC ran first: it precedes every branch and target,
          or dominates a derivative read reached through uniform control flow and a scalar-only prefix;
        - no other instruction that can write EXEC reaches it (so EXEC is still that mask and every VGPR write ran
          on all four lanes, helper lanes included);
        - every conditional branch that reaches it is wave-uniform (SCC proven uniform by the translator), or an
          execz/execnz branch, whose EXEC is the whole-quad mask by the rule above and so never taken. A vccz/vccnz
          branch, or SCC from lane data, can split a quad, and the read is refused.
        Then the four threads ran the same branch outcomes, so they are in the same block on the same dispatch
        pass. No wave size is assumed: the SM 6.0 quad operations need only the 2x2 quad of four lanes."""
        reads = getattr(self, 'quad_flow_reads', [])
        if not reads:
            return []
        index, block_of, succ, dom, _writes, _reached = self.flow
        after = {}

        def reaches(src, dst):
            a, b = index[src], index[dst]
            if block_of[a] == block_of[b] and a < b:
                return True
            start = block_of[a]
            if start not in after:
                seen, todo = set(), list(succ[start])
                while todo:
                    blk = todo.pop()
                    if blk not in seen:
                        seen.add(blk)
                        todo += succ[blk]
                after[start] = seen
            return block_of[b] in after[start]
        # No branch or target (a dead read behind an early s_endpgm): nothing for the s_wqm to precede.
        first = min([off for off, _ in branches] + list(targets), default=float('inf'))
        writers = [off for off, _n, name, asm in self.ins if writes_exec(name, parse(asm)[1])
                   and not (re.sub(r'_e(32|64)$', '', name) in ('s_wqm_b32', 's_wqm_b64')
                            and parse(asm)[1] in (['exec', 'exec'], ['exec_lo', 'exec_lo']))]
        wqm = self.wqm_set_at
        early_wqm = wqm is not None and wqm < first

        def wqm_dominates(off):
            if wqm is None:
                return False
            wqm_block, block = block_of[index[wqm]], block_of[index[off]]
            return wqm < off if wqm_block == block else wqm_block in dom[block]

        # Before WQM an exact-EXEC VGPR write can leave helpers with stale data.
        # Only extend the proof through a scalar prefix, including its branches.
        pre_wqm_vector_write = None
        if wqm is not None and not early_wqm:
            pre_wqm_vector_write = next(
                (off for off, _, name, asm in self.ins if reaches(off, wqm) and not wqm_dominates(off)
                 and any((reg_range(dst) or (None,))[0] == 'v' for dst in destinations(name, parse(asm)[1]))),
                None)
        out = []
        for at, base, allow_late_wqm in reads:
            if at in self.whole_quad_reads:
                split = next((b for b in sorted(self.divergent_branches) if reaches(b, at)), None)
                if split is not None:
                    out.append((at, '%s_dpp quad read at +0x%x: %s at +0x%x is not proven wave-uniform'
                                % (base, at, self.divergent_branches[split], split)))
                continue
            # A derivative can also enter WQM after a uniform branch. Require that
            # the WQM instruction dominates the read, including order within its block.
            late_wqm = not early_wqm and allow_late_wqm and wqm_dominates(at)
            if not (early_wqm or late_wqm):
                out.append((at, '%s_dpp quad read at +0x%x in a program with branches: the s_wqm exec, exec does not '
                                'run before every branch' % (base, at)))
                continue
            writer = next((w for w in writers if reaches(w, at)), None)
            if writer is not None:
                out.append((at, '%s_dpp quad read at +0x%x in a program with branches: the EXEC write at +0x%x reaches '
                                'it' % (base, at, writer)))
                continue
            split = next((b for b, name in sorted(self.divergent_branches.items())
                          if (name not in ('s_cbranch_execz', 's_cbranch_execnz')
                              or (late_wqm and not wqm_dominates(b))) and reaches(b, at)), None)
            if split is not None:
                out.append((at, '%s_dpp quad read at +0x%x in a program with branches: %s at +0x%x is not proven '
                                'wave-uniform' % (base, at, self.divergent_branches[split], split)))
            elif late_wqm and pre_wqm_vector_write is not None:
                out.append((at, '%s_dpp quad read at +0x%x: VGPR write at +0x%x before whole-quad EXEC'
                            % (base, at, pre_wqm_vector_write)))
        return out

    def dpp_pair(self, base, asm, ops, mods):
        """A DPP16 spelling whose SRC0 reads two selected lanes of the quad.

        RDNA 2 ISA section 12.19 (DPP16): lane i of a quad reads SRC0 of lane quad_perm[i] of the same quad. A pair
        broadcast ([1,1,3,3], [0,0,2,2], [2,3,2,3], [0,1,0,1]) moves one lane of every horizontal or vertical pair
        into the other and leaves the first lane alone. A projected exchange selects one row or column for both
        pairs, so a subtraction can use the same edge throughout the quad. Each lane is read with the shader
        model 6.0 QuadReadLaneAt (its lane argument must be a constant, so all four reads are unconditional) and this
        lane keeps the one quad_perm names for its own position in the quad. That needs all four lanes of the quad to
        run the instruction in the same flow, with a whole-quad EXEC mask established by s_wqm exec, exec.
        Branching code must pass quad_flow_refusals."""
        perms = [m for m in mods if m.startswith('quad_perm:')]
        rest = [m for m in mods if not m.startswith('quad_perm:')]
        if len(perms) != 1:
            raise Unsupported('%s_dpp with %s' % (base, ' '.join(mods)))
        for m in rest:
            if m not in ('row_mask:0xf', 'bank_mask:0xf', 'bound_ctrl:0', 'bound_ctrl:1'):
                raise Unsupported('%s_dpp with %s' % (base, m))
        # With all four lanes active no source lane is disabled, so bound_ctrl has nothing to replace.
        perm = tuple(int(c) for c in re.findall(r'\d', perms[0]))
        if perm not in DPP_PAIR and not (perm in DPP_COARSE and base in DPP_DERIVATIVE_OPS):
            raise Unsupported('%s_dpp with %s' % (base, perms[0]))
        if (self.kind != 'ps' or not (self.quad_flow_straight() or self.quad_flow_allowed()) or getattr(self, 'fetched', False) or
                (not (getattr(self, 'exec_known', False) and getattr(self, 'wqm_exec', False))
                 and not self.quad_subrev_block_proof(base))):
            raise Unsupported('%s_dpp %s outside straight-line pixel code with a whole-quad exec mask (s_wqm exec, exec)'
                              % (base, perms[0]))
        if len(ops) < 2:
            raise Unsupported('%s_dpp with %d operands' % (base, len(ops)))
        # The permuted word goes to a scratch register (v256, beyond the 256 hardware registers), so a second source
        # naming the same register, or a destination that is the source, still reads the right lane.
        src = re.fullmatch(r'([-|]*)v(\d+)(\|?)', ops[1])
        if not src:
            raise Unsupported('%s_dpp source %s' % (base, ops[1]))
        word = self.u('v%s' % src.group(2))  # records the original source as a used VGPR
        self.note_quad_flow(base, allow_late_wqm=base in DPP_DERIVATIVE_OPS)
        self.needs_sm6 = True
        self.used_v.add(DPP_SCRATCH)
        self.uniform_v.discard(DPP_SCRATCH)
        if isinstance(self.align4_v, set):
            self.align4_v.discard(DPP_SCRATCH)
        self.emit('{ uint x2w_q0 = QuadReadLaneAt(%s, 0), x2w_q1 = QuadReadLaneAt(%s, 1),'
                  ' x2w_q2 = QuadReadLaneAt(%s, 2), x2w_q3 = QuadReadLaneAt(%s, 3);'
                  ' uint x2w_lane = ((uint(i.pos.y) & 1u) << 1u) | (uint(i.pos.x) & 1u);'
                  ' v%d = x2w_lane == 0u ? x2w_q%d : x2w_lane == 1u ? x2w_q%d : x2w_lane == 2u ? x2w_q%d : x2w_q%d; }'
                  % ((word,) * 4 + (DPP_SCRATCH,) + perm))
        ops = [ops[0], '%sv%d%s' % (src.group(1), DPP_SCRATCH, src.group(3))] + ops[2:]
        return self.lower(base, asm, ops, [])

    def forget_writes(self, name, ops, drop_since=True):
        """Before an instruction runs: an EXEC write ends every recomputable expression, and an SGPR write ends the
        expressions that read it and (when drop_since is set) that SGPR's since-join mark (set_s marks it again when it proves the new value).
        The VGPR proofs stay: the entry-block read copies the VGPR, which neither write changes."""
        if writes_exec(name, ops):
            self.uniform_expr.clear()
            if getattr(self, 'entry_exec_proof', None) is not None:
                self.uniform_v.clear()
        regs = set(range(128)) if name.startswith('s_movreld') else set()  # M0 picks the destination
        for tok in destinations(name, ops):
            r = reg_range(tok)
            if r and r[0] == 's':
                regs.update(range(r[1], r[2] + 1))
        if not regs:
            return set()
        if drop_since and self.since_join is not None:
            self.since_join -= regs
        for v in [v for v, expr in self.uniform_expr.items()
                  if regs & {int(n) for n in re.findall(r'\bs(\d+)\b', expr)}]:
            del self.uniform_expr[v]
        return regs

    MASK_NAMES_LOW = ('vcc', 'vcc_lo', 'exec', 'exec_lo')
    MASK_NAMES_HIGH = ('vcc', 'vcc_hi', 'exec', 'exec_hi')

    def mask_source_tags(self, base, ops):
        """(low, high): whether a source operand of the instruction carries a lane-mask word. low = a mask's low word
        (vcc/exec, an SGPR in self.masks, a pair that starts in one); high = the high word of a wave64 mask (vcc_hi,
        exec_hi, a pair holding one, an SGPR in self.mask_high). Read before the instruction runs: it may overwrite
        its own source."""
        dests = destinations(base, ops)
        srcs = ops[len(dests):] if dests and ops[:len(dests)] == dests else list(ops)
        masks, high_regs = getattr(self, 'masks', ()), getattr(self, 'mask_high', ())
        low = high = False
        for tok in (t.strip().lstrip('-|').rstrip('|') for t in srcs):
            low |= tok in self.MASK_NAMES_LOW
            high |= tok in self.MASK_NAMES_HIGH and tok not in ('vcc_lo', 'exec_lo')
            r = reg_range(tok)
            if r and r[0] == 's':
                start_tagged = r[1] in masks
                low |= start_tagged
                high |= any(n in high_regs for n in range(r[1], r[2] + 1)) or (start_tagged and r[2] > r[1])
        return low, high

    def propagate_mask_tags(self, base, ops, tags):
        """After an instruction ran: the SGPRs it wrote carry the lane-mask tag of what they were made from. This is the one
        place the tag moves. A compare or carry writes its single-word mask itself (the decoder
        names a wave64 compare destination s0, not s[0:1]); a mask-only scalar op (mov, bitwise, cselect, saveexec)
        copies the tag of its sources to its destination, both words of a b64 pair; any other write cleared the tag
        already (set_s). self.masks holds the low word, self.mask_high the high word of a wave64 pair: no numeric read of
        either is lowered (lower())."""
        low, high = tags
        self.tag_compare_pair(base, ops)
        saveexec = base.startswith('s_') and 'saveexec' in base
        if MASK_COPY_SALU_RE.match(base) and not base.startswith('s_wqm') and not (low or high):
            # (s_wqm is a no-op in _lower(): it writes nothing, so the old mask stays in its destination.)
            # a 32-bit copy of plain data: the destination no longer holds a lane mask, whatever an earlier path or join
            # left. A 64-bit pair copy keeps its tag: scalar() lowers every b64 move as a mask copy (the lane bit only),
            # so the truncated word must stay refused as numeric data.
            for tok in destinations(base, ops):
                r = reg_range(tok)
                if r and r[0] == 's' and r[2] == r[1]:
                    self.masks.discard(r[1])
        scalar = bool(MASK_COPY_SALU_RE.match(base)) and (low or high)
        pair_select = base.startswith('s_cselect') and base.endswith('b64')
        if not (saveexec or scalar or pair_select):
            return
        for tok in destinations(base, ops):
            r = reg_range(tok)
            if not r or r[0] != 's':
                continue
            if r[2] > r[1]:   # a pair: low word, and the high word it carries
                self.masks.add(r[1])
                self.mask_high = getattr(self, 'mask_high', set()) | {r[1] + 1}
            elif low or saveexec or pair_select:
                self.masks.add(r[1])
            if r[2] == r[1] and high and not (saveexec or pair_select):
                self.mask_high = getattr(self, 'mask_high', set()) | {r[1]}

    def tag_compare_pair(self, base, ops):
        """A compare or carry-out into SGPR sN (the decoder prints a wave64 pair as s[N:N+1] by its low word only). In
        wave64 the destination is an even-aligned pair and s(N+1) holds the other 32 lanes, which no one-lane value
        has: tag s(N+1) as the high word. An odd sN is a wave32 destination (no high word); an even one that is a wave32
        destination loses the numeric reads of s(N+1), a refusal and never a wrong value. If s(N+1) already holds its own lane
        mask (a wave32 compare or carry in the neighbouring register) the destination is not a pair: nothing is tagged."""
        if base.startswith('v_cmp') and not base.startswith('v_cmpx'):
            tok = ops[0] if ops else ''
        elif VOP3B_SDST.match(base) and len(ops) > 1:
            tok = ops[1]
        else:
            return
        r = reg_range(tok.strip())
        if r and r[0] == 's' and r[1] == r[2] and r[1] in self.masks and r[1] % 2 == 0 and r[1] + 1 not in self.masks:
            self.mask_high = getattr(self, 'mask_high', set()) | {r[1] + 1}

    def lane_mask_exact(self):
        """True where the one-lane model of a wave mask is exact: a compute group of one thread has one participating lane
        per wave, so the lane's bit is the whole mask and its high word is 0. Anywhere else the mask holds other lanes'
        bits that a numeric read would see."""
        group = getattr(getattr(self, 'pipe', None), 'group', None)
        return getattr(self, 'kind', None) == 'cs' and bool(group) and group[0] * group[1] * group[2] == 1

    def reads_mask_high(self, toks):
        """First single-register operand that is the high word of a wave64 lane mask, else None."""
        high = getattr(self, 'mask_high', ())
        for t in toks:
            r = reg_range(t.strip().lstrip('-|').rstrip('|'))
            if r and r[0] == 's' and r[1] == r[2] and r[1] in high:
                return t.strip()
        return None

    def lower_mask_word_sop2(self, base, ops, mods):
        """Lower whole-word reads of entry EXEC and straight-line compare masks."""
        if (base not in MASK_WORD_SOP2 or mods or len(ops) != 3 or getattr(self, 'kind', None) != 'cs'
                or getattr(self, 'has_branches', False) or not getattr(self, 'exec_known', False)):
            return False
        # SM 6.0 guarantees at least four lanes per host wave. The group fits both the host
        # wave and the guest's low word. Use logical thread IDs so host lane numbering is irrelevant.
        group = getattr(getattr(self, 'pipe', None), 'group', None)
        if group not in ((2, 1, 1), (3, 1, 1), (4, 1, 1)):
            return False
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            return False
        sources = ops[1:]
        masks = [self.is_lane_mask(tok) for tok in sources]
        if not any(masks) or (base == 's_nor_b32' and all(masks)):
            return False  # keep the existing all-mask NOR projection
        if self.reads_mask_high(sources):
            return False
        keys = [tok.removesuffix('_lo') for tok in sources]
        # A tagged bit alone is insufficient. Only entry EXEC and a compare issued with
        # entry EXEC prove that bits for absent lanes are zero.
        proven = {'exec'}
        for off, _, name, asm in self.ins:
            if off >= self.pc:
                break
            operands = parse(asm)[1]
            if writes_exec(name, operands) or 'movrel' in name or name.startswith(('s_setpc', 's_swappc')):
                return False
            _, writes, may = scalar_effects(name, asm)
            proven.difference_update(may)
            if name.startswith('v_cmp'):
                for word in writes - {'vcc_hi', 'exec_hi'}:
                    reg = reg_range(word)
                    if reg:
                        if reg[1] % 2 == 0 and getattr(self.pipe, 'wave_size', None) != 32:
                            # The decoder may print only the low destination of a wave64 compare.
                            proven.discard('s%d' % (reg[1] + 1))
                        if reg[2] != reg[1]:
                            continue
                    proven.add(word)
        if any(key not in proven if flag else not self.uniform_source(tok)
               for tok, flag, key in zip(sources, masks, keys)):
            return False
        if any(not flag and (reg := reg_range(tok)) and
               (reg[0] != 's' or reg[1] != reg[2] or reg[1] > 105)
               for tok, flag in zip(sources, masks)):
            return False

        values = [self.u(tok) for tok in sources]
        for i, flag in enumerate(masks):
            if flag:
                temp = 'x2w_mask_%x_%d' % (self.pc, i)
                self.emit('uint %s = x2w_ballot(%s, tid.x);' % (temp, values[i]))
                values[i] = temp
        self.uses_mask_ballot = True
        expr = SCA[base][0].format(*values)
        d = dst[1]
        if base == 's_nor_b32':
            word = 'x2w_nor_%x' % self.pc
            self.emit('uint %s = %s;' % (word, expr))
            self.set_scc('%s != 0u' % word)
            self.set_s(d, '(%s >> tid.x) & 1u' % word, uniform=False)
            self.masks.add(d)
        else:
            if base in SHIFT_ADD:
                self.set_scc(shift_add_carry(base, values, expr))
            self.set_s(d, expr)
        return True

    def lower(self, base, asm, ops, mods):
        """Lower one instruction; a vector ALU form that reads a lane mask as a number is refused.

        The model keeps one lane's bit of a wave mask. A vector ALU source that reads the mask as a number (the
        llvm.amdgcn.ballot word) sees every lane's bit on hardware, so the lowered word would be silently wrong.
        The mask-aware forms (select, carry, readlane) and v_mov_b32 (which the readfirstlane proofs handle) pass.
        The check runs after the form's own refusals, so those keep their messages."""
        if base == MESH_MASK_SELECT:
            return self.mesh_mask_select(ops)
        ballot_regs = getattr(self, 'mesh_ballot_regs', None)
        if ballot_regs:
            # These legacy mask consumers expect a single per-lane bit rather than a scalar mask word.
            if base.endswith('_b64'):
                mask_sources = scalar_registers(', '.join(ops[len(destinations(base, ops)):]))
            elif base == 'v_cndmask_b16' or base.endswith('_co_ci_u32'):
                low_words = scalar_registers(ops[-1]) if len(ops) >= 4 else set()
                mask_sources = low_words | {n + 1 for n in low_words}
            else:
                mask_sources = set()
            if mask_sources & ballot_regs:
                raise Unsupported('slot 57 mesh: %s on a materialized mask word is not covered' % base)
        self.current_asm = asm      # named by scc_refusal
        if self.wave_reduction(base, ops):
            return None
        if self.lower_mask_word_sop2(base, ops, mods):
            return None
        masks = []
        if base.startswith('v_') and not base.startswith('v_mov_b32') and base not in MASK_AWARE_VALU and not MASK_AWARE_VALU_RE.search(base):
            masks = [t for t in (o.strip().lstrip('-|').rstrip('|') for o in ops[1:]) if self.is_lane_mask(t)]
        if base.startswith('s_') and not MASK_AWARE_SALU_RE.match(base):
            masks = [t for t in (ops if base.startswith(('s_cmp', 's_bitcmp')) else ops[1:]) if self.is_lane_mask(t.strip())]
        if base.startswith(('v_', 's_')) and not masks:   # the high word of a wave64 mask has no one-lane value
            skip = 2 if VOP3B_SDST.match(base) else 1   # the carry-out of a VOP3b form is a destination, not a read
            high = self.reads_mask_high(ops if base.startswith(('s_cmp', 's_bitcmp')) else ops[skip:])
            masks = [high] if high else []
        tags = self.mask_source_tags(base, ops)
        result = self._lower(base, asm, ops, mods)
        for tok in destinations(base, ops):   # an overwritten register no longer holds a mask's high word
            r = reg_range(tok)
            if r and r[0] == 's':
                getattr(self, 'mask_high', set()).difference_update(range(r[1], r[2] + 1))
        if masks and not self.lane_mask_exact():
            raise Unsupported('%s reads lane mask %s as numeric data' % (base, masks[0]))
        self.propagate_mask_tags(base, ops, tags)
        return result

    def lower_image_instruction(self, proof, name, asm, ops, mods):
        pc = self.pc
        if pc in proof.captures:
            temp, operand = proof.captures[pc]
            self.emit('%s = %s;' % (temp, self.u(operand)))
        if pc in proof.loads:
            dst, addr = proof.loads[pc]
            index = ' + '.join(key for key, _, _ in addr.offsets)
            subscript = '(%s) >> 2' % index if index else '0u'
            self.set_v(dst, 'image_data_%x[%s]' % (pc, subscript))
            return None
        base = re.sub(r'_e(32|64)$', '', name)
        self.current_asm = asm
        self._uniform_written = set()
        for reg in proof.erased.get(pc, ()):
            if reg.startswith('v') and reg[1:].isdigit():
                n = int(reg[1:])
                self.align4_v.discard(n)
                self.clear_low16(n)
                getattr(self, 'uniform_expr', {}).pop(n, None)
                getattr(self, 'uniform_v', set()).discard(n)
            elif reg.startswith('s') and reg[1:].isdigit():
                n = int(reg[1:])
                self.sym.pop(n, None)
                self.masks.discard(n)
                self.mask_high.discard(n)
                self.align4_s.discard(n)
                self.varying_s.add(n)
        return self._lower(base, asm, ops, mods)

    def _lower(self, base, asm, ops, mods):
        pc = getattr(self, 'pc', None)
        calculated = getattr(self, 'calculated_pc', {})
        if pc in calculated:
            target = calculated[pc]
            if target is not None:
                return ('jump', target)
            # The proof checked that the code address is never read as data.
            # Do not materialize it as a fictional integer or keep an older
            # entry constant/resource/lane-mask proof for the overwritten SGPR.
            for reg in scalar_registers(ops[0]):
                self.sym.pop(reg, None)
                self.masks.discard(reg)
                if isinstance(self.align4_s, set):
                    self.align4_s.discard(reg)
                self.varying_s.add(reg)
            if base in ('s_add_u32', 's_addc_u32'):
                self.scc_uniform = False
                self.scc_verified = False
            return None
        if getattr(self, 'fetched', False):  # unit probes build a Stage without __init__
            # Behind a fetch call the definite-assignment guard relies on exec_known: every exec write must clear it.
            # s_wqm of exec into itself only adds lanes, so it stays a no-op; other s_wqm forms (a no-op here) and
            # exec writes outside the handlers that clear exec_known are refused. Other programs keep their handling.
            if base in ('s_wqm_b32', 's_wqm_b64') and not (ops[0] in ('exec', 'exec_lo') and ops[1:] == ops[:1]):
                raise Unsupported('%s %s after a fetch call' % (base, ', '.join(ops)))
            if ops and ops[0] in ('exec', 'exec_lo', 'exec_hi') and not base.startswith('s_wqm') and (
                    ops[0] == 'exec_hi' or not (base in SCA or base.startswith('v_cmp') or
                                                (base in ('s_cselect_b32', 's_cselect_b64') and ops[0] == 'exec'))):
                raise Unsupported('%s into %s after a fetch call' % (base, ops[0]))
        if base in ('s_wqm_b32', 's_wqm_b64'):
            if ops[0] in ('exec', 'exec_lo') and ops[1:] == ops[:1] and getattr(self, 'exec_known', False):
                self.wqm_exec = True   # every lane of a touched quad is now active, helper lanes included
                self.wqm_count = getattr(self, 'wqm_count', 0) + 1
                if getattr(self, 'wqm_set_at', False) is None:
                    self.wqm_set_at = getattr(self, 'pc', None)  # where the quad-read flow proof starts
            elif hasattr(self, 'lane_exec_breaks'):
                # Any other s_wqm is a no-op here, while the guest turns on the inactive lanes of every touched quad
                # (or writes a whole-quad mask somewhere): a lane's EXEC bit may be 0 where the guest's is 1.
                self.lane_exec_breaks.append((self.pc, '%s %s' % (base, ', '.join(ops))))
            if hasattr(self, 'uniform_v'):
                self.uniform_v.clear()
            if isinstance(self.align4_v, set):
                self.align4_v.clear()
            if hasattr(self, 'scc_uniform'):
                self.scc_uniform = False  # the hardware SCC result is not modelled
            self.scc_verified = False
            return None
        if base == 's_setreg_imm32_b32':
            # The write itself changes nothing the lowering models, except MODE[1:0] (the binary32 rounding mode):
            # v_cvt_pk_u8_f32 and the other conversions are lowered for round-to-nearest-even, and the hardware
            # follows MODE.round (round toward zero truncates). s_round_mode and s_setreg_b32 are refused already.
            field = re.fullmatch(r'hwreg\((HW_REG_MODE|1)(?:,\s*(\d+),\s*(\d+))?\)', ops[0].strip()) if ops else None
            if field and int(field.group(2) or 0) < 2:   # shortcut: an offset >= 2 cannot reach MODE[1:0]
                # The immediate is known: the write is harmless when it leaves MODE[1:0] at 0 (nearest even).
                offset, width = int(field.group(2) or 0), int(field.group(3) or 32)
                try:
                    value = int(ops[1], 0) & 0xffffffff
                except IndexError:
                    value = None
                except ValueError:
                    # The disassembler prints some inline constants as floats (1.0, 0.5): take their binary32 bits.
                    try:
                        value = struct.unpack('<I', struct.pack('<f', float(ops[1])))[0]
                    except (ValueError, OverflowError):
                        raise Unsupported('%s %s: immediate not decoded' % (base, ', '.join(ops)))
                if value is None:
                    raise Unsupported('%s %s: immediate not decoded' % (base, ', '.join(ops)))
                if ((value & ((1 << width) - 1)) << offset) & 3:
                    raise Unsupported('%s %s writes the binary32 rounding mode MODE[1:0]' % (base, ', '.join(ops)))
            if not ops or not ops[0].strip().startswith('hwreg('):
                raise Unsupported('%s %s: hardware register not decoded' % (base, ', '.join(ops)))
            return None
        # s_waitcnt_depctr only delays the wave until the named dependency counters reach their thresholds. LLVM
        # defines it with one immediate and no register definition, use or side effect (SOPInstructions.td,
        # S_WAITCNT_DEPCTR) and inserts it only as a hazard delay (GCNHazardRecognizer.cpp,
        # fixVMEMtoScalarWriteHazards). This model has no outstanding operation a counter could count.
        if base in ('s_inst_prefetch', 's_waitcnt', 's_waitcnt_depctr', 's_nop', 'v_nop', 's_clause', 's_sopp_0f',
                    's_code_end', 's_setprio'):
            return None
        if base == 's_endpgm':
            return ('end',)
        if base == 's_branch':
            return ('jump', int(ops[0].split('_')[1], 16))
        if base == 's_barrier' and getattr(self, 'kind', None) == 'cs' and self._lds_general_barrier_ok():
            if mods or ops not in ([], ['0'], ['0x0']):
                raise Unsupported('s_barrier with unexpected operands or modifiers')
            self._lds_require_alloc(base)
            self.emit('GroupMemoryBarrierWithGroupSync();')
            return None
        if base.startswith('s_cbranch_'):
            if base[10:].startswith('vcc'):
                self.read_vcc()
            if getattr(self, 'kind', None) == 'ms' and base in ('s_cbranch_scc0', 's_cbranch_scc1') and not self.scc_uniform:
                raise Unsupported('slot 57 mesh: a branch reads SCC that is not proven wave-uniform')
            cond = {'scc0': '!scc', 'scc1': 'scc', 'vccz': '(vcc & 1) == 0', 'vccnz': '(vcc & 1) != 0',
                    'execz': '(exec & 1) == 0', 'execnz': '(exec & 1) != 0'}[base[10:]]
            if getattr(self, 'pc', None) in getattr(self, 'wave_voted_branches', ()):
                # The guest skips the block when no lane of the wave has the bit; here every thread of the host wave
                # takes the same decision (convergence rule C2).
                self.needs_sm6 = True
                word = 'exec' if base[10:].startswith('exec') else 'vcc'
                bit = LE.EXEC_BIT if word == 'exec' else '(vcc & 1u) != 0u'
                # Keep the legacy vote spelling on the original translation path.
                zero = (base[10:] in ('execz', 'vccz') if getattr(self, 'retry_wave_reduction', False)
                        else base[10:].endswith('z'))
                cond = ('!' if zero else '') + 'WaveActiveAnyTrue(%s)' % bit
            elif base[10:] in ('execz', 'execnz') and getattr(self, 'pc', None) in getattr(self, 'voted_branches', ()):
                # The wave skips the block only when no lane is active; a quad with an active lane runs it whole, its
                # other lanes masked (their VGPRs kept for the derivatives, writes guarded by EXEC), as the guest does.
                self.uses_quad_vote = True
                cond = ('!' if base.endswith('execz') else '') + 'x2w_quad_any(exec)'
            return ('branch', cond, int(ops[0].split('_')[1], 16))
        if base == 'exp':
            return self.export(asm)
        if base in (*LDS_EXCHANGE_OPS, 's_barrier'):
            if getattr(self, 'kind', None) != 'cs':
                raise Unsupported(base)
            # The refusal census continues past failed instructions. Scan this program only once.
            if not hasattr(self, 'lds_error'):
                self.lds_error = None
                try:
                    if getattr(self, 'pipe', None) is None:
                        raise Unsupported('LDS exchange needs pipeline metadata')
                    self.lds_exchange_words = check_lds_exchange(self.ins, self.pipe.group, self.pipe.lds_words)
                    self.uses_lds = True
                except Unsupported as reason:
                    self.lds_error = str(reason)
            if self.lds_error is not None:
                raise Unsupported(self.lds_error)
            if base == 's_barrier':
                self.emit('GroupMemoryBarrierWithGroupSync();')
            else:
                write = base.startswith('ds_write')
                address = self.u(ops[0 if write else 1])
                offsets = lds_byte_offsets(base, mods)
                if base in ('ds_write_b32', 'ds_read_b32', 'ds_read2_b32'):
                    if write:
                        self.emit('lds[(%s + %du) >> 2] = %s;' % (address, offsets[0], self.u(ops[1])))
                    else:
                        if base == 'ds_read2_b32':
                            # Either destination can overlap the incoming address VGPR.
                            self.emit('a = %s;' % address)
                            address = 'a'
                        start = reg_range(ops[0])[1]
                        for index, offset in enumerate(offsets):
                            self.set_v(start + index, 'lds[(%s + %du) >> 2]' % (address, offset))
                else:
                    dwords = 4 if base.endswith('_b128') else 2
                    lo = _lds_vgpr_span(ops[1 if write else 0], dwords)
                    if write:
                        vals = [self.u('v%d' % (lo + k)) for k in range(dwords)]
                        self.emit('a = (%s + %du) >> 2;' % (address, offsets[0]))
                        for k, val in enumerate(vals):
                            self.emit('lds[%s] = %s;' % ('a' if k == 0 else 'a + %du' % k, val))
                    else:
                        self.emit('a = (%s + %du) >> 2;' % (address, offsets[0]))
                        for k in range(dwords):
                            self.set_v(lo + k, 'lds[%s]' % ('a' if k == 0 else 'a + %du' % k))
            return None
        if base.endswith('_dpp'):
            return self.dpp(base[:-len('_dpp')], ops, mods)
        handler = getattr(self, 'op_' + base, None)
        if handler:
            if mods and handler.__name__ in MODIFIER_FREE_HANDLERS:
                raise Unsupported('%s with %s' % (base, mods))
            return handler(ops, mods)
        if base in MUBUF_ATOMIC_OPS:
            return self.buffer_atomic(base, ops, mods)
        if base in LDS_DUAL_OPS:
            return self.lds_dual(base, ops, mods)
        if base in LDS_ATOMIC_OPS:
            return self.lds_atomic(base, ops, mods)
        if base.endswith('_dpp'):
            return self.dpp(base[:-len('_dpp')], ops, mods)
        if base.startswith('v_cmp'):
            # Before the VOP1/VOP2 SDWA route below: VOPC SDWA names also end in _sdwa.
            return self.compare(base, ops, mods)
        if base.endswith('_sdwa'):
            return self.sdwa(base[:-len('_sdwa')], ops, mods)
        if base in VEC_H:
            return self.half_alu(base, ops, mods)
        if base == 'v_ldexp_f32':
            # RDNA 2 ISA VOP3 v_ldexp_f32: D = S0.f * 2^S1.i, S1 a signed 32-bit exponent of any size. x2w_ldexp
            # (prelude) works on the binary32 fields, so every exponent, denormal and signed zero rounds like the product.
            if len(ops) != 3:
                raise Unsupported('%s with %d operands, expected 3' % (base, len(ops)))
            if any(m not in ('clamp', 'mul:2', 'mul:4', 'div:2') for m in mods):
                raise Unsupported('%s with %s' % (base, mods))
            dst = reg_range(ops[0])
            if not dst or dst[0] != 'v' or dst[1] != dst[2] or dst[1] > 255:
                raise Unsupported('%s destination %s' % (base, ops[0]))
            if ops[2].startswith(('-', '|')) and not re.fullmatch(r'-(?:0x[0-9a-fA-F]+|\d+)', ops[2]):
                raise Unsupported('%s exponent with a source modifier' % base)
            pipe = getattr(self, 'pipe', None)
            if pipe is not None:
                pipe.uses_ldexp = True
            return self.set_vf(dst[1], 'asfloat(x2w_ldexp(asuint(%s), asint(%s)))' % (self.f(ops[1]), self.u(ops[2])), mods)
        if base in VEC_F:
            arity, text = VEC_F[base]
            # A missing source used to raise IndexError and an extra one was dropped without a word.
            if len(ops) != arity + 1:
                raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
            args = [self.f(o) for o in ops[1:1 + arity]]
            if base in ('v_fmac_f32', 'v_mac_f32'):
                args.append(self.f(ops[0]))
            return self.set_vf(reg_range(ops[0])[1], text.format(*args), mods)
        if base in VEC_U_TO_F:
            arity, text_f = VEC_U_TO_F[base]
            if len(ops) != arity + 1:
                raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
            # Only the output modifiers a conversion can carry; op_sel and the source modifiers are not
            # accepted here, because ignoring them would silently change the operand.
            if any(m not in ('clamp', 'mul:2', 'mul:4', 'div:2') for m in mods):
                raise Unsupported('%s with %s' % (base, mods))
            dst = reg_range(ops[0])[1]
            low16 = base in ('v_cvt_f32_f16', 'v_cvt_f32_ubyte0', 'v_cvt_f32_ubyte1', 'v_cvt_off_f32_i4')
            self.set_vf(dst, text_f.format(*[self.u(o, low16_only=low16) for o in ops[1:1 + arity]]), mods)
            return None
        if base in VEC_U:
            arity, text = VEC_U[base]
            high = (False, False)
            if base == 'v_mad_u32_u16' and len(mods) == 1 and mods[0].startswith('op_sel:'):
                # RDNA 2 VOP3 op_sel: bit i set takes the high halfword of source i for a 16-bit operand. S2 is a
                # full 32-bit addend and the destination is 32 bits wide, so only bits 0 and 1 are meaningful here;
                # a set bit 2 or 3 stays refused.
                bits = re.fullmatch(r'op_sel:\[([01]),([01]),([01])(?:,([01]))?\]', mods[0].replace(' ', ''))
                if not bits or bits[3] == '1' or bits[4] == '1':
                    raise Unsupported('%s with %s' % (base, mods))
                high, mods = (bits[1] == '1', bits[2] == '1'), []
            if mods:
                raise Unsupported('%s with %s' % (base, mods))
            if len(ops) != arity + 1:
                raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
            if base in VOP3_ONLY_VEC_U:
                dst = reg_range(ops[0])
                if not dst or dst[0] != 'v' or dst[1] != dst[2] or dst[1] > 255:
                    raise Unsupported('%s destination %s' % (base, ops[0]))
                for src in ops[1:1 + arity]:
                    if self.is_lane_mask(src):
                        raise Unsupported('%s on a lane mask' % base)
                    sr = reg_range(src)
                    if sr is not None:
                        if sr[1] != sr[2] or (sr[0] == 'v' and sr[1] > 255) or (sr[0] == 's' and sr[1] > 105):
                            raise Unsupported('%s source %s' % (base, src))
                    elif src not in FLOATS and src not in ('m0', 'null'):
                        try:
                            val = int(src, 0)
                        except ValueError:
                            raise Unsupported('operand %r' % src)
                        if not -0x80000000 <= val <= 0xffffffff:
                            raise Unsupported('%s source %s' % (base, src))
            if base == 'v_and_b32':
                c0, c1 = self.const_u32(ops[1]), self.const_u32(ops[2])
                args = [self.u(ops[1], low16_only=c1 is not None and (c1 & 0xffff0000) == 0),
                        self.u(ops[2], low16_only=c0 is not None and (c0 & 0xffff0000) == 0)]
            elif base == 'v_lshlrev_b32':
                shift = self.const_u32(ops[1])
                args = [self.u(ops[1]), self.u(ops[2], low16_only=shift is not None and (shift & 31) >= 16)]
            elif base == 'v_mad_u32_u16':
                # S0 and S1 read only bits [15:0] (or [31:16] with op_sel); S2 reads all 32 bits.
                args = [self.u(ops[1], low16_only=not high[0]),
                        self.u(ops[2], low16_only=not high[1]),
                        self.u(ops[3])]
                text = '(%s & 0xffffu) * (%s & 0xffffu) + {2}' % (
                    '({0} >> 16u)' if high[0] else '{0}', '({1} >> 16u)' if high[1] else '{1}')
            else:
                args = [self.u(o) for o in ops[1:1 + arity]]
            if base in ('v_sad_u8', 'v_sad_hi_u8', 'v_sad_u16', 'v_sad_u32', 'v_msad_u8'):
                pipe = getattr(self, 'pipe', None)
                if pipe is not None and isinstance(getattr(pipe, 'uses_sad', None), set):
                    pipe.uses_sad.add(base)
            align4 = self.valu_align4(base, ops[1:1 + arity])
            return self.set_v(reg_range(ops[0])[1], text.format(*args), align4=align4)
        if base in SCA:
            return self.scalar(base, ops, mods) if mods else self.scalar(base, ops)
        if base == 's_addc_u32':
            # Only the high word of an advanced descriptor pointer; the carry is not modelled.
            hi = reg_range(ops[1]) if len(ops) == 3 else None
            if hasattr(self, 'scc_uniform') and self.tracks_uniformity():
                self.scc_uniform = False
            if hi and hi[0] == 's' and self.pointer_add(base, ops[0], ops[1:], [self.sym.get(hi[1]), None]):
                return None
            raise Unsupported(base)
        if base == 's_getpc_b64':
            # The calculated branch and program-image proofs elide GETPC before
            # scalar lowering. Keep the established diagnostic for unproven uses.
            raise Unsupported('%s: the shader load address is only elided for a program that also calculates a branch target' % base)
        raise Unsupported(base)

    # VOP1/VOP2 with DPP16 quad broadcast (#2125 item A): quad_perm:[k,k,k,k] makes every lane read SRC0 from
    # lane k of its own quad (RDNA 2 ISA, "DPP": a lane reads lane (lane & ~3) | perm[lane & 3] of the row; LLVM 15
    # AMDGPU DPP quad_perm). A pixel stage's four consecutive lanes are a 2x2 pixel quad (lane = x & 1 | (y & 1) << 1),
    # which is what SM 6.0 QuadReadLaneAt(value, k) reads, so the stage is compiled as ps_6_0 (Stage.needs_sm6).
    # Accepted only where the whole quad is proven active: the entry straight-line prefix of a pixel stage that ran
    # s_wqm exec, exec and whose EXEC is still that mask (helper lanes are active, so no source lane is off). A later
    # uniform loop or forward branch elsewhere in the stage does not drop the proof: every path still reaches the
    # broadcast with the same EXEC and cannot jump back to the prefix. bound_ctrl then has nothing to
    # decide (the source lane is never disabled), so it is accepted and ignored. A broadcast after a branch target,
    # after EXEC changed, or with divergent EXEC stays refused, as do other DPP controls (pair broadcasts, row shifts,
    # mirrors, DPP8), a masked row or bank, and a vertex stage (dpp() drops fi:1 first).
    DPP_TEMP = 256   # hidden VGPR that holds the quad read; the hardware has v0..v255 only
    DPP_OUTPUT_MODS = ('clamp', 'mul:2', 'mul:4', 'div:2')

    def dpp_broadcast(self, base, ops, mods, fetch_inactive=False):
        fields, rest = {}, []
        for m in mods:
            key, _, value = m.partition(':')
            if key in ('quad_perm', 'row_mask', 'bank_mask', 'bound_ctrl') and value and key not in fields:
                fields[key] = value
            elif m in self.DPP_OUTPUT_MODS:
                rest.append(m)
            else:
                raise Unsupported('%s_dpp with modifier %s' % (base, m))
        lanes = re.fullmatch(r'\[([0-3]),([0-3]),([0-3]),([0-3])\]', fields.get('quad_perm', ''))
        if not lanes or len(set(lanes.groups())) != 1:
            raise Unsupported('%s_dpp %s: only quad_perm:[k,k,k,k] broadcasts are lowered'
                              % (base, fields.get('quad_perm', 'without quad_perm')))
        if fields.get('row_mask') != '0xf' or fields.get('bank_mask') != '0xf':
            raise Unsupported('%s_dpp with a partial row_mask or bank_mask' % base)
        if self.kind != 'ps':
            raise Unsupported('%s_dpp outside a straight-line pixel stage with the entry EXEC mask' % base)
        # A branch to offset zero also makes the nominal entry block re-entrant.
        # The whole-program proof (quad_flow_refusals) replaces the entry-prefix test once translate() runs it.
        narrowed = False
        if not getattr(self, 'exec_known', False):
            if self.narrowing_proof():
                narrowed = True
            elif not self.quad_subrev_block_proof(base):
                raise Unsupported('%s_dpp outside the entry prefix of a pixel stage with the entry EXEC mask%s'
                                  % (base, self.quad_exec_cause()))
        elif not self.quad_flow_allowed() and getattr(self, 'past_entry', bool(getattr(self, 'blocks', ()))):
            raise Unsupported('%s_dpp outside the entry prefix of a pixel stage with the entry EXEC mask%s'
                              % (base, self.quad_exec_cause()))
        if not narrowed and fields.get('bound_ctrl', '1') != '1':
            raise Unsupported('%s_dpp bound_ctrl:%s' % (base, fields['bound_ctrl']))
        if not getattr(self, 'wqm_exec', False) and not self.quad_subrev_block_proof(base):
            # The entry EXEC can leave out lanes of a partly covered quad, and the guest then reads an inactive
            # source lane (bound_ctrl / fi), while QuadReadLaneAt would return a helper lane's value.
            raise Unsupported('%s_dpp without s_wqm exec, exec proving whole-quad activity' % base)
        if base != 'v_mov_b32' and base not in VEC_F and base not in VEC_U and base not in VEC_U_TO_F:
            raise Unsupported('DPP form of %s' % base)
        if len(ops) < 2:
            raise Unsupported('%s_dpp with %d operands' % (base, len(ops)))
        token = ops[1]
        neg = token.startswith('-') and base in VEC_F
        token = token[1:] if neg else token
        absolute = token.startswith('|') and token.endswith('|') and base in VEC_F
        token = token[1:-1] if absolute else token
        source = reg_range(token)
        if not source or source[0] != 'v' or source[1] != source[2]:
            raise Unsupported('%s_dpp source %s is not a VGPR' % (base, ops[1]))
        # Modifiers apply to the value the lane read, so they stay on the rewritten operand.
        rewritten = 'v%d' % self.DPP_TEMP
        rewritten = '|%s|' % rewritten if absolute else rewritten
        rewritten = '-' + rewritten if neg else rewritten
        self.note_quad_flow(base)
        self.needs_sm6 = True
        if narrowed:
            return self.dpp_broadcast_narrowed(base, ops, rest, rewritten, token, lanes.group(1), fields,
                                               fetch_inactive)
        self.set_v(self.DPP_TEMP, 'QuadReadLaneAt(%s, %su)' % (self.u(token), lanes.group(1)))
        operands = [ops[0], rewritten] + ops[2:]
        if base == 'v_mov_b32':
            if rest or len(operands) != 2:
                raise Unsupported('v_mov_b32_dpp with %d operands or an output modifier' % len(ops))
            return self.lower(base, '%s %s' % (base, ', '.join(operands)), operands, [])
        return self.lower(base, '%s %s %s' % (base, ', '.join(operands), ' '.join(rest)), operands, rest)

    DPP_VALID = 257  # hidden: the source lane's EXEC bit, read with the data

    NARROWING_WRITERS = ('v_cmpx_', 's_and_saveexec_', 's_and_b32', 's_and_b64', 's_andn2_b32', 's_andn2_b64')

    def narrowing_proof(self):
        """Whether EXEC at this point is the whole-quad mask of an entry s_wqm narrowed only by AND-type writes, in
        straight-line code: then every lane that is inactive here was active (helper lanes included) up to its
        narrowing, so the guest's and the host's VGPRs of an inactive lane agree and a quad read may take them."""
        wqm = getattr(self, 'wqm_set_at', None)
        if (self.kind != 'ps' or wqm is None or not getattr(self, 'wqm_exec', False) or getattr(self, 'fetched', False)
                or not self.quad_flow_straight()):
            return False
        for off, _n, name, asm in getattr(self, 'ins', ()):
            if off >= getattr(self, 'pc', 1 << 62):
                break
            ops = parse(asm)[1]
            if not writes_exec(name, ops):
                continue
            base = re.sub(r'_e(32|64)$', '', name)
            if off == wqm and base in ('s_wqm_b32', 's_wqm_b64'):
                continue
            if off < wqm or not base.startswith(self.NARROWING_WRITERS):
                return False
        return True

    def dpp_broadcast_narrowed(self, base, ops, rest, rewritten, token, lane, fields, fetch_inactive):
        """A quad broadcast behind a narrowed EXEC (RDNA 2 ISA DPP16; model section 2a). Per lane the source is valid
        when FI is set or the source lane's EXEC bit is 1; an invalid source reads 0 when BOUND_CTRL is set, and
        otherwise leaves the destination unwritten. The quad reads are unconditional (every thread of the quad runs
        them, as the quad intrinsics need); only the destination write is guarded by the thread's own EXEC bit."""
        self.used_v.update((self.DPP_TEMP, self.DPP_VALID))
        self.uniform_v.discard(self.DPP_TEMP)
        if isinstance(self.align4_v, set):
            self.align4_v.discard(self.DPP_TEMP)
        self.emit('v%d = QuadReadLaneAt(%s, %su);' % (self.DPP_TEMP, self.u(token), lane))
        bound = fields.get('bound_ctrl') == '1'
        guard = False
        if not fetch_inactive:
            self.emit('v%d = QuadReadLaneAt(exec & 1u, %su);' % (self.DPP_VALID, lane))
            if bound:
                self.emit('v%d = v%d != 0u ? v%d : 0u;' % (self.DPP_TEMP, self.DPP_VALID, self.DPP_TEMP))
            else:
                guard = True
        operands = [ops[0], rewritten] + ops[2:]
        if base == 'v_mov_b32' and (rest or len(operands) != 2):
            raise Unsupported('v_mov_b32_dpp with %d operands or an output modifier' % len(ops))
        if guard:
            self.emit('if (v%d != 0u) {' % self.DPP_VALID)
        result = self.lower(base, '%s %s %s' % (base, ', '.join(operands), ' '.join(rest)), operands, rest)
        if guard:
            self.emit('}')
        return result

    # VOP1/VOP2 with a DPP16 quad_perm source (AMD RDNA 2 ISA, "Data-Parallel Primitives"): lane i of
    # each aligned group of four reads src0 from lane quad_perm[i] of that group. The three exchanges
    # map onto the SM 6.0 quad intrinsics, which read the other lane of the 2x2 pixel quad and keep
    # helper lanes active. Only a pixel stage at entry EXEC qualifies, and only after `s_wqm exec, exec`
    # proved whole-quad EXEC: the hardware can return 0 or the old value for a source lane whose EXEC bit
    # is off (an uncovered pixel of an edge quad), and the intrinsics have no such case. Other permutations, fi:1 without that proof, DPP8, row shifts and partial masks stay refused.
    def dpp(self, base, ops, mods):
        if base in VOP3_ONLY_VEC_U:
            raise Unsupported('DPP form of %s' % base)
        # fi:1 (fetch inactive, DPP16 dword bit 18; LLVM AMDGPU `fi`, Mesa ACO fetch_inactive) only decides what a
        # disabled source lane returns. Every lowering below is taken only with the whole quad proven active, so no
        # source lane is disabled and it decides nothing.
        fetch_inactive = 'fi:1' in mods
        mods = [m for m in mods if m != 'fi:1']
        compute_float = getattr(self, 'kind', None) == 'cs' and base in COMPUTE_DPP_FLOAT
        if not compute_float and any(re.fullmatch(r'quad_perm:\[([0-3]),\1,\1,\1\]', mod) for mod in mods):
            return self.dpp_broadcast(base, ops, mods, fetch_inactive=fetch_inactive)
        perm = None
        row_ror8 = False
        row = None
        for m in mods:
            match = re.fullmatch(r'quad_perm:\[([0-3]),([0-3]),([0-3]),([0-3])\]', m)
            if match and perm is None and not row_ror8 and row is None:
                perm = tuple(int(x) for x in match.groups())
            elif m == 'row_ror:8' and perm is None and not row_ror8 and row is None:
                row_ror8 = True
            elif (compute_float and perm is None and not row_ror8 and row is None and
                  re.fullmatch(r'(?:row_shr:(?:[1-9]|1[0-5])|row_share:(?:[0-9]|1[0-5]))', m)):
                row = m
            elif m not in ('row_mask:0xf', 'bank_mask:0xf', 'bound_ctrl:0', 'bound_ctrl:1'):
                raise Unsupported('%s_dpp with %s' % (base, m))
        if row_ror8:
            # DPP16 row_ror:8 (RDNA 2 ISA, "Data-Parallel Primitives"): a row is 16 lanes and lane i reads the lane 8 places
            # round the row, which is lane i ^ 8 whichever way the rotation runs. Only compute, through groupshared memory.
            if getattr(self, 'kind', None) != 'cs':
                raise Unsupported('%s_dpp with row_ror:8 outside a compute program' % base)
            intrinsic = ROW_ROR8
        elif (perm in DPP_PAIR and not compute_float) or (
                perm in DPP_COARSE and base in DPP_DERIVATIVE_OPS and self.kind == 'ps'):
            return self.dpp_pair(base, '%s %s' % (base, ', '.join(ops)), ops, mods)
        else:
            intrinsic = QUAD_SWAPS.get(perm)
        indexed = compute_float and (row is not None or (perm is not None and intrinsic is None))
        if not intrinsic and not indexed:
            raise Unsupported('%s_dpp quad_perm %s: only the swaps [1,0,3,2], [2,3,0,1] and [3,2,1,0] are lowered'
                              % (base, perm))
        if base == 'v_mov_b32':
            arity, text, kind = 1, '{0}', 'u'
        elif base in VEC_U:
            (arity, text), kind = VEC_U[base], 'u'
        elif base in VEC_F and base not in ('v_fmac_f32', 'v_mac_f32'):
            (arity, text), kind = VEC_F[base], 'f'
        else:
            raise Unsupported('%s_dpp' % base)
        src_neg_abs = []
        if kind == 'f':
            # Float quad swaps accept the source neg/abs of the plain operation (RDNA 2 ISA 12.19;
            # LLVM 15 AMDGPU DPP prints them on the _dpp spelling). The permuted word is read first,
            # then neg/abs apply, so -vN negates the neighbour lane value. VGPR-only is kept: an SGPR
            # source stays refused below, as does a non-VGPR base.
            if len(ops) != arity + 1 or not re.fullmatch(r'v\d+', ops[0]):
                raise Unsupported('%s_dpp needs VGPR operands without modifiers' % base)
            for token in ops[1:1 + arity]:
                neg, rest = self.split_neg(token)
                absolute = rest.startswith('|') and rest.endswith('|')
                base_tok = rest[1:-1] if absolute else rest
                if not re.fullmatch(r'v\d+', base_tok):
                    raise Unsupported('%s_dpp needs VGPR operands without modifiers' % base)
                src_neg_abs.append((neg, absolute, base_tok))
        elif len(ops) != arity + 1 or any(not re.fullmatch(r'v\d+', o) for o in ops):
            raise Unsupported('%s_dpp needs VGPR operands without modifiers' % base)
        compute = getattr(self, 'kind', None) == 'cs'
        after_exec_write = (base == 'v_add_f32' and not self.exec_known and
                            getattr(self, 'pc', None) in getattr(self, 'whole_quad_sites', ()))
        if compute:
            self.compute_quad_gate(base, row_ror8, allow_flow=compute_float)
            if row is not None and math.prod(self.pipe.group) % 16:
                raise Unsupported('%s_dpp %s requires a group of whole rows (a multiple of 16 threads)' % (base, row))
        else:
            block_proof = self.quad_subrev_block_proof(base)
            if not after_exec_write:
                if (getattr(self, 'kind', None) != 'ps' or not (self.quad_flow_straight() or self.quad_flow_allowed())
                        or not (getattr(self, 'exec_known', False) or block_proof)):
                    raise Unsupported('%s_dpp requires a straight-line pixel shader with entry EXEC%s' % (base, self.quad_exec_cause()))
                if not (getattr(self, 'wqm_exec', False) or block_proof):
                    raise Unsupported('%s_dpp requires whole-quad EXEC (s_wqm of exec first): a partly covered quad has '
                                      'inactive source lanes that the hardware masks and QuadReadAcross does not' % base)
            else:
                self.whole_quad_reads.add(self.pc)
            self.note_quad_flow(base, allow_late_wqm=base in DPP_DERIVATIVE_OPS
                                and intrinsic in ('QuadReadAcrossX', 'QuadReadAcrossY'))
            self.needs_sm6 = True
        # src0 is the exchanged lane; a VOP2 src1 stays this lane's own register.
        valid = None
        if kind == 'f':
            neg0, abs0, base0 = src_neg_abs[0]
            if indexed:
                swapped, valid = self.compute_dpp_source(self.u(base0), perm, row)
                if valid and 'bound_ctrl:1' in mods:
                    swapped = '(%s ? %s : 0u)' % (valid, swapped)
                    valid = None
            else:
                swapped = self.quad_source(intrinsic, self.u(base0), compute)
            if after_exec_write:
                # All host threads of the quad execute the read, including when the guest quad is off.
                self.used_v.add(self.DPP_TEMP)
                self.emit('v%d = %s;' % (self.DPP_TEMP, swapped))
                swapped = 'v%d' % self.DPP_TEMP
            first = 'F(%s)' % swapped
            if abs0:
                first = 'abs(%s)' % first
            if neg0:
                first = '(-%s)' % first
            args = [first] + [self.f(o) for o in ops[2:1 + arity]]
        else:
            swapped = self.quad_source(intrinsic, self.u(ops[1]), compute)
            args = [swapped] + [self.u(o) for o in ops[2:1 + arity]]
        expr = text.format(*args)
        dst = reg_range(ops[0])[1]
        value = 'U(%s)' % expr if kind == 'f' else expr
        if valid:
            value = '(%s ? %s : %s)' % (valid, value, self.u(ops[0]))
        self.set_v(dst, value)
        if compute:
            self.emit('GroupMemoryBarrierWithGroupSync();')  # the next exchange may rewrite the array

    def compute_dpp_source(self, source, perm, row):
        """Stage source words and select a lane within the same logical quad or row."""
        flat = self.stage_quad_source(source)
        if row is None:
            control = sum(lane << (2 * i) for i, lane in enumerate(perm))
            return 'x2w_quad[(%s & ~3u) | ((%du >> (2u * (%s & 3u))) & 3u)]' % (flat, control, flat), None
        kind, lane = row.split(':')
        if kind == 'row_share':
            return 'x2w_quad[(%s & ~15u) | %su]' % (flat, lane), None
        valid = '(%s & 15u) >= %su' % (flat, lane)
        # Keep even the unselected read inside this row.
        return 'x2w_quad[(%s & ~15u) | ((%s - %su) & 15u)]' % (flat, flat, lane), valid

    def compute_quad_gate(self, base, row_ror8=False, *, allow_flow=False):
        """A DPP quad swap in a compute program is exchanged through groupshared memory (no quad intrinsic: SM 6 leaves the
        quad of a compute thread to the implementation). Lane i of an aligned group of four reads lane perm[i] (RDNA 2 ISA,
        DPP16 quad_perm); a wave holds 32 or 64 consecutive threads in x-major order, so lane bits 1:0 are the bits 1:0
        of the flat thread index whatever the wave size. A group of whole quads must reach the exchange with
        entry EXEC. Keep the original straight-line path and use the CFG proof for the selected float ALU."""
        x, y, z = self.pipe.group
        if x * y * z % 4:
            raise Unsupported('%s_dpp in a compute program requires a group of whole quads (a multiple of 4 threads)' % base)
        if row_ror8 and x * y * z % 16:
            raise Unsupported('%s_dpp row_ror:8 in a compute program requires a group of whole rows (a multiple of 16 threads)' % base)
        if self.quad_flow_straight() and getattr(self, 'exec_known', False):
            return
        block = getattr(self, 'compute_dpp_entry', {}).get(getattr(self, 'pc', None))
        if allow_flow and block is not None:
            self.compute_dpp_blocks.add(block)
            return
        raise Unsupported('%s_dpp requires a straight-line compute program with entry EXEC%s' % (base, self.quad_exec_cause()))

    def quad_source(self, intrinsic, source, compute):
        """Expression for the exchanged lane word. Pixel stage: the SM 6 quad intrinsic. Compute: write this thread's word to
        a groupshared slot, wait for the group, and read the slot of the lane the swap names."""
        if not compute:
            return '%s(%s)' % (intrinsic, source)
        flat = self.stage_quad_source(source)
        return 'x2w_quad[%s ^ %du]' % (flat, {'QuadReadAcrossX': 1, 'QuadReadAcrossY': 2, 'QuadReadAcrossDiagonal': 3, ROW_ROR8: 8}[intrinsic])

    def stage_quad_source(self, source):
        x, y, _ = self.pipe.group
        flat = '(tid.x + %du * (tid.y + %du * tid.z))' % (x, y)
        self.uses_quad_shared = True
        self.emit('x2w_quad[%s] = %s;' % (flat, source))
        self.emit('GroupMemoryBarrierWithGroupSync();')
        return flat

    def half_operand(self, tok, kind):
        """Float expression for a 16-bit operand: the low half of the register, or an inline constant."""
        if kind in ('i16', 'u16'):
            if tok in FLOATS:  # an inline float constant is a binary16 word at 16-bit width, not binary32
                word = struct.unpack('<H', struct.pack('<e', FLOATS[tok]))[0]
                src = '%du' % word
            else:
                src = self.u(tok)
            if kind == 'i16':
                return 'float(asint((%s & 0xffffu) << 16u) >> 16)' % src
            return 'float(%s & 0xffffu)' % src
        neg = False
        if tok.startswith('-') and (not re.match(r'-[\d.]', tok) or re.match(r'-0x', tok)):
            neg, tok = True, tok[1:]
        absolute = tok.startswith('|') and tok.endswith('|')
        if absolute:
            tok = tok[1:-1]
        if tok in FLOATS:
            text = repr(struct.unpack('<e', struct.pack('<e', FLOATS[tok]))[0])  # inline constants are binary16 here
        elif tok == '-0':
            text = '0.0'
        else:
            text = 'f16tof32(%s & 0xffffu)' % self.u(tok)
        if absolute:
            text = 'abs(%s)' % text
        return '(-%s)' % text if neg else text

    def half_alu(self, base, ops, mods):
        """16-bit VALU: operands are the low 16 bits, the result fills the low 16 bits and zeroes the rest (RDNA 2)."""
        arity, kind, text, result = VEC_H[base]
        if len(ops) != arity + 1:
            raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
        if any(m != 'clamp' for m in mods) or ('clamp' in mods and result != 'h'):
            raise Unsupported('%s with %s' % (base, mods))
        args = [self.half_operand(o, kind) for o in ops[1:1 + arity]]
        if base in ('v_fmac_f16', 'v_mac_f16'):
            args.append(self.half_operand(ops[0], 'h'))
        expr = text.format(*args)
        if result == 'h':
            if 'clamp' in mods:
                expr = 'saturate(%s)' % expr
            expr = 'f16rn(%s)' % expr
        self.set_v(reg_range(ops[0])[1], expr)

    # VOP1/VOP2 with sub-dword addressing: select the source fields, run the same operation, then place
    # the low bits of the result in the destination field. The integer forms reuse VEC_U unchanged,
    # v_mov_b32 is the same path with the selected source as its result, v_cndmask_b32 selects with VCC,
    # VEC_U_TO_F conversions apply output modifiers before dst_sel, VEC_F forms stay refused, and
    # dst_unused UNUSED_SEXT has no lowering.
    def sdwa_half(self, base, ops, mods):
        """SDWA binary16 arithmetic: select a half from each source, round once and write one half.

        Sources: src0_sel/src1_sel WORD_0 or WORD_1 pick the half; DWORD reads the low half, as the 16-bit
        operation does. Destination: dst_sel WORD_0 or WORD_1 with dst_unused UNUSED_PAD (other half zero) or
        UNUSED_PRESERVE (other half of the old register kept). dst_sel DWORD stays refused: the upper half of a
        16-bit result is not claimed (an open question, see the low-16-only contract), and a byte selection, sext(),
        clamp and the output modifiers have no lowering here. Sources are VGPRs with the abs/neg modifiers.
        """
        arity, _kind, text, _result = VEC_H[base]
        allowed_fields = {'src%d_sel' % index for index in range(arity)} | {'dst_sel', 'dst_unused'}
        fields = {}
        for m in mods:
            key, _, value = m.partition(':')
            if key in allowed_fields and value and key not in fields:
                fields[key] = value
            else:
                raise Unsupported('%s SDWA with modifier %s' % (base, m))
        if len(ops) != arity + 1:
            raise Unsupported('%s SDWA with %d operands, expected %d' % (base, len(ops), arity + 1))
        args = []
        for index in range(arity):
            selection = fields.get('src%d_sel' % index)
            if selection not in ('DWORD', 'WORD_0', 'WORD_1'):
                raise Unsupported('%s SDWA src%d_sel %s' % (base, index, selection))
            neg, token = self.split_neg(ops[1 + index])
            absolute = False
            while True:
                if token.startswith('|') and token.endswith('|') and len(token) > 2:
                    token = token[1:-1]
                elif token.startswith('abs(') and token.endswith(')'):
                    token = token[len('abs('):-1]
                else:
                    break
                absolute = True
            reg = reg_range(token)
            if not reg or reg[0] != 'v' or reg[1] != reg[2]:
                raise Unsupported('%s SDWA source %s: expected a single VGPR' % (base, ops[1 + index]))
            word = self.u(token, low16_only=selection != 'WORD_1')
            half = '(%s >> 16u)' % word if selection == 'WORD_1' else '(%s & 0xffffu)' % word
            value = 'f16tof32(%s)' % half
            if absolute:
                value = 'abs(%s)' % value
            args.append('(-%s)' % value if neg else value)
        select = fields.get('dst_sel')
        unused = fields.get('dst_unused', 'UNUSED_PAD')
        if select not in ('WORD_0', 'WORD_1'):
            raise Unsupported('%s SDWA dst_sel %s' % (base, select))
        if unused not in ('UNUSED_PAD', 'UNUSED_PRESERVE'):
            raise Unsupported('%s SDWA dst_unused %s' % (base, unused))
        destination = reg_range(ops[0])[1]
        result = '(f16rn(%s))' % text.format(*args)
        # Keeping the old high half of a register whose high half is undefined leaves it undefined.
        keep_low16 = select == 'WORD_0' and unused == 'UNUSED_PRESERVE' and destination in self.low16_only_v
        self.set_v(destination, sdwa_dest(result, select, unused, 'v%d' % destination), low16_only=keep_low16)

    def sdwa(self, base, ops, mods):
        if base in ('v_add_f16', 'v_sub_f16', 'v_subrev_f16', 'v_mul_f16', 'v_exp_f16', 'v_rcp_f16'):
            return self.sdwa_half(base, ops, mods)
        if base in ('v_cvt_off_f32_i4', *VOP3_ONLY_VEC_U):
            raise Unsupported('SDWA form of %s' % base)
        fields = {}
        out_mods = []
        float_result = base in VEC_U_TO_F
        for m in mods:
            if float_result and m in ('clamp', 'mul:2', 'mul:4', 'div:2'):
                out_mods.append(m)
            else:
                key, _, value = m.partition(':')
                if key in ('src0_sel', 'src1_sel', 'src2_sel', 'dst_sel', 'dst_unused') and value:
                    fields[key] = value
                else:
                    raise Unsupported('%s with SDWA modifier %s' % (base, m))
        if float_result:
            arity, text = VEC_U_TO_F[base]
        elif base in VEC_U:
            arity, text = VEC_U[base]
        elif base == 'v_mov_b32':
            arity, text = 1, '{0}'
            # A lane mask is one bit per lane here, so a byte or word of it is not the hardware's field.
            if fields.get('src0_sel') == 'DWORD':
                self.reject_lane_mask_data(base, ops[1])
            elif self.is_lane_mask(ops[1].removeprefix('sext(').removesuffix(')')):
                raise Unsupported('v_mov_b32 SDWA %s of lane mask %s' % (fields.get('src0_sel'), ops[1]))
        elif base == 'v_cndmask_b32' and len(ops) == 4 and ops[3] == 'vcc_lo':
            # The VOP2 select: its lane mask is always VCC, printed as the fourth operand, and read like the
            # plain select's mask (op_v_cndmask_b32).
            mask = self.u(ops[3]).replace('{', '{{').replace('}', '}}')
            arity, text = 2, '(%s & 1) ? {1} : {0}' % mask
        else:
            raise Unsupported('SDWA form of %s' % base)
        sources = []
        for index in range(arity):
            token = ops[1 + index]
            sext = token.startswith('sext(') and token.endswith(')')
            if sext:
                token = token[len('sext('):-1]
            selection = fields.get('src%d_sel' % index)
            if not selection:
                raise Unsupported('%s SDWA without src%d_sel' % (base, index))
            low16 = selection in ('BYTE_0', 'BYTE_1', 'WORD_0') or (
                selection == 'DWORD' and not sext and base in ('v_cvt_f32_f16', 'v_cvt_f32_ubyte0', 'v_cvt_f32_ubyte1'))
            sources.append(sdwa_source(self.u(token, low16_only=low16), selection, sext))
        result = text.format(*sources)
        if float_result:
            # Output modifiers scale and then clamp the float value; dst_sel writes its bit pattern.
            result = 'U(%s)' % self.f_mods(result, out_mods)
        select = fields.get('dst_sel')
        if not select:
            raise Unsupported('%s SDWA without dst_sel' % base)
        # The operation's HLSL carries its own operators, so it has to be grouped before the
        # destination field masks it: "v7 | v8 & 0xffff" would otherwise bind as "v7 | (v8 & 0xffff)".
        if SDWA_SEL[select][1] < 32:
            result = '(%s)' % result
        destination = reg_range(ops[0])[1]
        unused = fields.get('dst_unused', 'UNUSED_PAD')
        keep_low16 = (unused == 'UNUSED_PRESERVE' and select not in ('DWORD', 'WORD_1') and
                      destination in self.low16_only_v)
        self.set_v(destination, sdwa_dest(result, select, unused, 'v%d' % destination), low16_only=keep_low16)

    def pointer_add(self, base, dname, srcs, syms):
        """Propagate a descriptor pointer through s_add_u32 sD, sP, <literal> / s_addc_u32 sD+1, sP+1, 0.

        The pair advances a table or root pointer by a literal byte offset, into the same or
        another register pair. The offset rides in the low word as ('ptr', source, offset); the
        high word keeps ('ptrhi', source). The low-word carry is taken as zero: the base is a
        table or root address and the offset is below 1 MiB. Anything else returns False.
        """
        if base not in ('s_add_u32', 's_addc_u32') or len(srcs) != 2 or not syms[0]:
            return False
        if getattr(self, 'has_branches', False):
            if (not getattr(self, 'forward_table_adds', False) or base != 's_add_u32'
                    or syms[0][0] != 'ptr' or syms[0][1][0] != 'table'):
                return False
        reg = reg_range(dname)
        if not reg or reg[0] != 's' or reg[1] != reg[2] or reg[1] > 105:
            return False
        d = reg[1]
        if base == 's_add_u32':
            if syms[0][0] != 'ptr' or not re.fullmatch(r'(?:0[xX][0-9a-fA-F]+|\d+)', srcs[1]):
                return False
            added = int(srcs[1], 0)
            if added > 0xfffff:
                raise Unsupported('descriptor pointer offset %s' % srcs[1])
            before = syms[0][2] if len(syms[0]) > 2 else 0
            if not isinstance(before, int):
                raise Unsupported('add through a pointer with unknown byte offset')
            if getattr(self, 'forward_table_adds', False) and before + added > 0xfffff:
                raise Unsupported('accumulated table pointer offset %#x' % (before + added))
            self.sym[d] = ('ptr', syms[0][1], before + added)
            if syms[0][1][0] in PTR64_SOURCES and d + 1 <= 105:
                # The carry-in of an s_addc_u32 is only the carry of this very add: remember where the next
                # instruction starts so that the high-word add is accepted right behind it and nowhere else.
                size = next((n for off, n, _name, _asm in getattr(self, 'ins', ()) if off == self.pc), 0)
                self.ptr_low_add = (d, self.pc + size, syms[0][1], before, before + added)
                # The pair stays open until the matching s_addc_u32 completes it. The high register is left as it
                # is (in place it is still the original high source); a load through an open pair is refused.
                self.__dict__.setdefault('ptr_open_pairs', set()).add(d)
            return True
        low = self.sym.get(d - 1)
        mark = getattr(self, 'ptr_low_add', None)
        hi = reg_range(srcs[0])
        # The high operand is judged by what it holds, not by where it sits: a copied high word still counts.
        if (not mark or srcs[1] != '0' or not low or low[0] != 'ptr' or not hi or hi[0] != 's'
                or mark[:2] != (d - 1, getattr(self, 'pc', 0)) or low[1] != mark[2]
                or self.sym.get(hi[1]) != ptr_high_symbol(mark[2], mark[3])):
            return False
        self.sym[d] = ptr_high_symbol(mark[2], mark[4])
        getattr(self, 'ptr_open_pairs', set()).discard(d - 1)
        # This s_addc_u32 is the only reader of the low add's elided carry; any other SCC reader stays refused.
        self.__dict__.setdefault('ptr_addc_pcs', set()).add(self.pc)
        return True

    def exec_zero_keepable(self, target, reg, arriving, paths, branches):
        """Rule R1: the fall-through arm of an s_cbranch_execz skip wrote a descriptor word and no other edge into the
        target wrote the register, and exec is not written between the first skipping branch and the target (so on
        the fall-through exec is still the arm's mask, and on every taken edge it is 0)."""
        if len(paths) != 1 or not paths[0] or paths[0][0] != 'desc' or reg not in arriving[0][1]:
            return False
        if any(reg in w for _, w in arriving[1:]):
            return False
        first = min(pc for pc, to in branches if to == target)
        for off, _n, name, asm in self.ins:
            if first < off < target and writes_exec(name, parse(asm)[1]):
                return False
        return True

    def end_exec_keep(self, index, off, name, asm, targets):
        """Ends the R1 window: from the next join, branch or exec write on, a register kept after an execz join is
        joined again (a lane that skipped the arm is active again and would read the arm's stale word). A scalar
        instruction reading a kept register ends it for that register at once: scalar work runs with exec == 0 too.
        A descriptor load that redefines a kept register on every path retires its marker without poisoning it."""
        prev_name, prev_asm = self.ins[index - 1][2], self.ins[index - 1][3]
        if prev_name.startswith(('s_load', 's_buffer_load')):
            dest = reg_range(parse(prev_asm)[1][0])
            for reg in [r for r in self.exec_keep if dest and dest[0] == 's' and dest[1] <= r <= dest[2]]:
                del self.exec_keep[reg]
        if not self.exec_keep:
            return
        if (off in targets or name.startswith('s_cbranch') or name in ('s_branch', 's_setpc_b64')
                or writes_exec(prev_name, parse(prev_asm)[1])):
            self.retire_exec_keep(set(self.exec_keep))
        elif name.startswith('s_'):
            self.end_exec_keep_scalar_read(name, asm)

    def end_exec_keep_scalar_read(self, name, asm):
        doomed = set()
        for tok in parse(asm)[1][1:]:
            r = reg_range(tok)
            if r and r[0] == 's':
                doomed |= {reg for reg in self.exec_keep if r[1] <= reg <= r[2]}
        self.retire_exec_keep(doomed)

    def retire_exec_keep(self, doomed):
        for reg in doomed:
            if self.sym.get(reg) == self.exec_keep[reg]:
                self.sym[reg] = ('joined at a branch target',)
            del self.exec_keep[reg]

    # scalar ALU with descriptor-word bookkeeping
    def exec_absorbed_source(self, base, srcs):
        """Index of a joined source of the instruction at a branch target whose stale value cannot reach bit 0 of
        the result, or None.

        `s_and_b64 D, S, exec` (either order) and `s_orn2_b64 D, S, exec` are lane-mask operations: the emitted
        result is bit 0 of `S & exec` or `S | ~exec`. When S was written as a run-time number on the fall-through
        and every other path into the target is a taken s_cbranch_execz edge (exec_zero_join), the lane's exec bit
        is 0 on exactly the paths where the variable of S may not hold the register's value, and there the result
        is 0 (or 1 for orn2) whatever S holds. Only the instruction at the target qualifies: exec is unchanged."""
        off, regs = getattr(self, 'exec_zero_join', (None, ()))
        if off is None or off != getattr(self, 'pc', None) or len(srcs) != 2:
            return None
        for i in {'s_and_b64': (0, 1), 's_orn2_b64': (0,)}.get(base, ()):
            reg = reg_range(srcs[i])
            if srcs[1 - i] == 'exec' and reg and reg[0] == 's' and reg[1] in regs:
                return i
        return None

    def dead_vcc_high(self):
        """Follow each path until it exits or a full write replaces the old high word."""
        bounds, block_of, succ, _ = self.spill_cfg
        index = self.descriptor_instruction_index
        pending = [(block_of[index], index + 1)]
        seen = set()
        while pending:
            block, start = pending.pop()
            if (block, start) in seen:
                continue
            seen.add((block, start))
            for _, _, name, asm in self.ins[start:bounds[block][1]]:
                reads, writes, _ = scalar_effects(name, asm)
                if reads & {'vcc_hi', 's107'} or name.startswith(('s_movrel', 's_setpc', 's_swappc')):
                    return False
                if name.startswith('s_') and writes & {'vcc', 's106'} and writes & {'vcc_hi', 's107'}:
                    break
            else:
                pending.extend((target, bounds[target][0]) for target in succ[block])
        return True

    def scalar(self, base, ops, mods=()):
        # Every SALU form in SCA is dst, src (the two templates below use {0} and {1}); a short list would
        # index past the end and leave `translator crash in <op>: IndexError` in the census instead of a
        # refusal. s_movk_i32 keeps its own exact check for the immediate it folds.
        if base != 's_movk_i32' and len(ops) < 2:
            raise Unsupported('%s with %d operands, expected 2' % (base, len(ops)))
        if base == 's_movk_i32' and len(ops) != 2:
            raise Unsupported('s_movk_i32 with %d operands, expected 2' % len(ops))
        text, scc = SCA[base]
        dst = reg_range(ops[0])
        if base == 's_add_u32' and self.address_carry_reader is not None and self.pc == self.address_carry_reader:
            if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
                raise Unsupported('s_add_u32 destination %s' % ops[0])
            # The immediately preceding numeric add emitted the carry; no branch can enter here.
            self.set_s(dst[1], '(scc ? 1u : 0u)')
            self.set_scc('false')  # 0 + a single bit never produces another carry.
            return None
        pair = base.endswith('_b64')
        srcs = ops[1:]
        # 64-bit operands name the low register; exec/vcc name themselves.
        def low(tok):
            r = reg_range(tok)
            return ('s%d' % r[1]) if r and r[0] == 's' and r[1] != r[2] else tok
        srcs = [low(t) for t in srcs]
        dname = low(ops[0])
        if base == 's_movk_i32':
            srcs = [str(sopk_imm(srcs[0], True))]  # D.i32 = signext(SIMM16)
        shift_add = base in SHIFT_ADD
        word_complement = base in ('s_nand_b32', 's_nor_b32', 's_xnor_b32')
        lane_complement = False
        # s_mul_hi_{u,i}32 need the whole 32-bit word of each source, like the halfword pack.
        word_pack = base in PACK_B16 or base in ('s_sext_i32_i8', 's_sext_i32_i16', 's_mul_hi_u32', 's_mul_hi_i32')
        word_absdiff = base == 's_absdiff_i32'
        if shift_add or word_complement or word_pack or word_absdiff:
            # The per-lane mask cannot supply the full word needed by these operations. s_mul_hi_{u,i}32 into an
            # SGPR that held a mask replaces it with a plain word (set_s drops the mark), so only vcc/exec count.
            mask_dest = (dname in ('exec', 'exec_lo', 'exec_hi', 'vcc', 'vcc_lo', 'vcc_hi')
                         if base in ('s_mul_hi_u32', 's_mul_hi_i32')
                         else self.is_lane_mask(dname))
            # Bit i of ~(S0 op S1) depends only on bit i of each source, so with every source a lane mask the lane's
            # bit of the result is exact. Only an SGPR destination is taken; a mixed mask and plain word is not.
            def mask_source(tok):
                r = reg_range(tok)
                return tok in ('vcc', 'vcc_lo', 'exec', 'exec_lo') or bool(
                    r and r[0] == 's' and r[1] == r[2] and r[1] in self.masks)
            lane_complement = (word_complement and bool(dst) and dst[0] == 's' and dst[1] == dst[2] and dst[1] <= 105
                               and all(mask_source(t) for t in srcs))
            if not lane_complement and (mask_dest or any(self.is_lane_mask(t) for t in srcs)):
                raise Unsupported('%s on a lane mask' % base)
            if lane_complement and any(reg_range(t) and reg_range(t)[1] in getattr(self, 'join_masks', ())
                                       for t in srcs):
                raise Unsupported('%s on a lane mask joined at a branch target' % base)
        if shift_add and dname != 'm0' and (not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105):
            raise Unsupported('%s destination %s' % (base, dname))
        if word_pack:
            if base in ('s_mul_hi_u32', 's_mul_hi_i32') and len(srcs) != 2:
                raise Unsupported('%s with %d operands, expected 3' % (base, len(ops)))
            if dname != 'm0' and (dst is None or dst[0] != 's' or dst[1] > 105):
                raise Unsupported('%s destination %s' % (base, dname))
            for source in srcs:
                reg = reg_range(source)
                if reg and (reg[0] != 's' or reg[1] > 105):
                    raise Unsupported('%s source %s' % (base, source))
        if pair and ('m0' in srcs or dname == 'm0'):
            raise Unsupported('64-bit scalar operand m0')
        def src_sym(t):
            if t == 'm0':
                return self.sym.get('m0')
            r = reg_range(t)
            return self.sym.get(r[1]) if r and r[0] == 's' else None
        syms = [src_sym(t) for t in srcs]
        mask_dest = dname in ('exec', 'vcc', 'exec_lo', 'vcc_lo', 'exec_hi', 'vcc_hi')
        constant_pair = None
        if base == 's_mov_b64' and ops[0] == 'vcc' and len(ops) == 2:
            source = reg_range(ops[1])
            if source and source[0] == 's':
                # The decoder prints odd scalar operands as single registers.
                high = source[2]
                if high == source[1] and high % 2 and high < 105:
                    high += 1
                words = (syms[0], self.sym.get(high))
                tracked_pair = high == source[1] + 1 and high <= 107 and all(
                    word and word[0] in ('desc', 'const') for word in words)
                if any(word and word[0] == 'desc' for word in words):
                    wave = getattr(self.pipe, 'wave_size', None)
                    if (source[1] % 2 == 0 and source[2] == source[1] + 1 and source[1] in self.masks
                            and (not words[0] or words[0][0] != 'desc') and words[1] and words[1][0] == 'desc'
                            and (wave == 32 or wave is None and wave32_exec_proof(self.ins))
                            and self.dead_vcc_high()):
                        self.sym.pop(106, None)
                        self.sym.pop(107, None)
                        self.emit('vcc = (%s) & 1;' % self.S(source[1]))
                        return None
                    if not tracked_pair:
                        raise Unsupported('mask from descriptor data: vcc needs a pair of tracked descriptor words or constants')
                    self.sym[106], self.sym[107] = words
                    self.mark_vcc_descriptor(extended=bool(source[1] % 2 or any(w[0] == 'const' for w in words)))
                    return None
                if tracked_pair:
                    constant_pair = words
        tg_srcs = [
            reg for tok in ops[1:] for r in (reg_range(tok),) if r and r[0] == 's'
            for reg in range(r[1], r[2] + 1) if self.sym.get(reg) == ('tg_size',)
        ]
        if tg_srcs:
            # Wave-ID bits TG_SIZE[11:6] (mask 0xfc0) are identically 0 for <= 32 threads per group (1 wave).
            if (base == 's_and_b32' and not mods and len(srcs) == 2
                    and dst and dst[0] == 's' and dst[1] == dst[2] and dst[1] <= 105):
                other = (srcs[1] if syms[0] == ('tg_size',) and syms[1] != ('tg_size',)
                         else srcs[0] if syms[1] == ('tg_size',) and syms[0] != ('tg_size',)
                         else None)
                if other is not None and self.const_u32(other) == 0xfc0:
                    x, y, z = self.pipe.group
                    if x * y * z <= 32:
                        self.set_s(dst[1], '0u', ('const', 0), scc='false')
                        return None
                    # Wave32 waves hold 32 consecutive threads in x-major order: the wave ID is the flat thread index / 32,
                    # and differs between the waves of one group (so the register is not uniform).
                    flat = '(tid.x + %du * (tid.y + %du * tid.z))' % (x, y)
                    self.set_s(dst[1], '((%s >> 5u) << 6u)' % flat, scc='$ != 0u', uniform=False)
                    return None
            raise Unsupported('TG_SIZE register s%d read outside s_and_b32 with 0xfc0' % tg_srcs[0])
        if any(s and s[0] in ('desc', 'descnum', 'ptr', 'ptrhi') for s in syms):
            if (shift_add and getattr(self, 'kind', None) in ('ps', 'vs') and not mods
                    and dst and dst[0] == 's' and not self.is_lane_mask(dname)):
                result = self.descriptor_alu(text, srcs, syms)
                expr = self.descriptor_number(self.descriptor_number_node(result), fixed_only=True)
                values = [self.descriptor_operand(t) for t in srcs]
                self.set_scc(shift_add_carry(base, values, expr))
                self.mark_descriptor_scc()
                self.sym[dst[1]] = result
                return None
            if 'src_scc' in srcs and not getattr(self, 'scc_verified', False):
                raise Unsupported(self.scc_refusal('src_scc with unverified SCC', 'descriptor source'))
            if scc or base in SCC_UNMODELLED:
                self.scc_effect = 'descriptor'
                self.scc_verified = False
                if self.tracks_uniformity():
                    self.scc_uniform = False
            if shift_add or word_complement or word_absdiff:
                raise Unsupported(self.scc_refusal('scc from descriptor data', 'shift-add, complement or absdiff'))
            if word_pack:
                raise Unsupported('%s on descriptor data' % base)
            # Every path below writes the destination's symbol directly, not through set_s, or refuses. The
            # written word is descriptor data, so a lane-mask mark the SGPR held from an earlier write ends here as
            # set_s ends it. lower() has already classified the sources, so a mask read by this instruction is
            # still refused.
            if dst and dst[0] == 's':
                for reg in range(dst[1], dst[2] + 1):
                    getattr(self, 'masks', set()).discard(reg)
                    getattr(self, 'mask_high', set()).discard(reg)
            if self.pointer_add(base, dname, srcs, syms):
                return None
            if dname == 'm0':
                if scc:
                    raise Unsupported(self.scc_refusal('scc from descriptor data', 'm0 write'))
                self.sym['m0'] = syms[0] if base == 's_mov_b32' else ('descnum',)
                return None
            if mask_dest:
                raise Unsupported('mask from descriptor data')
            if dname == 'null':
                raise Unsupported(self.scc_refusal('scc from descriptor data', 'null destination') if (scc or base in SCC_UNMODELLED)
                                  else '%s into null on descriptor data' % base)
            if not dst or dst[0] != 's':
                raise Unsupported('%s destination %s' % (base, ops[0]))
            d = dst[1]
            own = self.sym.get(d)
            raw = (raw_descriptor_alu(base, syms, [self.const_u32(t) for t in srcs])
                   if getattr(self, 'kind', None) in ('cs', 'vs') and len(syms) == 2 else None)
            if base in ('s_mov_b32', 's_mov_b64'):
                self.sym[d] = syms[0]
                if pair:
                    high = reg_range(ops[1])[2]
                    hi = self.sym.get(high)
                    if hi and not run_time_symbol(hi):
                        self.sym[d + 1] = hi  # a descriptor word, a pointer or an unassigned constant: symbol only
                    else:
                        # A value the high word holds at run time is copied at run time too, so a join that later
                        # reads the register at run time finds it (a constant keeps its symbol for folding).
                        self.set_s(d + 1, self.S(high))
                        if hi and hi[0] in ('const', 'ud'):
                            self.sym[d + 1] = hi
                        elif self.tracks_uniformity():
                            self.varying_s.add(d + 1)
            elif (own and own[0] == 'ptr' and base in ('s_add_u32', 's_addc_u32')
                  and not getattr(self, 'has_branches', False)):
                # With branches the offset is not tracked: the 'added' mark below stays, and the join refuses it.
                raise Unsupported('%s adds a non-literal to a descriptor pointer' % base)
            elif raw is not None:
                self.sym[d] = raw
                if raw[0] == 'desc':
                    self.pipe.modified.add(desc_key(raw))
            elif own and own[0] in ('desc', 'ptr', 'ptrhi') and any(s == own for s in syms):
                if own[0] == 'desc':
                    self.pipe.modified.add(desc_key(own))
                    # Keep the resource identity for loads; stores refuse the unproved value.
                    self.sym[d] = own[:4] + ('modified',)
                    getattr(self.pipe, 'other_modified', set()).add(desc_key(own))
                else:
                    self.sym[d] = own[:2] + ('added',)  # this operation has no proven byte offset
            else:
                arity = max(int(i) for i in re.findall(r'\{(\d+)\}', text)) + 1
                self.sym[d] = (self.descriptor_alu(text, srcs, syms)
                               if not pair and not mods and arity == len(srcs)
                               else ('descnum', 'alu', ('unavailable', '%s in %s' % (', '.join(srcs), base))))
            return None
        # The table templates name their sources as {0}, {1}, ...; a short operand list would make
        # str.format raise IndexError, which the single-shader path records as a translator crash
        # rather than a refusal. The arity guards above cover the table entries that spell a count, so
        # this one covers the templates themselves: every placeholder the template uses must have a
        # source, and a template may not read a source the instruction does not carry.
        needed = max((int(i) for i in re.findall(r'\{(\d+)\}', text)), default=-1) + 1
        if len(srcs) < needed:
            raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), needed + 1))
        if dname == 'null':
            if any(self.is_lane_mask(t) for t in srcs):
                raise Unsupported('%s into null on a lane mask' % base)
            if pair:
                raise Unsupported('%s into null' % base)
        deferred_scc = None
        absorbed = self.exec_absorbed_source(base, srcs)
        if absorbed is not None:
            self.used_s.add(reg_range(srcs[absorbed])[1])
        values = [t if i == absorbed else self.u(t) for i, t in enumerate(srcs)]
        expr = text.format(*values)
        if base == 's_add_u32' and getattr(self, 'pc', None) in self.address_carry_pairs:
            for tok in srcs:
                self.reject_lane_mask_data(base, tok)
                r = reg_range(tok)
                if r and r[1] in getattr(self, 'join_masks', ()):
                    raise Unsupported('address carry on a lane mask joined at a branch target')
            # Compute carry before writing the low register, which can also be an addend.
            self.set_scc('(%s) < %s' % (expr, values[0]))
            self.address_carry_reader = self.address_carry_pairs[self.pc]
        elif base in SCC_UNMODELLED:
            scc_text = add_sub_min_max_scc(base, values[0], values[1], expr)
            if re.search(r'\bscc\b', scc_text):
                # An operand is the incoming SCC (src_scc): the destination must read it before this instruction's own
                # SCC replaces it, so the destination is written first and the SCC after, from sources it did not change.
                if dname in ('m0', 'exec', 'exec_lo', 'vcc', 'vcc_lo', 'null') or any(
                        reg_range(t) and dst and reg_range(t)[0] == 's' and dst[1] <= reg_range(t)[1] <= dst[2]
                        for t in srcs):
                    raise Unsupported('%s reads src_scc with a destination that is also a source or a special register'
                                      % base)
                deferred_scc = scc_text
            else:
                # From the sources, before the destination is written (it can be a source).
                self.set_scc(scc_text)
        if shift_add:
            # carry = bits shifted out of S0, or a wrap of the 32-bit sum; read before the destination is written
            self.set_scc(shift_add_carry(base, values, expr))
        if dname == 'm0':
            self.sym.pop('m0', None)
            self.emit('m0 = %s;' % expr)
            if scc:
                # Through set_scc, so an SCC derived from a lane mask (s_and_b32 m0, vcc_lo, -1)
                # is tracked as varying like any other scalar result.
                self.set_scc(scc.replace('$', 'm0'), proof=scc.replace('$', '(%s)' % expr))
            return None
        if dname in ('exec', 'exec_lo', 'vcc', 'vcc_lo'):
            var = dname[:4].rstrip('_') if dname.startswith('exec') else 'vcc'
            if var == 'exec':
                copy = base in ('s_mov_b64', 's_mov_b32') and dname == 'exec' and self.exec_known and (syms[0] == ('execcopy',))
                if copy:
                    if hasattr(self, 'lane_exec_breaks') and self.helper_off_copy(srcs[0], syms[0]):
                        self.lane_exec_breaks.append((self.pc, '%s %s' % (base, ', '.join(ops))))
                        self.helper_restore = True
                    # A restored copy can re-activate lanes a WQM added: drop every VGPR proof.
                    self.wqm_exec = False  # and the copy can be older than the WQM
                    if hasattr(self, 'uniform_v'):
                        self.uniform_v.clear()
                    if isinstance(self.align4_v, set):
                        self.align4_v.clear()
                    return None
                self.exec_known = self.exec_narrow_only = False
                if (hasattr(self, 'lane_exec_breaks')
                        and getattr(self, 'pc', None) not in getattr(self, 'reduction_restores', ())
                        and (dname == 'exec_lo' or not self.exec_write_exact(srcs, syms))):
                    # exec = (expr) & 1 writes every pixel and helper lane alike; the guest enables only the lanes the
                    # source selects, so the lane's EXEC bit may stop following the guest's from here.
                    self.lane_exec_breaks.append((self.pc, '%s %s' % (base, ', '.join(ops))))
            self.emit('%s = (%s) & 1;' % (var, expr))
            if constant_pair is not None:
                # Keep the numeric mask assignment and also make the literal words available to a resource read.
                self.sym[106], self.sym[107] = constant_pair
                self.vcc_constant_pair = True
                self.vcc_descriptor_overwritten = False
                if getattr(self, 'has_branches', False):
                    self.vcc_events.setdefault(self.pc, []).append(('load_constant_descriptor', constant_pair))
            if scc:
                self.set_scc(scc.replace('$', var))
            return None
        if dname == 'null':
            if scc:
                self.set_scc(scc.replace('$', '(%s)' % expr))
            return None
        if not dst or dst[0] != 's':
            raise Unsupported('%s destination %s' % (base, ops[0]))
        d = dst[1]
        is_mask = pair or any(t in ('exec', 'exec_lo', 'vcc', 'vcc_lo') or (reg_range(t) and reg_range(t)[0] == 's' and reg_range(t)[1] in self.masks)
                              for t in srcs) and base.split('_')[1] in ('mov', 'and', 'or', 'xor', 'andn2', 'orn2', 'nand', 'nor', 'xnor', 'not', 'cselect')
        if base in ('s_mov_b32', 's_mov_b64', 's_movk_i32') and re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', srcs[0]) and srcs[0] not in FLOATS:
            value = int(srcs[0], 0)
            self.set_s(d, values[0], ('const', value & 0xffffffff))
            if pair:
                self.set_s(d + 1, '0x%08xu' % ((value >> 32) & 0xffffffff), ('const', (value >> 32) & 0xffffffff))
            self.masks.discard(d)
            return None
        if base == 's_mov_b32' and not pair and len(srcs) == 1 and reg_range(srcs[0]) and reg_range(srcs[0])[0] == 's' \
                and reg_range(srcs[0])[1] == reg_range(srcs[0])[2] and len(self.sym.get(reg_range(srcs[0])[1]) or ()) == 2 \
                and self.is_const(reg_range(srcs[0])[1]) is not None:
            # A copy of an SGPR holding a plain known constant (not the tagged unassigned placeholder) is that constant (llc copies a zero into an inline
            # sampler word: s_mov_b32 s12, s15 after s_mov_b32 s15, 0).
            value = self.is_const(reg_range(srcs[0])[1])
            self.set_s(d, '0x%08xu' % value, ('const', value))
            self.masks.discard(d)
            return None
        if base in SCALAR_CONST_FOLD:
            # Constant operands give a known word, so a descriptor built from them (an inline sampler from
            # s_movk_i32, s_bfm_b32, s_mov_b32 and s_brev_b32 into s[N:N+3]) is a constant descriptor.
            def known(tok):
                r = reg_range(tok)
                if r and r[0] == 's' and r[1] == r[2]:
                    c = self.is_const(r[1])
                    return c if c is not None else getattr(self, '_pre_fconst', getattr(self, 'float_const_s', {})).get(r[1])
                if tok == 'null':
                    return 0
                if tok in FLOATS:
                    return fbits(FLOATS[tok])
                if re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', tok) and tok not in FLOATS:
                    return int(tok, 0) & 0xffffffff
                return None
            operands = [known(t) for t in srcs]
            if None not in operands:
                value = SCALAR_CONST_FOLD[base](*operands) & 0xffffffff
                self.set_s(d, '0x%08xu' % value, ('const', value))
                self.masks.discard(d)
                return None
        if base in ('s_mov_b64', 's_mov_b32') and srcs[0] in ('exec', 'exec_lo') and self.exec_known:
            self.set_s(d, 'exec')
            self.sym[d] = ('execcopy',)
            if not hasattr(self, 'execcopy_epoch'):
                self.execcopy_epoch = {}
            self.execcopy_epoch[d] = getattr(self, 'wqm_count', 0)   # the s_wqm count when the mask was saved
            self.masks.add(d)
            if pair:
                self.sym.pop(d + 1, None)
                if self.tracks_uniformity():
                    self.varying_s.add(d + 1)
            return None
        if is_mask:
            expr = '(%s) & 1' % expr
        if base == 's_mov_b64':
            r = reg_range(ops[1])
            high = r[2] if r and r[1] != r[2] else None
            # Classify the source high word before a destination write can change its status.
            high_source = 's%d' % high if high is not None else srcs[0]
            high_uniform = (self.tracks_uniformity() and high != d and
                            self.uniform_source(high_source))
        salu_align4 = not pair and not is_mask and self.salu_align4(base, srcs)
        self.set_s(d, expr, scc=None if lane_complement else scc, align4=salu_align4)
        if deferred_scc is not None:
            self.set_scc(deferred_scc)
        if base == 's_mov_b32' and srcs[0] in FLOATS and not is_mask:
            # Keep the runtime SGPR assignment intact while letting const_words() read inline float patterns
            # such as 0x40000000 (OPAQUE_BLACK, encoded as inline 244 / '2.0').
            self.__dict__.setdefault('float_const_s', {})[d] = fbits(FLOATS[srcs[0]])
        if lane_complement:
            # The guest SCC is (the whole complemented word != 0), which the bits of every other lane decide (an
            # idle lane's bit is set). It is not emitted: translate() refuses a program that can read it.
            self.scc_effect = 'a lane-mask complement'
            if self.tracks_uniformity():
                self.scc_uniform = False
        if hasattr(self, 'heap_index'):
            self.mark_heap_index(base, d, srcs, syms)
        if is_mask:
            self.masks.add(d)
        if pair and base != 's_mov_b64':
            self.set_s(d + 1, '0u')
        elif pair:
            self.set_s(d + 1, self.S(high) if high is not None else '0u', uniform=high_uniform)
        return None

    def heap_word_offset(self, tok, sym):
        """(param, word, scale, add) when tok, whose symbol before this instruction is sym, holds root constant word
        (param, word) times scale plus add, modulo 2^32 (see mark_heap_index); else None."""
        if sym and sym[0] == 'ud' and self.pipe.uds[sym[1]][0] == 'rootconst':
            return self.pipe.uds[sym[1]][1:] + (1, 0)
        r = reg_range(tok)
        mark = {**self.heap_index, **self.heap_index_was}.get(r[1]) if r and r[0] == 's' and r[1] == r[2] else None
        return mark[1:] if mark and mark[0] == 'rc' else None

    def mark_heap_index(self, base, d, srcs, syms):
        """Marks of SGPR d for heap_record. index << 5 is the byte offset of a 32-byte resource-heap record; the index
        is named when it is a root constant word (the dispatch knows it) or a word an s_buffer_load wrote (see
        op_s_buffer_load_dword). A root constant word times a literal plus a literal ('rc') is the byte offset such a
        load may read the index word at: the same for every thread of the dispatch."""
        lit = [int(t, 0) & 0xffffffff if re.fullmatch(r'0x[0-9a-f]+|[0-9]+', t) else None for t in srcs]
        r = reg_range(srcs[0])
        if base == 's_lshl_b32' and lit[1:] == [5]:
            word = {**self.heap_index, **self.heap_index_was}.get(r[1]) if r and r[0] == 's' else None
            if syms[0] and syms[0][0] == 'ud' and self.pipe.uds[syms[0][1]][0] == 'rootconst':
                self.heap_index[d] = self.pipe.uds[syms[0][1]]
            elif word and word[0] == 'bufword':
                self.heap_index[d] = ('buffer',) + word[1:]
            elif word == ('heap_index_word',):
                self.heap_index[d] = ('dynamic_buffer',)
        elif base in ('s_add_u32', 's_add_i32', 's_mul_i32') and len(srcs) == 2:
            for i in (0, 1):
                k = lit[1 - i]
                offset = self.heap_word_offset(srcs[i], syms[i]) if k is not None else None
                if offset:
                    p, w, scale, add = offset
                    self.heap_index[d] = (('rc', p, w, scale * k & 0xffffffff, add * k & 0xffffffff) if base == 's_mul_i32'
                                          else ('rc', p, w, scale, add + k & 0xffffffff))
                    break

    def set_exec_scc(self, expr):
        self.exec_known = self.exec_narrow_only = False
        self.scc_effect = 'emitted'
        self.scc_verified = True
        self.emit('exec = (%s) & 1; scc = exec != 0;' % expr)
        if hasattr(self, 'scc_uniform'):
            self.scc_uniform = False

    def op_s_and_saveexec_b64(self, ops, mods, text='exec & {0}'):
        if len(ops) != 2:
            raise Unsupported('saveexec with %d operands, expected 2' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's':
            raise Unsupported('saveexec destination %s' % ops[0])
        d = dst[1]
        src_reg = reg_range(ops[1])
        src = ops[1] if not src_reg else 's%d' % src_reg[1]
        # S0 is read at issue, before D is written (RDNA 2 ISA SOP1 S_AND_SAVEEXEC: D = EXEC; EXEC = S0 & EXEC).
        # With D == S0 (s_and_saveexec_b64 s[6:7], s[6:7]) writing D first would make the new exec exec & exec, so
        # S0 is copied to the scratch word w before D is written.
        if src_reg and src_reg[0] == 's' and src_reg[1] == d:
            self.emit('w = %s;' % self.u(src))
            value = 'w'
        else:
            value = self.u(src)
        self.set_s(d, 'exec')
        self.masks.add(d)
        self.set_exec_scc(text.format(value))

    op_s_and_saveexec_b32 = op_s_and_saveexec_b64

    def op_s_or_saveexec_b64(self, ops, mods):
        self.op_s_and_saveexec_b64(ops, mods, 'exec | {0}')

    op_s_or_saveexec_b32 = op_s_or_saveexec_b64

    def op_s_andn2_saveexec_b64(self, ops, mods):
        self.op_s_and_saveexec_b64(ops, mods, '{0} & ~exec')

    def op_s_andn2_wrexec_b64(self, ops, mods):
        if len(ops) != 2:
            raise Unsupported('wrexec with %d operands, expected 2' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's':
            raise Unsupported('wrexec destination %s' % ops[0])
        d = dst[1]
        src = ops[1] if not reg_range(ops[1]) else 's%d' % reg_range(ops[1])[1]
        self.set_exec_scc('%s & ~exec' % self.u(src))
        self.set_s(d, 'exec')
        self.masks.add(d)

    op_s_andn2_wrexec_b32 = op_s_andn2_wrexec_b64

    def scalar_accumulate(self, base, ops, mods):
        """s_addk_i32 / s_mulk_i32: D = D op signext(SIMM16), the destination is also the source.

        RDNA 2 ISA 12.2 (SOPK 15, 16): addk sets SCC to the signed overflow of the sum, mulk leaves SCC
        alone. LLVM 15 SOPInstructions.td declares Defs = [SCC] for both, so after mulk the SCC proof is
        dropped (the emitted scc keeps its value, which is the ISA meaning).
        """
        if mods or len(ops) != 2:
            raise Unsupported('%s with %d operands' % (base, len(ops)))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('%s destination %s' % (base, ops[0]))
        if not re.fullmatch(r'0x[0-9a-f]{1,4}', ops[1]):
            raise Unsupported('%s immediate %s' % (base, ops[1]))
        d = dst[1]
        # D is read before it is written: refuse a mask or descriptor source while the old state is intact.
        if d in self.masks or self.sym.get(d, ('',))[0] == 'execcopy':
            raise Unsupported('%s on a lane mask' % base)
        sym = self.sym.get(d)
        if sym and sym[0] not in ('const', 'ud', 'fixed_buffer_word'):
            raise Unsupported('%s on %s data' % (base, sym[0]))
        simm = int(ops[1], 16)
        imm = (simm - 0x10000 if simm & 0x8000 else simm) & 0xffffffff
        imm_text = '0x%08xu' % imm
        old = self.S(d)
        add = base == 's_addk_i32'
        result = '%s %s %s' % (old, '+' if add else '*', imm_text)
        if add:
            # signed overflow: the operands share a sign that the sum does not keep
            overflow = '((%s ^ (%s)) & ~(%s ^ %s) & 0x80000000u) != 0u' % (old, result, old, imm_text)
            if sym and sym[0] == 'const':
                total = (sym[1] + imm) & 0xffffffff
                overflow = 'true' if ((sym[1] ^ total) & ~(sym[1] ^ imm) & 0x80000000) else 'false'
            self.set_scc(overflow)
        elif self.tracks_uniformity():
            self.scc_uniform = False
        if sym and sym[0] == 'const':
            value = (sym[1] + imm if add else sym[1] * imm) & 0xffffffff
            self.set_s(d, '0x%08xu' % value, ('const', value))
        else:
            was_align4 = d in self.align4_s and sym is None
            acc_align4 = (was_align4 and (imm & 3) == 0) if add else (was_align4 or (imm & 3) == 0)
            self.set_s(d, result, align4=acc_align4)
        offset = self.heap_word_offset(ops[0], sym) if hasattr(self, 'heap_index') else None
        if offset:
            p, w, scale, plus = offset
            self.heap_index[d] = ('rc', p, w, scale, plus + imm & 0xffffffff) if add else (
                'rc', p, w, scale * imm & 0xffffffff, plus * imm & 0xffffffff)
        self.masks.discard(d)
        return None

    def op_s_addk_i32(self, ops, mods):
        return self.scalar_accumulate('s_addk_i32', ops, mods)

    def op_s_mulk_i32(self, ops, mods):
        return self.scalar_accumulate('s_mulk_i32', ops, mods)

    def op_s_cselect_b32(self, ops, mods):
        self.scalar_select(ops, False)

    def op_s_cselect_b64(self, ops, mods):
        self.scalar_select(ops, True)

    def op_s_cmov_b32(self, ops, mods):
        self.scalar_cmov(ops, mods, False)

    def op_s_cmov_b64(self, ops, mods):
        self.scalar_cmov(ops, mods, True)

    def op_s_cmovk_i32(self, ops, mods):
        # SOPK form: the 16-bit immediate is sign-extended to 32 bits.
        if len(ops) != 2 or mods:
            raise Unsupported('s_cmovk_i32 requires a destination and a 16-bit immediate')
        value = sopk_imm(ops[1], True)
        self.scalar_cmov([ops[0], '0x%x' % value], mods, False, 's_cmovk_i32')

    def op_s_bitset0_b32(self, ops, mods):
        return self.scalar_bitset(ops, mods, clear=True)

    def op_s_bitset1_b32(self, ops, mods):
        return self.scalar_bitset(ops, mods, clear=False)

    def scalar_bitset(self, ops, mods, clear):
        """s_bitset0/1_b32: clear or set bit S0[4:0] of D in place; SCC unchanged.

        RDNA 2 ISA SOP1 0x1b / 0x1d (and the pixel_aot / #1579 lowering): D.u &= ~(1u << (S0.u & 31))
        or D.u |= (1u << (S0.u & 31)). The destination is also a source (READ_MODIFY_WRITE). Only a
        plain user SGPR destination is accepted; lane masks and descriptor-bearing SGPRs stay refused
        so a partial word cannot be treated as a full 32-bit scalar.
        """
        name = 's_bitset%d_b32' % (0 if clear else 1)
        if mods or len(ops) != 2:
            raise Unsupported('%s with %d operands, expected 2 unmodified' % (name, len(ops)))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('%s destination %s' % (name, ops[0]))
        d = dst[1]
        if self.is_lane_mask(ops[0]) or d in self.masks:
            raise Unsupported('%s on a lane mask' % name)
        imm = re.fullmatch(r'-?(0x[0-9a-fA-F]+|\d+)', ops[1])
        bit_sym = None
        if ops[1] == 'null':
            index_imm = 0
        elif imm:
            index_imm = int(ops[1], 0) & 31
        else:
            src = reg_range(ops[1])
            if not src or src[0] != 's' or src[1] != src[2] or src[1] > 105:
                raise Unsupported('%s source %s' % (name, ops[1]))
            if self.is_lane_mask(ops[1]) or src[1] in self.masks:
                raise Unsupported('%s on a lane mask' % name)
            bit_sym = self.sym.get(src[1])
            if bit_sym and bit_sym[0] not in ('const', 'ud', 'fixed_buffer_word'):
                raise Unsupported('%s on %s data' % (name, bit_sym[0]))
            index_imm = bit_sym[1] & 31 if bit_sym and bit_sym[0] == 'const' else None
        sym = self.sym.get(d)
        if sym and sym[0] not in ('const', 'ud', 'fixed_buffer_word'):
            raise Unsupported('%s on %s data' % (name, sym[0]))
        prev = self.S(d)
        bit = self.u(ops[1])
        mask = '(1u << (%s & 31))' % bit
        result = '(%s & ~%s)' % (prev, mask) if clear else '(%s | %s)' % (prev, mask)
        if sym and sym[0] == 'const' and index_imm is not None:
            value = ((sym[1] & ~(1 << index_imm)) if clear else (sym[1] | (1 << index_imm))) & 0xffffffff
            self.set_s(d, '0x%08xu' % value, ('const', value))
        else:
            self.set_s(d, result)
        self.masks.discard(d)
        return None

    def scalar_cmov(self, ops, mods, pair, name=None):
        """D = SCC ? S0 : D on plain SGPR destinations; SCC is unchanged."""
        name = name or 's_cmov_b%d' % (64 if pair else 32)
        if len(ops) != 2 or mods:
            raise Unsupported('%s requires two unmodified operands' % name)
        dst, src = reg_range(ops[0]), reg_range(ops[1])
        if not dst or dst[0] != 's' or dst[2] > 105 or dst[2] - dst[1] != int(pair) or (pair and dst[1] % 2):
            raise Unsupported('%s destination %s' % (name, ops[0]))
        if pair and (not src or src[0] != 's' or src[2] > 105 or src[1] % 2 or src[2] != src[1] + 1):
            raise Unsupported('%s source %s' % (name, ops[1]))
        # A pair holds two words: a lane mask in either half cannot be moved as a full 32-bit scalar.
        halves = 2 if pair else 1
        regs = [('s%d' % (dst[1] + i)) for i in range(halves)]
        if src and src[0] == 's':
            regs += [('s%d' % (src[1] + i)) for i in range(halves)]
        if self.is_lane_mask(ops[0]) or self.is_lane_mask(ops[1]) or any(self.is_lane_mask(r) for r in regs):
            raise Unsupported('%s on a lane mask' % name)
        d = dst[1]
        low = ('s%d' % src[1]) if src and src[0] == 's' else ops[1]
        old = [self.S(d + i) for i in range(2 if pair else 1)]
        new = [self.u(low)]
        if pair:
            new.append(self.S(src[1] + 1))
        for i, (was, now) in enumerate(zip(old, new)):
            self.set_s(d + i, 'scc ? %s : %s' % (now, was))
        self.masks.discard(d)

    def scalar_select(self, ops, pair):
        if len(ops) != 3:
            raise Unsupported('scalar select with %d operands, expected 3' % len(ops))
        low = [('s%d' % reg_range(t)[1]) if reg_range(t) and reg_range(t)[0] == 's' else t for t in ops]
        if pair and 'm0' in low:
            raise Unsupported('64-bit scalar operand m0')
        if (not pair and hasattr(self, 'descriptor_scc_selects') and reg_range(low[0])
                and reg_range(low[0])[0] == 's' and reg_range(low[0])[1] <= 105
                and not any(self.is_lane_mask(token) for token in low)):
            self.descriptor_scc_selects.add(self.pc)
        expr = 'scc ? %s : %s' % (self.u(low[1]), self.u(low[2]))
        if low[0] == 'm0':
            self.sym.pop('m0', None)
            self.emit('m0 = %s;' % expr)
            return
        if low[0] in ('exec', 'vcc', 'exec_lo', 'vcc_lo'):
            # In wave32, exec_lo and vcc_lo hold the whole lane mask, as the generic scalar path reads them.
            mask = 'exec' if low[0].startswith('exec') else 'vcc'
            if mask == 'exec':
                self.exec_known = self.exec_narrow_only = False
            self.emit('%s = (%s) & 1;' % (mask, expr))
            return
        r = reg_range(low[0])
        if not r or r[0] != 's':
            raise Unsupported('s_cselect into %s' % low[0])
        d = r[1]
        self.set_s(d, '(%s) & 1' % expr if pair else expr)
        if pair:
            self.masks.add(d)
            self.set_s(d + 1, '0u')

    def smem_soffset(self, tok, record_use=True):
        """Runtime expression for an SMEM scalar offset operand (s0..s105 or m0)."""
        if tok == 'm0':
            return self.u(tok)
        r = reg_range(tok)
        if r and r[0] == 's' and r[1] == r[2] and r[1] <= 105:
            if record_use:
                return self.S(r[1])
            saved = set(self.used_s)
            try:
                return self.S(r[1])
            finally:
                self.used_s = saved
        raise Unsupported('operand %r' % tok)

    def const_soffset(self, tok):
        """The constant held by an SMEM offset register s0..s105, or None (m0 and unknown values)."""
        r = reg_range(tok)
        # Symbols are not merged at control-flow joins, so a constant proves the runtime value only without branches.
        if getattr(self, 'has_branches', True):
            return None
        if r and r[0] == 's' and r[1] == r[2] and r[1] <= 105:
            # A tagged constant (the ('const', 0, 'unassigned') default-sampler placeholder) is not a run-time value.
            sym = self.sym.get(r[1])
            value = self.is_const(r[1]) if sym and len(sym) == 2 else None
            if value is not None and 0 <= value <= 0xfffff:
                return value
        return None

    def op_s_load_dwordx4(self, ops, mods, count=4):
        imm = smem_mods(mods)
        if len(ops) != 3:
            raise Unsupported('s_load with %d operands' % len(ops))
        dst, base = reg_range(ops[0]), reg_range(ops[1])
        if not dst or dst[0] != 's' or dst[2] - dst[1] + 1 != count or dst[1] > 105 or dst[2] > 107:
            raise Unsupported('s_load destination %s for %d dwords' % (ops[0], count))
        # s106/s107 are vcc: only a descriptor load, which emits no code, may end there (s[104:107]).
        into_vcc = dst[2] > 105
        if into_vcc and (dst[1] > 104 or dst[2] != 107):
            raise Unsupported('s_load destination %s for %d dwords' % (ops[0], count))
        if not base or base[0] != 's' or base[2] - base[1] != 1 or base[2] > 105:
            raise Unsupported('s_load base %s' % ops[1])
        # A loaded descriptor or word is not a lane mask; some forms below only record a symbol, not set_s.
        for n in range(dst[1], dst[2] + 1):
            getattr(self, 'masks', set()).discard(n)
        p = self.sym.get(base[1])
        if not p or p[0] != 'ptr':
            raise Unsupported('s_load through s%d holding %s' % (base[1], p))
        waterfall = count == 8 and getattr(self, 'pc', None) in getattr(self, 'active_waterfall_loads', ())
        if waterfall and p[1][0] != 'table':
            raise Unsupported('compute readfirstlane waterfall: s_load_dwordx8 requires a fixed descriptor table')
        if len(p) > 2 and not isinstance(p[2], int):
            # The 'added' mark (an add whose offset is not tracked, as in a program with branches): no byte offset.
            raise Unsupported('s_load through s%d: the pointer was advanced by an untracked offset' % base[1])
        base_off = p[2] if len(p) > 2 else 0   # literal bytes added by s_add_u32 on the pointer
        if (base_off or len(p) > 2) and p[1][0] in PTR64_SOURCES:
            # An advanced 64-bit pointer needs its high word completed by the matching s_addc_u32;
            # a table pointer is 32 bits wide and has no high word.
            if base[1] in getattr(self, 'ptr_open_pairs', ()) or self.sym.get(base[1] + 1) != ptr_high_symbol(p[1], base_off):
                raise Unsupported('s_load through s%d: its high word is not the matching advanced pair' % base[1])
        # 'off' names no offset operand at all, so the byte offset is zero, as for the literal 0.
        token = smem_offset(ops[2])
        known = self.const_soffset(token) if p[1] != ('smpheap',) else None
        if known is not None:
            # A register offset that holds a known constant (s_movk before the load) is that literal byte
            # offset; the modifier adds to it. This reaches descriptors in a bindless heap (#1723).
            total = known + (imm or 0)
            if total > 0xfffff:
                raise Unsupported('SMEM register offset %#x' % total)
            token, imm = str(total), None
        if re.fullmatch(r'-?(?:0[xX][0-9a-fA-F]+|\d+)', token):
            off = int(token, 0)
            if off < 0 or off > 0xfffff:
                raise Unsupported('SMEM immediate offset %s' % ops[2])
            if imm is not None:
                raise Unsupported('SMEM literal offset %s with offset:%#x' % (ops[2], imm))
            if p[1][0] == 'root':
                if into_vcc:
                    raise Unsupported('s_load destination %s for %d dwords' % (ops[0], count))
                buf = self.pipe.resource((p[1], None), 'buf')
                for i in range(count):
                    self.set_s(dst[1] + i, '%s.Load(%d)' % (buf, base_off + off + 4 * i))
                    if getattr(self, 'kind', None) == 'cs' and hasattr(self, 'heap_index'):
                        self.heap_index[dst[1] + i] = ('heap_index_word',)
                return
            # A compute or vertex raw-store conversion must distinguish loads through different high words
            # even when the source and byte offset match. Keep the load PC for a changed pointer pair.
            # A 32-bit table pointer carries a high word the program sets to a constant: loads through the same
            # untouched low word and the same constant read the same words (tag -1 - constant).
            hi = self.sym.get(base[1] + 1)
            if getattr(self, 'kind', None) not in ('cs', 'vs') or (p == ('ptr', p[1]) and hi == ('ptrhi', p[1])):
                load = ()
            elif p == ('ptr', p[1]) and hi and hi[0] == 'const' and len(hi) == 2:
                load = (-1 - hi[1],)
            else:
                load = (self.pc,)
            if waterfall and (base_off | off) & 3:
                raise Unsupported('compute readfirstlane waterfall: descriptor table address requires dword alignment')
            for i in range(count):
                desc = ('desc', p[1], base_off + off, i)
                if len(p) > 2 and hasattr(self, 'advanced_descriptor_words'):
                    self.advanced_descriptor_words.add(desc)
                self.sym[dst[1] + i] = desc + load
            if into_vcc:
                self.mark_vcc_descriptor()
            return
        if into_vcc:
            raise Unsupported('s_load destination %s for %d dwords' % (ops[0], count))
        if base_off:
            raise Unsupported('s_load through %s advanced by %#x' % (p[1], base_off))
        if p[1] == ('smpheap',):
            if self.pipe.sampler_heap and count == 4:
                # A sampler picked at run time from the sampler heap: keep the byte offset in the
                # first destination register; image() indexes the heap table with it.
                d = dst[1]
                self.used_s.add(d)
                expr = self.smem_soffset(token)
                if imm:
                    expr = '%s + %d' % (expr, imm)
                self.emit('s%d = %s;' % (d, expr))
                for i in range(count):
                    self.sym[d + i] = ('smpheap', d, i)
                return
            self.smem_soffset(token, record_use=False)
            if count != 4:
                raise Unsupported('s_load of %d dwords from %s' % (count, p[1]))
            # Shortcut (default unless X2W_SAMPLER_HEAP=1): a fixed default sampler.
            for i in range(count):
                # Never assigned at run time (the third field says so): a join that reads it refuses (run_time_symbol).
                self.sym[dst[1] + i] = ('const', 0, 'unassigned')
            self.pipe.notes.add('dynamic sampler index replaced by a default sampler')
            return
        if p[1] == ('resheap',) and getattr(self, 'kind', None) == 'cs':
            return self.heap_record(dst, base, token, imm, count)
        if p[1] == ('resheap',) and not base_off and getattr(self, 'kind', None) in ('vs', 'ps'):
            return self.dynamic_heap_record(dst, base, token, imm, count,
                                            "s_load with a register offset through ('resheap',)")
        raise Unsupported('s_load with a register offset through %s' % (p[1],))

    def dynamic_heap_record(self, dst, base, token, imm, count, refusal):
        r = reg_range(token)
        if (count != 4 or dst[2] > 105 or self.has_branches or len(self.sym[base[1]]) != 2 or not r or r[0] != 's'
                or not self.uniform_source(token) or self.sym.get(base[2]) != ('const', 0)):
            raise Unsupported(refusal)
        name = 'heap_index_%x' % self.pc
        self.emit('uint %s = (%s + %du) >> 5;' % (name, self.smem_soffset(token), imm or 0))
        for i in range(count):
            self.sym[dst[1] + i] = ('desc', ('dynamic_heap', name), 0, i)

    def dynamic_heap_load(self, key, address):
        if key[1] != 0 or key in self.pipe.modified:
            raise Unsupported('dynamic resource heap descriptor was advanced')
        self.pipe.uses_dynamic_heap = True
        return 'x2w_heap_load(%s, %s)' % (key[0][1], address)

    def heap_record(self, dst, base, token, imm, count):
        """s_load_dwordx4 D, <resource heap pointer>, R in a compute program, where R is an index shifted left by 5
        with no later write of R and no branch target in between: D is the buffer descriptor in the 32-byte heap
        record that the index names. The index is a root constant word, or a word an s_buffer_load read from an
        unmodified table buffer at a literal byte offset or at a root constant word times a literal plus a literal
        (the marks, see mark_heap_index, end at a branch target). The manifest names the root constant or where the
        word is; the dispatch reads the word and the record. A uniform word from a non-table buffer instead
        uses a shader-side lookup."""
        what = 's_load of a resource heap record'
        r = reg_range(token)
        index = getattr(self, 'heap_index', {}).get(r[1]) if r and r[0] == 's' and r[1] == r[2] else None
        if index == ('dynamic_buffer',):
            return self.dynamic_heap_record(dst, base, token, imm, count,
                                            what + ' whose index is read from a buffer other than a table buffer '
                                            'at a dispatch-constant offset')
        if index == ('buffer',):
            raise Unsupported(what + ' whose index is read from a buffer other than a table buffer '
                              'at a dispatch-constant offset')
        if not index or index[0] not in ('rootconst', 'buffer'):
            raise Unsupported('%s whose offset is not a root constant shifted left by 5 since the last branch target' % what)
        if count != 4:
            raise Unsupported('%s of %d dwords' % (what, count))
        if imm is not None:
            raise Unsupported('%s with offset:%#x' % (what, imm))
        if self.sym.get(base[2]) != ('const', 0):
            raise Unsupported('%s through a pointer whose high word is not a constant zero' % what)
        source = ('resheap',) + index[1:] if index[0] == 'rootconst' else ('resheap', 'word') + index[1:]
        for i in range(count):
            self.sym[dst[1] + i] = ('desc', source, 0, i)

    def op_s_bfm_b32(self, ops, mods):
        """Fold constant bitfield masks: titles build inline sampler words with them."""
        if len(ops) != 3:
            raise Unsupported('s_bfm_b32 with %d operands, expected 3' % len(ops))
        vals = []
        for o in ops[1:3]:
            r = reg_range(o)
            if r and r[0] == 's':
                vals.append(self.is_const(r[1]))
            elif o == 'null':
                vals.append(0)
            else:
                try:
                    vals.append(int(o, 0))
                except ValueError:
                    vals.append(None)
        if ops[0] == 'm0':
            if None not in vals:
                value = (((1 << (vals[0] & 31)) - 1) << (vals[1] & 31)) & 0xffffffff
                expr = '0x%08xu' % value
            else:
                expr = '(((1u << (%s & 31)) - 1) << (%s & 31))' % (self.u(ops[1]), self.u(ops[2]))
            self.sym.pop('m0', None)
            self.emit('m0 = %s;' % expr)
            return None
        if ops[0] == 'null':
            if any(self.is_lane_mask(t) for t in ops[1:3]):
                raise Unsupported('s_bfm_b32 into null on a lane mask')
            self.u(ops[1])
            self.u(ops[2])
            return None
        dst = reg_range(ops[0])
        if not dst:
            raise Unsupported('s_bfm_b32 destination %s' % ops[0])
        d = dst[1]
        if None not in vals:
            value = (((1 << (vals[0] & 31)) - 1) << (vals[1] & 31)) & 0xffffffff
            self.set_s(d, '0x%08xu' % value, ('const', value))
            return None
        self.set_s(d, '(((1u << (%s & 31)) - 1) << (%s & 31))' % (self.u(ops[1]), self.u(ops[2])))
        return None

    def op_s_bfm_b64(self, ops, mods):
        """s_bfm_b64: D.u64 = ((1ULL << S0.u[5:0]) - 1) << S1.u[5:0], SCC unchanged.

        Unlike the mask-style 64-bit scalar ops (which keep one lane bit), this is a real 64-bit value, so both
        destination words are written. The width is masked to six bits here and the offset inside shl64, which
        also drops the bits shifted past bit 63; every HLSL shift count stays below 32."""
        if mods:
            raise Unsupported('s_bfm_b64 with %s' % (mods,))
        if len(ops) != 3:
            raise Unsupported('s_bfm_b64 with %d operands, expected 3' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[2] != dst[1] + 1 or dst[1] % 2 or dst[2] > 105:
            raise Unsupported('s_bfm_b64 destination %s' % ops[0])
        vals = []
        for o in ops[1:]:
            r = reg_range(o)
            if o in ('exec', 'exec_lo', 'exec_hi', 'vcc', 'vcc_lo', 'vcc_hi') or (
                    r and (r[0] != 's' or r[1] != r[2] or r[1] > 105)):
                raise Unsupported('s_bfm_b64 source %s' % o)
            if r and r[1] in self.masks:
                # A lane mask is not a number. Every source is read and checked before any destination state
                # changes, so a source aliasing the destination cannot lose its mask flag first.
                raise Unsupported('s_bfm_b64 source %s holds a lane mask' % o)
            if r:
                vals.append(self.is_const(r[1]))
            else:
                try:
                    vals.append(int(o, 0))
                except ValueError:
                    vals.append(None)
        d = dst[1]
        if None in vals:
            width, offset = self.u(ops[1]), self.u(ops[2])  # may refuse; still before any write
        # Both words are uniform only if both sources are. Judged before the destination changes, so an aliased
        # source is classified by its old value and an older proof for s[d] or s[d+1] cannot survive.
        uniform = self.tracks_uniformity() and all(self.uniform_source(o) for o in ops[1:])
        self.masks.discard(d)
        self.masks.discard(d + 1)
        if None not in vals:
            value = (((1 << (vals[0] & 63)) - 1) << (vals[1] & 63)) & 0xffffffffffffffff
            for i, word in enumerate((value & 0xffffffff, value >> 32)):
                self.set_s(d + i, '0x%08xu' % word, ('const', word))
            return None
        # Both words come from locals computed before either destination is written, so a source may be the destination.
        self.sym.pop(d, None)
        self.sym.pop(d + 1, None)
        if self.tracks_uniformity():
            for word in (d, d + 1):
                self.record_scalar_uniformity(word, uniform)
        self.used_s.update((d, d + 1))
        self.emit('{ uint bw = %s & 63u; uint2 t = shl64(bw >= 32u ? 0xffffffffu : (1u << (bw & 31u)) - 1u, '
                  'bw >= 32u ? (1u << (bw & 31u)) - 1u : 0u, %s); s%d = t.x; s%d = t.y; }' % (width, offset, d, d + 1))
        return None

    def scalar_const(self, tok):
        """Value of a 32-bit scalar source that is known at translation time, else None."""
        if tok == 'null':
            return 0
        r = reg_range(tok)
        if r:
            return self.is_const(r[1]) if r[0] == 's' and r[1] == r[2] else None
        try:
            return int(tok, 0) & 0xffffffff
        except ValueError:
            return None

    def scalar_wide_op(self, name, ops, mods, wide_src, wide_dst, helper, signed=None):
        """SOP2 shifts and bitfield extracts with a 64-bit source and/or result.

        D = f(S0, S1) with S1 a 32-bit word; SCC = (D != 0). Every source is read before any destination
        register is written, so an overlapping destination cannot feed a half-written value back."""
        if mods or len(ops) != 3:
            raise Unsupported('%s with %d operands and modifiers %s' % (name, len(ops), mods))
        dst = reg_range(ops[0])
        width = 1 + wide_dst
        if (not dst or dst[0] != 's' or dst[2] != dst[1] + wide_dst or dst[2] > 105
                or (wide_dst and dst[1] % 2)):
            raise Unsupported('%s destination %s' % (name, ops[0]))
        s1 = reg_range(ops[2])
        if ('m0' in ops[1:] or any(self.is_lane_mask(t) for t in ops[1:])
                or any(r[0] == 's' and any(w in self.masks for w in range(r[1], r[2] + 1))
                       for r in map(reg_range, ops[1:]) if r) or ops[2] in FLOATS or ops[1] in FLOATS
                or (s1 and (s1[0] != 's' or s1[1] != s1[2] or s1[1] > 105))):
            raise Unsupported('%s source %s' % (name, ', '.join(ops[1:])))
        if wide_src:
            r0 = reg_range(ops[1])
            if r0 and r0[0] != 's':
                raise Unsupported('%s source %s' % (name, ops[1]))
            low, high = self.i64_words(ops[1], helper == 'shr64i' or signed is True, True)
            known = [self.scalar_const('s%d' % r0[1]), self.scalar_const('s%d' % r0[2])] if r0 else [
                int(low[:-1]), int(high[:-1])]
        else:
            low, high = self.u(ops[1]), None
            known = [self.scalar_const(ops[1])]
        amount = self.u(ops[2])
        amount_known = self.scalar_const(ops[2])
        d = dst[1]
        self.masks.difference_update(range(d, d + width))
        if None not in known and amount_known is not None:
            x = known[0] | (known[1] << 32 if wide_src else 0)
            value = scalar_wide_fold(helper, x, amount_known, signed)
            for i in range(width):
                word = (value >> (32 * i)) & 0xffffffff
                self.set_s(d + i, '0x%08xu' % word, ('const', word))
            self.set_scc('true' if value else 'false')
            return None
        uniform = (self.tracks_uniformity()
                   and self.scalar_expr_uniform('%s %s %s' % (low, high or '', amount)))
        sg = '' if signed is None else ', %du' % (1 if signed else 0)
        if wide_dst:
            call = '%s(%s, %s, %s%s)' % (helper, low, high if wide_src else '0u', amount, sg)
            self.emit('{ uint2 t = %s; s%d = t.x; s%d = t.y; scc = (t.x | t.y) != 0u; }' % (call, d, d + 1))
        else:
            self.emit('{ uint t = %s(%s, %s%s); s%d = t; scc = t != 0u; }' % (helper, low, amount, sg, d))
        for r in range(d, d + width):
            self.sym.pop(r, None)
            if isinstance(self.align4_s, set):
                self.align4_s.discard(r)
            self.used_s.add(r)
            if self.tracks_uniformity():
                (self.varying_s.discard if uniform else self.varying_s.add)(r)
        self.scc_effect = 'emitted'
        self.scc_verified = True
        if self.tracks_uniformity():
            self.scc_uniform = uniform
        return None

    def op_s_lshl_b64(self, ops, mods):
        return self.scalar_wide_op('s_lshl_b64', ops, mods, True, True, 'shl64')

    def op_s_lshr_b64(self, ops, mods):
        return self.scalar_wide_op('s_lshr_b64', ops, mods, True, True, 'shr64u')

    def op_s_ashr_i64(self, ops, mods):
        return self.scalar_wide_op('s_ashr_i64', ops, mods, True, True, 'shr64i')

    def op_s_bfe_i32(self, ops, mods):
        return self.scalar_wide_op('s_bfe_i32', ops, mods, False, False, 'bfe32', True)

    def op_s_bfe_u64(self, ops, mods):
        return self.scalar_wide_op('s_bfe_u64', ops, mods, True, True, 'bfe64', False)

    def op_s_bfe_i64(self, ops, mods):
        return self.scalar_wide_op('s_bfe_i64', ops, mods, True, True, 'bfe64', True)

    def op_s_load_dwordx8(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 8)

    def op_s_load_dwordx16(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 16)

    def op_s_load_dwordx2(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 2)

    def op_s_load_dword(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 1)

    def op_s_buffer_load_dword(self, ops, mods, count=1):
        imm = smem_mods(mods)
        if len(ops) != 3:
            raise Unsupported('s_buffer_load with %d operands' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[2] - dst[1] + 1 != count or dst[2] > 105:
            raise Unsupported('s_buffer_load destination %s for %d dwords' % (ops[0], count))
        key = self.descriptor(ops[1])
        token = smem_offset(ops[2])
        waterfall = getattr(self, 'pc', None) in getattr(self, 'active_waterfall_loads', ())
        if waterfall and key in self.pipe.rw_keys:
            raise Unsupported('compute readfirstlane waterfall: scalar buffer load requires a read-only descriptor')
        dynamic = key is not None and key[0][0] == 'dynamic_heap'
        buf = None if dynamic or key is None else self.pipe.resource(key, 'buf')
        is_lit = bool(re.fullmatch(r'-?(?:0[xX][0-9a-fA-F]+|\d+)', token))
        if is_lit:
            val = int(token, 0)
            if val < 0 or val > 0xfffff:
                raise Unsupported('SMEM immediate offset %s' % ops[2])
            if imm is not None:
                raise Unsupported('SMEM literal offset %s with offset:%#x' % (ops[2], imm))
            offset = self.u(token)
            extra = 0
        else:
            offset = self.smem_soffset(token)
            extra = imm or 0
            sreg = reg_range(token)
            if sreg and offset == ('s%d' % sreg[1]) and dst[1] <= sreg[1] < dst[2]:
                self.emit('a = %s;' % offset)
                offset = 'a'
        # A word read from an unmodified table buffer at a literal offset or at a root constant word times a literal
        # plus a literal is the same for the whole dispatch: heap_record names it by (table parameter, table byte,
        # root constant parameter and word or None, scale, byte offset). Any other buffer word is only marked.
        where = ()
        if hasattr(self, 'heap_index') and key is not None:
            key, r, sreg = self.descriptor(ops[1]), reg_range(ops[1]), reg_range(token)
            first = self.sym.get(r[1])
            at = (None, None, 0, val) if is_lit else self.heap_word_offset(
                token, self.sym.get(sreg[1]) if sreg and sreg[0] == 's' else None)
            if (at and at[2] % 4 == 0 and (at[3] + extra) % 4 == 0  # a dword-aligned word for any root constant
                    and first[0] == 'desc' and key[0][0] == 'table' and key not in getattr(self.pipe, 'modified', ())  # probes build a bare pipe
                    and all(self.sym.get(r[1] + n) == first[:3] + (first[3] + n,) + first[4:] for n in range(4))):
                where = key[0][1:] + key[1:] + at[:3] + (at[3] + extra,)
        # RDNA 2 scalar buffer addressing clears the low two bits of the final address.
        # The raw view starts at the aligned descriptor base. Align once before loading either word.
        if waterfall and (extra % 4 or not self.operand_align4(token)):
            self.emit('a = (%s + %du) & ~3u;' % (offset, extra))
            offset, extra = 'a', 0
        # All words use the original offset, even when a destination overwrites its register.
        uniform = self.tracks_uniformity() and self.uniform_source(token)
        for i in range(count):
            address = '%s + %d' % (offset, extra + 4 * i)
            expr = (self.dynamic_heap_load(key, '(%s) & ~3u' % address) if dynamic
                    else '%s.Load(%s)' % (buf, address) if buf else '0u')
            self.set_s(dst[1] + i, expr, uniform=uniform)
            if is_lit:
                self.sym[dst[1] + i] = ('fixed_buffer_word',)
            if hasattr(self, 'heap_index'):
                self.heap_index[dst[1] + i] = ('bufword',) + (where[:5] + ((where[5] + 4 * i) & 0xffffffff,) if where else ())
                if (getattr(self, 'kind', None) == 'cs' and uniform and key is not None and key[0][0] != 'table'
                        and key not in getattr(self.pipe, 'rw_keys', ())):
                    self.heap_index[dst[1] + i] = ('heap_index_word',)

    def op_s_buffer_load_dwordx2(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 2)

    def op_s_buffer_load_dwordx4(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 4)

    def op_s_buffer_load_dwordx8(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 8)

    def op_s_buffer_load_dwordx16(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 16)

    def buffer_address(self, ops, mods, *, dynamic=False):
        # MUBUF and MTBUF share the same address calculation: optional index * stride (idxen),
        # optional byte VGPR offset (offen), scalar byte offset (soffset) and 12-bit immediate
        # offset. Cache-policy flags (glc, slc, dlc) do not change the loaded bytes; lds, tfe
        # and any other modifier change the destination or operand layout and are refused first.
        offset = 0
        for m in mods:
            if m.startswith('offset:') and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                offset += int(m[7:], 0)
            elif m not in ('idxen', 'offen', 'glc', 'slc', 'dlc'):
                raise Unsupported('buffer modifier %s' % m)
        if len(ops) != 4:
            raise Unsupported('buffer load with %d operands, expected 4' % len(ops))
        embedded = dynamic and self.embedded_buffer and self.pc in self.embedded_buffer[1]
        if embedded:
            key, buf = None, 'embedded_vs_load'
        else:
            key = self.descriptor(ops[2])
            buf = None if key is None or (dynamic and key[0][0] == 'dynamic_heap') else self.pipe.resource(key, 'buf')
        soffset = None
        if re.fullmatch(r'0x[0-9a-f]+|\d+', ops[3]):
            offset += int(ops[3], 0)
        else:
            sreg = reg_range(ops[3])
            if not sreg or sreg[0] != 's' or sreg[1] != sreg[2] or sreg[1] > 105:
                raise Unsupported('buffer soffset %s' % ops[3])
            soffset = self.S(sreg[1])
        idxen, offen = 'idxen' in mods, 'offen' in mods
        terms = []
        if idxen and buf is None and key is not None:
            raise Unsupported('indexed load through a dynamic resource heap descriptor')
        if idxen and offen:
            if key in self.pipe.modified:
                raise Unsupported('indexed load through a modified descriptor')
            r = reg_range(ops[1])
            if not r or r[0] != 'v' or r[2] - r[1] != 1 or r[2] > 255:
                raise Unsupported('buffer vaddr %s for idxen offen' % ops[1])
            stride = 'ud%d' % self.pipe.ud(('stride', self.srv_index(key))) if buf else '0u'
            terms.append('%s * %s' % (self.u('v%d' % r[1]), stride))
            terms.append(self.u('v%d' % r[2]))
        elif idxen:
            if key in self.pipe.modified:
                raise Unsupported('indexed load through a modified descriptor')
            r = reg_range(ops[1])
            if not r or r[0] != 'v' or r[1] != r[2] or r[1] > 255:
                raise Unsupported('buffer vaddr %s for idxen' % ops[1])
            stride = 'ud%d' % self.pipe.ud(('stride', self.srv_index(key))) if buf else '0u'
            terms.append('%s * %s' % (getattr(self, 'uniform_vaddr', None) or self.u(ops[1]), stride))
        elif offen:
            r = reg_range(ops[1])
            if not r or r[0] != 'v' or r[1] != r[2] or r[1] > 255:
                raise Unsupported('buffer vaddr %s for offen' % ops[1])
            terms.append(getattr(self, 'uniform_vaddr', None) or self.u(ops[1]))
        elif ops[1] != 'off':
            raise Unsupported('buffer vaddr %s without idxen or offen' % ops[1])
        if soffset is not None:
            terms.append(soffset)
        if embedded:
            # Keep the VGPR and scalar terms separate so overflow cannot wrap
            # a large byte offset back into the table.
            return buf, '%s, %s, %du' % (terms[0] if offen else '0u', soffset or '0u', offset)
        terms.append('%d' % offset)
        return buf if buf is not None else key, ' + '.join(terms)

    def op_buffer_load_dword(self, ops, mods, count=1):
        # ops[0] is read before buffer_address() can refuse a short operand list, so an instruction with no
        # operands at all would reach it as an index error. Everything longer is left to the shared check,
        # which sees a modifier first: the lds form decodes to three operands and must still refuse on it.
        if not ops:
            raise Unsupported('buffer load with 0 operands, expected 4')
        dst = reg_range(ops[0])
        buf, address = self.buffer_address(ops, mods, dynamic=True)
        if buf == 'embedded_vs_load':
            self.set_v(dst[1], '%s(%s)' % (buf, address))
            return
        self.emit('a = %s;' % address)
        for i in range(count):
            expr = (self.dynamic_heap_load(buf, 'a + %d' % (4 * i)) if isinstance(buf, tuple)
                    else '%s.Load(a + %d)' % (buf, 4 * i) if buf else '0u')
            self.set_v(dst[1] + i, expr)

    def op_buffer_load_dwordx2(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 2)

    def op_buffer_load_dwordx3(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 3)

    def op_buffer_load_dwordx4(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 4)

    def op_global_load_dword(self, ops, mods):
        # Compute-stage one-dword GLOBAL load in the saddr form (RDNA 2 ISA §9.2, LLVM 15 SelectGlobalSAddr):
        # ADDR = SADDR[63:0] + zext(VADDR[31:0]) + sext(OFFSET[11:0]), where SADDR is a proven 64-bit root
        # buffer pointer pair s[base:base+1] (optionally copied or advanced by completed literal s_add_u32 /
        # s_addc_u32). Modifiers are checked before the operand count so the two-operand `lds` form refuses on
        # its modifier rather than arity.
        if getattr(self, 'kind', None) != 'cs':
            raise Unsupported('global_load_dword')
        offset = 0
        for m in mods:
            if m.startswith('offset:') and re.fullmatch(r'-?(?:0x[0-9a-f]+|\d+)', m[7:]):
                imm = int(m[7:], 0)
                if not -2048 <= imm <= 2047:
                    raise Unsupported('global load offset %s' % m[7:])
                offset += imm
            elif m not in ('glc', 'slc', 'dlc'):
                raise Unsupported('global load modifier %s' % m)
        if len(ops) != 3:
            raise Unsupported('global load with %d operands, expected 3' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[1] != dst[2] or dst[1] > 255:
            raise Unsupported('global load destination %s' % ops[0])
        if ops[2] == 'off':
            raise Unsupported('global load without a scalar address pair')
        base = reg_range(ops[2])
        if not base or base[0] != 's' or base[2] - base[1] != 1 or base[1] % 2 or base[2] > 105:
            raise Unsupported('global load base %s' % ops[2])
        vaddr = reg_range(ops[1])
        if not vaddr or vaddr[0] != 'v' or vaddr[1] != vaddr[2] or vaddr[1] > 255:
            raise Unsupported('global load vaddr %s' % ops[1])
        p = self.sym.get(base[1])
        if not p or p[0] != 'ptr':
            raise Unsupported('global load through s%d holding %s' % (base[1], p))
        if len(p) > 2 and not isinstance(p[2], int):
            raise Unsupported('global load through s%d: the pointer was advanced by an untracked offset' % base[1])
        if p[1][0] != 'root':
            raise Unsupported('global load through %s' % (p[1],))
        base_off = p[2] if len(p) > 2 else 0
        if not isinstance(base_off, int) or isinstance(base_off, bool) or base_off < 0:
            raise Unsupported('global load through s%d: invalid pointer byte offset %r' % (base[1], base_off))
        if base[1] in getattr(self, 'ptr_open_pairs', ()) or self.sym.get(base[2]) != ptr_high_symbol(p[1], base_off):
            raise Unsupported('global load through s%d: its high word is not the matching pair' % base[1])
        # RDNA 2 low-bit GLOBAL alignment depends on SH_MEM_CONFIG.ALIGNMENT_MODE (DWORD rounds down, STRICT /
        # DWORD_STRICT faults, UNALIGNED gathers across neighbouring DWORDs; Scarlett's setting is unmeasured),
        # and HLSL ByteAddressBuffer.Load requires a multiple of four. Require proven 4-byte-aligned components.
        if offset % 4:
            raise Unsupported('global load at unaligned immediate offset %d' % offset)
        if base_off % 4:
            raise Unsupported('global load through s%d at unaligned pointer offset %d' % (base[1], base_off))
        if not self.operand_align4(ops[1]):
            raise Unsupported('global load vaddr %s is not proven 4-byte aligned' % ops[1])
        total_off = base_off + offset
        if total_off > 0xffffffff or total_off < -0xffffffff:
            raise Unsupported('global load byte offset %#x' % total_off)
        buf = self.pipe.resource((p[1], None), 'buf')
        # Guard the signed 64-bit view-relative byte offset (zext(vaddr) + base_off + sext(offset)) against
        # [0, 0xffffffff] before converting to a 32-bit ByteAddressBuffer offset so overflow above 4 GiB or
        # underflow below base cannot wrap into an in-bounds word of a bounded root view.
        self.emit('a = %s;' % self.u(ops[1]))
        if total_off > 0:
            load_expr = '(a <= 0x%08xu ? %s.Load(a + %du) : 0u)' % (0xffffffff - total_off, buf, total_off)
        elif total_off < 0:
            mag = -total_off
            load_expr = '(a >= %du ? %s.Load(a - %du) : 0u)' % (mag, buf, mag)
        else:
            load_expr = '%s.Load(a)' % buf
        self.set_v(dst[1], load_expr)

    def op_buffer_store_vs(self, ops, mods, count):
        # Vertex-stage store in the `off` form: scalar byte offset plus the 12-bit immediate, through a
        # descriptor read from a fixed root-table slot, written to a RWByteAddressBuffer (issue #1723).
        # The manifest carries the `uav` line the runtime binds; a descriptor may not also be loaded.
        if self.kind != 'vs':
            raise Unsupported('buffer store in the %s stage' % self.kind)
        offset = 0
        for m in mods:
            if m.startswith('offset:') and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                offset += int(m[7:], 0)
            else:
                raise Unsupported('buffer store modifier %s' % m)
        if len(ops) != 4 or ops[1] != 'off':
            raise Unsupported('buffer store without the off address form')
        src = reg_range(ops[0])
        if not src or src[0] != 'v' or src[2] - src[1] != count - 1 or src[2] > 255:
            raise Unsupported('buffer store data %s' % ops[0])
        key = self.descriptor(ops[2])
        if key is None:
            return
        if key[0][0] != 'table' or key[1] is None:
            raise Unsupported('buffer store through %s' % (key,))
        if key in self.pipe.modified:
            # Only the modifications a compute store accepts: the stride clear or the structured-to-raw conversion.
            self.store_descriptor(ops[2], 4, 'buffer store', stride0_ok=True, raw_conversion_ok=True)
        if re.fullmatch(r'0x[0-9a-f]+|\d+', ops[3]):
            offset += int(ops[3], 0)
            soffset = ''
            if offset % 4:
                raise Unsupported('buffer store at unaligned byte offset %d' % offset)
        else:
            sreg = reg_range(ops[3])
            if not sreg or sreg[0] != 's' or sreg[1] != sreg[2] or sreg[1] > 105:
                raise Unsupported('buffer soffset %s' % ops[3])
            if offset % 4:
                raise Unsupported('buffer store at unaligned immediate offset %d' % offset)
            soffset = '%s + ' % self.S(sreg[1])
        uav = self.pipe.uav(key, 'raw')
        words = ', '.join(self.u('v%d' % n) for n in range(src[1], src[2] + 1))
        store = '%s.Store%s(a, %s);' % (uav, '' if count == 1 else str(count), 'uint%d(%s)' % (count, words) if count > 1 else words)
        # Store* needs a multiple of four: the immediate is checked above; a register offset is rounded down.
        self.emit('a = (%s%d) & 0xfffffffcu;' % (soffset, offset) if soffset else 'a = %d;' % offset)
        self.emit(store if self.exec_known else 'if (exec & 1) %s' % store)

    def store_descriptor(self, tok, words, what, stride0_ok=False, raw_conversion_ok=False):
        """Resource key for a single loaded descriptor or a proven raw transformation of it."""
        rsrc = reg_range(tok)
        first = self.sym.get(rsrc[1]) if rsrc and rsrc[0] == 's' and rsrc[2] - rsrc[1] == words - 1 else None
        if not first or first[0] != 'desc' or first[1][0] != 'table' or first[2] is None or first[3] % 4:
            raise Unsupported('%s through %s holding %s' % (what, tok, first))
        key = desc_key(first)
        loaded = tuple(desc_word(first, first[3] + k) for k in range(words))
        actual = tuple(self.sym.get(rsrc[1] + k) for k in range(words))
        valid = names_one_value(first) and actual == loaded
        if names_one_value(first) and stride0_ok and words == 4:
            cleared = loaded[1] + ('stride0',)
            valid = valid or actual == (loaded[0], cleared, *loaded[2:])
            if not valid and raw_conversion_ok:
                valid = actual == (loaded[0], cleared, loaded[2] + ('byte_count',), loaded[3] + ('oob3',))
        if (not valid or key in getattr(self.pipe, 'other_modified', ())
                or (key in self.pipe.modified and not stride0_ok)):
            raise Unsupported('%s through a modified descriptor' % what)
        return key

    def raw_table_uav_store(self, kind_label, ops, mods, data_count, *, allow_glc=False, allow_index=False,
                            raw_conversion_ok=False, uniform_off=False):
        """Validate a fixed-table raw UAV descriptor, data range, byte address, and 4-byte alignment."""
        offset = 0
        glc = False
        offen = 'offen' in mods
        for m in mods:
            if m.startswith('offset:') and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                offset += int(m[7:], 0)
            elif m == 'glc' and allow_glc:
                glc = True
            elif m == 'idxen' and allow_index and not offen:
                continue
            elif m != 'offen':
                raise Unsupported('%s modifier %s' % (kind_label, m))
        idxen = 'idxen' in mods
        if len(ops) != 4 or not (offen or idxen or (uniform_off and ops[1] == 'off')):
            raise Unsupported('%s without the offen address form' % kind_label)
        data = reg_range(ops[0])
        if not data or data[0] != 'v' or data[2] - data[1] != data_count - 1 or data[2] > 255:
            raise Unsupported('%s data %s' % (kind_label, ops[0]))
        if offen or idxen:
            vaddr = reg_range(ops[1])
            if not vaddr or vaddr[0] != 'v' or vaddr[1] != vaddr[2] or vaddr[1] > 255:
                raise Unsupported('%s vaddr %s for %s' % (kind_label, ops[1], 'offen' if offen else 'idxen'))
        if kind_label == 'buffer store' and self.null_descriptor(ops[2]):
            self.descriptor(ops[2])  # verify any overlapping VCC words on all paths
            return None, None, None, glc
        key = self.store_descriptor(ops[2], 4, kind_label, stride0_ok=not idxen, raw_conversion_ok=raw_conversion_ok)
        terms = [self.u(ops[1])] if offen else []
        if re.fullmatch(r'0x[0-9a-f]+|\d+', ops[3]):
            offset += int(ops[3], 0)
        else:
            sreg = reg_range(ops[3])
            if not sreg or sreg[0] != 's' or sreg[1] != sreg[2] or sreg[1] > 105:
                raise Unsupported('buffer soffset %s' % ops[3])
            terms.append(self.S(sreg[1]))
        if offset % 4:
            raise Unsupported('%s at unaligned immediate offset %d' % (kind_label, offset))
        uav = self.pipe.uav(key, rw=kind_label == 'buffer store')
        if idxen:
            # The raw UAV view starts at the descriptor base. Its entry constant carries DW1[31:16].
            stride = self.pipe.ud(('uav_stride', self.pipe.uavs[key]['index']))
            terms.insert(0, '%s * (ud%d & 0x3fffu)' % (self.u(ops[1]), stride))
        terms.append(str(offset))
        return data, uav, '(%s) & 0xfffffffcu' % ' + '.join(terms), glc

    def op_buffer_store_dword(self, ops, mods, count=1):
        # Compute-stage store in the offen form: per-lane byte VGPR plus the scalar offset and the 12-bit immediate,
        # through a descriptor read from a fixed root-table slot, to a RWByteAddressBuffer (manifest `uav` line).
        # Like the loads, this assumes a raw descriptor (no swizzle or stride). Out-of-range stores: assumes a future
        # runtime sizes the view to the descriptor's range so both sides drop them; nothing binds or checks that yet.
        if self.kind == 'vs':
            return self.op_buffer_store_vs(ops, mods, count)
        if self.kind != 'cs':
            raise Unsupported('buffer_store_dword' + ('x%d' % count if count > 1 else ''))  # the unknown-op refusal
        src, uav, address, _ = self.raw_table_uav_store('buffer store', ops, mods, count, raw_conversion_ok=True,
                                                        uniform_off=True)
        if uav is None:
            return
        # The `off` form (no VGPR address, MUBUF offen = idxen = 0) gives every lane the same byte address, so the
        # lanes race; with one proven value in all of them any order stores the same words.
        if ops[1] == 'off' and not all(self.proven_uniform_v(n) for n in range(src[1], src[2] + 1)):
            raise Unsupported('buffer store off with data not proven wave-uniform')
        words = [self.u('v%d' % n) for n in range(src[1], src[2] + 1)]
        data = words[0] if count == 1 else 'uint%d(%s)' % (count, ', '.join(words))
        # Store* needs a multiple of four. As in #2059, the run-time part (VGPR and register soffset) is rounded down:
        # an assumption that a dword access ignores the low two address bits, not a measurement.
        self.emit('a = %s;' % address)
        store = '%s.Store%s(a, %s);' % (uav, '' if count == 1 else count, data)
        self.emit(store if self.exec_known else 'if (exec & 1) %s' % store)

    def buffer_atomic(self, name, ops, mods):
        """Lower 32-bit compute MUBUF atomics with optional glc pre-atomic return."""
        if self.kind != 'cs':
            raise Unsupported(name)
        method, signed, is_sub, data_count = MUBUF_ATOMIC_OPS[name]
        data, uav, address, glc = self.raw_table_uav_store(
            'buffer atomic', ops, mods, data_count, allow_glc=True, allow_index=True, uniform_off=True)
        words = [self.u('v%d' % n) for n in range(data[1], data[2] + 1)]
        self.emit('a = %s;' % address)
        if not isinstance(getattr(self, 'used_atomic', None), set):
            self.used_atomic = set()
        if name == 'buffer_atomic_cmpswap':
            # RDNA 2 puts replacement in vdata[0] and compare in vdata[1]; HLSL takes compare first.
            if glc:
                self.used_atomic.add('uint')
                call = '%s.InterlockedCompareExchange(a, %s, %s, ao);' % (uav, words[1], words[0])
            else:
                call = '%s.InterlockedCompareStore(a, %s, %s);' % (uav, words[1], words[0])
        elif name == 'buffer_atomic_swap':
            # HLSL RWByteAddressBuffer.InterlockedExchange always requires an output operand.
            self.used_atomic.add('uint')
            call = '%s.InterlockedExchange(a, %s, ao);' % (uav, words[0])
        else:
            arg = 'asint(%s)' % words[0] if signed else ('(0u - %s)' % words[0] if is_sub else words[0])
            if glc:
                out_var = 'ai' if signed else 'ao'
                self.used_atomic.add('int' if signed else 'uint')
                call = '%s.Interlocked%s(a, %s, %s);' % (uav, method, arg, out_var)
            else:
                call = '%s.Interlocked%s(a, %s);' % (uav, method, arg)
        self.emit(call if self.exec_known else 'if (exec & 1) %s' % call)
        if glc:
            self.set_v(data[1], 'asuint(ai)' if signed else 'ao')

    def _lds_general_barrier_ok(self):
        """True when a compute stage uses dual/atomic LDS ops in straight-line code without legacy exchange ops."""
        ins = getattr(self, 'ins', None)
        if not ins:
            return False
        has_general = False
        for i, (_, _, name, _) in enumerate(ins):
            op = re.sub(r'_e(32|64)$', '', name)
            if op in LEGACY_LDS_EXCHANGE_OPS:
                return False
            if op in LDS_DUAL_ATOMIC_OPS:
                has_general = True
            if op.startswith(('s_branch', 's_cbranch', 's_setpc', 's_swappc')) or (op == 's_endpgm' and i + 1 < len(ins)):
                return False
        return has_general

    def _lds_require_alloc(self, base):
        if getattr(self, 'pipe', None) is None:
            raise Unsupported('%s needs pipeline metadata' % base)
        words = self.pipe.lds_words
        if not 0 < words <= 8192:
            raise Unsupported('%s needs a nonzero LDS allocation of at most 32 KiB' % base)
        self.uses_lds = True
        return words * 4

    def lds_dual(self, name, ops, mods):
        """Lower dual-offset LDS operations onto the dword-indexed group array."""
        if getattr(self, 'kind', None) != 'cs':
            raise Unsupported(name)
        is_write, width, _ = LDS_DUAL_OPS[name]
        dwords = width // 4
        if is_write:
            first_reg = _lds_single if dwords == 1 else _lds_pair
            if (len(ops) != 3 or _lds_single(ops[0]) is None
                    or any(first_reg(op) is None for op in ops[1:])):
                shape = ('three single VGPRs (addr, data0, data1)' if dwords == 1
                         else 'a byte address VGPR and two 2-VGPR data pairs')
                raise Unsupported('%s needs %s' % (name, shape))
        else:
            if len(ops) != 2 or _lds_vgpr_span(ops[0], 2 * dwords) is None or _lds_single(ops[1]) is None:
                shape = '2-VGPR pair' if dwords == 1 else '4-VGPR range'
                raise Unsupported('%s needs a %s and a byte address VGPR' % (name, shape))
        offsets = lds_byte_offsets(name, mods)
        lds_bytes = self._lds_require_alloc(name)
        exec_prefix = '' if self.exec_known else '(exec & 1) && '
        self.emit('a = %s;' % self.u(ops[0 if is_write else 1]))
        for slot, off in enumerate(offsets):
            # Require the entire slot in bounds. Both widths have dword alignment.
            limit = lds_bytes - off - width + 4
            guard = '%s((a & 3u) == 0u) && a < %du' % (exec_prefix, limit)
            if is_write:
                data = first_reg(ops[slot + 1])
            else:
                dst = _lds_vgpr_span(ops[0], 2 * dwords) + slot * dwords
            for index in range(dwords):
                cell = 'lds[(a + %du) >> 2]' % (off + index * 4)
                if is_write:
                    value = self.u('v%d' % (data + index))
                    if limit > 0:
                        self.emit('if (%s) %s = %s;' % (guard, cell, value))
                elif limit > 0:
                    self.emit('w = 0u;')
                    self.emit('if (%s) w = %s;' % (guard, cell))
                    self.set_v(dst + index, 'w')
                else:
                    self.set_v(dst + index, '0u')

    def lds_atomic(self, name, ops, mods):
        """Lower 32-bit unsigned/bitwise compute LDS atomics (non-rtn and _rtn forms) onto groupshared uint lds[N]."""
        if getattr(self, 'kind', None) != 'cs':
            raise Unsupported(name)
        method, has_rtn = LDS_ATOMIC_OPS[name]
        if has_rtn:
            if (len(ops) != 3 or _lds_single(ops[0]) is None
                    or _lds_single(ops[1]) is None or _lds_single(ops[2]) is None):
                raise Unsupported('%s needs three single VGPRs (dst, addr, data)' % name)
            dst, addr_op, data_op = _lds_single(ops[0]), ops[1], ops[2]
        else:
            if len(ops) != 2 or _lds_single(ops[0]) is None or _lds_single(ops[1]) is None:
                raise Unsupported('%s needs two single VGPRs (addr, data)' % name)
            dst, addr_op, data_op = None, ops[0], ops[1]
        # Reuse ds_write_b32 single-offset validation (optional 16-bit byte offset, no gds/unknown mods).
        off, = lds_byte_offsets('ds_write_b32', mods)
        if off % 4:
            raise Unsupported('%s requires 4-byte aligned immediate offset, got %d' % (name, off))
        lds_bytes = self._lds_require_alloc(name)
        addr = self.u(addr_op)
        data = self.u(data_op)
        self.emit('a = %s;' % addr)
        exec_prefix = '' if self.exec_known else '(exec & 1) && '
        if off < lds_bytes:
            if has_rtn:
                if not isinstance(getattr(self, 'used_atomic', None), set):
                    self.used_atomic = set()
                self.used_atomic.add('uint')
                self.emit('ao = 0u;')
                self.emit('if (%s((a & 3u) == 0u) && a < %du) Interlocked%s(lds[(a + %du) >> 2], %s, ao);'
                          % (exec_prefix, lds_bytes - off, method, off, data))
                self.set_v(dst, 'ao')
            else:
                self.emit('if (%s((a & 3u) == 0u) && a < %du) Interlocked%s(lds[(a + %du) >> 2], %s);'
                          % (exec_prefix, lds_bytes - off, method, off, data))
        elif has_rtn:
            self.set_v(dst, '0u')

    def op_buffer_store_dwordx2(self, ops, mods):
        self.op_buffer_store_dword(ops, mods, 2)

    def op_buffer_store_dwordx3(self, ops, mods):
        self.op_buffer_store_dword(ops, mods, 3)

    def op_buffer_store_dwordx4(self, ops, mods):
        self.op_buffer_store_dword(ops, mods, 4)

    def tbuffer(self, ops, mods, count):
        if not ops:
            raise Unsupported('buffer load with 0 operands, expected 4')
        dst = reg_range(ops[0])
        formats = [m for m in mods if m.startswith('format:')]
        fmt = formats[0][8:-1] if formats else 'BUF_FMT_8_UNORM'  # the disassembler omits the default format
        buf, address = self.buffer_address(ops, [m for m in mods if not m.startswith('format:')])
        if buf is None:
            for i in range(count):
                self.set_v(dst[1] + i, '0u')
            return
        self.emit('a = %s;' % address)
        self.load_format(fmt, buf, dst[1], count)

    def load_format(self, fmt, buf, first, count):
        if fmt not in FETCH_FORMATS:
            raise Unsupported('buffer format %s' % fmt)
        load, comps = FETCH_FORMATS[fmt]
        if load:
            self.emit('w = %s;' % load.format(buf))
        for i in range(count):
            self.set_v(first + i, comps[i].format(buf))

    def op_s_swappc_b64(self, ops, mods):
        # Only the entry call into the driver's fetch routine is translated: as the loads that routine makes.
        if self.kind != 'vs' or self.pc != 0 or ops != ['s[4:5]', 's[4:5]']:
            raise Unsupported('s_swappc_b64 %s at +0x%x' % (', '.join(ops), self.pc))
        if not self.pipe.vs_fetch:
            raise Unsupported('vertex stage calls a fetch routine and the pipeline declares no input layout')
        base = self.pipe.ud(('basevertex',))
        for e in self.pipe.vs_fetch:
            if e.get('empty'):
                for i in range(e['count']):
                    self.set_v(e['vgpr'] + i, '0u')
                continue
            key = (('vbtab',), 16 * e['slot'])
            buf = self.pipe.resource(key, 'buf')
            stride = self.pipe.ud(('stride', self.srv_index(key)))
            self.emit('a = (vid + ud%d) * ud%d + %d;' % (base, stride, e['offset']))
            self.load_format(FETCH_DXGI[e['format']][0], buf, e['vgpr'], e['count'])
        self.fetched = True

    def srv_index(self, key):
        """The read-only slot of a buffer descriptor, for the stride and word 3 the runtime supplies with it. A compute
        descriptor the program also stores through has none (it is one RW view)."""
        if key in getattr(self.pipe, 'rw_keys', ()):
            raise Unsupported('indexed or formatted load through a read-write descriptor %s' % (key,))
        return self.pipe.resources[key]['index']

    @staticmethod
    def descriptor_scalar_writes(name, ops):
        regs = set()
        for token in destinations(name, ops):
            if token in ('vcc', 'vcc_lo', 'vcc_hi'):
                regs.update((106, 107))
            parsed = reg_range(token)
            if parsed and parsed[0] == 's':
                regs.update(range(parsed[1], parsed[2] + 1))
        if name.startswith('v_cmp') and len(ops) == 2:
            regs.update((106, 107))
        return regs

    def formatted_pointer_reaches(self, pointer, before):
        """Refuse a write to the original root pointer on any predecessor path of this read."""
        if not hasattr(self, 'descriptor_predecessors'):
            by_offset = {off: i for i, (off, *_rest) in enumerate(self.ins)}
            self.descriptor_predecessors = [[] for _ in self.ins]
            self.descriptor_writes_by_instruction = []
            for index, (offset, _, name, asm) in enumerate(self.ins):
                _, args, _ = parse(asm)
                self.descriptor_writes_by_instruction.append(self.descriptor_scalar_writes(name, args))
                calculated_target = self.calculated_pc.get(offset)
                if calculated_target is not None:
                    self.descriptor_predecessors[by_offset[calculated_target]].append(index)
                elif index + 1 < len(self.ins) and name not in ('s_branch', 's_endpgm'):
                    self.descriptor_predecessors[index + 1].append(index)
                if name == 's_branch' or name.startswith('s_cbranch_'):
                    target = by_offset.get(int(args[0].split('_')[1], 16))
                    if target is not None:
                        self.descriptor_predecessors[target].append(index)
        seen = set()
        pending = list(self.descriptor_predecessors[before])
        while pending:
            earlier = pending.pop()
            if earlier in seen:
                continue
            seen.add(earlier)
            if pointer in self.descriptor_writes_by_instruction[earlier]:
                return False
            pending.extend(self.descriptor_predecessors[earlier])
        return True

    def formatted_sole_writer(self, register, before):
        """Index of the one instruction that is the last write to this SGPR on every path from the entry to
        instruction `before`, or None.

        One writer found walking back, with no path that reaches the entry without a write, means that the write is
        on every path to the read and that no path from it to the read writes the register again."""
        self.formatted_pointer_reaches(register, before)  # builds the predecessor tables
        writers, seen, expand = set(), set(), [before]
        while expand:
            at = expand.pop()
            if at == 0 or not self.descriptor_predecessors[at]:
                return None  # the entry value of the register reaches the read
            for earlier in self.descriptor_predecessors[at]:
                if earlier in seen:
                    continue
                seen.add(earlier)
                _, _, name, asm = self.ins[earlier]
                wrote = self.descriptor_writes_by_instruction[earlier] | scalar_destinations(name, asm)
                if register in wrote or 'any' in wrote:
                    writers.add(earlier)
                else:
                    expand.append(earlier)
        if len(writers) != 1:
            return None
        index, = writers
        return None if 'any' in (self.descriptor_writes_by_instruction[index] | scalar_destinations(self.ins[index][2], self.ins[index][3])) else index

    def formatted_pointer_copy(self, pointer, before):
        """(source register, instruction index) of the one s_mov_b32 that is the last write to this pointer on
        every path from the entry to this read, or None."""
        index = self.formatted_sole_writer(pointer, before)
        if index is None:
            return None
        _, _, name, asm = self.ins[index]
        _, args, _ = parse(asm)
        source = reg_range(args[1]) if name == 's_mov_b32' and len(args) == 2 else None
        if args[0] != 's%d' % pointer or not source or source[0] != 's' or source[1] != source[2]:
            return None
        return source[1], index

    def spill_word_proven(self, reg, sym):
        """True when the descriptor word held in s<reg> would pass the formatted-load provenance proof here: it is
        still in the entry block, or it is a word of one untouched local table load (or an already proven reload)."""
        if sym[0] != 'desc' or not getattr(self, 'past_entry', False):
            return True
        return self.formatted_descriptor_from_block(('s', reg, reg), sym)

    def formatted_descriptor_from_block(self, reg, first):
        """Require one untouched local load and a table pointer that no incoming path can rewrite. A word written
        by a v_readlane_b32 reload counts when every v_writelane_b32 that can supply it passed this same proof."""
        if not first:
            return False
        start, end = self.descriptor_block_first, self.descriptor_instruction_index
        reloads = {}
        for w in range(reg[1], reg[2] + 1):
            for at in range(end - 1, start - 1, -1):
                _, _, name, asm = self.ins[at]
                if w in self.descriptor_scalar_writes(name, parse(asm)[1]):
                    if re.sub(r'_e(32|64)$', '', name) == 'v_readlane_b32':
                        reloads[w] = at
                    break
        if reloads:
            return all(
                getattr(self, 'reload_proven', {}).get(reloads[w], False) if w in reloads else
                self.formatted_descriptor_from_block(('s', w, w), self.sym.get(w)) for w in range(reg[1], reg[2] + 1))
        width = reg[2] - reg[1] + 1
        block = self.ins[start:end]
        words = set(range(reg[1], reg[2] + 1))
        for index in range(len(block) - 1, -1, -1):
            _, _, name, asm = block[index]
            _, args, _ = parse(asm)
            if not (self.descriptor_scalar_writes(name, args) & words):
                continue
            m = re.fullmatch(r's_load_dwordx(4|8|16)', name)
            load = reg_range(args[0])
            if not m or not load or load[1] + first[3] != reg[1] or int(m.group(1)) < first[3] + width:
                return False
            base = reg_range(args[1])
            if not base or base[0] != 's':
                return False
            pointer = base[1]
            read = start + index
            for previous in range(index - 1, -1, -1):
                _, _, earlier, prior = block[previous]
                _, src, _ = parse(prior)
                if pointer not in self.descriptor_scalar_writes(earlier, src):
                    continue
                copied_from = reg_range(src[1]) if earlier == 's_mov_b32' and len(src) == 2 else None
                if not copied_from or copied_from[0] != 's' or copied_from[1] != copied_from[2]:
                    return False
                pointer = copied_from[1]
                read = start + previous
                break
            else:
                copy = self.formatted_pointer_copy(pointer, read)
                if copy:
                    pointer, read = copy
            return self.entry_table_pointers.get(pointer) == first[1] and self.formatted_pointer_reaches(pointer, read)
        return self.formatted_descriptor_across_blocks(reg, first)

    def formatted_descriptor_across_blocks(self, reg, first):
        """The descriptor words were loaded in an earlier block: every word has the same one writer on every path
        to this read, that writer is one s_load of table memory, and its base pointer is proven as in the block."""
        end = self.descriptor_instruction_index
        writers = {self.formatted_sole_writer(w, end) for w in range(reg[1], reg[2] + 1)}
        if len(writers) != 1 or None in writers:
            return False
        at, = writers
        _, _, name, asm = self.ins[at]
        _, args, _ = parse(asm)
        m = re.fullmatch(r's_load_dwordx(4|8|16)', name)
        load, base = reg_range(args[0]), reg_range(args[1]) if len(args) > 1 else None
        if not m or not load or load[1] + first[3] != reg[1] or int(m.group(1)) < first[3] + reg[2] - reg[1] + 1:
            return False
        if not base or base[0] != 's':
            return False
        pointer, read = base[1], at
        copy = self.formatted_pointer_copy(pointer, read)
        if copy:
            pointer, read = copy
        return self.entry_table_pointers.get(pointer) == first[1] and self.formatted_pointer_reaches(pointer, read)

    def mubuf_format(self, ops, mods, count):
        """buffer_load_format_x/xy/xyz/xyzw: FORMAT and DST_SEL come from word 3 of the descriptor, known at draw time.

        The caller supplies that word as an entry constant. Each supported format becomes one case of a switch
        on the FORMAT field that decodes the four components as the tbuffer path does; DST_SEL then picks, for
        each destination register, a component or the constant 0 or 1. The draw builds the word from a hardware
        descriptor or from the title's typed buffer view (src/startup_stubs/buffer_format_word3.h) and refuses the
        draw when it has none or the FORMAT has no case here, so the default arm is not reached from a draw.
        Compute also accepts proven heap records. A view token needs a decoded format before dispatch, while
        an unknown format in a native heap record returns zero in every destination.
        """
        buf, address = self.buffer_address(ops, mods)
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[2] - dst[1] + 1 != count or dst[2] > 255:
            raise Unsupported('formatted buffer load destination %s for %d components' % (ops[0], count))
        if buf is None:
            for i in range(count):
                self.set_v(dst[1] + i, '0u')
            return
        # FORMAT and DST_SEL come from the bound descriptor, so all four words must be intact.
        r = reg_range(ops[2])
        first = self.sym.get(r[1]) if r else None
        source = first[1][0] if first and first[0] in ('desc', 'ptr') else None
        heap = source == 'resheap' and getattr(self, 'kind', None) == 'cs'
        if source not in (None, 'table') and not heap:
            raise Unsupported('formatted buffer load requires a hardware table descriptor (source: %s)' % source)
        if (not r or r[0] != 's' or r[2] - r[1] != 3 or not first or first[0] != 'desc'
                or not names_one_value(first) or any(
                    self.sym.get(r[1] + i) != desc_word(first, first[3] + i) for i in range(4))):
            raise Unsupported('formatted buffer load through a descriptor whose words were rewritten')
        if not heap and self.past_entry and not self.formatted_descriptor_from_block(r, first):
            raise Unsupported('formatted buffer load requires descriptor words proven in the entry block or a local table load')
        word3 = 'ud%d' % self.pipe.ud(('descword3', self.srv_index(self.descriptor(ops[2]))))
        self.emit('a = %s;' % address)
        self.emit('{ uint c0 = 0, c1 = 0, c2 = 0, c3 = 0, one = U(1.0);')
        self.emit('switch ((%s >> %d) & %du) {' % (word3, DESC_FORMAT_SHIFT, DESC_FORMAT_MASK))
        for fmt, code in sorted(DESC_FORMAT_CODES.items(), key=lambda kv: kv[1]):
            load, comps = FETCH_FORMATS[fmt]
            body = ['w = %s;' % load.format(buf)] if load else []
            body += ['c%d = %s;' % (i, comps[i].format(buf)) for i in range(4)]
            if fmt in DESC_ONE_INTEGER:
                body.append('one = 1u;')
            self.emit('case %du: %s break;' % (code, ' '.join(body)))
        if heap:
            self.emit('default: one = 0u; break; }')
        else:
            self.emit('default: c0 = c1 = c2 = c3 = %du; break; }' % DESC_FORMAT_UNKNOWN)
        pick = ('((%s >> %d) & 7u)' % (word3, 3 * i) for i in range(count))
        for i, sel in enumerate(pick):
            self.set_v(dst[1] + i, '%s == 4u ? c0 : %s == 5u ? c1 : %s == 6u ? c2 : %s == 7u ? c3 : %s == 1u ? one : 0u'
                       % (sel, sel, sel, sel, sel))
        self.emit('}')

    def op_buffer_load_format_x(self, ops, mods):
        self.mubuf_format(ops, mods, 1)

    def op_buffer_load_format_xy(self, ops, mods):
        self.mubuf_format(ops, mods, 2)

    def op_buffer_load_format_xyz(self, ops, mods):
        self.mubuf_format(ops, mods, 3)

    def op_buffer_load_format_xyzw(self, ops, mods):
        self.mubuf_format(ops, mods, 4)

    def op_buffer_store_format_d16_x(self, ops, mods):
        """Lower compute MUBUF buffer_store_format_d16_x (opcode 0x84) using V# word 3, stride and num_records.

        Writes one converted 16-bit component from the low 16 bits of vdata into a raw UAV buffer:
        - BUF_FMT_32_FLOAT (22): binary16 -> binary32, 4-byte aligned Store.
        - BUF_FMT_16_FLOAT (13), BUF_FMT_16_UINT (11), BUF_FMT_16_SINT (12): low 16 bits stored at a
          2-byte aligned offset via a lock-free InterlockedCompareExchange CAS loop on the enclosing dword.
        - BUF_FMT_16_UNORM (7), BUF_FMT_16_SNORM (8): binary16 -> binary32 -> D3D half-away-from-zero
          16-bit UNORM/SNORM (NaN -> 0), stored via the same 2-byte atomic CAS loop.
        Per RDNA 2 ISA §8.1.5 Table 35, OOB_SELECT 0 (structured, stride != 0) checks index < num_records
        and (voff + imm_off) + elem_bytes <= stride while soffset adds to the base address; OOB_SELECT 3
        (raw, stride == 0) checks (soffset + voff + imm_off) + elem_bytes <= num_records. OOB_SELECT 1/2,
        mismatched stride/OOB_SELECT, nonzero DW1[31:30] swizzle flags (uav_stride carries DW1 >> 16),
        unsupported formats/flags, misaligned addresses and out-of-bounds lanes are no-ops.
        """
        name = 'buffer_store_format_d16_x'
        if self.kind != 'cs':
            raise Unsupported(name)
        raw = getattr(self, 'raw_code', b'')
        pc = getattr(self, 'pc', None)
        if raw and pc is not None and pc + 8 <= len(raw):
            w0, w1 = struct.unpack_from('<II', raw, pc)
            if (w0 & 0x00030000) or (w1 & 0x00200000):
                raise Unsupported('%s illegal MUBUF encoding bits (lds/reserved)' % name)
        if len(ops) != 4:
            raise Unsupported('%s with %d operands, expected 4' % (name, len(ops)))
        idxen = offen = False
        imm_offset = None
        for m in mods:
            if m == 'idxen' and not idxen:
                idxen = True
            elif m == 'offen' and not offen:
                offen = True
            elif m.startswith('offset:') and imm_offset is None and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                val = int(m[7:], 0)
                if val > 0xfff:
                    raise Unsupported('%s modifier %s' % (name, m))
                imm_offset = val
            else:
                raise Unsupported('%s modifier %s' % (name, m))
        imm_offset = 0 if imm_offset is None else imm_offset
        src = reg_range(ops[0])
        if not src or src[0] != 'v' or src[1] != src[2] or src[1] > 255:
            raise Unsupported('%s data %s' % (name, ops[0]))
        if idxen and offen:
            vaddr = reg_range(ops[1])
            if not vaddr or vaddr[0] != 'v' or vaddr[2] - vaddr[1] != 1 or vaddr[2] > 255:
                raise Unsupported('%s vaddr %s for idxen offen' % (name, ops[1]))
            idx_expr, voff_expr = self.u('v%d' % vaddr[1]), self.u('v%d' % vaddr[2])
        elif idxen:
            vaddr = reg_range(ops[1])
            if not vaddr or vaddr[0] != 'v' or vaddr[1] != vaddr[2] or vaddr[1] > 255:
                raise Unsupported('%s vaddr %s for idxen' % (name, ops[1]))
            idx_expr, voff_expr = self.u(ops[1]), '0u'
        elif offen:
            vaddr = reg_range(ops[1])
            if not vaddr or vaddr[0] != 'v' or vaddr[1] != vaddr[2] or vaddr[1] > 255:
                raise Unsupported('%s vaddr %s for offen' % (name, ops[1]))
            idx_expr, voff_expr = '0u', self.u(ops[1])
        else:
            if ops[1] != 'off':
                raise Unsupported('%s vaddr %s without idxen or offen' % (name, ops[1]))
            idx_expr, voff_expr = '0u', '0u'
        if re.fullmatch(r'0x[0-9a-f]+|\d+', ops[3]):
            soff_imm = int(ops[3], 0)
            if soff_imm > 0xffffffff:
                raise Unsupported('%s soffset %s' % (name, ops[3]))
            soff_expr = '%du' % soff_imm
        else:
            if self.is_lane_mask(ops[3]):
                raise Unsupported('%s soffset %s is a lane mask' % (name, ops[3]))
            sreg = reg_range(ops[3])
            if not sreg or sreg[0] != 's' or sreg[1] != sreg[2] or sreg[1] > 105:
                raise Unsupported('%s soffset %s' % (name, ops[3]))
            soff_expr = self.S(sreg[1])
        rsrc = reg_range(ops[2])
        if not rsrc or rsrc[0] != 's' or rsrc[2] - rsrc[1] != 3 or rsrc[2] > 105:
            raise Unsupported('%s descriptor %s' % (name, ops[2]))
        if self.null_descriptor(ops[2]):
            self.descriptor(ops[2])
            return
        key = self.store_descriptor(ops[2], 4, name, stride0_ok=False)
        first = self.sym.get(rsrc[1])
        if getattr(self, 'past_entry', False) and not self.formatted_descriptor_from_block(rsrc, first):
            raise Unsupported('%s requires descriptor words proven in the entry block or a local table load' % name)
        uav = self.pipe.uav(key, 'raw')
        uav_index = self.pipe.uavs[key]['index']
        word3 = 'ud%d' % self.pipe.ud(('uav_descword3', uav_index))
        stride = 'ud%d' % self.pipe.ud(('uav_stride', uav_index))
        num_records = 'ud%d' % self.pipe.ud(('uav_numrecords', uav_index))
        val_expr = self.u(ops[0], low16_only=True)
        self.pipe.uses_store_d16_x = True
        call = 'store_format_d16_x(%s, %s, %s, %s, %s, %s, %s, %du, %s);' % (
            uav, word3, stride, num_records, idx_expr, voff_expr, soff_expr, imm_offset, val_expr)
        self.emit(call if self.exec_known else 'if (exec & 1) ' + call)

    def op_buffer_store_format_xyzw(self, ops, mods):
        """Lower compute MUBUF buffer_store_format_xyzw (opcode 0x07) in the idxen form with an inline soffset.

        FORMAT, stride and num_records are read at dispatch from the same entry constants as
        buffer_store_format_d16_x, with its bounds and OOB_SELECT rules. Only 4-channel formats write
        (STORE_FORMAT_XYZW_FORMATS); any other descriptor makes the store a no-op.
        """
        name = 'buffer_store_format_xyzw'
        if self.kind != 'cs':
            raise Unsupported(name)  # the unknown-op refusal
        raw = getattr(self, 'raw_code', b'')
        pc = getattr(self, 'pc', None)
        if raw and pc is not None and pc + 8 <= len(raw):
            w0, w1 = struct.unpack_from('<II', raw, pc)
            if (w0 & 0x00030000) or (w1 & 0x00200000):
                raise Unsupported('%s illegal MUBUF encoding bits (lds/reserved)' % name)
        if len(ops) != 4:
            raise Unsupported('%s with %d operands, expected 4' % (name, len(ops)))
        idxen = False
        imm_offset = None
        for m in mods:
            if m == 'idxen' and not idxen:
                idxen = True
            elif m == 'offen':
                raise Unsupported('%s offen address form (only idxen is modelled)' % name)
            elif m.startswith('offset:') and imm_offset is None and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                imm_offset = int(m[7:], 0)
                if imm_offset > 0xfff:
                    raise Unsupported('%s modifier %s' % (name, m))
            else:
                raise Unsupported('%s modifier %s' % (name, m))
        if not idxen:
            raise Unsupported('%s without idxen (only the idxen address form is modelled)' % name)
        src = reg_range(ops[0])
        if not src or src[0] != 'v' or src[2] - src[1] != 3 or src[2] > 255:
            raise Unsupported('%s data %s' % (name, ops[0]))
        vaddr = reg_range(ops[1])
        if not vaddr or vaddr[0] != 'v' or vaddr[1] != vaddr[2] or vaddr[1] > 255:
            raise Unsupported('%s vaddr %s for idxen' % (name, ops[1]))
        if not re.fullmatch(r'0x[0-9a-f]+|\d+', ops[3]):
            raise Unsupported('%s soffset %s (only a non-negative inline constant is modelled)' % (name, ops[3]))
        rsrc = reg_range(ops[2])
        if not rsrc or rsrc[0] != 's' or rsrc[2] - rsrc[1] != 3 or rsrc[2] > 105:
            raise Unsupported('%s descriptor %s' % (name, ops[2]))
        if self.null_descriptor(ops[2]):
            self.descriptor(ops[2])
            return
        key = self.store_descriptor(ops[2], 4, name, stride0_ok=False)
        first = self.sym.get(rsrc[1])
        if getattr(self, 'past_entry', False) and not self.formatted_descriptor_from_block(rsrc, first):
            raise Unsupported('%s requires descriptor words proven in the entry block or a local table load' % name)
        idx = self.u(ops[1])
        data = 'uint4(%s)' % ', '.join(self.u('v%d' % n) for n in range(src[1], src[2] + 1))
        uav = self.pipe.uav(key, 'raw')
        uav_index = self.pipe.uavs[key]['index']
        word3 = 'ud%d' % self.pipe.ud(('uav_descword3', uav_index))
        stride = 'ud%d' % self.pipe.ud(('uav_stride', uav_index))
        num_records = 'ud%d' % self.pipe.ud(('uav_numrecords', uav_index))
        self.pipe.uses_store_format_xyzw = self.pipe.uses_pack16_rne = True
        call = 'store_format_xyzw(%s, %s, %s, %s, %s, 0u, %du, %du, %s);' % (
            uav, word3, stride, num_records, idx, int(ops[3], 0), imm_offset or 0, data)
        self.emit(call if self.exec_known else 'if (exec & 1) ' + call)

    def op_tbuffer_load_format_x(self, ops, mods):
        self.tbuffer(ops, mods, 1)

    def op_tbuffer_load_format_xy(self, ops, mods):
        self.tbuffer(ops, mods, 2)

    def op_tbuffer_load_format_xyz(self, ops, mods):
        self.tbuffer(ops, mods, 3)

    def op_tbuffer_load_format_xyzw(self, ops, mods):
        self.tbuffer(ops, mods, 4)

    def image(self, ops, mods, how):
        if getattr(self, 'kind', None) == 'cs' and 'd16' in mods:
            raise Unsupported('compute d16 image load requires validated SRV formats')
        # Every image form reads vdata, an address and a descriptor, so a shorter operand list would reach
        # ops[1] or ops[2] as an index error rather than a refusal. Samplers are checked further down, where
        # the form decides whether one is required.
        if len(ops) < 3:
            raise Unsupported('image instruction with %d operands, expected at least 3' % len(ops))
        dst = reg_range(ops[0])
        dims = [m for m in mods if m.startswith('dim:')]
        dmasks = [m for m in mods if m.startswith('dmask:')]
        if not dims or not dmasks:
            raise Unsupported('image instruction without %s' % ('dim:' if not dims else 'dmask:'))
        dim = dims[0][4:]
        dmask = int(dmasks[0][6:], 0)
        kinds = {'SQ_RSRC_IMG_1D': 'tex1d', 'SQ_RSRC_IMG_2D': 'tex2d', 'SQ_RSRC_IMG_3D': 'tex3d', 'SQ_RSRC_IMG_2D_ARRAY': 'tex2da'}
        # Cube maps: only the compute level-zero and explicit-level samples (the runtime binds a TextureCube view for them).
        cube = dim == 'SQ_RSRC_IMG_CUBE' and how in ('lz', 'l') and getattr(self, 'kind', None) == 'cs'
        if cube:
            kinds[dim] = 'texcube'
        if dim not in kinds:
            raise Unsupported('image dimension %s' % dim)
        # A 2D array takes the slice as its third address component, where a volume takes z. Sample and
        # Load on a Texture2DArray take it in the same place, so only the declared type differs.
        one = dim == 'SQ_RSRC_IMG_1D'
        three = dim in ('SQ_RSRC_IMG_3D', 'SQ_RSRC_IMG_2D_ARRAY') or cube
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)]
        integer_srvs = getattr(getattr(self, 'pipe', None), 'integer_texture_srvs', False)
        # The legacy 1D path models only sampling. Typed SRVs also use the generic load
        # path below, whose Texture1D address has one coordinate and a mip index.
        if one and not (integer_srvs and how in ('load', 'load_mip')):
            expected = {'sample': 1, 'lz': 1, 'l': 2}.get(how)
            if expected is None or len(cs) != expected or not dmask:
                raise Unsupported('unverified 1D image %s address or modifiers' % how)
            for m in mods:
                if not (m.startswith(('dim:', 'dmask:')) or (how != 'sample' and m in ('dlc', 'glc', 'slc'))):
                    raise Unsupported('1D image %s modifier %s' % (how, m))
        sample_d16 = not one and how == 'lz' and 'd16' in mods
        lod = bias = None
        grads = None
        if how == 'd':
            # Address dwords are ddx then ddy (one per coordinate axis each) and then the coordinates, so a 2D
            # sample is six registers, a 2D array seven and a volume nine. Offsets, bias, clamp, a16 and the
            # other layout-changing modifiers are refused; only cache-policy bits are accepted.
            for m in mods:
                if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc')):
                    raise Unsupported('image_sample_d modifier %s' % m)
            if dmask == 0:
                raise Unsupported('image_sample_d with an empty dmask')
            if one:
                raise Unsupported('image_sample_d on a 1D image')
            axes = 3 if dim == 'SQ_RSRC_IMG_3D' else 2
            want = 3 * axes + (1 if dim == 'SQ_RSRC_IMG_2D_ARRAY' else 0)
            if want == 9 and len(cs) == 16 and not ops[1].startswith('['):
                # A contiguous nine-dword volume address is encoded as a sixteen-register range; the rest is padding.
                cs = cs[:9]
            if len(cs) != want:
                raise Unsupported('image_sample_d with %d address registers' % len(cs))
            grads = (cs[:axes], cs[axes:2 * axes])
            cs = cs[2 * axes:]
        if how in ('l', 'b'):
            # image_sample_l: the address dwords are the coordinates and then the explicit level. image_sample_b:
            # the RDNA 2 ISA (section 8.2.5, "{ offset } { bias } { z-compare } { derivative } { body }") puts one
            # float bias dword BEFORE the coordinates. Offsets, clamp and derivatives would add dwords or modifiers
            # this lowering does not read, so they are refused. Only the cache-policy
            # bits (glc, slc, dlc), which never change the sampled value, are accepted next to dim and dmask;
            # unorm, r128, tfe, lwe, a16, d16 and reserved bits would each change the address, resource or result
            # layout and are refused by name.
            name = 'image_sample_' + how
            for m in mods:
                if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc')):
                    raise Unsupported('%s modifier %s' % (name, m))
            if dmask == 0:
                raise Unsupported('%s with an empty dmask' % name)
            expected = 4 if three else (2 if one else 3)
            if len(cs) != expected:
                raise Unsupported('%s with %d address registers' % (name, len(cs)))
            if how == 'l':
                lod, cs = cs[-1], cs[:-1]
            else:
                # The bias is added to the LOD derived from the coordinate derivatives of the pixel quad: only a
                # pixel stage that still runs with its entry mask and wave-uniform control flow has them.
                # Elsewhere the plain sample is lowered to level 0, which would drop the bias, so the bias form is
                # refused instead. A uniform join (scc with scc_uniform) or later loop does not drop the proof
                # (M200-01 B/L). Divergent EXEC or a vcc/exec-conditioned branch before the sample stays refused.
                if (self.kind != 'ps' or not self.derivative_exec_ok() or
                        not getattr(self, 'uniform_cf', True)):
                    raise Unsupported('image_sample_b needs implicit derivatives: pixel stage with the entry EXEC '
                                     'mask and wave-uniform control flow')
                if hasattr(self, 'derivative_sites'):
                    self.derivative_sites.append((self.pc, 'image_sample_b'))
                if not self.exec_known and not self.uniform_bias(cs[0]):
                    # Past a divergent EXEC the op is only drawn with one bias for the quad (tests/startup_image_sample_b_
                    # flow_check.py); a bias that differs per lane (a join of different values) stays refused by name.
                    raise Unsupported('image_sample_b with a per-lane bias past a divergent EXEC')
                bias, cs = cs[0], cs[1:]
        if how == 'get_lod':
            # image_get_lod returns [0] the LOD a sample would use, clamped, and [1] the computed LOD before the clamp
            # (Mesa nir_texop_lod, which ACO emits as image_get_lod; GLSL textureQueryLod x/y). Only the 2D form with
            # two address registers and dmask 0x1-0x3 is lowered; the cache-policy bits are the only other modifiers.
            for m in mods:
                if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc')):
                    raise Unsupported('image_get_lod modifier %s' % m)
            if self.kind != 'ps' or dim != 'SQ_RSRC_IMG_2D' or not dmask or dmask & ~3 or len(cs) != 2:
                raise Unsupported('image_get_lod form %s %s' % (', '.join(ops), ' '.join(mods)))
            # Both HLSL calls take implicit derivatives, so the rule of image_sample_b applies.
            if not self.derivative_exec_ok() or not getattr(self, 'uniform_cf', True):
                raise Unsupported('image_get_lod needs implicit derivatives: pixel stage with the entry EXEC '
                                 'mask and wave-uniform control flow')
            if hasattr(self, 'derivative_sites'):
                self.derivative_sites.append((self.pc, 'image_get_lod'))
        desc_key = self.descriptor(ops[2])
        tex = self.pipe.resource(desc_key, kinds[dim]) if desc_key is not None else None
        if tex is not None and hasattr(self.pipe, 'note_use'):  # the manifest hint `srvuse`; unit-test stubs have none
            self.pipe.note_use(desc_key, 'texel' if how in ('load', 'load_mip') else 'lod' if how == 'get_lod' else 'sample')
        if how in ('load', 'load_mip'):
            # image_load_mip addresses x, y, [slice or z] and then the mip, so a 2D array is four registers
            # and Texture2DArray.Load takes int4(x, y, slice, mip) like a volume's int4(x, y, z, mip).
            axes = 1 if one else 3 if three else 2
            packed_2d = how == 'load' and dim == 'SQ_RSRC_IMG_2D' and 'a16' in mods
            packed_d16 = how in ('load', 'load_mip') and 'd16' in mods
            allowed = ('unorm', 'glc', 'slc', 'dlc')
            if packed_2d:
                allowed += ('a16',)
            if packed_d16:
                allowed += ('d16',)
            unsupported = any(not m.startswith(('dim:', 'dmask:')) and m not in allowed for m in mods)
            expected_addr = 1 if packed_2d else (axes + 1 if how == 'load_mip' else axes)
            wrong_address = len(cs) != expected_addr
            if unsupported or wrong_address or not dmask:
                raise Unsupported('image_%s address or modifiers: %s' % (how, ' '.join(mods)))
            if packed_d16:
                op_name = 'image_' + how
                expected_dst = (bin(dmask).count('1') + 1) // 2
                if not dst or dst[0] != 'v' or dst[2] - dst[1] + 1 != expected_dst or dst[2] > 255:
                    raise Unsupported('%s d16 destination %s for dmask %#x' % (op_name, ops[0], dmask))
                res = self.pipe.resources[desc_key]
                if res['index'] in self.pipe.srv_formats:
                    fmt = self.pipe.srv_formats[res['index']]
                    if fmt not in D16_EXACT_SRV_FORMATS:
                        raise Unsupported('%s d16 unsupported SRV format %d' % (op_name, fmt))
            if packed_2d:
                packed = self.u(cs[0])
                address = ['%s & 0xffffu' % packed, '%s >> 16' % packed]
            else:
                address = [self.u(c) for c in cs[:axes]]
            mip = self.u(cs[axes]) if how == 'load_mip' else '0'
            location = 'int%d(%s)' % (axes + 1, ', '.join(address + [mip]))
            call = '%s.Load(%s)' % (tex, location)
            if integer_srvs and not packed_d16:
                res = self.pipe.resources[desc_key]
                res['integer_load'] = True
                call = ('asfloat(%(tex)s_is_integer == 2u ? asuint(%(tex)s_sint.Load(%(loc)s)) : '
                        '%(tex)s_is_integer != 0u ? %(tex)s_uint.Load(%(loc)s) : asuint(%(float)s))') % {
                            'tex': tex, 'loc': location, 'float': call}
        else:
            packed_2d = False
            if not one and how in ('sample', 'lz'):
                op_name = 'image_sample' if how == 'sample' else 'image_sample_lz'
                packed_2d = dim == 'SQ_RSRC_IMG_2D' and 'a16' in mods
                allowed = ('dlc', 'glc', 'slc') + (('a16',) if packed_2d else ())
                if sample_d16:
                    allowed += ('d16',)
                for m in mods:
                    if not (m.startswith(('dim:', 'dmask:')) or m in allowed):
                        raise Unsupported('%s modifier %s' % (op_name, m))
                if not dmask:
                    raise Unsupported('%s with an empty dmask' % op_name)
                expected = 1 if packed_2d else (3 if three else 2)
                if len(cs) != expected:
                    raise Unsupported('%s with %d address registers' % (op_name, len(cs)))
            if sample_d16:
                expected_dst = (bin(dmask).count('1') + 1) // 2
                if not dst or dst[0] != 'v' or dst[2] - dst[1] + 1 != expected_dst or dst[2] > 255:
                    raise Unsupported('image_sample_lz d16 destination %s for dmask %#x' % (ops[0], dmask))
            if tex is None:
                self.null_image_read(dst, dmask)
                return
            heap = self.heap_sampler(ops[3])
            words = None if heap else self.const_words(ops[3])
            name = 'image_sample' + ('' if how == 'sample' else '_' + how)
            implicit_bias = False
            if how == 'b':
                # The guest clamps the biased LOD to the sampler's MIN_LOD and MAX_LOD. Only a constant descriptor
                # can be read here, and only its limits are carried to the static sampler (line smpl), so a sampler
                # from the heap or from a register is refused: its limits and bias cannot be shown to be irrelevant.
                words = self.sample_b_sampler(heap, words)
                smp = self.pipe.sampler(tuple(words), lod_limits=True)
            elif how == 'get_lod':
                # The clamped result depends on the sampler's MIN_LOD, MAX_LOD and bias, so the sampler must be a
                # constant descriptor under the same validation as image_sample_b, with its limits carried.
                if heap or not words or self.default_sampler_words(words):
                    raise Unsupported('image_get_lod with a sampler that is not constant (LOD limits and bias unknown)')
                smp = self.pipe.sampler(tuple(self.lod_sampler_words('image_get_lod', words)), lod_limits=True)
            elif how in ('sample', 'l', 'lz', 'd') and (heap or (words and self.default_sampler_words(words))):
                # A heap sampler, or the fixed default that stands in for one, keeps its unbounded static LOD range: the
                # descriptor is read at run time, so its MIN_LOD, MAX_LOD and bias are not validated here (#2221).
                self.pipe.lod_unvalidated.add(name)
                smp = heap or self.pipe.sampler(tuple(words))
            elif how in ('sample', 'l', 'lz', 'd') and words:
                # The smpl binding carries the validated LOD state. An unbiased descriptor with mip filter 0
                # (base level only) keeps the ordinary sampler.
                allow_bias = how in ('sample', 'lz') and self.kind == 'ps' and dim == 'SQ_RSRC_IMG_2D'
                words = self.lod_sampler_words(name, words, mip_filter_none_ok=True, allow_bias=allow_bias)
                implicit_bias = how == 'sample' and bool(words[2] & 0x3fff)
                smp = self.pipe.sampler(tuple(words), lod_limits=bool((words[2] >> 26) & 3))
            else:
                smp = heap or self.pipe.sampler(tuple(words) if words else self.descriptor(ops[3], allow_null=False))
            if packed_2d:
                packed = self.u(cs[0])
                uv = 'float2(f16tof32(%s), f16tof32(%s >> 16))' % (packed, packed)
            elif one:
                uv = self.f(cs[0])
            elif cube:
                # The address is S, T and the face index (RDNA 2 ISA, image_sample on a cube map; S and T come from
                # v_cubesc / v_cubetc divided by |v_cubema| and offset by 1.5, v_cubeid is the face). The sample is
                # taken from the TextureCube view at the direction those registers stand for (x2w_cube_dir).
                if len(cs) != 3:
                    raise Unsupported('image_sample on a cube map with %d address registers' % len(cs))
                self.pipe.uses_cube_dir = True
                uv = 'x2w_cube_dir(%s, %s, %s)' % (self.f(cs[0]), self.f(cs[1]), self.f(cs[2]))
            elif three:
                uv = 'float3(%s, %s, %s)' % (self.f(cs[0]), self.f(cs[1]), self.f(cs[2]))
            else:
                uv = 'float2(%s, %s)' % (self.f(cs[0]), self.f(cs[1]))
            if how == 'sample' and self.kind == 'cs':
                # A compute lane has no pixel quad to supply derivatives; level 0 would be a guess.
                raise Unsupported('image_sample with implicit LOD in a compute program')
            if how == 'd':
                axes = len(grads[0])
                dx, dy = ['float%d(%s)' % (axes, ', '.join(self.f(r) for r in g)) for g in grads]
                call = '%s.SampleGrad(%s, %s, %s, %s)' % (tex, smp, uv, dx, dy)
            elif how == 'l':
                call = '%s.SampleLevel(%s, %s, %s)' % (tex, smp, uv, self.f(lod))
            elif how == 'b':
                # SampleBias adds the bias to the implicit LOD, like the ISA's _B suffix ("Add this BIAS to the LOD
                # computed"). The sampler's LOD limits are carried by the smpl line; its LOD bias is refused in
                # sample_b_sampler.
                call = '%s.SampleBias(%s, %s, %s)' % (tex, smp, uv, self.f(bias))
            elif how == 'get_lod':
                # HLSL CalculateLevelOfDetail (clamped) and CalculateLevelOfDetailUnclamped (SM 4.1+). Which clamps
                # each side applies besides the sampler's (the view's mip range) is assumed to agree.
                call = 'float4(%s.CalculateLevelOfDetail(%s, %s), %s.CalculateLevelOfDetailUnclamped(%s, %s), 0.0, 0.0)' % (
                    tex, smp, uv, tex, smp, uv)
            elif how == 'lz' or not self.exec_known or self.kind != 'ps':
                if implicit_bias:
                    raise Unsupported('image_sample with a sampler LOD bias and no implicit LOD')
                call = '%s.SampleLevel(%s, %s, 0)' % (tex, smp, uv)
            elif len(self.blocks) > 0:
                # Past a join the guest still takes the LOD from the quad's derivatives. Where the rule of image_sample_b
                # holds (entry EXEC, wave-uniform control flow, no divergent back edge over the call) the call is the
                # implicit-LOD Sample; otherwise it stays level 0 (resolve_plain_samples).
                call = '%s.SampleLevel(%s, %s, 0)' % (tex, smp, uv)
                if getattr(self, 'uniform_cf', True) and hasattr(self, 'plain_samples'):
                    self.plain_samples[self.pc] = (None if implicit_bias else call, '%s.Sample(%s, %s)' % (tex, smp, uv))
                    call = 'X2W_PLAIN_SAMPLE<%x>' % self.pc
                elif implicit_bias:
                    raise Unsupported('image_sample with a sampler LOD bias and no implicit LOD')
            else:
                call = '%s.Sample(%s, %s)' % (tex, smp, uv)
        if tex is None:
            self.null_image_read(dst, dmask, packed='d16' in mods)
            return
        self.emit('t = %s;' % call)
        if sample_d16 or (how in ('load', 'load_mip') and 'd16' in mods):
            self.pipe.uses_pack16_rne = True
            if sample_d16:
                # Narrow the filtered float result. The exact UNORM16 load path reconstructs an
                # individual texel and would quantize filtered samples before converting them.
                pack = 'pack16_rne(t.%s)'
            else:
                res = self.pipe.resources[desc_key]
                res['d16_exact'] = True
                mode_arg = 'ud%d' % self.pipe.ud(('d16unorm16', res['index']))
                pack = 'pack16_d16(t.%s, ' + mode_arg + ')'
            halves = [pack % comp for bit, comp in enumerate('xyzw') if dmask & (1 << bit)]
            for i in range(0, len(halves), 2):
                odd_tail = i + 1 == len(halves)
                expr = halves[i] if odd_tail else '%s | (%s << 16)' % (halves[i], halves[i + 1])
                self.set_v(dst[1] + i // 2, expr, low16_only=odd_tail)
        else:
            n = dst[1]
            for bit, comp in enumerate('xyzw'):
                if dmask & (1 << bit):
                    self.set_v(n, 'U(t.%s)' % comp)
                    n += 1

    def null_image_read(self, dst, dmask, packed=False):
        if not 0 < dmask <= 15:
            raise Unsupported('invalid null image dmask')
        count = dmask.bit_count()
        if packed:
            count = (count + 1) // 2
        if not dst or dst[0] != 'v' or dst[2] - dst[1] + 1 != count or dst[2] > 255:
            raise Unsupported('null image destination does not match dmask')
        for i in range(count):
            self.set_v(dst[1] + i, '0u', low16_only=packed and i == count - 1 and bool(dmask.bit_count() % 2))

    def image_compare(self, ops, mods, how):
        """image_sample_c and image_sample_c_lz: a 2D or 2D-array depth comparison returning one float.

        The address dwords are the reference, then u and v (and the slice of a 2D array), which is what
        SampleCmp* takes as its third argument and its location. The comparison function and the filter live in the sampler descriptor,
        so they are read from its constant words by the decoder the pixel AOT path already checks against
        D3D12 (pixel_aot.comparison_sampler_state) and bound as a SamplerComparisonState. HLSL has no
        SampleCmpLevel before shader model 6.7 (the stages here compile as 5.1 and 6.0), so
        image_sample_c_l is refused instead of being sampled at level zero.
        """
        base = 'image_sample_' + how
        if how == 'c_l':
            raise Unsupported('image_sample_c_l: SampleCmpLevel needs shader model 6.7; only c and c_lz are lowered')
        for m in mods:
            if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc')):
                raise Unsupported('%s modifier %s' % (base, m))
        dims = [m[4:] for m in mods if m.startswith('dim:')]
        masks = [m[6:] for m in mods if m.startswith('dmask:')]
        if len(dims) != 1 or len(masks) != 1:
            raise Unsupported('%s without exactly one dim and dmask' % base)
        dim, dmask = dims[0], int(masks[0], 0)
        if dim not in ('SQ_RSRC_IMG_2D', 'SQ_RSRC_IMG_2D_ARRAY'):
            raise Unsupported('%s on %s' % (base, dim))
        array = dim == 'SQ_RSRC_IMG_2D_ARRAY'
        if dmask != 1:
            raise Unsupported('%s with dmask %#x: a comparison returns one channel' % (base, dmask))
        if len(ops) != 4:
            raise Unsupported('%s with %d operands' % (base, len(ops)))
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)]
        # The reference, u and v; a 2D array adds the slice as the last address register.
        if len(cs) != (4 if array else 3):
            raise Unsupported('%s with %d address registers' % (base, len(cs)))
        if self.heap_sampler(ops[3]):
            raise Unsupported('%s through a sampler heap' % base)
        words = self.const_words(ops[3])
        if words is None:
            raise Unsupported('%s with a sampler that is not constant' % base)
        import pixel_aot
        try:
            pixel_aot.comparison_sampler_state(tuple(words), level_zero=how == 'c_lz', allow_border=True)
        except ValueError as error:
            raise Unsupported('%s sampler: %s' % (base, error))
        key = self.descriptor(ops[2])
        if key is None:
            return self.null_image_read(reg_range(ops[0]), dmask)
        tex = self.pipe.resource(key, 'tex2da' if array else 'tex2d')
        smp = self.pipe.sampler(tuple(words), comparison=True)
        # Texture2DArray takes the slice as the third location component, like Sample and Load on the same type.
        location = ('float3(%s, %s, %s)' % (self.f(cs[1]), self.f(cs[2]), self.f(cs[3])) if array
                    else 'float2(%s, %s)' % (self.f(cs[1]), self.f(cs[2])))
        args = '%s, %s, %s' % (smp, location, self.f(cs[0]))
        # MAX_LOD (word 1 bits 23:12) is not carried into the static sampler, and the comparison-state mapping ignores
        # the mip filter and uses level zero. Only a source limited to level zero is therefore exact for the implicit
        # form; any other limit would need mip selection and blending that is not lowered in this slice, so it is refused.
        max_lod = (words[1] >> 12) & 0xfff
        if how == 'c' and max_lod:
            raise Unsupported('%s with implicit mip selection (sampler MAX_LOD %#x)' % (base, max_lod))
        call = '%s.SampleCmpLevelZero(%s)' % (tex, args)
        self.set_v(reg_range(ops[0])[1], 'U(%s)' % call)

    def op_image_sample_c(self, ops, mods):
        self.image_compare(ops, mods, 'c')

    def op_image_sample_c_lz(self, ops, mods):
        self.image_compare(ops, mods, 'c_lz')

    def op_image_sample_c_l(self, ops, mods):
        self.image_compare(ops, mods, 'c_l')

    def op_image_sample(self, ops, mods):
        self.image(ops, mods, 'sample')

    def op_image_sample_lz(self, ops, mods):
        self.image(ops, mods, 'lz')

    def op_image_sample_l(self, ops, mods):
        self.image(ops, mods, 'l')

    def sample_b_sampler(self, heap, words):
        """The constant sampler words of an image_sample_b, or Unsupported when the guest sampler state is not kept.

        SQ_IMG_SAMP_WORD1 bits 11:0 MIN_LOD and 23:12 MAX_LOD are u4.8 and are carried to MinLOD and MaxLOD of the
        static sampler (translated_sampler_bindings.h, line smpl). This form still refuses descriptor bias
        and a mip filter of 0 (RDNA 2 ISA: disable mipmapping and use the base level)."""
        if heap:
            raise Unsupported('image_sample_b through a sampler heap (LOD limits and bias unknown)')
        if not words:
            raise Unsupported('image_sample_b with a sampler that is not constant')
        if self.default_sampler_words(words):
            raise Unsupported('image_sample_b through a sampler heap (replaced by a default sampler)')
        return self.lod_sampler_words('image_sample_b', words)

    def default_sampler_words(self, words):
        """True for the four zero words that the default-sampler shortcut of a run-time heap load leaves behind."""
        return list(words) == [0, 0, 0, 0] and 'dynamic sampler index replaced by a default sampler' in self.pipe.notes

    def lod_sampler_words(self, name, words, mip_filter_none_ok=False, allow_bias=False):
        """Validate LOD state for a constant sampler and return the words to bind."""
        mip_filter = (words[2] >> 26) & 3
        primary = words[2] & 0x3fff
        secondary = words[2] & 0xfc000
        if secondary or (primary and (not allow_bias or mip_filter != 2)):
            raise Unsupported('%s with a sampler LOD bias (word 2 %#x)' % (name, words[2]))
        # AMD RDNA 2 Shader Instruction Set Architecture, sections 8.2.6 and 8.2.7:
        # T#.perf_modulation scales LOD_BIAS_SEC, which must be zero here. The
        # primary bias is a signed 14-bit field with eight fractional bits
        # (also AMD CIK 3D Registers, SQ_IMG_SAMP_WORD2). It maps exactly to
        # MipLODBias within D3D12_MIP_LOD_BIAS_MIN/MAX.
        bias = ((primary ^ 0x2000) - 0x2000) / 256
        if not -16.0 <= bias <= 15.99:
            raise Unsupported('%s with sampler LOD bias %g outside the D3D12 range' % (name, bias))
        if mip_filter == 0 and not mip_filter_none_ok:
            raise Unsupported('%s with mipmapping disabled (sampler mip filter 0)' % name)
        if words[1] & 0xfff > (words[1] >> 12) & 0xfff:
            raise Unsupported('%s with sampler MIN_LOD %#x above MAX_LOD %#x'
                              % (name, words[1] & 0xfff, (words[1] >> 12) & 0xfff))
        if primary and name == 'image_sample_lz':
            # RDNA 2 ISA §8.2.3, Sample Instruction Suffix Key: _LZ forces mip
            # level 0. D3D sample_l honors MIPLODBIAS, so bind the zero-bias
            # words here, separately from any implicit sample with the bias.
            return (*words[:2], words[2] & ~0x3fff, words[3])
        return words

    def op_image_sample_b(self, ops, mods):
        self.image(ops, mods, 'b')

    def op_image_sample_d(self, ops, mods):
        self.image(ops, mods, 'd')

    def op_image_sample_c_d(self, ops, mods):
        raise Unsupported('image_sample_c_d: SampleCmpGrad needs shader model 6.8; only c and c_lz are lowered')

    def op_image_get_lod(self, ops, mods):
        self.image(ops, mods, 'get_lod')

    def op_image_load(self, ops, mods):
        self.image(ops, mods, 'load')

    def op_image_load_mip(self, ops, mods):
        self.image(ops, mods, 'load_mip')

    def fixed_table_image(self, operand, name):
        """Return the table location if the manifest can bind the original eight-word image view."""
        image = reg_range(operand)
        first = self.sym.get(image[1]) if image and image[0] == 's' and image[2] - image[1] == 7 else None
        if not first or first[0] != 'desc' or first[1][0] != 'table' or first[2] is None or first[3] % 8:
            raise Unsupported('%s requires a fixed table image descriptor' % name)
        for i in range(8):
            word = desc_word(first, first[3] + i)
            # A write through a second load of this descriptor can leave these SGPR symbols intact.
            if self.sym.get(image[1] + i) != word or desc_key(word) in self.pipe.modified:
                raise Unsupported('%s through a modified image descriptor' % name)
        return self.descriptor(operand)

    def op_image_store(self, ops, mods):
        """Write the selected channels of a typed image texel.

        MIMG opcode 0x08 (LLVM AMDGPU MIMGInstructions.td, IMAGE_STORE via MIMG_Store): VDATA is the
        source, VADDR the integer texel address (x, y for dim 2D, x, y, layer for dim 2D-array) and SRSRC
        the image descriptor; the channels selected by DMASK are written in order from consecutive VDATA
        registers. A partial mask uses a typed UAV load to preserve the other channels. This assumes that
        each active lane owns its texel, including across groups, for the duration of the read-modify-write.
        The bound float or normalized view must support typed UAV stores and, for partial masks, loads.
        The store runs under the lane's exec bit, like the other per-lane writes, and D3D drops UAV writes
        from helper lanes as the hardware drops them for lanes outside exec.
        """
        name = 'image_store'
        if self.kind not in ('ps', 'cs'):
            raise Unsupported('%s outside a pixel stage' % name)
        d16 = self.kind == 'cs' and mods.count('d16') == 1
        if len(ops) != 3:
            raise Unsupported('%s with %d operands' % (name, len(ops)))
        for m in mods:
            if d16 and m == 'd16':
                continue
            # IMAGE_STORE always uses integer texel coordinates, so unorm is a no-op like on image_load.
            if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc', 'unorm')
                    or (m == 'a16' and mods.count('a16') == 1 and self.kind == 'ps')):
                raise Unsupported('%s modifier %s' % (name, m))
        dims = [m[4:] for m in mods if m.startswith('dim:')]
        # a16 packs 16-bit texel coordinates two to a VGPR (x in the low half, y in the high half). Only the 2D form
        # is lowered, like image_load; other dimensions would also pack the layer or z into a second register.
        packed_2d = 'a16' in mods
        if packed_2d and dims != ['SQ_RSRC_IMG_2D']:
            raise Unsupported('%s modifier a16 only on a 2D image' % name)
        masks = [m[6:] for m in mods if m.startswith('dmask:')]
        if len(dims) != 1 or len(masks) != 1:
            raise Unsupported('%s without exactly one dim and dmask' % name)
        # Compute also stores to 1D, 2D-array and 3D images: the address is x; x, y, layer; x, y, z, and the bound
        # view is a RWTexture1D, RWTexture2DArray or RWTexture3D (a 3D view keeps the caller's W range, as an
        # array view keeps its slices). Pixel stages store to 2D and 2D-array images (#2830); 1D and 3D stay compute-only.
        shapes = {'SQ_RSRC_IMG_2D': (2, 'tex2d', 'x and y'), 'SQ_RSRC_IMG_2D_ARRAY': (3, 'tex2da', 'x, y and layer'),
                  'SQ_RSRC_IMG_1D': (1, 'tex1d', 'x'), 'SQ_RSRC_IMG_3D': (3, 'tex3d', 'x, y and z')}
        if dims[0] not in shapes or (dims[0] in ('SQ_RSRC_IMG_1D', 'SQ_RSRC_IMG_3D') and self.kind != 'cs') or \
                (d16 and dims[0] in ('SQ_RSRC_IMG_1D', 'SQ_RSRC_IMG_3D')):
            raise Unsupported('%s on %s: only 2D images are lowered' % (name, dims[0]))
        address, image_kind, address_text = shapes[dims[0]]
        try:
            mask = int(masks[0], 0)
        except ValueError:
            raise Unsupported('%s with dmask %s: not an integer' % (name, masks[0])) from None
        # A partial mask writes only the selected channels, in order from consecutive VDATA registers. HLSL's typed
        # UAV store writes all four, so the other channels are read back and stored unchanged (read-modify-write).
        channels = [i for i in range(4) if mask >> i & 1]
        if not channels or mask > 0xf or mask < 0 or (d16 and mask != 0xf):
            raise Unsupported('%s with dmask %s: %s' % (
                name, masks[0], 'a 16-bit store writes all four channels' if channels and d16 and mask <= 0xf
                else 'no channel selected or out of range'))
        data = reg_range(ops[0])
        count = 2 if d16 else len(channels)
        if not data or data[0] != 'v' or data[2] - data[1] + 1 != count or data[2] > 255:
            raise Unsupported('%s data %s: %s VGPRs expected' % (name, ops[0], {1: 'one', 2: 'two', 3: 'three', 4: 'four'}[count]))
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)] if r and r[0] == 'v' else []
        if len(cs) != (1 if packed_2d else address):
            raise Unsupported('%s with %d address registers: %s expected' % (
                name, len(cs), 'one packed register' if packed_2d else address_text))
        words = [self.u('v%d' % n) for n in range(data[1], data[2] + 1)]
        if self.null_descriptor(ops[2]):
            self.descriptor(ops[2])  # verify any overlapping VCC words on all paths
            return
        if self.kind == 'cs':
            if not all(re.fullmatch(r'v\d+', c) and int(c[1:]) <= 255 for c in cs):
                raise Unsupported('compute image_store requires full-width VGPR coordinates')
            key = self.fixed_table_image(ops[2], name)
        else:
            key = self.descriptor(ops[2])
        if d16:
            # RDNA 2 ISA §8.2.5: RG occupy VDATA, BA occupy VDATA+1, low half first.
            rgba = ['f16tof32(%s & 0xffffu), f16tof32(%s >> 16)' % (word, word) for word in words]
            value = 'float4(%s)' % ', '.join(rgba)
        else:
            value = 'asfloat(uint4(%s))' % ', '.join(words)
        image = self.pipe.uav(key, image_kind)
        if packed_2d:
            packed = self.u(cs[0])
            index = 'uint2(%s & 0xffffu, %s >> 16)' % (packed, packed)
        else:
            index = self.u(cs[0]) if address == 1 else 'uint%d(%s)' % (address, ', '.join(self.u(c) for c in cs))
        if len(channels) == 4 or d16:
            body = '%s[%s] = %s;' % (image, index, value)
        else:
            sets = ' '.join('t.%s = asfloat(%s);' % ('xyzw'[c], word) for c, word in zip(channels, words))
            body = '{ t = %s[%s]; %s %s[%s] = t; }' % (image, index, sets, image, index)
            self.pipe.uavs[key]['load'] = True  # manifest 'uavload N': the runtime needs typed UAV load for it
        self.emit(body if self.exec_known else 'if (exec & 1) ' + body)

    def gather_2d_setup(self, name, ops, mods, addr_count, addr_label, *, mip_preclamp, regs_out=None):
        if self.kind not in ('ps', 'cs') or len(ops) != 4:
            raise Unsupported('%s requires a pixel or compute stage and four operands' % name)
        compute_lz = self.kind == 'cs' and name == 'image_gather4_lz'
        for modifier in mods:
            if compute_lz and modifier == 'unorm':
                continue
            if not (modifier.startswith(('dim:', 'dmask:')) or modifier in ('dlc', 'glc', 'slc')):
                raise Unsupported('%s modifier %s' % (name, modifier))
        if [m for m in mods if m.startswith('dim:')] != ['dim:SQ_RSRC_IMG_2D']:
            raise Unsupported('%s requires a 2D texture' % name)
        channels = {'dmask:0x1': ('Red', 0), 'dmask:0x2': ('Green', 1), 'dmask:0x4': ('Blue', 2), 'dmask:0x8': ('Alpha', 3)}
        masks = [m for m in mods if m.startswith('dmask:')]
        if len(masks) != 1 or masks[0] not in channels:
            raise Unsupported('%s requires a one-hot dmask' % name)
        dst, addr, sampler = (reg_range(ops[i]) for i in (0, 1, 3))
        if not dst or dst[0] != 'v' or dst[2] - dst[1] != 3 or dst[2] > 255:
            raise Unsupported('%s requires four destination registers' % name)
        if regs_out is not None and ops[1].startswith('['):
            # Non-sequential address: each register is named, in address order.
            listed = [reg_range(c.strip()) for c in ops[1][1:-1].split(',')]
            if (len(listed) != addr_count or not all(r and r[0] == 'v' and r[1] == r[2] and r[2] <= 255 for r in listed)):
                raise Unsupported('%s requires %s' % (name, addr_label))
            addr = ('v', listed[0][1], listed[-1][1])
            regs_out.extend(r[1] for r in listed)
        elif not addr or addr[0] != 'v' or addr[2] - addr[1] + 1 != addr_count or addr[2] > 255:
            raise Unsupported('%s requires %s' % (name, addr_label))
        elif regs_out is not None:
            regs_out.extend(range(addr[1], addr[2] + 1))
        words = self.const_words(ops[3]) if sampler and sampler[0] == 's' and sampler[2] - sampler[1] == 3 else None
        if words is None:
            raise Unsupported('%s requires a constant sampler' % name)
        from pixel_aot import check_gather_sampler
        profiled = gather_2d_profile_words(words)
        if compute_lz:
            w0, w1, w2, w3 = profiled
            # The static sampler decoder directly represents these RDNA 2 address modes.
            # Half-border and mirror-once-border have no matching D3D12 address mode.
            for shift in (0, 3):
                if ((w0 >> shift) & 7) not in (0, 1, 2, 3, 6):
                    raise Unsupported('%s sampler: unsupported X/Y address mode' % name)
            if w3 >> 30 == 3:
                raise Unsupported('%s sampler: register border color requires a palette' % name)
            # Check the remaining fields against the original profile. The real words are
            # still bound, and FORCE_UNNORMALIZED is handled by the coordinate lowering.
            if ((w2 >> 26) & 3) == 2:
                # At LOD zero the linear mip filter has no second contributing level.
                w2 = (w2 & ~(3 << 26)) | (1 << 26)
            profiled = ((w0 & ~(0x3f | (1 << 15))) | 0x12, w1, w2, w3 & 0x3fffffff)
        if not mip_preclamp and (words[2] & (1 << 28)):
            profiled = (profiled[0], profiled[1], profiled[2] | (1 << 28), profiled[3])
        try:
            # _lz reads mip zero of the bound view; _l checks all other sampler fields against
            # the 2D gather profile and enforces words[1] MAX_LOD plus view mip bounds in HLSL.
            check_gather_sampler(profiled, 1, comparison=False)
        except ValueError as error:
            if compute_lz:
                raise Unsupported('%s sampler: unsupported filter, LOD or descriptor flags' % name) from error
            raise Unsupported('%s sampler: %s' % (name, error)) from error
        if self.kind == 'cs' and not self.null_descriptor(ops[2]):
            key = self.fixed_table_image(ops[2], name)
        else:
            key = self.descriptor(ops[2])
        tex = self.pipe.resource(key, 'tex2d') if key is not None else None
        return dst, addr, words, channels[masks[0]], tex

    def op_image_gather4(self, ops, mods):
        raise Unsupported('image_gather4 requires the guest implicit mip level')

    def op_image_gather4_lz(self, ops, mods):
        regs = [] if self.kind == 'cs' else None
        dst, addr, words, (channel_name, _), tex = self.gather_2d_setup(
            'image_gather4_lz', ops, mods, 2, 'two VGPR address registers' if regs is not None else
            'two contiguous address registers', mip_preclamp=True, regs_out=regs)
        if tex is None:
            return self.null_image_read(dst, 15)
        smp = self.pipe.sampler(tuple(words))
        u, v = (self.f('v%d' % i) for i in (regs if regs is not None else range(addr[1], addr[2] + 1)))
        coords = 'float2(%s, %s)' % (u, v)
        if regs is not None and (words[0] & (1 << 15) or 'unorm' in mods):
            # GetDimensions with no mip argument returns the extent at view mip zero.
            self.emit('{ uint gw, gh; %s.GetDimensions(gw, gh); t = %s.Gather%s(%s, %s / float2(gw, gh)); }' %
                      (tex, tex, channel_name, smp, coords))
        else:
            self.emit('t = %s.Gather%s(%s, %s);' % (tex, channel_name, smp, coords))
        for i, component in enumerate('xyzw', dst[1]):
            self.set_v(i, 'U(t.%s)' % component)

    def constant_vgpr_word(self, n, name):
        """Value of VGPR v<n> when its only writer is one `v_mov_b32 v<n>, <literal>` of a straight-line program."""
        if getattr(self, 'has_branches', True):
            raise Unsupported('%s needs straight-line code to prove its offset register' % name)
        value = None
        for _, _, op, asm in getattr(self, 'ins', ()):
            ops = parse(asm)[1]
            base = re.sub(r'_e(32|64)$', '', op)
            if base.startswith(('exp', 'buffer_store', 'ds_write', 'v_cmpx')) or not ops:
                continue
            if any(r and r[0] == 'v' and r[1] <= n <= r[2] for r in map(reg_range, destinations(op, ops))):
                if value is not None or base != 'v_mov_b32' or len(ops) != 2:
                    raise Unsupported('%s offset register v%d is not one literal move' % (name, n))
                try:
                    value = int(ops[1], 0)
                except ValueError:
                    raise Unsupported('%s offset register v%d is not an integer literal' % (name, n)) from None
        if value is None:
            raise Unsupported('%s offset register v%d is never written' % (name, n))
        return value & 0xffffffff

    def op_image_gather4_lz_o(self, ops, mods):
        """image_gather4_lz_o: image_gather4_lz with a texel offset packed in the first address register.

        RDNA 2 MIMG address order for the _o forms is offset, then the coordinates; the offset register holds
        signed 6-bit X in bits 5:0 and Y in bits 13:8 (LLVM AMDGPU MIMG address layout). The three address
        registers may be contiguous or the non-sequential (NSA) list `[vA, vB, vC]`.
        A literal offset in -8..7 uses HLSL's Gather with a literal offset. Any other offset (a value computed
        at run time, or out of that range) is emulated: the gather footprint of a bilinear fetch at mip zero is
        the 2x2 texels from floor(uv * size - 0.5), moved by the offset and clamped into the image (the
        sampler is validated as normalized clamp), read with Load in the gather order x, y, z, w =
        (0,1), (1,1), (1,0), (0,0). The emulation rounds the footprint in float, not in the sampler's fixed
        point, so a coordinate within 2^-8 of a texel centre can pick the neighbouring footprint.
        """
        name = 'image_gather4_lz_o'
        regs = []
        dst, addr, words, (channel_name, channel_index), tex = self.gather_2d_setup(
            name, ops, mods, 3, 'three address registers', mip_preclamp=True, regs_out=regs)
        if tex is None:
            return self.null_image_read(dst, 15)
        smp = self.pipe.sampler(tuple(words))
        u, v = (self.f('v%d' % i) for i in regs[1:])
        fields = None
        try:
            packed = self.constant_vgpr_word(regs[0], name)
            if packed & ~0x3f3f:
                raise Unsupported('%s offset %#x sets bits outside the X and Y fields' % (name, packed))
            fields = []
            for shift in (0, 8):
                field = (packed >> shift) & 0x3f
                field -= 64 if field & 0x20 else 0
                if not -8 <= field <= 7:
                    raise Unsupported('%s offset %d is outside the HLSL gather range -8..7' % (name, field))
                fields.append(field)
        except Unsupported:
            fields = None
        if fields is not None:
            self.emit('t = %s.Gather%s(%s, float2(%s, %s), int2(%d, %d));' % (tex, channel_name, smp, u, v, *fields))
        else:
            off = self.u('v%d' % regs[0])
            self.emit(('{ uint gw, gh; %s.GetDimensions(gw, gh); int2 gs = int2(gw, gh); '
                       'int2 go = int2(asint((%s) << 26u) >> 26, asint((%s) << 18u) >> 26); '
                       'int2 gb = int2(floor(float2(%s, %s) * float2(gs) - 0.5)) + go; '
                       't = float4(%s.Load(int3(clamp(gb + int2(0, 1), 0, gs - 1), 0)).%s, '
                       '%s.Load(int3(clamp(gb + int2(1, 1), 0, gs - 1), 0)).%s, '
                       '%s.Load(int3(clamp(gb + int2(1, 0), 0, gs - 1), 0)).%s, '
                       '%s.Load(int3(clamp(gb, 0, gs - 1), 0)).%s); }')
                      % (tex, off, off, u, v, tex, 'rgba'[channel_index], tex, 'rgba'[channel_index],
                         tex, 'rgba'[channel_index], tex, 'rgba'[channel_index]))
        for i, component in enumerate('xyzw', dst[1]):
            self.set_v(i, 'U(t.%s)' % component)

    def op_image_gather4_l(self, ops, mods):
        """image_gather4_l on a 2D texture with explicit LOD and a validated constant sampler."""
        dst, addr, words, (_, channel_index), tex = self.gather_2d_setup(
            'image_gather4_l', ops, mods, 3, 'three contiguous address registers', mip_preclamp=False)
        if tex is None:
            return self.null_image_read(dst, 15)
        self.pipe.uses_gather_2d_lod = True
        max_lod = ((words[1] >> 12) & 0xfff) / 256.0
        u, v, lod = (self.f('v%d' % i) for i in range(addr[1], addr[2] + 1))
        self.emit('t = gather_2d_lod(%s, float2(%s, %s), %s, %d, %sf);' % (tex, u, v, lod, channel_index, repr(max_lod)))
        for i, component in enumerate('xyzw', dst[1]):
            self.set_v(i, 'U(t.%s)' % component)

    def op_image_gather4_c_lz(self, ops, mods):
        """image_gather4_c_lz: four red-channel depth comparisons at view mip zero.

        The address dwords are the reference, then u and v, as on the hardware gather. The comparison
        function and the depth compare live in the constant sampler, so the guest words are bound as a
        SamplerComparisonState and the sample profile is the one pixel_aot.check_gather_sampler already
        checks for comparison gathers. Only red is in contract: gather4_c compares the resource's red
        component and D3D's gather4_c takes a single-component view.
        """
        name = 'image_gather4_c_lz'
        if self.kind != 'ps' or len(ops) != 4:
            raise Unsupported('%s requires a pixel stage and four operands' % name)
        for modifier in mods:
            if not (modifier.startswith(('dim:', 'dmask:')) or modifier in ('dlc', 'glc', 'slc')):
                raise Unsupported('%s modifier %s' % (name, modifier))
        if [m for m in mods if m.startswith('dim:')] != ['dim:SQ_RSRC_IMG_2D']:
            raise Unsupported('%s requires a 2D texture' % name)
        masks = [m for m in mods if m.startswith('dmask:')]
        if masks != ['dmask:0x1']:
            raise Unsupported('%s requires red dmask 0x1' % name)
        dst, addr, image, sampler = (reg_range(ops[i]) for i in (0, 1, 2, 3))
        if not dst or dst[0] != 'v' or dst[2] - dst[1] != 3 or dst[2] > 255:
            raise Unsupported('%s requires four destination registers' % name)
        if not addr or addr[0] != 'v' or addr[2] - addr[1] != 2 or addr[2] > 255:
            raise Unsupported('%s requires three contiguous address registers' % name)
        if not image or image[0] != 's' or image[2] - image[1] != 7 or image[2] > 105:
            raise Unsupported('%s requires eight image descriptor registers' % name)
        if not sampler or sampler[0] != 's' or sampler[2] - sampler[1] != 3 or sampler[2] > 105:
            raise Unsupported('%s requires four sampler descriptor registers' % name)
        words = self.const_words(ops[3])
        if words is None:
            raise Unsupported('%s requires a constant sampler' % name)
        from pixel_aot import check_gather_sampler
        try:
            # Both the guest _lz and D3D gather compare at mip zero of the bound view.
            check_gather_sampler(words, 1, comparison=True)
        except ValueError as error:
            raise Unsupported('%s sampler: %s' % (name, error)) from error
        key = self.descriptor(ops[2])
        if key is None:
            return self.null_image_read(dst, 15)
        tex = self.pipe.resource(key, 'tex2d')
        smp = self.pipe.sampler(tuple(words), comparison=True)
        reference, u, v = (self.f('v%d' % i) for i in range(addr[1], addr[2] + 1))
        # The gather is read before any destination register is written, so an alias between the address
        # and destination ranges keeps the original operands.
        self.emit('t = %s.GatherCmp(%s, float2(%s, %s), %s);' % (tex, smp, u, v, reference))
        for i, component in enumerate('xyzw', dst[1]):
            self.set_v(i, 'U(t.%s)' % component)

    def op_v_cvt_i32_f32(self, ops, mods):
        if mods:
            raise Unsupported('v_cvt_i32_f32 with modifiers')
        if len(ops) != 2:
            raise Unsupported('v_cvt_i32_f32 with %d operands, expected 2' % len(ops))
        dst = reg_range(ops[0])[1]
        self.set_v(dst, 'uint(int(F(%s)))' % self.u(ops[1]))

    def op_v_pack_b32_f16(self, ops, mods):
        """D[15:0] = S0.f16, D[31:16] = S1.f16 (AMD RDNA 2 ISA, VOP3 V_PACK_B32_F16).

        op_sel[i] selects the high half of source i (LLVM 15 VOP3Instructions.td: V_PACK_B32_F16 is VOP3_OPSEL).
        The neg and abs source modifiers act on the sign bit (bit 15) of the selected half, abs first. The halves
        are copied as bits, like pixel_aot's lowering of the same instruction: no binary16 denormal flush and no
        NaN quieting is modelled (an assumption, not measured). Register sources only: an inline float constant
        on a binary16 operand is a binary16 value, not the binary32 word u() would give. A destination op_sel bit
        and clamp are refused, and so is any modifier other than the one op_sel:[a,b,c] spelling the decoder
        prints."""
        sel = [0, 0, 0]
        for m in mods:
            match = re.fullmatch(r'op_sel:\[([01]),([01]),([01])\]', m)
            if not match:
                raise Unsupported('v_pack_b32_f16 modifier %s' % m)
            sel = [int(x) for x in match.groups()]
        dst = reg_range(ops[0]) if ops else None
        if sel[2] or len(ops) != 3 or not dst or dst[0] != 'v' or dst[1] != dst[2]:
            raise Unsupported('v_pack_b32_f16 form %s %s' % (', '.join(ops), ' '.join(mods)))
        halves = []
        for i, tok in enumerate(ops[1:3]):
            neg = tok.startswith('-') and not re.match(r'-[\d.]', tok)
            tok = tok[1:] if neg else tok
            absolute = tok.startswith('|') and tok.endswith('|')
            tok = tok[1:-1] if absolute else tok
            reg = reg_range(tok)
            if not reg or reg[1] != reg[2] or reg[0] not in ('v', 's'):
                raise Unsupported('v_pack_b32_f16 source %s' % ops[1 + i])
            # The high half needs the whole word; the low half reads only bits 15:0, so a register whose high half
            # is undefined (after an odd-count d16 load) stays readable here.
            half = '(%s >> 16)' % self.u(tok) if sel[i] else '(%s & 0xffffu)' % self.u(tok, low16_only=True)
            if absolute:
                half = '(%s & 0x7fffu)' % half
            if neg:
                half = '(%s ^ 0x8000u)' % half
            halves.append(half)
        self.set_v(dst[1], '%s | (%s << 16)' % (halves[0], halves[1]))

    def op_v_cvt_pknorm_u16_f32(self, ops, mods):
        name = 'v_cvt_pknorm_u16_f32'
        if mods:
            raise Unsupported('%s with %s' % (name, ' '.join(mods)))
        if len(ops) != 3:
            raise Unsupported('%s needs a destination and two sources' % name)
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[1] != dst[2]:
            raise Unsupported('%s destination %s' % (name, ops[0]))
        lo, hi = (self.f(token) for token in ops[1:])
        self.pipe.uses_pknorm_u16 = True
        self.set_v(dst[1], 'pknorm_u16(asuint(%s)) | (pknorm_u16(asuint(%s)) << 16u)' % (lo, hi))

    def op_v_cvt_pknorm_i16_f32(self, ops, mods):
        name = 'v_cvt_pknorm_i16_f32'
        if mods or len(ops) != 3:
            raise Unsupported('%s needs a destination and two sources without modifiers' % name)
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[1] != dst[2]:
            raise Unsupported('%s destination %s' % (name, ops[0]))
        for token in ops[1:]:
            r = reg_range(token)
            if not (r and r[0] == 'v' and r[1] == r[2]) and self.const_u32(token) is None:
                raise Unsupported('%s source %s is not a lane register or constant' % (name, token))
        lo, hi = (self.u(token) for token in ops[1:])
        self.pipe.uses_pknorm_i16 = True
        self.set_v(dst[1], 'pknorm_i16(%s) | (pknorm_i16(%s) << 16u)' % (lo, hi))

    def op_v_cvt_pk_u8_f32(self, ops, mods):
        # VOP3A opcode 350 of the RDNA 1.0 and RDNA 2 ISA guides:
        #   D = (S2 & ~(0xff << 8 * S1[1:0])) | ((flt32_to_uint8(S0) & 0xff) << 8 * S1[1:0])
        # The guides leave flt32_to_uint8 undefined. Measured on RDNA 2 hardware (RX 6900 XT, gfx1030, 5432 rows,
        # see docs/experiments.md): round to nearest even, saturated to [0, 255], NaN and negatives (-0.0, -inf
        # included) give 0, +inf gives 255. MODE[1:0] = 3 (round toward zero) truncates instead, which is why a write
        # of those bits is refused in s_setreg_imm32_b32. s_denorm_mode 0 changed nothing.
        # Lowering: (bits - 1u) < 0x7f800000u holds exactly from the smallest positive subnormal to +inf and fails
        # for both zeros, every negative and every NaN, so those take the 0u arm. It is an integer test on the raw
        # word: a float compare may be folded under fast-math (DXC emits `fcmp fast ogt`, so NaN would depend on
        # the driver), and min/max treat a NaN operand differently between back ends.
        # Above zero, min(S0, 255.0) saturates (+inf included) and HLSL round() is round_ne (DXIL Round_ne), so the
        # conversion is exact and in range.
        if mods:
            raise Unsupported('v_cvt_pk_u8_f32 with modifiers')
        if len(ops) != 4:
            raise Unsupported('v_cvt_pk_u8_f32 with %d operands, expected 4' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[1] != dst[2]:
            raise Unsupported('v_cvt_pk_u8_f32 destination %s' % ops[0])
        if reg_range(ops[1]):
            src = reg_range(ops[1])
            if src[1] != src[2]:
                raise Unsupported('v_cvt_pk_u8_f32 source %s' % ops[1])
        bits, select, rest = self.u(ops[1]), self.u(ops[2]), self.u(ops[3])
        value = 'F(%s)' % bits
        shift = '((%s & 3u) << 3u)' % select
        self.set_v(dst[1], '((%s & ~(0xffu << %s)) | (((((%s - 1u) < 0x7f800000u) ? uint(round(min(%s, 255.0))) : 0u) & 0xffu) << %s))'
                   % (rest, shift, bits, value, shift))

    def op_s_ff0_i32_b32(self, ops, mods):
        """Find first zero bit; SCC unchanged (#1723).

        RDNA 2 ISA SOP1 / pixel_aot ff0_i32_b32: D = firstbitlow(~S0)
        (0xffffffff when every bit of S0 is set). Lane masks and descriptor-bearing
        SGPRs are refused.
        """
        if mods or len(ops) != 2:
            raise Unsupported('s_ff0_i32_b32 with %d operands, expected 2 unmodified' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('s_ff0_i32_b32 destination %s' % ops[0])
        if self.is_lane_mask(ops[0]) or dst[1] in self.masks:
            raise Unsupported('s_ff0_i32_b32 on a lane mask')
        imm = re.fullmatch(r'-?(0x[0-9a-fA-F]+|\d+)', ops[1])
        if imm:
            value = int(ops[1], 0) & 0xffffffff
            zeros = (~value) & 0xffffffff
            folded = 0xffffffff if zeros == 0 else (zeros & -zeros).bit_length() - 1
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded))
            return None
        src = reg_range(ops[1])
        if not src or src[0] != 's' or src[1] != src[2] or src[1] > 105:
            raise Unsupported('s_ff0_i32_b32 source %s' % ops[1])
        if self.is_lane_mask(ops[1]) or src[1] in self.masks:
            raise Unsupported('s_ff0_i32_b32 on a lane mask')
        sym = self.sym.get(src[1])
        if sym and sym[0] not in ('const', 'ud', 'fixed_buffer_word'):
            raise Unsupported('s_ff0_i32_b32 on %s data' % sym[0])
        if sym and sym[0] == 'const':
            value = sym[1] & 0xffffffff
            zeros = (~value) & 0xffffffff
            folded = 0xffffffff if zeros == 0 else (zeros & -zeros).bit_length() - 1
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded))
            return None
        src_u = self.u(ops[1])
        self.set_s(dst[1], 'firstbitlow(~%s)' % src_u)
        return None

    def op_s_ff1_i32_b32(self, ops, mods):
        """Find first set bit in a 32-bit scalar; SCC unchanged (#2022 Item F).

        The compute-entry special case on `exec_lo` still folds to lane 0; in a pixel stage `exec_lo` is the
        first active lane for a later v_readlane_b32 (first_active_lane). Every other accepted
        source uses `firstbitlow` (0xffffffff when the word is zero), matching `pixel_aot`.
        """
        if mods or len(ops) != 2:
            raise Unsupported('s_ff1_i32_b32 with %d operands, expected 2 unmodified' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('s_ff1_i32_b32 destination %s' % ops[0])
        if ops[1] == 'exec_lo':
            if self.kind == 'ps':
                self.first_active_lane(dst[1])
                return None
            if self.kind != 'cs':
                raise Unsupported('s_ff1_i32_b32 of exec_lo outside compute')
            if getattr(self, 'past_entry', False) or not getattr(self, 'exec_known', False):
                raise Unsupported('s_ff1_i32_b32 requires the compute entry EXEC mask')
            # At dispatch entry even the last partial wave starts at lane zero.
            self.set_s(dst[1], '0u', ('const', 0))
            return None
        if self.is_lane_mask(ops[0]) or self.is_lane_mask(ops[1]) or dst[1] in self.masks:
            raise Unsupported('s_ff1_i32_b32 on a lane mask')
        imm = re.fullmatch(r'-?(0x[0-9a-fA-F]+|\d+)', ops[1])
        if imm:
            value = int(ops[1], 0) & 0xffffffff
            # firstbitlow of zero is -1; otherwise the lowest set bit index.
            folded = 0xffffffff if value == 0 else (value & -value).bit_length() - 1
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded))
            return None
        src = reg_range(ops[1])
        if not src or src[0] != 's' or src[1] != src[2] or src[1] > 105:
            raise Unsupported('s_ff1_i32_b32 source %s' % ops[1])
        if src[1] in self.masks:
            raise Unsupported('s_ff1_i32_b32 on a lane mask')
        sym = self.sym.get(src[1])
        if sym and sym[0] not in ('const', 'ud', 'fixed_buffer_word'):
            raise Unsupported('s_ff1_i32_b32 on %s data' % sym[0])
        if sym and sym[0] == 'const':
            value = sym[1] & 0xffffffff
            folded = 0xffffffff if value == 0 else (value & -value).bit_length() - 1
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded))
            return None
        self.set_s(dst[1], 'firstbitlow(%s)' % self.u(ops[1]))
        return None

    def op_s_ff1_i32_b64(self, ops, mods):
        """Find first set bit of a 64-bit source; SCC unchanged (#2022 Item F).

        A 32-bit SOP1 literal or inline constant zero-extends to 64 bits (high word 0)
        and folds. An SGPR pair uses the same 64-bit firstbitlow scan as SCALAR_BITS
        (`_ctz64` / #2072). Kept as a Stage method so immediates are not refused by the
        pair-only SCALAR_BITS template.
        """
        if mods or len(ops) != 2:
            raise Unsupported('s_ff1_i32_b64 with %d operands, expected 2 unmodified' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('s_ff1_i32_b64 destination %s' % ops[0])
        if self.is_lane_mask(ops[0]) or dst[1] in self.masks:
            raise Unsupported('s_ff1_i32_b64 on a lane mask')
        imm = re.fullmatch(r'-?(0x[0-9a-fA-F]+|\d+)', ops[1])
        if imm:
            # Encoding: one 32-bit literal/inline; the ISA zero-extends it (hi = 0).
            value = int(ops[1], 0) & 0xffffffff
            folded = 0xffffffff if value == 0 else (value & -value).bit_length() - 1
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded))
            return None
        src = reg_range(ops[1])
        if src is None or src[0] != 's' or src[2] > 105 or src[2] - src[1] != 1:
            raise Unsupported('s_ff1_i32_b64 source %s' % ops[1])
        if any(r in self.masks for r in range(src[1], src[2] + 1)):
            raise Unsupported('s_ff1_i32_b64 on a lane mask')
        lo, hi = self.S(src[1]), self.S(src[2])
        self.masks.discard(dst[1])
        self.set_s(dst[1], _ctz64(lo, hi))
        return None

    def op_s_abs_i32(self, ops, mods):
        """Signed absolute value; SCC = result != 0 (#1723).

        RDNA 2 ISA SOP1 / pixel_aot: D = asint(S0) < 0 ? (0u - S0) : S0 (INT_MIN stays).
        Lane masks and descriptor-bearing SGPRs are refused so a projected bit is not treated
        as a full signed word.
        """
        if mods or len(ops) != 2:
            raise Unsupported('s_abs_i32 with %d operands, expected 2 unmodified' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('s_abs_i32 destination %s' % ops[0])
        if self.is_lane_mask(ops[0]) or dst[1] in self.masks:
            raise Unsupported('s_abs_i32 on a lane mask')
        imm = re.fullmatch(r'-?(0x[0-9a-fA-F]+|\d+)', ops[1])
        src_sym = None
        if imm:
            value = int(ops[1], 0) & 0xffffffff
        else:
            src = reg_range(ops[1])
            if not src or src[0] != 's' or src[1] != src[2] or src[1] > 105:
                raise Unsupported('s_abs_i32 source %s' % ops[1])
            if self.is_lane_mask(ops[1]) or src[1] in self.masks:
                raise Unsupported('s_abs_i32 on a lane mask')
            src_sym = self.sym.get(src[1])
            if src_sym and src_sym[0] not in ('const', 'ud', 'fixed_buffer_word'):
                raise Unsupported('s_abs_i32 on %s data' % src_sym[0])
            value = src_sym[1] & 0xffffffff if src_sym and src_sym[0] == 'const' else None
        if value is not None:
            folded = (-value if value & 0x80000000 else value) & 0xffffffff
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded), scc='$ != 0')
            return None
        expr = '(asint(%s) < 0 ? (0u - %s) : %s)' % ((self.u(ops[1]),) * 3)
        self.set_s(dst[1], expr, scc='$ != 0')
        return None

    def op_s_flbit_i32_b32(self, ops, mods):
        """Find first bit high (leading-zero count); SCC unchanged (#1723).

        RDNA 2 ISA SOP1 / pixel_aot dpp_ffbh_u32: D = S0 == 0 ? 0xffffffff : 31 - firstbithigh(S0).
        Lane masks and descriptor-bearing SGPRs are refused.
        """
        if mods or len(ops) != 2:
            raise Unsupported('s_flbit_i32_b32 with %d operands, expected 2 unmodified' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('s_flbit_i32_b32 destination %s' % ops[0])
        if self.is_lane_mask(ops[0]) or dst[1] in self.masks:
            raise Unsupported('s_flbit_i32_b32 on a lane mask')
        imm = re.fullmatch(r'-?(0x[0-9a-fA-F]+|\d+)', ops[1])
        if imm:
            value = int(ops[1], 0) & 0xffffffff
            folded = 0xffffffff if value == 0 else 31 - (value.bit_length() - 1)
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded))
            return None
        src = reg_range(ops[1])
        if not src or src[0] != 's' or src[1] != src[2] or src[1] > 105:
            raise Unsupported('s_flbit_i32_b32 source %s' % ops[1])
        if self.is_lane_mask(ops[1]) or src[1] in self.masks:
            raise Unsupported('s_flbit_i32_b32 on a lane mask')
        sym = self.sym.get(src[1])
        if sym and sym[0] not in ('const', 'ud', 'fixed_buffer_word'):
            raise Unsupported('s_flbit_i32_b32 on %s data' % sym[0])
        if sym and sym[0] == 'const':
            value = sym[1] & 0xffffffff
            folded = 0xffffffff if value == 0 else 31 - (value.bit_length() - 1)
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded))
            return None
        src_u = self.u(ops[1])
        self.set_s(dst[1], '(%s == 0u ? 0xffffffffu : 31u - firstbithigh(%s))' % (src_u, src_u))
        return None

    def op_s_quadmask_b64(self, ops, mods):
        """Collapse each nibble of a 64-bit source into bits of D.lo; D.hi = 0 (#1723).

        RDNA 2 ISA SOP1 / pixel_aot: bit g of D.lo is 1 iff nibble g of the 64-bit source is
        nonzero (sixteen nibbles → bits 0..15). SCC = (D.lo | D.hi) != 0.
        Integer inline constants are 64-bit; 32-bit literals zero-extend.
        Lane masks and descriptor SGPRs are refused.
        """
        if mods or len(ops) != 2:
            raise Unsupported('s_quadmask_b64 with %d operands, expected 2 unmodified' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[2] != dst[1] + 1 or dst[2] > 105 or dst[1] % 2:
            raise Unsupported('s_quadmask_b64 destination %s' % ops[0])
        if any(r in self.masks for r in range(dst[1], dst[2] + 1)) or self.is_lane_mask(ops[0]):
            raise Unsupported('s_quadmask_b64 on a lane mask')

        def nibble_mask(word):
            folded = 0
            for group in range(8):
                if word & (0xf << (group * 4)):
                    folded |= 1 << group
            return folded

        def nibble_expr(src_u):
            return '(%s)' % ' | '.join(
                '((%s & 0x%xu) != 0u ? %du : 0u)' % (src_u, 0xf << (g * 4), 1 << g)
                for g in range(8))

        imm = re.fullmatch(r'0x[0-9a-fA-F]{1,8}|-?\d+', ops[1])
        if imm:
            value = int(ops[1], 0) & 0xffffffffffffffff
            folded = nibble_mask(value & 0xffffffff) | (nibble_mask(value >> 32) << 8)
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded), scc='$ != 0')
            self.set_s(dst[1] + 1, '0u', ('const', 0))
            return None
        src = reg_range(ops[1])
        if src is None or src[0] != 's' or src[2] > 105 or src[2] - src[1] != 1 or src[1] % 2:
            raise Unsupported('s_quadmask_b64 source %s' % ops[1])
        if any(r in self.masks for r in range(src[1], src[2] + 1)):
            raise Unsupported('s_quadmask_b64 on a lane mask')
        for n in range(src[1], src[2] + 1):
            sym = self.sym.get(n)
            if sym and sym[0] not in ('const', 'ud', 'fixed_buffer_word'):
                raise Unsupported('s_quadmask_b64 on %s data' % sym[0])
        lo_sym, hi_sym = self.sym.get(src[1]), self.sym.get(src[2])
        if (lo_sym and lo_sym[0] == 'const' and hi_sym and hi_sym[0] == 'const'):
            folded = nibble_mask(lo_sym[1] & 0xffffffff) | (nibble_mask(hi_sym[1] & 0xffffffff) << 8)
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded), scc='$ != 0')
            self.set_s(dst[1] + 1, '0u', ('const', 0))
            return None
        lo_u, hi_u = self.S(src[1]), self.S(src[2])
        # Low 8 bits from the low word; next 8 bits from the high word. High result word is 0.
        expr = '((%s) | ((%s) << 8u))' % (nibble_expr(lo_u), nibble_expr(hi_u))
        self.set_s(dst[1], expr, scc='$ != 0')
        self.set_s(dst[1] + 1, '0u', ('const', 0))
        return None

    def op_s_bitreplicate_b64_b32(self, ops, mods):
        """Write each bit of a 32-bit source twice: D.u64 bit 2i and 2i+1 = S0 bit i; SCC unchanged (Run R, 3c50f16a).

        RDNA 2 ISA SOP1 S_BITREPLICATE_B64_B32 (the same result as pixel_aot's s_bitreplicate_b64_b32). The low
        destination dword holds the replicated source bits 15:0, the high dword bits 31:16. An integer inline
        constant or a 32-bit literal is the 32-bit source and folds. Lane masks and a source that is also a
        destination register are refused (the pair would be written before the second half reads the source).
        """
        if mods or len(ops) != 2:
            raise Unsupported('s_bitreplicate_b64_b32 with %d operands, expected 2 unmodified' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[2] != dst[1] + 1 or dst[2] > 105 or dst[1] % 2:
            raise Unsupported('s_bitreplicate_b64_b32 destination %s' % ops[0])
        if any(r in self.masks for r in range(dst[1], dst[2] + 1)) or self.is_lane_mask(ops[0]):
            raise Unsupported('s_bitreplicate_b64_b32 on a lane mask')

        def replicate(word, shift):
            folded = 0
            for bit in range(16):
                if (word >> (shift + bit)) & 1:
                    folded |= 3 << (2 * bit)
            return folded

        def fold(word):
            lo, hi = replicate(word, 0), replicate(word, 16)
            self.set_s(dst[1], '0x%08xu' % lo, ('const', lo))
            self.set_s(dst[1] + 1, '0x%08xu' % hi, ('const', hi))

        if re.fullmatch(r'-?(0x[0-9a-fA-F]+|\d+)', ops[1]):
            fold(int(ops[1], 0) & 0xffffffff)
            return None
        src = reg_range(ops[1])
        if not src or src[0] != 's' or src[1] != src[2] or src[1] > 105:
            raise Unsupported('s_bitreplicate_b64_b32 source %s' % ops[1])
        if src[1] in self.masks or self.is_lane_mask(ops[1]):
            raise Unsupported('s_bitreplicate_b64_b32 on a lane mask')
        sym = self.sym.get(src[1])
        if sym and sym[0] not in ('const', 'ud', 'fixed_buffer_word'):
            raise Unsupported('s_bitreplicate_b64_b32 on %s data' % sym[0])
        if sym and sym[0] == 'const':
            fold(sym[1] & 0xffffffff)
            return None
        if dst[1] <= src[1] <= dst[2]:
            raise Unsupported('s_bitreplicate_b64_b32 source %s is a destination register' % ops[1])
        value = self.S(src[1])

        def expr(shift):
            return '(%s)' % ' | '.join('((%s & 0x%xu) != 0u ? 0x%xu : 0u)' % (value, 1 << (shift + bit), 3 << (2 * bit))
                                       for bit in range(16))

        self.set_s(dst[1], expr(0))
        self.set_s(dst[1] + 1, expr(16))
        return None

    def op_s_quadmask_b32(self, ops, mods):
        """Collapse each nibble of S0 to one bit of D; SCC = result != 0 (#1723).

        RDNA 2 ISA SOP1 / pixel_aot: bit g of D is 1 iff nibble g of S0 is nonzero.
        Lane masks and descriptor-bearing SGPRs are refused.
        """
        if mods or len(ops) != 2:
            raise Unsupported('s_quadmask_b32 with %d operands, expected 2 unmodified' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('s_quadmask_b32 destination %s' % ops[0])
        if self.is_lane_mask(ops[0]) or dst[1] in self.masks:
            raise Unsupported('s_quadmask_b32 on a lane mask')
        imm = re.fullmatch(r'-?(0x[0-9a-fA-F]+|\d+)', ops[1])
        if imm:
            value = int(ops[1], 0) & 0xffffffff
            folded = 0
            for group in range(8):
                if value & (0xf << (group * 4)):
                    folded |= 1 << group
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded), scc='$ != 0')
            return None
        src = reg_range(ops[1])
        if not src or src[0] != 's' or src[1] != src[2] or src[1] > 105:
            raise Unsupported('s_quadmask_b32 source %s' % ops[1])
        if self.is_lane_mask(ops[1]) or src[1] in self.masks:
            raise Unsupported('s_quadmask_b32 on a lane mask')
        sym = self.sym.get(src[1])
        if sym and sym[0] not in ('const', 'ud', 'fixed_buffer_word'):
            raise Unsupported('s_quadmask_b32 on %s data' % sym[0])
        if sym and sym[0] == 'const':
            value = sym[1] & 0xffffffff
            folded = 0
            for group in range(8):
                if value & (0xf << (group * 4)):
                    folded |= 1 << group
            self.set_s(dst[1], '0x%08xu' % folded, ('const', folded), scc='$ != 0')
            return None
        src_u = self.u(ops[1])
        parts = ['((%s & 0x%xu) != 0u ? %du : 0u)' % (src_u, 0xf << (g * 4), 1 << g) for g in range(8)]
        self.set_s(dst[1], '(%s)' % ' | '.join(parts), scc='$ != 0')
        return None

    # ---- slot 57 mesh stage: instructions mesh_skeleton() puts in place of the group-memory exchanges ----
    def op_x2w_mesh_lane0(self, ops, mods):
        # `s_mov_b64 exec, 1` leaves lane 0 of each wave running. The saved copy of that mask is not modelled.
        r = reg_range(ops[0])
        for n in range(r[1], r[2] + 1):
            self.sym[n] = ('the saved mesh lane-0 mask',)
        self.emit('exec = (tid & 63u) == 0u ? 1u : 0u;')
        self.exec_known = self.exec_narrow_only = False

    def op_x2w_mesh_group(self, ops, mods):
        # The word is entry v5 >> 7 of wave 0, lane 0. What v5 holds at entry is not verified, so a use is refused.
        self.sym[reg_range(ops[0])[1]] = ('the mesh group index',)

    def op_x2w_mesh_firstlane(self, ops, mods):
        self.mesh_collect_reads[self.pc] = self.u(ops[1])
        self.set_s(reg_range(ops[0])[1], 'mesh_first', uniform=True)

    def mesh_mask_select(self, ops):
        if not (self.scc_verified and self.scc_uniform):
            raise Unsupported('slot 57 mesh: mask select needs a proven wave-uniform SCC')

        def contribution(tok):
            if tok in MESH_MASK_WORDS:
                bit, half = MESH_MASK_WORDS[tok]
                if bit == 'vcc':
                    self.read_vcc()
            else:
                value = pc_word(tok)
                if value == 0:
                    return '0u'
                if value == -1:
                    return 'tid & 32u ? 0u : 1u'
                reg = reg_range(tok)
                if not reg or reg[0] != 's' or reg[1] != reg[2]:
                    raise Unsupported('slot 57 mesh: unproven mask select source %s' % tok)
                n = reg[1]
                half = 32 if n in self.mask_high else 0
                low = n - 1 if half else n
                if low not in self.masks or {low, n} & self.join_masks:
                    raise Unsupported('slot 57 mesh: unproven mask select source %s' % tok)
                bit = self.S(low)
            return '((tid & 32u) == %du ? (%s & 1u) : 0u)' % (half, bit)

        yes, no = (contribution(tok) for tok in ops[1:])
        self.mesh_collect_masks[self.pc] = 'scc ? (%s) : (%s)' % (yes, no)
        dst = ops[0]
        if dst in MESH_MASK_WORDS:
            bit, half = MESH_MASK_WORDS[dst]
            # SOP2 writes just one half of the guest mask. Keep the other half as it was.
            self.emit('%s = (tid & 32u) == %du ? ((mesh_mask >> (tid & 31u)) & 1u) : %s;'
                      % (bit, half, bit))
            if bit == 'exec':
                self.exec_known = self.exec_narrow_only = False
        else:
            reg = reg_range(dst)
            if not reg or reg[0] != 's' or reg[1] != reg[2] or reg[1] > 105:
                raise Unsupported('slot 57 mesh: unsupported mask select destination %s' % dst)
            self.set_s(reg[1], 'mesh_mask', uniform=True)

    def check_mesh_collectives(self, sites, branches):
        # A logical wave must reach a collective with all its lanes or none. A lane-only
        # branch is safe before the collective if both paths reconverge there or both miss it.
        offsets = [off for off, _, _, _ in self.ins]
        indices = {off: i for i, off in enumerate(offsets)}
        jumps = dict(branches)
        for site in sites:
            if offsets.count(site) != 1 or any(target <= source for source, target in branches):
                raise Unsupported('slot 57 mesh: a wave collective in a loop is not covered')
            at = indices[site]
            reach = [0] * (len(offsets) + 1)
            reach[at] = 1
            for i in range(at - 1, -1, -1):
                off, _, name, _ = self.ins[i]
                paths = [] if name in ('s_branch', 's_endpgm') else [reach[i + 1]]
                if off in jumps:
                    paths.append(reach[indices[jumps[off]]])
                if off in self.divergent_branches and paths and (min(paths) != max(paths) or 2 in paths):
                    raise Unsupported('slot 57 mesh: a wave collective can be bypassed by part of a wave')
                reach[i] = paths[0] if paths and min(paths) == max(paths) else 2 if paths else 0

    def op_x2w_mesh_counts(self, ops, mods):
        prims, verts, temporary = (reg_range(tok)[1] for tok in ops)
        self.emit('if (tid == 0u) { lds[0] = %s; lds[1] = %s; }' % (self.S(prims), self.S(verts)))
        self.emit('GroupMemoryBarrierWithGroupSync();')
        # Thread t writes vertex t and primitive t, so a count above the thread count names outputs that no
        # thread writes. The clamp keeps the declared arrays in bounds and changes nothing a thread writes.
        for n, word in ((prims, 0), (verts, 1)):
            self.set_s(n, 'min(lds[%d], %du)' % (word, MESH_THREADS), uniform=True)
        # Without a vertex no index is valid, so no primitive is.
        self.emit('if (%s == 0u) %s = 0u;' % (self.S(verts), self.S(prims)))
        self.emit('SetMeshOutputCounts(%s, %s);' % (self.S(verts), self.S(prims)))
        self.mesh_verts = self.S(verts)
        self.sym[temporary] = ('a mesh allocation temporary',)

    def mesh_export(self, asm, target, vals, flags):
        if target == 'target_20':
            # First word: three 9-bit vertex indices at bits 0, 10 and 20, an edge flag above each (not modelled:
            # triangles are filled) and the null bit 31 (Mesa ac_nir_pack_ngg_prim_exp_arg).
            if vals[1:] != ['off'] * 3 or [f for f in flags if f != 'done']:
                raise Unsupported('mesh primitive export form %r' % asm)
            # An index at or past the vertex count has no defined vertex; the clamp keeps it inside the array.
            # A null primitive names one vertex three times, which covers no pixel.
            self.emit('if (exec & 1) { w = %s; tris[tid] = w >> 31 != 0u ? (uint3)0u : '
                      'min(uint3(w, w >> 10, w >> 20) & 511u, max(%s, 1u) - 1u); }'
                      % (self.u(vals[0]), self.mesh_verts))
            return None
        first = len(self.lines)
        self.in_mesh_export = True
        try:
            self.export(asm)
        finally:
            self.in_mesh_export = False
        self.lines[first:] = ['if (exec & 1) { %s mesh_vertex = true; }' % ' '.join(self.lines[first:])]
        return None

    def op_v_mbcnt_lo_u32_b32(self, ops, mods, high=False):
        name = 'v_mbcnt_%s_u32_b32' % ('hi' if high else 'lo')
        if self.kind == 'ms':
            # V_MBCNT counts the mask bits strictly below the lane, in the low or high half of a wave64 mask.
            dst = reg_range(ops[0]) if len(ops) == 3 else None
            if mods or not dst or dst[0] != 'v' or dst[1] != dst[2] or ops[1] != '-1':
                raise Unsupported('%s in a mesh stage requires a VGPR, the mask -1 and a base' % name)
            below = 'max(tid & 63u, 32u) - 32u' if high else 'min(tid & 63u, 32u)'
            self.set_v(dst[1], '(%s + %s)' % (self.u(ops[2]), below))
            return
        if self.kind != 'cs':
            raise Unsupported(name)
        if mods or len(ops) != 3:
            raise Unsupported('%s requires three unmodified operands' % name)
        x, y, z = self.pipe.group
        multi_wave = x * y * z > 32
        if multi_wave:
            wave = self.pipe.wave_size
            if wave != 32 and (wave is not None or not wave32_exec_proof(self.ins)):
                raise Unsupported('%s in a larger group requires a proven wave32 program' % name)
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[1] != dst[2] or dst[1] > 255:
            raise Unsupported('%s destination %s' % (name, ops[0]))
        # Both halves use the tested constant-mask profile, even though HI only copies the base.
        mask = self.const_u32(ops[1])
        if mask is None:
            raise Unsupported('%s requires a known mask' % name)
        base = reg_range(ops[2])
        base_is_vgpr = base and base[0] == 'v' and base[1] == base[2] and base[2] <= 255
        if not base_is_vgpr and self.const_u32(ops[2]) is None:
            raise Unsupported('%s requires a VGPR or known base' % name)
        result = self.u(ops[2])
        # RDNA 2 V_MBCNT counts mask bits strictly below the lane. A small group
        # fits in the low half of either guest wave size, including lane 31.
        if not high:
            lane = '(tid.x + %du * (tid.y + %du * tid.z))' % (x, y)
            if multi_wave:
                # The packed mask already supplies every lane's bit. Compute its prefix
                # directly in guest lane order, independent of host wave size and active lanes.
                lane = '(%s & 31u)' % lane
            result = '(countbits(0x%08xu & ((1u << %s) - 1u)) + %s)' % (mask, lane, result)
        self.set_v(dst[1], result)

    def op_v_mbcnt_hi_u32_b32(self, ops, mods):
        self.op_v_mbcnt_lo_u32_b32(ops, mods, high=True)

    def proven_uniform_v(self, n):
        """VGPR n holds one value in every lane EXEC enables now. In the entry block EXEC is known; past it, a
        recomputable expression proves no join, EXEC write or overwrite lies between the definition and here."""
        if not self.tracks_uniformity() or n not in self.uniform_v:
            return False
        # Past the entry block of a branch-free program is only dead code after s_endpgm: nothing to recompute.
        if getattr(self, 'past_entry', True):
            return n in getattr(self, 'uniform_expr', {}) and getattr(self, 'has_branches', False)
        return self.uniform_entry_exec()

    def firstlane_source(self, token, base):
        if base in VEC_F:
            token = re.sub(r'^-(?=[^\d.])', '', token).strip('|')
        reg = reg_range(token)
        if reg and reg[0] == 'v':
            return all(n in self.firstlane_v for n in range(reg[1], reg[2] + 1))
        return self.uniform_source(token) and (
            not reg or reg[1] in self.defined_uniform_s or
            (self.sym.get(reg[1]) or ('',))[0] in ('const', 'ud', 'fixed_buffer_word'))

    def firstlane_wave_allowed(self, src):
        # SM 6 waves have at least four lanes. A group this small fits in one wave
        # on both guest widths and does not need a host wave-size attribute.
        x, y, z = self.pipe.group
        if x * y * z > 4 or self.has_branches or self.blocks:
            return False
        for off, _, name, asm in self.ins:
            if off >= self.pc:
                break
            args = parse(asm)[1]
            if writes_exec(name, args) and not name.startswith('v_cmpx'):
                return False
            if name.startswith('v_writelane') and reg_range(args[0]) == src:
                return False
        return True

    def op_v_readfirstlane_b32(self, ops, mods):
        if len(ops) != 2:
            raise Unsupported(READFIRSTLANE_UNIFORM)
        src, dst = reg_range(ops[1]), reg_range(ops[0])
        past_entry = getattr(self, 'past_entry', True)
        waterfalls = getattr(self, 'waterfall_reads', {})
        # A value defined with entry EXEC is still in the VGPR if no path since then could overwrite it.
        direct = src and src[1] in getattr(self, 'entry_written_v', ()) and self.uniform_entry_exec()
        uniform = src and (direct or self.proven_uniform_v(src[1]))
        if getattr(self, 'pc', None) in waterfalls and not uniform:
            reason = waterfalls[self.pc]
            if reason:
                raise Unsupported('compute readfirstlane waterfall: ' + reason)
            self.active_waterfall_loads.update(self.waterfall_scalar_loads[self.pc])
            self.set_s(dst[1], self.u(ops[1]), uniform=False)  # the selected key varies between lanes
            return
        if (mods or not dst or dst[0] != 's' or dst[1] != dst[2] or
                not src or src[0] != 'v' or src[1] != src[2]):
            raise Unsupported(READFIRSTLANE_UNIFORM)
        if not uniform:
            if getattr(getattr(self, 'pipe', None), 'retry_firstlane', False):
                if (self.firstlane_cfg and self.uniform_cf and src[1] in self.firstlane_v
                        and (self.firstlane_ps_exec is None or self.pc in self.firstlane_ps_exec)):
                    self.set_s(dst[1], self.u(ops[1]), uniform=True)
                    return
                if (self.firstlane_cfg and self.uniform_cf and src[1] in self.firstlane_ps_load_v
                        and self.pc in self.firstlane_ps_full_exec):
                    self.set_s(dst[1], self.u(ops[1]), uniform=True)
                    return
                if getattr(self, 'kind', None) == 'cs' and self.firstlane_wave_allowed(src):
                    x, y, _ = self.pipe.group
                    # Elect by guest lane order. The winning host lane supplies its value
                    # even when host lane numbering differs or guest EXEC is empty.
                    self.emit('{ uint lane = tid.x + %du * (tid.y + %du * tid.z);' % (x, y))
                    self.emit('  uint first = WaveActiveMin((exec & 1u) != 0u ? lane : 0xffffffffu);')
                    self.emit('  uint value = 0u;')
                    self.emit('  [branch] if (lane == (first == 0xffffffffu ? 0u : first))'
                              ' value = WaveReadLaneFirst(%s);' % self.u(ops[1]))
                    self.set_s(dst[1], 'WaveActiveBitOr(value)', uniform=True)
                    self.emit('}')
                    return
            raise Unsupported(READFIRSTLANE_UNIFORM)
        if past_entry and not direct:
            # The first active lane ran the definition: no join, EXEC write or overwrite of the VGPR and no write
            # of a scalar it read lies between them. A lane inactive at the definition kept an older VGPR value,
            # so the read recomputes the definition from the scalars instead of reading this lane's copy.
            self.set_s(dst[1], self.uniform_expr[src[1]])
        else:
            self.set_s(dst[1], self.u(ops[1]))

    # ---- SGPR spill to VGPR lanes (#2125 item E; mesh #2413 T7 / #2127) ----------------------
    # The compiler parks a scalar in one lane of a VGPR: `v_writelane_b32 vN, sM, k` writes sM into lane k
    # and `v_readlane_b32 sD, vN, k` reads it back (AMD RDNA 2 ISA reference, VOP3 lane ops; LLVM 15
    # AMDGPU spill lowering). Both ignore EXEC. With a constant k and a VGPR that no other instruction
    # touches, vN is only a table of scalar slots, so one scalar variable per (vN, k) is an exact model
    # and no other lane is ever read. A mixed-use VGPR instead needs a per-lane write. Constant-lane
    # reloads of a mixed-use VGPR still need a separate proof because ordinary vector writes can change the slot.
    # Lanes 0..63 cover a wave64 program. A wave32 program uses only bits 4:0 of the lane select (RDNA 2 ISA,
    # v_readlane/v_writelane: the lane is taken modulo the wave size), so k and k + 32 name one slot there and two in
    # wave64; the translator does not know the wave size, so a VGPR whose lanes include such a pair is refused.
    # Slot 57 mesh is wave64 (#2127). Vertex stages use the same model: the two instructions touch no other lane and no stage input.
    #
    # With branches the slot variable is assigned on the pixel's own path, while the hardware slot holds the last
    # write on the wave's path. The two paths differ only inside the region of a branch not proven uniform (an
    # s_cbranch_execz or _vccz the pixel takes alone, or an SCC branch on varying data), where the wave also runs
    # the blocks this pixel skips. spill_reaching gives the writes that can supply a read, written on every path.
    # One write: every path into the read ran it, so the slot is one more scalar register; it is accepted unless
    # the write lies in such a region that the read is outside (a loop or arm the wave can run after this pixel
    # left it). Several writes: the value is chosen by the path, so every write must lie outside every such region;
    # the reload then keeps a constant only if all of them stored it, and is not claimed uniform.
    # Compute stages (cs) use the same proof, branches included. A spilled descriptor or pointer word (SPILL_SYMBOLS) is a
    # symbol with no run-time value: the slot carries the symbol, the reload restores it, and the word keeps the
    # provenance the formatted loads require (formatted_descriptor_from_block follows a reload to the writes that supply it).
    # Lane masks (compute only). The translator models a lane mask (vcc_lo, or an SGPR in self.masks) per invocation
    # as that invocation's own bit: `vcc` / `sN` hold bit 0 of the wave mask, and a v_cmp stores `w` (all lanes active)
    # or `exec & w`, so an inactive lane's bit is already 0 in the variable. `v_writelane_b32 vN, M, k` copies the whole
    # wave mask into lane k whatever EXEC is, `v_readlane_b32 D, vN, k` copies it back to every lane, so a mask that is
    # only ever used as a mask afterwards (reject_lane_mask_data and the mask rules refuse every numeric read of a marked
    # register, vcc included) has, after the reload, each invocation's bit equal to its bit at the write: the spill is
    # the identity on the per-invocation bit. The slot variable spN_k is therefore assigned the mask's own variable at
    # the write, in every invocation, and the reload copies it into the destination and marks it a mask again (or
    # assigns vcc for a `vcc_lo` destination). The slot proof of spill_join (written on every path, no loop-carried
    # read, no write in a region the wave runs while the invocation skips it; writes in s_cbranch_execz blocks are
    # refused by execz_refusals) decides what a reload sees. A reload that may see two writes of which one is a mask
    # is refused: nothing proves they store the same mask. exec, vcc and vcc_hi (64-bit forms), and m0 stay refused
    # as sources, a mask register in join_masks (mark not proven on every path) too.
    SPILL_LANES = 64
    SPILL_STAGES = ('ps', 'vs', 'ms', 'cs')
    # Symbols a spill carries by identity: the SGPR holds no run-time value in the translation, so the slot holds none either.
    SPILL_SYMBOLS = ('desc', 'ptr', 'ptrhi')

    def spill_reaching(self, cfg):
        """Instruction index of each v_readlane_b32 -> the v_writelane_b32 indices that can supply its (VGPR, lane), or
        None when some path from the entry reaches it with none. Forward must-dataflow over the blocks: a slot is
        written at a join only if it is written on every incoming edge (intersection) and the writes that can supply
        it are the union over those edges, iterated to a fixed point so a later write in a loop body reaches the loop
        head through the back edge."""
        bounds, _, succ, _ = cfg
        lane_ops = {}
        for i, (_, _, name, asm) in enumerate(self.ins):
            base = re.sub(r'_e(32|64)$', '', name)
            ops = parse(asm)[1] if base in ('v_writelane_b32', 'v_readlane_b32') else ()
            if len(ops) == 3:
                try:
                    lane = int(ops[2], 0)  # 3 and 0x3 are one slot
                except ValueError:
                    lane = ops[2]  # an SGPR lane, refused by spill_operands
                lane_ops[i] = (base == 'v_writelane_b32', ops[0 if base == 'v_writelane_b32' else 1], lane)
        into = [{}] + [None] * (len(bounds) - 1)  # None: no path reaches the block yet
        reach = {}
        changed = True
        while changed:
            changed = False
            for b, (lo, hi) in enumerate(bounds):
                if into[b] is None:
                    continue
                state = dict(into[b])
                for i in range(lo, hi):
                    if i in lane_ops:
                        write, vreg, lane = lane_ops[i]
                        if write:
                            state[vreg, lane] = frozenset([i])
                        else:
                            reach[i] = state.get((vreg, lane))
                for t in succ[b]:
                    new = state if into[t] is None else {k: into[t][k] | state[k] for k in into[t].keys() & state.keys()}
                    if new != into[t]:
                        into[t], changed = new, True
        return reach

    def spill_join(self, n, lane):
        """(sym, uniform) of the slot a v_readlane_b32 reads in a program with branches, or a refusal naming what
        failed (see the comment above SPILL_LANES)."""
        i = self.descriptor_instruction_index
        writes = self.spill_reach.get(i)
        if writes is None:
            raise Unsupported('v_readlane_b32 reads v%d lane %d with no v_writelane_b32 of it on some path' % (n, lane))
        if max(writes) > i:
            raise Unsupported('v_readlane_b32 reads v%d lane %d written later in a loop (loop-carried)' % (n, lane))
        if any(w not in self.spill_writes for w in writes):
            raise Unsupported('v_readlane_b32 reads v%d lane %d from a v_writelane_b32 that was not translated' % (n, lane))
        _, block_of, _, div_blocks = self.spill_cfg
        here = block_of[i]
        for off, blocks in sorted(div_blocks.items()):
            if off in self.uniform_branches or not any(block_of[w] in blocks for w in writes):
                continue
            if len(writes) > 1:
                raise Unsupported('v_readlane_b32 joins %d writes of v%d lane %d, one inside the region of the branch at '
                                  '%#x, which is not proven uniform' % (len(writes), n, lane, off))
            if here not in blocks:
                raise Unsupported('v_readlane_b32 reads v%d lane %d outside the region of the branch at %#x that writes '
                                  'it, which is not proven uniform' % (n, lane, off))
        stored = [self.spill_writes[w] for w in sorted(writes)]
        if any(s[3] for s in stored) and len(stored) > 1:
            # Which mask the reload holds would depend on the path; nothing proves the writes store the same one.
            raise Unsupported('v_readlane_b32 joins %d writes of v%d lane %d, one of them a lane mask' % (len(stored), n, lane))
        sym = stored[0][0] if all(s[0] == stored[0][0] for s in stored) else None
        if sym is None and any(s[0] is not None and s[0][0] in self.SPILL_SYMBOLS for s in stored):
            # A descriptor or pointer word has no run-time variable: only the same symbol on every path can be reloaded.
            raise Unsupported('v_readlane_b32 joins different descriptor or pointer words of v%d lane %d' % (n, lane))
        return sym, len(stored) == 1 and stored[0][1], all(s[2] for s in stored), stored[0][3]

    def require_mask_reads_only(self, key):
        """Refuse a reloaded lane mask (scalar state `key`: 'vcc' or 'sN') that an instruction reachable before its next
        write reads as anything but a mask. The translator marks the register a mask but does not refuse every numeric
        read of one, and a numeric read would see the whole wave mask where the model has one bit. Accepted reads: a lane
        select or carry-in (per_lane_read), a read by s_cbranch_vccz / _vccnz, a copy into exec_lo or vcc_lo, and another
        spill of it."""
        effects = [scalar_effects(name, asm) for _, _, name, asm in self.ins]
        index = {off: i for i, (off, *_) in enumerate(self.ins)}
        todo, seen = [self.descriptor_instruction_index + 1], set()
        while todo:
            j = todo.pop()
            if j in seen or j >= len(self.ins):
                continue
            seen.add(j)
            _, _, name, asm = self.ins[j]
            base = re.sub(r'_e(32|64)$', '', name)
            ops = parse(asm)[1]
            if key in effects[j][0]:
                copy = base == 's_mov_b32' and len(ops) == 2 and ops[0] in ('exec_lo', 'vcc_lo') and key in scalar_tokens(ops[1])
                if not (self.per_lane_read(j, key) or copy or base == 'v_writelane_b32' or
                        (key == 'vcc' and base in ('s_cbranch_vccz', 's_cbranch_vccnz'))):
                    raise Unsupported('v_readlane_b32 reloads a lane mask that %s reads as data' % name)
            if key in effects[j][1]:
                continue
            target = None
            if base == 's_branch' or base.startswith('s_cbranch'):
                target = index.get(int(ops[0].split('_')[1], 16))
            if base not in ('s_branch', 's_endpgm'):
                todo.append(j + 1)
            if target is not None:
                todo.append(target)

    def spills_lane_masks(self):
        return getattr(self, 'kind', None) == 'cs'

    def spill_operands(self, name, ops, mods):
        if mods or len(ops) != 3:
            raise Unsupported('%s with modifiers or %d operands' % (name, len(ops)))
        vreg, sreg = ((reg_range(ops[0]), reg_range(ops[1])) if name == 'v_writelane_b32'
                      else (reg_range(ops[1]), reg_range(ops[0])))
        if (not vreg or vreg[0] != 'v' or vreg[1] != vreg[2] or vreg[1] > 255 or
                (sreg is not None and (sreg[0] != 's' or sreg[1] != sreg[2] or sreg[1] > 103))):
            raise Unsupported('%s requires one VGPR and one plain SGPR' % name)
        try:
            lane = int(ops[2], 0)
        except ValueError:
            raise Unsupported('%s requires a constant lane, got %s' % (name, ops[2])) from None
        kind = getattr(self, 'kind', None)
        if not 0 <= lane < self.SPILL_LANES:
            raise Unsupported('%s lane %d outside 0..%d' % (name, lane, self.SPILL_LANES - 1))
        if kind not in self.SPILL_STAGES:
            raise Unsupported('%s is lowered in pixel, vertex, mesh and compute stages only' % name)
        if not self.tracks_uniformity() or (getattr(self, 'has_branches', True) and
                                            (kind not in ('ps', 'cs') or getattr(self, 'spill_reach', None) is None)):
            raise Unsupported('%s needs straight-line code to prove write-before-read' % name)
        n = vreg[1]
        vector_op = self.spill_vector_use(n)
        if vector_op:
            if name != 'v_writelane_b32' or kind == 'ms':
                raise Unsupported('%s: v%d is also used as a vector (%s)' % (name, n, vector_op))
        elif kind != 'ms':  # slot 57 mesh is always wave64, so k and k + 32 are two slots
            self.require_spill_lanes_apart(n, name)
        return n, sreg, lane, vector_op is not None

    def require_spill_lanes_apart(self, n, name):
        """Refuse when two lanes of spill VGPR v<n> differ by 32: one slot in wave32, two in wave64."""
        lanes = set()
        for _, _, op, asm in getattr(self, 'ins', ()):
            if re.sub(r'_e(32|64)$', '', op) not in ('v_writelane_b32', 'v_readlane_b32'):
                continue
            ops = parse(asm)[1]
            if len(ops) == 3 and n in [r[1] for r in map(reg_range, ops[:2]) if r and r[0] == 'v']:
                try:
                    lanes.add(int(ops[2], 0))
                except ValueError:
                    continue  # a register lane is refused where it is read
        for lane in lanes:
            if lane + 32 in lanes:
                raise Unsupported('%s: v%d lanes %d and %d are one slot in wave32 and two in wave64'
                                  % (name, n, lane, lane + 32))

    def spill_vector_use(self, n):
        """Return the first ordinary vector use of a lane register."""
        # Mesh: scan the pre-skeleton program. The accepted group-memory exchanges rewrite to
        # x2w_mesh_* / s_nop, which would hide ordinary use of exchange VGPRs (v5 and temps).
        source = getattr(self, 'mesh_source_ins', None) or getattr(self, 'ins', ())
        for _, _, op, asm in source:
            if re.sub(r'_e(32|64)$', '', op) in ('v_writelane_b32', 'v_readlane_b32'):
                continue
            for lo, hi, single in re.findall(r'\bv\[(\d+):(\d+)\]|\bv(\d+)\b', asm):
                a, b = (int(lo), int(hi)) if lo else (int(single), int(single))
                if a <= n <= b:
                    return op

    def op_v_writelane_b32(self, ops, mods):
        n, src, lane, vector_use = self.spill_operands('v_writelane_b32', ops, mods)
        if vector_use and self.has_branches:
            _, block_of, _, regions = self.spill_cfg
            block = block_of[self.descriptor_instruction_index]
            if any(block in blocks and off not in self.uniform_branches for off, blocks in regions.items()):
                raise Unsupported('v_writelane_b32 vector update in divergent control flow')
        mask = False
        if src is None:
            # Slot 57 mesh keeps ordinary scalar spills only (R0Y00 on #2637). Raw EXEC words are not
            # proven under the one-lane model (exec_hi would be wrong under full EXEC; exec_lo under
            # a lane-0 mask would reload per thread). mesh_entry_aot still models EXEC for #2127.
            if ops[1] == 'vcc_lo' and self.spills_lane_masks():
                value, sym, uniform, proven, mask = 'vcc', None, False, False, True
            elif ops[1] in ('vcc', 'vcc_lo', 'vcc_hi', 'exec', 'exec_lo', 'exec_hi', 'm0'):
                raise Unsupported('v_writelane_b32 source %s is a lane mask or m0' % ops[1])
            else:
                value, sym, uniform, proven = self.u(ops[1]), None, True, False
                if re.fullmatch(r'0x[0-9a-f]{8}u', value):
                    sym = ('const', int(value[2:-1], 16))
        else:
            held = self.sym.get(src[1]) or ('',)
            carried = held[0] in self.SPILL_SYMBOLS and getattr(self, 'kind', None) == 'cs'
            mask = self.spills_lane_masks() and src[1] in self.masks
            if mask:
                if src[1] in self.join_masks:
                    raise Unsupported('v_writelane_b32 source s%d holds a lane mask not proven on every path' % src[1])
                value, sym, uniform, proven = self.S(src[1]), (held if held[0] == 'execcopy' else None), False, False
            elif not carried and (self.is_lane_mask(ops[1]) or held[0] not in ('', 'const', 'ud', 'fixed_buffer_word')):
                raise Unsupported('v_writelane_b32 source s%d holds a lane mask or descriptor' % src[1])
            elif carried:
                # A descriptor or pointer word of a compute stage: the slot holds the symbol itself, so the reload
                # restores the same word. Nothing is emitted: the translation has no run-time value for it.
                if not names_one_value(held):
                    raise Unsupported('v_writelane_b32 source s%d holds a modified descriptor or pointer word' % src[1])
                value, sym, uniform, proven = None, held, True, self.spill_word_proven(src[1], held)
            else:
                value = self.S(src[1])
                sym = held if held[0] == 'const' else None
                uniform = self.uniform_source(ops[1])
                proven = False
        if vector_use:
            if value is None or mask or not uniform:
                raise Unsupported('v_writelane_b32 vector update requires a uniform numeric source')
            index = 'WaveGetLaneIndex()'
            wave_size = None
            if self.kind == 'cs':
                x, y, z = self.pipe.group
                if x * y * z > 32:
                    raise Unsupported('v_writelane_b32 vector update requires at most 32 threads per group')
                index = '(tid.x + %du * (tid.y + %du * tid.z))' % (x, y)
                wave_size = self.pipe.wave_size
            if lane >= 32:
                if wave_size is None and wave32_exec_proof(self.ins):
                    wave_size = 32
                if wave_size is None:
                    raise Unsupported('v_writelane_b32 vector update lane %d requires a known guest wave size' % lane)
                lane %= wave_size
            # This is a read-modify-write that ignores EXEC. Unselected lanes retain their old bits and
            # any outstanding low-half-only restriction. The new value is no longer proven uniform.
            self.set_v(n, '(%s == %du) ? %s : v%d' % (index, lane, value, n),
                       low16_only=n in (getattr(self, 'low16_only_v', None) or ()), ignore_exec=True)
            getattr(self, 'uniform_expr', {}).pop(n, None)
            return
        slots = self.__dict__.setdefault('spill_slots', {})
        slots[(n, lane)] = (sym, uniform, proven, mask)
        if getattr(self, 'spill_reach', None) is not None:
            self.spill_writes[self.descriptor_instruction_index] = (sym, uniform, proven, mask)
        if value is not None:
            self.emit('sp%d_%d = %s;' % (n, lane, value))

    # The first active lane of a pixel stage (#2618, the F rows). `s_ff1_i32_b32 sD, exec_lo` gives the index of the
    # lowest lane in EXEC (RDNA 2 ISA, S_FF1_I32_B32) and `v_readlane_b32 sX, vY, sD` copies vY from that lane, so the
    # pair reads vY from the first active lane. The console packs pixels into waves in its own way, so a correct
    # program cannot depend on which pixel that is (the contract of the #2835 wave reductions), and the pair becomes
    # WaveReadLaneFirst(vY) over the active lanes of the desktop wave. The lane number itself means nothing on the
    # desktop, so sD is a symbol with no run-time value and every other read of it is refused. Accepted only where the
    # one-lane model holds the whole wave's EXEC: a straight-line pixel stage whose EXEC is the entry mask at the
    # s_ff1 and at each readlane, without s_wqm (helper lanes are then out of EXEC on the console, as they are out of
    # desktop wave operations). The decoder prints wave32 syntax, and in wave64 exec_lo is half the wave (an empty
    # low half gives -1, which v_readlane reads as lane 63), so a program that names exec_hi or the 64-bit exec is
    # refused.
    def first_active_lane(self, n):
        what = 's_ff1_i32_b32 s%d, exec_lo in a pixel stage' % n
        if getattr(self, 'has_branches', True) or getattr(self, 'blocks', True):
            raise Unsupported('%s with branches: only straight-line pixel stages are lowered' % what)
        if not getattr(self, 'exec_known', False):
            raise Unsupported('%s: EXEC is not the entry mask' % what)
        if getattr(self, 'wqm_exec', False):
            raise Unsupported('%s after s_wqm exec, exec: the first lane may be a helper lane' % what)
        ins = self.ins
        if any('exec_hi' in scalar_tokens(' '.join(parse(asm)[1])) for _o, _n, _name, asm in ins):
            raise Unsupported('%s of a program that names exec_hi or the 64-bit exec (in wave64, exec_lo is half the '
                              'wave)' % what)
        index = next(i for i, item in enumerate(ins) if item[0] == self.pc)
        key = 's%d' % n
        for _off, _n, name, asm in ins[index + 1:]:
            reads, writes, may = scalar_effects(name, asm)
            ops = parse(asm)[1]
            if key in reads and not (re.sub(r'_e(32|64)$', '', name) == 'v_readlane_b32' and len(ops) == 3 and
                                     ops[2] == key):
                raise Unsupported('%s: %s reads the lane number as data (the desktop numbers its lanes in its own '
                                  'way); only the lane select of v_readlane_b32 is lowered' % (what, asm))
            if key in writes or key in may:
                break
        self.set_s(n, 'WaveReadLaneFirst(WaveGetLaneIndex())', sym=('first_lane',))

    def read_first_lane(self, ops, mods):
        """v_readlane_b32 sX, vY, sD with sD from first_active_lane: WaveReadLaneFirst(vY)."""
        what = 'v_readlane_b32 %s of the first active lane' % ', '.join(ops)
        dst, src = reg_range(ops[0]), reg_range(ops[1])
        if mods:
            raise Unsupported('%s with %s' % (what, ' '.join(mods)))
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105 or src[1] != src[2] or src[1] > 255:
            raise Unsupported('%s: expected one SGPR destination and one VGPR' % what)
        if not getattr(self, 'exec_known', False) or getattr(self, 'wqm_exec', False):
            raise Unsupported('%s: EXEC is no longer the entry mask of the s_ff1_i32_b32' % what)
        for _o, _n, name, asm in self.ins:
            other = parse(asm)[1]
            if re.sub(r'_e(32|64)$', '', name) == 'v_writelane_b32' and other and reg_range(other[0]) == src:
                raise Unsupported('%s: v%d is a lane spill written by v_writelane_b32' % (what, src[1]))
        self.set_s(dst[1], 'WaveReadLaneFirst(%s)' % self.u(ops[1]), uniform=False)

    def op_v_readlane_b32(self, ops, mods):
        if len(ops) != 3 or reg_range(ops[1]) is None or reg_range(ops[1])[0] != 'v':
            raise Unsupported('v_readlane_b32 requires a VGPR source')
        lane = reg_range(ops[2])
        if lane and lane[0] == 's' and lane[1] == lane[2] and getattr(self, 'sym', {}).get(lane[1]) == ('first_lane',):
            self.read_first_lane(ops, mods)
            return
        n, dst, lane, _ = self.spill_operands('v_readlane_b32', ops, mods)
        if getattr(self, 'has_branches', False):
            sym, uniform, proven, mask = self.spill_join(n, lane)
        else:
            slot = getattr(self, 'spill_slots', {}).get((n, lane))
            if slot is None:
                raise Unsupported('v_readlane_b32 reads v%d lane %d before any v_writelane_b32 wrote it' % (n, lane))
            sym, uniform, proven, mask = slot
        if dst is None:
            if not (mask and ops[0] == 'vcc_lo'):
                raise Unsupported('v_readlane_b32 requires an SGPR destination')
            self.emit('vcc = sp%d_%d;' % (n, lane))
            self.require_mask_reads_only('vcc')
            return
        self.set_s(dst[1], 'sp%d_%d' % (n, lane), sym=None if mask else sym, uniform=uniform)
        if mask:
            self.masks.add(dst[1])
            if sym is not None:
                self.sym[dst[1]] = sym
            self.require_mask_reads_only('s%d' % dst[1])
        if getattr(self, 'descriptor_instruction_index', None) is not None:  # unit probes build a Stage without translate()
            self.__dict__.setdefault('reload_proven', {})[self.descriptor_instruction_index] = proven

    def op_v_permlane16_b32(self, ops, mods):
        """v_permlane16_b32 vdst, vsrc0, ssrc1, ssrc2 (RDNA 2 ISA, VOP3 lane permutes; LLVM 15 gfx10): inside each row of
        16 lanes, lane i reads vsrc0 of lane sel[i] of the same row, where sel[i] is the 4-bit field i of ssrc1 (lanes
        0..7) or of ssrc2 (lanes 8..15). The op_sel flags (fetch-inactive, bound_ctrl) only decide what a disabled lane
        reads, so the form is taken only where every lane is enabled: a straight-line compute program with entry EXEC
        in a group of whole rows. Exchanged through groupshared memory like the DPP swaps (compute_quad_gate)."""
        name = 'v_permlane16_b32'
        dst, src = (reg_range(ops[0]), reg_range(ops[1])) if len(ops) == 4 else (None, None)
        if not dst or not src or dst[0] != 'v' or src[0] != 'v' or dst[1] != dst[2] or src[1] != src[2]:
            raise Unsupported('%s requires VGPR destination and source and two lane selects' % name)
        if any(not re.fullmatch(r'op_sel:\[[01],[01]\]', m) for m in mods):
            raise Unsupported('%s with %s' % (name, ' '.join(m for m in mods if not re.fullmatch(r'op_sel:\[[01],[01]\]', m))))
        if getattr(self, 'kind', None) != 'cs':
            raise Unsupported('%s outside a compute program' % name)
        self.compute_quad_gate(name, row_ror8=True)
        x, y, _z = self.pipe.group
        flat = '(tid.x + %du * (tid.y + %du * tid.z))' % (x, y)
        lane = '(%s & 15u)' % flat
        select = '(((%s < 8u ? %s : %s) >> ((%s & 7u) * 4u)) & 15u)' % (lane, self.u(ops[2]), self.u(ops[3]), lane)
        self.uses_quad_shared = True
        self.emit('x2w_quad[%s] = %s;' % (flat, self.u(ops[1])))
        self.emit('GroupMemoryBarrierWithGroupSync();')
        self.set_v(dst[1], 'x2w_quad[(%s & ~15u) | %s]' % (flat, select))
        self.emit('GroupMemoryBarrierWithGroupSync();')

    def op_v_mov_b32(self, ops, mods):
        if len(ops) != 2:
            raise Unsupported('v_mov_b32 with %d operands, expected 2' % len(ops))
        self.reject_lane_mask_data('v_mov_b32', ops[1])
        align4 = not mods and self.operand_align4(ops[1])
        dst = reg_range(ops[0])[1]
        src = reg_range(ops[1])
        copy = not mods and src and src[0] == 'v' and src[1] == src[2]
        if copy and getattr(self, 'has_branches', False) and hasattr(self, 'low16_copies'):
            self.low16_copies.setdefault(src[1], set()).add(dst)
        self.set_v(dst, self.u(ops[1], low16_only=copy),
                   low16_only=copy and src[1] in self.low16_only_v, align4=align4)

    def op_v_swap_b32(self, ops, mods):
        # VOP1 only (llvm-mc rejects the e64, DPP and SDWA spellings for gfx1013). Both registers exchange
        # their old values, so one is read into a temporary before either is written. The ISA allows no
        # input or output modifier and no scalar or constant operand: both operands are plain VGPRs.
        if mods:
            raise Unsupported('v_swap_b32 with %s' % mods)
        if len(ops) != 2:
            raise Unsupported('v_swap_b32 with %d operands, expected 2' % len(ops))
        regs = [reg_range(o) for o in ops]
        if any(not r or r[0] != 'v' or r[1] != r[2] or r[1] > 255 for r in regs):
            raise Unsupported('v_swap_b32 operands %s, %s: expected two single VGPRs' % (ops[0], ops[1]))
        dst, src = regs[0][1], regs[1][1]
        for reg in (dst, src):
            self.note_whole_read(reg)
        body = '{ uint t = v%d; v%d = v%d; v%d = t; }' % (dst, dst, src, src)
        # Uniformity travels with the value; under an unknown exec mask neither register keeps the proof.
        uniform = (dst in self.uniform_v and self.exec_known, src in self.uniform_v and self.exec_known)
        for reg, flag in zip((src, dst), uniform):
            (self.uniform_v.add if flag else self.uniform_v.discard)(reg)
        if isinstance(self.align4_v, set):
            aligned = (dst in self.align4_v and self.exec_known, src in self.align4_v and self.exec_known)
            for reg, flag in zip((src, dst), aligned):
                (self.align4_v.add if flag else self.align4_v.discard)(reg)
        exprs = getattr(self, 'uniform_expr', {})
        moved = {src: exprs.pop(dst, None), dst: exprs.pop(src, None)}
        exprs.update({reg: e for reg, e in moved.items() if e is not None and reg in self.uniform_v})
        self.used_v.update((dst, src))
        self.emit(body if self.exec_known else 'if (exec & 1) ' + body)

    def op_v_cndmask_b32(self, ops, mods):
        # The select takes no output modifier: a clamp or mul on it would change the value, so a
        # modifier that reaches here is refused rather than dropped from the emitted text.
        if mods:
            raise Unsupported('v_cndmask_b32 with %s' % mods)
        if len(ops) not in (3, 4):
            raise Unsupported('v_cndmask_b32 with %d operands, expected 3 or 4' % len(ops))
        for source in ops[1:3]:
            self.reject_lane_mask_data('v_cndmask_b32', source)
        mask = self.u(ops[3]) if len(ops) > 3 else 'vcc'
        if len(ops) == 4 and re.fullmatch(r's\d+', ops[3]):
            low = int(ops[3][1:])
            words = {low, low + 1}
            if words & getattr(self, 'mesh_ballot_regs', set()):
                if (low % 2 or low >= 105 or words & (self.masks | self.mask_high | self.join_masks)
                        or not all(self.uniform_source('s%d' % n) for n in words)):
                    raise Unsupported('slot 57 mesh: unproven scalar pair for v_cndmask_b32')
                mask = '((tid & 32u ? %s : %s) >> (tid & 31u))' % (self.S(low + 1), mask)
        align4 = self.operand_align4(ops[1]) and self.operand_align4(ops[2])
        self.set_v(reg_range(ops[0])[1],
                   '(%s & 1) ? %s : %s' % (mask, self.u_src(ops[2]), self.u_src(ops[1])), align4=align4)

    def op_v_interp_p1_f32(self, ops, mods):
        if ops[1].startswith(('-', '|')):
            raise Unsupported('interpolation source modifier %s' % ops[1])
        if hasattr(self, 'uniform_v'):
            self.uniform_v.discard(reg_range(ops[0])[1])
        if isinstance(self.align4_v, set):
            self.align4_v.discard(reg_range(ops[0])[1])
        self.note_attr(ops[2], 'linear')

    def op_v_interp_p2_f32(self, ops, mods):
        if ops[1].startswith(('-', '|')):
            raise Unsupported('interpolation source modifier %s' % ops[1])
        attr, comp = self.note_attr(ops[2], 'linear')
        self.set_v(reg_range(ops[0])[1], 'U(i.param%d.%s)' % (attr, comp))

    def op_v_interp_mov_f32(self, ops, mods):
        # Only p0 is the flat copy. p10 (P1 - P0) and p20 (P2 - P0) need the per-vertex attribute
        # values, which the FXC path does not have, so they are refused rather than read as p0.
        if ops[1] != 'p0':
            raise Unsupported('v_interp_mov_f32 parameter selector %s: only p0 is lowered' % ops[1])
        attr, comp = self.note_attr(ops[2], 'flat')
        self.set_v(reg_range(ops[0])[1], 'i.fparam%d.%s' % (attr, comp))

    def interp_f16(self, op, ops, mods, vsrc, n_ops):
        """Common checks of v_interp_p1ll_f16 / v_interp_p2_f16: -> (attr, field, high).

        The attribute channel holds two binary16 values; `field` is the interpolated float the pair reads
        (`param0.x`, `hparam0_lo.z`). Two vertex exports exist, with different mappings:
        - compr (see export()): the pair is split into float channels, so half h of attribute channel c is float
          channel 2 * c + h of the parameter: attrN.x = (param.x, param.y), attrN.y = (param.z, param.w). Channels
          z and w have no such pair.
        - plain (a 32-bit export, e.g. halves packed by v_cvt_pkrtz_f16_f32): half h of channel c is
          f16tof32(word_c >> 16 * h), carried by the extra outputs hparamN_lo (h = 0) and hparamN_hi (h = 1),
          channel c of each. All four channels exist.
        """
        allowed = ('high', INTERP_DST_HIGH) if op == 'v_interp_p2_f16' else ('high',)
        extra = [m for m in mods if m not in allowed]
        if extra:
            raise Unsupported('%s with %s: only the plain form and high are lowered' % (op, ' '.join(extra)))
        if len(ops) != n_ops:
            raise Unsupported('%s with %d operands, expected %d' % (op, len(ops), n_ops))
        for tok in (ops[0], ops[1]) + tuple(ops[3:]):
            if not re.fullmatch(r'v\d+', tok):
                raise Unsupported('%s operand %s: only plain VGPRs are lowered' % (op, tok))
        m = re.fullmatch(r'attr(\d+)\.([xyzw])', ops[2])
        if not m:
            raise Unsupported('%s interpolant %s: expected attrN.x/.y/.z/.w' % (op, ops[2]))
        attr, component = self.note_attr(ops[2], 'f16')
        chan, high = 'xyzw'.index(component), int('high' in mods)
        pipe = getattr(self, 'pipe', None)
        vs = getattr(pipe, 'vs', None)
        mesh_plain = getattr(pipe, 'mesh_plain_params', None)
        if (vs is not None and attr in vs.plain_params and attr not in vs.compr_params) or (
                mesh_plain is not None and attr in mesh_plain):
            self.f16_plain.setdefault(attr, set()).add(chan)
            return attr, 'hparam%d_%s.%s' % (attr, ('lo', 'hi')[high], component), high
        if chan > 1:
            raise Unsupported('%s on channel %s: a compr export fills only the x and y channels' % (op, m.group(2)))
        return attr, 'param%d.%s' % (attr, 'xyzw'[2 * chan + high]), high

    def interp_f16_pair(self, op, ops):
        """The entry I/J registers: the PERSP_CENTER pair of the pixel inputs, I in the first, J in the second."""
        base = getattr(getattr(self, 'pipe', None), 'persp_center_vgpr', 0)
        if base is None:
            raise Unsupported('%s: the pixel inputs do not enable the PERSP_CENTER barycentrics' % op)
        want = 'v%d' % (base + (0 if op == 'v_interp_p1ll_f16' else 1))
        if ops[1] != want:
            raise Unsupported('%s barycentric operand %s: only the PERSP_CENTER entry %s (%s) is lowered' % (
                op, ops[1], 'I' if op == 'v_interp_p1ll_f16' else 'J', want))
        return base

    def interp_f16_scan(self):
        """Whole-program proof for the f16 interpolation pair (no fixed-point analysis: any doubt is a refusal).

        The lowering reads the rasterizer's interpolant, so it is only faithful if (1) the entry I/J registers are
        never written (they still hold the entry barycentrics wherever the pair reads them), (2) v_interp_p1ll_f16,
        which emits nothing, has no consumer but the v_interp_p2_f16 that follows it in the same straight-line run
        and nothing reads the destination it leaves stale when p2 writes elsewhere, and (3) nothing reads the
        upper 16 bits of a plain p2 destination: whether gfx10 keeps or clears them is not established by a source
        (LLVM 15 SIInstrInfo/VOP3Instructions.td model the result as f16 only; the encoding states no
        high-half behaviour), so only a plain v_cvt_f32_f16 reading the low half may consume it, until a p2 with
        op_sel 0x8 writes the high half of that register (Mesa ACO keeps the other half, see check_encoding).
        """
        if self.interp_scanned:
            return
        pair = None
        targets = set()
        rows = []
        for off, _, name, asm in self.ins:
            n, ops, mods = parse(asm)
            n = re.sub(r'_e(32|64)$', '', n)
            rows.append((off, n, ops, mods))
            target = getattr(self, 'calculated_pc', {}).get(off)
            if n.startswith('s_cbranch') or n == 's_branch':
                target = int(ops[0].split('_')[1], 16)
            if target is not None:
                # A backward branch needs no refusal of its own: every branch instruction is a boundary below, where
                # the scan requires that no p1ll result is pending, no p2 result is stale and no destination has an
                # undefined upper half. Nothing the scan tracks therefore travels along a back edge, and the loop head
                # (also a boundary, entered with the same empty state) starts clean on every iteration.
                targets.add(target)

        def regs(tok):
            r = reg_range(tok)
            return set(range(r[1], r[2] + 1)) if r and r[0] == 'v' else set()

        pending, stale, high_undefined = {}, set(), set()
        for off, n, ops, mods in rows:
            branch_boundary = off in targets or n.startswith('s_cbranch') or n in ('s_branch', 's_setpc_b64')
            if branch_boundary and pending:
                raise Unsupported('a branch between v_interp_p1ll_f16 %s and its v_interp_p2_f16: p2 could be '
                                  'reached without the matching p1ll' % ', '.join('v%d' % r for r in sorted(pending)))
            if branch_boundary and (stale or high_undefined):
                raise Unsupported('unmodelled v_interp_p2_f16 result across a branch')
            store = n == 'exp' or n.startswith(INTERP_STORE_PREFIXES)
            # A malformed pair member is refused by its own handler; the scan models only the plain spelling.
            interp = (n in ('v_interp_p1ll_f16', 'v_interp_p2_f16') and len(ops) == (3 if n[10] == '1' else 4) and
                      all(re.fullmatch(r'v\d+', o) for o in ops[:2] + ops[3:]))
            if n in ('v_interp_p1ll_f16', 'v_interp_p2_f16') and not interp:
                return  # its own handler refuses the program; nothing after it is worth proving
            written = regs(ops[0]) if ops and not store else set()
            if interp:
                read = regs(ops[1])
                pair = self.interp_f16_pair(n, ops)
            else:
                read = set().union(*(regs(o) for o in ops)) if ops else set()
            low_reader = (n == 'v_cvt_f32_f16' and not mods and len(ops) == 2 and
                          all(re.fullmatch(r'v\d+', o) for o in ops))
            for r in sorted(read):
                if r in pending:
                    raise Unsupported('v%d is the v_interp_p1ll_f16 result, which emits nothing: %s reads it before '
                                      'the matching v_interp_p2_f16' % (r, n))
                if r in stale:
                    raise Unsupported('v%d is the v_interp_p1ll_f16 result, which emits nothing: %s reads it after '
                                      'a v_interp_p2_f16 that wrote another register' % (r, n))
                if r in high_undefined and not (low_reader and r == int(ops[1][1:])):
                    raise Unsupported('%s reads the upper 16 bits of the v_interp_p2_f16 result v%d: not modelled' %
                                      (n, r))
            if n == 'v_interp_p1ll_f16' and interp:
                pending[int(ops[0][1:])] = (ops[2], 'high' in mods)
            elif n == 'v_interp_p2_f16' and interp:
                d, out = int(ops[3][1:]), int(ops[0][1:])
                if pending.get(d) != (ops[2], 'high' in mods):
                    raise Unsupported('v_interp_p2_f16 source %s is not the v_interp_p1ll_f16 result of %s' %
                                      (ops[3], ops[2]))
                del pending[d]
                if out in pending:
                    raise Unsupported('v_interp_p2_f16 overwrites the pending v_interp_p1ll_f16 result v%d' % out)
                stale.discard(out)
                if out != d:
                    stale.add(d)
                if INTERP_DST_HIGH in mods:
                    # Completes a 16-bit pair: the low half it keeps must be a plain p2 result still waiting for its
                    # upper half (no other writer is modelled), after which every reader sees defined halves.
                    if out not in high_undefined:
                        raise Unsupported('v_interp_p2_f16 op_sel 0x8 into v%d, which holds no v_interp_p2_f16 low '
                                          'half: the half it keeps is not modelled' % out)
                    high_undefined.discard(out)
                else:
                    high_undefined.add(out)
            elif low_reader:
                high_undefined.discard(int(ops[0][1:]))
                stale.discard(int(ops[0][1:]))
        # I and J: never written anywhere
        base = pair if pair is not None else (getattr(getattr(self, 'pipe', None), 'persp_center_vgpr', 0) or 0)
        for off, n, ops, mods in rows:
            store = n == 'exp' or n.startswith(INTERP_STORE_PREFIXES)
            if ops and not store and regs(ops[0]) & {base, base + 1}:
                raise Unsupported('%s writes v%d: the entry barycentric registers must still hold the entry values '
                                  'where v_interp_p1ll_f16 / v_interp_p2_f16 read them' % (n, min(
                                      regs(ops[0]) & {base, base + 1})))
            if n == 'v_swap_b32' and len(ops) == 2 and regs(ops[1]) & {base, base + 1}:
                raise Unsupported('v_swap_b32 writes the entry barycentric registers')
        self.interp_scanned = True

    def op_v_interp_p1ll_f16(self, ops, mods):
        # P0 + I * P10 in f32 from a binary16 attribute. The rasterizer interpolates the float channel, so the
        # statement itself emits nothing; p2 reads the interpolated value. The destination is remembered so that
        # p2 can be checked to continue this very pass.
        attr, field, high = self.interp_f16('v_interp_p1ll_f16', ops, mods, 1, 3)
        self.interp_f16_pair('v_interp_p1ll_f16', ops)
        self.interp_f16_scan()
        if self.interp_p1 is None:
            self.interp_p1 = {}
        self.interp_p1[int(ops[0][1:])] = (attr, field, high)

    def op_v_interp_p2_f16(self, ops, mods):
        # P1 + J * P20, converted to binary16 into one half of the destination, the other half kept (LLVM: the
        # result is f16, the high bit selects the attribute half only). The plain form writes the low half; with
        # op_sel 0x8 (ACO's v_interp_p2_hi_f16) the high half. interp_f16_scan refuses every consumer of a half the
        # pair has not written, so a half that is never written is never observed.
        attr, field, high = self.interp_f16('v_interp_p2_f16', ops, mods, 1, 4)
        self.interp_f16_pair('v_interp_p2_f16', ops)
        self.interp_f16_scan()
        if (self.interp_p1 or {}).get(int(ops[3][1:])) != (attr, field, high):
            raise Unsupported('v_interp_p2_f16 source %s is not the v_interp_p1ll_f16 result of %s' % (ops[3], ops[2]))
        d = reg_range(ops[0])[1]
        self.used_v.add(d)
        if INTERP_DST_HIGH in mods:
            self.set_v(d, '(v%d & 0xffffu) | (f32tof16(i.%s) << 16)' % (d, field))
        else:
            self.set_v(d, '(v%d & 0xffff0000u) | f32tof16(i.%s)' % (d, field))

    def note_attr(self, tok, how):
        m = re.fullmatch(r'attr(\d+)\.([xyzw])', tok)
        attr, component = int(m.group(1)), m.group(2)
        links = getattr(getattr(self, 'pipe', None), 'ps_attr_map', None)  # unit probes may have no pipe
        if links is not None:  # stream pipelines: the vertex parameter the attribute is linked to
            if attr not in links:
                raise Unsupported('pixel stage reads attribute %d, which its interface table does not list' % attr)
            parameter, mask = links[attr]
            if not mask & (1 << 'xyzw'.index(component)):
                raise Unsupported('pixel stage reads undeclared component %s of attribute %d' % (component, attr))
            attr = parameter
        self.attrs.setdefault(attr, set()).add(how)
        return attr, component

    def mix(self, ops, mods, dest):
        """v_fma_mix_f32 / v_fma_mix{lo,hi}_f16: fma of sources that are binary32 words or binary16 halves.

        Per source (LLVM 15 SelectVOP3PMadMixModsImpl): op_sel_hi set means the source is binary16, and
        op_sel set picks its high half. abs, then neg, apply after widening. dest is 'f32', 'lo' or 'hi'.
        """
        half, hi, clamp = [0, 0, 0], [0, 0, 0], False
        for m in mods:
            if m.startswith('op_sel_hi:['):
                half = [int(x) for x in m[11:-1].split(',')]
            elif m.startswith('op_sel:['):
                hi = [int(x) for x in m[8:-1].split(',')]
            elif m == 'clamp':
                clamp = True
            else:
                raise Unsupported('fma_mix modifier %s' % m)

        def operand(tok, is_half, is_hi):
            if not is_half and is_hi:
                raise Unsupported('op_sel on a binary32 fma_mix source %s' % tok)
            if tok == '-0':
                # neg on inline 0 is -0.0 in either format; f() would print +0.0, which changes a -0 product.
                return '(-0.0)'
            if not is_half:
                if tok.startswith('--') and tok[1:] in FLOATS:
                    # neg on a negative inline float prints as `--0.5`; `(--0.5)` is an HLSL decrement.
                    return '(-(%s))' % self.f(tok[1:])
                if re.fullmatch(r'-(\d+|0x[0-9a-fA-F]+)', tok):
                    # The decoder prints neg on inline 1..64 or a literal as a negative integer, which f()
                    # reads without the sign-bit flip. Refuse until the decoder resolves it from the encoding.
                    raise Unsupported('negated integer or literal fma_mix source %s' % tok)
                return self.f(tok)
            neg = tok.startswith('-') and not re.match(r'-[\d.]', tok)
            if neg:
                tok = tok[1:]
            absolute = tok.startswith('|') and tok.endswith('|')
            if absolute:
                tok = tok[1:-1]
            r = reg_range(tok)
            if tok == '0':
                text = '0.0'
            elif re.fullmatch(r'0x[0-9a-fA-F]{1,8}', tok):
                word = (int(tok, 16) >> (16 if is_hi else 0)) & 0xffff
                text = 'f16tof32(0x%04xu)' % word
            elif tok == '0.5' and not is_hi:
                text = '0.5'
            elif r and r[1] == r[2]:
                # Only a high-half read needs the upper 16 bits defined.
                text = 'f16tof32(%s%s)' % (self.u(tok, low16_only=not is_hi), ' >> 16' if is_hi else '')
            else:
                raise Unsupported('binary16 fma_mix source %s' % tok)
            if absolute:
                text = 'abs(%s)' % text
            return '(-%s)' % text if neg else text

        # v_fma_mix* is fused, but HLSL may round `a * b` before the add, so binary32 sources with long
        # mantissas can differ from the hardware by 1 ULP (same as v_fma_f32 in VEC_F). mix32 is a precise
        # mad: without it an RDNA 1 and an RDNA 4 driver drop the sign of a -0 product (#2729, #2746).
        expr = 'mix32(%s, %s, %s)' % tuple(operand(*a) for a in zip(ops[1:4], half, hi))
        if clamp:
            expr = 'saturate(%s)' % expr
        d = reg_range(ops[0])[1]
        if dest == 'f32':
            self.set_v(d, 'U(%s)' % expr)
            return
        self.used_v.add(d)
        if dest == 'hi':
            self.set_v(d, '(v%d & 0xffffu) | (f32tof16(%s) << 16)' % (d, expr))
        else:
            keep_low16 = d in self.low16_only_v
            self.set_v(d, '(v%d & 0xffff0000u) | f32tof16(%s)' % (d, expr), low16_only=keep_low16)

    def op_v_fma_mix_f32(self, ops, mods):
        self.mix(ops, mods, 'f32')

    def op_v_fma_mixlo_f16(self, ops, mods):
        self.mix(ops, mods, 'lo')

    def op_v_fma_mixhi_f16(self, ops, mods):
        self.mix(ops, mods, 'hi')

    def packed_half(self, base, arity, text, ops, mods):
        """v_pk_{mul,add,fma,min,max}_f16: the same binary16 operation on two lanes of a VGPR pair of halves.

        Result low half = op(source i half op_sel[i], ...); high half = op(source i half op_sel_hi[i], ...). Absent
        modifiers mean op_sel [0,0,0] and op_sel_hi [1,1,1] (LLVM VOP3P defaults). Both lanes are written (no zeroing).
        The binary operations support neg_lo and neg_hi after half selection. Clamp and other modifiers are refused.
        Each source must be a plain VGPR, or a literal or inline float constant read through its low half in both lanes.
        """
        if len(ops) != arity + 1:
            raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
        sel = {'op_sel': [0] * arity, 'op_sel_hi': [1] * arity}
        if arity == 2:
            sel.update(neg_lo=[0] * arity, neg_hi=[0] * arity)
        for m in mods:
            key, _, value = m.partition(':')
            if key not in sel or not (value.startswith('[') and value.endswith(']')):
                raise Unsupported('%s with %s' % (base, m))
            bits = value[1:-1].split(',')
            if len(bits) != arity or any(b not in ('0', '1') for b in bits):
                raise Unsupported('%s with %s' % (base, m))
            sel[key] = [int(b) for b in bits]
        literal = {}   # source index -> its 32-bit literal, read through its LOW half only (see below)
        for i, tok in enumerate(ops):
            r = reg_range(tok)
            if r and r[0] == 'v' and r[1] == r[2] and tok == 'v%d' % r[1]:
                continue
            if i >= 1 and re.fullmatch(r'0x[0-9a-fA-F]{1,8}', tok):
                # A literal is one dword and each lane picks a half of it by op_sel / op_sel_hi. Only a source whose two lanes both
                # read the low half is accepted: that is the plain binary16 constant (LLVM prints `0x7bef op_sel_hi:[1,0]`); a read of
                # the high half of a literal, and an inline constant (how it reaches each half is not proven), stay refused.
                if sel['op_sel'][i - 1] or sel['op_sel_hi'][i - 1]:
                    raise Unsupported('%s literal source %s read through its high half' % (base, tok))
                literal[i - 1] = int(tok, 16) & 0xffff
                continue
            if i >= 1 and tok in PK_INLINE_F16:
                # Inline float constants (0.5, 1.0, 2.0, 4.0 and negatives) are the binary16 value in the low half; accepted on the same
                # terms as a literal: both lanes read the low half. Inline integers and a high-half read stay refused.
                if sel['op_sel'][i - 1] or sel['op_sel_hi'][i - 1]:
                    raise Unsupported('%s inline constant %s read through its high half' % (base, tok))
                literal[i - 1] = PK_INLINE_F16[tok]
                continue
            raise Unsupported('%s operand %s is not a single VGPR' % (base, tok))
        lanes = []
        for pick, neg in (('op_sel', 'neg_lo'), ('op_sel_hi', 'neg_hi')):
            args = []
            for n, (tok, hi, both) in enumerate(zip(ops[1:], sel[pick], zip(sel['op_sel'], sel['op_sel_hi']))):
                if n in literal:
                    half = '0x%xu' % literal[n]
                else:
                    word = self.u(tok, low16_only=not any(both))
                    half = '(%s >> 16) & 0xffffu' % word if hi else '%s & 0xffffu' % word
                if neg in sel and sel[neg][n]:
                    half = '(%s) ^ 0x8000u' % half
                args.append('f16tof32(%s)' % half)
            lanes.append('f16rn(%s)' % text.format(*args))
        d = reg_range(ops[0])[1]
        self.set_v(d, '%s | (%s << 16)' % tuple(lanes))

    def op_v_pk_mul_f16(self, ops, mods):
        self.packed_half('v_pk_mul_f16', 2, 'mul16({0}, {1})', ops, mods)

    def op_v_pk_add_f16(self, ops, mods):
        self.packed_half('v_pk_add_f16', 2, '{0} + {1}', ops, mods)

    def op_v_pk_fma_f16(self, ops, mods):
        self.packed_half('v_pk_fma_f16', 3, 'fma16o({0}, {1}, {2})', ops, mods)

    # v_pk_{min,max}_f16: the lane values are exact binary32 images of binary16, so the binary32 helper that orders -0 below +0 and passes a
    # NaN to D3D min/max (the one v_min_f32 and v_max_f32 use) is exact, and f16rn gives the result back its binary16 word.
    def op_v_pk_min_f16(self, ops, mods):
        self.packed_half('v_pk_min_f16', 2, 'minmax_zero32({0}, {1}, false)', ops, mods)

    def op_v_pk_max_f16(self, ops, mods):
        self.packed_half('v_pk_max_f16', 2, 'minmax_zero32({0}, {1}, true)', ops, mods)

    def cube(self, kind, ops, mods):
        """v_cube{id,sc,tc,ma}_f32: LLVM's ConstantFoldAMDGCNCubeIntrinsic.

        The major axis is z when |z| is at least |x| and |y|, otherwise y when |y| is at
        least |x|, otherwise x. A face is negative only for a nonzero, non-NaN negative
        coordinate. cubema returns twice the signed major coordinate.
        """
        if len(ops) != 4:
            raise Unsupported('v_cube%s_f32 with %d operands, expected 4' % (kind, len(ops)))
        if any(m != 'clamp' for m in mods):
            raise Unsupported('v_cube%s_f32 with %s' % (kind, ', '.join(mods)))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[2] != dst[1] or dst[2] > 255:
            raise Unsupported('v_cube%s_f32 destination %s' % (kind, ops[0]))
        x, y, z = [self.f(o) for o in ops[1:4]]
        temp = 'cube%d' % getattr(self, 'cube_temps', 0)
        self.cube_temps = getattr(self, 'cube_temps', 0) + 1
        self.emit('float3 %s = float3(%s, %s, %s);' % (temp, x, y, z))
        c = temp
        a = 'abs(%s)' % c
        zmajor = '(%s.z >= %s.x && %s.z >= %s.y)' % (a, a, a, a)
        ymajor = '(%s.y >= %s.x)' % (a, a)
        expression = {
            'id': '(%s ? (%s.z < 0.0f ? 5.0f : 4.0f) : %s ? (%s.y < 0.0f ? 3.0f : 2.0f) '
                  ': (%s.x < 0.0f ? 1.0f : 0.0f))' % (zmajor, c, ymajor, c, c),
            'sc': '(%s ? (%s.z < 0.0f ? -%s.x : %s.x) : %s ? %s.x '
                  ': (%s.x < 0.0f ? %s.z : -%s.z))' % (zmajor, c, c, c, ymajor, c, c, c, c),
            'tc': '(%s ? -%s.y : %s ? (%s.y < 0.0f ? -%s.z : %s.z) : -%s.y)' % (
                zmajor, c, ymajor, c, c, c, c),
            'ma': '((%s ? %s.z : %s ? %s.y : %s.x) * 2.0f)' % (zmajor, c, ymajor, c, c),
        }[kind]
        self.set_vf(dst[1], expression, mods)

    def op_v_cubeid_f32(self, ops, mods):
        self.cube('id', ops, mods)

    def op_v_cubesc_f32(self, ops, mods):
        self.cube('sc', ops, mods)

    def op_v_cubetc_f32(self, ops, mods):
        self.cube('tc', ops, mods)

    def op_v_cubema_f32(self, ops, mods):
        self.cube('ma', ops, mods)

    def i64_words(self, tok, signed=False, literal=False):
        """(low, high) uint expressions of a 64-bit integer source: a VGPR/SGPR pair, an inline int or a
        32-bit hex literal when `literal` is set (zero-extended, sign-extended when the operation is signed)."""
        pair = reg_range(tok)
        if pair and pair[2] == pair[1] + 1:
            if pair[0] == 'v' and pair[2] <= 255:
                return self.u('v%d' % pair[1]), self.u('v%d' % pair[2])
            if pair[0] == 's' and pair[1] % 2 == 0 and pair[2] <= 105:
                return self.u('s%d' % pair[1]), self.u('s%d' % pair[2])
            raise Unsupported('64-bit source %s' % tok)
        if re.fullmatch(r'0|-?[1-9]\d*', tok) and -16 <= int(tok) <= 64:
            value = int(tok) & 0xffffffffffffffff
            return '%du' % (value & 0xffffffff), '%du' % (value >> 32)
        if literal and re.fullmatch(r'0x[0-9a-f]{1,8}', tok):
            word = int(tok, 16)
            return '%du' % word, ('4294967295u' if signed and word >> 31 else '0u')
        raise Unsupported('64-bit source %s' % tok)

    def shift64(self, ops, mods, helper):
        if mods:
            raise Unsupported('64-bit shift with %s' % mods)
        if len(ops) != 3:
            raise Unsupported('64-bit shift with %d operands, expected 3' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[2] != dst[1] + 1 or dst[2] > 255:
            raise Unsupported('64-bit shift destination %s' % ops[0])
        sr = reg_range(ops[1])
        if (sr and (sr[1] != sr[2] or (sr[0] == 'v' and sr[1] > 255) or (sr[0] == 's' and sr[1] > 105))
                or ops[1] in ('vcc', 'exec') or ops[1] in FLOATS):
            raise Unsupported('64-bit shift amount %s' % ops[1])
        amount = self.u(ops[1])
        low, high = self.i64_words(ops[2], helper == 'shr64i', True)
        body = '{ uint2 t = %s(%s, %s, %s); v%d = t.x; v%d = t.y; }' % (
            helper, low, high, amount, dst[1], dst[2])
        if hasattr(self, 'uniform_v'):
            self.uniform_v.discard(dst[1])
            self.uniform_v.discard(dst[2])
        if isinstance(self.align4_v, set):
            self.align4_v.discard(dst[1])
            self.align4_v.discard(dst[2])
        self.clear_low16(dst[1])
        self.clear_low16(dst[2])
        self.used_v.update((dst[1], dst[2]))
        self.emit(body if self.exec_known else 'if (exec & 1) ' + body)
        # Scalar or literal sources give every lane the same result, under the conditions Stage.one uses for VALU.
        pair = reg_range(ops[2])
        sources = [ops[1]] + (['%s%d' % (pair[0], pair[1]), '%s%d' % (pair[0], pair[2])] if pair else [])
        if (self.tracks_uniformity() and not any(reg_range(t)[0] == 'v' for t in sources if reg_range(t)) and
                all(self.uniform_source(t) for t in sources) and
                (getattr(self, 'past_entry', False) and hasattr(self, 'uniform_expr') or
                 not getattr(self, 'past_entry', True) and getattr(self, 'exec_known', False))):
            self.uniform_v.update((dst[1], dst[2]))
            if hasattr(self, 'uniform_expr'):
                for n, half in ((dst[1], 'x'), (dst[2], 'y')):
                    self.uniform_expr[n] = '%s(%s, %s, %s).%s' % (helper, low, high, amount, half)

    def carry_out_op(self, ops, mods, kind):
        """v_{add,sub,subrev}_co_u32 D, SDST, S0, S1: a 32-bit add or subtract with a carry or borrow out.

        AMD RDNA 2 ISA: V_ADD_CO_U32 D = S0 + S1 with the carry out of bit 31 written to the lane's bit of SDST;
        V_SUB_CO_U32 D = S0 - S1 with a borrow when S1 > S0; V_SUBREV_CO_U32 swaps S0 and S1. These forms are VOP3b
        only and take no carry-in. The result and the carry go through the scratch registers a and w first, so D or
        SDST may overlap a source.
        """
        name = 'v_%s_co_u32' % kind
        if mods:
            raise Unsupported('%s with %s' % (name, ' '.join(mods)))
        if len(ops) != 4:
            raise Unsupported('%s with %d operands' % (name, len(ops)))
        d = reg_range(ops[0])
        if not d or d[0] != 'v' or d[1] != d[2] or d[1] > 255:
            raise Unsupported('%s destination %s' % (name, ops[0]))
        sdst = reg_range(ops[1])
        if ops[1] not in ('vcc', 'vcc_lo') and not (sdst and sdst[0] == 's' and sdst[1] == sdst[2] and sdst[1] <= 105):
            raise Unsupported('%s carry-out destination %s' % (name, ops[1]))
        a, b = self.u(ops[2]), self.u(ops[3])
        if kind == 'subrev':
            a, b = b, a
        if kind == 'add':
            self.emit('a = %s + %s;' % (a, b))
            self.emit('w = (%s + %s) < %s ? 1u : 0u;' % (a, b, a))
        else:
            self.emit('a = %s - %s;' % (a, b))
            self.emit('w = %s > %s ? 1u : 0u;' % (b, a))
        align4 = self.operand_align4(ops[2]) and self.operand_align4(ops[3])
        self.set_v(d[1], 'a', align4=align4)
        predicate = 'w' if self.exec_known else '(exec & w)'
        if ops[1] in ('vcc', 'vcc_lo'):
            self.emit('vcc = %s;' % predicate)
        else:
            # The carry is a per-lane bit: never a uniform scalar, whatever the scratch name hides.
            self.set_s(sdst[1], predicate, uniform=False)
            self.masks.add(sdst[1])

    def op_v_add_co_u32(self, ops, mods):
        self.carry_out_op(ops, mods, 'add')

    def op_v_sub_co_u32(self, ops, mods):
        self.carry_out_op(ops, mods, 'sub')

    def op_v_subrev_co_u32(self, ops, mods):
        self.carry_out_op(ops, mods, 'subrev')

    def carry_in_out_op(self, ops, mods, kind):
        """v_{add,sub,subrev}_co_ci_u32 D, SDST, S0, S1, S2: 32-bit add/sub with carry or borrow in and out.

        AMD RDNA 2 ISA: V_ADD_CO_CI_U32 D = S0 + S1 + S2[lane] with carry out to SDST[lane];
        V_SUB_CO_CI_U32 D = S0 - S1 - S2[lane] with borrow out to SDST[lane]; V_SUBREV_CO_CI_U32 swaps S0 and S1.
        Both VOP2 (_e32, implicit vcc_lo) and VOP3b (_e64) forms arrive with five operands. Results go through
        scratch a and w first so D or SDST may overlap any source.
        """
        name = 'v_%s_co_ci_u32' % kind
        if mods:
            raise Unsupported('%s with %s' % (name, ' '.join(mods)))
        if len(ops) != 5:
            raise Unsupported('%s with %d operands' % (name, len(ops)))
        d = reg_range(ops[0])
        if not d or d[0] != 'v' or d[1] != d[2] or d[1] > 255:
            raise Unsupported('%s destination %s' % (name, ops[0]))
        sdst = reg_range(ops[1])
        if ops[1] not in ('vcc', 'vcc_lo') and not (sdst and sdst[0] == 's' and sdst[1] == sdst[2] and sdst[1] <= 105):
            raise Unsupported('%s carry-out destination %s' % (name, ops[1]))
        cin_tok = ops[4]
        if cin_tok in ('vcc_hi', 'exec_hi'):
            raise Unsupported('%s carry-in %s' % (name, cin_tok))
        cin_const = self.const_u32(cin_tok)
        if cin_const in (0, 0xffffffff):
            ci = '1u' if cin_const else '0u'
        elif self.is_lane_mask(cin_tok):
            ci = '(%s & 1u)' % self.u(cin_tok)
        else:
            raise Unsupported('%s carry-in %s' % (name, cin_tok))
        a, b = self.u(ops[2]), self.u(ops[3])
        if kind == 'subrev':
            a, b = b, a
        if kind == 'add':
            self.emit('a = %s + %s;' % (a, b))
            self.emit('w = a < %s ? 1u : 0u;' % a)
            self.emit('a = a + %s;' % ci)
            self.emit('w = w | (a < %s ? 1u : 0u);' % ci)
        else:
            self.emit('a = %s - %s;' % (a, b))
            self.emit('w = %s > %s ? 1u : 0u;' % (b, a))
            self.emit('w = w | (%s > a ? 1u : 0u);' % ci)
            self.emit('a = a - %s;' % ci)
        self.set_v(d[1], 'a')
        predicate = 'w' if self.exec_known else '(exec & w)'
        if ops[1] in ('vcc', 'vcc_lo'):
            self.emit('vcc = %s;' % predicate)
        else:
            self.set_s(sdst[1], predicate, uniform=False)
            self.masks.add(sdst[1])

    def op_v_add_co_ci_u32(self, ops, mods):
        self.carry_in_out_op(ops, mods, 'add')

    def op_v_sub_co_ci_u32(self, ops, mods):
        self.carry_in_out_op(ops, mods, 'sub')

    def op_v_subrev_co_ci_u32(self, ops, mods):
        self.carry_in_out_op(ops, mods, 'subrev')

    def op_v_lshlrev_b64(self, ops, mods):
        self.shift64(ops, mods, 'shl64')

    def op_v_lshrrev_b64(self, ops, mods):
        self.shift64(ops, mods, 'shr64u')

    def op_v_ashrrev_i64(self, ops, mods):
        self.shift64(ops, mods, 'shr64i')

    def mad64(self, ops, mods, signed):
        """v_mad_u64_u32 / v_mad_i64_i32: D = S0 * S1 + S2 modulo 2^64, two 32-bit factors, 64-bit S2.

        The unsigned form also writes the carry-out of the 64-bit add to a lane mask. The signed carry-out
        is not established, so the signed form is accepted only with the discarding null destination."""
        name = 'v_mad_i64_i32' if signed else 'v_mad_u64_u32'
        if mods:
            raise Unsupported('%s with %s' % (name, mods))
        if len(ops) != 5:
            raise Unsupported('%s with %d operands, expected 5' % (name, len(ops)))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[2] != dst[1] + 1 or dst[2] > 255:
            raise Unsupported('%s destination %s' % (name, ops[0]))
        mask = ops[1]
        mask_reg = reg_range(mask)
        if signed:
            if mask != 'null':
                raise Unsupported('%s carry-out %s: only null is lowered' % (name, mask))
        elif mask not in ('vcc', 'vcc_lo') and not (
                mask_reg and mask_reg[0] == 's' and mask_reg[1] == mask_reg[2] and mask_reg[1] <= 105):
            raise Unsupported('%s carry-out %s' % (name, mask))
        for tok in ops[2:4]:
            r = reg_range(tok)
            if tok in FLOATS or tok in ('exec', 'exec_lo') or (
                    r and (r[1] != r[2] or (r[0] == 'v' and r[1] > 255) or (r[0] == 's' and r[1] > 105))):
                raise Unsupported('%s factor %s' % (name, tok))
        for tok in ops[2:5]:
            r = reg_range(tok)
            if tok in ('vcc', 'vcc_lo', 'vcc_hi') or (r and r[0] == 's' and any(
                    w in self.masks for w in range(r[1], r[2] + 1))):
                # u() would supply the per-lane 0/1 projection, not the architectural 32-bit mask.
                raise Unsupported('%s reads the lane mask %s as an integer' % (name, tok))
        low, high = self.i64_words(ops[4])
        self.emit('{ uint3 r = %s(%s, %s, %s, %s);' % ('mad64i' if signed else 'mad64u',
                                                         self.u(ops[2]), self.u(ops[3]), low, high))
        self.set_v(dst[1], 'r.x')
        self.set_v(dst[2], 'r.y')
        if not signed:
            expr = 'r.z' if self.exec_known else '(exec & r.z)'
            if mask in ('vcc', 'vcc_lo'):
                self.emit('vcc = %s;' % expr)
            else:
                # The carry is a per-lane bit: never a uniform scalar, whatever the scratch name hides.
                self.set_s(mask_reg[1], expr, uniform=False)
                self.masks.add(mask_reg[1])
        self.emit('}')

    def op_v_mad_u64_u32(self, ops, mods):
        self.mad64(ops, mods, False)

    def op_v_mad_i64_i32(self, ops, mods):
        self.mad64(ops, mods, True)

    def op_v_cvt_pkrtz_f16_f32(self, ops, mods):
        if len(ops) != 3:
            raise Unsupported('v_cvt_pkrtz_f16_f32 with %d operands, expected 3' % len(ops))
        self.set_v(reg_range(ops[0])[1], 'f32tof16(%s) | (f32tof16(%s) << 16)' % (self.f(ops[1]), self.f(ops[2])))

    # VOPC with sub-dword addressing (tools/disasm_rdna2.py _decode_vopc_sdwa prints it): each source
    # reads its src0_sel/src1_sel field, zero-extended or, under sext(), sign-extended, and the
    # compare then reads that 32-bit value as its type. A binary32 class compare reads the whole
    # float in SRC0 and selects an integer field in SRC1 for the mask.
    #
    # An f32 source (#1906) with a DWORD selection reads the register as binary32 with neg/abs as in
    # the e64 form. A BYTE or WORD selection is refused: LLVM's SIPeepholeSDWA models it as the field
    # zero-extended and read as binary32 bits (the RDNA 2 ISA guide gives SDWAB only its fields), so
    # the field is always +0 or a denormal and the result depends on the binary32 denormal mode, which
    # this translator does not read (pixel_aot refuses the same forms). sext() is valid for integer
    # operands only (LLVM AMDGPUModifierSyntax; LLVM 15 does not decode the bit on a float source).
    def compare_sdwa_sources(self, base, typ, ops, mods, *, float_src0=False, float_sources=False):
        fields = {}
        for m in mods:
            key, _, value = m.partition(':')
            if key not in ('src0_sel', 'src1_sel') or value not in SDWA_SEL or key in fields:
                raise Unsupported('%s with SDWA modifier %s' % (base, m))
            fields[key] = value
        sources = []
        for index, token in enumerate(ops):
            selection = fields.get('src%d_sel' % index)
            if not selection:
                raise Unsupported('%s SDWA without src%d_sel' % (base, index))
            sext = token.startswith('sext(') and token.endswith(')')
            if float_src0 and index == 0:
                if selection != 'DWORD' or sext:
                    raise Unsupported('%s requires a DWORD float source' % base)
                sources.append(self.f(token))
                continue
            if float_sources:
                if 'sext(' in token:
                    raise Unsupported('%s: sext() on a float source' % base)
                if selection != 'DWORD':
                    raise Unsupported('%s: f32 SDWA byte/word selection depends on the binary32 denormal mode'
                                      % base)
                # The whole word: the same text as the e64 form with the same modifiers.
                sources.append(self.f(token))
                continue
            if sext:
                token = token[len('sext('):-1]
            word = sdwa_source(self.u(token, low16_only=selection in ('BYTE_0', 'BYTE_1', 'WORD_0')), selection, sext)
            sources.append('int(%s)' % word if typ == 'i32' else word)
        return sources

    def check_compare_sdwa_destination(self, base, writes_exec, ops):
        # v_cmpx_*_sdwa prints just its sources. v_cmp_*_sdwa prints the scalar destination first.
        if len(ops) != (2 if writes_exec else 3):
            raise Unsupported('%s with %d operands' % (base, len(ops)))
        if not writes_exec and ops[0] not in ('vcc', 'vcc_lo', 'exec', 'exec_lo') and not (
                reg_range(ops[0]) and reg_range(ops[0])[0] == 's'):
            raise Unsupported('%s into %s' % (base, ops[0]))

    def _low16_source(self, base, tok, signed):
        """A source of a 16-bit integer compare: bits 15:0 of the operand, sign- or zero-extended (RDNA 2 VOPC
        S0.i16 / S0.u16). A float token is refused: on a 16-bit operand the inline constants 240-248 are binary16
        values (LLVM 15 prints inline 1.0 there as 0x3c00), while u() would give the binary32 word."""
        if tok in FLOATS:
            raise Unsupported('%s source %s: inline float constants are binary16 on 16-bit operands' % (base, tok))
        reg = reg_range(tok)
        if reg:
            if reg[1] != reg[2]:
                raise Unsupported('%s source %s' % (base, tok))
            x = self.u(tok, low16_only=True)
            return '(asint(%s << 16u) >> 16u)' % x if signed else '(%s & 0xffffu)' % x
        try:
            value = int(tok, 0)
        except ValueError:
            # vcc, exec, m0 and modifier spellings: not a source this lowering reads.
            raise Unsupported('%s source %s' % (base, tok))
        # The decoder prints inline integers -16..64 in decimal and a literal in hex; LLVM 15 accepts only 16-bit
        # literals on these operands.
        if not -16 <= value <= 0xffff:
            raise Unsupported('%s source %s: wider than 16 bits' % (base, tok))
        low = value & 0xffff
        return '%d' % (low - 0x10000 if low & 0x8000 else low) if signed else '0x%04xu' % low

    def compare(self, base, ops, mods=()):
        # A comparison reads its two operands directly, so a modifier that arrives here is one this
        # lowering does not model (op_sel selects a packed half, clamp/omod have no place in a compare):
        # refuse it rather than emit a predicate computed from the unselected operands.
        if mods and not base.endswith('_sdwa'):
            raise Unsupported('%s with %s' % (base, mods))
        # v_cmp_class_f32 / v_cmpx_class_f32 test the binary32 class of the value in SRC0 against the
        # class-bit mask in SRC1 (the ISA order: the float operand first, the integer mask second).
        name = base.removesuffix('_sdwa')
        if name in ('v_cmp_class_f32', 'v_cmpx_class_f32', 'v_cmp_class_f16', 'v_cmpx_class_f16'):
            # The half form (issue #1855) takes the raw binary16 pattern in the low half of the operand
            # word: its classes follow the half layout (exponent 5 bits, mantissa 10, quiet bit 0x0200)
            # and abs/neg act on bit 15, so the value is never converted to float here.
            f16 = name.endswith('_f16')
            writes_exec = name in ('v_cmpx_class_f32', 'v_cmpx_class_f16')
            sdwa = name != base
            if sdwa:
                self.check_compare_sdwa_destination(base, writes_exec, ops)
            if len(ops) == 3:
                dst, value, mask = ops
            elif writes_exec and len(ops) == 2:
                # The e32 and SDWA spellings write EXEC implicitly and print no destination.
                dst, (value, mask) = 'exec', ops
            else:
                raise Unsupported('%s with %d operands' % (base, len(ops)))
            if sdwa and f16:
                # SDWA half class: classes bits [15:0] of a plain register. src0_sel DWORD and WORD_0 both
                # leave those bits where class16 reads them; WORD_1, a byte, sext() and the abs/neg spellings
                # (which print |v|, -v, abs(), neg()) would classify other bits, so they are refused.
                selections = [m for m in mods if m.startswith('src0_sel:')]
                if selections not in (['src0_sel:DWORD'], ['src0_sel:WORD_0']):
                    raise Unsupported('SDWA form of %s with %s' % (name, ' '.join(selections) or 'no src0_sel'))
                if not reg_range(value):
                    raise Unsupported('%s with source %s' % (base, value))
                if mask in FLOATS:
                    raise Unsupported('%s mask %s: expected an integer class mask' % (base, mask))
                _word, mask = self.compare_sdwa_sources(base, 'u32', [value, mask], mods)
                value = self.u(value)
            elif sdwa:
                value, mask = self.compare_sdwa_sources(base, 'u32', [value, mask], mods, float_src0=True)
            elif f16:
                # abs() and - apply to bit 15 of the half, before the class test.
                # The decoder prints an absolute source as |v4| and a negated one as -v4, so both
                # spellings are accepted here as well as abs(...).
                token = value
                if re.fullmatch(r'-[1-9]\d*', token):
                    raise Unsupported('%s source %s: ambiguous signed inline half' % (base, value))
                negative = token.startswith('-')
                if negative:
                    token = token[1:]
                absolute = (token.startswith('|') and token.endswith('|')) or (
                    token.startswith('abs(') and token.endswith(')'))
                if absolute:
                    token = token[4:-1] if token.startswith('abs(') else token[1:-1]
                if token in FLOATS or re.fullmatch(r'-[1-9]\d*', token):
                    # A float-printed operand would otherwise be lowered as its binary32 bit pattern.
                    raise Unsupported('%s source %s: f16 class compares need the raw half' % (base, value))
                bits = self.u(token, low16_only=True)
                if absolute:
                    bits = '(%s & 0x7fffu)' % bits
                if negative:
                    bits = '(%s ^ 0x8000u)' % bits
                if mask in FLOATS:
                    raise Unsupported('%s mask %s: expected an integer class mask' % (base, mask))
                value, mask = bits, self.u(mask)
            else:
                value, mask = self.f(value), self.u(mask)
            cond = '(%s(%s) & %s) != 0u' % ('class16' if f16 else 'class32', value, mask)
            predicate = '((%s) ? 1u : 0u)' % cond
            # A narrowed EXEC must stay narrowed: intersect with it, as the generic compare path does.
            if not self.exec_known:
                predicate = '(exec & %s)' % predicate
            if writes_exec or dst in ('exec', 'exec_lo'):
                self.exec_known = False
                self.emit('exec = %s;' % predicate)
            elif dst in ('vcc', 'vcc_lo'):
                self.emit('vcc = %s;' % predicate)
            else:
                reg = reg_range(dst)
                if not reg or reg[0] != 's':
                    raise Unsupported('%s into %s' % (base, dst))
                d = reg[1]
                self.set_s(d, predicate)
                self.masks.add(d)
            return None

        m = re.fullmatch(r'v_cmp(x?)_(\w+?)_(f32|u32|i32|i16|u16)(_sdwa)?', base)
        if not m:
            raise Unsupported(base)
        writes_exec, cond, typ, sdwa = m.group(1) == 'x', m.group(2), m.group(3), bool(m.group(4))
        # The 16-bit integer compares have the six relations only (gfx10 has no 16-bit f or t). Their SDWA forms
        # are not lowered here (tools/pixel_aot.py has its own lowering of them).
        if typ in ('i16', 'u16') and (sdwa or cond not in ('lt', 'eq', 'le', 'gt', 'ne', 'ge')):
            raise Unsupported(base)
        # f32 lg is ordered and not equal (false when either source is NaN) and nlg its negation (#1920).
        # HLSL != is unordered, so lg reads as less or greater, behind {2}: both source words not NaN,
        # tested on their bits. DXC marks float compares fast unless -Gis, and a driver may then fold
        # less-or-greater into an unordered not-equal.
        # FXC without /Gis compiles !(a < b) to a >= b, false on NaN, and DXC without -Gis folds x != x
        # and x == x to constants (#1936). So nlt, nle, ngt and nge negate an ordered relation behind {2},
        # and o and u read {2} alone; the startup cache compiles with FXC and no flags.
        table = {'lt': '{0} < {1}', 'eq': '{0} == {1}', 'le': '{0} <= {1}', 'gt': '{0} > {1}',
                 'lg': '({2} && ({0} < {1} || {0} > {1}))', 'ne': '{0} != {1}',
                 'ge': '{0} >= {1}', 'neq': '!({0} == {1})', 'nlt': '!({2} && {0} < {1})',
                 'nle': '!({2} && {0} <= {1})', 'ngt': '!({2} && {0} > {1})', 'nge': '!({2} && {0} >= {1})',
                 'nlg': '!({2} && ({0} < {1} || {0} > {1}))', 'o': '({2})', 'u': '!({2})',
                 't': 'true', 'f': 'false'}
        # Conditions outside the table or dedicated lowering stay refused in both forms.
        if cond not in table:
            raise Unsupported(base)
        if sdwa:
            self.check_compare_sdwa_destination(base, writes_exec, ops)
        if len(ops) == 3:
            dst, a, b = ops
        else:
            dst, (a, b) = ('exec' if writes_exec else 'vcc'), ops
        if sdwa:
            # f32 sources: DWORD selections only, read as in the e64 form, so {2} tests the registers.
            x, y = self.compare_sdwa_sources(base, typ, [a, b], mods, float_sources=typ == 'f32')
        elif typ == 'f32':
            x, y = self.f(a), self.f(b)
        elif typ == 'i32':
            x, y = 'int(%s)' % self.u(a), 'int(%s)' % self.u(b)
        elif typ in ('i16', 'u16'):
            x, y = (self._low16_source(base, t, typ == 'i16') for t in (a, b))
        else:
            x, y = self.u(a), self.u(b)
        def word(tok):
            # A float source's word before its neg/abs modifiers, as f() reads them; they keep a NaN a NaN.
            tok = self.split_neg(tok)[1]
            if tok.startswith('|') and tok.endswith('|'):
                tok = tok[1:-1]
            return self.u(tok)
        ordered = ''
        if '{2}' in table[cond]:
            ordered = ' && '.join('(%s & 0x7fffffffu) <= 0x7f800000u' % word(t) for t in (a, b))
        expr = '((%s) ? 1u : 0u)' % table[cond].format(x, y, ordered)
        if not self.exec_known:
            expr = '(exec & %s)' % expr
        if writes_exec or dst in ('exec', 'exec_lo'):
            self.exec_known = False
            self.emit('exec = %s;' % expr)
        elif dst in ('vcc', 'vcc_lo'):
            self.emit('vcc = %s;' % expr)
        else:
            # vcc_hi, exec_hi, m0 or a literal are not an SGPR the one-lane model holds a mask in; reg_range
            # returned None for them and the index raised TypeError, which ended the whole translation.
            reg = reg_range(dst)
            if not reg or reg[0] != 's':
                raise Unsupported('%s into %s' % (base, dst))
            d = reg[1]
            self.set_s(d, expr)
            self.masks.add(d)

    def signed_pack_export(self, token):
        """Identify signed normalized words on all paths to this compressed export."""
        reg = reg_range(token)
        entry_exec = getattr(self, 'signed_pack_entry_exec', None)
        if not reg or entry_exec is None:
            return False
        bounds, block_of, succ, _ = self.spill_cfg
        pred = [[] for _ in bounds]
        for b, edges in enumerate(succ):
            for target in edges:
                pred[target].append(b)
        index = self.descriptor_instruction_index
        pending = [(block_of[index], index, reg[1])]
        seen, forms = set(), set()
        # A masked VGPR write can leave the previous value live in inactive lanes.
        while pending:
            item = pending.pop()
            if item in seen:
                continue
            seen.add(item)
            block, stop, word = item
            start = bounds[block][0]
            for i in range(stop - 1, start - 1, -1):
                off, _, name, asm = self.ins[i]
                _, ops, mods = parse(asm)
                if word not in lds_vgpr_writes(name, ops):
                    continue
                if name in ('v_mov_b32_e32', 'v_mov_b32_e64') and not mods:
                    src = reg_range(ops[1])
                    if src and src[0] == 'v' and src[1] == src[2]:
                        pending.append((block, i, src[1]))
                        if off in entry_exec:
                            break
                        continue
                forms.add(name == 'v_cvt_pknorm_i16_f32')
                partial = name.startswith(('v_movrel', 'v_writelane')) or name.endswith('_sdwa') or 'd16' in name
                if off in entry_exec and not partial:
                    break
            else:
                if block == 0 or not pred[block]:
                    forms.add(False)
                pending.extend((b, bounds[b][1], word) for b in pred[block])
        if forms == {True, False}:
            raise Unsupported('compressed export mixes signed normalized and other reaching values')
        return True in forms

    def export(self, asm):
        target, vals, flags = export_operands(asm)
        if getattr(self, 'kind', None) == 'ms' and not getattr(self, 'in_mesh_export', False):
            return self.mesh_export(asm, target, vals, flags)
        if any(f.startswith('en:') for f in flags):
            raise Unsupported('compressed export enable mask %s does not select whole channel pairs'
                              % next(f for f in flags if f.startswith('en:'))[3:])
        if 'vm' in flags and not self.exec_known:
            self.emit('if (!(exec & 1)) discard;')
            if hasattr(self, 'lane_exec_breaks'):
                self.lane_exec_breaks.append((self.pc, 'a discard'))
        if target == 'null':
            return None  # the null target writes no render target; its only effect is the vm discard above
        if target == 'mrtz':
            plain = (self.kind == 'ps' and 'depth' not in self.exports and 'compr' not in flags and len(vals) == 4
                     and re.fullmatch(r'v\d+', vals[0]) is not None and vals[1:] == ['off', 'off', 'off'])
            straight = not (self.has_branches or self.blocks) and self.exec_known
            deferred_vm = False
            if plain and not straight and 'vm' not in flags and 'done' not in flags:
                deferred_vm = (sum(name == 'exp' and asm.startswith('exp mrtz ') for _, _, name, asm in self.ins) == 1
                               and mrtz_followed_by_vm(self.ins, self.pc)
                               and mrtz_covers_every_path(self.ins, self.pc, deferred_vm=True))
            if not plain or not (straight or deferred_vm
                                 or ('vm' in flags and mrtz_covers_every_path(self.ins, self.pc))):
                raise Unsupported('MRTZ requires one uncompressed depth export in a straight-line pixel shader with entry EXEC')
            self.exports['depth'] = True
            guard = 'if (exec & 1) ' if deferred_vm and not self.exec_known else ''
            self.emit('%so.depth = %s;' % (guard, self.f(vals[0])))
            return None
        name = {'pos0': 'pos'}.get(target, target)
        if not re.fullmatch(r'pos|param\d+|mrt[0-7]', name):
            raise Unsupported('export target %s' % target)
        kind = self.pipe.mrt_kind(int(name[3:])) if name.startswith('mrt') else 'float'
        if len(vals) != 4:
            raise Unsupported('export with %d operands' % len(vals))
        # The channels the export writes. An 'off' channel is not written, so a render target keeps
        # what it held there; the pipeline's write mask carries that (manifest 'wmask', #1674).
        mask = sum(1 << c for c, v in enumerate(vals) if v != 'off')
        if 'compr' in flags:
            if name.startswith('param'):
                self.compr_params.add(int(name[5:]))
            if kind in ('uint', 'int'):
                raise Unsupported('packed integer export %s' % name)
            # Two binary16 channels per register: the low pair (0, 1) from the first, the high pair
            # (2, 3) from the second, each printed twice or as 'off', 'off'.
            if vals[1] != vals[0] or vals[3] != vals[2]:
                raise Unsupported('compressed export operands %s' % ', '.join(vals))
            if mask != 0xF and not name.startswith('mrt'):
                raise Unsupported('partial compressed export to %s' % name)
            pairs = []
            # R32_FLOAT uses R only. G cannot contribute to R through blending, and A
            # is handled separately. Require the explicit format of the primary target.
            red_only = name == 'mrt0' and tuple(self.pipe.rtv_formats[:1]) == (41,)  # DXGI_FORMAT_R32_FLOAT
            for c in (0, 2):
                v = vals[c]
                r = self.u(v, low16_only=red_only and c == 0) if v != 'off' else None
                if r and name.startswith('mrt') and self.signed_pack_export(v):
                    slot = int(name[3:])
                    formats = self.pipe.rtv_formats
                    fmt = formats[slot] if slot < len(formats) else 0
                    if fmt in (11, 35, 56):  # 16-bit UNORM, RGBA / RG / R
                        pairs += ['(float((%s) & 0xffffu) / 65535.0)' % r,
                                  '(float((%s) >> 16u) / 65535.0)' % r]
                        continue
                    if fmt in (13, 37, 58):  # 16-bit SNORM, RGBA / RG / R
                        pairs += ['max(-1.0, float(asint((%s) << 16u) >> 16) / 32767.0)' % r,
                                  'max(-1.0, float(asint(%s) >> 16) / 32767.0)' % r]
                        continue
                    if fmt not in (10, 34, 54):  # 16-bit FLOAT, RGBA / RG / R
                        raise Unsupported('signed normalized compressed export to %s with RTV format %d' % (name, fmt))
                pairs += ['f16tof32(%s)' % r, 'f16tof32(%s >> 16)' % r] if r else ['0.0', '0.0']
            exprs = pairs
            words = ['0u' if e == '0.0' else 'U(%s)' % e for e in exprs]
        elif kind in ('uint', 'int'):
            exprs = ['0u' if v == 'off' else self.u(v) for v in vals]
            words = exprs
        elif kind == 'bits':
            ud = self.pipe.ud(('mrtbits',))
            exprs = ['0.0' if v == 'off' else 'F((%s) ^ ud%d)' % (self.u(v), ud) for v in vals]
            words = None
        else:
            exprs = ['0.0' if v == 'off' else 'F(%s)' % self.u(v) for v in vals]
            words = ['0u' if v == 'off' else self.u(v) for v in vals]
            if name.startswith('param'):
                self.plain_params[int(name[5:])] = {c for c, v in enumerate(vals) if v == 'off'}
        if name.startswith('mrt'):
            # One pipeline write mask per target: an export of no channel, or two exports of one target
            # with different channels, cannot be represented and is refused.
            if not mask:
                raise Unsupported('export to %s writes no channel' % name)
            known = self.write_masks.setdefault(name, mask)
            if known != mask:
                raise Unsupported('exports to %s write channel masks 0x%x and 0x%x' % (name, known, mask))
        self.exports[name] = True
        expr = '%s(%s)' % ('uint4' if kind in ('uint', 'int') else 'float4', ', '.join(exprs))
        self.emit('o.%s = %s;' % (name, 'asint(%s)' % expr if kind == 'int' else expr))
        if name.startswith('param'):
            # The flat copy read by v_interp_mov_f32 carries the register words. A float copy would let
            # FXC fold asfloat() of a constant and flush integer bit patterns that are binary32
            # denormals (#1259). hlsl_vs() drops the line for attributes the pixel stage reads only
            # interpolated.
            self.emit('o.f%s = uint4(%s);' % (name, ', '.join(words)))
            if 'compr' not in flags and kind not in ('uint', 'int', 'bits'):
                # The binary16 halves of each word for v_interp_p1ll_f16 / v_interp_p2_f16 (see interp_f16);
                # interpolated like param, and dropped by hlsl_vs() unless the pixel stage reads them.
                for half, shift in (('lo', ''), ('hi', ' >> 16')):
                    self.emit('o.h%s_%s = float4(%s);' % (name, half, ', '.join(
                        '0.0' if v == 'off' else 'f16tof32(%s%s)' % (self.u(v), shift) for v in vals)))
        return None

    # ---- output ------------------------------------------------------------------------------
    def fold_execz_counted_loops(self):
        loops = getattr(self, 'execz_counted_loops', {})
        if not loops:
            return self.blocks
        # Only the guard can enter this region from outside. Keep the surrounding dispatch
        # and replace its three private blocks with a structured loop on SCC.
        blocks = {off: (lines, term) for off, lines, term in self.blocks}
        guard_blocks = {term[2]: off for off, (_, term) in blocks.items()
                        if term and term[:2] == ('branch', '(exec & 1) == 0')}
        for latch, (head, body, end) in loops.items():
            start = guard_blocks[latch]
            before, test_exec = blocks[start]
            compare, test_scc = blocks.pop(head)
            iteration, _ = blocks.pop(body)
            blocks.pop(latch)
            lines = [*before, 'if (!(%s)) {' % test_exec[1], '    [loop] while (true) {']
            lines.extend('        ' + line for line in compare)
            lines.append('        if (%s) break;' % test_scc[1])
            lines.extend('        ' + line for line in iteration)
            lines.extend(('    }', '}'))
            blocks[start] = lines, ('jump', end)
        return [(off, lines, term) for off, (lines, term) in blocks.items()]

    def exit_exports(self, goto):
        """Track whether each reachable exit has run an output assignment."""
        if self.kind not in ('vs', 'ps') or sum(
                bool(term and term[0] == 'end') for _, _, term in self.blocks) < 2:
            return set(), set(), set()
        if self.kind == 'ps' and not self.exports:
            return set(), set(), set()
        prefix = ('o.pos =',) if self.kind == 'vs' else ('o.mrt', 'o.depth =')
        writes = {i for i, (_, lines, _) in enumerate(self.blocks)
                  if any(line.lstrip().startswith(prefix) for line in lines)}
        with_export, without_export, seen = set(), set(), set()
        pending = [(0, False)]
        while pending:
            i, exported = pending.pop()
            if (i, exported) in seen or i == len(self.blocks):
                continue
            seen.add((i, exported))
            exported |= i in writes
            term = self.blocks[i][2]
            if term and term[0] == 'end':
                (with_export if exported else without_export).add(i)
                continue
            if term:
                pending.append((goto(term[-1]), exported))
            if not term or term[0] != 'jump':
                pending.append((i + 1, exported))
        if self.kind == 'vs' and without_export:
            raise Unsupported('early exit without exports in a vertex stage')
        if self.kind == 'ps' and kill_exit_block(self.ins):
            # This exit already discards through its null export's valid mask.
            without_export.discard(len(self.blocks) - 1)
        return without_export - with_export, without_export & with_export, writes

    def body(self):
        """Statements of the whole program as structured HLSL."""
        out = []
        blocks = self.fold_execz_counted_loops()
        offsets = [b[0] for b in blocks]
        index = {off: i for i, off in enumerate(offsets)}
        loops = any(t and t[0] in ('jump', 'branch') and t[-1] <= b[0] for b in blocks for t in [b[2]])
        compute_dpp_blocks = getattr(self, 'compute_dpp_blocks', ())
        if any(off not in index for off in compute_dpp_blocks):
            raise Unsupported('compute DPP exchange lost its converged block')
        if len(blocks) == 1:
            return ['    ' + line for line in blocks[0][1]]

        def goto(off):
            if off not in index:
                raise Unsupported('branch into the middle of a block (+0x%x)' % off)
            return index[off]
        discard, conditional_discard, export_blocks = self.exit_exports(goto)
        out.append('    uint pc = 0;')
        if conditional_discard:
            out.append('    bool wrote_output = false;')
        wave_load_blocks = {off + n for off, n, _, _ in self.ins if off in getattr(self, 'mesh_wave_loads', {})}
        if loops:
            out.append('    [loop] for (uint guard = 0; guard < 65536 && pc != 0xffffu; ++guard) {')
        for i, (off, lines, term) in enumerate(blocks):
            if off in getattr(self, 'mesh_collect_reads', {}):
                value = self.mesh_collect_reads[off]
                out.append('    uint mesh_first_%d = mesh_first_active(tid, pc == %d && (exec & 1u) != 0u, '
                           'pc == %d ? %s : 0u, false);' % (i, i, i, value))
            if off in getattr(self, 'mesh_collect_masks', {}):
                selected = self.mesh_collect_masks[off]
                out.append('    uint mesh_mask_%d = mesh_ballot(tid, pc == %d ? (%s) : 0u);' % (i, i, selected))
            if (off == getattr(self, 'mesh_allocation_block', None) or
                    off in compute_dpp_blocks):
                out.append('    {')
            else:
                attribute = '[branch] ' if off in wave_load_blocks else ''
                out.append('    %s%sif (pc == %d) {' % ('else ' if loops and i else '', attribute, i))
            if off in getattr(self, 'mesh_collect_reads', {}):
                out.append('        uint mesh_first = mesh_first_%d;' % i)
            if off in getattr(self, 'mesh_collect_masks', {}):
                out.append('        uint mesh_mask = mesh_mask_%d;' % i)
            out += ['        ' + line for line in lines]
            if conditional_discard and i in export_blocks:
                out.append('        wrote_output = true;')
            if off in getattr(self, 'mesh_collect_votes', {}):
                word = self.mesh_collect_votes[off]
                out += ['    }',
                        '    uint mesh_vote_%d = mesh_first_active(tid, pc == %d && (%s & 1u) != 0u, 0u, true);'
                        % (i, i, word),
                        '    if (pc == %d) {' % i,
                        '        uint mesh_vote = mesh_vote_%d;' % i]
            nxt = i + 1 if i + 1 < len(blocks) else 0xffff
            if not term:
                out.append('        pc = %d;' % nxt)
            elif term[0] == 'end':
                if i in discard:
                    out.append('        discard;')
                elif i in conditional_discard:
                    out.append('        if (!wrote_output) discard;')
                out.append('        pc = 0xffffu;')
            elif term[0] == 'jump':
                out.append('        pc = %d;' % goto(term[1]))
            else:
                out.append('        pc = (%s) ? %d : %d;' % (term[1], goto(term[2]), nxt))
            out.append('    }')
        if loops:
            out.append('    }')
        return out

    def declarations(self):
        lines = []
        if self.used_s:
            lines.append('    uint %s;' % ', '.join('s%d = 0' % n for n in sorted(self.used_s)))
        if self.used_v:
            lines.append('    uint %s;' % ', '.join('v%d = 0' % n for n in sorted(self.used_v)))
        packed = self.pipe.packed_position_vgpr
        if self.kind == 'ps' and packed in self.used_v:
            lines.append('    v%d = (uint(i.pos.x) & 0xffffu) | ((uint(i.pos.y) & 0xffffu) << 16);' % packed)
        if getattr(self, 'spill_slots', None):
            lines.append('    uint %s;' % ', '.join('sp%d_%d = 0' % k for k in sorted(self.spill_slots)))
        if getattr(self, 'uniform_load_temps', None):
            lines.append('    uint %s;' % ', '.join('%s = 0' % n for n in sorted(self.uniform_load_temps)))
        lines.append('    uint vcc = 0, exec = 1, m0 = 0, a = 0, w = 0; bool scc = false; float4 t = 0;')
        if any(bound is not None for _, bound, _, _ in getattr(self, 'mesh_wave_loads', {}).values()):
            lines.append('    bool wave_any = false;')
        if getattr(self, 'used_atomic', None):
            if 'uint' in self.used_atomic:
                lines.append('    uint ao = 0;')
            if 'int' in self.used_atomic:
                lines.append('    int ai = 0;')
        lines += ['    s%d = %s;' % kv for kv in sorted(getattr(self, 'materialized', {}).items())]
        return lines

    # ---- whole-wave reductions (#2125 item D) --------------------------------------------------------------------------
    # A wave32 reduction reads lanes outside the pixel quad, so its instructions cannot be lowered one at a time. The
    # whole chain is recognised instead, in the order the #2125 censuses report for every wave32 chain (LLVM 15
    # spelling; vT, vX, vP, sS, sD any registers, OP one operation, ID its identity):
    #   s_or_saveexec_b32 sS, -1                            save EXEC, enable all 32 lanes
    #   v_cndmask_b32_e64 vT, ID, vX, sS                    the identity fill: vX where sS has the lane, ID elsewhere
    #   OP_dpp vT, vT, vT quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf   lane pairs
    #   OP_dpp vT, vT, vT quad_perm:[2,3,0,1] ...           quads
    #   OP_dpp vT, vT, vT row_half_mirror ...               lane i with lane 7 - i: eight lanes
    #   OP_dpp vT, vT, vT row_mirror ...                    lane i with lane 15 - i: the row of sixteen
    #   v_permlanex16_b32 vP, vT, 0, 0                      lane 0 of the other row
    #   OP vT, vP, vT                                       every lane now holds the whole wave
    #   s_mov_b32 exec_lo, sS
    #   v_readlane_b32 sD, vT, 31
    # (RDNA 2 ISA: DPP16 quad_perm, row_half_mirror and row_mirror, V_PERMLANEX16_B32, V_READLANE_B32.) With every
    # lane enabled no DPP or permlane source is disabled, so bound_ctrl and fi have nothing to decide. sD is OP over
    # the lanes active before the chain, the others adding the identity. The census also shows a min chain whose
    # readlane comes later: after its restore, one v_cndmask_b32_sdwa, then the s_or_saveexec_b32 sS2, -1 of a max
    # chain, and only then v_readlane_b32 sD, vT, 31; the max chain takes that readlane between its saveexec and its
    # fill. Readlane ignores EXEC, so the min result is still the one in vT as long as nothing in between writes vT
    # (checked); the v_cndmask_b32_sdwa keeps its own handler, under the restored EXEC. The console packs pixels into
    # waves in its own way, so a correct program cannot depend on which pixels share a wave: the chain becomes one
    # WaveActiveBitOr / WaveActiveMin / WaveActiveMax of vX over the active lanes of the desktop wave, which leaves
    # helper lanes out. Accepted only in a straight-line pixel stage whose EXEC is the entry mask, not widened by
    # s_wqm (helper lanes would then be counted on the console). Every other shape that starts like a chain is refused
    # with its reason: an unproved wave64 combine, another fill, a wrong identity, invalid steps, other or mixed
    # operations, other lane selects, an unrestored EXEC, a misplaced readlane or
    # with vT rewritten before it, a branch target inside, or vP read before it is rewritten.
    WAVE_REDUCE = {'v_or_b32': ('WaveActiveBitOr({0})', 0), 'v_min_u32': ('WaveActiveMin({0})', 0xffffffff),
                   'v_max_u32': ('WaveActiveMax({0})', 0), 'v_and_b32': ('WaveActiveBitAnd({0})', 0xffffffff),
                   'v_xor_b32': ('WaveActiveBitXor({0})', 0), 'v_add_nc_u32': ('WaveActiveSum({0})', 0)}
    WAVE_REDUCE_STEPS = ('quad_perm:[1,0,3,2]', 'quad_perm:[2,3,0,1]', 'row_half_mirror', 'row_mirror')
    WAVE_REDUCE_CONTROL = re.compile(r'quad_perm:\[[0-3],[0-3],[0-3],[0-3]\]|row_half_mirror|row_mirror')

    @staticmethod
    def wave_steps_reduce(controls, op):
        """Whether four DPP16 steps (each a quad_perm, row_half_mirror or row_mirror, every lane valid) followed by
        v_permlanex16_b32 with lane selects 0, 0 and the combine leave lane 31 with every lane's input exactly once
        (ISA, DPP16 and V_PERMLANEX16_B32). Each lane holds, per source lane, how often that input went into its value:
        a step adds the permuted lane's counts to the lane's own. An idempotent operation (or, and, min, max) only needs
        every count at least once; xor and sum need exactly once."""
        def source(control, lane):
            if control == 'row_half_mirror':
                return (lane & ~7) | (7 - (lane & 7))
            if control == 'row_mirror':
                return (lane & ~15) | (15 - (lane & 15))
            sel = [int(x) for x in control[len('quad_perm:['):-1].split(',')]
            return (lane & ~3) | sel[lane & 3]
        held = [[int(i == lane) for i in range(32)] for lane in range(32)]
        for control in controls:
            held = [[a + b for a, b in zip(held[source(control, lane)], held[lane])] for lane in range(32)]
        held = [[a + b for a, b in zip(held[(lane & ~15 ^ 16)], held[lane])] for lane in range(32)]
        counts = held[31]
        return all(c == 1 for c in counts) if op in ('v_xor_b32', 'v_add_nc_u32') else all(c >= 1 for c in counts)

    @staticmethod
    def wave_reduce_cndmask_fill(stretch, ss):
        """The identity fill of the census: exactly v_cndmask_b32_e64 vT, 0 | -1, vX, sS (the input where the saved
        mask has the lane, the inline constant elsewhere). (vT, 'vX', identity), or None for any other stretch."""
        if len(stretch) != 1:
            return None
        base, ops, mods, _text = stretch[0]
        if base != 'v_cndmask_b32' or mods or len(ops) != 4 or ops[3] != 's%d' % ss or ops[1] not in ('0', '-1') or \
                not re.fullmatch(r'v\d+', ops[0]) or not re.fullmatch(r'v\d+', ops[2]) or ops[0] == ops[2]:
            return None
        return int(ops[0][1:]), ops[2], int(ops[1]) & 0xffffffff

    # The identity fill, between s_or_saveexec_b32 sS, -1 and the first quad_perm step: each entry is a function
    # (stretch, sS) -> (vT, input operand, identity) for a form it proves, or None; stretch is a list of
    # (base, operands, modifiers, text). The identity itself is checked against OP afterwards.
    WAVE_REDUCE_FILLS = (wave_reduce_cndmask_fill,)
    # A VALU write of the first operand that keeps or reads the old value of that register does not end vP's life.
    WAVE_REDUCE_KEEPS = re.compile(r'v_(f?mac|pk_fmac|dot\w*c|interp_p2|writelane|swap|movrel)|.*(16|_dpp|_dpp8|_sdwa)$')

    def skipped_scalar_read_later(self, branches):
        """Rule C5 of the active-lane model: a scalar register written inside a block that an EXEC / VCC branch can skip
        and read after the join. The guest runs the block for the whole wave when any lane is active, so every lane sees the
        new value (also a lane that is inactive in the block and reads it after EXEC is restored); a thread that skips
        the block on its own keeps the old one. Only a wave-wide decision of the branch keeps them equal."""
        ins = getattr(self, 'ins', None)
        if not isinstance(ins, list):
            return False
        wave_branch = {off: name for off, _n, name, _a in ins
                       if name.startswith('s_cbranch_') and name[10:] in ('execz', 'execnz', 'vccz', 'vccnz')}
        for pc, target in branches:
            if pc not in wave_branch:
                continue
            written = set()
            for off, _n, name, asm in ins:
                if pc < off < target:
                    reads, writes, _may = scalar_effects(name, asm)
                    if reads & {'exec', 'vcc'} or name.startswith('v_cmp'):
                        continue   # mask arithmetic: one bit per thread already, no wave-wide value
                    written |= {w for w in writes if re.fullmatch(r's\d+', w)}
            if written and any(off >= target and written & scalar_effects(name, asm)[0] for off, _n, name, asm in ins):
                return True
        return False

    def holds_wave_reduction(self):
        """Whether the program starts a whole-wave reduction chain: s_or_saveexec_b32 sS, -1 and, within the chain's
        length, a row_mirror / row_half_mirror DPP step or a v_permlanex16_b32 (the test wave_reduction itself uses)."""
        ins = getattr(self, 'ins', None)
        if not isinstance(ins, list):
            return False
        for index, (_off, _n, name, asm) in enumerate(ins):
            _name, ops, _mods = parse(asm)
            if name in ('s_or_saveexec_b32', 's_or_saveexec_b64') and len(ops) == 2 and ops[1] == '-1':
                for nxt, nops, nmods in (parse(item[3]) for item in ins[index + 1:index + 17]):
                    if nxt == 'v_permlanex16_b32' or nxt.endswith('_dpp') and ('row_mirror' in nmods or
                                                                               'row_half_mirror' in nmods):
                        return True
        return False

    def wave_reduction(self, base, ops):
        """True when this instruction belongs to a recognised wave reduction (its result is lowered at the restore of
        EXEC, its readlane at the readlane). False for anything that does not start like one; Unsupported for a chain of
        the wrong shape."""
        plan, pc = getattr(self, 'reduce_plan', None), getattr(self, 'pc', None)
        if plan and pc in plan:
            action = plan[pc]
            if action:
                action()
            return True
        if pc is None or not isinstance(getattr(self, 'ins', None), list) or not (
                base in ('s_or_saveexec_b32', 's_or_saveexec_b64') and len(ops) == 2 and ops[1] == '-1'):
            return False
        index = next((i for i, item in enumerate(self.ins) if item[0] == pc), None)
        if index is None:
            return False
        steps = [parse(asm) for _off, _n, _name, asm in self.ins[index:index + 17]]
        if not any(name == 'v_permlanex16_b32' or name.endswith('_dpp') and ('row_mirror' in mods or
                                                                             'row_half_mirror' in mods)
                   for name, _ops, mods in steps[1:]):
            return False   # not a reduction: the instruction keeps its own handler
        wave64 = base.endswith('_b64')
        ss = reg_range(ops[0])
        if wave64:
            if not ss or ss[0] != 's' or ss[1] % 2 or ss[2] != ss[1] + 1 or ss[2] > 105:
                raise Unsupported('wave reduction in the wave64 form: EXEC needs an even SGPR pair')
        elif not ss or ss[0] != 's' or ss[1] != ss[2]:
            raise Unsupported('wave reduction: s_or_saveexec_b32 %s, -1 does not save EXEC in one SGPR' % ops[0])
        ss = ss[1]
        first = next((k for k, (name, _o, m) in enumerate(steps)
                      if name.endswith('_dpp') and any(self.WAVE_REDUCE_CONTROL.fullmatch(x) for x in m)), len(steps))
        items = self.ins[index:index + first + 10]

        def at(k, what):
            if k >= len(items):
                raise Unsupported('wave reduction: the chain ends before %s' % what)
            name, o, m = parse(items[k][3])
            return re.sub(r'_e(32|64)$', '', name), o, m, items[k][3]

        stretch = [at(k, 'the first quad_perm step') for k in range(1, first)]
        pending = getattr(self, 'reduce_pending', {})
        taken = None   # the later readlane of the min chain just before, which this chain carries before its fill
        if not wave64 and stretch and stretch[0][0] == 'v_readlane_b32' and items[1][0] in pending:
            taken = pending[items[1][0]]
            stretch = stretch[1:]
        fill = next((found for found in (form(stretch, ss) for form in self.WAVE_REDUCE_FILLS) if found), None)
        if fill is None:
            raise Unsupported('wave reduction: the identity fill after %s %s, -1 (%s) is not a known '
                              'form' % (base, ops[0], ' / '.join(t[3] for t in stretch) or 'no instruction'))
        vt, source, identity = fill
        op = None
        seen = []
        for k in range(len(self.WAVE_REDUCE_STEPS)):
            b, o, m, text = at(first + k, 'the DPP step %d of 4' % (k + 1))
            if not b.endswith('_dpp'):
                raise Unsupported('wave reduction: missing step %d of 4 (found %s)' % (k + 1, text))
            controls = [x for x in m if not x.startswith(('row_mask:', 'bank_mask:', 'bound_ctrl:', 'fi:'))]
            if len(controls) != 1 or not self.WAVE_REDUCE_CONTROL.fullmatch(controls[0]):
                raise Unsupported('wave reduction: %s in place of a quad_perm, row_half_mirror or row_mirror step'
                                  % (' '.join(controls) or 'no DPP control'))
            seen.append(controls[0])
            if 'row_mask:0xf' not in m or 'bank_mask:0xf' not in m:
                raise Unsupported('wave reduction: %s with a partial row or bank mask' % text)
            this = b[:-len('_dpp')]
            if this not in self.WAVE_REDUCE:
                raise Unsupported('wave reduction of %s: only v_or_b32, v_and_b32, v_xor_b32, v_add_nc_u32, v_min_u32 and v_max_u32 are lowered' % this)
            if op not in (None, this):
                raise Unsupported('wave reduction mixes %s and %s' % (op, this))
            op = this
            if o != ['v%d' % vt] * 3:
                raise Unsupported('wave reduction: %s does not combine v%d with itself' % (text, vt))
        if not self.wave_steps_reduce(seen, op):
            raise Unsupported('wave reduction: the steps %s do not bring every lane\'s input into lane 31%s' % (
                ', '.join(seen), ' exactly once' if op in ('v_xor_b32', 'v_add_nc_u32') else ''))
        if identity != self.WAVE_REDUCE[op][1]:
            raise Unsupported('wave reduction of %s: the inactive lanes get 0x%08x, the identity is 0x%08x'
                              % (op, identity, self.WAVE_REDUCE[op][1]))
        if taken is not None and op != 'v_max_u32':
            raise Unsupported('wave reduction: a %s chain carries the later readlane of the v_min_u32 chain before it; '
                              'only a v_max_u32 chain is lowered there' % op)
        b, o, m, text = at(first + 4, 'v_permlanex16_b32')
        if b != 'v_permlanex16_b32' or len(o) != 4 or o[1] != 'v%d' % vt or not re.fullmatch(r'v\d+', o[0]) or \
                o[0] == o[1]:
            raise Unsupported('wave reduction: %s in place of v_permlanex16_b32 vN, v%d, 0, 0' % (text, vt))
        if m or o[2:] != ['0', '0']:
            raise Unsupported('wave reduction: %s: only the lane selects 0, 0 without modifiers are lowered' % text)
        vp = int(o[0][1:])
        b, o, m, text = at(first + 5, 'the two rows are combined')
        if b != op or m or len(o) != 3 or o[0] != 'v%d' % vt or sorted(o[1:]) != sorted(['v%d' % vt, 'v%d' % vp]):
            mixed = b in self.WAVE_REDUCE and b != op
            raise Unsupported('wave reduction %s' % ('mixes %s and %s' % (op, b) if mixed else
                                                     ': %s in place of %s v%d, v%d, v%d' % (text, op, vt, vt, vp)))
        if wave64:
            return self.wave64_sum(index, first, at, op, vt, vp, ss, source)
        b, o, m, text = at(first + 6, 'EXEC is restored')
        if b != 's_mov_b32' or o != ['exec_lo', 's%d' % ss] or m:
            raise Unsupported('wave reduction: %s in place of s_mov_b32 exec_lo, s%d (EXEC restored before the readlane)'
                              % (text, ss))
        read = first + 7   # the readlane: right after the restore, or later in the min chain
        b, o, m, text = at(read, 'v_readlane_b32')
        later = b == 'v_cndmask_b32_sdwa'
        if later:
            # The census's min form: restore, one v_cndmask_b32_sdwa (its own handler, under the restored EXEC), the
            # s_or_saveexec_b32 sS2, -1 of the max chain that follows, then this chain's readlane. Readlane ignores
            # EXEC; vT must still hold the result, and sS2 must survive the readlane for the max chain's fill and
            # restore.
            if op != 'v_min_u32':
                raise Unsupported('wave reduction of %s: %s after the restore in place of v_readlane_b32 sN, v%d, 31 '
                                  '(only the v_min_u32 chain reads its lane later)' % (op, text, vt))
            if o and reg_range(o[0]) and reg_range(o[0])[0] == 'v' and \
                    reg_range(o[0])[1] <= vt <= reg_range(o[0])[2]:
                raise Unsupported('wave reduction: %s rewrites v%d before the readlane of the v_min_u32 chain'
                                  % (text, vt))
            b, o, m, text = at(first + 8, 'the s_or_saveexec_b32 of the v_max_u32 chain')
            nxt = reg_range(o[0]) if b == 's_or_saveexec_b32' and len(o) == 2 and o[1] == '-1' and not m else None
            if not nxt or nxt[0] != 's' or nxt[1] != nxt[2]:
                raise Unsupported('wave reduction: %s in place of the s_or_saveexec_b32 sN, -1 of the v_max_u32 chain '
                                  'before the readlane of the v_min_u32 chain' % text)
            read = first + 9
            b, o, m, text = at(read, 'the readlane of the v_min_u32 chain')
        sd = reg_range(o[0]) if len(o) == 3 else None
        if b != 'v_readlane_b32' or m or not sd or sd[0] != 's' or sd[1] != sd[2] or sd[1] == ss or \
                o[1] != 'v%d' % vt:
            raise Unsupported('wave reduction: %s in place of v_readlane_b32 sN, v%d, 31' % (text, vt))
        if o[2] != '31':
            raise Unsupported('wave reduction: %s reads lane %s: only lane 31 is lowered' % (text, o[2]))
        if later and sd[1] == nxt[1]:
            raise Unsupported('wave reduction: the readlane of the v_min_u32 chain overwrites s%d, the EXEC saved by '
                              'the v_max_u32 chain' % sd[1])
        end = items[read][0]
        targets = {int(parse(asm)[1][0].split('_')[1], 16) for _off, _n, name, asm in self.ins
                   if name.startswith('s_cbranch') or name == 's_branch'}
        retry_wave = getattr(self, 'retry_wave_reduction', False)
        if retry_wave:
            targets.update(t for t in self.calculated_pc.values() if t is not None)
        inside = sorted(t for t in targets if pc < t <= end)
        if inside:
            raise Unsupported('wave reduction: branch target 0x%x inside the chain' % inside[0])
        if getattr(self, 'kind', None) != 'ps' or getattr(self, 'fetched', False):
            raise Unsupported('wave reduction outside a pixel stage')
        if retry_wave and pc not in self.reduction_entry_exec:
            raise Unsupported('wave reduction: EXEC is not proven to be the entry mask on every incoming path')
        if not getattr(self, 'wave_exec', False):
            if getattr(self, 'has_branches', True) or getattr(self, 'blocks', True):
                raise Unsupported(WAVE_REDUCTION_BRANCHES)
            if not getattr(self, 'exec_known', False):
                raise Unsupported('wave reduction: EXEC is not the entry mask')
        if getattr(self, 'wqm_exec', False):
            raise Unsupported('wave reduction after s_wqm exec, exec: helper lanes are in EXEC and would be counted')
        after_restore = index + first + 7
        if retry_wave:
            self.reduction_scratch[index + first + 4] = (vp, after_restore)
        else:
            for _off, _n, name, asm in self.ins[after_restore:]:
                _name, o, _m = parse(asm)
                hits = [i for i, tok in enumerate(o) if any(
                    int(r.group(3) or r.group(1)) <= vp <= int(r.group(3) or r.group(2))
                    for r in re.finditer(r'\bv\[(\d+):(\d+)\]|\bv(\d+)\b', tok))]
                if not hits:
                    continue
                short = re.sub(r'_e(32|64)$', '', name)
                if hits == [0] and short.startswith('v_') and not self.WAVE_REDUCE_KEEPS.match(short):
                    break
                raise self.wave_scratch_read(vp, asm)
        # Nothing of the chain is emitted until the restore, so every register read there still holds its value from
        # the fill (the only instruction lowered in between is the min chain's readlane carried here, an SGPR write
        # that is not this chain's saved EXEC). The readlane then copies vT, which nothing has rewritten since.
        plan = self.reduce_plan = getattr(self, 'reduce_plan', {})
        if taken is not None:
            taken[0] = True
        for k in range(1, read + 1):
            if k in (first + 7, first + 8) and later:
                continue   # the v_cndmask_b32_sdwa keeps its handler; the max chain's saveexec starts that chain
            plan.setdefault(items[k][0], None)
        plan[items[first + 6][0]] = lambda: self.lower_wave_reduction(op, vt, vp, ss, source)
        claim = [False]
        if later:
            self.reduce_pending = {**pending, items[read][0]: claim}
        plan[items[read][0]] = lambda: self.lower_wave_readlane(sd[1], vt, claim if later else None)
        return True

    def wave64_sum(self, index, first, at, op, vt, vp, ss, source):
        """The DPP/permlane prefix produces two modulo-32-bit partial sums, one in each half."""
        def refuse(reason):
            raise Unsupported('wave reduction in the wave64 form: ' + reason)

        if (self.kind != 'ps' or self.fetched or self.has_branches or self.blocks or self.calculated_pc
                or not self.exec_known or self.wqm_exec):
            refuse('requires a straight-line pixel stage with entry EXEC')
        if op != 'v_add_nc_u32':
            refuse('the two half-wave results must be sums')
        b, o, m, _ = at(first + 6, 'the wave64 EXEC restore')
        if (b, o, m) != ('s_mov_b64', ['exec', 's[%d:%d]' % (ss, ss + 1)], []):
            refuse('requires restoration of both EXEC words before the readlanes')
        halves = {}
        for k in (first + 7, first + 8):
            b, o, m, _ = at(k, 'a wave64 half-wave readlane')
            if (b != 'v_readlane_b32' or m or len(o) != 3 or o[1] != 'v%d' % vt
                    or o[2] not in ('31', '63') or not re.fullmatch(r's\d+', o[0])
                    or int(o[0][1:]) > 105):
                refuse('requires readlanes of lanes 31 and 63')
            halves[o[2]] = o[0]
        if len(halves) != 2 or len(set(halves.values())) != 2:
            refuse('requires distinct readlanes of lanes 31 and 63 into distinct SGPRs')
        end = first + 9
        b, o, m, _ = at(end, 'the scalar sum of the wave64 halves')
        if (b != 's_add_u32' or m or len(o) != 3 or set(o[1:]) != set(halves.values())
                or not re.fullmatch(r's\d+', o[0]) or int(o[0][1:]) > 105):
            refuse('requires s_add_u32 of the two half-wave sums')
        dest = int(o[0][1:])
        scalar_scratch = {'s%d' % ss, 's%d' % (ss + 1), *halves.values()} - {o[0]}
        scc_live = True
        for _, _, name, asm in self.ins[index + end + 1:]:
            reads, _, _ = scalar_effects(name, asm)
            if scc_live and 'scc' in reads:
                refuse('the carry of the half-wave add is read after the chain')
            # Use only explicit SCC writes that the translator can check. The half sums, the saved EXEC
            # and the vector temporaries have no desktop equivalents in this lowering.
            if name.startswith(('s_cmp', 's_bitcmp')):
                scc_live = False
            _, operands, _ = parse(asm)
            if scalar_scratch & scalar_tokens(asm) or 'movrel' in name or any(
                    r[0] == 'v' and any(r[1] <= v <= r[2] for v in (vt, vp))
                    for operand in operands if (r := reg_range(operand.strip('-|')))):
                refuse('a half-wave temporary is used after the chain')

        plan = self.reduce_plan = getattr(self, 'reduce_plan', {})
        for pc, _, _, _ in self.ins[index + 1:index + end + 1]:
            plan[pc] = None

        def lower():
            self.needs_sm6 = True
            self.set_s(dest, 'WaveActiveSum(%s)' % self.u(source), uniform=True)
            self.scc_effect = 'the carry of the wave64 half-wave add'
            self.scc_verified = False

        plan[self.ins[index + end][0]] = lower
        return True

    @staticmethod
    def wave_scratch_read(vp, asm):
        return Unsupported('wave reduction: v%d (the other row from v_permlanex16_b32) is read after the chain by %s'
                           % (vp, asm))

    def wave_scratch_refusal(self):
        """Find a use of a discarded permutation result on any path after the chain's restore."""
        bounds, block_of, succ, _ = self.spill_cfg
        for vp, after_restore in self.reduction_scratch.values():
            pending, seen = [after_restore], set()
            scratch = 'v%d' % vp
            while pending:
                i = pending.pop()
                if i in seen:
                    continue
                seen.add(i)
                off, _, _, asm = self.ins[i]
                name, ops, mods = parse(asm)
                # The next recognized permutation kills the old result. Its own paths are checked separately.
                if i in self.reduction_scratch and self.reduction_scratch[i][0] == vp:
                    continue
                if (name in ('v_mov_b32', 'v_mov_b32_e32', 'v_mov_b32_e64') and not mods
                        and ops[0] == scratch and ops[1] != scratch and off in self.reduction_entry_exec):
                    continue
                if vp in mesh_vregs(ops):
                    return self.wave_scratch_read(vp, asm)
                block = block_of[i]
                if i + 1 < bounds[block][1]:
                    pending.append(i + 1)
                else:
                    pending.extend(bounds[b][0] for b in succ[block])
        return None

    def lower_wave_reduction(self, op, vt, vp, ss, source):
        operand = self.u(source)
        if not self.exec_known and not getattr(self, 'retry_wave_reduction', False):
            # Past an EXEC write the lanes outside the old EXEC take the identity (the v_cndmask of the chain).
            operand = '((%s) ? %s : %du)' % (LE.EXEC_BIT, operand, self.WAVE_REDUCE[op][1] & 0xffffffff)
        value = self.WAVE_REDUCE[op][0].format(operand)
        self.needs_sm6 = True
        self.set_s(ss, 'exec')
        self.sym[ss] = ('execcopy',)
        self.masks.add(ss)
        # The guest runs the chain with every lane enabled, so vT is written in every lane and every thread of the host
        # wave takes part in the intrinsic: no `if (exec & 1)` around it (an inactive lane's readlane source must be set).
        was_known, self.exec_known = self.exec_known, True
        try:
            self.set_v(vt, value)
        finally:
            self.exec_known = was_known
        if hasattr(self, 'uniform_v'):
            self.uniform_v.discard(vp)
        self.set_scc('true')   # s_or_saveexec_b32 set SCC from the all-lanes EXEC

    def lower_wave_readlane(self, sd, vt, claim):
        if claim is not None and not claim[0]:
            raise Unsupported('wave reduction: the later readlane of the v_min_u32 chain is not carried by a v_max_u32 '
                              'chain')
        self.set_s(sd, self.u('v%d' % vt), uniform=False)


# 32-bit raw MUBUF atomics: opcode_name -> (RWByteAddressBuffer method suffix, signed, is_sub, data_vgprs)
MUBUF_ATOMIC_OPS = {
    'buffer_atomic_swap': ('Exchange', False, False, 1),
    'buffer_atomic_cmpswap': ('CompareExchange', False, False, 2),
    'buffer_atomic_add': ('Add', False, False, 1),
    'buffer_atomic_sub': ('Add', False, True, 1),
    'buffer_atomic_smin': ('Min', True, False, 1),
    'buffer_atomic_umin': ('Min', False, False, 1),
    'buffer_atomic_smax': ('Max', True, False, 1),
    'buffer_atomic_umax': ('Max', False, False, 1),
    'buffer_atomic_and': ('And', False, False, 1),
    'buffer_atomic_or': ('Or', False, False, 1),
    'buffer_atomic_xor': ('Xor', False, False, 1),
}

# Legacy straight-line LDS exchange ops checked by check_lds_exchange().
LEGACY_LDS_EXCHANGE_OPS = frozenset((
    'ds_write_b32', 'ds_read_b32', 'ds_read2_b32', 'ds_write_b64', 'ds_read_b64', 'ds_write_b128', 'ds_read_b128',
))

# Dual-offset LDS ops: name -> (is_write, slot_bytes, offset_stride_bytes)
LDS_DUAL_OPS = {
    'ds_write2_b32': (True, 4, 4),
    'ds_write2st64_b32': (True, 4, 256),
    'ds_read2st64_b32': (False, 4, 256),
    'ds_write2_b64': (True, 8, 8),
    'ds_write2st64_b64': (True, 8, 512),
    'ds_read2_b64': (False, 8, 8),
    'ds_read2st64_b64': (False, 8, 512),
}

# 32-bit unsigned/bitwise LDS atomics: name -> (HLSL Interlocked* suffix, has_rtn)
LDS_ATOMIC_OPS = {
    'ds_add_u32': ('Add', False),
    'ds_min_u32': ('Min', False),
    'ds_max_u32': ('Max', False),
    'ds_and_b32': ('And', False),
    'ds_or_b32': ('Or', False),
    'ds_add_rtn_u32': ('Add', True),
    'ds_min_rtn_u32': ('Min', True),
    'ds_max_rtn_u32': ('Max', True),
    'ds_and_rtn_b32': ('And', True),
    'ds_or_rtn_b32': ('Or', True),
    'ds_wrxchg_rtn_b32': ('Exchange', True),
}

LDS_DUAL_ATOMIC_OPS = frozenset((*LDS_DUAL_OPS, *LDS_ATOMIC_OPS))

# Unused w1 register field masks in RDNA 2 DS encoding (must be zero):
# w1[7:0]=ADDR, w1[15:8]=DATA0, w1[23:16]=DATA1, w1[31:24]=VDST.
LDS_UNUSED_W1_MASK = {
    'ds_write_b32': 0xffff0000,
    **{op: (0xff000000 if is_write else 0x00ffff00)
       for op, (is_write, _, _) in LDS_DUAL_OPS.items()},
    **{op: (0x00ff0000 if has_rtn else 0xffff0000) for op, (_, has_rtn) in LDS_ATOMIC_OPS.items()},
}

# name -> (source operand count, expression); operands are floats
VEC_F = {
    'v_add_f32': (2, '{0} + {1}'), 'v_sub_f32': (2, '{0} - {1}'), 'v_subrev_f32': (2, '{1} - {0}'),
    'v_mul_f32': (2, '{0} * {1}'), 'v_mul_legacy_f32': (2, 'mul_legacy({0}, {1})'),
    'v_min_f32': (2, 'minmax_zero32({0}, {1}, false)'), 'v_max_f32': (2, 'minmax_zero32({0}, {1}, true)'),
    'v_fmac_f32': (2, '{0} * {1} + {2}'), 'v_mac_f32': (2, 'mad32({0}, {1}, {2}, false)'),
    'v_fma_f32': (3, '{0} * {1} + {2}'), 'v_mad_f32': (3, 'mad32({0}, {1}, {2}, false)'),
    'v_mad_legacy_f32': (3, 'mad32({0}, {1}, {2}, true)'),
    'v_fmamk_f32': (3, '{0} * {1} + {2}'), 'v_fmaak_f32': (3, '{0} * {1} + {2}'),
    'v_madmk_f32': (3, 'mad32({0}, {1}, {2}, false)'), 'v_madak_f32': (3, 'mad32({0}, {1}, {2}, false)'),
    'v_rcp_f32': (1, '1.0 / {0}'), 'v_rcp_iflag_f32': (1, '1.0 / {0}'), 'v_rsq_f32': (1, 'rsqrt({0})'), 'v_sqrt_f32': (1, 'sqrt({0})'),
    'v_exp_f32': (1, 'exp2({0})'), 'v_log_f32': (1, 'log2({0})'), 'v_floor_f32': (1, 'floor({0})'), 'v_ceil_f32': (1, 'ceil({0})'),
    'v_trunc_f32': (1, 'trunc({0})'), 'v_fract_f32': (1, 'fract32({0})'), 'v_rndne_f32': (1, 'round({0})'),
    'v_sin_f32': (1, 'sin({0} * 6.283185307)'), 'v_cos_f32': (1, 'cos({0} * 6.283185307)'),
    'v_med3_f32': (3, 'med3_32({0}, {1}, {2})'),
    'v_min3_f32': (3, 'minmax_zero32({0}, minmax_zero32({1}, {2}, false), false)'),
    'v_max3_f32': (3, 'minmax_zero32({0}, minmax_zero32({1}, {2}, true), true)'),
}
# v_fma_mix*: precise keeps the sign of a zero result; mad still lets a backend fuse or round the product.
MIX32 = 'float mix32(float a, float b, float c) { precise float result = mad(a, b, c); return result; }'
# 16-bit VALU: (sources, source kind, float or uint expression of the sources, result kind). 'h' sources are
# binary16 read from the low half and computed in binary32; an 'h' result is rounded back to binary16. The
# i16/u16 conversions to binary16 read the low half as an integer; the conversions to integers truncate toward
# zero, saturate to the 16-bit range and read NaN as 0.
VEC_H = {
    'v_exp_f16': (1, 'h', 'exp2({0})', 'h'), 'v_rcp_f16': (1, 'h', '1.0 / {0}', 'h'),
    'v_add_f16': (2, 'h', '{0} + {1}', 'h'), 'v_sub_f16': (2, 'h', '{0} - {1}', 'h'),
    'v_subrev_f16': (2, 'h', '{1} - {0}', 'h'), 'v_mul_f16': (2, 'h', 'mul16({0}, {1})', 'h'),
    'v_min_f16': (2, 'h', 'min({0}, {1})', 'h'), 'v_max_f16': (2, 'h', 'max({0}, {1})', 'h'),
    'v_fmac_f16': (2, 'h', 'fma16o({0}, {1}, {2})', 'h'), 'v_fmamk_f16': (3, 'h', 'fma16o({0}, {1}, {2})', 'h'),
    'v_fmaak_f16': (3, 'h', 'fma16o({0}, {1}, {2})', 'h'),
    'v_cvt_f16_i16': (1, 'i16', '{0}', 'h'), 'v_cvt_f16_u16': (1, 'u16', '{0}', 'h'),
    'v_cvt_i16_f16': (1, 'h', 'isnan({0}) ? 0u : (uint(int(clamp(trunc({0}), -32768.0, 32767.0))) & 0xffffu)', 'u'),
    'v_cvt_u16_f16': (1, 'h', 'isnan({0}) ? 0u : uint(clamp(trunc({0}), 0.0, 65535.0))', 'u'),
}
# Conversions whose result is binary32. The expression is the float value, so an output modifier
# (clamp, mul:2, mul:4, div:2) can wrap it the way it wraps any other float result. The uint-result
# conversions stay in VEC_U.
VEC_U_TO_F = {
    'v_cvt_f32_u32': (1, 'float({0})'), 'v_cvt_f32_i32': (1, 'float(int({0}))'),
    'v_cvt_f32_ubyte0': (1, 'float({0} & 255)'), 'v_cvt_f32_ubyte1': (1, 'float(({0} >> 8) & 255)'),
    'v_cvt_f32_ubyte2': (1, 'float(({0} >> 16) & 255)'), 'v_cvt_f32_ubyte3': (1, 'float({0} >> 24)'),
    'v_cvt_f32_f16': (1, 'f16tof32({0})'),
    'v_cvt_off_f32_i4': (1, '(float(int({0} << 28) >> 28) * 0.0625)'),
}
# operands are uints; the result is a uint expression
VEC_U = {
    'v_add_nc_u32': (2, '{0} + {1}'), 'v_sub_nc_u32': (2, '{0} - {1}'), 'v_subrev_nc_u32': (2, '{1} - {0}'),
    'v_mul_lo_u32': (2, '{0} * {1}'), 'v_mul_hi_u32': (2, 'mul_hi_u32({0}, {1})'), 'v_mul_hi_i32': (2, 'mul_hi_i32({0}, {1})'), 'v_and_b32': (2, '{0} & {1}'), 'v_or_b32': (2, '{0} | {1}'), 'v_xor_b32': (2, '{0} ^ {1}'),
    'v_not_b32': (1, '~{0}'), 'v_lshlrev_b32': (2, '{1} << ({0} & 31)'), 'v_lshrrev_b32': (2, '{1} >> ({0} & 31)'),
    'v_ashrrev_i32': (2, 'uint(int({1}) >> ({0} & 31))'), 'v_min_u32': (2, 'min({0}, {1})'), 'v_max_u32': (2, 'max({0}, {1})'),
    'v_min_i32': (2, 'uint(min(int({0}), int({1})))'), 'v_max_i32': (2, 'uint(max(int({0}), int({1})))'),
    'v_min3_u32': (3, 'min(min({0}, {1}), {2})'),
    'v_max3_u32': (3, 'max(max({0}, {1}), {2})'),
    'v_med3_u32': (3, 'max(min({0}, {1}), min(max({0}, {1}), {2}))'),
    'v_min3_i32': (3, 'uint(min(min(int({0}), int({1})), int({2})))'),
    'v_max3_i32': (3, 'uint(max(max(int({0}), int({1})), int({2})))'),
    'v_med3_i32': (3, 'uint(max(min(int({0}), int({1})), min(max(int({0}), int({1})), int({2}))))'),
    'v_cvt_u32_f32': (1, 'uint(max(F({0}), 0.0))'),
    'v_cvt_f16_f32': (1, 'f32tof16(F({0}))'),
    'v_bfe_u32': (3, '({0} >> ({1} & 31)) & ((1u << ({2} & 31)) - 1)'), 'v_mad_u32_u24': (3, '({0} & 0xffffffu) * ({1} & 0xffffffu) + {2}'),
    'v_mad_i32_i24': (3, 'asuint(asint({0} << 8) >> 8) * asuint(asint({1} << 8) >> 8) + {2}'),
    'v_mad_u32_u16': (3, '({0} & 0xffffu) * ({1} & 0xffffu) + {2}'),
    'v_lshl_add_u32': (3, '({0} << ({1} & 31)) + {2}'), 'v_add_lshl_u32': (3, '({0} + {1}) << ({2} & 31)'), 'v_add3_u32': (3, '{0} + {1} + {2}'),
    'v_or3_b32': (3, '{0} | {1} | {2}'), 'v_xor3_b32': (3, '{0} ^ {1} ^ {2}'), 'v_and_or_b32': (3, '({0} & {1}) | {2}'), 'v_lshl_or_b32': (3, '({0} << ({1} & 31)) | {2}'),
    'v_xad_u32': (3, '({0} ^ {1}) + {2}'), 'v_bfi_b32': (3, '({0} & {1}) | (~{0} & {2})'),
    'v_mul_u32_u24': (2, '({0} & 0xffffffu) * ({1} & 0xffffffu)'),
    # v_bfm_b32: a mask of S0[4:0] ones shifted left by S1[4:0] (same field rule as s_bfm_b32).
    'v_bfm_b32': (2, '(((1u << ({0} & 31)) - 1) << ({1} & 31))'),
    'v_mul_i32_i24': (2, 'asuint(asint({0} << 8) >> 8) * asuint(asint({1} << 8) >> 8)'),
    # Per-lane bit operations (AMD RDNA 2 ISA reference). The bitscan results count from the most significant
    # bit for the "h" forms and read 0xffffffff when no bit qualifies; firstbitlow(0) is already 0xffffffff.
    # v_bfe_i32 extracts a field at S1[4:0] of width S2[4:0] (0 reads 0) and sign-extends it from its top bit,
    # taking the bits above bit 31 as copies of bit 31; v_bcnt_u32_b32 adds S1 to the population count of S0.
    'v_bfe_i32': (3, '(({2} & 31u) == 0u ? 0u : asuint(asint({0} << (32u - min(({1} & 31u) + ({2} & 31u), 32u))) '
                     '>> (32u - min(({1} & 31u) + ({2} & 31u), 32u) + ({1} & 31u))))'),
    'v_bcnt_u32_b32': (2, 'countbits({0}) + {1}'), 'v_ffbl_b32': (1, 'firstbitlow({0})'),
    'v_ffbh_u32': (1, '({0} == 0u ? 0xffffffffu : 31u - firstbithigh({0}))'),
    'v_ffbh_i32': (1, '({0} == 0u || {0} == 0xffffffffu ? 0xffffffffu : 31u - firstbithigh(asint({0}) < 0 ? ~{0} : {0}))'),
    'v_bfrev_b32': (1, 'reversebits({0})'),
    # Unsigned sums of absolute differences (AMD RDNA 2 ISA reference, VOP3 opcodes 0x15a to 0x15d and 0x171), all
    # modulo 2^32. src1 is the reference: v_msad_u8 leaves out each byte lane whose src1 byte is zero, as in
    # tools/pixel_aot.py.
    'v_sad_u8': (3, 'sad_u8({0}, {1}, {2})'),
    'v_sad_hi_u8': (3, 'sad_hi_u8({0}, {1}, {2})'),
    'v_sad_u16': (3, 'sad_u16({0}, {1}, {2})'),
    'v_sad_u32': (3, 'sad_u32({0}, {1}, {2})'),
    'v_msad_u8': (3, 'msad_u8({0}, {1}, {2})'),
    # Saturating 32-to-16-bit integer packs (AMD RDNA 2 ISA reference, VOP3A opcodes 0x36a and 0x36b; saturation
    # bounds per Mesa NIR pack_uvec2_to_uint / pack_ivec2_to_uint selected in ACO): src0 -> low 16 bits, src1 -> high 16 bits.
    'v_cvt_pk_u16_u32': (2, 'min({0}, 0xffffu) | (min({1}, 0xffffu) << 16u)'),
    'v_cvt_pk_i16_i32': (2, '(asuint(clamp(asint({0}), -32768, 32767)) & 0xffffu) | '
                            '(asuint(clamp(asint({1}), -32768, 32767)) << 16u)'),
}


def _mul24_hi_i32():
    """High 32 bits of the 48-bit product of two signed 24-bit fields, in 32-bit integer arithmetic.

    With a = ah * 2^12 + al (al in 0..4095, ah the sign-extended top 12 bits) the product is
    hh * 2^24 + mid * 2^12 + lo, so p >> 12 = hh * 2^12 + t with t = mid + (lo >> 12), and
    p >> 32 = (hh + (t >> 12)) >> 8 exactly (floor division nests), which stays inside 32 bits.
    """
    def top(n):
        return '(asint({%d} << 8) >> 20)' % n

    def low(n):
        return 'asint({%d} & 0xfffu)' % n
    hh = '(%s * %s)' % (top(0), top(1))
    mid = '(%s * %s + %s * %s)' % (top(0), low(1), low(0), top(1))
    t = '(%s + ((%s * %s) >> 12))' % (mid, low(0), low(1))
    return 'asuint(((%s + (%s >> 12)) >> 8))' % (hh, t)


VEC_U.update({
    # AMD RDNA 2 ISA reference: the 24-bit multiplies read the low 24 bits of each source, the signed forms as a
    # two's complement field. The mul and mad forms keep the low 32 bits of the product; the hi forms keep bits 63..32
    # of the 48-bit product. The 24-bit field is sign-extended with a left shift of 8 and an arithmetic right shift.
    # v_mul_hi_u32_u24 is deliberately not lowered. Its result is at most 0xffff, so as a float output word it is
    # always a binary32 denormal, and FXC 10.0.26100 at ps_5_1 (the startup path) folds a float output whose
    # provable range lies below 0x00800000 to a constant zero (any form tried, even a plain shift without a
    # product). The same words through a uint4 output survive, and DXC keeps them. v_mul_hi_i32_i24 can reach
    # the whole 32-bit range, so FXC keeps it. A refusal is visible; zeros are not.
    'v_mul_hi_i32_i24': (2, _mul24_hi_i32()),
    'v_xnor_b32': (2, '~({0} ^ {1})'),
    # v_alignbit_b32 takes bits [S2[4:0] + 31 : S2[4:0]] of the 64-bit value {S0, S1}; a zero shift returns S1 and
    # the guard avoids a shift by 32. v_alignbyte_b32 does the same with S2[1:0] * 8 bits.
    'v_alignbit_b32': (3, '(({2} & 31u) == 0u ? {1} : (({1} >> ({2} & 31u)) | ({0} << (32u - ({2} & 31u)))))'),
    'v_alignbyte_b32': (3, '(({2} & 3u) == 0u ? {1} : (({1} >> (({2} & 3u) * 8u)) | ({0} << (32u - ({2} & 3u) * 8u))))'),
})

VOP3_ONLY_VEC_U = frozenset(('v_cvt_pk_u16_u32', 'v_cvt_pk_i16_i32'))
# scalar ops: (expression over uint sources, SCC rule with $ for the result or None)
# These SCC rules use the inputs, so they cannot be expressed as a template over the result.
SCC_LEGACY_MIN_MAX = frozenset(('s_min_i32', 's_min_u32', 's_max_i32', 's_max_u32'))
SCC_UNMODELLED = SCC_LEGACY_MIN_MAX | {'s_add_i32', 's_add_u32', 's_sub_i32', 's_sub_u32'}


def shift_add_carry(base, values, result):
    return '(%s >> %d) != 0 || (%s) < %s' % (values[0], 32 - SHIFT_ADD[base], result, values[1])


def add_sub_min_max_scc(base, a, b, result):
    """The SCC expression of s_add/s_sub/s_min/s_max_{i,u}32 over the source texts a and b and the result text.
    s_add_u32: carry out; s_sub_u32: borrow (S0 < S1); s_add_i32 / s_sub_i32: signed overflow (the sign of the result
    differs from the sign the operands imply); s_min / s_max: SCC = (D == S0), the first source was chosen."""
    if base == 's_add_u32':
        return '(%s) < (%s)' % (result, a)
    if base == 's_sub_u32':
        return '(%s) < (%s)' % (a, b)
    if base == 's_add_i32':
        return '(((((%s) ^ (%s)) & ((%s) ^ (%s))) >> 31) != 0)' % (a, result, b, result)
    if base == 's_sub_i32':
        return '(((((%s) ^ (%s)) & ((%s) ^ (%s))) >> 31) != 0)' % (a, b, a, result)
    cast = 'int' if base.endswith('_i32') else ''
    return '(%s(%s) %s %s(%s))' % (cast, a, '<=' if base.startswith('s_min') else '>=', cast, b)

PACK_B16 = frozenset(('s_pack_ll_b32_b16', 's_pack_lh_b32_b16', 's_pack_hh_b32_b16'))
SCA = {
    's_mov_b32': ('{0}', None), 's_mov_b64': ('{0}', None), 's_movk_i32': ('{0}', None),
    's_and_b32': ('{0} & {1}', '$ != 0'), 's_or_b32': ('{0} | {1}', '$ != 0'), 's_xor_b32': ('{0} ^ {1}', '$ != 0'),
    's_andn2_b32': ('{0} & ~{1}', '$ != 0'), 's_orn2_b32': ('{0} | ~{1}', '$ != 0'), 's_not_b32': ('~{0}', '$ != 0'),
    's_nand_b32': ('~({0} & {1})', '$ != 0'), 's_nor_b32': ('~({0} | {1})', '$ != 0'), 's_xnor_b32': ('~({0} ^ {1})', '$ != 0'),
    's_and_b64': ('{0} & {1}', '$ != 0'), 's_or_b64': ('{0} | {1}', '$ != 0'), 's_xor_b64': ('{0} ^ {1}', '$ != 0'),
    's_andn2_b64': ('{0} & ~{1}', '$ != 0'), 's_orn2_b64': ('{0} | ~{1}', '$ != 0'), 's_not_b64': ('~{0}', '$ != 0'),
    's_nand_b64': ('~({0} & {1})', '$ != 0'), 's_nor_b64': ('~({0} | {1})', '$ != 0'), 's_xnor_b64': ('~({0} ^ {1})', '$ != 0'),
    's_lshl_b32': ('{0} << ({1} & 31)', '$ != 0'), 's_lshr_b32': ('{0} >> ({1} & 31)', '$ != 0'),
    's_ashr_i32': ('uint(int({0}) >> ({1} & 31))', '$ != 0'),
    's_add_i32': ('{0} + {1}', None), 's_add_u32': ('{0} + {1}', None), 's_sub_i32': ('{0} - {1}', None), 's_sub_u32': ('{0} - {1}', None),
    's_mul_i32': ('{0} * {1}', None), 's_min_u32': ('min({0}, {1})', None), 's_max_u32': ('max({0}, {1})', None),
    's_min_i32': ('uint(min(int({0}), int({1})))', None), 's_max_i32': ('uint(max(int({0}), int({1})))', None),
    's_bfe_u32': ('({0} >> ({1} & 31)) & ((({1} >> 16) & 127) >= 32 ? 0xffffffffu : ((1u << (({1} >> 16) & 31)) - 1))', '$ != 0'),
    's_brev_b32': ('reversebits({0})', None),
    # s_mul_hi_{u,i}32: high 32 bits of the unsigned/signed 64-bit product; SCC unchanged.
    's_mul_hi_u32': ('mul_hi_u32({0}, {1})', None),
    's_mul_hi_i32': ('mul_hi_i32({0}, {1})', None),
    # s_sext_i32_iN: sign-extend the low N bits of S0; SCC unchanged.
    's_sext_i32_i8': ('uint(int({0} << 24) >> 24)', None),
    's_sext_i32_i16': ('uint(int({0} << 16) >> 16)', None),
    # s_pack_*_b32_b16: two halfwords of the sources into one word; SCC unchanged.
    's_pack_ll_b32_b16': ('({0} & 0xffffu) | ({1} << 16)', None),
    's_pack_lh_b32_b16': ('({0} & 0xffffu) | ({1} & 0xffff0000u)', None),
    's_pack_hh_b32_b16': ('({0} >> 16) | ({1} & 0xffff0000u)', None),
    # s_absdiff_i32: D = S0 - S1 wrapped to 32 bits, negated if negative (INT_MIN stays); SCC = result != 0.
    's_absdiff_i32': ('asuint(abs(asint({0} - {1})))', '$ != 0'),
    # s_bfm_b32: a mask of S0[4:0] ones shifted left by S1[4:0]; SCC unchanged.
    's_bfm_b32': ('(((1u << ({0} & 31)) - 1) << ({1} & 31))', None),
    # s_lshlN_add_u32: low 32 bits of (S0 << N) + S1. SHIFT_ADD also supplies SCC for wide overflow.
    's_lshl1_add_u32': ('({0} << 1) + {1}', None), 's_lshl2_add_u32': ('({0} << 2) + {1}', None),
    's_lshl3_add_u32': ('({0} << 3) + {1}', None), 's_lshl4_add_u32': ('({0} << 4) + {1}', None),
}
# SCA operations folded when every operand is a known constant (Stage.scalar), computing what their SCA text
# computes (AMD RDNA 2 ISA: S_BREV_B32 D.u[31:0] = S0.u[0:31], no SCC write). s_bfm_b32 has its own handler.
SCALAR_CONST_FOLD = {
    's_brev_b32': lambda a: int('{:032b}'.format(a)[::-1], 2),
}
# shift amount of the scalar shift-and-add forms; their SCC is a carry from the sources, not a rule over the result
SHIFT_ADD = {'s_lshl1_add_u32': 1, 's_lshl2_add_u32': 2, 's_lshl3_add_u32': 3, 's_lshl4_add_u32': 4}
MASK_WORD_SOP2 = SHIFT_ADD.keys() | {'s_mul_i32', 's_nor_b32'}
for _name, _op in (('eq', '=='), ('lg', '!='), ('gt', '>'), ('ge', '>='), ('lt', '<'), ('le', '<=')):
    for _typ, _cast in (('i32', 'int({0}) %s int({1})'), ('u32', '{0} %s {1}')):
        def _make(op=_op, cast=_cast, name='s_cmp_%s_%s' % (_name, _typ)):
            def handler(self, ops, mods):
                # A scalar compare reads exactly two sources; refusing any other count keeps a malformed
                # instruction a refusal rather than an index error in the batch translator.
                if len(ops) != 2:
                    raise Unsupported('%s with %d operands, expected 2' % (name, len(ops)))
                symbols = [self.sym.get(r[1]) if (r := reg_range(t)) and r[0] == 's' else None for t in ops]
                descriptor = (getattr(self, 'kind', None) in ('ps', 'vs')
                              and any(s and s[0] in ('desc', 'descnum') for s in symbols))
                if descriptor:
                    if mods or any(self.is_lane_mask(t) or t == 'src_scc' for t in ops):
                        raise Unsupported('scc from descriptor data in %s with a lane mask, SCC or modifier' % name)
                    self.set_scc((cast % op).format(*(self.descriptor_operand(t) for t in ops)))
                    self.mark_descriptor_scc()
                else:
                    self.set_scc((cast % op).format(self.u(ops[0]), self.u(ops[1])))
            return handler
        setattr(Stage, 'op_s_cmp_%s_%s' % (_name, _typ), _make())
    for _typ, _cast in (('i32', 'int({0}) %s int({1})'), ('u32', '{0} %s {1}')):
        def _make_cmpk(op=_op, cast=_cast, signed=_typ == 'i32', name='s_cmpk_%s_%s' % (_name, _typ)):
            def handler(self, ops, mods):
                if len(ops) != 2:
                    raise Unsupported('%s with %d operands, expected 2' % (name, len(ops)))
                if mods:
                    raise Unsupported('%s with modifiers' % name)
                self.set_scc((cast % op).format(self.u(ops[0]), '0x%08xu' % sopk_imm(ops[1], signed)))
            return handler
        setattr(Stage, 'op_s_cmpk_%s_%s' % (_name, _typ), _make_cmpk())
# Scalar bit counts and scans with a 32-bit result. `lo`/`hi` are the uint words of the source: a 32-bit source has only
# `lo`. The scans return 0xffffffff (-1) when no bit qualifies. Only the counts set SCC (result != 0).
def _clz64(lo, hi):
    return ('(%(h)s != 0u ? 31u - firstbithigh(%(h)s) : (%(l)s != 0u ? 63u - firstbithigh(%(l)s) : 0xffffffffu))'
            % {'l': lo, 'h': hi})


def _ctz64(lo, hi):
    return ('(%(l)s != 0u ? firstbitlow(%(l)s) : (%(h)s != 0u ? 32u + firstbitlow(%(h)s) : 0xffffffffu))'
            % {'l': lo, 'h': hi})


def _flbit_i64(lo, hi):
    # first bit that differs from the sign bit: complement a negative value, then count leading zeros
    neg = '((%s & 0x80000000u) != 0u)' % hi
    return _clz64('(%s ? ~%s : %s)' % (neg, lo, lo), '(%s ? ~%s : %s)' % (neg, hi, hi))


# name -> (source is a register pair, expression over (lo, hi), SCC rule or None)
SCALAR_BITS = {
    's_bcnt0_i32_b32': (False, lambda lo, hi: '(32u - countbits(%s))' % lo, '$ != 0'),
    's_bcnt1_i32_b32': (False, lambda lo, hi: 'countbits(%s)' % lo, '$ != 0'),
    's_bcnt0_i32_b64': (True, lambda lo, hi: '(64u - countbits(%s) - countbits(%s))' % (lo, hi), '$ != 0'),
    's_bcnt1_i32_b64': (True, lambda lo, hi: '(countbits(%s) + countbits(%s))' % (lo, hi), '$ != 0'),
    's_ff0_i32_b64': (True, lambda lo, hi: _ctz64('(~%s)' % lo, '(~%s)' % hi), None),
    # s_ff1_i32_b64 stays on Stage.op_s_ff1_i32_b64 (immediates + SGPR pairs) so SCALAR_BITS
    # does not overwrite the zero-extended immediate fold from #2022 Item F.
    's_flbit_i32_b64': (True, _clz64, None),
    's_flbit_i32_i64': (True, _flbit_i64, None),
}
for _name, (_pair, _expr, _scc) in SCALAR_BITS.items():
    def _make_scalar_bits(name=_name, pair=_pair, expr=_expr, scc=_scc):
        def handler(self, ops, mods):
            if len(ops) != 2 or mods:
                raise Unsupported('%s requires two unmodified operands' % name)
            dst, src = reg_range(ops[0]), reg_range(ops[1])
            if dst is None or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
                raise Unsupported('%s destination %s' % (name, ops[0]))
            if src is None or src[0] != 's' or src[2] > 105 or src[2] - src[1] != (1 if pair else 0):
                raise Unsupported('%s source %s' % (name, ops[1]))
            if any(r in self.masks for r in range(src[1], src[2] + 1)):
                raise Unsupported('%s on a lane mask' % name)
            lo, hi = self.S(src[1]), self.S(src[2])
            self.masks.discard(dst[1])
            self.set_s(dst[1], expr(lo, hi), scc=scc)
        return handler
    setattr(Stage, 'op_' + _name, _make_scalar_bits())
# s_bitcmp1_b32 / s_bitcmp0_b32: SCC = S0[S1[4:0]] == 1 / == 0.
for _bit in (0, 1):
    def _make_bitcmp(bit=_bit):
        def handler(self, ops, mods):
            if len(ops) != 2 or mods:
                raise Unsupported('scalar bit test requires two unmodified operands')
            if any(self.is_lane_mask(tok) for tok in ops):
                raise Unsupported('scalar bit test on a lane mask')
            self.set_scc('((%s >> (%s & 31)) & 1u) == %du' % (self.u(ops[0]), self.u(ops[1]), bit))
        return handler
    setattr(Stage, 'op_s_bitcmp%d_b32' % _bit, _make_bitcmp())


def scalar_wide_fold(helper, x, amount, signed):
    """Translation-time value of one SOP2 shift or bitfield extract (the helper name picks the operation)."""
    m64 = 0xffffffffffffffff
    if helper == 'shl64':
        return (x << (amount & 63)) & m64
    if helper == 'shr64u':
        return x >> (amount & 63)
    if helper == 'shr64i':
        return ((x - (1 << 64) if x >> 63 else x) >> (amount & 63)) & m64
    bits = 32 if helper == 'bfe32' else 64
    off = amount & (bits - 1)
    wd = min((amount >> 16) & 127, bits - off)  # bits past the source MSB do not exist
    field = (x >> off) & ((1 << wd) - 1)
    if signed and 0 < wd < bits and field >> (wd - 1) & 1:
        field |= ((1 << bits) - 1) ^ ((1 << wd) - 1)
    return 0 if signed and wd == 0 else field


def parse_root(blob):
    if not blob or blob[:4] != b'DXBC':
        raise Unsupported('root signature blob missing')

    def read(fmt, data, offset):
        if offset < 0 or offset + struct.calcsize(fmt) > len(data):
            raise Unsupported('root signature blob truncated')
        return struct.unpack_from(fmt, data, offset)

    n, = read('<I', blob, 28)
    for o in read('<%dI' % n, blob, 32):
        if blob[o:o + 4] == b'RTS0':
            d = blob[o + 8:]
            ver, nparam, poff, nsamp, soff, flags = read('<6I', d, 0)
            params = []
            for i in range(nparam):
                ptype, vis, off = read('<3I', d, poff + 12 * i)
                p = dict(type=ptype, vis=vis)
                if ptype == 1:
                    p['reg'], p['space'], p['count'] = read('<3I', d, off)
                params.append(p)
            return dict(flags=flags, params=params)
    raise Unsupported('no RTS0 part')


# Run-time sampler heap indexing: an s_load from the sampler heap with a register offset becomes a
# read of a bounded table, SamplerState qh[SAMPLER_HEAP_SIZE] : register(s0, space1), the form a
# desktop SM 5.1 / 6.0 shader can index at run time. Opt-in with X2W_SAMPLER_HEAP=1 until the
# stand-in binds that table; otherwise such a load keeps the default-sampler shortcut.
#
# Contract at the manifest boundary ('smpheap <count> <space> fallback <entry>'):
# - entries 0 .. SAMPLER_HEAP_FALLBACK - 1 hold the title's sampler-heap entries, entry n at byte
#   offset 16 * n;
# - entry SAMPLER_HEAP_FALLBACK is reserved: the host fills it with a fallback sampler;
# - the shader computes the entry itself (x2w_sampler_entry): a byte offset that is not a multiple
#   of 16, or that lies at or past 16 * SAMPLER_HEAP_FALLBACK, reads the fallback entry. The host
#   cannot check the offset, which is computed in the shader, so the guard is in the shader.
SAMPLER_HEAP_SIZE = 2048  # D3D12's limit for a shader-visible sampler heap
SAMPLER_HEAP_SPACE = 1
SAMPLER_HEAP_FALLBACK = SAMPLER_HEAP_SIZE - 1


# DXGI_FORMAT values for integer RTVs and the binary32 float targets.
UINT_RTVS = frozenset((3, 7, 12, 17, 25, 30, 36, 42, 50, 57, 62))
SINT_RTVS = frozenset((4, 8, 14, 18, 32, 38, 43, 52, 59, 64))
FLOAT32_RTVS = frozenset((2, 6, 16, 41))
# DXGI_FORMAT values independently verified exact for D16 image_load and image_load_mip, matching
# d16_srv_mode() in src/startup_stubs/d16_image_format.h (#2023): 32-bit float and 8-bit UNORM views (exact via
# pack16_rne) plus 16-bit UNORM views (exact via pack16_unorm16).
D16_FLOAT_EXACT_SRV_FORMATS = frozenset((2, 6, 16, 21, 28, 41, 49, 61, 65, 87, 88))
D16_UNORM16_SRV_FORMATS = frozenset((11, 35, 56))
D16_EXACT_SRV_FORMATS = D16_FLOAT_EXACT_SRV_FORMATS | D16_UNORM16_SRV_FORMATS
# Uncompressed exports keep their 32-bit shader values. Narrow integer RTVs clamp
# each channel to the destination range in the output merger.
VERIFIED_UINT_RTVS = frozenset((3, 12, 17, 25, 30, 36, 42, 50, 57, 62))
VERIFIED_SINT_RTVS = frozenset((4, 14, 18, 32, 38, 43, 52, 59, 64))


def unaligned_dword_body(low, high):
    """Read a little-endian dword spanning two aligned words."""
    return ('uint k = (a & 3) * 8, lo = %s; if (k == 0) return lo; '
            'return (lo >> k) | (%s << (32 - k)); }' % (low, high))


PLACEHOLDER_PS_PREFIX = 'placeholder ps '
PLACEHOLDER_DRAWN = ' (placeholder drawn)'


def placeholder_ps_enabled():
    return os.environ.get('X2W_PLACEHOLDER_PS') == '1'


def placeholder_rgb():
    rgb = os.environ.get('X2W_PLACEHOLDER_RGB', '255,0,255').split(',')
    if len(rgb) == 3 and all(re.fullmatch(r'[0-9]{1,3}', c.strip()) and int(c) <= 255 for c in rgb):
        return tuple(int(c) for c in rgb)
    return 255, 0, 255


def placeholder_manifest_rgb(path):
    with suppress(OSError), open(path, encoding='utf-8', errors='replace') as result:
        for line in result:
            if line.startswith('placeholder rgb '):
                return line.strip()[16:]
    return None


def placeholder_rgb_text():
    return '%d,%d,%d' % placeholder_rgb() if placeholder_ps_enabled() else None


def pixel_export_targets(code):
    """Read the output signature without lowering the shader body."""
    raw = strip_header(code)
    if len(raw) % 4:
        raise Unsupported('unaligned placeholder pixel program')
    ins = decode(raw, exports_only=True)
    if not ins or ins[-1][2] != 's_endpgm':
        raise Unsupported('placeholder pixel program has no end')
    targets = set()
    for _, _, name, asm in ins:
        if name != 'exp':
            continue
        target, vals, flags = export_operands(asm)
        if (len(vals) != 4 or any(not re.fullmatch(r'off|v[0-9]+', v) for v in vals)
                or any(f.startswith('en:') for f in flags)):
            raise Unsupported('unreadable placeholder pixel export')
        if target == 'null':
            continue
        if not re.fullmatch(r'mrt[0-7z]', target):
            raise Unsupported('unreadable placeholder pixel export target')
        if target != 'mrtz':
            targets.add(target)
        elif vals[0] != 'off':
            targets.add('depth')
    return targets


class PlaceholderPixelStage:
    attrs = {}
    f16_plain = {}
    write_masks = {}
    needs_sm6 = False

    def __init__(self, pipe, code, reason):
        self.exports = pixel_export_targets(code)
        self.reason = fail_reason(reason).splitlines()[0]
        self.rgb = placeholder_rgb()
        self.kinds = {name: pipe.mrt_kind(int(name[3:])) for name in sorted(self.exports) if name.startswith('mrt')}

    def hlsl(self):
        depth = 'depth' in self.exports
        if not self.exports:
            return 'void main() {}\n'
        arg = 'float4 pos : SV_Position' if depth else ''
        fields, values = [], []
        if depth:
            fields.append('    float depth : SV_Depth;')
            values.append('    o.depth = pos.z;')
        color = 'float4(%d.0/255.0, %d.0/255.0, %d.0/255.0, 1.0)' % self.rgb
        for name, kind in self.kinds.items():
            vector = {'uint': 'uint4', 'int': 'int4'}.get(kind, 'float4')
            fields.append('    %s %s : SV_Target%s;' % (vector, name, name[3:]))
            value = color if vector == 'float4' else '%s(1, 2, 3, 1)' % vector
            values.append('    o.%s = %s;' % (name, value))
        return '\n'.join(['struct PSOut {'] + fields + ['};', 'PSOut main(%s) {' % arg, '    PSOut o;']
                         + values + ['    return o;', '}']) + '\n'


class Pipeline:
    vertex_kind = 'vs'
    uses_dynamic_heap = False
    uses_pack16_rne = False
    uses_gather_2d_lod = False
    uses_cube_dir = False
    uses_store_d16_x = False
    uses_sad = ()
    uses_store_format_xyzw = False
    srv_formats = ()
    integer_texture_srvs = False
    uavs = None  # key -> dict(index, kind): images a pixel stage stores to (u registers); None until the first store

    def _init_pixel_context(self, root_blob, sampler_heap=None, rtv_formats=(), ps_hw_inputs=None, vs_fetch=None,
                            ps_attr_map=None, srv_formats=()):
        self.root = parse_root(root_blob)
        self.ps_attr_map = ps_attr_map  # stream_interface(): pixel attribute -> (vertex parameter, component mask); None pairs by number
        self.vs_fetch = vs_fetch  # stream_fetch(): the loads the vertex stage's entry call stands for
        self.packed_position_vgpr = None
        # False without SPI_PS_INPUT_ENA / ADDR: Stage.derive_packed_position then reads the register from the program.
        self.ps_inputs_known = ps_hw_inputs is not None
        # First VGPR of the PERSP_CENTER I/J entry pair (SPI_PS_INPUT_ADDR bit 1, after the PERSP_SAMPLE pair when that
        # one is enabled). Without hardware inputs the model assumes the center pair alone: v0, v1.
        self.persp_center_vgpr = 0
        if ps_hw_inputs is not None:
            # SPI_PS_INPUT_ADDR allocates the entry VGPRs in bit order; SPI_PS_INPUT_ENA chooses which of the
            # allocated inputs are loaded. Every enabled input is also allocated (LLVM AMDGPUAsmPrinter
            # getSIProgramInfo: "all enabled inputs are tagged in InputAddr as well"; SIISelLowering
            # processPSInputArgs marks an input allocated, then enabled only when it is used).
            ena, addr = ps_hw_inputs
            if not (0 <= ena <= 0xffff and 0 <= addr <= 0xffff):
                raise Unsupported('unverified pixel input enable/address combination')
            if ena & ~addr:
                raise Unsupported('contradictory pixel input enable/address masks: an enabled input is not allocated')
            # Layout from the address mask; an input allocated but not enabled holds no loaded value.
            self.persp_center_vgpr = (2 if addr & 1 else 0) if ena & 2 else None
            if ena & (1 << 15):
                # SPI PS inputs 0..6 supply IJ pairs (the pull model supplies three);
                # line stipple through sample coverage each occupy one VGPR.
                widths = (2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1)
                self.packed_position_vgpr = sum(w for bit, w in enumerate(widths) if addr & (1 << bit))
        self.rtv_formats = rtv_formats
        self.srv_formats = dict(srv_formats) if isinstance(srv_formats, dict) else dict(enumerate(srv_formats))
        self.sampler_heap = os.environ.get('X2W_SAMPLER_HEAP') == '1' if sampler_heap is None else sampler_heap
        self.uses_sampler_heap = False
        self.uses_pack16_rne = False
        self.uses_gather_2d_lod = False
        self.uses_store_d16_x = False
        self.uses_sad = set()
        self.uses_store_format_xyzw = False
        self.integer_texture_srvs = os.environ.get('X2W_INTEGER_TEXTURE_SRVS') == '1'
        self.resources = {}   # key -> dict(index, kind)
        self.samplers = {}    # key -> index
        self.uds = []         # entry-constant sources
        self.notes = set()    # shortcuts taken (written to the manifest as comments)
        self.lod_unvalidated = set()  # image_sample forms read through a heap sampler (LOD fields unchecked, #2221)
        self.modified, self.other_modified = set(), set()  # other: unproved descriptor writes

    def _build_pixel_stage(self, code, refusals=None, decoded=None):
        # Translation can allocate bindings before refusing. Restore the state left
        # by the vertex stage so the retry allocates only the bindings it uses.
        state = deepcopy(self.__dict__, {id(self.vs): self.vs} if hasattr(self, 'vs') else None)
        before = len(refusals) if refusals is not None else 0
        try:
            stage = Stage(self, 'ps', code, refusals=refusals, decoded=decoded)
        except Unsupported as reason:
            if str(reason) != READFIRSTLANE_UNIFORM:
                raise
        else:
            if refusals is None or (READFIRSTLANE_UNIFORM, 'ps') not in refusals[before:]:
                return stage
        self.__dict__.clear()
        self.__dict__.update(state)
        self.retry_firstlane = True
        if refusals is not None:
            del refusals[before:]
        return Stage(self, 'ps', code, refusals=refusals, decoded=decoded)

    def _graphics_pixel_stage(self, code, pixel_refusal):
        if code is None:
            return None
        if not placeholder_ps_enabled():
            return self._build_pixel_stage(code)
        state = deepcopy(self.__dict__, {id(self.vs): self.vs})
        try:
            if pixel_refusal is not None:
                raise pixel_refusal
            return self._build_pixel_stage(code)
        except Unsupported as reason:
            self.__dict__.clear()
            self.__dict__.update(state)
            try:
                return PlaceholderPixelStage(self, code, reason)
            except Unsupported:
                raise reason from None

    def __init__(self, vs_code, ps_code, root_blob, sampler_heap=None, rtv_formats=(), ps_hw_inputs=None, vs_fetch=None,
                 refusals=None, decoded=None, ps_attr_map=None, srv_formats=(), *, pixel_refusal=None):
        self._init_pixel_context(root_blob, sampler_heap, rtv_formats, ps_hw_inputs, vs_fetch,
                                 ps_attr_map, srv_formats)
        self.rect = self.ud(('rect',))
        if refusals is None:
            self.vs = Stage(self, self.vertex_kind, vs_code)
            self.ps = self._graphics_pixel_stage(ps_code, pixel_refusal)
        else:
            decoded = {} if decoded is None else decoded
            self.vs = Stage(self, self.vertex_kind, vs_code, refusals=refusals, decoded=decoded.get('vs'))
            self.ps = self._build_pixel_stage(ps_code, refusals, decoded.get('ps')) if ps_code is not None else None
        if self.ps is None:
            self.params, self.flat, self.f16_plain = [], [], []
            if 'pos' not in self.vs.exports:
                raise Unsupported('vertex stage exports no position')
            return
        self.params = sorted(int(k[5:]) for k in self.vs.exports if k.startswith('param'))
        self.flat = sorted(a for a, how in self.ps.attrs.items() if 'flat' in how)
        for a, how in self.ps.attrs.items():
            if 'f16' not in how:
                continue
            compr, plain = a in self.vs.compr_params, a in self.vs.plain_params
            if how != {'f16'} or not (compr or plain):
                raise Unsupported('attribute %d is read as binary16 by the pixel stage but not only so, or not exported '
                                  'by the vertex stage with compr or as a plain 32-bit export' % a)
            if compr and plain:
                raise Unsupported('attribute %d is exported both with compr and plain by the vertex stage: the two '
                                  'binary16 layouts differ' % a)
            if plain and self.ps.f16_plain.get(a, set()) & self.vs.plain_params[a]:
                raise Unsupported('attribute %d is read as binary16 from channels the vertex stage exports as off' % a)
        self.f16_plain = sorted(a for a in self.ps.f16_plain if a in self.vs.plain_params and a not in self.vs.compr_params)
        missing = [a for a in self.ps.attrs if a not in self.params]
        if missing:
            raise Unsupported('pixel stage reads attributes %s the vertex stage does not export' % missing)
        if 'pos' not in self.vs.exports:
            raise Unsupported('vertex stage exports no position')

    def mrt_kind(self, slot):
        if slot >= len(self.rtv_formats):
            return 'float'
        fmt = self.rtv_formats[slot]
        if fmt in UINT_RTVS:
            if fmt not in VERIFIED_UINT_RTVS:
                raise Unsupported('unverified UINT RTV format %d' % fmt)
            return 'uint'
        if fmt in SINT_RTVS:
            if fmt not in VERIFIED_SINT_RTVS:
                raise Unsupported('unverified SINT RTV format %d' % fmt)
            return 'int'
        return 'bits' if fmt in FLOAT32_RTVS else 'float'

    def ud(self, source):
        if source not in self.uds:
            self.uds.append(source)
        return self.uds.index(source)


    def resource(self, key, kind):
        if key[0][0] == 'dynamic_heap':
            raise Unsupported('dynamic resource heap views require a raw buffer load')
        if key in self.stored_images():
            raise Unsupported('resource %s is both read and stored to' % (key,))
        r = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        if r['kind'] != kind:
            raise Unsupported('resource %s used as %s and %s' % (key, r['kind'], kind))
        r['calls'] = r.get('calls', 0) + 1
        return 'r%d' % r['index']

    def note_use(self, key, use):
        """Says what the shader does with the read resource it just asked for through resource(): 'sample' (Sample,
        SampleBias, SampleLevel, SampleGrad), 'texel' (Load) or 'lod' (CalculateLevelOfDetail). Only feeds the manifest
        line `srvuse` (srv_use_line); the emitted HLSL does not depend on it."""
        r = self.resources[key]
        r.setdefault('uses', set()).add(use)
        r['classified'] = r.get('classified', 0) + 1

    @staticmethod
    def srv_use_line(r):
        """`srvuse N sample|texel|lod` for a read texture whose every use was classified, else None: a runtime that
        wants to change the texture (opt-in, dropping its top mip) reads no line as unknown and leaves it alone."""
        uses = r.get('uses')
        if r['kind'] == 'buf' or not uses or r.get('classified', 0) != r.get('calls', 0):
            return None
        word = 'texel' if 'texel' in uses else 'lod' if 'lod' in uses else 'sample'
        return 'srvuse %d %s' % (r['index'], word)

    def stored_images(self):
        return self.uavs or {}

    def uav(self, key, kind, rw=False):
        if key in self.resources:
            raise Unsupported('resource %s is both read and stored to' % (key,))
        if self.uavs is None:
            self.uavs = {}
        r = self.uavs.setdefault(key, dict(index=len(self.uavs), kind=kind))
        if r['kind'] != kind:
            raise Unsupported('stored image %s used as %s and %s' % (key, r['kind'], kind))
        return ('u%d' if kind == 'raw' else 'w%d') % r['index']

    def sampler(self, key, comparison=False, lod_limits=False):
        if not isinstance(key[0], int):
            raise Unsupported('sampler from a descriptor')
        # A comparison sampler, or one that carries the descriptor's LOD state, is its own binding
        # even when its words equal an ordinary sampler's.
        key = ('cmp',) + tuple(key) if comparison else ('lod',) + tuple(key) if lod_limits else key
        return 'q%d' % self.samplers.setdefault(key, len(self.samplers))

    def sampler_lines(self):
        """Emit static sampler words as smp, smpc or smpl. The smpl words carry limits and any validated bias."""
        lines = []
        for key, i in self.samplers.items():
            prefixed = key[0] in ('cmp', 'lod')
            lines.append('%s %d %s' % ({'cmp': 'smpc', 'lod': 'smpl'}.get(key[0], 'smp'), i,
                                       ' '.join('0x%x' % w for w in key[prefixed:])))
        return lines

    def integer_srv_bindings(self):
        inputs = [r for r in self.resources.values() if r.get('integer_load')]
        if len(self.uds) + len(inputs) > 32:
            raise Unsupported('integer texture loads exceed the 32 entry values')
        for ordinal, res in enumerate(inputs):
            yield res, len(self.resources) + 2 * ordinal, len(self.uds) + ordinal

    def integer_srv_manifest(self):
        if self.integer_texture_srvs and any(r['kind'] != 'buf' for r in self.resources.values()):
            yield 'srvtypes 1'
            for res, reg, entry in self.integer_srv_bindings():
                yield 'srvint %d %d %d' % (res['index'], reg, entry)

    def common(self, *, embedded_vertex=False):
        import pixel_aot
        twins = list(self.integer_srv_bindings())
        out = ['#define F(x) asfloat(x)', '#define U(x) asuint(x)', FLOAT_MINMAX,
               'float mul_legacy(float a, float b) { return (a == 0.0 || b == 0.0) ? 0.0 : a * b; }',
               # Match pixel_aot: round the product before adding and flush denormals to signed zero.
               'float ftz32(float value) {',
               '    uint bits = asuint(value);',
               # Select the mask as an integer so FXC preserves the sign of a flushed zero.
               '    uint mask = (bits & 0x7fffffffu) < 0x00800000u ? 0x80000000u : 0xffffffffu;',
               '    return asfloat(bits & mask);',
               '}',
               'float mad32(float a, float b, float c, bool legacy) {',
               '    a = ftz32(a);',
               '    b = ftz32(b);',
               '    precise float product = ftz32(legacy ? mul_legacy(a, b) : a * b);',
               '    precise float sum = product + ftz32(c);',
               '    return ftz32(sum);',
               '}',
               # Match pixel_aot: x - floor(x) capped at 0x3f7fffff, so a tiny negative input cannot round up to
               # 1.0. The cap is a comparison, not min: min returns its other operand for a NaN.
               # The comparison is on the bits because FXC folds `frac(value) >= 1.0f` to false and emits a bare
               # frc for it, which loses the cap on an adapter whose frac returns 1.0. frac can only reach the cap
               # by rounding to exactly +1.0, so the equality covers the same case, and a NaN misses it and passes.
               'float fract32(float value) {',
               '    float f = frac(value);',
               '    return asuint(f) == 0x3f800000u ? asfloat(0x3f7fffffu) : f;',
               '}',
               # v_med3_f32 with a NaN source is the minimum of the three (ISA "343 V_MED3_F32"), so the test is
               # on the encoding: a float isnan() may be folded away by the compiler. D3D min() returns the
               # other operand, so the min3 chain over a NaN source is the smallest of the non-NaN ones. min, max
               # and med3 go through minmax_zero32, which orders -0 below +0 and passes a NaN to D3D min/max.
               # Same body as med3_32 in tools/pixel_aot.py.
               'float med3_32(float a, float b, float c) {',
               '    uint top = max(max(asuint(a) & 0x7fffffffu, asuint(b) & 0x7fffffffu), asuint(c) & 0x7fffffffu);',
               '    return top > 0x7f800000u',
               '        ? minmax_zero32(minmax_zero32(a, b, false), c, false)',
               '        : minmax_zero32(minmax_zero32(a, b, false), minmax_zero32(minmax_zero32(a, b, true), c, false), true);',
               '}',
               # High half of an unsigned 32x32 product, by halfword products so it fits 32-bit HLSL.
               'uint mul_hi_u32(uint a, uint b) { uint la = a & 0xffffu, lb = b & 0xffffu;'
               ' uint middle = (a >> 16) * lb + ((la * lb) >> 16);'
               ' return (a >> 16) * (b >> 16) + (middle >> 16) + (((middle & 0xffffu) + la * (b >> 16)) >> 16); }',
               'uint mul_hi_i32(uint a, uint b) { return mul_hi_u32(a, b) - ((a >> 31) != 0u ? b : 0u)'
               ' - ((b >> 31) != 0u ? a : 0u); }',
               # a * b + c for binary16 inputs with one rounding (#2073). The first form recovered the binary32 sum error
               # with a floating-point TwoSum; an NVIDIA driver (RTX 3070 Ti, 32.0.16.2101) simplified that error term to
               # zero despite `precise`, so ties rounded the wrong way. fma16_rne works on the encodings in 32-bit integer
               # arithmetic, so no driver floating-point rewrite can reach it. Infinities and NaNs keep float semantics.
               pixel_aot.SALU_HELPERS['fma16_rne'],
               # A binary16 product is exact in binary32. Keep its sign when the result is zero (#2746).
               'float mul16(float a, float b) { precise float product = a * b; return product; }',
               MIX32,
               'float fma16o(float a, float b, float c) {',
               '    if (!isfinite(a) || !isfinite(b) || !isfinite(c)) return a * b + c;',
               '    return f16tof32(fma16_rne(f32tof16(a), f32tof16(b), f32tof16(c)));',
               '}',
               # binary32 to binary16 rounded to nearest even: f32tof16 truncates, a binary16 result must round (#1723).
               'uint f16rn(float v) { uint b = asuint(v), s = (b >> 16) & 0x8000u, a = b & 0x7fffffffu;'
               ' if (a > 0x7f800000u) return s | 0x7e00u; if (a >= 0x47800000u) return s | 0x7c00u;'
               ' if (a < 0x38800000u) return s | uint(round(asfloat(a) * 16777216.0));'
               ' uint r = a + 0xfffu + ((a >> 13) & 1u); return s | min((r - 0x38000000u) >> 13, 0x7c00u); }',
               # The ten binary32 classes v_cmp_class_f32 tests, from the raw encoding (#1795).
               'uint class32(float value) { uint bits = asuint(value), magnitude = bits & 0x7fffffffu;'
               ' bool negative = bits >= 0x80000000u;'
               ' if (magnitude > 0x7f800000u) return (bits & 0x00400000u) != 0u ? 0x2u : 0x1u;'
               ' if (magnitude == 0x7f800000u) return negative ? 0x4u : 0x200u;'
               ' if (magnitude >= 0x00800000u) return negative ? 0x8u : 0x100u;'
               ' if (magnitude != 0u) return negative ? 0x10u : 0x80u;'
               ' return negative ? 0x20u : 0x40u; }',
               # The ten binary16 classes v_cmp_class_f16 tests, from the raw encoding (issue #1855):
               # signaling NaN, quiet NaN, -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf.
               'uint class16(uint bits) { uint h = bits & 0xffffu, magnitude = h & 0x7fffu;'
               ' bool negative = h >= 0x8000u;'
               ' if (magnitude > 0x7c00u) return (h & 0x0200u) != 0u ? 0x2u : 0x1u;'
               ' if (magnitude == 0x7c00u) return negative ? 0x4u : 0x200u;'
               ' if (magnitude >= 0x0400u) return negative ? 0x8u : 0x100u;'
               ' if (magnitude != 0u) return negative ? 0x10u : 0x80u;'
               ' return negative ? 0x20u : 0x40u; }',
               # The 64-bit shifts (issue #1815): a pair of words, so the count never reaches 32 in a 32-bit
               # shift and an in-place destination keeps the original value (the caller passes both words in).
               'uint2 shl64(uint lo, uint hi, uint sh) { sh &= 63u; uint sb = sh & 31u, sc = (32u - sb) & 31u;'
               ' uint cy = sb == 0u ? 0u : (lo >> sc);'
               ' return uint2(sh < 32u ? (lo << sb) : 0u, sh < 32u ? ((hi << sb) | cy) : (lo << sb)); }',
               'uint2 shr64u(uint lo, uint hi, uint sh) { sh &= 63u; uint sb = sh & 31u, sc = (32u - sb) & 31u;'
               ' uint cy = sb == 0u ? 0u : (hi << sc); uint h = hi >> sb;'
               ' return uint2(sh < 32u ? ((lo >> sb) | cy) : h, sh < 32u ? h : 0u); }',
               'uint2 shr64i(uint lo, uint hi, uint sh) { sh &= 63u; uint sb = sh & 31u, sc = (32u - sb) & 31u;'
               ' uint cy = sb == 0u ? 0u : (hi << sc); uint h = asuint(asint(hi) >> sb);'
               ' return uint2(sh < 32u ? ((lo >> sb) | cy) : h, sh < 32u ? h : asuint(asint(hi) >> 31u)); }',
               # v_mad_u64_u32 / v_mad_i64_i32: a * b + (chi:clo) from 16-bit partial products, so no 64-bit type.
               # z is the carry of the 64-bit add (the unsigned form's lane-mask carry-out).
               'uint mulhi32(uint a, uint b) {'
               ' uint a0 = a & 0xffffu, a1 = a >> 16, b0 = b & 0xffffu, b1 = b >> 16;'
               ' uint p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;'
               ' uint mid = (p00 >> 16) + (p01 & 0xffffu) + (p10 & 0xffffu);'
               ' return p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16); }',
               'uint3 mad64u(uint a, uint b, uint clo, uint chi) {'
               ' uint ph = mulhi32(a, b), pl = a * b, lo = pl + clo, c0 = lo < pl ? 1u : 0u, hs = ph + chi, hi = hs + c0;'
               ' return uint3(lo, hi, (hs < ph || hi < hs) ? 1u : 0u); }',
               'uint3 mad64i(uint a, uint b, uint clo, uint chi) {'
               ' uint ph = mulhi32(a, b) - (asint(a) < 0 ? b : 0u) - (asint(b) < 0 ? a : 0u), pl = a * b, lo = pl + clo;'
               ' return uint3(lo, ph + chi + (lo < pl ? 1u : 0u), 0u); }',
               'uint2 bfe64(uint lo, uint hi, uint s1, uint sg) { uint w = min((s1 >> 16u) & 127u, 64u - (s1 & 63u)); uint2 e = shr64u(lo, hi, s1);'
               ' e = uint2(e.x & (w >= 32u ? 0xffffffffu : ((1u << (w & 31u)) - 1u)),'
               ' e.y & (w >= 64u ? 0xffffffffu : (w > 32u ? ((1u << ((w - 32u) & 31u)) - 1u) : 0u)));'
               ' uint sh = (64u - w) & 63u; uint2 t = shl64(e.x, e.y, sh);'
               ' return sg != 0u ? shr64i(t.x, t.y, sh) : e; }',
               'uint bfe32(uint x, uint s1, uint sg) { uint w = min((s1 >> 16u) & 127u, 32u - (s1 & 31u));'
               ' uint e = (x >> (s1 & 31u)) & (w >= 32u ? 0xffffffffu : ((1u << (w & 31u)) - 1u));'
               ' uint sh = (32u - w) & 31u; return sg != 0u ? asuint(asint(e << sh) >> sh) : e; }',
               'uint ldu(ByteAddressBuffer b, uint a) { ' +
               unaligned_dword_body('b.Load(a & ~3u)', 'b.Load((a & ~3u) + 4)'),
               'cbuffer entry : register(b0) { %s };' % ' '.join('uint ud%d;' % i for i in range(len(self.uds) + len(twins)))]
        if self.uses_dynamic_heap:
            constants = ['uint ud%d;' % i for i in range(len(self.uds) + len(twins))]
            out[-1] = 'cbuffer entry : register(b0) { %s };' % ' '.join(
                constants + ['uint heap_count;'])
            out += ['uint x2w_heap_load(uint index, uint address) {',
                    '    if (index >= heap_count) return 0u;',
                    '    ByteAddressBuffer buffer = ResourceDescriptorHeap[NonUniformResourceIndex(index)];',
                    '    return buffer.Load(address);',
                    '}']
        for r, reg, entry in twins:
            kind = {'tex1d': 'Texture1D', 'tex2d': 'Texture2D', 'tex2da': 'Texture2DArray', 'tex3d': 'Texture3D'}[r['kind']]
            out.append('%s<uint4> r%d_uint : register(t%d);' % (kind, r['index'], reg))
            out.append('%s<int4> r%d_sint : register(t%d);' % (kind, r['index'], reg + 1))
            out.append('#define r%d_is_integer ud%d' % (r['index'], entry))
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            out.append('%s r%d : register(t%d);' % ({'buf': 'ByteAddressBuffer', 'tex1d': 'Texture1D<float4>', 'tex2d': 'Texture2D<float4>', 'tex2da': 'Texture2DArray<float4>', 'tex3d': 'Texture3D<float4>', 'texcube': 'TextureCube<float4>'}[r['kind']], r['index'], r['index']))
        for key, r in sorted(self.stored_images().items(), key=lambda kv: kv[1]['index']):
            out.append('%s %s%d : register(u%d);' % ({'tex1d': 'RWTexture1D<float4>', 'tex2d': 'RWTexture2D<float4>', 'tex2da': 'RWTexture2DArray<float4>', 'tex3d': 'RWTexture3D<float4>', 'raw': 'RWByteAddressBuffer'}[r['kind']], 'u' if r['kind'] == 'raw' else 'w', r['index'], r['index']))
        for key, i in self.samplers.items():
            out.append('%s q%d : register(s%d);' % ('SamplerComparisonState' if key[0] == 'cmp' else 'SamplerState', i, i))
        if self.uses_sampler_heap:
            out.append('SamplerState qh[%d] : register(s0, space%d);' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE))
            out.append('uint x2w_sampler_entry(uint offset) { return (offset & 15u) != 0 || offset >= %du ? %du : offset >> 4; }'
                       % (16 * SAMPLER_HEAP_FALLBACK, SAMPLER_HEAP_FALLBACK))
        if self.uses_pack16_rne:
            # IEEE binary32 to binary16 round-to-nearest-even (quieting NaN while preserving its sign,
            # matching RDNA 2 hardware on binary32 inputs), plus exact UNORM16 ratio-to-half conversion
            # (avoiding binary32 double-rounding at 0xffaf -> 0x3bfd and 0xffef -> 0x3bff).
            out.extend((
                'uint pack16_rne(float value) {',
                '    uint bits = asuint(value), sign = (bits >> 16) & 0x8000u, abs_bits = bits & 0x7fffffffu;',
                '    if (abs_bits > 0x7f800000u) return sign | 0x7e00u;',
                '    if (abs_bits >= 0x477ff000u) return sign | 0x7c00u;',
                '    if (abs_bits >= 0x38800000u) {',
                '        uint shifted = abs_bits - 0x38000000u;',
                '        return sign | ((shifted + 0x00000fffu + ((shifted >> 13) & 1u)) >> 13);',
                '    }',
                '    if (abs_bits < 0x33000000u) return sign;',
                '    uint sig = 0x00800000u | (abs_bits & 0x007fffffu), shift = 126u - (abs_bits >> 23);',
                '    uint half_m = sig >> shift, rem = sig & ((1u << shift) - 1u), mid = 1u << (shift - 1u);',
                '    if (rem > mid || (rem == mid && (half_m & 1u) != 0u)) half_m += 1u;',
                '    return sign | half_m;',
                '}',
                'uint pack16_unorm16(float value) {',
                '    uint u = uint(saturate(value) * 65535.0 + 0.5);',
                '    if (u <= 3u) return u << 8;',
                '    if (u >= 65535u) return 0x3c00u;',
                '    uint msb = firstbithigh(u);',
                '    uint num = u << (26u - msb);',
                '    uint q = num / 65535u;',
                '    if (num - q * 65535u > 32767u) q += 1u;',
                '    return ((msb - 1u) << 10) + (q - 1024u);',
                '}',
                'uint pack16_d16(float value, uint unorm16) {',
                '    return unorm16 != 0u ? pack16_unorm16(value) : pack16_rne(value);',
                '}',
            ))
        if getattr(self, 'uses_cube_dir', False):
            # The inverse of v_cubeid / v_cubesc / v_cubetc / v_cubema (LLVM ConstantFoldAMDGCNCubeIntrinsic, the same table
            # as Stage.cube): the face index and S = sc / (2 |major|) + 1.5, T = tc / (2 |major|) + 1.5 give the direction.
            out.extend((
                'float3 x2w_cube_dir(float s, float t, float face) {',
                '    float sc = 2.0f * s - 3.0f, tc = 2.0f * t - 3.0f;',
                '    uint f = (uint)clamp(floor(face + 0.5f), 0.0f, 5.0f);',
                '    if (f == 0u) return float3(1.0f, -tc, -sc);',
                '    if (f == 1u) return float3(-1.0f, -tc, sc);',
                '    if (f == 2u) return float3(sc, 1.0f, tc);',
                '    if (f == 3u) return float3(sc, -1.0f, -tc);',
                '    if (f == 4u) return float3(sc, -tc, 1.0f);',
                '    return float3(-sc, -tc, -1.0f);',
                '}'))
        if self.uses_gather_2d_lod:
            out.extend((
                'float4 gather_2d_lod(Texture2D<float4> tex, float2 uv, float lod, uint component, float max_lod) {',
                '    float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);',
                '    if ((asuint(uv.x) & 0x7f800000u) != 0x7f800000u &&',
                '        (asuint(uv.y) & 0x7f800000u) != 0x7f800000u &&',
                '        (asuint(lod) & 0x7f800000u) != 0x7f800000u) {',
                '        uint width, height, levels;',
                '        tex.GetDimensions(0u, width, height, levels);',
                '        if (levels != 0u) {',
                '            uint mip = min(uint(floor(clamp(lod, 0.0f, max_lod) + 0.5f)), levels - 1u);',
                '            tex.GetDimensions(mip, width, height, levels);',
                '            int2 edge = int2(width, height) - 1;',
                '            precise float2 scaled = uv * float2(width, height) - 0.5f;',
                '            int2 origin = int2(floor(clamp(scaled, float2(-1.0f, -1.0f), float2(width, height))));',
                '            result = float4(tex.Load(int3(clamp(origin + int2(0, 1), int2(0, 0), edge), int(mip)))[component],',
                '                            tex.Load(int3(clamp(origin + int2(1, 1), int2(0, 0), edge), int(mip)))[component],',
                '                            tex.Load(int3(clamp(origin + int2(1, 0), int2(0, 0), edge), int(mip)))[component],',
                '                            tex.Load(int3(clamp(origin,              int2(0, 0), edge), int(mip)))[component]);',
                '        }',
                '    }',
                '    return result;',
                '}',
            ))
        if self.uses_store_d16_x:
            out.extend((
                'uint d16_to_unorm16(uint h) {',
                '    float x = f16tof32(h);',
                '    if (isnan(x)) return 0u;',
                '    precise float saturated = saturate(x);',
                '    precise float scaled = saturated * 65535.0f;',
                '    return (uint)floor(scaled + 0.5f);',
                '}',
                'uint d16_to_snorm16(uint h) {',
                '    float x = f16tof32(h);',
                '    if (isnan(x)) return 0u;',
                '    precise float saturated = clamp(x, -1.0f, 1.0f);',
                '    precise float scaled = saturated * 32767.0f;',
                '    precise float rounded = scaled >= 0.0f ? floor(scaled + 0.5f) : ceil(scaled - 0.5f);',
                '    return ((uint)(int)rounded) & 0xffffu;',
                '}',
                'void store_format_d16_x(RWByteAddressBuffer buf, uint word3, uint stride, uint num_records, '
                'uint idx, uint voff, uint soff, uint imm_off, uint val) {',
                '    if ((word3 & 0xcef80000u) != 0u || (word3 & 7u) != 4u) return;',
                '    uint oob_sel = (word3 >> 28u) & 3u;',
                '    if ((oob_sel != 0u && oob_sel != 3u) || stride > 0x3fffu) return;',
                '    uint field = (word3 >> 12u) & 127u;',
                '    uint fmt = 0u;',
                '    if (((word3 >> 24u) & 1u) != 0u) {',
                '        fmt = field;',
                '    } else {',
                '        if (field == 16u) fmt = 7u;',
                '        else if (field == 17u) fmt = 8u;',
                '        else if (field == 20u) fmt = 11u;',
                '        else if (field == 21u) fmt = 12u;',
                '        else if (field == 23u) fmt = 13u;',
                '        else if (field == 39u) fmt = 22u;',
                '    }',
                '    uint elem_bytes = fmt == 22u ? 4u : (fmt == 7u || fmt == 8u || fmt == 11u || fmt == 12u || fmt == 13u ? 2u : 0u);',
                '    if (elem_bytes == 0u) return;',
                '    if (voff > 0xffffffffu - imm_off) return;',
                '    uint rec_off = voff + imm_off;',
                '    uint addr = 0u;',
                '    if (oob_sel == 3u) {',
                '        if (stride != 0u) return;',
                '        if (soff > 0xffffffffu - rec_off) return;',
                '        addr = soff + rec_off;',
                '        if (elem_bytes > num_records || addr > num_records - elem_bytes) return;',
                '    } else {',
                '        if (stride == 0u) return;',
                '        if (idx >= num_records || elem_bytes > stride || rec_off > stride - elem_bytes) return;',
                '        if (idx > (0xffffffffu - rec_off) / stride) return;',
                '        uint elem_addr = idx * stride + rec_off;',
                '        if (soff > 0xffffffffu - elem_addr) return;',
                '        addr = elem_addr + soff;',
                '    }',
                '    uint uav_bytes = 0u;',
                '    buf.GetDimensions(uav_bytes);',
                '    uint h = val & 0xffffu;',
                '    if (fmt == 22u) {',
                '        if ((addr & 3u) != 0u || uav_bytes < 4u || addr > uav_bytes - 4u) return;',
                '        buf.Store(addr, asuint(f16tof32(h)));',
                '        return;',
                '    }',
                '    if ((addr & 1u) != 0u) return;',
                '    uint out16 = h;',
                '    if (fmt == 7u) out16 = d16_to_unorm16(h);',
                '    else if (fmt == 8u) out16 = d16_to_snorm16(h);',
                '    uint base = addr & 0xfffffffcu;',
                '    if (uav_bytes < 4u || base > uav_bytes - 4u) return;',
                '    uint shift = (addr & 2u) << 3u;',
                '    uint keep = ~(0xffffu << shift);',
                '    uint ins = (out16 & 0xffffu) << shift;',
                '    uint cur = buf.Load(base);',
                '    for (;;) {',
                '        uint next = (cur & keep) | ins;',
                '        uint prev = 0u;',
                '        buf.InterlockedCompareExchange(base, cur, next, prev);',
                '        if (prev == cur) break;',
                '        cur = prev;',
                '    }',
                '}',
            ))
        if getattr(self, 'uses_pknorm_u16', False):
            # Compute the 40-bit significand product as (m << 16) - m, then divide
            # by the power of two with one nearest-even rounding.
            out.extend((
                'uint pknorm_u16(uint bits) {',
                '    if (bits > 0x7f800000u || bits < 0x37000000u) return 0u;',
                '    if (bits >= 0x3f800000u) return 65535u;',
                '    uint m = 0x800000u | (bits & 0x7fffffu);',
                '    uint lo = (m << 16u) - m;',
                '    uint hi = (m >> 16u) - uint((m << 16u) < m);',
                '    uint shift = 150u - (bits >> 23u);',
                '    uint q, guard; bool sticky;',
                '    if (shift > 32u) {',
                '        uint k = shift - 32u;',
                '        q = hi >> k;',
                '        guard = (hi >> (k - 1u)) & 1u;',
                '        sticky = lo != 0u || (hi & ((1u << (k - 1u)) - 1u)) != 0u;',
                '    } else {',
                '        q = shift == 32u ? hi : (hi << (32u - shift)) | (lo >> (shift & 31u));',
                '        guard = (lo >> (shift - 1u)) & 1u;',
                '        sticky = (lo & ((1u << (shift - 1u)) - 1u)) != 0u;',
                '    }',
                '    return q + (guard & uint(sticky || (q & 1u) != 0u));',
                '}',
            ))
        if getattr(self, 'uses_pknorm_i16', False):
            # RDNA 2 SNORM: round the exact significand product before applying the sign.
            # (m << 15) - m spans 39 bits. Split it into two uints for SM 5.1.
            out.extend((
                'uint pknorm_i16(uint bits) {',
                '    uint mag = bits & 0x7fffffffu;',
                '    if (mag > 0x7f800000u || mag < 0x37800000u) return 0u;',
                '    uint packed = 32767u;',
                '    if (mag < 0x3f800000u) {',
                '        uint m = 0x800000u | (mag & 0x7fffffu);',
                '        uint lo = (m << 15u) - m;',
                '        uint hi = (m >> 17u) - uint((m << 15u) < m);',
                '        uint shift = 150u - (mag >> 23u);',
                '        uint q, guard; bool sticky;',
                '        if (shift > 32u) {',
                '            uint k = shift - 32u;',
                '            q = hi >> k;',
                '            guard = (hi >> (k - 1u)) & 1u;',
                '            sticky = lo != 0u || (hi & ((1u << (k - 1u)) - 1u)) != 0u;',
                '        } else {',
                '            q = shift == 32u ? hi : (hi << (32u - shift)) | (lo >> (shift & 31u));',
                '            guard = (lo >> (shift - 1u)) & 1u;',
                '            sticky = (lo & ((1u << (shift - 1u)) - 1u)) != 0u;',
                '        }',
                '        packed = q + (guard & uint(sticky || (q & 1u) != 0u));',
                '    }',
                '    return ((bits >> 31u) != 0u ? 0u - packed : packed) & 0xffffu;',
                '}',
            ))
        if getattr(self, 'uses_ldexp', False):
            # Integer arithmetic on the binary32 fields (sign, biased exponent, 24-bit significand): x * 2^n with one
            # rounding to nearest even. The result exponent is e + n; at 255 or more it overflows to infinity, below 1
            # the significand is shifted right by 1 - E bits and rounded (a carry into bit 23 is the smallest normal),
            # and a shift past 25 bits is zero. A denormal source is normalised first. No float multiply is involved,
            # so denormal results and the sign of a zero survive compilers that flush or fold them.
            out.extend((
                'uint x2w_ldexp(uint b, int n) {',
                '    uint sign = b & 0x80000000u, m = b & 0x7fffffu;',
                '    int e = int((b >> 23) & 0xffu);',
                '    if (e == 255) return m != 0u ? (b | 0x00400000u) : b;',
                '    if (e == 0) {',
                '        if (m == 0u) return b;',
                '        int s = 23 - int(firstbithigh(m));',
                '        m <<= uint(s); e = 1 - s;',
                '    } else {',
                '        m |= 0x00800000u;',
                '    }',
                '    int E = e + clamp(n, -1000, 1000);',
                '    if (E >= 255) return sign | 0x7f800000u;',
                '    if (E >= 1) return sign | (uint(E) << 23) | (m & 0x7fffffu);',
                '    int sh = 1 - E;',
                '    if (sh > 25) return sign;',
                '    uint keep = m >> uint(sh), rem = m & ((1u << uint(sh)) - 1u), half = 1u << uint(sh - 1);',
                '    if (rem > half || (rem == half && (keep & 1u) != 0u)) keep += 1u;',
                '    return sign | keep;',
                '}',
            ))
        if 'v_sad_u8' in self.uses_sad:
            out.append(
                'uint sad_u8(uint a, uint b, uint c) {'
                ' return uint(abs(int(a & 0xffu) - int(b & 0xffu)))'
                ' + uint(abs(int((a >> 8) & 0xffu) - int((b >> 8) & 0xffu)))'
                ' + uint(abs(int((a >> 16) & 0xffu) - int((b >> 16) & 0xffu)))'
                ' + uint(abs(int((a >> 24) & 0xffu) - int((b >> 24) & 0xffu))) + c; }')
        if 'v_sad_u16' in self.uses_sad:
            out.append(
                'uint sad_u16(uint a, uint b, uint c) {'
                ' return uint(abs(int(a & 0xffffu) - int(b & 0xffffu)))'
                ' + uint(abs(int(a >> 16) - int(b >> 16))) + c; }')
        if 'v_sad_u32' in self.uses_sad:
            out.append(
                'uint sad_u32(uint a, uint b, uint c) {'
                ' return (a > b ? a - b : b - a) + c; }')
        # v_sad_hi_u8 and v_msad_u8 share the four-lane absolute-difference sum of sad_u8, so it is factored into
        # sad4_u8, whose `masked` argument leaves out each byte lane whose src1 byte is zero (v_msad_u8's reference
        # rule). The table formulas match tools/pixel_aot.py exactly: sad_hi_u8 is (sad4_u8(a, b, false) << 16) + c
        # and msad_u8 is sad4_u8(a, b, true) + c, both modulo 2^32. sad_u8 above keeps its inlined four-term form so
        # a translation that uses only v_sad_u8 is unchanged.
        if 'v_sad_hi_u8' in self.uses_sad or 'v_msad_u8' in self.uses_sad:
            out.append(
                'uint sad4_u8(uint a, uint b, bool masked) {'
                ' return (masked && ((b & 0xffu) == 0u) ? 0u : uint(abs(int(a & 0xffu) - int(b & 0xffu))))'
                ' + (masked && (((b >> 8) & 0xffu) == 0u) ? 0u : uint(abs(int((a >> 8) & 0xffu) - int((b >> 8) & 0xffu))))'
                ' + (masked && (((b >> 16) & 0xffu) == 0u) ? 0u : uint(abs(int((a >> 16) & 0xffu) - int((b >> 16) & 0xffu))))'
                ' + (masked && (((b >> 24) & 0xffu) == 0u) ? 0u : uint(abs(int((a >> 24) & 0xffu) - int((b >> 24) & 0xffu)))); }')
        if 'v_sad_hi_u8' in self.uses_sad:
            out.append(
                'uint sad_hi_u8(uint a, uint b, uint c) { return (sad4_u8(a, b, false) << 16u) + c; }')
        if 'v_msad_u8' in self.uses_sad:
            out.append(
                'uint msad_u8(uint a, uint b, uint c) { return sad4_u8(a, b, true) + c; }')
        if self.uses_store_format_xyzw:
            legacy = ['        %sif (field == %du) fmt = %du;' % ('' if i == 0 else 'else ', field, code)
                      for i, (field, code) in enumerate(sorted(STORE_FORMAT_XYZW_LEGACY_FORMATS.items()))]
            out.extend((
                'uint store_unorm(uint bits, float scale) {',
                '    float x = asfloat(bits);',
                '    if (isnan(x)) return 0u;',
                '    precise float saturated = saturate(x);',
                '    precise float scaled = saturated * scale;',
                '    return (uint)floor(scaled + 0.5f);',
                '}',
                'void store_format_xyzw(RWByteAddressBuffer buf, uint word3, uint stride, uint num_records, '
                'uint idx, uint voff, uint soff, uint imm_off, uint4 v) {',
                '    if ((word3 & 0xcef80000u) != 0u || (word3 & 0xfffu) != 0xfacu) return;',
                '    uint oob_sel = (word3 >> 28u) & 3u;',
                '    if ((oob_sel != 0u && oob_sel != 3u) || stride > 0x3fffu) return;',
                '    uint field = (word3 >> 12u) & 127u;',
                '    uint fmt = 0u;',
                '    if (((word3 >> 24u) & 1u) != 0u) {',
                '        fmt = field;',
                '    } else {',
                *legacy,
                '    }',
                '    uint elem_bytes = (fmt == 75u || fmt == 77u) ? 16u : fmt == 71u ? 8u : (fmt == 50u || fmt == 56u) ? 4u : 0u;',
                '    if (elem_bytes == 0u) return;',
                '    if (voff > 0xffffffffu - imm_off) return;',
                '    uint rec_off = voff + imm_off;',
                '    uint addr = 0u;',
                '    if (oob_sel == 3u) {',
                '        if (stride != 0u) return;',
                '        if (soff > 0xffffffffu - rec_off) return;',
                '        addr = soff + rec_off;',
                '        if (elem_bytes > num_records || addr > num_records - elem_bytes) return;',
                '    } else {',
                '        if (stride == 0u) return;',
                '        if (idx >= num_records || elem_bytes > stride || rec_off > stride - elem_bytes) return;',
                '        if (idx > (0xffffffffu - rec_off) / stride) return;',
                '        uint elem_addr = idx * stride + rec_off;',
                '        if (soff > 0xffffffffu - elem_addr) return;',
                '        addr = elem_addr + soff;',
                '    }',
                '    if ((addr & 3u) != 0u) return;',
                '    uint uav_bytes = 0u;',
                '    buf.GetDimensions(uav_bytes);',
                '    if (uav_bytes < elem_bytes || addr > uav_bytes - elem_bytes) return;',
                '    if (elem_bytes == 16u) {',
                '        buf.Store4(addr, v);',
                '    } else if (fmt == 71u) {',
                '        buf.Store2(addr, uint2(pack16_rne(asfloat(v.x)) | (pack16_rne(asfloat(v.y)) << 16u), '
                'pack16_rne(asfloat(v.z)) | (pack16_rne(asfloat(v.w)) << 16u)));',
                '    } else if (fmt == 56u) {',
                '        buf.Store(addr, store_unorm(v.x, 255.0f) | (store_unorm(v.y, 255.0f) << 8u) '
                '| (store_unorm(v.z, 255.0f) << 16u) | (store_unorm(v.w, 255.0f) << 24u));',
                '    } else {',
                '        buf.Store(addr, store_unorm(v.x, 1023.0f) | (store_unorm(v.y, 1023.0f) << 10u) '
                '| (store_unorm(v.z, 1023.0f) << 20u) | (store_unorm(v.w, 3.0f) << 30u));',
                '    }',
                '}',
            ))
        embedded = self.vs.embedded_buffer if embedded_vertex else None
        if embedded:
            _, _, size, words = embedded
            out += [
                'static const uint embedded_vs[%d] = { %s };' %
                (len(words), ', '.join('0x%08xu' % word for word in words)),
                'uint embedded_vs_load(uint lane_offset, uint scalar_offset, uint imm) {',
                '    const uint bound = %du;' % size,
                '    if (bound < 4u || imm > bound - 4u || scalar_offset > bound - 4u - imm || '
                'lane_offset > bound - 4u - imm - scalar_offset) return 0u;',
                '    uint a = lane_offset + scalar_offset + imm;',
                '    uint i = a >> 2u;',
                '    ' + unaligned_dword_body('embedded_vs[i]', 'embedded_vs[i + 1u]'),
            ]
        return out

    def interface(self, for_pixel):
        fields = []
        for p in range(max(self.params) + 1 if self.params else 0):
            fields.append('    float4 param%d : PARAM%d;' % (p, p))
        # Attributes read without interpolation get a second, flat copy of the vertex output.
        for p in self.flat:
            fields.append('    nointerpolation uint4 fparam%d : FPARAM%d;' % (p, p))
        for p in self.f16_plain:
            fields.append('    float4 hparam%d_lo : HPARAML%d;' % (p, p))
            fields.append('    float4 hparam%d_hi : HPARAMH%d;' % (p, p))
        fields.append('    float4 pos : SV_Position;')
        return fields

    def hlsl_vs(self):
        s = self.vs
        out = self.common(embedded_vertex=True) + ['struct VSOut {'] + self.interface(False) + ['};', 'VSOut body(uint vid) {', '    VSOut o = (VSOut)0;']
        s.used_v.add(0)
        flat = re.compile(r'\s*o\.fparam(\d+) = ')
        half = re.compile(r'\s*o\.hparam(\d+)_(?:lo|hi) = ')
        body = [line for line in s.body() if not (flat.match(line) and int(flat.match(line).group(1)) not in self.flat)
                and not (half.match(line) and int(half.match(line).group(1)) not in self.f16_plain)]
        if s.fetched:
            # Behind a fetch call a register read as 0 by default would hide a missing input, so every read must be
            # of a register defined on all paths. Proven here only for the entry block (straight-line code from the
            # call to the first branch or branch target): v0, the fetched registers, and registers an unpredicated
            # write in that block assigned before the read. A write under an exec predicate or in any later block
            # does not define a register.
            written = {0}
            for index, (start, lines, term) in enumerate(s.blocks):
                for line in lines + ([term[1]] if term and term[0] == 'branch' else []):
                    target = re.match(r'v(\d+) = (.*)', line)
                    unwritten = {int(n) for n in re.findall(r'\bv(\d+)\b', target.group(2) if target else line)} - written
                    if unwritten:
                        raise Unsupported('vertex stage reads v%d, which neither the fetch nor an unconditional write '
                                          'before the first branch defines' % min(unwritten))
                    if target and index == 0 and start == 0:
                        written.add(int(target.group(1)))
        out += s.declarations() + ['    v0 = vid;'] + body + ['    return o;', '}',
            '// Rectangle lists arrive as six vertices per rectangle: corners 0 1 2, 0 2 and the fourth corner 0 + 2 - 1.',
            'VSOut main(uint vid : SV_VertexID) {',
            '    if (ud%d == 0) return body(vid);' % self.rect,
            '    uint first = (vid / 6) * 3, corner = vid % 6;',
            '    if (corner < 3) return body(first + corner);',
            '    if (corner == 3) return body(first);',
            '    if (corner == 4) return body(first + 2);',
            '    VSOut p = body(first), q = body(first + 1), r = body(first + 2), o;']
        for p in range(max(self.params) + 1 if self.params else 0):
            out.append('    o.param%d = p.param%d + r.param%d - q.param%d;' % (p, p, p, p))
        out += ['    o.fparam%d = p.fparam%d;' % (p, p) for p in self.flat]
        for p in self.f16_plain:
            out += ['    o.hparam%d_%s = p.hparam%d_%s + r.hparam%d_%s - q.hparam%d_%s;' % (p, h, p, h, p, h, p, h)
                    for h in ('lo', 'hi')]
        out += ['    o.pos = p.pos + r.pos - q.pos;', '    return o;', '}']
        return '\n'.join(out) + '\n'

    def ps_profile(self):
        """The shader profile the pixel stage compiles with: ps_6_0 once it holds a quad read, else ps_5_1."""
        if self.uses_dynamic_heap:
            return 'ps_6_6'
        return 'ps_6_0' if self.ps.needs_sm6 else 'ps_5_1'

    def hlsl_ps(self):
        s = self.ps
        if isinstance(s, PlaceholderPixelStage):
            return s.hlsl()
        targets = sorted(int(k[3:]) for k in s.exports if k.startswith('mrt'))
        depth = 'depth' in s.exports
        quad = QUAD_ANY_HLSL if getattr(s, 'uses_quad_vote', False) else []
        out = self.common() + quad + ['struct PSIn {'] + self.interface(True) + ['};']
        if not targets and not depth:
            out += ['void main(PSIn i) {'] + s.declarations() + s.body() + ['}']
            return '\n'.join(out) + '\n'
        out.append('struct PSOut {')
        if depth:
            out.append('    float depth : SV_Depth;')
        for t in targets:
            kind = self.mrt_kind(t)
            out.append('    %s mrt%d : SV_Target%d;' % ({'int': 'int4', 'uint': 'uint4'}.get(kind, 'float4'), t, t))
        out += ['};', 'PSOut main(PSIn i) {', '    PSOut o = (PSOut)0;']
        out += s.declarations() + s.body() + ['    return o;', '}']
        return '\n'.join(out) + '\n'

    def manifest(self):
        # A manifest that needs a new runtime binding must be rejected by older runtimes.
        version = 2 if self.stored_images() else 1
        if any(source[0] == 'descword' for source in self.uds):
            version = 3
        lines = ['x2w-manifest %d' % version, 'ud %d' % len(self.uds)]
        if self.uses_dynamic_heap:
            lines.append('dynamic_heap')
        for i, source in enumerate(self.uds):
            lines.append('udsrc %d %s' % (i, ' '.join(str(x) for x in source)))
        lines.extend(self.integer_srv_manifest())
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            source, off = key
            where = {'table': 'table %d %d' % (source[1] if len(source) > 1 else 0, off or 0), 'vbtab': 'vb %d' % ((off or 0) // 16),
                     'root': 'root %d' % (source[1] if len(source) > 1 else 0)}.get(source[0])
            if where is None:
                raise Unsupported('resource through %s' % (source,))
            lines.append('srv %d %s %s' % (r['index'], r['kind'], where))
            if self.srv_use_line(r):
                lines.append(self.srv_use_line(r))
            if r.get('d16_exact'):
                lines.append('d16srv %d exact' % r['index'])
        for key, r in sorted(self.stored_images().items(), key=lambda kv: kv[1]['index']):
            source, off = key
            if source[0] != 'table':
                raise Unsupported('stored image through %s' % (source,))
            lines.append('uav %d %s table %d %d' % (r['index'], r['kind'], source[1] if len(source) > 1 else 0, off or 0))
            if r.get('load'):
                lines.append('uavload %d' % r['index'])
        lines += self.sampler_lines()
        if self.uses_sampler_heap:
            lines.append('smpheap %d %d fallback %d' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE, SAMPLER_HEAP_FALLBACK))
        lines.append('params %d' % (max(self.params) + 1 if self.params else 0))
        if self.ps is None:
            lines.append('ps none')
            return '\n'.join(lines) + '\n'
        if isinstance(self.ps, PlaceholderPixelStage):
            lines.append(PLACEHOLDER_PS_PREFIX + self.ps.reason)
            lines.append('placeholder rgb %d,%d,%d' % self.ps.rgb)
        lines.append('targets %s' % ' '.join(k[3:] for k in sorted(self.ps.exports) if k.startswith('mrt')))
        # 'wmask <target> <mask>': the export leaves the channels outside the mask unwritten, so the
        # runtime keeps them out of that target's RenderTargetWriteMask. Full masks are not written.
        for name, mask in sorted(self.ps.write_masks.items(), key=lambda kv: int(kv[0][3:])):
            if mask != 0xF:
                lines.append('wmask %d 0x%x' % (int(name[3:]), mask))
        return '\n'.join(lines) + '\n'


TG_SIZE_WAVES_REFUSAL = ('COMPUTE_PGM_RSRC2 TG_SIZE_EN requires at most 32 threads per group '
                         '(more needs whole wave32 waves and a program proved wave32)')


def wave32_exec_proof(ins):
    """True when the program names a wave32 exec write (a 32-bit scalar op into exec_lo, or s_*saveexec_b32) and no wave64
    exec form (the 64-bit exec pair or a 64-bit scalar op on it). The opcodes differ between the sizes (RDNA 2 ISA, SOP1
    S_AND_SAVEEXEC_B32 / _B64, SOP1 S_MOV_B32 / _B64), so a compiler's exec handling states the wave size; a program that
    never touches exec states nothing, and the caller must say wave_size=32."""
    wave32 = wave64 = False
    for _off, _size, name, asm in ins:
        if not name.startswith('s_'):
            continue
        if re.search(r'\bexec\b', asm):
            wave64 = True
        elif name.endswith('saveexec_b32'):
            wave32 = True
        elif re.search(r'\bexec_lo\b', asm) and name.endswith('_b32'):
            ops = parse(asm)[1]
            wave32 |= bool(ops) and ops[0] == 'exec_lo'
    return wave32 and not wave64


class ReadWriteBuffer(Unsupported):
    """A compute program loads and stores through one descriptor. A raw buffer is lowered as one RWByteAddressBuffer
    (ComputePipeline retries with the key in rw_keys); anything else keeps the refusal."""

    def __init__(self, key, lowerable):
        super().__init__('descriptor %s used for a load and a store' % (key,))
        self.key, self.lowerable = key, lowerable


class ComputePipeline(Pipeline):
    """One compute program with the root and resource model of Pipeline. COMPUTE_PGM_RSRC2 and the thread group
    size are pipeline register state the caller supplies; COMPUTE_PGM_RSRC2 is decoded as compute_store_aot does
    (#1608). Root parameters from s4 are an inference (see Stage.setup_entry)."""

    def __init__(self, cs_code, root_blob, rsrc2, group, sampler_heap=None, refusals=None, decoded=None, wave_size=None):
        from compute_store_aot import checked_group, entry_registers
        try:
            tg_size_en = (isinstance(rsrc2, int) and not isinstance(rsrc2, bool)
                          and 0 <= rsrc2 < (1 << 32) and bool((rsrc2 >> 10) & 1))
            self.entry = entry_registers(rsrc2 & ~(1 << 10) if tg_size_en else rsrc2)
            self.group = checked_group(group)
        except ValueError as reason:
            raise Unsupported(str(reason)) from None
        self.wave_size = wave_size  # the caller's statement that the program runs as wave32 (32), or None: judged from the program
        if tg_size_en and self.group[0] * self.group[1] * self.group[2] > 32 and (
                wave_size not in (None, 32) or self.group[0] * self.group[1] * self.group[2] % 32):
            raise Unsupported(TG_SIZE_WAVES_REFUSAL)
        user, groups, _ = self.entry
        self.tg_size_sgpr = (user + len(groups)) if tg_size_en else None
        self.root = parse_root(root_blob)
        self.vs_fetch, self.packed_position_vgpr, self.rtv_formats = None, None, ()
        self.sampler_heap = os.environ.get('X2W_SAMPLER_HEAP') == '1' if sampler_heap is None else sampler_heap
        self.uses_sampler_heap = False
        self.uses_gather_2d_lod = False
        self.uses_store_d16_x = False
        self.uses_sad = set()
        # LLVM AMDGPUUsage, COMPUTE_PGM_RSRC2.GRANULATED_LDS_SIZE on gfx10: 128 dwords per unit.
        self.lds_words = ((rsrc2 >> 15) & 0x1ff) * 128
        self.integer_texture_srvs = os.environ.get('X2W_INTEGER_TEXTURE_SRVS') == '1'
        self.rw_keys = set()
        self.retry_entry_exec = False
        self.retry_firstlane = False
        self._stage_args = (cs_code, refusals, decoded)
        self._refusals_before = len(refusals) if refusals is not None else 0
        self._build_stage()

    def _build_stage(self):
        """Translates the program (the Stage does it on construction). A buffer both loaded and stored is found
        part-way: translate again with that descriptor as one RW view, once per such descriptor.
        A readfirstlane refusal retries the existing EXEC proof before trying the wider proof and wave lowering."""
        while True:
            try:
                self._reset_stage()
                refusals = self._stage_args[1]
                if (refusals is not None and (READFIRSTLANE_UNIFORM, 'cs') in refusals[self._refusals_before:]
                        and self.retry_readfirstlane()):
                    continue
                return
            except ReadWriteBuffer as conflict:
                self._lower_read_write(conflict)
            except Unsupported as reason:
                if str(reason) != READFIRSTLANE_UNIFORM or not self.retry_readfirstlane():
                    raise

    def retry_readfirstlane(self):
        # Keep both existing lowerings ahead of the new proof and wave path.
        if not self.retry_entry_exec:
            self.retry_entry_exec = True
        elif not self.retry_firstlane:
            self.retry_firstlane = True
        else:
            return False
        return True

    def _lower_read_write(self, conflict):
        if not conflict.lowerable or conflict.key in self.rw_keys or len(self.rw_keys) >= 8:
            raise conflict
        self.rw_keys.add(conflict.key)

    def _reset_stage(self):
        """A fresh translation state: the descriptors, samplers, entry constants and the program's stage."""
        cs_code, refusals, decoded = self._stage_args
        self.resources, self.uavs, self.samplers, self.uds = {}, {}, {}, []
        self.uses_dynamic_heap = False
        self.plain_stored = set()  # raw buffer descriptors a plain buffer store went through
        self.notes, self.modified, self.other_modified = set(), set(), set()  # other: unproved descriptor writes
        self.lod_unvalidated = set()
        if refusals is not None:
            del refusals[self._refusals_before:]
        self.cs = Stage(self, 'cs', cs_code, refusals=refusals, decoded=decoded)

    def resource(self, key, kind):
        if key in self.rw_keys and kind == 'buf':
            return Pipeline.uav(self, key, 'raw')  # the program also stores through it: one RW view, loads read it
        if key in self.uavs:
            raise ReadWriteBuffer(key, key in self.plain_stored and self.uavs[key]['kind'] == 'raw' and kind == 'buf')
        return super().resource(key, kind)

    def uav(self, key, kind='raw', rw=False):
        # rw: a plain buffer store, which a read of the same descriptor may share a view with. Formatted stores and
        # atomics keep the refusal.
        if key in self.resources:
            raise ReadWriteBuffer(key, rw and self.resources[key]['kind'] == 'buf' and kind == 'raw')
        if rw:
            self.plain_stored.add(key)
        return super().uav(key, kind)

    def stored_images(self):
        return {key: r for key, r in self.uavs.items() if r['kind'] != 'raw'}

    def common(self):
        out = [line for line in super().common() if self.uds or self.uses_dynamic_heap or any(self.integer_srv_bindings()) or not line.startswith("cbuffer entry")]
        for r in self.uavs.values():
            if r['kind'] == 'raw':
                out.append('RWByteAddressBuffer u{0} : register(u{0});'.format(r['index']))
        if self.rw_keys:
            out.insert(1, 'uint ldu(RWByteAddressBuffer b, uint a) { ' +
                       unaligned_dword_body('b.Load(a & ~3u)', 'b.Load((a & ~3u) + 4)'))
        return out

    def hlsl_cs(self):
        # The body can also meet a buffer that is both loaded and stored: translate again with it as one RW view.
        while True:
            try:
                return self._hlsl_cs()
            except ReadWriteBuffer as conflict:
                self._lower_read_write(conflict)
                self._build_stage()

    def _hlsl_cs(self):
        s = self.cs
        if s.exports:
            raise Unsupported('export in a compute program')
        body = s.body()
        shared = []
        if getattr(s, 'uses_lds', False):
            shared.append('groupshared uint lds[%d];' % getattr(s, 'lds_exchange_words', self.lds_words))
        if getattr(s, 'uses_quad_shared', False):
            shared.append('groupshared uint x2w_quad[%d];' % (self.group[0] * self.group[1] * self.group[2]))
        if getattr(s, 'uses_mask_ballot', False):
            shared.extend([
                'uint x2w_ballot(uint bit, uint lane) {',
                '    uint word = 0u;',
                '    [unroll] for (uint i = 0u; i < %du; ++i) {' % self.group[0],
                '        if (any(WaveActiveBallot(bit != 0u && lane == i) != 0u)) word |= 1u << i;',
                '    }',
                '    return word;',
                '}',
            ])
        image = getattr(s, 'image_proof', None)
        tables = []
        captures = []
        if image:
            for pc, values in s.image_tables.items():
                tables.append('static const uint image_data_%x[%d] = {%s};' %
                              (pc, len(values), ', '.join('0x%08xu' % value for value in values)))
            captures = ['    uint %s = 0;' % key for key, _ in image.captures.values()]
        return '\n'.join(self.common() + shared + tables + ['[numthreads(%d, %d, %d)]' % self.group,
                                          'void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {']
                         + s.declarations() + captures + ['    ' + line for line in s.entry] + body + ['}']) + '\n'

    def manifest(self):
        # Version 2: `stage cs`, `group` and `uav` lines, which a graphics-only reader must not take for a draw.
        lines = ['x2w-manifest 2', 'stage cs', 'group %d %d %d' % self.group, 'ud %d' % len(self.uds)]
        if self.uses_dynamic_heap:
            lines.append('dynamic_heap')
        lines += ['udsrc %d %s' % (i, ' '.join(str(x) for x in source)) for i, source in enumerate(self.uds)]
        lines.extend(self.integer_srv_manifest())
        for (source, off), r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            if source[0] == 'table':
                where = 'table %d %d' % (source[1], off or 0)
            elif source[0] == 'root':
                where = 'root %d' % source[1]
            elif self.uses_dynamic_heap and source == ('resheap',) and r['kind'] == 'buf' and off is not None and off % 32 == 0:
                where = 'heap fixed %d' % off
            elif source[0] == 'resheap' and len(source) == 3 and r['kind'] == 'buf':
                where = 'heap rootconst %d %d' % source[1:]  # the 32-byte heap record that root constant word indexes
            elif source[0] == 'resheap' and source[1] == 'word' and r['kind'] == 'buf':
                # The record the word at byte `at` (+ root constant word * `times`) of table buffer <param> <byte> indexes.
                table, byte, param, word, scale, at = source[2:]
                where = 'heap word table %d %d at %d' % (table, byte, at) + (
                    '' if param is None else ' rootconst %d %d times %d' % (param, word, scale))
            else:
                raise Unsupported('resource through %s' % (source,))
            lines.append('srv %d %s %s' % (r['index'], r['kind'], where))
        for (source, off), r in self.uavs.items():
            if source[0] != 'table':
                raise Unsupported('store through %s' % (source,))
            where = 'table %d %d' % (source[1], off)
            if r['kind'] == 'raw':
                lines.append('uav %d %s' % (r['index'], where))
            else:
                lines.append('uav %d %s float %s channels 4' % (r['index'], r['kind'], where))
                if r.get('load'):
                    lines.append('uavload %d' % r['index'])
        lines += self.sampler_lines()
        if self.uses_sampler_heap:
            lines.append('smpheap %d %d fallback %d' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE, SAMPLER_HEAP_FALLBACK))
        return '\n'.join(lines) + '\n'


class MeshPixelStage(Pipeline):
    """Pixel stage paired with a mesh shader exporting param0..param12 (#2413 T4)."""

    def __init__(self, mesh_params, ps_code, root_blob, *, rtv_formats=(), ps_hw_inputs=None, ps_attr_map=None,
                 sampler_heap=None, srv_formats=(), refusals=None, decoded=None):
        from mesh_param_link import MAX_PARAM, validate_linkage
        self._init_pixel_context(root_blob, sampler_heap, rtv_formats, ps_hw_inputs, None,
                                 ps_attr_map, srv_formats)
        self.integer_texture_srvs = False  # the mesh runtime has a separate manifest and binding ABI
        clean_params = []
        for p in mesh_params:
            if type(p) is not int or not 0 <= p <= MAX_PARAM:
                raise Unsupported('mesh parameter %r outside 0..%d' % (p, MAX_PARAM))
            clean_params.append(p)
        self.params = sorted(set(clean_params))
        self.mesh_plain_params = frozenset(self.params)
        self.ps = self._build_pixel_stage(ps_code, refusals, decoded)
        self.flat = sorted(a for a, how in self.ps.attrs.items() if 'flat' in how)
        for a, how in self.ps.attrs.items():
            if 'f16' not in how:
                continue
            if how != {'f16'} or a not in self.params:
                raise Unsupported('attribute %d is read as binary16 by the pixel stage but not only so, or not '
                                  'exported as a plain 32-bit parameter by the mesh stage' % a)
        self.f16_plain = sorted(a for a in self.ps.f16_plain if a in self.params)
        missing = sorted(a for a in self.ps.attrs if a not in self.params)
        if missing:
            raise Unsupported('pixel stage reads attributes %s the mesh stage does not export' % missing)
        try:
            validate_linkage(self.params, pixel_attrs=sorted(self.ps.attrs), flat=self.flat,
                             f16_plain=self.f16_plain, semantic='param')
        except ValueError as error:
            raise Unsupported(str(error)) from None

    def common(self):
        return [line for line in super().common() if self.uds or self.uses_dynamic_heap or not line.startswith("cbuffer entry")]

    def linkage(self, *, distance_vectors=0):
        from mesh_param_link import validate_linkage
        return validate_linkage(self.params, pixel_attrs=sorted(self.ps.attrs), flat=self.flat,
                                f16_plain=self.f16_plain, distance_vectors=distance_vectors, semantic='param')


class MeshPipeline(Pipeline):
    """A slot 57 mesh pipeline with an optional pixel stage. The mesh stage is self.vs."""
    vertex_kind = 'ms'

    def hlsl_ms(self):
        s = self.vs
        lines = [re.sub(r' o\.fparam(\d+) = [^;]*;', lambda m: m[0] if int(m[1]) in self.flat else '',
                        re.sub(r' o\.hparam(\d+)_(?:lo|hi) = [^;]*;',
                               lambda m: m[0] if int(m[1]) in self.f16_plain else '', line))
                 for line in s.body()]
        if self.ps is None:
            # Nothing reads a parameter without a pixel stage: drop its writes but keep the rest of the export
            # line (its exec guard and mesh_vertex = true), as for unread fparam/hparam fields above.
            lines = [re.sub(r' o\.param\d+ = [^;]*;', '', line) for line in lines]
        mask_collectives = []
        if s.mesh_collect_masks:
            mask_collectives = [
                'groupshared uint mesh_ballot_word[2];',
                'uint mesh_ballot(uint tid, uint bit) {',
                '    uint wave = tid >> 6u;',
                '    if ((tid & 63u) == 0u) mesh_ballot_word[wave] = 0u;',
                '    GroupMemoryBarrierWithGroupSync();',
                '    if (bit != 0u) InterlockedOr(mesh_ballot_word[wave], 1u << (tid & 31u));',
                '    GroupMemoryBarrierWithGroupSync();',
                '    uint word = mesh_ballot_word[wave];',
                '    GroupMemoryBarrierWithGroupSync();',
                '    return word;',
                '}',
            ]
        collectives = []
        if s.mesh_collect_reads or s.mesh_collect_votes:
            # One logical wave spans 64 guest threads regardless of the desktop wave width.
            # Every group thread calls each site, including threads whose guest PC skips it.
            collectives = [
                'groupshared uint mesh_first_lane[2];',
                'groupshared uint mesh_lane_value[%d];' % MESH_THREADS,
                'uint mesh_first_active(uint tid, bool active, uint value, bool vote) {',
                '    uint wave = tid >> 6u, base = tid & ~63u;',
                '    if (tid == base) mesh_first_lane[wave] = 64u;',
                '    mesh_lane_value[tid] = value;',
                '    GroupMemoryBarrierWithGroupSync();',
                '    if (active) InterlockedMin(mesh_first_lane[wave], tid & 63u);',
                '    GroupMemoryBarrierWithGroupSync();',
                '    uint first = mesh_first_lane[wave];',
                # With empty EXEC, the guest read uses lane zero. 64 maps to zero here.
                '    uint result = vote ? uint(first < 64u) : mesh_lane_value[base + (first & 63u)];',
                '    GroupMemoryBarrierWithGroupSync();',
                '    return result;',
                '}',
            ]
        return '\n'.join(
            self.common() + collectives + mask_collectives + ['struct VSOut {'] + self.interface(False) + [
                '};', 'groupshared uint lds[2];', '[numthreads(%d, 1, 1)]' % MESH_THREADS, '[outputtopology("triangle")]',
                'void main(uint tid : SV_GroupIndex, out vertices VSOut verts[%d], out indices uint3 tris[%d]) {'
                % (MESH_THREADS, MESH_THREADS), '    VSOut o = (VSOut)0; bool mesh_vertex = false;']
            + s.declarations() + ['    ' + line for line in s.entry] + lines
            + ['    if (mesh_vertex) verts[tid] = o;', '}']) + '\n'


def fnv(data, h=1469598103934665603):
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def pipeline_key(vs_code, ps_code, root_blob, layout=b'', ps_inputs=None):
    """Hash the startup translation revision and the three blobs in runtime order, then a stream's input layout.

    A pipeline with a stream description (req_<key>.psin) also hashes its pixel input masks: b'psin1' and then
    SPI_PS_INPUT_ENA and SPI_PS_INPUT_ADDR as little-endian u32. Two pipelines with the same stages, root
    signature and input layout but different masks are then two requests with their own results. Without a
    description the key is unchanged."""
    key = fnv(root_blob, fnv(ps_code, fnv(vs_code, fnv(b'x2w-startup-shader-28'))))
    key = fnv(layout, key) if layout else key
    return key if ps_inputs is None else fnv(b'psin1' + struct.pack('<II', *ps_inputs), key)


def compute_key(cs_code, data, root_blob, desc, state_a=b'', state_b=b''):
    """The runtime key of a device slot 67 compute request (req_<key>.c67, .data, .root, .desc, .sa, .sb): the
    graphics revision followed by "cs", then each part preceded by its length as a little-endian u64: the code,
    the words 7-8 blob, the root signature, the 12 description words with their pointers zeroed and the two
    state blocks."""
    key = fnv(b'x2w-startup-shader-28cs')
    for part in (cs_code, data, root_blob, desc, state_a, state_b):
        key = fnv(part, fnv(struct.pack('<Q', len(part)), key))
    return key


S_ENDPGM = 0xbf810000


def compute_entry(state_a):
    """(COMPUTE_PGM_RSRC2, group size) from state block A. A hypothesis from a local observation, not decoded
    by the stand-in (docs/startup-runtime.md#compute-pipeline-translation): dword 3 = X | Y << 16, dword 4
    bits 15:0 = Z, dword 6 = COMPUTE_PGM_RSRC2."""
    if len(state_a) != 28:
        raise Unsupported('compute state block A is %d bytes, not 28' % len(state_a))
    words = struct.unpack('<7I', state_a)
    return words[6], (words[3] & 0xffff, words[3] >> 16, words[4] & 0xffff)


def load_compute_request(out_dir, key, refusals=None):
    """The ComputePipeline of a slot 67 request, after checking the files against .desc and the key."""
    name = os.path.join(out_dir, 'req_%016x' % key)
    try:
        parts = {ext: open(name + ext, 'rb').read() for ext in ('.c67', '.data', '.root', '.desc', '.sa', '.sb')}
    except OSError:
        raise Unsupported('unreadable shader request') from None  # the error text would name the cache path
    desc = parts['.desc']
    if len(desc) != 96:
        raise Unsupported('compute description is %d bytes, not 96' % len(desc))
    words = struct.unpack('<12Q', desc)
    for ext, word in (('.c67', 6), ('.data', 8), ('.sa', 2), ('.sb', 4)):
        if len(parts[ext]) != words[word]:
            raise Unsupported('compute request %s is %d bytes, the description says %d' % (ext, len(parts[ext]), words[word]))
    if compute_key(parts['.c67'], parts['.data'], parts['.root'], desc, parts['.sa'], parts['.sb']) != key:
        raise Unsupported('compute request files do not match the key in their name')
    code = parts['.c67']
    if not code or len(code) % 4:
        raise Unsupported('compute code of %d bytes is not whole dwords' % len(code))
    if S_ENDPGM not in struct.unpack('<%dI' % (len(code) // 4), code):
        raise Unsupported('compute code without s_endpgm')
    rsrc2, group = compute_entry(parts['.sa'])
    return ComputePipeline(code, parts['.root'], rsrc2, group, refusals=refusals)


def compute_refusals(out_dir, key):
    """Refusal sites of one slot 67 compute request, without publishing a translation."""
    refusals, op = [], 'request loading'
    try:
        pipe = load_compute_request(out_dir, key, refusals)
        op = 'hlsl_cs'
        pipe.hlsl_cs()
        op = 'manifest'
        pipe.manifest()
    except Unsupported as reason:
        refusals.append((str(reason), 'pipeline'))
    except Exception as error:  # noqa: BLE001 - one request must not end the census
        refusals.append((handler_error(op, error, refusals), 'pipeline'))
    return refusals


def log_unvalidated_lod(key, pipe):
    """Once per translated module: the sample forms that read a heap sampler keep its LOD fields unvalidated (#2221)."""
    forms = getattr(pipe, 'lod_unvalidated', None)  # test stand-ins for Pipeline do not set it
    if forms:
        print('%016x: %s through a sampler heap: sampler MIN_LOD, MAX_LOD and LOD bias are not validated (#2221)'
              % (key, ', '.join(sorted(forms))), file=sys.stderr, flush=True)


def translate_compute(out_dir, key):
    """Writes HLSL, then DXC bytecode and the manifest last, or <key>.fail with the reason.
    X2W_FXC set to the empty string publishes the manifest alone, as for graphics."""
    name = os.path.join(out_dir, '%016x' % key)
    with suppress(FileNotFoundError):
        os.remove(name + '.txt')  # the result is unpublished until validation and compilation finish
    try:
        pipe = load_compute_request(out_dir, key)
        hlsl, manifest = pipe.hlsl_cs(), pipe.manifest()
    except Unsupported as reason:
        reason = fail_reason(reason)
        write_failure(name + '.fail', reason)
        return 'unsupported: %s' % reason
    log_unvalidated_lod(key, pipe)
    open(name + '.cs.hlsl', 'w').write(hlsl)
    if os.environ.get('X2W_FXC') != '':
        model = '6.6' if pipe.uses_dynamic_heap else '6.0'
        compiler = find_dxc()
        if compiler is None:
            message = 'dxc not found (X2W_DXC=%s): a compute stage needs shader model %s' % (
                os.environ.get('X2W_DXC', '<unset>'), model)
            write_failure(name + '.fail', message)
            return message
        import subprocess
        with tempfile.TemporaryDirectory(prefix='.dxc-', dir=out_dir) as staging:
            bytecode = os.path.join(staging, 'cs.cso')
            profile = 'cs_' + model.replace('.', '_')
            r = subprocess.run([compiler, '-Gis', '-T', profile, '-E', 'main', '-Fo', bytecode, name + '.cs.hlsl'],
                               capture_output=True, text=True)
            if r.returncode or not has_bytecode(bytecode):
                # DXC names the source path; keep only the diagnostic after it.
                lines = [re.sub(r'^.*?\.cs\.hlsl(:\d+:\d+)?:\s*', '', line) for line in r.stderr.strip().splitlines()]
                detail = lines[0] if lines else ('?' if r.returncode else 'no bytecode produced')
                write_failure(name + '.fail', 'dxc cs: %s' % detail[:500])
                return 'dxc cs failed: %s' % detail[:500]
            os.replace(bytecode, name + '.cs.cso')
    temporary = '%s.txt.%d.tmp' % (name, os.getpid())
    with open(temporary, 'w') as result:
        result.write(manifest)
    os.replace(temporary, name + '.txt')
    with suppress(FileNotFoundError):
        os.remove(name + '.fail')
    return 'ok'


DEFAULT_FXC = r'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe'
KITS_BIN = r'C:\Program Files (x86)\Windows Kits\10\bin'


def find_fxc():
    """The fxc.exe to compile with, '' when X2W_FXC is the empty string (manifest only), or None when none exists.

    Order: X2W_FXC, the 10.0.26100.0 SDK, the newest KITS_BIN/<version>/x64, then PATH."""
    configured = os.environ.get('X2W_FXC')
    if configured == '':
        return ''
    if configured and os.path.isfile(configured):
        return configured
    if os.path.isfile(DEFAULT_FXC):
        return DEFAULT_FXC
    versions = []
    try:
        for entry in os.listdir(KITS_BIN):
            if re.fullmatch(r'[0-9]+(?:\.[0-9]+)*', entry):
                versions.append((tuple(map(int, entry.split('.'))), entry))
    except OSError:
        pass
    for _, entry in sorted(versions, reverse=True):
        candidate = os.path.join(KITS_BIN, entry, 'x64', 'fxc.exe')
        if os.path.isfile(candidate):
            return candidate
    import shutil
    return shutil.which('fxc.exe')


def find_dxc():
    """The dxc.exe for a pixel stage that needs shader model 6.0, or None. Order: X2W_DXC, the newest
    KITS_BIN/<version>/x64, then PATH."""
    configured = os.environ.get('X2W_DXC')
    if configured and os.path.isfile(configured):
        return configured
    versions = []
    try:
        for entry in os.listdir(KITS_BIN):
            if re.fullmatch(r'[0-9]+(?:\.[0-9]+)*', entry):
                versions.append((tuple(map(int, entry.split('.'))), entry))
    except OSError:
        pass
    for _, entry in sorted(versions, reverse=True):
        candidate = os.path.join(KITS_BIN, entry, 'x64', 'dxc.exe')
        if os.path.isfile(candidate):
            return candidate
    import shutil
    return shutil.which('dxc.exe') or shutil.which('dxc')


def has_bytecode(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def has_complete_mesh57_bytecode(path):
    """Check the DXContainer header and length without reading the shader body."""
    try:
        with open(path, 'rb') as shader:
            header = shader.read(32)
            if len(header) != 32 or header[:4] != b'DXBC':
                return False
            size, parts = struct.unpack_from('<II', header, 24)
            return size == os.fstat(shader.fileno()).st_size and 0 < parts <= (size - 32) // 4
    except OSError:
        return False


def is_placeholder_failure(path):
    with suppress(OSError), open(path, encoding='utf-8', errors='replace') as failure:
        return failure.readline().rstrip().endswith(PLACEHOLDER_DRAWN)
    return False


def discard_stale_placeholder(manifest):
    rgb = placeholder_manifest_rgb(manifest)
    if rgb is not None and rgb != placeholder_rgb_text():
        with suppress(FileNotFoundError):
            os.remove(manifest)


def write_graphics_failure(path, reason):
    write_failure(path, reason, context='x2w-placeholder-attempt' if placeholder_ps_enabled() else None)


def publish_placeholder_failure(name, stage, compile_enabled, bytecode_suffixes):
    if not compile_enabled:
        # Source-only publication cannot mark older bytecode as a placeholder.
        for suffix in bytecode_suffixes:
            with suppress(FileNotFoundError):
                os.remove(name + suffix)
    write_graphics_failure(name + '.fail', stage.reason + PLACEHOLDER_DRAWN)


def translate_to(out_dir, key, vs_code, ps_code, root_blob, check=True, rtv_formats=(), rtv_request='', vs_fetch=None,
                 ps_attr_map=None, ps_hw_inputs=None):
    """Writes <key>.vs.hlsl, <key>.ps.hlsl and <key>.txt (last), or <key>.fail with the reason.

    With check, the pair is compiled to .vs.cso and .ps.cso first and a missing fxc is a failure.
    check=False, or X2W_FXC set to the empty string, publishes the manifest without bytecode."""
    name = os.path.join(out_dir, '%016x' % key)
    discard_stale_placeholder(name + '.txt')
    try:
        pipe = Pipeline(vs_code, ps_code, root_blob, rtv_formats=rtv_formats, vs_fetch=vs_fetch, ps_attr_map=ps_attr_map,
                        ps_hw_inputs=ps_hw_inputs)
        vs, ps, manifest = pipe.hlsl_vs(), pipe.hlsl_ps(), pipe.manifest()
        if rtv_request:
            manifest += rtv_request + '\n'
    except Unsupported as reason:
        reason = fail_reason(reason)
        write_graphics_failure(name + '.fail', reason)
        return 'unsupported: %s' % reason
    log_unvalidated_lod(key, pipe)
    open(name + '.vs.hlsl', 'w').write(vs)
    open(name + '.ps.hlsl', 'w').write(ps)
    dynamic_heap = getattr(pipe, 'uses_dynamic_heap', False)
    if dynamic_heap:
        compile_enabled = check and os.environ.get('X2W_FXC') != ''
        fxc = ''
    else:
        fxc = find_fxc() if check else ''
        compile_enabled = bool(fxc)
    if fxc is None:
        message = 'fxc.exe not found (X2W_FXC=%s): set X2W_FXC to its path or install the Windows SDK' % os.environ.get('X2W_FXC', '<unset>')
        write_graphics_failure(name + '.fail', message)
        return message
    manifest_path = name + '.txt'
    write_manifest = True
    # A quad intrinsic exists from shader model 6.0 only, and a pipeline does not mix DXBC and DXIL stages.
    sm6 = dynamic_heap or any(getattr(getattr(pipe, stage, None), 'needs_sm6', False) for stage in ('vs', 'ps'))
    if compile_enabled and sm6:
        compiler = find_dxc()
        if compiler is None:
            feature = 'resource heap indexing needs shader model 6.6' if dynamic_heap else 'a DPP quad read needs shader model 6.0'
            message = 'dxc not found (X2W_DXC=%s): %s' % (os.environ.get('X2W_DXC', '<unset>'), feature)
            write_graphics_failure(name + '.fail', message)
            return message
        model = '6_6' if dynamic_heap else '6_0'
        profiles = (('vs', 'vs_' + model), ('ps', 'ps_' + model))
    else:
        compiler, profiles = fxc, (('vs', 'vs_5_1'), ('ps', 'ps_5_1'))
    if compile_enabled:
        import subprocess
        # An existing manifest can already point at these paths. Publish only bytecode from a successful pair.
        with tempfile.TemporaryDirectory(prefix='.fxc-', dir=out_dir) as staging:
            for stage, profile in profiles:
                bytecode = os.path.join(staging, stage + '.cso')
                if sm6:
                    command = [compiler, '-Gis', '-T', profile, '-E', 'main', '-Fo', bytecode, name + '.%s.hlsl' % stage]
                else:
                    command = [compiler, '/nologo', '/T', profile, '/E', 'main', '/Fo', bytecode, name + '.%s.hlsl' % stage]
                r = subprocess.run(command, capture_output=True, text=True)
                if r.returncode or not has_bytecode(bytecode):
                    detail = r.stderr.strip() or ('?' if r.returncode else 'no bytecode produced')
                    write_graphics_failure(name + '.fail', 'fxc %s: %s' % (stage, detail[-2000:]))
                    return 'fxc %s failed: %s' % (stage, detail.splitlines()[-1])
            with suppress(FileNotFoundError, UnicodeError):
                with open(manifest_path) as existing:
                    write_manifest = existing.read() != manifest
            # If the bindings changed, withdraw the old manifest before publishing new bytecode.
            if write_manifest:
                with suppress(FileNotFoundError):
                    os.remove(manifest_path)
            for stage in ('vs', 'ps'):
                os.replace(os.path.join(staging, stage + '.cso'), name + '.%s.cso' % stage)
    placeholder = isinstance(getattr(pipe, 'ps', None), PlaceholderPixelStage)
    if placeholder:
        publish_placeholder_failure(name, pipe.ps, compile_enabled, ('.vs.cso', '.ps.cso'))
    if write_manifest:
        temporary = '%s.%d.tmp' % (manifest_path, os.getpid())
        try:
            with open(temporary, 'w') as result:
                result.write(manifest)
            os.replace(temporary, manifest_path)
        finally:
            with suppress(FileNotFoundError):
                os.remove(temporary)
    if not placeholder and (fxc or is_placeholder_failure(name + '.fail')):
        with suppress(FileNotFoundError):
            os.remove(name + '.fail')
    return 'ok'


def classify_decoded(ins):
    text = ' '.join(x[3] for x in ins if x[2] == 'exp')
    return 'vs' if 'pos' in text else 'ps' if 'mrt' in text or 'null' in text or not text else 'other'  # no export: a depth-only pixel shader


def classify(blob):
    return classify_decoded(decode(strip_header(blob), embedded_vertex=True))


def load_request(out_dir, key, refusals=None, pair_info=None, *, validate_base_key=False):
    """Read a base request or an RTV variant and classify the source stages."""
    decoded = {}

    def classify_stage(blob):
        if refusals is None:
            try:
                return classify(blob)
            except Unsupported:
                if placeholder_ps_enabled():
                    try:
                        pixel_export_targets(blob)
                    except Unsupported:
                        pass
                    else:
                        return 'ps'
                raise
        ins = decode(strip_header(blob), refusals, embedded_vertex=True)
        kind = classify_decoded(ins)
        decoded[kind] = ins
        return kind

    request, base, formats, description = request_source(out_dir, key)
    name = os.path.join(out_dir, 'req_%016x' % base)
    a, b, root = (open(name + ext, 'rb').read() for ext in ('.a', '.b', '.root'))
    if pair_info is not None and not request:
        pair_info.append((hashlib.sha256(struct.pack('<Q', len(a)) + a + b).digest(), root))
    # A stream with an input layout publishes it as .ia (before .a) and hashes it into the key.
    layout = open(name + '.ia', 'rb').read() if os.path.isfile(name + '.ia') else b''
    if request or validate_base_key:
        # The masks of a stream description are part of the key (pipeline_key).
        masks = read_ps_inputs(description, base)
        if pipeline_key(a, b, root, layout, masks) != base:
            raise Unsupported('RTV variant source key mismatch' if request else
                              'pixel input description names another pipeline' if masks is not None else
                              'shader request source key mismatch')
    fetch = attr_map = None
    if a.startswith(b'DXBC') or b.startswith(b'DXBC'):
        # Slot 86 records complete containers in canonical (VS, PS) order.
        # Keep the container bytes in the cache key; translate only the bounded native code.
        import zlib
        from pixel_aot import container_code
        from resource_inventory import parts
        try:
            container, ps_container = a, b
            a, b = (container_code(parts(blob)[b'DXIL'], stage=stage, decoder=native_instructions,
                                   vertex_fetch=struct.pack('<I', FETCH_CALL))
                    for blob, stage in ((a, 1), (b, 0)))
            attr_map = stream_interface(container, ps_container)
            if starts_with_fetch_call(a):
                fetch = stream_fetch(container, layout.decode('ascii') if layout else None)
        except (KeyError, ValueError, IndexError, struct.error, zlib.error) as error:
            raise Unsupported('invalid stream shader: %s' % error) from error
        if classify_stage(a) != 'vs' or classify_stage(b) != 'ps':
            raise Unsupported('stream stage and shader export disagree')
        kinds = {'vs': a, 'ps': b}
    else:
        kinds = {classify_stage(a): a, classify_stage(b): b}
    return kinds, root, formats, request, fetch, decoded, attr_map


def request_source(out_dir, key):
    """The source of a graphics request: (rtv1 line or '', source key, RTV formats, stream description path).

    A base request is its own source. An RTV variant (req_<key>.rtv) names its source pipeline and is translated
    from that pipeline's blobs and stream description, req_<source>.psin. This is the one place that names the
    description a request depends on: the translation reads its masks from that path, and pending_keys checks
    the same path to withdraw a published result or retry a failure after the description changes or, for a
    result recorded as translated with it (dependency_record), is removed. The path is returned whether or not the
    file exists. An invalid variant line is refused by name."""
    name = os.path.join(out_dir, 'req_%016x' % key)
    request = ''
    formats = ()
    base = key
    if os.path.isfile(name + '.rtv'):
        with open(name + '.rtv', encoding='ascii') as file:
            request = file.read(256)
        if not re.fullmatch(r'rtv1 [0-9]{1,20}(?: [0-9]{1,10}){1,8}', request) or fnv(request.encode('ascii')) != key:
            raise Unsupported('invalid RTV variant request')
        base, *formats = map(int, request.split()[1:])
        if base > 0xffffffffffffffff or any(fmt > 0xffffffff for fmt in formats):
            raise Unsupported('invalid RTV variant values')
    return request, base, formats, os.path.join(out_dir, 'req_%016x.psin' % base)


def request_ps_inputs(out_dir, key):
    """The pixel input masks (SPI_PS_INPUT_ENA, SPI_PS_INPUT_ADDR) of one graphics request, or None.

    They come from the stream description request_source() names: req_<key>.psin, the separately captured
    description of exactly that pipeline (see read_ps_inputs), or for an RTV variant its source pipeline's. The
    masks are part of the request's identity: the key must equal pipeline_key() of the stage blobs, root
    signature, input layout and these masks, so a description copied from another pipeline, left stale after a
    stage change or rewritten with another mask pair is refused, never applied. No file means no captured
    description: the translation keeps its model without hardware inputs."""
    _, base, _, description = request_source(out_dir, key)
    masks = read_ps_inputs(description, base)
    if masks is None:
        return None
    name = os.path.join(out_dir, 'req_%016x' % base)
    a, b, root = (open(name + ext, 'rb').read() for ext in ('.a', '.b', '.root'))
    layout = open(name + '.ia', 'rb').read() if os.path.isfile(name + '.ia') else b''
    if pipeline_key(a, b, root, layout, masks) != base:
        raise Unsupported('pixel input description names another pipeline')
    return masks


def dependency_record(out_dir, key):
    """<key>.psdep: written when a graphics request is translated with a stream description, before its manifest.

    It marks a result as known to depend on the description request_source() names, so pending_keys can withdraw
    that result when the description is removed. A legacy request that never had a description has no record, and
    its inputs are not read again. The record holds the description line the result was translated with."""
    return os.path.join(out_dir, '%016x.psdep' % key)


def record_dependency(out_dir, key, base, masks):
    path = dependency_record(out_dir, key)
    line = 'psin1 %016x ena=0x%04x addr=0x%04x\n' % (base, *masks)
    with suppress(FileNotFoundError, UnicodeError), open(path) as existing:
        if existing.read() == line:
            return
    temporary = '%s.%d.tmp' % (path, os.getpid())
    try:
        with open(temporary, 'w') as record:
            record.write(line)
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.remove(temporary)


def read_ps_inputs(path, key):
    """The (enable, address) masks in the stream description at `path` (req_<key>.psin), one ASCII line
    'psin1 <key> ena=<mask> addr=<mask>' (the key as 16 hex digits, the masks as 0x-prefixed 16-bit values), or
    None without the file. The line must name `key`; a malformed line or a missing mask is refused by name."""
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as file:
        line = file.read(128)
    fields = line.decode('ascii', 'replace').split()
    if not re.fullmatch(rb'psin1 [0-9a-f]{16}(?: (?:ena|addr)=0x[0-9a-f]{1,4}){0,2}\n?', line):
        raise Unsupported('invalid pixel input description')
    masks = dict(field.split('=') for field in fields[2:])
    if len(masks) != len(fields) - 2:
        raise Unsupported('invalid pixel input description')
    if int(fields[1], 16) != key:
        raise Unsupported('pixel input description names another pipeline')
    for field, label in (('ena', 'enable'), ('addr', 'address')):
        if field not in masks:
            raise Unsupported('pixel input %s mask missing from the stream description' % label)
    return int(masks['ena'], 16), int(masks['addr'], 16)


MESH57_REVISION = 'x2w-startup-mesh-57-3'


def mesh57_result_line(key, mesh_digest, pixel_digest):
    return f'mesh57-result {MESH57_REVISION} {key:016x} {mesh_digest:016x} {pixel_digest:016x}'


def mesh57_key(ms, ps, root):
    value = fnv(MESH57_REVISION.encode('ascii'))
    for part in (ms, ps, root):
        value = fnv(part, fnv(struct.pack('<Q', len(part)), value))
    return value


def kill_exit_block(pixel):
    """True when the only null export of a decoded pixel stage is the early exit of a kill.

    LLVM SILateBranchLowering and Mesa ACO place that exit after the program's s_endpgm: the block is
    `exp null off, off, off, off done vm` then the final s_endpgm, and every branch into it enters with EXEC zero,
    so the export writes nothing. The stage's other lanes end at the earlier s_endpgm. An entry is s_cbranch_execz,
    or the demote form: s_cbranch_scc0 directly after `s_cselect exec, exec, 0` (s_cselect leaves SCC, so the taken
    branch follows the select of 0) with no branch landing on the s_cbranch_scc0 itself."""
    at = [i for i, (_, _, op, a) in enumerate(pixel) if op == 'exp' and a.split()[1] == 'null']
    if len(at) != 1 or at[0] != len(pixel) - 2 or pixel[at[0] - 1][2] != 's_endpgm':
        return False
    if pixel[at[0]][3] != 'exp null off, off, off, off done vm':
        return False
    label = 'label_0x%x' % pixel[at[0]][0]
    end = 'label_0x%x' % pixel[-1][0]
    targets = {a.split()[-1] for _, _, op, a in pixel if op.startswith('s_cbranch') or op == 's_branch'}
    entries = [i for i, (_, _, _, a) in enumerate(pixel) if a.split()[-1:] in ([label], [end])]

    def zero_exec(i):
        if pixel[i][2] == 's_cbranch_execz':
            return True
        return (pixel[i][2] == 's_cbranch_scc0' and i > 0
                and pixel[i - 1][3] in ('s_cselect_b64 exec, exec, 0', 's_cselect_b32 exec_lo, exec_lo, 0')
                and 'label_0x%x' % pixel[i][0] not in targets)
    return bool(entries) and all(zero_exec(i) for i in entries)


def validate_mesh57_pixel(ps):
    pixel = decode(strip_header(ps))
    targets = {a.split()[1] for _, _, op, a in pixel if op == 'exp'}
    null_only = targets == {'null'}
    if not pixel or pixel[-1][2] != 's_endpgm' or classify_decoded(pixel) != 'ps' and not null_only:
        raise Unsupported('slot 57 pixel stage has an incompatible export or termination')
    if 'null' in targets and not null_only and not kill_exit_block(pixel):
        raise Unsupported('slot 57 pixel stage mixes the null export with another export')


def translate_mesh57(out_dir, key):
    """Translate a fixed-descriptor mesh request of the accepted shape (mesh_skeleton), or raise Unsupported.

    Writes <key>.ms57.hlsl and the compiled .ms57.cso, their .ps57 counterparts if present, and <key>.ms57.txt (last).
    The manifest records the request identity and the compiled stage digests."""
    name = os.path.join(out_dir, '%016x' % key)
    discard_stale_placeholder(name + '.ms57.txt')
    source = os.path.join(out_dir, 'req_%016x' % key)
    ms, ps, root = (open(source + ext, 'rb').read() for ext in ('.m57', '.ps57', '.root'))
    if mesh57_key(ms, ps, root) != key:
        raise Unsupported('slot 57 mesh request key mismatch')
    parse_root(root)
    mesh = decode(strip_header(ms))
    exports = [assembly.split()[1] for _, _, op, assembly in mesh if op == 'exp']
    if not mesh or mesh[-1][2] != 's_endpgm' or not any(
        op == 's_sendmsg' and assembly == 's_sendmsg 0x9' for _, _, op, assembly in mesh
    ) or not any(target.startswith('pos') for target in exports) or 'target_20' not in exports:
        raise Unsupported('slot 57 mesh stage has no verified allocation and primitive export')
    pixel_refusal = None
    if ps:
        try:
            validate_mesh57_pixel(ps)
        except Unsupported as reason:
            if not placeholder_ps_enabled():
                raise
            pixel_refusal = reason  # The placeholder derives its signature after mesh translation.
    pipe = MeshPipeline(ms, ps or None, root, pixel_refusal=pixel_refusal)
    sources = [('ms', 'ms_6_5', pipe.hlsl_ms())]
    if ps:
        sources.append(('ps', 'ps_6_0', pipe.hlsl_ps()))
    manifest = pipe.manifest()
    digests = [0, 0]
    for stage, _, hlsl in sources:
        open('%s.%s57.hlsl' % (name, stage), 'w').write(hlsl)
    compile_enabled = os.environ.get('X2W_FXC') != ''
    if compile_enabled:
        compiler = find_dxc()
        if compiler is None:
            raise Unsupported('dxc not found (X2W_DXC=%s): a mesh stage needs shader model 6.5' % os.environ.get('X2W_DXC', '<unset>'))
        import subprocess
        # Publish bytecode only after all requested stages compile successfully.
        with tempfile.TemporaryDirectory(prefix='.dxc-', dir=out_dir) as staging:
            for index, (stage, profile, _) in enumerate(sources):
                bytecode = os.path.join(staging, stage + '.cso')
                r = subprocess.run([compiler, '-Gis', '-T', profile, '-E', 'main', '-Fo', bytecode, '%s.%s57.hlsl' % (name, stage)],
                                   capture_output=True, text=True)
                if r.returncode or not has_complete_mesh57_bytecode(bytecode):
                    # DXC names the source path; keep only the diagnostic after it.
                    lines = [re.sub(r'^.*?57\.hlsl(:\d+:\d+)?:\s*', '', line) for line in r.stderr.strip().splitlines()]
                    raise Unsupported('dxc %s: %s' % (stage, (lines[0] if lines else 'missing or incomplete bytecode')[:500]))
                with open(bytecode, 'rb') as compiled:
                    digests[index] = fnv(compiled.read())
            for stage, _, _ in sources:
                os.replace(os.path.join(staging, stage + '.cso'), '%s.%s57.cso' % (name, stage))
    placeholder = isinstance(pipe.ps, PlaceholderPixelStage)
    if placeholder:
        publish_placeholder_failure(name, pipe.ps, compile_enabled, ('.ms57.cso', '.ps57.cso'))
    temporary = '%s.ms57.txt.%d.tmp' % (name, os.getpid())
    with open(temporary, 'w') as result:
        header, body = manifest.split('\n', 1)
        result.write(f'{header}\n{mesh57_result_line(key, *digests)}\n{body}')
    os.replace(temporary, name + '.ms57.txt')
    if not placeholder:
        with suppress(FileNotFoundError):
            os.remove(name + '.fail')
    return 'ok'


def translate_request(out_dir, key):
    """Translate a base shader request, an RTV variant referencing its source blobs, or a slot 67 compute request."""
    if is_compute_request(out_dir, key):
        return translate_compute(out_dir, key)
    if os.path.isfile(os.path.join(out_dir, 'req_%016x.m57' % key)):
        try:
            return translate_mesh57(out_dir, key)
        except (Unsupported, OSError, UnicodeError, struct.error) as reason:
            write_graphics_failure(os.path.join(out_dir, '%016x.fail' % key), str(reason))
            return 'unsupported: %s' % reason
    try:
        try:
            _, base, _, description = request_source(out_dir, key)
            if os.path.isfile(dependency_record(out_dir, key)) and not os.path.isfile(description):
                raise Unsupported('pixel input description missing for a request translated with one')
            kinds, root, formats, request, fetch, _, attr_map = load_request(out_dir, key, validate_base_key=True)
            ps_inputs = request_ps_inputs(out_dir, key)
            if ps_inputs is not None:
                record_dependency(out_dir, key, base, ps_inputs)  # before the manifest is published
        except Unsupported:
            # A result published earlier was translated from sources or masks this request no longer confirms.
            with suppress(FileNotFoundError):
                os.remove(os.path.join(out_dir, '%016x.txt' % key))
            raise
    except (Unsupported, OSError, UnicodeError) as reason:
        write_graphics_failure(os.path.join(out_dir, '%016x.fail' % key), str(reason))
        return 'unsupported: %s' % reason
    if set(kinds) != {'vs', 'ps'}:
        write_graphics_failure(os.path.join(out_dir, '%016x.fail' % key), 'stages %s' % sorted(kinds))
        return 'stages %s' % sorted(kinds)
    result = translate_to(out_dir, key, kinds['vs'], kinds['ps'], root, rtv_formats=formats, rtv_request=request,
                          vs_fetch=fetch, ps_attr_map=attr_map, ps_hw_inputs=ps_inputs)
    if ps_inputs is not None and result == 'ok':
        # An unchanged manifest is not rewritten; mark it as checked against this description (pending_keys).
        with suppress(FileNotFoundError):
            os.utime(os.path.join(out_dir, '%016x.txt' % key))
    return result


REFUSALS_CAVEAT = ('Counts after the first refusal are approximate. Skipped instructions leave later state '
                   'unmodelled and can cause follow-on errors. A pipeline-scope refusal ends the pipeline, '
                   'so Only counts at pipeline scope reflect a lower bound on the reasons reached.')


def request_keys(out_dir, compute=False):
    """Graphics request keys (ready marker .a or .rtv), or with compute, slot 67 compute keys (.c67)."""
    pattern = r'req_([0-9a-f]{16})\.c67' if compute else r'req_([0-9a-f]{16})\.(?:a|rtv)'
    for name in sorted(os.listdir(out_dir)):
        match = re.fullmatch(pattern, name)
        if match:
            yield int(match.group(1), 16)


def is_compute_request(out_dir, key):
    return os.path.isfile(os.path.join(out_dir, 'req_%016x.c67' % key))


def normalize_refusal(reason):
    # Opaque decoder/container details can contain input bytes or disassembly.
    for label in ('undecodable word', 'invalid stream shader', 'unverified vertex fetch record',
                  'unreadable shader request', 'invalid binary shader request'):
        if reason.startswith(label):
            return label
    reason = re.sub(r'\b([sv])\[\d+(?::(\d+))?\]', lambda m: m[1] + ('[N:N]' if m[2] else '[N]'), reason)
    reason = re.sub(r'\b([sv])\d+\b', r'\1N', reason)
    reason = re.sub(r'\b(pos|param|mrt)\d+\b', r'\1N', reason)
    reason = re.sub(r'\blabel_[0-9a-f]+\b', 'label_N', reason, flags=re.IGNORECASE)
    reason = re.sub(r'\bword_0x[0-9a-f]+\b', 'word_N', reason, flags=re.IGNORECASE)
    # Preserve cause numbers. Redact hex operands and offsets, and decimal operands
    # in instruction operand lists, symbolic values, and named operand fields.
    number = r'[+-]?(?:0x[0-9a-f]+|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)'
    reason = re.sub(r'(?<!\w)[+-]?0x[0-9a-f]+\b', 'N', reason, flags=re.IGNORECASE)
    reason = re.sub(r'(^[a-z]\w*_\w+\s+)(' + number + r')(?=\s*(?:,|$))', r'\1N', reason)
    reason = re.sub(r'([,\[(]\s*|\b(?:offset|operand|destination|source|amount|selector|vaddr|soffset|register|attributes|dmask|op_sel|offset0|offset1)[: ]+)([\'"]?)'
                    + number + r'(?![\w.-])', r'\1\2N', reason, flags=re.IGNORECASE)
    return reason


def handler_error(op, error, refusals):
    label = 'internal error' if refusals else 'translator crash'
    return '%s in %s: %s' % (label, op, type(error).__name__)


def request_refusals(out_dir, key, pair_info=None):
    """Collect reached refusal sites for one request without publishing a translation."""
    refusals = []
    try:
        kinds, root, formats, request, fetch, decoded, attr_map = load_request(out_dir, key, refusals, pair_info)
        ps_inputs = request_ps_inputs(out_dir, key)
    except Unsupported as reason:
        refusals.append((str(reason), 'pipeline'))
        return refusals
    except struct.error:
        refusals.append(('invalid binary shader request', 'pipeline'))
        return refusals
    except (OSError, UnicodeError):
        # File errors can include the cache path and request key.
        refusals.append(('unreadable shader request', 'pipeline'))
        return refusals
    except Exception as error:
        refusals.append((handler_error('request loading', error, refusals), 'pipeline'))
        return refusals

    op = 'Pipeline'
    try:
        if set(kinds) != {'vs', 'ps'}:
            raise Unsupported('stages %s' % sorted(kinds))
        pipe = Pipeline(kinds['vs'], kinds['ps'], root, rtv_formats=formats, vs_fetch=fetch,
                        refusals=refusals, decoded=decoded, ps_attr_map=attr_map, ps_hw_inputs=ps_inputs)
        op = 'hlsl_vs'
        pipe.hlsl_vs()
        op = 'hlsl_ps'
        pipe.hlsl_ps()
        op = 'manifest'
        pipe.manifest()
    except Unsupported as reason:
        refusals.append((str(reason), 'pipeline'))
    except Exception as error:
        refusals.append((handler_error(op, error, refusals), 'pipeline'))
    return refusals


def refusals_report(out_dir):
    histogram, rows, pairs = {}, {}, {}
    for key in dict.fromkeys(request_keys(out_dir)):
        pair_info = []
        sites = [(normalize_refusal(reason), scope) for reason, scope in request_refusals(out_dir, key, pair_info)]
        count = len(sites)
        histogram[count] = histogram.get(count, 0) + 1
        distinct = len({reason for reason, _ in sites})
        for reason, scope in set(sites):
            row = rows.setdefault((reason, scope), dict(reason=reason, scope=scope, containing=0, first=0, only=0))
            row['containing'] += 1
            row['first'] += (reason, scope) == sites[0]
            row['only'] += distinct == 1
        if pair_info:
            pair, root = pair_info[0]
            pairs.setdefault(pair, {}).setdefault(root, set()).update(sites)

    pair_rows = {}
    for roots in pairs.values():
        every = set.intersection(*roots.values())
        for reason, scope in set.union(*roots.values()):
            row = pair_rows.setdefault((reason, scope), dict(reason=reason, scope=scope,
                                                            under_every_root=0, under_some_roots=0))
            column = 'under_every_root' if (reason, scope) in every else 'under_some_roots'
            row[column] += 1

    return dict(per_request=dict(examined=sum(histogram.values()), no_refusal=histogram.get(0, 0),
                                 refusal_sites_per_request=[dict(sites=count, requests=histogram[count])
                                                            for count in sorted(histogram)],
                                 reasons=sorted(rows.values(), key=lambda row: (-row['containing'], row['reason'], row['scope']))),
                per_code_pair=dict(examined=len(pairs),
                                   reasons=sorted(pair_rows.values(), key=lambda row: (row['reason'], row['scope']))),
                compute=compute_refusals_report(out_dir),
                caveat=REFUSALS_CAVEAT)


def compute_refusals_report(out_dir):
    """Slot 67 compute requests, kept apart so the graphics statistics above do not move."""
    examined, clean, rows = 0, 0, {}
    for key in request_keys(out_dir, compute=True):
        examined += 1
        sites = [normalize_refusal(reason) for reason, _ in compute_refusals(out_dir, key)]
        clean += not sites
        for reason in set(sites):
            row = rows.setdefault(reason, dict(reason=reason, containing=0, first=0))
            row['containing'] += 1
            row['first'] += reason == sites[0]
    return dict(examined=examined, no_refusal=clean,
                reasons=sorted(rows.values(), key=lambda row: (-row['containing'], row['reason'])))


def refusals_main(argv):
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Read-only diagnostic for all cached requests.',
                                     epilog=REFUSALS_CAVEAT, allow_abbrev=False)
    parser.add_argument('--refusals', required=True, metavar='CACHE_DIR')
    parser.add_argument('--json', action='store_true', help='emit the report as JSON')
    args = parser.parse_args(argv)
    if not os.path.isdir(args.refusals):
        parser.error('CACHE_DIR must be an existing directory')
    report = refusals_report(args.refusals)
    if args.json:
        print(json.dumps(report))
        return
    requests = report['per_request']
    print('Per request key')
    print('Requests examined: %d' % requests['examined'])
    print('No refusal recorded (not compiled): %d' % requests['no_refusal'])
    print('Refusal sites per request (sites, requests):')
    for bucket in requests['refusal_sites_per_request']:
        print('%d\t%d' % (bucket['sites'], bucket['requests']))
    print('Containing\tFirst\tOnly\tScope\tReason')
    for row in requests['reasons']:
        print('%(containing)d\t%(first)d\t%(only)d\t%(scope)s\t%(reason)s' % row)
    pairs = report['per_code_pair']
    print('Per code pair (excluding RTV variants)')
    print('Code pairs examined: %d' % pairs['examined'])
    print('Under every root\tUnder some roots\tScope\tReason')
    for row in pairs['reasons']:
        print('%(under_every_root)d\t%(under_some_roots)d\t%(scope)s\t%(reason)s' % row)
    compute = report['compute']
    print('Compute requests (device slot 67)')
    print('Compute requests examined: %d' % compute['examined'])
    print('No refusal recorded (not compiled): %d' % compute['no_refusal'])
    print('Containing\tFirst\tReason')
    for row in compute['reasons']:
        print('%(containing)d\t%(first)d\t%(reason)s' % row)
    print(REFUSALS_CAVEAT)


def main():
    if any(arg == '--refusals' or arg.startswith('--refusals=') for arg in sys.argv[1:]):
        return refusals_main(sys.argv[1:])
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    if sys.argv[1] == '--capture':
        # --capture <dir> <capture.bin>: write request files for every pipeline the capture binds
        import capture
        cap = capture.Capture(sys.argv[3])
        done = set()
        for c in cap.calls:
            w = c['args'][0]
            if c['slot'] != 25 or w in done:
                continue
            done.add(w)
            p = cap.psos.get(w)
            seen = {w}
            while p and p['path'] == 55 and p['base']:
                if p['base'] in seen:
                    print('pso %x: wrapper chain loops back to %x' % (w & 0xffffff, p['base'] & 0xffffff))
                    p = None
                    break
                seen.add(p['base'])
                p = cap.psos.get(p['base'])
            if not p or p['path'] != 57:
                continue
            if len(p['desc']) < 96:
                print('pso %x: pipeline desc is %d bytes, need 96' % (w & 0xffffff, len(p['desc'])))
                continue
            q = struct.unpack_from('<12Q', p['desc'], 0)
            a, b = cap.memory.read(q[5], q[6]), cap.memory.read(q[9], q[10])
            root = cap.roots.get(q[0], b'')
            if not a or not b or len(a) != q[6] or len(b) != q[10]:
                print('pso %x: code not captured' % (w & 0xffffff))
                continue
            key = pipeline_key(a, b, root)
            for ext, blob in (('.a', a), ('.b', b), ('.root', root)):
                open(os.path.join(out_dir, 'req_%016x%s' % (key, ext)), 'wb').write(blob)
            print('pso %x -> request %016x' % (w & 0xffffff, key))
    jobs = jobs_argument(sys.argv[3:])
    if sys.argv[1] == '--retry-lifted':
        # --retry-lifted <dir> [seconds] [--jobs N] [--weights <stderr.txt or runs directory>]
        given = [arg for arg in sys.argv[3:4] if not arg.startswith('--')]
        seconds = float(given[0]) if given else PRE_RUN_RETRY_SECONDS
        at = sys.argv.index('--weights') if '--weights' in sys.argv[3:] else None
        weights = run_weights(sys.argv[at + 1]) if at is not None and at + 1 < len(sys.argv) else None
        print(retry_lifted(out_dir, seconds, jobs=jobs, weights=weights).line(), flush=True)
        return
    if sys.argv[1] == '--watch':
        # --watch <dir> [seconds]: translate requests as a running title writes them, until stopped
        interval = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
        # In-run the pool translates only requests without a result or a failure: a failure an older translator
        # build wrote is retried by the pass after the run (--pending, translator_stamp), never here, so a translator
        # update does not turn the run into a retry of every old refusal (Run Pi: 1,823 retries, 28 workers, 5.8 GiB).
        # The pool is small (watch_jobs) and logs its size and commit every POOL_LOG_SECONDS.
        started = time.time()  # requests published after this are the running title's: they go first
        jobs, monitor = watch_jobs(), PoolMonitor()
        clock = TitleClock(os.environ.get('X2W_TITLE_START_FILE'), started)
        print('gfx2hlsl watch: pool of at most %d workers, old failures left for the pass after the run' % jobs,
              file=sys.stderr, flush=True)
        while True:
            watch_pass(out_dir, True, started, jobs=jobs, monitor=monitor, recycle=True, clock=clock)
            monitor.tick(None)
            time.sleep(interval)
    if family_retry_enabled():
        build = None if os.environ.get('X2W_RETRY_FAILED') == '1' else translator_version()
    else:
        build = translator_stamp()
    translate_pending(out_dir, since_translator=build, jobs=jobs)


def jobs_argument(args):
    """--jobs N among `args`: at most N workers, and never more than translation_jobs(); None when absent."""
    if '--jobs' not in args:
        return None
    at = args.index('--jobs')
    try:
        wanted = int(args[at + 1])
    except (IndexError, ValueError):
        raise SystemExit('gfx2hlsl: --jobs needs a whole number')
    return max(1, min(wanted, translation_jobs()))




def translator_stamp():
    """The newest modification time of the translator's own files (this module and the modules it loaded from its
    folder): a .fail older than this was written by another build and is tried again. None with X2W_RETRY_FAILED=1,
    which retries every failure."""
    if os.environ.get('X2W_RETRY_FAILED') == '1':
        return None
    here = os.path.dirname(os.path.abspath(__file__))
    files = {os.path.abspath(__file__)} | {
        os.path.abspath(module.__file__) for module in list(sys.modules.values())
        if getattr(module, '__file__', None) and os.path.dirname(os.path.abspath(module.__file__)) == here}
    return max(os.path.getmtime(name) for name in files if os.path.isfile(name))


def watch_pass(out_dir, skip_failed, since=None, since_translator=None, jobs=None, monitor=None, recycle=False,
               clock=None):
    """One --watch pass: translate_pending with rescans, so a request the title publishes during a long pass starts
    within about RESCAN_SECONDS. A worker killed from outside (out of memory) breaks the pool; the pass is
    reported and its requests are retried on the next pass instead of ending the watcher."""
    from concurrent.futures.process import BrokenProcessPool
    try:
        translate_pending(out_dir, skip_failed=skip_failed, rescan=True, since=since, since_translator=since_translator,
                          jobs=jobs, monitor=monitor, recycle=recycle, clock=clock)
        return True
    except BrokenProcessPool as error:
        print('translation pass interrupted, retrying: %s' % error, file=sys.stderr, flush=True)
        return False


def fail_reason(reason):
    """The .fail text of a refusal. An operand alone ("operand '-v30'") does not say which lowering refused it, so
    the instruction is added when the translation loop recorded it. The record is an attribute on the exception, so
    a refusal raised outside that loop, or rebuilt as a new Unsupported on the way here, gives the message alone."""
    text = str(reason)
    if text.startswith('operand ') and getattr(reason, 'instruction', None):
        text += ' in %s' % reason.instruction
    return text


def raise_site(error):
    """The traceback frame that raised error, skipping standard-library frames so an error raised inside a library
    call (json, struct helpers, ...) names the code that made the call. None without a traceback."""
    frames = traceback.extract_tb(error.__traceback__)
    roots = {os.path.normcase(os.path.abspath(sysconfig.get_paths()[name])) + os.sep
             for name in ('stdlib', 'platstdlib')}
    for frame in reversed(frames):
        if not os.path.normcase(os.path.abspath(frame.filename)).startswith(tuple(roots)):
            return frame
    return frames[-1] if frames else None


# Refusal families recent translators lift or narrow, by the reason an older build wrote (the first line of its .fail).
# In the run the --watch pool leaves old refusals alone and the pass after the run retries them, so a lift would reach
# only the next run: before the title starts tester_run retries every old refusal (--retry-lifted), these families
# first, within PRE_RUN_RETRY_SECONDS. A lift adds the reason it removes here so it goes first.
LIFTED_REASONS = (
    ('wave64-reduction', r'^wave reduction in the wave64 form \(s_or_saveexec_b64 .*only the wave32 chain is lowered'),
    ('embedded-vertex-table', r'^s_getpc_b64: the shader load address is only elided '
                              r'for a program that also calculates a branch target'),
    ('lod-bias', r'with a sampler LOD bias'),
    # Includes a plain MRTZ covered by the terminal done+vm export in its block.
    ('f16-unary', r'SDWA form of v_(exp|rcp)_f16|^v_(exp|rcp)_f16'),
    ('writelane-vector', r'^v_writelane_b32: v\d+ is also used as a vector \('),
    ('lds-fill', r'LDS fill needs the unchanged entry local X ID'),
    ('lds-qword', r'LDS exchange needs one bounded qword per local X ID'),
    ('mrtz', r'MRTZ requires one uncompressed depth export'),
    ('rw-buffer', r'used for a load and a store'),
    ('cube', r'image dimension SQ_RSRC_IMG_CUBE'),
    ('tg-size', r'TG_SIZE_EN requires at most \d+ threads per group'),
    ('dmask', r'image_store with dmask \S+: a typed UAV store writes all four channels'),
    ('dpp-pixel', r'_dpp requires a straight-line pixel shader with entry EXEC'),
    ('dpp-compute', r'_dpp requires a straight-line compute program with entry EXEC'),
    ('mesh57-wave-load', r'^slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes \(wave-load block:'),
    ('compute-execz-scalar', r'^(?:' + '|'.join(sorted(EXECZ_SCALAR_JOIN)) +
     r') inside a block s_cbranch_execz skips per lane: definition of .*?, live where the block joins'),
    ('mesh57-firstlane', r'^slot 57 mesh: v_readfirstlane_b32 outside the group-index read is not covered'),
)
PRE_RUN_RETRY_SECONDS = 180.0
# In the cache: the build that last ran the pre-run retry and the keys it left refused. A lifted family's text is also
# what this build writes for some cases its lift does not cover (such as TG_SIZE_EN groups outside wave32 coverage),
# so a newer .fail of a lifted family is retried once per build, not at the head of every run.
RETRIED_NAME = 'pre_run_retried.txt'
# The runtime's per-key use lines (tester_pack.key_weights ranks the zip's request files by the same lines).
NOT_DRAWN_LINE = re.compile(r'x2w-render not drawn key ([0-9a-f]{16}): (\d+)')
CENSUS_LINE = re.compile(r'x2w-render compute census key ([0-9a-f]{16}): (\d+) dispatches, (\d+) translated')
# One line per pipeline key the title used (graphics and compute): a run killed before its census still has these.
USE_LINE = re.compile(r'x2w-render key use \S+ ([0-9a-f]{16}) first \S+ s last \S+ s uses (\d+)')


def lifted_family(reason, families=LIFTED_REASONS):
    """The LIFTED_REASONS family whose pattern the refusal text contains, or None."""
    return next((name for name, pattern in families if re.search(pattern, reason)), None)


def family_retry_enabled():
    return os.environ.get('X2W_PRE_RUN_FAMILY_RETRY') == '1'


@functools.lru_cache(maxsize=1)
def translator_version():
    """Content fingerprint of the translator and its shader input and lowering modules."""
    script = Path(__file__).resolve()
    tools = script.parent.parent
    sources = (script, script.with_name('lane_exec.py'), script.with_name('ps_input_scan.py'),
               *(tools / (name + '.py') for name in (
                   'disasm_rdna2', 'pixel_aot', 'resource_inventory', 'verify_build',
                   'shader_minmax', 'compute_store_aot', 'mesh_param_link')))
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode('ascii') + b'\0')
        digest.update(source.read_bytes().replace(b'\r\n', b'\n') + b'\0')
    return digest.hexdigest()


def refusal_family(reason):
    from tester_pack import normal_reason

    known = lifted_family(reason)
    if known:
        return known
    normalized = normal_reason(reason.removeprefix('unsupported: ').strip())
    return 'reason-' + hashlib.sha256(normalized.encode('utf-8')).hexdigest()


FAILURE_METADATA = re.compile(r'x2w-refusal-v1 ([0-9a-f]{64}) ([a-z0-9-]+)')


def write_failure(path, reason, *, version=None, context=None):
    detail = reason + '\n' + (context + '\n' if context else '')
    if not family_retry_enabled():
        with open(path, 'w') as handle:
            handle.write(detail)
        return
    family = refusal_family(reason.split('\n', 1)[0])
    version = version or translator_version()
    atomic_retry_text(path, '%sx2w-refusal-v1 %s %s\n' % (detail, version, family))


def atomic_retry_text(path, content):
    handle = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=os.path.dirname(path) or '.',
                                         prefix='.retry-', delete=False)
    temporary = handle.name
    try:
        with handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.remove(temporary)


def read_failure(path):
    with open(path, encoding='utf-8', errors='replace') as handle:
        reason = handle.readline()
        metadata = None
        for line in handle:
            metadata = FAILURE_METADATA.fullmatch(line.rstrip())
    if metadata:
        return reason, metadata[2], metadata[1]
    return reason, refusal_family(reason), None


def lifted_failures(out_dir, since_translator, families=LIFTED_REASONS):
    """Keys of the requests an older build refused (a .fail older than since_translator, still pending) for a reason
    of a lifted family, oldest request first."""
    return [key for key, lifted in old_failures(out_dir, since_translator, families) if lifted]


def old_failures(out_dir, since_translator, families=LIFTED_REASONS, final=frozenset(), weights=None):
    """(key, in a lifted family) for every refused request worth a retry, in retry order: the requests the last run
    used first, most used first (`weights`: draws not drawn, dispatches without a translation, else the key's uses),
    then the lifted families, then
    graphics and mesh requests (what a title draws) before compute, each oldest request first. A .fail older than
    since_translator was written by another build and is listed; a newer one is final, unless its reason is in a lifted
    family and this build has not retried it yet (`final`): a .fail's time does not prove which build wrote it (Run P
    kept a pixel MRTZ request refused by a stale .fail although that build translates it)."""
    weights = weights or {}
    rows = []
    for order, key in enumerate(pending_keys(out_dir, False, None)):
        fail = os.path.join(out_dir, '%016x.fail' % key)
        try:
            newer = since_translator is not None and os.path.getmtime(fail) >= since_translator
            with open(fail, encoding='utf-8', errors='replace') as handle:
                reason = handle.readline()
        except OSError:
            continue
        lifted = lifted_family(reason, families) is not None
        if newer and (not lifted or key in final):
            continue
        compute = os.path.isfile(os.path.join(out_dir, 'req_%016x.c67' % key))
        rows.append((not weights.get(key, 0), -weights.get(key, 0), not lifted, compute, order, key, lifted))
    return [(key, lifted) for _u, _w, _l, _c, _o, key, lifted in sorted(rows)]


def retried_by_build(out_dir, stamp):
    """The keys this translator build (stamp) already retried before a run and left refused (RETRIED_NAME); an empty
    set when another build wrote the file, or there is none."""
    try:
        with open(os.path.join(out_dir, RETRIED_NAME), encoding='ascii') as handle:
            words = handle.read().split()
    except (OSError, UnicodeDecodeError):
        return frozenset()
    if stamp is None or not words or words[0] != 'build-%r' % stamp:
        return frozenset()
    return frozenset(int(word, 16) for word in words[1:] if re.fullmatch(r'[0-9a-f]{16}', word))


def note_retried(out_dir, stamp, keys):
    """Record that this build retried `keys` and left them refused (replaces the record of another build)."""
    if stamp is None:
        return
    path = os.path.join(out_dir, RETRIED_NAME)
    with suppress(OSError):
        with open(path + '.tmp', 'w', encoding='ascii') as handle:
            handle.write('build-%r\n' % stamp + ''.join('%016x\n' % key for key in sorted(keys)))
        os.replace(path + '.tmp', path)


def run_weights(path):
    """{key: draws not drawn (graphics) or dispatches without a translation (compute)} from a run's stderr.txt, each
    key's newest cumulative line; {} when the file cannot be read."""
    if family_retry_enabled():
        return scene_weights(Path(path))
    weights, uses = {}, {}
    try:
        with open(path, encoding='utf-8', errors='replace') as handle:
            for line in handle:
                if not line.startswith('x2w-render '):
                    continue
                drawn, census, use = NOT_DRAWN_LINE.match(line), CENSUS_LINE.match(line), USE_LINE.match(line)
                if drawn:
                    weights[int(drawn.group(1), 16)] = int(drawn.group(2))
                elif census:
                    weights[int(census.group(1), 16)] = max(0, int(census.group(2)) - int(census.group(3)))
                elif use:
                    uses[int(use.group(1), 16)] = int(use.group(2))
    except OSError:
        return {}
    # A key with no draw/dispatch line (a run killed before its census) counts by how often the title used it.
    return {**{key: count for key, count in uses.items() if count > 0}, **weights}


def scene_weights(path, need_census=True):
    from tester_pack import KEY_USE_RE

    if path.is_dir():
        candidates = []
        for run in path.glob('run_*'):
            with suppress(OSError):
                if run.is_dir():
                    candidates.append((run.stat().st_mtime_ns, run))
        # The newest run with scene census lines; a title killed before any census still ranks by key use.
        for need in (True, False):
            for _, run in sorted(candidates, reverse=True):
                weights = scene_weights(run / 'stderr.txt', need)
                if weights:
                    return weights
        return {}

    census = False
    weights = {}
    try:
        with path.open(encoding='utf-8', errors='replace') as handle:
            for line in handle:
                if (line.startswith(('x2w-render scene census', 'x2w-render compute census:',
                                     'x2w-render not drawn by pipeline key:'))
                        or CENSUS_LINE.match(line) or NOT_DRAWN_LINE.match(line)):
                    census = True
                match = KEY_USE_RE.fullmatch(line)
                if match:
                    key, count = int(match[2], 16), int(match[5])
                    weights[key] = max(count, weights.get(key, 0))
    except OSError:
        return {}
    return weights if census or not need_census else {}


class RetryReport:
    def __init__(self, keys, ok, refused, left, seconds, lifted=0, used=0, skipped=0):
        self.keys, self.ok, self.refused, self.left, self.seconds = keys, ok, refused, left, seconds
        self.lifted, self.used, self.skipped = lifted, used, skipped

    def line(self):
        """tester_run's log line: old refusals (of them in lifted families), translated now, still refused, left for
        the pass after the run."""
        left = ', %d left for the pass after the run (time cap)' % self.left if self.left else ''
        if self.skipped:
            left += ', %d skipped after family probes' % self.skipped
        used = ', %d used in the last run' % self.used if self.used else ''
        return 'pre-run retry: %d keys (%d in lifted families%s), ok %d, still refused %d%s, %.0f s' % (
            self.keys, self.lifted, used, self.ok, self.refused, left, self.seconds)


def retry_lifted(out_dir, seconds=PRE_RUN_RETRY_SECONDS, jobs=None, families=LIFTED_REASONS, clock=time.monotonic,
                 weights=None):
    """Translate again the old refusals (old_failures: lifted families first, then by the last run's use `weights`)
    before the title starts. No new request is started after `seconds`; what is left keeps its .fail for the pass after
    the run. What this build retried and still refuses is recorded (RETRIED_NAME) and not retried again by it."""
    if family_retry_enabled():
        return retry_families(out_dir, seconds, jobs, clock, weights or {})
    start = clock()
    stamp = translator_stamp()
    already = retried_by_build(out_dir, stamp)
    rows = old_failures(out_dir, stamp, families, already, weights)
    keys = [key for key, _lifted in rows]
    results = retry_batch(out_dir, keys, jobs, clock, start, seconds)
    clear_successful_failures(out_dir, results)
    if results:
        note_retried(out_dir, stamp, already | {key for key, result in results.items() if result != 'ok'})
    ok = sum(1 for result in results.values() if result == 'ok')
    return RetryReport(len(keys), ok, len(results) - ok, len(keys) - len(results), clock() - start,
                       sum(1 for _key, lifted in rows if lifted), sum(1 for key in keys if (weights or {}).get(key)))


def retry_batch(out_dir, keys, jobs, clock, start, seconds):
    results = {}
    if keys:
        jobs = max(1, min(jobs or translation_jobs(), len(keys)))
        if jobs == 1:
            for key in keys:
                if clock() - start >= seconds:
                    break
                results[key] = translate_one(out_dir, key)
        else:
            from concurrent.futures import FIRST_COMPLETED, wait
            queue, running = list(keys), {}
            with new_pool(jobs) as pool:
                while queue or running:
                    while queue and len(running) < jobs and clock() - start < seconds:
                        key = queue.pop(0)
                        running[pool.submit(translate_one, out_dir, key)] = key
                    if not running:
                        break
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in done:
                        results[running.pop(future)] = future.result()
    return results


def clear_successful_failures(out_dir, results):
    for key, result in results.items():
        if result == 'ok':
            with suppress(OSError):
                os.remove(os.path.join(out_dir, '%016x.fail' % key))


FAMILY_PROBES = 8
FAMILY_CURSOR = 'pre_run_family_cursor.txt'


def retry_families(out_dir, seconds, jobs, clock, weights):
    start = clock()
    version = translator_version()
    force = os.environ.get('X2W_RETRY_FAILED') == '1'
    groups = {}
    keys = sorted(set(pending_keys(out_dir, False)), key=lambda key: (-weights.get(key, 0), key))
    lifted = 0
    for key in keys:
        try:
            reason, family, written_by = read_failure(os.path.join(out_dir, '%016x.fail' % key))
        except OSError:
            continue
        if not force and written_by == version:
            continue
        groups.setdefault(family, []).append(key)
        lifted += lifted_family(reason) is not None

    # Interleave families so an expensive graphics family cannot occupy every probe slot.
    families = list(groups)
    cursor_path = Path(out_dir) / FAMILY_CURSOR
    with suppress(OSError, ValueError):
        offset = families.index(cursor_path.read_text(encoding='utf-8').strip()) + 1
        families = families[offset:] + families[:offset]
    probes = [(family, groups[family][index]) for index in range(FAMILY_PROBES)
              for family in families if index < len(groups[family])]
    results = retry_batch(out_dir, [key for _, key in probes], jobs, clock, start, seconds)
    for family, key in reversed(probes):
        if key in results:
            with suppress(OSError):
                atomic_retry_text(cursor_path, family + '\n')
            break

    passing_families = {family for family, key in probes if results.get(key) == 'ok'}
    incomplete = {family for family, key in probes if key not in results}
    rest = sorted((key for family in passing_families - incomplete for key in groups[family][FAMILY_PROBES:]),
                  key=lambda key: (-weights.get(key, 0), key))
    skipped = sum(max(0, len(groups[family]) - FAMILY_PROBES)
                  for family in groups.keys() - passing_families - incomplete)
    results.update(retry_batch(out_dir, rest, jobs, clock, start, seconds))
    clear_successful_failures(out_dir, results)
    total = sum(map(len, groups.values()))
    ok = sum(result == 'ok' for result in results.values())
    return RetryReport(total, ok, len(results) - ok, total - len(results) - skipped, clock() - start, lifted,
                       sum(bool(weights.get(key)) for group in groups.values() for key in group), skipped)


def translate_one(out_dir, key):
    """translate_request, with an unexpected translator error recorded as <key>.fail instead of raised."""
    try:
        return translate_request(out_dir, key)
    except Exception as error:  # noqa: BLE001 - one bad request must not stop a scan or the watcher
        reason = 'translator error: %s: %s' % (type(error).__name__, error)
        # Where it was raised (function, file name, line), so the .fail points at the fault without the program
        # that hit it. The line number moves with unrelated edits, so this text is for reading, not a stable key.
        frame = raise_site(error)
        if frame:
            reason += ' (in %s, %s line %d)' % (frame.name, os.path.basename(frame.filename), frame.lineno)
        write_failure(os.path.join(out_dir, '%016x.fail' % key), reason)
        return reason


WATCH_JOBS = 4          # --watch: workers beside a running title (X2W_TRANSLATE_JOBS overrides)
RECYCLE_AFTER = 50      # requests per pool worker before it is replaced, so its memory growth is given back
POOL_LOG_SECONDS = 30.0  # --watch: how often the pool's size and commit are logged


def watch_jobs():
    """Workers of the in-run (--watch) pool: X2W_TRANSLATE_JOBS when set, else at most WATCH_JOBS."""
    return translation_jobs() if os.environ.get('X2W_TRANSLATE_JOBS') else min(WATCH_JOBS, translation_jobs())


def new_pool(jobs, recycle=False):
    """A process pool; with recycle (the --watch command line, whose workers live as long as the run) each worker is
    replaced after RECYCLE_AFTER requests (Python 3.11 and later; it needs a spawning start method, which an in-process
    caller with a fork context does not have)."""
    from concurrent.futures import ProcessPoolExecutor
    if recycle and sys.version_info >= (3, 11):
        return ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=RECYCLE_AFTER)
    return ProcessPoolExecutor(max_workers=jobs)


def private_bytes(pid):
    """Committed private memory of a process in bytes (Windows PrivateUsage; elsewhere RssAnon + VmSwap), or None."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes as wt

        class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS_EX
            _fields_ = [('cb', wt.DWORD), ('faults', wt.DWORD)] + [(n, ctypes.c_size_t) for n in (
                'peak_ws', 'ws', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile',
                'private')]
        k32, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
        k32.OpenProcess.restype = wt.HANDLE
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            return counters.private if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb) else None
        finally:
            k32.CloseHandle(handle)
    try:
        with open('/proc/%d/status' % pid) as status:
            fields = dict(line.split(':', 1) for line in status if ':' in line)
        return sum(int(fields.get(name, '0 kB').split()[0]) for name in ('RssAnon', 'VmSwap')) * 1024
    except (OSError, ValueError):
        return None


TITLE_START_SLACK = 30.0  # s: a title.pid older than the watcher by more than this is an earlier run's


class TitleClock:
    """--watch: the ' [t=N s]' suffix of a result line (translate.txt), so a translation lines up with the run's
    timeline. N counts from the title's start, the mtime of `start_file` (the run's title.pid, written when the title
    is launched, a few seconds after the watcher) once it exists and is not older than the watcher; until then from the
    watcher's start, marked ' [t=N s, translator clock]'."""

    def __init__(self, start_file, started, now=time.time):
        self.start_file, self.started, self.now, self.zero = start_file, started, now, None

    def title_start(self):
        if self.zero is None and self.start_file:
            try:
                written = os.stat(self.start_file).st_mtime
            except OSError:
                return None
            if written >= self.started - TITLE_START_SLACK:  # a title.pid left by an earlier run is not this start
                self.zero = written
        return self.zero

    def __call__(self):
        zero = self.title_start()
        if zero is None:
            return ' [t=%.1f s, translator clock]' % (self.now() - self.started)
        return ' [t=%.1f s]' % (self.now() - zero)


class PoolMonitor:
    """Logs the --watch pool every POOL_LOG_SECONDS to stderr: workers alive, requests running, their committed
    private memory now and the peak seen."""

    def __init__(self, every=POOL_LOG_SECONDS, clock=time.monotonic, out=None):
        self.every, self.clock, self.out = every, clock, out
        self.next_at = clock()
        self.peak = 0

    def tick(self, pool, running=0):
        if self.clock() < self.next_at:
            return
        self.next_at = self.clock() + self.every
        pids = list(getattr(pool, '_processes', None) or {}) if pool is not None else []
        sizes = [size for size in map(private_bytes, pids) if size is not None]
        total = sum(sizes)
        self.peak = max(self.peak, total)
        print('gfx2hlsl watch pool: %d workers, %d requests running, commit %.0f MiB (peak %.0f MiB)'
              % (len(pids), running, total / 1048576.0, self.peak / 1048576.0), file=self.out or sys.stderr, flush=True)


def translation_jobs():
    try:
        jobs = int(os.environ.get('X2W_TRANSLATE_JOBS') or 0)
    except ValueError:
        jobs = 0
    jobs = jobs if jobs > 0 else os.cpu_count() or 1
    return min(jobs, 61) if os.name == 'nt' else jobs  # ProcessPoolExecutor refuses more than 61 on Windows


RESCAN_SECONDS = 1.0  # --watch: how often a pass looks for requests published since it started
READY_MARKERS = ('a', 'rtv', 'c67', 'm57')


def request_stamps(out_dir, keys):
    """{key: modification time of its ready marker}: when the runtime published the request (it writes the marker
    last). A request whose marker is gone sorts last."""
    stamps = {}
    for key in keys:
        stamps[key] = float('inf')
        for ext in READY_MARKERS:
            with suppress(OSError):
                stamps[key] = os.path.getmtime(os.path.join(out_dir, 'req_%016x.%s' % (key, ext)))
                break
    return stamps


def request_order(stamps, since=None):
    """Keys in the order the title asked for them: oldest ready marker first. With since (the watcher's start), the
    requests published after it, which the running title is waiting for, come before the older backlog."""
    return sorted(stamps, key=lambda key: (since is None or stamps[key] < since, stamps[key], key))


def translate_pending(out_dir, skip_failed=False, jobs=None, rescan=False, since=None, since_translator=None,
                      monitor=None, recycle=False, clock=None):
    """Translate every request that has no result yet, oldest published first (request_order). The runtime writes
    the ready markers last.

    Requests are independent (each writes only its own <key>.* files), so several run in parallel
    processes; X2W_TRANSLATE_JOBS (default: the processor count, at most 61 on Windows) sets how many.
    Each worker gets one request at a time. With rescan (--watch), the pending list is read again at most every
    RESCAN_SECONDS between results, so a request published during the pass is queued in its order instead of
    waiting for the next pass; a request already handled in this pass is queued again only when its marker is
    rewritten. Results are printed as they finish.
    """
    stamps = request_stamps(out_dir, dict.fromkeys(pending_keys(out_dir, skip_failed, since_translator)))
    queue = request_order(stamps, since)
    if not queue:
        return
    handled = {}  # key -> marker time it was queued with in this pass

    def refill(queue):
        # A failure written during this pass is final for the pass (skip_failed), as on the next pass.
        queued = set(queue)
        fresh = request_stamps(out_dir, [key for key in pending_keys(out_dir, True) if key not in queued])
        stamps.update((key, t) for key, t in fresh.items() if handled.get(key) != t)
        return request_order({key: stamps[key] for key in queued | {k for k in fresh if handled.get(k) != fresh[k]}},
                             since)

    jobs = jobs or translation_jobs()
    if not rescan:
        jobs = min(jobs, len(queue))
    last_scan = time.monotonic()
    pool = running = None
    if jobs >= 2:
        from concurrent.futures import FIRST_COMPLETED, wait
        pool, running = new_pool(jobs, recycle), {}
    try:
        while True:
            while queue and (pool is None or len(running) < jobs):
                key = queue.pop(0)
                handled[key] = stamps[key]
                if pool is None:
                    print('%016x: %s%s' % (key, translate_one(out_dir, key), clock() if clock else ''), flush=True)
                    break
                running[pool.submit(translate_one, out_dir, key)] = key
            if running:
                done, _ = wait(running, timeout=RESCAN_SECONDS if rescan else None, return_when=FIRST_COMPLETED)
                for future in done:
                    print('%016x: %s%s' % (running.pop(future), future.result(), clock() if clock else ''), flush=True)
            if monitor is not None:
                monitor.tick(pool, len(running or ()))
            idle = not queue and not running
            if rescan and (idle or time.monotonic() - last_scan >= RESCAN_SECONDS):
                queue = refill(queue)
                last_scan = time.monotonic()
                idle = not queue and not running
            if idle:
                return
    finally:
        if pool is not None:
            pool.shutdown()


def pending_keys(out_dir, skip_failed, since_translator=None):
    """Keys whose request has no finished result. A key with a .fail is skipped when the .fail is newer than its
    request and stream description. Outside watch mode it must also match the translator content fingerprint,
    or postdate the legacy translator timestamp. With neither cutoff, every failure is tried again."""
    keys = []
    version = since_translator if isinstance(since_translator, str) and not skip_failed else None
    manifest_only = os.environ.get('X2W_FXC') == ''
    names = sorted(os.listdir(out_dir))
    failures = {name for name in names if name.endswith('.fail')}
    rgb = placeholder_rgb_text()
    for name in names:
        m = re.fullmatch(r'req_([0-9a-f]{16})\.(a|rtv|c67|m57)', name)
        if not m:
            continue
        key = int(m.group(1), 16)
        result = os.path.join(out_dir, '%016x' % key)
        mesh = m.group(2) == 'm57'
        compute = m.group(2) == 'c67'
        stages = ('.cs.cso',) if m.group(2) == 'c67' else ('.ms57.cso', '.ps57.cso') if mesh else ('.vs.cso', '.ps.cso')
        ready_bytecode = has_complete_mesh57_bytecode if mesh else has_bytecode
        manifest = result + ('.ms57.txt' if mesh else '.txt')
        # A published manifest with bytecode (or manifest-only mode) is done; otherwise resume (#2008). The stream
        # description the translation reads (request_source: the request's own, or an RTV variant's source
        # pipeline's) is a dependency like the request file. Rewritten after the result was published, the result
        # is checked again, so one translated with other masks is not kept in silence; rewritten after a refusal,
        # the request is retried.
        described = None
        lost = False
        if m.group(2) in ('a', 'rtv'):
            with suppress(Unsupported, OSError, UnicodeError):  # no description, or a variant line refused later
                described = os.path.getmtime(request_source(out_dir, key)[3])
            # A result translated with a description recorded it (dependency_record). With that description gone,
            # the result is checked again and withdrawn; a legacy request without a record is not read again.
            lost = described is None and os.path.isfile(result + '.psdep')
        failure_name = '%016x.fail' % key
        fail = os.path.join(out_dir, failure_name)
        placeholder = not compute and failure_name in failures and is_placeholder_failure(fail)
        if placeholder and placeholder_manifest_rgb(manifest) != rgb:
            keys.append(key)
            continue
        if os.path.isfile(manifest) and not lost and not (
                described is not None and described > os.path.getmtime(manifest)):
            if mesh:
                with suppress(OSError), open(manifest, 'rb') as published:
                    rows = published.read().splitlines()
                    marker = (f'mesh57-result {MESH57_REVISION} {key:016x} ').encode('ascii')
                    digests = re.fullmatch(rb'([0-9a-f]{16}) ([0-9a-f]{16})', rows[1][len(marker):]) if (
                        len(rows) > 1 and rows[1].startswith(marker)) else None
                    if digests:
                        none = b'ps none' in rows
                        needed = stages[:1] if none else stages
                        mesh_digest, pixel_digest = (int(value, 16) for value in digests.groups())
                        if manifest_only or (mesh_digest and bool(pixel_digest) != none and
                                             all(ready_bytecode(result + s) for s in needed)):
                            continue
            elif manifest_only or all(ready_bytecode(result + s) for s in stages):
                continue
        if placeholder:
            keys.append(key)
            continue
        if not compute and rgb is not None and failure_name in failures:
            with suppress(OSError), open(fail, encoding='utf-8', errors='replace') as failure:
                if 'x2w-placeholder-attempt' not in failure.read().splitlines():
                    keys.append(key)
                    continue
        # A repeated pass retries a failure only once its request or its stream description is rewritten. A result
        # still published without its recorded description is checked even with an older .fail, which a success on
        # the manifest-only path leaves in place.
        recheck = lost and os.path.isfile(manifest)
        inputs = max(os.path.getmtime(os.path.join(out_dir, name)), described or 0)
        cutoff = inputs if skip_failed or version else (
            None if since_translator is None else max(inputs, since_translator))
        if cutoff is not None and not recheck:
            with suppress(OSError):
                if os.path.getmtime(fail) >= cutoff and (version is None or read_failure(fail)[2] == version):
                    continue
        keys.append(key)
    return keys

if __name__ == '__main__':
    main()
