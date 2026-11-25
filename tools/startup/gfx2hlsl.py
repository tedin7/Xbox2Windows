"""gfx1013 machine code -> HLSL for a vertex+pixel pipeline (startup tools; see docs/startup-runtime.md).

Model: one lane. Registers are uint variables; exec/vcc/mask registers hold the lane's bit in bit 0.
Scalar registers are also tracked symbolically at translation time so that pointer and descriptor
values (tables, vertex-buffer table, root buffers) turn into HLSL resources instead of arithmetic.
The output is HLSL for both stages plus a manifest that tells the runtime what to bind where.
"""
from contextlib import suppress
import os
import re
import struct
import sys
import sysconfig
import tempfile
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import disasm_rdna2 as D  # noqa: E402


class Unsupported(Exception):
    pass


PREFETCH = bytes.fromhex('0300a0bf')

# DPP16 quad pair broadcasts (quad_perm of one lane of every horizontal or vertical pair into the other). The lanes of
# a quad are the pixels 0 1 / 2 3, row by row; lane i reads lane quad_perm[i].
DPP_PAIR = frozenset(((1, 1, 3, 3), (0, 0, 2, 2), (2, 3, 2, 3), (0, 1, 0, 1)))
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
              37: ('BUF_FMT_16_16_SNORM', 4), 30: ('BUF_FMT_8_8_8_8_UINT', 4)}
# Typed buffer loads: (word load or None, the four component expressions). Components a format lacks read as
# 0, 0, 0, 1 here, as in the existing tbuffer translation; that default is an assumption, not a measurement.
FETCH_FORMATS = {
    'BUF_FMT_8_8_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', 'U(((w >> 16) & 255) / 255.0)', 'U((w >> 24) / 255.0)']),
    'BUF_FMT_8_8_8_8_UINT': ('ldu({0}, a)', ['w & 255u', '(w >> 8) & 255u', '(w >> 16) & 255u', 'w >> 24']),
    'BUF_FMT_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', '0u', 'U(1.0)']),
    'BUF_FMT_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', '0u', '0u', 'U(1.0)']),
    'BUF_FMT_8_UINT': ('{0}.Load(a & ~3u) >> ((a & 3u) * 8u)', ['w & 255u', '0u', '0u', '1u']),
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
# A FORMAT the draw should have refused reaches the switch default: every component becomes this NaN pattern
# (never a finite zero), so a missed validation shows in the readback instead of passing as black.
DESC_FORMAT_UNKNOWN = 0x7fc0f0f0
DESC_FORMAT_CODES = {
    'BUF_FMT_8_UNORM': 1, 'BUF_FMT_8_UINT': 5, 'BUF_FMT_8_8_UNORM': 14, 'BUF_FMT_32_FLOAT': 22,
    'BUF_FMT_16_16_UNORM': 23, 'BUF_FMT_16_16_SNORM': 24, 'BUF_FMT_16_16_UINT': 27, 'BUF_FMT_16_16_FLOAT': 29, 'BUF_FMT_8_8_8_8_UNORM': 56, 'BUF_FMT_8_8_8_8_UINT': 60,
    'BUF_FMT_32_32_FLOAT': 64, 'BUF_FMT_16_16_16_16_FLOAT': 71, 'BUF_FMT_32_32_32_FLOAT': 74,
    'BUF_FMT_32_32_32_32_FLOAT': 77}
DESC_FORMAT_SHIFT, DESC_FORMAT_MASK = 12, 127
# Integer formats read a DST_SEL constant 1 as the integer 1, every other format as 1.0.
DESC_ONE_INTEGER = ('BUF_FMT_8_UINT', 'BUF_FMT_8_8_8_8_UINT')

def starts_with_fetch_call(code):
    return len(code) >= 4 and struct.unpack_from('<I', code)[0] == FETCH_CALL


def parse_input_layout(text):
    """'ia1 <n>' then one '<semantic> <index> <format> <slot> <offset> <class> <step>' line per element."""
    lines = text.split('\n')
    if not re.fullmatch(r'ia1 ([1-9]|[12][0-9]|3[0-2])', lines[0]) or lines[-1] != '' or len(lines) != int(lines[0][4:]) + 2:
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
        # The shared disassembler ignores op_sel (w0[14:11]) here, as LLVM 15 does. Mesa ACO's
        # v_interp_p2_hi_f16 is p2 with op_sel bit 3 (w0[14]): it writes the high half of the destination. The
        # attribute half `high` is a different bit (w1[8]) and is printed, so it is not refused here.
        if struct.unpack_from('<I', code, off)[0] & 0x7800:
            raise Unsupported('%s with op_sel: only the plain form and high are lowered' % name)
    if name == 'image_load_mip':
        # The shared disassembler does not expose these gfx10 MIMG flags.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & (1 << 16):
            raise Unsupported('image_load_mip tfe')
        if word1 & (1 << 30):
            raise Unsupported('image_load_mip a16')
    if re.fullmatch(r'v_cmpx?_(lt|eq|le|gt|ne|ge)_[iu]16_e64', name):
        # The decoder prints NEG on an inline integer as a sign (inline 1 with NEG reads `-1`, the same text as
        # inline -1), and LLVM 15 rejects `-v1` and `|v1|` on these sources: refuse either bit (VOP3 w0[10:8] ABS,
        # w1[31:29] NEG) so a printed -N is always a genuine inline integer.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & 0x700 or word1 & 0xe0000000:
            raise Unsupported('%s with neg/abs: not a 16-bit integer source modifier' % name)
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
    return asm


def decode(code, refusals=None):
    out, off = [], 0
    while off + 4 <= len(code):
        try:
            n, name, asm, _ = D.decode_instruction(code, off)
        except Exception as error:
            raise Unsupported('undecodable word at +0x%x: %s' % (off, error))
        try:
            asm = check_encoding(code, off, n, name, asm)
        except Unsupported as reason:
            if refusals is None:
                raise
            refusals.append((str(reason), 'decode'))
        else:
            out.append((off, n, name, asm))
        off += n
    last = max((i for i, x in enumerate(out) if x[2] == 's_endpgm'), default=len(out) - 1)
    return out[:last + 1]


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


def parse(asm):
    """-> (mnemonic, [operands], [modifier tokens])"""
    name, _, rest = asm.partition(' ')
    ops = split_top(rest)
    mods = []
    if ops:
        tail = split_top(ops[-1], ' ')
        ops[-1], mods = tail[0], tail[1:]
    return name, ops, mods


FLOATS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0,
          '0.15915494': 0.15915494}


def fbits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


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


def exec_stays_clear(name, asm, saved):
    """True when an exec write cannot set the bit of a lane whose bit is clear: exec & x (s_and_saveexec, v_cmpx,
    s_and or s_andn2 with exec as the kept operand), or exec | x, exec ^ x or a move of x where x is only ever a copy
    of exec made inside the same skipped block (so x's bit is clear for that lane too)."""
    base = re.sub(r'_e(32|64)$', '', name)
    _, ops, _ = parse(asm)
    if base.startswith(('s_and_saveexec', 'v_cmpx')):
        return True
    if not ops or not is_exec(scalar_tokens(ops[0])):
        return False
    srcs = [scalar_tokens(op) for op in ops[1:]]
    # saved tracks register names, not which exec half was copied, so only a 32-bit restore is trusted
    copies = lambda regs: bool(regs) and regs <= saved and base.endswith('32')
    if base in ('s_and_b32', 's_and_b64'):
        return any(is_exec(regs) for regs in srcs)
    if base in ('s_andn2_b32', 's_andn2_b64'):
        return bool(srcs) and is_exec(srcs[0])
    if base in ('s_or_b32', 's_or_b64', 's_xor_b32', 's_xor_b64') and len(srcs) == 2:
        return all(is_exec(regs) or copies(regs) for regs in srcs)
    if base in ('s_mov_b32', 's_mov_b64') and len(srcs) == 1:
        return copies(srcs[0])
    return False


def exec_copies_after(name, asm, saved, may):
    """The registers that hold only copies of exec made in the block, after this instruction."""
    base = re.sub(r'_e(32|64)$', '', name)
    _, ops, _ = parse(asm)
    dests = scalar_tokens(ops[0]) - EXEC_KEYS - {'scc'} if ops else set()
    saved = saved - (may - EXEC_KEYS - {'scc'})
    if base.startswith('s_and_saveexec') or (base in ('s_mov_b32', 's_mov_b64') and len(ops) == 2
                                             and is_exec(scalar_tokens(ops[1]))):
        saved |= dests
    return saved


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
    two = base.startswith(TWO_DESTINATIONS)
    dests, srcs = ops[:2] if two else ops[:1], ops[2:] if two else ops[1:]
    writes = scalar_tokens(' '.join(dests))
    reads = scalar_tokens(' '.join(srcs))
    if base.startswith(READ_MODIFY_WRITE):
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
    return reads, writes, writes | may | ({'scc'} if base.startswith('s_') else set())

# Forms with no scalar destination: their printed operands are all sources (v_cmpx writes only EXEC, which the
# symbolic tracking does not hold).
NO_SCALAR_DESTINATION = ('s_cbranch', 's_branch', 's_waitcnt', 's_nop', 's_endpgm', 's_cmp', 's_bitcmp', 's_setpc',
                         's_barrier', 's_sendmsg', 's_inst_prefetch', 's_clause', 's_setprio', 's_sleep', 's_setreg',
                         's_code_end', 's_dcache', 's_icache', 's_ttrace', 'exp', 'ds_write', 'v_cmpx')
TWO_SCALAR_DESTINATIONS = ('v_add_co', 'v_sub_co', 'v_subrev_co', 'v_div_scale', 'v_mad_u64_u32', 'v_mad_i64_i32')
# Symbols whose register also holds the value at run time: constants are assigned when set, entry root constants are
# assigned at entry when a join needs it, and the others are assigned when tracked.
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
        return len(sym) == 4  # ('desc', source, load offset, dword); the 'modified' mark drops the written value
    # An add on a pointer keeps its symbol and marks it 'added' (the offset is not in the tuple), so only an
    # untouched pointer, exactly (kind, source), names one value.
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


def smem_offset(tok):
    """The SMEM offset operand as a token: 'off' (no offset operand) is the literal 0, anything else unchanged."""
    return '0' if tok == 'off' else tok


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
# Multiply, fused multiply-add (including the literal form), reciprocal and median of three use
# only local arithmetic. Move copies its input bits. None reads a lane ID or another lane.
UNIFORM_VALU = frozenset((
    'v_cvt_u32_f32', 'v_cvt_i32_f32', 'v_cvt_f32_u32', 'v_cvt_f32_ubyte0', 'v_mul_f32',
    'v_fma_f32', 'v_fmaak_f32', 'v_rcp_f32', 'v_med3_f32', 'v_mov_b32',
))


# DPP quad_perm selections that exchange lanes of a 2x2 pixel quad (lane = 2 * y + x), with the SM 6.0 intrinsic.
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
                                          'buffer_', 'tbuffer_', 'image_', 'exp', 'ds_')):
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


def _lds_single(tok):
    """Single hardware VGPR vN (v0..v255) for bounded LDS exchange validation."""
    m = re.fullmatch(r'v(\d+)', tok)
    return int(m.group(1)) if m and int(m.group(1)) < 256 else None


def _lds_pair(tok):
    """Exact contiguous 2-VGPR pair v[n:n+1] (v0..v255) for bounded 64-bit LDS exchanges."""
    m = re.fullmatch(r'v\[(\d+):(\d+)\]', tok)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return lo if hi == lo + 1 and hi < 256 else None


def lds_byte_offsets(name, mods):
    if name != 'ds_read2_b32':
        if not mods:
            return (0,)
        if len(mods) == 1 and re.fullmatch(r'offset:\d+', mods[0]):
            return (int(mods[0][7:]),)
        raise Unsupported('LDS exchange needs two VGPRs and an optional byte offset')
    # DS_READ2 encodes each offset independently in dwords. An omitted field is zero.
    offsets = {}
    for mod in mods:
        field = re.fullmatch(r'offset([01]):(\d+)', mod)
        if not field or field[1] in offsets or int(field[2]) > 255:
            raise Unsupported('LDS read2 needs ordinary dword offsets')
        offsets[field[1]] = 4 * int(field[2])
    first, second = offsets.get('0', 0), offsets.get('1', 0)
    if first == second:
        raise Unsupported('LDS read2 needs two distinct offsets')
    return first, second


def check_lds_exchange(ins, group, words):
    """Prove one race-free 32-bit or 64-bit fill of a contiguous LDS span, one uniform barrier, then reads of initialized words.

    RDNA 2 ISA (Ch. 9 DS): ds_write_b32 / ds_read_b32 / ds_read2_b32 access 4-byte dwords and
    ds_write_b64 / ds_read_b64 access two contiguous little-endian dwords (v[n] at addr+offset, v[n+1] at addr+offset+4).
    LLVM 15 AMDGPUUsage defines COMPUTE_PGM_RSRC2.GRANULATED_LDS_SIZE in 128-dword (512-byte) units on gfx10.
    """
    if not 0 < words <= 8192 or group[1:] != (1, 1):
        raise Unsupported('LDS exchange needs a 1D group and a nonzero allocation of at most 32 KiB')
    first = None
    elem_bytes = None
    published = False
    read = False
    prev = (None, [], [])
    for i, (_, _, _, asm) in enumerate(ins):
        name, ops, mods = parse(asm)
        base = name.removesuffix('_e32').removesuffix('_e64')
        if (base.startswith(('s_branch', 's_cbranch', 's_setpc', 's_swappc')) or writes_exec(base, ops)
                or base == 's_endpgm' and i + 1 < len(ins)):
            raise Unsupported('LDS exchange needs straight-line code and entry EXEC')
        if base == 's_barrier':
            if (first is None or published or prev != ('s_waitcnt', ['lgkmcnt(0)'], [])
                    or mods or ops not in ([], ['0'], ['0x0'])):
                raise Unsupported('LDS exchange needs one barrier after a completed fill')
            published = True
        elif base in ('ds_write_b32', 'ds_read_b32', 'ds_read2_b32', 'ds_write_b64', 'ds_read_b64'):
            write = base.startswith('ds_write')
            width = 8 if base.endswith('_b64') else 4
            addr_op = ops[0] if (write and len(ops) == 2) else (ops[1] if len(ops) == 2 else '')
            val_op = ops[1] if (write and len(ops) == 2) else (ops[0] if len(ops) == 2 else '')
            if base == 'ds_read2_b32':
                if len(ops) != 2 or _lds_pair(val_op) is None or _lds_single(addr_op) is None:
                    raise Unsupported('LDS read2 needs a VGPR pair and a byte address VGPR')
            elif (len(ops) != 2 or _lds_single(addr_op) is None
                  or (_lds_pair(val_op) is None if width == 8 else _lds_single(val_op) is None)):
                raise Unsupported('LDS exchange needs %s, %s and an optional byte offset'
                                  % ('two VGPRs' if width == 4 else 'a VGPR address',
                                     'an aligned 2-VGPR pair' if width == 8 else 'a matching data VGPR'))
            offsets = lds_byte_offsets(base, mods)
            producer, args, modifiers = prev
            if write:
                offset, = offsets
                shift = '3' if width == 8 else '2'
                # The prefix below preserves the entry X ID in v0. Each lane writes exactly one distinct element.
                # 8-byte alignment is not required for LDS dword-addressed writes.
                if (first is not None or producer != 'v_lshlrev_b32' or args != [addr_op, shift, 'v0']
                        or modifiers or offset % 4 or offset + group[0] * width > words * 4):
                    raise Unsupported('LDS exchange needs one bounded %s per local X ID'
                                      % ('qword' if width == 8 else 'dword'))
                first = offset
                elem_bytes = width
            else:
                if not published or modifiers or not args or args[0] != addr_op:
                    raise Unsupported('LDS read needs a bounded address after the fill barrier')
                if elem_bytes != width:
                    raise Unsupported('LDS read width (%dB) does not match initialized fill (%dB)'
                                      % (width, elem_bytes or 0))
                if producer == 'v_mov_b32' and len(args) == 2 and re.fullmatch(r'0x[0-9a-f]+|\d+', args[1]):
                    low = high = int(args[1], 0)
                elif (producer == 'v_and_b32' and len(args) == 3 and re.fullmatch(r'0x[0-9a-f]+|\d+', args[1])
                      and _lds_single(args[2]) is not None):
                    low, high = 0, int(args[1], 0)
                else:
                    raise Unsupported('LDS read needs a literal or literal-masked byte address')
                # Every byte offset in [a + offset, a + offset + width) must be 4-byte aligned and inside the filled span.
                if (low % 4 or high % 4
                        or any(offset % 4 or low + offset < first or high + offset + (width - 4) >= first + group[0] * width
                               for offset in offsets)):
                    raise Unsupported('LDS read outside the aligned, initialized span')
                if i + 1 >= len(ins) or ins[i + 1][3] != 's_waitcnt lgkmcnt(0)':
                    raise Unsupported('LDS read needs a completed LGKM wait')
                read = True
        elif base.startswith('ds_') or 'lds' in mods:
            raise Unsupported('LDS exchange does not cover ' + asm)
        elif first is None and not base.startswith('s_'):
            # Simple VALU instructions only write their single destination. Other forms, e.g. v_swap_b32,
            # can also change a source. After the fill, each read proves its own address without the entry X ID.
            if (base not in VEC_U and base not in VEC_F and base != 'v_mov_b32'
                    or not ops or not re.fullmatch(r'v[0-9]+', ops[0]) or ops[0] == 'v0'):
                raise Unsupported('LDS fill needs the unchanged entry local X ID')
        prev = base, ops, mods
    if not read:
        raise Unsupported('LDS exchange needs a fill, a barrier and a completed read')


class Stage:
    """Translates one stage. Shares the resource and entry-constant registries of its pipeline."""

    # Immutable fallbacks for unit probes that bypass __init__; __init__ and translate() assign fresh
    # per-instance sets, so nothing is shared across Stage instances.
    needs_sm6 = False         # the lowering uses a shader model 6.0 intrinsic, so the stage cannot compile as 5.1
    exec_known = False        # unit probes without __init__ customize this explicitly when needed
    low16_only_v = ()         # VGPRs whose upper 16 bits are undefined after odd-dmask d16 image_load
    pinned_low16_v = ()       # low16_only_v live across basic-block boundaries
    whole_read_v = ()         # VGPRs read outside a proven low-16-bit consumer
    byte_float_v = ()         # VGPRs proven to hold +0.0 or an integer in 1..255 (v_cvt_f32_ubyte0..3 result)
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

    def __init__(self, pipe, kind, code, refusals=None, decoded=None, structured_loops=True):
        self.pipe, self.kind = pipe, kind
        self.structured_loops = structured_loops
        self.refusals = refusals
        if refusals is None:
            self.ins = decode(strip_header(code))
        else:
            self.ins = decode(strip_header(code), refusals) if decoded is None else decoded
        # A loop header is translated before the back edges into it, so whether a back edge brings the symbol kept
        # there is known only afterwards: translate again with that register read at run time or refused there. A
        # register only moves from kept to 'run' to 'refuse', so this ends.
        self.force_joined = {}    # branch target -> register -> 'run' | 'refuse'
        first_refusal = len(refusals) if refusals is not None else 0
        while True:
            self.lines = []           # body statements (per block)
            self.sym = {}             # scalar register -> symbolic value
            self.used_s, self.used_v = set(), set()
            self.used_atomic = set()
            self.exec_known = True    # exec still equals the entry mask (the lane is active)
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
            self.translate()
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
        elif self.kind == 'vs':
            self.sym[0] = ('ptr', ('vbtab',))
            self.sym[1] = ('ptrhi', ('vbtab',))
            self.sym[2] = ('ud', self.pipe.ud(('basevertex',)))
            self.sym[3] = ('ud', self.pipe.ud(('startinstance',)))
        else:
            self.sym[reg] = ('const', 0)  # primitive mask: only ever moved to m0

    def setup_compute_entry(self, root_end):
        """Entry registers of a compute wave as compute_store_aot.entry_registers (#1608) gives them: the user SGPRs,
        then one SGPR per enabled group-ID axis (SV_GroupID), and v0..v2 holding SV_GroupThreadID. Where the root
        parameters sit inside the user SGPRs (from s4) is this translator's inference. User SGPRs the root does not
        place (s0-s3 and any past the root's span) have no known value, so any numeric read of them is refused."""
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
            raise Unsupported('numeric use of s%d holding %s' % (n, s[0]))
        self.used_s.add(n)
        return 's%d' % n

    def note_whole_read(self, reg):
        if reg in self.low16_only_v:
            raise Unsupported('whole-word read of v%d after odd d16 image_load' % reg)
        if isinstance(self.whole_read_v, set):
            self.whole_read_v.add(reg)

    def u(self, tok, *, low16_only=False):
        """uint expression for a 32-bit operand."""
        if tok in ('vcc_lo', 'vcc'):
            return 'vcc'
        if tok in ('exec_lo', 'exec'):
            return 'exec'
        if tok == 'm0':
            m = self.sym.get('m0')
            if m:
                raise Unsupported('numeric use of m0 holding %s' % m[0])
            return 'm0'
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
                or bool(reg and reg[0] == 's' and reg[1] in self.masks))

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

    def u_src(self, tok):
        """uint expression for an integer-context source that the disassembler may print negated."""
        # In an integer instruction the `-` modifier is the source's sign bit flipped: the same bit
        # pattern a float negation produces. Only handlers that model it ask for this; other integer
        # sources keep using u(), so an unexpected negation stays a refusal rather than a silent flip.
        neg, tok = self.split_neg(tok)
        return '(%s ^ 0x80000000u)' % self.u(tok) if neg else self.u(tok)

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

    def emit(self, text):
        self.lines.append(text)

    def clear_low16(self, n):
        if isinstance(self.low16_only_v, set) and self.exec_known and n not in self.pinned_low16_v:
            self.low16_only_v.discard(n)

    def set_v(self, n, expr, *, low16_only=False):
        if getattr(self, 'interp_p1', None):
            self.interp_p1.pop(n, None)
        if self.byte_float_v:
            self.byte_float_v.discard(n)
        if hasattr(self, 'uniform_v'):
            self.uniform_v.discard(n)
        self.last_v = (n, expr)
        if low16_only:
            if not isinstance(self.low16_only_v, set):
                self.low16_only_v = set()
            self.low16_only_v.add(n)
        else:
            self.clear_low16(n)
        self.used_v.add(n)
        self.emit('v%d = %s;' % (n, expr) if self.exec_known else 'if (exec & 1) v%d = %s;' % (n, expr))

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

    def scalar_expr_uniform(self, expr):
        # These identifiers are runtime inputs in the scalar expressions emitted below.
        # In particular, a compare can carry a lane mask into an otherwise scalar select.
        for token in re.findall(r'\b(?:s\d+|v\d+|vcc|exec|m0|scc)\b', expr):
            if token == 'scc':
                if not self.scc_uniform:
                    return False
            elif not self.uniform_source(token):
                return False
        return True

    def set_scc(self, expr, proof=None):
        """Emit an SCC assignment and track its uniformity from `proof` (an expression over the
        sources) when the emitted text names a register, such as m0, that the proof cannot judge."""
        self.scc_effect = 'emitted'
        if self.tracks_uniformity():
            self.scc_uniform = self.scalar_expr_uniform(expr if proof is None else proof)
        self.emit('scc = %s;' % expr)

    def set_s(self, n, expr, sym=None, scc=None, uniform=None):
        """Assign scalar register n at run time and/or symbolically.

        Any write ends n's life as a lane mask; a caller that writes a mask marks n again after this call.
        The uniformity of expr is judged first, while a source that is n itself still has its old mark."""
        if self.tracks_uniformity() and (sym is None or sym[0] == 'const'):
            if uniform is None:
                uniform = self.scalar_expr_uniform(expr)
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
            # Invariant: since_join gains an SGPR only here, when set_s proves it uniform; forget_writes drops any SGPR
            # an instruction writes first, so a write through an unmodelled path never keeps a stale mark.
            if getattr(self, 'since_join', None) is not None:
                (self.since_join.add if uniform else self.since_join.discard)(n)
        getattr(self, 'masks', set()).discard(n)  # unit probes build a Stage without __init__
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

    def descriptor(self, tok):
        """Resource key for a descriptor operand like s[8:11]: (source, byte offset or None)."""
        r = reg_range(tok)
        s = self.sym.get(r[1]) if r else None
        if s and s[0] == 'desc':
            # A descriptor can start inside a wider load (s_load_dwordx16 fetches two image
            # descriptors): its offset is the load's offset plus the first register's dword index.
            return s[1], (None if s[2] is None else s[2] + 4 * s[3])
        if s and s[0] == 'ptr':
            return s[1], None
        raise Unsupported('descriptor operand %s holds %s' % (tok, s))

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
        words = [self.is_const(n) for n in range(r[1], r[2] + 1)]
        return None if any(w is None for w in words) else words

    def execz_refusals(self):
        """{offset: reason} for compute blocks that s_cbranch_execz skips and the one-lane model cannot run.

        The wave runs such a block when any lane is active, so its scalar results are the same for every lane; the
        model skips the block for an inactive lane. That lane can differ from the wave only by reading, after the
        join, scalar state the block defines (no lane active means the wave skipped the block too). The block is
        therefore accepted unchanged when every scalar register, mask and SCC it may define is dead where it joins,
        and it has no effect outside scalar registers and no branch out of it. An exec write is accepted in a
        straight-line block (no branch in it, none into it) when it cannot set the bit of a lane whose bit is clear
        (exec_stays_clear): such a lane's bit is clear on entry, so the wave leaves it clear and its exec at the
        join is the model's. A live definition, another exec write, a refused effect, a branch leaving the block
        or a backward s_cbranch_execz is refused."""
        offsets = [ins[0] for ins in self.ins]
        index = {off: i for i, off in enumerate(offsets)}
        effects = [scalar_effects(name, asm) for _, _, name, asm in self.ins]
        succ = []
        targets = []  # every branch target, the proven calculated s_setpc_b64 ones included
        for i, (off, n, name, asm) in enumerate(self.ins):
            to = int(parse(asm)[1][0].split('_')[1], 16) if name.startswith(('s_cbranch', 's_branch')) else None
            if name == 's_setpc_b64':  # a proven calculated jump reaches its target
                to = getattr(self, 'calculated_pc', {}).get(off)
            nxt = [i + 1] if i + 1 < len(self.ins) and name not in ('s_branch', 's_endpgm') else []
            if to is not None:
                targets.append(to)
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
        refused = {}
        for b, (off, n, name, asm) in enumerate(self.ins):
            if name != 's_cbranch_execz':
                continue
            t = index.get(int(parse(asm)[1][0].split('_')[1], 16))
            if t is None or t <= b:
                refused.setdefault(off, 's_cbranch_execz to a block before it')
                continue
            live = live_in[t] | {'m0'}  # m0 is also read implicitly; treat it as live
            saved = set()  # registers whose every write in the block so far is a copy of exec
            # The copies are tracked in program order, which is every path only in a straight-line block: no branch
            # in it and no branch into it. Elsewhere an exec write is refused.
            straight = not any(self.ins[k][2].startswith(('s_cbranch', 's_branch')) for k in range(b + 1, t)) and \
                not any(index.get(target) in range(b + 1, t) for target in targets)
            for i in range(b + 1, t):
                at, _, inner, inner_asm = self.ins[i]
                may = effects[i][2]
                reason = None
                if inner.startswith(EXECZ_REFUSED):
                    reason = 'effect outside scalar registers'
                elif inner.startswith(('s_cbranch', 's_branch')) and not (
                        b < index.get(int(parse(inner_asm)[1][0].split('_')[1], 16), -1) <= t):
                    reason = 'branch out of the block'
                elif may & EXEC_KEYS and not (straight and exec_stays_clear(inner, inner_asm, saved)):
                    reason = 'exec write'
                elif (may - EXEC_KEYS) & live:
                    reason = 'definition of %s, live where the block joins' % ', '.join(sorted((may - EXEC_KEYS) & live))
                saved = exec_copies_after(inner, inner_asm, saved, may)
                if reason:
                    refused.setdefault(at, '%s inside a block s_cbranch_execz skips per lane: %s' % (inner, reason))
        return refused

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
        CFG region (between the branch and its immediate post-dominator, or over the loop range for a backward edge)."""
        ins = self.ins
        if not ins:
            return ({}, {}, {}) if with_cfg else {}
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
        div_targets = {}
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
            div_targets[off] = (regs, affected)
        return joined, forward_splits, div_targets

    def translate(self):
        refusals = getattr(self, 'refusals', None)  # unit probes build a Stage without __init__
        try:
            self.calculated_pc = calculated_pc_proof(self.ins, getattr(self, 'kind', None))
        except Unsupported as reason:
            if refusals is None:
                raise
            refusals.append((str(reason), self.kind))
            self.calculated_pc = {}
        targets = set()
        targets.update(target for target in self.calculated_pc.values() if target is not None)
        # (branch, target) of every branch, the proven s_setpc_b64 jumps included
        branches = [(pc, target) for pc, target in self.calculated_pc.items() if target is not None]
        for off, n, name, asm in self.ins:
            if name.startswith('s_cbranch') or name == 's_branch':
                target = int(parse(asm)[1][0].split('_')[1], 16)
                targets.add(target)
                branches.append((off, target))
        join_writes = self.join_writes(branches)
        refused = self.execz_refusals() if getattr(self, 'kind', None) == 'cs' else {}  # probes skip __init__
        joined, forward_splits, div_targets = self.joined_registers(with_cfg=True)
        init_sym = dict(self.sym)
        init_exec_known = getattr(self, 'exec_known', True)
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

        for _ in range(max(4, len(self.ins) + 2)):
            scc_effects = {}
            self.lines = []
            self.blocks = []          # (offset, [lines], terminator)
            self.sym = dict(init_sym)
            self.exec_known = init_exec_known
            self.wqm_exec = init_wqm_exec
            self.masks = set(init_masks)
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
            self.varying_s = set()
            self.scc_uniform = True
            self.low16_only_v = set()   # VGPRs whose upper 16 bits are undefined after odd-dmask d16 image_load
            self.pinned_low16_v = set() # low16_only_v live across basic-block boundaries
            self.whole_read_v = set()   # VGPRs read outside a proven low-16-bit consumer
            self.has_branches = bool(targets)
            self.entry_table_pointers = {
                reg: sym[1] for reg, sym in init_sym.items()
                if isinstance(reg, int) and sym[0] == 'ptr' and sym[1][0] == 'table'
            }
            self.descriptor_block_first = 0
            self.uniform_expr = {}    # proven VGPR -> uniform HLSL expression over scalars that defined it
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

            for instruction_index, (off, n, name, asm) in enumerate(self.ins):
                if off in targets and off != start:
                    self.pinned_low16_v.update(self.low16_only_v)
                    self.blocks.append((start, self.lines, None))
                    self.lines, start = [], off
                    self.descriptor_block_first = instruction_index
                self.pc = off
                self.descriptor_instruction_index = instruction_index
                self.scc_effect = None
                # Before the first branch target or terminator every path has run exactly the instructions above, so
                # the program-order tracking is exact there even in a program with branches. Code after s_endpgm in a
                # branch-free program is past it too (dead code).
                self.past_entry = bool(self.blocks) or 0 in targets
                if off in targets:
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
                    clobber = here if off in joined else every_sgpr
                    fwd_clobber = forward_splits.get(off, clobber)
                    blocked = force_unproven | force_at.get(off, set())
                    kept_u = {
                        r for r in pre_uniform
                        if r not in clobber or (r not in fwd_clobber and r not in blocked)
                    }
                    tentative_kept.update((kept_u & clobber) - blocked)
                    self.uniform_v.clear()
                    self.since_join, self.scc_uniform = kept_u, False
                    # A constant is only known here if no other path into the join writes the register: a loop counter
                    # set before the loop and incremented in it reaches the loop head with a new value on every pass.
                    # set_s also assigns constants at run time, so the register reads its real value once it is dropped.
                    for reg in join_writes.get(off, ()) | (clobber & every_sgpr):
                        if (self.sym.get(reg) or ('',))[0] == 'const':
                            del self.sym[reg]
                    self.defined_uniform_s = set(kept_u) | {
                        k for k, v in self.sym.items()
                        if isinstance(k, int) and v and (
                            (v[0] == 'const' and len(v) == 2) or v[0] in ('ud', 'fixed_buffer_word')
                        )
                    }
                if name.startswith('s_cbranch') or name == 's_branch' or (
                        name == 's_setpc_b64' and self.calculated_pc.get(off) is not None):
                    b_target = self.calculated_pc[off] if name == 's_setpc_b64' else int(parse(asm)[1][0].split('_')[1], 16)
                    there = joined.get(b_target, set())
                    if b_target > off:
                        incoming.setdefault(b_target, []).append(({reg: self.sym.get(reg) for reg in there}, written & there))
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
                try:
                    if off in refused:
                        raise Unsupported(refused[off])
                    base = re.sub(r'_e(32|64)$', '', name)
                    if base.startswith('s_cbranch_'):
                        cond = base[10:]
                        cond_uniform = cond in ('scc0', 'scc1') and self.scc_uniform
                        if not cond_uniform:
                            b_target = int(parse(asm)[1][0].split('_')[1], 16)
                            regs, affected = div_targets.get(off, (set(), set()))
                            regs = {r for r in regs if isinstance(r, int)}
                            if b_target <= off:
                                regs.update(r for r in joined.get(b_target, ()) if isinstance(r, int))
                            for tgt_off in affected:
                                bad_at_target.setdefault(tgt_off, set()).update(regs)
                    term = self.one(name, asm)
                except Unsupported as reason:
                    if refusals is None:
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
                if self.scc_effect:
                    scc_effects[off] = self.scc_effect
                if term:
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
            if raised_unsupported is not None:
                raise raised_unsupported
            break
        else:
            reason = 'v_readfirstlane_b32 uniformity proof exceeded pass limit'
            if refusals is None:
                raise Unsupported(reason)
            refusals.append((reason, self.kind))
        if self.lines:
            self.blocks.append((start, self.lines, None))
        self.materialized = materialize  # declarations() assigns these before the body (and before any loop)
        if 'descriptor' in scc_effects.values():
            for name in self.descriptor_scc_reads(dict(branches), scc_effects):
                reason = 'scc from descriptor data in %s' % name
                if refusals is None:
                    raise Unsupported(reason)
                refusals.append((reason, self.kind))
        self.pinned_low16_v.update(self.low16_only_v)
        leak = self.pinned_low16_v & self.whole_read_v
        if leak:
            reason = 'whole-word read of v%d after odd d16 image_load' % min(leak)
            if refusals is None:
                raise Unsupported(reason)
            refusals.append((reason, self.kind))

    def descriptor_scc_reads(self, branches, effects):
        """Find reads reachable with an elided SCC write, including through joins and back edges."""
        instructions = {off: (size, name, asm) for off, size, name, asm in self.ins}
        pending = [(0, False)]
        seen = set()
        while pending:
            off, stale = pending.pop()
            if (off, stale) in seen or off not in instructions:
                continue
            seen.add((off, stale))
            size, name, asm = instructions[off]
            if stale and off not in self.calculated_pc and reads_scc(name, asm):
                yield name
            if off in effects:
                stale = effects[off] == 'descriptor'
            if off in branches:
                pending.append((branches[off], stale))
            if name not in ('s_branch', 's_setpc_b64', 's_endpgm'):
                pending.append((off + size, stale))

    def join_writes(self, branches):
        """Branch target -> the scalar registers another path can write before it reaches the target: the loop body
        [target, branch] for a backward branch, and the skipped instructions (branch, target) for a forward one."""
        writes = {}
        for branch, target in branches:
            regs = writes.setdefault(target, set())
            for off, _n, name, asm in self.ins:
                if not (target <= off <= branch if target <= branch else branch < off < target):
                    continue
                _, ops, _ = parse(asm)
                if name.startswith('s_movreld'):
                    regs.update(range(128))  # M0 picks the destination
                for tok in destinations(name, ops):
                    r = reg_range(tok)
                    if r and r[0] == 's':
                        regs.update(range(r[1], r[2] + 1))
                    elif tok == 'm0':
                        regs.add('m0')
        return writes

    def tracks_uniformity(self):
        return all(hasattr(self, name) for name in ('uniform_v', 'varying_s', 'scc_uniform'))

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
        if tok in FLOATS:
            return True
        try:
            int(tok, 0)
        except ValueError:
            return False
        return True

    def one(self, name, asm):
        _, ops, mods = parse(asm)
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
                 not getattr(self, 'past_entry', True) and getattr(self, 'exec_known', False)) and
                (not mods or (base in VEC_F or base in VEC_U_TO_F) and
                 all(m in ('clamp', 'mul:2', 'mul:4', 'div:2') for m in mods))):
            candidate = reg_range(ops[0])
            if candidate and candidate[0] == 'v' and candidate[1] == candidate[2]:
                sources = ops[1:]
                if base in VEC_F:
                    sources = [re.sub(r'^-(?=[^\d.])', '', tok).strip('|') for tok in sources]
                if sources and all(self.uniform_source(tok) for tok in sources):
                    dst = candidate[1]
        self.last_v = None
        term = self.lower(base, asm, ops, mods)
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
            self.uniform_v.add(dst)
            if hasattr(self, 'uniform_expr'):
                # Proven sources are scalars, constants and VGPRs; each VGPR needs its own current expression.
                expr = self.last_v[1] if self.last_v and self.last_v[0] == dst else None
                vs = [int(n) for n in re.findall(r'\bv(\d+)\b', expr or '')]
                if expr is None or any(n not in self.uniform_expr for n in vs):
                    self.uniform_expr.pop(dst, None)
                else:
                    self.uniform_expr[dst] = re.sub(r'\bv(\d+)\b', lambda m: '(%s)' % self.uniform_expr[int(m[1])],
                                                    expr)
        return term

    def dpp_pair(self, base, asm, ops, mods):
        """A DPP16 spelling whose SRC0 is a quad pair broadcast (#2125 item C).

        RDNA 2 ISA section 12.19 (DPP16): lane i of a quad reads SRC0 of lane quad_perm[i] of the same quad. A pair
        broadcast ([1,1,3,3], [0,0,2,2], [2,3,2,3], [0,1,0,1]) moves one lane of every horizontal or vertical pair
        into the other and leaves the first lane alone. The word of each of the four lanes is read with the shader
        model 6.0 QuadReadLaneAt (its lane argument must be a constant, so all four reads are unconditional) and this
        lane keeps the one quad_perm names for its own position in the quad. That needs all four lanes of the quad to
        run the instruction in the same flow, as in a pixel shader with helper lanes: straight-line code and a
        whole-quad exec mask (proved by s_wqm exec, exec on the entry mask; a restored exact mask loses it). Every
        other control (other permutations, row_*, wave_*, a partial row or bank mask, fi, a vertex stage) is refused by
        name, since the single lane cannot read a neighbour."""
        perms = [m for m in mods if m.startswith('quad_perm:')]
        rest = [m for m in mods if not m.startswith('quad_perm:')]
        if len(perms) != 1:
            raise Unsupported('%s_dpp with %s' % (base, ' '.join(mods)))
        for m in rest:
            if m not in ('row_mask:0xf', 'bank_mask:0xf', 'bound_ctrl:0', 'bound_ctrl:1'):
                raise Unsupported('%s_dpp with %s' % (base, m))
        # With all four lanes active no source lane is disabled, so bound_ctrl has nothing to replace.
        perm = tuple(int(c) for c in re.findall(r'\d', perms[0]))
        if perm not in DPP_PAIR:
            raise Unsupported('%s_dpp with %s' % (base, perms[0]))
        if (self.kind != 'ps' or getattr(self, 'has_branches', True) or not getattr(self, 'exec_known', False) or
                not getattr(self, 'wqm_exec', False) or getattr(self, 'fetched', False)):
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
        self.needs_sm6 = True
        self.used_v.add(DPP_SCRATCH)
        self.uniform_v.discard(DPP_SCRATCH)
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

    def lower(self, base, asm, ops, mods):
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
                self.varying_s.add(reg)
            if base in ('s_add_u32', 's_addc_u32'):
                self.scc_uniform = False
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
            if hasattr(self, 'uniform_v'):
                self.uniform_v.clear()
            if hasattr(self, 'scc_uniform'):
                self.scc_uniform = False  # the hardware SCC result is not modelled
            return None
        if base in ('s_inst_prefetch', 's_waitcnt', 's_nop', 'v_nop', 's_clause', 's_sopp_0f',
                    's_setreg_imm32_b32', 's_code_end', 's_setprio'):
            return None
        if base == 's_endpgm':
            return ('end',)
        if base == 's_branch':
            return ('jump', int(ops[0].split('_')[1], 16))
        if base.startswith('s_cbranch_'):
            cond = {'scc0': '!scc', 'scc1': 'scc', 'vccz': '(vcc & 1) == 0', 'vccnz': '(vcc & 1) != 0',
                    'execz': '(exec & 1) == 0', 'execnz': '(exec & 1) != 0'}[base[10:]]
            return ('branch', cond, int(ops[0].split('_')[1], 16))
        if base == 'exp':
            return self.export(asm)
        if base in ('ds_write_b32', 'ds_read_b32', 'ds_read2_b32', 'ds_write_b64', 'ds_read_b64', 's_barrier'):
            if getattr(self, 'kind', None) != 'cs':
                raise Unsupported(base)
            # The refusal census continues past failed instructions. Scan this program only once.
            if not hasattr(self, 'lds_error'):
                self.lds_error = None
                try:
                    if getattr(self, 'pipe', None) is None:
                        raise Unsupported('LDS exchange needs pipeline metadata')
                    check_lds_exchange(self.ins, self.pipe.group, self.pipe.lds_words)
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
                    lo = _lds_pair(ops[1 if write else 0])
                    if write:
                        d0, d1 = self.u('v%d' % lo), self.u('v%d' % (lo + 1))
                        self.emit('a = (%s + %du) >> 2;' % (address, offsets[0]))
                        self.emit('lds[a] = %s;' % d0)
                        self.emit('lds[a + 1u] = %s;' % d1)
                    else:
                        self.emit('a = (%s + %du) >> 2;' % (address, offsets[0]))
                        self.set_v(lo, 'lds[a]')
                        self.set_v(lo + 1, 'lds[a + 1u]')
            return None
        if base.endswith('_dpp'):
            return self.dpp(base[:-len('_dpp')], ops, mods)
        handler = getattr(self, 'op_' + base, None)
        if handler:
            return handler(ops, mods)
        if base in MUBUF_ATOMIC_OPS:
            return self.buffer_atomic(base, ops, mods)
        if base.endswith('_dpp'):
            return self.dpp(base[:-len('_dpp')], ops, mods)
        if base.startswith('v_cmp'):
            # Before the VOP1/VOP2 SDWA route below: VOPC SDWA names also end in _sdwa.
            return self.compare(base, ops, mods)
        if base.endswith('_sdwa'):
            return self.sdwa(base[:-len('_sdwa')], ops, mods)
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
            low16 = base in ('v_cvt_f32_f16', 'v_cvt_f32_ubyte0', 'v_cvt_f32_ubyte1')
            self.set_vf(dst, text_f.format(*[self.u(o, low16_only=low16) for o in ops[1:1 + arity]]), mods)
            if (base.startswith('v_cvt_f32_ubyte') and not mods and self.exec_known and
                    not getattr(self, 'has_branches', True)):
                # The whole register now holds float(0..255), exactly +0.0 or an integer in 1..255, until the next write.
                # Every VGPR write goes through set_v, which drops the proof; a path writing a VGPR without set_v
                # would have to drop it too.
                if not isinstance(self.byte_float_v, set):
                    self.byte_float_v = set()
                self.byte_float_v.add(dst)
            return None
        if base in VEC_U:
            arity, text = VEC_U[base]
            if mods:
                raise Unsupported('%s with %s' % (base, mods))
            if len(ops) != arity + 1:
                raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
            if base == 'v_and_b32':
                c0, c1 = self.const_u32(ops[1]), self.const_u32(ops[2])
                args = [self.u(ops[1], low16_only=c1 is not None and (c1 & 0xffff0000) == 0),
                        self.u(ops[2], low16_only=c0 is not None and (c0 & 0xffff0000) == 0)]
            else:
                args = [self.u(o) for o in ops[1:1 + arity]]
            return self.set_v(reg_range(ops[0])[1], text.format(*args))
        if base in SCA:
            return self.scalar(base, ops)
        if base == 'buffer_load_format_xyzw':
            # Not translated: the four-component form has no startup request that needs it yet.
            raise Unsupported('%s requires descriptor FORMAT and DST_SEL' % base)
        raise Unsupported(base)

    # VOP1/VOP2 with DPP16 quad broadcast (#2125 item A): quad_perm:[k,k,k,k] makes every lane read SRC0 from
    # lane k of its own quad (RDNA 2 ISA, "DPP": a lane reads lane (lane & ~3) | perm[lane & 3] of the row; LLVM 15
    # AMDGPU DPP quad_perm). A pixel stage's four consecutive lanes are a 2x2 pixel quad (lane = x & 1 | (y & 1) << 1),
    # which is what SM 6.0 QuadReadLaneAt(value, k) reads, so the stage is compiled as ps_6_0 (Stage.needs_sm6).
    # Accepted only where the whole quad is proven active: a straight-line pixel stage that ran s_wqm exec, exec and
    # whose EXEC is still that mask (helper lanes are active, so no source lane is off). bound_ctrl then has
    # nothing to decide (the source lane is never disabled), so it is accepted and ignored. Everything else a DPP word can say stays refused: other controls (pair
    # broadcasts, row shifts, mirrors, DPP8), a masked row or bank, fi, and a vertex stage.
    DPP_TEMP = 256   # hidden VGPR that holds the quad read; the hardware has v0..v255 only
    DPP_OUTPUT_MODS = ('clamp', 'mul:2', 'mul:4', 'div:2')

    def dpp_broadcast(self, base, ops, mods):
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
        if fields.get('bound_ctrl', '1') != '1':
            raise Unsupported('%s_dpp bound_ctrl:%s' % (base, fields['bound_ctrl']))
        if (self.kind != 'ps' or getattr(self, 'has_branches', True) or getattr(self, 'blocks', [])
                or not getattr(self, 'exec_known', False)):
            raise Unsupported('%s_dpp outside a straight-line pixel stage with the entry EXEC mask' % base)
        if not getattr(self, 'wqm_exec', False):
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
        self.needs_sm6 = True
        self.set_v(self.DPP_TEMP, 'QuadReadLaneAt(%s, %su)' % (self.u(token), lanes.group(1)))
        operands = [ops[0], rewritten] + ops[2:]
        if base == 'v_mov_b32':
            if rest or len(operands) != 2:
                raise Unsupported('v_mov_b32_dpp with %d operands or an output modifier' % len(ops))
            return self.lower(base, '%s %s' % (base, ', '.join(operands)), operands, [])
        return self.lower(base, '%s %s %s' % (base, ', '.join(operands), ' '.join(rest)), operands, rest)

    # VOP1/VOP2 with a DPP16 quad_perm source (AMD RDNA 2 ISA, "Data-Parallel Primitives"): lane i of
    # each aligned group of four reads src0 from lane quad_perm[i] of that group. The three exchanges
    # map onto the SM 6.0 quad intrinsics, which read the other lane of the 2x2 pixel quad and keep
    # helper lanes active. Only a pixel stage at entry EXEC qualifies, and only after `s_wqm exec, exec`
    # proved whole-quad EXEC: the hardware can return 0 or the old value for a source lane whose EXEC bit
    # is off (an uncovered pixel of an edge quad), and the intrinsics have no such case. Other permutations, DPP8, row shifts and partial masks stay refused.
    def dpp(self, base, ops, mods):
        if any(re.fullmatch(r'quad_perm:\[([0-3]),\1,\1,\1\]', mod) for mod in mods):
            return self.dpp_broadcast(base, ops, mods)
        perm = None
        for m in mods:
            match = re.fullmatch(r'quad_perm:\[([0-3]),([0-3]),([0-3]),([0-3])\]', m)
            if match and perm is None:
                perm = tuple(int(x) for x in match.groups())
            elif m not in ('row_mask:0xf', 'bank_mask:0xf', 'bound_ctrl:0', 'bound_ctrl:1'):
                raise Unsupported('%s_dpp with %s' % (base, m))
        if perm in DPP_PAIR:
            return self.dpp_pair(base, '%s %s' % (base, ', '.join(ops)), ops, mods)
        intrinsic = QUAD_SWAPS.get(perm)
        if not intrinsic:
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
        if len(ops) != arity + 1 or any(not re.fullmatch(r'v\d+', o) for o in ops):
            raise Unsupported('%s_dpp needs VGPR operands without modifiers' % base)
        if (getattr(self, 'kind', None) != 'ps' or getattr(self, 'has_branches', True) or getattr(self, 'blocks', True)
                or not getattr(self, 'exec_known', False)):
            raise Unsupported('%s_dpp requires a straight-line pixel shader with entry EXEC' % base)
        if not getattr(self, 'wqm_exec', False):
            raise Unsupported('%s_dpp requires whole-quad EXEC (s_wqm of exec first): a partly covered quad has '
                              'inactive source lanes that the hardware masks and QuadReadAcross does not' % base)
        self.needs_sm6 = True
        # src0 is the exchanged lane; a VOP2 src1 stays this lane's own register.
        swapped = '%s(%s)' % (intrinsic, self.u(ops[1]))
        args = [('F(%s)' % swapped) if kind == 'f' else swapped] + \
               [self.f(o) if kind == 'f' else self.u(o) for o in ops[2:1 + arity]]
        expr = text.format(*args)
        self.set_v(reg_range(ops[0])[1], 'U(%s)' % expr if kind == 'f' else expr)

    # VOP1/VOP2 with sub-dword addressing: select the source fields, run the same operation, then place
    # the low bits of the result in the destination field. The integer forms reuse VEC_U unchanged,
    # v_mov_b32 is the same path with the selected source as its result,
    # VEC_U_TO_F conversions apply output modifiers before dst_sel, VEC_F forms stay refused, and
    # dst_unused UNUSED_SEXT has no lowering.
    def sdwa(self, base, ops, mods):
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
            if fields.get('src0_sel') != 'DWORD' and self.is_lane_mask(ops[1].removeprefix('sext(').removesuffix(')')):
                raise Unsupported('v_mov_b32 SDWA %s of lane mask %s' % (fields.get('src0_sel'), ops[1]))
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

    # scalar ALU with descriptor-word bookkeeping
    def scalar(self, base, ops):
        if base == 's_movk_i32' and len(ops) != 2:
            raise Unsupported('s_movk_i32 with %d operands, expected 2' % len(ops))
        text, scc = SCA[base]
        if base in SCC_UNMODELLED and self.tracks_uniformity():
            self.scc_uniform = False
        dst = reg_range(ops[0])
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
        # s_mul_hi_u32 needs the whole 32-bit word of each source, like the halfword pack.
        word_pack = base in ('s_pack_hh_b32_b16', 's_sext_i32_i8', 's_sext_i32_i16', 's_mul_hi_u32')
        if shift_add or word_complement or word_pack:
            # The per-lane mask cannot supply the full word needed by these operations. s_mul_hi_u32 into an
            # SGPR that held a mask replaces it with a plain word (set_s drops the mark), so only vcc/exec count.
            mask_dest = (dname in ('exec', 'exec_lo', 'exec_hi', 'vcc', 'vcc_lo', 'vcc_hi') if base == 's_mul_hi_u32'
                         else self.is_lane_mask(dname))
            if mask_dest or any(self.is_lane_mask(t) for t in srcs):
                raise Unsupported('%s on a lane mask' % base)
        if shift_add and dname != 'm0' and (not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105):
            raise Unsupported('%s destination %s' % (base, dname))
        if word_pack:
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
        if any(s and s[0] in ('desc', 'descnum', 'ptr', 'ptrhi') for s in syms):
            if scc or base in SCC_UNMODELLED:
                self.scc_effect = 'descriptor'
                if self.tracks_uniformity():
                    self.scc_uniform = False
            if shift_add or word_complement:
                raise Unsupported('scc from descriptor data')
            if word_pack:
                raise Unsupported('%s on descriptor data' % base)
            if dname == 'm0':
                if scc:
                    raise Unsupported('scc from descriptor data')
                self.sym['m0'] = syms[0] if base == 's_mov_b32' else ('descnum',)
                return None
            if dname in ('exec', 'vcc', 'exec_lo', 'vcc_lo'):
                raise Unsupported('mask from descriptor data')
            d = reg_range(dname)[1]
            own = self.sym.get(d)
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
            elif own and own[0] in ('desc', 'ptr', 'ptrhi') and any(s == own for s in syms):
                if own[0] == 'desc':
                    self.pipe.modified.add(desc_key(own))
                    # The word keeps its descriptor identity for loads; the store check (store_descriptor) sees the
                    # mark. In a compute stage, clearing the stride field of dword 1 (bits 29:16, keeping the base
                    # address bits 15:0 and bits 31:30) is marked on its own: a raw byte-address store accepts it.
                    stride0 = (getattr(self, 'kind', None) == 'cs' and base == 's_and_b32' and len(own) == 4
                               and own[3] % 4 == 1 and sorted(srcs) == sorted([dname, '0xc000ffff']))
                    self.sym[d] = own[:4] + ('stride0' if stride0 else 'modified',)
                    if not stride0:
                        getattr(self.pipe, 'other_modified', set()).add(desc_key(own))
                else:
                    self.sym[d] = own if len(own) > 2 else own + ('added',)  # the added offset is not tracked
            else:
                self.sym[d] = ('descnum',)
            return None
        values = [self.u(t) for t in srcs]
        expr = text.format(*values)
        if shift_add:
            # carry = bits shifted out of S0, or a wrap of the 32-bit sum; read before the destination is written
            n = SHIFT_ADD[base]
            self.set_scc('(%s >> %d) != 0 || (%s) < %s' % (values[0], 32 - n, expr, values[1]))
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
                copy = base in ('s_mov_b64', 's_mov_b32') and self.exec_known and (syms[0] == ('execcopy',))
                if copy:
                    # A restored copy can re-activate lanes a WQM added: drop every VGPR proof.
                    self.wqm_exec = False  # and the copy can be older than the WQM
                    if hasattr(self, 'uniform_v'):
                        self.uniform_v.clear()
                    return None
                self.exec_known = False
            self.emit('%s = (%s) & 1;' % (var, expr))
            if scc:
                self.set_scc(scc.replace('$', var))
            return None
        d = reg_range(dname)[1]
        is_mask = pair or any(t in ('exec', 'exec_lo', 'vcc', 'vcc_lo') or (reg_range(t) and reg_range(t)[0] == 's' and reg_range(t)[1] in self.masks)
                              for t in srcs) and base.split('_')[1] in ('mov', 'and', 'or', 'xor', 'andn2', 'orn2', 'nand', 'nor', 'xnor', 'not', 'cselect')
        if base in ('s_mov_b32', 's_mov_b64', 's_movk_i32') and re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', srcs[0]) and srcs[0] not in FLOATS:
            value = int(srcs[0], 0)
            self.set_s(d, values[0], ('const', value & 0xffffffff))
            if pair:
                self.set_s(d + 1, '0x%08xu' % ((value >> 32) & 0xffffffff), ('const', (value >> 32) & 0xffffffff))
            self.masks.discard(d)
            return None
        if base in SCALAR_CONST_FOLD:
            # Constant operands give a known word, so a descriptor built from them (an inline sampler from
            # s_movk_i32, s_bfm_b32, s_mov_b32 and s_brev_b32 into s[N:N+3]) is a constant descriptor.
            def known(tok):
                r = reg_range(tok)
                if r and r[0] == 's' and r[1] == r[2]:
                    return self.is_const(r[1])
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
        self.set_s(d, expr, scc=scc)
        if is_mask:
            self.masks.add(d)
        if pair and base != 's_mov_b64':
            self.set_s(d + 1, '0u')
        elif pair:
            self.set_s(d + 1, self.S(high) if high is not None else '0u', uniform=high_uniform)
        return None

    def set_exec_scc(self, expr):
        self.exec_known = False
        self.scc_effect = 'emitted'
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
            self.set_s(d, result)
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
        expr = 'scc ? %s : %s' % (self.u(low[1]), self.u(low[2]))
        if low[0] == 'm0':
            self.sym.pop('m0', None)
            self.emit('m0 = %s;' % expr)
            return
        if low[0] in ('exec', 'vcc', 'exec_lo', 'vcc_lo'):
            # In wave32, exec_lo and vcc_lo hold the whole lane mask, as the generic scalar path reads them.
            mask = 'exec' if low[0].startswith('exec') else 'vcc'
            if mask == 'exec':
                self.exec_known = False
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

    def op_s_load_dwordx4(self, ops, mods, count=4):
        imm = smem_mods(mods)
        if len(ops) != 3:
            raise Unsupported('s_load with %d operands' % len(ops))
        dst, base = reg_range(ops[0]), reg_range(ops[1])
        if not dst or dst[0] != 's' or dst[2] - dst[1] + 1 != count or dst[2] > 105:
            raise Unsupported('s_load destination %s for %d dwords' % (ops[0], count))
        if not base or base[0] != 's' or base[2] - base[1] != 1 or base[2] > 105:
            raise Unsupported('s_load base %s' % ops[1])
        # A loaded descriptor or word is not a lane mask; some forms below only record a symbol, not set_s.
        for n in range(dst[1], dst[2] + 1):
            getattr(self, 'masks', set()).discard(n)
        p = self.sym.get(base[1])
        if not p or p[0] != 'ptr':
            raise Unsupported('s_load through s%d holding %s' % (base[1], p))
        # 'off' names no offset operand at all, so the byte offset is zero, as for the literal 0.
        token = smem_offset(ops[2])
        if re.fullmatch(r'-?(?:0[xX][0-9a-fA-F]+|\d+)', token):
            off = int(token, 0)
            if off < 0 or off > 0xfffff:
                raise Unsupported('SMEM immediate offset %s' % ops[2])
            if imm is not None:
                raise Unsupported('SMEM literal offset %s with offset:%#x' % (ops[2], imm))
            if p[1][0] == 'root':
                buf = self.pipe.resource((p[1], None), 'buf')
                for i in range(count):
                    self.set_s(dst[1] + i, '%s.Load(%d)' % (buf, off + 4 * i))
                return
            for i in range(count):
                self.sym[dst[1] + i] = ('desc', p[1], off, i)
            return
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
        raise Unsupported('s_load with a register offset through %s' % (p[1],))

    def op_s_bfm_b32(self, ops, mods):
        """Fold constant bitfield masks: titles build inline sampler words with them."""
        vals = []
        for o in ops[1:3]:
            r = reg_range(o)
            if r and r[0] == 's':
                vals.append(self.is_const(r[1]))
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
        d = reg_range(ops[0])[1]
        if None not in vals:
            value = (((1 << (vals[0] & 31)) - 1) << (vals[1] & 31)) & 0xffffffff
            self.set_s(d, '0x%08xu' % value, ('const', value))
            return None
        self.set_s(d, '(((1u << (%s & 31)) - 1) << (%s & 31))' % (self.u(ops[1]), self.u(ops[2])))
        return None

    def scalar_const(self, tok):
        """Value of a 32-bit scalar source that is known at translation time, else None."""
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
            low, high = self.i64_words(ops[1])
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
            self.used_s.add(r)
            if self.tracks_uniformity():
                (self.varying_s.discard if uniform else self.varying_s.add)(r)
        self.scc_effect = 'emitted'
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
        buf = self.pipe.resource(self.descriptor(ops[1]), 'buf')
        token = smem_offset(ops[2])
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
        # All words use the original offset, even when a destination overwrites its register.
        uniform = self.tracks_uniformity() and self.uniform_source(token)
        for i in range(count):
            self.set_s(dst[1] + i, '%s.Load(%s + %d)' % (buf, offset, extra + 4 * i), uniform=uniform)
            if is_lit:
                self.sym[dst[1] + i] = ('fixed_buffer_word',)

    def op_s_buffer_load_dwordx2(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 2)

    def op_s_buffer_load_dwordx4(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 4)

    def op_s_buffer_load_dwordx8(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 8)

    def op_s_buffer_load_dwordx16(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 16)

    def buffer_address(self, ops, mods):
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
        key = self.descriptor(ops[2])
        buf = self.pipe.resource(key, 'buf')
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
        if idxen and offen:
            if key in self.pipe.modified:
                raise Unsupported('indexed load through a modified descriptor')
            r = reg_range(ops[1])
            if not r or r[0] != 'v' or r[2] - r[1] != 1 or r[2] > 255:
                raise Unsupported('buffer vaddr %s for idxen offen' % ops[1])
            stride = self.pipe.ud(('stride', self.pipe.resources[key]['index']))
            terms.append('%s * ud%d' % (self.u('v%d' % r[1]), stride))
            terms.append(self.u('v%d' % r[2]))
        elif idxen:
            if key in self.pipe.modified:
                raise Unsupported('indexed load through a modified descriptor')
            r = reg_range(ops[1])
            if not r or r[0] != 'v' or r[1] != r[2] or r[1] > 255:
                raise Unsupported('buffer vaddr %s for idxen' % ops[1])
            stride = self.pipe.ud(('stride', self.pipe.resources[key]['index']))
            terms.append('%s * ud%d' % (self.u(ops[1]), stride))
        elif offen:
            r = reg_range(ops[1])
            if not r or r[0] != 'v' or r[1] != r[2] or r[1] > 255:
                raise Unsupported('buffer vaddr %s for offen' % ops[1])
            terms.append(self.u(ops[1]))
        elif ops[1] != 'off':
            raise Unsupported('buffer vaddr %s without idxen or offen' % ops[1])
        if soffset is not None:
            terms.append(soffset)
        terms.append('%d' % offset)
        return buf, ' + '.join(terms)

    def op_buffer_load_dword(self, ops, mods, count=1):
        # ops[0] is read before buffer_address() can refuse a short operand list, so an instruction with no
        # operands at all would reach it as an index error. Everything longer is left to the shared check,
        # which sees a modifier first: the lds form decodes to three operands and must still refuse on it.
        if not ops:
            raise Unsupported('buffer load with 0 operands, expected 4')
        dst = reg_range(ops[0])
        buf, address = self.buffer_address(ops, mods)
        self.emit('a = %s;' % address)
        for i in range(count):
            self.set_v(dst[1] + i, '%s.Load(a + %d)' % (buf, 4 * i))

    def op_buffer_load_dwordx2(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 2)

    def op_buffer_load_dwordx3(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 3)

    def op_buffer_load_dwordx4(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 4)

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
        if key[0][0] != 'table' or key[1] is None:
            raise Unsupported('buffer store through %s' % (key,))
        if key in self.pipe.modified:
            raise Unsupported('buffer store through a modified descriptor')
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

    def store_descriptor(self, tok, words, what, stride0_ok=False):
        """Resource key of a store's descriptor operand: all its words must still be the loaded words of one
        descriptor at a fixed table offset. A moved, overwritten or modified word (stride, num_records, format)
        would change what the store does. With stride0_ok (a raw byte-address store), dword 1 may carry a cleared
        stride field: that is the raw descriptor such a store already assumes."""
        rsrc = reg_range(tok)
        first = self.sym.get(rsrc[1]) if rsrc and rsrc[0] == 's' and rsrc[2] - rsrc[1] == words - 1 else None
        if not first or first[0] != 'desc' or first[1][0] != 'table' or first[2] is None or first[3] % 4:
            raise Unsupported('%s through %s holding %s' % (what, tok, first))
        key = desc_key(first)
        def word_ok(k):
            sym, want = self.sym.get(rsrc[1] + k), first[:3] + (first[3] + k,)
            return sym == want or (stride0_ok and k == 1 and sym == want + ('stride0',))
        if (not all(word_ok(k) for k in range(words)) or key in getattr(self.pipe, 'other_modified', ())
                or (key in self.pipe.modified and not stride0_ok)):
            raise Unsupported('%s through a modified descriptor' % what)
        return key

    def raw_table_uav_store(self, kind_label, ops, mods, data_count, *, allow_glc=False, allow_scalar_address=False):
        """Validate a fixed-table raw UAV descriptor, data range, byte address, and 4-byte alignment."""
        offset = 0
        glc = False
        for m in mods:
            if m.startswith('offset:') and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                offset += int(m[7:], 0)
            elif m == 'glc' and allow_glc:
                glc = True
            elif m != 'offen':
                raise Unsupported('%s modifier %s' % (kind_label, m))
        offen = 'offen' in mods
        if len(ops) != 4 or (not offen and not (allow_scalar_address and glc and ops[1] == 'off')):
            raise Unsupported('%s without the offen address form' % kind_label)
        data = reg_range(ops[0])
        if not data or data[0] != 'v' or data[2] - data[1] != data_count - 1 or data[2] > 255:
            raise Unsupported('%s data %s' % (kind_label, ops[0]))
        if offen:
            vaddr = reg_range(ops[1])
            if not vaddr or vaddr[0] != 'v' or vaddr[1] != vaddr[2] or vaddr[1] > 255:
                raise Unsupported('%s vaddr %s for offen' % (kind_label, ops[1]))
        key = self.store_descriptor(ops[2], 4, kind_label, stride0_ok=True)
        terms = [self.u(ops[1])] if offen else []
        if re.fullmatch(r'0x[0-9a-f]+|\d+', ops[3]):
            if not offen:
                raise Unsupported('%s off requires a scalar register soffset' % kind_label)
            offset += int(ops[3], 0)
        else:
            sreg = reg_range(ops[3])
            if not sreg or sreg[0] != 's' or sreg[1] != sreg[2] or sreg[1] > 105:
                raise Unsupported('buffer soffset %s' % ops[3])
            terms.append(self.S(sreg[1]))
        if offset % 4:
            raise Unsupported('%s at unaligned immediate offset %d' % (kind_label, offset))
        terms.append(str(offset))
        return data, self.pipe.uav(key), '(%s) & 0xfffffffcu' % ' + '.join(terms), glc

    def op_buffer_store_dword(self, ops, mods, count=1):
        # Compute-stage store in the offen form: per-lane byte VGPR plus the scalar offset and the 12-bit immediate,
        # through a descriptor read from a fixed root-table slot, to a RWByteAddressBuffer (manifest `uav` line).
        # Like the loads, this assumes a raw descriptor (no swizzle or stride). Out-of-range stores: assumes a future
        # runtime sizes the view to the descriptor's range so both sides drop them; nothing binds or checks that yet.
        if self.kind == 'vs':
            return self.op_buffer_store_vs(ops, mods, count)
        if self.kind != 'cs':
            raise Unsupported('buffer_store_dword' + ('x%d' % count if count > 1 else ''))  # the unknown-op refusal
        src, uav, address, _ = self.raw_table_uav_store('buffer store', ops, mods, count)
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
            'buffer atomic', ops, mods, data_count, allow_glc=True, allow_scalar_address=name == 'buffer_atomic_add')
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
            key = (('vbtab',), 16 * e['slot'])
            buf = self.pipe.resource(key, 'buf')
            stride = self.pipe.ud(('stride', self.pipe.resources[key]['index']))
            self.emit('a = (vid + ud%d) * ud%d + %d;' % (base, stride, e['offset']))
            self.load_format(FETCH_DXGI[e['format']][0], buf, e['vgpr'], e['count'])
        self.fetched = True

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
            for index, (_, _, name, asm) in enumerate(self.ins):
                _, args, _ = parse(asm)
                self.descriptor_writes_by_instruction.append(self.descriptor_scalar_writes(name, args))
                if index + 1 < len(self.ins) and name not in ('s_branch', 's_endpgm'):
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

    def formatted_descriptor_from_block(self, reg, first):
        """Require one untouched local load and a table pointer that no incoming path can rewrite."""
        start, end = self.descriptor_block_first, self.descriptor_instruction_index
        block = self.ins[start:end]
        words = set(range(reg[1], reg[2] + 1))
        for index in range(len(block) - 1, -1, -1):
            _, _, name, asm = block[index]
            _, args, _ = parse(asm)
            if not (self.descriptor_scalar_writes(name, args) & words):
                continue
            m = re.fullmatch(r's_load_dwordx(4|8|16)', name)
            load = reg_range(args[0])
            if not m or not load or load[1] + first[3] != reg[1] or int(m.group(1)) < first[3] + 4:
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
            return self.entry_table_pointers.get(pointer) == first[1] and self.formatted_pointer_reaches(pointer, read)
        return False

    def mubuf_format(self, ops, mods, count):
        """buffer_load_format_x/xy/xyz: FORMAT and DST_SEL come from word 3 of the descriptor, known at draw time.

        The runtime passes that word as an entry constant. Each supported format becomes one case of a switch
        on the FORMAT field that decodes the four components as the tbuffer path does; DST_SEL then picks, for
        each destination register, a component or the constant 0 or 1. The draw validates FORMAT and descriptor
        availability before submission, so the generated switch only sees the supported cases.
        """
        if self.kind not in ('vs', 'ps'):
            raise Unsupported('formatted buffer load requires a graphics draw')
        dst = reg_range(ops[0])
        buf, address = self.buffer_address(ops, mods)
        # FORMAT and DST_SEL are read from the table word, so word 3 must still be the loaded one.
        r = reg_range(ops[2])
        first = self.sym.get(r[1]) if r else None
        if not r or r[2] - r[1] != 3 or not first or first[0] != 'desc' or any(
                self.sym.get(r[1] + i) != ('desc', first[1], first[2], first[3] + i) for i in range(4)):
            raise Unsupported('formatted buffer load through a descriptor whose words were rewritten')
        if first[1][0] != 'table':
            raise Unsupported('formatted buffer load requires a hardware table descriptor')
        if self.past_entry and not self.formatted_descriptor_from_block(r, first):
            raise Unsupported('formatted buffer load requires descriptor words proven in the entry block or a local table load')
        word3 = 'ud%d' % self.pipe.ud(('descword3', self.pipe.resources[self.descriptor(ops[2])]['index']))
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
        dim = [m for m in mods if m.startswith('dim:')][0][4:]
        dmask = int([m for m in mods if m.startswith('dmask:')][0][6:], 0)
        kinds = {'SQ_RSRC_IMG_1D': 'tex1d', 'SQ_RSRC_IMG_2D': 'tex2d', 'SQ_RSRC_IMG_3D': 'tex3d', 'SQ_RSRC_IMG_2D_ARRAY': 'tex2da'}
        if dim not in kinds:
            raise Unsupported('image dimension %s' % dim)
        # A 2D array takes the slice as its third address component, where a volume takes z. Sample and
        # Load on a Texture2DArray take it in the same place, so only the declared type differs.
        one = dim == 'SQ_RSRC_IMG_1D'
        three = dim in ('SQ_RSRC_IMG_3D', 'SQ_RSRC_IMG_2D_ARRAY')
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)]
        if one and (how != 'sample' or len(cs) != 1 or not dmask or any(not m.startswith(('dim:', 'dmask:')) for m in mods)):
            raise Unsupported('unverified 1D image sample address or modifiers')
        lod = bias = None
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
            if len(cs) != (4 if three else 3):
                raise Unsupported('%s with %d address registers' % (name, len(cs)))
            if how == 'l':
                lod, cs = cs[-1], cs[:-1]
            else:
                # The bias is added to the LOD derived from the coordinate derivatives of the pixel quad: only a
                # pixel stage that still runs with its entry mask has them. Elsewhere the plain sample is lowered
                # to level 0, which would drop the bias, so the bias form is refused instead.
                if self.kind != 'ps' or not self.exec_known or len(self.blocks) > 0:
                    raise Unsupported('image_sample_b needs implicit derivatives: pixel stage, entry mask, no branches')
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
            if not self.exec_known or len(self.blocks) > 0:
                raise Unsupported('image_get_lod needs implicit derivatives: pixel stage, entry mask, no branches')
        desc_key = self.descriptor(ops[2])
        tex = self.pipe.resource(desc_key, kinds[dim])
        if how in ('load', 'load_mip'):
            # image_load_mip addresses x, y, [slice or z] and then the mip, so a 2D array is four registers
            # and Texture2DArray.Load takes int4(x, y, slice, mip) like a volume's int4(x, y, z, mip).
            axes = 3 if three else 2
            packed_2d = how == 'load' and dim == 'SQ_RSRC_IMG_2D' and 'a16' in mods
            packed_d16 = how == 'load' and 'd16' in mods
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
                expected_dst = (bin(dmask).count('1') + 1) // 2
                if not dst or dst[0] != 'v' or dst[2] - dst[1] + 1 != expected_dst or dst[2] > 255:
                    raise Unsupported('image_load d16 destination %s for dmask %#x' % (ops[0], dmask))
                res = self.pipe.resources[desc_key]
                if res['index'] in self.pipe.srv_formats:
                    fmt = self.pipe.srv_formats[res['index']]
                    if fmt not in D16_EXACT_SRV_FORMATS:
                        raise Unsupported('image_load d16 unsupported SRV format %d' % fmt)
            if packed_2d:
                packed = self.u(cs[0])
                address = ['%s & 0xffffu' % packed, '%s >> 16' % packed]
            else:
                address = [self.u(c) for c in cs[:axes]]
            mip = self.u(cs[axes]) if how == 'load_mip' else '0'
            call = '%s.Load(int%d(%s))' % (tex, axes + 1, ', '.join(address + [mip]))
        else:
            packed_2d = False
            if not one and how in ('sample', 'lz'):
                op_name = 'image_sample' if how == 'sample' else 'image_sample_lz'
                packed_2d = dim == 'SQ_RSRC_IMG_2D' and 'a16' in mods
                allowed = ('dlc', 'glc', 'slc') + (('a16',) if packed_2d else ())
                for m in mods:
                    if not (m.startswith(('dim:', 'dmask:')) or m in allowed):
                        raise Unsupported('%s modifier %s' % (op_name, m))
                if not dmask:
                    raise Unsupported('%s with an empty dmask' % op_name)
                expected = 1 if packed_2d else (3 if three else 2)
                if len(cs) != expected:
                    raise Unsupported('%s with %d address registers' % (op_name, len(cs)))
            heap = self.heap_sampler(ops[3])
            words = None if heap else self.const_words(ops[3])
            name = 'image_sample' + ('' if how == 'sample' else '_' + how)
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
            elif how in ('sample', 'l', 'lz') and (heap or (words and self.default_sampler_words(words))):
                # A heap sampler, or the fixed default that stands in for one, keeps its unbounded static LOD range: the
                # descriptor is read at run time, so its MIN_LOD, MAX_LOD and bias are not validated here (#2221).
                self.pipe.lod_unvalidated.add(name)
                smp = heap or self.pipe.sampler(tuple(words))
            elif how in ('sample', 'l', 'lz') and words:
                # A constant descriptor carries its LOD limits like image_sample_b, under the same bias and range
                # validation. A mip filter of 0 (base level only) keeps the ordinary sampler, as before: refusing it
                # would drop every pipeline with an all-zero inline sampler, and only the bias form needs it refused.
                self.lod_sampler_words(name, words, mip_filter_none_ok=True)
                smp = self.pipe.sampler(tuple(words), lod_limits=bool((words[2] >> 26) & 3))
            else:
                smp = heap or self.pipe.sampler(tuple(words) if words else self.descriptor(ops[3]))
            if packed_2d:
                packed = self.u(cs[0])
                uv = 'float2(f16tof32(%s), f16tof32(%s >> 16))' % (packed, packed)
            elif one:
                uv = self.f(cs[0])
            elif three:
                uv = 'float3(%s, %s, %s)' % (self.f(cs[0]), self.f(cs[1]), self.f(cs[2]))
            else:
                uv = 'float2(%s, %s)' % (self.f(cs[0]), self.f(cs[1]))
            if how == 'sample' and self.kind == 'cs':
                # A compute lane has no pixel quad to supply derivatives; level 0 would be a guess.
                raise Unsupported('image_sample with implicit LOD in a compute program')
            if how == 'l':
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
            elif how == 'lz' or not self.exec_known or self.kind != 'ps' or len(self.blocks) > 0:
                call = '%s.SampleLevel(%s, %s, 0)' % (tex, smp, uv)
            else:
                call = '%s.Sample(%s, %s)' % (tex, smp, uv)
        self.emit('t = %s;' % call)
        if how == 'load' and 'd16' in mods:
            res = self.pipe.resources[desc_key]
            res['d16_exact'] = True
            mode_arg = 'ud%d' % self.pipe.ud(('d16unorm16', res['index']))
            self.pipe.uses_pack16_rne = True
            halves = ['pack16_d16(t.%s, %s)' % (comp, mode_arg) for bit, comp in enumerate('xyzw') if dmask & (1 << bit)]
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
            pixel_aot.comparison_sampler_state(tuple(words), level_zero=how == 'c_lz')
        except ValueError as error:
            raise Unsupported('%s sampler: %s' % (base, error))
        tex = self.pipe.resource(self.descriptor(ops[2]), 'tex2da' if array else 'tex2d')
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
        static sampler (translated_sampler_bindings.h, line smpl). SQ_IMG_SAMP_WORD2 bits 13:0 LOD_BIAS (s6.8) and
        19:14 LOD_BIAS_SEC (s2.4) are not carried: the ISA scales them by T#.perf_modulation, which the D3D12 sampler
        has no field for, so a nonzero bias is refused. Mip filter 0 disables mipmapping (ISA: "0=none (disable
        mipmapping, use base-level)"), which a D3D12 sampler cannot express and which would make the bias count, so
        it is refused too. Field layout: Mesa 24.0.0 src/amd/registers/gfx10-rsrc.json, SQ_IMG_SAMP_WORD1/2."""
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

    def lod_sampler_words(self, name, words, mip_filter_none_ok=False):
        """The words of a constant sampler whose LOD limits are carried to the static sampler, or Unsupported.

        Shared by image_sample_b and by image_sample, _l and _lz (#2177). Only the bias form refuses a mip filter of
        0 (mip_filter_none_ok False): there the bias would count; the other forms leave that sampler as it was."""
        if words[2] & 0xfffff:
            raise Unsupported('%s with a sampler LOD bias (word 2 %#x)' % (name, words[2]))
        if (words[2] >> 26) & 3 == 0 and not mip_filter_none_ok:
            raise Unsupported('%s with mipmapping disabled (sampler mip filter 0)' % name)
        if words[1] & 0xfff > (words[1] >> 12) & 0xfff:
            raise Unsupported('%s with sampler MIN_LOD %#x above MAX_LOD %#x'
                              % (name, words[1] & 0xfff, (words[1] >> 12) & 0xfff))
        return words

    def op_image_sample_b(self, ops, mods):
        self.image(ops, mods, 'b')

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
            word = first[:3] + (first[3] + i,)
            # A write through a second load of this descriptor can leave these SGPR symbols intact.
            if self.sym.get(image[1] + i) != word or desc_key(word) in self.pipe.modified:
                raise Unsupported('%s through a modified image descriptor' % name)
        return self.descriptor(operand)

    def op_image_store(self, ops, mods):
        """Write one four-channel 2D texel from pixel FP32 or compute packed FP16 data.

        MIMG opcode 0x08 (LLVM AMDGPU MIMGInstructions.td, IMAGE_STORE via MIMG_Store): VDATA is the
        source, VADDR the integer texel address (x, y for dim 2D) and SRSRC the image descriptor; the
        channels selected by DMASK are written in order from consecutive VDATA registers. HLSL's typed UAV
        store writes every channel of the view, so only DMASK 0xf is exact here; a partial mask would need a
        read-modify-write and is refused. The data is stored as float4, so the bound view must be of a
        float or normalized format (the runtime side is not part of this lowering). The store runs under the
        lane's exec bit, like the other per-lane writes, and D3D drops UAV writes from helper lanes as the
        hardware drops them for lanes outside exec.
        """
        name = 'image_store'
        d16 = self.kind == 'cs' and mods.count('d16') == 1
        if self.kind == 'cs' and not d16:
            raise Unsupported('compute image_store requires d16')
        if not d16 and self.kind != 'ps':
            raise Unsupported('%s outside a pixel stage' % name)
        if len(ops) != 3:
            raise Unsupported('%s with %d operands' % (name, len(ops)))
        for m in mods:
            if d16 and m == 'd16':
                continue
            if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc')):
                raise Unsupported('%s modifier %s' % (name, m))
        dims = [m[4:] for m in mods if m.startswith('dim:')]
        masks = [m[6:] for m in mods if m.startswith('dmask:')]
        if len(dims) != 1 or len(masks) != 1:
            raise Unsupported('%s without exactly one dim and dmask' % name)
        if dims[0] != 'SQ_RSRC_IMG_2D':
            raise Unsupported('%s on %s: only 2D images are lowered' % (name, dims[0]))
        try:
            mask = int(masks[0], 0)
        except ValueError:
            raise Unsupported('%s with dmask %s: not an integer' % (name, masks[0])) from None
        if mask != 0xf:
            raise Unsupported('%s with dmask %s: a typed UAV store writes all four channels' % (name, masks[0]))
        data = reg_range(ops[0])
        count = 2 if d16 else 4
        if not data or data[0] != 'v' or data[2] - data[1] + 1 != count or data[2] > 255:
            raise Unsupported('%s data %s: %s VGPRs expected' % (name, ops[0], 'two' if d16 else 'four'))
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)] if r and r[0] == 'v' else []
        if len(cs) != 2:
            raise Unsupported('%s with %d address registers: x and y expected' % (name, len(cs)))
        words = [self.u('v%d' % n) for n in range(data[1], data[2] + 1)]
        if d16:
            if not all(re.fullmatch(r'v\d+', c) and int(c[1:]) <= 255 for c in cs):
                raise Unsupported('compute image_store requires full-width VGPR coordinates')
            key = self.fixed_table_image(ops[2], name)
            # RDNA 2 ISA §8.2.5: RG occupy VDATA, BA occupy VDATA+1, low half first.
            rgba = ['f16tof32(%s & 0xffffu), f16tof32(%s >> 16)' % (word, word) for word in words]
            value = 'float4(%s)' % ', '.join(rgba)
        else:
            key = self.descriptor(ops[2])
            value = 'asfloat(uint4(%s))' % ', '.join(words)
        image = self.pipe.uav(key, 'tex2d')
        body = '%s[uint2(%s, %s)] = %s;' % (image, self.u(cs[0]), self.u(cs[1]), value)
        self.emit(body if self.exec_known else 'if (exec & 1) ' + body)

    def gather_2d_setup(self, name, ops, mods, addr_count, addr_label, *, mip_preclamp):
        if self.kind not in ('ps', 'cs') or len(ops) != 4:
            raise Unsupported('%s requires a pixel or compute stage and four operands' % name)
        for modifier in mods:
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
        if not addr or addr[0] != 'v' or addr[2] - addr[1] + 1 != addr_count or addr[2] > 255:
            raise Unsupported('%s requires %s' % (name, addr_label))
        words = self.const_words(ops[3]) if sampler and sampler[0] == 's' and sampler[2] - sampler[1] == 3 else None
        if words is None:
            raise Unsupported('%s requires a constant sampler' % name)
        from pixel_aot import check_gather_sampler
        profiled = gather_2d_profile_words(words)
        if not mip_preclamp and (words[2] & (1 << 28)):
            profiled = (profiled[0], profiled[1], profiled[2] | (1 << 28), profiled[3])
        try:
            # _lz reads mip zero of the bound view; _l checks all other sampler fields against
            # the 2D gather profile and enforces words[1] MAX_LOD plus view mip bounds in HLSL.
            check_gather_sampler(profiled, 1, comparison=False)
        except ValueError as error:
            raise Unsupported('%s sampler: %s' % (name, error)) from error
        key = self.fixed_table_image(ops[2], name) if self.kind == 'cs' else self.descriptor(ops[2])
        tex = self.pipe.resource(key, 'tex2d')
        return dst, addr, words, channels[masks[0]], tex

    def op_image_gather4_lz(self, ops, mods):
        dst, addr, words, (channel_name, _), tex = self.gather_2d_setup(
            'image_gather4_lz', ops, mods, 2, 'two contiguous address registers', mip_preclamp=True)
        smp = self.pipe.sampler(tuple(words))
        u, v = (self.f('v%d' % i) for i in range(addr[1], addr[2] + 1))
        self.emit('t = %s.Gather%s(%s, float2(%s, %s));' % (tex, channel_name, smp, u, v))
        for i, component in enumerate('xyzw', dst[1]):
            self.set_v(i, 'U(t.%s)' % component)

    def op_image_gather4_l(self, ops, mods):
        """image_gather4_l on a 2D texture with explicit LOD and a validated constant sampler."""
        dst, addr, words, (_, channel_index), tex = self.gather_2d_setup(
            'image_gather4_l', ops, mods, 3, 'three contiguous address registers', mip_preclamp=False)
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
        tex = self.pipe.resource(self.descriptor(ops[2]), 'tex2d')
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
        dst = reg_range(ops[0])[1]
        self.set_v(dst, 'uint(int(F(%s)))' % self.u(ops[1]))

    def op_v_cvt_pk_u8_f32(self, ops, mods):
        # VOP3A opcode 350 of the RDNA 1.0 and RDNA 2 ISA guides:
        #   D = (S2 & ~(0xff << 8 * S1[1:0])) | ((flt32_to_uint8(S0) & 0xff) << 8 * S1[1:0])
        # The guides leave flt32_to_uint8 undefined (rounding, bounds, NaN and negative results are not
        # stated), so S0 must be +0.0 or an integer in [1, 255], where every reading gives the same byte.
        # A constant S0 outside that set is refused. A register is accepted only when proven: it was written
        # by v_cvt_f32_ubyte0..3 (no modifier, straight-line code, full exec) with no write since. Any other
        # register is refused unless the caller selects the pk_u8_s0_contract of the Pipeline, which states
        # that every S0 register holds +0.0 or an integer in 1..255. The clamp only keeps the HLSL
        # conversion defined.
        if mods:
            raise Unsupported('v_cvt_pk_u8_f32 with modifiers')
        if len(ops) != 4:
            raise Unsupported('v_cvt_pk_u8_f32 with %d operands, expected 4' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[1] != dst[2]:
            raise Unsupported('v_cvt_pk_u8_f32 destination %s' % ops[0])
        if not reg_range(ops[1]):
            src = self.u(ops[1])
            bits = int(src[:-1], 16)
            value = struct.unpack('<f', struct.pack('<I', bits))[0]
            if bits != 0 and not (value.is_integer() and 1 <= value <= 255):
                raise Unsupported('v_cvt_pk_u8_f32 with constant source %s outside +0.0 and 1..255' % ops[1])
        else:
            src = reg_range(ops[1])
            if src[0] != 'v' or src[1] != src[2]:
                raise Unsupported('v_cvt_pk_u8_f32 source %s' % ops[1])
            if src[1] not in self.byte_float_v:
                if not getattr(getattr(self, 'pipe', None), 'pk_u8_s0_contract', False):
                    raise Unsupported('v_cvt_pk_u8_f32 source %s is not proven to be +0.0 or an integer in 1..255'
                                      % ops[1])
                self.pipe.notes.add('v_cvt_pk_u8_f32 source registers assumed +0.0 or integers in 1..255 '
                                    '(pk_u8_s0_contract)')
        value, select, rest = self.u(ops[1]), self.u(ops[2]), self.u(ops[3])
        shift = '((%s & 3u) << 3u)' % select
        self.set_v(dst[1], '((%s & ~(0xffu << %s)) | ((uint(clamp(F(%s), 0.0, 255.0)) & 0xffu) << %s))'
                   % (rest, shift, value, shift))

    def op_s_ff1_i32_b32(self, ops, mods):
        if self.kind != 'cs' or mods or len(ops) != 2 or ops[1] != 'exec_lo':
            raise Unsupported('s_ff1_i32_b32 requires compute exec_lo')
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('s_ff1_i32_b32 requires one SGPR destination')
        if self.past_entry or not self.exec_known:
            raise Unsupported('s_ff1_i32_b32 requires the compute entry EXEC mask')
        # At dispatch entry even the last partial wave starts at lane zero.
        # FF1 reports its index and leaves SCC unchanged.
        self.set_s(dst[1], '0u', ('const', 0))

    def op_v_mbcnt_lo_u32_b32(self, ops, mods, high=False):
        name = 'v_mbcnt_%s_u32_b32' % ('hi' if high else 'lo')
        if self.kind != 'cs':
            raise Unsupported(name)
        if mods or len(ops) != 3:
            raise Unsupported('%s requires three unmodified operands' % name)
        x, y, z = self.pipe.group
        if x * y * z > 32:
            raise Unsupported('%s requires at most 32 threads per group' % name)
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
        # RDNA 2 V_MBCNT counts mask bits strictly below the lane. All threads of
        # this group fit in the low half of either guest wave size, including lane 31.
        if not high:
            lane = '(tid.x + %du * (tid.y + %du * tid.z))' % (x, y)
            result = '(countbits(0x%08xu & ((1u << %s) - 1u)) + %s)' % (mask, lane, result)
        self.set_v(dst[1], result)

    def op_v_mbcnt_hi_u32_b32(self, ops, mods):
        self.op_v_mbcnt_lo_u32_b32(ops, mods, high=True)

    def op_v_readfirstlane_b32(self, ops, mods):
        src, dst = reg_range(ops[1]), reg_range(ops[0])
        past_entry = getattr(self, 'past_entry', True)
        # Past the entry block of a branch-free program is only dead code after s_endpgm: nothing to recompute.
        if (mods or not self.tracks_uniformity() or not dst or dst[0] != 's' or dst[1] != dst[2] or
                not src or src[0] != 'v' or src[1] != src[2] or src[1] not in self.uniform_v or
                (src[1] not in getattr(self, 'uniform_expr', {}) or not getattr(self, 'has_branches', False)
                 if past_entry else not getattr(self, 'exec_known', False))):
            raise Unsupported('v_readfirstlane_b32 requires a proven uniform source')
        if past_entry:
            # The first active lane ran the definition: no join, EXEC write or overwrite of the VGPR and no write
            # of a scalar it read lies between them. A lane inactive at the definition kept an older VGPR value,
            # so the read recomputes the definition from the scalars instead of reading this lane's copy.
            self.set_s(dst[1], self.uniform_expr[src[1]])
        else:
            self.set_s(dst[1], self.u(ops[1]))

    # ---- SGPR spill to VGPR lanes (#2125, item E) --------------------------------------------
    # The compiler parks a scalar in one lane of a VGPR: `v_writelane_b32 vN, sM, k` writes sM into lane k
    # and `v_readlane_b32 sD, vN, k` reads it back (AMD RDNA 2 ISA reference, VOP3 lane ops; LLVM 15
    # AMDGPU spill lowering). Both ignore EXEC. With a constant k and a VGPR that no other instruction
    # touches, vN is only a table of scalar slots, so one scalar variable per (vN, k) is an exact model
    # and no other lane is ever read. Anything else keeps the one-lane model out of reach and is refused.
    SPILL_LANES = 32

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
        if not 0 <= lane < self.SPILL_LANES:
            raise Unsupported('%s lane %d outside 0..%d' % (name, lane, self.SPILL_LANES - 1))
        if getattr(self, 'kind', None) != 'ps':
            raise Unsupported('%s is lowered in pixel stages only' % name)
        if getattr(self, 'has_branches', True) or not self.tracks_uniformity():
            raise Unsupported('%s needs straight-line code to prove write-before-read' % name)
        self.require_spill_only(vreg[1], name)
        return vreg[1], sreg, lane

    def require_spill_only(self, n, name):
        """Refuse unless every instruction naming v<n> is a lane write or read of it."""
        for _, _, op, asm in getattr(self, 'ins', ()):
            if re.sub(r'_e(32|64)$', '', op) in ('v_writelane_b32', 'v_readlane_b32'):
                continue
            for lo, hi, single in re.findall(r'\bv\[(\d+):(\d+)\]|\bv(\d+)\b', asm):
                a, b = (int(lo), int(hi)) if lo else (int(single), int(single))
                if a <= n <= b:
                    raise Unsupported('%s: v%d is also used as a vector (%s)' % (name, n, op))

    def op_v_writelane_b32(self, ops, mods):
        n, src, lane = self.spill_operands('v_writelane_b32', ops, mods)
        if src is None:
            if ops[1] in ('vcc', 'vcc_lo', 'vcc_hi', 'exec', 'exec_lo', 'exec_hi', 'm0'):
                raise Unsupported('v_writelane_b32 source %s is a lane mask or m0' % ops[1])
            value, sym, uniform = self.u(ops[1]), None, True
            if re.fullmatch(r'0x[0-9a-f]{8}u', value):
                sym = ('const', int(value[2:-1], 16))
        else:
            if self.is_lane_mask(ops[1]) or (self.sym.get(src[1]) or ('',))[0] not in ('', 'const', 'ud', 'fixed_buffer_word'):
                raise Unsupported('v_writelane_b32 source s%d holds a lane mask or descriptor' % src[1])
            value = self.S(src[1])
            sym = self.sym.get(src[1]) if (self.sym.get(src[1]) or ('',))[0] == 'const' else None
            uniform = self.uniform_source(ops[1])
        slots = self.__dict__.setdefault('spill_slots', {})
        slots[(n, lane)] = (sym, uniform)
        self.emit('sp%d_%d = %s;' % (n, lane, value))

    def op_v_readlane_b32(self, ops, mods):
        if len(ops) != 3 or reg_range(ops[1]) is None or reg_range(ops[1])[0] != 'v':
            raise Unsupported('v_readlane_b32 requires a VGPR source')
        n, dst, lane = self.spill_operands('v_readlane_b32', ops, mods)
        if dst is None:
            raise Unsupported('v_readlane_b32 requires an SGPR destination')
        slot = getattr(self, 'spill_slots', {}).get((n, lane))
        if slot is None:
            raise Unsupported('v_readlane_b32 reads v%d lane %d before any v_writelane_b32 wrote it' % (n, lane))
        sym, uniform = slot
        self.set_s(dst[1], 'sp%d_%d' % (n, lane), sym=sym, uniform=uniform)

    def op_v_mov_b32(self, ops, mods):
        self.set_v(reg_range(ops[0])[1], self.u(ops[1]))

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
        mask = self.u(ops[3]) if len(ops) > 3 else 'vcc'
        self.set_v(reg_range(ops[0])[1],
                   '(%s & 1) ? %s : %s' % (mask, self.u_src(ops[2]), self.u_src(ops[1])))

    def op_v_interp_p1_f32(self, ops, mods):
        if hasattr(self, 'uniform_v'):
            self.uniform_v.discard(reg_range(ops[0])[1])
        self.note_attr(ops[2], 'linear')

    def op_v_interp_p2_f32(self, ops, mods):
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
        extra = [m for m in mods if m != 'high']
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
        vs = getattr(getattr(self, 'pipe', None), 'vs', None)
        if vs is not None and attr in vs.plain_params and attr not in vs.compr_params:
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
        upper 16 bits of the p2 destination: whether gfx10 keeps or clears them is not established by a source
        (LLVM 15 SIInstrInfo/VOP3Instructions.td model the result as f16 only; the encoding states no
        high-half behaviour), so only a plain v_cvt_f32_f16 reading the low half may consume it.
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
                if target <= off:
                    raise Unsupported('a backward branch in a program with v_interp_p1ll_f16 / v_interp_p2_f16')
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
        # P1 + J * P20, converted to binary16 into the low half of the destination (LLVM: the result is f16, the
        # high bit selects the attribute half only). The upper half is not modelled: interp_f16_scan refuses
        # every consumer of it, so the preserved half emitted here is never observed.
        attr, field, high = self.interp_f16('v_interp_p2_f16', ops, mods, 1, 4)
        self.interp_f16_pair('v_interp_p2_f16', ops)
        self.interp_f16_scan()
        if (self.interp_p1 or {}).get(int(ops[3][1:])) != (attr, field, high):
            raise Unsupported('v_interp_p2_f16 source %s is not the v_interp_p1ll_f16 result of %s' % (ops[3], ops[2]))
        d = reg_range(ops[0])[1]
        self.used_v.add(d)
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

    def mix(self, ops, mods, high):
        """v_fma_mix{lo,hi}_f16: fma of operands that are f32 or f16 halves; the f16 result goes to one half."""
        sel = [0, 0, 0]
        for m in mods:
            if m.startswith('op_sel_hi:'):
                sel = [int(x) for x in m[11:-1].split(',')]
            elif m.startswith('op_sel:'):
                raise Unsupported('op_sel on fma_mix')
        def operand(tok, is_half):
            if not is_half or tok in ('-0', '0'):
                return self.f(tok)
            return 'f16tof32(%s)' % self.u(tok, low16_only=True)
        expr = '%s * %s + %s' % tuple(operand(t, s) for t, s in zip(ops[1:4], sel))
        if 'clamp' in mods:
            expr = 'saturate(%s)' % expr
        d = reg_range(ops[0])[1]
        self.used_v.add(d)
        if high:
            self.set_v(d, '(v%d & 0xffffu) | (f32tof16(%s) << 16)' % (d, expr))
        else:
            keep_low16 = d in self.low16_only_v
            self.set_v(d, '(v%d & 0xffff0000u) | f32tof16(%s)' % (d, expr), low16_only=keep_low16)

    def op_v_fma_mixlo_f16(self, ops, mods):
        self.mix(ops, mods, False)

    def op_v_fma_mixhi_f16(self, ops, mods):
        self.mix(ops, mods, True)

    def i64_words(self, tok):
        """(low, high) uint expressions of a 64-bit integer source: a VGPR/SGPR pair or an inline int."""
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
        low, high = self.i64_words(ops[2])
        body = '{ uint2 t = %s(%s, %s, %s); v%d = t.x; v%d = t.y; }' % (
            helper, low, high, amount, dst[1], dst[2])
        if hasattr(self, 'uniform_v'):
            self.uniform_v.discard(dst[1])
            self.uniform_v.discard(dst[2])
        self.clear_low16(dst[1])
        self.clear_low16(dst[2])
        self.used_v.update((dst[1], dst[2]))
        self.emit(body if self.exec_known else 'if (exec & 1) ' + body)

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
        self.set_v(d[1], 'a')
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

    def op_v_lshlrev_b64(self, ops, mods):
        self.shift64(ops, mods, 'shl64')

    def op_v_lshrrev_b64(self, ops, mods):
        self.shift64(ops, mods, 'shr64u')

    def op_v_ashrrev_i64(self, ops, mods):
        self.shift64(ops, mods, 'shr64i')
    def op_v_cvt_pkrtz_f16_f32(self, ops, mods):
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
            if f16 and sdwa:
                raise Unsupported('SDWA form of %s' % name)
            if sdwa:
                self.check_compare_sdwa_destination(base, writes_exec, ops)
            if len(ops) == 3:
                dst, value, mask = ops
            elif writes_exec and len(ops) == 2:
                # The e32 and SDWA spellings write EXEC implicitly and print no destination.
                dst, (value, mask) = 'exec', ops
            else:
                raise Unsupported('%s with %d operands' % (base, len(ops)))
            if sdwa:
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

    def export(self, asm):
        # disasm_rdna2 prints a compressed EN that is not a whole channel pair (0x0, 0x3, 0xC or 0xF) as
        # an explicit en:0x.. flag. As in pixel_aot, such a mask is refused rather than guessed.
        m = re.fullmatch(r'exp (\w+) (.*?)((?: done| compr| vm| en:0x[0-9a-f]+)*)', asm)
        if not m:
            raise Unsupported('export form %r' % asm)
        target, vals, flags = m.group(1), [v.strip() for v in m.group(2).split(',')], m.group(3).split()
        if any(f.startswith('en:') for f in flags):
            raise Unsupported('compressed export enable mask %s does not select whole channel pairs'
                              % next(f for f in flags if f.startswith('en:'))[3:])
        if 'vm' in flags and not self.exec_known:
            self.emit('if (!(exec & 1)) discard;')
        if target == 'mrtz':
            if (self.kind != 'ps' or self.has_branches or self.blocks or not self.exec_known or 'depth' in self.exports
                    or 'compr' in flags or len(vals) != 4 or not re.fullmatch(r'v\d+', vals[0])
                    or vals[1:] != ['off', 'off', 'off']):
                raise Unsupported('MRTZ requires one uncompressed depth export in a straight-line pixel shader with entry EXEC')
            self.exports['depth'] = True
            self.emit('o.depth = %s;' % self.f(vals[0]))
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
            for v in (vals[0], vals[2]):
                r = self.u(v) if v != 'off' else None
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
    def _termination(self):
        """The statement a structured block uses when the dispatcher would set pc = 0xffff."""
        has_output = self.kind == 'vs' or any(k.startswith('mrt') for k in self.exports)
        return 'return o;' if has_output else 'return;'

    def _structured_body(self, index):
        """Natural loops with one latch and one forward exit, or None for the dispatcher.

        The supported shape is deliberately small: a loop occupies consecutive blocks, its header
        dominates every block in that range, one edge returns to the header, and one edge leaves the
        range for the block immediately after it. Nested loops may occupy a consecutive subrange. A
        loop that is entered anywhere except its header, has several exits, or overlaps another loop
        without being nested in it keeps the existing pc dispatcher.
        """
        n = len(self.blocks)
        edges, preds = [], [set() for _ in range(n)]

        def target(off):
            if off not in index:
                raise Unsupported('branch into the middle of a block (+0x%x)' % off)
            return index[off]

        for i, (_, _, term) in enumerate(self.blocks):
            nxt = i + 1 if i + 1 < n else -1
            if not term:
                out = [(nxt, None)]
            elif term[0] == 'end':
                out = [(-1, None)]
            elif term[0] == 'jump':
                out = [(target(term[1]), None)]
            else:
                out = [(target(term[2]), term[1]), (nxt, '!(%s)' % term[1])]
            if len({e[0] for e in out}) != len(out):
                return None
            edges.append(out)
            for to, _ in out:
                if to >= 0:
                    preds[to].add(i)

        reachable, stack = {0}, [0]
        while stack:
            for to, _ in edges[stack.pop()]:
                if to >= 0 and to not in reachable:
                    reachable.add(to)
                    stack.append(to)
        if len(reachable) != n:
            return None

        dom = [set(range(n)) for _ in range(n)]
        dom[0] = {0}
        changed = True
        while changed:
            changed = False
            for i in range(1, n):
                incoming = preds[i]
                if not incoming:
                    continue
                common = set.intersection(*(dom[p] for p in incoming))
                common.add(i)
                if common != dom[i]:
                    dom[i] = common
                    changed = True

        backs = {}
        for u, out in enumerate(edges):
            for v, cond in out:
                if v >= 0 and v in dom[u] and v <= u:
                    backs.setdefault(v, []).append((u, cond))
        if not backs:
            return None
        if any(len(entries) != 1 for entries in backs.values()):
            return None

        loops = []
        for header, ((latch, _),) in backs.items():
            nodes, work = {header}, [latch]
            while work:
                node = work.pop()
                if node == header or node in nodes:
                    continue
                nodes.add(node)
                work.extend(preds[node])
            first, last = min(nodes), max(nodes)
            if header != first or latch != last or nodes != set(range(first, last + 1)):
                return None
            # The dominance check above already excludes a side entry into a natural loop.
            # Keep this explicit test so a future relaxation of the range rule cannot admit one.
            for node in nodes:
                if node != header and any(p not in nodes for p in preds[node]):
                    return None
            exits = [(u, cond, to) for u in nodes for to, cond in edges[u] if to not in nodes]
            if len(exits) != 1:
                return None
            exit_src, exit_cond, exit_to = exits[0]
            if exit_to != last + 1 or exit_cond is None:
                return None
            loops.append(dict(header=header, latch=latch, nodes=nodes, exit=exit_to, exit_src=exit_src,
                              exit_cond=exit_cond, children=[]))

        # Natural loops are either disjoint or nested. A partial overlap is outside this shape.
        for i, a in enumerate(loops):
            for b in loops[i + 1:]:
                if a['nodes'] & b['nodes'] and not (a['nodes'] <= b['nodes'] or b['nodes'] <= a['nodes']):
                    return None
        for loop in sorted(loops, key=lambda item: len(item['nodes'])):
            parents = [p for p in loops if p is not loop and loop['nodes'] < p['nodes']]
            if parents:
                min(parents, key=lambda item: len(item['nodes']))['children'].append(loop)

        def validate(loop):
            header, latch, nodes, exit_src, exit_to = (loop['header'], loop['latch'], loop['nodes'],
                                                      loop['exit_src'], loop['exit'])
            covered = set()
            for child in loop['children']:
                if not (header < child['header'] and child['latch'] < latch
                        and child['exit'] == child['latch'] + 1):
                    return None
                if covered & child['nodes']:
                    return None
                covered |= child['nodes']
                if validate(child) is None:
                    return None
            chain, at = [], header
            while at <= latch:
                child = next((c for c in loop['children'] if c['header'] == at), None)
                if child:
                    chain.append(('loop', child))
                    at = child['latch'] + 1
                    continue
                if at in covered:
                    return None
                chain.append(('block', at))
                at += 1
            loop['chain'] = chain
            for kind, step in chain:
                if kind == 'loop':
                    continue
                out = edges[step]
                if step == header and header != latch:
                    internal = [e for e in out if e[0] == header + 1]
                    outside = [e for e in out if e[0] not in nodes and e[0] != exit_to]
                    if len(internal) != 1 or outside or len(out) not in (1, 2):
                        return None
                    if len(out) == 2 and exit_src != header:
                        return None
                elif step == latch:
                    back = [e for e in out if e[0] == header]
                    outside = [e for e in out if e[0] not in nodes and e[0] != exit_to]
                    if len(back) != 1 or outside:
                        return None
                    if len(out) == 2 and exit_src != latch:
                        return None
                    if len(out) == 1 and (exit_src == latch or out[0][1] is not None):
                        return None
                else:
                    if len(out) != 1 or out[0][0] != step + 1:
                        return None
            if exit_src not in (header, latch):
                return None
            if exit_src == header and header != latch:
                outside = [e for e in edges[header] if e[0] == exit_to]
                if len(outside) != 1 or len(edges[header]) != 2:
                    return None
            if exit_src == latch:
                if len(edges[latch]) != 2 or not any(e[0] == exit_to for e in edges[latch]):
                    return None
            if header == latch:
                if len(edges[header]) != 2 or not any(e[0] == header for e in edges[header]):
                    return None
            return loop

        top = [loop for loop in loops if not any(loop['nodes'] < p['nodes'] for p in loops)]
        top_by_header = {loop['header']: loop for loop in top}
        if any(validate(loop) is None for loop in top):
            return None
        # A loop may be nested only under a validated top-level loop.
        if any(validate(loop) is None for loop in loops):
            return None

        out = ['    uint guard = 0;']

        def block(i, indent):
            pad = '    ' * indent
            out.append('%sif (guard >= 65536u) %s' % (pad, self._termination()))
            out.append('%s++guard;' % pad)
            out.extend(pad + line for line in self.blocks[i][1])

        def loop_code(loop, indent):
            pad = '    ' * indent
            out.append('%sfor (;;) {' % pad)
            for kind, step in loop['chain']:
                if kind == 'loop':
                    loop_code(step, indent + 1)
                else:
                    block(step, indent + 1)
                    if step == loop['exit_src']:
                        out.append('%s    if (%s) break;' % (pad, loop['exit_cond']))
            out.append('%s}' % pad)

        at = 0
        while at < n:
            loop = top_by_header.get(at)
            if loop:
                loop_code(loop, 1)
                at = loop['latch'] + 1
                continue
            if any(at in loop['nodes'] for loop in loops):
                return None
            term = self.blocks[at][2]
            block(at, 1)
            if term and term[0] == 'end':
                if at != n - 1:
                    return None
                at += 1
            elif term and term[0] == 'jump':
                if target(term[1]) != at + 1:
                    return None
                at += 1
            elif term and term[0] == 'branch':
                return None
            else:
                at += 1
        return out

    def body(self):
        """Statements of the whole program as structured HLSL."""
        out = []
        offsets = [b[0] for b in self.blocks]
        index = {off: i for i, off in enumerate(offsets)}
        loops = any(t and t[0] in ('jump', 'branch') and t[-1] <= b[0] for b in self.blocks for t in [b[2]])
        if loops and getattr(self, 'structured_loops', True):
            structured = self._structured_body(index)
            if structured is not None:
                return structured
        if len(self.blocks) == 1 and not loops:
            return ['    ' + line for line in self.blocks[0][1]]

        def goto(off):
            if off not in index:
                raise Unsupported('branch into the middle of a block (+0x%x)' % off)
            return index[off]
        out.append('    uint pc = 0;')
        if loops:
            out.append('    [loop] for (uint guard = 0; guard < 65536 && pc != 0xffffu; ++guard) {')
        for i, (off, lines, term) in enumerate(self.blocks):
            out.append('    %sif (pc == %d) {' % ('else ' if loops and i else '', i))
            out += ['        ' + line for line in lines]
            nxt = i + 1 if i + 1 < len(self.blocks) else 0xffff
            if not term:
                out.append('        pc = %d;' % nxt)
            elif term[0] == 'end':
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
        lines.append('    uint vcc = 0, exec = 1, m0 = 0, a = 0, w = 0; bool scc = false; float4 t = 0;')
        if getattr(self, 'used_atomic', None):
            if 'uint' in self.used_atomic:
                lines.append('    uint ao = 0;')
            if 'int' in self.used_atomic:
                lines.append('    int ai = 0;')
        lines += ['    s%d = %s;' % kv for kv in sorted(getattr(self, 'materialized', {}).items())]
        return lines


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

# name -> (source operand count, expression); operands are floats
VEC_F = {
    'v_add_f32': (2, '{0} + {1}'), 'v_sub_f32': (2, '{0} - {1}'), 'v_subrev_f32': (2, '{1} - {0}'),
    'v_mul_f32': (2, '{0} * {1}'), 'v_mul_legacy_f32': (2, 'mul_legacy({0}, {1})'),
    'v_min_f32': (2, 'min({0}, {1})'), 'v_max_f32': (2, 'max({0}, {1})'),
    'v_fmac_f32': (2, '{0} * {1} + {2}'), 'v_mac_f32': (2, 'mad32({0}, {1}, {2}, false)'),
    'v_fma_f32': (3, '{0} * {1} + {2}'), 'v_mad_f32': (3, 'mad32({0}, {1}, {2}, false)'),
    'v_mad_legacy_f32': (3, 'mad32({0}, {1}, {2}, true)'),
    'v_fmamk_f32': (3, '{0} * {1} + {2}'), 'v_fmaak_f32': (3, '{0} * {1} + {2}'),
    'v_madmk_f32': (3, 'mad32({0}, {1}, {2}, false)'), 'v_madak_f32': (3, 'mad32({0}, {1}, {2}, false)'),
    'v_rcp_f32': (1, '1.0 / {0}'), 'v_rcp_iflag_f32': (1, '1.0 / {0}'), 'v_rsq_f32': (1, 'rsqrt({0})'), 'v_sqrt_f32': (1, 'sqrt({0})'),
    'v_exp_f32': (1, 'exp2({0})'), 'v_log_f32': (1, 'log2({0})'), 'v_floor_f32': (1, 'floor({0})'), 'v_ceil_f32': (1, 'ceil({0})'),
    'v_trunc_f32': (1, 'trunc({0})'), 'v_fract_f32': (1, 'frac({0})'), 'v_rndne_f32': (1, 'round({0})'),
    'v_sin_f32': (1, 'sin({0} * 6.283185307)'), 'v_cos_f32': (1, 'cos({0} * 6.283185307)'),
    'v_med3_f32': (3, 'med3({0}, {1}, {2})'), 'v_min3_f32': (3, 'min({0}, min({1}, {2}))'), 'v_max3_f32': (3, 'max({0}, max({1}, {2}))'),
}
# Conversions whose result is binary32. The expression is the float value, so an output modifier
# (clamp, mul:2, mul:4, div:2) can wrap it the way it wraps any other float result. The uint-result
# conversions stay in VEC_U.
VEC_U_TO_F = {
    'v_cvt_f32_u32': (1, 'float({0})'), 'v_cvt_f32_i32': (1, 'float(int({0}))'),
    'v_cvt_f32_ubyte0': (1, 'float({0} & 255)'), 'v_cvt_f32_ubyte1': (1, 'float(({0} >> 8) & 255)'),
    'v_cvt_f32_ubyte2': (1, 'float(({0} >> 16) & 255)'), 'v_cvt_f32_ubyte3': (1, 'float({0} >> 24)'),
    'v_cvt_f32_f16': (1, 'f16tof32({0})'),
}
# operands are uints; the result is a uint expression
VEC_U = {
    'v_add_nc_u32': (2, '{0} + {1}'), 'v_sub_nc_u32': (2, '{0} - {1}'), 'v_subrev_nc_u32': (2, '{1} - {0}'),
    'v_mul_lo_u32': (2, '{0} * {1}'), 'v_and_b32': (2, '{0} & {1}'), 'v_or_b32': (2, '{0} | {1}'), 'v_xor_b32': (2, '{0} ^ {1}'),
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
    'v_lshl_add_u32': (3, '({0} << ({1} & 31)) + {2}'), 'v_add_lshl_u32': (3, '({0} + {1}) << ({2} & 31)'), 'v_add3_u32': (3, '{0} + {1} + {2}'),
    'v_or3_b32': (3, '{0} | {1} | {2}'), 'v_xor3_b32': (3, '{0} ^ {1} ^ {2}'), 'v_and_or_b32': (3, '({0} & {1}) | {2}'), 'v_lshl_or_b32': (3, '({0} << ({1} & 31)) | {2}'),
    'v_xad_u32': (3, '({0} ^ {1}) + {2}'), 'v_bfi_b32': (3, '({0} & {1}) | (~{0} & {2})'),
    'v_mul_u32_u24': (2, '({0} & 0xffffffu) * ({1} & 0xffffffu)'),
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
}
# scalar ops: (expression over uint sources, SCC rule with $ for the result or None)
# Scalar ops that write SCC on the hardware (carry, overflow, or which source was selected) but have
# no SCC template here: the emitted scc keeps its previous value, so the uniformity proof stops
# trusting it.
SCC_UNMODELLED = frozenset(('s_add_i32', 's_add_u32', 's_sub_i32', 's_sub_u32',
                            's_min_i32', 's_min_u32', 's_max_i32', 's_max_u32'))

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
    # s_mul_hi_u32: high 32 bits of the unsigned 64-bit product; SCC unchanged.
    's_mul_hi_u32': ('mul_hi_u32({0}, {1})', None),
    # s_sext_i32_iN: sign-extend the low N bits of S0; SCC unchanged.
    's_sext_i32_i8': ('uint(int({0} << 24) >> 24)', None),
    's_sext_i32_i16': ('uint(int({0} << 16) >> 16)', None),
    's_pack_hh_b32_b16': ('({0} >> 16) | ({1} & 0xffff0000u)', None),
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
for _name, _op in (('eq', '=='), ('lg', '!='), ('gt', '>'), ('ge', '>='), ('lt', '<'), ('le', '<=')):
    for _typ, _cast in (('i32', 'int({0}) %s int({1})'), ('u32', '{0} %s {1}')):
        def _make(op=_op, cast=_cast, name='s_cmp_%s_%s' % (_name, _typ)):
            def handler(self, ops, mods):
                # A scalar compare reads exactly two sources; refusing any other count keeps a malformed
                # instruction a refusal rather than an index error in the batch translator.
                if len(ops) != 2:
                    raise Unsupported('%s with %d operands, expected 2' % (name, len(ops)))
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
# DXGI_FORMAT values independently verified exact for D16 image_load, matching d16_srv_mode()
# in src/startup_stubs/d16_image_format.h (#2023): 32-bit float and 8-bit UNORM views (exact via
# pack16_rne) plus 16-bit UNORM views (exact via pack16_unorm16).
D16_FLOAT_EXACT_SRV_FORMATS = frozenset((2, 6, 16, 21, 28, 41, 49, 61, 65, 87, 88))
D16_UNORM16_SRV_FORMATS = frozenset((11, 35, 56))
D16_EXACT_SRV_FORMATS = D16_FLOAT_EXACT_SRV_FORMATS | D16_UNORM16_SRV_FORMATS
# Uncompressed exports keep their 32-bit shader values. RGBA8 integer RTVs clamp each
# channel to the destination range in the output merger.
VERIFIED_UINT_RTVS = frozenset((3, 17, 30, 42))
VERIFIED_SINT_RTVS = frozenset((4, 18, 32, 43))


class Pipeline:
    uses_pack16_rne = False
    uses_gather_2d_lod = False
    srv_formats = ()
    uavs = None  # key -> dict(index, kind): images a pixel stage stores to (u registers); None until the first store

    def __init__(self, vs_code, ps_code, root_blob, sampler_heap=None, rtv_formats=(), ps_hw_inputs=None, vs_fetch=None,
                 refusals=None, decoded=None, ps_attr_map=None, srv_formats=(), pk_u8_s0_contract=False,
                 structured_loops=True):
        # pk_u8_s0_contract: the caller guarantees that every v_cvt_pk_u8_f32 S0 register not proven in domain
        # holds +0.0 or an integer in 1..255 (fixtures only; ordinary translation never selects it).
        self.pk_u8_s0_contract = pk_u8_s0_contract
        self.root = parse_root(root_blob)
        self.ps_attr_map = ps_attr_map  # stream_interface(): pixel attribute -> (vertex parameter, component mask); None pairs by number
        self.vs_fetch = vs_fetch  # stream_fetch(): the loads the vertex stage's entry call stands for
        self.packed_position_vgpr = None
        # First VGPR of the PERSP_CENTER I/J entry pair (SPI_PS_INPUT_ADDR bit 1, after the PERSP_SAMPLE pair when that
        # one is enabled). Without hardware inputs the model assumes the center pair alone: v0, v1.
        self.persp_center_vgpr = 0
        if ps_hw_inputs is not None:
            ena, addr = ps_hw_inputs
            if ena != addr or not 0 <= ena <= 0xffff:
                raise Unsupported('unverified pixel input enable/address combination')
            self.persp_center_vgpr = (2 if addr & 1 else 0) if addr & 2 else None
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
        self.resources = {}   # key -> dict(index, kind)
        self.samplers = {}    # key -> index
        self.uds = []         # entry-constant sources
        self.notes = set()    # shortcuts taken (written to the manifest as comments)
        self.lod_unvalidated = set()  # image_sample forms read through a heap sampler (LOD fields unchecked, #2221)
        self.modified = set()
        self.rect = self.ud(('rect',))
        self.structured_loops = structured_loops
        if refusals is None:
            self.vs = Stage(self, 'vs', vs_code, structured_loops=structured_loops)
            self.ps = Stage(self, 'ps', ps_code, structured_loops=structured_loops)
        else:
            decoded = {} if decoded is None else decoded
            self.vs = Stage(self, 'vs', vs_code, refusals=refusals, decoded=decoded.get('vs'),
                            structured_loops=structured_loops)
            self.ps = Stage(self, 'ps', ps_code, refusals=refusals, decoded=decoded.get('ps'),
                            structured_loops=structured_loops)
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
        if key in self.stored_images():
            raise Unsupported('resource %s is both read and stored to' % (key,))
        r = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        if r['kind'] != kind:
            raise Unsupported('resource %s used as %s and %s' % (key, r['kind'], kind))
        return 'r%d' % r['index']

    def stored_images(self):
        return self.uavs or {}

    def uav(self, key, kind):
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
        # A comparison sampler, or one that carries the descriptor's LOD limits (image_sample_b), is its own binding
        # even when its words equal an ordinary sampler's.
        key = ('cmp',) + tuple(key) if comparison else ('lod',) + tuple(key) if lod_limits else key
        return 'q%d' % self.samplers.setdefault(key, len(self.samplers))

    def common(self):
        out = ['#define F(x) asfloat(x)', '#define U(x) asuint(x)',
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
               'float med3(float a, float b, float c) { return max(min(a, b), min(max(a, b), c)); }',
               # High half of an unsigned 32x32 product, by halfword products so it fits 32-bit HLSL.
               'uint mul_hi_u32(uint a, uint b) { uint la = a & 0xffffu, lb = b & 0xffffu;'
               ' uint middle = (a >> 16) * lb + ((la * lb) >> 16);'
               ' return (a >> 16) * (b >> 16) + (middle >> 16) + (((middle & 0xffffu) + la * (b >> 16)) >> 16); }',
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
               'uint2 bfe64(uint lo, uint hi, uint s1, uint sg) { uint w = min((s1 >> 16u) & 127u, 64u - (s1 & 63u)); uint2 e = shr64u(lo, hi, s1);'
               ' e = uint2(e.x & (w >= 32u ? 0xffffffffu : ((1u << (w & 31u)) - 1u)),'
               ' e.y & (w >= 64u ? 0xffffffffu : (w > 32u ? ((1u << ((w - 32u) & 31u)) - 1u) : 0u)));'
               ' uint sh = (64u - w) & 63u; uint2 t = shl64(e.x, e.y, sh);'
               ' return sg != 0u ? shr64i(t.x, t.y, sh) : e; }',
               'uint bfe32(uint x, uint s1, uint sg) { uint w = min((s1 >> 16u) & 127u, 32u - (s1 & 31u));'
               ' uint e = (x >> (s1 & 31u)) & (w >= 32u ? 0xffffffffu : ((1u << (w & 31u)) - 1u));'
               ' uint sh = (32u - w) & 31u; return sg != 0u ? asuint(asint(e << sh) >> sh) : e; }',
               'uint ldu(ByteAddressBuffer b, uint a) { uint k = (a & 3) * 8, lo = b.Load(a & ~3u); if (k == 0) return lo; '
               'return (lo >> k) | (b.Load((a & ~3u) + 4) << (32 - k)); }',
               'cbuffer entry : register(b0) { %s };' % ' '.join('uint ud%d;' % i for i in range(len(self.uds)))]
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            out.append('%s r%d : register(t%d);' % ({'buf': 'ByteAddressBuffer', 'tex1d': 'Texture1D<float4>', 'tex2d': 'Texture2D<float4>', 'tex2da': 'Texture2DArray<float4>', 'tex3d': 'Texture3D<float4>'}[r['kind']], r['index'], r['index']))
        for key, r in sorted(self.stored_images().items(), key=lambda kv: kv[1]['index']):
            out.append('%s %s%d : register(u%d);' % ({'tex2d': 'RWTexture2D<float4>', 'raw': 'RWByteAddressBuffer'}[r['kind']], 'u' if r['kind'] == 'raw' else 'w', r['index'], r['index']))
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
        out = self.common() + ['struct VSOut {'] + self.interface(False) + ['};', 'VSOut body(uint vid) {', '    VSOut o = (VSOut)0;']
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
        return 'ps_6_0' if self.ps.needs_sm6 else 'ps_5_1'

    def hlsl_ps(self):
        s = self.ps
        targets = sorted(int(k[3:]) for k in s.exports if k.startswith('mrt'))
        depth = 'depth' in s.exports
        out = self.common() + ['struct PSIn {'] + self.interface(True) + ['};']
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
        # A manifest that names a writable view is version 2, so a runtime that reads only version 1 lines can tell.
        lines = ['x2w-manifest %d' % (2 if self.stored_images() else 1), 'ud %d' % len(self.uds)]
        for i, source in enumerate(self.uds):
            lines.append('udsrc %d %s' % (i, ' '.join(str(x) for x in source)))
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            source, off = key
            where = {'table': 'table %d %d' % (source[1] if len(source) > 1 else 0, off or 0), 'vbtab': 'vb %d' % ((off or 0) // 16),
                     'root': 'root %d' % (source[1] if len(source) > 1 else 0)}.get(source[0])
            if where is None:
                raise Unsupported('resource through %s' % (source,))
            lines.append('srv %d %s %s' % (r['index'], r['kind'], where))
            if r.get('d16_exact'):
                lines.append('d16srv %d exact' % r['index'])
        for key, r in sorted(self.stored_images().items(), key=lambda kv: kv[1]['index']):
            source, off = key
            if source[0] != 'table':
                raise Unsupported('stored image through %s' % (source,))
            lines.append('uav %d %s table %d %d' % (r['index'], r['kind'], source[1] if len(source) > 1 else 0, off or 0))
        for key, i in self.samplers.items():
            prefixed = key[0] in ('cmp', 'lod')
            lines.append('%s %d %s' % ({'cmp': 'smpc', 'lod': 'smpl'}.get(key[0], 'smp'), i,
                                       ' '.join('0x%x' % w for w in key[prefixed:])))
        if self.uses_sampler_heap:
            lines.append('smpheap %d %d fallback %d' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE, SAMPLER_HEAP_FALLBACK))
        lines.append('params %d' % (max(self.params) + 1 if self.params else 0))
        lines.append('targets %s' % ' '.join(k[3:] for k in sorted(self.ps.exports) if k.startswith('mrt')))
        # 'wmask <target> <mask>': the export leaves the channels outside the mask unwritten, so the
        # runtime keeps them out of that target's RenderTargetWriteMask. Full masks are not written.
        for name, mask in sorted(self.ps.write_masks.items(), key=lambda kv: int(kv[0][3:])):
            if mask != 0xF:
                lines.append('wmask %d 0x%x' % (int(name[3:]), mask))
        return '\n'.join(lines) + '\n'


class ComputePipeline(Pipeline):
    """One compute program with the root and resource model of Pipeline. COMPUTE_PGM_RSRC2 and the thread group
    size are pipeline register state the caller supplies; COMPUTE_PGM_RSRC2 is decoded as compute_store_aot does
    (#1608). Root parameters from s4 are an inference (see Stage.setup_entry)."""

    def __init__(self, cs_code, root_blob, rsrc2, group, sampler_heap=None, refusals=None, decoded=None):
        from compute_store_aot import checked_group, entry_registers
        try:
            self.entry = entry_registers(rsrc2)  # (user SGPRs, {axis: group-ID SGPR}, thread-ID VGPRs)
            self.group = checked_group(group)
        except ValueError as reason:
            raise Unsupported(str(reason)) from None
        self.root = parse_root(root_blob)
        self.vs_fetch, self.packed_position_vgpr, self.rtv_formats = None, None, ()
        self.sampler_heap = os.environ.get('X2W_SAMPLER_HEAP') == '1' if sampler_heap is None else sampler_heap
        self.uses_sampler_heap = False
        self.uses_gather_2d_lod = False
        self.resources, self.uavs, self.samplers, self.uds = {}, {}, {}, []
        self.notes, self.modified, self.other_modified = set(), set(), set()  # other: anything but a stride clear
        self.lod_unvalidated = set()
        # LLVM AMDGPUUsage, COMPUTE_PGM_RSRC2.GRANULATED_LDS_SIZE on gfx10: 128 dwords per unit.
        self.lds_words = ((rsrc2 >> 15) & 0x1ff) * 128
        self.cs = Stage(self, 'cs', cs_code, refusals=refusals, decoded=decoded)

    def resource(self, key, kind):
        if key in self.uavs:
            raise Unsupported('descriptor %s used for a load and a store' % (key,))
        return super().resource(key, kind)

    def uav(self, key, kind='raw'):
        if key in self.resources:
            raise Unsupported('descriptor %s used for a load and a store' % (key,))
        return super().uav(key, kind)

    def stored_images(self):
        return {key: r for key, r in self.uavs.items() if r['kind'] != 'raw'}

    def common(self):
        out = [line for line in super().common() if self.uds or not line.startswith('cbuffer entry')]
        for r in self.uavs.values():
            if r['kind'] == 'raw':
                out.append('RWByteAddressBuffer u{0} : register(u{0});'.format(r['index']))
        return out

    def hlsl_cs(self):
        s = self.cs
        if s.exports:
            raise Unsupported('export in a compute program')
        body = s.body()
        shared = ['groupshared uint lds[%d];' % self.lds_words] if getattr(s, 'uses_lds', False) else []
        return '\n'.join(self.common() + shared + ['[numthreads(%d, %d, %d)]' % self.group,
                                          'void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {']
                         + s.declarations() + ['    ' + line for line in s.entry] + body + ['}']) + '\n'

    def manifest(self):
        # Version 2: `stage cs`, `group` and `uav` lines, which a graphics-only reader must not take for a draw.
        lines = ['x2w-manifest 2', 'stage cs', 'group %d %d %d' % self.group, 'ud %d' % len(self.uds)]
        lines += ['udsrc %d %s' % (i, ' '.join(str(x) for x in source)) for i, source in enumerate(self.uds)]
        for (source, off), r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            if source[0] == 'table':
                where = 'table %d %d' % (source[1], off or 0)
            elif source[0] == 'root':
                where = 'root %d' % source[1]
            else:
                raise Unsupported('resource through %s' % (source,))
            lines.append('srv %d %s %s' % (r['index'], r['kind'], where))
        for (source, off), r in self.uavs.items():
            where = 'table %d %d' % (source[1], off)
            if r['kind'] == 'raw':
                lines.append('uav %d %s' % (r['index'], where))
            else:
                lines.append('uav %d tex2d float %s channels 4' % (r['index'], where))
        for key, i in self.samplers.items():
            comparison = key[0] == 'cmp'
            lines.append('%s %d %s' % ('smpc' if comparison else 'smp', i, ' '.join('0x%x' % w for w in key[comparison:])))
        if self.uses_sampler_heap:
            lines.append('smpheap %d %d fallback %d' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE, SAMPLER_HEAP_FALLBACK))
        return '\n'.join(lines) + '\n'


def fnv(data, h=1469598103934665603):
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def pipeline_key(vs_code, ps_code, root_blob, layout=b''):
    """Hash the startup translation revision and the three blobs in runtime order, then a stream's input layout."""
    key = fnv(root_blob, fnv(ps_code, fnv(vs_code, fnv(b'x2w-startup-shader-14'))))
    return fnv(layout, key) if layout else key


def compute_key(cs_code, data, root_blob, desc, state_a=b'', state_b=b''):
    """The runtime key of a device slot 67 compute request (req_<key>.c67, .data, .root, .desc, .sa, .sb): the
    graphics revision followed by "cs", then each part preceded by its length as a little-endian u64: the code,
    the words 7-8 blob, the root signature, the 12 description words with their pointers zeroed and the two
    state blocks."""
    key = fnv(b'x2w-startup-shader-14cs')
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
    """Writes <key>.cs.hlsl, then <key>.cs.cso (DXC cs_6_0) and <key>.txt last, or <key>.fail with the reason.
    X2W_FXC set to the empty string publishes the manifest alone, as for graphics."""
    name = os.path.join(out_dir, '%016x' % key)
    with suppress(FileNotFoundError):
        os.remove(name + '.txt')  # the result is unpublished until validation and compilation finish
    try:
        pipe = load_compute_request(out_dir, key)
        hlsl, manifest = pipe.hlsl_cs(), pipe.manifest()
    except Unsupported as reason:
        reason = fail_reason(reason)
        open(name + '.fail', 'w').write(reason + '\n')
        return 'unsupported: %s' % reason
    log_unvalidated_lod(key, pipe)
    open(name + '.cs.hlsl', 'w').write(hlsl)
    if os.environ.get('X2W_FXC') != '':
        compiler = find_dxc()
        if compiler is None:
            message = 'dxc not found (X2W_DXC=%s): a compute stage needs shader model 6.0' % os.environ.get('X2W_DXC', '<unset>')
            open(name + '.fail', 'w').write(message + '\n')
            return message
        import subprocess
        with tempfile.TemporaryDirectory(prefix='.dxc-', dir=out_dir) as staging:
            bytecode = os.path.join(staging, 'cs.cso')
            r = subprocess.run([compiler, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', bytecode, name + '.cs.hlsl'],
                               capture_output=True, text=True)
            if r.returncode or not has_bytecode(bytecode):
                # DXC names the source path; keep only the diagnostic after it.
                lines = [re.sub(r'^.*?\.cs\.hlsl(:\d+:\d+)?:\s*', '', line) for line in r.stderr.strip().splitlines()]
                detail = lines[0] if lines else ('?' if r.returncode else 'no bytecode produced')
                open(name + '.fail', 'w').write('dxc cs: %s\n' % detail[:500])
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


def translate_to(out_dir, key, vs_code, ps_code, root_blob, check=True, rtv_formats=(), rtv_request='', vs_fetch=None,
                 ps_attr_map=None):
    """Writes <key>.vs.hlsl, <key>.ps.hlsl and <key>.txt (last), or <key>.fail with the reason.

    With check, the pair is compiled to .vs.cso and .ps.cso first and a missing fxc is a failure.
    check=False, or X2W_FXC set to the empty string, publishes the manifest without bytecode."""
    name = os.path.join(out_dir, '%016x' % key)
    try:
        pipe = Pipeline(vs_code, ps_code, root_blob, rtv_formats=rtv_formats, vs_fetch=vs_fetch, ps_attr_map=ps_attr_map)
        vs, ps, manifest = pipe.hlsl_vs(), pipe.hlsl_ps(), pipe.manifest()
        if rtv_request:
            manifest += rtv_request + '\n'
    except Unsupported as reason:
        reason = fail_reason(reason)
        open(name + '.fail', 'w').write(reason + '\n')
        return 'unsupported: %s' % reason
    log_unvalidated_lod(key, pipe)
    open(name + '.vs.hlsl', 'w').write(vs)
    open(name + '.ps.hlsl', 'w').write(ps)
    fxc = find_fxc() if check else ''
    if fxc is None:
        message = 'fxc.exe not found (X2W_FXC=%s): set X2W_FXC to its path or install the Windows SDK' % os.environ.get('X2W_FXC', '<unset>')
        open(name + '.fail', 'w').write(message + '\n')
        return message
    manifest_path = name + '.txt'
    write_manifest = True
    # A quad intrinsic exists from shader model 6.0 only, and a pipeline does not mix DXBC and DXIL stages.
    sm6 = any(getattr(getattr(pipe, stage, None), 'needs_sm6', False) for stage in ('vs', 'ps'))
    if fxc and sm6:
        compiler = find_dxc()
        if compiler is None:
            message = 'dxc not found (X2W_DXC=%s): a DPP quad read needs shader model 6.0' % os.environ.get('X2W_DXC', '<unset>')
            open(name + '.fail', 'w').write(message + '\n')
            return message
        profiles = (('vs', 'vs_6_0'), ('ps', 'ps_6_0'))
    else:
        compiler, profiles = fxc, (('vs', 'vs_5_1'), ('ps', 'ps_5_1'))
    if fxc:
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
                    open(name + '.fail', 'w').write('fxc %s: %s\n' % (stage, detail[-2000:]))
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
    if write_manifest:
        temporary = '%s.%d.tmp' % (manifest_path, os.getpid())
        try:
            with open(temporary, 'w') as result:
                result.write(manifest)
            os.replace(temporary, manifest_path)
        finally:
            with suppress(FileNotFoundError):
                os.remove(temporary)
    if fxc:
        with suppress(FileNotFoundError):
            os.remove(name + '.fail')
    return 'ok'


def classify_decoded(ins):
    text = ' '.join(x[3] for x in ins if x[2] == 'exp')
    return 'vs' if 'pos' in text else 'ps' if 'mrt' in text or not text else 'other'  # no export: a depth-only pixel shader


def classify(blob):
    return classify_decoded(decode(strip_header(blob)))


def load_request(out_dir, key, refusals=None, pair_info=None):
    """Read a base request or an RTV variant and classify the source stages."""
    decoded = {}

    def classify_stage(blob):
        if refusals is None:
            return classify(blob)
        ins = decode(strip_header(blob), refusals)
        kind = classify_decoded(ins)
        decoded[kind] = ins
        return kind

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
        name = os.path.join(out_dir, 'req_%016x' % base)
    a, b, root = (open(name + ext, 'rb').read() for ext in ('.a', '.b', '.root'))
    if pair_info is not None and not request:
        import hashlib
        pair_info.append((hashlib.sha256(struct.pack('<Q', len(a)) + a + b).digest(), root))
    # A stream with an input layout publishes it as .ia (before .a) and hashes it into the key.
    layout = open(name + '.ia', 'rb').read() if os.path.isfile(name + '.ia') else b''
    if request and pipeline_key(a, b, root, layout) != base:
        raise Unsupported('RTV variant source key mismatch')
    fetch = attr_map = None
    if a.startswith(b'DXBC') or b.startswith(b'DXBC'):
        # Slot 86 records complete containers in canonical (VS, PS) order.
        # Keep the container bytes in the cache key; translate only the bounded native code.
        import zlib
        from pixel_aot import container_code
        from resource_inventory import parts
        try:
            container, ps_container = a, b
            a, b = (container_code(parts(blob)[b'DXIL'], stage=stage)
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


def mesh57_key(ms, ps, root):
    value = fnv(b'x2w-startup-mesh-57-1')
    for part in (ms, ps, root):
        value = fnv(part, fnv(struct.pack('<Q', len(part)), value))
    return value


def inspect_mesh57_request(out_dir, key):
    """Read a fixed-descriptor mesh request without falling back to a vertex translation."""
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
    if ps:
        pixel = decode(strip_header(ps))
        if not pixel or pixel[-1][2] != 's_endpgm' or classify_decoded(pixel) != 'ps':
            raise Unsupported('slot 57 pixel stage has an incompatible export or termination')
    for offset, _, op, _ in mesh:
        if op.startswith('ds_'):
            raise Unsupported('slot 57 mesh workgroup memory: %s at +0x%x' % (op, offset))
    raise Unsupported('slot 57 mesh code emission is unavailable')


def translate_request(out_dir, key):
    """Translate a base shader request, an RTV variant referencing its source blobs, or a slot 67 compute request."""
    if is_compute_request(out_dir, key):
        return translate_compute(out_dir, key)
    if os.path.isfile(os.path.join(out_dir, 'req_%016x.m57' % key)):
        try:
            inspect_mesh57_request(out_dir, key)
        except (Unsupported, OSError, UnicodeError, struct.error) as reason:
            open(os.path.join(out_dir, '%016x.fail' % key), 'w').write(str(reason) + '\n')
            return 'unsupported: %s' % reason
    try:
        kinds, root, formats, request, fetch, _, attr_map = load_request(out_dir, key)
    except (Unsupported, OSError, UnicodeError) as reason:
        open(os.path.join(out_dir, '%016x.fail' % key), 'w').write(str(reason) + '\n')
        return 'unsupported: %s' % reason
    if set(kinds) != {'vs', 'ps'}:
        open(os.path.join(out_dir, '%016x.fail' % key), 'w').write('stages %s\n' % sorted(kinds))
        return 'stages %s' % sorted(kinds)
    return translate_to(out_dir, key, kinds['vs'], kinds['ps'], root, rtv_formats=formats, rtv_request=request, vs_fetch=fetch,
                        ps_attr_map=attr_map)


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
        kinds, root, formats, _, fetch, decoded, attr_map = load_request(out_dir, key, refusals, pair_info)
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
                        refusals=refusals, decoded=decoded, ps_attr_map=attr_map)
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
    if sys.argv[1] == '--watch':
        # --watch <dir> [seconds]: translate requests as a running title writes them, until stopped
        interval = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
        # The first pass retries earlier failures once (the runtime never rewrites a request, so a
        # translator fix would otherwise not reach them); later passes skip failures already written.
        skip_failed = False
        while True:
            if watch_pass(out_dir, skip_failed):
                skip_failed = True
            time.sleep(interval)
    translate_pending(out_dir)




def watch_pass(out_dir, skip_failed):
    """One --watch pass. A worker killed from outside (out of memory) breaks the pool; the pass is
    reported and its requests are retried on the next pass instead of ending the watcher."""
    from concurrent.futures.process import BrokenProcessPool
    try:
        translate_pending(out_dir, skip_failed=skip_failed)
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
        with open(os.path.join(out_dir, '%016x.fail' % key), 'w') as fail:
            fail.write(reason + '\n')
        return reason


def translation_jobs():
    try:
        jobs = int(os.environ.get('X2W_TRANSLATE_JOBS') or 0)
    except ValueError:
        jobs = 0
    jobs = jobs if jobs > 0 else os.cpu_count() or 1
    return min(jobs, 61) if os.name == 'nt' else jobs  # ProcessPoolExecutor refuses more than 61 on Windows


def translate_pending(out_dir, skip_failed=False, jobs=None):
    """Translate every request that has no result yet. The runtime writes the ready markers last.

    Requests are independent (each writes only its own <key>.* files), so several run in parallel
    processes; X2W_TRANSLATE_JOBS (default: the processor count, at most 61 on Windows) sets how many.
    """
    keys = pending_keys(out_dir, skip_failed)
    jobs = min(jobs or translation_jobs(), len(keys))
    if jobs < 2:
        for key in keys:
            print('%016x: %s' % (key, translate_one(out_dir, key)), flush=True)
        return
    from concurrent.futures import ProcessPoolExecutor
    chunk = max(1, len(keys) // (jobs * 4))  # small batches keep every worker busy
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for key, result in zip(keys, pool.map(translate_one, [out_dir] * len(keys), keys, chunksize=chunk)):
            print('%016x: %s' % (key, result), flush=True)


def pending_keys(out_dir, skip_failed):
    keys = []
    manifest_only = os.environ.get('X2W_FXC') == ''
    for name in sorted(os.listdir(out_dir)):
        m = re.fullmatch(r'req_([0-9a-f]{16})\.(a|rtv|c67|m57)', name)
        if not m:
            continue
        key = int(m.group(1), 16)
        result = os.path.join(out_dir, '%016x' % key)
        stages = ('.cs.cso',) if m.group(2) == 'c67' else ('.vs.cso', '.ps.cso')
        # A published manifest with bytecode (or manifest-only mode) is done; otherwise resume (#2008).
        if m.group(2) != 'm57' and os.path.isfile(result + '.txt') and (manifest_only or all(has_bytecode(result + s) for s in stages)):
            continue
        fail = os.path.join(out_dir, '%016x.fail' % key)
        # A repeated pass retries a failure only once its request is rewritten.
        if skip_failed and os.path.exists(fail) and os.path.getmtime(fail) >= os.path.getmtime(os.path.join(out_dir, name)):
            continue
        keys.append(key)
    return keys

if __name__ == '__main__':
    main()
