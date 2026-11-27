"""Run the translated vertex fetch of every buffer format gfx2hlsl lowers and read all four components.

usage: startup_vertex_fetch_formats_check.py <dxc> <offscreen_check> <adapter>
       startup_vertex_fetch_formats_check.py controls

tools/startup/gfx2hlsl.py lowers tbuffer_load_format_* for the supported buffer formats from a table of HLSL
expressions: the load of the word or words, the field each component takes and its conversion, and the
values of the missing components. Only BUF_FMT_8_UINT and BUF_FMT_8_UNORM were run before
(startup_r8_uint_check.py). This drives each table entry the way that check does: a source-built vertex
stage with one tbuffer_load_format_xyzw from vertex buffer 1 is translated, its generated vertex body is
called from a compute wrapper, and the four position components of 16 vertices are read back through
offscreen_check and compared with a reference that decodes the same bytes from the
format definitions (little-endian fields from the lowest byte, UNORM n / (2^bits - 1), binary16 widened to
binary32, missing X, Y and Z as 0 and a missing W as 1.0, or 1 for an integer format).

Each format with sub-dword fields uses a stride and offset that put elements at more than one byte
alignment, so a field that crosses a dword is read through ldu(); the 32-bit formats keep aligned elements. UNORM components are compared as round(v * (2^bits - 1)), so the
check does not depend on how many ulps the division is off by; every other component is compared bit for bit.
The same table also serves buffer_load_format_x/xy/xyz, whose format and component selection are not in the
instruction but in word 3 of the buffer descriptor (FORMAT bits 12..18, DST_SEL bits 0..11): the runtime passes
that word as an entry constant and the translation switches on it. Each format is run through a MUBUF
xyz load with its own descriptor word, then descriptor words with swizzles and constants 0 and 1 run through
the same translated body.
Each TBUFFER format runs its matching reference and each wrong reference that changes a word in one renderer
batch. The first mismatched word must match the predicted index and values. Each distinct vertex or compute
source compiles once in the fixture's temporary directory. Mutated translations dispatch separately.
"""
import hashlib
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from compute_load_format_check import check_references
from startup_sampler_heap_check import root_blob
from startup_stream_check import gfx2hlsl

SENTINEL = 0xc5c5c5c5
ONE = 0x3f800000
# offscreen_check --compute-store-case fills the raw t0 with this little-endian pattern.
DATA = b''.join(struct.pack('<I', (0x8b02a7f1 + i * 0x413b1) & 0xffffffff) for i in range(320))


def f16(bits):
    return struct.unpack('<I', struct.pack('<f', struct.unpack('<e', struct.pack('<H', bits))[0]))[0]


# name: (format code, stride, offset, field bits, field kind, number of fields). Kinds: 'unorm' (compared as
# round(v * max)), 'uint', 'f16' and 'f32' (compared as bits). The code is the gfx10 unified format number.
FORMATS = {
    'BUF_FMT_8_UNORM': (1, 3, 1, 8, 'unorm', 1),
    'BUF_FMT_8_UINT': (5, 3, 1, 8, 'uint', 1),
    'BUF_FMT_8_8_UNORM': (14, 3, 2, 8, 'unorm', 2),
    'BUF_FMT_16_16_UNORM': (23, 6, 2, 16, 'unorm', 2),
    # One dword: x, y, z are 10-bit fields (bits 9:0, 19:10, 29:20) and w the 2-bit field in bits 31:30. The field
    # width is listed as 16 only so the stride and offset checks see sub-dword alignment; see PACKED_2101010.
    'BUF_FMT_2_10_10_10_UNORM': (50, 6, 2, 16, 'unorm2101010', 4),
    'BUF_FMT_16_16_SNORM': (24, 6, 2, 16, 'snorm', 2),
    'BUF_FMT_16_16_UINT': (27, 6, 2, 16, 'uint', 2),
    'BUF_FMT_16_16_FLOAT': (29, 10, 10, 16, 'f16', 2),
    'BUF_FMT_16_16_16_16_FLOAT': (71, 18, 10, 16, 'f16', 4),
    'BUF_FMT_8_8_8_8_UNORM': (56, 5, 3, 8, 'unorm', 4),
    'BUF_FMT_8_8_8_8_UINT': (60, 5, 3, 8, 'uint', 4),
    'BUF_FMT_32_FLOAT': (22, 8, 4, 32, 'f32', 1),
    'BUF_FMT_32_32_FLOAT': (64, 12, 0, 32, 'f32', 2),
    'BUF_FMT_32_32_32_FLOAT': (74, 16, 4, 32, 'f32', 3),
    'BUF_FMT_32_32_32_32_FLOAT': (77, 20, 8, 32, 'f32', 4),
}


PACKED_2101010 = ((0, 10), (10, 10), (20, 10), (30, 2))  # (low bit, width) of x, y, z, w


def snorm_words(fields, unsigned=False, no_clamp=False):
    """SNORM fields as the compared integers round(v * 32767): two's complement, -32768 clamped to -32767."""
    out = []
    for f in fields:
        v = f if unsigned else f - 0x10000 if f & 0x8000 else f
        out.append((v if no_clamp else max(v, -32767)) & 0xffffffff)
    return out


def reference(name, vertex, *, offset=None, swap=False, shift=False, w_zero=False, unorm_pow2=False, unsigned=False, no_clamp=False):
    """The four compared words of one vertex, from the format definition. The keywords give wrong references."""
    _, stride, own_offset, bits, kind, count = FORMATS[name]
    address, size = vertex * stride + (own_offset if offset is None else offset), bits // 8
    if kind == 'unorm2101010':
        word = int.from_bytes(DATA[address + (1 if shift else 0):address + (1 if shift else 0) + 4], 'little')
        fields = [(word >> low) & ((1 << width) - 1) for low, width in PACKED_2101010]
        if swap:
            fields[0], fields[1] = fields[1], fields[0]
        if unorm_pow2:
            fields = [round(f * ((1 << w) - 1) / (1 << w)) for f, (_, w) in zip(fields, PACKED_2101010)]
        return fields
    fields = [int.from_bytes(DATA[address + size * i:address + size * (i + 1)], 'little') for i in range(count)]
    if shift and bits < 32:
        # Each field read one field higher, as a wrong shift would; the last reads past the element.
        fields = [int.from_bytes(DATA[address + size * (i + 1):address + size * (i + 2)], 'little') for i in range(count)]
    if swap and count > 1:
        fields[0], fields[1] = fields[1], fields[0]
    if kind == 'unorm' and unorm_pow2:
        # v = field / 2^bits, compared as round(v * (2^bits - 1)).
        fields = [round(f * ((1 << bits) - 1) / (1 << bits)) for f in fields]
    if kind == 'snorm':
        fields = snorm_words(fields, unsigned=unsigned, no_clamp=no_clamp)
    words = [f16(f) if kind == 'f16' else f for f in fields]
    one = 1 if kind == 'uint' else ONE
    missing = [0, 0, 0, 0 if w_zero else one]
    return words + missing[count:]


def compared(name):
    """Return the integer readback scale for each normalized component."""
    _, _, _, bits, kind, count = FORMATS[name]
    if kind == 'unorm2101010':
        return [(1 << width) - 1 for _, width in PACKED_2101010]
    scale = {'unorm': (1 << bits) - 1, 'snorm': (1 << (bits - 1)) - 1}.get(kind)
    return [scale if i < count else None for i in range(4)]


def expected(name, **wrong):
    words = [SENTINEL] * 320
    for vertex in range(16):
        words[vertex * 4:vertex * 4 + 4] = reference(name, vertex, **wrong)
    return words


def pipeline(name):
    code, _, offset, _, _, _ = FORMATS[name]
    # s_load_dwordx4 s[16:19], s[0:1], 16  (vertex-buffer descriptor 1)
    # tbuffer_load_format_xyzw v[4:7], v0, s[16:19], 0 idxen offset:<offset> format:<code>
    # exp pos0 v4, v5, v6, v7 done
    words = (0xf4080400, 0xfa000010, 0xe8032000 | offset | code << 19, 0x80040400,
             0xf80008cf, 0x07060504, 0xbf810000)
    vs = struct.pack('<7I', *words)
    ps = struct.pack('<3I', 0xf800180f, 0, 0xbf810000)
    result = gfx2hlsl.Pipeline(vs, ps, root_blob(flags=0))
    assert 'srv 0 buf vb 1' in result.manifest().splitlines()
    return result


def wrapper(name):
    parts = ['bits.%s' % c if scale is None else '(uint)(int)round(p.%s * %d.0)' % (c, scale)
             for c, scale in zip('xyzw', compared(name))]
    return '''
RWByteAddressBuffer outputWords : register(u0);
[numthreads(16, 1, 1)]
void readback(uint id : SV_GroupIndex) {
    float4 p = body(id).pos;
    uint4 bits = asuint(p);
    outputWords.Store4(id * 16, uint4(%s));
}
''' % ', '.join(parts)


# Translations a wrong table entry would produce: (format, label, generated text, replacement).
MUTATIONS = (
    ('BUF_FMT_8_8_8_8_UNORM', 'second byte from the third', 'U(((w >> 8) & 255) / 255.0)', 'U(((w >> 16) & 255) / 255.0)'),
    ('BUF_FMT_8_8_8_8_UNORM', 'fourth byte as 255', 'U((w >> 24) / 255.0)', 'U(1.0)'),
    ('BUF_FMT_2_10_10_10_UNORM', 'y from the x field', 'v5 = U(((w >> 10) & 1023) / 1023.0);', 'v5 = U((w & 1023) / 1023.0);'),
    ('BUF_FMT_2_10_10_10_UNORM', 'z from the y field', 'v6 = U(((w >> 20) & 1023) / 1023.0);', 'v6 = U(((w >> 10) & 1023) / 1023.0);'),
    ('BUF_FMT_2_10_10_10_UNORM', 'w over 4', 'v7 = U((w >> 30) / 3.0);', 'v7 = U((w >> 30) / 4.0);'),
    ('BUF_FMT_8_8_8_8_UINT', 'normalized green byte', 'v5 = (w >> 8) & 255u;', 'v5 = U(((w >> 8) & 255u) / 255.0);'),
    ('BUF_FMT_8_8_8_8_UINT', 'blue byte from green', 'v6 = (w >> 16) & 255u;', 'v6 = (w >> 8) & 255u;'),
    ('BUF_FMT_8_8_8_8_UINT', 'alpha byte from blue', 'v7 = w >> 24;', 'v7 = (w >> 16) & 255u;'),
    ('BUF_FMT_8_8_UNORM', 'aligned load instead of ldu', 'ldu(r0, a)', 'r0.Load(a & ~3u)'),
    ('BUF_FMT_16_16_UNORM', 'high half over 65536', 'U((w >> 16) / 65535.0)', 'U((w >> 16) / 65536.0)'),
    ('BUF_FMT_16_16_UINT', 'normalized low half', 'v4 = w & 65535u;', 'v4 = U((w & 65535u) / 65535.0);'),
    ('BUF_FMT_16_16_UINT', 'high half from the low half', 'v5 = w >> 16;', 'v5 = w & 65535u;'),
    ('BUF_FMT_16_16_UINT', 'missing W as 1.0', 'v7 = 1u;', 'v7 = U(1.0);'),
    ('BUF_FMT_16_16_SNORM', 'low half read unsigned', 'float(int(w << 16) >> 16)', 'float((w << 16) >> 16)'),
    ('BUF_FMT_16_16_FLOAT', 'low half from the high half', 'v4 = U(f16tof32(w));', 'v4 = U(f16tof32(w >> 16));'),
    ('BUF_FMT_16_16_16_16_FLOAT', 'second dword aligned down',
     'v7 = U(f16tof32(ldu(r0, a + 4) >> 16));', 'v7 = U(f16tof32(r0.Load((a + 4) & ~3u) >> 16));'),
    ('BUF_FMT_32_32_32_FLOAT', 'missing W as 0', 'U(1.0)', '0u'),
    ('BUF_FMT_32_32_32_32_FLOAT', 'W from the third word', 'r0.Load(a + 12)', 'r0.Load(a + 8)'),
)
# Wrong references; each must fail the readback of every format where it changes a word.
WRONG = (('swapped channels', dict(swap=True)), ('fields one higher', dict(shift=True)),
         ('missing W as 0', dict(w_zero=True)), ('UNORM over 2^bits', dict(unorm_pow2=True)),
         ('SNORM read unsigned', dict(unsigned=True)))

# The generated SNORM expressions evaluated on fixed halves (the device fills t0 with a fixed pattern that
# holds no 0x8000, so the boundary values are checked here, on the text the translator emits).
HALVES = {0x8000: -1.0, 0x8001: -1.0, 0x7fff: 1.0, 0x0000: 0.0, 0xffff: -1.0 / 32767.0, 0x4000: 16384 / 32767.0}


class Word(int):
    """A uint: shifts left wrap at 32 bits."""
    def __lshift__(self, n):
        return Word((int(self) << n) & 0xffffffff)


def to_int(x):
    x = int(x) & 0xffffffff
    return x - (1 << 32) if x >> 31 else x


def snorm_value(expression, word):
    """Evaluate one generated component expression as HLSL would (uint w, int() as a bit cast)."""
    text = expression.replace('U(', 'identity(').replace('int(', 'to_int(')
    return eval(text, {'identity': lambda x: x, 'to_int': to_int, 'w': Word(word), 'max': max, 'float': float})


def snorm_boundaries(components):
    """Raise AssertionError unless both halves of the word decode every HALVES entry exactly."""
    for half, value in HALVES.items():
        for low_first in (True, False):
            other = 0x1234  # a nonzero neighbour: the other half must not leak in
            word = (other << 16 | half) if low_first else (half << 16 | other)
            got = snorm_value(components[0 if low_first else 1], word)
            assert abs(got - value) < 1e-6, (hex(half), low_first, got, value)


IDENTITY = 4 | 5 << 3 | 6 << 6 | 7 << 9  # DST_SEL X, Y, Z, W: the four components in order


# How the descriptor reaches the load: straight from the table pointer, through a copy of that pointer made in an
# earlier block, or loaded in an earlier block than the formatted load.
LAYOUTS = {
    'plain': ((), (0xf4080404, 0xfa000000), ()),
    'pointer copy': ((0xbe8a0308, 0xbf820000), (0xf4080405, 0xfa000000), ()),  # s_mov_b32 s10, s8; s_branch +0; load via s[10:11]
    'split blocks': ((), (0xf4080404, 0xfa000000), (0xbf820000,)),  # s_branch +0 between the load and the formatted load
}


def pipeline_mubuf(offset, op=2, layout='plain'):
    """A source-built vertex stage: buffer_load_format_xyz (op 2; 0 is _x, 1 is _xy) from a hardware table descriptor."""
    # s_load_dwordx4 s[16:19], s[8:9], 0
    # buffer_load_format_xyz v[4:6], v0, s[16:19], 0 idxen offset:<offset>
    # exp pos0 v4, v5, v6, v7 done
    before, load, between = LAYOUTS[layout]
    words = (*before, *load, *between, 0xe0002000 | op << 18 | offset, 0x80040400,
             0xf80008cf, 0x07060504, 0xbf810000)
    ps = struct.pack('<3I', 0xf800180f, 0, 0xbf810000)
    result = gfx2hlsl.Pipeline(struct.pack('<%dI' % len(words), *words), ps, root_blob(flags=0))
    assert 'srv 0 buf table 0 0' in result.manifest().splitlines()
    assert 'udsrc 5 descword3 0' in result.manifest().splitlines()
    return result


def descriptor_word3(name, selects=(4, 5, 6, 7)):
    """Word 3 of a buffer descriptor: FORMAT in bits 12..18, then DST_SEL x, y, z, w in three bits each."""
    return FORMATS[name][0] << 12 | sum(sel << 3 * i for i, sel in enumerate(selects))


def mubuf_expected(name, count, selects=(4, 5, 6, 7), integer_one=False):
    """The four compared words per vertex of a buffer_load_format_* with `count` registers and these DST_SEL values."""
    words = [SENTINEL] * 320
    one = 1 if integer_one else ONE
    for vertex in range(16):
        comps = reference(name, vertex)
        comps = [comps[0], comps[1], comps[2], comps[3]]
        out = [{4: comps[0], 5: comps[1], 6: comps[2], 7: comps[3], 1: one}.get(sel, 0) for sel in selects]
        words[vertex * 4:vertex * 4 + 4] = out[:count] + [0] * (4 - count)
    return words


def mutate_case(source, code, old, new):
    """Replace `old` by `new` inside the switch case of one format code only."""
    lines = source.split('\n')
    marker = 'case %du:' % code
    hits = [i for i, line in enumerate(lines) if line.strip().startswith(marker) and old in line]
    assert len(hits) == 1, (code, old)
    lines[hits[0]] = lines[hits[0]].replace(old, new, 1)
    return '\n'.join(lines)


# Translations a wrong switch would produce on the descriptor path: (label, old text, new text).
DESC_MUTATIONS = (
    ('format field ignored (always 4 half floats)', '(ud5 >> 12) & 127u', '71u'),
    ('format field one bit low', '(ud5 >> 12) & 127u', '(ud5 >> 11) & 127u'),
    ('DST_SEL y from the x field', '((ud5 >> 3) & 7u)', '((ud5 >> 0) & 7u)'),
    ('DST_SEL constant 1 as 0', 'one = U(1.0)', 'one = 0u'),
)
# Integer 1 for the constant of an integer format, and the switch cases themselves, are mutated per format below.
# buffer_load_format_xyzw device cases: (label, format, DST_SEL x, y, z, w). The fourth register is never 0 here.
XYZW_CASES = (
    ('xyzw', 'BUF_FMT_32_32_32_32_FLOAT', (4, 5, 6, 7)),
    ('xyzw with missing z and w', 'BUF_FMT_16_16_FLOAT', (4, 5, 6, 7)),
    ('xyzw read as w, 1, 0, x', 'BUF_FMT_32_32_32_32_FLOAT', (7, 1, 0, 4)),
)


def mubuf_cases():
    """(label, format used for the offset and the reference, descriptor word 3, expected words) for the device run."""
    cases = []
    for name in FORMATS:
        cases.append(('%s xyz' % name, name, descriptor_word3(name), mubuf_expected(name, 3)))
    wide, small = 'BUF_FMT_32_32_32_32_FLOAT', 'BUF_FMT_8_UINT'
    cases.append(('xyzw read as y, x, 1', wide, descriptor_word3(wide, (5, 4, 1, 7)), mubuf_expected(wide, 3, (5, 4, 1))))
    cases.append(('xyzw read as w, 0, z', wide, descriptor_word3(wide, (7, 0, 6, 7)), mubuf_expected(wide, 3, (7, 0, 6))))
    cases.append(('uint constant 1 stays integer', small, descriptor_word3(small, (4, 1, 0, 7)),
                  mubuf_expected(small, 3, (4, 1, 0), integer_one=True)))
    return cases


def controls():
    """Translation-side checks: every table entry is reached, and the reference is not degenerate."""
    for name in FORMATS:
        source = pipeline(name).hlsl_vs()
        assert 'v7' in source, name
    for name, label, old, new in MUTATIONS:
        source = pipeline(name).hlsl_vs()
        assert old in source, (name, label)
    runtime = (Path(__file__).resolve().parents[1] / 'src' / 'startup_stubs' / 'formatted_buffer.h').read_text()
    assert set(map(int, re.findall(r'case (\d+):', runtime))) == set(gfx2hlsl.DESC_FORMAT_CODES.values())
    # SNORM: the emitted expressions hit the boundaries, and each wrong form is caught by the same check.
    components = gfx2hlsl.FETCH_FORMATS['BUF_FMT_16_16_SNORM'][1][:2]
    snorm_boundaries(components)
    wrong_forms = (('read unsigned', lambda c: c.replace('int(w << 16) >> 16', '(w << 16) >> 16').replace('int(w) >> 16', 'w >> 16')),
                   ('no clamp', lambda c: c.replace('max(', '(').replace(', -1.0)', ')')),
                   ('divide by 32768', lambda c: c.replace('32767.0', '32768.0')),
                   ('low half only', lambda c: components[0]))
    for label, edit in wrong_forms:
        try:
            snorm_boundaries([edit(c) for c in components])
        except AssertionError:
            continue
        raise AssertionError('SNORM wrong form %r escaped detection' % label)
    # The reference reads every byte of each element: changing any byte of vertex 5 changes a word.
    for name, (_, stride, offset, bits, kind, count) in FORMATS.items():
        for label, wrong in WRONG:
            if expected(name, **wrong) != expected(name):
                break
        else:
            raise AssertionError('no wrong reference changes %s' % name)
    # 16-bit floats in the chosen elements are normal, so f16tof32 is exact on every adapter.
    for name, (_, stride, offset, _, kind, count) in FORMATS.items():
        if kind != 'f16':
            continue
        for vertex in range(16):
            address = vertex * stride + offset
            halves = struct.unpack_from('<' + 'H' * count, DATA, address)
            assert all(0 < ((half >> 10) & 31) < 31 for half in halves), (name, vertex, halves)
            if count == 4:
                assert len(set(halves)) == 4, (name, vertex, halves)
    # The words of the 32-bit formats are finite and normal, so the position passes through unchanged.
    for name in ('BUF_FMT_32_FLOAT', 'BUF_FMT_32_32_FLOAT', 'BUF_FMT_32_32_32_FLOAT', 'BUF_FMT_32_32_32_32_FLOAT'):
        _, _, _, _, _, count = FORMATS[name]
        for vertex in range(16):
            for word in reference(name, vertex)[:count]:
                assert 0 < (word >> 23) & 0xff < 0xff, (name, vertex, hex(word))
    # Several elements of each format with sub-dword fields sit at different byte alignments; the 32-bit
    # formats are read with aligned Load() and keep 4-byte aligned elements.
    for name, (_, stride, offset, bits, _, _) in FORMATS.items():
        alignments = {(v * stride + offset) % 4 for v in range(16)}
        assert len(alignments) > 1 if bits < 32 else alignments == {0}, name

    # buffer_load_format_x/xy/xyz: every table format has a switch case, and the codes match the disassembler's.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
    from disasm_rdna2 import MTBUF_FORMATS
    assert gfx2hlsl.DESC_FORMAT_CODES.keys() == gfx2hlsl.FETCH_FORMATS.keys()
    def table_matches(table):
        for name, (code, *_rest) in FORMATS.items():
            if code == 1:  # the omitted tbuffer format: LLVM prints nothing, the disassembler has no name for it
                if code in table or name != 'BUF_FMT_8_UNORM':
                    return False
            elif table.get(code) != name:
                return False
        return True
    for name, (code, *_rest) in FORMATS.items():
        assert gfx2hlsl.DESC_FORMAT_CODES[name] == code, name
    assert table_matches(MTBUF_FORMATS)
    swapped = dict(MTBUF_FORMATS)  # a wrong format name for one code must be caught
    victim = next(c for n, (c, *_r) in FORMATS.items() if c != 1)
    swapped[victim] = 'BUF_FMT_INVALID'
    assert not table_matches(swapped), 'format mutation not caught'
    assert not table_matches({**MTBUF_FORMATS, 1: 'BUF_FMT_8_UNORM'}), 'code 1 must stay the omitted default'
    for label, name, selects in XYZW_CASES:
        words = mubuf_expected(name, 4, selects)
        assert all(words[v * 4 + 3] for v in range(16)), label  # an xyz translation leaves this register 0
    for op, count in ((0, 1), (1, 2), (2, 3), (3, 4)):
        source = pipeline_mubuf(4, op).hlsl_vs()
        assert source.count('case ') == len(FORMATS), op
        # An unvalidated FORMAT yields the NaN sentinel in every component, never silent zeros.
        assert 'default: c0 = c1 = c2 = c3 = %du;' % gfx2hlsl.DESC_FORMAT_UNKNOWN in source, op
        assert 'default: break' not in source, op
        written = {int(n) for n in re.findall(r'^\s*v(\d+) = ', source, re.M) if int(n) >= 4}
        assert written == set(range(4, 4 + count)), (op, written)
    # A shader that rewrites descriptor word 3 (s_mov_b32 s19, 0x50fac) after the load must not be translated
    # with the table word: the formatted load is refused.
    try:
        gfx2hlsl.Pipeline(struct.pack('<8I', 0xf4080404, 0xfa000000, 0xbe9303ff, 0x50fac, 0xe0002000 | 2 << 18, 0x80040400,
                                      0xf80008cf, 0x07060504) + struct.pack('<I', 0xbf810000),
                          struct.pack('<3I', 0xf800180f, 0, 0xbf810000), root_blob(flags=0))
    except gfx2hlsl.Unsupported as error:
        assert 'rewritten' in str(error), error
    else:
        raise AssertionError('formatted load through a rewritten descriptor word 3 must be refused')
    # A branch can bypass the descriptor load, even though the decoder sees the load before the use.
    branch = (0xbf840002, 0xf4080404, 0xfa000000, 0xe0002000, 0x80040400,
              0xf80008cf, 0x07060504, 0xbf810000)
    ps = struct.pack('<3I', 0xf800180f, 0, 0xbf810000)
    try:
        gfx2hlsl.Pipeline(struct.pack('<8I', *branch), ps, root_blob(flags=0))
    except gfx2hlsl.Unsupported as error:
        assert 'proven in the entry block' in str(error) or 'joined at a branch target' in str(error), error
    else:
        raise AssertionError('formatted load accepted when a branch could skip the descriptor load')
    def stage(words):
        return gfx2hlsl.Pipeline(struct.pack('<%dI' % len(words), *words), ps, root_blob(flags=0))

    load = (0xf4080404, 0xfa000000)
    formatted_tail = (0xe0002000, 0x80040400, 0xf80008cf, 0x07060504, 0xbf810000)
    # Both edges reach a block that reloads all four words from the unmodified table pointer.
    local_reload = stage((0xbf840001, 0xbf800000, *load, *formatted_tail))
    assert 'srv 0 buf table 0 0' in local_reload.manifest().splitlines()
    # One edge clobbers the entry pointer before the reload. The last lexical path restores it,
    # but the branch can skip that restore, so a proof based only on translation order is wrong.
    changed_pointer = (0xbe920308, 0xbe880380, 0xbf840001, 0xbe880312, *load, *formatted_tail)
    try:
        stage(changed_pointer)
    except gfx2hlsl.Unsupported as error:
        assert 'local table load' in str(error) or 'joined at a branch target' in str(error), error
    else:
        raise AssertionError('formatted load accepted after a possibly clobbered root pointer')
    # A copy of the table pointer made in a block above the load is proven, for graphics as for compute stages.
    copied = stage((0xbe8a0308, 0xbf820000, 0xf4080405, 0xfa000000, *formatted_tail))
    assert 'srv 0 buf table 0 0' in copied.manifest().splitlines()
    # The descriptor loaded in an earlier block than the formatted load is proven the same way.
    split = stage((*load, 0xbf820000, *formatted_tail))
    assert 'srv 0 buf table 0 0' in split.manifest().splitlines()
    # Control: a copy of something that is not the table pointer, a rewritten copy, and a word rewritten after the load.
    for words in ((0xbe8a0314, 0xbf820000, 0xf4080405, 0xfa000000, *formatted_tail),      # s_mov_b32 s10, s20
                  (0xbe8a0308, 0xbf820000, 0xbe8a0380, 0xf4080405, 0xfa000000, *formatted_tail),  # copy, then s_mov_b32 s10, 0
                  (*load, 0xbf820000, 0xbe900380, *formatted_tail)):                        # s_mov_b32 s16, 0 before the use
        try:
            stage(words)
        except gfx2hlsl.Unsupported:
            pass
        else:
            raise AssertionError('formatted load accepted without a proven descriptor: %r' % (words,))
    source = pipeline_mubuf(0).hlsl_vs()
    for label, old, new in DESC_MUTATIONS:
        assert old in source, label
    runtime_gate_controls()
    # Each reference is built from the format table: no case may be degenerate (an all-zero or all-sentinel set).
    for label, name, word3, words in mubuf_cases():
        assert words != [SENTINEL] * 320 and (any(words[:64]) or 'zeros' in label), label
    assert mubuf_expected('BUF_FMT_32_32_32_32_FLOAT', 3, (5, 4, 1)) != mubuf_expected('BUF_FMT_32_32_32_32_FLOAT', 3)


RUNTIME = Path(__file__).resolve().parents[1] / 'src' / 'startup_stubs' / 'd3d12_draw.inl'


def runtime_gate(source):
    """Check that both FORMAT metadata and usable backing are required before submission."""
    assert 'formatted_buffers_ready(*tr, pass, formatted_buffers)' in source
    decoder = source[source.index('const char* formatted_buffer_word3('):source.index('bool formatted_buffers_ready(')]
    assert 'formatted_buffer_supported(' in decoder
    preflight = source[source.index('bool formatted_buffers_ready('):source.index('bool draw_translated(')]
    for condition in ('x2w_local::readable(', 'view_token_magic', 'formatted_buffer_word3('):
        assert condition in preflight, condition
    start = source.index('bool draw_translated(')
    gate = source.index('if (!have && formatted.entry)', start)
    assert source.index('formatted_buffers_ready(*tr, pass, formatted_buffers)', start) < gate
    assert gate < source.index('if (!have && identity_for_null)', gate)
    assert gate < source.index('list->SetPipelineState(pipeline)', gate)


def runtime_gate_controls():
    source = RUNTIME.read_text(encoding='utf-8')
    runtime_gate(source)
    decoder = source[source.index('const char* formatted_buffer_word3('):source.index('bool formatted_buffers_ready(')]
    try:
        runtime_gate(source.replace(decoder, decoder.replace('formatted_buffer_supported(', 'omitted')))
    except AssertionError:
        pass
    else:
        raise AssertionError('descriptor decoder without FORMAT support checking must fail the control')
    preflight = source[source.index('bool formatted_buffers_ready('):source.index('bool draw_translated(')]
    for dropped in ('x2w_local::readable(', 'view_token_magic', 'formatted_buffer_word3('):
        try:
            runtime_gate(source.replace(preflight, preflight.replace(dropped, 'omitted')))
        except AssertionError:
            continue
        raise AssertionError('gate without %s must fail the control' % dropped)
    start = source.index('bool draw_translated(')
    gate = 'if (!have && formatted.entry)'
    start_cmd = 'list->SetPipelineState(pipeline);'
    late = source[:start] + source[start:].replace(gate, '', 1).replace(start_cmd, start_cmd + '\n' + gate, 1)
    try:
        runtime_gate(late)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('gate after binding fallback or list commands must fail the control')


def compile_fetch(directory, compiler, name, source):
    hlsl, output = directory / 'fetch.hlsl', directory / 'fetch.cso'
    # Descriptor words can change without changing the vertex program. The readback wrapper
    # also depends on the format, so key each stage by its complete input in this run's temporary directory.
    for profile, entry, text in (('vs_6_0', 'main', source), ('cs_6_0', 'readback', source + wrapper(name))):
        binary = directory / ('%s-%s.cso' % (profile, hashlib.sha256(text.encode('ascii')).hexdigest()))
        if not binary.exists():
            hlsl.write_text(text, encoding='ascii')
            subprocess.run([compiler, '-T', profile, '-E', entry, '-Fo', output, hlsl], check=True)
            output.replace(binary)
    return binary


def write_reference(case, name, words, translated, word3=None):
    constants = [0] * 16
    constants[translated.uds.index(('stride', 0))] = FORMATS[name][1]
    if word3 is not None:
        constants[translated.uds.index(('descword3', 0))] = word3
    case.write_bytes(struct.pack('<337I', 1, *words, *constants))


def run_reference(directory, consumer, adapter, binary, name, words, translated=None, word3=None):
    case = directory / 'fetch.case'
    write_reference(case, name, words, translated or pipeline(name), word3)
    return subprocess.run([consumer, adapter, '--compute-store-case', binary, case], capture_output=True, text=True)


def run(directory, compiler, consumer, adapter, name, source, words, translated=None, word3=None):
    binary = compile_fetch(directory, compiler, name, source)
    return run_reference(directory, consumer, adapter, binary, name, words, translated, word3)


def check(directory, compiler, consumer, adapter):
    for name in FORMATS:
        translated = pipeline(name)
        binary = compile_fetch(directory, compiler, name, translated.hlsl_vs())
        correct = expected(name)
        case = directory / 'fetch-correct.case'
        write_reference(case, name, correct, translated)
        references = [(case, '16 vertices, stride %d, offset %d' % FORMATS[name][1:3], None)]
        for label, wrong in WRONG:
            words = expected(name, **wrong)
            if words == correct:
                continue
            first = next(i for i, (a, b) in enumerate(zip(words, correct)) if a != b)
            case = directory / f'fetch-wrong-{len(references)}.case'
            write_reference(case, name, words, translated)
            diagnostic = f'word={first} expected={words[first]:08x} actual={correct[first]:08x}'
            references.append((case, label, diagnostic))
        if check_references(name, consumer, adapter, binary, references, allow_skip=True) == 77:
            return 77
    for name, label, old, new in MUTATIONS:
        source = pipeline(name).hlsl_vs()
        binary = compile_fetch(directory, compiler, name, source.replace(old, new, 1))
        bad = run_reference(directory, consumer, adapter, binary, name, expected(name))
        if bad.returncode != 1 or 'Compute readback differs' not in bad.stderr:
            raise AssertionError('%s: mutation %r escaped detection\n%s' % (name, label, bad.stderr))
        print('%s: mutation %r fails' % (name, label))
    print('%s: %d buffer formats, 16 vertices each, every wrong reference and %d mutations fail'
          % (adapter, len(FORMATS), len(MUTATIONS)))
    mubuf_runs = 0
    for label, name, word3, words in mubuf_cases():
        translated = pipeline_mubuf(FORMATS[name][2])
        source = translated.hlsl_vs()
        result = run(directory, compiler, consumer, adapter, name, source, words, translated, word3)
        if result.returncode:
            raise RuntimeError('descriptor %s: %s%s' % (label, result.stdout, result.stderr))
        print('descriptor %s: %s' % (label, [l for l in result.stdout.splitlines() if 'mismatches=' in l][0]))
        mubuf_runs += 1
    # The descriptor reached through a pointer copy or loaded in an earlier block reads the same bytes.
    for layout in ('pointer copy', 'split blocks'):
        for label, name, word3, words in mubuf_cases():
            if 'xyz' not in label and 'as' not in label:
                continue
            translated = pipeline_mubuf(FORMATS[name][2], 2, layout)
            result = run(directory, compiler, consumer, adapter, name, translated.hlsl_vs(), words, translated, word3)
            if result.returncode:
                raise RuntimeError('%s, descriptor %s: %s%s' % (layout, label, result.stdout, result.stderr))
            mubuf_runs += 1
        translated = pipeline_mubuf(FORMATS['BUF_FMT_32_32_32_32_FLOAT'][2], 2, layout)
        # Control: the descriptor word is really read through this layout, so another format must fail the reference.
        bad = run(directory, compiler, consumer, adapter, 'BUF_FMT_32_32_32_32_FLOAT', translated.hlsl_vs(),
                  mubuf_expected('BUF_FMT_32_32_32_32_FLOAT', 3), translated, descriptor_word3('BUF_FMT_32_32_FLOAT'))
        if bad.returncode != 1 or 'Compute readback differs' not in bad.stderr:
            raise AssertionError('%s: another descriptor format passed the reference\n%s' % (layout, bad.stderr))
        print('%s: matches the reference, another format fails' % layout)
    # The same translated body must fail when the descriptor word names another format, or when it is mutated.
    name = 'BUF_FMT_32_32_32_32_FLOAT'
    translated = pipeline_mubuf(FORMATS[name][2])
    source = translated.hlsl_vs()
    for other in ('BUF_FMT_32_32_FLOAT', 'BUF_FMT_16_16_16_16_FLOAT'):
        bad = run(directory, compiler, consumer, adapter, name, source, mubuf_expected(name, 3), translated, descriptor_word3(other))
        if bad.returncode != 1 or 'Compute readback differs' not in bad.stderr:
            raise AssertionError('descriptor word of %s did not change the load\n%s' % (other, bad.stderr))
    for label, old, new in DESC_MUTATIONS:
        for case_label, case_name, word3, words in mubuf_cases():
            if 'xyz' not in case_label and 'as' not in case_label and 'integer' not in case_label:
                continue
            translated = pipeline_mubuf(FORMATS[case_name][2])
            bad = run(directory, compiler, consumer, adapter, case_name, translated.hlsl_vs().replace(old, new), words, translated, word3)
            if bad.returncode == 1 and 'Compute readback differs' in bad.stderr:
                print('descriptor mutation %r fails on %s' % (label, case_label))
                break
        else:
            raise AssertionError('descriptor mutation %r escaped detection' % label)
    for name, label, old, new in MUTATIONS:
        translated = pipeline_mubuf(FORMATS[name][2])
        # The descriptor switch names its components c0..c3 where the tbuffer translation names v4..v7.
        old, new = (re.sub(r'\bv([4-7]) =', lambda m: 'c%d =' % (int(m.group(1)) - 4), t) for t in (old, new))
        source = mutate_case(translated.hlsl_vs(), FORMATS[name][0], old, new)
        # An xyz load never reads W, so a W-only mutation needs a word routing W into the third register.
        for selects in ((4, 5, 6, 7), (4, 5, 7, 7)):
            bad = run(directory, compiler, consumer, adapter, name, source, mubuf_expected(name, 3, selects), translated,
                      descriptor_word3(name, selects))
            if bad.returncode == 1 and 'Compute readback differs' in bad.stderr:
                break
        else:
            raise AssertionError('%s: descriptor-path mutation %r escaped detection\n%s' % (name, label, bad.stderr))
        print('%s: descriptor-path mutation %r fails' % (name, label))
    print('%s: %d buffer_load_format descriptor cases, wrong descriptor words and %d mutations fail'
          % (adapter, mubuf_runs, len(MUTATIONS) + len(DESC_MUTATIONS)))
    # buffer_load_format_xyzw: the fourth register holds what DST_SEL w routes there, and the xyz translation of the
    # same load (fourth register left 0) must fail each reference.
    for label, name, selects in XYZW_CASES:
        word3, words = descriptor_word3(name, selects), mubuf_expected(name, 4, selects)
        translated = pipeline_mubuf(FORMATS[name][2], 3)
        result = run(directory, compiler, consumer, adapter, name, translated.hlsl_vs(), words, translated, word3)
        if result.returncode:
            raise RuntimeError('descriptor %s: %s%s' % (label, result.stdout, result.stderr))
        print('descriptor %s: %s' % (label, [l for l in result.stdout.splitlines() if 'mismatches=' in l][0]))
        xyz = pipeline_mubuf(FORMATS[name][2], 2)
        bad = run(directory, compiler, consumer, adapter, name, xyz.hlsl_vs(), words, xyz, word3)
        if bad.returncode != 1 or 'Compute readback differs' not in bad.stderr:
            raise AssertionError('descriptor %s: the xyz form passed the xyzw reference\n%s' % (label, bad.stderr))
        print('descriptor %s: the xyz form fails' % label)
    return 0


if __name__ == '__main__':
    controls()
    if sys.argv[1:] == ['controls']:
        print('vertex fetch format controls passed')
        sys.exit(0)
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
