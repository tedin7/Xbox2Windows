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
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import disasm_rdna2 as D  # noqa: E402


class Unsupported(Exception):
    pass


PREFETCH = bytes.fromhex('0300a0bf')

# SDWA (sub-dword addressing) selections, with the same meaning as tools/pixel_aot.py: a source
# selection is zero-extended, or sign-extended under sext(); a destination selection writes the low
# bits of the result into that field, UNUSED_PAD zeroes the others and UNUSED_PRESERVE keeps them.
SDWA_SEL = {'BYTE_0': (0, 8), 'BYTE_1': (8, 8), 'BYTE_2': (16, 8), 'BYTE_3': (24, 8),
            'WORD_0': (0, 16), 'WORD_1': (16, 16), 'DWORD': (0, 32)}


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
              61: ('BUF_FMT_8_UNORM', 1), 34: ('BUF_FMT_16_16_FLOAT', 4), 35: ('BUF_FMT_16_16_UNORM', 4)}
# Typed buffer loads: (word load or None, the four component expressions). Components a format lacks read as
# 0, 0, 0, 1 here, as in the existing tbuffer translation; that default is an assumption, not a measurement.
FETCH_FORMATS = {
    'BUF_FMT_8_8_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', 'U(((w >> 16) & 255) / 255.0)', 'U((w >> 24) / 255.0)']),
    'BUF_FMT_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', '0u', 'U(1.0)']),
    'BUF_FMT_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', '0u', '0u', 'U(1.0)']),
    'BUF_FMT_8_UINT': ('{0}.Load(a & ~3u) >> ((a & 3u) * 8u)', ['w & 255u', '0u', '0u', '1u']),
    'BUF_FMT_16_16_UNORM': ('ldu({0}, a)', ['U((w & 65535) / 65535.0)', 'U((w >> 16) / 65535.0)', '0u', 'U(1.0)']),
    'BUF_FMT_16_16_FLOAT': ('ldu({0}, a)', ['U(f16tof32(w))', 'U(f16tof32(w >> 16))', '0u', 'U(1.0)']),
    # Four half channels in one 8-byte element: x/y in the first dword, z/w in the second.
    'BUF_FMT_16_16_16_16_FLOAT': (None, ['U(f16tof32(ldu({0}, a)))', 'U(f16tof32(ldu({0}, a) >> 16))',
                                         'U(f16tof32(ldu({0}, a + 4)))', 'U(f16tof32(ldu({0}, a + 4) >> 16))']),
    'BUF_FMT_32_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', '{0}.Load(a + 12)']),
    'BUF_FMT_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', 'U(1.0)']),
    'BUF_FMT_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '0u', 'U(1.0)']),
    'BUF_FMT_32_FLOAT': (None, ['{0}.Load(a)', '0u', '0u', 'U(1.0)'])}


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


# Binary32 words for inline source encodings 240 through 248, in encoding order.
CNDMASK_INLINE_FLOAT_BITS = (0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
                            0x40000000, 0xc0000000, 0x40800000, 0xc0800000,
                            0x3e22f983)  # 1/(2*pi)


def check_encoding(code, off, n, name, asm):
    if name == 'image_load_mip':
        # The shared disassembler does not expose these gfx10 MIMG flags.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & (1 << 16):
            raise Unsupported('image_load_mip tfe')
        if word1 & (1 << 30):
            raise Unsupported('image_load_mip a16')
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
# Multiply, fused multiply-add (including the literal form), and reciprocal use only
# local arithmetic. Move copies its input bits. None reads a lane ID or another lane.
UNIFORM_VALU = frozenset((
    'v_cvt_u32_f32', 'v_cvt_i32_f32', 'v_cvt_f32_ubyte0', 'v_mul_f32',
    'v_fma_f32', 'v_fmaak_f32', 'v_rcp_f32', 'v_mov_b32',
))


class Stage:
    """Translates one stage. Shares the resource and entry-constant registries of its pipeline."""

    def __init__(self, pipe, kind, code, refusals=None, decoded=None):
        self.pipe, self.kind = pipe, kind
        self.refusals = refusals
        if refusals is None:
            self.ins = decode(strip_header(code))
        else:
            self.ins = decode(strip_header(code), refusals) if decoded is None else decoded
        self.lines = []           # body statements (per block)
        self.sym = {}             # scalar register -> symbolic value
        self.used_s, self.used_v = set(), set()
        self.exec_known = True    # exec still equals the entry mask (the lane is active)
        self.exports = {}         # 'pos0' / 'paramN' / 'mrtN' -> [4 expressions]
        self.attrs = {}           # pixel: attr index -> 'linear' | 'flat'
        self.uses_pos = False
        self.masks = set()        # scalar registers currently holding lane masks
        self.fetched = False      # vertex: the entry call into the fetch routine was translated as loads
        self.setup_entry()
        self.translate()

    # ---- entry state ---------------------------------------------------------------------
    def setup_entry(self):
        root = self.pipe.root
        reg = 8
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
        if self.kind == 'vs':
            self.sym[0] = ('ptr', ('vbtab',))
            self.sym[1] = ('ptrhi', ('vbtab',))
            self.sym[2] = ('ud', self.pipe.ud(('basevertex',)))
            self.sym[3] = ('ud', self.pipe.ud(('startinstance',)))
        else:
            self.sym[reg] = ('const', 0)  # primitive mask: only ever moved to m0

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

    def u(self, tok):
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
                self.used_v.add(r[1])
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

    def u_src(self, tok):
        """uint expression for an integer-context source that the disassembler may print negated."""
        # In an integer instruction the `-` modifier is the source's sign bit flipped: the same bit
        # pattern a float negation produces. Only handlers that model it ask for this; other integer
        # sources keep using u(), so an unexpected negation stays a refusal rather than a silent flip.
        if tok.startswith('-') and not re.match(r'-[\d.]', tok):
            return '(%s ^ 0x80000000u)' % self.u(tok[1:])
        return self.u(tok)

    def f(self, tok):
        """float expression, with the inline neg/abs operand modifiers."""
        neg = False
        if tok.startswith('-') and not re.match(r'-[\d.]', tok):
            neg, tok = True, tok[1:]
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

    def emit(self, text):
        self.lines.append(text)

    def set_v(self, n, expr):
        if hasattr(self, 'uniform_v'):
            self.uniform_v.discard(n)
        self.used_v.add(n)
        self.emit('v%d = %s;' % (n, expr) if self.exec_known else 'if (exec & 1) v%d = %s;' % (n, expr))

    def set_vf(self, n, expr, mods=()):
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
        self.set_v(n, 'U(%s)' % expr)

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
        if self.tracks_uniformity():
            self.scc_uniform = self.scalar_expr_uniform(expr if proof is None else proof)
        self.emit('scc = %s;' % expr)

    def set_s(self, n, expr, sym=None, scc=None, uniform=None):
        """Assign scalar register n at run time and/or symbolically."""
        if self.tracks_uniformity() and (sym is None or sym[0] == 'const'):
            if uniform is None:
                uniform = self.scalar_expr_uniform(expr)
            if uniform:
                self.varying_s.discard(n)
            else:
                self.varying_s.add(n)
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

    # ---- instruction translation -------------------------------------------------------------
    def translate(self):
        targets = set()
        for off, n, name, asm in self.ins:
            if name.startswith('s_cbranch') or name == 's_branch':
                targets.add(int(parse(asm)[1][0].split('_')[1], 16))
        self.blocks = []          # (offset, [lines], terminator)
        self.uniform_v = set()
        self.varying_s = set()
        self.scc_uniform = True
        self.has_branches = bool(targets)
        start = 0
        refusals = getattr(self, 'refusals', None)  # unit probes build a Stage without __init__
        for off, n, name, asm in self.ins:
            if off in targets and off != start:
                self.blocks.append((start, self.lines, None))
                self.lines, start = [], off
            self.pc = off
            try:
                term = self.one(name, asm)
            except Unsupported as reason:
                if refusals is None:
                    raise
                refusals.append((str(reason), self.kind))
                continue
            except Exception as error:
                if refusals is None:
                    raise
                refusals.append((handler_error(name, error, refusals), self.kind))
                continue
            if term:
                self.blocks.append((start, self.lines, term))
                self.lines, start = [], off + n
        if self.lines:
            self.blocks.append((start, self.lines, None))

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
        # Unit probes build a Stage without __init__: without the tracking state, prove nothing.
        if (base in UNIFORM_VALU and ops and self.tracks_uniformity() and
                not getattr(self, 'has_branches', True) and getattr(self, 'exec_known', False) and
                (not mods or (base in VEC_F or base in VEC_U_TO_F) and
                 all(m in ('clamp', 'mul:2', 'mul:4', 'div:2') for m in mods))):
            candidate = reg_range(ops[0])
            if candidate and candidate[0] == 'v' and candidate[1] == candidate[2]:
                sources = ops[1:]
                if base in VEC_F:
                    sources = [re.sub(r'^-(?=[^\d.])', '', tok).strip('|') for tok in sources]
                if sources and all(self.uniform_source(tok) for tok in sources):
                    dst = candidate[1]
        term = self.lower(base, asm, ops, mods)
        if dst is not None:
            self.uniform_v.add(dst)
        return term

    def lower(self, base, asm, ops, mods):
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
        handler = getattr(self, 'op_' + base, None)
        if handler:
            return handler(ops, mods)
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
            return self.set_vf(reg_range(ops[0])[1],
                               text_f.format(*[self.u(o) for o in ops[1:1 + arity]]), mods)
        if base in VEC_U:
            arity, text = VEC_U[base]
            if mods:
                raise Unsupported('%s with %s' % (base, mods))
            if len(ops) != arity + 1:
                raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
            return self.set_v(reg_range(ops[0])[1], text.format(*[self.u(o) for o in ops[1:1 + arity]]))
        if base in SCA:
            return self.scalar(base, ops)
        if base in ('buffer_load_format_x', 'buffer_load_format_xy',
                    'buffer_load_format_xyz', 'buffer_load_format_xyzw'):
            # MUBUF formatted loads take both fields from V# word 3. The startup request carries
            # the root signature layout, while the descriptor table is supplied at draw time.
            raise Unsupported('%s requires descriptor FORMAT and DST_SEL' % base)
        raise Unsupported(base)

    # VOP1/VOP2 with sub-dword addressing: select the source fields, run the same operation, then place
    # the low bits of the result in the destination field. The integer forms reuse VEC_U unchanged;
    # float forms are not lowered yet, and dst_unused UNUSED_SEXT has no lowering.
    def sdwa(self, base, ops, mods):
        fields = {}
        for m in mods:
            key, _, value = m.partition(':')
            if key in ('src0_sel', 'src1_sel', 'src2_sel', 'dst_sel', 'dst_unused') and value:
                fields[key] = value
            else:
                raise Unsupported('%s with SDWA modifier %s' % (base, m))
        if base in VEC_U_TO_F:
            arity, text = VEC_U_TO_F[base]
            float_result = True
        elif base in VEC_U:
            arity, text = VEC_U[base]
            float_result = False
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
            sources.append(sdwa_source(self.u(token), selection, sext))
        result = text.format(*sources)
        if float_result:
            # The table holds the float value; the destination selection writes its bit pattern.
            result = 'U(%s)' % result
        select = fields.get('dst_sel')
        if not select:
            raise Unsupported('%s SDWA without dst_sel' % base)
        # The operation's HLSL carries its own operators, so it has to be grouped before the
        # destination field masks it: "v7 | v8 & 0xffff" would otherwise bind as "v7 | (v8 & 0xffff)".
        if SDWA_SEL[select][1] < 32:
            result = '(%s)' % result
        destination = reg_range(ops[0])[1]
        self.set_v(destination, sdwa_dest(result, select, fields.get('dst_unused', 'UNUSED_PAD'),
                                          'v%d' % destination))

    # scalar ALU with descriptor-word bookkeeping
    def scalar(self, base, ops):
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
        shift_add = base in SHIFT_ADD
        word_complement = base in ('s_nand_b32', 's_nor_b32', 's_xnor_b32')
        word_pack = base == 's_pack_hh_b32_b16'
        if shift_add or word_complement or word_pack:
            # The per-lane mask cannot supply the full word needed by these operations.
            if self.is_lane_mask(dname) or any(self.is_lane_mask(t) for t in srcs):
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
            if scc and self.tracks_uniformity():
                self.scc_uniform = False  # nothing is emitted here, so the run-time scc is stale
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
                    hi = self.sym.get(reg_range(ops[1])[2])
                    if hi and hi[0] in ('desc', 'descnum', 'ptr', 'ptrhi', 'const', 'ud'):
                        self.sym[d + 1] = hi
                    else:
                        self.sym.pop(d + 1, None)
                        if self.tracks_uniformity():
                            self.varying_s.add(d + 1)
            elif own and own[0] in ('desc', 'ptr', 'ptrhi') and any(s == own for s in syms):
                if own[0] == 'desc':
                    self.pipe.modified.add((own[1], own[2]))
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
            self.masks.add(d)
        else:
            self.masks.discard(d)
        if base == 's_mov_b64':
            r = reg_range(ops[1])
            high = r[2] if r and r[1] != r[2] else None
            # Classify the source high word before a destination write can change its status.
            high_source = 's%d' % high if high is not None else srcs[0]
            high_uniform = (self.tracks_uniformity() and high != d and
                            self.uniform_source(high_source))
        self.set_s(d, expr, scc=scc)
        if pair and base != 's_mov_b64':
            self.set_s(d + 1, '0u')
        elif pair:
            self.set_s(d + 1, self.S(high) if high is not None else '0u', uniform=high_uniform)
        return None

    def op_s_and_saveexec_b64(self, ops, mods, text='exec & {0}'):
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's':
            raise Unsupported('saveexec destination %s' % ops[0])
        d = dst[1]
        src = ops[1] if not reg_range(ops[1]) else 's%d' % reg_range(ops[1])[1]
        self.set_s(d, 'exec')
        self.masks.add(d)
        self.exec_known = False
        self.emit('exec = (%s) & 1; scc = exec != 0;' % text.format(self.u(src)))
        if hasattr(self, 'scc_uniform'):
            self.scc_uniform = False

    op_s_and_saveexec_b32 = op_s_and_saveexec_b64

    def op_s_or_saveexec_b64(self, ops, mods):
        self.op_s_and_saveexec_b64(ops, mods, 'exec | {0}')

    op_s_or_saveexec_b32 = op_s_or_saveexec_b64

    def op_s_andn2_saveexec_b64(self, ops, mods):
        self.op_s_and_saveexec_b64(ops, mods, '{0} & ~exec')

    def op_s_andn2_wrexec_b64(self, ops, mods):
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's':
            raise Unsupported('wrexec destination %s' % ops[0])
        d = dst[1]
        src = ops[1] if not reg_range(ops[1]) else 's%d' % reg_range(ops[1])[1]
        self.exec_known = False
        self.emit('exec = (%s & ~exec) & 1; scc = exec != 0;' % self.u(src))
        if hasattr(self, 'scc_uniform'):
            self.scc_uniform = False
        self.set_s(d, 'exec')
        self.masks.add(d)

    op_s_andn2_wrexec_b32 = op_s_andn2_wrexec_b64

    def op_s_cselect_b32(self, ops, mods):
        self.scalar_select(ops, False)

    def op_s_cselect_b64(self, ops, mods):
        self.scalar_select(ops, True)

    def scalar_select(self, ops, pair):
        low = [('s%d' % reg_range(t)[1]) if reg_range(t) and reg_range(t)[0] == 's' else t for t in ops]
        if pair and 'm0' in low:
            raise Unsupported('64-bit scalar operand m0')
        expr = 'scc ? %s : %s' % (self.u(low[1]), self.u(low[2]))
        if low[0] == 'm0':
            self.sym.pop('m0', None)
            self.emit('m0 = %s;' % expr)
            return
        if low[0] in ('exec', 'vcc'):
            if low[0] == 'exec':
                self.exec_known = False
            self.emit('%s = (%s) & 1;' % (low[0], expr))
            return
        d = reg_range(low[0])[1]
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
                self.sym[dst[1] + i] = ('const', 0)
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
            raise Unsupported('buffer load with %d operands' % len(ops))
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

    def tbuffer(self, ops, mods, count):
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

    def op_tbuffer_load_format_x(self, ops, mods):
        self.tbuffer(ops, mods, 1)

    def op_tbuffer_load_format_xy(self, ops, mods):
        self.tbuffer(ops, mods, 2)

    def op_tbuffer_load_format_xyz(self, ops, mods):
        self.tbuffer(ops, mods, 3)

    def op_tbuffer_load_format_xyzw(self, ops, mods):
        self.tbuffer(ops, mods, 4)

    def image(self, ops, mods, how):
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
        lod = None
        if how == 'l':
            # Address dwords are the coordinates and then the explicit level; offsets, bias, clamp and derivatives
            # would add dwords or modifiers this lowering does not read, so they are refused. Only the cache-policy
            # bits (glc, slc, dlc), which never change the sampled value, are accepted next to dim and dmask;
            # unorm, r128, tfe, lwe, a16, d16 and reserved bits would each change the address, resource or result
            # layout and are refused by name.
            for m in mods:
                if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc')):
                    raise Unsupported('image_sample_l modifier %s' % m)
            if dmask == 0:
                raise Unsupported('image_sample_l with an empty dmask')
            if len(cs) != (4 if three else 3):
                raise Unsupported('image_sample_l with %d address registers' % len(cs))
            lod, cs = cs[-1], cs[:-1]
        tex = self.pipe.resource(self.descriptor(ops[2]), kinds[dim])
        if how in ('load', 'load_mip'):
            # image_load_mip addresses x, y, [slice or z] and then the mip, so a 2D array is four registers
            # and Texture2DArray.Load takes int4(x, y, slice, mip) like a volume's int4(x, y, z, mip).
            axes = 3 if three else 2
            packed_2d = how == 'load' and dim == 'SQ_RSRC_IMG_2D' and 'a16' in mods
            allowed = ('unorm', 'glc', 'slc', 'dlc')
            if packed_2d:
                allowed += ('a16',)
            unsupported = any(not m.startswith(('dim:', 'dmask:')) and m not in allowed for m in mods)
            wrong_address = (packed_2d and len(cs) != 1) or (how == 'load_mip' and len(cs) != axes + 1)
            if unsupported or wrong_address or not dmask:
                raise Unsupported('image_%s address or modifiers: %s' % (how, ' '.join(mods)))
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
            if how == 'l':
                call = '%s.SampleLevel(%s, %s, %s)' % (tex, smp, uv, self.f(lod))
            elif how == 'lz' or not self.exec_known or self.kind != 'ps' or len(self.blocks) > 0:
                call = '%s.SampleLevel(%s, %s, 0)' % (tex, smp, uv)
            else:
                call = '%s.Sample(%s, %s)' % (tex, smp, uv)
        self.emit('t = %s;' % call)
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
            pixel_aot.comparison_sampler_state(tuple(words))
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

    def op_image_load(self, ops, mods):
        self.image(ops, mods, 'load')

    def op_image_load_mip(self, ops, mods):
        self.image(ops, mods, 'load_mip')

    def op_image_gather4_lz(self, ops, mods):
        name = 'image_gather4_lz'
        if self.kind != 'ps' or len(ops) != 4:
            raise Unsupported('%s requires a pixel stage and four operands' % name)
        for modifier in mods:
            if not (modifier.startswith(('dim:', 'dmask:')) or modifier in ('dlc', 'glc', 'slc')):
                raise Unsupported('%s modifier %s' % (name, modifier))
        if [m for m in mods if m.startswith('dim:')] != ['dim:SQ_RSRC_IMG_2D']:
            raise Unsupported('%s requires a 2D texture' % name)
        channels = {'dmask:0x1': 'Red', 'dmask:0x2': 'Green', 'dmask:0x4': 'Blue', 'dmask:0x8': 'Alpha'}
        masks = [m for m in mods if m.startswith('dmask:')]
        if len(masks) != 1 or masks[0] not in channels:
            raise Unsupported('%s requires a one-hot dmask' % name)
        dst, addr, sampler = (reg_range(ops[i]) for i in (0, 1, 3))
        if not dst or dst[0] != 'v' or dst[2] - dst[1] != 3 or dst[2] > 255:
            raise Unsupported('%s requires four destination registers' % name)
        if not addr or addr[0] != 'v' or addr[2] - addr[1] != 1 or addr[2] > 255:
            raise Unsupported('%s requires two contiguous address registers' % name)
        words = self.const_words(ops[3]) if sampler and sampler[0] == 's' and sampler[2] - sampler[1] == 3 else None
        if words is None:
            raise Unsupported('%s requires a constant sampler' % name)
        from pixel_aot import check_gather_sampler
        try:
            # Both the guest _lz and D3D Gather select mip zero of the bound view.
            # Validate the sampler for that level even when the view has more mips.
            check_gather_sampler(words, 1, comparison=False)
        except ValueError as error:
            raise Unsupported('%s sampler: %s' % (name, error)) from error
        tex = self.pipe.resource(self.descriptor(ops[2]), 'tex2d')
        smp = self.pipe.sampler(tuple(words))
        u, v = (self.f('v%d' % i) for i in range(addr[1], addr[2] + 1))
        self.emit('t = %s.Gather%s(%s, float2(%s, %s));' % (tex, channels[masks[0]], smp, u, v))
        for i, component in enumerate('xyzw', dst[1]):
            self.set_v(i, 'U(t.%s)' % component)

    def op_v_cvt_i32_f32(self, ops, mods):
        if mods:
            raise Unsupported('v_cvt_i32_f32 with modifiers')
        dst = reg_range(ops[0])[1]
        self.set_v(dst, 'uint(int(F(%s)))' % self.u(ops[1]))

    def float_bits(self, tok):
        """uint bits of a float source, with the inline neg/abs operand modifiers applied."""
        return 'U(%s)' % self.f(tok)

    def op_v_frexp_mant_f32(self, ops, mods):
        # Sign and mantissa in [0.5, 1) of a normal input; zero and denormals read as a signed zero;
        # infinity and NaN pass through unchanged (RDNA 2 ISA, V_FREXP_MANT_F32).
        if len(ops) != 2:
            raise Unsupported('v_frexp_mant_f32 with %d operands, expected 2' % len(ops))
        b = self.float_bits(ops[1])
        e = '((%s >> 23) & 255u)' % b
        expr = 'F(%s == 255u ? %s : %s == 0u ? (%s & 0x80000000u) : ((%s & 0x807fffffu) | 0x3f000000u))' % (
            e, b, e, b, b)
        self.set_vf(reg_range(ops[0])[1], expr, mods)

    def op_v_frexp_exp_i32_f32(self, ops, mods):
        # Exponent e with input = mantissa * 2^e and mantissa in [0.5, 1). Zero, denormal, infinity and NaN give 0.
        if mods:
            raise Unsupported('v_frexp_exp_i32_f32 with modifiers')
        if len(ops) != 2:
            raise Unsupported('v_frexp_exp_i32_f32 with %d operands, expected 2' % len(ops))
        e = '((%s >> 23) & 255u)' % self.float_bits(ops[1])
        self.set_v(reg_range(ops[0])[1], '(%s == 0u || %s == 255u) ? 0u : (%s - 126u)' % (e, e, e))

    def float_to_int(self, base, ops, mods, rounded):
        # Saturating conversion of the rounded float; NaN reads 0. v_cvt_flr rounds toward -inf, v_cvt_rpi
        # rounds floor(x + 0.5). A value at or above 2^31 reads INT_MAX; one below -2^31 reads INT_MIN.
        if mods:
            raise Unsupported('%s with modifiers' % base)
        if len(ops) != 2:
            raise Unsupported('%s with %d operands, expected 2' % (base, len(ops)))
        x = self.f(ops[1])
        r = 'floor(%s)' % (rounded % x)
        self.set_v(reg_range(ops[0])[1], '(%s != %s) ? 0u : (%s >= 2147483648.0) ? 0x7fffffffu : uint(int(max(%s, -2147483648.0)))' % (
            x, x, r, r))

    def op_v_cvt_flr_i32_f32(self, ops, mods):
        self.float_to_int('v_cvt_flr_i32_f32', ops, mods, '%s')

    def op_v_cvt_rpi_i32_f32(self, ops, mods):
        self.float_to_int('v_cvt_rpi_i32_f32', ops, mods, '%s + 0.5')

    def op_v_ldexp_f32(self, ops, mods):
        # D = S0 * 2^S1 with S1 a signed integer, clamped to +-300 (the result is 0 or inf past that anyway).
        # The scale is applied in three parts of at most 2^100 each, so that no factor overflows binary32
        # when the product is finite.
        if len(ops) != 3:
            raise Unsupported('v_ldexp_f32 with %d operands, expected 3' % len(ops))
        n = 'clamp(int(%s), -300, 300)' % self.u(ops[2])
        a = '(%s / 3)' % n
        b = '((%s - %s) / 2)' % (n, a)
        c = '(%s - %s - %s)' % (n, a, b)
        expr = '((%s * exp2(float(%s))) * exp2(float(%s))) * exp2(float(%s))' % (self.f(ops[1]), a, b, c)
        self.set_vf(reg_range(ops[0])[1], expr, mods)

    def op_v_readfirstlane_b32(self, ops, mods):
        src, dst = reg_range(ops[1]), reg_range(ops[0])
        if (mods or not self.tracks_uniformity() or getattr(self, 'has_branches', True) or
                not getattr(self, 'exec_known', False) or not dst or dst[0] != 's' or dst[1] != dst[2] or
                not src or src[0] != 'v' or src[1] != src[2] or src[1] not in self.uniform_v):
            raise Unsupported('v_readfirstlane_b32 requires a proven uniform source')
        self.set_s(reg_range(ops[0])[1], self.u(ops[1]))

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
        body = '{ uint t = v%d; v%d = v%d; v%d = t; }' % (dst, dst, src, src)
        # Uniformity travels with the value; under an unknown exec mask neither register keeps the proof.
        uniform = (dst in self.uniform_v and self.exec_known, src in self.uniform_v and self.exec_known)
        for reg, flag in zip((src, dst), uniform):
            (self.uniform_v.add if flag else self.uniform_v.discard)(reg)
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

    def note_attr(self, tok, how):
        m = re.fullmatch(r'attr(\d+)\.([xyzw])', tok)
        attr = int(m.group(1))
        self.attrs.setdefault(attr, set()).add(how)
        return attr, m.group(2)

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
            return 'f16tof32(%s)' % self.u(tok)
        expr = '%s * %s + %s' % tuple(operand(t, s) for t, s in zip(ops[1:4], sel))
        if 'clamp' in mods:
            expr = 'saturate(%s)' % expr
        d = reg_range(ops[0])[1]
        self.used_v.add(d)
        if high:
            self.set_v(d, '(v%d & 0xffffu) | (f32tof16(%s) << 16)' % (d, expr))
        else:
            self.set_v(d, '(v%d & 0xffff0000u) | f32tof16(%s)' % (d, expr))

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
        self.used_v.update((dst[1], dst[2]))
        self.emit(body if self.exec_known else 'if (exec & 1) ' + body)

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
    def compare_sdwa_sources(self, base, typ, ops, mods, *, float_src0=False):
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
            if sext:
                token = token[len('sext('):-1]
            word = sdwa_source(self.u(token), selection, sext)
            sources.append('int(%s)' % word if typ == 'i32' else word)
        return sources

    def check_compare_sdwa_destination(self, base, writes_exec, ops):
        # v_cmpx_*_sdwa prints just its sources. v_cmp_*_sdwa prints the scalar destination first.
        if len(ops) != (2 if writes_exec else 3):
            raise Unsupported('%s with %d operands' % (base, len(ops)))
        if not writes_exec and ops[0] not in ('vcc', 'vcc_lo', 'exec', 'exec_lo') and not (
                reg_range(ops[0]) and reg_range(ops[0])[0] == 's'):
            raise Unsupported('%s into %s' % (base, ops[0]))

    def compare(self, base, ops, mods=()):
        # A comparison reads its two operands directly, so a modifier that arrives here is one this
        # lowering does not model (op_sel selects a packed half, clamp/omod have no place in a compare):
        # refuse it rather than emit a predicate computed from the unselected operands.
        if mods and not base.endswith('_sdwa'):
            raise Unsupported('%s with %s' % (base, mods))
        # v_cmp_class_f32 / v_cmpx_class_f32 test the binary32 class of the value in SRC0 against the
        # class-bit mask in SRC1 (the ISA order: the float operand first, the integer mask second).
        name = base.removesuffix('_sdwa')
        if name in ('v_cmp_class_f32', 'v_cmpx_class_f32'):
            writes_exec = name == 'v_cmpx_class_f32'
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
            if sdwa:
                value, mask = self.compare_sdwa_sources(base, 'u32', [value, mask], mods, float_src0=True)
            else:
                value, mask = self.f(value), self.u(mask)
            cond = '(class32(%s) & %s) != 0u' % (value, mask)
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
                d = reg_range(dst)[1]
                self.set_s(d, predicate)
                self.masks.add(d)
            return None

        m = re.fullmatch(r'v_cmp(x?)_(\w+?)_(f32|u32|i32)(_sdwa)?', base)
        if not m:
            raise Unsupported(base)
        writes_exec, cond, typ, sdwa = m.group(1) == 'x', m.group(2), m.group(3), bool(m.group(4))
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
        if sdwa and typ == 'f32':
            # The f32 predicates are verified on the plain and e64 forms only (#1920); the SDWA forms
            # stay refused until they are verified too.
            raise Unsupported('%s: f32 SDWA compares are not lowered' % base)
        if sdwa:
            self.check_compare_sdwa_destination(base, writes_exec, ops)
        if len(ops) == 3:
            dst, a, b = ops
        else:
            dst, (a, b) = ('exec' if writes_exec else 'vcc'), ops
        if sdwa:
            x, y = self.compare_sdwa_sources(base, typ, [a, b], mods)
        elif typ == 'f32':
            x, y = self.f(a), self.f(b)
        elif typ == 'i32':
            x, y = 'int(%s)' % self.u(a), 'int(%s)' % self.u(b)
        else:
            x, y = self.u(a), self.u(b)
        def word(tok):
            # A float source's word before its neg/abs modifiers, as f() reads them; they keep a NaN a NaN.
            if tok.startswith('-') and not re.match(r'-[\d.]', tok):
                tok = tok[1:]
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
            d = reg_range(dst)[1]
            self.set_s(d, expr)
            self.masks.add(d)

    def export(self, asm):
        m = re.fullmatch(r'exp (\w+) (.*?)((?: done| compr| vm)*)', asm)
        target, vals, flags = m.group(1), [v.strip() for v in m.group(2).split(',')], m.group(3).split()
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
        if 'compr' in flags:
            if kind in ('uint', 'int'):
                raise Unsupported('packed integer export %s' % name)
            a, b = self.u(vals[0]), self.u(vals[2])
            exprs = ['f16tof32(%s)' % a, 'f16tof32(%s >> 16)' % a, 'f16tof32(%s)' % b, 'f16tof32(%s >> 16)' % b]
            words = ['U(%s)' % e for e in exprs]
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
        self.exports[name] = True
        expr = '%s(%s)' % ('uint4' if kind in ('uint', 'int') else 'float4', ', '.join(exprs))
        self.emit('o.%s = %s;' % (name, 'asint(%s)' % expr if kind == 'int' else expr))
        if name.startswith('param'):
            # The flat copy read by v_interp_mov_f32 carries the register words. A float copy would let
            # FXC fold asfloat() of a constant and flush integer bit patterns that are binary32
            # denormals (#1259). hlsl_vs() drops the line for attributes the pixel stage reads only
            # interpolated.
            self.emit('o.f%s = uint4(%s);' % (name, ', '.join(words)))
        return None

    # ---- output ------------------------------------------------------------------------------
    def body(self):
        """Statements of the whole program as structured HLSL."""
        out = []
        offsets = [b[0] for b in self.blocks]
        index = {off: i for i, off in enumerate(offsets)}
        loops = any(t and t[0] in ('jump', 'branch') and t[-1] <= b[0] for b in self.blocks for t in [b[2]])
        if len(self.blocks) == 1:
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
        lines.append('    uint vcc = 0, exec = 1, m0 = 0, a = 0, w = 0; bool scc = false; float4 t = 0;')
        return lines


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
    'v_or3_b32': (3, '{0} | {1} | {2}'), 'v_and_or_b32': (3, '({0} & {1}) | {2}'), 'v_lshl_or_b32': (3, '({0} << ({1} & 31)) | {2}'),
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
    's_bfe_u32': ('({0} >> ({1} & 31)) & ((1u << (({1} >> 16) & 127)) - 1)', '$ != 0'),
    's_brev_b32': ('reversebits({0})', None),
    's_pack_hh_b32_b16': ('({0} >> 16) | ({1} & 0xffff0000u)', None),
    # s_bfm_b32: a mask of S0[4:0] ones shifted left by S1[4:0]; SCC unchanged.
    's_bfm_b32': ('(((1u << ({0} & 31)) - 1) << ({1} & 31))', None),
    # s_lshlN_add_u32: low 32 bits of (S0 << N) + S1. SHIFT_ADD also supplies SCC for wide overflow.
    's_lshl1_add_u32': ('({0} << 1) + {1}', None), 's_lshl2_add_u32': ('({0} << 2) + {1}', None),
    's_lshl3_add_u32': ('({0} << 3) + {1}', None), 's_lshl4_add_u32': ('({0} << 4) + {1}', None),
}
# shift amount of the scalar shift-and-add forms; their SCC is a carry from the sources, not a rule over the result
SHIFT_ADD = {'s_lshl1_add_u32': 1, 's_lshl2_add_u32': 2, 's_lshl3_add_u32': 3, 's_lshl4_add_u32': 4}
for _name, _op in (('eq', '=='), ('lg', '!='), ('gt', '>'), ('ge', '>='), ('lt', '<'), ('le', '<=')):
    for _typ, _cast in (('i32', 'int({0}) %s int({1})'), ('u32', '{0} %s {1}')):
        def _make(op=_op, cast=_cast):
            def handler(self, ops, mods):
                self.set_scc((cast % op).format(self.u(ops[0]), self.u(ops[1])))
            return handler
        setattr(Stage, 'op_s_cmp_%s_%s' % (_name, _typ), _make())
        setattr(Stage, 'op_s_cmpk_%s_%s' % (_name, _typ), _make())
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


def parse_root(blob):
    if not blob or blob[:4] != b'DXBC':
        raise Unsupported('root signature blob missing')
    n, = struct.unpack_from('<I', blob, 28)
    for o in struct.unpack_from('<%dI' % n, blob, 32):
        if blob[o:o + 4] == b'RTS0':
            d = blob[o + 8:]
            ver, nparam, poff, nsamp, soff, flags = struct.unpack_from('<6I', d, 0)
            params = []
            for i in range(nparam):
                ptype, vis, off = struct.unpack_from('<3I', d, poff + 12 * i)
                p = dict(type=ptype, vis=vis)
                if ptype == 1:
                    p['reg'], p['space'], p['count'] = struct.unpack_from('<3I', d, off)
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


class Pipeline:
    def __init__(self, vs_code, ps_code, root_blob, sampler_heap=None, rtv_formats=(), ps_hw_inputs=None, vs_fetch=None,
                 refusals=None, decoded=None):
        self.root = parse_root(root_blob)
        self.vs_fetch = vs_fetch  # stream_fetch(): the loads the vertex stage's entry call stands for
        self.packed_position_vgpr = None
        if ps_hw_inputs is not None:
            ena, addr = ps_hw_inputs
            if ena != addr or not 0 <= ena <= 0xffff:
                raise Unsupported('unverified pixel input enable/address combination')
            if ena & (1 << 15):
                # SPI PS inputs 0..6 supply IJ pairs (the pull model supplies three);
                # line stipple through sample coverage each occupy one VGPR.
                widths = (2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1)
                self.packed_position_vgpr = sum(w for bit, w in enumerate(widths) if addr & (1 << bit))
        self.rtv_formats = rtv_formats
        self.sampler_heap = os.environ.get('X2W_SAMPLER_HEAP') == '1' if sampler_heap is None else sampler_heap
        self.uses_sampler_heap = False
        self.resources = {}   # key -> dict(index, kind)
        self.samplers = {}    # key -> index
        self.uds = []         # entry-constant sources
        self.notes = set()    # shortcuts taken (written to the manifest as comments)
        self.modified = set()
        self.rect = self.ud(('rect',))
        if refusals is None:
            self.vs = Stage(self, 'vs', vs_code)
            self.ps = Stage(self, 'ps', ps_code)
        else:
            decoded = {} if decoded is None else decoded
            self.vs = Stage(self, 'vs', vs_code, refusals=refusals, decoded=decoded.get('vs'))
            self.ps = Stage(self, 'ps', ps_code, refusals=refusals, decoded=decoded.get('ps'))
        self.params = sorted(int(k[5:]) for k in self.vs.exports if k.startswith('param'))
        self.flat = sorted(a for a, how in self.ps.attrs.items() if 'flat' in how)
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
            if fmt != 42:
                raise Unsupported('unverified UINT RTV format %d' % fmt)
            return 'uint'
        if fmt in SINT_RTVS:
            if fmt != 43:
                raise Unsupported('unverified SINT RTV format %d' % fmt)
            return 'int'
        return 'bits' if fmt in FLOAT32_RTVS else 'float'

    def ud(self, source):
        if source not in self.uds:
            self.uds.append(source)
        return self.uds.index(source)

    def resource(self, key, kind):
        r = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        if r['kind'] != kind:
            raise Unsupported('resource %s used as %s and %s' % (key, r['kind'], kind))
        return 'r%d' % r['index']

    def sampler(self, key, comparison=False):
        if not isinstance(key[0], int):
            raise Unsupported('sampler from a descriptor')
        # A comparison sampler is its own binding even when its words equal an ordinary sampler's.
        key = ('cmp',) + tuple(key) if comparison else key
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
               # The ten binary32 classes v_cmp_class_f32 tests, from the raw encoding (#1795).
               'uint class32(float value) { uint bits = asuint(value), magnitude = bits & 0x7fffffffu;'
               ' bool negative = bits >= 0x80000000u;'
               ' if (magnitude > 0x7f800000u) return (bits & 0x00400000u) != 0u ? 0x2u : 0x1u;'
               ' if (magnitude == 0x7f800000u) return negative ? 0x4u : 0x200u;'
               ' if (magnitude >= 0x00800000u) return negative ? 0x8u : 0x100u;'
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
               'uint ldu(ByteAddressBuffer b, uint a) { uint k = (a & 3) * 8, lo = b.Load(a & ~3u); if (k == 0) return lo; '
               'return (lo >> k) | (b.Load((a & ~3u) + 4) << (32 - k)); }',
               'cbuffer entry : register(b0) { %s };' % ' '.join('uint ud%d;' % i for i in range(len(self.uds)))]
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            out.append('%s r%d : register(t%d);' % ({'buf': 'ByteAddressBuffer', 'tex1d': 'Texture1D<float4>', 'tex2d': 'Texture2D<float4>', 'tex2da': 'Texture2DArray<float4>', 'tex3d': 'Texture3D<float4>'}[r['kind']], r['index'], r['index']))
        for key, i in self.samplers.items():
            out.append('%s q%d : register(s%d);' % ('SamplerComparisonState' if key[0] == 'cmp' else 'SamplerState', i, i))
        if self.uses_sampler_heap:
            out.append('SamplerState qh[%d] : register(s0, space%d);' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE))
            out.append('uint x2w_sampler_entry(uint offset) { return (offset & 15u) != 0 || offset >= %du ? %du : offset >> 4; }'
                       % (16 * SAMPLER_HEAP_FALLBACK, SAMPLER_HEAP_FALLBACK))
        return out

    def interface(self, for_pixel):
        fields = []
        for p in range(max(self.params) + 1 if self.params else 0):
            fields.append('    float4 param%d : PARAM%d;' % (p, p))
        # Attributes read without interpolation get a second, flat copy of the vertex output.
        for p in self.flat:
            fields.append('    nointerpolation uint4 fparam%d : FPARAM%d;' % (p, p))
        fields.append('    float4 pos : SV_Position;')
        return fields

    def hlsl_vs(self):
        s = self.vs
        out = self.common() + ['struct VSOut {'] + self.interface(False) + ['};', 'VSOut body(uint vid) {', '    VSOut o = (VSOut)0;']
        s.used_v.add(0)
        flat = re.compile(r'\s*o\.fparam(\d+) = ')
        body = [line for line in s.body() if not (flat.match(line) and int(flat.match(line).group(1)) not in self.flat)]
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
        out += ['    o.pos = p.pos + r.pos - q.pos;', '    return o;', '}']
        return '\n'.join(out) + '\n'

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
        lines = ['x2w-manifest 1', 'ud %d' % len(self.uds)]
        for i, source in enumerate(self.uds):
            lines.append('udsrc %d %s' % (i, ' '.join(str(x) for x in source)))
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            source, off = key
            where = {'table': 'table %d %d' % (source[1] if len(source) > 1 else 0, off or 0), 'vbtab': 'vb %d' % ((off or 0) // 16),
                     'root': 'root %d' % (source[1] if len(source) > 1 else 0)}.get(source[0])
            if where is None:
                raise Unsupported('resource through %s' % (source,))
            lines.append('srv %d %s %s' % (r['index'], r['kind'], where))
        for key, i in self.samplers.items():
            comparison = key[0] == 'cmp'
            lines.append('%s %d %s' % ('smpc' if comparison else 'smp', i, ' '.join('0x%x' % w for w in key[comparison:])))
        if self.uses_sampler_heap:
            lines.append('smpheap %d %d fallback %d' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE, SAMPLER_HEAP_FALLBACK))
        lines.append('params %d' % (max(self.params) + 1 if self.params else 0))
        lines.append('targets %s' % ' '.join(k[3:] for k in sorted(self.ps.exports) if k.startswith('mrt')))
        return '\n'.join(lines) + '\n'


def fnv(data, h=1469598103934665603):
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def pipeline_key(vs_code, ps_code, root_blob, layout=b''):
    """Hash the startup translation revision and the three blobs in runtime order, then a stream's input layout."""
    key = fnv(root_blob, fnv(ps_code, fnv(vs_code, fnv(b'x2w-startup-shader-6'))))
    return fnv(layout, key) if layout else key


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


def has_bytecode(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def translate_to(out_dir, key, vs_code, ps_code, root_blob, check=True, rtv_formats=(), rtv_request='', vs_fetch=None):
    """Writes <key>.vs.hlsl, <key>.ps.hlsl and <key>.txt (last), or <key>.fail with the reason.

    With check, the pair is compiled to .vs.cso and .ps.cso first and a missing fxc is a failure.
    check=False, or X2W_FXC set to the empty string, publishes the manifest without bytecode."""
    name = os.path.join(out_dir, '%016x' % key)
    try:
        pipe = Pipeline(vs_code, ps_code, root_blob, rtv_formats=rtv_formats, vs_fetch=vs_fetch)
        vs, ps, manifest = pipe.hlsl_vs(), pipe.hlsl_ps(), pipe.manifest()
        if rtv_request:
            manifest += rtv_request + '\n'
    except Unsupported as reason:
        open(name + '.fail', 'w').write(str(reason) + '\n')
        return 'unsupported: %s' % reason
    open(name + '.vs.hlsl', 'w').write(vs)
    open(name + '.ps.hlsl', 'w').write(ps)
    fxc = find_fxc() if check else ''
    if fxc is None:
        message = 'fxc.exe not found (X2W_FXC=%s): set X2W_FXC to its path or install the Windows SDK' % os.environ.get('X2W_FXC', '<unset>')
        open(name + '.fail', 'w').write(message + '\n')
        return message
    manifest_path = name + '.txt'
    write_manifest = True
    if fxc:
        import subprocess
        # An existing manifest can already point at these paths. Publish only bytecode from a successful pair.
        with tempfile.TemporaryDirectory(prefix='.fxc-', dir=out_dir) as staging:
            for stage, profile in (('vs', 'vs_5_1'), ('ps', 'ps_5_1')):
                bytecode = os.path.join(staging, stage + '.cso')
                r = subprocess.run([fxc, '/nologo', '/T', profile, '/E', 'main', '/Fo', bytecode, name + '.%s.hlsl' % stage],
                                   capture_output=True, text=True)
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
    fetch = None
    if a.startswith(b'DXBC') or b.startswith(b'DXBC'):
        # Slot 86 records complete containers in canonical (VS, PS) order.
        # Keep the container bytes in the cache key; translate only the bounded native code.
        import zlib
        from pixel_aot import container_code
        from resource_inventory import parts
        try:
            container = a
            a, b = (container_code(parts(blob)[b'DXIL'], stage=stage)
                    for blob, stage in ((a, 1), (b, 0)))
            if starts_with_fetch_call(a):
                fetch = stream_fetch(container, layout.decode('ascii') if layout else None)
        except (KeyError, ValueError, IndexError, struct.error, zlib.error) as error:
            raise Unsupported('invalid stream shader: %s' % error) from error
        if classify_stage(a) != 'vs' or classify_stage(b) != 'ps':
            raise Unsupported('stream stage and shader export disagree')
        kinds = {'vs': a, 'ps': b}
    else:
        kinds = {classify_stage(a): a, classify_stage(b): b}
    return kinds, root, formats, request, fetch, decoded


def translate_request(out_dir, key):
    """Translate a base shader request or an RTV variant referencing its source blobs."""
    try:
        kinds, root, formats, request, fetch, _ = load_request(out_dir, key)
    except (Unsupported, OSError, UnicodeError) as reason:
        open(os.path.join(out_dir, '%016x.fail' % key), 'w').write(str(reason) + '\n')
        return 'unsupported: %s' % reason
    if set(kinds) != {'vs', 'ps'}:
        open(os.path.join(out_dir, '%016x.fail' % key), 'w').write('stages %s\n' % sorted(kinds))
        return 'stages %s' % sorted(kinds)
    return translate_to(out_dir, key, kinds['vs'], kinds['ps'], root, rtv_formats=formats, rtv_request=request, vs_fetch=fetch)


REFUSALS_CAVEAT = ('Counts after the first refusal are approximate. Skipped instructions leave later state '
                   'unmodelled and can cause follow-on errors. A pipeline-scope refusal ends the pipeline, '
                   'so Only counts at pipeline scope reflect a lower bound on the reasons reached.')


def request_keys(out_dir):
    for name in sorted(os.listdir(out_dir)):
        match = re.fullmatch(r'req_([0-9a-f]{16})\.(?:a|rtv)', name)
        if match:
            yield int(match.group(1), 16)


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
        kinds, root, formats, _, fetch, decoded = load_request(out_dir, key, refusals, pair_info)
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
                        refusals=refusals, decoded=decoded)
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
                caveat=REFUSALS_CAVEAT)


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
    manifest_only = os.environ.get('X2W_FXC') == ''
    for key in request_keys(out_dir):
        name = os.path.join(out_dir, '%016x' % key)
        if os.path.isfile(name + '.txt') and (
            manifest_only or (has_bytecode(name + '.vs.cso') and has_bytecode(name + '.ps.cso'))
        ):
            continue
        print('%016x: %s' % (key, translate_request(out_dir, key)))


if __name__ == '__main__':
    main()
