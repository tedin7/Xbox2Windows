"""Two-channel signed-normalized typed buffer loads in tools/startup/gfx2hlsl.py (issue #1723).

usage: startup_gfx2hlsl_tbuffer_16_16_snorm_check.py

`BUF_FMT_16_16_SNORM` stores two signed 16-bit channels in one dword. The format table did not name it, so a typed
load of this format was refused with `buffer format BUF_FMT_16_16_SNORM`.

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, the MTBUF branch): no game-derived bytes are used. Each case encodes the instruction,
decodes it, and only then translates it, so the test exercises the same path the pipeline uses.
"""
import os
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


# --- synthetic MTBUF encodings -------------------------------------------------------------------------
# Layout taken from tools/disasm_rdna2.py:
#   w0: [31:26] = 0b111010 (MTBUF), [25:19] format, [18:16] opcode low bits, [15] dlc, [14] glc, [13] idxen,
#       [12] offen, [11:0] immediate offset
#   w1: [7:0] vaddr, [15:8] vdata, [20:16] srsrc/4, [21] opcode high bit, [22] slc, [23] tfe, [31:24] soffset
MTBUF_LOAD = {'x': 0, 'xy': 1, 'xyz': 2, 'xyzw': 3}
MTBUF_LOAD_D16 = {'x': 8, 'xy': 9, 'xyz': 10, 'xyzw': 11}
MTBUF_FORMAT_16_16_SNORM = 24
MTBUF_FORMAT_8_UNORM = 1          # the format the disassembler treats as its default and omits
MTBUF_FORMAT_UNNAMED = 0x7f       # not in the disassembler's format table


def encode_mtbuf(op, fmt, *, vdata=0, vaddr=0, srsrc=4, soffset=128, idxen=True, offen=False, imm=0,
                 glc=0, dlc=0, tfe=0, slc=0):
    """Encode one MTBUF instruction; returns the 8 bytes the decoder reads."""
    w0 = (0b111010 << 26) | ((fmt & 0x7f) << 19) | ((op & 0x7) << 16)
    w0 |= (dlc & 1) << 15 | (glc & 1) << 14 | (idxen & 1) << 13 | (offen & 1) << 12 | (imm & 0xfff)
    w1 = ((vaddr & 0xff) | ((vdata & 0xff) << 8) | (((srsrc // 4) & 0x1f) << 16) | (((op >> 3) & 1) << 21)
          | ((slc & 1) << 22) | ((tfe & 1) << 23) | ((soffset & 0xff) << 24))
    return struct.pack('<II', w0, w1)


class Pipeline:
    """Just enough pipeline state for a typed buffer load."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()

    def resource(self, key, kind):
        entry = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        return 'r%d' % entry['index']

    def ud(self, source):
        return 0


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()

    def descriptor(self, tok):
        return ('buffer', tok)


def translate_bytes(code):
    """Decode the bytes the way the pipeline does, then translate every decoded instruction."""
    decoded = G.decode(code)
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines)


def refused(code):
    try:
        translate_bytes(code)
    except G.Unsupported:
        return True
    return False


X_EXPR = 'U(max(float(asint(w << 16) >> 16) / 32767.0, -1.0))'
Y_EXPR = 'U(max(float(asint(w) >> 16) / 32767.0, -1.0))'

# 1. A typed xy load of the format decodes, translates, and reads one dword.
code = encode_mtbuf(MTBUF_LOAD['xy'], MTBUF_FORMAT_16_16_SNORM, vdata=1, vaddr=2, srsrc=4)
asm, emitted = translate_bytes(code)
check('encodes and decodes a typed xy load',
      asm == ['tbuffer_load_format_xy v[1:2], v2, s[4:7], 0 format:[BUF_FMT_16_16_SNORM] idxen'], repr(asm))
check('one unaligned dword load', 'w = ldu(r0, a);' in emitted, emitted)
check('x sign-extends the low half and clamps at -1.0', 'v1 = ' + X_EXPR in emitted, emitted)
check('y sign-extends the high half and clamps at -1.0', 'v2 = ' + Y_EXPR in emitted, emitted)

# 2. The one-channel form reads only x.
code = encode_mtbuf(MTBUF_LOAD['x'], MTBUF_FORMAT_16_16_SNORM, vdata=1, vaddr=2, srsrc=4)
asm, emitted = translate_bytes(code)
check('x form reads one channel', 'v1 = ' + X_EXPR in emitted and 'v2 =' not in emitted, emitted)

# 3. The four-channel form fills the missing channels as 0, 0, 1.0.
code = encode_mtbuf(MTBUF_LOAD['xyzw'], MTBUF_FORMAT_16_16_SNORM, vdata=1, vaddr=2, srsrc=4)
asm, emitted = translate_bytes(code)
check('missing z is 0', 'v3 = 0u' in emitted, emitted)
check('missing w is 1.0', 'v4 = U(1.0)' in emitted, emitted)

# 4. Refusals that must not widen: a format the table does not name, and the d16 typed-load form.
check('an unnamed format stays refused',
      refused(encode_mtbuf(MTBUF_LOAD['xy'], MTBUF_FORMAT_UNNAMED, vdata=1, vaddr=2, srsrc=4)))
check('the d16 form stays refused',
      refused(encode_mtbuf(MTBUF_LOAD_D16['xy'], MTBUF_FORMAT_16_16_SNORM, vdata=1, vaddr=2, srsrc=4)))

# 5. The default-format instruction (format 1, which the disassembler omits) still translates.
code = encode_mtbuf(MTBUF_LOAD['xy'], MTBUF_FORMAT_8_UNORM, vdata=1, vaddr=2, srsrc=4)
asm, emitted = translate_bytes(code)
check('default format still translates', 'format:[' not in asm[0] and 'v1 =' in emitted, repr(asm))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
