"""Four-channel half-float typed buffer loads in tools/startup/gfx2hlsl.py (issue #1695).

usage: startup_gfx2hlsl_tbuffer_16f_check.py

`BUF_FMT_16_16_16_16_FLOAT` stores four binary16 channels in one 8-byte element: x and y in the first dword,
z and w in the second. The format table had the two-channel form only, so a typed load of this format was
refused with `buffer format BUF_FMT_16_16_16_16_FLOAT`; that refusal covered 5 of the 29 refused shaders in the
first main-game loading-screen cache (#870).

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, the MTBUF branch): no game-derived bytes are used. Each case encodes the instruction,
decodes it, and only then translates it, so the test exercises the same path the pipeline uses.
"""
import hashlib
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
MTBUF_FORMAT_16_16_16_16_FLOAT = 71
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


# 1. The encoder produces the instruction the decoder reads; a layout change shows up here first.
code = encode_mtbuf(MTBUF_LOAD['xyz'], MTBUF_FORMAT_16_16_16_16_FLOAT, vdata=1, vaddr=2, srsrc=4)
asm, emitted = translate_bytes(code)
check('encodes and decodes a typed xyz load',
      asm == ['tbuffer_load_format_xyz v[1:3], v2, s[4:7], 0 format:[BUF_FMT_16_16_16_16_FLOAT] idxen'],
      repr(asm))

# 2. x and y come from the first dword, z from the second, all converted from binary16.
check('x from the first dword', 'v1 = U(f16tof32(ldu(r0, a)))' in emitted, emitted)
check('y from the high half of the first dword', 'v2 = U(f16tof32(ldu(r0, a) >> 16))' in emitted, emitted)
check('z from the second dword', 'v3 = U(f16tof32(ldu(r0, a + 4)))' in emitted, emitted)

# 3. The four-channel form also lowers, and w comes from the high half of the second dword.
code = encode_mtbuf(MTBUF_LOAD['xyzw'], MTBUF_FORMAT_16_16_16_16_FLOAT, vdata=1, vaddr=2, srsrc=4)
asm, emitted = translate_bytes(code)
check('encodes and decodes the xyzw form', asm[0].startswith('tbuffer_load_format_xyzw '), repr(asm))
check('w from the high half of the second dword', 'v4 = U(f16tof32(ldu(r0, a + 4) >> 16))' in emitted, emitted)

# 4. The one-channel form lowers and reads only the first half.
code = encode_mtbuf(MTBUF_LOAD['x'], MTBUF_FORMAT_16_16_16_16_FLOAT, vdata=1, vaddr=2, srsrc=4)
asm, emitted = translate_bytes(code)
check('encodes and decodes the x form', asm[0].startswith('tbuffer_load_format_x '), repr(asm))
check('x form reads one channel', 'v1 = U(f16tof32(ldu(r0, a)))' in emitted and 'v2 =' not in emitted, emitted)


# 5. Refusals that must not widen: a format the table does not name, and the d16 typed-load form.
check('an unnamed format stays refused',
      refused(encode_mtbuf(MTBUF_LOAD['xyz'], MTBUF_FORMAT_UNNAMED, vdata=1, vaddr=2, srsrc=4)))
check('the d16 form stays refused',
      refused(encode_mtbuf(MTBUF_LOAD_D16['xyz'], MTBUF_FORMAT_16_16_16_16_FLOAT, vdata=1, vaddr=2, srsrc=4)))

# 6. The default-format instruction (format 1, which the disassembler omits) still translates.
code = encode_mtbuf(MTBUF_LOAD['xyz'], MTBUF_FORMAT_8_UNORM, vdata=1, vaddr=2, srsrc=4)
asm, emitted = translate_bytes(code)
check('default format still translates', 'format:[' not in asm[0] and 'v1 =' in emitted, repr(asm))


def check_formatted_heap():
    from startup_compute_dispatch_heap_check import (
        FORMATTED_ASM, FORMATTED_BINDINGS, FORMATTED_CODE, FORMATTED_GROUP, FORMATTED_RSRC2,
        FORMATTED_INDEXED_ASM, FORMATTED_INDEXED_BINDINGS, FORMATTED_INDEXED_CODE,
    )
    from startup_gfx2hlsl_compute_check import assemble, refused, translate
    from startup_gfx2hlsl_heap_record_check import root_blob

    if len(sys.argv) == 3 and sys.argv[1] == '--llvm-mc':
        assert assemble(sys.argv[2], FORMATTED_ASM) == FORMATTED_CODE
        assert assemble(sys.argv[2], FORMATTED_INDEXED_ASM) == FORMATTED_INDEXED_CODE

    def pipeline(code):
        return G.ComputePipeline(code, root_blob(table_range=True), FORMATTED_RSRC2, FORMATTED_GROUP)

    program = pipeline(FORMATTED_CODE)
    assert program.manifest().endswith(FORMATTED_BINDINGS), program.manifest()
    indexed = pipeline(FORMATTED_INDEXED_CODE).manifest()
    assert indexed.endswith(FORMATTED_INDEXED_BINDINGS), indexed
    # Replacing word 3 with zero after the load must invalidate the heap proof.
    load_at = FORMATTED_CODE.index(bytes.fromhex('00200ce0000c0880'))
    rewritten = FORMATTED_CODE[:load_at] + bytes.fromhex('8003a3be') + FORMATTED_CODE[load_at:]  # s_mov_b32 s35, 0
    refused('words were rewritten', lambda: pipeline(rewritten).manifest())

    stage = Stage()
    stage.sym[4] = ('ptr', ('root', 0))
    refused('hardware table descriptor (source: root)',
            lambda: stage.op_buffer_load_format_xyzw(['v[0:3]', 'off', 's[4:7]', '0'], []))

    # The pinned table is the legacy format set (plus the prelude helpers main added, e.g. ldexp32); extended formats (default on) add cases.
    os.environ['X2W_EXTENDED_BUFFER_FORMATS'] = '0'
    try:
        table = translate('L-uint').hlsl_cs().encode()
    finally:
        del os.environ['X2W_EXTENDED_BUFFER_FORMATS']
    assert hashlib.sha256(table).hexdigest() == 'caf208eeba14c16771ec99e4fbf30d091614d9d82550ed681f32c66ab17a4c30'
    print('formatted heap manifest, rewritten-word refusal and byte-identical table HLSL: ok')


check_formatted_heap()

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')