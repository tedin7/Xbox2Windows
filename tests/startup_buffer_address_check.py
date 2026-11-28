"""MUBUF and MTBUF buffer addressing and modifier checks in tools/startup/gfx2hlsl.py.

usage: startup_buffer_address_check.py [controls]
       startup_buffer_address_check.py <dxc> <offscreen_check> <warp|hardware> [fxc]

In RDNA 2 MUBUF and MTBUF loads, idxen and offen can be set together: vaddr is then a two-VGPR
pair v[n:n+1] holding (vindex, voffset), and the byte address is
vindex * stride + voffset + soffset + inst_offset. Stage.buffer_address() previously used an
idxen / elif offen branch, which read v[n] as the index and silently dropped v[n+1]. It also
ignored tfe, raised IndexError on lds (whose assembly omits vdata), and raised ValueError when
soffset named an SGPR or non-integer operand instead of an integer literal.

All instruction bytes below are constructed from the RDNA 2 / tools/disasm_rdna2.py bit layouts.
With no arguments (or 'controls'), this script checks decoding, generated HLSL, refusals and an
unguarded control without a GPU. Given DXC and offscreen_check, it runs three source-built vertex
shaders through the compute readback wrapper on WARP or hardware (plus FXC vs_5_1/ps_5_1 compile
verification when FXC is supplied) and checks that mutations and wrong references fail readback.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402
from triage_run import classify_shader_reason  # noqa: E402

SENTINEL = 0xc5c5c5c5
# offscreen_check --compute-store-case fills the raw t0 buffer with this 320-dword pattern.
DATA = b''.join(struct.pack('<I', (0x8b02a7f1 + i * 0x413b1) & 0xffffffff) for i in range(320))

MUBUF_LOAD_DWORD = 0x0c
MUBUF_LOAD_DWORDX4 = 0x0e
MTBUF_LOAD_XYZW = 0x03
BUF_FMT_8_8_8_8_UNORM = 56


def encode_mubuf(op=MUBUF_LOAD_DWORD, *, imm=0, offen=0, idxen=0, glc=0, dlc=0, lds=0,
                 vaddr=0, vdata=4, srsrc=16, slc=0, tfe=0, soffset=128):
    """Construct an 8-byte RDNA 2 MUBUF instruction from its hardware fields."""
    w0 = ((0b111000 << 26) | ((op & 0xff) << 18) | ((lds & 1) << 16) | ((dlc & 1) << 15)
          | ((glc & 1) << 14) | ((idxen & 1) << 13) | ((offen & 1) << 12) | (imm & 0xfff))
    w1 = (((soffset & 0xff) << 24) | ((tfe & 1) << 23) | ((slc & 1) << 22)
          | (((srsrc // 4) & 0x1f) << 16) | ((vdata & 0xff) << 8) | (vaddr & 0xff))
    return struct.pack('<II', w0, w1)


def encode_mtbuf(op=MTBUF_LOAD_XYZW, fmt=BUF_FMT_8_8_8_8_UNORM, *, imm=0, offen=0, idxen=0,
                 glc=0, dlc=0, vaddr=0, vdata=4, srsrc=16, slc=0, tfe=0, soffset=128):
    """Construct an 8-byte RDNA 2 MTBUF instruction from its hardware fields."""
    w0 = ((0b111010 << 26) | ((fmt & 0x7f) << 19) | ((op & 0x7) << 16) | ((dlc & 1) << 15)
          | ((glc & 1) << 14) | ((idxen & 1) << 13) | ((offen & 1) << 12) | (imm & 0xfff))
    w1 = (((soffset & 0xff) << 24) | ((tfe & 1) << 23) | ((slc & 1) << 22)
          | (((op >> 3) & 1) << 21) | (((srsrc // 4) & 0x1f) << 16)
          | ((vdata & 0xff) << 8) | (vaddr & 0xff))
    return struct.pack('<II', w0, w1)


def encode_vop2(op, vdst, src0, vsrc1):
    """Construct a 4-byte RDNA 2 VOP2 instruction."""
    w0 = ((op & 0x3f) << 25) | ((vdst & 0xff) << 17) | ((vsrc1 & 0xff) << 9) | (src0 & 0x1ff)
    return struct.pack('<I', w0)


def encode_vop1(op, vdst, src0):
    """Construct a 4-byte RDNA 2 VOP1 instruction."""
    w0 = 0x7e000000 | ((vdst & 0xff) << 17) | ((op & 0xff) << 9) | (src0 & 0x1ff)
    return struct.pack('<I', w0)


def root_with_one_constant():
    """Minimal DXBC RTS0 container declaring one 32-bit root constant (placed at s8)."""
    header = struct.pack('<6I', 2, 1, 24, 0, 0, 0)
    param = struct.pack('<3I', 1, 0, 36)
    const = struct.pack('<3I', 0, 0, 1)
    rts0 = header + param + const
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return b'DXBC' + bytes(16) + struct.pack('<III', 1, 36 + len(part), 1) + struct.pack('<I', 36) + part


class DummyPipeline:
    """Minimal Pipeline stand-in for isolated Stage instruction checks."""

    def __init__(self):
        self.resources = {}
        self.uds = [('rect',)]
        self.notes = set()
        self.modified = set()
        self.sampler_heap = False

    def ud(self, source):
        if source not in self.uds:
            self.uds.append(source)
        return self.uds.index(source)

    def resource(self, key, kind):
        entry = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        return 'r%d' % entry['index']


class ProbeStage(G.Stage):
    """Isolated Stage with a descriptor at s[16:19], a root constant at s8 and a constant at s5."""

    def __init__(self):
        self.pipe = DummyPipeline()
        self.kind = 'vs'
        self.lines = []
        self.sym = {
            5: ('const', 0x20),
            8: ('ud', self.pipe.ud(('rootconst', 0, 0))),
            **{16 + i: ('desc', ('vbtab',), 16, i) for i in range(4)},
        }
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports, self.attrs, self.masks = {}, {}, set()


def translate_bytes(code, *, modify_desc=False):
    decoded = G.decode(code)
    stage = ProbeStage()
    if modify_desc:
        stage.pipe.modified.add((('vbtab',), 16))
    for _off, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], stage


def translate_asm(asm, *, modify_desc=False):
    name = asm.split(' ', 1)[0]
    stage = ProbeStage()
    if modify_desc:
        stage.pipe.modified.add((('vbtab',), 16))
    stage.one(name, asm)
    return stage


def must_refuse_bytes(code, expected_substring='', *, modify_desc=False):
    try:
        translate_bytes(code, modify_desc=modify_desc)
    except G.Unsupported as err:
        if expected_substring and expected_substring not in str(err):
            raise AssertionError(f'expected {expected_substring!r} in Unsupported({str(err)!r})')
        return str(err)
    raise AssertionError('expected Unsupported, but translation succeeded')


def must_refuse_asm(asm, expected_substring='', *, modify_desc=False):
    try:
        translate_asm(asm, modify_desc=modify_desc)
    except G.Unsupported as err:
        if expected_substring and expected_substring not in str(err):
            raise AssertionError(f'expected {expected_substring!r} in Unsupported({str(err)!r})')
        return str(err)
    raise AssertionError(f'expected Unsupported for {asm!r}, but translation succeeded')


def old_buffer_address(stage, ops, mods):
    """Pre-fix Stage.buffer_address implementation used as a negative control."""
    key = stage.descriptor(ops[2])
    buf = stage.pipe.resource(key, 'buf')
    offset = int(ops[3], 0)
    for m in mods:
        if m.startswith('offset:'):
            offset += int(m[7:], 0)
    if 'idxen' in mods:
        if key in stage.pipe.modified:
            raise G.Unsupported('indexed load through a modified descriptor')
        stride = stage.pipe.ud(('stride',) + (stage.pipe.resources[key]['index'],))
        address = '%s * ud%d + %d' % (stage.u(ops[1]), stride, offset)
    elif 'offen' in mods:
        address = '%s + %d' % (stage.u(ops[1]), offset)
    else:
        address = '%d' % offset
    return buf, address


def formatted_mubuf_request_controls():
    # One SRV descriptor table at s[8:9]. The shader loads its second 16-byte entry.
    rts0 = b''.join((
        struct.pack('<6I', 2, 1, 24, 0, 0, 0),  # Root signature header
        struct.pack('<3I', 0, 0, 36),            # Descriptor-table parameter
        struct.pack('<2I', 1, 44),               # Descriptor table
        struct.pack('<6I', 0, 2, 0, 0, 0, 0),    # SRV range with two entries
    ))
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    root = b'DXBC' + bytes(16) + struct.pack('<4I', 1, 36 + len(part), 1, 36) + part
    load_descriptor = struct.pack('<3I', 0xf4080804, 0xfa000010, 0xbf8cc07f)  # s_load_dwordx4; lgkmcnt(0)
    export_position = struct.pack('<4I', 0xbf8c3f70, 0xf80008cf, 0x06060606, 0xbf810000)  # vmcnt(0); exp; end
    ps = struct.pack('<3I', 0xf800180f, 0, 0xbf810000)

    with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'X2W_FXC': ''}):
        folder = Path(temp)

        def request(load):
            vs = load_descriptor + load + export_position
            key = G.pipeline_key(vs, ps, root)
            for suffix, data in (('a', vs), ('b', ps), ('root', root)):
                (folder / f'req_{key:016x}.{suffix}').write_bytes(data)
            return G.translate_request(temp, key), folder / f'{key:016x}'

        # With the same descriptor and address, adjacent raw and MTBUF loads can publish a translation.
        for load, hlsl in (
            (encode_mubuf(0x0f, vdata=6, srsrc=32), 'v8 = r0.Load(a + 8);'),
            (encode_mtbuf(0x02, vdata=6, srsrc=32), 'v8 = U(((w >> 16) & 255) / 255.0);'),
        ):
            result, name = request(load)
            assert result == 'ok', result
            assert 'srv 0 buf table 0 16\n' in name.with_suffix('.txt').read_text()
            assert hlsl in name.with_suffix('.vs.hlsl').read_text(), hlsl

        formats = (('x', 'v6'), ('xy', 'v[6:7]'), ('xyz', 'v[6:8]'), ('xyzw', 'v[6:9]'))
        for op, (suffix, dest) in enumerate(formats):
            load = encode_mubuf(op, vdata=6, srsrc=32)
            mnemonic = 'buffer_load_format_' + suffix
            assert G.decode(load) == [(0, 8, mnemonic, f'{mnemonic} {dest}, off, s[32:35], 0')]
            result, name = request(load)
            # All four forms take FORMAT and DST_SEL from descriptor word 3 at draw time, one register per component.
            assert result == 'ok', result
            source = name.with_suffix('.vs.hlsl').read_text()
            assert 'switch ((ud' in source
            written = {int(n) for n in re.findall(r'^\s*v(\d+) = ', source, re.M)} & set(range(6, 10))
            assert written == set(range(6, 7 + op)), (mnemonic, written)
        # The old refusal text is still classified for caches written by earlier translators.
        assert classify_shader_reason('buffer_load_format_xyzw requires descriptor FORMAT and DST_SEL') == 'buffer_* instruction'


def controls():
    formatted_mubuf_request_controls()

    # 1. Existing single-mode forms keep their exact HLSL output.
    asm, st = translate_bytes(encode_mubuf(imm=12, soffset=128 + 16))
    assert asm == ['buffer_load_dword v4, off, s[16:19], 16 offset:12'], asm
    assert st.lines == ['a = 28;', 'v4 = r0.Load(a + 0);'], st.lines

    asm, st = translate_bytes(encode_mubuf(imm=8, idxen=1, vaddr=2))
    assert asm == ['buffer_load_dword v4, v2, s[16:19], 0 idxen offset:8'], asm
    assert st.lines == ['a = v2 * ud2 + 8;', 'v4 = r0.Load(a + 0);'], st.lines
    assert st.used_v == {2, 4}, st.used_v

    asm, st = translate_bytes(encode_mubuf(imm=8, offen=1, vaddr=3))
    assert asm == ['buffer_load_dword v4, v3, s[16:19], 0 offen offset:8'], asm
    assert st.lines == ['a = v3 + 8;', 'v4 = r0.Load(a + 0);'], st.lines
    assert st.used_v == {3, 4}, st.used_v

    # 2. Combined idxen + offen reads v[n] as index and v[n+1] as byte offset on both MUBUF and MTBUF.
    asm, st = translate_bytes(encode_mubuf(MUBUF_LOAD_DWORDX4, imm=12, idxen=1, offen=1, vaddr=2))
    assert asm == ['buffer_load_dwordx4 v[4:7], v[2:3], s[16:19], 0 idxen offen offset:12'], asm
    assert st.lines[0] == 'a = v2 * ud2 + v3 + 12;', st.lines
    assert {2, 3}.issubset(st.used_v), st.used_v

    asm, st = translate_bytes(encode_mtbuf(MTBUF_LOAD_XYZW, BUF_FMT_8_8_8_8_UNORM, imm=5,
                                           idxen=1, offen=1, vaddr=6, vdata=8))
    assert asm == ['tbuffer_load_format_xyzw v[8:11], v[6:7], s[16:19], 0 '
                   'format:[BUF_FMT_8_8_8_8_UNORM] idxen offen offset:5'], asm
    assert st.lines[0] == 'a = v6 * ud2 + v7 + 5;', st.lines
    assert {6, 7}.issubset(st.used_v), st.used_v

    # 3. Scalar soffset from an SGPR (root constant, folded scalar constant, or runtime SGPR, up to s105).
    asm, st = translate_bytes(encode_mubuf(imm=12, idxen=1, offen=1, vaddr=2, soffset=8))
    assert asm == ['buffer_load_dword v4, v[2:3], s[16:19], s8 idxen offen offset:12'], asm
    assert st.lines[0] == 'a = v2 * ud2 + v3 + ud1 + 12;', st.lines

    asm, st = translate_bytes(encode_mubuf(imm=4, offen=1, vaddr=2, soffset=5))
    assert st.lines[0] == 'a = v2 + 0x00000020u + 4;', st.lines

    asm, st = translate_bytes(encode_mubuf(imm=4, offen=1, vaddr=2, soffset=105))
    assert st.lines[0] == 'a = v2 + s105 + 4;' and 105 in st.used_s, (st.lines, st.used_s)

    # 4. Cache-policy flags (glc, slc, dlc) are accepted without changing the address expression.
    _, st = translate_bytes(encode_mubuf(imm=8, idxen=1, offen=1, vaddr=2, glc=1, slc=1, dlc=1))
    assert st.lines[0] == 'a = v2 * ud2 + v3 + 8;', st.lines
    _, st = translate_bytes(encode_mtbuf(imm=8, idxen=1, offen=1, vaddr=2, glc=1, slc=1, dlc=1))
    assert st.lines[0] == 'a = v2 * ud2 + v3 + 8;', st.lines

    # 5. Highest valid 2-VGPR pair v[254:255] translates; vaddr=255 (v[255:256]) is refused.
    _, st = translate_bytes(encode_mubuf(imm=4, idxen=1, offen=1, vaddr=254))
    assert st.lines[0] == 'a = v254 * ud2 + v255 + 4;', st.lines
    must_refuse_bytes(encode_mubuf(imm=4, idxen=1, offen=1, vaddr=255), 'v[255:256]')

    # 6. Unmodeled modifiers (lds, tfe, unknown) are refused with Unsupported, never dropped or crashed.
    must_refuse_bytes(encode_mubuf(imm=8, offen=1, vaddr=2, lds=1), 'buffer modifier lds')
    must_refuse_bytes(encode_mubuf(imm=8, lds=1), 'buffer modifier lds')
    must_refuse_bytes(encode_mubuf(imm=8, offen=1, vaddr=2, tfe=1), 'buffer modifier tfe')
    must_refuse_bytes(encode_mtbuf(imm=8, offen=1, vaddr=2, tfe=1), 'buffer modifier tfe')
    must_refuse_asm('buffer_load_dword v4, off, s[16:19], 0 offset:-4', 'buffer modifier offset:-4')
    must_refuse_asm('buffer_load_dword v4, off, s[16:19], 0 unknown_mod', 'buffer modifier unknown_mod')

    # 7. Unmodeled or invalid soffset encodings raise Unsupported rather than ValueError.
    for soffset, label in (
        (106, 'vcc_lo'),     # just above s105
        (124, 'm0'),
        (125, 'null'),
        (193, '-1'),         # negative inline integer low edge
        (208, '-16'),        # negative inline integer high edge
        (209, 's209'),       # reserved scalar source code printed as s209
        (240, '0.5'),        # inline float low edge
        (248, '0.15915494'), # inline float high edge
    ):
        must_refuse_bytes(encode_mubuf(imm=8, offen=1, vaddr=2, soffset=soffset), f'buffer soffset {label}')
    # Descriptor-holding SGPR in soffset is refused by Stage.S().
    must_refuse_bytes(encode_mubuf(imm=8, offen=1, vaddr=2, soffset=16), '16')

    # 8. Modified descriptor on combined idxen + offen and mismatched vaddr shapes are refused.
    must_refuse_bytes(encode_mubuf(imm=8, idxen=1, offen=1, vaddr=2),
                      'indexed load through a modified descriptor', modify_desc=True)
    must_refuse_asm('buffer_load_dword v4, v2, s[16:19], 0 idxen offen', 'for idxen offen')
    must_refuse_asm('buffer_load_dword v4, v[2:3], s[16:19], 0 idxen', 'for idxen')
    must_refuse_asm('buffer_load_dword v4, v[2:3], s[16:19], 0 offen', 'for offen')
    must_refuse_asm('buffer_load_dword v4, v2, s[16:19], 0', 'without idxen or offen')

    # 9. Negative control: the pre-fix implementation drops v3, crashes with IndexError on lds,
    #    and crashes with ValueError on an SGPR soffset.
    st_ctrl = ProbeStage()
    _, addr_old = old_buffer_address(st_ctrl, ['v4', 'v[2:3]', 's[16:19]', '0'], ['idxen', 'offen', 'offset:12'])
    assert addr_old == 'v2 * ud2 + 12' and 'v3' not in addr_old, addr_old
    for fn, exc_type in (
        (lambda: old_buffer_address(ProbeStage(), ['off', 's[16:19]', '0'], ['lds']), (G.Unsupported, IndexError)),
        (lambda: old_buffer_address(ProbeStage(), ['v4', 'v2', 's[16:19]', 's8'], ['offen']), ValueError),
    ):
        try:
            fn()
        except exc_type:
            pass
        else:
            raise AssertionError('pre-fix negative control unexpectedly succeeded')


    # 10. All 32-bit words read through float4 pos in the MUBUF GPU cases are normal floats,
    #     so returning float4 from body(id) preserves all 32 bits on every adapter.
    for words in (expected_mubuf_idxen_offen(), expected_mubuf_offen_soffset()):
        for word in words[:64]:
            assert 0 < ((word >> 23) & 0xff) < 0xff, hex(word)


# ---- D3D12 compute-wrapped VS readback cases -----------------------------------------------------------

def build_mubuf_idxen_offen_pipeline():
    """VS: v2 = v0; v3 = (v0 & 3) * 8; buffer_load_dwordx4 v[4:7], v[2:3], s[16:19], s8 idxen offen offset:12."""
    vs = b''.join((
        struct.pack('<2I', 0xf4080400, 0xfa000010),                 # s_load_dwordx4 s[16:19], s[0:1], 16
        encode_vop1(0x01, 2, 256 + 0),                              # v_mov_b32_e32 v2, v0
        encode_vop2(0x1b, 3, 128 + 3, 0),                           # v_and_b32_e32 v3, 3, v0
        encode_vop2(0x1a, 3, 128 + 3, 3),                           # v_lshlrev_b32_e32 v3, 3, v3
        encode_mubuf(MUBUF_LOAD_DWORDX4, imm=12, idxen=1, offen=1, vaddr=2, vdata=4, srsrc=16, soffset=8),
        struct.pack('<3I', 0xf80008cf, 0x07060504, 0xbf810000),     # exp pos0 v4, v5, v6, v7 done; s_endpgm
    ))
    ps = struct.pack('<3I', 0xf800180f, 0, 0xbf810000)
    return G.Pipeline(vs, ps, root_with_one_constant())


def build_mtbuf_idxen_offen_pipeline():
    """VS: v2 = v0; v3 = (v0 * 3) & 7; tbuffer_load_format_xyzw (8_8_8_8_UNORM) v[4:7], v[2:3], s[16:19], s8."""
    vs = b''.join((
        struct.pack('<2I', 0xf4080400, 0xfa000010),
        encode_vop1(0x01, 2, 256 + 0),                              # v_mov_b32_e32 v2, v0
        encode_vop2(0x0b, 3, 128 + 3, 0),                           # v_mul_u32_u24_e32 v3, 3, v0
        encode_vop2(0x1b, 3, 128 + 7, 3),                           # v_and_b32_e32 v3, 7, v3
        encode_mtbuf(MTBUF_LOAD_XYZW, BUF_FMT_8_8_8_8_UNORM, imm=5,
                     idxen=1, offen=1, vaddr=2, vdata=4, srsrc=16, soffset=8),
        struct.pack('<3I', 0xf80008cf, 0x07060504, 0xbf810000),
    ))
    ps = struct.pack('<3I', 0xf800180f, 0, 0xbf810000)
    return G.Pipeline(vs, ps, root_with_one_constant())


def build_mubuf_offen_soffset_pipeline():
    """VS: v3 = v0 << 4; buffer_load_dwordx4 v[4:7], v3, s[16:19], s8 offen offset:8."""
    vs = b''.join((
        struct.pack('<2I', 0xf4080400, 0xfa000010),
        encode_vop2(0x1a, 3, 128 + 4, 0),                           # v_lshlrev_b32_e32 v3, 4, v0
        encode_mubuf(MUBUF_LOAD_DWORDX4, imm=8, offen=1, vaddr=3, vdata=4, srsrc=16, soffset=8),
        struct.pack('<3I', 0xf80008cf, 0x07060504, 0xbf810000),
    ))
    ps = struct.pack('<3I', 0xf800180f, 0, 0xbf810000)
    return G.Pipeline(vs, ps, root_with_one_constant())


def expected_mubuf_idxen_offen(*, stride=24, soffset=32, imm=12, drop_voffset=False, drop_soffset=False):
    words = [SENTINEL] * 320
    for vid in range(16):
        voff = 0 if drop_voffset else (vid & 3) * 8
        soff = 0 if drop_soffset else soffset
        addr = vid * stride + voff + soff + imm
        words[vid * 4:vid * 4 + 4] = struct.unpack_from('<4I', DATA, addr)
    return words


def expected_mtbuf_idxen_offen(*, stride=13, soffset=19, imm=5, drop_voffset=False):
    words = [SENTINEL] * 320
    for vid in range(16):
        voff = 0 if drop_voffset else ((vid * 3) & 7)
        addr = vid * stride + voff + soffset + imm
        words[vid * 4:vid * 4 + 4] = list(DATA[addr:addr + 4])
    return words


def expected_mubuf_offen_soffset(*, soffset=48, imm=8, drop_soffset=False):
    words = [SENTINEL] * 320
    for vid in range(16):
        soff = 0 if drop_soffset else soffset
        addr = (vid * 16) + soff + imm
        words[vid * 4:vid * 4 + 4] = struct.unpack_from('<4I', DATA, addr)
    return words


def wrapper(unorm_8=False):
    if unorm_8:
        expr = 'uint4((uint)round(p.x * 255.0), (uint)round(p.y * 255.0), (uint)round(p.z * 255.0), (uint)round(p.w * 255.0))'
    else:
        expr = 'asuint(p)'
    return f'''
RWByteAddressBuffer outputWords : register(u0);
[numthreads(16, 1, 1)]
void readback(uint id : SV_GroupIndex) {{
    float4 p = body(id).pos;
    outputWords.Store4(id * 16, {expr});
}}
'''


def run_case(directory, compiler, consumer, adapter, pipe, vs_source, words, *, stride=0, soffset=0, unorm_8=False, fxc=None):
    hlsl, binary, case = (directory / n for n in ('addr.hlsl', 'addr.cso', 'addr.case'))
    constants = [0] * 16
    if ('rootconst', 0, 0) in pipe.uds:
        constants[pipe.uds.index(('rootconst', 0, 0))] = soffset
    if ('stride', 0) in pipe.uds:
        constants[pipe.uds.index(('stride', 0))] = stride
    case.write_bytes(struct.pack('<337I', 1, *words, *constants))
    hlsl.write_text(vs_source, encoding='ascii')
    if fxc:
        ps_hlsl = directory / 'addr_ps.hlsl'
        ps_hlsl.write_text(pipe.hlsl_ps(), encoding='ascii')
        subprocess.run([fxc, '/nologo', '/T', 'vs_5_1', '/E', 'main', '/Fo', str(directory / 'vs51.cso'), str(hlsl)],
                       check=True, capture_output=True, text=True)
        subprocess.run([fxc, '/nologo', '/T', 'ps_5_1', '/E', 'main', '/Fo', str(directory / 'ps51.cso'), str(ps_hlsl)],
                       check=True, capture_output=True, text=True)
    subprocess.run([compiler, '-T', 'vs_6_0', '-E', 'main', '-Fo', str(binary), str(hlsl)], check=True)
    with hlsl.open('a', encoding='ascii') as handle:
        handle.write(wrapper(unorm_8=unorm_8))
    subprocess.run([compiler, '-T', 'cs_6_0', '-E', 'readback', '-Fo', str(binary), str(hlsl)], check=True)
    return subprocess.run([consumer, adapter, '--compute-store-case', str(binary), str(case)],
                          capture_output=True, text=True)


def check_gpu(directory, compiler, consumer, adapter, fxc=None):
    # 1. MUBUF dwordx4 with combined idxen + offen + SGPR soffset + imm_offset.
    p_mubuf = build_mubuf_idxen_offen_pipeline()
    src_mubuf = p_mubuf.hlsl_vs()
    stride_slot = p_mubuf.uds.index(('stride', 0))
    soff_slot = p_mubuf.uds.index(('rootconst', 0, 0))
    addr_expr = f'v2 * ud{stride_slot} + v3 + ud{soff_slot} + 12'
    assert f'a = {addr_expr};' in src_mubuf, src_mubuf
    ref_mubuf = expected_mubuf_idxen_offen()
    res = run_case(directory, compiler, consumer, adapter, p_mubuf, src_mubuf, ref_mubuf,
                   stride=24, soffset=32, fxc=fxc)
    if res.returncode == 77:
        print(res.stdout, res.stderr)
        return 77
    if res.returncode:
        raise RuntimeError('mubuf idxen+offen:\n' + res.stdout + res.stderr)
    print('mubuf idxen+offen+soffset:', [l for l in res.stdout.splitlines() if 'mismatches=' in l][0])

    # 2. MTBUF 8_8_8_8_UNORM with combined idxen + offen + SGPR soffset + unaligned ldu().
    p_mtbuf = build_mtbuf_idxen_offen_pipeline()
    src_mtbuf = p_mtbuf.hlsl_vs()
    assert f'a = v2 * ud{stride_slot} + v3 + ud{soff_slot} + 5;' in src_mtbuf, src_mtbuf
    ref_mtbuf = expected_mtbuf_idxen_offen()
    res = run_case(directory, compiler, consumer, adapter, p_mtbuf, src_mtbuf, ref_mtbuf,
                   stride=13, soffset=19, unorm_8=True, fxc=fxc)
    if res.returncode:
        raise RuntimeError('mtbuf idxen+offen:\n' + res.stdout + res.stderr)
    print('mtbuf idxen+offen+soffset (unaligned UNORM):',
          [l for l in res.stdout.splitlines() if 'mismatches=' in l][0])

    # 3. MUBUF offen + SGPR soffset (no idxen).
    p_offen = build_mubuf_offen_soffset_pipeline()
    src_offen = p_offen.hlsl_vs()
    assert f'a = v3 + ud{soff_slot} + 8;' in src_offen, src_offen
    ref_offen = expected_mubuf_offen_soffset()
    res = run_case(directory, compiler, consumer, adapter, p_offen, src_offen, ref_offen,
                   soffset=48, fxc=fxc)
    if res.returncode:
        raise RuntimeError('mubuf offen+soffset:\n' + res.stdout + res.stderr)
    print('mubuf offen+soffset:', [l for l in res.stdout.splitlines() if 'mismatches=' in l][0])

    # 4. Wrong references must fail at their first differing word.
    for label, pipe, src, good, bad, kw in (
        ('mubuf reference without voffset', p_mubuf, src_mubuf, ref_mubuf,
         expected_mubuf_idxen_offen(drop_voffset=True), dict(stride=24, soffset=32)),
        ('mubuf reference without soffset', p_mubuf, src_mubuf, ref_mubuf,
         expected_mubuf_idxen_offen(drop_soffset=True), dict(stride=24, soffset=32)),
        ('mtbuf reference without voffset', p_mtbuf, src_mtbuf, ref_mtbuf,
         expected_mtbuf_idxen_offen(drop_voffset=True), dict(stride=13, soffset=19, unorm_8=True)),
        ('offen reference without soffset', p_offen, src_offen, ref_offen,
         expected_mubuf_offen_soffset(drop_soffset=True), dict(soffset=48)),
    ):
        first = next(i for i, (a, b) in enumerate(zip(bad, good)) if a != b)
        fail = run_case(directory, compiler, consumer, adapter, pipe, src, bad, **kw)
        if fail.returncode != 1 or f'word={first} expected=' not in fail.stderr:
            raise AssertionError(f'{label} escaped detection:\n{fail.stderr}')
        print(f'control {label!r}: fails at word {first} as expected')

    # 5. Mutated translations (including the pre-fix dropped-voffset bug and swapped vaddr pair) must fail.
    for label, old, new in (
        ('pre-fix dropped voffset', addr_expr, f'v2 * ud{stride_slot} + ud{soff_slot} + 12'),
        ('swapped vindex and voffset', addr_expr, f'v3 * ud{stride_slot} + v2 + ud{soff_slot} + 12'),
        ('dropped scalar soffset', addr_expr, f'v2 * ud{stride_slot} + v3 + 12'),
        ('dropped immediate offset', addr_expr, f'v2 * ud{stride_slot} + v3 + ud{soff_slot} + 0'),
    ):
        assert old in src_mubuf, label
        fail = run_case(directory, compiler, consumer, adapter, p_mubuf,
                        src_mubuf.replace(old, new, 1), ref_mubuf, stride=24, soffset=32)
        if fail.returncode != 1 or 'Compute readback differs' not in fail.stderr:
            raise AssertionError(f'mutation {label!r} escaped detection:\n{fail.stderr}')
        print(f'mutation {label!r}: fails readback as expected')

    print(f'{adapter}: 3 buffer-addressing cases, 4 wrong references and 4 translation mutations verified')
    return 0


def main():
    controls()
    if len(sys.argv) == 1 or sys.argv[1:] == ['controls']:
        print('startup buffer address controls passed')
        return 0
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    dxc, consumer, adapter = sys.argv[1:4]
    fxc = sys.argv[4] if len(sys.argv) == 5 else None
    with tempfile.TemporaryDirectory() as temp:
        return check_gpu(Path(temp), dxc, consumer, adapter, fxc=fxc)


if __name__ == '__main__':
    sys.exit(main())
