"""SMEM combined register+immediate addressing and modifier checks in tools/startup/gfx2hlsl.py.

usage: startup_smem_address_check.py [controls]
       startup_smem_address_check.py binary <llvm-mc>
       startup_smem_address_check.py <dxc> <offscreen_check> <warp|hardware> [fxc]

In RDNA 2 SMEM loads (s_load_dword* and s_buffer_load_dword*), SOFFSET[31:25] and the 21-bit
signed immediate OFFSET[20:0] can be active together: tools/disasm_rdna2.py prints the register
operand in ops[2] and 'offset:0x...' in mods. Previously, Stage.op_s_buffer_load_dword() and
Stage.op_s_load_dwordx4() ignored mods, silently dropping 'offset:0x...' on register-offset buffer
and sampler-heap loads, accepting unknown modifiers, reading negative 21-bit signed immediates
(0x100000..0x1fffff) as +1..2 MiB, and clobbering a runtime SGPR soffset when a multi-dword
s_buffer_load_dwordx* wrote over that register first.

All instruction bytes below are constructed from the RDNA 2 / tools/disasm_rdna2.py bit layouts.
With no arguments (or 'controls'), this script checks decoding, refusals, destination-overlap
protection and pre-fix negative controls without a GPU. Given DXC and offscreen_check, it executes
two compute-wrapped vertex buffer-load cases and one sampler-heap pixel draw on WARP or hardware
(plus FXC SM 5.1 compilation when FXC is supplied) and verifies that mutations and wrong references
fail readback.
"""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402

SENTINEL = 0xc5c5c5c5
# offscreen_check --compute-store-case fills the raw t0 buffer with this 320-dword pattern.
DATA = b''.join(struct.pack('<I', (0x8b02a7f1 + i * 0x413b1) & 0xffffffff) for i in range(320))

SMEM_LOAD_DWORD = 0x00
SMEM_LOAD_DWORDX2 = 0x01
SMEM_LOAD_DWORDX4 = 0x02
SMEM_LOAD_DWORDX8 = 0x03
SMEM_LOAD_DWORDX16 = 0x04
SMEM_BUFFER_LOAD_DWORD = 0x08
SMEM_BUFFER_LOAD_DWORDX2 = 0x09
SMEM_BUFFER_LOAD_DWORDX4 = 0x0a
SMEM_BUFFER_LOAD_DWORDX8 = 0x0b
SMEM_BUFFER_LOAD_DWORDX16 = 0x0c

SAMPLER_HEAP_FLAG = 0x800  # D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED

FULLSCREEN_VS = """\
float4 main(uint id : SV_VertexID) : SV_Position
{
    return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
}
"""


def encode_smem(op, sdata, sbase, *, soffset=125, imm=0, glc=0, dlc=0):
    """Construct an 8-byte RDNA 2 SMEM instruction from its hardware fields."""
    assert sbase % 2 == 0, sbase
    w0 = ((0b111101 << 26) | ((op & 0xff) << 18) | ((glc & 1) << 16) | ((dlc & 1) << 14)
          | ((sdata & 0x7f) << 6) | ((sbase >> 1) & 0x3f))
    w1 = ((soffset & 0x7f) << 25) | (imm & 0x1fffff)
    return struct.pack('<II', w0, w1)


def encode_sop1(op, sdst, ssrc0):
    """Construct a 4-byte RDNA 2 SOP1 instruction."""
    w0 = 0xbe800000 | ((sdst & 0x7f) << 16) | ((op & 0xff) << 8) | (ssrc0 & 0xff)
    return struct.pack('<I', w0)


def encode_sop2(op, sdst, ssrc0, ssrc1):
    """Construct a 4-byte RDNA 2 SOP2 instruction."""
    w0 = 0x80000000 | ((op & 0x7f) << 23) | ((sdst & 0x7f) << 16) | ((ssrc1 & 0xff) << 8) | (ssrc0 & 0xff)
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


def sampler_heap_root_blob(flags=SAMPLER_HEAP_FLAG):
    """DXBC RTS0 container: parameter 0 descriptor table (s8), parameter 1 root constant (s9), sampler heap (s10)."""
    params = [(0, 0), (1, 0)]
    poff = 24
    coff = poff + 12 * len(params)
    rts0 = struct.pack('<6I', 2, len(params), poff, 0, 0, flags)
    rts0 += struct.pack('<3I', 0, 0, coff + 12) + struct.pack('<3I', 1, 0, coff)
    rts0 += struct.pack('<3I', 0, 0, 1)
    rts0 += struct.pack('<5I', 0, 0, 0, 0, 0)
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return b'DXBC' + bytes(16) + struct.pack('<III', 1, 36 + len(part), 1) + struct.pack('<I', 36) + part


class DummyPipeline:
    """Minimal Pipeline stand-in for isolated Stage instruction checks."""

    def __init__(self, *, sampler_heap=False):
        self.resources = {}
        self.uds = [('rect',)]
        self.notes = set()
        self.modified = set()
        self.sampler_heap = sampler_heap
        self.uses_sampler_heap = False

    def ud(self, source):
        if source not in self.uds:
            self.uds.append(source)
        return self.uds.index(source)

    def resource(self, key, kind):
        entry = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        return 'r%d' % entry['index']


class ProbeStage(G.Stage):
    """Isolated Stage with buffer descriptor s[16:19], root buffer s[4:5], table s[6:7], sampler heap s[10:11]."""

    def __init__(self, *, sampler_heap=False):
        self.pipe = DummyPipeline(sampler_heap=sampler_heap)
        self.kind = 'vs'
        self.lines = []
        self.sym = {
            4: ('ptr', ('root', 0)),
            5: ('ptrhi', ('root', 0)),
            6: ('ptr', ('table', 0)),
            8: ('ud', self.pipe.ud(('rootconst', 0, 0))),
            9: ('const', 0x20),
            10: ('ptr', ('smpheap',)),
            **{16 + i: ('desc', ('vbtab',), 16, i) for i in range(4)},
        }
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports, self.attrs, self.masks = {}, {}, set()


def run_bytes(code, *, sampler_heap=False):
    _, name, asm, _ = D.decode_instruction(code, 0)
    stage = ProbeStage(sampler_heap=sampler_heap)
    stage.one(name, asm)
    return asm, stage


def must_refuse_bytes(code, reason, *, sampler_heap=False):
    try:
        run_bytes(code, sampler_heap=sampler_heap)
    except G.Unsupported as exc:
        assert reason in str(exc), f'expected {reason!r} in {str(exc)!r}'
    else:
        _, _, asm, _ = D.decode_instruction(code, 0)
        raise AssertionError(f'expected Unsupported({reason!r}) for {asm!r}, but translation succeeded')


def must_refuse_asm(asm, reason, *, sampler_heap=False):
    name = asm.split()[0]
    stage = ProbeStage(sampler_heap=sampler_heap)
    try:
        stage.one(name, asm)
    except G.Unsupported as exc:
        assert reason in str(exc), f'expected {reason!r} in {str(exc)!r}'
    else:
        raise AssertionError(f'expected Unsupported({reason!r}) for {asm!r}, but translation succeeded')


def controls():
    # 1. Combined SGPR/m0 soffset + 21-bit immediate offset across all s_buffer_load_dword* widths.
    for op, width in (
        (SMEM_BUFFER_LOAD_DWORD, 1),
        (SMEM_BUFFER_LOAD_DWORDX2, 2),
        (SMEM_BUFFER_LOAD_DWORDX4, 4),
        (SMEM_BUFFER_LOAD_DWORDX8, 8),
        (SMEM_BUFFER_LOAD_DWORDX16, 16),
    ):
        asm, st = run_bytes(encode_smem(op, 24, 16, soffset=8, imm=0x18, glc=1, dlc=1))
        assert 'offset:0x18' in asm and 'glc' in asm and 'dlc' in asm, asm
        assert len(st.lines) == width, st.lines
        for i, line in enumerate(st.lines):
            assert f'+ {0x18 + 4 * i})' in line and 'ud1' in line, line
            assert (24 + i) not in st.sym, st.sym

        asm_m0, st_m0 = run_bytes(encode_smem(op, 24, 16, soffset=124, imm=0x10))
        assert 'm0 offset:0x10' in asm_m0, asm_m0
        # m0 cannot overlap a destination SGPR range, so no scratch 'a =' copy is emitted.
        assert len(st_m0.lines) == width and all(not line.startswith('a =') for line in st_m0.lines), st_m0.lines
        for i, line in enumerate(st_m0.lines):
            assert f'm0 + {0x10 + 4 * i})' in line, line

    # 2. Maximum non-negative 21-bit signed immediate (0xfffff) is accepted on both literal and register forms.
    _, st_max_lit = run_bytes(encode_smem(SMEM_BUFFER_LOAD_DWORD, 24, 16, soffset=125, imm=0xfffff, glc=1))
    assert '0x000fffffu' in st_max_lit.lines[0] and st_max_lit.sym.get(24) == ('fixed_buffer_word',), st_max_lit.lines
    _, st_max_reg = run_bytes(encode_smem(SMEM_BUFFER_LOAD_DWORD, 24, 16, soffset=8, imm=0xfffff))
    assert f'+ {0xfffff})' in st_max_reg.lines[0], st_max_reg.lines
    # Uppercase hex in synthetic assembly is also accepted:
    st_upper = ProbeStage()
    st_upper.one('s_buffer_load_dword', 's_buffer_load_dword s24, s[16:19], s8 offset:0X2A')
    assert '+ 42)' in st_upper.lines[0], st_upper.lines

    # 3. Destination-overlapping runtime SGPR soffset preserves the base address across multi-dword loads.
    _, st_overlap = run_bytes(encode_smem(SMEM_BUFFER_LOAD_DWORDX4, 20, 16, soffset=20, imm=0x10))
    assert st_overlap.lines[0] == 'a = s20;', st_overlap.lines
    for i in range(4):
        assert f'a + {0x10 + 4 * i})' in st_overlap.lines[1 + i], st_overlap.lines

    # 4. Dynamic sampler-heap s_load_dwordx4 carries the immediate offset in opt-in mode and keeps the shortcut clean.
    _, st_heap = run_bytes(encode_smem(SMEM_LOAD_DWORDX4, 24, 10, soffset=8, imm=0x20, glc=1), sampler_heap=True)
    assert any('ud1 + 32' in line for line in st_heap.lines), st_heap.lines
    assert [st_heap.sym.get(24 + i) for i in range(4)] == [('smpheap', 24, i) for i in range(4)]

    _, st_shortcut = run_bytes(encode_smem(SMEM_LOAD_DWORDX4, 24, 10, soffset=20, imm=0x20), sampler_heap=False)
    assert not st_shortcut.lines and 20 not in st_shortcut.used_s, (st_shortcut.lines, st_shortcut.used_s)
    assert st_shortcut.pipe.notes == {'dynamic sampler index replaced by a default sampler'}

    # 5. Negative 21-bit signed SMEM immediate encodings (0x100000..0x1fffff) are refused on all paths.
    for neg_imm in (0x100000, 0x1ffff0, 0x1ffffc, 0x1fffff):
        must_refuse_bytes(encode_smem(SMEM_BUFFER_LOAD_DWORD, 24, 16, soffset=125, imm=neg_imm),
                          'SMEM immediate offset')
        must_refuse_bytes(encode_smem(SMEM_BUFFER_LOAD_DWORDX4, 24, 16, soffset=8, imm=neg_imm),
                          'SMEM immediate offset')
        must_refuse_bytes(encode_smem(SMEM_LOAD_DWORD, 24, 4, soffset=125, imm=neg_imm),
                          'SMEM immediate offset')
        must_refuse_bytes(encode_smem(SMEM_LOAD_DWORDX8, 24, 6, soffset=125, imm=neg_imm),
                          'SMEM immediate offset')
        must_refuse_bytes(encode_smem(SMEM_LOAD_DWORDX4, 24, 10, soffset=8, imm=neg_imm),
                          'SMEM immediate offset', sampler_heap=True)
        must_refuse_bytes(encode_smem(SMEM_LOAD_DWORDX4, 24, 10, soffset=8, imm=neg_imm),
                          'SMEM immediate offset', sampler_heap=False)
    must_refuse_asm('s_buffer_load_dword s24, s[16:19], -4', 'SMEM immediate offset')
    must_refuse_asm('s_load_dword s24, s[4:5], -4', 'SMEM immediate offset')
    for token in ('-0x4', '-4', '0x1ffffc'):
        must_refuse_asm(f's_buffer_load_dword s24, s[16:19], s0 offset:{token}', 'SMEM immediate offset')
        must_refuse_asm(f's_load_dword s24, s[4:5], 0x0 offset:{token}', 'SMEM immediate offset')

    # 6. Invalid SMEM soffset operands (special registers, SGPRs > 105, VGPRs, pairs, descriptor SGPRs) are refused.
    for soffset, expected in (
        (106, "operand 'vcc_lo'"),
        (107, "operand 'vcc_hi'"),
        (108, "operand 'ttmp0'"),
        (126, "operand 'exec_lo'"),
        (127, "operand 'exec_hi'"),
        (16, 'numeric use of s16 holding desc'),
    ):
        must_refuse_bytes(encode_smem(SMEM_BUFFER_LOAD_DWORD, 24, 16, soffset=soffset, imm=8), expected)
        must_refuse_bytes(encode_smem(SMEM_LOAD_DWORDX4, 24, 10, soffset=soffset, imm=16),
                          expected, sampler_heap=True)
        must_refuse_bytes(encode_smem(SMEM_LOAD_DWORDX4, 24, 10, soffset=soffset, imm=16),
                          expected, sampler_heap=False)
    for bad_op in ('v0', 's108', 's[8:9]', 'vcc', 'exec', '1.0'):
        must_refuse_asm(f's_buffer_load_dword s24, s[16:19], {bad_op}', 'operand')

    # 7. Unmodeled SMEM modifiers, duplicate offset tokens and wrong register widths are refused.
    for mod in ('slc', 'lds', 'tfe', 'offset:bad'):
        must_refuse_asm(f's_buffer_load_dword s24, s[16:19], s8 {mod}', 'SMEM modifier')
        must_refuse_asm(f's_load_dword s24, s[4:5], 0x0 {mod}', 'SMEM modifier')
    must_refuse_asm('s_buffer_load_dword s24, s[16:19], s8 offset:0x10 offset:0x20', 'SMEM modifier offset:0x20')
    must_refuse_asm('s_buffer_load_dword s24, s[16:19], 0x10 offset:0x20', 'SMEM literal offset')
    must_refuse_asm('s_load_dword s24, s[4:5], 0x10 offset:0x20', 'SMEM literal offset')
    must_refuse_asm('s_buffer_load_dwordx2 s24, s[16:19], 0x0', 's_buffer_load destination')
    must_refuse_asm('s_load_dwordx4 s24, s[4:5], 0x0', 's_load destination')
    must_refuse_bytes(encode_smem(SMEM_LOAD_DWORDX2, 24, 10, soffset=8, imm=0x10),
                      "s_load of 2 dwords from ('smpheap',)", sampler_heap=False)

    # 8. Live in-process negative control: when smem_mods ignores modifiers, both handlers drop 'offset:0x18'.
    real_smem_mods = G.smem_mods
    G.smem_mods = lambda _mods: None
    try:
        _, st_dropped_buf = run_bytes(encode_smem(SMEM_BUFFER_LOAD_DWORD, 20, 16, soffset=8, imm=0x18))
        assert '+ 0)' in st_dropped_buf.lines[0] and '+ 24)' not in st_dropped_buf.lines[0], st_dropped_buf.lines
        _, st_dropped_heap = run_bytes(encode_smem(SMEM_LOAD_DWORDX4, 24, 10, soffset=8, imm=0x18), sampler_heap=True)
        assert '+ 24' not in st_dropped_heap.lines[0], st_dropped_heap.lines
    finally:
        G.smem_mods = real_smem_mods

    # 9. All 32-bit words exported through float4 pos in the GPU buffer cases are normal IEEE-754 floats.
    for soffset in (32, 64):
        for words in (expected_smem_dwordx4(soffset=soffset), expected_smem_m0_and_overlap(soffset=soffset)):
            for word in words[:64]:
                assert 0 < ((word >> 23) & 0xff) < 0xff, hex(word)


# ---- D3D12 GPU readback cases -------------------------------------------------------------------------

def build_smem_dwordx4_pipeline():
    """VS: s_load_dwordx4 s[16:19], s[0:1], 0x10; s_buffer_load_dwordx4 s[20:23], s[16:19], s8 offset:0x18."""
    vs = b''.join((
        encode_smem(SMEM_LOAD_DWORDX4, 16, 0, soffset=125, imm=0x10),
        encode_smem(SMEM_BUFFER_LOAD_DWORDX4, 20, 16, soffset=8, imm=0x18, glc=1),
        encode_vop1(0x01, 4, 20),                                    # v_mov_b32_e32 v4, s20
        encode_vop1(0x01, 5, 21),                                    # v_mov_b32_e32 v5, s21
        encode_vop1(0x01, 6, 22),                                    # v_mov_b32_e32 v6, s22
        encode_vop1(0x01, 7, 23),                                    # v_mov_b32_e32 v7, s23
        struct.pack('<3I', 0xf80008cf, 0x07060504, 0xbf810000),      # exp pos0 v4, v5, v6, v7 done; s_endpgm
    ))
    ps = struct.pack('<3I', 0xf800180f, 0, 0xbf810000)
    return G.Pipeline(vs, ps, root_with_one_constant())


def build_smem_m0_and_overlap_pipeline():
    """VS: m0 + offset:0x14 dlc, const SGPR + offset:0x08, and overlapping runtime SGPR s[20:21] + offset:0x1c glc."""
    vs = b''.join((
        encode_smem(SMEM_LOAD_DWORDX4, 16, 0, soffset=125, imm=0x10),
        encode_smem(SMEM_BUFFER_LOAD_DWORD, 22, 16, soffset=124, imm=0x14, dlc=1),  # m0 (=0) + 0x14
        encode_sop1(0x03, 21, 128 + 32),                             # s_mov_b32 s21, 32
        encode_smem(SMEM_BUFFER_LOAD_DWORD, 23, 16, soffset=21, imm=0x08),          # const s21 (32) + 8 = 40
        encode_sop2(0x00, 20, 8, 128 + 12),                          # s_add_u32 s20, s8, 12
        encode_smem(SMEM_BUFFER_LOAD_DWORDX2, 20, 16, soffset=20, imm=0x1c, glc=1), # overlapping s[20:21] at s8 + 40..44
        encode_vop1(0x01, 4, 22),                                    # v_mov_b32_e32 v4, s22
        encode_vop1(0x01, 5, 23),                                    # v_mov_b32_e32 v5, s23
        encode_vop1(0x01, 6, 20),                                    # v_mov_b32_e32 v6, s20
        encode_vop1(0x01, 7, 21),                                    # v_mov_b32_e32 v7, s21
        struct.pack('<3I', 0xf80008cf, 0x07060504, 0xbf810000),
    ))
    ps = struct.pack('<3I', 0xf800180f, 0, 0xbf810000)
    return G.Pipeline(vs, ps, root_with_one_constant())


def build_sampler_heap_offset_pipeline():
    """PS: s_load_dwordx4 s[24:27], s[10:11], s9 offset:0x10 -> samples at (1.25, 0.5) via dynamic heap."""
    vs = bytes.fromhex('f202087e' '80020a7e' 'cf0800f805050504' '000081bf')
    ps = b''.join((
        encode_smem(SMEM_LOAD_DWORDX8, 16, 8, soffset=125, imm=0x0),
        encode_smem(SMEM_LOAD_DWORDX4, 24, 10, soffset=9, imm=0x10),
        bytes.fromhex('ff02007e0000a03f' 'f002027e' '7fc08cbf'
                      '080f80f00000c400' '703f8cbf' '0f1800f800010203' '000081bf'),
    ))
    return G.Pipeline(vs, ps, sampler_heap_root_blob(), sampler_heap=True)


def expected_smem_dwordx4(*, soffset=32, imm=0x18, drop_soffset=False, drop_imm=False):
    base = (0 if drop_soffset else soffset) + (0 if drop_imm else imm)
    quad = list(struct.unpack_from('<4I', DATA, base))
    return quad * 16 + [SENTINEL] * (320 - 64)


def expected_smem_m0_and_overlap(*, soffset=32, drop_imm=False):
    m0_addr = 0 if drop_imm else 0x14
    const_addr = 32 + (0 if drop_imm else 0x08)
    ov_base = (soffset + 12) + (0 if drop_imm else 0x1c)
    w0, = struct.unpack_from('<I', DATA, m0_addr)
    w1, = struct.unpack_from('<I', DATA, const_addr)
    w2, w3 = struct.unpack_from('<2I', DATA, ov_base)
    return [w0, w1, w2, w3] * 16 + [SENTINEL] * (320 - 64)


def wrapper():
    return '''
RWByteAddressBuffer outputWords : register(u0);
[numthreads(16, 1, 1)]
void readback(uint id : SV_GroupIndex) {
    float4 p = body(id).pos;
    outputWords.Store4(id * 16, asuint(p));
}
'''


def run_buffer_case(directory, compiler, consumer, adapter, pipe, vs_source, words, *, soffset=0, fxc=None):
    hlsl, binary, case = (directory / n for n in ('smem.hlsl', 'smem.cso', 'smem.case'))
    constants = [0] * 16
    if ('rootconst', 0, 0) in pipe.uds:
        constants[pipe.uds.index(('rootconst', 0, 0))] = soffset
    case.write_bytes(struct.pack('<337I', 1, *words, *constants))
    hlsl.write_text(vs_source, encoding='ascii')
    if fxc:
        ps_hlsl = directory / 'smem_ps.hlsl'
        ps_hlsl.write_text(pipe.hlsl_ps(), encoding='ascii')
        subprocess.run([fxc, '/nologo', '/T', 'vs_5_1', '/E', 'main', '/Fo', str(directory / 'vs51.cso'), str(hlsl)],
                       check=True, capture_output=True, text=True)
        subprocess.run([fxc, '/nologo', '/T', 'ps_5_1', '/E', 'main', '/Fo', str(directory / 'ps51.cso'), str(ps_hlsl)],
                       check=True, capture_output=True, text=True)
    subprocess.run([compiler, '-T', 'vs_6_0', '-E', 'main', '-Fo', str(binary), str(hlsl)], check=True)
    with hlsl.open('a', encoding='ascii') as handle:
        handle.write(wrapper())
    subprocess.run([compiler, '-T', 'cs_6_0', '-E', 'readback', '-Fo', str(binary), str(hlsl)], check=True)
    return subprocess.run([consumer, adapter, '--compute-store-case', str(binary), str(case)],
                          capture_output=True, text=True)


def compile_stage(compiler, kind, source, stem, profile, directory):
    hlsl = directory / f'{stem}.hlsl'
    cso = directory / f'{stem}.cso'
    hlsl.write_text(source, encoding='ascii')
    args = ([compiler, '/nologo', '/T', profile, '/E', 'main', '/Fo', str(cso), str(hlsl)]
            if kind == 'fxc' else
            [compiler, '-T', profile, '-E', 'main', '-Fo', str(cso), str(hlsl)])
    subprocess.run(args, check=True, capture_output=True, text=True)
    return cso


def check_gpu(directory, compiler, consumer, adapter, fxc=None):
    # 1. s_buffer_load_dwordx4 with SGPR soffset + offset:0x18 at two distinct runtime soffsets.
    p_x4 = build_smem_dwordx4_pipeline()
    src_x4 = p_x4.hlsl_vs()
    for soffset in (32, 64):
        ref = expected_smem_dwordx4(soffset=soffset)
        res = run_buffer_case(directory, compiler, consumer, adapter, p_x4, src_x4, ref,
                              soffset=soffset, fxc=fxc if soffset == 32 else None)
        if res.returncode == 77:
            print(res.stdout, res.stderr)
            return 77
        if res.returncode:
            raise RuntimeError(f's_buffer_load_dwordx4 soffset={soffset}:\n' + res.stdout + res.stderr)
        print(f's_buffer_load_dwordx4 (soffset={soffset}, offset:0x18):',
              [l for l in res.stdout.splitlines() if 'mismatches=' in l][0])

    # 2. m0 + offset:0x14 dlc and destination-overlapping runtime SGPR s[20:21] + offset:0x1c glc.
    p_m0 = build_smem_m0_and_overlap_pipeline()
    src_m0 = p_m0.hlsl_vs()
    ref_m0 = expected_smem_m0_and_overlap(soffset=32)
    res_m0 = run_buffer_case(directory, compiler, consumer, adapter, p_m0, src_m0, ref_m0,
                             soffset=32, fxc=fxc)
    if res_m0.returncode:
        raise RuntimeError('s_buffer_load m0+overlap:\n' + res_m0.stdout + res_m0.stderr)
    print('s_buffer_load_dwordx2 (m0+0x14 and overlapping s20+0x1c):',
          [l for l in res_m0.stdout.splitlines() if 'mismatches=' in l][0])

    # 3. Wrong references (dropped imm_offset or dropped soffset) must fail at word 0.
    ref_x4_32 = expected_smem_dwordx4(soffset=32)
    for label, pipe, src, good, bad in (
        ('dwordx4 reference without immediate offset', p_x4, src_x4, ref_x4_32,
         expected_smem_dwordx4(soffset=32, drop_imm=True)),
        ('dwordx4 reference without soffset', p_x4, src_x4, ref_x4_32,
         expected_smem_dwordx4(soffset=32, drop_soffset=True)),
        ('m0+overlap reference without immediate offsets', p_m0, src_m0, ref_m0,
         expected_smem_m0_and_overlap(soffset=32, drop_imm=True)),
    ):
        first = next(i for i, (a, b) in enumerate(zip(bad, good)) if a != b)
        fail = run_buffer_case(directory, compiler, consumer, adapter, pipe, src, bad, soffset=32)
        if fail.returncode != 1 or f'word={first} expected=' not in fail.stderr:
            raise AssertionError(f'{label} escaped detection:\n{fail.stderr}')
        print(f'control {label!r}: fails at word {first} as expected')

    # 4. Translation mutations (pre-fix dropped offset:0x18 and reading clobbered s20 instead of saved a)
    #    must fail. Mask the clobbered s20 word into the 1280-byte buffer so root-SRV hardware does not fault.
    dropped_imm_hlsl = re.sub(r'\+ (24|28|32|36)\)', lambda m: f'+ {int(m.group(1)) - 24})', src_x4)
    assert dropped_imm_hlsl != src_x4
    clobbered_overlap_hlsl = src_m0.replace('Load(a + 32)', 'Load((s20 & 0x1fcu) + 32)', 1)
    assert clobbered_overlap_hlsl != src_m0
    for label, pipe, mutant, ref in (
        ('pre-fix dropped offset:0x18', p_x4, dropped_imm_hlsl, ref_x4_32),
        ('clobbered overlapping s20 soffset', p_m0, clobbered_overlap_hlsl, ref_m0),
    ):
        fail = run_buffer_case(directory, compiler, consumer, adapter, pipe, mutant, ref, soffset=32)
        if fail.returncode != 1 or 'Compute readback differs' not in fail.stderr:
            raise AssertionError(f'mutation {label!r} escaped detection:\n{fail.stderr}')
        print(f'mutation {label!r}: fails readback as expected')

    # 5. Sampler-heap pixel draw: s_load_dwordx4 s[24:27], s[10:11], s9 offset:0x10.
    p_heap = build_sampler_heap_offset_pipeline()
    manifest = directory / 'heap_manifest.txt'
    manifest.write_text(p_heap.manifest().rstrip() + '\nsmp 0 0 0 0 0\n', encoding='ascii')
    compilers = [('dxc', compiler, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    for kind, bin_compiler, vs_prof, ps_prof in compilers:
        vs_cso = compile_stage(bin_compiler, kind, FULLSCREEN_VS, f'fs_vs_{kind}', vs_prof, directory)
        ps_src = p_heap.hlsl_ps()
        ps_cso = compile_stage(bin_compiler, kind, ps_src, f'heap_ps_{kind}', ps_prof, directory)
        # Entry 0 = wrap (0.25 -> 0x3e800000), entry 1 = clamp (0.75 -> 0x3f400000),
        # entry 2+ = black border (0.0 -> 0x00000000), entry 2047 = fallback (0.5 -> 0x3f000000).
        # With offset:0x10 (+16 bytes), s9=0 selects entry 1, s9=16 selects entry 2, s9=2046*16 hits fallback 2047.
        for s9_bytes, expected_red in ((0, 0x3f400000), (16, 0x00000000), (2046 * 16, 0x3f000000)):
            draw = subprocess.run([consumer, adapter, '--startup-sampler-heap', str(vs_cso), str(ps_cso),
                                   str(s9_bytes), f'{expected_red:08x}', str(manifest)],
                                  capture_output=True, text=True)
            if draw.returncode:
                raise RuntimeError(f'sampler heap ({kind}, s9={s9_bytes}):\n' + draw.stdout + draw.stderr)
        # Pre-fix mutation (dropping + 16 on the sampler heap byte offset) must read entry 0 instead of entry 1.
        dropped_heap = re.sub(r'\+ 16\b', '', ps_src)
        assert dropped_heap != ps_src
        bad_cso = compile_stage(bin_compiler, kind, dropped_heap, f'heap_bad_{kind}', ps_prof, directory)
        bad_draw = subprocess.run([consumer, adapter, '--startup-sampler-heap', str(vs_cso), str(bad_cso),
                                   '0', '3f400000', str(manifest)],
                                  capture_output=True, text=True)
        if bad_draw.returncode == 0:
            raise AssertionError(f'sampler heap dropped-offset mutation ({kind}) escaped detection')
        print(f'sampler heap offset:0x10 ({kind}): 3 draws and 1 dropped-offset mutation verified')

    print(f'{adapter}: SMEM buffer and sampler-heap offset readbacks verified')
    return 0


def check_binary(llvm_mc):
    cases = (
        ('s_buffer_load_dwordx4 s[20:23], s[16:19], s8 offset:0x18 glc',
         encode_smem(SMEM_BUFFER_LOAD_DWORDX4, 20, 16, soffset=8, imm=0x18, glc=1)),
        ('s_buffer_load_dword s22, s[16:19], m0 offset:0x14 dlc',
         encode_smem(SMEM_BUFFER_LOAD_DWORD, 22, 16, soffset=124, imm=0x14, dlc=1)),
        ('s_buffer_load_dwordx2 s[20:21], s[16:19], s20 offset:0x1c glc',
         encode_smem(SMEM_BUFFER_LOAD_DWORDX2, 20, 16, soffset=20, imm=0x1c, glc=1)),
        ('s_load_dwordx4 s[24:27], s[10:11], s9 offset:0x10',
         encode_smem(SMEM_LOAD_DWORDX4, 24, 10, soffset=9, imm=0x10)),
    )
    source = '\n'.join(asm for asm, _ in cases) + '\n'
    want = b''.join(raw for _, raw in cases)
    proc = subprocess.run([llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-show-encoding'],
                          input=source, capture_output=True, text=True, check=True)
    built = bytearray()
    for line in proc.stdout.splitlines():
        m = re.search(r'encoding: \[([^\]]*)\]', line)
        if m:
            built += bytes(int(b, 16) for b in m.group(1).split(','))
    if bytes(built) != want:
        raise AssertionError(f'llvm-mc encoding mismatch: got {bytes(built).hex()}, expected {want.hex()}')
    print('SMEM combined-offset encodings match llvm-mc')
    return 0


def main():
    controls()
    if len(sys.argv) == 1 or sys.argv[1:] == ['controls']:
        print('startup SMEM address controls passed')
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == 'binary':
        return check_binary(sys.argv[2])
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    dxc, consumer, adapter = sys.argv[1:4]
    fxc = sys.argv[4] if len(sys.argv) == 5 else None
    with tempfile.TemporaryDirectory() as temp:
        return check_gpu(Path(temp), dxc, consumer, adapter, fxc=fxc)


if __name__ == '__main__':
    sys.exit(main())
