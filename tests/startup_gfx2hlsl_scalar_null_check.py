"""RDNA 2 scalar null operand (code 125) and Stage.scalar destination guards in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation, CPU SCC evaluation across 14 32-bit SALU ops, null-source
                                       folding/uniformity, TypeError/Unsupported destination sweep, pure Python
  binary <llvm-mc>                     reassemble every fixture and instruction spelling and compare with recorded words
  decode <libLLVM>                     disassemble the canonical encodings with LLVM's C API and compare with disasm_rdna2.py
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile and draw the pixel fixture (FXC ps_5_1 + DXC ps_6_0) and execute the
                                       root-constant SALU null-destination + branch buffer fixture with negative controls

AMD RDNA 2 ISA reference (scalar/vector source and destination encoding code 125 = null):
  - Reading null returns 0 (wave-uniform 32-bit zero).
  - Writing null discards the destination register result while still performing modeled 32-bit SCC updates.
  - Unmodelled scalar destinations (vcc_hi, exec_hi, ttmp*, _b64/lane-mask/descriptor/unmodelled-SCC writes to null)
    raise gfx2hlsl.Unsupported instead of TypeError (#2618, #1723, #2463).
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    Expr,
    MASK,
    POST_SOURCE,
    POST_WORDS,
    PRE_SOURCE,
    PRE_WORDS,
    check,
    signed,
    vs_source,
    vs_words,
    words_text,
)

NULL = 125
M0 = 124
VCC_LO, VCC_HI = 106, 107
TTMP0 = 108
EXEC_LO, EXEC_HI = 126, 127

SENTINEL = 0xc5c5c5c5
# Matches offscreen_check --compute-store-case's built-in t0 ByteAddressBuffer pattern.
DATA = b''.join(struct.pack('<I', (0x8b02a7f1 + i * 0x413b1) & 0xffffffff) for i in range(320))


def sop1(op, sdst, ssrc0, lit=None):
    out = struct.pack('<I', 0xbe800000 | ((sdst & 0x7f) << 16) | ((op & 0xff) << 8) | (ssrc0 & 0xff))
    return out if lit is None else out + struct.pack('<I', lit & MASK)


def sop2(op, sdst, ssrc0, ssrc1, lit=None):
    out = struct.pack('<I', 0x80000000 | ((op & 0x7f) << 23) | ((sdst & 0x7f) << 16) | ((ssrc1 & 0xff) << 8) | (ssrc0 & 0xff))
    return out if lit is None else out + struct.pack('<I', lit & MASK)


def sopk(op, sdst, simm16):
    return struct.pack('<I', 0xb0000000 | ((op & 0x1f) << 23) | ((sdst & 0x7f) << 16) | (simm16 & 0xffff))


def sopc(op, ssrc0, ssrc1):
    return struct.pack('<I', 0xbf000000 | ((op & 0x7f) << 16) | ((ssrc1 & 0xff) << 8) | (ssrc0 & 0xff))


def sopp(op, simm16=0):
    return struct.pack('<I', 0xbf800000 | ((op & 0x7f) << 16) | (simm16 & 0xffff))


def vop1(op, vdst, src0):
    return struct.pack('<I', 0x7e000000 | ((vdst & 0xff) << 17) | ((op & 0xff) << 9) | (src0 & 0x1ff))


def vop2(op, vdst, src0, vsrc1):
    return struct.pack('<I', ((op & 0x3f) << 25) | ((vdst & 0xff) << 17) | ((vsrc1 & 0xff) << 9) | (src0 & 0x1ff))


def vop3a(op, vdst, src0, src1, src2=0):
    return struct.pack('<2I', 0xd4000000 | (op << 16) | (vdst & 0xff), (src2 << 18) | (src1 << 9) | src0)


def smem(op, sdata, sbase, *, soffset=NULL, imm=0):
    w0 = (0b111101 << 26) | ((op & 0xff) << 18) | ((sdata & 0x7f) << 6) | ((sbase >> 1) & 0x3f)
    w1 = ((soffset & 0x7f) << 25) | (imm & 0x1fffff)
    return struct.pack('<2I', w0, w1)


def root_with_one_constant():
    header = struct.pack('<6I', 2, 1, 24, 0, 0, 0)
    param = struct.pack('<3I', 1, 0, 36)
    const = struct.pack('<3I', 0, 0, 1)
    rts0 = header + param + const
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return b'DXBC' + bytes(16) + struct.pack('<III', 1, 36 + len(part), 1) + struct.pack('<I', 36) + part


# Canonical LLVM 15 round-trip lines (disasm_rdna2 <-> llvm-mc <-> libLLVM).
CANONICAL_CASES = (
    ('s_and_b32 null, s8, 4', sop2(0x0e, NULL, 8, 128 + 4)),
    ('s_bfe_u32 null, s8, 0x30008', sop2(0x27, NULL, 8, 255, lit=0x30008)),
    ('s_mov_b32 null, s8', sop1(0x03, NULL, 8)),
    ('s_not_b32 null, s8', sop1(0x07, NULL, 8)),
    ('s_movk_i32 null, 0x100', sopk(0x00, NULL, 0x100)),
    ('s_cselect_b32 s2, 32, null', sop2(0x0a, 2, 128 + 32, NULL)),
    ('s_or_b32 m0, null, s2', sop2(0x10, M0, NULL, 2)),
    ('s_cmp_eq_u32 s8, null', sopc(0x06, 8, NULL)),
    ('s_bfm_b32 null, s8, 2', sop2(0x24, NULL, 8, 128 + 2)),
    ('s_and_b64 null, exec, s[0:1]', sop2(0x0f, NULL, EXEC_LO, 0)),
    ('v_subrev_nc_u32_e32 v12, null, v0', vop2(0x27, 12, NULL, 0)),
    ('v_xor3_b32 v13, v1, v5, null', vop3a(0x178, 13, 257, 261, NULL)),
    ('v_add_f32_e32 v14, null, v2', vop2(0x03, 14, NULL, 2)),
    ('v_cvt_pk_u16_u32 v15, null, v3', vop3a(0x36a, 15, NULL, 259)),
)

# Pixel-stage draw fixture (FXC ps_5_1 + DXC ps_6_0): exercises null as VOP2/VOP3A source and SALU null-dst + null-src.
PS_BODY_SOURCE = """\
v_subrev_nc_u32_e32 v12, null, v0
v_xor3_b32 v13, v1, v5, null
v_add_f32_e32 v14, null, v2
s_mov_b32 s4, 1.0
s_and_b32 null, s4, 0x7f800000
s_cselect_b32 s5, 0x55, null
v_xor_b32_e32 v15, s5, v3
"""
PS_BODY_WORDS = b''.join((
    vop2(0x27, 12, NULL, 0),
    vop3a(0x178, 13, 257, 261, NULL),
    vop2(0x03, 14, NULL, 2),
    sop1(0x03, 4, 242),
    sop2(0x0e, NULL, 4, 255, lit=0x7f800000),
    sop2(0x0a, 5, 255, NULL, lit=0x55),
    vop2(0x1d, 15, 5, 3),
))
PS_WORDS = PRE_WORDS + PS_BODY_WORDS + POST_WORDS

PS_DRAW_INPUTS = [
    # Set 0: normal binary32 words in v0..v3, plus v5 (set index 1's second element) for v_xor3_b32.
    (0x3f800011, 0, 0),
    (0x40200022, 0x0000ff00, 0),
    (0x40600000, 0, 0),  # 3.5f
    (0x41000033, 0, 0),
    # Set 1:
    (0xbf800044, 0, 0),
    (0xc1200055, 0x00aa5500, 0),
    (0xc0100000, 0, 0),  # -2.25f
    (0x3e800066, 0, 0),
]


def f32_add_zero(bits, addend_bits=0):
    a = struct.unpack('<f', struct.pack('<I', addend_bits))[0]
    b = struct.unpack('<f', struct.pack('<I', bits))[0]
    return struct.unpack('<I', struct.pack('<f', a + b))[0]


def ref_ps_chunk(chunk):
    return [
        chunk[0][0] & MASK,
        (chunk[1][0] ^ chunk[1][1]) & MASK,
        f32_add_zero(chunk[2][0], 0),
        (chunk[3][0] ^ 0x55) & MASK,
    ]


# 14 32-bit SCA operations that write modeled SCC ($ != 0).
SCC_32_OPS = (
    ('s_and_b32', 2, lambda a, b: (a & b) != 0),
    ('s_or_b32', 2, lambda a, b: (a | b) != 0),
    ('s_xor_b32', 2, lambda a, b: (a ^ b) != 0),
    ('s_andn2_b32', 2, lambda a, b: (a & (~b & MASK)) != 0),
    ('s_orn2_b32', 2, lambda a, b: (a | (~b & MASK)) != 0),
    ('s_not_b32', 1, lambda a, _b: (~a & MASK) != 0),
    ('s_nand_b32', 2, lambda a, b: (~(a & b) & MASK) != 0),
    ('s_nor_b32', 2, lambda a, b: (~(a | b) & MASK) != 0),
    ('s_xnor_b32', 2, lambda a, b: (~(a ^ b) & MASK) != 0),
    ('s_lshl_b32', 2, lambda a, b: ((a << (b & 31)) & MASK) != 0),
    ('s_lshr_b32', 2, lambda a, b: ((a >> (b & 31)) & MASK) != 0),
    ('s_ashr_i32', 2, lambda a, b: ((signed(a) >> (b & 31)) & MASK) != 0),
    ('s_bfe_u32', 2, lambda a, b: ((a >> (b & 31)) & (MASK if ((b >> 16) & 127) >= 32 else ((1 << ((b >> 16) & 31)) - 1))) != 0),
    ('s_absdiff_i32', 2, lambda a, b: (abs(signed((a - b) & MASK)) & MASK) != 0),
)

SCC_PAIRS = (
    (0, 0), (0, MASK), (MASK, 0), (MASK, MASK), (0x55555555, 0xaaaaaaaa),
    (0x80000000, 1), (0x80000000, 31), (1, 31), (0x0000ff00, 0x00080008),
    (0x0000ff00, 0x00040000), (0x00010000, 0x00200010), (0xdeadbeef, 0xdeadbeef),
)


class DummyPipe:
    sampler_heap = False
    rtv_formats = ()
    srv_formats = {}

    def __init__(self):
        self.resources = {}
        self.uavs = {}
        self.uds = []
        self.modified = set()
        self.other_modified = set()

    def ud(self, tag):
        if tag not in self.uds:
            self.uds.append(tag)
        return self.uds.index(tag)

    def resource(self, key, kind):
        entry = self.resources.setdefault(key, {'index': len(self.resources), 'kind': kind})
        return 'r%d' % entry['index']


class ProbeStage(G.Stage):
    def __init__(self, kind='ps'):
        self.pipe = DummyPipe()
        self.kind = kind
        self.lines = []
        self.sym = {
            5: ('const', 0x12),
            10: ('ptr', ('root', 1)),
            11: ('ptrhi', ('root', 1)),
            **{16 + i: ('desc', ('table', 0), 16, i) for i in range(4)},
        }
        self.used_s, self.used_v = set(), set()
        self.uniform_v, self.varying_s = set(), set()
        self.scc_uniform = True
        self.defined_uniform_s = {0, 1, 2, 5}
        self.bad_sgprs = set()
        self.uniform_expr = {}
        self.since_join = None
        self.low16_only_v, self.pinned_low16_v, self.whole_read_v = set(), set(), set()
        self.exec_known = True
        self.exports, self.attrs, self.write_masks, self.masks = {}, {}, {}, set()


def eval_scc_line(line, s0, s1):
    assert line.startswith('scc = ') and line.endswith(';'), line
    return bool(Expr(line[len('scc = '):-1]).run({'s0': s0 & MASK, 's1': s1 & MASK})[0])


def build_root_null_pipeline():
    """VS exercising runtime root-constant s8 through SALU null-destination SCC updates, no-SCC null discards,
    null scalar sources, and s_cbranch_scc0 into s_buffer_load_dwordx4."""
    vs = b''.join((
        smem(0x02, 16, 0, soffset=NULL, imm=0x10),          # s_load_dwordx4 s[16:19], s[0:1], 0x10
        sop2(0x0e, NULL, 8, 128 + 4),                       # s_and_b32 null, s8, 4          -> scc = (s8 & 4) != 0
        sop1(0x03, NULL, 8),                                # s_mov_b32 null, s8             -> no-op, keeps scc
        sop2(0x26, NULL, 8, 128 + 7),                       # s_mul_i32 null, s8, 7          -> no-op, keeps scc
        sop2(0x24, NULL, 8, 128 + 2),                       # s_bfm_b32 null, s8, 2          -> no-op, keeps scc
        sop2(0x0a, 2, 128 + 32, NULL),                      # s_cselect_b32 s2, 32, null     -> s2 = scc ? 32 : 0
        sop2(0x27, NULL, 8, 255, lit=0x30008),              # s_bfe_u32 null, s8, 0x30008    -> scc = ((s8 >> 8) & 7) != 0
        sopp(0x04, 2),                                      # s_cbranch_scc0 +2 dwords (skips the 8-byte literal add)
        sop2(0x00, 2, 2, 255, lit=0x40),                    # s_add_u32 s2, s2, 0x40         -> s2 += 64 when scc
        sop2(0x10, M0, NULL, 2),                            # s_or_b32 m0, null, s2          -> m0 = 0 | s2
        smem(0x0a, 20, 16, soffset=M0),                     # s_buffer_load_dwordx4 s[20:23], s[16:19], m0
        vop1(0x01, 4, 20),
        vop1(0x01, 5, 21),
        vop1(0x01, 6, 22),
        vop1(0x01, 7, 23),
        struct.pack('<3I', 0xf80008cf, 0x07060504, 0xbf810000),
    ))
    ps = struct.pack('<3I', 0xf800180f, 0, 0xbf810000)
    return G.Pipeline(vs, ps, root_with_one_constant())


def expected_root_null(s8, *, drop_and_scc=False, clobber_mov_scc=False, invert_bfe_scc=False, bad_null_src=False):
    scc1 = False if drop_and_scc else ((s8 & 4) != 0)
    if clobber_mov_scc:
        scc1 = True
    null_val = 16 if bad_null_src else 0
    s2 = 32 if scc1 else null_val
    bfe_nz = (((s8 >> 8) & 7) != 0)
    scc2 = (not bfe_nz) if invert_bfe_scc else bfe_nz
    if scc2:
        s2 = (s2 + 64) & MASK
    addr = (null_val | s2) & MASK
    quad = list(struct.unpack_from('<4I', DATA, addr))
    return quad * 16 + [SENTINEL] * (320 - 64)


def null_sample_pipeline(nonzero=False, store=False):
    from startup_image_sample_a16_check import DIM_2D, IMG_SAMPLE_LZ, encode_mimg

    # A hand-written zero image descriptor, two zero coordinates, then add 0.25 to red.
    body = b''.join(sop1(4, reg, 128) for reg in range(64, 72, 2))
    if nonzero:
        body += sop1(3, 69, 137)
    body += vop1(1, 20, 128) + vop1(1, 21, 128)
    body += encode_mimg(IMG_SAMPLE_LZ, dim=DIM_2D, vaddr=20, vdata=12, srsrc=64, ssamp=84)
    body += struct.pack('<I', 0xbf8c3f70)
    body += vop2(3, 12, 255, 12) + struct.pack('<f', 0.25)
    if store:
        body += encode_mimg(8, dim=DIM_2D, vaddr=20, vdata=12, srsrc=64)
    return G.Pipeline(vs_words([(0, 0, 0)] * 4), body + POST_WORDS, root_blob())


def check_null_image():
    pipe = null_sample_pipeline()
    assert not pipe.resources
    assert null_sample_pipeline(store=True).hlsl_ps() == pipe.hlsl_ps()
    try:
        null_sample_pipeline(nonzero=True)
    except G.Unsupported as error:
        assert "descriptor operand s[64:71] holds ('const', 0x9)" in str(error), error
    else:
        raise AssertionError('nonzero image descriptor translated')


def draw_null_image(directory, dxc, renderer, adapter):
    pipe = null_sample_pipeline()
    source = pipe.hlsl_ps()
    vs = compile_hlsl(dxc, 'dxc', pipe.hlsl_vs(), 'vs_null_image', 'vs_6_0', directory)
    expected = words_text([G.fbits(0.25), 0, 0, 0])
    mutant = source.replace('v12 = 0u;', 'v12 = U(1.0);', 1)
    assert mutant != source
    for label, shader in (('null_image', source), ('null_returns_one', mutant)):
        ps = compile_hlsl(dxc, 'dxc', shader, label, 'ps_6_0', directory)
        result = draw(renderer, adapter, vs, ps, expected)
        if shader == source:
            assert result.returncode == 0, (result.stdout, result.stderr)
        else:
            assert result.returncode == 1 and 'actual=3fa00000,00000000,00000000,00000000' in result.stderr, result.stderr


def controls():
    check_null_image()
    # 1. Canonical encodings round-trip through disasm_rdna2.decode_instruction.
    for text, raw in CANONICAL_CASES:
        n, _, asm, _ = D.decode_instruction(raw, 0)
        check(f'disasm_rdna2 round-trip: {text}', n == len(raw) and asm == text)
    off, body_lines = 0, []
    while off < len(PS_BODY_WORDS):
        n, _, asm, _ = D.decode_instruction(PS_BODY_WORDS, off)
        body_lines.append(asm + '\n')
        off += n
    check('PS_BODY_WORDS matches PS_BODY_SOURCE', ''.join(body_lines) == PS_BODY_SOURCE)

    # 2. All 14 32-bit SCC-writing SCA ops into null emit only `scc = ...;` and match ISA SCC across boundary pairs.
    for op, arity, ref_scc in SCC_32_OPS:
        st = ProbeStage()
        asm = f'{op} null, s0' if arity == 1 else f'{op} null, s0, s1'
        st.one(op, asm)
        check(f'{op} null: emits single SCC assignment and leaves SGPRs unwritten',
              len(st.lines) == 1 and st.lines[0].startswith('scc = ') and st.scc_uniform)
        seen = set()
        mismatches = 0
        for a, b in SCC_PAIRS:
            got = eval_scc_line(st.lines[0], a, b)
            want = ref_scc(a, b)
            seen.add(got)
            if got != want:
                mismatches += 1
        check(f'{op} null: {len(SCC_PAIRS)} boundary pairs match ISA SCC (reaches both True and False)',
              mismatches == 0 and seen == {False, True})

    # 3. 32-bit non-SCC ops into null validate sources, emit no assignment, and leave SCC / scc_uniform untouched.
    for op, asm in (
        ('s_mov_b32', 's_mov_b32 null, s0'),
        ('s_movk_i32', 's_movk_i32 null, 0x100'),
        ('s_brev_b32', 's_brev_b32 null, s0'),
        ('s_mul_i32', 's_mul_i32 null, s0, s1'),
        ('s_bfm_b32', 's_bfm_b32 null, s0, s1'),
    ):
        st = ProbeStage()
        st.scc_uniform = False
        st.one(op, asm)
        check(f'{op} into null: emits nothing and preserves scc_uniform',
              st.lines == [] and st.scc_uniform is False)

    # 4. null (code 125) as a 32-bit source reads as 0x00000000u, folds constants, and preserves UNIFORM_VALU.
    st_fold = ProbeStage()
    st_fold.one('s_brev_b32', 's_brev_b32 s2, null')
    st_fold.one('s_bfm_b32', 's_bfm_b32 s3, 8, null')
    st_fold.one('s_mov_b32', 's_mov_b32 s4, 1')
    st_fold.one('s_bitset0_b32', 's_bitset0_b32 s4, null')
    st_fold.one('s_bfe_i32', 's_bfe_i32 s6, null, 0x40000')
    check('null source folds in s_brev_b32, s_bfm_b32, s_bitset0_b32 and s_bfe_i32',
          st_fold.sym.get(2) == ('const', 0)
          and st_fold.sym.get(3) == ('const', 0xff)
          and st_fold.sym.get(4) == ('const', 0)
          and st_fold.sym.get(6) == ('const', 0))

    def run_stage(*instructions, varying=()):
        s = G.Stage.__new__(G.Stage)
        s.pipe, s.kind, s.lines = DummyPipe(), 'ps', []
        s.attrs, s.exports, s.write_masks = {}, {}, {}
        s.sym, s.used_s, s.used_v, s.masks = {}, set(), set(), set()
        s.uniform_v, s.varying_s, s.scc_uniform = set(), set(varying), True
        s.entry_uniform_s = {2}
        s.exec_known = True
        s.ins = [(4 * i, 4, asm.split()[0], asm) for i, asm in enumerate(instructions)]
        s.translate()
        return [line for _, lines, _ in s.blocks for line in lines]

    u_lines = run_stage(
        's_and_b32 null, s2, 4',
        's_cselect_b32 s3, 10, null',
        'v_cvt_pk_u16_u32 v7, null, s3',
        'v_readfirstlane_b32 s9, v7',
    )
    check('uniform s_and_b32 null + null source propagates to v_readfirstlane_b32',
          u_lines[-1] == 's9 = v7;')
    try:
        run_stage(
            'v_cmp_lt_f32_e32 vcc_lo, v0, v1',
            's_and_b32 m0, vcc_lo, -1',
            's_cselect_b32 s8, 4, 0',
            's_cmp_eq_u32 s2, s2',
            's_and_b32 null, s8, 4',
            's_cselect_b32 s3, 10, null',
            'v_mov_b32_e32 v7, s3',
            'v_readfirstlane_b32 s9, v7',
        )
    except G.Unsupported as exc:
        check('varying s_and_b32 null clears scc_uniform and refuses v_readfirstlane_b32',
              'proven uniform source' in str(exc))
    else:
        raise AssertionError('varying s_and_b32 null unexpectedly accepted by v_readfirstlane_b32')

    # ComputePipeline lowering with s_and_b32 null + s_cbranch_scc0 + VALU null source.
    cs_blob = (
        sop2(0x0e, NULL, 4, 128 + 4)
        + sopp(0x04, 1)
        + vop2(0x25, 1, NULL, 0)
        + sopp(0x01, 0)
    )
    cs_hlsl = G.ComputePipeline(cs_blob, root_blob(), (4 << 1) | (1 << 7), (1, 1, 1)).hlsl_cs()
    check('ComputePipeline: s_and_b32 null + s_cbranch_scc0 + v_add_nc_u32 null lowers cleanly',
          'scc = (s4 & 0x00000004u) != 0;' in cs_hlsl and '0x00000000u + v0' in cs_hlsl)

    # 5. Complete TypeError elimination & explicit Unsupported refusal sweep across all SCA ops + s_bfm_b32.
    bad_dests = ('vcc_hi', 'exec_hi', 'ttmp0')
    for op in sorted(G.SCA):
        arity = 1 if op in ('s_mov_b32', 's_mov_b64', 's_movk_i32', 's_not_b32', 's_not_b64',
                            's_brev_b32', 's_sext_i32_i8', 's_sext_i32_i16') else 2
        src_a = '0x10' if op == 's_movk_i32' else ('s[0:1]' if op.endswith('_b64') else 's0')
        src_b = 's[2:3]' if op.endswith('_b64') else 's1'
        rhs = src_a if arity == 1 else f'{src_a}, {src_b}'
        for dst in bad_dests:
            for use_desc in (False, True):
                st = ProbeStage()
                if use_desc and not op.endswith('_b64') and op != 's_movk_i32':
                    st.sym[0] = ('desc', ('table', 0), 16, 0)
                try:
                    st.one(op, f'{op} {dst}, {rhs}')
                except G.Unsupported as exc:
                    assert dst in str(exc) or 'descriptor' in str(exc) or 'lane mask' in str(exc), (op, dst, exc)
                except Exception as exc:
                    raise AssertionError(f'{op} into {dst} (desc={use_desc}) raised {type(exc).__name__}: {exc}')
                else:
                    raise AssertionError(f'{op} into {dst} unexpectedly accepted: {st.lines}')

    for dst in ('vcc_lo', 'vcc_hi', 'exec_lo', 'exec_hi', 'ttmp0'):
        st = ProbeStage()
        try:
            st.one('s_bfm_b32', f's_bfm_b32 {dst}, s0, s1')
        except G.Unsupported as exc:
            assert f's_bfm_b32 destination {dst}' == str(exc), exc
        else:
            raise AssertionError(f's_bfm_b32 into {dst} accepted')
    check(f'all {len(G.SCA)} SCA ops + s_bfm_b32 refuse unmodelled non-SGPR destinations without TypeError', True)

    # Specific null-destination refusals: _b64, lane-mask sources (#2618 wave64 pattern), SCC_UNMODELLED, descriptors.
    specific_refusals = (
        ('s_and_b64', 's_and_b64 null, exec, s[0:1]', 's_and_b64 into null on a lane mask'),
        ('s_or_b64', 's_or_b64 null, vcc, s[0:1]', 's_or_b64 into null on a lane mask'),
        ('s_and_b64', 's_and_b64 null, s[0:1], s[2:3]', 's_and_b64 into null'),
        ('s_mov_b64', 's_mov_b64 null, s[0:1]', 's_mov_b64 into null'),
        ('s_and_b32', 's_and_b32 null, exec_lo, s0', 's_and_b32 into null on a lane mask'),
        ('s_or_b32', 's_or_b32 null, vcc_lo, s0', 's_or_b32 into null on a lane mask'),
        ('s_nand_b32', 's_nand_b32 null, vcc_lo, exec_lo', 's_nand_b32 on a lane mask'),
        ('s_bfm_b32', 's_bfm_b32 null, vcc_lo, 4', 's_bfm_b32 into null on a lane mask'),
        ('s_and_b32', 's_and_b32 null, s16, 0xff', 'scc from descriptor data [null destination: s_and_b32, destination null]'),
        ('s_add_u32', 's_add_u32 null, s10, 4', 'scc from descriptor data [null destination: s_add_u32, destination null]'),
        ('s_mov_b32', 's_mov_b32 null, s16', 's_mov_b32 into null on descriptor data'),
    )
    for op, asm, expected_reason in specific_refusals:
        st = ProbeStage()
        try:
            st.one(op, asm)
        except G.Unsupported as exc:
            check(f'refuses {asm} ({exc})', str(exc) == expected_reason and not st.lines)
        else:
            raise AssertionError(f'{asm} unexpectedly accepted: {st.lines}')

    # A null destination keeps only the SCC the operation computes (no register is written, nothing else is emitted).
    for asm, scc_line in (
            ('s_add_u32 null, s0, s1', 'scc = (s0 + s1) < (s0);'),
            ('s_sub_u32 null, s0, s1', 'scc = (s0) < (s1);'),
            ('s_add_i32 null, s0, s1', 'scc = (((((s0) ^ (s0 + s1)) & ((s1) ^ (s0 + s1))) >> 31) != 0);'),
            ('s_sub_i32 null, s0, s1', 'scc = (((((s0) ^ (s1)) & ((s0) ^ (s0 - s1))) >> 31) != 0);'),
            ('s_min_u32 null, s0, s1', 'scc = ((s0) <= (s1));'),
            ('s_max_u32 null, s0, s1', 'scc = ((s0) >= (s1));'),
            ('s_min_i32 null, s0, s1', 'scc = (int(s0) <= int(s1));'),
            ('s_max_i32 null, s0, s1', 'scc = (int(s0) >= int(s1));')):
        st = ProbeStage()
        st.one(asm.split()[0], asm)
        check(f'{asm} computes only its SCC {st.lines}', st.lines == [scc_line])

    # SGPR holding a compare mask must also be refused when combined into null.
    st_mask = ProbeStage()
    st_mask.masks.add(0)
    try:
        st_mask.one('s_and_b32', 's_and_b32 null, s0, s1')
    except G.Unsupported as exc:
        check('s_and_b32 null with SGPR compare mask refused', str(exc) == 's_and_b32 into null on a lane mask')
    else:
        raise AssertionError('accepted SGPR compare mask into null')

    # 6. Unlowered negative control: reverting Stage.u('null') restores the pre-change refusal.
    real_u = G.Stage.u

    def no_null_u(self, tok, *, low16_only=False):
        if tok == 'null':
            raise G.Unsupported("operand 'null'")
        return real_u(self, tok, low16_only=low16_only)

    G.Stage.u = no_null_u
    try:
        try:
            G.Pipeline(vs_words(PS_DRAW_INPUTS[:4]), PS_WORDS, root_blob())
        except G.Unsupported as exc:
            check("unlowered control: removing null from Stage.u restores operand 'null' refusal",
                  str(exc) == "operand 'null'")
        else:
            raise AssertionError('PS_WORDS translated with Stage.u null handling disabled')
    finally:
        G.Stage.u = real_u


def check_binary(llvm_mc):
    for text, expected in CANONICAL_CASES:
        built = assemble(llvm_mc, text + '\n')
        assert built == expected, f'{text}: assembled {built.hex()}, recorded {expected.hex()}'
    assert assemble(llvm_mc, PS_BODY_SOURCE) == PS_BODY_WORDS, 'PS_BODY_WORDS differs from llvm-mc'
    for name, source, raw in (('prologue', PRE_SOURCE, PRE_WORDS), ('export', POST_SOURCE, POST_WORDS)):
        assert assemble(llvm_mc, source) == raw, f'{name} fixture differs from llvm-mc'
    for idx in range(0, len(PS_DRAW_INPUTS), 4):
        chunk = PS_DRAW_INPUTS[idx:idx + 4]
        assert assemble(llvm_mc, vs_source(chunk)) == vs_words(chunk), chunk
    print('llvm-mc: all scalar/vector null cases and draw fixtures match recorded words')


def check_decode(library):
    from pixel_aot import disassemble
    blob = b''.join(raw for _, raw in CANONICAL_CASES)
    llvm_lines = disassemble(blob, library)
    ours = [text for text, _ in CANONICAL_CASES]
    assert llvm_lines == ours, f'LLVM vs disasm_rdna2 mismatch: {list(zip(llvm_lines, ours))}'
    print(f'libLLVM: {len(CANONICAL_CASES)} encodings with null match disasm_rdna2.py verbatim')


def run_root_case(directory, dxc, renderer, adapter, pipe, vs_hlsl, ref_words, *, s8=0, fxc=None):
    hlsl, binary, case = (directory / n for n in ('root_null.hlsl', 'root_null.cso', 'root_null.case'))
    constants = [0] * 16
    if ('rootconst', 0, 0) in pipe.uds:
        constants[pipe.uds.index(('rootconst', 0, 0))] = s8
    case.write_bytes(struct.pack('<337I', 1, *ref_words, *constants))
    hlsl.write_text(vs_hlsl, encoding='ascii')
    if fxc:
        ps_hlsl = directory / 'root_null_ps.hlsl'
        ps_hlsl.write_text(pipe.hlsl_ps(), encoding='ascii')
        subprocess.run([fxc, '/nologo', '/T', 'vs_5_1', '/E', 'main', '/Fo', str(directory / 'vs51.cso'), str(hlsl)],
                       check=True, capture_output=True, text=True)
        subprocess.run([fxc, '/nologo', '/T', 'ps_5_1', '/E', 'main', '/Fo', str(directory / 'ps51.cso'), str(ps_hlsl)],
                       check=True, capture_output=True, text=True)
    subprocess.run([dxc, '-T', 'vs_6_0', '-E', 'main', '-Fo', str(binary), str(hlsl)],
                   check=True, capture_output=True, text=True)
    wrapper = (
        '\nRWByteAddressBuffer outputWords : register(u0);\n'
        '[numthreads(16, 1, 1)]\n'
        'void readback(uint id : SV_GroupIndex) {\n'
        '    float4 p = body(id).pos;\n'
        '    outputWords.Store4(id * 16, asuint(p));\n'
        '}\n'
    )
    with hlsl.open('a', encoding='ascii') as handle:
        handle.write(wrapper)
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'readback', '-Fo', str(binary), str(hlsl)],
                   check=True, capture_output=True, text=True)
    return subprocess.run([renderer, adapter, '--compute-store-case', str(binary), str(case)],
                          capture_output=True, text=True)


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    ps_draws = ps_muts = 0
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        draw_null_image(directory, dxc, renderer, adapter)
        ps_text = G.Pipeline(vs_words([(0, 0, 0)] * 4), PS_WORDS, root_blob()).hlsl_ps()

        # Part A: Graphics VS+PS draw across FXC ps_5_1 and DXC ps_6_0 + three pixel-stage mutations.
        ps_mutations = (
            (
                'subrev_null_1',
                'v0 - 0x00000000u',
                'v0 - 0x00000001u',
                lambda c: [(c[0][0] - 1) & MASK, (c[1][0] ^ c[1][1]) & MASK, f32_add_zero(c[2][0]), (c[3][0] ^ 0x55) & MASK],
            ),
            (
                'f32_null_one',
                'F(0x00000000u) + F(v2)',
                '1.0 + F(v2)',
                lambda c: [c[0][0] & MASK, (c[1][0] ^ c[1][1]) & MASK, f32_add_zero(c[2][0], 0x3f800000), (c[3][0] ^ 0x55) & MASK],
            ),
            (
                'drop_null_scc',
                'scc = (s4 & 0x7f800000u) != 0;',
                'scc = false;',
                lambda c: [c[0][0] & MASK, (c[1][0] ^ c[1][1]) & MASK, f32_add_zero(c[2][0]), c[3][0] & MASK],
            ),
        )
        for kind, compiler, vs_profile, ps_profile in compilers:
            good_ps = compile_hlsl(compiler, kind, ps_text, f'ps_null_{kind}', ps_profile, directory)
            for idx in range(0, len(PS_DRAW_INPUTS), 4):
                chunk = PS_DRAW_INPUTS[idx:idx + 4]
                vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(chunk), PS_WORDS, root_blob()).hlsl_vs(),
                                  f'vs_null_{idx}_{kind}', vs_profile, directory)
                expected = words_text(ref_ps_chunk(chunk))
                run = draw(renderer, adapter, vs, good_ps, expected)
                if run.returncode == 77:
                    print(run.stdout, run.stderr)
                    sys.exit(77)
                if run.returncode:
                    raise AssertionError(f'{kind} PS null draw {idx}\n{run.stdout}\n{run.stderr}')
                ps_draws += 1
                print(f'{kind} PS null draw {idx // 4}: {expected}', flush=True)
                if idx:
                    continue
                for mut_name, target, repl, mut_fn in ps_mutations:
                    assert target in ps_text, (mut_name, ps_text)
                    bad_ps = compile_hlsl(compiler, kind, ps_text.replace(target, repl, 1),
                                          f'mut_ps_{mut_name}_{kind}', ps_profile, directory)
                    wrong = words_text(mut_fn(chunk))
                    mrun = draw(renderer, adapter, vs, bad_ps, expected)
                    if mrun.returncode == 0 or f'actual={wrong}' not in mrun.stderr:
                        raise AssertionError(f'{kind} mutation {mut_name} did not read {wrong}\n{mrun.stdout}\n{mrun.stderr}')
                    ps_muts += 1
                    print(f'{kind} PS mutation ({mut_name}): fails as expected, reads {wrong}', flush=True)

        # Part B: Runtime root-constant s8 through SALU null-destination SCC updates + branch + buffer load.
        root_pipe = build_root_null_pipeline()
        vs_hlsl = root_pipe.hlsl_vs()
        for s8 in (0x000, 0x004, 0x300, 0x504):
            ref = expected_root_null(s8)
            res = run_root_case(directory, dxc, renderer, adapter, root_pipe, vs_hlsl, ref,
                                s8=s8, fxc=fxc if s8 == 0 else None)
            if res.returncode == 77:
                print(res.stdout, res.stderr)
                sys.exit(77)
            if res.returncode:
                raise AssertionError(f'root null case s8={s8:#x}\n{res.stdout}\n{res.stderr}')
            print(f'root null SALU+branch (s8={s8:#x}):',
                  [l for l in res.stdout.splitlines() if 'mismatches=' in l][0], flush=True)

        # Four HLSL mutations on the runtime root-constant shader must fail GPU readback.
        ref_504 = expected_root_null(0x504)
        ref_300 = expected_root_null(0x300)
        and_mut, n1 = re.subn(r'scc = \(ud\d+ & 0x00000004u\) != 0;', 'scc = false;', vs_hlsl, count=1)
        clob_mut, n2 = re.subn(r's2 = scc \? 0x00000020u : 0x00000000u;',
                               'scc = true; s2 = scc ? 0x00000020u : 0x00000000u;', vs_hlsl, count=1)
        bfe_mut, n3 = re.subn(r'(scc = \(\(ud\d+ >> \(0x00030008u & 31\)\).*?)\) != 0;', r'\1) == 0;', vs_hlsl, count=1)
        src_mut, n4 = re.subn(r'm0 = 0x00000000u \| s2;', 'm0 = 0x00000010u | s2;', vs_hlsl, count=1)
        assert (n1, n2, n3, n4) == (1, 1, 1, 1), (n1, n2, n3, n4, vs_hlsl)
        root_mutants = (
            ('dropped s_and_b32 null SCC write', and_mut, 0x504, ref_504),
            ('no-SCC null discard clobbering SCC', clob_mut, 0x300, ref_300),
            ('inverted s_bfe_u32 null SCC write', bfe_mut, 0x504, ref_504),
            ('non-zero null source in s_or_b32', src_mut, 0x504, ref_504),
        )
        for label, mutant_hlsl, s8_val, good_ref in root_mutants:
            assert mutant_hlsl != vs_hlsl, (label, vs_hlsl)
            fail = run_root_case(directory, dxc, renderer, adapter, root_pipe, mutant_hlsl, good_ref, s8=s8_val)
            if fail.returncode != 1 or 'Compute readback differs' not in fail.stderr:
                raise AssertionError(f'root mutation {label!r} escaped detection:\n{fail.stdout}\n{fail.stderr}')
            print(f'root mutation ({label}): fails GPU readback as expected', flush=True)

    print(f'{adapter}: {ps_draws} PS draws + 4 root-constant runs matched; {ps_muts} PS + 4 root mutations caught')


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary' and len(args) == 1:
        check_binary(args[0])
    elif mode == 'decode' and len(args) == 1:
        check_decode(args[0])
    elif mode == 'draw' and len(args) in (3, 4):
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
