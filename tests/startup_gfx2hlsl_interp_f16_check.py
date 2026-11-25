"""v_interp_p1ll_f16 / v_interp_p2_f16 in tools/startup/gfx2hlsl.py (pixel stage, issue #1996 item C).

  (no arguments)                       translation controls and mutations, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  compile <dxc>                        compile the translated pair with DXC vs_6_0 / ps_6_0 (no rendering)
  draw <dxc> <offscreen_check> <warp|hardware>
                                       draw the compr pair and the plain-export pair (both halves) and compare the four result words (Windows, D3D12)

Semantics (LLVM 15 AMDGPU, VOP3Instructions.td and IntrinsicsAMDGPU.td; Mesa aco emit_interp_instr):
  v_interp_p1ll_f16 vdst, vI, attrN.c [high]   vdst(f32) = P0 + I * P10 of the binary16 half `high` of channel c
  v_interp_p2_f16   vdst, vJ, attrN.c, vP1 [high]   vdst(f16) = P1 + J * P20 (same attribute half)
The pair gives the interpolated binary16 value. The vertex stage exports it with `exp paramN ... compr`: two binary16
values per dword, channel x = (v6.lo, v6.hi), channel y = (v7.lo, v7.hi). gfx2hlsl splits the pair into four float
channels, so the half h of channel c is float channel 2 * c + h.

The fixture is a source-built gfx1013 vertex and pixel pair. The vertex stage places a full-screen triangle and exports
the constant binary16 values 1.5, -2.0 (channel x), 0.5 and 3.0 (channel y) to param0; the pixel stage interpolates
the four halves, converts each to f32 and exports them to mrt0. The reference is written from the ISA layout above
with struct ('e' format) conversions, not from the lowering: the translated text is evaluated on the CPU by a small
interpreter of exactly the statements the fixture produces. Negative controls: the lowering removed (the pair is
refused), the high bit inverted on both instructions (the evaluated words differ from the reference), and every
modifier or form the fixture does not cover (refused),
including every op_sel bit (ACO's v_interp_p2_hi_f16 writes the high half of the destination).

Plain export (follow-up of #2052): real programs pack the binary16 halves into a plain 32-bit export, e.g. with
v_cvt_pkrtz_f16_f32, and read all four channels. Half h of 32-bit channel c is f16tof32(word_c >> 16 * h) (the low
16 bits for h = 0), which is not the compr mapping (attrN.x = (param.x, param.y) of a compr export). The fixture packs
eight binary16 values into four words (`exp param0 v6, v7, v8, v9`), two pixel stages read the low and the high halves
of channels x..w. Controls: the compr fixture with its export flipped to plain reads other words than the compr
reference, the high bit inverted on the plain fixture reads the other halves, an attribute exported both ways and a
read of a channel exported as `off` are refused.

The binary mode compares with llvm-mc after clearing the unused src2 field of v_interp_p1ll_f16: LLVM 15 encodes it
as 0, newer llvm-mc encodes the inline constant 0 (128), and the words below are the LLVM 15 ones.
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import VS_WORDS as F32_VS_WORDS, assemble, compile_hlsl, draw, root_blob  # noqa: E402

VS_SOURCE = """\
v_mov_b32 v3, 0x40400000
v_mov_b32 v10, 0xc0400000
v_cmp_eq_u32 vcc_lo, 1, v0
v_cndmask_b32 v1, -1.0, v3, vcc_lo
v_cmp_eq_u32 vcc_lo, 2, v0
v_cndmask_b32 v2, 1.0, v10, vcc_lo
v_mov_b32 v4, 1.0
v_mov_b32 v5, 0
v_mov_b32 v6, 0xc0003e00
v_mov_b32 v7, 0x42003800
exp pos0 v1, v2, v5, v4 done
exp param0 v6, v6, v7, v7 compr
s_endpgm
"""
VS_WORDS = bytes.fromhex(
    'ff02067e00004040ff02147e000040c08100847df30602028200847df2140402f202087e80020a7eff020c7e003e00c0'
    'ff020e7e00380042cf0800f8010205040f0600f806070000000081bf')
PS_SOURCE = """\
s_mov_b32 m0, s2
v_interp_p1ll_f16 v4, v0, attr0.x
v_interp_p2_f16 v4, v1, attr0.x, v4
v_interp_p1ll_f16 v5, v0, attr0.x high
v_interp_p2_f16 v5, v1, attr0.x, v5 high
v_interp_p1ll_f16 v6, v0, attr0.y
v_interp_p2_f16 v6, v1, attr0.y, v6
v_interp_p1ll_f16 v7, v0, attr0.y high
v_interp_p2_f16 v7, v1, attr0.y, v7 high
v_cvt_f32_f16 v4, v4
v_cvt_f32_f16 v5, v5
v_cvt_f32_f16 v6, v6
v_cvt_f32_f16 v7, v7
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
PS_WORDS = bytes.fromhex(
    '0203fcbe040042d70000020004005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04'
    '070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf')
# Variants of PS_SOURCE for the barycentric, omitted-result and upper-half refusals. The words are llvm-mc output
# (src2 of p1ll cleared as for PS_WORDS), verified by `binary`; the sources are in PS_VARIANT_SOURCES.
V_BODY = PS_SOURCE.split('\n', 1)[1]
V_FIRST = 'v_interp_p1ll_f16 v4, v0, attr0.x\nv_interp_p2_f16 v4, v1, attr0.x, v4\n'
V_TAIL = 'exp mrt0 v4, v4, v4, v4 done vm\ns_endpgm\n'
PS_VARIANT_SOURCES = {
    'I clobbered': 's_mov_b32 m0, s2\nv_mov_b32 v0, 1.0\n' + V_BODY,
    'J clobbered': 's_mov_b32 m0, s2\nv_mov_b32 v1, 1.0\n' + V_BODY,
    'I and J swapped': ('s_mov_b32 m0, s2\n' + V_BODY).replace('v0,', 'vI,').replace('v1,', 'v0,').replace('vI,', 'v1,'),
    'p1ll result read between the pair':
        's_mov_b32 m0, s2\nv_interp_p1ll_f16 v4, v0, attr0.x\nv_mov_b32 v8, v4\n'
        'v_interp_p2_f16 v4, v1, attr0.x, v4\n' + V_BODY.split('\n', 2)[2],
    'p1ll result read after p2 wrote elsewhere':
        's_mov_b32 m0, s2\nv_interp_p1ll_f16 v4, v0, attr0.x\nv_interp_p2_f16 v8, v1, attr0.x, v4\n'
        'v_cvt_f32_f16 v8, v8\nv_mov_b32 v9, v4\nexp mrt0 v8, v9, v9, v9 done vm\ns_endpgm\n',
    'upper half read by a shift': 's_mov_b32 m0, s2\n' + V_FIRST + 'v_lshrrev_b32 v8, 16, v4\n'
        'v_cvt_f32_f16 v4, v4\nexp mrt0 v4, v8, v8, v8 done vm\ns_endpgm\n',
    'upper half exported': 's_mov_b32 m0, s2\n' + V_FIRST + V_TAIL,
    'branch into p2': 's_mov_b32 m0, s2\ns_cbranch_vccz L0\nv_interp_p1ll_f16 v4, v0, attr0.x\nL0:\n'
        'v_interp_p2_f16 v4, v1, attr0.x, v4\nv_cvt_f32_f16 v4, v4\n' + V_TAIL,
    'unrelated read of v2': 's_mov_b32 m0, s2\nv_mov_b32 v8, v2\n' + V_BODY,
    'unrelated write of v9': 's_mov_b32 m0, s2\nv_mov_b32 v9, 1.0\n' + V_BODY,
}
PS_VARIANTS = {name: bytes.fromhex(h) for name, h in (
    ('I clobbered', '0203fcbef202007e040042d70000020004005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
    ('J clobbered', '0203fcbef202027e040042d70000020004005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
    ('I and J swapped', '0203fcbe040042d70002020004005ad700001204050042d70003020005005ad700011604060042d74002020006005ad740001a04070042d74003020007005ad740011e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
    ('p1ll result read between the pair', '0203fcbe040042d7000002000403107e04005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
    ('p1ll result read after p2 wrote elsewhere', '0203fcbe040042d70000020008005ad7000212040817107e0403127e0f1800f808090909000081bf'),
    ('upper half read by a shift', '0203fcbe040042d70000020004005ad7000212049008102c0417087e0f1800f804080808000081bf'),
    ('upper half exported', '0203fcbe040042d70000020004005ad7000212040f1800f804040404000081bf'),
    ('branch into p2', '0203fcbe020086bf040042d70000020004005ad7000212040417087e0f1800f804040404000081bf'),
    ('unrelated read of v2', '0203fcbe0203107e040042d70000020004005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
    ('unrelated write of v9', '0203fcbef202127e040042d70000020004005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
)}
# Plain-export fixture: the vertex stage packs 8 binary16 values with v_cvt_pkrtz_f16_f32 (inputs are the f32 values
# below) and exports the four words without compr; PLAIN_PS_LO / PLAIN_PS_HI read half 0 / half 1 of channels x..w.
PLAIN_VALUES = ((1.5, -2.0), (0.5, 3.0), (4.0, -0.25), (8.0, 0.125))  # (low, high) half of each word
_VS_HEAD = VS_SOURCE.split('v_mov_b32 v6')[0]
_VS_PACK = ''.join('v_mov_b32 v%d, 0x%08x\n' % (11 + i, c) for i, c in enumerate(
    (0x3fc00000, 0xc0000000, 0x3f000000, 0x40400000, 0x40800000, 0xbe800000, 0x41000000, 0x3e000000))) + ''.join(
    'v_cvt_pkrtz_f16_f32 v%d, v%d, v%d\n' % (6 + i, 11 + 2 * i, 12 + 2 * i) for i in range(4))


def plain_vs_source(exports):
    return _VS_HEAD + _VS_PACK + 'exp pos0 v1, v2, v5, v4 done\n' + exports + 's_endpgm\n'


def plain_ps_source(high):
    body = ''.join('v_interp_p1ll_f16 v%d, v0, attr0.%s%s\nv_interp_p2_f16 v%d, v1, attr0.%s, v%d%s\n'
                   % (r, c, high, r, c, r, high) for r, c in zip((4, 5, 6, 7), 'xyzw'))
    return ('s_mov_b32 m0, s2\n' + body + ''.join('v_cvt_f32_f16 v%d, v%d\n' % (r, r) for r in (4, 5, 6, 7))
            + 'exp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n')


PLAIN_VS_SOURCES = {
    'plain': 'exp param0 v6, v7, v8, v9\n',
    'off': 'exp param0 v6, v7, v8, off\n',
    'mixed': 'exp param0 v6, v6, v7, v7 compr\nexp param0 v6, v7, v8, v9\n',
}
PLAIN_VS = {name: bytes.fromhex(h) for name, h in (
    ('plain', 'ff02067e00004040ff02147e000040c08100847df30602028200847df2140402f202087e80020a7eff02167e0000c03ff502187ef0021a7eff021c7e00004040f6021e7eff02207e000080beff02227e00000041ff02247e0000003e0b190c5e0d1d0e5e0f21105e1125125ecf0800f8010205040f0200f806070809000081bf'),
    ('off', 'ff02067e00004040ff02147e000040c08100847df30602028200847df2140402f202087e80020a7eff02167e0000c03ff502187ef0021a7eff021c7e00004040f6021e7eff02207e000080beff02227e00000041ff02247e0000003e0b190c5e0d1d0e5e0f21105e1125125ecf0800f801020504070200f806070800000081bf'),
    ('mixed', 'ff02067e00004040ff02147e000040c08100847df30602028200847df2140402f202087e80020a7eff02167e0000c03ff502187ef0021a7eff021c7e00004040f6021e7eff02207e000080beff02227e00000041ff02247e0000003e0b190c5e0d1d0e5e0f21105e1125125ecf0800f8010205040f0600f8060700000f0200f806070809000081bf'),
)}
PLAIN_PS = {name: bytes.fromhex(h) for name, h in (
    ('lo', '0203fcbe040042d70000020004005ad700021204050042d74000020005005ad740021604060042d78000020006005ad780021a04070042d7c000020007005ad7c0021e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
    ('hi', '0203fcbe040042d70001020004005ad700031204050042d74001020005005ad740031604060042d78001020006005ad780031a04070042d7c001020007005ad7c0031e040417087e05170a7e06170c7e07170e7e0f1800f804050607000081bf'),
)}
P1LL_SRC2_MASK = 0x07fc0000  # w1 bits 26:18, the unused third source

# The reference, from the ISA layout: two binary16 values per dword of the compr export.
EXPORT = (0xc0003e00, 0x42003800)  # v6, v7 of VS_SOURCE
HALVES = ((1.5, -2.0), (0.5, 3.0))


def f16_bits(x):
    return struct.unpack('<H', struct.pack('<e', x))[0]


def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def reference():
    """The four words the pixel stage exports: half h of channel c of the attribute, as f32."""
    for c in range(2):
        assert EXPORT[c] == f16_bits(HALVES[c][0]) | f16_bits(HALVES[c][1]) << 16
    words = []
    for c, h in ((0, 0), (0, 1), (1, 0), (1, 1)):
        half = EXPORT[c] >> (16 * h) & 0xffff
        words.append(f32_bits(struct.unpack('<e', struct.pack('<H', half))[0]))
    return tuple(words)


def f16_to_f32_bits(h):
    return f32_bits(struct.unpack('<e', struct.pack('<H', h))[0])


PLAIN_WORDS = tuple(f16_bits(lo) | f16_bits(hi) << 16 for lo, hi in PLAIN_VALUES)  # v6..v9 of the plain fixture


def plain_reference(half):
    """The four words of a plain-fixture pixel stage: half `half` of channels x..w, as f32 (from the ISA layout)."""
    return tuple(f16_to_f32_bits(word >> (16 * half) & 0xffff) for word in PLAIN_WORDS)


def evaluate(vs, ps, channels=None, regs=None):
    """Evaluate the translated text: the vertex compr export, then the pixel conversions. Only the statements
    the fixture produces are understood; anything else fails the check. `channels` replaces the vertex export with
    four interpolated f32 words (what the rasterizer hands the pixel stage)."""
    regs = {'v6': EXPORT[0], 'v7': EXPORT[1]} if regs is None else regs
    if channels is None:
        m = re.search(r'o\.param0 = float4\(([^;]*)\);\n', vs)
        channels = []
        for expr in re.findall(r'f16tof32\((v\d+)( >> 16)?\)', m.group(1)):
            channels.append(f16_to_f32_bits((regs[expr[0]] >> (16 if expr[1] else 0)) & 0xffff))
        assert len(channels) == 4 or 'F(v' in m.group(1), m.group(0)  # a plain export has no compr channels
    # The plain export's extra outputs: o.hparam0_lo / _hi = float4(f16tof32(vN [>> 16]) or 0.0, ...)
    halves = {}
    for m in re.finditer(r'o\.hparam0_(lo|hi) = float4\(([^;]*)\);\n', vs):
        args = re.findall(r'f16tof32\((v\d+)( >> 16)?\)|(0\.0)', m.group(2))
        assert len(args) == 4, m.group(0)
        halves[m.group(1)] = [0 if zero else f16_to_f32_bits((regs[r] >> (16 if sh else 0)) & 0xffff)
                              for r, sh, zero in args]
    state = {}
    for line in ps.splitlines():
        stmt = re.fullmatch(r'    (v\d+) = \((v\d+) & 0xffff0000u\) \| f32tof16\(i\.(param0|hparam0_lo|hparam0_hi)\.([xyzw])\);',
                            line)
        if stmt:
            assert stmt.group(1) == stmt.group(2), line
            # f32tof16 of the interpolated value; a constant triangle makes it the exported value
            source = channels if stmt.group(3) == 'param0' else halves[stmt.group(3)[-2:]]
            f32 = source['xyzw'.index(stmt.group(4))]
            state[stmt.group(1)] = f16_bits(struct.unpack('<f', struct.pack('<I', f32))[0])
            continue
        stmt = re.fullmatch(r'    (v\d+) = U\(f16tof32\((v\d+)\)\);', line)
        if stmt:
            state[stmt.group(1)] = f16_to_f32_bits(state[stmt.group(2)] & 0xffff)
    return tuple(state['v%d' % n] for n in (4, 5, 6, 7))


def translate(ps_words=PS_WORDS, vs_words=VS_WORDS, **kwargs):
    return G.Pipeline(vs_words, ps_words, root_blob(), **kwargs)


def patch(words, index, w0=0, w1=0):
    """Instruction `index` of PS_WORDS (after the 4-byte s_mov_b32): w0 / w1 are XORed into its two dwords."""
    out = bytearray(words)
    off = 4 + 8 * index
    a, b = struct.unpack_from('<II', out, off)
    struct.pack_into('<II', out, off, a ^ w0, b ^ w1)
    return bytes(out)


def refused(label, words, needle, vs_words=VS_WORDS, **kwargs):
    try:
        translate(words, vs_words, **kwargs)
    except G.Unsupported as error:
        assert needle in str(error), (label, str(error))
        print('refused: %s (%s)' % (label, error))
        return
    except Exception as error:
        raise AssertionError('%s crashed instead of being refused: %r' % (label, error)) from error
    raise AssertionError('%s was translated' % label)


def controls():
    expected = reference()
    assert expected == (0x3fc00000, 0xc0000000, 0x3f000000, 0x40400000), expected
    pipe = translate()
    vs, ps = pipe.hlsl_vs(), pipe.hlsl_ps()
    got = evaluate(vs, ps)
    assert got == expected, ('translated words', [hex(w) for w in got], [hex(w) for w in expected])
    print('translated: %s' % ','.join('%08x' % w for w in got))

    # The p2 lowering keeps the upper half of the destination with this mask; dropping it must fail the check.
    assert ps.count('& 0xffff0000u) | f32tof16(') == 4, ps
    # The whole-program scan runs once per translated program, not once per p1ll / p2 statement.
    calls = []
    scan = G.Stage.interp_f16_scan

    def counting(self):
        calls.append(self.interp_scanned)
        return scan(self)

    G.Stage.interp_f16_scan = counting
    try:
        translate()
    finally:
        G.Stage.interp_f16_scan = scan
    assert calls.count(False) == 1 and len(calls) == 8, calls
    print('scan: %d calls, 1 full run' % len(calls))

    # A decoded instruction must be the source line (LLVM's spelling).
    lines = PS_SOURCE.splitlines()
    for (_, size, name, asm), line in zip(G.decode(G.strip_header(PS_WORDS)), lines):
        assert asm.replace('_e32', '') == line, (asm, line)

    # Negative control 1: the lowering removed. The pair must be refused, not translated by some fallback.
    saved = {n: getattr(G.Stage, n) for n in ('op_v_interp_p1ll_f16', 'op_v_interp_p2_f16')}
    try:
        for n in saved:
            delattr(G.Stage, n)
        refused('lowering disabled', PS_WORDS, 'v_interp_p1ll_f16')
    finally:
        for n, f in saved.items():
            setattr(G.Stage, n, f)

    # Negative control 2: the high bit inverted on every instruction of the pair. The pair stays consistent, so it
    # is translated, but it reads the other half: the evaluated words differ from the reference.
    flipped = PS_WORDS
    for index in range(8):
        flipped = patch(flipped, index, w1=0x100)  # w1 bit 8: attribute half select
    pipe = translate(flipped)
    swapped = evaluate(pipe.hlsl_vs(), pipe.hlsl_ps())
    assert swapped == (expected[1], expected[0], expected[3], expected[2]), [hex(w) for w in swapped]
    assert swapped != expected
    print('negative control: high inverted reads %s' % ','.join('%08x' % w for w in swapped))
    # ... and inverting only one of the pair is refused.
    refused('p2 high differs from p1ll', patch(PS_WORDS, 1, w1=0x100), 'is not the v_interp_p1ll_f16 result')

    # What the fixture does not cover is refused.
    refused('p1ll clamp', patch(PS_WORDS, 0, w0=0x8000), 'clamp')
    refused('p1ll div:2', patch(PS_WORDS, 0, w1=3 << 27), 'div:2')
    refused('p1ll mul:2', patch(PS_WORDS, 0, w1=1 << 27), 'mul:2')
    for bit in range(11, 15):
        refused('p1ll op_sel bit %d' % (bit - 11), patch(PS_WORDS, 0, w0=1 << bit), 'op_sel')
        refused('p2 op_sel bit %d' % (bit - 11), patch(PS_WORDS, 1, w0=1 << bit), 'op_sel')
    refused('p2 clamp', patch(PS_WORDS, 1, w0=0x8000), 'clamp')
    refused('p2 abs on j', patch(PS_WORDS, 1, w0=0x200), 'plain VGPRs')
    refused('p2 neg on p1', patch(PS_WORDS, 1, w1=4 << 29), 'plain VGPRs')
    refused('p1ll abs on i', patch(PS_WORDS, 0, w0=0x200), 'plain VGPRs')
    # attr0.z: change channel bits 7:6 of the p1ll and p2 from x (0) to z (2)
    z = patch(patch(PS_WORDS, 0, w1=2 << 6), 1, w1=2 << 6)
    refused('attribute channel z', z, 'channel z')
    # An attribute spelled without a channel is refused, not an AttributeError (decode always prints a channel,
    # so the instruction goes straight to a Stage probe).
    for op, operands in (('v_interp_p1ll_f16', 'v5, v6, attr0'), ('v_interp_p2_f16', 'v5, v6, attr0, v4')):
        try:
            G.Stage.__new__(G.Stage).one(op, '%s %s' % (op, operands))
        except G.Unsupported as error:
            assert 'expected attrN.x/.y/.z/.w' in str(error), (op, str(error))
            print('refused: %s without a channel (%s)' % (op, error))
        else:
            raise AssertionError('%s without a channel was lowered' % op)
    # An attribute the vertex stage does not export (attr5; the vertex stage exports only param0): the whole pair
    # moves to attr5 (w1 bits 5:0), and the pipeline refuses it later, so interp_f16 needs no bound of its own.
    out_of_range = PS_WORDS
    for index in range(8):
        out_of_range = patch(out_of_range, index, w1=5)
    refused('attribute not exported by the vertex stage', out_of_range, 'attribute 5')
    # p2 whose src2 is not the p1ll result: v4 -> v8 in the third source
    other = patch(PS_WORDS, 1, w1=(0x104 ^ 0x108) << 18)
    refused('p2 source is not the p1ll result', other, 'is not the v_interp_p1ll_f16 result')
    # p1ll result overwritten before p2 reads it
    # (an overwrite between the pair: v_mov_b32 v4, v0 = 7e080300 inserted after the p1ll)
    clobbered = PS_WORDS[:12] + bytes.fromhex('0003087e') + PS_WORDS[12:]
    refused('p1ll result overwritten', clobbered, 'before the matching v_interp_p2_f16')
    # A plain float export of the same attribute is lowered as a plain export since the follow-up of #2052 (it used to
    # be refused as "not exported with compr"); plain_controls() covers it.
    assert 'hparam0_lo' in translate(PS_WORDS, F32_VS_WORDS).hlsl_ps()
    barycentric_controls(expected)
    plain_controls(expected)
    print('controls ok')


def plain_pair(half=0, vs_name='plain', ps_words=None):
    pipe = translate(PLAIN_PS['lo' if half == 0 else 'hi'] if ps_words is None else ps_words, PLAIN_VS[vs_name])
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def plain_controls(expected):
    """Plain 32-bit export of binary16 halves (follow-up of #2052)."""
    lo_ref, hi_ref = plain_reference(0), plain_reference(1)
    assert lo_ref == (0x3fc00000, 0x3f000000, 0x40800000, 0x41000000), [hex(w) for w in lo_ref]
    assert hi_ref == (0xc0000000, 0x40400000, 0xbe800000, 0x3e000000), [hex(w) for w in hi_ref]
    regs = {'v%d' % (6 + i): w for i, w in enumerate(PLAIN_WORDS)}
    for half, ref in ((0, lo_ref), (1, hi_ref)):
        vs, ps = plain_pair(half)
        got = evaluate(vs, ps, regs=regs)
        assert got == ref, ('plain half', half, [hex(w) for w in got], [hex(w) for w in ref])
        assert 'hparam0_lo' in vs and 'o.fparam0' not in vs, vs
        print('plain export half %d: %s' % (half, ','.join('%08x' % w for w in got)))
    # Runtime interface tables can link a pixel attribute to a differently numbered vertex parameter.
    remapped = PLAIN_PS['lo']
    for index in range(8):
        remapped = patch(remapped, index, w1=3)
    pipe = translate(remapped, PLAIN_VS['plain'], ps_attr_map={3: (0, 0xf)})
    assert evaluate(pipe.hlsl_vs(), pipe.hlsl_ps(), regs=regs) == lo_ref
    refused('interface masks off an interpolated channel', remapped, 'undeclared component w',
            vs_words=PLAIN_VS['plain'], ps_attr_map={3: (0, 0x7)})
    # The compr fixture with its export flipped to plain (clear the compr bit, w0 bit 10 of the param0 export:
    # words v6, v7, v0, v0) is the same attribute in the ISA (attrN.x = dword 0, attrN.y = dword 1), so the pixel
    # stage reads the compr reference. Only the lowering differs: hparam0_* instead of param0.x .. w.
    flipped_vs = bytearray(VS_WORDS)
    flipped_vs[VS_WORDS.index(bytes.fromhex('0f0600f8')) + 1] ^= 0x04
    pipe = translate(PS_WORDS, bytes(flipped_vs))
    assert 'hparam0_lo' in pipe.hlsl_ps() and 'param0.' not in pipe.hlsl_ps().replace('hparam0', ''), pipe.hlsl_ps()
    got = evaluate(pipe.hlsl_vs(), pipe.hlsl_ps(), regs={'v0': 0, 'v6': EXPORT[0], 'v7': EXPORT[1]})
    assert got == expected, ([hex(w) for w in got], [hex(w) for w in expected])
    print('compr fixture exported plain reads the compr reference %s' % ','.join('%08x' % w for w in got))
    # Negative control 1: the plain fixture read through the compr mapping (attrN.x = param.x, param.y; attrN.y =
    # param.z, param.w; channels z and w read as float channels 2c + h) gives other words: the mappings differ.
    vs, ps = plain_pair(0)
    compr_mapped = re.sub(r'i\.hparam0_(lo|hi)\.([xyzw])', lambda m: 'i.param0.%s' % 'xyzw'[
        2 * 'xyzw'.index(m.group(2)) % 4 + (m.group(1) == 'hi')], ps)
    got = evaluate(vs, compr_mapped, channels=[f16_to_f32_bits(PLAIN_WORDS[c] >> (16 * h) & 0xffff) for c, h in ((0, 0), (0, 1), (1, 0), (1, 1))],
                   regs=regs)
    assert got != lo_ref, 'compr mapping read the plain words'
    print('negative control: plain fixture read with the compr mapping gives %s' % ','.join('%08x' % w for w in got))
    # Negative control 2: the high bit inverted on the plain fixture reads the other halves.
    inverted = PLAIN_PS['lo']
    for index in range(8):
        inverted = patch(inverted, index, w1=0x100)
    got = evaluate(*plain_pair(ps_words=inverted), regs=regs)
    assert got == hi_ref and got != lo_ref, [hex(w) for w in got]
    print('negative control: plain high inverted reads %s' % ','.join('%08x' % w for w in got))
    # Negative control 3: the lowering removed refuses the pair.
    saved = {n: getattr(G.Stage, n) for n in ('op_v_interp_p1ll_f16', 'op_v_interp_p2_f16')}
    try:
        for n in saved:
            delattr(G.Stage, n)
        refused('plain lowering disabled', PLAIN_PS['lo'], 'v_interp_p1ll_f16', vs_words=PLAIN_VS['plain'])
    finally:
        for n, f in saved.items():
            setattr(G.Stage, n, f)
    # Refusals: an attribute exported both ways, and a channel exported as `off` that is read as binary16.
    refused('exported both compr and plain', PS_WORDS, 'exported both with compr and plain',
            vs_words=PLAIN_VS['mixed'])
    refused('off channel read as binary16', PLAIN_PS['lo'], 'channels the vertex stage exports as off',
            vs_words=PLAIN_VS['off'])
    # A compr export still refuses channels z and w.
    refused('compr attribute channel z', patch(patch(PS_WORDS, 0, w1=2 << 6), 1, w1=2 << 6), 'channel z')
    print('plain controls ok')


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def barycentric_controls(expected):
    # 1. The operand must be the entry PERSP_CENTER pair, still holding the entry values.
    refused('I clobbered before use', PS_VARIANTS['I clobbered'], 'writes v0')
    refused('J clobbered before use', PS_VARIANTS['J clobbered'], 'writes v1')
    refused('I and J swapped', PS_VARIANTS['I and J swapped'], 'barycentric operand v1')
    # One operand on another register: p1ll reads v2 as I, p2 reads v0 as J (the VGPR is src1, w1 bits 17:9).
    for label, index, field in (('p1ll reads v2', 0, 2), ('p2 reads v0', 1, 0)):
        words = bytearray(PS_WORDS)
        off = 4 + 8 * index
        w1 = struct.unpack_from('<I', words, off + 4)[0]
        w1 = (w1 & ~(0x1ff << 9)) | ((256 + field) << 9)
        struct.pack_into('<I', words, off + 4, w1)
        refused(label, bytes(words), 'barycentric operand')
    # Entry layout: the pair follows the PERSP_SAMPLE pair when that one is enabled, and PERSP_CENTER must be enabled.
    translate(ps_hw_inputs=(2, 2))
    print('accepted: PERSP_CENTER alone (v0, v1)')
    try:
        translate(ps_hw_inputs=(3, 3))
    except G.Unsupported as error:
        assert 'barycentric operand v0' in str(error), str(error)
        print('refused: PERSP_CENTER at v2, v3 (%s)' % error)
    else:
        raise AssertionError('v0, v1 accepted although they hold PERSP_SAMPLE')
    try:
        translate(ps_hw_inputs=(1, 1))
    except G.Unsupported as error:
        assert 'PERSP_CENTER' in str(error), str(error)
        print('refused: PERSP_CENTER not enabled (%s)' % error)
    else:
        raise AssertionError('accepted without PERSP_CENTER')
    # Positive control, varying triangle: the three vertices carry different binary16 pairs and the pixel reads an
    # interior point. ISA arithmetic: p1ll gives P0 + I * (P1 - P0) in f32, p2 adds J * (P2 - P0) and rounds to
    # binary16. The lowering's input is the rasterizer's value of the same interpolation.
    triangle = ((1.5, -2.0, 0.5, 3.0), (2.5, 1.0, -1.5, 0.25), (-0.5, 4.0, 2.0, -3.0))
    for i, j in ((0.25, 0.5), (0.5, 0.25), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.125, 0.375)):
        isa = []
        for c in range(4):
            p0, p1, p2 = (v[c] for v in triangle)
            partial = f32(p0 + i * (p1 - p0))
            isa.append(f16_to_f32_bits(f16_bits(f32(partial + j * (p2 - p0)))))
        interpolated = [f32_bits(f32(p0 + i * (p1 - p0) + j * (p2 - p0)))
                        for p0, p1, p2 in zip(*triangle)]
        pipe = translate()
        got = evaluate(pipe.hlsl_vs(), pipe.hlsl_ps(), interpolated)
        assert got == tuple(isa), ((i, j), [hex(w) for w in got], [hex(w) for w in isa])
    assert tuple(isa) != expected
    print('varying triangle: translated words equal the ISA arithmetic at 6 interior and corner points')
    # Registers other than the pair may be read or written freely.
    for label in ('unrelated read of v2', 'unrelated write of v9'):
        pipe = translate(PS_VARIANTS[label])
        assert evaluate(pipe.hlsl_vs(), pipe.hlsl_ps()) == expected, label
        print('accepted: %s' % label)

    # 2. The omitted p1ll result and the upper half of the p2 result.
    refused('p1ll result read between the pair', PS_VARIANTS['p1ll result read between the pair'],
            'reads it before the matching v_interp_p2_f16')
    refused('p1ll result read after p2 wrote elsewhere', PS_VARIANTS['p1ll result read after p2 wrote elsewhere'],
            'after a v_interp_p2_f16 that wrote another register')
    refused('upper half read by a shift', PS_VARIANTS['upper half read by a shift'], 'upper 16 bits')
    refused('upper half exported', PS_VARIANTS['upper half exported'], 'upper 16 bits')
    refused('branch into p2 without the matching p1ll', PS_VARIANTS['branch into p2'], 'a branch between')
    # The calculated branch jumps from byte 16 to byte 28, skipping the first p1ll.
    calculated = PS_WORDS[:4] + bytes.fromhex('001f94be1494148015801582142080be') + PS_WORDS[4:]
    refused('calculated branch into p2', calculated, 'a branch between')
    pair_and_export = PS_VARIANTS['upper half exported']
    # s_cbranch_vccz skips the one-word full-width conversion and reaches the export.
    skip_conversion = pair_and_export[:20] + bytes.fromhex('010086bf0417087e') + pair_and_export[20:]
    refused('branch skips the full-width conversion', skip_conversion, 'result across a branch')


def normalise(words):
    """Clear the unused src2 field of v_interp_p1ll_f16, which newer llvm-mc encodes as 128 and LLVM 15 as 0."""
    out = bytearray(words)
    off = 0
    for _, size, name, _ in G.decode(G.strip_header(words)):
        if name.startswith('v_interp_p1ll_f16'):
            w1 = struct.unpack_from('<I', out, off + 4)[0] & ~P1LL_SRC2_MASK
            struct.pack_into('<I', out, off + 4, w1)
        off += size
    return bytes(out)


def check_binary(llvm_mc):
    for name, source, words in (('vertex', VS_SOURCE, VS_WORDS), ('pixel', PS_SOURCE, PS_WORDS)):
        built = normalise(assemble(llvm_mc, source))
        if built != words:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    for name, source in PLAIN_VS_SOURCES.items():
        if assemble(llvm_mc, plain_vs_source(source)) != PLAIN_VS[name]:
            raise AssertionError('plain-export %s vertex fixture differs from llvm-mc' % name)
    for name, high in (('lo', ''), ('hi', ' high')):
        if normalise(assemble(llvm_mc, plain_ps_source(high))) != PLAIN_PS[name]:
            raise AssertionError('plain-export %s pixel fixture differs from llvm-mc' % name)
    for name, source in PS_VARIANT_SOURCES.items():
        built = normalise(assemble(llvm_mc, source))
        # a forward branch is an unresolved fixup in llvm-mc's listing (placeholder 0x0a): the offset is 2 dwords
        built = built.replace(bytes.fromhex('0a0a86bf'), bytes.fromhex('020086bf'))
        if built != PS_VARIANTS[name]:
            raise AssertionError('%s variant: assembled %s, recorded %s' % (name, built.hex(), PS_VARIANTS[name].hex()))
    print('fixture words match llvm-mc')


def pair():
    pipe = translate()
    return pipe.hlsl_vs(), pipe.hlsl_ps()


def check_compile(dxc):
    vs, ps = pair()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        compile_hlsl(dxc, 'dxc', vs, 'vs', 'vs_6_0', directory)
        compile_hlsl(dxc, 'dxc', ps, 'ps', 'ps_6_0', directory)
    for half in (0, 1):
        vs, ps = plain_pair(half)
        with tempfile.TemporaryDirectory() as tmp:
            compile_hlsl(dxc, 'dxc', vs, 'vs', 'vs_6_0', Path(tmp))
            compile_hlsl(dxc, 'dxc', ps, 'ps', 'ps_6_0', Path(tmp))
    print('dxc vs_6_0 / ps_6_0: compiled (compr pair and the plain-export pair, both halves)')


def check_draw(dxc, renderer, adapter):
    expected = ','.join('%08x' % w for w in reference())
    vs, ps = pair()
    flipped_words = PS_WORDS
    for index in range(8):
        flipped_words = patch(flipped_words, index, w1=0x100)
    flipped = translate(flipped_words)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs_cso = compile_hlsl(dxc, 'dxc', vs, 'vs', 'vs_6_0', directory)
        ps_cso = compile_hlsl(dxc, 'dxc', ps, 'ps', 'ps_6_0', directory)
        run = draw(renderer, adapter, vs_cso, ps_cso, expected)
        if run.returncode:
            raise AssertionError('translated pair\n%s\n%s' % (run.stdout, run.stderr))
        print('translated pair: %s' % expected, flush=True)
        bad_ps = compile_hlsl(dxc, 'dxc', flipped.hlsl_ps(), 'ps_high', 'ps_6_0', directory)
        run = draw(renderer, adapter, vs_cso, bad_ps, expected)
        if run.returncode == 0:
            raise AssertionError('high inverted: control read the reference words')
        print('high inverted: fails as expected', flush=True)
        # The plain export: low and high halves of channels x..w, and the high bit inverted.
        for half in (0, 1):
            expected_plain = ','.join('%08x' % w for w in plain_reference(half))
            pvs, pps = plain_pair(half)
            vs_cso = compile_hlsl(dxc, 'dxc', pvs, 'pvs', 'vs_6_0', directory)
            ps_cso = compile_hlsl(dxc, 'dxc', pps, 'pps', 'ps_6_0', directory)
            run = draw(renderer, adapter, vs_cso, ps_cso, expected_plain)
            if run.returncode:
                raise AssertionError('plain export half %d\n%s\n%s' % (half, run.stdout, run.stderr))
            print('plain export half %d: %s' % (half, expected_plain), flush=True)
            wrong = compile_hlsl(dxc, 'dxc', plain_pair(1 - half)[1], 'pps_other', 'ps_6_0', directory)
            if draw(renderer, adapter, vs_cso, wrong, expected_plain).returncode == 0:
                raise AssertionError('plain export: the other half read the reference words')
            print('plain export, other half: fails as expected', flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'compile':
        controls()
        check_compile(*args)
    elif mode == 'draw':
        controls()
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
