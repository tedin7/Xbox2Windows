"""v_interp_p1ll_f16 / v_interp_p2_f16 in tools/startup/gfx2hlsl.py (pixel stage, issue #1996 item C).

  (no arguments)                       translation controls and mutations, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  compile <dxc>                        compile the translated pair with DXC vs_6_0 / ps_6_0 (no rendering)
  draw_exec_halves <dxc> <offscreen_check> <warp|hardware>
                                       draw the EXEC coverage and explicit-LOD sample fixtures and their mutants
  draw_helper_taint <dxc> <offscreen_check> <warp>
                                       compare a partial-coverage triangle and its mutants against a CPU reference
  draw_flow <dxc> <offscreen_check> <warp|hardware>
                                       draw the varying, branching pair and mixed-FMA fixture and its mutants
  draw <dxc> <offscreen_check> <warp|hardware>
                                       draw the compr pair and the plain-export pair (both halves) and compare the four result words (Windows, D3D12)

Semantics (RDNA 2 ISA PDF pp.102, 188, 190):
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
modifier or form the fixture does not cover (refused). A p2 with op_sel 0x8 writes the high half and preserves
the previous low half. When p2 aliases p1ll the preserved half belongs to the omitted partial interpolant.

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
import gpu_case_runner  # noqa: E402
from startup_flat_param_check import VS_WORDS as F32_VS_WORDS, assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_image_3d_check import root_blob as image_root_blob  # noqa: E402

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
# Loops: the full-width conversions replace the omitted halves before the back edge of a constant-trip loop.
# The two refused loops carry a pending p1ll or its omitted high half along the back edge. The back-edge offset is patched into the llvm-mc listing (an unresolved fixup, 0x0a).
LOOP_HEAD = 's_mov_b32 m0, s2\ns_mov_b32 s20, 2\n'
LOOP_BACK = 's_sub_u32 s20, s20, 1\ns_cmp_lg_u32 s20, 0\ns_cbranch_scc1 L0\n'
LOOP_CVT = ''.join('v_cvt_f32_f16 v%d, v%d\n' % (n, n) for n in range(4, 8))
LOOP_SOURCES = {
    'loop': LOOP_HEAD + 'L0:\n' + V_BODY.split('v_cvt_f32_f16')[0] + LOOP_CVT + LOOP_BACK + 'exp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n',
    'loop p1ll pending across the back edge':
        LOOP_HEAD + 'v_interp_p1ll_f16 v4, v0, attr0.x\nL0:\nv_interp_p2_f16 v4, v1, attr0.x, v4\nv_cvt_f32_f16 v4, v4\n'
        'v_interp_p1ll_f16 v4, v0, attr0.x\n' + LOOP_BACK + V_TAIL,
    'loop p2 result not converted before the back edge':
        LOOP_HEAD + 'L0:\n' + V_FIRST + LOOP_BACK + 'v_cvt_f32_f16 v4, v4\n' + V_TAIL,
}
LOOP_WORDS = {name: bytes.fromhex(h) for name, h in (
    ('loop', '0203fcbe820394be040042d70000020004005ad700021204050042d70001020005005ad700031604060042d74000020006005ad740021a04'
             '070042d74001020007005ad740031e040417087e05170a7e06170c7e07170e7e14819480148007bfe9ff85bf0f1800f804050607000081bf'),
    ('loop p1ll pending across the back edge',
     '0203fcbe820394be040042d70000020004005ad7000212040417087e040042d70000020014819480148007bff8ff85bf0f1800f804040404000081bf'),
    ('loop p2 result not converted before the back edge',
     '0203fcbe820394be040042d70000020004005ad70002120414819480148007bff9ff85bf0417087e0f1800f804040404000081bf'),
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
        stmt = re.fullmatch(r' +(v\d+) = \((v\d+) & 0xffff0000u\) \| f16rn\(i\.(param0|hparam0_lo|hparam0_hi)\.([xyzw])\);',
                            line)
        if stmt:
            assert stmt.group(1) == stmt.group(2), line
            # f16rn of the interpolated value; a constant triangle makes it the exported value
            source = channels if stmt.group(3) == 'param0' else halves[stmt.group(3)[-2:]]
            f32 = source['xyzw'.index(stmt.group(4))]
            state[stmt.group(1)] = f16_bits(struct.unpack('<f', struct.pack('<I', f32))[0])
            continue
        stmt = re.fullmatch(r' +(v\d+) = U\(f16tof32\((v\d+)\)\);', line)
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
    assert ps.count('& 0xffff0000u) | f16rn(') == 4, ps
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
        refused('p2 op_sel bit %d' % (bit - 11), patch(PS_WORDS, 1, w0=1 << bit),
                'op_sel' if bit < 14 else 'preserved half of an omitted')
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
    flow_controls()
    extended_flow_controls()
    half_source_controls()
    monotone_exec_controls()
    cfg_exec_and_memory_controls()
    sample_and_subset_controls()
    helper_taint_controls()
    assert len(set(flow_reference(8, 4)[::4])) == 8
    loop_controls(expected)
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


def words(*values):
    return struct.pack('<%dI' % len(values), *values)


def branch(op, body):
    """Skip a dword-aligned instruction block."""
    return words(0xbf800000 | op << 16 | (len(body) // 4 & 0xffff))


def mov(dst, value):
    return words(0x7e0002ff | dst << 17, value)


def interp_words(p1, dst, channel, *, high=False, dst_high=False):
    attr = channel << 6 | int(high) << 8
    return (words(0xd7420000 | p1, attr | 256 << 9),
            words(0xd75a0000 | int(dst_high) << 14 | dst, attr | 257 << 9 | (256 + p1) << 18))


def mix_words(dst, src, *, half=False, opcode=0x20):
    # src * 1.0 + 0.0; the other two sources are always binary32.
    return words(0xcc000000 | opcode << 16 | dst, 256 + src | 242 << 9 | 128 << 18 | int(half) << 27)


END = words(0xbf810000)
ROUND_ZERO = words(0xbfa4000c)
M0 = words(0xbefc0302)


def export_words(*regs):
    return words(0xf800180f, sum(reg << (8 * i) for i, reg in enumerate(regs)))


def flow_fixture(*, stale_read=False):
    p1, p2 = zip(interp_words(12, 20, 0, high=True),
                 interp_words(13, 21, 1),
                 interp_words(0, 22, 1, high=True, dst_high=True))
    prefix = (M0 + mov(20, 0x3f800000) + mov(22, 0x3c00) + b''.join(p1 + p2)
              + mix_words(23, 20, half=True))
    # 1.5 < the widened interpolation. The two arms sample different halves of
    # the target and both overwrite the omitted p1ll results before the join.
    compare = words(0x7c020000 | 23 << 9 | 255, 0x3fc00000)
    left = mix_words(24, 20) + mov(12, 0x3f800000)
    right = mix_words(24, 23) + (mix_words(12, 12) if stale_read else mov(12, 0x40000000))
    jump = branch(2, right)
    split = compare + branch(6, left + jump) + left + jump + right
    epilogue = (mov(13, 0x3f800000) + ROUND_ZERO + mix_words(0, 12) + mix_words(1, 13)
                + mov(25, 0x3f801800) + mov(26, 0xbf801800)
                + mix_words(27, 25, opcode=0x21) + mix_words(27, 26, opcode=0x22)
                + words(0x2f << 25 | 28 << 17 | 1 << 9 | 256)
                + export_words(24, 22, 27, 28) + END)
    # Empty EXEC takes the null export without changing MODE.
    return branch(8, prefix + split + epilogue) + prefix + split + epilogue + words(0xf8000890, 0) + END


def flow_controls():
    ps = translate(flow_fixture(), varying_vertex_words()).hlsl_ps()
    assert 'v20 & 0xffff0000u' in ps, 'low p2 must preserve the previous high half'
    assert 'v22 & 0xffffu' in ps, 'high p2 must preserve the previous low half'
    assert 'f16tof32(v20)' in ps and 'mix32(F(v20)' in ps, ps
    assert ps.count('f16rz(mix32(') == 2 and 'uint f16rz(float value)' in ps, 'both mixed FMAs must use f16rz'
    refused('one arm reads stale p1ll', flow_fixture(stale_read=True), 'reads it after a v_interp_p2_f16')

    p1, p2 = interp_words(12, 20, 0)
    prefix = M0 + p1 + p2
    narrow = words(0xbe942480)  # s_and_saveexec_b64 s[20:21], 0
    restore = words(0xbefe0414)  # s_mov_b64 exec, s[20:21]
    overwrite = mov(12, 0x3f800000)
    use = mix_words(13, 12)
    tail = export_words(13, 13, 13, 13) + END
    translate(prefix + narrow + overwrite + use + restore + export_words(20, 20, 20, 20) + END)
    refused('whole write under narrower EXEC before restore', prefix + narrow + overwrite + restore + use + tail,
            'reads it after a v_interp_p2_f16')
    refused('partial replacement under original EXEC', prefix + mix_words(12, 20, opcode=0x21) + use + tail,
            'reads it after a v_interp_p2_f16')
    f32_interp = words(0xc8000000 | 11 << 18 | 1 << 10)  # v_interp_p1_f32 v11, v0, attr1.x
    separate_attr = VS_WORDS[:-4] + words(0xf800021f, 0x04050201) + END
    refused('I write before a later barycentric read', prefix + mov(0, 0) + f32_interp + END, 'writes v0',
            vs_words=separate_attr)
    # A later p2_f32 writes the modeled 32-bit register before its ordinary use.
    p2_f32 = words(0xc8010001 | 12 << 18 | 1 << 10)
    overwrite_f32 = words(0xc8000000 | 12 << 18 | 1 << 10) + p2_f32
    translate(prefix + overwrite_f32 + use + tail, separate_attr)
    # The other arm writes under the original EXEC. At the join that is insufficient.
    masked_arm = narrow + overwrite + restore
    plain_arm = overwrite
    jump = branch(2, plain_arm)
    split = words(0x7c020000 | 20 << 9 | 242) + branch(6, masked_arm + jump) + masked_arm + jump + plain_arm
    refused('one arm only writes under narrower EXEC', prefix + split + use + tail, 'reads it after a v_interp_p2_f16')
    loop = p1 + p2 + overwrite + mov(0, 0)
    refused('I write reaches an earlier barycentric read through a loop',
            M0 + loop + words(0xbf840000 | (-len(loop) // 4 - 1) & 0xffff) + END, 'writes v0')

    for immediate in (3, 4, 8):
        refused('unsupported rounding immediate %#x' % immediate, M0 + words(0xbfa40000 | immediate) + END,
                'binary32 rounding' if immediate == 3 else 'directed binary16/binary64 rounding')
    for name, op in (
            ('v_add_f16', words(0x32 << 25 | 12 << 17 | 11 << 9 | 266)),
            ):
        refused(name + ' in RTZ', M0 + ROUND_ZERO + op + END, name + ' with binary16/binary64 rounding toward zero')
    cvt = words(0x7e000000 | 10 << 17 | 0x0a << 9 | 242)
    for name, op in (('conversion', cvt), ('interpolation', p1 + p2)):
        shader = translate(M0 + ROUND_ZERO + op + END).hlsl_ps()
        assert 'f16rz(' in shader and 'uint f16rz(float value)' in shader, name
        mixed_op = M0 + branch(8, ROUND_ZERO) + ROUND_ZERO + op + END
        refused('mixed modes at ' + name, mixed_op, 'mixed binary16/binary64 rounding modes')
    mixed = M0 + branch(8, ROUND_ZERO) + ROUND_ZERO + mix_words(16, 17, opcode=0x21) + END
    refused('mixed modes at a binary16 producer', mixed, 'mixed binary16/binary64 rounding modes')
    reset = translate(M0 + ROUND_ZERO + words(0xbfa40000) + mix_words(16, 17, opcode=0x21) + END).hlsl_ps()
    assert 'f16rn(mix32(' in reset and 'f16rz' not in reset, reset


def save_exec(dst, source=128):
    return words(0xbe802400 | dst << 16 | source)


def restore_exec(src):
    return words(0xbefe0400 | src)


def compr_export(slot, first, second):
    return words(0xf8001c0f | slot << 4, first | second << 8)


def half_exec_fixture(*, slot=0, restore=True, nonlocal_read=False, alpha_stale=False):
    p1, p2 = interp_words(0, 20, 0, high=True)
    compare = words(0x7c020000 | 21 << 9 | 255, 0x3fc00000)  # 1.5 < v21
    cvt = words(0x7e001700 | 21 << 17 | 20)  # v_cvt_f32_f16 v21, v20
    mixlo = words(0xcc210000, 276 | 240 << 9 | 128 << 18 | 1 << 27)  # half(v20) * 0.5
    readlane = words(0xd760001e, 256 | 128 << 9) if nonlocal_read else b''
    return (M0 + p1 + p2 + cvt + compare + save_exec(20, 106) + mixlo + save_exec(22)
            + (restore_exec(22) if restore else b'') + readlane
            + mov(28, 0) + compr_export(slot, 0, 0 if alpha_stale else 28) + END)


def restored_exec_fixture(*, clobber=b'', wide=False):
    saved = words(0xbe94047e)  # s_mov_b64 s[20:21], exec
    p1, p2 = interp_words(12, 20, 0)
    if wide:
        p1_hi, p2_hi = interp_words(13, 21, 1)
        p1 += p1_hi
        p2 += p2_hi
        # Shift the quadword 0x1fe000001fc00000 left once. Consume both destination words.
        replace = (mov(16, 0x1fc00000) + mov(17, 0x1fe00000)
                   + words(0xd6ff000c, 129 | 272 << 9))
        export = export_words(12, 13, 12, 13)
    else:
        replace = mov(12, 0x3f800000)
        export = export_words(12, 12, 12, 12)
    return (M0 + saved + restore_exec(20) + p1 + p2 + save_exec(24) + clobber
            + restore_exec(20) + replace + restore_exec(24) + export + END)


def scan_decoded(*instructions, pixel=False):
    """Probe scan effects that do not need descriptor setup or shader lowering."""
    stage = G.Stage.__new__(G.Stage)
    if pixel:
        stage.kind = 'ps'
    stage.ins = [(4 * i, 4, G.parse(asm)[0], asm) for i, asm in enumerate(instructions)]
    stage.interp_f16_scan()
    return stage


def half_source_fixture(*, high_operand=None, wrong_compare=False):
    # The first pair leaves v20's partial unmaterialized. The second pair
    # replaces only v20's low half, which both subsequent instructions use.
    pairs = b''.join(interp_words(20, 19, 0) + interp_words(24, 20, 0))
    compare = 0xd4cc0028 if wrong_compare else 0xd4c90028  # v_cmp_{gt,lt}_f16_e64 s40
    select_high = 1 << 12  # op_sel[1]
    read = words(compare | (select_high if high_operand == 'compare' else 0), 128 | 276 << 9)
    if high_operand == 'add':
        read += words(0xd5320015 | select_high, 240 | 276 << 9)
    else:
        read += words(0x32 << 25 | 21 << 17 | 20 << 9 | 240)  # v_add_f16 v21, 0.5, v20
    result = words(0x7e001700 | 22 << 17 | 21, 0xd5010017, 128 | 242 << 9 | 40 << 18)
    return M0 + pairs + read + result + export_words(22, 23, 22, 23) + END


def half_source_controls():
    ps = translate(half_source_fixture()).hlsl_ps()
    assert 's40 = ((f16tof32(0x0000u) < f16tof32(v20)) ? 1u : 0u);' in ps, ps
    assert 'v21 = f16rn(0.5 + f16tof32(v20 & 0xffffu));' in ps, ps
    stale = ('v20 is the v_interp_p1ll_f16 result, which emits nothing: %s reads it after '
             'a v_interp_p2_f16 that wrote another register')
    for operand, op in (('compare', 'v_cmp_lt_f16'), ('add', 'v_add_f16')):
        refused('high operand in ' + op, half_source_fixture(high_operand=operand), stale % op)

    alias = ('v_interp_p1ll_f16 v20, v0, attr0.x', 'v_interp_p2_f16 v20, v1, attr0.x, v20')
    for read in (
            'v_cmp_class_f16 vcc_lo, v21, v20',
            'v_cmp_lt_u16 vcc_lo, 0, v20',
            'v_cmp_lt_i16 vcc_lo, v20, v21',
            'v_cvt_f32_f16 v21, v20',
            'v_add_f16 v22, v21, v20 op_sel:[1,0]',
            'v_mad_u32_u16 v22, v20, v21, v23',
            'v_fmac_f16 v22, v21, v20 op_sel:[1,0]'):
        scan_decoded(*alias, read)
    for read in (
            'v_cmp_lt_f16 s40, 0, v20 op_sel:[0,1]',
            'v_fmac_f16 v22, v21, v20 op_sel:[0,1]',
            'v_mad_u32_u16 v22, v21, v23, v20',
            'v_sad_u16 v22, v21, v20, v23',
            'v_sat_pk_u8_i16 v22, v20',
            'v_pk_add_f16 v22, v21, v20',
            'v_dot2_f32_f16 v22, v21, v20, v23',
            'v_cvt_f16_f32 v22, v20',
            'v_cvt_pkrtz_f16_f32 v22, v21, v20'):
        try:
            scan_decoded(*alias, read)
        except G.Unsupported as error:
            assert 'reads the preserved half of an omitted v_interp_p1ll_f16 result v20' in str(error), error
        else:
            raise AssertionError(read)


def monotone_exec_controls():
    masked = (
        's_mov_b64 s[20:21], exec',
        's_and_saveexec_b64 s[36:37], vcc',
        's_mov_b64 s[22:23], exec',
        's_mov_b64 exec, s[20:21]',
        'v_interp_p1ll_f16 v12, v0, attr0.x',
        'v_interp_p2_f16 v20, v1, attr0.x, v12',
        's_mov_b64 exec, s[22:23]',
        'v_mov_b32 v12, v20',
    )
    read = 'v_mov_b32 v30, v12'
    scan_decoded(*masked, 's_and_b64 exec, s[34:35], exec', read, 's_andn2_b64 exec, s[22:23], exec', read)
    for narrow in (
            's_and_b64 exec, exec, s[34:35]',
            's_andn2_b64 exec, exec, s[34:35]',
            's_and_saveexec_b64 s[26:27], s[34:35]',
            's_andn1_saveexec_b64 s[26:27], s[34:35]',
            'v_cmpx_lt_f32 v2, v3',
            'v_cmpx_lt_f32_e64 v2, v3',
            'v_cmpx_lt_f32_sdwa v2, v3',
    ):
        scan_decoded(*masked, narrow, read)
    for bound in (
            's_and_b64 exec, s[22:23], s[34:35]',
            's_and_b64 exec, s[34:35], s[22:23]',
            's_andn2_b64 exec, s[22:23], s[34:35]',
            's_andn2_saveexec_b64 s[22:23], s[22:23]',
    ):
        scan_decoded(*masked, 's_mov_b64 exec, s[34:35]', bound, read)
    scan_decoded(*masked, 's_andn2_saveexec_b64 s[26:27], s[34:35]',
                 's_mov_b64 exec, s[26:27]', read)  # save the old, covered EXEC
    scan_decoded(*masked, 's_mov_b64 exec, s[34:35]', 'v_mov_b32 v12, v20',
                 's_mov_b64 exec, s[22:23]', read)  # a second write retains the older mask

    for label, widen in (
            ('OR can expose stale lanes', 's_or_b64 exec, s[34:35], exec'),
            ('ANDN2 with unproven first operand', 's_andn2_b64 exec, s[34:35], exec'),
    ):
        try:
            scan_decoded(*masked, widen, read)
        except G.Unsupported as error:
            assert 'reads it after' in str(error), (label, error)
        else:
            raise AssertionError(label)
    # Exercise the same source selection through the decoder and pixel lowering.
    p1, p2 = interp_words(12, 20, 0)
    cmp = words(0x7c020000 | 2 << 9 | 242)  # v_cmp_lt_f32 vcc_lo, 1.0, v2
    and_exec = words(0x80000000 | 0x0f << 23 | 126 << 16 | 126 << 8 | 20)
    andn2_exec = words(0x80000000 | 0x15 << 23 | 126 << 16 | 126 << 8 | 22)
    overwrite = words(0x7e000314 | 12 << 17)  # v_mov_b32 v12, v20
    read12 = words(0x7e00030c | 30 << 17)
    code = (M0 + mov(2, 0x40000000) + cmp + save_exec(20, 106) + words(0xbe96047e)
            + restore_exec(20) + p1 + p2 + restore_exec(22) + overwrite
            + and_exec + read12 + andn2_exec + read12 + export_words(30, 30, 30, 30) + END)
    ps = translate(code).hlsl_ps()
    assert ps.count('v30 = v12;') == 2
    print('EXEC narrowing, older covered masks, and widening controls passed')


def cfg_exec_and_memory_controls():
    pair = ('v_interp_p1ll_f16 v12, v0, attr0.x', 'v_interp_p2_f16 v20, v1, attr0.x, v12')
    overwrite = 'v_mov_b32 v12, v20'
    read = 'v_mov_b32 v30, v12'

    def refuses(label, *instructions):
        try:
            scan_decoded(*instructions)
        except G.Unsupported as error:
            assert 'reads it after' in str(error), (label, error)
        else:
            raise AssertionError(label)

    def arm(join_clobber='s_nop 0', arm_clobber='s_nop 0', replacement=overwrite):
        # s24 preserves the entry EXEC on both paths. The condition and direct jump
        # split the narrowed arm before p1ll. s20 may be clobbered on just one path.
        return ('s_mov_b64 s[36:37], exec', 's_and_saveexec_b64 s[20:21], vcc',
                's_cbranch_scc0 label_0010', join_clobber, 's_branch label_0018', 's_nop 0',
                *pair, arm_clobber, 's_mov_b64 exec, s[20:21]', replacement,
                's_mov_b64 exec, s[36:37]', read)

    scan_decoded(*arm())
    refuses('saved EXEC overwritten in the arm', *arm(arm_clobber='s_load_dwordx2 s[20:21], s[8:9], 0'))
    refuses('saved EXEC overwritten on one incoming path', *arm(join_clobber='s_mov_b32 s21, 0'))
    refuses('saved EXEC overwritten by scalar buffer load',
            *arm(join_clobber='s_buffer_load_dwordx2 s[20:21], s[32:35], 0'))
    refuses('restored superset does not define the omitted value before a write', *arm(replacement=read))

    # A restore of an already equal mask preserves other equal copies, as in
    # the original straight-line proof.
    scan_decoded('s_mov_b64 s[20:21], exec', 's_mov_b64 s[22:23], exec', 's_mov_b64 exec, s[20:21]',
                 *pair, 's_and_saveexec_b64 s[36:37], vcc', 's_mov_b64 exec, s[22:23]', overwrite,
                 's_mov_b64 exec, s[20:21]', read)

    # The full shader path also decodes and lowers an if-arm with two branch
    # instructions between saveexec and p1ll.
    p1, p2 = interp_words(12, 20, 0)
    save_entry = words(0xbe98047e)
    cmp = words(0x7c020000 | 2 << 9 | 242)  # v_cmp_lt_f32 vcc_lo, 1.0, v2
    nop = words(0xbf800000)
    write12 = words(0x7e000314 | 12 << 17)
    guarded = (M0 + mov(2, 0x40000000) + cmp + save_entry + save_exec(20, 106)
               + branch(4, nop) + nop + branch(2, nop) + nop + p1 + p2
               + restore_exec(20) + write12 + restore_exec(24) + export_words(12, 12, 12, 12) + END)
    assert 'v12 = v20;' in translate(guarded).hlsl_ps()

    # Equality must disappear at saveexec, even though the previously equal
    # pair and the pair just saved both bound the narrowed EXEC.
    stage = G.Stage.__new__(G.Stage)
    instructions = ('s_mov_b64 s[36:37], exec', 's_and_saveexec_b64 s[20:21], vcc')
    effects = []
    for asm in instructions:
        name, ops, _ = G.parse(asm)
        effects.append(stage.interp_exec_effects(name, asm, ops, G.writes_exec(name, ops)))
    effects.append(stage.interp_exec_effects('s_nop', 's_nop 0', ['0'], False))
    facts = stage.interp_exec_entry_facts([[1], [2], []], effects, set())
    assert facts[2][:2] == (frozenset(), {frozenset({20, 21}), frozenset({36, 37})}), 'saveexec must clear EQ'

    masked = ('s_mov_b64 s[36:37], exec', *pair, 's_and_saveexec_b64 s[20:21], vcc', overwrite)
    load = 'image_load v30, [v12, v13], s[32:39]'
    store = 'buffer_store_dword v12, v13, s[32:35], 0 offen'
    scan_decoded(*masked, load, store)
    for masked_read in (
            'image_store v12, v[16:17], s[32:39]',
            'image_atomic_add v12, v[16:17], s[32:39]',
            'tbuffer_load_format_x v30, v12, s[32:35], 0 offen',
            'global_atomic_add v30, v[12:13], v14, off',
            'image_sample_lz v30, v[12:13], s[32:39], s[40:43]',
            'image_sample_l v30, v[12:14], s[32:39], s[40:43]',
            'image_gather4_c_lz_o v[30:33], v[12:16], s[32:39], s[40:43]',
    ):
        scan_decoded(*masked, masked_read)
    plain = scan_decoded(*masked, 'image_sample v30, v[12:13], s[32:39], s[40:43]')
    assert set(plain.interp_sample_obligations) == {plain.ins[-1][0]}
    for quad_read in (
            'image_sample_b v30, v[12:14], s[32:39], s[40:43]',
            'image_sample_d v30, v[12:17], s[32:39], s[40:43]',
            'image_get_lod v30, v[12:13], s[32:39], s[40:43]',
            'image_gather4 v[30:33], v[12:13], s[32:39], s[40:43]',
            'ds_swizzle_b32 v30, v12',
            'ds_permute_b32 v30, v12, v13',
            'ds_bpermute_b32 v30, v12, v13',
    ):
        refuses('cross-lane read must still fail: ' + quad_read, *masked, quad_read)
    refuses('restored active lanes may read a stale address', *masked, 's_mov_b64 exec, s[36:37]', load)
    print('EXEC CFG facts and masked memory controls passed')


def extended_flow_controls():
    vertex = varying_vertex_words()
    for fmt in (41, 54, 56, 58, 61, 63):
        for slot, formats in ((0, (fmt,)), (1, (10, fmt))):
            ps = translate(half_exec_fixture(slot=slot), vertex, rtv_formats=formats).hlsl_ps()
            assert 'o.mrt%d = float4(f16tof32(v0), f16tof32(v0 >> 16)' % slot in ps
    translate(half_exec_fixture(restore=False), vertex, rtv_formats=(61,))
    refused('inactive lanes are visible to readlane', half_exec_fixture(restore=False, nonlocal_read=True),
            'v_readlane_b32 reads it after', vs_words=vertex, rtv_formats=(61,))

    for wide in (False, True):
        translate(restored_exec_fixture(wide=wide), vertex)
    for clobber in (words(0xbe940380), words(0xbe950380)):
        refused('copy loses identity when either saved SGPR is overwritten',
                restored_exec_fixture(clobber=clobber), 'reads it after', vs_words=vertex)

    pair = ('v_interp_p1ll_f16 v12, v0, attr0.x', 'v_interp_p2_f16 v20, v1, attr0.x, v12')
    use = 'v_add_co_ci_u32 v30, vcc_lo, v12, v13, vcc_lo'
    for op in ('v_lshlrev_b64', 'v_ashrrev_i64', 'v_add_f64', 'v_mul_lo_u64'):
        scan_decoded(*pair, op + ' v[12:13], v[16:17], v[18:19]', use)
    for op in ('image_load v[11:12], [v8, v9], s[32:39]',
               'image_sample v[12:13], v[8:9], s[32:39], s[40:43]',
               'image_gather4 v[10:13], v[8:9], s[32:39], s[40:43]'):
        scan_decoded(*pair, op, use)
        for flag in ('d16', 'tfe', 'lwe'):
            try:
                scan_decoded(*pair, op + ' ' + flag, use)
            except G.Unsupported as error:
                assert 'reads it after' in str(error), error
            else:
                raise AssertionError('partial image destination treated as whole: ' + op + ' ' + flag)

    low = 'v_fma_mixlo_f16 v12, v20, 1.0, 0'
    high = 'v_fma_mixhi_f16 v12, v20, 1.0, 0'
    scan_decoded(*pair, low, 'v_cvt_f32_f16 v30, v12')
    scan_decoded(*pair, high, 'v_cvt_f32_f16_sdwa v30, v12 src0_sel:WORD_1')
    scan_decoded(*pair, low, 'v_fma_mix_f32 v30, v12, 1.0, 0 op_sel_hi:[1,0,0]')
    scan_decoded(*pair, high, 'v_fma_mix_f32 v30, v12, 1.0, 0 op_sel:[1,0,0] op_sel_hi:[1,0,0]')
    for word, byte in ((0, 1), (1, 3)):
        scan_decoded(*pair, 'v_mov_b32_sdwa v12, v20 dst_sel:WORD_%d dst_unused:UNUSED_PRESERVE' % word,
                     'v_mov_b32_sdwa v30, v12 src0_sel:BYTE_%d' % byte)
    def refuses_read(label, *instructions):
        try:
            scan_decoded(*instructions)
        except G.Unsupported as error:
            assert 'reads it after' in str(error), (label, error)
        else:
            raise AssertionError(label)

    for write, read in (
            ('', 'v_cvt_f32_f16 v30, v12'),
            (' ' + G.INTERP_DST_HIGH, 'v_cvt_f32_f16_sdwa v30, v12 src0_sel:WORD_1')):
        scan_decoded(*pair, 'v_interp_p1ll_f16 v22, v0, attr0.y',
                     'v_interp_p2_f16 v12, v1, attr0.y, v22' + write, read)
    refuses_read('image address list reads an omitted register', *pair,
                 'image_load v30, [v8, v12], s[32:39]')
    refuses_read('low-half writer killed the high half', *pair, low, use)
    for fmt in (10, 65):  # RGBA16_FLOAT and A8_UNORM
        refused('RGBA or alpha target needs the other half', half_exec_fixture(), 'reads it after',
                vs_words=vertex, rtv_formats=(fmt,))
    refused('alpha can reach blending on R-only targets', half_exec_fixture(alpha_stale=True),
            'reads it after', vs_words=vertex, rtv_formats=(61,))
    refuses_read('overlapping scalar load left a stale EXEC identity',
                 's_mov_b64 exec, s[20:21]', 's_mov_b64 s[36:37], exec', *pair,
                 's_load_dwordx4 s[18:21], s[8:9], 0', 's_mov_b64 exec, s[20:21]', 'v_mov_b32 v12, v20',
                 's_mov_b64 exec, s[36:37]', use)


def wqm(src=126):
    return words(0xbefe0a00 | src)


def copy_exec(dst):
    return words(0xbe80047e | dst << 16)


def sample_fixture(form='plain', flow='narrowed'):
    """Use v12 as U after replacing the omitted partial within the sample's EXEC."""
    p1, p2 = interp_words(12, 20, 0)  # attr0.x low half is 1.5 in VS_WORDS
    # Texture table slot 0 and an inline point sampler. Ordinary sampling
    # exposes all three mips. Comparison sampling is limited to level zero.
    sampler = (0x88001012, 0, 0, 0) if form == 'cmp' else (0, 0x200000, 0x04000000, 0)
    scalar = words(0xf40c0404, 0xfa000000)
    for reg, value in enumerate(sampler, 24):
        scalar += words(0xbe8003ff | reg << 16, value)
    scalar += words(0xbf8cc07f)
    saved_entry = copy_exec(32) if flow in ('entry', 'placeholder') else b''
    prefix = M0 + scalar + saved_entry + wqm() + p1 + p2
    # 1.25 < interpolated 1.5 on the covered pixel. Keep the scalar branch and
    # mask in the translated shader even though this one-pixel draw takes the body.
    prefix += words(0x7e001700 | 21 << 17 | 20, 0x7c020000 | 21 << 9 | 255, 0x3fa00000)
    opcode = {'plain': 0x20, 'bias': 0x25, 'cmp': 0x28, 'offset': 0x30}[form]
    body = mov(12, 0x3ec00000) + mov(13, 0x3f200000)  # (3/8, 5/8)
    if form != 'plain':
        body += mov(11, {'bias': 0, 'cmp': 0x3e000000, 'offset': 1}[form])
    body += words(0xf0000108 | opcode << 18, 0x00c41e00 | (12 if form == 'plain' else 11))
    body += words(0xbf8c3f70)
    if flow == 'narrowed':
        active = save_exec(32, 106) + branch(8, body) + body + restore_exec(32)
    elif flow == 'whole':
        active = body
    else:
        # Returning to the entry mask is a no-op in the main walk. In the
        # guest it removes the helper lanes before the coordinate overwrite.
        active = restore_exec(32)
        if flow == 'placeholder':
            active += branch(2, words(0xbf800000)) + words(0xbf800000) + scalar
        active += body
    return prefix + active + export_words(30, 30, 30, 30) + END


def sample_stage(form='plain', flow='narrowed'):
    return G.Pipeline(VS_WORDS, sample_fixture(form, flow), image_root_blob())


def subset_fixture(control=None, *, shrink_op=0x15):
    live, other, quad = 32, 34, 36
    p1, p2 = interp_words(12, 20, 0, high=True)
    prefix = M0 + copy_exec(live) + copy_exec(other) + wqm() + p1 + p2
    prefix += words(0x7e001700 | 21 << 17 | 20, 0x7c020000 | 21 << 9 | 255, 0x3fc00000)
    shrink = words(0x80000000 | shrink_op << 23 | live << 16 | 106 << 8 | live)
    cover = mov(12, 0x3fa00000)
    if control == 'shrink reverse':
        # The copies initially match. Shrinking live must break quad <= live.
        masks = wqm() + copy_exec(quad) + copy_exec(live) + shrink + restore_exec(live) + cover + restore_exec(quad)
    elif control == 'helpers':
        # The pre-WQM live mask cannot cover the helper lanes in quad.
        masks = copy_exec(quad) + restore_exec(live) + cover + restore_exec(quad)
    else:
        # Both paths enter the join under wqm(live), then save it in quad.
        narrowed = mov(29, 0x40000000)
        # Shrink again after the join. The AND is idempotent and must keep
        # the live <= EXEC fact learned from both WQM paths.
        masks = (shrink + wqm(live) + branch(6, wqm(live)) + wqm(live) + shrink
                 + save_exec(quad, 106) + branch(8, narrowed) + narrowed + restore_exec(quad) + cover)
        if control == 'rewritten':
            masks += words(0xbe800480 | live << 16)  # s_mov_b64 live, 0
        if control == 'different pair':
            masks += wqm(other) + copy_exec(other)
        masks += restore_exec(other if control == 'different pair' else live)
    # Match the comparison result and the overwritten coordinate itself.
    read = words(0x7c020000 | 21 << 9 | (256 + 12))
    select = mov(31, 0x3f800000) + words(0x02000000 | 30 << 17 | 31 << 9 | 128)
    return prefix + masks + read + select + export_words(12, 30, 21, 31) + END


def sample_and_subset_controls():
    for form, call in (
            ('plain', 'SampleLevel(q0, float2(F(v12), F(v13)), 0)'),
            ('cmp', 'SampleCmpLevelZero(q0, float2(F(v12), F(v13)), F(v11))')):
        ps = sample_stage(form).hlsl_ps()
        assert call in ps, (form, ps)
    stale = ('v12 is the v_interp_p1ll_f16 result, which emits nothing: %s reads it after '
             'a v_interp_p2_f16 that wrote another register')
    for label, form, flow, reason in (
            ('exec_known', 'plain', 'entry', stale % 'image_sample'),
            ('plain placeholder', 'plain', 'placeholder', stale % 'image_sample'),
            ('bias', 'bias', 'narrowed', stale % 'image_sample_b'),
            ('offset under narrowed EXEC', 'offset', 'narrowed', stale % 'image_sample_o'),
            ('offset under whole EXEC', 'offset', 'whole', 'image_sample_o')):
        try:
            sample_stage(form, flow).hlsl_ps()
        except G.Unsupported as error:
            assert str(error) == reason, (label, error)
        else:
            raise AssertionError('%s was translated' % label)
    for label in ('rewritten', 'different pair', 'helpers', 'shrink reverse'):
        refused(label, subset_fixture(label), stale % 'v_cmp_lt_f32', vs_words=varying_vertex_words())
    for shrink_op in (0x15, 0x0f):  # S_ANDN2_B64 and S_AND_B64
        ps = translate(subset_fixture(shrink_op=shrink_op), varying_vertex_words()).hlsl_ps()
        assert 'F(v12) < F(v21)' in ps, ps
        assert ps.count('v12 = 0x3fa00000u;') == 1, ps
    print('covered samples and EXEC subset controls passed')


def helper_taint_fixture(case='lerp'):
    """Read an omitted partial in WQM after overwriting it in exact EXEC."""
    live, quad = 32, 36
    p1, p2 = interp_words(20, 20, 0, high=True)
    setup = words(0xf40c0404, 0xfa000000)
    for reg, value in enumerate((0, 0x200000, 0x04000000, 0), 24):
        setup += words(0xbe8003ff | reg << 16, value)
    setup += words(0xbf8cc07f)

    def valu(op, dst, src0, src1):
        return words(op << 25 | dst << 17 | src1 << 9 | (256 + src0))

    def sample(dst, coords, opcode=0x20):
        return words(0xf0000108 | opcode << 18, 0x00c40000 | dst << 8 | coords, 0xbf8c3f70)

    code = (M0 + setup + copy_exec(live) + wqm() + p1 + p2
            + words(0x7e001700 | 21 << 17 | 20))  # unpack the full interpolant to v21
    # An independent implicit sample needs helpers at the triangle edge. Its
    # U gradient spans four texels per pixel, so the CPU expects mip 2.
    code += (mov(28, f32_bits(4)) + valu(0x08, 29, 21, 28)
             + mov(30, f32_bits(3 / 8)) + sample(31, 29))
    clobbered = case in ('clobbered', 'unrelated_kill')
    shrink = clobbered or case in ('lerp', 'lowered_level', 'rewritten')
    p1, p2 = interp_words(12, 20, 0, high=True)
    if clobbered:
        first_quad, null_desc = 38, 40
        code += copy_exec(first_quad) + restore_exec(first_quad) + p1 + p2
        # Load from a null descriptor so the test needs no buffer binding.
        code += words(0xbe800480 | null_desc << 16, 0xbe800480 | (null_desc + 2) << 16,
                      0xf4240000 | first_quad << 6 | null_desc >> 1, 0xfa000000, 0xbf8cc07f)
    if shrink:
        code += (mov(28, f32_bits(2.25)) + words(0x7c020000 | 21 << 9 | (256 + 28))
                 + words(0x8a800000 | live << 16 | 106 << 8 | live) + wqm(live))
    code += save_exec(quad, 106) if clobbered else p1 + p2 + copy_exec(quad)
    overwrite_mask = live
    if case == 'rewritten':
        # Rewrite before the overwrite so the new L still reaches masked.
        code += words(0xbe800480 | live << 16)  # s_mov_b64 L, 0
    elif case == 'unrelated_kill':
        overwrite_mask = 44
        code += words(0xbe800480 | overwrite_mask << 16)
    code += (restore_exec(overwrite_mask) + mov(12, f32_bits(5 / 16))
             + mov(13, f32_bits(3 / 8)) + sample(16, 12 if shrink else 29) + restore_exec(quad))
    code += (mov(24, f32_bits(3 / 8)) + mov(25, f32_bits(1 / 2))
             + valu(0x04, 23, 21, 12) + valu(0x2b, 12, 23, 25))  # lerp into v12
    last = 16
    if case in ('implicit', 'whole_clear', 'exact_clear', 'explicit', 'lowered_level'):
        if case == 'exact_clear':
            code += restore_exec(live)
        if case in ('whole_clear', 'exact_clear'):
            code += mov(23, f32_bits(3 / 8))
        if case == 'exact_clear':
            code += restore_exec(quad)
        code += sample(17, 23, opcode=0x27 if case == 'explicit' else 0x20)
        last = 17
    elif case == 'compare':
        nop = words(0xbf800000)
        code += words(0x7c020000 | 24 << 9 | (256 + 23)) + branch(6, nop) + nop
    elif case == 'dpp':
        code += words(0x7e0002fa | 26 << 17, 0xff00b117)  # v_mov_b32_dpp v26, v23 quad_perm:[1,0,3,2]
    if shrink:
        code += restore_exec(live)  # export with the shrunken coverage mask
    return code + export_words(12, 21, 31, last) + END


def helper_taint_stage(case):
    return G.Pipeline(helper_triangle_words(), helper_taint_fixture(case), image_root_blob())


def helper_taint_controls():
    for case in ('lerp', 'clobbered'):
        lerp = helper_taint_stage(case).hlsl_ps()
        assert 'F(v21) - F(v12)' in lerp and 'F(v23)' in lerp and '.SampleLevel(' in lerp, lerp
    whole = helper_taint_stage('whole_clear').hlsl_ps()
    assert re.search(r'\.Sample\(\w+, float2\(F\(v23\), F\(v24\)\)\)', whole), whole
    for case in ('explicit', 'lowered_level'):
        explicit = helper_taint_stage(case).hlsl_ps()
        assert re.search(r'\.SampleLevel\(\w+, float2\(F\(v23\), F\(v24\)\)', explicit), explicit
    reason = 'v23 may hold a stale v_interp_p1ll_f16 value in helper lanes: '
    stale = ('v12 is the v_interp_p1ll_f16 result, which emits nothing: v_sub_f32 reads it after '
             'a v_interp_p2_f16 that wrote another register')
    for case, diagnostic in (
            ('implicit', reason + 'image_sample reads it across lanes'),
            ('exact_clear', reason + 'image_sample reads it across lanes'),
            ('compare', reason + 'v_cmp_lt_f32 uses it to write a lane mask'),
            ('dpp', reason + 'v_mov_b32_dpp reads it across lanes'),
            ('rewritten', stale),
            ('unrelated_kill', stale)):
        try:
            helper_taint_stage(case).hlsl_ps()
        except G.Unsupported as error:
            assert str(error) == diagnostic, (case, error)
        else:
            raise AssertionError('%s was translated' % case)

    exact = 's_mov_b64 exec, s[32:33]'
    quad = 's_mov_b64 exec, s[36:37]'
    prefix = ('s_mov_b64 s[32:33], exec', 's_wqm_b64 exec, exec', 's_mov_b64 s[36:37], exec',
              'v_interp_p1ll_f16 v12, v0, attr0.x', 'v_interp_p2_f16 v20, v1, attr0.x, v12',
              exact, 'v_mov_b32 v12, 1.0', quad, 'v_sub_f32 v23, v21, v12')
    store = 'buffer_store_dword v23, v13, s[16:19], 0 offen'
    scan_decoded(*prefix, exact, store, pixel=True)
    for tail, diagnostic in (
            ((store,), reason + 'buffer_store_dword uses it in a memory write with helper lanes enabled'),
            (('image_atomic_add v30, v[22:23], s[16:23]',),
             reason + 'image_atomic_add uses it in a memory write with helper lanes enabled'),
            (('v_readfirstlane_b32 s20, v23',), reason + 'v_readfirstlane_b32 reads it across lanes'),
            (('v_cmpx_lt_f32 v23, v24',), reason + 'v_cmpx_lt_f32 uses it to write a lane mask'),
            (('v_add_co_u32 v20, vcc_lo, v23, v24',), reason + 'v_add_co_u32 uses it to write a lane mask'),
            (('v_mov_b32_sdwa v23, v30 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD', store),
             reason + 'buffer_store_dword uses it in a memory write with helper lanes enabled'),
            (('s_branch label_0024',), 'unmodelled v_interp_p1ll_f16 helper-lane taint across a backward branch')):
        try:
            scan_decoded(*prefix, *tail, pixel=True)
        except G.Unsupported as error:
            assert str(error) == diagnostic, (tail, error)
        else:
            raise AssertionError('%s cleared or observed helper-lane taint' % (tail,))
    # At the join, the taken edge bypasses a whole overwrite.
    sample = 'image_sample v30, v[23:24], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D'
    joined = scan_decoded(*prefix, 's_cbranch_scc0 label_002c', 'v_mov_b32 v23, 0', sample, pixel=True)
    assert joined.interp_sample_obligations == {44: reason + 'image_sample reads it across lanes'}
    helper_wqm_clear_controls(sample, reason + 'image_sample reads it across lanes')
    print('helper-lane interpolation taint controls passed')


def helper_wqm_clear_controls(sample, diagnostic):
    # Use the decoded clobbered-mask draw through the lerp. The real shader's
    # explicit-level fallback can satisfy the obligation, so test the implicit
    # call at the lowering gate with the obligation produced by the scan.
    prefix = [asm for _, _, _, asm in G.decode(helper_taint_fixture('clobbered'))][:-3]
    assert 's_buffer_load_dwordx2 s[38:39], s[40:43], 0x0' in prefix, prefix
    assert prefix[-2].startswith('v_sub_f32') and prefix[-1].startswith('v_fmac_f32'), prefix
    scan_decoded(*prefix[:-2], 's_and_b64 exec, exec, vcc', *prefix[-2:], pixel=True)
    quad = 's_mov_b64 exec, s[36:37]'
    clear = 'v_mov_b32 v23, 0'
    implicit = 'r0.Sample(q0, float2(F(v23), F(v24)))'
    live_wqm = 's_wqm_b64 exec, s[32:33]'
    save = 's_and_saveexec_b64 s[46:47], vcc'
    restore = 's_mov_b64 exec, s[46:47]'
    for setup in ((live_wqm,), (live_wqm, save, restore)):
        stage = scan_decoded(*prefix, *setup, clear, quad, sample, pixel=True)
        stage.pc = stage.ins[-1][0]
        stage.check_interp_sample_call(implicit)

    wqm_index = prefix.index('s_wqm_b64 exec, exec')
    sibling = prefix[:wqm_index] + ['s_mov_b64 s[44:45], exec'] + prefix[wqm_index:]
    for label, body, write_x in (
            ('unrelated_wqm_clear', prefix, 's_mov_b64 s[44:45], 0'),
            ('independently_shrunk_wqm_clear', sibling, 's_and_b64 s[44:45], s[44:45], 0')):
        stage = scan_decoded(*body, write_x, 's_wqm_b64 exec, s[44:45]', clear, quad, sample, pixel=True)
        stage.pc = stage.ins[-1][0]
        try:
            stage.check_interp_sample_call(implicit)
        except G.Unsupported as error:
            assert str(error) == diagnostic, (label, error)
        else:
            raise AssertionError('%s allowed an implicit sample' % label)


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
    refused('upper half read by a shift', PS_VARIANTS['upper half read by a shift'], 'preserved half of an omitted')
    refused('upper half exported', PS_VARIANTS['upper half exported'], 'preserved half of an omitted')
    refused('branch into p2 without the matching p1ll', PS_VARIANTS['branch into p2'], 'a branch between')
    # The calculated branch jumps from byte 16 to byte 28, skipping the first p1ll.
    calculated = PS_WORDS[:4] + bytes.fromhex('001f94be1494148015801582142080be') + PS_WORDS[4:]
    refused('calculated branch into p2', calculated, 'a branch between')
    pair_and_export = PS_VARIANTS['upper half exported']
    # s_cbranch_vccz skips the one-word full-width conversion and reaches the export.
    skip_conversion = pair_and_export[:20] + bytes.fromhex('010086bf0417087e') + pair_and_export[20:]
    refused('branch skips the full-width conversion', skip_conversion, 'preserved half of an omitted')


def loop_controls(expected):
    """A constant-trip loop around the pair translates (the pair is recomputed every iteration, so the exported words
    are the reference words); a p1ll result or an unconverted p2 result carried along the back edge stays refused."""
    pipe = translate(LOOP_WORDS['loop'])
    ps = pipe.hlsl_ps()
    assert 'pc = (scc) ? 1 : 2;' in ps and ps.count('f16rn(i.param0.') == 4, ps
    got = evaluate(pipe.hlsl_vs(), ps)
    assert got == expected, ('loop translated words', [hex(w) for w in got], [hex(w) for w in expected])
    print('translated loop: %s' % ','.join('%08x' % w for w in got))
    refused('loop: p1ll result pending across the back edge', LOOP_WORDS['loop p1ll pending across the back edge'],
            'a branch between')
    refused('loop: p2 result not converted before the back edge',
            LOOP_WORDS['loop p2 result not converted before the back edge'], 'result across a backward branch')
    # The high bit inverted on the pair inside the loop reads the other halves: translated, other words.
    flipped = bytearray(LOOP_WORDS['loop'])
    for index in range(8):  # the pair starts after two dwords (s_mov m0, s_mov s20)
        flipped[8 + 8 * index + 4 + 1] ^= 0x01
    flipped = bytes(flipped)
    other = evaluate(*(lambda p: (p.hlsl_vs(), p.hlsl_ps()))(translate(flipped)))
    assert other != expected, 'loop with the high bit inverted evaluated to the reference words'


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
    for name, source in LOOP_SOURCES.items():
        built = normalise(assemble(llvm_mc, source))
        # a backward branch is an unresolved fixup too (0x0a): the offset counts dwords from the label to the branch
        label = len(normalise(assemble(llvm_mc, source.split('L0:')[0]))) // 4
        branch = built.index(bytes.fromhex('0a0a85bf')) // 4
        built = built.replace(bytes.fromhex('0a0a85bf'), struct.pack('<HBB', (label - branch - 1) & 0xffff, 0x85, 0xbf))
        if built != LOOP_WORDS[name]:
            raise AssertionError('%s: assembled %s, recorded %s' % (name, built.hex(), LOOP_WORDS[name].hex()))
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
        compile_hlsl(dxc, 'dxc', translate(LOOP_WORDS['loop']).hlsl_ps(), 'ps_loop', 'ps_6_0', directory)
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
        # The constant-trip loop around the pair exports the reference words; its high bit inverted does not.
        loop = translate(LOOP_WORDS['loop'])
        loop_ps = compile_hlsl(dxc, 'dxc', loop.hlsl_ps(), 'ps_loop', 'ps_6_0', directory)
        run = draw(renderer, adapter, vs_cso, loop_ps, expected)
        if run.returncode:
            raise AssertionError('loop around the pair\n%s\n%s' % (run.stdout, run.stderr))
        print('loop around the pair: %s' % expected, flush=True)
        loop_flipped = bytearray(LOOP_WORDS['loop'])
        for index in range(8):
            loop_flipped[8 + 8 * index + 4 + 1] ^= 0x01
        bad_loop = compile_hlsl(dxc, 'dxc', translate(bytes(loop_flipped)).hlsl_ps(), 'ps_loop_high', 'ps_6_0', directory)
        if draw(renderer, adapter, vs_cso, bad_loop, expected).returncode == 0:
            raise AssertionError('loop with the high bit inverted read the reference words')
        print('loop, high inverted: fails as expected', flush=True)
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


def varying_vertex_words(right_x=3.0, bottom_y=-3.0, near=False):
    # Default: only vertex 1 has x = right_x and attr0.x's high half = 3; the others have x = -1 and high half 1.
    # near: attr0.x high varies from 1 to the next binary16 value, so at eight pixel centers the interpolant
    # is exactly (2*x+1)/16 half ULP above 1.
    def select(dst, original, replacement):
        return words(0x02000000 | dst << 17 | replacement << 9 | (256 + original))  # v_cndmask_b32

    return (mov(6, f32_bits(right_x)) + mov(7, f32_bits(bottom_y)) + mov(8, 0xbf800000) + mov(9, 0x3f800000)
            + mov(15, 0x3c004000) + mov(16, 0x3c014000 if near else 0x42004000) + mov(17, 0x42003800)
            + words(0x7d840081) + select(2, 8, 6) + select(15, 15, 16)  # v_cmp_eq_u32 1, v0
            + words(0x7d840082) + select(3, 9, 7)                       # v_cmp_eq_u32 2, v0
            + mov(4, 0) + mov(5, 0x3f800000)
            + words(0xf80008cf, 0x05040302, 0xf800060f, 0x0000110f) + END)


def helper_triangle_words():
    # On the 8x8 target, the triangle has screen vertices (0,0), (8,0), (0,7).
    # The interpolated high half of attr0.x ranges from 1 on the left to 3 at x=8.
    return varying_vertex_words(right_x=1.0, bottom_y=-0.75)


def flow_reference(width, height, near_input=False):
    """ISA reference at pixel centers; interpolants are exact binary32 values."""
    expected = []
    # 1 + 3*2^-12 is three quarters of a half ULP above 1.
    near = 1 + 3 * 2**-12
    truncation = 0x3c00
    assert f16_bits(near) == truncation + 1
    assert f16_bits(1 + (4.5 / 8) * 2**-10) == 0x3c01
    assert 1 + (4.5 / 8) * 2**-10 > 1 + 2**-11
    for _ in range(height):
        for x in range(width):
            interp = 1 + ((x + 0.5) / width) * (2**-10 if near_input else 1)
            half = f16_bits(interp)
            if interp > 1.5:
                first = 0x3f800000 | half
                overwritten = 1.0
            else:
                first = f16_to_f32_bits(half)
                overwritten = 2.0
            high_p2 = f16_bits(3.0) << 16 | f16_bits(1.0)
            mix = (0x8000 | truncation) << 16 | truncation
            packed = f16_bits(1.0) << 16 | f16_bits(overwritten)
            expected.extend((first, high_p2, mix, packed))
    return expected


def check_flow_draw(dxc, renderer, adapter):
    width, height = 8, 4
    pipe = translate(flow_fixture(), varying_vertex_words())
    ps = pipe.hlsl_ps()
    cases = {
        'flow': ps,
        'clear_high': ps.replace('(v20 & 0xffff0000u) | f16rn(i.param0.y)', 'f16rn(i.param0.y)'),
        'clear_low': ps.replace('(v22 & 0xffffu) | (f16rn(i.param0.w) << 16)', '(f16rn(i.param0.w) << 16)'),
        'round_nearest': ps.replace('f16rz(mix32(', 'f16rn(mix32('),
    }
    assert all(mutant != ps for name, mutant in cases.items() if name != 'flow')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        reference_path = directory / 'flow.bin'
        reference_path.write_bytes(words(*flow_reference(width, height)))
        vs = compile_hlsl(dxc, 'dxc', pipe.hlsl_vs(), 'flow_vs', 'vs_6_0', directory)
        for name, source in cases.items():
            pixel = compile_hlsl(dxc, 'dxc', source, name, 'ps_6_0', directory)
            run = gpu_case_runner.run(renderer, adapter, ['--startup-pixel-words', vs, pixel,
                                                           width, height, reference_path])
            diagnostic = run.stdout + run.stderr
            if run.returncode == 77 and name == 'flow':
                print(diagnostic, flush=True)
                raise SystemExit(77)
            if name == 'flow':
                assert run.returncode == 0 and 'all 32 pixels matched' in diagnostic, diagnostic
            else:
                assert run.returncode == 1 and 'pixel mismatch' in diagnostic, (name, diagnostic)
            print(name, diagnostic, flush=True)
        near_pipe = translate(flow_fixture(), varying_vertex_words(near=True))
        near_reference = directory / 'near.bin'
        near_reference.write_bytes(words(*flow_reference(width, height, near_input=True)))
        near_vs = compile_hlsl(dxc, 'dxc', near_pipe.hlsl_vs(), 'near_vs', 'vs_6_0', directory)
        near_ps = compile_hlsl(dxc, 'dxc', near_pipe.hlsl_ps(), 'near_ps', 'ps_6_0', directory)
        run = gpu_case_runner.run(renderer, adapter, ['--startup-pixel-words', near_vs, near_ps,
                                                       width, height, near_reference])
        diagnostic = run.stdout + run.stderr
        if run.returncode == 77:
            print(diagnostic, flush=True)
            raise SystemExit(77)
        assert run.returncode == 0 and 'all 32 pixels matched' in diagnostic, diagnostic
        print('near-half', diagnostic, flush=True)


def check_extended_flow_draw(dxc, renderer, adapter):
    width, height = 8, 4
    vertex = varying_vertex_words()
    # First four columns are inactive. Active columns export half(1 + (x + 0.5)/8) * 0.5.
    # This is an exact dyadic value, so the reference can quantize it to UNORM directly.
    red = [0xffff if x < 4 else round((17 + 2 * x) * 65535 / 32) for x in range(width)]
    red_words = [word for value in red * height for word in (value, 0, 0, 0)]
    rgba = '--startup-pixel-words'
    # The low half of attr0.x is 2.0 at every vertex. Add 0.5 and test 0 < 2.
    half_source_words = list(map(f32_bits, (2.0 + 0.5, float(0 < 2.0)))) * (2 * width * height)
    cases = (
        ('masked_r16', half_exec_fixture(), (56,), red_words, '--startup-r16-unorm', False),
        ('missing_restore', half_exec_fixture(restore=False), (56,), red_words, '--startup-r16-unorm', True),
        ('same_exec', restored_exec_fixture(), (), [0x3f800000] * (width * height * 4), rgba, False),
        ('wide', restored_exec_fixture(wide=True), (),
         [0x3f800000, 0x3fc00000] * (width * height * 2), rgba, False),
        ('half_sources', half_source_fixture(), (), half_source_words, rgba, False),
        ('half_sources_wrong_compare', half_source_fixture(wrong_compare=True), (), half_source_words, rgba, True),
    )
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name, code, formats, reference, operation, mismatch in cases:
            pipe = translate(code, vertex, rtv_formats=formats)
            vs = compile_hlsl(dxc, 'dxc', pipe.hlsl_vs(), name + '_vs', 'vs_6_0', directory)
            ps = compile_hlsl(dxc, 'dxc', pipe.hlsl_ps(), name + '_ps', 'ps_6_0', directory)
            expected = directory / (name + '.bin')
            expected.write_bytes(words(*reference))
            args = [operation, vs, ps, width, height, expected]
            if formats:
                args.append('0')  # Unused root constant for the existing R16_UNORM operation.
            run = gpu_case_runner.run(renderer, adapter, args)
            diagnostic = run.stdout + run.stderr
            if run.returncode == 77 and name == 'masked_r16':
                print(diagnostic, flush=True)
                raise SystemExit(77)
            if mismatch:
                assert run.returncode == 1 and 'pixel mismatch' in diagnostic, (name, diagnostic)
            else:
                assert run.returncode == 0 and 'all 32 pixels matched' in diagnostic, (name, diagnostic)
            print(name, diagnostic, flush=True)
        check_subset_draw(dxc, renderer, adapter, directory, vertex, width, height)
        check_covered_sample_draw(dxc, renderer, adapter, directory)


def check_subset_draw(dxc, renderer, adapter, directory, vertex, width, height):
    # S drops pixels with an interpolated value greater than 1.5. The
    # surviving pixels compare the replacement value 1.25 to interpolation.
    reference = []
    for x in range(width):
        interpolated = 1 + (x + 0.5) / width
        pixel = ((1.25, float(1.25 < interpolated), interpolated, 1.0)
                 if interpolated <= 1.5 else (-1.0,) * 4)
        reference.extend(map(f32_bits, pixel))
    expected = directory / 'exec_subset.bin'
    expected.write_bytes(words(*(reference * height)))
    pipe = translate(subset_fixture(), vertex)
    vs = compile_hlsl(dxc, 'dxc', pipe.hlsl_vs(), 'exec_subset_vs', 'vs_6_0', directory)
    source = pipe.hlsl_ps()
    mutant = source.replace('v12 = 0x3fa00000u;', 'v12 = 0x3fb00000u;')  # 1.375
    assert mutant != source
    for name, text in (('exec_subset', source), ('exec_subset_wrong_value', mutant)):
        ps = compile_hlsl(dxc, 'dxc', text, name, 'ps_6_0', directory)
        run = gpu_case_runner.run(renderer, adapter, ['--startup-pixel-words', vs, ps, width, height, expected])
        diagnostic = run.stdout + run.stderr
        if text == source:
            assert run.returncode == 0 and 'all %d pixels matched' % (width * height) in diagnostic, diagnostic
        else:
            assert run.returncode == 1 and 'pixel mismatch' in diagnostic, diagnostic
        print(name, diagnostic, flush=True)


def check_covered_sample_draw(dxc, renderer, adapter, directory):
    def texel(mip):
        # Point sampling of (3/8, 5/8) in the 4x4 mip fixture, slice zero.
        side = 4 >> mip
        x, y = (3 * side) // 8, (5 * side) // 8
        return (1 + x + 4 * y + 32 * mip) / 128

    vs = compile_hlsl(dxc, 'dxc', sample_stage().hlsl_vs(), 'covered_sample_vs', 'vs_6_0', directory)
    for form, reference, old, new, wrong in (
            ('plain', texel(0), 'F(v13)), 0)', 'F(v13)), 1)', texel(1)),
            ('cmp', float(0.125 < texel(0)), 'F(v13)), F(v11))', 'F(v13)), 0.0)', float(0.0 < texel(0)))):
        pipe = sample_stage(form)
        source = pipe.hlsl_ps()
        sampler, = (line for line in pipe.manifest().splitlines() if line.startswith(('smpl ', 'smpc ')))
        assert source.count(old) == 1 and wrong != reference
        for name, text, value in ((form, source, reference), (form + '_mutant', source.replace(old, new), wrong)):
            pixel = compile_hlsl(dxc, 'dxc', text, 'covered_sample_' + name, 'ps_6_0', directory)
            run = gpu_case_runner.run(renderer, adapter, ['--startup-image-2d-view', vs, pixel, sampler, '0/3'])
            expected = ','.join(['%08x' % f32_bits(value)] * 4)
            diagnostic = run.stdout + run.stderr
            assert run.returncode == 0 and 'view_rgba=' + expected in diagnostic, (name, diagnostic, expected)
            print(name, diagnostic, flush=True)


def helper_taint_reference(case):
    """Evaluate coverage, the two binary16 vertex values, and point mip sampling on the CPU."""
    reference = []
    for y in range(8):
        for x in range(8):
            interpolated = 1 + (x + 0.5) / 4
            covered = 7 * (2 * x + 1) + 8 * (2 * y + 1) < 112  # edge through (8,0) and (0,7)
            if not covered or (case in ('lerp', 'lowered_level', 'clobbered') and interpolated > 2.25):
                reference.extend([f32_bits(-1)] * 4)
                continue
            # This interpolant is exactly representable in binary16. v12 starts
            # the lerp at 5/16 and the texture's level-zero V selects row 1.
            diff = interpolated - 5 / 16
            texel_x = int(4 * diff) % 4 if case in ('explicit', 'lowered_level') else 1
            reference.extend(map(f32_bits, (5 / 16 + diff / 2, interpolated, 65 / 128, (5 + texel_x) / 128)))
    return reference


def check_helper_taint_draw(dxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for case in ('lerp', 'whole_clear', 'explicit', 'lowered_level', 'clobbered'):
            pipe = helper_taint_stage(case)
            source = pipe.hlsl_ps()
            sampler, = (line for line in pipe.manifest().splitlines() if line.startswith('smpl '))
            vs = compile_hlsl(dxc, 'dxc', pipe.hlsl_vs(), case + '_vs', 'vs_6_0', directory)
            expected = directory / (case + '.bin')
            expected.write_bytes(words(*helper_taint_reference(case)))
            if case == 'whole_clear':
                old, new = 'v23 = 0x3ec00000u;', 'v23 = 0x3f200000u;'  # Select the next texel after clearing.
            else:
                old, new = 'v25 = 0x3f000000u;', 'v25 = 0x3e800000u;'  # Lerp weight 1/2 becomes 1/4.
            assert source.count(old) == 1, source
            for name, text, mismatch in ((case, source, False), (case + '_mutant', source.replace(old, new), True)):
                ps = compile_hlsl(dxc, 'dxc', text, name, 'ps_6_0', directory)
                run = gpu_case_runner.run(renderer, adapter, ['--startup-pixel-image-words',
                                                               vs, ps, 8, 8, expected, sampler])
                diagnostic = run.stdout + run.stderr
                if run.returncode == 77 and name == 'lerp':
                    print(diagnostic, flush=True)
                    raise SystemExit(77)
                if mismatch:
                    assert run.returncode == 1 and 'pixel mismatch' in diagnostic, (name, diagnostic)
                else:
                    assert run.returncode == 0 and 'all 64 pixels matched' in diagnostic, (name, diagnostic)
                print(name, diagnostic, flush=True)


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
        check_flow_draw(*args)
    elif mode == 'draw_flow':
        check_flow_draw(*args)
    elif mode == 'draw_exec_halves':
        check_extended_flow_draw(*args)
    elif mode == 'draw_helper_taint':
        check_helper_taint_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
