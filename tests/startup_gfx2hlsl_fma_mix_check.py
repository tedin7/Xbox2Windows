"""Mixed-precision VOP3P FMA (`v_fma_mix_f32`, `v_fma_mix{lo,hi}_f16`) in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_fma_mix_check.py [fxc <fxc> | dispatch <dxc> <offscreen_check> warp|hardware |
                                        warp-heap <offscreen_check> <dxc>]

With `fxc`, every translated line is also compiled for ps_5_1 (the profile the startup tools use),
so emitted text that is not valid HLSL fails here and not only in a later draw. With `dispatch`, every
translated line runs in a DXC cs_6_0 shader on WARP or a hardware adapter, and each result word is
compared with a CPU reference that decodes the VOP3P word itself. Exit code 77 is a capability skip.

`v_fma_mix_f32` was refused outright (58 code pairs in the #1723 refusal snapshot). It now shares
`Stage.mix()` with the two binary16-destination forms.

Source selection follows LLVM 15 (`AMDGPUDAGToDAGISel::SelectVOP3PMadMixModsImpl`, llvmorg-15.0.7):
"If the source's op_sel_hi is set, it indicates to do a conversion from fp16. If the sources's op_sel
is set, it picks the high half of the source register." `AMDGPUInstPrinter::printOpSelHi` prints
`SISrcMods::OP_SEL_1`, the bit that function sets for an fp16 source. abs, then neg, apply after
the conversion.

The instruction bytes are **constructed here** from the VOP3P layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, VOP3P branch):
  w0: [31:23] = 0b110011000, [22:16] opcode, [15] clamp, [14] op_sel_hi[2], [13:11] op_sel,
      [10:8] abs (neg_hi on other VOP3P), [7:0] vdst
  w1: [8:0] src0, [17:9] src1, [26:18] src2, [28:27] op_sel_hi[1:0], [31:29] neg (neg_lo)
"""
import math
import os
import re
import struct
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []
translated = []   # every emitted line, compiled together in fxc mode
cases = []        # (label, encoding, emitted line), run in dispatch mode

MIX_F32, MIXLO_F16, MIXHI_F16 = 0x20, 0x21, 0x22
V = 0x100            # VGPR source base
S5 = 5
ZERO = 128           # inline integer 0
ONE = 129            # inline integer 1
SEVENTEEN = 145      # inline integer 17
MINUS_ONE = 193      # inline integer -1
MINUS_HALF_F = 241   # inline float -0.5
ONE_F = 242          # inline float 1.0
LITERAL = 255


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


def encode(op, vdst, srcs, *, op_sel=(0, 0, 0), op_sel_hi=(0, 0, 0), abs_bits=(0, 0, 0),
           neg=(0, 0, 0), clamp=0, literal=None):
    w0 = (0b110011000 << 23) | (op << 16) | (clamp << 15) | (op_sel_hi[2] << 14) | (vdst & 0xff)
    w1 = srcs[0] | (srcs[1] << 9) | (srcs[2] << 18) | (op_sel_hi[0] << 27) | (op_sel_hi[1] << 28)
    for i in range(3):
        w0 |= op_sel[i] << (11 + i)
        w0 |= abs_bits[i] << (8 + i)
        w1 |= neg[i] << (29 + i)
    code = struct.pack('<II', w0, w1)
    return code + struct.pack('<I', literal) if literal is not None else code


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self, low16_only_v=()):
        self.lines, self.sym = [], {}
        self.low16_only_v = set(low16_only_v)
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()


def translate(code, low16_only_v=()):
    decoded = G.decode(code)
    stage = Stage(low16_only_v)
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return ' | '.join(asm for _o, _s, _n, asm in decoded), ' '.join(stage.lines)


def expect(label, code, asm, hlsl, low16_only_v=()):
    got_asm, got = translate(code, low16_only_v)
    translated.append(got)
    cases.append((label, code, got))
    check(label + ' (decode)', got_asm == asm, f'got {got_asm!r}')
    check(label, got == hlsl, f'got {got!r}\n  want {hlsl!r}')
    return got


def expect_refused(label, code, asm, low16_only_v=()):
    decoded = G.decode(code)
    check(label + ' (decode)', decoded[0][3] == asm, f'got {decoded[0][3]!r}')
    try:
        _asm, got = translate(code, low16_only_v)
    except G.Unsupported as e:
        check(label + ' refused', True)
        return str(e)
    check(label + ' refused', False, f'translated to {got!r}')
    return None


v2, v3, v4 = V + 2, V + 3, V + 4

# The encoder must reproduce the LLVM-assembled words pinned in tests/rdna2_tools_check.py, so it does
# not only mirror the decoder it feeds.
check('encoder matches llvm-mc: v_fma_mix_f32 v12, s44, v19, v20',
      encode(MIX_F32, 12, (44, V + 19, V + 20)).hex() == '0c0020cc2c265204')
check('encoder matches llvm-mc: v_fma_mix_f32 v12, |s44|, -v19, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0]',
      encode(MIX_F32, 12, (44, V + 19, V + 20), op_sel=(1, 1, 0), op_sel_hi=(1, 0, 0),
             abs_bits=(1, 0, 0), neg=(0, 1, 0)).hex() == '0c1920cc2c26524c')

# --- v_fma_mix_f32: binary32 result, sources selected per operand -----------------------------------
plain = expect('f32 sources',
               encode(MIX_F32, 1, (v2, v3, v4)),
               'v_fma_mix_f32 v1, v2, v3, v4',
               'v1 = U(mix32(F(v2), F(v3), F(v4)));')
low = expect('binary16 low half of src0',
             encode(MIX_F32, 1, (v2, v3, v4), op_sel_hi=(1, 0, 0)),
             'v_fma_mix_f32 v1, v2, v3, v4 op_sel_hi:[1,0,0]',
             'v1 = U(mix32(f16tof32(v2), F(v3), F(v4)));')
high = expect('binary16 high half of src0',
              encode(MIX_F32, 1, (v2, v3, v4), op_sel=(1, 0, 0), op_sel_hi=(1, 0, 0)),
              'v_fma_mix_f32 v1, v2, v3, v4 op_sel:[1,0,0] op_sel_hi:[1,0,0]',
              'v1 = U(mix32(f16tof32(v2 >> 16), F(v3), F(v4)));')
check('control: op_sel changes the selected half', low != high)
check('control: op_sel_hi changes the source format', plain != low)
expect('abs then neg after widening a high half, scalar source, clamp',
       encode(MIX_F32, 1, (v2, S5, v4), op_sel=(0, 1, 0), op_sel_hi=(0, 1, 0),
              abs_bits=(0, 1, 0), neg=(0, 1, 0), clamp=1),
       'v_fma_mix_f32 v1, v2, -|s5|, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0] clamp',
       'v1 = U(saturate(mix32(F(v2), (-abs(f16tof32(s5 >> 16))), F(v4))));')
expect('inline 0 as a binary16 addend',
       encode(MIX_F32, 1, (v2, v3, ZERO), op_sel_hi=(1, 1, 1)),
       'v_fma_mix_f32 v1, v2, v3, 0 op_sel_hi:[1,1,1]',
       'v1 = U(mix32(f16tof32(v2), f16tof32(v3), 0.0));')
expect('negated inline 0 addend keeps a -0 product',
       encode(MIX_F32, 1, (v2, v3, ZERO), op_sel=(0, 1, 0), op_sel_hi=(0, 1, 0), neg=(0, 0, 1)),
       'v_fma_mix_f32 v1, v2, v3, -0 op_sel:[0,1,0] op_sel_hi:[0,1,0]',
       'v1 = U(mix32(F(v2), f16tof32(v3 >> 16), (-0.0)));')
expect('inline float on a binary32 source',
       encode(MIX_F32, 1, (v2, ONE_F, v4), op_sel_hi=(1, 0, 0)),
       'v_fma_mix_f32 v1, v2, 1.0, v4 op_sel_hi:[1,0,0]',
       'v1 = U(mix32(f16tof32(v2), 1.0, F(v4)));')
# neg on inline -0.5 prints as `--0.5`; `(--0.5)` would be an HLSL decrement (review on #2055).
expect('negated negative inline float',
       encode(MIX_F32, 1, (MINUS_HALF_F, v3, v4), neg=(1, 0, 0)),
       'v_fma_mix_f32 v1, --0.5, v3, v4',
       'v1 = U(mix32((-(-0.5)), F(v3), F(v4)));')
expect('control: an unnegated literal on a binary32 source still translates',
       encode(MIX_F32, 1, (LITERAL, v3, v4), literal=0x3f800000),
       'v_fma_mix_f32 v1, 0x3f800000, v3, v4',
       'v1 = U(mix32(F(0x3f800000u), F(v3), F(v4)));')

# --- the binary16-destination forms keep their packing --------------------------------------------
expect('mixlo writes the low half',
       encode(MIXLO_F16, 1, (v2, v3, v4), op_sel_hi=(1, 1, 0)),
       'v_fma_mixlo_f16 v1, v2, v3, v4 op_sel_hi:[1,1,0]',
       'v1 = (v1 & 0xffff0000u) | f16rn(mix32(f16tof32(v2), f16tof32(v3), F(v4)));')
expect('mixlo with a negated inline 0 on a binary16 source',
       encode(MIXLO_F16, 1, (v2, v3, ZERO), op_sel_hi=(1, 1, 1), neg=(0, 0, 1)),
       'v_fma_mixlo_f16 v1, v2, v3, -0 op_sel_hi:[1,1,1]',
       'v1 = (v1 & 0xffff0000u) | f16rn(mix32(f16tof32(v2), f16tof32(v3), (-0.0)));')
expect('mixhi writes the high half from a high-half source',
       encode(MIXHI_F16, 1, (v2, v3, v4), op_sel=(0, 0, 1), op_sel_hi=(0, 0, 1)),
       'v_fma_mixhi_f16 v1, v2, v3, v4 op_sel:[0,0,1] op_sel_hi:[0,0,1]',
       'v1 = (v1 & 0xffffu) | (f16rn(mix32(F(v2), F(v3), f16tof32(v4 >> 16))) << 16);')

# A register whose upper 16 bits are undefined (odd-dmask D16 image_load) can feed a low-half read.
expect('low half of a register with undefined upper bits',
       encode(MIX_F32, 1, (v2, v3, v4), op_sel_hi=(1, 0, 0)),
       'v_fma_mix_f32 v1, v2, v3, v4 op_sel_hi:[1,0,0]',
       'v1 = U(mix32(f16tof32(v2), F(v3), F(v4)));', low16_only_v={2})

# --- refusals -------------------------------------------------------------------------------------
expect_refused('high half of a register with undefined upper bits',
               encode(MIX_F32, 1, (v2, v3, v4), op_sel=(1, 0, 0), op_sel_hi=(1, 0, 0)),
               'v_fma_mix_f32 v1, v2, v3, v4 op_sel:[1,0,0] op_sel_hi:[1,0,0]', low16_only_v={2})
expect_refused('binary32 use of a register with undefined upper bits',
               encode(MIX_F32, 1, (v2, v3, v4)),
               'v_fma_mix_f32 v1, v2, v3, v4', low16_only_v={2})
expect_refused('op_sel on a binary32 negated inline 0',
               encode(MIX_F32, 1, (v2, v3, ZERO), op_sel=(0, 0, 1), neg=(0, 0, 1)),
               'v_fma_mix_f32 v1, v2, v3, -0 op_sel:[0,0,1]')
expect_refused('op_sel without op_sel_hi (high half of a binary32 source)',
               encode(MIX_F32, 1, (v2, v3, v4), op_sel=(1, 0, 0)),
               'v_fma_mix_f32 v1, v2, v3, v4 op_sel:[1,0,0]')
for src, value in enumerate((0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0), 240):
    expect(f'low-half binary16 inline {value}',
           encode(MIX_F32, 1, (v2, src, v4), op_sel_hi=(0, 1, 0)),
           f'v_fma_mix_f32 v1, v2, {value}, v4 op_sel_hi:[0,1,0]',
           f'v1 = U(mix32(F(v2), {value}, F(v4)));')
expect('low-half binary16 inline 1.0 in mixlo',
       encode(MIXLO_F16, 1, (v2, ONE_F, v4), op_sel_hi=(0, 1, 0)),
       'v_fma_mixlo_f16 v1, v2, 1.0, v4 op_sel_hi:[0,1,0]',
       'v1 = (v1 & 0xffff0000u) | f16rn(mix32(F(v2), 1.0, F(v4)));')
expect_refused('high half of inline 1.0',
               encode(MIX_F32, 1, (v2, ONE_F, v4), op_sel=(0, 1, 0), op_sel_hi=(0, 1, 0)),
               'v_fma_mix_f32 v1, v2, 1.0, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]')
expect_refused('binary16 inline reciprocal two pi',
               encode(MIX_F32, 1, (v2, 248, v4), op_sel_hi=(0, 1, 0)),
               'v_fma_mix_f32 v1, v2, 0.15915494, v4 op_sel_hi:[0,1,0]')
expect('literal on a binary16 source',
               encode(MIX_F32, 1, (LITERAL, v3, v4), op_sel_hi=(1, 0, 0), literal=0x3c00),
               'v_fma_mix_f32 v1, 0x3c00, v3, v4 op_sel_hi:[1,0,0]',
               'v1 = U(mix32(f16tof32(0x3c00u), F(v3), F(v4)));')
# The decoder prints neg on inline 1..64 or a literal as a negative integer. Read as an integer, the
# sign-bit flip is lost (neg on inline 1 is 0x80000001, not 0xffffffff), so these stay refused.
# Inline -1 prints the same text as neg on inline 1, so it is refused too.
expect_refused('neg on inline 1 (binary32)',
               encode(MIX_F32, 1, (ONE, v3, v4), neg=(1, 0, 0)),
               'v_fma_mix_f32 v1, -1, v3, v4')
expect_refused('inline -1 (same text as neg on inline 1)',
               encode(MIX_F32, 1, (MINUS_ONE, v3, v4)),
               'v_fma_mix_f32 v1, -1, v3, v4')
expect_refused('neg on inline 17 (binary32)',
               encode(MIX_F32, 1, (SEVENTEEN, v3, v4), neg=(1, 0, 0)),
               'v_fma_mix_f32 v1, -17, v3, v4')
expect_refused('neg on a literal (binary32)',
               encode(MIX_F32, 1, (LITERAL, v3, v4), neg=(1, 0, 0), literal=0x3f800000),
               'v_fma_mix_f32 v1, -0x3f800000, v3, v4')

# Both halves contain normal binary16 numbers: 1.0 in the low word and 2.0 in the high word.
for high, bits in ((0, 0x3c00), (1, 0x4000)):
    raw = encode(MIX_F32, 1, (LITERAL, v3, v4), literal=0x40003c00,
                 op_sel=(high, 0, 0), op_sel_hi=(1, 0, 0))
    suffix = (' op_sel:[1,0,0]' if high else '') + ' op_sel_hi:[1,0,0]'
    expect('select literal half %d' % high, raw,
           'v_fma_mix_f32 v1, 0x40003c00, v3, v4' + suffix,
           'v1 = U(mix32(f16tof32(0x%04xu), F(v3), F(v4)));' % bits)

expect('inline half on a low binary16 source',
       encode(MIX_F32, 1, (240, v3, v4), op_sel_hi=(1, 0, 0)),
       'v_fma_mix_f32 v1, 0.5, v3, v4 op_sel_hi:[1,0,0]',
       'v1 = U(mix32(0.5, F(v3), F(v4)));')
expect_refused('unverified high half of inline constant',
               encode(MIX_F32, 1, (240, v3, v4), op_sel=(1, 0, 0), op_sel_hi=(1, 0, 0)),
               'v_fma_mix_f32 v1, 0.5, v3, v4 op_sel:[1,0,0] op_sel_hi:[1,0,0]')


def compile_translated(fxc):
    """Compile every translated line in one ps_5_1 entry point; returns FXC's exit status and output."""
    regs = sorted({int(n) for line in translated for n in re.findall(r'\bv(\d+)\b', line)})
    body = ['#define F asfloat', '#define U asuint', G.MIX32, G.F16RN,
            'uint main(nointerpolation uint4 p : TEXCOORD0) : SV_Target {',
            '    uint s5 = p.y;']
    body += ['    uint v%d = p.x + %du;' % (r, r) for r in regs]
    body += ['    ' + line for line in translated]
    body += ['    return %s;' % ' ^ '.join('v%d' % r for r in regs), '}']
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'fma_mix.hlsl')
        with open(src, 'w') as fh:
            fh.write('\n'.join(body) + '\n')
        run = subprocess.run([fxc, '/nologo', '/T', 'ps_5_1', '/E', 'main', '/Fo', os.path.join(d, 'fma_mix.cso'), src],
                             capture_output=True, text=True)
    return run.returncode, (run.stdout + run.stderr).strip()


# --- dispatch mode: run the translated lines on a D3D12 device and compare with a CPU reference ----
#
# The reference decodes each VOP3P word itself (layout in the module docstring), without
# tools/disasm_rdna2.py or gfx2hlsl, and applies the LLVM 15 reading: op_sel_hi = binary16 source,
# op_sel = its high half, abs then neg after widening, clamp = saturate. Inputs have short mantissas,
# so the product and sum are exact in binary32 and a fused and an unfused multiply-add give the same
# word. The rounding cases exercise sums that need rounding when stored as binary16.

INLINE_F32 = {240: 0.5, 241: -0.5, 242: 1.0, 243: -1.0, 244: 2.0, 245: -2.0, 246: 4.0, 247: -4.0}
F32_INPUT = {'v2': 1.25, 'v3': -0.75, 'v4': 3.0, 's5': 0.5}                                # binary32 reads
F16_INPUT = {'v2': (1.5, -2.5), 'v3': (0.75, 4.0), 'v4': (-1.25, 0.5), 's5': (2.0, -3.0)}  # (low, high)
V1_BEFORE = 0x13579bdf  # mixlo/mixhi keep one half of the destination
# The -0 case needs a zero product: a +0 addend would give +0, the -0 addend keeps -0.
OVERRIDES = {'negated inline 0 addend keeps a -0 product': {'v2': 0x80000000}}

# Each row is v2 * 1.0 + v4, run with both destination halves.
ROUNDING_INPUTS = (
    ('below midpoint', 1.0, 2**-12),
    ('even lower neighbor tie', 1.0, 2**-11),
    ('odd lower neighbor tie', 1.0 + 2**-10, 2**-11),
    ('above midpoint', 1.0, 3 * 2**-12),
    ('carry into exponent', 2047.0, 0.5),
    ('largest finite', 65504.0, 0.0),
    ('rounds to largest finite', 65504.0, 15.0),
    ('overflow midpoint', 65504.0, 16.0),
    ('above overflow midpoint', 65504.0, 17.0),
    ('denormal even lower neighbor tie', 2**-23, 2**-25),
    ('denormal odd lower neighbor tie', 2**-24, 2**-25),
    ('negative above midpoint in magnitude', -1.0, -3 * 2**-12),
)


def f32_word(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def f16_word(value):
    try:
        packed = struct.pack('<e', value)
    except OverflowError:
        # struct raises when rounding a finite input to infinity.
        packed = struct.pack('<e', math.copysign(math.inf, value))
    return struct.unpack('<H', packed)[0]


def decode_mix(code):
    w0, w1 = struct.unpack_from('<II', code)
    literal = struct.unpack_from('<I', code, 8)[0] if len(code) >= 12 else None
    return {
        'op': (w0 >> 16) & 0x7f, 'clamp': (w0 >> 15) & 1,
        'srcs': [w1 & 0x1ff, (w1 >> 9) & 0x1ff, (w1 >> 18) & 0x1ff],
        'op_sel': [(w0 >> (11 + i)) & 1 for i in range(3)],
        'op_sel_hi': [(w1 >> 27) & 1, (w1 >> 28) & 1, (w0 >> 14) & 1],
        'abs': [(w0 >> (8 + i)) & 1 for i in range(3)],
        'neg': [(w1 >> (29 + i)) & 1 for i in range(3)],
        'literal': literal,
    }


def is_register(src):
    return src >= V or src < 106


def register_name(src):
    return 'v%d' % (src - V) if src >= V else 's%d' % src


def case_inputs(label, fields):
    """Register words for one case: each source register gets a value in the format it is read as."""
    words = {'v1': V1_BEFORE}
    for src, half in zip(fields['srcs'], fields['op_sel_hi']):
        if is_register(src):
            name = register_name(src)
            if half:
                low, high = F16_INPUT[name]
                words[name] = f16_word(low) | (f16_word(high) << 16)
            else:
                words[name] = f32_word(F32_INPUT[name])
    words.update(OVERRIDES.get(label, {}))
    return words


def source_value(src, half, high, words, literal):
    if is_register(src):
        word = words[register_name(src)]
    elif src == 255:
        word = literal
    elif src == 128:
        return 0.0
    elif src in INLINE_F32:
        assert not high, 'the translator refuses the high half of an inline constant'
        return INLINE_F32[src]
    else:
        raise ValueError('source %d has no reference' % src)
    if half:
        return struct.unpack('<e', struct.pack('<H', (word >> (16 * high)) & 0xffff))[0]
    return struct.unpack('<f', struct.pack('<I', word))[0]


def reference(fields, words):
    args = []
    for i in range(3):
        value = source_value(fields['srcs'][i], fields['op_sel_hi'][i], fields['op_sel'][i], words,
                             fields['literal'])
        if fields['abs'][i]:
            value = abs(value)
        if fields['neg'][i]:
            value = -value
        args.append(value)
    result = args[0] * args[1] + args[2]
    if fields['clamp']:
        result = min(max(result, 0.0), 1.0)
    if fields['op'] == MIX_F32:
        return f32_word(result)
    half = f16_word(result)
    before = words['v1']
    if fields['op'] == MIXLO_F16:
        return (before & 0xffff0000) | half
    return (before & 0xffff) | (half << 16)


def raw_fill(index):
    # Keep in step with raw_uav_fill_word in tests/spirv_window/raw_uav_dump.cpp.
    return ((index + 1) * 0x9E3779B1 & 0xffffffff) | 1


def dispatch_source(rows, lines, helper=G.MIX32):
    """One thread: per case, load opaque inputs, run the translated line, store v1."""
    body = ['#define F asfloat', '#define U asuint', helper, G.F16RN, 'RWByteAddressBuffer data : register(u0);',
            '[numthreads(1, 1, 1)]', 'void main() {']
    for index, ((_label, words), line) in enumerate(zip(rows, lines)):
        base = index * 6
        decls = []
        for slot, name in enumerate(('v1', 'v2', 'v3', 'v4', 's5')):
            opaque = raw_fill(base + slot) ^ words.get(name, 0)
            decls.append('uint %s = data.Load(%d) ^ 0x%08xu;' % (name, (base + slot) * 4, opaque))
        body.append('    { ' + ' '.join(decls) + ' ' + line + ' data.Store(%d, v1); }' % ((base + 5) * 4))
    body.append('}')
    return '\n'.join(body) + '\n'


def run_dispatch(dxc, runner, adapter, source, words_total, directory, tag):
    src, cso, out = (os.path.join(directory, tag + ext) for ext in ('.hlsl', '.cso', '.bin'))
    with open(src, 'w') as fh:
        fh.write(source)
    subprocess.run([dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', cso, src], check=True)
    size = (words_total * 4 + 15) // 16 * 16
    run = subprocess.run([runner, adapter, '--raw-uav-dump', cso, '1', str(size), out],
                         capture_output=True, text=True)
    print(run.stdout, end='')
    if run.returncode == 77:
        sys.exit(77)
    if run.returncode or run.stderr:
        raise AssertionError(run.stderr or 'raw UAV dump exited with %d' % run.returncode)
    with open(out, 'rb') as fh:
        data = fh.read()
    return struct.unpack('<%dI' % (len(data) // 4), data)


def check_dispatch(dxc, runner, adapter):
    runner = os.path.abspath(runner)
    dispatch_cases = cases.copy()
    rows = [(label, case_inputs(label, decode_mix(code))) for label, code, _line in cases]
    for op, name in ((MIXLO_F16, 'mixlo'), (MIXHI_F16, 'mixhi')):
        code = encode(op, 1, (v2, v3, v4))
        _, line = translate(code)
        for label, base, addend in ROUNDING_INPUTS:
            label = f'{name}: {label}'
            dispatch_cases.append((label, code, line))
            rows.append((label, {'v1': V1_BEFORE, 'v2': f32_word(base), 'v3': f32_word(1.0), 'v4': f32_word(addend)}))
    expected = [reference(decode_mix(code), words) for (_l, code, _), (_, words) in zip(dispatch_cases, rows)]
    lines = [line for _l, _c, line in dispatch_cases]
    # Information, not a check: 4097 * 4097 = 16785409 rounds to 16785408 in binary32, so a fused mad
    # returns 1.0 and an unfused one 0.0. mix32 allows either (#2729).
    rows.append(('cancellation', {'v2': f32_word(4097.0), 'v3': f32_word(4097.0), 'v4': f32_word(-16785408.0)}))
    lines.append('v1 = U(mix32(F(v2), F(v3), F(v4)));')
    out_dir = os.path.join(here, '..', 'out')
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as directory:
        words = run_dispatch(dxc, runner, adapter, dispatch_source(rows, lines), len(rows) * 6, directory,
                             'fma_mix')
        for index, (label, _c, _l) in enumerate(dispatch_cases):
            got = words[index * 6 + 5]
            check('%s: %s readback' % (adapter, label), got == expected[index],
                  'expected %08x, got %08x' % (expected[index], got))
        fused = words[len(dispatch_cases) * 6 + 5]
        print('%s: mad(4097, 4097, -16785408) = %08x (%s)' % (
            adapter, fused, {0x3f800000: 'fused', 0: 'unfused'}.get(fused, 'neither fused nor unfused')))
        # Negative control: a helper that clears the sign of a zero result, as the RDNA 1 and RDNA 4 drivers
        # did without precise, must fail exactly the -0 case.
        lossy = G.MIX32.replace('return result;', 'return asfloat(asuint(result) == 0x80000000u ? 0u : asuint(result));')
        assert lossy != G.MIX32
        lost = run_dispatch(dxc, runner, adapter, dispatch_source(rows, lines, lossy), len(rows) * 6, directory,
                            'fma_mix_sign_control')
        caught = [label for i, (label, _c, _l) in enumerate(dispatch_cases) if lost[i * 6 + 5] != expected[i]]
        check('%s: control that clears a -0 result fails only the -0 product case' % adapter,
              caught == ['negated inline 0 addend keeps a -0 product'], repr(caught))
        # Negative control: drop the high-half shift, so the device reads the low half. The readback must
        # then match a reference with op_sel cleared, and differ from the correct result somewhere.
        # (A zero product, as in the -0 case, hides the half it reads.)
        mutated = [line.replace(' >> 16)', ')') for line in lines]
        changed = [i for i, (a, b) in enumerate(zip(lines, mutated)) if a != b]
        control = run_dispatch(dxc, runner, adapter, dispatch_source(rows, mutated), len(rows) * 6, directory,
                               'fma_mix_control')
        low_reads = []
        for i in changed:
            fields = decode_mix(dispatch_cases[i][1])
            fields['op_sel'] = [0, 0, 0]
            low_reads.append(reference(fields, rows[i][1]))
        follows = all(control[i * 6 + 5] == want for i, want in zip(changed, low_reads))
        caught = [i for i in changed if control[i * 6 + 5] != expected[i]]
        check('%s: control without the high-half shift reads the low half in all %d high-half cases'
              % (adapter, len(changed)), bool(changed) and follows)
        check('%s: control changes %d of %d high-half results' % (adapter, len(caught), len(changed)),
              len(caught) > 0)


def heap_shaders():
    from startup_gfx2hlsl_dynamic_heap_check import IMAGE_HEAD, IMAGE_LOAD, IMAGE_TAIL, VS_WORDS, root_blob

    mix = encode(MIX_F32, 4, (V + 4, ONE_F, V + 5), op_sel_hi=(0, 1, 0))
    pipe = G.Pipeline(VS_WORDS, bytes.fromhex(IMAGE_HEAD + IMAGE_LOAD) + mix + bytes.fromhex(IMAGE_TAIL), root_blob())
    ps = pipe.hlsl_ps()
    check('heap pixel uses a binary16 inline 1.0', 'mix32(F(v4), 1.0, F(v5))' in ps)
    return pipe, pipe.hlsl_vs(), ps


def warp_heap(offscreen, dxc, pipe, vs_source, ps_source):
    from pathlib import Path
    from startup_gfx2hlsl_dynamic_heap_check import compile_shader, draw, mutate

    expected = [f32_word(value) for value in (1.25 * 1.0 + 2.5, 2.5, 4, 8)]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_shader(dxc, directory, 'vs_6_6', vs_source)
        ps = compile_shader(dxc, directory, 'ps_6_6', ps_source)
        draw(offscreen, vs, ps, 96, expected, len(pipe.uds), heap_count=5)
        wrong = mutate(ps_source, 'mix32(F(v4), 1.0, F(v5))', 'mix32(F(v4), 0.0, F(v5))')
        ps = compile_shader(dxc, directory, 'ps_6_6', wrong)
        draw(offscreen, vs, ps, 96, expected, len(pipe.uds), negative=True, heap_count=5)


if len(sys.argv) == 3 and sys.argv[1] == 'fxc':
    status, output = compile_translated(sys.argv[2])
    check(f'fxc ps_5_1 compiles all {len(translated)} translated lines', status == 0, output)
elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
    check_dispatch(*sys.argv[2:])
elif len(sys.argv) == 4 and sys.argv[1] == 'warp-heap':
    warp_heap(*sys.argv[2:], *heap_shaders())
elif len(sys.argv) == 1:
    heap_shaders()
elif len(sys.argv) != 1:
    print(__doc__)
    sys.exit(2)

if failures:
    print(f'FAILED: {len(failures)} check(s): {", ".join(failures)}')
    sys.exit(1)
print('all fma_mix checks passed')
