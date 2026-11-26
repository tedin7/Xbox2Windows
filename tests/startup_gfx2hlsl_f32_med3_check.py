"""v_med3_f32 in tools/startup/gfx2hlsl.py: a NaN source is the minimum of the three (#2044).

  (no arguments)                       translation controls, pure Python
  decode                               decode the fixture words with tools/disasm_rdna2.py and compare the text
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC vs_5_1/ps_5_1 when given and with DXC
                                       vs_6_0/ps_6_0, draw one pixel and compare the four exported words; then the
                                       branchless lowering must disagree on the NaN lane
                                       (needs tests/offscreen_check.cpp built as offscreen_check)

The RDNA 2 ISA defines V_MED3_F32 (\"343 V_MED3_F32\") as:

    if (isNan(S0.f) || isNan(S1.f) || isNan(S2.f)) D.f = V_MIN3_F32(S0.f, S1.f, S2.f);
    else if (V_MAX3_F32(S0.f, S1.f, S2.f) == S0.f) D.f = V_MAX_F32(S1.f, S2.f);
    ...

A NaN source therefore makes the result the smallest of the three. The lowering used to emit
`max(min(a, b), min(max(a, b), c))`, which on a NaN source returns 2.0 for med3(NaN, 2.0, 1.0) where the ISA
returns 1.0: measured on an RX 9070 XT (driver 32.0.32029.2010) with a = NaN, 3.0, 1.5 and 0.5, the old form
gave 2.0, 2.0, 1.5, 1.0 and the ISA demands 1.0, 2.0, 1.5, 1.0.

The table now points at `med3_32`, whose body is that of `tools/pixel_aot.py`: a NaN source is recognised on
the encoding (`asuint(x) & 0x7fffffffu > 0x7f800000u`) because a float `isnan()` may be folded away by the
compiler, and the result is `min(min(a, b), c)`, which D3D's min() reduces to the smallest non-NaN source.
The integer medians (`v_med3_u32`, `v_med3_i32`) are untouched: they have no NaN.

What the measurement establishes and what it does not: the emitted text renders 2.0 on the card for a NaN
operand, and the ISA's rule is the lower of the non-NaN sources. That the hardware instruction itself
computes V_MIN3_F32 on a NaN source is **not** established by this fixture: HLSL has no way to emit
`v_med3_f32` (the compiler does not select it from the median pattern), so the draw compares the translated
text against the ISA rule, not against the instruction. This limit is recorded in docs/experiments.md.

Controls, all on the emitted HLSL text (no compiler, no GPU):
  - the preamble defines `med3_32` with the encoding test on all three sources, the `>` of `0x7f800000u` and
    the signed-zero `minmax_zero32` min3 chain for the NaN branch; no line calls a `med3(` helper;
  - the binary32 table points `v_med3_f32` at `med3_32` and the integer forms are unchanged;
  - every lane of the fixture lowers to `U(med3_32(F(v1), F(v2), F(v3)))` reading s44..s47, which the entry
    constant buffer fills at draw time;
  - four mutations must be rejected: the old branchless helper, the threshold `>=`, an `isnan(` test, and a
    table that points back at a helper without the branch.

The draw fixture is a source-built pair. The vertex stage places a full-screen triangle and exports the
position only. The pixel stage moves the four entry constants s8..s11 into s44..s47, then for each lane
`v_mov_b32 v1, s44+lane`, `v_mov_b32 v2, 2.0`, `v_mov_b32 v3, 1.0` and `v_med3_f32 v(12+lane), v1, v2, v3`,
exporting the four words to mrt0. The constants are root 32-bit constants (the form
tools/startup/gfx2hlsl.py lowers a constant buffer to), written by the renderer per draw: a literal would be
folded at compile time and the lowering under test would never run. The four inputs are 0x7fc00000 (quiet
NaN), 0x40400000 (3.0), 0x3fc00000 (1.5) and 0x3f000000 (0.5); the expected words are 0x3f800000 (1.0),
0x40000000 (2.0), 0x3fc00000 (1.5) and 0x3f800000 (1.0).

The control that counts: the branchless helper restored in the same pair, computed as `precise`, must MISMATCH on
the NaN lane, with 0x40000000 where the ISA asks 0x3f800000, and match the other three lanes. Without `precise`
a driver may fold the median against the constant bounds into a clamp and draw the ISA's 0x3f800000 there (#2730). A draw whose negative control
passes proves nothing; this check reports that as a failure rather than accepting it.

The instruction words are constructed here from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`): VOP3A w0 = 0b110101 << 26 | opcode << 16 | vdst, w1 = src0 | src1 << 9 | src2 << 18,
with the VGPR codes 256+n; v_med3_f32 is opcode 0x157, so `v_med3_f32 v12, v1, v2, v3` is
0c0057d501050e04. The decode mode proves that with the repository decoder.
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
import gfx2hlsl as G  # noqa: E402
import disasm_rdna2  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl  # noqa: E402

SOP1_MOV_B32 = 0xbe800300
VOP1_E32 = 0x7e000000
VOP1_MOV_B32 = 0x01
VGPR, LITERAL = 256, 255
INLINE = {0x40000000: 0xf4, 0x3f800000: 0xf2}  # 2.0 and 1.0, the coding llvm-mc picks for a float literal

# The four inputs, then the four words the ISA rule gives for med3(a, 2.0, 1.0).
INPUTS = (0x7fc00000, 0x40400000, 0x3fc00000, 0x3f000000)
EXPECTED = (0x3f800000, 0x40000000, 0x3fc00000, 0x3f800000)
# The branchless lowering's NaN lane: max(min(a, b), min(max(a, b), c)) with a = NaN is 2.0.
BRANCHLESS_NAN = 0x40000000

MED3_32 = ('float med3_32(float a, float b, float c) {\n'
           '    uint top = max(max(asuint(a) & 0x7fffffffu, asuint(b) & 0x7fffffffu), asuint(c) & 0x7fffffffu);\n'
           '    return top > 0x7f800000u\n'
           '        ? minmax_zero32(minmax_zero32(a, b, false), c, false)\n'
           '        : minmax_zero32(minmax_zero32(a, b, false), minmax_zero32(minmax_zero32(a, b, true), c, false), true);\n'
           '}')
BRANCHLESS = 'float med3_32(float a, float b, float c) {\n    return max(min(a, b), min(max(a, b), c));\n}'
# The draw control is the same helper computed as `precise`. Without it, a driver may fold the median against the
# fixture's constant bounds into a clamp whose NaN result is the ISA's: an RX 5700 XT drew 0x3f800000 on the NaN
# lane, so the defect escaped (#2730). `precise` keeps D3D's min/max NaN rule, so the NaN lane draws 2.0.
BRANCHLESS_DRAW = ('float med3_32(float a, float b, float c) {\n'
                   '    precise float r = max(min(a, b), min(max(a, b), c));\n    return r;\n}')
OLD_ISNAN = ('float med3_32(float a, float b, float c) {\n'
             '    return isnan(a) || isnan(b) || isnan(c) ? min(min(a, b), c)'
             ' : max(min(a, b), min(max(a, b), c));\n}')

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


# ---- the fixture -----------------------------------------------------------------------------------------------
VS_SOURCE = ('v_mov_b32 v3, 0x40400000\n'
             'v_mov_b32 v10, 0xc0400000\n'
             'v_cmp_eq_u32 vcc_lo, 1, v0\n'
             'v_cndmask_b32 v1, -1.0, v3, vcc_lo\n'
             'v_cmp_eq_u32 vcc_lo, 2, v0\n'
             'v_cndmask_b32 v2, 1.0, v10, vcc_lo\n'
             'v_mov_b32 v4, 1.0\n'
             'v_mov_b32 v5, 0\n'
             'exp pos0 v1, v2, v5, v4 done\n'
             's_endpgm\n')
VS_WORDS = (bytes.fromhex('ff02067e00004040' 'ff02147e000040c0' '8100847d' 'f3060202' '8200847d' 'f2140402'
                          'f202087e' '80020a7e')
            + bytes.fromhex('cf0800f801020504' '000081bf'))
PS_EXPORT = bytes.fromhex('0f1800f80c0d0e0f000081bf')  # exp mrt0 v12..v15 done vm ; s_endpgm


def sop1_mov_b32(sdst, ssrc0):
    return struct.pack('<I', SOP1_MOV_B32 | sdst << 16 | ssrc0)


def vop1_e32(opcode, vdst, src0):
    return struct.pack('<I', VOP1_E32 | vdst << 17 | opcode << 9 | src0)


def vop3_med3_f32(vdst, src0, src1, src2):
    w0 = 0xd4000000 | (0x157 << 16) | vdst
    return struct.pack('<II', w0, src0 | src1 << 9 | src2 << 18)


# The fixture supplies eight words: ud0 for the rectangle list, the four inputs, three unused. The
# translator declares three uds past the blob, so the pair reads eleven.
FIXTURE_WORDS = 8
PS_PREFIX_SOURCE = ''.join('s_mov_b32 s%d, s%d\n' % (44 + lane, 8 + lane) for lane in range(4))
PS_PREFIX_WORDS = b''.join(sop1_mov_b32(44 + lane, 8 + lane) for lane in range(4))


def body_source(lane):
    return ('v_mov_b32 v1, s%d\n'
            'v_mov_b32 v2, 2.0\n'
            'v_mov_b32 v3, 1.0\n'
            'v_med3_f32 v%d, v1, v2, v3\n' % (44 + lane, 12 + lane))


def body_words(lane):
    return (vop1_e32(VOP1_MOV_B32, 1, 44 + lane)
            + vop1_e32(VOP1_MOV_B32, 2, INLINE[0x40000000])
            + vop1_e32(VOP1_MOV_B32, 3, INLINE[0x3f800000])
            + vop3_med3_f32(12 + lane, VGPR + 1, VGPR + 2, VGPR + 3))


PS_SOURCE = (PS_PREFIX_SOURCE + ''.join(body_source(lane) for lane in range(4))
             + 'exp mrt0 v12, v13, v14, v15 done vm\ns_endpgm\n')
PS_WORDS = (PS_PREFIX_WORDS + b''.join(body_words(lane) for lane in range(4)) + PS_EXPORT)


def constant_root_blob(count):
    """A DXBC container with one root parameter of `count` 32-bit constants at b0, visibility ALL."""
    payload = struct.pack('<3I', 0, 0, count)
    params = struct.pack('<3I', 1, 0, 36)
    d = struct.pack('<6I', 2, 1, 24, 0, 0, 0) + params + payload
    part = b'RTS0' + struct.pack('<I', len(d)) + d
    return (b'DXBC' + bytes(16) + struct.pack('<III', 1, 32 + 4 + len(part), 1) + struct.pack('<I', 36) + part)


def entry_counts(vs_hlsl, ps_hlsl):
    """How many entry constants the emitted pair declares, and where the fixture's four land.

    ud0 is the rectangle-list switch the vertex stage reads; the root signature the fixture describes supplies
    exactly the words both stages declare, so a read of ud0..udN never falls outside it.
    """
    uds = {int(n) for n in re.findall(r'uint ud(\d+);', ps_hlsl)}
    assert uds == set(range(len(uds))), sorted(uds)
    assert uds == {int(n) for n in re.findall(r'uint ud(\d+);', vs_hlsl)}, (uds, vs_hlsl)
    return len(uds)


def pipeline():
    # The count depends only on the translated program, not on the values: translate once with a placeholder
    # root, then rebuild with the exact number of words the shaders declare.
    probe = G.Pipeline(VS_WORDS, PS_WORDS, constant_root_blob(FIXTURE_WORDS))
    count = entry_counts(probe.hlsl_vs(), probe.hlsl_ps())
    return G.Pipeline(VS_WORDS, PS_WORDS, constant_root_blob(FIXTURE_WORDS))


def words_text(values):
    return ','.join('%08x' % v for v in values)


# ---- text controls ---------------------------------------------------------------------------------------------
STATEMENT = 'v%d = U(med3_32(F(v1), F(v2), F(v3)));'


def check_text(hlsl):
    """Every control on one emitted pixel shader. Raises AssertionError with the first disagreement."""
    assert MED3_32 in hlsl, ('med3_32 missing or changed', hlsl)
    assert not re.search(r'(?<!med3_32)(?<![A-Za-z0-9_])med3\s*\(', hlsl.split('cbuffer')[0]), \
        ('a med3( helper is still in the preamble', hlsl)
    assert '0x7f800000u' in hlsl and 'asuint(a) & 0x7fffffffu' in hlsl, ('the encoding test is gone', hlsl)
    assert 'isnan(' not in hlsl, ('an isnan() test in the preamble', hlsl)
    statements = [line.strip() for line in hlsl.splitlines() if 'med3_32(' in line and '= U(' in line]
    assert statements == [STATEMENT % (12 + lane) for lane in range(4)], (statements, hlsl)
    for lane in range(4):
        assert 's%d = ud%d;' % (44 + lane, 1 + lane) in hlsl, \
            ('lane %d does not read entry constant ud%d' % (lane, 1 + lane), hlsl)
    for word in INPUTS:
        assert '0x%08xu' % word not in hlsl, ('a fixture input is a literal', hex(word), hlsl)


def check_controls():
    assert G.VEC_F['v_med3_f32'] == (3, 'med3_32({0}, {1}, {2})'), G.VEC_F['v_med3_f32']
    # The integer medians have no NaN and keep their own lowering.
    assert G.VEC_U['v_med3_u32'] == (3, 'max(min({0}, {1}), min(max({0}, {1}), {2}))'), G.VEC_U['v_med3_u32']
    assert G.VEC_U['v_med3_i32'] == (3, 'uint(max(min(int({0}), int({1})), min(max(int({0}), int({1})), int({2}))))'), \
        G.VEC_U['v_med3_i32']
    print('the binary32 table points v_med3_f32 at med3_32; the integer medians are unchanged')
    hlsl = pipeline().hlsl_ps()
    check_text(hlsl)
    # Depend on the spelling, so feed one() directly with the spellings a decoder cannot produce.
    class Pipeline:
        def __init__(self):
            self.resources, self.notes, self.modified = {}, set(), set()

    class Probe(G.Stage):
        def __init__(self):
            self.lines, self.sym = [], {}
            self.used_s, self.used_v, self.uniform_v = set(), set(), set()
            self.exec_known = True
            self.exports = {}
            self.pipe = Pipeline()

    for name in ('v_med3_f32', 'v_med3_f32_e32', 'v_med3_f32_e64'):
        stage = Probe()
        stage.one(name, '%s v12, v1, v2, v3' % name)
        assert stage.lines == [STATEMENT % 12], (name, stage.lines)
    print('both spellings lower to med3_32')


def mutations(hlsl):
    return [
        ('the branchless helper restored', lambda t: t.replace(MED3_32, BRANCHLESS)),
        ('the threshold changed to >=', lambda t: t.replace('top > 0x7f800000u', 'top >= 0x7f800000u')),
        ('an isnan() test instead of the encoding test', lambda t: t.replace(MED3_32, OLD_ISNAN)),
        ('the table pointing back at a helper without the branch',
         lambda t: t.replace(STATEMENT % 12, 'v12 = U(med3(F(v1), F(v2), F(v3)));')),
    ]


def check_mutations():
    hlsl = pipeline().hlsl_ps()
    for label, mutate in mutations(hlsl):
        changed = mutate(hlsl)
        rejected = changed != hlsl
        if rejected:
            try:
                check_text(changed)
            except AssertionError:
                pass
            else:
                rejected = False
        check('%s is rejected' % label, rejected)


# ---- the words ------------------------------------------------------------------------------------------------
def check_decode():
    decoded = disasm_rdna2.disassemble_binary(PS_WORDS)
    assembly = [ins['assembly'] for ins in decoded]
    assert assembly[0:4] == ['s_mov_b32 s%d, s%d' % (44 + lane, 8 + lane) for lane in range(4)], assembly[0:4]
    for lane in range(4):
        block = assembly[4 + 4 * lane:8 + 4 * lane]
        assert block == ['v_mov_b32_e32 v1, s%d' % (44 + lane), 'v_mov_b32_e32 v2, 2.0',
                         'v_mov_b32_e32 v3, 1.0', 'v_med3_f32 v%d, v1, v2, v3' % (12 + lane)], block
    assert assembly[-2:] == ['exp mrt0 v12, v13, v14, v15 done vm', 's_endpgm'], assembly[-2:]
    assert PS_WORDS[28:36].hex() == '0c0057d501050e04', PS_WORDS[28:36].hex()
    print('the fixture decodes to s44..s47, v_med3_f32 on each lane and the mrt0 export')


def check_binary(llvm_mc):
    for name, source, words in (('vertex', VS_SOURCE, VS_WORDS), ('pixel', PS_SOURCE, PS_WORDS)):
        built = assemble(llvm_mc, source)
        if built != words:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    print('fixture words match llvm-mc')


# ---- the draw -------------------------------------------------------------------------------------------------
def draw(renderer, adapter, vs, ps, constants, expected):
    """One pixel through offscreen_check --startup-entry-words; exit 77 (skip) propagates."""
    run = subprocess.run([renderer, adapter, '--startup-entry-words', str(vs), str(ps), constants, expected],
                         capture_output=True, text=True)
    sys.stdout.write(run.stdout)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(77)
    return run


def actual_words(run):
    """The four words the renderer reports as `actual=` in its mismatch line."""
    m = re.search(r'actual=([0-9a-f]{8}),([0-9a-f]{8}),([0-9a-f]{8}),([0-9a-f]{8})', run.stderr)
    return [int(word, 16) for word in m.groups()] if m else None


def naive(hlsl):
    """The branchless helper restored as `precise` (BRANCHLESS_DRAW), the table still calling med3_32."""
    return hlsl.replace(MED3_32, BRANCHLESS_DRAW)


def check_draw(dxc, renderer, adapter, fxc=None):
    check_controls()
    check_mutations()
    check_decode()
    if failures:
        sys.exit(1)
    # ud0 is the vertex stage's rectangle-list switch; the four inputs follow it and are what the pixel
    # stage moves into s44..s47. The list is as long as the constant buffer both shaders declare.
    translated = pipeline()
    entry = [0] * entry_counts(translated.hlsl_vs(), translated.hlsl_ps())
    entry[1:1 + len(INPUTS)] = INPUTS
    constants = words_text(entry)
    expected = words_text(EXPECTED)
    print('constants %s expected %s' % (constants, expected))
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, pipeline().hlsl_vs(), 'vs_med3_%s' % kind, vs_profile, directory)
            good = compile_hlsl(compiler, kind, pipeline().hlsl_ps(), 'ps_med3_%s' % kind, ps_profile, directory)
            run = draw(renderer, adapter, vs, good, constants, expected)
            if run.returncode:
                raise AssertionError('%s translated pair, constants %s\n%s\n%s'
                                     % (kind, constants, run.stdout, run.stderr))
            print('%s %s: %s' % (kind, adapter, expected), flush=True)
            bad = compile_hlsl(compiler, kind, naive(pipeline().hlsl_ps()), 'ps_med3_bad_%s' % kind,
                               ps_profile, directory)
            run = draw(renderer, adapter, vs, bad, constants, expected)
            actual = actual_words(run)
            label = '%s branchless lowering on %s' % (kind, adapter)
            if run.returncode == 0:
                check('%s was accepted: the draw does not catch the defect' % label, False)
            elif actual is None:
                check('%s failed without an actual= line: %s' % (label, run.stderr.strip()), False)
            else:
                nan_lane = actual[0] == BRANCHLESS_NAN and EXPECTED[0] != BRANCHLESS_NAN
                others = actual[1:] == list(EXPECTED[1:])
                check('%s mismatched on the NaN lane (%s) and matched the other three'
                      % (label, words_text(actual)), nan_lane and others)
    if failures:
        sys.exit(1)


def main():
    if len(sys.argv) == 1:
        check_controls()
        check_mutations()
        check_decode()
        if failures:
            sys.exit(1)
        return
    mode, *args = sys.argv[1:]
    if mode == 'decode':
        check_decode()
    elif mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
