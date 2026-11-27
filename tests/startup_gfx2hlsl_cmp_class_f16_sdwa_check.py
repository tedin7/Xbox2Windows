"""The SDWA forms of `v_cmp_class_f16` / `v_cmpx_class_f16` in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation controls and value checks, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile source-built pixel programs with FXC ps_5_1 when given and with DXC
                                       ps_6_0, draw 16 x 16 pixels and compare every pixel with a CPU reference

The plain e32/e64 register and encoded-literal path (including the census form `0x7c01`) is covered by
startup_gfx2hlsl_cmp_class_f16_check.py. This file covers what that one does not:

  * SDWA: `src0_sel` DWORD and WORD_0 are lowered to the shared class16() helper on the register word (which reads
    bits [15:0] only); `src1_sel` selects and sign-extends the integer mask as for the binary32 class compare. WORD_1,
    each byte, sext() on src0, the abs/neg spellings, a missing select, a float mask and a bad destination are refused.
    Words are built with the VOP_SDWA9Be layout tools/disasm_rdna2.py reads, and two of them are compared with the
    bytes clang -cc1as (LLVM 15, gfx1013) emits for the same text.
  * Value level: the emitted predicate is evaluated over boundary patterns of every class with garbage in the upper
    half and against an independent reference written from the binary16 layout (struct 'e'). Three mutated helpers
    (upper half read, sNaN and qNaN swapped, +0 and -0 swapped) must disagree with that reference.
  * Draw: a register holding 0x7c01 in its low half (sNaN, class bit 0x1), a register with the same low half and a
    qNaN (0x7e00, class bit 0x2) in the upper half, the encoded literal 0x7c01, the e32 spelling and both SDWA
    selects all give the same class: the 16 x 16 pixels keep or discard by the class bit against a per-pixel mask,
    and every variant must match one reference image. Two controls must fail it: classifying the upper half instead
    of the low one (the independent high-half control) and swapping the sNaN/qNaN return values. A third reads the
    whole mask dword instead of the BYTE_1 select.

Encoded NEG and inline float constants are not lowered through the SDWA path and stay refused on the plain path
(tests/startup_gfx2hlsl_cmp_class_f16_check.py).
"""
import math
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
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402
from startup_gfx2hlsl_vopc_sdwa_check import (  # noqa: E402
    CLEAR, EXPORT_WORDS, HEIGHT, MASK, MUL_WORDS, MULTIPLIER, PACKED, WIDTH, draw, field, source, vopc_sdwa,
    words_file)
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


CMP_CLASS_F16, CMPX_CLASS_F16 = 0x8F, 0x9F    # VOPC e32 opcodes
SGPR_DST = 6
V1, V2 = 0x101, 0x102


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for decoded instructions to be translated."""

    def __init__(self, exec_known=True):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = exec_known


def translate(code, exec_known=True):
    decoded = G.decode(code)
    assert len(decoded) == 1 and decoded[0][1] == 8, decoded
    stage = Probe(exec_known)
    stage.one(decoded[0][2], decoded[0][3])
    return decoded[0][3], stage


def refusal_of(name, asm):
    try:
        stage = Probe()
        stage.one(name, asm)
    except G.Unsupported as error:
        return str(error)
    return None


# ---- the independent reference and the helper under test -------------------------------------------------------
def ref_class16(word):
    """The class bit of S0[15:0] from the binary16 layout read through struct 'e'; the upper half is ignored."""
    low = word & 0xFFFF
    value = struct.unpack('<e', struct.pack('<H', low))[0]
    negative = bool(low & 0x8000)
    if math.isnan(value):
        return 0x2 if low & 0x0200 else 0x1      # struct cannot tell sNaN from qNaN: mantissa bit 9 decides
    if math.isinf(value):
        return 0x4 if negative else 0x200
    if value == 0.0:
        return 0x20 if negative else 0x40
    if abs(value) < 2.0 ** -14:
        return 0x10 if negative else 0x80
    return 0x8 if negative else 0x100


# (binary16 pattern, the one class bit it must hit), written from the ISA class list, not from the lowering.
BOUNDARIES = [(0x0000, 6), (0x8000, 5), (0x0001, 7), (0x8001, 4), (0x03FF, 7), (0x83FF, 4), (0x0400, 8),
              (0x8400, 3), (0x7BFF, 8), (0xFBFF, 3), (0x7C00, 9), (0xFC00, 2), (0x7C01, 0), (0xFC01, 0),
              (0x7DFF, 0), (0x7E00, 1), (0xFE00, 1), (0x7FFF, 1), (0xFFFF, 1), (0x3C00, 8), (0xBC00, 3)]
UPPERS = [0x0000, 0xFFFF, 0x7C00, 0x7E00, 0x8000, 0xDEAD]


def class16_helper():
    pipe = object.__new__(G.Pipeline)
    pipe.uds, pipe.resources, pipe.samplers, pipe.uses_sampler_heap = [], {}, {}, False
    return [entry for entry in pipe.common() if 'uint class16(' in entry][0]


def run_helper(helper, word):
    """Evaluate the emitted helper body for one register word."""
    body = helper[helper.index('{') + 1:helper.rindex('}')]
    env = {'bits': word}
    for statement in (piece.strip() for piece in body.split(';')):
        if not statement:
            continue
        if statement.startswith(('uint ', 'bool ')):
            for part in statement.split(' ', 1)[1].split(','):
                name, expression = (piece.strip() for piece in part.split('=', 1))
                env[name] = int(bool(Expr(expression).run(env)[0])) if statement.startswith('bool ') \
                    else Expr(expression).run(env)[0]
        elif statement.startswith('if ') and ') return ' in statement:
            condition, consequence = statement[3:].split(') return ', 1)
            if Expr(condition[1:]).run(env)[0]:
                return Expr(consequence).run(env)[0]
        elif statement.startswith('return '):
            return Expr(statement[7:]).run(env)[0]
    raise ValueError('the helper returned nothing for %#x' % word)


HELPER = class16_helper()
HELPER_MUTATIONS = {
    'upper half read': ('uint h = bits & 0xffffu', 'uint h = bits'),
    'sNaN and qNaN swapped': ('? 0x2u : 0x1u', '? 0x1u : 0x2u'),
    '+0 and -0 swapped': ('negative ? 0x20u : 0x40u', 'negative ? 0x40u : 0x20u'),
}


def mutated(name):
    old, new = HELPER_MUTATIONS[name]
    assert HELPER.count(old) == 1, name
    return HELPER.replace(old, new)


def split_call(text, name):
    """(text before, argument, text after) of the one `name(...)` call in text."""
    start = text.index(name + '(')
    depth, at = 0, start + len(name)
    for at in range(start + len(name), len(text)):
        depth += text[at] == '('
        depth -= text[at] == ')'
        if depth == 0:
            break
    return text[:start], text[start + len(name) + 1:at], text[at + 1:]


def predicate(stage, word, mask_word, helper=HELPER, lane=1):
    """The value the emitted line computes for a source word and a mask register word."""
    expression = stage.lines[0].split(' = ', 1)[1].rstrip(';')
    before, argument, after = split_call(expression, 'class16')
    env = {'v1': mask_word, 'v2': word, 's5': word, 's6': mask_word, 'exec': lane}
    env['cls'] = run_helper(helper, Expr(argument).run(env)[0])
    return Expr(before + 'cls' + after).run(env)[0]


# ---- controls -------------------------------------------------------------------------------------------------
SELS_SRC1 = [('DWORD', False), ('WORD_0', False), ('BYTE_1', False), ('BYTE_1', True), ('WORD_1', True)]


def accepted_forms():
    for destination in ('vcc', 'sgpr', 'exec'):
        for sel0 in ('DWORD', 'WORD_0'):
            for sel1, sext1 in SELS_SRC1:
                for src0_kind, src0_number in (('v', 2), ('s', 5)):
                    yield destination, sel0, sel1, sext1, src0_kind, src0_number


def controls():
    # 1. Words built here against the bytes LLVM assembles for the same text.
    llvm_cmp = bytes([0xF9, 0x00, 0x1E, 0x7D, 0x04, 0x00, 0x04, 0x06])
    llvm_cmpx = bytes([0xF9, 0x00, 0x3E, 0x7D, 0x04, 0x00, 0x04, 0x06])
    built_cmp = vopc_sdwa(CMP_CLASS_F16, source('v', 4), source('v', 0), None, 'WORD_0', 'DWORD')
    built_cmpx = vopc_sdwa(CMPX_CLASS_F16, source('v', 4), source('v', 0), None, 'WORD_0', 'DWORD')
    check('built SDWA words equal the clang -cc1as bytes', built_cmp == llvm_cmp and built_cmpx == llvm_cmpx,
          '%s %s' % (built_cmp.hex(), built_cmpx.hex()))
    asm, stage = translate(llvm_cmp)
    check('the LLVM word decodes and lowers', asm == 'v_cmp_class_f16_sdwa vcc_lo, v4, v0 src0_sel:WORD_0 '
          'src1_sel:DWORD' and stage.lines == ['vcc = (((class16(v4) & v0) != 0u) ? 1u : 0u);'], repr((asm, stage.lines)))
    asm, stage = translate(llvm_cmpx, exec_known=False)
    check('the LLVM cmpx word writes exec behind the narrowed mask',
          asm == 'v_cmpx_class_f16_sdwa v4, v0 src0_sel:WORD_0 src1_sel:DWORD' and stage.exec_known is False
          and stage.lines == ['exec = (exec & (((class16(v4) & v0) != 0u) ? 1u : 0u));'], repr((asm, stage.lines)))

    # 2. Every accepted form: destination, src0 select and register file, src1 select and sext. The predicate
    # classifies the low half of the source whatever sits in the upper half, against the selected mask.
    errors, count = [], 0
    for destination, sel0, sel1, sext1, kind, number in accepted_forms():
        op = CMPX_CLASS_F16 if destination == 'exec' else CMP_CLASS_F16
        code = vopc_sdwa(op, source(kind, number), source('v', 1, sext=sext1),
                         SGPR_DST if destination == 'sgpr' else None, sel0, sel1)
        asm, stage = translate(code, exec_known=False)
        count += 1
        target = {'vcc': 'vcc', 'sgpr': 's%d' % SGPR_DST, 'exec': 'exec'}[destination]
        if (len(stage.lines) != 1 or not stage.lines[0].startswith(target + ' = ') or stage.exec_known
                or (SGPR_DST in stage.masks) != (destination == 'sgpr')):
            errors.append((asm, stage.lines, stage.masks))
            continue
        for word, _bit in BOUNDARIES:
            for upper in UPPERS:
                for mask_raw in (0x00000001, 0x00000200, 0x000003FF, 0x00000100, 0x0000FF80, 0xFFFF0001, 0x01FF0000):
                    for lane in (0, 1):
                        got = predicate(stage, (upper << 16) | word, mask_raw, lane=lane)
                        wanted = lane & int(bool(ref_class16(word) & field(mask_raw, sel1, sext1)))
                        if got != wanted:
                            errors.append((asm, hex((upper << 16) | word), hex(mask_raw), lane, got, wanted))
    check('%d accepted SDWA forms classify S0[15:0] against the selected mask and write their destination' % count,
          count == 3 * 2 * len(SELS_SRC1) * 2 and not errors, repr(errors[:3]))

    # 3. Refusals, each loud and with its own reason.
    def expect(label, code, message):
        name, asm = (lambda d: (d[0][2], d[0][3]))(G.decode(code))
        got = refusal_of(name, asm)
        check('refused: %s (%s)' % (label, got), got is not None and message in got, asm)

    for sel0 in ('WORD_1', 'BYTE_0', 'BYTE_1', 'BYTE_2', 'BYTE_3'):
        expect('src0_sel:%s' % sel0, vopc_sdwa(CMP_CLASS_F16, source('v', 2), source('v', 1), None, sel0, 'DWORD'),
               'SDWA form of v_cmp_class_f16 with src0_sel:%s' % sel0)
    expect('cmpx src0_sel:WORD_1', vopc_sdwa(CMPX_CLASS_F16, source('v', 2), source('v', 1), None, 'WORD_1', 'DWORD'),
           'SDWA form of v_cmpx_class_f16 with src0_sel:WORD_1')
    # abs and neg on the SDWA half source print |v2|, -v2, abs() or neg(): the sign bit they act on is bit 15 of
    # the half, which the SDWA path does not model, so they are refused instead of classifying the unmodified bits.
    expect('abs on src0', vopc_sdwa(CMP_CLASS_F16, source('v', 2, absolute=True), source('v', 1), None, 'WORD_0',
                                    'DWORD'), 'with source')
    expect('neg on src0', vopc_sdwa(CMP_CLASS_F16, source('v', 2, neg=True), source('v', 1), None, 'WORD_0',
                                    'DWORD'), 'with source')
    expect('neg and abs on src0', vopc_sdwa(CMP_CLASS_F16, source('v', 2, neg=True, absolute=True), source('v', 1),
                                            None, 'WORD_0', 'DWORD'), 'with source')
    for asm, message in (
            # The decoder cannot encode sext() on a float source, so the text is given directly.
            ('v_cmp_class_f16_sdwa vcc_lo, sext(v2), v1 src0_sel:WORD_0 src1_sel:DWORD', 'with source sext(v2)'),
            ('v_cmp_class_f16_sdwa vcc_lo, v2, v1 src1_sel:DWORD', 'SDWA form of v_cmp_class_f16 with no src0_sel'),
            ('v_cmp_class_f16_sdwa vcc_lo, v2, 1.0 src0_sel:WORD_0 src1_sel:DWORD', 'expected an integer class mask'),
            ('v_cmp_class_f16_sdwa vcc_lo, 1.0, v1 src0_sel:WORD_0 src1_sel:DWORD', 'with source 1.0'),
            ('v_cmp_class_f16_sdwa vcc_lo, 0x7c01, v1 src0_sel:WORD_0 src1_sel:DWORD', 'with source 0x7c01'),
            ('v_cmp_class_f16_sdwa vcc_lo, v2, v1 src0_sel:WORD_0 src0_sel:DWORD src1_sel:DWORD', 'with src0_sel'),
            ('v_cmp_class_f16_sdwa vcc_lo, v2, v1 src0_sel:WORD_0 src1_sel:DWORD clamp', 'clamp'),
            ('v_cmp_class_f16_sdwa null, v2, v1 src0_sel:WORD_0 src1_sel:DWORD', 'into null'),
            ('v_cmp_class_f16_sdwa vcc_lo, v2 src0_sel:WORD_0 src1_sel:DWORD', 'with 2 operands'),
            ('v_cmpx_class_f16_sdwa vcc_lo, v2, v1 src0_sel:WORD_0 src1_sel:DWORD', 'with 3 operands'),
            ('v_cmp_lt_f16_sdwa vcc_lo, v2, v1 src0_sel:BYTE_0 src1_sel:WORD_0',
             'f16 SDWA requires WORD_0 or WORD_1')):
        got = refusal_of(asm.split()[0], asm)
        check('refused: %s (%s)' % (asm, got), got is not None and message in got)
    code = vopc_sdwa(0xc9, source('v', 2), source('v', 1), None, 'WORD_0', 'WORD_0')
    asm, stage = translate(code)
    check('binary16 lt with WORD selectors lowers as a relational compare',
          'f16tof32' in stage.lines[0] and 'class16(' not in stage.lines[0], asm)

    # 4. The reference separates the controls. Each mutated helper, over the whole 16-bit space with upper halves
    # that differ from the low half, must disagree with it; the shipped helper agrees everywhere.
    def disagreements(helper, uppers, first_only=False):
        out = []
        for low in range(0x10000):
            for upper in uppers:
                word = (upper << 16) | low
                if run_helper(helper, word) != ref_class16(word):
                    out.append(hex(word))
                    if first_only:
                        return out
        return out

    check('the helper agrees with the binary16 reference on every low half under two upper halves',
          not disagreements(HELPER, (0x0000, 0x7E00)), repr(disagreements(HELPER, (0x0000, 0x7E00))[:4]))
    for name in HELPER_MUTATIONS:
        found = disagreements(mutated(name), (0x7E00,), first_only=True)
        check('control %s disagrees with the reference (first at %s)' % (name, found[:1]), len(found) > 0)
    _asm, stage = translate(vopc_sdwa(CMP_CLASS_F16, source('v', 2), source('v', 1), None, 'WORD_0', 'DWORD'))
    for name in HELPER_MUTATIONS:
        bad = [(hex(word), hex(upper)) for word, _bit in BOUNDARIES for upper in UPPERS
               if predicate(stage, (upper << 16) | word, ref_class16(word), helper=mutated(name)) != 1]
        check('control %s fails the emitted predicate on a boundary pattern' % name, bool(bad))


# ---- the draw -------------------------------------------------------------------------------------------------
LOW = 0x7C01          # a signaling NaN: class bit 0x1
UPPER_QNAN = 0x7E00   # a quiet NaN in the upper half: class bit 0x2, what a translation reading bits [31:16] sees


def v_mov_literal(vdst, literal):
    return struct.pack('<II', (0b0111111 << 25) | (vdst << 17) | (1 << 9) | 255, literal)


def e32_cmpx(src0, vsrc1):
    return struct.pack('<I', (0b0111110 << 25) | (CMPX_CLASS_F16 << 17) | (vsrc1 << 9) | src0)


def e64_cmpx_literal(literal, src1):
    return struct.pack('<III', (0b110101 << 26) | (CMPX_CLASS_F16 << 16) | 106, 255 | (src1 << 9), literal)


def sdwa_cmpx(sel0, sel1):
    return vopc_sdwa(CMPX_CLASS_F16, source('v', 2), source('v', 1), None, sel0, sel1)


# name: (source text, words, mask select). Every variant tests class bit 0x1 of the half 0x7c01 against the mask
# v1 = position * MULTIPLIER; only the way the half reaches the compare changes.
VARIANTS = {
    'register, upper qNaN, WORD_0 / DWORD': (
        'v_mov_b32_e32 v2, 0x7e007c01\nv_cmpx_class_f16_sdwa v2, v1 src0_sel:WORD_0 src1_sel:DWORD',
        v_mov_literal(2, (UPPER_QNAN << 16) | LOW) + sdwa_cmpx('WORD_0', 'DWORD'), None),
    'register, upper qNaN, DWORD / DWORD': (
        'v_mov_b32_e32 v2, 0x7e007c01\nv_cmpx_class_f16_sdwa v2, v1 src0_sel:DWORD src1_sel:DWORD',
        v_mov_literal(2, (UPPER_QNAN << 16) | LOW) + sdwa_cmpx('DWORD', 'DWORD'), None),
    'register, clean upper, WORD_0 / DWORD': (
        'v_mov_b32_e32 v2, 0x7c01\nv_cmpx_class_f16_sdwa v2, v1 src0_sel:WORD_0 src1_sel:DWORD',
        v_mov_literal(2, LOW) + sdwa_cmpx('WORD_0', 'DWORD'), None),
    'register, upper qNaN, e32': (
        'v_mov_b32_e32 v2, 0x7e007c01\nv_cmpx_class_f16_e32 v2, v1',
        v_mov_literal(2, (UPPER_QNAN << 16) | LOW) + e32_cmpx(V2, 1), None),
    'encoded literal 0x7c01, e64': (
        'v_cmpx_class_f16_e64 vcc_lo, 0x7c01, v1', e64_cmpx_literal(LOW, V1), None),
    'register, upper qNaN, WORD_0 / BYTE_1': (
        'v_mov_b32_e32 v2, 0x7e007c01\nv_cmpx_class_f16_sdwa v2, v1 src0_sel:WORD_0 src1_sel:BYTE_1',
        v_mov_literal(2, (UPPER_QNAN << 16) | LOW) + sdwa_cmpx('WORD_0', 'BYTE_1'), 'BYTE_1'),
}
PS_HEAD = 'v_mul_u32_u24_e32 v1, 0x9b5d, v0\n'
CONTROLS = {
    'upper half classified': ('class16(v2)', 'class16(v2 >> 16u)'),
    'sNaN and qNaN swapped': ('(h & 0x0200u) != 0u ? 0x2u : 0x1u', '(h & 0x0200u) != 0u ? 0x1u : 0x2u'),
}


def pixel_words(name):
    return MUL_WORDS + VARIANTS[name][1] + EXPORT_WORDS


def pixel_hlsl(name, control=None):
    text = G.Pipeline(VS_WORDS, pixel_words(name), root_blob(), ps_hw_inputs=PACKED).hlsl_ps()
    if control:
        old, new = CONTROLS[control]
        if old == 'class16(v2)' and name.startswith('encoded literal'):
            old, new = 'class16(0x00007c01u)', 'class16(0x00007c01u >> 16u)'
        assert text.count(old) == 1, (name, control, old)
        text = text.replace(old, new)
    return text


def image(select, class_bit):
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            position = x | (y << 16)
            mask = (position * MULTIPLIER) & MASK
            if select:
                mask = field(mask, select, False)
            words += [position if mask & class_bit else CLEAR] * 4
    return words


def check_fixture():
    for name, (text, _words, select) in VARIANTS.items():
        decoded = [asm for _o, _s, _n, asm in G.decode(pixel_words(name))]
        wanted = (PS_HEAD + text + '\nexp mrt0 v0, v0, v0, v0 done vm\ns_endpgm').splitlines()
        wanted = [line.replace('v_mov_b32_e32', 'v_mov_b32_e32') for line in wanted]
        check('%s: fixture decodes to its source' % name, decoded == wanted, '%s\n%s' % (decoded, wanted))
        hlsl = pixel_hlsl(name)
        check('%s: lowered through class16 behind exec' % name,
              'class16(' in hlsl and 'if (!(exec & 1)) discard;' in hlsl, hlsl)
        reference = image(select, 0x1)
        kept = sum(reference[i] != CLEAR for i in range(0, len(reference), 4))
        check('%s: reference keeps %d of %d pixels' % (name, kept, WIDTH * HEIGHT), 0 < kept < WIDTH * HEIGHT)
    # The controls must be distinguishable by the image, or the draw could not tell them apart.
    check('classifying the upper half (class bit 0x2) changes the image', image(None, 0x1) != image(None, 0x2))
    check('swapping sNaN and qNaN changes the image', image(None, 0x1) != image(None, 0x2))
    check('ignoring the BYTE_1 select changes the image', image('BYTE_1', 0x1) != image(None, 0x1))
    for name in VARIANTS:
        for control in CONTROLS:
            pixel_hlsl(name, control)    # every control applies to exactly one place in every variant


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        plain = words_file(directory, 'plain', image(None, 0x1))
        selected = words_file(directory, 'selected', image('BYTE_1', 0x1))
        upper = words_file(directory, 'upper', image(None, 0x2))
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            for index, (name, (_text, _words, select)) in enumerate(VARIANTS.items()):
                ps = compile_hlsl(compiler, kind, pixel_hlsl(name), 'ps_%d_%s' % (index, kind), ps_profile, directory)
                run = draw(renderer, adapter, vs, ps, selected if select else plain)
                check('%s %s: %s matches all %d pixels' % (kind, adapter, name, WIDTH * HEIGHT),
                      run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                      run.stdout + run.stderr)
            for control in CONTROLS:
                name = 'register, upper qNaN, WORD_0 / DWORD'
                bad = compile_hlsl(compiler, kind, pixel_hlsl(name, control), 'ps_bad_%s_%s' % (control[:5], kind),
                                   ps_profile, directory)
                run = draw(renderer, adapter, vs, bad, plain)
                check('%s %s: control %s fails the draw' % (kind, adapter, control),
                      run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)
            # A translation that classified the upper half draws the image of the upper class bit.
            bad = compile_hlsl(compiler, kind, pixel_hlsl('register, upper qNaN, WORD_0 / DWORD',
                                                          'upper half classified'), 'ps_up_%s' % kind, ps_profile,
                               directory)
            run = draw(renderer, adapter, vs, bad, upper)
            check('%s %s: upper-half control matches the upper-class image' % (kind, adapter),
                  run.returncode == 0, run.stdout + run.stderr)
            # The select control: the same program with the mask read as a whole dword draws the plain image.
            name = 'register, upper qNaN, WORD_0 / BYTE_1'
            text = pixel_hlsl(name)
            assert text.count('((v1 >> 8u) & 0xffu)') == 1, text
            bad = compile_hlsl(compiler, kind, text.replace('((v1 >> 8u) & 0xffu)', 'v1'), 'ps_sel_%s' % kind,
                               ps_profile, directory)
            run = draw(renderer, adapter, vs, bad, selected)
            check('%s %s: mask select control fails the draw' % (kind, adapter),
                  run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)
            run = draw(renderer, adapter, vs, bad, plain)
            check('%s %s: mask select control matches the whole-dword image' % (kind, adapter),
                  run.returncode == 0, run.stdout + run.stderr)


def main():
    if len(sys.argv) == 1:
        controls()
        check_fixture()
    elif sys.argv[1] == 'draw':
        check_fixture()
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
