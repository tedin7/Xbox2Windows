"""VOPC SDWA compares in tools/startup/gfx2hlsl.py: v_cmp_*_sdwa and v_cmpx_*_sdwa (#1906).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     assemble the decoded text of every accepted form and the pixel fixture with
                                       llvm-mc and compare with the words built here
  decode <libLLVM>                     disassemble the same words with LLVM's C API and compare with
                                       tools/disasm_rdna2.py
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the pixel fixture with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw 16 x 16 pixels and compare every pixel with a CPU reference

The instruction words are built here from the VOP_SDWA9Be layout that tools/disasm_rdna2.py reads:
  w0: [31:25] = 0b0111110, [24:17] opcode, [16:9] vsrc1, [8:0] = 249 (SDWA)
  w1: [7:0] src0, [14:8] sdst, [15] sdst_en, [18:16] src0_sel, [19] src0_sext, [20] src0_neg, [21] src0_abs,
      [23] src0_sgpr, [26:24] src1_sel, [27] src1_sext, [28] src1_neg, [29] src1_abs, [31] src1_sgpr

Integer compares: every i32 and u32 compare the decoder names (f, lt, eq, le, gt, ne, ge, t; v_cmp into vcc_lo and
into an SGPR, v_cmpx into EXEC) times every select and sext() on each source is decoded, translated, and the emitted
HLSL is evaluated on the CPU (the integer evaluator of startup_gfx2hlsl_bit_alu_check.py) over source words whose
bytes and words differ from the whole dword and carry the sign bit. The reference is written here with plain integer
arithmetic from the ISA meaning: the selected field, zero- or sign-extended to 32 bits, compared as the type says.
The value set is checked to separate a lowering that ignores the select or sext() from the reference.

Binary32 class compares read a DWORD float in SRC0 and a selected, optionally sign-extended integer mask in SRC1.
The emitted predicate is checked against the selected mask for each of the ten classes. Ordinary float predicates
with DWORD selections must lower as their e64 form with the same neg/abs; tests/startup_gfx2hlsl_f32_sdwa_check.py
checks them against the ISA reference.

Refused: a sub-dword or sign-extended class value, a byte or word select on an ordinary float source (its result
depends on the binary32 denormal mode), sext() on a float source, an SDWA modifier the decoder does not print for
compares, a missing or repeated select, and a destination that is not vcc_lo, EXEC or an SGPR.

The draw: a source-built pixel program multiplies the packed screen position by a constant and keeps a pixel when
v_cmpx_le_i32_sdwa -16, sext(v1) src0_sel:DWORD src1_sel:BYTE_1 holds. Survivors export the packed position; a
discarded pixel keeps the clear value -1.0. Two mutated translations, one that reads the whole dword (no select) and
one that zero-extends the byte (no sext()), must each fail the draw against the reference and match their own
plain-integer image.
The binary32 class draw selects the negative-normal mask bit from BYTE_1. Reading the whole dword must fail.
"""
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
from startup_flat_param_check import assemble  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

MASK = 0xffffffff
SELS = ['BYTE_0', 'BYTE_1', 'BYTE_2', 'BYTE_3', 'WORD_0', 'WORD_1', 'DWORD']
SEL_CODE = {name: code for code, name in enumerate(SELS)}
FIELD = {'BYTE_0': (0, 8), 'BYTE_1': (8, 8), 'BYTE_2': (16, 8), 'BYTE_3': (24, 8),
         'WORD_0': (0, 16), 'WORD_1': (16, 16), 'DWORD': (0, 32)}
SGPR_DST = 6

# Every 32-bit VOPC compare the decoder names, by opcode: (cmpx, condition, type).
COMPARES = {}
for _op, (_name, _) in D.VOPC_OPS.items():
    _parts = _name[:-len('_e32')].split('_')
    if _parts[-1] in ('f32', 'i32', 'u32') and _parts[1] in ('cmp', 'cmpx'):
        COMPARES[_op] = (_parts[1] == 'cmpx', '_'.join(_parts[2:-1]), _parts[-1])
INTEGER_CONDITIONS = ('f', 'lt', 'eq', 'le', 'gt', 'ne', 'ge', 't')

# Source words: every byte and word of most differs from the whole dword, both signs occur in every field, and the
# pairs include equal fields.
VALUES = [0x80ff7f01, 0x7f0180fe, 0x01fe807f, 0xfe7f0180, 0x00000000, 0xffffffff, 0x80008000, 0x7fff7fff]
PAIRS = [(a, b) for a in VALUES for b in VALUES]


def signed(value):
    return value - (1 << 32) if value >> 31 else value


def field(word, sel, sext):
    """The selected field of a 32-bit word, extended to 32 bits."""
    offset, width = FIELD[sel]
    value = (word >> offset) & ((1 << width) - 1)
    if sext and value >> (width - 1):
        value -= 1 << width
    return value & MASK


def reference(cond, typ, a, b):
    x, y = (signed(a), signed(b)) if typ == 'i32' else (a, b)
    return int({'f': False, 'lt': x < y, 'eq': x == y, 'le': x <= y, 'gt': x > y, 'ne': x != y, 'ge': x >= y,
                't': True}[cond])


# ---- encoding ------------------------------------------------------------------------------------------------------
def source(kind, number, sext=False, neg=False, absolute=False):
    return dict(kind=kind, number=number, sext=sext, neg=neg, absolute=absolute)


def vopc_sdwa(op, src0, src1, sdst=None, sel0='DWORD', sel1='DWORD'):
    """One VOPC SDWA instruction. sdst None leaves sdst_en clear (vcc_lo, or EXEC for v_cmpx)."""
    w0 = (0b0111110 << 25) | (op << 17) | (src1['number'] << 9) | 249
    w1 = src0['number'] | (SEL_CODE[sel0] << 16) | (SEL_CODE[sel1] << 24)
    if sdst is not None:
        w1 |= (sdst << 8) | (1 << 15)
    for shift, src in ((19, src0), (27, src1)):
        w1 |= (src['sext'] << shift) | (src['neg'] << (shift + 1)) | (src['absolute'] << (shift + 2))
        w1 |= (src['kind'] == 's') << (shift + 4)
    return struct.pack('<II', w0, w1)


# ---- translation ---------------------------------------------------------------------------------------------------
class Probe(G.Stage):
    """A Stage with no pipeline: enough state for decoded instructions to be translated."""

    def __init__(self, exec_known=True):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = exec_known


def decode_one(code):
    decoded = G.decode(code)
    assert len(decoded) == 1 and decoded[0][1] == 8, decoded
    return decoded[0][2], decoded[0][3]


def translate(code, exec_known=True):
    name, asm = decode_one(code)
    stage = Probe(exec_known)
    stage.one(name, asm)
    return asm, stage


def refusal(name, asm):
    try:
        Probe().one(name, asm)
    except G.Unsupported as error:
        return str(error)
    return None


failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def integer_forms():
    """(op, destination, src0 sel, src0 sext, src1 sel, src1 sext) for every accepted integer form."""
    for op, (cmpx, cond, typ) in sorted(COMPARES.items()):
        if typ == 'f32':
            continue
        for destination in (('exec',) if cmpx else ('vcc', 'sgpr')):
            for sel0 in SELS:
                for sext0 in (False, True):
                    for sel1 in SELS:
                        for sext1 in (False, True):
                            yield op, destination, sel0, sext0, sel1, sext1


def integer_code(op, destination, sel0, sext0, sel1, sext1):
    return vopc_sdwa(op, source('v', 1, sext0), source('v', 2, sext1),
                     SGPR_DST if destination == 'sgpr' else None, sel0, sel1)


def class_forms():
    for op, destination in ((0x88, 'vcc'), (0x88, 's6'), (0x98, 'exec')):
        for selection in SELS:
            for sext in (False, True):
                encoded = vopc_sdwa(op, source('v', 1), source('v', 2, sext=sext),
                                    SGPR_DST if destination == 's6' else None, 'DWORD', selection)
                yield destination, selection, sext, encoded


def controls():
    # 1. The encoder writes what the decoder reads, checked on LLVM 15 encodings from tests/rdna2_tools_check.py.
    known = [
        (vopc_sdwa(0xc4, source('v', 1), source('v', 2)), 'f904887d01000606'),
        (vopc_sdwa(0xc4, source('v', 1, sext=True), source('v', 2), SGPR_DST), 'f904887d01860e06'),
        (vopc_sdwa(0xc4, source('v', 1), source('v', 2), SGPR_DST, 'DWORD', 'WORD_1'), 'f904887d01860605'),
        (vopc_sdwa(0x01, source('v', 1, neg=True, absolute=True), source('v', 2), SGPR_DST), 'f904027c01863606'),
        (vopc_sdwa(0xd4, source('s', 1), source('v', 2)), 'f904a87d01008606'),
    ]
    check('encoder matches LLVM 15 encodings', all(code.hex() == want for code, want in known),
          str([(code.hex(), want) for code, want in known if code.hex() != want]))
    integer_ops = sorted(op for op, (_, cond, typ) in COMPARES.items() if typ != 'f32')
    check('32 integer compares, every condition', len(integer_ops) == 32 and all(
        sorted(cond for (cmpx, cond, typ) in COMPARES.values() if typ == t and cmpx == x)
        == sorted(INTEGER_CONDITIONS) for t in ('i32', 'u32') for x in (False, True)))

    # 2. The value set separates the reference from a lowering that ignores the select or sext().
    for sel in SELS[:-1]:
        for sext in (False, True):
            check('%s%s: a value whose field differs from the dword' % (sel, ' sext' if sext else ''),
                  any(field(v, sel, sext) != v for v in VALUES))
        check('%s: a value whose sign-extended field differs from the zero-extended one' % sel,
              any(field(v, sel, True) != field(v, sel, False) for v in VALUES))
    blind = []
    for cond in INTEGER_CONDITIONS[1:-1]:
        for typ in ('i32', 'u32'):
            for sel in SELS[:-1]:
                for sext in (False, True):
                    want = [reference(cond, typ, field(a, sel, sext), b) for a, b in PAIRS]
                    no_select = [reference(cond, typ, a, b) for a, b in PAIRS]
                    no_sext = [reference(cond, typ, field(a, sel, False), b) for a, b in PAIRS]
                    if want == no_select or (sext and want == no_sext):
                        blind.append((cond, typ, sel, sext))
    check('every condition separates the select and sext() on the value pairs', not blind, str(blind))

    # 3. Every accepted integer form evaluates to the reference on every pair.
    count, wrong = 0, []
    texts = {}
    for form in integer_forms():
        op, destination, sel0, sext0, sel1, sext1 = form
        cmpx, cond, typ = COMPARES[op]
        asm, stage = translate(integer_code(*form))
        target = {'exec': 'exec', 'vcc': 'vcc', 'sgpr': 's%d' % SGPR_DST}[destination]
        if len(stage.lines) != 1 or not stage.lines[0].startswith(target + ' = ') or not stage.lines[0].endswith(';'):
            wrong.append((asm, stage.lines))
            continue
        if (destination == 'sgpr') != (SGPR_DST in stage.masks) or stage.exec_known == cmpx:
            wrong.append((asm, 'destination bookkeeping', stage.masks, stage.exec_known))
            continue
        text = stage.lines[0][len(target) + 3:-1]
        texts[form] = text
        expr = Expr(text)
        for a, b in PAIRS:
            got = expr.run({'v1': a, 'v2': b, 'true': 1, 'false': 0})[0]
            want = reference(cond, typ, field(a, sel0, sext0), field(b, sel1, sext1))
            count += 1
            if got != want:
                wrong.append((asm, hex(a), hex(b), got, want))
                break
    check('%d integer forms x %d pairs (%d evaluations) equal the reference' % (
        len(texts), len(PAIRS), count), not wrong, str(wrong[:3]))
    check('v_cmp into vcc_lo, into an SGPR and v_cmpx into EXEC all covered',
          {d for _, d, *_ in texts} == {'exec', 'vcc', 'sgpr'} and len(texts) == 48 * 196)

    # 4. EXEC already reduced: the compare result is masked by the lane's EXEC bit, as in the non-SDWA path.
    wrong = []
    for op, (cmpx, cond, typ) in sorted(COMPARES.items()):
        if typ == 'f32':
            continue
        code = integer_code(op, 'exec' if cmpx else 'vcc', 'BYTE_1', True, 'WORD_0', False)
        asm, stage = translate(code, exec_known=False)
        expr = Expr(stage.lines[0].split(' = ', 1)[1][:-1])
        for a, b in PAIRS:
            for lane in (0, 1):
                want = lane & reference(cond, typ, field(a, 'BYTE_1', True), field(b, 'WORD_0', False))
                if expr.run({'v1': a, 'v2': b, 'exec': lane, 'true': 1, 'false': 0})[0] != want:
                    wrong.append((asm, hex(a), hex(b), lane))
    check('inactive lane: compare result masked by EXEC', not wrong, str(wrong[:3]))

    # 5. Scalar and inline-constant sources take the same select and sext().
    asm, stage = translate(vopc_sdwa(0x93, source('s', 208), source('v', 1, sext=True), None, 'DWORD', 'BYTE_1'))
    check('inline constant and sext(v1) BYTE_1 (the draw form)',
          asm == 'v_cmpx_le_i32_sdwa -16, sext(v1) src0_sel:DWORD src1_sel:BYTE_1' and stage.lines ==
          ['exec = ((int(0xfffffff0u) <= int(asuint(asint(v1 << 16u) >> 24u))) ? 1u : 0u);'], str((asm, stage.lines)))
    asm, stage = translate(vopc_sdwa(0x85, source('s', 3, sext=True), source('v', 2), SGPR_DST, 'BYTE_3', 'WORD_1'))
    expr = Expr(stage.lines[0].split(' = ', 1)[1][:-1])
    check('sext(s3) BYTE_3 against v2 WORD_1 into s6',
          asm == 'v_cmp_ne_i32_sdwa s6, sext(s3), v2 src0_sel:BYTE_3 src1_sel:WORD_1'
          and all(expr.run({'s3': a, 'v2': b})[0] == int(field(a, 'BYTE_3', True) != field(b, 'WORD_1', False))
                  for a, b in PAIRS), str((asm, stage.lines)))

    # 6. Float compares in their SDWA form (#1906): with DWORD selections every condition, modifier and destination
    # emits the text of the e64 form with the same modifiers. Byte and word selections, NaN and the compiler models
    # are checked in tests/startup_gfx2hlsl_f32_sdwa_check.py.
    differ, float_forms = [], 0
    modifiers = [(False, False), (True, False), (False, True), (True, True)]

    def spelled(reg, neg, absolute):
        reg = '|%s|' % reg if absolute else reg
        return '-' + reg if neg else reg

    for op, (cmpx, cond, typ) in sorted(COMPARES.items()):
        if typ != 'f32' or cond in ('class', 'tru'):
            continue
        for (neg0, abs0), (neg1, abs1) in ((m, n) for m in modifiers for n in modifiers):
            for destination in (('exec',) if cmpx else ('vcc', 'sgpr')):
                code = vopc_sdwa(op, source('v', 1, neg=neg0, absolute=abs0), source('v', 2, neg=neg1, absolute=abs1),
                                 SGPR_DST if destination == 'sgpr' else None)
                name, asm = decode_one(code)
                float_forms += 1
                e64 = name.replace('_sdwa', '_e64')
                operands = '%s, %s' % (spelled('v1', neg0, abs0), spelled('v2', neg1, abs1))
                if not cmpx:
                    operands = '%s, %s' % ('s%d' % SGPR_DST if destination == 'sgpr' else 'vcc_lo', operands)
                if refusal(name, asm) is not None:
                    differ.append((asm, refusal(name, asm)))
                    continue
                _, sdwa_stage = translate(code)
                e64_stage = Probe()
                e64_stage.one(e64, '%s %s' % (e64, operands))
                if sdwa_stage.lines != e64_stage.lines or sdwa_stage.masks != e64_stage.masks:
                    differ.append((asm, sdwa_stage.lines, e64_stage.lines))
    check('%d float SDWA forms with DWORD selections lower as the e64 form' % float_forms,
          not differ and float_forms == 15 * 16 * 3, str(differ[:2]))

    # 7. Refusals, each with its own message.
    def expect(label, name, asm, message):
        got = refusal(name, asm)
        check('refused: %s (%s)' % (label, got), got == message)

    class_count, class_errors = 0, []
    for destination, selection, sext, encoded in class_forms():
        name, asm = decode_one(encoded)
        try:
            _, class_stage = translate(encoded, exec_known=False)
        except G.Unsupported as error:
            class_errors.append((asm, str(error)))
            continue
        class_count += 1
        cmpx = destination == 'exec'
        if (name != ('v_cmpx_class_f32_sdwa' if cmpx else 'v_cmp_class_f32_sdwa')
                or len(class_stage.lines) != 1 or not class_stage.lines[0].startswith(destination + ' = ')
                or 'class32(F(v1))' not in class_stage.lines[0]
                or (SGPR_DST in class_stage.masks) != (destination == 's6')
                or class_stage.exec_known):
            class_errors.append((asm, class_stage.lines, class_stage.masks))
            continue
        expression = class_stage.lines[0].split(' = ', 1)[1][:-1].replace('class32(F(v1))', 'class_bit')
        emitted = Expr(expression)
        for raw in VALUES + [0x200, 0x20000]:
            for index in range(10):
                class_bit = 1 << index
                for lane in (0, 1):
                    actual = emitted.run({'class_bit': class_bit, 'v2': raw, 'exec': lane})[0]
                    expected = lane & int(bool(class_bit & field(raw, selection, sext)))
                    if actual != expected:
                        class_errors.append((asm, raw, index, lane, actual, expected))
    check('42 binary32 class SDWA forms use the selected mask and destination',
          class_count == 42 and not class_errors, str(class_errors[:3]))
    op = 0x88
    name, asm = decode_one(vopc_sdwa(op, source('v', 1), source('v', 2), None, 'WORD_0', 'BYTE_1'))
    expect('class float src0 sub-dword', name, asm, 'v_cmp_class_f32_sdwa requires a DWORD float source')
    _, class_stage = translate(vopc_sdwa(op, source('v', 1, neg=True, absolute=True), source('v', 2),
                                        None, 'DWORD', 'BYTE_1'))
    check('class float source preserves abs and neg', 'class32((-abs(F(v1))))' in class_stage.lines[0],
          str(class_stage.lines))
    # A byte or word select on an ordinary float source stays refused (#1906): the field is +0 or a denormal, so the
    # result depends on the binary32 denormal mode.
    for index in (0, 1):
        for sel in SELS[:-1]:
            sels = ('DWORD', sel) if index else (sel, 'DWORD')
            name, asm = decode_one(vopc_sdwa(0x01, source('v', 1), source('v', 2), None, *sels))
            expect('float src%d_sel:%s' % (index, sel), name, asm,
                   'v_cmp_lt_f32_sdwa: f32 SDWA byte/word selection depends on the binary32 denormal mode')
    expect('sext() on a float source', 'v_cmp_lt_f32_sdwa',
           'v_cmp_lt_f32_sdwa vcc_lo, sext(v1), v2 src0_sel:BYTE_0 src1_sel:DWORD',
           'v_cmp_lt_f32_sdwa: sext() on a float source')
    for extra in ('clamp', 'mul:2'):
        expect('float compare modifier %s' % extra, 'v_cmp_lt_f32_sdwa',
               'v_cmp_lt_f32_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD %s' % extra,
               'v_cmp_lt_f32_sdwa with SDWA modifier %s' % extra)
    base = 'v_cmp_lt_i32_sdwa vcc_lo, v1, v2'
    for extra in ('clamp', 'dst_sel:DWORD', 'dst_unused:UNUSED_PAD', 'src2_sel:DWORD', 'mul:2'):
        expect('modifier %s' % extra, 'v_cmp_lt_i32_sdwa', '%s src0_sel:DWORD src1_sel:DWORD %s' % (base, extra),
               'v_cmp_lt_i32_sdwa with SDWA modifier %s' % extra)
    expect('unknown select', 'v_cmp_lt_i32_sdwa', base + ' src0_sel:BYTE_4 src1_sel:DWORD',
           'v_cmp_lt_i32_sdwa with SDWA modifier src0_sel:BYTE_4')
    expect('repeated select', 'v_cmp_lt_i32_sdwa', base + ' src0_sel:DWORD src0_sel:BYTE_0 src1_sel:DWORD',
           'v_cmp_lt_i32_sdwa with SDWA modifier src0_sel:BYTE_0')
    expect('missing src1_sel', 'v_cmp_lt_i32_sdwa', base + ' src0_sel:DWORD', 'v_cmp_lt_i32_sdwa SDWA without src1_sel')
    expect('missing src0_sel', 'v_cmp_lt_i32_sdwa', base + ' src1_sel:DWORD', 'v_cmp_lt_i32_sdwa SDWA without src0_sel')
    expect('null destination', 'v_cmp_lt_i32_sdwa', 'v_cmp_lt_i32_sdwa null, v1, v2 src0_sel:DWORD src1_sel:DWORD',
           'v_cmp_lt_i32_sdwa into null')
    expect('missing destination', 'v_cmp_lt_i32_sdwa', 'v_cmp_lt_i32_sdwa v1, v2 src0_sel:DWORD src1_sel:DWORD',
           'v_cmp_lt_i32_sdwa with 2 operands')
    expect('cmpx destination', 'v_cmpx_lt_i32_sdwa', 'v_cmpx_lt_i32_sdwa s6, v1, v2 src0_sel:DWORD src1_sel:DWORD',
           'v_cmpx_lt_i32_sdwa with 3 operands')
    name, asm = decode_one(vopc_sdwa(0x0f, source('v', 1), source('v', 2)))
    expect('tru (no condition in the table, as in the non-SDWA form)', name, asm, 'v_cmp_tru_f32_sdwa')
    expect('16-bit compare', 'v_cmp_lt_i16_sdwa', 'v_cmp_lt_i16_sdwa s6, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD',
           'v_cmp_lt_i16_sdwa')

    # 8. The non-SDWA forms are unchanged.
    stage = Probe()
    stage.one('v_cmp_lt_i32_e32', 'v_cmp_lt_i32_e32 vcc_lo, v1, v2')
    stage.one('v_cmpx_gt_u32_e64', 'v_cmpx_gt_u32_e64 exec_lo, v3, s4')
    check('non-SDWA compares unchanged', stage.lines == ['vcc = ((int(v1) < int(v2)) ? 1u : 0u);',
                                                         'exec = ((v3 > s4) ? 1u : 0u);'], str(stage.lines))
    return texts


# ---- the pixel fixture ---------------------------------------------------------------------------------------------
WIDTH = HEIGHT = 16
MULTIPLIER = 0x9b5d
THRESHOLD = -16
CLEAR = 0xbf800000  # -1.0, the target's clear value: a discarded pixel keeps it
PS_SOURCE = """\
v_mul_u32_u24_e32 v1, 0x9b5d, v0
v_cmpx_le_i32_sdwa -16, sext(v1) src0_sel:DWORD src1_sel:BYTE_1
exp mrt0 v0, v0, v0, v0 done vm
s_endpgm
"""
MUL_WORDS = struct.pack('<II', (0x0b << 25) | (1 << 17) | 255, MULTIPLIER)
EXPORT_WORDS = struct.pack('<III', 0xf800180f, 0, 0xbf810000)
PS_WORDS = (MUL_WORDS + vopc_sdwa(0x93, source('s', 208), source('v', 1, sext=True), None, 'DWORD', 'BYTE_1')
            + EXPORT_WORDS)
CLASS_PS_SOURCE = PS_SOURCE.replace('v_cmpx_le_i32_sdwa -16, sext(v1)', 'v_cmpx_class_f32_sdwa -1.0, v1')
CLASS_PS_WORDS = (MUL_WORDS + vopc_sdwa(0x98, source('s', 243), source('v', 1), None, 'DWORD', 'BYTE_1')
                  + EXPORT_WORDS)
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
SELECTED = 'asuint(asint(v1 << 16u) >> 24u)'
CLASS_SELECTED = '((v1 >> 8u) & 0xffu)'
MUTATIONS = {
    'no select': (SELECTED, 'v1'),                     # reads the whole dword
    'no sext': (SELECTED, '((v1 >> 8u) & 0xffu)'),      # zero-extends the byte
}


def survives(position, how):
    product = (position * MULTIPLIER) & MASK  # both factors are below 2^24
    value = {'reference': field(product, 'BYTE_1', True), 'no select': product,
             'no sext': field(product, 'BYTE_1', False)}[how]
    return THRESHOLD <= signed(value)


def image(how):
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            position = x | (y << 16)
            words += [position] * 4 if survives(position, how) else [CLEAR] * 4
    return words


def pixel_hlsl(how='reference'):
    text = G.Pipeline(VS_WORDS, PS_WORDS, root_blob(), ps_hw_inputs=PACKED).hlsl_ps()
    if how != 'reference':
        old, new = MUTATIONS[how]
        assert text.count(old) == 1, how
        text = text.replace(old, new)
    return text


def class_image(select=True):
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            position = x | (y << 16)
            mask = (position * MULTIPLIER) & MASK
            if select:
                mask = field(mask, 'BYTE_1', False)
            words += [position if mask & 0x8 else CLEAR] * 4  # -1.0 is negative normal
    return words


def class_pixel_hlsl(select=True):
    text = G.Pipeline(VS_WORDS, CLASS_PS_WORDS, root_blob(), ps_hw_inputs=PACKED).hlsl_ps()
    if not select:
        assert text.count(CLASS_SELECTED) == 1
        text = text.replace(CLASS_SELECTED, 'v1')
    return text


def check_fixture():
    decoded = [asm for _o, _s, _n, asm in G.decode(PS_WORDS)]
    check('pixel fixture decodes to its source', decoded == PS_SOURCE.splitlines(), str(decoded))
    text = pixel_hlsl()
    check('pixel fixture keeps a pixel through EXEC',
          'exec = ((int(0xfffffff0u) <= int(%s)) ? 1u : 0u);' % SELECTED in text
          and 'if (!(exec & 1)) discard;' in text, text)
    reference = image('reference')
    kept = sum(reference[i] != CLEAR for i in range(0, len(reference), 4))
    check('reference keeps %d of %d pixels' % (kept, WIDTH * HEIGHT), 0 < kept < WIDTH * HEIGHT)
    for how in MUTATIONS:
        other = image(how)
        differing = sum(other[i:i + 4] != reference[i:i + 4] for i in range(0, len(other), 4))
        check('%s mutation changes %d pixels' % (how, differing), differing > 0)
    # The evaluator on each pixel stage's compare reads the same decisions as the plain-integer images.
    for how in ('reference', *MUTATIONS):
        line = next(ln.strip() for ln in pixel_hlsl(how).splitlines() if ln.strip().startswith('exec = '))
        expr = Expr(line[len('exec = '):-1])
        product = lambda p: (p * MULTIPLIER) & MASK  # noqa: E731
        check('%s: evaluated compare equals the image' % how, all(
            expr.run({'v1': product(x | y << 16)})[0] == survives(x | y << 16, how)
            for y in range(HEIGHT) for x in range(WIDTH)))
    decoded_class = [asm for _o, _s, _n, asm in G.decode(CLASS_PS_WORDS)]
    check('class pixel fixture decodes to its source', decoded_class == CLASS_PS_SOURCE.splitlines(), str(decoded_class))
    class_reference = class_image()
    kept = sum(class_reference[i] != CLEAR for i in range(0, len(class_reference), 4))
    check('class reference keeps %d of %d pixels' % (kept, WIDTH * HEIGHT), 0 < kept < WIDTH * HEIGHT)
    check('ignoring the class mask select changes the image', class_reference != class_image(select=False))


def all_codes():
    codes = [integer_code(*form) for form in integer_forms()]
    modifiers = [(False, False), (True, False), (False, True), (True, True)]
    for op, (cmpx, cond, typ) in sorted(COMPARES.items()):
        if typ == 'f32' and cond not in ('class', 'tru'):
            for (neg0, abs0), (neg1, abs1) in ((m, n) for m in modifiers for n in modifiers):
                codes.append(vopc_sdwa(op, source('v', 1, neg=neg0, absolute=abs0),
                                       source('v', 2, neg=neg1, absolute=abs1), None if cmpx else SGPR_DST))
    codes.extend(encoded for _, _, _, encoded in class_forms())
    codes.append(vopc_sdwa(0x85, source('s', 3, sext=True), source('v', 2), SGPR_DST, 'BYTE_3', 'WORD_1'))
    return codes


def check_binary(llvm_mc):
    codes = all_codes()
    source_text = ''.join(decode_one(code)[1] + '\n' for code in codes)
    built = assemble(llvm_mc, source_text)
    check('llvm-mc assembles %d decoded VOPC SDWA forms to the same words' % len(codes), built == b''.join(codes))
    check('llvm-mc assembles the pixel fixture to the same words', assemble(llvm_mc, PS_SOURCE) == PS_WORDS)
    check('llvm-mc assembles the class pixel fixture to the same words',
          assemble(llvm_mc, CLASS_PS_SOURCE) == CLASS_PS_WORDS)


def check_decode(library):
    from pixel_aot import disassemble
    codes = all_codes()
    texts = disassemble(b''.join(codes), library)
    ours = [decode_one(code)[1] for code in codes]
    wrong = [(a, b) for a, b in zip(texts, ours) if a != b]
    check('LLVM disassembles %d VOPC SDWA forms as tools/disasm_rdna2.py does' % len(codes),
          len(texts) == len(codes) and not wrong, str(wrong[:3]))
    check('LLVM disassembles the pixel fixture as its source', disassemble(PS_WORDS, library) == PS_SOURCE.splitlines(),
          str(disassemble(PS_WORDS, library)))
    class_text = disassemble(CLASS_PS_WORDS, library)
    check('LLVM disassembles the class pixel fixture as its source', class_text == CLASS_PS_SOURCE.splitlines(),
          str(class_text))


def words_file(directory, name, words):
    path = directory / ('%s.bin' % name)
    path.write_bytes(struct.pack('<%dI' % len(words), *words))
    return path


def draw(renderer, adapter, vs, ps, expected):
    run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT),
                          str(expected)], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(1 if failures else 77)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        reference = words_file(directory, 'reference', image('reference'))
        class_reference = words_file(directory, 'class_reference', class_image())
        class_unselected = words_file(directory, 'class_unselected', class_image(select=False))
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            ps = compile_hlsl(compiler, kind, pixel_hlsl(), 'ps_%s' % kind, ps_profile, directory)
            run = draw(renderer, adapter, vs, ps, reference)
            print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                                  if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
            check('%s %s: all %d pixels match the reference' % (kind, adapter, WIDTH * HEIGHT),
                  run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                  run.stdout + run.stderr)
            for how in MUTATIONS:
                bad = compile_hlsl(compiler, kind, pixel_hlsl(how), 'ps_%s_%s' % (how.replace(' ', '_'), kind),
                                   ps_profile, directory)
                run = draw(renderer, adapter, vs, bad, reference)
                lines = [ln for ln in run.stderr.splitlines() if 'pixel' in ln]
                check('%s %s %s mutation fails against the reference: %s' % (kind, adapter, how, ' / '.join(lines)),
                      run.returncode != 0 and lines and lines[0].startswith('pixel mismatch'), run.stdout + run.stderr)
                own = words_file(directory, how.replace(' ', '_'), image(how))
                run = draw(renderer, adapter, vs, bad, own)
                check('%s %s %s mutation matches its own plain-integer image' % (kind, adapter, how),
                      run.returncode == 0, run.stdout + run.stderr)
            class_ps = compile_hlsl(compiler, kind, class_pixel_hlsl(), 'ps_class_%s' % kind, ps_profile, directory)
            run = draw(renderer, adapter, vs, class_ps, class_reference)
            check('%s %s: class compare matches all %d pixels' % (kind, adapter, WIDTH * HEIGHT),
                  run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                  run.stdout + run.stderr)
            bad_class = compile_hlsl(compiler, kind, class_pixel_hlsl(select=False), 'ps_class_bad_%s' % kind,
                                     ps_profile, directory)
            run = draw(renderer, adapter, vs, bad_class, class_reference)
            check('%s %s: class compare without the select fails' % (kind, adapter),
                  run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)
            run = draw(renderer, adapter, vs, bad_class, class_unselected)
            check('%s %s: class mutation matches the whole-dword image' % (kind, adapter),
                  run.returncode == 0, run.stdout + run.stderr)


def main():
    if len(sys.argv) == 1:
        controls()
        check_fixture()
    else:
        mode, *args = sys.argv[1:]
        if mode == 'binary':
            check_binary(*args)
        elif mode == 'decode':
            check_decode(*args)
        elif mode == 'draw':
            check_fixture()
            check_draw(*args)
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
