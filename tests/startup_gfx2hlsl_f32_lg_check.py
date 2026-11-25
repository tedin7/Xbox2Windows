"""f32 VOPC conditions in tools/startup/gfx2hlsl.py, lg and nlg on NaN operands among them (#1920).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     assemble the decoded text of every checked form and of the pixel fixtures with
                                       llvm-mc and compare with the words built here
  decode <libLLVM>                     disassemble the same words with LLVM's C API and compare with
                                       tools/disasm_rdna2.py
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the pixel fixtures with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw 16 x 16 pixels each and compare every pixel with the ISA reference
                                       (needs tests/offscreen_check.cpp built as offscreen_check)

In the gfx10 ISA V_CMP_LG_F32 is "less or greater": ordered and not equal, false when either source is NaN.
V_CMP_NLG_F32 is its negation, true when either source is NaN. HLSL != is unordered (true with a NaN), so the old
lowering `a != b` and `!(a != b)` gave the opposite result whenever a source was NaN. The lowering is now
`(ORDERED && (a < b || a > b))` and its negation, ORDERED testing both source words for NaN on their bits: DXC
marks float compares fast unless -Gis, and the draw records whether the same text without the bit test holds.

The reference is written here from the ISA definitions with integer arithmetic on the bit patterns: a NaN is an
exponent of all ones with a non-zero mantissa; ordered sources compare by sign and magnitude, with +0 equal to -0 and
denormals kept; lt, eq, le, gt, lg and ge are false when a source is NaN; the n-forms are their negations; o is both
sources ordered and u either unordered; f and t are constant.

Controls: every f32 condition in the compare table (v_cmp and v_cmpx; the e32 form and the e64 form under every
neg/abs combination on both sources) is decoded from words built here, translated, and the emitted HLSL is evaluated
on the CPU with IEEE binary32 comparisons over every pair of 16 source words: +-0, +-min denormal, +-1.0, +-max
finite, +-inf, +-quiet NaN, +-signalling NaN and +-NaN with a payload. The same forms with the old `!=` lowering
restored must disagree with the reference. The f32 SDWA lg/nlg forms with DWORD selections must lower as the e64
form; tests/startup_gfx2hlsl_f32_sdwa_check.py checks every f32 SDWA form.

The draws: a source-built pixel program builds one of those 16 words from the pixel's x and one from its y with
integer operations, so 16 x 16 pixels cover every pair: finite unequal, equal (+0 and -0 among them), infinite and
NaN sources. Fixture `cmp` writes four v_cmp results as 1.0 or 0.0: v_cmp_lg_f32_e32, v_cmp_nlg_f32_e64 (-a, |b|),
v_cmp_lg_f32_e64 (-|a|, -b) and v_cmp_nlg_f32_e32. Each `cmpx_*` fixture keeps a pixel when its v_cmpx form holds
and exports the packed position; a discarded pixel keeps the clear value -1.0. D3D lets a backend flush binary32
denormal sources of a compare to signed zero, so every fixture must match the reference image with denormals kept
or the one with them flushed, the same choice for all fixtures of one compiler and adapter. Each translation with
the old `!=` lowering restored must fail its draw against both images and match its own image.
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
from startup_flat_param_check import assemble  # noqa: E402
from startup_gfx2hlsl_vopc_sdwa_check import COMPARES, Probe, source, vopc_sdwa  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

MASK = 0xffffffff
SIGN = 0x80000000
SGPR_DST = 6
# The 16 source words: each class with the sign bit clear and set.
CLASSES = [0x00000000, 0x00000001, 0x3f800000, 0x7f7fffff, 0x7f800000, 0x7fc00000, 0x7f800001, 0x7fc12345]
VALUES = [c | sign for c in CLASSES for sign in (0, SIGN)]
PAIRS = [(a, b) for a in VALUES for b in VALUES]
# Every f32 condition the compare table lowers; v_cmp_class_f32 and v_cmp_tru_f32 stay refused.
F32_OPS = sorted(op for op, (_, cond, typ) in COMPARES.items() if typ == 'f32' and cond not in ('class', 'tru'))
MODIFIERS = [(neg, absolute) for neg in (False, True) for absolute in (False, True)]


# ---- the ISA reference, on bit patterns -------------------------------------------------------------------------
def is_nan(bits):
    return (bits & 0x7fffffff) > 0x7f800000


def key(bits):
    """Order of an ordered binary32 word: sign and magnitude, +0 and -0 equal, denormals kept."""
    magnitude = bits & 0x7fffffff
    return -magnitude if bits & SIGN else magnitude


def modified(bits, neg, absolute):
    """The VOP3/SDWA source modifiers: abs clears the sign bit, then neg flips it."""
    if absolute:
        bits &= 0x7fffffff
    return bits ^ SIGN if neg else bits


ORDERED = {'lt': lambda x, y: x < y, 'eq': lambda x, y: x == y, 'le': lambda x, y: x <= y,
           'gt': lambda x, y: x > y, 'lg': lambda x, y: x != y, 'ge': lambda x, y: x >= y}


def reference(cond, a, b):
    unordered = is_nan(a) or is_nan(b)
    if cond in ('f', 't'):
        return int(cond == 't')
    if cond in ('o', 'u'):
        return int(unordered == (cond == 'u'))
    if cond in ORDERED:
        return int(not unordered and ORDERED[cond](key(a), key(b)))
    assert cond[0] == 'n' and cond[1:] in ORDERED, cond
    return 1 - reference(cond[1:], a, b)


def old_reference(cond, a, b):
    """The lowering before #1920: lg as unordered not-equal, nlg its negation."""
    if cond in ('lg', 'nlg'):
        ne = is_nan(a) or is_nan(b) or key(a) != key(b)
        return int(ne != (cond == 'nlg'))
    return reference(cond, a, b)


# ---- a small evaluator for the compare HLSL the lowering emits (uint words, binary32 floats, bools) -------------
TOKEN = re.compile(r'\s*(0x[0-9a-fA-F]+u?|\d+\.\d+|\d+u?|[A-Za-z_]\w*|<=|>=|==|!=|\|\||&&|[-&!<>?:(),])')
LEVELS = [['||'], ['&&'], ['&'], ['==', '!='], ['<', '>', '<=', '>=']]


def as_float(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


class Expr:
    """Values are ('u', int) for uint words, ('f', float) for binary32 and ('b', 0|1) for bools. Python float
    comparisons follow IEEE 754: every ordered relation is false with a NaN and != is true."""

    def __init__(self, text):
        self.tokens = TOKEN.findall(text)
        assert ''.join(self.tokens) == re.sub(r'\s+', '', text), text
        self.pos = 0
        self.tree = self.ternary()
        assert self.pos == len(self.tokens), text

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self):
        self.pos += 1
        return self.tokens[self.pos - 1]

    def ternary(self):
        cond = self.binary(0)
        if self.peek() != '?':
            return cond
        self.take()
        yes = self.ternary()
        assert self.take() == ':'
        return ('?', cond, yes, self.ternary())

    def binary(self, level):
        if level == len(LEVELS):
            return self.unary()
        node = self.binary(level + 1)
        while self.peek() in LEVELS[level]:
            node = (self.take(), node, self.binary(level + 1))
        return node

    def unary(self):
        token = self.take()
        if token in ('-', '!'):
            return ('u' + token, self.unary())
        if token == '(':
            node = self.ternary()
            assert self.take() == ')'
            return node
        if '.' in token:
            return ('const', ('f', float(token)))
        if token[0].isdigit():
            return ('const', ('u', int(token.rstrip('u'), 0) & MASK))
        if token in ('true', 'false'):
            return ('const', ('b', int(token == 'true')))
        if self.peek() != '(':
            return ('var', token)
        self.take()
        args = [self.ternary()]
        while self.peek() == ',':
            self.take()
            args.append(self.ternary())
        assert self.take() == ')'
        return ('call', token, args)

    def run(self, env, node=None):
        return self.eval(env, node or self.tree)[1]

    def eval(self, env, node):
        kind = node[0]
        if kind == 'const':
            return node[1]
        if kind == 'var':
            return 'u', env[node[1]]
        if kind == '?':
            return self.eval(env, node[2] if self.truth(self.eval(env, node[1])) else node[3])
        if kind == 'call':
            (k, v), = [self.eval(env, a) for a in node[2]]
            if node[1] in ('F', 'asfloat'):
                assert k == 'u'
                return 'f', as_float(v)
            assert node[1] == 'abs' and k == 'f', node
            return 'f', math.fabs(v)
        if kind == 'u-':
            k, v = self.eval(env, node[1])
            assert k == 'f'
            return 'f', -v
        if kind == 'u!':
            return 'b', int(not self.truth(self.eval(env, node[1])))
        if kind in ('||', '&&'):
            a = self.truth(self.eval(env, node[1]))
            b = self.truth(self.eval(env, node[2]))
            return 'b', int(a or b) if kind == '||' else int(a and b)
        (ka, a), (kb, b) = self.eval(env, node[1]), self.eval(env, node[2])
        if kind == '&':
            assert 'f' not in (ka, kb)
            return 'u', a & b
        assert ka == kb and ka in ('f', 'u'), node
        return 'b', int({'==': a == b, '!=': a != b, '<': a < b, '>': a > b, '<=': a <= b, '>=': a >= b}[kind])

    @staticmethod
    def truth(value):
        kind, v = value
        assert kind in ('b', 'u'), value
        return v != 0


# ---- encoding ---------------------------------------------------------------------------------------------------
VGPR = 256
VCC_LO = 106
INLINE = {0: 128, 1: 129, 15: 143, 16: 144, 31: 159, 0x3f800000: 242}  # integers and 1.0


def operand(value):
    """(source code, literal or None) of a 32-bit constant."""
    if value in INLINE:
        return INLINE[value], None
    if 1 <= value <= 64:
        return 128 + value, None
    return 255, value


def vop1_mov(vdst, value):
    code, literal = operand(value)
    return struct.pack('<I', (0x3f << 25) | (vdst << 17) | (0x01 << 9) | code) + (
        struct.pack('<I', literal) if literal is not None else b'')


def vop2(op, vdst, src0, vsrc1):
    """src0 is a constant (int) or ('v', n)."""
    if isinstance(src0, tuple):
        code, literal = VGPR + src0[1], None
    else:
        code, literal = operand(src0)
    return struct.pack('<I', (op << 25) | (vdst << 17) | (vsrc1 << 9) | code) + (
        struct.pack('<I', literal) if literal is not None else b'')


V_CNDMASK, V_LSHRREV, V_LSHLREV, V_AND, V_XOR = 0x01, 0x16, 0x1a, 0x1b, 0x1d
V_CMP_NE_U32 = 0xc5


def vopc_e32(op, src0, vsrc1):
    """src0 is a constant (int) or ('v', n); vsrc1 a VGPR number."""
    code = VGPR + src0[1] if isinstance(src0, tuple) else operand(src0)[0]
    return struct.pack('<I', (0b0111110 << 25) | (op << 17) | (vsrc1 << 9) | code)


def vop3_compare(op, sdst, src0, src1, neg=(False, False), absolute=(False, False)):
    """VOP3 compare of two VGPRs. A v_cmpx keeps sdst 0, as llvm-mc encodes it for gfx10."""
    w0 = (0b110101 << 26) | (op << 16) | ((absolute[0] | absolute[1] << 1) << 8) | sdst
    w1 = (VGPR + src0) | ((VGPR + src1) << 9) | ((neg[0] | neg[1] << 1) << 29)
    return struct.pack('<II', w0, w1)


def cndmask_e64(vdst, mask):
    """v_cndmask_b32_e64 vdst, 0, 1.0, mask: 1.0 where the lane's bit of mask is set."""
    return struct.pack('<II', (0b110101 << 26) | (0x101 << 16) | vdst, 128 | (242 << 9) | (mask << 18))


def compare_code(op, form, mods=((False, False), (False, False))):
    """One compare of v1 and v2. form is 'e32' (vcc_lo or EXEC) or 'e64' (s6, or EXEC for v_cmpx)."""
    cmpx = COMPARES[op][0]
    (neg0, abs0), (neg1, abs1) = mods
    if form == 'e32':
        return vopc_e32(op, ('v', 1), 2)
    return vop3_compare(op, 0 if cmpx else SGPR_DST, 1, 2, (neg0, neg1), (abs0, abs1))


def all_forms():
    """(op, form, mods, destination) for every checked compare form."""
    for op in F32_OPS:
        cmpx = COMPARES[op][0]
        yield op, 'e32', ((False, False), (False, False)), 'exec' if cmpx else 'vcc'
        for mods in ((m, n) for m in MODIFIERS for n in MODIFIERS):
            yield op, 'e64', mods, 'exec' if cmpx else 'sgpr'


def llvm_text(asm):
    """tools/disasm_rdna2.py prints the unused sdst field of a gfx10 v_cmpx_*_e64; LLVM 15 prints no sdst there."""
    return re.sub(r'^(v_cmpx_\w+_e64) s0, ', r'\1 ', asm)


# ---- translation ------------------------------------------------------------------------------------------------
failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def decode_one(code):
    decoded = G.decode(code)
    assert len(decoded) == 1 and decoded[0][1] == len(code), decoded
    return decoded[0][2], decoded[0][3]


def translate(code, exec_known=True):
    name, asm = decode_one(code)
    stage = Probe(exec_known)
    stage.one(name, asm)
    return asm, stage


# The new lg text, `(ORDERED && (X < Y || X > Y))` with ORDERED the bit test of both source words, and the old one,
# `(X != Y)`: X and Y are F(vN) under optional abs() and negation, so neither holds a ' < ', ' > ' or '||'.
ORDERED_TEXT = r'\((?P<w{0}>\w+) & 0x7fffffffu\) <= 0x7f800000u'
LG_TEXT = re.compile(r'\(%s && %s && \((?P<x>[^<>|]+?) < (?P<y>[^<>|]+?) \|\| (?P=x) > (?P=y)\)\)' % (
    ORDERED_TEXT.format(0), ORDERED_TEXT.format(1)))


def old_lowering(text):
    """The same text with lg and nlg lowered as before #1920."""
    return LG_TEXT.sub(lambda m: '(%s != %s)' % (m.group('x'), m.group('y')), text)


def unguarded(text):
    """lg as less or greater without the bit test: exact in HLSL, not under DXC's fast compares on every driver."""
    return LG_TEXT.sub(lambda m: '(%s < %s || %s > %s)' % (m.group('x'), m.group('y'), m.group('x'), m.group('y')),
                       text)


def first_mismatch(expr, cond, mods, model=reference):
    (neg0, abs0), (neg1, abs1) = mods
    for a, b in PAIRS:
        want = model(cond, modified(a, neg0, abs0), modified(b, neg1, abs1))
        got = expr.run({'v1': a, 'v2': b})
        if got != want:
            return a, b, got, want
    return None


# The lowering of every other condition before #1920, for v_cmp_*_f32_e32 vcc_lo, v1, v2. nlt, nle, ngt, nge, o and u
# changed with #1936 and are checked in tests/startup_gfx2hlsl_f32_nan_check.py.
UNCHANGED = {'f': 'false', 'lt': 'F(v1) < F(v2)', 'eq': 'F(v1) == F(v2)', 'le': 'F(v1) <= F(v2)',
             'gt': 'F(v1) > F(v2)', 'ge': 'F(v1) >= F(v2)', 'neq': '!(F(v1) == F(v2))'}
NAN_FORMS = ('nlt', 'nle', 'ngt', 'nge', 'o', 'u')


def controls():
    # 1. The value set and the reference.
    check('16 source words: +-0, +-min denormal, +-1.0, +-max finite, +-inf, +-quiet, signalling and payload NaN',
          len(set(VALUES)) == 16 and sum(map(is_nan, VALUES)) == 6
          and sum(math.isnan(as_float(v)) for v in VALUES) == 6
          and {as_float(v) for v in VALUES if not is_nan(v)} == {0.0, 2.0 ** -149, -2.0 ** -149, 1.0, -1.0,
                                                                   as_float(0x7f7fffff), -as_float(0x7f7fffff),
                                                                   math.inf, -math.inf})
    ordered = [(a, b) for a, b in PAIRS if not is_nan(a) and not is_nan(b)]
    check('reference: the ordered relations match IEEE binary32 on every ordered pair', all(
        reference(cond, a, b) == int(ORDERED[cond](as_float(a), as_float(b))) for cond in ORDERED for a, b in ordered))
    check('reference: +0 and -0 are equal, the min denormals are not',
          reference('eq', 0, SIGN) and not reference('lg', 0, SIGN) and reference('lg', 1, SIGN | 1)
          and reference('lt', SIGN | 1, 0))
    nan_pairs = [(a, b) for a, b in PAIRS if is_nan(a) or is_nan(b)]
    check('reference: %d pairs with a NaN: lg false, nlg true, u true, o false' % len(nan_pairs), all(
        (reference('lg', a, b), reference('nlg', a, b), reference('u', a, b), reference('o', a, b)) == (0, 1, 1, 0)
        for a, b in nan_pairs))
    check('reference: the old lowering differs on exactly the NaN pairs', all(
        (old_reference(cond, a, b) != reference(cond, a, b)) == (is_nan(a) or is_nan(b))
        for cond in ('lg', 'nlg') for a, b in PAIRS))
    check('every f32 condition of the table is checked (%d opcodes)' % len(F32_OPS), len(F32_OPS) == 30 and sorted(
        {COMPARES[op][1] for op in F32_OPS}) == sorted(set(UNCHANGED) | {'lg', 'nlg'} | set(NAN_FORMS)))

    # 2. The new lowering's text; every other condition keeps its text.
    asm, stage = translate(compare_code(0x05, 'e32'))
    ordered = '(v1 & 0x7fffffffu) <= 0x7f800000u && (v2 & 0x7fffffffu) <= 0x7f800000u'
    check('v_cmp_lg_f32_e32: %s' % stage.lines, stage.lines == [
        'vcc = (((%s && (F(v1) < F(v2) || F(v1) > F(v2)))) ? 1u : 0u);' % ordered])
    asm, stage = translate(compare_code(0x0a, 'e32'))
    check('v_cmp_nlg_f32_e32: %s' % stage.lines, stage.lines == [
        'vcc = ((!(%s && (F(v1) < F(v2) || F(v1) > F(v2)))) ? 1u : 0u);' % ordered])
    asm, stage = translate(compare_code(0x1a, 'e64', ((True, True), (False, True))))
    check('v_cmpx_nlg_f32_e64 tests the words under the modifiers: %s' % stage.lines, stage.lines == [
        'exec = ((!(%s && ((-abs(F(v1))) < abs(F(v2)) || (-abs(F(v1))) > abs(F(v2))))) ? 1u : 0u);' % ordered])
    changed = []
    for op in F32_OPS:
        cmpx, cond, _ = COMPARES[op]
        if cmpx or cond in ('lg', 'nlg') + NAN_FORMS:
            continue
        asm, stage = translate(compare_code(op, 'e32'))
        if stage.lines != ['vcc = ((%s) ? 1u : 0u);' % UNCHANGED[cond]]:
            changed.append((asm, stage.lines))
    check('the %d other conditions keep their lowering' % len(UNCHANGED), not changed, str(changed))

    # 3. Every form of every condition equals the reference on every pair; the old lg/nlg lowering does not.
    forms = evaluations = 0
    wrong, caught, missed, first = [], 0, [], None
    for op, form, mods, destination in all_forms():
        cmpx, cond, _ = COMPARES[op]
        asm, stage = translate(compare_code(op, form, mods))
        target = {'exec': 'exec', 'vcc': 'vcc', 'sgpr': 's%d' % SGPR_DST}[destination]
        if (len(stage.lines) != 1 or not stage.lines[0].startswith(target + ' = ')
                or (destination == 'sgpr') != (SGPR_DST in stage.masks) or stage.exec_known == cmpx):
            wrong.append((asm, stage.lines, 'destination'))
            continue
        text = stage.lines[0][len(target) + 3:-1]
        forms += 1
        evaluations += len(PAIRS)
        miss = first_mismatch(Expr(text), cond, mods)
        if miss:
            wrong.append((asm, ['%08x' % miss[0], '%08x' % miss[1]], miss[2:]))
        if cond in ('lg', 'nlg'):
            # Under IEEE comparisons the bit test changes nothing; it is there for fast compares.
            loose = unguarded(text)
            if loose == text or first_mismatch(Expr(loose), cond, mods):
                wrong.append((asm, 'without the bit test', loose))
            old = old_lowering(text)
            miss = first_mismatch(Expr(old), cond, mods) if old != text else None
            if miss and not first_mismatch(Expr(old), cond, mods, old_reference):
                caught += 1
                first = first or '%s with %s: a=%08x b=%08x gives %d, the ISA %d' % (asm, old, *miss)
            else:
                missed.append(asm)
    check('%d f32 forms x %d pairs (%d evaluations) equal the ISA reference' % (forms, len(PAIRS), evaluations),
          not wrong and forms == len(F32_OPS) * 17, str(wrong[:3]))
    check('old != lowering fails %d of %d lg/nlg forms against the reference and matches its own model' % (
        caught, caught + len(missed)), caught == 4 * 17 and not missed, str(missed[:3]))
    print('  first: ' + str(first))

    # 4. EXEC already reduced: the compare result is masked by the lane's EXEC bit.
    wrong = []
    for op in F32_OPS:
        cmpx, cond, _ = COMPARES[op]
        if cond not in ('lg', 'nlg'):
            continue
        for form in ('e32', 'e64'):
            mods = ((True, False), (False, True))
            asm, stage = translate(compare_code(op, form, mods), exec_known=False)
            expr = Expr(stage.lines[0].split(' = ', 1)[1][:-1])
            for a, b in PAIRS:
                for lane in (0, 1):
                    want = lane & reference(cond, modified(a, True, False), modified(b, False, True))
                    if form == 'e32':
                        want = lane & reference(cond, a, b)
                    if expr.run({'v1': a, 'v2': b, 'exec': lane}) != want:
                        wrong.append((asm, hex(a), hex(b), lane))
    check('inactive lane: lg/nlg results masked by EXEC in the e32 and e64 forms', not wrong, str(wrong[:3]))

    # 5. The f32 SDWA lg/nlg compares (#1906) with DWORD selections: the same text as the e64 form with the same
    # modifiers. Every f32 SDWA form is checked in tests/startup_gfx2hlsl_f32_sdwa_check.py.
    differ = []
    for op in F32_OPS:
        cmpx, cond, _ = COMPARES[op]
        if cond not in ('lg', 'nlg'):
            continue
        for neg, absolute in MODIFIERS:
            name, asm = decode_one(vopc_sdwa(op, source('v', 1, neg=neg, absolute=absolute), source('v', 2),
                                             None if cmpx else SGPR_DST))
            try:
                stage = Probe()
                stage.one(name, asm)
            except G.Unsupported as error:
                differ.append((asm, str(error)))
                continue
            _asm, e64 = translate(compare_code(op, 'e64', ((neg, absolute), (False, False))))
            if stage.lines != e64.lines:
                differ.append((asm, stage.lines, e64.lines))
    check('f32 SDWA lg/nlg compares with DWORD selections lower as the e64 form (16 forms)', not differ,
          str(differ[:3]))


# ---- the pixel fixtures -----------------------------------------------------------------------------------------
WIDTH = HEIGHT = 16
CLEAR = 0xbf800000  # -1.0, the target's clear value: a discarded pixel keeps it
ONE = 0x3f800000
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0


def build_value(dst, index):
    """dst = CLASSES[index >> 1] with the sign bit (index & 1), from integer operations; v5 is scratch."""
    text = 'v_lshrrev_b32_e32 v5, 1, v%d\nv_mov_b32_e32 v%d, 0\n' % (index, dst)
    words = vop2(V_LSHRREV, 5, 1, index) + vop1_mov(dst, 0)
    for k in range(1, 8):
        constant = {1: '1', ONE: '1.0'}.get(CLASSES[k], '0x%x' % CLASSES[k])
        text += 'v_cmp_ne_u32_e32 vcc_lo, %d, v5\nv_cndmask_b32_e32 v%d, %s, v%d, vcc_lo\n' % (k, dst, constant, dst)
        words += vopc_e32(V_CMP_NE_U32, k, 5) + vop2(V_CNDMASK, dst, CLASSES[k], dst)
    text += 'v_lshlrev_b32_e32 v5, 31, v%d\nv_xor_b32_e32 v%d, v%d, v5\n' % (index, dst, dst)
    words += vop2(V_LSHLREV, 5, 31, index) + vop2(V_XOR, dst, ('v', dst), 5)
    return text, words


def prologue():
    """v3 = VALUES[x], v4 = VALUES[y] for the packed position x | y << 16 in v0."""
    text = 'v_and_b32_e32 v1, 15, v0\nv_lshrrev_b32_e32 v2, 16, v0\n'
    words = vop2(V_AND, 1, 15, 0) + vop2(V_LSHRREV, 2, 16, 0)
    for dst, index in ((3, 1), (4, 2)):
        t, w = build_value(dst, index)
        text, words = text + t, words + w
    return text, words


EXPORT_PACKED = ('exp mrt0 v0, v0, v0, v0 done vm\ns_endpgm\n', struct.pack('<III', 0xf800180f, 0, 0xbf810000))
NEG, ABS = (True, False), (False, True)
NEG_ABS, PLAIN = (True, True), (False, False)
# name -> (body text, body words, [(condition, src0 modifiers, src1 modifiers)] per exported word, or one for v_cmpx)
FIXTURES = {
    'cmp': (
        'v_cmp_lg_f32_e32 vcc_lo, v3, v4\nv_cndmask_b32_e64 v6, 0, 1.0, vcc_lo\n'
        'v_cmp_nlg_f32_e64 s6, -v3, |v4|\nv_cndmask_b32_e64 v7, 0, 1.0, s6\n'
        'v_cmp_lg_f32_e64 s8, -|v3|, -v4\nv_cndmask_b32_e64 v8, 0, 1.0, s8\n'
        'v_cmp_nlg_f32_e32 vcc_lo, v3, v4\nv_cndmask_b32_e64 v9, 0, 1.0, vcc_lo\n'
        'exp mrt0 v6, v7, v8, v9 done vm\ns_endpgm\n',
        vopc_e32(0x05, ('v', 3), 4) + cndmask_e64(6, VCC_LO)
        + vop3_compare(0x0a, 6, 3, 4, (True, False), (False, True)) + cndmask_e64(7, 6)
        + vop3_compare(0x05, 8, 3, 4, (True, True), (True, False)) + cndmask_e64(8, 8)
        + vopc_e32(0x0a, ('v', 3), 4) + cndmask_e64(9, VCC_LO)
        + struct.pack('<III', 0xf800180f, 0x09080706, 0xbf810000),
        [('lg', PLAIN, PLAIN), ('nlg', NEG, ABS), ('lg', NEG_ABS, NEG), ('nlg', PLAIN, PLAIN)]),
    'cmpx_lg_e32': ('v_cmpx_lg_f32_e32 v3, v4\n', vopc_e32(0x15, ('v', 3), 4), [('lg', PLAIN, PLAIN)]),
    'cmpx_nlg_e32': ('v_cmpx_nlg_f32_e32 v3, v4\n', vopc_e32(0x1a, ('v', 3), 4), [('nlg', PLAIN, PLAIN)]),
    'cmpx_lg_e64': ('v_cmpx_lg_f32_e64 -v3, |v4|\n', vop3_compare(0x15, 0, 3, 4, (True, False), (False, True)),
                    [('lg', NEG, ABS)]),
    'cmpx_nlg_e64': ('v_cmpx_nlg_f32_e64 |v3|, -v4\n', vop3_compare(0x1a, 0, 3, 4, (False, True), (True, False)),
                     [('nlg', ABS, NEG)]),
}


def classify(a, b):
    """The kind of a source pair: 'nan' when either is NaN, else 'equal' or 'unequal'."""
    if is_nan(a) or is_nan(b):
        return 'nan'
    return 'equal' if key(a) == key(b) else 'unequal'


def fixture(name):
    """(source text, words) of one pixel program."""
    head_text, head_words = prologue()
    body_text, body_words, _ = FIXTURES[name]
    if name.startswith('cmpx'):
        body_text, body_words = body_text + EXPORT_PACKED[0], body_words + EXPORT_PACKED[1]
    return head_text + body_text, head_words + body_words


def flushed(bits):
    """A binary32 denormal flushed to zero of the same sign; other words unchanged."""
    return bits & SIGN if bits & 0x7f800000 == 0 else bits


DENORMALS = ('ieee', 'flushed')


def image(name, model=reference, denormals='ieee'):
    """The expected target words. D3D lets a backend flush binary32 denormal sources of a compare to signed zero;
    denormals='flushed' is that image."""
    words = []
    results = FIXTURES[name][2]
    prepare = flushed if denormals == 'flushed' else (lambda bits: bits)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            a, b = VALUES[x], VALUES[y]
            got = [model(cond, prepare(modified(a, *m0)), prepare(modified(b, *m1))) for cond, m0, m1 in results]
            if name == 'cmp':
                words += [ONE if r else 0 for r in got]
            else:
                words += [x | y << 16] * 4 if got[0] else [CLEAR] * 4
    return words


def pixel_hlsl(name, old=False):
    text = G.Pipeline(VS_WORDS, fixture(name)[1], root_blob(), ps_hw_inputs=PACKED).hlsl_ps()
    if old:
        mutated = old_lowering(text)
        assert mutated.count('!=') - text.count('!=') == len(FIXTURES[name][2]), (name, mutated)
        text = mutated
    return text


def check_fixtures():
    for name in FIXTURES:
        text, words = fixture(name)
        decoded = [llvm_text(asm) for _o, _s, _n, asm in G.decode(words)]
        check('%s: fixture decodes to its source (%d instructions)' % (name, len(decoded)),
              decoded == text.splitlines(), str([(a, b) for a, b in zip(decoded, text.splitlines()) if a != b][:3]))
        hlsl = pixel_hlsl(name)
        check('%s: %d lg/nlg compares lowered as less or greater' % (name, len(FIXTURES[name][2])),
              len(LG_TEXT.findall(hlsl)) == len(FIXTURES[name][2]) and ' != F(' not in hlsl)
        # What the image asserts, per result: lg is 1 on finite or infinite unequal sources and 0 on equal ones
        # and on every NaN pair; nlg is the opposite.
        for index, (cond, m0, m1) in enumerate(FIXTURES[name][2]):
            seen = {}
            for b in VALUES:
                for a in VALUES:
                    a2, b2 = modified(a, *m0), modified(b, *m1)
                    seen.setdefault(classify(a2, b2), set()).add(reference(cond, a2, b2))
            want = {'unequal': {1}, 'equal': {0}, 'nan': {0}} if cond == 'lg' else {
                'unequal': {0}, 'equal': {1}, 'nan': {1}}
            counts = {kind: sum(classify(modified(a, *m0), modified(b, *m1)) == kind for a, b in PAIRS)
                      for kind in ('unequal', 'equal', 'nan')}
            check('%s result %d (%s): unequal -> %d, equal -> %d, NaN -> %d; pixels %s' % (
                name, index, cond, *(min(want[k]) for k in ('unequal', 'equal', 'nan')), counts), seen == want,
                str(seen))
        for denormals in DENORMALS:
            good, bad = image(name, denormals=denormals), image(name, old_reference, denormals)
            differing = sum(good[i:i + 4] != bad[i:i + 4] for i in range(0, len(good), 4))
            check('%s (%s denormals): the old lowering changes %d of %d pixels' % (
                name, denormals, differing, WIDTH * HEIGHT), differing == 156)
        ieee, flush = image(name), image(name, denormals='flushed')
        differing = [(i // 4 % WIDTH, i // 4 // WIDTH) for i in range(0, len(ieee), 4)
                     if ieee[i:i + 4] != flush[i:i + 4]]
        check('%s: flushing denormals changes %d pixels, each with a denormal source and no NaN' % (
            name, len(differing)), bool(differing) and all(
            any(VALUES[c] & 0x7f800000 == 0 and VALUES[c] & 0x7fffff for c in (x, y))
            and not is_nan(VALUES[x]) and not is_nan(VALUES[y]) for x, y in differing))
        good = ieee
        if name != 'cmp':
            kept = sum(good[i] != CLEAR for i in range(0, len(good), 4))
            check('%s: reference keeps %d of %d pixels' % (name, kept, WIDTH * HEIGHT), 0 < kept < WIDTH * HEIGHT)
    # The evaluator on each pixel stage's compare lines reads the same decisions as the images.
    for name in FIXTURES:
        for old in (False, True):
            lines = [ln.strip() for ln in pixel_hlsl(name, old).splitlines()
                     if re.match(r'\s*(vcc|exec|s6|s8) = .*F\(v3\)', ln)]
            model = old_reference if old else reference
            results = FIXTURES[name][2]
            ok = len(lines) == len(results)
            for line, (cond, m0, m1) in zip(lines, results):
                expr = Expr(line.split(' = ', 1)[1][:-1])
                ok = ok and all(expr.run({'v3': a, 'v4': b}) == model(cond, modified(a, *m0), modified(b, *m1))
                                for a, b in PAIRS)
            check('%s%s: evaluated compares equal the %s' % (name, ' (old)' if old else '',
                                                              'old model' if old else 'ISA reference'), ok, str(lines))


def all_codes():
    return [compare_code(op, form, mods) for op, form, mods, _ in all_forms()]


def check_binary(llvm_mc):
    codes = all_codes()
    source_text = ''.join(llvm_text(decode_one(code)[1]) + '\n' for code in codes)
    built = assemble(llvm_mc, source_text)
    check('llvm-mc assembles %d decoded f32 compare forms to the same words' % len(codes), built == b''.join(codes))
    for name in FIXTURES:
        text, words = fixture(name)
        check('llvm-mc assembles the %s fixture to the same words' % name, assemble(llvm_mc, text) == words)


def check_decode(library):
    from pixel_aot import disassemble
    codes = all_codes()
    texts = disassemble(b''.join(codes), library)
    ours = [llvm_text(decode_one(code)[1]) for code in codes]
    wrong = [(a, b) for a, b in zip(texts, ours) if a != b]
    check('LLVM disassembles %d f32 compare forms as tools/disasm_rdna2.py does' % len(codes),
          len(texts) == len(codes) and not wrong, str(wrong[:3]))
    for name in FIXTURES:
        text, words = fixture(name)
        check('LLVM disassembles the %s fixture as its source' % name, disassemble(words, library) == text.splitlines())


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
    """Each fixture must match the ISA image with denormals kept or flushed (the same choice for every fixture of
    one compiler and adapter); the old != lowering must match neither and must match its own image."""
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            modes = set()
            for name in FIXTURES:
                files = {d: words_file(directory, '%s_%s' % (name, d), image(name, denormals=d)) for d in DENORMALS}
                ps = compile_hlsl(compiler, kind, pixel_hlsl(name), 'ps_%s_%s' % (name, kind), ps_profile, directory)
                matched, report = None, []
                for denormals in DENORMALS:
                    run = draw(renderer, adapter, vs, ps, files[denormals])
                    report.append(run.stdout + run.stderr)
                    if run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout:
                        matched = denormals
                        break
                print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                                      if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
                modes.add(matched)
                check('%s %s %s: all %d pixels match the ISA reference (%s denormals)' % (
                    kind, adapter, name, WIDTH * HEIGHT, matched), matched is not None, '\n'.join(report))
                bad = compile_hlsl(compiler, kind, pixel_hlsl(name, old=True), 'ps_%s_old_%s' % (name, kind),
                                   ps_profile, directory)
                for denormals in DENORMALS:
                    run = draw(renderer, adapter, vs, bad, files[denormals])
                    lines = [ln for ln in run.stderr.splitlines() if 'pixel' in ln]
                    check('%s %s %s old != lowering fails against the %s image: %s' % (
                        kind, adapter, name, denormals, ' / '.join(lines)),
                        run.returncode != 0 and lines and lines[0].startswith('pixel mismatch'),
                        run.stdout + run.stderr)
                own = words_file(directory, name + '_old', image(name, old_reference, matched or 'ieee'))
                run = draw(renderer, adapter, vs, bad, own)
                check('%s %s %s old != lowering matches its own image' % (kind, adapter, name),
                      run.returncode == 0, run.stdout + run.stderr)
                # Recorded, not checked: the same less-or-greater without the bit test of the source words.
                loose = compile_hlsl(compiler, kind, unguarded(pixel_hlsl(name)), 'ps_%s_loose_%s' % (name, kind),
                                     ps_profile, directory)
                run = draw(renderer, adapter, vs, loose, files[matched or 'ieee'])
                print('  recorded: %s %s %s without the NaN bit test: %s' % (kind, adapter, name, ' / '.join(
                    ln for ln in (run.stdout + run.stderr).splitlines() if 'pixel' in ln)))
            check('%s %s: one denormal behaviour across the fixtures: %s' % (kind, adapter, sorted(map(str, modes))),
                  len(modes) == 1 and None not in modes)

def main():
    if len(sys.argv) == 1:
        controls()
        check_fixtures()
    else:
        mode, *args = sys.argv[1:]
        if mode == 'binary':
            check_binary(*args)
        elif mode == 'decode':
            check_decode(*args)
        elif mode == 'draw':
            check_fixtures()
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
