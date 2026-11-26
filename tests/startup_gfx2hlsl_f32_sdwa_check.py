"""f32 VOPC SDWA compares in tools/startup/gfx2hlsl.py: v_cmp_*_f32_sdwa and v_cmpx_*_f32_sdwa (#1906).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     assemble the decoded text of every f32 SDWA form built here and of the pixel
                                       fixtures with llvm-mc and compare with the words built here
  decode <libLLVM>                     disassemble the same words with LLVM's C API and compare with
                                       tools/disasm_rdna2.py; LLVM must refuse the sext bit on a float source
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the pixel fixtures with FXC ps_5_1 when given and with DXC ps_6_0,
                                       all without flags, draw 16 x 16 pixels each and compare every pixel with the
                                       ISA reference (needs tests/offscreen_check.cpp built as offscreen_check)

What is lowered. DWORD selections read the register as binary32 with neg/abs as in the e64 form, so a
src0_sel:DWORD src1_sel:DWORD compare emits the e64 text of the same modifiers, NaN bit test included. Byte and word
selections are refused because the selected field is always +0 or a denormal and its result depends on the
binary32 denormal mode, which gfx2hlsl does not read (tools/pixel_aot.py refuses the same forms). sext() is
integer-only (llvm/docs/AMDGPUModifierSyntax.rst) and refused; LLVM 15 does not decode the bit on a float source.
LLVM's SIPeepholeSDWA models the sub-dword case as zero-extension read as binary32 bits; the RDNA 2 ISA guide gives
only the SDWAB fields (section 13.3.8) and says VOPC SDWA has no OMOD, CLMP, DST_U or DST_SEL (section 6.8).

Controls: every f32 SDWA form is built from the VOP_SDWA9Be fields and decoded: the 15 conditions of the compare
table; v_cmp into vcc_lo and into an SGPR, v_cmpx into EXEC; every select on each source; every neg/abs combination
on each source. Each DWORD/DWORD form must emit the e64 text and its HLSL is evaluated over the 256 pairs of the 16
source words of tests/startup_gfx2hlsl_f32_lg_check.py under the three compiler models of
tests/startup_gfx2hlsl_f32_nan_check.py (IEEE; FXC pushing a negation into the opposite relation; DXC folding
reflexive relations and reading less-or-greater as not-equal) against the ISA reference on bit patterns. Every form
with a BYTE or WORD selection on either source must be refused with the denormal-mode message. Negative control: the
text before #1930/#1949 on the same operands must fail under the model of the compiler that broke it.

The draws build one source word from the pixel's x and one from its y, as the lg check does. Fixture cmp_<cond>
writes four v_cmp_<cond>_f32_sdwa DWORD/DWORD results as 1.0 or 0.0 under (a, b), (-a, |b|), (-|a|, -b) and
(|a|, b), into vcc_lo and SGPRs; fixture cmpx_<cond> keeps a pixel when v_cmpx_<cond>_f32_sdwa -v3, |v4| holds.
Each must match the image with denormals kept or the one with them flushed, the same choice for every fixture whose
image depends on it, and the report names it. The text before #1930/#1949 must fail against both images on the
compiler that broke it (lg/nlg on both) and match that compiler's model. The exception is o/u under DXC, which folds
the reflexive compares only for some of the modified operand shapes, so that match is recorded. The old text on the
other compiler and strict compares (/Gis, -Gis) are recorded too.
"""
import functools
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_gfx2hlsl_f32_lg_check as L  # noqa: E402
import gpu_case_runner  # noqa: E402
import startup_gfx2hlsl_f32_nan_check as N  # noqa: E402
import startup_gfx2hlsl_vopc_sdwa_check as S  # noqa: E402
from startup_image_3d_check import CompileQueue  # noqa: E402
from startup_gfx2hlsl_f32_lg_check import (  # noqa: E402
    CLEAR, HEIGHT, ONE, PAIRS, VALUES, WIDTH, G, as_float, assemble, compile_hlsl, flushed, modified,
    old_reference, reference)

SELS = S.SELS
SUB = SELS[:-1]
MODIFIERS = L.MODIFIERS
PLAIN, NEG, ABS, NEG_ABS = (False, False), (True, False), (False, True), (True, True)
SGPR_DST = 6
F32_OPS = list(L.F32_OPS)
CONDS = sorted({S.COMPARES[op][1] for op in F32_OPS})
OPCODE = {cond: op for op, (cmpx, cond, typ) in S.COMPARES.items() if typ == 'f32' and not cmpx}
OPCODE_X = {cond: op for op, (cmpx, cond, typ) in S.COMPARES.items() if typ == 'f32' and cmpx}
# Conditions whose text tests the source words for NaN ({2} in the table), and the compiler model under which the
# text before #1930 (lg, nlg) and #1949 (the others) gives a wrong result.
GUARDED = ('lg', 'nlg', 'nlt', 'nle', 'ngt', 'nge', 'o', 'u')
BROKEN_BY = dict(N.BROKEN_BY, lg='ieee', nlg='ieee')
OLD = dict(N.OLD, lg='({0} != {1})', nlg='!({0} != {1})')
MODELS = N.MODELS
DENORMAL_MODE = '%s: f32 SDWA byte/word selection depends on the binary32 denormal mode'

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


# ---- the ISA reference, on bit patterns -----------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def ref(cond, x, y):
    return reference(cond, x, y)


def blind(cond, x, y):
    """What the text before #1930/#1949 gives under the compiler that broke it."""
    if cond in ('lg', 'nlg'):
        return old_reference(cond, x, y)
    return N.blind(cond, x, y)


def want(cond, mods0, mods1, how='reference'):
    model = blind if how == 'old' else ref
    return [model(cond, modified(a, *mods0), modified(b, *mods1)) for a, b in PAIRS]


# ---- a vectorised evaluator for the emitted HLSL, under the compiler models of the nan check -------------------
TOKEN = re.compile(r'\s*(0x[0-9a-fA-F]+u?|\d+\.\d+|\d+u?|[A-Za-z_]\w*|<=|>=|==|!=|\|\||&&|[-&!<>?:(),])')
LEVELS = [['||'], ['&&'], ['&'], ['==', '!='], ['<', '>', '<=', '>=']]
OPPOSITE = N.OPPOSITE


def parse(text):
    """The expression tree of L.Expr, as hashable tuples."""
    tokens = TOKEN.findall(text)
    assert ''.join(tokens) == re.sub(r'\s+', '', text), text
    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def take():
        pos[0] += 1
        return tokens[pos[0] - 1]

    def ternary():
        cond = binary(0)
        if peek() != '?':
            return cond
        take()
        yes = ternary()
        assert take() == ':'
        return ('?', cond, yes, ternary())

    def binary(level):
        if level == len(LEVELS):
            return unary()
        node = binary(level + 1)
        while peek() in LEVELS[level]:
            node = (take(), node, binary(level + 1))
        return node

    def unary():
        token = take()
        if token in ('-', '!'):
            return ('u' + token, unary())
        if token == '(':
            node = ternary()
            assert take() == ')'
            return node
        if '.' in token:
            return ('const', 'f', float(token))
        if token[0].isdigit():
            return ('const', 'u', int(token.rstrip('u'), 0) & 0xffffffff)
        if token in ('true', 'false'):
            return ('const', 'b', int(token == 'true'))
        if peek() != '(':
            return ('var', token)
        take()
        args = [ternary()]
        while peek() == ',':
            take()
            args.append(ternary())
        assert take() == ')'
        return ('call', token, tuple(args))

    tree = ternary()
    assert pos[0] == len(tokens), text
    return tree


class Vector:
    """Evaluates trees over every pair at once: values are (kind, list), kind 'u', 'f' or 'b'. The model rules are
    those of startup_gfx2hlsl_f32_nan_check.Model; Python floats compare as IEEE binary32 does on exact values."""

    def __init__(self, env, model):
        self.env, self.model, self.memo = env, model, {}
        self.size = len(next(iter(env.values())))

    def run(self, text):
        kind, values = self.eval(parse(text))
        assert kind in ('u', 'b'), text
        return values

    def floats(self, node):
        return node[0] in OPPOSITE and self.eval(node[1])[0] == 'f'

    def negate(self, node):
        if node[0] == 'u!':
            return node[1]
        if node[0] in ('&&', '||'):
            return ('||' if node[0] == '&&' else '&&', self.negate(node[1]), self.negate(node[2]))
        if node[0] in OPPOSITE:
            return (OPPOSITE[node[0]], node[1], node[2])
        return ('u!', node)

    def eval(self, node):
        if node not in self.memo:
            self.memo[node] = self.compute(node)
        return self.memo[node]

    def compute(self, node):
        if self.model == 'fxc' and node[0] == 'u!':
            inner = node[1]
            if inner[0] in ('&&', '||') or self.floats(inner):
                return self.eval(self.negate(inner))
        if self.model == 'dxc':
            if self.floats(node) and node[1] == node[2]:
                return 'b', [int(node[0] in ('==', '<=', '>='))] * self.size
            if (node[0] == '||' and node[1][0] == '<' and node[2][0] == '>' and node[1][1:] == node[2][1:]
                    and self.floats(node[1])):
                return self.eval(('!=',) + node[1][1:])
        kind = node[0]
        if kind == 'const':
            return node[1], [node[2]] * self.size
        if kind == 'var':
            return 'u', self.env[node[1]]
        if kind == '?':
            cond = self.truth(self.eval(node[1]))
            (k1, yes), (k2, no) = self.eval(node[2]), self.eval(node[3])
            assert k1 == k2
            return k1, [y if c else n for c, y, n in zip(cond, yes, no)]
        if kind == 'call':
            (k, v), = [self.eval(a) for a in node[2]]
            if node[1] in ('F', 'asfloat'):
                assert k == 'u'
                return 'f', [as_float(w) for w in v]
            assert node[1] == 'abs' and k == 'f', node
            return 'f', [abs(f) for f in v]
        if kind == 'u-':
            k, v = self.eval(node[1])
            assert k == 'f'
            return 'f', [-f for f in v]
        if kind == 'u!':
            return 'b', [int(not t) for t in self.truth(self.eval(node[1]))]
        if kind in ('||', '&&'):
            a, b = self.truth(self.eval(node[1])), self.truth(self.eval(node[2]))
            return 'b', [int(p or q) if kind == '||' else int(p and q) for p, q in zip(a, b)]
        (ka, a), (kb, b) = self.eval(node[1]), self.eval(node[2])
        if kind == '&':
            assert ka == kb == 'u', node
            return 'u', [p & q for p, q in zip(a, b)]
        assert ka == kb and ka in ('f', 'u'), node
        compare = {'==': lambda p, q: p == q, '!=': lambda p, q: p != q, '<': lambda p, q: p < q,
                   '>': lambda p, q: p > q, '<=': lambda p, q: p <= q, '>=': lambda p, q: p >= q}[kind]
        return 'b', [int(compare(p, q)) for p, q in zip(a, b)]

    @staticmethod
    def truth(value):
        kind, values = value
        assert kind in ('b', 'u'), value
        return [v != 0 for v in values]


A = [a for a, _b in PAIRS]
B = [b for _a, b in PAIRS]


# ---- forms -----------------------------------------------------------------------------------------------------
def sdwa_code(op, sel0, sel1, mods0, mods1, sdst=None, src=(1, 2)):
    return S.vopc_sdwa(op, S.source('v', src[0], neg=mods0[0], absolute=mods0[1]),
                       S.source('v', src[1], neg=mods1[0], absolute=mods1[1]), sdst, sel0, sel1)


def all_forms():
    """(op, destination, sel0, sel1, mods0, mods1) for every f32 SDWA form checked here."""
    for op in F32_OPS:
        cmpx = S.COMPARES[op][0]
        for destination in (('exec',) if cmpx else ('vcc', 'sgpr')):
            for sel0 in SELS:
                for sel1 in SELS:
                    for mods0 in MODIFIERS:
                        for mods1 in MODIFIERS:
                            yield op, destination, sel0, sel1, mods0, mods1


def form_code(op, destination, sel0, sel1, mods0, mods1):
    return sdwa_code(op, sel0, sel1, mods0, mods1, SGPR_DST if destination == 'sgpr' else None)


def translate(code, exec_known=True):
    """(asm, stage) or (asm, the refusal)."""
    name, asm = S.decode_one(code)
    stage = S.Probe(exec_known)
    try:
        stage.one(name, asm)
    except G.Unsupported as error:
        return asm, str(error)
    return asm, stage


class Refused(Exception):
    """A form the translator refused where the check needs its text."""


def lines_of(stage):
    return ['<refused: %s>' % stage] if isinstance(stage, str) else stage.lines


def body(stage, target):
    if isinstance(stage, str):
        raise Refused(stage)
    line = stage.lines[0]
    assert line.startswith(target + ' = ') and line.endswith(';'), line
    return line[len(target) + 3:-1]


def operand_texts(mods0, mods1, src=('v1', 'v2')):
    """X and Y, the float operand texts of this modifier set, from the DWORD/DWORD lt compare."""
    _asm, stage = translate(sdwa_code(OPCODE['lt'], 'DWORD', 'DWORD', mods0, mods1,
                                      src=(int(src[0][1:]), int(src[1][1:]))))
    text = body(stage, 'vcc')
    assert text.startswith('((') and text.endswith(') ? 1u : 0u)'), text
    x, y = text[2:-len(') ? 1u : 0u)')].split(' < ')
    return x, y


def old_text(cond, mods0, mods1, src=('v1', 'v2')):
    return '((%s) ? 1u : 0u)' % OLD[cond].format(*operand_texts(mods0, mods1, src))


def controls():
    # 1. Why byte and word selections are refused: the field of any word is +0 or a positive denormal.
    fields = {S.field(v, sel, False) for v in VALUES for sel in SUB}
    check('a BYTE or WORD field fits in 16 bits: as binary32 it is +0 or a positive denormal (%d distinct of the 16 '
          'words)' % len(fields), all(w <= 0xffff for w in fields) and 0 in fields and any(fields)
          and all(as_float(w) < 2.0 ** -126 for w in fields))

    # 2. The emitted text: DWORD/DWORD is the e64 form with the same modifiers, the NaN test on the registers.
    _asm, stage = translate(sdwa_code(OPCODE['lg'], 'DWORD', 'DWORD', NEG, ABS, SGPR_DST))
    expected = 's6 = ((((v1 & 0x7fffffffu) <= 0x7f800000u && (v2 & 0x7fffffffu) <= 0x7f800000u && ' \
               '((-F(v1)) < abs(F(v2)) || (-F(v1)) > abs(F(v2))))) ? 1u : 0u);'
    check('v_cmp_lg_f32_sdwa s6, -v1, |v2| DWORD/DWORD: %s' % lines_of(stage), lines_of(stage) == [expected])
    differ = []
    for op in F32_OPS:
        cmpx = S.COMPARES[op][0]
        for mods in ((m, n) for m in MODIFIERS for n in MODIFIERS):
            _asm, sdwa = translate(sdwa_code(op, 'DWORD', 'DWORD', *mods, None if cmpx else SGPR_DST))
            _asm, e64 = translate(L.compare_code(op, 'e64', mods))
            if lines_of(sdwa) != lines_of(e64) or isinstance(sdwa, str) or sdwa.masks != e64.masks:
                differ.append((op, mods, lines_of(sdwa), lines_of(e64)))
    check('DWORD/DWORD SDWA forms emit the e64 text with the same modifiers (%d forms)' % (len(F32_OPS) * 16),
          not differ, str(differ[:2]))

    # 3. Every form: DWORD/DWORD is lowered into its destination and equals the reference under every compiler model;
    # every form with a BYTE or WORD selection on either source is refused for the denormal mode.
    texts, wrong, count, refused, accepted = {}, [], 0, 0, []
    for form in all_forms():
        op, destination, sel0, sel1, mods0, mods1 = form
        cmpx, cond, _ = S.COMPARES[op]
        asm, stage = translate(form_code(*form))
        if (sel0, sel1) != ('DWORD', 'DWORD'):
            if stage == DENORMAL_MODE % asm.split()[0]:
                refused += 1
            else:
                accepted.append((asm, lines_of(stage)))
            continue
        target = {'exec': 'exec', 'vcc': 'vcc', 'sgpr': 's%d' % SGPR_DST}[destination]
        if isinstance(stage, str):
            wrong.append((asm, stage))
            continue
        if (len(stage.lines) != 1 or not stage.lines[0].startswith(target + ' = ')
                or (destination == 'sgpr') != (SGPR_DST in stage.masks) or stage.exec_known == cmpx):
            wrong.append((asm, stage.lines, 'destination'))
            continue
        count += 1
        texts.setdefault((cond, mods0, mods1), set()).add(body(stage, target))
    check('%d BYTE/WORD forms (15 conditions; v_cmp into vcc_lo and an SGPR, v_cmpx; a sub-dword select on src0, '
          'src1 or both; every neg/abs) refused: "%s"' % (refused, DENORMAL_MODE % '<name>'),
          not accepted and refused == 15 * 3 * 48 * 16, str(accepted[:3]))
    check('%d DWORD/DWORD forms translate, each into its destination' % count,
          not wrong and count == 15 * 3 * 16, str(wrong[:3]))
    check('v_cmp into vcc_lo or an SGPR and v_cmpx into EXEC emit one text per form (%d texts)' % len(texts),
          all(len(t) == 1 for t in texts.values()) and len(texts) == 15 * 16)
    texts = {key: next(iter(t)) for key, t in texts.items()}
    evaluators = {model: Vector({'v1': A, 'v2': B}, model) for model in MODELS}
    wrong, evaluations = [], 0
    for key, text in texts.items():
        expected = want(*key)
        for model in MODELS:
            evaluations += len(PAIRS)
            got = evaluators[model].run(text)
            if got != expected:
                i = next(i for i, (g, w) in enumerate(zip(got, expected)) if g != w)
                wrong.append((key, model, '%08x %08x gives %d, the ISA %d' % (A[i], B[i], got[i], expected[i])))
    check('%d forms (%d texts) x %d pairs x 3 models (%d evaluations) equal the ISA reference' % (
        count, len(texts), len(PAIRS), evaluations), not wrong and len(texts) == 15 * 16, str(wrong[:3]))

    # Negative control: the text before #1930/#1949 on the same operands fails under the compiler that broke it,
    # holds under IEEE compares (lg/nlg already fail there), and matches that compiler's model.
    caught, missed = 0, []
    for (cond, mods0, mods1) in texts:
        if cond not in GUARDED:
            continue
        old = old_text(cond, mods0, mods1)
        expected = want(cond, mods0, mods1)
        broken = evaluators[BROKEN_BY[cond]].run(old)
        if (broken != expected and broken == want(cond, mods0, mods1, 'old')
                and (BROKEN_BY[cond] == 'ieee' or evaluators['ieee'].run(old) == expected)):
            caught += 1
        else:
            missed.append((cond, mods0, mods1))
    check('negative control: the text before #1930/#1949 fails %d of %d guarded forms under its compiler '
          '(lg/nlg already under IEEE) and matches that compiler\'s model' % (caught, caught + len(missed)),
          not missed and caught == 8 * 16, str(missed[:3]))

    # 4. EXEC already reduced: every condition, masked by the lane's EXEC bit, under every model.
    wrong = []
    for cond in CONDS:
        for op in (OPCODE[cond], OPCODE_X[cond]):
            for mods0, mods1 in ((NEG, ABS), (NEG_ABS, PLAIN)):
                _asm, stage = translate(sdwa_code(op, 'DWORD', 'DWORD', mods0, mods1), exec_known=False)
                if isinstance(stage, str):
                    wrong.append((cond, op, stage))
                    continue
                text = stage.lines[0].split(' = ', 1)[1][:-1]
                expected = want(cond, mods0, mods1)
                for model in MODELS:
                    for lane in (0, 1):
                        got = Vector({'v1': A, 'v2': B, 'exec': [lane] * len(A)}, model).run(text)
                        if got != [lane & w for w in expected]:
                            wrong.append((cond, op, mods0, model, lane))
    check('inactive lane: every condition masked by EXEC under every model', not wrong, str(wrong[:3]))

    # 5. Scalar and inline-constant sources: DWORD lowered, a sub-dword select refused.
    s3 = [0x7fc12345, 0xbf800000, 0x00010000, 0xffff0001]
    words = [(s, b) for s in s3 for b in VALUES]
    env = {'s3': [s for s, _ in words], 'v2': [b for _, b in words]}
    asm, stage = translate(S.vopc_sdwa(OPCODE['nlt'], S.source('s', 3, neg=True), S.source('v', 2, absolute=True),
                                       None, 'DWORD', 'DWORD'))
    got = Vector(env, 'fxc').run(body(stage, 'vcc'))
    check('SGPR source: %s' % asm, asm == 'v_cmp_nlt_f32_sdwa vcc_lo, -s3, |v2| src0_sel:DWORD src1_sel:DWORD'
          and got == [reference('nlt', s ^ 0x80000000, b & 0x7fffffff) for s, b in words], str(lines_of(stage)))
    asm, stage = translate(S.vopc_sdwa(OPCODE_X['ge'], S.source('s', 242, neg=True), S.source('v', 2), None,
                                       'DWORD', 'DWORD'))
    got = Vector({'v2': VALUES}, 'ieee').run(body(stage, 'exec'))
    check('inline constant neg(1.0): %s' % asm,
          asm == 'v_cmpx_ge_f32_sdwa neg(1.0), v2 src0_sel:DWORD src1_sel:DWORD'
          and got == [reference('ge', 0xbf800000, b) for b in VALUES], str(lines_of(stage)))
    for code in (S.vopc_sdwa(OPCODE['nlt'], S.source('s', 3, neg=True), S.source('v', 2), None, 'WORD_1', 'DWORD'),
                 S.vopc_sdwa(OPCODE_X['ge'], S.source('s', 242, neg=True), S.source('v', 2), None, 'DWORD', 'BYTE_1')):
        asm, stage = translate(code)
        check('refused: %s (%s)' % (asm, stage), stage == DENORMAL_MODE % asm.split()[0])

    # 6. Other refusals, each with its own message.
    def expect(label, name, asm, message):
        try:
            S.Probe().one(name, asm)
            got = None
        except G.Unsupported as error:
            got = str(error)
        check('refused: %s (%s)' % (label, got), got == message)

    base = 'v_cmp_lt_f32_sdwa vcc_lo, v1, v2'
    expect('sext() on a float source', 'v_cmp_lt_f32_sdwa',
           'v_cmp_lt_f32_sdwa vcc_lo, sext(v1), v2 src0_sel:DWORD src1_sel:DWORD',
           'v_cmp_lt_f32_sdwa: sext() on a float source')
    expect('sext() on a float src1 with a word select', 'v_cmpx_nlg_f32_sdwa',
           'v_cmpx_nlg_f32_sdwa v1, -sext(v2) src0_sel:DWORD src1_sel:WORD_1',
           'v_cmpx_nlg_f32_sdwa: sext() on a float source')
    for extra in ('clamp', 'mul:2', 'div:2', 'dst_sel:DWORD', 'dst_unused:UNUSED_PAD'):
        expect('modifier %s' % extra, 'v_cmp_lt_f32_sdwa', '%s src0_sel:DWORD src1_sel:DWORD %s' % (base, extra),
               'v_cmp_lt_f32_sdwa with SDWA modifier %s' % extra)
    expect('missing src1_sel', 'v_cmp_lt_f32_sdwa', base + ' src0_sel:DWORD',
           'v_cmp_lt_f32_sdwa SDWA without src1_sel')
    name, asm = S.decode_one(S.vopc_sdwa(OPCODE['lt'], S.source('v', 1, sext=True), S.source('v', 2), None,
                                         'BYTE_0', 'DWORD'))
    check('the decoder refuses the sext bit on a float source: %s' % asm, name == 'vopc_sdwa'
          and asm == 'v_cmp_lt_f32_e32 <unsupported SDWA>')
    name, asm = S.decode_one(S.vopc_sdwa(0x0f, S.source('v', 1), S.source('v', 2)))
    expect('tru (no condition in the table, as in the non-SDWA form)', name, asm, 'v_cmp_tru_f32_sdwa')
    name, asm = S.decode_one(S.vopc_sdwa(0x88, S.source('v', 1), S.source('v', 2), None, 'WORD_0', 'BYTE_1'))
    expect('class float src0 sub-dword (class compares are unchanged)', name, asm,
           'v_cmp_class_f32_sdwa requires a DWORD float source')


# ---- the pixel fixtures ---------------------------------------------------------------------------------------
# Per cmp_<cond> result: (destination SGPR or None for vcc_lo, mods0, mods1), as the e64 results of the nan check.
CMP_RESULTS = [(None, PLAIN, PLAIN), (6, NEG, ABS), (8, NEG_ABS, NEG), (None, ABS, PLAIN)]
CMPX_RESULT = (NEG, ABS)


def fixtures():
    """name -> (body words, [(cond, mods0, mods1)] per exported word, or one for v_cmpx)."""
    out = {}
    for cond in CONDS:
        words, results = b'', []
        for index, (sdst, mods0, mods1) in enumerate(CMP_RESULTS):
            words += sdwa_code(OPCODE[cond], 'DWORD', 'DWORD', mods0, mods1, sdst, src=(3, 4))
            words += L.cndmask_e64(6 + index, L.VCC_LO if sdst is None else sdst)
            results.append((cond, mods0, mods1))
        out['cmp_' + cond] = (words + struct.pack('<III', 0xf800180f, 0x09080706, 0xbf810000), results)
    for cond in CONDS:
        out['cmpx_' + cond] = (sdwa_code(OPCODE_X[cond], 'DWORD', 'DWORD', *CMPX_RESULT, src=(3, 4))
                               + L.EXPORT_PACKED[1], [(cond, *CMPX_RESULT)])
    return out


FIXTURES = fixtures()


def program(name):
    return L.prologue()[1] + FIXTURES[name][0]


def image(name, how='reference', denormals='ieee'):
    """The expected target words. D3D lets a backend flush binary32 denormal compare sources to signed zero;
    denormals='flushed' is that image."""
    prepare = flushed if denormals == 'flushed' else (lambda bits: bits)
    model = blind if how == 'old' else ref
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            a, b = VALUES[x], VALUES[y]
            got = [model(cond, prepare(modified(a, *m0)), prepare(modified(b, *m1))) for cond, m0, m1 in
                   FIXTURES[name][1]]
            if name.startswith('cmp_'):
                words += [ONE if r else 0 for r in got]
            else:
                words += [x | y << 16] * 4 if got[0] else [CLEAR] * 4
    return words


COMPARE_LINE = re.compile(r'^(\s*)(vcc|exec|s6|s8) = (\(\(.*\) \? 1u : 0u\));$')


def is_compare(m):
    return m and ('F(' in m.group(3) or '0x7fffffffu' in m.group(3))


def pixel_hlsl(name, how='reference'):
    text = G.Pipeline(L.VS_WORDS, program(name), L.root_blob(), ps_hw_inputs=L.PACKED).hlsl_ps()
    if how != 'old':
        return text
    results, out = iter(FIXTURES[name][1]), []
    for line in text.splitlines():
        m = COMPARE_LINE.match(line)
        if is_compare(m):
            cond, mods0, mods1 = next(results)
            if cond in GUARDED:
                line = '%s%s = %s;' % (m.group(1), m.group(2), old_text(cond, mods0, mods1, ('v3', 'v4')))
        out.append(line)
    return '\n'.join(out) + '\n'


def compare_lines(text):
    return [m.group(3) for m in map(COMPARE_LINE.match, text.splitlines()) if is_compare(m)]


def fixture_text(words):
    return [L.llvm_text(asm) for _o, _s, _n, asm in G.decode(words)]


def check_fixtures():
    env = {'v3': A, 'v4': B}
    evaluators = {model: Vector(env, model) for model in MODELS}
    for name, (_words, results) in FIXTURES.items():
        cond = results[0][0]
        lines = compare_lines(pixel_hlsl(name))
        ok = len(lines) == (0 if cond == 'f' else len(results))
        for line, (_cond, m0, m1) in zip(lines, results):
            for model in MODELS:
                ok = ok and evaluators[model].run(line) == want(cond, m0, m1)
        check('%s: %d compares evaluate to the ISA reference under every model' % (name, len(lines)), ok, str(lines))
        if cond not in GUARDED:
            check('%s: no NaN test, the old text is the same' % name, pixel_hlsl(name, 'old') == pixel_hlsl(name))
            continue
        lines = compare_lines(pixel_hlsl(name, 'old'))
        ok = len(lines) == len(results) and all(
            evaluators[BROKEN_BY[cond]].run(line) == want(cond, m0, m1, 'old')
            for line, (_cond, m0, m1) in zip(lines, results))
        check('%s old text: evaluated under %s equals that model' % (name, BROKEN_BY[cond]), ok)
        for denormals in L.DENORMALS:
            differing = sum(p != q for p, q in zip(image(name, denormals=denormals), image(name, 'old', denormals)))
            check('%s (%s denormals): the old text under its compiler changes %d words' % (
                name, denormals, differing), differing > 0)


def all_codes():
    codes = [form_code(*form) for form in all_forms()]
    codes.append(S.vopc_sdwa(OPCODE['nlt'], S.source('s', 3, neg=True), S.source('v', 2, absolute=True), None,
                             'DWORD', 'DWORD'))
    codes.append(S.vopc_sdwa(OPCODE_X['ge'], S.source('s', 242, neg=True), S.source('v', 2), None, 'DWORD', 'DWORD'))
    return codes


def check_binary(llvm_mc):
    codes = all_codes()
    source_text = ''.join(S.decode_one(code)[1] + '\n' for code in codes)
    check('llvm-mc assembles %d decoded f32 SDWA forms (lowered and refused) to the same words' % len(codes),
          assemble(llvm_mc, source_text) == b''.join(codes))
    for name in FIXTURES:
        words = program(name)
        text = fixture_text(words)
        check('llvm-mc assembles the decoded %s fixture (%d instructions) to the same words' % (name, len(text)),
              assemble(llvm_mc, '\n'.join(text) + '\n') == words)


def check_decode(library):
    from pixel_aot import disassemble
    codes = all_codes()
    texts = disassemble(b''.join(codes), library)
    ours = [S.decode_one(code)[1] for code in codes]
    wrong = [(a, b) for a, b in zip(texts, ours) if a != b]
    check('LLVM disassembles %d f32 SDWA forms as tools/disasm_rdna2.py does' % len(codes),
          len(texts) == len(codes) and not wrong, str(wrong[:3]))
    # The sext bit on a float source: tools/disasm_rdna2.py refuses it, and so does LLVM 15.
    for index in (0, 1):
        sources = [S.source('v', 1), S.source('v', 2)]
        sources[index] = S.source('v', index + 1, sext=True)
        code = S.vopc_sdwa(OPCODE['lt'], *sources, None, 'BYTE_0', 'BYTE_0')
        try:
            decoded = disassemble(code, library)
        except Exception as error:  # noqa: BLE001 - pixel_aot reports an undecodable word this way
            decoded = 'refused: %s' % error
        check('LLVM refuses sext on float src%d (%s): %s' % (index, code.hex(), decoded),
              str(decoded).startswith('refused') and S.decode_one(code)[0] == 'vopc_sdwa')
    for name in FIXTURES:
        words = program(name)
        check('LLVM disassembles the %s fixture as tools/disasm_rdna2.py does' % name,
              disassemble(words, library) == fixture_text(words))


def draw(renderer, adapter, vs, ps, expected):
    run = gpu_case_runner.run(renderer, adapter, ['--startup-pixel-words', vs, ps, WIDTH, HEIGHT, expected])
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(1 if failures else 77)
    return run


def pixel_lines(run):
    return ' / '.join(ln for ln in (run.stdout + run.stderr).splitlines() if 'pixel' in ln)


def prefetch(queue, compilers, directory):
    """Start every compile check_draw will ask for, in the order it asks. A compile missed here is made inline."""
    for kind, compiler, vs_profile, ps_profile in compilers:
        queue.submit(compile_hlsl, compiler, kind, L.FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
        for name, (_words, results) in FIXTURES.items():
            queue.submit(compile_hlsl, compiler, kind, pixel_hlsl(name), 'ps_%s_%s' % (name, kind), ps_profile,
                         directory)
            if results[0][0] in GUARDED:
                queue.submit(compile_hlsl, compiler, kind, pixel_hlsl(name, 'old'), 'ps_%s_old_%s' % (name, kind),
                             ps_profile, directory)
            queue.submit(N.compile_strict, compiler, kind, pixel_hlsl(name), 'ps_%s_strict_%s' % (name, kind),
                         ps_profile, directory)


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    if not fxc:
        print('  no FXC given: the nlt/nle/ngt/nge old-text controls are not drawn')
    with tempfile.TemporaryDirectory() as tmp, CompileQueue() as queue:
        directory = Path(tmp)
        prefetch(queue, compilers, directory)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = queue.call(compile_hlsl, compiler, kind, L.FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            modes, header = set(), None
            for name, (_words, results) in FIXTURES.items():
                cond = results[0][0]
                files = {d: L.words_file(directory, '%s_%s' % (name, d), image(name, denormals=d))
                         for d in L.DENORMALS}
                ps = queue.call(compile_hlsl, compiler, kind, pixel_hlsl(name), 'ps_%s_%s' % (name, kind), ps_profile,
                                directory)
                matched, report = None, []
                for denormals in L.DENORMALS:
                    run = draw(renderer, adapter, vs, ps, files[denormals])
                    report.append(run.stdout + run.stderr)
                    if run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout:
                        matched = denormals
                        break
                header = header or ' '.join(ln for ln in run.stdout.splitlines()
                                            if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors=')))
                sensitive = image(name) != image(name, denormals='flushed')
                if sensitive or matched is None:
                    modes.add(matched)
                check('%s %s %s: all %d pixels match the ISA reference (%s denormals)' % (
                    kind, adapter, name, WIDTH * HEIGHT, matched if sensitive else 'either'), matched is not None,
                    '\n'.join(report))
                if cond in GUARDED:
                    bad = queue.call(compile_hlsl, compiler, kind, pixel_hlsl(name, 'old'), 'ps_%s_old_%s' % (name, kind),
                                     ps_profile, directory)
                    if BROKEN_BY[cond] in ('ieee', kind):
                        for denormals in L.DENORMALS:
                            run = draw(renderer, adapter, vs, bad, files[denormals])
                            check('%s %s %s old text fails against the %s image: %s' % (
                                kind, adapter, name, denormals, pixel_lines(run)),
                                run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)
                        own = L.words_file(directory, name + '_old', image(name, 'old', matched or 'ieee'))
                        run = draw(renderer, adapter, vs, bad, own)
                        if cond in ('o', 'u'):
                            # The old o/u text keeps each source's modifiers, and DXC folds x == x only for some of
                            # those operand shapes (the nan check draws it on the plain words), so this is recorded.
                            print('  recorded: %s %s %s old text against the %s model: %s' % (
                                kind, adapter, name, BROKEN_BY[cond], pixel_lines(run) or 'all pixels matched'))
                        else:
                            check('%s %s %s old text matches the image of the %s model' % (
                                kind, adapter, name, BROKEN_BY[cond]), run.returncode == 0, run.stdout + run.stderr)
                    else:
                        run = draw(renderer, adapter, vs, bad, files[matched or 'ieee'])
                        print('  recorded: %s %s %s old text: %s' % (
                            kind, adapter, name, pixel_lines(run) or 'all pixels matched'))
                # Diagnostic, not checked: the text with IEEE-strict compares, which the startup cache does not use.
                strict = queue.call(N.compile_strict, compiler, kind, pixel_hlsl(name), 'ps_%s_strict_%s' % (name, kind),
                                    ps_profile, directory)
                outcome = 'no image matched'
                for denormals in L.DENORMALS:
                    run = draw(renderer, adapter, vs, strict, files[denormals])
                    if run.returncode == 0:
                        outcome = 'all %d pixels matched (%s denormals)' % (
                            WIDTH * HEIGHT, denormals if sensitive else 'either')
                        break
                print('  recorded: %s %s %s with %s: %s' % (kind, adapter, name, N.STRICT[kind], outcome))
            print('  ' + str(header))
            check('%s %s: one denormal behaviour across the denormal-sensitive fixtures: %s' % (
                kind, adapter, sorted(map(str, modes))), len(modes) == 1 and None not in modes)


def guarded(part):
    """Run one part of the check; a refusal where a text is needed is a failure of that part, not a crash."""
    try:
        part()
    except (Refused, G.Unsupported) as error:
        check('%s: every form it needs translates' % part.__name__, False, 'refused: %s' % error)


def main():
    if len(sys.argv) == 1:
        guarded(controls)
        guarded(check_fixtures)
    else:
        mode, *args = sys.argv[1:]
        if mode == 'binary':
            check_binary(*args)
        elif mode == 'decode':
            check_decode(*args)
        elif mode == 'draw':
            guarded(check_fixtures)
            if failures:
                print('FAIL: the fixtures do not translate as checked; not drawn')
                sys.exit(1)
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
