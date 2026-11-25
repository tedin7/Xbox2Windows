"""f32 VOPC nlt, nle, ngt, nge, o and u in tools/startup/gfx2hlsl.py under FXC and DXC without -Gis (#1936).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     assemble the decoded text of the pixel fixtures with llvm-mc and compare with
                                       the words built here
  decode <libLLVM>                     disassemble the same words with LLVM's C API and compare with
                                       tools/disasm_rdna2.py
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the pixel fixtures with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw 16 x 16 pixels each and compare every pixel with the ISA reference
                                       (needs tests/offscreen_check.cpp built as offscreen_check)

The startup cache compiles gfx2hlsl pairs with FXC and no flags. Neither FXC nor DXC keeps float compares
IEEE-strict unless asked (/Gis, -Gis), and both rewrite the text these conditions had before #1936:

  FXC   !(a < b) becomes a >= b (and so on for <=, >, >=), false when a source is NaN: nlt, nle, ngt and nge
        gave 0 on every NaN pair where the ISA gives 1.
  DXC   a != a and a == a fold to false and true: u gave 0 and o gave 1 on every pair.

The lowering is now `!(ORDERED && a OP b)` for the n-forms and ORDERED and !ORDERED for o and u, where ORDERED
tests both source words for NaN on their bits, as lg and nlg do since #1920. Any rewrite of the float relation is
then behind an integer NaN test.

Controls: every f32 condition of the compare table, in every form (v_cmp and v_cmpx; e32, and e64 under every
neg/abs combination), is translated and its HLSL is evaluated over all 256 pairs of the 16 source words of
tests/startup_gfx2hlsl_f32_lg_check.py under three models of the compiler: IEEE binary32 compares; the FXC
rewrite, a negation pushed through && and || into the opposite relation; and the DXC folds, a relation between
two identical float operands folded to its value on equal operands and a < b || a > b read as a != b (#1920).
Each must equal the ISA reference. The text before #1936, rebuilt from the same operands, must fail under the
model of the compiler that broke it and hold under IEEE compares.

The draws: as in tests/startup_gfx2hlsl_f32_lg_check.py, a source-built pixel program builds one of the 16 words
from x and one from y, so the 16 x 16 pixels cover every pair. Fixture `cmp_<cond>` writes four results of one
condition as 1.0 or 0.0: v_cmp_<cond>_f32_e32, and the e64 form with (-a, |b|), (-|a|, -b) and (|a|, b).
Fixture `cmpx_<cond>` keeps a pixel when v_cmpx_<cond>_f32_e64 (-a, |b|) holds. All 13 lowered conditions
other than lg and nlg are drawn; every fixture must match the ISA image with denormals kept or flushed, the same
choice across the fixtures of one compiler and adapter whose image depends on it (f, o and u do not). The text
before #1936 must fail its draw against both images on the compiler that broke it and must match the image of that
compiler's rewrite. As a diagnostic, not checked, the old and the new text of the six are also drawn with /Gis and
-Gis, which the startup cache does not use.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_gfx2hlsl_f32_lg_check as L  # noqa: E402
from startup_gfx2hlsl_f32_lg_check import (  # noqa: E402
    COMPARES, F32_OPS, PAIRS, VALUES, CLEAR, ONE, WIDTH, HEIGHT, Expr, G, assemble, compile_hlsl, flushed,
    is_nan, modified, reference)
from startup_image_3d_check import CompileQueue  # noqa: E402

NAN_FORMS = ('nlt', 'nle', 'ngt', 'nge', 'o', 'u')
BROKEN_BY = {'nlt': 'fxc', 'nle': 'fxc', 'ngt': 'fxc', 'nge': 'fxc', 'o': 'dxc', 'u': 'dxc'}
OPPOSITE = {'<': '>=', '>=': '<', '>': '<=', '<=': '>', '==': '!=', '!=': '=='}
# The text before #1936, from the float operand texts X and Y.
OLD = {'nlt': '!({0} < {1})', 'nle': '!({0} <= {1})', 'ngt': '!({0} > {1})', 'nge': '!({0} >= {1})',
       'o': '({0} == {0} && {1} == {1})', 'u': '({0} != {0} || {1} != {1})'}
ORDERED = '(v1 & 0x7fffffffu) <= 0x7f800000u && (v2 & 0x7fffffffu) <= 0x7f800000u'
OPCODE = {cond: op for op, (cmpx, cond, typ) in COMPARES.items() if typ == 'f32' and not cmpx}
OPCODE_X = {cond: op for op, (cmpx, cond, typ) in COMPARES.items() if typ == 'f32' and cmpx}

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


# ---- the compiler models ---------------------------------------------------------------------------------------
class Model(Expr):
    """Expr under a compiler model: 'ieee', 'fxc' (negation pushed into the opposite relation) or 'dxc'
    (reflexive float relations folded, less-or-greater read as unordered not-equal)."""

    def __init__(self, text, model):
        super().__init__(text)
        self.model = model

    def floats(self, env, node):
        return node[0] in OPPOSITE and self.eval(env, node[1])[0] == 'f'

    def negate(self, env, node):
        if node[0] == 'u!':
            return node[1]
        if node[0] in ('&&', '||'):
            return ('||' if node[0] == '&&' else '&&', self.negate(env, node[1]), self.negate(env, node[2]))
        if node[0] in OPPOSITE:
            return (OPPOSITE[node[0]], node[1], node[2])
        return ('u!', node)

    def eval(self, env, node):
        if self.model == 'fxc' and node[0] == 'u!':
            inner = node[1]
            if inner[0] in ('&&', '||') or self.floats(env, inner):
                return self.eval(env, self.negate(env, inner))
        if self.model == 'dxc':
            if self.floats(env, node) and node[1] == node[2]:
                return 'b', int(node[0] in ('==', '<=', '>='))
            if (node[0] == '||' and node[1][0] == '<' and node[2][0] == '>' and node[1][1:] == node[2][1:]
                    and self.floats(env, node[1])):
                return self.eval(env, ('!=',) + node[1][1:])
        return super().eval(env, node)


MODELS = ('ieee', 'fxc', 'dxc')


def operands(form, mods, cmpx):
    """The float operand texts X and Y of this form, read from the lt compare of the same form."""
    op = OPCODE_X['lt'] if cmpx else OPCODE['lt']
    _asm, stage = L.translate(L.compare_code(op, form, mods))
    text = stage.lines[0].split(' = ', 1)[1]
    assert text.startswith('((') and text.endswith(') ? 1u : 0u);'), text
    x, y = text[2:-len(') ? 1u : 0u);')].split(' < ')
    return x, y


def first_mismatch(expr, cond, mods):
    (neg0, abs0), (neg1, abs1) = mods
    for a, b in PAIRS:
        want = reference(cond, modified(a, neg0, abs0), modified(b, neg1, abs1))
        got = expr.run({'v1': a, 'v2': b})
        if got != want:
            return '%08x %08x gives %d, the ISA %d' % (a, b, got, want)
    return None


def controls():
    # 1. The emitted text.
    want = {'nlt': '!(%s && F(v1) < F(v2))', 'nle': '!(%s && F(v1) <= F(v2))', 'ngt': '!(%s && F(v1) > F(v2))',
            'nge': '!(%s && F(v1) >= F(v2))', 'o': '(%s)', 'u': '!(%s)'}
    wrong = []
    for cond in NAN_FORMS:
        _asm, stage = L.translate(L.compare_code(OPCODE[cond], 'e32'))
        line = 'vcc = ((%s) ? 1u : 0u);' % (want[cond] % ORDERED)
        if stage.lines != [line]:
            wrong.append((cond, stage.lines, line))
    check('v_cmp_{nlt,nle,ngt,nge,o,u}_f32_e32 test both source words on their bits', not wrong, str(wrong))
    _asm, stage = L.translate(L.compare_code(OPCODE_X['nge'], 'e64', ((True, True), (False, True))))
    check('v_cmpx_nge_f32_e64 -|a|, |b| tests the words under the modifiers: %s' % stage.lines, stage.lines == [
        'exec = ((!(%s && (-abs(F(v1))) >= abs(F(v2)))) ? 1u : 0u);' % ORDERED])

    # 2. The models: on the text before #1936 they reproduce what was drawn.
    x, y = 'F(v1)', 'F(v2)'
    seen = {}
    for cond in NAN_FORMS:
        for model in MODELS:
            expr = Model(OLD[cond].format(x, y), model)
            seen[(cond, model)] = first_mismatch(expr, cond, ((False, False), (False, False))) is None
    check('models: the old text holds under IEEE compares and fails exactly under the compiler that broke it',
          all(seen[(c, m)] == (m == 'ieee' or m != BROKEN_BY[c]) for c in NAN_FORMS for m in MODELS),
          str(sorted(k for k, v in seen.items() if v != (k[1] == 'ieee' or k[1] != BROKEN_BY[k[0]]))))
    lg = Model('(F(v1) < F(v2) || F(v1) > F(v2))', 'dxc')
    check('models: dxc reads the #1920 less-or-greater as unordered not-equal',
          lg.run({'v1': 0x7fc00000, 'v2': 0}) == 1 and Model('(F(v1) < F(v2) || F(v1) > F(v2))', 'ieee').run(
              {'v1': 0x7fc00000, 'v2': 0}) == 0)

    # 3. Every form of every condition under every model; the old text of the six fails under its compiler.
    forms, wrong, caught, missed = 0, [], 0, []
    for op, form, mods, _destination in L.all_forms():
        cmpx, cond, _ = COMPARES[op]
        asm, stage = L.translate(L.compare_code(op, form, mods))
        text = stage.lines[0].split(' = ', 1)[1][:-1]
        forms += 1
        for model in MODELS:
            miss = first_mismatch(Model(text, model), cond, mods)
            if miss:
                wrong.append((asm, model, miss))
        if cond in NAN_FORMS:
            old = '((%s) ? 1u : 0u)' % OLD[cond].format(*operands(form, mods, cmpx))
            if (first_mismatch(Model(old, 'ieee'), cond, mods) is None
                    and first_mismatch(Model(old, BROKEN_BY[cond]), cond, mods)):
                caught += 1
            else:
                missed.append(asm)
    check('%d f32 forms x %d pairs equal the ISA reference under IEEE, FXC and DXC compares' % (forms, len(PAIRS)),
          not wrong and forms == len(F32_OPS) * 17, str(wrong[:3]))
    check('the text before #1936 fails %d of %d nlt/nle/ngt/nge/o/u forms under its compiler and holds under IEEE' % (
        caught, caught + len(missed)), caught == 6 * 2 * 17 and not missed, str(missed[:3]))

    # 4. EXEC already reduced: the result is masked by the lane's EXEC bit.
    wrong = []
    for cond in NAN_FORMS:
        for op in (OPCODE[cond], OPCODE_X[cond]):
            mods = ((True, False), (False, True))
            asm, stage = L.translate(L.compare_code(op, 'e64', mods), exec_known=False)
            text = stage.lines[0].split(' = ', 1)[1][:-1]
            for model in MODELS:
                expr = Model(text, model)
                for a, b in PAIRS:
                    for lane in (0, 1):
                        want = lane & reference(cond, modified(a, True, False), modified(b, False, True))
                        if expr.run({'v1': a, 'v2': b, 'exec': lane}) != want:
                            wrong.append((asm, model, hex(a), hex(b), lane))
    check('inactive lane: the six results are masked by EXEC under every model', not wrong, str(wrong[:3]))


# ---- the pixel fixtures -----------------------------------------------------------------------------------------
DRAWN = [c for c in ('f', 'lt', 'eq', 'le', 'gt', 'ge', 'neq') + NAN_FORMS]
CMP_MODS = [((False, False), (False, False)), ((True, False), (False, True)), ((True, True), (True, False)),
            ((False, True), (False, False))]
SDST = 6


def fixtures():
    """name -> (body words, [(condition, src0 (neg, abs), src1 (neg, abs))] per exported word, or one for v_cmpx)."""
    out = {}
    for cond in DRAWN:
        words, results = b'', []
        for index, mods in enumerate(CMP_MODS):
            (n0, a0), (n1, a1) = mods
            if index == 0:
                words += L.vopc_e32(OPCODE[cond], ('v', 3), 4) + L.cndmask_e64(6, L.VCC_LO)
            else:
                words += L.vop3_compare(OPCODE[cond], SDST, 3, 4, (n0, n1), (a0, a1)) + L.cndmask_e64(6 + index, SDST)
            results.append((cond, mods[0], mods[1]))
        words += struct.pack('<III', 0xf800180f, 0x09080706, 0xbf810000)
        out['cmp_' + cond] = (words, results)
    for cond in NAN_FORMS:
        mods = CMP_MODS[1]
        words = L.vop3_compare(OPCODE_X[cond], 0, 3, 4, (True, False), (False, True)) + L.EXPORT_PACKED[1]
        out['cmpx_' + cond] = (words, [(cond, mods[0], mods[1])])
    return out


FIXTURES = fixtures()


def program(name):
    return L.prologue()[1] + FIXTURES[name][0]


def blind(cond, a, b):
    """What the text before #1936 computes under the compiler that broke it."""
    if cond in NAN_FORMS and (is_nan(a) or is_nan(b)):
        return int(cond == 'o')
    return reference(cond, a, b)


def image(name, model=reference, denormals='ieee'):
    prepare = flushed if denormals == 'flushed' else (lambda bits: bits)
    words = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            a, b = VALUES[x], VALUES[y]
            got = [model(cond, prepare(modified(a, *m0)), prepare(modified(b, *m1)))
                   for cond, m0, m1 in FIXTURES[name][1]]
            if name.startswith('cmp_'):
                words += [ONE if r else 0 for r in got]
            else:
                words += [x | y << 16] * 4 if got[0] else [CLEAR] * 4
    return words


def pixel_hlsl(name, old=False):
    text = G.Pipeline(L.VS_WORDS, program(name), L.root_blob(), ps_hw_inputs=L.PACKED).hlsl_ps()
    if old:
        text = old_text(text)
    return text


def old_text(text):
    """The same stage with the six conditions lowered as before #1936."""
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if ' = ((' in stripped and '0x7fffffffu' in stripped and stripped.endswith(') ? 1u : 0u);'):
            target, expr = stripped.split(' = ', 1)
            body = expr[2:-len(') ? 1u : 0u);')]
            body = body.replace('((v3 & 0x7fffffffu) <= 0x7f800000u && (v4 & 0x7fffffffu) <= 0x7f800000u)', 'ORD')
            body = body.replace('(v3 & 0x7fffffffu) <= 0x7f800000u && (v4 & 0x7fffffffu) <= 0x7f800000u', 'ORD')
            for cond in NAN_FORMS:
                rel = {'nlt': '<', 'nle': '<=', 'ngt': '>', 'nge': '>='}.get(cond)
                if rel and body.startswith('!(ORD && ') and (' %s ' % rel) in body and body.count(' %s ' % rel) == 1:
                    xy = body[len('!(ORD && '):-1].split(' %s ' % rel)
                    if len(xy) == 2 and not any(r in body for r in (' < ', ' <= ', ' > ', ' >= ') if r != ' %s ' % rel):
                        body = OLD[cond].format(*xy)
                        break
            else:
                if body in ('ORD', '!ORD'):
                    body = OLD['o' if body == 'ORD' else 'u'].format('F(v3)', 'F(v4)')
            line = line[:len(line) - len(stripped)] + '%s = ((%s) ? 1u : 0u);' % (target, body)
        out.append(line)
    return '\n'.join(out) + '\n'


def check_fixtures():
    for name, (_words, results) in FIXTURES.items():
        conds = {c for c, _m0, _m1 in results}
        text, old = pixel_hlsl(name), pixel_hlsl(name, old=True)
        guarded = text.count('(v3 & 0x7fffffffu)')
        if conds & set(NAN_FORMS):
            check('%s: %d compares test the source words; the old text has none' % (name, guarded),
                  guarded == len(results) and old.count('(v3 & 0x7fffffffu)') == 0 and old != text)
        else:
            check('%s: no bit test, the old text is the same' % name, guarded == 0 and old == text)
        for denormals in L.DENORMALS:
            good, bad = image(name, denormals=denormals), image(name, blind, denormals)
            differing = sum(good[i:i + 4] != bad[i:i + 4] for i in range(0, len(good), 4))
            want = 156 if conds & set(NAN_FORMS) else 0
            check('%s (%s denormals): the compiler-rewritten old text changes %d of %d pixels' % (
                name, denormals, differing, WIDTH * HEIGHT), differing == want)
    # The evaluator on the compare lines of each stage reads the same decisions as the images, under every model.
    for name, (_words, results) in FIXTURES.items():
        for old in (False, True):
            lines = [ln.strip() for ln in pixel_hlsl(name, old).splitlines()
                     if ln.strip().startswith(('vcc = ', 's6 = ', 'exec = ')) and ('F(v3)' in ln or '(v3 & ' in ln)]
            ok = len(lines) == len(results) or (results[0][0] == 'f' and not lines)
            for line, (cond, m0, m1) in zip(lines, results):
                for model in MODELS:
                    if old and model != 'ieee':
                        continue
                    expr = Model(line.split(' = ', 1)[1][:-1], model)
                    ok = ok and all(expr.run({'v3': a, 'v4': b}) == reference(cond, modified(a, *m0), modified(b, *m1))
                                    for a, b in PAIRS)
            check('%s%s: evaluated compares equal the ISA reference' % (name, ' (old, IEEE)' if old else ''), ok,
                  str(lines))


def fixture_text(words):
    return [L.llvm_text(asm) for _o, _s, _n, asm in G.decode(words)]


def check_binary(llvm_mc):
    for name in FIXTURES:
        words = program(name)
        text = fixture_text(words)
        check('llvm-mc assembles the decoded %s fixture (%d instructions) to the same words' % (name, len(text)),
              assemble(llvm_mc, '\n'.join(text) + '\n') == words)


def check_decode(library):
    from pixel_aot import disassemble
    for name in FIXTURES:
        words = program(name)
        check('LLVM disassembles the %s fixture as tools/disasm_rdna2.py does' % name,
              disassemble(words, library) == fixture_text(words))


def draw(renderer, adapter, vs, ps, expected):
    run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT),
                          str(expected)], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(1 if failures else 77)
    return run


STRICT = {'fxc': '/Gis', 'dxc': '-Gis'}


def compile_strict(compiler, kind, source, out, profile, directory):
    """compile_hlsl with IEEE-strict compares, for the recorded comparison only."""
    path = directory / ('%s.hlsl' % out)
    path.write_text(source, encoding='ascii')
    target = directory / ('%s.cso' % out)
    if kind == 'fxc':
        args = [compiler, '/nologo', STRICT[kind], '/T', profile, '/E', 'main', '/Fo', str(target), str(path)]
    else:
        args = [compiler, STRICT[kind], '-T', profile, '-E', 'main', '-Fo', str(target), str(path)]
    subprocess.run(args, check=True, capture_output=True, text=True)
    return target


def prefetch(queue, compilers, directory):
    """Start every compile check_draw will ask for, in the order it asks. A compile missed here is made inline."""
    for kind, compiler, vs_profile, ps_profile in compilers:
        queue.submit(compile_hlsl, compiler, kind, L.FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
        for name, (_words, results) in FIXTURES.items():
            queue.submit(compile_hlsl, compiler, kind, pixel_hlsl(name), 'ps_%s_%s' % (name, kind), ps_profile,
                         directory)
            if not {c for c, _m0, _m1 in results} & set(NAN_FORMS):
                continue
            queue.submit(compile_hlsl, compiler, kind, pixel_hlsl(name, old=True), 'ps_%s_old_%s' % (name, kind),
                         ps_profile, directory)
            for label, old in (('old', True), ('new', False)):
                queue.submit(compile_strict, compiler, kind, pixel_hlsl(name, old=old),
                             'ps_%s_%s_strict_%s' % (name, label, kind), ps_profile, directory)


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    if not fxc:
        print('  no FXC given: the nlt/nle/ngt/nge negative controls are not drawn')
    with tempfile.TemporaryDirectory() as tmp, CompileQueue() as queue:
        directory = Path(tmp)
        prefetch(queue, compilers, directory)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = queue.call(compile_hlsl, compiler, kind, L.FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            modes, header = set(), None
            for name, (_words, results) in FIXTURES.items():
                conds = {c for c, _m0, _m1 in results}
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
                # f, o and u do not read denormals: both images are the same and say nothing about the mode.
                sensitive = image(name) != image(name, denormals='flushed')
                if sensitive or matched is None:
                    modes.add(matched)
                check('%s %s %s: all %d pixels match the ISA reference (%s denormals)' % (
                    kind, adapter, name, WIDTH * HEIGHT, matched if sensitive else 'either'), matched is not None,
                    '\n'.join(report))
                if not conds & set(NAN_FORMS):
                    continue
                bad = queue.call(compile_hlsl, compiler, kind, pixel_hlsl(name, old=True), 'ps_%s_old_%s' % (name, kind),
                                 ps_profile, directory)
                if BROKEN_BY[min(conds)] == kind:
                    for denormals in L.DENORMALS:
                        run = draw(renderer, adapter, vs, bad, files[denormals])
                        lines = [ln for ln in run.stderr.splitlines() if 'pixel' in ln]
                        check('%s %s %s old text fails against the %s image: %s' % (
                            kind, adapter, name, denormals, ' / '.join(lines)),
                            run.returncode != 0 and lines and lines[0].startswith('pixel mismatch'),
                            run.stdout + run.stderr)
                    own = L.words_file(directory, name + '_blind', image(name, blind, matched or 'ieee'))
                    run = draw(renderer, adapter, vs, bad, own)
                    check('%s %s %s old text matches the image of the %s rewrite' % (kind, adapter, name, kind),
                          run.returncode == 0, run.stdout + run.stderr)
                else:
                    run = draw(renderer, adapter, vs, bad, files[matched or 'ieee'])
                    print('  recorded: %s %s %s old text: %s' % (kind, adapter, name, ' / '.join(
                        ln for ln in (run.stdout + run.stderr).splitlines() if 'pixel' in ln)))
                # Diagnostic, not checked: the old and the new text with IEEE-strict compares (/Gis, -Gis), which
                # the startup cache does not use.
                for label, old in (('old', True), ('new', False)):
                    strict = queue.call(compile_strict, compiler, kind, pixel_hlsl(name, old=old),
                                        'ps_%s_%s_strict_%s' % (name, label, kind), ps_profile, directory)
                    outcome = 'no image matched'
                    for denormals in L.DENORMALS:
                        run = draw(renderer, adapter, vs, strict, files[denormals])
                        if run.returncode == 0:
                            outcome = 'all %d pixels matched (%s denormals)' % (
                                WIDTH * HEIGHT, denormals if sensitive else 'either')
                            break
                    print('  recorded: %s %s %s %s text with %s: %s' % (
                        kind, adapter, name, label, STRICT[kind], outcome))
            print('  ' + str(header))
            check('%s %s: one denormal behaviour across the denormal-sensitive fixtures: %s' % (
                kind, adapter, sorted(map(str, modes))), len(modes) == 1 and None not in modes)


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
