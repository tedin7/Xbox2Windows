"""D3D12 draw check for v_frexp_mant_f32, v_frexp_exp_i32_f32, v_cvt_flr_i32_f32, v_cvt_rpi_i32_f32 and
v_ldexp_f32 in tools/startup/gfx2hlsl.py (Part of #1723).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture bodies and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per input chunk and compare the four result words

The fixture reuses the flat-parameter pixel prologue of startup_gfx2hlsl_bit_alu_check.py: twelve words arrive in
v0..v11 and v12..v15 are exported. Each operation runs on four lanes per pixel (source words v0..v3, the ldexp
exponents v4..v7).

The expected words come from the references of startup_gfx2hlsl_frexp_cvt_check.py (math.frexp, math.floor and
math.ldexp on Python doubles), not from the lowering. Each lowering has a mutation (a changed constant) whose draw
must read other words, so a draw that cannot see the difference fails the check. Denormal inputs and results are
left out: the denormal mode decides them.
"""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_frexp_cvt_check as R  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (POST_SOURCE, POST_WORDS, PRE_SOURCE, PRE_WORDS,  # noqa: E402
                                            vs_source, vs_words)

MASK = 0xffffffff
UNARY = ('v_frexp_mant_f32', 'v_frexp_exp_i32_f32', 'v_cvt_flr_i32_f32', 'v_cvt_rpi_i32_f32')
OPS = UNARY + ('v_ldexp_f32',)
REFERENCE = {'v_frexp_mant_f32': R.ref_frexp_mant, 'v_frexp_exp_i32_f32': R.ref_frexp_exp,
             'v_cvt_flr_i32_f32': R.ref_cvt_flr, 'v_cvt_rpi_i32_f32': R.ref_cvt_rpi}
# mnemonic -> (text the lowering must contain, replacement that changes the result)
MUTATION = {'v_frexp_mant_f32': ('0x3f000000u', '0x3f800000u'),
            'v_frexp_exp_i32_f32': ('- 126u', '- 127u'),
            'v_cvt_flr_i32_f32': ('floor(', 'ceil('),
            'v_cvt_rpi_i32_f32': ('>= 0.5', '>= 0.75'),
            'v_ldexp_f32': ('exp2(float(', '-exp2(float(')}
BODY_SOURCE = {op: ''.join('%s v%d, v%d\n' % (op, 12 + i, i) for i in range(4)) for op in UNARY}
BODY_SOURCE['v_ldexp_f32'] = ''.join('v_ldexp_f32 v%d, v%d, v%d\n' % (12 + i, i, 4 + i) for i in range(4))

# Normal, finite, inf and NaN inputs; the cvt inputs include the saturating ends.
UNARY_INPUTS = [R.bits(x) for x in (1.0, -1.0, 0.5, 0.75, -2.5, 3.0, 0.1, -0.1, 0.4999, 0.5001, 123456.789,
                                    -98765.4, 1.0e20, -1.0e20, 2.0 ** 127, 8388608.5, 2147483520.0,
                                    -2147483648.0, 2.0 ** 31, -2.0 ** 33, 3.4028235e38, 0.0, -0.0, 2.0 ** -126)]
UNARY_INPUTS += [R.INF, R.INF | 0x80000000, R.NAN, 0x7f7fffff,
                0xffc00000, 0x3effffff, 0x4b000001, 0xff800001]
LDEXP_INPUTS = [(R.bits(a), n & MASK) for a, n in (
    (1.0, 1), (1.5, -1), (-1.5, 5), (0.1, 24), (123456.789, -20), (-3.0, 100), (1.0, 127), (1.0, -126),
    (1.0, 200), (-1.0, -200), (0.0, 5), (-0.0, -5), (1.0, 2 ** 31 - 1), (1.0, -2 ** 31), (2.0, 300), (1.5, 0))]
LDEXP_INPUTS += [(R.INF, 3), (R.INF | 0x80000000, -3 & MASK), (R.NAN, 7), (R.bits(3.0), 2)]


def chunks(op):
    """(12 parameter words, 4 expected result words) per pixel."""
    out = []
    if op == 'v_ldexp_f32':
        for i in range(0, len(LDEXP_INPUTS), 4):
            part = LDEXP_INPUTS[i:i + 4]
            out.append(([a for a, _ in part] + [n for _, n in part] + [0] * 4, [R.ref_ldexp(a, n) for a, n in part]))
    else:
        for i in range(0, len(UNARY_INPUTS), 4):
            part = UNARY_INPUTS[i:i + 4]
            out.append((part + [0] * 8, [REFERENCE[op](w) & MASK for w in part]))
    return out


def mutate_text(op, text):
    old, new = MUTATION[op]
    mutated = text.replace(old, new)
    assert mutated != text, op
    return mutated


def sets_for(words):
    """The four (src0, src1, src2) tuples vs_words() turns into parameter words 0..11 in order."""
    return [(words[lane], words[4 + lane], words[8 + lane]) for lane in range(4)]


def ps_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


BODY_WORDS = {
    'v_frexp_mant_f32': bytes.fromhex('0081187e01811a7e02811c7e03811e7e'),
    'v_frexp_exp_i32_f32': bytes.fromhex('007f187e017f1a7e027f1c7e037f1e7e'),
    'v_cvt_flr_i32_f32': bytes.fromhex('001b187e011b1a7e021b1c7e031b1e7e'),
    'v_cvt_rpi_i32_f32': bytes.fromhex('0019187e01191a7e02191c7e03191e7e'),
    'v_ldexp_f32': bytes.fromhex('0c0062d7000902000d0062d7010b02000e0062d7020d02000f0062d7030f0200'),
}


def translate(op, mutate=False):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words(op), root_blob())
    text = pipe.hlsl_ps()
    return pipe, mutate_text(op, text) if mutate else text


def check(label, ok):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        raise AssertionError(label)


def controls():
    for op in OPS:
        check('%s fixture has 12 words per pixel' % op, all(len(w) == 12 and len(e) == 4 for w, e in chunks(op)))
        check('%s inputs fill every lane' % op, sum(len(e) for _, e in chunks(op)) == (
            len(LDEXP_INPUTS) if op == 'v_ldexp_f32' else len(UNARY_INPUTS)))
        check('%s mutation changes the translation' % op, translate(op, True)[1] != translate(op)[1])


def check_binary(llvm_mc):
    for op, source in BODY_SOURCE.items():
        words = assemble(llvm_mc, source)
        if op in BODY_WORDS and BODY_WORDS[op] != words:
            raise AssertionError('%s: assembled %s, recorded %s' % (op, words.hex(), BODY_WORDS[op].hex()))
        print("    '%s': bytes.fromhex('%s')," % (op, words.hex()))
    for name, source, words in (('prologue', PRE_SOURCE, PRE_WORDS), ('export', POST_SOURCE, POST_WORDS)):
        if assemble(llvm_mc, source) != words:
            raise AssertionError('%s fixture differs from llvm-mc' % name)
    for op in OPS:
        for words, _ in chunks(op):
            sets = sets_for(words)
            if assemble(llvm_mc, vs_source(sets)) != vs_words(sets):
                raise AssertionError('vertex fixture for %s differs from llvm-mc' % (sets,))
    print('fixture words match llvm-mc')


def words_text(values):
    return ','.join('%08x' % v for v in values)


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op in OPS:
                good_ps = compile_hlsl(compiler, kind, translate(op)[1], 'ps_%s_%s' % (op, kind), ps_profile, directory)
                bad_ps = compile_hlsl(compiler, kind, translate(op, True)[1], 'mutated_%s_%s' % (op, kind),
                                      ps_profile, directory)
                caught = 0
                for index, (words, expected) in enumerate(chunks(op)):
                    vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(sets_for(words)), ps_words(op),
                                                                 root_blob()).hlsl_vs(),
                                      'vs_%s_%d_%s' % (op, index, kind), vs_profile, directory)
                    want = words_text(expected)
                    run = draw(renderer, adapter, vs, good_ps, want)
                    if run.returncode:
                        raise AssertionError('%s %s pixel %d expected %s\n%s\n%s' % (kind, op, index, want, run.stdout,
                                                                                    run.stderr))
                    print('%s %s pixel %d: %s' % (kind, op, index, want), flush=True)
                    run = draw(renderer, adapter, vs, bad_ps, want)
                    if run.returncode:
                        mismatch = f'startup_flat_param mismatch expected={want} actual='
                        debug_ok = ('debug_layer=enabled' not in run.stdout
                                    or 'debug_errors=0 debug_warnings=0' in run.stdout)
                        if run.returncode != 1 or mismatch not in run.stderr or not debug_ok:
                            raise AssertionError(f'{kind} {op} mutation failed before producing a readback\n'
                                                 f'{run.stdout}\n{run.stderr}')
                        caught += 1
                if not caught:
                    raise AssertionError('%s %s: no pixel told the mutated lowering from the reference' % (kind, op))
                print('%s %s mutation: %d pixels read other words' % (kind, op, caught), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        print('gfx2hlsl frexp/cvt draw fixture: controls passed')
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
