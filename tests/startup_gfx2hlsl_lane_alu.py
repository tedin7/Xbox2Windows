"""Shared runner for per-lane ALU table checks of tools/startup/gfx2hlsl.py.

A family module describes operations of the translator's VEC_U, VEC_F or SCA tables and calls main(family):

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel bodies and compare them with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

The pixel fixture is the one of startup_gfx2hlsl_bit_alu_check.py: it reads three flat parameters (v0..v3, v4..v7,
v8..v11), applies one operation per lane with sources v(4s + lane) and exports v12..v15 to mrt0, whose words the
renderer compares exactly. The vertex stage places the inputs in the parameters, so the compilers cannot fold them.

A family is a dict with:
  ops:       mnemonic -> (source count, reference over ints, VOP2 or VOP1 so the e32 and e64 spellings both exist)
  draw:      mnemonic -> input sets (src0, src1, src2), a multiple of four
  mutations: mnemonic -> (pattern in the translated pixel text, replacement, reference of the mutated lowering)
  body:      mnemonic -> the recorded pixel body words (llvm-mc -mcpu=gfx1013)
  refused:   assembly lines that must stay refused
  accepted:  assembly lines that must translate (one instruction, no pipeline)
  pipeline:  optional keyword arguments of the Pipeline built for the fixture (a selected contract)
Every reference is written from the ISA description with plain integer arithmetic, not from the lowering.
"""
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (POST_SOURCE, POST_WORDS, PRE_SOURCE, PRE_WORDS, Probe,  # noqa: E402
                                            vs_source, vs_words, words_text)

MASK = 0xffffffff


def body_source(family, op):
    arity, _, alternate = family['ops'][op]
    lines = []
    for lane in range(4):
        sources = ', '.join('v%d' % (4 * s + lane) for s in range(arity))
        lines.append('%s%s v%d, %s\n' % (op, '_e64' if alternate and lane % 2 else '', 12 + lane, sources))
    return ''.join(lines)


def translate(family, op, mutate=False):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), PRE_WORDS + family['body'][op] + POST_WORDS, root_blob(),
                      **family.get('pipeline', {}))
    text = pipe.hlsl_ps()
    if mutate:
        pattern, replacement, _ = family['mutations'][op]
        mutated, count = re.subn(pattern, replacement, text)
        assert count == 4 and mutated != text, (op, count)
        text = mutated
    return pipe, text


def refused(assembly):
    try:
        Probe().one(assembly.split(' ', 1)[0], assembly)
    except G.Unsupported:
        return True
    return False


def check(label, ok):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        raise AssertionError(label)


def expected(family, op, sets, mutated=False):
    arity, reference, _ = family['ops'][op]
    function = family['mutations'][op][2] if mutated else reference
    return [function(*x[:arity]) & MASK for x in sets]


def controls(family):
    for op, (arity, _, _) in family['ops'].items():
        _, text = translate(family, op)
        lanes = re.findall(r'^\s*v(1[2-5]) = (.*);$', text, re.M)
        check('%s: four lanes translated' % op, [lane for lane, _ in lanes] == ['12', '13', '14', '15'])
        translate(family, op, True)
        sets = family['draw'][op]
        check('%s: %d draw sets' % (op, len(sets)), sets and len(sets) % 4 == 0)
        # Each mutation reads a wrong word in the first draw, so the mutated draw cannot pass by accident.
        first = sets[:4]
        check('%s: mutation changes draw set 0' % op,
              expected(family, op, first, True) != expected(family, op, first))
    check('accepted forms translate', not any(refused(a) for a in family['accepted']))
    check('refused forms stay refused', all(refused(a) for a in family['refused']))


def check_binary(family, llvm_mc):
    for op in family['ops']:
        built = assemble(llvm_mc, body_source(family, op))
        if built != family['body'][op]:
            raise AssertionError('%s: assembled %s, recorded %s' % (op, built.hex(), family['body'][op].hex()))
    for name, source, words in (('prologue', PRE_SOURCE, PRE_WORDS), ('export', POST_SOURCE, POST_WORDS)):
        if assemble(llvm_mc, source) != words:
            raise AssertionError('%s fixture differs from llvm-mc' % name)
    for op, sets in family['draw'].items():
        for i in range(0, len(sets), 4):
            if assemble(llvm_mc, vs_source(sets[i:i + 4])) != vs_words(sets[i:i + 4]):
                raise AssertionError('vertex fixture for %s %s differs from llvm-mc' % (op, sets[i:i + 4]))
    print('fixture words match llvm-mc')


def check_draw(family, dxc, renderer, adapter, fxc=None):
    controls(family)
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    draws = mutations = 0
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op, sets in family['draw'].items():
                arity = family['ops'][op][0]
                good = compile_hlsl(compiler, kind, translate(family, op)[1], 'ps_%s_%s' % (op, kind), ps_profile,
                                    directory)
                for index in range(0, len(sets), 4):
                    chunk = sets[index:index + 4]
                    vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(chunk), PRE_WORDS + family['body'][op]
                                                                 + POST_WORDS, root_blob(),
                                                                 **family.get('pipeline', {})).hlsl_vs(),
                                      'vs_%s_%d_%s' % (op, index, kind), vs_profile, directory)
                    want = words_text(expected(family, op, chunk))
                    run = draw(renderer, adapter, vs, good, want)
                    if run.returncode:
                        raise AssertionError('%s %s inputs %s\n%s\n%s' % (kind, op, chunk, run.stdout, run.stderr))
                    draws += 1
                    print('%s %s %s: %s' % (kind, op, [tuple('%x' % v for v in x[:arity]) for x in chunk], want),
                          flush=True)
                    if index:
                        continue
                    # The same draw with the mutated lowering must read the mutated words, not the reference.
                    wrong = words_text(expected(family, op, chunk, True))
                    bad = compile_hlsl(compiler, kind, translate(family, op, True)[1], 'mutated_%s_%s' % (op, kind),
                                       ps_profile, directory)
                    run = draw(renderer, adapter, vs, bad, want)
                    if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                        raise AssertionError('%s %s mutation did not read %s\n%s\n%s'
                                             % (kind, op, wrong, run.stdout, run.stderr))
                    mutations += 1
                    print('%s %s mutation: fails as expected, reads %s' % (kind, op, wrong), flush=True)
    print('%s: %d draws matched, %d mutated draws read their mutated words' % (adapter, draws, mutations))


def main(family, doc):
    if len(sys.argv) == 1:
        controls(family)
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(family, *args)
    elif mode == 'draw':
        check_draw(family, *args)
    else:
        raise SystemExit(doc)


def words(hex_text):
    return bytes.fromhex(hex_text)


def signed(v):
    return v - (1 << 32) if v >> 31 else v


def body_words(family, llvm_mc):
    """Development aid: print the body words llvm-mc gives for every operation of a family."""
    for op in family['ops']:
        print("    '%s': words('%s')," % (op, assemble(llvm_mc, body_source(family, op)).hex()))
