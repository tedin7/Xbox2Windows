"""Binary16 VOPC compares in tools/startup/gfx2hlsl.py, drawn on a real D3D12 adapter (Part of #1723).

  (no arguments)                       translation controls and reference self-checks, pure Python
  binary <llvm-mc>                     reassemble the fixture bodies and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair (FXC ps_5_1 when given, DXC ps_6_0), draw one pixel
                                       per four input pairs and compare the four result words with the CPU reference

v_cmp_<relation>_f16 for the fifteen relations of the ISA (every one except tru, which no compare lowering accepts), e64 into
an SGPR on the even lanes and e32 into VCC on the odd ones, each followed by v_cndmask_b32 that turns the mask bit into
1.0 or 0.0. The pixel fixture is the one of startup_gfx2hlsl_bit_alu_check.py: three flat parameters (v0..v3, v4..v7,
v8..v11), the lanes' results in v12..v15 exported to mrt0; the vertex stage places the inputs in the parameters, so the
pixel stage cannot fold them. Every source word carries garbage in its upper half: the compares read the low half only.

The reference is the IEEE relation of the two exact binary32 images of the halves (Python floats), written from the ISA
meaning of each relation: ordered relations are false with a NaN, the negated ones true, lg is less-or-greater, o and u
test for NaN. The input pairs cover +0 and -0, an infinity, quiet and signalling NaNs of both signs and with payloads,
denormals, the largest finite half, adjacent halves and equal halves.

Negative controls: the translated shader with a relation weakened (< as <=, >= as >), a negation removed, the NaN test
forced to ordered, or the two sources exchanged must each differ from the reference on some pair.
"""
import re
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    POST_WORDS, PRE_WORDS, vs_source, vs_words, words_text)

ONE = 0x3f800000
HIGH = (0xabcd0000, 0x5a5a0000, 0xdead0000, 0x12340000)


def h2f(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 0xffff))[0]


def nan(x):
    return x != x


# relation -> reference over two binary32 images (Python float comparisons are IEEE: false with a NaN)
RELATIONS = {
    'f': lambda a, b: False,
    'lt': lambda a, b: a < b,
    'eq': lambda a, b: a == b,
    'le': lambda a, b: a <= b,
    'gt': lambda a, b: a > b,
    'lg': lambda a, b: a < b or a > b,
    'ge': lambda a, b: a >= b,
    'o': lambda a, b: not nan(a) and not nan(b),
    'u': lambda a, b: nan(a) or nan(b),
    'nge': lambda a, b: not a >= b,
    'nlg': lambda a, b: not (a < b or a > b),
    'ngt': lambda a, b: not a > b,
    'nle': lambda a, b: not a <= b,
    'neq': lambda a, b: not a == b,
    'nlt': lambda a, b: not a < b,
}

SPECIAL = (0x0000, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0x7e00)  # +0 -0 1.0 -1.0 +inf qNaN
EXTRA = ((0x3c01, 0x3c00), (0x3c00, 0x3c01), (0x03ff, 0x0001), (0x8001, 0x0001), (0x7bff, 0x7c00), (0xfc00, 0x7c00),
         (0x7c01, 0x3c00), (0xfe00, 0xfc00), (0xffff, 0xffff), (0x7d55, 0x7d55), (0x0001, 0x8001), (0x7bff, 0x7bff))
PAIRS = [(a, b) for a in SPECIAL for b in SPECIAL] + list(EXTRA)
assert len(PAIRS) % 4 == 0, len(PAIRS)


def sets(pairs):
    """Four lanes of (a, b, unused) per draw with garbage in the upper half of every word."""
    return [(a | HIGH[i % 4], b | HIGH[(i + 1) % 4], HIGH[(i + 2) % 4]) for i, (a, b) in enumerate(pairs)]


def reference(rel, pair):
    return ONE if RELATIONS[rel](h2f(pair[0]), h2f(pair[1])) else 0


def body_source(rel):
    lines = []
    for lane in range(4):
        a, b, dst = lane, 4 + lane, 12 + lane
        if lane % 2 == 0:
            lines.append('v_cmp_%s_f16_e64 s20, v%d, v%d\nv_cndmask_b32_e64 v%d, 0, 1.0, s20\n' % (rel, a, b, dst))
        else:
            lines.append('v_cmp_%s_f16_e32 vcc_lo, v%d, v%d\nv_cndmask_b32_e64 v%d, 0, 1.0, vcc_lo\n' % (rel, a, b, dst))
    return ''.join(lines)


# Recorded llvm-mc (gfx1013) encodings of body_source(rel); `binary <llvm-mc>` regenerates and compares them.
BODY_WORDS = {
    'f': bytes.fromhex(
        '1400c8d4000902000c0001d580e45100010b907d0d0001d580e4a9011400c8d4020d02000e0001d580e45100030f907d0f0001d580e4a901'),
    'lt': bytes.fromhex(
        '1400c9d4000902000c0001d580e45100010b927d0d0001d580e4a9011400c9d4020d02000e0001d580e45100030f927d0f0001d580e4a901'),
    'eq': bytes.fromhex(
        '1400cad4000902000c0001d580e45100010b947d0d0001d580e4a9011400cad4020d02000e0001d580e45100030f947d0f0001d580e4a901'),
    'le': bytes.fromhex(
        '1400cbd4000902000c0001d580e45100010b967d0d0001d580e4a9011400cbd4020d02000e0001d580e45100030f967d0f0001d580e4a901'),
    'gt': bytes.fromhex(
        '1400ccd4000902000c0001d580e45100010b987d0d0001d580e4a9011400ccd4020d02000e0001d580e45100030f987d0f0001d580e4a901'),
    'lg': bytes.fromhex(
        '1400cdd4000902000c0001d580e45100010b9a7d0d0001d580e4a9011400cdd4020d02000e0001d580e45100030f9a7d0f0001d580e4a901'),
    'ge': bytes.fromhex(
        '1400ced4000902000c0001d580e45100010b9c7d0d0001d580e4a9011400ced4020d02000e0001d580e45100030f9c7d0f0001d580e4a901'),
    'o': bytes.fromhex(
        '1400cfd4000902000c0001d580e45100010b9e7d0d0001d580e4a9011400cfd4020d02000e0001d580e45100030f9e7d0f0001d580e4a901'),
    'u': bytes.fromhex(
        '1400e8d4000902000c0001d580e45100010bd07d0d0001d580e4a9011400e8d4020d02000e0001d580e45100030fd07d0f0001d580e4a901'),
    'nge': bytes.fromhex(
        '1400e9d4000902000c0001d580e45100010bd27d0d0001d580e4a9011400e9d4020d02000e0001d580e45100030fd27d0f0001d580e4a901'),
    'nlg': bytes.fromhex(
        '1400ead4000902000c0001d580e45100010bd47d0d0001d580e4a9011400ead4020d02000e0001d580e45100030fd47d0f0001d580e4a901'),
    'ngt': bytes.fromhex(
        '1400ebd4000902000c0001d580e45100010bd67d0d0001d580e4a9011400ebd4020d02000e0001d580e45100030fd67d0f0001d580e4a901'),
    'nle': bytes.fromhex(
        '1400ecd4000902000c0001d580e45100010bd87d0d0001d580e4a9011400ecd4020d02000e0001d580e45100030fd87d0f0001d580e4a901'),
    'neq': bytes.fromhex(
        '1400edd4000902000c0001d580e45100010bda7d0d0001d580e4a9011400edd4020d02000e0001d580e45100030fda7d0f0001d580e4a901'),
    'nlt': bytes.fromhex(
        '1400eed4000902000c0001d580e45100010bdc7d0d0001d580e4a9011400eed4020d02000e0001d580e45100030fdc7d0f0001d580e4a901'),
}


def same_source_words(rel):
    """BODY_WORDS[rel] with every compare reading src0 for both sources (v_cmp_<rel>_f16 v0, v0): DXC folds a reflexive
    float compare to a constant, which a NaN half must not follow. VOP3 (0xd4 opcode byte): src1 field is bits 9..17 of the
    second dword; VOPC e32 (0x7d): VSRC1 is bits 9..16 of the first."""
    data = BODY_WORDS[rel]
    words = [int.from_bytes(data[i:i + 4], 'little') for i in range(0, len(data), 4)]
    patched, i = [], 0
    while i < len(words):
        w = words[i]
        if w >> 24 == 0xd4:  # VOP3 compare: two dwords
            src0 = words[i + 1] & 0x1ff
            patched += [w, (words[i + 1] & ~(0x1ff << 9)) | (src0 << 9)]
            i += 2
        elif w >> 24 == 0xd5:  # v_cndmask_b32_e64: two dwords, unchanged
            patched += [w, words[i + 1]]
            i += 2
        else:  # VOPC e32 compare (0x7d...)
            src0 = w & 0x1ff
            patched.append((w & ~(0xff << 9)) | ((src0 & 0xff) << 9))
            i += 1
    return b''.join(x.to_bytes(4, 'little') for x in patched)


def translate(rel, same=False):
    body = same_source_words(rel) if same else BODY_WORDS[rel]
    return G.Pipeline(vs_words(sets(PAIRS[:4])), PRE_WORDS + body + POST_WORDS, root_blob()).hlsl_ps()


# Relations whose reflexive compare a compiler may fold (eq, le, ge true, neq false) and so need the NaN guard.
SAME_SOURCE = ('eq', 'le', 'ge', 'neq')


def mutants():
    """(label, relation, mutation of the translated text); each must change a result on some input pair."""
    return (
        ('lt as le', 'lt', lambda t: t.replace(' < f16tof32', ' <= f16tof32')),
        ('ge as gt', 'ge', lambda t: t.replace(' >= f16tof32', ' > f16tof32')),
        ('neq without the negation', 'neq', lambda t: t.replace('((!(', '(((')),
        ('nlt without the negation', 'nlt', lambda t: t.replace('((!(', '(((')),
        ('o always true', 'o', lambda t: t.replace('<= 0x7c00u', '<= 0xffffu')),
        ('u always false', 'u', lambda t: t.replace('<= 0x7c00u', '<= 0xffffu')),
        ('lt with exchanged sources', 'lt', lambda t: re.sub(r'f16tof32\((v\d) & 0xffffu\) < f16tof32\((v\d) & 0xffffu\)',
                                                              r'f16tof32(\2 & 0xffffu) < f16tof32(\1 & 0xffffu)', t)),
    )


def controls():
    failures = []

    def check(label, ok):
        print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
        if not ok:
            failures.append(label)

    # The model itself, by hand.
    n, one, zero, mzero, inf = 0x7e00, 0x3c00, 0x0000, 0x8000, 0x7c00
    check('model: -0 == +0', reference('eq', (mzero, zero)) == ONE)
    check('model: NaN < 1 is false, nlt is true', reference('lt', (n, one)) == 0 and reference('nlt', (n, one)) == ONE)
    check('model: NaN == NaN is false, neq is true', reference('eq', (n, n)) == 0 and reference('neq', (n, n)) == ONE)
    check('model: lg is false for equal and for NaN', reference('lg', (one, one)) == 0 and reference('lg', (n, one)) == 0)
    check('model: lg is true for 1 vs inf', reference('lg', (one, inf)) == ONE)
    check('model: o and u', reference('o', (one, inf)) == ONE and reference('u', (one, n)) == ONE and reference('o', (n, one)) == 0)
    check('model: a denormal is below 1 and above zero', reference('lt', (0x0001, one)) == ONE and reference('gt', (0x0001, zero)) == ONE)
    check('model: -inf < the smallest negative denormal', reference('lt', (0xfc00, 0x8001)) == ONE)
    check('model: f is false', all(reference('f', p) == 0 for p in PAIRS))
    check('pairs: every relation has a true and a false pair (except f)',
          all({reference(r, p) for p in PAIRS} == {0, ONE} for r in RELATIONS if r != 'f'))
    check('pairs: NaN operands present on both sides, both signs',
          {a for a, _ in PAIRS if nan(h2f(a))} >= {0x7e00, 0x7c01, 0xfe00, 0xffff, 0x7d55} and
          {b for _, b in PAIRS if nan(h2f(b))} >= {0x7e00, 0xffff, 0x7d55})
    check('draw: the pairs fill whole draws of four lanes', len(PAIRS) % 4 == 0)
    for rel in RELATIONS:
        try:
            text = translate(rel)
        except G.Unsupported as e:
            check('%s: translates (%s)' % (rel, e), False)
            continue
        check('%s: translates to four lane results' % rel, all(('v%d = ' % r) in text for r in range(12, 16)))
        check('%s: reads the low half through f16tof32' % rel, rel == 'f' or 'f16tof32(v0 & 0xffffu)' in text or rel in ('o', 'u'))
    for rel in SAME_SOURCE:
        text = translate(rel, True)
        check('%s: same-source compare keeps the NaN guard' % rel, '(v0 & 0x7fffu) <= 0x7c00u' in text)
        check('%s: same-source model: NaN against itself' % rel,
              reference(rel, (n, n)) == (ONE if rel == 'neq' else 0) and reference(rel, (one, one)) == (0 if rel == 'neq' else ONE))
    if failures:
        raise SystemExit('FAILED: ' + ', '.join(failures))


def check_binary(llvm_mc):
    for rel in RELATIONS:
        built = assemble(llvm_mc, body_source(rel))
        if built != BODY_WORDS[rel]:
            raise AssertionError('%s: assembled %s, recorded %s' % (rel, built.hex(), BODY_WORDS[rel].hex()))
    for i in range(0, len(PAIRS), 4):
        if assemble(llvm_mc, vs_source(sets(PAIRS[i:i + 4]))) != vs_words(sets(PAIRS[i:i + 4])):
            raise AssertionError('vertex fixture %d differs from llvm-mc' % i)
    print('fixture words match llvm-mc')


def run_draws(kind, compiler, vs_profile, ps_profile, renderer, adapter, directory, label, rel, text, expect_fail, same=False):
    """Draw every chunk of PAIRS; returns the failing chunk indexes (all of them must pass, or for a mutant some must fail)."""
    ps = compile_hlsl(compiler, kind, text, 'ps_%s_%s_%s' % (label, rel, kind), ps_profile, directory)
    bad = []
    for index in range(0, len(PAIRS), 4):
        chunk = PAIRS[index:index + 4]
        pipe_ps = PRE_WORDS + (same_source_words(rel) if same else BODY_WORDS[rel]) + POST_WORDS
        vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(sets(chunk)), pipe_ps, root_blob()).hlsl_vs(),
                          'vs_%s_%d_%s' % (rel, index, kind), vs_profile, directory)
        expected = words_text(reference(rel, (p[0], p[0]) if same else p) for p in chunk)
        run = draw(renderer, adapter, vs, ps, expected)
        if run.returncode:
            bad.append((index, chunk, run.stderr.strip()))
            if expect_fail:
                return bad  # one differing chunk proves the mutant is caught
    return bad


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for rel in RELATIONS:
                bad = run_draws(kind, compiler, vs_profile, ps_profile, renderer, adapter, directory, 'good', rel,
                                translate(rel), False)
                if bad:
                    failures.append('%s v_cmp_%s_f16' % (kind, rel))
                    print('%s %s MISMATCH chunk %d pairs %s\n%s' % (kind, rel, bad[0][0] // 4, bad[0][1], bad[0][2]), flush=True)
                else:
                    print('%s v_cmp_%s_f16: %d pairs match' % (kind, rel, len(PAIRS)), flush=True)
            for rel in SAME_SOURCE:
                bad = run_draws(kind, compiler, vs_profile, ps_profile, renderer, adapter, directory, 'same', rel,
                                translate(rel, True), False, True)
                if bad:
                    failures.append('%s v_cmp_%s_f16 v0, v0' % (kind, rel))
                    print('%s %s same source MISMATCH chunk %d pairs %s\n%s' % (kind, rel, bad[0][0] // 4, bad[0][1], bad[0][2]), flush=True)
                else:
                    print('%s v_cmp_%s_f16 same source: %d pairs match' % (kind, rel, len(PAIRS)), flush=True)
            if kind == 'dxc':  # DXC without -Gis is the compiler that folds x == x; the guard must not be dropped
                text = translate('eq', True).replace('<= 0x7c00u', '<= 0xffffu')
                bad = run_draws(kind, compiler, vs_profile, ps_profile, renderer, adapter, directory, 'same_noguard', 'eq',
                                text, True, True)
                if not bad:
                    failures.append('%s control same-source eq without the NaN guard was not detected' % kind)
                    print('%s control same-source eq without the NaN guard: NOT DETECTED' % kind, flush=True)
                else:
                    print('%s control same-source eq without the NaN guard: fails as expected' % kind, flush=True)
            for label, rel, mutate in mutants():
                text = mutate(translate(rel))
                if text == translate(rel):
                    raise AssertionError('mutant %s changed nothing' % label)
                bad = run_draws(kind, compiler, vs_profile, ps_profile, renderer, adapter, directory,
                                label.replace(' ', '_'), rel, text, True)
                if not bad:
                    failures.append('%s control %s was not detected' % (kind, label))
                    print('%s control %s: NOT DETECTED' % (kind, label), flush=True)
                else:
                    print('%s control %s: fails as expected (chunk %d)' % (kind, label, bad[0][0] // 4), flush=True)
    if failures:
        raise AssertionError('draw mismatches: ' + ', '.join(failures))


def main():
    if len(sys.argv) == 1:
        controls()
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
