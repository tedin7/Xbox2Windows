"""Unsigned 64-bit VOPC compares in tools/startup/gfx2hlsl.py: v_cmp_*_u64 and v_cmpx_*_u64, e32 and e64.

  (no arguments)                       translation controls and fixture checks, pure Python
  binary <llvm-mc> <llvm-objcopy>      assemble every fixture with LLVM (-filetype=obj, .text extracted with
                                       llvm-objcopy) and compare with the words built here
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the pixel fixtures with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw 16 x 16 pixels and compare every pixel with the plain-integer reference

Semantics, from the AMD RDNA 2 ISA guide (VOPC opcodes): V_CMP_<C>_U64 sets the lane bit to S0.u64 <C> S1.u64 over
the register pairs (low dword first); v_cmpx writes the result to EXEC. The lowering compares the high words first and
the low words only when the high words are equal, and ANDs the result with the incoming EXEC when EXEC is not known.
The reference below is plain Python integer arithmetic on the 64-bit value (high << 32 | low).

Controls: the six relations of v_cmp and v_cmpx in e32 and e64 forms (vcc, vcc_lo, s4 and exec destinations) evaluated
on the CPU with the integer evaluator of startup_gfx2hlsl_bit_alu_check.py over a value set whose members disagree
between the high and the low halves (the high word smaller while the low word is larger, and the reverse), with SGPR
pair, inline (-1 sign-extends to 64 bits) and VGPR sources; the EXEC intersection when EXEC is not known; the
refusals (literal and float sources, vcc/exec/lane-mask sources, an odd SGPR pair, NEG/ABS bits, clamp, op_sel, the
f and t conditions, the signed i64 form which stays refused).

The draws: a pixel program builds two 64-bit values per pixel from the packed position p = x | y << 16: A = (a, x & 3)
and B = (a when y is odd else b, x >> 2 & 3), with a = p * 0x9b5d and b = a ^ (p * 0x6d2b & 0xffff800f) (32-bit
products), so the high words take all 16 combinations, odd rows have equal low words, and a pixel exists where the high
word is smaller and the low word larger and the reverse.
  K  v_cmpx_lt_u64_e32 v[4:5], v[6:7]   (a killed pixel keeps the clear value -1.0)
  M  for each of lt, eq, le, gt, ne, ge (bit k): channel 0 v_cmp_C_u64_e32 vcc_lo, v[4:5], v[6:7]; channel 1
     v_cmp_C_u64_e64 s4, v[6:7], v[4:5]; channel 2 v_cmp_C_u64_e64 s4, 5, v[4:5]; channel 3 v_cmp_C_u64_e64 s4, -1,
     v[6:7]; each bit folded in with v_cndmask_b32_e64 and v_or_b32, then every word ORed with 1.0 and exported.
Three mutated translations: `high only` (the tie term is dropped from the ordered relations), `no tie gate` (the low
compare is ORed in whatever the high words say), `zext inline` (inline -1 reads as 0x00000000ffffffff). Each mutated
draw must fail against the reference and match the image of the same mutated formula.

Not covered: gfx1013 hardware (the draws run the translated HLSL on desktop adapters), literal sources (refused), the
signed i64 compares (still refused), and the DPP forms.
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
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_cmp16_check as C  # noqa: E402
import startup_image_3d_check as I3  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
CONDITIONS = C.CONDITIONS
RELATION = C.RELATION
check = C.check
V, inline, e32, e64 = C.V, C.inline, C.e32, C.e64
WIDTH, HEIGHT, CLEAR, ONE, PACKED = C.WIDTH, C.HEIGHT, C.CLEAR, C.ONE, C.PACKED
VCC_LO, SGPR_DST, LITERAL = C.VCC_LO, C.SGPR_DST, C.LITERAL
OPCODES = {}
for _op, (_name, _) in D.VOPC_OPS.items():
    _m = re.fullmatch(r'v_cmp(x?)_(\w+)_([iu]64)_e32', _name)
    if _m:
        OPCODES[_m.group(1) == 'x', _m.group(2), _m.group(3)] = _op
OP = {key[:2]: op for key, op in OPCODES.items() if key[2] == 'u64' and key[1] in CONDITIONS}

# Values whose halves disagree: the high word is smaller while the low word is larger, and the reverse.
VALUES = [0, 1, 0xffffffff, 0x100000000, 0x1ffffffff, 0xffffffff00000000, 0x8000000000000000, 0x7fffffffffffffff,
          0xffffffffffffffff, 0x200000001, 0x100000002, 0x123456789abcdef0]
PAIRS = [(a, b) for a in VALUES for b in VALUES]


def words(value):
    return value & MASK, value >> 32


def reference(cond, a, b):
    return int(RELATION[cond](a, b))


def evaluate(expr, env, exec_value=None):
    env = dict(env)
    if exec_value is not None:
        env['exec'] = exec_value
    return Expr(expr).run(env)[0]


def controls():
    check('24 opcodes found: 6 conditions x v_cmp/v_cmpx x e32 in the decoder table (u64)', len(OP) == 12
          and sorted(OP) == sorted((x, c) for x in (False, True) for c in CONDITIONS), str(sorted(OP)))
    # 1. Every form, every pair of values, against the 64-bit reference.
    sources = (('v[2:3] v[4:5]', V(2), V(4), None), ('s[2:3] v[4:5]', 2, V(4), None))
    for cmpx in (False, True):
        for cond in CONDITIONS:
            op = OP[cmpx, cond]
            forms = []
            for label, src0, src1, _ in sources:
                if cmpx:
                    forms += [('e32 ' + label, e32(op, src0, src1 - 256), 'exec'),
                              ('e64 ' + label, e64(op, 0, src0, src1), 'exec')]
                else:
                    forms += [('e32 ' + label, e32(op, src0, src1 - 256), 'vcc'),
                              ('e64 vcc_lo ' + label, e64(op, VCC_LO, src0, src1), 'vcc'),
                              ('e64 s4 ' + label, e64(op, SGPR_DST, src0, src1), 's4')]
            forms += [('e64 inline -1 v[4:5]', e64(op, 0 if cmpx else SGPR_DST, inline(-1), V(4)),
                       'exec' if cmpx else 's4'),
                      ('e64 inline 5 v[4:5]', e64(op, 0 if cmpx else SGPR_DST, inline(5), V(4)),
                       'exec' if cmpx else 's4')]
            for label, code, dst in forms:
                asm, got_dst, expr = C.translate(code)
                ok = got_dst == dst
                inline_a = 0xffffffffffffffff if 'inline -1' in label else 5 if 'inline 5' in label else None
                for a, b in PAIRS:
                    a = inline_a if inline_a is not None else a
                    env = {}
                    for names, value in ((('v2', 'v3'), a), (('s2', 's3'), a), (('v4', 'v5'), b)):
                        env[names[0]], env[names[1]] = words(value)
                    ok = ok and evaluate(expr, env) == reference(cond, a, b)
                check('%s %s: %s -> %s %s' % ('v_cmpx' if cmpx else 'v_cmp', cond, label, asm, dst), ok, expr)
            # The EXEC intersection when EXEC is not known: the result is ANDed with the incoming EXEC.
            _, _, expr = C.translate(e32(op, V(2), 4), exec_known=False)
            ok = all(evaluate(expr, dict(zip(('v2', 'v3', 'v4', 'v5'), words(a) + words(b))), 0xdeadbeef)
                     == (0xdeadbeef & reference(cond, a, b)) for a, b in PAIRS)
            check('%s %s: result ANDed with the incoming EXEC' % ('v_cmpx' if cmpx else 'v_cmp', cond), ok, expr)

    # 2. The value set separates the high-first rule from the common mistakes.
    def high_only(cond, a, b):
        (al, ah), (bl, bh) = words(a), words(b)
        return int(ah < bh if cond in ('lt', 'le') else ah > bh) if cond in ('lt', 'le', 'gt', 'ge') else reference(cond, a, b)

    def low_only(cond, a, b):
        return int(RELATION[cond](a & MASK, b & MASK))

    for cond in CONDITIONS:
        check('values separate %s from a low-word-only compare' % cond,
              any(low_only(cond, a, b) != reference(cond, a, b) for a, b in PAIRS))
        if cond in ('lt', 'le', 'gt', 'ge'):
            check('values separate %s from a high-word-only compare' % cond,
                  any(high_only(cond, a, b) != reference(cond, a, b) for a, b in PAIRS))

    # 3. Refusals.
    op = OP[False, 'lt']
    cases = {
        'literal source': e32(op, LITERAL, 4, 0x1234),
        'odd SGPR pair': e32(op, 3, 4),
        'vcc source': e32(op, 106, 4),
        'exec source': e32(op, 126, 4),
        'float source': e32(op, 240, 4),
        'NEG bit': e64(op, SGPR_DST, V(2), V(4), neg=1),
        'ABS bit': e64(op, SGPR_DST, V(2), V(4), absolute=1),
        'clamp': e64(op, SGPR_DST, V(2), V(4), clamp=1),
        'op_sel': e64(op, SGPR_DST, V(2), V(4), opsel=1),
    }
    for label, code in cases.items():
        found = refusal_text(code)
        check('refused: %s' % label, found is not None, str(found))
    for cond in ('f', 't'):
        code = e32(next(o for o, (n, _) in D.VOPC_OPS.items() if n == 'v_cmp_%s_u64_e32' % cond), V(2), 4)
        check('refused: v_cmp_%s_u64' % cond, refusal_text(code) is not None)
    code = e32(next(o for o, (n, _) in D.VOPC_OPS.items() if n == 'v_cmp_lt_i64_e32'), V(2), 4)
    check('refused: the signed v_cmp_lt_i64 stays unlowered', refusal_text(code) is not None)
    stage = C.Probe()
    stage.masks = {2}
    name, asm = C.decode_one(e32(op, 2, 4))
    try:
        stage.one(name, asm)
        refused = False
    except G.Unsupported:
        refused = True
    check('refused: an SGPR pair holding a lane mask', refused)


def refusal_text(code):
    return C.refusal(code)


# ---- the pixel fixtures --------------------------------------------------------------------------------------------
def vop2(op, vdst, src0, vsrc1):
    return struct.pack('<I', (op << 25) | (vdst << 17) | (vsrc1 << 9) | src0)


def vop1(op, vdst, src0):
    return struct.pack('<I', (0b0111111 << 25) | (vdst << 17) | (op << 9) | src0)


def cndmask_e64(vdst, src0, src1, mask):
    return struct.pack('<II', 0xd5010000 | vdst, src0 | (src1 << 9) | (mask << 18))


V_MOV, V_AND, V_OR, V_LSHRREV = 1, 0x1b, 0x1c, 0x16
CMP_EQ_U32 = next(o for o, (n, _) in D.VOPC_OPS.items() if n == 'v_cmp_eq_u32_e32')
PROLOGUE = C.PROLOGUE
PROLOGUE_BYTES = bytes.fromhex(C.PROLOGUE_WORDS)
TAIL_K = bytes.fromhex('f20006380f1800f803030303000081bf')   # v_or_b32 v3, 1.0, v0; exp mrt0 v3 x4 done vm; s_endpgm
TAIL_M = bytes.fromhex('f2181838f21a1a38f21c1c38f21e1e380f1800f80c0d0e0f000081bf')

# v4 = a; v5 = x & 3; v6 = a (odd y) or b (even y); v7 = x >> 2 & 3.
PAIR_SOURCE = ('v_mov_b32_e32 v4, v1\nv_and_b32_e32 v5, 3, v0\nv_lshrrev_b32_e32 v8, 16, v0\nv_and_b32_e32 v8, 1, v8\n'
               'v_cmp_eq_u32_e32 vcc_lo, 1, v8\nv_cndmask_b32_e64 v6, v2, v1, vcc_lo\n'
               'v_lshrrev_b32_e32 v7, 2, v0\nv_and_b32_e32 v7, 3, v7\n')
PAIR_BYTES = (vop1(1, 4, V(1)) + vop2(V_AND, 5, inline(3), 0) + vop2(V_LSHRREV, 8, inline(16), 0)
              + vop2(V_AND, 8, inline(1), 8) + C.e32(CMP_EQ_U32, inline(1), 8)
              + cndmask_e64(6, V(2), V(1), VCC_LO) + vop2(V_LSHRREV, 7, inline(2), 0)
              + vop2(V_AND, 7, inline(3), 7))
K_SOURCE = PROLOGUE + PAIR_SOURCE + 'v_cmpx_lt_u64_e32 v[4:5], v[6:7]\n' + 'v_or_b32_e32 v3, 1.0, v0\nexp mrt0 v3, v3, v3, v3 done vm\ns_endpgm\n'
K_WORDS = PROLOGUE_BYTES + PAIR_BYTES + e32(OP[True, 'lt'], V(4), 6) + TAIL_K
M_CHANNELS = (('v_cmp_%s_u64_e32 vcc_lo, v[4:5], v[6:7]', 'vcc_lo', lambda c: e32(OP[False, c], V(4), 6), VCC_LO),
              ('v_cmp_%s_u64_e64 s4, v[6:7], v[4:5]', 's4', lambda c: e64(OP[False, c], SGPR_DST, V(6), V(4)), SGPR_DST),
              ('v_cmp_%s_u64_e64 s4, 5, v[4:5]', 's4', lambda c: e64(OP[False, c], SGPR_DST, inline(5), V(4)), SGPR_DST),
              ('v_cmp_%s_u64_e64 s4, -1, v[6:7]', 's4', lambda c: e64(OP[False, c], SGPR_DST, inline(-1), V(6)), SGPR_DST))
M_SOURCE = PROLOGUE + PAIR_SOURCE + ''.join('v_mov_b32_e32 v%d, 0\n' % r for r in range(12, 16)) + ''.join(
    (cmp % cond) + '\nv_cndmask_b32_e64 v10, 0, %d, %s\nv_or_b32_e32 v%d, v%d, v10\n' % (1 << k, mask, 12 + c, 12 + c)
    for k, cond in enumerate(CONDITIONS) for c, (cmp, mask, _, _) in enumerate(M_CHANNELS)) + ''.join(
    'v_or_b32_e32 v%d, 1.0, v%d\n' % (r, r) for r in range(12, 16)) + 'exp mrt0 v12, v13, v14, v15 done vm\ns_endpgm\n'
M_WORDS = PROLOGUE_BYTES + PAIR_BYTES + b''.join(vop1(1, r, 128) for r in range(12, 16)) + b''.join(
    make(cond) + cndmask_e64(10, 128, 128 + (1 << k), mask_reg) + vop2(V_OR, 12 + c, V(12 + c), 10)
    for k, cond in enumerate(CONDITIONS) for c, (_, _, make, mask_reg) in enumerate(M_CHANNELS)) + TAIL_M
FIXTURES = {'K': (K_SOURCE, K_WORDS), 'M': (M_SOURCE, M_WORDS)}

MUTATIONS = {
    'high only': (r' \| \(\((\w+) == (\w+)\) & \((\w+) [<>]=? (\w+)\)\)', '', ('K', 'M')),
    'no tie gate': (r'\(\((\w+) == (\w+)\) & \((\w+) ([<>]=?) (\w+)\)\)', r'(\3 \4 \5)', ('K', 'M')),
    'zext inline': (r'\(\(4294967295u ', '((0u ', ('M',)),
}


def pairs(position):
    a, b = C.operands(position)
    x, y = position & 0xffff, position >> 16
    low_b = a if y & 1 else b
    return (a, x & 3), (low_b, x >> 2 & 3)


def holds(cond, x, y, how, inline_neg=False):
    """x and y are (low, high). inline_neg: x is the constant -1 (sign-extended unless the mutation zero-extends)."""
    (xl, xh), (yl, yh) = x, y
    if how == 'zext inline' and inline_neg:
        xh = 0
    if how == 'high only' and cond in ('lt', 'le', 'gt', 'ge'):
        return int(xh < yh if cond in ('lt', 'le') else xh > yh)
    if how == 'no tie gate' and cond in ('lt', 'le', 'gt', 'ge'):
        strict = RELATION['lt' if cond in ('lt', 'le') else 'gt']
        return int(strict(xh, yh) or RELATION[cond](xl, yl))
    return int(RELATION[cond](xh << 32 | xl, yh << 32 | yl))


def image(program, how='reference'):
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            position = x | (y << 16)
            a, b = pairs(position)
            if program == 'K':
                out += [ONE | position] * 4 if holds('lt', a, b, how) else [CLEAR] * 4
            else:
                channels = [(a, b, False), (b, a, False), ((5, 0), a, False), ((MASK, MASK), b, True)]
                out += [ONE | sum(holds(cond, p, q, how, neg) << k for k, cond in enumerate(CONDITIONS))
                        for p, q, neg in channels]
    return out


def pixel_hlsl(program, how='reference'):
    text = G.Pipeline(I3.VS_WORDS, FIXTURES[program][1], I3.root_blob(), ps_hw_inputs=PACKED).hlsl_ps()
    if how != 'reference':
        pattern, replacement, programs = MUTATIONS[how]
        text, count = re.subn(pattern, replacement, text)
        assert count > 0 or program not in programs, (program, how)
    return text


def expected_decode(source):
    lines = [line.replace('0x3f800000', '1.0') for line in source.splitlines()]
    return [re.sub(r'^(v_cmpx_\w+_e64) ', r'\1 s0, ', line) for line in lines]


def check_fixture():
    for name, (source, code) in FIXTURES.items():
        decoded = [asm for _o, _s, _n, asm in G.decode(code)]
        check('%s fixture decodes to its source' % name, decoded == expected_decode(source), str(decoded))
    recorded = [(cndmask_e64(10, 128, 129, VCC_LO), '0a0001d58002a901'), (vop2(V_OR, 12, V(12), 10), '0c151838'),
                (vop1(1, 12, 128), '8002187e')]
    check('encoders match recorded LLVM 15 encodings', all(code.hex() == want for code, want in recorded),
          str([(code.hex(), want) for code, want in recorded if code.hex() != want]))
    reference_image = image('K')
    kept = sum(reference_image[i] != CLEAR for i in range(0, len(reference_image), 4))
    check('K keeps %d of %d pixels' % (kept, WIDTH * HEIGHT), 0 < kept < WIDTH * HEIGHT)
    cases = {(x, y): pairs(x | y << 16) for y in range(HEIGHT) for x in range(WIDTH)}
    check('a pixel with the high word smaller and the low word larger',
          any(a[1] < b[1] and a[0] > b[0] for a, b in cases.values()))
    check('a pixel with the high word larger and the low word smaller',
          any(a[1] > b[1] and a[0] < b[0] for a, b in cases.values()))
    check('a pixel with equal high words and a smaller low word', any(a[1] == b[1] and a[0] < b[0] for a, b in cases.values()))
    check('a pixel with equal 64-bit values', any(a == b for a, b in cases.values()))
    m = image('M')
    for channel in (0, 1):
        for k, cond in enumerate(CONDITIONS):
            bits = {m[4 * i + channel] >> k & 1 for i in range(WIDTH * HEIGHT)}
            check('M channel %d %s takes both values' % (channel, cond), bits == {0, 1})
    for how, (_, _, programs) in MUTATIONS.items():
        for program in programs:
            pixel_hlsl(program, how)
            differing = sum(a != b for a, b in zip(image(program), image(program, how)))
            check('%s: %s mutation changes %d words' % (program, how, differing), differing > 0)


# ---- LLVM ----------------------------------------------------------------------------------------------------------
def check_binary(llvm_mc, objcopy):
    for name, (source, code) in FIXTURES.items():
        built = C.assemble(llvm_mc, objcopy, source)
        check('%s: LLVM assembles the fixture to the words built here' % name, built == code,
              '%s != %s' % (built.hex(), code.hex()))


# ---- draws ---------------------------------------------------------------------------------------------------------
def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = I3.compile_hlsl(compiler, kind, I3.FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            for program in ('K', 'M'):
                reference_file = C.words_file(directory, '%s_reference' % program, image(program))
                ps = I3.compile_hlsl(compiler, kind, pixel_hlsl(program), 'ps_%s_%s' % (program, kind), ps_profile,
                                     directory)
                run = C.draw(renderer, adapter, vs, ps, reference_file)
                print('  ' + ' '.join(ln for ln in run.stdout.splitlines()
                                      if ln.startswith(('backend=', 'adapter=', 'driver=', 'debug_errors='))))
                check('%s %s %s: all %d pixels match the reference' % (kind, adapter, program, WIDTH * HEIGHT),
                      run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                      run.stdout + run.stderr)
                for how, (_, _, programs) in MUTATIONS.items():
                    if program not in programs:
                        continue
                    tag = '%s_%s_%s' % (program, how.replace(' ', '_'), kind)
                    bad = I3.compile_hlsl(compiler, kind, pixel_hlsl(program, how), 'ps_' + tag, ps_profile, directory)
                    run = C.draw(renderer, adapter, vs, bad, reference_file)
                    lines = [ln for ln in run.stderr.splitlines() if 'pixel' in ln]
                    check('%s %s %s %s mutation fails against the reference: %s'
                          % (kind, adapter, program, how, ' / '.join(lines[:1])),
                          run.returncode != 0 and lines and lines[0].startswith('pixel mismatch'),
                          run.stdout + run.stderr)
                    own = C.words_file(directory, tag, image(program, how))
                    run = C.draw(renderer, adapter, vs, bad, own)
                    check('%s %s %s %s mutation matches its own image' % (kind, adapter, program, how),
                          run.returncode == 0, run.stdout + run.stderr)


def main():
    if len(sys.argv) == 1:
        controls()
        check_fixture()
    else:
        mode, *args = sys.argv[1:]
        if mode == 'binary' and len(args) == 2:
            check_fixture()
            check_binary(*args)
        elif mode == 'draw' and len(args) in (3, 4):
            check_fixture()
            check_draw(*args)
        else:
            raise SystemExit(__doc__)
    C.finish()


if __name__ == '__main__':
    main()
