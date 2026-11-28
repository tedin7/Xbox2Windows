"""v_mad_u64_u32 and v_mad_i64_i32 in tools/startup/gfx2hlsl.py (Part of #1723).

D = S0 * S1 + S2 modulo 2^64 with two 32-bit factors and a 64-bit S2, written to a VGPR pair. The unsigned form
also writes the carry-out of the 64-bit add to a lane mask; the signed form is accepted only with `null`.

  (no arguments)                       pure Python controls described below
  binary <llvm-mc>                     reassemble the pixel fixtures and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pixel shaders (FXC ps_5_1 when given, DXC ps_6_0),
                                       draw one pixel per input and compare mrt0 with the 64-bit reference

Draw pixel: v0 = S0, v1 = S1, v[2:3] = the 64-bit S2. The unsigned shader exports (lo, hi, 1.0f when carry else 0, 0), the signed
shader exports (lo, hi, 0, 0). A mutation that drops the low-word carry into the high word must read the exact
mutated words, so the draw cannot pass by accident.

Controls: translate each form, evaluate the emitted helpers in 32-bit arithmetic against an independent
Python 64-bit reference over a value grid, and check the refusals. Pure Python, no GPU and no assembler.
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
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    MASK, POST_SOURCE, POST_WORDS, PRE_SOURCE, PRE_WORDS, Expr, Probe, check, vs_source, vs_words, words_text)

M64 = (1 << 64) - 1
VALUES = (0, 1, 2, 0xffff, 0x10000, 0x7fffffff, 0x80000000, 0x89abcdef, 0xfffffffe, 0xffffffff)
ADDENDS = (0, 1, M64, 1 << 63, 0x00000001ffffffff, 0xfedcba9876543210, 0x7fffffffffffffff)

BODY_SOURCE = {
    False: ('v_mad_u64_u32 v[12:13], vcc_lo, v0, v1, v[2:3]\n'
            'v_cndmask_b32_e64 v14, 0, 1.0, vcc_lo\nv_mov_b32 v15, 0\n'),
    True: 'v_mad_i64_i32 v[12:13], null, v0, v1, v[2:3]\nv_mov_b32 v14, 0\nv_mov_b32 v15, 0\n',
}
BODY_WORDS = {
    False: bytes.fromhex('0c6a76d500030a040e0001d580e4a90180021e7e'),
    True: bytes.fromhex('0c7d77d500030a0480021c7e80021e7e'),
}
# (S0, S1, S2): carry out of the low word, carry out of the 64-bit add, negative factors, the largest products.
DRAWS = (
    (0xffffffff, 0xffffffff, 0xffffffffffffffff),
    (0x89abcdef, 0x76543210, 0xfedcba9876543210),
    (0xffffffff, 0x00000002, 0x00000001ffffffff),
    (0x80000000, 0x80000000, 0x8000000000000000),
    (0x80000000, 0x7fffffff, 0x7fffffffffffffff),
    (0xfffffffe, 0x00010001, 0x00000000ffff0000),
    (0x0000ffff, 0x0000ffff, 0),
    (0x00000003, 0x00000005, 0x000000000000000f),
)


def signed32(x):
    return x - (1 << 32) if x >> 31 else x


def reference(signed, a, b, c):
    """(result, carry) from plain Python integers: the exact product plus the 64-bit addend."""
    product = (signed32(a) * signed32(b) if signed else a * b) & M64
    total = product + c
    return total & M64, int(total > M64)


class Evaluator:
    """Runs the emitted mulhi32/mad64u/mad64i helpers with the 32-bit evaluator of the bit-ALU check."""

    def __init__(self, text):
        self.helpers = {}
        for name in ('mulhi32', 'mad64u', 'mad64i'):
            m = re.search(r'uint3? %s\((.*?)\) \{ (.*?) \}$' % name, text, re.M)
            assert m, '%s missing from the generated HLSL' % name
            self.helpers[name] = ([p.split()[1] for p in m.group(1).split(',')], m.group(2))
        self.original_call = Expr.call
        Expr.call = staticmethod(self.call)

    def close(self):
        Expr.call = staticmethod(self.original_call)

    def call(self, name, args):
        if name == 'mulhi32':
            return self.run('mulhi32', [a[0] for a in args])[0], False
        return self.original_call(name, args)

    def run(self, name, values):
        params, body = self.helpers[name]
        env = dict(zip(params, values))
        stmts = [s.strip() for s in body.split(';') if s.strip()]
        for stmt in stmts[:-1]:
            for item in re.split(r',\s*(?=\w+ = )', stmt[len('uint '):]):
                var, rhs = [x.strip() for x in item.split('=', 1)]
                env[var] = Expr(rhs).run(env)[0]
        ret = stmts[-1]
        inner = re.fullmatch(r'return (?:uint3\((.*)\)|(.*))', ret)
        if inner.group(2) is not None:
            return [Expr(inner.group(2)).run(env)[0]]
        parts, depth, start = [], 0, 0
        text = inner.group(1)
        for i, ch in enumerate(text):
            depth += ch == '('
            depth -= ch == ')'
            if ch == ',' and depth == 0:
                parts.append(text[start:i])
                start = i + 1
        parts.append(text[start:])
        return [Expr(p).run(env)[0] for p in parts]


def translate(asm):
    probe = Probe()
    probe.masks, probe.varying_s = set(), set()
    probe.one(asm.split(' ', 1)[0], asm)
    return probe


def refused(asm):
    try:
        translate(asm)
    except G.Unsupported:
        return True
    return False


def helper_source():
    """The helpers as a pixel stage emits them (a program that only exports)."""
    return G.Pipeline(vs_words([(0, 0, 0)] * 4), PRE_WORDS + POST_WORDS, root_blob()).hlsl_ps()


def controls():
    ev = Evaluator(helper_source())
    try:
        for signed, name in ((False, 'mad64u'), (True, 'mad64i')):
            bad = None
            for a in VALUES:
                for b in VALUES:
                    for c in ADDENDS:
                        lo, hi, carry = ev.run(name, [a, b, c & MASK, c >> 32])
                        want, want_carry = reference(signed, a, b, c)
                        if (hi << 32 | lo) != want or (not signed and carry != want_carry):
                            bad = bad or (hex(a), hex(b), hex(c), hex(hi << 32 | lo), hex(want), carry)
            check('%s: %d inputs match the 64-bit reference' % (name, len(VALUES) ** 2 * len(ADDENDS)), bad is None)
    finally:
        ev.close()

    u = translate('v_mad_u64_u32 v[12:13], vcc_lo, v18, v19, v[20:21]')
    check('unsigned form writes the pair and vcc',
          u.lines == ['{ uint3 r = mad64u(v18, v19, v20, v21);', 'v12 = r.x;', 'v13 = r.y;', 'vcc = r.z;', '}'])
    s = translate('v_mad_u64_u32 v[12:13], s60, v18, v19, -1')
    check('inline S2 is sign-extended and an SGPR mask is tracked',
          s.lines[0] == '{ uint3 r = mad64u(v18, v19, 4294967295u, 4294967295u);' and 60 in s.masks)
    i = translate('v_mad_i64_i32 v[12:13], null, v18, v19, s[46:47]')
    check('signed form with null writes only the pair',
          i.lines == ['{ uint3 r = mad64i(v18, v19, s46, s47);', 'v12 = r.x;', 'v13 = r.y;', '}'])
    overlap = translate('v_mad_u64_u32 v[20:21], s60, v18, v19, v[20:21]')
    check('a destination overlapping S2 reads S2 before it writes', overlap.lines[0].endswith('v20, v21);'))

    for asm in ('v_mad_u64_u32 v[14:15], vcc_lo, s60, 1, 0',
                'v_mad_u64_u32 v[14:15], vcc_lo, vcc_lo, 1, 0',
                'v_mad_u64_u32 v[14:15], vcc_lo, v18, v19, s[60:61]',
                'v_mad_u64_u32 v[14:15], vcc_lo, v18, v19, s[59:60]'):
        masked = Probe()
        masked.masks, masked.varying_s = {60}, set()
        try:
            masked.one('v_mad_u64_u32', asm)
            ok = False
        except G.Unsupported:
            ok = True
        check('refused lane-mask input: %s' % asm, ok)
    unmasked = Probe()
    unmasked.masks, unmasked.varying_s = {60}, set()
    unmasked.one('v_mad_u64_u32', 'v_mad_u64_u32 v[14:15], vcc_lo, s58, 1, s[56:57]')
    check('a scalar that is not a lane mask is still accepted', bool(unmasked.lines))
    tracked = Probe()
    tracked.masks, tracked.varying_s = set(), set()
    tracked.uniform_v, tracked.scc_uniform = set(), True
    tracked.one('v_mad_u64_u32', 'v_mad_u64_u32 v[12:13], s60, v18, v19, v[20:21]')
    check('an SGPR carry mask is marked varying, not uniform',
          60 in tracked.masks and 60 in tracked.varying_s and not tracked.uniform_source('s60'))
    for asm in ('v_mad_i64_i32 v[12:13], vcc_lo, v18, v19, v[20:21]',
                'v_mad_u64_u32 v[12:13], null, v18, v19, v[20:21]',
                'v_mad_u64_u32 v[12:13], vcc_lo, v18, v19, v[20:21] clamp',
                'v_mad_u64_u32 v12, vcc_lo, v18, v19, v[20:21]',
                'v_mad_u64_u32 v[12:13], vcc_lo, v[18:19], v19, v[20:21]',
                'v_mad_u64_u32 v[12:13], vcc_lo, v18, v19',
                'v_mad_u64_u32 v[12:13], vcc_lo, v18, v19, v20'):
        check('refused: %s' % asm, refused(asm))

    # Discarded carry destinations have no VCC effect, even on forms the translator refuses.
    for name, asm, may in (
            ('v_mad_u64_u32', 'v_mad_u64_u32 v[12:13], vcc_lo, v18, v19, v[20:21]', {'vcc', 'vcc_hi'}),
            ('v_mad_u64_u32', 'v_mad_u64_u32 v[12:13], null, v18, v19, v[20:21]', set()),
            ('v_mad_i64_i32', 'v_mad_i64_i32 v[12:13], null, v18, v19, v[20:21]', set()),
            ('v_add_co_u32', 'v_add_co_u32 v4, vcc_lo, v5, v6', {'vcc', 'vcc_hi'}),
            ('v_cmp_gt_u32_e64', 'v_cmp_gt_u32_e64 vcc_lo, v1, v2', {'vcc', 'vcc_hi'})):
        _, writes, may_write = G.scalar_effects(name, asm)
        check('%s may-writes %s' % (asm, ' and '.join(sorted(may)) or 'no VCC half'),
              (writes | may_write) & G.VCC_KEYS == may and 'vcc_hi' not in writes)
    # A skipped-block join must not retain a stale high-half clear fact.
    start = {('exec', 0), ('exec_hi', 1), ('vcc_hi', 0)}
    mad = 'v_mad_u64_u32 v[12:13], vcc_lo, v18, v19, v[20:21]'
    after = G.clear_after('v_mad_u64_u32', mad, G.scalar_effects('v_mad_u64_u32', mad)[2], start)
    check('a clear vcc_hi fact does not survive the unsigned mad', after == start - {('vcc_hi', 0)})
    null_mad = 'v_mad_u64_u32 v[12:13], null, v18, v19, v[20:21]'
    after_null = G.clear_after('v_mad_u64_u32', null_mad, G.scalar_effects('v_mad_u64_u32', null_mad)[2], start)
    check('a discarded null carry keeps the vcc_hi fact', after_null == start)


def pack_draw(a, b, c):
    return [(a, 0, 0), (b, 0, 0), (c & MASK, 0, 0), (c >> 32, 0, 0)]


def ps_words(signed):
    return PRE_WORDS + BODY_WORDS[signed] + POST_WORDS


def pixel_text(signed, mutate=False):
    text = G.Pipeline(vs_words(pack_draw(0, 0, 0)), ps_words(signed), root_blob()).hlsl_ps()
    if mutate:
        helper = 'mad64i' if signed else 'mad64u'
        # Drop the low-word carry into the high word, in the helper this shader calls.
        mutated = re.sub(r'(uint3 %s\(.*?\) \{.*?)(?:hi = hs \+ c0|\+ \(lo < pl \? 1u : 0u\))' % helper,
                         lambda m: m.group(1) + ('hi = hs + 0u' if not signed else '+ 0u'), text, count=1)
        assert mutated != text, helper
        text = mutated
    return text


def expected_words(signed, a, b, c):
    result, carry = reference(signed, a, b, c)
    return [result & MASK, result >> 32, 0 if signed else (0x3f800000 if carry else 0), 0]


def mutated_words(signed, a, b, c):
    """The same reference with the low-word carry dropped from the high word (the carry-out stays exact)."""
    product = (signed32(a) * signed32(b) if signed else a * b) & M64
    low = (product & MASK) + (c & MASK)
    hi = ((product >> 32) + (c >> 32)) & MASK
    total = product + c
    return [low & MASK, hi, 0 if signed else (0x3f800000 if total > M64 else 0), 0]


def check_binary(llvm_mc):
    for signed in (False, True):
        built = assemble(llvm_mc, BODY_SOURCE[signed])
        if built != BODY_WORDS[signed]:
            raise AssertionError('signed=%s: assembled %s, recorded %s' % (signed, built.hex(), BODY_WORDS[signed].hex()))
    for name, source, words in (('prologue', PRE_SOURCE, PRE_WORDS), ('export', POST_SOURCE, POST_WORDS)):
        if assemble(llvm_mc, source) != words:
            raise AssertionError('%s fixture differs from llvm-mc' % name)
    for draw_inputs in DRAWS:
        chunk = pack_draw(*draw_inputs)
        if assemble(llvm_mc, vs_source(chunk)) != vs_words(chunk):
            raise AssertionError('vertex fixture for %s differs from llvm-mc' % (draw_inputs,))
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for signed in (False, True):
                label = 'i64' if signed else 'u64'
                good_ps = compile_hlsl(compiler, kind, pixel_text(signed), 'ps_mad_%s_%s' % (label, kind),
                                       ps_profile, directory)
                for index, inputs in enumerate(DRAWS):
                    vs = compile_hlsl(compiler, kind,
                                      G.Pipeline(vs_words(pack_draw(*inputs)), ps_words(signed), root_blob()).hlsl_vs(),
                                      'vs_mad_%s_%d_%s' % (label, index, kind), vs_profile, directory)
                    expected = words_text(expected_words(signed, *inputs))
                    run = draw(renderer, adapter, vs, good_ps, expected)
                    if run.returncode:
                        raise AssertionError('%s %s inputs %s\n%s\n%s' % (kind, label, inputs, run.stdout, run.stderr))
                    print('%s %s draw %d: %s' % (kind, label, index, expected), flush=True)
                    if index:
                        continue
                    wrong = words_text(mutated_words(signed, *inputs))
                    bad_ps = compile_hlsl(compiler, kind, pixel_text(signed, True), 'mutated_mad_%s_%s' % (label, kind),
                                          ps_profile, directory)
                    run = draw(renderer, adapter, vs, bad_ps, expected)
                    if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                        raise AssertionError('%s %s mutation did not read %s\n%s\n%s'
                                             % (kind, label, wrong, run.stdout, run.stderr))
                    print('%s %s mutation: fails as expected, reads %s' % (kind, label, wrong), flush=True)


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
