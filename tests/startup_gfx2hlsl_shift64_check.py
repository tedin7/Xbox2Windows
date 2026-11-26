"""64-bit VOP3 vector shifts (v_lshlrev_b64, v_lshrrev_b64, v_ashrrev_i64) in tools/startup/gfx2hlsl.py (Refs #1723).

  (no arguments)                       translation, helper-math, overlap, refusal and mutation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pairs with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per two 64-bit shift inputs and compare the four result words

Each pixel fixture reads two 64-bit values (v[0:1] and v[2:3]) and two shift amounts (v8 and v9) through flat
parameters so the pixel stage cannot constant-fold them. The first shift writes a disjoint destination pair (v[12:13]);
the second writes a partially overlapping destination pair (v[3:4] from v[2:3]), then copies v3..v4 to v14..v15 and
exports v12..v15 to mrt0.

Every result is compared with an independent Python 64-bit integer reference. Each emitted HLSL helper (shl64,
shr64u, shr64i) is parsed and evaluated in 32-bit integer arithmetic across all 64 shift counts plus wrapped counts
above 63, and checked to ensure no 32-bit shift ever receives a count of 32 or more. Per-operation mutations (dropped
cross-word carry for lshl/lshr, logical instead of arithmetic right shift for ashr) must fail with exact expected
mutated words on both CPU evaluation and desktop GPU draws.
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
    Expr,
    MASK,
    POST_SOURCE,
    POST_WORDS,
    PRE_SOURCE,
    PRE_WORDS,
    Probe,
    check,
    refused,
    vs_source,
    vs_words,
    words_text,
)

MASK64 = (1 << 64) - 1
OPS = (
    ('v_lshlrev_b64', 'shl64'),
    ('v_lshrrev_b64', 'shr64u'),
    ('v_ashrrev_i64', 'shr64i'),
)

BODY_SOURCE = {
    op: (
        '%s v[12:13], v8, v[0:1]\n'
        '%s v[3:4], v9, v[2:3]\n'
        'v_mov_b32 v14, v3\n'
        'v_mov_b32 v15, v4\n'
    ) % (op, op)
    for op, _ in OPS
}
BODY_WORDS = {
    'v_lshlrev_b64': bytes.fromhex('0c00ffd6080102000300ffd60905020003031c7e04031e7e'),
    'v_lshrrev_b64': bytes.fromhex('0c0000d708010200030000d70905020003031c7e04031e7e'),
    'v_ashrrev_i64': bytes.fromhex('0c0001d708010200030001d70905020003031c7e04031e7e'),
}

GRID_VALUES = (
    (0x00000000, 0x00000000),
    (0x00000001, 0x00000000),
    (0x80000000, 0x00000000),
    (0x00000000, 0x00000001),
    (0x00000000, 0x80000000),
    (0xffffffff, 0x7fffffff),
    (0xffffffff, 0x80000000),
    (0x89abcdef, 0x01234567),
    (0x76543210, 0xfedcba98),
)
GRID_AMOUNTS = tuple(range(64)) + (64, 65, 95, 96, 127, 0x80000020, 0xffffffc1, MASK)

# Each entry holds two (lo, hi, amount) cases: pair 0 is disjoint (v[12:13]), pair 1 overlaps (v[3:4] <- v[2:3]).
# Draw 0 separates dropped cross-word carry (lshl/lshr), logical-vs-arithmetic fill (ashr) and overlap corruption.
DRAWS = (
    ((0x89abcdef, 0x01234567, 1), (0x76543210, 0x80000001, 31)),
    ((0x89abcdef, 0x01234567, 0), (0x76543210, 0xfedcba98, 32)),
    ((0x89abcdef, 0x01234567, 33), (0xffffffff, 0x80000000, 63)),
    ((0xdeadbeef, 0xcafebabe, 64), (0x00000001, 0x80000001, 0xffffffc1)),
)


def pack_draw(pair):
    (lo0, hi0, amt0), (lo1, hi1, amt1) = pair
    return [(lo0, 0, amt0), (hi0, 0, amt1), (lo1, 0, 0), (hi1, 0, 0)]


def ps_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


def ref_shift64(op, lo, hi, amount):
    """Independent 64-bit integer reference: D = S1 << S0[5:0], S1 >> S0[5:0], or signext(S1) >> S0[5:0]."""
    value, count = ((hi & MASK) << 32) | (lo & MASK), amount & 63
    if op == 'v_lshlrev_b64':
        res = (value << count) & MASK64
    elif op == 'v_lshrrev_b64':
        res = value >> count
    elif op == 'v_ashrrev_i64':
        signed64 = value - (1 << 64) if value >> 63 else value
        res = (signed64 >> count) & MASK64
    else:
        raise ValueError(op)
    return res & MASK, res >> 32


def mut_shift64(op, lo, hi, amount):
    """Expected result of each op's deliberate mutation."""
    sh = amount & 63
    sb = sh & 31
    if op == 'v_lshlrev_b64':
        # Cross-word carry dropped: lo bits shifted past bit 31 do not enter hi.
        return ((lo << sb) & MASK, (hi << sb) & MASK) if sh < 32 else (0, (lo << sb) & MASK)
    if op == 'v_lshrrev_b64':
        # Cross-word carry dropped: hi bits shifted below bit 0 do not enter lo.
        return ((lo >> sb) & MASK, (hi >> sb) & MASK) if sh < 32 else ((hi >> sb) & MASK, 0)
    if op == 'v_ashrrev_i64':
        # Logical right shift substituted for arithmetic right shift.
        return ref_shift64('v_lshrrev_b64', lo, hi, amount)
    raise ValueError(op)


def mutate_hlsl(op, helper, text):
    if op in ('v_lshlrev_b64', 'v_lshrrev_b64'):
        pattern = r'(uint2 %s\([^)]+\) \{[^}]+)\| cy(\))' % helper
        mutated = re.sub(pattern, r'\1| 0u\2', text, count=1)
    elif op == 'v_ashrrev_i64':
        mutated = text.replace('= shr64i(', '= shr64u(')
    else:
        raise ValueError(op)
    assert mutated != text, (op, text)
    return mutated


def translate(op, helper, mutate=False):
    pipe = G.Pipeline(vs_words(pack_draw(DRAWS[0])), ps_words(op), root_blob())
    text = pipe.hlsl_ps()
    if mutate:
        text = mutate_hlsl(op, helper, text)
    return pipe, text


def eval_emitted_helper(helper, ps_text):
    """Parse an emitted uint2 shift64 helper and return a 32-bit evaluator for (lo, hi, sh)."""
    m = re.search(r'uint2 %s\(uint lo, uint hi, uint sh\) \{ (.*?) \}' % helper, ps_text)
    assert m, '%s helper missing from generated HLSL' % helper
    body = m.group(1)
    # Check every shift inside the helper uses only sb, sc, or 31u (never >= 32).
    shifts = re.findall(r'(?:<<|>>)\s*(?:sb|sc|31u)\b', body)
    assert len(shifts) == len(re.findall(r'<<|>>', body)), (helper, body)
    stmts = [part.strip() for part in body.split(';') if part.strip()]
    decls = []
    for stmt in stmts[:-1]:
        if stmt == 'sh &= 63u':
            decls.append(('sh', Expr('sh & 63u')))
            continue
        assert stmt.startswith('uint '), stmt
        for item in stmt[5:].split(','):
            name, rhs = [x.strip() for x in item.split('=', 1)]
            decls.append((name, Expr(rhs)))
    ret = stmts[-1]
    m_ret = re.fullmatch(r'return uint2\((.+)\)', ret)
    assert m_ret, ret
    # Split top-level comma inside uint2(lo_expr, hi_expr) respecting nested parentheses.
    inner, depth, split_at = m_ret.group(1), 0, None
    for idx, ch in enumerate(inner):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            split_at = idx
            break
    assert split_at is not None, inner
    lo_e, hi_e = Expr(inner[:split_at]), Expr(inner[split_at + 1:])

    def run(lo, hi, s):
        env = {'lo': lo & MASK, 'hi': hi & MASK, 'sh': s & MASK}
        for name, expr in decls:
            env[name] = expr.run(env)[0]
        return lo_e.run(env)[0], hi_e.run(env)[0]

    return run


def controls():
    for op, helper in OPS:
        _, text = translate(op, helper)
        expected_lines = [
            '{ uint2 t = %s(v0, v1, v8); v12 = t.x; v13 = t.y; }' % helper,
            '{ uint2 t = %s(v2, v3, v9); v3 = t.x; v4 = t.y; }' % helper,
        ]
        stripped = [line.strip() for line in text.splitlines()]
        idx = stripped.index(expected_lines[0])
        check('%s: disjoint and overlapping calls emitted through packed temp' % op,
              stripped[idx:idx + len(expected_lines)] == expected_lines)

        fn = eval_emitted_helper(helper, text)
        mismatch = next(
            ((lo, hi, amt, fn(lo, hi, amt), ref_shift64(op, lo, hi, amt))
             for lo, hi in GRID_VALUES for amt in GRID_AMOUNTS
             if fn(lo, hi, amt) != ref_shift64(op, lo, hi, amt)),
            None,
        )
        check('%s: emitted helper matches reference on %d cases' % (op, len(GRID_VALUES) * len(GRID_AMOUNTS)),
              mismatch is None)

        # Mutation control on the emitted text for draw 0.
        mut_text = mutate_hlsl(op, helper, text)
        mut_helper = 'shr64u' if op == 'v_ashrrev_i64' else helper
        mut_fn = eval_emitted_helper(mut_helper, mut_text)
        p0, p1 = DRAWS[0]
        ref_words = list(ref_shift64(op, *p0)) + list(ref_shift64(op, *p1))
        mut_words = list(mut_shift64(op, *p0)) + list(mut_shift64(op, *p1))
        eval_mut = list(mut_fn(*p0)) + list(mut_fn(*p1))
        check('%s: draw 0 mutation reads %s (differs from %s)' % (op, words_text(mut_words), words_text(ref_words)),
              mut_words != ref_words and eval_mut == mut_words)

    # Partial-overlap negative control: overwriting v3 before computing v4 from v3 corrupts v4 for every op on draw 0.
    lo1, hi1, amt1 = DRAWS[0][1]
    for op, _ in OPS:
        new_lo, new_hi = ref_shift64(op, lo1, hi1, amt1)
        corrupted_hi = ref_shift64(op, lo1, new_lo, amt1)[1]
        check('%s partial overlap v[3:4] <- v[2:3]: sequential write without temp corrupts v4 (%08x != %08x)'
              % (op, corrupted_hi, new_hi), corrupted_hi != new_hi)

    # Accepted operand forms (VGPR pair, unaligned VGPR pair, _e64 spelling, aligned SGPR pair, inline -16..64,
    # SGPR/literal/m0 shift count) and inline sign-extension check.
    accepted = [
        'v_lshrrev_b64 v[0:1], v2, v[4:5]',
        'v_lshrrev_b64_e64 v[1:2], v0, v[1:2]',
        'v_lshlrev_b64 v[0:1], s2, v[2:3]',
        'v_ashrrev_i64 v[0:1], v2, s[4:5]',
        'v_lshrrev_b64 v[0:1], 0x80000020, v[2:3]',
        'v_lshrrev_b64 v[0:1], m0, v[2:3]',
        'v_lshrrev_b64 v[0:1], 32, -1',
        'v_lshlrev_b64 v[0:1], 1, 64',
        'v_ashrrev_i64 v[0:1], 1, -16',
    ]
    check('accepted operand forms translate', not any(refused(a) for a in accepted))

    p_neg = Probe()
    p_neg.one('v_lshrrev_b64', 'v_lshrrev_b64 v[0:1], 1, -1')
    p_pos = Probe()
    p_pos.one('v_lshrrev_b64', 'v_lshrrev_b64 v[0:1], 1, 1')
    check('64-bit inline constants sign-extend (-1 -> 4294967295u,4294967295u; 1 -> 1u,0u)',
          'shr64u(4294967295u, 4294967295u, 0x00000001u)' in p_neg.lines[0]
          and 'shr64u(1u, 0u, 0x00000001u)' in p_pos.lines[0])

    # exec-masked emission guards both destination registers.
    masked = Probe()
    masked.exec_known = False
    masked.one('v_lshrrev_b64', 'v_lshrrev_b64 v[4:5], v2, v[4:5]')
    check('exec-masked v_lshrrev_b64 guards both destination registers',
          masked.lines == ['if (exec & 1) { uint2 t = shr64u(v4, v5, v2); v4 = t.x; v5 = t.y; }'])

    # Destination registers must be removed from uniform_v when overwritten by a 64-bit shift.
    uni = Probe()
    uni.uniform_v.update((4, 5))
    uni.one('v_lshrrev_b64', 'v_lshrrev_b64 v[4:5], v2, v[4:5]')
    check('64-bit shift invalidates uniform_v for both destination registers',
          4 not in uni.uniform_v and 5 not in uni.uniform_v)

    # Symbolic non-numeric SGPR pair (pointer/descriptor) must stay refused.
    sym_probe = Probe()
    sym_probe.sym[0] = ('ptr', ('vbtab',))
    sym_probe.sym[1] = ('ptrhi', ('vbtab',))
    sym_refused = False
    try:
        sym_probe.one('v_lshrrev_b64', 'v_lshrrev_b64 v[0:1], v2, s[0:1]')
    except G.Unsupported:
        sym_refused = True
    check('pointer-holding SGPR pair refused', sym_refused)

    # Unmodeled modifiers, invalid destinations/sources and wrong operand counts must stay refused.
    refusals = [
        'v_lshrrev_b64 v[0:1], v2, v[4:5] clamp',
        'v_lshlrev_b64 v[0:1], v2, v[4:5] mul:2',
        'v_ashrrev_i64 v[0:1], v2, v[4:5] div:2',
        'v_lshrrev_b64 v[0:1], v2, v[4:5] op_sel:[1,0,0,0]',
        'v_lshrrev_b64 v[0:1], -v2, v[4:5]',
        'v_lshrrev_b64 v[0:1], |v2|, v[4:5]',
        'v_lshrrev_b64 v[0:1], v2, -v[4:5]',
        'v_lshrrev_b64 v[0:1], v2, |v[4:5]|',
        'v_lshrrev_b64 v[0:1], v2, neg(v[4:5])',
        'v_lshrrev_b64 v0, v2, v[4:5]',
        'v_lshrrev_b64 s[0:1], v2, v[4:5]',
        'v_lshrrev_b64 v[0:0], v2, v[4:5]',
        'v_lshrrev_b64 v[0:2], v2, v[4:5]',
        'v_lshrrev_b64 v[2:1], v2, v[4:5]',
        'v_lshrrev_b64 v[255:256], v2, v[4:5]',
        'v_lshrrev_b64 v[0:1], v[2:3], v[4:5]',
        'v_lshrrev_b64 v[0:1], s[2:3], v[4:5]',
        'v_lshrrev_b64 v[0:1], v256, v[4:5]',
        'v_lshrrev_b64 v[0:1], s106, v[4:5]',
        'v_lshrrev_b64 v[0:1], vcc, v[4:5]',
        'v_lshrrev_b64 v[0:1], exec, v[4:5]',
        'v_lshrrev_b64 v[0:1], 1.0, v[4:5]',
        'v_lshrrev_b64 v[0:1], v2, v4',
        'v_lshrrev_b64 v[0:1], v2, s0',
        'v_lshrrev_b64 v[0:1], v2, s[1:2]',
        'v_lshrrev_b64 v[0:1], v2, s[104:106]',
        'v_lshrrev_b64 v[0:1], v2, v[4:6]',
        'v_lshrrev_b64 v[0:1], v2, v[255:256]',
        'v_lshrrev_b64 v[0:1], v2, vcc',
        'v_lshrrev_b64 v[0:1], v2, exec',
        'v_lshrrev_b64 v[0:1], v2, 0x41',
        'v_lshrrev_b64 v[0:1], v2, 65',
        'v_lshrrev_b64 v[0:1], v2, -17',
        'v_lshrrev_b64 v[0:1], v2, 1.0',
        'v_lshrrev_b64 v[0:1]',
        'v_lshrrev_b64 v[0:1], v2',
        'v_lshrrev_b64 v[0:1], v2, v[4:5], v6',
        'v_lshrrev_b64 v[0:1], vcc_lo, v[2:3]',  # a wave mask is not a shift count in the one-lane model
        'v_lshrrev_b64 v[0:1], exec_lo, v[2:3]',
    ]
    check('modifiers, invalid operands and wrong counts refused (%d cases)' % len(refusals),
          all(refused(a) for a in refusals))


def check_binary(llvm_mc):
    for op, _ in OPS:
        built = assemble(llvm_mc, BODY_SOURCE[op])
        if built != BODY_WORDS[op]:
            raise AssertionError('%s: assembled %s, recorded %s' % (op, built.hex(), BODY_WORDS[op].hex()))
    for name, source, words in (('prologue', PRE_SOURCE, PRE_WORDS), ('export', POST_SOURCE, POST_WORDS)):
        if assemble(llvm_mc, source) != words:
            raise AssertionError('%s fixture differs from llvm-mc' % name)
    for pair in DRAWS:
        chunk = pack_draw(pair)
        if assemble(llvm_mc, vs_source(chunk)) != vs_words(chunk):
            raise AssertionError('vertex fixture for %s differs from llvm-mc' % (pair,))
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op, helper in OPS:
                good_ps = compile_hlsl(compiler, kind, translate(op, helper)[1],
                                       'ps_%s_%s' % (op, kind), ps_profile, directory)
                for index, pair in enumerate(DRAWS):
                    chunk = pack_draw(pair)
                    vs = compile_hlsl(compiler, kind,
                                      G.Pipeline(vs_words(chunk), ps_words(op), root_blob()).hlsl_vs(),
                                      'vs_%s_%d_%s' % (op, index, kind), vs_profile, directory)
                    p0, p1 = pair
                    expected = words_text(list(ref_shift64(op, *p0)) + list(ref_shift64(op, *p1)))
                    run = draw(renderer, adapter, vs, good_ps, expected)
                    if run.returncode:
                        raise AssertionError('%s %s inputs %s\n%s\n%s' % (kind, op, pair, run.stdout, run.stderr))
                    print('%s %s draw %d: %s' % (kind, op, index, expected), flush=True)
                    if index:
                        continue
                    wrong = words_text(list(mut_shift64(op, *p0)) + list(mut_shift64(op, *p1)))
                    bad_ps = compile_hlsl(compiler, kind, translate(op, helper, mutate=True)[1],
                                          'mutated_%s_%s' % (op, kind), ps_profile, directory)
                    run = draw(renderer, adapter, vs, bad_ps, expected)
                    if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                        raise AssertionError('%s %s mutation did not read %s\n%s\n%s'
                                             % (kind, op, wrong, run.stdout, run.stderr))
                    print('%s %s mutation: fails as expected, reads %s' % (kind, op, wrong), flush=True)


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
