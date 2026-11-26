"""Lower v_mad_u32_u16 in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the encoded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input sets and compare the four result words

RDNA 2 ISA semantics for v_mad_u32_u16 (no clamp/omod/abs/neg):
  D.u32 = S0.u16 * S1.u16 + S2.u32  (32-bit unsigned wraparound), where op_sel bit 0 / bit 1 takes the high halfword
  [31:16] of S0 / S1 instead of the low one. op_sel bit 2 (S2, a full dword) and bit 3 (the destination) stay refused.

S0 and S1 read only bits [15:0] (allowed on low16-only odd-dmask d16 image_load results), while S2 reads all
32 bits (refused on low16-only registers). Any other modifier (op_sel bits 2 and 3, clamp, omod, abs, neg) or
wrong operand count is refused.
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
from startup_gfx2hlsl_bit_alu_check import (Expr, POST_SOURCE, POST_WORDS, PRE_SOURCE, PRE_WORDS,  # noqa: E402
                                            lane_expressions, lane_results, vs_source, vs_words, words_text)

MASK = 0xffffffff
VOP3_MAD_U32_U16 = 0x373


def ref_mad_u32_u16(s0, s1, s2, sel=(0, 0)):
    a = (s0 >> 16 if sel[0] else s0) & 0xffff
    b = (s1 >> 16 if sel[1] else s1) & 0xffff
    return (a * b + s2) & MASK


def encode_mad_u32_u16(vdst, src0, src1, src2, *, literal=None, op_sel=(0, 0, 0, 0),
                       abs_bits=(0, 0, 0), neg=(0, 0, 0), clamp=0, omod=0):
    """Encode synthetic gfx1013 VOP3A bytes for v_mad_u32_u16 (opcode 0x373)."""
    w0 = (0b110101 << 26) | (VOP3_MAD_U32_U16 << 16) | ((clamp & 1) << 15) | (vdst & 0xff)
    for i, bit in enumerate(op_sel):
        w0 |= (bit & 1) << (11 + i)
    for i, bit in enumerate(abs_bits):
        w0 |= (bit & 1) << (8 + i)
    w1 = ((src0 & 0x1ff) | ((src1 & 0x1ff) << 9) | ((src2 & 0x1ff) << 18) |
          ((omod & 0x3) << 27) | ((neg[0] & 1) << 29) | ((neg[1] & 1) << 30) | ((neg[2] & 1) << 31))
    if 255 in (src0, src1, src2):
        assert literal is not None
        return struct.pack('<3I', w0, w1, literal & MASK)
    assert literal is None
    return struct.pack('<2I', w0, w1)


BODY_WORDS = b''.join(encode_mad_u32_u16(12 + lane, 256 + lane, 256 + 4 + lane, 256 + 8 + lane)
                      for lane in range(4))
PS_WORDS = PRE_WORDS + BODY_WORDS + POST_WORDS
SELECTS = ((1, 0), (0, 1), (1, 1))  # op_sel:[1,0,0], [0,1,0] (the shape the title uses), [1,1,0]
SEL_WORDS = {sel: b''.join(encode_mad_u32_u16(12 + lane, 256 + lane, 256 + 4 + lane, 256 + 8 + lane,
                                              op_sel=(sel[0], sel[1], 0, 0)) for lane in range(4))
             for sel in SELECTS}
SEL_PS_WORDS = {sel: PRE_WORDS + words + POST_WORDS for sel, words in SEL_WORDS.items()}
BODY_SOURCE = ''.join('v_mad_u32_u16 v%d, v%d, v%d, v%d\n' % (12 + lane, lane, 4 + lane, 8 + lane)
                      for lane in range(4))
SEL_SOURCE = {sel: ''.join('v_mad_u32_u16 v%d, v%d, v%d, v%d op_sel:[%d,%d,0]\n'
                           % (12 + lane, lane, 4 + lane, 8 + lane, sel[0], sel[1]) for lane in range(4))
              for sel in SELECTS}

# Synthetic non-register forms decoded through G.decode() and translated via Stage.
SYNTHETIC_CASES = [
    (
        'literal src0 + vgpr src1 + inline 0 src2',
        encode_mad_u32_u16(1, 255, 256 + 15, 128 + 0, literal=0x1f0),
        'v_mad_u32_u16 v1, 0x1f0, v15, 0',
        'v1 = (0x000001f0u & 0xffffu) * (v15 & 0xffffu) + 0x00000000u;',
        lambda v: ref_mad_u32_u16(0x1f0, v['v15'], 0),
    ),
    (
        'sgpr src0 + inline -1 src1 + vgpr src2',
        encode_mad_u32_u16(3, 4, 193, 256 + 9),
        'v_mad_u32_u16 v3, s4, -1, v9',
        'v3 = (s4 & 0xffffu) * (0xffffffffu & 0xffffu) + v9;',
        lambda v: ref_mad_u32_u16(v['s4'], 0xffffffff, v['v9']),
    ),
    (
        'vgpr src0 + inline 7 src1 + literal src2',
        encode_mad_u32_u16(2, 256 + 5, 128 + 7, 255, literal=0x12345678),
        'v_mad_u32_u16 v2, v5, 7, 0x12345678',
        'v2 = (v5 & 0xffffu) * (0x00000007u & 0xffffu) + 0x12345678u;',
        lambda v: ref_mad_u32_u16(v['v5'], 7, 0x12345678),
    ),
]

S01_EDGES = [
    0, 1, 2, 0x7fff, 0x8000, 0xffff,
    0x00010000, 0xdead0000, 0xffff0000,
    0xffff0001, 0xdeadbeef, 0x7fffffff, 0x80000000, 0x8000ffff, MASK,
]
S2_EDGES = [0, 1, 0x12345678, 0x7fffffff, 0x80000000, 0xfffe0000, 0xffff0000, MASK]
GRID = [(s0, s1, s2) for s0 in S01_EDGES for s1 in S01_EDGES for s2 in S2_EDGES]

# 4 input sets per chunk; chunk 0 simultaneously distinguishes all three mutations below.
DRAW_SETS = [
    # Chunk 0: non-zero upper-16 bits on S0, S1, and S2; all distinct from mutations.
    (0xdead0003, 0xbeef0005, 0x12345678),
    (0xffff0007, 0x80000009, 0xabcd0000),
    (0x0001ffff, 0x0002ffff, 0x00020000),
    (0x12340002, 0x56780003, 0x89abcdef),
    # Chunk 1: max product 0xffff * 0xffff = 0xfffe0001, wraparound with 0xffffffff, zero product + upper S2.
    (0x0000ffff, 0x0000ffff, 0x00000000),
    (0x0000ffff, 0x0000ffff, MASK),
    (0xdead0000, 0xbeef0000, 0x76543210),
    (0xffff0003, 0xffff0005, MASK),
    # Chunk 2: boundary bits 0x8000 * 0x8000, 0xffff * 1 + 0xffff0000, wraparound to 0, large addend.
    (0x00008000, 0x00008000, 0x00000000),
    (0xabcdffff, 0x12340001, 0xffff0000),
    (0x00010001, 0x0001ffff, 0xffff0000),
    (0xffffffff, 0xffffffff, 0x0001ffff),
]

MUTATIONS = {
    'drop_s0_mask': (
        r'\((v[0-3]) & 0xffffu\) \* \((v[4-7]) & 0xffffu\) \+ (v(?:[89]|1[01]))',
        r'\1 * (\2 & 0xffffu) + \3',
        lambda s0, s1, s2: (s0 * (s1 & 0xffff) + s2) & MASK,
    ),
    'drop_s1_mask': (
        r'\((v[0-3]) & 0xffffu\) \* \((v[4-7]) & 0xffffu\) \+ (v(?:[89]|1[01]))',
        r'(\1 & 0xffffu) * \2 + \3',
        lambda s0, s1, s2: ((s0 & 0xffff) * s1 + s2) & MASK,
    ),
    'mask_s2': (
        r'\((v[0-3]) & 0xffffu\) \* \((v[4-7]) & 0xffffu\) \+ (v(?:[89]|1[01]))',
        r'(\1 & 0xffffu) * (\2 & 0xffffu) + (\3 & 0xffffu)',
        lambda s0, s1, s2: ((s0 & 0xffff) * (s1 & 0xffff) + (s2 & 0xffff)) & MASK,
    ),
}


class Probe(G.Stage):
    """Minimal Stage for single-instruction / decoded-byte translation probes."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.low16_only_v = set()
        self.pinned_low16_v = set()
        self.whole_read_v = set()


def check(label, ok, detail=''):
    print('%s: %s%s' % (label, 'ok' if ok else 'MISMATCH', '  ' + detail if detail and not ok else ''))
    if not ok:
        raise AssertionError(label)


def translate_ps(mutate=None, sel=None):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), PS_WORDS if sel is None else SEL_PS_WORDS[sel], root_blob())
    text = pipe.hlsl_ps()
    if mutate is not None:
        pattern, replacement, _ = MUTATIONS[mutate]
        mutated, count = re.subn(pattern, replacement, text)
        assert count == 4 and mutated != text, (mutate, count, text)
        text = mutated
    return pipe, text


def ignore_op_sel(text):
    """A broken lowering that drops the halfword select: every `(vN >> 16u)` operand reads the low half instead."""
    mutated, count = re.subn(r'\((v[0-7]) >> 16u\) & 0xffffu', r'\1 & 0xffffu', text)
    assert count >= 4 and mutated != text, count
    return mutated


def decode_one(word_bytes):
    decoded = G.decode(word_bytes)
    assert len(decoded) == 1, decoded
    _, size, name, asm = decoded[0]
    assert size == len(word_bytes), (size, len(word_bytes), decoded)
    return name, asm


def translate_bytes(word_bytes):
    name, asm = decode_one(word_bytes)
    stage = Probe()
    stage.one(name, asm)
    return name, asm, stage


def refused_bytes(word_bytes):
    try:
        translate_bytes(word_bytes)
    except G.Unsupported as exc:
        return str(exc)
    return None


def refused_asm(asm):
    try:
        stage = Probe()
        stage.one(asm.split(' ', 1)[0], asm)
    except G.Unsupported as exc:
        return str(exc)
    return None


def controls():
    # 1. Decode check on the 4-lane pixel body bytes.
    body_decoded = G.decode(BODY_WORDS)
    expected_body = ['v_mad_u32_u16 v%d, v%d, v%d, v%d' % (12 + i, i, 4 + i, 8 + i) for i in range(4)]
    check('synthetic BODY_WORDS decode',
          [asm for _, _, _, asm in body_decoded] == expected_body
          and all(name == 'v_mad_u32_u16' for _, _, name, _ in body_decoded))

    # 2. Value-level verification over all 4 lanes across the 1800-case grid.
    _, ps_text = translate_ps()
    exprs = lane_expressions(ps_text)
    mismatch = None
    for index, (s0, s1, s2) in enumerate(GRID):
        lane = index % 4
        env = {'v%d' % lane: s0, 'v%d' % (4 + lane): s1, 'v%d' % (8 + lane): s2}
        got = exprs[lane].run(env)[0]
        want = ref_mad_u32_u16(s0, s1, s2)
        if got != want:
            mismatch = (lane, (s0, s1, s2), got, want)
            break
    check('pixel lowering: %d 32-bit inputs match reference' % len(GRID), mismatch is None, repr(mismatch))

    # 2b. op_sel: the selected halfword of S0 / S1 reaches the multiply (different high and low halves, a nonzero
    #     32-bit addend), and the unselected operand still reads its low half.
    for sel in SELECTS:
        _, sel_text = translate_ps(sel=sel)
        sel_exprs = lane_expressions(sel_text)
        bad = None
        for index, (s0, s1, s2) in enumerate(GRID):
            lane = index % 4
            env = {'v%d' % lane: s0, 'v%d' % (4 + lane): s1, 'v%d' % (8 + lane): s2}
            if sel_exprs[lane].run(env)[0] != ref_mad_u32_u16(s0, s1, s2, sel):
                bad = (lane, (s0, s1, s2))
                break
        check('op_sel:[%d,%d,0] lowering: %d 32-bit inputs match reference' % (sel[0], sel[1], len(GRID)), bad is None,
              repr(bad))
        broken = lane_expressions(ignore_op_sel(sel_text))
        differs = any(broken[i % 4].run({'v%d' % (i % 4): a, 'v%d' % (4 + i % 4): b, 'v%d' % (8 + i % 4): c})[0]
                      != ref_mad_u32_u16(a, b, c, sel) for i, (a, b, c) in enumerate(GRID))
        check('op_sel:[%d,%d,0]: ignoring the select is caught by the grid' % sel, differs)
        check('op_sel:[%d,%d,0]: the draw chunk 0 separates the halves' % sel,
              [ref_mad_u32_u16(*row, sel) for row in DRAW_SETS[:4]] != [ref_mad_u32_u16(*row) for row in DRAW_SETS[:4]])

    # 3. Synthetic non-register forms: literal, inline integer, SGPR.
    sample_vals = {
        'v5': 0xdead0011, 'v9': 0x87654321, 'v15': 0xbeef0005, 's4': 0xcafe0009,
    }
    for label, word_bytes, want_asm, want_line, ref_fn in SYNTHETIC_CASES:
        name, asm, stage = translate_bytes(word_bytes)
        check('decode %s' % label, name == 'v_mad_u32_u16' and asm == want_asm, asm)
        check('lower %s' % label, stage.lines == [want_line], repr(stage.lines))
        parsed = Expr(stage.lines[0].split(' = ', 1)[1].rstrip(';'))
        for hi_s in (0x00000000, 0xdead0000, 0xffff0000):
            for lo_s in (0x0000, 0x0001, 0xffff):
                env = dict(sample_vals)
                val = hi_s | lo_s
                env['v5'] = val
                env['v15'] = val
                env['s4'] = val
                env['v9'] = (hi_s ^ 0x12345678) | lo_s
                got = parsed.run(env)[0]
                want = ref_fn(env)
                assert got == want, (label, env, got, want)
        check('evaluate %s' % label, True)

    # 4. low16_only contract: S0 and S1 allow odd-dmask d16 low16_only_v; S2 must refuse whole-word read;
    #    writing dst clears low16_only_v on dst.
    stage = Probe()
    stage.low16_only_v.update({4, 5})
    stage.one('v_mad_u32_u16', 'v_mad_u32_u16 v4, v4, v5, v7')
    check('low16_only allowed on S0/S1 and cleared on dst',
          4 not in stage.low16_only_v and 4 not in stage.whole_read_v and 5 not in stage.whole_read_v
          and 7 in stage.whole_read_v)

    stage_s2 = Probe()
    stage_s2.low16_only_v.add(7)
    try:
        stage_s2.one('v_mad_u32_u16', 'v_mad_u32_u16 v6, v4, v5, v7')
    except G.Unsupported as exc:
        s2_reason = str(exc)
    else:
        s2_reason = None
    check('low16_only refused on S2', s2_reason == 'whole-word read of v7 after odd d16 image_load', repr(s2_reason))

    # 5. Refusal controls from synthetic VOP3A bytes and malformed operand counts.
    byte_refusals = [
        ('op_sel:[0,0,1]', encode_mad_u32_u16(1, 256 + 2, 256 + 3, 256 + 4, op_sel=(0, 0, 1, 0)), 'v_mad_u32_u16 with'),
        ('op_sel:[0,0,0,1]', encode_mad_u32_u16(1, 256 + 2, 256 + 3, 256 + 4, op_sel=(0, 0, 0, 1)), 'v_mad_u32_u16 with'),
        ('clamp', encode_mad_u32_u16(1, 256 + 2, 256 + 3, 256 + 4, clamp=1), 'v_mad_u32_u16 with'),
        ('omod mul:2', encode_mad_u32_u16(1, 256 + 2, 256 + 3, 256 + 4, omod=1), 'v_mad_u32_u16 with'),
        ('abs', encode_mad_u32_u16(1, 256 + 2, 256 + 3, 256 + 4, abs_bits=(1, 0, 0)), 'operand'),
        ('neg', encode_mad_u32_u16(1, 256 + 2, 256 + 3, 256 + 4, neg=(1, 0, 0)), 'operand'),
    ]
    for label, word_bytes, needle in byte_refusals:
        reason = refused_bytes(word_bytes)
        check('refused bytes (%s)' % label, reason is not None and needle in reason, repr(reason))

    for bad_asm in ('v_mad_u32_u16 v1, v2, v3', 'v_mad_u32_u16 v1, v2, v3, v4, v5'):
        reason = refused_asm(bad_asm)
        check('refused arity (%s)' % bad_asm,
              reason is not None and 'operands, expected 4' in reason, repr(reason))

    # 6. Mutation controls: each mutated HLSL must disagree on GRID and match its own MUTATIONS reference on DRAW_SETS.
    first_chunk = DRAW_SETS[:4]
    first_ref = [ref_mad_u32_u16(*row) for row in first_chunk]
    for mut_name, (_, _, mut_ref) in MUTATIONS.items():
        _, mut_text = translate_ps(mut_name)
        mut_exprs = lane_expressions(mut_text)
        found = None
        for index, (s0, s1, s2) in enumerate(GRID):
            lane = index % 4
            env = {'v%d' % lane: s0, 'v%d' % (4 + lane): s1, 'v%d' % (8 + lane): s2}
            got = mut_exprs[lane].run(env)[0]
            if got != ref_mad_u32_u16(s0, s1, s2):
                found = (lane, (s0, s1, s2), got)
                break
        check('mutation %s breaks GRID' % mut_name, found is not None)
        all_match_mut = all(
            mut_exprs[i % 4].run({'v%d' % (i % 4): s0, 'v%d' % (4 + i % 4): s1, 'v%d' % (8 + i % 4): s2})[0]
            == mut_ref(s0, s1, s2)
            for i, (s0, s1, s2) in enumerate(GRID)
        )
        check('mutation %s matches its own reference across %d inputs' % (mut_name, len(GRID)), all_match_mut)
        wrong_words = [mut_ref(*row) for row in first_chunk]
        check('mutation %s differs from reference on draw chunk 0' % mut_name, wrong_words != first_ref)
        check('mutation %s evaluator matches draw chunk 0' % mut_name,
              lane_results(mut_exprs, first_chunk) == wrong_words)

    # 7. Unlowered negative control: removing v_mad_u32_u16 from VEC_U must refuse.
    saved = G.VEC_U.pop('v_mad_u32_u16')
    try:
        reason = refused_bytes(BODY_WORDS[:8])
        check('unlowered control refuses v_mad_u32_u16', reason == 'v_mad_u32_u16', repr(reason))
    finally:
        G.VEC_U['v_mad_u32_u16'] = saved


def check_binary(llvm_mc):
    built = assemble(llvm_mc, BODY_SOURCE)
    if built != BODY_WORDS:
        raise AssertionError('body: assembled %s, recorded %s' % (built.hex(), BODY_WORDS.hex()))
    for sel in SELECTS:
        if assemble(llvm_mc, SEL_SOURCE[sel]) != SEL_WORDS[sel]:
            raise AssertionError('op_sel %s body differs from llvm-mc' % (sel,))
    for name, source, words in (('prologue', PRE_SOURCE, PRE_WORDS), ('export', POST_SOURCE, POST_WORDS)):
        if assemble(llvm_mc, source) != words:
            raise AssertionError('%s fixture differs from llvm-mc' % name)
    for _, word_bytes, want_asm, _, _ in SYNTHETIC_CASES:
        built = assemble(llvm_mc, want_asm + '\n')
        if built != word_bytes:
            raise AssertionError('%s: assembled %s, recorded %s' % (want_asm, built.hex(), word_bytes.hex()))
    for i in range(0, len(DRAW_SETS), 4):
        chunk = DRAW_SETS[i:i + 4]
        if assemble(llvm_mc, vs_source(chunk)) != vs_words(chunk):
            raise AssertionError('vertex fixture for %s differs from llvm-mc' % (chunk,))
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            good_ps = compile_hlsl(compiler, kind, translate_ps()[1], 'ps_mad_u32_u16_%s' % kind, ps_profile, directory)
            for index in range(0, len(DRAW_SETS), 4):
                chunk = DRAW_SETS[index:index + 4]
                vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(chunk), PS_WORDS, root_blob()).hlsl_vs(),
                                  'vs_mad_u32_u16_%d_%s' % (index, kind), vs_profile, directory)
                expected = words_text(ref_mad_u32_u16(*x) for x in chunk)
                run = draw(renderer, adapter, vs, good_ps, expected)
                if run.returncode:
                    raise AssertionError('%s chunk %d inputs %s\n%s\n%s'
                                         % (kind, index // 4, chunk, run.stdout, run.stderr))
                print('%s chunk %d %s: %s' % (kind, index // 4,
                                              [tuple('%08x' % v for v in x) for x in chunk], expected), flush=True)
                if index:
                    continue
                for mut_name, (_, _, mut_ref) in MUTATIONS.items():
                    wrong = words_text(mut_ref(*x) for x in chunk)
                    bad_ps = compile_hlsl(compiler, kind, translate_ps(mut_name)[1],
                                          'mutated_%s_%s' % (mut_name, kind), ps_profile, directory)
                    run = draw(renderer, adapter, vs, bad_ps, expected)
                    if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                        raise AssertionError('%s mutation %s did not read %s\n%s\n%s'
                                             % (kind, mut_name, wrong, run.stdout, run.stderr))
                    print('%s mutation %s: fails as expected, reads %s' % (kind, mut_name, wrong), flush=True)
        check_draw_sel(dxc, renderer, adapter, directory)


def check_draw_sel(dxc, renderer, adapter, directory):
    """op_sel draws on DXC ps_6_0: every select reproduces the reference; ignoring the select must fail."""
    for sel in SELECTS:
        good_ps = compile_hlsl(dxc, 'dxc', translate_ps(sel=sel)[1], 'ps_sel_%d%d' % sel, 'ps_6_0', directory)
        bad_ps = compile_hlsl(dxc, 'dxc', ignore_op_sel(translate_ps(sel=sel)[1]), 'bad_sel_%d%d' % sel, 'ps_6_0',
                              directory)
        for index in range(0, len(DRAW_SETS), 4):
            chunk = DRAW_SETS[index:index + 4]
            vs = compile_hlsl(dxc, 'dxc', G.Pipeline(vs_words(chunk), SEL_PS_WORDS[sel], root_blob()).hlsl_vs(),
                              'vs_sel_%d%d_%d' % (sel + (index,)), 'vs_6_0', directory)
            expected = words_text(ref_mad_u32_u16(*x, sel) for x in chunk)
            run = draw(renderer, adapter, vs, good_ps, expected)
            if run.returncode:
                raise AssertionError('op_sel %s chunk %d\n%s\n%s' % (sel, index // 4, run.stdout, run.stderr))
            if index == 0:
                run = draw(renderer, adapter, vs, bad_ps, expected)
                if run.returncode == 0:
                    raise AssertionError('op_sel %s ignored by the lowering was not caught\n%s' % (sel, run.stdout))
        print('op_sel:[%d,%d,0] draws match the reference; the select-ignoring lowering fails' % sel, flush=True)


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
