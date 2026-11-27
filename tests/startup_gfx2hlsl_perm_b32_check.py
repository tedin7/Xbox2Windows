"""Byte permutation (v_perm_b32) in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation, on-demand helper, exhaustive selector 0..255 evaluation,
                                       refusal and mutation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input triples and compare the four result words

AMD RDNA 2 ISA V_PERM_B32 (VOP3A opcode 0x344):
  Input bytes in[0..3] = S1 bytes 0..3, in[4..7] = S0 bytes 0..3.
  For each output byte i in 0..3, sel = (S2 >> (8*i)) & 0xff:
    0..7   -> in[sel]
    8      -> in[1][7] ? 0xff : 0x00
    9      -> in[3][7] ? 0xff : 0x00
    10     -> in[5][7] ? 0xff : 0x00
    11     -> in[7][7] ? 0xff : 0x00
    12     -> 0x00
    >= 13  -> 0xff
"""
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    MASK,
    POST_WORDS,
    PRE_WORDS,
    Probe,
    check,
    vs_words,
    words_text,
)
from startup_gfx2hlsl_sad_check import HlslExpr, vop3a  # noqa: E402

OPCODE = 0x344
BODY_WORDS = b''.join(vop3a(OPCODE, 12 + lane, 256 + lane, 256 + 4 + lane, 256 + 8 + lane) for lane in range(4))
PS_WORDS = PRE_WORDS + BODY_WORDS + POST_WORDS


def ref_perm_byte(hi, lo, sel):
    sel &= 0xff
    if sel >= 13:
        return 0xff
    if sel == 12:
        return 0x00
    if sel >= 8:
        word = hi if sel >= 10 else lo
        bit = 31 if (sel & 1) else 15
        return 0xff if ((word >> bit) & 1) else 0x00
    word = hi if sel >= 4 else lo
    return (word >> ((sel & 3) * 8)) & 0xff


def ref_perm_b32(hi, lo, sel):
    hi, lo, sel = hi & MASK, lo & MASK, sel & MASK
    out = 0
    for shift in (0, 8, 16, 24):
        out |= ref_perm_byte(hi, lo, (sel >> shift) & 0xff) << shift
    return out


# Pairs that distinguish odd-byte sign bits (bytes 1, 3, 5, 7) from even-byte sign bits (bytes 0, 2, 4, 6).
WORD_PAIRS = (
    (0x81020384, 0x05868708),
    (0x01828304, 0x85060788),
    (0x00800080, 0x80008000),
    (0x80008000, 0x00800080),
    (0x7f7f7f7f, 0x80808080),
    (0x11223344, 0x55667788),
)

DRAW_SETS = (
    (
        (0x11223344, 0x55667788, 0x07040300),  # bytes from S0 and S1
        (0x81020384, 0x05868708, 0x0b0a0908),  # sign-ext selectors 8,9,10,11 (even vs odd sign bits differ)
        (0x01828304, 0x85060788, 0x0b0a0908),  # sign-ext selectors 8,9,10,11 (flipped signs)
        (0x11223344, 0x55667788, 0xfe0d0c05),  # 0xfe (>=13 -> 0xff), 13 (-> 0xff), 12 (-> 0x00), 5 (S0 byte 1)
    ),
    (
        (0xdeadbeef, 0xcafebabe, 0x00010203),  # reverse bytes of S1
        (0xdeadbeef, 0xcafebabe, 0x04050607),  # reverse bytes of S0
        (0x00800080, 0x80008000, 0x08090a0b),  # reverse sign-ext selectors
        (0x01234567, 0x89abcdef, 0x0c0c0d0f),  # constant 0x00 and 0xff selectors
    ),
)


def extract_helper(hlsl, name):
    m = re.search(rf'uint {name}\(uint hi, uint lo, uint sel\)\s*\{{\s*return\s+(.+?);\s*\}}', hlsl)
    assert m, f'{name} helper missing from HLSL'
    return m.group(1)


def controls():
    pipe = G.Pipeline(vs_words(DRAW_SETS[0]), PS_WORDS, root_blob())
    ps = pipe.hlsl_ps()
    check('perm_byte helper emitted on demand', 'uint perm_byte(uint hi, uint lo, uint sel)' in ps)
    check('perm_b32 helper emitted on demand', 'uint perm_b32(uint hi, uint lo, uint sel)' in ps)
    for lane in range(4):
        expected_line = f'v{12 + lane} = perm_b32(v{lane}, v{4 + lane}, v{8 + lane});'
        check(f'lane {lane} assignment emitted', expected_line in ps)

    plain_ps = G.Pipeline(vs_words(DRAW_SETS[0]), PRE_WORDS + POST_WORDS, root_blob()).hlsl_ps()
    check('perm_b32 omitted when unused', 'perm_b32(' not in plain_ps and 'perm_byte(' not in plain_ps)

    # Exhaustively evaluate the emitted perm_byte helper over all 256 selectors x 6 word pairs, plus perm_b32.
    byte_eval = HlslExpr(extract_helper(ps, 'perm_byte'))
    b32_eval = HlslExpr(extract_helper(ps, 'perm_b32'),
                        helpers={'perm_byte': lambda hi, lo, s: byte_eval.run({'hi': hi, 'lo': lo, 'sel': s})[0]})
    for hi, lo in WORD_PAIRS:
        for sel in range(256):
            got = byte_eval.run({'hi': hi, 'lo': lo, 'sel': sel})[0]
            want = ref_perm_byte(hi, lo, sel)
            assert got == want, (hex(hi), hex(lo), sel, hex(got), hex(want))
    for triples in DRAW_SETS:
        for hi, lo, sel in triples:
            got = b32_eval.run({'hi': hi, 'lo': lo, 'sel': sel})[0]
            want = ref_perm_b32(hi, lo, sel)
            assert got == want, (hex(hi), hex(lo), hex(sel), hex(got), hex(want))
    print(f'v_perm_b32: emitted helpers match reference on {len(WORD_PAIRS) * 256} selector cases + draw sets: ok')

    # Accepted operand forms and exec-masked emission.
    for asm in (
        'v_perm_b32 v12, v0, v4, v8',
        'v_perm_b32 v12, s4, v4, v8',
        'v_perm_b32 v12, 1, v4, 0',
        'v_perm_b32 v12, 0x11223344, v4, v8',
    ):
        probe = Probe()
        probe.lower('v_perm_b32', asm, [o.strip() for o in asm.split(None, 1)[1].split(',')], [])
        assert probe.lines and 'perm_b32(' in probe.lines[0], (asm, probe.lines)
    probe_masked = Probe()
    probe_masked.exec_known = False
    probe_masked.lower('v_perm_b32', 'v_perm_b32 v12, v0, v4, v8', ['v12', 'v0', 'v4', 'v8'], [])
    check('exec-masked v_perm_b32 guards assignment',
          probe_masked.lines == ['if (exec & 1) v12 = perm_b32(v0, v4, v8);'])
    print('accepted operand forms translate: ok')

    # Refusals: VOP3A neg/abs on VGPR and inline constants, clamp, omod, op_sel, wrong operand counts.
    def must_refuse(fn, label, needle=''):
        try:
            fn()
        except G.Unsupported as error:
            check(f'{label} refused', needle in str(error) if needle else bool(str(error)))
        else:
            raise AssertionError(f'{label} was accepted')

    for label, bad, needle in (
        ('abs0', vop3a(OPCODE, 12, 256, 260, 264, abs_bits=1), 'with neg/abs'),
        ('neg2', vop3a(OPCODE, 12, 256, 260, 264, neg=4), 'with neg/abs'),
        ('neg_inline0', vop3a(OPCODE, 12, 129, 260, 264, neg=1), 'with neg/abs'),
        ('abs_inline0', vop3a(OPCODE, 12, 129, 260, 264, abs_bits=1), 'with neg/abs'),
        ('clamp', vop3a(OPCODE, 12, 256, 260, 264, clamp=1), ''),
        ('omod', vop3a(OPCODE, 12, 256, 260, 264, omod=1), ''),
        ('op_sel', vop3a(OPCODE, 12, 256, 260, 264, op_sel=1), ''),
    ):
        must_refuse(lambda b=bad: G.Pipeline(vs_words(DRAW_SETS[0]), PRE_WORDS + b + POST_WORDS, root_blob()), label, needle)
    for bad_ops in (['v12', 'v0', 'v4'], ['v12', 'v0', 'v4', 'v8', 'v9']):
        must_refuse(lambda ops=bad_ops: Probe().lower('v_perm_b32', 'v_perm_b32', ops, []), f'arity {len(bad_ops)}')
    print('modifiers and wrong operand counts refused: ok')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = [('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []
    compilers.append(('dxc', dxc, 'vs_6_0', 'ps_6_0'))
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for tag, compiler, vs_profile, ps_profile in compilers:
            for idx, triples in enumerate(DRAW_SETS):
                pipe = G.Pipeline(vs_words(triples), PS_WORDS, root_blob())
                vs_bin = compile_hlsl(compiler, tag, pipe.hlsl_vs(), f'{tag}_vs_{idx}', vs_profile, directory)
                ps_text = pipe.hlsl_ps()
                ps_bin = compile_hlsl(compiler, tag, ps_text, f'{tag}_ps_{idx}', ps_profile, directory)
                want = words_text(ref_perm_b32(*t) for t in triples)
                run = draw(renderer, adapter, vs_bin, ps_bin, want)
                sys.stdout.write(run.stdout)
                if run.returncode == 77:
                    sys.exit(77)
                assert run.returncode == 0, f'{tag} set {idx} failed:\n{run.stdout}\n{run.stderr}'
                print(f'{tag} v_perm_b32 draw {idx}: {want}')
                if idx == 0:
                    # Mutation 1: swap S0 and S1.
                    mut1 = ps_text.replace('perm_b32(v0, v4, v8)', 'perm_b32(v4, v0, v8)')
                    assert mut1 != ps_text
                    mut1_bin = compile_hlsl(compiler, tag, mut1, f'{tag}_ps_mut_swap', ps_profile, directory)
                    run_m1 = draw(renderer, adapter, vs_bin, mut1_bin, want)
                    assert run_m1.returncode != 0, f'{tag} S0/S1 swap mutation escaped'
                    print(f'{tag} v_perm_b32 swap mutation: fails as expected (rc={run_m1.returncode})')
                    # Mutation 2: use byte 0/4 sign bit (bit 7) instead of byte 1/5 sign bit (bit 15).
                    mut2 = ps_text.replace('31u : 15u', '31u : 7u')
                    assert mut2 != ps_text
                    mut2_bin = compile_hlsl(compiler, tag, mut2, f'{tag}_ps_mut_sign', ps_profile, directory)
                    run_m2 = draw(renderer, adapter, vs_bin, mut2_bin, want)
                    assert run_m2.returncode != 0, f'{tag} sign-bit mutation escaped'
                    print(f'{tag} v_perm_b32 sign-bit mutation: fails as expected (rc={run_m2.returncode})')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        check_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
    else:
        raise SystemExit(__doc__)
