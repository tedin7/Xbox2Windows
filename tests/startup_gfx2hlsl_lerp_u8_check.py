"""Unsigned per-byte average with rounding (v_lerp_u8) in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation, on-demand helper, value-evaluation, refusal and mutation
                                       controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input triples and compare the four result words

AMD RDNA 2 ISA V_LERP_U8 (VOP3A opcode 0x14d):
  D.u[7:0]   = (S0.u[7:0]   + S1.u[7:0]   + S2.u[0])  >> 1
  D.u[15:8]  = (S0.u[15:8]  + S1.u[15:8]  + S2.u[8])  >> 1
  D.u[23:16] = (S0.u[23:16] + S1.u[23:16] + S2.u[16]) >> 1
  D.u[31:24] = (S0.u[31:24] + S1.u[31:24] + S2.u[24]) >> 1
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
    refused,
    vs_words,
    words_text,
)
from startup_gfx2hlsl_sad_check import HlslExpr, vop3a  # noqa: E402

OPCODE = 0x14d
BODY_WORDS = b''.join(vop3a(OPCODE, 12 + lane, 256 + lane, 256 + 4 + lane, 256 + 8 + lane) for lane in range(4))
PS_WORDS = PRE_WORDS + BODY_WORDS + POST_WORDS


def ref_lerp_u8(a, b, c):
    a, b, c = a & MASK, b & MASK, c & MASK
    out = 0
    for shift in (0, 8, 16, 24):
        out |= ((((a >> shift) & 0xff) + ((b >> shift) & 0xff) + ((c >> shift) & 1)) >> 1) << shift
    return out


U8_WORDS = (
    0x00000000, 0x01020304, 0x04030201, 0x7f80ff00, 0x807f00ff,
    0xfefdfcfb, 0x01010101, 0x7fffffff, 0x80000000, 0xffffffff,
)
ROUND_WORDS = (
    0x00000000, 0x01000100, 0x00010001, 0xfe02fe00, 0xff01ff01, 0xffffffff,
)

DRAW_SETS = (
    (
        (0x01020304, 0x04030201, 0xfe02fe00),  # upper S2 byte bits set, all LSBs 0
        (0x01020304, 0x04030201, 0x01010101),  # all S2 byte LSBs 1
        (0x7f80ff00, 0x807f00ff, 0x01000100),  # alternating LSBs + boundary sums
        (0xffffffff, 0xffffffff, 0xffffffff),  # max sum 511 >> 1 == 255 in every byte
    ),
    (
        (0x00000000, 0x01010101, 0xfe00fe00),  # 0+1+0 >> 1 == 0 vs upper bits ignored
        (0x00000000, 0x01010101, 0x01010101),  # 0+1+1 >> 1 == 1
        (0xfefdfcfb, 0x01020304, 0x00010001),  # sum 255 + carry -> 256 >> 1 == 128
        (0x12345678, 0x9abcdef0, 0xab01cd00),  # mixed bytes and S2 upper bits
    ),
)


def extract_helper(hlsl):
    m = re.search(r'uint lerp_u8\(uint a, uint b, uint c\)\s*\{\s*return\s+(.+?);\s*\}', hlsl)
    assert m, 'lerp_u8 helper missing from HLSL'
    return m.group(1)


def controls():
    pipe = G.Pipeline(vs_words(DRAW_SETS[0]), PS_WORDS, root_blob())
    ps = pipe.hlsl_ps()
    check('lerp_u8 helper emitted on demand', 'uint lerp_u8(uint a, uint b, uint c)' in ps)
    for lane in range(4):
        expected_line = f'v{12 + lane} = lerp_u8(v{lane}, v{4 + lane}, v{8 + lane});'
        check(f'lane {lane} assignment emitted', expected_line in ps)

    # On-demand check: a pixel shader without v_lerp_u8 must not emit lerp_u8().
    plain_ps = G.Pipeline(vs_words(DRAW_SETS[0]), PRE_WORDS + POST_WORDS, root_blob()).hlsl_ps()
    check('lerp_u8 omitted when unused', 'lerp_u8(' not in plain_ps)

    # Evaluate the exact emitted helper body over 600 edge triples.
    evaluator = HlslExpr(extract_helper(ps))
    for a in U8_WORDS:
        for b in U8_WORDS:
            for c in ROUND_WORDS:
                got = evaluator.run({'a': a, 'b': b, 'c': c})[0]
                want = ref_lerp_u8(a, b, c)
                assert got == want, (hex(a), hex(b), hex(c), hex(got), hex(want))
    print(f'v_lerp_u8: emitted helper matches reference on {len(U8_WORDS) * len(U8_WORDS) * len(ROUND_WORDS)} cases: ok')

    # Accepted operand forms (VGPR, SGPR, inline constants, literal) and exec-masked emission.
    for asm in (
        'v_lerp_u8 v12, v0, v4, v8',
        'v_lerp_u8 v12, s4, v4, v8',
        'v_lerp_u8 v12, 1, v4, 0',
        'v_lerp_u8 v12, 0x01020304, v4, v8',
    ):
        probe = Probe()
        probe.lower('v_lerp_u8', asm, [o.strip() for o in asm.split(None, 1)[1].split(',')], [])
        assert probe.lines and 'lerp_u8(' in probe.lines[0], (asm, probe.lines)
    probe_masked = Probe()
    probe_masked.exec_known = False
    probe_masked.lower('v_lerp_u8', 'v_lerp_u8 v12, v0, v4, v8', ['v12', 'v0', 'v4', 'v8'], [])
    check('exec-masked v_lerp_u8 guards assignment',
          probe_masked.lines == ['if (exec & 1) v12 = lerp_u8(v0, v4, v8);'])
    print('accepted operand forms translate: ok')

    # Refusals: VOP3A encoding neg/abs/clamp/omod/op_sel, plus wrong operand counts.
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
        must_refuse(lambda ops=bad_ops: Probe().lower('v_lerp_u8', 'v_lerp_u8', ops, []), f'arity {len(bad_ops)}')
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
                want = words_text(ref_lerp_u8(*t) for t in triples)
                run = draw(renderer, adapter, vs_bin, ps_bin, want)
                sys.stdout.write(run.stdout)
                if run.returncode == 77:
                    sys.exit(77)
                assert run.returncode == 0, f'{tag} set {idx} failed:\n{run.stdout}\n{run.stderr}'
                print(f'{tag} v_lerp_u8 draw {idx}: {want}')
                if idx == 0:
                    # Mutation 1: drop `& 1u` so upper S2 byte bits leak into the byte average.
                    mut1 = ps_text.replace('(c & 1u)', '(c & 0xffu)').replace('((c >> 8) & 1u)', '((c >> 8) & 0xffu)')
                    assert mut1 != ps_text
                    mut1_bin = compile_hlsl(compiler, tag, mut1, f'{tag}_ps_mut_mask', ps_profile, directory)
                    run_m1 = draw(renderer, adapter, vs_bin, mut1_bin, want)
                    assert run_m1.returncode != 0, f'{tag} upper-bit mask mutation escaped'
                    print(f'{tag} v_lerp_u8 mask mutation: fails as expected (rc={run_m1.returncode})')
                    # Mutation 2: drop rounding bits (S2 = 0).
                    mut2 = ps_text.replace('lerp_u8(v0, v4, v8)', 'lerp_u8(v0, v4, 0u)').replace(
                        'lerp_u8(v1, v5, v9)', 'lerp_u8(v1, v5, 0u)')
                    assert mut2 != ps_text
                    mut2_bin = compile_hlsl(compiler, tag, mut2, f'{tag}_ps_mut_round', ps_profile, directory)
                    run_m2 = draw(renderer, adapter, vs_bin, mut2_bin, want)
                    assert run_m2.returncode != 0, f'{tag} drop-rounding mutation escaped'
                    print(f'{tag} v_lerp_u8 round mutation: fails as expected (rc={run_m2.returncode})')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        check_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) == 6 else None)
    else:
        raise SystemExit(__doc__)
