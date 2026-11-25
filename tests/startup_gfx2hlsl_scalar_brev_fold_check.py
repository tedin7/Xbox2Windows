"""s_brev_b32 of a constant operand folds to a known word in tools/startup/gfx2hlsl.py (Stage.scalar).

Compilers can build an inline sampler descriptor from scalar constants with s_movk_i32, s_bfm_b32, s_mov_b32 and
s_brev_b32 (s_brev_b32 sN, 1 sets only bit 31) before the sample. The sampler must be four known words; s_mov_b32,
s_movk_i32 and s_bfm_b32 (op_s_bfm_b32) already were, s_brev_b32 was a run-time expression (reversebits).
AMD RDNA 2 ISA: S_BREV_B32 D[31:0] = S0[0:31], no SCC write.

Hand-assembled pixel stage (SOP2 0b10 << 30 | op << 23 | sdst << 16 | ssrc1 << 8 | ssrc0, s_bfm_b32 0x24; SOP1
0xbe800000 | sdst << 16 | op << 8 | ssrc0, s_brev_b32 0x0b; inline integers 128 + n), the decoder reads it back.
Expected values are computed here bit by bit, not with the translator's table. Controls: an operand that is not
known (the user SGPR s5) is not folded and keeps reversebits; a brev of the folded s_bfm_b32 chains; without the
fold table the brev results are not constants (the previous behaviour).
Sampler: the recorded image_3d stage with its fourth sampler word written by s_brev_b32 s27, 0 samples through the
constant sampler 0, 0, 0, 0; without the table the descriptor is refused as partly constant, as before.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import PS_WORDS, VS_WORDS, root_blob  # noqa: E402


def sop2(op, d, s0, s1):
    return struct.pack('<I', 0x80000000 | op << 23 | d << 16 | s1 << 8 | s0)


def sop1(op, d, s0):
    return struct.pack('<I', 0xbe800000 | d << 16 | op << 8 | s0)


def v_mov_s(v, s):
    return struct.pack('<I', 0x7e000000 | v << 17 | 1 << 9 | s)  # v_mov_b32_e32 vN, sM


def program():
    w = sop2(0x24, 0, 128 + 2, 128 + 22)          # s_bfm_b32 s0, 2, 22
    w += sop1(0x0b, 1, 128 + 1)                    # s_brev_b32 s1, 1
    w += sop1(0x0b, 2, 5)                          # s_brev_b32 s2, s5 (s5 not known)
    w += sop1(0x0b, 3, 0)                          # s_brev_b32 s3, s0 (a folded value)
    w += b''.join(v_mov_s(i, i) for i in range(4))
    return w + struct.pack('<III', 0xf800180f, 0x03020100, 0xbf810000)  # exp mrt0 v0, v1, v2, v3 done vm


def reverse(x):
    return sum(((x >> i) & 1) << (31 - i) for i in range(32))


def main():
    asm = [a for _, _, _, a in G.decode(program())]
    assert asm[:4] == ['s_bfm_b32 s0, 2, 22', 's_brev_b32 s1, 1', 's_brev_b32 s2, s5', 's_brev_b32 s3, s0'], asm
    bfm = sum(1 << i for i in range(22, 24))
    want = {0: bfm, 1: reverse(1), 3: reverse(bfm)}
    assert (bfm, want[1], want[3]) == (0x00c00000, 0x80000000, 0x00000300)
    pipe = G.Pipeline(VS_WORDS, program(), root_blob())
    ps = pipe.hlsl_ps()
    for n, value in want.items():
        assert pipe.ps.is_const(n) == value, (n, pipe.ps.sym.get(n))
        assert '    s%d = 0x%08xu;' % (n, value) in ps.split('\n'), ps
    assert pipe.ps.is_const(2) is None and '    s2 = reversebits(s5);' in ps.split('\n'), ps
    saved = dict(G.SCALAR_CONST_FOLD)
    G.SCALAR_CONST_FOLD.clear()
    try:
        old = G.Pipeline(VS_WORDS, program(), root_blob())
        old.hlsl_ps()
        assert old.ps.is_const(0) == bfm and all(old.ps.is_const(n) is None for n in (1, 2, 3)), old.ps.sym
    finally:
        G.SCALAR_CONST_FOLD.update(saved)
    sampler = PS_WORDS.replace(bytes.fromhex('80039bbe'), sop1(0x0b, 27, 128))  # s_mov_b32 s27, 0 -> s_brev_b32
    assert 's_brev_b32 s27, 0' in [a for _, _, _, a in G.decode(sampler)]
    pipe = G.Pipeline(VS_WORDS, sampler, root_blob())
    text = pipe.hlsl_ps()
    assert 'smp 0 0x0 0x0 0x0 0x0' in pipe.manifest().splitlines() and '.Sample(q0,' in text, pipe.manifest()
    G.SCALAR_CONST_FOLD.clear()
    try:
        G.Pipeline(VS_WORDS, sampler, root_blob()).hlsl_ps()
    except G.Unsupported as error:
        assert 'descriptor operand s[24:27]' in str(error), str(error)
    else:
        raise AssertionError('brev-built sampler accepted without the fold table')
    finally:
        G.SCALAR_CONST_FOLD.update(saved)
    print('scalar const fold: s_brev_b32 1 = 0x80000000, brev of the folded s_bfm_b32 2, 22 (0x00c00000) = '
          '0x00000300; controls: unknown operand keeps reversebits, no brev constants without the table; '
          'a sampler word from s_brev_b32 is a constant descriptor (refused without the table)')


if __name__ == '__main__':
    main()
