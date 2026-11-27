"""RDNA 2 binary16 FMA, packed FMAC and LDEXP controls and D3D12 readback."""
import os
import re
import struct
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

sys.path[:0] = [os.path.join(os.path.dirname(__file__), '..', 'tools', 'startup'),
                os.path.join(os.path.dirname(__file__), '..', 'tools')]
import gfx2hlsl as G
import f16_packed_model as M
from startup_compute_lds_dual_atomics_check import (PREAMBLE_LINES, SENTINEL, USER7, WORDS,
    check_debug_layer, mubuf_store, root_blob, run_shader, sopp, v_mov_imm, vop2)
from startup_gfx2hlsl_cvt_pk_int16_check import vop3a
from startup_gfx2hlsl_f16_alu_draw_check import NAN_HLSL


def stage():
    s = G.Stage.__new__(G.Stage)
    s.lines, s.sym, s.used_s, s.used_v, s.masks = [], {}, set(), set(), set()
    s.exec_known = True
    s.low16_only_v = set()
    return s


def lower(asm, low=()):
    s = stage()
    s.low16_only_v.update(low)
    s.one(asm.split()[0], asm)
    return s


def controls():
    a, b, c = 0x2b90, 0x38dc, 0x292f
    assert M.half_op('fma', a, b, c) != M.half_op('add', M.half_op('mul', a, b), c)
    for bits in (0, 1, 2, 3, 4, 15, 14, 12):
        code = vop3a(0x34b, 3, 257, 258, 259, op_sel=bits)
        asm = G.decode(code)[0][3]
        s = lower(asm)
        assert 'fma16o(' in s.lines[0]
        assert ('<< 16' in s.lines[0]) == bool(bits & 8)
    assert 3 in lower('v_fma_f16 v3, v1, v2, v0', (3,)).low16_only_v
    assert 3 not in lower('v_fma_f16 v3, v1, v2, v0 op_sel:[0,0,0,1]', (3,)).low16_only_v
    assert lower('v_ldexp_f16 v3, v1, v2', (1, 2)).lines
    for asm in ('v_fma_f16 v3, v1, v2, -v0', 'v_fma_f16 v3, 0x1234, v2, 1.0',
                'v_fma_f16 v3, 2.0, v2, -1.0', 'v_fma_f16 v3, |v1|, v2, v0 clamp',
                'v_pk_fmac_f16 v3, s0, v2', 'v_pk_fmac_f16 v3, 0x12345678, v2',
                'v_pk_fmac_f16 v3, 30, v2', 'v_pk_fmac_f16 v3, -4, v2',
                'v_ldexp_f16_e32 v3, 13, v2', 'v_ldexp_f16_e64 v3, s0, v2',
                'v_ldexp_f16_e64 v3, v1, -1.0'):
        assert lower(asm).lines
    for asm in ('v_fma_f16 v3, v1, v2, v0 mul:2', 'v_fma_f16 v3, v1, v2, v0 op_sel:[2,0,0]',
                'v_pk_fmac_f16 v3, v1, v2 clamp', 'v_pk_fmac_f16 v3, v1, s2',
                'v_ldexp_f16 v3, v1, -v2', 'v_ldexp_f16 v3, v1, v2 mul:2',
                'v_fma_f16 v3, v[1:2], v2, v0'):
        try:
            lower(asm)
        except G.Unsupported:
            pass
        else:
            raise AssertionError('accepted ' + asm)
    print('f16 ops controls passed')


def scaled(a, n):
    if (a & 0x7fff) == 0 or (a & 0x7c00) == 0x7c00:
        return a
    return M.round_exact(Fraction(M.h2f(a)) * Fraction(2) ** n)


def fixture():
    code = bytearray(b''.join(raw for _, raw in PREAMBLE_LINES))
    code += v_mov_imm(4, 0)
    expected = []
    def emit(raw, word):
        nonlocal code
        code += raw + mubuf_store(0x1c, 3, 4, offset=4 * len(expected))
        expected.append(word)
    # Distinct halves, plus an exact fused-rounding counterexample.
    words = (0x3c012b90, 0x400038dc, 0x3555292f)
    for reg, word in enumerate(words, 1):
        code += v_mov_imm(reg, word)
    for bits in (0, 1, 2, 3, 4, 15, 14, 12):
        code += v_mov_imm(3, words[2])  # source C and destination alias
        picked = [M.pick(words[i], (bits >> i) & 1) for i in range(3)]
        value = M.half_op('fma', *picked)
        result = (words[2] & (0xffff if bits & 8 else 0xffff0000)) | (value << 16 if bits & 8 else value)
        if bits & 8:
            assert result != ((words[2] & 0xffff0000) | value)
        emit(vop3a(0x34b, 3, 257, 258, 259, op_sel=bits), result)
    # Explicit destination and source C are separate, with negated C.
    code += v_mov_imm(3, 0xa55a5aa5) + v_mov_imm(5, words[2])
    value = M.half_op('fma', words[0] & 0xffff, words[1] & 0xffff, (words[2] & 0xffff) ^ 0x8000)
    emit(vop3a(0x34b, 3, 257, 258, 261, neg=4), 0xa55a0000 | value)
    code += v_mov_imm(1, 1) + v_mov_imm(2, 0x3c00) + v_mov_imm(5, 0)
    code += v_mov_imm(3, 0xa55a5aa5)
    emit(vop3a(0x34b, 3, 257, 258, 261), 0xa55a0001)
    code += v_mov_imm(1, words[0]) + v_mov_imm(2, words[1])
    # Both packed halves and literal / inline constants.
    for src, encoded, literal in ((words[0], 257, None), (0x42003c00, 255, 0x42003c00), (30, 158, None)):
        dest = 0x40003c00
        code += v_mov_imm(3, dest)
        lo = M.half_op('fma', src & 0xffff, words[1] & 0xffff, dest & 0xffff)
        hi = M.half_op('fma', src >> 16, words[1] >> 16, dest >> 16)
        emit(vop2(0x3c, 3, encoded, 2, lit=literal), lo | (hi << 16))
        assert (lo | hi << 16) != (hi | lo << 16)
    # A zero high inline constant still multiplies infinity, producing NaN.
    code += v_mov_imm(2, 0x7c003c00) + v_mov_imm(3, 0x40003c00)
    emit(vop2(0x3c, 3, 158, 2), 0x7e00 << 16 | M.half_op('fma', 30, 0x3c00, 0x3c00))
    for n in (-30, -15, -1, 0, 1, 15, 30, 100, -100):
        for a in (0x3c01, 0x0001, 0x7bff):
            code += v_mov_imm(1, a) + v_mov_imm(2, n & 0xffff)
            emit(vop2(0x3b, 3, 257, 2), scaled(a, n))
    code += v_mov_imm(1, 0x3c01) + v_mov_imm(2, 0xffff)
    emit(vop3a(0x13b, 3, 257, 258), scaled(0x3c01, -1))
    code += v_mov_imm(2, 1)
    emit(vop2(0x3b, 3, 141, 2), scaled(13, 1))
    code += sopp(1)
    hlsl = G.ComputePipeline(bytes(code), root_blob(), USER7, (1, 1, 1)).hlsl_cs()
    hlsl, stores = re.subn(r'\bStore\(a, v3\);', 'Store(a, nanc(v3));', hlsl)
    assert stores == len(expected)
    return NAN_HLSL + hlsl, expected + [SENTINEL] * (WORDS - len(expected))


controls()
source, reference = fixture()
if sys.argv[1:2] == ['dispatch']:
    out = Path(__file__).resolve().parent.parent / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        result = run_shader('f16-ops', *sys.argv[2:], Path(tmp), source, reference)
        print(result.stdout, end='')
        if result.returncode == 77:
            sys.exit(77)
        check_debug_layer('f16-ops', result.stdout, result.stderr)
        assert result.returncode == 0, result.stderr
