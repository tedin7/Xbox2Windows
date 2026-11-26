"""16-bit vector ALU lowering in tools/startup/gfx2hlsl.py, drawn on a real D3D12 adapter (Part of #1723).

  (no arguments)                       translation controls and reference self-checks, pure Python
  binary <llvm-mc>                     reassemble the fixture bodies and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair (FXC ps_5_1 when given, DXC ps_6_0), draw one pixel
                                       per four input sets and compare the four result words with the CPU reference

The pixel fixture is the one of startup_gfx2hlsl_bit_alu_check.py: three flat parameters (v0..v3, v4..v7, v8..v11),
one operation per lane, results in v12..v15 exported to mrt0. The vertex stage places the inputs in the parameters, so
the pixel stage cannot fold them. Every source word carries garbage in its upper half: the 16-bit operations read the
low half only and write a result whose upper half is zero.

The v_pk_{mul,add,fma}_f16 keys (name:half:group) draw four selector pairs per key and isolate one half of the result;
their reference is tests/f16_packed_model.py (exact rationals, NaN/inf/-0/denormal/overflow/tie values), and one
wrong expected word per run must be rejected.

The expected words come from an exact rational model of binary16 arithmetic (Fraction, then one round to binary16),
not from the lowering's expressions.
"""
import math
import re
import struct
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import f16_packed_model as M  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    POST_WORDS, PRE_WORDS, vs_source, vs_words, words_text)

K = 0x4900  # literal operand of v_fmamk_f16 / v_fmaak_f16: 10.0


def h2f(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 0xffff))[0]


def f2h(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def exact(bits):
    return Fraction(h2f(bits))


def round_half(value):
    """Nearest binary16 (ties to even) of an exact rational; overflow gives infinity."""
    try:
        return f2h(float(value))
    except OverflowError:
        return 0xfc00 if value < 0 else 0x7c00


def trunc_clamp(bits, lo, hi):
    value = h2f(bits)
    if math.isnan(value):
        return 0
    return int(max(lo, min(hi, math.trunc(value)))) if not math.isinf(value) else (hi if value > 0 else lo)


def signed16(v):
    v &= 0xffff
    return v - 0x10000 if v & 0x8000 else v


def f_min(a, b):
    return a if exact(a) < exact(b) else b


def f_max(a, b):
    return a if exact(a) > exact(b) else b


# mnemonic -> (source count, VOP1 spelling, reference over (s0, s1, s2) words)
OPS = {
    'v_add_f16': (2, False, lambda a, b, c: round_half(exact(a) + exact(b))),
    'v_sub_f16': (2, False, lambda a, b, c: round_half(exact(a) - exact(b))),
    'v_subrev_f16': (2, False, lambda a, b, c: round_half(exact(b) - exact(a))),
    'v_mul_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(b))),
    'v_min_f16': (2, False, lambda a, b, c: f_min(a & 0xffff, b & 0xffff)),
    'v_max_f16': (2, False, lambda a, b, c: f_max(a & 0xffff, b & 0xffff)),
    'v_fmac_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(b) + exact(c))),
    'v_fmamk_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(K) + exact(b))),
    'v_fmaak_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(b) + exact(K))),
    'v_cvt_f16_i16': (1, True, lambda a, b, c: round_half(Fraction(signed16(a)))),
    'v_cvt_f16_u16': (1, True, lambda a, b, c: round_half(Fraction(a & 0xffff))),
    'v_cvt_i16_f16': (1, True, lambda a, b, c: trunc_clamp(a, -32768, 32767) & 0xffff),
    'v_cvt_u16_f16': (1, True, lambda a, b, c: trunc_clamp(a, 0, 65535)),
}

G_ = {name: f2h(value) for name, value in (
    ('1', 1.0), ('2', 2.0), ('3', 3.0), ('half', 0.5), ('m1', -1.0), ('m2.5', -2.5), ('100', 100.0), ('0.1', 0.1),
    ('1000', 1000.0), ('m3', -3.0), ('7', 7.0), ('2.75', 2.75), ('m2.75', -2.75), ('60000', 60000.0),
    ('m60000', -60000.0), ('3.25', 3.25), ('300', 300.0), ('0', 0.0), ('1.5', 1.5), ('0.3', 0.3))}
HIGH = (0xabcd0000, 0x5a5a0000, 0xdead0000, 0x12340000)


def sets(pairs, accumulators=None):
    """Eight lanes of (s0, s1, s2) with garbage in the upper half of every word."""
    out = []
    for i, (a, b) in enumerate(pairs):
        c = accumulators[i] if accumulators else 0
        out.append((a | HIGH[i % 4], b | HIGH[(i + 1) % 4], c | HIGH[(i + 2) % 4]))
    return out


F = G_
ARITH = [(F['1'], F['2']), (F['3'], F['m1']), (F['0.1'], F['0.3']), (F['100'], F['1000']),
         (F['m2.5'], F['half']), (F['1.5'], F['7']), (F['300'], F['m3']), (F['3.25'], F['2.75'])]
ACC = [F['1'], F['m1'], F['half'], F['100'], F['2'], F['0'], F['3'], F['7']]
# (a, b, c) binary16 triples where rounding the binary32 sum and then binary16 differs from one rounding (#2073).
# Measured on AMD RX 9070 XT and WARP: the earlier f16rn(a*b + c) lowering was one binary16 unit low on each.
FMAC_DOUBLE_ROUNDING = [(0x4a54, 0x1ef4, 0x4e2d), (0x51ea, 0xbf3e, 0x1d5f), (0x2bd5, 0xbe83, 0x4528),
                        (0x3915, 0x387a, 0x4b71)]
INTS = [(0x0003, 0), (0xfffd, 0), (0x8403, 0), (0x7fff, 0), (0x8000, 0), (0, 0), (0xffff, 0), (0x00ff, 0)]
FLOATS = [(F['2.75'], 0), (F['m2.75'], 0), (F['60000'], 0), (F['m60000'], 0), (F['half'], 0), (F['m1'], 0),
          (F['100'], 0), (f2h(0.99951171875), 0)]
DRAW = {}
for _name, (_arity, _vop1, _ref) in OPS.items():
    if _name.startswith('v_cvt_f16'):
        DRAW[_name] = sets(INTS)
    elif _name.startswith('v_cvt_'):
        DRAW[_name] = sets(FLOATS)
    else:
        DRAW[_name] = sets(ARITH, ACC if _name == 'v_fmac_f16' else None)
# Four double-rounding lanes replace the first four fixture lanes of v_fmac_f16 (eight lanes stay two draws).
DRAW['v_fmac_f16'] = [tuple(w | HIGH[(i + k) % 4] for k, w in enumerate(t)) for i, t in
                      enumerate(FMAC_DOUBLE_ROUNDING)] + DRAW['v_fmac_f16'][4:]


# v_pk_{mul,add,fma,min,max}_f16 keys 'name:half:group'. One key draws four lanes, each with its own op_sel / op_sel_hi pair
# (every pair for mul and add, a covering sample of sixteen for fma), then isolates the low or the high half of the
# result with an integer op, so the readback word is a 16-bit value (tagged below like the integer conversions).
PK_OPS = {'v_pk_mul_f16': ('mul', 2), 'v_pk_add_f16': ('add', 2), 'v_pk_fma_f16': ('fma', 3),
          'v_pk_min_f16': ('min', 2), 'v_pk_max_f16': ('max', 2)}


# Group 4 of min/max: source 1 is the literal 0x7bef, read through its low half in both lanes (the only form the translator
# takes: op_sel[1] = 0 and op_sel_hi[1] = 0); source 0 takes every lane selector.
LIT = 0x7bef
LIT_OF = {4: (0x7bef, '0x7bef'), 5: (0x3c00, '1.0')}   # group -> (binary16 word, assembler text): a literal and an inline constant
LIT_PAIRS = [([0, 0], [1, 0]), ([1, 0], [0, 0]), ([0, 0], [0, 0]), ([1, 0], [1, 0])]


def pk_parse(key):
    name, half, group = key.split(':')
    kind, arity = PK_OPS[name]
    pairs = LIT_PAIRS if int(group) >= 4 else M.combos(arity)[int(group) * 4:int(group) * 4 + 4]
    return name, kind, arity, half, pairs


def pk_lane_ok(kind, words, sel, hi, half):
    """A lane whose result half is a NaN must get it from a positive NaN operand: the sign of a NaN made by inf * 0 or
    inf - inf differs between adapters, so such a lane could not be compared word for word."""
    picks = [M.pick(w, (hi if half == 'hi' else sel)[i]) for i, w in enumerate(words)]
    if kind in ('min', 'max'):
        # No signalling NaN source (whether the shader mode quiets it first is not part of what this fixture proves) and not two NaN
        # sources (the payload of the NaN the adapter returns is not fixed).
        return not any(M.is_nan(h) and not h & 0x200 for h in picks) and sum(M.is_nan(h) for h in picks) < 2
    result = M.half_op(kind, *picks)
    if not M.is_nan(result):
        return True
    return any(M.is_nan(h) for h in picks) and not any(M.is_nan(h) and M.sign(h) for h in picks)


def pk_lanes(key):
    name, kind, arity, half, pairs = pk_parse(key)
    seeds = M.sample(sum(map(ord, key)), 400, arity)
    if key[-2] == ':' and int(key[-1]) >= 4:
        seeds = [(w[0], LIT_OF[int(key[-1])][0]) + tuple(w[2:]) for w in seeds]
    lanes = []
    for i in range(8):
        sel, hi = pairs[i % 4]
        word = next(w for w in seeds if pk_lane_ok(kind, w, sel, hi, half) and w not in lanes)
        lanes.append(word)
        seeds = [w for w in seeds if w != word]
    return [tuple(w) + (0,) * (3 - arity) for w in lanes]


PK_KEYS = ['%s:%s:%d' % (n, h, g) for n in PK_OPS for g in range(6 if n in ('v_pk_min_f16', 'v_pk_max_f16') else 4) for h in ('lo', 'hi')]
for _key in PK_KEYS:
    DRAW[_key] = pk_lanes(_key)


def pk_reference(key, x, lane):
    name, kind, arity, half, pairs = pk_parse(key)
    sel, hi = pairs[lane % 4]
    word = M.packed(kind, x[:arity], sel, hi)
    return word >> 16 if half == 'hi' else word & 0xffff


def pk_body_source(key):
    name, kind, arity, half, pairs = pk_parse(key)
    lines = []
    for lane, (sel, hi) in enumerate(pairs):
        dst = 12 + lane
        srcs = ', '.join('v%d' % (lane + 4 * i) for i in range(arity))
        if int(key[-1]) >= 4:
            srcs = 'v%d, %s' % (lane, LIT_OF[int(key[-1])][1])
        lines.append('%s v%d, %s op_sel:[%s] op_sel_hi:[%s]\n' % (name, dst, srcs, ','.join(map(str, sel)), ','.join(map(str, hi))))
        lines.append('v_and_b32 v%d, 0xffff, v%d\n' % (dst, dst) if half == 'lo' else 'v_lshrrev_b32 v%d, 16, v%d\n' % (dst, dst))
    return ''.join(lines)


def body_source(op):
    if op in PK_KEYS:
        return pk_body_source(op)
    arity, vop1, _ = OPS[op]
    lines = []
    for lane in range(4):
        dst, a, b = 12 + lane, lane, 4 + lane
        if op == 'v_fmac_f16':
            lines.append('v_mov_b32 v%d, v%d\n' % (dst, 8 + lane))
        if op == 'v_fmamk_f16':
            lines.append('%s v%d, v%d, 0x%x, v%d\n' % (op, dst, a, K, b))
        elif op == 'v_fmaak_f16':
            lines.append('%s v%d, v%d, v%d, 0x%x\n' % (op, dst, a, b, K))
        elif vop1:
            lines.append('%s%s v%d, v%d\n' % (op, '_e64' if lane % 2 else '', dst, a))
        else:
            lines.append('%s v%d, v%d, v%d\n' % (op, dst, a, b))
    return ''.join(lines)


# Recorded llvm-mc (gfx1013) encodings of body_source(op); `binary <llvm-mc>` regenerates and compares them.
BODY_WORDS = {
    'v_add_f16': bytes.fromhex(
        '00091864010b1a64020d1c64030f1e64'),
    'v_sub_f16': bytes.fromhex(
        '00091866010b1a66020d1c66030f1e66'),
    'v_subrev_f16': bytes.fromhex(
        '00091868010b1a68020d1c68030f1e68'),
    'v_mul_f16': bytes.fromhex(
        '0009186a010b1a6a020d1c6a030f1e6a'),
    'v_min_f16': bytes.fromhex(
        '00091874010b1a74020d1c74030f1e74'),
    'v_max_f16': bytes.fromhex(
        '00091872010b1a72020d1c72030f1e72'),
    'v_fmac_f16': bytes.fromhex(
        '0803187e0009186c09031a7e010b1a6c0a031c7e020d1c6c0b031e7e030f1e6c'),
    'v_fmamk_f16': bytes.fromhex(
        '0009186e00490000010b1a6e00490000020d1c6e00490000030f1e6e00490000'),
    'v_fmaak_f16': bytes.fromhex(
        '0009187000490000010b1a7000490000020d1c7000490000030f1e7000490000'),
    'v_cvt_f16_i16': bytes.fromhex(
        '00a3187e0d00d1d50101000002a31c7e0f00d1d503010000'),
    'v_cvt_f16_u16': bytes.fromhex(
        '00a1187e0d00d0d50101000002a11c7e0f00d0d503010000'),
    'v_cvt_i16_f16': bytes.fromhex(
        '00a7187e0d00d3d50101000002a71c7e0f00d3d503010000'),
    'v_cvt_u16_f16': bytes.fromhex(
        '00a5187e0d00d2d50101000002a51c7e0f00d2d503010000'),
}


# Recorded llvm-mc (gfx1013) encodings of the v_pk_*_f16 fixtures (see pk_body_source).
BODY_WORDS.update({
    'v_pk_mul_f16:lo:0': bytes.fromhex(
        '0c4010cc00090200ff181836ffff00000d4010cc010b0208ff1a1a36ffff00000e4010cc020d0210ff1c1c36ffff00000f4010cc030f0218ff1e1e36ffff0000'),
    'v_pk_mul_f16:hi:0': bytes.fromhex(
        '0c4010cc000902009018182c0d4010cc010b0208901a1a2c0e4010cc020d0210901c1c2c0f4010cc030f0218901e1e2c'),
    'v_pk_mul_f16:lo:1': bytes.fromhex(
        '0c4810cc00090200ff181836ffff00000d4810cc010b0208ff1a1a36ffff00000e4810cc020d0210ff1c1c36ffff00000f4810cc030f0218ff1e1e36ffff0000'),
    'v_pk_mul_f16:hi:1': bytes.fromhex(
        '0c4810cc000902009018182c0d4810cc010b0208901a1a2c0e4810cc020d0210901c1c2c0f4810cc030f0218901e1e2c'),
    'v_pk_mul_f16:lo:2': bytes.fromhex(
        '0c5010cc00090200ff181836ffff00000d5010cc010b0208ff1a1a36ffff00000e5010cc020d0210ff1c1c36ffff00000f5010cc030f0218ff1e1e36ffff0000'),
    'v_pk_mul_f16:hi:2': bytes.fromhex(
        '0c5010cc000902009018182c0d5010cc010b0208901a1a2c0e5010cc020d0210901c1c2c0f5010cc030f0218901e1e2c'),
    'v_pk_mul_f16:lo:3': bytes.fromhex(
        '0c5810cc00090200ff181836ffff00000d5810cc010b0208ff1a1a36ffff00000e5810cc020d0210ff1c1c36ffff00000f5810cc030f0218ff1e1e36ffff0000'),
    'v_pk_mul_f16:hi:3': bytes.fromhex(
        '0c5810cc000902009018182c0d5810cc010b0208901a1a2c0e5810cc020d0210901c1c2c0f5810cc030f0218901e1e2c'),
    'v_pk_add_f16:lo:0': bytes.fromhex(
        '0c400fcc00090200ff181836ffff00000d400fcc010b0208ff1a1a36ffff00000e400fcc020d0210ff1c1c36ffff00000f400fcc030f0218ff1e1e36ffff0000'),
    'v_pk_add_f16:hi:0': bytes.fromhex(
        '0c400fcc000902009018182c0d400fcc010b0208901a1a2c0e400fcc020d0210901c1c2c0f400fcc030f0218901e1e2c'),
    'v_pk_add_f16:lo:1': bytes.fromhex(
        '0c480fcc00090200ff181836ffff00000d480fcc010b0208ff1a1a36ffff00000e480fcc020d0210ff1c1c36ffff00000f480fcc030f0218ff1e1e36ffff0000'),
    'v_pk_add_f16:hi:1': bytes.fromhex(
        '0c480fcc000902009018182c0d480fcc010b0208901a1a2c0e480fcc020d0210901c1c2c0f480fcc030f0218901e1e2c'),
    'v_pk_add_f16:lo:2': bytes.fromhex(
        '0c500fcc00090200ff181836ffff00000d500fcc010b0208ff1a1a36ffff00000e500fcc020d0210ff1c1c36ffff00000f500fcc030f0218ff1e1e36ffff0000'),
    'v_pk_add_f16:hi:2': bytes.fromhex(
        '0c500fcc000902009018182c0d500fcc010b0208901a1a2c0e500fcc020d0210901c1c2c0f500fcc030f0218901e1e2c'),
    'v_pk_add_f16:lo:3': bytes.fromhex(
        '0c580fcc00090200ff181836ffff00000d580fcc010b0208ff1a1a36ffff00000e580fcc020d0210ff1c1c36ffff00000f580fcc030f0218ff1e1e36ffff0000'),
    'v_pk_add_f16:hi:3': bytes.fromhex(
        '0c580fcc000902009018182c0d580fcc010b0208901a1a2c0e580fcc020d0210901c1c2c0f580fcc030f0218901e1e2c'),
    'v_pk_fma_f16:lo:0': bytes.fromhex(
        '0c000ecc0009221cff181836ffff00000d080ecc010b2604ff1a1a36ffff00000e500ecc020d2a0cff1c1c36ffff00000f180ecc030f2e14ff1e1e36ffff0000'),
    'v_pk_fma_f16:hi:0': bytes.fromhex(
        '0c000ecc0009221c9018182c0d080ecc010b2604901a1a2c0e500ecc020d2a0c901c1c2c0f180ecc030f2e14901e1e2c'),
    'v_pk_fma_f16:lo:1': bytes.fromhex(
        '0c600ecc0009221cff181836ffff00000d680ecc010b2604ff1a1a36ffff00000e300ecc020d2a0cff1c1c36ffff00000f780ecc030f2e14ff1e1e36ffff0000'),
    'v_pk_fma_f16:hi:1': bytes.fromhex(
        '0c600ecc0009221c9018182c0d680ecc010b2604901a1a2c0e300ecc020d2a0c901c1c2c0f780ecc030f2e14901e1e2c'),
    'v_pk_fma_f16:lo:2': bytes.fromhex(
        '0c080ecc00092204ff181836ffff00000d200ecc010b260cff1a1a36ffff00000e380ecc020d2a14ff1c1c36ffff00000f100ecc030f2e1cff1e1e36ffff0000'),
    'v_pk_fma_f16:hi:2': bytes.fromhex(
        '0c080ecc000922049018182c0d200ecc010b260c901a1a2c0e380ecc020d2a14901c1c2c0f100ecc030f2e1c901e1e2c'),
    'v_pk_fma_f16:lo:3': bytes.fromhex(
        '0c680ecc00092204ff181836ffff00000d400ecc010b260cff1a1a36ffff00000e580ecc020d2a14ff1c1c36ffff00000f700ecc030f2e1cff1e1e36ffff0000'),
    'v_pk_fma_f16:hi:3': bytes.fromhex(
        '0c680ecc000922049018182c0d400ecc010b260c901a1a2c0e580ecc020d2a14901c1c2c0f700ecc030f2e1c901e1e2c'),
})


def ps_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


# Recorded llvm-mc (gfx1013) encodings of the v_pk_min_f16 / v_pk_max_f16 fixtures.
BODY_WORDS.update({
    'v_pk_min_f16:lo:0': bytes.fromhex(
        '0c4011cc00090200ff181836ffff00000d4011cc010b0208ff1a1a36ffff00000e4011cc020d0210ff1c1c36ffff00000f4011cc030f0218ff1e1e36ffff0000'),
    'v_pk_min_f16:hi:0': bytes.fromhex(
        '0c4011cc000902009018182c0d4011cc010b0208901a1a2c0e4011cc020d0210901c1c2c0f4011cc030f0218901e1e2c'),
    'v_pk_min_f16:lo:1': bytes.fromhex(
        '0c4811cc00090200ff181836ffff00000d4811cc010b0208ff1a1a36ffff00000e4811cc020d0210ff1c1c36ffff00000f4811cc030f0218ff1e1e36ffff0000'),
    'v_pk_min_f16:hi:1': bytes.fromhex(
        '0c4811cc000902009018182c0d4811cc010b0208901a1a2c0e4811cc020d0210901c1c2c0f4811cc030f0218901e1e2c'),
    'v_pk_min_f16:lo:2': bytes.fromhex(
        '0c5011cc00090200ff181836ffff00000d5011cc010b0208ff1a1a36ffff00000e5011cc020d0210ff1c1c36ffff00000f5011cc030f0218ff1e1e36ffff0000'),
    'v_pk_min_f16:hi:2': bytes.fromhex(
        '0c5011cc000902009018182c0d5011cc010b0208901a1a2c0e5011cc020d0210901c1c2c0f5011cc030f0218901e1e2c'),
    'v_pk_min_f16:lo:3': bytes.fromhex(
        '0c5811cc00090200ff181836ffff00000d5811cc010b0208ff1a1a36ffff00000e5811cc020d0210ff1c1c36ffff00000f5811cc030f0218ff1e1e36ffff0000'),
    'v_pk_min_f16:hi:3': bytes.fromhex(
        '0c5811cc000902009018182c0d5811cc010b0208901a1a2c0e5811cc020d0210901c1c2c0f5811cc030f0218901e1e2c'),
    'v_pk_max_f16:lo:0': bytes.fromhex(
        '0c4012cc00090200ff181836ffff00000d4012cc010b0208ff1a1a36ffff00000e4012cc020d0210ff1c1c36ffff00000f4012cc030f0218ff1e1e36ffff0000'),
    'v_pk_max_f16:hi:0': bytes.fromhex(
        '0c4012cc000902009018182c0d4012cc010b0208901a1a2c0e4012cc020d0210901c1c2c0f4012cc030f0218901e1e2c'),
    'v_pk_max_f16:lo:1': bytes.fromhex(
        '0c4812cc00090200ff181836ffff00000d4812cc010b0208ff1a1a36ffff00000e4812cc020d0210ff1c1c36ffff00000f4812cc030f0218ff1e1e36ffff0000'),
    'v_pk_max_f16:hi:1': bytes.fromhex(
        '0c4812cc000902009018182c0d4812cc010b0208901a1a2c0e4812cc020d0210901c1c2c0f4812cc030f0218901e1e2c'),
    'v_pk_max_f16:lo:2': bytes.fromhex(
        '0c5012cc00090200ff181836ffff00000d5012cc010b0208ff1a1a36ffff00000e5012cc020d0210ff1c1c36ffff00000f5012cc030f0218ff1e1e36ffff0000'),
    'v_pk_max_f16:hi:2': bytes.fromhex(
        '0c5012cc000902009018182c0d5012cc010b0208901a1a2c0e5012cc020d0210901c1c2c0f5012cc030f0218901e1e2c'),
    'v_pk_max_f16:lo:3': bytes.fromhex(
        '0c5812cc00090200ff181836ffff00000d5812cc010b0208ff1a1a36ffff00000e5812cc020d0210ff1c1c36ffff00000f5812cc030f0218ff1e1e36ffff0000'),
    'v_pk_max_f16:hi:3': bytes.fromhex(
        '0c5812cc000902009018182c0d5812cc010b0208901a1a2c0e5812cc020d0210901c1c2c0f5812cc030f0218901e1e2c'),
    'v_pk_min_f16:lo:4': bytes.fromhex(
        '0c4011cc00ff0108ef7b0000ff181836ffff00000d4811cc01ff0100ef7b0000ff1a1a36ffff00000e4011cc02ff0100ef7b0000ff1c1c36ffff00000f4811cc03ff0108ef7b0000ff1e1e36ffff0000'),
    'v_pk_min_f16:hi:4': bytes.fromhex(
        '0c4011cc00ff0108ef7b00009018182c0d4811cc01ff0100ef7b0000901a1a2c0e4011cc02ff0100ef7b0000901c1c2c0f4811cc03ff0108ef7b0000901e1e2c'),
    'v_pk_max_f16:lo:4': bytes.fromhex(
        '0c4012cc00ff0108ef7b0000ff181836ffff00000d4812cc01ff0100ef7b0000ff1a1a36ffff00000e4012cc02ff0100ef7b0000ff1c1c36ffff00000f4812cc03ff0108ef7b0000ff1e1e36ffff0000'),
    'v_pk_max_f16:hi:4': bytes.fromhex(
        '0c4012cc00ff0108ef7b00009018182c0d4812cc01ff0100ef7b0000901a1a2c0e4012cc02ff0100ef7b0000901c1c2c0f4812cc03ff0108ef7b0000901e1e2c'),
    'v_pk_min_f16:lo:5': bytes.fromhex(
        '0c4011cc00e50108ff181836ffff00000d4811cc01e50100ff1a1a36ffff00000e4011cc02e50100ff1c1c36ffff00000f4811cc03e50108ff1e1e36ffff0000'),
    'v_pk_min_f16:hi:5': bytes.fromhex(
        '0c4011cc00e501089018182c0d4811cc01e50100901a1a2c0e4011cc02e50100901c1c2c0f4811cc03e50108901e1e2c'),
    'v_pk_max_f16:lo:5': bytes.fromhex(
        '0c4012cc00e50108ff181836ffff00000d4812cc01e50100ff1a1a36ffff00000e4012cc02e50100ff1c1c36ffff00000f4812cc03e50108ff1e1e36ffff0000'),
    'v_pk_max_f16:hi:5': bytes.fromhex(
        '0c4012cc00e501089018182c0d4812cc01e50100901a1a2c0e4012cc02e50100901c1c2c0f4812cc03e50108901e1e2c'),
})


# ---- SDWA binary16 arithmetic (v_add/sub/subrev/mul_f16_sdwa) -------------------------------------------------------
# One configuration per draw group: four lanes run the same instruction (src0_sel, src1_sel, dst_sel, dst_unused and
# the abs/neg spelling of the sources) over different words. Lane i reads v<i> and v<4+i>; the old value of the
# destination is the third parameter word, copied into v<12+i> first so that UNUSED_PRESERVE has something to keep.
# Every source word carries a different, non-NaN value in the half that is not selected.
SDWA_BASE = ('WORD_0', 'UNUSED_PAD')
SDWA_CONFIGS = [(op, s0, s1) + SDWA_BASE + ('', '') for op in ('v_add_f16',) for s0 in ('DWORD', 'WORD_0', 'WORD_1')
                for s1 in ('DWORD', 'WORD_0', 'WORD_1')]
SDWA_CONFIGS += [('v_add_f16', 'WORD_1', 'WORD_0', d, u, '', '') for d, u in
                 (('WORD_1', 'UNUSED_PAD'), ('WORD_0', 'UNUSED_PRESERVE'), ('WORD_1', 'UNUSED_PRESERVE'))]
SDWA_CONFIGS += [(op, 'WORD_1', 'WORD_0', 'WORD_1', 'UNUSED_PRESERVE', '', '')
                 for op in ('v_sub_f16', 'v_subrev_f16', 'v_mul_f16')]
SDWA_CONFIGS += [('v_add_f16', 'WORD_0', 'WORD_1', 'WORD_0', 'UNUSED_PRESERVE', '-', '|'),
                 ('v_sub_f16', 'WORD_1', 'WORD_0', 'WORD_1', 'UNUSED_PAD', '-|', ''),
                 ('v_mul_f16', 'WORD_0', 'WORD_0', 'WORD_0', 'UNUSED_PAD', '|', '-')]
# Edge values of binary16 (zeros of both signs, the smallest and largest denormal, the smallest normal, the largest
# finite value, infinities, quiet and signalling NaN, a NaN with payload) and the roundings that matter.
EDGE_PAIRS = [(0x0000, 0x0000), (0x8000, 0x8000), (0x8000, 0x0000), (0x0001, 0x0001), (0x03ff, 0x0001),
              (0x7bff, 0x7bff), (0x0000, 0x7c00), (0xfbff, 0xfbff), (0x7c00, 0x7c00), (0x7c00, 0xfc00),
              (0x7e00, 0x3c00), (0x3c00, 0x7c01), (0x7fff, 0x7fff), (0x3c00, 0x1000), (0x3c01, 0x1000),
              (0x0400, 0x8001)]
SDWA_EDGE_CONFIGS = [(op, 'WORD_1', 'WORD_0', 'WORD_1', 'UNUSED_PRESERVE', '', '')
                     for op in ('v_add_f16', 'v_sub_f16', 'v_subrev_f16', 'v_mul_f16')]
GARBAGE = (0xabcd, 0x5a5a, 0xdead, 0x1234)
OLD = (0x5a5a1234, 0xdead5678, 0x1234abcd, 0x2bcd3456)
NAN_HALF = 0x7e00
# The target is a float format: a result that fits in 16 bits reads back as binary32 zero or a subnormal and FXC drops it.
# Both the drawn word and the expected word are XORed with a tag that keeps every lane a finite normal float (0x3f800000 would turn some 0x4000-class upper halves into infinity or NaN; the control below checks every lane).
SDWA_TAG = 0x60000000


def sdwa_place(value, selection, lane, salt):
    """A register word holding value in the selected half and a different non-NaN value in the other."""
    other = GARBAGE[(lane + salt) % 4]
    return (value << 16) | other if selection == 'WORD_1' else (other << 16) | value


def sdwa_lanes(cfg, pairs):
    _op, s0, s1 = cfg[:3]
    return [(sdwa_place(a, s0, i, 0), sdwa_place(b, s1, i, 1), OLD[i % 4]) for i, (a, b) in enumerate(pairs)]


SDWA_DRAW = {}
for _k, _cfg in enumerate(SDWA_CONFIGS):
    SDWA_DRAW[_cfg] = sdwa_lanes(_cfg, [ARITH[(_i + _k) % 8] for _i in range(4)])
for _cfg in SDWA_EDGE_CONFIGS:
    SDWA_DRAW[_cfg + ('edge',)] = sdwa_lanes(_cfg, EDGE_PAIRS)


def sdwa_name(cfg):
    return '_'.join(p.lower().replace('unused_', '') for p in cfg[:5]) + ''.join(
        '_' + {'-': 'neg', '|': 'abs', '-|': 'negabs', '': 'plain'}[m] for m in cfg[5:7]) + (
        '_edge' if len(cfg) > 7 else '')


def sdwa_operand(modifier, reg):
    return {'': 'v%d', '-': '-v%d', '|': '|v%d|', '-|': '-|v%d|'}[modifier] % reg


def sdwa_body_source(cfg):
    op, s0, s1, dsel, unused, m0, m1 = cfg[:7]
    lines = []
    for lane in range(4):
        dst = 12 + lane
        lines.append('v_mov_b32 v%d, v%d\n' % (dst, 8 + lane))
        lines.append('%s_sdwa v%d, %s, %s dst_sel:%s dst_unused:%s src0_sel:%s src1_sel:%s\n' % (
            op, dst, sdwa_operand(m0, lane), sdwa_operand(m1, 4 + lane), dsel, unused, s0, s1))
    return ''.join(lines)


def half_special(op, a, b):
    """Reference binary16 result word of a op b over words, from exact rationals and the IEEE special cases."""
    def nan(x):
        return (x & 0x7fff) > 0x7c00

    def inf(x):
        return (x & 0x7fff) == 0x7c00
    if op == 'v_subrev_f16':
        op, a, b = 'v_sub_f16', b, a
    if op == 'v_sub_f16':
        op, b = 'v_add_f16', b ^ 0x8000
    if nan(a) or nan(b):
        return NAN_HALF
    sign_a, sign_b = a & 0x8000, b & 0x8000
    if op == 'v_mul_f16':
        if (inf(a) and (b & 0x7fff) == 0) or (inf(b) and (a & 0x7fff) == 0):
            return NAN_HALF
        if inf(a) or inf(b):
            return 0x7c00 | (sign_a ^ sign_b)
        value = exact(a) * exact(b)
        return round_half(value) if value else (sign_a ^ sign_b)
    if inf(a) and inf(b):
        return NAN_HALF if sign_a != sign_b else a
    if inf(a) or inf(b):
        return a if inf(a) else b
    value = exact(a) + exact(b)
    if value == 0:
        return 0x8000 if (a & 0x7fff) == 0 and (b & 0x7fff) == 0 and sign_a and sign_b else 0
    return round_half(value)


def sdwa_half_of(word, selection, modifier):
    half = word >> 16 if selection == 'WORD_1' else word & 0xffff
    if 'abs' in modifier or '|' in modifier:
        half &= 0x7fff
    if modifier.startswith('-'):
        half ^= 0x8000
    return half


def sdwa_reference(cfg, lane, selection_swap=False, keep_swap=False, half_swap=False):
    """Expected register word. The three flags build the deliberately wrong readings of the control mutants."""
    op, s0, s1, dsel, unused, m0, m1 = cfg[:7]
    w0, w1, old = lane
    if selection_swap:
        s0, s1 = ('WORD_0' if sel == 'WORD_1' else 'WORD_1' for sel in (s0, s1))
    result = half_special(op, sdwa_half_of(w0, s0, m0), sdwa_half_of(w1, s1, m1))
    if half_swap:
        dsel = 'WORD_0' if dsel == 'WORD_1' else 'WORD_1'
    if keep_swap:
        unused = 'UNUSED_PAD' if unused == 'UNUSED_PRESERVE' else 'UNUSED_PRESERVE'
    shift = 16 if dsel == 'WORD_1' else 0
    kept = (old & ~(0xffff << shift)) & 0xffffffff if unused == 'UNUSED_PRESERVE' else 0
    return canon_word(kept | (result << shift))


def canon_half(h):
    return NAN_HALF if (h & 0x7fff) > 0x7c00 else h


def canon_word(word):
    return canon_half(word & 0xffff) | (canon_half(word >> 16) << 16)


# A NaN carries no portable sign or payload, so the drawn pair canonicalises every NaN half to 0x7e00 on both sides.
NAN_HLSL = ('uint nanh(uint h) { return (h & 0x7fffu) > 0x7c00u ? 0x7e00u : h; }\n'
            'uint nanc(uint w) { return nanh(w & 0xffffu) | (nanh(w >> 16u) << 16u); }\n')


def sdwa_draw_text(cfg):
    text = G.Pipeline(vs_words([(0, 0, 0)] * 4), sdwa_ps_words(cfg), root_blob()).hlsl_ps()
    return NAN_HLSL + re.sub(r'F\(v(1[2-5])\)', r'F(nanc(v\1) ^ 0x%xu)' % SDWA_TAG, text)


# Recorded llvm-mc (gfx1013) encodings of sdwa_body_source(cfg), by configuration name; `binary <llvm-mc>` compares.
SDWA_BODY_WORDS = {
    'v_add_f16_dword_dword_word_0_pad_plain_plain': bytes.fromhex(
        '0803187ef90818640004060609031a7ef90a1a64010406060a031c7ef90c1c64020406060b031e7ef90e1e6403040606'),
    'v_add_f16_dword_word_0_word_0_pad_plain_plain': bytes.fromhex(
        '0803187ef90818640004060409031a7ef90a1a64010406040a031c7ef90c1c64020406040b031e7ef90e1e6403040604'),
    'v_add_f16_dword_word_1_word_0_pad_plain_plain': bytes.fromhex(
        '0803187ef90818640004060509031a7ef90a1a64010406050a031c7ef90c1c64020406050b031e7ef90e1e6403040605'),
    'v_add_f16_word_0_dword_word_0_pad_plain_plain': bytes.fromhex(
        '0803187ef90818640004040609031a7ef90a1a64010404060a031c7ef90c1c64020404060b031e7ef90e1e6403040406'),
    'v_add_f16_word_0_word_0_word_0_pad_plain_plain': bytes.fromhex(
        '0803187ef90818640004040409031a7ef90a1a64010404040a031c7ef90c1c64020404040b031e7ef90e1e6403040404'),
    'v_add_f16_word_0_word_1_word_0_pad_plain_plain': bytes.fromhex(
        '0803187ef90818640004040509031a7ef90a1a64010404050a031c7ef90c1c64020404050b031e7ef90e1e6403040405'),
    'v_add_f16_word_1_dword_word_0_pad_plain_plain': bytes.fromhex(
        '0803187ef90818640004050609031a7ef90a1a64010405060a031c7ef90c1c64020405060b031e7ef90e1e6403040506'),
    'v_add_f16_word_1_word_0_word_0_pad_plain_plain': bytes.fromhex(
        '0803187ef90818640004050409031a7ef90a1a64010405040a031c7ef90c1c64020405040b031e7ef90e1e6403040504'),
    'v_add_f16_word_1_word_1_word_0_pad_plain_plain': bytes.fromhex(
        '0803187ef90818640004050509031a7ef90a1a64010405050a031c7ef90c1c64020405050b031e7ef90e1e6403040505'),
    'v_add_f16_word_1_word_0_word_1_pad_plain_plain': bytes.fromhex(
        '0803187ef90818640005050409031a7ef90a1a64010505040a031c7ef90c1c64020505040b031e7ef90e1e6403050504'),
    'v_add_f16_word_1_word_0_word_0_preserve_plain_plain': bytes.fromhex(
        '0803187ef90818640014050409031a7ef90a1a64011405040a031c7ef90c1c64021405040b031e7ef90e1e6403140504'),
    'v_add_f16_word_1_word_0_word_1_preserve_plain_plain': bytes.fromhex(
        '0803187ef90818640015050409031a7ef90a1a64011505040a031c7ef90c1c64021505040b031e7ef90e1e6403150504'),
    'v_sub_f16_word_1_word_0_word_1_preserve_plain_plain': bytes.fromhex(
        '0803187ef90818660015050409031a7ef90a1a66011505040a031c7ef90c1c66021505040b031e7ef90e1e6603150504'),
    'v_subrev_f16_word_1_word_0_word_1_preserve_plain_plain': bytes.fromhex(
        '0803187ef90818680015050409031a7ef90a1a68011505040a031c7ef90c1c68021505040b031e7ef90e1e6803150504'),
    'v_mul_f16_word_1_word_0_word_1_preserve_plain_plain': bytes.fromhex(
        '0803187ef908186a0015050409031a7ef90a1a6a011505040a031c7ef90c1c6a021505040b031e7ef90e1e6a03150504'),
    'v_add_f16_word_0_word_1_word_0_preserve_neg_abs': bytes.fromhex(
        '0803187ef90818640014142509031a7ef90a1a64011414250a031c7ef90c1c64021414250b031e7ef90e1e6403141425'),
    'v_sub_f16_word_1_word_0_word_1_pad_negabs_plain': bytes.fromhex(
        '0803187ef90818660005350409031a7ef90a1a66010535040a031c7ef90c1c66020535040b031e7ef90e1e6603053504'),
    'v_mul_f16_word_0_word_0_word_0_pad_abs_neg': bytes.fromhex(
        '0803187ef908186a0004241409031a7ef90a1a6a010424140a031c7ef90c1c6a020424140b031e7ef90e1e6a03042414'),
    'v_add_f16_word_1_word_0_word_1_preserve_plain_plain_edge': bytes.fromhex(
        '0803187ef90818640015050409031a7ef90a1a64011505040a031c7ef90c1c64021505040b031e7ef90e1e6403150504'),
    'v_sub_f16_word_1_word_0_word_1_preserve_plain_plain_edge': bytes.fromhex(
        '0803187ef90818660015050409031a7ef90a1a66011505040a031c7ef90c1c66021505040b031e7ef90e1e6603150504'),
    'v_subrev_f16_word_1_word_0_word_1_preserve_plain_plain_edge': bytes.fromhex(
        '0803187ef90818680015050409031a7ef90a1a68011505040a031c7ef90c1c68021505040b031e7ef90e1e6803150504'),
    'v_mul_f16_word_1_word_0_word_1_preserve_plain_plain_edge': bytes.fromhex(
        '0803187ef908186a0015050409031a7ef90a1a6a011505040a031c7ef90c1c6a021505040b031e7ef90e1e6a03150504'),
}


def sdwa_ps_words(cfg):
    return PRE_WORDS + SDWA_BODY_WORDS[sdwa_name(cfg)] + POST_WORDS


def sdwa_controls(check):
    check('sdwa model: 1 + 2 = 3', half_special('v_add_f16', F['1'], F['2']) == F['3'])
    check('sdwa model: -0 + -0 is -0, -0 + +0 is +0, x - x is +0',
          (half_special('v_add_f16', 0x8000, 0x8000), half_special('v_add_f16', 0x8000, 0),
           half_special('v_sub_f16', F['1'], F['1'])) == (0x8000, 0, 0))
    check('sdwa model: inf - inf, 0 * inf and NaN input give the NaN class',
          (half_special('v_sub_f16', 0x7c00, 0x7c00), half_special('v_mul_f16', 0, 0x7c00),
           half_special('v_add_f16', 0x7c01, F['1'])) == (NAN_HALF,) * 3)
    check('sdwa model: 65504 + 65504 overflows to inf; the smallest denormal survives an add of zero',
          (half_special('v_add_f16', 0x7bff, 0x7bff), half_special('v_add_f16', 0x0001, 0)) == (0x7c00, 0x0001))
    check('sdwa model: ties to even (1 + 2^-11 stays, 1.0009765625 + 2^-11 rounds up)',
          (half_special('v_add_f16', 0x3c00, 0x1000), half_special('v_add_f16', 0x3c01, 0x1000)) == (0x3c00, 0x3c02))
    check('sdwa model: the smallest denormals add and a mixed product underflows to a signed zero',
          (half_special('v_add_f16', 0x0001, 0x0001), half_special('v_mul_f16', 0x0001, 0x8001)) == (0x0002, 0x8000))
    for cfg, lanes in SDWA_DRAW.items():
        name = sdwa_name(cfg)
        check('sdwa %s: four or sixteen lanes in groups of four' % name, len(lanes) % 4 == 0 and len(lanes) >= 4)
        refs = [sdwa_reference(cfg, x) for x in lanes]
        check('sdwa %s: every tagged word is a normal finite binary32' % name,
              all(0 < ((r ^ SDWA_TAG) >> 23) & 0xff < 0xff for r in refs))
        # A wrong lowering must change at least one lane of this configuration (the value-is-wrong control).
        for label, flags in (('the other half of the sources', {'selection_swap': True}),
                             ('the other destination half', {'half_swap': True}),
                             ('preserve and pad exchanged', {'keep_swap': True})):
            detected = any(sdwa_reference(cfg, x, **flags) != r for x, r in zip(lanes, refs))
            check('sdwa %s: control "%s" detects a wrong lowering' % (name, label), detected)
    check('sdwa draw: every configuration has an assembled body',
          all(sdwa_name(cfg) in SDWA_BODY_WORDS for cfg in SDWA_DRAW) or not SDWA_BODY_WORDS)


def reference(op, x, lane=0):
    if op in PK_KEYS:
        return pk_reference(op, x, lane)
    return OPS[op][2](*x) & 0xffffffff


def translate(op):
    return G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words(op), root_blob()).hlsl_ps()


# The target is R32G32B32A32_FLOAT, so a small integer result is a binary32 denormal and FXC drops it when it comes
# from a float to integer conversion. The integer-result operations therefore set bit 30 on the way out of the pixel
# stage and in the expected word; the translated statements themselves are unchanged.
INTEGER_RESULT = ('v_cvt_i16_f16', 'v_cvt_u16_f16', *PK_KEYS)
TAG = 0x40000000


def draw_text(op):
    text = translate(op)
    return re.sub(r'F\(v(1[2-5])\)', r'F(v\1 | 0x%xu)' % TAG, text) if op in INTEGER_RESULT else text


def draw_word(op, x, lane=0):
    return reference(op, x, lane) | (TAG if op in INTEGER_RESULT else 0)


def controls():
    failures = []

    def check(label, ok):
        print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
        if not ok:
            failures.append(label)

    # The model itself, by hand: 0.1 + 0.3 in binary16, saturation, truncation toward zero, NaN reads 0.
    check('model: 1 + 2 = 3', OPS['v_add_f16'][2](F['1'], F['2'], 0) == F['3'])
    check('model: 60000 truncates to the i16 limit', OPS['v_cvt_i16_f16'][2](F['60000'], 0, 0) == 0x7fff)
    check('model: -60000 saturates to -32768', OPS['v_cvt_i16_f16'][2](F['m60000'], 0, 0) == 0x8000)
    check('model: a negative value converts to u16 zero', OPS['v_cvt_u16_f16'][2](F['m1'], 0, 0) == 0)
    check('model: NaN converts to zero', OPS['v_cvt_i16_f16'][2](0x7e00, 0, 0) == 0)
    check('model: 0xfffd is -3 as i16', OPS['v_cvt_f16_i16'][2](0xfffd, 0, 0) == F['m3'])
    for a, b, c, want in ((0x4a54, 0x1ef4, 0x4e2d, 0x4e33), (0x51ea, 0xbf3e, 0x1d5f, 0xd55b),
                          (0x2bd5, 0xbe83, 0x4528, 0x450f), (0x3915, 0x387a, 0x4b71, 0x4b9f)):
        check('model: fmac %04x*%04x+%04x is one rounding' % (a, b, c), OPS['v_fmac_f16'][2](a, b, c) == want)
    check('draw: fmac covers the double-rounding triples',
          all(tuple(w & 0xffff for w in x) in FMAC_DOUBLE_ROUNDING
                                            for x in DRAW['v_fmac_f16'][:4]))
    for op in OPS:
        check('%s: every draw reference has a zero upper half' % op,
              all(reference(op, x) >> 16 == 0 for x in DRAW[op]))
        check('%s: draw sets cover two draws of four lanes' % op, len(DRAW[op]) == 8)
    for key in PK_KEYS:
        check('%s: every draw reference has a zero upper half' % key,
              all(reference(key, x, i) >> 16 == 0 for i, x in enumerate(DRAW[key])))
        check('%s: draw sets cover two draws of four lanes' % key, len(DRAW[key]) == 8)
        check('%s: no lane is a generated NaN' % key,
              all(pk_lane_ok(pk_parse(key)[1], x[:pk_parse(key)[2]], *pk_parse(key)[4][i % 4], pk_parse(key)[3])
                  for i, x in enumerate(DRAW[key])))
    if BODY_WORDS:
        # Non-precise multiplication can lose -0 on RDNA 4 while giving the expected word elsewhere.
        # Check the qualifier and its use even when this host's driver preserves both forms.
        products = (('plain', translate('v_mul_f16'), 4),
                    ('packed', translate('v_pk_mul_f16:lo:0'), 8),
                    ('SDWA', sdwa_draw_text(next(cfg for cfg in SDWA_DRAW if cfg[0] == 'v_mul_f16')), 4))
        for form, text, count in products:
            check('%s binary16 multiply preserves its zero sign' % form,
                  'precise float product = a * b; return product;' in text and text.count('f16rn(mul16(') == count)
        for op in (*OPS, *PK_KEYS):
            try:
                text = translate(op)
            except G.Unsupported as e:
                check('%s: translates (%s)' % (op, e), False)
                continue
            check('%s: translates to four lane results' % op,
                  all(('v%d = ' % r) in text for r in range(12, 16)))
    sdwa_controls(check)
    for cfg in SDWA_DRAW:
        try:
            text = sdwa_draw_text(cfg)
        except G.Unsupported as e:
            check('sdwa %s: translates (%s)' % (sdwa_name(cfg), e), False)
            continue
        check('sdwa %s: the four exports carry the float-target tag' % sdwa_name(cfg),
              text.count('^ 0x%xu' % SDWA_TAG) == 4)
        check('sdwa %s: translates to four lane results' % sdwa_name(cfg),
              all(('v%d = ' % r) in text for r in range(12, 16)))
    if failures:
        raise SystemExit('FAILED: ' + ', '.join(failures))


def check_binary(llvm_mc):
    for op in (*OPS, *PK_KEYS):
        built = assemble(llvm_mc, body_source(op))
        if built != BODY_WORDS[op]:
            raise AssertionError('%s: assembled %s, recorded %s' % (op, built.hex(), BODY_WORDS[op].hex()))
    for op in DRAW:
        for i in range(0, 8, 4):
            if assemble(llvm_mc, vs_source(DRAW[op][i:i + 4])) != vs_words(DRAW[op][i:i + 4]):
                raise AssertionError('vertex fixture for %s differs from llvm-mc' % op)
    for cfg in SDWA_DRAW:
        built = assemble(llvm_mc, sdwa_body_source(cfg))
        if built != SDWA_BODY_WORDS[sdwa_name(cfg)]:
            raise AssertionError('%s: assembled %s, recorded %s' % (sdwa_name(cfg), built.hex(),
                                                                     SDWA_BODY_WORDS[sdwa_name(cfg)].hex()))
        for i in range(0, len(SDWA_DRAW[cfg]), 4):
            if assemble(llvm_mc, vs_source(SDWA_DRAW[cfg][i:i + 4])) != vs_words(SDWA_DRAW[cfg][i:i + 4]):
                raise AssertionError('vertex fixture for %s differs from llvm-mc' % sdwa_name(cfg))
    print('fixture words match llvm-mc')


def draw_shaders(submit, name, lanes, pixel_words, pixel_text):
    pixel = submit('ps', pixel_text, 'ps_' + name)
    vertices = []
    for index in range(0, len(lanes), 4):
        chunk = lanes[index:index + 4]
        source = G.Pipeline(vs_words(chunk), pixel_words, root_blob()).hlsl_vs()
        vertices.append((index, chunk, submit('vs', source, 'vs_%s_%d' % (name, index))))
    ps = pixel.result()
    for index, chunk, vertex in vertices:
        yield index, chunk, vertex.result(), ps


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    failures = []
    # A case group needs one pixel shader and up to two vertex shaders. Only schedule the current group,
    # and consume each compiler result in draw order so that a later failure cannot hide an earlier one.
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=3) as pool:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            stages = {}
            profiles = {'vs': vs_profile, 'ps': ps_profile}

            def submit(stage, source, stem):
                profile = profiles[stage]
                key = profile, source
                if key not in stages:
                    stages[key] = pool.submit(compile_hlsl, compiler, kind, source, stem + '_' + kind, profile, directory)
                return stages[key]

            for op, lanes in DRAW.items():
                shaders = draw_shaders(submit, op.replace(':', '_'), lanes, ps_words(op), draw_text(op))
                for index, chunk, vs, ps in shaders:
                    expected = words_text(draw_word(op, x, index + n) for n, x in enumerate(chunk))
                    run = draw(renderer, adapter, vs, ps, expected)
                    if op == PK_KEYS[0] and index == 0:
                        # A wrong expected word must be reported: the comparison is not vacuous.
                        wrong = words_text([draw_word(op, chunk[0], 0) ^ 1] + [draw_word(op, x, n + 1) for n, x in enumerate(chunk[1:])])
                        if not draw(renderer, adapter, vs, ps, wrong).returncode:
                            failures.append('%s %s wrong-value control accepted' % (kind, op))
                    if run.returncode:
                        failures.append('%s %s set %d' % (kind, op, index // 4))
                        print('%s %s set %d MISMATCH inputs %s\n%s' % (kind, op, index // 4, chunk, run.stderr.strip()),
                              flush=True)
                        continue
                    print('%s %s set %d: %s' % (kind, op, index // 4, expected), flush=True)
            for cfg, lanes in SDWA_DRAW.items():
                name = sdwa_name(cfg)
                shaders = draw_shaders(submit, 'sdwa_' + name, lanes, sdwa_ps_words(cfg), sdwa_draw_text(cfg))
                for index, chunk, vs, ps in shaders:
                    if cfg == SDWA_CONFIGS[0] and index == 0:
                        control_vs, control_ps = vs, ps
                    expected = words_text(sdwa_reference(cfg, x) ^ SDWA_TAG for x in chunk)
                    run = draw(renderer, adapter, vs, ps, expected)
                    if run.returncode:
                        failures.append('%s sdwa %s set %d' % (kind, name, index // 4))
                        print('%s sdwa %s set %d MISMATCH inputs %s\n%s' % (kind, name, index // 4, chunk,
                                                                          run.stderr.strip()), flush=True)
                        continue
                    print('%s sdwa %s set %d: %s' % (kind, name, index // 4, expected), flush=True)
            # Wrong-value control: a drawn word that differs by one bit must be reported as a mismatch.
            cfg = SDWA_CONFIGS[0]
            chunk = SDWA_DRAW[cfg][:4]
            wrong = [sdwa_reference(cfg, x) ^ SDWA_TAG for x in chunk]
            wrong[0] ^= 0x10000
            if not draw(renderer, adapter, control_vs, control_ps, words_text(wrong)).returncode:
                failures.append('%s sdwa wrong-value control was accepted' % kind)
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
