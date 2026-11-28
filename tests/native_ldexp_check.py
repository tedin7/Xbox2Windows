"""v_ldexp_f32 in the direct compute path (#3013).

  native_ldexp_check.py controls <compile_dxil> <sdk bin dir>
  native_ldexp_check.py hardware <native_store_replay> <sdk bin dir>

The semantics are tools/pixel_aot.py's ldexp32: exact binary32 scaling of S0 by 2^S1, with zero, Inf and NaN passed
unchanged, a denormal S0 normalized rather than flushed, a denormal result rounded to nearest even, S1 a signed
integer clamped to [-512, 512]. clamp saturates with integer tests: NaN and negative values (also -0) give +0,
values from 1.0 give 1.0 and a denormal stays, so nothing depends on a float denormal mode. The VOP3 words were assembled with LLVM 15.0.7 (clang -cc1as,
gfx1013). The prologue has the instructions of native_dpp_quad_check.py with the shift amounts 7 and 12, so
v8 = 128 * lane and s7 = 4096 * group (32 words per lane), and the store words follow native_lane_access_check.py.

Each case sets S0 and S1 by v_mov literals and stores v_ldexp_f32 of S0, of -|S0| and with clamp; every lane of two
groups stores all cases. References that flush denormals, truncate or round half away from zero, or add the exponent
without clamping S1 (so it wraps) must each differ from the expected words.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

CASES = [
    (0x3F800000, 3),            # 1.0 * 2^3
    (0x00000001, 23),           # a denormal input, normalized: 2^-126
    (0x3FC00000, -149),         # 1.5 * 2^-149: half way between 1 and 2 ulp, rounds to even 2 (also clamped)
    (0x3F800000, -150),         # 2^-150: half way between 0 and 1 ulp, rounds to even 0
    (0x7FC00001, 5),            # NaN with a payload passes unchanged
    (0x80000000, 0x7FFFFFFF),   # -0 passes unchanged; clamp gives +0
    (0x3F400000, 1),            # 0.75 * 2 = 1.5, saturated to 1.0
    (0xC0400000, 0x7FFFFFFF),   # -3 * 2^huge: -Inf
]
PROLOGUE = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020, 0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
# v_ldexp_f32 v4, v1, v2 / v_ldexp_f32 v5, -|v1|, v2 / v_ldexp_f32 v6, v1, v2 clamp
OPS = [(0xD7620004, 0x00020501, 4), (0xD7620105, 0x20020501, 5), (0xD7628006, 0x00020501, 6)]
# Refused: neg on S1 (word 1 bit 30), omod mul:2 (word 1 bits 28:27), abs on S1 (word 0 bit 9), op_sel bit 11 and
# a nonzero src2 field (word 1 bits 26:18, here v1).
REFUSED = [('neg on S1', (0xD7620004, 0x40020501)), ('omod', (0xD7620004, 0x08020501)),
           ('abs on S1', (0xD7620204, 0x00020501)), ('op_sel', (0xD7620804, 0x00020501)),
           ('src2 field', (0xD7620004, 0x04060501))]
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064


def program(ops=OPS):
    code = list(PROLOGUE)
    for index, (a, b) in enumerate(CASES):
        code += [0x7E0202FF, a, 0x7E0402FF, b & 0xFFFFFFFF]
        for w0, w1, _ in ops:
            code += [w0, w1]
        for slot, (_, _, vgpr) in enumerate(ops):
            code += [0xE0701000 | (16 * index + 4 * slot), 0x80080008 | vgpr << 8]
    return code + [0xBF810000]


def signed(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def ldexp32(bits, delta, flush=False, rounding='even', clamp_delta=True):
    sign, exp, mant = bits & 0x80000000, (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if exp == 0xFF or (exp == 0 and mant == 0):
        return bits
    if exp == 0:
        if flush:
            return sign
        shift = 23 - (mant.bit_length() - 1)
        mant, e = (mant << shift) & 0x7FFFFF, 1 - shift
    else:
        e = exp
    delta = signed(delta)
    if clamp_delta:
        e += max(-512, min(512, delta))
    else:
        e = signed(e + delta)
    if e >= 255:
        return sign | 0x7F800000
    if e <= 0:
        if flush or e < -23:
            return sign
        full, shift = mant | 0x800000, 1 - e
        half, rest, q = 1 << (shift - 1), full & ((1 << shift) - 1), full >> shift
        if rounding == 'even':
            q += rest > half or (rest == half and q & 1)
        elif rounding == 'away':
            q += rest >= half
        return sign | q
    return sign | (e << 23) | mant


def saturate(bits, float_clamp=False):
    """NaN and negative values (also -0) give +0, values from 1.0 give 1.0, others (also denormals) stay."""
    if float_clamp:  # a float clamp that flushes denormals and keeps -0
        if (bits & 0x7F800000) == 0x7F800000 and bits & 0x7FFFFF:
            return 0
        if (bits & 0x7F800000) == 0:
            return bits & 0x80000000
        value = struct.unpack('<f', struct.pack('<I', bits))[0]
        return struct.unpack('<I', struct.pack('<f', min(max(value, 0.0), 1.0)))[0]
    if (bits & 0x7FFFFFFF) > 0x7F800000 or bits & 0x80000000:
        return 0
    return min(bits, 0x3F800000)


def results(a, b, float_clamp=False, **mutation):
    return [ldexp32(a, b, **mutation), ldexp32(a | 0x80000000, b, **mutation),
            saturate(ldexp32(a, b, **mutation), float_clamp)]


def reference(**mutation):
    expected = [SENTINEL] * OUTPUT_WORDS
    for lane in range(64):
        base = GUARD_WORDS + lane * 32
        for index, (a, b) in enumerate(CASES):
            expected[base + 4 * index:base + 4 * index + 3] = results(a, b, **mutation)
    return expected


def compile_program(cli, sdk, work, root, code, suffix):
    marker = request(work, root, code)
    output = marker.with_suffix(suffix)
    return subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                          capture_output=True, text=True), output


def controls(cli, sdk, work, root):
    result, output = compile_program(cli, sdk, work, root, program(), '.cso')
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    # Integer bit operations only: no float multiply and no DXIL unary float operation (such as Saturate).
    assert 'fmul' not in dump and '@dx.op.unary.f32(' not in dump and 'shl i32' in dump, dump
    for n, (label, words) in enumerate(REFUSED):
        result, _ = compile_program(cli, sdk, work, root, program([words + (4,)]), f'.refused{n}.cso')
        assert result.returncode != 0 and 'v_ldexp_f32 operand modifiers' in result.stdout + result.stderr, (
            label, result.stdout, result.stderr)
    expected = reference()
    for label, mutation in (('flushed denormals', {'flush': True}), ('truncated', {'rounding': 'down'}),
                            ('half away from zero', {'rounding': 'away'}), ('S1 not clamped', {'clamp_delta': False}),
                            ('float clamp', {'float_clamp': True})):
        assert reference(**mutation) != expected, label
    print('compiled three v_ldexp_f32 forms with integer operations only; five modifier forms are refused; the '
          'flushing, truncating, half-away, unclamped-exponent and float-clamp references differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    actual = list(struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes()))
    expected = reference()
    diff = [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(actual, expected)) if x != y]
    assert not diff, diff[:10]
    print('v_ldexp_f32: all 2064 words matched')
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
