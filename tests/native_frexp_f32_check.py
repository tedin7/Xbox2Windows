"""Direct compute v_frexp_exp_i32_f32 and v_frexp_mant_f32 (VOP1 and VOP3A)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969

CASES = (
    0x00000000,  # +0.0
    0x80000000,  # -0.0
    0x3F800000,  # +1.0 -> mant=0.5, exp=1
    0xC0400000,  # -3.0 -> mant=-0.75, exp=2
    0x00800000,  # smallest normal 2^-126 -> mant=0.5, exp=-125
    0x00400000,  # largest subnormal 2^-127 -> mant=0.5, exp=-126
    0x80000001,  # smallest negative subnormal -2^-149 -> mant=-0.5, exp=-148
    0x7F800000,  # +Inf -> mant=+Inf (clamped to 1.0), exp=0
)


def v_mov_lit(vdst, lit):
    return [0x7E0002FF | ((vdst & 0xFF) << 17), lit & MASK]


def vop1(op, vdst, src0):
    return 0x7E000000 | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)


def vop3_unary(op, vdst, src0, *, abs0=0, neg0=0, clamp=0, omod=0):
    w0 = 0xD4000000 | ((op & 0x3FF) << 16) | ((clamp & 1) << 15) | ((abs0 & 1) << 8) | (vdst & 0xFF)
    w1 = ((neg0 & 1) << 29) | ((omod & 3) << 27) | (src0 & 0x1FF)
    return [w0, w1]


def program():
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for ci, bits in enumerate(CASES):
        code += v_mov_lit(2, bits)
        # 0: VOP1 v_frexp_exp_i32_f32 v4, v2
        code += [vop1(0x3F, 4, 258), 0xE0701000 | (4 * (ci * 4 + 0)), 0x80080408]
        # 1: VOP1 v_frexp_mant_f32 v4, v2
        code += [vop1(0x40, 4, 258), 0xE0701000 | (4 * (ci * 4 + 1)), 0x80080408]
        # 2: VOP3A v_frexp_exp_i32_f32 v4, -v2 (negated input)
        code += vop3_unary(0x1BF, 4, 258, neg0=1) + [0xE0701000 | (4 * (ci * 4 + 2)), 0x80080408]
        # 3: VOP3A v_frexp_mant_f32 v4, |v2| clamp
        code += vop3_unary(0x1C0, 4, 258, abs0=1, clamp=1) + [0xE0701000 | (4 * (ci * 4 + 3)), 0x80080408]
    return code + [0xBF810000]


def frexp_f32(bits, *, flush_denormals=False, clamp=False):
    sign = bits & 0x80000000
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0xFF or (bits & 0x7FFFFFFF) == 0:
        out_exp, out_mant = 0, bits
    elif exp == 0:
        if flush_denormals:
            out_exp, out_mant = 0, sign
        else:
            steps = 0
            x = mant
            while (x & 0x800000) == 0:
                x <<= 1
                steps += 1
            out_exp = (1 - steps - 126) & MASK
            out_mant = sign | 0x3F000000 | (x & 0x7FFFFF)
    else:
        out_exp = (exp - 126) & MASK
        out_mant = sign | 0x3F000000 | mant
    if clamp:
        abs_m = out_mant & 0x7FFFFFFF
        if abs_m > 0x7F800000 or (out_mant & 0x80000000):
            out_mant = 0
        elif out_mant > 0x3F7FFFFF:
            out_mant = 0x3F800000
    return out_exp, out_mant


def reference(*, flush_denormals=False, no_clamp=False):
    words = [SENTINEL] * 8
    for _lane in range(64):
        for bits in CASES:
            e0, m0 = frexp_f32(bits, flush_denormals=flush_denormals)
            e_neg, _ = frexp_f32(bits ^ 0x80000000, flush_denormals=flush_denormals)
            _, m_abs_clamp = frexp_f32(bits & 0x7FFFFFFF, flush_denormals=flush_denormals, clamp=not no_clamp)
            words.extend((e0, m0, e_neg, m_abs_clamp))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    marker = request(work, root, program())
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                          text=True, check=True).stdout
    assert 'shl i32 ' in dump and 'lshr i32 ' in dump, dump
    for tag, bad, msg in (
        ('exp_clamp', vop3_unary(0x1BF, 4, 258, clamp=1) + [0xBF810000], 'unsupported v_frexp_exp_i32_f32 operand modifiers'),
        ('mant_omod', vop3_unary(0x1C0, 4, 258, omod=1) + [0xBF810000], 'unsupported v_frexp_mant_f32 operand modifiers'),
    ):
        sub = work / tag
        sub.mkdir(exist_ok=True)
        bad_marker = request(sub, root, bad)
        bad_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', bad_marker,
                                  bad_marker.with_suffix('.cso')], capture_output=True, text=True)
        assert bad_res.returncode != 0 and msg in (bad_res.stdout + bad_res.stderr), (tag, bad_res.stdout, bad_res.stderr)
    assert reference() != reference(flush_denormals=True)
    assert reference() != reference(no_clamp=True)
    print('v_frexp_exp_i32_f32/mant_f32 DXIL verified; invalid modifiers refused; denormal and clamp controls differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    expected = reference()
    actual = struct.unpack(f'<{len(expected)}I', output.read_bytes())
    mismatch = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
    if mismatch is not None:
        raise AssertionError(f'word {mismatch}: got {actual[mismatch]:08x}, expected {expected[mismatch]:08x}')
    print('v_frexp_exp_i32_f32/mant_f32: 2048 result words and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_frexp_f32_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
