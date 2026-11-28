"""Direct compute v_frexp_mant_f16 and v_frexp_exp_i16_f16 (VOP1 and VOP3A, high-half preserving)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969
HIGH_SEED = 0xBEEF0000

CASES = (
    0xCAFE0000,  # +0.0h (with high junk in src0)
    0x12348000,  # -0.0h
    0x00003C00,  # +1.0h -> mant=0.5h (0x3800), exp=1
    0x5555C200,  # -3.0h -> mant=-0.75h (0xBA00), exp=2
    0x00000400,  # smallest normal 2^-14 -> mant=0.5h, exp=-13
    0x00000200,  # largest subnormal 2^-15 -> mant=0.5h, exp=-14
    0x00008001,  # smallest negative subnormal -2^-24 -> mant=-0.5h (0xB800), exp=-23
    0x00007C00,  # +Inf -> mant=+Inf (clamped to 1.0h = 0x3C00), exp=0
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
        # 0: VOP1 v_frexp_exp_i16_f16 v4, v2 (pre-seed v4 high half)
        code += v_mov_lit(4, HIGH_SEED | 0x1111) + [vop1(0x5A, 4, 258), 0xE0701000 | (4 * (ci * 4 + 0)), 0x80080408]
        # 1: VOP1 v_frexp_mant_f16 v4, v2
        code += v_mov_lit(4, HIGH_SEED | 0x2222) + [vop1(0x59, 4, 258), 0xE0701000 | (4 * (ci * 4 + 1)), 0x80080408]
        # 2: VOP3A v_frexp_exp_i16_f16 v4, -v2
        code += v_mov_lit(4, HIGH_SEED | 0x3333) + vop3_unary(0x1DA, 4, 258, neg0=1) + [0xE0701000 | (4 * (ci * 4 + 2)), 0x80080408]
        # 3: VOP3A v_frexp_mant_f16 v4, |v2| clamp
        code += v_mov_lit(4, HIGH_SEED | 0x4444) + vop3_unary(0x1D9, 4, 258, abs0=1, clamp=1) + [0xE0701000 | (4 * (ci * 4 + 3)), 0x80080408]
    return code + [0xBF810000]


def frexp_f16(bits, *, flush_denormals=False, clamp=False):
    bits &= 0xFFFF
    sign = bits & 0x8000
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0x1F or (bits & 0x7FFF) == 0:
        out_exp, out_mant = 0, bits
    elif exp == 0:
        if flush_denormals:
            out_exp, out_mant = 0, sign
        else:
            steps = 0
            x = mant
            while (x & 0x400) == 0:
                x <<= 1
                steps += 1
            out_exp = (1 - steps - 14) & 0xFFFF
            out_mant = sign | 0x3800 | (x & 0x3FF)
    else:
        out_exp = (exp - 14) & 0xFFFF
        out_mant = sign | 0x3800 | mant
    if clamp:
        abs_m = out_mant & 0x7FFF
        if abs_m > 0x7C00 or (out_mant & 0x8000):
            out_mant = 0
        elif out_mant > 0x3BFF:
            out_mant = 0x3C00
    return out_exp, out_mant


def reference(*, flush_denormals=False, no_clamp=False, zero_high=False):
    hi = 0 if zero_high else HIGH_SEED
    words = [SENTINEL] * 8
    for _lane in range(64):
        for bits in CASES:
            e0, m0 = frexp_f16(bits, flush_denormals=flush_denormals)
            e_neg, _ = frexp_f16((bits & 0xFFFF) ^ 0x8000, flush_denormals=flush_denormals)
            _, m_abs_clamp = frexp_f16(bits & 0x7FFF, flush_denormals=flush_denormals, clamp=not no_clamp)
            words.extend((hi | e0, hi | m0, hi | e_neg, hi | m_abs_clamp))
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
        ('exp_clamp', vop3_unary(0x1DA, 4, 258, clamp=1) + [0xBF810000], 'unsupported v_frexp_exp_i16_f16 operand modifiers'),
        ('mant_omod', vop3_unary(0x1D9, 4, 258, omod=1) + [0xBF810000], 'unsupported v_frexp_mant_f16 operand modifiers'),
    ):
        sub = work / tag
        sub.mkdir(exist_ok=True)
        bad_marker = request(sub, root, bad)
        bad_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', bad_marker,
                                  bad_marker.with_suffix('.cso')], capture_output=True, text=True)
        assert bad_res.returncode != 0 and msg in (bad_res.stdout + bad_res.stderr), (tag, bad_res.stdout, bad_res.stderr)
    assert reference() != reference(flush_denormals=True)
    assert reference() != reference(no_clamp=True)
    assert reference() != reference(zero_high=True)
    print('v_frexp_mant_f16/exp_i16_f16 DXIL verified; invalid modifiers refused; denormal, clamp and high-half controls differ')
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
    print('v_frexp_mant_f16/exp_i16_f16: 2048 result words (high halves kept) and 16 guards matched')
    return 0


def ir(dumper):
    """No GPU and no DXC: evaluate lane 0 of the module the direct frontend builds.

    An inline float source (240-248) is a binary16 operand here, so it must read the binary16 value
    of the constant (as v_add_f16 does), not the low half of its f32 pattern.
    """
    from native_direct_container_check import words
    from native_f16_alu_check import INLINE_F16, SHIFT, compute_container, interpret, store
    from native_scalar_lds_check import scalar, sop1, vmove

    # Pinned rows, independent of the lowering: 1.0 = 0.5 * 2^1 and 1/(2pi) = 0x3118 = 0.6367 * 2^-2.
    assert frexp_f16(0x3C00) == (1, 0x3800) and frexp_f16(0x3118) == (0xFFFE, 0x3918)
    cases = []
    for src, half in INLINE_F16.items():
        cases.append(([vop1(0x59, 3, src)], frexp_f16(half)[1]))
        cases.append(([vop1(0x5A, 3, src)], frexp_f16(half)[0]))
    cases.append((vop3_unary(0x1D9, 3, 245, abs0=1, clamp=1), frexp_f16(0x4000, clamp=True)[1]))  # |-2.0| clamp
    cases.append((vop3_unary(0x1D9, 3, 242, neg0=1), frexp_f16(0xBC00)[1]))                         # -1.0
    cases.append((vop3_unary(0x1DA, 3, 248, neg0=1), frexp_f16(0xB118)[0]))                         # -1/(2pi)
    cases.append((v_mov_lit(1, 0x12343C00) + [vop1(0x59, 3, 257)], 0x3800))                         # VGPR source
    code = [sop1(3, 20, 4), sop1(3, 21, 128), 0xF408080A, 0xFA000020,
            scalar(0x1E, 7, 5, 128 + SHIFT), vmove(8, 7), sop1(3, 126, 129)]
    want = []
    for k, (op, result) in enumerate(cases):
        code += v_mov_lit(3, HIGH_SEED | (0x1000 + k)) + op + store(4 * k, 3)
        want.append(HIGH_SEED | result)
    code.append(0xBF810000)
    with tempfile.TemporaryDirectory(prefix='x2w_native_frexp_f16_ir_') as temp:
        path = Path(temp) / 'frexp.cdx'
        path.write_bytes(compute_container(words(*code)))
        run = subprocess.run([dumper, str(path)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    stores = sorted(interpret(run.stdout))
    assert [address for address, _ in stores] == [4 * k for k in range(len(want))], stores[:4]
    bad = [(k, f'{got:08x}', f'{expected:08x}') for k, ((_, got), expected) in enumerate(zip(stores, want))
           if got != expected]
    assert not bad, bad
    print(f'v_frexp_mant_f16/exp_i16_f16 IR: {len(want)} results matched, inline floats read as binary16')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_frexp_f16_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'ir':
        raise SystemExit(ir(sys.argv[2]))
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
