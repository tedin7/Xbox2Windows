"""Direct compute v_ldexp_f16 (VOP2 and VOP3A, exact integer binary16 scaling, high-half preserving)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969
HIGH_SEED = 0xBEEF0000

# Two 8-case batches covering normals, denormals, subnormal ties, ±0, ±Inf, qNaN, and large/min deltas.
BATCHES = (
    (
        (0xCAFE3C00, 0x00000002),  # 1.0h * 2^2 = 4.0h
        (0x1234C200, 0xABCDFFFE),  # -3.0h * 2^-2 = -0.75h (delta = -2)
        (0x00000001, 0x0000000A),  # smallest subnormal 2^-24 * 2^10 = smallest normal 2^-14
        (0x00000400, 0x0000FFFF),  # smallest normal 2^-14 * 2^-1 = largest subnormal 2^-15
        (0x00000403, 0x0000FFFE),  # subnormal round-up (rest > half)
        (0x00000402, 0x0000FFFE),  # subnormal tie-to-even (even q stays)
        (0x00000406, 0x0000FFFE),  # subnormal tie-to-even (odd q rounds up)
        (0x00007BFF, 0x00000001),  # max finite overflow -> +Inf (clamped to 1.0h)
    ),
    (
        (0x00000000, 0x00000005),  # +0.0h pass-through
        (0x00008000, 0x0000FFFB),  # -0.0h pass-through
        (0x00007C00, 0x0000FFF0),  # +Inf pass-through (clamped to 1.0h)
        (0x0000FC00, 0x00000010),  # -Inf pass-through (clamped to 0.0h)
        (0x00007E01, 0x00000003),  # qNaN pass-through (clamped to 0.0h)
        (0x00003C00, 0x00000064),  # 1.0h * 2^100 -> +Inf (clamped delta +100 -> +64)
        (0x00003C00, 0x0000FF9C),  # 1.0h * 2^-100 -> +0.0h (clamped delta -100 -> -64)
        (0x00003C00, 0x00008000),  # 1.0h * 2^-32768 -> +0.0h (INT16_MIN delta)
    ),
)


def v_mov_lit(vdst, lit):
    return [0x7E0002FF | ((vdst & 0xFF) << 17), lit & MASK]


def vop2(op, vdst, src0, vsrc1):
    return ((op & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)


def vop3_binary(op, vdst, src0, src1, *, abs_m=0, neg_m=0, clamp=0, omod=0):
    w0 = 0xD4000000 | ((op & 0x3FF) << 16) | ((clamp & 1) << 15) | ((abs_m & 7) << 8) | (vdst & 0xFF)
    w1 = ((neg_m & 7) << 29) | ((omod & 3) << 27) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return [w0, w1]


def program(batch_idx):
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for ci, (s0, s1) in enumerate(BATCHES[batch_idx]):
        code += v_mov_lit(2, s0) + v_mov_lit(3, s1)
        # 0: VOP2 v_ldexp_f16 v4, v2, v3
        code += v_mov_lit(4, HIGH_SEED | 0x1111) + [vop2(0x3B, 4, 258, 3), 0xE0701000 | (4 * (ci * 4 + 0)), 0x80080408]
        # 1: batch 0 -> VOP3A v_ldexp_f16 v4, v2, v3; batch 1 -> VOP2 inline float constant 1.0 (src0=242 -> 0x3C00)
        if batch_idx == 0:
            code += v_mov_lit(4, HIGH_SEED | 0x2222) + vop3_binary(0x13B, 4, 258, 259) + [
                0xE0701000 | (4 * (ci * 4 + 1)), 0x80080408]
        else:
            code += v_mov_lit(4, HIGH_SEED | 0x2222) + [vop2(0x3B, 4, 242, 3),
                                                        0xE0701000 | (4 * (ci * 4 + 1)), 0x80080408]
        # 2: VOP3A v_ldexp_f16 v4, -v2, v3
        code += v_mov_lit(4, HIGH_SEED | 0x3333) + vop3_binary(0x13B, 4, 258, 259, neg_m=1) + [
            0xE0701000 | (4 * (ci * 4 + 2)), 0x80080408]
        # 3: VOP3A v_ldexp_f16 v4, |v2|, v3 clamp
        code += v_mov_lit(4, HIGH_SEED | 0x4444) + vop3_binary(0x13B, 4, 258, 259, abs_m=1, clamp=1) + [
            0xE0701000 | (4 * (ci * 4 + 3)), 0x80080408]
    return code + [0xBF810000]


def sext16(v):
    v &= 0xFFFF
    return v - 0x10000 if (v & 0x8000) else v


def ldexp_f16(bits, delta_raw, *, flush_denormals=False, truncate_subnormal=False, drop_keep=False, clamp=False):
    bits &= 0xFFFF
    delta = max(-64, min(64, sext16(delta_raw)))
    sign = bits & 0x8000
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if not drop_keep and (exp == 0x1F or (bits & 0x7FFF) == 0):
        res = bits
    else:
        if exp == 0:
            if flush_denormals:
                res = sign
                exp = None
            else:
                steps = 0
                x = mant
                while x and (x & 0x400) == 0:
                    x <<= 1
                    steps += 1
                mant = x & 0x3FF
                e = 1 - steps + delta
        else:
            e = exp + delta
        if exp is not None:
            if e > 30:
                res = sign | 0x7C00
            elif e > 0:
                res = sign | (e << 10) | mant
            elif e < -10:
                res = sign
            else:
                full = 0x400 | mant
                amt = 1 - e
                half = 1 << (amt - 1)
                rest = full & ((1 << amt) - 1)
                q = full >> amt
                up = 0 if truncate_subnormal else int(rest > half or (rest == half and (q & 1) != 0))
                res = sign | (q + up)
    if clamp:
        abs_r = res & 0x7FFF
        if abs_r > 0x7C00 or (res & 0x8000):
            res = 0
        elif res > 0x3BFF:
            res = 0x3C00
    return res & 0xFFFF


def reference(batch_idx, *, flush_denormals=False, truncate_subnormal=False, drop_keep=False,
              inline_as_zero=False, no_clamp=False, zero_high=False):
    hi = 0 if zero_high else HIGH_SEED
    words = [SENTINEL] * 8
    for _lane in range(64):
        for s0, s1 in BATCHES[batch_idx]:
            r0 = ldexp_f16(s0, s1, flush_denormals=flush_denormals, truncate_subnormal=truncate_subnormal,
                           drop_keep=drop_keep)
            if batch_idx == 0:
                r1 = r0
            else:
                r1 = ldexp_f16(0x0000 if inline_as_zero else 0x3C00, s1,
                               flush_denormals=flush_denormals, truncate_subnormal=truncate_subnormal,
                               drop_keep=drop_keep)
            r_neg = ldexp_f16((s0 & 0xFFFF) ^ 0x8000, s1, flush_denormals=flush_denormals,
                              truncate_subnormal=truncate_subnormal, drop_keep=drop_keep)
            r_abs_clamp = ldexp_f16(s0 & 0x7FFF, s1, flush_denormals=flush_denormals,
                                    truncate_subnormal=truncate_subnormal, drop_keep=drop_keep, clamp=not no_clamp)
            words.extend((hi | r0, hi | r1, hi | r_neg, hi | r_abs_clamp))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    for bi in range(len(BATCHES)):
        sub = work / f'b{bi}'
        sub.mkdir(exist_ok=True)
        marker = request(sub, root, program(bi))
        output = marker.with_suffix('.cso')
        result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                                capture_output=True, text=True)
        assert result.returncode == 0 and output.is_file(), (bi, result.stdout, result.stderr)
        dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                              text=True, check=True).stdout
        assert 'shl i32 ' in dump and 'lshr i32 ' in dump, dump
        if bi == 1:
            assert '15360' in dump, 'inline float 1.0 must lower to 0x3c00 (15360)'
    for tag, bad, msg in (
        ('omod', vop3_binary(0x13B, 4, 258, 259, omod=1) + [0xBF810000], 'unsupported v_ldexp_f16 operand modifiers'),
        ('neg_src1', vop3_binary(0x13B, 4, 258, 259, neg_m=2) + [0xBF810000], 'unsupported v_ldexp_f16 operand modifiers'),
        ('bad_mode', [0xBFA4000C, vop2(0x3B, 4, 258, 3), 0xBF810000], 'unsupported binary16 rounding or denormal mode'),
    ):
        sub = work / tag
        sub.mkdir(exist_ok=True)
        bad_marker = request(sub, root, bad)
        bad_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', bad_marker,
                                  bad_marker.with_suffix('.cso')], capture_output=True, text=True)
        assert bad_res.returncode != 0 and msg in (bad_res.stdout + bad_res.stderr), (
            tag, bad_res.stdout, bad_res.stderr)
    assert reference(0) != reference(0, flush_denormals=True)
    assert reference(0) != reference(0, truncate_subnormal=True)
    assert reference(0) != reference(0, no_clamp=True)
    assert reference(0) != reference(0, zero_high=True)
    assert reference(1) != reference(1, drop_keep=True)
    assert reference(1) != reference(1, inline_as_zero=True)
    print('v_ldexp_f16 DXIL verified; invalid modifiers/mode refused; all negative controls differ')
    return 0


def hardware(replay, sdk, work, root):
    for bi in range(len(BATCHES)):
        sub = work / f'hw{bi}'
        sub.mkdir(exist_ok=True)
        marker = request(sub, root, program(bi))
        output = work / f'out_{bi}.bin'
        result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        expected = reference(bi)
        actual = struct.unpack(f'<{len(expected)}I', output.read_bytes())
        mismatch = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
        if mismatch is not None:
            raise AssertionError(f'batch={bi} word {mismatch}: got {actual[mismatch]:08x}, expected {expected[mismatch]:08x}')
        print(f'batch={bi}: 2048 result words (high halves kept) and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_ldexp_f16_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
