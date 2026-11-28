"""Direct compute v_floor_f16, v_ceil_f16, v_trunc_f16 and v_rndne_f16 (VOP1/VOP3A, high-half preserving)."""
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
    0xCAFE8000,  # -0.0h (with high junk in src0)
    0x00000001,  # +min_sub (2^-24)
    0x00008001,  # -min_sub (-2^-24)
    0x00003800,  # +0.5h (rndne ties to +0.0h)
    0x00003E00,  # +1.5h (rndne ties to +2.0h)
    0x00004100,  # +2.5h (rndne ties to +2.0h)
    0x0000BE00,  # -1.5h (floor=-2, ceil=-1, trunc=-1, rndne=-2)
    0x000063FF,  # +1023.5h (ceil/rndne carry into exponent -> 1024.0h = 0x6400)
)


def v_mov_lit(vdst, lit):
    return [0x7E0002FF | ((vdst & 0xFF) << 17), lit & MASK]


def vop1(op, vdst, src0):
    return 0x7E000000 | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)


def vop3_unary(op, vdst, src0, *, abs0=0, neg0=0, clamp=0, omod=0, op_sel=0):
    w0 = (0xD4000000 | ((op & 0x3FF) << 16) | ((clamp & 1) << 15) |
          ((op_sel & 0xF) << 11) | ((abs0 & 1) << 8) | (vdst & 0xFF))
    w1 = ((neg0 & 1) << 29) | ((omod & 3) << 27) | (src0 & 0x1FF)
    return [w0, w1]


def program():
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for ci, bits in enumerate(CASES):
        code += v_mov_lit(2, bits)
        # 0: VOP1 v_floor_f16
        code += v_mov_lit(4, HIGH_SEED | 0x1111) + [vop1(0x5B, 4, 258), 0xE0701000 | (4 * (ci * 4 + 0)), 0x80080408]
        # 1: VOP1 v_ceil_f16
        code += v_mov_lit(4, HIGH_SEED | 0x2222) + [vop1(0x5C, 4, 258), 0xE0701000 | (4 * (ci * 4 + 1)), 0x80080408]
        # 2: VOP3A v_trunc_f16
        code += v_mov_lit(4, HIGH_SEED | 0x3333) + vop3_unary(0x1DD, 4, 258) + [0xE0701000 | (4 * (ci * 4 + 2)), 0x80080408]
        # 3: VOP3A v_rndne_f16
        code += v_mov_lit(4, HIGH_SEED | 0x4444) + vop3_unary(0x1DE, 4, 258) + [0xE0701000 | (4 * (ci * 4 + 3)), 0x80080408]
    return code + [0xBF810000]


def round_f16(bits, mode, *, ties_away=False, lose_neg_zero=False):
    bits &= 0xFFFF
    sign = 0 if lose_neg_zero else (bits & 0x8000)
    neg = (bits & 0x8000) != 0
    mag = bits & 0x7FFF
    exp = (bits >> 10) & 0x1F
    if exp >= 25:
        return bits
    if exp < 15:
        if mode == 'trunc':
            return sign
        if mode == 'floor':
            return 0xBC00 if (mag != 0 and neg) else sign
        if mode == 'ceil':
            return 0x3C00 if (mag != 0 and not neg) else sign
        # rndne
        if ties_away:
            return (bits & 0x8000) | 0x3C00 if mag >= 0x3800 else sign
        return (bits & 0x8000) | 0x3C00 if mag > 0x3800 else sign
    frac_bits = 25 - exp
    unit = 1 << frac_bits
    mask = unit - 1
    half = unit >> 1
    frac = bits & mask
    base = bits & (~mask & 0xFFFF)
    if mode == 'trunc':
        return base
    if mode == 'floor':
        return (base + unit) & 0xFFFF if (neg and frac != 0) else base
    if mode == 'ceil':
        return (base + unit) & 0xFFFF if ((not neg) and frac != 0) else base
    # rndne
    if ties_away:
        inc = frac >= half
    else:
        inc = frac > half or (frac == half and (base & unit) != 0)
    return (base + unit) & 0xFFFF if inc else base


def reference(*, ties_away=False, lose_neg_zero=False, zero_high=False):
    hi = 0 if zero_high else HIGH_SEED
    words = [SENTINEL] * 8
    for _lane in range(64):
        for bits in CASES:
            words.extend((
                hi | round_f16(bits, 'floor', ties_away=ties_away, lose_neg_zero=lose_neg_zero),
                hi | round_f16(bits, 'ceil', ties_away=ties_away, lose_neg_zero=lose_neg_zero),
                hi | round_f16(bits, 'trunc', ties_away=ties_away, lose_neg_zero=lose_neg_zero),
                hi | round_f16(bits, 'rndne', ties_away=ties_away, lose_neg_zero=lose_neg_zero),
            ))
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
        ('omod', vop3_unary(0x1DB, 4, 258, omod=1) + [0xBF810000], 'unsupported v_floor_f16 operand modifiers'),
        ('op_sel', vop3_unary(0x1DE, 4, 258, op_sel=1) + [0xBF810000], 'unsupported v_rndne_f16 operand modifiers'),
    ):
        sub = work / tag
        sub.mkdir(exist_ok=True)
        bad_marker = request(sub, root, bad)
        bad_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', bad_marker,
                                  bad_marker.with_suffix('.cso')], capture_output=True, text=True)
        assert bad_res.returncode != 0 and msg in (bad_res.stdout + bad_res.stderr), (tag, bad_res.stdout, bad_res.stderr)
    assert reference() != reference(ties_away=True)
    assert reference() != reference(lose_neg_zero=True)
    assert reference() != reference(zero_high=True)
    print('v_floor/ceil/trunc/rndne_f16 DXIL verified; invalid modifiers refused; tie, -0 and high-half controls differ')
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
    print('v_floor/ceil/trunc/rndne_f16: 2048 result words (high halves kept) and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_round_f16_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
