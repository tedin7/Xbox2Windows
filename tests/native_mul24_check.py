"""Direct compute VOP2/VOP3A 24-bit integer multiplies (v_mul_i32_i24, v_mul_hi_i32_i24, v_mul_hi_u32_u24)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
from native_scalar_lds_check import sop1

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969
INACTIVE = 0x13579BDF

# Eight pairs covering upper-byte noise (must be ignored), +/-2^23 boundaries, max 24-bit, and sign combinations.
CASES = (
    (0xDE000003, 0xBE000007),
    (0xFF7FFFFF, 0xAA7FFFFF),  # (+8388607) * (+8388607)
    (0x00800000, 0x00800000),  # (-8388608) * (-8388608) = +2^46 -> hi = +0x4000
    (0x12800000, 0x347FFFFF),  # (-8388608) * (+8388607) -> negative hi
    (0x56FFFFFF, 0x78FFFFFF),  # (-1)*(-1)=+1 signed; (2^24-1)^2 unsigned -> hi = 0xFFFE
    (0x9AFFFFFF, 0xBC000002),  # (-1)*(+2)=-2 -> hi = 0xFFFFFFFF signed; hi = 0x0001 unsigned
    (0xAB123456, 0xCD654321),  # general positive/unsigned product
    (0xEF800001, 0x10FFFFFE),  # near-min * -2
)
# (vop2_op, wide) -> covers all 3 opcodes across both VOP2 (_e32) and VOP3A (_e64) encodings.
FORMS = (
    (0x09, False),  # v_mul_i32_i24_e32
    (0x0A, False),  # v_mul_hi_i32_i24_e32
    (0x0C, False),  # v_mul_hi_u32_u24_e32
    (0x0A, True),   # v_mul_hi_i32_i24_e64
)


def vop3a(op, vdst, src0, src1, src2=0, *, abs_bits=0, op_sel=0, clamp=0, omod=0, neg=0):
    w0 = 0xD4000000 | ((op & 0x3FF) << 16) | (clamp << 15) | ((op_sel & 0xF) << 11) | ((abs_bits & 7) << 8) | (vdst & 0xFF)
    w1 = ((neg & 7) << 29) | ((omod & 3) << 27) | ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return [w0, w1]


def program(*, sparse=False, first_modifier=0, second_modifier=0):
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for case, (a, b) in enumerate(CASES):
        code += [vop2(0x1D, 1, 255, 0), a,  # v_xor_b32 v1, a, v0
                 0x7E0402FF, b]             # v_mov_b32 v2, b
        for mode, (op2, wide) in enumerate(FORMS):
            if sparse:
                code += [0x7E0802FF, INACTIVE, sop1(3, 126, 255), 0x55555555]
            if wide:
                w0, w1 = vop3a(0x100 + op2, 4, 257, 258)
                if (first_modifier or second_modifier) and case == 0:
                    w0 ^= first_modifier
                    w1 ^= second_modifier
                code += [w0, w1]
            else:
                code += [vop2(op2, 4, 257, 2)]
            if sparse:
                code += [sop1(3, 126, 193)]  # restore EXEC before storing
            offset = 4 * (case * len(FORMS) + mode)
            code += [0xE0701000 | offset, 0x80080408]  # buffer_store_dword v4, v8, s[32:35], 0 offen
    return code + [0xBF810000]


def sext24(v):
    v &= 0xFFFFFF
    return v - 0x1000000 if v >= 0x800000 else v


def eval_mul24(op2, a, b):
    if op2 == 0x09:
        return (sext24(a) * sext24(b)) & MASK
    if op2 == 0x0A:
        return ((sext24(a) * sext24(b)) >> 32) & MASK
    return (((a & 0xFFFFFF) * (b & 0xFFFFFF)) >> 32) & MASK


def reference(sparse=False):
    words = [SENTINEL] * 8
    for lane in range(64):
        for a, b in CASES:
            for op2, _wide in FORMS:
                words.append(INACTIVE if sparse and lane % 2 else eval_mul24(op2, a ^ (lane % 32), b))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    # Valid encoding + 5 refused VOP3A encodings:
    # clamp (w0[15]), op_sel (w0[11]), abs (w0[8]), neg (w1[29]), nonzero src2 (w1[18]).
    refusal_cases = (
        (0, 0, ''),
        (1 << 15, 0, 'modifier'),
        (1 << 11, 0, 'op_sel'),
        (1 << 8, 0, 'modifier'),
        (0, 1 << 29, 'modifier'),
        (0, 1 << 18, 'unused src2'),
    )
    for first, second, needle in refusal_cases:
        marker = request(work, root, program(first_modifier=first, second_modifier=second))
        output = marker.with_suffix('.cso')
        result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                                capture_output=True, text=True)
        if first or second:
            assert result.returncode and needle in result.stderr, (first, second, result.stderr)
            assert not output.exists()
        else:
            assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
            dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                                  text=True, check=True).stdout
            assert 'ashr i32 ' in dump and ', 20' in dump and 'and i32 ' in dump and ', 4095' in dump, dump
    assert reference() != reference(sparse=True)
    print('mul24 DXIL verified; five VOP3A modifier/unused-src2 encodings refused; EXEC control differs')
    return 0


def hardware(replay, sdk, work, root):
    for sparse in (False, True):
        marker = request(work, root, program(sparse=sparse))
        output = work / 'out.bin'
        result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        expected = reference(sparse)
        actual = struct.unpack(f'<{len(expected)}I', output.read_bytes())
        mismatch = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
        if mismatch is not None:
            raise AssertionError(f'sparse={sparse} word {mismatch}: got {actual[mismatch]:08x}, '
                                 f'expected {expected[mismatch]:08x}')
        print(f'sparse={sparse}: 2048 result words and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_mul24_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
