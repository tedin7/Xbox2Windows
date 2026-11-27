"""Direct compute VOP1/VOP3A v_cvt_off_f32_i4 (signed 4-bit offset [-8..+7] / 16.0 to binary32)."""
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

# Eight input words whose low nibbles span 0..7 and 8..15 (with non-zero upper 28 bits that must be ignored).
CASES = (
    0xDEAD0000,  # 0 -> 0.0
    0xBEEF0001,  # +1 -> +0.0625
    0xCAFE0004,  # +4 -> +0.25
    0xBABE0007,  # +7 -> +0.4375
    0x12340008,  # -8 -> -0.5
    0x5678000C,  # -4 -> -0.25
    0x9ABC000F,  # -1 -> -0.0625
    0xFFFFFFF2,  # +2 -> +0.125
)
# (wide, omod, clamp)
FORMS = (
    (False, 0, 0),  # VOP1 v_cvt_off_f32_i4_e32
    (True, 0, 0),   # VOP3A v_cvt_off_f32_i4_e64
    (True, 1, 0),   # VOP3A mul:2
    (True, 0, 1),   # VOP3A clamp [0.0, 1.0]
)


def vop1(op, vdst, src0):
    return 0x7E000000 | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)


def vop3a(op, vdst, src0, src1=0, src2=0, *, abs_bits=0, op_sel=0, clamp=0, omod=0, neg=0):
    w0 = 0xD4000000 | ((op & 0x3FF) << 16) | (clamp << 15) | ((op_sel & 0xF) << 11) | ((abs_bits & 7) << 8) | (vdst & 0xFF)
    w1 = ((neg & 7) << 29) | ((omod & 3) << 27) | ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return [w0, w1]


def program(*, sparse=False, drop_clamp=False, first_modifier=0, second_modifier=0):
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for case, val in enumerate(CASES):
        code += [vop2(0x1D, 1, 255, 0), val]  # v_xor_b32 v1, val, v0 (exercises all 16 nibble values across lanes)
        for mode, (wide, omod, clamp) in enumerate(FORMS):
            if sparse:
                code += [0x7E0802FF, INACTIVE, sop1(3, 126, 255), 0x55555555]
            if wide:
                w0, w1 = vop3a(0x18E, 4, 257, omod=omod, clamp=1 if (clamp and not drop_clamp) else 0)
                if (first_modifier or second_modifier) and case == 0 and mode == 1:
                    w0 ^= first_modifier
                    w1 ^= second_modifier
                code += [w0, w1]
            else:
                code += [vop1(0x0E, 4, 257)]
            if sparse:
                code += [sop1(3, 126, 193)]  # restore EXEC before storing
            offset = 4 * (case * len(FORMS) + mode)
            code += [0xE0701000 | offset, 0x80080408]  # buffer_store_dword v4, v8, s[32:35], 0 offen
    return code + [0xBF810000]


def cvt_off_f32_i4(word, omod, clamp):
    nib = word & 0xF
    s4 = nib - 16 if nib >= 8 else nib
    val = s4 * 0.0625
    if omod == 1:
        val *= 2.0
    elif omod == 2:
        val *= 4.0
    elif omod == 3:
        val *= 0.5
    if clamp:
        val = max(0.0, min(1.0, val))
    return struct.unpack('<I', struct.pack('<f', val))[0]


def reference(sparse=False, ignore_clamp=False):
    words = [SENTINEL] * 8
    for lane in range(64):
        for val in CASES:
            for _wide, omod, clamp in FORMS:
                words.append(INACTIVE if sparse and lane % 2 else
                             cvt_off_f32_i4(val ^ (lane % 32), omod, clamp and not ignore_clamp))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    # Valid encoding + 5 refused VOP3A encodings:
    # abs (w0[8]), neg (w1[29]), op_sel (w0[11]), nonzero src1 (w1[9]), nonzero src2 (w1[18]).
    refusal_cases = (
        (0, 0, ''),
        (1 << 8, 0, 'integer conversion source modifier'),
        (0, 1 << 29, 'integer conversion source modifier'),
        (1 << 11, 0, 'op_sel'),
        (0, 1 << 9, 'unused src1/src2'),
        (0, 1 << 18, 'unused src1/src2'),
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
            assert 'ashr i32 ' in dump and ', 28' in dump and 'sitofp i32 ' in dump, dump
    assert reference() != reference(ignore_clamp=True)
    assert reference() != reference(sparse=True)
    print('v_cvt_off_f32_i4 DXIL verified; five VOP3A modifier/unused-src encodings refused; clamp and EXEC differ')
    return 0


def hardware(replay, sdk, work, root):
    for sparse, drop_clamp in ((False, False), (True, False), (False, True)):
        marker = request(work, root, program(sparse=sparse, drop_clamp=drop_clamp))
        output = work / 'out.bin'
        result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        expected = reference(sparse, ignore_clamp=drop_clamp)
        actual = struct.unpack(f'<{len(expected)}I', output.read_bytes())
        mismatch = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
        if mismatch is not None:
            raise AssertionError(f'sparse={sparse} drop_clamp={drop_clamp} word {mismatch}: got {actual[mismatch]:08x}, '
                                 f'expected {expected[mismatch]:08x}')
        if drop_clamp:
            changed = sum(a != b for a, b in zip(actual, reference()))
            assert changed > 0, 'dropping clamp did not change the readback'
            print(f'no-clamp control: {changed} words differed from the clamped reference')
        else:
            print(f'sparse={sparse}: 2048 result words and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_cvt_off_f32_i4_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
