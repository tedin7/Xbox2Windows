"""Direct compute VOP1/VOP3A v_sat_pk_u8_i16 (pack two signed 16-bit halves saturated to u8, zero high half)."""
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

# 16 input words covering negative (-32768, -1), boundary (0, 1, 255, 256), and large positive (+32767) in both halves.
CASES = (
    0x00000000,
    0x00FF0001,
    0x010000FF,
    0x7FFF8000,
    0x80007FFF,
    0xFFFF0000,
    0x0000FFFF,
    0xFF000080,
    0x0080FF00,
    0x00FE0002,
    0x0101FFFE,
    0x12345678,
    0x80017FFE,
    0x007F0080,
    0x00FF0100,
    0xDEADBEEF,
)


def vop1(op, vdst, src0):
    return 0x7E000000 | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)


def vop3a(op, vdst, src0, src1=0, src2=0, *, abs_bits=0, op_sel=0, clamp=0, omod=0, neg=0):
    w0 = 0xD4000000 | ((op & 0x3FF) << 16) | (clamp << 15) | ((op_sel & 0xF) << 11) | ((abs_bits & 7) << 8) | (vdst & 0xFF)
    w1 = ((neg & 7) << 29) | ((omod & 3) << 27) | ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return [w0, w1]


def program(*, sparse=False, first_modifier=0, second_modifier=0):
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for case, val in enumerate(CASES):
        code += [vop2(0x1D, 1, 255, 0), val]  # v_xor_b32 v1, val, v0
        for wide in (False, True):
            # Preload v4 with 0xABCD0000 so we prove D[31:16] is zeroed (or preserved when inactive).
            code += [0x7E0802FF, INACTIVE if sparse else 0xABCD0000]
            if sparse:
                code += [sop1(3, 126, 255), 0x55555555]
            if wide:
                w0, w1 = vop3a(0x1E2, 4, 257)
                if (first_modifier or second_modifier) and case == 0:
                    w0 ^= first_modifier
                    w1 ^= second_modifier
                code += [w0, w1]
            else:
                code += [vop1(0x62, 4, 257)]
            if sparse:
                code += [sop1(3, 126, 193)]  # restore EXEC before storing
            offset = 4 * (case * 2 + int(wide))
            code += [0xE0701000 | offset, 0x80080408]  # buffer_store_dword v4, v8, s[32:35], 0 offen
    return code + [0xBF810000]


def sext16(v):
    v &= 0xFFFF
    return v - 0x10000 if v >= 0x8000 else v


def sat_pk_u8_i16(word):
    lo = max(0, min(255, sext16(word)))
    hi = max(0, min(255, sext16(word >> 16)))
    return lo | (hi << 8)


def reference(sparse=False):
    words = [SENTINEL] * 8
    for lane in range(64):
        for val in CASES:
            for _wide in (False, True):
                words.append(INACTIVE if sparse and lane % 2 else sat_pk_u8_i16(val ^ (lane % 32)))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    # Valid encoding + 6 refused VOP3A modifier/reserved-bit encodings:
    # clamp (w0[15]), op_sel (w0[11]), abs (w0[8]), neg (w1[29]), nonzero src1 (w1[9]), nonzero src2 (w1[18]).
    refusal_cases = (
        (0, 0, ''),
        (1 << 15, 0, 'modifier'),
        (1 << 11, 0, 'op_sel'),
        (1 << 8, 0, 'modifier'),
        (0, 1 << 29, 'modifier'),
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
            assert 'icmp slt i32 ' in dump and 'icmp sgt i32 ' in dump and '255' in dump, dump
    assert reference() != reference(sparse=True)
    print('v_sat_pk_u8_i16 DXIL verified; six VOP3A modifier/unused-src encodings refused; EXEC control differs')
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
    with tempfile.TemporaryDirectory(prefix='x2w_native_sat_pk_u8_i16_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
