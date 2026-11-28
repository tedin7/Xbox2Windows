"""Direct compute VOP3P packed 16-bit multiply/mad (v_pk_mul_lo_u16, v_pk_mad_u16, v_pk_mad_i16)."""
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

CASES = (
    (0x00030005, 0x0007000B, 0x00010002),  # small positive products & sums
    (0x01000003, 0x01000005, 0x00010007),  # unsigned overflow 0x10001 -> 0xFFFF clamp vs 1 wrap
    (0xFFFF8000, 0xFFFF0002, 0xFFFF0000),  # max u16 * max u16 + max u16 = 0xFFFF0000 (no u32 overflow)
    (0x80007FFF, 0x00028000, 0x7FFF0001),  # signed (-32768)*(-32768) -> +32767 saturation
    (0x80008000, 0x7FFF7FFF, 0x80008000),  # signed (-32768)*(+32767)-32768 -> -32768 saturation
    (0x0003FFFD, 0x00050002, 0x0001FFFE),  # signed negative product without saturation
    (0x00FF0000, 0x0100FFFF, 0x00FF0001),  # exact unsigned boundary 255*256+255 = 0xFFFF
    (0xABCD0100, 0x00040100, 0xDEADBEEF),  # general wrapping patterns
)
# (opcode, clamp, op_sel, op_sel_hi)
FORMS = (
    (0x01, False, 0, 3),  # v_pk_mul_lo_u16 default
    (0x01, False, 1, 2),  # v_pk_mul_lo_u16 swizzled op_sel:[1,0] op_sel_hi:[0,1]
    (0x09, True, 0, 7),   # v_pk_mad_u16 clamp default
    (0x00, True, 5, 2),   # v_pk_mad_i16 clamp swizzled op_sel:[1,0,1] op_sel_hi:[0,1,0]
)


def vop3p(opcode, vdst, src0, src1, src2=0, *, op_sel=0, op_sel_hi=7, clamp=0, neg_lo=0, neg_hi=0):
    w0 = (0xCC000000 | ((opcode & 0x7F) << 16) | (clamp << 15) | (((op_sel_hi >> 2) & 1) << 14)
          | ((op_sel & 7) << 11) | ((neg_hi & 7) << 8) | (vdst & 0xFF))
    w1 = (((neg_lo & 7) << 29) | ((op_sel_hi & 3) << 27)
          | ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF))
    return [w0, w1]


def program(*, sparse=False, drop_clamp=False, first_modifier=0, second_modifier=0, mod_opcode=0x01):
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for case, (a, b, c) in enumerate(CASES):
        code += [vop2(0x1D, 1, 255, 0), a,       # v_xor_b32 v1, a, v0
                 0x7E0402FF, b, 0x7E0602FF, c]   # v_mov_b32 v2, b; v_mov_b32 v3, c
        for mode, (opcode, clamp, op_sel, op_sel_hi) in enumerate(FORMS):
            if sparse:
                code += [0x7E0802FF, INACTIVE, sop1(3, 126, 255), 0x55555555]
            is_mad = opcode != 0x01
            w0, w1 = vop3p(opcode, 4, 257, 258, 259 if is_mad else 0,
                           op_sel=op_sel, op_sel_hi=op_sel_hi if is_mad else (op_sel_hi | 4),
                           clamp=1 if (clamp and not drop_clamp) else 0)
            if (first_modifier or second_modifier) and opcode == mod_opcode and case == 0:
                w0 ^= first_modifier
                w1 ^= second_modifier
            code += [w0, w1]
            if sparse:
                code += [sop1(3, 126, 193)]  # restore EXEC before storing
            offset = 4 * (case * len(FORMS) + mode)
            code += [0xE0701000 | offset, 0x80080408]  # buffer_store_dword v4, v8, s[32:35], 0 offen
    return code + [0xBF810000]


def sext16(v):
    v &= 0xFFFF
    return v - 0x10000 if v >= 0x8000 else v


def eval_mul_mad(opcode, a, b, c, clamp, op_sel, op_sel_hi):
    def pick(word, hi):
        return ((word >> 16) if hi else word) & 0xFFFF

    def lane(sel):
        x = pick(a, sel & 1)
        y = pick(b, (sel >> 1) & 1)
        if opcode == 0x01:
            return (x * y) & 0xFFFF
        z = pick(c, (sel >> 2) & 1)
        if opcode == 0x00:
            r = sext16(x) * sext16(y) + sext16(z)
            if clamp:
                r = max(-32768, min(32767, r))
            return r & 0xFFFF
        r = x * y + z
        if clamp:
            r = min(0xFFFF, r)
        return r & 0xFFFF

    return (lane(op_sel) | (lane(op_sel_hi) << 16)) & MASK


def reference(sparse=False, ignore_clamp=False):
    words = [SENTINEL] * 8
    for lane in range(64):
        for a, b, c in CASES:
            for opcode, clamp, op_sel, op_sel_hi in FORMS:
                words.append(INACTIVE if sparse and lane % 2 else
                             eval_mul_mad(opcode, a ^ (lane % 32), b, c,
                                          clamp and not ignore_clamp, op_sel, op_sel_hi))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    # Valid encoding + 5 refused modifier/reserved-bit encodings:
    # mul_lo: clamp (w0[15]), op_sel[2] (w0[13]), nonzero src2 (w1[18]);
    # mad_i16: neg_hi (w0[8]), neg_lo (w1[29]).
    cases = (
        (0x01, 0, 0),
        (0x01, 1 << 15, 0),
        (0x01, 1 << 13, 0),
        (0x01, 0, 1 << 18),
        (0x00, 1 << 8, 0),
        (0x00, 0, 1 << 29),
    )
    for mod_op, first, second in cases:
        marker = request(work, root, program(first_modifier=first, second_modifier=second, mod_opcode=mod_op))
        output = marker.with_suffix('.cso')
        result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                                capture_output=True, text=True)
        if first or second:
            assert result.returncode and 'packed 16-bit mul/mad operand modifiers' in result.stderr, result.stderr
            assert not output.exists()
        else:
            assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
            dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                                  text=True, check=True).stdout
            assert 'mul i32 ' in dump and 'icmp sgt i32 ' in dump and '32767' in dump, dump
            assert 'icmp ugt i32 ' in dump and '65535' in dump, dump
    assert reference() != reference(ignore_clamp=True)
    assert reference() != reference(sparse=True)
    print('v_pk_mul_lo_u16/mad_16 DXIL verified; five modifier encodings refused; clamp and EXEC controls differ')
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
    with tempfile.TemporaryDirectory(prefix='x2w_native_pk_mul_mad_16_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
