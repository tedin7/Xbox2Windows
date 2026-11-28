"""Direct compute VOP3P packed 16-bit add/sub (v_pk_add/sub_i16, v_pk_add/sub_u16) with and without clamp."""
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
    (0x7FFF8000, 0x0001FFFF),  # signed hi overflow (+32767+1), lo underflow (-32768-1)
    (0x4000C000, 0x4000C000),  # signed hi/lo saturation both directions
    (0x00008000, 0x00018000),  # 0 - (-32768) -> +32767 saturation; unsigned underflow -> 0
    (0xFFFF8000, 0x00018000),  # unsigned wrap & saturation to 0xFFFF
    (0x00010005, 0x00020003),  # unsigned sub clamp to 0 vs exact diff
    (0x7FFF8000, 0x80007FFF),  # opposite-sign cancellation
    (0x0003FFFD, 0x00050002),  # small positive/negative without saturation
    (0xDEADBEEF, 0x11112222),  # general bit patterns
)
# (opcode, clamp, op_sel, op_sel_hi)
FORMS = (
    (0x02, True, 0, 3),   # v_pk_add_i16 clamp default
    (0x03, True, 1, 2),   # v_pk_sub_i16 clamp swizzled op_sel:[1,0] op_sel_hi:[0,1]
    (0x0A, True, 0, 3),   # v_pk_add_u16 clamp default
    (0x0B, True, 0, 0),   # v_pk_sub_u16 clamp broadcast low halves op_sel:[0,0] op_sel_hi:[0,0]
)


def vop3p(opcode, vdst, src0, src1, *, op_sel=0, op_sel_hi=3, clamp=0, neg_lo=0, neg_hi=0, src2=0):
    w0 = (0xCC004000 | ((opcode & 0x7F) << 16) | (clamp << 15) | ((op_sel & 7) << 11)
          | ((neg_hi & 7) << 8) | (vdst & 0xFF))
    w1 = (((neg_lo & 7) << 29) | ((op_sel_hi & 3) << 27)
          | ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF))
    return [w0, w1]


def program(*, sparse=False, drop_clamp=False, first_modifier=0, second_modifier=0):
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for case, (a, b) in enumerate(CASES):
        code += [vop2(0x1D, 1, 255, 0), a,  # v_xor_b32 v1, a, v0
                 0x7E0402FF, b]             # v_mov_b32 v2, b
        for mode, (opcode, clamp, op_sel, op_sel_hi) in enumerate(FORMS):
            if sparse:
                code += [0x7E0802FF, INACTIVE, sop1(3, 126, 255), 0x55555555]
            w0, w1 = vop3p(opcode, 4, 257, 258, op_sel=op_sel, op_sel_hi=op_sel_hi,
                           clamp=1 if (clamp and not drop_clamp) else 0)
            code += [w0 ^ first_modifier, w1 ^ second_modifier]
            if sparse:
                code += [sop1(3, 126, 193)]  # restore EXEC before storing
            offset = 4 * (case * len(FORMS) + mode)
            code += [0xE0701000 | offset, 0x80080408]  # buffer_store_dword v4, v8, s[32:35], 0 offen
    return code + [0xBF810000]


def sext16(v):
    v &= 0xFFFF
    return v - 0x10000 if v >= 0x8000 else v


def eval_add_sub(opcode, a, b, clamp, op_sel, op_sel_hi):
    signed = opcode in (0x02, 0x03)
    sub = opcode in (0x03, 0x0B)

    def pick(word, hi):
        return ((word >> 16) if hi else word) & 0xFFFF

    def lane(sel):
        x = pick(a, sel & 1)
        y = pick(b, (sel >> 1) & 1)
        if signed:
            xv, yv = sext16(x), sext16(y)
            r = xv - yv if sub else xv + yv
            if clamp:
                r = max(-32768, min(32767, r))
            return r & 0xFFFF
        r = x - y if sub else x + y
        if clamp:
            r = max(0, min(0xFFFF, r))
        return r & 0xFFFF

    return (lane(op_sel) | (lane(op_sel_hi) << 16)) & MASK


def reference(sparse=False, ignore_clamp=False):
    words = [SENTINEL] * 8
    for lane in range(64):
        for a, b in CASES:
            for opcode, clamp, op_sel, op_sel_hi in FORMS:
                words.append(INACTIVE if sparse and lane % 2 else
                             eval_add_sub(opcode, a ^ (lane % 32), b, clamp and not ignore_clamp, op_sel, op_sel_hi))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    # Valid encoding + 4 refused modifier/reserved-bit encodings:
    # op_sel[2] (w0[13]), neg_hi (w0[8]), neg_lo (w1[29]), nonzero src2 (w1[18]).
    for first, second in ((0, 0), (1 << 13, 0), (1 << 8, 0), (0, 1 << 29), (0, 1 << 18)):
        marker = request(work, root, program(first_modifier=first, second_modifier=second))
        output = marker.with_suffix('.cso')
        result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                                capture_output=True, text=True)
        if first or second:
            assert result.returncode and 'packed 16-bit add/sub operand modifiers' in result.stderr, result.stderr
            assert not output.exists()
        else:
            assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
            dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                                  text=True, check=True).stdout
            assert 'icmp sgt i32 ' in dump and '32767' in dump, dump
            assert 'icmp ugt i32 ' in dump and '65535' in dump, dump
    assert reference() != reference(ignore_clamp=True)
    assert reference() != reference(sparse=True)
    print('v_pk_add/sub_16 DXIL verified; four modifier encodings refused; clamp and EXEC controls differ')
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
    with tempfile.TemporaryDirectory(prefix='x2w_native_pk_add_sub_16_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
