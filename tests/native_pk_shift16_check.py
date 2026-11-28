"""Direct compute VOP3P packed 16-bit shifts (v_pk_lshlrev_b16, v_pk_lshrrev_b16, v_pk_ashrrev_i16)."""
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

# (shift_word, value_word) covering counts 0..15, >15 masking, sign bit 0x8000, and lane-bleed hazards.
CASES = (
    (0x00040001, 0x00018001),
    (0x000F0010, 0x80007FFF),
    (0x001F0008, 0xFFFF1234),
    (0x0007000C, 0xDEADBEEF),
    (0x00000001, 0x80017FFE),
    (0x00140004, 0xFF0000F0),
    (0x00000000, 0x7FFF8000),
    (0x000F000F, 0x80007FFF),
)
# (opcode, op_sel, op_sel_hi)
FORMS = (
    (0x04, 0, 3),  # v_pk_lshlrev_b16 default
    (0x05, 1, 2),  # v_pk_lshrrev_b16 swizzled op_sel:[1,0] op_sel_hi:[0,1]
    (0x06, 0, 3),  # v_pk_ashrrev_i16 default
    (0x06, 0, 0),  # v_pk_ashrrev_i16 broadcast low halves op_sel:[0,0] op_sel_hi:[0,0]
)


def vop3p(opcode, vdst, src0, src1, *, op_sel=0, op_sel_hi=3, clamp=0, neg_lo=0, neg_hi=0, src2=0):
    w0 = (0xCC004000 | ((opcode & 0x7F) << 16) | (clamp << 15) | ((op_sel & 7) << 11)
          | ((neg_hi & 7) << 8) | (vdst & 0xFF))
    w1 = (((neg_lo & 7) << 29) | ((op_sel_hi & 3) << 27)
          | ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF))
    return [w0, w1]


def program(*, sparse=False, first_modifier=0, second_modifier=0):
    # Group index and local lane identify a 24-word row (6 cases * 4 forms).
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    # Multiply row index by 24/32? Actually native_store_replay allocates 32 dwords per lane (64*32 = 2048 words).
    # Keep stride 32 dwords per lane (offset < 128 bytes) to match native_store_replay's buffer layout.
    for case, (amt_w, val_w) in enumerate(CASES):
        code += [vop2(0x1D, 1, 255, 0), amt_w,  # v_xor_b32 v1, amt_w, v0 (varies shift counts per lane)
                 0x7E0402FF, val_w]             # v_mov_b32 v2, val_w
        for mode, (opcode, op_sel, op_sel_hi) in enumerate(FORMS):
            if sparse:
                code += [0x7E0802FF, INACTIVE, sop1(3, 126, 255), 0x55555555]
            w0, w1 = vop3p(opcode, 4, 257, 258, op_sel=op_sel, op_sel_hi=op_sel_hi)
            code += [w0 ^ first_modifier, w1 ^ second_modifier]
            if sparse:
                code += [sop1(3, 126, 193)]  # restore EXEC before storing
            offset = 4 * (case * len(FORMS) + mode)
            code += [0xE0701000 | offset, 0x80080408]  # buffer_store_dword v4, v8, s[32:35], 0 offen
        # Fill remaining 8 dwords of the 32-word row with 0? Only 24 dwords written per lane; write last 8 on case==0.
    return code + [0xBF810000]


def sext16(v):
    v &= 0xFFFF
    return v - 0x10000 if v >= 0x8000 else v


def eval_shift(opcode, a, b, op_sel, op_sel_hi):
    def pick(word, hi):
        return ((word >> 16) if hi else word) & 0xFFFF

    def lane(sel):
        amt = pick(a, sel & 1) & 15
        val = pick(b, (sel >> 1) & 1)
        if opcode == 0x04:
            return (val << amt) & 0xFFFF
        if opcode == 0x05:
            return (val >> amt) & 0xFFFF
        return (sext16(val) >> amt) & 0xFFFF

    return (lane(op_sel) | (lane(op_sel_hi) << 16)) & MASK


def reference(sparse=False):
    words = [SENTINEL] * 8
    for lane in range(64):
        for amt_w, val_w in CASES:
            for opcode, op_sel, op_sel_hi in FORMS:
                words.append(INACTIVE if sparse and lane % 2 else
                             eval_shift(opcode, amt_w ^ (lane % 32), val_w, op_sel, op_sel_hi))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    # Test valid encoding + 5 refused modifier/reserved-bit encodings:
    # clamp (w0[15]), op_sel[2] (w0[13]), neg_hi (w0[8]), neg_lo (w1[29]), nonzero src2 (w1[18]).
    for first, second in ((0, 0), (1 << 15, 0), (1 << 13, 0), (1 << 8, 0), (0, 1 << 29), (0, 1 << 18)):
        marker = request(work, root, program(first_modifier=first, second_modifier=second))
        output = marker.with_suffix('.cso')
        result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                                capture_output=True, text=True)
        if first or second:
            assert result.returncode and 'packed 16-bit shift operand modifiers' in result.stderr, result.stderr
            assert not output.exists()
        else:
            assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
            dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                                  text=True, check=True).stdout
            assert 'shl i32 ' in dump and 'lshr i32 ' in dump and 'ashr i32 ' in dump, dump
    assert reference() != reference(sparse=True)
    print('v_pk_shift16 DXIL verified; five modifier encodings refused; EXEC control differs')
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
    with tempfile.TemporaryDirectory(prefix='x2w_native_pk_shift16_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
