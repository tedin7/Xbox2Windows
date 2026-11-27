"""Direct compute signed/unsigned VOP3P byte dot products and accumulator saturation."""
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
    (0x01020304, 0x02050813, 29),
    (0x80808080, 0x80808080, 0x7FFFFFF0),  # signed positive overflow
    (0x80808080, 0x7F7F7F7F, 0x80000010),  # signed negative overflow
    (MASK, MASK, 0xFFFFFFF0),             # unsigned carry
    (0xFF80FE7F, 0x7F1002FE, 0x80010001),
    (0x80000000, 0x80000000, 0x7FFFBFFF),  # exact signed upper boundary at lane 0
    (0x80000000, 0x7F000000, 0x80003F80),  # exact signed lower boundary at lane 0
    (0x00000080, 0xFFFFFF00, MASK),        # exact unsigned upper boundary
)
FORMS = ((True, False), (True, True), (False, False), (False, True))


def program(*, sparse=False, drop_clamp=False, first_modifier=0, second_modifier=0):
    # Group index and local lane identify a 32-word row.
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for case, (a, b, acc) in enumerate(CASES):
        code += [vop2(0x1D, 1, 255, 0), a,       # v_xor_b32 v1, a, v0
                 0x7E0402FF, b, 0x7E0602FF, acc] # v_mov_b32 v2, b; v_mov_b32 v3, acc
        for mode, (signed, clamp) in enumerate(FORMS):
            if sparse:
                code += [0x7E0802FF, INACTIVE, sop1(3, 126, 255), 0x55555555]
            opcode = 0x16 if signed else 0x17
            word = 0xCC004004 | (opcode << 16) | ((clamp and not drop_clamp) << 15)
            operands = 0x1C000000 | (259 << 18) | (258 << 9) | 257
            code += [word ^ first_modifier, operands ^ second_modifier]
            if sparse:
                code += [sop1(3, 126, 193)]  # restore EXEC before storing
            offset = 4 * (case * len(FORMS) + mode)
            code += [0xE0701000 | offset, 0x80080408]  # buffer_store_dword v4, v8, s[32:35], 0 offen
    return code + [0xBF810000]


def as_signed(value, width):
    return value - (1 << width) if value >> (width - 1) else value


def dot(a, b, acc, signed, clamp):
    total = as_signed(acc, 32) if signed else acc
    for shift in range(0, 32, 8):
        x, y = (a >> shift) & 255, (b >> shift) & 255
        total += as_signed(x, 8) * as_signed(y, 8) if signed else x * y
    if clamp:
        lo, hi = (-(1 << 31), (1 << 31) - 1) if signed else (0, MASK)
        total = max(lo, min(hi, total))
    return total & MASK


def reference(sparse=False, ignore_clamp=False):
    words = [SENTINEL] * 8
    for lane in range(64):
        for a, b, acc in CASES:
            for signed, clamp in FORMS:
                words.append(INACTIVE if sparse and lane % 2 else
                             dot(a ^ (lane % 32), b, acc, signed, clamp and not ignore_clamp))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    for first, second in ((0, 0), (1 << 11, 0), (1 << 14, 0), (0, 1 << 27),
                          (1 << 8, 0), (0, 1 << 29)):
        marker = request(work, root, program(first_modifier=first, second_modifier=second))
        output = marker.with_suffix('.cso')
        result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                                capture_output=True, text=True)
        if first or second:
            assert result.returncode and 'packed byte dot operand modifiers' in result.stderr, result.stderr
            assert not output.exists()
        else:
            assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
            dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                                  text=True, check=True).stdout
            for opcode in (163, 164):
                assert dump.count(f'@dx.op.dot4AddPacked.i32(i32 {opcode},') == 2 * len(CASES), dump
    assert reference() != reference(ignore_clamp=True)
    assert reference() != reference(sparse=True)
    print('signed/unsigned dot4 DXIL verified; five modifier encodings refused; clamp and EXEC controls differ')
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
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
