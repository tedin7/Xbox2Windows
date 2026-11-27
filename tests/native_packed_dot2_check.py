"""Read back gfx1013 signed and unsigned 16-bit dot products in direct compute."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
from native_scalar_lds_check import sop1

MASK = 0xFFFFFFFF
INT_MIN, INT_MAX = -(1 << 31), (1 << 31) - 1
SENTINEL, INACTIVE = 0x69696969, 0x13579BDF
CASES = (
    (0xFFFF0002, 0x0003FFFC, 13),
    (0x80008000, 0x80008000, 0x80000000),  # intermediate signed overflow cancels
    (0x80008000, 0x80008000, 0x7FFFFFFF),  # final signed positive overflow
    (0x80008000, 0x7FFF7FFF, 0x80000000),  # final signed negative overflow
    (MASK, MASK, MASK),                    # unsigned total carries twice
    (0x80000000, 0x80000000, 0x3FFFFFFF),  # exact INT_MAX
    (0x80000000, 0x7FFF0000, 0xBFFF8000),  # exact INT_MIN
    (0xFFFF0000, 0xFFFF0000, 0x0001FFFE),  # exact UINT_MAX
)
FORMS = ((True, False), (True, True), (False, False), (False, True))


def program(*, sparse=False, drop_clamp=False, low_modifier=0, high_modifier=0):
    # v8 is the byte offset of this lane's 32-word output row.
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for case, (a, b, acc) in enumerate(CASES):
        code += [vop2(0x1D, 1, 255, 0), a,       # v_xor_b32 v1, a, v0
                 0x7E0402FF, b, 0x7E0602FF, acc]  # v_mov_b32 v2, b; v_mov_b32 v3, acc
        for form, (signed, clamp) in enumerate(FORMS):
            if sparse:
                code += [0x7E0802FF, INACTIVE, sop1(3, 126, 255), 0x55555555]
            opcode = 0x14 if signed else 0x15
            word = 0xCC004004 | (opcode << 16) | ((clamp and not drop_clamp) << 15)
            operands = 0x1C000000 | (259 << 18) | (258 << 9) | 257
            code += [word ^ low_modifier, operands ^ high_modifier]
            if sparse:
                code += [sop1(3, 126, 193)]  # restore EXEC before storing
            offset = 4 * (case * len(FORMS) + form)
            code += [0xE0701000 | offset, 0x80080408]  # buffer_store_dword v4, v8, s[32:35], 0 offen
    return code + [0xBF810000]


def signed_word(value, width):
    return value - (1 << width) if value >> (width - 1) else value


def dot(a, b, acc, signed, clamp, *, early_clamp=False):
    lo, hi = (INT_MIN, INT_MAX) if signed else (0, MASK)
    x = (a & 65535, a >> 16)
    y = (b & 65535, b >> 16)
    if signed:
        x = tuple(signed_word(value, 16) for value in x)
        y = tuple(signed_word(value, 16) for value in y)
        acc = signed_word(acc, 32)
    product = x[0] * y[0] + x[1] * y[1]
    if clamp and early_clamp:
        product = max(lo, min(hi, product))
    total = product + acc
    if clamp:
        total = max(lo, min(hi, total))
    return total & MASK


def reference(sparse=False, drop_clamp=False):
    words = [SENTINEL] * 8
    for lane in range(64):
        for a, b, acc in CASES:
            for signed, clamp in FORMS:
                words.append(INACTIVE if sparse and lane % 2 else
                             dot(a ^ (lane % 32), b, acc, signed, clamp and not drop_clamp))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    for low, high in ((0, 0), (1 << 11, 0), (1 << 14, 0), (0, 1 << 27),
                      (1 << 8, 0), (0, 1 << 29)):
        marker = request(work, root, program(low_modifier=low, high_modifier=high))
        output = marker.with_suffix('.cso')
        run = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                             capture_output=True, text=True)
        if low or high:
            assert run.returncode and 'packed halfword dot operand modifiers' in run.stderr, run.stderr
            assert not output.exists()
        else:
            assert run.returncode == 0 and output.is_file(), (run.stdout, run.stderr)
            dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                                  text=True, check=True).stdout
            assert 'mul i32' in dump and 'add i64' not in dump, dump
    assert dot(*CASES[1], True, True) == 0
    assert dot(*CASES[1], True, True, early_clamp=True) == MASK
    assert (2 * 65535 ** 2 + MASK) >> 32 == 2
    print('32-bit DXIL verified; five modifier encodings refused; the exact cancellation catches early clamp')
    return 0


def hardware(replay, sdk, work, root):
    for sparse, drop_clamp in ((False, False), (True, False), (False, True)):
        marker = request(work, root, program(sparse=sparse, drop_clamp=drop_clamp))
        output = work / 'out.bin'
        run = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
        if run.returncode == 77:
            return 77
        run.check_returncode()
        expected = reference(sparse, drop_clamp)
        actual = struct.unpack(f'<{len(expected)}I', output.read_bytes())
        for i, (word, wanted) in enumerate(zip(actual, expected)):
            if word != wanted:
                raise AssertionError(f'sparse={sparse} drop_clamp={drop_clamp} word {i}: '
                                     f'got {word:08x}, expected {wanted:08x}')
        if drop_clamp:
            changed = sum(word != clamped for word, clamped in zip(actual, reference()))
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
