"""v_add_nc_i32, v_sub_nc_i32 and the clamp forms of v_add_nc_u32, v_sub_nc_u32 and v_subrev_nc_u32 in the direct
compute path (#3009).

  native_add_sub_sat_check.py controls <compile_dxil> <sdk bin dir>
  native_add_sub_sat_check.py hardware <native_store_replay> <sdk bin dir>

The signed forms wrap like the unsigned ones; clamp saturates, signed for v_*_nc_i32 and unsigned for v_*_nc_u32.
LLVM 15.0.7 selects exactly these clamp forms for llvm.sadd.sat, llvm.ssub.sat, llvm.uadd.sat and llvm.usub.sat
(clang -nogpulib -target amdgcn-- -mcpu=gfx1013 -O2 -S). The VOP3 words were assembled with LLVM 15.0.7
(clang -cc1as, gfx1013); the first prologue words follow native_dpp_quad_check.py, with v8 = 128 * lane and s7 = 4096 * group (32 words
per lane), and the store words follow native_lane_access_check.py.

Each case sets S0 and S1 by v_mov literals and stores seven results; every lane of two groups stores all cases.
Unclamped, sign-swapped (unsigned saturation for the signed forms and the reverse), saturate-toward-S1 and
unswapped-subrev references must each differ from the expected words.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

CASES = [
    (0x7FFFFFFF, 0x00000001),
    (0x80000000, 0x00000001),
    (0xFFFFFFFF, 0x80000000),
    (0x7FFFFFFF, 0xFFFFFFFF),
]
PROLOGUE = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020, 0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
# v_add_nc_i32 v4 / v_add_nc_i32 v5 clamp / v_sub_nc_i32 v6 / v_sub_nc_i32 v7 clamp / v_add_nc_u32_e64 v9 clamp /
# v_sub_nc_u32_e64 v10 clamp / v_subrev_nc_u32_e64 v11 clamp, each with S0 = v1 and S1 = v2.
OPS = [('add_i32', 0xD77F0004, 4), ('add_i32 clamp', 0xD77F8005, 5), ('sub_i32', 0xD7760006, 6),
       ('sub_i32 clamp', 0xD7768007, 7), ('add_u32 clamp', 0xD5258009, 9), ('sub_u32 clamp', 0xD526800A, 10),
       ('subrev_u32 clamp', 0xD527800B, 11)]
SOURCES = 0x00020501
NEG_SOURCES = 0x20020501  # the same operands with neg on S0
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
MASK = 0xFFFFFFFF


def program(sources=SOURCES):
    code = list(PROLOGUE)
    for index, (a, b) in enumerate(CASES):
        code += [0x7E0202FF, a, 0x7E0402FF, b]
        for _, word, _ in OPS:
            code += [word, sources]
        for slot, (_, _, vgpr) in enumerate(OPS):
            code += [0xE0701000 | (32 * index + 4 * slot), 0x80080008 | vgpr << 8]
    return code + [0xBF810000]


def signed(x):
    return x - (1 << 32) if x & 0x80000000 else x


def op(name, a, b, clamp=True, swap_sign=False, toward_b=False, subrev_swap=True):
    sat = name.endswith('clamp') and clamp
    is_signed = name.endswith('i32') or name.endswith('i32 clamp')
    if swap_sign:
        is_signed = not is_signed
    if name.startswith('subrev') and subrev_swap:
        a, b = b, a
    if is_signed:
        x, y = signed(a), signed(b)
        true = x + y if name.startswith('add') else x - y
        if sat and not -(1 << 31) <= true < (1 << 31):
            sign = (b if toward_b else a) & 0x80000000
            return 0x80000000 if sign else 0x7FFFFFFF
        return true & MASK
    true = a + b if name.startswith('add') else a - b
    if sat and true > MASK:
        return MASK
    if sat and true < 0:
        return 0
    return true & MASK


def reference(**mutation):
    expected = [SENTINEL] * OUTPUT_WORDS
    for lane in range(64):
        base = GUARD_WORDS + lane * 32
        for index, (a, b) in enumerate(CASES):
            for slot, (name, _, _) in enumerate(OPS):
                expected[base + 8 * index + slot] = op(name, a, b, **mutation)
    return expected


def compile_program(cli, sdk, work, root, code, name):
    marker = request(work, root, code)
    output = marker.with_suffix(name)
    return subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                          capture_output=True, text=True), output


def controls(cli, sdk, work, root):
    result, output = compile_program(cli, sdk, work, root, program(), '.cso')
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    assert 'add i32' in dump and 'sub i32' in dump and 'select i1' in dump, dump
    result, _ = compile_program(cli, sdk, work, root, program(NEG_SOURCES), '.neg.cso')
    assert result.returncode != 0 and 'v_add_nc_i32 operand modifiers' in result.stdout + result.stderr, (
        result.stdout, result.stderr)
    expected = reference()
    for label, mutation in (('unclamped', {'clamp': False}), ('signedness swapped', {'swap_sign': True}),
                            ('saturated toward S1', {'toward_b': True}),
                            ('subrev without the swap', {'subrev_swap': False})):
        assert reference(**mutation) != expected, label
    print('compiled seven add/sub forms with saturating selects; a neg modifier is refused; the unclamped, '
          'signedness-swapped, saturate-toward-S1 and unswapped-subrev references differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    actual = list(struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes()))
    expected = reference()
    diff = [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(actual, expected)) if x != y]
    assert not diff, diff[:10]
    print('seven add/sub forms: all 2064 words matched')
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
