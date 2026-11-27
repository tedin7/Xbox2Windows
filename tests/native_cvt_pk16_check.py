"""v_cvt_pk_u16_u32 and v_cvt_pk_i16_i32 in the direct compute path (#2995).

  native_cvt_pk16_check.py controls <compile_dxil> <sdk bin dir>
  native_cvt_pk16_check.py hardware <native_store_replay> <sdk bin dir>

Both saturate S0 and S1 to 16 bits and pack S0 into the low and S1 into the high half (gfx2hlsl.py's VOP3 table):
v_cvt_pk_u16_u32 takes min(S, 0xffff) of the unsigned values, v_cvt_pk_i16_i32 clamps the signed values to
[-32768, 32767]. A nonzero unused SRC2 field and op_sel are refused (gfx2hlsl.py refuses the SRC2 field). The VOP3 words were assembled with
LLVM 15.0.7 (clang -cc1as, gfx1013); the v_mov and store words follow the encodings of native_align_bfm_check.py.

Each operand pair below is set by v_mov literals and converted by both instructions; every lane of two groups of
32 stores the sixteen results. The pairs cross both bounds of both forms. References that truncate instead of
saturating and that apply the unsigned bound to the signed form must each differ from the expected words.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

PAIRS = [(0, 0xFFFF), (0x10000, 0xFFFFFFFF), (0xFFFF8000, 0xFFFF7FFF), (0x7FFF, 0x8000),
         (0x12345, 0x7FFFFFFF), (0x80000000, 5), (0xFFFFFFFE, 0xFFFE), (1, 0x00010001)]
# s_mov_b32 s20, s4 / s_mov_b32 s21, 0 / s_load_dwordx4 s[32:35], s[20:21], 0x20 / s_lshl_b32 s7, s5, 11 /
# v_lshlrev_b32 v8, 6, v0 / v_add_nc_u32 v8, s7, v8: 64 bytes per lane, 2048 per group of 32
PROLOGUE = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020, 0x8F078B05, 0x34100086, 0x4A101007]
CVT_U16, CVT_I16 = (0xD76A0004, 0x00020501), (0xD76B0005, 0x00020501)  # v4 / v5 from v1, v2
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064


def program(src2=0):
    code = PROLOGUE + [0xBF8CC07F]  # s_waitcnt lgkmcnt(0) before the stores use s[32:35]
    for index, (x, y) in enumerate(PAIRS):
        code += [0x7E0202FF, x, 0x7E0402FF, y,
                 CVT_U16[0], CVT_U16[1] | src2 << 18, CVT_I16[0], CVT_I16[1] | src2 << 18]
        for slot, vgpr in enumerate((4, 5)):
            code += [0xE0701000 | (8 * index + 4 * slot), 0x80080008 | vgpr << 8]  # buffer_store_dword v, v8, s[32:35]
    return code + [0xBF810000]


def signed(x):
    return x - (1 << 32) if x & 0x80000000 else x


def pack(x, y, truncate=False, unsigned_bound=False):
    if truncate:
        return (x & 0xFFFF) | ((y & 0xFFFF) << 16), (x & 0xFFFF) | ((y & 0xFFFF) << 16)
    u = min(x, 0xFFFF) | (min(y, 0xFFFF) << 16)
    clamp = (lambda v: min(v, 0x7FFF)) if unsigned_bound else (lambda v: max(-32768, min(32767, signed(v))))
    i = (clamp(x) & 0xFFFF) | ((clamp(y) & 0xFFFF) << 16)
    return u, i


def reference(**mutation):
    expected = [SENTINEL] * OUTPUT_WORDS
    for lane in range(64):
        base = GUARD_WORDS + lane * 16
        for index, (x, y) in enumerate(PAIRS):
            expected[base + 2 * index:base + 2 * index + 2] = pack(x, y, **mutation)
    return expected


def compile_program(cli, sdk, work, root, code):
    marker = request(work, root, code)
    return marker, subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker,
                                   marker.with_suffix('.cso')], capture_output=True, text=True)


def controls(cli, sdk, work, root):
    marker, result = compile_program(cli, sdk, work, root, program())
    assert result.returncode == 0 and marker.with_suffix('.cso').is_file(), (result.stdout, result.stderr)
    _, result = compile_program(cli, sdk, work, root, program(src2=3))
    assert result.returncode and 'with unused src2 field set' in result.stderr, (result.stdout, result.stderr)
    code = program()
    code[code.index(CVT_U16[0])] |= 1 << 11  # op_sel bit of src0
    _, result = compile_program(cli, sdk, work, root, code)
    assert result.returncode and 'with op_sel' in result.stderr, (result.stdout, result.stderr)
    expected = reference()
    for label, mutant in (('truncated', reference(truncate=True)), ('unsigned bound', reference(unsigned_bound=True))):
        assert mutant != expected, label
    print('compiled; a nonzero src2 field and op_sel refused; the truncating and unsigned-bound references differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    actual = list(struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes()))
    diff = [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(actual, reference())) if x != y]
    assert not diff, diff[:10]
    print('v_cvt_pk_u16_u32 and v_cvt_pk_i16_i32: all 2064 words matched (1024 results, guards and sentinels)')
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
