"""Direct compute SOP1 s_sext_i32_i8 and s_sext_i32_i16 (sign extension preserving SCC)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request
from native_scalar_lds_check import sop1

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969

# (value, incoming_scc_for_i8, incoming_scc_for_i16)
CASES = (
    (0x00000000, 1, 0),  # zero result must preserve SCC=1; zero must preserve SCC=0
    (0x0000007F, 0, 1),  # max positive i8, positive i16
    (0x00000080, 0, 1),  # min negative i8 (0xFFFFFF80), positive i16 (0x00000080)
    (0x123456FF, 1, 0),  # upper junk + -1 i8, positive i16
    (0x12347FFF, 0, 1),  # max positive i16, -1 i8
    (0x12348000, 1, 0),  # min negative i16 (0xFFFF8000), 0 i8 (preserve SCC=1 and SCC=0)
    (0x8765807F, 0, 1),  # negative i16, positive i8
    (0xFFFFFFFF, 0, 0),  # all ones (-1 for both) while preserving SCC=0
)


def sopc_set_scc(val):
    # s_cmp_eq_i32 0, 0 -> SCC=1; s_cmp_eq_i32 0, 1 -> SCC=0
    return 0xBF008080 if val else 0xBF008180


def program():
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for ci, (val, scc8, scc16) in enumerate(CASES):
        code += [sop1(3, 6, 255), val]  # s_mov_b32 s6, val
        # s_sext_i32_i8 (SOP1 0x19) preserving incoming SCC
        code += [
            sopc_set_scc(scc8),
            sop1(0x19, 8, 6),           # s_sext_i32_i8 s8, s6
            sop1(3, 9, 253),            # s_mov_b32 s9, src_scc
            0x7E080208,                 # v_mov_b32 v4, s8
            0xE0701000 | (4 * (ci * 4 + 0)), 0x80080408,
            0x7E080209,                 # v_mov_b32 v4, s9 (SCC)
            0xE0701000 | (4 * (ci * 4 + 1)), 0x80080408,
        ]
        # s_sext_i32_i16 (SOP1 0x1a) preserving incoming SCC
        code += [
            sopc_set_scc(scc16),
            sop1(0x1A, 8, 6),           # s_sext_i32_i16 s8, s6
            sop1(3, 9, 253),            # s_mov_b32 s9, src_scc
            0x7E080208,                 # v_mov_b32 v4, s8
            0xE0701000 | (4 * (ci * 4 + 2)), 0x80080408,
            0x7E080209,                 # v_mov_b32 v4, s9 (SCC)
            0xE0701000 | (4 * (ci * 4 + 3)), 0x80080408,
        ]
    return code + [0xBF810000]


def sext(val, bits, zero_extend=False):
    m = (1 << bits) - 1
    v = val & m
    if zero_extend:
        return v
    sign = 1 << (bits - 1)
    return (v - (1 << bits)) & MASK if (v & sign) else v


def reference(zero_extend=False, clobber_scc=False):
    words = [SENTINEL] * 8
    for _lane in range(64):
        for val, scc8, scc16 in CASES:
            r8 = sext(val, 8, zero_extend=zero_extend)
            r16 = sext(val, 16, zero_extend=zero_extend)
            s8 = int(r8 != 0) if clobber_scc else scc8
            s16 = int(r16 != 0) if clobber_scc else scc16
            words.extend((r8, s8, r16, s16))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    marker = request(work, root, program())
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                          text=True, check=True).stdout
    assert 'ashr i32 ' in dump and 'shl i32 ' in dump, dump
    assert reference() != reference(zero_extend=True)
    assert reference() != reference(clobber_scc=True)
    print('s_sext_i32_i8/i16 DXIL verified; zero-extend and clobber-SCC controls differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    expected = reference()
    actual = struct.unpack(f'<{len(expected)}I', output.read_bytes())
    mismatch = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
    if mismatch is not None:
        raise AssertionError(f'word {mismatch}: got {actual[mismatch]:08x}, expected {expected[mismatch]:08x}')
    print('s_sext_i32_i8/i16: 2048 result/SCC words and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_s_sext_i32_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
