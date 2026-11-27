"""Direct compute SOP2 s_lshl{1,2,3,4}_add_u32 (D = (S0 << k) + S1, SCC = 64-bit unsigned overflow)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request
from native_scalar_lds_check import sop1

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969

# (s0, s1) covering no-carry, add-carry-only, shift-out-overflow-only (s1=0), and exact 2^32-1 / 2^32 boundaries.
CASES = (
    (0x00000005, 0x00000003),  # small positive, no carry
    (0x80000000, 0x00000000),  # shift-out overflow only (k=1..4 -> SCC=1, D=0)
    (0x40000000, 0x00000000),  # k=1 -> SCC=0; k=2..4 -> shift-out overflow SCC=1
    (0x10000000, 0x00000000),  # k=1..3 -> SCC=0; k=4 -> shift-out overflow SCC=1
    (0x0FFFFFFF, 0x0000000F),  # k=4: (0x0FFFFFFF<<4)+15 = 0xFFFFFFFF (exact max, SCC=0)
    (0x0FFFFFFF, 0x00000010),  # k=4: (0x0FFFFFFF<<4)+16 = 0x100000000 (add carry, SCC=1, D=0)
    (0x01234567, 0xF0000000),  # add carry for k=2..4, no carry for k=1
    (0xFFFFFFFF, 0xFFFFFFFF),  # both shift-out and add carry
)
# SOP2 opcodes 0x2e..0x31 -> k = 1..4
SHIFTS = ((0x2E, 1), (0x2F, 2), (0x30, 3), (0x31, 4))


def sop2(op, sdst, src0, src1):
    return 0x80000000 | ((op & 0x7F) << 23) | ((sdst & 0x7F) << 16) | ((src1 & 0xFF) << 8) | (src0 & 0xFF)


def program_batch(batch):
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for ci, (s0, s1) in enumerate(batch):
        code += [sop1(3, 6, 255), s0,  # s_mov_b32 s6, s0
                 sop1(3, 7, 255), s1]  # s_mov_b32 s7, s1
        for si, (op, _k) in enumerate(SHIFTS):
            code += [
                sop2(op, 8, 6, 7),         # s_lshl{k}_add_u32 s8, s6, s7
                sop1(3, 9, 253),           # s_mov_b32 s9, src_scc
                0x7E080208,                # v_mov_b32 v4, s8
                0xE0701000 | (4 * (ci * 8 + si * 2)), 0x80080408,
                0x7E080209,                # v_mov_b32 v4, s9 (SCC)
                0xE0701000 | (4 * (ci * 8 + si * 2 + 1)), 0x80080408,
            ]
    return code + [0xBF810000]


def ref_op(s0, s1, k, ignore_shift_overflow=False):
    full = ((s0 & MASK) << k) + (s1 & MASK)
    d = full & MASK
    if ignore_shift_overflow:
        shifted = ((s0 & MASK) << k) & MASK
        scc = int(((shifted + (s1 & MASK)) & MASK) < shifted)
    else:
        scc = int(full >= (1 << 32))
    return d, scc


def reference(batch, ignore_shift_overflow=False):
    words = [SENTINEL] * 8
    for _lane in range(64):
        for s0, s1 in batch:
            for _op, k in SHIFTS:
                d, scc = ref_op(s0, s1, k, ignore_shift_overflow=ignore_shift_overflow)
                words.extend((d, scc))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    for bi, batch in enumerate((CASES[:4], CASES[4:])):
        marker = request(work, root, program_batch(batch))
        output = marker.with_suffix('.cso')
        result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                                capture_output=True, text=True)
        assert result.returncode == 0 and output.is_file(), (bi, result.stdout, result.stderr)
        dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                              text=True, check=True).stdout
        assert 'shl i32 ' in dump and 'icmp ult i32 ' in dump, dump
    assert reference(CASES[:4]) != reference(CASES[:4], ignore_shift_overflow=True)
    print('s_lshl1..4_add_u32 DXIL verified; shift-out overflow SCC control differs')
    return 0


def hardware(replay, sdk, work, root):
    for bi, batch in enumerate((CASES[:4], CASES[4:])):
        marker = request(work, root, program_batch(batch))
        output = work / f'out_{bi}.bin'
        result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        expected = reference(batch)
        actual = struct.unpack(f'<{len(expected)}I', output.read_bytes())
        mismatch = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
        if mismatch is not None:
            raise AssertionError(f'batch={bi} word {mismatch}: got {actual[mismatch]:08x}, '
                                 f'expected {expected[mismatch]:08x}')
        print(f'batch={bi}: 2048 result/SCC words and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_s_lshl_add_u32_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
