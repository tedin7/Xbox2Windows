"""Direct compute SOP1 s_quadmask_b32 and s_quadmask_b64."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request
from native_scalar_lds_check import sop1

SENTINEL = 0x69696969

CASES_32 = (
    0x00000000,  # all zero -> 0x00, SCC=0
    0x00000001,  # nibble 0 bit 0 -> 0x01, SCC=1
    0x00000008,  # nibble 0 bit 3 -> 0x01, SCC=1
    0x80000000,  # nibble 7 bit 3 -> 0x80, SCC=1
    0x0F00F00F,  # nibbles 0, 3, 6 -> 0x49, SCC=1
    0x10204080,  # odd nibbles 1, 3, 5, 7 -> 0xAA, SCC=1
    0x01020408,  # even nibbles 0, 2, 4, 6 -> 0x55, SCC=1
    0xFFFFFFFF,  # all nibbles -> 0xFF, SCC=1
)

CASES_64 = (
    (0x00000000, 0x00000000),  # zero -> (0x0000, 0, SCC=0)
    (0x00000000, 0x00000001),  # hi-only non-zero -> (0x0100, 0, SCC=1)
    (0x80000001, 0x10204080),  # mixed lo and hi -> (0xAA81, 0, SCC=1)
    (0x01020408, 0xFFFFFFFF),  # all hi + even lo -> (0xFF55, 0, SCC=1)
)


def program():
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    # Words 0..15: 8 x s_quadmask_b32 (D, SCC)
    for ci, x in enumerate(CASES_32):
        code += [
            sop1(3, 6, 255), x,         # s_mov_b32 s6, x
            sop1(0x2C, 8, 6),           # s_quadmask_b32 s8, s6
            sop1(3, 9, 253),            # s_mov_b32 s9, src_scc
            0x7E080208,                 # v_mov_b32 v4, s8
            0xE0701000 | (4 * (ci * 2 + 0)), 0x80080408,
            0x7E080209,                 # v_mov_b32 v4, s9 (SCC)
            0xE0701000 | (4 * (ci * 2 + 1)), 0x80080408,
        ]
    # Words 16..31: 4 x s_quadmask_b64 (D.lo, D.hi, SCC, aliased-in-place D.lo)
    for ci, (lo, hi) in enumerate(CASES_64):
        base = 16 + ci * 4
        code += [
            sop1(3, 6, 255), lo,        # s_mov_b32 s6, lo
            sop1(3, 7, 255), hi,        # s_mov_b32 s7, hi
            sop1(0x2D, 8, 6),           # s_quadmask_b64 s[8:9], s[6:7]
            sop1(3, 10, 253),           # s_mov_b32 s10, src_scc
            sop1(0x2D, 6, 6),           # s_quadmask_b64 s[6:7], s[6:7] (aliased dst==src)
            0x7E080208,                 # v_mov_b32 v4, s8 (D.lo)
            0xE0701000 | (4 * (base + 0)), 0x80080408,
            0x7E080209,                 # v_mov_b32 v4, s9 (D.hi, must be 0)
            0xE0701000 | (4 * (base + 1)), 0x80080408,
            0x7E08020A,                 # v_mov_b32 v4, s10 (SCC)
            0xE0701000 | (4 * (base + 2)), 0x80080408,
            0x7E080206,                 # v_mov_b32 v4, s6 (aliased D.lo)
            0xE0701000 | (4 * (base + 3)), 0x80080408,
        ]
    return code + [0xBF810000]


def quadmask32(x, all_bits=False):
    out = 0
    for q in range(8):
        nib = (x >> (4 * q)) & 0xF
        if (nib == 0xF) if all_bits else (nib != 0):
            out |= 1 << q
    return out


def reference(all_bits=False, ignore_hi=False):
    words = [SENTINEL] * 8
    for _lane in range(64):
        for x in CASES_32:
            q = quadmask32(x, all_bits=all_bits)
            words.extend((q, int(q != 0)))
        for lo, hi in CASES_64:
            qlo = quadmask32(lo, all_bits=all_bits)
            qhi = 0 if ignore_hi else quadmask32(hi, all_bits=all_bits)
            out_lo = qlo | (qhi << 8)
            words.extend((out_lo, 0, int(out_lo != 0), out_lo))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    marker = request(work, root, program())
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                          text=True, check=True).stdout
    assert 'and i32 ' in dump and '286331153' in dump, dump
    for tag, bad, msg in (
        ('odd_dst', [sop1(0x2D, 7, 6), 0xBF810000], 'invalid s_quadmask_b64 destination'),
        ('odd_src', [sop1(0x2D, 8, 7), 0xBF810000], 'invalid s_quadmask_b64 source'),
    ):
        sub = work / tag
        sub.mkdir(exist_ok=True)
        bad_marker = request(sub, root, bad)
        bad_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', bad_marker,
                                  bad_marker.with_suffix('.cso')], capture_output=True, text=True)
        assert bad_res.returncode != 0 and msg in (bad_res.stdout + bad_res.stderr), (tag, bad_res.stdout, bad_res.stderr)
    assert reference() != reference(all_bits=True)
    assert reference() != reference(ignore_hi=True)
    print('s_quadmask_b32/b64 DXIL verified; odd dst/src refused; all-bits and ignore-hi controls differ')
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
    print('s_quadmask_b32/b64: 2048 result/SCC words and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_s_quadmask_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
