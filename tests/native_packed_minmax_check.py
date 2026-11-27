"""Source-built direct DXIL checks for the six packed 16-bit min/max opcodes."""
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_direct_container_check import patched_pixel
from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
from native_scalar_lds_check import sop1
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

OPS = {'i16': (8, 7), 'u16': (13, 12), 'f16': (17, 18)}
CASES = (
    (0x00008000, 0x80000000, 0, 3),
    (0x8000FFFF, 0x7FFF0001, 1, 2),
    (0x7E013C00, 0x4000FE00, 0, 3),
    (0x7C010001, 0xFE7F8001, 0, 3),
    (0xFC007C00, 0x7BFFFBFF, 3, 0),
    (0x80010000, 0x00017C00, 2, 1),
    (0xFFFF0000, 0x0000FFFF, 0, 3),
    (0x4400C000, 0x3C003800, 1, 1),
    (0x00000000, 0x00018000, 3, 2),
)
SENTINEL = 0x69696969
INACTIVE = 0x13579bdf
PROLOGUE = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020, 0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]


def packed(op, dst, lo, hi, *, first=257, second=258, bad=0):
    return [0xCC004000 | op << 16 | lo << 11 | dst | bad, first | second << 9 | hi << 27]


def compute_program(kind, *, bad=0, swap=False, flush=False, sparse=False):
    code = list(PROLOGUE)
    if flush:
        code.append(0xBFA50003)  # s_denorm_mode: keep f32 denormals and flush f16
    for slot, (a, b, lo, hi) in enumerate(CASES):
        source = {1: 23, 2: 255, 8: 128}.get(slot, 257)
        if source == 23:
            code += [0xBE9703FF, a]  # s_mov_b32 s23, literal
        elif source == 257:
            code += [0x7E0202FF, a]
        code += [0x7E0402FF, b]
        for j, op in enumerate(OPS[kind]):
            if sparse:
                code += [0x7E0802FF, INACTIVE, sop1(3, 126, 255), 0x55555555]
            code += packed(op, 4, hi if swap else lo, lo if swap else hi, first=source, bad=bad)
            if source == 255:
                code.append(a)
            if sparse:
                code.append(sop1(3, 126, 193))
            code += [0xE0701000 | (slot * 2 + j) * 4, 0x80080408]
    return code + [0xBF810000]


def compile_compute(cli, sdk, work, root, kind, **options):
    source = request(work, root, compute_program(kind, **options))
    output = source.with_suffix('.cso')
    run = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', source, output],
                         capture_output=True, text=True, timeout=30)
    return output, run


def root_signature(sdk, work):
    hlsl = work / 'root.hlsl'
    hlsl.write_text('#define RS "DescriptorTable(UAV(u0,numDescriptors=8))"')
    root = work / 'root.bin'
    subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'RS', '-Fo', root, hlsl], check=True)
    return root.read_bytes()


def controls(cli, sdk, work, root, baseline=False):
    for kind in OPS:
        _, run = compile_compute(cli, sdk, work, root, kind)
        if baseline:
            assert run.returncode, (kind, run.stdout)
            print(kind, run.stderr.strip())
            continue
        assert run.returncode == 0, run.stdout + run.stderr
        for bad in (1 << 15, 1 << 8, 1 << 13):
            _, run = compile_compute(cli, sdk, work, root, kind, bad=bad)
            assert run.returncode and 'packed 16-bit min/max encoding' in run.stderr, (kind, run.stderr)
    if not baseline:
        _, run = compile_compute(cli, sdk, work, root, 'f16', flush=True)
        assert run.returncode and 'denormal mode' in run.stderr, run.stderr
        print('six opcodes compiled; clamp, negation, noncanonical third selector and f16 flush mode refused')


def half_result(kind, maximum, a, b):
    if kind == 'f16':
        x, y = (struct.unpack('<e', struct.pack('<H', v))[0] for v in (a, b))
        if math.isnan(x):
            return 0x7e00 if math.isnan(y) else b
        if math.isnan(y):
            return a
        if x == y == 0:
            return a & b if maximum else a | b
    elif kind == 'i16':
        x, y = (struct.unpack('<h', struct.pack('<H', v))[0] for v in (a, b))
    else:
        x, y = a, b
    return a if (x > y if maximum else x < y) else b


def expected(kind, case, maximum, *, swap=False):
    a, b, lo, hi = case
    halves = []
    for select in (hi, lo) if swap else (lo, hi):
        x, y = (a >> (16 * (select & 1))) & 0xffff, (b >> (16 * (select >> 1))) & 0xffff
        halves.append(half_result(kind, maximum, x, y))
    return halves[0] | halves[1] << 16


def compute_replay(replay, sdk, work, root, kind, **options):
    source = request(work, root, compute_program(kind, **options))
    output = work / 'readback.bin'
    run = subprocess.run([replay, sdk, source, work / 'cache', '0', output, '--direct'])
    if run.returncode == 77:
        raise SystemExit(77)
    run.check_returncode()
    return struct.unpack('<2064I', output.read_bytes())


def expected_buffer(kind, *, swap=False, sparse=False):
    lane_words = [expected(kind, case, maximum, swap=swap) for case in CASES for maximum in (False, True)]
    unused = [SENTINEL] * (32 - len(lane_words))
    inactive = [INACTIVE] * len(lane_words) if sparse else lane_words
    pair = lane_words + unused + inactive + unused
    return (SENTINEL,) * 8 + tuple(pair * 32) + (SENTINEL,) * 8


def draw_compute(replay, sdk, work, root):
    for kind in OPS:
        actual = compute_replay(replay, sdk, work, root, kind)
        reference = expected_buffer(kind)
        for index, (word, want) in enumerate(zip(actual, reference, strict=True)):
            assert word == want, (kind, index, hex(word), hex(want))
        print(f'{kind}: 2064 Compute readback words matched')
    actual = compute_replay(replay, sdk, work, root, 'f16', swap=True)
    assert actual == expected_buffer('f16', swap=True)
    count = sum(a != b for a, b in zip(actual, expected_buffer('f16'), strict=True))
    assert count > 0
    print(f'Compute swapped selectors: {count} readback words differ from the original reference')
    actual = compute_replay(replay, sdk, work, root, 'f16', sparse=True)
    assert actual == expected_buffer('f16', sparse=True)
    print('Compute sparse EXEC: inactive lanes kept their original packed word')


def pixel_program(kind, case):
    a, b, lo, hi = case
    code = [0x7E0202FF, a, 0x7E0402FF, b]
    for j, op in enumerate(OPS[kind]):
        src, low, high = 4 + j, 8 + 2 * j, 9 + 2 * j
        code += packed(op, src, lo, hi)
        code += [vop2(0x1b, low, 255, src), 65535, vop2(0x16, high, 144, src)]
        # Each unsigned halfword is exactly representable as binary32, even
        # if the packed result as a whole would encode a float NaN.
        code += [0x7E000C00 | reg << 17 | (256 + reg) for reg in (low, high)]
    return patched_pixel(len(code), code, export=(0xF800180F, 0x0B0A0908))


def draw_pixel(cli, sdk, renderer, adapter, work):
    vs = compile_hlsl(str(Path(sdk) / 'dxc.exe'), 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', work)
    for kind in OPS:
        for index in (0, 1, 3, 7):
            case = CASES[index]
            shader = work / f'{kind}-{index}.ps'
            shader.write_bytes(pixel_program(kind, case))
            cso = shader.with_suffix('.cso')
            build = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-ps', shader, cso],
                                   capture_output=True, text=True, timeout=30)
            assert build.returncode == 0, build.stdout + build.stderr
            minimum, maximum = (expected(kind, case, flag) for flag in (False, True))
            reference = work / 'expected.bin'
            reference.write_bytes(struct.pack('<4f', minimum & 0xffff, minimum >> 16, maximum & 0xffff, maximum >> 16))
            run = subprocess.run([renderer, adapter, '--startup-pixel-words', vs, cso, '1', '1', reference],
                                 capture_output=True, text=True, timeout=30)
            if run.returncode == 77:
                raise SystemExit(77)
            assert run.returncode == 0, (kind, index, run.stdout, run.stderr)
            if 'debug_layer=enabled' in run.stdout:
                assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
        print(f'{kind}: four Pixel cases matched for both packed results')


if __name__ == '__main__':
    mode, tool, sdk, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        if mode == 'pixel':
            draw_pixel(tool, sdk, *args, work)
        elif mode == 'compute':
            draw_compute(tool, sdk, work, root_signature(sdk, work))
        else:
            controls(tool, sdk, work, root_signature(sdk, work), baseline=mode == 'baseline')
