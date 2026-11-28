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
MOD_CASES = (
    # a, b, op_sel, op_sel_hi, neg_lo, neg_hi, clamp
    (0x38004000, 0x3C00BC00, 0, 3, 1, 2, 0),  # neg_lo[0] vs neg_hi[1] asymmetric
    (0xC0000001, 0x80003C00, 1, 2, 2, 1, 0),  # neg_lo[1] vs neg_hi[0] cross-lane
    (0x00008000, 0x80000000, 0, 3, 3, 0, 0),  # neg_lo both (+/-0) without neg_hi
    (0x7C007E01, 0xFC003800, 0, 3, 0, 3, 0),  # neg_hi both (+/-Inf, NaN payload)
    (0x8000C000, 0x3C000000, 0, 3, 0, 0, 1),  # clamp -2.0 -> +0.0 and -0.0 -> +0.0
    (0x00013800, 0x3C000000, 0, 3, 0, 0, 1),  # clamp preserves positive subnormal 0x0001 and 0.5..1.0
    (0x7C004000, 0xFC000000, 0, 3, 0, 0, 1),  # clamp >1.0 / +Inf -> 1.0, -Inf -> +0.0
    (0x7E013800, 0x7E027BFF, 0, 3, 0, 0, 1),  # clamp both-NaN -> +0.0, one-NaN picks numeric -> clamp
    (0x3800B800, 0xC0004000, 0, 3, 3, 2, 1),  # combined neg_lo + neg_hi + clamp
)
SENTINEL = 0x69696969
INACTIVE = 0x13579bdf
PROLOGUE = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020, 0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]


def packed(op, dst, lo, hi, *, first=257, second=258, bad=0, neg_lo=0, neg_hi=0, clamp=0):
    w0 = 0xCC004000 | (op << 16) | ((clamp & 1) << 15) | (lo << 11) | ((neg_hi & 7) << 8) | dst | bad
    w1 = ((neg_lo & 7) << 29) | (hi << 27) | (second << 9) | first
    return [w0, w1]


def compute_program(kind, *, bad=0, swap=False, flush=False, sparse=False, neg_lo=0, neg_hi=0, clamp=0, mods=False):
    code = list(PROLOGUE)
    if flush:
        code.append(0xBFA50003)  # s_denorm_mode: keep f32 denormals and flush f16
    rows = MOD_CASES if mods else CASES
    for slot, row in enumerate(rows):
        if mods:
            a, b, lo, hi, m_lo, m_hi, m_clamp = row
        else:
            a, b, lo, hi = row
            m_lo, m_hi, m_clamp = neg_lo, neg_hi, clamp
        source = {1: 23, 2: 255, 8: 128}.get(slot, 257) if not mods else {2: 23, 4: 255}.get(slot, 257)
        if source == 23:
            code += [0xBE9703FF, a]  # s_mov_b32 s23, literal
        elif source == 257:
            code += [0x7E0202FF, a]
        code += [0x7E0402FF, b]
        for j, op in enumerate(OPS[kind]):
            if sparse:
                code += [0x7E0802FF, INACTIVE, sop1(3, 126, 255), 0x55555555]
            code += packed(op, 4, hi if swap else lo, lo if swap else hi, first=source, bad=bad,
                           neg_lo=m_lo, neg_hi=m_hi, clamp=m_clamp)
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
        if kind == 'f16':
            _, run = compile_compute(cli, sdk, work, root, kind, mods=True)
            assert run.returncode == 0, run.stdout + run.stderr
            _, run = compile_compute(cli, sdk, work, root, kind, neg_lo=3, neg_hi=3, clamp=1)
            assert run.returncode == 0, run.stdout + run.stderr
            for bad_kwargs in ({'bad': 1 << 10}, {'bad': 1 << 13}, {'neg_lo': 4}):
                _, run = compile_compute(cli, sdk, work, root, kind, **bad_kwargs)
                assert run.returncode and 'packed 16-bit min/max encoding' in run.stderr, (kind, bad_kwargs, run.stderr)
        else:
            for bad in (1 << 15, 1 << 8, 1 << 13):
                _, run = compile_compute(cli, sdk, work, root, kind, bad=bad)
                assert run.returncode and 'packed 16-bit min/max encoding' in run.stderr, (kind, run.stderr)
    if not baseline:
        _, run = compile_compute(cli, sdk, work, root, 'f16', flush=True)
        assert run.returncode and 'denormal mode' in run.stderr, run.stderr
        print('six opcodes compiled (f16 with neg_lo/neg_hi/clamp); reserved/integer mods and f16 flush mode refused')


def half_result(kind, maximum, a, b, *, neg_a=False, neg_b=False, clamp=False):
    if kind == 'f16':
        if neg_a:
            a ^= 0x8000
        if neg_b:
            b ^= 0x8000
        x, y = (struct.unpack('<e', struct.pack('<H', v))[0] for v in (a, b))
        if math.isnan(x):
            res = 0x7e00 if math.isnan(y) else b
        elif math.isnan(y):
            res = a
        elif x == y == 0:
            res = a & b if maximum else a | b
        else:
            res = a if (x > y if maximum else x < y) else b
        if clamp:
            if (res & 0x7fff) > 0x7c00 or (res & 0x8000):
                return 0x0000
            if res > 0x3bff:
                return 0x3c00
        return res
    elif kind == 'i16':
        x, y = (struct.unpack('<h', struct.pack('<H', v))[0] for v in (a, b))
    else:
        x, y = a, b
    return a if (x > y if maximum else x < y) else b


def expected(kind, case, maximum, *, swap=False):
    if len(case) == 7:
        a, b, lo, hi, neg_lo, neg_hi, clamp = case
    else:
        a, b, lo, hi = case
        neg_lo, neg_hi, clamp = 0, 0, 0
    halves = []
    pairs = ((hi, neg_hi), (lo, neg_lo)) if swap else ((lo, neg_lo), (hi, neg_hi))
    for select, neg in pairs:
        x, y = (a >> (16 * (select & 1))) & 0xffff, (b >> (16 * (select >> 1))) & 0xffff
        halves.append(half_result(kind, maximum, x, y, neg_a=bool(neg & 1), neg_b=bool(neg & 2), clamp=bool(clamp)))
    return halves[0] | halves[1] << 16


def compute_replay(replay, sdk, work, root, kind, **options):
    source = request(work, root, compute_program(kind, **options))
    output = work / 'readback.bin'
    run = subprocess.run([replay, sdk, source, work / 'cache', '0', output, '--direct'])
    if run.returncode == 77:
        raise SystemExit(77)
    run.check_returncode()
    return struct.unpack('<2064I', output.read_bytes())


def expected_buffer(kind, *, swap=False, sparse=False, mods=False):
    rows = MOD_CASES if mods else CASES
    lane_words = [expected(kind, case, maximum, swap=swap) for case in rows for maximum in (False, True)]
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
    actual = compute_replay(replay, sdk, work, root, 'f16', mods=True)
    reference = expected_buffer('f16', mods=True)
    for index, (word, want) in enumerate(zip(actual, reference, strict=True)):
        assert word == want, ('f16_mods', index, hex(word), hex(want))
    print(f'f16 neg_lo/neg_hi/clamp: {len(MOD_CASES)}x2 Compute readback cases matched across 2064 words')
    actual = compute_replay(replay, sdk, work, root, 'f16', swap=True)
    assert actual == expected_buffer('f16', swap=True)
    count = sum(a != b for a, b in zip(actual, expected_buffer('f16'), strict=True))
    assert count > 0
    print(f'Compute swapped selectors: {count} readback words differ from the original reference')
    actual = compute_replay(replay, sdk, work, root, 'f16', sparse=True)
    assert actual == expected_buffer('f16', sparse=True)
    print('Compute sparse EXEC: inactive lanes kept their original packed word')


def pixel_program(kind, case):
    if len(case) == 7:
        a, b, lo, hi, neg_lo, neg_hi, clamp = case
    else:
        a, b, lo, hi = case
        neg_lo, neg_hi, clamp = 0, 0, 0
    code = [0x7E0202FF, a, 0x7E0402FF, b]
    for j, op in enumerate(OPS[kind]):
        src, low, high = 4 + j, 8 + 2 * j, 9 + 2 * j
        code += packed(op, src, lo, hi, neg_lo=neg_lo, neg_hi=neg_hi, clamp=clamp)
        code += [vop2(0x1b, low, 255, src), 65535, vop2(0x16, high, 144, src)]
        # Each unsigned halfword is exactly representable as binary32, even
        # if the packed result as a whole would encode a float NaN.
        code += [0x7E000C00 | reg << 17 | (256 + reg) for reg in (low, high)]
    return patched_pixel(len(code), code, export=(0xF800180F, 0x0B0A0908))


def draw_pixel(cli, sdk, renderer, adapter, work):
    vs = compile_hlsl(str(Path(sdk) / 'dxc.exe'), 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', work)
    for kind in OPS:
        indices = (0, 1, 3, 7)
        cases = [(f'{kind}-{index}', CASES[index]) for index in indices]
        if kind == 'f16':
            cases += [(f'f16-mod-{index}', case) for index, case in enumerate(MOD_CASES)]
        for label, case in cases:
            shader = work / f'{label}.ps'
            shader.write_bytes(pixel_program(kind, case))
            cso = shader.with_suffix('.cso')
            build = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-ps', shader, cso],
                                   capture_output=True, text=True, timeout=30)
            assert build.returncode == 0, (label, build.stdout + build.stderr)
            minimum, maximum = (expected(kind, case, flag) for flag in (False, True))
            reference = work / 'expected.bin'
            reference.write_bytes(struct.pack('<4f', minimum & 0xffff, minimum >> 16, maximum & 0xffff, maximum >> 16))
            run = subprocess.run([renderer, adapter, '--startup-pixel-words', vs, cso, '1', '1', reference],
                                 capture_output=True, text=True, timeout=30)
            if run.returncode == 77:
                raise SystemExit(77)
            assert run.returncode == 0, (label, run.stdout, run.stderr)
            if 'debug_layer=enabled' in run.stdout:
                assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
        print(f'{kind}: {len(cases)} Pixel cases matched for both packed results')


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
