"""Run direct packed binary16 arithmetic against an exact-rational reference."""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from f16_packed_model import h2f, half_op, round_exact
from native_direct_container_check import patched_pixel
from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
from native_scalar_lds_check import sop1
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

OPS = {'add': 0x0f, 'mul': 0x10, 'fma': 0x0e}
GUARD = 0x69696969
INACTIVE = 0x13579bdf
PROLOGUE = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020, 0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]


@dataclass(frozen=True)
class Case:
    a: int
    b: int
    c: int
    low: int = 0
    high: int = 7
    neg_lo: int = 0
    neg_hi: int = 0

    @property
    def inputs(self):
        return self.a, self.b, self.c


CASES = (
    # Both products are binary16 midpoints. A minimum subnormal addend
    # changes the exact rounded result but vanishes on a binary32 tie.
    Case(0x3c033c01, 0x3e003e00, 0x00018001),
    Case(0x6001bc01, 0x3e003e00, 0x80010001),
    Case(0x04000001, 0x38003800, 0x03ff0000),
    Case(0x7bfffbff, 0x40004000, 0x80000000),
    Case(0xbc003c00, 0x3c003c00, 0x3c00bc00),
    Case(0x80008000, 0xbc003c00, 0x80008000),
    Case(0x7c00fc00, 0x00003c00, 0x3c007c00),
    Case(0x7d017e01, 0x3c003c00, 0x3c003c00),
    Case(0x42003c00, 0xc4004000, 0x4400c000, low=5, high=2, neg_lo=1, neg_hi=6),
    Case(0x00018001, 0x35553c00, 0x7bff83ff, low=3, high=4, neg_lo=2, neg_hi=1),
)


def packed_instruction(name, dst, case, bad=0):
    active = 7 if name == 'fma' else 3
    # Set the conventional high selector for the unused third source.
    high = case.high if name == 'fma' else case.high | 4
    word0 = (0xcc000000 | OPS[name] << 16 | dst | (case.low & active) << 11 |
             (high & 4) << 12 | (case.neg_hi & active) << 8 | bad)
    last = 259 if name == 'fma' else 0
    word1 = 257 | 258 << 9 | last << 18 | (high & 3) << 27 | (case.neg_lo & active) << 29
    return [word0, word1]


def moves(case):
    return [word for index, value in enumerate(case.inputs, 1) for word in (0x7e0002ff | index << 17, value)]


def compute_program(cases, *, sparse=False, bad=0, mode=()):
    code = PROLOGUE + list(mode)
    for slot, case in enumerate(cases):
        code += moves(case)
        for j, op in enumerate(OPS):
            if sparse:
                code += [0x7E0802FF, INACTIVE, sop1(3, 126, 255), 0x55555555]
            code += packed_instruction(op, 4, case, bad)
            if sparse:
                code.append(sop1(3, 126, 193))
            code += [0xE0701000 | (slot * len(OPS) + j) * 4, 0x80080408]
    return code + [0xBF810000]


def root_signature(sdk, work):
    hlsl = work / 'root.hlsl'
    hlsl.write_text('#define RS "DescriptorTable(UAV(u0,numDescriptors=8))"')
    root = work / 'root.bin'
    subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'RS', '-Fo', root, hlsl], check=True)
    return root.read_bytes()


def compile_shader(cli, sdk, work, profile, source):
    output = source.with_suffix('.cso')
    run = subprocess.run([cli, sdk, work / 'cache', profile, source, output],
                         capture_output=True, text=True, timeout=30)
    return output, run


def controls(cli, sdk, work):
    root = root_signature(sdk, work)
    source = request(work, root, compute_program(CASES[:1]))
    output, run = compile_shader(cli, sdk, work, 'gfx1013-direct-cs@0', source)
    assert run.returncode == 0 and output.is_file(), run.stdout + run.stderr
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], check=True,
                          capture_output=True, text=True).stdout
    assert 'fadd float' in dump and 'fmul float' in dump and '@dx.op.unaryBits.i32(i32 33,' in dump, dump
    clamped_source = request(work, root, compute_program(CASES[:1], bad=0x8000))
    clamped_output, clamped_run = compile_shader(cli, sdk, work, 'gfx1013-direct-cs@0', clamped_source)
    assert clamped_run.returncode == 0 and clamped_output.is_file(), clamped_run.stdout + clamped_run.stderr
    for options, message in (
        ({'bad': 0x0400}, 'packed binary16 arithmetic encoding'),
        ({'bad': 0x2000}, 'packed binary16 arithmetic encoding'),
        ({'mode': (0xBFA50003,)}, 'binary16 rounding or denormal mode'),
        ({'mode': (0xBFA40004,)}, 'binary16 rounding or denormal mode'),
    ):
        source = request(work, root, compute_program(CASES[:1], **options))
        _, run = compile_shader(cli, sdk, work, 'gfx1013-direct-cs@0', source)
        assert run.returncode and message in run.stderr, (options, run.stderr)
    print('ADD/MUL and integer FMA compiled (including clamp); unsupported modifiers and modes were refused')


def operands(case, high):
    selected = case.high if high else case.low
    neg = case.neg_hi if high else case.neg_lo
    return tuple(((word >> (16 * ((selected >> i) & 1))) & 0xffff) ^ (((neg >> i) & 1) << 15)
                 for i, word in enumerate(case.inputs))


def expected(case, op):
    low, high = (half_op(op, *operands(case, h)) for h in (False, True))
    return low | high << 16


def double_rounded(case):
    def lane(high):
        a, b, c = map(h2f, operands(case, high))
        rounded_f32, = struct.unpack('<f', struct.pack('<f', a * b + c))
        return round_exact(Fraction(rounded_f32))
    return lane(False) | lane(True) << 16


def replay(tool, sdk, work, root, cases, sparse=False):
    source = request(work, root, compute_program(cases, sparse=sparse))
    output = work / 'readback.bin'
    run = subprocess.run([tool, sdk, source, work / 'cache', '0', output, '--direct'])
    if run.returncode == 77:
        raise SystemExit(77)
    run.check_returncode()
    return struct.unpack('<2064I', output.read_bytes())


def verify_compute(tool, sdk, work, root, cases=CASES, sparse=False):
    values = [expected(case, op) for case in cases for op in OPS]
    unused = [GUARD] * (32 - len(values))
    inactive = [INACTIVE] * len(values) if sparse else values
    pair = values + unused + inactive + unused
    want = (GUARD,) * 8 + tuple(pair * 32) + (GUARD,) * 8
    actual = replay(tool, sdk, work, root, cases, sparse)
    mismatch = next((i for i, (x, y) in enumerate(zip(actual, want, strict=True)) if x != y), None)
    assert mismatch is None, (mismatch, hex(actual[mismatch]), hex(want[mismatch]))
    return actual


def draw_compute(tool, sdk, work):
    root = root_signature(sdk, work)
    values = verify_compute(tool, sdk, work, root)
    verify_compute(tool, sdk, work, root, sparse=True)
    for i in (0, 1):
        offset = 8 + i * 3 + 2
        assert values[offset] == expected(CASES[i], 'fma') and values[offset] != double_rounded(CASES[i])
        print(f'FMA case {i}: exact {values[offset]:08x}, binary32 intermediate {double_rounded(CASES[i]):08x}')
    print('2064 words matched in each full and sparse EXEC dispatch')


def pixel_program(case, other_op):
    code = moves(case)
    for i, op in enumerate(('fma', other_op)):
        result, low, high = 4 + i, 8 + 2 * i, 9 + 2 * i
        code += packed_instruction(op, result, case)
        code += [vop2(0x1b, low, 255, result), 65535, vop2(0x16, high, 144, result)]
        code += [0x7E000C00 | reg << 17 | (256 + reg) for reg in (low, high)]
    return patched_pixel(len(code), code, export=(0xF800180F, 0x0B0A0908))


def draw_pixel(tool, sdk, renderer, adapter, work):
    vs = compile_hlsl(str(Path(sdk) / 'dxc.exe'), 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', work)
    for index, case in enumerate(CASES):
        other_op = 'add' if index % 2 == 0 else 'mul'
        shader = work / f'packed-{index}.ps'
        shader.write_bytes(pixel_program(case, other_op))
        cso, build = compile_shader(tool, sdk, work, 'gfx1013-container-ps', shader)
        assert build.returncode == 0, build.stdout + build.stderr
        fused, other = expected(case, 'fma'), expected(case, other_op)
        reference = work / 'expected.bin'
        reference.write_bytes(struct.pack('<4f', fused & 0xffff, fused >> 16, other & 0xffff, other >> 16))
        run = subprocess.run([renderer, adapter, '--startup-pixel-words', vs, cso, '1', '1', reference],
                             capture_output=True, text=True, timeout=30)
        if run.returncode == 77:
            raise SystemExit(77)
        assert run.returncode == 0, (index, run.stdout, run.stderr)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
    print(f'{len(CASES)} Pixel readbacks matched, including both exact FMA tie cases')


if __name__ == '__main__':
    mode, tool, sdk, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        if mode == 'controls':
            controls(tool, sdk, work)
        elif mode == 'compute':
            draw_compute(tool, sdk, work)
        elif mode == 'pixel':
            draw_pixel(tool, sdk, *args, work)
        else:
            raise ValueError(mode)
