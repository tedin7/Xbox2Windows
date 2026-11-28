"""Check that direct DXIL sampling uses the guest's consecutive-lane quads."""

import math
from pathlib import Path
import subprocess
import sys
import tempfile

from compute_gather4_l_check import FLOAT_TEXTURE_FLAG, HEIGHT, SENTINEL, WIDTH, bits, texels
from native_direct_container_check import compute, words
from native_gather_lod_check import render
from startup_compute_gather_l_check import reference as gather_reference
from startup_image_sample_c_check import LESS, POINT


CASES = (
    ((4, 1, 1), True),
    ((12, 1, 1), True),
    ((32, 1, 1), True),
    ((64, 1, 1), False),
    ((96, 1, 1), True),
    ((128, 1, 1), False),
    ((8, 4, 1), True),
    ((8, 4, 2), False),
)
MIP_COLORS = tuple(bits(value) for value in (1.0, 2.0, 4.0))
MIP_ZERO_FLAG = 1 << 22  # tests/compute_check.cpp texture_mip_zero_sampler_flag
DEPTH_FLAG = 1 << 23     # tests/compute_check.cpp texture_comparison_flag
POINT_MIP_SAMPLER = (0x12, 0xFFF000, 0x04500000, 0)


def vop2(op, dst, src0, src1):
    return (op << 25) | (dst << 17) | (src1 << 9) | src0


def setup(wave32, sampler):
    # One group at ID (0,0,0), with guest local IDs in v0/v1/v2.
    # The texture descriptor is at table + 0 and output at table + 0x20.
    code = [
        0xBEFE03C1 if wave32 else 0xBEFE04C1,  # s_mov_b32 exec_lo, -1 / s_mov_b64 exec, -1
        0xBE810380, 0xF40C0200, 0xFA000000,
        0xF4080800, 0xFA000020,
    ]
    for reg, value in enumerate(sampler, start=4):
        code += [0xBE8003FF | reg << 16, value]
    return code


def program(group, wave32, implicit):
    code = setup(wave32, POINT_MIP_SAMPLER)
    _, y, z = group
    if z > 1:
        code += [
            vop2(26, 2, 133, 2),      # v2 = z << 5
            vop2(28, 0, 258, 0),      # v0 |= v2
        ]
    if y > 1:
        code += [
            vop2(26, 1, 131, 1),      # v1 = y << 3
            vop2(28, 0, 257, 0),      # v0 |= v1, completing the linear index
        ]
    code += [
        vop2(26, 5, 130, 0),          # v5 = linear index * 4, the output byte offset
    ]
    if y > 1:
        # V ramps within each pair of guest rows. Consecutive quads step two lanes
        # in ddy, covering one texel (mip 0). Without flattening, the 2D spatial
        # quad steps eight lanes in ddy, covering four texels (mip 2).
        code += [
            0x7E0202F0,              # v_mov_b32 v1, 0.5 (U)
            vop2(27, 2, 143, 0),      # v2 = linear index & 15
            0x7E040D02,              # v_cvt_f32_u32 v2, v2
            vop2(8, 2, 255, 2), bits(1 / (2 * HEIGHT)),  # V
            0x7E060280,              # v_mov_b32 v3, 0 (explicit LOD)
        ]
    else:
        code += [
            vop2(22, 3, 130, 0),      # v3 = linear index >> 2
            vop2(27, 3, 131, 3),      # v3 &= 3
            vop2(19, 3, 130, 3),      # v3 = min(v3, 2), the expected mip
            vop2(26, 4, 151, 3),      # v4 = mip << 23
            vop2(37, 4, 255, 4), bits(1 / WIDTH),  # v4 = bits for (1 / WIDTH) * 2**mip
            vop2(27, 1, 129, 0),      # v1 = linear index & 1
            0x7E020D01,              # v_cvt_f32_u32 v1, v1
            vop2(8, 1, 260, 1),      # v1 = U, giving each quad ddx(U) = 2**mip / WIDTH
            vop2(27, 4, 136, 0),      # v4 = linear index & 8
            0x7E080D04,              # v_cvt_f32_u32 v4, v4
            vop2(8, 4, 255, 4), bits(0.0625),  # v4 = 0.5 when bit 3 of the index is set
            vop2(3, 1, 260, 1),      # Add an offset constant within each quad.
            0x7E0402F0,              # v_mov_b32 v2, 0.5 (V)
            0x7E060D03,              # v_cvt_f32_u32 v3, v3 (explicit LOD)
        ]
    code += [
        0xBF8CC07F,                  # s_waitcnt lgkmcnt(0)
        0xF0000108 | ((32 if implicit else 36) << 18), 0x00220401,
        0xBF8C3F70,                  # s_waitcnt vmcnt(0)
        0xE0701000, 0x80080405,      # buffer_store_dword v4, v5, s[32:35], 0 offen
        0xBF810000,
    ]
    return compute(words(*code), table=True, wave32=wave32, group=group)


def compile_shader(cli, sdk, work, group, wave32, implicit):
    name = '-'.join(map(str, (*group, int(wave32), int(implicit))))
    return compile_program(cli, sdk, work, name, program(group, wave32, implicit))


def compile_program(cli, sdk, work, name, binary):
    source = work / f'{name}.c67'
    source.write_bytes(binary)
    shader = source.with_suffix('.cso')
    run = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-cs', source, shader],
                         capture_output=True, text=True, timeout=30)
    return shader, run


def compile_pair(cli, sdk, work, group, wave32):
    for implicit in (True, False):
        shader, run = compile_shader(cli, sdk, work, group, wave32, implicit)
        assert run.returncode == 0 and shader.is_file(), (group, wave32, implicit, run.stdout, run.stderr)
        yield shader


def mip_payload(colors, channels=4):
    return [color for mip, color in enumerate(colors)
            for _ in range((WIDTH >> mip) * (HEIGHT >> mip) * channels)]


def write_case(path, expected, payload, flags=FLOAT_TEXTURE_FLAG):
    expected = [*expected, *([SENTINEL] * (320 - len(expected)))]
    path.write_bytes(words(flags | 1, *expected, *payload))


def write_reference(path, group):
    threads = math.prod(group)
    if group[1] > 1:
        expected = [MIP_COLORS[0]] * threads
    else:
        expected = [MIP_COLORS[min((lane // 4) % 4, 2)] for lane in range(threads)]
    write_case(path, expected, mip_payload(MIP_COLORS))


def operation_program(op, *, step=2 / WIDTH, operand=0.0, mask=1, components=1,
                      sampler=POINT_MIP_SAMPLER):
    code = setup(True, sampler)
    code += [
        vop2(26, 8, {1: 130, 2: 131, 4: 132}[components], 0),  # byte offset
        vop2(27, 7, 129, 0),      # lane & 1
        0x7E0E0D07,              # convert v7 to float
        vop2(8, 1, 255, 7), bits(step),
        vop2(3, 1, 240, 1),      # U = 0.5 + step * (lane & 1)
        0x7E0402F0,              # V = 0.5
        0x7E0002FF, bits(operand),
    ]
    address = 0 if op in (37, 40) else 1
    if op == 40:
        # Comparison reference before U and V, as in startup_image_sample_c_check.
        code += [vop2(8, 0, 255, 7), bits(0.25), vop2(3, 0, 240, 0)]
    if op in (36, 68):
        code += [0x7E060300]     # v_mov_b32 v3, v0: explicit LOD after U and V
    code += [
        0xBF8CC07F,
        0xF0000008 | op << 18 | mask << 8, 0x00220400 | address,
        0xBF8C3F70,
        {1: 0xE0701000, 2: 0xE0741000, 4: 0xE0781000}[components], 0x80080408,
        0xBF810000,
    ]
    return compute(words(*code), wave32=True, group=(4, 1, 1), table=True)


def operation_cases():
    colors = mip_payload(MIP_COLORS)
    yield 'bias-base', operation_program(32), [MIP_COLORS[1]] * 4, colors, FLOAT_TEXTURE_FLAG
    yield 'bias-plus-one', operation_program(37, operand=1.0), [MIP_COLORS[2]] * 4, colors, FLOAT_TEXTURE_FLAG
    yield 'bias-explicit', operation_program(36, operand=2.0), [MIP_COLORS[2]] * 4, colors, FLOAT_TEXTURE_FLAG

    comparison = (POINT | LESS, 0xFFF000, 0x04000000, 0)
    depth = mip_payload(tuple(map(bits, (0.25, 0.625, 1.0))), channels=1)
    yield 'compare', operation_program(40, sampler=comparison), [bits(1.0), bits(0.0)] * 2, depth, DEPTH_FLAG

    mip_zero = (0x12, 0, 0x04500000, 0)
    query = operation_program(96, step=4 / WIDTH, mask=3, components=2, sampler=mip_zero)
    yield 'get-lod', query, [bits(0.0), bits(2.0)] * 4, colors, FLOAT_TEXTURE_FLAG | MIP_ZERO_FLAG

    step = 1 / (4 * WIDTH)
    taps = gather_reference(0, 0, (0.5, 0.5))[:4]
    assert taps == gather_reference(0, 0, (0.5 + step, 0.5))[:4]
    for name, op in (('gather', 64), ('gather-explicit', 68)):
        binary = operation_program(op, step=step, components=4, sampler=mip_zero)
        yield name, binary, taps * 4, texels(), FLOAT_TEXTURE_FLAG | MIP_ZERO_FLAG


def check_readback(renderer, adapter, shader, reference):
    result = render(renderer, adapter, '--compute-texture-case', shader, reference)
    assert result.returncode == 0 and 'compared_words=320 mismatches=0' in result.stdout, (
        shader.name, result.stdout, result.stderr)


def main(cli, sdk, *render_args):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if not render_args:
            for group, wave32 in CASES:
                tuple(compile_pair(cli, sdk, work, group, wave32))
            for group, reason in (
                ((36, 1, 1), 'partial final wave in multi-wave group'),
                ((6, 1, 1), 'implicit derivatives require a pixel stage'),
            ):
                _, run = compile_shader(cli, sdk, work, group, True, True)
                assert run.returncode != 0 and reason in run.stderr, (group, run.stdout, run.stderr)
            _, run = compile_program(cli, sdk, work, 'gather-unsupported-sampler',
                                     operation_program(64, components=4))
            assert run.returncode != 0 and 'implicit gather requires a sampler fixed to mip zero' in run.stderr, (
                run.stdout, run.stderr)
            group = (6, 1, 1)
            shader, run = compile_shader(cli, sdk, work, group, True, False)
            assert run.returncode == 0 and shader.is_file(), run.stdout + run.stderr
            for name, binary, _, _, _ in operation_cases():
                shader, run = compile_program(cli, sdk, work, name, binary)
                assert run.returncode == 0 and shader.is_file(), (name, run.stdout, run.stderr)
            return

        renderer, adapter = render_args
        wave = render(renderer, adapter, '--shader-wave32')
        assert wave.returncode == 0, wave.stdout + wave.stderr
        reference = work / 'expected.bin'
        for group, wave32 in CASES:
            write_reference(reference, group)
            for shader in compile_pair(cli, sdk, work, group, wave32):
                check_readback(renderer, adapter, shader, reference)
            print(f'{group} wave{32 if wave32 else 64}: implicit and explicit LOD matched guest readback')
        for name, binary, expected, payload, flags in operation_cases():
            shader, run = compile_program(cli, sdk, work, name, binary)
            assert run.returncode == 0 and shader.is_file(), (name, run.stdout, run.stderr)
            write_case(reference, expected, payload, flags)
            check_readback(renderer, adapter, shader, reference)
            print(f'{name}: exact readback')


if __name__ == '__main__':
    main(*sys.argv[1:])
