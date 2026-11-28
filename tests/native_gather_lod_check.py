"""Compare direct DXIL 2D image_gather4_l against independent mip/tap references."""
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import compute_gather4_l_check as compute_data
from native_direct_container_check import compute, patched_pixel, words
from pixel_gather_check import SIDE, texel, write_case
from startup_compute_gather_l_check import reference as compute_reference


@dataclass(frozen=True)
class Case:
    name: str
    lod: float
    relative: int
    channel: int
    first: int = 1
    uv: tuple[float, float] = (0.546875, 0.546875)
    max_lod: int = 512  # unsigned 4.8


CASES = (
    Case('view_first', 0.0, 0, 0),
    Case('view_next', 1.0, 1, 1),
    Case('view_last', 7.0, 1, 2, max_lod=0xFFF),
    Case('half_up', 0.5, 1, 3),
    Case('below_half', 0.499, 0, 1),
    Case('negative_lod', -1.0, 0, 2),
    Case('sampler_limit', 1.0, 0, 3, max_lod=127),
    Case('outside_uv', 0.0, 0, 0, uv=(-1.0e30, 1.25)),
    Case('nonfinite_uv', 0.0, 0, 1, uv=(float('nan'), 0.546875)),
    Case('nonfinite_lod', float('inf'), 0, 2),
    Case('full_view', 2.0, 2, 0, first=0),
)
ORDER = ((0, 1), (1, 1), (1, 0), (0, 0))


def float_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def program(case, stage, *, dim=1, mask=None, flags=0, sampler=0x12, filter_word=0x04500000,
            override_descriptor=False, dynamic=False):
    code = [0xBE810380, 0xF40C0200, 0xFA000000]  # read the texture descriptor into s[8:15]
    if stage == 'cs':
        code += [0xF4080800, 0xFA000020]  # output descriptor s[32:35]
        # Save the lane's byte offset and LOD before overwriting v0 with U.
        code += [0x34060084, 0x7E040D00] if dynamic else [0x7E060280]
    if override_descriptor:
        code.append(0xBE8E0380)  # clobber s14 without changing the descriptor base
    for reg, value in enumerate((sampler, case.max_lod << 12, filter_word, 0), start=4):
        code += [0xBE8003FF | reg << 16, value]
    for reg, value in enumerate(case.uv if dynamic else (*case.uv, case.lod)):
        code += [0x7E0002FF | reg << 17, float_bits(value)]
    mask = 1 << case.channel if mask is None else mask
    code += [0xBF8CC07F, 0xF0000000 | 68 << 18 | dim << 3 | mask << 8 | flags, 0x00220400, 0xBF8C3F70]
    if stage == 'cs':
        return compute(words(*code, 0xE0781000, 0x80080403, 0xBF810000), threads=4 if dynamic else 1, table=True)
    return patched_pixel(len(code), code, table=True, export=(0xF800180F, 0x07060504))


def compile_shader(cli, sdk, work, case, stage, **overrides):
    source = work / f'{case.name}-{stage}'
    source.write_bytes(program(case, stage, **overrides))
    cso = source.with_suffix('.cso')
    run = subprocess.run([cli, sdk, work / 'cache', f'gfx1013-container-{stage}', source, cso],
                         capture_output=True, text=True, timeout=30)
    return cso, run


def controls(cli, sdk, work):
    for stage in ('ps', 'cs'):
        shader, run = compile_shader(cli, sdk, work, CASES[1], stage)
        assert run.returncode == 0 and shader.is_file(), run.stdout + run.stderr
        dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', shader], check=True,
                              capture_output=True, text=True).stdout
        assert dump.count('call %dx.types.ResRet.f32 @dx.op.textureLoad.f32(') == 4, dump
        assert dump.count('call %dx.types.Dimensions @dx.op.getDimensions(') == 2, dump
        assert '@dx.op.textureGather' not in dump, dump
        for overrides, reason in (
            ({'dim': 5}, 'explicit 2D gather operands'),
            ({'mask': 3}, 'explicit 2D gather operands'),
            ({'flags': 0x1000}, 'explicit 2D gather operands'),
            ({'sampler': 0x13}, 'explicit 2D gather sampler'),
            ({'filter_word': 0x08500000}, 'explicit 2D gather sampler'),
            ({'filter_word': 0x14500000}, 'explicit 2D gather sampler'),
            ({'override_descriptor': True}, 'non-contiguous resource descriptor'),
        ):
            _, run = compile_shader(cli, sdk, work, CASES[1], stage, **overrides)
            assert run.returncode and reason in run.stderr, (stage, overrides, run.stderr)
    print('both stages emitted four loads and rejected seven unsupported cases')


def taps(case):
    if not all(map(math.isfinite, (*case.uv, case.lod))):
        return (0.0,) * 4
    # The pixel helper loads the first payload into MostDetailedMip.
    # Payload dimensions and texel tags therefore start at relative mip zero.
    level = case.relative
    extent = SIDE >> level
    x, y = (math.floor(value * extent - 0.5) for value in case.uv)
    return tuple(texel(level, min(max(x + dx, 0), extent - 1), min(max(y + dy, 0), extent - 1), case.channel)
                 for dx, dy in ORDER)


def render(renderer, adapter, operation, *args):
    run = subprocess.run([renderer, adapter, operation, *args], capture_output=True, text=True, timeout=30)
    if run.returncode == 77:
        print(run.stdout + run.stderr, end='')
        raise SystemExit(77)
    return run


def render_compute(renderer, adapter, shader, reference, first, expected):
    flags = compute_data.FLOAT_TEXTURE_FLAG | (compute_data.VIEW_MIP1_FLAG if first else 0)
    reference.write_bytes(words(compute_data.GROUPS | flags, *expected, *compute_data.texels()))
    return render(renderer, adapter, '--compute-texture-case', shader, reference)


def draw(cli, sdk, renderer, adapter, stage, work):
    if stage == 'cs':
        wave = render(renderer, adapter, '--shader-wave32')
        assert wave.returncode == 0, wave.stdout + wave.stderr
    for case in CASES:
        shader, run = compile_shader(cli, sdk, work, case, stage)
        assert run.returncode == 0, (case.name, run.stdout, run.stderr)
        reference = work / 'expected.bin'
        if stage == 'ps':
            write_case(reference, 3 - case.first, case.first, [taps(case)] * (SIDE ** 2))
            result = render(renderer, adapter, '--check-pixel-gather', shader, reference)
            assert result.returncode == 0 and 'pixels=64 channels=256 max_absolute_error=0' in result.stdout, (
                case.name, result.stdout, result.stderr)
        else:
            finite = all(map(math.isfinite, (*case.uv, case.lod)))
            expected = (compute_reference(case.first + case.relative, case.channel, case.uv)
                        if finite else [0] * 4 + [compute_data.SENTINEL] * 316)
            result = render_compute(renderer, adapter, shader, reference, case.first, expected)
            assert result.returncode == 0 and 'compared_words=320 mismatches=0' in result.stdout, (
                case.name, result.stdout, result.stderr)
        if 'debug_layer=enabled' in result.stdout:
            assert 'debug_errors=0 debug_warnings=0' in result.stdout, result.stdout
        print(f'{stage} {case.name}: exact readback')
    if stage == 'cs':
        case = Case('dynamic_lod', 0.0, 0, 1)
        shader, run = compile_shader(cli, sdk, work, case, stage, dynamic=True)
        assert run.returncode == 0, run.stdout + run.stderr
        expected = [tap for level in (1, 2, 2, 2) for tap in compute_reference(level, case.channel, case.uv)[:4]]
        expected += [compute_data.SENTINEL] * 304
        result = render_compute(renderer, adapter, shader, reference, case.first, expected)
        assert result.returncode == 0 and 'compared_words=320 mismatches=0' in result.stdout, (
            result.stdout, result.stderr)
        print('cs dynamic LOD: four lanes selected resource mips 1, 2, 2, 2')
    # Keep the mip-1 expectation and force the shader to use view-relative mip zero.
    case = CASES[1]
    forced = Case('forced_mip_zero', 0.0, 0, case.channel)
    shader, run = compile_shader(cli, sdk, work, forced, stage)
    assert run.returncode == 0, run.stdout + run.stderr
    if stage == 'ps':
        write_case(reference, 2, case.first, [taps(case)] * (SIDE ** 2))
        result = render(renderer, adapter, '--check-pixel-gather', shader, reference)
        assert result.returncode == 1 and 'pixel_mismatch x=0 y=0' in result.stderr, (result.stdout, result.stderr)
    else:
        expected = compute_reference(case.first + case.relative, case.channel, case.uv)
        result = render_compute(renderer, adapter, shader, reference, case.first, expected)
        assert result.returncode == 1 and 'compared_words=320 mismatches=4' in result.stdout, (
            result.stdout, result.stderr)
    print(f'{stage} forced mip zero: readback rejected')


def main(mode, cli, sdk, *args):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if mode == 'controls':
            controls(cli, sdk, work)
        else:
            draw(cli, sdk, *args, mode, work)


if __name__ == '__main__':
    main(*sys.argv[1:])
