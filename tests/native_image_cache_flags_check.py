"""Verify GLC and DLC cache-policy flags on direct read-only image ops (#3242)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_direct_container_check import compute, patched_pixel, words
import native_gather_lod_check as G

GLC = 0x2000
DLC = 0x0080
SLC = 0x02000000
ALL_CACHE = GLC | DLC | SLC
REFUSED = 'direct image opcode or descriptor modifiers'


def ro_program(op, stage, *, flags=0, w1_flags=0):
    code = [0xBE810380, 0xF40C0200, 0xFA000000]  # s[8:15] texture descriptor
    if stage == 'cs':
        code += [0xF4080800, 0xFA000020, 0x7E060280]  # s[32:35] UAV, v3 = 0
    for reg, value in enumerate((0x12, 512 << 12, 0x04500000, 0), start=4):
        code += [0xBE8003FF | (reg << 16), value]
    coords = (0.5, 0.5, 0.0) if op >= 32 else (0, 0, 0)
    for reg, value in enumerate(coords):
        encoded = struct.unpack('<I', struct.pack('<f', value))[0] if op >= 32 else value
        code += [0x7E0002FF | (reg << 17), encoded]
    sampler_field = (1 << 21) if op >= 32 else 0
    dmask = 1 if op in (68, 71) else 0xF
    code += [0xBF8CC07F, 0xF0000008 | (op << 18) | (dmask << 8) | flags, 0x00020400 | sampler_field | w1_flags, 0xBF8C3F70]
    if stage == 'cs':
        return compute(words(*code, 0xE0781000, 0x80080403, 0xBF810000), threads=1, table=True)
    return patched_pixel(len(code), code, table=True, export=(0xF800180F, 0x07060504))


def store_program(stage, *, flags=0):
    code = [0xBE810380, 0xF40C0200, 0xFA000000, 0xBF8CC07F,
            0x7E000280, 0x7E020280,
            0xF0000F08 | (8 << 18) | flags, 0x00020400, 0xBF810000]
    if stage == 'cs':
        return compute(words(*code), threads=1, table=True)
    return patched_pixel(len(code), code, table=True, export=(0xF800180F, 0x07060504))


def compile_bytes(cli, sdk, work, label, stage, blob):
    src = work / f'{label}.{stage}'
    src.write_bytes(blob)
    cso = src.with_suffix('.cso')
    run = subprocess.run([cli, sdk, work / 'cache', f'gfx1013-container-{stage}', src, cso],
                         capture_output=True, text=True, timeout=30)
    return cso, run


def controls(cli, sdk, work):
    ops = (
        ('image_load', 0, '@dx.op.textureLoad.f32'),
        ('image_load_mip', 1, '@dx.op.textureLoad.f32'),
        ('image_getresinfo', 14, '@dx.op.getDimensions'),
        ('image_sample_lz', 39, '@dx.op.sampleLevel.f32'),
        ('image_gather4_lz', 71, '@dx.op.textureGather.f32'),
        ('image_gather4_l', 68, '@dx.op.textureLoad.f32'),
    )
    accepted = 0
    for stage in ('cs', 'ps'):
        for name, op, expected_dxil in ops:
            base_cso, base_run = compile_bytes(cli, sdk, work, f'{name}-base', stage, ro_program(op, stage))
            assert base_run.returncode == 0 and base_cso.is_file(), (stage, name, base_run.stdout, base_run.stderr)
            base_dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', base_cso],
                                       capture_output=True, text=True, check=True).stdout
            assert expected_dxil in base_dump, (stage, name, expected_dxil, base_dump)
            for tag, fl in (('glc', GLC), ('dlc', DLC), ('slc', SLC), ('all', ALL_CACHE)):
                cso, run = compile_bytes(cli, sdk, work, f'{name}-{tag}', stage, ro_program(op, stage, flags=fl))
                assert run.returncode == 0 and cso.is_file(), (stage, name, tag, run.stdout, run.stderr)
                dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', cso],
                                      capture_output=True, text=True, check=True).stdout
                assert dump == base_dump, (stage, name, tag)
                accepted += 1

    refusals = (
        ('store_glc', lambda st: store_program(st, flags=GLC)),
        ('store_dlc', lambda st: store_program(st, flags=DLC)),
        ('sample_r128', lambda st: ro_program(39, st, flags=0x8000)),
        ('sample_tfe', lambda st: ro_program(39, st, flags=0x10000)),
        ('sample_lwe', lambda st: ro_program(39, st, flags=0x20000)),
        ('sample_res_bit6', lambda st: ro_program(39, st, flags=0x0040)),
        ('sample_res_bit14', lambda st: ro_program(39, st, flags=0x4000)),
        ('sample_res_w1_bit26', lambda st: ro_program(39, st, w1_flags=1 << 26)),
    )
    refused = 0
    for stage in ('cs', 'ps'):
        for rname, builder in refusals:
            _, run = compile_bytes(cli, sdk, work, f'ref-{rname}', stage, builder(stage))
            assert run.returncode != 0 and REFUSED in run.stderr, (stage, rname, run.returncode, run.stderr)
            refused += 1

    print(f'native_image_cache_flags controls: {accepted} flagged DXIL modules matched plain baselines; {refused} refusals matched')
    return 0


def execute(cli, sdk, renderer, adapter, stage, work):
    if stage == 'cs':
        wave = G.render(renderer, adapter, '--shader-wave32')
        assert wave.returncode == 0, wave.stdout + wave.stderr
    cases = (G.CASES[0], G.CASES[1], G.CASES[3], G.CASES[7])
    flag_sets = (('glc', GLC), ('dlc', DLC), ('glc_dlc_slc', ALL_CACHE))
    for case in cases:
        for tag, flags in flag_sets:
            cso, run = G.compile_shader(cli, sdk, work, case, stage, flags=flags)
            assert run.returncode == 0, (stage, case.name, tag, run.stdout, run.stderr)
            reference = work / 'expected.bin'
            if stage == 'ps':
                G.write_case(reference, 3 - case.first, case.first, [G.taps(case)] * (G.SIDE ** 2))
                res = G.render(renderer, adapter, '--check-pixel-gather', cso, reference)
                assert res.returncode == 0 and 'pixels=64 channels=256 max_absolute_error=0' in res.stdout, (
                    stage, case.name, tag, res.stdout, res.stderr)
            else:
                expected = G.compute_reference(case.first + case.relative, case.channel, case.uv)
                res = G.render_compute(renderer, adapter, cso, reference, case.first, expected)
                assert res.returncode == 0 and 'compared_words=320 mismatches=0' in res.stdout, (
                    stage, case.name, tag, res.stdout, res.stderr)
            if 'debug_layer=enabled' in res.stdout:
                assert 'debug_errors=0 debug_warnings=0' in res.stdout, res.stdout
    print(f'native_image_cache_flags {stage} ({adapter}): {len(cases) * len(flag_sets)} cached-flag cases matched')
    return 0


def main(mode, cli, sdk, *args):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if mode == 'controls':
            return controls(cli, sdk, work)
        return execute(cli, sdk, args[0], args[1], mode, work)


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
