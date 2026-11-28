"""Check that direct 2D image_sample_lz accepts all five sampler index bits."""
from pathlib import Path
import subprocess
import sys
import tempfile

import compute_gather4_l_check as texture
from native_direct_container_check import compute, words
from native_gather_lod_check import render, render_compute


def program(sampler_base, reserved_bits):
    code = [
        0xBE810380, 0xF40C0200, 0xFA000000,  # texture descriptor into s[8:15]
        0xF4080800, 0xFA000020,              # output descriptor into s[32:35]
        0x7E060280,                          # v3 = output byte offset 0
    ]
    for reg, value in enumerate(texture.SAMPLER, start=sampler_base):
        code.extend((0xBE8003FF | reg << 16, value))
    code.extend((
        0x7E0002FF, 0x3C800000,  # v0 = 0.5 / 32
        0x7E0202FF, 0x3D800000,  # v1 = 0.5 / 8
        0xBF8CC07F,
        # image_sample_lz v[4:7], v[0:1], s[8:15], s[sampler_base:sampler_base+3]
        0xF09C0F08, ((sampler_base // 4) << 21) | 0x00020400 | reserved_bits,
        0xBF8C3F70,
        0xE0781000, 0x80080403,  # buffer_store_dwordx4 v[4:7]
        0xBF810000,
    ))
    return compute(words(*code), threads=1, table=True)


def main(cli, sdk, renderer, adapter):
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        wave = render(renderer, adapter, '--shader-wave32')
        assert wave.returncode == 0, wave.stdout + wave.stderr
        # The bilinear sample is centered on texel (0, 0) of mip 0.
        expected = [texture.texel(0, 0, 0, channel) for channel in range(4)] + [texture.SENTINEL] * 316
        for name, base, reserved_bits in (('low', 4, 0), ('high', 68, 0), ('reserved', 68, 1 << 26)):
            source = work / name
            source.write_bytes(program(base, reserved_bits))
            shader = source.with_suffix('.cso')
            compiled = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-cs', source, shader],
                                      capture_output=True, text=True, timeout=30)
            if reserved_bits:
                assert compiled.returncode != 0 and 'direct image opcode or descriptor modifiers' in compiled.stderr, (
                    compiled.stdout, compiled.stderr)
                continue
            assert compiled.returncode == 0, (name, compiled.stdout, compiled.stderr)
            result = render_compute(renderer, adapter, shader, work / 'expected.bin', 0, expected)
            assert result.returncode == 0 and 'compared_words=320 mismatches=0' in result.stdout, (
                name, result.stdout, result.stderr)
        print('Both sampler quads matched the CPU reference. Reserved bit 26 was refused.')


if __name__ == '__main__':
    main(*sys.argv[1:])
