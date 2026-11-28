"""Differential for gathering at a derivative-selected mip in a nonzero-base SRV."""
import math
from pathlib import Path
import struct
import sys
import tempfile

import startup_image_gather_lz_check as gather
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl
from startup_image_get_lod_view_check import UNPACK_XY, read_pixel

VIEW = '1/2'
MIP_BASE = 8
UV = 0.5625
LOD_QUERY = struct.pack('<II', 0xF1800108, 0x00C40200)  # image_get_lod v2, v[0:1], ... dmask:1
WAIT_IMAGE = gather.SUFFIX[:4]
TAPS = ((0, 1), (1, 1), (1, 0), (0, 0))


def float_op(op, dst, value):
    # VOP2 with a literal source and the same VGPR as the other source.
    return struct.pack('<If', op << 25 | dst << 17 | dst << 9 | 255, value)


def prefix(step):
    # Same UV at the readback pixel for both cases. The derivative of the
    # window-relative UV changes from 1 to 2 texels per pixel.
    before, original, after = gather.PREFIX.partition(bytes.fromhex('f002007eff02027e0000803e'))
    assert original
    before = before.replace(bytes.fromhex('800399be'), struct.pack('<II', 0xBE9903FF, 2 << 20))
    xy = UNPACK_XY + b''.join(float_op(0x08, axis, step) + float_op(0x03, axis, UV) for axis in range(2))
    return before + xy + after


def pipeline(step, selected):
    lead = prefix(step)
    if selected:
        # Pass the image_get_lod result in v2 to the existing explicit gather.
        lead += LOD_QUERY + WAIT_IMAGE
    code = gather.shader(prefix=lead, opcode=0x44 if selected else 0x47)
    return gather.G.Pipeline(gather.VS_WORDS, code, gather.root_blob(flags=0), ps_hw_inputs=(0x8000, 0x8000))


def render(dxc, renderer, adapter, vs, pipe, name, view, work):
    sampler, = (line for line in pipe.manifest().splitlines() if line.startswith(('smp ', 'smpl ')))
    ps = compile_hlsl(dxc, 'dxc', pipe.hlsl_ps(), name, 'ps_6_0', work)
    return struct.unpack('>4f', read_pixel(renderer, adapter, vs, ps, sampler, view, mip_base=MIP_BASE))


def expected_taps(resource_mip):
    # Both visible levels contain four different interior taps.
    extent = MIP_BASE >> resource_mip
    origin = math.floor(UV * extent - 0.5)
    assert 0 <= origin < extent - 1
    area = MIP_BASE ** 2
    return tuple((1 + origin + x + MIP_BASE * (origin + y) + 2 * area * resource_mip) / (8 * area)
                 for x, y in TAPS)


def expect(label, observed, wanted):
    for tap, (actual, expected) in enumerate(zip(observed, wanted, strict=True)):
        assert actual == expected, f'{label}, pixel (0,0), tap {tap}: expected {expected}, got {actual}'


def main(dxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', work)
        for step, relative_mip in ((0.25, 0), (0.5, 1)):
            resource_mip = 1 + relative_mip
            selected = pipeline(step, selected=True)
            fixed = pipeline(step, selected=False)
            wanted = expected_taps(resource_mip)
            prefix_name = f'level{relative_mip}'
            actual = render(dxc, renderer, adapter, vs, selected, prefix_name, VIEW, work)
            expect(prefix_name, actual, wanted)
            oracle = render(dxc, renderer, adapter, vs, fixed, prefix_name + '_fixed', f'{resource_mip}/1', work)
            expect(prefix_name + ' fixed-view oracle', oracle, wanted)
            first = render(dxc, renderer, adapter, vs, fixed, prefix_name + '_first', VIEW, work)
            expect(prefix_name + ' first-visible control', first, expected_taps(1))
            assert (first == actual) == (relative_mip == 0), (step, first, actual)
            print(f'view={VIEW} step={step} relative_mip={relative_mip} resource_mip={resource_mip} taps={actual}')
            if relative_mip:
                print('the first-visible control differed at all four taps')
                assert all(a != b for a, b in zip(first, actual, strict=True))


if __name__ == '__main__':
    main(*sys.argv[1:])
