"""Draw gfx1013 image_get_lod through translated static samplers and 2D/array mip views."""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_gfx2hlsl_image_get_lod_check as fixture
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

# LLVM 15 gfx1013 encodings. v0 initially holds the packed pixel position.
UNPACK_XY = bytes.fromhex(
    '9000022c'          # v_lshrrev_b32 v1, 16, v0
    'ff000036ffff0000'  # v_and_b32 v0, 0xffff, v0
    '000d007e'          # v_cvt_f32_u32 v0, v0
    '010d027e')         # v_cvt_f32_u32 v1, v1
SCALE_EIGHTH = bytes.fromhex(
    'ff0000100000003e'  # v_mul_f32 v0, 0.125, v0
    'ff0202100000003e')  # v_mul_f32 v1, 0.125, v1

DIRECT_PS = """Texture2D<float4> image : register(t0);
SamplerState sampler0 : register(s0);
float4 main(float4 position : SV_Position) : SV_Target0 {
    float2 uv = floor(position.xy) * STEP;
    float clamped = image.CalculateLevelOfDetail(sampler0, uv);
    float raw = image.CalculateLevelOfDetailUnclamped(sampler0, uv);
    return float4(clamped, raw, clamped, raw);
}
"""
SLICE_LIT = bytes.fromhex('ff02047e')  # v_mov_b32 v2, <imm32> (VOP1 literal): the slice dword of the array address
DIRECT_ARRAY_PS = DIRECT_PS.replace('Texture2D<', 'Texture2DArray<', 1)


def patch_sampler(code, low, high):
    # s_mov_b32 s25, literal: sampler word 1 carries the two unsigned 4.8 LOD limits.
    at = fixture.W2_AT - 4
    assert code[at:at + 4] == bytes.fromhex('800399be')
    limits = int(low * 256) | (int(high * 256) << 12)
    return code[:at] + struct.pack('<II', 0xbe9903ff, limits) + code[at + 4:]


def array_pipeline(step, slice_index, low, high, mip_filter):
    """The dim 0x5 (SQ_RSRC_IMG_2D_ARRAY) stage: v[0:2] is u, v and the slice; the u/v moves are the same
    pixel-position unpacking and scaling as the 2D lane, and the slice dword becomes a literal move (the
    fixture's VOP1 v2 = 1.0 cannot hold 0.0 as an inline constant)."""
    code = fixture.stage(dmask=3, dim=5, word2=mip_filter << 26)
    coords = bytes.fromhex('f002007eff02027e0000803ef202047ef402067e')
    assert code.count(coords) == 1
    scale = {1: b'', 0.125: SCALE_EIGHTH}[step]
    lit = struct.pack('<I', 0 if slice_index == 0 else 0x3f800000)
    # v2 (the slice) and v3 (unused) keep moving constants; only their encodings shift with the insertion.
    code = code.replace(coords, UNPACK_XY + scale + SLICE_LIT + lit + bytes.fromhex('f402067e'))
    code = patch_sampler(code, low, high)
    return fixture.G.Pipeline(fixture.VS_WORDS, code, fixture.root_blob(flags=0), ps_hw_inputs=(0x8000, 0x8000))


def pipeline(step, low, high, mip_filter):
    code = fixture.stage(dmask=3, word2=mip_filter << 26)
    # Replace the four constant coordinate moves preceding s_waitcnt lgkmcnt(0).
    coords = bytes.fromhex('f002007eff02027e0000803ef202047ef402067e')
    assert code.count(coords) == 1
    scale = {1: b'', 0.125: SCALE_EIGHTH}[step]
    code = code.replace(coords, UNPACK_XY + scale)
    code = patch_sampler(code, low, high)
    return fixture.G.Pipeline(fixture.VS_WORDS, code, fixture.root_blob(flags=0), ps_hw_inputs=(0x8000, 0x8000))


def read_pixel(renderer, adapter, vs, ps, sampler, view, *, selector=None, array=False, mip_base=None):
    mode = '--startup-image-2d-array-view' if array else '--startup-image-2d-view'
    args = [renderer, adapter, mode, vs, ps, sampler, view]
    if selector is not None or mip_base is not None:
        args.append(str(0 if selector is None else selector))
    if mip_base is not None:
        args.append(str(mip_base))
    run = subprocess.run(args, capture_output=True, text=True, timeout=30)
    print(run.stdout, end='')
    if run.returncode == 77:
        print(run.stderr, file=sys.stderr)
        raise SystemExit(77)
    if run.returncode or ('debug_layer=enabled' in run.stdout and 'debug_errors=0 debug_warnings=0' not in run.stdout):
        raise AssertionError(run.stdout + run.stderr)
    match = re.search(r'^view_rgba=((?:[0-9a-f]{8},){3}[0-9a-f]{8})$', run.stdout, re.M)
    assert match, run.stdout
    return bytes.fromhex(match[1].replace(',', ''))


def check_color(pixel, expected):
    values = struct.unpack('>4f', pixel)
    # One D3D12 mip LOD fractional step. The translated and direct words still have to match exactly.
    assert all(abs(value - ref) <= 1 / 256 for value, ref in zip(values, expected * 2)), (values, expected)


def draw(dxc, renderer, adapter):
    # SRV first/count, UV step, sampler min/max, mip filter (1 point, 2 linear), expected (clamped, raw).
    # The raw LOD is log2(step * width of the first visible mip); the clamped LOD also uses the view and filter.
    cases = (
        ('0/3', 1, 0, 2, 1, (2, 2)),
        ('1/2', 1, 0, 2, 1, (1, 1)),
        ('1/1', 1, 0, 2, 1, (0, 1)),
        ('0/1', 1, 0, 2, 1, (0, 2)),
        ('0/3', 1, 0, 0.75, 1, (1, 2)),
        ('0/3', 1, 0, 0.75, 2, (0.75, 2)),
        ('0/3', 0.125, 0.5, 2, 2, (0.5, -1)),
    )
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', work)
        for view, step, low, high, mip_filter, expected in cases:
            pipe = pipeline(step, low, high, mip_filter)
            sampler, = (line for line in pipe.manifest().splitlines() if line.startswith('smpl '))
            guest = compile_hlsl(dxc, 'dxc', pipe.hlsl_ps(), 'translated', 'ps_6_0', work)
            direct = compile_hlsl(dxc, 'dxc', DIRECT_PS.replace('STEP', str(float(step))), 'direct', 'ps_6_0', work)
            pixel = read_pixel(renderer, adapter, vs, guest, sampler, view)
            reference = read_pixel(renderer, adapter, vs, direct, sampler, view)
            assert pixel == reference, (pixel.hex(), reference.hex())
            check_color(pixel, expected)
            print(f'view={view} step={step} min={low} max={high} mip_filter={mip_filter} rgba={pixel.hex()}')
            if high == 0.75 and mip_filter == 2:
                # Only the tag changes. The ordinary static sampler ignores word 1's limits.
                unlimited = read_pixel(renderer, adapter, vs, guest, sampler.replace('smpl ', 'smp ', 1), view)
                check_color(unlimited, (2, 2))
                assert unlimited != pixel, (unlimited.hex(), pixel.hex())
                print(f'ordinary sampler control rgba={unlimited.hex()}')


def draw_array(dxc, renderer, adapter):
    """The array lane: the dim 0x5 get_lod stage against the same 4x4 two-slice three-mip texture through
    --startup-image-2d-array-view. The whole resource is bound (srv tex2da table 0 0: no SRV mip window), so
    the raw LOD is log2(step * 4) and the clamped LOD is that value clamped by the sampler's MIN_LOD/MAX_LOD
    word. Two different valid slices (0 and 1; a slice coordinate is not normalized) must return the same LOD
    pair: the slice is consumed but not part of the query. The translated and direct HLSL readbacks must
    match exactly, and check_color keeps both near the CPU reference (mip filter 1 = point, 2 = linear)."""
    # UV step, slice, sampler min/max, mip filter (1 point, 2 linear), expected (clamped, raw).
    cases = (
        (1, 0, 0, 2, 1, (2, 2)),
        (1, 1, 0, 2, 1, (2, 2)),
        (0.125, 0, 0.5, 2, 2, (0.5, -1)),
        (0.125, 1, 0.5, 2, 2, (0.5, -1)),
    )
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'fullscreen_array', 'vs_6_0', work)
        for step, slice_index, low, high, mip_filter, expected in cases:
            pipe = array_pipeline(step, slice_index, low, high, mip_filter)
            sampler, = (line for line in pipe.manifest().splitlines() if line.startswith('smpl '))
            guest = compile_hlsl(dxc, 'dxc', pipe.hlsl_ps(), 'translated_array_%d' % slice_index, 'ps_6_0', work)
            direct = compile_hlsl(dxc, 'dxc', DIRECT_ARRAY_PS.replace('STEP', str(float(step))),
                                  'direct_array_%d' % slice_index, 'ps_6_0', work)
            pixel = read_pixel(renderer, adapter, vs, guest, sampler, '0/3', array=True)
            reference = read_pixel(renderer, adapter, vs, direct, sampler, '0/3', array=True)
            assert pixel == reference, (slice_index, pixel.hex(), reference.hex())
            check_color(pixel, expected)
            print(f'array slice={slice_index} step={step} min={low} max={high} mip_filter={mip_filter} '
                  f'rgba={pixel.hex()}')


if __name__ == '__main__':
    draw(*sys.argv[1:])
    draw_array(*sys.argv[1:])
