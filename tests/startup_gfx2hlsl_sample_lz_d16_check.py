"""Packed D16 level-zero pixel sampling and D3D12 WARP readback."""
import struct
import sys
import tempfile
from pathlib import Path

from startup_image_load_d16_check import (
    FLOAT_MIP1, FLOAT_TEXELS, FULLSCREEN_VS, G, PACK16_RNE_HLSL, PACK16_RTZ_HLSL, UINT_RTV, VS_WORDS,
    compile_hlsl, expect_refused, pack16_rtz_ref, pack_pair, root_blob, run_d16_draw,
)


# s_load_dwordx8 s[16:23], s[8:9], 0
# s_mov_b32 s24, 0; s_mov_b32 s25, 0x100000; s_mov_b32 s26, 0x4000000; s_mov_b32 s27, 0
# v_mov_b32 v0, 0.0625; v_mov_b32 v1, 0.25; s_waitcnt lgkmcnt(0)
PREFIX = bytes.fromhex(
    '04040cf4000000fa 800398be ff0399be00001000 ff039abe00000004 80039bbe '
    'ff02007e0000803d ff02027e0000803e 7fc08cbf'
)
# image_sample_lz v[4:5], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D d16
FULL = bytes.fromhex('080f9cf00004c480')
# image_sample_lz v[6:7], v[0:1], s[16:23], s[24:27] dmask:0xb dim:SQ_RSRC_IMG_2D d16
SPARSE = bytes.fromhex('080b9cf00006c480')
# s_waitcnt vmcnt(0); v_and_b32 v7, 0xffff, v7; exp mrt0 v4, v5, v6, v7 done vm; s_endpgm
SUFFIX = bytes.fromhex('703f8cbf ff0e0e36ffff0000 0f1800f804050607 000081bf')

LEVEL_ZERO = 'SampleLevel(q0, float2(F(v0), F(v1)), 0)'
SAMPLER = 'smpl 0 0x0 0x100000 0x4000000 0x0'


def expected(texel, narrow):
    x, y, z, w = map(narrow, texel)
    return (pack_pair(x, y), pack_pair(z, w), pack_pair(x, y), w) * 16


def half_rne(word):
    value, = struct.unpack('<f', struct.pack('<I', word))
    return int.from_bytes(struct.pack('<e', value), 'little')


def translate(first_sample):
    return G.Pipeline(VS_WORDS, PREFIX + first_sample + SPARSE + SUFFIX, root_blob(), rtv_formats=UINT_RTV)


def controls():
    pipe = translate(FULL)
    source = pipe.hlsl_ps()
    assert source.count(LEVEL_ZERO) == 2, source
    assert 'pack16_rne(t.z) | (pack16_rne(t.w) << 16)' in source
    assert 'v7 = pack16_rne(t.w);' in source
    manifest = pipe.manifest()
    assert SAMPLER in manifest.splitlines(), manifest
    assert 'd16srv' not in manifest and 'd16unorm16' not in manifest, manifest
    # A 2-register destination starting at v255 overruns the VGPR file. TFE and
    # result packing for implicit-LOD sampling remain outside this lowering.
    expect_refused(lambda: translate(FULL[:5] + b'\xff' + FULL[6:]),
                   'image_sample_lz d16 destination v[255:256] for dmask 0xf', 'destination overrun')
    expect_refused(lambda: translate(FULL[:2] + bytes([FULL[2] | 1]) + FULL[3:]),
                   'image_sample_lz modifier tfe', 'TFE sample')
    expect_refused(lambda: translate(FULL[:2] + b'\x80' + FULL[3:]),
                   'image_sample modifier d16', 'implicit-LOD D16')
    reference = expected(FLOAT_TEXELS[0], half_rne)
    mutants = {
        'mip_one': (source.replace(LEVEL_ZERO, LEVEL_ZERO.replace(', 0)', ', 1)')), expected(FLOAT_MIP1[0], half_rne)),
        'truncate': (source.replace(PACK16_RNE_HLSL, PACK16_RTZ_HLSL, 1), expected(FLOAT_TEXELS[0], pack16_rtz_ref)),
    }
    for shader, words in mutants.values():
        assert shader != source and words != reference
    print('image_sample_lz d16: emitted packed full and sparse masks, both CPU mutant references differ')
    return source, reference, mutants


def draw(dxc, renderer):
    source, reference, mutants = controls()
    out = Path(__file__).resolve().parents[1] / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sample-lz-d16-', dir=out) as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs', 'vs_6_0', directory)
        ps = compile_hlsl(dxc, 'dxc', source, 'ps', 'ps_6_0', directory)
        mode = 'rgba32f_sampled'
        run = run_d16_draw(renderer, 'warp', mode, vs, ps, reference, directory, 'reference')
        assert run.returncode == 0, run.stdout + run.stderr
        for name, (shader, words) in mutants.items():
            ps = compile_hlsl(dxc, 'dxc', shader, name, 'ps_6_0', directory)
            run = run_d16_draw(renderer, 'warp', mode, vs, ps, reference, directory, name + '_negative')
            assert run.returncode == 1 and 'pixel (0,0) mismatch' in run.stderr, run.stdout + run.stderr
            run = run_d16_draw(renderer, 'warp', mode, vs, ps, words, directory, name + '_model')
            assert run.returncode == 0, run.stdout + run.stderr
        print('WARP matched 64 words and rejected both mutants against the CPU reference')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) == 4 and sys.argv[1] == 'draw':
        draw(*sys.argv[2:])
    else:
        raise SystemExit('usage: startup_gfx2hlsl_sample_lz_d16_check.py [draw <dxc> <offscreen_check>]')
