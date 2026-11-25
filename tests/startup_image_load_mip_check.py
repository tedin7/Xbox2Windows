"""Check explicit mip operands in source-built gfx1013 texture loads."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

from startup_image_3d_check import G, VS_WORDS, assemble, root_blob


PREFIX = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
v_mov_b32 v0, 0
v_mov_b32 v1, 0
v_mov_b32 v2, 2
v_mov_b32 v3, 1
s_waitcnt lgkmcnt(0)
"""
SUFFIX = """\
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
PREFIX_WORDS = bytes.fromhex('04040cf4000000fa 8002007e 8002027e 8202047e 8102067e 7fc08cbf')
SUFFIX_WORDS = bytes.fromhex('703f8cbf 0f1800f804050607 000081bf')
CASES = (
    ('2D packed', 'image_load_mip v[4:7], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D',
     '080f04f000040400', 'int3(v0, v1, v2)', 'tex2d'),
    ('3D NSA', 'image_load_mip v[4:7], [v0, v1, v3, v2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_3D',
     '120f04f00004040001030200', 'int4(v0, v1, v3, v2)', 'tex3d'),
    ('2D implicit', 'image_load v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D',
     '080f00f000040400', 'int3(v0, v1, 0)', 'tex2d'),
)


def shader(image):
    return G.Pipeline(VS_WORDS, PREFIX_WORDS + image + SUFFIX_WORDS, root_blob())


def check_plain_load_layouts(llvm_mc):
    # Pack x=1 and y=2 into v0. v1=63 distinguishes the packed and full-width forms.
    prefix = PREFIX.replace('v_mov_b32 v0, 0', 'v_mov_b32 v0, 0x20001').replace(
        'v_mov_b32 v1, 0', 'v_mov_b32 v1, 63')
    prefix_words = PREFIX_WORDS.replace(bytes.fromhex('8002007e 8002027e'),
                                        bytes.fromhex('ff02007e 01000200 bf02027e'))
    a16 = bytes.fromhex('080f00f000040440')
    if llvm_mc:
        source = prefix + 'image_load v[4:7], v0, s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D a16\n' + SUFFIX
        assert assemble(llvm_mc, source) == prefix_words + a16 + SUFFIX_WORDS
    ps_code = prefix_words + a16 + SUFFIX_WORDS
    root = root_blob()
    pipe = G.Pipeline(VS_WORDS, ps_code, root)
    ps = pipe.hlsl_ps()
    assert 'r0.Load(int3(v0 & 0xffffu, v0 >> 16, 0))' in ps, ps
    assert 'srv 0 tex2d table 0 0' in pipe.manifest()
    assert 'r0.Load(int3(v0, v1, 0))' in shader(bytes.fromhex('882f00f000040400')).hlsl_ps()

    old_key = G.fnv(root, G.fnv(ps_code, G.fnv(VS_WORDS, G.fnv(b'x2w-startup-shader-4'))))
    key = G.pipeline_key(VS_WORDS, ps_code, root)
    assert key != old_key, 'cached pre-fix shader would be reused'
    runtime = Path(__file__).resolve().parents[1] / 'src' / 'startup_stubs' / 'd3d12_draw.inl'
    revision = re.search(rb'constexpr char revision\[\] = "([^"]+)";', runtime.read_bytes())
    assert revision, 'desktop startup cache revision missing'
    runtime_key = G.fnv(root, G.fnv(ps_code, G.fnv(VS_WORDS, G.fnv(revision[1]))))
    assert key == runtime_key, 'offline and desktop startup cache revisions differ'

    # LLVM accepts [v0] as the same packed encoding and rejects a second address VGPR.
    # Patch the length field and append a fake NSA word to check that the decoder preserves the refusal.
    invalid_nsa = '0a0f00f00004044001000000'

    d16_ps = shader(bytes.fromhex('080f00f000040480')).hlsl_ps()
    assert 'v4 = pack16_d16(t.x, ud4) | (pack16_d16(t.y, ud4) << 16);' in d16_ps, d16_ps
    assert 'v5 = pack16_d16(t.z, ud4) | (pack16_d16(t.w, ud4) << 16);' in d16_ps, d16_ps
    try:
        shader(bytes.fromhex('080f04f000040480'))
    except G.Unsupported as error:
        assert 'image_load_mip' in str(error) and 'd16' in str(error), error
    else:
        raise AssertionError('d16 image_load_mip accepted')

    for flag, encoded in (('tfe', '080f01f000040400'),
                          ('r128', '088f00f000040400'), ('lwe', '080f02f000040400'),
                          ('reserved', '084f00f000040400'), ('dmask:0x0', '080000f000040400'),
                          ('a16', '100f00f000040440'), ('invalid_nsa', invalid_nsa)):
        try:
            shader(bytes.fromhex(encoded))
        except G.Unsupported as error:
            assert 'image_load' in str(error) and flag in str(error), error
        else:
            raise AssertionError(f'{flag} image load accepted with unsupported layout')
    try:
        pipe.ps.image(['v[4:7]', '[v0, v1]', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0xf', 'a16'], 'load')
    except G.Unsupported as error:
        assert 'image_load address' in str(error), error
    else:
        raise AssertionError('A16 image load accepted two address registers')
    print('plain 2D load: packed A16 coordinates, cache policy unchanged, unsupported layouts refused')


def check(llvm_mc):
    for name, instruction, encoded, address, kind in CASES:
        image = bytes.fromhex(encoded)
        if llvm_mc:
            source = PREFIX + instruction + '\n' + SUFFIX
            assert assemble(llvm_mc, source) == PREFIX_WORDS + image + SUFFIX_WORDS, name
        pipe = shader(image)
        assert f'r0.Load({address})' in pipe.hlsl_ps(), name
        assert f'srv 0 {kind} table 0 0' in pipe.manifest(), name
        assert 'SamplerState' not in pipe.hlsl_ps(), name
        print(f'{name}: {address}, {kind}, no sampler')
    # Packed data, packed addresses and fault-status returns need separate lowering.
    for flag, byte, mask in (('d16', 7, 0x80), ('a16', 7, 0x40), ('tfe', 2, 0x01)):
        unsupported = bytearray.fromhex(CASES[0][2])
        unsupported[byte] |= mask
        try:
            shader(unsupported)
        except G.Unsupported as error:
            assert 'image_load_mip' in str(error) and flag in str(error), error
        else:
            raise AssertionError(f'{flag} image load accepted')
    check_plain_load_layouts(llvm_mc)


def draw(dxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as directory:
        hlsl = Path(directory) / 'load_mip.hlsl'
        dxil = hlsl.with_suffix('.dxil')
        for index, level in ((0, 2), (2, 0)):
            source = shader(bytes.fromhex(CASES[index][2])).hlsl_ps()
            # The offscreen fixture's vertex shader exports TEXCOORD0 before
            # SV_Position. Match its input signature without changing the load.
            source = source.replace('struct PSIn {', 'struct PSIn {\n    float4 uv : TEXCOORD0;')
            hlsl.write_text(source, encoding='ascii')
            subprocess.run([dxc, '-T', 'ps_6_0', '-E', 'main', '-Fo', dxil, hlsl], check=True)
            result = subprocess.run([renderer, adapter, '--check-pixel-mip-exact', dxil, str(level)],
                                    capture_output=True, text=True)
            output = result.stdout + result.stderr
            print(f'Mip {level}:\n{output}', end='', flush=True)
            if result.returncode:
                return result.returncode
            if 'pixels=64 channels=256 max_absolute_error=0' not in output:
                raise RuntimeError('mip draw did not report a complete exact readback')
        result = subprocess.run([renderer, adapter, '--check-pixel-mip-exact', dxil, '2'],
                                capture_output=True, text=True)
        output = result.stdout + result.stderr
        if result.returncode != 1 or 'pixel_mismatch x=0 y=0' not in output:
            raise RuntimeError(f'wrong-mip control did not produce a pixel mismatch:\n{output}')
        print(f'Wrong-mip control rejected:\n{output}', end='')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc', help='reassemble the synthetic fixtures')
    parser.add_argument('--draw', nargs=3, metavar=('DXC', 'RENDERER', 'ADAPTER'))
    args = parser.parse_args()
    check(args.llvm_mc)
    if args.draw:
        raise SystemExit(draw(*args.draw))
