"""Check explicit mip operands in source-built gfx1013 texture loads."""
import argparse
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

from startup_image_3d_check import G, VS_WORDS, assemble, root_blob
from startup_stream_check import container
from startup_stream_draw_check import vertex, vop1, vop2


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
    d16_mip_pipe = shader(bytes.fromhex('080f04f000040480'))
    d16_mip_ps = d16_mip_pipe.hlsl_ps()
    assert 'r0.Load(int3(v0, v1, v2))' in d16_mip_ps, d16_mip_ps
    assert 'v4 = pack16_d16(t.x, ud4) | (pack16_d16(t.y, ud4) << 16);' in d16_mip_ps, d16_mip_ps
    assert 'v5 = pack16_d16(t.z, ud4) | (pack16_d16(t.w, ud4) << 16);' in d16_mip_ps, d16_mip_ps
    assert 'd16srv 0 exact' in d16_mip_pipe.manifest().splitlines(), d16_mip_pipe.manifest()

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
    # Packed addresses and fault-status returns remain refused on image_load_mip.
    for flag, byte, mask in (('a16', 7, 0x40), ('tfe', 2, 0x01)):
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


IDENTIFIERS = (0, 1, 0x00ffffff, 0x01000001, 0xffffffff)
FLOAT_BITS = struct.unpack('<5I', struct.pack('<5f', 0, 1, -2.5, 0.125, 65536))


DIMENSIONS = {'1d': 0, '2d': 1, '3d': 2, '2da': 5}


def identifier_pixel(filtered=False, dim='2d', explicit_mip=False):
    # The load draws across five columns. Split the packed position before
    # fetching red, then export the register bits to MRT0.
    words = [0xf40c0404, 0xfa000000]
    if filtered:
        words += [0xbe980480, 0xbe9a0480, vop1(1, 2, 128), vop1(1, 3, 128)]
    else:
        words += [vop2(0x1b, 2, 255, 0), 0xffff, vop2(0x16, 3, 144, 0)]
    if dim in ('3d', '2da'):
        words += [vop1(1, 4, 128)]
    # Three coordinates use NSA so the implicit-mip form has exactly three address operands.
    nsa = dim in ('3d', '2da')
    opcode = 0x27 if filtered else int(explicit_mip)
    words += [0xbf8cc07f, 0xf0000100 | DIMENSIONS[dim] << 3 | (2 if nsa else 0) | opcode << 18,
              0x00040902 | ((6 << 21) if filtered else 0)]
    if nsa:
        words += [0x0403]
    words += [0xbf8c3f70,
              0xf8000001, 9, vop1(1, 10, 128), 0xf800181f, 0x0a0a0a0a, 0xbf810000]
    return words


def check_integer_sources():
    def pipe(words):
        return G.Pipeline(VS_WORDS, struct.pack(f'<{len(words)}I', *words), root_blob(), rtv_formats=(42, 28))

    with patch.dict(os.environ, X2W_INTEGER_TEXTURE_SRVS='0'):
        legacy = pipe(identifier_pixel())
        legacy_hlsl = legacy.hlsl_ps()
        assert 'srvint' not in legacy.manifest() and '_is_integer' not in legacy_hlsl
        sample_hlsl = pipe(identifier_pixel(filtered=True)).hlsl_ps()

    with patch.dict(os.environ, X2W_INTEGER_TEXTURE_SRVS='1'):
        for dim, shape in zip(DIMENSIONS, ('1D', '2D', '3D', '2DArray')):
            typed = pipe(identifier_pixel(dim=dim))
            hlsl = typed.hlsl_ps()
            assert f'Texture{shape}<uint4> r0_uint : register(t1);' in hlsl
            assert f'Texture{shape}<int4> r0_sint : register(t2);' in hlsl
            assert 'i.pos.x' in hlsl and f'srv 0 tex{dim} table 0 0' in typed.manifest()
            assert 'srvint 0 1 4' in typed.manifest()
            for mutant in ('inverted', 'swapped registers'):
                assert typed_mutant(hlsl, mutant) != hlsl
        filtered = pipe(identifier_pixel(filtered=True))
        assert filtered.hlsl_ps() == sample_hlsl and 'srvint' not in filtered.manifest()
        assert 'srvtypes 1' in filtered.manifest()
    print('Typed SRV lowering: four dimensions, UINT and SINT declarations, filtered HLSL unchanged')


def typed_mutant(source, kind):
    if kind == 'inverted':
        assert '_is_integer != 0u' in source
        return source.replace('_is_integer != 0u', '_is_integer == 0u')
    float_decl = re.search(r'(Texture\w+)<float4> (r\d+) : register\(t(\d+)\);', source)
    assert float_decl
    twin_decl = re.search(float_decl[1] + r'<uint4> ' + float_decl[2] + r'_uint : register\(t(\d+)\);', source)
    assert twin_decl
    return (source.replace(float_decl[0], float_decl[0].replace(f't{float_decl[3]}', f't{twin_decl[1]}'))
                  .replace(twin_decl[0], twin_decl[0].replace(f't{twin_decl[1]}', f't{float_decl[3]}')))


def debug_errors(result):
    return re.findall(r'x2w-d3d12-debug[^\n]*sev=[01]\b', result.stderr)


def identifier_draw(dxc, consumer, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        vs, ps, root, pixels = (work / name for name in ('vs', 'ps', 'root', 'pixels'))
        vs.write_bytes(vertex())
        cases = [(dim, False, True, dim, False) for dim in DIMENSIONS]
        cases += [('1d-mip', False, True, '1d', True), ('filtered', True, True, '2d', False),
                  ('legacy', False, False, '2d', False)]
        for name, filtered, typed, dim, explicit_mip in cases:
            cache = work / name
            cache.mkdir()
            ps.write_bytes(container(0, identifier_pixel(filtered, dim, explicit_mip)))
            env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
                       X2W_D3D12_DEBUG='1', X2W_TEST_DEBUG_LAYER='1', X2W_FXC='', X2W_SHADER_CACHE=str(cache),
                       X2W_INTEGER_TEXTURE_SRVS='1' if typed else '0', X2W_TEST_IDENTIFIER_DIMENSION=dim)
            if filtered:
                env['X2W_TEST_IDENTIFIER_FILTERED'] = '1'
            args = [consumer, vs, ps, root, adapter, pixels, '42']

            def run(fmt):
                result = subprocess.run(args, env=dict(env, X2W_TEST_IDENTIFIER_SRV=fmt),
                                        capture_output=True, text=True)
                if result.returncode == 77:
                    raise SystemExit(77)
                assert result.returncode == 0, result.stderr
                assert 'x2w-d3d12 debug layer: 0x00000000' in result.stderr, result.stderr
                raw = pixels.read_bytes()
                assert len(raw) == 40, len(raw)
                return struct.unpack_from('<5I', raw), result

            run('float')
            req, = cache.glob('req_*.rtv')
            key = req.stem[4:]
            subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], env=env, check=True)
            lines = (cache / f'{key}.txt').read_text().splitlines()
            assert ('srvtypes 1' in lines) == typed
            assert any(line.startswith('srvint ') for line in lines) == (typed and not filtered)
            for stage in ('vs', 'ps'):
                stem = cache / f'{key}.{stage}'
                subprocess.run([dxc, '-T', f'{stage}_6_0', '-E', 'main', '-Fo', f'{stem}.cso', f'{stem}.hlsl'],
                               check=True)
            if filtered:
                actual, result = run('uint')
                assert actual == (0,) * 5 and not debug_errors(result), (actual, result.stderr)
                assert '-> drawn;' in result.stderr and result.stderr.count('srv 0: integer_view_filtered') == 1, result.stderr
                continue

            actual, result = run('float')
            assert actual == FLOAT_BITS and not debug_errors(result), (actual, result.stderr)
            if not typed:
                continue
            for fmt in ('uint', 'sint'):
                actual, result = run(fmt)
                assert actual == IDENTIFIERS and not debug_errors(result), (fmt, actual, result.stderr)
            source = Path(f'{stem}.hlsl').read_text()
            mutant = work / 'mutant.hlsl'
            for kind in ('inverted', 'swapped registers'):
                mutant.write_text(typed_mutant(source, kind), encoding='ascii')
                subprocess.run([dxc, '-T', 'ps_6_0', '-E', 'main', '-Fo', f'{stem}.cso', mutant], check=True)
                actual, result = run('uint')
                assert '-> drawn;' in result.stderr, result.stderr
                assert actual != IDENTIFIERS or debug_errors(result), (kind, actual, result.stderr)
    print(f'Identifier RTV on {adapter}: all four dimensions match UINT, SINT and FLOAT readback. '
          'Both mutants rejected in each dimension, filtered null and legacy bindings checked.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc', help='reassemble the synthetic fixtures')
    parser.add_argument('--draw', nargs=3, metavar=('DXC', 'RENDERER', 'ADAPTER'))
    parser.add_argument('--consumer', help='wrapped draw fixture for typed SRVs')
    args = parser.parse_args()
    check(args.llvm_mc)
    check_integer_sources()
    if args.draw:
        result = draw(*args.draw)
        if not result and args.consumer:
            identifier_draw(args.draw[0], args.consumer, args.draw[2])
        raise SystemExit(result)
