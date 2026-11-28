"""3D image sampling in tools/startup/gfx2hlsl.py (dim:SQ_RSRC_IMG_3D, #1149).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given) and
                                       draw: the readback must name the texel the 3D coordinate selects

The fixture is a source-built gfx1013 pixel stage. It loads a texture descriptor from table 0, builds
an all-zero inline sampler (s_mov_b32), takes x = 0.75 and y = 0.25 from literals and z from root
constant 1 (s9), samples with dim:SQ_RSRC_IMG_3D and exports the result. tests/startup_image_3d.cpp
binds a 2x2x2 R32G32B32A32_FLOAT texture whose texel (x, y, z) holds ((1 + x + 2y + 4z) / 8, x, y, z)
and a point sampler with wrap addressing.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

# Pixel stage: s8 = descriptor table 0, s9 = root constant 1 (the z coordinate's bits).
PS_SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, 0
s_mov_b32 s26, 0
s_mov_b32 s27, 0
v_mov_b32 v0, 0x3f400000
v_mov_b32 v1, 0x3e800000
v_mov_b32 v2, s9
s_waitcnt lgkmcnt(0)
image_sample v[0:3], v[0:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_3D
s_waitcnt vmcnt(0)
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
"""
SAMPLE_3D = bytes.fromhex('100f80f00000c400')
PS_WORDS = bytes.fromhex(
    '04040cf4000000fa' '800398be' '800399be' '80039abe' '80039bbe' 'ff02007e0000403f' 'ff02027e0000803e'
    '0902047e' '7fc08cbf') + SAMPLE_3D + bytes.fromhex('703f8cbf' '0f1800f800010203' '000081bf')

# The same program with dim:SQ_RSRC_IMG_2D (two address registers), and with dim:SQ_RSRC_IMG_1D.
SOURCE_2D = PS_SOURCE.replace('v[0:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_3D',
                              'v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D')
WORDS_2D = PS_WORDS.replace(SAMPLE_3D, bytes.fromhex('080f80f00000c400'))
SOURCE_1D = PS_SOURCE.replace('v[0:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_3D',
                              'v0, s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_1D')
WORDS_1D = PS_WORDS.replace(SAMPLE_3D, bytes.fromhex('000f80f00000c400'))

VS_WORDS = bytes.fromhex('f202087e' '80020a7e' 'cf0800f805050504' '000081bf')

# Desktop full-screen triangle for the draw; the translated pixel stage reads only SV_Position.
FULLSCREEN_VS = """\
float4 main(uint id : SV_VertexID) : SV_Position
{
    return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
}
"""

SAMPLE_CALL = 'r0.Sample(q0, float3(F(v0), F(v1), F(v2)))'


def root_blob():
    """A DXBC container with one RTS0 part: parameter 0 a descriptor table, parameter 1 one root constant."""
    params = [(0, 0), (1, 0)]  # (type, visibility)
    poff = 24
    coff = poff + 12 * len(params)
    rts0 = struct.pack('<6I', 2, len(params), poff, 0, 0, 0)
    rts0 += struct.pack('<3I', 0, 0, coff + 12) + struct.pack('<3I', 1, 0, coff)
    rts0 += struct.pack('<3I', 0, 0, 1)        # constants: b0, space 0, one value
    rts0 += struct.pack('<5I', 0, 0, 0, 0, 0)  # table body (gfx2hlsl does not read it)
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return (b'DXBC' + bytes(16) + struct.pack('<III', 1, 32 + 4 + len(part), 1) + struct.pack('<I', 36) + part)


def translate(ps_words):
    return G.Pipeline(VS_WORDS, ps_words, root_blob())


def controls():
    # dim:SQ_RSRC_IMG_3D: a Texture3D, sampled with all three address registers in order.
    pipe = translate(PS_WORDS)
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'Texture3D<float4> r0 : register(t0);' in ps and 'Texture2D' not in ps, ps
    assert 't = %s;' % SAMPLE_CALL in ps, ps
    assert re.search(r'\bv2 = ud1;', ps), ps
    assert 'srv 0 tex3d table 0 0' in manifest and 'udsrc 1 rootconst 1 0' in manifest, manifest
    assert 'smp 0 0x0 0x0 0x0 0x0' in manifest and not pipe.notes, (manifest, pipe.notes)
    print('dim 3D: Texture3D r0, %s with v2 = ud1 (root constant 1); manifest srv 0 tex3d' % SAMPLE_CALL)

    # Only the dim field differs: dim:SQ_RSRC_IMG_2D gives a Texture2D and a float2 coordinate.
    pipe = translate(WORDS_2D)
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'Texture2D<float4> r0 : register(t0);' in ps and 'Texture3D' not in ps, ps
    assert 't = r0.Sample(q0, float2(F(v0), F(v1)));' in ps, ps
    assert 'srv 0 tex2d table 0 0' in manifest, manifest
    print('dim 2D (same words, dim field changed): Texture2D r0, float2(F(v0), F(v1)); manifest srv 0 tex2d')

    pipe = translate(WORDS_1D)
    assert 'Texture1D<float4> r0' in pipe.hlsl_ps()
    assert 'r0.Sample(q0, F(v0))' in pipe.hlsl_ps()
    assert 'srv 0 tex1d table 0 0' in pipe.manifest()
    # The MIMG words carry no address count: the disassembler sizes the address from the dimension. With the cube
    # dimension (field 3) the same words are a three-register cube sample, translated since #2943; the 2D MSAA
    # dimension (field 6) stays an unverified dimension and must be refused.
    cube = translate(WORDS_1D.replace(bytes.fromhex('000f80f0'), bytes.fromhex('180f80f0'))).hlsl_ps()
    assert 'TextureCube<float4> r0' in cube and 'r0.Sample(q0, x2w_cube_dir(F(v0), F(v1), F(v2)))' in cube, cube
    unsupported = WORDS_1D.replace(bytes.fromhex('000f80f0'), bytes.fromhex('300f80f0'))
    try:
        translate(unsupported)
    except G.Unsupported as error:
        assert 'image dimension SQ_RSRC_IMG_2D_MSAA' in str(error), error
    else:
        raise AssertionError('MSAA image dimension accepted')


def assemble(llvm_mc, source):
    run = subprocess.run([llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-show-encoding'], input=source,
                         capture_output=True, text=True, check=True)
    words = bytearray()
    for line in run.stdout.splitlines():
        m = re.search(r'encoding: \[([^\]]*)\]', line)
        if m:
            words += bytes(int(b, 16) for b in m.group(1).split(','))
    return bytes(words)


def check_binary(llvm_mc):
    for name, source, words in (('3D pixel', PS_SOURCE, PS_WORDS), ('2D pixel', SOURCE_2D, WORDS_2D),
                                ('1D pixel', SOURCE_1D, WORDS_1D)):
        built = assemble(llvm_mc, source)
        if built != words:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    print('fixture words match llvm-mc')


def compile_hlsl(compiler, kind, source, out, profile, directory):
    path = directory / ('%s.hlsl' % out)
    path.write_text(source, encoding='ascii')
    if kind == 'fxc':
        args = [compiler, '/nologo', '/T', profile, '/E', 'main', '/Fo', str(directory / ('%s.cso' % out)), str(path)]
    else:
        args = [compiler, '-T', profile, '-E', 'main', '-Fo', str(directory / ('%s.cso' % out)), str(path)]
    subprocess.run(args, check=True, capture_output=True, text=True)
    return directory / ('%s.cso' % out)


class CompileQueue:
    """Runs independent shader compiles on a thread pool, and hands each result back in source order.

    submit() starts a compile; call() returns that compile's result, or compiles inline when nothing was
    submitted for the same arguments. A failed compile raises where call() reaches it, so the first failure in
    source order is still the one reported. Every submitted compile needs its own output name."""

    def __init__(self, workers=None):
        self.pool = ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1))
        self.futures = {}

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def submit(self, function, *args):
        assert (function, args) not in self.futures, 'duplicate compile %r' % (args[3],)
        self.futures[(function, args)] = self.pool.submit(function, *args)

    def call(self, function, *args):
        future = self.futures.pop((function, args), None)
        return future.result() if future else function(*args)


def texel(x, y, z):
    """The fixture texture's value at integer texel (x, y, z), as four binary32 hex words."""
    return ','.join('%08x' % struct.unpack('<I', struct.pack('<f', c))[0]
                    for c in ((1 + x + 2 * y + 4 * z) / 8, x, y, z))


def check_draw(dxc, renderer, adapter, fxc=None):
    # (z, expected): x = 0.75 and y = 0.25 select texel column 1, row 0. z = 0.25 and 0.75 select the
    # two slices; 1.75 wraps to slice 1.
    cases = [('0.25', texel(1, 0, 0)), ('0.75', texel(1, 0, 1)), ('1.75', texel(1, 0, 1))]
    # Negative controls on the translated HLSL: z replaced by a constant in the first slice, and the x and z
    # address registers exchanged. Each must disagree with the texel the fixture's coordinate selects.
    mutants = [('z fixed to slice 0', SAMPLE_CALL.replace('F(v2)', '0.25'), '0.75', texel(1, 0, 0)),
               ('x and z exchanged', 'r0.Sample(q0, float3(F(v2), F(v1), F(v0)))', '0.25', texel(0, 0, 1))]
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    translated = translate(PS_WORDS).hlsl_ps()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'fullscreen_%s' % kind, vs_profile, directory)
            shader = compile_hlsl(compiler, kind, translated, 'ps_%s' % kind, ps_profile, directory)
            # Select only alpha, export it through all four channels, and change the input x.
            alpha = WORDS_1D.replace(bytes.fromhex('000f80f0'), bytes.fromhex('000880f0')).replace(
                bytes.fromhex('0f1800f800010203'), bytes.fromhex('0f1800f800000000'))
            for word, expected in [(0x3f400000, '3f400000'), (0x3e800000, '3f000000')]:
                code = alpha.replace(struct.pack('<I', 0x3f400000), struct.pack('<I', word))
                text = translate(code).hlsl_ps()
                ps1 = compile_hlsl(compiler, kind, text, 'ps_1d_' + kind, ps_profile, directory)
                args = [renderer, adapter, '--startup-image-1d', str(vs), str(ps1), '0', ','.join([expected] * 4)]
                run = subprocess.run(args, capture_output=True, text=True)
                if run.returncode == 77:
                    sys.exit(77)
                assert run.returncode == 0, (run.stdout, run.stderr)
                mutant = text.replace('U(t.w)', 'U(t.x)')
                assert mutant != text
                args[4] = str(compile_hlsl(compiler, kind, mutant, 'ps_1d_wrong', ps_profile, directory))
                failed = subprocess.run(args, capture_output=True, text=True)
                wrong = '3e800000' if word == 0x3f400000 else '3e000000'
                assert failed.returncode != 0 and 'actual=' + ','.join([wrong] * 4) in failed.stderr, failed.stderr
            for z, expected in cases:
                run = subprocess.run([renderer, adapter, '--startup-image-3d', str(vs), str(shader), z, expected],
                                     capture_output=True, text=True)
                sys.stdout.write(run.stdout)
                if run.returncode == 77:
                    print(run.stdout + run.stderr)
                    sys.exit(77)
                if run.returncode:
                    raise AssertionError('%s z=%s\n%s\n%s' % (kind, z, run.stdout, run.stderr))
                print('%s z=%s: %s' % (kind, z, expected), flush=True)
            for label, call, z, actual in mutants:
                source = translated.replace(SAMPLE_CALL, call)
                assert source != translated
                mutant = compile_hlsl(compiler, kind, source, 'ps_mutant_%s' % kind, ps_profile, directory)
                expected = dict(cases)[z]
                run = subprocess.run([renderer, adapter, '--startup-image-3d', str(vs), str(mutant), z, expected],
                                     capture_output=True, text=True)
                sys.stdout.write(run.stdout)
                if run.returncode == 0 or 'actual=%s' % actual not in run.stderr:
                    raise AssertionError('%s %s: control did not read texel %s\n%s\n%s'
                                         % (kind, label, actual, run.stdout, run.stderr))
                print('%s %s: fails as expected, reads %s' % (kind, label, actual), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        controls()
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
