"""Cube-map sampling in a pixel stage (image_sample, dim:SQ_RSRC_IMG_CUBE).

usage: startup_image_cube_check.py [binary <llvm-mc> | draw <dxc> <offscreen_check> <warp|hardware>]

  (no arguments)  translation controls, pure Python
  binary          the fixture words match llvm-mc
  draw            draw a pixel stage that samples the six faces of a cube map and read the colour back

The fixture is tests/startup_image_3d_check.py's pixel stage with a cube sample: S and T are literals (in [1, 2], the
ISA's range after the divide by |2 * major| and the 1.5 offset) and the face index is root constant 1. The texture holds
six 2x2 faces whose texel (x, y) of face f is ((1 + x + 2y + 4f) / 8, x, y, f), so each of the 24 texels is distinct. The
face-local texel is (floor((S - 1) * 2), floor((T - 1) * 2)), which does not use the direction table. Controls: a table
with the first two faces swapped and S and T exchanged.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_image_3d_check as T  # noqa: E402

G = T.G
SAMPLE_CUBE = bytes.fromhex('180f80f00000c400')  # image_sample v[0:3], v[0:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_CUBE
SOURCE = T.PS_SOURCE.replace('dim:SQ_RSRC_IMG_3D', 'dim:SQ_RSRC_IMG_CUBE')
WORDS = T.PS_WORDS.replace(T.SAMPLE_3D, SAMPLE_CUBE)
LITERAL_S, LITERAL_T = bytes.fromhex('ff02007e0000403f'), bytes.fromhex('ff02027e0000803e')
CALL = 'x2w_cube_dir(F(v0), F(v1), F(v2))'
# (S, T): texel (1, 0) and texel (0, 1), neither at a texel centre.
POINTS = ((1.7, 1.3), (1.2, 1.8))


def words(s, t):
    def literal(opcode, value):
        return bytes.fromhex(opcode) + struct.pack('<f', value)
    return WORDS.replace(LITERAL_S, literal('ff02007e', s)).replace(LITERAL_T, literal('ff02027e', t))


def expected(s, t, face):
    return T.texel(int((s - 1) * 2), int((t - 1) * 2), face)


def controls():
    pipe = T.translate(WORDS)
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'TextureCube<float4> r0 : register(t0);' in ps and 'Texture3D' not in ps, ps
    assert 't = r0.Sample(q0, %s);' % CALL in ps, ps
    assert ps.count('float3 x2w_cube_dir(float s, float t, float face) {') == 1
    assert 'srv 0 texcube table 0 0' in manifest, manifest
    # image_sample_lz and image_sample_l take the same address and give SampleLevel.
    lz = T.translate(WORDS.replace(SAMPLE_CUBE, bytes.fromhex('180f9cf00000c400'))).hlsl_ps()
    assert 'r0.SampleLevel(q0, %s, 0)' % CALL in lz, lz
    # The other 3D-shaped forms keep their refusals: a cube compare sample, and a cube map with fewer address registers.
    for fragment, word in (('image dimension SQ_RSRC_IMG_CUBE', bytes.fromhex('180f88f00000c400')),):
        try:
            T.translate(WORDS.replace(SAMPLE_CUBE, word))
        except G.Unsupported as reason:
            assert fragment in str(reason), str(reason)
        else:
            raise AssertionError('accepted %s' % word.hex())
    # A program without a cube map has no direction helper.
    assert 'x2w_cube_dir' not in T.translate(T.PS_WORDS).hlsl_ps()
    print('dim cube: TextureCube r0, %s; manifest srv 0 texcube' % CALL)


def check_binary(llvm_mc):
    assert T.assemble(llvm_mc, SOURCE) == WORDS, 'the fixture words no longer match llvm-mc'
    print('fixture words match llvm-mc')


def run(renderer, adapter, vs, ps, face, want):
    return subprocess.run([renderer, adapter, '--startup-image-cube', str(vs), str(ps), str(face), want],
                          capture_output=True, text=True)


def check_draw(dxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = T.compile_hlsl(dxc, 'dxc', T.FULLSCREEN_VS, 'fullscreen', 'vs_6_0', directory)
        for index, (s, t) in enumerate(POINTS):
            source = T.translate(words(s, t)).hlsl_ps()
            ps = T.compile_hlsl(dxc, 'dxc', source, 'ps_%d' % index, 'ps_6_0', directory)
            for face in range(6):
                want = expected(s, t, face)
                result = run(renderer, adapter, vs, ps, face, want)
                sys.stdout.write(result.stdout)
                if result.returncode == 77:
                    sys.exit(77)
                assert result.returncode == 0, 'face %d S=%g T=%g\n%s\n%s' % (face, s, t, result.stdout, result.stderr)
            print('S=%g T=%g: all six faces read texel (%d, %d)' % (s, t, int((s - 1) * 2), int((t - 1) * 2)), flush=True)
        # Mutants on the first point: faces 0 and 1 swapped in the table; S and T exchanged.
        s, t = POINTS[0]
        source = T.translate(words(s, t)).hlsl_ps()
        swapped = source.replace('f == 0u', 'f == @u').replace('f == 1u', 'f == 0u').replace('f == @u', 'f == 1u')
        transposed = source.replace(CALL, 'x2w_cube_dir(F(v1), F(v0), F(v2))')
        for label, text, face, actual in (('faces 0 and 1 swapped', swapped, 0, expected(s, t, 1)),
                                          ('S and T exchanged', transposed, 2, T.texel(0, 1, 2))):
            assert text != source, label
            ps = T.compile_hlsl(dxc, 'dxc', text, 'ps_mutant', 'ps_6_0', directory)
            result = run(renderer, adapter, vs, ps, face, expected(s, t, face))
            if result.returncode == 0 or 'actual=%s' % actual.replace(',', ',') not in result.stderr:
                raise AssertionError('%s: control did not read %s\n%s\n%s' % (label, actual, result.stdout, result.stderr))
            print('%s: fails as expected, reads %s' % (label, actual), flush=True)


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
