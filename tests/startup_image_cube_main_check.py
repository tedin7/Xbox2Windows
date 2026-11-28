"""Cube images in tools/startup/gfx2hlsl.py (dim:SQ_RSRC_IMG_CUBE, #2943).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware>
                                       translate, compile (DXC ps_6_0) and draw against a six-face TextureCube

A gfx10 cube sample addresses the face coordinates s and t and then the face id. Mesa's prepare_cube_coords
(src/amd/common/nir/ac_nir_lower_tex.c) builds s = sc/|ma| + 1.5 and t = tc/|ma| + 1.5 from v_cubesc, v_cubetc
and v_cubema, and LLVM's ConstantFoldAMDGCNCubeIntrinsic defines v_cubema as twice the major coordinate, so s and
t lie in [1, 2] and s - 1, t - 1 are the position on the face. The translator rebuilds a direction from (s, t,
face) for TextureCube (x2w_cube_dir in the HLSL).

tests/startup_image_3d.cpp, run as --startup-image-cube, binds six 4x4 R32G32B32A32_FLOAT faces as a TextureCube
whose texel (x, y) of face f holds ((1 + x + 4y + 16f) / 128, x, y, f), so all 96 texels differ. The sampler is the
all-zero inline sampler (point, wrap); every (s, t) is a texel centre (in {1.125, 1.375, 1.625, 1.875}).

What the draws show: the literal cases feed (s, t, face) directly, without v_cube, and the reference is only the
address meaning above (x = floor(4 * (s - 1)), y = floor(4 * (t - 1))). The desktop driver selects the
face of the rebuilt direction itself, so a wrong face, sign, axis or 2.0 * (s - 1.5) / 2.0 * (t - 1.5) scale in
x2w_cube_dir reads another texel. The mutants edit one row of the emitted helper (sign of the u term flipped, u and v
exchanged, faces 2 and 3 exchanged, and dropping the 2.0 * scale on u or v) and must read the texel a D3D cube
lookup of the mutated direction gives. The end-to-end cases run the v_cube chain on six directions and must equal
both that reference and a desktop PS that samples the direction.
They do not show that the console's face order and orientation equal D3D's: that rests on LLVM's cube table
being the D3D11.3 cube face table. Only one mip is bound, so LOD and bias values are not tested; the image_sample_l
and image_sample_b draws only show that the extra address dword is not taken as a coordinate.
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, assemble, compile_hlsl, root_blob  # noqa: E402

INLINE_FLOATS = {0.0: 128, 0.5: 240, -0.5: 241, 1.0: 242, -1.0: 243, 2.0: 244, -2.0: 245, 4.0: 246, -4.0: 247}
PROLOGUE = bytes.fromhex('04040cf4000000fa' '800398be' '800399be' '80039abe' '80039bbe')
PROLOGUE_SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, 0
s_mov_b32 s26, 0
s_mov_b32 s27, 0
"""
EPILOGUE = bytes.fromhex('703f8cbf' '0f1800f800010203' '000081bf')
EPILOGUE_SOURCE = """\
s_waitcnt vmcnt(0)
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
"""
WAIT = bytes.fromhex('7fc08cbf')
# First and second dword of each MIMG form on dim:SQ_RSRC_IMG_CUBE (field value 3), vdata v[0:3], vaddr v0,
# s[16:23] and s[24:27]; recorded from LLVM 15.
MIMG = {
    'sample': ('image_sample v[0:3], v[0:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_CUBE', '180f80f00000c400'),
    'lz': ('image_sample_lz v[0:3], v[0:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_CUBE', '180f9cf00000c400'),
    'l': ('image_sample_l v[0:3], v[0:3], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_CUBE', '180f90f00000c400'),
    'b': ('image_sample_b v[0:3], v[0:3], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_CUBE', '180f94f00000c400'),
}


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def v_mov(dst, value):
    """v_mov_b32 vdst, value: an inline constant where LLVM uses one, else a literal."""
    if value in INLINE_FLOATS:
        return struct.pack('<I', 0x7e000200 | dst << 17 | INLINE_FLOATS[value]), 'v_mov_b32 v%d, %s' % (dst, value)
    return (struct.pack('<I', 0x7e0002ff | dst << 17) + struct.pack('<I', bits(value)),
            'v_mov_b32 v%d, 0x%08x' % (dst, bits(value)))


# image_sample_b needs a sampler with mipmapping (word 2 mip filter 1) and MAX_LOD 2 (word 1), as in
# startup_image_sample_b_check.py SAMPLER; only one mip is bound, so neither changes the texel.
B_SAMPLER = bytes.fromhex('04040cf4000000fa' '800398be' 'ff0399be00002000' 'ff039abe00000004' '80039bbe')
B_SAMPLER_SOURCE = PROLOGUE_SOURCE.replace('s_mov_b32 s25, 0', 's_mov_b32 s25, 0x200000').replace(
    's_mov_b32 s26, 0', 's_mov_b32 s26, 0x4000000')


def program(moves, form='lz', middle=b'', middle_source=''):
    """The prologue, the given v_mov constants, an optional middle part, one cube sample and the export."""
    words, source = (bytearray(B_SAMPLER), B_SAMPLER_SOURCE) if form == 'b' else (bytearray(PROLOGUE), PROLOGUE_SOURCE)
    for dst, value in moves:
        w, s = v_mov(dst, value)
        words += w
        source += s + '\n'
    words += middle
    source += middle_source
    text, mimg = MIMG[form]
    words += WAIT + bytes.fromhex(mimg) + EPILOGUE
    source += 's_waitcnt lgkmcnt(0)\n' + text + '\n' + EPILOGUE_SOURCE
    return bytes(words), source


def texel(x, y, face):
    return ','.join('%08x' % bits(c) for c in ((1 + x + 4 * y + 16 * face) / 128, x, y, face))


def address_texel(s, t, face):
    """The AMD cube address meaning: face-local position s - 1 and t - 1 on a 4x4 face."""
    return (min(3, max(0, int((s - 1.0) * 4))), min(3, max(0, int((t - 1.0) * 4))), face)


# The per-face (sc, tc) of v_cubesc and v_cubetc (LLVM ConstantFoldAMDGCNCubeIntrinsic), used here to model a D3D
# TextureCube lookup of a direction; LLVM's table is the D3D11.3 face table.
def cube_lookup(d):
    x, y, z = d
    ax, ay, az = abs(x), abs(y), abs(z)
    if az >= ax and az >= ay:
        face, sc, tc, ma = (5, -x, -y, z) if z < 0 else (4, x, -y, z)
    elif ay >= ax:
        face, sc, tc, ma = (3, x, -z, y) if y < 0 else (2, x, z, y)
    else:
        face, sc, tc, ma = (1, z, -y, x) if x < 0 else (0, -z, -y, x)
    inv = 1.0 / abs(2 * ma)
    return address_texel(sc * inv + 1.5, tc * inv + 1.5, face)


# The emitted helper rows, face 0 to 5, as (x, y, z) expressions in u and v.
ROWS = [('1.0', '-v', '-u'), ('-1.0', '-v', 'u'), ('u', '1.0', 'v'), ('u', '-1.0', '-v'), ('u', '-v', '1.0'),
        ('-u', '-v', '-1.0')]


def row_line(face, row):
    body = 'return float3(%s, %s, %s);' % row
    return body if face == 5 else 'if (k == %d) %s' % (face, body)


def helper_dir(rows, s, t, face, su=2.0, sv=2.0):
    u, v = su * (s - 1.5), sv * (t - 1.5)
    return tuple(eval(c, {}, {'u': u, 'v': v}) for c in rows[face])  # noqa: S307 (fixed expressions above)


def flip_u(row):
    return tuple(c.replace('-u', '#').replace('u', '-u').replace('#', 'u') for c in row)


def swap_uv(row):
    return tuple(c.replace('u', '#').replace('v', 'u').replace('#', 'v') for c in row)


def mutants():
    """(label, edits [(old line, new line)], mutated rows, (su, sv))."""
    out = []
    for face in range(6):
        for label, change in (('u sign flipped', flip_u), ('u and v exchanged', swap_uv)):
            rows = list(ROWS)
            rows[face] = change(ROWS[face])
            out.append(('face %d %s' % (face, label), [(row_line(face, ROWS[face]), row_line(face, rows[face]))],
                        rows, (2.0, 2.0)))
    rows = list(ROWS)
    rows[2], rows[3] = ROWS[3], ROWS[2]
    out.append(('faces 2 and 3 exchanged', [(row_line(2, ROWS[2]), row_line(2, ROWS[3])),
                                            (row_line(3, ROWS[3]), row_line(3, ROWS[2]))], rows, (2.0, 2.0)))
    out.append(('u scale 2 dropped', [('clamp(2.0 * (s - 1.5), -1.0, 1.0)', 'clamp((s - 1.5), -1.0, 1.0)')],
                ROWS, (1.0, 2.0)))
    out.append(('v scale 2 dropped', [('clamp(2.0 * (t - 1.5), -1.0, 1.0)', 'clamp((t - 1.5), -1.0, 1.0)')],
                ROWS, (2.0, 1.0)))
    return out


# Literal (s, t, face) cases: outer-ring 4x4 texel centres so every face has x != y and dropping scale 2 moves texels.
CASES = [(1.125, 1.875, 0), (1.875, 1.125, 1), (1.125, 1.875, 2), (1.875, 1.125, 3), (1.875, 1.125, 4), (1.125, 1.875, 5),
         (1.875, 1.875, 2)]
# End-to-end directions, one per face, each on a 4x4 texel centre with x != y.
DIRECTIONS = [(1.0, -0.75, 0.25), (-1.0, -0.75, -0.25), (0.75, 1.0, -0.25), (0.75, -1.0, 0.25), (0.75, 0.25, 1.0),
              (-0.75, 0.25, -1.0)]
CHAIN_SOURCE = """\
v_cubema_f32 v7, v4, v5, v6
v_cubesc_f32 v8, v4, v5, v6
v_cubetc_f32 v9, v4, v5, v6
v_cubeid_f32 v2, v4, v5, v6
v_rcp_f32_e64 v7, |v7|
v_mov_b32 v10, 0x3fc00000
v_fma_f32 v0, v8, v7, v10
v_fma_f32 v1, v9, v7, v10
"""
CHAIN = bytes.fromhex('070047d5040b1a04' '080045d5040b1a04' '090046d5040b1a04' '020044d5040b1a04'
                      '0701aad507010000' 'ff02147e0000c03f' '00004bd5080f2a04' '01004bd5090f2a04')


def literal_program(s, t, face, form='lz'):
    if form in ('lz', 'sample'):
        return program([(0, s), (1, t), (2, float(face))], form)
    if form == 'l':
        return program([(0, s), (1, t), (2, float(face)), (3, 0.0)], form)
    return program([(0, 0.0), (1, s), (2, t), (3, float(face))], form)  # b: bias first


def chain_program(d):
    return program([(4, d[0]), (5, d[1]), (6, d[2])], 'lz', CHAIN, CHAIN_SOURCE)


def desktop_ps(d):
    return ('TextureCube<float4> r0 : register(t0);\nSamplerState q0 : register(s0);\n'
            'float4 main() : SV_Target { return r0.SampleLevel(q0, float3(%r, %r, %r), 0); }\n' % d)


def translate(words):
    return G.Pipeline(VS_WORDS, words, root_blob())


def expect_refused(call, needle):
    try:
        call()
    except G.Unsupported as error:
        assert needle in str(error), (needle, str(error))
        return str(error)
    raise AssertionError('accepted, expected refusal containing %r' % needle)


def bare_image(how, mods, kind='ps'):
    stage = G.Stage.__new__(G.Stage)
    stage.kind = kind
    return lambda: stage.image(['v[0:3]', 'v[0:2]', 's[16:23]', 's[24:27]'], mods, how)


def model_controls():
    values = {texel(x, y, f) for f in range(6) for y in range(4) for x in range(4)}
    assert len(values) == 96, len(values)
    for s, t, face in CASES:
        # The helper rebuilds a direction that a D3D lookup maps back to the address texel.
        assert cube_lookup(helper_dir(ROWS, s, t, face)) == address_texel(s, t, face), (s, t, face)
    assert {f for s, t, f in CASES if address_texel(s, t, f)[0] != address_texel(s, t, f)[1]} == set(range(6))
    for d in DIRECTIONS:
        x, y, face = cube_lookup(d)
        assert x != y, d
    assert sorted(cube_lookup(d)[2] for d in DIRECTIONS) == list(range(6))
    for label, _, rows, scale in mutants():
        assert any(cube_lookup(helper_dir(rows, s, t, f, *scale)) != address_texel(s, t, f)
                   for s, t, f in CASES), label
    print('model: 96 distinct texels; helper rows invert a D3D lookup on %d cases; %d mutants each change a case'
          % (len(CASES), len(mutants())))


def controls():
    model_controls()
    words, _ = literal_program(1.875, 1.125, 1)
    pipe = translate(words)
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'TextureCube<float4> r0 : register(t0);' in ps and 'Texture2DArray' not in ps, ps
    assert 'r0.SampleLevel(q0, x2w_cube_dir(F(v0), F(v1), F(v2)), 0)' in ps, ps
    assert 'float u = clamp(2.0 * (s - 1.5), -1.0, 1.0);' in ps, ps
    assert 'float v = clamp(2.0 * (t - 1.5), -1.0, 1.0);' in ps, ps
    for face, row in enumerate(ROWS):
        assert row_line(face, row) in ps, (face, ps)
    assert ps.count('float3 x2w_cube_dir(') == 1, ps
    assert 'srv 0 texcube table 0 0' in manifest.splitlines(), manifest
    for form, call in (('sample', 'r0.Sample(q0, x2w_cube_dir(F(v0), F(v1), F(v2)))'),
                       ('l', 'r0.SampleLevel(q0, x2w_cube_dir(F(v0), F(v1), F(v2)), F(v3))'),
                       ('b', 'r0.SampleBias(q0, x2w_cube_dir(F(v1), F(v2), F(v3)), F(v0))')):
        ps = translate(literal_program(1.875, 1.125, 1, form)[0]).hlsl_ps()
        assert call in ps, (form, ps)
    # Other dimensions keep their declarations and calls.
    from startup_image_2d_array_check import SAMPLE_WORDS, SAMPLE_CALL  # noqa: E402
    ps = translate(SAMPLE_WORDS).hlsl_ps()
    assert SAMPLE_CALL in ps and 'x2w_cube_dir' not in ps and 'TextureCube' not in ps, ps
    print('translation: TextureCube r0 with x2w_cube_dir for sample, lz, l and b; manifest srv 0 texcube')
    refusals = [
        (bare_image('load', ['dim:SQ_RSRC_IMG_CUBE', 'dmask:0xf']), 'image dimension SQ_RSRC_IMG_CUBE: Load'),
        (bare_image('load_mip', ['dim:SQ_RSRC_IMG_CUBE', 'dmask:0xf']), 'image dimension SQ_RSRC_IMG_CUBE: Load'),
        (bare_image('d', ['dim:SQ_RSRC_IMG_CUBE', 'dmask:0xf']), 'image_sample_d on SQ_RSRC_IMG_CUBE'),
        (bare_image('get_lod', ['dim:SQ_RSRC_IMG_CUBE', 'dmask:0x3']), 'image_get_lod on SQ_RSRC_IMG_CUBE'),
        (bare_image('lz', ['dim:SQ_RSRC_IMG_CUBE', 'dmask:0xf', 'a16']), 'image_sample_lz on SQ_RSRC_IMG_CUBE with a16'),
        (bare_image('sample', ['dim:SQ_RSRC_IMG_CUBE', 'dmask:0xf', 'd16']),
         'image_sample on SQ_RSRC_IMG_CUBE with d16'),
        (bare_image('l', ['dim:SQ_RSRC_IMG_CUBE', 'dmask:0xf'], kind='cs'), 'cube image in a compute program'),
    ]
    for call, needle in refusals:
        expect_refused(call, needle)
    # A constant sampler with DISABLE_CUBE_WRAP (word 0 bit 28) is refused; the bit clear is accepted above.
    wrap = bytes.fromhex('ff0398be00000010')
    words = literal_program(1.75, 1.25, 1)[0].replace(bytes.fromhex('800398be'), wrap, 1)
    expect_refused(lambda: translate(words), 'DISABLE_CUBE_WRAP')
    print('refused: cube load and load_mip, sample_d, get_lod, a16, d16, compute, DISABLE_CUBE_WRAP')


def check_binary(llvm_mc):
    fixtures = [('literal %s' % (case,), *reversed(literal_program(*case))) for case in CASES]
    fixtures += [('%s' % form, *reversed(literal_program(1.875, 1.125, 1, form))) for form in ('sample', 'l', 'b')]
    fixtures += [('chain %s' % (d,), *reversed(chain_program(d))) for d in DIRECTIONS]
    for name, source, words in fixtures:
        built = assemble(llvm_mc, source)
        if built != words:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    print('fixture words match llvm-mc (%d programs)' % len(fixtures))


def check_draw(dxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', directory)
        count = [0, 0]  # reference draws that matched, controls and mutants that failed as predicted

        def draw(shader, expected, label, fail_reads=None, sampler=None):
            run = subprocess.run([renderer, adapter, '--startup-image-cube', str(vs), str(shader), '0', expected]
                                 + ([sampler] if sampler else []), capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                sys.exit(77)
            if not any(count):
                for line in run.stdout.splitlines():
                    if line.startswith(('backend=', 'adapter=', 'driver=')):
                        print(line)
            if 'debug_layer=enabled' in run.stdout:
                assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
            if fail_reads is None:
                if run.returncode:
                    raise AssertionError('%s\n%s\n%s' % (label, run.stdout, run.stderr))
                print('%s: %s' % (label, expected), flush=True)
                count[0] += 1
            else:
                if run.returncode == 0 or 'actual=%s' % fail_reads not in run.stderr:
                    raise AssertionError('%s: control did not read %s\n%s\n%s'
                                         % (label, fail_reads, run.stdout, run.stderr))
                print('%s: fails as expected, reads %s' % (label, fail_reads), flush=True)
                count[1] += 1

        def build(words, name):
            return compile_hlsl(dxc, 'dxc', translate(words).hlsl_ps(), name, 'ps_6_0', directory)

        def sampler_line(words):
            lines = [line for line in translate(words).manifest().splitlines() if line.startswith(('smp ', 'smpl '))]
            assert len(lines) == 1, lines
            return lines[0]

        for i, (s, t, face) in enumerate(CASES):
            draw(build(literal_program(s, t, face)[0], 'lit%d' % i), texel(*address_texel(s, t, face)),
                 'image_sample_lz s=%g t=%g face=%d' % (s, t, face))
        case = (1.875, 1.125, 1)
        for form in ('sample', 'l', 'b'):
            words = literal_program(*case, form)[0]
            draw(build(words, 'form_%s' % form), texel(*address_texel(*case)),
                 'image_sample%s s=%g t=%g face=%d' % ('' if form == 'sample' else '_' + form, *case),
                 sampler=sampler_line(words))
        # Guest-input controls: the face one higher and s, t exchanged must read those texels.
        s, t, face = case
        draw(build(literal_program(s, t, face + 1)[0], 'face_plus'), texel(*address_texel(*case)),
             'control face + 1', texel(*address_texel(s, t, face + 1)))
        draw(build(literal_program(t, s, face)[0], 'swap_st'), texel(*address_texel(*case)),
             'control s and t exchanged', texel(*address_texel(t, s, face)))
        # l and b with the face in the slot the translator reads as LOD or bias: another texel.
        draw(build(program([(0, s), (1, t), (2, 0.0), (3, float(face))], 'l')[0], 'l_swapped'),
             texel(*address_texel(*case)), 'control image_sample_l face and lod exchanged',
             texel(*address_texel(s, t, 0)))
        words = program([(0, float(face)), (1, s), (2, t), (3, 0.0)], 'b')[0]
        draw(build(words, 'b_swapped'), texel(*address_texel(*case)), 'control image_sample_b face and bias exchanged',
             texel(*address_texel(s, t, 0)), sampler=sampler_line(words))
        # Translator mutants: each edits the emitted helper and must read the texel a D3D lookup of its direction gives.
        for index, (label, edits, rows, scale) in enumerate(mutants()):
            hit = next((c for c in CASES if cube_lookup(helper_dir(rows, *c, *scale)) != address_texel(*c)))
            source = translate(literal_program(*hit)[0]).hlsl_ps()
            mutated = source
            for old, new in edits:
                assert mutated.count(old) == 1, (label, old)
                mutated = mutated.replace(old, new)
            assert mutated != source, label
            shader = compile_hlsl(dxc, 'dxc', mutated, 'mutant%d' % index, 'ps_6_0', directory)
            draw(shader, texel(*address_texel(*hit)), 'mutant %s (case %s)' % (label, hit),
                 texel(*cube_lookup(helper_dir(rows, *hit, *scale))))
        # End to end: the v_cube chain on six directions, against the reference and a desktop TextureCube draw.
        for i, d in enumerate(DIRECTIONS):
            expected = texel(*cube_lookup(d))
            draw(build(chain_program(d)[0], 'chain%d' % i), expected, 'v_cube chain d=%s' % (d,))
            native = compile_hlsl(dxc, 'dxc', desktop_ps(d), 'native%d' % i, 'ps_6_0', directory)
            draw(native, expected, 'desktop TextureCube d=%s' % (d,))
        print('%d cube draws matched their references; %d controls and mutants read the texel they predict'
              % tuple(count))


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(args[0])
    elif mode == 'draw':
        controls()
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
