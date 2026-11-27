"""v_cube{id,sc,tc,ma}_f32 in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair and compare four output words per
                                       direction on a source-built full-screen triangle
  draw <dxc> <startup_cube_sample_check> warp --sample
                                       test cube sampling through the translated draw path on WARP

The four operations follow LLVM's ConstantFoldAMDGCNCubeIntrinsic. The major axis is z when |z| is at
least |x| and |y|, otherwise y when |y| is at least |x|, otherwise x. A face is negative only for a
nonzero, non-NaN negative coordinate. cubema returns twice the signed major coordinate.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_image_3d_check import root_blob as table_root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    POST_SOURCE, POST_WORDS, PRE_SOURCE, PRE_WORDS, vs_source, vs_words,
)

OPS = ('v_cubeid_f32', 'v_cubesc_f32', 'v_cubetc_f32', 'v_cubema_f32')
BODY_SOURCE = '''\
v_cubeid_f32 v12, v0, v4, v8
v_cubesc_f32 v13, v1, v5, v9
v_cubetc_f32 v14, v2, v6, v10
v_cubema_f32 v15, v3, v7, v11
'''
BODY_WORDS = bytes.fromhex(
    '0c0044d500092204' '0d0045d5010b2604' '0e0046d5020d2a04' '0f0047d5030f2e04')
# The first pair is swapped as a negative control: the same inputs then write sc to v12 and id to v13.
BAD_BODY = bytes.fromhex(
    '0c0045d500092204' '0d0044d5010b2604' '0e0046d5020d2a04' '0f0047d5030f2e04')

# Each direction covers a face; the first group also contains y/x and z/x ties and -0.0 on the major axis.
DIRECTIONS = (
    (2.0, -2.0, 1.0),
    (-1.5, 0.25, -1.5),
    (-3.0, 1.0, 2.0),
    (0.0, 0.0, -0.0),
    (1.0, 0.5, -0.25),
    (0.25, 1.0, 0.5),
    (0.5, -1.0, -0.5),
    (-0.25, -0.5, 1.0),
)


SAMPLE_POSITION = """\
v_cmp_eq_u32 vcc_lo, 1, v0
v_cndmask_b32_e64 v20, -1.0, 0x40400000, vcc_lo
v_cmp_eq_u32 vcc_lo, 2, v0
v_cndmask_b32_e64 v21, 1.0, 0xc0400000, vcc_lo
v_mov_b32 v22, 0
v_mov_b32 v23, 1.0
exp pos0 v20, v21, v22, v23 done
"""
SAMPLE_SETUP = """\
s_load_dwordx8 s[32:39], s[8:9], 0x0
s_mov_b32 s40, 0
s_mov_b32 s41, 0x100000
s_mov_b32 s42, 0x4000000
s_mov_b32 s43, 0
v_mov_b32 v40, 0
v_mov_b32 v41, s9
v_mov_b32 v42, s10
v_mov_b32 v43, s11
v_mov_b32 v44, 0
s_waitcnt lgkmcnt(0)
"""
SAMPLE_INSTRUCTION = 'image_sample{suffix} v[48:51], {address}, s[32:39], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_CUBE\n'
SAMPLE_PIXEL_OUT = 'exp mrt0 v48, v49, v50, v51 done vm\ns_endpgm\n'
SAMPLE_VERTEX_OUT = 'exp param0 v48, v49, v50, v51\ns_endpgm\n'
SAMPLE_PASS_PIXEL = 's_mov_b32 m0, s12\n' + ''.join(f'v_interp_mov_f32 v{48+i}, p0, attr0.{c}\n' for i,c in enumerate('xyzw'))

# Machine words for the assembly above (gfx1013).
SAMPLE_PARTS = (
    (SAMPLE_POSITION, bytes.fromhex(
        '8100847d140001d5f3fea901000040408200847d150001d5f2fea901000040c080022c7ef2022e7ecf0800f814151617')),
    (SAMPLE_SETUP, bytes.fromhex(
        '04080cf4000000fa8003a8beff03a9be00001000ff03aabe000000048003abbe8002507e0902527e0a02547e0b02567e8002587e7fc08cbf')),
    (SAMPLE_PIXEL_OUT, bytes.fromhex('0f1800f830313233000081bf')),
    (SAMPLE_VERTEX_OUT, bytes.fromhex('0f0200f830313233000081bf')),
    (SAMPLE_PASS_PIXEL, bytes.fromhex('0c03fcbe0200c2c80201c6c80202cac80203cec8')),
)
SAMPLE_FORMS = {
    '': ('v[41:43]', '180f80f029304801', 'Sample'),
    '_lz': ('v[41:43]', '180f9cf029304801', 'SampleLevel'),
    '_l': ('v[41:44]', '180f90f029304801', 'SampleLevel'),
    '_b': ('v[40:43]', '180f94f028304801', 'SampleBias'),
    '_d': ('[v40, v41, v42, v43, v44, v45, v46]', '1c0f88f028304801292a2b2c2d2e0000', None),
}
SAMPLE_DIRECTIONS = (
    (4, 1, -2), (-4, -1, 2), (1, 4, 2), (-1, -4, -2), (1, -2, 4), (-1, 2, -4),
    (31, 0, 32), (32, 0, 31),
)
FACE_COLOURS = (
    (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1),
    (1, 1, 0, 1), (0, 1, 1, 1), (1, 0, 1, 1),
)


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for one instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True


def one(assembly):
    Probe().one(assembly.split(' ', 1)[0], assembly)


def refused(assembly):
    try:
        one(assembly)
    except G.Unsupported as reason:
        return str(reason)
    return None


def word(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def direction_words(directions):
    """The flat parameters are 32-bit words, so pass the binary32 bits of each coordinate."""
    return tuple((word(x), word(y), word(z)) for x, y, z in directions)


def cube_reference(x, y, z):
    if abs(z) >= abs(x) and abs(z) >= abs(y):
        face, sc, tc, ma = (5.0, -x, -y, z) if z < 0 else (4.0, x, -y, z)
    elif abs(y) >= abs(x):
        face, sc, tc, ma = (3.0, x, -z, y) if y < 0 else (2.0, x, z, y)
    else:
        face, sc, tc, ma = (1.0, z, -y, x) if x < 0 else (0.0, -z, -y, x)
    return [face, sc, tc, ma + ma]


def expected_text(directions):
    return ','.join('%08x' % word(value) for d in directions for value in cube_reference(*d))


def translate(directions, body=BODY_WORDS):
    return G.Pipeline(vs_words(direction_words(directions)), PRE_WORDS + body + POST_WORDS, root_blob())


def controls():
    checks = {
        'v_cubeid_f32': ('? 5.0f : 4.0f', '? 3.0f : 2.0f', '? 1.0f : 0.0f'),
        'v_cubesc_f32': ('-cube0.x', 'cube0.x', '-cube0.z', 'cube0.z'),
        'v_cubetc_f32': ('-cube0.y', '-cube0.z', 'cube0.z'),
        'v_cubema_f32': ('* 2.0f',),
    }
    for op in OPS:
        stage = Probe()
        stage.one(op, '%s v12, v0, v4, v8' % op)
        assert stage.lines[0] == 'float3 cube0 = float3(F(v0), F(v4), F(v8));', stage.lines
        assert stage.lines[1].startswith('v12 = U('), stage.lines
        for needle in checks[op]:
            assert needle in stage.lines[1], (op, needle, stage.lines)
        print('%s: translated' % op)

    clamp = Probe()
    clamp.one('v_cubeid_f32', 'v_cubeid_f32 v12, v0, v4, v8 clamp')
    assert 'saturate(' in clamp.lines[1], clamp.lines
    print('clamp: saturates the result')

    modified = Probe()
    modified.one('v_cubeid_f32', 'v_cubeid_f32 v12, -v0, |v4|, v8')
    assert '(-F(v0))' in modified.lines[0] and 'abs(F(v4))' in modified.lines[0], modified.lines
    print('source neg/abs: applied before the cube operation')

    separate = Probe()
    separate.one('v_cubeid_f32', 'v_cubeid_f32 v12, v0, v4, v8')
    first_temp = separate.lines[0]
    separate.lines = []
    separate.one('v_cubema_f32', 'v_cubema_f32 v13, v1, v5, v9')
    assert 'float3 cube0' in first_temp and 'float3 cube1' in separate.lines[0], (first_temp, separate.lines)
    print('separate blocks: temporary names stay unique')

    for assembly, needle in (
        ('v_cubeid_f32 v12, v0, v4', 'expected 4'),
        ('v_cubeid_f32 v256, v0, v4, v8', 'destination'),
        ('v_cubeid_f32 v12, v0, v4, v8 mul:2', 'with mul:2'),
        ('v_cubeid_f32 v12, v0, v4, v8 op_sel:[1,0,0]', 'with op_sel'),
    ):
        reason = refused(assembly)
        assert reason and needle in reason, (assembly, reason)
        print('refused: %s (%s)' % (assembly, reason))


def check_binary(llvm_mc):
    assert assemble(llvm_mc, BODY_SOURCE) == BODY_WORDS
    assert assemble(llvm_mc, PRE_SOURCE) == PRE_WORDS
    assert assemble(llvm_mc, POST_SOURCE) == POST_WORDS
    for index in range(0, len(DIRECTIONS), 4):
        directions = DIRECTIONS[index:index + 4]
        assert assemble(llvm_mc, vs_source(direction_words(directions))) == vs_words(direction_words(directions))
    for source, code in SAMPLE_PARTS:
        assert assemble(llvm_mc, source) == code
    for suffix, (address, code, _) in SAMPLE_FORMS.items():
        assert assemble(llvm_mc, SAMPLE_INSTRUCTION.format(suffix=suffix, address=address)) == bytes.fromhex(code)
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for index, direction in enumerate(DIRECTIONS):
                pipe = translate((direction,) * 4)
                vs = compile_hlsl(compiler, kind, pipe.hlsl_vs(),
                                  'cube_vs_%d_%s' % (index, kind), vs_profile, directory)
                ps = compile_hlsl(compiler, kind, pipe.hlsl_ps(),
                                  'cube_ps_%d_%s' % (index, kind), ps_profile, directory)
                expected = expected_text((direction,))
                run = draw(renderer, adapter, vs, ps, expected)
                if run.returncode:
                    raise AssertionError('%s cube direction %s\n%s\n%s' % (kind, direction, run.stdout, run.stderr))
                print('%s cube direction %s: %s' % (kind, direction, expected), flush=True)

            first = DIRECTIONS[0]
            first_pipe = translate((first,) * 4)
            first_vs = compile_hlsl(compiler, kind, first_pipe.hlsl_vs(),
                                    'cube_first_vs_%s' % kind, vs_profile, directory)
            first_expected = expected_text((first,))

            bad_ps = compile_hlsl(compiler, kind, translate((first,) * 4, BAD_BODY).hlsl_ps(),
                                  'cube_bad_%s' % kind, ps_profile, directory)
            run = draw(renderer, adapter, first_vs, bad_ps, first_expected)
            if run.returncode == 0:
                raise AssertionError('%s swapped id/sc did not mismatch\n%s\n%s' % (kind, run.stdout, run.stderr))
            print('%s swapped id/sc: mismatch as expected' % kind, flush=True)

            tie = DIRECTIONS[1]
            tie_pipe = translate((tie,) * 4)
            tie_vs = compile_hlsl(compiler, kind, tie_pipe.hlsl_vs(),
                                  'cube_tie_vs_%s' % kind, vs_profile, directory)
            tie_ps = compile_hlsl(compiler, kind, tie_pipe.hlsl_ps().replace('>=', '>'),
                                  'cube_tie_ps_%s' % kind, ps_profile, directory)
            run = draw(renderer, adapter, tie_vs, tie_ps, expected_text((tie,)))
            if run.returncode == 0:
                raise AssertionError('%s tie-order control did not mismatch\n%s\n%s' % (kind, run.stdout, run.stderr))
            print('%s tie-order control: mismatch as expected' % kind, flush=True)

            doubled_ps = compile_hlsl(compiler, kind, first_pipe.hlsl_ps().replace('* 2.0f', '* 1.0f'),
                                      'cube_double_%s' % kind, ps_profile, directory)
            run = draw(renderer, adapter, first_vs, doubled_ps, first_expected)
            if run.returncode == 0:
                raise AssertionError('%s cubema factor control did not mismatch\n%s\n%s'
                                     % (kind, run.stdout, run.stderr))
            print('%s cubema factor control: mismatch as expected' % kind, flush=True)


def sample_pipeline(stage, suffix):
    position, setup, pixel_out, vertex_out, forward = (code for _, code in SAMPLE_PARTS)
    sample = setup + bytes.fromhex(SAMPLE_FORMS[suffix][1]) + bytes.fromhex('703f8cbf')  # s_waitcnt vmcnt(0)
    if stage == 'vs':
        vs, ps = position + sample + vertex_out, forward + pixel_out
    else:
        vs, ps = position + bytes.fromhex('000081bf'), sample + pixel_out
    root = bytearray(table_root_blob())
    rts0 = 44  # DXBC header, chunk offset and RTS0 chunk header.
    params = struct.unpack_from('<I', root, rts0 + 8)[0]
    constants = struct.unpack_from('<I', root, rts0 + params + 12 + 8)[0]
    struct.pack_into('<I', root, rts0 + constants + 8, 3)  # S, T and face.
    return G.Pipeline(vs, ps, root), vs, ps


def sample_cases():
    for stage, forms in (('ps', ('', '_lz', '_l', '_b')), ('vs', ('', '_lz', '_l'))):
        for suffix in forms:
            pipeline, vs, ps = sample_pipeline(stage, suffix)
            source = pipeline.hlsl_ps() if stage == 'ps' else pipeline.hlsl_vs()
            method = SAMPLE_FORMS[suffix][2] if stage == 'ps' else 'SampleLevel'
            assert f'r0.{method}(q0, x2w_cube_dir(' in source
            manifest = pipeline.manifest()
            assert manifest.startswith('x2w-manifest 4\n')
            assert 'srv 0 texcube table 0 0\n' in manifest
            yield stage, suffix, pipeline, vs, ps


def sample_refusals():
    for stage, suffix, expected in (
        ('ps', '_d', 'image_sample_d on a cube image: face-space gradients are unsupported'),
        ('vs', '_d', 'image_sample_d on a cube image: face-space gradients are unsupported'),
        ('vs', '_b', 'image_sample_b needs implicit derivatives: pixel stage with the entry EXEC mask and wave-uniform control flow'),
    ):
        try:
            sample_pipeline(stage, suffix)
        except G.Unsupported as error:
            assert str(error) == expected, str(error)
        else:
            raise AssertionError(f'{stage} image_sample{suffix} translated')


def swap_x_faces(source):
    positive = 'return float3(1.0f, -tc, -sc);'
    negative = 'return float3(-1.0f, -tc, sc);'
    assert source.count(positive) == source.count(negative) == 1
    return source.replace(positive, '@POSITIVE@').replace(negative, positive).replace('@POSITIVE@', negative)


def check_sample_draw(dxc, renderer, adapter):
    if adapter != 'warp':
        raise SystemExit('the translated cube draw fixture uses WARP')
    print(f'Windows build {sys.getwindowsversion().build}', flush=True)
    sample_refusals()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        faces = directory / 'faces.bin'
        faces.write_bytes(b''.join(struct.pack('<4f', *colour) * 16 for colour in FACE_COLOURS))
        samples = directory / 'samples.bin'
        addresses = []
        expected = []
        for direction in SAMPLE_DIRECTIONS:
            face, sc, tc, ma = cube_reference(*direction)
            addresses.append(struct.pack('<3f', 1.5 + sc / abs(ma), 1.5 + tc / abs(ma), face))
            axis = max(range(3), key=lambda axis: abs(direction[axis]))
            colour = FACE_COLOURS[2 * axis + (direction[axis] < 0)]
            expected.append(bytes(channel * 255 for channel in colour))
        samples.write_bytes(b''.join(addresses))
        output = directory / 'out.bin'

        def readback(prefix, binding):
            run = subprocess.run([renderer, 'draw', prefix, faces, samples, output, binding],
                                 capture_output=True, text=True, timeout=45)
            print(run.stdout + run.stderr, end='', flush=True)
            if run.returncode:
                raise SystemExit(run.returncode)
            if binding == 'cube-array':
                assert 'cube arrays are unsupported; binding null' in run.stderr
            data = output.read_bytes()
            assert len(data) == 4 * len(expected)
            return [data[i:i + 4] for i in range(0, len(data), 4)]

        for stage, suffix, pipeline, vs, ps in sample_cases():
            name = stage + '_sample' + suffix
            prefix = directory / name
            prefix.with_suffix('.a').write_bytes(vs)
            prefix.with_suffix('.b').write_bytes(ps)
            prefix.with_suffix('.txt').write_text(pipeline.manifest(), encoding='ascii')
            sources = {'vs': pipeline.hlsl_vs(), 'ps': pipeline.hlsl_ps()}
            for kind, source in sources.items():
                compile_hlsl(dxc, 'dxc', source, name + '.' + kind, kind + '_6_0', directory)
            result = readback(prefix, 'cube')
            assert result == expected, (name, 'cube', result, expected)
            if stage == 'ps' and not suffix:
                for binding in ('2d', 'short-cube', 'cube-array'):
                    result = readback(prefix, binding)
                    assert result == [bytes(4)] * len(expected), (name, binding, result)
            compile_hlsl(dxc, 'dxc', swap_x_faces(sources[stage]), name + '.' + stage, stage + '_6_0', directory)
            result = readback(prefix, 'cube')
            swapped = [i for i, (actual, wanted) in enumerate(zip(result, expected)) if actual != wanted]
            assert swapped == [0, 1, 7], (name, 'swapped X faces', result, expected)
            assert [result[i] for i in swapped] == [expected[1], expected[0], expected[1]]
            print(name + ': eight exact samples, swapped X faces detected', flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        sample_refusals()
        print(f'{len(list(sample_cases()))} graphics cube forms translated; gradient and vertex bias controls refused')
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw' and args[-1:] == ['--sample']:
        check_sample_draw(*args[:-1])
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
