"""Synthetic gfx1013 cube samples. Run with no arguments, binary <llvm-mc>,
or draw <dxc> <offscreen_check> <warp|hardware> [fxc].
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_image_3d_check import FULLSCREEN_VS, G, VS_WORDS, assemble, compile_hlsl, root_blob

# The three literals are patched with the direction for each face. The cube
# instructions produce sc, tc, face and twice the signed major coordinate.
PS_INPUT = (0x8000, 0x8000)  # v0 is the packed fixed-point pixel position.
PREFIX = """\
s_wqm_b64 exec, exec
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, 0x100000
s_mov_b32 s26, 0x4000000
s_mov_b32 s27, 0
v_mov_b32 v8, 0x3f900000
v_mov_b32 v9, 0x3fa00000
v_mov_b32 v10, 0x3fb00000
v_cubesc_f32 v1, v8, v9, v10
v_cubetc_f32 v2, v8, v9, v10
v_cubeid_f32 v3, v8, v9, v10
v_cubema_f32 v4, v8, v9, v10
v_and_b32_e32 v5, 0xffff, v0
v_cvt_f32_u32_e32 v5, v5
v_fma_f32 v1, v5, 0.25, v1
v_rcp_f32_e64 v4, |v4|
v_fma_f32 v1, v1, v4, 0x3fc00000
v_fma_f32 v2, v2, v4, 0x3fc00000
v_mov_b32 v0, 2.0
v_mov_b32 v4, 1.0
s_waitcnt lgkmcnt(0)
"""
SUFFIX = """\
s_waitcnt vmcnt(0)
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
"""
FORMS = {
    'sample': ('v[1:3]', 'Sample', '', 0),
    'sample_lz': ('v[1:3]', 'SampleLevel', ', 0', 0),
    'sample_l': ('v[1:4]', 'SampleLevel', ', F(v4)', 1),
    'sample_b': ('v[0:3]', 'SampleBias', ', F(v0)', 1),
}
PREFIX_WORDS = bytes.fromhex(
    '7e0afebe04040cf4000000fa800398beff0399be00001000ff039abe0000000480039bbeff02107e0000903fff02127e'
    '0000a03fff02147e0000b03f010045d508132a04020046d508132a04030044d508132a04040047d508132a04ff000a36'
    'ffff0000050d0a7e01004bd505ff05040000803e0401aad50401000001004bd50109fe030000c03f02004bd50209fe03'
    '0000c03ff402007ef202087e7fc08cbf')
SUFFIX_WORDS = bytes.fromhex('703f8cbf0f1800f800010203000081bf')
UNIFORM_JOIN = """\
s_cmp_lg_u32 s9, 0
s_cbranch_scc1 1
v_mov_b32 v5, 0
"""
JOIN_WORDS = bytes.fromhex('098007bf010085bf80020a7e')
OP_WORDS = {
    'sample': bytes.fromhex('180f80f00100c400'),
    'sample_lz': bytes.fromhex('180f9cf00100c400'),
    'sample_l': bytes.fromhex('180f90f00100c400'),
    'sample_b': bytes.fromhex('180f94f00000c400'),
    'sample_d': bytes.fromhex('180f88f00000c400'),
    'sample_c': bytes.fromhex('180fa0f00000c400'),
    'gather4': bytes.fromhex('180100f10100c400'),
    'gather4_lz': bytes.fromhex('18011cf10100c400'),
    'get_lod': bytes.fromhex('180380f10100c400'),
    'sample_a16': bytes.fromhex('180f80f00100c440'),
    'sample_lz_d16': bytes.fromhex('180f9cf00100c480'),
}
# All directions are well inside the selected face. Its projected coordinate
# at pixel (0, 0) is (0.6875, 0.3125), selecting (2, 1) at mip 0 and
# (1, 0) at mip 1. u increases by 1/8 per pixel on a 4x4 face (LOD -1).
# The bias of +2 therefore selects mip 1.
DIRECTIONS = (
    (1.0, 0.375, -0.375), (-1.0, 0.375, 0.375),
    (0.375, 1.0, -0.375), (0.375, -1.0, 0.375),
    (0.375, 0.375, 1.0), (-0.375, 0.375, -1.0),
)
DIM = 'image dimension SQ_RSRC_IMG_CUBE'
X_FACES = (
    '    if (f == 0u) return float3(1.0f, -tc, -sc);\n'
    '    if (f == 1u) return float3(-1.0f, -tc, sc);'
)
SWAPPED_X_FACES = (
    '    if (f == 0u) return float3(-1.0f, -tc, sc);\n'
    '    if (f == 1u) return float3(1.0f, -tc, -sc);'
)


def words(form, direction):
    prefix = PREFIX_WORDS
    for literal, coord in zip((0x3f900000, 0x3fa00000, 0x3fb00000), direction):
        prefix = prefix.replace(struct.pack('<I', literal), struct.pack('<f', coord))
    return prefix + OP_WORDS[form] + SUFFIX_WORDS


def translate_pixel(source):
    return G.Pipeline(VS_WORDS, source, root_blob(), ps_hw_inputs=PS_INPUT)


def refused(source, stage, reason):
    try:
        if stage == 'ps':
            translate_pixel(source)
        else:
            G.Pipeline(source, VS_WORDS, root_blob())
    except G.Unsupported as exc:
        assert reason in str(exc), str(exc)
    else:
        raise AssertionError(f'{stage} accepted {reason}')


def controls():
    uv = 'x2w_cube_dir(F(v1), F(v2), F(v3))'
    for form, (_, method, suffix, _) in FORMS.items():
        pipe = translate_pixel(words(form, DIRECTIONS[0]))
        hlsl = pipe.hlsl_ps()
        assert f'r0.{method}(q0, {uv}{suffix})' in hlsl, [line for line in hlsl.splitlines() if 'r0.' in line]
        assert 'TextureCube<float4> r0 : register(t0);' in hlsl
        assert hlsl.count('float3 x2w_cube_dir(') == 1
        assert 'X2W_PLAIN_SAMPLE' not in hlsl
        assert 'srv 0 texcube table 0 0' in pipe.manifest()
    sample = words('sample', DIRECTIONS[0])
    sample_at = len(PREFIX_WORDS)
    joined = translate_pixel(sample[:sample_at] + JOIN_WORDS + sample[sample_at:])
    assert joined.ps.plain_samples
    joined_hlsl = joined.hlsl_ps()
    assert f'r0.Sample(q0, {uv})' in joined_hlsl and 'X2W_PLAIN_SAMPLE' not in joined_hlsl
    for form, reason in (('sample_d', DIM), ('get_lod', DIM),
                         ('sample_c', 'image_sample_c on SQ_RSRC_IMG_CUBE'),
                         ('gather4', 'image_gather4 requires the guest implicit mip level'),
                         ('gather4_lz', 'image_gather4_lz requires a 2D texture')):
        refused(words(form, DIRECTIONS[0]), 'ps', reason)
    for form in ('sample_a16', 'sample_lz_d16'):
        refused(words(form, DIRECTIONS[0]), 'ps', 'cube sampling with a16 or d16')
    refused(OP_WORDS['sample'] + SUFFIX_WORDS, 'vs', DIM)
    print('Cube pixel calls, declarations, address order and retained refusals pass')


def check_binary(llvm_mc):
    assert assemble(llvm_mc, PREFIX) == PREFIX_WORDS
    assert assemble(llvm_mc, SUFFIX) == SUFFIX_WORDS
    assert assemble(llvm_mc, UNIFORM_JOIN) == JOIN_WORDS
    for form, binary in OP_WORDS.items():
        assert assemble(llvm_mc, operation_source(form)) == binary, form


def operation_source(form):
    if form in FORMS:
        opcode, address, dst, mask, modifier = form, FORMS[form][0], 'v[0:3]', 15, ''
    else:
        opcode, address, dst, mask, modifier = {
            'sample_d': ('sample_d', 'v[0:6]', 'v[0:3]', 15, ''),
            'sample_c': ('sample_c', 'v[0:3]', 'v[0:3]', 15, ''),
            'gather4': ('gather4', 'v[1:3]', 'v[0:3]', 1, ''),
            'gather4_lz': ('gather4_lz', 'v[1:3]', 'v[0:3]', 1, ''),
            'get_lod': ('get_lod', 'v[1:3]', 'v[0:1]', 3, ''),
            'sample_a16': ('sample', 'v[1:2]', 'v[0:3]', 15, ' a16'),
            'sample_lz_d16': ('sample_lz', 'v[1:3]', 'v[0:1]', 15, ' d16'),
        }[form]
    return (f'image_{opcode} {dst}, {address}, s[16:23], s[24:27] '
            f'dmask:0x{mask:x} dim:SQ_RSRC_IMG_CUBE{modifier}\n')


def pixel(face, mip):
    return ','.join(f'{word:08x}' for word in struct.unpack('<4I', struct.pack(
        '<4f', face / 8, 1.0 if mip else 2.0, 0.0 if mip else 1.0, 1.0 + mip)))


def draw_with(compiler, kind, renderer, adapter, vs_profile, ps_profile):
    def readback(shader, expected, mutant_actual=None):
        run = subprocess.run(
            [renderer, adapter, '--startup-image-cube', str(vs), str(shader), "0", expected, sampler],
            capture_output=True, text=True)
        print(run.stdout, end='')
        if run.returncode == 77:
            sys.exit(77)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
        if mutant_actual is None:
            assert run.returncode == 0, (run.stdout, run.stderr)
        else:
            assert run.returncode != 0 and f'actual={mutant_actual}' in run.stderr, (run.stdout, run.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'cube_vs', vs_profile, path)
        for form, (_, _, _, mip) in FORMS.items():
            for face, direction in enumerate(DIRECTIONS):
                pipe = translate_pixel(words(form, direction))
                sampler, = (line for line in pipe.manifest().splitlines() if line.startswith(('smp ', 'smpl ')))
                source = pipe.hlsl_ps()
                shader = compile_hlsl(compiler, kind, source, f'{form}_{face}', ps_profile, path)
                readback(shader, pixel(face, mip))
                if form == 'sample' and face == 0:
                    mutant = source.replace(X_FACES, SWAPPED_X_FACES)
                    assert mutant != source
                    shader = compile_hlsl(compiler, kind, mutant, 'cube_wrong_face', ps_profile, path)
                    readback(shader, pixel(0, 0), mutant_actual=pixel(1, 0))
                    print('Wrong-face mutant failed with the -X texel')


def draw(dxc, renderer, adapter, fxc=None):
    draw_with(dxc, 'dxc', renderer, adapter, 'vs_6_0', 'ps_6_0')
    if fxc:
        draw_with(fxc, 'fxc', renderer, adapter, 'vs_5_1', 'ps_5_1')


if __name__ == '__main__':
    if sys.argv[1:2] == ['binary']:
        check_binary(sys.argv[2])
    else:
        controls()
        if sys.argv[1:2] == ['draw']:
            draw(*sys.argv[2:])
