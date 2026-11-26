"""Constant-sampler LOD limits for image_sample, image_sample_l and image_sample_lz in tools/startup/gfx2hlsl.py (#2177).

  controls [form]                      translation controls, pure Python (the default); without a form, also the
                                       compute manifest sampler lines
  binary <llvm-mc>                     reassemble every fixture source and compare it with the recorded words
  draw-bias <dxc> <offscreen_check>     translated WARP draws with CPU mip selection and negative controls
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given) and draw each form
                                       on its own: the readback must name the mip level the descriptor's limit selects

Semantics. SQ_IMG_SAMP_WORD1 bits 11:0 MIN_LOD and 23:12 MAX_LOD (u4.8, Mesa 24.0.0 src/amd/registers/gfx10-rsrc.json)
clamp the LOD of every sample form, not only of image_sample_b (#2051). A constant sampler of image_sample, _l and _lz now
reaches the static sampler through the same smpl manifest line, with the validation of image_sample_b: a descriptor LOD
bias uses smplb when the primary field is in range and the secondary field is zero. MIN_LOD above MAX_LOD is refused. A mip filter of 0 (word 2 bits 27:26) is NOT refused here
(only the bias form needs it: the bias would count): such a sampler keeps its ordinary smp line, as before. A sampler read
at run time (heap table or the default-sampler shortcut) keeps its ordinary static sampler too, is not validated, and is
logged once per translated module (#2221).

Each form is drawn alone against the 4x4, two-slice, three-mip 2D array of tests/startup_image_3d.cpp
(--startup-image-2d-array-mips) with a point sampler, mip filter "point" (word 2 bits 27:26 = 1), where every
(mip, slice) texel differs. The control of every form is the same program through the ordinary smp line (the
translation before this change), which must read a DIFFERENT mip:

  image_sample_l   explicit LOD 2.0 above MAX_LOD 1:     smpl reads mip 1, smp reads mip 2
  image_sample     MIN_LOD 2 above the implicit LOD 1:   smpl reads mip 2, smp reads mip 1
  image_sample_lz  MIN_LOD 1 above the zero LOD:         smpl reads mip 1, smp reads mip 0

A constant u has no derivatives, so for image_sample the draw replaces only the u initialiser of the translated HLSL by
0.75 + 0.5 * (SV_Position.x - 0.5) (as tests/startup_image_sample_b_check.py does): u changes by 0.5 per pixel, a
footprint of two texels on the 4x4 base level, an implicit LOD of exactly 1. The Sample call is not touched.
"""
import contextlib
import io
import math
import os
import re
import subprocess
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_sampler_heap_check as H  # noqa: E402
from startup_compute_gather_l_check import COMPUTE_TAIL, compute_prefix, cs_pipe, gather_insn  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, assemble, compile_hlsl, root_blob  # noqa: E402
from startup_image_sample_b_check import texel_cell  # noqa: E402

POINT_MIP = 0x4000000  # word 2 bits 27:26 = 1
PROGRAM = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, {w1}
s_mov_b32 s26, {w2}
s_mov_b32 s27, 0
v_mov_b32 v0, 0x3f400000
v_mov_b32 v1, 0x3e800000
v_mov_b32 v2, s9
v_mov_b32 v3, 0x40000000
s_waitcnt lgkmcnt(0)
{instruction}
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
# The sampler words come from a run-time load of the sampler heap at the byte offset in root constant 1 (s9).
HEAP_PROGRAM = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_load_dwordx4 s[24:27], s[10:11], s9
v_mov_b32 v0, 0x3f400000
v_mov_b32 v1, 0x3e800000
v_mov_b32 v2, 0
v_mov_b32 v3, 0x40000000
s_waitcnt lgkmcnt(0)
{instruction}
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
TAIL = 's[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY'
# form: (instruction, manifest/HLSL name, MIN_LOD and MAX_LOD word the draw uses, mip the limits read, mip an
# unlimited sampler reads)
FORMS = {
    'l': ('image_sample_l v[4:7], v[0:3], ' + TAIL, 'image_sample_l', 0x100000, 1, 2),
    'sample': ('image_sample v[4:7], v[0:2], ' + TAIL, 'image_sample', 0x200200, 2, 1),
    'lz': ('image_sample_lz v[4:7], v[0:2], ' + TAIL, 'image_sample_lz', 0x200100, 1, 0),
}
# variant: (word 1 as a function of the form's limit word, word 2)
VARIANTS = {
    'limits': (lambda w1: w1, POINT_MIP),
    'bias': (lambda w1: w1, POINT_MIP | 1),
    'bias sec': (lambda w1: w1, POINT_MIP | 0x4000),
    'range': (lambda w1: 0x200300, POINT_MIP),  # MIN_LOD 3 above MAX_LOD 2
    'mip filter none': (lambda w1: w1, 0),
}
CALLS = {  # the one texture call each form lowers to (kind 2D array, so the slice is the third component)
    'l': 'r0.SampleLevel(q0, float3(F(v0), F(v1), F(v2)), F(v3))',
    'sample': 'r0.Sample(q0, float3(F(v0), F(v1), F(v2)))',
    'lz': 'r0.SampleLevel(q0, float3(F(v0), F(v1), F(v2)), 0)',
}
U_INIT = 'v0 = 0x3f400000u;'
U_DERIVED = 'v0 = U(0.75 + 0.5 * (i.pos.x - 0.5));'


def fixtures():
    out = {}
    for form, (instruction, _, w1, _, _) in FORMS.items():
        for variant, (words1, w2) in VARIANTS.items():
            out['%s %s' % (form, variant)] = PROGRAM.format(instruction=instruction, w1='0x%x' % words1(w1), w2='0x%x' % w2)
        out['%s heap' % form] = HEAP_PROGRAM.format(instruction=instruction)
    return out


# BEGIN WORDS
WORDS = {
    'l limits': (
        '04040cf4000000fa800398beff0399be00001000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f90f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'l bias': (
        '04040cf4000000fa800398beff0399be00001000ff039abe0100000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f90f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'l bias sec': (
        '04040cf4000000fa800398beff0399be00001000ff039abe0040000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f90f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'l range': (
        '04040cf4000000fa800398beff0399be00032000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f90f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'l mip filter none': (
        '04040cf4000000fa800398beff0399be0000100080039abe80039bbeff02007e'
        '0000403fff02027e0000803e0902047ef402067e7fc08cbf280f90f00004c400'
        '703f8cbf0f1800f804050607000081bf'
    ),
    'l heap': (
        '04040cf4000000fa050608f400000012ff02007e0000403fff02027e0000803e'
        '8002047ef402067e7fc08cbf280f90f00004c400703f8cbf0f1800f804050607'
        '000081bf'
    ),
    'sample limits': (
        '04040cf4000000fa800398beff0399be00022000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f80f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'sample bias': (
        '04040cf4000000fa800398beff0399be00022000ff039abe0100000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f80f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'sample bias sec': (
        '04040cf4000000fa800398beff0399be00022000ff039abe0040000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f80f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'sample range': (
        '04040cf4000000fa800398beff0399be00032000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f80f0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'sample mip filter none': (
        '04040cf4000000fa800398beff0399be0002200080039abe80039bbeff02007e'
        '0000403fff02027e0000803e0902047ef402067e7fc08cbf280f80f00004c400'
        '703f8cbf0f1800f804050607000081bf'
    ),
    'sample heap': (
        '04040cf4000000fa050608f400000012ff02007e0000403fff02027e0000803e'
        '8002047ef402067e7fc08cbf280f80f00004c400703f8cbf0f1800f804050607'
        '000081bf'
    ),
    'lz limits': (
        '04040cf4000000fa800398beff0399be00012000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f9cf0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'lz bias': (
        '04040cf4000000fa800398beff0399be00012000ff039abe0100000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f9cf0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'lz bias sec': (
        '04040cf4000000fa800398beff0399be00012000ff039abe0040000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f9cf0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'lz range': (
        '04040cf4000000fa800398beff0399be00032000ff039abe0000000480039bbe'
        'ff02007e0000403fff02027e0000803e0902047ef402067e7fc08cbf280f9cf0'
        '0004c400703f8cbf0f1800f804050607000081bf'
    ),
    'lz mip filter none': (
        '04040cf4000000fa800398beff0399be0001200080039abe80039bbeff02007e'
        '0000403fff02027e0000803e0902047ef402067e7fc08cbf280f9cf00004c400'
        '703f8cbf0f1800f804050607000081bf'
    ),
    'lz heap': (
        '04040cf4000000fa050608f400000012ff02007e0000403fff02027e0000803e'
        '8002047ef402067e7fc08cbf280f9cf00004c400703f8cbf0f1800f804050607'
        '000081bf'
    ),
}
# END WORDS


def stage(name, sampler_heap=None):
    if name.endswith(' heap'):
        return G.Pipeline(H.VS_WORDS, bytes.fromhex(WORDS[name]), H.root_blob(), sampler_heap=sampler_heap)
    return G.Pipeline(VS_WORDS, bytes.fromhex(WORDS[name]), root_blob())


def sampler_line(pipe):
    lines = [line for line in pipe.manifest().splitlines() if line.startswith(('smp ', 'smpl '))]
    assert len(lines) == 1, pipe.manifest()
    return lines[0]


def refused(name, text):
    try:
        stage(name).hlsl_ps()
    except G.Unsupported as error:
        assert str(error) == text, (name, str(error))
    else:
        raise AssertionError('%s was accepted' % name)


def controls(only=None):
    assert WORDS, 'no recorded words'
    for form, (_, op, w1, mip, plain_mip) in FORMS.items():
        if only and form != only:
            continue
        name = '%s limits' % form
        pipe = stage(name)
        ps = pipe.hlsl_ps()
        # The call is the form's own and the sampler carries the limits (manifest line smpl).
        assert CALLS[form] in ps, (name, ps)
        line = 'smpl 0 0x0 0x%x 0x%x 0x0' % (w1, POINT_MIP)
        assert line in pipe.manifest().splitlines() and 'smp ' not in pipe.manifest(), (name, pipe.manifest())
        assert not pipe.lod_unvalidated and not pipe.notes, (name, pipe.lod_unvalidated)
        assert mip != plain_mip
        # Refusals use the validation shared with image_sample_b, under this form's name.
        biased = stage('%s bias' % form)
        assert CALLS[form] in biased.hlsl_ps()
        assert biased.manifest() == pipe.manifest().replace('x2w-manifest 1', 'x2w-manifest 3', 1).replace(
            line, 'smplb 0 0x0 0x%x 0x%x 0x0' % (w1, POINT_MIP | 1))
        refused('%s bias sec' % form, '%s with sampler LOD_BIAS_SEC (word 2 %#x)' % (op, POINT_MIP | 0x4000))
        refused('%s range' % form, '%s with sampler MIN_LOD 0x300 above MAX_LOD 0x200' % op)
        # A mip filter of 0 keeps today's ordinary sampler: no smpl line, nothing refused.
        none = stage('%s mip filter none' % form)
        none.hlsl_ps()
        assert 'smp 0 0x0 0x%x 0x0 0x0' % w1 in none.manifest().splitlines() and 'smpl' not in none.manifest(), none.manifest()
        # A sampler from the heap (table or default-sampler shortcut) is still accepted, on the ordinary path, and
        # the module logs once that its LOD fields are not validated.
        for heap in (True, False):
            heaped = stage('%s heap' % form, sampler_heap=heap)
            heaped.hlsl_ps()
            manifest = heaped.manifest()
            assert 'smpl' not in manifest and heaped.lod_unvalidated == {op}, (form, heap, manifest, heaped.lod_unvalidated)
            assert ('smpheap' in manifest) == heap, (form, heap, manifest)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                G.log_unvalidated_lod(0x1234, heaped)
            text = stderr.getvalue()
            assert text.count('\n') == 1 and op in text and '#2221' in text and '0000000000001234' in text, text
        clean = io.StringIO()
        with contextlib.redirect_stderr(clean):
            G.log_unvalidated_lod(0x1234, pipe)
        assert not clean.getvalue(), clean.getvalue()
        print('%s: smpl line and call; refused: LOD_BIAS_SEC, MIN_LOD > MAX_LOD; mip filter 0 stays smp; heap accepted and '
              'logged once' % op)
    # The log is written by the translation entry point, once per module, and only for a heap sampler.
    for name, logged in (('sample heap', True), ('sample limits', False)):
        vs, root = (H.VS_WORDS, H.root_blob()) if logged else (VS_WORDS, root_blob())
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = G.translate_to(tmp, 0x42, vs, bytes.fromhex(WORDS[name]), root, check=False)
        assert result == 'ok', (name, result)
        assert stderr.getvalue().count('image_sample through a sampler heap') == int(logged), (name, stderr.getvalue())
    print('translate_to logs the unvalidated heap sampler once, and nothing for a constant sampler')
    if only:
        return
    check_compute_manifest()
    check_bias_translation()
    # The three forms differ from each other and from the ordinary line, and the model has distinct mips per form.
    assert len({texel_cell(m, s) for m in range(3) for s in range(2)}) == 6
    assert len({tuple(FORMS[f][3:5]) for f in FORMS}) == 3
    print('draw model: every form reads its limit mip and its control reads another')


def check_compute_manifest():
    """The compute manifest uses the same sampler lines as the graphics one (#2463).

    The programs reuse the helpers of startup_compute_gather_l_check: its compute prefix (sampler words in s24-s27,
    s24 = 0x12), and gather_insn with a 2D sample opcode in place of the gather one. The binary mode of this check
    does not reassemble them with llvm-mc; each case asserts the sample call the translator decodes."""
    for form, opcode in (('lz', 0x27), ('l', 0x24)):
        for variant, w2, tag in (('limits', 0x4500000, 'smpl'), ('bias', 0x4503f80, 'smplb'),
                                 ('mip filter none', 0x500000, 'smp')):
            pipe = cs_pipe(compute_prefix(w2=w2) + gather_insn(0xf, 1, opcode) + COMPUTE_TAIL)
            assert '.SampleLevel(q0,' in pipe.hlsl_cs(), (form, variant)
            manifest = pipe.manifest().splitlines()
            assert manifest[0] == 'x2w-manifest %d' % (3 if tag == 'smplb' else 2), manifest
            lines = [line for line in manifest if line.startswith('smp')]
            assert lines == ['%s 0 0x12 0xfff000 0x%x 0x0' % (tag, w2)], (form, variant, lines)
    pipe = cs_pipe(compute_prefix() + gather_insn(1, 1, 0x2f) + COMPUTE_TAIL)
    assert '.SampleCmpLevelZero(q0,' in pipe.hlsl_cs()
    lines = [line for line in pipe.manifest().splitlines() if line.startswith('smp')]
    assert lines == ['smpc 0 0x12 0xfff000 0x4500000 0x0'], lines
    print('compute manifest: smpl and smplb (lz, l); controls: smp with mip filter 0, smpc for comparison')


def check_binary(llvm_mc):
    for name, source in fixtures().items():
        built = assemble(llvm_mc, source)
        if built != bytes.fromhex(WORDS[name]):
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), WORDS[name]))
    assert set(WORDS) == set(fixtures()), 'recorded and generated fixtures differ'
    print('fixture words match llvm-mc')


def drawable(form, hlsl):
    """The translated stage; for the implicit form, u made to change by 0.5 per pixel (only its initialiser)."""
    if form != 'sample':
        return hlsl
    assert hlsl.count(U_INIT) == 1, hlsl
    return hlsl.replace(U_INIT, U_DERIVED)


def check_draw(dxc, renderer, adapter, fxc=None):
    operation = '--startup-image-2d-array-mips'
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])

    def draw_once(vs, shader, slice_index, expected, sampler):
        run = subprocess.run([renderer, adapter, operation, str(vs), str(shader), str(slice_index), expected, sampler],
                             capture_output=True, text=True)
        sys.stdout.write(run.stdout)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            sys.exit(77)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
        return run

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'fullscreen_lod_%s' % kind, vs_profile, directory)
            for form, (_, op, w1, mip, plain_mip) in FORMS.items():
                pipe = stage('%s limits' % form)
                shader = compile_hlsl(compiler, kind, drawable(form, pipe.hlsl_ps()), 'ps_lod_%s_%s' % (form, kind),
                                      ps_profile, directory)
                line = sampler_line(pipe)
                assert line.startswith('smpl '), line
                for slice_index in (0, 1):
                    expected = texel_cell(mip, slice_index)
                    run = draw_once(vs, shader, slice_index, expected, line)
                    if run.returncode:
                        raise AssertionError('%s %s slice=%d\n%s\n%s' % (kind, op, slice_index, run.stdout, run.stderr))
                    print('%s %s slice=%d reads mip %d: %s' % (kind, op, slice_index, mip, expected), flush=True)
                # Control: the same words through the ordinary smp line (limits not carried) read another mip.
                plain = 'smp ' + line[len('smpl '):]
                for slice_index in (0, 1):
                    run = draw_once(vs, shader, slice_index, texel_cell(mip, slice_index), plain)
                    if run.returncode == 0 or 'actual=%s' % texel_cell(plain_mip, slice_index) not in run.stderr:
                        raise AssertionError('%s %s: smp control did not read mip %d\n%s\n%s'
                                             % (kind, op, plain_mip, run.stdout, run.stderr))
                print('%s %s: plain smp control fails as expected, reads mip %d' % (kind, op, plain_mip), flush=True)


def bias_stage(form, word2):
    # Reuse the source-built sample program with a [0, 2] mip window and the selected sampler word.
    code = bytes.fromhex(WORDS[form + ' limits'])
    original = struct.pack('<I', FORMS[form][2])
    assert code.count(original) == 1 and code.count(struct.pack('<I', POINT_MIP)) == 1
    code = code.replace(original, struct.pack('<I', 0x200000))
    code = code.replace(struct.pack('<I', POINT_MIP), struct.pack('<I', word2))
    return G.Pipeline(VS_WORDS, code, root_blob())


def check_bias_translation():
    for form in ('sample', 'lz'):
        for fixed in (-4096, -128, 0, 256, 4093):
            word2 = POINT_MIP | (fixed & 0x3fff)
            pipe = bias_stage(form, word2)
            assert CALLS[form] in pipe.hlsl_ps()
            manifest = pipe.manifest().splitlines()
            tag, version = ('smplb', 3) if fixed else ('smpl', 1)
            assert manifest[0] == 'x2w-manifest %d' % version, manifest
            assert '%s 0 0x0 0x200000 0x%x 0x0' % (tag, word2) in manifest, manifest
        for word2, reason in ((POINT_MIP | 0x2fff, 'outside the D3D12 range'),
                              (POINT_MIP | 4094, 'outside the D3D12 range'),
                              (POINT_MIP | 0x1000, 'outside the D3D12 range'),
                              (0x100, 'with mipmapping disabled')):
            try:
                bias_stage(form, word2).hlsl_ps()
            except G.Unsupported as error:
                assert reason in str(error), str(error)
            else:
                raise AssertionError((form, hex(word2), reason))
    assert G.lifted_family('image_sample with a sampler LOD bias (word 2 0x4000001)') == 'lod-bias'
    print('graphics sampler LOD bias: endpoints, overflow, no mipmapping, cache retry and legacy manifest')


def check_bias_draw(dxc, renderer):
    check_bias_translation()
    switch = 'X2W_SAMPLER_LOD_BIAS'

    def mip_color(form, bias):
        # A 7/16 UV delta on a 4-wide texture yields a 7/4 texel footprint.
        lod = math.log2(7 / 4) if form == 'sample' else 0
        mip = max(0, min(2, math.floor(lod + bias + 0.5)))
        return ','.join('%08x' % bits for bits in struct.unpack('<4I', struct.pack('<4f', 0.25 * (mip + 1), 0, 0, 1)))

    def draw(vs, ps, line, expected, enabled, wrong=None):
        env = os.environ.copy()
        env.pop(switch, None)
        if enabled:
            env[switch] = '1'
        run = subprocess.run([renderer, 'warp', '--startup-sampler-lod-bias', str(vs), str(ps), '0', expected, line],
                             capture_output=True, text=True, env=env)
        print(run.stdout, end='')
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
        if wrong is None:
            assert run.returncode == 0, (line, run.stdout, run.stderr)
        else:
            assert run.returncode == 1 and 'actual=' + wrong in run.stderr, (line, run.stdout, run.stderr)

    def manifest_line(bias):
        return 'smplb 0 0x0 0x200000 0x%x 0x0' % (POINT_MIP | (round(bias * 256) & 0x3fff))

    clear = ','.join(['bf800000'] * 4)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'sampler_lod_bias_vs', 'vs_6_0', directory)
        for form in ('sample', 'lz'):
            for bias in (-0.5, 1.0, 0.0):
                pipe = bias_stage(form, POINT_MIP | (round(bias * 256) & 0x3fff))
                hlsl = pipe.hlsl_ps()
                line = next(line for line in pipe.manifest().splitlines() if line.startswith(('smpl ', 'smplb ')))
                if form == 'sample':
                    assert hlsl.count(U_INIT) == 1, hlsl
                    hlsl = hlsl.replace(U_INIT, 'v0 = U(0.75 + 0.4375 * (i.pos.x - 0.5));')
                ps = compile_hlsl(dxc, 'dxc', hlsl, 'sampler_lod_bias_%s_%s' % (form, bias), 'ps_6_0', directory)
                expected = mip_color(form, bias)
                draw(vs, ps, line, expected, enabled=True)
                draw(vs, ps, line, clear if bias else expected, enabled=False)
                # For explicit zero, +1 discriminates both mutants without using the +0.5 mip boundary.
                if bias == 1.0 or (form == 'sample' and bias == -0.5):
                    flipped, unbiased = mip_color(form, -bias), mip_color(form, 0)
                    assert expected != flipped and expected != unbiased
                    draw(vs, ps, manifest_line(-bias), expected, enabled=True, wrong=flipped)
                    dropped = line.replace('smplb ', 'smpl ', 1)
                    draw(vs, ps, dropped, expected, enabled=True, wrong=unbiased)
                print('%s bias=%g CPU reference=%s' % (form, bias, expected), flush=True)
            # +16 is invalid. A 13-bit decoder would read it as the legal value -16 and submit a draw.
            draw(vs, ps, manifest_line(16), clear, enabled=True)


def main():
    args = sys.argv[1:]
    if not args or args[0] == 'controls':
        controls(args[1] if len(args) > 1 else None)
    elif args[0] == 'binary' and len(args) == 2:
        check_binary(args[1])
    elif args[0] == 'draw-bias' and len(args) == 3:
        check_bias_draw(*args[1:])
    elif args[0] == 'draw' and len(args) in (4, 5):
        controls()
        check_draw(*args[1:])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
