"""Inline sampler masks in tools/startup/gfx2hlsl.py (#1149).

  (no arguments)                                        pure translation controls
  binary <llvm-mc>                                      reassemble the gfx1013 fixtures
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]      compile and draw with the folded sampler
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_image_3d_check import (
    FULLSCREEN_VS, G, PS_SOURCE as IMAGE_SOURCE, PS_WORDS as IMAGE_WORDS,
    SAMPLE_CALL, assemble, compile_hlsl, texel, translate,
)


# s_bfm_b32 uses the low five bits of the width and offset. Word 0 uses width
# and offset 33 to set only bit 1 (clamp U); word 1 uses width 32 to produce zero.
# Word 2 sets the linear magnification bit. Word 3 uses literal width zero.
SAMPLER_SOURCE = """\
s_movk_i32 s28, 0x21
s_bfm_b32 s24, s28, 33
s_bfm_b32 s25, 32, 5
s_bfm_b32 s26, 1, 20
s_bfm_b32 s27, 0, 31
"""
SAMPLER_WORDS = bytes.fromhex('21001cb0' '1ca11892' 'a0851992' '81941a92' '809f1b92')
CLAMP_SOURCE = IMAGE_SOURCE.replace(
    ''.join('s_mov_b32 s%d, 0\n' % i for i in range(24, 28)), SAMPLER_SOURCE,
).replace('v_mov_b32 v0, 0x3f400000', 'v_mov_b32 v0, 0x3fa00000')  # x = 1.25
CLAMP_WORDS = IMAGE_WORDS.replace(
    bytes.fromhex('800398be' '800399be' '80039abe' '80039bbe'), SAMPLER_WORDS,
).replace(bytes.fromhex('ff02007e0000403f'), bytes.fromhex('ff02007e0000a03f'))
WRAP_SOURCE = CLAMP_SOURCE.replace('s_bfm_b32 s24, s28, 33', 's_bfm_b32 s24, 0, 33')
WRAP_WORDS = CLAMP_WORDS.replace(bytes.fromhex('1ca11892'), bytes.fromhex('80a11892'))

# The clamp sampler with word 0 built in s29 and copied into s24 (s_mov_b32 s24, s29): only the copy fold lets
# the four words be read as constants, so the draw below depends on it.
COPY_CLAMP_SOURCE = CLAMP_SOURCE.replace('s_bfm_b32 s24, s28, 33', 's_bfm_b32 s29, s28, 33\ns_mov_b32 s24, s29')
COPY_CLAMP_WORDS = CLAMP_WORDS.replace(bytes.fromhex('1ca11892'), bytes.fromhex('1ca11d92' '1d0398be'))

# s9 is root constant 1. Overwrite a previously folded word to check that the
# run-time write clears its constant value before the sampler is consumed.
DYNAMIC_SOURCE = CLAMP_SOURCE.replace('v_mov_b32 v0,', 's_bfm_b32 s24, s9, 33\nv_mov_b32 v0,')
DYNAMIC_WORDS = CLAMP_WORDS.replace(bytes.fromhex('ff02007e0000a03f'), bytes.fromhex('09a11892ff02007e0000a03f'))


def check_compare_sampler_constant():
    """A comparison sample needs the sampler's four words as constants it can read.

    image_compare refuses a sampler it cannot resolve to a constant run, because the comparison
    function and the filter have to be taken from those words. The refusal is decided from the
    symbol table, so a bare stage reaches it and a stage with four constant words gets past it.
    """
    def bare_stage(symbols=None):
        stage = G.Stage.__new__(G.Stage)
        stage.sym = dict(symbols or {})
        stage.used_s = set()
        stage.pipe = None
        return stage

    try:
        bare_stage().image_compare(['v5', '[v1, v2, v3]', 's0', 's4'],
                                   ['dim:SQ_RSRC_IMG_2D', 'dmask:0x1'], 'c')
    except G.Unsupported as error:
        assert str(error) == 'image_sample_c with a sampler that is not constant', error
    else:
        raise AssertionError('image_sample_c accepted an unresolved sampler')

    # Four constant words must get past this guard: the bare stage then stops where it needs a
    # pipeline, not here.
    words = {n: ('const', 0) for n in range(4, 8)}
    try:
        bare_stage(words).image_compare(['v5', '[v1, v2, v3]', 's0', 's4'],
                                          ['dim:SQ_RSRC_IMG_2D', 'dmask:0x1'], 'c')
    except G.Unsupported as error:
        assert 'sampler that is not constant' not in str(error), error
    except AttributeError:
        pass

# llc builds a constant sampler by copying one zeroed SGPR into the other words (s_mov_b32 s25, s24).
COPY_WORDS = IMAGE_WORDS.replace(
    bytes.fromhex('800398be' '800399be' '80039abe' '80039bbe'),
    bytes.fromhex('800398be' '180399be' '18039abe' '18039bbe'),
)


def check_copied_constant_sampler():
    """A sampler word copied from a constant SGPR is that constant, not an unknown value."""
    assert COPY_WORDS != IMAGE_WORDS
    expected = [l for l in translate(IMAGE_WORDS).manifest().splitlines() if l.startswith('smp ')]
    got = [l for l in translate(COPY_WORDS).manifest().splitlines() if l.startswith('smp ')]
    assert got == expected and got, (got, expected)
    print('copied constant sampler folds to %s' % got[0])


def copy_symbol(symbols, copy='s_mov_b32 s12, s15'):
    """The symbol s12 holds after `copy` runs on a bare stage that starts with `symbols`."""
    class Pipe:
        sampler_heap = False

    stage = G.Stage.__new__(G.Stage)
    stage.pipe, stage.sym, stage.lines = Pipe(), dict(symbols), []
    stage.used_s, stage.used_v, stage.uniform_v, stage.exec_known = set(), set(), set(), True
    stage.has_branches, stage.masks = False, set()
    stage.one(copy.split()[0], copy)
    return stage.sym.get(12)


def check_copy_only_folds_plain_constants():
    """Only a plain ('const', value) copies as a constant; every other source must not fold."""
    assert copy_symbol({15: ('const', 0)}) == ('const', 0)
    # The default-sampler placeholder is tagged 'unassigned': a copy of it is no run-time constant.
    assert copy_symbol({15: ('const', 0, 'unassigned')}) != ('const', 0)
    # A register that holds a root constant or nothing known is not a constant.
    assert copy_symbol({15: ('ud', 1)}) != ('const', 1)
    assert copy_symbol({}) is None
    # A copy after the source was overwritten takes the new value, never the old one.
    assert copy_symbol({15: ('const', 7)}) == ('const', 7)
    print('copies fold only plain constants')


def check_copied_sampler_after_overwrite():
    """A sampler word copied after its source was rewritten by a run-time value or another constant."""
    prefix = bytes.fromhex('800398be')  # s_mov_b32 s24, 0
    copies = bytes.fromhex('180399be' '18039abe' '18039bbe')
    anchor = prefix + copies
    assert anchor in COPY_WORDS
    runtime = COPY_WORDS.replace(anchor, prefix + bytes.fromhex('090398be') + copies)  # s24 = s9 (root constant)
    try:
        translate(runtime)
    except G.Unsupported as error:
        assert 'holds' in str(error), error
    else:
        raise AssertionError('a sampler copied from a run-time register was accepted')
    # s28 = 0, then s28 = 1, then s27 = s28; s24..s26 stay literal zeros
    one = COPY_WORDS.replace(anchor, bytes.fromhex('800398be' '800399be' '80039abe' '80039cbe' '81039cbe' '1c039bbe'))
    smp = [l for l in translate(one).manifest().splitlines() if l.startswith('smp ')]
    assert smp == ['smp 0 0x0 0x0 0x0 0x1'], smp
    print('copy after overwrite: run-time source refused, new constant wins (%s)' % smp[0])


def controls():
    check_compare_sampler_constant()
    check_copied_constant_sampler()
    check_copy_only_folds_plain_constants()
    check_copied_sampler_after_overwrite()
    fixtures = []
    for name, code, words in (
        ('clamp', CLAMP_WORDS, (0x2, 0, 0x00100000, 0)),
        ('wrap', WRAP_WORDS, (0, 0, 0x00100000, 0)),
        ('copied clamp', COPY_CLAMP_WORDS, (0x2, 0, 0x00100000, 0)),
    ):
        pipe = translate(code)
        manifest = pipe.manifest()
        lines = [line for line in manifest.splitlines() if line.startswith(('smp ', 'smpheap '))]
        expected = 'smp 0 ' + ' '.join(hex(word) for word in words)
        assert lines == [expected] and not pipe.notes, (manifest, pipe.notes)
        ps = pipe.hlsl_ps()
        assert 'SamplerState q0 : register(s0);' in ps and 't = %s;' % SAMPLE_CALL in ps, ps
        fixtures.append((ps, lines[0]))
        print('%s folds to %s' % (name, lines[0]))

    try:
        translate(DYNAMIC_WORDS)
    except G.Unsupported as error:
        assert str(error) == 'descriptor operand s[24:27] holds None', error
        print('root-constant mask width: inline sampler refused (%s)' % error)
    else:
        raise AssertionError('run-time write to s24 was accepted as an inline sampler')
    (clamp_ps, clamp_line), (_, wrap_line), (copy_ps, copy_line) = fixtures
    assert copy_line == clamp_line, (copy_line, clamp_line)
    return clamp_ps, clamp_line, wrap_line, copy_ps


def check_binary(llvm_mc):
    for name, source, recorded in (
        ('clamp', CLAMP_SOURCE, CLAMP_WORDS),
        ('wrap', WRAP_SOURCE, WRAP_WORDS),
        ('root constant', DYNAMIC_SOURCE, DYNAMIC_WORDS),
        ('copied clamp', COPY_CLAMP_SOURCE, COPY_CLAMP_WORDS),
    ):
        built = assemble(llvm_mc, source)
        assert built == recorded, '%s: assembled %s, recorded %s' % (name, built.hex(), recorded.hex())
    print('all four fixture programs match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    ps, clamp, wrap, copy_ps = controls()
    expected_clamp = texel(1, 0, 0)
    expected_wrap = texel(0, 0, 0)
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    if fxc:
        compilers.append(('fxc', fxc, 'vs_5_1', 'ps_5_1'))
    else:
        print('FXC ps_5_1 skipped: compiler unavailable', flush=True)
    out = Path(__file__).resolve().parents[1] / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='inline-sampler-', dir=out) as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'fullscreen', vs_profile, directory)
            shader = compile_hlsl(compiler, kind, ps, 'inline_sampler', ps_profile, directory)
            copy_shader = compile_hlsl(compiler, kind, copy_ps, 'inline_sampler_copy', ps_profile, directory)
            # Hold shader, texture, and reference fixed. Only the manifest sampler changes. The copied-word shader
            # is drawn with its own sampler (must read the clamped texel) and with the wrap sampler (must mismatch).
            for label, shader_path, line, result in (('clamp', shader, clamp, 0), ('wrap control', shader, wrap, 1),
                                                     ('copied-word clamp', copy_shader, clamp, 0),
                                                     ('copied-word wrap control', copy_shader, wrap, 1)):
                command = [renderer, adapter, '--startup-image-3d', str(vs), str(shader_path), '0.25', expected_clamp]
                run = subprocess.run([*command, line], capture_output=True, text=True)
                print(run.stdout, end='', flush=True)
                if run.returncode == 77:
                    print(run.stderr, end='', flush=True)
                    sys.exit(77)
                assert run.returncode == result, (kind, label, run.returncode, run.stderr)
                if 'debug_layer=enabled' in run.stdout:
                    assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
                if result:
                    assert ('expected=%s actual=%s' % (expected_clamp, expected_wrap)) in run.stderr, run.stderr
                    print('%s %s: correctly mismatches, reads %s' % (kind, label, expected_wrap), flush=True)
                else:
                    print('%s %s: %s' % (kind, label, expected_clamp), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
