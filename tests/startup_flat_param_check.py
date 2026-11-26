"""Flat parameters in tools/startup/gfx2hlsl.py keep the exported words (v_interp_mov_f32, #1259).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile with FXC vs_5_1/ps_5_1 (the --pending path) when
                                       given and with DXC vs_6_0/ps_6_0, draw one pixel and compare the words

The fixture is a source-built gfx1013 vertex and pixel pair. The vertex stage places a full-screen
triangle from the vertex index (v0) and exports the constant words
1, 2, 0x7fc00001 and 0x80000001 to param0. The pixel stage reads them with v_interp_mov_f32 p0 and exports
them to mrt0. As binary32, 1, 2 and 0x80000001 are denormals and 0x7fc00001 is a NaN with a payload. The
words must reach the render target unchanged.

Before #1259 the flat copy was a float4 written from the float export. FXC folds asfloat() of a constant
and flushed the three denormal patterns to 0. The draw keeps that form as a negative control, and the same
form with the words computed at run time as a second control: that one is exact, so the loss is the
constant fold, not the flat interface.

The controls also replace the first read with each parameter selector in both encodings (#1890): p0 must
give the same pixel stage, and p10, p20 and an invalid selector must be refused, not read as p0.
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
import gfx2hlsl as G  # noqa: E402
import dxc_cache  # noqa: E402
import gpu_case_runner  # noqa: E402

VS_SOURCE = """\
v_mov_b32 v3, 0x40400000
v_mov_b32 v10, 0xc0400000
v_cmp_eq_u32 vcc_lo, 1, v0
v_cndmask_b32 v1, -1.0, v3, vcc_lo
v_cmp_eq_u32 vcc_lo, 2, v0
v_cndmask_b32 v2, 1.0, v10, vcc_lo
v_mov_b32 v4, 1.0
v_mov_b32 v5, 0
v_mov_b32 v6, 1
v_mov_b32 v7, 2
v_mov_b32 v8, 0x7fc00001
v_mov_b32 v9, 0x80000001
exp pos0 v1, v2, v5, v4 done
exp param0 v6, v7, v8, v9
s_endpgm
"""
VS_WORDS = bytes.fromhex(
    'ff02067e00004040' 'ff02147e000040c0' '8100847d' 'f3060202' '8200847d' 'f2140402' 'f202087e' '80020a7e'
    '81020c7e' '82020e7e' 'ff02107e0100c07f' 'ff02127e01000080' 'cf0800f801020504' '0f0200f806070809' '000081bf')

PS_SOURCE = """\
s_mov_b32 m0, s2
v_interp_mov_f32 v0, p0, attr0.x
v_interp_mov_f32 v1, p0, attr0.y
v_interp_mov_f32 v2, p0, attr0.z
v_interp_mov_f32 v3, p0, attr0.w
exp mrt0 v0, v1, v2, v3 done vm
s_endpgm
"""
PS_WORDS = bytes.fromhex('0203fcbe' '020002c8' '020106c8' '02020ac8' '02030ec8' '0f1800f800010203' '000081bf')

# The same vertex stage exporting param0 twice is not needed: an interpolated read (v_interp_p1/p2)
# keeps its float parameter and gets no flat copy. This pixel stage reads attr0.x interpolated only.
LINEAR_PS_SOURCE = """\
s_mov_b32 m0, s2
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
exp mrt0 v2, v2, v2, v2 done vm
s_endpgm
"""
LINEAR_PS_WORDS = bytes.fromhex('0203fcbe' '000008c8' '010009c8' '0f1800f802020202' '000081bf')

# The parameter selector of the first read, in both encodings (#1890). Only p0 is the flat copy; p10 (P1 - P0)
# and p20 (P2 - P0) need per-vertex attribute values, which the FXC path does not have, so they are refused.
# Each line replaces the first read of PS_WORDS; the decoded text must be the source line (LLVM's spelling).
SELECTOR_SOURCE = """\
v_interp_mov_f32_e32 v0, p10, attr0.x
v_interp_mov_f32_e32 v0, p20, attr0.x
v_interp_mov_f32_e32 v0, p0, attr0.x
v_interp_mov_f32_e64 v0, p10, attr0.x
v_interp_mov_f32_e64 v0, p20, attr0.x
v_interp_mov_f32_e64 v0, p0, attr0.x
"""
SELECTOR_WORDS = tuple(bytes.fromhex(w) for w in (
    '000002c8', '010002c8', '020002c8', '000002d600000000', '000002d600020000', '000002d600040000'))
# VINTRP vsrc 3: llvm-objdump prints invalid_param_3 and llvm-mc accepts no such selector.
INVALID_SELECTOR_WORD = bytes.fromhex('030002c8')

WORDS = (0x00000001, 0x00000002, 0x7fc00001, 0x80000001)
EXPECTED = ','.join('%08x' % w for w in WORDS)
FLUSHED = '00000000,00000000,7fc00001,00000000'


def root_blob():
    """A DXBC container with one RTS0 part and no parameters the fixture reads."""
    rts0 = struct.pack('<6I', 2, 0, 24, 0, 0, 0)
    part = b'RTS0' + struct.pack('<I', len(rts0)) + rts0
    return (b'DXBC' + bytes(16) + struct.pack('<III', 1, 32 + 4 + len(part), 1) + struct.pack('<I', 36) + part)


def translate(ps_words=PS_WORDS):
    return G.Pipeline(VS_WORDS, ps_words, root_blob())


def float_flat(vs, ps):
    """The flat copy as it was before #1259: a float4 copied from the float export, read with asuint."""
    vs = vs.replace('nointerpolation uint4 fparam0', 'nointerpolation float4 fparam0')
    vs = vs.replace('    o.fparam0 = uint4(v6, v7, v8, v9);\n', '    o.fparam0 = o.param0;\n')
    ps = ps.replace('nointerpolation uint4 fparam0', 'nointerpolation float4 fparam0')
    ps = re.sub(r'= i\.fparam0\.([xyzw]);', r'= U(i.fparam0.\1);', ps)
    return vs, ps


def run_time(vs):
    """The words computed at run time. (vid >> 31) << 23 is 0 for the three vertices, but FXC cannot prove
    that: with it the word may be a normal float, so FXC cannot fold the conversion either. A plain
    vid >> 31 is not enough; FXC finds that 1 + (vid >> 31) is always a denormal and flushes it."""
    for n, word in zip(range(6, 10), WORDS):
        vs = vs.replace('    v%d = 0x%08xu;\n' % (n, word), '    v%d = 0x%08xu + ((vid >> 31) << 23);\n' % (n, word))
    return vs


def controls():
    pipe = translate()
    vs, ps = pipe.hlsl_vs(), pipe.hlsl_ps()
    # The flat copy is uint4, written from the exported registers before any float conversion.
    for text in (vs, ps):
        assert 'nointerpolation uint4 fparam0 : FPARAM0;' in text and 'float4 fparam0' not in text, text
    assert '    o.param0 = float4(F(v6), F(v7), F(v8), F(v9));\n    o.fparam0 = uint4(v6, v7, v8, v9);\n' in vs, vs
    assert '    o.fparam0 = p.fparam0;' in vs, vs  # the rectangle-list corner takes the first vertex's words
    for comp, reg in zip('xyzw', range(4)):
        assert '    v%d = i.fparam0.%s;' % (reg, comp) in ps, ps
    assert 'U(i.fparam' not in ps, ps
    print('flat: VS o.fparam0 = uint4(v6, v7, v8, v9); PS v0..v3 = i.fparam0.xyzw, no float conversion')

    check_selectors(ps)

    # An attribute read only interpolated gets no flat copy in either stage.
    pipe = translate(LINEAR_PS_WORDS)
    vs, ps = pipe.hlsl_vs(), pipe.hlsl_ps()
    assert 'fparam' not in vs and 'fparam' not in ps, (vs, ps)
    assert 'i.param0.x' in ps, ps
    print('interpolated only: no fparam0 in either stage')

    # The controls really are the old form and the run-time form.
    old_vs, old_ps = float_flat(*(lambda p: (p.hlsl_vs(), p.hlsl_ps()))(translate()))
    assert 'o.fparam0 = o.param0;' in old_vs and 'U(i.fparam0.w)' in old_ps
    assert run_time(old_vs).count('+ ((vid >> 31) << 23)') == 4
    print('controls: float4 flat copy (before #1259) and its run-time form')


def with_first_read(word):
    """PS_WORDS with its first read (v_interp_mov_f32 v0, p0, attr0.x) replaced by `word`."""
    assert PS_WORDS[4:8] == SELECTOR_WORDS[2]
    return PS_WORDS[:4] + word + PS_WORDS[8:]


def check_selectors(ps):
    """p0 is lowered in both encodings; p10, p20 and an invalid selector are refused, never read as p0 (#1890)."""
    for line, word in zip(SELECTOR_SOURCE.splitlines(), SELECTOR_WORDS):
        (_, size, name, asm), = G.decode(word)
        assert (size, asm) == (len(word), line), (word.hex(), size, asm)
        selector = line.split(', ')[1]
        try:
            translated = translate(with_first_read(word)).hlsl_ps()
        except G.Unsupported as error:
            assert selector != 'p0', (line, error)
            assert 'parameter selector %s' % selector in str(error), (line, error)
            print('refused: %s (%s)' % (line, error))
            continue
        assert selector == 'p0', 'translated %s as %s' % (line, re.findall(r'v0 = i\.[^;]*;', translated))
        assert translated == ps, translated
        print('lowered: %s -> v0 = i.fparam0.x;' % line)
    (_, _, name, asm), = G.decode(INVALID_SELECTOR_WORD)
    assert name.endswith('_unsupported'), asm
    try:
        translate(with_first_read(INVALID_SELECTOR_WORD))
    except G.Unsupported as error:
        print('refused: VINTRP selector 3 (%s)' % error)
    else:
        raise AssertionError('VINTRP selector 3 was translated')


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
    for name, source, words in (('vertex', VS_SOURCE, VS_WORDS), ('pixel', PS_SOURCE, PS_WORDS),
                                ('interpolated pixel', LINEAR_PS_SOURCE, LINEAR_PS_WORDS)):
        built = assemble(llvm_mc, source)
        if built != words:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    for line, words in zip(SELECTOR_SOURCE.splitlines(), SELECTOR_WORDS):
        built = assemble(llvm_mc, line + '\n')
        if built != words:
            raise AssertionError('selector fixture %s: assembled %s, recorded %s' % (line, built.hex(), words.hex()))
    print('fixture words match llvm-mc')


def compile_hlsl(compiler, kind, source, out, profile, directory):
    path = directory / ('%s.hlsl' % out)
    path.write_text(source, encoding='ascii')
    if kind == 'fxc':
        args = [compiler, '/nologo', '/T', profile, '/E', 'main', '/Fo', str(directory / ('%s.cso' % out)), str(path)]
    else:
        args = [compiler, '-T', profile, '-E', 'main', '-Fo', str(directory / ('%s.cso' % out)), str(path)]
    # DXC compiles go through the opt-in X2W_DXC_CACHE cache (tools/dxc_cache.py); FXC compiles and everything else are untouched.
    dxc_cache.with_dxc_cache(subprocess).run(args, check=True, capture_output=True, text=True)
    return directory / ('%s.cso' % out)


def draw(renderer, adapter, vs, ps, expected):
    # One long-lived offscreen_check per test (one D3D12 device); the one-shot call stays as the fallback.
    run = gpu_case_runner.run(renderer, adapter, ['--startup-flat-param', vs, ps, expected])
    sys.stdout.write(run.stdout)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(77)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    pipe = translate()
    vs, ps = pipe.hlsl_vs(), pipe.hlsl_ps()
    old_vs, old_ps = float_flat(vs, ps)
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            def build(name, source, profile):
                return compile_hlsl(compiler, kind, source, '%s_%s' % (name, kind), profile, directory)

            # The translated pair: the exported words reach the target exactly.
            run = draw(renderer, adapter, build('vs', vs, vs_profile), build('ps', ps, ps_profile), EXPECTED)
            if run.returncode:
                raise AssertionError('%s translated pair\n%s\n%s' % (kind, run.stdout, run.stderr))
            print('%s translated pair: %s' % (kind, EXPECTED), flush=True)

            # The float4 flat copy with run-time words: exact, so the interface keeps the bits.
            old_ps_cso = build('old_ps', old_ps, ps_profile)
            run = draw(renderer, adapter, build('runtime_vs', run_time(old_vs), vs_profile), old_ps_cso, EXPECTED)
            if run.returncode:
                raise AssertionError('%s float4 flat copy, run-time words\n%s\n%s' % (kind, run.stdout, run.stderr))
            print('%s float4 flat copy, run-time words: %s' % (kind, EXPECTED), flush=True)

            # The float4 flat copy with constant words: FXC folds asfloat() and flushes the denormals.
            run = draw(renderer, adapter, build('old_vs', old_vs, vs_profile), old_ps_cso, EXPECTED)
            if kind == 'fxc':
                if run.returncode == 0 or 'actual=%s' % FLUSHED not in run.stderr:
                    raise AssertionError('fxc float4 flat copy: control did not read %s\n%s\n%s'
                                         % (FLUSHED, run.stdout, run.stderr))
                print('fxc float4 flat copy, constant words: fails as expected, reads %s' % FLUSHED, flush=True)
            else:
                # DXC keeps constant denormals; the control only separates compilers.
                result = 'exact' if run.returncode == 0 else run.stderr.strip()
                print('dxc float4 flat copy, constant words: %s' % result, flush=True)


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
