"""Lane-local DPP spellings of VOP2 operations in tools/startup/gfx2hlsl.py, read back from a real D3D12 draw.

  (no arguments)                       translation controls, pure Python (CPU evaluation of the translated text)
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input sets and compare the four result words

The pixel fixture reads two flat parameters (v0..v3, v4..v7), applies one `<op>_dpp ... quad_perm:[0,1,2,3] row_mask:0xf
bank_mask:0xf` to each of the four lanes and exports v12..v15 to mrt0. The vertex stage places the inputs in the
parameters, so the pixel stage cannot fold them. The reference words come from plain float32 / integer arithmetic
written here, not from the lowering. A mutation (the DPP form lowered as the wrong operation) must read a different
word, so a draw that cannot tell the operations apart cannot pass.
"""
import re
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    POST_SOURCE, POST_WORDS, PRE_WORDS, vs_source, vs_words)

MASK = 0xffffffff
CONTROL = 'quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf'


def f32(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


# mnemonic -> reference over two source words, plain arithmetic (floats are normal, finite, exactly representable sums)
OPS = {
    'v_add_f32': lambda a, b: bits(f32(a) + f32(b)),
    'v_sub_f32': lambda a, b: bits(f32(a) - f32(b)),
    'v_or_b32': lambda a, b: a | b,
    'v_add_nc_u32': lambda a, b: (a + b) & MASK,
}
# mnemonic -> (pattern in the translated text, replacement): the operation the DPP form must not be lowered as
MUTATIONS = {
    'v_add_f32': (r'(F\(v\d+\)) \+ (F\(v\d+\))', r'\1 - \2'),
    'v_sub_f32': (r'(F\(v\d+\)) - (F\(v\d+\))', r'\1 + \2'),
    'v_or_b32': (r'(v\d+) \| (v\d+)', r'\1 & \2'),
    'v_add_nc_u32': (r'= (v\d+) \+ (v\d+);', r'= \1 - \2;'),
}
MUTATION_REF = {
    'v_add_f32': lambda a, b: bits(f32(a) - f32(b)),
    'v_sub_f32': lambda a, b: bits(f32(a) + f32(b)),
    'v_or_b32': lambda a, b: a & b,
    'v_add_nc_u32': lambda a, b: (a - b) & MASK,
}
FLOATS = [(bits(a), bits(b)) for a, b in (
    (1.5, 2.25), (-3.0, 0.5), (100.0, -0.125), (0.0, 7.0), (1024.0, 1.0), (-2.5, -2.5), (0.375, 8.0), (65536.0, 3.0),
    (-0.75, 0.25), (12.0, 12.0), (5.5, -1.5), (2.0, 0.0625))]
INTS = [(0x80, 1), (0xff00ff00, 0x00ff00ff), (MASK, 1), (0x12345678, 0x0f0f0f0f), (0x80000000, 0x80000000), (5, 3),
        (0xdeadbeef, 0x01010101), (0, MASK), (0x00ff0000, 0x0000ff00), (0x7fffffff, 1), (0xaaaaaaaa, 0x55555555),
        (0x10, 0x20)]
DRAW = {'v_add_f32': FLOATS, 'v_sub_f32': FLOATS, 'v_or_b32': INTS, 'v_add_nc_u32': INTS}

BODY_WORDS = {
    'v_add_f32': bytes.fromhex(
        'fa08180600e400fffa0a1a0601e400fffa0c1c0602e400fffa0e1e0603e400ff'),
    'v_sub_f32': bytes.fromhex(
        'fa08180800e400fffa0a1a0801e400fffa0c1c0802e400fffa0e1e0803e400ff'),
    'v_or_b32': bytes.fromhex(
        'fa08183800e400fffa0a1a3801e400fffa0c1c3802e400fffa0e1e3803e400ff'),
    'v_add_nc_u32': bytes.fromhex(
        'fa08184a00e400fffa0a1a4a01e400fffa0c1c4a02e400fffa0e1e4a03e400ff'),
}


def body_source(op):
    return ''.join('%s_dpp v%d, v%d, v%d %s\n' % (op, 12 + lane, lane, 4 + lane, CONTROL) for lane in range(4))


def sets_of(chunk):
    return [(a, b, 0) for a, b in chunk]


def pixel_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


def translate(op, mutate=False):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), pixel_words(op), root_blob())
    text = pipe.hlsl_ps()
    if mutate:
        pattern, replacement = MUTATIONS[op]
        mutated = re.sub(pattern, replacement, text)
        assert mutated != text, op
        text = mutated
    return text


def check(label, ok):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        raise AssertionError(label)


def words_text(values):
    return ','.join('%08x' % v for v in values)


def check_binary(llvm_mc):
    """The DPP words decode to the same HLSL statements as the plain spelling of the same operation."""
    for op in OPS:
        plain = PRE_WORDS + assemble(llvm_mc, ''.join(
            '%s v%d, v%d, v%d\n' % (op, 12 + lane, lane, 4 + lane) for lane in range(4))) + POST_WORDS
        check(op + ' fixture words match llvm-mc', assemble(llvm_mc, body_source(op)) == BODY_WORDS[op])
        dpp = pixel_words(op)
        check(op + ' dpp binary decodes like the plain op', dpp != plain and
              G.Pipeline(vs_words([(0, 0, 0)] * 4), dpp, root_blob()).hlsl_ps() ==
              G.Pipeline(vs_words([(0, 0, 0)] * 4), plain, root_blob()).hlsl_ps())
    print('dpp fixture words decode')


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op, pairs in DRAW.items():
                reference = OPS[op]
                good_ps = compile_hlsl(compiler, kind, translate(op), 'ps_%s_%s' % (op, kind), ps_profile,
                                       directory)
                for index in range(0, len(pairs), 4):
                    chunk = pairs[index:index + 4]
                    vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(sets_of(chunk)), pixel_words(op),
                                                                 root_blob()).hlsl_vs(),
                                      'vs_%s_%d_%s' % (op, index, kind), vs_profile, directory)
                    expected = words_text(reference(a, b) for a, b in chunk)
                    run = draw(renderer, adapter, vs, good_ps, expected)
                    if run.returncode:
                        raise AssertionError('%s %s_dpp inputs %s\n%s\n%s' % (kind, op, chunk, run.stdout, run.stderr))
                    print('%s %s_dpp %s: %s' % (kind, op, [tuple('%x' % v for v in x) for x in chunk], expected),
                          flush=True)
                    if index:
                        continue
                    wrong = words_text(MUTATION_REF[op](a, b) for a, b in chunk)
                    check('%s mutation differs from reference' % op, wrong != expected)
                    bad_ps = compile_hlsl(compiler, kind, translate(op, True), 'mutated_%s_%s' % (op, kind),
                                          ps_profile, directory)
                    run = draw(renderer, adapter, vs, bad_ps, expected)
                    if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                        raise AssertionError('%s %s mutation did not read %s\n%s\n%s' % (
                            kind, op, wrong, run.stdout, run.stderr))
                    print('%s %s mutation: fails as expected, reads %s' % (kind, op, wrong), flush=True)


def main():
    if len(sys.argv) == 1:
        raise SystemExit(__doc__)
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
