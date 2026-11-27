"""Packed binary16 min/max translation and D3D12 readback (#2022).

Run without arguments for translation controls, or use
draw <dxc> <offscreen_check> <warp|hardware> to compare both halves with the CPU reference.
"""
import math
import re
import struct
import sys
import tempfile
from pathlib import Path

from startup_gfx2hlsl_bit_alu_check import G, POST_WORDS, PRE_WORDS, vs_words, words_text
from startup_flat_param_check import compile_hlsl, draw, root_blob


OPS = {'min': 'false', 'max': 'true'}
DEFAULT_LO, DEFAULT_HI = (0, 0), (1, 1)
LITERAL = 0x4400c000  # high +4, low -2
TAG = 0x40000000


def lower(asm, sym=None, masks=()):
    stage = G.Stage.__new__(G.Stage)
    stage.lines, stage.sym, stage.masks = [], sym or {}, set(masks)
    stage.used_s, stage.used_v, stage.exec_known = set(), set(), True
    stage.one(asm.split()[0], asm)
    return stage.lines


def controls():
    for op, flag in OPS.items():
        mnemonic = f'v_pk_{op}_f16'
        for operands, words in (
            ('v0, v1', ('v0', 'v1')),
            ('v0, 0', ('v0', '0x00000000u')),
            ('0x4400c000, v1', ('0x4400c000u', 'v1')),
            ('v0, s20', ('v0', 's20')),
        ):
            for low, high, modifiers in (
                (DEFAULT_LO, DEFAULT_HI, ''),
                ((1, 1), (0, 0), ' op_sel:[1,1] op_sel_hi:[0,0]'),
                ((1, 0), (0, 1), ' op_sel:[1,0] op_sel_hi:[0,1]'),
            ):
                expected = []
                for picks in (low, high):
                    args = [f'({w} >> 16) & 0xffffu' if h else f'{w} & 0xffffu' for w, h in zip(words, picks)]
                    expected.append(f'minmax16_bits({", ".join(args)}, {flag})')
                asm = f'{mnemonic} v3, {operands}{modifiers}'
                got = lower(asm)
                assert got == [f'v3 = {expected[0]} | ({expected[1]} << 16);'], (asm, got)
                print(f'{asm}: packed HLSL matches')

        for tail in ('v0, v1 clamp', 'v0, v1 mul:2',
                     'v0, v1 op_sel:[1]', 'v0, v1 op_sel_hi:[0,2]'):
            refuses(f'{mnemonic} v3, {tail}', tail.split()[-1])
        for source in ('1', '-1', '0.0', 'vcc', 'exec', 'm0', 'null', 'src_scc', 'v[0:1]',
                       's[20:20]', 'v256', 's106', '-v0', '0x100000000'):
            refuses(f'{mnemonic} v3, {source}, v1', f'source {source} requires')
        refuses(f'{mnemonic} v256, v0, v1', 'destination v256 is out of range')
        refuses(f'{mnemonic} v3, v0, s20', 'numeric use of s20 holding root', sym={20: ('root',)})
        refuses(f'{mnemonic} v3, v0, s20', 'is a lane mask', masks=(20,))

    for op, tail in (('add', ''), ('mul', ''), ('fma', ', v2')):
        refuses(f'v_pk_{op}_f16 v3, v0, 0x4000{tail}', 'is not a single VGPR')


def refuses(asm, reason, **kwargs):
    try:
        lower(asm, **kwargs)
    except G.Unsupported as error:
        assert reason in str(error), (asm, error)
    else:
        raise AssertionError(f'accepted {asm}')
    print(f'{asm}: refused')


def half_reference(op, a, b):
    x, y = (struct.unpack('<e', struct.pack('<H', bits))[0] for bits in (a, b))
    if math.isnan(x) and math.isnan(y):
        return 0x7e00
    if math.isnan(x):
        return b
    if math.isnan(y):
        return a
    if x == y == 0.0:
        return (a | b) if op == 'min' else (a & b)
    return a if (x <= y if op == 'min' else x >= y) else b


def reference(op, a, b, low, high):
    lo, hi = (half_reference(op, (a >> (16 * i)) & 0xffff, (b >> (16 * j)) & 0xffff)
              for i, j in (low, high))
    return lo | (hi << 16)


# Four output channels per group. Each row holds two source words, optional special source, and low/high picks.
# Manual mutants that must fail the draws: swap op_sel/op_sel_hi, use max for min, and replace the NaN tests
# with float isnan() folded to false (model that fold by replacing both tests with false).
GROUPS = {
    'zeros': (
        (0x80000000, 0x00008000, '', DEFAULT_LO, DEFAULT_HI),
        (0x00008000, 0x80000000, '', DEFAULT_LO, DEFAULT_HI),
        (0x80000000, 0x80000000, '', DEFAULT_LO, DEFAULT_HI),
        (0x00008000, 0x00008000, '', DEFAULT_LO, DEFAULT_HI),
    ),
    'nan_and_range': (
        (0xfe017d01, 0x40003c00, '', DEFAULT_LO, DEFAULT_HI),
        (0x40003c00, 0x7e02fd02, '', DEFAULT_LO, DEFAULT_HI),
        (0xfe017d01, 0x7e02fd02, '', DEFAULT_LO, DEFAULT_HI),
        (0x7c008001, 0xfc000001, '', DEFAULT_LO, DEFAULT_HI),
    ),
    'selectors': (
        (LITERAL, 0x42003c00, 'literal', (1, 0), (0, 1)),
        (0x42003c00, LITERAL, 'sgpr', (0, 1), (1, 0)),
        (0x4200bc00, 0, 'zero', DEFAULT_LO, (1, 0)),
        (0x42003c00, 0x4400c000, '', DEFAULT_LO, DEFAULT_HI),
    ),
}


def pixel_words(op, cases):
    """Encode the source-built fixture using the two-source gfx1013 VOP3P fields."""
    body = bytearray()
    if any(kind == 'sgpr' for _, _, kind, _, _ in cases):
        body += struct.pack('<2I', 0xbe9403ff, LITERAL)  # s_mov_b32 s20, literal
    opcode = 0x11 if op == 'min' else 0x12
    for channel, (_, _, kind, low, high) in enumerate(cases):
        src0 = 256 + channel if kind != 'literal' else 255
        src1 = {'sgpr': 20, 'zero': 128}.get(kind, 260 + channel)
        w0 = 0xcc004000 | (opcode << 16) | (12 + channel) | (low[0] << 11) | (low[1] << 12)
        w1 = src0 | (src1 << 9) | (high[0] << 27) | (high[1] << 28)
        body += struct.pack('<2I', w0, w1)
        if kind == 'literal':
            body += struct.pack('<I', LITERAL)
    return PRE_WORDS + bytes(body) + POST_WORDS


def check_draw(dxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for group, cases in GROUPS.items():
            vs_code = vs_words([(a, b, 0) for a, b, _, _, _ in cases])
            for op in OPS:
                name = f'{group}_{op}'
                pipeline = G.Pipeline(vs_code, pixel_words(op, cases), root_blob())
                vs = compile_hlsl(dxc, 'dxc', pipeline.hlsl_vs(), f'vs_{name}', 'vs_6_0', directory)
                packed_ps = pipeline.hlsl_ps()
                for shift in (0, 16):
                    # Export each half in a normal binary32 word so D3D preserves every result bit.
                    ps_text, count = re.subn(r'F\((v1[2-5])\)',
                                             rf'F(((\1 >> {shift}) & 0xffffu) | 0x{TAG:x}u)', packed_ps)
                    assert count == 4, packed_ps
                    ps = compile_hlsl(dxc, 'dxc', ps_text, f'ps_{name}_{shift}', 'ps_6_0', directory)
                    expected = words_text(TAG | ((reference(op, a, b, low, high) >> shift) & 0xffff)
                                          for a, b, _, low, high in cases)
                    result = draw(renderer, adapter, vs, ps, expected)
                    assert result.returncode == 0, (name, shift, expected, result.stderr)
                    print(f'{name}, shift {shift}: {expected}')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) == 5 and sys.argv[1] == 'draw' and sys.argv[4] in ('warp', 'hardware'):
        check_draw(*sys.argv[2:])
    else:
        sys.exit(__doc__)
