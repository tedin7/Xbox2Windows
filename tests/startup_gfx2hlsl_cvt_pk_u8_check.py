"""v_cvt_pk_u8_f32 of tools/startup/gfx2hlsl.py (item K of #2023).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel bodies and compare them with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

The RDNA 1.0 and RDNA 2 ISA guides give (VOP3A opcode 350)
  D = (S2 & ~(0xff << 8 * S1[1:0])) | ((flt32_to_uint8(S0) & 0xff) << 8 * S1[1:0])
and leave flt32_to_uint8 undefined: rounding, bounds, NaN and negative results are not stated. The lowering and this
fixture therefore cover only an S0 that is +0.0 or an integer in [1, 255], where truncation, any rounding mode and any
clamp give the same byte (the form tools/compute_store_aot.py already lowers, #1546). Fractional, negative, above 255
and NaN values are not exercised, and a constant S0 outside that set is refused. A register S0 is accepted in ordinary
translation only when it was written by v_cvt_f32_ubyte0..3 with no write since; any other register is refused unless
the Pipeline selects pk_u8_s0_contract, which the fixture draws do (their S0 words come from flat parameters). S1 keeps its bits above [1:0] in every
draw set, and every S2 holds four distinct bytes, so a wrong byte or an overwritten S2 changes the word.
"""
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_gfx2hlsl_lane_alu import G, MASK, Probe, check, check_draw, main, refused, words  # noqa: E402


def f32(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def cvt_pk_u8(s0, s1, s2):
    shift = 8 * (s1 & 3)
    return (s2 & ~(0xff << shift) & MASK) | ((int(struct.unpack('<f', struct.pack('<I', s0))[0]) & 0xff) << shift)


def dropped_src2(s0, s1, s2):
    return cvt_pk_u8(s0, s1, 0)


OPS = {'v_cvt_pk_u8_f32': (3, cvt_pk_u8, False)}
# (S0 bits, S1, S2): the four byte selectors twice, S1 high bits set or clear, distinct bytes in every S2.
DRAW_SETS = [
    (f32(0.0), 0x00000000, 0x11223344), (f32(1.0), 0xfffffff5, 0xa1b2c3d4), (f32(127.0), 0x12345676, 0x01fe7f80),
    (f32(255.0), 0x80000003, 0xdeadbeef), (f32(255.0), 0x00000000, 0x55aa33cc), (f32(2.0), 0x00000001, 0x0f1e2d3c),
    (f32(200.0), 0xabcdef02, 0x89abcdef), (f32(16.0), 0x7ffffffb, 0xc0ffee11),
]
FAMILY = {
    'pipeline': {'pk_u8_s0_contract': True},
    'ops': OPS, 'draw': {'v_cvt_pk_u8_f32': DRAW_SETS},
    # A lowering that loses S2: the reference of the mutation keeps only the inserted byte.
    'mutations': {'v_cvt_pk_u8_f32': (r'\((v\d+) & ~\(0xffu << \(\(v\d+ & 3u\) << 3u\)\)\)', '0u', dropped_src2)},
    'body': {'v_cvt_pk_u8_f32': words('0c005ed500092204' '0d005ed5010b2604' '0e005ed5020d2a04' '0f005ed5030f2e04')},
    'accepted': ['v_cvt_pk_u8_f32 v1, 1.0, v3, v4', 'v_cvt_pk_u8_f32 v1, 0, 2, 0xa1b2c3d4',
                 'v_cvt_pk_u8_f32 v1, 0x437f0000, v3, v4'],
    'refused': ['v_cvt_pk_u8_f32 v1, v2, v3', 'v_cvt_pk_u8_f32 v1, v2, v3, v4, v5',
                'v_cvt_pk_u8_f32 v1, v2, v3, v4 clamp', 'v_cvt_pk_u8_f32 v1, v2, v3, v4 mul:2',
                'v_cvt_pk_u8_f32 v1, -v2, v3, v4', 'v_cvt_pk_u8_f32 v1, |v2|, v3, v4',
                'v_cvt_pk_u8_f32 v1, 0.5, v3, v4', 'v_cvt_pk_u8_f32 v1, -1.0, v3, v4',
                'v_cvt_pk_u8_f32 v1, 0x43800000, v3, v4', 'v_cvt_pk_u8_f32 v1, 0x7fc00000, v3, v4',
                'v_cvt_pk_u8_f32 v1, 0x3fc00000, v3, v4', 'v_cvt_pk_u8_f32 v1, 0x80000000, v3, v4',
                'v_cvt_pk_u8_f32 s1, 1.0, v3, v4', 'v_cvt_pk_u8_f32 v[1:2], v2, v3, v4'],
}


class Pipe:
    """The two Pipeline attributes the op reads: the selected contract and the notes."""

    def __init__(self, contract):
        self.pk_u8_s0_contract, self.notes = contract, set()


def make_probe(contract=False):
    probe = Probe()
    probe.pipe, probe.has_branches = Pipe(contract), False
    return probe


def run(lines, contract=False):
    probe = make_probe(contract)
    for asm in lines:
        probe.one(asm.split(' ', 1)[0], asm)
    return probe


def emitted(asm, contract=True):
    return run([asm], contract).lines[-1]


def refuses(lines, contract=False):
    try:
        run(lines, contract)
    except G.Unsupported as error:
        return str(error)
    return None


def evaluate(line, regs):
    """Run the emitted assignment as Python: v-registers hold uints, F() reads binary32, clamp and uint() as HLSL."""
    expr = line.split(' = ', 1)[1].rstrip(';')
    expr = re.sub(r'\b0x([0-9a-f]+)u\b', r'0x\1', expr)
    expr = re.sub(r'\b(\d+)u\b', r'\1', expr)
    env = {'F': lambda word: struct.unpack('<f', struct.pack('<I', word))[0], 'uint': int,
           'clamp': lambda x, lo, hi: min(max(x, lo), hi), **regs}
    return eval(expr, {'__builtins__': {}}, env) & MASK  # the expression is the translator's own text


def selector_controls():
    # All four byte selectors, high S1 bits, and an S2 of four distinct bytes, through the translated text.
    line = emitted('v_cvt_pk_u8_f32 v12, v0, v4, v8')
    shifted = line.replace('<< 3u', '<< 2u')
    for s0, s1, s2 in DRAW_SETS:
        regs = {'v0': s0, 'v4': s1, 'v8': s2}
        check('translated text reads %08x %08x %08x' % (s0, s1, s2), evaluate(line, regs) == cvt_pk_u8(s0, s1, s2))
    # Negative controls on the text: a wrong byte position and a lost S2 must disagree with the reference.
    check('wrong byte position is detected',
          any(evaluate(shifted, {'v0': a, 'v4': b, 'v8': c}) != cvt_pk_u8(a, b, c) for a, b, c in DRAW_SETS))
    check('overwritten S2 is detected',
          any(dropped_src2(a, b, c) != cvt_pk_u8(a, b, c) for a, b, c in DRAW_SETS))
    check('all four selectors drawn', {s1 & 3 for _, s1, _ in DRAW_SETS} == {0, 1, 2, 3})
    # The reference itself, worked by hand: byte 2 of 0x11223344 replaced by 0xff.
    check('hand-worked word', cvt_pk_u8(f32(255.0), 2, 0x11223344) == 0x11ff3344)
    shifted_high = evaluate(line, {'v0': f32(9.0), 'v4': 0x100, 'v8': 0x01020304})
    check('S1 bits above [1:0] are ignored', shifted_high == evaluate(line, {'v0': f32(9.0), 'v4': 0, 'v8': 0x01020304}))


def run_controls():
    selector_controls()
    check('constant inside the set is lowered', 'F(0x3f800000u)' in emitted('v_cvt_pk_u8_f32 v1, 1.0, v3, v4'))
    check('S1 and S2 constants are lowered', not refused('v_cvt_pk_u8_f32 v1, 1.0, 7, 0xffffffff'))
    contract_controls()


def contract_controls():
    pk = 'v_cvt_pk_u8_f32 v1, v2, v3, v4'
    # An unproven register S0 is refused in ordinary translation, with a message naming the register.
    for asm in (pk, 'v_cvt_pk_u8_f32_e64 v1, v2, 3, s4', 'v_cvt_pk_u8_f32 v1, s2, v3, v4'):
        check('unproven S0 refused without the contract: ' + asm, refuses([asm]) is not None)
    check('refusal message', refuses([pk]) == 'v_cvt_pk_u8_f32 source v2 is not proven to be +0.0 or an integer in 1..255')
    # The same words translate when the contract is selected, and the note records it.
    probe = run([pk, 'v_cvt_pk_u8_f32_e64 v1, v2, 3, s4'], True)
    check('unproven S0 accepted with the contract', len(probe.lines) == 2 and 'clamp(F(v2)' in probe.lines[0])
    check('the contract is recorded in the notes', any('pk_u8_s0_contract' in n for n in probe.pipe.notes))
    check('no note without a contract use', not run(['v_cvt_pk_u8_f32 v1, 1.0, v3, v4'], True).pipe.notes)
    # Proven forms, each of the four bytes, need no contract.
    for n in range(4):
        cvt = 'v_cvt_f32_ubyte%d v2, v5' % n
        check('ubyte%d result is proven' % n, refuses([cvt, pk]) is None)
        check('ubyte%d result by the e64 spelling is proven' % n, refuses(['v_cvt_f32_ubyte%d_e64 v2, v5' % n, pk]) is None)
    # Anything that writes the register afterwards, or a different register, is unproven again.
    for label, lines in (('a float multiply', ['v_cvt_f32_ubyte0 v2, v5', 'v_mul_f32 v2, v2, v6', pk]),
                         ('a computed register', ['v_mul_f32 v2, v5, v6', pk]),
                         ('a move', ['v_cvt_f32_ubyte0 v2, v5', 'v_mov_b32 v2, v7', pk]),
                         ('another register', ['v_cvt_f32_ubyte0 v8, v5', pk]),
                         ('a saturating conversion', ['v_cvt_f32_ubyte0 v2, v5 clamp', pk]),
                         ('a scaled conversion', ['v_cvt_f32_ubyte0 v2, v5 mul:2', pk]),
                         ('an integer conversion', ['v_cvt_f32_u32 v2, v5', pk])):
        check('%s is refused without the contract' % label, refuses(lines) is not None)
        check('%s is accepted with the contract' % label, refuses(lines, True) is None)
    # The proof needs straight-line code and the entry exec mask.
    probe = make_probe()
    probe.has_branches = True
    probe.one('v_cvt_f32_ubyte0', 'v_cvt_f32_ubyte0 v2, v5')
    check('no proof in a program with branches', 2 not in getattr(probe, 'byte_float_v', ()))
    probe = make_probe()
    probe.exec_known = False
    probe.one('v_cvt_f32_ubyte0', 'v_cvt_f32_ubyte0 v2, v5')
    check('no proof under a partial exec mask', 2 not in getattr(probe, 'byte_float_v', ()))


if __name__ == '__main__':
    if len(sys.argv) == 6 and sys.argv[1] == 'draw':
        _, _, dxc, renderer, adapter, fxc = sys.argv
        check_draw(FAMILY, dxc, renderer, adapter, fxc)
    else:
        run_controls()
        main(FAMILY, __doc__)
