"""Per-lane bit operations in tools/startup/gfx2hlsl.py: v_bfe_i32, v_bcnt_u32_b32, v_ffbl_b32, v_ffbh_u32,
v_ffbh_i32, v_bfrev_b32 and v_xor3_b32.

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair with FXC ps_5_1 when given and with DXC ps_6_0,
                                       draw one pixel per four input sets and compare the four result words

The pixel fixture reads three flat parameters (v0..v3, v4..v7, v8..v11), applies one operation to each of the four
lanes (VOP1 operations alternate the e32 and e64 encodings) and exports v12..v15 to mrt0. The vertex stage places the
inputs in the parameters, so the pixel stage cannot fold them.

Every result is compared with a reference written from the ISA description with plain integer arithmetic, string
reversal and bit loops, not with the lowering's own expressions. The translated text is evaluated on the CPU by a
small HLSL integer evaluator over a grid that includes every bfe offset and width. Each lowering also has a mutation
(no 31 minus, unsigned shift, subtracted addend, wrong zero result, no reversal) that must read a wrong word. The words
the mutated draw must read come from MUTATION_REF, a plain integer definition of each mutation, and the evaluator on
the mutated text must agree with it. A mutation that keeps the same inputs read as the correct lowering is used so
the pixel stage has the same interface in both draws.
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
from startup_flat_param_check import VS_SOURCE, VS_WORDS, assemble, compile_hlsl, draw, root_blob  # noqa: E402

MASK = 0xffffffff


def ref_bfe_i32(x, off, width):
    off, width = off & 31, width & 31
    if width == 0:
        return 0
    field = ((x - (1 << 32) if x >> 31 else x) >> off) & ((1 << width) - 1)
    return (field - (1 << width) if field >> (width - 1) else field) & MASK


def ref_ffbh_i32(x):
    return next((k for k in range(32) if (x >> (31 - k) & 1) != x >> 31), MASK)


# mnemonic -> (source count, VOP1 so the e32 and e64 spellings both exist, reference)
OPS = {
    'v_bfe_i32': (3, False, ref_bfe_i32),
    'v_bcnt_u32_b32': (2, False, lambda x, y: (bin(x).count('1') + y) & MASK),
    'v_ffbl_b32': (1, True, lambda x: next((i for i in range(32) if x >> i & 1), MASK)),
    'v_ffbh_u32': (1, True, lambda x: next((k for k in range(32) if x >> (31 - k) & 1), MASK)),
    'v_ffbh_i32': (1, True, ref_ffbh_i32),
    'v_bfrev_b32': (1, True, lambda x: int('{:032b}'.format(x)[::-1], 2)),
    'v_xor3_b32': (3, False, lambda x, y, z: x ^ y ^ z),  # LLVM 15 VOP3 v_xor3_b32: D.u = S0.u ^ S1.u ^ S2.u
}
# mnemonic -> (pattern in the translated text, replacement): a wrong lowering of the same operation
MUTATIONS = {
    'v_bfe_i32': (r'asuint\(asint\(', 'asuint(('),                              # the field is not sign-extended
    'v_bcnt_u32_b32': (r'countbits\((v\d+)\) \+ (v\d+)', r'countbits(\1) - \2'),  # the addend is subtracted
    'v_ffbl_b32': (r'firstbitlow\((v\d+)\)', r'(\1 == 0u ? 0u : firstbitlow(\1))'),  # zero reads 0
    'v_ffbh_u32': (r'31u - firstbithigh', 'firstbithigh'),                    # counts from the low end
    'v_ffbh_i32': (r'asint\((v\d+)\) < 0 \? ~v\d+ : v\d+', r'\1'),            # negative values scanned as is
    'v_bfrev_b32': (r'reversebits\((v\d+)\)', r'\1'),                         # not reversed
    'v_xor3_b32': (r'(v\d+) \^ (v\d+) \^ v\d+', r'\1 ^ \2'),                    # the third operand is dropped
}
# mnemonic -> the word the mutated lowering must produce, defined from the mutation and the inputs with plain integer
# arithmetic (not from the evaluator and not from a readback)
MUTATION_REF = {
    'v_bfe_i32': lambda x, off, w: 0 if w & 31 == 0 else (x >> (off & 31)) & ((1 << (w & 31)) - 1),
    'v_bcnt_u32_b32': lambda x, y: (bin(x).count('1') - y) & MASK,
    'v_ffbl_b32': lambda x: 0 if x == 0 else (x & -x).bit_length() - 1,
    'v_ffbh_u32': lambda x: MASK if x == 0 else x.bit_length() - 1,
    'v_ffbh_i32': lambda x: MASK if x in (0, MASK) else 32 - x.bit_length(),
    'v_bfrev_b32': lambda x: x,
    'v_xor3_b32': lambda x, y, z: x ^ y,
}
SPECIAL = [0, 1, 2, 3, 0x7f, 0x80, 0xff, 0x7fffffff, 0x80000000, 0x80000001, 0xfffffffe, MASK, 0x55555555,
           0xaaaaaaaa, 0x12345678, 0xdeadbeef, 0x00ff0000]
BITS = SPECIAL + [1 << k for k in range(32)] + [~(1 << k) & MASK for k in range(32)]
XOR3_WORDS = [0, 1, 0x80000000, MASK, 0x13579bdf, 0x2468ace0, 0xfedcba98, 0x0f0f0f0f, 0xa5c30f96]
GRID = {
    'v_bfe_i32': [(x, off, w) for x in SPECIAL for off in range(32) for w in range(32)]
                 + [(x, off, w) for x in SPECIAL for off, w in ((32, 33), (63, 0xffffffe5), (0x25, 0x1f))],
    'v_bcnt_u32_b32': [(x, y, 0) for x in BITS for y in (0, 1, 32, MASK)],
    'v_xor3_b32': [(x, y, z) for x in XOR3_WORDS for y in XOR3_WORDS for z in XOR3_WORDS],
}
for _name, (_arity, _vop1, _ref) in OPS.items():
    GRID.setdefault(_name, [(x, 0, 0) for x in BITS])
# Four input sets per draw: (src0, src1, src2) per lane. The first set separates every mutation from the reference.
DRAW = {
    'v_bfe_i32': [(0x80, 0, 8), (0x80000000, 31, 1), (0x12345678, 8, 0), (0xdeadbeef, 20, 31),
                  (0xf0, 4, 4), (0x7f, 0, 8), (0x80000000, 16, 16), (MASK, 28, 4),
                  (0x00ff0000, 16, 8), (0x00800000, 16, 8), (0x12345678, 32, 5), (0xdeadbeef, 5, 37)],
    'v_bcnt_u32_b32': [(0, 0, 0), (0xff, 5, 0), (MASK, MASK, 0), (0x80000001, 1, 0),
                       (0x55555555, 0, 0), (0, MASK, 0), (0xf0f0f0f0, 7, 0), (0x00010000, 0x80000000, 0)],
    # Three different sources in every byte (13/24/fe, 57/68/dc, 9b/ac/ba, df/e0/98), so dropping any one source,
    # or OR, AND or an add for the XOR, reads a different word; the other sets check zero, all ones and a repeat.
    'v_xor3_b32': [(0x13579bdf, 0x2468ace0, 0xfedcba98), (0x2468ace0, 0xfedcba98, 0x13579bdf),
                   (0xfedcba98, 0x13579bdf, 0x2468ace0), (0x0f0f0f0f, 0xa5c30f96, 0x80000001),
                   (0, 0, 0), (MASK, MASK, MASK), (0x13579bdf, 0x13579bdf, 0x2468ace0), (0, 0xdeadbeef, MASK)],
}
for _name in ('v_ffbl_b32', 'v_ffbh_u32', 'v_ffbh_i32', 'v_bfrev_b32'):
    DRAW[_name] = [(x, 0, 0) for x in (0, 0x100, 0x80000000, 0xfffffff0, 1, MASK, 0x7fffffff, 0xdeadbeef,
                                         0x00010000, 0x80000001, 0x0000ff00, 0xf0000000)]

PRE_WORDS = bytes.fromhex(
    '0203fcbe020002c8020106c802020ac802030ec8020412c8020516c802061ac8'
    '02071ec8020822c8020926c8020a2ac8020b2ec8')
POST_WORDS = bytes.fromhex('0f1800f80c0d0e0f000081bf')
BODY_WORDS = {
    'v_bfe_i32': bytes.fromhex(
        '0c0049d5000922040d0049d5010b26040e0049d5020d2a040f0049d5030f2e04'),
    'v_bcnt_u32_b32': bytes.fromhex(
        '0c0064d7000902000d0064d7010b02000e0064d7020d02000f0064d7030f0200'),
    'v_ffbl_b32': bytes.fromhex(
        '0075187e0d00bad50101000002751c7e0f00bad503010000'),
    'v_ffbh_u32': bytes.fromhex(
        '0073187e0d00b9d50101000002731c7e0f00b9d503010000'),
    'v_ffbh_i32': bytes.fromhex(
        '0077187e0d00bbd50101000002771c7e0f00bbd503010000'),
    'v_bfrev_b32': bytes.fromhex(
        '0071187e0d00b8d50101000002711c7e0f00b8d503010000'),
    'v_xor3_b32': bytes.fromhex(
        '0c0078d5000922040d0078d5010b26040e0078d5020d2a040f0078d5030f2e04'),
}
PRE_SOURCE = 's_mov_b32 m0, s2\n' + ''.join('v_interp_mov_f32 v%d, p0, attr%d.%s\n' % (4 * p + c, p, 'xyzw'[c])
                                           for p in range(3) for c in range(4))
POST_SOURCE = 'exp mrt0 v12, v13, v14, v15 done vm\ns_endpgm\n'


def body_source(op):
    arity, vop1, _ = OPS[op]
    lines = []
    for lane in range(4):
        sources = ', '.join('v%d' % (4 * s + lane) for s in range(arity))
        lines.append('%s%s v%d, %s\n' % (op, '_e64' if vop1 and lane % 2 else '', 12 + lane, sources))
    return ''.join(lines)


def ps_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


# The vertex stage: the recorded full-screen triangle prefix, then 12 literals in v11..v22 exported as param0..2.
def mov_literal(reg, value):
    """v_mov_b32 reg, value with the operand coding llvm-mc picks: an inline constant when there is one."""
    inline = {0x3f000000: 0xf0, 0xbf000000: 0xf1, 0x3f800000: 0xf2, 0xbf800000: 0xf3, 0x40000000: 0xf4,
              0xc0000000: 0xf5, 0x40800000: 0xf6, 0xc0800000: 0xf7, 0x3e22f983: 0xf8}
    code = inline.get(value, 0x80 + value if value <= 64 else 0xc0 + (1 << 32) - value if value >= MASK - 15 else None)
    if code is not None:
        return struct.pack('<I', 0x7e000200 | reg << 17 | code)
    return struct.pack('<2I', 0x7e0002ff | reg << 17, value)


def vs_source(sets):
    words = [w for s in range(3) for w in (x[s] for x in sets)]
    return (''.join(VS_SOURCE.splitlines(True)[:8]) + ''.join('v_mov_b32 v%d, 0x%x\n' % (11 + i, w)
            for i, w in enumerate(words)) + 'exp pos0 v1, v2, v5, v4 done\n' + ''.join(
            'exp param%d v%d, v%d, v%d, v%d\n' % (p, 11 + 4 * p, 12 + 4 * p, 13 + 4 * p, 14 + 4 * p) for p in range(3))
            + 's_endpgm\n')


def vs_words(sets):
    words = [w for s in range(3) for w in (x[s] for x in sets)]
    out = VS_WORDS[:40] + b''.join(mov_literal(11 + i, w) for i, w in enumerate(words))
    out += bytes.fromhex('cf0800f801020504')
    for p in range(3):
        out += struct.pack('<2I', 0xf800000f | (32 + p) << 4, sum((11 + 4 * p + c) << (8 * c) for c in range(4)))
    return out + bytes.fromhex('000081bf')


# ---- a small evaluator for the integer HLSL the lowering emits (32-bit, uint and int) ------------------------
TOKEN = re.compile(r'\s*(0x[0-9a-fA-F]+u?|\d+u?|[A-Za-z_]\w*|<<|>>|<=|>=|==|!=|\|\||[-+*&|^~!<>?:(),])')
LEVELS = [['||'], ['|'], ['^'], ['&'], ['==', '!='], ['<', '>', '<=', '>='], ['<<', '>>'], ['+', '-'], ['*']]


def signed(v):
    return v - (1 << 32) if v >> 31 else v


class Expr:
    def __init__(self, text):
        self.tokens = TOKEN.findall(text)
        assert ''.join(self.tokens) == re.sub(r'\s+', '', text), text
        self.pos = 0
        self.tree = self.ternary()
        assert self.pos == len(self.tokens), text

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self):
        self.pos += 1
        return self.tokens[self.pos - 1]

    def ternary(self):
        cond = self.binary(0)
        if self.peek() != '?':
            return cond
        self.take()
        yes = self.ternary()
        assert self.take() == ':'
        return ('?', cond, yes, self.ternary())

    def binary(self, level):
        if level == len(LEVELS):
            return self.unary()
        node = self.binary(level + 1)
        while self.peek() in LEVELS[level]:
            op = self.take()
            node = (op, node, self.binary(level + 1))
        return node

    def unary(self):
        token = self.take()
        if token in ('~', '-', '!'):
            return ('u' + token, self.unary())
        if token == '(':
            node = self.ternary()
            assert self.take() == ')'
            return node
        if token[0].isdigit():
            return ('num', int(token.rstrip('u'), 0), not token.endswith('u'))
        if self.peek() != '(':
            return ('var', token)
        self.take()
        args = [self.ternary()]
        while self.peek() == ',':
            self.take()
            args.append(self.ternary())
        assert self.take() == ')'
        return ('call', token, args)

    def run(self, env, node=None):
        node = node or self.tree
        kind = node[0]
        if kind == 'num':
            return node[1] & MASK, node[2]
        if kind == 'var':
            return env[node[1]], False
        if kind == '?':
            return self.run(env, node[2] if self.run(env, node[1])[0] else node[3])
        if kind == 'call':
            return self.call(node[1], [self.run(env, a) for a in node[2]])
        if kind[0] == 'u' and len(node) == 2:
            v, s = self.run(env, node[1])
            return {'u~': ~v, 'u-': -v, 'u!': int(v == 0)}[kind] & MASK, s
        (a, sa), (b, sb) = self.run(env, node[1]), self.run(env, node[2])
        both = sa and sb
        if kind in ('<<', '>>'):
            count = b & 31
            return ((a << count) & MASK if kind == '<<' else (signed(a) >> count if sa else a >> count) & MASK), sa
        if kind in ('==', '!=', '<', '>', '<=', '>='):
            x, y = (signed(a), signed(b)) if both else (a, b)
            return int({'==': x == y, '!=': x != y, '<': x < y, '>': x > y, '<=': x <= y, '>=': x >= y}[kind]), True
        if kind == '||':
            return int(bool(a or b)), True
        return {'|': a | b, '^': a ^ b, '&': a & b, '+': a + b, '-': a - b, '*': a * b}[kind] & MASK, both

    @staticmethod
    def call(name, args):
        (v, s), rest = args[0], args[1:]
        if name in ('asint', 'int'):
            return v, True
        if name in ('asuint', 'uint'):
            return v, False
        if name == 'min':
            assert not s and not rest[0][1]
            return min(v, rest[0][0]), False
        assert not s, name  # the lowering only scans unsigned values
        return {'countbits': bin(v).count('1'), 'reversebits': int('{:032b}'.format(v)[::-1], 2),
                'firstbitlow': (v & -v).bit_length() - 1 if v else MASK,
                'firstbithigh': v.bit_length() - 1 if v else MASK}[name] & MASK, False


def lane_expressions(ps_text):
    found = re.findall(r'^\s*v(1[2-5]) = (.*);$', ps_text, re.M)
    assert [r for r, _ in found] == ['12', '13', '14', '15'], found
    return [Expr(text) for _, text in found]


def lane_results(exprs, sets):
    return [e.run({'v%d' % (4 * s + lane): x[s] for s in range(3)})[0] for lane, (e, x) in enumerate(zip(exprs, sets))]


def translate(op, mutate=False):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words(op), root_blob())
    text = pipe.hlsl_ps()
    if mutate:
        pattern, replacement = MUTATIONS[op]
        mutated = re.sub(pattern, replacement, text)
        assert mutated != text, op
        text = mutated
    return pipe, text


def first_mismatch(op, text):
    """The first input whose lane result differs from the reference, or None."""
    exprs, reference = lane_expressions(text), OPS[op][2]
    for index, inputs in enumerate(GRID[op]):
        lane = index % 4
        got = exprs[lane].run({'v%d' % (4 * s + lane): inputs[s] for s in range(3)})[0]
        if got != reference(*inputs[:OPS[op][0]]):
            return lane, inputs, got
    return None


def check(label, ok):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        raise AssertionError(label)


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for one instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True


def refused(assembly):
    try:
        Probe().one(assembly.split(' ', 1)[0], assembly)
    except G.Unsupported:
        return True
    return False


def controls():
    for op in OPS:
        pipe, text = translate(op)
        check('%s: %d inputs match the reference (e32 and e64 lanes)' % (op, len(GRID[op])),
              first_mismatch(op, text) is None)
        mutated = translate(op, True)[1]
        found = first_mismatch(op, mutated)
        check('%s: mutation reads a wrong word %s' % (op, found), found is not None)
        arity, mutated_exprs = OPS[op][0], lane_expressions(mutated)
        check('%s: the mutated text equals the mutation definition on all %d inputs' % (op, len(GRID[op])),
              all(mutated_exprs[i % 4].run({'v%d' % (4 * s + i % 4): x[s] for s in range(3)})[0]
                  == MUTATION_REF[op](*x[:arity]) for i, x in enumerate(GRID[op])))
        first = DRAW[op][:4]
        wrong = [MUTATION_REF[op](*x[:arity]) for x in first]
        check('%s: draw set 0 mutated words %s differ from the reference and are not all zero' % (op, words_text(wrong)),
              wrong != [OPS[op][2](*x[:arity]) for x in first] and any(wrong))
        check('%s: evaluator on the mutated text reads the same words' % op,
              lane_results(mutated_exprs, first) == wrong)
    # Source-built shader texts name the operands the evaluator reads: v_bfe_i32 field values by hand.
    exprs = lane_expressions(translate('v_bfe_i32')[1])
    check('bfe: offset 0 width 8 of 0x80 is -128, width 0 is 0, offset+width above 32 follows bit 31',
          lane_results(exprs, [(0x80, 0, 8), (0x12345678, 8, 0), (0x80000000, 16, 31), (0x7fffffff, 31, 31)])
          == [0xffffff80, 0, 0xffff8000, 0])
    check('the same forms without a modifier or operand error translate', not any(refused(a) for a in (
        'v_ffbh_i32_e64 v1, v2', 'v_bfe_i32 v1, v2, v3, v4', 'v_bcnt_u32_b32 v1, v2, v3', 'v_bfrev_b32 v1, v2',
        'v_xor3_b32 v1, v2, v3, v4')))
    check('modifiers refused', all(refused(a) for a in (
        'v_ffbh_u32_e64 v1, v2 clamp', 'v_bfe_i32 v1, v2, v3, v4 clamp', 'v_bcnt_u32_b32 v1, v2, v3 clamp',
        'v_ffbl_b32_e64 v1, v2 mul:2', 'v_bfrev_b32_e64 v1, -v2', 'v_ffbh_i32_e64 v1, |v2|',
        'v_bfe_i32 v1, v2, v3, v4 op_sel:[1,0,0,0]', 'v_xor3_b32 v1, v2, v3, v4 clamp',
        'v_xor3_b32 v1, -v2, v3, v4', 'v_xor3_b32 v1, v2, |v3|, v4', 'v_xor3_b32 v1, v2, v3, v4 op_sel:[1,0,0,0]')))
    check('wrong operand count refused', all(refused(a) for a in (
        'v_bfe_i32 v1, v2, v3', 'v_bcnt_u32_b32 v1, v2', 'v_ffbl_b32 v1, v2, v3', 'v_bfrev_b32 v1',
        'v_xor3_b32 v1, v2, v3', 'v_xor3_b32 v1, v2, v3, v4, v5')))


def check_binary(llvm_mc):
    for op in OPS:
        built = assemble(llvm_mc, body_source(op))
        if built != BODY_WORDS[op]:
            raise AssertionError('%s: assembled %s, recorded %s' % (op, built.hex(), BODY_WORDS[op].hex()))
    for name, source, words in (('prologue', PRE_SOURCE, PRE_WORDS), ('export', POST_SOURCE, POST_WORDS)):
        if assemble(llvm_mc, source) != words:
            raise AssertionError('%s fixture differs from llvm-mc' % name)
    for sets in (DRAW[op][i:i + 4] for op in DRAW for i in range(0, len(DRAW[op]), 4)):
        if assemble(llvm_mc, vs_source(sets)) != vs_words(sets):
            raise AssertionError('vertex fixture for %s differs from llvm-mc' % (sets,))
    print('fixture words match llvm-mc')


def words_text(values):
    return ','.join('%08x' % v for v in values)


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op, sets in DRAW.items():
                reference = OPS[op][2]
                good_ps = compile_hlsl(compiler, kind, translate(op)[1], 'ps_%s_%s' % (op, kind), ps_profile, directory)
                for index in range(0, len(sets), 4):
                    chunk = sets[index:index + 4]
                    vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(chunk), ps_words(op), root_blob()).hlsl_vs(),
                                      'vs_%s_%d_%s' % (op, index, kind), vs_profile, directory)
                    expected = words_text(reference(*x[:OPS[op][0]]) for x in chunk)
                    run = draw(renderer, adapter, vs, good_ps, expected)
                    if run.returncode:
                        raise AssertionError('%s %s inputs %s\n%s\n%s' % (kind, op, chunk, run.stdout, run.stderr))
                    print('%s %s %s: %s' % (kind, op, [tuple('%x' % v for v in x[:OPS[op][0]]) for x in chunk], expected),
                          flush=True)
                    if index:
                        continue
                    # The same draw with the mutated lowering must read the mutated words, not the reference.
                    text = translate(op, True)[1]
                    wrong = words_text(MUTATION_REF[op](*x[:OPS[op][0]]) for x in chunk)
                    bad_ps = compile_hlsl(compiler, kind, text, 'mutated_%s_%s' % (op, kind), ps_profile, directory)
                    run = draw(renderer, adapter, vs, bad_ps, expected)
                    if run.returncode == 0 or 'actual=%s' % wrong not in run.stderr:
                        raise AssertionError('%s %s mutation did not read %s\n%s\n%s' % (kind, op, wrong, run.stdout, run.stderr))
                    print('%s %s mutation: fails as expected, reads %s' % (kind, op, wrong), flush=True)


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
