"""Scalar 64-bit shifts and bitfield extracts in tools/startup/gfx2hlsl.py (Part of #1723).

usage: startup_gfx2hlsl_scalar_shift64_bfe_check.py

Covers s_lshl_b64, s_lshr_b64, s_ashr_i64, s_bfe_i32, s_bfe_u64 and s_bfe_i64. Pure Python, no GPU:
  * constant operands fold to exact 32-bit words and SCC = (result != 0), against hand-written values;
  * the emitted HLSL helpers (shl64, shr64u, shr64i, bfe32, bfe64) are translated to Python and run on a
    grid of inputs, then compared with an independent bit-string reference;
  * run-time operands emit one block that reads every source before writing a destination;
  * malformed shapes are refused.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_gfx2hlsl_scalar_m0_check as M  # noqa: E402
import startup_gfx2hlsl_shift64_check as S  # noqa: E402

G = M.G
M32, M64 = 0xffffffff, 0xffffffffffffffff
OPCODES = {'s_lshl_b64': 0x1f, 's_lshr_b64': 0x21, 's_ashr_i64': 0x23,
           's_bfe_i32': 0x28, 's_bfe_u64': 0x29, 's_bfe_i64': 0x2a}
WIDE_RESULT = {'s_lshl_b64', 's_lshr_b64', 's_ashr_i64', 's_bfe_u64', 's_bfe_i64'}
HELPERS = {'s_lshl_b64': ('shl64', 0), 's_lshr_b64': ('shr64u', 0), 's_ashr_i64': ('shr64i', 0),
           's_bfe_i32': ('bfe32', 1), 's_bfe_u64': ('bfe64', 0), 's_bfe_i64': ('bfe64', 1)}
DST, SRC, AMT = 20, 6, 9      # s[20:21] = op(s[6:7], s9)


# ---- an independent reference, written from the instruction definitions on bit strings ------------------
def reference(op, x, amount):
    if op in ('s_lshl_b64', 's_lshr_b64', 's_ashr_i64'):
        n = amount & 63
        bits = format(x, '064b')
        if op == 's_lshl_b64':
            return int((bits[n:] + '0' * n)[:64], 2)
        fill = bits[0] if op == 's_ashr_i64' else '0'
        return int((fill * n + bits)[:64], 2)
    size = 32 if op == 's_bfe_i32' else 64
    offset, width = amount & (size - 1), (amount >> 16) & 127
    bits = format(x, '0%db' % size)[::-1]                 # index = bit number
    field = bits[offset:offset + width][:size]            # bits that exist, lowest first
    if op != 's_bfe_u64' and width and field:
        field = field + field[-1] * (size - len(field))   # replicate the top bit of the field
    return int((field + '0' * size)[:size][::-1], 2)


# ---- run the emitted HLSL helper text as Python ------------------------------------------------------------
class U(int):
    def __new__(cls, v):
        return int.__new__(cls, v & M32)

    def __lshift__(self, n):
        assert 0 <= int(n) < 32, 'shift count %d' % int(n)
        return U(int(self) << int(n))

    def __rshift__(self, n):
        assert 0 <= int(n) < 32, 'shift count %d' % int(n)
        return U(int(self) >> int(n))

    __add__ = lambda a, b: U(int(a) + int(b))
    __sub__ = lambda a, b: U(int(a) - int(b))
    __and__ = lambda a, b: U(int(a) & int(b))
    __or__ = lambda a, b: U(int(a) | int(b))
    __invert__ = lambda a: U(~int(a))


class I(int):
    def __rshift__(self, n):
        assert 0 <= int(n) < 32, 'shift count %d' % int(n)
        return I(int(self) >> int(n))


def _asint(v):
    return I(int(v) - (1 << 32) if int(v) >> 31 else int(v))


def _ternary(flat):
    cond, mark, rest = flat.partition('?')
    if not mark:
        return flat
    yes, _, no = rest.partition(':')
    return '(%s if %s else %s)' % (_ternary(yes.strip()), cond.strip(), _ternary(no.strip()))


def to_python(expr):
    expr = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'U(\1)', expr)
    expr = re.sub(r'\.x\b', '[0]', re.sub(r'\.y\b', '[1]', expr)).replace('uint2(', 'T(')
    expr = re.sub(r'\buint\(', 'U(', expr).replace('||', ' or ').replace('&&', ' and ')
    group = re.compile(r'\(([^()]*)\)')
    while group.search(expr):
        expr = group.sub(lambda m: '\x00%s\x01' % ','.join(_ternary(p.strip()) for p in split_top(m.group(1))), expr)
    return ','.join(_ternary(p.strip()) for p in split_top(expr)).replace('\x00', '(').replace('\x01', ')')


def split_top(text):
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += (ch == '\x00') - (ch == '\x01')
        if ch == ',' and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    return parts + [text[start:]]


def load_helper(text, name):
    m = re.search(r'uint2? %s\((.*?)\) \{ (.*?) \}(?:\',|$)' % name, text, re.S) or \
        re.search(r'uint2? %s\(([^)]*)\) \{ (.*?)\}\n' % name, text)
    assert m, name + ' helper missing'
    params = [p.split()[-1] for p in m.group(1).split(',')]
    lines = []
    for stmt in [s.strip() for s in m.group(2).split(';') if s.strip()]:
        branch = re.fullmatch(r'if \((.*?)\) (return .*)', stmt)
        if branch:
            lines.append('if %s: %s' % (to_python(branch.group(1)), to_python(branch.group(2))))
            continue
        stmt = re.sub(r'^uint2? ', '', stmt)
        parts = re.split(r',\s*(?=\w+ = )', stmt)
        for part in parts:
            if part.startswith('return '):
                lines.append('return ' + to_python(part[7:]))
            elif '&=' in part:
                var, rhs = part.split('&=')
                lines.append('%s = %s & %s' % (var.strip(), var.strip(), to_python(rhs)))
            else:
                var, rhs = part.split(' = ', 1)
                lines.append('%s = %s' % (var.strip(), to_python(rhs)))
    source = 'def f(%s):\n    ' % ', '.join(params) + '\n    '.join(lines) + '\n'
    env = {'min': min, 'U': U, 'T': lambda a, b: (a, b), 'asint': _asint, 'asuint': lambda v: U(int(v))}
    exec(source, env)  # noqa: S102 - our own generated helper text
    return env['f'], env


def make_helpers(text):
    env = {}
    funcs = {}
    for name in ('shl64', 'shr64u', 'shr64i', 'bfe32', 'bfe64'):
        f, e = load_helper(text, name)
        funcs[name] = f
        env[name] = f
    for f in funcs.values():
        f.__globals__.update(env)
    return funcs


GRID_X = (0, 1, 0x80000000, 0xffffffff, 0x100000000, 0x7fffffffffffffff, 0x8000000000000000,
          0xffffffffffffffff, 0x123456789abcdef0, 0xdeadbeefcafef00d, 0x00000000f0f0f0f0)
GRID_SHIFT = (0, 1, 5, 31, 32, 33, 63, 64, 65, 0xffffffc1)
GRID_BFE = tuple(off | (width << 16) for off in (0, 1, 4, 16, 31, 32, 33, 63) for width in
                 (0, 1, 3, 8, 31, 32, 33, 40, 63, 64, 70, 127))


def helper_grid():
    funcs = make_helpers(S.translate('v_lshlrev_b64', 'shl64')[1])
    checked = 0
    for op, (name, signed) in HELPERS.items():
        amounts = GRID_BFE if 'bfe' in op else GRID_SHIFT
        for x in GRID_X:
            if op == 's_bfe_i32':
                x &= M32
            for amount in amounts:
                lo, hi = x & M32, x >> 32
                if name == 'bfe32':
                    got = int(funcs[name](U(lo), U(amount), U(signed)))
                elif name == 'bfe64':
                    r = funcs[name](U(lo), U(hi), U(amount), U(signed))
                    got = int(r[0]) | int(r[1]) << 32
                else:
                    r = funcs[name](U(lo), U(hi), U(amount))
                    got = int(r[0]) | int(r[1]) << 32
                want = reference(op, x, amount)
                assert got == want, '%s x=%#x amount=%#x: helper %#x, reference %#x' % (op, x, amount, got, want)
                checked += 1
    return checked


# ---- translation -------------------------------------------------------------------------------------------
def stage_for(op, *, dst=DST, src=SRC, amt=AMT, x=None, amount=None):
    stage = M.ProbeStage()
    if x is not None:
        stage.sym[src], stage.sym[src + 1] = ('const', x & M32), ('const', x >> 32)
    if amount is not None:
        stage.sym[amt] = ('const', amount)
    code = M.encode_sop2(OPCODES[op], dst, src, amt)
    _, name, asm, _ = M.D.decode_instruction(code, 0)
    assert name == op, (name, op)
    stage.one(name, asm)
    return stage


def result_words(stage, op):
    count = 2 if op in WIDE_RESULT else 1
    words = [stage.sym[DST + i] for i in range(count)]
    assert all(w[0] == 'const' for w in words), words
    return sum(w[1] << (32 * i) for i, w in enumerate(words))


def const_folding():
    cases = {
        's_lshl_b64': ((1, 63, 0x8000000000000000), (0xffffffff, 32, 0xffffffff00000000),
                       (0x00000000f0f0f0f0, 4, 0x0000000f0f0f0f00), (3, 64, 3)),
        's_lshr_b64': ((0x8000000000000000, 63, 1), (0xffffffff00000000, 32, 0xffffffff),
                       (0x123456789abcdef0, 8, 0x00123456789abcde), (5, 65, 2)),
        's_ashr_i64': ((0x8000000000000000, 63, M64), (0xf000000000000000, 60, M64),
                       (0x7fffffffffffffff, 62, 1), (0x8000000000000000, 32, 0xffffffff80000000)),
        's_bfe_i32': ((0x000000f0, 4 | 4 << 16, 0xffffffff), (0x00000030, 4 | 3 << 16, 0x00000003),
                      (0x80000000, 31 | 1 << 16, 0xffffffff), (0x80000000, 31 | 2 << 16, 0xffffffff), (0x12345678, 0, 0), (0xdeadbeef, 32 << 16, 0xdeadbeef)),
        's_bfe_u64': ((0x123456789abcdef0, 32 | 16 << 16, 0x5678), (M64, 63 | 8 << 16, 1),
                      (0x8000000000000000, 60 | 4 << 16, 8), (M64, 0 | 64 << 16, M64)),
        's_bfe_i64': ((0x00000000000000f0, 4 | 4 << 16, M64), (0x0000000000000030, 4 | 3 << 16, 3),
                      (0x8000000000000000, 63 | 1 << 16, M64), (0x8000000000000000, 63 | 2 << 16, M64),
                      (0x123456789abcdef0, 32 | 16 << 16, 0x5678),
                      (0xff00000000000000, 56 | 8 << 16, M64)),
    }
    for op, rows in cases.items():
        for x, amount, want in rows:
            stage = stage_for(op, x=x if op != 's_bfe_i32' else x & M32, amount=amount)
            got = result_words(stage, op)
            assert got == want, '%s %#x,%#x -> %#x, expected %#x' % (op, x, amount, got, want)
            assert stage.lines[-1] == 'scc = %s;' % ('true' if want else 'false'), (op, stage.lines)
            if op != 's_bfe_i32':
                got_ref = reference(op, x, amount)
                assert got_ref == want, (op, hex(x), hex(amount), hex(got_ref), hex(want))


def runtime_blocks():
    for op in OPCODES:
        wide = op in WIDE_RESULT
        stage = stage_for(op)                      # s6, s7 and s9 are unknown at translation time
        helper = HELPERS[op][0]
        block = [line for line in stage.lines if helper + '(' in line]
        assert len(block) == 1, (op, stage.lines)
        text = block[0]
        assert text.count('s%d = ' % DST) == 1 and ('s%d = ' % (DST + 1) in text) == wide, text
        assert text.index('s%d' % AMT) < text.index('s%d = ' % DST), text      # sources first
        assert 'scc = ' in text and stage.scc_effect == 'emitted', text
        assert all(r not in stage.sym for r in (DST, DST + 1)), stage.sym
        assert ({DST, DST + 1} if wide else {DST}) <= stage.used_s, stage.used_s

    # The destination overlaps the source pair: the whole source is read before s6 is written.
    stage = stage_for('s_lshl_b64', dst=SRC)
    text = [line for line in stage.lines if 'shl64(' in line][0]
    assert text.index('shl64(s6, s7') < text.index('s6 = '), text


def uniformity():
    for op in OPCODES:
        stage = M.ProbeStage()
        stage.varying_s, stage.scc_uniform = set(), True
        code = M.encode_sop2(OPCODES[op], DST, SRC, AMT)
        _, name, asm, _ = M.D.decode_instruction(code, 0)
        stage.sym[AMT] = ('ud', 1)               # a root constant is uniform
        stage.sym[SRC], stage.sym[SRC + 1] = ('ud', 1), ('ud', 1)
        stage.one(name, asm)
        assert DST not in stage.varying_s and stage.scc_uniform, (op, stage.varying_s)
        stage = M.ProbeStage()
        stage.varying_s, stage.scc_uniform = set(), True
        stage.masks = {SRC}                       # a lane mask is not a uniform scalar
        try:
            stage.one(name, asm)
        except G.Unsupported:
            continue
        raise AssertionError('%s on a lane mask should be refused' % op)


def overwritten_mask():
    for op in OPCODES:
        stage = M.ProbeStage()
        stage.masks = {DST} | ({DST + 1} if op in WIDE_RESULT else set())
        stage.sym[SRC], stage.sym[SRC + 1] = ('const', 0x10), ('const', 0)
        stage.sym[AMT] = ('const', 1)
        code = M.encode_sop2(OPCODES[op], DST, SRC, AMT)
        _, name, asm, _ = M.D.decode_instruction(code, 0)
        stage.one(name, asm)
        assert not stage.masks, (op, stage.masks)
        stage = M.ProbeStage()
        stage.masks = {DST} | ({DST + 1} if op in WIDE_RESULT else set())
        code = M.encode_sop2(OPCODES[op], DST, SRC, AMT)
        _, name, asm, _ = M.D.decode_instruction(code, 0)
        stage.one(name, asm)
        assert not stage.masks, (op, 'runtime', stage.masks)


def refusals():
    def refuse(op, dst, src, amt, label, sym=None, masks=()):
        stage = M.ProbeStage()
        stage.masks.update(masks)
        stage.sym.update(sym or {})
        code = M.encode_sop2(OPCODES[op], dst, src, amt)
        _, name, asm, _ = M.D.decode_instruction(code, 0)
        try:
            stage.one(name, asm)
        except G.Unsupported:
            return
        raise AssertionError('%s should be refused: %s' % (op, label))

    for op in OPCODES:
        refuse(op, M.M0, SRC, AMT, 'm0 destination')
        refuse(op, DST, SRC, M.M0, 'm0 shift or field selector')
        refuse(op, 106, SRC, AMT, 'vcc destination')
    for op in WIDE_RESULT:
        refuse(op, DST + 1, SRC, AMT, 'odd destination pair')
        refuse(op, DST, SRC + 1, AMT, 'odd source pair')
    for op in ('s_lshr_b64', 's_bfe_u64'):
        refuse(op, DST, SRC, AMT, 'lane mask in the high source word', masks={SRC + 1})
    refuse('s_bfe_i32', DST, M.M0, AMT, 'm0 source data')
    refuse('s_lshl_b64', DST, SRC, 106, 'vcc as the shift amount')
    refuse('s_lshl_b64', DST, SRC, 126, 'exec as the shift amount')


def main():
    checked = helper_grid()
    const_folding()
    runtime_blocks()
    uniformity()
    refusals()
    overwritten_mask()
    print('startup scalar 64-bit shift and bitfield extract controls passed (%d helper cases)' % checked)
    return 0


if __name__ == '__main__':
    sys.exit(main())
