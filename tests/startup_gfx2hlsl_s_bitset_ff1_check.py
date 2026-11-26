"""Scalar bit set/clear and find-first-one in tools/startup/gfx2hlsl.py (Run 20 census #18, #1723).

usage: startup_gfx2hlsl_s_bitset_ff1_check.py

s_bitset1_b32 sets bit S0[4:0] of the destination, s_bitset0_b32 clears it, and s_ff1_i32_b32 returns the index
of the lowest set bit of S0 (-1 when none). None of them touch SCC. The translator refused all three with
`Unsupported`. The instruction bytes are constructed here from the SOP1 layout the checked-in decoder reads:
  [31:23] = 0b101111101, [22:16] sdst, [15:8] opcode, [7:0] ssrc0
and the emitted HLSL is evaluated against an integer reference.
"""
import os
import re
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


S_FF1_I32_B32 = 0x13
S_BITSET0_B32 = 0x1b
S_BITSET1_B32 = 0x1d
EXEC_LO = 126


def encode_sop1(op, sdst, ssrc0):
    return struct.pack('<I', (0b101111101 << 23) | (sdst << 16) | (op << 8) | ssrc0)


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified, self.sampler_heap = {}, set(), set(), False


class Stage(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports, self.masks = {}, set()
        self.pipe = Pipeline()


def translate(code):
    decoded = G.decode(code)
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines)


def refused(code):
    try:
        translate(code)
    except G.Unsupported:
        return True
    return False


def evaluate(emitted, dst, s1, s2):
    """Run the emitted `sN = expr;` statement for s1 = old destination (or source) and s2 = bit/source."""
    match = re.search(r'\bs%d = (.*?);' % dst, emitted)
    expr = match.group(1)
    expr = re.sub(r'\b(0x[0-9a-f]+|\d+)u\b', r'\1', expr)
    expr = expr.replace('firstbitlow', 'FBL')
    expr = re.sub(r'\((\S+) == 0 \? (\S+) : (.*)\)$', r'(\2 if \1 == 0 else \3)', expr)
    expr = re.sub(r'\bs1\b', str(s1), expr)
    expr = re.sub(r'\bs2\b', str(s2), expr)
    return eval(expr, {'FBL': lambda x: (x & -x).bit_length() - 1}) & 0xffffffff


# 1. The encoder produces what the decoder reads.
asm, emitted = translate(encode_sop1(S_BITSET1_B32, 1, 2))
check('decodes s_bitset1_b32', asm == ['s_bitset1_b32 s1, s2'], repr(asm))
asm, _ = translate(encode_sop1(S_BITSET0_B32, 1, 2))
check('decodes s_bitset0_b32', asm == ['s_bitset0_b32 s1, s2'], repr(asm))
asm, _ = translate(encode_sop1(S_FF1_I32_B32, 3, 2))
check('decodes s_ff1_i32_b32', asm == ['s_ff1_i32_b32 s3, s2'], repr(asm))

# 2. s_bitset1_b32 sets bit S0[4:0] of the old destination; only the low five bits of S0 count.
_, emitted = translate(encode_sop1(S_BITSET1_B32, 1, 2))
check('bitset1 sets bit 5', evaluate(emitted, 1, 0, 5) == 0x20, emitted)
check('bitset1 keeps other bits', evaluate(emitted, 1, 0x81, 4) == 0x91, emitted)
check('bitset1 uses only S0[4:0]', evaluate(emitted, 1, 0, 0x25) == 0x20, emitted)
check('bitset1 bit 31', evaluate(emitted, 1, 0, 31) == 0x80000000, emitted)
check('bitset1 is idempotent', evaluate(emitted, 1, 0x20, 5) == 0x20, emitted)
check('bitset1 leaves SCC alone', 'scc' not in emitted, emitted)

# 3. s_bitset0_b32 clears it.
_, emitted = translate(encode_sop1(S_BITSET0_B32, 1, 2))
check('bitset0 clears bit 5', evaluate(emitted, 1, 0xffffffff, 5) == 0xffffffdf, emitted)
check('bitset0 of a clear bit is a no-op', evaluate(emitted, 1, 0x1, 5) == 0x1, emitted)
check('bitset0 leaves SCC alone', 'scc' not in emitted, emitted)

# 4. s_ff1_i32_b32: lowest set bit index, -1 for zero.
_, emitted = translate(encode_sop1(S_FF1_I32_B32, 3, 2))
for value, want in ((0, 0xffffffff), (1, 0), (0x80, 7), (0xf0, 4), (0x80000000, 31), (0xffffffff, 0)):
    check('ff1 of 0x%x' % value, evaluate(emitted, 3, 0, value) == want, emitted)
check('ff1 leaves SCC alone', 'scc' not in emitted, emitted)

# 5. Refusals that must not widen: a lane mask is not a plain word, and exec is never a destination here.
check('bitset1 of a lane mask refuses', refused(encode_sop1(S_BITSET1_B32, EXEC_LO, 2)))
check('ff1 into exec_lo refuses', refused(encode_sop1(S_FF1_I32_B32, EXEC_LO, 2)))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
