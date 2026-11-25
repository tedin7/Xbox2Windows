"""s_sext_i32_i8, s_sext_i32_i16, s_cmov_b32, s_cmov_b64 and s_cmovk_i32 in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_s_sext_cmov_check.py

Sign-extends: D = sign-extended low 8 / 16 bits of S0, SCC unchanged. Conditional moves: D = SCC ? S : D, SCC unchanged;
s_cmovk_i32 sign-extends its 16-bit immediate. The emitted lines are run in order on the CPU by the evaluator of
startup_gfx2hlsl_bit_alu_check.py; the reference is plain Python integer arithmetic. Lane-mask operands are refused.
"""
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
SOP1 = {name: op for op, (name, _) in D.SOP1_OPS.items()}
SOPK = {name: op for op, (name, _) in D.SOPK_OPS.items()}
VALUES = [0, 1, 0x7f, 0x80, 0xff, 0x100, 0x7fff, 0x8000, 0xffff, 0x10000, 0x80000000, MASK, 0x12345680, 0xdeadbeef,
          0xcafe7fff]
SCC_UNSET = 2
VCC_LO, EXEC_LO = 106, 126

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def sop1(name, sdst, ssrc):
    return struct.pack('<I', (0b101111101 << 23) | (sdst << 16) | (SOP1[name] << 8) | ssrc)


def sopk(name, sdst, imm):
    return struct.pack('<I', (0b1011 << 28) | (SOPK[name] << 23) | (sdst << 16) | imm)


class Probe(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True


def translate(words, masks=()):
    decoded = G.decode(words)
    assert len(decoded) == 1 and decoded[0][1] == len(words), decoded
    stage = Probe()
    stage.masks = set(masks)
    try:
        stage.one(decoded[0][2], decoded[0][3])
    except G.Unsupported as error:
        stage.lines = ['refused: %s' % error]
    return stage.lines


def run(lines, env):
    env = dict(env)
    for line in lines:
        m = re.fullmatch(r'(\w+) = (.*);', line)
        assert m, line
        env[m.group(1)] = Expr(m.group(2)).run(env)[0]
    return env


def sext(value, bits):
    value &= (1 << bits) - 1
    return (value - (1 << bits) if value >> (bits - 1) else value) & MASK


for name, bits in (('s_sext_i32_i8', 8), ('s_sext_i32_i16', 16)):
    bad = None
    lines = translate(sop1(name, 3, 1))
    for a in VALUES:
        out = run(lines, {'s1': a, 's3': 0x55, 'scc': SCC_UNSET})
        want = (sext(a, bits), SCC_UNSET)
        if (out['s3'], out['scc']) != want:
            bad = (hex(a), (out['s3'], out['scc']), want)
            break
    check('%s result and SCC' % name, bad is None, str(bad))
    refused = [translate(sop1(name, d, s))[0] for d, s in ((3, VCC_LO), (3, EXEC_LO), (VCC_LO, 1), (EXEC_LO, 1))]
    check('%s refuses lane-mask operands' % name, all(r.startswith('refused: ') for r in refused), str(refused))

for scc in (0, 1):
    lines = translate(sop1('s_cmov_b32', 3, 1))
    bad = None
    for a in VALUES:
        out = run(lines, {'s1': a, 's3': 0x1234, 'scc': scc})
        want = (a if scc else 0x1234, scc)
        if (out['s3'], out['scc']) != want:
            bad = (hex(a), (out['s3'], out['scc']), want)
    check('s_cmov_b32 with SCC=%d' % scc, bad is None, str(bad))

    lines = translate(sop1('s_cmov_b64', 4, 2))
    bad = None
    for a in VALUES:
        out = run(lines, {'s2': a, 's3': ~a & MASK, 's4': 0x11, 's5': 0x22, 'scc': scc})
        want = ((a, ~a & MASK) if scc else (0x11, 0x22), scc)
        if ((out['s4'], out['s5']), out['scc']) != want:
            bad = (hex(a), ((out['s4'], out['s5']), out['scc']), want)
    check('s_cmov_b64 with SCC=%d' % scc, bad is None, str(bad))

    bad = None
    for imm in (0, 1, 0x7fff, 0x8000, 0xffff, 0x1234):
        lines = translate(sopk('s_cmovk_i32', 3, imm))
        out = run(lines, {'s3': 0x1234, 'scc': scc})
        want = (sext(imm, 16) if scc else 0x1234, scc)
        if (out['s3'], out['scc']) != want:
            bad = (hex(imm), (out['s3'], out['scc']), want)
    check('s_cmovk_i32 with SCC=%d sign-extends the immediate' % scc, bad is None, str(bad))

refused = [translate(sop1(n, d, s))[0] for n in ('s_cmov_b32', 's_cmov_b64')
           for d, s in ((VCC_LO, 2), (EXEC_LO, 2), (4, VCC_LO), (4, EXEC_LO), (125, 2))]
check('s_cmov refuses lane masks and the discard destination', all(r.startswith('refused: ') for r in refused),
      str(refused))
halves = [translate(sop1('s_cmov_b64', d, s), {m})[0] for d, s, m in ((4, 2, 3), (4, 2, 5), (4, 2, 2), (4, 2, 4))]
check('s_cmov_b64 refuses a lane mask held in either half of either pair',
      all(r.startswith('refused: ') for r in halves), str(halves))
check('s_cmov_b64 still moves pairs that hold no lane mask', not translate(sop1('s_cmov_b64', 4, 2), {7})[0].startswith('refused'))
check('s_cmovk_i32 refuses a lane-mask destination', translate(sopk('s_cmovk_i32', VCC_LO, 1))[0].startswith('refused: '))

for instruction in ('s_cmovk_i32 s3, 0x10000', 's_cmovk_i32 s3, -32769',
                    's_cmov_b64 s[104:106], s[2:3]', 's_cmov_b64 s[4:5], s[104:106]',
                    's_cmov_b32 s[4:5], s2'):
    stage = Probe()
    try:
        stage.one(instruction.split()[0], instruction)
    except G.Unsupported:
        check(instruction + ' refused', True)
    else:
        check(instruction + ' refused', False)

if failures:
    print('FAILED: %s' % ', '.join(failures))
    sys.exit(1)
print('all ok')
