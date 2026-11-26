"""The scalar shift-and-add forms s_lshl1..4_add_u32 in tools/startup/gfx2hlsl.py (Run 20 census #16).

usage: startup_gfx2hlsl_s_lshl_add_check.py [descriptor_gpu <dxc> <offscreen_check> warp]

D = (S0 << N) + S1 on 32 bits, and SCC = carry out of the wide sum (S0 << N) + S1 (bits shifted out of S0 count).
Words are built from the SOP2 layout tools/disasm_rdna2.py reads: [31:30] = 0b10, [29:23] opcode, [22:16] sdst,
[15:8] ssrc1, [7:0] ssrc0, plus one literal dword when a source is 255. The emitted lines are run in order on the CPU
by the evaluator of startup_gfx2hlsl_bit_alu_check.py; the reference is plain Python integer arithmetic.
SCC starts at a value neither 0 nor 1, so a form that does not write it fails. Lane-mask forms are refused.
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
from startup_gfx2hlsl_descriptor_scc_gpu import check_gpu, shader  # noqa: E402

MASK = 0xffffffff
OPCODE = {name: op for op, (name, _) in D.SOP2_OPS.items()}
NAMES = {n: 's_lshl%d_add_u32' % n for n in (1, 2, 3, 4)}
SPECIAL = [0, MASK, 0xaaaaaaaa, 0x55555555, 0x80000000, 0x7fffffff, 1, 0xfffffffe, 0xf0000000, 0x0fffffff,
           0x12345678, 0xdeadbeef]
VALUES = SPECIAL + [1 << k for k in range(32)] + [~(1 << k) & MASK for k in range(32)]
PAIRS = [(a, b) for a in SPECIAL for b in VALUES] + [(a, b) for a in VALUES for b in SPECIAL]
SCC_UNSET = 2
VCC_LO, M0, NULL, EXEC_LO, LITERAL = 106, 124, 125, 126, 255
INLINE = {128: 0, 129: 1, 193: MASK}
# (sdst, ssrc0, ssrc1, literal)
FORMS = [(3, 1, 2, None), (1, 1, 2, None), (2, 1, 2, None), (3, 1, 1, None), (3, 128, 2, None),
         (3, 193, 2, None), (3, 1, 129, None), (3, LITERAL, 2, 0xaaaaaaaa), (3, 1, LITERAL, 0x55555555),
         (M0, 1, 2, None)]
MASK_FORMS = [(3, VCC_LO, 2, None), (3, 1, EXEC_LO, None), (VCC_LO, 1, 2, None), (EXEC_LO, 1, 2, None)]

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def sop2(name, sdst, ssrc0, ssrc1, literal=None):
    word = (0b10 << 30) | (OPCODE[name] << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0
    return struct.pack('<I', word) + (b'' if literal is None else struct.pack('<I', literal))


class Probe(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True


def translate(words):
    decoded = G.decode(words)
    assert len(decoded) == 1 and decoded[0][1] == len(words), decoded
    stage = Probe()
    try:
        stage.one(decoded[0][2], decoded[0][3])
    except G.Unsupported as error:
        stage.lines = ['refused: %s' % error]
    return stage.lines


def source_value(code, literal, env):
    if code == LITERAL:
        return literal
    return INLINE[code] if code in INLINE else env['s%d' % code]


def run(lines, env):
    env = dict(env)
    for line in lines:
        m = re.fullmatch(r'(\w+) = (.*);', line)
        assert m, line
        env[m.group(1)] = Expr(m.group(2)).run(env)[0]
    return env


def reference(n, a, b):
    wide = (a << n) + b
    return wide & MASK, int(wide > MASK)


def first_mismatch(n):
    name = NAMES[n]
    for sdst, ssrc0, ssrc1, literal in FORMS:
        lines = translate(sop2(name, sdst, ssrc0, ssrc1, literal))
        if lines[0].startswith('refused: '):
            return name, sdst, ssrc0, ssrc1, lines[0]
        dest = 'm0' if sdst == M0 else 's%d' % sdst
        for a, b in PAIRS:
            env = {'s1': a, 's2': b, 'scc': SCC_UNSET}
            want = reference(n, source_value(ssrc0, literal, env), source_value(ssrc1, literal, env))
            out = run(lines, env)
            if (out.get(dest), out['scc']) != want:
                return name, sdst, ssrc0, ssrc1, hex(a), hex(b), (out.get(dest), out['scc']), want
    return None


# Five as the second operand exercises carry from the sum as well as shifted-out bits.
descriptor_operations = [
    (name, '5', lambda word, n=n: reference(n, word, 5))
    for n, name in NAMES.items()
]


def incoming_scc_reference(word):
    # The fixture seeds SCC opposite to carry. With shift 2 and a one-bit addend,
    # only the shift can set carry.
    return reference(2, word, int(word < 0x40000000))


descriptor_operations.append((NAMES[2], 'src_scc', incoming_scc_reference))
stride_operations = [
    ('stride', (NAMES[2], '9', lambda word: reference(2, word, 9))),
    ('stride_min1', (NAMES[2], '9', lambda word: reference(2, max(word, 1), 9))),
]

if len(sys.argv) == 5 and sys.argv[1] == 'descriptor_gpu':
    check_gpu(*sys.argv[2:], descriptor_operations)
    for field, operation in stride_operations:
        check_gpu(*sys.argv[2:], (operation,), field=field, cases=(0, 7, 0x3fff))
    sys.exit(0)

for kind in ('ps', 'cs'):
    for name, rhs, _ in descriptor_operations:
        shader(kind, name, rhs)
for field, (name, rhs, _) in stride_operations:
    shader('cs', name, rhs, field)

for n in NAMES:
    mismatch = first_mismatch(n)
    check('%s result and carry across %d forms and %d pairs' % (NAMES[n], len(FORMS), len(PAIRS)),
          mismatch is None, str(mismatch))

for n in NAMES:
    refused = [translate(sop2(NAMES[n], *form))[0] for form in MASK_FORMS]
    check('%s refuses lane-mask sources and destinations' % NAMES[n],
          all(r.startswith('refused: ') for r in refused), str(refused))
    refused_null = translate(sop2(NAMES[n], NULL, 129, 129))
    check('%s refuses the discard destination without raising a translator error' % NAMES[n],
          refused_null == ['refused: %s destination null' % NAMES[n]], str(refused_null))
    stage = Probe()
    stage.varying_s = {1}
    stage.scc_uniform = True
    stage.one(NAMES[n], f'{NAMES[n]} s3, s1, s2')
    check('%s clears a stale SCC uniformity proof' % NAMES[n], not stage.scc_uniform)

if failures:
    print('FAILED: %s' % ', '.join(failures))
    sys.exit(1)
print('all ok')
