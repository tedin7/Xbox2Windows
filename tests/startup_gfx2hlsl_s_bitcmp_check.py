"""s_bitcmp1_b32 and s_bitcmp0_b32 in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_s_bitcmp_check.py

RDNA 2 ISA, SOPC: S_BITCMP1_B32 sets SCC = S0.u[S1.u[4:0]] == 1; S_BITCMP0_B32 sets SCC = S0.u[S1.u[4:0]] == 0.
The reference is a plain Python bit test; the translated text is evaluated on the CPU for the sources below,
including a bit index above 31 (only its low five bits count).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
import gfx2hlsl as G  # noqa: E402

failures = []


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for single instructions to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.masks = set()


def run(line, regs):
    stage = Probe()
    stage.one(line.split(' ', 1)[0], line)
    assert len(stage.lines) == 1, stage.lines
    target, _, expression = stage.lines[0].rstrip(';').partition(' = ')
    assert target == 'scc', stage.lines
    expression = re.sub(r'\b(0x[0-9a-f]+|\d+)u\b', r'\1', expression)
    return bool(eval(expression, {}, regs))


WORDS = (0x00000000, 0xffffffff, 0x80000001, 0x00010020, 0x55555555, 0xaaaaaaaa)
INDEXES = (0, 1, 5, 16, 31, 32, 33, 0x21, 0xffffffe5)
for mnemonic, want in (('s_bitcmp1_b32', 1), ('s_bitcmp0_b32', 0)):
    for word in WORDS:
        for index in INDEXES:
            expected = ((word >> (index & 31)) & 1) == want
            for label, line, regs in (
                    ('imm', '%s s44, %d' % (mnemonic, index), {'s44': word}),
                    ('reg', '%s s44, s45' % mnemonic, {'s44': word, 's45': index})):
                try:
                    got = run(line, regs)
                except Exception as exc:  # the translator refused or emitted something unexpected
                    got = '%s: %s' % (type(exc).__name__, exc)
                ok = got == expected
                if not ok:
                    failures.append((line, word, index, got, expected))
                    print('MISMATCH %s word=%#x index=%#x got=%s want=%s' % (line, word, index, got, expected))

for mnemonic in ('s_bitcmp0_b32', 's_bitcmp1_b32'):
    stage = Probe()
    stage.varying_s = {44}
    stage.scc_uniform = True
    stage.one(mnemonic, f'{mnemonic} s44, 1')
    assert not stage.scc_uniform, stage.lines
    for operands in ('exec_lo, 1', 'vcc_lo, 1', 's12, 1', 's44, s12', 's44, vcc_lo'):
        stage = Probe()
        stage.masks.add(12)
        try:
            stage.one(mnemonic, f'{mnemonic} {operands}')
        except G.Unsupported as exc:
            assert 'lane mask' in str(exc), exc
        else:
            raise AssertionError(f'{mnemonic} {operands} accepted a projected lane mask')
        assert not stage.lines, stage.lines

print('s_bitcmp: %s' % ('ok (216 scalar cases, 10 lane-mask refusals)' if not failures
                       else '%d mismatches' % len(failures)))
sys.exit(1 if failures else 0)
