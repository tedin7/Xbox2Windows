"""s_mulk_i32 and s_addk_i32 in tools/startup/gfx2hlsl.py (issue #1723, Run 20 census #5).

usage: startup_gfx2hlsl_s_mulk_addk_check.py

The two SOPK arithmetic forms `s_mulk_i32 sN, simm16` and `s_addk_i32 sN, simm16` were refused. Both read and write
SDST and take a 16-bit immediate that is sign-extended; s_addk_i32 also sets SCC to the signed overflow of the add.

The instruction words are constructed here from the layout the checked-in decoder reads (`tools/disasm_rdna2.py`,
SOPK branch): 0xb0000000, opcode in bits 27:23 (0x0f add, 0x10 mul), SDST in bits 22:16, SIMM16 in bits 15:0.
The emitted statements are executed on 32-bit values and compared with the architectural result.
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


SOPK_ADDK, SOPK_MULK = 0x0F, 0x10
MASK = 0xFFFFFFFF


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self):
        self.lines, self.sym, self.masks = [], {}, set()
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()


def encode(op, sdst, simm):
    return struct.pack('<I', 0xB0000000 | (op << 23) | (sdst << 16) | simm)


def translate(code):
    stage = Stage()
    decoded = G.decode(code)
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return decoded[0][3], stage


def run(lines, s4, scc=0):
    """Execute the emitted statements on 32-bit unsigned values (HLSL uint wraps)."""
    env = {'s4': s4, 'scc': bool(scc)}
    for line in lines:
        text = re.sub(r'0x([0-9a-f]{8})u', r'0x\1', line.rstrip(';'))
        text = re.sub(r'\b0u\b', '0', text).replace('&&', ' and ')
        name, _, expr = text.partition(' = ')
        value = eval(expr, {}, env)
        env[name] = value if name == 'scc' else value & MASK
    return env['s4'], env['scc']


def signed(value):
    return value - (1 << 32) if value & 0x80000000 else value


def expected_add(a, imm):
    total = signed(a) + imm
    return total & MASK, not -(1 << 31) <= total < (1 << 31)


for simm in (0x0000, 0x0001, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF):
    imm = simm - 0x10000 if simm >= 0x8000 else simm
    asm, stage = translate(encode(SOPK_MULK, 4, simm))
    check('decodes s_mulk_i32 0x%x' % simm, asm == 's_mulk_i32 s4, 0x%x' % simm, asm)
    ok = all(run(stage.lines, a)[0] == (signed(a) * imm) & MASK
             for a in (0, 1, 3, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x12345678))
    check('s_mulk_i32 0x%x multiplies by the sign-extended immediate' % simm, ok, ' '.join(stage.lines))
    check('s_mulk_i32 0x%x leaves SCC alone' % simm, all('scc' not in line for line in stage.lines), ' '.join(stage.lines))

    asm, stage = translate(encode(SOPK_ADDK, 4, simm))
    check('decodes s_addk_i32 0x%x' % simm, asm == 's_addk_i32 s4, 0x%x' % simm, asm)
    ok = all(run(stage.lines, a) == expected_add(a, imm)
             for a in (0, 1, 3, 0x7FFF, 0x7FFFFFFF, 0x7FFFFFF0, 0x80000000, 0x80000001, 0xFFFFFFFF, 0x12345678))
    check('s_addk_i32 0x%x adds and sets SCC to the signed overflow' % simm, ok, ' '.join(stage.lines))

# SCC is written from the old value, so the result write must come after it.
_asm, stage = translate(encode(SOPK_ADDK, 4, 1))
check('s_addk_i32 reads the old destination for SCC', stage.lines[0].startswith('scc =') and stage.lines[1].startswith('s4 ='),
      ' '.join(stage.lines))

# A destination that is not a plain SGPR is refused rather than lowered wrongly.
refused = True
try:
    translate(encode(SOPK_MULK, 106, 2))
except G.Unsupported:
    pass
else:
    refused = False
check('a non-SGPR destination is refused', refused)

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
