"""v_swap_b32 in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_swap_check.py [binary <llvm-mc>]

(no arguments)       pure Python: decode the VOP1 words, translate, evaluate the translated text on the CPU, refusals
binary <llvm-mc>     also reassemble the source lines with LLVM 15 (gfx1013) and compare them with the words built here

Semantics (RDNA 2 ISA, VOP1 opcode 101, "V_SWAP_B32: Swap operands. Input and output modifiers not supported; this is an
untyped operation."; the SDWA and DPP limitation lists in section 12.19 also name V_SWAP_B32): VDST and SRC0 are both
VGPRs and exchange their old values, so both are read before either is written. The reference below is a plain Python
exchange of two list entries, not the lowering's expression. LLVM 15 for gfx1013 assembles only the 32-bit VOP1 form:
the e64, dpp and sdwa spellings, a modifier, a scalar or constant or negated or 64-bit operand are all rejected by
llvm-mc, so the translator refuses them too.

The translated text is run by a small evaluator over the statements the fixture programs produce. Two mutations of the
lowering (the first write happens before the read; only one destination is written) are loaded from the translator
source and must read wrong words, so the check cannot pass with a wrong lowering.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
import gfx2hlsl as G  # noqa: E402
import disasm_rdna2 as D  # noqa: E402

MASK = 0xffffffff
LOWERING = "        body = '{ uint t = v%d; v%d = v%d; v%d = t; }' % (dst, dst, src, src)"
MUTATIONS = {
    # the destination is written before the source is read: both registers end with the source's value
    'write before read': "        body = '{ v%d = v%d; uint t = v%d; v%d = t; }' % (dst, src, dst, src)",
    # only the destination is written: the source keeps its value
    'one destination': "        body = '{ uint t = v%d; v%d = v%d; v%d = v%d; }' % (dst, dst, src, src, src)",
}
failures = []


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH  ' + detail))
    if not condition:
        failures.append(label)


def probe(module, exec_known=True, uniform=()):
    class Probe(module.Stage):
        """A Stage with no pipeline: enough state for single instructions to be translated."""

        def __init__(self):
            self.lines, self.sym = [], {}
            self.used_s, self.used_v, self.uniform_v = set(), set(), set(uniform)
            self.exec_known = exec_known
    return Probe()


def translate(module, source, **kwargs):
    stage = probe(module, **kwargs)
    for line in source:
        stage.one(line.split(' ', 1)[0], line)
    return stage


def evaluate(lines, regs):
    """Run translated statements (assignments, a temporary in braces, an optional exec guard) on a register dict."""
    regs = dict(regs)
    for line in lines:
        line = re.sub(r'^if \(exec & 1\) ', '', line)
        for statement in re.sub(r'[{}]', '', line).split(';'):
            statement = re.sub(r'^\s*uint ', '', statement.strip())
            if not statement:
                continue
            target, _, expression = statement.partition(' = ')
            expression = re.sub(r'\b(0x[0-9a-f]+)u\b', r'\1', expression)
            if not re.fullmatch(r'[0-9a-fxtv +^-]*', expression) or not re.fullmatch(r't|v\d+', target):
                raise AssertionError('evaluator does not know: %s' % statement)
            regs[target] = eval(expression, {}, regs) & MASK
    return regs


def vop1_word(vdst, src):
    """VOP1 word: 0x7e000000 | vdst << 17 | opcode 101 << 9 | src0 (0x100 + VGPR)."""
    return (0x7e000000 | vdst << 17 | 101 << 9 | (0x100 + src)).to_bytes(4, 'little')


def swap_ref(values, dst, src):
    """Reference: exchange two list entries."""
    values = list(values)
    values[dst], values[src] = values[src], values[dst]
    return values


SEED = {1: 0x11111111, 2: 0x22222222, 3: 0x33333333, 4: 0x44444444, 255: 0xdeadbeef, 0: 0x00000001}


def mov(reg):
    return 'v_mov_b32 v%d, 0x%08x' % (reg, SEED[reg])


def run(module, program):
    stage = translate(module, program)
    return evaluate(stage.lines, {}), stage


def cases(module):
    """(label, program, expected registers) for the positive fixtures."""
    s = SEED
    return [
        ('plain swap, then both registers are read',
         [mov(1), mov(2), 'v_swap_b32 v1, v2', 'v_sub_nc_u32 v3, v1, v2'],
         {'v1': s[2], 'v2': s[1], 'v3': (s[2] - s[1]) & MASK}),
        ('operand order does not matter',
         [mov(1), mov(2), 'v_swap_b32 v2, v1', 'v_sub_nc_u32 v3, v1, v2'],
         {'v1': s[2], 'v2': s[1], 'v3': (s[2] - s[1]) & MASK}),
        ('self swap keeps the value',
         [mov(3), 'v_swap_b32 v3, v3', 'v_xor_b32 v4, v3, v3'],
         {'v3': s[3], 'v4': 0}),
        ('a register written by another instruction first',
         [mov(1), mov(2), 'v_add_nc_u32 v1, v1, v2', 'v_swap_b32 v1, v2'],
         {'v1': s[2], 'v2': (s[1] + s[2]) & MASK}),
        ('two swaps rotate three registers',
         [mov(1), mov(2), mov(3), 'v_swap_b32 v1, v2', 'v_swap_b32 v2, v3'],
         {'v1': s[2], 'v2': s[3], 'v3': s[1]}),
        ('the same swap twice restores both',
         [mov(1), mov(2), 'v_swap_b32 v1, v2', 'v_swap_b32 v1, v2'],
         {'v1': s[1], 'v2': s[2]}),
        ('highest and lowest VGPR',
         [mov(255), mov(0), 'v_swap_b32 v255, v0'],
         {'v255': s[0], 'v0': s[255]}),
    ]


def run_checks(module):
    """The positive fixtures against the reference; returns the labels of the mismatching ones."""
    bad = []
    for label, program, expected in cases(module):
        try:
            regs, stage = run(module, program)
        except Exception as error:  # noqa: BLE001  report which exception a mutated or old translator raises
            bad.append((label, type(error).__name__ + ': ' + str(error)))
            continue
        got = {k: regs.get(k) for k in expected}
        if got != expected:
            bad.append((label, 'got %r expected %r' % (got, expected)))
    return bad


def refused(module, name, line, **kwargs):
    try:
        probe(module, **kwargs).one(name, line)
    except module.Unsupported as error:
        return str(error)
    except Exception as error:  # noqa: BLE001
        return 'other exception %s' % type(error).__name__
    return None


# 1. the documented VOP1 encoding decodes to the spelling the translator reads, then translates
for dst, src in ((1, 2), (3, 3), (255, 0), (0, 255)):
    n, name, asm, _ = D.decode_instruction(vop1_word(dst, src), 0)
    check('decode v%d, v%d' % (dst, src), (n, name, asm) == (4, 'v_swap_b32', 'v_swap_b32 v%d, v%d' % (dst, src)),
          repr((n, name, asm)))

# 2. translated text against the reference
bad = run_checks(G)
check('fixtures: translated text matches the reference exchange (%d programs)' % len(cases(G)), not bad, repr(bad))
ref = swap_ref([SEED[1], SEED[2], SEED[3]], 0, 1)
check('reference helper exchanges two entries', ref == [SEED[2], SEED[1], SEED[3]])

# 3. the emitted statement reads before it writes and names both registers
stage = translate(G, ['v_swap_b32 v1, v2'])
check('text of one swap', stage.lines == ['{ uint t = v1; v1 = v2; v2 = t; }'], repr(stage.lines))
check('both registers are marked used', {1, 2} <= stage.used_v, repr(stage.used_v))
stage = translate(G, ['v_swap_b32 v1, v2'], exec_known=False)
check('unknown exec guards the swap', stage.lines == ['if (exec & 1) { uint t = v1; v1 = v2; v2 = t; }'],
      repr(stage.lines))

# 4. uniformity travels with the value, and is dropped when exec is unknown
stage = translate(G, ['v_swap_b32 v1, v2'], uniform=(1,))
check('uniform flag moves from v1 to v2', stage.uniform_v == {2}, repr(stage.uniform_v))
stage = translate(G, ['v_swap_b32 v2, v1'], uniform=(1,))
check('uniform flag moves with the operand order reversed', stage.uniform_v == {2}, repr(stage.uniform_v))
stage = translate(G, ['v_swap_b32 v1, v2'], uniform=(1, 2))
check('two uniform registers stay uniform', stage.uniform_v == {1, 2}, repr(stage.uniform_v))
stage = translate(G, ['v_swap_b32 v1, v2'], exec_known=False, uniform=(1, 2))
check('unknown exec drops the uniform proof', stage.uniform_v == set(), repr(stage.uniform_v))

# 5. refusals: every form llvm-mc rejects, and the operands the lowering does not model
for label, name, line in (
        ('clamp', 'v_swap_b32', 'v_swap_b32 v1, v2 clamp'),
        ('omod', 'v_swap_b32', 'v_swap_b32 v1, v2 mul:2'),
        ('op_sel', 'v_swap_b32', 'v_swap_b32 v1, v2 op_sel:[1,0]'),
        ('scalar source', 'v_swap_b32', 'v_swap_b32 v1, s2'),
        ('scalar destination', 'v_swap_b32', 'v_swap_b32 s2, v1'),
        ('inline constant', 'v_swap_b32', 'v_swap_b32 v1, 1'),
        ('negated source', 'v_swap_b32', 'v_swap_b32 v1, -v2'),
        ('64-bit destination', 'v_swap_b32', 'v_swap_b32 v[1:2], v3'),
        ('64-bit source', 'v_swap_b32', 'v_swap_b32 v1, v[2:3]'),
        ('VGPR 256', 'v_swap_b32', 'v_swap_b32 v256, v1'),
        ('one operand', 'v_swap_b32', 'v_swap_b32 v1'),
        ('three operands', 'v_swap_b32', 'v_swap_b32 v1, v2, v3'),
        ('dpp', 'v_swap_b32_dpp', 'v_swap_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf'),
        ('sdwa', 'v_swap_b32_sdwa', 'v_swap_b32_sdwa v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD'),
        ('decoder record of the VOP3A spelling', 'v_swap_b32_unsupported', 'v_swap_b32 <unsupported VOP3A encoding>')):
    message = refused(G, name, line)
    check('refused: %s' % label, message is not None and not message.startswith('other exception'), repr(message))
# the e64 words decode to the unsupported record, which has no handler
n, name, asm, _ = D.decode_instruction((0xd5e50001).to_bytes(4, 'little') + (0x102).to_bytes(4, 'little'), 0)
check('VOP3A words of v_swap_b32 decode to the unsupported record', (n, name) == (8, 'v_swap_b32_unsupported'),
      repr((n, name, asm)))
check('...and that record is refused', refused(G, name, asm) is not None)

# 6. mutation controls: load the translator with one wrong lowering and require the fixtures to fail
source = Path(G.__file__).read_text(encoding='utf-8')
check('control: the lowering line is in the translator once', source.count(LOWERING) == 1)
for label, wrong in MUTATIONS.items():
    mutated = {'__file__': G.__file__, '__name__': 'gfx2hlsl_mutated'}
    exec(compile(source.replace(LOWERING, wrong), G.__file__, 'exec'), mutated)

    class Module:
        Stage, Unsupported = mutated['Stage'], mutated['Unsupported']
    bad = run_checks(Module)
    check('control: "%s" fails the fixtures (%d of %d)' % (label, len(bad), len(cases(G))), bool(bad))

# 7. optional: the source lines assemble to the words built here
if sys.argv[1:2] == ['binary']:
    from startup_flat_param_check import assemble  # noqa: E402
    for dst, src in ((1, 2), (3, 3), (255, 0)):
        built = assemble(sys.argv[2], 'v_swap_b32 v%d, v%d\n' % (dst, src))
        check('llvm-mc v_swap_b32 v%d, v%d' % (dst, src), built == vop1_word(dst, src), built.hex())

print('v_swap_b32 checks: %d failed' % len(failures))
sys.exit(1 if failures else 0)
