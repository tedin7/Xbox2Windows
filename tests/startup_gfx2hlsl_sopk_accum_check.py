"""Scalar SOPK accumulate forms s_addk_i32 and s_mulk_i32 in tools/startup/gfx2hlsl.py.

  (no arguments)   translation controls, pure Python, no device

Semantics (RDNA 2 ISA, section 12.2 SOPK opcodes 15 and 16, cross-checked with LLVM 15 SOPInstructions.td):
  s_addk_i32  tmp = D.i32; D.i32 = D.i32 + signext(SIMM16); SCC = (tmp[31] == SIMM16[15] && tmp[31] != D.i32[31]),
              the signed overflow of the sum.
  s_mulk_i32  D.i32 = D.i32 * signext(SIMM16). The ISA names no SCC write, so the lowering leaves the emitted scc alone.
              LLVM 15 lists SCC under Defs for both forms (one `let Defs = [SCC]` block), so the translator also stops
              trusting its SCC uniformity proof after s_mulk_i32.
D is destination and source. The decoder prints the 16-bit field as unsigned hex (`s_addk_i32 s0, 0x8000`); llvm-mc
(LLVM 15, gfx1013) accepts that spelling and encodes it to the words pinned in ENCODINGS below.

The emitted lines are run in order by the integer evaluator of startup_gfx2hlsl_bit_alu_check.py. The reference here is
plain Python integer arithmetic: the exact sum or product of the signed operands, reduced to 32 bits, and SCC set
exactly when the exact signed sum leaves [-2^31, 2^31). SCC starts at 2 (not a value SCC can take), so a form that
writes it or fails to preserve it is visible.

Refused before any state changes: D holding a lane mask, descriptor or pointer data, destinations that are not a single
SGPR below 106, immediates other than the decoder's 0x0..0xffff spelling, and extra operands or modifiers.

Mutations (each must turn the battery red): immediate not sign-extended, unsigned carry as SCC, wrong constant-fold
SCC, s_mulk_i32 writing SCC, no mask refusal, and the lowering removed (the behavior of a translator without it).
"""
import re
import struct
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
OPCODE = {name: op for op, (name, _) in D.SOPK_OPS.items()}
IMMEDIATES = [0, 1, 0x7fff, 0x8000, 0xffff]
VALUES = [0, 1, 2, 0x7fff, 0x8000, 0x7ffffffe, 0x7fffffff, 0x80000000, 0x80000001, 0xffff8000, 0xffff8001,
          0xfffffffe, MASK, 0x12345678, 0xedcba987]
SCC_UNSET = 2
REG = 5
ENCODINGS = {  # llvm-mc LLVM 15.0.7, amdgcn-amd-amdhsa gfx1013: (name, sdst, simm) -> little-endian bytes
    ('s_addk_i32', 0, 0x8000): '008080b7', ('s_mulk_i32', 0, 0xfffe): 'feff00b8',
    ('s_addk_i32', 0, 0xffff): 'ffff80b7', ('s_mulk_i32', 1, 0x7fff): 'ff7f01b8',
    ('s_addk_i32', 0, 0): '000080b7',
}


def sopk(name, sdst, simm):
    return struct.pack('<I', (0b1011 << 28) | (OPCODE[name] << 23) | (sdst << 16) | simm)


def s32(value):
    return value - (1 << 32) if value >> 31 else value


def sext16(value):
    return value - 0x10000 if value & 0x8000 else value


def reference(name, value, imm):
    """(result, overflow flag or None) from exact integer arithmetic."""
    if name == 's_addk_i32':
        total = s32(value) + sext16(imm)
        return total & MASK, int(not -(1 << 31) <= total < (1 << 31))
    return (s32(value) * sext16(imm)) & MASK, None


def decode_one(words):
    decoded = G.decode(words)
    assert len(decoded) == 1 and decoded[0][1] == len(words), decoded
    return decoded[0][2], decoded[0][3]


def make_probe(mod):
    class Probe(mod.Stage):
        """A Stage with no pipeline and with the uniformity state of a real one."""

        def __init__(self):
            self.lines, self.sym = [], {}
            self.used_s, self.used_v, self.uniform_v = set(), set(), set()
            self.masks, self.varying_s = set(), set()
            self.scc_uniform = True
            self.exec_known = True

    return Probe


def run_lines(lines, env):
    env = dict(env, true=1, false=0)
    for line in lines:
        m = re.fullmatch(r'(\w+) = (.*);', line)
        assert m, line
        env[m.group(1)] = Expr(m.group(2)).run(env)[0]
    return env


def battery(mod):
    """Every check as (label, ok, detail) against the translator module `mod`."""
    results = []
    Probe = make_probe(mod)

    def check(label, ok, detail=''):
        results.append((label, bool(ok), detail))

    def translate(name, imm, sdst=REG, setup=None):
        words = sopk(name, sdst, imm)
        text_name, asm = decode_one(words)
        stage = Probe()
        if setup:
            setup(stage)
        stage.one(text_name, asm)
        return asm, stage

    def refusal(asm_name, asm, setup=None):
        stage = Probe()
        if setup:
            setup(stage)
        before = (set(stage.masks), dict(stage.sym), set(stage.varying_s), stage.scc_uniform)
        try:
            stage.one(asm_name, asm)
        except mod.Unsupported as error:
            after = (stage.masks, stage.sym, stage.varying_s, stage.scc_uniform)
            return str(error), not stage.lines and after == before
        return None, False

    check('encoder equals the llvm-mc words and the decoder reads them back',
          all(sopk(*key).hex() == word and decode_one(sopk(*key))[1] == '%s s%d, 0x%x' % key
              for key, word in ENCODINGS.items()))

    # 1. Values and SCC against the integer reference.
    for name in ('s_addk_i32', 's_mulk_i32'):
        bad = None
        for imm in IMMEDIATES:
            asm, stage = translate(name, imm)
            if asm != '%s s%d, 0x%x' % (name, REG, imm):
                bad = (asm, 'spelling')
                break
            for value in VALUES:
                for scc_in in (SCC_UNSET, 0, 1):
                    env = run_lines(stage.lines, {'s%d' % REG: value, 'scc': scc_in})
                    want, flag = reference(name, value, imm)
                    want_scc = scc_in if flag is None else flag
                    if (env['s%d' % REG], env['scc']) != (want, want_scc):
                        bad = (asm, hex(value), scc_in, (env['s%d' % REG], env['scc']), (want, want_scc))
                        break
                if bad:
                    break
            if bad:
                break
        check('%s: %d immediates x %d values x 3 SCC inputs equal the reference' % (name, len(IMMEDIATES), len(VALUES)),
              bad is None, str(bad))

    # 2. Named overflow cases, including the signed and unsigned carries that differ.
    for value, imm, overflow, result in (
            (0x7fffffff, 1, 1, 0x80000000),          # up
            (0x80000000, 0xffff, 1, 0x7fffffff),     # down: INT_MIN + (-1)
            (0x7ffffffe, 1, 0, 0x7fffffff),          # no overflow, last positive value
            (0x80000001, 0xffff, 0, 0x80000000),     # no overflow, reaches INT_MIN
            (MASK, 0xffff, 0, 0xfffffffe),           # -1 + -1: an unsigned carry here, no signed overflow
            (0x7fffffff, 0, 0, 0x7fffffff),
            (0x7fffffff, 0x7fff, 1, 0x80007ffe),
            (0x80000000, 0x8000, 1, 0x7fff8000)):
        _, stage = translate('s_addk_i32', imm)
        env = run_lines(stage.lines, {'s%d' % REG: value, 'scc': SCC_UNSET})
        check('addk 0x%08x + 0x%04x -> 0x%08x, SCC %d' % (value, imm, result, overflow),
              (env['s%d' % REG], env['scc']) == (result, overflow), str((env['s%d' % REG], env['scc'])))

    # 3. A later SCC consumer reads the overflow; s_mulk_i32 keeps a previous compare result.
    def consume(first_lines):
        stage = Probe()
        for text in first_lines:
            name, asm = text if isinstance(text, tuple) else (text.split()[0], text)
            stage.one(name, asm)
        stage.one('s_cselect_b32', 's_cselect_b32 s9, 1, 0')
        return stage

    seen = []
    for value, imm in ((0x7fffffff, 1), (0x7ffffffe, 1), (0x80000000, 0xffff), (MASK, 0xffff)):
        stage = consume(['s_addk_i32 s%d, 0x%x' % (REG, imm)])
        env = run_lines(stage.lines, {'s%d' % REG: value, 'scc': SCC_UNSET})
        seen.append((env['s9'], reference('s_addk_i32', value, imm)[1]))
    check('s_cselect_b32 after addk reads the overflow: %s' % seen, all(a == b for a, b in seen), str(seen))
    seen = []
    for compare, want in (('s_cmp_eq_u32 s7, s7', 1), ('s_cmp_lg_u32 s7, s7', 0)):
        stage = consume([compare, 's_mulk_i32 s%d, 0xfffe' % REG])
        env = run_lines(stage.lines, {'s%d' % REG: 0x7fffffff, 's7': 3, 'scc': SCC_UNSET})
        seen.append((env['s9'], want, env['s%d' % REG]))
    check('s_cselect_b32 after mulk still reads the earlier compare: %s' % seen,
          all(a == b and c == 2 for a, b, c in seen), str(seen))

    # 4. Constant folding: a known D folds to a known result and a known SCC, run-time lines agree.
    bad = None
    for name in ('s_addk_i32', 's_mulk_i32'):
        for imm in IMMEDIATES:
            for value in VALUES:
                asm, stage = translate(name, imm, setup=lambda s, v=value: s.sym.__setitem__(REG, ('const', v)))
                want, flag = reference(name, value, imm)
                env = run_lines(stage.lines, {'scc': SCC_UNSET})
                got = (stage.sym.get(REG), env['s%d' % REG], env['scc'])
                if got != (('const', want), want, SCC_UNSET if flag is None else flag):
                    bad = (asm, hex(value), got)
                    break
            if bad:
                break
    check('constants fold to the reference (result, symbol and SCC)', bad is None, str(bad))
    _, stage = translate('s_addk_i32', 1, setup=lambda s: s.sym.__setitem__(REG, ('const', 5)))
    check('folded addk emits plain constants', stage.lines == ['scc = false;', 's%d = 0x00000006u;' % REG],
          str(stage.lines))

    # 5. Uniformity: D uniform before stays uniform, a varying or unknown D stays varying, SCC follows D.
    _, stage = translate('s_addk_i32', 1)
    check('addk of an untracked-uniform D keeps D and SCC uniform',
          stage.uniform_source('s%d' % REG) and stage.scc_uniform, str((stage.varying_s, stage.scc_uniform)))
    _, stage = translate('s_addk_i32', 1, setup=lambda s: s.varying_s.add(REG))
    check('addk of a varying D keeps D varying and SCC varying',
          not stage.uniform_source('s%d' % REG) and not stage.scc_uniform, str((stage.varying_s, stage.scc_uniform)))
    _, stage = translate('s_mulk_i32', 3, setup=lambda s: s.varying_s.add(REG))
    check('mulk of a varying D keeps D varying', not stage.uniform_source('s%d' % REG), str(stage.varying_s))
    _, stage = translate('s_mulk_i32', 3)
    check('mulk of a uniform D keeps D uniform and drops the SCC proof',
          stage.uniform_source('s%d' % REG) and not stage.scc_uniform, str((stage.varying_s, stage.scc_uniform)))
    _, stage = translate('s_mulk_i32', 3, setup=lambda s: s.sym.__setitem__(REG, ('ud', 3)))
    check('mulk of uniform data is uniform', stage.uniform_source('s%d' % REG) and stage.sym.get(REG) is None)

    # 6. Refusals leave masks, symbols and uniformity state as they were and emit nothing.
    def as_mask(stage):
        stage.masks.add(REG)

    message, intact = refusal('s_addk_i32', 's_addk_i32 s%d, 0x1' % REG, as_mask)
    check('addk on a compare-mask SGPR is refused before D is touched', message == 's_addk_i32 on a lane mask' and intact,
          str((message, intact)))
    message, intact = refusal('s_mulk_i32', 's_mulk_i32 s%d, 0x1' % REG, as_mask)
    check('mulk on a compare-mask SGPR is refused before D is touched', message == 's_mulk_i32 on a lane mask' and intact,
          str((message, intact)))
    stage = Probe()
    stage.one('v_cmp_lt_f32_e64', 'v_cmp_lt_f32_e64 s%d, v0, v1' % REG)
    check('a real compare marks the SGPR as a mask', REG in stage.masks, str(stage.masks))
    try:
        stage.one('s_addk_i32', 's_addk_i32 s%d, 0x1' % REG)
        ok = False
    except mod.Unsupported:
        ok = REG in stage.masks
    check('addk on the destination of a real compare is refused and keeps the mask flag', ok)
    for kind in ('desc', 'descnum', 'ptr', 'ptrhi', 'execcopy', 'smpheap'):
        for name in ('s_addk_i32', 's_mulk_i32'):
            key = ('desc', ('table', 0), 0, 0) if kind == 'desc' else (kind, 0)
            message, intact = refusal(name, '%s s%d, 0x1' % (name, REG),
                                      lambda s, key=key: s.sym.__setitem__(REG, key))
            want = '%s on a lane mask' % name if kind == 'execcopy' else '%s on %s data' % (name, kind)
            check('%s on %s tracking is refused and keeps the symbol' % (name, kind), message == want and intact,
                  str((message, intact)))
    for operands in ('s[2:3], 0x1', 'vcc_lo, 0x1', 'exec_lo, 0x1', 'm0, 0x1', 'v1, 0x1', 's106, 0x1', 's5, s6', 's5',
                     's5, 0x1, 0x2', 's5, 0x10000', 's5, -1', 's5, 1', 's5, 0X1', 's5, 0x', 's5, 1.0'):
        for name in ('s_addk_i32', 's_mulk_i32'):
            message, intact = refusal(name, '%s %s' % (name, operands))
            check('%s %s is refused' % (name, operands), message is not None and intact, str((message, intact)))
    for name in ('s_addk_i32', 's_mulk_i32'):
        message, _ = refusal(name, '%s s5, 0x1 clamp' % name)
        check('%s with a modifier is refused' % name, message is not None, str(message))
    return results


def load_mutant(old, new):
    """gfx2hlsl.py with one source line replaced, loaded as its own module."""
    source = Path(G.__file__).read_text(encoding='utf-8').replace('\r\n', '\n')
    assert source.count(old) == 1, old
    module = types.ModuleType('gfx2hlsl_mutant')
    module.__file__ = G.__file__
    exec(compile(source.replace(old, new), G.__file__, 'exec'), module.__dict__)
    return module


MUTATIONS = [
    ('immediate not sign-extended',
     'imm = (simm - 0x10000 if simm & 0x8000 else simm) & 0xffffffff', 'imm = simm'),
    ('unsigned carry instead of signed overflow',
     "overflow = '((%s ^ (%s)) & ~(%s ^ %s) & 0x80000000u) != 0u' % (old, result, old, imm_text)",
     "overflow = '(%s) < %s' % (result, old)"),
    ('constant fold with the unsigned carry',
     "overflow = 'true' if ((sym[1] ^ total) & ~(sym[1] ^ imm) & 0x80000000) else 'false'",
     "overflow = 'true' if (sym[1] + imm) > MASK_ else 'false'"),
    ('mulk writes SCC',
     'elif self.tracks_uniformity():\n            self.scc_uniform = False',
     "else:\n            self.set_scc('%s != 0u' % result)"),
    ('no mask refusal',
     "if d in self.masks or self.sym.get(d, ('',))[0] == 'execcopy':", 'if False:'),
    ('no descriptor refusal',
     "if sym and sym[0] not in ('const', 'ud', 'fixed_buffer_word'):", 'if False:'),
    ('lowering removed', 'def op_s_addk_i32(self, ops, mods):', 'def op_unused_addk(self, ops, mods):'),
]


def main():
    failures = []
    for label, ok, detail in battery(G):
        print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
        if not ok:
            failures.append(label)
            if detail:
                print('  ' + detail)
    for label, old, new in MUTATIONS:
        mutant = load_mutant(old, new.replace('MASK_', '0xffffffff'))
        try:
            red = [result[0] for result in battery(mutant) if not result[1]]
        except Exception as error:  # a mutant that breaks the translation outright is also caught
            red = ['%s: %s' % (type(error).__name__, error)]
        print('mutation %s: %s' % (label, 'caught by %d checks' % len(red) if red else 'NOT CAUGHT'), flush=True)
        if not red:
            failures.append('mutation not caught: ' + label)
    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
