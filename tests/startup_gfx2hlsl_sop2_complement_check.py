"""Scalar SOP2 complement logic (#1919), high-halfword packing (#1996) and unsigned multiply-high in
tools/startup/gfx2hlsl.py.

  (no arguments)        translation controls, pure Python
  nand-draw <dxc> <offscreen_check>  WARP pixel readback and a missing-NOT control
  binary <llvm-mc>      assemble the decoded text of every form with llvm-mc and compare with the words built here
  decode <libLLVM>      disassemble the same words with LLVM's C API and compare with tools/disasm_rdna2.py

The instruction words are built here from the SOP2 layout that tools/disasm_rdna2.py reads: [31:30] = 0b10,
[29:23] opcode, [22:16] sdst, [15:8] ssrc1, [7:0] ssrc0, and one literal dword after the word when a source is 255.

Each form is decoded and translated, and the emitted lines (the result, then SCC) are run in order on the CPU by the
integer evaluator of startup_gfx2hlsl_bit_alu_check.py over edge values: 0, all ones, alternating bits, every single
bit and its complement, and mixed words. The reference is written here with plain integer arithmetic from the ISA
meaning: D = ~(S0 op S1) on 32 bits and SCC = (D != 0). SCC starts at a value neither 0 nor 1, so a form that does not
write it fails. The value set gives SCC = 0 and SCC = 1 for every operation.

The lowering text and SCC rule equal pixel_aot's SCALAR entries for the same operations (#561). Every numeric form
(SGPR, inline, literal or m0 operands) emits the lines of its non-complement sibling (s_and_b32, s_or_b32, s_xor_b32)
with the operation wrapped in ~( ).

Lane masks: vcc_lo, exec_lo or an SGPR holding a compare mask is the whole wave's mask, of which the translator keeps
one lane's bit. Bit i of ~(S0 op S1) depends only on bit i of each source, so with every source a lane mask and an
SGPR destination the lane's bit of the result is exact: the form is accepted and checked here on every lane of the
edge words. SCC is set when any bit of the complemented 32-bit result is, which the other lanes decide (the bit of an
idle lane is set); the per-lane projection cannot show that, so SCC is not emitted and a program that can read it
before the next SCC write is refused (through branches too). NAND can take a uniform-bit immediate (all zeroes or
all ones) alongside a lane mask. Other mixed inputs and vcc_lo or exec_lo destinations stay refused. The mask mark
follows program order, so a mark is a proof at a branch
target only if the register's last write dominates the target and no block on a path from that write to the target
writes the register again; otherwise the complement is refused until the register is written again. Mutations: the
join rule disabled and the SCC read scan disabled must each accept a refused program.

Mutations: the table entry with the complement dropped, and the entry with the SCC write dropped, must each read a
wrong result or SCC. Refused: every SOP2 operation the decoder names that still has no lowering.

The `s_pack_hh_b32_b16` forms use the same encoder and evaluator. The reference places the high half of src0
in the low result half and the high half of src1 in the high result half, preserving SCC. Swapping the inputs
must fail the value check.

The `s_bfe_u32` forms use the same encoder and evaluator. The reference reads the ISA text: offset = S1[4:0], width =
S1[22:16], a width of 0 gives 0, offset + width < 32 gives the field, and any wider field gives S0 >> offset. SCC is
D != 0. HLSL reduces a shift count to 5 bits, so a lowering that shifts by the 7-bit width gives 0 at width 32 and a
truncated mask at 33 to 127. The inputs reach widths 0 to 127 with the source sign bit set, as SGPR, inline and
literal controls. The mutated table entries must read a wrong word.

The `s_mul_hi_u32` forms use the same encoder and evaluator. The emitted mul_hi_u32 helper is taken from the
translator's common preamble and run statement by statement in 32-bit arithmetic; the reference is (a * b) >> 32 on
Python integers, SCC unchanged. The value set adds 0 x 0, 1 x 1, all ones squared, 0x80000000 x 2 and seeded random
pairs. Mutations that must each fail: the low word of the product, the signed high word, the helper without the
low-halfword carry, the helper without the final carry limb, an added SCC write and a disabled lowering.
"""
import random
import re
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
import pixel_aot as P  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    POST_WORDS, PRE_WORDS, Expr as BitExpr, vs_words, words_text)

MASK = 0xffffffff
OPCODE = {name: op for op, (name, _) in D.SOP2_OPS.items()}
PACK_HH = 's_pack_hh_b32_b16'
BFE = 's_bfe_u32'
# operation -> (non-complement sibling, reference)
OPS = {
    's_nand_b32': ('s_and_b32', lambda a, b: ~(a & b) & MASK),
    's_nor_b32': ('s_or_b32', lambda a, b: ~(a | b) & MASK),
    's_xnor_b32': ('s_xor_b32', lambda a, b: ~(a ^ b) & MASK),
}
SIBLING_OPERATOR = {'s_and_b32': '&', 's_or_b32': '|', 's_xor_b32': '^'}

SPECIAL = [0, MASK, 0xaaaaaaaa, 0x55555555, 0x80000000, 0x7fffffff, 1, 0xfffffffe, 0xffff0000, 0x0000ffff,
           0x12345678, 0xedcba987, 0xdeadbeef, 0x0f0f0f0f]
VALUES = SPECIAL + [1 << k for k in range(32)] + [~(1 << k) & MASK for k in range(32)]
PAIRS = [(a, b) for a in SPECIAL for b in VALUES] + [(a, b) for a in VALUES for b in SPECIAL]
SCC_UNSET = 2  # not a value SCC can take: a form that leaves SCC alone keeps it

VCC_LO, M0, EXEC_LO, LITERAL = 106, 124, 126, 255
INLINE = {128: 0, 129: 1, 193: MASK}  # ssrc codes: 0, 1, -1


def sop2(name, sdst, ssrc0, ssrc1, literal=None):
    word = (0b10 << 30) | (OPCODE[name] << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0
    return struct.pack('<I', word) + (b'' if literal is None else struct.pack('<I', literal))


# (sdst, ssrc0, ssrc1, literal): the forms with an SGPR or m0 destination and no mask source are checked against
# the integer reference; the others only against the sibling.
VALUE_FORMS = [
    (3, 1, 2, None),            # s3 = s1 op s2
    (1, 1, 2, None),            # in place, first source
    (2, 1, 2, None),            # in place, second source
    (3, 1, 1, None),            # one register on both sides
    (3, 128, 2, None),          # inline 0
    (3, 193, 2, None),          # inline -1
    (3, 1, 129, None),          # inline 1
    (3, 1, 193, None),          # inline -1 as the second source
    (3, LITERAL, 2, 0xaaaaaaaa),
    (3, 1, LITERAL, 0x55555555),
    (M0, 1, 2, None),
]
MASK_FORMS = [
    (3, VCC_LO, 2, None),       # a lane-mask source marks the destination as a mask
    (3, 1, EXEC_LO, None),
    (VCC_LO, 1, 2, None),
    (EXEC_LO, 1, 2, None),
]


def code(name, form):
    return sop2(name, *form)


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for decoded instructions to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True


def decode_one(words):
    decoded = G.decode(words)
    assert len(decoded) == 1 and decoded[0][1] == len(words), decoded
    return decoded[0][2], decoded[0][3]


def translate(words):
    name, asm = decode_one(words)
    stage = Probe()
    stage.one(name, asm)
    return asm, stage


def attempt(words):
    """translate(), with a refusal returned as a stage whose only line names it."""
    try:
        return translate(words)
    except G.Unsupported as error:
        stage = Probe()
        stage.lines = ['refused: %s' % error]
        return decode_one(words)[1], stage


def source_value(code_, literal, env):
    if code_ == LITERAL:
        return literal
    if code_ in INLINE:
        return INLINE[code_]
    return env['s%d' % code_]


def common_lines():
    pipe = object.__new__(G.Pipeline)
    pipe.uds, pipe.resources, pipe.samplers, pipe.uses_sampler_heap = [], {}, {}, False
    return pipe.common()


def mul_hi_helper():
    """The emitted mul_hi_u32 body as (declarations, return expression), parsed for the evaluator."""
    text = ''.join(entry for entry in common_lines() if entry.startswith('uint mul_hi_u32('))
    m = re.fullmatch(r'uint mul_hi_u32\(uint a, uint b\) \{ (.*) \}', text)
    assert m, text
    *statements, result = [part.strip() for part in m.group(1).split(';') if part.strip()]
    assert result.startswith('return '), result
    declarations = []
    for statement in statements:
        assert statement.startswith('uint '), statement
        for item in statement[5:].split(', '):
            name, value = item.split(' = ', 1)
            declarations.append((name, BitExpr(value)))
    return declarations, BitExpr(result[7:])


class Expr(BitExpr):
    """The bit-ALU evaluator, plus the translator's own mul_hi_u32 helper text."""
    helper = None

    def call(self, name, args):
        if name != 'mul_hi_u32':
            return BitExpr.call(name, args)
        assert not args[0][1] and not args[1][1], 'mul_hi_u32 reads uint operands'
        declarations, result = Expr.helper
        env = {'a': args[0][0], 'b': args[1][0]}
        for local, value in declarations:
            env[local] = value.run(env)[0]
        return result.run(env)[0], False


def run_lines(lines, env):
    env = dict(env)
    for line in lines:
        m = re.fullmatch(r'(\w+) = (.*);', line)
        assert m, line
        env[m.group(1)] = Expr(m.group(2)).run(env)[0]
    return env


failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def first_mismatch(name, reference=None, updates_scc=True):
    """The first (form, a, b, got, want) whose result or SCC differs from the reference, or None."""
    if reference is None:
        reference = OPS[name][1]
    for form in VALUE_FORMS:
        sdst, ssrc0, ssrc1, literal = form
        asm, stage = attempt(code(name, form))
        if stage.lines[0].startswith('refused: '):
            return asm, stage.lines[0]
        destination = 'm0' if sdst == M0 else 's%d' % sdst
        for a, b in PAIRS:
            env = {'s1': a, 's2': b, 'scc': SCC_UNSET}
            want = reference(source_value(ssrc0, literal, env), source_value(ssrc1, literal, env))
            out = run_lines(stage.lines, env)
            got = (out.get(destination), out['scc'])
            expected = (want, int(want != 0) if updates_scc else SCC_UNSET)
            if got != expected:
                return asm, hex(a), hex(b), got, expected
    return None


def run_text(*instructions):
    """Translate disassembly text with the branch scan of a real stage -> emitted lines."""
    stage = Probe()
    stage.attrs = {}
    stage.ins = [(4 * i, 4, asm.split()[0], asm) for i, asm in enumerate(instructions)]
    stage.translate()
    return [line for _, lines, _ in stage.blocks for line in lines]


def text_refusal(*instructions):
    try:
        return 'accepted: %s' % run_text(*instructions)
    except G.Unsupported as error:
        return str(error)


def lane_mask_controls():
    operator = {name: SIBLING_OPERATOR[sibling] for name, (sibling, _) in OPS.items()}
    # 1. Every source a lane mask, SGPR destination: one line, the lane's bit of the complement, and no SCC write.
    forms = [('s3, s1, s2', 's3', 's1', 's2'), ('s1, s2, s1', 's1', 's2', 's1'), ('s3, vcc_lo, s1', 's3', 'vcc', 's1'),
             ('s3, s2, exec_lo', 's3', 's2', 'exec'), ('s3, vcc_lo, exec_lo', 's3', 'vcc', 'exec')]
    for name, (_, reference) in OPS.items():
        wrong = []
        for operands, dest, left, right in forms:
            stage = Probe()
            stage.masks.update((1, 2))
            stage.one(name, '%s %s' % (name, operands))
            expected = ['%s = (~(%s %s %s)) & 1;' % (dest, left, operator[name], right)]
            if stage.lines != expected or int(dest[1:]) not in stage.masks or stage.scc_effect != 'a lane-mask complement':
                wrong.append((operands, stage.lines, expected, sorted(stage.masks), stage.scc_effect))
                continue
            for a, b in ((a, b) for a in SPECIAL for b in SPECIAL):
                for lane in range(32):
                    env = {left: (a >> lane) & 1, right: (b >> lane) & 1, 'scc': SCC_UNSET}
                    out = run_lines(stage.lines, env)
                    if (out[dest], out['scc']) != ((reference(a, b) >> lane) & 1, SCC_UNSET):
                        wrong.append((operands, hex(a), hex(b), lane, out[dest], out['scc']))
                        break
        check('%s: %d all-mask forms give the lane bit of the reference on 32 lanes x %d pairs, SCC not written'
              % (name, len(forms), len(SPECIAL) ** 2), not wrong, str(wrong[:2]))

    # 2. Still refused: a constant or plain-word source next to a mask, a mask destination, a high-half mask name.
    for name in OPS:
        wrong = []
        refused_forms = ('s3, s1, s9', 's3, s9, vcc_lo', 's3, s1, -2', 's3, 1, vcc_lo', 'vcc_lo, s1, s2',
                         'exec_lo, s1, s2', 's3, s1, vcc_hi', 'm0, s1, s2')
        if name != 's_nand_b32':
            refused_forms += ('s3, s1, -1', 's3, 0, vcc_lo')
        for operands in refused_forms:
            stage = Probe()
            stage.masks.update((1, 2))
            try:
                stage.one(name, '%s %s' % (name, operands))
                wrong.append((operands, stage.lines))
            except G.Unsupported as error:
                if str(error) != '%s on a lane mask' % name or stage.lines:
                    wrong.append((operands, str(error), stage.lines))
        check('%s: mixed, constant and mask-destination forms stay refused' % name, not wrong, str(wrong[:2]))

    # 3. Whole programs. SCC after the complement is not the guest's: a read before the next SCC write is refused.
    masks = ('v_cmp_eq_f32_e64 s5, 0, v1', 'v_cmp_lt_f32_e64 s4, 0, v2')
    for name in OPS:
        complement = '%s s4, s5, s4' % name
        line = 's4 = (~(s5 %s s4)) & 1;' % operator[name]
        lines = run_text(*masks, complement, 's_and_saveexec_b32 s4, s4', 's_endpgm')
        at = lines.index(line) if line in lines else -1
        check('%s between two compares and s_and_saveexec_b32 is accepted' % name,
              at >= 0 and lines[at + 1:at + 3] == ['w = s4;', 's4 = exec;'] and 'scc = s4 != 0;' not in lines, str(lines))
        lines = run_text(*masks, complement, 's_cmp_eq_u32 s2, 0', 's_cselect_b32 s6, 1, 0', 's_endpgm')
        check('%s then an SCC write then a read is accepted' % name,
              line in lines and any(text.startswith('s6 = scc ?') for text in lines), str(lines))
        programs = {
            's_cselect_b32': (*masks, complement, 's_cselect_b32 s6, 1, 0', 's_endpgm'),
            's_cbranch_scc1': (*masks, complement, 's_cbranch_scc1 label_0010', 's_endpgm'),
            # the read sits behind a jump that skips the SCC write
            's_cmov_b32': (*masks, complement, 's_branch label_0014', 's_cmp_eq_u32 s2, 0', 's_cmov_b32 s6, s7',
                           's_endpgm'),
            # the read sits at a loop head that the complement reaches over the back edge
            's_cselect_b64': ('s_cmp_eq_u32 s2, 0', 's_cselect_b64 s[6:7], 1, 0', *masks, complement,
                              's_cbranch_vccz label_0004', 's_endpgm'),
        }
        got = {reader: text_refusal(*program) for reader, program in programs.items()}
        want = {reader: 'scc from a lane-mask complement in %s' % reader for reader in programs}
        check('%s: an SCC read before the next SCC write is refused (%d programs)' % (name, len(programs)),
              got == want, str(got))

    # 4. The mask mark at a branch target.
    skip = ('s_cmp_eq_u32 s2, 0', 's_cbranch_scc1 label_0010', *masks)   # the branch skips both compares
    loop = (*masks, 's_nand_b32 s6, s5, s4', 's_mov_b32 s5, s3', 's_cmp_eq_u32 s2, 0', 's_cbranch_scc0 label_0008',
            's_endpgm')                                                   # the body overwrites s5 before the back edge
    arms = ('v_cmp_eq_f32_e64 s5, 0, v1', 's_cmp_eq_u32 s2, 0', 's_cbranch_scc1 label_0014', 'v_cmp_lt_f32_e64 s4, 0, v2',
            's_branch label_0020', 'v_cmp_lt_f32_e64 s4, 0, v3', 's_nand_b32 s4, s5, s4', 's_and_saveexec_b32 s4, s4',
            's_endpgm')                                                   # s5 is written before the split only
    joined = 's_nand_b32 on a lane mask joined at a branch target'
    got = [text_refusal(*skip, 's_nand_b32 s6, s5, s4', 's_endpgm'), text_refusal(*loop)]
    check('a mask that another path into the branch target changes is refused (skipped write, loop body write)',
          got == [joined, joined], str(got))
    lines = run_text(*arms)
    check('a mask written before the split and on no arm crosses the join', 's4 = (~(s5 & s4)) & 1;' in lines, str(lines))
    lines = run_text(*skip, *masks, 's_nand_b32 s6, s5, s4', 's_endpgm')
    check('a mask written again after the join is proven again', 's6 = (~(s5 & s4)) & 1;' in lines, str(lines))
    got = text_refusal(*masks, 's_mov_b32 s5, s3', 's_nand_b32 s6, s5, s4', 's_endpgm')
    check('an overwritten mask is a plain word: refused', got == 's_nand_b32 on a lane mask', got)

    # 5. Mutations: each rule disabled accepts a program refused above.
    saved = G.Stage.unjoined
    G.Stage.unjoined = lambda self, target, last_write, regs: set()
    try:
        got = [text_refusal(*skip, 's_nand_b32 s6, s5, s4', 's_endpgm'), text_refusal(*loop)]
    finally:
        G.Stage.unjoined = saved
    check('mutation without the join rule accepts both programs', all(t.startswith('accepted') for t in got), str(got))
    saved = G.Stage.descriptor_scc_reads
    G.Stage.descriptor_scc_reads = lambda self, branches, effects: iter(())
    try:
        got = text_refusal(*masks, 's_nand_b32 s4, s5, s4', 's_cselect_b32 s6, 1, 0', 's_endpgm')
    finally:
        G.Stage.descriptor_scc_reads = saved
    check('mutation without the SCC read scan accepts the read', got.startswith('accepted'), got)


def nand_draw_fixture():
    # Two compare-derived masks and two forms reading EXEC. Select normal floats for exact MRT readback.
    inputs = (0, 7, 0, 0)
    values = (0x3f800000, 0x40000000)
    vs = vs_words([(bits, *values) for bits in inputs])
    ps = bytearray(PRE_WORDS)
    for index in range(2):
        # v_cmp_ne_i32_e64 s20/s21, 0, v0/v1
        ps.extend(struct.pack('<2I', 0xd4850014 + index, 128 | ((256 + index) << 9)))
    forms = ((20, 193, None), (LITERAL, 21, MASK), (EXEC_LO, 128, None), (193, EXEC_LO, None))
    for index, (src0, src1, literal) in enumerate(forms):
        ps.extend(sop2('s_nand_b32', 24 + index, src0, src1, literal))
        # v_cndmask_b32_e64 v12..v15, v4..v7, v8..v11, s24..s27
        ps.extend(struct.pack('<2I', 0xd501000c + index,
                              (260 + index) | ((264 + index) << 9) | ((24 + index) << 18)))
    pipe = G.Pipeline(vs, bytes(ps) + POST_WORDS, root_blob())
    and_bits = (bool(inputs[0]), bool(inputs[1]), False, True)
    correct = words_text(values[not bit] for bit in and_bits)
    wrong = words_text(values[bit] for bit in and_bits)
    return pipe, correct, wrong


def nand_draw(dxc, runner):
    pipe, correct, wrong = nand_draw_fixture()
    shader = pipe.hlsl_ps()
    mutant, edits = re.subn(r'(s2[4-7] = \()~', r'\1', shader)
    assert edits == 4
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', pipe.hlsl_vs(), 'nand_vs', 'vs_6_0', directory)
        ps = compile_hlsl(dxc, 'dxc', shader, 'nand_ps', 'ps_6_0', directory)
        run = draw(runner, 'warp', vs, ps, correct)
        if run.returncode:
            raise AssertionError(run.stderr)
        ps = compile_hlsl(dxc, 'dxc', mutant, 'nand_mutant_ps', 'ps_6_0', directory)
        run = draw(runner, 'warp', vs, ps, correct)
        if run.returncode != 1 or f'actual={wrong}' not in run.stderr:
            raise AssertionError(f'NAND mutant should fail readback with {wrong}: {run.stderr}')
        print('NAND mutant rejected at the expected pixel values')



def pack_controls():
    def reference(a, b):
        return (b // 65536) * 65536 + a // 65536

    check('pack_hh matches the LLVM 15 SOP2 encoding', sop2(PACK_HH, 0, 1, 2).hex() == '0102009a')
    mismatch = first_mismatch(PACK_HH, reference, updates_scc=False)
    check('pack_hh result and untouched SCC across %d forms and %d input pairs' % (len(VALUE_FORMS), len(PAIRS)),
          mismatch is None, str(mismatch))
    if mismatch is not None:
        return

    # The shader's per-lane masks cannot stand in for the wave's full 32-bit words.
    mask_forms = MASK_FORMS + [(3, 107, 2, None), (127, 1, 2, None)]
    masked = [(form, attempt(code(PACK_HH, form))[1].lines) for form in mask_forms]
    check('pack_hh refuses whole-wave mask sources and destinations',
          all(lines == ['refused: %s on a lane mask' % PACK_HH] for _, lines in masked), str(masked))
    _, reserved = attempt(sop2(PACK_HH, 3, 209, 2))
    check('pack_hh refuses a reserved scalar source encoding',
          reserved.lines == ['refused: %s source s209' % PACK_HH], str(reserved.lines))

    for symbolic in ('mask', 'descriptor'):
        stage = Probe()
        if symbolic == 'mask':
            stage.masks.add(1)
        else:
            stage.sym[1] = ('ptr', 0, 0)
        try:
            stage.one(PACK_HH, decode_one(sop2(PACK_HH, 3, 1, 2))[1])
        except G.Unsupported as error:
            reason = 'a lane mask' if symbolic == 'mask' else 'descriptor data'
            check('pack_hh refuses a symbolic %s' % symbolic,
                  str(error) == '%s on %s' % (PACK_HH, reason) and not stage.lines, str(error))
        else:
            check('pack_hh refuses a symbolic %s' % symbolic, False, str(stage.lines))

    saved = G.SCA[PACK_HH]
    try:
        del G.SCA[PACK_HH]
        disabled = first_mismatch(PACK_HH, reference, updates_scc=False)
        check('pack_hh negative control catches a disabled lowering',
              disabled is not None and disabled[-1] == 'refused: %s' % PACK_HH, str(disabled))
        text, scc = saved
        G.SCA[PACK_HH] = (text.format('{1}', '{0}'), scc)
        check('pack_hh negative control catches reversed source halves',
              first_mismatch(PACK_HH, reference, updates_scc=False) is not None)
    finally:
        G.SCA[PACK_HH] = saved


BFE_OFFSETS = [0, 1, 4, 8, 16, 24, 31]
BFE_WIDTHS = [0, 1, 8, 16, 31, 32, 33, 40, 48, 63, 64, 65, 96, 127]
BFE_SOURCES = [MASK, 0x80000000, 0x9a345f50, 0xdeadbeef, 0x12345678, 1, 0, 0xaaaaaaaa, 0x55555555]
# Control bits 15:5 and 31:23 are ignored; the second spelling of each control sets them all.
BFE_CONTROLS = sorted({(w << 16) | o | junk for o in BFE_OFFSETS for w in BFE_WIDTHS for junk in (0, 0xff80ffe0)}
                      | {0xffffffff, 0x007f0000, 0x00200000, 0x00400000})
BFE_PAIRS = [(a, c) for a in BFE_SOURCES for c in BFE_CONTROLS]
# Pairs that separate the old lowering from the ISA: width 32 and 64 read 0 before, a wide field loses its mask.
BFE_WIDE = [(0x80000000, 0x00200000), (MASK, 0x00400000), (0x9a345f50, 0x00280000), (0x80000001, 0x007f0000),
            (0x80000000, 0x003f0000), (0xdeadbeef, 0x00200008)]


def bfe_reference(src, ctrl):
    offset, width = ctrl & 31, (ctrl >> 16) & 127
    if width == 0:
        return 0
    if offset + width < 32:
        return (src >> offset) & ((1 << width) - 1)
    return src >> offset


def bfe_first_mismatch(reference=bfe_reference, pairs=BFE_PAIRS):
    """The first (form, source, control, got, want) of s_bfe_u32 that differs from the ISA reference, or None.

    Control forms: SGPR (separate, in place on either source), m0 destination, inline 0, 1 and -1, and a literal
    for the control and for the source, so each operand kind of the translator is evaluated."""
    cases = [((3, 1, 2, None), pairs), ((1, 1, 2, None), pairs), ((2, 1, 2, None), pairs), ((M0, 1, 2, None), pairs)]
    cases += [((3, 1, code_, None), [(a, c) for a, c in pairs if c == INLINE[code_]]) for code_ in INLINE]
    cases += [((3, 1, LITERAL, c), [(a, x) for a, x in pairs if x == c]) for c in sorted({x for _, x in pairs})]
    cases += [((3, LITERAL, 2, a), [(x, c) for x, c in pairs if x == a]) for a in sorted({x for x, _ in pairs})]
    for form, inputs in cases:
        sdst, ssrc0, ssrc1, literal = form
        asm, stage = attempt(code(BFE, form))
        if stage.lines[0].startswith('refused: '):
            return asm, stage.lines[0]
        destination = 'm0' if sdst == M0 else 's%d' % sdst
        for a, c in inputs:
            env = {'s1': a, 's2': c, 'scc': SCC_UNSET}
            want = reference(source_value(ssrc0, literal, env), source_value(ssrc1, literal, env))
            out = run_lines(stage.lines, env)
            got = (out.get(destination), out['scc'])
            if got != (want, int(want != 0)):
                return asm, hex(a), hex(c), got, (want, int(want != 0))
    return None


def bfe_controls():
    check('s_bfe_u32 matches the LLVM 15 SOP2 encoding', sop2(BFE, 0, 1, 2).hex() == '01028093')
    widths = {(c >> 16) & 127 for _, c in BFE_PAIRS}
    wide = [bfe_reference(a, c) for a, c in BFE_WIDE]
    check('s_bfe_u32 inputs reach widths 0 to 127, and wide fields with a nonzero result',
          widths >= {0, 31, 32, 33, 40, 63, 64, 127} and all(wide), str(wide))
    mismatch = bfe_first_mismatch()
    check('s_bfe_u32 result and SCC across %d (source, control) pairs and all operand kinds' % len(BFE_PAIRS),
          mismatch is None, str(mismatch))
    mismatch = bfe_first_mismatch(pairs=BFE_WIDE)
    check('s_bfe_u32 wide fields (width 32 and above)', mismatch is None, str(mismatch))

    saved = G.SCA[BFE]
    old = '({0} >> ({1} & 31)) & ((1u << (({1} >> 16) & 127)) - 1)'
    try:
        # The reference itself would pass a formula that agrees with it; the old table entry must not.
        G.SCA[BFE] = (old, saved[1])
        found = bfe_first_mismatch()
        check('s_bfe_u32 negative control: the width taken mod 32 by the shift fails: %s' % (found,),
              found is not None)
        G.SCA[BFE] = (saved[0], None)
        found = bfe_first_mismatch()
        check('s_bfe_u32 negative control: no SCC write fails: %s' % (found,), found is not None)
        G.SCA[BFE] = ('({0} >> ({1} & 31)) & ((1u << (({1} >> 16) & 63)) - 1)', saved[1])
        found = bfe_first_mismatch()
        check('s_bfe_u32 negative control: a 6-bit width fails: %s' % (found,), found is not None)
        G.SCA[BFE] = ('{0} >> ({1} & 31)', saved[1])
        found = bfe_first_mismatch()
        check('s_bfe_u32 negative control: no field mask fails: %s' % (found,), found is not None)
    finally:
        G.SCA[BFE] = saved


MUL_HI = 's_mul_hi_u32'
MUL_EDGES = [(0, 0), (1, 1), (MASK, MASK), (0x80000000, 2), (2, 0x80000000), (0x10000, 0x10000), (0xffff, 0xffff),
             (0x1ffff, 0x1ffff), (MASK, 1), (0x80000000, 0x80000000), (0xaaaaaaab, 0xfffffffd)]
_random = random.Random(0x1723)
MUL_PAIRS = MUL_EDGES + PAIRS + [(_random.getrandbits(32), _random.getrandbits(32)) for _ in range(512)]


def mul_reference(a, b):
    return (a * b) >> 32


def mul_mismatch():
    """first_mismatch() over the multiply value set, with the helper re-read from the preamble."""
    global PAIRS
    Expr.helper = mul_hi_helper()
    saved, PAIRS = PAIRS, MUL_PAIRS
    try:
        return first_mismatch(MUL_HI, mul_reference, updates_scc=False)
    finally:
        PAIRS = saved


def mul_controls():
    check('mul_hi matches the LLVM 15 SOP2 encoding', sop2(MUL_HI, 0, 1, 2).hex() == '0102809a')
    text, scc = G.SCA.get(MUL_HI, (None, None))
    arity, aot_text, aot_scc = P.SCALAR['mul_hi_u32']
    check('mul_hi: %r SCC %r equals pixel_aot %r SCC %r' % (text, scc, aot_text, aot_scc),
          arity == 2 and text == aot_text and scc is None and aot_scc is None and MUL_HI not in G.SCC_UNMODELLED)
    check('mul_hi: reference gives 0xfffffffe for all ones squared and 1 for 0x80000000 x 2',
          mul_reference(MASK, MASK) == 0xfffffffe and mul_reference(0x80000000, 2) == 1)
    mismatch = mul_mismatch()
    check('mul_hi result and untouched SCC across %d forms and %d input pairs' % (len(VALUE_FORMS), len(MUL_PAIRS)),
          mismatch is None, str(mismatch))
    asm, stage = translate(sop2(MUL_HI, 3, LITERAL, 2, 0xaaaaaaab))
    check('mul_hi literal form emits the helper call', stage.lines == ['s3 = mul_hi_u32(0xaaaaaaabu, s2);'],
          '%s -> %s' % (asm, stage.lines))

    masked = [(form, attempt(code(MUL_HI, form))[1].lines) for form in MASK_FORMS + [(3, 107, 2, None)]]
    check('mul_hi refuses whole-wave mask sources and destinations',
          all(lines == ['refused: %s on a lane mask' % MUL_HI] for _, lines in masked), str(masked))
    _, reserved = attempt(sop2(MUL_HI, 3, 209, 2))
    check('mul_hi refuses a reserved scalar source encoding',
          reserved.lines == ['refused: %s source s209' % MUL_HI], str(reserved.lines))
    for symbolic in ('mask', 'descriptor'):
        stage = Probe()
        if symbolic == 'mask':
            stage.masks.add(1)
        else:
            stage.sym[1] = ('ptr', 0, 0)
        try:
            stage.one(MUL_HI, decode_one(sop2(MUL_HI, 3, 1, 2))[1])
        except G.Unsupported as error:
            reason = 'a lane mask' if symbolic == 'mask' else 'descriptor data'
            check('mul_hi refuses a symbolic %s' % symbolic,
                  str(error) == '%s on %s' % (MUL_HI, reason) and not stage.lines, str(error))
        else:
            check('mul_hi refuses a symbolic %s' % symbolic, False, str(stage.lines))
    # An SGPR that held a compare mask is overwritten with a plain word, and is no longer a mask.
    stage = Probe()
    stage.masks.add(3)
    stage.one(MUL_HI, decode_one(sop2(MUL_HI, 3, 1, 2))[1])
    check('mul_hi over a mask-marked destination writes the word and drops the mark',
          stage.lines == ['s3 = mul_hi_u32(s1, s2);'] and 3 not in stage.masks, str((stage.lines, stage.masks)))

    # Mutations: each must read a wrong word or SCC somewhere in the value set.
    saved_entry, saved_common = G.SCA[MUL_HI], G.Pipeline.common
    table = {
        'the low word of the product': ('{0} * {1}', None),
        'the signed high word': ('mul_hi_u32({0}, {1}) - (({0} >> 31) != 0u ? {1} : 0u)'
                                 ' - (({1} >> 31) != 0u ? {0} : 0u)', None),
        'an added SCC write': (saved_entry[0], '$ != 0'),
    }
    helper = {
        'the low-halfword carry dropped': (' + ((la * lb) >> 16)', ''),
        'the final carry limb dropped': ('(((middle & 0xffffu) + la * (b >> 16)) >> 16)', '((la * (b >> 16)) >> 16)'),
    }
    try:
        for label, entry in table.items():
            G.SCA[MUL_HI] = entry
            found = mul_mismatch()
            check('mul_hi mutation with %s fails: %s' % (label, found), found is not None)
        G.SCA[MUL_HI] = saved_entry
        for label, (old, new) in helper.items():
            def mutated(self, old=old, new=new):
                lines = saved_common(self)
                assert sum(old in line for line in lines) == 1, old
                return [line.replace(old, new) for line in lines]
            G.Pipeline.common = mutated
            found = mul_mismatch()
            check('mul_hi mutation with %s fails: %s' % (label, found), found is not None)
        G.Pipeline.common = saved_common
        del G.SCA[MUL_HI]
        disabled = mul_mismatch()
        check('mul_hi negative control catches a disabled lowering',
              disabled is not None and disabled[-1] == 'refused: %s' % MUL_HI, str(disabled))
    finally:
        G.SCA[MUL_HI] = saved_entry
        G.Pipeline.common = saved_common


def controls():
    # 1. The encoder writes what the decoder reads, checked on the LLVM 15 encodings in tests/rdna2_tools_check.py.
    known = [(sop2('s_nand_b32', 0, 1, 2), '0102008c'), (sop2('s_nor_b32', 0, 1, 2), '0102008d'),
             (sop2('s_xnor_b32', 0, 1, 2), '0102008e')]
    check('encoder matches LLVM 15 encodings', all(w.hex() == want for w, want in known),
          str([(w.hex(), want) for w, want in known]))
    decoded = [decode_one(code(name, form))[1] for name in OPS for form in VALUE_FORMS + MASK_FORMS]
    check('decoder names every form (%d)' % len(decoded), all(t.split(' ', 1)[0] in OPS for t in decoded), str(decoded))

    # 2. The table entries are pixel_aot's lowering of the same operations, with the logic-op SCC rule.
    for name in OPS:
        text, scc = G.SCA.get(name, (None, None))
        arity, aot_text, aot_scc = P.SCALAR[name[2:]]
        sibling = G.SCA[OPS[name][0]]
        check('%s: %r SCC %r equals pixel_aot %r SCC %r and the sibling SCC rule'
              % (name, text, scc, aot_text, aot_scc),
              arity == 2 and text == aot_text and aot_scc == P.NONZERO and scc == '$ != 0' and sibling[1] == scc)

    # 3. The value set reaches SCC = 0 and SCC = 1 for every operation.
    for name, (_, reference) in OPS.items():
        results = {reference(a, b) for a, b in PAIRS}
        check('%s: %d pairs give a zero and a nonzero result' % (name, len(PAIRS)), 0 in results and len(results) > 1)

    # 4. Every value form evaluates to the reference: result and SCC.
    for name in OPS:
        found = first_mismatch(name)
        check('%s: %d forms x %d pairs equal the reference (result and SCC)' % (name, len(VALUE_FORMS), len(PAIRS)),
              found is None, str(found))

    # 5. Every numeric form emits the sibling's lines with the operation complemented.
    for name, (sibling, _) in OPS.items():
        wrong = []
        for form in VALUE_FORMS:
            asm, stage = attempt(code(name, form))
            plain_asm, plain = translate(code(sibling, form))
            operands = plain_asm.split(' ', 1)[1].split(', ')
            probe = Probe()
            inner = '%s %s %s' % (probe.u(operands[1]), SIBLING_OPERATOR[sibling], probe.u(operands[2]))
            expected = [line.replace(inner, '~(%s)' % inner, 1) for line in plain.lines]
            if (asm != plain_asm.replace(sibling, name, 1) or stage.lines != expected or expected == plain.lines
                    or stage.masks != plain.masks or stage.exec_known != plain.exec_known):
                wrong.append((asm, stage.lines, plain.lines, stage.masks, plain.masks))
        check('%s: %d numeric forms emit the %s lines complemented, same mask and EXEC bookkeeping'
              % (name, len(VALUE_FORMS), sibling), not wrong, str(wrong[:2]))

    # 5b. Refused: a lane mask next to a plain word, or as the destination (lane_mask_controls has the accepted forms).
    for name in OPS:
        accepted = []
        for form in MASK_FORMS:
            asm, stage = attempt(code(name, form))
            if stage.lines != ['refused: %s on a lane mask' % name]:
                accepted.append((asm, stage.lines))
        stage = Probe()
        stage.masks.add(1)  # s1 holds a compare result
        try:
            stage.one(name, '%s s3, s1, s2' % name)
            accepted.append(('%s s3, s1(mask), s2' % name, stage.lines))
        except G.Unsupported as error:
            if str(error) != '%s on a lane mask' % name:
                accepted.append(('%s s3, s1(mask), s2' % name, [str(error)]))
        check('%s: %d lane-mask forms refused' % (name, len(MASK_FORMS) + 1), not accepted, str(accepted[:2]))

    # Symbolic descriptor sources cannot silently leave the previous SCC in place.
    for name in OPS:
        stage = Probe()
        stage.sym[1] = ('ptr', 0, 0)
        try:
            stage.one(name, '%s s3, s1, s2' % name)
        except G.Unsupported as error:
            check('%s refuses a descriptor before writing SCC' % name,
                  str(error).startswith('scc from descriptor data') and not stage.lines, str(error))
        else:
            check('%s refuses a descriptor before writing SCC' % name, False, str(stage.lines))

    # 6. Mutated table entries, built from the sibling's, must read a wrong word or SCC.
    saved = dict(G.SCA)
    try:
        for name, (sibling, _) in OPS.items():
            G.SCA[name] = saved[sibling]
            found = first_mismatch(name)
            check('%s mutation without the complement fails: %s' % (name, found), found is not None)
            G.SCA[name] = ('~(%s)' % saved[sibling][0], None)
            found = first_mismatch(name)
            check('%s mutation without the SCC write fails: %s' % (name, found), found is not None)
            G.SCA.clear()
            G.SCA.update(saved)
    finally:
        G.SCA.clear()
        G.SCA.update(saved)

    # 7. SOP2 operations that still have no lowering are refused by name.
    missing = [name for _op, (name, _) in sorted(D.SOP2_OPS.items())
               if name not in G.SCA and not hasattr(G.Stage, 'op_' + name)]
    wrong = []
    for name in missing:
        asm = decode_one(sop2(name, 2, 4, 6))[1]
        try:
            Probe().one(name, asm)
            wrong.append((asm, 'accepted'))
        except G.Unsupported as error:
            if str(error) != name:
                wrong.append((asm, str(error)))
    check('%d SOP2 operations without a lowering are refused by name (%s)' % (len(missing), ', '.join(missing)),
          missing and not wrong and not set(OPS) & set(missing), str(wrong))
    lane_mask_controls()
    pipe, expected, mutant = nand_draw_fixture()
    hlsl = pipe.hlsl_ps()
    check('NAND with uniform-bit literals reaches the pixel export',
          all(f's{reg} = (~(' in hlsl for reg in range(24, 28)) and expected != mutant, hlsl)
    got = text_refusal('s_nand_b32 s24, exec_lo, -1', 's_cselect_b32 s25, 1, 0', 's_endpgm')
    check('NAND with a uniform-bit literal still guards a whole-word SCC read',
          got == 'scc from a lane-mask complement in s_cselect_b32', got)
    pack_controls()
    bfe_controls()
    mul_controls()


def all_codes():
    logic = [code(name, form) for name in list(OPS) + [s for s, _ in OPS.values()] for form in VALUE_FORMS + MASK_FORMS]
    return (logic + [code(name, form) for name in (PACK_HH, MUL_HI) for form in VALUE_FORMS]
            + [code(BFE, form) for form in VALUE_FORMS])


def check_binary(llvm_mc):
    codes = all_codes()
    text = ''.join(decode_one(c)[1] + '\n' for c in codes)
    check('llvm-mc assembles %d decoded SOP2 forms to the same words' % len(codes),
          assemble(llvm_mc, text) == b''.join(codes))


def check_decode(library):
    codes = all_codes()
    texts = P.disassemble(b''.join(codes), library)
    ours = [decode_one(c)[1] for c in codes]
    wrong = [(a, b) for a, b in zip(texts, ours) if a != b]
    check('LLVM disassembles %d SOP2 forms as tools/disasm_rdna2.py does' % len(codes),
          len(texts) == len(codes) and not wrong, str(wrong[:3]))


def main():
    if len(sys.argv) == 1:
        controls()
    else:
        mode, *args = sys.argv[1:]
        if mode == 'nand-draw':
            nand_draw(*args)
        elif mode == 'binary':
            check_binary(*args)
        elif mode == 'decode':
            check_decode(*args)
        else:
            raise SystemExit(__doc__)
    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
