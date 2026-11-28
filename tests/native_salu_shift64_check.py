"""Direct compute 64-bit scalar shift (s_lshl_b64, s_lshr_b64, s_ashr_i64) controls and readback (#3062)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
from native_scalar_lds_check import scalar, sop1, vmove


LSHL64, LSHR64, ASHR64, CSELECT32 = 0x1F, 0x21, 0x23, 0x0A
ZERO, ONE, NEG_ONE, NEG_SIXTEEN, SIXTY_FOUR, VCC, EXEC, FLOAT_ONE, LITERAL = 128, 129, 193, 208, 192, 106, 126, 242, 255
PAIR = (0x89ABCDEF, 0x81234567)
LITERAL_VALUE = 0x80000001
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
LANES = 64
PLANE_BYTES = LANES * 8
GUARD_WORDS = 8
OUTPUT_WORDS = 2064
SENTINEL = 0x69696969


def set_pair(words):
    return [sop1(3, 22, LITERAL), words[0], sop1(3, 23, LITERAL), words[1]]


def set_scc(value):
    # s_cmp_eq_u32 0, (0 for true, 1 for false)
    return 0xBF060080 | (ZERO if value else ONE) << 8


def compute_program(instructions, dst):
    # Two 32-lane groups, with 8 bytes per lane and 256 bytes per group in each plane.
    program = [sop1(3, 20, 4), sop1(3, 21, ZERO),
               0xF4080000 | 32 << 6 | 10, 0xFA000020,
               scalar(0x1E, 7, 5, 136), vop2(0x1A, 8, 131, 0), vop2(0x25, 8, 7, 8)]
    program += instructions
    program += [vmove(2, dst), vmove(3, dst + 1), 0xE0741000, 0x80080208,
                scalar(CSELECT32, 26, ONE, ZERO), vmove(2, 26), vmove(3, ZERO),
                0xE0741000 | PLANE_BYTES, 0x80080208, 0xBF810000]
    return program


def eval_shift(op, lo, hi, src_kind, amount, *, logical_ashr=False, zero_ext_neg=False,
               zero_ext_lit=False, sign_ext_lit=False, mask_amount=True):
    shift = amount & 63 if mask_amount else amount
    if src_kind == 'neg_inline':
        hi_word = 0 if zero_ext_neg else MASK32
    elif src_kind == 'pos_inline':
        hi_word = 0
    elif src_kind == 'literal':
        if zero_ext_lit:
            hi_word = 0
        elif (op == ASHR64 or sign_ext_lit) and (lo >> 31):
            hi_word = MASK32
        else:
            hi_word = 0
    else:
        hi_word = hi
    val = (hi_word << 32) | (lo & MASK32)
    if op == LSHL64:
        res = (val << shift) & MASK64
    elif op == LSHR64 or logical_ashr:
        res = val >> shift
    else:
        signed_val = val - (1 << 64) if (val >> 63) else val
        res = (signed_val >> shift) & MASK64
    return (res & MASK32, (res >> 32) & MASK32)


def cases():
    # SGPR pair across boundary amounts and odd SGPR wrapped shift amount (65 & 63 == 1).
    for op, opname in ((LSHL64, 'lshl'), (LSHR64, 'lshr'), (ASHR64, 'ashr')):
        for amount in (0, 1, 31, 32, 33, 63):
            code = set_pair(PAIR) + [set_scc(0), scalar(op, 24, 22, ZERO + amount)]
            yield (f'{opname} pair,{amount}', code, 24, op, PAIR[0], PAIR[1], 'pair', amount)
        code = set_pair(PAIR) + [sop1(3, 27, LITERAL), 65, set_scc(0), scalar(op, 24, 22, 27)]
        yield (f'{opname} pair,s27(65)', code, 24, op, PAIR[0], PAIR[1], 'pair', 65)

    # Inline integers: negative (-1, -16) and positive (1, 64).
    for op, opname in ((LSHL64, 'lshl'), (LSHR64, 'lshr'), (ASHR64, 'ashr')):
        for label, enc, lo, hi, src_kind in (('neg1', NEG_ONE, MASK32, MASK32, 'neg_inline'),
                                             ('neg16', NEG_SIXTEEN, 0xFFFFFFF0, MASK32, 'neg_inline'),
                                             ('pos1', ONE, 1, 0, 'pos_inline'),
                                             ('pos64', SIXTY_FOUR, 64, 0, 'pos_inline')):
            code = [set_scc(0), scalar(op, 24, enc, ZERO + 5)]
            yield (f'{opname} {label},5', code, 24, op, lo, hi, src_kind, 5)

    # 32-bit literal 0x80000001: zero-extended for logical shifts, sign-extended for s_ashr_i64.
    for op, opname in ((LSHR64, 'lshr'), (ASHR64, 'ashr')):
        code = [set_scc(0), scalar(op, 24, LITERAL, ZERO + 4), LITERAL_VALUE]
        yield (f'{opname} lit,4', code, 24, op, LITERAL_VALUE, 0, 'literal', 4)

    # Overlapping in-place destinations: s[22:23] written after reading s[22:23] and s22/s23.
    code = set_pair((0x12345678, 0x9ABCDEF0)) + [set_scc(0), scalar(LSHR64, 22, 22, 22)]
    yield ('lshr overlap s22', code, 22, LSHR64, 0x12345678, 0x9ABCDEF0, 'pair', 0x12345678)
    code = set_pair((0x12345678, 0x9ABCDEF0)) + [set_scc(0), scalar(LSHL64, 22, 22, 23)]
    yield ('lshl overlap s23', code, 22, LSHL64, 0x12345678, 0x9ABCDEF0, 'pair', 0x9ABCDEF0)

    # High-word-only nonzero into VCC (SCC=1), and zero result clearing primed SCC=1.
    yield ('lshl vcc high only', [set_scc(0), scalar(LSHL64, VCC, ONE, ZERO + 32)], VCC, LSHL64, 1, 0, 'pos_inline', 32)
    yield ('lshr zero clears scc', [set_scc(1), scalar(LSHR64, 24, ONE, ONE)], 24, LSHR64, 1, 0, 'pos_inline', 1)


def full_reference(*, scc_low_only=False, **mutant_flags):
    out = []
    for _, _, _, op, lo, hi, src_kind, amount in cases():
        words = eval_shift(op, lo, hi, src_kind, amount, **mutant_flags)
        scc = int(words[0] != 0) if scc_low_only else int(any(words))
        out.append((words, scc))
    return out


def controls(cli, sdk, work, root):
    # Compile representative program covering all three opcodes and verify DXIL 64-bit shifts.
    ir_program = compute_program(
        set_pair(PAIR) + [
            scalar(LSHL64, 24, 22, ZERO + 5),
            scalar(LSHR64, 24, 24, ONE),
            scalar(ASHR64, 24, 24, ONE),
        ], 24)
    marker = request(work, root, ir_program)
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    for insn in ('shl i64', 'lshr i64', 'ashr i64'):
        assert insn in dump, (insn, dump)

    # Symbolic-descriptor overwrite regression for s_ashr_i64: s_load_dwordx2 s[24:25], s[20:21], 0x20
    # tags both s24 and s25 as descriptor; s_ashr_i64 must clear both symbolic tags so vmove(3, 25) succeeds.
    sym_prog = compute_program([0xF4040000 | 24 << 6 | 10, 0xFA000020, scalar(ASHR64, 24, NEG_ONE, ONE)], 24)
    sym_marker = request(work, root, sym_prog)
    sym_out = sym_marker.with_suffix('.sym.cso')
    sym_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', sym_marker, sym_out],
                             capture_output=True, text=True)
    assert sym_res.returncode == 0 and sym_out.is_file(), (sym_res.stdout, sym_res.stderr)

    # Refused operand/destination controls.
    refused = (
        ([scalar(LSHL64, EXEC, 22, ONE), 0xBF810000], 'invalid s_lshl_b64 destination'),
        ([scalar(LSHR64, 25, 22, ONE), 0xBF810000], 'invalid s_lshr_b64 destination'),
        ([scalar(ASHR64, 24, 23, ONE), 0xBF810000], 'invalid s_ashr_i64 source'),
        ([scalar(LSHL64, 24, VCC, ONE), 0xBF810000], 'invalid s_lshl_b64 source'),
        ([scalar(LSHR64, 24, FLOAT_ONE, ONE), 0xBF810000], 'invalid s_lshr_b64 source'),
        ([scalar(ASHR64, 24, 22, FLOAT_ONE), 0xBF810000], 'invalid s_ashr_i64 source'),
        ([scalar(LSHL64, 24, 22, VCC), 0xBF810000], 'invalid s_lshl_b64 source'),
    )
    for n, (words, error) in enumerate(refused):
        m = request(work, root, words)
        o = m.with_suffix(f'.refused{n}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o],
                           capture_output=True, text=True)
        assert r.returncode and error in r.stderr, (error, r.stdout, r.stderr)
        assert not o.exists(), o

    # Mutants must each alter the reference expectations.
    baseline = full_reference()
    for label, flags in (
        ('logical ashr', {'logical_ashr': True}),
        ('zero-extended negative inline', {'zero_ext_neg': True}),
        ('zero-extended literal for ashr', {'zero_ext_lit': True}),
        ('sign-extended literal for lshr', {'sign_ext_lit': True}),
        ('SCC from low word only', {'scc_low_only': True}),
        ('unmasked shift amount', {'mask_amount': False}),
    ):
        assert full_reference(**flags) != baseline, label

    print('compiled s_lshl_b64, s_lshr_b64 and s_ashr_i64 (including descriptor overwrite); '
          '7 invalid operand/destination forms refused; 6 reference mutants differ')
    return 0


def hardware(replay, sdk, work, root):
    for name, code, dst, op, lo, hi, src_kind, amount in cases():
        words = eval_shift(op, lo, hi, src_kind, amount)
        scc = int(any(words))
        marker = request(work, root, compute_program(code, dst))
        output = work / 'out.bin'
        result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        actual = struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes())
        end = GUARD_WORDS + LANES * 4
        assert actual[:GUARD_WORDS] == (SENTINEL,) * GUARD_WORDS, name
        assert actual[GUARD_WORDS:end] == words * LANES + (scc, 0) * LANES, (name, words, scc)
        assert actual[end:] == (SENTINEL,) * (OUTPUT_WORDS - end), name
        print(f'{name}: result words, SCC and guards matched')
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG',
                        '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
