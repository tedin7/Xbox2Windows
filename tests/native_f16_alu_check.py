"""v_add_f16, v_sub_f16, v_subrev_f16 and v_mul_f16 in the direct C++ compute path.

Each source contributes its low 16 bits. abs clears bit 15 of that half, then neg toggles it.
The operation runs in f32 and rounds to binary16 the way v_cvt_f16_f32 does. The result is the low
16 bits and the destination's high half keeps its value (LLVM GCNSubtarget::zeroesHigh16BitsOfDest:
"On gfx10, all 16-bit instructions preserve the high bits"). Every destination holds HI_A in its high
half before the operation, so each stored word is HI_A | result. The reference is half_op in f16_packed_model.py (sub is an add of
the negated second source, subrev swaps the sources). NaN payloads are compared by class.

  native_f16_alu_check.py model
      no dumper and no GPU: the case table, the f32 rounding cross-check, and the two mutants
  native_f16_alu_check.py ir DUMPER
      build one compute container, dump its DXIL text, evaluate it, and require the named refusals
  native_f16_alu_check.py gpu STORE_REPLAY SDK
      compute readback of the same words and the DPP16 cases (exit 77 without wave32)
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import f16_packed_model as M
from native_compute_clear_check import write_request
from native_direct_container_check import pack, words
from native_scalar_lds_check import scalar, sop1, vmove

GROUPS = 2
WORDS = 2064
SENTINEL = 0x69696969  # native_store_replay fills untouched words with 0x69
BASE = 8
STRIDE = 128
SHIFT = 9
assert 4 * STRIDE == 1 << SHIFT

OPS = ('add', 'sub', 'subrev', 'mul')
OP_CODE = {'add': 0x32, 'sub': 0x33, 'subrev': 0x34, 'mul': 0x35}
# Binary16 rounding of the nine inline float encodings, in encoding order.
INLINE_F16 = {
    240: 0x3800, 241: 0xb800, 242: 0x3c00, 243: 0xbc00, 244: 0x4000,
    245: 0xc000, 246: 0x4400, 247: 0xc400, 248: 0x3118,
}
PAIRS = [
    (0x3c00, 0x4000), (0x0000, 0x8000), (0x8000, 0x8000), (0x0001, 0x0000),
    (0x0001, 0x0001), (0x7c00, 0x3c00), (0x7c00, 0xfc00), (0x7e00, 0x3c00),
    (0x7bff, 0x7bff), (0x3c00, 0x1000), (0x3c01, 0x1000), (0x7c01, 0x0001),
]
HI_A = 0xABCD0000
HI_B = 0x789A0000
DPP_EXEC = 0xFFFFFEFD  # Lanes 1 and 8 are inactive during the DPP instructions.
DPP_OLD = HI_A | 0x3555
DPP_CASES = (
    # op, DPP_CTRL, FI, BOUND_CTRL, row mask, bank mask, src1 low half
    ('add', 0xB1, True, False, 0xF, 0xF, 0x1000),  # quad_perm:[1,0,3,2], ties to even
    ('mul', 0x111, False, False, 0xF, 0xF, 0x3C01),  # row_shr:1, preserve at row edges
    ('sub', 0xB1, True, False, 0x5, 0xF, 0x3C01),
    ('subrev', 0xB1, False, True, 0xF, 0x5, 0x3C01),
)


def binary16(op, a, b):
    """Low 16 bits of one RDNA 2 operation. sub negates the second source; subrev swaps."""
    a &= 0xffff
    b &= 0xffff
    if op == 'sub':
        return M.half_op('add', a, b ^ 0x8000)
    if op == 'subrev':
        return M.half_op('add', b, a ^ 0x8000)
    return M.half_op(op, a, b)


def modified(bits, abs_mask, neg, index):
    if abs_mask & (1 << index):
        bits &= 0x7fff
    if neg & (1 << index):
        bits ^= 0x8000
    return bits & 0xffff


def same_half(got, want):
    """Bit equality of two stored words: the high half must be the kept HI_A, the low halves equal. Any NaN
    matches any NaN."""
    if got >> 16 != HI_A >> 16 or want >> 16 not in (0, HI_A >> 16):
        return False
    return same_binary16(got & 0xffff, want & 0xffff)


def same_binary16(got, want):
    """Bit equality of two binary16 values (16 bits each). Any NaN matches any NaN."""
    if got >> 16 or want >> 16:
        return False
    if M.is_nan(got) or M.is_nan(want):
        return M.is_nan(got) and M.is_nan(want)
    return got == want


def _u32(value):
    return value & 0xffffffff


def packed_half_rne(bits):
    """The integer round used by packed_half_rne in the direct compute lowering."""
    bits = _u32(bits)
    sign = (bits >> 16) & 32768
    mag = bits & 0x7fffffff
    exp = mag >> 23
    frac = mag & 0x7fffff

    def round_shift(value, shift, half):
        odd = _u32(value >> shift) & 1
        bias = _u32(_u32(half - 1) + odd)
        return _u32(_u32(value + bias) >> shift)

    normal = round_shift(_u32(mag - 939524096), 13, 4096)
    sub_range = 102 <= exp <= 112
    shift = _u32(126 - exp) if sub_range else 14
    half = _u32(1 << _u32(shift - 1))
    sub = round_shift(frac | 8388608, shift, half)
    if exp <= 112:
        result = sub if sub_range else 0
    else:
        result = normal
    if exp > 142:
        result = 31744
    if mag > 2139095040:
        result = (frac >> 13) | 32256
    return (result | sign) & 65535


def f32_half(op, a, b):
    """f32 operation of two exact f16 values, then the integer binary16 round."""
    left, right = M.h2f(a), M.h2f(b)
    value = {'add': left + right, 'sub': left - right, 'subrev': right - left, 'mul': left * right}[op]
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return packed_half_rne(bits)


def vop2(op, dst, src0, src1):
    return [(op << 25) | ((dst & 255) << 17) | ((src1 & 255) << 9) | (src0 & 511)]


def vop3(op, dst, src0, src1, abs_mask=0, neg=0, clamp=0, omod=0, opsel=0):
    word = (0xd4000000 | ((0x100 + op) << 16) | ((abs_mask & 7) << 8) | ((opsel & 15) << 11) |
            ((clamp & 1) << 15) | (dst & 255))
    other = (src0 & 511) | ((src1 & 511) << 9) | ((omod & 3) << 27) | ((neg & 7) << 29)
    return [word, other]


def mov(reg, value):
    return [0x7e0002ff | (reg << 17), value & 0xffffffff]


def smov(reg, value):
    return [0xbe8003ff | (reg << 16), value & 0xffffffff]


def store(byte_off, data):
    return [0xe0701000 | (byte_off & 4095), 0x80080008 | ((data & 255) << 8)]


class Case:
    def __init__(self, op, a, b, kind='vop2', abs_mask=0, neg=0, inline_src=242):
        self.op = op
        self.kind = kind
        self.abs_mask = abs_mask
        self.neg = neg
        self.inline_src = inline_src
        self.dest = 3
        if kind == 'inline':
            a = INLINE_F16[inline_src]
        elif kind == 'inline_int':
            a = (192 - 193) & 0xffff
        self.a = a & 0xffff
        self.b = b & 0xffff
        self.want = binary16(op, modified(self.a, abs_mask, neg, 0), modified(self.b, abs_mask, neg, 1))

    def emit(self):
        a_word, b_word = HI_A | self.a, HI_B | self.b
        op = OP_CODE[self.op]
        if self.kind == 'overlap':
            self.dest = 1
            return mov(1, a_word) + mov(2, b_word) + vop2(op, 1, 257, 2)
        if self.kind == 'issue':
            self.dest = 0
            word = vop2(0x32, 0, 257, 2)
            assert word == [0x64000501]
            return mov(1, a_word) + mov(2, b_word) + mov(0, HI_A) + word
        if self.kind == 'literal':
            return mov(2, b_word) + mov(3, HI_A) + vop2(op, 3, 255, 2) + [a_word]
        if self.kind == 'inline':
            return mov(2, b_word) + mov(3, HI_A) + vop2(op, 3, self.inline_src, 2)
        if self.kind == 'inline_int':
            return mov(2, b_word) + mov(3, HI_A) + vop2(op, 3, 193, 2)
        if self.kind == 'sgpr':
            return mov(2, b_word) + mov(3, HI_A) + smov(8, a_word) + vop2(op, 3, 8, 2)
        if self.kind == 'vop3':
            return mov(1, a_word) + mov(2, b_word) + mov(3, HI_A) + vop3(op, 3, 257, 258, self.abs_mask, self.neg)
        return mov(1, a_word) + mov(2, b_word) + mov(3, HI_A) + vop2(op, 3, 257, 2)


def all_cases():
    items = [Case(op, a, b) for a, b in PAIRS for op in OPS]
    for abs_mask, neg in ((1, 0), (0, 2), (3, 1), (3, 3)):
        items.extend(Case(op, 0xbc00, 0x4000, 'vop3', abs_mask, neg) for op in OPS)
    items.append(Case('add', 0x3c00, 0x4000, 'vop3', abs_mask=4))
    for op in OPS:
        items.append(Case(op, 0, 0x4000, 'inline'))
        items.append(Case(op, 0x3c01, 0x3c00, 'literal'))
        items.append(Case(op, 0x3c00, 0x4000, 'sgpr'))
    items.append(Case('add', 0, 0x3c00, 'inline', inline_src=248))
    items.append(Case('add', 0, 0, 'inline_int'))
    items.append(Case('add', 0x3c00, 0x4000, 'overlap'))
    items.append(Case('subrev', 0x3c00, 0x4000, 'overlap'))
    items.append(Case('add', 0x3c00, 0x4000, 'issue'))
    assert len(items) < STRIDE
    return items


def preamble():
    """s4 is the descriptor table and s5 is the group id."""
    return [sop1(3, 20, 4), sop1(3, 21, 128), 0xf408080a, 0xfa000020,
            scalar(0x1e, 7, 5, 128 + SHIFT), vmove(8, 7), sop1(3, 126, 129),
            0xbfa40000, 0xbfa40010, 0xbfa5000f, 0xba800881, 0]


def program(items):
    code = preamble()
    want = []
    for case in items:
        code += case.emit()
        code += store(4 * len(want), case.dest)
        want.append(case.want)
    code.append(0xbf810000)
    return code, want


def dpp_program():
    assert STRIDE == 32 * len(DPP_CASES)
    code = preamble()
    code += [sop1(3, 126, 193)]  # the preamble leaves EXEC at lane 0; set up every lane
    code += vop2(0x1A, 8, 132, 0) + vop2(0x25, 8, 7, 8)  # 16 * lane + group byte base
    code += mov(1, HI_B | 0x3c00) + vop2(0x1C, 1, 256, 1)  # lane-distinct src0
    for slot, (op, pattern, fi, bound_zero, rows, banks, src1) in enumerate(DPP_CASES):
        control = 1 | pattern << 8 | fi << 18 | bound_zero << 19 | banks << 24 | rows << 28
        code += mov(2, HI_B | src1) + mov(3, DPP_OLD)
        code += [sop1(3, 126, 255), DPP_EXEC, *vop2(OP_CODE[op], 3, 250, 2), control]
        # Store under full EXEC to observe both lane masks and preserved destinations.
        code += [sop1(3, 126, 193), *store(4 * slot, 3)]
    return code + [0xbf810000]


def dpp_reference():
    results = []
    for lane in range(32):
        for op, _, fi, bound_zero, rows, banks, src1 in DPP_CASES:
            peer = lane - 1 if op == 'mul' else lane ^ 1
            in_row = peer // 16 == lane // 16
            usable = in_row and (fi or DPP_EXEC & (1 << peer))
            row, bank = lane // 16, lane // 4 % 4
            enabled = DPP_EXEC & (1 << lane) and rows & (1 << row) and banks & (1 << bank)
            if enabled and (usable or bound_zero):
                results.append(HI_A | binary16(op, 0x3c00 + peer if usable else 0, src1))
            else:
                results.append(DPP_OLD)
    return results


def swapped_want(case):
    op = {'sub': 'subrev', 'subrev': 'sub'}.get(case.op, case.op)
    return binary16(op, modified(case.a, case.abs_mask, case.neg, 0), modified(case.b, case.abs_mask, case.neg, 1))


def reject_mutants(items):
    """The comparator must fail a sub/subrev swap and a result whose high half was zeroed."""
    wants = [case.want for case in items]
    swapped = [swapped_want(case) for case in items]
    assert any(left != right for left, right in zip(wants, swapped))
    assert not all(same_half(HI_A | got, want) for got, want in zip(swapped, wants))
    assert not any(same_half(got, want) for got, want in zip(wants, wants))


def check_model(items):
    assert binary16('add', 0x3c00, 0x1000) == 0x3c00
    assert binary16('add', 0x3c01, 0x1000) == 0x3c02
    assert binary16('add', 0x7bff, 0x7bff) == 0x7c00
    assert binary16('add', 0x8000, 0x8000) == 0x8000
    assert binary16('add', 0x8000, 0x0000) == 0x0000
    assert binary16('add', 0x0001, 0x0001) == 0x0002
    assert binary16('mul', 0x0001, 0x8001) == 0x8000
    assert binary16('sub', 0x4200, 0x3c00) == 0x4000
    assert binary16('subrev', 0x3c00, 0x4200) == 0x4000
    assert binary16('sub', 0x3c00, 0x4000) != binary16('subrev', 0x3c00, 0x4000)
    for case in items:
        left = modified(case.a, case.abs_mask, case.neg, 0)
        right = modified(case.b, case.abs_mask, case.neg, 1)
        got = f32_half(case.op, left, right)
        assert same_half(HI_A | got, case.want), (case.op, case.kind, hex(left), hex(right), hex(got), hex(case.want))
    wants = [case.want for case in items]
    assert 0x8000 in wants and 0x0000 in wants and 0x7c00 in wants and 0x3c00 in wants and 0x3c02 in wants
    assert any(M.is_nan(value) for value in wants)
    assert any(0 < (value & 0x7fff) < 0x0400 for value in wants)
    reject_mutants(items)


def refusal_programs():
    add = vop2(0x32, 3, 257, 2)
    programs = []
    programs.append((vop2(0x32, 3, 233, 2) + [0xfac68801, 0xbf810000], 'SDWA or DPP vector operands'))
    # Clamp needs a wide encoding. A VOP3 with clamp and the DPP16 source selector is invalid.
    programs.append((vop3(0x32, 3, 250, 258, clamp=1) + [0xbf810000], 'unsupported v_add_f16 clamp or omod'))
    programs.append((vop3(0x35, 3, 257, 258, clamp=1) + [0xbf810000], 'unsupported v_mul_f16 clamp or omod'))
    programs.append((vop3(0x33, 3, 257, 258, omod=1) + [0xbf810000], 'unsupported v_sub_f16 clamp or omod'))
    programs.append((vop3(0x34, 3, 257, 258, opsel=1) + [0xbf810000], 'unsupported v_subrev_f16 high-half op_sel'))
    programs.append((vop3(0x32, 3, 257, 258, opsel=8) + [0xbf810000], 'unsupported v_add_f16 high-half op_sel'))
    programs.append(([0xbfa4000c] + add + [0xbf810000], 'unsupported binary16 rounding or denormal mode'))
    programs.append(([0xbfa50000] + add + [0xbf810000], 'unsupported binary16 rounding or denormal mode'))
    programs.append(([0xba8005c1, 1] + add + [0xbf810000], 'unsupported binary16 rounding or denormal mode'))
    programs.append(([0xb98005c1] + add + [0xbf810000], 'unsupported binary16 rounding or denormal mode'))
    programs.append(([0xbfa40001] + add + [0xbf810000], 'unsupported compute control instruction s_round_mode'))
    programs.append(([0xbfa5000c] + add + [0xbf810000], 'unsupported compute control instruction s_denorm_mode'))
    programs.append((vop2(0x3c, 3, 257, 2) + [0xbf810000], 'direct DXIL lowering for v_pk_fmac_f16'))
    programs.append((vop2(0x3a, 3, 257, 2) + [0xbf810000], 'direct DXIL lowering for v_min_f16'))
    return programs


def compute_container(code_bytes):
    """One compute stage. The descriptor table is s4 and the group id is s5, matching the .c67 preamble."""
    body = bytearray(256)
    struct.pack_into('<Q', body, 0, 6)
    for offset in (8, 16, 24, 32, 40, 48, 72):
        struct.pack_into('<Q', body, offset, 0xDEADBEEFDEADBEEF)
    record = 112
    struct.pack_into('<Q', body, 96, record)
    struct.pack_into('<Q', body, record + 40, len(body))
    struct.pack_into('<I', body, record + 56, 1)
    body.extend(words(3, 0, 1, 0, 4))
    struct.pack_into('<QQ', body, record, len(body), len(code_bytes))
    struct.pack_into('<I', body, record + 20, 32)
    struct.pack_into('<II', body, record + 76, 512, 4)
    struct.pack_into('<IIII', body, record + 120, 0x8a, 32, 1, 1)
    body.extend(code_bytes)
    return pack(body)


def _i32(token, values):
    if token.startswith('%'):
        kind, bits = values[token]
        if kind != 'i32':
            raise AssertionError((token, kind, bits))
        return bits & 0xffffffff
    if token == 'undef':
        return 0
    return int(token, 0) & 0xffffffff


def _i1(token, values):
    if token == 'true':
        return 1
    if token == 'false':
        return 0
    kind, bits = values[token]
    if kind != 'i1':
        raise AssertionError((token, kind))
    return bits


def _signed(bits):
    return bits - 0x100000000 if bits & 0x80000000 else bits


def _icmp(pred, left, right):
    sl, sr = _signed(left), _signed(right)
    return {
        'eq': left == right, 'ne': left != right,
        'ugt': left > right, 'uge': left >= right, 'ult': left < right, 'ule': left <= right,
        'sgt': sl > sr, 'sge': sl >= sr, 'slt': sl < sr, 'sle': sl <= sr,
    }[pred]


def _f32(bits):
    return struct.unpack('<f', struct.pack('<I', bits & 0xffffffff))[0]


def _fbin(op, left, right):
    value = {'fadd': _f32(left) + _f32(right), 'fsub': _f32(left) - _f32(right),
             'fmul': _f32(left) * _f32(right)}[op]
    return struct.unpack('<I', struct.pack('<f', value))[0]


def _widen(bits):
    return struct.unpack('<I', struct.pack('<f', struct.unpack('<e', struct.pack('<H', bits & 0xffff))[0]))[0]


def _branch(line, values):
    match = re.fullmatch(r'br label %(\S+)', line)
    if match:
        return match.group(1)
    match = re.fullmatch(r'br i1 (\S+), label %(\S+), label %(\S+)', line)
    if not match:
        raise AssertionError(line)
    return match.group(2) if _i1(match.group(1), values) else match.group(3)


def _exec(line, values, memory, stores):
    if line.startswith('store '):
        match = re.fullmatch(r'store i32 (\S+), i32\* (\S+), align 4', line)
        if not match:
            raise AssertionError(line)
        memory[match.group(2)] = _i32(match.group(1), values)
        return
    if 'rawBufferStore.i32' in line:
        match = re.search(r'rawBufferStore\.i32\(i32 140, %dx\.types\.Handle \S+, i32 (\S+), i32 \S+, i32 (\S+),', line)
        if not match:
            raise AssertionError(line)
        stores.append((_i32(match.group(1), values), _i32(match.group(2), values)))
        return
    if line.startswith('call '):
        raise AssertionError(line)
    match = re.fullmatch(r'(%\w+) = (.+)', line)
    if not match:
        raise AssertionError(line)
    name, rhs = match.group(1), match.group(2)
    if rhs.startswith('alloca '):
        memory[name] = None
        values[name] = ('ptr', 0)
        return
    loaded = re.fullmatch(r'load i32, i32\* (%\w+), align 4', rhs)
    if loaded:
        # A destination keeps its previous bits on an inactive lane. Lane 0 is active
        # for every store in this fixture, so an unread register contributes nothing.
        values[name] = ('i32', 0 if memory.get(loaded.group(1)) is None else memory[loaded.group(1)])
        return
    compared = re.fullmatch(r'icmp (\w+) i32 (\S+), (\S+)', rhs)
    if compared:
        values[name] = ('i1', int(_icmp(compared.group(1), _i32(compared.group(2), values), _i32(compared.group(3), values))))
        return
    selected = re.fullmatch(r'select i1 (\S+), i32 (\S+), i32 (\S+)', rhs)
    if selected:
        taken = _i32(selected.group(2), values) if _i1(selected.group(1), values) else _i32(selected.group(3), values)
        values[name] = ('i32', taken)
        return
    binary = re.fullmatch(r'(add|sub|mul|and|or|xor|shl|lshr|ashr|urem|udiv) i32 (\S+), (\S+)', rhs)
    if binary:
        op, left, right = binary.group(1), _i32(binary.group(2), values), _i32(binary.group(3), values)
        if op == 'shl':
            result = left << (right & 31)
        elif op == 'lshr':
            result = left >> (right & 31)
        elif op == 'ashr':
            result = _signed(left) >> (right & 31)
        elif op in ('urem', 'udiv'):
            result = 0 if right == 0 else (left % right if op == 'urem' else left // right)
        else:
            result = {'add': left + right, 'sub': left - right, 'mul': left * right,
                      'and': left & right, 'or': left | right, 'xor': left ^ right}[op]
        values[name] = ('i32', result & 0xffffffff)
        return
    logic = re.fullmatch(r'(and|or|xor) i1 (\S+), (\S+)', rhs)
    if logic:
        left, right = _i1(logic.group(2), values), _i1(logic.group(3), values)
        result = {'and': left & right, 'or': left | right, 'xor': left ^ right}[logic.group(1)]
        values[name] = ('i1', result)
        return
    extended = re.fullmatch(r'zext i1 (\S+) to i32', rhs)
    if extended:
        values[name] = ('i32', _i1(extended.group(1), values))
        return
    if rhs.startswith('bitcast i32 '):
        token = rhs.removeprefix('bitcast i32 ').removesuffix(' to float')
        values[name] = ('float', _i32(token, values))
        return
    if rhs.startswith('bitcast float '):
        token = rhs.removeprefix('bitcast float ').removesuffix(' to i32')
        kind, bits = values[token]
        if kind != 'float':
            raise AssertionError(rhs)
        values[name] = ('i32', bits)
        return
    floated = re.fullmatch(r'(fadd|fsub|fmul) float (\S+), (\S+)', rhs)
    if floated:
        def bits_of(token):
            if token.startswith('%'):
                kind, bits = values[token]
                if kind != 'float':
                    raise AssertionError(rhs)
                return bits
            return struct.unpack('<I', struct.pack('<f', float(token)))[0]
        values[name] = ('float', _fbin(floated.group(1), bits_of(floated.group(2)), bits_of(floated.group(3))))
        return
    extracted = re.fullmatch(r'extractvalue %dx\.types\.fouri32 (\S+), (\d+)', rhs)
    if extracted:
        kind, _ = values[extracted.group(1)]
        if kind != 'ballot' or extracted.group(2) != '0':
            raise AssertionError(rhs)
        values[name] = ('i32', 0xffffffff)
        return
    if rhs.startswith('call '):
        if 'waveGetLaneIndex' in rhs or 'groupId.i32' in rhs or 'flattenedThreadIdInGroup' in rhs:
            values[name] = ('i32', 0)
        elif 'waveActiveBallot' in rhs:
            values[name] = ('ballot', 0)
        elif 'legacyF16ToF32' in rhs:
            argument = re.search(r'i32 (\S+)\)$', rhs)
            values[name] = ('float', _widen(_i32(argument.group(1), values)))
        elif 'unaryBits.i32(i32 33,' in rhs:
            argument = re.search(r'i32 33,\s*i32 ([^,\s\)]+)\)', rhs)
            x = _i32(argument.group(1), values) & 0xffffffff
            values[name] = ('i32', (32 - x.bit_length()) if x else 0xffffffff)
        elif 'createHandleFromBinding' in rhs or 'annotateHandle' in rhs:
            values[name] = ('handle', 0)
        else:
            raise AssertionError(rhs)
        return
    raise AssertionError(line)


def interpret(ir):
    """Lane 0 of a straight-line compute module. Returns (byte address, stored word) pairs."""
    body = ir.split('define void @main()', 1)[1].split('\n}', 1)[0]
    blocks, label, buf = {}, None, []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line == '{':
            continue
        if line.endswith(':') and ' ' not in line:
            if label is not None:
                blocks[label] = buf
            label, buf = line[:-1], []
            continue
        if label is None:
            raise AssertionError(line)
        buf.append(line)
    if label is not None:
        blocks[label] = buf
    values, memory, stores = {}, {}, []
    label, steps = 'entry', 0
    while steps < 200000:
        steps += 1
        followed = False
        for line in blocks[label]:
            if line.startswith('br '):
                label = _branch(line, values)
                followed = True
                break
            if line == 'ret void':
                return stores
            _exec(line, values, memory, stores)
        if not followed:
            raise AssertionError('block fell through: ' + label)
    raise AssertionError('step limit')


def dump(dumper, work, code, name):
    path = work / (name + '.cdx')
    path.write_bytes(compute_container(words(*code)))
    return subprocess.run([dumper, str(path)], capture_output=True, text=True)


def run_model():
    items = all_cases()
    check_model(items)
    code, want = program(items)
    assert vop2(0x32, 0, 257, 2) == [0x64000501]
    assert len(want) == len(items) and len(code) > len(want)
    print('native_f16_alu model: %d results match f32-then-round; '
          'a sub/subrev swap and a zeroed upper half are rejected' % len(items))


def run_ir(dumper):
    items = all_cases()
    check_model(items)
    code, want = program(items)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = dump(dumper, work, code, 'alu')
        assert run.returncode == 0, run.stderr
        text = run.stdout
        assert 'legacyF16ToF32' in text and 'legacyF32ToF16' not in text, 'rounding opcode'
        assert 'fadd float' in text and 'fsub float' in text and 'fmul float' in text
        assert '939524096' in text, 'integer nearest-even rounding'
        stores = sorted(interpret(text))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, hex(got), hex(expected)) for i, ((_, got), expected) in enumerate(zip(stores, want))
               if not same_half(got, expected)]
        assert not bad, bad[:8]
        run = dump(dumper, work, dpp_program(), 'dpp')
        assert run.returncode == 0 and 'waveReadLaneAt.i32' in run.stdout, (run.stdout, run.stderr)
        for refused, why in refusal_programs():
            run = dump(dumper, work, refused, 'refused')
            assert run.returncode != 0 and why in run.stderr, (why, run.returncode, run.stderr)
    print('native_f16_alu IR: %d results match the model; %d refusals carry the named diagnostic' %
          (len(want), len(refusal_programs())))


def group_offsets(count):
    offsets = [[BASE + group * STRIDE + i for i in range(count)] for group in range(GROUPS)]
    flat = [index for group in offsets for index in group]
    assert len(set(flat)) == len(flat)
    assert 0 <= min(flat) and max(flat) < WORDS
    return offsets


def run_gpu(gpu, sdk):
    code, want = program(all_cases())
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        src, root = work / 'root.hlsl', work / 'root.bin'
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([str(Path(sdk) / 'dxc.exe'), '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, src], check=True)
        sa = struct.pack('<7I', 0, 0, 0, 32 | (1 << 16), 1 | (1 << 20), 0, 0x80 | (5 << 1))
        for name, instructions, reference in (('alu', code, want), ('dpp', dpp_program(), dpp_reference())):
            marker = write_request(work, instructions, root.read_bytes(), sa)
            dest = work / (name + '-readback.bin')
            run = subprocess.run([gpu, sdk, str(marker), str(work / 'cache'), '0', str(dest), '--direct'])
            if run.returncode == 77:
                sys.exit(77)
            run.check_returncode()
            results = {slot: value for offsets in group_offsets(len(reference))
                       for slot, value in zip(offsets, reference)}
            actual = struct.unpack('<2064I', dest.read_bytes())
            bad = []
            for index, got in enumerate(actual):
                model = results.get(index, SENTINEL)
                # Compare NaNs by class, keep the destination high half, and check the guard words.
                ok = same_half(got, model) if index in results else got == model
                if not ok:
                    bad.append((index, hex(got), hex(model)))
            assert not bad, (name, bad[:10])
            print('native_f16_alu GPU %s: %d words matched the model, high half kept' % (name, len(results)))


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'model':
        run_model()
    elif len(sys.argv) == 3 and sys.argv[1] == 'ir':
        run_ir(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == 'gpu':
        run_gpu(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
