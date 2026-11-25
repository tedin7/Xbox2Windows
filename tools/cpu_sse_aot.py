"""Recompile bounded Windows x64 leaf functions that use scalar SSE/SSE2 arithmetic to C++.

Each export is named with its Windows x64 signature, `name=<result>:<slots>`, where every
letter is `f` (float), `d` (double) or `q` (a 64-bit integer or pointer). Slot N arrives in
XMM0-XMM3 when it is floating point and in RCX, RDX, R8 or R9 otherwise; the result leaves in
XMM0 or RAX. Control flow and the integer instructions are those of cpu_switch_aot.

Scalar SQRT, MIN and MAX are lifted too. CVTSS2SI and CVTSD2SI (rounded by the MXCSR rounding control, DAZ read
at run time, FTZ not involved) are lifted next to the truncating CVTTSS2SI and CVTTSD2SI; the invalid and precision
exception flags they set are not observed. The generated code runs under the host MXCSR: a guest MXCSR write is not
lifted, so the rounding mode, DAZ and FTZ are the host's and the exception flags are not observed. SQRT, MIN and
MAX read their operands at run time, so a constant operand is not folded with the default environment; ADD, SUB,
MUL and DIV have no such barrier.

Only the low quadword of an XMM register is modelled. Every supported instruction computes
that quadword from low quadwords alone, so the upper half never reaches a result. The two
dwords of the low quadword are tracked separately as defined or undefined: a float argument
defines the low dword only, and a read of an undefined part is rejected.
"""

import argparse
from pathlib import Path
import re

import pefile
from iced_x86 import Code, ConditionCode as CC, FlowControl, MemorySizeExt, Mnemonic as M, OpKind, Register as R, RegisterExt

from cpu_leaf_aot import export_rva
from cpu_switch_aot import (CMOVCC, FLAG_PRODUCERS, Function, Lifter, NAMES, PRELUDE, SETCC, VOLATILE,
                            flag_producer, full, signed, unsigned)


SLOTS = (R.RCX, R.RDX, R.R8, R.R9)
XMM_SLOTS = (R.XMM0, R.XMM1, R.XMM2, R.XMM3)
XMM = {getattr(R, f"XMM{number}"): f"xmm{number}" for number in range(16)}
VOLATILE_XMM = {getattr(R, f"XMM{number}") for number in range(6)}
TYPES = {4: "float", 8: "double"}
MOVES = {M.MOVSS: 4, M.MOVSD: 8}
COPIES = {M.MOVAPS, M.MOVAPD, M.MOVUPS, M.MOVUPD}
ARITHMETIC = {M.ADDSS: ("+", 4), M.SUBSS: ("-", 4), M.MULSS: ("*", 4), M.DIVSS: ("/", 4),
              M.ADDSD: ("+", 8), M.SUBSD: ("-", 8), M.MULSD: ("*", 8), M.DIVSD: ("/", 8)}
FROM_INTEGER = {M.CVTSI2SS: 4, M.CVTSI2SD: 8}
TRUNCATIONS = {M.CVTTSS2SI: 4, M.CVTTSD2SI: 8}
ROUNDINGS = {M.CVTSS2SI: 4, M.CVTSD2SI: 8}
RESIZES = {M.CVTSS2SD: 4, M.CVTSD2SS: 8}
COMPARES = {M.UCOMISS: 4, M.COMISS: 4, M.UCOMISD: 8, M.COMISD: 8}
ROOTS = {M.SQRTSS: 4, M.SQRTSD: 8}
# MIN and MAX with the comparison that keeps the destination operand.
EXTREMA = {M.MINSS: ("<", 4), M.MAXSS: (">", 4), M.MINSD: ("<", 8), M.MAXSD: (">", 8)}
ZEROING = {M.XORPS, M.XORPD, M.PXOR}
GPR_STORE_MOVES = {Code.MOVD_RM32_XMM: (4, RegisterExt.is_gpr32), Code.MOVQ_RM64_XMM: (8, RegisterExt.is_gpr64)}
GPR_LOAD_MOVES = {Code.MOVD_XMM_RM32: (4, RegisterExt.is_gpr32), Code.MOVQ_XMM_RM64: (8, RegisterExt.is_gpr64)}
SCALAR = {*MOVES, *COPIES, *ARITHMETIC, *ROOTS, *EXTREMA, *FROM_INTEGER, *TRUNCATIONS, *ROUNDINGS, *RESIZES, *COMPARES, *ZEROING,
          M.MOVD, M.MOVQ}
# fc holds ZF, PF and CF of the last compare as bits 2, 1 and 0. An unordered compare sets all three.
CONDITIONS = {CC.E: "(fc & 4u) != 0", CC.NE: "(fc & 4u) == 0", CC.B: "(fc & 1u) != 0", CC.AE: "(fc & 1u) == 0",
              CC.BE: "(fc & 5u) != 0", CC.A: "(fc & 5u) == 0", CC.P: "(fc & 2u) != 0", CC.NP: "(fc & 2u) == 0"}


def parts(register, width):
    """Definedness keys of the low dword and, for a double, of bits 32-63."""
    return {(register, 0)} if width == 4 else {(register, 0), (register, 1)}


def describe(key):
    if isinstance(key, tuple):
        return XMM[key[0]] + ("[63:32]" if key[1] else "")
    return NAMES[key]


def parse_export(text):
    match = re.fullmatch(r"(\w+)=([fdq]):([fdq]{0,4})", text)
    if not match:
        raise ValueError(f"invalid signature {text!r}: expected name=<result>:<slots> of f, d and q")
    return match.groups()


class LeafFunction(Function):
    def constants(self):
        """No register is tracked as a table base, so every indirect jump is rejected."""
        return {}


class ScalarLifter(Lifter):
    def __init__(self, result):
        super().__init__()
        self.result = result
        self.vectors = set()
        self.compare = False
        self.merges = []

    def xmm(self, ins, number):
        register = ins.op_register(number)
        if ins.op_kind(number) != OpKind.REGISTER or register not in XMM:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")
        self.vectors.add(register)
        return register

    def pointer(self, ins, reads):
        address = self.address(ins)
        reads.update(full(r) for r in (ins.memory_base, ins.memory_index) if r != R.NONE)
        return address

    def source(self, ins, number, width, reads):
        """The bits of a float or double operand, from an XMM register or from memory."""
        if ins.op_kind(number) == OpKind.MEMORY and MemorySizeExt.size(ins.memory_size) == width:
            return f"load<{unsigned(width)}>({self.pointer(ins, reads)})"
        register = self.xmm(ins, number)
        reads.update(parts(register, width))
        return XMM[register] if width == 8 else f"static_cast<std::uint32_t>({XMM[register]})"

    def assign(self, register, width, bits):
        name = XMM[register]
        return f"{name} = {bits};" if width == 8 else f"{name} = low({name}, {bits});"

    def statement(self, ins, next_ins, sets_flags=False, producer=None):
        mnemonic = ins.mnemonic
        reads, writes = set(), set()
        self.merges = []
        if mnemonic == M.RET and self.result == "q":
            return "return rax;", {R.RAX}, writes
        if mnemonic == M.RET:
            width = 4 if self.result == "f" else 8
            self.vectors.add(R.XMM0)
            bits = "xmm0" if width == 8 else "static_cast<std::uint32_t>(xmm0)"
            return f"return cast<{TYPES[width]}>({bits});", parts(R.XMM0, width), writes
        vector = any(ins.op_kind(n) == OpKind.REGISTER and not RegisterExt.is_gpr(ins.op_register(n))
                     for n in range(ins.op_count))
        unsupported = ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")
        if mnemonic not in SCALAR or ins.op_count != 2 or not all(
                ins.op_kind(n) in (OpKind.REGISTER, OpKind.MEMORY) for n in range(2)):
            if vector:
                raise unsupported
            text, gpr_reads, gpr_writes = super().statement(
                ins, sets_flags or (next_ins is not None and next_ins.flow_control == FlowControl.CONDITIONAL_BRANCH),
                producer)
            # This leaf profile tracks whole GPRs. A byte or word write retains the previous high bits.
            partial = {register for register, width in gpr_writes.items() if width < 4}
            return text, set(gpr_reads) | partial, set(gpr_writes)

        if mnemonic in MOVES or mnemonic in (M.MOVD, M.MOVQ):
            width = MOVES.get(mnemonic, 4 if mnemonic == M.MOVD else 8)
            if ins.op0_kind == OpKind.MEMORY:
                if MemorySizeExt.size(ins.memory_size) != width:
                    raise unsupported
                bits = self.source(ins, 1, width, reads)
                return f"store<{unsigned(width)}>({self.pointer(ins, reads)}, {bits});", reads, writes
            if ins.op0_register not in XMM:
                # MOVD r32, xmm zero-extends like any 32-bit write; MOVQ r64, xmm transfers all 64 bits verbatim.
                if ins.code not in GPR_STORE_MOVES:
                    raise unsupported
                gpr_width, is_gpr_width = GPR_STORE_MOVES[ins.code]
                if not is_gpr_width(ins.op0_register):
                    raise unsupported
                bits = self.source(ins, 1, gpr_width, reads)
                writes.add(full(ins.op0_register))
                return self.write(ins, gpr_width, bits), reads, writes
            destination = self.xmm(ins, 0)
            if ins.op1_kind == OpKind.REGISTER and ins.op1_register not in XMM:
                if ins.code not in GPR_LOAD_MOVES:
                    raise unsupported
                gpr_width, is_gpr_width = GPR_LOAD_MOVES[ins.code]
                if not is_gpr_width(ins.op1_register):
                    raise unsupported
                reads.add(full(ins.op1_register))
                bits = self.operand(ins, 1, gpr_width)
            else:
                bits = self.source(ins, 1, width, reads)
            if mnemonic == M.MOVSS and ins.op1_kind == OpKind.REGISTER:
                writes.update(parts(destination, 4))
                return self.assign(destination, 4, bits), reads, writes
            # Loads, MOVD, MOVQ and MOVSD between registers define the whole low quadword.
            writes.update(parts(destination, 8))
            return f"{XMM[destination]} = {bits};", reads, writes
        if mnemonic in TRUNCATIONS or mnemonic in ROUNDINGS:
            width = RegisterExt.size(ins.op0_register) if ins.op0_kind == OpKind.REGISTER else 0
            if width not in (4, 8) or not RegisterExt.is_gpr(ins.op0_register):
                raise unsupported
            source = {**TRUNCATIONS, **ROUNDINGS}[mnemonic]
            bits = self.source(ins, 1, source, reads)
            writes.add(full(ins.op0_register))
            convert = "truncate" if mnemonic in TRUNCATIONS else "round_to_integer"
            value = f"{convert}<{signed(width)}>(cast<{TYPES[source]}>({bits}))"
            return self.write(ins, width, value), reads, writes

        destination = self.xmm(ins, 0)
        name = XMM[destination]
        if mnemonic in COPIES or mnemonic in ZEROING:
            source = self.xmm(ins, 1)
            if mnemonic in ZEROING:
                # Only the zeroing idiom: any other use combines the two registers bit by bit.
                if source != destination:
                    raise unsupported
                writes.update(parts(destination, 8))
                return f"{name} = 0;", reads, writes
            # A register copy may carry undefined parts, which stay undefined in the copy.
            self.merges = [((destination, half), (source, half)) for half in (0, 1)]
            return f"{name} = {XMM[source]};", reads, writes
        if mnemonic in ARITHMETIC:
            operator, width = ARITHMETIC[mnemonic]
            reads.update(parts(destination, width))
            a = name if width == 8 else f"static_cast<std::uint32_t>({name})"
            b = self.source(ins, 1, width, reads)
            return self.assign(destination, width, f"arith<{TYPES[width]}>({a}, {b}, '{operator}')"), reads, writes
        if mnemonic in ROOTS:
            # The source alone is read. The destination keeps every bit above the result.
            width = ROOTS[mnemonic]
            bits = self.source(ins, 1, width, reads)
            writes.update(parts(destination, width))
            return self.assign(destination, width, f"root<{TYPES[width]}>({bits})"), reads, writes
        if mnemonic in EXTREMA:
            operator, width = EXTREMA[mnemonic]
            reads.update(parts(destination, width))
            a = name if width == 8 else f"static_cast<std::uint32_t>({name})"
            b = self.source(ins, 1, width, reads)
            return self.assign(destination, width, f"extreme<{TYPES[width]}>({a}, {b}, '{operator}')"), reads, writes
        if mnemonic in FROM_INTEGER:
            width = FROM_INTEGER[mnemonic]
            if ins.op1_kind == OpKind.REGISTER:
                source = RegisterExt.size(ins.op1_register)
                if source not in (4, 8) or not RegisterExt.is_gpr(ins.op1_register):
                    raise unsupported
                reads.add(full(ins.op1_register))
                integer = self.operand(ins, 1, source)
            else:
                source = MemorySizeExt.size(ins.memory_size)
                integer = f"load<{unsigned(source)}>({self.pointer(ins, reads)})"
            writes.update(parts(destination, width))
            value = f"static_cast<{TYPES[width]}>(static_cast<{signed(source)}>({integer}))"
            return self.assign(destination, width, f"cast<{unsigned(width)}>({value})"), reads, writes
        if mnemonic in RESIZES:
            source = RESIZES[mnemonic]
            width = 12 - source
            bits = self.source(ins, 1, source, reads)
            writes.update(parts(destination, width))
            value = f"resize<{TYPES[width]}>(cast<{TYPES[source]}>({bits}))"
            return self.assign(destination, width, f"cast<{unsigned(width)}>({value})"), reads, writes
        width = COMPARES[mnemonic]
        if next_ins is None or next_ins.flow_control != FlowControl.CONDITIONAL_BRANCH:
            raise ValueError(f"float compare without a conditional branch at RVA {ins.ip:#x}")
        reads.update(parts(destination, width))
        a = name if width == 8 else f"static_cast<std::uint32_t>({name})"
        b = self.source(ins, 1, width, reads)
        self.compare = True
        return f"fc = compare(cast<{TYPES[width]}>({a}), cast<{TYPES[width]}>({b}));", reads, writes


def condition(function, edges, lifter, jcc):
    """A float compare may feed a run of conditional jumps; integer flags feed the next jump only."""
    current = jcc
    while True:
        producer = function.ending.get(current.ip)
        if current.ip in edges or producer is None or producer.flow_control not in (
                FlowControl.NEXT, FlowControl.CONDITIONAL_BRANCH):
            raise ValueError(f"flags cross a join before RVA {jcc.ip:#x}")
        if producer.flow_control == FlowControl.NEXT:
            break
        current = producer
    if producer.mnemonic in COMPARES:
        if jcc.condition_code not in CONDITIONS:
            raise ValueError(f"unsupported condition after a float compare at RVA {jcc.ip:#x}")
        return CONDITIONS[jcc.condition_code]
    if current is not jcc:
        raise ValueError(f"flags cross a branch before RVA {jcc.ip:#x}")
    if producer.mnemonic not in FLAG_PRODUCERS:
        raise ValueError(f"unsupported flag producer at RVA {producer.ip:#x}")
    return lifter.condition(producer, jcc)


def lift(pe, name, result, slots):
    entry = export_rva(pe, name)
    function = LeafFunction(pe, entry)
    edges = function.explore()
    lifter = ScalarLifter(result)
    order = sorted(function.instructions)
    instructions = function.instructions
    producers = {ip: flag_producer(function, edges, ins) for ip, ins in instructions.items()
                 if ins.mnemonic in SETCC | CMOVCC}
    sets_flags = {producer.ip for producer in producers.values()}
    body = {}
    effects = {}
    for ip in order:
        ins = instructions[ip]
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix:
            raise ValueError(f"unsupported prefix at RVA {ip:#x}")
        following = instructions.get(ins.next_ip) if ins.flow_control == FlowControl.NEXT else None
        effects[ip] = (set(), set(), [])
        if ins.flow_control == FlowControl.CONDITIONAL_BRANCH:
            body[ip] = f"if ({condition(function, edges, lifter, ins)}) goto loc_{ins.near_branch_target:x};"
        elif ins.flow_control == FlowControl.UNCONDITIONAL_BRANCH:
            body[ip] = f"goto loc_{ins.near_branch_target:x};"
        else:
            if ins.mnemonic != M.RET and ins.is_stack_instruction:
                raise ValueError(f"stack operation in leaf profile at RVA {ip:#x}")
            body[ip], reads, writes = lifter.statement(ins, following, ip in sets_flags, producers.get(ip))
            effects[ip] = (reads, writes, lifter.merges)
        reads, writes, merges = effects[ip]
        touched = reads | writes | {key for pair in merges for key in pair}
        saved = sorted(describe(key) for key in touched
                       if (key[0] not in VOLATILE_XMM if isinstance(key, tuple) else key not in VOLATILE))
        if saved:
            raise ValueError(f"nonvolatile register {saved[0]} at RVA {ip:#x}")

    # Must-defined registers and XMM parts: the arguments at entry, intersected at every join.
    arguments = set()
    for number, kind in enumerate(slots):
        arguments |= {SLOTS[number]} if kind == "q" else parts(XMM_SLOTS[number], 4 if kind == "f" else 8)
    everything = {*NAMES, *((register, half) for register in XMM for half in (0, 1))}
    defined = {ip: set(everything) for ip in order}
    defined[entry] = set(arguments)
    changed = True
    while changed:
        changed = False
        for ip in order:
            _, writes, merges = effects[ip]
            out = defined[ip] | writes
            for destination, source in merges:
                if source not in defined[ip]:
                    out.discard(destination)
                else:
                    out.add(destination)
            for successor in function.successors[ip]:
                merged = defined[successor] & out
                if merged != defined[successor]:
                    defined[successor] = merged
                    changed = True
    for ip in order:
        missing = sorted(describe(key) for key in effects[ip][0] - defined[ip])
        if missing:
            raise ValueError(f"undefined register {missing[0]} at RVA {ip:#x}")

    labels = set(edges)
    for position, ip in enumerate(order):
        ins = instructions[ip]
        following = order[position + 1] if position + 1 < len(order) else None
        if ins.flow_control in (FlowControl.NEXT, FlowControl.CONDITIONAL_BRANCH) and following != ins.next_ip:
            labels.add(ins.next_ip)
    integers = lifter.used | ({R.RAX} if result == "q" else set())
    parameters = []
    declarations = []
    for number, kind in enumerate(slots):
        if kind == "q":
            parameters.append("word" + (f" {NAMES[SLOTS[number]]}" if SLOTS[number] in integers else ""))
        elif XMM_SLOTS[number] in lifter.vectors:
            width = 4 if kind == "f" else 8
            parameters.append(f"{TYPES[width]} a{number}")
            declarations.append(f"word xmm{number} = cast<{unsigned(width)}>(a{number});")
        else:
            parameters.append(TYPES[4 if kind == "f" else 8])
    passed = {SLOTS[n] if kind == "q" else XMM_SLOTS[n] for n, kind in enumerate(slots)}
    declarations += [f"word {NAMES[r]} = 0;" for r in sorted(integers - passed, key=lambda r: NAMES[r])]
    declarations += [f"word {XMM[r]} = 0;" for r in sorted(lifter.vectors - passed)]
    if lifter.flags:
        declarations.append("word fa = 0, fb = 0;")
    if lifter.compare:
        declarations.append("unsigned fc = 0;")
    text = []
    for position, ip in enumerate(order):
        if ip in labels:
            text.append(f"loc_{ip:x}:")
        if body[ip]:
            text.append("    " + body[ip])
        ins = instructions[ip]
        following = order[position + 1] if position + 1 < len(order) else None
        if ins.flow_control in (FlowControl.NEXT, FlowControl.CONDITIONAL_BRANCH) and following != ins.next_ip:
            text.append(f"    goto loc_{ins.next_ip:x};")
    returned = {"f": "float", "d": "double", "q": "word"}[result]
    signature = f'extern "C" __declspec(dllexport) {returned} lifted_{name}({", ".join(parameters)})'
    source = [signature + " {", *("    " + d for d in declarations), *text, "}"]
    return "\n".join(source), function


FLOAT_PRELUDE = PRELUDE + """#include <cmath>
#include <emmintrin.h>
#include <limits>
// One host instruction per guest operation: no fused multiply-add.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract(off)
#else
#pragma STDC FP_CONTRACT OFF
#endif
template <class To, class From> static To cast(From value) {
    static_assert(sizeof(To) == sizeof(From), "bit cast between sizes");
    To out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}
// A scalar single result replaces the low dword and keeps bits 32-63 of the destination.
template <class Bits> static word low(word old, Bits value) {
    return (old & ~0xffffffffull) | static_cast<std::uint32_t>(value);
}
// ADD, SUB, MUL and DIV with exceptions masked. A NaN destination operand is returned quieted,
// otherwise a NaN source operand; an invalid operation gives the default NaN, whose sign is set.
// The host compiler may swap the operands of its own addition, so the rule is applied here.
template <class Float, class Bits> static Bits arith(Bits a, Bits b, char op) {
    const Bits quiet = static_cast<Bits>(1) << (std::numeric_limits<Float>::digits - 2);
    const Float x = cast<Float>(a), y = cast<Float>(b);
    if (x != x) return a | quiet;
    if (y != y) return b | quiet;
    const Float r = op == '+' ? x + y : op == '-' ? x - y : op == '*' ? x * y : x / y;
    return r != r ? static_cast<Bits>(~(quiet - 1)) : cast<Bits>(r);
}
// The host reads a value from a volatile so that no operation on it is evaluated by the compiler: a constant
// operand would otherwise be folded with the default rounding and without DAZ instead of under the host MXCSR.
template <class Float> static Float runtime(Float value) {
    volatile Float input = value;
    return input;
}
// With DAZ set a denormal source operand is read as a zero of its sign, also when it is returned unchanged
// as the result of MIN or MAX. The host compare of a denormal with zero shows whether DAZ is set.
template <class Float, class Bits> static Bits daz(Bits bits) {
    const Bits sign = static_cast<Bits>(1) << (8 * sizeof(Bits) - 1);
    const Bits fraction = (static_cast<Bits>(1) << (std::numeric_limits<Float>::digits - 1)) - 1;
    const Bits magnitude = bits & ~sign;
    if (magnitude == 0 || magnitude > fraction) return bits;
    return runtime(cast<Float>(static_cast<Bits>(1))) == static_cast<Float>(0) ? bits & sign : bits;
}
// SQRTSS and SQRTSD run on the processor itself, through the intrinsics, and not through the host library
// function: the processor reads a denormal operand as a zero of its sign under DAZ, returns -0 for -0, quiets a
// NaN operand, gives the default NaN for any other negative operand and rounds as the MXCSR says. A C library
// root decides those cases on its own (a runtime that tests the sign of the bits returns a NaN for a negative
// denormal under DAZ, where the processor returns -0).
static float hardware_root(float value) { return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(value))); }
static double hardware_root(double value) {
    const __m128d operand = _mm_set_sd(value);
    return _mm_cvtsd_f64(_mm_sqrt_sd(operand, operand));
}
template <class Float, class Bits> static Bits root(Bits b) {
    return cast<Bits>(hardware_root(runtime(cast<Float>(b))));
}
// MINSS, MAXSS, MINSD and MAXSD return the destination only when the comparison is true. Equal values
// (so zeros of either sign) and any NaN, which compares false, give the source operand unchanged.
template <class Float, class Bits> static Bits extreme(Bits a, Bits b, char op) {
    const Float x = runtime(cast<Float>(a)), y = runtime(cast<Float>(b));
    return daz<Float>((op == '<' ? x < y : x > y) ? a : b);
}
// CVTTSS2SI and CVTTSD2SI: a NaN or a value outside the destination range gives the integer
// indefinite, which is the sign bit alone.
template <class Integer, class Float> static Integer truncate(Float value) {
    const Float limit = -static_cast<Float>(std::numeric_limits<Integer>::min());
    return value >= -limit && value < limit ? static_cast<Integer>(value) : std::numeric_limits<Integer>::min();
}
// CVTSS2SI and CVTSD2SI round by the MXCSR rounding control and read a denormal as a zero under DAZ. They run on
// the processor, through the intrinsics, and not through lrint or llrint of the host C library: that library does
// not follow the instruction for a NaN or a value out of range and, on Windows, for a rounding control other than
// the default. The processor gives the integer indefinite, the sign bit alone, for a NaN, an infinity or a result
// that does not fit. The operand is read at run time, so a constant is not rounded by the compiler.
static std::int32_t hardware_round(float value, std::int32_t) { return _mm_cvtss_si32(_mm_set_ss(value)); }
static std::int64_t hardware_round(float value, std::int64_t) { return _mm_cvtss_si64(_mm_set_ss(value)); }
static std::int32_t hardware_round(double value, std::int32_t) { return _mm_cvtsd_si32(_mm_set_sd(value)); }
static std::int64_t hardware_round(double value, std::int64_t) { return _mm_cvtsd_si64(_mm_set_sd(value)); }
template <class Integer, class Float> static Integer round_to_integer(Float value) {
    return hardware_round(runtime(value), Integer());
}
// CVTSS2SD and CVTSD2SS: the volatile input keeps every conversion, so a pair is not folded to an
// identity. Each one quiets a signalling NaN and, with DAZ or FTZ set, flushes denormals, as on hardware.
template <class To, class From> static To resize(From value) {
    volatile From input = value;
    return static_cast<To>(input);
}
// ZF, PF and CF of COMISS, UCOMISS, COMISD and UCOMISD as bits 2, 1 and 0.
template <class Float> static unsigned compare(Float a, Float b) {
    return a != a || b != b ? 7u : a < b ? 1u : a == b ? 4u : 0u;
}
"""


def translate_pe(pe, exports):
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    functions = [lift(pe, *parse_export(text)) for text in exports]
    return FLOAT_PRELUDE + "\n".join(f"\n{source}" for source, _ in functions) + "\n", [f for _, f in functions]


def translate(image, exports):
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, exports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated name=<result>:<slots> signatures")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        source, functions = translate(args.image, args.exports.split(","))
        args.output.write_text(source, encoding="utf-8")
        for function in functions:
            print(f"RVA {function.entry:#x}: {len(function.instructions)} instructions")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
