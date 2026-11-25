"""Lift bounded Windows x64 leaf functions with BMI2 RORX, SARX, SHLX, and SHRX to C++.

Each export uses the leaf ABI:
    std::uint64_t lifted_<name>(word rcx, word rdx, word r8)
where rcx, rdx, and r8 enter as 64-bit integer arguments or flat pointers to caller-owned
buffers and RAX carries the 64-bit return value.

Supported BMI2 shift/rotate instructions (docs/coverage/x64_zen2.tsv group BMI/ADX):
  - RORX r32, r/m32, imm8  |  RORX r64, r/m64, imm8   (count masked to 5/6 bits)
  - SARX r32, r/m32, r32   |  SARX r64, r/m64, r64    (arithmetic right shift, 5/6-bit mask)
  - SHLX r32, r/m32, r32   |  SHLX r64, r/m64, r64    (logical left shift, 5/6-bit mask)
  - SHRX r32, r/m32, r32   |  SHRX r64, r/m64, r64    (logical right shift, 5/6-bit mask)

All four BMI2 forms leave EFLAGS completely unmodified (CF, ZF, SF, OF, PF preserved across
CMP -> BMI2 shift -> SETcc). 32-bit destinations zero-extend into bits [63:32], and counts
mask modulo 32 (so count 32 shifts by 0) or modulo 64. Memory operands are flat
[rcx|rdx|r8 + disp] loads (m32/m64) read unconditionally via load_always() so /O2 cannot
eliminate faulting reads.

Transport / flag-packing glue:
  - Volatile scratch GPR (rax/r9/r10/r11) mov, movzx, cmp, or, setcc (reg or [r8 + disp]),
    flat [r8 + disp] stores, and terminal ret.

EVEX/XOP encodings, legacy SHL/SAR/SHR/ROR, BMI1 (ANDN/BEXTR/...), other BMI2 (MULX/PDEP/
PEXT/BZHI), nonvolatile GPRs (RBX/RBP/RSI/RDI/R12..R15/RSP), writes to argument GPRs
(rcx/rdx/r8), high-byte GPRs, undefined GPR or flag reads, RIP-relative/SIB-indexed/
non-argument memory operands, stores outside [r8], instruction prefixes, branches/calls,
and missing or trailing instructions after ret are rejected.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import (
    Code,
    Decoder,
    EncodingKind,
    FlowControl,
    MemorySizeExt,
    OpKind,
    Register as R,
    RegisterExt,
)

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}
ARGUMENT_NAMES = {"rcx", "rdx", "r8"}

VOLATILE_GPR64 = {
    R.RAX: "rax",
    R.RCX: "rcx",
    R.RDX: "rdx",
    R.R8: "r8",
    R.R9: "r9",
    R.R10: "r10",
    R.R11: "r11",
}
VOLATILE_GPR32 = {
    R.EAX: "rax",
    R.ECX: "rcx",
    R.EDX: "rdx",
    R.R8D: "r8",
    R.R9D: "r9",
    R.R10D: "r10",
    R.R11D: "r11",
}
VOLATILE_GPR16 = {
    R.AX: "rax",
    R.CX: "rcx",
    R.DX: "rdx",
    R.R8W: "r8",
    R.R9W: "r9",
    R.R10W: "r10",
    R.R11W: "r11",
}
VOLATILE_GPR8 = {
    R.AL: "rax",
    R.CL: "rcx",
    R.DL: "rdx",
    R.R8L: "r8",
    R.R9L: "r9",
    R.R10L: "r10",
    R.R11L: "r11",
}
HIGH_BYTE_GPRS = {R.AH, R.CH, R.DH, R.BH}
NONVOLATILE_GPR = {
    R.RBX, R.EBX, R.BX, R.BL,
    R.RBP, R.EBP, R.BP, R.BPL,
    R.RSI, R.ESI, R.SI, R.SIL,
    R.RDI, R.EDI, R.DI, R.DIL,
    R.RSP, R.ESP, R.SP, R.SPL,
    R.R12, R.R12D, R.R12W, R.R12L,
    R.R13, R.R13D, R.R13W, R.R13L,
    R.R14, R.R14D, R.R14W, R.R14L,
    R.R15, R.R15D, R.R15W, R.R15L,
}

# (width_bits, helper_name, is_imm8)
BMI2_SHIFT_SPECS = {
    Code.VEX_RORX_R32_RM32_IMM8: (32, "op_rorx32", True),
    Code.VEX_RORX_R64_RM64_IMM8: (64, "op_rorx64", True),
    Code.VEX_SARX_R32_RM32_R32: (32, "op_sarx32", False),
    Code.VEX_SARX_R64_RM64_R64: (64, "op_sarx64", False),
    Code.VEX_SHLX_R32_RM32_R32: (32, "op_shlx32", False),
    Code.VEX_SHLX_R64_RM64_R64: (64, "op_shlx64", False),
    Code.VEX_SHRX_R32_RM32_R32: (32, "op_shrx32", False),
    Code.VEX_SHRX_R64_RM64_R64: (64, "op_shrx64", False),
}

MOV_WIDTHS = {
    Code.MOV_R64_RM64: 64,
    Code.MOV_RM64_R64: 64,
    Code.MOV_RM64_IMM32: 64,
    Code.MOV_R64_IMM64: 64,
    Code.MOV_R32_RM32: 32,
    Code.MOV_RM32_R32: 32,
    Code.MOV_RM32_IMM32: 32,
    Code.MOV_R32_IMM32: 32,
    Code.MOV_R8_RM8: 8,
    Code.MOV_RM8_R8: 8,
    Code.MOV_RM8_IMM8: 8,
    Code.MOV_R8_IMM8: 8,
}
MOVZX_SPECS = {
    Code.MOVZX_R32_RM8: (32, 8),
    Code.MOVZX_R64_RM8: (64, 8),
}
CMP_WIDTHS = {
    Code.CMP_R32_RM32: 32,
    Code.CMP_RM32_R32: 32,
    Code.CMP_RM32_IMM8: 32,
    Code.CMP_RM32_IMM32: 32,
    Code.CMP_EAX_IMM32: 32,
    Code.CMP_R64_RM64: 64,
    Code.CMP_RM64_R64: 64,
    Code.CMP_RM64_IMM8: 64,
    Code.CMP_RM64_IMM32: 64,
    Code.CMP_RAX_IMM32: 64,
}
OR_WIDTHS = {
    Code.OR_R32_RM32: 32,
    Code.OR_RM32_R32: 32,
    Code.OR_RM32_IMM8: 32,
    Code.OR_R64_RM64: 64,
    Code.OR_RM64_R64: 64,
    Code.OR_RM64_IMM8: 64,
}
SETCC_SPECS = {
    Code.SETE_RM8: ("zf", "flags.zf"),
    Code.SETNE_RM8: ("zf", "!flags.zf"),
    Code.SETB_RM8: ("cf", "flags.cf"),
    Code.SETAE_RM8: ("cf", "!flags.cf"),
    Code.SETS_RM8: ("sf", "flags.sf"),
    Code.SETNS_RM8: ("sf", "!flags.sf"),
    Code.SETO_RM8: ("of", "flags.of"),
    Code.SETNO_RM8: ("of", "!flags.of"),
    Code.SETP_RM8: ("pf", "flags.pf"),
    Code.SETNP_RM8: ("pf", "!flags.pf"),
}

PRELUDE = """#include <atomic>
#include <cstdint>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

using word = std::uint64_t;

struct Flags {
    bool cf = false;
    bool zf = false;
    bool sf = false;
    bool of = false;
    bool pf = false;
};

template <class T>
static inline T load_always(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const T v = *reinterpret_cast<const volatile T __unaligned*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return v;
}

template <class T>
static inline void store_always(word address, T value) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    *reinterpret_cast<volatile T __unaligned*>(address) = value;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline bool parity_even(std::uint8_t byte) {
    byte ^= static_cast<std::uint8_t>(byte >> 4);
    byte &= 0x0Fu;
    return ((0x9669u >> byte) & 1u) != 0u;
}

static inline word update_gpr_width(word cur, std::uint64_t val, unsigned width) {
    if (width == 8u) {
        return (cur & ~0xFFull) | (val & 0xFFull);
    }
    if (width == 32u) {
        return val & 0xFFFFFFFFull;
    }
    return val;
}

// --- Flagless BMI2 shift / rotate helpers (5-bit count for 32-bit, 6-bit count for 64-bit) ---

static inline word op_rorx32(word val, std::uint8_t imm8) {
    const std::uint32_t v = static_cast<std::uint32_t>(val);
    // Mask to 5 bits before (32u - c) to avoid C++ shift-count UB when imm8 >= 32.
    const unsigned c = static_cast<unsigned>(imm8 & 31u);
    const std::uint32_t r = (c == 0u) ? v : ((v >> c) | (v << (32u - c)));
    return static_cast<word>(r);
}

static inline word op_rorx64(word val, std::uint8_t imm8) {
    // Mask to 6 bits before (64u - c) to avoid C++ shift-count UB when imm8 >= 64.
    const unsigned c = static_cast<unsigned>(imm8 & 63u);
    return (c == 0u) ? val : ((val >> c) | (val << (64u - c)));
}

static inline word op_shlx32(word val, word cnt) {
    const std::uint32_t v = static_cast<std::uint32_t>(val);
    const unsigned c = static_cast<unsigned>(cnt & 31ull);
    return static_cast<word>(v << c);
}

static inline word op_shlx64(word val, word cnt) {
    const unsigned c = static_cast<unsigned>(cnt & 63ull);
    return val << c;
}

static inline word op_shrx32(word val, word cnt) {
    const std::uint32_t v = static_cast<std::uint32_t>(val);
    const unsigned c = static_cast<unsigned>(cnt & 31ull);
    return static_cast<word>(v >> c);
}

static inline word op_shrx64(word val, word cnt) {
    const unsigned c = static_cast<unsigned>(cnt & 63ull);
    return val >> c;
}

// Build sign fill explicitly on unsigned types so arithmetic right shift does not rely on signed >>.
static inline word op_sarx32(word val, word cnt) {
    const std::uint32_t v = static_cast<std::uint32_t>(val);
    const unsigned c = static_cast<unsigned>(cnt & 31ull);
    if (c == 0u) {
        return static_cast<word>(v);
    }
    const std::uint32_t sign = (v & 0x80000000u) ? (~0u << (32u - c)) : 0u;
    return static_cast<word>((v >> c) | sign);
}

static inline word op_sarx64(word val, word cnt) {
    const unsigned c = static_cast<unsigned>(cnt & 63ull);
    if (c == 0u) {
        return val;
    }
    const word sign = (val & 0x8000000000000000ull) ? (~0ull << (64u - c)) : 0ull;
    return (val >> c) | sign;
}

static inline void compute_cmp32(word lhs_w, word rhs_w, Flags& flags) {
    const std::uint32_t a = static_cast<std::uint32_t>(lhs_w);
    const std::uint32_t b = static_cast<std::uint32_t>(rhs_w);
    const std::uint32_t r = a - b;
    flags.cf = a < b;
    flags.zf = r == 0u;
    flags.sf = (r & 0x80000000u) != 0u;
    flags.of = (((a ^ b) & (a ^ r)) & 0x80000000u) != 0u;
    flags.pf = parity_even(static_cast<std::uint8_t>(r));
}

static inline void compute_cmp64(word a, word b, Flags& flags) {
    const word r = a - b;
    flags.cf = a < b;
    flags.zf = r == 0ull;
    flags.sf = (r >> 63) != 0ull;
    flags.of = (((a ^ b) & (a ^ r)) >> 63) != 0ull;
    flags.pf = parity_even(static_cast<std::uint8_t>(r));
}

static inline void compute_logic_flags(word result, unsigned width, Flags& flags) {
    const std::uint64_t masked = (width == 32u) ? (result & 0xFFFFFFFFull) : result;
    flags.cf = false;
    flags.of = false;
    flags.zf = masked == 0ull;
    flags.sf = ((masked >> (width - 1u)) & 1ull) != 0ull;
    flags.pf = parity_even(static_cast<std::uint8_t>(masked));
}

#if defined(_MSC_VER)
extern "C" __declspec(dllexport) std::uint32_t host_has_bmi2() {
    int info[4]{};
    __cpuid(info, 0);
    if (info[0] < 7) return 0u;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 8)) ? 1u : 0u;
}
#endif
"""


def gpr_info(register, ins):
    """Map volatile GPR (64/32/16/8-bit low) to (name, width_bits), rejecting nonvolatile/high-byte."""
    if register in VOLATILE_GPR64:
        return VOLATILE_GPR64[register], 64
    if register in VOLATILE_GPR32:
        return VOLATILE_GPR32[register], 32
    if register in VOLATILE_GPR16:
        return VOLATILE_GPR16[register], 16
    if register in VOLATILE_GPR8:
        return VOLATILE_GPR8[register], 8
    if register in HIGH_BYTE_GPRS:
        raise ValueError(f"unsupported high-byte GPR {register} at RVA {ins.ip:#x}: {ins}")
    if register in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported GPR {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, expected_bytes, *, write=False):
    """Lower a flat [base + disp] operand rooted in an argument GPR64 (stores only to [r8])."""
    if ins.is_ip_rel_memory_operand or ins.memory_base in (R.RIP, R.EIP):
        raise ValueError(f"RIP-relative memory operand at RVA {ins.ip:#x}: {ins}")
    if ins.memory_index != R.NONE:
        raise ValueError(f"SIB index memory operand at RVA {ins.ip:#x}: {ins}")
    base = ins.memory_base
    if base == R.NONE or not RegisterExt.is_gpr64(base) or base not in ARGUMENTS:
        raise ValueError(f"bad base register in memory operand at RVA {ins.ip:#x}: {ins}")
    if write and base != R.R8:
        raise ValueError(f"non-r8 store memory operand at RVA {ins.ip:#x}: {ins}")
    actual_bytes = MemorySizeExt.size(ins.memory_size)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"unexpected {actual_bytes}-byte memory operand (expected {expected_bytes}) at RVA {ins.ip:#x}: {ins}"
        )
    disp = ins.memory_displacement & 0xFFFFFFFFFFFFFFFF
    name = ARGUMENTS[base]
    return f"({name} + {disp:#x}ull)" if disp else name


def _immediate_value(ins, op_index, width):
    kind = ins.op_kind(op_index)
    if kind == OpKind.IMMEDIATE8:
        return ins.immediate8 & 0xFF
    if kind == OpKind.IMMEDIATE32:
        return ins.immediate32 & 0xFFFFFFFF
    if kind == OpKind.IMMEDIATE64:
        return ins.immediate64 & 0xFFFFFFFFFFFFFFFF
    if kind == OpKind.IMMEDIATE8TO32:
        return ins.immediate8to32 & 0xFFFFFFFF
    if kind == OpKind.IMMEDIATE8TO64:
        return ins.immediate8to64 & 0xFFFFFFFFFFFFFFFF
    if kind == OpKind.IMMEDIATE32TO64:
        return ins.immediate32to64 & 0xFFFFFFFFFFFFFFFF
    raise ValueError(f"unsupported immediate kind {kind} for {width}-bit op at RVA {ins.ip:#x}: {ins}")


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line BMI2 shift/rotate leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    defined_gpr = {"rcx", "rdx", "r8"}
    used_scratch = set()
    defined_flags = set()
    statements = []
    returned = False

    def note_scratch(reg_name):
        if reg_name in ("r9", "r10", "r11"):
            used_scratch.add(reg_name)

    def read_reg(op_index, ins, expected_width):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        reg_name, width = gpr_info(ins.op_register(op_index), ins)
        if width != expected_width:
            raise ValueError(f"unexpected GPR width {width} at RVA {ins.ip:#x}: {ins}")
        if reg_name not in defined_gpr:
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        note_scratch(reg_name)
        mask = {8: "0xFFull", 32: "0xFFFFFFFFull"}.get(width)
        return f"({reg_name} & {mask})" if mask else reg_name

    def check_dst_reg(op_index, ins, expected_width, *, require_defined=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        reg_name, width = gpr_info(ins.op_register(op_index), ins)
        if width != expected_width:
            raise ValueError(f"unexpected GPR destination width {width} at RVA {ins.ip:#x}: {ins}")
        if reg_name in ARGUMENT_NAMES:
            raise ValueError(f"cannot write argument GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        if require_defined and reg_name not in defined_gpr:
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        note_scratch(reg_name)
        return reg_name

    def read_rm(op_index, ins, width):
        kind = ins.op_kind(op_index)
        if kind == OpKind.REGISTER:
            return read_reg(op_index, ins, width)
        if kind == OpKind.MEMORY:
            addr = flat_address(ins, width // 8, write=False)
            ctype = {8: "std::uint8_t", 32: "std::uint32_t", 64: "std::uint64_t"}[width]
            load_var = f"ld_{ins.ip:x}_{op_index}"
            statements.append(f"const word {load_var} = static_cast<word>(load_always<{ctype}>({addr}));")
            return load_var
        raise ValueError(f"unsupported operand at RVA {ins.ip:#x}: {ins}")

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")

        op_code = ins.op_code()
        encoding = op_code.encoding
        if encoding == EncodingKind.EVEX or ins.op_mask != R.NONE or ins.zeroing_masking or ins.is_broadcast:
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        if encoding == EncodingKind.XOP:
            raise ValueError(f"unsupported XOP instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code != Code.RETNQ and ins.flow_control != FlowControl.NEXT:
            raise ValueError(f"non-leaf branch or call at RVA {ins.ip:#x}: {ins}")

        code = ins.code
        if code in BMI2_SHIFT_SPECS:
            if encoding != EncodingKind.VEX:
                raise ValueError(f"expected VEX encoding for BMI2 shift at RVA {ins.ip:#x}: {ins}")
            width, helper, is_imm8 = BMI2_SHIFT_SPECS[code]
            src_expr = read_rm(1, ins, width)
            if is_imm8:
                if ins.op_kind(2) != OpKind.IMMEDIATE8:
                    raise ValueError(f"expected imm8 operand for rorx at RVA {ins.ip:#x}: {ins}")
                rhs_expr = f"static_cast<std::uint8_t>({ins.immediate8 & 0xFF:#x}u)"
            else:
                rhs_expr = read_reg(2, ins, width)
            dst = check_dst_reg(0, ins, width, require_defined=False)
            statements.append(f"{dst} = {helper}({src_expr}, {rhs_expr});")
            defined_gpr.add(dst)
        elif encoding == EncodingKind.VEX:
            raise ValueError(f"unsupported VEX instruction at RVA {ins.ip:#x}: {ins}")
        elif code == Code.RETNQ:
            if "rax" not in defined_gpr:
                raise ValueError(f"return with undefined RAX at RVA {ins.ip:#x}")
            statements.append("return rax;")
            returned = True
        elif code in MOV_WIDTHS:
            width = MOV_WIDTHS[code]
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind in (OpKind.REGISTER, OpKind.MEMORY):
                src_expr = read_rm(1, ins, width)
                dst = check_dst_reg(0, ins, width, require_defined=(width < 32))
                statements.append(f"{dst} = update_gpr_width({dst}, {src_expr}, {width}u);")
                if width >= 32:
                    defined_gpr.add(dst)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src_expr = read_rm(1, ins, width)
                addr = flat_address(ins, width // 8, write=True)
                ctype = {8: "std::uint8_t", 32: "std::uint32_t", 64: "std::uint64_t"}[width]
                statements.append(f"store_always<{ctype}>({addr}, static_cast<{ctype}>({src_expr}));")
            elif ins.op0_kind == OpKind.REGISTER:
                imm = _immediate_value(ins, 1, width)
                dst = check_dst_reg(0, ins, width, require_defined=(width < 32))
                statements.append(f"{dst} = update_gpr_width({dst}, {imm:#x}ull, {width}u);")
                if width >= 32:
                    defined_gpr.add(dst)
            elif ins.op0_kind == OpKind.MEMORY:
                imm = _immediate_value(ins, 1, width)
                addr = flat_address(ins, width // 8, write=True)
                ctype = {8: "std::uint8_t", 32: "std::uint32_t", 64: "std::uint64_t"}[width]
                statements.append(f"store_always<{ctype}>({addr}, static_cast<{ctype}>({imm:#x}ull));")
            else:
                raise ValueError(f"unsupported mov operands at RVA {ins.ip:#x}: {ins}")
        elif code in MOVZX_SPECS:
            dst_width, src_width = MOVZX_SPECS[code]
            src_expr = read_rm(1, ins, src_width)
            dst = check_dst_reg(0, ins, dst_width, require_defined=False)
            statements.append(f"{dst} = {src_expr};")
            defined_gpr.add(dst)
        elif code in CMP_WIDTHS:
            width = CMP_WIDTHS[code]
            lhs_expr = read_rm(0, ins, width)
            if ins.op1_kind in (OpKind.REGISTER, OpKind.MEMORY):
                rhs_expr = read_rm(1, ins, width)
            else:
                rhs_expr = f"{_immediate_value(ins, 1, width):#x}ull"
            statements.append(f"compute_cmp{width}({lhs_expr}, {rhs_expr}, flags);")
            defined_flags.update(("cf", "zf", "sf", "of", "pf"))
        elif code in OR_WIDTHS:
            width = OR_WIDTHS[code]
            dst = check_dst_reg(0, ins, width, require_defined=True)
            lhs_expr = read_reg(0, ins, width)
            if ins.op1_kind in (OpKind.REGISTER, OpKind.MEMORY):
                rhs_expr = read_rm(1, ins, width)
            else:
                rhs_expr = f"{_immediate_value(ins, 1, width):#x}ull"
            statements.append(f"{dst} = update_gpr_width({dst}, {lhs_expr} | {rhs_expr}, {width}u);")
            statements.append(f"compute_logic_flags({dst}, {width}u, flags);")
            defined_flags.update(("cf", "zf", "sf", "of", "pf"))
        elif code in SETCC_SPECS:
            flag_name, cond_expr = SETCC_SPECS[code]
            if flag_name not in defined_flags:
                raise ValueError(f"undefined flags read by {ins} at RVA {ins.ip:#x}")
            if ins.op0_kind == OpKind.REGISTER:
                dst = check_dst_reg(0, ins, 8, require_defined=True)
                statements.append(f"{dst} = update_gpr_width({dst}, ({cond_expr}) ? 1ull : 0ull, 8u);")
            elif ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins, 1, write=True)
                statements.append(
                    f"store_always<std::uint8_t>({addr}, static_cast<std::uint8_t>(({cond_expr}) ? 1u : 0u));"
                )
            else:
                raise ValueError(f"unsupported setcc operand at RVA {ins.ip:#x}: {ins}")
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = ["    word rax = 0ull;"]
    for gpr in ("r9", "r10", "r11"):
        if gpr in used_scratch:
            decls.append(f"    word {gpr} = 0ull;")
    decls.append("    Flags flags{};")
    body = [f"    {stmt}" for stmt in statements]
    lines = [
        f'extern "C" __declspec(dllexport) std::uint64_t lifted_{name}(word rcx, word rdx, word r8) {{',
        "    (void)rcx; (void)rdx; (void)r8;",
        *decls,
        *body,
        "}",
    ]
    return "\n".join(lines), len(instructions)


def lift(pe, name):
    entry = export_rva(pe, name)
    raw = bounded_code(pe, entry)
    source, count = lift_code(name, entry, raw)
    return source, entry, count


def translate_pe(pe, names):
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    lifted = [lift(pe, name) for name in names]
    source = PRELUDE + "\n".join(f"\n{text}" for text, _, _ in lifted) + "\n"
    meta = [(name, entry, count) for name, (_, entry, count) in zip(names, lifted)]
    return source, meta


def translate(image, names):
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated export names")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        source, meta = translate(args.image, [n for n in args.exports.split(",") if n])
        args.output.write_text(source, encoding="utf-8")
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x}: {count} instructions")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
