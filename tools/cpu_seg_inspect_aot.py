"""Lift bounded Windows x64 leaf functions with LAR, LSL, VERR, and VERW to C++.

Each export uses the leaf ABI std::uint64_t lifted_<name>(word rcx, word rdx, word r8),
where rcx, rdx, and r8 enter as 64-bit integer arguments or flat pointers to caller-owned
buffers and RAX carries the 64-bit return value.

Supported segment inspection instructions (docs/coverage/x64_zen2.tsv group Integer):
  - lar  r16/r32/r64, r/m16  (load access rights byte)
  - lsl  r16/r32/r64, r/m16  (load segment limit)
  - verr r/m16               (verify segment for reading)
  - verw r/m16               (verify segment for writing)

Exact architectural semantics modeled for standard Windows x64 user-mode (CPL=3) GDT
descriptors:
  - LAR/LSL: on ZF=1, r16 updates only bits [15:0] preserving [63:16], while r32 and r64
    zero-extend the 32-bit result to 64 bits; on ZF=0, all 64 bits of dst stay untouched.
    Only ZF is updated; CF, PF, SF, and OF are preserved.
  - VERR/VERW: sets ZF=1 only for readable/writable user-mode GDT selectors (respecting
    CPL=3, RPL, TI, present, S=1, DPL, and code/data type bits), clears ZF=0 otherwise,
    and leaves all other flags and all GPRs untouched.
  - Memory m16 selector operands are read unconditionally via volatile load16_always() so
    optimizing compilers (/O2) never eliminate faulting reads, even when the result is unused.

Non-legacy encodings (VEX, EVEX, XOP), segment/LOCK/REP prefixes, RIP-relative addressing,
SIB indexed or non-argument-rooted memory operands, clobbered pointer bases, nonvolatile GPR
reads/writes, high-byte GPRs, non-leaf branches/calls, and undefined reads are rejected.
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
    Mnemonic as M,
    OpKind,
    Register as R,
    RegisterExt,
)

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}

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

LAR_CODES = {
    Code.LAR_R16_RM16: 16,
    Code.LAR_R32_R32M16: 32,
    Code.LAR_R64_R64M16: 64,
}
LSL_CODES = {
    Code.LSL_R16_RM16: 16,
    Code.LSL_R32_R32M16: 32,
    Code.LSL_R64_R64M16: 64,
}
VERR_CODES = {Code.VERR_RM16, Code.VERR_R32M16, Code.VERR_R64M16}
VERW_CODES = {Code.VERW_RM16, Code.VERW_R32M16, Code.VERW_R64M16}

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
CMOVCC_SPECS = {
    Code.CMOVE_R32_RM32: (32, "zf", "flags.zf"),
    Code.CMOVNE_R32_RM32: (32, "zf", "!flags.zf"),
    Code.CMOVE_R64_RM64: (64, "zf", "flags.zf"),
    Code.CMOVNE_R64_RM64: (64, "zf", "!flags.zf"),
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
    Code.MOV_R16_RM16: 16,
    Code.MOV_RM16_R16: 16,
    Code.MOV_RM16_IMM16: 16,
    Code.MOV_R16_IMM16: 16,
    Code.MOV_R8_RM8: 8,
    Code.MOV_RM8_R8: 8,
    Code.MOV_RM8_IMM8: 8,
    Code.MOV_R8_IMM8: 8,
}
MOVZX_SPECS = {
    Code.MOVZX_R32_RM8: (32, 8),
    Code.MOVZX_R64_RM8: (64, 8),
    Code.MOVZX_R32_RM16: (32, 16),
    Code.MOVZX_R64_RM16: (64, 16),
}
XOR_WIDTHS = {
    Code.XOR_R32_RM32: 32,
    Code.XOR_RM32_R32: 32,
    Code.XOR_R64_RM64: 64,
    Code.XOR_RM64_R64: 64,
}
OR_WIDTHS = {
    Code.OR_R32_RM32: 32,
    Code.OR_RM32_R32: 32,
    Code.OR_RM32_IMM8: 32,
    Code.OR_R64_RM64: 64,
    Code.OR_RM64_R64: 64,
    Code.OR_RM64_IMM8: 64,
}
ADD_WIDTHS = {
    Code.ADD_R32_RM32: 32,
    Code.ADD_RM32_R32: 32,
    Code.ADD_RM32_IMM8: 32,
    Code.ADD_RM32_IMM32: 32,
    Code.ADD_EAX_IMM32: 32,
    Code.ADD_R64_RM64: 64,
    Code.ADD_RM64_R64: 64,
    Code.ADD_RM64_IMM8: 64,
    Code.ADD_RM64_IMM32: 64,
    Code.ADD_RAX_IMM32: 64,
}

PRELUDE = """#include <atomic>
#include <cstdint>

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

struct SegDesc {
    std::uint32_t lo;
    std::uint32_t hi;
};

// Standard Windows x64 GDT descriptors (user-mode CPL=3 inspection model).
static constexpr SegDesc kWin64Gdt[] = {
    {0x00000000u, 0x00000000u},  // 0x00: null
    {0x00000000u, 0x00000000u},  // 0x08: unused / non-present
    {0x00000000u, 0x00209B00u},  // 0x10: KGDT64_R0_CODE (kernel 64-bit code, DPL=0)
    {0x0000FFFFu, 0x00CF9300u},  // 0x18: KGDT64_R0_DATA (kernel data, DPL=0)
    {0x0000FFFFu, 0x00CFFB00u},  // 0x20..0x23: KGDT64_R3_CMCODE (32-bit user code)
    {0x0000FFFFu, 0x00CFF300u},  // 0x28..0x2B: KGDT64_R3_DATA (user data/stack)
    {0x00000000u, 0x0020FB00u},  // 0x30..0x33: KGDT64_R3_CODE (64-bit user code)
};

static inline bool lookup_user_desc(std::uint16_t sel, SegDesc& out) {
    const std::uint16_t ti = static_cast<std::uint16_t>((sel >> 2) & 1u);
    const std::uint16_t idx = static_cast<std::uint16_t>(sel >> 3);
    const std::uint16_t rpl = static_cast<std::uint16_t>(sel & 3u);
    if (ti != 0u || (sel & 0xFFFCu) == 0u || idx >= (sizeof(kWin64Gdt) / sizeof(kWin64Gdt[0]))) {
        return false;
    }
    const SegDesc d = kWin64Gdt[idx];
    const std::uint32_t present = (d.hi >> 15) & 1u;
    const std::uint32_t non_sys = (d.hi >> 12) & 1u;
    if (present == 0u || non_sys == 0u) {
        return false;
    }
    const std::uint32_t dpl = (d.hi >> 13) & 3u;
    const std::uint32_t type = (d.hi >> 8) & 0xFu;
    const bool is_code = (type & 8u) != 0u;
    const bool is_conforming = is_code && ((type & 4u) != 0u);
    constexpr std::uint32_t cpl = 3u;
    if (!is_conforming && (dpl < cpl || dpl < rpl)) {
        return false;
    }
    out = d;
    return true;
}

static inline bool seg_lar(std::uint16_t sel, std::uint32_t& out_ar) {
    SegDesc d{};
    if (!lookup_user_desc(sel, d)) {
        return false;
    }
    out_ar = d.hi & 0x00F0FF00u;
    return true;
}

static inline bool seg_lsl(std::uint16_t sel, std::uint32_t& out_limit) {
    SegDesc d{};
    if (!lookup_user_desc(sel, d)) {
        return false;
    }
    const std::uint32_t raw_lim = (d.lo & 0xFFFFu) | (d.hi & 0x000F0000u);
    const bool gran = (d.hi & (1u << 23)) != 0u;
    out_limit = gran ? ((raw_lim << 12) | 0xFFFu) : raw_lim;
    return true;
}

static inline bool seg_verr(std::uint16_t sel) {
    SegDesc d{};
    if (!lookup_user_desc(sel, d)) {
        return false;
    }
    const std::uint32_t type = (d.hi >> 8) & 0xFu;
    return ((type & 8u) == 0u) || ((type & 2u) != 0u);
}

static inline bool seg_verw(std::uint16_t sel) {
    SegDesc d{};
    if (!lookup_user_desc(sel, d)) {
        return false;
    }
    const std::uint32_t type = (d.hi >> 8) & 0xFu;
    return ((type & 8u) == 0u) && ((type & 2u) != 0u);
}

static inline word update_gpr_width(word cur, std::uint64_t val, unsigned width) {
    if (width == 8u) {
        return (cur & ~0xFFull) | (val & 0xFFull);
    }
    if (width == 16u) {
        return (cur & ~0xFFFFull) | (val & 0xFFFFull);
    }
    if (width == 32u) {
        return val & 0xFFFFFFFFull;
    }
    return val;
}

static inline void compute_add32(word lhs_w, word rhs_w, word& dst, Flags& flags) {
    const std::uint32_t a = static_cast<std::uint32_t>(lhs_w);
    const std::uint32_t b = static_cast<std::uint32_t>(rhs_w);
    const std::uint32_t r = a + b;
    dst = static_cast<word>(r);
    flags.cf = r < a;
    flags.zf = r == 0u;
    flags.sf = (r & 0x80000000u) != 0u;
    flags.of = (((a ^ r) & (b ^ r)) & 0x80000000u) != 0u;
    flags.pf = parity_even(static_cast<std::uint8_t>(r));
}

static inline void compute_add64(word a, word b, word& dst, Flags& flags) {
    const word r = a + b;
    dst = r;
    flags.cf = r < a;
    flags.zf = r == 0ull;
    flags.sf = (r >> 63) != 0ull;
    flags.of = (((a ^ r) & (b ^ r)) >> 63) != 0ull;
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


def flat_address(ins, clobbered_ptrs, expected_bytes):
    """Lower a flat [base + disp] operand rooted in an unclobbered argument GPR64."""
    if ins.is_ip_rel_memory_operand or ins.memory_base in (R.RIP, R.EIP):
        raise ValueError(f"RIP-relative memory operand at RVA {ins.ip:#x}: {ins}")
    if ins.memory_index != R.NONE:
        raise ValueError(f"SIB index memory operand at RVA {ins.ip:#x}: {ins}")
    base = ins.memory_base
    if base == R.NONE or not RegisterExt.is_gpr64(base) or base not in ARGUMENTS:
        raise ValueError(f"bad base register in memory operand at RVA {ins.ip:#x}: {ins}")
    if base in clobbered_ptrs:
        raise ValueError(f"memory operand uses clobbered pointer {ARGUMENTS[base]} at RVA {ins.ip:#x}: {ins}")
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
    if kind == OpKind.IMMEDIATE16:
        return ins.immediate16 & 0xFFFF
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
    """Decode and lift one bounded straight-line segment-inspection leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    defined_gpr = {"rcx", "rdx", "r8"}
    used_scratch = set()
    clobbered_ptrs = set()
    defined_flags = set()
    statements = []
    returned = False

    def note_scratch(reg_name):
        if reg_name in ("r9", "r10", "r11"):
            used_scratch.add(reg_name)

    def read_reg(op_index, ins, expected_width=None):
        reg_name, width = gpr_info(ins.op_register(op_index), ins)
        if expected_width is not None and width != expected_width:
            raise ValueError(f"unexpected GPR width {width} at RVA {ins.ip:#x}: {ins}")
        if reg_name not in defined_gpr:
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        note_scratch(reg_name)
        return reg_name, width

    def check_dst_reg(op_index, ins, expected_width=None, require_defined=False):
        reg_name, width = gpr_info(ins.op_register(op_index), ins)
        if expected_width is not None and width != expected_width:
            raise ValueError(f"unexpected GPR destination width {width} at RVA {ins.ip:#x}: {ins}")
        if require_defined and reg_name not in defined_gpr:
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        note_scratch(reg_name)
        return reg_name, width

    def commit_write(reg_name, width):
        if reg_name == "rcx":
            clobbered_ptrs.add(R.RCX)
        elif reg_name == "rdx":
            clobbered_ptrs.add(R.RDX)
        elif reg_name == "r8":
            clobbered_ptrs.add(R.R8)
        if width >= 32:
            defined_gpr.add(reg_name)

    def read_selector(op_index, ins):
        kind = ins.op_kind(op_index)
        if kind == OpKind.REGISTER:
            reg_name, _ = read_reg(op_index, ins)
            return f"static_cast<std::uint16_t>({reg_name} & 0xFFFFull)"
        if kind == OpKind.MEMORY:
            addr = flat_address(ins, clobbered_ptrs, 2)
            sel_var = f"sel_{ins.ip:x}"
            statements.append(f"const std::uint16_t {sel_var} = load_always<std::uint16_t>({addr});")
            return sel_var
        raise ValueError(f"unsupported selector operand at RVA {ins.ip:#x}: {ins}")

    def read_rm(op_index, ins, width):
        kind = ins.op_kind(op_index)
        if kind == OpKind.REGISTER:
            reg_name, _ = read_reg(op_index, ins, width)
            mask = {8: "0xFFull", 16: "0xFFFFull", 32: "0xFFFFFFFFull"}.get(width)
            return f"({reg_name} & {mask})" if mask else reg_name
        if kind == OpKind.MEMORY:
            addr = flat_address(ins, clobbered_ptrs, width // 8)
            ctype = {8: "std::uint8_t", 16: "std::uint16_t", 32: "std::uint32_t", 64: "std::uint64_t"}[width]
            load_var = f"ld_{ins.ip:x}"
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
        if encoding == EncodingKind.VEX:
            raise ValueError(f"unsupported VEX instruction at RVA {ins.ip:#x}: {ins}")
        if encoding == EncodingKind.EVEX or ins.op_mask != R.NONE or ins.zeroing_masking or ins.is_broadcast:
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        if encoding == EncodingKind.XOP:
            raise ValueError(f"unsupported XOP instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code != Code.RETNQ and ins.flow_control != FlowControl.NEXT:
            raise ValueError(f"non-leaf branch or call at RVA {ins.ip:#x}: {ins}")

        code = ins.code
        if code == Code.RETNQ:
            if "rax" not in defined_gpr:
                raise ValueError(f"return with undefined RAX at RVA {ins.ip:#x}")
            statements.append("return rax;")
            returned = True
        elif code in LAR_CODES or code in LSL_CODES:
            is_lar = code in LAR_CODES
            width = LAR_CODES[code] if is_lar else LSL_CODES[code]
            helper = "seg_lar" if is_lar else "seg_lsl"
            # On ZF=0 all 64 bits of dst stay untouched, so dst must already be defined.
            dst, _ = check_dst_reg(0, ins, width, require_defined=True)
            sel = read_selector(1, ins)
            val_var = f"val_{ins.ip:x}"
            if width == 16:
                update = f"{dst} = ({dst} & ~0xFFFFull) | (static_cast<word>({val_var}) & 0xFFFFull);"
            else:
                update = f"{dst} = static_cast<word>({val_var});"
            statements.append(
                f"{{ std::uint32_t {val_var} = 0u; "
                f"if ({helper}({sel}, {val_var})) {{ {update} flags.zf = true; }} "
                f"else {{ flags.zf = false; }} }}"
            )
            commit_write(dst, width)
            defined_flags.add("zf")
        elif code in VERR_CODES or code in VERW_CODES:
            helper = "seg_verr" if code in VERR_CODES else "seg_verw"
            sel = read_selector(0, ins)
            statements.append(f"flags.zf = {helper}({sel});")
            defined_flags.add("zf")
        elif code in MOV_WIDTHS:
            width = MOV_WIDTHS[code]
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src_expr = read_rm(1, ins, width)
                dst, _ = check_dst_reg(0, ins, width, require_defined=(width < 32))
                statements.append(f"{dst} = update_gpr_width({dst}, {src_expr}, {width}u);")
                commit_write(dst, width)
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst, _ = check_dst_reg(0, ins, width, require_defined=(width < 32))
                src_expr = read_rm(1, ins, width)
                statements.append(f"{dst} = update_gpr_width({dst}, {src_expr}, {width}u);")
                commit_write(dst, width)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src_expr = read_rm(1, ins, width)
                addr = flat_address(ins, clobbered_ptrs, width // 8)
                ctype = {8: "std::uint8_t", 16: "std::uint16_t", 32: "std::uint32_t", 64: "std::uint64_t"}[width]
                statements.append(f"store_always<{ctype}>({addr}, static_cast<{ctype}>({src_expr}));")
            elif ins.op0_kind == OpKind.REGISTER:
                imm = _immediate_value(ins, 1, width)
                dst, _ = check_dst_reg(0, ins, width, require_defined=(width < 32))
                statements.append(f"{dst} = update_gpr_width({dst}, {imm:#x}ull, {width}u);")
                commit_write(dst, width)
            elif ins.op0_kind == OpKind.MEMORY:
                imm = _immediate_value(ins, 1, width)
                addr = flat_address(ins, clobbered_ptrs, width // 8)
                ctype = {8: "std::uint8_t", 16: "std::uint16_t", 32: "std::uint32_t", 64: "std::uint64_t"}[width]
                statements.append(f"store_always<{ctype}>({addr}, static_cast<{ctype}>({imm:#x}ull));")
            else:
                raise ValueError(f"unsupported mov operands at RVA {ins.ip:#x}: {ins}")
        elif code in MOVZX_SPECS:
            dst_width, src_width = MOVZX_SPECS[code]
            src_expr = read_rm(1, ins, src_width)
            dst, _ = check_dst_reg(0, ins, dst_width, require_defined=False)
            statements.append(f"{dst} = {src_expr};")
            commit_write(dst, dst_width)
        elif code in XOR_WIDTHS or code in OR_WIDTHS:
            is_xor = code in XOR_WIDTHS
            width = XOR_WIDTHS[code] if is_xor else OR_WIDTHS[code]
            if (is_xor and ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER
                    and ins.op0_register == ins.op1_register):
                dst, _ = check_dst_reg(0, ins, width, require_defined=False)
                statements.append(f"{dst} = 0ull;")
                statements.append("flags.cf = false; flags.of = false; flags.sf = false; flags.zf = true; flags.pf = true;")
                commit_write(dst, width)
            else:
                if ins.op0_kind != OpKind.REGISTER:
                    raise ValueError(f"unsupported {'xor' if is_xor else 'or'} destination at RVA {ins.ip:#x}: {ins}")
                dst, _ = check_dst_reg(0, ins, width, require_defined=True)
                lhs, _ = read_reg(0, ins, width)
                if ins.op1_kind in (OpKind.REGISTER, OpKind.MEMORY):
                    rhs = read_rm(1, ins, width)
                else:
                    rhs = f"{_immediate_value(ins, 1, width):#x}ull"
                op_sym = "^" if is_xor else "|"
                statements.append(f"{dst} = update_gpr_width({dst}, {lhs} {op_sym} {rhs}, {width}u);")
                statements.append(f"compute_logic_flags({dst}, {width}u, flags);")
                commit_write(dst, width)
            defined_flags.update(("cf", "zf", "sf", "of", "pf"))
        elif code in ADD_WIDTHS:
            width = ADD_WIDTHS[code]
            if ins.op0_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported add destination at RVA {ins.ip:#x}: {ins}")
            dst, _ = check_dst_reg(0, ins, width, require_defined=True)
            lhs, _ = read_reg(0, ins, width)
            if ins.op1_kind in (OpKind.REGISTER, OpKind.MEMORY):
                rhs = read_rm(1, ins, width)
            else:
                rhs = f"{_immediate_value(ins, 1, width):#x}ull"
            statements.append(f"compute_add{width}({lhs}, {rhs}, {dst}, flags);")
            commit_write(dst, width)
            defined_flags.update(("cf", "zf", "sf", "of", "pf"))
        elif code in SETCC_SPECS:
            flag_name, cond_expr = SETCC_SPECS[code]
            if flag_name not in defined_flags:
                raise ValueError(f"undefined flags read by {ins} at RVA {ins.ip:#x}")
            if ins.op0_kind == OpKind.REGISTER:
                dst, _ = check_dst_reg(0, ins, 8, require_defined=True)
                statements.append(f"{dst} = update_gpr_width({dst}, ({cond_expr}) ? 1ull : 0ull, 8u);")
                commit_write(dst, 8)
            elif ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins, clobbered_ptrs, 1)
                statements.append(
                    f"store_always<std::uint8_t>({addr}, static_cast<std::uint8_t>(({cond_expr}) ? 1u : 0u));"
                )
            else:
                raise ValueError(f"unsupported setcc operand at RVA {ins.ip:#x}: {ins}")
        elif code in CMOVCC_SPECS:
            width, flag_name, cond_expr = CMOVCC_SPECS[code]
            if flag_name not in defined_flags:
                raise ValueError(f"undefined flags read by {ins} at RVA {ins.ip:#x}")
            dst, _ = check_dst_reg(0, ins, width, require_defined=True)
            src_expr = read_rm(1, ins, width)
            if width == 32:
                statements.append(
                    f"if ({cond_expr}) {{ {dst} = update_gpr_width({dst}, {src_expr}, 32u); }} "
                    f"else {{ {dst} &= 0xFFFFFFFFull; }}"
                )
            else:
                statements.append(f"if ({cond_expr}) {{ {dst} = update_gpr_width({dst}, {src_expr}, {width}u); }}")
            commit_write(dst, width)
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
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=("Use only for the fixed Windows x64 CPL=3 GDT profile. The generated code "
                "does not read the live GDT/LDT or model other selector layouts."),
    )
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
