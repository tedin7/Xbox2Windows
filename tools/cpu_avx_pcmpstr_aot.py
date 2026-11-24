"""Lift bounded Windows x64 leaf functions with AVX VEX.128 packed string compares to C++.

Each export uses the leaf ABI std::uint64_t fn(word rcx, word rdx, word r8),
where rcx, rdx and r8 carry flat host pointers to caller-owned buffers and
RAX carries the scalar return value (packed EFLAGS or index). YMM0..YMM5 are
modeled as portable 256-bit value structs (four 64-bit lanes); every VEX.128
write to XMM0..XMM5 (including implicit XMM0 writes from vpcmpestrm and
vpcmpistrm) zeroes bits 128..255 of the corresponding YMM register.

Supported AVX instructions (docs/coverage/x64_zen2.tsv group AVX):
  - Packed string compares (VEX.128, W=0, xmm or flat m128 src2):
    vpcmpestri, vpcmpestrm, vpcmpistri, vpcmpistrm
  - 64-bit explicit-length packed string compares (VEX.128, W=1, xmm or flat
    m128 src2): vpcmpestri64, vpcmpestrm64. The lengths are the full signed
    64-bit RAX and RDX; W=0 forms use only the signed low 32 bits.

Transport and ABI glue:
  - vmovdqu (128/256-bit flat load, store and register copy)
  - vzeroupper
  - flat 32/64-bit load/store/reg mov and immediate mov for volatile GPRs
  - reg-reg / imm8 or/xor, reg imm8 shl, setcc (setb/sete/sets/seto/setp),
    movzx r32, r8, and terminal ret.

VEX.256 (L=1), EVEX encodings, legacy SSE4.2 pcmp*str*, nonvolatile registers
(XMM6..15 / YMM6..15, RBX/RBP/RSI/RDI/R12..R15/RSP), writes to r8,
unsupported addressing modes, prefixes, branches, and all other instructions
are rejected.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import (
    Code,
    Decoder,
    EncodingKind,
    MemorySizeExt,
    Mnemonic as M,
    OpKind,
    Register as R,
    RegisterExt,
)

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}

VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
VOLATILE_YMM = {getattr(R, f"YMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = (
    {getattr(R, f"XMM{i}") for i in range(6, 16)}
    | {getattr(R, f"YMM{i}") for i in range(6, 16)}
)

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
VOLATILE_GPR8 = {
    R.AL: "rax",
    R.CL: "rcx",
    R.DL: "rdx",
    R.R8L: "r8",
    R.R9L: "r9",
    R.R10L: "r10",
    R.R11L: "r11",
}
NONVOLATILE_GPR = {
    R.RBX, R.EBX, R.BL, R.BH,
    R.RBP, R.EBP, R.BPL,
    R.RSI, R.ESI, R.SIL,
    R.RDI, R.EDI, R.DIL,
    R.RSP, R.ESP, R.SPL,
    R.R12, R.R12D, R.R12L,
    R.R13, R.R13D, R.R13L,
    R.R14, R.R14D, R.R14L,
    R.R15, R.R15D, R.R15L,
}

# (is_explicit, is_mask, is_len64)
PCMP_SPECS = {
    Code.VEX_VPCMPESTRI_XMM_XMMM128_IMM8: (True, False, False),
    Code.VEX_VPCMPESTRM_XMM_XMMM128_IMM8: (True, True, False),
    Code.VEX_VPCMPISTRI_XMM_XMMM128_IMM8: (False, False, False),
    Code.VEX_VPCMPISTRM_XMM_XMMM128_IMM8: (False, True, False),
    Code.VEX_VPCMPESTRI64_XMM_XMMM128_IMM8: (True, False, True),
    Code.VEX_VPCMPESTRM64_XMM_XMMM128_IMM8: (True, True, True),
}

LEGACY_PCMP_CODES = {
    Code.PCMPESTRI_XMM_XMMM128_IMM8,
    Code.PCMPESTRI64_XMM_XMMM128_IMM8,
    Code.PCMPESTRM_XMM_XMMM128_IMM8,
    Code.PCMPESTRM64_XMM_XMMM128_IMM8,
    Code.PCMPISTRI_XMM_XMMM128_IMM8,
    Code.PCMPISTRM_XMM_XMMM128_IMM8,
}

SETCC_FIELDS = {
    Code.SETB_RM8: "cf",
    Code.SETE_RM8: "zf",
    Code.SETS_RM8: "sf",
    Code.SETO_RM8: "of",
    Code.SETP_RM8: "pf",
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.VEX_VMOVDQU_XMM_XMMM128,
    Code.VEX_VMOVDQU_XMMM128_XMM,
    Code.VEX_VMOVDQU_YMM_YMMM256,
    Code.VEX_VMOVDQU_YMMM256_YMM,
    Code.VEX_VZEROUPPER,
    Code.MOV_R32_RM32,
    Code.MOV_R64_RM64,
    Code.MOV_RM32_R32,
    Code.MOV_RM64_R64,
    Code.MOV_R32_IMM32,
    Code.MOV_RM64_IMM32,
    Code.OR_R32_RM32,
    Code.OR_RM32_R32,
    Code.OR_R64_RM64,
    Code.OR_RM64_R64,
    Code.OR_RM32_IMM8,
    Code.OR_RM64_IMM8,
    Code.XOR_R32_RM32,
    Code.XOR_RM32_R32,
    Code.XOR_R64_RM64,
    Code.XOR_RM64_R64,
    Code.SHL_RM32_1,
    Code.SHL_RM64_1,
    Code.SHL_RM32_IMM8,
    Code.SHL_RM64_IMM8,
    Code.MOVZX_R32_RM8,
    *SETCC_FIELDS,
    *PCMP_SPECS,
}


PRELUDE = """#include <cstdint>
#include <cstring>
#include <intrin.h>

extern "C" __declspec(dllexport) std::uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}

using word = std::uint64_t;

struct Vec256 {
    std::uint64_t q[4];
};

struct Flags {
    bool cf = false;
    bool zf = false;
    bool sf = false;
    bool of = false;
    bool pf = false;
    bool af = false;
};

struct PcmpResult {
    std::uint16_t mask;
    std::uint32_t index;
    Flags flags;
};

static std::uint32_t load32(word address) {
    return *reinterpret_cast<const volatile std::uint32_t __unaligned*>(address);
}

static std::uint64_t load64(word address) {
    return *reinterpret_cast<const volatile std::uint64_t __unaligned*>(address);
}

template<unsigned Bytes> static Vec256 load_vector(word address) {
    // The guest reads the full operand even if the result is discarded.
    Vec256 v{};
    const auto* src = reinterpret_cast<const volatile std::uint8_t*>(address);
    auto* dst = reinterpret_cast<std::uint8_t*>(v.q);
    for (unsigned i = 0; i < Bytes; ++i) dst[i] = src[i];
    return v;
}

static Vec256 load128(word address) { return load_vector<16>(address); }
static Vec256 load256(word address) { return load_vector<32>(address); }

static void store32(word address, std::uint32_t val) {
    *reinterpret_cast<volatile std::uint32_t __unaligned*>(address) = val;
}

static void store64(word address, std::uint64_t val) {
    *reinterpret_cast<volatile std::uint64_t __unaligned*>(address) = val;
}

static void store128(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
}

static void store256(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 32);
}

static Vec256 copy128(const Vec256& v) {
    return {{v.q[0], v.q[1], 0ull, 0ull}};
}

static bool even_parity8(std::uint8_t byte_val) {
    unsigned parity = byte_val;
    parity ^= parity >> 4;
    parity ^= parity >> 2;
    parity ^= parity >> 1;
    return (parity & 1u) == 0u;
}

static Flags logic_flags(std::uint64_t result, bool is64) {
    const std::uint64_t masked = is64 ? result : (result & 0xffffffffull);
    const std::uint64_t sign_bit = is64 ? (masked >> 63) : ((masked >> 31) & 1ull);
    return {false, masked == 0ull, sign_bit != 0ull, false, even_parity8(static_cast<std::uint8_t>(masked & 0xffull)), false};
}

static std::uint32_t pcmp_explicit_len(std::int64_t raw, std::uint32_t limit) {
    const std::uint64_t mag = raw < 0 ? (0ull - static_cast<std::uint64_t>(raw)) : static_cast<std::uint64_t>(raw);
    return mag < static_cast<std::uint64_t>(limit) ? static_cast<std::uint32_t>(mag) : limit;
}

static std::int32_t pcmp_elem(const std::uint8_t p[16], std::uint32_t idx, bool is_word, bool is_signed) {
    if (is_word) {
        std::uint16_t u = 0u;
        std::memcpy(&u, p + idx * 2u, 2);
        return is_signed ? static_cast<std::int32_t>(static_cast<std::int16_t>(u)) : static_cast<std::int32_t>(u);
    }
    return is_signed ? static_cast<std::int32_t>(static_cast<std::int8_t>(p[idx])) : static_cast<std::int32_t>(p[idx]);
}

static PcmpResult pcmp_str(const Vec256& a_vec, const Vec256& b_vec,
                           std::uint8_t imm, bool is_explicit, std::int64_t raw_la, std::int64_t raw_lb) {
    std::uint8_t a[16]{};
    std::uint8_t b[16]{};
    std::memcpy(a, a_vec.q, 16);
    std::memcpy(b, b_vec.q, 16);
    const bool is_word = (imm & 1u) != 0u;
    const bool is_signed = (imm & 2u) != 0u;
    const std::uint32_t limit = is_word ? 8u : 16u;
    const std::uint32_t agg = (imm >> 2) & 3u;
    const std::uint32_t pol = (imm >> 4) & 3u;
    const bool high_sel = (imm & 0x40u) != 0u;
    std::uint32_t len1 = 0u;
    std::uint32_t len2 = 0u;
    if (is_explicit) {
        len1 = pcmp_explicit_len(raw_la, limit);
        len2 = pcmp_explicit_len(raw_lb, limit);
    } else {
        while (len1 < limit && pcmp_elem(a, len1, is_word, is_signed) != 0) ++len1;
        while (len2 < limit && pcmp_elem(b, len2, is_word, is_signed) != 0) ++len2;
    }
    std::uint16_t int_res1 = 0u;
    for (std::uint32_t i = 0u; i < limit; ++i) {
        bool bit = false;
        if (agg == 0u) {
            if (i < len2) {
                const std::int32_t bv = pcmp_elem(b, i, is_word, is_signed);
                for (std::uint32_t j = 0u; j < len1; ++j) {
                    if (pcmp_elem(a, j, is_word, is_signed) == bv) { bit = true; break; }
                }
            }
        } else if (agg == 1u) {
            if (i < len2) {
                const std::int32_t bv = pcmp_elem(b, i, is_word, is_signed);
                for (std::uint32_t j = 0u; j + 1u < len1; j += 2u) {
                    if (bv >= pcmp_elem(a, j, is_word, is_signed) &&
                        bv <= pcmp_elem(a, j + 1u, is_word, is_signed)) { bit = true; break; }
                }
            }
        } else if (agg == 2u) {
            if (i < len1 && i < len2) bit = pcmp_elem(a, i, is_word, is_signed) == pcmp_elem(b, i, is_word, is_signed);
            else if (i >= len1 && i >= len2) bit = true;
        } else {
            bit = true;
            for (std::uint32_t j = 0u; j < len1 && i + j < limit; ++j) {
                if (i + j >= len2 || pcmp_elem(a, j, is_word, is_signed) != pcmp_elem(b, i + j, is_word, is_signed)) {
                    bit = false;
                    break;
                }
            }
        }
        if (bit) int_res1 = static_cast<std::uint16_t>(int_res1 | (1u << i));
    }
    const std::uint16_t all_mask = static_cast<std::uint16_t>((1u << limit) - 1u);
    std::uint16_t int_res2 = int_res1;
    if (pol == 1u) int_res2 = static_cast<std::uint16_t>(int_res1 ^ all_mask);
    else if (pol == 3u) int_res2 = static_cast<std::uint16_t>(int_res1 ^ static_cast<std::uint16_t>((1u << len2) - 1u));
    std::uint32_t index = limit;
    if (int_res2 != 0u) {
        if (!high_sel) { index = 0u; while (((int_res2 >> index) & 1u) == 0u) ++index; }
        else { index = limit - 1u; while (((int_res2 >> index) & 1u) == 0u) --index; }
    }
    Flags f{int_res2 != 0u, len2 < limit, len1 < limit, (int_res2 & 1u) != 0u, false, false};
    return {int_res2, index, f};
}

static Vec256 pcmp_write_mask(std::uint16_t mask, std::uint8_t imm) {
    std::uint8_t out[16]{};
    if ((imm & 0x40u) == 0u) {
        std::memcpy(out, &mask, 2);
    } else if ((imm & 1u) == 0u) {
        for (std::uint32_t k = 0u; k < 16u; ++k) out[k] = ((mask >> k) & 1u) ? 0xffu : 0x00u;
    } else {
        for (std::uint32_t k = 0u; k < 8u; ++k) {
            const std::uint16_t v = ((mask >> k) & 1u) ? 0xffffu : 0x0000u;
            std::memcpy(out + 2u * k, &v, 2);
        }
    }
    Vec256 v{};
    std::memcpy(v.q, out, 16);
    return v;
}
"""


def vec_index(register, ins):
    """Map XMM0..XMM5 or YMM0..YMM5 to (index, is_ymm), rejecting all others."""
    if register in VOLATILE_XMM:
        return VOLATILE_XMM[register], False
    if register in VOLATILE_YMM:
        return VOLATILE_YMM[register], True
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def gpr_info(register, ins):
    """Map volatile GPR (64/32/8-bit low) to (name, width_bits), rejecting nonvolatile/high-byte."""
    if register in VOLATILE_GPR64:
        return VOLATILE_GPR64[register], 64
    if register in VOLATILE_GPR32:
        return VOLATILE_GPR32[register], 32
    if register in VOLATILE_GPR8:
        return VOLATILE_GPR8[register], 8
    if register in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported GPR {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, clobbered_ptrs):
    """Lower a flat [base + disp] pointer operand rooted in an unclobbered argument GPR64."""
    base = ins.memory_base
    if (
        ins.is_ip_rel_memory_operand
        or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
        or ins.memory_index != R.NONE
        or not RegisterExt.is_gpr64(base)
        or base not in ARGUMENTS
    ):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    if base in clobbered_ptrs:
        raise ValueError(f"memory operand uses clobbered pointer {ARGUMENTS[base]} at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    name = ARGUMENTS[base]
    return f"({name} + {disp:#x}ull)" if disp else name


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line AVX VEX.128 packed string compare leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used_vec = set()
    defined_vec = set()
    defined_gpr = {"rcx", "rdx", "r8"}
    defined_gpr8 = set()
    clobbered_ptrs = set()
    flags_defined = False
    statements = []
    returned = False

    def read_vec(op_index, ins, allow_ymm=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if wide and not allow_ymm:
            raise ValueError(f"unsupported VEX.256 (L=1) encoding at RVA {ins.ip:#x}: {ins}")
        if idx not in defined_vec:
            raise ValueError(f"undefined vector register ymm{idx} at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return f"ymm{idx}", wide

    def write_vec(op_index, ins, allow_ymm=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if wide and not allow_ymm:
            raise ValueError(f"unsupported VEX.256 (L=1) encoding at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return idx, f"ymm{idx}", wide

    def mark_gpr_written(reg_name, width):
        if reg_name == "r8":
            raise ValueError("unsupported write to output buffer pointer r8")
        if reg_name == "rcx":
            clobbered_ptrs.add(R.RCX)
        elif reg_name == "rdx":
            clobbered_ptrs.add(R.RDX)
        if width >= 32:
            defined_gpr.add(reg_name)
        defined_gpr8.add(reg_name)

    def read_gpr(op_index, ins, expected_width):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR operand {op_index} at RVA {ins.ip:#x}: {ins}")
        reg_name, width = gpr_info(ins.op_register(op_index), ins)
        if width != expected_width:
            raise ValueError(f"unexpected GPR width {width} at RVA {ins.ip:#x}: {ins}")
        ready = defined_gpr8 if width == 8 else defined_gpr
        if reg_name not in ready:
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        return reg_name

    def write_gpr(op_index, ins, expected_width):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR destination at RVA {ins.ip:#x}: {ins}")
        reg_name, width = gpr_info(ins.op_register(op_index), ins)
        if width != expected_width:
            raise ValueError(f"unexpected GPR destination width {width} at RVA {ins.ip:#x}: {ins}")
        if reg_name == "r8":
            raise ValueError(f"unsupported write to output buffer pointer r8 at RVA {ins.ip:#x}: {ins}")
        return reg_name

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        op_code = ins.op_code()
        if (
            op_code.encoding == EncodingKind.EVEX
            or ins.op_mask != R.NONE
            or ins.zeroing_masking
            or ins.is_broadcast
        ):
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in PCMP_SPECS and (
            op_code.encoding == EncodingKind.VEX and op_code.w == 1 and ins.mnemonic in (
                M.VPCMPESTRI, M.VPCMPESTRM, M.VPCMPISTRI, M.VPCMPISTRM
            )
        ):
            raise ValueError(f"unsupported VEX.W=1 64-bit packed string compare at RVA {ins.ip:#x}: {ins}")
        if ins.code in LEGACY_PCMP_CODES:
            raise ValueError(f"unsupported legacy SSE4.2 packed string compare at RVA {ins.ip:#x}: {ins}")
        if op_code.encoding == EncodingKind.VEX and op_code.l == 1 and ins.mnemonic != M.VMOVDQU:
            raise ValueError(f"unsupported VEX.256 (L=1) instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        mnemonic = ins.mnemonic
        if ins.code == Code.RETNQ:
            if "rax" not in defined_gpr:
                raise ValueError(f"return with undefined RAX at RVA {ins.ip:#x}")
            statements.append("return rax;")
            returned = True
        elif ins.code == Code.VEX_VZEROUPPER:
            for idx in sorted(defined_vec):
                statements.append(f"ymm{idx}.q[2] = 0ull; ymm{idx}.q[3] = 0ull;")
        elif mnemonic == M.VMOVDQU:
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst_idx, dst, wide = write_vec(0, ins, allow_ymm=True)
                addr = flat_address(ins, clobbered_ptrs)
                statements.append(f"{dst} = {'load256' if wide else 'load128'}({addr});")
                defined_vec.add(dst_idx)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src, wide = read_vec(1, ins, allow_ymm=True)
                addr = flat_address(ins, clobbered_ptrs)
                statements.append(f"{'store256' if wide else 'store128'}({addr}, {src});")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src, _ = read_vec(1, ins, allow_ymm=True)
                dst_idx, dst, wide = write_vec(0, ins, allow_ymm=True)
                statements.append(f"{dst} = {src};" if wide else f"{dst} = copy128({src});")
                defined_vec.add(dst_idx)
            else:
                raise ValueError(f"unsupported vmovdqu operands at RVA {ins.ip:#x}: {ins}")
        elif ins.code in PCMP_SPECS:
            is_explicit, is_mask, is_len64 = PCMP_SPECS[ins.code]
            src1, _ = read_vec(0, ins)
            if ins.op1_kind == OpKind.REGISTER:
                src2, _ = read_vec(1, ins)
            elif ins.op1_kind == OpKind.MEMORY:
                if MemorySizeExt.size(ins.memory_size) != 16:
                    raise ValueError(f"expected 128-bit memory operand at RVA {ins.ip:#x}: {ins}")
                src2 = f"load128({flat_address(ins, clobbered_ptrs)})"
            else:
                raise ValueError(f"unsupported vpcmp*str* operand at RVA {ins.ip:#x}: {ins}")
            imm = ins.immediate8 & 0xff
            if is_explicit:
                if "rax" not in defined_gpr or "rdx" not in defined_gpr:
                    raise ValueError(f"undefined explicit length register at RVA {ins.ip:#x}: {ins}")
                if is_len64:
                    la_expr = "static_cast<std::int64_t>(rax)"
                    lb_expr = "static_cast<std::int64_t>(rdx)"
                else:
                    la_expr = "static_cast<std::int64_t>(static_cast<std::int32_t>(static_cast<std::uint32_t>(rax)))"
                    lb_expr = "static_cast<std::int64_t>(static_cast<std::int32_t>(static_cast<std::uint32_t>(rdx)))"
            else:
                la_expr, lb_expr = "0", "0"
            if is_mask:
                used_vec.add(0)
                defined_vec.add(0)
                out_stmt = f"ymm0 = pcmp_write_mask(r.mask, {imm:#x}u);"
            else:
                mark_gpr_written("rcx", 64)
                out_stmt = "rcx = static_cast<word>(r.index);"
            statements.append(
                f"{{ const PcmpResult r = pcmp_str({src1}, {src2}, {imm:#x}u, "
                f"{'true' if is_explicit else 'false'}, {la_expr}, {lb_expr}); "
                f"{out_stmt} flags = r.flags; }}"
            )
            flags_defined = True
        elif mnemonic == M.MOV:
            width = 64 if ins.code in (Code.MOV_R64_RM64, Code.MOV_RM64_R64, Code.MOV_RM64_IMM32) else 32
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst = write_gpr(0, ins, width)
                addr = flat_address(ins, clobbered_ptrs)
                statements.append(f"{dst} = {'load64' if width == 64 else 'static_cast<word>(load32'}({addr}){'' if width == 64 else ')'};")
                mark_gpr_written(dst, width)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src = read_gpr(1, ins, width)
                addr = flat_address(ins, clobbered_ptrs)
                statements.append(f"store{width}({addr}, {'static_cast<std::uint32_t>(' + src + ')' if width == 32 else src});")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src = read_gpr(1, ins, width)
                dst = write_gpr(0, ins, width)
                statements.append(f"{dst} = {src} & 0xffffffffull;" if width == 32 else f"{dst} = {src};")
                mark_gpr_written(dst, width)
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind in (OpKind.IMMEDIATE32, OpKind.IMMEDIATE32TO64):
                dst = write_gpr(0, ins, width)
                imm = ins.immediate32to64 & 0xffffffffffffffff if width == 64 else ins.immediate32 & 0xffffffff
                statements.append(f"{dst} = {imm:#x}ull;")
                mark_gpr_written(dst, width)
            else:
                raise ValueError(f"unsupported mov operands at RVA {ins.ip:#x}: {ins}")
        elif mnemonic in (M.OR, M.XOR):
            width = 64 if ins.code in (
                Code.OR_R64_RM64, Code.OR_RM64_R64, Code.OR_RM64_IMM8, Code.XOR_R64_RM64, Code.XOR_RM64_R64
            ) else 32
            op_sym = "|" if mnemonic == M.OR else "^"
            dst = write_gpr(0, ins, width)
            if ins.op1_kind == OpKind.REGISTER:
                if mnemonic == M.XOR and ins.op0_register == ins.op1_register:
                    statements.append(f"{dst} = 0ull;")
                else:
                    lhs = read_gpr(0, ins, width)
                    rhs = read_gpr(1, ins, width)
                    expr = f"({lhs} {op_sym} {rhs})"
                    if width == 32:
                        expr += " & 0xffffffffull"
                    statements.append(f"{dst} = {expr};")
            elif ins.op1_kind in (OpKind.IMMEDIATE8TO32, OpKind.IMMEDIATE8TO64):
                lhs = read_gpr(0, ins, width)
                imm_val = (ins.immediate8to64 if width == 64 else ins.immediate8to32) & (
                    0xffffffffffffffff if width == 64 else 0xffffffff
                )
                expr = f"({lhs} {op_sym} {imm_val:#x}ull)"
                if width == 32:
                    expr += " & 0xffffffffull"
                statements.append(f"{dst} = {expr};")
            else:
                raise ValueError(f"unsupported {mnemonic} operands at RVA {ins.ip:#x}: {ins}")
            statements.append(f"flags = logic_flags({dst}, {'true' if width == 64 else 'false'});")
            mark_gpr_written(dst, width)
            flags_defined = True
        elif mnemonic == M.SHL:
            width = 64 if ins.code in (Code.SHL_RM64_IMM8, Code.SHL_RM64_1) else 32
            is_one = ins.code in (Code.SHL_RM32_1, Code.SHL_RM64_1)
            if ins.op0_kind != OpKind.REGISTER or (not is_one and ins.op1_kind != OpKind.IMMEDIATE8):
                raise ValueError(f"unsupported shl operands at RVA {ins.ip:#x}: {ins}")
            dst = write_gpr(0, ins, width)
            src = read_gpr(0, ins, width)
            count = 1 if is_one else (ins.immediate8 & (63 if width == 64 else 31))
            if width == 64:
                statements.append(f"{dst} = {src} << {count}u;")
            else:
                statements.append(f"{dst} = (static_cast<std::uint32_t>({src}) << {count}u) & 0xffffffffull;")
            mark_gpr_written(dst, width)
            if count != 0:
                flags_defined = False
        elif ins.code in SETCC_FIELDS:
            if not flags_defined:
                raise ValueError(f"undefined flags read by {ins} at RVA {ins.ip:#x}")
            dst = write_gpr(0, ins, 8)
            field = SETCC_FIELDS[ins.code]
            statements.append(f"{dst} = ({dst} & ~0xffull) | (flags.{field} ? 1ull : 0ull);")
            mark_gpr_written(dst, 8)
        elif ins.code == Code.MOVZX_R32_RM8:
            src = read_gpr(1, ins, 8)
            dst = write_gpr(0, ins, 32)
            statements.append(f"{dst} = {src} & 0xffull;")
            mark_gpr_written(dst, 32)
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec256 ymm{idx}{{}};" for idx in sorted(used_vec)]
    decls.append("    word rax = 0ull;")
    for gpr in ("r9", "r10", "r11"):
        if gpr in defined_gpr or gpr in defined_gpr8:
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
        source, meta = translate(args.image, args.exports.split(","))
        args.output.write_text(source, encoding="utf-8")
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x}: {count} instructions")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
