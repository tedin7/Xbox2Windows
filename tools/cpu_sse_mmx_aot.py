"""Lift bounded Windows x64 leaf functions with legacy SSE/SSE2/MMX conversion, shuffle and masked-store ops to C++.

Each export uses the leaf ABI void fn(word rcx, word rdx, word r8), where rcx,
rdx and r8 carry host pointers to caller-owned buffers (16-byte-aligned bases
when 16-byte-aligned memory operands are used; rcx and rdx are inputs, r8 is the
output). XMM0..XMM5 are modeled as portable 128-bit value structs (Vec128, two
64-bit lanes), MM0..MM7 are modeled as std::uint64_t mm[8], and volatile scratch
GPRs (rax, r9, r10, r11) carry 64-bit movq bridge values and RDI save/restore
slots. Only legacy encodings are accepted; VEX, EVEX and other non-legacy forms
reject.

Supported instructions (docs/coverage/x64_zen2.tsv groups SSE and SSE2):
  - cvtpi2ps: convert two int32_t in mm/m64 to float32 in xmm[63:0], preserving xmm[127:64]
  - cvtps2pi: convert two float32 in xmm[63:0]/m64 to int32_t in mm (round-to-nearest-even)
  - cvttps2pi: convert two float32 in xmm[63:0]/m64 to int32_t in mm (truncate toward zero)
  - pshufw: 16-bit word permutation of mm/m64 by imm8 into mm
  - pavgb, pavgw, pmulhuw, psadbw: packed unsigned byte/word averaging,
    unsigned high 16-bit multiply, and sum of absolute byte differences
    (both MMX mm/m64 with unaligned m64 allowed and SSE2 xmm/16B-aligned m128)
  - movntq: non-temporal 64-bit store from mm to [r8 + disp]
  - maskmovq: conditional 8-byte store from mm1 under mm2 MSB mask to implicit [rdi]
  - maskmovdqu: conditional 16-byte store from xmm1 under xmm2 MSB mask to implicit [rdi]
    (portable first slice for maskmovq/maskmovdqu: requires rdi save/restore via
    a volatile scratch GPR around mov rdi, r8 and writable ordinary memory for the
    full 8- or 16-byte destination; touches only bytes with (mask[i] & 0x80) != 0)

Transport/ABI glue: movq (between mm, scratch r64 rax/r9/r10/r11, and flat
[rcx|rdx|r8] memory), 64-bit mov for rdi save/set-to-r8/restore, movaps/movdqa
(16-byte aligned 128-bit load/store/reg copy), movdqu (unaligned 128-bit
load/store/reg copy), emms, and terminal ret.

Call with default MXCSR controls, an empty x87 stack with no pending unmasked
exceptions, and no live MMX values. The caller owns the buffers and observes
the output bytes after return. Floating-point status and physical x87 tag
updates, non-temporal caching hints, and implementation-dependent partial-page
or all-zero-mask fault behavior are outside this leaf profile.

MMX state policy:
  - mmx_active starts false and becomes true whenever any MM0..MM7 register is
    touched; ret rejects while mmx_active is true.
  - emms clears mmx_active and invalidates active MM register definitions
    unconditionally.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import Code, Decoder, EncodingKind, OpKind, Register as R, RegisterExt

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}
ARGUMENT_GPRS = {R.RCX, R.ECX, R.RDX, R.EDX, R.R8, R.R8D}
VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = {getattr(R, f"XMM{i}") for i in range(6, 16)}
MM_REGISTERS = {getattr(R, f"MM{i}"): i for i in range(8)}

SCRATCH_GPR64 = {R.RAX: "rax", R.R9: "r9", R.R10: "r10", R.R11: "r11"}
NONVOLATILE_GPR = {
    R.EBX, R.RBX, R.EBP, R.RBP, R.ESI, R.RSI, R.EDI, R.RDI, R.ESP, R.RSP,
    R.R12D, R.R12, R.R13D, R.R13, R.R14D, R.R14, R.R15D, R.R15,
}

CVT_PS2PI_HELPERS = {
    Code.CVTPS2PI_MM_XMMM64: "op_cvtps2pi",
    Code.CVTTPS2PI_MM_XMMM64: "op_cvttps2pi",
}

MM_PACKED_HELPERS = {
    Code.PAVGB_MM_MMM64: "op_pavgb64",
    Code.PAVGW_MM_MMM64: "op_pavgw64",
    Code.PMULHUW_MM_MMM64: "op_pmulhuw64",
    Code.PSADBW_MM_MMM64: "op_psadbw64",
}

XMM_PACKED_HELPERS = {
    Code.PAVGB_XMM_XMMM128: "op_pavgb128",
    Code.PAVGW_XMM_XMMM128: "op_pavgw128",
    Code.PMULHUW_XMM_XMMM128: "op_pmulhuw128",
    Code.PSADBW_XMM_XMMM128: "op_psadbw128",
}

LOAD_ALIGNED_VEC_CODES = {Code.MOVAPS_XMM_XMMM128, Code.MOVDQA_XMM_XMMM128}
STORE_ALIGNED_VEC_CODES = {Code.MOVAPS_XMMM128_XMM, Code.MOVDQA_XMMM128_XMM}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.EMMS,
    Code.CVTPI2PS_XMM_MMM64,
    *CVT_PS2PI_HELPERS,
    *MM_PACKED_HELPERS,
    *XMM_PACKED_HELPERS,
    Code.PSHUFW_MM_MMM64_IMM8,
    Code.MOVNTQ_M64_MM,
    Code.MASKMOVQ_RDI_MM_MM,
    Code.MASKMOVDQU_RDI_XMM_XMM,
    Code.MOV_R64_RM64,
    Code.MOV_RM64_R64,
    Code.MOVAPS_XMM_XMMM128,
    Code.MOVAPS_XMMM128_XMM,
    Code.MOVDQA_XMM_XMMM128,
    Code.MOVDQA_XMMM128_XMM,
    Code.MOVDQU_XMM_XMMM128,
    Code.MOVDQU_XMMM128_XMM,
    Code.MOVQ_MM_MMM64,
    Code.MOVQ_MMM64_MM,
    Code.MOVQ_MM_RM64,
    Code.MOVQ_RM64_MM,
}

PRELUDE = """#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

using word = std::uint64_t;

struct Vec128 {
    std::uint64_t q[2];
};

// Guest reads can fault even when the translated result is dead. Keep surrounding guest
// writes on their original side of the read when the compiler inlines these helpers.
static inline std::uint64_t load64(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto value = *reinterpret_cast<const volatile std::uint64_t __unaligned*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return value;
}

static inline void store64(word address, std::uint64_t val) {
    std::memcpy(reinterpret_cast<void*>(address), &val, 8);
}

static inline Vec128 load128(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    Vec128 value;
    std::memcpy(value.q, reinterpret_cast<const void*>(address), sizeof(value.q));
    const volatile Vec128 observed = value;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{observed.q[0], observed.q[1]}};
}

static inline void store128(word address, const Vec128& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
}

static inline float f_from_bits(std::uint32_t bits) {
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

static inline std::uint32_t bits_from_f(float v) {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &v, 4);
    return bits;
}

static inline std::uint64_t op_cvtpi2ps(std::uint64_t src) {
    const auto i0 = static_cast<std::int32_t>(static_cast<std::uint32_t>(src));
    const auto i1 = static_cast<std::int32_t>(static_cast<std::uint32_t>(src >> 32));
    const std::uint32_t f0 = bits_from_f(static_cast<float>(i0));
    const std::uint32_t f1 = bits_from_f(static_cast<float>(i1));
    return static_cast<std::uint64_t>(f0) | (static_cast<std::uint64_t>(f1) << 32);
}

// Default MXCSR round-to-nearest-even; NaN, infinity and out-of-int32_t range
// after rounding yield the 32-bit integer indefinite value (0x80000000).
// Widen to double before range-checking because float(INT32_MAX) rounds to 2^31.
static inline std::uint32_t cvt_round_f32_i32(std::uint32_t bits) {
    const float r = std::nearbyintf(f_from_bits(bits));
    const double rd = static_cast<double>(r);
    const bool valid = std::isfinite(r) && (rd >= -2147483648.0 && rd <= 2147483647.0);
    return valid
        ? static_cast<std::uint32_t>(static_cast<std::int32_t>(r))
        : 0x80000000u;
}

// Truncate toward zero; NaN, infinity and out-of-int32_t range yield 0x80000000.
static inline std::uint32_t cvt_trunc_f32_i32(std::uint32_t bits) {
    const float r = std::truncf(f_from_bits(bits));
    const double rd = static_cast<double>(r);
    const bool valid = std::isfinite(r) && (rd >= -2147483648.0 && rd <= 2147483647.0);
    return valid
        ? static_cast<std::uint32_t>(static_cast<std::int32_t>(r))
        : 0x80000000u;
}

static inline std::uint64_t op_cvtps2pi(std::uint64_t src) {
    const std::uint32_t lo = cvt_round_f32_i32(static_cast<std::uint32_t>(src));
    const std::uint32_t hi = cvt_round_f32_i32(static_cast<std::uint32_t>(src >> 32));
    return static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32);
}

static inline std::uint64_t op_cvttps2pi(std::uint64_t src) {
    const std::uint32_t lo = cvt_trunc_f32_i32(static_cast<std::uint32_t>(src));
    const std::uint32_t hi = cvt_trunc_f32_i32(static_cast<std::uint32_t>(src >> 32));
    return static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32);
}

static inline std::uint64_t op_pshufw(std::uint64_t src, std::uint8_t imm8) {
    std::uint64_t out = 0ull;
    for (unsigned i = 0; i < 4u; ++i) {
        const unsigned sel = (imm8 >> (i * 2u)) & 3u;
        const std::uint64_t w = (src >> (sel * 16u)) & 0xffffull;
        out |= w << (i * 16u);
    }
    return out;
}

static inline void op_maskmovq(word address, std::uint64_t src, std::uint64_t mask) {
    auto* dst = reinterpret_cast<volatile std::uint8_t*>(address);
    // Portable first slice: require [address..address+7] to be writable ordinary
    // memory and touch only bytes selected by mask MSB ((mask[i] & 0x80) != 0).
    for (unsigned i = 0; i < 8u; ++i) {
        const auto mask_byte = static_cast<std::uint8_t>(mask >> (i * 8u));
        if ((mask_byte & 0x80u) != 0u) {
            std::atomic_signal_fence(std::memory_order_seq_cst);
            dst[i] = static_cast<std::uint8_t>(src >> (i * 8u));
        }
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void op_maskmovdqu(word address, const Vec128& src, const Vec128& mask) {
    std::uint8_t src_bytes[16]{};
    std::uint8_t mask_bytes[16]{};
    std::memcpy(src_bytes, src.q, 16);
    std::memcpy(mask_bytes, mask.q, 16);
    auto* dst = reinterpret_cast<volatile std::uint8_t*>(address);
    // Portable first slice: require [address..address+15] to be writable ordinary
    // memory and touch only bytes selected by mask MSB ((mask[i] & 0x80) != 0).
    for (unsigned i = 0; i < 16u; ++i) {
        if ((mask_bytes[i] & 0x80u) != 0u) {
            std::atomic_signal_fence(std::memory_order_seq_cst);
            dst[i] = src_bytes[i];
        }
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline std::uint64_t op_pavgb64(std::uint64_t a, std::uint64_t b) {
    std::uint64_t out = 0ull;
    for (unsigned i = 0u; i < 8u; ++i) {
        const std::uint16_t ab = static_cast<std::uint8_t>(a >> (i * 8u));
        const std::uint16_t bb = static_cast<std::uint8_t>(b >> (i * 8u));
        const std::uint64_t avg = (ab + bb + 1u) >> 1u;
        out |= (avg & 0xFFull) << (i * 8u);
    }
    return out;
}

static inline std::uint64_t op_pavgw64(std::uint64_t a, std::uint64_t b) {
    std::uint64_t out = 0ull;
    for (unsigned i = 0u; i < 4u; ++i) {
        const std::uint32_t aw = static_cast<std::uint16_t>(a >> (i * 16u));
        const std::uint32_t bw = static_cast<std::uint16_t>(b >> (i * 16u));
        const std::uint64_t avg = (aw + bw + 1u) >> 1u;
        out |= (avg & 0xFFFFull) << (i * 16u);
    }
    return out;
}

static inline std::uint64_t op_pmulhuw64(std::uint64_t a, std::uint64_t b) {
    std::uint64_t out = 0ull;
    for (unsigned i = 0u; i < 4u; ++i) {
        const std::uint32_t aw = static_cast<std::uint16_t>(a >> (i * 16u));
        const std::uint32_t bw = static_cast<std::uint16_t>(b >> (i * 16u));
        const std::uint64_t prod = (aw * bw) >> 16u;
        out |= (prod & 0xFFFFull) << (i * 16u);
    }
    return out;
}

static inline std::uint64_t op_psadbw64(std::uint64_t a, std::uint64_t b) {
    std::uint32_t sum = 0u;
    for (unsigned i = 0u; i < 8u; ++i) {
        const unsigned ab = static_cast<std::uint8_t>(a >> (i * 8u));
        const unsigned bb = static_cast<std::uint8_t>(b >> (i * 8u));
        sum += ab >= bb ? (ab - bb) : (bb - ab);
    }
    return static_cast<std::uint64_t>(sum);
}

static inline Vec128 op_pavgb128(const Vec128& a, const Vec128& b) {
    return {{op_pavgb64(a.q[0], b.q[0]), op_pavgb64(a.q[1], b.q[1])}};
}

static inline Vec128 op_pavgw128(const Vec128& a, const Vec128& b) {
    return {{op_pavgw64(a.q[0], b.q[0]), op_pavgw64(a.q[1], b.q[1])}};
}

static inline Vec128 op_pmulhuw128(const Vec128& a, const Vec128& b) {
    return {{op_pmulhuw64(a.q[0], b.q[0]), op_pmulhuw64(a.q[1], b.q[1])}};
}

static inline Vec128 op_psadbw128(const Vec128& a, const Vec128& b) {
    return {{op_psadbw64(a.q[0], b.q[0]), op_psadbw64(a.q[1], b.q[1])}};
}
"""


def vec_index(register, ins):
    """Map XMM0..XMM5 to an index, rejecting nonvolatile and unsupported registers."""
    if register in VOLATILE_XMM:
        return VOLATILE_XMM[register]
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def mm_index(register, ins):
    """Map MM0..MM7 to an index, rejecting all others."""
    if register in MM_REGISTERS:
        return MM_REGISTERS[register]
    raise ValueError(f"unsupported MMX register {register} at RVA {ins.ip:#x}: {ins}")


def gpr64_scratch(register, ins):
    """Map volatile scratch GPR64 (rax, r9, r10, r11) to name, rejecting others."""
    if register in SCRATCH_GPR64:
        return SCRATCH_GPR64[register]
    if register in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {register} at RVA {ins.ip:#x}: {ins}")
    if register in ARGUMENT_GPRS:
        raise ValueError(f"unsupported argument GPR operand {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported GPR {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, *, alignment=1, store=False):
    """Lower a flat [base + disp] pointer operand rooted in an argument GPR64."""
    base = ins.memory_base
    if (ins.is_ip_rel_memory_operand or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
            or ins.memory_index != R.NONE or not RegisterExt.is_gpr64(base)):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    if base in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR base {base} in memory operand at RVA {ins.ip:#x}: {ins}")
    if base not in ARGUMENTS:
        raise ValueError(f"unsupported memory base {base} at RVA {ins.ip:#x}: {ins}")
    if store and base != R.R8:
        raise ValueError(f"non-r8 store memory operand at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    if disp % alignment:
        raise ValueError(f"unaligned {alignment}-byte memory operand at RVA {ins.ip:#x}: {ins}")
    name = ARGUMENTS[base]
    return f"({name} + {disp:#x}ull)" if disp else name


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line legacy SSE/MMX leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used_xmm = set()
    defined_xmm = set()
    used_mm = False
    defined_mm = set()
    used_gpr = set()
    defined_gpr = set()
    # Track caller's nonvolatile RDI across save/set-to-r8/restore for maskmovq/maskmovdqu.
    caller_rdi_holders = {"rdi"}
    rdi_expr = None
    mmx_active = False
    statements = []
    returned = False

    def read_vec(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(ins.op_register(op_index), ins)
        if idx not in defined_xmm:
            raise ValueError(f"undefined vector register xmm{idx} at RVA {ins.ip:#x}: {ins}")
        used_xmm.add(idx)
        return f"xmm{idx}"

    def write_vec(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(ins.op_register(op_index), ins)
        used_xmm.add(idx)
        return idx, f"xmm{idx}"

    def read_mm(op_index, ins):
        nonlocal used_mm, mmx_active
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected MMX register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx = mm_index(ins.op_register(op_index), ins)
        if idx not in defined_mm:
            raise ValueError(f"undefined MMX register mm{idx} at RVA {ins.ip:#x}: {ins}")
        used_mm = True
        mmx_active = True
        return f"mm[{idx}]"

    def write_mm(op_index, ins):
        nonlocal used_mm, mmx_active
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected MMX register destination at RVA {ins.ip:#x}: {ins}")
        idx = mm_index(ins.op_register(op_index), ins)
        used_mm = True
        mmx_active = True
        return idx, f"mm[{idx}]"

    def read_gpr64(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR operand {op_index} at RVA {ins.ip:#x}: {ins}")
        reg_name = gpr64_scratch(ins.op_register(op_index), ins)
        if reg_name not in defined_gpr:
            if reg_name in caller_rdi_holders:
                raise ValueError(f"GPR {reg_name} holds saved rdi at RVA {ins.ip:#x}: {ins}")
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        used_gpr.add(reg_name)
        return reg_name

    def write_gpr64(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR destination at RVA {ins.ip:#x}: {ins}")
        reg_name = gpr64_scratch(ins.op_register(op_index), ins)
        used_gpr.add(reg_name)
        return reg_name

    def vec_src128(op_index, ins, *, alignment):
        if ins.op_kind(op_index) == OpKind.MEMORY:
            return f"load128({flat_address(ins, alignment=alignment)})"
        return read_vec(op_index, ins)

    def mm_src64(op_index, ins):
        if ins.op_kind(op_index) == OpKind.MEMORY:
            return f"load64({flat_address(ins)})"
        return read_mm(op_index, ins)

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if (ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix
                or ins.has_segment_prefix or ins.has_xacquire_prefix or ins.has_xrelease_prefix):
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        if ins.op_code().encoding != EncodingKind.LEGACY:
            raise ValueError(f"non-legacy encoding at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            if mmx_active:
                raise ValueError(f"missing emms before ret with active MMX state at RVA {ins.ip:#x}: {ins}")
            if "rdi" not in caller_rdi_holders:
                raise ValueError(f"unrestored nonvolatile GPR rdi at ret RVA {ins.ip:#x}: {ins}")
            statements.append("return;")
            returned = True
        elif ins.code == Code.EMMS:
            mmx_active = False
            defined_mm.clear()
            statements.append("/* emms */")
        elif ins.code == Code.CVTPI2PS_XMM_MMM64:
            read_vec(0, ins)
            src = mm_src64(1, ins)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name}.q[0] = op_cvtpi2ps({src});")
            defined_xmm.add(dst_idx)
        elif ins.code in CVT_PS2PI_HELPERS:
            if ins.op1_kind == OpKind.MEMORY:
                src = f"load64({flat_address(ins)})"
            else:
                src = f"{read_vec(1, ins)}.q[0]"
            dst_idx, dst_name = write_mm(0, ins)
            statements.append(f"{dst_name} = {CVT_PS2PI_HELPERS[ins.code]}({src});")
            defined_mm.add(dst_idx)
        elif ins.code == Code.PSHUFW_MM_MMM64_IMM8:
            imm = ins.immediate8 & 0xff
            src = mm_src64(1, ins)
            dst_idx, dst_name = write_mm(0, ins)
            statements.append(f"{dst_name} = op_pshufw({src}, {imm:#x}u);")
            defined_mm.add(dst_idx)
        elif ins.code in MM_PACKED_HELPERS:
            dst = read_mm(0, ins)
            src = mm_src64(1, ins)
            dst_idx, dst_name = write_mm(0, ins)
            statements.append(f"{dst_name} = {MM_PACKED_HELPERS[ins.code]}({dst}, {src});")
            defined_mm.add(dst_idx)
        elif ins.code in XMM_PACKED_HELPERS:
            dst = read_vec(0, ins)
            src = vec_src128(1, ins, alignment=16)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = {XMM_PACKED_HELPERS[ins.code]}({dst}, {src});")
            defined_xmm.add(dst_idx)
        elif ins.code == Code.MOVNTQ_M64_MM:
            if ins.op0_kind != OpKind.MEMORY:
                raise ValueError(f"expected memory destination at RVA {ins.ip:#x}: {ins}")
            src = read_mm(1, ins)
            statements.append(f"store64({flat_address(ins, store=True)}, {src});")
        elif ins.code == Code.MASKMOVQ_RDI_MM_MM:
            if ins.op0_kind != OpKind.MEMORY_SEG_RDI or ins.memory_segment != R.DS:
                raise ValueError(f"unsupported maskmovq target at RVA {ins.ip:#x}: {ins}")
            if rdi_expr != "r8":
                raise ValueError(f"maskmovq requires r8-rooted rdi at RVA {ins.ip:#x}: {ins}")
            src = read_mm(1, ins)
            mask = read_mm(2, ins)
            statements.append(f"op_maskmovq({rdi_expr}, {src}, {mask});")
        elif ins.code == Code.MASKMOVDQU_RDI_XMM_XMM:
            if ins.op0_kind != OpKind.MEMORY_SEG_RDI or ins.memory_segment != R.DS:
                raise ValueError(f"unsupported maskmovdqu target at RVA {ins.ip:#x}: {ins}")
            if rdi_expr != "r8":
                raise ValueError(f"maskmovdqu requires r8-rooted rdi at RVA {ins.ip:#x}: {ins}")
            src = read_vec(1, ins)
            mask = read_vec(2, ins)
            statements.append(f"op_maskmovdqu({rdi_expr}, {src}, {mask});")
        elif ins.code in (Code.MOV_R64_RM64, Code.MOV_RM64_R64):
            if ins.op0_kind != OpKind.REGISTER or ins.op1_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")
            op0_reg = ins.op0_register
            op1_reg = ins.op1_register
            if op0_reg not in (R.RDI, R.EDI) and op1_reg not in (R.RDI, R.EDI):
                raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")
            if not (RegisterExt.is_gpr64(op0_reg) and RegisterExt.is_gpr64(op1_reg)):
                raise ValueError(f"unsupported mov operand width at RVA {ins.ip:#x}: {ins}")
            if op1_reg == R.RDI:
                if op0_reg == R.RDI:
                    raise ValueError(f"nonvolatile GPR {op0_reg} at RVA {ins.ip:#x}: {ins}")
                dst_gpr = gpr64_scratch(op0_reg, ins)
                if "rdi" not in caller_rdi_holders:
                    raise ValueError(f"cannot save clobbered rdi at RVA {ins.ip:#x}: {ins}")
                # Caller's RDI is outside this ABI: keep dst_gpr only for RDI restoration.
                defined_gpr.discard(dst_gpr)
                caller_rdi_holders.add(dst_gpr)
            elif op0_reg == R.RDI:
                if op1_reg == R.R8:
                    if not any(s in caller_rdi_holders for s in SCRATCH_GPR64.values()):
                        raise ValueError(f"rdi modified without saving to volatile scratch at RVA {ins.ip:#x}: {ins}")
                    caller_rdi_holders.discard("rdi")
                    rdi_expr = "r8"
                elif op1_reg in SCRATCH_GPR64 and SCRATCH_GPR64[op1_reg] in caller_rdi_holders:
                    caller_rdi_holders.add("rdi")
                    rdi_expr = None
                else:
                    raise ValueError(f"non-r8 or unsaved write to nonvolatile GPR rdi at RVA {ins.ip:#x}: {ins}")
        elif ins.code in (*LOAD_ALIGNED_VEC_CODES, Code.MOVDQU_XMM_XMMM128):
            alignment = 16 if ins.code in LOAD_ALIGNED_VEC_CODES else 1
            src = vec_src128(1, ins, alignment=alignment)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = {src};")
            defined_xmm.add(dst_idx)
        elif ins.code in (*STORE_ALIGNED_VEC_CODES, Code.MOVDQU_XMMM128_XMM):
            alignment = 16 if ins.code in STORE_ALIGNED_VEC_CODES else 1
            src = read_vec(1, ins)
            if ins.op0_kind == OpKind.MEMORY:
                statements.append(f"store128({flat_address(ins, alignment=alignment, store=True)}, {src});")
            else:
                dst_idx, dst_name = write_vec(0, ins)
                statements.append(f"{dst_name} = {src};")
                defined_xmm.add(dst_idx)
        elif ins.code == Code.MOVQ_MM_MMM64:
            src = mm_src64(1, ins)
            dst_idx, dst_name = write_mm(0, ins)
            statements.append(f"{dst_name} = {src};")
            defined_mm.add(dst_idx)
        elif ins.code == Code.MOVQ_MMM64_MM:
            src = read_mm(1, ins)
            if ins.op0_kind == OpKind.MEMORY:
                statements.append(f"store64({flat_address(ins, store=True)}, {src});")
            else:
                dst_idx, dst_name = write_mm(0, ins)
                statements.append(f"{dst_name} = {src};")
                defined_mm.add(dst_idx)
        elif ins.code == Code.MOVQ_MM_RM64:
            src = f"load64({flat_address(ins)})" if ins.op1_kind == OpKind.MEMORY else read_gpr64(1, ins)
            dst_idx, dst_name = write_mm(0, ins)
            statements.append(f"{dst_name} = {src};")
            defined_mm.add(dst_idx)
        elif ins.code == Code.MOVQ_RM64_MM:
            src = read_mm(1, ins)
            if ins.op0_kind == OpKind.MEMORY:
                statements.append(f"store64({flat_address(ins, store=True)}, {src});")
            else:
                reg_name = write_gpr64(0, ins)
                caller_rdi_holders.discard(reg_name)
                statements.append(f"{reg_name} = {src};")
                defined_gpr.add(reg_name)
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec128 xmm{idx}{{}};" for idx in sorted(used_xmm)]
    if used_mm:
        decls.append("    std::uint64_t mm[8]{};")
    decls.extend(f"    word {reg_name} = 0ull;" for reg_name in sorted(used_gpr))
    body = [f"    {stmt}" for stmt in statements]
    lines = [
        f'extern "C" __declspec(dllexport) void lifted_{name}(word rcx, word rdx, word r8) {{',
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
