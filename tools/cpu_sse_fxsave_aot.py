"""Lift bounded Windows x64 leaf functions with SSE FXSAVE/FXRSTOR state ops to C++.

Each export uses the leaf ABI void fn(word rcx, word rdx, word r8), where rcx,
rdx and r8 carry host pointers to caller-owned buffers (16-byte-aligned bases
for m512byte and aligned m128 operands; rcx and rdx are inputs, r8 is the
output). The translator models the full 512-byte FXSAVE/FXRSTOR architectural
state in a local portable FxState struct (leaving host XMM6..15, x87/MMX and
MXCSR untouched across lifted calls).

Entry-state contract (`--entry-state`):
  - defined (default): the three-pointer ABI carries no incoming x87/MMX/MXCSR/XMM
    state; every fxsave/fxsave64 must follow a supported fxrstor/fxrstor64 that
    establishes FxState. Bare fxsave/fxsave64 before fxrstor is rejected.
  - canonical (opt-in): allows fxsave/fxsave64 before fxrstor assuming the
    native caller establishes canonical architectural state on entry
    (FCW=0x037F, FSW=0, FTW=0, FOP=0, FIP=0, FCS=0, FDP=0, FDS=0,
    MXCSR=0x1F80, ST0..7=0, XMM0..15=0).

Supported SSE state instructions (4 forms from docs/coverage/x64_zen2.tsv group SSE):
  - fxsave / fxsave64:   store 416 architectural state bytes to 16-byte-aligned
                         [r8 + disp], zeroing reserved byte 5, reserved ST/MM
                         bytes [10..15] (and [14..15]/[22..23] for 32-bit fxsave),
                         writing host MXCSR_MASK at [28..31], handling FCS/FDS
                         per CPUID.7.0:EBX[13], and leaving [416..511] untouched.
  - fxrstor / fxrstor64: load 416 architectural state bytes from 16-byte-aligned
                         [base + disp], validating MXCSR reserved bits against
                         host MXCSR_MASK (#GP before mutating state), masking FOP
                         to 11 bits (0x7FF), clearing FSW bits 15 and 7, and
                         zero-extending (32-bit) or preserving 64-bit FIP/FDP.

Transport/ABI glue: movdqa/movdqu (XMM0..XMM5), paddd (XMM0..XMM5), movq
(MM0..MM7 <-> m64 / volatile scratch r64 rax/r9/r10/r11), emms, 64-bit mov on
volatile GPRs, and terminal ret.

XSAVE-family instructions, VEX/EVEX/XOP encodings, maskmovq/maskmovdqu,
nonvolatile registers (XMM6..15, RBX/RBP/RSI/RDI/R12..R15/RSP), non-argument
memory bases, stores outside r8, RIP-relative addressing, instruction prefixes,
branches, unclosed MMX state before ret, and trailing bytes are rejected.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import Code, Decoder, EncodingKind, MemorySizeExt, OpKind, Register as R, RegisterExt

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}
VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = (
    {getattr(R, f"XMM{i}") for i in range(6, 16)}
    | {getattr(R, f"YMM{i}") for i in range(16)}
    | {getattr(R, f"ZMM{i}") for i in range(16)}
)
MM_REGISTERS = {getattr(R, f"MM{i}"): i for i in range(8)}

SCRATCH_GPR64 = {R.RAX: "rax", R.R9: "r9", R.R10: "r10", R.R11: "r11"}
VOLATILE_GPR64 = {**ARGUMENTS, **SCRATCH_GPR64}
NONVOLATILE_GPR = {
    R.RBX, R.EBX, R.BX, R.BL, R.BH,
    R.RBP, R.EBP, R.BP, R.BPL,
    R.RSI, R.ESI, R.SI, R.SIL,
    R.RDI, R.EDI, R.DI, R.DIL,
    R.RSP, R.ESP, R.SP, R.SPL,
    R.R12, R.R12D, R.R12W, R.R12L,
    R.R13, R.R13D, R.R13W, R.R13L,
    R.R14, R.R14D, R.R14W, R.R14L,
    R.R15, R.R15D, R.R15W, R.R15L,
}

FXSAVE_CODES = {
    Code.FXSAVE_M512BYTE: False,
    Code.FXSAVE64_M512BYTE: True,
}
FXRSTOR_CODES = {
    Code.FXRSTOR_M512BYTE: False,
    Code.FXRSTOR64_M512BYTE: True,
}
ENTRY_STATES = ("defined", "canonical")

XSAVE_REJECT_CODES = {
    Code.XSAVE_MEM,
    Code.XSAVE64_MEM,
    Code.XRSTOR_MEM,
    Code.XRSTOR64_MEM,
    Code.XSAVEOPT_MEM,
    Code.XSAVEOPT64_MEM,
    Code.XSAVEC_MEM,
    Code.XSAVEC64_MEM,
    Code.XSAVES_MEM,
    Code.XSAVES64_MEM,
    Code.XRSTORS_MEM,
    Code.XRSTORS64_MEM,
}

MASKMOV_REJECT_CODES = {
    Code.MASKMOVQ_RDI_MM_MM,
    Code.MASKMOVDQU_RDI_XMM_XMM,
}

VEC_LOAD_CODES = {
    Code.MOVDQA_XMM_XMMM128: True,
    Code.MOVDQU_XMM_XMMM128: False,
}
VEC_STORE_CODES = {
    Code.MOVDQA_XMMM128_XMM: True,
    Code.MOVDQU_XMMM128_XMM: False,
}
MOVQ_MM_CODES = {
    Code.MOVQ_MM_MMM64,
    Code.MOVQ_MMM64_MM,
    Code.MOVQ_MM_RM64,
    Code.MOVQ_RM64_MM,
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.EMMS,
    Code.PADDD_XMM_XMMM128,
    Code.MOV_R64_RM64,
    Code.MOV_RM64_R64,
    *FXSAVE_CODES,
    *FXRSTOR_CODES,
    *VEC_LOAD_CODES,
    *VEC_STORE_CODES,
    *MOVQ_MM_CODES,
}


PRELUDE = """#include <atomic>
#include <cstdint>
#include <cstring>
#include <intrin.h>

extern "C" __declspec(dllimport) void __stdcall RaiseException(
    unsigned long, unsigned long, unsigned long, const std::uintptr_t*);

extern "C" __declspec(dllexport) std::uint32_t host_has_fxsr() {
    int info[4]{};
    __cpuid(info, 1);
    return (info[3] & (1 << 24)) != 0 ? 1u : 0u;
}

extern "C" __declspec(dllexport) std::uint32_t host_deprecates_fpu_cs_ds() {
    int info0[4]{};
    __cpuid(info0, 0);
    if (info0[0] < 7) return 0u;
    int info7[4]{};
    __cpuidex(info7, 7, 0);
    return (info7[1] & (1 << 13)) != 0 ? 1u : 0u;
}

extern "C" __declspec(dllexport) std::uint32_t host_fp_err_cpuid_info() {
    // Bit 0: CPUID.07H.0:EBX[6] FDP_EXCPTN_ONLY
    // Bit 1: CPUID.07H.0:EBX[13] ZERO_FCS_FDS
    // Bit 2: CPUID.80000008H:EBX[2] RstrFpErrPtrs
    std::uint32_t flags = 0u;
    int info0[4]{};
    __cpuid(info0, 0);
    if (info0[0] >= 7) {
        int info7[4]{};
        __cpuidex(info7, 7, 0);
        if ((info7[1] & (1 << 6)) != 0) flags |= 1u;
        if ((info7[1] & (1 << 13)) != 0) flags |= 2u;
    }
    int ext0[4]{};
    __cpuid(ext0, static_cast<int>(0x80000000u));
    if (static_cast<std::uint32_t>(ext0[0]) >= 0x80000008u) {
        int ext8[4]{};
        __cpuid(ext8, static_cast<int>(0x80000008u));
        if ((ext8[1] & (1 << 2)) != 0) flags |= 4u;
    }
    return flags;
}

extern "C" __declspec(dllexport) std::uint32_t host_mxcsr_mask() {
    alignas(16) std::uint8_t area[512]{};
    _fxsave64(area);
    std::uint32_t mask = 0u;
    std::memcpy(&mask, area + 28, 4);
    return mask != 0u ? mask : 0x0000FFBFu;
}

using word = std::uint64_t;

struct Vec128 {
    std::uint64_t q[2];
};

struct FxState {
    std::uint16_t fcw = 0x037Fu;
    std::uint16_t fsw = 0u;
    std::uint8_t ftw = 0u;
    std::uint16_t fop = 0u;
    std::uint64_t fip = 0ull;
    std::uint16_t fcs = 0u;
    std::uint64_t fdp = 0ull;
    std::uint16_t fds = 0u;
    std::uint32_t mxcsr = 0x1F80u;
    std::uint32_t mxcsr_mask = 0x0000FFBFu;
    std::uint8_t st[8][16]{};
    std::uint8_t xmm[16][16]{};
};

[[noreturn]] static inline void raise_gp() {
    // #GP(0) on Windows x64 is reported as an access violation reading address -1.
    const std::uintptr_t info[2] = {0u, ~static_cast<std::uintptr_t>(0u)};
    RaiseException(0xC0000005ul, 0ul, 2ul, info);
    __assume(0);
}

static inline std::uint64_t load64(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto value = *reinterpret_cast<const volatile std::uint64_t __unaligned*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return value;
}

static inline void store64(word address, std::uint64_t val) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    *reinterpret_cast<volatile std::uint64_t __unaligned*>(address) = val;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline Vec128 load128(word address, bool aligned) {
    if (aligned && (address & 15ull) != 0ull) raise_gp();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* src = reinterpret_cast<const volatile std::uint64_t __unaligned*>(address);
    const std::uint64_t q0 = src[0];
    const std::uint64_t q1 = src[1];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1}};
}

static inline void store128(word address, const Vec128& v, bool aligned) {
    if (aligned && (address & 15ull) != 0ull) raise_gp();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto* dst = reinterpret_cast<volatile std::uint64_t __unaligned*>(address);
    dst[0] = v.q[0];
    dst[1] = v.q[1];
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline Vec128 get_xmm(const FxState& state, unsigned idx) {
    Vec128 v{};
    std::memcpy(v.q, state.xmm[idx], 16);
    return v;
}

static inline void set_xmm(FxState& state, unsigned idx, const Vec128& v) {
    std::memcpy(state.xmm[idx], v.q, 16);
}

static inline Vec128 op_paddd(const Vec128& a, const Vec128& b) {
    std::uint32_t da[4]{};
    std::uint32_t db[4]{};
    std::memcpy(da, a.q, 16);
    std::memcpy(db, b.q, 16);
    for (unsigned i = 0; i < 4u; ++i) da[i] += db[i];
    Vec128 out{};
    std::memcpy(out.q, da, 16);
    return out;
}

// Every MMX instruction but EMMS, one that only reads an MM register included, clears TOP and marks all tags valid.
static inline void enter_mmx(FxState& state) {
    state.ftw = 0xFFu;
    state.fsw = static_cast<std::uint16_t>(state.fsw & ~0x3800u);
}

static inline std::uint64_t get_mm(FxState& state, unsigned idx) {
    enter_mmx(state);
    std::uint64_t v = 0ull;
    std::memcpy(&v, state.st[idx], 8);
    return v;
}

static inline void set_mm(FxState& state, unsigned idx, std::uint64_t val) {
    std::memcpy(state.st[idx], &val, 8);
    state.st[idx][8] = 0xFFu;
    state.st[idx][9] = 0xFFu;
    std::memset(state.st[idx] + 10, 0, 6);
    enter_mmx(state);
}

static inline void op_emms(FxState& state) {
    state.ftw = 0u;
    state.fsw = static_cast<std::uint16_t>(state.fsw & ~0x3800u);
}

static inline void op_fxrstor(FxState& state, word address, bool is64) {
    if ((address & 15ull) != 0ull) raise_gp();
    std::uint8_t raw[416]{};
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* src = reinterpret_cast<const volatile std::uint8_t*>(address);
    for (unsigned i = 0; i < 416u; ++i) raw[i] = src[i];
    std::atomic_signal_fence(std::memory_order_seq_cst);

    std::uint32_t new_mxcsr = 0u;
    std::memcpy(&new_mxcsr, raw + 24, 4);
    if ((new_mxcsr & ~state.mxcsr_mask) != 0u) raise_gp();

    std::uint16_t fcw_raw = 0u;
    std::memcpy(&fcw_raw, raw + 0, 2);
    state.fcw = static_cast<std::uint16_t>((fcw_raw & 0x1F7Fu) | 0x0040u);
    std::uint16_t fsw_raw = 0u;
    std::memcpy(&fsw_raw, raw + 2, 2);
    state.fsw = static_cast<std::uint16_t>(fsw_raw & 0x7F7Fu);
    state.ftw = raw[4];
    std::uint16_t fop_raw = 0u;
    std::memcpy(&fop_raw, raw + 6, 2);
    state.fop = static_cast<std::uint16_t>(fop_raw & 0x07FFu);

    if (is64) {
        std::memcpy(&state.fip, raw + 8, 8);
        state.fcs = 0u;
        std::memcpy(&state.fdp, raw + 16, 8);
        state.fds = 0u;
    } else {
        std::uint32_t fip32 = 0u;
        std::uint32_t fdp32 = 0u;
        std::memcpy(&fip32, raw + 8, 4);
        std::memcpy(&fdp32, raw + 16, 4);
        state.fip = static_cast<std::uint64_t>(fip32);
        state.fdp = static_cast<std::uint64_t>(fdp32);
        if (host_deprecates_fpu_cs_ds() != 0u) {
            state.fcs = 0u;
            state.fds = 0u;
        } else {
            std::memcpy(&state.fcs, raw + 12, 2);
            std::memcpy(&state.fds, raw + 20, 2);
        }
    }
    state.mxcsr = new_mxcsr;
    const unsigned top = (state.fsw >> 11) & 7u;
    for (unsigned i = 0; i < 8u; ++i) {
        const unsigned phys = (top + i) & 7u;
        std::memcpy(state.st[phys], raw + 32u + i * 16u, 10);
        std::memset(state.st[phys] + 10, 0, 6);
    }
    for (unsigned i = 0; i < 16u; ++i) {
        std::memcpy(state.xmm[i], raw + 160u + i * 16u, 16);
    }
}

static inline void op_fxsave(const FxState& state, word address, bool is64) {
    if ((address & 15ull) != 0ull) raise_gp();
    std::uint8_t out[416]{};
    std::memcpy(out + 0, &state.fcw, 2);
    const std::uint16_t fsw_out = static_cast<std::uint16_t>(state.fsw & 0x7F7Fu);
    std::memcpy(out + 2, &fsw_out, 2);
    out[4] = state.ftw;
    out[5] = 0u;
    const std::uint16_t fop_out = static_cast<std::uint16_t>(state.fop & 0x07FFu);
    std::memcpy(out + 6, &fop_out, 2);

    if (is64) {
        std::memcpy(out + 8, &state.fip, 8);
        std::memcpy(out + 16, &state.fdp, 8);
    } else {
        const auto fip32 = static_cast<std::uint32_t>(state.fip);
        const auto fdp32 = static_cast<std::uint32_t>(state.fdp);
        const bool dep = host_deprecates_fpu_cs_ds() != 0u;
        const std::uint16_t fcs_out = dep ? 0u : state.fcs;
        const std::uint16_t fds_out = dep ? 0u : state.fds;
        std::memcpy(out + 8, &fip32, 4);
        std::memcpy(out + 12, &fcs_out, 2);
        std::memcpy(out + 16, &fdp32, 4);
        std::memcpy(out + 20, &fds_out, 2);
    }
    std::memcpy(out + 24, &state.mxcsr, 4);
    std::memcpy(out + 28, &state.mxcsr_mask, 4);
    const unsigned top = (state.fsw >> 11) & 7u;
    for (unsigned i = 0; i < 8u; ++i) {
        const unsigned phys = (top + i) & 7u;
        std::memcpy(out + 32u + i * 16u, state.st[phys], 10);
    }
    for (unsigned i = 0; i < 16u; ++i) {
        std::memcpy(out + 160u + i * 16u, state.xmm[i], 16);
    }

    std::uint64_t words[52]{};
    std::memcpy(words, out, sizeof(out));
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto* dst = reinterpret_cast<volatile std::uint64_t*>(address);
    for (unsigned i = 0; i < 52u; ++i) dst[i] = words[i];
    std::atomic_signal_fence(std::memory_order_seq_cst);
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


def gpr64_name(register, ins, *, scratch_only=False):
    """Map volatile 64-bit GPRs to C++ names, rejecting nonvolatile and sub-64-bit GPRs."""
    allowed = SCRATCH_GPR64 if scratch_only else VOLATILE_GPR64
    if register in allowed:
        return allowed[register]
    if register in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {register} at RVA {ins.ip:#x}: {ins}")
    if scratch_only and register in ARGUMENTS:
        raise ValueError(f"unsupported argument GPR operand {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported GPR {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, clobbered_ptrs, *, expected_bytes=None, write=False):
    """Lower a flat [base + disp] pointer operand rooted in an unclobbered argument GPR64."""
    base = ins.memory_base
    if (
        ins.is_ip_rel_memory_operand
        or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
        or ins.memory_index != R.NONE
        or not RegisterExt.is_gpr64(base)
    ):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    if base in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR base {base} in memory operand at RVA {ins.ip:#x}: {ins}")
    if base not in ARGUMENTS:
        raise ValueError(f"unsupported memory base {base} at RVA {ins.ip:#x}: {ins}")
    if base in clobbered_ptrs:
        raise ValueError(f"memory operand uses clobbered pointer {ARGUMENTS[base]} at RVA {ins.ip:#x}: {ins}")
    if write and base != R.R8:
        raise ValueError(f"non-r8 store memory operand at RVA {ins.ip:#x}: {ins}")
    if expected_bytes is not None and MemorySizeExt.size(ins.memory_size) != expected_bytes:
        raise ValueError(f"expected {expected_bytes}-byte memory operand at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    name = ARGUMENTS[base]
    return f"({name} + {disp:#x}ull)" if disp else name


def lift_code(name, start, raw, *, entry_state="defined"):
    """Decode and lift one bounded straight-line SSE FXSAVE/FXRSTOR leaf function."""
    if entry_state not in ENTRY_STATES:
        raise ValueError(f"unsupported entry-state profile {entry_state!r}")
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    defined_xmm = set()
    defined_mm = set()
    used_gpr = set()
    defined_gpr = {"rcx", "rdx", "r8"}
    clobbered_ptrs = set()
    mmx_active = False
    fx_state_defined = False
    used_canonical_entry = False
    statements = []
    returned = False

    def read_xmm(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected XMM operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(ins.op_register(op_index), ins)
        if idx not in defined_xmm:
            raise ValueError(f"undefined vector register xmm{idx} at RVA {ins.ip:#x}: {ins}")
        return f"get_xmm(state, {idx})"

    def write_xmm_idx(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected XMM destination at RVA {ins.ip:#x}: {ins}")
        return vec_index(ins.op_register(op_index), ins)

    def read_mm(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected MMX operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx = mm_index(ins.op_register(op_index), ins)
        if idx not in defined_mm:
            raise ValueError(f"undefined MMX register mm{idx} at RVA {ins.ip:#x}: {ins}")
        return idx

    def read_gpr(op_index, ins, *, scratch_only=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR operand {op_index} at RVA {ins.ip:#x}: {ins}")
        reg_name = gpr64_name(ins.op_register(op_index), ins, scratch_only=scratch_only)
        if reg_name not in defined_gpr:
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        used_gpr.add(reg_name)
        return reg_name

    def write_gpr(op_index, ins, *, scratch_only=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR destination at RVA {ins.ip:#x}: {ins}")
        reg_name = gpr64_name(ins.op_register(op_index), ins, scratch_only=scratch_only)
        if reg_name == "r8":
            raise ValueError(f"unsupported write to output buffer pointer r8 at RVA {ins.ip:#x}: {ins}")
        return reg_name

    def clobber_gpr(reg_name):
        if reg_name == "rcx":
            clobbered_ptrs.add(R.RCX)
        elif reg_name == "rdx":
            clobbered_ptrs.add(R.RDX)

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if (
            ins.has_lock_prefix
            or ins.has_rep_prefix
            or ins.has_repne_prefix
            or ins.has_segment_prefix
            or ins.has_xacquire_prefix
            or ins.has_xrelease_prefix
        ):
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        encoding = ins.op_code().encoding
        if encoding in (EncodingKind.VEX, EncodingKind.EVEX, EncodingKind.XOP):
            raise ValueError(f"unsupported VEX/EVEX instruction at RVA {ins.ip:#x}: {ins}")
        if ins.op_mask != R.NONE or ins.zeroing_masking or ins.is_broadcast:
            raise ValueError(f"unsupported VEX/EVEX instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code in XSAVE_REJECT_CODES:
            raise ValueError(f"unsupported XSAVE-family instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code in MASKMOV_REJECT_CODES:
            raise ValueError(f"unsupported masked-store instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            if mmx_active:
                raise ValueError(f"unclosed MMX state before ret at RVA {ins.ip:#x}: {ins}")
            statements.append("return;")
            returned = True

        elif ins.code in FXSAVE_CODES:
            if ins.op0_kind != OpKind.MEMORY:
                raise ValueError(f"expected memory operand for fxsave at RVA {ins.ip:#x}: {ins}")
            addr = flat_address(ins, clobbered_ptrs, expected_bytes=512, write=True)
            if not fx_state_defined:
                if entry_state != "canonical":
                    raise ValueError(
                        f"fxsave before fxrstor requires explicit --entry-state canonical at RVA {ins.ip:#x}: {ins}"
                    )
                used_canonical_entry = True
            is64 = "true" if FXSAVE_CODES[ins.code] else "false"
            statements.append(f"op_fxsave(state, {addr}, {is64});")

        elif ins.code in FXRSTOR_CODES:
            if ins.op0_kind != OpKind.MEMORY:
                raise ValueError(f"expected memory operand for fxrstor at RVA {ins.ip:#x}: {ins}")
            addr = flat_address(ins, clobbered_ptrs, expected_bytes=512, write=False)
            is64 = "true" if FXRSTOR_CODES[ins.code] else "false"
            statements.append(f"op_fxrstor(state, {addr}, {is64});")
            fx_state_defined = True
            defined_xmm.update(range(6))
            defined_mm.update(range(8))

        elif ins.code == Code.EMMS:
            statements.append("op_emms(state);")
            defined_mm.clear()
            mmx_active = False

        elif ins.code in VEC_LOAD_CODES:
            aligned = "true" if VEC_LOAD_CODES[ins.code] else "false"
            dst_idx = write_xmm_idx(0, ins)
            if ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=16, write=False)
                statements.append(f"set_xmm(state, {dst_idx}, load128({addr}, {aligned}));")
            else:
                src_expr = read_xmm(1, ins)
                statements.append(f"set_xmm(state, {dst_idx}, {src_expr});")
            defined_xmm.add(dst_idx)

        elif ins.code in VEC_STORE_CODES:
            aligned = "true" if VEC_STORE_CODES[ins.code] else "false"
            if ins.op0_kind == OpKind.MEMORY:
                src_expr = read_xmm(1, ins)
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=16, write=True)
                statements.append(f"store128({addr}, {src_expr}, {aligned});")
            else:
                src_expr = read_xmm(1, ins)
                dst_idx = write_xmm_idx(0, ins)
                statements.append(f"set_xmm(state, {dst_idx}, {src_expr});")
                defined_xmm.add(dst_idx)

        elif ins.code == Code.PADDD_XMM_XMMM128:
            dst_expr = read_xmm(0, ins)
            dst_idx = write_xmm_idx(0, ins)
            if ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=16, write=False)
                rhs_expr = f"load128({addr}, true)"
            else:
                rhs_expr = read_xmm(1, ins)
            statements.append(f"set_xmm(state, {dst_idx}, op_paddd({dst_expr}, {rhs_expr}));")

        elif ins.code in MOVQ_MM_CODES:
            if ins.op0_kind == OpKind.MEMORY:
                src_mm = read_mm(1, ins)
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=8, write=True)
                statements.append(f"store64({addr}, get_mm(state, {src_mm}));")
                mmx_active = True
            elif ins.op1_kind == OpKind.MEMORY:
                dst_mm = mm_index(ins.op0_register, ins)
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=8, write=False)
                statements.append(f"set_mm(state, {dst_mm}, load64({addr}));")
                defined_mm.add(dst_mm)
                mmx_active = True
            elif ins.op0_register in MM_REGISTERS and ins.op1_register in MM_REGISTERS:
                src_mm = read_mm(1, ins)
                dst_mm = mm_index(ins.op0_register, ins)
                statements.append(f"set_mm(state, {dst_mm}, get_mm(state, {src_mm}));")
                defined_mm.add(dst_mm)
                mmx_active = True
            elif ins.op0_register in MM_REGISTERS:
                src_gpr = read_gpr(1, ins, scratch_only=True)
                dst_mm = mm_index(ins.op0_register, ins)
                statements.append(f"set_mm(state, {dst_mm}, {src_gpr});")
                defined_mm.add(dst_mm)
                mmx_active = True
            else:
                src_mm = read_mm(1, ins)
                dst_gpr = write_gpr(0, ins, scratch_only=True)
                used_gpr.add(dst_gpr)
                statements.append(f"{dst_gpr} = get_mm(state, {src_mm});")
                defined_gpr.add(dst_gpr)
                mmx_active = True

        elif ins.code in (Code.MOV_R64_RM64, Code.MOV_RM64_R64):
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src_gpr = read_gpr(1, ins)
                dst_gpr = write_gpr(0, ins)
                clobber_gpr(dst_gpr)
                used_gpr.add(dst_gpr)
                statements.append(f"{dst_gpr} = {src_gpr};")
                defined_gpr.add(dst_gpr)
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst_gpr = write_gpr(0, ins)
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=8, write=False)
                clobber_gpr(dst_gpr)
                used_gpr.add(dst_gpr)
                statements.append(f"{dst_gpr} = load64({addr});")
                defined_gpr.add(dst_gpr)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src_gpr = read_gpr(1, ins)
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=8, write=True)
                statements.append(f"store64({addr}, {src_gpr});")
            else:
                raise ValueError(f"unsupported mov operands at RVA {ins.ip:#x}: {ins}")

        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [
        "    FxState state{};",
        "    state.mxcsr_mask = host_mxcsr_mask();",
    ]
    for gpr in ("rax", "r9", "r10", "r11"):
        if gpr in used_gpr:
            decls.append(f"    word {gpr} = 0ull;")
    body = [f"    {stmt}" for stmt in statements]
    if used_canonical_entry:
        contract = (
            "// Entry-state profile: canonical (caller must establish FCW=0x037F, FSW=0, "
            "FTW=0, FOP=0, FIP=0, FCS=0, FDP=0, FDS=0, MXCSR=0x1F80, ST0..7=0, XMM0..15=0)."
        )
    else:
        contract = f"// Entry-state profile: {entry_state} (all fxsave/fxsave64 follow fxrstor/fxrstor64)."
    lines = [
        contract,
        f'extern "C" __declspec(dllexport) void lifted_{name}(word rcx, word rdx, word r8) {{',
        "    (void)rcx; (void)rdx; (void)r8;",
        *decls,
        *body,
        "}",
    ]
    return "\n".join(lines), len(instructions)


def lift(pe, name, *, entry_state="defined"):
    entry = export_rva(pe, name)
    raw = bounded_code(pe, entry)
    source, count = lift_code(name, entry, raw, entry_state=entry_state)
    return source, entry, count


def translate_pe(pe, names, *, entry_state="defined"):
    if entry_state not in ENTRY_STATES:
        raise ValueError(f"unsupported entry-state profile {entry_state!r}")
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    lifted = [lift(pe, name, entry_state=entry_state) for name in names]
    is_canonical = 1 if entry_state == "canonical" else 0
    profile_block = (
        f'\n// Entry-state profile: {entry_state}\n'
        f'extern "C" __declspec(dllexport) const char* lifted_entry_state_profile() {{ return "{entry_state}"; }}\n'
        f'extern "C" __declspec(dllexport) std::uint32_t lifted_requires_canonical_entry() {{ return {is_canonical}u; }}\n'
    )
    source = PRELUDE + profile_block + "\n".join(f"\n{text}" for text, _, _ in lifted) + "\n"
    meta = [(name, entry, count) for name, (_, entry, count) in zip(names, lifted)]
    return source, meta


def translate(image, names, *, entry_state="defined"):
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, names, entry_state=entry_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated export names")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--entry-state",
        choices=ENTRY_STATES,
        default="defined",
        help="entry-state profile: defined (default, requires fxrstor before fxsave) or canonical (opt-in caller-initialized canonical FX state)",
    )
    args = parser.parse_args()
    try:
        source, meta = translate(args.image, args.exports.split(","), entry_state=args.entry_state)
        args.output.write_text(source, encoding="utf-8")
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x} ({args.entry_state}): {count} instructions")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
