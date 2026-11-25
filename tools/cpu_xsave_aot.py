"""Lift bounded Windows x64 leaf functions with XSAVE/XRSTOR state ops to C++.

Each export uses the leaf ABI void fn(word rcx, word rdx, word r8), where rcx,
rdx and r8 carry host pointers to caller-owned buffers (64-byte-aligned bases
for XSAVE/XRSTOR memory operands; rcx and rdx are inputs, r8 is the output).
The translator models the bounded 832-byte x87|SSE|AVX standard-format XSAVE
state (XCR0[2:0] = 7) in a local portable XState struct (leaving host XMM6..15,
YMM6..15, x87/MMX and MXCSR untouched across lifted calls).

Entry-state contract (`--entry-state`) and XRSTOR format contract (`--compacted-xrstor`):
  - defined (default): the three-pointer ABI carries no incoming x87/SSE/AVX
    state; every xsave/xsave64 must only save components previously initialized
    by a supported xrstor/xrstor64 (or vzeroupper for AVX). Entry-only xsave*
    and partially initialized restore/save paths that observe uninitialized
    components from entry are rejected. MXCSR is tracked apart from the
    components: by default (standard-only XRSTOR), an xrstor whose RFBM selects
    SSE or AVX defines MXCSR, so vzeroupper alone does not, and a save that
    stores MXCSR (xsave/xsaveopt with SSE or AVX selected, xsavec with SSE
    selected) is rejected without it.
  - --compacted-xrstor (opt-in): permits XRSTOR to accept compacted headers
    (XCOMP_BV[63] set) in addition to standard headers. Because the header
    format is data and cannot be distinguished at lift time, a compacted
    restore selecting AVX without SSE leaves MXCSR untouched; therefore, with
    --compacted-xrstor, only an xrstor whose RFBM selects SSE defines MXCSR
    under the defined profile, and AVX-only xrstor -> xsaveopt(AVX) requires
    --entry-state canonical (or a prior SSE restore).
  - canonical (opt-in): permits xsave/xsave64 before full component definition
    assuming the native caller establishes canonical architectural XSAVE state
    on entry (FCW=0x037F, FSW=0, FTW=0, FOP=0, FIP=0, FCS=0, FDP=0, FDS=0,
    MXCSR=0x1F80, ST0..7=0, XMM0..15=0, YMM_HI0..15=0, XINUSE=0).

Supported XSAVE state instructions (6 forms from docs/coverage/x64_zen2.tsv
group XSAVE):
  - xsave / xsave64:   require 64-byte-aligned [base + disp] and bounded mask
                       (edx == 0 && (eax & ~7) == 0); write requested legacy
                       x87/SSE and AVX YMM[255:128] regions while leaving
                       legacy [416..511], XCOMP_BV [520..527], and header
                       [528..575] untouched, and updating XSTATE_BV[2:0] =
                       (old & ~rfbm) | (xinuse & rfbm). MXCSR and MXCSR_MASK are stored when
                       RFBM selects SSE or AVX.
  - xsavec / xsavec64: same alignment and mask contract as xsave; write the
                       compacted format: only components with XINUSE & RFBM set
                       are stored (SSE is in use when any XMM register is live
                       or MXCSR differs from 0x1F80; x87/SSE in the legacy region, AVX at the
                       first extended offset 576, which is also its compacted
                       offset because x87/SSE own no extended slot), header
                       XSTATE_BV = XINUSE & RFBM, XCOMP_BV = bit 63 | RFBM, and
                       the reserved header bytes [528..575] left untouched.
                       MXCSR and MXCSR_MASK are stored with selected live SSE state.
                       Alignment and mask checks run before the first store.
  - xsaveopt / xsaveopt64: same alignment and mask contract as xsave, standard format, init optimization
                       only: a component with XINUSE clear is not written and its XSTATE_BV bit is
                       cleared; XSTATE_BV = (old & ~rfbm) | (XINUSE & rfbm). SSE counts as in use
                       while MXCSR differs from 0x1F80. MXCSR and MXCSR_MASK are stored whenever
                       RFBM selects SSE or AVX, and the header bytes [520..575] stay untouched. The
                       modified optimization (XMODIFIED after a prior XRSTOR to the same address) is
                       not modeled: the lifted code writes every component that is in use.
  - xrstor / xrstor64: require 64-byte-aligned [base + disp] and bounded mask
                       (edx == 0 && (eax & ~7) == 0). The header selects the format.
                       Standard (XCOMP_BV[63] clear): validate XCOMP_BV == 0,
                       header [528..535] == 0, (XSTATE_BV & ~7) == 0, and
                       (when rfbm & 6 != 0) MXCSR against host MXCSR_MASK
                       (#GP before mutating state), then restore or init each
                       requested component and update XINUSE[2:0].
                       Compacted (XCOMP_BV[63] set, the layout xsavec writes):
                       validate that XCOMP_BV[62:0] lists nothing outside
                       x87|SSE|AVX, that XSTATE_BV is a subset of XCOMP_BV[62:0]
                       and that every header byte [528..575] is zero (#GP before
                       mutating state). A requested component restores from its
                       compacted slot (x87 and SSE in the legacy area, AVX at 576, the
                       first extended slot) when XSTATE_BV and XCOMP_BV both list it,
                       and is initialized otherwise, which includes a component
                       XCOMP_BV does not list. MXCSR belongs to the SSE component:
                       loaded (and checked against MXCSR_MASK) when RFBM[1] and
                       XSTATE_BV[1] are set, reset to 0x1F80 when RFBM[1] is set and
                       XSTATE_BV[1] is clear, untouched when RFBM[1] is clear, so an
                       AVX-only restore keeps the current MXCSR. The xsavec writer
                       leaves bytes [528..575] alone, so a caller that restores a
                       buffer xsavec filled must have zeroed them first.

Transport/ABI glue: vmovdqa/vmovdqu (XMM0..5, YMM0..5), vzeroupper,
movdqa/movdqu (XMM0..5), paddd (XMM0..5), movq (MM0..7 <-> m64 / volatile
scratch r64), emms, 32/64-bit volatile GPR mov/xor, and terminal ret.

XSAVES/XRSTORS, FXSAVE/FXRSTOR, EVEX encodings, nonvolatile
registers (XMM6..15, YMM6..15, ZMM0..15, RBX/RBP/RSI/RDI/R12..R15/RSP),
non-argument memory bases, stores outside r8, RIP-relative/indexed addressing,
prefixes, branches, unclosed MMX state before ret, and trailing bytes reject.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import Code, Decoder, EncodingKind, MemorySizeExt, OpKind, Register as R, RegisterExt

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}
VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
VOLATILE_YMM = {getattr(R, f"YMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = (
    {getattr(R, f"XMM{i}") for i in range(6, 16)}
    | {getattr(R, f"YMM{i}") for i in range(6, 16)}
    | {getattr(R, f"ZMM{i}") for i in range(16)}
)
MM_REGISTERS = {getattr(R, f"MM{i}"): i for i in range(8)}

SCRATCH_GPR64 = {R.RAX: "rax", R.R9: "r9", R.R10: "r10", R.R11: "r11"}
VOLATILE_GPR64 = {**ARGUMENTS, **SCRATCH_GPR64}
VOLATILE_GPR32 = {
    R.EAX: "rax",
    R.ECX: "rcx",
    R.EDX: "rdx",
    R.R8D: "r8",
    R.R9D: "r9",
    R.R10D: "r10",
    R.R11D: "r11",
}
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

XSAVE_CODES = {
    Code.XSAVE_MEM: False,
    Code.XSAVE64_MEM: True,
}
XRSTOR_CODES = {
    Code.XRSTOR_MEM: False,
    Code.XRSTOR64_MEM: True,
}
XSAVEC_CODES = {
    Code.XSAVEC_MEM: False,
    Code.XSAVEC64_MEM: True,
}
XSAVEOPT_CODES = {
    Code.XSAVEOPT_MEM: False,
    Code.XSAVEOPT64_MEM: True,
}
ENTRY_STATES = ("defined", "canonical")
XCOMP_NAMES = ((1, "x87"), (2, "sse"), (4, "avx"))

REJECT_XSAVE_FAMILY_CODES = {
    Code.XSAVES_MEM,
    Code.XSAVES64_MEM,
    Code.XRSTORS_MEM,
    Code.XRSTORS64_MEM,
}
REJECT_FXSAVE_CODES = {
    Code.FXSAVE_M512BYTE,
    Code.FXSAVE64_M512BYTE,
    Code.FXRSTOR_M512BYTE,
    Code.FXRSTOR64_M512BYTE,
}

VEX_LOAD_CODES = {
    Code.VEX_VMOVDQA_XMM_XMMM128: (False, True),
    Code.VEX_VMOVDQU_XMM_XMMM128: (False, False),
    Code.VEX_VMOVDQA_YMM_YMMM256: (True, True),
    Code.VEX_VMOVDQU_YMM_YMMM256: (True, False),
}
VEX_STORE_CODES = {
    Code.VEX_VMOVDQA_XMMM128_XMM: (False, True),
    Code.VEX_VMOVDQU_XMMM128_XMM: (False, False),
    Code.VEX_VMOVDQA_YMMM256_YMM: (True, True),
    Code.VEX_VMOVDQU_YMMM256_YMM: (True, False),
}
SSE_LOAD_CODES = {
    Code.MOVDQA_XMM_XMMM128: True,
    Code.MOVDQU_XMM_XMMM128: False,
}
SSE_STORE_CODES = {
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
    Code.VEX_VZEROUPPER,
    Code.PADDD_XMM_XMMM128,
    Code.MOV_R64_RM64,
    Code.MOV_RM64_R64,
    Code.MOV_R32_RM32,
    Code.MOV_RM32_R32,
    Code.MOV_R32_IMM32,
    Code.XOR_R32_RM32,
    Code.XOR_RM32_R32,
    *XSAVE_CODES,
    *XSAVEC_CODES,
    *XSAVEOPT_CODES,
    *XRSTOR_CODES,
    *VEX_LOAD_CODES,
    *VEX_STORE_CODES,
    *SSE_LOAD_CODES,
    *SSE_STORE_CODES,
    *MOVQ_MM_CODES,
}


PRELUDE = """#include <atomic>
#include <cstdint>
#include <cstring>
#include <intrin.h>

extern "C" __declspec(dllimport) void __stdcall RaiseException(
    unsigned long, unsigned long, unsigned long, const std::uintptr_t*);

extern "C" __declspec(dllexport) std::uint32_t host_has_xsave() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 26) | (1 << 27) | (1 << 28);  // XSAVE, OSXSAVE, AVX
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 7u) == 7u ? 1u : 0u;
}

extern "C" __declspec(dllexport) std::uint32_t host_has_xsavec() {
    if (host_has_xsave() == 0u) return 0u;
    int info0[4]{};
    __cpuid(info0, 0);
    if (info0[0] < 0xD) return 0u;
    int sub[4]{};
    __cpuidex(sub, 0xD, 1);
    return (sub[0] & 2) != 0 ? 1u : 0u;  // CPUID.(0DH,1):EAX[1] = XSAVEC
}

extern "C" __declspec(dllexport) std::uint32_t host_has_xsaveopt() {
    if (host_has_xsave() == 0u) return 0u;
    int info0[4]{};
    __cpuid(info0, 0);
    if (info0[0] < 0xD) return 0u;
    int sub[4]{};
    __cpuidex(sub, 0xD, 1);
    return (sub[0] & 1) != 0 ? 1u : 0u;  // CPUID.(0DH,1):EAX[0] = XSAVEOPT
}

extern "C" __declspec(dllexport) std::uint64_t host_xcr0() {
    if (host_has_xsave() == 0u) return 0ull;
    return _xgetbv(0);
}

extern "C" __declspec(dllexport) std::uint32_t host_xsave_area_size() {
    int features[4]{};
    __cpuidex(features, 0xD, 0);
    return static_cast<std::uint32_t>(features[1]);
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

struct Vec256 {
    std::uint64_t q[4];
};

struct XState {
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
    std::uint8_t ymm_hi[16][16]{};
    std::uint64_t xinuse = 0ull;
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

static inline void store64(word address, std::uint64_t value) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    *reinterpret_cast<volatile std::uint64_t __unaligned*>(address) = value;
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

static inline Vec256 load256(word address, bool aligned) {
    if (aligned && (address & 31ull) != 0ull) raise_gp();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* src = reinterpret_cast<const volatile std::uint64_t __unaligned*>(address);
    const std::uint64_t q0 = src[0];
    const std::uint64_t q1 = src[1];
    const std::uint64_t q2 = src[2];
    const std::uint64_t q3 = src[3];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, q2, q3}};
}

static inline void store128(word address, const Vec128& v, bool aligned) {
    if (aligned && (address & 15ull) != 0ull) raise_gp();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto* dst = reinterpret_cast<volatile std::uint64_t __unaligned*>(address);
    dst[0] = v.q[0];
    dst[1] = v.q[1];
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store256(word address, const Vec256& v, bool aligned) {
    if (aligned && (address & 31ull) != 0ull) raise_gp();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto* dst = reinterpret_cast<volatile std::uint64_t __unaligned*>(address);
    dst[0] = v.q[0];
    dst[1] = v.q[1];
    dst[2] = v.q[2];
    dst[3] = v.q[3];
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline bool any_ymm_hi_nonzero(const XState& state) {
    for (unsigned i = 0; i < 16u; ++i) {
        for (unsigned j = 0; j < 16u; ++j) {
            if (state.ymm_hi[i][j] != 0u) return true;
        }
    }
    return false;
}

static inline Vec128 get_xmm(const XState& state, unsigned idx) {
    Vec128 v{};
    std::memcpy(v.q, state.xmm[idx], 16);
    return v;
}

static inline Vec256 get_ymm(const XState& state, unsigned idx) {
    Vec256 v{};
    std::memcpy(v.q, state.xmm[idx], 16);
    std::memcpy(v.q + 2, state.ymm_hi[idx], 16);
    return v;
}

static inline void set_xmm_legacy(XState& state, unsigned idx, const Vec128& v) {
    std::memcpy(state.xmm[idx], v.q, 16);
    state.xinuse |= 2ull;
}

static inline void set_xmm_vex(XState& state, unsigned idx, const Vec128& v) {
    std::memcpy(state.xmm[idx], v.q, 16);
    std::memset(state.ymm_hi[idx], 0, 16);
    state.xinuse |= 2ull;
    if (!any_ymm_hi_nonzero(state)) state.xinuse &= ~4ull;
}

static inline void set_ymm(XState& state, unsigned idx, const Vec256& v) {
    std::memcpy(state.xmm[idx], v.q, 16);
    std::memcpy(state.ymm_hi[idx], v.q + 2, 16);
    state.xinuse |= 2ull;
    if (v.q[2] != 0ull || v.q[3] != 0ull) state.xinuse |= 4ull;
    else if (!any_ymm_hi_nonzero(state)) state.xinuse &= ~4ull;
}

static inline void op_vzeroupper(XState& state) {
    for (unsigned i = 0; i < 16u; ++i) std::memset(state.ymm_hi[i], 0, 16);
    state.xinuse &= ~4ull;
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

// Every MMX instruction but EMMS, one that only reads an MM register included, clears TOP, marks all tags valid
// and leaves the x87 component in use.
static inline void enter_mmx(XState& state) {
    state.ftw = 0xFFu;
    state.fsw = static_cast<std::uint16_t>(state.fsw & ~0x3800u);
    state.xinuse |= 1ull;
}

static inline std::uint64_t get_mm(XState& state, unsigned idx) {
    enter_mmx(state);
    std::uint64_t v = 0ull;
    std::memcpy(&v, state.st[idx], 8);
    return v;
}

static inline void set_mm(XState& state, unsigned idx, std::uint64_t val) {
    std::memcpy(state.st[idx], &val, 8);
    state.st[idx][8] = 0xFFu;
    state.st[idx][9] = 0xFFu;
    std::memset(state.st[idx] + 10, 0, 6);
    enter_mmx(state);
}

static inline void op_emms(XState& state) {
    state.ftw = 0u;
    state.fsw = static_cast<std::uint16_t>(state.fsw & ~0x3800u);
}

static inline void init_x87(XState& state) {
    state.fcw = 0x037Fu;
    state.fsw = 0u;
    state.ftw = 0u;
    state.fop = 0u;
    state.fip = 0ull;
    state.fcs = 0u;
    state.fdp = 0ull;
    state.fds = 0u;
    for (unsigned i = 0; i < 8u; ++i) std::memset(state.st[i], 0, 16);
}

static inline void op_xrstor(XState& state, word address, word rax_val, word rdx_val, bool is64, bool allow_compacted = false) {
    const std::uint32_t mask_lo = static_cast<std::uint32_t>(rax_val);
    const std::uint32_t mask_hi = static_cast<std::uint32_t>(rdx_val);
    if ((address & 63ull) != 0ull) raise_gp();
    if (mask_hi != 0u || (mask_lo & ~7u) != 0u) raise_gp();
    const std::uint64_t rfbm = static_cast<std::uint64_t>(mask_lo) & 7ull;

    std::uint8_t raw[832]{};
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* src = reinterpret_cast<const volatile std::uint8_t*>(address);
    for (unsigned i = 0; i < 832u; ++i) raw[i] = src[i];
    std::atomic_signal_fence(std::memory_order_seq_cst);

    std::uint64_t xstate_bv = 0ull;
    std::uint64_t xcomp_bv = 0ull;
    std::uint64_t hdr_reserved = 0ull;
    std::memcpy(&xstate_bv, raw + 512, 8);
    std::memcpy(&xcomp_bv, raw + 520, 8);
    std::memcpy(&hdr_reserved, raw + 528, 8);
    // XCOMP_BV[63] selects the compacted format when allow_compacted is enabled.
    const bool compacted = allow_compacted && ((xcomp_bv >> 63) != 0ull);
    const std::uint64_t xcomp_components = xcomp_bv & ~(1ull << 63);
    if (compacted) {
        // Compacted header: every header byte after XCOMP_BV is reserved, XSTATE_BV must be a subset of the
        // components XCOMP_BV lists, and XCOMP_BV may list nothing outside x87|SSE|AVX.
        for (unsigned i = 528u; i < 576u; i += 8u) {
            std::uint64_t word = 0ull;
            std::memcpy(&word, raw + i, 8);
            if (word != 0ull) raise_gp();
        }
        if ((xcomp_components & ~7ull) != 0ull) raise_gp();
        if ((xstate_bv & ~xcomp_components) != 0ull) raise_gp();
    } else {
        if (xcomp_bv != 0ull || hdr_reserved != 0ull) raise_gp();
        if ((xstate_bv & ~7ull) != 0ull) raise_gp();
    }

    // MXCSR: the standard format loads it whenever RFBM selects SSE or AVX. The compacted format keeps it with
    // the SSE component only: loaded when RFBM[1] and XSTATE_BV[1] are set, reset to 0x1F80 when RFBM[1] is set
    // and XSTATE_BV[1] is clear, and left alone when RFBM[1] is clear (an AVX-only restore does not touch it).
    // The #GP check applies to a loaded value only.
    std::uint32_t new_mxcsr = 0u;
    bool restores_mxcsr = false;
    if (!compacted) {
        restores_mxcsr = (rfbm & 6ull) != 0ull;
        if (restores_mxcsr) {
            std::memcpy(&new_mxcsr, raw + 24, 4);
            if ((new_mxcsr & ~state.mxcsr_mask) != 0u) raise_gp();
        }
    } else if ((rfbm & 2ull) != 0ull) {
        restores_mxcsr = true;
        new_mxcsr = 0x1F80u;
        if ((xstate_bv & 2ull) != 0ull) {
            std::memcpy(&new_mxcsr, raw + 24, 4);
            if ((new_mxcsr & ~state.mxcsr_mask) != 0u) raise_gp();
        }
    }

    if ((rfbm & 1ull) != 0ull) {
        if ((xstate_bv & 1ull) != 0ull) {
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
            const unsigned top = (state.fsw >> 11) & 7u;
            for (unsigned i = 0; i < 8u; ++i) {
                const unsigned phys = (top + i) & 7u;
                std::memcpy(state.st[phys], raw + 32u + i * 16u, 10);
                std::memset(state.st[phys] + 10, 0, 6);
            }
            state.xinuse |= 1ull;
        } else {
            init_x87(state);
            state.xinuse &= ~1ull;
        }
    }

    if (restores_mxcsr) {
        state.mxcsr = new_mxcsr;
    }

    if ((rfbm & 2ull) != 0ull) {
        if ((xstate_bv & 2ull) != 0ull) {
            for (unsigned i = 0; i < 16u; ++i) {
                std::memcpy(state.xmm[i], raw + 160u + i * 16u, 16);
            }
            state.xinuse |= 2ull;
        } else {
            for (unsigned i = 0; i < 16u; ++i) std::memset(state.xmm[i], 0, 16);
            state.xinuse &= ~2ull;
        }
    }

    if ((rfbm & 4ull) != 0ull) {
        if ((xstate_bv & 4ull) != 0ull) {
            // The compacted AVX slot is the first extended slot, 576, as x87 and SSE live in the legacy area.
            for (unsigned i = 0; i < 16u; ++i) {
                std::memcpy(state.ymm_hi[i], raw + 576u + i * 16u, 16);
            }
            state.xinuse |= 4ull;
        } else {
            for (unsigned i = 0; i < 16u; ++i) std::memset(state.ymm_hi[i], 0, 16);
            state.xinuse &= ~4ull;
        }
    }
}

// XSAVE stores MXCSR and MXCSR_MASK when RFBM selects SSE or AVX (mxcsr_with_avx); XSAVEC only with SSE.
static inline void op_xsave(const XState& state, word address, word rax_val, word rdx_val, bool is64,
                            bool mxcsr_with_avx = true) {
    const std::uint32_t mask_lo = static_cast<std::uint32_t>(rax_val);
    const std::uint32_t mask_hi = static_cast<std::uint32_t>(rdx_val);
    if ((address & 63ull) != 0ull) raise_gp();
    if (mask_hi != 0u || (mask_lo & ~7u) != 0u) raise_gp();
    const std::uint64_t rfbm = static_cast<std::uint64_t>(mask_lo) & 7ull;

    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto* dst = reinterpret_cast<volatile std::uint64_t*>(address);

    if ((rfbm & 1ull) != 0ull) {
        std::uint8_t x87_hdr[24]{};
        std::memcpy(x87_hdr + 0, &state.fcw, 2);
        const std::uint16_t fsw_out = static_cast<std::uint16_t>(state.fsw & 0x7F7Fu);
        std::memcpy(x87_hdr + 2, &fsw_out, 2);
        x87_hdr[4] = state.ftw;
        x87_hdr[5] = 0u;
        const std::uint16_t fop_out = static_cast<std::uint16_t>(state.fop & 0x07FFu);
        std::memcpy(x87_hdr + 6, &fop_out, 2);
        if (is64) {
            std::memcpy(x87_hdr + 8, &state.fip, 8);
            std::memcpy(x87_hdr + 16, &state.fdp, 8);
        } else {
            const auto fip32 = static_cast<std::uint32_t>(state.fip);
            const auto fdp32 = static_cast<std::uint32_t>(state.fdp);
            const bool dep = host_deprecates_fpu_cs_ds() != 0u;
            const std::uint16_t fcs_out = dep ? 0u : state.fcs;
            const std::uint16_t fds_out = dep ? 0u : state.fds;
            std::memcpy(x87_hdr + 8, &fip32, 4);
            std::memcpy(x87_hdr + 12, &fcs_out, 2);
            std::memcpy(x87_hdr + 16, &fdp32, 4);
            std::memcpy(x87_hdr + 20, &fds_out, 2);
        }
        std::uint64_t hdr_words[3]{};
        std::memcpy(hdr_words, x87_hdr, 24);
        for (unsigned i = 0; i < 3u; ++i) dst[i] = hdr_words[i];

        std::uint8_t st_area[128]{};
        const unsigned top = (state.fsw >> 11) & 7u;
        for (unsigned i = 0; i < 8u; ++i) {
            const unsigned phys = (top + i) & 7u;
            std::memcpy(st_area + i * 16u, state.st[phys], 10);
        }
        std::uint64_t st_words[16]{};
        std::memcpy(st_words, st_area, 128);
        for (unsigned i = 0; i < 16u; ++i) dst[4u + i] = st_words[i];
    }

    if ((rfbm & (mxcsr_with_avx ? 6ull : 2ull)) != 0ull) {
        dst[3] = static_cast<std::uint64_t>(state.mxcsr) | (static_cast<std::uint64_t>(state.mxcsr_mask) << 32);
    }

    if ((rfbm & 2ull) != 0ull) {
        std::uint64_t xmm_words[32]{};
        std::memcpy(xmm_words, state.xmm, 256);
        for (unsigned i = 0; i < 32u; ++i) dst[20u + i] = xmm_words[i];
    }

    const std::uint64_t old_bv = dst[64];
    const std::uint64_t new_bv = (old_bv & ~rfbm) | (state.xinuse & rfbm);
    dst[64] = new_bv;

    if ((rfbm & 4ull) != 0ull) {
        std::uint64_t ymm_words[32]{};
        std::memcpy(ymm_words, state.ymm_hi, 256);
        for (unsigned i = 0; i < 32u; ++i) dst[72u + i] = ymm_words[i];
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void op_xsavec(const XState& state, word address, word rax_val, word rdx_val, bool is64) {
    // Every fault check precedes the first store, so a rejected call leaves memory intact.
    if ((address & 63ull) != 0ull) raise_gp();
    if (static_cast<std::uint32_t>(rdx_val) != 0u || (static_cast<std::uint32_t>(rax_val) & ~7u) != 0u) raise_gp();
    const std::uint64_t rfbm = static_cast<std::uint64_t>(rax_val) & 7ull;
    // SDM vol.1 13.6: SSE is also in use when MXCSR differs from its reset value, even if no XMM register is live.
    const std::uint64_t sse_live = state.mxcsr != 0x1F80u ? 2ull : 0ull;
    const std::uint64_t used = rfbm & (state.xinuse | sse_live);
    op_xsave(state, address, used, 0ull, is64, false);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto* dst = reinterpret_cast<volatile std::uint64_t*>(address);
    dst[64] = used;
    dst[65] = (1ull << 63) | rfbm;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void op_xsaveopt(const XState& state, word address, word rax_val, word rdx_val, bool is64) {
    // Every fault check precedes the first store, so a rejected call leaves memory intact.
    if ((address & 63ull) != 0ull) raise_gp();
    if (static_cast<std::uint32_t>(rdx_val) != 0u || (static_cast<std::uint32_t>(rax_val) & ~7u) != 0u) raise_gp();
    const std::uint64_t rfbm = static_cast<std::uint64_t>(rax_val) & 7ull;
    // Init optimization only: SSE is in use while MXCSR is not at its reset value (SDM vol.1 13.6).
    const std::uint64_t sse_live = state.mxcsr != 0x1F80u ? 2ull : 0ull;
    const std::uint64_t inuse = state.xinuse | sse_live;
    op_xsave(state, address, rfbm & inuse, 0ull, is64, false);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto* dst = reinterpret_cast<volatile std::uint64_t*>(address);
    if ((rfbm & 6ull) != 0ull) {
        dst[3] = static_cast<std::uint64_t>(state.mxcsr) | (static_cast<std::uint64_t>(state.mxcsr_mask) << 32);
    }
    dst[64] = (dst[64] & ~rfbm) | (inuse & rfbm);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}
"""


def vec_index(register, ins):
    """Map XMM0..XMM5 or YMM0..YMM5 to (index, is_ymm), rejecting nonvolatile and ZMM."""
    if register in VOLATILE_XMM:
        return VOLATILE_XMM[register], False
    if register in VOLATILE_YMM:
        return VOLATILE_YMM[register], True
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def mm_index(register, ins):
    """Map MM0..MM7 to an index, rejecting all others."""
    if register in MM_REGISTERS:
        return MM_REGISTERS[register]
    raise ValueError(f"unsupported MMX register {register} at RVA {ins.ip:#x}: {ins}")


def gpr_name(register, ins, *, scratch_only=False):
    """Map volatile 32/64-bit GPRs to (base_name, is_32bit), rejecting nonvolatile and narrow GPRs."""
    if register in SCRATCH_GPR64:
        return SCRATCH_GPR64[register], False
    if not scratch_only and register in VOLATILE_GPR64:
        return VOLATILE_GPR64[register], False
    if not scratch_only and register in VOLATILE_GPR32:
        return VOLATILE_GPR32[register], True
    if register in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {register} at RVA {ins.ip:#x}: {ins}")
    if scratch_only and register in ARGUMENTS:
        raise ValueError(f"unsupported argument GPR operand {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported GPR {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, ptr_origins, *, expected_bytes=None, write=False):
    """Lower a flat [base + disp] pointer operand rooted in a live argument provenance."""
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
    if base not in VOLATILE_GPR64:
        raise ValueError(f"unsupported memory base {base} at RVA {ins.ip:#x}: {ins}")
    base_name = VOLATILE_GPR64[base]
    origin = ptr_origins.get(base_name)
    if origin is None:
        if base in ARGUMENTS:
            raise ValueError(f"memory operand uses clobbered pointer {base_name} at RVA {ins.ip:#x}: {ins}")
        raise ValueError(f"unsupported memory base {base} at RVA {ins.ip:#x}: {ins}")
    if write and origin != R.R8:
        raise ValueError(f"non-r8 store memory operand at RVA {ins.ip:#x}: {ins}")
    if expected_bytes is not None and MemorySizeExt.size(ins.memory_size) != expected_bytes:
        raise ValueError(f"expected {expected_bytes}-byte memory operand at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    return f"({base_name} + {disp:#x}ull)" if disp else base_name


def lift_code(name, start, raw, *, entry_state="defined", compacted_xrstor=False):
    """Decode and lift one bounded straight-line XSAVE/XRSTOR leaf function."""
    if entry_state not in ENTRY_STATES:
        raise ValueError(f"unsupported entry-state profile {entry_state!r}")
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    defined_xmm = set()
    defined_ymm = set()
    defined_mm = set()
    used_gpr = set()
    defined_gpr = {"rcx", "rdx", "r8"}
    ptr_origins = {"rcx": R.RCX, "rdx": R.RDX, "r8": R.R8}
    const_gpr = {"rcx": None, "rdx": None, "r8": None, "rax": None, "r9": None, "r10": None, "r11": None}
    defined_xcomp = 0
    # MXCSR is tracked apart from the SSE/AVX components: vzeroupper defines AVX state but never MXCSR.
    defined_mxcsr = False
    used_canonical_entry = False
    mmx_active = False
    statements = []
    returned = False

    def read_vec(op_index, ins, *, want_ymm):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected vector operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx, is_ymm = vec_index(ins.op_register(op_index), ins)
        if is_ymm != want_ymm:
            raise ValueError(f"mismatched vector width at RVA {ins.ip:#x}: {ins}")
        if want_ymm:
            if idx not in defined_ymm:
                raise ValueError(f"undefined vector register ymm{idx} at RVA {ins.ip:#x}: {ins}")
            return f"get_ymm(state, {idx})"
        if idx not in defined_xmm:
            raise ValueError(f"undefined vector register xmm{idx} at RVA {ins.ip:#x}: {ins}")
        return f"get_xmm(state, {idx})"

    def write_vec_idx(op_index, ins, *, want_ymm):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected vector destination at RVA {ins.ip:#x}: {ins}")
        idx, is_ymm = vec_index(ins.op_register(op_index), ins)
        if is_ymm != want_ymm:
            raise ValueError(f"mismatched vector destination width at RVA {ins.ip:#x}: {ins}")
        return idx

    def read_mm(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected MMX operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx = mm_index(ins.op_register(op_index), ins)
        if idx not in defined_mm:
            raise ValueError(f"undefined MMX register mm{idx} at RVA {ins.ip:#x}: {ins}")
        return idx

    def read_gpr(op_index, ins, *, expect_32=None):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR operand {op_index} at RVA {ins.ip:#x}: {ins}")
        reg_name, is32 = gpr_name(ins.op_register(op_index), ins)
        if expect_32 is not None and is32 != expect_32:
            raise ValueError(f"mismatched GPR width at RVA {ins.ip:#x}: {ins}")
        if reg_name not in defined_gpr:
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        used_gpr.add(reg_name)
        return reg_name

    def write_gpr(op_index, ins, *, scratch_only=False, expect_32=None):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR destination at RVA {ins.ip:#x}: {ins}")
        reg_name, is32 = gpr_name(ins.op_register(op_index), ins, scratch_only=scratch_only)
        if expect_32 is not None and is32 != expect_32:
            raise ValueError(f"mismatched GPR width at RVA {ins.ip:#x}: {ins}")
        if reg_name == "r8":
            raise ValueError(f"unsupported write to output buffer pointer r8 at RVA {ins.ip:#x}: {ins}")
        return reg_name

    def clobber_gpr(reg_name, *, copy_from=None):
        if copy_from is not None and copy_from in ptr_origins:
            ptr_origins[reg_name] = ptr_origins[copy_from]
        else:
            ptr_origins.pop(reg_name, None)

    def require_xsave_masks(ins):
        for reg_name in ("rax", "rdx"):
            if reg_name not in defined_gpr:
                raise ValueError(f"undefined mask GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
            used_gpr.add(reg_name)

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
        if ins.code in REJECT_XSAVE_FAMILY_CODES:
            raise ValueError(f"unsupported XSAVE-family instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code in REJECT_FXSAVE_CODES:
            raise ValueError(f"unsupported FXSAVE/FXRSTOR instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            if mmx_active:
                raise ValueError(f"unclosed MMX state before ret at RVA {ins.ip:#x}: {ins}")
            statements.append("return;")
            returned = True

        elif ins.code in XSAVE_CODES or ins.code in XSAVEC_CODES or ins.code in XSAVEOPT_CODES:
            is64 = {**XSAVE_CODES, **XSAVEC_CODES, **XSAVEOPT_CODES}[ins.code]
            op_name = "op_xsavec" if ins.code in XSAVEC_CODES else "op_xsaveopt" if ins.code in XSAVEOPT_CODES else "op_xsave"
            require_xsave_masks(ins)
            addr = flat_address(ins, ptr_origins, write=True)
            rax_val = const_gpr.get("rax")
            rdx_val = const_gpr.get("rdx")
            if rdx_val == 0 and rax_val is not None and (rax_val & ~7) == 0:
                needed_xcomp = rax_val & 7
            else:
                needed_xcomp = 7
            missing = needed_xcomp & ~defined_xcomp
            # xsave and xsaveopt store MXCSR with SSE or AVX; xsavec only with SSE.
            mxcsr_stored = bool(needed_xcomp & (2 if ins.code in XSAVEC_CODES else 6))
            if missing != 0:
                if entry_state != "canonical":
                    missing_names = ", ".join(lbl for bit, lbl in XCOMP_NAMES if missing & bit)
                    raise ValueError(
                        f"xsave observes uninitialized XSAVE component(s) ({missing_names}) before definition; "
                        f"requires explicit --entry-state canonical at RVA {ins.ip:#x}: {ins}"
                    )
                used_canonical_entry = True
            if mxcsr_stored and not defined_mxcsr:
                if entry_state != "canonical":
                    raise ValueError(
                        f"xsave stores MXCSR before any restore defined it; "
                        f"requires explicit --entry-state canonical at RVA {ins.ip:#x}: {ins}"
                    )
                used_canonical_entry = True
            statements.append(f"{op_name}(state, {addr}, rax, rdx, {'true' if is64 else 'false'});")

        elif ins.code in XRSTOR_CODES:
            is64 = XRSTOR_CODES[ins.code]
            require_xsave_masks(ins)
            addr = flat_address(ins, ptr_origins, write=False)
            statements.append(
                f"op_xrstor(state, {addr}, rax, rdx, {'true' if is64 else 'false'}, "
                f"{'true' if compacted_xrstor else 'false'});"
            )
            rax_val = const_gpr.get("rax")
            rdx_val = const_gpr.get("rdx")
            if rdx_val == 0 and rax_val is not None and (rax_val & ~7) == 0:
                defined_xcomp |= rax_val & 7
                # Standard-only XRSTOR defines MXCSR whenever RFBM selects SSE or AVX.
                # When compacted inputs are allowed, only RFBM[1] (SSE) guarantees MXCSR is defined,
                # because compacted XRSTOR with RFBM=4 (AVX only) leaves MXCSR unchanged.
                mxcsr_mask_bits = 2 if compacted_xrstor else 6
                if rax_val & mxcsr_mask_bits:
                    defined_mxcsr = True
            defined_xmm.update(range(6))
            defined_ymm.update(range(6))
            defined_mm.update(range(8))

        elif ins.code == Code.VEX_VZEROUPPER:
            statements.append("op_vzeroupper(state);")
            defined_xcomp |= 4
            defined_ymm.update(defined_xmm)

        elif ins.code in VEX_LOAD_CODES:
            wide, aligned = VEX_LOAD_CODES[ins.code]
            dst_idx = write_vec_idx(0, ins, want_ymm=wide)
            if ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, ptr_origins, expected_bytes=32 if wide else 16)
                src = f"load{'256' if wide else '128'}({addr}, {'true' if aligned else 'false'})"
            else:
                src = read_vec(1, ins, want_ymm=wide)
            statements.append(f"{'set_ymm' if wide else 'set_xmm_vex'}(state, {dst_idx}, {src});")
            defined_xmm.add(dst_idx)
            defined_ymm.add(dst_idx)

        elif ins.code in VEX_STORE_CODES:
            wide, aligned = VEX_STORE_CODES[ins.code]
            src = read_vec(1, ins, want_ymm=wide)
            if ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins, ptr_origins, expected_bytes=32 if wide else 16, write=True)
                statements.append(f"store{'256' if wide else '128'}({addr}, {src}, {'true' if aligned else 'false'});")
            else:
                dst_idx = write_vec_idx(0, ins, want_ymm=wide)
                statements.append(f"{'set_ymm' if wide else 'set_xmm_vex'}(state, {dst_idx}, {src});")
                defined_xmm.add(dst_idx)
                defined_ymm.add(dst_idx)

        elif ins.code in SSE_LOAD_CODES:
            aligned = SSE_LOAD_CODES[ins.code]
            dst_idx = write_vec_idx(0, ins, want_ymm=False)
            if ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, ptr_origins, expected_bytes=16)
                src = f"load128({addr}, {'true' if aligned else 'false'})"
            else:
                src = read_vec(1, ins, want_ymm=False)
            statements.append(f"set_xmm_legacy(state, {dst_idx}, {src});")
            defined_xmm.add(dst_idx)

        elif ins.code in SSE_STORE_CODES:
            aligned = SSE_STORE_CODES[ins.code]
            src = read_vec(1, ins, want_ymm=False)
            if ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins, ptr_origins, expected_bytes=16, write=True)
                statements.append(f"store128({addr}, {src}, {'true' if aligned else 'false'});")
            else:
                dst_idx = write_vec_idx(0, ins, want_ymm=False)
                statements.append(f"set_xmm_legacy(state, {dst_idx}, {src});")
                defined_xmm.add(dst_idx)

        elif ins.code == Code.PADDD_XMM_XMMM128:
            lhs = read_vec(0, ins, want_ymm=False)
            dst_idx = write_vec_idx(0, ins, want_ymm=False)
            if ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, ptr_origins, expected_bytes=16)
                rhs = f"load128({addr}, true)"
            else:
                rhs = read_vec(1, ins, want_ymm=False)
            statements.append(f"set_xmm_legacy(state, {dst_idx}, op_paddd({lhs}, {rhs}));")
            defined_xmm.add(dst_idx)

        elif ins.code == Code.EMMS:
            statements.append("op_emms(state);")
            mmx_active = False

        elif ins.code in MOVQ_MM_CODES:
            if ins.op0_kind == OpKind.MEMORY:
                src_mm = read_mm(1, ins)
                addr = flat_address(ins, ptr_origins, expected_bytes=8, write=True)
                statements.append(f"store64({addr}, get_mm(state, {src_mm}));")
                mmx_active = True
            elif ins.op1_kind == OpKind.MEMORY:
                dst_mm = mm_index(ins.op0_register, ins)
                addr = flat_address(ins, ptr_origins, expected_bytes=8)
                statements.append(f"set_mm(state, {dst_mm}, load64({addr}));")
                defined_mm.add(dst_mm)
                mmx_active = True
            elif ins.code == Code.MOVQ_MM_RM64:
                dst_mm = mm_index(ins.op0_register, ins)
                src_gpr = read_gpr(1, ins, expect_32=False)
                statements.append(f"set_mm(state, {dst_mm}, {src_gpr});")
                defined_mm.add(dst_mm)
                mmx_active = True
            elif ins.code == Code.MOVQ_RM64_MM:
                src_mm = read_mm(1, ins)
                dst_gpr = write_gpr(0, ins, scratch_only=True, expect_32=False)
                clobber_gpr(dst_gpr)
                const_gpr[dst_gpr] = None
                used_gpr.add(dst_gpr)
                statements.append(f"{dst_gpr} = get_mm(state, {src_mm});")
                defined_gpr.add(dst_gpr)
                mmx_active = True
            else:
                raise ValueError(f"unsupported movq operand shape at RVA {ins.ip:#x}: {ins}")

        elif ins.code in (Code.MOV_R64_RM64, Code.MOV_RM64_R64):
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src_gpr = read_gpr(1, ins, expect_32=False)
                dst_gpr = write_gpr(0, ins, expect_32=False)
                clobber_gpr(dst_gpr, copy_from=src_gpr)
                const_gpr[dst_gpr] = const_gpr.get(src_gpr)
                used_gpr.add(dst_gpr)
                statements.append(f"{dst_gpr} = {src_gpr};")
                defined_gpr.add(dst_gpr)
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst_gpr = write_gpr(0, ins, expect_32=False)
                addr = flat_address(ins, ptr_origins, expected_bytes=8)
                clobber_gpr(dst_gpr)
                const_gpr[dst_gpr] = None
                used_gpr.add(dst_gpr)
                statements.append(f"{dst_gpr} = load64({addr});")
                defined_gpr.add(dst_gpr)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src_gpr = read_gpr(1, ins, expect_32=False)
                addr = flat_address(ins, ptr_origins, expected_bytes=8, write=True)
                statements.append(f"store64({addr}, {src_gpr});")
            else:
                raise ValueError(f"unsupported mov operands at RVA {ins.ip:#x}: {ins}")

        elif ins.code in (Code.MOV_R32_RM32, Code.MOV_RM32_R32):
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src_gpr = read_gpr(1, ins, expect_32=True)
                dst_gpr = write_gpr(0, ins, expect_32=True)
                clobber_gpr(dst_gpr)
                src_val = const_gpr.get(src_gpr)
                const_gpr[dst_gpr] = (src_val & 0xFFFFFFFF) if src_val is not None else None
                used_gpr.add(dst_gpr)
                statements.append(f"{dst_gpr} = static_cast<std::uint32_t>({src_gpr});")
                defined_gpr.add(dst_gpr)
            else:
                raise ValueError(f"unsupported 32-bit mov operands at RVA {ins.ip:#x}: {ins}")

        elif ins.code == Code.MOV_R32_IMM32:
            dst_gpr = write_gpr(0, ins, expect_32=True)
            clobber_gpr(dst_gpr)
            used_gpr.add(dst_gpr)
            imm = ins.immediate32 & 0xFFFFFFFF
            const_gpr[dst_gpr] = imm
            statements.append(f"{dst_gpr} = {imm:#x}ull;")
            defined_gpr.add(dst_gpr)

        elif ins.code in (Code.XOR_R32_RM32, Code.XOR_RM32_R32):
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER and ins.op0_register == ins.op1_register:
                dst_gpr = write_gpr(0, ins, expect_32=True)
                clobber_gpr(dst_gpr)
                const_gpr[dst_gpr] = 0
                used_gpr.add(dst_gpr)
                statements.append(f"{dst_gpr} = 0ull;")
                defined_gpr.add(dst_gpr)
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                dst_gpr = write_gpr(0, ins, expect_32=True)
                lhs = read_gpr(0, ins, expect_32=True)
                rhs = read_gpr(1, ins, expect_32=True)
                clobber_gpr(dst_gpr)
                lhs_val = const_gpr.get(lhs)
                rhs_val = const_gpr.get(rhs)
                const_gpr[dst_gpr] = (
                    ((lhs_val ^ rhs_val) & 0xFFFFFFFF)
                    if lhs_val is not None and rhs_val is not None
                    else None
                )
                used_gpr.add(dst_gpr)
                statements.append(f"{dst_gpr} = static_cast<std::uint32_t>({lhs} ^ {rhs});")
                defined_gpr.add(dst_gpr)
            else:
                raise ValueError(f"unsupported xor operands at RVA {ins.ip:#x}: {ins}")

        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = ["    XState state{};", "    state.mxcsr_mask = host_mxcsr_mask();"]
    for gpr in ("rax", "r9", "r10", "r11"):
        if gpr in used_gpr:
            decls.append(f"    word {gpr} = 0ull;")
    body = [f"    {stmt}" for stmt in statements]
    if used_canonical_entry:
        contract = (
            "// Entry-state profile: canonical (caller must establish FCW=0x037F, FSW=0, "
            "FTW=0, FOP=0, FIP=0, FCS=0, FDP=0, FDS=0, MXCSR=0x1F80, ST0..7=0, "
            "XMM0..15=0, YMM_HI0..15=0, XINUSE=0)."
        )
    else:
        contract = f"// Entry-state profile: {entry_state} (all saved XSAVE components defined before xsave)."
    lines = [
        contract,
        f'extern "C" __declspec(dllexport) void lifted_{name}(word rcx, word rdx, word r8) {{',
        "    (void)rcx; (void)rdx; (void)r8;",
        *decls,
        *body,
        "}",
    ]
    return "\n".join(lines), len(instructions)


def lift(pe, name, *, entry_state="defined", compacted_xrstor=False):
    entry = export_rva(pe, name)
    raw = bounded_code(pe, entry)
    source, count = lift_code(name, entry, raw, entry_state=entry_state, compacted_xrstor=compacted_xrstor)
    return source, entry, count


def translate_pe(pe, names, *, entry_state="defined", compacted_xrstor=False):
    if entry_state not in ENTRY_STATES:
        raise ValueError(f"unsupported entry-state profile {entry_state!r}")
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    lifted = [lift(pe, name, entry_state=entry_state, compacted_xrstor=compacted_xrstor) for name in names]
    is_canonical = 1 if entry_state == "canonical" else 0
    is_compacted = 1 if compacted_xrstor else 0
    profile_block = (
        f'\n// Entry-state profile: {entry_state} (compacted_xrstor={is_compacted})\n'
        f'extern "C" __declspec(dllexport) const char* lifted_entry_state_profile() {{ return "{entry_state}"; }}\n'
        f'extern "C" __declspec(dllexport) std::uint32_t lifted_requires_canonical_entry() {{ return {is_canonical}u; }}\n'
        f'extern "C" __declspec(dllexport) std::uint32_t lifted_allows_compacted_xrstor() {{ return {is_compacted}u; }}\n'
    )
    source = PRELUDE + profile_block + "\n".join(f"\n{text}" for text, _, _ in lifted) + "\n"
    meta = [(name, entry, count) for name, (_, entry, count) in zip(names, lifted)]
    return source, meta


def translate(image, names, *, entry_state="defined", compacted_xrstor=False):
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, names, entry_state=entry_state, compacted_xrstor=compacted_xrstor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated export names")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--entry-state",
        choices=ENTRY_STATES,
        default="defined",
        help="entry-state profile: defined (default, requires prior component definition before xsave) or canonical (opt-in caller-initialized canonical XSAVE state)",
    )
    parser.add_argument(
        "--compacted-xrstor",
        action="store_true",
        help="opt-in: allow compacted XRSTOR headers (XCOMP_BV[63] set); under defined entry-state, requires SSE RFBM[1] to define MXCSR",
    )
    args = parser.parse_args()
    try:
        source, meta = translate(
            args.image,
            args.exports.split(","),
            entry_state=args.entry_state,
            compacted_xrstor=args.compacted_xrstor,
        )
        args.output.write_text(source, encoding="utf-8")
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x} ({args.entry_state}): {count} instructions")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
