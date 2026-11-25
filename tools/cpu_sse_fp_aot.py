"""Lift bounded Windows x64 leaf functions with legacy SSE FP arithmetic, compares and moves to C++.

Each export uses the leaf ABI void fn(word rcx, word rdx, word r8), where rcx,
rdx and r8 carry 16-byte-aligned host pointers to caller-owned buffers (rcx
and rdx are inputs, r8 is the output). XMM0..XMM5 are modeled as portable
128-bit value structs (two 64-bit lanes). Only legacy encodings are accepted;
VEX, EVEX and other non-legacy forms reject.

Supported SSE and SSE2 instructions (32 FP forms from docs/coverage/x64_zen2.tsv groups SSE, SSE2):
  - Packed float arithmetic: addps, subps, mulps, divps, minps, maxps (xmm and
    aligned m128 source, destination read and replaced) and sqrtps (xmm and
    aligned m128 source, destination only written)
  - Packed float bitwise logic: andps, andnps, orps, xorps (xmm and aligned
    m128 source; xorps xmmN, xmmN self-zeroing accepts an undefined xmmN)
  - Packed double arithmetic: addpd, subpd, mulpd, divpd, minpd, maxpd and sqrtpd,
    with the same operand rules (two 64-bit lanes)
  - Packed double bitwise logic: andpd, andnpd, orpd, xorpd (xmm and aligned
    m128 source; xorpd xmmN, xmmN self-zeroing accepts an undefined xmmN)
  - Packed float compare: cmpps (imm8 predicates 0..7, xmm and aligned m128)
  - Packed double compare: cmppd (imm8 predicates 0..7, xmm and aligned m128; imm8 >= 8 is
    an AVX-only predicate and rejects; each 64-bit lane becomes an all-ones or all-zero mask)
  - Scalar float compare: cmpss (imm8 predicates 0..7, xmm and m32; bits 127:32 preserved)
  - Scalar double compare: cmpsd (imm8 predicates 0..7, xmm and m64; bits 127:64 preserved)
  - Half-qword float moves: movlps, movhps (m64 load-merge preserving the other
    64 bits, and m64 store)
  - Half-qword double moves: movlpd, movhpd (m64 load-merge preserving the other
    64 bits, and m64 store)
  - Non-temporal stores: movntps, movntpd (aligned m128 store)

Arithmetic runs under the host MXCSR: the generated code executes the same
SSE operation at run time, so the rounding mode, DAZ and FTZ are the host's
and the translator never folds a result at translation time. A guest MXCSR
write (ldmxcsr, vldmxcsr) is not lifted, exception flags are not observed and
the other scalar forms, conversions, rcpps and rsqrtps stay rejected. NaN operands are resolved in software (the first
source operand that is a NaN, made quiet, else the second) so that a compiler
which commutes an addition cannot change a payload.

Transport/ABI glue: movaps, movdqa (aligned 128-bit load/store/reg copy),
movdqu (unaligned 128-bit load/store/reg copy) and terminal ret. All memory
stores must target an r8-rooted flat address; loads may read from rcx, rdx or
r8. Full-width 128-bit aligned operands (cmpps m128, cmppd m128, movntps, movntpd,
movaps m128, movdqa m128) require displacements divisible by 16.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import Code, Decoder, EncodingKind, OpKind, Register as R, RegisterExt

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}
VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = {getattr(R, f"XMM{i}") for i in range(6, 16)}
NONVOLATILE_GPR = {
    R.EBX, R.RBX, R.EBP, R.RBP, R.ESI, R.RSI, R.EDI, R.RDI, R.ESP, R.RSP,
    R.R12D, R.R12, R.R13D, R.R13, R.R14D, R.R14, R.R15D, R.R15,
}

BINARY_ARITH_FUNCTIONS = {
    Code.ADDPS_XMM_XMMM128: "op_addps",
    Code.SUBPS_XMM_XMMM128: "op_subps",
    Code.MULPS_XMM_XMMM128: "op_mulps",
    Code.DIVPS_XMM_XMMM128: "op_divps",
    Code.MINPS_XMM_XMMM128: "op_minps",
    Code.MAXPS_XMM_XMMM128: "op_maxps",
    Code.ADDPD_XMM_XMMM128: "op_addpd",
    Code.SUBPD_XMM_XMMM128: "op_subpd",
    Code.MULPD_XMM_XMMM128: "op_mulpd",
    Code.DIVPD_XMM_XMMM128: "op_divpd",
    Code.MINPD_XMM_XMMM128: "op_minpd",
    Code.MAXPD_XMM_XMMM128: "op_maxpd",
}

BITWISE_LOGIC_FUNCTIONS = {
    Code.ANDPS_XMM_XMMM128: "op_andps",
    Code.ANDNPS_XMM_XMMM128: "op_andnps",
    Code.ORPS_XMM_XMMM128: "op_orps",
    Code.XORPS_XMM_XMMM128: "op_xorps",
    Code.ANDPD_XMM_XMMM128: "op_andps",
    Code.ANDNPD_XMM_XMMM128: "op_andnps",
    Code.ORPD_XMM_XMMM128: "op_orps",
    Code.XORPD_XMM_XMMM128: "op_xorps",
}

LOAD_ALIGNED_VEC_CODES = {Code.MOVAPS_XMM_XMMM128, Code.MOVDQA_XMM_XMMM128}
STORE_ALIGNED_VEC_CODES = {Code.MOVAPS_XMMM128_XMM, Code.MOVDQA_XMMM128_XMM}

ALLOWED_CODES = {
    Code.RETNQ,
    *BINARY_ARITH_FUNCTIONS,
    *BITWISE_LOGIC_FUNCTIONS,
    Code.SQRTPS_XMM_XMMM128,
    Code.SQRTPD_XMM_XMMM128,
    Code.CMPPS_XMM_XMMM128_IMM8,
    Code.CMPPD_XMM_XMMM128_IMM8,
    Code.CMPSS_XMM_XMMM32_IMM8,
    Code.CMPSD_XMM_XMMM64_IMM8,
    Code.MOVLPS_XMM_M64,
    Code.MOVLPS_M64_XMM,
    Code.MOVHPS_XMM_M64,
    Code.MOVHPS_M64_XMM,
    Code.MOVLPD_XMM_M64,
    Code.MOVLPD_M64_XMM,
    Code.MOVHPD_XMM_M64,
    Code.MOVHPD_M64_XMM,
    Code.MOVNTPS_M128_XMM,
    Code.MOVNTPD_M128_XMM,
    Code.MOVAPS_XMM_XMMM128,
    Code.MOVAPS_XMMM128_XMM,
    Code.MOVDQA_XMM_XMMM128,
    Code.MOVDQA_XMMM128_XMM,
    Code.MOVDQU_XMM_XMMM128,
    Code.MOVDQU_XMMM128_XMM,
}

PRELUDE = """#include <atomic>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>

using word = std::uint64_t;

struct Vec128 {
    std::uint64_t q[2];
};

// Guest reads can fault even when the translated result is dead. Keep surrounding guest
// writes on their original side of the read when the compiler inlines these helpers.
static inline std::uint32_t load32(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto value = *reinterpret_cast<const volatile std::uint32_t __unaligned*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return value;
}

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

static inline float d_as_float(std::uint32_t bits) {
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

static inline std::uint32_t lane32(const Vec128& v, unsigned i) {
    return static_cast<std::uint32_t>(v.q[i >> 1] >> ((i & 1u) * 32u));
}

static inline void set_lane32(Vec128& v, unsigned i, std::uint32_t val) {
    const unsigned shift = (i & 1u) * 32u;
    v.q[i >> 1] = (v.q[i >> 1] & ~(0xffffffffull << shift)) | (static_cast<std::uint64_t>(val) << shift);
}

// CMPPS/CMPSS predicates 0..7: EQ_OQ, LT_OS, LE_OS, UNORD_Q, NEQ_UQ, NLT_US, NLE_US, ORD_Q.
static inline std::uint32_t cmp_f32(float x, float y, std::uint8_t imm8) {
    bool res;
    switch (imm8 & 7u) {
        case 0u: res = (x == y); break;
        case 1u: res = (x < y); break;
        case 2u: res = (x <= y); break;
        case 3u: res = (x != x) || (y != y); break;
        case 4u: res = !(x == y); break;
        case 5u: res = !(x < y); break;
        case 6u: res = !(x <= y); break;
        default: res = (x == x) && (y == y); break;
    }
    return res ? 0xffffffffu : 0u;
}

static inline Vec128 op_cmpps(const Vec128& a, const Vec128& b, std::uint8_t imm8) {
    Vec128 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane32(out, i, cmp_f32(d_as_float(lane32(a, i)), d_as_float(lane32(b, i)), imm8));
    }
    return out;
}

static inline __m128 to_m128(const Vec128& v) {
    __m128 out;
    std::memcpy(&out, v.q, sizeof(out));
    return out;
}

static inline Vec128 from_m128(__m128 v) {
    Vec128 out;
    std::memcpy(out.q, &v, sizeof(out.q));
    return out;
}

static inline bool is_nan_bits(std::uint32_t bits) {
    return (bits & 0x7fffffffu) > 0x7f800000u;
}

// A lane with a NaN source returns the first source operand when it is a NaN, else the
// second one, in both cases made quiet. The hardware does the same, but a compiler may
// swap the operands of a commutative operation, so the choice is made here. Lanes
// without a NaN source keep the result of the host instruction, which follows the host
// MXCSR (rounding mode, DAZ and FTZ).
static inline Vec128 with_nan_sources(const Vec128& a, const Vec128& b, Vec128 result) {
    for (unsigned i = 0; i < 4u; ++i) {
        const std::uint32_t x = lane32(a, i);
        const std::uint32_t y = lane32(b, i);
        if (is_nan_bits(x)) {
            set_lane32(result, i, x | 0x00400000u);
        } else if (is_nan_bits(y)) {
            set_lane32(result, i, y | 0x00400000u);
        }
    }
    return result;
}

static inline Vec128 op_addps(const Vec128& a, const Vec128& b) {
    return with_nan_sources(a, b, from_m128(_mm_add_ps(to_m128(a), to_m128(b))));
}

static inline Vec128 op_subps(const Vec128& a, const Vec128& b) {
    return with_nan_sources(a, b, from_m128(_mm_sub_ps(to_m128(a), to_m128(b))));
}

static inline Vec128 op_mulps(const Vec128& a, const Vec128& b) {
    return with_nan_sources(a, b, from_m128(_mm_mul_ps(to_m128(a), to_m128(b))));
}

static inline Vec128 op_divps(const Vec128& a, const Vec128& b) {
    return with_nan_sources(a, b, from_m128(_mm_div_ps(to_m128(a), to_m128(b))));
}

// MINPS and MAXPS are not commutative: equal operands (including +0 and -0) and any NaN
// return the second source operand, a signaling NaN included. The host instruction is
// used as is, so denormal inputs follow the host DAZ bit.
static inline Vec128 op_minps(const Vec128& a, const Vec128& b) {
    return from_m128(_mm_min_ps(to_m128(a), to_m128(b)));
}

static inline Vec128 op_maxps(const Vec128& a, const Vec128& b) {
    return from_m128(_mm_max_ps(to_m128(a), to_m128(b)));
}

// Only the source is read. A NaN lane comes back quiet with its payload; a negative
// operand other than -0 gives the host's default NaN.
static inline Vec128 op_sqrtps(const Vec128& src) {
    Vec128 result = from_m128(_mm_sqrt_ps(to_m128(src)));
    for (unsigned i = 0; i < 4u; ++i) {
        const std::uint32_t x = lane32(src, i);
        if (is_nan_bits(x)) {
            set_lane32(result, i, x | 0x00400000u);
        }
    }
    return result;
}

static inline __m128d to_m128d(const Vec128& v) {
    __m128d out;
    std::memcpy(&out, v.q, sizeof(out));
    return out;
}

static inline Vec128 from_m128d(__m128d v) {
    Vec128 out;
    std::memcpy(out.q, &v, sizeof(out.q));
    return out;
}

// CMPPD predicates 0..7 on the host instruction, so the processor decides NaN operands and, under
// DAZ, treats denormal operands as zero (C operators and the C library may differ). Each 64-bit
// lane is all ones or all zero. The operands are read through volatile copies: a compiler must not
// fold the compare, which depends on the run-time MXCSR.
static inline __m128d cmp_f64(__m128d x, __m128d y, std::uint8_t imm8) {
    switch (imm8 & 7u) {
        case 0u: return _mm_cmpeq_pd(x, y);
        case 1u: return _mm_cmplt_pd(x, y);
        case 2u: return _mm_cmple_pd(x, y);
        case 3u: return _mm_cmpunord_pd(x, y);
        case 4u: return _mm_cmpneq_pd(x, y);
        case 5u: return _mm_cmpnlt_pd(x, y);
        case 6u: return _mm_cmpnle_pd(x, y);
        default: return _mm_cmpord_pd(x, y);
    }
}

static inline Vec128 op_cmppd(const Vec128& a, const Vec128& b, std::uint8_t imm8) {
    const volatile Vec128 va = a;
    const volatile Vec128 vb = b;
    const Vec128 x = {{va.q[0], va.q[1]}};
    const Vec128 y = {{vb.q[0], vb.q[1]}};
    return from_m128d(cmp_f64(to_m128d(x), to_m128d(y), imm8));
}

static inline bool is_nan64_bits(std::uint64_t bits) {
    return (bits & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
}

// The double lanes follow the single rule: the first NaN source, made quiet, else the second.
// The host instruction supplies every other lane, so rounding, DAZ and FTZ stay the host's.
static inline Vec128 with_nan_sources_pd(const Vec128& a, const Vec128& b, Vec128 result) {
    for (unsigned i = 0; i < 2u; ++i) {
        if (is_nan64_bits(a.q[i])) {
            result.q[i] = a.q[i] | 0x0008000000000000ull;
        } else if (is_nan64_bits(b.q[i])) {
            result.q[i] = b.q[i] | 0x0008000000000000ull;
        }
    }
    return result;
}

static inline Vec128 op_addpd(const Vec128& a, const Vec128& b) {
    return with_nan_sources_pd(a, b, from_m128d(_mm_add_pd(to_m128d(a), to_m128d(b))));
}

static inline Vec128 op_subpd(const Vec128& a, const Vec128& b) {
    return with_nan_sources_pd(a, b, from_m128d(_mm_sub_pd(to_m128d(a), to_m128d(b))));
}

static inline Vec128 op_mulpd(const Vec128& a, const Vec128& b) {
    return with_nan_sources_pd(a, b, from_m128d(_mm_mul_pd(to_m128d(a), to_m128d(b))));
}

static inline Vec128 op_divpd(const Vec128& a, const Vec128& b) {
    return with_nan_sources_pd(a, b, from_m128d(_mm_div_pd(to_m128d(a), to_m128d(b))));
}

// MINPD and MAXPD return the second source operand for equal values and for any NaN, as the
// single forms do; the host instruction is used as is.
static inline Vec128 op_minpd(const Vec128& a, const Vec128& b) {
    return from_m128d(_mm_min_pd(to_m128d(a), to_m128d(b)));
}

static inline Vec128 op_maxpd(const Vec128& a, const Vec128& b) {
    return from_m128d(_mm_max_pd(to_m128d(a), to_m128d(b)));
}

// The root comes from the host instruction on a run-time operand (never the C library, which
// can disagree with the hardware on denormal inputs under DAZ). A NaN lane comes back quiet.
static inline Vec128 op_sqrtpd(const Vec128& src) {
    Vec128 result = from_m128d(_mm_sqrt_pd(to_m128d(src)));
    for (unsigned i = 0; i < 2u; ++i) {
        if (is_nan64_bits(src.q[i])) {
            result.q[i] = src.q[i] | 0x0008000000000000ull;
        }
    }
    return result;
}

static inline std::uint32_t op_cmpss_d(std::uint32_t a, std::uint32_t b, std::uint8_t imm8) {
    return cmp_f32(d_as_float(a), d_as_float(b), imm8);
}

// CMPSD predicates 0..7 on the low double via host SSE2, so live MXCSR (DAZ) and NaN rules match
// hardware. Volatile copies prevent compile-time folding across MXCSR changes.
static inline std::uint64_t op_cmpsd_q(std::uint64_t a, std::uint64_t b, std::uint8_t imm8) {
    const volatile std::uint64_t va = a;
    const volatile std::uint64_t vb = b;
    const __m128d x = _mm_castsi128_pd(_mm_cvtsi64_si128(static_cast<long long>(va)));
    const __m128d y = _mm_castsi128_pd(_mm_cvtsi64_si128(static_cast<long long>(vb)));
    __m128d r;
    switch (imm8 & 7u) {
        case 0u: r = _mm_cmpeq_sd(x, y); break;
        case 1u: r = _mm_cmplt_sd(x, y); break;
        case 2u: r = _mm_cmple_sd(x, y); break;
        case 3u: r = _mm_cmpunord_sd(x, y); break;
        case 4u: r = _mm_cmpneq_sd(x, y); break;
        case 5u: r = _mm_cmpnlt_sd(x, y); break;
        case 6u: r = _mm_cmpnle_sd(x, y); break;
        default: r = _mm_cmpord_sd(x, y); break;
    }
    return static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_castpd_si128(r)));
}

// Bitwise SSE packed-FP logic preserves all 128 bits identically (NaNs, -0.0, denormals, Inf).
static inline Vec128 op_andps(const Vec128& dst, const Vec128& src) {
    return {{dst.q[0] & src.q[0], dst.q[1] & src.q[1]}};
}

static inline Vec128 op_andnps(const Vec128& dst, const Vec128& src) {
    return {{(~dst.q[0]) & src.q[0], (~dst.q[1]) & src.q[1]}};
}

static inline Vec128 op_orps(const Vec128& dst, const Vec128& src) {
    return {{dst.q[0] | src.q[0], dst.q[1] | src.q[1]}};
}

static inline Vec128 op_xorps(const Vec128& dst, const Vec128& src) {
    return {{dst.q[0] ^ src.q[0], dst.q[1] ^ src.q[1]}};
}

// Keep self-zeroed vectors opaque to /O2 constant folding so (+0.0 + -0.0) and MXCSR rules hold.
static inline Vec128 op_zero_vec() {
    const volatile Vec128 z{};
    return {{z.q[0], z.q[1]}};
}
"""


def vec_index(register, ins):
    """Map XMM0..XMM5 to an index, rejecting nonvolatile and unsupported registers."""
    if register in VOLATILE_XMM:
        return VOLATILE_XMM[register]
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, *, alignment=1, write=False):
    """Lower a flat [base + disp] pointer operand rooted in an argument GPR64."""
    base = ins.memory_base
    if (ins.is_ip_rel_memory_operand or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
            or ins.memory_index != R.NONE or not RegisterExt.is_gpr64(base)):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    if base in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR base {base} in memory operand at RVA {ins.ip:#x}: {ins}")
    if base not in ARGUMENTS:
        raise ValueError(f"unsupported memory base {base} at RVA {ins.ip:#x}: {ins}")
    if write and base != R.R8:
        raise ValueError(f"non-r8 store memory operand at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    if disp % alignment:
        raise ValueError(f"unaligned {alignment}-byte memory operand at RVA {ins.ip:#x}: {ins}")
    name = ARGUMENTS[base]
    return f"({name} + {disp:#x}ull)" if disp else name


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line legacy SSE FP leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used = set()
    defined = set()
    statements = []
    returned = False

    def read_vec(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(ins.op_register(op_index), ins)
        if idx not in defined:
            raise ValueError(f"undefined vector register xmm{idx} at RVA {ins.ip:#x}: {ins}")
        used.add(idx)
        return f"xmm{idx}"

    def write_vec(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(ins.op_register(op_index), ins)
        used.add(idx)
        return idx, f"xmm{idx}"

    def vec_src128(op_index, ins, *, alignment=16):
        if ins.op_kind(op_index) == OpKind.MEMORY:
            return f"load128({flat_address(ins, alignment=alignment)})"
        return read_vec(op_index, ins)

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        if ins.op_code().encoding != EncodingKind.LEGACY:
            raise ValueError(f"non-legacy encoding at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            statements.append("return;")
            returned = True
        elif ins.code in BINARY_ARITH_FUNCTIONS:
            dst = read_vec(0, ins)
            src = vec_src128(1, ins, alignment=16)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = {BINARY_ARITH_FUNCTIONS[ins.code]}({dst}, {src});")
            defined.add(dst_idx)
        elif ins.code in BITWISE_LOGIC_FUNCTIONS:
            if (
                ins.code in (Code.XORPS_XMM_XMMM128, Code.XORPD_XMM_XMMM128)
                and ins.op0_kind == OpKind.REGISTER
                and ins.op1_kind == OpKind.REGISTER
                and ins.op0_register == ins.op1_register
            ):
                dst_idx, dst_name = write_vec(0, ins)
                statements.append(f"{dst_name} = op_zero_vec();")
            else:
                dst = read_vec(0, ins)
                src = vec_src128(1, ins, alignment=16)
                dst_idx, dst_name = write_vec(0, ins)
                statements.append(f"{dst_name} = {BITWISE_LOGIC_FUNCTIONS[ins.code]}({dst}, {src});")
            defined.add(dst_idx)
        elif ins.code == Code.SQRTPS_XMM_XMMM128:
            src = vec_src128(1, ins, alignment=16)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = op_sqrtps({src});")
            defined.add(dst_idx)
        elif ins.code == Code.SQRTPD_XMM_XMMM128:
            src = vec_src128(1, ins, alignment=16)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = op_sqrtpd({src});")
            defined.add(dst_idx)
        elif ins.code == Code.CMPPS_XMM_XMMM128_IMM8:
            imm = ins.immediate8 & 0xff
            if imm > 7:
                raise ValueError(f"unsupported cmpps predicate {imm:#x} at RVA {ins.ip:#x}: {ins}")
            dst = read_vec(0, ins)
            src = vec_src128(1, ins, alignment=16)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = op_cmpps({dst}, {src}, {imm:#x}u);")
            defined.add(dst_idx)
        elif ins.code == Code.CMPPD_XMM_XMMM128_IMM8:
            imm = ins.immediate8 & 0xff
            if imm > 7:
                raise ValueError(f"unsupported cmppd predicate {imm:#x} at RVA {ins.ip:#x}: {ins}")
            dst = read_vec(0, ins)
            src = vec_src128(1, ins, alignment=16)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = op_cmppd({dst}, {src}, {imm:#x}u);")
            defined.add(dst_idx)
        elif ins.code == Code.CMPSS_XMM_XMMM32_IMM8:
            imm = ins.immediate8 & 0xff
            if imm > 7:
                raise ValueError(f"unsupported cmpss predicate {imm:#x} at RVA {ins.ip:#x}: {ins}")
            dst = read_vec(0, ins)
            if ins.op1_kind == OpKind.MEMORY:
                src32 = f"load32({flat_address(ins)})"
            else:
                src32 = f"lane32({read_vec(1, ins)}, 0u)"
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(
                f"set_lane32({dst_name}, 0u, op_cmpss_d(lane32({dst}, 0u), {src32}, {imm:#x}u));"
            )
            defined.add(dst_idx)
        elif ins.code == Code.CMPSD_XMM_XMMM64_IMM8:
            imm = ins.immediate8 & 0xff
            if imm > 7:
                raise ValueError(f"unsupported cmpsd predicate {imm:#x} at RVA {ins.ip:#x}: {ins}")
            dst = read_vec(0, ins)
            if ins.op1_kind == OpKind.MEMORY:
                src64 = f"load64({flat_address(ins)})"
            else:
                src64 = f"{read_vec(1, ins)}.q[0]"
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name}.q[0] = op_cmpsd_q({dst}.q[0], {src64}, {imm:#x}u);")
            defined.add(dst_idx)
        elif ins.code in (
            Code.MOVLPS_XMM_M64, Code.MOVHPS_XMM_M64,
            Code.MOVLPD_XMM_M64, Code.MOVHPD_XMM_M64,
        ):
            if ins.op1_kind != OpKind.MEMORY:
                raise ValueError(f"expected memory source at RVA {ins.ip:#x}: {ins}")
            read_vec(0, ins)
            dst_idx, dst_name = write_vec(0, ins)
            q_idx = 1 if ins.code in (Code.MOVHPS_XMM_M64, Code.MOVHPD_XMM_M64) else 0
            statements.append(f"{dst_name}.q[{q_idx}] = load64({flat_address(ins)});")
            defined.add(dst_idx)
        elif ins.code in (
            Code.MOVLPS_M64_XMM, Code.MOVHPS_M64_XMM,
            Code.MOVLPD_M64_XMM, Code.MOVHPD_M64_XMM,
        ):
            if ins.op0_kind != OpKind.MEMORY:
                raise ValueError(f"expected memory destination at RVA {ins.ip:#x}: {ins}")
            src = read_vec(1, ins)
            q_idx = 1 if ins.code in (Code.MOVHPS_M64_XMM, Code.MOVHPD_M64_XMM) else 0
            statements.append(f"store64({flat_address(ins, write=True)}, {src}.q[{q_idx}]);")
        elif ins.code in (Code.MOVNTPS_M128_XMM, Code.MOVNTPD_M128_XMM):
            if ins.op0_kind != OpKind.MEMORY:
                raise ValueError(f"expected memory destination at RVA {ins.ip:#x}: {ins}")
            src = read_vec(1, ins)
            statements.append(f"store128({flat_address(ins, alignment=16, write=True)}, {src});")
        elif ins.code in (*LOAD_ALIGNED_VEC_CODES, Code.MOVDQU_XMM_XMMM128):
            align = 16 if ins.code in LOAD_ALIGNED_VEC_CODES else 1
            src = vec_src128(1, ins, alignment=align)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = {src};")
            defined.add(dst_idx)
        elif ins.code in (*STORE_ALIGNED_VEC_CODES, Code.MOVDQU_XMMM128_XMM):
            align = 16 if ins.code in STORE_ALIGNED_VEC_CODES else 1
            src = read_vec(1, ins)
            if ins.op0_kind == OpKind.MEMORY:
                statements.append(
                    f"store128({flat_address(ins, alignment=align, write=True)}, {src});"
                )
            else:
                dst_idx, dst_name = write_vec(0, ins)
                statements.append(f"{dst_name} = {src};")
                defined.add(dst_idx)
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec128 xmm{idx}{{}};" for idx in sorted(used)]
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
