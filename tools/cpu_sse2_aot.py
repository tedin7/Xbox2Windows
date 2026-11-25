"""Lift bounded Windows x64 leaf functions with legacy SSE2 instructions to C++.

Each export uses the leaf ABI void fn(word rcx, word rdx, word r8), where rcx,
rdx and r8 carry 16-byte-aligned host pointers to caller-owned buffers. XMM0..XMM5 are
modeled as portable 128-bit value structs (two 64-bit lanes). Volatile scratch
GPRs (rax, r9, r10, r11) carry cvtsd2si results and movnti/mov transport
values. Only legacy encodings are accepted; VEX, EVEX and other non-legacy forms reject.

Supported SSE2 instructions (23 of the 64 forms in docs/coverage/x64_zen2.tsv
group SSE2):
  - Packed compare: cmppd (imm8 predicates 0..7, xmm and aligned m128 source)
  - Scalar compare: cmpsd (imm8 predicates 0..7 on the low double, xmm or m64
    source, bits 127:64 of dest preserved; this SSE2 form shares its mnemonic
    with the string compare, which is a different iced Code and is not accepted)
  - Packed converts: cvtdq2pd, cvtdq2ps, cvtpd2dq, cvtpd2ps, cvtps2dq,
    cvtps2pd, cvttpd2dq, cvttps2dq
  - MMX-source convert: cvtpi2pd (m64 memory form only)
  - Scalar convert: cvtsd2si (r32/r64 destination, xmm or m64 source)
  - Scalar double ops: maxsd, minsd, sqrtsd (bits 127:64 of dest preserved)
  - Fences: lfence, mfence (host x64 SSE2 fences)
  - Aligned moves: movapd, movdqa (reg copy, aligned load, aligned store)
  - Half-qword moves: movhpd, movlpd (m64 load merge and m64 store)
  - Non-temporal stores: movntdq, movntpd, movnti (r32 and r64)

Transport/ABI glue: flat argument-rooted 32/64-bit mov between scratch GPRs
and [base + disp], plus terminal ret.

Rejected: the 36 SSE2-group forms reserved for sibling translators (see
docs/experiments.md), the five task forms that cannot be expressed through the
leaf ABI (cvtpd2pi, cvttpd2pi, movdq2q, movq2dq use MMX register operands;
maskmovdqu uses implicit RDI), the mm-register source of cvtpi2pd, cmppd
and cmpsd immediates above 7, non-legacy (VEX/EVEX) encodings, MMX forms, instruction prefixes,
nonvolatile or argument registers, non-flat memory operands, branches and all
other instructions.
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

SCRATCH_GPR32 = {R.EAX: "rax", R.R9D: "r9", R.R10D: "r10", R.R11D: "r11"}
SCRATCH_GPR64 = {R.RAX: "rax", R.R9: "r9", R.R10: "r10", R.R11: "r11"}
NONVOLATILE_GPR = {
    R.EBX, R.RBX, R.EBP, R.RBP, R.ESI, R.RSI, R.EDI, R.RDI, R.ESP, R.RSP,
    R.R12D, R.R12, R.R13D, R.R13, R.R14D, R.R14, R.R15D, R.R15,
}

MM_REGISTERS = {getattr(R, f"MM{i}") for i in range(8)}

# Task forms whose operands live in the MMX register file. mm registers have no
# leaf-ABI value model, so every encoding of these four forms rejects.
MMX_OPERAND_CODES = {
    Code.CVTPD2PI_MM_XMMM128,
    Code.CVTTPD2PI_MM_XMMM128,
    Code.MOVDQ2Q_MM_XMM,
    Code.MOVQ2DQ_XMM_MM,
}

# maskmovdqu stores through implicit DS:rdi, not through a memory operand, so
# its address cannot be bound to an argument buffer.
IMPLICIT_ADDRESS_CODES = {Code.MASKMOVDQU_RDI_XMM_XMM}

# Packed converts whose source is a full 128-bit value (xmm or aligned m128).
PACKED_CVT_HELPERS = {
    Code.CVTDQ2PS_XMM_XMMM128: "op_cvtdq2ps",
    Code.CVTPD2DQ_XMM_XMMM128: "op_cvtpd2dq",
    Code.CVTPD2PS_XMM_XMMM128: "op_cvtpd2ps",
    Code.CVTPS2DQ_XMM_XMMM128: "op_cvtps2dq",
    Code.CVTTPD2DQ_XMM_XMMM128: "op_cvttpd2dq",
    Code.CVTTPS2DQ_XMM_XMMM128: "op_cvttps2dq",
}

# Packed converts whose source is the low 64 bits (xmm low half or m64).
QWORD_CVT_HELPERS = {
    Code.CVTDQ2PD_XMM_XMMM64: "op_cvtdq2pd",
    Code.CVTPS2PD_XMM_XMMM64: "op_cvtps2pd",
}

# Scalar double ops that merge into the destination's low qword.
SCALAR_MERGE_HELPERS = {
    Code.MAXSD_XMM_XMMM64: "op_maxsd_q",
    Code.MINSD_XMM_XMMM64: "op_minsd_q",
}

ALLOWED_CODES = {
    # Terminal and ABI glue.
    Code.RETNQ,
    Code.LFENCE,
    Code.MFENCE,
    Code.MOV_R32_RM32,
    Code.MOV_R64_RM64,
    Code.MOV_RM32_R32,
    Code.MOV_RM64_R64,
    # Packed and scalar compare.
    Code.CMPPD_XMM_XMMM128_IMM8,
    Code.CMPSD_XMM_XMMM64_IMM8,
    # Packed and scalar converts.
    *PACKED_CVT_HELPERS,
    *QWORD_CVT_HELPERS,
    Code.CVTPI2PD_XMM_MMM64,
    Code.CVTSD2SI_R32_XMMM64,
    Code.CVTSD2SI_R64_XMMM64,
    # Scalar double ops.
    *SCALAR_MERGE_HELPERS,
    Code.SQRTSD_XMM_XMMM64,
    # Aligned moves.
    Code.MOVAPD_XMM_XMMM128,
    Code.MOVAPD_XMMM128_XMM,
    Code.MOVDQA_XMM_XMMM128,
    Code.MOVDQA_XMMM128_XMM,
    # Half-qword moves.
    Code.MOVHPD_XMM_M64,
    Code.MOVHPD_M64_XMM,
    Code.MOVLPD_XMM_M64,
    Code.MOVLPD_M64_XMM,
    # Non-temporal stores.
    Code.MOVNTDQ_M128_XMM,
    Code.MOVNTPD_M128_XMM,
    Code.MOVNTI_M32_R32,
    Code.MOVNTI_M64_R64,
}

PRELUDE = """#include <emmintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>

using word = std::uint64_t;

struct Vec128 {
    std::uint64_t q[2];
};

static std::uint32_t load32(word address) {
    std::uint32_t val = 0u;
    std::memcpy(&val, reinterpret_cast<const void*>(address), 4);
    return val;
}

static std::uint64_t load64(word address) {
    std::uint64_t val = 0ull;
    std::memcpy(&val, reinterpret_cast<const void*>(address), 8);
    return val;
}

static void store32(word address, std::uint32_t val) {
    std::memcpy(reinterpret_cast<void*>(address), &val, 4);
}

static void store64(word address, std::uint64_t val) {
    std::memcpy(reinterpret_cast<void*>(address), &val, 8);
}

static Vec128 load128(word address) {
    Vec128 v{};
    std::memcpy(v.q, reinterpret_cast<const void*>(address), 16);
    return v;
}

static void store128(word address, const Vec128& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
}

static double q_as_double(std::uint64_t bits) {
    double v;
    std::memcpy(&v, &bits, 8);
    return v;
}

static std::uint64_t double_as_q(double v) {
    std::uint64_t bits;
    std::memcpy(&bits, &v, 8);
    return bits;
}

static float d_as_float(std::uint32_t bits) {
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

static std::uint32_t float_as_d(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return bits;
}

static std::uint32_t lane32(const Vec128& v, unsigned i) {
    return static_cast<std::uint32_t>(v.q[i >> 1] >> ((i & 1u) * 32u));
}

static void set_lane32(Vec128& v, unsigned i, std::uint32_t val) {
    const unsigned shift = (i & 1u) * 32u;
    v.q[i >> 1] = (v.q[i >> 1] & ~(0xffffffffull << shift)) | (static_cast<std::uint64_t>(val) << shift);
}

static std::int32_t s32(std::uint32_t v) {
    std::int32_t out;
    std::memcpy(&out, &v, 4);
    return out;
}

// CMPPD predicates 0..7: EQ_OQ, LT_OS, LE_OS, UNORD_Q, NEQ_UQ, NLT_US, NLE_US, ORD_Q.
static Vec128 op_cmppd(const Vec128& a, const Vec128& b, std::uint8_t imm8) {
    Vec128 out{};
    for (unsigned i = 0; i < 2u; ++i) {
        const double x = q_as_double(a.q[i]);
        const double y = q_as_double(b.q[i]);
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
        out.q[i] = res ? 0xffffffffffffffffull : 0ull;
    }
    return out;
}

// CMPSD predicates 0..7 on the low double, same predicate order as CMPPD. The
// host SSE2 compare runs on the operands' raw bits, so the live MXCSR (DAZ
// reads subnormal inputs as zero) applies exactly as for the instruction, and
// a NaN input follows the predicate's ordered or unordered rule. The result is
// an all-ones or all-zero qword; the caller keeps bits 127:64 of the destination.
static std::uint64_t op_cmpsd_q(std::uint64_t a, std::uint64_t b, std::uint8_t imm8) {
    const __m128d x = _mm_castsi128_pd(_mm_cvtsi64_si128(static_cast<long long>(a)));
    const __m128d y = _mm_castsi128_pd(_mm_cvtsi64_si128(static_cast<long long>(b)));
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

// MAXSD/MINSD run on the processor, on run-time operands: it returns src2 on ties, unordered inputs and equal
// values, and under MXCSR.DAZ it reads a denormal operand as a zero of its sign, also when that operand is the
// result (a C comparison would return the denormal bits unchanged).
static std::uint64_t op_maxsd_q(std::uint64_t a, std::uint64_t b) {
    volatile std::uint64_t va = a, vb = b;
    const __m128d x = _mm_castsi128_pd(_mm_cvtsi64_si128(static_cast<long long>(va)));
    const __m128d y = _mm_castsi128_pd(_mm_cvtsi64_si128(static_cast<long long>(vb)));
    return static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_castpd_si128(_mm_max_sd(x, y))));
}

static std::uint64_t op_minsd_q(std::uint64_t a, std::uint64_t b) {
    volatile std::uint64_t va = a, vb = b;
    const __m128d x = _mm_castsi128_pd(_mm_cvtsi64_si128(static_cast<long long>(va)));
    const __m128d y = _mm_castsi128_pd(_mm_cvtsi64_si128(static_cast<long long>(vb)));
    return static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_castpd_si128(_mm_min_sd(x, y))));
}

// SQRTSD keeps the NaN payload (quieted) and yields the real-indefinite QNaN
// for negative nonzero inputs, matching the host instruction.
static std::uint64_t op_sqrtsd_q(std::uint64_t b) {
    const double v = q_as_double(b);
    if (v != v) {
        return b | 0x0008000000000000ull;
    }
    if (v < 0.0) {
        return 0xfff8000000000000ull;
    }
    return double_as_q(std::sqrt(v));
}

// MXCSR rounding under the default round-to-nearest-even mode. NaN, infinity
// and out-of-range rounded results return the integer indefinite value.
static std::int32_t cvt_round_i32(double v) {
    const double r = std::nearbyint(v);
    return (r >= -2147483648.0 && r <= 2147483647.0)
               ? static_cast<std::int32_t>(r)
               : static_cast<std::int32_t>(0x80000000u);
}

static std::int32_t cvt_trunc_i32(double v) {
    const double r = std::trunc(v);
    return (r >= -2147483648.0 && r <= 2147483647.0)
               ? static_cast<std::int32_t>(r)
               : static_cast<std::int32_t>(0x80000000u);
}

static std::uint32_t op_cvtsd2si32(std::uint64_t bits) {
    return static_cast<std::uint32_t>(cvt_round_i32(q_as_double(bits)));
}

static std::uint64_t op_cvtsd2si64(std::uint64_t bits) {
    const double r = std::nearbyint(q_as_double(bits));
    const bool valid = (r >= -0x1p63 && r < 0x1p63);
    return static_cast<std::uint64_t>(valid ? static_cast<std::int64_t>(r)
                                            : static_cast<std::int64_t>(0x8000000000000000ull));
}

static Vec128 op_cvtpd2dq(const Vec128& a) {
    const std::uint32_t lo = static_cast<std::uint32_t>(cvt_round_i32(q_as_double(a.q[0])));
    const std::uint32_t hi = static_cast<std::uint32_t>(cvt_round_i32(q_as_double(a.q[1])));
    return {{static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32), 0ull}};
}

static Vec128 op_cvttpd2dq(const Vec128& a) {
    const std::uint32_t lo = static_cast<std::uint32_t>(cvt_trunc_i32(q_as_double(a.q[0])));
    const std::uint32_t hi = static_cast<std::uint32_t>(cvt_trunc_i32(q_as_double(a.q[1])));
    return {{static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32), 0ull}};
}

static Vec128 op_cvtpd2ps(const Vec128& a) {
    const std::uint32_t lo = float_as_d(static_cast<float>(q_as_double(a.q[0])));
    const std::uint32_t hi = float_as_d(static_cast<float>(q_as_double(a.q[1])));
    return {{static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32), 0ull}};
}

static Vec128 op_cvtdq2ps(const Vec128& a) {
    Vec128 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane32(out, i, float_as_d(static_cast<float>(s32(lane32(a, i)))));
    }
    return out;
}

static Vec128 op_cvtps2dq(const Vec128& a) {
    Vec128 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane32(out, i, static_cast<std::uint32_t>(
                               cvt_round_i32(static_cast<double>(d_as_float(lane32(a, i))))));
    }
    return out;
}

static Vec128 op_cvttps2dq(const Vec128& a) {
    Vec128 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane32(out, i, static_cast<std::uint32_t>(
                               cvt_trunc_i32(static_cast<double>(d_as_float(lane32(a, i))))));
    }
    return out;
}

static Vec128 op_cvtdq2pd(std::uint64_t a) {
    const std::int32_t lo = s32(static_cast<std::uint32_t>(a & 0xffffffffull));
    const std::int32_t hi = s32(static_cast<std::uint32_t>(a >> 32));
    return {{double_as_q(static_cast<double>(lo)), double_as_q(static_cast<double>(hi))}};
}

static Vec128 op_cvtps2pd(std::uint64_t a) {
    const std::uint32_t lo = static_cast<std::uint32_t>(a & 0xffffffffull);
    const std::uint32_t hi = static_cast<std::uint32_t>(a >> 32);
    return {{double_as_q(static_cast<double>(d_as_float(lo))),
             double_as_q(static_cast<double>(d_as_float(hi)))}};
}

static Vec128 op_cvtpi2pd(std::uint64_t a) {
    return op_cvtdq2pd(a);
}
"""


def vec_index(register, ins):
    """Map XMM0..XMM5 to an index, rejecting all others."""
    if register in VOLATILE_XMM:
        return VOLATILE_XMM[register]
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def gpr_scratch(register, ins):
    """Map volatile scratch GPR32/GPR64 (rax, r9, r10, r11) to (name, is_64), rejecting others."""
    if register in SCRATCH_GPR32:
        return SCRATCH_GPR32[register], False
    if register in SCRATCH_GPR64:
        return SCRATCH_GPR64[register], True
    if register in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {register} at RVA {ins.ip:#x}: {ins}")
    if register in ARGUMENT_GPRS:
        raise ValueError(f"unsupported argument GPR operand {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported GPR {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, *, alignment=1):
    """Lower a flat [base + disp] pointer operand rooted in an argument GPR64."""
    base = ins.memory_base
    if (ins.is_ip_rel_memory_operand or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
            or ins.memory_index != R.NONE or not RegisterExt.is_gpr64(base)
            or base not in ARGUMENTS):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    if disp % alignment:
        raise ValueError(f"unaligned {alignment}-byte memory operand at RVA {ins.ip:#x}: {ins}")
    name = ARGUMENTS[base]
    return f"({name} + {disp:#x}ull)" if disp else name


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line legacy SSE2 leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used = set()
    defined = set()
    used_gpr = set()
    defined_gpr = set()
    statements = []
    returned = False

    def read_vec(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        register = ins.op_register(op_index)
        if register in MM_REGISTERS:
            raise ValueError(f"MMX register operand outside the leaf ABI at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(register, ins)
        if idx not in defined:
            raise ValueError(f"undefined vector register xmm{idx} at RVA {ins.ip:#x}: {ins}")
        used.add(idx)
        return f"xmm{idx}"

    def write_vec(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        register = ins.op_register(op_index)
        if register in MM_REGISTERS:
            raise ValueError(f"MMX register operand outside the leaf ABI at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(register, ins)
        used.add(idx)
        return idx, f"xmm{idx}"

    def read_gpr(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR operand {op_index} at RVA {ins.ip:#x}: {ins}")
        reg_name, is64 = gpr_scratch(ins.op_register(op_index), ins)
        if reg_name not in defined_gpr:
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        used_gpr.add(reg_name)
        return reg_name, is64

    def write_gpr(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR destination at RVA {ins.ip:#x}: {ins}")
        reg_name, is64 = gpr_scratch(ins.op_register(op_index), ins)
        used_gpr.add(reg_name)
        return reg_name, is64

    def vec_src128(op_index, ins):
        """Source of a full-width vector operand: xmm name or m128 load expression."""
        if ins.op_kind(op_index) == OpKind.MEMORY:
            return f"load128({flat_address(ins, alignment=16)})"
        return read_vec(op_index, ins)

    def vec_src64(op_index, ins):
        """Source of a low-qword operand: xmm.q[0] or m64 load64 expression."""
        if ins.op_kind(op_index) == OpKind.MEMORY:
            return f"load64({flat_address(ins)})"
        return f"{read_vec(op_index, ins)}.q[0]"

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        if ins.op_code().encoding != EncodingKind.LEGACY:
            raise ValueError(f"non-legacy encoding at RVA {ins.ip:#x}: {ins}")
        if ins.code in MMX_OPERAND_CODES:
            raise ValueError(f"MMX register operand outside the leaf ABI at RVA {ins.ip:#x}: {ins}")
        if ins.code in IMPLICIT_ADDRESS_CODES:
            raise ValueError(f"implicit RDI memory operand outside the leaf ABI at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            statements.append("return;")
            returned = True
        elif ins.code == Code.LFENCE:
            statements.append("_mm_lfence();")
        elif ins.code == Code.MFENCE:
            statements.append("_mm_mfence();")
        elif ins.code in (Code.MOV_R32_RM32, Code.MOV_R64_RM64):
            if ins.op0_kind != OpKind.REGISTER or ins.op1_kind != OpKind.MEMORY:
                raise ValueError(f"unsupported mov instruction at RVA {ins.ip:#x}: {ins}")
            reg_name, is64 = write_gpr(0, ins)
            addr = flat_address(ins)
            if is64:
                statements.append(f"{reg_name} = load64({addr});")
            else:
                statements.append(f"{reg_name} = static_cast<word>(load32({addr}));")
            defined_gpr.add(reg_name)
        elif ins.code in (Code.MOV_RM32_R32, Code.MOV_RM64_R64):
            if ins.op0_kind != OpKind.MEMORY or ins.op1_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported mov instruction at RVA {ins.ip:#x}: {ins}")
            reg_name, is64 = read_gpr(1, ins)
            addr = flat_address(ins)
            if is64:
                statements.append(f"store64({addr}, {reg_name});")
            else:
                statements.append(f"store32({addr}, static_cast<std::uint32_t>({reg_name}));")
        elif ins.code in (Code.MOVNTI_M32_R32, Code.MOVNTI_M64_R64):
            reg_name, is64 = read_gpr(1, ins)
            addr = flat_address(ins)
            if ins.code == Code.MOVNTI_M64_R64:
                if not is64:
                    raise ValueError(f"movnti width mismatch at RVA {ins.ip:#x}: {ins}")
                statements.append(f"store64({addr}, {reg_name});")
            else:
                statements.append(f"store32({addr}, static_cast<std::uint32_t>({reg_name}));")
        elif ins.code == Code.CMPPD_XMM_XMMM128_IMM8:
            imm = ins.immediate8 & 0xff
            if imm > 7:
                raise ValueError(f"unsupported cmppd predicate {imm:#x} at RVA {ins.ip:#x}: {ins}")
            dst = read_vec(0, ins)
            src = vec_src128(1, ins)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = op_cmppd({dst}, {src}, {imm:#x}u);")
            defined.add(dst_idx)
        elif ins.code == Code.CMPSD_XMM_XMMM64_IMM8:
            imm = ins.immediate8 & 0xff
            if imm > 7:
                raise ValueError(f"unsupported cmpsd predicate {imm:#x} at RVA {ins.ip:#x}: {ins}")
            dst = read_vec(0, ins)
            src = vec_src64(1, ins)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name}.q[0] = op_cmpsd_q({dst}.q[0], {src}, {imm:#x}u);")
            defined.add(dst_idx)
        elif ins.code in PACKED_CVT_HELPERS:
            src = vec_src128(1, ins)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = {PACKED_CVT_HELPERS[ins.code]}({src});")
            defined.add(dst_idx)
        elif ins.code in QWORD_CVT_HELPERS:
            src = vec_src64(1, ins)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = {QWORD_CVT_HELPERS[ins.code]}({src});")
            defined.add(dst_idx)
        elif ins.code == Code.CVTPI2PD_XMM_MMM64:
            if ins.op1_kind != OpKind.MEMORY:
                raise ValueError(f"MMX register operand outside the leaf ABI at RVA {ins.ip:#x}: {ins}")
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = op_cvtpi2pd(load64({flat_address(ins)}));")
            defined.add(dst_idx)
        elif ins.code in (Code.CVTSD2SI_R32_XMMM64, Code.CVTSD2SI_R64_XMMM64):
            src = vec_src64(1, ins)
            reg_name, is64 = write_gpr(0, ins)
            if ins.code == Code.CVTSD2SI_R64_XMMM64:
                if not is64:
                    raise ValueError(f"cvtsd2si width mismatch at RVA {ins.ip:#x}: {ins}")
                statements.append(f"{reg_name} = op_cvtsd2si64({src});")
            else:
                if is64:
                    raise ValueError(f"cvtsd2si width mismatch at RVA {ins.ip:#x}: {ins}")
                statements.append(f"{reg_name} = static_cast<word>(op_cvtsd2si32({src}));")
            defined_gpr.add(reg_name)
        elif ins.code in SCALAR_MERGE_HELPERS:
            dst = read_vec(0, ins)
            src = vec_src64(1, ins)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name}.q[0] = {SCALAR_MERGE_HELPERS[ins.code]}({dst}.q[0], {src});")
            defined.add(dst_idx)
        elif ins.code == Code.SQRTSD_XMM_XMMM64:
            dst = read_vec(0, ins)
            src = vec_src64(1, ins)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name}.q[0] = op_sqrtsd_q({src});")
            defined.add(dst_idx)
        elif ins.code in (Code.MOVAPD_XMM_XMMM128, Code.MOVDQA_XMM_XMMM128):
            if ins.op1_kind == OpKind.MEMORY:
                dst_idx, dst_name = write_vec(0, ins)
                statements.append(f"{dst_name} = load128({flat_address(ins, alignment=16)});")
            else:
                src = read_vec(1, ins)
                dst_idx, dst_name = write_vec(0, ins)
                statements.append(f"{dst_name} = {src};")
            defined.add(dst_idx)
        elif ins.code in (Code.MOVAPD_XMMM128_XMM, Code.MOVDQA_XMMM128_XMM):
            src = read_vec(1, ins)
            statements.append(f"store128({flat_address(ins, alignment=16)}, {src});")
        elif ins.code in (Code.MOVHPD_XMM_M64, Code.MOVLPD_XMM_M64):
            dst = read_vec(0, ins)
            dst_idx, dst_name = write_vec(0, ins)
            q_idx = 1 if ins.code == Code.MOVHPD_XMM_M64 else 0
            statements.append(f"{dst_name}.q[{q_idx}] = load64({flat_address(ins)});")
            defined.add(dst_idx)
        elif ins.code in (Code.MOVHPD_M64_XMM, Code.MOVLPD_M64_XMM):
            src = read_vec(1, ins)
            q_idx = 1 if ins.code == Code.MOVHPD_M64_XMM else 0
            statements.append(f"store64({flat_address(ins)}, {src}.q[{q_idx}]);")
        elif ins.code in (Code.MOVNTDQ_M128_XMM, Code.MOVNTPD_M128_XMM):
            src = read_vec(1, ins)
            statements.append(f"store128({flat_address(ins, alignment=16)}, {src});")
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec128 xmm{idx}{{}};" for idx in sorted(used)]
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
