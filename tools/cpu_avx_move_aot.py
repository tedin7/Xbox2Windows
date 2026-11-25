"""Lift bounded Windows x64 leaf functions with AVX move/shuffle/blend/logical instructions to C++.

Each export uses the leaf ABI void fn(word rcx, word rdx, word r8), where rcx,
rdx and r8 carry flat host pointers to caller-owned buffers. YMM0..YMM5 are
modeled as portable 256-bit value structs (four 64-bit lanes); every VEX write
to an XMM destination zeroes bits 128..255 of the corresponding YMM register.
Volatile scratch GPRs (rax/eax, r9/r9d, r10/r10d, r11/r11d) support GPR-mask
and scalar transfer instructions (vmovmskpd, vmovmskps, vextractps) plus flat
argument-rooted mov load/store glue.

Supported AVX instructions (50 surviving forms from docs/coverage/x64_zen2.tsv
group AVX, excluding vmovdqa/vmovups/vmovaps/vmovq/vmovd carved out to
cpu_vex_aot.py (PR #325); vmovdqu remains as transport glue):
  - Logical bitwise ops: vandpd, vandps, vandnpd, vandnps, vorpd, vorps,
    vxorpd, vxorps
  - Blends: vblendpd, vblendps, vblendvpd, vblendvps
  - Broadcasts: vbroadcastf128 (m128), vbroadcastsd (m64/xmm),
    vbroadcastss (m32/xmm)
  - Lane ops: vextractf128, vinsertf128, vextractps, vinsertps
  - Permutes: vperm2f128, vpermilpd (imm8 & reg), vpermilps (imm8 & reg)
  - Shuffles: vshufpd, vshufps
  - Unpacks: vunpcklpd, vunpcklps, vunpckhpd, vunpckhps
  - Vector & scalar moves: vmovapd, vmovupd, vlddqu, vmovntdq, vmovntdqa,
    vmovntpd, vmovntps, vmovddup, vmovshdup, vmovsldup, vmovhlps, vmovlhps,
    vmovhpd, vmovhps, vmovlpd, vmovlps, vmovsd, vmovss, vmovmskpd, vmovmskps
  - Masked loads and stores: vmaskmovpd, vmaskmovps

Transport/ABI glue: vmovdqu (128/256-bit flat load, store and register copy),
flat argument-rooted 32/64-bit mov between volatile scratch GPRs and
[base + disp], vzeroupper and terminal ret.

All operations transfer or combine raw bits with no FP arithmetic, no NaN
canonicalization and no MXCSR involvement. The eight logical ops also take a flat
[base + disp] memory operand as their last source (m128 for VEX.128, m256 for
VEX.256), read as raw bits. Memory operands on the 20 other register-only AVX
forms that have memory encodings, the carved-out move forms
(vmovdqa, vmovups, vmovaps, vmovq, vmovd), nonvolatile vector or GPR registers,
EVEX encodings, instruction prefixes, unsupported addressing modes, branches
and all other instructions are rejected.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import Code, Decoder, EncodingKind, Mnemonic as M, OpKind, Register as R, RegisterExt

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}
VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
VOLATILE_YMM = {getattr(R, f"YMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = ({getattr(R, f"XMM{i}") for i in range(6, 16)}
                   | {getattr(R, f"YMM{i}") for i in range(6, 16)})

SCRATCH_GPR32 = {R.EAX: "rax", R.R9D: "r9", R.R10D: "r10", R.R11D: "r11"}
SCRATCH_GPR64 = {R.RAX: "rax", R.R9: "r9", R.R10: "r10", R.R11: "r11"}
NONVOLATILE_GPR = {
    R.EBX, R.RBX, R.EBP, R.RBP, R.ESI, R.RSI, R.EDI, R.RDI,
    R.R12D, R.R12, R.R13D, R.R13, R.R14D, R.R14, R.R15D, R.R15,
}

# Purely register-domain AVX forms in this translator's policy (memory operands rejected).
# The eight logical ops are not listed: their last source may be a flat memory operand.
REGISTER_ONLY_AVX = {
    M.VBLENDPD, M.VBLENDPS, M.VBLENDVPD, M.VBLENDVPS,
    M.VEXTRACTF128, M.VINSERTF128, M.VEXTRACTPS, M.VINSERTPS,
    M.VPERM2F128, M.VPERMILPD, M.VPERMILPS,
    M.VSHUFPD, M.VSHUFPS,
    M.VUNPCKLPD, M.VUNPCKLPS, M.VUNPCKHPD, M.VUNPCKHPS,
    M.VMOVDDUP, M.VMOVSHDUP, M.VMOVSLDUP,
}

# Full-width moves (vmovapd/vmovupd plus vmovdqu transport glue).
FULL_VEC_MOVES = {
    M.VMOVAPD, M.VMOVDQU, M.VMOVUPD,
}

NT_STORES = {M.VMOVNTDQ, M.VMOVNTPD, M.VMOVNTPS}
PLAIN_LOADS = {M.VLDDQU, M.VMOVNTDQA}

LOGICAL_HELPERS = {
    M.VANDPD: "op_and",
    M.VANDPS: "op_and",
    M.VANDNPD: "op_andn",
    M.VANDNPS: "op_andn",
    M.VORPD: "op_or",
    M.VORPS: "op_or",
    M.VXORPD: "op_xor",
    M.VXORPS: "op_xor",
}

UNPACK_HELPERS = {
    M.VUNPCKLPD: "op_vunpcklpd",
    M.VUNPCKHPD: "op_vunpckhpd",
    M.VUNPCKLPS: "op_vunpcklps",
    M.VUNPCKHPS: "op_vunpckhps",
}

DUP_HELPERS = {
    M.VMOVDDUP: "op_vmovddup",
    M.VMOVSHDUP: "op_vmovshdup",
    M.VMOVSLDUP: "op_vmovsldup",
}

# (is_store, element_bits, is_ymm)
MASKMOV_SPECS = {
    Code.VEX_VMASKMOVPS_XMM_XMM_M128: (False, 32, False),
    Code.VEX_VMASKMOVPS_YMM_YMM_M256: (False, 32, True),
    Code.VEX_VMASKMOVPD_XMM_XMM_M128: (False, 64, False),
    Code.VEX_VMASKMOVPD_YMM_YMM_M256: (False, 64, True),
    Code.VEX_VMASKMOVPS_M128_XMM_XMM: (True, 32, False),
    Code.VEX_VMASKMOVPS_M256_YMM_YMM: (True, 32, True),
    Code.VEX_VMASKMOVPD_M128_XMM_XMM: (True, 64, False),
    Code.VEX_VMASKMOVPD_M256_YMM_YMM: (True, 64, True),
}

ALLOWED_CODES = {
    # Transport / ABI glue
    Code.VEX_VZEROUPPER,
    Code.RETNQ,
    Code.VEX_VMOVDQU_XMM_XMMM128,
    Code.VEX_VMOVDQU_XMMM128_XMM,
    Code.VEX_VMOVDQU_YMM_YMMM256,
    Code.VEX_VMOVDQU_YMMM256_YMM,
    Code.MOV_R32_RM32,
    Code.MOV_R64_RM64,
    Code.MOV_RM32_R32,
    Code.MOV_RM64_R64,
    # Logical bitwise ops (8 forms)
    Code.VEX_VANDPD_XMM_XMM_XMMM128,
    Code.VEX_VANDPD_YMM_YMM_YMMM256,
    Code.VEX_VANDPS_XMM_XMM_XMMM128,
    Code.VEX_VANDPS_YMM_YMM_YMMM256,
    Code.VEX_VANDNPD_XMM_XMM_XMMM128,
    Code.VEX_VANDNPD_YMM_YMM_YMMM256,
    Code.VEX_VANDNPS_XMM_XMM_XMMM128,
    Code.VEX_VANDNPS_YMM_YMM_YMMM256,
    Code.VEX_VORPD_XMM_XMM_XMMM128,
    Code.VEX_VORPD_YMM_YMM_YMMM256,
    Code.VEX_VORPS_XMM_XMM_XMMM128,
    Code.VEX_VORPS_YMM_YMM_YMMM256,
    Code.VEX_VXORPD_XMM_XMM_XMMM128,
    Code.VEX_VXORPD_YMM_YMM_YMMM256,
    Code.VEX_VXORPS_XMM_XMM_XMMM128,
    Code.VEX_VXORPS_YMM_YMM_YMMM256,
    # Blends (4 forms)
    Code.VEX_VBLENDPD_XMM_XMM_XMMM128_IMM8,
    Code.VEX_VBLENDPD_YMM_YMM_YMMM256_IMM8,
    Code.VEX_VBLENDPS_XMM_XMM_XMMM128_IMM8,
    Code.VEX_VBLENDPS_YMM_YMM_YMMM256_IMM8,
    Code.VEX_VBLENDVPD_XMM_XMM_XMMM128_XMM,
    Code.VEX_VBLENDVPD_YMM_YMM_YMMM256_YMM,
    Code.VEX_VBLENDVPS_XMM_XMM_XMMM128_XMM,
    Code.VEX_VBLENDVPS_YMM_YMM_YMMM256_YMM,
    # Broadcasts (3 forms)
    Code.VEX_VBROADCASTF128_YMM_M128,
    Code.VEX_VBROADCASTSD_YMM_M64,
    Code.VEX_VBROADCASTSD_YMM_XMM,
    Code.VEX_VBROADCASTSS_XMM_M32,
    Code.VEX_VBROADCASTSS_YMM_M32,
    Code.VEX_VBROADCASTSS_XMM_XMM,
    Code.VEX_VBROADCASTSS_YMM_XMM,
    # Lane ops (4 forms)
    Code.VEX_VEXTRACTF128_XMMM128_YMM_IMM8,
    Code.VEX_VINSERTF128_YMM_YMM_XMMM128_IMM8,
    Code.VEX_VEXTRACTPS_RM32_XMM_IMM8,
    Code.VEX_VEXTRACTPS_R64M32_XMM_IMM8,
    Code.VEX_VINSERTPS_XMM_XMM_XMMM32_IMM8,
    # Permutes (3 forms)
    Code.VEX_VPERM2F128_YMM_YMM_YMMM256_IMM8,
    Code.VEX_VPERMILPD_XMM_XMMM128_IMM8,
    Code.VEX_VPERMILPD_YMM_YMMM256_IMM8,
    Code.VEX_VPERMILPD_XMM_XMM_XMMM128,
    Code.VEX_VPERMILPD_YMM_YMM_YMMM256,
    Code.VEX_VPERMILPS_XMM_XMMM128_IMM8,
    Code.VEX_VPERMILPS_YMM_YMMM256_IMM8,
    Code.VEX_VPERMILPS_XMM_XMM_XMMM128,
    Code.VEX_VPERMILPS_YMM_YMM_YMMM256,
    # Shuffles (2 forms)
    Code.VEX_VSHUFPD_XMM_XMM_XMMM128_IMM8,
    Code.VEX_VSHUFPD_YMM_YMM_YMMM256_IMM8,
    Code.VEX_VSHUFPS_XMM_XMM_XMMM128_IMM8,
    Code.VEX_VSHUFPS_YMM_YMM_YMMM256_IMM8,
    # Unpacks (4 forms)
    Code.VEX_VUNPCKLPD_XMM_XMM_XMMM128,
    Code.VEX_VUNPCKLPD_YMM_YMM_YMMM256,
    Code.VEX_VUNPCKHPD_XMM_XMM_XMMM128,
    Code.VEX_VUNPCKHPD_YMM_YMM_YMMM256,
    Code.VEX_VUNPCKLPS_XMM_XMM_XMMM128,
    Code.VEX_VUNPCKLPS_YMM_YMM_YMMM256,
    Code.VEX_VUNPCKHPS_XMM_XMM_XMMM128,
    Code.VEX_VUNPCKHPS_YMM_YMM_YMMM256,
    # Full-width moves (2 surviving forms: vmovapd, vmovupd)
    Code.VEX_VMOVAPD_XMM_XMMM128,
    Code.VEX_VMOVAPD_XMMM128_XMM,
    Code.VEX_VMOVAPD_YMM_YMMM256,
    Code.VEX_VMOVAPD_YMMM256_YMM,
    Code.VEX_VMOVUPD_XMM_XMMM128,
    Code.VEX_VMOVUPD_XMMM128_XMM,
    Code.VEX_VMOVUPD_YMM_YMMM256,
    Code.VEX_VMOVUPD_YMMM256_YMM,
    # Unaligned load and non-temporal moves (5 forms)
    Code.VEX_VLDDQU_XMM_M128,
    Code.VEX_VLDDQU_YMM_M256,
    Code.VEX_VMOVNTDQ_M128_XMM,
    Code.VEX_VMOVNTDQ_M256_YMM,
    Code.VEX_VMOVNTDQA_XMM_M128,
    Code.VEX_VMOVNTDQA_YMM_M256,
    Code.VEX_VMOVNTPD_M128_XMM,
    Code.VEX_VMOVNTPD_M256_YMM,
    Code.VEX_VMOVNTPS_M128_XMM,
    Code.VEX_VMOVNTPS_M256_YMM,
    # Dup moves (3 forms)
    Code.VEX_VMOVDDUP_XMM_XMMM64,
    Code.VEX_VMOVDDUP_YMM_YMMM256,
    Code.VEX_VMOVSHDUP_XMM_XMMM128,
    Code.VEX_VMOVSHDUP_YMM_YMMM256,
    Code.VEX_VMOVSLDUP_XMM_XMMM128,
    Code.VEX_VMOVSLDUP_YMM_YMMM256,
    # Half-XMM register and memory moves (6 forms)
    Code.VEX_VMOVHLPS_XMM_XMM_XMM,
    Code.VEX_VMOVLHPS_XMM_XMM_XMM,
    Code.VEX_VMOVHPD_XMM_XMM_M64,
    Code.VEX_VMOVHPD_M64_XMM,
    Code.VEX_VMOVHPS_XMM_XMM_M64,
    Code.VEX_VMOVHPS_M64_XMM,
    Code.VEX_VMOVLPD_XMM_XMM_M64,
    Code.VEX_VMOVLPD_M64_XMM,
    Code.VEX_VMOVLPS_XMM_XMM_M64,
    Code.VEX_VMOVLPS_M64_XMM,
    # Scalar moves (2 surviving forms: vmovsd, vmovss)
    Code.VEX_VMOVSD_XMM_M64,
    Code.VEX_VMOVSD_M64_XMM,
    Code.VEX_VMOVSD_XMM_XMM_XMM,
    Code.VEX_VMOVSD_XMM_XMM_XMM_0F11,
    Code.VEX_VMOVSS_XMM_M32,
    Code.VEX_VMOVSS_M32_XMM,
    Code.VEX_VMOVSS_XMM_XMM_XMM,
    Code.VEX_VMOVSS_XMM_XMM_XMM_0F11,
    # Sign-bit mask to GPR (2 forms)
    Code.VEX_VMOVMSKPD_R32_XMM,
    Code.VEX_VMOVMSKPD_R32_YMM,
    Code.VEX_VMOVMSKPD_R64_XMM,
    Code.VEX_VMOVMSKPD_R64_YMM,
    Code.VEX_VMOVMSKPS_R32_XMM,
    Code.VEX_VMOVMSKPS_R32_YMM,
    Code.VEX_VMOVMSKPS_R64_XMM,
    Code.VEX_VMOVMSKPS_R64_YMM,
    # Masked loads and stores (2 forms)
    *MASKMOV_SPECS,
}

PRELUDE = """#include <atomic>
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#define X2W_UNALIGNED __unaligned
#else
#define X2W_UNALIGNED
#endif

using word = std::uint64_t;

struct Vec256 {
    std::uint64_t q[4];
};

static inline std::uint32_t load32(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto val = *reinterpret_cast<const volatile std::uint32_t X2W_UNALIGNED*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return val;
}

static inline std::uint64_t load64(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto val = *reinterpret_cast<const volatile std::uint64_t X2W_UNALIGNED*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return val;
}

static inline void store32(word address, std::uint32_t val) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), &val, 4);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store64(word address, std::uint64_t val) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), &val, 8);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline Vec256 load128(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* p = reinterpret_cast<const volatile std::uint64_t X2W_UNALIGNED*>(address);
    const std::uint64_t q0 = p[0];
    const std::uint64_t q1 = p[1];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, 0ull, 0ull}};
}

static inline Vec256 load256(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* p = reinterpret_cast<const volatile std::uint64_t X2W_UNALIGNED*>(address);
    const std::uint64_t q0 = p[0];
    const std::uint64_t q1 = p[1];
    const std::uint64_t q2 = p[2];
    const std::uint64_t q3 = p[3];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, q2, q3}};
}

static inline Vec256 load_vec32(word address) {
    Vec256 v{};
    v.q[0] = static_cast<std::uint64_t>(load32(address));
    return v;
}

static inline Vec256 load_vec64(word address) {
    Vec256 v{};
    v.q[0] = load64(address);
    return v;
}

static inline void store128(word address, const Vec256& v) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store256(word address, const Vec256& v) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), v.q, 32);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static Vec256 copy128(const Vec256& v) {
    return {{v.q[0], v.q[1], 0ull, 0ull}};
}

static std::uint32_t lane32(const Vec256& v, unsigned i) {
    return static_cast<std::uint32_t>(v.q[i >> 1] >> ((i & 1u) * 32u));
}

static void set_lane32(Vec256& v, unsigned i, std::uint32_t val) {
    const unsigned shift = (i & 1u) * 32u;
    v.q[i >> 1] = (v.q[i >> 1] & ~(0xffffffffull << shift)) | (static_cast<std::uint64_t>(val) << shift);
}

static Vec256 op_and(const Vec256& a, const Vec256& b, bool wide) {
    Vec256 out{};
    const unsigned n = wide ? 4u : 2u;
    for (unsigned i = 0; i < n; ++i) out.q[i] = a.q[i] & b.q[i];
    return out;
}

static Vec256 op_andn(const Vec256& a, const Vec256& b, bool wide) {
    Vec256 out{};
    const unsigned n = wide ? 4u : 2u;
    for (unsigned i = 0; i < n; ++i) out.q[i] = (~a.q[i]) & b.q[i];
    return out;
}

static Vec256 op_or(const Vec256& a, const Vec256& b, bool wide) {
    Vec256 out{};
    const unsigned n = wide ? 4u : 2u;
    for (unsigned i = 0; i < n; ++i) out.q[i] = a.q[i] | b.q[i];
    return out;
}

static Vec256 op_xor(const Vec256& a, const Vec256& b, bool wide) {
    Vec256 out{};
    const unsigned n = wide ? 4u : 2u;
    for (unsigned i = 0; i < n; ++i) out.q[i] = a.q[i] ^ b.q[i];
    return out;
}

static Vec256 op_vblendpd(const Vec256& a, const Vec256& b, std::uint8_t imm8, bool wide) {
    Vec256 out{};
    const unsigned n = wide ? 4u : 2u;
    for (unsigned i = 0; i < n; ++i) out.q[i] = ((imm8 >> i) & 1u) ? b.q[i] : a.q[i];
    return out;
}

static Vec256 op_vblendps(const Vec256& a, const Vec256& b, std::uint8_t imm8, bool wide) {
    Vec256 out{};
    const unsigned n = wide ? 8u : 4u;
    for (unsigned i = 0; i < n; ++i) {
        set_lane32(out, i, ((imm8 >> i) & 1u) ? lane32(b, i) : lane32(a, i));
    }
    return out;
}

static Vec256 op_vblendvpd(const Vec256& a, const Vec256& b, const Vec256& mask, bool wide) {
    Vec256 out{};
    const unsigned n = wide ? 4u : 2u;
    for (unsigned i = 0; i < n; ++i) {
        out.q[i] = (mask.q[i] & 0x8000000000000000ull) ? b.q[i] : a.q[i];
    }
    return out;
}

static Vec256 op_vblendvps(const Vec256& a, const Vec256& b, const Vec256& mask, bool wide) {
    Vec256 out{};
    const unsigned n = wide ? 8u : 4u;
    for (unsigned i = 0; i < n; ++i) {
        set_lane32(out, i, (lane32(mask, i) & 0x80000000u) ? lane32(b, i) : lane32(a, i));
    }
    return out;
}

static Vec256 op_vbroadcastf128(word address) {
    Vec256 v = load128(address);
    v.q[2] = v.q[0];
    v.q[3] = v.q[1];
    return v;
}

static Vec256 op_vbroadcastsd_q(std::uint64_t q) {
    return {{q, q, q, q}};
}

static Vec256 op_vbroadcastss_d(std::uint32_t d, bool wide) {
    const std::uint64_t q = static_cast<std::uint64_t>(d) | (static_cast<std::uint64_t>(d) << 32);
    return wide ? Vec256{{q, q, q, q}} : Vec256{{q, q, 0ull, 0ull}};
}

static Vec256 op_vextractf128(const Vec256& src, std::uint8_t imm8) {
    if (imm8 & 1u) return {{src.q[2], src.q[3], 0ull, 0ull}};
    return {{src.q[0], src.q[1], 0ull, 0ull}};
}

static Vec256 op_vinsertf128(const Vec256& src1, const Vec256& src2, std::uint8_t imm8) {
    if (imm8 & 1u) return {{src1.q[0], src1.q[1], src2.q[0], src2.q[1]}};
    return {{src2.q[0], src2.q[1], src1.q[2], src1.q[3]}};
}

static word op_vextractps(const Vec256& src, std::uint8_t imm8) {
    return static_cast<word>(lane32(src, imm8 & 3u));
}

static Vec256 op_vinsertps(const Vec256& src1, const Vec256& src2, std::uint8_t imm8) {
    Vec256 out{{src1.q[0], src1.q[1], 0ull, 0ull}};
    const std::uint32_t tmp = lane32(src2, (imm8 >> 6) & 3u);
    set_lane32(out, (imm8 >> 4) & 3u, tmp);
    for (unsigned i = 0; i < 4u; ++i) {
        if ((imm8 >> i) & 1u) set_lane32(out, i, 0u);
    }
    return out;
}

static void pick_lane128(const Vec256& src1, const Vec256& src2, unsigned sel,
                         std::uint64_t& lo, std::uint64_t& hi) {
    if (sel & 8u) {
        lo = 0ull;
        hi = 0ull;
        return;
    }
    switch (sel & 3u) {
        case 0u: lo = src1.q[0]; hi = src1.q[1]; break;
        case 1u: lo = src1.q[2]; hi = src1.q[3]; break;
        case 2u: lo = src2.q[0]; hi = src2.q[1]; break;
        default: lo = src2.q[2]; hi = src2.q[3]; break;
    }
}

static Vec256 op_vperm2f128(const Vec256& src1, const Vec256& src2, std::uint8_t imm8) {
    Vec256 out{};
    pick_lane128(src1, src2, imm8 & 0xfu, out.q[0], out.q[1]);
    pick_lane128(src1, src2, (imm8 >> 4) & 0xfu, out.q[2], out.q[3]);
    return out;
}

static Vec256 op_vpermilpd_imm(const Vec256& src, std::uint8_t imm8, bool wide) {
    Vec256 out{};
    out.q[0] = ((imm8 >> 0) & 1u) ? src.q[1] : src.q[0];
    out.q[1] = ((imm8 >> 1) & 1u) ? src.q[1] : src.q[0];
    if (wide) {
        out.q[2] = ((imm8 >> 2) & 1u) ? src.q[3] : src.q[2];
        out.q[3] = ((imm8 >> 3) & 1u) ? src.q[3] : src.q[2];
    }
    return out;
}

static Vec256 op_vpermilpd_var(const Vec256& src, const Vec256& ctrl, bool wide) {
    Vec256 out{};
    out.q[0] = ((ctrl.q[0] >> 1) & 1ull) ? src.q[1] : src.q[0];
    out.q[1] = ((ctrl.q[1] >> 1) & 1ull) ? src.q[1] : src.q[0];
    if (wide) {
        out.q[2] = ((ctrl.q[2] >> 1) & 1ull) ? src.q[3] : src.q[2];
        out.q[3] = ((ctrl.q[3] >> 1) & 1ull) ? src.q[3] : src.q[2];
    }
    return out;
}

static Vec256 op_vpermilps_imm(const Vec256& src, std::uint8_t imm8, bool wide) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane32(out, i, lane32(src, (imm8 >> (2u * i)) & 3u));
    }
    if (wide) {
        for (unsigned i = 0; i < 4u; ++i) {
            set_lane32(out, 4u + i, lane32(src, 4u + ((imm8 >> (2u * i)) & 3u)));
        }
    }
    return out;
}

static Vec256 op_vpermilps_var(const Vec256& src, const Vec256& ctrl, bool wide) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane32(out, i, lane32(src, lane32(ctrl, i) & 3u));
    }
    if (wide) {
        for (unsigned i = 0; i < 4u; ++i) {
            set_lane32(out, 4u + i, lane32(src, 4u + (lane32(ctrl, 4u + i) & 3u)));
        }
    }
    return out;
}

static Vec256 op_vshufpd(const Vec256& a, const Vec256& b, std::uint8_t imm8, bool wide) {
    Vec256 out{};
    out.q[0] = ((imm8 >> 0) & 1u) ? a.q[1] : a.q[0];
    out.q[1] = ((imm8 >> 1) & 1u) ? b.q[1] : b.q[0];
    if (wide) {
        out.q[2] = ((imm8 >> 2) & 1u) ? a.q[3] : a.q[2];
        out.q[3] = ((imm8 >> 3) & 1u) ? b.q[3] : b.q[2];
    }
    return out;
}

static Vec256 op_vshufps(const Vec256& a, const Vec256& b, std::uint8_t imm8, bool wide) {
    Vec256 out{};
    set_lane32(out, 0u, lane32(a, (imm8 >> 0) & 3u));
    set_lane32(out, 1u, lane32(a, (imm8 >> 2) & 3u));
    set_lane32(out, 2u, lane32(b, (imm8 >> 4) & 3u));
    set_lane32(out, 3u, lane32(b, (imm8 >> 6) & 3u));
    if (wide) {
        set_lane32(out, 4u, lane32(a, 4u + ((imm8 >> 0) & 3u)));
        set_lane32(out, 5u, lane32(a, 4u + ((imm8 >> 2) & 3u)));
        set_lane32(out, 6u, lane32(b, 4u + ((imm8 >> 4) & 3u)));
        set_lane32(out, 7u, lane32(b, 4u + ((imm8 >> 6) & 3u)));
    }
    return out;
}

static Vec256 op_vunpcklpd(const Vec256& a, const Vec256& b, bool wide) {
    return wide ? Vec256{{a.q[0], b.q[0], a.q[2], b.q[2]}}
                : Vec256{{a.q[0], b.q[0], 0ull, 0ull}};
}

static Vec256 op_vunpckhpd(const Vec256& a, const Vec256& b, bool wide) {
    return wide ? Vec256{{a.q[1], b.q[1], a.q[3], b.q[3]}}
                : Vec256{{a.q[1], b.q[1], 0ull, 0ull}};
}

static Vec256 op_vunpcklps(const Vec256& a, const Vec256& b, bool wide) {
    Vec256 out{};
    set_lane32(out, 0u, lane32(a, 0u));
    set_lane32(out, 1u, lane32(b, 0u));
    set_lane32(out, 2u, lane32(a, 1u));
    set_lane32(out, 3u, lane32(b, 1u));
    if (wide) {
        set_lane32(out, 4u, lane32(a, 4u));
        set_lane32(out, 5u, lane32(b, 4u));
        set_lane32(out, 6u, lane32(a, 5u));
        set_lane32(out, 7u, lane32(b, 5u));
    }
    return out;
}

static Vec256 op_vunpckhps(const Vec256& a, const Vec256& b, bool wide) {
    Vec256 out{};
    set_lane32(out, 0u, lane32(a, 2u));
    set_lane32(out, 1u, lane32(b, 2u));
    set_lane32(out, 2u, lane32(a, 3u));
    set_lane32(out, 3u, lane32(b, 3u));
    if (wide) {
        set_lane32(out, 4u, lane32(a, 6u));
        set_lane32(out, 5u, lane32(b, 6u));
        set_lane32(out, 6u, lane32(a, 7u));
        set_lane32(out, 7u, lane32(b, 7u));
    }
    return out;
}

static Vec256 op_vmovddup(const Vec256& a, bool wide) {
    return wide ? Vec256{{a.q[0], a.q[0], a.q[2], a.q[2]}}
                : Vec256{{a.q[0], a.q[0], 0ull, 0ull}};
}

static Vec256 op_vmovshdup(const Vec256& a, bool wide) {
    Vec256 out{};
    set_lane32(out, 0u, lane32(a, 1u));
    set_lane32(out, 1u, lane32(a, 1u));
    set_lane32(out, 2u, lane32(a, 3u));
    set_lane32(out, 3u, lane32(a, 3u));
    if (wide) {
        set_lane32(out, 4u, lane32(a, 5u));
        set_lane32(out, 5u, lane32(a, 5u));
        set_lane32(out, 6u, lane32(a, 7u));
        set_lane32(out, 7u, lane32(a, 7u));
    }
    return out;
}

static Vec256 op_vmovsldup(const Vec256& a, bool wide) {
    Vec256 out{};
    set_lane32(out, 0u, lane32(a, 0u));
    set_lane32(out, 1u, lane32(a, 0u));
    set_lane32(out, 2u, lane32(a, 2u));
    set_lane32(out, 3u, lane32(a, 2u));
    if (wide) {
        set_lane32(out, 4u, lane32(a, 4u));
        set_lane32(out, 5u, lane32(a, 4u));
        set_lane32(out, 6u, lane32(a, 6u));
        set_lane32(out, 7u, lane32(a, 6u));
    }
    return out;
}

static Vec256 op_vmovhlps(const Vec256& a, const Vec256& b) {
    return {{b.q[1], a.q[1], 0ull, 0ull}};
}

static Vec256 op_vmovlhps(const Vec256& a, const Vec256& b) {
    return {{a.q[0], b.q[0], 0ull, 0ull}};
}

static Vec256 op_vmovhp_load(const Vec256& a, word address) {
    return {{a.q[0], load64(address), 0ull, 0ull}};
}

static Vec256 op_vmovlp_load(const Vec256& a, word address) {
    return {{load64(address), a.q[1], 0ull, 0ull}};
}

static Vec256 op_vmovsd_merge(const Vec256& a, const Vec256& b) {
    return {{b.q[0], a.q[1], 0ull, 0ull}};
}

static Vec256 op_vmovss_merge(const Vec256& a, const Vec256& b) {
    return {{(a.q[0] & 0xffffffff00000000ull) | (b.q[0] & 0xffffffffull), a.q[1], 0ull, 0ull}};
}

static Vec256 op_zext32(std::uint32_t v) {
    return {{static_cast<std::uint64_t>(v), 0ull, 0ull, 0ull}};
}

static Vec256 op_zext64(std::uint64_t v) {
    return {{v, 0ull, 0ull, 0ull}};
}

static word op_vmovmskpd(const Vec256& src, bool wide) {
    word mask = 0ull;
    const unsigned n = wide ? 4u : 2u;
    for (unsigned i = 0; i < n; ++i) {
        if (src.q[i] & 0x8000000000000000ull) mask |= (1ull << i);
    }
    return mask;
}

static word op_vmovmskps(const Vec256& src, bool wide) {
    word mask = 0ull;
    const unsigned n = wide ? 8u : 4u;
    for (unsigned i = 0; i < n; ++i) {
        if (lane32(src, i) & 0x80000000u) mask |= (1ull << i);
    }
    return mask;
}

static Vec256 op_vmaskmovps_load(const Vec256& mask, word address, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        if ((lane32(mask, i) & 0x80000000u) != 0u) {
            set_lane32(out, i, load32(address + static_cast<word>(i) * 4ull));
        }
    }
    return out;
}

static Vec256 op_vmaskmovpd_load(const Vec256& mask, word address, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 4u : 2u;
    for (unsigned i = 0; i < lanes; ++i) {
        if ((mask.q[i] & 0x8000000000000000ull) != 0ull) {
            out.q[i] = load64(address + static_cast<word>(i) * 8ull);
        }
    }
    return out;
}

static void op_vmaskmovps_store(word address, const Vec256& mask, const Vec256& src, bool wide) {
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        if ((lane32(mask, i) & 0x80000000u) != 0u) {
            store32(address + static_cast<word>(i) * 4ull, lane32(src, i));
        }
    }
}

static void op_vmaskmovpd_store(word address, const Vec256& mask, const Vec256& src, bool wide) {
    const unsigned lanes = wide ? 4u : 2u;
    for (unsigned i = 0; i < lanes; ++i) {
        if ((mask.q[i] & 0x8000000000000000ull) != 0ull) {
            store64(address + static_cast<word>(i) * 8ull, src.q[i]);
        }
    }
}
#if defined(_MSC_VER)
extern "C" __declspec(dllexport) std::uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}
#endif
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


def gpr_scratch(register, ins):
    """Map volatile scratch GPR32/GPR64 (rax, r9, r10, r11) to (name, is_64), rejecting others."""
    if register in SCRATCH_GPR32:
        return SCRATCH_GPR32[register], False
    if register in SCRATCH_GPR64:
        return SCRATCH_GPR64[register], True
    if register in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported GPR {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, *, store=False):
    """Lower a flat [base + disp] pointer operand rooted in an argument GPR64."""
    base = ins.memory_base
    allowed = {R.R8} if store else ARGUMENTS
    if (ins.is_ip_rel_memory_operand or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
            or ins.memory_index != R.NONE or not RegisterExt.is_gpr64(base)
            or base not in allowed):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    name = ARGUMENTS[base]
    return f"({name} + {disp:#x}ull)" if disp else name


def lift_code(name, start, raw, *, allow_memory=False):
    """Decode and lift one bounded straight-line AVX bit-move/shuffle/blend/logical leaf function."""
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
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if idx not in defined:
            raise ValueError(f"undefined vector register ymm{idx} at RVA {ins.ip:#x}: {ins}")
        used.add(idx)
        return f"ymm{idx}", wide

    def read_vec_or_mem(op_index, ins, mem_loader):
        if ins.op_kind(op_index) == OpKind.MEMORY:
            return f"{mem_loader}({flat_address(ins)})"
        expr, _ = read_vec(op_index, ins)
        return expr

    def write_vec(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        used.add(idx)
        return idx, f"ymm{idx}", wide

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

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        if (ins.op_code().encoding == EncodingKind.EVEX or ins.op_mask != R.NONE
                or ins.zeroing_masking or ins.is_broadcast):
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        mnemonic = ins.mnemonic
        if mnemonic in REGISTER_ONLY_AVX and not allow_memory and any(
                ins.op_kind(i) == OpKind.MEMORY for i in range(ins.op_count)):
            raise ValueError(
                f"unsupported memory operand for register-domain AVX instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            statements.append("return;")
            returned = True
        elif ins.code == Code.VEX_VZEROUPPER:
            for idx in sorted(defined):
                statements.append(f"ymm{idx}.q[2] = 0ull; ymm{idx}.q[3] = 0ull;")
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
        elif mnemonic in LOGICAL_HELPERS:
            src1, _ = read_vec(1, ins)
            dst_idx, dst, wide = write_vec(0, ins)
            if ins.op_kind(2) == OpKind.MEMORY:
                # The memory operand is the last source: dest = op(src1, mem), so
                # VANDN inverts the register operand and never the memory one.
                src2 = f"{'load256' if wide else 'load128'}({flat_address(ins)})"
            else:
                src2, _ = read_vec(2, ins)
            helper = LOGICAL_HELPERS[mnemonic]
            statements.append(f"{dst} = {helper}({src1}, {src2}, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic in (M.VBLENDPD, M.VBLENDPS):
            src1, _ = read_vec(1, ins)
            dst_idx, dst, wide = write_vec(0, ins)
            src2 = read_vec_or_mem(2, ins, "load256" if wide else "load128")
            imm = ins.immediate8 & 0xff
            helper = "op_vblendpd" if mnemonic == M.VBLENDPD else "op_vblendps"
            statements.append(f"{dst} = {helper}({src1}, {src2}, {imm:#x}u, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic in (M.VBLENDVPD, M.VBLENDVPS):
            src1, _ = read_vec(1, ins)
            mask, _ = read_vec(3, ins)
            dst_idx, dst, wide = write_vec(0, ins)
            src2 = read_vec_or_mem(2, ins, "load256" if wide else "load128")
            helper = "op_vblendvpd" if mnemonic == M.VBLENDVPD else "op_vblendvps"
            statements.append(f"{dst} = {helper}({src1}, {src2}, {mask}, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic == M.VBROADCASTF128:
            dst_idx, dst, _ = write_vec(0, ins)
            addr = flat_address(ins)
            statements.append(f"{dst} = op_vbroadcastf128({addr});")
            defined.add(dst_idx)
        elif mnemonic == M.VBROADCASTSD:
            if ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins)
                dst_idx, dst, _ = write_vec(0, ins)
                statements.append(f"{dst} = op_vbroadcastsd_q(load64({addr}));")
            else:
                src, _ = read_vec(1, ins)
                dst_idx, dst, _ = write_vec(0, ins)
                statements.append(f"{dst} = op_vbroadcastsd_q({src}.q[0]);")
            defined.add(dst_idx)
        elif mnemonic == M.VBROADCASTSS:
            if ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins)
                dst_idx, dst, wide = write_vec(0, ins)
                statements.append(f"{dst} = op_vbroadcastss_d(load32({addr}), {'true' if wide else 'false'});")
            else:
                src, _ = read_vec(1, ins)
                dst_idx, dst, wide = write_vec(0, ins)
                statements.append(f"{dst} = op_vbroadcastss_d(lane32({src}, 0u), {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic == M.VEXTRACTF128:
            src, _ = read_vec(1, ins)
            imm = ins.immediate8 & 0xff
            if ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins, store=True)
                statements.append(f"store128({addr}, op_vextractf128({src}, {imm:#x}u));")
            else:
                dst_idx, dst, _ = write_vec(0, ins)
                statements.append(f"{dst} = op_vextractf128({src}, {imm:#x}u);")
                defined.add(dst_idx)
        elif mnemonic == M.VINSERTF128:
            src1, _ = read_vec(1, ins)
            src2 = read_vec_or_mem(2, ins, "load128")
            dst_idx, dst, _ = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = op_vinsertf128({src1}, {src2}, {imm:#x}u);")
            defined.add(dst_idx)
        elif mnemonic == M.VEXTRACTPS:
            src, _ = read_vec(1, ins)
            imm = ins.immediate8 & 0xff
            if ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins, store=True)
                statements.append(f"store32({addr}, static_cast<std::uint32_t>(op_vextractps({src}, {imm:#x}u)));")
            else:
                reg_name, _ = write_gpr(0, ins)
                statements.append(f"{reg_name} = op_vextractps({src}, {imm:#x}u);")
                defined_gpr.add(reg_name)
        elif mnemonic == M.VINSERTPS:
            src1, _ = read_vec(1, ins)
            src2 = read_vec_or_mem(2, ins, "load_vec32")
            dst_idx, dst, _ = write_vec(0, ins)
            imm = (ins.immediate8 & 0x3f) if ins.op2_kind == OpKind.MEMORY else (ins.immediate8 & 0xff)
            statements.append(f"{dst} = op_vinsertps({src1}, {src2}, {imm:#x}u);")
            defined.add(dst_idx)
        elif mnemonic == M.VPERM2F128:
            src1, _ = read_vec(1, ins)
            src2 = read_vec_or_mem(2, ins, "load256")
            dst_idx, dst, _ = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = op_vperm2f128({src1}, {src2}, {imm:#x}u);")
            defined.add(dst_idx)
        elif mnemonic in (M.VPERMILPD, M.VPERMILPS):
            is_pd = mnemonic == M.VPERMILPD
            if ins.op2_kind == OpKind.IMMEDIATE8:
                dst_idx, dst, wide = write_vec(0, ins)
                src = read_vec_or_mem(1, ins, "load256" if wide else "load128")
                imm = ins.immediate8 & 0xff
                helper = "op_vpermilpd_imm" if is_pd else "op_vpermilps_imm"
                statements.append(f"{dst} = {helper}({src}, {imm:#x}u, {'true' if wide else 'false'});")
            else:
                src, _ = read_vec(1, ins)
                dst_idx, dst, wide = write_vec(0, ins)
                ctrl = read_vec_or_mem(2, ins, "load256" if wide else "load128")
                helper = "op_vpermilpd_var" if is_pd else "op_vpermilps_var"
                statements.append(f"{dst} = {helper}({src}, {ctrl}, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic in (M.VSHUFPD, M.VSHUFPS):
            src1, _ = read_vec(1, ins)
            dst_idx, dst, wide = write_vec(0, ins)
            src2 = read_vec_or_mem(2, ins, "load256" if wide else "load128")
            imm = ins.immediate8 & 0xff
            helper = "op_vshufpd" if mnemonic == M.VSHUFPD else "op_vshufps"
            statements.append(f"{dst} = {helper}({src1}, {src2}, {imm:#x}u, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic in UNPACK_HELPERS:
            src1, _ = read_vec(1, ins)
            dst_idx, dst, wide = write_vec(0, ins)
            src2 = read_vec_or_mem(2, ins, "load256" if wide else "load128")
            helper = UNPACK_HELPERS[mnemonic]
            statements.append(f"{dst} = {helper}({src1}, {src2}, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic in DUP_HELPERS:
            dst_idx, dst, wide = write_vec(0, ins)
            mem_loader = ("load256" if wide else "load_vec64") if mnemonic == M.VMOVDDUP else (
                "load256" if wide else "load128")
            src = read_vec_or_mem(1, ins, mem_loader)
            helper = DUP_HELPERS[mnemonic]
            statements.append(f"{dst} = {helper}({src}, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic in FULL_VEC_MOVES:
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst_idx, dst, wide = write_vec(0, ins)
                addr = flat_address(ins)
                statements.append(f"{dst} = {'load256' if wide else 'load128'}({addr});")
                defined.add(dst_idx)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src, wide = read_vec(1, ins)
                addr = flat_address(ins)
                statements.append(f"{'store256' if wide else 'store128'}({addr}, {src});")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src, _ = read_vec(1, ins)
                dst_idx, dst, wide = write_vec(0, ins)
                statements.append(f"{dst} = {src};" if wide else f"{dst} = copy128({src});")
                defined.add(dst_idx)
            else:
                raise ValueError(f"unsupported vector move operands at RVA {ins.ip:#x}: {ins}")
        elif mnemonic in PLAIN_LOADS:
            dst_idx, dst, wide = write_vec(0, ins)
            addr = flat_address(ins)
            statements.append(f"{dst} = {'load256' if wide else 'load128'}({addr});")
            defined.add(dst_idx)
        elif mnemonic in NT_STORES:
            src, wide = read_vec(1, ins)
            addr = flat_address(ins)
            statements.append(f"{'store256' if wide else 'store128'}({addr}, {src});")
        elif mnemonic in (M.VMOVHLPS, M.VMOVLHPS):
            src1, _ = read_vec(1, ins)
            src2, _ = read_vec(2, ins)
            dst_idx, dst, _ = write_vec(0, ins)
            helper = "op_vmovhlps" if mnemonic == M.VMOVHLPS else "op_vmovlhps"
            statements.append(f"{dst} = {helper}({src1}, {src2});")
            defined.add(dst_idx)
        elif mnemonic in (M.VMOVHPD, M.VMOVHPS, M.VMOVLPD, M.VMOVLPS):
            is_high = mnemonic in (M.VMOVHPD, M.VMOVHPS)
            if ins.op0_kind == OpKind.MEMORY:
                src, _ = read_vec(1, ins)
                addr = flat_address(ins)
                q_idx = 1 if is_high else 0
                statements.append(f"store64({addr}, {src}.q[{q_idx}]);")
            else:
                src1, _ = read_vec(1, ins)
                dst_idx, dst, _ = write_vec(0, ins)
                addr = flat_address(ins)
                helper = "op_vmovhp_load" if is_high else "op_vmovlp_load"
                statements.append(f"{dst} = {helper}({src1}, {addr});")
                defined.add(dst_idx)
        elif mnemonic in (M.VMOVSD, M.VMOVSS):
            is_sd = mnemonic == M.VMOVSD
            if ins.op_count == 3:
                src1, _ = read_vec(1, ins)
                src2, _ = read_vec(2, ins)
                dst_idx, dst, _ = write_vec(0, ins)
                helper = "op_vmovsd_merge" if is_sd else "op_vmovss_merge"
                statements.append(f"{dst} = {helper}({src1}, {src2});")
                defined.add(dst_idx)
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst_idx, dst, _ = write_vec(0, ins)
                addr = flat_address(ins)
                if is_sd:
                    statements.append(f"{dst} = op_zext64(load64({addr}));")
                else:
                    statements.append(f"{dst} = op_zext32(load32({addr}));")
                defined.add(dst_idx)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src, _ = read_vec(1, ins)
                addr = flat_address(ins)
                if is_sd:
                    statements.append(f"store64({addr}, {src}.q[0]);")
                else:
                    statements.append(f"store32({addr}, lane32({src}, 0u));")
            else:
                raise ValueError(f"unsupported scalar move operands at RVA {ins.ip:#x}: {ins}")
        elif mnemonic in (M.VMOVMSKPD, M.VMOVMSKPS):
            src, wide = read_vec(1, ins)
            reg_name, _ = write_gpr(0, ins)
            helper = "op_vmovmskpd" if mnemonic == M.VMOVMSKPD else "op_vmovmskps"
            statements.append(f"{reg_name} = {helper}({src}, {'true' if wide else 'false'});")
            defined_gpr.add(reg_name)
        elif ins.code in MASKMOV_SPECS:
            is_store, elem_bits, wide = MASKMOV_SPECS[ins.code]
            if is_store:
                if ins.op0_kind != OpKind.MEMORY or ins.op1_kind != OpKind.REGISTER or ins.op2_kind != OpKind.REGISTER:
                    raise ValueError(f"unsupported vmaskmov store operands at RVA {ins.ip:#x}: {ins}")
                mask_name, mask_wide = read_vec(1, ins)
                src_name, src_wide = read_vec(2, ins)
                if mask_wide != wide or src_wide != wide:
                    raise ValueError(f"vmaskmov width mismatch at RVA {ins.ip:#x}: {ins}")
                addr = flat_address(ins)
                helper = "op_vmaskmovps_store" if elem_bits == 32 else "op_vmaskmovpd_store"
                statements.append(f"{helper}({addr}, {mask_name}, {src_name}, {'true' if wide else 'false'});")
            else:
                if ins.op0_kind != OpKind.REGISTER or ins.op1_kind != OpKind.REGISTER or ins.op2_kind != OpKind.MEMORY:
                    raise ValueError(f"unsupported vmaskmov load operands at RVA {ins.ip:#x}: {ins}")
                mask_name, mask_wide = read_vec(1, ins)
                dst_idx, dst_name, dst_wide = write_vec(0, ins)
                if mask_wide != wide or dst_wide != wide:
                    raise ValueError(f"vmaskmov width mismatch at RVA {ins.ip:#x}: {ins}")
                addr = flat_address(ins)
                helper = "op_vmaskmovps_load" if elem_bits == 32 else "op_vmaskmovpd_load"
                statements.append(f"{dst_name} = {helper}({mask_name}, {addr}, {'true' if wide else 'false'});")
                defined.add(dst_idx)
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec256 ymm{idx}{{}};" for idx in sorted(used)]
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


def lift(pe, name, *, allow_memory=False):
    entry = export_rva(pe, name)
    raw = bounded_code(pe, entry)
    source, count = lift_code(name, entry, raw, allow_memory=allow_memory)
    return source, entry, count


def translate_pe(pe, names, *, allow_memory=False):
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    lifted = [lift(pe, name, allow_memory=allow_memory) for name in names]
    source = PRELUDE + "\n".join(f"\n{text}" for text, _, _ in lifted) + "\n"
    meta = [(name, entry, count) for name, (_, entry, count) in zip(names, lifted)]
    return source, meta


def translate(image, names, *, allow_memory=False):
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, names, allow_memory=allow_memory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated export names")
    parser.add_argument("output", type=Path)
    parser.add_argument("--mem-ops", action="store_true", help="allow memory operands on register-domain AVX forms")
    args = parser.parse_args()
    try:
        source, meta = translate(args.image, args.exports.split(","), allow_memory=args.mem_ops)
        args.output.write_text(source, encoding="utf-8")
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x}: {count} instructions")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
