"""Lift bounded Windows x64 leaf functions with AVX2 instructions to C++.

Each export uses the leaf ABI void fn(word rcx, word rdx, word r8), where rcx,
rdx and r8 carry flat host pointers to caller-owned buffers. YMM0..YMM5 are
modeled as portable 256-bit value structs (four 64-bit lanes); every VEX write
to an XMM destination zeroes bits 128..255 of the corresponding YMM register.

Supported AVX2 instructions (all 28 forms from docs/coverage/x64_zen2.tsv group AVX2):
  - 128-bit lane broadcast/permute: vbroadcasti128 (m128), vextracti128,
    vinserti128, vperm2i128
  - Element broadcasts: vpbroadcastb, vpbroadcastw, vpbroadcastd, vpbroadcastq
  - Lane permutes: vpermd, vpermpd, vpermps, vpermq
  - Blend: vpblendd
  - Variable shifts: vpsllvd, vpsllvq, vpsravd, vpsrlvd, vpsrlvq
  - Indexed gathers (VSIB): vgatherdps, vgatherdpd, vgatherqps, vgatherqpd,
    vpgatherdd, vpgatherdq, vpgatherqd, vpgatherqq
  - Masked loads and stores: vpmaskmovd, vpmaskmovq

Supported FMA3 instructions (all 60 forms from docs/coverage/x64_zen2.tsv group FMA):
  - Scalar single/double (XMM, reg or m32/m64 op2): vfmadd/vfmsub/vfnmadd/vfnmsub
    x {132, 213, 231} x {ss, sd}
  - Packed single/double (XMM/YMM, reg or m128/m256 op2): vfmadd/vfmsub/vfnmadd/
    vfnmsub/vfmaddsub/vfmsubadd x {132, 213, 231} x {ps, pd}

Transport/ABI glue: vmovdqu (128/256-bit flat load, store and register copy),
vzeroupper and terminal ret.

Memory operands on the 17 register-only AVX2 forms, overlapping gather
dest/index/mask registers, invalid VSIB scales or addressing modes, nonvolatile
vector registers (XMM6..XMM15 / YMM6..YMM15), EVEX encodings, instruction
prefixes, branches and all other instructions are rejected.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import Code, Decoder, DecoderOptions, EncodingKind, Mnemonic as M, OpKind, Register as R, RegisterExt

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}
VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
VOLATILE_YMM = {getattr(R, f"YMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = ({getattr(R, f"XMM{i}") for i in range(6, 16)}
                   | {getattr(R, f"YMM{i}") for i in range(6, 16)})

REGISTER_ONLY_AVX2 = {
    M.VEXTRACTI128, M.VINSERTI128, M.VPERM2I128,
    M.VPBROADCASTB, M.VPBROADCASTW, M.VPBROADCASTD, M.VPBROADCASTQ,
    M.VPERMD, M.VPERMPD, M.VPERMPS, M.VPERMQ,
    M.VPBLENDD,
    M.VPSLLVD, M.VPSLLVQ, M.VPSRAVD, M.VPSRLVD, M.VPSRLVQ,
}

# (helper_name, element_count, dest_and_mask_is_ymm, index_is_ymm, index_elem_bits)
GATHER_SPECS = {
    Code.VEX_VGATHERDPS_XMM_VM32X_XMM: ("op_gather_d32_i32", 4, False, False, 32),
    Code.VEX_VGATHERDPS_YMM_VM32Y_YMM: ("op_gather_d32_i32", 8, True, True, 32),
    Code.VEX_VPGATHERDD_XMM_VM32X_XMM: ("op_gather_d32_i32", 4, False, False, 32),
    Code.VEX_VPGATHERDD_YMM_VM32Y_YMM: ("op_gather_d32_i32", 8, True, True, 32),
    Code.VEX_VGATHERDPD_XMM_VM32X_XMM: ("op_gather_d64_i32", 2, False, False, 32),
    Code.VEX_VGATHERDPD_YMM_VM32X_YMM: ("op_gather_d64_i32", 4, True, False, 32),
    Code.VEX_VPGATHERDQ_XMM_VM32X_XMM: ("op_gather_d64_i32", 2, False, False, 32),
    Code.VEX_VPGATHERDQ_YMM_VM32X_YMM: ("op_gather_d64_i32", 4, True, False, 32),
    Code.VEX_VGATHERQPS_XMM_VM64X_XMM: ("op_gather_d32_i64", 2, False, False, 64),
    Code.VEX_VGATHERQPS_XMM_VM64Y_XMM: ("op_gather_d32_i64", 4, False, True, 64),
    Code.VEX_VPGATHERQD_XMM_VM64X_XMM: ("op_gather_d32_i64", 2, False, False, 64),
    Code.VEX_VPGATHERQD_XMM_VM64Y_XMM: ("op_gather_d32_i64", 4, False, True, 64),
    Code.VEX_VGATHERQPD_XMM_VM64X_XMM: ("op_gather_d64_i64", 2, False, False, 64),
    Code.VEX_VGATHERQPD_YMM_VM64Y_YMM: ("op_gather_d64_i64", 4, True, True, 64),
    Code.VEX_VPGATHERQQ_XMM_VM64X_XMM: ("op_gather_d64_i64", 2, False, False, 64),
    Code.VEX_VPGATHERQQ_YMM_VM64Y_YMM: ("op_gather_d64_i64", 4, True, True, 64),
}

# (is_store, element_bits, is_ymm)
MASKMOV_SPECS = {
    Code.VEX_VPMASKMOVD_XMM_XMM_M128: (False, 32, False),
    Code.VEX_VPMASKMOVD_YMM_YMM_M256: (False, 32, True),
    Code.VEX_VPMASKMOVQ_XMM_XMM_M128: (False, 64, False),
    Code.VEX_VPMASKMOVQ_YMM_YMM_M256: (False, 64, True),
    Code.VEX_VPMASKMOVD_M128_XMM_XMM: (True, 32, False),
    Code.VEX_VPMASKMOVD_M256_YMM_YMM: (True, 32, True),
    Code.VEX_VPMASKMOVQ_M128_XMM_XMM: (True, 64, False),
    Code.VEX_VPMASKMOVQ_M256_YMM_YMM: (True, 64, True),
}

# (kind: "ss"|"sd"|"ps"|"pd", order: 132|213|231, mode: 0..5, is_ymm: bool)
# mode: 0=madd, 1=msub, 2=nmadd, 3=nmsub, 4=maddsub, 5=msubadd
def _build_fma3_specs():
    specs = {}
    families = (
        ("VFMADD", 0, True),
        ("VFMSUB", 1, True),
        ("VFNMADD", 2, True),
        ("VFNMSUB", 3, True),
        ("VFMADDSUB", 4, False),
        ("VFMSUBADD", 5, False),
    )
    for fam, mode, has_scalar in families:
        for order in (132, 213, 231):
            specs[getattr(Code, f"VEX_{fam}{order}PS_XMM_XMM_XMMM128")] = ("ps", order, mode, False)
            specs[getattr(Code, f"VEX_{fam}{order}PS_YMM_YMM_YMMM256")] = ("ps", order, mode, True)
            specs[getattr(Code, f"VEX_{fam}{order}PD_XMM_XMM_XMMM128")] = ("pd", order, mode, False)
            specs[getattr(Code, f"VEX_{fam}{order}PD_YMM_YMM_YMMM256")] = ("pd", order, mode, True)
            if has_scalar:
                specs[getattr(Code, f"VEX_{fam}{order}SS_XMM_XMM_XMMM32")] = ("ss", order, mode, False)
                specs[getattr(Code, f"VEX_{fam}{order}SD_XMM_XMM_XMMM64")] = ("sd", order, mode, False)
    return specs


FMA3_SPECS = _build_fma3_specs()

ALLOWED_CODES = {
    Code.VEX_VMOVDQU_XMM_XMMM128,
    Code.VEX_VMOVDQU_XMMM128_XMM,
    Code.VEX_VMOVDQU_YMM_YMMM256,
    Code.VEX_VMOVDQU_YMMM256_YMM,
    Code.VEX_VZEROUPPER,
    Code.RETNQ,
    Code.VEX_VBROADCASTI128_YMM_M128,
    Code.VEX_VEXTRACTI128_XMMM128_YMM_IMM8,
    Code.VEX_VINSERTI128_YMM_YMM_XMMM128_IMM8,
    Code.VEX_VPERM2I128_YMM_YMM_YMMM256_IMM8,
    Code.VEX_VPBROADCASTB_XMM_XMMM8,
    Code.VEX_VPBROADCASTB_YMM_XMMM8,
    Code.VEX_VPBROADCASTW_XMM_XMMM16,
    Code.VEX_VPBROADCASTW_YMM_XMMM16,
    Code.VEX_VPBROADCASTD_XMM_XMMM32,
    Code.VEX_VPBROADCASTD_YMM_XMMM32,
    Code.VEX_VPBROADCASTQ_XMM_XMMM64,
    Code.VEX_VPBROADCASTQ_YMM_XMMM64,
    Code.VEX_VPERMD_YMM_YMM_YMMM256,
    Code.VEX_VPERMPD_YMM_YMMM256_IMM8,
    Code.VEX_VPERMPS_YMM_YMM_YMMM256,
    Code.VEX_VPERMQ_YMM_YMMM256_IMM8,
    Code.VEX_VPBLENDD_XMM_XMM_XMMM128_IMM8,
    Code.VEX_VPBLENDD_YMM_YMM_YMMM256_IMM8,
    Code.VEX_VPSLLVD_XMM_XMM_XMMM128,
    Code.VEX_VPSLLVD_YMM_YMM_YMMM256,
    Code.VEX_VPSLLVQ_XMM_XMM_XMMM128,
    Code.VEX_VPSLLVQ_YMM_YMM_YMMM256,
    Code.VEX_VPSRAVD_XMM_XMM_XMMM128,
    Code.VEX_VPSRAVD_YMM_YMM_YMMM256,
    Code.VEX_VPSRLVD_XMM_XMM_XMMM128,
    Code.VEX_VPSRLVD_YMM_YMM_YMMM256,
    Code.VEX_VPSRLVQ_XMM_XMM_XMMM128,
    Code.VEX_VPSRLVQ_YMM_YMM_YMMM256,
    *GATHER_SPECS,
    *MASKMOV_SPECS,
    *FMA3_SPECS,
}

BROADCAST_HELPERS = {
    M.VPBROADCASTB: ("op_vpbroadcastb", "load_vec8"),
    M.VPBROADCASTW: ("op_vpbroadcastw", "load_vec16"),
    M.VPBROADCASTD: ("op_vpbroadcastd", "load_vec32"),
    M.VPBROADCASTQ: ("op_vpbroadcastq", "load_vec64"),
}

SHIFT_HELPERS = {
    M.VPSLLVD: "op_vpsllvd",
    M.VPSLLVQ: "op_vpsllvq",
    M.VPSRAVD: "op_vpsravd",
    M.VPSRLVD: "op_vpsrlvd",
    M.VPSRLVQ: "op_vpsrlvq",
}

PRELUDE = """#include <atomic>
#include <cmath>
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

static Vec256 op_vbroadcasti128(word address) {
    Vec256 v = load128(address);
    v.q[2] = v.q[0];
    v.q[3] = v.q[1];
    return v;
}

static Vec256 op_vextracti128(const Vec256& src, std::uint8_t imm8) {
    if (imm8 & 1u) return {{src.q[2], src.q[3], 0ull, 0ull}};
    return {{src.q[0], src.q[1], 0ull, 0ull}};
}

static Vec256 op_vinserti128(const Vec256& src1, const Vec256& src2, std::uint8_t imm8) {
    if (imm8 & 1u) return {{src1.q[0], src1.q[1], src2.q[0], src2.q[1]}};
    return {{src2.q[0], src2.q[1], src1.q[2], src1.q[3]}};
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

static Vec256 op_vperm2i128(const Vec256& src1, const Vec256& src2, std::uint8_t imm8) {
    Vec256 out{};
    pick_lane128(src1, src2, imm8 & 0xfu, out.q[0], out.q[1]);
    pick_lane128(src1, src2, (imm8 >> 4) & 0xfu, out.q[2], out.q[3]);
    return out;
}

static Vec256 op_vpbroadcastb(const Vec256& src, bool wide) {
    const std::uint64_t b = src.q[0] & 0xffull;
    const std::uint64_t q = b * 0x0101010101010101ull;
    return wide ? Vec256{{q, q, q, q}} : Vec256{{q, q, 0ull, 0ull}};
}

static Vec256 op_vpbroadcastw(const Vec256& src, bool wide) {
    const std::uint64_t w = src.q[0] & 0xffffull;
    const std::uint64_t q = w * 0x0001000100010001ull;
    return wide ? Vec256{{q, q, q, q}} : Vec256{{q, q, 0ull, 0ull}};
}

static Vec256 op_vpbroadcastd(const Vec256& src, bool wide) {
    const std::uint64_t d = src.q[0] & 0xffffffffull;
    const std::uint64_t q = d | (d << 32);
    return wide ? Vec256{{q, q, q, q}} : Vec256{{q, q, 0ull, 0ull}};
}

static Vec256 op_vpbroadcastq(const Vec256& src, bool wide) {
    const std::uint64_t q = src.q[0];
    return wide ? Vec256{{q, q, q, q}} : Vec256{{q, q, 0ull, 0ull}};
}

static Vec256 op_vpermd(const Vec256& idx, const Vec256& src) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        set_lane32(out, i, lane32(src, lane32(idx, i) & 7u));
    }
    return out;
}

static Vec256 op_vpermq(const Vec256& src, std::uint8_t imm8) {
    return {{
        src.q[(imm8 >> 0) & 3u],
        src.q[(imm8 >> 2) & 3u],
        src.q[(imm8 >> 4) & 3u],
        src.q[(imm8 >> 6) & 3u],
    }};
}

static Vec256 op_vpblendd(const Vec256& src1, const Vec256& src2, std::uint8_t imm8, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        set_lane32(out, i, ((imm8 >> i) & 1u) ? lane32(src2, i) : lane32(src1, i));
    }
    return out;
}

static Vec256 op_vpsllvd(const Vec256& src1, const Vec256& src2, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        const std::uint32_t c = lane32(src2, i);
        set_lane32(out, i, c >= 32u ? 0u : (lane32(src1, i) << c));
    }
    return out;
}

static Vec256 op_vpsrlvd(const Vec256& src1, const Vec256& src2, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        const std::uint32_t c = lane32(src2, i);
        set_lane32(out, i, c >= 32u ? 0u : (lane32(src1, i) >> c));
    }
    return out;
}

static Vec256 op_vpsravd(const Vec256& src1, const Vec256& src2, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        const std::uint32_t v = lane32(src1, i);
        const std::uint32_t c = lane32(src2, i);
        const bool neg = (v & 0x80000000u) != 0u;
        std::uint32_t r;
        if (c >= 32u) {
            r = neg ? 0xffffffffu : 0u;
        } else {
            r = (v >> c) | ((neg && c != 0u) ? (~0u << (32u - c)) : 0u);
        }
        set_lane32(out, i, r);
    }
    return out;
}

static Vec256 op_vpsllvq(const Vec256& src1, const Vec256& src2, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 4u : 2u;
    for (unsigned i = 0; i < lanes; ++i) {
        const std::uint64_t c = src2.q[i];
        out.q[i] = c >= 64ull ? 0ull : (src1.q[i] << c);
    }
    return out;
}

static Vec256 op_vpsrlvq(const Vec256& src1, const Vec256& src2, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 4u : 2u;
    for (unsigned i = 0; i < lanes; ++i) {
        const std::uint64_t c = src2.q[i];
        out.q[i] = c >= 64ull ? 0ull : (src1.q[i] >> c);
    }
    return out;
}

static inline std::uint8_t load8(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto val = *reinterpret_cast<const volatile std::uint8_t*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return val;
}

static inline std::uint16_t load16(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto val = *reinterpret_cast<const volatile std::uint16_t X2W_UNALIGNED*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return val;
}

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

static void store32(word address, std::uint32_t val) {
    std::memcpy(reinterpret_cast<void*>(address), &val, 4);
}

static void store64(word address, std::uint64_t val) {
    std::memcpy(reinterpret_cast<void*>(address), &val, 8);
}

static word sign_extend32(std::uint32_t v) {
    return static_cast<word>(static_cast<std::int64_t>(static_cast<std::int32_t>(v)));
}

static void op_gather_d32_i32(Vec256& dest, word base, word disp, word scale,
                              const Vec256& vindex, Vec256& mask, unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
        if ((lane32(mask, i) & 0x80000000u) != 0u) {
            const word addr = base + sign_extend32(lane32(vindex, i)) * scale + disp;
            set_lane32(dest, i, load32(addr));
        }
    }
    if (count == 4u) {
        dest.q[2] = 0ull;
        dest.q[3] = 0ull;
    }
    mask = Vec256{};
}

static void op_gather_d64_i32(Vec256& dest, word base, word disp, word scale,
                              const Vec256& vindex, Vec256& mask, unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
        if ((mask.q[i] & 0x8000000000000000ull) != 0ull) {
            const word addr = base + sign_extend32(lane32(vindex, i)) * scale + disp;
            dest.q[i] = load64(addr);
        }
    }
    if (count == 2u) {
        dest.q[2] = 0ull;
        dest.q[3] = 0ull;
    }
    mask = Vec256{};
}

static void op_gather_d32_i64(Vec256& dest, word base, word disp, word scale,
                              const Vec256& vindex, Vec256& mask, unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
        if ((lane32(mask, i) & 0x80000000u) != 0u) {
            const word addr = base + vindex.q[i] * scale + disp;
            set_lane32(dest, i, load32(addr));
        }
    }
    if (count == 2u) {
        dest.q[1] = 0ull;
    }
    dest.q[2] = 0ull;
    dest.q[3] = 0ull;
    mask = Vec256{};
}

static void op_gather_d64_i64(Vec256& dest, word base, word disp, word scale,
                              const Vec256& vindex, Vec256& mask, unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
        if ((mask.q[i] & 0x8000000000000000ull) != 0ull) {
            const word addr = base + vindex.q[i] * scale + disp;
            dest.q[i] = load64(addr);
        }
    }
    if (count == 2u) {
        dest.q[2] = 0ull;
        dest.q[3] = 0ull;
    }
    mask = Vec256{};
}

static Vec256 op_vpmaskmovd_load(const Vec256& mask, word address, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        if ((lane32(mask, i) & 0x80000000u) != 0u) {
            set_lane32(out, i, load32(address + static_cast<word>(i) * 4ull));
        }
    }
    return out;
}

static Vec256 op_vpmaskmovq_load(const Vec256& mask, word address, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 4u : 2u;
    for (unsigned i = 0; i < lanes; ++i) {
        if ((mask.q[i] & 0x8000000000000000ull) != 0ull) {
            out.q[i] = load64(address + static_cast<word>(i) * 8ull);
        }
    }
    return out;
}

static void op_vpmaskmovd_store(word address, const Vec256& mask, const Vec256& src, bool wide) {
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        if ((lane32(mask, i) & 0x80000000u) != 0u) {
            store32(address + static_cast<word>(i) * 4ull, lane32(src, i));
        }
    }
}

static void op_vpmaskmovq_store(word address, const Vec256& mask, const Vec256& src, bool wide) {
    const unsigned lanes = wide ? 4u : 2u;
    for (unsigned i = 0; i < lanes; ++i) {
        if ((mask.q[i] & 0x8000000000000000ull) != 0ull) {
            store64(address + static_cast<word>(i) * 8ull, src.q[i]);
        }
    }
}

static inline Vec256 load_vec8(word address) {
    Vec256 v{};
    v.q[0] = static_cast<std::uint64_t>(load8(address));
    return v;
}

static inline Vec256 load_vec16(word address) {
    Vec256 v{};
    v.q[0] = static_cast<std::uint64_t>(load16(address));
    return v;
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

static float f32_from_bits(std::uint32_t bits) {
    float f = 0.0f;
    std::memcpy(&f, &bits, 4);
    return f;
}

static std::uint32_t f32_to_bits(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, 4);
    return bits;
}

static double f64_from_bits(std::uint64_t bits) {
    double d = 0.0;
    std::memcpy(&d, &bits, 8);
    return d;
}

static std::uint64_t f64_to_bits(double d) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &d, 8);
    return bits;
}

static std::uint32_t fma_lane_f32(std::uint32_t op0_b, std::uint32_t op1_b, std::uint32_t op2_b,
                                  unsigned order, bool neg_prod, bool neg_add) {
    const std::uint32_t a_b = (order == 132u) ? op0_b : op1_b;
    const std::uint32_t b_b = (order == 213u) ? op0_b : op2_b;
    const std::uint32_t c_b = (order == 132u) ? op1_b : ((order == 213u) ? op2_b : op0_b);
    const std::uint32_t ops[3] = {a_b, b_b, c_b};
    for (std::uint32_t b : ops) {
        if ((b & 0x7fffffffu) > 0x7f800000u) return b | 0x00400000u;
    }
    const float a = f32_from_bits(neg_prod ? (a_b ^ 0x80000000u) : a_b);
    const float b = f32_from_bits(b_b);
    const float c = f32_from_bits(neg_add ? (c_b ^ 0x80000000u) : c_b);
    const float r = std::fmaf(a, b, c);
    if (std::isnan(r)) return 0xffc00000u;
    return f32_to_bits(r);
}

static std::uint64_t fma_lane_f64(std::uint64_t op0_b, std::uint64_t op1_b, std::uint64_t op2_b,
                                  unsigned order, bool neg_prod, bool neg_add) {
    const std::uint64_t a_b = (order == 132u) ? op0_b : op1_b;
    const std::uint64_t b_b = (order == 213u) ? op0_b : op2_b;
    const std::uint64_t c_b = (order == 132u) ? op1_b : ((order == 213u) ? op2_b : op0_b);
    const std::uint64_t ops[3] = {a_b, b_b, c_b};
    for (std::uint64_t b : ops) {
        if ((b & 0x7fffffffffffffffull) > 0x7ff0000000000000ull) return b | 0x0008000000000000ull;
    }
    const double a = f64_from_bits(neg_prod ? (a_b ^ 0x8000000000000000ull) : a_b);
    const double b = f64_from_bits(b_b);
    const double c = f64_from_bits(neg_add ? (c_b ^ 0x8000000000000000ull) : c_b);
    const double r = std::fma(a, b, c);
    if (std::isnan(r)) return 0xfff8000000000000ull;
    return f64_to_bits(r);
}

static void fma_lane_signs(unsigned mode, unsigned lane, bool& neg_prod, bool& neg_add) {
    neg_prod = (mode == 2u || mode == 3u);
    if (mode == 4u) {
        neg_add = ((lane & 1u) == 0u);
    } else if (mode == 5u) {
        neg_add = ((lane & 1u) != 0u);
    } else {
        neg_add = (mode == 1u || mode == 3u);
    }
}

static Vec256 op_fma_ss(const Vec256& op0, const Vec256& op1, const Vec256& op2,
                        unsigned order, unsigned mode) {
    Vec256 out = copy128(op0);
    bool neg_prod = false, neg_add = false;
    fma_lane_signs(mode, 0u, neg_prod, neg_add);
    set_lane32(out, 0u, fma_lane_f32(lane32(op0, 0u), lane32(op1, 0u), lane32(op2, 0u), order, neg_prod, neg_add));
    return out;
}

static Vec256 op_fma_sd(const Vec256& op0, const Vec256& op1, const Vec256& op2,
                        unsigned order, unsigned mode) {
    Vec256 out = copy128(op0);
    bool neg_prod = false, neg_add = false;
    fma_lane_signs(mode, 0u, neg_prod, neg_add);
    out.q[0] = fma_lane_f64(op0.q[0], op1.q[0], op2.q[0], order, neg_prod, neg_add);
    return out;
}

static Vec256 op_fma_ps(const Vec256& op0, const Vec256& op1, const Vec256& op2,
                        unsigned order, unsigned mode, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 8u : 4u;
    for (unsigned i = 0; i < lanes; ++i) {
        bool neg_prod = false, neg_add = false;
        fma_lane_signs(mode, i, neg_prod, neg_add);
        set_lane32(out, i, fma_lane_f32(lane32(op0, i), lane32(op1, i), lane32(op2, i), order, neg_prod, neg_add));
    }
    return out;
}

static Vec256 op_fma_pd(const Vec256& op0, const Vec256& op1, const Vec256& op2,
                        unsigned order, unsigned mode, bool wide) {
    Vec256 out{};
    const unsigned lanes = wide ? 4u : 2u;
    for (unsigned i = 0; i < lanes; ++i) {
        bool neg_prod = false, neg_add = false;
        fma_lane_signs(mode, i, neg_prod, neg_add);
        out.q[i] = fma_lane_f64(op0.q[i], op1.q[i], op2.q[i], order, neg_prod, neg_add);
    }
    return out;
}
#if defined(_MSC_VER)
extern "C" __declspec(dllexport) std::uint32_t host_has_avx2() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    if ((_xgetbv(0) & 6u) != 6u) return 0u;
    __cpuid(info, 0);
    if (info[0] < 7) return 0u;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) ? 1u : 0u;
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


def vsib_address(ins, expected_idx_wide, expected_idx_bits):
    """Validate and extract (base_name, disp_u64, scale, idx_reg_num) from a VSIB memory operand."""
    if not ins.is_vsib:
        raise ValueError(f"expected VSIB memory operand at RVA {ins.ip:#x}: {ins}")
    if (expected_idx_bits == 32 and not ins.is_vsib32) or (expected_idx_bits == 64 and not ins.is_vsib64):
        raise ValueError(f"VSIB index width mismatch at RVA {ins.ip:#x}: {ins}")
    base = ins.memory_base
    if (ins.is_ip_rel_memory_operand or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
            or not RegisterExt.is_gpr64(base) or base not in ARGUMENTS):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    scale = ins.memory_index_scale
    if scale not in (1, 2, 4, 8):
        raise ValueError(f"unsupported VSIB scale {scale} at RVA {ins.ip:#x}: {ins}")
    idx_num, idx_wide = vec_index(ins.memory_index, ins)
    if idx_wide != expected_idx_wide:
        raise ValueError(f"VSIB index register width mismatch at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    return ARGUMENTS[base], disp, scale, idx_num


def lift_code(name, start, raw, *, allow_memory=False):
    """Decode and lift one bounded straight-line AVX2 leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        probe = next(iter(Decoder(64, raw[bad.ip - start:], DecoderOptions.NO_INVALID_CHECK, ip=bad.ip)), None)
        if probe is not None and not probe.is_invalid and probe.code in GATHER_SPECS:
            dst_idx, _ = vec_index(probe.op_register(0), probe)
            idx_idx, _ = vec_index(probe.memory_index, probe)
            mask_idx, _ = vec_index(probe.op_register(2), probe)
            if len({dst_idx, idx_idx, mask_idx}) != 3:
                raise ValueError(
                    f"overlapping gather registers (dest=ymm{dst_idx}, index=ymm{idx_idx}, mask=ymm{mask_idx}) "
                    f"at RVA {bad.ip:#x}: {probe}")
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used = set()
    defined = set()
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

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        if (ins.op_code().encoding == EncodingKind.EVEX or ins.op_mask != R.NONE
                or ins.zeroing_masking or ins.is_broadcast):
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        mnemonic = ins.mnemonic
        if mnemonic in REGISTER_ONLY_AVX2 and not allow_memory and any(
                ins.op_kind(i) == OpKind.MEMORY for i in range(ins.op_count)):
            raise ValueError(
                f"unsupported memory operand for register-domain AVX2 instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            statements.append("return;")
            returned = True
        elif ins.code == Code.VEX_VZEROUPPER:
            for idx in sorted(defined):
                statements.append(f"ymm{idx}.q[2] = 0ull; ymm{idx}.q[3] = 0ull;")
        elif mnemonic == M.VMOVDQU:
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
                raise ValueError(f"unsupported vmovdqu operands at RVA {ins.ip:#x}: {ins}")
        elif mnemonic == M.VBROADCASTI128:
            dst_idx, dst, _ = write_vec(0, ins)
            addr = flat_address(ins)
            statements.append(f"{dst} = op_vbroadcasti128({addr});")
            defined.add(dst_idx)
        elif mnemonic == M.VEXTRACTI128:
            src, _ = read_vec(1, ins)
            imm = ins.immediate8 & 0xff
            if ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins, store=True)
                statements.append(f"store128({addr}, op_vextracti128({src}, {imm:#x}u));")
            else:
                dst_idx, dst, _ = write_vec(0, ins)
                statements.append(f"{dst} = op_vextracti128({src}, {imm:#x}u);")
                defined.add(dst_idx)
        elif mnemonic == M.VINSERTI128:
            src1, _ = read_vec(1, ins)
            src2 = read_vec_or_mem(2, ins, "load128")
            dst_idx, dst, _ = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = op_vinserti128({src1}, {src2}, {imm:#x}u);")
            defined.add(dst_idx)
        elif mnemonic == M.VPERM2I128:
            src1, _ = read_vec(1, ins)
            src2 = read_vec_or_mem(2, ins, "load256")
            dst_idx, dst, _ = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = op_vperm2i128({src1}, {src2}, {imm:#x}u);")
            defined.add(dst_idx)
        elif mnemonic in BROADCAST_HELPERS:
            helper, mem_loader = BROADCAST_HELPERS[mnemonic]
            src = read_vec_or_mem(1, ins, mem_loader)
            dst_idx, dst, wide = write_vec(0, ins)
            statements.append(f"{dst} = {helper}({src}, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic in (M.VPERMD, M.VPERMPS):
            src1, _ = read_vec(1, ins)
            src2 = read_vec_or_mem(2, ins, "load256")
            dst_idx, dst, _ = write_vec(0, ins)
            statements.append(f"{dst} = op_vpermd({src1}, {src2});")
            defined.add(dst_idx)
        elif mnemonic in (M.VPERMQ, M.VPERMPD):
            src = read_vec_or_mem(1, ins, "load256")
            dst_idx, dst, _ = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = op_vpermq({src}, {imm:#x}u);")
            defined.add(dst_idx)
        elif mnemonic == M.VPBLENDD:
            src1, wide = read_vec(1, ins)
            src2 = read_vec_or_mem(2, ins, "load256" if wide else "load128")
            dst_idx, dst, dst_wide = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = op_vpblendd({src1}, {src2}, {imm:#x}u, {'true' if dst_wide else 'false'});")
            defined.add(dst_idx)
        elif mnemonic in SHIFT_HELPERS:
            src1, wide = read_vec(1, ins)
            src2 = read_vec_or_mem(2, ins, "load256" if wide else "load128")
            dst_idx, dst, dst_wide = write_vec(0, ins)
            helper = SHIFT_HELPERS[mnemonic]
            statements.append(f"{dst} = {helper}({src1}, {src2}, {'true' if dst_wide else 'false'});")
            defined.add(dst_idx)
        elif ins.code in GATHER_SPECS:
            helper, count, dst_wide_exp, idx_wide_exp, idx_bits = GATHER_SPECS[ins.code]
            if ins.op0_kind != OpKind.REGISTER or ins.op1_kind != OpKind.MEMORY or ins.op2_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported gather operands at RVA {ins.ip:#x}: {ins}")
            dst_idx, dst_wide = vec_index(ins.op_register(0), ins)
            mask_idx, mask_wide = vec_index(ins.op_register(2), ins)
            base_name, disp, scale, idx_idx = vsib_address(ins, idx_wide_exp, idx_bits)
            if dst_wide != dst_wide_exp or mask_wide != dst_wide_exp:
                raise ValueError(f"gather operand width mismatch at RVA {ins.ip:#x}: {ins}")
            if len({dst_idx, idx_idx, mask_idx}) != 3:
                raise ValueError(
                    f"overlapping gather registers (dest=ymm{dst_idx}, index=ymm{idx_idx}, mask=ymm{mask_idx}) "
                    f"at RVA {ins.ip:#x}: {ins}")
            for reg_idx in (dst_idx, idx_idx, mask_idx):
                if reg_idx not in defined:
                    raise ValueError(f"undefined vector register ymm{reg_idx} at RVA {ins.ip:#x}: {ins}")
                used.add(reg_idx)
            statements.append(
                f"{helper}(ymm{dst_idx}, {base_name}, {disp:#x}ull, {scale}ull, "
                f"ymm{idx_idx}, ymm{mask_idx}, {count}u);")
        elif ins.code in MASKMOV_SPECS:
            is_store, elem_bits, wide = MASKMOV_SPECS[ins.code]
            if is_store:
                if ins.op0_kind != OpKind.MEMORY or ins.op1_kind != OpKind.REGISTER or ins.op2_kind != OpKind.REGISTER:
                    raise ValueError(f"unsupported vpmaskmov store operands at RVA {ins.ip:#x}: {ins}")
                mask_name, mask_wide = read_vec(1, ins)
                src_name, src_wide = read_vec(2, ins)
                if mask_wide != wide or src_wide != wide:
                    raise ValueError(f"vpmaskmov width mismatch at RVA {ins.ip:#x}: {ins}")
                addr = flat_address(ins)
                helper = "op_vpmaskmovd_store" if elem_bits == 32 else "op_vpmaskmovq_store"
                statements.append(f"{helper}({addr}, {mask_name}, {src_name}, {'true' if wide else 'false'});")
            else:
                if ins.op0_kind != OpKind.REGISTER or ins.op1_kind != OpKind.REGISTER or ins.op2_kind != OpKind.MEMORY:
                    raise ValueError(f"unsupported vpmaskmov load operands at RVA {ins.ip:#x}: {ins}")
                mask_name, mask_wide = read_vec(1, ins)
                dst_idx, dst_name, dst_wide = write_vec(0, ins)
                if mask_wide != wide or dst_wide != wide:
                    raise ValueError(f"vpmaskmov width mismatch at RVA {ins.ip:#x}: {ins}")
                addr = flat_address(ins)
                helper = "op_vpmaskmovd_load" if elem_bits == 32 else "op_vpmaskmovq_load"
                statements.append(f"{dst_name} = {helper}({mask_name}, {addr}, {'true' if wide else 'false'});")
                defined.add(dst_idx)
        elif ins.code in FMA3_SPECS:
            kind, order, mode, wide = FMA3_SPECS[ins.code]
            op0_name, op0_wide = read_vec(0, ins)
            op1_name, op1_wide = read_vec(1, ins)
            if op0_wide != wide or op1_wide != wide:
                raise ValueError(f"FMA3 operand width mismatch at RVA {ins.ip:#x}: {ins}")
            if ins.op2_kind == OpKind.REGISTER:
                op2_expr, op2_wide = read_vec(2, ins)
                if op2_wide != wide:
                    raise ValueError(f"FMA3 operand width mismatch at RVA {ins.ip:#x}: {ins}")
            elif ins.op2_kind == OpKind.MEMORY:
                addr = flat_address(ins)
                if kind == "ss":
                    op2_expr = f"load_vec32({addr})"
                elif kind == "sd":
                    op2_expr = f"load_vec64({addr})"
                else:
                    op2_expr = f"{'load256' if wide else 'load128'}({addr})"
            else:
                raise ValueError(f"unsupported FMA3 operand 2 at RVA {ins.ip:#x}: {ins}")
            if kind in ("ss", "sd"):
                statements.append(f"{op0_name} = op_fma_{kind}({op0_name}, {op1_name}, {op2_expr}, {order}u, {mode}u);")
            else:
                statements.append(
                    f"{op0_name} = op_fma_{kind}({op0_name}, {op1_name}, {op2_expr}, {order}u, {mode}u, {'true' if wide else 'false'});")
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec256 ymm{idx}{{}};" for idx in sorted(used)]
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
    parser.add_argument("--mem-ops", action="store_true", help="allow memory operands on register-domain AVX2 forms")
    args = parser.parse_args()
    try:
        source, meta = translate(args.image, args.exports.split(","), allow_memory=args.mem_ops)
        args.output.write_text(source, encoding="utf-8")
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x}: {count} instructions")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
