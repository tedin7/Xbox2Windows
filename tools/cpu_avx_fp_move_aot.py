"""Lift bounded Windows x64 leaf functions with AVX quadword/GPR and masked-byte moves to C++.

Each export uses the leaf ABI void fn(word rcx, word rdx, word r8), where rcx,
rdx and r8 carry flat host pointers to caller-owned buffers (rcx and rdx are
inputs, r8 is the output). YMM0..YMM5 are modeled as portable 256-bit value
structs (four 64-bit lanes); every VEX.128 write to XMM0..XMM5 zeroes
[255:128], and vmovq XMM writes zero [255:64]. Volatile scratch GPRs
(rax, r9, r10, r11) support vmovq GPR transfers, rdi save/restore around
vmaskmovdqu, and 64-bit flat load/store glue.

Supported AVX instructions (docs/coverage/x64_zen2.tsv group AVX):
  - Quadword / GPR move: vmovq across all four VEX.128 Code encodings
    (VEX_VMOVQ_XMM_XMMM64, VEX_VMOVQ_XMMM64_XMM, VEX_VMOVQ_XMM_RM64,
    VEX_VMOVQ_RM64_XMM) covering all 5 operand shapes (xmm<-xmm, xmm<-m64,
    m64<-xmm, xmm<-r64, r64<-xmm)
  - Conditional byte store: vmaskmovdqu (VEX.128 xmm1, xmm2 to implicit [rdi],
    requiring rdi save/restore via a volatile scratch GPR around mov rdi, r8).
    Precondition for this portable slice: the entire 16-byte destination
    [rdi..rdi+15] is writable ordinary memory; the lifted helper touches only
    bytes whose mask MSB is set ((mask[i] & 0x80) != 0) and does not model
    implementation-dependent partial-fault or all-zero-mask page-edge faults.

Transport/ABI glue: vmovups, vmovaps, vmovdqu, vmovdqa (128/256-bit load, store
and register copy), vzeroupper, 64-bit mov (volatile GPR <-> [base + disp],
volatile GPR reg-reg, and rdi save/set-to-r8/restore), and terminal ret.

Alternating FP add/subtract: vaddsubps and vaddsubpd (VEX.128 and VEX.256; register or
flat m128/m256 second source). Even-indexed elements compute src1 - src2 and odd-indexed
elements compute src1 + src2 in every 128-bit lane (vaddsubpd: element 0 subtracts,
element 1 adds). The arithmetic is performed at run time by the matching host intrinsic
(_mm_addsub_ps, _mm256_addsub_ps, _mm_addsub_pd, _mm256_addsub_pd), so NaN propagation,
signed zeros, exceptions-as-masked and the MXCSR rounding, DAZ and FTZ bits are those of the
host instruction; nothing is folded at translation time. VEX.128 forms zero [255:128] of the
destination. Other FP arithmetic is not supported.

Legacy SSE forms (movq, maskmovdqu, addsubps, addsubpd), EVEX encodings, other FP arithmetic
(for example vaddps), nonvolatile registers (XMM6..15, YMM6..15,
RBX/RBP/RSI/R12..R15/RSP and unrestored, non-r8, or leaked RDI), non-argument
memory bases, stores outside r8, unaligned vmovaps/vmovdqa memory operands,
prefixes, RIP-relative addressing, branches, and trailing bytes after ret are
rejected.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import (
    Code,
    Decoder,
    EncodingKind,
    MemorySizeExt,
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

SCRATCH_GPR64 = {R.RAX: "rax", R.R9: "r9", R.R10: "r10", R.R11: "r11"}
VOLATILE_GPR64 = {**ARGUMENTS, **SCRATCH_GPR64}
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

VMOVQ_CODES = {
    Code.VEX_VMOVQ_XMM_XMMM64,
    Code.VEX_VMOVQ_XMMM64_XMM,
    Code.VEX_VMOVQ_XMM_RM64,
    Code.VEX_VMOVQ_RM64_XMM,
}

FULL_VEC_LOAD_CODES = {
    Code.VEX_VMOVUPS_XMM_XMMM128: (False, False),
    Code.VEX_VMOVUPS_YMM_YMMM256: (True, False),
    Code.VEX_VMOVDQU_XMM_XMMM128: (False, False),
    Code.VEX_VMOVDQU_YMM_YMMM256: (True, False),
    Code.VEX_VMOVAPS_XMM_XMMM128: (False, True),
    Code.VEX_VMOVAPS_YMM_YMMM256: (True, True),
    Code.VEX_VMOVDQA_XMM_XMMM128: (False, True),
    Code.VEX_VMOVDQA_YMM_YMMM256: (True, True),
}

FULL_VEC_STORE_CODES = {
    Code.VEX_VMOVUPS_XMMM128_XMM: (False, False),
    Code.VEX_VMOVUPS_YMMM256_YMM: (True, False),
    Code.VEX_VMOVDQU_XMMM128_XMM: (False, False),
    Code.VEX_VMOVDQU_YMMM256_YMM: (True, False),
    Code.VEX_VMOVAPS_XMMM128_XMM: (False, True),
    Code.VEX_VMOVAPS_YMMM256_YMM: (True, True),
    Code.VEX_VMOVDQA_XMMM128_XMM: (False, True),
    Code.VEX_VMOVDQA_YMMM256_YMM: (True, True),
}

LEGACY_SSE_REJECT_CODES = {
    Code.MOVQ_XMM_XMMM64,
    Code.MOVQ_XMMM64_XMM,
    Code.MOVQ_XMM_RM64,
    Code.MOVQ_RM64_XMM,
    Code.MOVQ_MM_MMM64,
    Code.MOVQ_MMM64_MM,
    Code.MASKMOVDQU_RDI_XMM_XMM,
}

# Code -> (256-bit form, lifted helper name).
VADDSUB_CODES = {
    Code.VEX_VADDSUBPS_XMM_XMM_XMMM128: (False, "op_vaddsubps128"),
    Code.VEX_VADDSUBPS_YMM_YMM_YMMM256: (True, "op_vaddsubps256"),
    Code.VEX_VADDSUBPD_XMM_XMM_XMMM128: (False, "op_vaddsubpd128"),
    Code.VEX_VADDSUBPD_YMM_YMM_YMMM256: (True, "op_vaddsubpd256"),
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.VEX_VZEROUPPER,
    Code.MOV_R64_RM64,
    Code.MOV_RM64_R64,
    Code.VEX_VMASKMOVDQU_RDI_XMM_XMM,
    *VMOVQ_CODES,
    *FULL_VEC_LOAD_CODES,
    *FULL_VEC_STORE_CODES,
    *VADDSUB_CODES,
}


PRELUDE = """#include <atomic>
#include <cstdint>
#include <cstring>
#include <intrin.h>
#include <immintrin.h>

extern "C" __declspec(dllexport) std::uint32_t host_exchange_mxcsr(std::uint32_t next) {
    const std::uint32_t current = _mm_getcsr();
    _mm_setcsr(next);
    return current;
}

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

// Guest reads can fault even when the translated result is dead. Keep surrounding guest
// writes on their original side of the read when the compiler inlines these helpers.
static inline std::uint64_t load64(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto value = *reinterpret_cast<const volatile std::uint64_t __unaligned*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return value;
}

static inline Vec256 load128(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* p = reinterpret_cast<const volatile std::uint64_t __unaligned*>(address);
    const std::uint64_t q0 = p[0];
    const std::uint64_t q1 = p[1];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, 0ull, 0ull}};
}

static inline Vec256 load256(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* p = reinterpret_cast<const volatile std::uint64_t __unaligned*>(address);
    const std::uint64_t q0 = p[0];
    const std::uint64_t q1 = p[1];
    const std::uint64_t q2 = p[2];
    const std::uint64_t q3 = p[3];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, q2, q3}};
}

static inline void store64(word address, std::uint64_t val) {
    std::memcpy(reinterpret_cast<void*>(address), &val, 8);
}

static inline void store128(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
}

static inline void store256(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 32);
}

static inline Vec256 copy128(const Vec256& v) {
    return {{v.q[0], v.q[1], 0ull, 0ull}};
}

// VEX alternating add/subtract: even elements src1 - src2, odd elements src1 + src2.
// Executed by the host instruction at run time so MXCSR (rounding, DAZ, FTZ), NaN
// propagation and signed zeros are the hardware's. The 128-bit forms zero [255:128].
static inline Vec256 op_vaddsubps128(const Vec256& a, const Vec256& b) {
    Vec256 r{};
    const __m128 x = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a.q)));
    const __m128 y = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b.q)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(r.q), _mm_castps_si128(_mm_addsub_ps(x, y)));
    return r;
}

static inline Vec256 op_vaddsubpd128(const Vec256& a, const Vec256& b) {
    Vec256 r{};
    const __m128d x = _mm_castsi128_pd(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a.q)));
    const __m128d y = _mm_castsi128_pd(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b.q)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(r.q), _mm_castpd_si128(_mm_addsub_pd(x, y)));
    return r;
}

static inline Vec256 op_vaddsubps256(const Vec256& a, const Vec256& b) {
    Vec256 r{};
    const __m256 x = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q)));
    const __m256 y = _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.q)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r.q), _mm256_castps_si256(_mm256_addsub_ps(x, y)));
    return r;
}

static inline Vec256 op_vaddsubpd256(const Vec256& a, const Vec256& b) {
    Vec256 r{};
    const __m256d x = _mm256_castsi256_pd(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q)));
    const __m256d y = _mm256_castsi256_pd(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.q)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r.q), _mm256_castpd_si256(_mm256_addsub_pd(x, y)));
    return r;
}

static inline void op_vmaskmovdqu(word address, const Vec256& src, const Vec256& mask) {
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


def gpr64_name(register, ins):
    """Map volatile 64-bit GPRs to C++ names, rejecting nonvolatile and sub-64-bit GPRs."""
    if register in VOLATILE_GPR64:
        return VOLATILE_GPR64[register]
    if register in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported GPR {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, clobbered_ptrs, *, expected_bytes=None, alignment=1, write=False):
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
    if disp % alignment:
        raise ValueError(f"unaligned {alignment}-byte memory operand at RVA {ins.ip:#x}: {ins}")
    name = ARGUMENTS[base]
    return f"({name} + {disp:#x}ull)" if disp else name


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line AVX quadword and byte-masked move leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used_vec = set()
    defined_vec = set()
    used_gpr = set()
    defined_gpr = {"rcx", "rdx", "r8"}
    clobbered_ptrs = set()
    # Track caller's nonvolatile RDI across save/set-to-r8/restore for vmaskmovdqu.
    caller_rdi_holders = {"rdi"}
    rdi_expr = None
    statements = []
    returned = False

    def read_vec(op_index, ins, *, require_xmm=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if require_xmm and wide:
            raise ValueError(f"unexpected YMM operand in VEX.128 instruction at RVA {ins.ip:#x}: {ins}")
        if idx not in defined_vec:
            raise ValueError(f"undefined vector register ymm{idx} at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return f"ymm{idx}", wide

    def write_vec(op_index, ins, *, require_xmm=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if require_xmm and wide:
            raise ValueError(f"unexpected YMM destination in VEX.128 instruction at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return idx, f"ymm{idx}", wide

    def read_gpr(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR operand {op_index} at RVA {ins.ip:#x}: {ins}")
        reg_name = gpr64_name(ins.op_register(op_index), ins)
        if reg_name not in defined_gpr:
            if reg_name in caller_rdi_holders:
                raise ValueError(f"GPR {reg_name} holds saved rdi at RVA {ins.ip:#x}: {ins}")
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        used_gpr.add(reg_name)
        return reg_name

    def write_gpr(op_index, ins, *, scratch_only=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR destination at RVA {ins.ip:#x}: {ins}")
        reg = ins.op_register(op_index)
        reg_name = gpr64_name(reg, ins)
        if reg_name == "r8":
            raise ValueError(f"unsupported write to output buffer pointer r8 at RVA {ins.ip:#x}: {ins}")
        if scratch_only and reg not in SCRATCH_GPR64:
            raise ValueError(f"expected volatile scratch GPR destination at RVA {ins.ip:#x}: {ins}")
        return reg_name

    def clobber_gpr(reg_name):
        caller_rdi_holders.discard(reg_name)
        if reg_name == "rcx":
            clobbered_ptrs.add(R.RCX)
        elif reg_name == "rdx":
            clobbered_ptrs.add(R.RDX)

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
        if ins.code in LEGACY_SSE_REJECT_CODES:
            raise ValueError(f"unsupported legacy SSE encoding at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            if "rdi" not in caller_rdi_holders:
                raise ValueError(f"unrestored nonvolatile GPR rdi at ret RVA {ins.ip:#x}: {ins}")
            statements.append("return;")
            returned = True

        elif ins.code == Code.VEX_VZEROUPPER:
            for idx in sorted(defined_vec):
                statements.append(f"ymm{idx}.q[2] = 0ull; ymm{idx}.q[3] = 0ull;")

        elif ins.code in VMOVQ_CODES:
            if ins.op0_kind == OpKind.MEMORY:
                src, _ = read_vec(1, ins, require_xmm=True)
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=8, write=True)
                statements.append(f"store64({addr}, {src}.q[0]);")
            elif ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=8)
                dst_idx, dst, _ = write_vec(0, ins, require_xmm=True)
                statements.append(f"{dst} = {{{{load64({addr}), 0ull, 0ull, 0ull}}}};")
                defined_vec.add(dst_idx)
            elif ins.code == Code.VEX_VMOVQ_XMM_RM64:
                src_gpr = read_gpr(1, ins)
                dst_idx, dst, _ = write_vec(0, ins, require_xmm=True)
                statements.append(f"{dst} = {{{{{src_gpr}, 0ull, 0ull, 0ull}}}};")
                defined_vec.add(dst_idx)
            elif ins.code == Code.VEX_VMOVQ_RM64_XMM:
                src, _ = read_vec(1, ins, require_xmm=True)
                dst_gpr = write_gpr(0, ins)
                clobber_gpr(dst_gpr)
                used_gpr.add(dst_gpr)
                statements.append(f"{dst_gpr} = {src}.q[0];")
                defined_gpr.add(dst_gpr)
            else:
                src, _ = read_vec(1, ins, require_xmm=True)
                dst_idx, dst, _ = write_vec(0, ins, require_xmm=True)
                statements.append(f"{dst} = {{{{{src}.q[0], 0ull, 0ull, 0ull}}}};")
                defined_vec.add(dst_idx)

        elif ins.code == Code.VEX_VMASKMOVDQU_RDI_XMM_XMM:
            if ins.op0_kind != OpKind.MEMORY_SEG_RDI or ins.memory_segment != R.DS:
                raise ValueError(f"unsupported vmaskmovdqu target at RVA {ins.ip:#x}: {ins}")
            if rdi_expr != "r8":
                raise ValueError(f"vmaskmovdqu requires r8-rooted rdi at RVA {ins.ip:#x}: {ins}")
            src, _ = read_vec(1, ins, require_xmm=True)
            mask, _ = read_vec(2, ins, require_xmm=True)
            statements.append(f"op_vmaskmovdqu({rdi_expr}, {src}, {mask});")

        elif ins.code in VADDSUB_CODES:
            wide, helper = VADDSUB_CODES[ins.code]
            src1, _ = read_vec(1, ins, require_xmm=not wide)
            if ins.op2_kind == OpKind.MEMORY:
                expected = 32 if wide else 16
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=expected)
                src2 = f"{'load256' if wide else 'load128'}({addr})"
            else:
                src2, _ = read_vec(2, ins, require_xmm=not wide)
            dst_idx, dst, _ = write_vec(0, ins, require_xmm=not wide)
            statements.append(f"{dst} = {helper}({src1}, {src2});")
            defined_vec.add(dst_idx)

        elif ins.code in FULL_VEC_LOAD_CODES:
            wide, aligned = FULL_VEC_LOAD_CODES[ins.code]
            expected = 32 if wide else 16
            align = expected if aligned else 1
            if ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=expected, alignment=align)
                dst_idx, dst, _ = write_vec(0, ins, require_xmm=not wide)
                statements.append(f"{dst} = {'load256' if wide else 'load128'}({addr});")
                defined_vec.add(dst_idx)
            else:
                src, _ = read_vec(1, ins, require_xmm=not wide)
                dst_idx, dst, _ = write_vec(0, ins, require_xmm=not wide)
                statements.append(f"{dst} = {src};" if wide else f"{dst} = copy128({src});")
                defined_vec.add(dst_idx)

        elif ins.code in FULL_VEC_STORE_CODES:
            wide, aligned = FULL_VEC_STORE_CODES[ins.code]
            expected = 32 if wide else 16
            align = expected if aligned else 1
            if ins.op0_kind == OpKind.MEMORY:
                src, _ = read_vec(1, ins, require_xmm=not wide)
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=expected, alignment=align, write=True)
                statements.append(f"{'store256' if wide else 'store128'}({addr}, {src});")
            else:
                src, _ = read_vec(1, ins, require_xmm=not wide)
                dst_idx, dst, _ = write_vec(0, ins, require_xmm=not wide)
                statements.append(f"{dst} = {src};" if wide else f"{dst} = copy128({src});")
                defined_vec.add(dst_idx)

        elif ins.code in (Code.MOV_R64_RM64, Code.MOV_RM64_R64):
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                op0_reg = ins.op0_register
                op1_reg = ins.op1_register
                if not (RegisterExt.is_gpr64(op0_reg) and RegisterExt.is_gpr64(op1_reg)):
                    raise ValueError(f"unsupported mov operand width at RVA {ins.ip:#x}: {ins}")
                if op1_reg == R.RDI:
                    dst_gpr = write_gpr(0, ins, scratch_only=True)
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
                else:
                    src_gpr = read_gpr(1, ins)
                    dst_gpr = write_gpr(0, ins)
                    clobber_gpr(dst_gpr)
                    used_gpr.add(dst_gpr)
                    statements.append(f"{dst_gpr} = {src_gpr};")
                    defined_gpr.add(dst_gpr)
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst_gpr = write_gpr(0, ins)
                addr = flat_address(ins, clobbered_ptrs, expected_bytes=8)
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

    decls = [f"    Vec256 ymm{idx}{{}};" for idx in sorted(used_vec)]
    for gpr in ("rax", "r9", "r10", "r11"):
        if gpr in used_gpr:
            decls.append(f"    word {gpr} = 0ull;")
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
