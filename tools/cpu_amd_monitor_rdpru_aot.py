"""Lift bounded Windows x64 leaf functions using AMD MONITORX, MWAITX, and RDPRU to C++.

Each export uses the straight-line leaf ABI:
    std::uint64_t lifted_<name>(word rcx, word rdx, word r8)
where rcx, rdx, and r8 carry flat caller-owned 64-bit argument values/pointers
(r8 is the output buffer; stores are allowed only to [r8 + disp] within the
bounded 64-byte output slot).

Supported AMD Zen 2 user-mode instructions (docs/coverage/x64_zen2.tsv group Integer):
  - RDPRU    (0F 01 FD): statically validates ECX in {0, 1} (0 = MPERF, 1 = APERF),
                         zero-extends EAX -> RAX and EDX -> RDX, preserves RCX and
                         RFLAGS, and resists /O2 CSE/DCE via volatile read + signal
                         fences. Bounded approximation: both selectors (0 and 1) map
                         to a serialized host RDTSC read (monotonic progression only;
                         does not return distinct hardware MPERF/APERF counter values).
  - MONITORX (0F 01 FA): arms address-range monitor on flat caller-owned
                         [rcx|rdx|r8] linear address in RAX with ECX == 0 and
                         EDX == 0; probes address readability, preserves all GPRs
                         and RFLAGS, and prevents elision/reordering of
                         surrounding observable stores.
  - MWAITX   (0F 01 FB): hint-driven wait using defined RAX hints and statically
                         validated ECX extensions (0 or 1; ECX[1] timer mode is
                         rejected because it implicitly reads nonvolatile EBX, and
                         reserved bits ECX[31:2] are rejected); preserves all GPRs
                         and RFLAGS without eliding surrounding observable stores.
                         Bounded approximation: lowers to a fenced _mm_pause()
                         (an allowed early wake-up).

Bounded volatile GPR transport (RAX, RCX, RDX, R8..R11 only):
  - mov (reg-reg, reg-imm, [r8 + disp] 32/64-bit store)
  - lea r64, [rcx|rdx|r8] (flat zero-displacement pointer copy)
  - xor, or, shl, add, cmp (32/64-bit reg-reg / reg-imm with CF/ZF/SF/OF model)
  - setb, sete, sets, seto (volatile 8-bit low GPRs; require defined flag & reg)
  - terminal ret (requires defined RAX)

MONITOR, MWAIT, RDTSC, RDTSCP, 16/32-bit MONITORX overrides, VEX/EVEX/non-legacy
encodings, vector/x87 instructions, nonvolatile GPRs (RBX, RBP, RSI, RDI, RSP,
R12..R15), undefined or out-of-range ECX on RDPRU, non-flat/unowned RAX or
undefined/non-zero ECX/EDX on MONITORX, undefined/timer/reserved ECX on MWAITX,
undefined GPR/flag reads, stores outside [r8], branches, prefixes, missing ret,
and trailing instructions after ret are rejected.
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
    OpKind,
    Register as R,
    RegisterExt,
)

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}

VOLATILE_GPR64 = {
    R.RAX: "rax", R.RCX: "rcx", R.RDX: "rdx",
    R.R8: "r8", R.R9: "r9", R.R10: "r10", R.R11: "r11",
}
VOLATILE_GPR32 = {
    R.EAX: "rax", R.ECX: "rcx", R.EDX: "rdx",
    R.R8D: "r8", R.R9D: "r9", R.R10D: "r10", R.R11D: "r11",
}
VOLATILE_GPR8 = {
    R.AL: "rax", R.CL: "rcx", R.DL: "rdx",
    R.R8L: "r8", R.R9L: "r9", R.R10L: "r10", R.R11L: "r11",
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

LEGACY_SYS_REJECT_CODES = {
    Code.MONITORW,
    Code.MONITORD,
    Code.MONITORQ,
    Code.MWAIT,
    Code.RDTSC,
    Code.RDTSCP,
}

MONITORX_BAD_ADDR_CODES = {
    Code.MONITORXW,
    Code.MONITORXD,
}

SETCC_CODES = {
    Code.SETB_RM8: "cf",
    Code.SETE_RM8: "zf",
    Code.SETS_RM8: "sf",
    Code.SETO_RM8: "of",
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.MONITORXQ,
    Code.MWAITX,
    Code.RDPRU,
    # mov
    Code.MOV_R64_RM64,
    Code.MOV_RM64_R64,
    Code.MOV_R64_IMM64,
    Code.MOV_RM64_IMM32,
    Code.MOV_R32_RM32,
    Code.MOV_RM32_R32,
    Code.MOV_R32_IMM32,
    Code.MOV_RM32_IMM32,
    # lea
    Code.LEA_R64_M,
    # xor
    Code.XOR_R64_RM64,
    Code.XOR_RM64_R64,
    Code.XOR_RM64_IMM8,
    Code.XOR_RM64_IMM32,
    Code.XOR_RAX_IMM32,
    Code.XOR_R32_RM32,
    Code.XOR_RM32_R32,
    Code.XOR_RM32_IMM8,
    Code.XOR_RM32_IMM32,
    Code.XOR_EAX_IMM32,
    # or
    Code.OR_R64_RM64,
    Code.OR_RM64_R64,
    Code.OR_RM64_IMM8,
    Code.OR_RM64_IMM32,
    Code.OR_RAX_IMM32,
    Code.OR_R32_RM32,
    Code.OR_RM32_R32,
    Code.OR_RM32_IMM8,
    Code.OR_RM32_IMM32,
    Code.OR_EAX_IMM32,
    # shl
    Code.SHL_RM64_1,
    Code.SHL_RM64_IMM8,
    Code.SHL_RM32_1,
    Code.SHL_RM32_IMM8,
    # add
    Code.ADD_R64_RM64,
    Code.ADD_RM64_R64,
    Code.ADD_RM64_IMM8,
    Code.ADD_RM64_IMM32,
    Code.ADD_RAX_IMM32,
    Code.ADD_R32_RM32,
    Code.ADD_RM32_R32,
    Code.ADD_RM32_IMM8,
    Code.ADD_RM32_IMM32,
    Code.ADD_EAX_IMM32,
    # cmp
    Code.CMP_R64_RM64,
    Code.CMP_RM64_R64,
    Code.CMP_RM64_IMM8,
    Code.CMP_RM64_IMM32,
    Code.CMP_RAX_IMM32,
    Code.CMP_R32_RM32,
    Code.CMP_RM32_R32,
    Code.CMP_RM32_IMM8,
    Code.CMP_RM32_IMM32,
    Code.CMP_EAX_IMM32,
    # setcc
    *SETCC_CODES,
}

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
MAX_OUT_BYTES = 64


PRELUDE = r"""#include <atomic>
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER)
#include <emmintrin.h>
#include <intrin.h>
#endif

#if defined(_MSC_VER)
// Host AMD CPUID feature probes:
//   MWAITX: CPUID Fn8000_0001_ECX[29]
//   RDPRU : CPUID Fn8000_0008_EBX[4]
extern "C" __declspec(dllexport) std::uint32_t host_has_mwaitx() {
    int info[4]{};
    __cpuid(info, static_cast<int>(0x80000000u));
    if (static_cast<std::uint32_t>(info[0]) < 0x80000001u) return 0u;
    __cpuid(info, static_cast<int>(0x80000001u));
    return (static_cast<std::uint32_t>(info[2]) & (1u << 29)) != 0u ? 1u : 0u;
}

extern "C" __declspec(dllexport) std::uint32_t host_has_rdpru() {
    int info[4]{};
    __cpuid(info, static_cast<int>(0x80000000u));
    if (static_cast<std::uint32_t>(info[0]) < 0x80000008u) return 0u;
    __cpuid(info, static_cast<int>(0x80000008u));
    return (static_cast<std::uint32_t>(info[1]) & (1u << 4)) != 0u ? 1u : 0u;
}

extern "C" __declspec(dllexport) std::uint64_t host_serial_tsc() {
    _mm_lfence();
    const volatile std::uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}
#endif

using word = std::uint64_t;

static inline void store32(word address, std::uint32_t value) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(value));
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store64(word address, std::uint64_t value) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(value));
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

// RDPRU bounded approximation: ECX (0 = MPERF, 1 = APERF) is statically validated,
// and both selectors map to a serialized host RDTSC read (monotonic only, not true
// hardware MPERF/APERF). The volatile capture and signal fences prevent /O2 from
// CSE-merging consecutive reads or DCE-eliminating overwritten reads.
static inline std::uint64_t op_rdpru_read(std::uint32_t selector) {
    (void)selector;
    std::atomic_signal_fence(std::memory_order_seq_cst);
#if defined(_MSC_VER)
    const volatile std::uint64_t counter = __rdtsc();
#else
    std::uint32_t lo = 0u, hi = 0u;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    const volatile std::uint64_t counter = (static_cast<std::uint64_t>(hi) << 32) | lo;
#endif
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return counter;
}

// MONITORX: probes readability of the flat linear address (preserving #PF fault
// semantics and surrounding store ordering) without modifying memory, GPRs, or RFLAGS.
static inline void op_monitorx(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const volatile std::uint8_t armed = *reinterpret_cast<const volatile std::uint8_t*>(address);
    (void)armed;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

// MWAITX bounded approximation: lowers to a fenced _mm_pause() (an allowed early
// wake-up; ECX in {0, 1}; timer mode ECX[1] rejected at lift time). Preserves all
// GPRs and RFLAGS and keeps surrounding observable stores ordered.
static inline void op_mwaitx(std::uint32_t hints, std::uint32_t extensions) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const volatile std::uint32_t observed = hints ^ (extensions << 16);
    (void)observed;
#if defined(_MSC_VER)
    _mm_pause();
#endif
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline std::uint64_t op_add64(std::uint64_t a, std::uint64_t b, bool& cf, bool& zf, bool& sf, bool& of) {
    const std::uint64_t r = a + b;
    cf = r < a;
    zf = r == 0ull;
    sf = (r >> 63) != 0ull;
    of = (((~(a ^ b)) & (a ^ r)) >> 63) != 0ull;
    return r;
}

static inline std::uint64_t op_add32(std::uint64_t a64, std::uint64_t b64, bool& cf, bool& zf, bool& sf, bool& of) {
    const std::uint32_t a = static_cast<std::uint32_t>(a64);
    const std::uint32_t b = static_cast<std::uint32_t>(b64);
    const std::uint32_t r = a + b;
    cf = r < a;
    zf = r == 0u;
    sf = (r >> 31) != 0u;
    of = (((~(a ^ b)) & (a ^ r)) >> 31) != 0u;
    return static_cast<std::uint64_t>(r);
}

static inline void op_cmp64(std::uint64_t a, std::uint64_t b, bool& cf, bool& zf, bool& sf, bool& of) {
    const std::uint64_t r = a - b;
    cf = a < b;
    zf = r == 0ull;
    sf = (r >> 63) != 0ull;
    of = (((a ^ b) & (a ^ r)) >> 63) != 0ull;
}

static inline void op_cmp32(std::uint64_t a64, std::uint64_t b64, bool& cf, bool& zf, bool& sf, bool& of) {
    const std::uint32_t a = static_cast<std::uint32_t>(a64);
    const std::uint32_t b = static_cast<std::uint32_t>(b64);
    const std::uint32_t r = a - b;
    cf = a < b;
    zf = r == 0u;
    sf = (r >> 31) != 0u;
    of = (((a ^ b) & (a ^ r)) >> 31) != 0u;
}

static inline std::uint64_t op_logic64(std::uint64_t r, bool& cf, bool& zf, bool& sf, bool& of) {
    cf = false;
    of = false;
    zf = r == 0ull;
    sf = (r >> 63) != 0ull;
    return r;
}

static inline std::uint64_t op_logic32(std::uint32_t r, bool& cf, bool& zf, bool& sf, bool& of) {
    cf = false;
    of = false;
    zf = r == 0u;
    sf = (r >> 31) != 0u;
    return static_cast<std::uint64_t>(r);
}

static inline std::uint64_t op_shl64(std::uint64_t a, unsigned count, bool& cf, bool& zf, bool& sf, bool& of) {
    count &= 63u;
    if (count == 0u) return a;
    const std::uint64_t r = a << count;
    cf = ((a >> (64u - count)) & 1ull) != 0ull;
    zf = r == 0ull;
    sf = (r >> 63) != 0ull;
    if (count == 1u) of = ((r >> 63) != 0ull) != cf;
    return r;
}

static inline std::uint64_t op_shl32(std::uint64_t a64, unsigned count, bool& cf, bool& zf, bool& sf, bool& of) {
    const std::uint32_t a = static_cast<std::uint32_t>(a64);
    count &= 31u;
    if (count == 0u) return static_cast<std::uint64_t>(a);
    const std::uint32_t r = a << count;
    cf = ((a >> (32u - count)) & 1u) != 0u;
    zf = r == 0u;
    sf = (r >> 31) != 0u;
    if (count == 1u) of = ((r >> 31) != 0u) != cf;
    return static_cast<std::uint64_t>(r);
}
"""


def _check_reg_class(reg, table, ins):
    if reg in table:
        return table[reg]
    if reg in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {reg} at RVA {ins.ip:#x}: {ins}")
    if RegisterExt.is_vector_register(reg):
        raise ValueError(f"unsupported vector register {reg} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported register {reg} at RVA {ins.ip:#x}: {ins}")


def _store_r8_address(ins, width):
    """Lower a flat [r8 + disp] output store within [0, MAX_OUT_BYTES - width]."""
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
    if base != R.R8:
        raise ValueError(f"store outside output buffer r8 at RVA {ins.ip:#x}: {ins}")
    if MemorySizeExt.size(ins.memory_size) != width:
        raise ValueError(f"unsupported memory width at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & MASK64
    if disp > MAX_OUT_BYTES - width:
        raise ValueError(f"out-of-buffer store displacement {disp:#x} at RVA {ins.ip:#x}: {ins}")
    return f"(r8 + {disp:#x}ull)" if disp else "r8"


def _lea_flat_base(ins):
    """Lower a flat zero-displacement [rcx|rdx|r8] LEA address."""
    base = ins.memory_base
    if (
        ins.is_ip_rel_memory_operand
        or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
        or ins.memory_index != R.NONE
        or not RegisterExt.is_gpr64(base)
    ):
        raise ValueError(f"unsupported lea memory operand at RVA {ins.ip:#x}: {ins}")
    if base in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR base {base} in lea at RVA {ins.ip:#x}: {ins}")
    if base not in ARGUMENTS:
        raise ValueError(f"unsupported lea base {base} at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & MASK64
    if disp != 0:
        raise ValueError(f"unsupported displaced lea operand at RVA {ins.ip:#x}: {ins}")
    return ARGUMENTS[base]


def _imm_value(ins, op_index, *, width):
    kind = ins.op_kind(op_index)
    if kind == OpKind.IMMEDIATE8:
        return ins.immediate8 & 0xFF
    if kind == OpKind.IMMEDIATE8TO32:
        return ins.immediate8to32 & MASK32
    if kind == OpKind.IMMEDIATE8TO64:
        return ins.immediate8to64 & MASK64
    if kind == OpKind.IMMEDIATE32:
        return ins.immediate32 & MASK32
    if kind == OpKind.IMMEDIATE32TO64:
        return ins.immediate32to64 & MASK64
    if kind == OpKind.IMMEDIATE64:
        return ins.immediate64 & MASK64
    raise ValueError(f"unsupported immediate kind {kind} at RVA {ins.ip:#x}: {ins}")


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line AMD MONITORX/MWAITX/RDPRU leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    defined = {"rcx", "rdx", "r8"}
    const_val = {}
    flat_ptr = {"rcx", "rdx", "r8"}
    used_temps = set()
    flags_defined = set()
    used_flags = False
    statements = []
    returned = False

    def mark_reg(reg_name):
        if reg_name not in ("rcx", "rdx", "r8"):
            used_temps.add(reg_name)

    def read_reg(op_index, table, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        rname = _check_reg_class(ins.op_register(op_index), table, ins)
        if rname not in defined:
            raise ValueError(f"undefined register {rname} at RVA {ins.ip:#x}: {ins}")
        mark_reg(rname)
        return rname

    def write_reg(op_index, table, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        rname = _check_reg_class(ins.op_register(op_index), table, ins)
        mark_reg(rname)
        return rname

    def kill_provenance(rname):
        const_val.pop(rname, None)
        flat_ptr.discard(rname)

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
        op_code = ins.op_code()
        if (
            op_code.encoding == EncodingKind.EVEX
            or ins.op_mask != R.NONE
            or ins.zeroing_masking
            or ins.is_broadcast
        ):
            raise ValueError(f"unsupported EVEX encoding at RVA {ins.ip:#x}: {ins}")
        if op_code.encoding == EncodingKind.VEX:
            raise ValueError(f"unsupported VEX encoding at RVA {ins.ip:#x}: {ins}")
        if op_code.encoding != EncodingKind.LEGACY:
            raise ValueError(f"unsupported non-legacy encoding at RVA {ins.ip:#x}: {ins}")
        if ins.code in LEGACY_SYS_REJECT_CODES:
            raise ValueError(f"unsupported legacy monitor/mwait/rdtsc instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code in MONITORX_BAD_ADDR_CODES:
            raise ValueError(f"unsupported non-64-bit monitorx address size at RVA {ins.ip:#x}: {ins}")
        if ins.flow_control not in (FlowControl.NEXT, FlowControl.RETURN):
            raise ValueError(f"unsupported control-flow instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        code = ins.code

        if code == Code.RETNQ:
            if "rax" not in defined:
                raise ValueError(f"undefined return register rax at ret RVA {ins.ip:#x}")
            mark_reg("rax")
            statements.append("return rax;")
            returned = True

        elif code == Code.RDPRU:
            if "rcx" not in defined:
                raise ValueError(f"undefined ECX for rdpru at RVA {ins.ip:#x}")
            if "rcx" not in const_val:
                raise ValueError(f"non-constant/unproven ECX for rdpru at RVA {ins.ip:#x}")
            sel = const_val["rcx"] & MASK32
            if sel not in (0, 1):
                raise ValueError(f"out-of-range ECX={sel:#x} for rdpru at RVA {ins.ip:#x}")
            mark_reg("rax")
            statements.append(
                f"{{ const std::uint64_t pru = op_rdpru_read({sel}u); "
                f"rax = static_cast<std::uint32_t>(pru); rdx = pru >> 32; }}"
            )
            defined.add("rax")
            defined.add("rdx")
            kill_provenance("rax")
            kill_provenance("rdx")

        elif code == Code.MONITORXQ:
            if "rax" not in defined or "rax" not in flat_ptr:
                raise ValueError(f"non-flat or unowned RAX address for monitorx at RVA {ins.ip:#x}")
            if "rcx" not in defined or "rcx" not in const_val or (const_val["rcx"] & MASK32) != 0:
                raise ValueError(f"undefined or non-zero ECX for monitorx at RVA {ins.ip:#x}")
            if "rdx" not in defined or "rdx" not in const_val or (const_val["rdx"] & MASK32) != 0:
                raise ValueError(f"undefined or non-zero EDX for monitorx at RVA {ins.ip:#x}")
            mark_reg("rax")
            statements.append("op_monitorx(rax);")

        elif code == Code.MWAITX:
            if "rax" not in defined:
                raise ValueError(f"undefined RAX hints for mwaitx at RVA {ins.ip:#x}")
            if "rcx" not in defined or "rcx" not in const_val:
                raise ValueError(f"undefined or non-constant ECX extensions for mwaitx at RVA {ins.ip:#x}")
            ext = const_val["rcx"] & MASK32
            if ext & 0x2:
                raise ValueError(
                    f"unsupported MWAITX timer extension ECX[1] (requires nonvolatile EBX) at RVA {ins.ip:#x}"
                )
            if ext & ~0x3:
                raise ValueError(f"unsupported reserved MWAITX extension bits ECX={ext:#x} at RVA {ins.ip:#x}")
            mark_reg("rax")
            statements.append(f"op_mwaitx(static_cast<std::uint32_t>(rax), {ext}u);")

        elif code == Code.LEA_R64_M:
            dst = write_reg(0, VOLATILE_GPR64, ins)
            src_base = _lea_flat_base(ins)
            if src_base not in defined:
                raise ValueError(f"undefined lea base {src_base} at RVA {ins.ip:#x}: {ins}")
            statements.append(f"{dst} = {src_base};")
            defined.add(dst)
            kill_provenance(dst)
            if src_base in flat_ptr:
                flat_ptr.add(dst)
            if src_base in const_val:
                const_val[dst] = const_val[src_base]

        elif code in (Code.MOV_R64_RM64, Code.MOV_RM64_R64):
            if ins.op0_kind == OpKind.MEMORY:
                src = read_reg(1, VOLATILE_GPR64, ins)
                addr = _store_r8_address(ins, 8)
                statements.append(f"store64({addr}, {src});")
            elif ins.op1_kind == OpKind.MEMORY:
                raise ValueError(f"unsupported memory load at RVA {ins.ip:#x}: {ins}")
            else:
                src = read_reg(1, VOLATILE_GPR64, ins)
                dst = write_reg(0, VOLATILE_GPR64, ins)
                statements.append(f"{dst} = {src};")
                defined.add(dst)
                if src in const_val:
                    const_val[dst] = const_val[src]
                else:
                    const_val.pop(dst, None)
                if src in flat_ptr:
                    flat_ptr.add(dst)
                else:
                    flat_ptr.discard(dst)

        elif code in (Code.MOV_R64_IMM64, Code.MOV_RM64_IMM32):
            if ins.op0_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported memory immediate store at RVA {ins.ip:#x}: {ins}")
            dst = write_reg(0, VOLATILE_GPR64, ins)
            imm = _imm_value(ins, 1, width=64)
            statements.append(f"{dst} = {imm:#x}ull;")
            defined.add(dst)
            flat_ptr.discard(dst)
            const_val[dst] = imm

        elif code in (Code.MOV_R32_RM32, Code.MOV_RM32_R32):
            if ins.op0_kind == OpKind.MEMORY:
                src = read_reg(1, VOLATILE_GPR32, ins)
                addr = _store_r8_address(ins, 4)
                statements.append(f"store32({addr}, static_cast<std::uint32_t>({src}));")
            elif ins.op1_kind == OpKind.MEMORY:
                raise ValueError(f"unsupported memory load at RVA {ins.ip:#x}: {ins}")
            else:
                src = read_reg(1, VOLATILE_GPR32, ins)
                dst = write_reg(0, VOLATILE_GPR32, ins)
                statements.append(f"{dst} = static_cast<std::uint32_t>({src});")
                defined.add(dst)
                flat_ptr.discard(dst)
                if src in const_val:
                    const_val[dst] = const_val[src] & MASK32
                else:
                    const_val.pop(dst, None)

        elif code in (Code.MOV_R32_IMM32, Code.MOV_RM32_IMM32):
            if ins.op0_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported memory immediate store at RVA {ins.ip:#x}: {ins}")
            dst = write_reg(0, VOLATILE_GPR32, ins)
            imm = _imm_value(ins, 1, width=32)
            statements.append(f"{dst} = {imm:#x}ull;")
            defined.add(dst)
            flat_ptr.discard(dst)
            const_val[dst] = imm

        elif code in (
            Code.XOR_R64_RM64, Code.XOR_RM64_R64, Code.XOR_RM64_IMM8, Code.XOR_RM64_IMM32, Code.XOR_RAX_IMM32,
            Code.XOR_R32_RM32, Code.XOR_RM32_R32, Code.XOR_RM32_IMM8, Code.XOR_RM32_IMM32, Code.XOR_EAX_IMM32,
        ):
            is64 = code in (Code.XOR_R64_RM64, Code.XOR_RM64_R64, Code.XOR_RM64_IMM8, Code.XOR_RM64_IMM32, Code.XOR_RAX_IMM32)
            table = VOLATILE_GPR64 if is64 else VOLATILE_GPR32
            if ins.op0_kind != OpKind.REGISTER or ins.op1_kind == OpKind.MEMORY:
                raise ValueError(f"unsupported memory xor at RVA {ins.ip:#x}: {ins}")
            used_flags = True
            flags_defined = {"cf", "zf", "sf", "of"}
            if ins.op1_kind == OpKind.REGISTER and ins.op0_register == ins.op1_register:
                dst = write_reg(0, table, ins)
                statements.append(f"{dst} = op_logic{'64' if is64 else '32'}(0u, cf, zf, sf, of);")
                defined.add(dst)
                flat_ptr.discard(dst)
                const_val[dst] = 0
            else:
                dst = read_reg(0, table, ins)
                if ins.op1_kind == OpKind.REGISTER:
                    src = read_reg(1, table, ins)
                    src_expr = src if is64 else f"static_cast<std::uint32_t>({src})"
                    src_c = const_val.get(src)
                else:
                    imm = _imm_value(ins, 1, width=64 if is64 else 32)
                    src_expr = f"{imm:#x}ull" if is64 else f"{imm:#x}u"
                    src_c = imm
                dst_expr = dst if is64 else f"static_cast<std::uint32_t>({dst})"
                statements.append(f"{dst} = op_logic{'64' if is64 else '32'}({dst_expr} ^ {src_expr}, cf, zf, sf, of);")
                flat_ptr.discard(dst)
                if dst in const_val and src_c is not None:
                    const_val[dst] = (const_val[dst] ^ src_c) & (MASK64 if is64 else MASK32)
                else:
                    const_val.pop(dst, None)

        elif code in (
            Code.OR_R64_RM64, Code.OR_RM64_R64, Code.OR_RM64_IMM8, Code.OR_RM64_IMM32, Code.OR_RAX_IMM32,
            Code.OR_R32_RM32, Code.OR_RM32_R32, Code.OR_RM32_IMM8, Code.OR_RM32_IMM32, Code.OR_EAX_IMM32,
        ):
            is64 = code in (Code.OR_R64_RM64, Code.OR_RM64_R64, Code.OR_RM64_IMM8, Code.OR_RM64_IMM32, Code.OR_RAX_IMM32)
            table = VOLATILE_GPR64 if is64 else VOLATILE_GPR32
            if ins.op0_kind != OpKind.REGISTER or ins.op1_kind == OpKind.MEMORY:
                raise ValueError(f"unsupported memory or at RVA {ins.ip:#x}: {ins}")
            used_flags = True
            flags_defined = {"cf", "zf", "sf", "of"}
            dst = read_reg(0, table, ins)
            if ins.op1_kind == OpKind.REGISTER:
                src = read_reg(1, table, ins)
                src_expr = src if is64 else f"static_cast<std::uint32_t>({src})"
                src_c = const_val.get(src)
            else:
                imm = _imm_value(ins, 1, width=64 if is64 else 32)
                src_expr = f"{imm:#x}ull" if is64 else f"{imm:#x}u"
                src_c = imm
            dst_expr = dst if is64 else f"static_cast<std::uint32_t>({dst})"
            statements.append(f"{dst} = op_logic{'64' if is64 else '32'}({dst_expr} | {src_expr}, cf, zf, sf, of);")
            flat_ptr.discard(dst)
            if dst in const_val and src_c is not None:
                const_val[dst] = (const_val[dst] | src_c) & (MASK64 if is64 else MASK32)
            else:
                const_val.pop(dst, None)

        elif code in (Code.SHL_RM64_1, Code.SHL_RM64_IMM8, Code.SHL_RM32_1, Code.SHL_RM32_IMM8):
            is64 = code in (Code.SHL_RM64_1, Code.SHL_RM64_IMM8)
            table = VOLATILE_GPR64 if is64 else VOLATILE_GPR32
            if ins.op0_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported memory shl at RVA {ins.ip:#x}: {ins}")
            dst = read_reg(0, table, ins)
            cnt = 1 if code in (Code.SHL_RM64_1, Code.SHL_RM32_1) else (ins.immediate8 & 0xFF)
            eff = cnt & (63 if is64 else 31)
            used_flags = True
            statements.append(f"{dst} = op_shl{'64' if is64 else '32'}({dst}, {cnt}u, cf, zf, sf, of);")
            if eff == 1:
                flags_defined = {"cf", "zf", "sf", "of"}
            elif eff > 1:
                flags_defined = {"cf", "zf", "sf"}
            flat_ptr.discard(dst)
            if dst in const_val:
                mask = MASK64 if is64 else MASK32
                const_val[dst] = ((const_val[dst] & mask) << eff) & mask

        elif code in (
            Code.ADD_R64_RM64, Code.ADD_RM64_R64, Code.ADD_RM64_IMM8, Code.ADD_RM64_IMM32, Code.ADD_RAX_IMM32,
            Code.ADD_R32_RM32, Code.ADD_RM32_R32, Code.ADD_RM32_IMM8, Code.ADD_RM32_IMM32, Code.ADD_EAX_IMM32,
        ):
            is64 = code in (Code.ADD_R64_RM64, Code.ADD_RM64_R64, Code.ADD_RM64_IMM8, Code.ADD_RM64_IMM32, Code.ADD_RAX_IMM32)
            table = VOLATILE_GPR64 if is64 else VOLATILE_GPR32
            if ins.op0_kind != OpKind.REGISTER or ins.op1_kind == OpKind.MEMORY:
                raise ValueError(f"unsupported memory add at RVA {ins.ip:#x}: {ins}")
            used_flags = True
            flags_defined = {"cf", "zf", "sf", "of"}
            dst = read_reg(0, table, ins)
            if ins.op1_kind == OpKind.REGISTER:
                src = read_reg(1, table, ins)
                src_expr = src
                src_c = const_val.get(src)
            else:
                imm = _imm_value(ins, 1, width=64 if is64 else 32)
                src_expr = f"{imm:#x}ull"
                src_c = imm
            statements.append(f"{dst} = op_add{'64' if is64 else '32'}({dst}, {src_expr}, cf, zf, sf, of);")
            flat_ptr.discard(dst)
            if dst in const_val and src_c is not None:
                const_val[dst] = (const_val[dst] + src_c) & (MASK64 if is64 else MASK32)
            else:
                const_val.pop(dst, None)

        elif code in (
            Code.CMP_R64_RM64, Code.CMP_RM64_R64, Code.CMP_RM64_IMM8, Code.CMP_RM64_IMM32, Code.CMP_RAX_IMM32,
            Code.CMP_R32_RM32, Code.CMP_RM32_R32, Code.CMP_RM32_IMM8, Code.CMP_RM32_IMM32, Code.CMP_EAX_IMM32,
        ):
            is64 = code in (Code.CMP_R64_RM64, Code.CMP_RM64_R64, Code.CMP_RM64_IMM8, Code.CMP_RM64_IMM32, Code.CMP_RAX_IMM32)
            table = VOLATILE_GPR64 if is64 else VOLATILE_GPR32
            if ins.op0_kind != OpKind.REGISTER or ins.op1_kind == OpKind.MEMORY:
                raise ValueError(f"unsupported memory cmp at RVA {ins.ip:#x}: {ins}")
            used_flags = True
            flags_defined = {"cf", "zf", "sf", "of"}
            lhs = read_reg(0, table, ins)
            if ins.op1_kind == OpKind.REGISTER:
                rhs_expr = read_reg(1, table, ins)
            else:
                imm = _imm_value(ins, 1, width=64 if is64 else 32)
                rhs_expr = f"{imm:#x}ull"
            statements.append(f"op_cmp{'64' if is64 else '32'}({lhs}, {rhs_expr}, cf, zf, sf, of);")

        elif code in SETCC_CODES:
            if ins.op0_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported memory setcc at RVA {ins.ip:#x}: {ins}")
            flag = SETCC_CODES[code]
            if flag not in flags_defined:
                raise ValueError(f"undefined flag {flag} read at RVA {ins.ip:#x}: {ins}")
            dst = read_reg(0, VOLATILE_GPR8, ins)
            used_flags = True
            statements.append(f"{dst} = ({dst} & ~0xffull) | ({flag} ? 1ull : 0ull);")
            kill_provenance(dst)

        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    word {reg} = 0ull;" for reg in sorted(used_temps)]
    if used_flags:
        decls.append("    bool cf = false, zf = false, sf = false, of = false;")
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
