"""Lift bounded Windows x64 leaf functions with MMX/SSE2 conversion and bridge ops to C++.

Each export uses the leaf ABI void fn(word rcx, word rdx, word r8), where rcx,
rdx and r8 carry host pointers to caller-owned buffers (16-byte-aligned bases
when 16-byte-aligned memory operands are used). XMM0..XMM5 are modeled as
portable 128-bit value structs (Vec128, two 64-bit lanes) and MM0..MM7 are
modeled as std::uint64_t mm[8]. Volatile scratch GPRs (rax, r9, r10, r11) carry
64-bit movq register bridge values. Only legacy encodings are accepted; VEX,
EVEX and other non-legacy forms reject.

Supported instructions:
  - MMX/SSE2 conversion & bridge ops: cvtpd2pi, cvttpd2pi, cvtpi2pd, movdq2q,
    movq2dq, emms
  - Transport & ABI glue: movdqa (16-byte aligned), movdqu, movq (between
    mm/scratch-r64/[rcx|rdx|r8]), and terminal ret

MMX state policy:
  - Call with an empty x87 stack, no live MMX values, and default MXCSR
    controls (round-to-nearest-even, exceptions masked, DAZ and FTZ off).
    Floating-point status flags and physical x87 tag transitions are outside
    the ABI. The exported result is the caller's output bytes.
  - mmx_active starts false and becomes true whenever any MM0..MM7 register is
    touched (cvtpi2pd xmm, m64 reads memory without touching MMX registers or
    setting mmx_active); ret rejects while mmx_active is true.
  - emms clears mmx_active and invalidates active MM register definitions, and
    is allowed unconditionally (even when mmx_active is false).

Memory policy:
  - All memory operands must be flat [base + disp] rooted in rcx, rdx or r8.
  - Memory stores are restricted to r8.
  - cvtpd2pi/cvttpd2pi m128 and movdqa m128 require disp % 16 == 0; cvtpi2pd m64
    and movq m64 allow any byte displacement.
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

CVT_PD2PI_HELPERS = {
    Code.CVTPD2PI_MM_XMMM128: "op_cvtpd2pi",
    Code.CVTTPD2PI_MM_XMMM128: "op_cvttpd2pi",
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.EMMS,
    *CVT_PD2PI_HELPERS,
    Code.CVTPI2PD_XMM_MMM64,
    Code.MOVDQ2Q_MM_XMM,
    Code.MOVQ2DQ_XMM_MM,
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

static inline double q_as_double(std::uint64_t bits) {
    double v;
    std::memcpy(&v, &bits, 8);
    return v;
}

static inline std::uint64_t double_as_q(double v) {
    std::uint64_t bits = 0ull;
    std::memcpy(&bits, &v, 8);
    return bits;
}

static inline Vec128 op_cvtpi2pd(std::uint64_t src) {
    const auto lo = static_cast<std::int32_t>(static_cast<std::uint32_t>(src));
    const auto hi = static_cast<std::int32_t>(static_cast<std::uint32_t>(src >> 32));
    return {{double_as_q(static_cast<double>(lo)), double_as_q(static_cast<double>(hi))}};
}

// Default MXCSR round-to-nearest-even; NaN, infinity and out-of-int32_t range
// after rounding yield the 32-bit integer indefinite value (0x80000000).
static inline std::uint32_t cvt_round_i32(double v) {
    const double r = std::nearbyint(v);
    const bool valid = std::isfinite(r) && (r >= -2147483648.0 && r <= 2147483647.0);
    return valid
        ? static_cast<std::uint32_t>(static_cast<std::int32_t>(r))
        : 0x80000000u;
}

// Truncate toward zero; NaN, infinity and out-of-int32_t range yield 0x80000000.
static inline std::uint32_t cvt_trunc_i32(double v) {
    const double r = std::trunc(v);
    const bool valid = std::isfinite(r) && (r >= -2147483648.0 && r <= 2147483647.0);
    return valid
        ? static_cast<std::uint32_t>(static_cast<std::int32_t>(r))
        : 0x80000000u;
}

static inline std::uint64_t op_cvtpd2pi(const Vec128& a) {
    const std::uint32_t lo = cvt_round_i32(q_as_double(a.q[0]));
    const std::uint32_t hi = cvt_round_i32(q_as_double(a.q[1]));
    return static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32);
}

static inline std::uint64_t op_cvttpd2pi(const Vec128& a) {
    const std::uint32_t lo = cvt_trunc_i32(q_as_double(a.q[0]));
    const std::uint32_t hi = cvt_trunc_i32(q_as_double(a.q[1]));
    return static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32);
}
"""


def vec_index(register, ins):
    """Map XMM0..XMM5 to an index, rejecting all others."""
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
            or ins.memory_index != R.NONE or not RegisterExt.is_gpr64(base)
            or base not in ARGUMENTS):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    if store and base != R.R8:
        raise ValueError(f"store outside the output buffer r8 at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    if disp % alignment:
        raise ValueError(f"unaligned {alignment}-byte memory operand at RVA {ins.ip:#x}: {ins}")
    name = ARGUMENTS[base]
    return f"({name} + {disp:#x}ull)" if disp else name


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line MMX/SSE2 leaf function."""
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
            statements.append("return;")
            returned = True
        elif ins.code == Code.EMMS:
            mmx_active = False
            defined_mm.clear()
            statements.append("/* emms */")
        elif ins.code in CVT_PD2PI_HELPERS:
            src = vec_src128(1, ins, alignment=16)
            dst_idx, dst_name = write_mm(0, ins)
            statements.append(f"{dst_name} = {CVT_PD2PI_HELPERS[ins.code]}({src});")
            defined_mm.add(dst_idx)
        elif ins.code == Code.CVTPI2PD_XMM_MMM64:
            src = mm_src64(1, ins)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = op_cvtpi2pd({src});")
            defined_xmm.add(dst_idx)
        elif ins.code == Code.MOVDQ2Q_MM_XMM:
            src = read_vec(1, ins)
            dst_idx, dst_name = write_mm(0, ins)
            statements.append(f"{dst_name} = {src}.q[0];")
            defined_mm.add(dst_idx)
        elif ins.code == Code.MOVQ2DQ_XMM_MM:
            src = read_mm(1, ins)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = {{{{{src}, 0ull}}}};")
            defined_xmm.add(dst_idx)
        elif ins.code in (Code.MOVDQA_XMM_XMMM128, Code.MOVDQU_XMM_XMMM128):
            alignment = 16 if ins.code == Code.MOVDQA_XMM_XMMM128 else 1
            src = vec_src128(1, ins, alignment=alignment)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = {src};")
            defined_xmm.add(dst_idx)
        elif ins.code in (Code.MOVDQA_XMMM128_XMM, Code.MOVDQU_XMMM128_XMM):
            alignment = 16 if ins.code == Code.MOVDQA_XMMM128_XMM else 1
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
