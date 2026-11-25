"""Lift bounded Windows x64 leaf functions using AVX VPTEST, VTESTPS, and VTESTPD to C++.

Each export uses the straight-line leaf ABI:
    std::uint64_t lifted_<name>(word rcx, word rdx, word r8)
where rcx, rdx, and r8 carry flat host pointers to caller-owned buffers (rcx/rdx
inputs, r8 output/scratch). YMM0..YMM5 are modeled as portable 256-bit value
structs (four 64-bit lanes); every VEX.128 write to XMM0..XMM5 zeroes [255:128].

Supported AVX test encodings (docs/coverage/x64_zen2.tsv group AVX):
  - VPTEST  xmm, xmm/m128  |  ymm, ymm/m256  (full 128/256 bits: ZF=(a&b)==0, CF=(~a&b)==0)
  - VTESTPS xmm, xmm/m128  |  ymm, ymm/m256  (bit 31 per f32 lane only)
  - VTESTPD xmm, xmm/m128  |  ymm, ymm/m256  (bit 63 per f64 lane only)

Transport / flag-packing glue:
  - vmovdqu (128/256-bit flat [rcx|rdx|r8] loads, [r8] stores, reg-reg copies)
  - vzeroupper
  - Volatile scratch GPR (rax/r9/r10/r11) mov, or, xor, shl, movzx, setb, sete
    packing (CF << 0) | (ZF << 1) into rax, and terminal ret.

Legacy SSE PTEST, EVEX forms, nonvolatile XMM/YMM/GPR registers, writes to
argument GPRs (rcx/rdx/r8), undefined vector/GPR/flag reads, non-flat or
mis-sized memory operands, stores outside [r8], prefixes, branches, missing ret,
and code after ret are rejected.
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
ARGUMENT_GPRS = {R.RCX, R.ECX, R.CL, R.RDX, R.EDX, R.DL, R.R8, R.R8D, R.R8L}

VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
VOLATILE_YMM = {getattr(R, f"YMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = (
    {getattr(R, f"XMM{i}") for i in range(6, 32)}
    | {getattr(R, f"YMM{i}") for i in range(6, 32)}
    | {getattr(R, f"ZMM{i}") for i in range(32)}
)

SCRATCH_GPR64 = {R.RAX: "rax", R.R9: "r9", R.R10: "r10", R.R11: "r11"}
SCRATCH_GPR32 = {R.EAX: "rax", R.R9D: "r9", R.R10D: "r10", R.R11D: "r11"}
SCRATCH_GPR8 = {R.AL: "rax", R.R9L: "r9", R.R10L: "r10", R.R11L: "r11"}
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

# (helper_name, is_ymm)
TEST_CODES = {
    Code.VEX_VPTEST_XMM_XMMM128: ("op_vptest_128", False),
    Code.VEX_VPTEST_YMM_YMMM256: ("op_vptest_256", True),
    Code.VEX_VTESTPS_XMM_XMMM128: ("op_vtestps_128", False),
    Code.VEX_VTESTPS_YMM_YMMM256: ("op_vtestps_256", True),
    Code.VEX_VTESTPD_XMM_XMMM128: ("op_vtestpd_128", False),
    Code.VEX_VTESTPD_YMM_YMMM256: ("op_vtestpd_256", True),
}

VMOVDQU_CODES = {
    Code.VEX_VMOVDQU_XMM_XMMM128: False,
    Code.VEX_VMOVDQU_XMMM128_XMM: False,
    Code.VEX_VMOVDQU_YMM_YMMM256: True,
    Code.VEX_VMOVDQU_YMMM256_YMM: True,
}

MOV_REG_CODES = {
    Code.MOV_R32_RM32: False,
    Code.MOV_RM32_R32: False,
    Code.MOV_R64_RM64: True,
    Code.MOV_RM64_R64: True,
}
MOV_IMM_CODES = {
    Code.MOV_R32_IMM32: False,
    Code.MOV_RM32_IMM32: False,
    Code.MOV_R64_IMM64: True,
    Code.MOV_RM64_IMM32: True,
}
XOR_REG_CODES = {
    Code.XOR_R32_RM32: False,
    Code.XOR_RM32_R32: False,
    Code.XOR_R64_RM64: True,
    Code.XOR_RM64_R64: True,
}
OR_REG_CODES = {
    Code.OR_R32_RM32: False,
    Code.OR_RM32_R32: False,
    Code.OR_R64_RM64: True,
    Code.OR_RM64_R64: True,
}
SHL_IMM_CODES = {
    Code.SHL_RM32_1: (False, True),
    Code.SHL_RM32_IMM8: (False, False),
    Code.SHL_RM64_1: (True, True),
    Code.SHL_RM64_IMM8: (True, False),
}
MOVZX_BYTE_CODES = {
    Code.MOVZX_R32_RM8: False,
    Code.MOVZX_R64_RM8: True,
}
SETCC_CODES = {
    Code.SETB_RM8: "flag_cf",
    Code.SETE_RM8: "flag_zf",
}

LEGACY_SSE_REJECT_CODES = {
    Code.PTEST_XMM_XMMM128,
    Code.MOVDQU_XMM_XMMM128,
    Code.MOVDQU_XMMM128_XMM,
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.VEX_VZEROUPPER,
    *TEST_CODES,
    *VMOVDQU_CODES,
    *MOV_REG_CODES,
    *MOV_IMM_CODES,
    *XOR_REG_CODES,
    *OR_REG_CODES,
    *SHL_IMM_CODES,
    *MOVZX_BYTE_CODES,
    *SETCC_CODES,
}


PRELUDE = """#include <atomic>
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_MSC_VER)
extern "C" __declspec(dllexport) std::uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}
#endif

using word = std::uint64_t;

struct Vec256 {
    std::uint64_t q[4];
};

struct TestFlags {
    bool cf;
    bool zf;
};

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

static inline void store128(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
}

static inline void store256(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 32);
}

static inline Vec256 copy128(const Vec256& v) {
    return {{v.q[0], v.q[1], 0ull, 0ull}};
}

static inline TestFlags test_masked(const Vec256& a, const Vec256& b, unsigned lanes, std::uint64_t mask) {
    std::uint64_t z_acc = 0ull;
    std::uint64_t c_acc = 0ull;
    for (unsigned i = 0; i < lanes; ++i) {
        const std::uint64_t am = a.q[i] & mask;
        const std::uint64_t bm = b.q[i] & mask;
        z_acc |= (am & bm);
        c_acc |= ((~am) & bm);
    }
    return {c_acc == 0ull, z_acc == 0ull};
}

static inline TestFlags op_vptest_128(const Vec256& a, const Vec256& b) {
    return test_masked(a, b, 2u, ~0ull);
}

static inline TestFlags op_vptest_256(const Vec256& a, const Vec256& b) {
    return test_masked(a, b, 4u, ~0ull);
}

static inline TestFlags op_vtestps_128(const Vec256& a, const Vec256& b) {
    return test_masked(a, b, 2u, 0x8000000080000000ull);
}

static inline TestFlags op_vtestps_256(const Vec256& a, const Vec256& b) {
    return test_masked(a, b, 4u, 0x8000000080000000ull);
}

static inline TestFlags op_vtestpd_128(const Vec256& a, const Vec256& b) {
    return test_masked(a, b, 2u, 0x8000000000000000ull);
}

static inline TestFlags op_vtestpd_256(const Vec256& a, const Vec256& b) {
    return test_masked(a, b, 4u, 0x8000000000000000ull);
}
"""


def vec_index(register, ins):
    """Map volatile XMM0..XMM5 or YMM0..YMM5 to (index, is_ymm)."""
    if register in VOLATILE_XMM:
        return VOLATILE_XMM[register], False
    if register in VOLATILE_YMM:
        return VOLATILE_YMM[register], True
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def scratch_gpr(register, ins, *, width, write=False):
    """Map a volatile scratch GPR of the requested width (1, 4, or 8 bytes) to its canonical 64-bit name."""
    if register in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {register} at RVA {ins.ip:#x}: {ins}")
    if register in ARGUMENT_GPRS:
        action = "write to" if write else "read of"
        raise ValueError(f"unsupported {action} argument GPR {register} at RVA {ins.ip:#x}: {ins}")
    table = {1: SCRATCH_GPR8, 4: SCRATCH_GPR32, 8: SCRATCH_GPR64}[width]
    if register not in table:
        raise ValueError(f"unsupported GPR {register} at RVA {ins.ip:#x}: {ins}")
    return table[register]


def flat_address(ins, *, expected_bytes, store=False):
    """Lower a flat [rcx|rdx|r8] (zero-displacement, no index, exact width) memory operand."""
    base = ins.memory_base
    if (ins.is_ip_rel_memory_operand or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
            or ins.memory_index != R.NONE or ins.memory_displacement != 0
            or not RegisterExt.is_gpr64(base) or base not in ARGUMENTS
            or MemorySizeExt.size(ins.memory_size) != expected_bytes):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    if store and base != R.R8:
        raise ValueError(f"store outside output buffer r8 at RVA {ins.ip:#x}: {ins}")
    return ARGUMENTS[base]


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line AVX test-flag leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used_vec = set()
    defined_vec = set()
    used_gpr = set()
    defined_gpr = set()
    defined_byte = set()
    flags_defined = False
    flags_used = False
    statements = []
    returned = False

    def read_vec(op_index, ins, *, expect_ymm):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected vector register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if wide != expect_ymm:
            raise ValueError(f"mismatched vector width at RVA {ins.ip:#x}: {ins}")
        if idx not in defined_vec:
            raise ValueError(f"undefined vector register ymm{idx} at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return f"ymm{idx}"

    def write_vec(op_index, ins, *, expect_ymm):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected vector register destination at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if wide != expect_ymm:
            raise ValueError(f"mismatched vector width at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return idx, f"ymm{idx}"

    def read_gpr(op_index, ins, *, width):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        reg_name = scratch_gpr(ins.op_register(op_index), ins, width=width, write=False)
        required = defined_byte if width == 1 else defined_gpr
        if reg_name not in required:
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        used_gpr.add(reg_name)
        return reg_name

    def write_gpr(op_index, ins, *, width):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR register destination at RVA {ins.ip:#x}: {ins}")
        reg_name = scratch_gpr(ins.op_register(op_index), ins, width=width, write=True)
        used_gpr.add(reg_name)
        return reg_name

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        op_code = ins.op_code()
        if (op_code.encoding == EncodingKind.EVEX or ins.op_mask != R.NONE
                or ins.zeroing_masking or ins.is_broadcast):
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code in LEGACY_SSE_REJECT_CODES:
            raise ValueError(f"unsupported legacy SSE encoding at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            if "rax" not in defined_gpr:
                raise ValueError(f"undefined GPR rax at ret RVA {ins.ip:#x}")
            used_gpr.add("rax")
            statements.append("return rax;")
            returned = True

        elif ins.code == Code.VEX_VZEROUPPER:
            for idx in sorted(defined_vec):
                statements.append(f"ymm{idx}.q[2] = 0ull; ymm{idx}.q[3] = 0ull;")

        elif ins.code in VMOVDQU_CODES:
            wide = VMOVDQU_CODES[ins.code]
            expected_bytes = 32 if wide else 16
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst_idx, dst = write_vec(0, ins, expect_ymm=wide)
                addr = flat_address(ins, expected_bytes=expected_bytes, store=False)
                statements.append(f"{dst} = {'load256' if wide else 'load128'}({addr});")
                defined_vec.add(dst_idx)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src = read_vec(1, ins, expect_ymm=wide)
                addr = flat_address(ins, expected_bytes=expected_bytes, store=True)
                statements.append(f"{'store256' if wide else 'store128'}({addr}, {src});")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src = read_vec(1, ins, expect_ymm=wide)
                dst_idx, dst = write_vec(0, ins, expect_ymm=wide)
                statements.append(f"{dst} = {src};" if wide else f"{dst} = copy128({src});")
                defined_vec.add(dst_idx)
            else:
                raise ValueError(f"unsupported vmovdqu operands at RVA {ins.ip:#x}: {ins}")

        elif ins.code in TEST_CODES:
            helper, wide = TEST_CODES[ins.code]
            src0 = read_vec(0, ins, expect_ymm=wide)
            if ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, expected_bytes=32 if wide else 16, store=False)
                src1 = f"{'load256' if wide else 'load128'}({addr})"
            else:
                src1 = read_vec(1, ins, expect_ymm=wide)
            statements.append(f"const TestFlags tf_{ins.ip:x} = {helper}({src0}, {src1});")
            statements.append(f"flag_cf = tf_{ins.ip:x}.cf; flag_zf = tf_{ins.ip:x}.zf;")
            flags_defined = True
            flags_used = True

        elif ins.code in SETCC_CODES:
            if not flags_defined:
                raise ValueError(f"flag read without preceding vptest/vtestps/vtestpd at RVA {ins.ip:#x}: {ins}")
            dst = write_gpr(0, ins, width=1)
            flag_var = SETCC_CODES[ins.code]
            statements.append(f"{dst} = ({dst} & ~0xffull) | ({flag_var} ? 1ull : 0ull);")
            defined_byte.add(dst)

        elif ins.code in MOVZX_BYTE_CODES:
            is64 = MOVZX_BYTE_CODES[ins.code]
            src = read_gpr(1, ins, width=1)
            dst = write_gpr(0, ins, width=8 if is64 else 4)
            statements.append(f"{dst} = ({src} & 0xffull);")
            defined_gpr.add(dst)
            defined_byte.add(dst)

        elif ins.code in MOV_REG_CODES:
            is64 = MOV_REG_CODES[ins.code]
            width = 8 if is64 else 4
            src = read_gpr(1, ins, width=width)
            dst = write_gpr(0, ins, width=width)
            if is64:
                statements.append(f"{dst} = {src};")
            else:
                statements.append(f"{dst} = static_cast<word>(static_cast<std::uint32_t>({src}));")
            defined_gpr.add(dst)
            defined_byte.add(dst)

        elif ins.code in MOV_IMM_CODES:
            is64 = MOV_IMM_CODES[ins.code]
            dst = write_gpr(0, ins, width=8 if is64 else 4)
            if ins.code == Code.MOV_R64_IMM64:
                imm = ins.immediate64 & 0xFFFFFFFFFFFFFFFF
            elif ins.code == Code.MOV_RM64_IMM32:
                imm = ins.immediate32to64 & 0xFFFFFFFFFFFFFFFF
            else:
                imm = ins.immediate32 & 0xFFFFFFFF
            statements.append(f"{dst} = {imm:#x}ull;")
            defined_gpr.add(dst)
            defined_byte.add(dst)

        elif ins.code in XOR_REG_CODES:
            is64 = XOR_REG_CODES[ins.code]
            width = 8 if is64 else 4
            if ins.op0_kind != OpKind.REGISTER or ins.op1_kind != OpKind.REGISTER:
                raise ValueError(f"expected register operands for xor at RVA {ins.ip:#x}: {ins}")
            if ins.op0_register == ins.op1_register:
                dst = write_gpr(0, ins, width=width)
                statements.append(f"{dst} = 0ull;")
            else:
                src = read_gpr(1, ins, width=width)
                dst = read_gpr(0, ins, width=width)
                write_gpr(0, ins, width=width)
                if is64:
                    statements.append(f"{dst} = {dst} ^ {src};")
                else:
                    statements.append(
                        f"{dst} = static_cast<word>(static_cast<std::uint32_t>({dst}) ^ static_cast<std::uint32_t>({src}));")
            defined_gpr.add(dst)
            defined_byte.add(dst)
            flags_defined = False

        elif ins.code in OR_REG_CODES:
            is64 = OR_REG_CODES[ins.code]
            width = 8 if is64 else 4
            src = read_gpr(1, ins, width=width)
            dst = read_gpr(0, ins, width=width)
            write_gpr(0, ins, width=width)
            if is64:
                statements.append(f"{dst} = {dst} | {src};")
            else:
                statements.append(
                    f"{dst} = static_cast<word>(static_cast<std::uint32_t>({dst}) | static_cast<std::uint32_t>({src}));")
            defined_gpr.add(dst)
            defined_byte.add(dst)
            flags_defined = False

        elif ins.code in SHL_IMM_CODES:
            is64, is_one = SHL_IMM_CODES[ins.code]
            width = 8 if is64 else 4
            dst = read_gpr(0, ins, width=width)
            write_gpr(0, ins, width=width)
            count = 1 if is_one else (ins.immediate8 & (63 if is64 else 31))
            if is64:
                statements.append(f"{dst} = {dst} << {count}u;")
            else:
                statements.append(f"{dst} = static_cast<word>(static_cast<std::uint32_t>({dst}) << {count}u);")
            defined_gpr.add(dst)
            defined_byte.add(dst)
            flags_defined = False

        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec256 ymm{idx}{{}};" for idx in sorted(used_vec)]
    decls += [f"    word {gpr} = 0ull;" for gpr in sorted(used_gpr)]
    if flags_used:
        decls.append("    bool flag_cf = false, flag_zf = false;")
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
