"""Lift bounded Windows x64 leaf functions using Intel SHA-1 NI instructions to C++.

Each export uses the straight-line leaf ABI:
    void lifted_<name>(word rcx, word rdx, word r8)
where rcx, rdx, and r8 carry flat host pointers to caller-owned 16-byte buffers
(rcx and rdx are inputs, r8 is the output/scratch). XMM0..XMM5 are modeled as
portable 128-bit value structs (four 32-bit lanes, pure portable C++ without
<immintrin.h>). Only legacy encodings are accepted; VEX, EVEX, and other
non-legacy encodings are rejected.

Supported SHA-1 NI instructions (docs/coverage/x64_zen2.tsv group Crypto):
  - SHA1RNDS4 xmm1, xmm2/m128, imm8  (NP 0F 3A CC /r ib): four SHA-1 rounds;
    imm8[1:0] selects f0 Ch + 0x5A827999, f1 Parity + 0x6ED9EBA1,
    f2 Maj + 0x8F1BBCDC, f3 Parity + 0xCA62C1D6 (imm8[7:2] ignored)
  - SHA1NEXTE xmm1, xmm2/m128        (NP 0F 38 C8 /r): add rol32(dst[3], 30)
    to src[3] and copy src[2:0]
  - SHA1MSG1  xmm1, xmm2/m128        (NP 0F 38 C9 /r): first SHA-1 message
    schedule step
  - SHA1MSG2  xmm1, xmm2/m128        (NP 0F 38 CA /r): second SHA-1 message
    schedule step

Transport / ABI glue:
  - movdqu (128-bit flat [rcx|rdx|r8] load, [r8] store, reg-reg copy)
  - movdqa (16-byte-aligned flat [rcx|rdx|r8] load, [r8] store, reg-reg copy)
  - terminal ret

SHA-256 instructions, VEX/EVEX encodings, nonvolatile XMM6..XMM15, undefined
XMM reads, unaligned movdqa, non-flat/displaced/indexed/out-of-buffer memory,
stores outside [r8], instruction prefixes, branches, missing ret, and trailing
instructions after ret are rejected.
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

VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = (
    {getattr(R, f"XMM{i}") for i in range(6, 32)}
    | {getattr(R, f"YMM{i}") for i in range(32)}
    | {getattr(R, f"ZMM{i}") for i in range(32)}
)
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

SHA1_BINARY_CODES = {
    Code.SHA1NEXTE_XMM_XMMM128: "op_sha1nexte",
    Code.SHA1MSG1_XMM_XMMM128: "op_sha1msg1",
    Code.SHA1MSG2_XMM_XMMM128: "op_sha1msg2",
}

SHA256_REJECT_CODES = {
    Code.SHA256RNDS2_XMM_XMMM128,
    Code.SHA256MSG1_XMM_XMMM128,
    Code.SHA256MSG2_XMM_XMMM128,
}

LOAD_CODES = {
    Code.MOVDQU_XMM_XMMM128: False,
    Code.MOVDQA_XMM_XMMM128: True,
}

STORE_CODES = {
    Code.MOVDQU_XMMM128_XMM: False,
    Code.MOVDQA_XMMM128_XMM: True,
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.SHA1RNDS4_XMM_XMMM128_IMM8,
    *SHA1_BINARY_CODES,
    *LOAD_CODES,
    *STORE_CODES,
}


PRELUDE = """#include <atomic>
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_MSC_VER)
extern "C" __declspec(dllexport) std::uint32_t host_has_sha() {
    int info[4]{};
    __cpuid(info, 0);
    if (info[0] < 7) return 0u;
    __cpuidex(info, 7, 0);
    return (static_cast<std::uint32_t>(info[1]) & (1u << 29)) != 0u ? 1u : 0u;
}
#endif

using word = std::uint64_t;

struct Vec128 {
    std::uint32_t w[4];
};

// Guest reads can fault even when the translated result is dead. Keep surrounding guest
// writes on their original side of the read when the compiler inlines these helpers.
static inline Vec128 load128(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    Vec128 out{};
    std::memcpy(out.w, reinterpret_cast<const void*>(address), sizeof(out.w));
    const volatile Vec128 observed = out;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{observed.w[0], observed.w[1], observed.w[2], observed.w[3]}};
}

static inline void store128(word address, const Vec128& v) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), v.w, sizeof(v.w));
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline std::uint32_t rol32(std::uint32_t x, unsigned n) {
    return (x << n) | (x >> (32u - n));
}

static inline Vec128 op_sha1rnds4(const Vec128& dst, const Vec128& src, unsigned imm8) {
    std::uint32_t a = dst.w[3];
    std::uint32_t b = dst.w[2];
    std::uint32_t c = dst.w[1];
    std::uint32_t d = dst.w[0];
    std::uint32_t e = 0u;
    const std::uint32_t w[4] = {src.w[3], src.w[2], src.w[1], src.w[0]};
    const unsigned sel = imm8 & 3u;
    const std::uint32_t k = (sel == 0u) ? 0x5a827999u
                          : (sel == 1u) ? 0x6ed9eba1u
                          : (sel == 2u) ? 0x8f1bbcdcu
                                        : 0xca62c1d6u;
    for (unsigned i = 0u; i < 4u; ++i) {
        const std::uint32_t f = (sel == 0u) ? ((b & c) ^ (~b & d))
                              : (sel == 2u) ? ((b & c) ^ (b & d) ^ (c & d))
                                            : (b ^ c ^ d);
        const std::uint32_t next_a = f + rol32(a, 5u) + w[i] + k + e;
        e = d;
        d = c;
        c = rol32(b, 30u);
        b = a;
        a = next_a;
    }
    return {{d, c, b, a}};
}

static inline Vec128 op_sha1nexte(const Vec128& dst, const Vec128& src) {
    return {{
        src.w[0],
        src.w[1],
        src.w[2],
        src.w[3] + rol32(dst.w[3], 30u),
    }};
}

static inline Vec128 op_sha1msg1(const Vec128& dst, const Vec128& src) {
    return {{
        dst.w[0] ^ src.w[2],
        dst.w[1] ^ src.w[3],
        dst.w[2] ^ dst.w[0],
        dst.w[3] ^ dst.w[1],
    }};
}

static inline Vec128 op_sha1msg2(const Vec128& dst, const Vec128& src) {
    const std::uint32_t w16 = rol32(dst.w[3] ^ src.w[2], 1u);
    const std::uint32_t w17 = rol32(dst.w[2] ^ src.w[1], 1u);
    const std::uint32_t w18 = rol32(dst.w[1] ^ src.w[0], 1u);
    const std::uint32_t w19 = rol32(dst.w[0] ^ w16, 1u);
    return {{w19, w18, w17, w16}};
}
"""


def vec_index(register, ins):
    """Map volatile XMM0..XMM5 to an index, rejecting nonvolatile and unsupported registers."""
    if register in VOLATILE_XMM:
        return VOLATILE_XMM[register]
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, *, aligned=False, store=False):
    """Lower a flat zero-displacement 16-byte [rcx|rdx|r8] memory operand."""
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
    if MemorySizeExt.size(ins.memory_size) != 16:
        raise ValueError(f"unsupported memory width at RVA {ins.ip:#x}: {ins}")
    if store and base != R.R8:
        raise ValueError(f"store outside output buffer r8 at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    if aligned and disp % 16 != 0:
        raise ValueError(f"unaligned 16-byte memory operand at RVA {ins.ip:#x}: {ins}")
    if disp != 0:
        raise ValueError(f"unsupported displaced/out-of-buffer memory operand at RVA {ins.ip:#x}: {ins}")
    return ARGUMENTS[base]


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line SHA-1 NI leaf function."""
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

    def vec_src128(op_index, ins, *, aligned=False):
        if ins.op_kind(op_index) == OpKind.MEMORY:
            return f"load128({flat_address(ins, aligned=aligned, store=False)})"
        return read_vec(op_index, ins)

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
        if ins.code in SHA256_REJECT_CODES:
            raise ValueError(f"unsupported SHA-256 instruction at RVA {ins.ip:#x}: {ins}")
        if ins.flow_control not in (FlowControl.NEXT, FlowControl.RETURN):
            raise ValueError(f"unsupported control-flow instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            statements.append("return;")
            returned = True

        elif ins.code == Code.SHA1RNDS4_XMM_XMMM128_IMM8:
            dst = read_vec(0, ins)
            src = vec_src128(1, ins, aligned=True)
            dst_idx, dst_name = write_vec(0, ins)
            imm8 = ins.immediate8 & 0xFF
            statements.append(f"{dst_name} = op_sha1rnds4({dst}, {src}, {imm8:#x}u);")
            defined.add(dst_idx)

        elif ins.code in SHA1_BINARY_CODES:
            dst = read_vec(0, ins)
            src = vec_src128(1, ins, aligned=True)
            dst_idx, dst_name = write_vec(0, ins)
            helper = SHA1_BINARY_CODES[ins.code]
            statements.append(f"{dst_name} = {helper}({dst}, {src});")
            defined.add(dst_idx)

        elif ins.code in LOAD_CODES:
            aligned = LOAD_CODES[ins.code]
            src = vec_src128(1, ins, aligned=aligned)
            dst_idx, dst_name = write_vec(0, ins)
            statements.append(f"{dst_name} = {src};")
            defined.add(dst_idx)

        elif ins.code in STORE_CODES:
            aligned = STORE_CODES[ins.code]
            src = read_vec(1, ins)
            if ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins, aligned=aligned, store=True)
                statements.append(f"store128({addr}, {src});")
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
