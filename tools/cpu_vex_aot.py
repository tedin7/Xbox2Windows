"""Recompile Windows x64 leaf functions that use VEX.128 packed-integer word and byte forms.

Profile: RCX and RDX hold the addresses of two 16-byte inputs and R8 the address of a 16-byte
output; there is no return value. Memory operands are exactly [RCX], [RDX] or [R8] with a 16-byte
access, loads from any of them and stores only to [R8]. Each XMM register becomes a 16-byte value
and each form an explicit per-lane loop, so the generated C++ spells out wraparound, saturation,
shift-count and rounding rules instead of relying on the host compiler's intrinsics. Any other
instruction, a VEX.256 (YMM) operand or another operand shape is rejected with its RVA.

Encoding: only VEX instructions are accepted. An EVEX form such as vpaddw xmm0{k1} has the same
mnemonic but adds masking, so it is rejected rather than lifted without its mask. Registers: only
the volatile XMM0-XMM5 may be read or written; XMM6-XMM15 are nonvolatile in the Windows x64 ABI.

Alignment: vmovdqu accesses any address. vmovdqa requires 16-byte-aligned buffers, as on the CPU;
the generated C++ does not check this, so callers must meet the precondition the native code
would fault on. Aliases: the output may be one of the inputs exactly, and the two inputs may be the
same buffer. Every memory read and the store happen in program order, each as one 16-byte copy, so
a read after a store to an aliased buffer sees the stored value, as on the CPU. Partial overlaps
between buffers are not covered by the tests.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import (Decoder, EncodingKind, Instruction, MemorySizeExt, Mnemonic as M, OpKind as K,
                      Register as R)

from cpu_leaf_aot import bounded_code, export_rva

# Only the volatile registers: a leaf that wrote XMM6-XMM15 would have to save them first.
XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
NONVOLATILE = {getattr(R, f"XMM{i}") for i in range(6, 32)}
YMM = {getattr(R, f"YMM{i}") for i in range(16)}
POINTERS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}

# Three-operand forms: name of the C++ helper that combines two 16-byte values.
UNARY = {
    M.VPABSW: "pabsw",
}
BINARY = {
    M.VPADDW: "paddw", M.VPSUBW: "psubw", M.VPMAXSW: "pmaxsw", M.VPMINSW: "pminsw",
    M.VPMULHW: "pmulhw", M.VPAVGW: "pavgw", M.VPXOR: "pxor", M.VPAND: "pand", M.VPOR: "por",
    M.VPUNPCKLBW: "punpcklbw", M.VPUNPCKHBW: "punpckhbw", M.VPUNPCKLWD: "punpcklwd",
    M.VPUNPCKHWD: "punpckhwd", M.VPACKUSWB: "packuswb",
    M.VPADDSW: "paddsw", M.VPADDUSW: "paddusw", M.VPSUBSW: "psubsw", M.VPSUBUSW: "psubusw",
    M.VPMULLW: "pmullw", M.VPMULHUW: "pmulhuw", M.VPMAXUW: "pmaxuw", M.VPMINUW: "pminuw",
    M.VPCMPEQW: "pcmpeqw", M.VPCMPGTW: "pcmpgtw",
    M.VPSIGNW: "psignw", M.VPMULHRSW: "pmulhrsw", M.VPHADDW: "phaddw",
    M.VPANDN: "pandn", M.VPAVGB: "pavgb", M.VPMADDUBSW: "pmaddubsw", M.VPSHUFB: "pshufb",
    M.VPADDB: "paddb", M.VPSUBB: "psubb", M.VPCMPEQB: "pcmpeqb", M.VPCMPGTB: "pcmpgtb",
}
SHIFTS = {M.VPSRAW: "psraw", M.VPSLLW: "psllw", M.VPSRLW: "psrlw"}
MOVES = {M.VMOVDQU, M.VMOVDQA}

PRELUDE = r"""#include <cstdint>
#include <cstring>
#include <intrin.h>

namespace {
struct V { std::uint8_t b[16]; };

std::int16_t word(const V& v, int i) { std::int16_t w; std::memcpy(&w, v.b + 2 * i, 2); return w; }
void set_word(V& v, int i, std::int32_t w) {
    const auto u = static_cast<std::uint16_t>(w);  // wraps modulo 2^16
    std::memcpy(v.b + 2 * i, &u, 2);
}
std::uint16_t uword(const V& v, int i) { return static_cast<std::uint16_t>(word(v, i)); }

V load(const std::uint8_t* p) { V v; std::memcpy(v.b, p, 16); return v; }
void store(std::uint8_t* p, const V& v) { std::memcpy(p, v.b, 16); }

#define WORDWISE(name, expr) \
    V name(const V& a, const V& b) { V r{}; for (int i = 0; i < 8; ++i) set_word(r, i, (expr)); return r; }
WORDWISE(paddw, word(a, i) + word(b, i))
WORDWISE(psubw, word(a, i) - word(b, i))
WORDWISE(pmaxsw, word(a, i) > word(b, i) ? word(a, i) : word(b, i))
WORDWISE(pminsw, word(a, i) < word(b, i) ? word(a, i) : word(b, i))
// The high 16 bits of the signed 32-bit product.
WORDWISE(pmulhw, (static_cast<std::int32_t>(word(a, i)) * word(b, i)) >> 16)
// Unsigned average, rounded up: (a + b + 1) >> 1 without overflow.
WORDWISE(pavgw, (static_cast<std::uint32_t>(uword(a, i)) + uword(b, i) + 1) >> 1)
// Saturating add and subtract: clamp the exact result to the signed or unsigned word range.
std::int32_t sat_s16(std::int32_t x) { return x < -32768 ? -32768 : x > 32767 ? 32767 : x; }
std::int32_t sat_u16(std::int32_t x) { return x < 0 ? 0 : x > 65535 ? 65535 : x; }
WORDWISE(paddsw, sat_s16(word(a, i) + word(b, i)))
WORDWISE(psubsw, sat_s16(word(a, i) - word(b, i)))
WORDWISE(paddusw, sat_u16(static_cast<std::int32_t>(uword(a, i)) + uword(b, i)))
WORDWISE(psubusw, sat_u16(static_cast<std::int32_t>(uword(a, i)) - uword(b, i)))
// The low 16 bits of the product are the same for signed and unsigned inputs; the high 16 bits
// of the unsigned 32-bit product need an unsigned multiply.
WORDWISE(pmullw, static_cast<std::int32_t>(word(a, i)) * word(b, i))
WORDWISE(pmulhuw, (static_cast<std::uint32_t>(uword(a, i)) * uword(b, i)) >> 16)
// Unary packed signed word abs: INT16_MIN (0x8000) wraps to 0x8000 without signed negation UB.
V pabsw(const V& a) {
    V r{};
    for (int i = 0; i < 8; ++i) {
        const std::int16_t w = word(a, i);
        const std::uint16_t u = static_cast<std::uint16_t>(w);
        set_word(r, i, w < 0 ? static_cast<std::int32_t>(static_cast<std::uint16_t>(0u - u)) : w);
    }
    return r;
}
// Sign mask, rounding multiply-high ((a*b + 0x4000) >> 15) and pairwise horizontal word add.
WORDWISE(psignw, word(b, i) < 0 ? static_cast<std::int32_t>(static_cast<std::uint16_t>(0u - uword(a, i)))
                                : word(b, i) == 0 ? 0 : word(a, i))
WORDWISE(pmulhrsw, (static_cast<std::int32_t>(word(a, i)) * word(b, i) + 0x4000) >> 15)
V phaddw(const V& a, const V& b) {
    V r{};
    for (int i = 0; i < 4; ++i) {
        set_word(r, i, word(a, 2 * i) + word(a, 2 * i + 1));
        set_word(r, 4 + i, word(b, 2 * i) + word(b, 2 * i + 1));
    }
    return r;
}
// Unsigned word maximum and minimum.
WORDWISE(pmaxuw, uword(a, i) > uword(b, i) ? uword(a, i) : uword(b, i))
WORDWISE(pminuw, uword(a, i) < uword(b, i) ? uword(a, i) : uword(b, i))
// Compares set a lane to all ones or all zeros; pcmpgtw compares as signed words.
WORDWISE(pcmpeqw, word(a, i) == word(b, i) ? -1 : 0)
WORDWISE(pcmpgtw, word(a, i) > word(b, i) ? -1 : 0)
#undef WORDWISE

V pxor(const V& a, const V& b) { V r; for (int i = 0; i < 16; ++i) r.b[i] = a.b[i] ^ b.b[i]; return r; }
V pand(const V& a, const V& b) { V r; for (int i = 0; i < 16; ++i) r.b[i] = a.b[i] & b.b[i]; return r; }
V pandn(const V& a, const V& b) { V r; for (int i = 0; i < 16; ++i) r.b[i] = static_cast<std::uint8_t>((~a.b[i]) & b.b[i]); return r; }
V por(const V& a, const V& b) { V r; for (int i = 0; i < 16; ++i) r.b[i] = a.b[i] | b.b[i]; return r; }
// Wraparound byte add/subtract and byte-lane equality / signed greater-than compares.
#define BYTEWISE(name, expr) \
    V name(const V& a, const V& b) { V r; for (int i = 0; i < 16; ++i) r.b[i] = static_cast<std::uint8_t>(expr); return r; }
BYTEWISE(paddb, a.b[i] + b.b[i])
BYTEWISE(psubb, a.b[i] - b.b[i])
BYTEWISE(pcmpeqb, a.b[i] == b.b[i] ? 0xFFu : 0u)
BYTEWISE(pcmpgtb, static_cast<std::int8_t>(a.b[i]) > static_cast<std::int8_t>(b.b[i]) ? 0xFFu : 0u)
#undef BYTEWISE
// Unsigned byte average rounded up: (a + b + 1) >> 1 without 8-bit overflow.
V pavgb(const V& a, const V& b) {
    V r;
    for (int i = 0; i < 16; ++i) r.b[i] = static_cast<std::uint8_t>((static_cast<unsigned>(a.b[i]) + b.b[i] + 1u) >> 1);
    return r;
}
// Unsigned byte * signed byte pairwise products added and saturated to signed 16-bit [-32768, 32767].
V pmaddubsw(const V& a, const V& b) {
    V r{};
    for (int i = 0; i < 8; ++i) {
        const std::int32_t lo = static_cast<std::int32_t>(a.b[2 * i]) * static_cast<std::int8_t>(b.b[2 * i]);
        const std::int32_t hi = static_cast<std::int32_t>(a.b[2 * i + 1]) * static_cast<std::int8_t>(b.b[2 * i + 1]);
        set_word(r, i, sat_s16(lo + hi));
    }
    return r;
}
// Byte shuffle: mask bit 7 zeroes the output byte, otherwise selects a[mask & 0x0f].
V pshufb(const V& a, const V& b) {
    V r;
    for (int i = 0; i < 16; ++i) r.b[i] = (b.b[i] & 0x80u) ? 0u : a.b[b.b[i] & 0x0Fu];
    return r;
}

// Interleave bytes or words from the low or high half of each source, first source first.
V punpcklbw(const V& a, const V& b) { V r; for (int i = 0; i < 8; ++i) { r.b[2*i] = a.b[i]; r.b[2*i+1] = b.b[i]; } return r; }
V punpckhbw(const V& a, const V& b) { V r; for (int i = 0; i < 8; ++i) { r.b[2*i] = a.b[8+i]; r.b[2*i+1] = b.b[8+i]; } return r; }
V punpcklwd(const V& a, const V& b) { V r{}; for (int i = 0; i < 4; ++i) { set_word(r, 2*i, word(a, i)); set_word(r, 2*i+1, word(b, i)); } return r; }
V punpckhwd(const V& a, const V& b) { V r{}; for (int i = 0; i < 4; ++i) { set_word(r, 2*i, word(a, 4+i)); set_word(r, 2*i+1, word(b, 4+i)); } return r; }

// Signed words to unsigned bytes with saturation: the first source fills bytes 0..7.
std::uint8_t sat_u8(std::int16_t w) { return w < 0 ? 0 : w > 255 ? 255 : static_cast<std::uint8_t>(w); }
V packuswb(const V& a, const V& b) { V r; for (int i = 0; i < 8; ++i) { r.b[i] = sat_u8(word(a, i)); r.b[8+i] = sat_u8(word(b, i)); } return r; }

// Shifts: a count above 15 fills arithmetic shifts with the sign and clears logical ones.
V psraw(const V& a, unsigned n) { V r{}; for (int i = 0; i < 8; ++i) set_word(r, i, word(a, i) >> (n > 15 ? 15 : n)); return r; }
V psllw(const V& a, unsigned n) { V r{}; for (int i = 0; i < 8; ++i) set_word(r, i, n > 15 ? 0 : uword(a, i) << n); return r; }
V psrlw(const V& a, unsigned n) { V r{}; for (int i = 0; i < 8; ++i) set_word(r, i, n > 15 ? 0 : uword(a, i) >> n); return r; }
// The count of the XMM or 16-byte memory form is the low 64 bits of the operand as one unsigned
// integer; the upper quadword is ignored. Any count above 15, however large, saturates to 16, so
// no bit of the quadword is lost to a narrower type.
unsigned shift_count(const V& c) {
    std::uint64_t q; std::memcpy(&q, c.b, 8);
    return q > 15 ? 16u : static_cast<unsigned>(q);
}
}  // namespace

extern "C" __declspec(dllexport) std::uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}
"""


def operand(instruction, index, rva):
    """C++ expression for operand `index`: an XMM register or a 16-byte load."""
    kind = instruction.op_kind(index)
    if kind == K.REGISTER:
        register = instruction.op_register(index)
        if register in YMM:
            raise ValueError(f"VEX.256 form at RVA {rva:#x}")
        if register in NONVOLATILE:
            raise ValueError(f"nonvolatile XMM register at RVA {rva:#x}")
        if register not in XMM:
            raise ValueError(f"unsupported register operand at RVA {rva:#x}")
        return f"xmm[{XMM[register]}]"
    if kind == K.MEMORY:
        return f"load({memory(instruction, rva)})"
    raise ValueError(f"unsupported operand at RVA {rva:#x}")


def memory(instruction, rva):
    if (instruction.memory_base not in POINTERS or instruction.memory_index != R.NONE
            or instruction.memory_displacement != 0 or instruction.segment_prefix != R.NONE
            or MemorySizeExt.size(instruction.memory_size) != 16):
        raise ValueError(f"memory operand outside the 16-byte buffers at RVA {rva:#x}")
    return POINTERS[instruction.memory_base]


def destination(instruction, rva):
    if instruction.op0_kind != K.REGISTER:
        raise ValueError(f"unsupported destination at RVA {rva:#x}")
    register = instruction.op0_register
    if register in YMM:
        raise ValueError(f"VEX.256 form at RVA {rva:#x}")
    if register in NONVOLATILE:
        raise ValueError(f"nonvolatile XMM register at RVA {rva:#x}")
    if register not in XMM:
        raise ValueError(f"unsupported destination at RVA {rva:#x}")
    return f"xmm[{XMM[register]}]"


def lift(start, code):
    """Return the C++ statements of one leaf function."""
    decoder = Decoder(64, code, ip=start)
    instruction = Instruction()
    lines = []
    returned = False
    defined = set()
    while decoder.can_decode:
        decoder.decode_out(instruction)
        rva = instruction.ip
        if returned:
            raise ValueError(f"code after ret at RVA {rva:#x}")
        op = instruction.mnemonic
        if instruction.is_invalid:
            raise ValueError(f"invalid instruction at RVA {rva:#x}")
        if op != M.RET and instruction.encoding != EncodingKind.VEX:
            raise ValueError(f"non-VEX encoding at RVA {rva:#x}")
        if op == M.RET and instruction.op_count == 0:
            returned = True
        elif op == M.VZEROUPPER:
            pass  # only the upper YMM halves change, and the profile never reads them
        elif op in MOVES and instruction.op_count == 2:
            if instruction.op0_kind == K.MEMORY:
                # Check the source first, so a VEX.256 store is reported as such rather than as a
                # 32-byte access.
                source = operand(instruction, 1, rva)
                if memory(instruction, rva) != "r8":
                    raise ValueError(f"store outside the output buffer at RVA {rva:#x}")
                lines.append(f"store(r8, {source});")
            else:
                lines.append(f"{destination(instruction, rva)} = {operand(instruction, 1, rva)};")
        elif op in UNARY and instruction.op_count == 2:
            lines.append(f"{destination(instruction, rva)} = {UNARY[op]}({operand(instruction, 1, rva)});")
        elif op in BINARY and instruction.op_count == 3:
            lines.append(f"{destination(instruction, rva)} = {BINARY[op]}("
                         f"{operand(instruction, 1, rva)}, {operand(instruction, 2, rva)});")
        elif op in SHIFTS and instruction.op_count == 3 and instruction.op2_kind == K.IMMEDIATE8:
            lines.append(f"{destination(instruction, rva)} = {SHIFTS[op]}("
                         f"{operand(instruction, 1, rva)}, {instruction.immediate8}u);")
        elif op in SHIFTS and instruction.op_count == 3 and instruction.op2_kind in (K.REGISTER, K.MEMORY):
            # The count is an XMM register or a 16-byte memory operand, read as one 16-byte value.
            lines.append(f"{destination(instruction, rva)} = {SHIFTS[op]}("
                         f"{operand(instruction, 1, rva)}, shift_count({operand(instruction, 2, rva)}));")
        else:
            raise ValueError(f"unsupported instruction {instruction} at RVA {rva:#x}")
        if op in MOVES or op in UNARY or op in BINARY or op in SHIFTS:
            zeroing = (op == M.VPXOR and instruction.op1_kind == K.REGISTER
                       and instruction.op2_kind == K.REGISTER
                       and instruction.op1_register == instruction.op2_register)
            if zeroing:
                lines[-1] = f"{destination(instruction, rva)} = V{{}};"
            else:
                for index in range(1, instruction.op_count):
                    if instruction.op_kind(index) == K.REGISTER and instruction.op_register(index) not in defined:
                        raise ValueError(f"undefined XMM source at RVA {rva:#x}")
            if instruction.op0_kind == K.REGISTER:
                defined.add(instruction.op0_register)
    if not returned:
        raise ValueError(f"missing ret at RVA {start + len(code):#x}")
    return lines


def translate(image, exports):
    parts = [PRELUDE]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in exports:
            if not name.isascii() or not name.isidentifier():
                raise ValueError(f"unsupported export name {name!r}")
            start = export_rva(pe, name)
            body = lift(start, bounded_code(pe, start))
            parts.append(f'extern "C" __declspec(dllexport) void {name}('
                         "const std::uint8_t* rcx, const std::uint8_t* rdx, std::uint8_t* r8) {")
            parts.append("    V xmm[16] = {};")
            parts.extend("    " + line for line in body)
            parts.append("}\n")
    return "\n".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("exports", nargs="+")
    args = parser.parse_args()
    try:
        args.output.write_text(translate(args.image, args.exports), encoding="utf-8")
    except (OSError, ValueError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
