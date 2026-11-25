"""Lift bounded Windows x64 leaf functions with AVX packed-integer instructions to C++.

Each export uses the leaf ABI void fn(word rcx, word rdx, word r8), where rcx,
rdx and r8 carry flat host pointers to caller-owned buffers. YMM0..YMM5 are
modeled as portable 256-bit value structs (four 64-bit lanes); every VEX.128
write to an XMM destination zeroes bits 128..255 of the corresponding YMM
register. Volatile scratch GPRs (rax, r9, r10, r11) are modeled as 64-bit words
for vpmovmskb, vpextr*, vpinsr* and flat load/store mov transport.

Supported AVX instructions (90 surviving forms from docs/coverage/x64_zen2.tsv group AVX,
excluding the 28 forms carved out to cpu_vex_aot.py (PR #325)):
  - State: vzeroall, vzeroupper
  - Abs: vpabsb, vpabsw, vpabsd
  - Pack: vpacksswb, vpackssdw, vpackusdw
  - Add / Sub: vpaddb/q, vpaddsb/usb/usw, vpsubb/d/q, vpsubsb/sw/usb/usw
  - Align / Blend: vpalignr, vpblendvb, vpblendw
  - Compare: vpcmpeqb/w/d/q, vpcmpgtb/w/d/q
  - Min / Max: vpmaxsb/sd/ub/uw/ud, vpminsb/sd/ub/uw/ud
  - Horizontal: vphaddw/d/sw, vphsubw/d/sw, vphminposuw
  - Multiply: vpmaddwd, vpmuldq, vpmuludq, vpmulhrsw, vpmulhuw, vpmulld
  - SAD: vpsadbw, vmpsadbw
  - Shuffles: vpshufhw, vpshuflw
  - Sign: vpsignb, vpsignw, vpsignd
  - Extend: vpmovsxbw/bd/bq/wd/wq/dq, vpmovzxbw/bd/bq/wd/wq/dq
  - Unpack: vpunpckldq/hdq/lqdq/hqdq
  - Shifts: vpslld/q, vpsrld/q (imm8 and xmm count)
  - Mask / Extract / Insert: vpmovmskb, vpextrb/w/d/q (reg/mem), vpinsrb/w/d/q (reg/mem)

  - 256-bit (VEX.L=1, AVX2, register operands only): vpaddb/w/d/q, vpsubb/w/d/q,
    vpcmpeqb/w/d/q, vpcmpgtb/w/d/q. Each 128-bit half is computed independently
    (no carry or compare crosses the lane boundary) and all 256 bits of the
    destination are written.
  - 256-bit (VEX.L=1, AVX2, register operands only) min / max: vpmaxsb/sw/sd/ub/uw/ud,
    vpminsb/sw/sd/ub/uw/ud. The 128-bit vpmaxsw and vpminsw encodings stay with cpu_vex_aot.py.
  - 256-bit (VEX.L=1, AVX2, register operands only) saturating add / subtract: vpaddsb/sw/usb/usw,
    vpsubsb/sw/usb/usw (destination = first source - second source). The 128-bit vpaddsw
    encoding stays with cpu_vex_aot.py.
  - 256-bit (VEX.L=1, AVX2, register operands only) horizontal add / subtract: vphaddw/d/sw,
    vphsubw/d/sw. Each 128-bit lane is computed independently.
  - 256-bit (VEX.L=1, AVX2, register operands only) pack with saturation: vpacksswb, vpackssdw,
    vpackusdw. Each 128-bit lane is computed independently (128-bit vpackuswb stays with
    cpu_vex_aot.py).

Transport/ABI glue: vmovdqu (128/256-bit flat load, store and register copy),
flat 32/64-bit GPR load/store mov for scratch GPRs, and terminal ret.

Every other VEX.L=1 packed-integer encoding, the 256-bit forms above with a
memory operand, memory operands
on register-only packed-integer instructions, the 28 carved-out packed-integer
forms owned by cpu_vex_aot.py (PR #325), undefined vector or GPR registers, nonvolatile
registers (XMM6..15 / YMM6..15, RBX/RBP/RSI/RDI/R12..R15/RSP), writes to
argument GPRs, EVEX encodings, instruction prefixes, branches and all other
instructions are rejected.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import Code, Decoder, EncodingKind, Mnemonic as M, OpKind, Register as R, RegisterExt

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}
ARGUMENT_GPRS = {R.RCX, R.ECX, R.RDX, R.EDX, R.R8, R.R8D}

VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
VOLATILE_YMM = {getattr(R, f"YMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = ({getattr(R, f"XMM{i}") for i in range(6, 16)}
                   | {getattr(R, f"YMM{i}") for i in range(6, 16)})

SCRATCH_GPR32 = {R.EAX: "rax", R.R9D: "r9", R.R10D: "r10", R.R11D: "r11"}
SCRATCH_GPR64 = {R.RAX: "rax", R.R9: "r9", R.R10: "r10", R.R11: "r11"}
NONVOLATILE_GPR = {
    R.RBX, R.EBX, R.RBP, R.EBP, R.RSI, R.ESI, R.RDI, R.EDI, R.RSP, R.ESP,
    R.R12, R.R12D, R.R13, R.R13D, R.R14, R.R14D, R.R15, R.R15D,
}

# 256-bit (VEX.L=1) register-only forms: mnemonic -> 128-bit helper applied to each half.
# vpaddw, vpaddd and vpsubw exist here only as 256-bit forms; their 128-bit encodings
# stay with cpu_vex_aot.py.
WIDE_BINARY_HELPERS = {
    getattr(M, name.upper()): f"op_{name}"
    for name in (
        "vpaddb", "vpaddw", "vpaddd", "vpaddq",
        "vpsubb", "vpsubw", "vpsubd", "vpsubq",
        "vpcmpeqb", "vpcmpeqw", "vpcmpeqd", "vpcmpeqq",
        "vpcmpgtb", "vpcmpgtw", "vpcmpgtd", "vpcmpgtq",
        "vpmaxsb", "vpmaxsw", "vpmaxsd", "vpmaxub", "vpmaxuw", "vpmaxud",
        "vpminsb", "vpminsw", "vpminsd", "vpminub", "vpminuw", "vpminud",
        "vpsignb", "vpsignw", "vpsignd",
        "vpaddsb", "vpaddsw", "vpaddusb", "vpaddusw",
        "vpsubsb", "vpsubsw", "vpsubusb", "vpsubusw",
        "vphaddw", "vphaddd", "vphaddsw", "vphsubw", "vphsubd", "vphsubsw",
        "vpmuldq", "vpmuludq", "vpmulhrsw", "vpsadbw",
        "vpacksswb", "vpackssdw", "vpackusdw",
    )
}
WIDE_BINARY_CODES = {
    getattr(Code, f"VEX_{name}_YMM_YMM_YMMM256")
    for name in (
        "VPADDB", "VPADDW", "VPADDD", "VPADDQ",
        "VPSUBB", "VPSUBW", "VPSUBD", "VPSUBQ",
        "VPCMPEQB", "VPCMPEQW", "VPCMPEQD", "VPCMPEQQ",
        "VPCMPGTB", "VPCMPGTW", "VPCMPGTD", "VPCMPGTQ",
        "VPMAXSB", "VPMAXSW", "VPMAXSD", "VPMAXUB", "VPMAXUW", "VPMAXUD",
        "VPMINSB", "VPMINSW", "VPMINSD", "VPMINUB", "VPMINUW", "VPMINUD",
        "VPSIGNB", "VPSIGNW", "VPSIGND",
        "VPADDSB", "VPADDSW", "VPADDUSB", "VPADDUSW",
        "VPSUBSB", "VPSUBSW", "VPSUBUSB", "VPSUBUSW",
        "VPHADDW", "VPHADDD", "VPHADDSW", "VPHSUBW", "VPHSUBD", "VPHSUBSW",
        "VPMULDQ", "VPMULUDQ", "VPMULHRSW", "VPSADBW",
        "VPACKSSWB", "VPACKSSDW", "VPACKUSDW",
    )
}
# 256-bit one-source forms: the 128-bit helper runs on each half. vpshufhw/vpshuflw use the
# same imm8 for both halves.
WIDE_UNARY_HELPERS = {
    getattr(M, name.upper()): f"op_{name}" for name in ("vpabsb", "vpabsw", "vpabsd")
}
WIDE_UNARY_IMM_HELPERS = {
    getattr(M, name.upper()): f"op_{name}" for name in ("vpshufhw", "vpshuflw")
}
WIDE_UNARY_CODES = {
    getattr(Code, f"VEX_{name}_YMM_YMMM256") for name in ("VPABSB", "VPABSW", "VPABSD")
}
WIDE_UNARY_IMM_CODES = {
    getattr(Code, f"VEX_{name}_YMM_YMMM256_IMM8") for name in ("VPSHUFHW", "VPSHUFLW")
}

# Unary: dst = helper(src)
UNARY_HELPERS = {
    M.VPABSB: "op_vpabsb",
    M.VPABSW: "op_vpabsw",
    M.VPABSD: "op_vpabsd",
    M.VPHMINPOSUW: "op_vphminposuw",
    M.VPMOVSXBW: "op_vpmovsxbw",
    M.VPMOVSXBD: "op_vpmovsxbd",
    M.VPMOVSXBQ: "op_vpmovsxbq",
    M.VPMOVSXWD: "op_vpmovsxwd",
    M.VPMOVSXWQ: "op_vpmovsxwq",
    M.VPMOVSXDQ: "op_vpmovsxdq",
    M.VPMOVZXBW: "op_vpmovzxbw",
    M.VPMOVZXBD: "op_vpmovzxbd",
    M.VPMOVZXBQ: "op_vpmovzxbq",
    M.VPMOVZXWD: "op_vpmovzxwd",
    M.VPMOVZXWQ: "op_vpmovzxwq",
    M.VPMOVZXDQ: "op_vpmovzxdq",
}

# Unary + imm8: dst = helper(src, imm8)
UNARY_IMM_HELPERS = {
    M.VPSHUFHW: "op_vpshufhw",
    M.VPSHUFLW: "op_vpshuflw",
}

# Binary: dst = helper(src1, src2)
BINARY_HELPERS = {
    M.VPACKSSWB: "op_vpacksswb",
    M.VPACKSSDW: "op_vpackssdw",
    M.VPACKUSDW: "op_vpackusdw",
    M.VPADDB: "op_vpaddb",
    M.VPADDQ: "op_vpaddq",
    M.VPADDSB: "op_vpaddsb",
    M.VPADDUSB: "op_vpaddusb",
    M.VPADDUSW: "op_vpaddusw",
    M.VPSUBB: "op_vpsubb",
    M.VPSUBD: "op_vpsubd",
    M.VPSUBQ: "op_vpsubq",
    M.VPSUBSB: "op_vpsubsb",
    M.VPSUBSW: "op_vpsubsw",
    M.VPSUBUSB: "op_vpsubusb",
    M.VPSUBUSW: "op_vpsubusw",
    M.VPCMPEQB: "op_vpcmpeqb",
    M.VPCMPEQW: "op_vpcmpeqw",
    M.VPCMPEQD: "op_vpcmpeqd",
    M.VPCMPEQQ: "op_vpcmpeqq",
    M.VPCMPGTB: "op_vpcmpgtb",
    M.VPCMPGTW: "op_vpcmpgtw",
    M.VPCMPGTD: "op_vpcmpgtd",
    M.VPCMPGTQ: "op_vpcmpgtq",
    M.VPMAXSB: "op_vpmaxsb",
    M.VPMAXSD: "op_vpmaxsd",
    M.VPMAXUB: "op_vpmaxub",
    M.VPMAXUW: "op_vpmaxuw",
    M.VPMAXUD: "op_vpmaxud",
    M.VPMINSB: "op_vpminsb",
    M.VPMINSD: "op_vpminsd",
    M.VPMINUB: "op_vpminub",
    M.VPMINUW: "op_vpminuw",
    M.VPMINUD: "op_vpminud",
    M.VPHADDW: "op_vphaddw",
    M.VPHADDD: "op_vphaddd",
    M.VPHADDSW: "op_vphaddsw",
    M.VPHSUBW: "op_vphsubw",
    M.VPHSUBD: "op_vphsubd",
    M.VPHSUBSW: "op_vphsubsw",
    M.VPMADDWD: "op_vpmaddwd",
    M.VPMULDQ: "op_vpmuldq",
    M.VPMULUDQ: "op_vpmuludq",
    M.VPMULHRSW: "op_vpmulhrsw",
    M.VPMULHUW: "op_vpmulhuw",
    M.VPMULLD: "op_vpmulld",
    M.VPSADBW: "op_vpsadbw",
    M.VPSIGNB: "op_vpsignb",
    M.VPSIGNW: "op_vpsignw",
    M.VPSIGND: "op_vpsignd",
    M.VPUNPCKLDQ: "op_vpunpckldq",
    M.VPUNPCKHDQ: "op_vpunpckhdq",
    M.VPUNPCKLQDQ: "op_vpunpcklqdq",
    M.VPUNPCKHQDQ: "op_vpunpckhqdq",
}

# Binary + imm8: dst = helper(src1, src2, imm8)
BINARY_IMM_HELPERS = {
    M.VPALIGNR: "op_vpalignr",
    M.VPBLENDW: "op_vpblendw",
    M.VMPSADBW: "op_vmpsadbw",
}

# Variable / imm8 shifts: dst = helper(src1, count_u64)
SHIFT_HELPERS = {
    M.VPSLLD: "op_vpslld",
    M.VPSLLQ: "op_vpsllq",
    M.VPSRLD: "op_vpsrld",
    M.VPSRLQ: "op_vpsrlq",
}

EXTRACT_SPECS = {
    M.VPEXTRB: ("op_vpextrb", "store8"),
    M.VPEXTRW: ("op_vpextrw", "store16"),
    M.VPEXTRD: ("op_vpextrd", "store32"),
    M.VPEXTRQ: ("op_vpextrq", "store64"),
}

INSERT_SPECS = {
    M.VPINSRB: ("op_vpinsrb", "load8", "static_cast<std::uint8_t>"),
    M.VPINSRW: ("op_vpinsrw", "load16", "static_cast<std::uint16_t>"),
    M.VPINSRD: ("op_vpinsrd", "load32", "static_cast<std::uint32_t>"),
    M.VPINSRQ: ("op_vpinsrq", "load64", ""),
}

# All 88 surviving packed-integer mnemonics whose AVX1 encodings are strictly VEX.128 (L=0).
AVX_INT_128_ONLY_MNEMONICS = (
    set(UNARY_HELPERS)
    | set(UNARY_IMM_HELPERS)
    | set(BINARY_HELPERS)
    | set(BINARY_IMM_HELPERS)
    | set(SHIFT_HELPERS)
    | {M.VPBLENDVB, M.VPMOVMSKB}
    | set(EXTRACT_SPECS)
    | set(INSERT_SPECS)
)

# Register-only packed-integer mnemonics (all except vpextr*/vpinsr* which allow flat memory operands).
REGISTER_ONLY_AVX_INT = AVX_INT_128_ONLY_MNEMONICS - set(EXTRACT_SPECS) - set(INSERT_SPECS)

ALLOWED_CODES = {
    # Transport / ABI glue
    Code.RETNQ,
    Code.VEX_VMOVDQU_XMM_XMMM128,
    Code.VEX_VMOVDQU_XMMM128_XMM,
    Code.VEX_VMOVDQU_YMM_YMMM256,
    Code.VEX_VMOVDQU_YMMM256_YMM,
    Code.MOV_R32_RM32,
    Code.MOV_R64_RM64,
    Code.MOV_RM32_R32,
    Code.MOV_RM64_R64,
    # State management
    Code.VEX_VZEROALL,
    Code.VEX_VZEROUPPER,
    # Unary
    Code.VEX_VPABSB_XMM_XMMM128,
    Code.VEX_VPABSW_XMM_XMMM128,
    Code.VEX_VPABSD_XMM_XMMM128,
    Code.VEX_VPHMINPOSUW_XMM_XMMM128,
    Code.VEX_VPMOVSXBW_XMM_XMMM64,
    Code.VEX_VPMOVSXBD_XMM_XMMM32,
    Code.VEX_VPMOVSXBQ_XMM_XMMM16,
    Code.VEX_VPMOVSXWD_XMM_XMMM64,
    Code.VEX_VPMOVSXWQ_XMM_XMMM32,
    Code.VEX_VPMOVSXDQ_XMM_XMMM64,
    Code.VEX_VPMOVZXBW_XMM_XMMM64,
    Code.VEX_VPMOVZXBD_XMM_XMMM32,
    Code.VEX_VPMOVZXBQ_XMM_XMMM16,
    Code.VEX_VPMOVZXWD_XMM_XMMM64,
    Code.VEX_VPMOVZXWQ_XMM_XMMM32,
    Code.VEX_VPMOVZXDQ_XMM_XMMM64,
    # Unary + imm8
    Code.VEX_VPSHUFHW_XMM_XMMM128_IMM8,
    Code.VEX_VPSHUFLW_XMM_XMMM128_IMM8,
    # Binary
    Code.VEX_VPACKSSWB_XMM_XMM_XMMM128,
    Code.VEX_VPACKSSDW_XMM_XMM_XMMM128,
    Code.VEX_VPACKUSDW_XMM_XMM_XMMM128,
    Code.VEX_VPADDB_XMM_XMM_XMMM128,
    Code.VEX_VPADDQ_XMM_XMM_XMMM128,
    Code.VEX_VPADDSB_XMM_XMM_XMMM128,
    Code.VEX_VPADDUSB_XMM_XMM_XMMM128,
    Code.VEX_VPADDUSW_XMM_XMM_XMMM128,
    Code.VEX_VPSUBB_XMM_XMM_XMMM128,
    Code.VEX_VPSUBD_XMM_XMM_XMMM128,
    Code.VEX_VPSUBQ_XMM_XMM_XMMM128,
    Code.VEX_VPSUBSB_XMM_XMM_XMMM128,
    Code.VEX_VPSUBSW_XMM_XMM_XMMM128,
    Code.VEX_VPSUBUSB_XMM_XMM_XMMM128,
    Code.VEX_VPSUBUSW_XMM_XMM_XMMM128,
    Code.VEX_VPCMPEQB_XMM_XMM_XMMM128,
    Code.VEX_VPCMPEQW_XMM_XMM_XMMM128,
    Code.VEX_VPCMPEQD_XMM_XMM_XMMM128,
    Code.VEX_VPCMPEQQ_XMM_XMM_XMMM128,
    Code.VEX_VPCMPGTB_XMM_XMM_XMMM128,
    Code.VEX_VPCMPGTW_XMM_XMM_XMMM128,
    Code.VEX_VPCMPGTD_XMM_XMM_XMMM128,
    Code.VEX_VPCMPGTQ_XMM_XMM_XMMM128,
    Code.VEX_VPMAXSB_XMM_XMM_XMMM128,
    Code.VEX_VPMAXSD_XMM_XMM_XMMM128,
    Code.VEX_VPMAXUB_XMM_XMM_XMMM128,
    Code.VEX_VPMAXUW_XMM_XMM_XMMM128,
    Code.VEX_VPMAXUD_XMM_XMM_XMMM128,
    Code.VEX_VPMINSB_XMM_XMM_XMMM128,
    Code.VEX_VPMINSD_XMM_XMM_XMMM128,
    Code.VEX_VPMINUB_XMM_XMM_XMMM128,
    Code.VEX_VPMINUW_XMM_XMM_XMMM128,
    Code.VEX_VPMINUD_XMM_XMM_XMMM128,
    Code.VEX_VPHADDW_XMM_XMM_XMMM128,
    Code.VEX_VPHADDD_XMM_XMM_XMMM128,
    Code.VEX_VPHADDSW_XMM_XMM_XMMM128,
    Code.VEX_VPHSUBW_XMM_XMM_XMMM128,
    Code.VEX_VPHSUBD_XMM_XMM_XMMM128,
    Code.VEX_VPHSUBSW_XMM_XMM_XMMM128,
    Code.VEX_VPMADDWD_XMM_XMM_XMMM128,
    Code.VEX_VPMULDQ_XMM_XMM_XMMM128,
    Code.VEX_VPMULUDQ_XMM_XMM_XMMM128,
    Code.VEX_VPMULHRSW_XMM_XMM_XMMM128,
    Code.VEX_VPMULHUW_XMM_XMM_XMMM128,
    Code.VEX_VPMULLD_XMM_XMM_XMMM128,
    Code.VEX_VPSADBW_XMM_XMM_XMMM128,
    Code.VEX_VPSIGNB_XMM_XMM_XMMM128,
    Code.VEX_VPSIGNW_XMM_XMM_XMMM128,
    Code.VEX_VPSIGND_XMM_XMM_XMMM128,
    Code.VEX_VPUNPCKLDQ_XMM_XMM_XMMM128,
    Code.VEX_VPUNPCKHDQ_XMM_XMM_XMMM128,
    Code.VEX_VPUNPCKLQDQ_XMM_XMM_XMMM128,
    Code.VEX_VPUNPCKHQDQ_XMM_XMM_XMMM128,
    # Binary + imm8
    Code.VEX_VPALIGNR_XMM_XMM_XMMM128_IMM8,
    Code.VEX_VPBLENDW_XMM_XMM_XMMM128_IMM8,
    Code.VEX_VMPSADBW_XMM_XMM_XMMM128_IMM8,
    # Ternary
    Code.VEX_VPBLENDVB_XMM_XMM_XMMM128_XMM,
    # Shifts (imm8 and xmm count)
    Code.VEX_VPSLLD_XMM_XMM_IMM8,
    Code.VEX_VPSLLD_XMM_XMM_XMMM128,
    Code.VEX_VPSLLQ_XMM_XMM_IMM8,
    Code.VEX_VPSLLQ_XMM_XMM_XMMM128,
    Code.VEX_VPSRLD_XMM_XMM_IMM8,
    Code.VEX_VPSRLD_XMM_XMM_XMMM128,
    Code.VEX_VPSRLQ_XMM_XMM_IMM8,
    Code.VEX_VPSRLQ_XMM_XMM_XMMM128,
    # Mask / Extract / Insert
    Code.VEX_VPMOVMSKB_R32_XMM,
    Code.VEX_VPMOVMSKB_R64_XMM,
    Code.VEX_VPEXTRB_R32M8_XMM_IMM8,
    Code.VEX_VPEXTRB_R64M8_XMM_IMM8,
    Code.VEX_VPEXTRW_R32_XMM_IMM8,
    Code.VEX_VPEXTRW_R64_XMM_IMM8,
    Code.VEX_VPEXTRW_R32M16_XMM_IMM8,
    Code.VEX_VPEXTRW_R64M16_XMM_IMM8,
    Code.VEX_VPEXTRD_RM32_XMM_IMM8,
    Code.VEX_VPEXTRQ_RM64_XMM_IMM8,
    Code.VEX_VPINSRB_XMM_XMM_R32M8_IMM8,
    Code.VEX_VPINSRB_XMM_XMM_R64M8_IMM8,
    Code.VEX_VPINSRW_XMM_XMM_R32M16_IMM8,
    Code.VEX_VPINSRW_XMM_XMM_R64M16_IMM8,
    Code.VEX_VPINSRD_XMM_XMM_RM32_IMM8,
    Code.VEX_VPINSRQ_XMM_XMM_RM64_IMM8,
}


ALLOWED_CODES |= WIDE_BINARY_CODES | WIDE_UNARY_CODES | WIDE_UNARY_IMM_CODES


PRELUDE = """#include <cstdint>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

using word = std::uint64_t;

struct Vec256 {
    std::uint64_t q[4];
};

static std::uint8_t load8(word address) {
    std::uint8_t val = 0u;
    std::memcpy(&val, reinterpret_cast<const void*>(address), 1);
    return val;
}

static std::uint16_t load16(word address) {
    std::uint16_t val = 0u;
    std::memcpy(&val, reinterpret_cast<const void*>(address), 2);
    return val;
}

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

static Vec256 load128(word address) {
    Vec256 v{};
    std::memcpy(v.q, reinterpret_cast<const void*>(address), 16);
    return v;
}

static Vec256 load256(word address) {
    Vec256 v{};
    std::memcpy(v.q, reinterpret_cast<const void*>(address), 32);
    return v;
}

static void store8(word address, std::uint8_t val) {
    std::memcpy(reinterpret_cast<void*>(address), &val, 1);
}

static void store16(word address, std::uint16_t val) {
    std::memcpy(reinterpret_cast<void*>(address), &val, 2);
}

static void store32(word address, std::uint32_t val) {
    std::memcpy(reinterpret_cast<void*>(address), &val, 4);
}

static void store64(word address, std::uint64_t val) {
    std::memcpy(reinterpret_cast<void*>(address), &val, 8);
}

static void store128(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
}

static void store256(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 32);
}

static Vec256 copy128(const Vec256& v) {
    return {{v.q[0], v.q[1], 0ull, 0ull}};
}

static std::uint8_t lane8(const Vec256& v, unsigned i) {
    return static_cast<std::uint8_t>((v.q[i >> 3] >> ((i & 7u) * 8u)) & 0xffull);
}

static void set_lane8(Vec256& v, unsigned i, std::uint8_t val) {
    const unsigned shift = (i & 7u) * 8u;
    v.q[i >> 3] = (v.q[i >> 3] & ~(0xffull << shift)) | (static_cast<std::uint64_t>(val) << shift);
}

static std::uint16_t lane16(const Vec256& v, unsigned i) {
    return static_cast<std::uint16_t>((v.q[i >> 2] >> ((i & 3u) * 16u)) & 0xffffull);
}

static void set_lane16(Vec256& v, unsigned i, std::uint16_t val) {
    const unsigned shift = (i & 3u) * 16u;
    v.q[i >> 2] = (v.q[i >> 2] & ~(0xffffull << shift)) | (static_cast<std::uint64_t>(val) << shift);
}

static std::uint32_t lane32(const Vec256& v, unsigned i) {
    return static_cast<std::uint32_t>((v.q[i >> 1] >> ((i & 1u) * 32u)) & 0xffffffffull);
}

static void set_lane32(Vec256& v, unsigned i, std::uint32_t val) {
    const unsigned shift = (i & 1u) * 32u;
    v.q[i >> 1] = (v.q[i >> 1] & ~(0xffffffffull << shift)) | (static_cast<std::uint64_t>(val) << shift);
}

static std::int8_t s8(std::uint8_t v) { return static_cast<std::int8_t>(v); }
static std::int16_t s16(std::uint16_t v) { return static_cast<std::int16_t>(v); }
static std::int32_t s32(std::uint32_t v) { return static_cast<std::int32_t>(v); }
static std::int64_t s64(std::uint64_t v) { return static_cast<std::int64_t>(v); }

static std::uint8_t sat_s8(std::int32_t v) {
    if (v > 127) return 0x7fu;
    if (v < -128) return 0x80u;
    return static_cast<std::uint8_t>(static_cast<std::int8_t>(v));
}

static std::uint16_t sat_s16(std::int32_t v) {
    if (v > 32767) return 0x7fffu;
    if (v < -32768) return 0x8000u;
    return static_cast<std::uint16_t>(static_cast<std::int16_t>(v));
}

static std::uint16_t sat_u16(std::int32_t v) {
    if (v > 65535) return 0xffffu;
    if (v < 0) return 0u;
    return static_cast<std::uint16_t>(v);
}

// --- Packed absolute value ---

static Vec256 op_vpabsb(const Vec256& a) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) {
        const std::uint8_t u = lane8(a, i);
        set_lane8(out, i, s8(u) < 0 ? static_cast<std::uint8_t>(0u - u) : u);
    }
    return out;
}

static Vec256 op_vpabsw(const Vec256& a) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        const std::uint16_t u = lane16(a, i);
        set_lane16(out, i, s16(u) < 0 ? static_cast<std::uint16_t>(0u - u) : u);
    }
    return out;
}

static Vec256 op_vpabsd(const Vec256& a) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        const std::uint32_t u = lane32(a, i);
        set_lane32(out, i, s32(u) < 0 ? (0u - u) : u);
    }
    return out;
}

// --- Pack with signed/unsigned saturation ---

static Vec256 op_vpacksswb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        set_lane8(out, i, sat_s8(s16(lane16(a, i))));
        set_lane8(out, i + 8u, sat_s8(s16(lane16(b, i))));
    }
    return out;
}

static Vec256 op_vpackssdw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane16(out, i, sat_s16(s32(lane32(a, i))));
        set_lane16(out, i + 4u, sat_s16(s32(lane32(b, i))));
    }
    return out;
}

static Vec256 op_vpackusdw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane16(out, i, sat_u16(s32(lane32(a, i))));
        set_lane16(out, i + 4u, sat_u16(s32(lane32(b, i))));
    }
    return out;
}

// --- Packed add ---

static Vec256 op_vpaddb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) set_lane8(out, i, static_cast<std::uint8_t>(lane8(a, i) + lane8(b, i)));
    return out;
}

static Vec256 op_vpaddw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) set_lane16(out, i, static_cast<std::uint16_t>(lane16(a, i) + lane16(b, i)));
    return out;
}

static Vec256 op_vpaddd(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) set_lane32(out, i, lane32(a, i) + lane32(b, i));
    return out;
}

static Vec256 op_vpaddq(const Vec256& a, const Vec256& b) {
    return {{a.q[0] + b.q[0], a.q[1] + b.q[1], 0ull, 0ull}};
}

static Vec256 op_vpaddsb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) set_lane8(out, i, sat_s8(static_cast<std::int32_t>(s8(lane8(a, i))) + s8(lane8(b, i))));
    return out;
}

// The 128-bit vpaddsw encoding stays with cpu_vex_aot.py; this helper only serves the 256-bit form.
static Vec256 op_vpaddsw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) set_lane16(out, i, sat_s16(static_cast<std::int32_t>(s16(lane16(a, i))) + s16(lane16(b, i))));
    return out;
}

static Vec256 op_vpaddusb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) {
        const unsigned sum = static_cast<unsigned>(lane8(a, i)) + static_cast<unsigned>(lane8(b, i));
        set_lane8(out, i, static_cast<std::uint8_t>(sum > 255u ? 255u : sum));
    }
    return out;
}

static Vec256 op_vpaddusw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        const unsigned sum = static_cast<unsigned>(lane16(a, i)) + static_cast<unsigned>(lane16(b, i));
        set_lane16(out, i, static_cast<std::uint16_t>(sum > 65535u ? 65535u : sum));
    }
    return out;
}

// --- Packed subtract ---

static Vec256 op_vpsubb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) set_lane8(out, i, static_cast<std::uint8_t>(lane8(a, i) - lane8(b, i)));
    return out;
}

static Vec256 op_vpsubw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) set_lane16(out, i, static_cast<std::uint16_t>(lane16(a, i) - lane16(b, i)));
    return out;
}

static Vec256 op_vpsubd(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) set_lane32(out, i, lane32(a, i) - lane32(b, i));
    return out;
}

static Vec256 op_vpsubq(const Vec256& a, const Vec256& b) {
    return {{a.q[0] - b.q[0], a.q[1] - b.q[1], 0ull, 0ull}};
}

static Vec256 op_vpsubsb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) set_lane8(out, i, sat_s8(static_cast<std::int32_t>(s8(lane8(a, i))) - s8(lane8(b, i))));
    return out;
}

static Vec256 op_vpsubsw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) set_lane16(out, i, sat_s16(static_cast<std::int32_t>(s16(lane16(a, i))) - s16(lane16(b, i))));
    return out;
}

static Vec256 op_vpsubusb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) {
        const std::uint8_t x = lane8(a, i), y = lane8(b, i);
        set_lane8(out, i, x > y ? static_cast<std::uint8_t>(x - y) : 0u);
    }
    return out;
}

static Vec256 op_vpsubusw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        const std::uint16_t x = lane16(a, i), y = lane16(b, i);
        set_lane16(out, i, x > y ? static_cast<std::uint16_t>(x - y) : 0u);
    }
    return out;
}

// --- Align and blend ---

static Vec256 op_vpalignr(const Vec256& a, const Vec256& b, std::uint8_t imm8) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) {
        const unsigned idx = i + static_cast<unsigned>(imm8);
        std::uint8_t val = 0u;
        if (idx < 16u) val = lane8(b, idx);
        else if (idx < 32u) val = lane8(a, idx - 16u);
        set_lane8(out, i, val);
    }
    return out;
}

static Vec256 op_vpblendvb(const Vec256& a, const Vec256& b, const Vec256& mask) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) {
        set_lane8(out, i, (lane8(mask, i) & 0x80u) ? lane8(b, i) : lane8(a, i));
    }
    return out;
}

static Vec256 op_vpblendw(const Vec256& a, const Vec256& b, std::uint8_t imm8) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        set_lane16(out, i, ((imm8 >> i) & 1u) ? lane16(b, i) : lane16(a, i));
    }
    return out;
}

// --- Packed compare ---

static Vec256 op_vpcmpeqb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) set_lane8(out, i, lane8(a, i) == lane8(b, i) ? 0xffu : 0u);
    return out;
}

static Vec256 op_vpcmpeqw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) set_lane16(out, i, lane16(a, i) == lane16(b, i) ? 0xffffu : 0u);
    return out;
}

static Vec256 op_vpcmpeqd(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) set_lane32(out, i, lane32(a, i) == lane32(b, i) ? 0xffffffffu : 0u);
    return out;
}

static Vec256 op_vpcmpeqq(const Vec256& a, const Vec256& b) {
    return {{a.q[0] == b.q[0] ? ~0ull : 0ull, a.q[1] == b.q[1] ? ~0ull : 0ull, 0ull, 0ull}};
}

static Vec256 op_vpcmpgtb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) set_lane8(out, i, s8(lane8(a, i)) > s8(lane8(b, i)) ? 0xffu : 0u);
    return out;
}

static Vec256 op_vpcmpgtw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) set_lane16(out, i, s16(lane16(a, i)) > s16(lane16(b, i)) ? 0xffffu : 0u);
    return out;
}

static Vec256 op_vpcmpgtd(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) set_lane32(out, i, s32(lane32(a, i)) > s32(lane32(b, i)) ? 0xffffffffu : 0u);
    return out;
}

static Vec256 op_vpcmpgtq(const Vec256& a, const Vec256& b) {
    return {{s64(a.q[0]) > s64(b.q[0]) ? ~0ull : 0ull, s64(a.q[1]) > s64(b.q[1]) ? ~0ull : 0ull, 0ull, 0ull}};
}

// --- 256-bit forms: the two 128-bit halves are independent ---

using Op128 = Vec256 (*)(const Vec256&, const Vec256&);

static Vec256 wide128(Op128 op, const Vec256& a, const Vec256& b) {
    const Vec256 lo = op({{a.q[0], a.q[1], 0ull, 0ull}}, {{b.q[0], b.q[1], 0ull, 0ull}});
    const Vec256 hi = op({{a.q[2], a.q[3], 0ull, 0ull}}, {{b.q[2], b.q[3], 0ull, 0ull}});
    return {{lo.q[0], lo.q[1], hi.q[0], hi.q[1]}};
}

using Op128Unary = Vec256 (*)(const Vec256&);
using Op128UnaryImm = Vec256 (*)(const Vec256&, std::uint8_t);

static Vec256 wide128_unary(Op128Unary op, const Vec256& a) {
    const Vec256 lo = op({{a.q[0], a.q[1], 0ull, 0ull}});
    const Vec256 hi = op({{a.q[2], a.q[3], 0ull, 0ull}});
    return {{lo.q[0], lo.q[1], hi.q[0], hi.q[1]}};
}

static Vec256 wide128_unary_imm(Op128UnaryImm op, const Vec256& a, std::uint8_t imm8) {
    const Vec256 lo = op({{a.q[0], a.q[1], 0ull, 0ull}}, imm8);
    const Vec256 hi = op({{a.q[2], a.q[3], 0ull, 0ull}}, imm8);
    return {{lo.q[0], lo.q[1], hi.q[0], hi.q[1]}};
}

// --- Packed min / max ---

static Vec256 op_vpmaxsb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) {
        const std::uint8_t x = lane8(a, i), y = lane8(b, i);
        set_lane8(out, i, s8(x) > s8(y) ? x : y);
    }
    return out;
}

static Vec256 op_vpmaxsw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        const std::uint16_t x = lane16(a, i), y = lane16(b, i);
        set_lane16(out, i, s16(x) > s16(y) ? x : y);
    }
    return out;
}

static Vec256 op_vpmaxsd(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        const std::uint32_t x = lane32(a, i), y = lane32(b, i);
        set_lane32(out, i, s32(x) > s32(y) ? x : y);
    }
    return out;
}

static Vec256 op_vpmaxub(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) {
        const std::uint8_t x = lane8(a, i), y = lane8(b, i);
        set_lane8(out, i, x > y ? x : y);
    }
    return out;
}

static Vec256 op_vpmaxuw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        const std::uint16_t x = lane16(a, i), y = lane16(b, i);
        set_lane16(out, i, x > y ? x : y);
    }
    return out;
}

static Vec256 op_vpmaxud(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        const std::uint32_t x = lane32(a, i), y = lane32(b, i);
        set_lane32(out, i, x > y ? x : y);
    }
    return out;
}

static Vec256 op_vpminsb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) {
        const std::uint8_t x = lane8(a, i), y = lane8(b, i);
        set_lane8(out, i, s8(x) < s8(y) ? x : y);
    }
    return out;
}

static Vec256 op_vpminsw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        const std::uint16_t x = lane16(a, i), y = lane16(b, i);
        set_lane16(out, i, s16(x) < s16(y) ? x : y);
    }
    return out;
}

static Vec256 op_vpminsd(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        const std::uint32_t x = lane32(a, i), y = lane32(b, i);
        set_lane32(out, i, s32(x) < s32(y) ? x : y);
    }
    return out;
}

static Vec256 op_vpminub(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) {
        const std::uint8_t x = lane8(a, i), y = lane8(b, i);
        set_lane8(out, i, x < y ? x : y);
    }
    return out;
}

static Vec256 op_vpminuw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        const std::uint16_t x = lane16(a, i), y = lane16(b, i);
        set_lane16(out, i, x < y ? x : y);
    }
    return out;
}

static Vec256 op_vpminud(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        const std::uint32_t x = lane32(a, i), y = lane32(b, i);
        set_lane32(out, i, x < y ? x : y);
    }
    return out;
}

// --- Horizontal operations ---

static Vec256 op_vphaddw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane16(out, i, static_cast<std::uint16_t>(lane16(a, 2u * i) + lane16(a, 2u * i + 1u)));
        set_lane16(out, i + 4u, static_cast<std::uint16_t>(lane16(b, 2u * i) + lane16(b, 2u * i + 1u)));
    }
    return out;
}

static Vec256 op_vphaddd(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 2u; ++i) {
        set_lane32(out, i, lane32(a, 2u * i) + lane32(a, 2u * i + 1u));
        set_lane32(out, i + 2u, lane32(b, 2u * i) + lane32(b, 2u * i + 1u));
    }
    return out;
}

static Vec256 op_vphaddsw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane16(out, i, sat_s16(static_cast<std::int32_t>(s16(lane16(a, 2u * i))) + s16(lane16(a, 2u * i + 1u))));
        set_lane16(out, i + 4u, sat_s16(static_cast<std::int32_t>(s16(lane16(b, 2u * i))) + s16(lane16(b, 2u * i + 1u))));
    }
    return out;
}

static Vec256 op_vphsubw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane16(out, i, static_cast<std::uint16_t>(lane16(a, 2u * i) - lane16(a, 2u * i + 1u)));
        set_lane16(out, i + 4u, static_cast<std::uint16_t>(lane16(b, 2u * i) - lane16(b, 2u * i + 1u)));
    }
    return out;
}

static Vec256 op_vphsubd(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 2u; ++i) {
        set_lane32(out, i, lane32(a, 2u * i) - lane32(a, 2u * i + 1u));
        set_lane32(out, i + 2u, lane32(b, 2u * i) - lane32(b, 2u * i + 1u));
    }
    return out;
}

static Vec256 op_vphsubsw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane16(out, i, sat_s16(static_cast<std::int32_t>(s16(lane16(a, 2u * i))) - s16(lane16(a, 2u * i + 1u))));
        set_lane16(out, i + 4u, sat_s16(static_cast<std::int32_t>(s16(lane16(b, 2u * i))) - s16(lane16(b, 2u * i + 1u))));
    }
    return out;
}

static Vec256 op_vphminposuw(const Vec256& a) {
    Vec256 out{};
    std::uint16_t best = lane16(a, 0u);
    std::uint16_t idx = 0u;
    for (unsigned i = 1u; i < 8u; ++i) {
        const std::uint16_t v = lane16(a, i);
        if (v < best) {
            best = v;
            idx = static_cast<std::uint16_t>(i);
        }
    }
    set_lane16(out, 0u, best);
    set_lane16(out, 1u, idx);
    return out;
}

// --- Packed multiply and multiply-add ---

static Vec256 op_vpmaddwd(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        const std::uint32_t p0 = static_cast<std::uint32_t>(
            static_cast<std::int32_t>(s16(lane16(a, 2u * i))) * static_cast<std::int32_t>(s16(lane16(b, 2u * i))));
        const std::uint32_t p1 = static_cast<std::uint32_t>(
            static_cast<std::int32_t>(s16(lane16(a, 2u * i + 1u))) * static_cast<std::int32_t>(s16(lane16(b, 2u * i + 1u))));
        set_lane32(out, i, p0 + p1);
    }
    return out;
}

static Vec256 op_vpmuldq(const Vec256& a, const Vec256& b) {
    const std::int64_t p0 = static_cast<std::int64_t>(s32(lane32(a, 0u))) * static_cast<std::int64_t>(s32(lane32(b, 0u)));
    const std::int64_t p1 = static_cast<std::int64_t>(s32(lane32(a, 2u))) * static_cast<std::int64_t>(s32(lane32(b, 2u)));
    return {{static_cast<std::uint64_t>(p0), static_cast<std::uint64_t>(p1), 0ull, 0ull}};
}

static Vec256 op_vpmuludq(const Vec256& a, const Vec256& b) {
    const std::uint64_t p0 = static_cast<std::uint64_t>(lane32(a, 0u)) * static_cast<std::uint64_t>(lane32(b, 0u));
    const std::uint64_t p1 = static_cast<std::uint64_t>(lane32(a, 2u)) * static_cast<std::uint64_t>(lane32(b, 2u));
    return {{p0, p1, 0ull, 0ull}};
}

static Vec256 op_vpmulhrsw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        const std::int32_t prod = static_cast<std::int32_t>(s16(lane16(a, i))) * static_cast<std::int32_t>(s16(lane16(b, i)));
        const std::uint32_t rounded = static_cast<std::uint32_t>(prod + 0x4000);
        set_lane16(out, i, static_cast<std::uint16_t>((rounded >> 15) & 0xffffu));
    }
    return out;
}

static Vec256 op_vpmulhuw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        const std::uint32_t prod = static_cast<std::uint32_t>(lane16(a, i)) * static_cast<std::uint32_t>(lane16(b, i));
        set_lane16(out, i, static_cast<std::uint16_t>(prod >> 16));
    }
    return out;
}

static Vec256 op_vpmulld(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) set_lane32(out, i, lane32(a, i) * lane32(b, i));
    return out;
}

// --- Sum of absolute differences ---

static std::uint16_t abs_diff_u8(std::uint8_t x, std::uint8_t y) {
    return x >= y ? static_cast<std::uint16_t>(x - y) : static_cast<std::uint16_t>(y - x);
}

static Vec256 op_vpsadbw(const Vec256& a, const Vec256& b) {
    std::uint64_t sum0 = 0ull, sum1 = 0ull;
    for (unsigned i = 0; i < 8u; ++i) {
        sum0 += abs_diff_u8(lane8(a, i), lane8(b, i));
        sum1 += abs_diff_u8(lane8(a, i + 8u), lane8(b, i + 8u));
    }
    return {{sum0, sum1, 0ull, 0ull}};
}

static Vec256 op_vmpsadbw(const Vec256& a, const Vec256& b, std::uint8_t imm8) {
    Vec256 out{};
    const unsigned a_off = ((imm8 >> 2) & 1u) * 4u;
    const unsigned b_off = (imm8 & 3u) * 4u;
    for (unsigned k = 0; k < 8u; ++k) {
        std::uint16_t sum = 0u;
        for (unsigned j = 0; j < 4u; ++j) {
            sum = static_cast<std::uint16_t>(sum + abs_diff_u8(lane8(a, a_off + k + j), lane8(b, b_off + j)));
        }
        set_lane16(out, k, sum);
    }
    return out;
}

// --- Shuffles ---

static Vec256 op_vpshufhw(const Vec256& a, std::uint8_t imm8) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane16(out, i, lane16(a, i));
        set_lane16(out, i + 4u, lane16(a, 4u + ((imm8 >> (2u * i)) & 3u)));
    }
    return out;
}

static Vec256 op_vpshuflw(const Vec256& a, std::uint8_t imm8) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane16(out, i, lane16(a, (imm8 >> (2u * i)) & 3u));
        set_lane16(out, i + 4u, lane16(a, i + 4u));
    }
    return out;
}

// --- Packed sign ---

static Vec256 op_vpsignb(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 16u; ++i) {
        const std::uint8_t x = lane8(a, i);
        const std::int8_t c = s8(lane8(b, i));
        set_lane8(out, i, c < 0 ? static_cast<std::uint8_t>(0u - x) : (c == 0 ? 0u : x));
    }
    return out;
}

static Vec256 op_vpsignw(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) {
        const std::uint16_t x = lane16(a, i);
        const std::int16_t c = s16(lane16(b, i));
        set_lane16(out, i, c < 0 ? static_cast<std::uint16_t>(0u - x) : (c == 0 ? 0u : x));
    }
    return out;
}

static Vec256 op_vpsignd(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        const std::uint32_t x = lane32(a, i);
        const std::int32_t c = s32(lane32(b, i));
        set_lane32(out, i, c < 0 ? (0u - x) : (c == 0 ? 0u : x));
    }
    return out;
}

// --- Sign-extend and zero-extend moves ---

static Vec256 op_vpmovsxbw(const Vec256& a) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) set_lane16(out, i, static_cast<std::uint16_t>(static_cast<std::int16_t>(s8(lane8(a, i)))));
    return out;
}

static Vec256 op_vpmovsxbd(const Vec256& a) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) set_lane32(out, i, static_cast<std::uint32_t>(static_cast<std::int32_t>(s8(lane8(a, i)))));
    return out;
}

static Vec256 op_vpmovsxbq(const Vec256& a) {
    return {{static_cast<std::uint64_t>(static_cast<std::int64_t>(s8(lane8(a, 0u)))),
             static_cast<std::uint64_t>(static_cast<std::int64_t>(s8(lane8(a, 1u)))), 0ull, 0ull}};
}

static Vec256 op_vpmovsxwd(const Vec256& a) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) set_lane32(out, i, static_cast<std::uint32_t>(static_cast<std::int32_t>(s16(lane16(a, i)))));
    return out;
}

static Vec256 op_vpmovsxwq(const Vec256& a) {
    return {{static_cast<std::uint64_t>(static_cast<std::int64_t>(s16(lane16(a, 0u)))),
             static_cast<std::uint64_t>(static_cast<std::int64_t>(s16(lane16(a, 1u)))), 0ull, 0ull}};
}

static Vec256 op_vpmovsxdq(const Vec256& a) {
    return {{static_cast<std::uint64_t>(static_cast<std::int64_t>(s32(lane32(a, 0u)))),
             static_cast<std::uint64_t>(static_cast<std::int64_t>(s32(lane32(a, 1u)))), 0ull, 0ull}};
}

static Vec256 op_vpmovzxbw(const Vec256& a) {
    Vec256 out{};
    for (unsigned i = 0; i < 8u; ++i) set_lane16(out, i, lane8(a, i));
    return out;
}

static Vec256 op_vpmovzxbd(const Vec256& a) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) set_lane32(out, i, lane8(a, i));
    return out;
}

static Vec256 op_vpmovzxbq(const Vec256& a) {
    return {{lane8(a, 0u), lane8(a, 1u), 0ull, 0ull}};
}

static Vec256 op_vpmovzxwd(const Vec256& a) {
    Vec256 out{};
    for (unsigned i = 0; i < 4u; ++i) set_lane32(out, i, lane16(a, i));
    return out;
}

static Vec256 op_vpmovzxwq(const Vec256& a) {
    return {{lane16(a, 0u), lane16(a, 1u), 0ull, 0ull}};
}

static Vec256 op_vpmovzxdq(const Vec256& a) {
    return {{lane32(a, 0u), lane32(a, 1u), 0ull, 0ull}};
}

// --- Unpack / interleave ---

static Vec256 op_vpunpckldq(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 2u; ++i) {
        set_lane32(out, 2u * i, lane32(a, i));
        set_lane32(out, 2u * i + 1u, lane32(b, i));
    }
    return out;
}

static Vec256 op_vpunpckhdq(const Vec256& a, const Vec256& b) {
    Vec256 out{};
    for (unsigned i = 0; i < 2u; ++i) {
        set_lane32(out, 2u * i, lane32(a, i + 2u));
        set_lane32(out, 2u * i + 1u, lane32(b, i + 2u));
    }
    return out;
}

static Vec256 op_vpunpcklqdq(const Vec256& a, const Vec256& b) {
    return {{a.q[0], b.q[0], 0ull, 0ull}};
}

static Vec256 op_vpunpckhqdq(const Vec256& a, const Vec256& b) {
    return {{a.q[1], b.q[1], 0ull, 0ull}};
}

// --- Packed shifts (imm8 and XMM-count; counts >= lane width saturate, never wrap) ---

static Vec256 op_vpslld(const Vec256& a, std::uint64_t count) {
    Vec256 out{};
    if (count >= 32ull) return out;
    const unsigned c = static_cast<unsigned>(count);
    for (unsigned i = 0; i < 4u; ++i) set_lane32(out, i, lane32(a, i) << c);
    return out;
}

static Vec256 op_vpsllq(const Vec256& a, std::uint64_t count) {
    if (count >= 64ull) return Vec256{};
    const unsigned c = static_cast<unsigned>(count);
    return {{a.q[0] << c, a.q[1] << c, 0ull, 0ull}};
}

static Vec256 op_vpsrld(const Vec256& a, std::uint64_t count) {
    Vec256 out{};
    if (count >= 32ull) return out;
    const unsigned c = static_cast<unsigned>(count);
    for (unsigned i = 0; i < 4u; ++i) set_lane32(out, i, lane32(a, i) >> c);
    return out;
}

static Vec256 op_vpsrlq(const Vec256& a, std::uint64_t count) {
    if (count >= 64ull) return Vec256{};
    const unsigned c = static_cast<unsigned>(count);
    return {{a.q[0] >> c, a.q[1] >> c, 0ull, 0ull}};
}

// --- Movemask, extract and insert ---

static std::uint32_t op_vpmovmskb(const Vec256& a) {
    std::uint32_t mask = 0u;
    for (unsigned i = 0; i < 16u; ++i) {
        if (lane8(a, i) & 0x80u) mask |= (1u << i);
    }
    return mask;
}

static std::uint8_t op_vpextrb(const Vec256& a, std::uint8_t imm8) {
    return lane8(a, imm8 & 15u);
}

static std::uint16_t op_vpextrw(const Vec256& a, std::uint8_t imm8) {
    return lane16(a, imm8 & 7u);
}

static std::uint32_t op_vpextrd(const Vec256& a, std::uint8_t imm8) {
    return lane32(a, imm8 & 3u);
}

static std::uint64_t op_vpextrq(const Vec256& a, std::uint8_t imm8) {
    return a.q[imm8 & 1u];
}

static Vec256 op_vpinsrb(const Vec256& a, std::uint8_t val, std::uint8_t imm8) {
    Vec256 out = copy128(a);
    set_lane8(out, imm8 & 15u, val);
    return out;
}

static Vec256 op_vpinsrw(const Vec256& a, std::uint16_t val, std::uint8_t imm8) {
    Vec256 out = copy128(a);
    set_lane16(out, imm8 & 7u, val);
    return out;
}

static Vec256 op_vpinsrd(const Vec256& a, std::uint32_t val, std::uint8_t imm8) {
    Vec256 out = copy128(a);
    set_lane32(out, imm8 & 3u, val);
    return out;
}

static Vec256 op_vpinsrq(const Vec256& a, std::uint64_t val, std::uint8_t imm8) {
    Vec256 out = copy128(a);
    out.q[imm8 & 1u] = val;
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


def scratch_gpr(register, ins):
    """Map volatile scratch GPR (rax/r9/r10/r11 or 32-bit halves) to (name, is_64)."""
    if register in SCRATCH_GPR64:
        return SCRATCH_GPR64[register], True
    if register in SCRATCH_GPR32:
        return SCRATCH_GPR32[register], False
    if register in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {register} at RVA {ins.ip:#x}: {ins}")
    if register in ARGUMENT_GPRS:
        raise ValueError(f"unsupported argument GPR operand {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported GPR {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins):
    """Lower a flat [base + disp] pointer operand rooted in an argument GPR64."""
    base = ins.memory_base
    if (ins.is_ip_rel_memory_operand or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
            or ins.memory_index != R.NONE or not RegisterExt.is_gpr64(base)
            or base not in ARGUMENTS):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    name = ARGUMENTS[base]
    return f"({name} + {disp:#x}ull)" if disp else name


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line AVX packed-integer leaf function."""
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
    statements = []
    returned = False

    def read_vec(op_index, ins, allow_ymm=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if wide and not allow_ymm:
            raise ValueError(f"unsupported VEX.256 (L=1) packed-integer encoding at RVA {ins.ip:#x}: {ins}")
        if idx not in defined_vec:
            raise ValueError(f"undefined vector register ymm{idx} at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return f"ymm{idx}", wide

    def write_vec(op_index, ins, allow_ymm=False):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if wide and not allow_ymm:
            raise ValueError(f"unsupported VEX.256 (L=1) packed-integer encoding at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return idx, f"ymm{idx}", wide

    def read_gpr(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR operand {op_index} at RVA {ins.ip:#x}: {ins}")
        reg_name, is64 = scratch_gpr(ins.op_register(op_index), ins)
        if reg_name not in defined_gpr:
            raise ValueError(f"undefined GPR {reg_name} at RVA {ins.ip:#x}: {ins}")
        used_gpr.add(reg_name)
        return reg_name, is64

    def write_gpr(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR destination at RVA {ins.ip:#x}: {ins}")
        reg_name, is64 = scratch_gpr(ins.op_register(op_index), ins)
        used_gpr.add(reg_name)
        return reg_name, is64

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        op_code = ins.op_code()
        if (op_code.encoding == EncodingKind.EVEX or ins.op_mask != R.NONE
                or ins.zeroing_masking or ins.is_broadcast):
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        mnemonic = ins.mnemonic
        wide_unary = ins.code in WIDE_UNARY_CODES
        wide_unary_imm = ins.code in WIDE_UNARY_IMM_CODES
        wide_binary = ins.code in WIDE_BINARY_CODES
        wide_any = wide_unary or wide_unary_imm or wide_binary
        if (mnemonic in AVX_INT_128_ONLY_MNEMONICS and not wide_any
                and op_code.encoding == EncodingKind.VEX and op_code.l == 1):
            raise ValueError(f"unsupported VEX.256 (L=1) packed-integer encoding at RVA {ins.ip:#x}: {ins}")
        if (mnemonic in REGISTER_ONLY_AVX_INT or wide_any) and any(
                ins.op_kind(i) == OpKind.MEMORY for i in range(ins.op_count)):
            raise ValueError(
                f"unsupported memory operand for register-domain AVX integer instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            statements.append("return;")
            returned = True
        elif ins.code == Code.VEX_VZEROUPPER:
            for idx in sorted(defined_vec):
                statements.append(f"ymm{idx}.q[2] = 0ull; ymm{idx}.q[3] = 0ull;")
        elif ins.code == Code.VEX_VZEROALL:
            for idx in range(6):
                used_vec.add(idx)
                defined_vec.add(idx)
                statements.append(f"ymm{idx} = Vec256{{}};")
        elif mnemonic == M.VMOVDQU:
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst_idx, dst, wide = write_vec(0, ins, allow_ymm=True)
                addr = flat_address(ins)
                statements.append(f"{dst} = {'load256' if wide else 'load128'}({addr});")
                defined_vec.add(dst_idx)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src, wide = read_vec(1, ins, allow_ymm=True)
                addr = flat_address(ins)
                statements.append(f"{'store256' if wide else 'store128'}({addr}, {src});")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src, _ = read_vec(1, ins, allow_ymm=True)
                dst_idx, dst, wide = write_vec(0, ins, allow_ymm=True)
                statements.append(f"{dst} = {src};" if wide else f"{dst} = copy128({src});")
                defined_vec.add(dst_idx)
            else:
                raise ValueError(f"unsupported vmovdqu operands at RVA {ins.ip:#x}: {ins}")
        elif mnemonic == M.MOV:
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst_gpr, is64 = write_gpr(0, ins)
                addr = flat_address(ins)
                statements.append(f"{dst_gpr} = {'load64' if is64 else 'static_cast<word>(load32'}({addr}){'' if is64 else ')'};")
                defined_gpr.add(dst_gpr)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src_gpr, is64 = read_gpr(1, ins)
                addr = flat_address(ins)
                if is64:
                    statements.append(f"store64({addr}, {src_gpr});")
                else:
                    statements.append(f"store32({addr}, static_cast<std::uint32_t>({src_gpr}));")
            else:
                raise ValueError(f"unsupported mov operands at RVA {ins.ip:#x}: {ins}")
        elif wide_unary:
            src, _ = read_vec(1, ins, allow_ymm=True)
            dst_idx, dst, _ = write_vec(0, ins, allow_ymm=True)
            statements.append(f"{dst} = wide128_unary({WIDE_UNARY_HELPERS[mnemonic]}, {src});")
            defined_vec.add(dst_idx)
        elif wide_unary_imm:
            src, _ = read_vec(1, ins, allow_ymm=True)
            dst_idx, dst, _ = write_vec(0, ins, allow_ymm=True)
            imm = ins.immediate8 & 0xff
            statements.append(
                f"{dst} = wide128_unary_imm({WIDE_UNARY_IMM_HELPERS[mnemonic]}, {src}, {imm:#x}u);")
            defined_vec.add(dst_idx)
        elif mnemonic in UNARY_HELPERS:
            src, _ = read_vec(1, ins)
            dst_idx, dst, _ = write_vec(0, ins)
            statements.append(f"{dst} = {UNARY_HELPERS[mnemonic]}({src});")
            defined_vec.add(dst_idx)
        elif mnemonic in UNARY_IMM_HELPERS:
            src, _ = read_vec(1, ins)
            dst_idx, dst, _ = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = {UNARY_IMM_HELPERS[mnemonic]}({src}, {imm:#x}u);")
            defined_vec.add(dst_idx)
        elif wide_binary:
            src1, _ = read_vec(1, ins, allow_ymm=True)
            src2, _ = read_vec(2, ins, allow_ymm=True)
            dst_idx, dst, _ = write_vec(0, ins, allow_ymm=True)
            statements.append(f"{dst} = wide128({WIDE_BINARY_HELPERS[mnemonic]}, {src1}, {src2});")
            defined_vec.add(dst_idx)
        elif mnemonic in BINARY_HELPERS:
            src1, _ = read_vec(1, ins)
            src2, _ = read_vec(2, ins)
            dst_idx, dst, _ = write_vec(0, ins)
            statements.append(f"{dst} = {BINARY_HELPERS[mnemonic]}({src1}, {src2});")
            defined_vec.add(dst_idx)
        elif mnemonic in BINARY_IMM_HELPERS:
            src1, _ = read_vec(1, ins)
            src2, _ = read_vec(2, ins)
            dst_idx, dst, _ = write_vec(0, ins)
            imm = ins.immediate8 & 0xff
            statements.append(f"{dst} = {BINARY_IMM_HELPERS[mnemonic]}({src1}, {src2}, {imm:#x}u);")
            defined_vec.add(dst_idx)
        elif mnemonic == M.VPBLENDVB:
            src1, _ = read_vec(1, ins)
            src2, _ = read_vec(2, ins)
            mask, _ = read_vec(3, ins)
            dst_idx, dst, _ = write_vec(0, ins)
            statements.append(f"{dst} = op_vpblendvb({src1}, {src2}, {mask});")
            defined_vec.add(dst_idx)
        elif mnemonic in SHIFT_HELPERS:
            src1, _ = read_vec(1, ins)
            if ins.op_kind(2) == OpKind.IMMEDIATE8:
                count_expr = f"{ins.immediate8 & 0xff:#x}ull"
            else:
                src2, _ = read_vec(2, ins)
                count_expr = f"{src2}.q[0]"
            dst_idx, dst, _ = write_vec(0, ins)
            statements.append(f"{dst} = {SHIFT_HELPERS[mnemonic]}({src1}, {count_expr});")
            defined_vec.add(dst_idx)
        elif mnemonic == M.VPMOVMSKB:
            src, _ = read_vec(1, ins)
            dst_gpr, _ = write_gpr(0, ins)
            statements.append(f"{dst_gpr} = static_cast<word>(op_vpmovmskb({src}));")
            defined_gpr.add(dst_gpr)
        elif mnemonic in EXTRACT_SPECS:
            helper, store_fn = EXTRACT_SPECS[mnemonic]
            src, _ = read_vec(1, ins)
            imm = ins.immediate8 & 0xff
            if ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins)
                statements.append(f"{store_fn}({addr}, {helper}({src}, {imm:#x}u));")
            else:
                dst_gpr, _ = write_gpr(0, ins)
                statements.append(f"{dst_gpr} = static_cast<word>({helper}({src}, {imm:#x}u));")
                defined_gpr.add(dst_gpr)
        elif mnemonic in INSERT_SPECS:
            helper, load_fn, cast_prefix = INSERT_SPECS[mnemonic]
            src1, _ = read_vec(1, ins)
            imm = ins.immediate8 & 0xff
            if ins.op2_kind == OpKind.MEMORY:
                addr = flat_address(ins)
                val_expr = f"{load_fn}({addr})"
            else:
                src_gpr, _ = read_gpr(2, ins)
                val_expr = f"{cast_prefix}({src_gpr})" if cast_prefix else src_gpr
            dst_idx, dst, _ = write_vec(0, ins)
            statements.append(f"{dst} = {helper}({src1}, {val_expr}, {imm:#x}u);")
            defined_vec.add(dst_idx)
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec256 ymm{idx}{{}};" for idx in sorted(used_vec)]
    decls += [f"    word {gpr} = 0ull;" for gpr in sorted(used_gpr)]
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
