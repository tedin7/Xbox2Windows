"""Differential, exact-width, discarded-read, and rejection checks for opt-in AVX move/blend/shuffle memory ops."""

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

from iced_x86 import Code, Decoder, Formatter, FormatterSyntax, Instruction, MemoryOperand, Register as R
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from avx_move_mem_check import (  # noqa: E402
    BUFFER_SIZE,
    OUT_FILL,
    OUT_SIZE,
    PATTERNS,
    encode_seq,
    invoke,
    require_equal,
)
from cpu_avx_move_aot import lift_code, translate  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

EXPORTS = (
    "op_vblendpd_xmm_mem",
    "op_vblendpd_ymm_mem",
    "op_vblendps_xmm_mem",
    "op_vblendps_ymm_mem",
    "op_vblendvpd_xmm_mem",
    "op_vblendvpd_ymm_mem",
    "op_vblendvps_xmm_mem",
    "op_vblendvps_ymm_mem",
    "op_vextractf128_mem_lo",
    "op_vextractf128_mem_hi",
    "op_vextractf128_mem_disp",
    "op_vinsertf128_mem_lo",
    "op_vinsertf128_mem_hi",
    "op_vextractps_mem",
    "op_vinsertps_mem",
    "op_vperm2f128_mem",
    "op_vpermilpd_imm_xmm_mem",
    "op_vpermilpd_imm_ymm_mem",
    "op_vpermilpd_var_ymm_mem",
    "op_vpermilps_imm_xmm_mem",
    "op_vpermilps_imm_ymm_mem",
    "op_vpermilps_var_ymm_mem",
    "op_vshufpd_xmm_mem",
    "op_vshufpd_ymm_mem",
    "op_vshufps_xmm_mem",
    "op_vshufps_ymm_mem",
    "op_vunpcklpd_xmm_mem",
    "op_vunpcklpd_ymm_mem",
    "op_vunpckhpd_xmm_mem",
    "op_vunpckhpd_ymm_mem",
    "op_vunpcklps_xmm_mem",
    "op_vunpcklps_ymm_mem",
    "op_vunpckhps_xmm_mem",
    "op_vunpckhps_ymm_mem",
    "op_vmovddup_xmm_mem",
    "op_vmovddup_ymm_mem",
    "op_vmovshdup_xmm_mem",
    "op_vmovshdup_ymm_mem",
    "op_vmovsldup_xmm_mem",
    "op_vmovsldup_ymm_mem",
    "op_discarded_mem_read",
)


def _u32s(buf):
    return struct.unpack(f"<{len(buf) // 4}I", buf)


def _u64s(buf):
    return struct.unpack(f"<{len(buf) // 8}Q", buf)


def _p32(vals):
    return struct.pack(f"<{len(vals)}I", *vals)


def _p64(vals):
    return struct.pack(f"<{len(vals)}Q", *vals)


def _pad(buf):
    return buf + b"\x00" * (32 - len(buf))


def _vblendpd(a, b, imm8, wide):
    n = 4 if wide else 2
    qa, qb = _u64s(a[:8 * n]), _u64s(b[:8 * n])
    return _pad(_p64([qb[i] if ((imm8 >> i) & 1) else qa[i] for i in range(n)]))


def _vblendps(a, b, imm8, wide):
    n = 8 if wide else 4
    da, db = _u32s(a[:4 * n]), _u32s(b[:4 * n])
    return _pad(_p32([db[i] if ((imm8 >> i) & 1) else da[i] for i in range(n)]))


def _vblendvpd(a, b, mask, wide):
    n = 4 if wide else 2
    qa, qb, qm = _u64s(a[:8 * n]), _u64s(b[:8 * n]), _u64s(mask[:8 * n])
    return _pad(_p64([qb[i] if (qm[i] >> 63) else qa[i] for i in range(n)]))


def _vblendvps(a, b, mask, wide):
    n = 8 if wide else 4
    da, db, dm = _u32s(a[:4 * n]), _u32s(b[:4 * n]), _u32s(mask[:4 * n])
    return _pad(_p32([db[i] if (dm[i] >> 31) else da[i] for i in range(n)]))


def _vperm2f128(a, b, imm8):
    lanes = (a[:16], a[16:32], b[:16], b[16:32])
    def pick(n):
        return b"\x00" * 16 if (n & 8) else lanes[n & 3]
    return pick(imm8 & 0xF) + pick((imm8 >> 4) & 0xF)


def _vshufpd(a, b, imm8, wide):
    qa, qb = _u64s(a[:32]), _u64s(b[:32])
    out = [qa[(imm8 >> 0) & 1], qb[(imm8 >> 1) & 1]]
    if wide:
        out += [qa[2 + ((imm8 >> 2) & 1)], qb[2 + ((imm8 >> 3) & 1)]]
    return _pad(_p64(out))


def _vshufps(a, b, imm8, wide):
    da, db = _u32s(a[:32]), _u32s(b[:32])
    out = [da[(imm8 >> 0) & 3], da[(imm8 >> 2) & 3], db[(imm8 >> 4) & 3], db[(imm8 >> 6) & 3]]
    if wide:
        out += [da[4 + ((imm8 >> 0) & 3)], da[4 + ((imm8 >> 2) & 3)],
                db[4 + ((imm8 >> 4) & 3)], db[4 + ((imm8 >> 6) & 3)]]
    return _pad(_p32(out))


def reference(name, a, b, _c, altered=False):
    if name == "op_vblendpd_xmm_mem":
        return _vblendpd(a, b[16:32], 2, False)
    if name == "op_vblendpd_ymm_mem":
        return _vblendpd(a, b[32:64], 5 if altered else 0xA, True)
    if name == "op_vblendps_xmm_mem":
        return _vblendps(a, a[32:48], 0xA, False)
    if name == "op_vblendps_ymm_mem":
        return _vblendps(a, b[16:48], 0x5A, True)
    if name == "op_vblendvpd_xmm_mem":
        return _vblendvpd(a, a[48:64], b[32:48], False)
    if name == "op_vblendvpd_ymm_mem":
        return _vblendvpd(a, b[:32], b[32:64], True)
    if name == "op_vblendvps_xmm_mem":
        return _vblendvps(a, b[16:32], a[48:64], False)
    if name == "op_vblendvps_ymm_mem":
        return _vblendvps(a, a[32:64], b[32:64], True)
    if name == "op_vextractf128_mem_lo":
        return a[:16] + OUT_FILL[16:]
    if name == "op_vextractf128_mem_hi":
        return a[16:32] + OUT_FILL[16:]
    if name == "op_vextractf128_mem_disp":
        return a[16:32] + a[:16]
    if name == "op_vinsertf128_mem_lo":
        return b[3:19] + a[16:32]
    if name == "op_vinsertf128_mem_hi":
        return a[:16] + b[19:35]
    if name == "op_vextractps_mem":
        return a[8:12] + a[12:16] + OUT_FILL[8:]
    if name == "op_vinsertps_mem":
        imm = 0xD2 if altered else 0xD4
        out = list(_u32s(a[:16]))
        out[(imm >> 4) & 3] = _u32s(b[7:11])[0]
        for i in range(4):
            if (imm >> i) & 1:
                out[i] = 0
        return _pad(_p32(out))
    if name == "op_vperm2f128_mem":
        return _vperm2f128(a[:32], b[16:48], 0x31 if altered else 0x20)
    if name == "op_vpermilpd_imm_xmm_mem":
        q = _u64s(a[3:19])
        return _pad(_p64([q[1], q[0]]))
    if name == "op_vpermilpd_imm_ymm_mem":
        q = _u64s(a[3:35])
        return _p64([q[1], q[0], q[3], q[2]])
    if name == "op_vpermilpd_var_ymm_mem":
        qa, qc = _u64s(a[:32]), _u64s(b[8:40])
        return _p64([qa[(qc[0] >> 1) & 1], qa[(qc[1] >> 1) & 1],
                     qa[2 + ((qc[2] >> 1) & 1)], qa[2 + ((qc[3] >> 1) & 1)]])
    if name == "op_vpermilps_imm_xmm_mem":
        d = _u32s(a[5:21])
        return _pad(_p32([d[(0x1B >> (2 * i)) & 3] for i in range(4)]))
    if name == "op_vpermilps_imm_ymm_mem":
        d = _u32s(a[5:37])
        return _p32([d[(0x93 >> (2 * i)) & 3] for i in range(4)] +
                    [d[4 + ((0x93 >> (2 * i)) & 3)] for i in range(4)])
    if name == "op_vpermilps_var_ymm_mem":
        da, dc = _u32s(a[:32]), _u32s(b[8:40])
        return _p32([da[dc[i] & 3] for i in range(4)] + [da[4 + (dc[4 + i] & 3)] for i in range(4)])
    if name == "op_vshufpd_xmm_mem":
        return _vshufpd(a, b[16:32], 2, False)
    if name == "op_vshufpd_ymm_mem":
        return _vshufpd(a, b[16:48], 9, True)
    if name == "op_vshufps_xmm_mem":
        return _vshufps(a, b[16:32], 0xE4, False)
    if name == "op_vshufps_ymm_mem":
        return _vshufps(a, b[16:48], 0x4E, True)
    if name == "op_vunpcklpd_xmm_mem":
        qa, qb = _u64s(a[:16]), _u64s(b[16:32])
        return _pad(_p64([qa[0], qb[0]]))
    if name == "op_vunpcklpd_ymm_mem":
        qa, qb = _u64s(a[:32]), _u64s(b[16:48])
        return _p64([qa[0], qb[0], qa[2], qb[2]])
    if name == "op_vunpckhpd_xmm_mem":
        qa, qb = _u64s(a[:16]), _u64s(b[16:32])
        return _pad(_p64([qa[1], qb[1]]))
    if name == "op_vunpckhpd_ymm_mem":
        qa, qb = _u64s(a[:32]), _u64s(b[16:48])
        return _p64([qa[1], qb[1], qa[3], qb[3]])
    if name == "op_vunpcklps_xmm_mem":
        da, db = _u32s(a[:16]), _u32s(b[16:32])
        return _pad(_p32([da[0], db[0], da[1], db[1]]))
    if name == "op_vunpcklps_ymm_mem":
        da, db = _u32s(a[:32]), _u32s(b[16:48])
        return _p32([da[0], db[0], da[1], db[1], da[4], db[4], da[5], db[5]])
    if name == "op_vunpckhps_xmm_mem":
        da, db = _u32s(a[:16]), _u32s(b[16:32])
        return _pad(_p32([da[2], db[2], da[3], db[3]]))
    if name == "op_vunpckhps_ymm_mem":
        da, db = _u32s(a[:32]), _u32s(b[16:48])
        return _p32([da[2], db[2], da[3], db[3], da[6], db[6], da[7], db[7]])
    if name == "op_vmovddup_xmm_mem":
        return _pad(a[3:11] * 2)
    if name == "op_vmovddup_ymm_mem":
        return a[3:11] * 2 + a[19:27] * 2
    if name == "op_vmovshdup_xmm_mem":
        d = _u32s(a[5:21])
        return _pad(_p32([d[1], d[1], d[3], d[3]]))
    if name == "op_vmovshdup_ymm_mem":
        d = _u32s(a[5:37])
        return _p32([d[1], d[1], d[3], d[3], d[5], d[5], d[7], d[7]])
    if name == "op_vmovsldup_xmm_mem":
        d = _u32s(a[7:23])
        return _pad(_p32([d[0], d[0], d[2], d[2]]))
    if name == "op_vmovsldup_ymm_mem":
        d = _u32s(a[7:39])
        return _p32([d[0], d[0], d[2], d[2], d[4], d[4], d[6], d[6]])
    if name == "op_discarded_mem_read":
        return a[:32]
    raise ValueError(name)


def build_buffers():
    rng = random.Random(0xA7E311)
    cases = []
    # Pattern cases covering IEEE-754 SNaN/QNaN/inf/-0 bit preservation and sign-bit masks.
    for i in range(len(PATTERNS)):
        a = bytearray(PATTERNS[i] + PATTERNS[(i * 3 + 1) % len(PATTERNS)])
        a += PATTERNS[(i * 5 + 2) % len(PATTERNS)] + PATTERNS[(i * 7 + 3) % len(PATTERNS)]
        a += PATTERNS[(i * 11 + 4) % len(PATTERNS)] + PATTERNS[(i * 13 + 5) % len(PATTERNS)]
        b = bytearray(PATTERNS[(i + 7) % len(PATTERNS)] + PATTERNS[(i * 3 + 9) % len(PATTERNS)])
        b += PATTERNS[(i * 5 + 11) % len(PATTERNS)] + PATTERNS[(i * 7 + 13) % len(PATTERNS)]
        b += PATTERNS[(i * 11 + 15) % len(PATTERNS)] + PATTERNS[(i * 13 + 1) % len(PATTERNS)]
        c = bytearray(OUT_FILL + rng.randbytes(BUFFER_SIZE - OUT_SIZE))
        cases.append((bytes(a[:BUFFER_SIZE]), bytes(b[:BUFFER_SIZE]), bytes(c)))
    for _ in range(16):
        cases.append((rng.randbytes(BUFFER_SIZE), rng.randbytes(BUFFER_SIZE),
                      OUT_FILL + rng.randbytes(BUFFER_SIZE - OUT_SIZE)))
    return cases


def bind(path, prefix):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


def host_has_avx(lifted_lib):
    if hasattr(lifted_lib, "host_has_avx"):
        lifted_lib.host_has_avx.argtypes = []
        lifted_lib.host_has_avx.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_avx())
    try:
        return " avx " in Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace")
    except OSError:
        return False


def check_translator(original_path):
    source_cpp, _ = translate(original_path, EXPORTS, allow_memory=True)
    if "host_has_avx" not in source_cpp:
        raise AssertionError("translated C++ missing host_has_avx")

    ld_ymm0 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX))
    ld_ymm1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM1, MemoryOperand(R.RCX))
    ld_ymm3 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM3, MemoryOperand(R.RDX))
    ld_xmm1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RCX))
    ret_ins = Instruction.create(Code.RETNQ)
    mem_rdx = MemoryOperand(R.RDX)
    mem_rcx = MemoryOperand(R.RCX)

    # All 20 register-only AVX forms must reject by default (allow_memory=False) and lift under --mem-ops.
    twenty_forms = [
        ("vblendpd_mem", (ld_ymm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VBLENDPD_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, mem_rdx, 0xA), ret_ins)),
        ("vblendps_mem", (ld_ymm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VBLENDPS_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, mem_rdx, 0x5A), ret_ins)),
        ("vblendvpd_mem", (ld_ymm1, ld_ymm3, Instruction.create_reg_reg_mem_reg(
            Code.VEX_VBLENDVPD_YMM_YMM_YMMM256_YMM, R.YMM0, R.YMM1, mem_rdx, R.YMM3), ret_ins)),
        ("vblendvps_mem", (ld_ymm1, ld_ymm3, Instruction.create_reg_reg_mem_reg(
            Code.VEX_VBLENDVPS_YMM_YMM_YMMM256_YMM, R.YMM0, R.YMM1, mem_rdx, R.YMM3), ret_ins)),
        ("vextractf128_mem", (ld_ymm0, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTF128_XMMM128_YMM_IMM8, MemoryOperand(R.R8), R.YMM0, 1), ret_ins)),
        ("vinsertf128_mem", (ld_ymm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VINSERTF128_YMM_YMM_XMMM128_IMM8, R.YMM0, R.YMM1, mem_rdx, 1), ret_ins)),
        ("vextractps_mem", (ld_xmm1, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTPS_RM32_XMM_IMM8, MemoryOperand(R.R8), R.XMM1, 2), ret_ins)),
        ("vinsertps_mem", (ld_xmm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VINSERTPS_XMM_XMM_XMMM32_IMM8, R.XMM0, R.XMM1, mem_rdx, 0xD4), ret_ins)),
        ("vperm2f128_mem", (ld_ymm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VPERM2F128_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, mem_rdx, 0x20), ret_ins)),
        ("vpermilpd_imm_mem", (Instruction.create_reg_mem_u32(
            Code.VEX_VPERMILPD_YMM_YMMM256_IMM8, R.YMM0, mem_rcx, 0xA), ret_ins)),
        ("vpermilps_imm_mem", (Instruction.create_reg_mem_u32(
            Code.VEX_VPERMILPS_YMM_YMMM256_IMM8, R.YMM0, mem_rcx, 0x93), ret_ins)),
        ("vshufpd_mem", (ld_ymm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VSHUFPD_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, mem_rdx, 9), ret_ins)),
        ("vshufps_mem", (ld_ymm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VSHUFPS_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, mem_rdx, 0x4E), ret_ins)),
        ("vunpcklpd_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VUNPCKLPD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem_rdx), ret_ins)),
        ("vunpcklps_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VUNPCKLPS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem_rdx), ret_ins)),
        ("vunpckhpd_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VUNPCKHPD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem_rdx), ret_ins)),
        ("vunpckhps_mem", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VUNPCKHPS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, mem_rdx), ret_ins)),
        ("vmovddup_xmm_mem", (Instruction.create_reg_mem(
            Code.VEX_VMOVDDUP_XMM_XMMM64, R.XMM0, mem_rcx), ret_ins)),
        ("vmovshdup_mem", (Instruction.create_reg_mem(
            Code.VEX_VMOVSHDUP_YMM_YMMM256, R.YMM0, mem_rcx), ret_ins)),
        ("vmovsldup_mem", (Instruction.create_reg_mem(
            Code.VEX_VMOVSLDUP_YMM_YMMM256, R.YMM0, mem_rcx), ret_ins)),
    ]
    for label, seq in twenty_forms:
        raw = encode_seq(*seq)
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if "unsupported memory operand for register-domain AVX instruction" not in str(exc):
                raise AssertionError(f"{label} default reject: got {exc!r}") from exc
        else:
            raise AssertionError(f"{label} unexpectedly accepted with allow_memory=False")
        src, _ = lift_code(label, 0x1000, raw, allow_memory=True)
        assert f"lifted_{label}" in src, label
        if label == "vmovddup_xmm_mem":
            assert "load_vec64(" in src and "load128(" not in src, src
        if label == "vinsertps_mem":
            assert "load_vec32(" in src and "0x14u" in src, src  # 0xD4 & 0x3F == 0x14

    evex_blend = Instruction.create_reg_reg_mem(
        Code.EVEX_VBLENDMPD_YMM_K1Z_YMM_YMMM256B64, R.YMM0, R.YMM1, mem_rdx)
    evex_blend.op_mask = R.K1
    evex_permil = Instruction.create_reg_mem_u32(
        Code.EVEX_VPERMILPD_YMM_K1Z_YMMM256B64_IMM8, R.YMM0, mem_rcx, 0xA)
    evex_permil.op_mask = R.K1

    mem_rejects = [
        ("extractf128_store_rcx", (ld_ymm0, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTF128_XMMM128_YMM_IMM8, MemoryOperand(R.RCX), R.YMM0, 0), ret_ins),
         "unsupported memory operand"),
        ("extractf128_store_rdx", (ld_ymm0, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTF128_XMMM128_YMM_IMM8, MemoryOperand(R.RDX), R.YMM0, 1), ret_ins),
         "unsupported memory operand"),
        ("extractps_store_rcx", (ld_xmm1, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTPS_RM32_XMM_IMM8, MemoryOperand(R.RCX), R.XMM1, 1), ret_ins),
         "unsupported memory operand"),
        ("extractps_store_rdx", (ld_xmm1, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTPS_RM32_XMM_IMM8, MemoryOperand(R.RDX), R.XMM1, 0), ret_ins),
         "unsupported memory operand"),
        ("rip_rel_mem", (Instruction.create_reg_mem(
            Code.VEX_VMOVDDUP_YMM_YMMM256, R.YMM0, MemoryOperand(R.RIP, displ=0x1010, displ_size=1)), ret_ins),
         "unsupported memory operand"),
        ("indexed_mem", (ld_ymm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VSHUFPS_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, MemoryOperand(R.RDX, R.RCX, 4), 0xE4), ret_ins),
         "unsupported memory operand"),
        ("addr32_mem", (Instruction.create_reg_mem(
            Code.VEX_VMOVSHDUP_YMM_YMMM256, R.YMM0, MemoryOperand(R.ECX)), ret_ins),
         "unsupported memory operand"),
        ("non_arg_base_rax", (ld_ymm1, Instruction.create_reg_reg_mem(
            Code.VEX_VUNPCKLPD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, MemoryOperand(R.RAX)), ret_ins),
         "unsupported memory operand"),
        ("evex_blend", (ld_ymm1, evex_blend, ret_ins), "unsupported EVEX instruction"),
        ("evex_permil", (evex_permil, ret_ins), "unsupported EVEX instruction"),
        ("nonvol_ymm6", (Instruction.create_reg_mem(
            Code.VEX_VMOVSLDUP_YMM_YMMM256, R.YMM6, mem_rcx), ret_ins),
         "nonvolatile vector register"),
        ("nonvol_ymm15", (ld_ymm0, Instruction.create_mem_reg_u32(
            Code.VEX_VEXTRACTF128_XMMM128_YMM_IMM8, MemoryOperand(R.R8), R.YMM15, 0), ret_ins),
         "nonvolatile vector register"),
        ("undef_ymm3", (Instruction.create_reg_reg_mem_u32(
            Code.VEX_VSHUFPD_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM3, mem_rdx, 0), ret_ins),
         "undefined vector register ymm3"),
        ("missing_ret", (Instruction.create_reg_mem(
            Code.VEX_VMOVDDUP_XMM_XMMM64, R.XMM0, mem_rcx),),
         "missing terminal ret"),
    ]
    for label, seq, reason in mem_rejects:
        try:
            lift_code(label, 0x1000, encode_seq(*seq), allow_memory=True)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: accepted under allow_memory=True")

    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_vblendps_ymm_mem")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12)
                       if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry
    with TemporaryDirectory() as folder:
        patched = Path(folder) / "patched.dll"
        for label, seq, reason in mem_rejects[:4]:
            patch = encode_seq(*seq)
            assert len(patch) <= max_len
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched.write_bytes(damaged)
            try:
                translate(patched, ["op_vblendps_ymm_mem"], allow_memory=True)
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"PE {label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"PE {label}: accepted rejected patch")

    print(f"translator validated: 20 default mem rejections, 20 --mem-ops acceptances, "
          f"{len(mem_rejects)} unit + 4 PE --mem-ops rejections")


def check_page_edge_and_discarded(native, lifted):
    """Verify exact m32/m64 load widths (no overread across a guard page) and discarded-read ordering."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualProtect.argtypes = [
        wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
    ]
    k32.VirtualProtect.restype = wintypes.BOOL
    k32.VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
    k32.VirtualFree.restype = wintypes.BOOL

    base = k32.VirtualAlloc(None, 8192, 0x3000, 0x04)  # MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE
    assert base, "VirtualAlloc failed"
    try:
        old = wintypes.DWORD(0)
        assert k32.VirtualProtect(base + 4096, 4096, 0x01, ctypes.byref(old))  # PAGE_NOACCESS
        a_buf = ctypes.create_string_buffer(bytes(range(1, 33)), 32)
        b_buf = ctypes.create_string_buffer(bytes(range(101, 133)), 32)
        out_buf = ctypes.create_string_buffer(32)

        # 1) m64 vmovddup: [rcx+3] sits at [page_end - 8 .. page_end - 1].
        src8 = b"ABCDEFGH"
        ctypes.memmove(base + 4096 - 8, src8, 8)
        for label, funcs in (("native", native), ("lifted", lifted)):
            ctypes.memset(out_buf, 0x5A, 32)
            funcs["op_vmovddup_xmm_mem"](base + 4096 - 11, ctypes.addressof(b_buf), ctypes.addressof(out_buf))
            assert out_buf.raw == src8 * 2 + b"\x00" * 16, (label, out_buf.raw)

        # 2) m32 vinsertps in op_discarded_mem_read: [rdx] sits at [page_end - 4 .. page_end - 1],
        # plus NULL rdx fault must occur after storing low 16 bytes.
        ctypes.memmove(base + 4096 - 4, b"\x11\x22\x33\x44", 4)
        for label, funcs in (("native", native), ("lifted", lifted)):
            fn = funcs["op_discarded_mem_read"]
            ctypes.memset(out_buf, 0x5A, 32)
            fn(ctypes.addressof(a_buf), base + 4096 - 4, ctypes.addressof(out_buf))
            assert out_buf.raw == a_buf.raw, (label, out_buf.raw)

            ctypes.memset(out_buf, 0x5A, 32)
            try:
                fn(ctypes.addressof(a_buf), 0, ctypes.addressof(out_buf))
            except OSError as exc:
                assert "access violation reading" in str(exc).lower(), (label, exc)
            else:
                raise AssertionError(f"{label} op_discarded_mem_read discarded faulting m32 read")
            assert out_buf.raw == a_buf.raw[:16] + b"\x5A" * 16, (label, out_buf.raw)
    finally:
        k32.VirtualFree(base, 0, 0x8000)


def check_differential(original_path, lifted_path, altered_lifted_path):
    _, native = bind(original_path, "")
    _, lifted = bind(lifted_path, "lifted_")
    _, altered = bind(altered_lifted_path, "lifted_")
    cases = build_buffers()

    total = 0
    for name in EXPORTS:
        for idx, (a, b, c) in enumerate(cases):
            want = reference(name, a, b, c)
            require_equal("native", name, idx, invoke(native[name], a, b, c), want)
            require_equal("lifted", name, idx, invoke(lifted[name], a, b, c), want)
            total += 1

    altered_diffs = {}
    for name in ("op_vblendpd_ymm_mem", "op_vinsertps_mem", "op_vperm2f128_mem"):
        diffs = 0
        for idx, (a, b, c) in enumerate(cases):
            want_alt = reference(name, a, b, c, altered=True)
            want_orig = reference(name, a, b, c, altered=False)
            got_alt = invoke(altered[name], a, b, c)
            require_equal("altered", name, idx, got_alt, want_alt)
            diffs += got_alt != want_orig
        assert diffs > 0, f"altered {name} did not diverge"
        altered_diffs[name] = diffs

    check_page_edge_and_discarded(native, lifted)
    print(f"{total} native/lifted/reference cases across {len(EXPORTS)} exports matched bit-for-bit; "
          f"page-edge m32/m64 no-overread + discarded-read fault + altered controls ({altered_diffs}) passed")


def record_covered_forms(original_path):
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    formatter = Formatter(FormatterSyntax.INTEL)
    forms = set()
    with pefile.PE(str(original_path), fast_load=True) as pe:
        for name in EXPORTS:
            entry = export_rva(pe, name)
            raw = bounded_code(pe, entry)
            for ins in Decoder(64, raw, ip=entry):
                forms.add(formatter.format_mnemonic(ins).lower())
    out_path = Path(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(word + "\n" for word in sorted(forms)), encoding="utf-8")


def main():
    if os.environ.get("FORM_COVERAGE_RECORD"):
        Path(os.environ["FORM_COVERAGE_RECORD"]).unlink(missing_ok=True)

    original_path, lifted_path, altered_lifted_path = (Path(p) for p in sys.argv[1:4])
    check_translator(original_path)

    lifted_lib = ctypes.CDLL(str(lifted_path.resolve()))
    if not host_has_avx(lifted_lib):
        print("Host lacks CPUID.01H:ECX[28] + XGETBV; skipping native differential check")
        sys.exit(77)

    check_differential(original_path, lifted_path, altered_lifted_path)
    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
