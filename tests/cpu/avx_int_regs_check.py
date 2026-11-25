"""Differential checks, mutation controls and rejection controls for AVX packed-integer AOT."""

import ctypes
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Encoder, Instruction, MemoryOperand, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_avx_int_aot import lift_code, translate
from cpu_leaf_aot import export_rva

MASK8 = 0xff
MASK16 = 0xffff
MASK32 = 0xffffffff
MASK64 = 0xffffffffffffffff
GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))
FILL = b"\x5A" * 32
FILL_192 = b"\x5A" * 192
ZERO16 = b"\x00" * 16

EXPORTS = (
    "op_pabsb", "op_pabsw", "op_pabsd",
    "op_packsswb", "op_packssdw", "op_packusdw",
    "op_paddb", "op_paddq",
    "op_paddsb", "op_paddusb", "op_paddusw",
    "op_psubb", "op_psubd", "op_psubq",
    "op_psubsb", "op_psubsw", "op_psubusb", "op_psubusw",
    "op_palignr_0", "op_palignr_5", "op_palignr_16", "op_palignr_23", "op_palignr_32",
    "op_pblendvb", "op_pblendw", "op_pblendw_all", "op_pblendw_none",
    "op_pcmpeqb", "op_pcmpeqw", "op_pcmpeqd", "op_pcmpeqq",
    "op_pcmpgtb", "op_pcmpgtw", "op_pcmpgtd", "op_pcmpgtq",
    "op_pmaxsb", "op_pmaxsd", "op_pmaxub", "op_pmaxuw", "op_pmaxud",
    "op_pminsb", "op_pminsd", "op_pminub", "op_pminuw", "op_pminud",
    "op_phaddw", "op_phaddd", "op_phaddsw", "op_phsubw", "op_phsubd", "op_phsubsw",
    "op_phminposuw",
    "op_pmaddwd", "op_pmuldq", "op_pmuludq",
    "op_pmulhrsw", "op_pmulhuw", "op_pmulld",
    "op_psadbw", "op_mpsadbw_0", "op_mpsadbw_3", "op_mpsadbw_7_ignored",
    "op_pshufhw", "op_pshuflw",
    "op_psignb", "op_psignw", "op_psignd",
    "op_pmovsxbw", "op_pmovsxbd", "op_pmovsxbq", "op_pmovsxwd", "op_pmovsxwq", "op_pmovsxdq",
    "op_pmovzxbw", "op_pmovzxbd", "op_pmovzxbq", "op_pmovzxwd", "op_pmovzxwq", "op_pmovzxdq",
    "op_punpckldq", "op_punpckhdq", "op_punpcklqdq", "op_punpckhqdq",
    "op_pslld_xmm", "op_psllq_xmm",
    "op_psrld_xmm", "op_psrlq_xmm",
    "op_pslld_imm", "op_pslld_imm_ovf",
    "op_psllq_imm", "op_psllq_imm_ovf",
    "op_psrld_imm", "op_psrld_imm_ovf", "op_psrlq_imm", "op_psrlq_imm_ovf",
    "op_pmovmskb",
    "op_pextrb_reg", "op_pextrb_mem", "op_pextrw_reg", "op_pextrw_mem",
    "op_pextrd_reg", "op_pextrd_mem", "op_pextrq_reg", "op_pextrq_mem",
    "op_pinsrb_reg", "op_pinsrb_mem", "op_pinsrw_reg", "op_pinsrw_mem",
    "op_pinsrd_reg", "op_pinsrd_mem", "op_pinsrq_reg", "op_pinsrq_mem",
    "op_zeroupper", "op_pipeline", "op_movdqu_copy128", "op_mov_store32",
    "op_movdqu_regcopy", "op_movdqu_store128",
    "op_palignr_69", "op_mpsadbw_6", "op_pshuflw_1b", "op_psllq_imm_129",
    "op_pextrw_reg_2", "op_pextrd_reg_2", "op_pinsrb_reg_upper", "op_pinsrw_reg_3",
    "op_zeroupper_ymm1", "op_movdqu_load128",
    "op_palignr_255", "op_pmovsxbq_wide_src", "op_zeroupper_ymm2",
)

ALL_EXPORTS = EXPORTS + ("op_zeroall",)


# --- Independent Python ISA reference model ---

def u8s(buf): return struct.unpack("<16B", buf[:16])
def s8s(buf): return struct.unpack("<16b", buf[:16])
def u16s(buf): return struct.unpack("<8H", buf[:16])
def s16s(buf): return struct.unpack("<8h", buf[:16])
def u32s(buf): return struct.unpack("<4I", buf[:16])
def s32s(buf): return struct.unpack("<4i", buf[:16])
def u64s(buf): return struct.unpack("<2Q", buf[:16])
def s64s(buf): return struct.unpack("<2q", buf[:16])


def pack_u8(vals): return struct.pack("<16B", *(v & MASK8 for v in vals)) + ZERO16
def pack_s8(vals): return struct.pack("<16b", *vals) + ZERO16
def pack_u16(vals): return struct.pack("<8H", *(v & MASK16 for v in vals)) + ZERO16
def pack_s16(vals): return struct.pack("<8h", *vals) + ZERO16
def pack_u32(vals): return struct.pack("<4I", *(v & MASK32 for v in vals)) + ZERO16
def pack_s32(vals): return struct.pack("<4i", *vals) + ZERO16
def pack_u64(vals): return struct.pack("<2Q", *(v & MASK64 for v in vals)) + ZERO16
def pack_s64(vals): return struct.pack("<2q", *vals) + ZERO16


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def ref_pabs(a, bits):
    if bits == 8:
        return pack_u8(abs(x) & MASK8 for x in s8s(a))
    if bits == 16:
        return pack_u16(abs(x) & MASK16 for x in s16s(a))
    return pack_u32(abs(x) & MASK32 for x in s32s(a))


def ref_palignr(a, b, imm8):
    cat = b[:16] + a[:16]
    if imm8 >= 32:
        return b"\x00" * 32
    shifted = cat[imm8:] + b"\x00" * imm8
    return shifted[:16] + ZERO16


def ref_pblendvb(a, b, mask):
    return pack_u8(b[i] if (mask[i] & 0x80) else a[i] for i in range(16))


def ref_pblendw(a, b, imm8):
    wa, wb = u16s(a), u16s(b)
    return pack_u16(wb[i] if ((imm8 >> i) & 1) else wa[i] for i in range(8))


def ref_phminposuw(a):
    wa = u16s(a)
    best, idx = wa[0], 0
    for i in range(1, 8):
        if wa[i] < best:
            best, idx = wa[i], i
    return pack_u16((best, idx, 0, 0, 0, 0, 0, 0))


def ref_pmaddwd(a, b):
    sa, sb = s16s(a), s16s(b)
    return pack_u32((sa[2 * i] * sb[2 * i] + sa[2 * i + 1] * sb[2 * i + 1]) & MASK32 for i in range(4))


def ref_pmulhrsw(a, b):
    sa, sb = s16s(a), s16s(b)
    return pack_u16((((sa[i] * sb[i]) >> 14) + 1) >> 1 for i in range(8))


def ref_psadbw(a, b):
    ua, ub = u8s(a), u8s(b)
    s0 = sum(abs(ua[i] - ub[i]) for i in range(8))
    s1 = sum(abs(ua[i] - ub[i]) for i in range(8, 16))
    return pack_u64((s0, s1))


def ref_mpsadbw(a, b, imm8):
    ua, ub = u8s(a), u8s(b)
    a_off = ((imm8 >> 2) & 1) * 4
    b_off = (imm8 & 3) * 4
    return pack_u16(sum(abs(ua[a_off + k + j] - ub[b_off + j]) for j in range(4)) for k in range(8))


def ref_pshufhw(a, imm8):
    wa = u16s(a)
    return pack_u16(list(wa[:4]) + [wa[4 + ((imm8 >> (2 * i)) & 3)] for i in range(4)])


def ref_pshuflw(a, imm8):
    wa = u16s(a)
    return pack_u16([wa[(imm8 >> (2 * i)) & 3] for i in range(4)] + list(wa[4:]))


def ref_psign(a, b, bits):
    if bits == 8:
        sa, sb = s8s(a), s8s(b)
        return pack_u8((-x if c < 0 else (0 if c == 0 else x)) & MASK8 for x, c in zip(sa, sb))
    if bits == 16:
        sa, sb = s16s(a), s16s(b)
        return pack_u16((-x if c < 0 else (0 if c == 0 else x)) & MASK16 for x, c in zip(sa, sb))
    sa, sb = s32s(a), s32s(b)
    return pack_u32((-x if c < 0 else (0 if c == 0 else x)) & MASK32 for x, c in zip(sa, sb))


def ref_shift(a, count, kind, bits):
    if kind == "sll":
        if bits == 32:
            return pack_u32(0 if count >= 32 else ((x << count) & MASK32) for x in u32s(a))
        return pack_u64(0 if count >= 64 else ((x << count) & MASK64) for x in u64s(a))
    if bits == 32:
        return pack_u32(0 if count >= 32 else (x >> count) for x in u32s(a))
    return pack_u64(0 if count >= 64 else (x >> count) for x in u64s(a))


def ref_pmovmskb_val(a):
    return sum((1 << i) for i in range(16) if (a[i] & 0x80))


def ref_pinsr(a, val, width, imm8):
    out = bytearray(a[:16])
    lanes = 16 // width
    idx = imm8 & (lanes - 1)
    fmt = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[width]
    mask = (1 << (8 * width)) - 1
    struct.pack_into(fmt, out, idx * width, val & mask)
    return bytes(out) + ZERO16


def ref_pipeline(a, b):
    xmm1 = a[:16] + ZERO16
    xmm2 = b[:16] + ZERO16
    xmm3 = pack_s8(clamp(x + y, -128, 127) for x, y in zip(s8s(xmm1), s8s(xmm2)))
    xmm4 = ref_psign(xmm3, xmm2, 8)
    xmm5 = ref_pshuflw(xmm4, 0x4E)
    xmm3 = pack_u8(0xff if x > y else 0 for x, y in zip(s8s(xmm5), s8s(xmm2)))
    xmm4 = ref_pblendvb(xmm1, xmm5, xmm3)
    eax = ref_pmovmskb_val(xmm4)
    xmm4 = ref_pinsr(xmm4, eax, 4, 3)
    xmm5 = ref_phminposuw(xmm4)
    return ref_palignr(xmm4, xmm5, 6)


def reference(name, a, b, altered=False):
    # Unary abs
    if name == "op_pabsb": return ref_pabs(a, 8)
    if name == "op_pabsw": return ref_pabs(a, 16)
    if name == "op_pabsd": return ref_pabs(a, 32)
    # Pack
    if name == "op_packsswb": return pack_s8([clamp(x, -128, 127) for x in s16s(a)] + [clamp(x, -128, 127) for x in s16s(b)])
    if name == "op_packssdw": return pack_s16([clamp(x, -32768, 32767) for x in s32s(a)] + [clamp(x, -32768, 32767) for x in s32s(b)])
    if name == "op_packusdw": return pack_u16([clamp(x, 0, 65535) for x in s32s(a)] + [clamp(x, 0, 65535) for x in s32s(b)])
    # Add
    if name == "op_paddb": return pack_u8(x + y for x, y in zip(u8s(a), u8s(b)))
    if name == "op_paddq": return pack_u64(x + y for x, y in zip(u64s(a), u64s(b)))
    if name == "op_paddsb": return pack_s8(clamp(x + y, -128, 127) for x, y in zip(s8s(a), s8s(b)))
    if name == "op_paddusb": return pack_u8(min(x + y, 255) for x, y in zip(u8s(a), u8s(b)))
    if name == "op_paddusw": return pack_u16(min(x + y, 65535) for x, y in zip(u16s(a), u16s(b)))
    # Sub
    if name == "op_psubb": return pack_u8(x - y for x, y in zip(u8s(a), u8s(b)))
    if name == "op_psubd": return pack_u32(x - y for x, y in zip(u32s(a), u32s(b)))
    if name == "op_psubq": return pack_u64(x - y for x, y in zip(u64s(a), u64s(b)))
    if name == "op_psubsb": return pack_s8(clamp(x - y, -128, 127) for x, y in zip(s8s(a), s8s(b)))
    if name == "op_psubsw": return pack_s16(clamp(x - y, -32768, 32767) for x, y in zip(s16s(a), s16s(b)))
    if name == "op_psubusb": return pack_u8(max(x - y, 0) for x, y in zip(u8s(a), u8s(b)))
    if name == "op_psubusw": return pack_u16(max(x - y, 0) for x, y in zip(u16s(a), u16s(b)))
    # Align & blend
    if name == "op_palignr_0": return ref_palignr(a, b, 0)
    if name == "op_palignr_5": return ref_palignr(a, b, 11 if altered else 5)
    if name == "op_palignr_16": return ref_palignr(a, b, 16)
    if name == "op_palignr_23": return ref_palignr(a, b, 23)
    if name == "op_palignr_32": return ref_palignr(a, b, 32)
    if name == "op_pblendvb": return ref_pblendvb(a[:16], b[:16], a[16:32])
    if name == "op_pblendw": return ref_pblendw(a, b, 0x55 if altered else 0xAA)
    if name == "op_pblendw_all": return ref_pblendw(a, b, 0xFF)
    if name == "op_pblendw_none": return ref_pblendw(a, b, 0x00)
    # Compare
    if name == "op_pcmpeqb": return pack_u8(MASK8 if x == y else 0 for x, y in zip(u8s(a), u8s(b)))
    if name == "op_pcmpeqw": return pack_u16(MASK16 if x == y else 0 for x, y in zip(u16s(a), u16s(b)))
    if name == "op_pcmpeqd": return pack_u32(MASK32 if x == y else 0 for x, y in zip(u32s(a), u32s(b)))
    if name == "op_pcmpeqq": return pack_u64(MASK64 if x == y else 0 for x, y in zip(u64s(a), u64s(b)))
    if name == "op_pcmpgtb": return pack_u8(MASK8 if x > y else 0 for x, y in zip(s8s(a), s8s(b)))
    if name == "op_pcmpgtw": return pack_u16(MASK16 if x > y else 0 for x, y in zip(s16s(a), s16s(b)))
    if name == "op_pcmpgtd": return pack_u32(MASK32 if x > y else 0 for x, y in zip(s32s(a), s32s(b)))
    if name == "op_pcmpgtq": return pack_u64(MASK64 if x > y else 0 for x, y in zip(s64s(a), s64s(b)))
    # Min / Max
    if name == "op_pmaxsb": return pack_s8(max(x, y) for x, y in zip(s8s(a), s8s(b)))
    if name == "op_pmaxsd": return pack_s32(max(x, y) for x, y in zip(s32s(a), s32s(b)))
    if name == "op_pmaxub": return pack_u8(max(x, y) for x, y in zip(u8s(a), u8s(b)))
    if name == "op_pmaxuw": return pack_u16(max(x, y) for x, y in zip(u16s(a), u16s(b)))
    if name == "op_pmaxud": return pack_u32(max(x, y) for x, y in zip(u32s(a), u32s(b)))
    if name == "op_pminsb": return pack_s8(min(x, y) for x, y in zip(s8s(a), s8s(b)))
    if name == "op_pminsd": return pack_s32(min(x, y) for x, y in zip(s32s(a), s32s(b)))
    if name == "op_pminub": return pack_u8(min(x, y) for x, y in zip(u8s(a), u8s(b)))
    if name == "op_pminuw": return pack_u16(min(x, y) for x, y in zip(u16s(a), u16s(b)))
    if name == "op_pminud": return pack_u32(min(x, y) for x, y in zip(u32s(a), u32s(b)))
    # Horizontal
    if name == "op_phaddw":
        wa, wb = u16s(a), u16s(b)
        return pack_u16([wa[2 * i] + wa[2 * i + 1] for i in range(4)] + [wb[2 * i] + wb[2 * i + 1] for i in range(4)])
    if name == "op_phaddd":
        da, db = u32s(a), u32s(b)
        return pack_u32([da[2 * i] + da[2 * i + 1] for i in range(2)] + [db[2 * i] + db[2 * i + 1] for i in range(2)])
    if name == "op_phaddsw":
        sa, sb = s16s(a), s16s(b)
        return pack_s16([clamp(sa[2 * i] + sa[2 * i + 1], -32768, 32767) for i in range(4)]
                        + [clamp(sb[2 * i] + sb[2 * i + 1], -32768, 32767) for i in range(4)])
    if name == "op_phsubw":
        wa, wb = u16s(a), u16s(b)
        return pack_u16([wa[2 * i] - wa[2 * i + 1] for i in range(4)] + [wb[2 * i] - wb[2 * i + 1] for i in range(4)])
    if name == "op_phsubd":
        da, db = u32s(a), u32s(b)
        return pack_u32([da[2 * i] - da[2 * i + 1] for i in range(2)] + [db[2 * i] - db[2 * i + 1] for i in range(2)])
    if name == "op_phsubsw":
        sa, sb = s16s(a), s16s(b)
        return pack_s16([clamp(sa[2 * i] - sa[2 * i + 1], -32768, 32767) for i in range(4)]
                        + [clamp(sb[2 * i] - sb[2 * i + 1], -32768, 32767) for i in range(4)])
    if name == "op_phminposuw": return ref_phminposuw(a)
    # Multiply
    if name == "op_pmaddwd": return ref_pmaddwd(a, b)
    if name == "op_pmuldq":
        sa, sb = s32s(a), s32s(b)
        return pack_s64((sa[0] * sb[0], sa[2] * sb[2]))
    if name == "op_pmuludq":
        ua, ub = u32s(a), u32s(b)
        return pack_u64((ua[0] * ub[0], ua[2] * ub[2]))
    if name == "op_pmulhrsw": return ref_pmulhrsw(a, b)
    if name == "op_pmulhuw": return pack_u16((x * y) >> 16 for x, y in zip(u16s(a), u16s(b)))
    if name == "op_pmulld": return pack_u32(x * y for x, y in zip(u32s(a), u32s(b)))
    # SAD
    if name == "op_psadbw": return ref_psadbw(a, b)
    if name == "op_mpsadbw_0": return ref_mpsadbw(a, b, 0x05 if altered else 0x00)
    if name == "op_mpsadbw_3": return ref_mpsadbw(a, b, 0x03)
    if name == "op_mpsadbw_7_ignored": return ref_mpsadbw(a, b, 0xFF)
    # Shuffles
    if name == "op_pshufhw": return ref_pshufhw(a, 0x1B if altered else 0x93)
    if name == "op_pshuflw": return ref_pshuflw(a, 0x4E)
    # Sign
    if name == "op_psignb": return ref_psign(a, b, 8)
    if name == "op_psignw": return ref_psign(a, b, 16)
    if name == "op_psignd": return ref_psign(a, b, 32)
    # Extend
    if name == "op_pmovsxbw": return pack_s16(s8s(a)[:8])
    if name == "op_pmovsxbd": return pack_s32(s8s(a)[:4])
    if name == "op_pmovsxbq": return pack_s64(s8s(a)[:2])
    if name == "op_pmovsxwd": return pack_s32(s16s(a)[:4])
    if name == "op_pmovsxwq": return pack_s64(s16s(a)[:2])
    if name == "op_pmovsxdq": return pack_s64(s32s(a)[:2])
    if name == "op_pmovzxbw": return pack_u16(u8s(a)[:8])
    if name == "op_pmovzxbd": return pack_u32(u8s(a)[:4])
    if name == "op_pmovzxbq": return pack_u64(u8s(a)[:2])
    if name == "op_pmovzxwd": return pack_u32(u16s(a)[:4])
    if name == "op_pmovzxwq": return pack_u64(u16s(a)[:2])
    if name == "op_pmovzxdq": return pack_u64(u32s(a)[:2])
    # Unpack
    if name == "op_punpckldq":
        ua, ub = u32s(a), u32s(b)
        return pack_u32((ua[0], ub[0], ua[1], ub[1]))
    if name == "op_punpckhdq":
        ua, ub = u32s(a), u32s(b)
        return pack_u32((ua[2], ub[2], ua[3], ub[3]))
    if name == "op_punpcklqdq":
        ua, ub = u64s(a), u64s(b)
        return pack_u64((ua[0], ub[0]))
    if name == "op_punpckhqdq":
        ua, ub = u64s(a), u64s(b)
        return pack_u64((ua[1], ub[1]))
    # Shifts (XMM count reads bits 63:0; bits 127:64 are ignored per SDM)
    cnt = struct.unpack_from("<Q", b, 0)[0]
    if name == "op_pslld_xmm": return ref_shift(a, cnt, "sll", 32)
    if name == "op_psllq_xmm": return ref_shift(a, cnt, "sll", 64)
    if name == "op_psrld_xmm": return ref_shift(a, cnt, "srl", 32)
    if name == "op_psrlq_xmm": return ref_shift(a, cnt, "srl", 64)
    # Shifts (imm8)
    if name == "op_pslld_imm": return ref_shift(a, 7, "sll", 32)
    if name == "op_pslld_imm_ovf": return ref_shift(a, 32, "sll", 32)
    if name == "op_psllq_imm": return ref_shift(a, 13, "sll", 64)
    if name == "op_psllq_imm_ovf": return ref_shift(a, 64, "sll", 64)
    if name == "op_psrld_imm": return ref_shift(a, 7, "srl", 32)
    if name == "op_psrld_imm_ovf": return ref_shift(a, 32, "srl", 32)
    if name == "op_psrlq_imm": return ref_shift(a, 13, "srl", 64)
    if name == "op_psrlq_imm_ovf": return ref_shift(a, 64, "srl", 64)
    # Movemask / Extract / Insert
    if name == "op_pmovmskb":
        return struct.pack("<Q", ref_pmovmskb_val(a)) + FILL[8:]
    if name == "op_pextrb_reg":
        idx = 3 if altered else 13
        return struct.pack("<Q", a[idx]) + FILL[8:]
    if name == "op_pextrb_mem":
        out = bytearray(FILL)
        out[3] = a[0xF7 & 15]
        return bytes(out)
    if name == "op_pextrw_reg":
        return struct.pack("<Q", u16s(a)[5]) + FILL[8:]
    if name == "op_pextrw_mem":
        out = bytearray(FILL)
        struct.pack_into("<H", out, 4, u16s(a)[0xFE & 7])
        return bytes(out)
    if name == "op_pextrd_reg":
        return struct.pack("<Q", u32s(a)[3]) + FILL[8:]
    if name == "op_pextrd_mem":
        out = bytearray(FILL)
        struct.pack_into("<I", out, 8, u32s(a)[0xFD & 3])
        return bytes(out)
    if name == "op_pextrq_reg":
        return struct.pack("<Q", u64s(a)[1]) + FILL[8:]
    if name == "op_pextrq_mem":
        out = bytearray(FILL)
        struct.pack_into("<Q", out, 8, u64s(a)[0xFE & 1])
        return bytes(out)
    if name == "op_pinsrb_reg":
        return ref_pinsr(a, struct.unpack_from("<I", b, 0)[0], 1, 11)
    if name == "op_pinsrb_mem":
        return ref_pinsr(a, b[3], 1, 0xF5)
    if name == "op_pinsrw_reg":
        return ref_pinsr(a, struct.unpack_from("<I", b, 0)[0], 2, 6)
    if name == "op_pinsrw_mem":
        return ref_pinsr(a, struct.unpack_from("<H", b, 4)[0], 2, 0xFA)
    if name == "op_pinsrd_reg":
        return ref_pinsr(a, struct.unpack_from("<I", b, 0)[0], 4, 2)
    if name == "op_pinsrd_mem":
        return ref_pinsr(a, struct.unpack_from("<I", b, 8)[0], 4, 0xFF)
    if name == "op_pinsrq_reg":
        return ref_pinsr(a, struct.unpack_from("<Q", b, 0)[0], 8, 1)
    if name == "op_pinsrq_mem":
        return ref_pinsr(a, struct.unpack_from("<Q", b, 8)[0], 8, 0xFE)
    # State & pipeline
    if name == "op_zeroupper":
        return a[:16] + ZERO16
    if name == "op_pipeline":
        return ref_pipeline(a, b)
    if name == "op_movdqu_copy128":
        return a[:16] + ZERO16
    if name == "op_mov_store32":
        return b[:4] + ZERO16[:4] + FILL[8:16] + b[:4] + FILL[20:]
    if name == "op_movdqu_regcopy":
        return a[:32]
    if name == "op_movdqu_store128":
        return a[:16] + FILL[16:]
    # Mutation-sensitive forms
    if name == "op_palignr_69": return ref_palignr(a, b, 69)
    if name == "op_mpsadbw_6": return ref_mpsadbw(a, b, 0x06)
    if name == "op_pshuflw_1b": return ref_pshuflw(a, 0x1B)
    if name == "op_psllq_imm_129": return ref_shift(a, 0x81, "sll", 64)
    if name == "op_pextrw_reg_2":
        return struct.pack("<Q", u16s(a)[2]) + FILL[8:]
    if name == "op_pextrd_reg_2":
        return struct.pack("<Q", u32s(a)[2]) + FILL[8:]
    if name == "op_pinsrb_reg_upper":
        return ref_pinsr(a, struct.unpack_from("<I", b, 0)[0], 1, 4)
    if name == "op_pinsrw_reg_3":
        return ref_pinsr(a, struct.unpack_from("<I", b, 0)[0], 2, 3)
    if name == "op_zeroupper_ymm1":
        return a[:16] + ZERO16
    if name == "op_movdqu_load128":
        return a[:16] + ZERO16
    if name == "op_palignr_255": return ref_palignr(a, b, 255)
    if name == "op_pmovsxbq_wide_src": return pack_s64(s8s(a)[:2])
    if name == "op_zeroupper_ymm2":
        return a[:16] + ZERO16
    raise ValueError(f"unknown export {name}")


# --- Input vectors ---

def pad32(lo16, hi16=None):
    return lo16 + (bytes(range(0xE0, 0xF0)) if hi16 is None else hi16)


def general_pairs():
    rng = random.Random(0xA71A)
    distinct_a = bytes(range(0x01, 0x21))
    distinct_b = bytes(range(0x81, 0xA1))
    edges_u8_a = bytes((0x00, 0x01, 0x7E, 0x7F, 0x80, 0x81, 0xFE, 0xFF,
                        0x7F, 0x80, 0xFF, 0x00, 0x55, 0xAA, 0x12, 0x34))
    edges_u8_b = bytes((0x00, 0xFF, 0x01, 0x7F, 0x80, 0x7F, 0x02, 0x01,
                        0x01, 0xFF, 0xFF, 0x80, 0xAA, 0x55, 0x12, 0x35))
    edges_u16_a = struct.pack("<8H", 0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF, 0x4000)
    edges_u16_b = struct.pack("<8H", 0x0000, 0xFFFF, 0x0001, 0x8000, 0x7FFF, 0x0002, 0xFFFF, 0xC000)
    edges_u32_a = struct.pack("<4I", 0x00000000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
    edges_u32_b = struct.pack("<4I", 0xFFFFFFFF, 0x00000001, 0x80000000, 0x7FFFFFFF)
    pairs = [
        (distinct_a, distinct_b),
        (b"\x00" * 32, b"\xFF" * 32),
        (b"\xFF" * 32, b"\x00" * 32),
        (distinct_a, distinct_a),
        (pad32(edges_u8_a), pad32(edges_u8_b)),
        (pad32(edges_u16_a), pad32(edges_u16_b)),
        (pad32(edges_u32_a), pad32(edges_u32_b)),
    ]
    for _ in range(5):
        pairs.append((rng.randbytes(32), rng.randbytes(32)))
    return pairs


def pack_sat_pairs():
    """Saturation boundary inputs for vpack* (signed/unsigned clamping at byte/word bounds)."""
    w_a = struct.pack("<8h", -32768, -129, -128, -1, 0, 127, 128, 32767)
    w_b = struct.pack("<8h", -1, 0, 1, 255, 256, 32767, -32768, 127)
    d_a = struct.pack("<4i", -2147483648, -32769, -32768, -1)
    d_b = struct.pack("<4i", 0, 32767, 65535, 65536)
    d_c = struct.pack("<4i", 2147483647, 32768, -65536, 1)
    return [
        (pad32(w_a), pad32(w_b)),
        (pad32(w_b), pad32(w_a)),
        (pad32(d_a), pad32(d_b)),
        (pad32(d_b), pad32(d_c)),
    ]


def sign_matrix_pairs(bits):
    """Full sign matrix {-min, -1, 0, +1, +max} x {-min, -1, 0, +1, +max}."""
    if bits == 8:
        vals = (-128, -127, -1, 0, 1, 127)
        a = struct.pack("<16b", *(vals[i % len(vals)] for i in range(16)))
        b = struct.pack("<16b", *(vals[(i // 3) % len(vals)] for i in range(16)))
        c = struct.pack("<16b", *(vals[(i * 5 + 1) % len(vals)] for i in range(16)))
        return [(pad32(a), pad32(b)), (pad32(a), pad32(c)), (pad32(b), pad32(a))]
    if bits == 16:
        vals = (-32768, -32767, -1, 0, 1, 32767, -42, 42)
        a = struct.pack("<8h", *vals)
        b = struct.pack("<8h", -32768, -1, 0, 1, 32767, -1, 0, 1)
        c = struct.pack("<8h", 0, 1, -1, -32768, 0, 32767, -32768, 0)
        return [(pad32(a), pad32(b)), (pad32(a), pad32(c))]
    a1 = struct.pack("<4i", -2147483648, -1, 0, 2147483647)
    b1 = struct.pack("<4i", -2147483648, -1, 0, 1)
    b2 = struct.pack("<4i", 1, 0, -1, -2147483648)
    b3 = struct.pack("<4i", 0, 2147483647, -2147483648, 0)
    return [(pad32(a1), pad32(b1)), (pad32(a1), pad32(b2)), (pad32(a1), pad32(b3))]


def shift_xmm_pairs(width):
    """Boundary shift counts: 0, 1, width-1, width, width+1, max, and upper-64-bits nonzero."""
    val_a = struct.pack("<2Q", 0x80017FFF0001FFFF, 0x800000017FFFFFFF)
    val_b = struct.pack("<2Q", 0xFFFFFFFF00000000, 0xA5A55A5A12345678)
    counts = [
        (0, 0),
        (1, 0),
        (1, 0xFFFFFFFFFFFFFFFF),  # Proves bits 127:64 are ignored!
        (width - 1, 0),
        (width - 1, 0x123456789ABCDEF0),
        (width, 0),
        (width + 1, 0),
        (63, 0),
        (64, 0),
        (65, 0),
        (255, 0),
        (0x100000001, 0),
        (0x7FFFFFFFFFFFFFFF, 0),
        (0x8000000000000000, 0),
        (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
    ]
    return [(pad32(v), pad32(struct.pack("<2Q", lo, hi))) for v in (val_a, val_b) for lo, hi in counts]


def minpos_pairs():
    vecs = [
        struct.pack("<8H", 5, 3, 9, 3, 7, 3, 1, 1),          # tie at index 6 & 7 -> picks 6
        struct.pack("<8H", 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
        struct.pack("<8H", 0xFFFF, 0xFFFE, 0x8000, 0x7FFF, 0x0001, 0x0002, 0x0001, 0x0000),  # min at 7
        struct.pack("<8H", 0x8000, 0x8001, 0xFFFF, 0xFFFE, 0x8000, 0x9000, 0xA000, 0xB000),  # unsigned >0x7fff
        struct.pack("<8H", 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000),
    ]
    return [(pad32(v), pad32(bytes(range(0x30, 0x40)))) for v in vecs]


def blendvb_pairs():
    src1 = bytes(range(0x10, 0x20))
    src2 = bytes(range(0xA0, 0xB0))
    masks = [
        b"\x00" * 16,
        b"\x7F" * 16,  # MSB=0 -> must pick src1 everywhere
        b"\x80" * 16,  # MSB=1 -> must pick src2 everywhere
        b"\xFF" * 16,
        bytes((0x7F, 0x80, 0x00, 0xFF, 0x01, 0xFE, 0x7E, 0x81,
               0x80, 0x7F, 0xFF, 0x00, 0x8F, 0x70, 0x40, 0xC0)),
    ]
    return [(src1 + m, pad32(src2)) for m in masks]


def extend_pairs():
    vecs = [
        bytes((0x00, 0x01, 0x7F, 0x80, 0x81, 0xFE, 0xFF, 0x42,
               0xAA, 0x55, 0x00, 0xFF, 0x7F, 0x80, 0x01, 0xFE)),
        struct.pack("<8H", 0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF, 0x1234, 0xFEDC),
        struct.pack("<4I", 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x00000000),
    ]
    return [(pad32(v), pad32(bytes(range(0x60, 0x70)))) for v in vecs]


def mulhrsw_pairs():
    # Exact rounding boundaries: (-32768)*(-32768)=0x40000000 -> 0x8000; (-16384)*1 -> 0; (-16385)*1 -> -1
    a = struct.pack("<8h", -32768, -32768, 32767, -16384, -16385, 16384, 16383, -1)
    b = struct.pack("<8h", -32768, 32767, 32767, 1, 1, 1, 1, 1)
    return [(pad32(a), pad32(b)), (pad32(b), pad32(a))]


def inputs_for(name):
    if name in ("op_packsswb", "op_packssdw", "op_packusdw"):
        return pack_sat_pairs() + general_pairs()
    if name == "op_psignb":
        return sign_matrix_pairs(8) + general_pairs()
    if name == "op_psignw":
        return sign_matrix_pairs(16) + general_pairs()
    if name == "op_psignd":
        return sign_matrix_pairs(32) + general_pairs()
    if name in ("op_pslld_xmm", "op_psrld_xmm"):
        return shift_xmm_pairs(32)
    if name in ("op_psllq_xmm", "op_psrlq_xmm"):
        return shift_xmm_pairs(64)
    if name == "op_phminposuw":
        return minpos_pairs() + general_pairs()
    if name == "op_pblendvb":
        return blendvb_pairs() + general_pairs()
    if name.startswith(("op_pmovsx", "op_pmovzx")):
        return extend_pairs() + general_pairs()
    if name == "op_pmulhrsw":
        return mulhrsw_pairs() + general_pairs()
    if name == "op_pmaddwd":
        min_pair = pad32(struct.pack("<2h12x", -32768, -32768))
        return general_pairs() + [(min_pair, min_pair)]
    return general_pairs()


# --- Execution & comparison ---

def invoke(fn, a, b, out_size=32):
    buf_a = ctypes.create_string_buffer(a, 32)
    buf_b = ctypes.create_string_buffer(b, 32)
    fill = FILL_192 if out_size == 192 else FILL
    guarded = ctypes.create_string_buffer(GUARD_PRE + fill + GUARD_POST, out_size + 32)
    out_ptr = ctypes.addressof(guarded) + len(GUARD_PRE)
    fn(ctypes.addressof(buf_a), ctypes.addressof(buf_b), out_ptr)
    raw = guarded.raw
    if raw[:16] != GUARD_PRE or raw[16 + out_size:] != GUARD_POST:
        raise AssertionError("guard bytes corrupted around output buffer")
    return raw[16:16 + out_size]


def assert_equal_output(label, name, idx, got, want):
    if got != want:
        raise AssertionError(
            f"{label} {name} case {idx}: got {got.hex()} != expected {want.hex()}")


def bind_exports(dll_path, prefix="", names=None):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in (ALL_EXPORTS if names is None else names):
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = None
        funcs[name] = fn
    return funcs


def check_differential(native_path, lifted_path, altered_path):
    native = bind_exports(native_path)
    lifted = bind_exports(lifted_path, "lifted_")
    altered = bind_exports(altered_path, "lifted_")

    total = 0
    for name in EXPORTS:
        for idx, (a, b) in enumerate(inputs_for(name)):
            want = reference(name, a, b, altered=False)
            got_native = invoke(native[name], a, b)
            got_lifted = invoke(lifted[name], a, b)
            assert_equal_output("native", name, idx, got_native, want)
            assert_equal_output("lifted", name, idx, got_lifted, want)
            total += 1

    # Dedicated 192-byte check for vzeroall across all six modeled YMM registers.
    want_zeroall = b"\x00" * 192
    for idx, (a, b) in enumerate(general_pairs()):
        got_native = invoke(native["op_zeroall"], a, b, out_size=192)
        got_lifted = invoke(lifted["op_zeroall"], a, b, out_size=192)
        assert_equal_output("native", "op_zeroall", idx, got_native, want_zeroall)
        assert_equal_output("lifted", "op_zeroall", idx, got_lifted, want_zeroall)
        total += 1

    # ALTER_IMM negative controls.
    altered_exports = (
        "op_palignr_5", "op_pblendw", "op_mpsadbw_0", "op_pshufhw", "op_pextrb_reg",
    )
    changed = {}
    for name in altered_exports:
        diffs = 0
        for idx, (a, b) in enumerate(inputs_for(name)):
            want_alt = reference(name, a, b, altered=True)
            want_orig = reference(name, a, b, altered=False)
            got_alt = invoke(altered[name], a, b)
            assert_equal_output("altered", name, idx, got_alt, want_alt)
            if got_alt != want_orig:
                diffs += 1
        if not diffs:
            raise AssertionError(f"altered control {name} did not diverge from original")
        changed[name] = diffs

    # Explicit comparator mutation control: flipping one bit must fail assert_equal_output.
    sample_a, sample_b = general_pairs()[0]
    expected = reference("op_pipeline", sample_a, sample_b)
    mutated = bytearray(expected)
    mutated[9] ^= 0x01
    try:
        assert_equal_output("mutation", "op_pipeline", 0, bytes(mutated), expected)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated output was not caught by comparator")

    print(f"{total} native/lifted/reference AVX packed-integer cases across {len(ALL_EXPORTS)} exports "
          f"matched bit-for-bit; altered controls diverged on {changed}; output mutation control caught")


# --- Translator rejection controls ---

def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections(original_path):
    """Verify VEX.L=1 integer ops, register-only memory ops, EVEX, out-of-scope, and ABI/state violations reject."""
    ld_xmm0 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_xmm1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RCX))
    ld_xmm2 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM2, MemoryOperand(R.RDX))
    ld_xmm3 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM3, MemoryOperand(R.RDX))
    ld_ymm0 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX))
    ld_ymm1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM1, MemoryOperand(R.RCX))
    ld_ymm2 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM2, MemoryOperand(R.RDX))
    ld_ymm3 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM3, MemoryOperand(R.RDX))
    ld_eax = Instruction.create_reg_mem(Code.MOV_R32_RM32, R.EAX, MemoryOperand(R.RDX))
    ret_ins = Instruction.create(Code.RETNQ)

    # 1) VEX.L=1 (256-bit) packed-integer encodings of surviving forms must be rejected explicitly (AVX2, not AVX1).
    # The 256-bit add, subtract, saturating add and subtract, compare, minimum, maximum, absolute value,
    # sign, word shuffle, horizontal, and pack forms are lowered and are covered by avx_int_ymm_addsub_check.py,
    # avx_int_ymm_sat_check.py, avx_int_ymm_minmax_check.py, avx_int_ymm_abs_sign_shuf_check.py,
    # avx_int_ymm_horizontal_check.py, and avx_int_ymm_pack_check.py.
    vex256_int_cases = [
        ("vpmulhuw_ymm", Instruction.create_reg_reg_reg(Code.VEX_VPMULHUW_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)),
        ("vpunpckhdq_ymm", Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKHDQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)),
        ("vpalignr_ymm", Instruction.create_reg_reg_reg_u32(Code.VEX_VPALIGNR_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, R.YMM2, 4)),
        ("vpblendvb_ymm", Instruction.create_reg_reg_reg_reg(Code.VEX_VPBLENDVB_YMM_YMM_YMMM256_YMM, R.YMM0, R.YMM1, R.YMM2, R.YMM3)),
        ("vpblendw_ymm", Instruction.create_reg_reg_reg_u32(Code.VEX_VPBLENDW_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, R.YMM2, 0xAA)),
        ("vpmaddwd_ymm", Instruction.create_reg_reg_reg(Code.VEX_VPMADDWD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)),
        ("vpmulld_ymm", Instruction.create_reg_reg_reg(Code.VEX_VPMULLD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)),
        ("vmpsadbw_ymm", Instruction.create_reg_reg_reg_u32(Code.VEX_VMPSADBW_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, R.YMM2, 0)),
        ("vpmovsxbw_ymm", Instruction.create_reg_reg(Code.VEX_VPMOVSXBW_YMM_XMMM128, R.YMM0, R.XMM1)),
        ("vpmovzxdq_ymm", Instruction.create_reg_reg(Code.VEX_VPMOVZXDQ_YMM_XMMM128, R.YMM0, R.XMM1)),
        ("vpunpckldq_ymm", Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKLDQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)),
        ("vpslld_ymm_imm", Instruction.create_reg_reg_u32(Code.VEX_VPSLLD_YMM_YMM_IMM8, R.YMM0, R.YMM1, 4)),
        ("vpslld_ymm_xmm", Instruction.create_reg_reg_reg(Code.VEX_VPSLLD_YMM_YMM_XMMM128, R.YMM0, R.YMM1, R.XMM2)),
        ("vpsrlq_ymm_imm", Instruction.create_reg_reg_u32(Code.VEX_VPSRLQ_YMM_YMM_IMM8, R.YMM0, R.YMM1, 4)),
        ("vpmovmskb_ymm", Instruction.create_reg_reg(Code.VEX_VPMOVMSKB_R32_YMM, R.EAX, R.YMM1)),
    ]
    vex256_by_label = dict(vex256_int_cases)

    # Alternate valid encodings: r64/W1 GPR bridge forms MASM may not emit by default.
    alt_accept = [
        ("vpmovmskb_r64", (ld_xmm1, Instruction.create_reg_reg(
            Code.VEX_VPMOVMSKB_R64_XMM, R.RAX, R.XMM1), ret_ins)),
        ("vpextrb_r64", (ld_xmm1, Instruction.create_reg_reg_u32(
            Code.VEX_VPEXTRB_R64M8_XMM_IMM8, R.RAX, R.XMM1, 5), ret_ins)),
        ("vpextrw_r64", (ld_xmm1, Instruction.create_reg_reg_u32(
            Code.VEX_VPEXTRW_R64_XMM_IMM8, R.RAX, R.XMM1, 3), ret_ins)),
        ("vpextrw_r64m16", (ld_xmm1, Instruction.create_reg_reg_u32(
            Code.VEX_VPEXTRW_R64M16_XMM_IMM8, R.RAX, R.XMM1, 6), ret_ins)),
        ("vpinsrb_r64", (ld_xmm1, ld_eax, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VPINSRB_XMM_XMM_R64M8_IMM8, R.XMM0, R.XMM1, R.RAX, 5), ret_ins)),
        ("vpinsrw_r64", (ld_xmm1, ld_eax, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VPINSRW_XMM_XMM_R64M16_IMM8, R.XMM0, R.XMM1, R.RAX, 6), ret_ins)),
    ]
    for label, seq in alt_accept:
        src, count = lift_code(label, 0x1000, encode_seq(*seq))
        assert f"lifted_{label}" in src and count == len(seq), (label, count)

    # 2) Memory operands on surviving register-only AVX packed-integer instructions must be rejected explicitly.
    reg_only_mem_cases = [
        ("vpabsb_mem", Instruction.create_reg_mem(Code.VEX_VPABSB_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))),
        ("vpacksswb_mem", Instruction.create_reg_reg_mem(Code.VEX_VPACKSSWB_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vpaddq_mem", Instruction.create_reg_reg_mem(Code.VEX_VPADDQ_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vpsubsb_mem", Instruction.create_reg_reg_mem(Code.VEX_VPSUBSB_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vpaddb_mem", Instruction.create_reg_reg_mem(Code.VEX_VPADDB_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vpsubd_mem", Instruction.create_reg_reg_mem(Code.VEX_VPSUBD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vpalignr_mem", Instruction.create_reg_reg_mem_u32(Code.VEX_VPALIGNR_XMM_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, MemoryOperand(R.RDX), 4)),
        ("vpblendvb_mem", Instruction.create_reg_reg_mem_reg(Code.VEX_VPBLENDVB_XMM_XMM_XMMM128_XMM, R.XMM0, R.XMM1, MemoryOperand(R.RDX), R.XMM3)),
        ("vpblendw_mem", Instruction.create_reg_reg_mem_u32(Code.VEX_VPBLENDW_XMM_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, MemoryOperand(R.RDX), 0xAA)),
        ("vpcmpeqq_mem", Instruction.create_reg_reg_mem(Code.VEX_VPCMPEQQ_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vpmaxud_mem", Instruction.create_reg_reg_mem(Code.VEX_VPMAXUD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vphaddsw_mem", Instruction.create_reg_reg_mem(Code.VEX_VPHADDSW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vphminposuw_mem", Instruction.create_reg_mem(Code.VEX_VPHMINPOSUW_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))),
        ("vpmaddwd_mem", Instruction.create_reg_reg_mem(Code.VEX_VPMADDWD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vpmulhrsw_mem", Instruction.create_reg_reg_mem(Code.VEX_VPMULHRSW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vmpsadbw_mem", Instruction.create_reg_reg_mem_u32(Code.VEX_VMPSADBW_XMM_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, MemoryOperand(R.RDX), 0)),
        ("vpshufhw_mem", Instruction.create_reg_mem_u32(Code.VEX_VPSHUFHW_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RCX), 0x1B)),
        ("vpshuflw_mem", Instruction.create_reg_mem_u32(Code.VEX_VPSHUFLW_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RCX), 0x4E)),
        ("vpsignb_mem", Instruction.create_reg_reg_mem(Code.VEX_VPSIGNB_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vpmovsxbw_mem", Instruction.create_reg_mem(Code.VEX_VPMOVSXBW_XMM_XMMM64, R.XMM0, MemoryOperand(R.RCX))),
        ("vpmovzxbd_mem", Instruction.create_reg_mem(Code.VEX_VPMOVZXBD_XMM_XMMM32, R.XMM0, MemoryOperand(R.RCX))),
        ("vpunpckldq_mem", Instruction.create_reg_reg_mem(Code.VEX_VPUNPCKLDQ_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vpslld_mem", Instruction.create_reg_reg_mem(Code.VEX_VPSLLD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vpsrld_mem", Instruction.create_reg_reg_mem(Code.VEX_VPSRLD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
        ("vpsrlq_mem", Instruction.create_reg_reg_mem(Code.VEX_VPSRLQ_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RDX))),
    ]

    # 3) Carved-out packed-integer forms owned by cpu_vex_aot.py (PR #325) (all 28 forms must reject as unsupported).
    carved_out_cases = [
        ("carved_vpackuswb", Instruction.create_reg_reg_reg(Code.VEX_VPACKUSWB_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpaddd", Instruction.create_reg_reg_reg(Code.VEX_VPADDD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpaddw", Instruction.create_reg_reg_reg(Code.VEX_VPADDW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpand", Instruction.create_reg_reg_reg(Code.VEX_VPAND_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpor", Instruction.create_reg_reg_reg(Code.VEX_VPOR_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpxor", Instruction.create_reg_reg_reg(Code.VEX_VPXOR_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpavgw", Instruction.create_reg_reg_reg(Code.VEX_VPAVGW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpmaxsw", Instruction.create_reg_reg_reg(Code.VEX_VPMAXSW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpminsw", Instruction.create_reg_reg_reg(Code.VEX_VPMINSW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpmulhw", Instruction.create_reg_reg_reg(Code.VEX_VPMULHW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpsllw_imm", Instruction.create_reg_reg_u32(Code.VEX_VPSLLW_XMM_XMM_IMM8, R.XMM0, R.XMM1, 4)),
        ("carved_vpsllw_xmm", Instruction.create_reg_reg_reg(Code.VEX_VPSLLW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpsraw_imm", Instruction.create_reg_reg_u32(Code.VEX_VPSRAW_XMM_XMM_IMM8, R.XMM0, R.XMM1, 4)),
        ("carved_vpsraw_xmm", Instruction.create_reg_reg_reg(Code.VEX_VPSRAW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpsrlw_imm", Instruction.create_reg_reg_u32(Code.VEX_VPSRLW_XMM_XMM_IMM8, R.XMM0, R.XMM1, 4)),
        ("carved_vpsrlw_xmm", Instruction.create_reg_reg_reg(Code.VEX_VPSRLW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpsubw", Instruction.create_reg_reg_reg(Code.VEX_VPSUBW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpunpckhbw", Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKHBW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpunpckhwd", Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKHWD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpunpcklbw", Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKLBW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpunpcklwd", Instruction.create_reg_reg_reg(Code.VEX_VPUNPCKLWD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpavgb", Instruction.create_reg_reg_reg(Code.VEX_VPAVGB_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpshufb", Instruction.create_reg_reg_reg(Code.VEX_VPSHUFB_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpmaddubsw", Instruction.create_reg_reg_reg(Code.VEX_VPMADDUBSW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpaddsw", Instruction.create_reg_reg_reg(Code.VEX_VPADDSW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpmullw", Instruction.create_reg_reg_reg(Code.VEX_VPMULLW_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpandn", Instruction.create_reg_reg_reg(Code.VEX_VPANDN_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpsrad_imm", Instruction.create_reg_reg_u32(Code.VEX_VPSRAD_XMM_XMM_IMM8, R.XMM0, R.XMM1, 4)),
        ("carved_vpsrad_xmm", Instruction.create_reg_reg_reg(Code.VEX_VPSRAD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("carved_vpsrldq", Instruction.create_reg_reg_u32(Code.VEX_VPSRLDQ_XMM_XMM_IMM8, R.XMM0, R.XMM1, 4)),
        ("carved_vpslldq", Instruction.create_reg_reg_u32(Code.VEX_VPSLLDQ_XMM_XMM_IMM8, R.XMM0, R.XMM1, 4)),
        ("carved_vpshufd", Instruction.create_reg_reg_u32(Code.VEX_VPSHUFD_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0x1B)),
    ]

    # 4) Out-of-scope instructions (AVX2-only, AVX FP, legacy SSE/MMX, branches, GPR ALU, ud2).
    out_of_scope_cases = [
        ("vpermd", Instruction.create_reg_reg_reg(Code.VEX_VPERMD_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2)),
        ("vpbroadcastd", Instruction.create_reg_reg(Code.VEX_VPBROADCASTD_XMM_XMMM32, R.XMM0, R.XMM1)),
        ("vpsllvd", Instruction.create_reg_reg_reg(Code.VEX_VPSLLVD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("vaddps", Instruction.create_reg_reg_reg(Code.VEX_VADDPS_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)),
        ("vptest", Instruction.create_reg_reg(Code.VEX_VPTEST_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("legacy_paddd", Instruction.create_reg_reg(Code.PADDD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("vmovdqa", Instruction.create_reg_mem(Code.VEX_VMOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))),
        ("add_gpr", Instruction.create_reg_reg(Code.ADD_R32_RM32, R.EAX, R.ECX)),
        ("jmp_rel8", Instruction.create_branch(Code.JMP_REL8_64, 0x1020)),
        ("ud2", Instruction.create(Code.UD2)),
    ]

    # 5) EVEX, prefixes, ABI, GPR, register-state and addressing violations.
    evex_paddq = Instruction.create_reg_reg_reg(
        Code.EVEX_VPADDQ_XMM_K1Z_XMM_XMMM128B64, R.XMM0, R.XMM1, R.XMM2)
    evex_paddq.op_mask = R.K1
    seg_vmov = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    seg_vmov.segment_prefix = R.GS
    seg_pinsr = Instruction.create_reg_reg_mem_u32(
        Code.VEX_VPINSRD_XMM_XMM_RM32_IMM8, R.XMM0, R.XMM1, MemoryOperand(R.RDX), 0)
    seg_pinsr.segment_prefix = R.FS
    other_cases = [
        ("evex_vpaddq", (ld_xmm0, ld_xmm1, ld_xmm2, evex_paddq, ret_ins),
         "unsupported EVEX instruction"),
        ("nonvolatile_xmm6", (ld_xmm1, ld_xmm2, Instruction.create_reg_reg_reg(
            Code.VEX_VPADDQ_XMM_XMM_XMMM128, R.XMM6, R.XMM1, R.XMM2), ret_ins),
         "nonvolatile vector register"),
        ("nonvolatile_ymm6", (Instruction.create_reg_mem(
            Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM6, MemoryOperand(R.RCX)), ret_ins),
         "nonvolatile vector register"),
        ("nonvolatile_gpr_ebx", (ld_xmm1, Instruction.create_reg_reg(
            Code.VEX_VPMOVMSKB_R32_XMM, R.EBX, R.XMM1), ret_ins),
         "nonvolatile GPR"),
        ("arg_gpr_write_ecx", (ld_xmm1, Instruction.create_reg_reg(
            Code.VEX_VPMOVMSKB_R32_XMM, R.ECX, R.XMM1), ret_ins),
         "unsupported argument GPR operand"),
        ("arg_gpr_read_edx", (ld_xmm1, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VPINSRD_XMM_XMM_RM32_IMM8, R.XMM0, R.XMM1, R.EDX, 0), ret_ins),
         "unsupported argument GPR operand"),
        ("undef_xmm3", (ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VPADDQ_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM3), ret_ins),
         "undefined vector register ymm3"),
        ("undef_gpr_eax", (ld_xmm1, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VPINSRD_XMM_XMM_RM32_IMM8, R.XMM0, R.XMM1, R.EAX, 0), ret_ins),
         "undefined GPR rax"),
        ("mov_reg_reg", (ld_eax, Instruction.create_reg_reg(Code.MOV_R32_RM32, R.R9D, R.EAX), ret_ins),
         "unsupported mov operands"),
        ("vpextrb_rip_rel", (ld_xmm1, Instruction.create_mem_reg_u32(
            Code.VEX_VPEXTRB_R32M8_XMM_IMM8, MemoryOperand(R.RIP, displ=0x1010, displ_size=1), R.XMM1, 0), ret_ins),
         "unsupported memory operand"),
        ("vpinsrd_rsp_base", (ld_xmm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VPINSRD_XMM_XMM_RM32_IMM8, R.XMM0, R.XMM1, MemoryOperand(R.RSP), 0), ret_ins),
         "unsupported memory operand"),
        ("vpinsrd_indexed", (ld_xmm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VPINSRD_XMM_XMM_RM32_IMM8, R.XMM0, R.XMM1, MemoryOperand(R.RDX, R.RCX, 4), 0), ret_ins),
         "unsupported memory operand"),
        ("vpextrd_addr32", (ld_xmm1, Instruction.create_mem_reg_u32(
            Code.VEX_VPEXTRD_RM32_XMM_IMM8, MemoryOperand(R.R8D), R.XMM1, 0), ret_ins),
         "unsupported memory operand"),
        ("vpextrd_non_arg_base", (ld_xmm1, Instruction.create_mem_reg_u32(
            Code.VEX_VPEXTRD_RM32_XMM_IMM8, MemoryOperand(R.RAX), R.XMM1, 0), ret_ins),
         "unsupported memory operand"),
        ("segment_prefix_vmov", (seg_vmov, ret_ins), "unsupported instruction prefix"),
        ("segment_prefix_pinsr", (ld_xmm1, seg_pinsr, ret_ins), "unsupported instruction prefix"),
        ("missing_ret", (ld_xmm0,), "missing terminal ret"),
        ("after_ret", (ret_ins, ld_xmm0), "unreachable instruction after ret"),
        ("undef_blend_mask", (ld_xmm1, ld_xmm2, Instruction.create_reg_reg_reg_reg(
            Code.VEX_VPBLENDVB_XMM_XMM_XMMM128_XMM, R.XMM0, R.XMM1, R.XMM2, R.XMM3), ret_ins),
         "undefined vector register ymm3"),
    ]

    all_unit = [
        *((lbl, (ld_ymm0, ld_ymm1, ld_ymm2, ld_ymm3, ins, ret_ins),
           "unsupported VEX.256 (L=1) packed-integer encoding")
          for lbl, ins in vex256_int_cases),
        *((lbl, (ld_xmm0, ld_xmm1, ld_xmm2, ld_xmm3, ins, ret_ins),
           "unsupported memory operand for register-domain AVX integer instruction")
          for lbl, ins in reg_only_mem_cases),
        *((lbl, (ld_xmm0, ld_xmm1, ld_xmm2, ins, ret_ins), "unsupported instruction")
          for lbl, ins in carved_out_cases),
        *((lbl, (ld_ymm0, ld_ymm1, ld_ymm2, ld_eax, ins, ret_ins), "unsupported instruction")
          for lbl, ins in out_of_scope_cases),
        *other_cases,
    ]

    for label, seq, expected_reason in all_unit:
        raw = encode_seq(*seq)
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator unexpectedly accepted rejected control")

    # End-to-end PE patching controls on a copy of the built native DLL.
    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_pipeline")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12)
                       if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        orig_end = struct.unpack_from("<I", records, rec_idx + 4)[0]
        max_len = orig_end - entry

    pe_patches = [
        ("pe_vpmulhuw_ymm", encode_seq(ld_ymm1, ld_ymm2, vex256_by_label["vpmulhuw_ymm"], ret_ins),
         "unsupported VEX.256 (L=1) packed-integer encoding"),
        ("pe_vpslld_ymm_xmm", encode_seq(ld_ymm1, ld_xmm2, vex256_by_label["vpslld_ymm_xmm"], ret_ins),
         "unsupported VEX.256 (L=1) packed-integer encoding"),
        ("pe_vpaddq_mem", encode_seq(ld_xmm1, reg_only_mem_cases[2][1], ret_ins),
         "unsupported memory operand for register-domain AVX integer instruction"),
        ("pe_vphminposuw_mem", encode_seq(reg_only_mem_cases[12][1], ret_ins),
         "unsupported memory operand for register-domain AVX integer instruction"),
        ("pe_carved_vpaddd", encode_seq(ld_xmm1, ld_xmm2, carved_out_cases[1][1], ret_ins),
         "unsupported instruction"),
        ("pe_evex_vpaddq", encode_seq(ld_xmm1, ld_xmm2, evex_paddq, ret_ins),
         "unsupported EVEX instruction"),
        ("pe_out_of_scope_vaddps", encode_seq(ld_xmm1, ld_xmm2, out_of_scope_cases[3][1], ret_ins),
         "unsupported instruction"),
        ("pe_nonvolatile_xmm6", encode_seq(*other_cases[1][1]),
         "nonvolatile vector register"),
        ("pe_undef_xmm3", encode_seq(*other_cases[6][1]),
         "undefined vector register ymm3"),
        ("pe_missing_ret", encode_seq(ld_xmm0), "missing terminal ret"),
        ("pe_after_ret", encode_seq(ret_ins, ld_xmm0), "unreachable instruction after ret"),
    ]
    with TemporaryDirectory() as folder:
        patched_path = Path(folder) / "patched.dll"
        for label, patch, expected_reason in pe_patches:
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds op_pipeline ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["op_pipeline"])
            except ValueError as exc:
                if expected_reason not in str(exc):
                    raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: PE translate accepted rejected patch")

    print(f"{len(all_unit)} unit rejection controls ({len(vex256_int_cases)} VEX.L=1 integer, "
          f"{len(reg_only_mem_cases)} register-domain memory, {len(carved_out_cases)} carved-out, "
          f"{len(out_of_scope_cases)} out-of-scope, {len(other_cases)} EVEX/ABI/state) and "
          f"{len(pe_patches)} PE-patched rejection controls passed")


if __name__ == "__main__":
    check_differential(*sys.argv[1:4])
    check_rejections(sys.argv[1])
