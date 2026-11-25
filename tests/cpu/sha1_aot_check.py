"""Differential, NIST FIPS 180-4, aliasing, mutation, and rejection checks for SHA-1 NI AOT lifting."""

import ctypes
import hashlib
import os
from pathlib import Path
import random
import struct
import sys

from iced_x86 import (
    Code,
    Decoder,
    Encoder,
    Formatter,
    FormatterSyntax,
    Instruction,
    MemoryOperand,
    Register as R,
)
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402
from cpu_sha1_aot import lift_code  # noqa: E402

EXPORTS = (
    "op_sha1rnds4_0_reg",
    "op_sha1rnds4_1_reg",
    "op_sha1rnds4_2_reg",
    "op_sha1rnds4_3_reg",
    "op_sha1rnds4_hi_imm",
    "op_sha1rnds4_0_mem",
    "op_sha1rnds4_1_mem",
    "op_sha1rnds4_2_mem",
    "op_sha1rnds4_3_mem",
    "op_sha1rnds4_self",
    "op_sha1nexte_reg",
    "op_sha1nexte_mem",
    "op_sha1nexte_self",
    "op_sha1msg1_reg",
    "op_sha1msg1_mem",
    "op_sha1msg1_self",
    "op_sha1msg2_reg",
    "op_sha1msg2_mem",
    "op_sha1msg2_self",
    "op_sha1_pipeline",
)

UNALIGNED_SAFE_EXPORTS = (
    "op_sha1rnds4_0_reg",
    "op_sha1rnds4_1_reg",
    "op_sha1rnds4_2_reg",
    "op_sha1rnds4_3_reg",
    "op_sha1rnds4_hi_imm",
    "op_sha1rnds4_self",
    "op_sha1nexte_reg",
    "op_sha1nexte_self",
    "op_sha1msg1_reg",
    "op_sha1msg1_self",
    "op_sha1msg2_reg",
    "op_sha1msg2_self",
)

ALTERED_EXPORTS = (
    "op_sha1rnds4_0_reg",
    "op_sha1rnds4_1_reg",
    "op_sha1rnds4_2_reg",
    "op_sha1rnds4_3_reg",
    "op_sha1nexte_reg",
    "op_sha1msg1_reg",
    "op_sha1msg2_reg",
)

GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))


# --- Independent Python reference for SHA-1 NI instructions ---

def _rol32(x, n):
    x &= 0xFFFFFFFF
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def _unpack4(buf):
    return struct.unpack("<4I", buf)


def _pack4(w0, w1, w2, w3):
    return struct.pack("<4I", w0 & 0xFFFFFFFF, w1 & 0xFFFFFFFF, w2 & 0xFFFFFFFF, w3 & 0xFFFFFFFF)


def ref_sha1rnds4(dst_bytes, src_bytes, imm8):
    d0, d1, d2, d3 = _unpack4(dst_bytes)
    s0, s1, s2, s3 = _unpack4(src_bytes)
    a, b, c, d, e = d3, d2, d1, d0, 0
    w = (s3, s2, s1, s0)
    sel = imm8 & 3
    k = (0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6)[sel]
    for i in range(4):
        if sel == 0:
            f = (b & c) ^ ((~b & 0xFFFFFFFF) & d)
        elif sel == 2:
            f = (b & c) ^ (b & d) ^ (c & d)
        else:
            f = b ^ c ^ d
        next_a = (f + _rol32(a, 5) + w[i] + k + e) & 0xFFFFFFFF
        e = d
        d = c
        c = _rol32(b, 30)
        b = a
        a = next_a
    return _pack4(d, c, b, a)


def ref_sha1nexte(dst_bytes, src_bytes):
    _, _, _, d3 = _unpack4(dst_bytes)
    s0, s1, s2, s3 = _unpack4(src_bytes)
    return _pack4(s0, s1, s2, (s3 + _rol32(d3, 30)) & 0xFFFFFFFF)


def ref_sha1msg1(dst_bytes, src_bytes):
    d0, d1, d2, d3 = _unpack4(dst_bytes)
    _, _, s2, s3 = _unpack4(src_bytes)
    return _pack4(d0 ^ s2, d1 ^ s3, d2 ^ d0, d3 ^ d1)


def ref_sha1msg2(dst_bytes, src_bytes):
    d0, d1, d2, d3 = _unpack4(dst_bytes)
    s0, s1, s2, _ = _unpack4(src_bytes)
    w16 = _rol32(d3 ^ s2, 1)
    w17 = _rol32(d2 ^ s1, 1)
    w18 = _rol32(d1 ^ s0, 1)
    w19 = _rol32(d0 ^ w16, 1)
    return _pack4(w19, w18, w17, w16)


def reference(name, a, b, *, altered=False):
    if name == "op_sha1rnds4_0_reg":
        return ref_sha1rnds4(a, b, 1 if altered else 0)
    if name == "op_sha1rnds4_1_reg":
        return ref_sha1rnds4(a, b, 2 if altered else 1)
    if name == "op_sha1rnds4_2_reg":
        return ref_sha1rnds4(a, b, 3 if altered else 2)
    if name == "op_sha1rnds4_3_reg":
        return ref_sha1rnds4(a, b, 0 if altered else 3)
    if name == "op_sha1rnds4_hi_imm":
        v = ref_sha1rnds4(a, b, 0xFC)
        v = ref_sha1rnds4(v, b, 0xFD)
        v = ref_sha1rnds4(v, b, 0xFE)
        return ref_sha1rnds4(v, b, 0xFF)
    if name == "op_sha1rnds4_0_mem":
        return ref_sha1rnds4(a, b, 0)
    if name == "op_sha1rnds4_1_mem":
        return ref_sha1rnds4(a, b, 1)
    if name == "op_sha1rnds4_2_mem":
        return ref_sha1rnds4(a, b, 2)
    if name == "op_sha1rnds4_3_mem":
        return ref_sha1rnds4(a, b, 3)
    if name == "op_sha1rnds4_self":
        return ref_sha1rnds4(a, a, 2)
    if name == "op_sha1nexte_reg":
        return ref_sha1nexte(b, a) if altered else ref_sha1nexte(a, b)
    if name == "op_sha1nexte_mem":
        return ref_sha1nexte(a, b)
    if name == "op_sha1nexte_self":
        return ref_sha1nexte(a, a)
    if name == "op_sha1msg1_reg":
        return ref_sha1msg2(a, b) if altered else ref_sha1msg1(a, b)
    if name == "op_sha1msg1_mem":
        return ref_sha1msg1(a, b)
    if name == "op_sha1msg1_self":
        return ref_sha1msg1(a, a)
    if name == "op_sha1msg2_reg":
        return ref_sha1msg1(a, b) if altered else ref_sha1msg2(a, b)
    if name == "op_sha1msg2_mem":
        return ref_sha1msg2(a, b)
    if name == "op_sha1msg2_self":
        return ref_sha1msg2(a, a)
    if name == "op_sha1_pipeline":
        x2 = ref_sha1msg2(ref_sha1msg1(a, b), b)
        x3 = ref_sha1rnds4(a, b, 0)
        x4 = ref_sha1nexte(x3, x2)
        return ref_sha1rnds4(x3, x4, 1)
    raise KeyError(name)


def all_pairs():
    """Construct edge, carry-wrap, lane-isolation, FIPS-state, and deterministic random 16-byte input pairs."""
    z16 = b"\x00" * 16
    f16 = b"\xff" * 16
    # Standard FIPS 180-4 SHA-1 initial IV words packed in XMM lane order (w3=H0..w0=H3).
    fips_iv = _pack4(0x10325476, 0x98BADCFE, 0xEFCDAB89, 0x67452301)
    pairs = [
        (z16, z16),
        (f16, f16),
        (z16, f16),
        (f16, z16),
        (fips_iv, _pack4(0x00000000, 0x00000000, 0x00000000, 0x61626380 + 0xC3D2E1F0)),
        # Carry wrap in SHA1NEXTE: rol32(0xFFFFFFFF, 30) == 0xFFFFFFFF; 0xFFFFFFFF + 1 == 0.
        (_pack4(0x11111111, 0x22222222, 0x33333333, 0xFFFFFFFF), _pack4(0x44444444, 0x55555555, 0x66666666, 0x00000001)),
        # Rotation bit boundaries in lane 3 (bits 0, 1, 2, 30, 31) and carry across 32-bit wrap.
        (_pack4(0x00000001, 0x80000000, 0x00000003, 0x00000001), _pack4(0x80000000, 0x00000001, 0x40000000, 0xC0000000)),
        (_pack4(0x80000000, 0x40000000, 0x20000000, 0x80000000), _pack4(0x00000001, 0x00000002, 0x00000004, 0x80000000)),
        # Single-lane isolation probes (each dword isolated in dst or src).
        *[
            (_pack4(*(0xA5A55A5A if j == i else 0 for j in range(4))),
             _pack4(*(0x3C3CC3C3 if j == k else 0 for j in range(4))))
            for i in range(4) for k in range(4)
        ],
    ]
    rng = random.Random(0x5A1A07)
    for _ in range(96):
        pairs.append((rng.randbytes(16), rng.randbytes(16)))
    return pairs


def alloc_guarded(payload, *, misalign=0):
    """Allocate a guarded 16-byte buffer whose payload begins at (16B-aligned + misalign)."""
    size = 48 + len(GUARD_PRE) + len(payload) + len(GUARD_POST)
    raw = ctypes.create_string_buffer(size)
    base = ctypes.addressof(raw)
    aligned = (base + len(GUARD_PRE) + 15) & ~15
    data_addr = aligned + misalign
    pre_off = (data_addr - len(GUARD_PRE)) - base
    data_off = data_addr - base
    post_off = data_off + len(payload)
    raw[pre_off:data_off] = GUARD_PRE
    raw[data_off:post_off] = payload
    raw[post_off:post_off + len(GUARD_POST)] = GUARD_POST
    return raw, pre_off, data_off, post_off, data_addr


def invoke(fn, a, b, *, a_misalign=0, b_misalign=0, out_misalign=0):
    buf_a, a_pre, a_off, a_post, a_addr = alloc_guarded(a, misalign=a_misalign)
    buf_b, b_pre, b_off, b_post, b_addr = alloc_guarded(b, misalign=b_misalign)
    buf_out, o_pre, o_off, o_post, o_addr = alloc_guarded(b"\x5a" * 16, misalign=out_misalign)
    fn(a_addr, b_addr, o_addr)
    assert buf_a.raw[a_pre:a_off] == GUARD_PRE and buf_a.raw[a_post:a_post + 16] == GUARD_POST, "input A guards corrupted"
    assert buf_b.raw[b_pre:b_off] == GUARD_PRE and buf_b.raw[b_post:b_post + 16] == GUARD_POST, "input B guards corrupted"
    assert buf_out.raw[o_pre:o_off] == GUARD_PRE and buf_out.raw[o_post:o_post + 16] == GUARD_POST, "output guards corrupted"
    assert buf_a.raw[a_off:a_post] == a, "input A mutated unexpectedly"
    assert buf_b.raw[b_off:b_post] == b, "input B mutated unexpectedly"
    return buf_out.raw[o_off:o_post]


def invoke_aliased(fn, a, b, mode):
    """Invoke fn with aliased argument pointers (rcx==rdx, r8==rcx, r8==rdx, or all three)."""
    buf_a, a_pre, a_off, a_post, a_addr = alloc_guarded(a)
    buf_b, b_pre, b_off, b_post, b_addr = alloc_guarded(b)
    buf_out, o_pre, o_off, o_post, o_addr = alloc_guarded(b"\x5a" * 16)
    if mode == "rcx==rdx":
        fn(a_addr, a_addr, o_addr)
        res = buf_out.raw[o_off:o_post]
    elif mode == "r8==rcx":
        fn(a_addr, b_addr, a_addr)
        res = buf_a.raw[a_off:a_post]
    elif mode == "r8==rdx":
        fn(a_addr, b_addr, b_addr)
        res = buf_b.raw[b_off:b_post]
    elif mode == "all":
        fn(a_addr, a_addr, a_addr)
        res = buf_a.raw[a_off:a_post]
    else:
        raise ValueError(f"unknown alias mode {mode}")
    assert buf_a.raw[a_pre:a_off] == GUARD_PRE and buf_a.raw[a_post:a_post + 16] == GUARD_POST, f"{mode}: A guards corrupted"
    assert buf_b.raw[b_pre:b_off] == GUARD_PRE and buf_b.raw[b_post:b_post + 16] == GUARD_POST, f"{mode}: B guards corrupted"
    assert buf_out.raw[o_pre:o_off] == GUARD_PRE and buf_out.raw[o_post:o_post + 16] == GUARD_POST, f"{mode}: out guards corrupted"
    return res


def bind(path, prefix):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


def host_has_sha(lifted_lib):
    if hasattr(lifted_lib, "host_has_sha"):
        lifted_lib.host_has_sha.argtypes = []
        lifted_lib.host_has_sha.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_sha())
    try:
        return " sha_ni" in Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace")
    except OSError:
        return False


def encode_seq(*instructions):
    enc = Encoder(64)
    rip = 0x1000
    for ins in instructions:
        rip += enc.encode(ins, rip)
    return enc.take_buffer()


def check_translator():
    ret = Instruction.create(Code.RETNQ)
    ld0 = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld1 = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX))
    ld0_a = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    st0 = Instruction.create_mem_reg(Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)
    st0_a = Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)

    # Positive checks for all 4 SHA-1 NI instructions (reg & flat [rdx]/[r8] mem, plus movdqa/movdqu glue).
    accepted = 0
    for imm in (0, 1, 2, 3, 0xFC, 0xFF):
        src_reg, _ = lift_code(
            f"sha1rnds4_reg_{imm}", 0x1000,
            encode_seq(ld0, ld1, Instruction.create_reg_reg_u32(Code.SHA1RNDS4_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, imm), st0, ret),
        )
        assert f"op_sha1rnds4(xmm0, xmm1, {imm:#x}u)" in src_reg
        src_mem, _ = lift_code(
            f"sha1rnds4_mem_{imm}", 0x1000,
            encode_seq(ld0_a, Instruction.create_reg_mem_u32(Code.SHA1RNDS4_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RDX), imm), st0_a, ret),
        )
        assert f"op_sha1rnds4(xmm0, load128(rdx), {imm:#x}u)" in src_mem
        accepted += 1
    for code, helper in (
        (Code.SHA1NEXTE_XMM_XMMM128, "op_sha1nexte"),
        (Code.SHA1MSG1_XMM_XMMM128, "op_sha1msg1"),
        (Code.SHA1MSG2_XMM_XMMM128, "op_sha1msg2"),
    ):
        src_reg, _ = lift_code(
            f"{helper}_reg", 0x1000,
            encode_seq(ld0, ld1, Instruction.create_reg_reg(code, R.XMM0, R.XMM1), st0, ret),
        )
        assert f"{helper}(xmm0, xmm1)" in src_reg
        src_mem, _ = lift_code(
            f"{helper}_mem", 0x1000,
            encode_seq(ld0_a, Instruction.create_reg_mem(code, R.XMM0, MemoryOperand(R.RDX)), st0_a, ret),
        )
        assert f"{helper}(xmm0, load128(rdx))" in src_mem
        accepted += 1

    # Verify generated C++ does not include <immintrin.h>.
    assert "<immintrin.h>" not in src_reg

    # Negative rejection controls.
    rep_ret = Instruction.create(Code.RETNQ)
    rep_ret.has_rep_prefix = True
    fs_ld = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    fs_ld.segment_prefix = R.FS

    cases = [
        ("sha256rnds2", (ld0, ld1, Instruction.create_reg_reg(Code.SHA256RNDS2_XMM_XMMM128, R.XMM1, R.XMM0), st0, ret),
         "unsupported SHA-256 instruction"),
        ("sha256msg1", (ld0, ld1, Instruction.create_reg_reg(Code.SHA256MSG1_XMM_XMMM128, R.XMM0, R.XMM1), st0, ret),
         "unsupported SHA-256 instruction"),
        ("sha256msg2", (ld0, ld1, Instruction.create_reg_reg(Code.SHA256MSG2_XMM_XMMM128, R.XMM0, R.XMM1), st0, ret),
         "unsupported SHA-256 instruction"),
        ("vex_vmovdqu", (Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), st0, ret),
         "unsupported VEX encoding"),
        ("vex_vpxor", (ld0, ld1, Instruction.create_reg_reg_reg(Code.VEX_VPXOR_XMM_XMM_XMMM128, R.XMM0, R.XMM0, R.XMM1), st0, ret),
         "unsupported VEX encoding"),
        ("evex_vmovdqu32", (Instruction.create_reg_mem(Code.EVEX_VMOVDQU32_XMM_K1Z_XMMM128, R.XMM0, MemoryOperand(R.RCX)), st0, ret),
         "unsupported EVEX encoding"),
        ("nonvol_xmm6_dst", (Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM6, MemoryOperand(R.RCX)), ret),
         "nonvolatile vector register"),
        ("nonvol_xmm15_src", (ld0, Instruction.create_reg_reg(Code.SHA1MSG1_XMM_XMMM128, R.XMM0, R.XMM15), st0, ret),
         "nonvolatile vector register"),
        ("undef_xmm_dst_rnds4", (ld1, Instruction.create_reg_reg_u32(Code.SHA1RNDS4_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0), st0, ret),
         "undefined vector register xmm0"),
        ("undef_xmm_src_nexte", (ld0, Instruction.create_reg_reg(Code.SHA1NEXTE_XMM_XMMM128, R.XMM0, R.XMM2), st0, ret),
         "undefined vector register xmm2"),
        ("undef_xmm_dst_msg1", (ld1, Instruction.create_reg_reg(Code.SHA1MSG1_XMM_XMMM128, R.XMM3, R.XMM1), ret),
         "undefined vector register xmm3"),
        ("undef_xmm_src_msg2", (ld0, Instruction.create_reg_reg(Code.SHA1MSG2_XMM_XMMM128, R.XMM0, R.XMM4), st0, ret),
         "undefined vector register xmm4"),
        ("undef_xmm_store", (st0, ret),
         "undefined vector register xmm0"),
        ("unaligned_movdqa_load", (Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=8, displ_size=1)), st0, ret),
         "unaligned 16-byte memory operand"),
        ("unaligned_movdqa_store", (ld0, Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8, displ=4, displ_size=1), R.XMM0), ret),
         "unaligned 16-byte memory operand"),
        ("unaligned_sha1_mem", (ld0, Instruction.create_reg_mem(Code.SHA1MSG1_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX, displ=8, displ_size=1)), st0, ret),
         "unaligned 16-byte memory operand"),
        ("displaced_movdqu", (Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=16, displ_size=1)), st0, ret),
         "unsupported displaced/out-of-buffer memory operand"),
        ("displaced_movdqa_oob", (Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=16, displ_size=1)), st0, ret),
         "unsupported displaced/out-of-buffer memory operand"),
        ("displaced_sha1_oob", (ld0, Instruction.create_reg_mem(Code.SHA1NEXTE_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX, displ=16, displ_size=1)), st0, ret),
         "unsupported displaced/out-of-buffer memory operand"),
        ("indexed_mem", (Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, index=R.RDX)), st0, ret),
         "unsupported memory operand"),
        ("rip_rel_mem", (Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RIP, displ=0x1100, displ_size=4)), st0, ret),
         "unsupported memory operand"),
        ("non_arg_base_rax", (Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RAX)), st0, ret),
         "unsupported memory base"),
        ("nonvol_base_rbx", (Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RBX)), st0, ret),
         "nonvolatile GPR base"),
        ("store_to_rcx", (ld0, Instruction.create_mem_reg(Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.RCX), R.XMM0), ret),
         "store outside output buffer r8"),
        ("store_to_rdx_movdqa", (ld0_a, Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.RDX), R.XMM0), ret),
         "store outside output buffer r8"),
        ("prefix_segment_fs", (fs_ld, st0, ret),
         "unsupported instruction prefix"),
        ("prefix_rep_ret", (ld0, st0, rep_ret),
         "unsupported instruction prefix"),
        ("branch_jmp", (ld0, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), st0, ret),
         "unsupported control-flow instruction"),
        ("missing_ret", (ld0, ld1, Instruction.create_reg_reg(Code.SHA1MSG1_XMM_XMMM128, R.XMM0, R.XMM1), st0),
         "missing terminal ret"),
        ("trailing_after_ret", (ld0, st0, ret, ld0),
         "unreachable instruction after ret"),
    ]
    for label, seq, reason in cases:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted rejected control")
    return accepted, len(cases)


# --- Full NIST FIPS 180-4 SHA-1 compression driven through lifted DLL exports ---

def _pack_be_lanes(w0, w1, w2, w3):
    return _pack4(w3, w2, w1, w0)


def _unpack_be_lanes(buf):
    w3, w2, w1, w0 = _unpack4(buf)
    return (w0, w1, w2, w3)


def _xor16(a, b):
    return bytes(x ^ y for x, y in zip(a, b))


def sha1_digest_via_funcs(funcs, message, *, use_mem=False):
    """Compute a full FIPS 180-4 SHA-1 digest using only the lifted SHA-1 NI exports."""
    suffix = "mem" if use_mem else "reg"
    rnds4 = [funcs[f"op_sha1rnds4_{i}_{suffix}"] for i in range(4)]
    nexte = funcs[f"op_sha1nexte_{suffix}"]
    msg1 = funcs[f"op_sha1msg1_{suffix}"]
    msg2 = funcs[f"op_sha1msg2_{suffix}"]

    bit_len = len(message) * 8
    padded = message + b"\x80"
    padded += b"\x00" * ((56 - len(padded) % 64) % 64)
    padded += struct.pack(">Q", bit_len)

    h0, h1, h2, h3, h4 = 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    for off in range(0, len(padded), 64):
        words = struct.unpack(">16I", padded[off:off + 64])
        msg = [_pack_be_lanes(*words[4 * i:4 * i + 4]) for i in range(4)]
        for t in range(4, 20):
            t1 = invoke(msg1, msg[t - 4], msg[t - 3])
            t2 = _xor16(t1, msg[t - 2])
            msg.append(invoke(msg2, t2, msg[t - 1]))

        abcd = _pack_be_lanes(h0, h1, h2, h3)
        # Seed round 0 with H4 via SHA1NEXTE: rol32(rol32(h4, 2), 30) == h4.
        e0_seed = _pack_be_lanes(_rol32(h4, 2), 0, 0, 0)
        w0e = invoke(nexte, e0_seed, msg[0])
        prev_abcd = abcd
        abcd = invoke(rnds4[0], abcd, w0e)

        for g in range(1, 20):
            e_plus_w = invoke(nexte, prev_abcd, msg[g])
            prev_abcd = abcd
            abcd = invoke(rnds4[g // 5], abcd, e_plus_w)

        a, b, c, d = _unpack_be_lanes(abcd)
        h4_vec = invoke(nexte, prev_abcd, _pack_be_lanes(h4, 0, 0, 0))
        h0 = (h0 + a) & 0xFFFFFFFF
        h1 = (h1 + b) & 0xFFFFFFFF
        h2 = (h2 + c) & 0xFFFFFFFF
        h3 = (h3 + d) & 0xFFFFFFFF
        h4 = _unpack_be_lanes(h4_vec)[0]

    return struct.pack(">5I", h0, h1, h2, h3, h4).hex()


FIPS_180_4_VECTORS = (
    (b"", "da39a3ee5e6b4b0d3255bfef95601890afd80709"),
    (b"abc", "a9993e364706816aba3e25717850c26c9cd0d89d"),
    (b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "84983e441c3bd26ebaae4aa1f95129e5e54670f1"),
    (
        b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
        b"ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
        "a49b2446a02c645bf419f995b67091253a04a259",
    ),
)


def check_lifted_unconditional(lifted, altered_lifted, pairs):
    """Run portable reference, NIST FIPS 180-4, aliasing, altered-build, and mutant checks on lifted DLLs."""
    total = 0
    for name in EXPORTS:
        for index, (a, b) in enumerate(pairs):
            want = reference(name, a, b)
            got = invoke(lifted[name], a, b)
            if got != want:
                raise AssertionError(
                    f"lifted {name} case {index}: got {got.hex()}, expected {want.hex()} "
                    f"(a={a.hex()} b={b.hex()})"
                )
            total += 1

    # Unaligned buffers on movdqu-only exports.
    a, b = pairs[4]
    for name in UNALIGNED_SAFE_EXPORTS:
        want = reference(name, a, b)
        for mis in ((1, 0, 0), (0, 3, 0), (0, 0, 5), (7, 5, 3)):
            got = invoke(lifted[name], a, b, a_misalign=mis[0], b_misalign=mis[1], out_misalign=mis[2])
            if got != want:
                raise AssertionError(f"lifted {name} misalign {mis}: got {got.hex()}, expected {want.hex()}")
            total += 1

    # Aliasing checks: rcx==rdx, r8==rcx, r8==rdx, and all three aliased.
    for name in (
        "op_sha1rnds4_0_reg", "op_sha1rnds4_2_mem",
        "op_sha1nexte_reg", "op_sha1nexte_mem",
        "op_sha1msg1_reg", "op_sha1msg1_mem",
        "op_sha1msg2_reg", "op_sha1msg2_mem",
        "op_sha1_pipeline",
    ):
        for mode in ("rcx==rdx", "r8==rcx", "r8==rdx", "all"):
            ea, eb = (a, a) if mode in ("rcx==rdx", "all") else (a, b)
            want = reference(name, ea, eb)
            got = invoke_aliased(lifted[name], a, b, mode)
            if got != want:
                raise AssertionError(f"lifted {name} alias {mode}: got {got.hex()}, expected {want.hex()}")
            total += 1

    # Full NIST FIPS 180-4 SHA-1 digests through both reg and mem lifted exports.
    for use_mem in (False, True):
        for msg, expected_hex in FIPS_180_4_VECTORS:
            assert hashlib.sha1(msg).hexdigest() == expected_hex
            got_hex = sha1_digest_via_funcs(lifted, msg, use_mem=use_mem)
            if got_hex != expected_hex:
                raise AssertionError(
                    f"FIPS 180-4 SHA-1 (mem={use_mem}) on {msg!r}: got {got_hex}, expected {expected_hex}"
                )
            total += 1

    # Altered lifted build matches altered reference and diverges from original lifted.
    for name in ALTERED_EXPORTS:
        changed = 0
        for a_in, b_in in pairs[:24]:
            orig = invoke(lifted[name], a_in, b_in)
            alt = invoke(altered_lifted[name], a_in, b_in)
            want_alt = reference(name, a_in, b_in, altered=True)
            if alt != want_alt:
                raise AssertionError(f"altered_lifted {name}: got {alt.hex()}, expected {want_alt.hex()}")
            changed += alt != orig
        if not changed:
            raise AssertionError(f"altered_lifted did not diverge on {name}")

    # Faulty mutant models must disagree with lifted outputs on at least one pair.
    def _mut_nexte_no_rol(x, y):
        _, _, _, d3 = _unpack4(x)
        s0, s1, s2, s3 = _unpack4(y)
        return _pack4(s0, s1, s2, (s3 + d3) & 0xFFFFFFFF)

    def _mut_nexte_keep_dst_low(x, y):
        d0, d1, d2, d3 = _unpack4(x)
        _, _, _, s3 = _unpack4(y)
        return _pack4(d0, d1, d2, (s3 + _rol32(d3, 30)) & 0xFFFFFFFF)

    def _mut_msg2_no_w16_dep(x, y):
        d0, d1, d2, d3 = _unpack4(x)
        s0, s1, s2, s3 = _unpack4(y)
        return _pack4(_rol32(d0 ^ s3, 1), _rol32(d1 ^ s0, 1), _rol32(d2 ^ s1, 1), _rol32(d3 ^ s2, 1))

    mutants = (
        ("op_sha1rnds4_0_reg", lambda x, y: ref_sha1rnds4(x, y, 1)),
        ("op_sha1rnds4_1_reg", lambda x, y: ref_sha1rnds4(x, y, 2)),
        ("op_sha1rnds4_2_reg", lambda x, y: ref_sha1rnds4(x, y, 3)),
        ("op_sha1rnds4_3_reg", lambda x, y: ref_sha1rnds4(x, y, 0)),
        ("op_sha1rnds4_0_reg", lambda x, y: ref_sha1rnds4(y, x, 0)),
        ("op_sha1nexte_reg", _mut_nexte_no_rol),
        ("op_sha1nexte_reg", _mut_nexte_keep_dst_low),
        ("op_sha1msg1_reg", lambda x, y: ref_sha1msg2(x, y)),
        ("op_sha1msg2_reg", _mut_msg2_no_w16_dep),
        ("op_sha1_pipeline", lambda x, y: ref_sha1rnds4(x, y, 0)),
    )
    for name, wrong in mutants:
        diffs = sum(invoke(lifted[name], x, y) != wrong(x, y) for x, y in pairs[:32])
        if not diffs:
            raise AssertionError(f"mutant on {name} agreed with lifted on every pair")

    return total, len(mutants)


def check_native_differential(native, lifted, altered, altered_lifted, pairs):
    """Compare native SHA-NI hardware execution bit-for-bit against lifted DLLs."""
    total = 0
    for name in EXPORTS:
        for index, (a, b) in enumerate(pairs):
            nat = invoke(native[name], a, b)
            lif = invoke(lifted[name], a, b)
            if nat != lif:
                raise AssertionError(
                    f"native vs lifted {name} case {index}: native={nat.hex()} lifted={lif.hex()}"
                )
            total += 1

    # Unaligned buffers on movdqu-only exports.
    a, b = pairs[4]
    for name in UNALIGNED_SAFE_EXPORTS:
        for mis in ((1, 0, 0), (0, 3, 0), (0, 0, 5), (7, 5, 3)):
            nat = invoke(native[name], a, b, a_misalign=mis[0], b_misalign=mis[1], out_misalign=mis[2])
            lif = invoke(lifted[name], a, b, a_misalign=mis[0], b_misalign=mis[1], out_misalign=mis[2])
            if nat != lif:
                raise AssertionError(f"native vs lifted {name} misalign {mis}: native={nat.hex()} lifted={lif.hex()}")
            total += 1

    # Aliasing checks on native vs lifted.
    for name in (
        "op_sha1rnds4_0_reg", "op_sha1rnds4_2_mem",
        "op_sha1nexte_reg", "op_sha1nexte_mem",
        "op_sha1msg1_reg", "op_sha1msg1_mem",
        "op_sha1msg2_reg", "op_sha1msg2_mem",
        "op_sha1_pipeline",
    ):
        for mode in ("rcx==rdx", "r8==rcx", "r8==rdx", "all"):
            nat = invoke_aliased(native[name], a, b, mode)
            lif = invoke_aliased(lifted[name], a, b, mode)
            if nat != lif:
                raise AssertionError(f"native vs lifted {name} alias {mode}: native={nat.hex()} lifted={lif.hex()}")
            total += 1

    # Altered build checks: altered lifted matches altered native and diverges from original native.
    for name in ALTERED_EXPORTS:
        changed = 0
        for a_in, b_in in pairs[:24]:
            alt_nat = invoke(altered[name], a_in, b_in)
            alt_lif = invoke(altered_lifted[name], a_in, b_in)
            if alt_lif != alt_nat:
                raise AssertionError(f"altered {name}: lifted={alt_lif.hex()} != native={alt_nat.hex()}")
            orig_nat = invoke(native[name], a_in, b_in)
            changed += alt_nat != orig_nat
        if not changed:
            raise AssertionError(f"altered native build did not diverge on {name}")

    return total


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

    original_path, lifted_path, altered_path, altered_lifted_path = (Path(p) for p in sys.argv[1:5])

    accepted, rejected = check_translator()
    print(f"{accepted} SHA-1 NI forms accepted; {rejected} rejection controls passed")

    lifted_lib, lifted = bind(lifted_path, "lifted_")
    _, altered_lifted = bind(altered_lifted_path, "lifted_")
    pairs = all_pairs()
    lifted_total, mutant_count = check_lifted_unconditional(lifted, altered_lifted, pairs)
    print(
        f"{lifted_total} portable reference / NIST FIPS 180-4 / aliasing checks on lifted DLLs passed; "
        f"{mutant_count} mutant controls + {len(ALTERED_EXPORTS)} altered-build controls caught"
    )

    if not host_has_sha(lifted_lib):
        print("Host lacks CPUID.07H.00H:EBX[29] (SHA-NI); skipping native differential check")
        sys.exit(77)

    _, native = bind(original_path, "")
    _, altered = bind(altered_path, "")
    native_total = check_native_differential(native, lifted, altered, altered_lifted, pairs)
    print(f"{native_total} native vs lifted SHA-NI differential cases matched bit for bit")

    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
