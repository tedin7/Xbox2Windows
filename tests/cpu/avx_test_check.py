"""Differential, boundary, mutation, and rejection checks for AVX VPTEST, VTESTPS, and VTESTPD."""

import ctypes
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
from cpu_avx_test_aot import lift_code  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

EXPORTS = (
    "op_vptest_xmm_reg",
    "op_vptest_xmm_mem",
    "op_vptest_ymm_reg",
    "op_vptest_ymm_mem",
    "op_vtestps_xmm_reg",
    "op_vtestps_xmm_mem",
    "op_vtestps_ymm_reg",
    "op_vtestps_ymm_mem",
    "op_vtestpd_xmm_reg",
    "op_vtestpd_xmm_mem",
    "op_vtestpd_ymm_reg",
    "op_vtestpd_ymm_mem",
    "op_vptest_ymm_after_xmm_zero",
    "op_vtestps_ymm_after_xmm_zero",
    "op_vtest_self",
    "op_vtest_r8_store_reload",
)

GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))


# --- Independent Python reference: (CF << 0) | (ZF << 1) ---

def _ref_flags(a, b, *, lanes, mask):
    qa = struct.unpack(f"<{lanes}Q", a[:8 * lanes])
    qb = struct.unpack(f"<{lanes}Q", b[:8 * lanes])
    z_acc = 0
    c_acc = 0
    for x, y in zip(qa, qb):
        xm = x & mask
        ym = y & mask
        z_acc |= (xm & ym)
        c_acc |= ((~xm) & ym)
    cf = 1 if c_acc == 0 else 0
    zf = 1 if z_acc == 0 else 0
    return cf | (zf << 1)


def reference(name, a, b):
    full = 0xFFFFFFFFFFFFFFFF
    ps = 0x8000000080000000
    pd = 0x8000000000000000
    if name in ("op_vptest_xmm_reg", "op_vptest_xmm_mem"):
        return _ref_flags(a, b, lanes=2, mask=full)
    if name in ("op_vptest_ymm_reg", "op_vptest_ymm_mem"):
        return _ref_flags(a, b, lanes=4, mask=full)
    if name in ("op_vtestps_xmm_reg", "op_vtestps_xmm_mem"):
        return _ref_flags(a, b, lanes=2, mask=ps)
    if name in ("op_vtestps_ymm_reg", "op_vtestps_ymm_mem"):
        return _ref_flags(a, b, lanes=4, mask=ps)
    if name in ("op_vtestpd_xmm_reg", "op_vtestpd_xmm_mem"):
        return _ref_flags(a, b, lanes=2, mask=pd)
    if name in ("op_vtestpd_ymm_reg", "op_vtestpd_ymm_mem", "op_vtest_r8_store_reload"):
        return _ref_flags(a, b, lanes=4, mask=pd)
    if name == "op_vptest_ymm_after_xmm_zero":
        return _ref_flags(a[:16] + b"\x00" * 16, b, lanes=4, mask=full)
    if name == "op_vtestps_ymm_after_xmm_zero":
        return _ref_flags(a[:16] + b"\x00" * 16, b, lanes=4, mask=ps)
    if name == "op_vtest_self":
        return _ref_flags(a, a, lanes=4, mask=full)
    raise KeyError(name)


def all_pairs():
    """Construct 32-byte input pairs exercising all 4 (CF, ZF) outcomes, float sign bits, NaNs, -0.0, and cross-lane independence."""
    z32 = b"\x00" * 32
    f32 = b"\xff" * 32
    pairs = [
        # All 4 (CF, ZF) combinations for VPTEST/VTESTPS/VTESTPD across 128 and 256 bits.
        (z32, z32),                                           # CF=1, ZF=1 -> 3
        (z32, f32),                                           # CF=0, ZF=1 -> 2
        (f32, f32),                                           # CF=1, ZF=0 -> 1
        (b"\x80" * 16 + b"\x00" * 16, f32),                   # CF=0, ZF=0 -> 0 (256-bit), CF=1/ZF=0 (128-bit)
        (b"\x00" * 16 + b"\x80" * 16, b"\x00" * 16 + b"\x80" * 16),  # upper-lane only
        (b"\x00" * 16 + b"\xff" * 16, b"\xff" * 16 + b"\x00" * 16),  # disjoint halves
        # Float32 sign-bit witnesses: -0.0 (0x80000000), +0.0, +NaN (0x7FC00001), -NaN (0xFFC00001), subnormals.
        (struct.pack("<8I", 0x7FC00001, 0x00000001, 0x7F800000, 0x3F800000,
                     0x7FC00001, 0x00000001, 0x7F800000, 0x3F800000),
         struct.pack("<8I", 0x7FC00001, 0x00000001, 0x7F800000, 0x3F800000,
                     0x7FC00001, 0x00000001, 0x7F800000, 0x3F800000)),  # non-sign bits only -> VTESTPS/PD = 3, VPTEST = 1
        (struct.pack("<8I", 0x80000000, 0x00000000, 0xFFC00001, 0x7FC00001,
                     0x80000001, 0x00000000, 0x80000000, 0x00000000),
         struct.pack("<8I", 0x80000000, 0x7FC00001, 0x80000000, 0x00000001,
                     0xFFC00001, 0x7F7FFFFF, 0x80000000, 0x00000000)),  # sign subset -> VTESTPS CF=1, ZF=0
        (struct.pack("<8I", 0x80000000, 0x00000000, 0x00000000, 0x00000000,
                     0x00000000, 0x00000000, 0x00000000, 0x00000000),
         struct.pack("<8I", 0x80000000, 0x80000000, 0x00000000, 0x00000000,
                     0x00000000, 0x00000000, 0x00000000, 0x80000000)),  # both a&b and ~a&b non-zero in sign bits -> 0
        # Float64 sign-bit witnesses: bit 31 set vs bit 63 set (-0.0, +NaN, -NaN, subnormal).
        (struct.pack("<4Q", 0x0000000080000000, 0x7FF8000000000001, 0x0000000000000001, 0x7FF0000000000000),
         struct.pack("<4Q", 0x0000000080000000, 0x7FF8000000000001, 0x0000000000000001, 0x7FF0000000000000)),
        (struct.pack("<4Q", 0x8000000000000000, 0xFFF8000000000001, 0x0000000080000000, 0x8000000000000001),
         struct.pack("<4Q", 0x8000000000000000, 0x0000000080000000, 0x0000000080000000, 0x8000000000000000)),
        (struct.pack("<4Q", 0x8000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000),
         struct.pack("<4Q", 0x8000000000000000, 0x8000000000000000, 0x0000000000000000, 0x8000000000000000)),
        (struct.pack("<4Q", 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000),
         struct.pack("<4Q", 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x8000000000000000)),
        # Single-lane bit 7 / bit 31 / bit 63 sweep across all 8 dwords.
        *[
            (struct.pack("<8I", *(0x80000000 if j == i else 0 for j in range(8))),
             struct.pack("<8I", *(0x80000000 if j == k else 0 for j in range(8))))
            for i in range(8) for k in (i, (i + 1) % 8, 7)
        ],
    ]
    # Sparse/structured sign-bit combinations so all 4 outcomes are densely hit.
    rng = random.Random(0xA7E5)
    for _ in range(120):
        a = rng.randbytes(32)
        b = rng.randbytes(32)
        pairs.append((a, b))
        # Also construct subset b = a & b to ensure CF=1 cases appear frequently for full-bit VPTEST.
        pairs.append((a, bytes(x & y for x, y in zip(a, b))))
    return pairs


def invoke(fn, a, b, a_offset=0, out_offset=0):
    buf_a = ctypes.create_string_buffer(GUARD_PRE + b"\x00" * a_offset + a + GUARD_POST,
                                        len(GUARD_PRE) + a_offset + 32 + len(GUARD_POST))
    buf_b = ctypes.create_string_buffer(GUARD_PRE + b + GUARD_POST,
                                        len(GUARD_PRE) + 32 + len(GUARD_POST))
    buf_out = ctypes.create_string_buffer(GUARD_PRE + b"\x5a" * (out_offset + 32) + GUARD_POST,
                                          len(GUARD_PRE) + out_offset + 32 + len(GUARD_POST))
    res = fn(
        ctypes.addressof(buf_a) + len(GUARD_PRE) + a_offset,
        ctypes.addressof(buf_b) + len(GUARD_PRE),
        ctypes.addressof(buf_out) + len(GUARD_PRE) + out_offset,
    )
    assert buf_a.raw[:16] == GUARD_PRE and buf_a.raw[-16:] == GUARD_POST, "input A guards corrupted"
    assert buf_b.raw[:16] == GUARD_PRE and buf_b.raw[-16:] == GUARD_POST, "input B guards corrupted"
    assert buf_out.raw[:16] == GUARD_PRE and buf_out.raw[-16:] == GUARD_POST, "output guards corrupted"
    return res, buf_out.raw[len(GUARD_PRE) + out_offset:len(GUARD_PRE) + out_offset + 32]


def bind(path, prefix):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = ctypes.c_uint64
        funcs[name] = fn
    return lib, funcs


def host_has_avx(lifted_lib):
    if hasattr(lifted_lib, "host_has_avx"):
        lifted_lib.host_has_avx.argtypes = []
        lifted_lib.host_has_avx.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_avx())
    try:
        return " avx" in Path("/proc/cpuinfo").read_text(encoding="ascii", errors="replace")
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
    ld_x0 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_x1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX))
    ld_y0 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX))
    ld_y1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM1, MemoryOperand(R.RDX))
    setb_al = Instruction.create_reg(Code.SETB_RM8, R.AL)
    sete_r9b = Instruction.create_reg(Code.SETE_RM8, R.R9L)
    movzx_eax = Instruction.create_reg_reg(Code.MOVZX_R32_RM8, R.EAX, R.AL)
    movzx_r9d = Instruction.create_reg_reg(Code.MOVZX_R32_RM8, R.R9D, R.R9L)
    shl_r9 = Instruction.create_reg_u32(Code.SHL_RM32_1, R.R9D, 1)
    or_eax = Instruction.create_reg_reg(Code.OR_R32_RM32, R.EAX, R.R9D)
    pack = (setb_al, sete_r9b, movzx_eax, movzx_r9d, shl_r9, or_eax, ret)

    # Positive checks for all 6 Zen2 AVX test encodings (reg & mem).
    accepted = 0
    for code, wide, helper in (
        (Code.VEX_VPTEST_XMM_XMMM128, False, "op_vptest_128"),
        (Code.VEX_VPTEST_YMM_YMMM256, True, "op_vptest_256"),
        (Code.VEX_VTESTPS_XMM_XMMM128, False, "op_vtestps_128"),
        (Code.VEX_VTESTPS_YMM_YMMM256, True, "op_vtestps_256"),
        (Code.VEX_VTESTPD_XMM_XMMM128, False, "op_vtestpd_128"),
        (Code.VEX_VTESTPD_YMM_YMMM256, True, "op_vtestpd_256"),
    ):
        r0, r1 = (R.YMM0, R.YMM1) if wide else (R.XMM0, R.XMM1)
        loads = (ld_y0, ld_y1) if wide else (ld_x0, ld_x1)
        src_reg, _ = lift_code(f"{helper}_reg", 0x1000, encode_seq(*loads, Instruction.create_reg_reg(code, r0, r1), *pack))
        assert f"{helper}(ymm0, ymm1)" in src_reg, helper
        src_mem, _ = lift_code(
            f"{helper}_mem", 0x1000,
            encode_seq(loads[0], Instruction.create_reg_mem(code, r0, MemoryOperand(R.RDX)), *pack))
        assert f"{helper}(ymm0, {'load256' if wide else 'load128'}(rdx))" in src_mem, helper
        accepted += 1

    # Negative controls: legacy SSE PTEST, EVEX, nonvol XMM/YMM/GPR, argument GPR writes,
    # undefined regs, non-flat mem, flag read without v*test*, clobbered flags, missing/trailing ret.
    cases = [
        ("legacy_ptest", (ld_x0, ld_x1, Instruction.create_reg_reg(Code.PTEST_XMM_XMMM128, R.XMM0, R.XMM1), *pack),
         "unsupported legacy SSE encoding"),
        ("evex_vptestmd", (ld_x0, ld_x1, Instruction.create_reg_reg_reg(Code.EVEX_VPTESTMD_KR_K1_XMM_XMMM128B32, R.K1, R.XMM0, R.XMM1), ret),
         "unsupported EVEX instruction"),
        ("evex_vptestmq", (ld_y0, ld_y1, Instruction.create_reg_reg_reg(Code.EVEX_VPTESTMQ_KR_K1_YMM_YMMM256B64, R.K1, R.YMM0, R.YMM1), ret),
         "unsupported EVEX instruction"),
        ("nonvol_xmm6", (ld_x0, Instruction.create_reg_reg(Code.VEX_VPTEST_XMM_XMMM128, R.XMM0, R.XMM6), *pack),
         "nonvolatile vector register"),
        ("nonvol_ymm15", (ld_y0, Instruction.create_reg_reg(Code.VEX_VTESTPS_YMM_YMMM256, R.YMM0, R.YMM15), *pack),
         "nonvolatile vector register"),
        ("nonvol_gpr_rbx", (ld_x0, ld_x1, Instruction.create_reg_reg(Code.VEX_VPTEST_XMM_XMMM128, R.XMM0, R.XMM1),
                            Instruction.create_reg(Code.SETB_RM8, R.BL), ret),
         "nonvolatile GPR"),
        ("nonvol_gpr_r12", (Instruction.create_reg_reg(Code.XOR_R32_RM32, R.R12D, R.R12D), ret),
         "nonvolatile GPR"),
        ("write_arg_r8_xor", (Instruction.create_reg_reg(Code.XOR_R32_RM32, R.R8D, R.R8D), ret),
         "unsupported write to argument GPR"),
        ("write_arg_r8_setb", (ld_x0, ld_x1, Instruction.create_reg_reg(Code.VEX_VPTEST_XMM_XMMM128, R.XMM0, R.XMM1),
                               Instruction.create_reg(Code.SETB_RM8, R.R8L), ret),
         "unsupported write to argument GPR"),
        ("write_arg_rcx_mov", (Instruction.create_reg_reg(Code.XOR_R32_RM32, R.EAX, R.EAX),
                               Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RCX, R.RAX), ret),
         "unsupported write to argument GPR"),
        ("undef_vec_src0", (ld_x1, Instruction.create_reg_reg(Code.VEX_VTESTPD_XMM_XMMM128, R.XMM2, R.XMM1), *pack),
         "undefined vector register ymm2"),
        ("undef_vec_src1", (ld_y0, Instruction.create_reg_reg(Code.VEX_VPTEST_YMM_YMMM256, R.YMM0, R.YMM3), *pack),
         "undefined vector register ymm3"),
        ("undef_rax_ret", (ret,), "undefined GPR rax"),
        ("undef_rax_partial_setb", (ld_x0, ld_x1, Instruction.create_reg_reg(Code.VEX_VPTEST_XMM_XMMM128, R.XMM0, R.XMM1),
                                    setb_al, ret),
         "undefined GPR rax"),
        ("undef_r9_or", (Instruction.create_reg_reg(Code.XOR_R32_RM32, R.EAX, R.EAX), or_eax, ret),
         "undefined GPR r9"),
        ("undef_r9b_movzx", (Instruction.create_reg_reg(Code.MOVZX_R32_RM8, R.EAX, R.R9L), ret),
         "undefined GPR r9"),
        ("mem_displaced", (ld_x0, Instruction.create_reg_mem(Code.VEX_VPTEST_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX, displ=16, displ_size=1)), *pack),
         "unsupported memory operand"),
        ("mem_indexed", (ld_y0, Instruction.create_reg_mem(Code.VEX_VTESTPS_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX, index=R.RDX)), *pack),
         "unsupported memory operand"),
        ("mem_non_arg_base", (ld_x0, Instruction.create_reg_mem(Code.VEX_VTESTPD_XMM_XMMM128, R.XMM0, MemoryOperand(R.RAX)), *pack),
         "unsupported memory operand"),
        ("mem_rip_rel", (ld_x0, Instruction.create_reg_mem(Code.VEX_VPTEST_XMM_XMMM128, R.XMM0, MemoryOperand(R.RIP, displ=0x1100, displ_size=4)), *pack),
         "unsupported memory operand"),
        ("store_to_rcx", (ld_y0, Instruction.create_mem_reg(Code.VEX_VMOVDQU_YMMM256_YMM, MemoryOperand(R.RCX), R.YMM0),
                          Instruction.create_reg_reg(Code.XOR_R32_RM32, R.EAX, R.EAX), ret),
         "store outside output buffer r8"),
        ("flag_read_without_test_setb", (setb_al, movzx_eax, ret),
         "flag read without preceding vptest/vtestps/vtestpd"),
        ("flag_read_without_test_sete", (sete_r9b, movzx_r9d, Instruction.create_reg_reg(Code.MOV_R32_RM32, R.EAX, R.R9D), ret),
         "flag read without preceding vptest/vtestps/vtestpd"),
        ("flag_read_after_xor_clobber", (ld_x0, ld_x1, Instruction.create_reg_reg(Code.VEX_VPTEST_XMM_XMMM128, R.XMM0, R.XMM1),
                                         Instruction.create_reg_reg(Code.XOR_R32_RM32, R.R9D, R.R9D), sete_r9b, ret),
         "flag read without preceding vptest/vtestps/vtestpd"),
        ("missing_ret", (ld_x0, ld_x1, Instruction.create_reg_reg(Code.VEX_VPTEST_XMM_XMMM128, R.XMM0, R.XMM1), setb_al, movzx_eax),
         "missing terminal ret"),
        ("trailing_after_ret", (Instruction.create_reg_reg(Code.XOR_R32_RM32, R.EAX, R.EAX), ret,
                                Instruction.create_reg_reg(Code.XOR_R32_RM32, R.EAX, R.EAX)),
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


def check_differential(native, lifted, altered, altered_lifted):
    pairs = all_pairs()
    total = 0
    seen_outcomes = {name: set() for name in EXPORTS}
    for name in EXPORTS:
        for index, (a, b) in enumerate(pairs):
            want = reference(name, a, b)
            for label, funcs in (("native", native), ("lifted", lifted)):
                got, scratch = invoke(funcs[name], a, b)
                if got != want:
                    raise AssertionError(
                        f"{label} {name} case {index}: got {got}, expected {want} "
                        f"(a={a.hex()} b={b.hex()})")
                if name == "op_vtest_r8_store_reload" and scratch != b:
                    raise AssertionError(f"{label} {name} case {index}: scratch mismatch")
            seen_outcomes[name].add(want)
            total += 1

    # Confirm all 4 (CF, ZF) combinations (0..3) were hit for all 6 base encodings (reg & mem).
    for name in EXPORTS[:12]:
        if seen_outcomes[name] != {0, 1, 2, 3}:
            raise AssertionError(f"{name}: only covered outcomes {sorted(seen_outcomes[name])}, expected [0, 1, 2, 3]")
    if seen_outcomes["op_vtest_self"] != {1, 3}:
        raise AssertionError(f"op_vtest_self: expected outcomes [1, 3], got {sorted(seen_outcomes['op_vtest_self'])}")

    # Unaligned buffers; guards must remain intact.
    a, b = pairs[7]
    for name in EXPORTS:
        for a_offset, out_offset in ((1, 0), (0, 3), (7, 5)):
            want = reference(name, a, b)
            for label, funcs in (("native", native), ("lifted", lifted)):
                got, _ = invoke(funcs[name], a, b, a_offset, out_offset)
                if got != want:
                    raise AssertionError(f"{label} {name} offsets {a_offset}/{out_offset}")
            total += 1

    # Altered build checks: altered lifted matches altered native and diverges from original.
    altered_names = (
        "op_vptest_xmm_reg", "op_vptest_ymm_reg",
        "op_vtestps_xmm_reg", "op_vtestps_ymm_reg",
        "op_vtestpd_xmm_reg", "op_vtestpd_ymm_reg",
    )
    for name in altered_names:
        changed = 0
        for a, b in pairs:
            alt_nat, _ = invoke(altered[name], a, b)
            alt_lif, _ = invoke(altered_lifted[name], a, b)
            if alt_lif != alt_nat:
                raise AssertionError(f"altered {name}: lifted ({alt_lif}) != native ({alt_nat})")
            orig_nat, _ = invoke(native[name], a, b)
            changed += alt_nat != orig_nat
        if not changed:
            raise AssertionError(f"altered build did not diverge on {name}")

    # Faulty Python mutant models: must disagree with lifted outputs on at least one pair.
    mutants = (
        ("op_vtestps_xmm_reg", lambda x, y: reference("op_vptest_xmm_reg", x, y)),
        ("op_vtestps_ymm_reg", lambda x, y: reference("op_vptest_ymm_reg", x, y)),
        ("op_vtestpd_xmm_reg", lambda x, y: reference("op_vtestps_xmm_reg", x, y)),
        ("op_vtestpd_ymm_reg", lambda x, y: reference("op_vtestps_ymm_reg", x, y)),
        ("op_vptest_ymm_reg", lambda x, y: reference("op_vptest_xmm_reg", x, y)),
        ("op_vptest_ymm_reg", lambda x, y: reference("op_vptest_ymm_reg", y, x)),
        ("op_vtestps_ymm_reg", lambda x, y: reference("op_vtestps_xmm_reg", x, y)),
        ("op_vtestpd_ymm_reg", lambda x, y: reference("op_vtestpd_xmm_reg", x, y)),
        ("op_vptest_ymm_after_xmm_zero", lambda x, y: reference("op_vptest_ymm_reg", x, y)),
        ("op_vtestps_ymm_after_xmm_zero", lambda x, y: reference("op_vtestps_ymm_reg", x, y)),
    )
    for name, wrong in mutants:
        diffs = sum(invoke(lifted[name], x, y)[0] != wrong(x, y) for x, y in pairs)
        if not diffs:
            raise AssertionError(f"mutant on {name} agreed with lifted on every pair")

    return total, len(mutants), len(altered_names)


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
    print(f"{accepted} AVX test forms accepted; {rejected} rejection controls passed")

    lifted_lib, lifted = bind(lifted_path, "lifted_")
    if not host_has_avx(lifted_lib):
        print("Host lacks CPUID.01H:ECX[27:28] or OS XGETBV XMM/YMM; skipping native differential check")
        sys.exit(77)

    _, native = bind(original_path, "")
    _, altered = bind(altered_path, "")
    _, altered_lifted = bind(altered_lifted_path, "lifted_")
    total, caught, altered_count = check_differential(native, lifted, altered, altered_lifted)
    print(f"{total} native/lifted/reference cases across {len(EXPORTS)} exports matched bit for bit; "
          f"{caught} mutant controls + {altered_count} altered-build controls caught")

    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
