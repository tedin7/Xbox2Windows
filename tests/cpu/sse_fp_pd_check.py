"""Differential checks, mutation controls and rejection controls for packed double SSE2 arithmetic and compares.

ADDPD, SUBPD, MULPD, DIVPD, MINPD, MAXPD and SQRTPD (xmm and aligned m128 sources) are run natively and
through the translator in the default control word and under all sixteen combinations of MXCSR rounding
mode, DAZ and FTZ, each against the exact reference in sse_fp_pd_ref.py. CMPPD (predicates 0..7, xmm and
aligned m128 sources) gets the same treatment against the predicate rules of that reference. The packed
single checks in sse_fp_check.py provide the buffer, driver and encoding helpers."""

from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Decoder, Instruction, MemoryOperand, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_sse_fp_aot import lift_code, translate
import sse_fp_check as base
import sse_fp_pd_ref as ref

FILL = base.FILL
OPS = ("add", "sub", "mul", "div", "min", "max", "sqrt")
SUFFIXES = ("", "_mem", "_mem_disp", "_self")
EXPORTS = tuple(f"op_{op}pd{suffix}" for op in OPS for suffix in SUFFIXES) + ("op_pd_pipeline",)
CMP_EXPORTS = tuple(f"op_cmppd_{p}" for p in range(8)) + ("op_cmppd_mem", "op_cmppd_mem_disp", "op_cmppd_self")
CMP_PREDICATE = {**{f"op_cmppd_{p}": p for p in range(8)}, "op_cmppd_mem": 5, "op_cmppd_mem_disp": 2,
                 "op_cmppd_self": 0}
# The ALTER_IMM build assembles these register forms with another operation.
ALTERED_OP = {"add": "sub", "sub": "add", "mul": "div", "div": "mul", "min": "max", "max": "min",
              "sqrt": "add"}

SPECIAL = (
    0x0000000000000000, 0x8000000000000000, 0x3FF0000000000000, 0xBFF0000000000000,
    0x3FF8000000000000, 0xC002000000000000, 0x4000000000000000, 0x3FD5555555555555,
    0x3CB0000000000000, 0x3FF0000000000001, 0x4340000000000001, 0x7FF0000000000000,
    0xFFF0000000000000, 0x7FF8000000000000, 0x7FF0000000000001, 0xFFF8000000000001,
    0x7FF4000000000000, 0x0000000000000001, 0x8000000000000003, 0x000FFFFFFFFFFFFF,
    0x0010000000000000, 0x0010000000000001, 0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF,
    0x7FE0000000000000, 0x3FE0000000000000, 0x0008000000000000, 0x5FF0000000000000,
)


def pk_q(*vals):
    return struct.pack("<" + "Q" * len(vals), *(v & ref.MASK64 for v in vals))


def arith_pairs():
    """Every ordered pair of special values in lanes 0 and 1 (the second vector repeats the source
    lanes, so register, [rdx] and [rdx+16] forms see the same pair), then random biased exponents."""
    lane_pairs = [(x, y) for x in SPECIAL for y in SPECIAL]
    pairs = []
    for start in range(0, len(lane_pairs), 2):
        (x0, y0), (x1, y1) = lane_pairs[start:start + 2]
        (x2, _), (x3, _) = lane_pairs[(start * 7 + 2) % len(lane_pairs):][:2]
        pairs.append((pk_q(x0, x1, x2, x3), pk_q(y0, y1, y0, y1)))
    rng = random.Random(0x5A11D)
    exponents = (0, 1, 2, 511, 1022, 1023, 1024, 1534, 2045, 2046)

    def lane():
        return (rng.getrandbits(1) << 63) | (rng.choice(exponents) << 52) | rng.getrandbits(52)

    for _ in range(20):
        pairs.append((pk_q(*(lane() for _ in range(4))), pk_q(*(lane() for _ in range(4)))))
    return pairs


def arith_reference(name, a, b, mode=ref.DEFAULT_MODE, altered=False, quirk=None):
    """(expected 32-byte output, wildcard mask) of an export under an MXCSR mode; a wildcard byte
    marks a lane sse_fp_pd_ref cannot decide (zero everywhere in the default mode)."""
    out, wild = bytearray(FILL), bytearray(32)
    if name == "op_pd_pipeline":
        low, high = ref.chain(a, b, mode, quirk, multiply="div" if altered else "mul")
        out[0:16], wild[0:16] = ref.pack(low)
        out[16:32], wild[16:32] = ref.pack(high)
        return bytes(out), bytes(wild)
    op, suffix = name[3:].split("pd", 1)
    first, source = a[0:16], {"": b[0:16], "_mem": b[0:16], "_mem_disp": b[16:32], "_self": a[0:16]}[suffix]
    if altered and suffix == "":
        op = ALTERED_OP[op]
    elif altered and suffix == "_self":
        op, source = ("add" if op == "sqrt" else op), b[0:16]
    elif altered:
        source = b[16:32] if suffix == "_mem" else b[0:16]
    out[0:16], wild[0:16] = ref.pack(ref.apply(op, first, source, mode, quirk))
    return bytes(out), bytes(wild)


def cmp_reference(name, a, b, mode=ref.DEFAULT_MODE, altered=False, quirk=None):
    """Expected 32-byte output of a compare export: two 64-bit masks, the upper half keeps its fill.
    The ALTER_IMM build assembles predicate (n xor 4)."""
    pred = CMP_PREDICATE[name] ^ (4 if altered else 0)
    source = {"op_cmppd_mem": b[0:16], "op_cmppd_mem_disp": b[16:32], "op_cmppd_self": a[0:16]}.get(name, b[0:16])
    return pk_q(*ref.compare(a[0:16], source, pred, mode, quirk)) + FILL[16:]


def check_cmp_differential(native_path, lifted_path, altered_path):
    native = base.bind_exports(native_path, "", CMP_EXPORTS)
    lifted = base.bind_exports(lifted_path, "lifted_", CMP_EXPORTS)
    altered = base.bind_exports(altered_path, "lifted_", CMP_EXPORTS)
    pairs = arith_pairs()
    for name in CMP_EXPORTS:
        diverged = 0
        for idx, (a, b) in enumerate(pairs):
            want = cmp_reference(name, a, b)
            base.assert_equal_output("native", name, idx, base.invoke(native[name], a, b), want)
            base.assert_equal_output("lifted", name, idx, base.invoke(lifted[name], a, b), want)
            got_alt = base.invoke(altered[name], a, b)
            base.assert_equal_output("altered", name, idx, got_alt, cmp_reference(name, a, b, altered=True))
            diverged += got_alt != want
        if not diverged:
            raise AssertionError(f"altered control {name} did not diverge from original")
    want = cmp_reference("op_cmppd_0", *pairs[0])
    mutated = bytearray(want)
    mutated[9] ^= 0x80
    try:
        base.assert_equal_output("mutation", "op_cmppd_0", 0, bytes(mutated), want)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated compare mask was not caught by comparator")
    print(f"{len(CMP_EXPORTS) * len(pairs)} native/lifted/reference packed double compare cases across "
          f"{len(CMP_EXPORTS)} exports matched bit for bit; altered predicates diverged on every export")


# Each wrong rule must disagree with the processor on some call in the given mode.
CMP_QUIRKS = (
    ("nan_ordered", "op_cmppd_3"), ("nan_ordered", "op_cmppd_7"), ("nan_ordered", "op_cmppd_0"),
    ("zero_signed", "op_cmppd_0"), ("zero_signed", "op_cmppd_1"), ("eq_for_neq", "op_cmppd_4"),
    ("lt_for_le", "op_cmppd_2"), ("swap_ord", "op_cmppd_3"), ("swap_ord", "op_cmppd_7"),
    ("mask32", "op_cmppd_0"), ("numeric", "op_cmppd_5"), ("ignore_daz", "op_cmppd_0"),
    ("ignore_daz", "op_cmppd_1"), ("ignore_daz", "op_cmppd_6"),
)
# UNORD and ORD only look at NaN, and a register compared with itself is equal to itself under any DAZ:
# DAZ cannot change those exports. FTZ and rounding never matter to a compare.
DAZ_BLIND = ("op_cmppd_3", "op_cmppd_7", "op_cmppd_self")


def check_cmp_modes(native_path, lifted_path, harness_path):
    driver = base.bind_mxcsr_driver(harness_path)
    native = base.bind_exports(native_path, "", CMP_EXPORTS)
    lifted = base.bind_exports(lifted_path, "lifted_", CMP_EXPORTS)
    pairs = arith_pairs()
    seen = {}
    modes = tuple(ref.Mode(rc, daz, ftz) for rc in range(4) for daz in (False, True) for ftz in (False, True))
    for mode in modes:
        for name in CMP_EXPORTS:
            for idx, (a, b) in enumerate(pairs):
                got_n = base.invoke_under(driver, native[name], mode.mxcsr(), a, b)
                got_l = base.invoke_under(driver, lifted[name], mode.mxcsr(), a, b)
                want = cmp_reference(name, a, b, mode)
                base.assert_equal_output("native", f"{name} [{mode.label()}]", idx, got_n, want)
                base.assert_equal_output("lifted", f"{name} [{mode.label()}]", idx, got_l, want)
                seen[(name, mode.rc, mode.daz, mode.ftz, idx)] = got_n
    for (name, rc, daz, ftz, idx), got in seen.items():
        if got != seen[(name, 0, daz, False, idx)]:
            raise AssertionError(f"{name} case {idx}: rounding mode or FTZ changed a compare result")
    for name in CMP_EXPORTS:
        moved = sum(seen[(name, 0, True, False, i)] != seen[(name, 0, False, False, i)] for i in range(len(pairs)))
        if bool(moved) == (name in DAZ_BLIND):
            raise AssertionError(f"{name}: DAZ changed {moved} results, expected {'none' if name in DAZ_BLIND else 'some'}")
    for quirk, name in CMP_QUIRKS:
        mode = ref.Mode(ref.RN, True, False) if quirk == "ignore_daz" else ref.DEFAULT_MODE
        caught = sum(
            cmp_reference(name, a, b, mode, quirk=quirk) != seen[(name, mode.rc, mode.daz, mode.ftz, idx)]
            for idx, (a, b) in enumerate(pairs))
        if not caught:
            raise AssertionError(f"reference quirk {quirk} on {name} [{mode.label()}] was never detected")
    print(f"{len(seen)} packed double compare runs in {len(modes)} MXCSR states: native, lifted and reference agree; "
          f"rounding mode and FTZ changed nothing, DAZ changed every export except {', '.join(DAZ_BLIND)}; "
          f"{len(CMP_QUIRKS)} wrong predicate rules detected")


def check_differential(native_path, lifted_path, altered_path):
    native = base.bind_exports(native_path, "", EXPORTS)
    lifted = base.bind_exports(lifted_path, "lifted_", EXPORTS)
    altered = base.bind_exports(altered_path, "lifted_", EXPORTS)
    pairs = arith_pairs()
    total = 0
    for name in EXPORTS:
        for idx, (a, b) in enumerate(pairs):
            want, wild = arith_reference(name, a, b)
            assert not any(wild), name
            base.assert_equal_output("native", name, idx, base.invoke(native[name], a, b), want)
            base.assert_equal_output("lifted", name, idx, base.invoke(lifted[name], a, b), want)
            total += 1
    changed = {}
    for name in EXPORTS:
        diffs = 0
        for idx, (a, b) in enumerate(pairs):
            want_alt, _ = arith_reference(name, a, b, altered=True)
            want_orig, _ = arith_reference(name, a, b)
            got_alt = base.invoke(altered[name], a, b)
            base.assert_equal_output("altered", name, idx, got_alt, want_alt)
            diffs += got_alt != want_orig
        if not diffs:
            raise AssertionError(f"altered control {name} did not diverge from original")
        changed[name] = diffs
    expected, _ = arith_reference("op_pd_pipeline", *pairs[0])
    mutated = bytearray(expected)
    mutated[3] ^= 0x01
    try:
        base.assert_equal_output("mutation", "op_pd_pipeline", 0, bytes(mutated), expected)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated output was not caught by comparator")
    print(f"{total} native/lifted/reference packed double cases across {len(EXPORTS)} exports matched bit for bit; "
          f"altered controls diverged on all {len(changed)} exports (fewest differing cases {min(changed.values())}); "
          f"output mutation control caught")


# Each wrong rule must disagree with the processor on some decided lane in the given mode.
REFERENCE_QUIRKS = (
    ("nan_second_source", "op_addpd", ref.DEFAULT_MODE),
    ("no_quieting", "op_mulpd", ref.DEFAULT_MODE),
    ("mixed_zero_negative", "op_addpd", ref.DEFAULT_MODE),
    ("divide_zero_nan", "op_divpd", ref.DEFAULT_MODE),
    ("zero_returns_first", "op_minpd", ref.DEFAULT_MODE),
    ("zero_returns_first", "op_maxpd", ref.DEFAULT_MODE),
    ("min_as_max", "op_minpd", ref.DEFAULT_MODE),
    ("nan_first_operand", "op_maxpd", ref.DEFAULT_MODE),
    ("sqrt_negative_zero_positive", "op_sqrtpd", ref.DEFAULT_MODE),
    ("sqrt_default_nan_positive", "op_sqrtpd", ref.DEFAULT_MODE),
    ("ignore_rounding_mode", "op_addpd", ref.Mode(ref.RU)),
    ("ignore_rounding_mode", "op_divpd", ref.Mode(ref.RZ)),
    ("ignore_rounding_mode", "op_sqrtpd", ref.Mode(ref.RU)),
    ("ignore_daz", "op_mulpd", ref.Mode(ref.RN, True, False)),
    ("ignore_ftz", "op_mulpd", ref.Mode(ref.RN, False, True)),
)


def check_modes(native_path, lifted_path, harness_path):
    driver = base.bind_mxcsr_driver(harness_path)
    native = base.bind_exports(native_path, "", EXPORTS)
    lifted = base.bind_exports(lifted_path, "lifted_", EXPORTS)
    pairs = arith_pairs()
    seen = {}
    decided = wildcards = 0
    modes = tuple(ref.Mode(rc, daz, ftz) for rc in range(4) for daz in (False, True) for ftz in (False, True))
    for mode in modes:
        for name in EXPORTS:
            for idx, (a, b) in enumerate(pairs):
                label = f"{name} [{mode.label()}]"
                got_n = base.invoke_under(driver, native[name], mode.mxcsr(), a, b)
                got_l = base.invoke_under(driver, lifted[name], mode.mxcsr(), a, b)
                want, wild = arith_reference(name, a, b, mode)
                base.assert_equal_output("native", label, idx, got_n, want, wild)
                # Where the reference abstains, the lifted code must still equal the processor.
                base.assert_equal_output("lifted", label, idx, got_l, got_n)
                seen[(name, mode.mxcsr(), idx)] = got_n
                undecided = sum(1 for byte in wild if byte) // 8
                wildcards += undecided
                decided += 4 - undecided
    default = ref.DEFAULT_MODE.mxcsr()
    for mode in (ref.Mode(ref.RD), ref.Mode(ref.RU), ref.Mode(ref.RZ), ref.Mode(ref.RN, True, False),
                 ref.Mode(ref.RN, False, True)):
        moved = [key for key, got in seen.items() if key[1] == mode.mxcsr() and got != seen[(key[0], default, key[2])]]
        if not moved:
            raise AssertionError(f"MXCSR {mode.label()} did not change any result: the harness is not applying it")
    for quirk, name, mode in REFERENCE_QUIRKS:
        caught = 0
        for idx, (a, b) in enumerate(pairs):
            want, wild = arith_reference(name, a, b, mode, quirk=quirk)
            got = seen[(name, mode.mxcsr(), idx)]
            if any((g ^ w) & ~m & 0xFF for g, w, m in zip(got, want, wild)):
                caught += 1
        if not caught:
            raise AssertionError(f"reference quirk {quirk} on {name} [{mode.label()}] was never detected")
    for name in ("op_addpd", "op_divpd", "op_sqrtpd"):
        for idx, (a, b) in enumerate(pairs):
            base.invoke_under(driver, native[name], ref.Mode(ref.RZ, True, True).mxcsr(), a, b)
            want, wild = arith_reference(name, a, b)
            base.assert_equal_output("restored", name, idx, base.invoke(native[name], a, b), want, wild)
            base.assert_equal_output("restored lifted", name, idx, base.invoke(lifted[name], a, b), want, wild)
    print(f"{len(seen)} packed double runs in {len(modes)} MXCSR states: native, lifted and reference agree on "
          f"{decided} lanes ({wildcards} lanes left undecided by the reference, native equals lifted there); "
          f"each non-default state moved a result; {len(REFERENCE_QUIRKS)} wrong reference rules detected; "
          f"driver restored the control word")


def check_rejections(original_path):
    """Alternate encodings lift to the expected statement; everything else the lifter does not model rejects."""
    ld_xmm0 = Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_xmm1 = Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX))
    ld_xmm4 = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM4, MemoryOperand(R.RCX))
    ld_xmm5 = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM5, MemoryOperand(R.RDX, displ=3, displ_size=1))
    st_xmm0 = Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)
    st_xmm4 = Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM4)
    ret_ins = Instruction.create(Code.RETNQ)

    def reg(code, dst, src):
        return Instruction.create_reg_reg(code, dst, src)

    def mem(code, dst, base_reg, displ=0):
        return Instruction.create_reg_mem(code, dst, MemoryOperand(base_reg, displ=displ, displ_size=1 if displ else 0))

    alt_accept = [
        ("addpd_xmm4_xmm5", (ld_xmm4, ld_xmm5, reg(Code.ADDPD_XMM_XMMM128, R.XMM4, R.XMM5), st_xmm4, ret_ins),
         "xmm4 = op_addpd(xmm4, xmm5);"),
        ("subpd_neg_disp", (ld_xmm0, mem(Code.SUBPD_XMM_XMMM128, R.XMM0, R.RDX, -32), st_xmm0, ret_ins),
         "xmm0 = op_subpd(xmm0, load128((rdx + 0xffffffffffffffe0ull)));"),
        ("mulpd_rcx_source", (ld_xmm0, mem(Code.MULPD_XMM_XMMM128, R.XMM0, R.RCX, 16), st_xmm0, ret_ins),
         "xmm0 = op_mulpd(xmm0, load128((rcx + 0x10ull)));"),
        ("divpd_xmm1_xmm0", (ld_xmm0, ld_xmm1, reg(Code.DIVPD_XMM_XMMM128, R.XMM1, R.XMM0),
                             Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM1), ret_ins),
         "xmm1 = op_divpd(xmm1, xmm0);"),
        ("minpd_r8_source", (ld_xmm0, mem(Code.MINPD_XMM_XMMM128, R.XMM0, R.R8, 16), st_xmm0, ret_ins),
         "xmm0 = op_minpd(xmm0, load128((r8 + 0x10ull)));"),
        ("maxpd_self", (ld_xmm0, reg(Code.MAXPD_XMM_XMMM128, R.XMM0, R.XMM0), st_xmm0, ret_ins),
         "xmm0 = op_maxpd(xmm0, xmm0);"),
        ("sqrtpd_undefined_dest", (ld_xmm1, reg(Code.SQRTPD_XMM_XMMM128, R.XMM5, R.XMM1),
                                   Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM5),
                                   ret_ins), "xmm5 = op_sqrtpd(xmm1);"),
        ("sqrtpd_mem_xmm4", (mem(Code.SQRTPD_XMM_XMMM128, R.XMM4, R.RCX, 0x20), st_xmm4, ret_ins),
         "xmm4 = op_sqrtpd(load128((rcx + 0x20ull)));"),
    ]
    alt_accept += [
        ("cmppd_xmm4_xmm5", (ld_xmm4, ld_xmm5, Instruction.create_reg_reg_u32(Code.CMPPD_XMM_XMMM128_IMM8, R.XMM4, R.XMM5, 6),
                             st_xmm4, ret_ins), "xmm4 = op_cmppd(xmm4, xmm5, 0x6u);"),
        ("cmppd_neg_disp", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPPD_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RDX, displ=-32, displ_size=1), 3), st_xmm0, ret_ins),
         "xmm0 = op_cmppd(xmm0, load128((rdx + 0xffffffffffffffe0ull)), 0x3u);"),
        ("cmppd_rcx_source", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPPD_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RCX, displ=16, displ_size=1), 7), st_xmm0, ret_ins),
         "xmm0 = op_cmppd(xmm0, load128((rcx + 0x10ull)), 0x7u);"),
    ]
    for label, seq, expected in alt_accept:
        raw = base.encode_seq(*seq)
        decoded = list(Decoder(64, raw, ip=0x1000))
        assert len(decoded) == len(seq) and not any(ins.is_invalid for ins in decoded), label
        src, count = lift_code(label, 0x1000, raw)
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if expected not in src:
            raise AssertionError(f"alternate encoding {label} lowered without expected {expected!r}")

    seg_load = mem(Code.ADDPD_XMM_XMMM128, R.XMM0, R.RDX)
    seg_load.segment_prefix = R.GS
    nonlegacy = [
        ("vaddpd_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VADDPD_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins)),
        ("vsubpd_ymm_vex", (Instruction.create_reg_reg_reg(
            Code.VEX_VSUBPD_YMM_YMM_YMMM256, R.YMM2, R.YMM0, R.YMM1), ret_ins)),
        ("vminpd_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VMINPD_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins)),
        ("vsqrtpd_vex", (ld_xmm1, Instruction.create_reg_reg(Code.VEX_VSQRTPD_XMM_XMMM128, R.XMM2, R.XMM1), ret_ins)),
        ("vaddpd_evex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.EVEX_VADDPD_XMM_K1Z_XMM_XMMM128B64, R.XMM2, R.XMM0, R.XMM1), ret_ins)),
        ("vsqrtpd_evex", (ld_xmm1, Instruction.create_reg_reg(
            Code.EVEX_VSQRTPD_XMM_K1Z_XMMM128B64, R.XMM2, R.XMM1), ret_ins)),
    ]
    out_of_scope = [
        (name, (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(code, R.XMM0, R.XMM1), ret_ins))
        for name, code in (
            ("unpckhpd", Code.UNPCKHPD_XMM_XMMM128), ("cvtdq2pd", Code.CVTDQ2PD_XMM_XMMM64),
            ("unpcklpd", Code.UNPCKLPD_XMM_XMMM128), ("haddpd", Code.HADDPD_XMM_XMMM128),
            ("addsubpd", Code.ADDSUBPD_XMM_XMMM128), ("cvtps2pd", Code.CVTPS2PD_XMM_XMMM64),
            ("cvtpd2ps", Code.CVTPD2PS_XMM_XMMM128), ("addsd", Code.ADDSD_XMM_XMMM64),
            ("addss", Code.ADDSS_XMM_XMMM32), ("sqrtsd", Code.SQRTSD_XMM_XMMM64),
            ("minsd", Code.MINSD_XMM_XMMM64), ("maxss", Code.MAXSS_XMM_XMMM32),
            ("rcpps", Code.RCPPS_XMM_XMMM128), ("rsqrtps", Code.RSQRTPS_XMM_XMMM128),
        )
    ] + [
        ("movupd", (Instruction.create_reg_mem(Code.MOVUPD_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret_ins)),
        ("shufpd", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_u32(Code.SHUFPD_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0), ret_ins)),
        ("cvtpi2pd", (Instruction.create_reg_mem(Code.CVTPI2PD_XMM_MMM64, R.XMM0, MemoryOperand(R.RDX)), ret_ins)),
    ]
    def cmppd(dst, src, imm):
        return Instruction.create_reg_reg_u32(Code.CMPPD_XMM_XMMM128_IMM8, dst, src, imm)

    cmppd_rejects = [
        ("cmppd_imm_8", (ld_xmm0, ld_xmm1, cmppd(R.XMM0, R.XMM1, 8), ret_ins), "unsupported cmppd predicate"),
        ("cmppd_imm_1f", (ld_xmm0, ld_xmm1, cmppd(R.XMM0, R.XMM1, 0x1F), ret_ins), "unsupported cmppd predicate"),
        ("cmppd_imm_ff", (ld_xmm0, ld_xmm1, cmppd(R.XMM0, R.XMM1, 0xFF), ret_ins), "unsupported cmppd predicate"),
        ("vcmppd_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VCMPPD_XMM_XMM_XMMM128_IMM8, R.XMM2, R.XMM0, R.XMM1, 0), ret_ins), "non-legacy encoding"),
        ("vcmppd_ymm_vex", (Instruction.create_reg_reg_reg_u32(
            Code.VEX_VCMPPD_YMM_YMM_YMMM256_IMM8, R.YMM2, R.YMM0, R.YMM1, 0), ret_ins), "non-legacy encoding"),
        ("vcmppd_evex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg_u32(
            Code.EVEX_VCMPPD_KR_K1_XMM_XMMM128B64_IMM8, R.K1, R.XMM0, R.XMM1, 0), ret_ins), "non-legacy encoding"),
        ("nonvolatile_xmm7_cmppd_dst", (cmppd(R.XMM7, R.XMM1, 0), ret_ins), "nonvolatile vector register"),
        ("nonvolatile_xmm9_cmppd_src", (ld_xmm0, cmppd(R.XMM0, R.XMM9, 0), ret_ins), "nonvolatile vector register"),
        ("undefined_cmppd_dst", (ld_xmm1, cmppd(R.XMM0, R.XMM1, 0), ret_ins), "undefined vector register xmm0"),
        ("undefined_cmppd_src", (ld_xmm0, cmppd(R.XMM0, R.XMM3, 0), ret_ins), "undefined vector register xmm3"),
        ("unaligned_cmppd_m128", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPPD_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RDX, displ=8, displ_size=1), 0), ret_ins),
         "unaligned 16-byte memory operand"),
        ("unaligned_cmppd_m128_plus1", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPPD_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RCX, displ=1, displ_size=1), 4), ret_ins),
         "unaligned 16-byte memory operand"),
        ("non_arg_base_rax_cmppd", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPPD_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RAX), 0), ret_ins), "unsupported memory base"),
    ]
    other = cmppd_rejects + [
        ("nonvolatile_xmm7_addpd_dst", (reg(Code.ADDPD_XMM_XMMM128, R.XMM7, R.XMM1), ret_ins), "nonvolatile vector register"),
        ("nonvolatile_xmm9_sqrtpd_dst", (ld_xmm1, reg(Code.SQRTPD_XMM_XMMM128, R.XMM9, R.XMM1), ret_ins),
         "nonvolatile vector register"),
        ("nonvolatile_xmm15_minpd_src", (ld_xmm0, reg(Code.MINPD_XMM_XMMM128, R.XMM0, R.XMM15), ret_ins),
         "nonvolatile vector register"),
        ("undefined_addpd_dst", (ld_xmm1, reg(Code.ADDPD_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins), "undefined vector register xmm0"),
        ("undefined_mulpd_src", (ld_xmm0, reg(Code.MULPD_XMM_XMMM128, R.XMM0, R.XMM3), ret_ins), "undefined vector register xmm3"),
        ("undefined_sqrtpd_src", (reg(Code.SQRTPD_XMM_XMMM128, R.XMM0, R.XMM2), ret_ins), "undefined vector register xmm2"),
        ("unaligned_divpd_m128", (ld_xmm0, mem(Code.DIVPD_XMM_XMMM128, R.XMM0, R.RDX, 8), ret_ins),
         "unaligned 16-byte memory operand"),
        ("unaligned_subpd_m128", (ld_xmm0, mem(Code.SUBPD_XMM_XMMM128, R.XMM0, R.RCX, -4), ret_ins),
         "unaligned 16-byte memory operand"),
        ("unaligned_sqrtpd_m128", (mem(Code.SQRTPD_XMM_XMMM128, R.XMM0, R.RCX, 4), ret_ins),
         "unaligned 16-byte memory operand"),
        ("unaligned_maxpd_m128_plus1", (ld_xmm0, mem(Code.MAXPD_XMM_XMMM128, R.XMM0, R.RDX, 1), ret_ins),
         "unaligned 16-byte memory operand"),
        ("non_arg_base_rax_addpd", (ld_xmm0, mem(Code.ADDPD_XMM_XMMM128, R.XMM0, R.RAX), ret_ins), "unsupported memory base"),
        ("non_arg_base_r9_minpd", (ld_xmm0, mem(Code.MINPD_XMM_XMMM128, R.XMM0, R.R9), ret_ins), "unsupported memory base"),
        ("nonvolatile_base_rbx_sqrtpd", (mem(Code.SQRTPD_XMM_XMMM128, R.XMM0, R.RBX), ret_ins), "nonvolatile GPR base"),
        ("rip_rel_addpd", (ld_xmm0, mem(Code.ADDPD_XMM_XMMM128, R.XMM0, R.RIP, 0x1010), ret_ins), "unsupported memory operand"),
        ("rsp_mulpd", (ld_xmm0, mem(Code.MULPD_XMM_XMMM128, R.XMM0, R.RSP), ret_ins), "unsupported memory operand"),
        ("addr32_divpd", (ld_xmm0, mem(Code.DIVPD_XMM_XMMM128, R.XMM0, R.ECX), ret_ins), "unsupported memory operand"),
        ("indexed_subpd", (ld_xmm0, Instruction.create_reg_mem(
            Code.SUBPD_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, R.RDX, 4)), ret_ins), "unsupported memory operand"),
        ("segment_prefix_addpd", (ld_xmm0, seg_load, ret_ins), "unsupported instruction prefix"),
        ("missing_ret", (ld_xmm0, reg(Code.ADDPD_XMM_XMMM128, R.XMM0, R.XMM0)), "missing terminal ret"),
        ("after_ret", (ret_ins, ld_xmm0), "unreachable instruction after ret"),
    ]
    raw_cases = [
        ("prefix_f2_addsd", bytes.fromhex("f20f58c1") + base.encode_seq(ret_ins), "unsupported instruction"),
        ("prefix_f3_addss", bytes.fromhex("f30f58c1") + base.encode_seq(ret_ins), "unsupported instruction"),
        ("prefix_f2_sqrtsd", bytes.fromhex("f20f51c1") + base.encode_seq(ret_ins), "unsupported instruction"),
        ("prefix_f3_sqrtss", bytes.fromhex("f30f51c1") + base.encode_seq(ret_ins), "unsupported instruction"),
        ("truncated_addpd", bytes.fromhex("660f58"), "invalid instruction"),
    ]
    unit = (
        [(label, seq, "non-legacy encoding") for label, seq in nonlegacy]
        + [(label, seq, "unsupported instruction") for label, seq in out_of_scope]
        + other
        + [(label, raw, reason) for label, raw, reason in raw_cases]
    )
    for label, seq, reason in unit:
        raw = seq if isinstance(seq, bytes) else base.encode_seq(*seq)
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator unexpectedly accepted rejected control")

    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_pd_pipeline")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry
    by_label = {label: (seq, reason) for label, seq, reason in unit}
    pe_patches = ("vaddpd_vex", "movupd", "cmppd_imm_8", "unaligned_cmppd_m128", "unpckhpd", "addsd", "unaligned_divpd_m128", "nonvolatile_xmm7_addpd_dst",
                  "undefined_sqrtpd_src", "nonvolatile_base_rbx_sqrtpd", "missing_ret")
    with TemporaryDirectory() as folder:
        patched_path = Path(folder) / "patched.dll"
        for label in pe_patches:
            seq, reason = by_label[label]
            patch = seq if isinstance(seq, bytes) else base.encode_seq(*seq)
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds op_pd_pipeline ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["op_pd_pipeline"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"pe_{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"pe_{label}: PE translate accepted rejected patch")
    print(f"{len(alt_accept)} alternate-encoding lift checks, {len(unit)} unit rejection controls "
          f"({len(nonlegacy)} VEX/EVEX, {len(out_of_scope)} out-of-scope forms, {len(other) + len(raw_cases)} "
          f"ABI/state/addressing/prefix/stream) and {len(pe_patches)} PE-patched rejection controls passed")


if __name__ == "__main__":
    native_dll, lifted_dll, altered_dll, harness_dll = sys.argv[1:5]
    check_differential(native_dll, lifted_dll, altered_dll)
    check_modes(native_dll, lifted_dll, harness_dll)
    check_cmp_differential(native_dll, lifted_dll, altered_dll)
    check_cmp_modes(native_dll, lifted_dll, harness_dll)
    check_rejections(native_dll)
