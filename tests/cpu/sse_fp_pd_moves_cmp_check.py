"""Differential, MXCSR, /O2, mutant, and rejection checks for legacy SSE2 CMPSD, MOVLPD, MOVHPD, and MOVNTPD."""

import ctypes
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Decoder, Formatter, FormatterSyntax, Instruction, MemoryOperand, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402
from cpu_sse_fp_aot import lift_code, translate  # noqa: E402
import sse_fp_check as base  # noqa: E402
import sse_fp_pd_ref as ref  # noqa: E402

FILL = base.FILL
MASK64 = ref.MASK64

CMPSD_EXPORTS = (
    tuple(f"op_cmpsd_{p}" for p in range(8))
    + ("op_cmpsd_mem", "op_cmpsd_mem_plus1", "op_cmpsd_self")
)
MOVE_EXPORTS = (
    "op_movlpd_load",
    "op_movlpd_store",
    "op_movhpd_load",
    "op_movhpd_store",
    "op_movlpd_load_plus1",
    "op_movhpd_load_plus1",
    "op_movlpd_store_plus1",
    "op_movhpd_store_plus1",
    "op_movntpd",
    "op_movntpd_disp16",
    "op_pd_moves_cmp_pipeline",
    "op_pd_moves_cmp_discarded_read",
)
EXPORTS = CMPSD_EXPORTS + MOVE_EXPORTS

ALTERED_EXPORTS = (
    tuple(f"op_cmpsd_{p}" for p in range(8))
    + (
        "op_cmpsd_mem",
        "op_cmpsd_mem_plus1",
        "op_cmpsd_self",
        "op_movlpd_load",
        "op_movlpd_store",
        "op_movhpd_load",
        "op_movhpd_store",
        "op_movntpd",
        "op_pd_moves_cmp_pipeline",
    )
)

DOUBLE_EDGE_QWORDS = (
    0x0000000000000000,  # +0.0
    0x8000000000000000,  # -0.0
    0x3FF0000000000000,  # +1.0
    0xBFF0000000000000,  # -1.0
    0x3FF8000000000000,  # +1.5
    0xC002000000000000,  # -2.25
    0x7FF0000000000000,  # +Inf
    0xFFF0000000000000,  # -Inf
    0x7FF8000000000000,  # +qNaN canonical
    0x7FF8000000000001,  # +qNaN payload
    0xFFF8000000000000,  # -qNaN canonical
    0xFFF8000000000001,  # -qNaN payload
    0x7FF0000000000001,  # +sNaN payload
    0xFFF0000000000001,  # -sNaN payload
    0x7FF7FFFFFFFFFFFF,  # +sNaN max payload
    0x0000000000000001,  # min +subnormal
    0x000FFFFFFFFFFFFF,  # max +subnormal
    0x8000000000000001,  # min -subnormal
    0x800FFFFFFFFFFFFF,  # max -subnormal
    0x0010000000000000,  # min +normal
    0x7FEFFFFFFFFFFFFF,  # max +finite
    0xFFEFFFFFFFFFFFFF,  # max -finite
)


def pk_q(*vals):
    return struct.pack("<" + "Q" * len(vals), *(v & MASK64 for v in vals))


def u64(buf):
    return struct.unpack("<Q", buf)[0]


def u64s(buf):
    return struct.unpack("<" + "Q" * (len(buf) // 8), buf)


def cmpsd_pairs():
    """Ordered pairs across DOUBLE_EDGE_QWORDS with dirty upper/shifted bytes, plus random pairs."""
    rng = random.Random(0xC35D64)
    pairs = []
    # Sweep every ordered pair of DOUBLE_EDGE_QWORDS on lane 0, with dirty [127:64] and [255:128].
    for idx, x in enumerate(DOUBLE_EDGE_QWORDS):
        for jdx, y in enumerate(DOUBLE_EDGE_QWORDS):
            upper_a = DOUBLE_EDGE_QWORDS[(idx * 5 + jdx + 1) % len(DOUBLE_EDGE_QWORDS)]
            upper_b = DOUBLE_EDGE_QWORDS[(jdx * 3 + idx + 2) % len(DOUBLE_EDGE_QWORDS)]
            a = pk_q(x, upper_a, upper_b, x ^ 0xA5A5A5A55A5A5A5A)
            # Place y at +0 and +16, plus shifted bytes for +1/+4/+8 offsets in random tests below.
            b = pk_q(y, upper_b, upper_a, y ^ 0x5A5A5A5AA5A5A5A5)
            pairs.append((a, b))
    # Explicit +1 unaligned offset pairs for every edge class (zeros, Inf, qNaN, sNaN, subnormals).
    for x in (0x0000000000000000, 0x8000000000000000, 0x7FF0000000000000, 0xFFF0000000000000,
              0x7FF8000000000001, 0xFFF0000000000001, 0x0000000000000001, 0x800FFFFFFFFFFFFF):
        for y in (0x0000000000000000, 0x8000000000000000, 0x7FF8000000000000, 0x7FF0000000000001,
                  0x0000000000000001, 0x3FF0000000000000):
            a = pk_q(x, rng.getrandbits(64)) + rng.randbytes(16)
            b = b"\xAA" + pk_q(y, rng.getrandbits(64)) + rng.randbytes(15)
            pairs.append((a, b[:32]))
    for _ in range(16):
        pairs.append((rng.randbytes(32), rng.randbytes(32)))
    return pairs


def move_pairs():
    pairs = [
        (bytes(range(0x10, 0x30)), bytes(range(0x80, 0xA0))),
        (b"\xFF" * 32, b"\x00" * 32),
        (bytes((0x17 * i + 3) & 0xFF for i in range(32)), bytes((0x31 * i + 7) & 0xFF for i in range(32))),
        (pk_q(0x0000000000000000, 0x8000000000000000, 0x7FF8000000000001, 0xFFF0000000000001),
         pk_q(0x7FF0000000000000, 0x0000000000000001, 0x800FFFFFFFFFFFFF, 0x3FF8000000000000)),
    ]
    rng = random.Random(0x55E2D64)
    for _ in range(12):
        pairs.append((rng.randbytes(32), rng.randbytes(32)))
    return pairs


def inputs_for(name):
    if name in CMPSD_EXPORTS or name == "op_pd_moves_cmp_pipeline":
        return cmpsd_pairs()
    return move_pairs()


def reference(name, a, b, mode=ref.DEFAULT_MODE, altered=False, quirk=None):
    """Expected 32-byte output region; unwritten bytes keep FILL."""
    out = bytearray(FILL)
    aq0, aq1 = u64s(a[:16])
    bq0, bq1 = u64s(b[:16])

    if name.startswith("op_cmpsd_"):
        if name == "op_cmpsd_mem":
            pred = 5
            src = u64(b[16:24]) if altered else bq0
        elif name == "op_cmpsd_mem_plus1":
            pred = 6 if altered else 2
            src = u64(b[1:9])
        elif name == "op_cmpsd_self":
            pred = 0
            src = bq0 if altered else aq0
        else:
            pred = int(name[-1]) ^ (4 if altered else 0)
            src = bq0
        out[0:16] = pk_q(ref.cmp_lane(aq0, src, pred, mode, quirk), aq1)
    elif name == "op_movlpd_load":
        out[0:16] = pk_q(aq0, bq0) if altered else pk_q(bq0, aq1)
    elif name == "op_movhpd_load":
        out[0:16] = pk_q(bq0, aq1) if altered else pk_q(aq0, bq0)
    elif name == "op_movlpd_store":
        out[8:16] = pk_q(aq1 if altered else aq0)
    elif name == "op_movhpd_store":
        out[8:16] = pk_q(aq0 if altered else aq1)
    elif name == "op_movlpd_load_plus1":
        out[0:16] = pk_q(u64(b[1:9]), aq1)
    elif name == "op_movhpd_load_plus1":
        out[0:16] = pk_q(aq0, u64(b[1:9]))
    elif name == "op_movlpd_store_plus1":
        out[9:17] = pk_q(aq0)
    elif name == "op_movhpd_store_plus1":
        out[9:17] = pk_q(aq1)
    elif name == "op_movntpd":
        if altered:
            out[16:32] = a[0:16]
        else:
            out[0:16] = a[0:16]
    elif name == "op_movntpd_disp16":
        out[16:32] = a[1:17]
    elif name == "op_pd_moves_cmp_pipeline":
        xmm0_lo = u64(b[8:16])
        xmm0_hi = aq1
        xmm1_lo = bq0
        xmm2_lo = u64(b[16:24])
        xmm2_hi = u64(a[24:32])
        pred0 = 0 if altered else 4
        xmm0_lo = ref.cmp_lane(xmm0_lo, xmm1_lo, pred0, mode, quirk)
        xmm2_lo = ref.cmp_lane(xmm2_lo, u64(b[4:12]), 2, mode, quirk)
        out[0:16] = pk_q(xmm0_lo, xmm0_hi)
        out[16:24] = pk_q(xmm2_hi)
        out[24:32] = pk_q(xmm2_lo)
    elif name == "op_pd_moves_cmp_discarded_read":
        out[0:16] = a[0:16]
    else:
        raise KeyError(name)
    return bytes(out)


def bind_exports(dll_path, prefix=""):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def check_discarded_read(fns, label):
    payload = move_pairs()[0][0]
    buf_a, ptr_a, _, _, _ = base.alloc_aligned_buffer(payload)
    buf_out, ptr_out, o_pre, o_off, o_post = base.alloc_aligned_buffer(FILL)
    fns["op_pd_moves_cmp_discarded_read"](ptr_a, ptr_a, ptr_out)
    got_ok = base.check_guards_and_read(buf_out, o_pre, o_off, o_post, "output")
    if got_ok != payload[:16] + FILL[16:]:
        raise AssertionError(f"{label} op_pd_moves_cmp_discarded_read(valid): {got_ok.hex()}")

    buf_out2, ptr_out2, o_pre2, o_off2, o_post2 = base.alloc_aligned_buffer(FILL)
    try:
        fns["op_pd_moves_cmp_discarded_read"](ptr_a, 0, ptr_out2)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_pd_moves_cmp_discarded_read(NULL) dropped faulting m64 read")
    if base.check_guards_and_read(buf_out2, o_pre2, o_off2, o_post2, "output") != FILL:
        raise AssertionError(f"{label} op_pd_moves_cmp_discarded_read(NULL) wrote output after fault")


def verify_positive_suite(native_fns, target_fns, label):
    total = 0
    for name in EXPORTS:
        fn_n, fn_t = native_fns[name], target_fns[name]
        for idx, (a, b) in enumerate(inputs_for(name)):
            want = reference(name, a, b)
            base.assert_equal_output("native", name, idx, base.invoke(fn_n, a, b), want)
            base.assert_equal_output(label, name, idx, base.invoke(fn_t, a, b), want)
            total += 1
    check_discarded_read(native_fns, "native")
    check_discarded_read(target_fns, label)
    return total


def check_altered_and_mutants(lifted_fns, altered_native_fns, altered_lifted_fns):
    diverged = {}
    for name in ALTERED_EXPORTS:
        diffs = 0
        for idx, (a, b) in enumerate(inputs_for(name)):
            want_alt = reference(name, a, b, altered=True)
            want_orig = reference(name, a, b)
            got_an = base.invoke(altered_native_fns[name], a, b)
            got_al = base.invoke(altered_lifted_fns[name], a, b)
            base.assert_equal_output("altered_native", name, idx, got_an, want_alt)
            base.assert_equal_output("altered_lifted", name, idx, got_al, want_alt)
            if got_al != want_orig:
                diffs += 1
        if not diffs:
            raise AssertionError(f"altered control {name} did not diverge from original")
        diverged[name] = diffs

    # Mutant models for moves & compares must diverge from lifted outputs.
    cmp_sample = cmpsd_pairs()
    move_sample = move_pairs()
    mutant_specs = [
        ("op_movlpd_load", move_sample, lambda x, y: reference("op_movhpd_load", x, y)),
        ("op_movhpd_load", move_sample, lambda x, y: reference("op_movlpd_load", x, y)),
        ("op_movlpd_store", move_sample, lambda x, y: reference("op_movhpd_store", x, y)),
        ("op_movhpd_store", move_sample, lambda x, y: reference("op_movlpd_store", x, y)),
        ("op_movlpd_load", move_sample, lambda x, y: pk_q(u64(y[:8]), 0) + FILL[16:]),  # clobbered [127:64]
        ("op_movhpd_load", move_sample, lambda x, y: pk_q(0, u64(y[:8])) + FILL[16:]),  # clobbered [63:0]
        ("op_cmpsd_0", cmp_sample, lambda x, y: pk_q(u64(reference("op_cmpsd_0", x, y)[:8]), 0) + FILL[16:]),
        ("op_cmpsd_0", cmp_sample, lambda x, y: pk_q(u64(reference("op_cmpsd_0", x, y)[:8]) & 0xFFFFFFFF, u64(x[8:16])) + FILL[16:]),
        ("op_movntpd", move_sample, lambda x, y: reference("op_movntpd", x, y, altered=True)),
    ]
    for name, pairs, wrong in mutant_specs:
        if not any(base.invoke(lifted_fns[name], a, b) != wrong(a, b) for a, b in pairs):
            raise AssertionError(f"mutant on {name} agreed with lifted on every pair")

    # Output mutation negative control: 1-bit flip must be caught.
    sample_a, sample_b = cmp_sample[0]
    expected = reference("op_pd_moves_cmp_pipeline", sample_a, sample_b)
    mutated = bytearray(expected)
    mutated[25] ^= 0x01
    try:
        base.assert_equal_output("mutation", "op_pd_moves_cmp_pipeline", 0, bytes(mutated), expected)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated output was not caught by comparator")
    return diverged, len(mutant_specs)


CMP_QUIRKS = (
    ("nan_ordered", "op_cmpsd_3"),
    ("nan_ordered", "op_cmpsd_7"),
    ("nan_ordered", "op_cmpsd_0"),
    ("zero_signed", "op_cmpsd_0"),
    ("zero_signed", "op_cmpsd_1"),
    ("eq_for_neq", "op_cmpsd_4"),
    ("lt_for_le", "op_cmpsd_2"),
    ("swap_ord", "op_cmpsd_3"),
    ("swap_ord", "op_cmpsd_7"),
    ("mask32", "op_cmpsd_0"),
    ("numeric", "op_cmpsd_5"),
    ("ignore_daz", "op_cmpsd_0"),
    ("ignore_daz", "op_cmpsd_1"),
    ("ignore_daz", "op_cmpsd_6"),
)
DAZ_BLIND = ("op_cmpsd_3", "op_cmpsd_7", "op_cmpsd_self")


def check_cmpsd_modes(native_fns, lifted_fns, harness_path):
    driver = base.bind_mxcsr_driver(harness_path)
    pairs = cmpsd_pairs()
    modes = tuple(ref.Mode(rc, daz, ftz) for rc in range(4) for daz in (False, True) for ftz in (False, True))
    seen = {}
    for mode in modes:
        for name in CMPSD_EXPORTS:
            for idx, (a, b) in enumerate(pairs):
                got_n = base.invoke_under(driver, native_fns[name], mode.mxcsr(), a, b)
                got_l = base.invoke_under(driver, lifted_fns[name], mode.mxcsr(), a, b)
                want = reference(name, a, b, mode)
                base.assert_equal_output("native", f"{name} [{mode.label()}]", idx, got_n, want)
                base.assert_equal_output("lifted", f"{name} [{mode.label()}]", idx, got_l, want)
                seen[(name, mode.rc, mode.daz, mode.ftz, idx)] = got_n

    for (name, rc, daz, ftz, idx), got in seen.items():
        if got != seen[(name, 0, daz, False, idx)]:
            raise AssertionError(f"{name} case {idx}: rounding mode or FTZ changed a cmpsd result")
    for name in CMPSD_EXPORTS:
        moved = sum(seen[(name, 0, True, False, i)] != seen[(name, 0, False, False, i)] for i in range(len(pairs)))
        if bool(moved) == (name in DAZ_BLIND):
            raise AssertionError(f"{name}: DAZ changed {moved} results, expected {'none' if name in DAZ_BLIND else 'some'}")
    for quirk, name in CMP_QUIRKS:
        mode = ref.Mode(ref.RN, True, False) if quirk == "ignore_daz" else ref.DEFAULT_MODE
        caught = sum(
            reference(name, a, b, mode, quirk=quirk) != seen[(name, mode.rc, mode.daz, mode.ftz, idx)]
            for idx, (a, b) in enumerate(pairs)
        )
        if not caught:
            raise AssertionError(f"reference quirk {quirk} on {name} [{mode.label()}] was never detected")
    return len(seen), len(modes), len(CMP_QUIRKS)


def compile_o2_and_verify(original_path, native_fns, cxx_path):
    source_text, _ = translate(original_path, list(EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "sse_fp_pd_moves_cmp_o2.cpp"
        dll_path = tmp / "sse_fp_pd_moves_cmp_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'sse_fp_pd_moves_cmp_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind_exports(dll_path, "lifted_")
        try:
            return verify_positive_suite(native_fns, o2_fns, "lifted_o2")
        finally:
            o2_fns.clear()
            unload_lib(o2_lib)


def check_alt_encodings_and_rejections(original_path):
    ld_xmm0 = Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_xmm1 = Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX))
    ld_xmm4 = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM4, MemoryOperand(R.RCX))
    ld_xmm5 = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM5, MemoryOperand(R.RDX, displ=3, displ_size=1))
    st_xmm0 = Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)
    st_xmm4 = Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM4)
    ret_ins = Instruction.create(Code.RETNQ)
    ret_bytes = base.encode_seq(ret_ins)

    alt_accept = [
        ("cmpsd_xmm4_xmm5", (ld_xmm4, ld_xmm5, Instruction.create_reg_reg_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM4, R.XMM5, 6), st_xmm4, ret_ins),
         "xmm4.q[0] = op_cmpsd_q(xmm4.q[0], xmm5.q[0], 0x6u);"),
        ("cmpsd_rcx_plus3", (ld_xmm4, Instruction.create_reg_mem_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM4, MemoryOperand(R.RCX, displ=3, displ_size=1), 4), st_xmm4, ret_ins),
         "xmm4.q[0] = op_cmpsd_q(xmm4.q[0], load64((rcx + 0x3ull)), 0x4u);"),
        ("cmpsd_r8_neg_disp", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, MemoryOperand(R.R8, displ=-8, displ_size=1), 1), st_xmm0, ret_ins),
         "xmm0.q[0] = op_cmpsd_q(xmm0.q[0], load64((r8 + 0xfffffffffffffff8ull)), 0x1u);"),
        ("movlpd_read_r8_disp", (ld_xmm0, Instruction.create_reg_mem(
            Code.MOVLPD_XMM_M64, R.XMM0, MemoryOperand(R.R8, displ=0x10, displ_size=1)), st_xmm0, ret_ins),
         "xmm0.q[0] = load64((r8 + 0x10ull));"),
        ("movhpd_read_rcx_plus3", (ld_xmm4, Instruction.create_reg_mem(
            Code.MOVHPD_XMM_M64, R.XMM4, MemoryOperand(R.RCX, displ=3, displ_size=1)), st_xmm4, ret_ins),
         "xmm4.q[1] = load64((rcx + 0x3ull));"),
        ("movhpd_store_disp", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVHPD_M64_XMM, MemoryOperand(R.R8, displ=0x18, displ_size=1), R.XMM0), ret_ins),
         "store64((r8 + 0x18ull), xmm0.q[1]);"),
        ("movlpd_store_disp_plus1", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVLPD_M64_XMM, MemoryOperand(R.R8, displ=0x19, displ_size=1), R.XMM0), ret_ins),
         "store64((r8 + 0x19ull), xmm0.q[0]);"),
        ("movntpd_disp32", (ld_xmm4, Instruction.create_mem_reg(
            Code.MOVNTPD_M128_XMM, MemoryOperand(R.R8, displ=0x20, displ_size=1), R.XMM4), ret_ins),
         "store128((r8 + 0x20ull), xmm4);"),
    ]
    for label, seq, expected in alt_accept:
        raw = base.encode_seq(*seq)
        decoded = list(Decoder(64, raw, ip=0x1000))
        assert len(decoded) == len(seq) and not any(ins.is_invalid for ins in decoded), label
        src, count = lift_code(label, 0x1000, raw)
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if expected not in src:
            raise AssertionError(f"alternate encoding {label} lowered without expected {expected!r}")

    # 1) Non-legacy VEX / EVEX rejections.
    nonlegacy = [
        ("vcmpsd_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VCMPSD_XMM_XMM_XMMM64_IMM8, R.XMM2, R.XMM0, R.XMM1, 0), ret_ins), "non-legacy encoding"),
        ("vmovlpd_load_vex", (ld_xmm0, Instruction.create_reg_reg_mem(
            Code.VEX_VMOVLPD_XMM_XMM_M64, R.XMM2, R.XMM0, MemoryOperand(R.RDX)), ret_ins), "non-legacy encoding"),
        ("vmovlpd_store_vex", (ld_xmm0, Instruction.create_mem_reg(
            Code.VEX_VMOVLPD_M64_XMM, MemoryOperand(R.R8), R.XMM0), ret_ins), "non-legacy encoding"),
        ("vmovhpd_load_vex", (ld_xmm0, Instruction.create_reg_reg_mem(
            Code.VEX_VMOVHPD_XMM_XMM_M64, R.XMM2, R.XMM0, MemoryOperand(R.RDX)), ret_ins), "non-legacy encoding"),
        ("vmovhpd_store_vex", (ld_xmm0, Instruction.create_mem_reg(
            Code.VEX_VMOVHPD_M64_XMM, MemoryOperand(R.R8), R.XMM0), ret_ins), "non-legacy encoding"),
        ("vmovntpd_vex", (ld_xmm0, Instruction.create_mem_reg(
            Code.VEX_VMOVNTPD_M128_XMM, MemoryOperand(R.R8), R.XMM0), ret_ins), "non-legacy encoding"),
        ("vmovntpd_ymm_vex", (Instruction.create_mem_reg(
            Code.VEX_VMOVNTPD_M256_YMM, MemoryOperand(R.R8), R.YMM0), ret_ins), "non-legacy encoding"),
        ("vcmpsd_evex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg_u32(
            Code.EVEX_VCMPSD_KR_K1_XMM_XMMM64_IMM8_SAE, R.K1, R.XMM0, R.XMM1, 0), ret_ins), "non-legacy encoding"),
        ("vmovlpd_evex", (ld_xmm0, Instruction.create_reg_reg_mem(
            Code.EVEX_VMOVLPD_XMM_XMM_M64, R.XMM2, R.XMM0, MemoryOperand(R.RDX)), ret_ins), "non-legacy encoding"),
        ("vmovhpd_evex", (ld_xmm0, Instruction.create_mem_reg(
            Code.EVEX_VMOVHPD_M64_XMM, MemoryOperand(R.R8), R.XMM0), ret_ins), "non-legacy encoding"),
        ("vmovntpd_evex", (ld_xmm0, Instruction.create_mem_reg(
            Code.EVEX_VMOVNTPD_M128_XMM, MemoryOperand(R.R8), R.XMM0), ret_ins), "non-legacy encoding"),
    ]

    # 2) Reserved predicates, unaligned MOVNTPD, non-r8 stores, XMM6..15, undefined reads, addressing/stream errors.
    seg_movlpd = Instruction.create_reg_mem(Code.MOVLPD_XMM_M64, R.XMM0, MemoryOperand(R.RDX))
    seg_movlpd.segment_prefix = R.GS
    other = [
        ("cmpsd_imm_8", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, R.XMM1, 8), ret_ins), "unsupported cmpsd predicate"),
        ("cmpsd_imm_1f", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, R.XMM1, 0x1F), ret_ins), "unsupported cmpsd predicate"),
        ("cmpsd_imm_ff", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, R.XMM1, 0xFF), ret_ins), "unsupported cmpsd predicate"),
        ("movhlps_reg_reg", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(
            Code.MOVHLPS_XMM_XMM, R.XMM0, R.XMM1), ret_ins), "unsupported instruction"),
        ("movlhps_reg_reg", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(
            Code.MOVLHPS_XMM_XMM, R.XMM0, R.XMM1), ret_ins), "unsupported instruction"),
        ("unaligned_movntpd_8", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVNTPD_M128_XMM, MemoryOperand(R.R8, displ=8, displ_size=1), R.XMM0), ret_ins),
         "unaligned 16-byte memory operand"),
        ("unaligned_movntpd_1", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVNTPD_M128_XMM, MemoryOperand(R.R8, displ=1, displ_size=1), R.XMM0), ret_ins),
         "unaligned 16-byte memory operand"),
        ("unaligned_movntpd_neg4", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVNTPD_M128_XMM, MemoryOperand(R.R8, displ=-4, displ_size=1), R.XMM0), ret_ins),
         "unaligned 16-byte memory operand"),
        ("non_r8_store_movntpd_rcx", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVNTPD_M128_XMM, MemoryOperand(R.RCX), R.XMM0), ret_ins), "non-r8 store memory operand"),
        ("non_r8_store_movntpd_rdx", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVNTPD_M128_XMM, MemoryOperand(R.RDX, displ=16, displ_size=1), R.XMM0), ret_ins),
         "non-r8 store memory operand"),
        ("non_r8_store_movlpd_rcx", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVLPD_M64_XMM, MemoryOperand(R.RCX), R.XMM0), ret_ins), "non-r8 store memory operand"),
        ("non_r8_store_movhpd_rdx", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVHPD_M64_XMM, MemoryOperand(R.RDX, displ=8, displ_size=1), R.XMM0), ret_ins),
         "non-r8 store memory operand"),
        ("nonvol_xmm6_cmpsd_dst", (ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM6, R.XMM1, 0), ret_ins), "nonvolatile vector register"),
        ("nonvol_xmm15_cmpsd_src", (ld_xmm0, Instruction.create_reg_reg_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, R.XMM15, 0), ret_ins), "nonvolatile vector register"),
        ("nonvol_xmm7_movlpd_load", (Instruction.create_reg_mem(
            Code.MOVLPD_XMM_M64, R.XMM7, MemoryOperand(R.RDX)), ret_ins), "nonvolatile vector register"),
        ("nonvol_xmm8_movhpd_store", (Instruction.create_mem_reg(
            Code.MOVHPD_M64_XMM, MemoryOperand(R.R8), R.XMM8), ret_ins), "nonvolatile vector register"),
        ("nonvol_xmm9_movntpd", (Instruction.create_mem_reg(
            Code.MOVNTPD_M128_XMM, MemoryOperand(R.R8), R.XMM9), ret_ins), "nonvolatile vector register"),
        ("undef_cmpsd_dst", (ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, R.XMM1, 0), ret_ins), "undefined vector register xmm0"),
        ("undef_cmpsd_src", (ld_xmm0, Instruction.create_reg_reg_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, R.XMM3, 0), ret_ins), "undefined vector register xmm3"),
        ("undef_movlpd_dst", (Instruction.create_reg_mem(
            Code.MOVLPD_XMM_M64, R.XMM0, MemoryOperand(R.RDX)), ret_ins), "undefined vector register xmm0"),
        ("undef_movhpd_dst", (Instruction.create_reg_mem(
            Code.MOVHPD_XMM_M64, R.XMM2, MemoryOperand(R.RDX)), ret_ins), "undefined vector register xmm2"),
        ("undef_movlpd_store_src", (Instruction.create_mem_reg(
            Code.MOVLPD_M64_XMM, MemoryOperand(R.R8), R.XMM4), ret_ins), "undefined vector register xmm4"),
        ("undef_movhpd_store_src", (Instruction.create_mem_reg(
            Code.MOVHPD_M64_XMM, MemoryOperand(R.R8), R.XMM5), ret_ins), "undefined vector register xmm5"),
        ("undef_movntpd_src", (Instruction.create_mem_reg(
            Code.MOVNTPD_M128_XMM, MemoryOperand(R.R8), R.XMM4), ret_ins), "undefined vector register xmm4"),
        ("non_arg_base_rax_cmpsd", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, MemoryOperand(R.RAX), 0), ret_ins), "unsupported memory base"),
        ("non_arg_base_r9_movlpd", (ld_xmm0, Instruction.create_reg_mem(
            Code.MOVLPD_XMM_M64, R.XMM0, MemoryOperand(R.R9)), ret_ins), "unsupported memory base"),
        ("nonvol_base_rbx_movhpd", (ld_xmm0, Instruction.create_reg_mem(
            Code.MOVHPD_XMM_M64, R.XMM0, MemoryOperand(R.RBX)), ret_ins), "nonvolatile GPR base"),
        ("rip_rel_cmpsd", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, MemoryOperand(R.RIP, displ=0x1010, displ_size=1), 0), ret_ins),
         "unsupported memory operand"),
        ("rsp_movlpd", (ld_xmm0, Instruction.create_reg_mem(
            Code.MOVLPD_XMM_M64, R.XMM0, MemoryOperand(R.RSP)), ret_ins), "unsupported memory operand"),
        ("addr32_movhpd", (ld_xmm0, Instruction.create_reg_mem(
            Code.MOVHPD_XMM_M64, R.XMM0, MemoryOperand(R.ECX)), ret_ins), "unsupported memory operand"),
        ("indexed_movntpd", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVNTPD_M128_XMM, MemoryOperand(R.R8, R.RDX, 4), R.XMM0), ret_ins), "unsupported memory operand"),
        ("segment_prefix_movlpd", (ld_xmm0, seg_movlpd, st_xmm0, ret_ins), "unsupported instruction prefix"),
        ("missing_ret", (ld_xmm0,), "missing terminal ret"),
        ("after_ret", (ret_ins, ld_xmm0), "unreachable instruction after ret"),
    ]

    # 3) Raw invalid encodings for reg-reg movlpd/movhpd/movntpd (mod=11 is #UD) & prefix/stream errors.
    ld0_bytes = base.encode_seq(ld_xmm0)
    raw_cases = [
        ("movlpd_load_reg_reg_raw", ld0_bytes + bytes.fromhex("660f12c1") + ret_bytes, "invalid instruction"),
        ("movlpd_store_reg_reg_raw", ld0_bytes + bytes.fromhex("660f13c1") + ret_bytes, "invalid instruction"),
        ("movhpd_load_reg_reg_raw", ld0_bytes + bytes.fromhex("660f16c1") + ret_bytes, "invalid instruction"),
        ("movhpd_store_reg_reg_raw", ld0_bytes + bytes.fromhex("660f17c1") + ret_bytes, "invalid instruction"),
        ("movntpd_reg_reg_raw", ld0_bytes + bytes.fromhex("660f2bc1") + ret_bytes, "invalid instruction"),
        ("lock_cmpxchg", bytes.fromhex("f00fb109") + ret_bytes, "unsupported instruction prefix"),
        ("rep_cmpsd_string", bytes.fromhex("f3a7") + ret_bytes, "unsupported instruction prefix"),
        ("cmpsd_string_unprefixed", bytes.fromhex("a7") + ret_bytes, "unsupported instruction"),
    ]

    unit = nonlegacy + other
    for label, seq, reason in unit:
        raw = base.encode_seq(*seq)
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator unexpectedly accepted rejected control")

    for label, raw, reason in raw_cases:
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator unexpectedly accepted raw rejected control")

    # 4) End-to-end PE-patched rejections + CLI non-traceback rejection.
    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_pd_moves_cmp_pipeline")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry

    by_label = {label: (base.encode_seq(*seq), reason) for label, seq, reason in unit}
    by_label.update({label: (raw, reason) for label, raw, reason in raw_cases})
    pe_patch_labels = (
        "vcmpsd_vex",
        "vmovntpd_evex",
        "cmpsd_imm_8",
        "unaligned_movntpd_8",
        "non_r8_store_movntpd_rcx",
        "non_r8_store_movlpd_rcx",
        "nonvol_xmm6_cmpsd_dst",
        "undef_movhpd_dst",
        "movlpd_load_reg_reg_raw",
        "segment_prefix_movlpd",
    )
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_sse_fp_aot.py"
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        patched_path = tmp / "patched.dll"
        out_cpp = tmp / "rejected.cpp"
        for label in pe_patch_labels:
            patch, reason = by_label[label]
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds max_len ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["op_pd_moves_cmp_pipeline"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"pe_{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"pe_{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched_path), "op_pd_moves_cmp_pipeline", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 1 or "Traceback" in proc.stderr or "unsupported instruction prefix" not in proc.stderr:
            raise AssertionError(f"CLI rejection unexpected: rc={proc.returncode}, stderr={proc.stderr!r}")
        if out_cpp.exists():
            raise AssertionError("CLI failure left behind output file")

    print(
        f"rejections passed: {len(alt_accept)} alt + {len(nonlegacy)} VEX/EVEX + {len(other)} IR + "
        f"{len(raw_cases)} raw + {len(pe_patch_labels)} PE-patched + CLI"
    )


def record_covered_forms(original_path):
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    formatter = Formatter(FormatterSyntax.INTEL)
    formatter.use_pseudo_ops = False
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

    original_path, lifted_path, altered_path, altered_lifted_path, harness_path, cxx_path = (
        Path(p) for p in sys.argv[1:7]
    )
    check_alt_encodings_and_rejections(original_path)

    orig_lib, orig_fns = bind_exports(original_path)
    lifted_lib, lifted_fns = bind_exports(lifted_path, "lifted_")
    alt_lib, alt_fns = bind_exports(altered_path)
    alt_lifted_lib, alt_lifted_fns = bind_exports(altered_lifted_path, "lifted_")
    try:
        lifted_cases = verify_positive_suite(orig_fns, lifted_fns, "lifted")
        diverged, mutants = check_altered_and_mutants(lifted_fns, alt_fns, alt_lifted_fns)
        mxcsr_runs, mode_count, quirk_count = check_cmpsd_modes(orig_fns, lifted_fns, harness_path)
        o2_cases = compile_o2_and_verify(original_path, orig_fns, cxx_path)
    finally:
        for fn_map, lib in (
            (orig_fns, orig_lib),
            (lifted_fns, lifted_lib),
            (alt_fns, alt_lib),
            (alt_lifted_fns, alt_lifted_lib),
        ):
            fn_map.clear()
            unload_lib(lib)

    record_covered_forms(original_path)
    print(
        f"sse_fp_pd_moves_cmp passed: {lifted_cases} lifted + {o2_cases} /O2 cases across {len(EXPORTS)} exports; "
        f"{mxcsr_runs} MXCSR CMPSD runs ({mode_count} states, {quirk_count} quirks); "
        f"ALTER_IMM diverged on {len(diverged)} exports + {mutants} mutants caught"
    )


if __name__ == "__main__":
    main()
