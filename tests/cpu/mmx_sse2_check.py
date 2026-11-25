"""Differential checks, mutation controls and rejection controls for MMX/SSE2 AOT."""

import ctypes
import math
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Encoder, Instruction, MemoryOperand, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_mmx_sse2_aot import lift_code, translate

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
GUARD_PRE = bytes(0xA0 + (i % 16) for i in range(32))
GUARD_POST = bytes(0xC0 + (i % 16) for i in range(32))
FILL = b"\x5A" * 32
INT32_INDEFINITE = 0x80000000

EXPORTS = (
    "op_cvtpd2pi_reg",
    "op_cvtpd2pi_mem",
    "op_cvtpd2pi_mem_disp",
    "op_cvttpd2pi_reg",
    "op_cvttpd2pi_mem",
    "op_movdq2q",
    "op_movq2dq",
    "op_emms_standalone",
    "op_movq_reg_mem",
    "op_movdqa_reg_ls",
    "op_movdqu_reg_ls",
    "op_mmx_sse2_bridge",
    "op_cvtpi2pd_reg",
    "op_cvtpi2pd_mem",
    "op_cvtpi2pd_mem_disp",
    "op_cvtpi2pd_discarded_read",
    "op_cvtpi2pd_bridge",
)

ALTERED_EXPORTS = (
    "op_cvtpd2pi_reg",
    "op_cvttpd2pi_reg",
    "op_cvtpi2pd_reg",
    "op_cvtpi2pd_mem",
)


# --- Independent Python ISA reference model ---

def d64(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def u64s(buf):
    return struct.unpack("<2Q", buf)


def pk_d(*vals):
    return struct.pack("<" + "I" * len(vals), *(v & MASK32 for v in vals))


def cvt_i32(v, trunc):
    """Round-to-nearest-even (default MXCSR) or truncate toward zero with 0x80000000 indefinite."""
    if math.isnan(v) or math.isinf(v):
        return INT32_INDEFINITE
    r = math.trunc(v) if trunc else round(v)
    return r & MASK32 if -0x80000000 <= r <= 0x7FFFFFFF else INT32_INDEFINITE


def ref_pd2pi(buf16, trunc):
    q0, q1 = u64s(buf16)
    return pk_d(cvt_i32(d64(q0), trunc), cvt_i32(d64(q1), trunc))


def ref_pi2pd(buf8):
    lo, hi = struct.unpack("<2i", buf8[:8])
    return struct.pack("<2d", float(lo), float(hi))


def reference(name, a, b, altered=False):
    """Expected 32-byte output region; unwritten bytes keep the FILL pattern."""
    out = bytearray(FILL)
    if name == "op_cvtpd2pi_reg":
        out[0:8] = ref_pd2pi(a[:16], trunc=altered)
    elif name == "op_cvtpd2pi_mem":
        out[0:8] = ref_pd2pi(a[:16], trunc=False)
    elif name == "op_cvtpd2pi_mem_disp":
        out[8:16] = ref_pd2pi(b[16:32], trunc=False)
    elif name == "op_cvttpd2pi_reg":
        out[0:8] = ref_pd2pi(a[:16], trunc=not altered)
    elif name == "op_cvttpd2pi_mem":
        out[0:8] = ref_pd2pi(b[:16], trunc=True)
    elif name == "op_movdq2q":
        out[0:8] = a[:8]
    elif name == "op_movq2dq":
        out[0:16] = b[:8] + b"\x00" * 8
    elif name == "op_emms_standalone":
        out[1:17] = a[1:17]
    elif name == "op_movq_reg_mem":
        out[0:8] = a[:8]
    elif name == "op_movdqa_reg_ls":
        out[0:16] = a[:16]
    elif name == "op_movdqu_reg_ls":
        out[3:19] = a[3:19]
    elif name == "op_mmx_sse2_bridge":
        pi_a = ref_pd2pi(a[:16], trunc=False)
        pi_b = ref_pd2pi(b[:16], trunc=True)
        out[0:16] = pi_a + b"\x00" * 8
        out[16:24] = pi_b
        out[24:32] = pi_a
    elif name == "op_cvtpi2pd_reg":
        out[0:16] = ref_pi2pd(b[:8] if altered else a[:8])
    elif name == "op_cvtpi2pd_mem":
        out[0:16] = ref_pi2pd(b[:8] if altered else a[:8])
    elif name == "op_cvtpi2pd_mem_disp":
        out[3:19] = ref_pi2pd(b[1:9])
    elif name == "op_cvtpi2pd_discarded_read":
        out[0:16] = a[:16]
        out[16:32] = a[:16]
    elif name == "op_cvtpi2pd_bridge":
        converted = ref_pi2pd(a[:8])
        out[0:16] = converted
        out[16:32] = ref_pi2pd(ref_pd2pi(converted, trunc=False))
    else:
        raise ValueError(f"unknown export {name}")
    return bytes(out)


DOUBLE_BITS = (
    0x3FF8000000000000,  # 1.5 (ties to even -> 2)
    0xBFF8000000000000,  # -1.5 (ties to even -> -2)
    0x4004000000000000,  # 2.5 (ties to even -> 2)
    0xC004000000000000,  # -2.5 (ties to even -> -2)
    0xC002000000000000,  # -2.25
    0x3FF4000000000000,  # 1.25
    0xBFFC000000000000,  # -1.75
    0x0000000000000000,  # +0.0
    0x8000000000000000,  # -0.0
    0x3FE0000000000000,  # 0.5 (ties to even -> 0)
    0xBFE0000000000000,  # -0.5 (ties to even -> 0)
    0x3FF0000000000000,  # 1.0
    0xBFF0000000000000,  # -1.0
    0x41DFFFFFFFC00000,  # 2147483647.0
    0x41DFFFFFFFDFFFFF,  # largest double < 2147483647.5
    0x41DFFFFFFFE00000,  # 2147483647.5 (rounds to 2^31 -> indefinite, truncates to 2147483647)
    0x41E0000000000000,  # 2147483648.0 (out of range -> indefinite)
    0xC1E0000000000000,  # -2147483648.0 (exact INT32_MIN)
    0xC1E0000000100000,  # -2147483648.5 (rounds to -2147483648, truncates to -2147483648)
    0xC1E0000000180000,  # -2147483648.75 (rounds to -2147483649 -> indefinite, truncates to -2147483648)
    0xC1E0000000200000,  # -2147483649.0 (out of range -> indefinite)
    0x7FF0000000000000,  # +inf
    0xFFF0000000000000,  # -inf
    0x7FF8000000000000,  # canonical qNaN
    0x7FF8000000000001,  # qNaN with payload
    0x7FF0000000000001,  # sNaN with payload
    0xFFF0000000000001,  # negative sNaN
    0xFFF8000000000000,  # real indefinite
    0x0000000000000001,  # min denormal
    0x000FFFFFFFFFFFFF,  # max denormal
    0x7FEFFFFFFFFFFFFF,  # +max finite
    0xFFEFFFFFFFFFFFFF,  # -max finite
)


def double_pairs():
    """(a, b) 32-byte pairs sweeping edge and boundary doubles across both lanes."""
    pairs = []
    n = len(DOUBLE_BITS)
    for i in range(0, n, 2):
        lo = DOUBLE_BITS[i]
        hi = DOUBLE_BITS[(i + 1) % n]
        lo2 = DOUBLE_BITS[(i + 3) % n]
        hi2 = DOUBLE_BITS[(i + 5) % n]
        pairs.append((struct.pack("<4Q", lo, hi, lo2, hi2), struct.pack("<4Q", hi2, lo2, hi, lo)))
    # Also test swapped lane ordering on key boundaries (e.g. 2147483647.5 in hi lane).
    pairs.append((
        struct.pack("<4Q", 0x3FF0000000000000, 0x41DFFFFFFFE00000, 0xC1E0000000100000, 0x8000000000000000),
        struct.pack("<4Q", 0xC1E0000000180000, 0x41E0000000000000, 0x7FF8000000000000, 0xC1E0000000200000),
    ))
    rng = random.Random(0x5EED3)
    for _ in range(8):
        pairs.append((rng.randbytes(32), rng.randbytes(32)))
    return pairs


def byte_pairs():
    """(a, b) 32-byte pairs with non-zero upper qwords for move/bridge checks."""
    pairs = [
        (bytes(range(0x20, 0x40)), bytes(range(0x80, 0xA0))),
        (b"\xFF" * 32, b"\x00" * 16 + b"\xAA" * 16),
        (b"\xAA" * 16 + b"\x55" * 16, b"\x11\x22\x33\x44\x55\x66\x77\x88" + b"\x99" * 24),
        (bytes((0x13 * i + 7) & 0xFF for i in range(32)), bytes((0x29 * i + 3) & 0xFF for i in range(32))),
    ]
    pairs += double_pairs()[:6]
    return pairs


INT32_BITS = (
    -2147483648,  # INT32_MIN
    -2147483647,  # INT32_MIN + 1
    -1,
    0,
    1,
    2147483646,   # INT32_MAX - 1
    2147483647,   # INT32_MAX
    -16777216,    # -(2^24)
    -16777217,    # -(2^24 + 1)
    16777217,     # 2^24 + 1
    0x12345678,
    -0x12345678,
)


def int32_pairs():
    """(a, b) 32-byte pairs sweeping int32 edges, lane orders and unaligned m64 offsets."""
    pairs = []
    n = len(INT32_BITS)
    for i in range(0, n, 2):
        lo_a = INT32_BITS[i]
        hi_a = INT32_BITS[(i + 1) % n]
        lo_b = INT32_BITS[(i + 2) % n]
        hi_b = INT32_BITS[(i + 3) % n]
        a = struct.pack("<2i", lo_a, hi_a) + struct.pack("<6i", hi_a, lo_a, lo_b, hi_b, 0, -1)
        b = struct.pack("<2i", lo_b, hi_b) + b"\xA5" + struct.pack("<5i", hi_a, lo_a, -2147483648, 2147483647, -1) + b"\x5A\x3C\x7E"
        pairs.append((a, b))
    explicit = [
        ((-2147483648, 2147483647), (2147483647, -2147483648), (-2147483648, -1)),
        ((-1, 0), (0, 1), (1, 2147483647)),
        ((1, -1), (-2147483648, 0), (0, -2147483648)),
    ]
    for a_pair, b_pair, unaligned_b in explicit:
        a = struct.pack("<2i", *a_pair) + bytes(range(0x10, 0x28))
        b = struct.pack("<2i", *b_pair) + b"\x77" * 16 + b"\x33" * 8
        b = b[:1] + struct.pack("<2i", *unaligned_b) + b[9:]
        pairs.append((a, b))
    rng = random.Random(0x5EED4)
    for _ in range(6):
        a = rng.randbytes(32)
        b = rng.randbytes(32)
        if a[:8] == b[:8]:
            b = bytes([b[0] ^ 0xFF]) + b[1:]
        pairs.append((a, b))
    return pairs


def inputs_for(name):
    if name.startswith(("op_cvtpd2pi", "op_cvttpd2pi", "op_mmx_sse2_bridge")):
        return double_pairs()
    if name.startswith("op_cvtpi2pd"):
        return int32_pairs()
    return byte_pairs()


# --- Guarded buffer helpers ---

def alloc_aligned_buffer(payload, align=16):
    total = len(GUARD_PRE) + len(payload) + len(GUARD_POST) + align
    buf = ctypes.create_string_buffer(total)
    base_addr = ctypes.addressof(buf)
    start_after_guard = base_addr + len(GUARD_PRE)
    pad = (align - start_after_guard % align) % align
    payload_off = len(GUARD_PRE) + pad
    pre_off = payload_off - len(GUARD_PRE)
    post_off = payload_off + len(payload)
    buf[pre_off:payload_off] = GUARD_PRE
    buf[payload_off:post_off] = payload
    buf[post_off:post_off + len(GUARD_POST)] = GUARD_POST
    return buf, base_addr + payload_off, pre_off, payload_off, post_off


def check_guards_and_read(buf, pre_off, payload_off, post_off, label):
    raw = buf.raw
    if raw[pre_off:payload_off] != GUARD_PRE or raw[post_off:post_off + len(GUARD_POST)] != GUARD_POST:
        raise AssertionError(f"guard bytes corrupted around {label}")
    return raw[payload_off:post_off]


def invoke(fn, a, b):
    buf_a, ptr_a, a_pre, a_off, a_post = alloc_aligned_buffer(a)
    buf_b, ptr_b, b_pre, b_off, b_post = alloc_aligned_buffer(b)
    buf_out, ptr_out, o_pre, o_off, o_post = alloc_aligned_buffer(FILL)
    fn(ptr_a, ptr_b, ptr_out)
    check_guards_and_read(buf_a, a_pre, a_off, a_post, "input_a")
    check_guards_and_read(buf_b, b_pre, b_off, b_post, "input_b")
    return check_guards_and_read(buf_out, o_pre, o_off, o_post, "output")


def assert_equal_output(label, name, idx, got, want):
    if got != want:
        raise AssertionError(f"{label} {name} case {idx}: got {got.hex()} != expected {want.hex()}")


def bind_exports(dll_path, prefix="", names=EXPORTS):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in names:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = None
        funcs[name] = fn
    return funcs


def check_discarded_read(native_path, lifted_path):
    """Verify volatile load ordering on a discarded cvtpi2pd m64 source read."""
    payload = bytes(range(32))
    name = "op_cvtpi2pd_discarded_read"
    for label, dll, prefix in (("native", native_path, ""), ("lifted", lifted_path, "lifted_")):
        fn = bind_exports(dll, prefix, (name,))[name]
        source, ptr, _, _, _ = alloc_aligned_buffer(payload)
        for readable in (True, False):
            result, result_ptr, pre, off, post = alloc_aligned_buffer(FILL)
            try:
                fn(ptr, ptr if readable else 0, result_ptr)
            except OSError as error:
                assert not readable and "access violation reading" in str(error).lower(), (label, error)
            else:
                assert readable, f"{label} discarded a faulting CVTPI2PD read"
            expected = payload[:16] + (payload[:16] if readable else FILL[16:])
            assert_equal_output(
                label, name, readable, check_guards_and_read(result, pre, off, post, label), expected
            )
        _ = source


def check_differential(native_path, lifted_path, altered_path):
    native = bind_exports(native_path)
    lifted = bind_exports(lifted_path, "lifted_")
    altered = bind_exports(altered_path, "lifted_")

    total = 0
    for name in EXPORTS:
        for idx, (a, b) in enumerate(inputs_for(name)):
            want = reference(name, a, b)
            got_n = invoke(native[name], a, b)
            got_l = invoke(lifted[name], a, b)
            assert_equal_output("native", name, idx, got_n, want)
            assert_equal_output("lifted", name, idx, got_l, want)
            total += 1

    changed = {}
    for name in ALTERED_EXPORTS:
        diffs = 0
        for idx, (a, b) in enumerate(inputs_for(name)):
            want_alt = reference(name, a, b, altered=True)
            want_orig = reference(name, a, b)
            got_alt = invoke(altered[name], a, b)
            assert_equal_output("altered", name, idx, got_alt, want_alt)
            if got_alt != want_orig:
                diffs += 1
        if not diffs:
            raise AssertionError(f"altered control {name} did not diverge from original")
        changed[name] = diffs

    # Output-buffer mutation negative control.
    sample_a, sample_b = double_pairs()[0]
    expected = reference("op_mmx_sse2_bridge", sample_a, sample_b)
    mutated = bytearray(expected)
    mutated[5] ^= 0x01
    try:
        assert_equal_output("mutation", "op_mmx_sse2_bridge", 0, bytes(mutated), expected)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated output was not caught by comparator")

    # Mutated signedness and lane-order negative controls for cvtpi2pd.
    signed_a = struct.pack("<2i", -2147483648, 2147483647) + b"\x00" * 24
    signed_expected = reference("op_cvtpi2pd_reg", signed_a, signed_a)
    unsigned_mutated = struct.pack("<2d", float(0x80000000), float(2147483647)) + signed_expected[16:]
    swapped_lane_mutated = signed_expected[8:16] + signed_expected[0:8] + signed_expected[16:]
    for mut_label, mut_bytes in (("signedness", unsigned_mutated), ("lane_order", swapped_lane_mutated)):
        try:
            assert_equal_output(f"mutation_{mut_label}", "op_cvtpi2pd_reg", 0, mut_bytes, signed_expected)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"mutated {mut_label} output was not caught by comparator")

    print(f"{total} native/lifted/reference MMX/SSE2 cases across {len(EXPORTS)} exports matched "
          f"bit-for-bit; altered controls diverged on {changed}; output/signedness/lane-order mutation controls caught")


# --- Alternate-encoding acceptance & translator rejection controls ---

def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_alt_encodings_and_rejections(original_path):
    ld_xmm0 = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_xmm1 = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX))
    ld_mm0 = Instruction.create_reg_mem(Code.MOVQ_MM_MMM64, R.MM0, MemoryOperand(R.RCX))
    ld_mm7 = Instruction.create_reg_mem(Code.MOVQ_MM_MMM64, R.MM7, MemoryOperand(R.RCX))
    st_xmm0 = Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)
    st_xmm2 = Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM2)
    st_xmm3 = Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM3)
    st_xmm5 = Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM5)
    st_mm0 = Instruction.create_mem_reg(Code.MOVQ_MMM64_MM, MemoryOperand(R.R8), R.MM0)
    emms_ins = Instruction.create(Code.EMMS)
    ret_ins = Instruction.create(Code.RETNQ)

    alt_accept = [
        ("cvtpi2pd_xmm5_mm7", (
            ld_mm7,
            Instruction.create_reg_reg(Code.CVTPI2PD_XMM_MMM64, R.XMM5, R.MM7),
            st_xmm5, emms_ins, ret_ins)),
        ("cvtpi2pd_mem_r8_neg_disp", (
            Instruction.create_reg_mem(Code.CVTPI2PD_XMM_MMM64, R.XMM2, MemoryOperand(R.R8, displ=-3, displ_size=1)),
            st_xmm2, ret_ins)),
        ("cvtpi2pd_mem_rcx_unaligned", (
            Instruction.create_reg_mem(Code.CVTPI2PD_XMM_MMM64, R.XMM0, MemoryOperand(R.RCX, displ=5, displ_size=1)),
            st_xmm0, ret_ins)),
        ("cvtpi2pd_undef_xmm_dst", (
            ld_mm0,
            Instruction.create_reg_reg(Code.CVTPI2PD_XMM_MMM64, R.XMM3, R.MM0),
            st_xmm3, emms_ins, ret_ins)),
        ("movq_r10_r11", (
            ld_mm0,
            Instruction.create_reg_reg(Code.MOVQ_RM64_MM, R.R10, R.MM0),
            Instruction.create_reg_reg(Code.MOVQ_MM_RM64, R.MM2, R.R10),
            Instruction.create_reg_reg(Code.MOVQ_RM64_MM, R.R11, R.MM2),
            Instruction.create_reg_reg(Code.MOVQ_MM_RM64, R.MM3, R.R11),
            Instruction.create_mem_reg(Code.MOVQ_MMM64_MM, MemoryOperand(R.R8), R.MM3),
            emms_ins, ret_ins)),
        ("movq_0f7f_reg", (
            ld_mm0,
            Instruction.create_reg_reg(Code.MOVQ_MMM64_MM, R.MM4, R.MM0),
            Instruction.create_mem_reg(Code.MOVQ_MMM64_MM, MemoryOperand(R.R8), R.MM4),
            emms_ins, ret_ins)),
        ("movdqa_0f7f_reg", (
            ld_xmm0,
            Instruction.create_reg_reg(Code.MOVDQA_XMMM128_XMM, R.XMM2, R.XMM0),
            Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM2),
            ret_ins)),
        ("movdqu_0f7f_reg", (
            ld_xmm0,
            Instruction.create_reg_reg(Code.MOVDQU_XMMM128_XMM, R.XMM2, R.XMM0),
            Instruction.create_mem_reg(Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.R8), R.XMM2),
            ret_ins)),
        ("load_from_r8", (
            Instruction.create_reg_mem(Code.CVTPD2PI_MM_XMMM128, R.MM0, MemoryOperand(R.R8, displ=16, displ_size=1)),
            st_mm0, emms_ins, ret_ins)),
        ("cvttpd2pi_disp", (
            Instruction.create_reg_mem(Code.CVTTPD2PI_MM_XMMM128, R.MM5, MemoryOperand(R.RDX, displ=16, displ_size=1)),
            Instruction.create_mem_reg(Code.MOVQ_MMM64_MM, MemoryOperand(R.R8), R.MM5),
            emms_ins, ret_ins)),
        ("emms_then_xmm_only", (
            ld_mm0, st_mm0, emms_ins, ld_xmm0, st_xmm0, ret_ins)),
        ("xmm_only_no_emms", (
            ld_xmm0, st_xmm0, ret_ins)),
    ]
    alt_expected = {
        "cvtpi2pd_xmm5_mm7": "xmm5 = op_cvtpi2pd(mm[7]);",
        "cvtpi2pd_mem_r8_neg_disp": "xmm2 = op_cvtpi2pd(load64((r8 + 0xfffffffffffffffdull)));",
        "cvtpi2pd_mem_rcx_unaligned": "xmm0 = op_cvtpi2pd(load64((rcx + 0x5ull)));",
        "cvtpi2pd_undef_xmm_dst": "xmm3 = op_cvtpi2pd(mm[0]);",
        "movq_r10_r11": "mm[3] = r11;",
        "movq_0f7f_reg": "mm[4] = mm[0];",
        "movdqa_0f7f_reg": "xmm2 = xmm0;",
        "movdqu_0f7f_reg": "xmm2 = xmm0;",
        "load_from_r8": "mm[0] = op_cvtpd2pi(load128((r8 + 0x10ull)));",
        "cvttpd2pi_disp": "mm[5] = op_cvttpd2pi(load128((rdx + 0x10ull)));",
        "emms_then_xmm_only": "store128(r8, xmm0);",
        "xmm_only_no_emms": "store128(r8, xmm0);",
    }
    for label, seq in alt_accept:
        src, count = lift_code(label, 0x1000, encode_seq(*seq))
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if alt_expected[label] not in src:
            raise AssertionError(
                f"alternate encoding {label} lowered without {alt_expected[label]!r}")

    seg_load = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    seg_load.segment_prefix = R.FS

    reject_cases = [
        # Missing emms when mmx_active is true
        ("missing_emms_cvtpd2pi", (
            ld_xmm0,
            Instruction.create_reg_reg(Code.CVTPD2PI_MM_XMMM128, R.MM0, R.XMM0),
            st_mm0, ret_ins), "missing emms before ret"),
        ("missing_emms_cvttpd2pi", (
            Instruction.create_reg_mem(Code.CVTTPD2PI_MM_XMMM128, R.MM0, MemoryOperand(R.RCX)),
            st_mm0, ret_ins), "missing emms before ret"),
        ("missing_emms_movdq2q", (
            ld_xmm0,
            Instruction.create_reg_reg(Code.MOVDQ2Q_MM_XMM, R.MM0, R.XMM0),
            st_mm0, ret_ins), "missing emms before ret"),
        ("missing_emms_movq", (ld_mm0, st_mm0, ret_ins), "missing emms before ret"),
        ("missing_emms_reactivated", (
            ld_mm0, st_mm0, emms_ins, ld_mm0, ret_ins), "missing emms before ret"),
        ("missing_emms_movq2dq", (
            ld_mm0,
            Instruction.create_reg_reg(Code.MOVQ2DQ_XMM_MM, R.XMM0, R.MM0),
            st_xmm0, ret_ins), "missing emms before ret"),
        ("missing_emms_cvtpi2pd_reg", (
            ld_mm0,
            Instruction.create_reg_reg(Code.CVTPI2PD_XMM_MMM64, R.XMM0, R.MM0),
            st_xmm0, ret_ins), "missing emms before ret"),
        ("missing_emms_prior_mm_then_cvtpi2pd_mem", (
            ld_mm0,
            Instruction.create_reg_mem(Code.CVTPI2PD_XMM_MMM64, R.XMM0, MemoryOperand(R.RCX)),
            st_xmm0, ret_ins), "missing emms before ret"),
        # Reading MM after emms without redefining
        ("read_mm_after_emms", (
            ld_mm0, emms_ins,
            Instruction.create_reg_reg(Code.MOVQ2DQ_XMM_MM, R.XMM0, R.MM0),
            emms_ins, ret_ins), "undefined MMX register mm0"),
        # VEX / EVEX rejections
        ("vex_vcvtpd2dq", (
            ld_xmm0,
            Instruction.create_reg_reg(Code.VEX_VCVTPD2DQ_XMM_XMMM128, R.XMM1, R.XMM0),
            ret_ins), "non-legacy encoding"),
        ("vex_vcvtdq2pd", (
            ld_xmm0,
            Instruction.create_reg_reg(Code.VEX_VCVTDQ2PD_XMM_XMMM64, R.XMM1, R.XMM0),
            ret_ins), "non-legacy encoding"),
        ("vex_vmovdqa", (
            Instruction.create_reg_mem(Code.VEX_VMOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)),
            ret_ins), "non-legacy encoding"),
        ("vex_vmovdqu", (
            Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)),
            ret_ins), "non-legacy encoding"),
        ("vex_vmovq", (
            Instruction.create_reg_reg(Code.VEX_VMOVQ_XMM_RM64, R.XMM0, R.RAX),
            ret_ins), "non-legacy encoding"),
        ("evex_vmovdqa64", (
            Instruction.create_reg_mem(Code.EVEX_VMOVDQA64_XMM_K1Z_XMMM128, R.XMM0, MemoryOperand(R.RCX)),
            ret_ins), "non-legacy encoding"),
        # Nonvolatile XMM6..XMM15
        ("nonvolatile_xmm6_dst", (
            Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM6, MemoryOperand(R.RCX)),
            ret_ins), "nonvolatile vector register"),
        ("nonvolatile_xmm6_cvtpi2pd_reg", (
            ld_mm0,
            Instruction.create_reg_reg(Code.CVTPI2PD_XMM_MMM64, R.XMM6, R.MM0),
            emms_ins, ret_ins), "nonvolatile vector register"),
        ("nonvolatile_xmm6_cvtpi2pd_mem", (
            Instruction.create_reg_mem(Code.CVTPI2PD_XMM_MMM64, R.XMM6, MemoryOperand(R.RCX)),
            ret_ins), "nonvolatile vector register"),
        ("nonvolatile_xmm15_src", (
            Instruction.create_reg_reg(Code.CVTPD2PI_MM_XMMM128, R.MM0, R.XMM15),
            emms_ins, ret_ins), "nonvolatile vector register"),
        ("nonvolatile_xmm7_movq2dq", (
            ld_mm0,
            Instruction.create_reg_reg(Code.MOVQ2DQ_XMM_MM, R.XMM7, R.MM0),
            emms_ins, ret_ins), "nonvolatile vector register"),
        # Nonvolatile and argument GPRs
        ("nonvolatile_gpr_rbx", (
            ld_mm0,
            Instruction.create_reg_reg(Code.MOVQ_RM64_MM, R.RBX, R.MM0),
            emms_ins, ret_ins), "nonvolatile GPR"),
        ("nonvolatile_gpr_r12", (
            ld_mm0,
            Instruction.create_reg_reg(Code.MOVQ_RM64_MM, R.R12, R.MM0),
            emms_ins, ret_ins), "nonvolatile GPR"),
        ("arg_gpr_rcx", (
            ld_mm0,
            Instruction.create_reg_reg(Code.MOVQ_RM64_MM, R.RCX, R.MM0),
            emms_ins, ret_ins), "unsupported argument GPR operand"),
        # Undefined registers
        ("undefined_xmm2", (
            Instruction.create_reg_reg(Code.CVTPD2PI_MM_XMMM128, R.MM0, R.XMM2),
            emms_ins, ret_ins), "undefined vector register xmm2"),
        ("undefined_mm3", (
            Instruction.create_reg_reg(Code.MOVQ2DQ_XMM_MM, R.XMM0, R.MM3),
            emms_ins, ret_ins), "undefined MMX register mm3"),
        ("undefined_mm_cvtpi2pd", (
            Instruction.create_reg_reg(Code.CVTPI2PD_XMM_MMM64, R.XMM0, R.MM2),
            emms_ins, ret_ins), "undefined MMX register mm2"),
        ("undefined_gpr_rax", (
            Instruction.create_reg_reg(Code.MOVQ_MM_RM64, R.MM0, R.RAX),
            emms_ins, ret_ins), "undefined GPR rax"),
        # Memory base & addressing violations
        ("non_arg_base_rax", (
            Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RAX)),
            ret_ins), "unsupported memory operand"),
        ("non_arg_base_cvtpi2pd_rax", (
            Instruction.create_reg_mem(Code.CVTPI2PD_XMM_MMM64, R.XMM0, MemoryOperand(R.RAX)),
            ret_ins), "unsupported memory operand"),
        ("non_arg_base_cvtpi2pd_rbx", (
            Instruction.create_reg_mem(Code.CVTPI2PD_XMM_MMM64, R.XMM0, MemoryOperand(R.RBX)),
            ret_ins), "unsupported memory operand"),
        ("rip_rel_mem", (
            Instruction.create_reg_mem(Code.CVTPD2PI_MM_XMMM128, R.MM0,
                                       MemoryOperand(R.RIP, displ=0x1010, displ_size=1)),
            emms_ins, ret_ins), "unsupported memory operand"),
        ("rsp_mem", (
            Instruction.create_reg_mem(Code.MOVQ_MM_MMM64, R.MM0, MemoryOperand(R.RSP)),
            emms_ins, ret_ins), "unsupported memory operand"),
        ("addr32_mem", (
            Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.ECX)),
            ret_ins), "unsupported memory operand"),
        ("indexed_mem", (
            Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, R.RDX, 4)),
            ret_ins), "unsupported memory operand"),
        # Stores outside r8
        ("store_movdqa_rcx", (
            ld_xmm0,
            Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.RCX), R.XMM0),
            ret_ins), "store outside the output buffer r8"),
        ("store_movdqu_rdx", (
            ld_xmm0,
            Instruction.create_mem_reg(Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.RDX), R.XMM0),
            ret_ins), "store outside the output buffer r8"),
        ("store_movq_rcx", (
            ld_mm0,
            Instruction.create_mem_reg(Code.MOVQ_MMM64_MM, MemoryOperand(R.RCX), R.MM0),
            emms_ins, ret_ins), "store outside the output buffer r8"),
        ("store_movq_rdx", (
            ld_mm0,
            Instruction.create_mem_reg(Code.MOVQ_MMM64_MM, MemoryOperand(R.RDX), R.MM0),
            emms_ins, ret_ins), "store outside the output buffer r8"),
        # Unaligned 16-byte accesses
        ("unaligned_cvtpd2pi_m128", (
            Instruction.create_reg_mem(Code.CVTPD2PI_MM_XMMM128, R.MM0, MemoryOperand(R.RCX, displ=8)),
            emms_ins, ret_ins), "unaligned 16-byte memory operand"),
        ("unaligned_cvttpd2pi_m128", (
            Instruction.create_reg_mem(Code.CVTTPD2PI_MM_XMMM128, R.MM0, MemoryOperand(R.RDX, displ=1)),
            emms_ins, ret_ins), "unaligned 16-byte memory operand"),
        ("unaligned_movdqa_load", (
            Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=8)),
            ret_ins), "unaligned 16-byte memory operand"),
        ("unaligned_movdqa_store", (
            ld_xmm0,
            Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8, displ=8), R.XMM0),
            ret_ins), "unaligned 16-byte memory operand"),
        # Unsupported instructions
        ("unsupported_femms", (Instruction.create(Code.FEMMS), ret_ins), "unsupported instruction"),
        ("unsupported_cvtps2pi", (
            ld_xmm0, Instruction.create_reg_reg(Code.CVTPS2PI_MM_XMMM64, R.MM0, R.XMM0),
            emms_ins, ret_ins), "unsupported instruction"),
        ("unsupported_movq_xmm", (
            ld_xmm1, Instruction.create_reg_reg(Code.MOVQ_XMM_XMMM64, R.XMM0, R.XMM1),
            ret_ins), "unsupported instruction"),
        ("unsupported_mov_gpr", (
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RAX, MemoryOperand(R.RCX)),
            ret_ins), "unsupported instruction"),
        ("unsupported_addpd", (
            ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.ADDPD_XMM_XMMM128, R.XMM0, R.XMM1),
            ret_ins), "unsupported instruction"),
        # Prefixes, missing ret, unreachable after ret
        ("segment_prefix", (seg_load, ret_ins), "unsupported instruction prefix"),
        ("missing_ret", (ld_xmm0,), "missing terminal ret"),
        ("after_ret", (ret_ins, ld_xmm0), "unreachable instruction after ret"),
    ]

    raw_cases = [
        ("lock_cmpxchg", bytes.fromhex("f00fb109") + encode_seq(ret_ins), "unsupported instruction prefix"),
        ("rep_movsb", bytes.fromhex("f3a4") + encode_seq(ret_ins), "unsupported instruction prefix"),
        ("maskmovdqu", bytes.fromhex("660ff7c8") + encode_seq(ret_ins), "unsupported instruction"),
    ]

    for label, seq, expected_reason in reject_cases:
        raw = encode_seq(*seq)
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator unexpectedly accepted rejected control")

    for label, raw, expected_reason in raw_cases:
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
        entry = export_rva(pe, "op_mmx_sse2_bridge")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12)
                       if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        orig_end = struct.unpack_from("<I", records, rec_idx + 4)[0]
        max_len = orig_end - entry

    seq_map = {label: seq for label, seq, _ in reject_cases}
    pe_patches = [
        ("pe_missing_emms", encode_seq(*seq_map["missing_emms_cvtpd2pi"]), "missing emms before ret"),
        ("pe_missing_emms_cvtpi2pd", encode_seq(*seq_map["missing_emms_cvtpi2pd_reg"]), "missing emms before ret"),
        ("pe_undef_mm_cvtpi2pd", encode_seq(*seq_map["undefined_mm_cvtpi2pd"]), "undefined MMX register mm2"),
        ("pe_nonvolatile_xmm6_cvtpi2pd", encode_seq(*seq_map["nonvolatile_xmm6_cvtpi2pd_mem"]), "nonvolatile vector register"),
        ("pe_non_arg_base_cvtpi2pd", encode_seq(*seq_map["non_arg_base_cvtpi2pd_rax"]), "unsupported memory operand"),
        ("pe_vex_vmovdqa", encode_seq(*seq_map["vex_vmovdqa"]), "non-legacy encoding"),
        ("pe_nonvolatile_xmm6", encode_seq(*seq_map["nonvolatile_xmm6_dst"]), "nonvolatile vector register"),
        ("pe_unaligned_cvtpd2pi", encode_seq(*seq_map["unaligned_cvtpd2pi_m128"]), "unaligned 16-byte memory operand"),
        ("pe_store_movq_rcx", encode_seq(*seq_map["store_movq_rcx"]), "store outside the output buffer r8"),
        ("pe_non_arg_base", encode_seq(*seq_map["non_arg_base_rax"]), "unsupported memory operand"),
        ("pe_femms", encode_seq(*seq_map["unsupported_femms"]), "unsupported instruction"),
        ("pe_missing_ret", encode_seq(*seq_map["missing_ret"]), "missing terminal ret"),
        ("pe_after_ret", encode_seq(*seq_map["after_ret"]), "unreachable instruction after ret"),
    ]
    with TemporaryDirectory() as folder:
        patched_path = Path(folder) / "patched.dll"
        for label, patch, expected_reason in pe_patches:
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds op_mmx_sse2_bridge ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                translate(patched_path, ["op_mmx_sse2_bridge"])
            except ValueError as exc:
                if expected_reason not in str(exc):
                    raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: PE translate accepted rejected patch")

    print(f"{len(alt_accept)} alternate-encoding lift checks, {len(reject_cases) + len(raw_cases)} "
          f"unit rejection controls and {len(pe_patches)} PE-patched rejection controls passed")


if __name__ == "__main__":
    check_discarded_read(*sys.argv[1:3])
    check_differential(*sys.argv[1:4])
    check_alt_encodings_and_rejections(sys.argv[1])
