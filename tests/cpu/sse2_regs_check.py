"""Differential checks, mutation controls and rejection controls for legacy SSE2 AOT."""

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
from cpu_sse2_aot import lift_code, translate
from cpu_leaf_aot import export_rva

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
GUARD_PRE = bytes(0xA0 + (i % 16) for i in range(32))
GUARD_POST = bytes(0xC0 + (i % 16) for i in range(32))
FILL = b"\x5A" * 32
INT32_INDEFINITE = 0x80000000
INT64_INDEFINITE = 0x8000000000000000
REAL_INDEFINITE = 0xFFF8000000000000

EXPORTS = (
    # cmppd predicates 0..7 (all eight covered) + aligned m128 source (9)
    "op_cmppd_0", "op_cmppd_1", "op_cmppd_2", "op_cmppd_3",
    "op_cmppd_4", "op_cmppd_5", "op_cmppd_6", "op_cmppd_7", "op_cmppd_mem",
    # cmpsd predicates 0..7, m64 source, unaligned +1 m64 source (10)
    "op_cmpsd_0", "op_cmpsd_1", "op_cmpsd_2", "op_cmpsd_3",
    "op_cmpsd_4", "op_cmpsd_5", "op_cmpsd_6", "op_cmpsd_7",
    "op_cmpsd_mem", "op_cmpsd_mem_plus1",
    # Scalar double ops (6)
    "op_maxsd_reg", "op_maxsd_mem", "op_minsd_reg", "op_minsd_mem",
    "op_sqrtsd_reg", "op_sqrtsd_mem",
    # cvtsd2si r32/r64, reg/mem (4)
    "op_cvtsd2si_r32_reg", "op_cvtsd2si_r64_reg",
    "op_cvtsd2si_r32_mem", "op_cvtsd2si_r64_mem",
    # Packed converts (17)
    "op_cvtdq2pd_reg", "op_cvtdq2pd_mem", "op_cvtdq2ps_reg", "op_cvtdq2ps_mem",
    "op_cvtpd2dq_reg", "op_cvtpd2dq_mem", "op_cvtpd2ps_reg", "op_cvtpd2ps_mem",
    "op_cvtpi2pd_mem",
    "op_cvtps2dq_reg", "op_cvtps2dq_mem", "op_cvtps2pd_reg", "op_cvtps2pd_mem",
    "op_cvttpd2dq_reg", "op_cvttpd2dq_mem",
    "op_cvttps2dq_reg", "op_cvttps2dq_mem",
    # Aligned moves and half-qword merges (12), unaligned +1 loads/stores (4)
    "op_movapd_ls", "op_movapd_reg", "op_movdqa_ls", "op_movdqa_reg",
    "op_movhpd_load", "op_movhpd_store", "op_movlpd_load", "op_movlpd_store",
    "op_movhpd_load_plus1", "op_movlpd_load_plus1",
    "op_movhpd_store_plus1", "op_movlpd_store_plus1",
    # Unaligned +1 m64 sources (2), non-temporal vector stores (2)
    "op_maxsd_mem_plus1", "op_cvtsd2si_mem_plus1",
    "op_movntdq", "op_movntpd",
    # Non-temporal GPR stores (2)
    "op_movnti32", "op_movnti64",
    # Fences (2)
    "op_lfence", "op_mfence",
    # Chained pipeline (1)
    "op_pipeline",
)

# Exports whose compare predicate flips in the ALTER_IMM build.
ALTERED_PREDICATES = {
    "op_cmppd_1": (1, 5), "op_cmppd_3": (3, 7), "op_cmppd_6": (6, 2),
    "op_cmpsd_1": (1, 5), "op_cmpsd_3": (3, 7), "op_cmpsd_6": (6, 2),
}
# Fixed predicates of the cmpsd memory-source exports.
CMPSD_MEM_PREDICATES = {"op_cmpsd_mem": 6, "op_cmpsd_mem_plus1": 2}


# --- Independent Python ISA reference model ---

def u64s(buf):
    return struct.unpack("<2Q", buf)


def u64(buf):
    return struct.unpack("<Q", buf)[0]


def u32s(buf):
    return struct.unpack("<4I", buf)


def si32(v):
    return struct.unpack("<i", struct.pack("<I", v & MASK32))[0]


def d64(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def q64(v):
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def pk_q(*vals):
    return struct.pack("<" + "Q" * len(vals), *(v & MASK64 for v in vals))


def pk_d(*vals):
    return struct.pack("<" + "I" * len(vals), *(v & MASK32 for v in vals))


def cmp_pred(x, y, imm):
    """CMPPD predicates 0..7 (EQ_OQ LT_OS LE_OS UNORD_Q NEQ_UQ NLT_US NLE_US ORD_Q)."""
    unord = (x != x) or (y != y)
    if imm == 0:
        return not unord and x == y
    if imm == 1:
        return not unord and x < y
    if imm == 2:
        return not unord and x <= y
    if imm == 3:
        return unord
    if imm == 4:
        return unord or x != y
    if imm == 5:
        return unord or not (x < y)
    if imm == 6:
        return unord or not (x <= y)
    return not unord


def ref_maxsd(a_bits, b_bits):
    """MAXSD/MINSD return src2 on ties, equal values and unordered inputs."""
    return a_bits if d64(a_bits) > d64(b_bits) else b_bits


def ref_minsd(a_bits, b_bits):
    return a_bits if d64(a_bits) < d64(b_bits) else b_bits


def ref_sqrtsd(b_bits):
    v = d64(b_bits)
    if v != v:
        return b_bits | 0x0008000000000000
    if v < 0.0:
        return REAL_INDEFINITE
    return q64(math.sqrt(v))


def cvt_i32(v, trunc):
    """Round (default MXCSR) or truncate a double to i32 with the indefinite fallback."""
    if math.isnan(v) or math.isinf(v):
        return INT32_INDEFINITE
    r = math.trunc(v) if trunc else round(v)
    return r & MASK32 if -0x80000000 <= r <= 0x7FFFFFFF else INT32_INDEFINITE


def ref_f64_to_i32(bits, trunc=False):
    return cvt_i32(d64(bits), trunc)


def ref_f32_to_i32(bits, trunc=False):
    return cvt_i32(f32(bits), trunc)


def ref_f64_to_i64(bits):
    v = d64(bits)
    if math.isnan(v) or math.isinf(v):
        return INT64_INDEFINITE
    r = round(v)
    return r & MASK64 if -(1 << 63) <= r <= (1 << 63) - 1 else INT64_INDEFINITE


def ref_f64_to_f32(bits):
    """Double to float bits: NaN narrows to a quiet float NaN keeping sign and top payload."""
    if (bits >> 52) & 0x7FF == 0x7FF and bits & 0xFFFFFFFFFFFFF:
        return (((bits >> 63) & 1) << 31) | 0x7F800000 | (((bits & 0xFFFFFFFFFFFFF) >> 29) | 0x400000)
    v = d64(bits)
    try:
        return struct.unpack("<I", struct.pack("<f", v))[0]
    except OverflowError:
        return (((bits >> 63) & 1) << 31) | 0x7F800000


def ref_f32_to_f64(bits):
    """Float to double bits: NaN widens to a quiet double NaN keeping sign and top payload."""
    if (bits >> 23) & 0xFF == 0xFF and bits & 0x7FFFFF:
        return (((bits >> 31) & 1) << 63) | 0x7FF0000000000000 | (
            ((bits & 0x7FFFFF) << 29) | 0x0008000000000000)
    return q64(f32(bits))


def ref_cvtdq2pd(b):
    lo, hi = si32(u32s(b[:16])[0]), si32(u32s(b[:16])[1])
    return struct.pack("<2d", float(lo), float(hi))


def ref_cvtdq2ps(b):
    return b"".join(struct.pack("<f", float(si32(w))) for w in u32s(b[:16]))


def ref_cvtpd2dq(b, trunc):
    q0, q1 = u64s(b[:16])
    return pk_d(cvt_i32(d64(q0), trunc), cvt_i32(d64(q1), trunc)) + b"\x00" * 8


def ref_cvtpd2ps(b):
    q0, q1 = u64s(b[:16])
    return pk_d(ref_f64_to_f32(q0), ref_f64_to_f32(q1)) + b"\x00" * 8


def ref_cvtps2dq(b, trunc):
    return pk_d(*(ref_f32_to_i32(w, trunc) for w in u32s(b[:16])))


def ref_cvtps2pd(b):
    d0, d1 = u32s(b[:16])[:2]
    return pk_q(ref_f32_to_f64(d0), ref_f32_to_f64(d1))


def reference(name, a, b, altered=False):
    """Expected 32-byte output region; unwritten bytes keep the FILL pattern."""
    out = bytearray(FILL)
    aq0, aq1 = u64s(a[:16])
    bq0, bq1 = u64s(b[:16])

    if name.startswith("op_cmppd"):
        pred = 5 if name == "op_cmppd_mem" else int(name[-1])
        if altered and name in ALTERED_PREDICATES:
            pred = ALTERED_PREDICATES[name][1]
        out[0:16] = pk_q(MASK64 if cmp_pred(d64(aq0), d64(bq0), pred) else 0,
                         MASK64 if cmp_pred(d64(aq1), d64(bq1), pred) else 0)
    elif name.startswith("op_cmpsd"):
        pred = CMPSD_MEM_PREDICATES.get(name)
        if pred is None:
            pred = int(name[-1])
        if altered and name in ALTERED_PREDICATES:
            pred = ALTERED_PREDICATES[name][1]
        src = u64(b[1:9]) if name == "op_cmpsd_mem_plus1" else bq0
        # Only the low double is compared; bits 127:64 of the destination are kept.
        out[0:16] = pk_q(MASK64 if cmp_pred(d64(aq0), d64(src), pred) else 0, aq1)
    elif name == "op_maxsd_mem_plus1":
        out[0:16] = pk_q(ref_maxsd(aq0, u64(b[1:9])), aq1)
    elif name.startswith("op_maxsd"):
        out[0:16] = pk_q(ref_maxsd(aq0, bq0), aq1)
    elif name.startswith("op_minsd"):
        out[0:16] = pk_q(ref_minsd(aq0, bq0), aq1)
    elif name.startswith("op_sqrtsd"):
        out[0:16] = pk_q(ref_sqrtsd(bq0), aq1)
    elif name == "op_cvtsd2si_r32_reg":
        out[0:4] = pk_d(ref_f64_to_i32(aq0))
    elif name == "op_cvtsd2si_r32_mem":
        out[0:4] = pk_d(ref_f64_to_i32(bq0))
    elif name == "op_cvtsd2si_r64_reg":
        out[0:8] = pk_q(ref_f64_to_i64(aq0))
    elif name == "op_cvtsd2si_r64_mem":
        out[0:8] = pk_q(ref_f64_to_i64(bq0))
    elif name.startswith("op_cvtdq2pd") or name == "op_cvtpi2pd_mem":
        out[0:16] = ref_cvtdq2pd(b)
    elif name.startswith("op_cvtdq2ps"):
        out[0:16] = ref_cvtdq2ps(b)
    elif name.startswith("op_cvtpd2dq"):
        out[0:16] = ref_cvtpd2dq(b, trunc=False)
    elif name.startswith("op_cvttpd2dq"):
        out[0:16] = ref_cvtpd2dq(b, trunc=True)
    elif name.startswith("op_cvtpd2ps"):
        out[0:16] = ref_cvtpd2ps(b)
    elif name.startswith("op_cvtps2dq"):
        out[0:16] = ref_cvtps2dq(b, trunc=False)
    elif name.startswith("op_cvttps2dq"):
        out[0:16] = ref_cvtps2dq(b, trunc=True)
    elif name.startswith("op_cvtps2pd"):
        out[0:16] = ref_cvtps2pd(b)
    elif name == "op_movhpd_load":
        out[0:16] = pk_q(aq0, bq0)
    elif name == "op_movlpd_load":
        out[0:16] = pk_q(bq0, aq1)
    elif name == "op_movhpd_store":
        out[8:16] = pk_q(aq1)
    elif name == "op_movlpd_store":
        out[8:16] = pk_q(aq0)
    elif name == "op_movhpd_load_plus1":
        out[0:16] = pk_q(aq0, u64(b[1:9]))
    elif name == "op_movlpd_load_plus1":
        out[0:16] = pk_q(u64(b[1:9]), aq1)
    elif name == "op_movhpd_store_plus1":
        out[9:17] = pk_q(aq1)
    elif name == "op_movlpd_store_plus1":
        out[9:17] = pk_q(aq0)
    elif name == "op_cvtsd2si_mem_plus1":
        out[0:4] = pk_d(ref_f64_to_i32(u64(b[1:9])))
    elif name == "op_movnti32":
        out[0:4] = b[0:4]
    elif name == "op_movnti64":
        out[0:8] = b[0:8]
    elif name == "op_pipeline":
        cq0, cq1 = u64s(b[16:32])
        m0 = MASK64 if cmp_pred(d64(aq0), d64(bq0), 2) else 0
        m1 = MASK64 if cmp_pred(d64(aq1), d64(bq1), 2) else 0
        xmm3_q0 = ref_f64_to_f32(m0) | (ref_f64_to_f32(m1) << 32)
        xmm3 = pk_q(xmm3_q0, cq1)
        x = ref_maxsd(u64s(xmm3)[0], bq0)
        x = ref_minsd(x, cq0)
        eax = ref_f64_to_i32(x)
        out[0:16] = pk_q(ref_sqrtsd(x), cq1)
        out[16:20] = pk_d(eax)
    else:
        # Full-vector copies and stores: movapd/movdqa/movntdq/movntpd/fences move a to out.
        out[0:16] = a[:16]
    return bytes(out)


DOUBLE_BITS = (
    0x3FF8000000000000,  # 1.5
    0xC002000000000000,  # -2.25
    0x0000000000000000,  # +0.0
    0x8000000000000000,  # -0.0
    0x3FF0000000000000,  # 1.0
    0xBFF0000000000000,  # -1.0
    0x7FF0000000000000,  # +inf
    0xFFF0000000000000,  # -inf
    0x7FF8000000000000,  # canonical qNaN
    0x7FF8000000000001,  # qNaN with payload
    0xFFF0000000000001,  # sNaN negative payload
    0x7FF0000000000001,  # sNaN payload
    0xFFF8000000000000,  # real indefinite
    0x0000000000000001,  # min denormal
    0x000FFFFFFFFFFFFF,  # max denormal
    0x7FEFFFFFFFFFFFFF,  # max finite
    0x41DFFFFFFFC00000,  # 2147483647.0
    0x41DFFFFFFFFFFFFF,  # largest double below 2147483648.0
    0x41E0000000000000,  # 2147483648.0 (i32 boundary)
    0xC1E0000000200000,  # -2147483649.0
    0x43DFFFFFFFFFFFFF,  # 9223372036854774784.0 (largest i64-reachable double)
    0x43E0000000000000,  # 2^63 (i64 boundary)
    0xC3E0000000000000,  # -2^63
    0x3FE0000000000000,  # 0.5
    0xBFE0000000000000,  # -0.5
    0xBFE8000000000000,  # -0.75
)


def double_pairs():
    """(a, b) 32-byte pairs whose qwords sweep double edge values on both lanes."""
    e = DOUBLE_BITS
    combos = [
        (e[0], e[1]), (e[1], e[0]), (e[2], e[3]), (e[3], e[2]),
        (e[4], e[4]), (e[5], e[5]), (e[6], e[7]), (e[7], e[6]),
        (e[8], e[0]), (e[0], e[8]), (e[9], e[10]), (e[10], e[11]),
        (e[12], e[13]), (e[14], e[0]), (e[15], e[1]), (e[16], e[17]),
        (e[18], e[19]), (e[20], e[21]), (e[22], e[23]), (e[24], e[25]),
    ]
    pairs = [(pk_q(x, y) + pk_q(y, x), pk_q(y, x) + pk_q(x, y)) for x, y in combos]
    rng = random.Random(0x5EED2)
    for _ in range(8):
        pairs.append((rng.randbytes(32), rng.randbytes(32)))
    return pairs


def int32_inputs():
    """b carries four i32 lanes swept for the dq/pi converts."""
    quads = [
        (0, 1, -1, 0x7FFFFFFF),
        (-0x80000000, 0x40000000, -0x40000000, 123456789),
        (16777217, -16777217, 16777215, -16777215),
        (0x01020304, -0x01020304, 0x55555555, -0x55555555),
        (1073741824, -1073741824, 536870912, -536870912),
        (2139095040, -2139095040, 2000000000, -2000000000),
        (1, 2, 4, 8),
        (-1, -2, -4, -8),
    ]
    rng = random.Random(0x1E32)
    out = []
    for quad in quads:
        a = rng.randbytes(32)
        b = pk_d(*quad) + pk_d(*reversed(quad))
        out.append((a, b))
    for _ in range(4):
        out.append((rng.randbytes(32), rng.randbytes(32)))
    return out


def float_inputs():
    """b carries four f32 lanes swept for the ps converts."""
    quads = [
        (0x3FC00000, 0xC0100000, 0x3F800000, 0xBF800000),
        (0x00000000, 0x80000000, 0x7F800000, 0xFF800000),
        (0x7FC00000, 0x7FC00001, 0x7F800001, 0xFF800001),
        (0x00000001, 0x007FFFFF, 0x7F7FFFFF, 0xFF7FFFFF),
        (0x4EFFFFFF, 0x4F000000, 0xCF000000, 0xCF000001),
        (0x3F000000, 0xBF000000, 0x4B000000, 0xCB000000),
        (0x4B800001, 0xCB800001, 0x4B7FFFFF, 0xCB7FFFFF),
        (0x00000002, 0x80000002, 0x4F000001, 0xCF000002),
    ]
    rng = random.Random(0xF32C)
    out = []
    for quad in quads:
        a = rng.randbytes(32)
        b = pk_d(*quad) + pk_d(*reversed(quad))
        out.append((a, b))
    for _ in range(4):
        out.append((rng.randbytes(32), rng.randbytes(32)))
    return out


def cmpsd_pairs(offset=0):
    """Every ordered pair of DOUBLE_BITS on the compared lane, dirty distinct upper qwords.

    With a nonzero offset the second operand starts that many bytes into b, for the
    unaligned m64 export; the bytes before it are a filler pattern.
    """
    rng = random.Random(0xC3D5 + offset)
    pairs = []
    for x in DOUBLE_BITS:
        for y in DOUBLE_BITS:
            a = pk_q(x, rng.getrandbits(64)) + rng.randbytes(16)
            b = b"\x77" * offset + pk_q(y, rng.getrandbits(64)) + rng.randbytes(16)
            pairs.append((a, b[:32]))
    return pairs + [(rng.randbytes(32), rng.randbytes(32)) for _ in range(8)]


def general_pairs():
    pairs = [
        (bytes(range(0x20, 0x40)), bytes(range(0x80, 0xA0))),
        (b"\xFF" * 32, b"\x00" * 32),
        (bytes(0x11 * i & 0xFF for i in range(32)), bytes(0x07 * i & 0xFF for i in range(32))),
        (b"\xAA" * 16 + b"\x55" * 16, b"\x33" * 16 + b"\xCC" * 16),
    ]
    pairs += double_pairs()[:6]
    return pairs


DQ_OPS = ("op_cvtdq2pd", "op_cvtdq2ps", "op_cvtpi2pd")
PS_OPS = ("op_cvtps2dq", "op_cvttps2dq", "op_cvtps2pd")
DBL_OPS = ("op_cmppd", "op_maxsd", "op_minsd", "op_sqrtsd", "op_cvtpd2dq", "op_cvtpd2ps",
           "op_cvttpd2dq", "op_cvtsd2si", "op_movhpd_load", "op_movlpd_load", "op_pipeline")


def inputs_for(name):
    if name.startswith("op_cmpsd"):
        return cmpsd_pairs(1 if name == "op_cmpsd_mem_plus1" else 0)
    if name.startswith(DQ_OPS):
        return int32_inputs()
    if name.startswith(PS_OPS):
        return float_inputs()
    if name.startswith(DBL_OPS):
        return double_pairs()
    return general_pairs()


# --- Guarded buffer helpers ---

def alloc_aligned_buffer(payload, align=16):
    """Allocate a guarded buffer whose payload starts at an aligned address."""
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
        raise AssertionError(
            f"{label} {name} case {idx}: got {got.hex()} != expected {want.hex()}")


def bind_exports(dll_path, prefix="", names=EXPORTS):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in names:
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
            want = reference(name, a, b)
            got_n = invoke(native[name], a, b)
            got_l = invoke(lifted[name], a, b)
            assert_equal_output("native", name, idx, got_n, want)
            assert_equal_output("lifted", name, idx, got_l, want)
            total += 1

    # Altered-immediate divergence controls: each export must match the altered
    # predicate's reference and diverge from the original somewhere.
    changed = {}
    for name in ALTERED_PREDICATES:
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

    # Output mutation negative control.
    sample_a, sample_b = double_pairs()[0]
    expected = reference("op_pipeline", sample_a, sample_b)
    mutated = bytearray(expected)
    mutated[7] ^= 0x01
    try:
        assert_equal_output("mutation", "op_pipeline", 0, bytes(mutated), expected)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated output was not caught by comparator")

    print(f"{total} native/lifted/reference SSE2 cases across {len(EXPORTS)} exports matched "
          f"bit-for-bit; altered controls diverged on {changed}; output mutation control caught")


def check_daz_extrema(native_path, lifted_path):
    """MAXSD and MINSD under MXCSR.DAZ return a denormal operand as a zero of its sign."""
    names = ("op_maxsd_reg", "op_minsd_reg")
    native, lifted = bind_exports(native_path, "", names), bind_exports(lifted_path, "lifted_", names)
    host = ctypes.CDLL(str(Path(native_path).resolve()))
    for function in (host.read_mxcsr, host.write_mxcsr):
        function.argtypes, function.restype = [ctypes.POINTER(ctypes.c_uint32)], None
    denormal, sign, minus_one, one = 0x000123456789ABCD, 1 << 63, 0xBFF0000000000000, 0x3FF0000000000000
    # (export, destination, source, result): the result is the second operand or the first, flushed by DAZ.
    cases = (("op_maxsd_reg", denormal, minus_one, 0), ("op_maxsd_reg", denormal | sign, minus_one, sign),
             ("op_minsd_reg", one, denormal | sign, sign), ("op_minsd_reg", one, denormal, 0))
    saved = ctypes.c_uint32()
    host.read_mxcsr(ctypes.byref(saved))
    try:
        host.write_mxcsr(ctypes.byref(ctypes.c_uint32(0x1FC0)))
        for name, a0, b0, result in cases:
            a, b = pk_q(a0, 7), pk_q(b0, 9)
            unflushed = a0 if name == "op_maxsd_reg" else b0  # what a C comparison returns
            if unflushed == result:
                raise AssertionError(f"{name}: control case cannot tell a flushed result apart")
            for label, function in (("native", native[name]), ("lifted", lifted[name])):
                assert_equal_output(f"{label} DAZ", name, hex(a0), invoke(function, a, b),
                                    pk_q(result, 7) + FILL[16:])
    finally:
        host.write_mxcsr(ctypes.byref(saved))
    print(f"{len(cases)} MAXSD/MINSD denormal results under MXCSR.DAZ matched the native instruction")


# --- Alternative-encoding acceptance & translator rejection controls ---

def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_alt_encodings_and_rejections(original_path):
    """Verify alternate encodings lift cleanly and all out-of-scope/illegal forms reject."""
    ld_xmm0 = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_xmm1 = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM1, MemoryOperand(R.RCX))
    ld_xmm2 = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM2, MemoryOperand(R.RDX))
    st_xmm0 = Instruction.create_mem_reg(Code.MOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)
    ret_ins = Instruction.create(Code.RETNQ)

    # Encodings MASM does not emit in the fixture: non-rax cvtsd2si destinations,
    # non-rax movnti sources, displaced memory forms and standalone fences.
    alt_accept = [
        ("cvtsd2si_r9", (ld_xmm1, Instruction.create_reg_reg(
            Code.CVTSD2SI_R64_XMMM64, R.R9, R.XMM1),
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.R8), R.R9), ret_ins)),
        ("cvtsd2si_r10d", (ld_xmm1, Instruction.create_reg_reg(
            Code.CVTSD2SI_R32_XMMM64, R.R10D, R.XMM1),
            Instruction.create_mem_reg(Code.MOV_RM32_R32, MemoryOperand(R.R8), R.R10D), ret_ins)),
        ("movnti_r9d", (Instruction.create_reg_mem(Code.MOV_R32_RM32, R.R9D, MemoryOperand(R.RDX)),
            Instruction.create_mem_reg(Code.MOVNTI_M32_R32, MemoryOperand(R.R8), R.R9D), ret_ins)),
        ("movnti_r10", (Instruction.create_reg_mem(Code.MOV_R64_RM64, R.R10, MemoryOperand(R.RDX)),
            Instruction.create_mem_reg(Code.MOVNTI_M64_R64, MemoryOperand(R.R8), R.R10), ret_ins)),
        ("lfence_standalone", (Instruction.create(Code.LFENCE), ret_ins)),
        ("mfence_standalone", (Instruction.create(Code.MFENCE), ret_ins)),
        ("cmppd_mem_disp", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPPD_XMM_XMMM128_IMM8, R.XMM0,
            MemoryOperand(R.RDX, displ=0x10, displ_size=1), 3), st_xmm0, ret_ins)),
        ("movhpd_load_disp", (ld_xmm0, Instruction.create_reg_mem(
            Code.MOVHPD_XMM_M64, R.XMM0,
            MemoryOperand(R.RDX, displ=0x18, displ_size=1)), st_xmm0, ret_ins)),
        ("movlpd_store_disp", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVLPD_M64_XMM, MemoryOperand(R.R8, displ=0x18, displ_size=1), R.XMM0), ret_ins)),
        ("cmppd_self", (ld_xmm0, Instruction.create_reg_reg_u32(
            Code.CMPPD_XMM_XMMM128_IMM8, R.XMM0, R.XMM0, 0), st_xmm0, ret_ins)),
        ("cmpsd_mem_disp", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0,
            MemoryOperand(R.RDX, displ=0x18, displ_size=1), 3), st_xmm0, ret_ins)),
        ("cmpsd_self", (ld_xmm0, Instruction.create_reg_reg_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, R.XMM0, 7), st_xmm0, ret_ins)),
    ]
    alt_expected = {
        "cvtsd2si_r9": "r9 = op_cvtsd2si64(xmm1.q[0])",
        "cvtsd2si_r10d": "r10 = static_cast<word>(op_cvtsd2si32(xmm1.q[0]))",
        "movnti_r9d": "store32(r8, static_cast<std::uint32_t>(r9))",
        "movnti_r10": "store64(r8, r10)",
        "lfence_standalone": "_mm_lfence();",
        "mfence_standalone": "_mm_mfence();",
        "cmppd_mem_disp": "load128((rdx + 0x10ull))",
        "movhpd_load_disp": ".q[1] = load64((rdx + 0x18ull))",
        "movlpd_store_disp": "store64((r8 + 0x18ull), xmm0.q[0])",
        "cmppd_self": "op_cmppd(xmm0, xmm0, 0x0u)",
        "cmpsd_mem_disp": "xmm0.q[0] = op_cmpsd_q(xmm0.q[0], load64((rdx + 0x18ull)), 0x3u)",
        "cmpsd_self": "xmm0.q[0] = op_cmpsd_q(xmm0.q[0], xmm0.q[0], 0x7u)",
    }
    for label, seq in alt_accept:
        src, count = lift_code(label, 0x1000, encode_seq(*seq))
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if alt_expected[label] not in src:
            raise AssertionError(
                f"alternate encoding {label} lowered without the expected statement "
                f"{alt_expected[label]!r}")

    # 1) The 36 SSE2-group forms reserved for sibling translators must all reject.
    reserved_cases = [
        ("addpd", Instruction.create_reg_reg(Code.ADDPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("addsd", Instruction.create_reg_reg(Code.ADDSD_XMM_XMMM64, R.XMM0, R.XMM1)),
        ("andnpd", Instruction.create_reg_reg(Code.ANDNPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("andpd", Instruction.create_reg_reg(Code.ANDPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("comisd", Instruction.create_reg_reg(Code.COMISD_XMM_XMMM64, R.XMM0, R.XMM1)),
        ("cvtsi2sd", Instruction.create_reg_reg(Code.CVTSI2SD_XMM_RM32, R.XMM0, R.EAX)),
        ("cvttsd2si", Instruction.create_reg_reg(Code.CVTTSD2SI_R32_XMMM64, R.EAX, R.XMM1)),
        ("cvtss2sd", Instruction.create_reg_reg(Code.CVTSS2SD_XMM_XMMM32, R.XMM0, R.XMM1)),
        ("cvtsd2ss", Instruction.create_reg_reg(Code.CVTSD2SS_XMM_XMMM64, R.XMM0, R.XMM1)),
        ("divpd", Instruction.create_reg_reg(Code.DIVPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("divsd", Instruction.create_reg_reg(Code.DIVSD_XMM_XMMM64, R.XMM0, R.XMM1)),
        ("maxpd", Instruction.create_reg_reg(Code.MAXPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("minpd", Instruction.create_reg_reg(Code.MINPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("movdqu", Instruction.create_reg_reg(Code.MOVDQU_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("movmskpd", Instruction.create_reg_reg(Code.MOVMSKPD_R32_XMM, R.EAX, R.XMM1)),
        ("movupd", Instruction.create_reg_reg(Code.MOVUPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("mulpd", Instruction.create_reg_reg(Code.MULPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("mulsd", Instruction.create_reg_reg(Code.MULSD_XMM_XMMM64, R.XMM0, R.XMM1)),
        ("orpd", Instruction.create_reg_reg(Code.ORPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("pmuludq", Instruction.create_reg_reg(Code.PMULUDQ_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("pshufd", Instruction.create_reg_reg_u32(Code.PSHUFD_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0x1B)),
        ("pshufhw", Instruction.create_reg_reg_u32(Code.PSHUFHW_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0x1B)),
        ("pshuflw", Instruction.create_reg_reg_u32(Code.PSHUFLW_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0x1B)),
        ("pslldq", Instruction.create_reg_u32(Code.PSLLDQ_XMM_IMM8, R.XMM0, 4)),
        ("psrldq", Instruction.create_reg_u32(Code.PSRLDQ_XMM_IMM8, R.XMM0, 4)),
        ("psubq", Instruction.create_reg_reg(Code.PSUBQ_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("punpckhqdq", Instruction.create_reg_reg(Code.PUNPCKHQDQ_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("punpcklqdq", Instruction.create_reg_reg(Code.PUNPCKLQDQ_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("shufpd", Instruction.create_reg_reg_u32(Code.SHUFPD_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 1)),
        ("sqrtpd", Instruction.create_reg_reg(Code.SQRTPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("subpd", Instruction.create_reg_reg(Code.SUBPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("subsd", Instruction.create_reg_reg(Code.SUBSD_XMM_XMMM64, R.XMM0, R.XMM1)),
        ("ucomisd", Instruction.create_reg_reg(Code.UCOMISD_XMM_XMMM64, R.XMM0, R.XMM1)),
        ("unpckhpd", Instruction.create_reg_reg(Code.UNPCKHPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("unpcklpd", Instruction.create_reg_reg(Code.UNPCKLPD_XMM_XMMM128, R.XMM0, R.XMM1)),
        ("xorpd", Instruction.create_reg_reg(Code.XORPD_XMM_XMMM128, R.XMM0, R.XMM1)),
    ]

    # 2) ABI-infeasible task forms: MMX register operands and the implicit-RDI store.
    abi_infeasible_cases = [
        ("cvtpd2pi_mm", (Instruction.create_reg_reg(
            Code.CVTPD2PI_MM_XMMM128, R.MM0, R.XMM1),), "MMX register operand outside the leaf ABI"),
        ("cvtpd2pi_mm_mem", (Instruction.create_reg_mem(
            Code.CVTPD2PI_MM_XMMM128, R.MM0, MemoryOperand(R.RCX)),), "MMX register operand outside the leaf ABI"),
        ("cvttpd2pi_mm", (Instruction.create_reg_reg(
            Code.CVTTPD2PI_MM_XMMM128, R.MM0, R.XMM1),), "MMX register operand outside the leaf ABI"),
        ("movdq2q_mm", (Instruction.create_reg_reg(
            Code.MOVDQ2Q_MM_XMM, R.MM0, R.XMM1),), "MMX register operand outside the leaf ABI"),
        ("movq2dq_mm", (Instruction.create_reg_reg(
            Code.MOVQ2DQ_XMM_MM, R.XMM0, R.MM1),), "MMX register operand outside the leaf ABI"),
        ("cvtpi2pd_mm_src", (Instruction.create_reg_reg(
            Code.CVTPI2PD_XMM_MMM64, R.XMM0, R.MM1),), "MMX register operand outside the leaf ABI"),
    ]

    # 3) cmppd immediates above 7 are a reserved encoding and reject by policy.
    cmppd_imm_case = ("cmppd_imm8", (ld_xmm0, Instruction.create_reg_reg_u32(
        Code.CMPPD_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0x1B), ret_ins), "unsupported cmppd predicate")

    # The cmpsd immediate above 7 is reserved in the legacy encoding and rejects too.
    cmpsd_imm_case = ("cmpsd_imm8", (ld_xmm0, Instruction.create_reg_reg_u32(
        Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, R.XMM1, 0x1B), ret_ins), "unsupported cmpsd predicate")

    # 4) VEX/EVEX encodings of implemented forms reject as non-legacy.
    nonlegacy_cases = [
        ("vaddpd_vex", (ld_xmm1, ld_xmm2, Instruction.create_reg_reg_reg(
            Code.VEX_VADDPD_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2), ret_ins), "non-legacy encoding"),
        ("vmaxsd_vex", (ld_xmm1, ld_xmm2, Instruction.create_reg_reg_reg(
            Code.VEX_VMAXSD_XMM_XMM_XMMM64, R.XMM0, R.XMM1, R.XMM2), ret_ins), "non-legacy encoding"),
        ("vcmpsd_vex", (ld_xmm1, ld_xmm2, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VCMPSD_XMM_XMM_XMMM64_IMM8, R.XMM0, R.XMM1, R.XMM2, 0), ret_ins), "non-legacy encoding"),
        ("vmovapd_vex", (Instruction.create_reg_mem(
            Code.VEX_VMOVAPD_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret_ins), "non-legacy encoding"),
        ("vmovapd_evex", (Instruction.create_reg_mem(
            Code.EVEX_VMOVAPD_XMM_K1Z_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret_ins), "non-legacy encoding"),
    ]

    # 5) Prefixes, register-file policy, state tracking and addressing violations.
    seg_load = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    seg_load.segment_prefix = R.FS
    other_cases = [
        ("unaligned_movdqa_source", (Instruction.create_reg_mem(
            Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=1)), ret_ins),
         "unaligned 16-byte memory operand"),
        ("unaligned_cmppd_source", (ld_xmm0, Instruction.create_reg_mem_i32(
            Code.CMPPD_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RDX, displ=8), 0), ret_ins),
         "unaligned 16-byte memory operand"),
        ("unaligned_movntdq_destination", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVNTDQ_M128_XMM, MemoryOperand(R.R8, displ=8), R.XMM0), ret_ins),
         "unaligned 16-byte memory operand"),
        ("nonvolatile_xmm6", (Instruction.create_reg_mem(
            Code.MOVDQA_XMM_XMMM128, R.XMM6, MemoryOperand(R.RCX)), ret_ins), "nonvolatile vector register"),
        ("nonvolatile_xmm6_cmpsd", (ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM6, R.XMM1, 0), ret_ins), "nonvolatile vector register"),
        ("undefined_cmpsd_dst", (ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPSD_XMM_XMMM64_IMM8, R.XMM0, R.XMM1, 0), ret_ins), "undefined vector register xmm0"),
        ("nonvolatile_xmm15", (Instruction.create_reg_reg(
            Code.MAXSD_XMM_XMMM64, R.XMM15, R.XMM1), ret_ins), "nonvolatile vector register"),
        ("nonvolatile_gpr_ebx", (ld_xmm0, Instruction.create_reg_reg(
            Code.CVTSD2SI_R32_XMMM64, R.EBX, R.XMM0), ret_ins), "nonvolatile GPR"),
        ("arg_gpr_ecx", (ld_xmm0, Instruction.create_reg_reg(
            Code.CVTSD2SI_R32_XMMM64, R.ECX, R.XMM0), ret_ins), "unsupported argument GPR operand"),
        ("undefined_xmm3", (ld_xmm0, Instruction.create_reg_reg(
            Code.MAXSD_XMM_XMMM64, R.XMM0, R.XMM3), ret_ins), "undefined vector register xmm3"),
        ("undefined_dst_merge", (Instruction.create_reg_mem(
            Code.MOVHPD_XMM_M64, R.XMM0, MemoryOperand(R.RDX)), ret_ins), "undefined vector register xmm0"),
        ("undefined_gpr", (Instruction.create_mem_reg(
            Code.MOVNTI_M32_R32, MemoryOperand(R.R8), R.EAX), ret_ins), "undefined GPR rax"),
        ("mov_reg_to_reg", (Instruction.create_reg_reg(
            Code.MOV_R64_RM64, R.RAX, R.RCX), ret_ins), "unsupported mov instruction"),
        ("rip_rel_mem", (Instruction.create_reg_mem(
            Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RIP, displ=0x1010, displ_size=1)), ret_ins),
            "unsupported memory operand"),
        ("rsp_mem", (Instruction.create_reg_mem(
            Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RSP)), ret_ins),
            "unsupported memory operand"),
        ("addr32_mem", (Instruction.create_reg_mem(
            Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.ECX)), ret_ins),
            "unsupported memory operand"),
        ("indexed_mem", (Instruction.create_reg_mem(
            Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, R.RDX, 4)), ret_ins),
            "unsupported memory operand"),
        ("non_arg_base_rax", (Instruction.create_reg_mem(
            Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RAX)), ret_ins),
            "unsupported memory operand"),
        ("segment_prefix", (seg_load, ret_ins), "unsupported instruction prefix"),
        ("missing_ret", (ld_xmm0,), "missing terminal ret"),
        ("after_ret", (ret_ins, ld_xmm0), "unreachable instruction after ret"),
    ]
    # Raw-byte controls: iced reports lock on cmpxchg and rep on movsb, and cannot
    # encode maskmovdqu's implicit DS:rdi operand (66 0F F7 /r emits it raw).
    raw_prefix_cases = [
        ("lock_cmpxchg", bytes.fromhex("f00fb109") + encode_seq(ret_ins), "unsupported instruction prefix"),
        ("rep_movsb", bytes.fromhex("f3a4") + encode_seq(ret_ins), "unsupported instruction prefix"),
        ("maskmovdqu", bytes.fromhex("660ff7c8") + encode_seq(ret_ins),
         "implicit RDI memory operand outside the leaf ABI"),
    ]

    all_unit = (
        [(lbl, (ld_xmm0, ld_xmm1, ins, ret_ins), "unsupported instruction")
         for lbl, ins in reserved_cases]
        + [(lbl, seq + (ret_ins,), reason) for lbl, seq, reason in abi_infeasible_cases]
        + [cmppd_imm_case, cmpsd_imm_case]
        + nonlegacy_cases
        + other_cases
    )

    for label, seq, expected_reason in all_unit:
        raw = encode_seq(*seq)
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator unexpectedly accepted rejected control")

    for label, raw, expected_reason in raw_prefix_cases:
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

    other_sequences = {label: seq for label, seq, _ in other_cases}
    pe_patches = [
        ("pe_addpd", encode_seq(ld_xmm1, reserved_cases[0][1], ret_ins), "unsupported instruction"),
        ("pe_movdqu_load", encode_seq(Instruction.create_reg_mem(
            Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret_ins), "unsupported instruction"),
        ("pe_pshufd", encode_seq(ld_xmm1, reserved_cases[20][1], ret_ins), "unsupported instruction"),
        ("pe_cvtpd2pi", encode_seq(*abi_infeasible_cases[0][1], ret_ins), "MMX register operand"),
        ("pe_maskmovdqu", bytes.fromhex("660ff7c8") + encode_seq(ret_ins), "implicit RDI"),
        ("pe_vex_vaddpd", encode_seq(*nonlegacy_cases[0][1]), "non-legacy encoding"),
        ("pe_cmppd_imm8", encode_seq(*cmppd_imm_case[1]), "unsupported cmppd predicate"),
        ("pe_cmpsd_imm8", encode_seq(*cmpsd_imm_case[1]), "unsupported cmpsd predicate"),
        ("pe_nonvolatile_xmm6", encode_seq(*other_sequences["nonvolatile_xmm6"]), "nonvolatile vector register"),
        ("pe_undef_xmm3", encode_seq(*other_sequences["undefined_xmm3"]), "undefined vector register xmm3"),
        ("pe_rsp_mem", encode_seq(*other_sequences["rsp_mem"]), "unsupported memory operand"),
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

    unit_total = len(all_unit) + len(raw_prefix_cases)
    print(f"{len(alt_accept)} alternate-encoding lift checks, {unit_total} unit rejection controls "
          f"({len(reserved_cases)} reserved SSE2 forms, {len(abi_infeasible_cases) + 1} MMX/implicit-address, "
          f"2 cmppd/cmpsd immediate, {len(nonlegacy_cases)} VEX/EVEX, "
          f"{len(other_cases) + len(raw_prefix_cases) - 1} ABI/state/addressing/prefix) "
          f"and {len(pe_patches)} PE-patched rejection controls passed")


if __name__ == "__main__":
    check_differential(*sys.argv[1:4])
    check_daz_extrema(*sys.argv[1:3])
    check_alt_encodings_and_rejections(sys.argv[1])
