"""Differential checks, mutation controls and rejection controls for legacy SSE FP AOT.

The packed arithmetic (ADDPS, SUBPS, MULPS, DIVPS, MINPS, MAXPS, SQRTPS) is also run under all
sixteen combinations of MXCSR rounding mode, DAZ and FTZ, each against the exact reference in
sse_fp_ref.py and against the original processor code."""

import ctypes
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Decoder, Encoder, Instruction, MemoryOperand, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_sse_fp_aot import lift_code, translate
import sse_fp_ref as ref

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
GUARD_PRE = bytes(0xA0 + (i % 16) for i in range(32))
GUARD_POST = bytes(0xC0 + (i % 16) for i in range(32))
FILL = b"\x5A" * 32

EXPORTS = (
    # cmpps predicates 0..7 + aligned m128 / disp / self (11)
    "op_cmpps_0", "op_cmpps_1", "op_cmpps_2", "op_cmpps_3",
    "op_cmpps_4", "op_cmpps_5", "op_cmpps_6", "op_cmpps_7",
    "op_cmpps_mem", "op_cmpps_mem_disp", "op_cmpps_self",
    # cmpss predicates 0..7 + m32 / unaligned +1 / self (11)
    "op_cmpss_0", "op_cmpss_1", "op_cmpss_2", "op_cmpss_3",
    "op_cmpss_4", "op_cmpss_5", "op_cmpss_6", "op_cmpss_7",
    "op_cmpss_mem", "op_cmpss_mem_plus1", "op_cmpss_self",
    # movlps / movhps aligned & +1 unaligned loads/stores (8)
    "op_movlps_load", "op_movlps_store", "op_movhps_load", "op_movhps_store",
    "op_movlps_load_plus1", "op_movhps_load_plus1",
    "op_movlps_store_plus1", "op_movhps_store_plus1",
    # movntps aligned stores (2)
    "op_movntps", "op_movntps_disp16",
    # Transport & chained pipeline (2)
    "op_movaps_reg", "op_pipeline",
)

ARITH_OPS = ("add", "sub", "mul", "div", "min", "max", "sqrt")
ARITH_SUFFIXES = ("", "_mem", "_mem_disp", "_self")
ARITH_EXPORTS = tuple(f"op_{op}ps{suffix}" for op in ARITH_OPS for suffix in ARITH_SUFFIXES) + (
    "op_arith_pipeline",)
EXPORTS = EXPORTS + ARITH_EXPORTS
# The ALTER_IMM build assembles these register forms with another operation.
ALTERED_OP = {"add": "sub", "sub": "add", "mul": "div", "div": "mul", "min": "max", "max": "min",
              "sqrt": "add"}

ALTERED_PREDICATES = {
    "op_cmpps_1": (1, 5),
    "op_cmpps_3": (3, 7),
    "op_cmpps_6": (6, 2),
    "op_cmpss_1": (1, 5),
    "op_cmpss_3": (3, 7),
    "op_cmpss_6": (6, 2),
}


# --- Independent Python IEEE-754 single-precision reference model ---

def u32s(buf):
    return struct.unpack("<" + "I" * (len(buf) // 4), buf)


def u32(buf):
    return struct.unpack("<I", buf)[0]


def u64s(buf):
    return struct.unpack("<2Q", buf)


def u64(buf):
    return struct.unpack("<Q", buf)[0]


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def pk_d(*vals):
    return struct.pack("<" + "I" * len(vals), *(v & MASK32 for v in vals))


def pk_q(*vals):
    return struct.pack("<" + "Q" * len(vals), *(v & MASK64 for v in vals))


def cmp_pred(x, y, imm):
    """CMPPS/CMPSS predicates 0..7 (EQ_OQ LT_OS LE_OS UNORD_Q NEQ_UQ NLT_US NLE_US ORD_Q)."""
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


def ref_cmpps(a_vec, b_vec, pred):
    ad = u32s(a_vec)
    bd = u32s(b_vec)
    return pk_d(*(MASK32 if cmp_pred(f32(x), f32(y), pred) else 0 for x, y in zip(ad, bd)))


def ref_cmpss(a_vec, b_bits, pred):
    ad = list(u32s(a_vec))
    ad[0] = MASK32 if cmp_pred(f32(ad[0]), f32(b_bits), pred) else 0
    return pk_d(*ad)


def arith_reference(name, a, b, mode=ref.DEFAULT_MODE, altered=False, quirk=None):
    """(expected 32-byte output, wildcard mask) of a packed arithmetic export under an MXCSR mode.

    A wildcard byte marks a lane sse_fp_ref cannot decide; it is zero everywhere in the default mode."""
    out, wild = bytearray(FILL), bytearray(32)
    if name == "op_arith_pipeline":
        low, high = ref.chain(a, b, mode, quirk, multiply="div" if altered else "mul")
        out[0:16], wild[0:16] = ref.pack(low)
        out[16:32], wild[16:32] = ref.pack(high)
        return bytes(out), bytes(wild)
    op, suffix = name[3:].split("ps", 1)
    first, source = a[0:16], {"": b[0:16], "_mem": b[0:16], "_mem_disp": b[16:32], "_self": a[0:16]}[suffix]
    if altered and suffix == "":
        op = ALTERED_OP[op]
    elif altered and suffix == "_self":
        op, source = ("add" if op == "sqrt" else op), b[0:16]
    elif altered:
        source = b[16:32] if suffix == "_mem" else b[0:16]
    out[0:16], wild[0:16] = ref.pack(ref.apply(op, first, source, mode, quirk))
    return bytes(out), bytes(wild)


def reference(name, a, b, altered=False):
    """Expected 32-byte output region; unwritten bytes keep the FILL pattern."""
    if name in ARITH_EXPORTS:
        out, wild = arith_reference(name, a, b, altered=altered)
        assert not any(wild), name
        return out
    out = bytearray(FILL)
    aq0, aq1 = u64s(a[:16])
    bq0, _ = u64s(b[:16])

    if name.startswith("op_cmpps"):
        if name == "op_cmpps_mem":
            pred, src = 5, b[0:16]
        elif name == "op_cmpps_mem_disp":
            pred, src = 2, b[16:32]
        elif name == "op_cmpps_self":
            pred, src = 0, a[0:16]
        else:
            pred, src = int(name[-1]), b[0:16]
        if altered and name in ALTERED_PREDICATES:
            pred = ALTERED_PREDICATES[name][1]
        out[0:16] = ref_cmpps(a[0:16], src, pred)
    elif name.startswith("op_cmpss"):
        if name == "op_cmpss_mem":
            pred, src32 = 5, u32(b[0:4])
        elif name == "op_cmpss_mem_plus1":
            pred, src32 = 2, u32(b[1:5])
        elif name == "op_cmpss_self":
            pred, src32 = 0, u32(a[0:4])
        else:
            pred, src32 = int(name[-1]), u32(b[0:4])
        if altered and name in ALTERED_PREDICATES:
            pred = ALTERED_PREDICATES[name][1]
        out[0:16] = ref_cmpss(a[0:16], src32, pred)
    elif name == "op_movlps_load":
        out[0:16] = pk_q(bq0, aq1)
    elif name == "op_movhps_load":
        out[0:16] = pk_q(aq0, bq0)
    elif name == "op_movlps_store":
        out[8:16] = pk_q(aq0)
    elif name == "op_movhps_store":
        out[8:16] = pk_q(aq1)
    elif name == "op_movlps_load_plus1":
        out[0:16] = pk_q(u64(b[1:9]), aq1)
    elif name == "op_movhps_load_plus1":
        out[0:16] = pk_q(aq0, u64(b[1:9]))
    elif name == "op_movlps_store_plus1":
        out[9:17] = pk_q(aq0)
    elif name == "op_movhps_store_plus1":
        out[9:17] = pk_q(aq1)
    elif name == "op_movntps_disp16":
        out[16:32] = a[1:17]
    elif name == "op_pipeline":
        xmm0 = pk_q(u64(b[8:16]), aq1)
        xmm1 = b[0:16]
        xmm2 = pk_q(u64(b[16:24]), u64(a[24:32]))
        xmm0 = ref_cmpps(xmm0, xmm1, 2)
        xmm2 = ref_cmpss(xmm2, u32(b[4:8]), 4)
        out[0:16] = xmm0
        out[16:24] = xmm2[8:16]
        out[24:32] = xmm2[0:8]
    else:
        # op_movntps, op_movaps_reg
        out[0:16] = a[0:16]
    return bytes(out)


FLOAT_BITS = (
    0x3FC00000,  # 1.5
    0xC0100000,  # -2.25
    0x00000000,  # +0.0
    0x80000000,  # -0.0
    0x3F800000,  # 1.0
    0xBF800000,  # -1.0
    0x7F800000,  # +inf
    0xFF800000,  # -inf
    0x7FC00000,  # canonical qNaN
    0x7FC00001,  # qNaN with payload
    0xFFC00000,  # negative qNaN
    0x7F800001,  # sNaN positive payload
    0xFF800001,  # sNaN negative payload
    0x7FBFFFFF,  # sNaN max payload
    0x00000001,  # min positive subnormal
    0x007FFFFF,  # max positive subnormal
    0x80000001,  # min negative subnormal
    0x00800000,  # min positive normal
    0x7F7FFFFF,  # max finite
    0xFF7FFFFF,  # -max finite
    0x3F000000,  # 0.5
    0xBF000000,  # -0.5
    0x4B800001,  # 2^24 + 1
    0xCB800001,  # -(2^24 + 1)
)


def float_pairs():
    """(a, b) 32-byte pairs sweeping single-precision edge values across all 4 lanes."""
    e = FLOAT_BITS
    combos = [
        (e[0], e[1], e[2], e[3], e[1], e[0], e[3], e[2]),
        (e[2], e[3], e[4], e[5], e[3], e[2], e[4], e[5]),
        (e[4], e[5], e[6], e[7], e[4], e[5], e[7], e[6]),
        (e[6], e[7], e[8], e[0], e[6], e[7], e[0], e[8]),
        (e[8], e[9], e[10], e[11], e[0], e[8], e[11], e[10]),
        (e[11], e[12], e[13], e[14], e[1], e[12], e[14], e[13]),
        (e[12], e[14], e[15], e[16], e[12], e[15], e[16], e[2]),
        (e[14], e[15], e[16], e[17], e[2], e[3], e[17], e[16]),
        (e[17], e[18], e[19], e[20], e[18], e[17], e[20], e[19]),
        (e[18], e[19], e[21], e[22], e[19], e[18], e[22], e[23]),
        (e[11], e[4], e[6], e[3], e[9], e[4], e[6], e[2]),
        (e[13], e[1], e[7], e[15], e[13], e[13], e[7], e[16]),
    ]
    pairs = []
    for a0, a1, a2, a3, b0, b1, b2, b3 in combos:
        a = pk_d(a0, a1, a2, a3, a3, a2, a1, a0)
        b = pk_d(b0, b1, b2, b3, b3, b2, b1, b0)
        pairs.append((a, b))
    # Ensure lane 0 explicitly tests unaligned +1 byte shifts against FLOAT_BITS edges.
    for x, y in ((e[8], e[4]), (e[11], e[12]), (e[2], e[3]), (e[14], e[16]), (e[6], e[7])):
        a = pk_d(x, e[18], e[19], e[9], e[0], e[1], e[4], e[5])
        b = b"\xAA" + pk_d(y, e[11], e[15], e[20], e[21], e[22], e[23]) + b"\x55\x33\x11"
        pairs.append((a, b))
    rng = random.Random(0x55EF9)
    for _ in range(10):
        pairs.append((rng.randbytes(32), rng.randbytes(32)))
    return pairs


def move_pairs():
    pairs = [
        (bytes(range(0x10, 0x30)), bytes(range(0x80, 0xA0))),
        (b"\xFF" * 32, b"\x00" * 32),
        (bytes((0x13 * i) & 0xFF for i in range(32)), bytes((0x29 * i) & 0xFF for i in range(32))),
    ]
    pairs.extend(float_pairs()[:6])
    return pairs


ARITH_SPECIAL = (
    0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3FC00000, 0xC0100000, 0x40000000, 0x3EAAAAAB,
    0x33800000, 0x3F800001, 0x4B800001, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC00001,
    0x00000001, 0x80000003, 0x007FFFFF, 0x00800000, 0x00800001, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F000000,
    0x5F800000, 0x1F800000, 0x3F000000, 0x00400000,
)


def arith_pairs():
    """Every ordered pair of special values (8 lanes per case), then random biased exponents."""
    lane_pairs = [(x, y) for x in ARITH_SPECIAL for y in ARITH_SPECIAL]
    pairs = []
    for start in range(0, len(lane_pairs), 8):
        chunk = lane_pairs[start:start + 8]
        pairs.append((pk_d(*(x for x, _ in chunk)), pk_d(*(y for _, y in chunk))))
    rng = random.Random(0x5A11)
    exponents = (0, 1, 2, 63, 126, 127, 128, 190, 253, 254)
    def lane():
        return (rng.getrandbits(1) << 31) | (rng.choice(exponents) << 23) | rng.getrandbits(23)
    for _ in range(30):
        pairs.append((pk_d(*(lane() for _ in range(8))), pk_d(*(lane() for _ in range(8)))))
    return pairs


def inputs_for(name):
    if name in ARITH_EXPORTS:
        return arith_pairs()
    if name.startswith(("op_cmpps", "op_cmpss")) or name == "op_pipeline":
        return float_pairs()
    return move_pairs()


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


def assert_equal_output(label, name, idx, got, want, wild=bytes(32)):
    if any((g ^ w) & ~m & 0xFF for g, w, m in zip(got, want, wild)):
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


def check_discarded_read(native_path, lifted_path):
    payload = bytes(range(32))
    for label, dll, prefix in (("native", native_path, ""), ("lifted", lifted_path, "lifted_")):
        name = "op_discarded_read"
        fn = bind_exports(dll, prefix, (name,))[name]
        source, ptr, _, _, _ = alloc_aligned_buffer(payload)
        for readable in (True, False):
            result, result_ptr, pre, off, post = alloc_aligned_buffer(FILL)
            try:
                fn(ptr, ptr if readable else 0, result_ptr)
            except OSError as error:
                assert not readable and "access violation reading" in str(error).lower(), (label, error)
            else:
                assert readable, f"{label} discarded a faulting MOVAPS read"
            expected = payload[:16] + (payload[:16] if readable else FILL[16:])
            assert_equal_output(label, name, readable, check_guards_and_read(result, pre, off, post, label), expected)


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

    # Altered-immediate divergence controls for both cmpps and cmpss.
    changed = {}
    for name in (*ALTERED_PREDICATES, *ARITH_EXPORTS):
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
    sample_a, sample_b = float_pairs()[0]
    expected = reference("op_pipeline", sample_a, sample_b)
    mutated = bytearray(expected)
    mutated[3] ^= 0x01
    try:
        assert_equal_output("mutation", "op_pipeline", 0, bytes(mutated), expected)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated output was not caught by comparator")

    print(f"{total} native/lifted/reference SSE FP cases across {len(EXPORTS)} exports matched "
          f"bit-for-bit; altered controls diverged on all {len(changed)} exports "
          f"(fewest differing cases {min(changed.values())}); output mutation control caught")


# --- Packed arithmetic under every MXCSR rounding mode, DAZ and FTZ ---

def function_address(fn):
    return ctypes.cast(fn, ctypes.c_void_p).value


def bind_mxcsr_driver(harness_path):
    """op_call_with_mxcsr(fn, a, b, out, mxcsr): the native-only harness image (never translated) that
    runs fn under a control word."""
    driver = ctypes.CDLL(str(Path(harness_path).resolve())).op_call_with_mxcsr
    driver.argtypes = [ctypes.c_uint64] * 4 + [ctypes.c_uint32]
    driver.restype = None
    return driver


def invoke_under(driver, fn, mxcsr, a, b):
    buf_a, ptr_a, a_pre, a_off, a_post = alloc_aligned_buffer(a)
    buf_b, ptr_b, b_pre, b_off, b_post = alloc_aligned_buffer(b)
    buf_out, ptr_out, o_pre, o_off, o_post = alloc_aligned_buffer(FILL)
    driver(function_address(fn), ptr_a, ptr_b, ptr_out, mxcsr)
    check_guards_and_read(buf_a, a_pre, a_off, a_post, "input_a")
    check_guards_and_read(buf_b, b_pre, b_off, b_post, "input_b")
    return check_guards_and_read(buf_out, o_pre, o_off, o_post, "output")


# Each wrong rule must disagree with the processor on some decided lane in the given mode.
REFERENCE_QUIRKS = (
    ("nan_second_source", "op_addps", ref.DEFAULT_MODE),
    ("no_quieting", "op_mulps", ref.DEFAULT_MODE),
    ("mixed_zero_negative", "op_addps", ref.DEFAULT_MODE),
    ("divide_zero_nan", "op_divps", ref.DEFAULT_MODE),
    ("zero_returns_first", "op_minps", ref.DEFAULT_MODE),
    ("zero_returns_first", "op_maxps", ref.DEFAULT_MODE),
    ("min_as_max", "op_minps", ref.DEFAULT_MODE),
    ("nan_first_operand", "op_maxps", ref.DEFAULT_MODE),
    ("sqrt_negative_zero_positive", "op_sqrtps", ref.DEFAULT_MODE),
    ("sqrt_default_nan_positive", "op_sqrtps", ref.DEFAULT_MODE),
    ("ignore_rounding_mode", "op_addps", ref.Mode(ref.RU)),
    ("ignore_rounding_mode", "op_divps", ref.Mode(ref.RZ)),
    ("ignore_rounding_mode", "op_sqrtps", ref.Mode(ref.RU)),
    ("ignore_daz", "op_mulps", ref.Mode(ref.RN, True, False)),
    ("ignore_ftz", "op_mulps", ref.Mode(ref.RN, False, True)),
)


def check_arithmetic_modes(native_path, lifted_path, harness_path):
    driver = bind_mxcsr_driver(harness_path)
    native = bind_exports(native_path, "", ARITH_EXPORTS)
    lifted = bind_exports(lifted_path, "lifted_", ARITH_EXPORTS)
    pairs = arith_pairs()
    seen = {}
    decided = wildcards = 0
    for mode in ref.ALL_MODES + tuple(ref.Mode(rc, daz, ftz) for rc in range(4)
                                      for daz, ftz in ((True, False), (False, True))):
        for name in ARITH_EXPORTS:
            for idx, (a, b) in enumerate(pairs):
                label = f"{name} [{mode.label()}]"
                got_n = invoke_under(driver, native[name], mode.mxcsr(), a, b)
                got_l = invoke_under(driver, lifted[name], mode.mxcsr(), a, b)
                want, wild = arith_reference(name, a, b, mode)
                assert_equal_output("native", label, idx, got_n, want, wild)
                # Where the reference abstains, the lifted code must still equal the processor.
                assert_equal_output("lifted", label, idx, got_l, got_n)
                seen[(name, mode.mxcsr(), idx)] = got_n
                wildcards += sum(1 for byte in wild if byte) // 4
                decided += 8 - sum(1 for byte in wild if byte) // 4
    # The control word must change results: each mode differs from the default on some call.
    default = ref.DEFAULT_MODE.mxcsr()
    for mode in (ref.Mode(ref.RD), ref.Mode(ref.RU), ref.Mode(ref.RZ), ref.Mode(ref.RN, True, False),
                 ref.Mode(ref.RN, False, True)):
        moved = [key for key, got in seen.items() if key[1] == mode.mxcsr() and got != seen[(key[0], default, key[2])]]
        if not moved:
            raise AssertionError(f"MXCSR {mode.label()} did not change any result: the harness is not applying it")
    # Wrong reference rules must be caught, so the inputs reach the rule being checked.
    for quirk, name, mode in REFERENCE_QUIRKS:
        caught = 0
        for idx, (a, b) in enumerate(pairs):
            want, wild = arith_reference(name, a, b, mode, quirk=quirk)
            got = seen[(name, mode.mxcsr(), idx)]
            if any((g ^ w) & ~m & 0xFF for g, w, m in zip(got, want, wild)):
                caught += 1
        if not caught:
            raise AssertionError(f"reference quirk {quirk} on {name} [{mode.label()}] was never detected")
    # The driver restores the caller's control word.
    for name in ("op_addps", "op_divps", "op_sqrtps"):
        for idx, (a, b) in enumerate(pairs):
            invoke_under(driver, native[name], ref.Mode(ref.RZ, True, True).mxcsr(), a, b)
            want, wild = arith_reference(name, a, b)
            assert_equal_output("restored", name, idx, invoke(native[name], a, b), want, wild)
            assert_equal_output("restored lifted", name, idx, invoke(lifted[name], a, b), want, wild)
    states = len({key[1] for key in seen})
    print(f"{len(seen)} packed arithmetic runs in {states} MXCSR states: native, lifted and reference agree on "
          f"{decided} lanes ({wildcards} lanes left undecided by the reference, native equals lifted there); "
          f"each non-default state moved a result; {len(REFERENCE_QUIRKS)} wrong reference rules detected; "
          f"driver restored the control word")


# --- Alternative-encoding acceptance & translator rejection controls ---

def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_alt_encodings_and_rejections(original_path):
    """Verify alternate encodings lift cleanly and all out-of-scope/illegal forms reject."""
    ld_xmm0 = Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_xmm1 = Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX))
    ld_xmm4 = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM4, MemoryOperand(R.RCX))
    ld_xmm5 = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM5, MemoryOperand(R.RDX, displ=3, displ_size=1))
    st_xmm0 = Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)
    ret_ins = Instruction.create(Code.RETNQ)

    # Encodings MASM does not emit directly: XMM4/XMM5, negative displacements,
    # reg-reg store opcode (0F 29 / 66 0F 7F / F3 0F 7F), rcx/r8 reads, etc.
    alt_accept = [
        ("cmpps_xmm4_xmm5", (ld_xmm4, ld_xmm5, Instruction.create_reg_reg_u32(
            Code.CMPPS_XMM_XMMM128_IMM8, R.XMM4, R.XMM5, 6),
            Instruction.create_mem_reg(Code.MOVNTPS_M128_XMM, MemoryOperand(R.R8), R.XMM4), ret_ins)),
        ("cmpps_neg_disp", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPPS_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RDX, displ=-16, displ_size=1), 3),
            st_xmm0, ret_ins)),
        ("cmpss_xmm4_rcx_plus3", (ld_xmm4, Instruction.create_reg_mem_u32(
            Code.CMPSS_XMM_XMMM32_IMM8, R.XMM4, MemoryOperand(R.RCX, displ=3, displ_size=1), 4),
            Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM4), ret_ins)),
        ("movlps_read_r8_disp", (ld_xmm0, Instruction.create_reg_mem(
            Code.MOVLPS_XMM_M64, R.XMM0, MemoryOperand(R.R8, displ=0x10, displ_size=1)),
            st_xmm0, ret_ins)),
        ("movhps_store_disp", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVHPS_M64_XMM, MemoryOperand(R.R8, displ=0x18, displ_size=1), R.XMM0), ret_ins)),
        ("movaps_alt_reg_reg", (ld_xmm0, Instruction.create_reg_reg(
            Code.MOVAPS_XMMM128_XMM, R.XMM2, R.XMM0),
            Instruction.create_mem_reg(Code.MOVNTPS_M128_XMM, MemoryOperand(R.R8), R.XMM2), ret_ins)),
        ("movdqa_alt_reg_reg", (ld_xmm0, Instruction.create_reg_reg(
            Code.MOVDQA_XMMM128_XMM, R.XMM3, R.XMM0),
            Instruction.create_mem_reg(Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.R8, displ=1, displ_size=1), R.XMM3),
            ret_ins)),
        ("addps_xmm4_xmm5", (ld_xmm4, ld_xmm5, Instruction.create_reg_reg(
            Code.ADDPS_XMM_XMMM128, R.XMM4, R.XMM5),
            Instruction.create_mem_reg(Code.MOVNTPS_M128_XMM, MemoryOperand(R.R8), R.XMM4), ret_ins)),
        ("divps_neg_disp", (ld_xmm0, Instruction.create_reg_mem(
            Code.DIVPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX, displ=-32, displ_size=1)), st_xmm0, ret_ins)),
        ("maxps_r8_source", (ld_xmm0, Instruction.create_reg_mem(
            Code.MAXPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.R8, displ=0x10, displ_size=1)), st_xmm0, ret_ins)),
        ("sqrtps_undefined_dest", (ld_xmm1, Instruction.create_reg_reg(
            Code.SQRTPS_XMM_XMMM128, R.XMM5, R.XMM1),
            Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM5), ret_ins)),
        ("sqrtps_mem_xmm4", (Instruction.create_reg_mem(
            Code.SQRTPS_XMM_XMMM128, R.XMM4, MemoryOperand(R.RCX, displ=0x20, displ_size=1)),
            Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM4), ret_ins)),
    ]
    alt_expected = {
        "cmpps_xmm4_xmm5": "xmm4 = op_cmpps(xmm4, xmm5, 0x6u);",
        "cmpps_neg_disp": "load128((rdx + 0xfffffffffffffff0ull))",
        "cmpss_xmm4_rcx_plus3": "set_lane32(xmm4, 0u, op_cmpss_d(lane32(xmm4, 0u), load32((rcx + 0x3ull)), 0x4u));",
        "movlps_read_r8_disp": "xmm0.q[0] = load64((r8 + 0x10ull));",
        "movhps_store_disp": "store64((r8 + 0x18ull), xmm0.q[1]);",
        "movaps_alt_reg_reg": "xmm2 = xmm0;",
        "movdqa_alt_reg_reg": "store128((r8 + 0x1ull), xmm3);",
        "addps_xmm4_xmm5": "xmm4 = op_addps(xmm4, xmm5);",
        "divps_neg_disp": "xmm0 = op_divps(xmm0, load128((rdx + 0xffffffffffffffe0ull)));",
        "maxps_r8_source": "xmm0 = op_maxps(xmm0, load128((r8 + 0x10ull)));",
        "sqrtps_undefined_dest": "xmm5 = op_sqrtps(xmm1);",
        "sqrtps_mem_xmm4": "xmm4 = op_sqrtps(load128((rcx + 0x20ull)));",
    }
    for label, seq in alt_accept:
        raw = encode_seq(*seq)
        decoded = list(Decoder(64, raw, ip=0x1000))
        assert len(decoded) == len(seq) and not any(ins.is_invalid for ins in decoded), label
        src, count = lift_code(label, 0x1000, raw)
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if alt_expected[label] not in src:
            raise AssertionError(
                f"alternate encoding {label} lowered without expected {alt_expected[label]!r}")

    # 1) Non-legacy VEX and EVEX encodings must reject.
    nonlegacy_cases = [
        ("vcmpps_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VCMPPS_XMM_XMM_XMMM128_IMM8, R.XMM2, R.XMM0, R.XMM1, 0), ret_ins), "non-legacy encoding"),
        ("vcmpss_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VCMPSS_XMM_XMM_XMMM32_IMM8, R.XMM2, R.XMM0, R.XMM1, 0), ret_ins), "non-legacy encoding"),
        ("vmovlps_vex", (ld_xmm0, Instruction.create_reg_reg_mem(
            Code.VEX_VMOVLPS_XMM_XMM_M64, R.XMM2, R.XMM0, MemoryOperand(R.RDX)), ret_ins), "non-legacy encoding"),
        ("vmovhps_vex", (ld_xmm0, Instruction.create_mem_reg(
            Code.VEX_VMOVHPS_M64_XMM, MemoryOperand(R.R8), R.XMM0), ret_ins), "non-legacy encoding"),
        ("vmovntps_vex", (ld_xmm0, Instruction.create_mem_reg(
            Code.VEX_VMOVNTPS_M128_XMM, MemoryOperand(R.R8), R.XMM0), ret_ins), "non-legacy encoding"),
        ("vmovaps_vex", (Instruction.create_reg_mem(
            Code.VEX_VMOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret_ins), "non-legacy encoding"),
        ("vcmpps_evex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg_u32(
            Code.EVEX_VCMPPS_KR_K1_XMM_XMMM128B32_IMM8, R.K1, R.XMM0, R.XMM1, 0), ret_ins), "non-legacy encoding"),
        ("vmovntps_evex", (ld_xmm0, Instruction.create_mem_reg(
            Code.EVEX_VMOVNTPS_M128_XMM, MemoryOperand(R.R8), R.XMM0), ret_ins), "non-legacy encoding"),
        ("vmovaps_evex", (Instruction.create_reg_mem(
            Code.EVEX_VMOVAPS_XMM_K1Z_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret_ins), "non-legacy encoding"),
        ("vaddps_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VADDPS_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
        ("vminps_vex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VMINPS_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
        ("vsqrtps_vex", (ld_xmm1, Instruction.create_reg_reg(
            Code.VEX_VSQRTPS_XMM_XMMM128, R.XMM2, R.XMM1), ret_ins), "non-legacy encoding"),
        ("vaddps_evex", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg(
            Code.EVEX_VADDPS_XMM_K1Z_XMM_XMMM128B32, R.XMM2, R.XMM0, R.XMM1), ret_ins), "non-legacy encoding"),
    ]

    # 2) Reserved immediate predicates >= 8 reject.
    imm_cases = [
        ("cmpps_imm_8", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 8), ret_ins), "unsupported cmpps predicate"),
        ("cmpps_imm_ff", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0xFF), ret_ins), "unsupported cmpps predicate"),
        ("cmpss_imm_8", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPSS_XMM_XMMM32_IMM8, R.XMM0, R.XMM1, 8), ret_ins), "unsupported cmpss predicate"),
        ("cmpss_imm_1f", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPSS_XMM_XMMM32_IMM8, R.XMM0, R.XMM1, 0x1F), ret_ins), "unsupported cmpss predicate"),
    ]

    # 3) Unsupported instructions (other SSE/SSE2/GPR ops).
    unsupported_insns = [
        ("rsqrtps", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.RSQRTPS_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins)),
        ("rcpps", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.RCPPS_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins)),
        ("haddpd", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.HADDPD_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins)),
        ("cvtps2pd", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.CVTPS2PD_XMM_XMMM64, R.XMM0, R.XMM1), ret_ins)),
        ("cvtdq2pd", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.CVTDQ2PD_XMM_XMMM64, R.XMM0, R.XMM1), ret_ins)),
        ("sqrtss", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.SQRTSS_XMM_XMMM32, R.XMM0, R.XMM1), ret_ins)),
        ("maxss", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.MAXSS_XMM_XMMM32, R.XMM0, R.XMM1), ret_ins)),
        ("addss", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.ADDSS_XMM_XMMM32, R.XMM0, R.XMM1), ret_ins)),
        ("cvtpd2ps", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.CVTPD2PS_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins)),
        ("unpcklpd", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.UNPCKLPD_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins)),
        ("unpckhpd", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.UNPCKHPD_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins)),
        ("shufpd", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_u32(Code.SHUFPD_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0), ret_ins)),
        ("movupd", (Instruction.create_reg_mem(Code.MOVUPD_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret_ins)),
        ("movhlps", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.MOVHLPS_XMM_XMM, R.XMM0, R.XMM1), ret_ins)),
        ("movlhps", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.MOVLHPS_XMM_XMM, R.XMM0, R.XMM1), ret_ins)),
        ("movss", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.MOVSS_XMM_XMMM32, R.XMM0, R.XMM1), ret_ins)),
        ("movups", (Instruction.create_reg_mem(Code.MOVUPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret_ins)),
        ("shufps", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_u32(Code.SHUFPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0), ret_ins)),
        ("cvtpi2ps", (ld_xmm0, Instruction.create_reg_mem(Code.CVTPI2PS_XMM_MMM64, R.XMM0, MemoryOperand(R.RDX)), ret_ins)),
        ("ldmxcsr", (Instruction.create_mem(Code.LDMXCSR_M32, MemoryOperand(R.RCX)), ret_ins)),
        ("mov_gpr", (Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RBX, R.RCX), ret_ins)),
    ]

    # 4) Register policy, alignment, store-base, addressing, prefix, and stream checks.
    seg_load = Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    seg_load.segment_prefix = R.GS
    other_cases = [
        ("nonvolatile_xmm6_dst", (Instruction.create_reg_mem(
            Code.MOVAPS_XMM_XMMM128, R.XMM6, MemoryOperand(R.RCX)), ret_ins), "nonvolatile vector register"),
        ("nonvolatile_xmm15_src", (ld_xmm0, Instruction.create_reg_reg_u32(
            Code.CMPPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM15, 0), ret_ins), "nonvolatile vector register"),
        ("nonvolatile_xmm8_movntps", (Instruction.create_mem_reg(
            Code.MOVNTPS_M128_XMM, MemoryOperand(R.R8), R.XMM8), ret_ins), "nonvolatile vector register"),
        ("undefined_cmpps_dst", (ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0), ret_ins), "undefined vector register xmm0"),
        ("undefined_cmpps_src", (ld_xmm0, Instruction.create_reg_reg_u32(
            Code.CMPPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM3, 0), ret_ins), "undefined vector register xmm3"),
        ("undefined_cmpss_dst", (ld_xmm1, Instruction.create_reg_reg_u32(
            Code.CMPSS_XMM_XMMM32_IMM8, R.XMM0, R.XMM1, 0), ret_ins), "undefined vector register xmm0"),
        ("undefined_movlps_dst", (Instruction.create_reg_mem(
            Code.MOVLPS_XMM_M64, R.XMM0, MemoryOperand(R.RDX)), ret_ins), "undefined vector register xmm0"),
        ("undefined_movhps_dst", (Instruction.create_reg_mem(
            Code.MOVHPS_XMM_M64, R.XMM2, MemoryOperand(R.RDX)), ret_ins), "undefined vector register xmm2"),
        ("undefined_movntps_src", (Instruction.create_mem_reg(
            Code.MOVNTPS_M128_XMM, MemoryOperand(R.R8), R.XMM4), ret_ins), "undefined vector register xmm4"),
        ("non_r8_store_movntps_rcx", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVNTPS_M128_XMM, MemoryOperand(R.RCX), R.XMM0), ret_ins), "non-r8 store memory operand"),
        ("non_r8_store_movlps_rdx", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVLPS_M64_XMM, MemoryOperand(R.RDX), R.XMM0), ret_ins), "non-r8 store memory operand"),
        ("non_r8_store_movhps_rcx", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVHPS_M64_XMM, MemoryOperand(R.RCX, displ=8, displ_size=1), R.XMM0), ret_ins),
            "non-r8 store memory operand"),
        ("non_r8_store_movaps_rdx", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.RDX), R.XMM0), ret_ins), "non-r8 store memory operand"),
        ("unaligned_cmpps_m128", (ld_xmm0, Instruction.create_reg_mem_u32(
            Code.CMPPS_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RDX, displ=8, displ_size=1), 0), ret_ins),
            "unaligned 16-byte memory operand"),
        ("unaligned_movntps_m128", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVNTPS_M128_XMM, MemoryOperand(R.R8, displ=4, displ_size=1), R.XMM0), ret_ins),
            "unaligned 16-byte memory operand"),
        ("unaligned_movaps_m128", (Instruction.create_reg_mem(
            Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=1, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("nonvolatile_gpr_base_rbx", (Instruction.create_reg_mem(
            Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RBX)), ret_ins), "nonvolatile GPR base"),
        ("nonvolatile_gpr_base_r12", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVNTPS_M128_XMM, MemoryOperand(R.R12), R.XMM0), ret_ins), "nonvolatile GPR base"),
        ("non_arg_base_rax", (Instruction.create_reg_mem(
            Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RAX)), ret_ins), "unsupported memory base"),
        ("non_arg_base_r9", (ld_xmm0, Instruction.create_reg_mem(
            Code.MOVLPS_XMM_M64, R.XMM0, MemoryOperand(R.R9)), ret_ins), "unsupported memory base"),
        ("rip_rel_mem", (Instruction.create_reg_mem(
            Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RIP, displ=0x1010, displ_size=1)), ret_ins),
            "unsupported memory operand"),
        ("rsp_mem", (Instruction.create_reg_mem(
            Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RSP)), ret_ins), "unsupported memory operand"),
        ("addr32_mem", (Instruction.create_reg_mem(
            Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.ECX)), ret_ins), "unsupported memory operand"),
        ("indexed_mem", (Instruction.create_reg_mem(
            Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, R.RDX, 4)), ret_ins),
            "unsupported memory operand"),
        ("nonvolatile_xmm7_addps_dst", (Instruction.create_reg_reg(
            Code.ADDPS_XMM_XMMM128, R.XMM7, R.XMM1), ret_ins), "nonvolatile vector register"),
        ("nonvolatile_xmm9_sqrtps_dst", (ld_xmm1, Instruction.create_reg_reg(
            Code.SQRTPS_XMM_XMMM128, R.XMM9, R.XMM1), ret_ins), "nonvolatile vector register"),
        ("undefined_addps_dst", (ld_xmm1, Instruction.create_reg_reg(
            Code.ADDPS_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins), "undefined vector register xmm0"),
        ("undefined_mulps_src", (ld_xmm0, Instruction.create_reg_reg(
            Code.MULPS_XMM_XMMM128, R.XMM0, R.XMM3), ret_ins), "undefined vector register xmm3"),
        ("undefined_minps_src", (ld_xmm0, Instruction.create_reg_reg(
            Code.MINPS_XMM_XMMM128, R.XMM0, R.XMM2), ret_ins), "undefined vector register xmm2"),
        ("undefined_sqrtps_src", (Instruction.create_reg_reg(
            Code.SQRTPS_XMM_XMMM128, R.XMM0, R.XMM2), ret_ins), "undefined vector register xmm2"),
        ("unaligned_divps_m128", (ld_xmm0, Instruction.create_reg_mem(
            Code.DIVPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX, displ=8, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("unaligned_subps_m128", (ld_xmm0, Instruction.create_reg_mem(
            Code.SUBPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=-4, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("unaligned_sqrtps_m128", (Instruction.create_reg_mem(
            Code.SQRTPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, displ=4, displ_size=1)), ret_ins),
            "unaligned 16-byte memory operand"),
        ("non_arg_base_maxps", (ld_xmm0, Instruction.create_reg_mem(
            Code.MAXPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RAX)), ret_ins), "unsupported memory base"),
        ("indexed_mulps", (ld_xmm0, Instruction.create_reg_mem(
            Code.MULPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX, R.RDX, 4)), ret_ins),
            "unsupported memory operand"),
        ("nonvolatile_base_sqrtps", (Instruction.create_reg_mem(
            Code.SQRTPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RBX)), ret_ins), "nonvolatile GPR base"),
        ("segment_prefix", (seg_load, ret_ins), "unsupported instruction prefix"),
        ("missing_ret", (ld_xmm0,), "missing terminal ret"),
        ("after_ret", (ret_ins, ld_xmm0), "unreachable instruction after ret"),
    ]

    raw_cases = [
        ("lock_cmpxchg", bytes.fromhex("f00fb109") + encode_seq(ret_ins), "unsupported instruction prefix"),
        ("rep_movsb", bytes.fromhex("f3a4") + encode_seq(ret_ins), "unsupported instruction prefix"),
        ("invalid_ud_bytes", bytes.fromhex("fffe") + encode_seq(ret_ins), "invalid instruction"),
        ("truncated_insn", encode_seq(ld_xmm0)[:-1], "invalid instruction"),
        ("empty_stream", b"", "truncated instruction stream"),
    ]

    all_unit = (
        nonlegacy_cases
        + imm_cases
        + [(lbl, seq, "unsupported instruction") for lbl, seq in unsupported_insns]
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
        ("pe_vcmpps_vex", encode_seq(*nonlegacy_cases[0][1]), "non-legacy encoding"),
        ("pe_cmpps_imm8", encode_seq(*imm_cases[0][1]), "unsupported cmpps predicate"),
        ("pe_cmpss_imm8", encode_seq(*imm_cases[2][1]), "unsupported cmpss predicate"),
        ("pe_rsqrtps", encode_seq(*unsupported_insns[0][1]), "unsupported instruction"),
        ("pe_nonvolatile_xmm6", encode_seq(*other_sequences["nonvolatile_xmm6_dst"]), "nonvolatile vector register"),
        ("pe_undef_movlps", encode_seq(*other_sequences["undefined_movlps_dst"]), "undefined vector register xmm0"),
        ("pe_non_r8_movntps", encode_seq(*other_sequences["non_r8_store_movntps_rcx"]), "non-r8 store memory operand"),
        ("pe_unaligned_movntps", encode_seq(*other_sequences["unaligned_movntps_m128"]), "unaligned 16-byte memory operand"),
        ("pe_nonvolatile_rbx", encode_seq(*other_sequences["nonvolatile_gpr_base_rbx"]), "nonvolatile GPR base"),
        ("pe_missing_ret", encode_seq(ld_xmm0), "missing terminal ret"),
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

    unit_total = len(all_unit) + len(raw_cases)
    print(f"{len(alt_accept)} alternate-encoding lift checks, {unit_total} unit rejection controls "
          f"({len(nonlegacy_cases)} VEX/EVEX, {len(imm_cases)} predicate imm8, "
          f"{len(unsupported_insns)} unsupported instructions, {len(other_cases) + len(raw_cases)} "
          f"ABI/state/addressing/prefix/stream) and {len(pe_patches)} PE-patched rejection controls passed")


if __name__ == "__main__":
    check_discarded_read(*sys.argv[1:3])
    check_differential(*sys.argv[1:4])
    check_arithmetic_modes(*sys.argv[1:3], sys.argv[4])
    check_alt_encodings_and_rejections(sys.argv[1])
