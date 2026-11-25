"""Differential checks and rejection controls for recompiled scalar SSE/SSE2 leaf functions."""

import ctypes
from fractions import Fraction
import math
from pathlib import Path
import random
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Decoder, Instruction, Mnemonic as M, OpKind, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from cpu_sse_aot import ScalarLifter, translate

SIGNATURES = {"blend_ss": "f:ffqf", "blend_sd": "d:ddqd", "convert": "q:fdq", "from_int": "d:qq",
              "classify_ss": "q:ff", "classify_sd": "q:dq", "round_trip": "f:f",
              "zero_xorps": "d:d", "zero_pxor": "d:d", "zero_xorpd": "d:d", "probe_ss": "q:qff",
              "neg_jcc": "q:q", "setcc_mix": "q:qqq", "cmovcc_mix": "q:qqq",
              "min_ss": "f:ffq", "max_ss": "f:ffq", "min_sd": "d:ddq", "max_sd": "d:ddq",
              "root_ss": "f:fq", "root_sd": "d:dq", "root_const_ss": "f:", "root_const_sd": "d:",
              "round_cvt": "q:fdq", "movq_gpr": "q:dq"}
# MIN, MAX and SQRT: the seed behind the pointer and how far each export writes into it.
SEED = struct.pack("<II", 0x0badf00d, 0x1badcafe)
EXTREMA = {"min_ss": ("min", 4), "max_ss": ("max", 4), "min_sd": ("min", 8), "max_sd": ("max", 8)}
ROOTS = {"root_ss": 4, "root_sd": 8}
# The host MXCSR words of the MIN, MAX and SQRT checks: the default, each rounding control (nearest, down, up,
# toward zero) and DAZ and FTZ. A native call and its translation run under the same word.
DEFAULT_MXCSR, DAZ, FTZ = 0x1f80, 0x40, 0x8000
ROUNDING_WORDS = [DEFAULT_MXCSR, 0x3f80, 0x5f80, 0x7f80]
EXTREMA_WORDS = [DEFAULT_MXCSR | DAZ, DEFAULT_MXCSR | DAZ | FTZ, DEFAULT_MXCSR | FTZ]
ROOT_WORDS = ROUNDING_WORDS + [DEFAULT_MXCSR | DAZ, 0x5f80 | DAZ, 0x3f80 | DAZ | FTZ, DEFAULT_MXCSR | FTZ, 0x5f80 | FTZ]
# Results fixed from the instruction set reference, (MXCSR word, export, operands, result bits). DAZ reads a
# denormal operand as a zero of its sign, so the root of a negative denormal is -0 and not the default NaN, and a
# denormal MIN or MAX result leaves as a signed zero. FTZ alone leaves the operands as they are.
DAZ_WORDS = [DEFAULT_MXCSR | DAZ, DEFAULT_MXCSR | DAZ | FTZ]
DENORMAL_ANSWERS = [
    (word, name, (operand, None), result) for word in DAZ_WORDS for name, operand, result in (
        ("root_ss", 0x807fffff, 0x80000000), ("root_ss", 0x80000001, 0x80000000), ("root_ss", 0x00000001, 0),
        ("root_ss", 0x007fffff, 0), ("root_sd", 0x800fffffffffffff, 1 << 63), ("root_sd", 0x8000000000000001, 1 << 63),
        ("root_sd", 1, 0), ("root_sd", 0x000fffffffffffff, 0))
] + [
    (word, name, (a, b, None), result) for word in DAZ_WORDS for name, a, b, result in (
        ("min_ss", 0x807fffff, 0x3f800000, 0x80000000), ("min_ss", 0x00000001, 0x3f800000, 0),
        ("max_ss", 0x807fffff, 0xbf800000, 0x80000000), ("max_ss", 0x00000001, 0xbf800000, 0),
        ("min_sd", 0x800fffffffffffff, 0x3ff0000000000000, 1 << 63), ("min_sd", 1, 0x3ff0000000000000, 0),
        ("max_sd", 0x800fffffffffffff, 0xbff0000000000000, 1 << 63), ("max_sd", 1, 0xbff0000000000000, 0))
] + [
    (DEFAULT_MXCSR | FTZ, name, values, result) for name, values, result in (
        ("root_ss", (0x807fffff, None), 0xffc00000), ("root_sd", (0x800fffffffffffff, None), 0xfff8000000000000),
        ("min_ss", (0x807fffff, 0x3f800000, None), 0x807fffff), ("max_sd", (1, 0xbff0000000000000, None), 1))
]
# The operands of the constant-source exports, as an integer built into the code: sqrt(2) as a float and
# sqrt(3) as a double.
CONSTANT_ROOTS = {"root_const_ss": (2, 4), "root_const_sd": (3, 8)}
WINDOWS = {"min_ss": 40, "max_ss": 40, "min_sd": 48, "max_sd": 48, "root_ss": 40, "root_sd": 48, "round_cvt": 80}
GUARD = bytes.fromhex("5a5a1234a5a54321")
FILL = 0xa5
MASK32 = 0xffffffff
# A float argument occupies the low dword of its XMM register. The rest is not part of the value.
JUNK = 0xdeadbeef << 32

SINGLES = [
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 0x3fc00000, 0x40200000, 0xc0200000,
    0x3e99999a, 0x3f7fffff, 0xbf7fffff, 0x33800000, 0x4b800000, 0x4b800001,
    0x00000001, 0x807fffff, 0x00800000, 0x7f7fffff, 0xff7fffff,          # denormals, smallest normal, largest
    0x7f800000, 0xff800000,                                              # infinities
    0x7fc00000, 0xffc00000, 0x7fc12345, 0xffe54321,                      # quiet NaNs
    0x7f800001, 0xffa00000, 0x7fbfffff,                                  # signalling NaNs
    0x4f000000, 0x4effffff, 0xcf000000, 0xcf000001,                      # around +-2^31
    0x5f000000, 0x5effffff, 0xdf000000, 0xdf000001,                      # around +-2^63
]
DOUBLES = [
    0x0000000000000000, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000, 0x3fe0000000000000,
    0x3ff8000000000000, 0x4004000000000000, 0xc004000000000000, 0x3fb999999999999a, 0x3fefffffffffffff,
    0x0000000000000001, 0x800fffffffffffff, 0x0010000000000000, 0x7fefffffffffffff, 0xffefffffffffffff,
    0x7ff0000000000000, 0xfff0000000000000,
    0x7ff8000000000000, 0xfff8000000000000, 0x7ff8000012345678, 0xfffc0000deadbeef,
    0x7ff0000000000001, 0xfff4000000000000, 0x7ff7ffffffffffff,
    0x41e0000000000000, 0x41dfffffffe00000, 0xc1e0000000000000, 0xc1e0000000100000, 0xc1e0000000200000,
    0x43e0000000000000, 0x43dfffffffffffff, 0xc3e0000000000000, 0xc3e0000000000001,
    0x47efffffefffffff, 0x47effffff0000000, 0xc7effffff0000000,          # rounds to FLT_MAX, to infinity
    0x36a0000000000000, 0x3690000000000000, 0x380fffffffffffff,          # float denormal range and below
    0x3ff0000010000000, 0x3ff0000030000000,                              # ties when narrowed to float
]
INTEGERS = [
    0, 1, 2, 0x7fffffff, 0x80000000, 0xffffffff, 0x100000000, 0x1000001, 0x1000003, 0xfffffffffeffffff,
    (1 << 53) + 1, (1 << 53) + 3, (1 << 63) - 1, 1 << 63, (1 << 64) - 1, 0x7fffff4000000000,
    0x7fffffbfffffffff, 0x7ffffffffffffbff, 0x0123456789abcdef, 0xfedcba9876543210, 0x123456780fffffff,
]
_rng = random.Random(2027)
SINGLES += [_rng.getrandbits(32) for _ in range(12)]
DOUBLES += [_rng.getrandbits(64) for _ in range(12)]
# Random bit patterns rarely land in a convertible range, so add values of moderate size too.
SINGLES += [struct.unpack("<I", struct.pack("<f", _rng.uniform(-3e9, 3e9)))[0] for _ in range(6)]
DOUBLES += [struct.unpack("<Q", struct.pack("<d", _rng.uniform(-3e19, 3e19)))[0] for _ in range(6)]
INTEGERS += [_rng.getrandbits(64) for _ in range(40)]


def is_nan(bits, width):
    exponent, fraction = (0xff, 23) if width == 4 else (0x7ff, 52)
    return bits >> fraction & exponent == exponent and bits & ((1 << fraction) - 1) != 0


def number(bits, width):
    return struct.unpack("<f" if width == 4 else "<d", struct.pack("<I" if width == 4 else "<Q", bits))[0]


def exact_root(bits, width, rounding):
    """The square root of a positive finite number, by integer arithmetic, rounded as the RC field says:
    0 to nearest even, 1 down, 2 up, 3 toward zero."""
    precision, fraction_bits, exponent_bits, bias = (24, 23, 8, 127) if width == 4 else (53, 52, 11, 1023)
    exponent = bits >> fraction_bits & (1 << exponent_bits) - 1
    fraction = bits & (1 << fraction_bits) - 1
    mantissa, power = (fraction | 1 << fraction_bits, exponent - bias - fraction_bits) if exponent \
        else (fraction, 1 - bias - fraction_bits)
    if power & 1:
        mantissa, power = mantissa << 1, power - 1
    scale = 2 * (precision + 2)
    root = math.isqrt(mantissa << scale)
    exact = root * root == mantissa << scale
    drop = root.bit_length() - precision
    kept, rest = root >> drop, root & (1 << drop) - 1
    half = 1 << drop - 1
    if rounding == 0:
        kept += rest > half or (rest == half and (not exact or kept & 1) != 0)
    elif rounding == 2:
        kept += rest != 0 or not exact
    value = math.ldexp(kept, drop + power // 2 - scale // 2)
    return struct.unpack("<I" if width == 4 else "<Q", struct.pack("<f" if width == 4 else "<d", value))[0]


def reference(name, values, payload, variant=None, mxcsr=DEFAULT_MXCSR):
    """Result bits and the whole buffer for MIN, MAX and SQRT, from the instruction set reference alone:
    exact Python arithmetic, no host instruction and no code of the translator. The variant, when given, is a
    deliberately wrong rule that the negative controls must see fail. The MXCSR word gives the rounding of a
    root and DAZ, which reads a denormal source operand as a zero of its sign, also when MIN or MAX returns it."""
    def flush(bits, width):
        sign = 1 << 8 * width - 1
        magnitude = bits & ~sign
        return bits & sign if mxcsr & DAZ and 0 < magnitude < 1 << (23 if width == 4 else 52) else bits

    def extreme(op, a, b, width):
        x, y = number(flush(a, width), width), number(flush(b, width), width)
        keep = x < y if op == "min" else x > y
        if variant == "tie-keeps-destination":
            keep = x <= y if op == "min" else x >= y
        elif variant == "nan-keeps-destination":
            keep = not (x >= y if op == "min" else x <= y)
        elif variant == "min-is-max":
            keep = x > y if op == "min" else x < y
        chosen = a if keep else b
        return chosen if variant == "daz-result-kept" else flush(chosen, width)

    def root(bits, width):
        quiet = 1 << (22 if width == 4 else 51)
        indefinite = 0xffc00000 if width == 4 else 0xfff8000000000000
        raw, bits = bits, flush(bits, width)
        x = number(bits, width)
        if variant == "library-sign-test" and raw != bits and raw >> 8 * width - 1:
            return indefinite  # a runtime that tests the sign of the bits before the processor reads the operand
        if x != x:
            return bits if variant == "snan-not-quieted" else bits | quiet
        if x < 0:
            return indefinite & ~(1 << (31 if width == 4 else 63)) if variant == "negative-positive-nan" else indefinite
        if x == 0:
            return 0 if variant == "negative-zero-positive" else bits
        if x == math.inf:
            return bits
        return exact_root(bits, width, 0 if variant == "nearest-only" else mxcsr >> 13 & 3)

    if name in EXTREMA:
        op, width = EXTREMA[name]
        mask = (1 << 8 * width) - 1
        operands = (values[0] & mask, values[1] & mask)
        result = extreme(op, *operands, width)
    else:
        width = ROOTS[name]
        operands = (values[0] & ((1 << 8 * width) - 1),)
        result = root(operands[0], width)
    return result, buffer(name, operands, payload, result, variant == "root-clears-upper")


def buffer(name, operands, payload, result, clear_upper=False):
    """The buffer after one call. The register form lands at +8 and the memory form after the operand copies.
    A single result replaces the low dword and keeps the dword above it from the seed."""
    width = EXTREMA[name][1] if name in EXTREMA else ROOTS[name]
    pack = "<I" if width == 4 else "<Q"
    memory = bytearray(payload)
    if width == 4:
        wide = struct.pack("<II", result, 0 if clear_upper else struct.unpack_from("<I", payload, 4)[0])
    else:
        wide = struct.pack("<Q", result)
    for number_, operand in enumerate(operands):
        memory[17 + number_ * width:17 + (number_ + 1) * width] = struct.pack(pack, operand)
    second = (28 if name in EXTREMA else 24) if width == 4 else (36 if name in EXTREMA else 32)
    memory[8:16] = wide
    memory[second:second + 8] = wide
    return bytes(memory)


# Each wrong rule and the export it breaks. The test needs every one to disagree with the processor on some call.
MUTANTS = [("min_ss", "tie-keeps-destination"), ("max_sd", "tie-keeps-destination"),
           ("max_ss", "nan-keeps-destination"), ("min_sd", "nan-keeps-destination"),
           ("min_sd", "min-is-max"), ("root_ss", "snan-not-quieted"), ("root_sd", "snan-not-quieted"),
           ("root_ss", "negative-positive-nan"), ("root_sd", "negative-positive-nan"),
           ("root_sd", "negative-zero-positive"), ("root_ss", "root-clears-upper")]


def cases():
    """(export, slot values with None at the pointer slot, initial bytes behind the pointer)."""
    found = []
    for i, a in enumerate(SINGLES):
        for j, b in enumerate(SINGLES):
            c = SINGLES[(5 * i + 3 * j) % len(SINGLES)]
            d = SINGLES[(i + 7 * j + 2) % len(SINGLES)]
            found.append(("blend_ss", (a, b, None, d), struct.pack("<II", c, 0x0badf00d)))
            found.append(("classify_ss", (a, b), b""))
    for i, a in enumerate(DOUBLES):
        for j, b in enumerate(DOUBLES):
            c = DOUBLES[(5 * i + 3 * j) % len(DOUBLES)]
            d = DOUBLES[(i + 7 * j + 2) % len(DOUBLES)]
            found.append(("blend_sd", (a, b, None, d), struct.pack("<QQ", c, 0x0badf00d0badf00d)))
            found.append(("classify_sd", (a, None), struct.pack("<Q", b)))
    for i, f in enumerate(SINGLES):
        for step in (0, 1, 5, 11, 17, 23, 29, 31):
            found.append(("convert", (f, DOUBLES[(i + step * (i + 1)) % len(DOUBLES)], None), bytes([FILL]) * 80))
    for name in ("zero_xorps", "zero_pxor", "zero_xorpd"):
        found.extend((name, (value,), b"") for value in DOUBLES)
    found += [("from_int", (n, None), bytes([FILL]) * 32) for n in INTEGERS]
    found += [("round_trip", (f,), b"") for f in SINGLES]
    for name, (op, width) in EXTREMA.items():
        pool = SINGLES if width == 4 else DOUBLES
        for a in pool:
            for b in pool:
                found.append((name, (a, b, None), SEED + bytes([FILL]) * (WINDOWS[name] - 8)))
    for name, width in ROOTS.items():
        found += [(name, (a, None), SEED + bytes([FILL]) * (WINDOWS[name] - 8)) for a in (SINGLES if width == 4 else DOUBLES)]
    found += [("root_const_ss", (), b""), ("root_const_sd", (), b"")]
    found += [("round_cvt", (f, d, None), bytes([FILL]) * WINDOWS["round_cvt"]) for f in ROUND_SINGLES for d in ROUND_DOUBLES]
    name, values, payload, _, _ = probe_answer(2.0, 1.0)
    found.append((name, values, payload))
    flag_values = [0, 1, 2, 0x7fffffff, 0x80000000, 0xffffffff, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
    found += [("neg_jcc", (value,), b"") for value in flag_values]
    for a in flag_values:
        for b in flag_values:
            found.append(("setcc_mix", (a, b, None), bytes([FILL]) * 4))
            found.append(("cmovcc_mix", (a, b, None), struct.pack("<QQ", 0xdeadbeefcafebabe, 0xaaaa555511112222)))
    for d in DOUBLES:
        for q in INTEGERS:
            found.append(("movq_gpr", (d, q), b""))
    return found


def run(call, name, values, payload):
    """Call one export. Returns its result bits and the guarded bytes behind the pointer slot."""
    buffer = ctypes.create_string_buffer(GUARD + payload + GUARD, len(payload) + 16)
    kinds = SIGNATURES[name][2:]
    slots = [ctypes.addressof(buffer) + 8 if value is None else value | JUNK if kind == "f" else value
             for kind, value in zip(kinds, values)]
    return call(name, slots), buffer.raw


def mxcsr_access(harness):
    """The host MXCSR of the calling thread: (get, set) of the native-only harness."""
    library = ctypes.CDLL(str(Path(harness).resolve()))
    library.get_mxcsr.argtypes, library.get_mxcsr.restype = [], ctypes.c_uint32
    library.set_mxcsr.argtypes, library.set_mxcsr.restype = [ctypes.c_uint32], None
    return library.get_mxcsr, library.set_mxcsr


def under(access, word, call, name, values, payload):
    """run() with the host MXCSR set to a word for the call alone."""
    get, put = access
    saved = get()
    put(word)
    try:
        return run(call, name, values, payload)
    finally:
        put(saved)


def library_caller(path, prefix=""):
    """Floating-point slots travel as doubles, so every bit pattern reaches XMM unchanged."""
    library = ctypes.CDLL(str(path))
    functions = {}
    for name, signature in SIGNATURES.items():
        function = getattr(library, prefix + name)
        function.argtypes = [ctypes.c_uint64 if kind == "q" else ctypes.c_double for kind in signature[2:]]
        function.restype = ctypes.c_uint64 if signature[0] == "q" else ctypes.c_double
        functions[name] = function

    def call(name, slots):
        signature = SIGNATURES[name]
        arguments = [slot if kind == "q" else struct.unpack("<d", struct.pack("<Q", slot))[0]
                     for kind, slot in zip(signature[2:], slots)]
        result = functions[name](*arguments)
        if signature[0] == "q":
            return result
        bits = struct.unpack("<Q", struct.pack("<d", result))[0]
        return bits & MASK32 if signature[0] == "f" else bits
    return call


def single(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def double(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]



# CVTSS2SI and CVTSD2SI of round_cvt. Each word of the output, its source width and its integer size in bytes: the
# 32-bit forms start from all ones, so a result that keeps the upper half or extends the sign shows.
ROUND_WINDOW = 80
ROUND_LAYOUT = {0: (4, 4), 8: (8, 4), 16: (4, 8), 24: (8, 8), 32: (4, 4), 56: (8, 4), 64: (4, 8), 72: (8, 8)}
ROUND_32 = tuple(offset for offset, (_, size) in ROUND_LAYOUT.items() if size == 4)
ROUND_CONTROL_WORDS = [DEFAULT_MXCSR | control << 13 | daz | ftz for control in range(4) for daz in (0, DAZ) for ftz in (0, FTZ)]
INDEFINITE = {4: 1 << 31, 8: 1 << 63}
ROUND_SINGLES = [single(v) for v in (
    0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, 0.25, -0.75, 1.0, -1.0, 4.5, 5.5, 1 - 2.0**-24, 1 + 2.0**-23,
    1.5 - 2.0**-23, 1.5 + 2.0**-23, 2.0**23 - 0.5, 2.0**23, 3e9, -3e9, 1e20, 2.0**31 - 128, 2.0**31, -2.0**31,
    -2.0**31 - 256, 2.0**32, 2.0**63 - 2.0**39, 2.0**63, -2.0**63, -2.0**63 - 2.0**40)] + [
    0x00000001, 0x007fffff, 0x807fffff, 0x80000001, 0x00800000, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000,
    0x7f800001, 0xffa00000, 0x7fc12345]
ROUND_DOUBLES = [double(v) for v in (
    0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, 0.5 - 2.0**-54, 0.5 + 2.0**-53, 1.0, -1.0, 2.0**31 - 0.5,
    2.0**31 - 0.5 - 2.0**-22, -2.0**31 - 0.5, -2.0**31, -2.0**31 + 0.5, 2.0**31, 2.0**31 - 1, -2.0**31 - 1,
    2.0**52 - 0.5, 3e9, 4.4e18, 2.0**63 - 1024, 2.0**63, -2.0**63, -2.0**63 - 2048, 1e300, -1e300)] + [
    0x0000000000000001, 0x000fffffffffffff, 0x800fffffffffffff, 0x8000000000000001, 0x0010000000000000,
    0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000, 0xfff8000000000000, 0x7ff0000000000001,
    0xfff7ffffffffffff, 0x7ff8000012345678]
# Wrong rules and the words they must break, as the guard of the test: a rule that rounds, narrows or extends
# wrongly has to disagree with the processor on its own words and on no other.
ROUND_MUTANTS = {"truncate": tuple(ROUND_LAYOUT), "ties-away": tuple(ROUND_LAYOUT), "indefinite-zero": tuple(ROUND_LAYOUT),
                 "ignore-rc": tuple(ROUND_LAYOUT), "daz-ignored": tuple(ROUND_LAYOUT), "keep-upper": ROUND_32,
                 "sign-extended": ROUND_32,
                 "r64-narrowed": tuple(offset for offset in ROUND_LAYOUT if offset not in ROUND_32)}
# Fixed from the instruction set reference: (MXCSR word, source, source bits, 32-bit result, 64-bit result).
ROUND_ANSWERS = [(DEFAULT_MXCSR, kind, *answer) for kind, *answer in (
    ("f", single(0.5), 0, 0), ("f", single(1.5), 2, 2), ("f", single(2.5), 2, 2), ("f", single(-0.5), 0, 0),
    ("f", single(-1.5), 0xfffffffe, (1 << 64) - 2), ("f", single(-2.5), 0xfffffffe, (1 << 64) - 2),
    ("f", 0x7fc00000, 1 << 31, 1 << 63), ("f", 0x7f800000, 1 << 31, 1 << 63), ("f", 0xff800000, 1 << 31, 1 << 63),
    ("f", single(2.0**31), 1 << 31, 1 << 31), ("f", single(2.0**63), 1 << 31, 1 << 63),
    ("f", single(-2.0**63), 1 << 31, 1 << 63), ("f", single(-2.0**31), 1 << 31, (1 << 64) - (1 << 31)),
    ("d", double(2.0**31 - 0.5), 1 << 31, 1 << 31), ("d", double(-2.0**31 - 0.5), 1 << 31, (1 << 64) - (1 << 31)),
    ("d", double(2.0**63 - 1024), 1 << 31, (1 << 63) - 1024), ("d", 0xfff8000000000000, 1 << 31, 1 << 63),
    ("d", double(0.5 + 2.0**-53), 1, 1))] + [
    (0x7f80, "d", double(2.0**31 - 0.5), 0x7fffffff, 0x7fffffff), (0x3f80, "f", single(-0.5), 0xffffffff, (1 << 64) - 1),
    (0x5f80, "f", single(0.5), 1, 1), (0x5f80, "f", single(-0.5), 0, 0), (0x3f80, "d", double(2.5), 2, 2),
    (0x7f80, "f", single(-2.5), 0xfffffffe, (1 << 64) - 2), (0x5f80, "d", double(2.5), 3, 3),
    (DEFAULT_MXCSR, "f", 1, 0, 0), (0x5f80, "f", 1, 1, 1), (0x5f80 | DAZ, "f", 1, 0, 0), (0x3f80 | DAZ, "f", 0x80000001, 0, 0),
    (0x3f80, "f", 0x80000001, 0xffffffff, (1 << 64) - 1), (0x5f80 | DAZ | FTZ, "d", 1, 0, 0), (0x5f80 | FTZ, "d", 1, 1, 1)]


def rounded(bits, width, size, mxcsr, variant=None):
    """CVTSS2SI or CVTSD2SI of one source, from exact rational arithmetic: the integer bits for the rounding control
    of the MXCSR word (0 nearest even, 1 down, 2 up, 3 toward zero), the integer indefinite for a NaN, an infinity or
    a result that does not fit, and zero for a denormal under DAZ. FTZ plays no part. A variant is a wrong rule."""
    fraction_bits, exponent_mask = (23, 0xff) if width == 4 else (52, 0x7ff)
    size = 4 if variant == "r64-narrowed" else size
    indefinite = 0 if variant == "indefinite-zero" else INDEFINITE[size]
    exponent = bits >> fraction_bits & exponent_mask
    if exponent == exponent_mask:
        return indefinite
    if exponent == 0 and mxcsr & DAZ and variant != "daz-ignored":
        return 0
    x = Fraction(number(bits, width))
    control = 3 if variant == "truncate" else 0 if variant == "ignore-rc" else mxcsr >> 13 & 3
    floor = math.floor(x)
    rest = x - floor
    if control == 0 and variant == "ties-away":
        result = (1 if x >= 0 else -1) * math.floor(abs(x) + Fraction(1, 2))
    elif control == 0:
        result = floor + (rest > Fraction(1, 2) or (rest == Fraction(1, 2) and floor % 2 == 1))
    else:
        result = floor if control == 1 or (control == 3 and x >= 0) else floor + (rest != 0)
    if not -INDEFINITE[size] <= result < INDEFINITE[size]:
        return indefinite
    return result & (1 << 8 * size) - 1


def rounded_answer(values, payload, mxcsr, variant=None):
    """The result and the whole buffer of round_cvt for the slot values (f, d, pointer)."""
    memory = bytearray(payload)
    memory[40:44] = struct.pack("<I", values[0] & MASK32)
    memory[48:56] = struct.pack("<Q", values[1])
    for offset, (width, size) in ROUND_LAYOUT.items():
        word = rounded(values[0] & MASK32 if width == 4 else values[1], width, size, mxcsr, variant)
        if size == 4 and variant == "keep-upper":
            word |= (1 << 64) - (1 << 32)
        if size == 4 and variant == "sign-extended" and word >> 31:
            word |= (1 << 64) - (1 << 32)
        memory[offset:offset + 8] = struct.pack("<Q", word)
    return struct.unpack_from("<Q", memory, 0)[0], bytes(memory)


def probe_answer(a, b):
    seed = double(2.75)
    integer = struct.pack("<i", -1) + bytes(4)
    operand = 1.5
    inputs = struct.pack("<Q8sf4x", seed, integer, operand)
    low, high = struct.unpack("<fI", struct.pack("<Q", seed))
    memory = inputs + struct.pack("<QfIfIfI", seed, low + operand, high,
                                  float(struct.unpack_from("<i", integer)[0]), high,
                                  operand, 0)
    result = 2 * (a > b) + 3 * (a < b) + (a != a or b != b)
    return "probe_ss", (None, single(a), single(b)), inputs + bytes([FILL]) * 32, result, memory


def known_answers():
    """Results fixed by the instruction set reference, independent of either build."""
    nan32, nan64 = 0xffc00000, 0xfff8000000000000
    min32, min64 = 0x80000000, 1 << 63
    def convert(f, d, *fields):
        memory = struct.pack("<IIQQQIIQIIQQQ", *fields[:9], FILL * 0x01010101, *fields[9:], d & ~MASK32 | f)
        return "convert", (f, d, None), bytes([FILL]) * 80, fields[0] + fields[1] & MASK32, memory

    def minmax(name, a, b, result):
        payload = SEED + bytes([FILL]) * (WINDOWS[name] - 8)
        return name, (a, b, None), payload, result, buffer(name, (a, b), payload, result)

    def root(name, a, result):
        payload = SEED + bytes([FILL]) * (WINDOWS[name] - 8)
        return name, (a, None), payload, result, buffer(name, (a,), payload, result)

    quiet_nan, signalling_nan, minus_zero32 = 0x7fc12345, 0x7f800001, 0x80000000
    return [
        # MIN and MAX return the source operand, unchanged, when the compare is false: equal values (zeros of
        # either sign) and a NaN on either side.
        minmax("min_ss", 0, minus_zero32, minus_zero32), minmax("min_ss", minus_zero32, 0, 0),
        minmax("max_ss", 0, minus_zero32, minus_zero32), minmax("max_ss", minus_zero32, 0, 0),
        minmax("min_ss", quiet_nan, single(1.0), single(1.0)), minmax("max_ss", signalling_nan, single(1.0), single(1.0)),
        minmax("min_ss", single(1.0), quiet_nan, quiet_nan), minmax("max_ss", single(1.0), signalling_nan, signalling_nan),
        minmax("min_ss", single(2.0), single(1.0), single(1.0)), minmax("max_ss", single(2.0), single(1.0), single(2.0)),
        minmax("min_ss", 0xff800000, single(1.0), 0xff800000), minmax("max_ss", 0x7f800000, single(1.0), 0x7f800000),
        minmax("min_sd", 0, 1 << 63, 1 << 63), minmax("min_sd", 1 << 63, 0, 0),
        minmax("max_sd", 0, 1 << 63, 1 << 63), minmax("max_sd", 1 << 63, 0, 0),
        minmax("min_sd", 0x7ff8000012345678, double(1.0), double(1.0)),
        minmax("max_sd", double(1.0), 0xfff4000000000000, 0xfff4000000000000),
        minmax("min_sd", double(2.0), double(1.0), double(1.0)), minmax("max_sd", double(2.0), double(1.0), double(2.0)),
        # SQRT: exact roots, the default NaN for a negative operand, -0 kept, a quieted NaN with its payload.
        root("root_ss", single(4.0), single(2.0)), root("root_ss", single(2.0), 0x3fb504f3),
        root("root_ss", single(-1.0), 0xffc00000), root("root_ss", 0xff800000, 0xffc00000),
        root("root_ss", minus_zero32, minus_zero32), root("root_ss", 0, 0), root("root_ss", 0x7f800000, 0x7f800000),
        root("root_ss", signalling_nan, 0x7fc00001), root("root_ss", quiet_nan, quiet_nan),
        root("root_ss", 0xffa00000, 0xffe00000), root("root_ss", 1, 0x1a3504f3),
        root("root_sd", double(9.0), double(3.0)), root("root_sd", double(2.0), 0x3ff6a09e667f3bcd),
        root("root_sd", double(-4.0), 0xfff8000000000000), root("root_sd", 0xfff0000000000000, 0xfff8000000000000),
        root("root_sd", 1 << 63, 1 << 63), root("root_sd", 0x7ff0000000000000, 0x7ff0000000000000),
        root("root_sd", 0x7ff0000000000001, 0x7ff8000000000001), root("root_sd", 1, 0x1e60000000000000),
        ("root_const_ss", (), b"", 0x3fb504f3, b""), ("root_const_sd", (), b"", 0x3ffbb67ae8584caa, b""),
        # (1.5 + 2.25) * 2 - 0.5 = 7; 1.5 / 7.
        ("blend_ss", (single(1.5), single(2.25), None, single(0.5)), struct.pack("<II", single(2.0), 0),
         single(1.5 / 7.0), struct.pack("<II", single(2.0), single(7.0))),
        # inf + -inf is invalid: the default NaN has its sign set, and then propagates.
        ("blend_ss", (0x7f800000, 0xff800000, None, single(1.0)), struct.pack("<II", single(2.0), 0),
         nan32, struct.pack("<II", single(2.0), nan32)),
        # A signalling NaN in the destination is quieted and wins over the source NaN.
        ("blend_ss", (0x7f800001, 0xffe54321, None, 0x7fc12345), struct.pack("<II", 0xffa00000, 0),
         0x7fc00001, struct.pack("<II", 0xffa00000, 0x7fc00001)),
        # A NaN source propagates when the destination is a number: 0 * NaN.
        ("blend_ss", (0, 0, None, single(1.0)), struct.pack("<II", 0x7f800001, 0),
         0x7fc00001, struct.pack("<II", 0x7f800001, 0x7fc00001)),
        # (0.1 - 1) * 3 + 0.25, divided by 3, with double rounding at every step.
        ("blend_sd", (double(0.1), double(1.0), None, double(0.25)), struct.pack("<QQ", double(3.0), 0),
         double(((0.1 - 1.0) * 3.0 + 0.25) / 3.0), struct.pack("<QQ", double(3.0), double((0.1 - 1.0) * 3.0 + 0.25))),
        # 0 * inf is invalid.
        ("blend_sd", (0, 0, None, double(1.0)), struct.pack("<QQ", 0x7ff0000000000000, 0),
         nan64, struct.pack("<QQ", 0x7ff0000000000000, nan64)),
        # -0 - -0 = +0, +0 * -1 = -0, -0 + -0 = -0, -0 / -1 = +0.
        ("blend_sd", (min64, min64, None, min64), struct.pack("<QQ", double(-1.0), 0),
         0, struct.pack("<QQ", double(-1.0), min64)),
        # Truncation toward zero, then the integer indefinite for NaN and for values out of range.
        convert(single(-2.75), double(2.75), 0xfffffffe, 2, (1 << 64) - 2, 2, double(-2.75), single(2.75),
                single(-2.75), double(2.75), 2, double(2.75), double(2.75) & ~MASK32 | single(2.75)),
        convert(0x7fc00000, 0xfff8000000000000, min32, min32, min64, min64, 0x7ff8000000000000, 0xffc00000,
                0x7fc00000, 0xfff8000000000000, min32, 0xfff8000000000000, 0xfff80000ffc00000),
        # 3e9 fits in 64 bits only; 2^63 fits in neither; -2^31 - 0.5 truncates to -2^31.
        convert(single(3e9), double(-2147483648.5), min32, min32, 3000000000, (1 << 64) - (1 << 31),
                double(3e9), single(-2147483648.5), single(3e9), double(-2147483648.5), min32,
                double(-2147483648.0), double(-2147483648.5) & ~MASK32 | single(-2147483648.5)),
        convert(0x5f000000, 0x43e0000000000000, min32, min32, min64, min64, 0x43e0000000000000, 0x5f000000,
                0x5f000000, 0x43e0000000000000, min32, 0x43e0000000000000, 0x43e000005f000000),
        # 2^24 + 1 is a tie in single precision and rounds to even; the low dword alone is 1.
        ("from_int", (0x100000001000001, None), bytes([FILL]) * 32, double(float(0x100000001000001) + 16777217.0),
         struct.pack("<IIQQQ", single(16777216.0), single(float(1 << 56)), double(16777217.0),
                     double(float(0x100000001000001)), 0x100000001000001)),
        # -1 as a dword and as a qword.
        ("from_int", (0xffffffff, None), bytes([FILL]) * 32, double(4294967294.0),
         struct.pack("<IIQQQ", single(-1.0), single(4294967296.0), double(-1.0), double(4294967295.0), 0xffffffff)),
        ("classify_ss", (single(1.0), single(1.0)), b"", 1, b""),
        ("classify_ss", (0x80000000, 0), b"", 1, b""),
        ("classify_ss", (single(2.0), single(1.0)), b"", 2, b""),
        ("classify_ss", (single(1.0), single(2.0)), b"", 4, b""),
        ("classify_ss", (0x7fc00000, single(1.0)), b"", 8, b""),
        ("classify_ss", (single(1.0), 0x7f800001), b"", 8, b""),
        ("classify_sd", (double(1.0), None), struct.pack("<Q", double(1.0)), 1, struct.pack("<Q", double(1.0))),
        ("classify_sd", (double(2.0), None), struct.pack("<Q", double(1.0)), 2, struct.pack("<Q", double(1.0))),
        ("classify_sd", (double(1.0), None), struct.pack("<Q", double(2.0)), 4, struct.pack("<Q", double(2.0))),
        ("classify_sd", (0x7ff0000000000001, None), struct.pack("<Q", 0), 8, struct.pack("<Q", 0)),
        ("zero_xorps", (double(1.0),), b"", 0, b""),
        ("zero_pxor", (double(-2.0),), b"", 0, b""),
        ("zero_xorpd", (double(3.0),), b"", 0, b""),
        # The conversion pair quiets a signalling NaN and keeps a quiet one and a number unchanged.
        ("round_trip", (0x7f800001,), b"", 0x7fc00001, b""),
        ("round_trip", (0xffa00000,), b"", 0xffe00000, b""),
        ("round_trip", (0x7fc12345,), b"", 0x7fc12345, b""),
        ("round_trip", (single(-2.75),), b"", single(-2.75), b""),
        probe_answer(1.0, 1.0),
        probe_answer(float("nan"), 1.0),
        ("neg_jcc", (0,), b"", 0, b""),
        ("neg_jcc", (1,), b"", 7, b""),
        ("neg_jcc", (0xffffffff,), b"", 3, b""),
        ("neg_jcc", (0x80000000,), b"", 0xf, b""),
        ("setcc_mix", (0xffffffff, 1, None), bytes([FILL]) * 4, 0x123456789abcde01, b"\x00\x01\x00\x00"),
        ("setcc_mix", (0x80000000, 0x80000000, None), bytes([FILL]) * 4, 0x123456789abcde00, b"\x01\x01\x01\x00"),
        ("setcc_mix", (0, 1, None), bytes([FILL]) * 4, 0x123456789abcde01, b"\x00\x00\x00\x01"),
        ("cmovcc_mix", (1, 2, None), struct.pack("<QQ", 0x12345678deadbeef, 0xaaaa555511112222),
         0x76543210, struct.pack("<QQ", 0x12345678deadbeef, 0xaaaa555511112222)),
        ("cmovcc_mix", (5, 2, None), struct.pack("<QQ", 0x12345678deadbeef, 0xaaaa555511112222),
         0xdeadbeef, struct.pack("<QQ", 0x12345678deadbeef, 0xaaaa555511112222)),
        ("cmovcc_mix", (0, 99, None), struct.pack("<QQ", 0xdeadbeef, 0xaaaa555511112222),
         99, struct.pack("<QQ", 0xdeadbeef, 0xaaaa555511112222)),
        ("cmovcc_mix", (1 << 63, 7, None), struct.pack("<QQ", 0xdeadbeef, 0xaaaa555511112222),
         0xaaaa555511112222, struct.pack("<QQ", 0xdeadbeef, 0xaaaa555511112222)),
        # MOVQ between XMM and 64-bit GPR preserves all 64 bits verbatim across -0.0, NaNs, subnormals, and +/-inf.
        ("movq_gpr", (min64, 0), b"", min64, b""),
        ("movq_gpr", (0x7ff0000000000001, 0), b"", 0x7ff0000000000001, b""),
        ("movq_gpr", (0xfff4000000000000, 0), b"", 0xfff4000000000000, b""),
        ("movq_gpr", (0x7ff7ffffffffffff, 0), b"", 0x7ff7ffffffffffff, b""),
        ("movq_gpr", (0x7ff8000012345678, 0), b"", 0x7ff8000012345678, b""),
        ("movq_gpr", (0xfffc0000deadbeef, 0), b"", 0xfffc0000deadbeef, b""),
        ("movq_gpr", (0x0000000000000001, 0), b"", 0x0000000000000001, b""),
        ("movq_gpr", (0x800fffffffffffff, 0), b"", 0x800fffffffffffff, b""),
        ("movq_gpr", (0x7ff0000000000000, 0), b"", 0x7ff0000000000000, b""),
        ("movq_gpr", (0xfff0000000000000, 0), b"", 0xfff0000000000000, b""),
        ("movq_gpr", (0, min64), b"", min64, b""),
        ("movq_gpr", (0, 0x7ff0000000000001), b"", 0x7ff0000000000001, b""),
        ("movq_gpr", (0x7ff8000012345678, 0xfedcba9876543210), b"", 0x7ff8000012345678 ^ 0xfedcba9876543210, b""),
    ]


def check(original, lifted, swapped, swapped_lifted):
    """Each argument is call(name, slots) -> result bits for one build of the fixture."""
    for name, values, payload, result, memory in known_answers():
        for label, call in (("source fixture", original), ("translation", lifted)):
            got, raw = run(call, name, values, payload)
            if got != result or raw != GUARD + memory + GUARD:
                raise AssertionError(f"{label} {name}{values}: {got:#x} {raw.hex()}")
    matched = dict.fromkeys(SIGNATURES, 0)
    changed = dict.fromkeys(SIGNATURES, 0)
    for name, values, payload in cases():
        native = run(original, name, values, payload)
        actual = run(lifted, name, values, payload)
        if actual != native:
            raise AssertionError(f"translation mismatch for {name}{values}: {actual} != {native}")
        if native[1][:8] != GUARD or native[1][-8:] != GUARD:
            raise AssertionError(f"{name}{values} wrote outside its buffer")
        if name in EXTREMA or name in ROOTS:
            expected = reference(name, values, payload)
            if native != (expected[0], GUARD + expected[1] + GUARD):
                raise AssertionError(f"{name}{values}: the processor gave {native}, the reference {expected}")
        if name == "round_cvt":
            expected = rounded_answer(values, payload, DEFAULT_MXCSR)
            if native != (expected[0], GUARD + expected[1] + GUARD):
                raise AssertionError(f"{name}{values}: the processor gave {native}, the reference {expected}")
        if name == "movq_gpr":
            expected = values[0] ^ values[1]
            if native != (expected, GUARD + GUARD):
                raise AssertionError(f"{name}{values}: the processor gave {native}, expected {expected:#x}")
        control = run(swapped, name, values, payload)
        actual = run(swapped_lifted, name, values, payload)
        if actual != control:
            raise AssertionError(f"control translation mismatch for {name}{values}: {actual} != {control}")
        matched[name] += 1
        if control != native:
            changed[name] += 1
            # b + a differs from a + b only in which NaN payload survives.
            if name == "blend_ss" and not (is_nan(values[0], 4) and is_nan(values[1], 4)):
                raise AssertionError(f"swapped addition changed {name}{values}")
            # Swapped MIN and MAX operands differ only for a NaN or two equal values, where the source wins.
            if name in EXTREMA:
                width = EXTREMA[name][1]
                if not (is_nan(values[0], width) or is_nan(values[1], width)
                        or number(values[0], width) == number(values[1], width)):
                    raise AssertionError(f"swapped operands changed {name}{values}")
            # Dropping the conversion pair differs only for a signalling NaN.
            if name == "round_trip" and not (is_nan(values[0], 4) and not values[0] & 0x400000):
                raise AssertionError(f"dropping the conversion pair changed {name}{values}")
    if not all(changed.values()):
        raise AssertionError(f"swapped-operand control matched the original: {changed}")
    for name, variant in MUTANTS:
        # A wrong rule must fail on at least one call, and the first call it fails on is reported.
        failing = [(values, payload) for case, values, payload in cases() if case == name
                   and reference(name, values, payload, variant) != reference(name, values, payload)]
        if not failing:
            raise AssertionError(f"wrong rule {variant!r} for {name} agrees with the reference on every call")
        native = run(original, name, failing[0][0], failing[0][1])
        wrong = reference(name, *failing[0], variant)
        if native == (wrong[0], GUARD + wrong[1] + GUARD):
            raise AssertionError(f"wrong rule {variant!r} for {name} matches the processor on {failing[0][0]}")
    print(f"{len(MUTANTS)} wrong rules each fail on a call of their export")
    print(f"{len(known_answers())} fixed results and {matched} calls match on the result and the guarded "
          f"buffer for the original and the swapped-operand build; swapped operands changed {changed}")


def mxcsr_cases():
    """(MXCSR word, export, slot values, initial bytes): MIN and MAX with DAZ and FTZ over every pair of
    operands, SQRT under every rounding control and DAZ, and the constant-source roots under every control."""
    for name, (_, width) in EXTREMA.items():
        pool = SINGLES if width == 4 else DOUBLES
        for word in EXTREMA_WORDS:
            for a in pool:
                for b in pool:
                    yield word, name, (a, b, None), SEED + bytes([FILL]) * (WINDOWS[name] - 8)
    for name, width in ROOTS.items():
        for word in ROOT_WORDS:
            for a in (SINGLES if width == 4 else DOUBLES):
                yield word, name, (a, None), SEED + bytes([FILL]) * (WINDOWS[name] - 8)
    for name in CONSTANT_ROOTS:
        for word in ROUNDING_WORDS:
            yield word, name, (), b""


def constant_root(name, word):
    """The result bits of a constant-source root, from the integer in the code and the rounding control."""
    operand, width = CONSTANT_ROOTS[name]
    bits = struct.unpack("<I" if width == 4 else "<Q", struct.pack("<f" if width == 4 else "<d", float(operand)))[0]
    return exact_root(bits, width, word >> 13 & 3)


# Each wrong rule of the MXCSR checks and the export it breaks.
MXCSR_MUTANTS = [("min_ss", "daz-result-kept"), ("max_ss", "daz-result-kept"), ("min_sd", "daz-result-kept"),
                 ("max_sd", "daz-result-kept"), ("root_ss", "nearest-only"), ("root_sd", "nearest-only"),
                 ("root_ss", "library-sign-test"), ("root_sd", "library-sign-test")]


def check_mxcsr(original, lifted, access):
    """MIN, MAX and SQRT of the fixture and of its translation under the same host MXCSR word, against the
    reference. A translation that lets the compiler fold a constant root, or that returns a denormal MIN or MAX
    operand unflushed under DAZ, disagrees with the processor here."""
    counts = {}
    seen = {}
    for word, name, values, payload in mxcsr_cases():
        native = under(access, word, original, name, values, payload)
        actual = under(access, word, lifted, name, values, payload)
        if actual != native:
            raise AssertionError(f"translation mismatch under MXCSR {word:#x} for {name}{values}: {actual} != {native}")
        if name in CONSTANT_ROOTS:
            expected = constant_root(name, word)
            if native != (expected, GUARD + payload + GUARD):
                raise AssertionError(f"{name} under MXCSR {word:#x}: the processor gave {native}, the reference {expected:#x}")
            seen.setdefault((name, expected), word)
        else:
            expected = reference(name, values, payload, mxcsr=word)
            if native != (expected[0], GUARD + expected[1] + GUARD):
                raise AssertionError(f"{name}{values} under MXCSR {word:#x}: the processor gave {native}, the reference {expected}")
            for case, variant in MXCSR_MUTANTS:
                if case == name:
                    wrong = reference(name, values, payload, variant, word)
                    if wrong != expected:
                        counts[(name, variant)] = counts.get((name, variant), 0) + 1
        counts[name] = counts.get(name, 0) + 1
    for word, name, values, result in DENORMAL_ANSWERS:
        payload = SEED + bytes([FILL]) * (WINDOWS[name] - 8)
        for label, call in (("processor", original), ("translation", lifted)):
            got = under(access, word, call, name, values, payload)[0]
            if got != result:
                raise AssertionError(f"{label} {name}{values} under MXCSR {word:#x}: {got:#x}, not {result:#x}")
    for name in CONSTANT_ROOTS:
        if len({bits for (case, bits) in seen if case == name}) < 2:
            raise AssertionError(f"{name} gave one result under every rounding control")
    for case, variant in MXCSR_MUTANTS:
        if not counts.get((case, variant)):
            raise AssertionError(f"wrong rule {variant!r} for {case} agrees with the reference on every call")
    print(f"{sum(v for k, v in counts.items() if isinstance(k, str))} calls match the processor under MXCSR words with "
          f"rounding controls, DAZ and FTZ; {len(MXCSR_MUTANTS)} wrong rules each disagree on a call")


def check_rounding(original, lifted, access):
    """round_cvt of the fixture and of its translation under the same host MXCSR word, for every rounding control with
    and without DAZ and FTZ, against the reference and the fixed answers. The wrong rules must each break their own words."""
    pairs = [(f, ROUND_DOUBLES[(7 * i + 11 * k) % len(ROUND_DOUBLES)]) for i, f in enumerate(ROUND_SINGLES) for k in range(6)]
    payload = bytes([FILL]) * WINDOWS["round_cvt"]
    broken = {variant: set() for variant in ROUND_MUTANTS}
    for word in ROUND_CONTROL_WORDS:
        for f, d in pairs:
            values = (f, d, None)
            native = under(access, word, original, "round_cvt", values, payload)
            actual = under(access, word, lifted, "round_cvt", values, payload)
            if actual != native:
                raise AssertionError(f"translation mismatch under MXCSR {word:#x} for round_cvt{values}: {actual} != {native}")
            expected = rounded_answer(values, payload, word)
            if native != (expected[0], GUARD + expected[1] + GUARD):
                raise AssertionError(f"round_cvt{values} under MXCSR {word:#x}: the processor gave {native}, the reference {expected}")
            for variant in ROUND_MUTANTS:
                wrong = rounded_answer(values, payload, word, variant)[1]
                broken[variant].update(o for o in ROUND_LAYOUT if wrong[o:o + 8] != expected[1][o:o + 8])
    for word, kind, bits, small, large in ROUND_ANSWERS:
        values = (bits, 0, None) if kind == "f" else (0, bits, None)
        for label, call in (("processor", original), ("translation", lifted)):
            raw = under(access, word, call, "round_cvt", values, payload)[1][8:-8]
            for offset, (width, size) in ROUND_LAYOUT.items():
                if width == (4 if kind == "f" else 8) and struct.unpack_from("<Q", raw, offset)[0] != (small, large)[size == 8]:
                    raise AssertionError(f"{label} round_cvt{values} under MXCSR {word:#x}: word {offset} is not "
                                         f"{(small, large)[size == 8]:#x}")
    for variant, offsets in ROUND_MUTANTS.items():
        if broken[variant] != set(offsets):
            raise AssertionError(f"wrong rule {variant!r} breaks words {sorted(broken[variant])}, not {sorted(offsets)}")
    print(f"round_cvt matches the processor and the reference for {len(ROUND_CONTROL_WORDS)} MXCSR words; "
          f"{len(ROUND_ANSWERS)} fixed answers; {len(ROUND_MUTANTS)} wrong rules each break their own words")


def check_rejections(original_path):
    data = Path(original_path).read_bytes()
    with pefile.PE(data=data, fast_load=True) as pe:
        code = {}
        for name in SIGNATURES:
            entry = export_rva(pe, name)
            (begin, end), = function_ranges(pe, entry)
            code[name] = list(Decoder(64, pe.get_data(begin, end - begin), ip=begin))
        offset = pe.get_offset_from_rva

    def first(name, mnemonic):
        return next(i for i in code[name] if i.mnemonic == mnemonic)

    def patch(start, encoding, *more):
        """Replace whole instructions, starting at one, by an encoding padded with NOPs."""
        length = start.len + sum(i.len for i in more)
        raw = bytes.fromhex(encoding)
        assert len(raw) <= length and all(a.next_ip == b.ip for a, b in zip((start, *more), more))
        return start.ip, raw.ljust(length, b"\x90")

    add, copy, multiply = first("blend_ss", M.ADDSS), first("blend_ss", M.MOVAPS), first("blend_ss", M.MULSS)
    load, subtract = first("blend_ss", M.MOVSS), first("blend_ss", M.SUBSS)
    truncate = next(i for i in code["convert"] if i.mnemonic == M.CVTTSS2SI and i.len == 5)
    round_ss, round_sd = first("round_cvt", M.CVTSS2SI), first("round_cvt", M.CVTSD2SI)
    round_mem = next(i for i in code["round_cvt"] if i.mnemonic == M.CVTSS2SI and i.op1_kind == OpKind.MEMORY)
    round_mem_next = next(i for i in code["round_cvt"] if i.ip == round_mem.next_ip)
    compare, parity, below_equal = (first("classify_ss", m) for m in (M.UCOMISS, M.JP, M.JBE))
    classify_jne = first("classify_ss", M.JNE)
    min_reg, min_mem = (i for i in code["min_ss"] if i.mnemonic == M.MINSS)
    max_sd_reg = first("max_sd", M.MAXSD)
    root_ss_reg, root_sd_reg = first("root_ss", M.SQRTSS), first("root_sd", M.SQRTSD)
    min_mem_next = next(i for i in code["min_ss"] if i.ip == min_mem.next_ip)
    neg_je = first("neg_jcc", M.JE)
    neg_after_je = next(i for i in code["neg_jcc"] if i.ip == neg_je.next_ip)
    neg_movs = [i for i in code["neg_jcc"] if i.mnemonic == M.MOV]
    neg_join = neg_movs[2]
    neg_join_next = next(i for i in code["neg_jcc"] if i.ip == neg_join.next_ip)
    setcc_mov_r9 = next(i for i in code["setcc_mix"] if i.mnemonic == M.MOV and i.ip > first("setcc_mix", M.CMP).ip)
    setcc_setl = first("setcc_mix", M.SETL)
    cmov_32 = first("cmovcc_mix", M.CMOVG)
    movq_to_gpr = next(i for i in code["movq_gpr"] if i.code == Code.MOVQ_RM64_XMM)
    movq_from_gpr = next(i for i in code["movq_gpr"] if i.code == Code.MOVQ_XMM_RM64)
    movq_from_rdx = next(i for i in code["movq_gpr"] if i.code == Code.MOVQ_XMM_RM64 and i.op1_register == R.RDX)
    assert load.next_ip == multiply.ip and compare.next_ip == parity.ip
    everything = [f"{name}={signature}" for name, signature in SIGNATURES.items()]
    controls = [
        (patch(add, "0f 58 d1"), "unsupported instruction", "packed ADDPS"),
        (patch(add, "f3 0f 52 d1"), "unsupported instruction", "RSQRTSS, which is only an approximation"),
        (patch(add, "c5 ea 58 d1"), "unsupported instruction", "VEX-encoded VADDSS"),
        (patch(add, "f2 0f 58 d1"), "undefined register xmm1[63:32]", "ADDSD on float arguments"),
        (patch(min_reg, "0f 5d d1"), "unsupported instruction", "packed MINPS"),
        (patch(min_reg, "c5 ea 5f d1"), "unsupported instruction", "VEX-encoded VMAXSS"),
        (patch(min_mem, "f3 0f 5d 1d 00 00 00 00", min_mem_next), "unsupported memory operand", "RIP-relative MINSS source"),
        (patch(max_sd_reg, "f2 0f 5f f1"), "nonvolatile register xmm6", "MAXSD into XMM6"),
        (patch(root_ss_reg, "f3 0f 51 f0"), "nonvolatile register xmm6", "SQRTSS into XMM6"),
        (patch(root_ss_reg, "f3 0f 51 d5"), "undefined register xmm5", "SQRTSS of an unwritten register"),
        (patch(root_sd_reg, "0f 51 d0"), "unsupported instruction", "packed SQRTPS"),
        (patch(subtract, "0f 57 d3"), "unsupported instruction", "XORPS of two registers"),
        (patch(copy, "0f 28 f0"), "nonvolatile register xmm6", "write to XMM6"),
        (patch(multiply, "f3 0f 59 d5"), "undefined register xmm5", "read of an unwritten register"),
        (patch(load, "f3 0f 10 25 00 00 00 00", multiply), "unsupported memory operand", "RIP-relative load"),
        (patch(truncate, "0f 2d d0"), "unsupported instruction", "CVTPS2PI into an MMX register"),
        (patch(round_ss, "66 0f 2d c0"), "unsupported instruction", "CVTPD2PI into an MMX register"),
        (patch(round_ss, "c5 fa 2d c0"), "unsupported instruction", "VEX-encoded VCVTSS2SI"),
        (patch(round_sd, "c5 fb 2d c1"), "unsupported instruction", "VEX-encoded VCVTSD2SI"),
        (patch(round_ss, "f3 0f 2d c5"), "undefined register xmm5", "CVTSS2SI of an unwritten register"),
        (patch(round_ss, "f3 0f 2d d8"), "nonvolatile register rbx", "CVTSS2SI into RBX"),
        (patch(round_mem, "f3 0f 2d 0d 00 00 00 00", round_mem_next), "unsupported memory operand", "RIP-relative CVTSS2SI source"),
        (patch(load, "66 48 0f 7e c8"), "undefined register xmm1[63:32]", "MOVQ reading XMM with only low 32-bit float defined"),
        (patch(movq_to_gpr, "66 48 0f 7e e8"), "undefined register xmm5", "MOVQ reading unwritten XMM register"),
        (patch(movq_from_rdx, "66 49 0f 6e db"), "undefined register r11", "MOVQ reading undefined 64-bit GPR"),
        (patch(movq_to_gpr, "66 48 0f 7e c3"), "nonvolatile register rbx", "MOVQ targeting nonvolatile RBX"),
        (patch(movq_from_gpr, "66 49 0f 6e f1"), "nonvolatile register xmm6", "MOVQ targeting nonvolatile XMM6"),
        (patch(movq_from_gpr, "66 48 0f 6e d3"), "nonvolatile register rbx", "MOVQ reading nonvolatile RBX"),
        (patch(movq_to_gpr, "66 48 0f 7e f0"), "nonvolatile register xmm6", "MOVQ reading nonvolatile XMM6"),
        (patch(movq_to_gpr, "48 0f 7e c2"), "unsupported instruction", "MOVQ from MMX register to 64-bit GPR"),
        (patch(movq_from_gpr, "48 0f 6e c2"), "unsupported instruction", "MOVQ from 64-bit GPR to MMX register"),
        ((below_equal.ip, b"\x7e"), "unsupported condition after a float compare", "JLE after COMISS"),
        (patch(below_equal, ""), "float compare without a conditional branch", "COMISS without a jump"),
        # CMP EAX, 1 and JB in place of UCOMISS and JP: the JNE that follows no longer has a producer.
        ((compare.ip, bytes.fromhex("83 f8 01 72")), "flags cross a branch", "second jump after CMP"),
        (patch(parity, "0f 94 c0", classify_jne), "unsupported flag producer", "SETcc after UCOMISS"),
        (patch(parity, "0f 44 c1", classify_jne), "unsupported flag producer", "CMOVcc after UCOMISS"),
        (patch(neg_after_je, "0f 94 c0"), "flags cross a join", "SETcc across a branch after NEG"),
        (patch(neg_join, "0f 44 c1", neg_join_next), "flags cross a join", "CMOVcc at a join target"),
        (patch(neg_after_je, "72 00"), "flags cross a branch", "second jump after NEG"),
        (patch(setcc_mov_r9, "41 0f 9c c1", setcc_setl), "undefined register r9", "SETcc partial write to undefined GPR"),
        (patch(cmov_32, "66 0f 4f c1"), "unsupported partial register write", "16-bit CMOVcc"),
    ]
    with TemporaryDirectory() as folder:
        path = Path(folder) / "control.dll"
        for (rva, raw), reason, label in controls:
            damaged = bytearray(data)
            damaged[offset(rva):offset(rva) + len(raw)] = raw
            path.write_bytes(damaged)
            try:
                translate(path, everything)
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"{label}: {exc}") from exc
            else:
                raise AssertionError(f"accepted control: {label}")
    signatures = [
        ("blend_ss=f:qfqf", "undefined register xmm2"),
        ("blend_ss=f:ffq", "undefined register xmm3"),
        ("blend_ss=d:ffqf", "undefined register xmm0[63:32]"),
        ("blend_sd=d:fdqd", "undefined register xmm0[63:32]"),
        ("from_int=d:qf", "undefined register rdx"),
        ("min_sd=d:ffq", "undefined register xmm0[63:32]"),
        ("root_sd=d:fq", "undefined register xmm0[63:32]"),
        ("max_ss=f:fqq", "undefined register xmm1"),
        ("root_ss=f:qf", "undefined register rdx"),
        ("from_int=q:qq", "undefined register rax"),
        ("round_cvt=q:fqq", "undefined register xmm1 at"),
        ("round_cvt=q:ffq", "undefined register xmm1[63:32]"),
        ("movq_gpr=q:fq", "undefined register xmm0[63:32]"),
        ("movq_gpr=q:df", "undefined register rdx"),
        ("blend_ss=f:ffqfq", "invalid signature"),
        ("blend_ss=i:ffqf", "invalid signature"),
    ]
    for export, reason in signatures:
        try:
            translate(original_path, [export])
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{export}: {exc}") from exc
        else:
            raise AssertionError(f"accepted signature: {export}")
    width_controls = [
        Instruction.create_reg_reg(Code.MOVD_RM32_XMM, R.RAX, R.XMM0),
        Instruction.create_reg_reg(Code.MOVD_XMM_RM32, R.XMM0, R.RAX),
        Instruction.create_reg_reg(Code.MOVQ_RM64_XMM, R.EAX, R.XMM0),
        Instruction.create_reg_reg(Code.MOVQ_XMM_RM64, R.XMM0, R.EAX),
    ]
    for ins in width_controls:
        try:
            ScalarLifter("q").statement(ins, None)
        except ValueError as exc:
            if "unsupported instruction" not in str(exc):
                raise AssertionError(f"unexpected rejection for {ins}: {exc}") from exc
        else:
            raise AssertionError(f"accepted width-mismatched instruction: {ins}")
    print(f"{len(controls)} patched-instruction rejections, {len(signatures)} signature rejections, "
          f"and {len(width_controls)} width-control rejections passed")


if __name__ == "__main__":
    original, lifted, swapped, swapped_lifted, harness = sys.argv[1:]
    check(library_caller(original), library_caller(lifted, "lifted_"),
          library_caller(swapped), library_caller(swapped_lifted, "lifted_"))
    check_mxcsr(library_caller(original), library_caller(lifted, "lifted_"), mxcsr_access(harness))
    check_rounding(library_caller(original), library_caller(lifted, "lifted_"), mxcsr_access(harness))
    check_rejections(original)
