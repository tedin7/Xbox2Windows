"""Check descriptor-format buffer_load_format in the compute store profile: exact lowering, refusals and readback.

The t0 format is V# word 3, read from a constant word at run time (--load-format-constant), as the
descriptor-format stores read theirs. The case file supplies the t0 words, so each format sees chosen edge values.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base  # noqa: E402

aot = base.aot
NAME = "descriptor_load_format"
GROUP = (4, 1, 1)
SOURCE_STRIDE = 80
OUTPUT_STRIDE = 120
THREADS = 8
# opcode suffix, t0 byte offset, GFX10 format code, legacy word 3, u0 byte offset of the results.
LOADS = (
    ("xyzw", 0, 56, False, 0),    # 8_8_8_8_UNORM
    ("xy", 4, 23, False, 16),     # 16_16_UNORM
    ("xyzw", 8, 50, False, 24),   # 2_10_10_10_UNORM: x in bits 9:0, w in bits 31:30
    ("xy", 12, 29, True, 40),     # 16_16_FLOAT, legacy DFMT/NFMT
    ("xyzw", 16, 71, False, 48),  # 16_16_16_16_FLOAT
    ("x", 24, 7, False, 64),      # 16_UNORM
    ("x", 28, 8, False, 68),      # 16_SNORM
    ("x", 32, 11, False, 72),     # 16_UINT
    ("x", 36, 12, False, 76),     # 16_SINT
    ("x", 40, 13, False, 80),     # 16_FLOAT
    ("xyz", 44, 72, False, 84),   # 32_32_32_UINT
    ("xyz", 56, 74, True, 96),    # 32_32_32_FLOAT, legacy
    ("x", 68, 20, False, 108),    # 32_UINT
    ("xy", 72, 64, False, 112),   # 32_32_FLOAT
)
# Format words that make one load invalid: load index, V# word 3. Each must leave every u0 word unwritten.
IDENTITY_32_UINT = aot.descriptor_format_word(20)
INVALID = (
    ("unknown format 8_SNORM", 0, aot.DESCRIPTOR_GFX10_FIXED | (2 << 12)),
    ("xyz opcode on a four-channel format", 10, aot.descriptor_format_word(75)),
    ("xy opcode on a one-channel format", 13, aot.descriptor_format_word(22)),
    ("non-identity DST_SEL", 12, IDENTITY_32_UINT ^ 1),
    # 10_10_10_2_UNORM has a 2-bit x in bits 1:0; only the 2_10_10_10 layout is handled.
    ("10_10_10_2_UNORM", 2, aot.DESCRIPTOR_GFX10_FIXED | (44 << 12)),
)
# Thread 0-3 t0 words: edge values per field. Threads 4-7 are pseudo-random.
CURATED = (
    (0x8001ff00, 0x0001ffff, 0xa00007ff, 0x80000001, 0xfc007c00, 0x7bff7e01, 0xdeadffff, 0x12348000, 0xabcdffff,
     0x55558000, 0xffff03ff, 0x89abcdef, 0x01234567, 0xfedcba98, 0x3e800000, 0xbfc00000, 0x7f800001, 0xcafef00d,
     0x3f800000, 0x40000000),
    (0x7f3c0a05, 0x80007fff, 0x00000000, 0x04000400, 0x00013c00, 0x83ff8400, 0xffff0001, 0x00008001, 0x00000000,
     0x00007fff, 0x00003c00, 0x00000000, 0xffffffff, 0x80000000, 0x00000000, 0x80000000, 0x7fc00000, 0x00000001,
     0xc0490fdb, 0x00000000),
    (0x01fe7f80, 0x5555aaaa, 0x7fffffff, 0x3c00bc00, 0x35553555, 0x00010001, 0x00008000, 0x00007fff, 0x12340001,
     0x00000001, 0x00000001, 0x11111111, 0x22222222, 0x33333333, 0x42f60000, 0x00000001, 0x80800000, 0x13579bdf,
     0x7f7fffff, 0x00800000),
    (0xfefdfcfb, 0xfffe0002, 0x40100401, 0xfbff7bff, 0x80018001, 0x03ff0000, 0x00005555, 0x0000ffff, 0x00010000,
     0x0000ffff, 0x0000fbff, 0xdeadbeef, 0x0badf00d, 0xfeedface, 0xbf800000, 0x3f7fffff, 0x00000000, 0x00000000,
     0xffffffff, 0x7f800000),
)
MASK = 0xffffffff


def source_words():
    words, state = [], 0x2545f491
    for thread in range(THREADS):
        for word in range(SOURCE_STRIDE // 4):
            if thread < len(CURATED):
                words.append(CURATED[thread][word])
            else:
                state = (state * 1103515245 + 12345) & MASK
                words.append(state ^ (state >> 13))
    return words + [base.SENTINEL] * (320 - len(words))


def f32_bits(value):
    """binary32 bits of a Fraction or exact value, rounded to nearest even (no double rounding)."""
    value = Fraction(value)
    if value == 0:
        return 0
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    while value >= Fraction(2) ** exponent * 2:
        exponent += 1
    while value < Fraction(2) ** exponent:
        exponent -= 1
    assert -126 <= exponent < 128, value
    mantissa = round(value / Fraction(2) ** (exponent - 23))
    if mantissa == 1 << 24:
        mantissa, exponent = 1 << 23, exponent + 1
    return sign | ((exponent + 127) << 23) | (mantissa - (1 << 23))


def half_to_f32(half):
    """IEEE 754 widening of binary16: exact for numbers; a NaN keeps its payload (IEEE 754-2008 6.2.3)."""
    sign, exponent, mantissa = (half >> 15) & 1, (half >> 10) & 31, half & 0x3ff
    if exponent == 31:
        return (sign << 31) | 0x7f800000 | (mantissa << 13)
    return struct.unpack("<I", struct.pack("<f", struct.unpack("<e", struct.pack("<H", half))[0]))[0]


def unorm(x, bits, *, divisor=None, truncate=False):
    d = divisor or (1 << bits) - 1
    if truncate:
        value = Fraction(x, d)
        if value == 0:
            return 0
        bits32 = f32_bits(value)
        # Round toward zero instead: step back when nearest rounding went up.
        return bits32 - 1 if struct.unpack("<f", struct.pack("<I", bits32))[0] > value else bits32
    return f32_bits(Fraction(x, d))


def snorm16(half, *, clamp=True):
    value = half - 0x10000 if half & 0x8000 else half
    if clamp and value <= -32767:
        return 0xbf800000
    return f32_bits(Fraction(value, 32767))


def load(code, words, channels, *, wrong=()):
    """Independent reference for one load: the definition of each format, from the t0 words at its offset.

    The profile handles an opcode that reads exactly the format's channels; anything else is a refusal.
    """
    _code, kind, format_channels = aot.decode_buffer_descriptor_format(aot.descriptor_format_word(code))
    assert channels == format_channels, (code, channels)
    w = words[0]
    nu = dict(divisor=256) if "unorm divisor" in wrong else dict(truncate=True) if "unorm truncated" in wrong else {}
    half = (lambda h: (h & 0x8000) << 16 if "flush halves" in wrong and (h >> 10) & 31 == 0 else half_to_f32(h))
    if kind == "raw32":
        result = words[:format_channels]
    elif kind == "unorm8":
        result = [unorm((w >> (8 * i)) & 0xff, 8, **nu) for i in range(4)]
    elif kind == "unorm16x2":
        result = [unorm(w & 0xffff, 16, **nu), unorm(w >> 16, 16, **nu)]
    elif kind == "unorm10":
        result = [unorm((w >> (10 * i)) & 0x3ff, 10, **nu) for i in range(3)] + [unorm(w >> 30, 2, **nu)]
    elif kind == "f16x2":
        result = [half(w & 0xffff), half(w >> 16)]
        if "swap halves" in wrong:
            result.reverse()
    elif kind == "f16x4":
        result = [half(w & 0xffff), half(w >> 16), half(words[1] & 0xffff), half(words[1] >> 16)]
    elif kind == "unorm16":
        result = [unorm(w & 0xffff, 16, **nu)]
    elif kind == "snorm16":
        result = [snorm16(w & 0xffff, clamp="snorm unclamped" not in wrong)]
    elif kind == "uint16":
        result = [w & 0xffff]
    elif kind == "sint16":
        result = [w & 0xffff if "sint zero-extended" in wrong else (((w & 0xffff) ^ 0x8000) - 0x8000) & MASK]
    else:
        result = [half(w & 0xffff)]
    return result[:channels]


def expected_words(groups, *, wrong=(), source=None, placeholder=None):
    """u0 after a valid dispatch. placeholder=(load index, words) publishes those words for that load instead."""
    source = source or source_words()
    words = [base.SENTINEL] * 320
    for thread in range(GROUP[0] * groups):
        index = thread + 1 if "neighbour element" in wrong else thread
        element = source[index * SOURCE_STRIDE // 4:(index + 1) * SOURCE_STRIDE // 4]
        for number, (suffix, src, code, legacy, dst) in enumerate(LOADS):
            if placeholder and placeholder[0] == number:
                values = placeholder[1](element[src // 4:])
            elif legacy and "legacy ignored" in wrong:
                values = [0] * len(suffix)
            else:
                values = load(code, element[src // 4:], len(suffix), wrong=wrong)
            start = (thread * OUTPUT_STRIDE + dst) // 4
            words[start:start + len(values)] = values
    return words


def constants():
    words = [aot.descriptor_format_word(code, legacy=legacy) for _s, _o, code, legacy, _d in LOADS]
    return words + [0] * (16 - len(words))


def program():
    lines = [
        "s_mov_b32 s12, s7", "s_mov_b32 s13, 0",
        "s_load_dwordx4 s[20:23], s[12:13], 0x30",
        "s_load_dwordx4 s[16:19], s[12:13], 0x40",
        "s_load_dwordx4 s[40:43], s[12:13], 0x50",
        "v_lshl_add_u32 v8, s11, 2, v0",
        "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dwordx4 s[24:27], s[16:19], null",
        "s_buffer_load_dwordx4 s[28:31], s[16:19], 0x10",
        "s_buffer_load_dwordx4 s[32:35], s[16:19], 0x20",
        "s_buffer_load_dwordx2 s[36:37], s[16:19], 0x30",
        "s_waitcnt lgkmcnt(0)",
    ]
    reg, stores = 10, []
    for suffix, src, _code, _legacy, dst in LOADS:
        n = len(suffix)
        regs = f"v{reg}" if n == 1 else f"v[{reg}:{reg + n - 1}]"
        lines.append(f"buffer_load_format_{suffix} {regs}, v8, s[40:43], 0 idxen" + (f" offset:{src}" if src else ""))
        width = "" if n == 1 else f"x{n}"
        stores.append(f"buffer_store_dword{width} {regs}, v8, s[20:23], 0 idxen" + (f" offset:{dst}" if dst else ""))
        reg += n
    lines += ["s_waitcnt vmcnt(0)", *stores, "s_endpgm"]
    binding = dict(table_reg=7, descriptor_offset=0x30, constant_descriptor=0x40, source_descriptor=0x50,
                   source_stride=SOURCE_STRIDE, group_reg=11, stride=OUTPUT_STRIDE, cpu="gfx1013",
                   load_format_constant=tuple(range(len(LOADS))))
    return lines, binding


LINES, BINDING = program()
FIRST_LOAD = ("    bool format_valid_0;\n"
              "    uint4 format_load_0 = load_descriptor_format(params[0].x, tid.x * 80u + 0u, 4u, format_valid_0);")
FIRST_STORE = ("    outputData.Store4(tid.x * 120u + 0u, uint4(format_load_0.x, format_load_0.y, format_load_0.z, "
               "format_load_0.w));")
LAST_LOAD = "    uint4 format_load_13 = load_descriptor_format(params[3].y, tid.x * 80u + 72u, 2u, format_valid_13);"
REFUSE = "    if (!(" + " && ".join(f"format_valid_{i}" for i in range(len(LOADS))) + ")) return;"
WRONG = (("UNORM divided by 2^n", ("unorm divisor",)), ("UNORM truncated", ("unorm truncated",)),
         ("SNORM without the clamp", ("snorm unclamped",)), ("SINT16 zero-extended", ("sint zero-extended",)),
         ("FLOAT16 halves swapped", ("swap halves",)), ("FLOAT16 denormals flushed", ("flush halves",)),
         ("legacy format ignored", ("legacy ignored",)), ("neighbouring element", ("neighbour element",)))
FIXED = f"{aot.DESCRIPTOR_LEGACY_FIXED:#x}u"
# label, old, new, and the run it must fail: None for the valid two-group run, else an INVALID index.
MUTATIONS = (
    ("ratio never rounds up", "    if ((bits & 1u) != 0u && (r != 0u || (result & 1u) != 0u)) ++result;\n", "", None),
    ("denormal halves flushed", "    if (m == 0u) return sign;\n", "    return sign;\n", None),
    # Dropping the -1 clamp alone is not observable: the long division of 32768 by 32767 also gives 1.0.
    ("SNORM sign dropped", "0x80000000u | descriptor_ratio(", "descriptor_ratio(", None),
    ("SNORM divided by 32768", "descriptor_ratio((uint)value, 32767u)", "descriptor_ratio((uint)value, 32768u)", None),
    ("SINT16 zero-extended", "(uint)((int)(raw.x << 16u) >> 16u)", "(raw.x & 0xffffu)", None),
    ("legacy fixed bits ignored", f"}} else if (fixedBits == {FIXED}) {{", "} else if (false) {", None),
    ("format word from params[0].y", "load_descriptor_format(params[0].x,", "load_descriptor_format(params[0].y,", None),
    ("invalid loads still store", REFUSE, "", 0),
    ("narrower opcode accepted", "format == 77u) && channels == 4u)", "format == 77u) && channels <= 4u)", 1),
    ("wider opcode accepted", "format == 22u) && channels == 1u)", "format == 22u) && channels >= 1u)", 2),
    ("selectors ignored", f"uint fixedBits = formatWord & {aot.DESCRIPTOR_FIXED_MASK:#x}u;",
     f"uint fixedBits = formatWord & {aot.DESCRIPTOR_FIXED_MASK & ~0xfff:#x}u | 0xfacu;", 3),
)


def lower(lines=None, **binding):
    return aot.lower(LINES if lines is None else lines, GROUP, **(BINDING | binding))


def controls():
    """Exact staged loads and stores, model checks, then one rejection per form outside the profile."""
    assert f32_bits(Fraction(1, 255)) == 0x3b808081 and f32_bits(Fraction(128, 255)) == 0x3f008081
    assert unorm(1, 16) == 0x37800080 and unorm(2, 2) == 0x3f2aaaab and snorm16(0x8000) == 0xbf800000
    assert snorm16(0x8000, clamp=False) == 0xbf800100 and snorm16(0xffff) == 0xb8000100
    assert half_to_f32(0x0001) == 0x33800000 and half_to_f32(0x03ff) == 0x387fc000 and half_to_f32(0x7e01) == 0x7fc02000
    source = lower()
    lines = source.splitlines()
    assert source.count("= load_descriptor_format(") == len(LOADS)
    assert FIRST_LOAD in source and LAST_LOAD in lines and FIRST_STORE in lines and REFUSE in lines, source
    assert "outputData.Store(tid.x * 120u + 108u, format_load_12.x);" in source
    assert "outputData.Store2(tid.x * 120u + 112u, uint2(format_load_13.x, format_load_13.y));" in source
    # Every load and the refusal come before the first store.
    assert lines.index(LAST_LOAD) < lines.index(REFUSE) < lines.index(FIRST_STORE)
    assert "&& channels == 4u) {" in source and "channels <=" not in source and "valid = false;" in source
    # Without the load constants the profile keeps the gfx1030 caller-format path and refuses gfx1013.
    base.rejects("Unsupported formatted buffer read target or address mode", lower, load_format_constant=None)
    for _name, old_lines, old_binding in base.mubuf_format_cases():
        old = aot.lower(old_lines, 32, **old_binding)
        assert "load_descriptor_format(" not in old and "inputData.Load" in old and " return;" not in old
    xyzw = LINES[12]
    assert xyzw == "buffer_load_format_xyzw v[10:13], v8, s[40:43], 0 idxen", xyzw
    wait = LINES.index("s_waitcnt lgkmcnt(0)", 7)
    for message, lines_, binding in (
        ("Descriptor-format d16 loads are not lowered",
         base.variant(LINES, (xyzw, "buffer_load_format_d16_xyzw v[10:11], v8, s[40:43], 0 idxen")), {}),
        ("Descriptor-format d16 loads are not lowered",
         base.variant(LINES, (xyzw, "buffer_load_format_d16_x v10, v8, s[40:43], 0 idxen")), {}),
        ("Unsupported formatted buffer read target or address mode",
         base.variant(LINES, (xyzw, xyzw.replace(" idxen", " offen"))), {}),
        ("Unsupported formatted buffer read target or address mode",
         base.variant(LINES, (xyzw, xyzw.replace(" idxen", ""))), {}),
        ("Unverified read offset", base.variant(LINES, (xyzw, xyzw + " offset:2")), {}),
        ("Read leaves its source element", base.variant(LINES, (xyzw, xyzw + " offset:68")), {}),
        ("Unsupported compute instruction", base.variant(LINES, (xyzw, xyzw + " glc")), {}),
        ("Formatted buffer load is missing a format constant", None, dict(load_format_constant=(0,))),
        ("Unused buffer load format constant", None, dict(load_format_constant=tuple(range(len(LOADS) + 1)))),
        ("Formatted buffer load format constant is not ready", None,
         dict(load_format_constant=(15,) + BINDING["load_format_constant"][1:])),
        ("Formatted buffer load format constant is not ready", LINES[:wait] + LINES[wait + 1:], {}),
        ("Invalid buffer load format constant", None, dict(cpu="gfx1030")),
        ("Unsupported formatted source binding", None, dict(source_format="BUF_FMT_32_32_32_32_UINT")),
        ("Invalid buffer load format constant", None, dict(cpu="gfx1030", source_format="BUF_FMT_32_32_32_32_UINT")),
        ("Invalid buffer load format constant", None, dict(load_format_constant=(16,))),
        ("Invalid buffer load format constant", None, dict(load_format_constant=())),
        ("Invalid buffer load format constant", None, dict(constant_descriptor=None)),
        ("Invalid buffer load format constant", None, dict(source_stride=None)),
        # A load result is data only.
        ("Unsupported store address", base.variant(LINES, (", v8, s[20:23], 0 idxen offset:16", ", v10, s[20:23], 0 idxen offset:16")), {}),
    ):
        base.rejects(message, lower, lines_, **binding)


def binary(directory, assembler, objcopy, library):
    """LLVM 15 assembles the fixture; lower of the disassembly and the CLI both match lower of the source."""
    controls()
    base.descriptor_store_round_trip(directory, LINES, BINDING, "mc", assembler, objcopy, library)
    print(f"{NAME}: llvm15_round_trip=match")
    return 0


def draw(directory, compiler, renderer, backend):
    """Dispatch with chosen t0 words; wrong references and mutated sources must fail.

    Every case owns its case and cso files, so the launches run on a small thread pool. Results are
    checked afterwards in the original order with the original messages.
    """
    src, cso = directory / "load.hlsl", directory / "load.cso"
    source = lower()
    src.write_text(source, encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    words = source_words()
    jobs = []  # (check, label, extra, case bytes, shader, mutated source or None)

    def add(check, label, groups, reference, formats=None, extra=None, mutated=None):
        data = (struct.pack("<321I", groups, *reference) + struct.pack("<16I", *(formats or constants()))
                + struct.pack("<320I", *words))
        jobs.append((check, label, extra, data, mutated))

    def run(number, job):
        _check, _label, _extra, data, mutated = job
        case, shader = directory / f"case{number}.bin", cso
        case.write_bytes(data)
        if mutated is not None:
            shader = directory / f"mutated{number}.cso"
            bad_src = directory / f"mutated{number}.hlsl"
            bad_src.write_text(mutated, encoding="ascii")
            subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", shader, bad_src], check=True)
        return subprocess.run([renderer, backend, "--compute-store-case", shader, case],
                              capture_output=True, text=True)

    for groups in (1, 2):
        add("pass", "dispatch", groups, expected_words(groups))

    def invalid_formats(number):
        formats = constants()
        formats[INVALID[number][1]] = INVALID[number][2]
        return formats

    # Each invalid format must leave all of u0 at the sentinel, and so fail a reference that publishes
    # placeholder words for that load: zeros, or the D3D (0, 0, 0, 1) fill for the wider opcode.
    unwritten = [base.SENTINEL] * 320
    zeros = (lambda channels: (lambda words: [0] * channels))
    for number, (label, index, _word) in enumerate(INVALID):
        add("pass", f"refused {label}", 2, unwritten, invalid_formats(number))
        channels = len(LOADS[index][0])
        placeholders = [("published as zeros", zeros(channels))]
        if number == 2:
            placeholders.append(("published with the (0, 0, 0, 1) fill", lambda words: [words[0], 0]))
        for what, values in placeholders:
            published = expected_words(2, placeholder=(index, values))
            add("placeholder", f"{label} {what}", 2, published, invalid_formats(number),
                extra=f"word=0 expected={published[0]:08x} actual=c5c5c5c5")
    expected = expected_words(2)
    for label, wrong in WRONG:
        reference = expected_words(2, wrong=wrong)
        word = next((i for i, (a, b) in enumerate(zip(expected, reference)) if a != b), None)
        if word is None:
            raise AssertionError(f"{NAME}: reference '{label}' matches the expected words")
        add("wrong", label, 2, reference, extra=f"word={word} expected=")
    for label, old, new, run_index in MUTATIONS:
        mutated = source.replace(old, new, 1)
        if mutated == source:
            raise AssertionError(f"{NAME}: mutation '{label}' did not change the source")
        if run_index is None:
            add("mutation", label, 2, expected, mutated=mutated)
        else:
            add("mutation", label, 2, unwritten, invalid_formats(run_index), mutated=mutated)

    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        results = list(pool.map(lambda item: run(*item), enumerate(jobs)))
    for (check, label, extra, _data, _mutated), result in zip(jobs, results):
        if check == "pass":
            print(f"{NAME}: {label}", flush=True)
            sys.stdout.write(result.stdout)
            if result.returncode:
                sys.stderr.write(result.stderr)
                return 1
        elif check == "placeholder":
            if result.returncode != 1 or extra not in result.stderr:
                raise AssertionError(f"{NAME}: {label} escaped detection\n{result.stderr}")
        elif check == "wrong":
            if result.returncode != 1 or extra not in result.stderr:
                raise AssertionError(f"{NAME}: reference '{label}' escaped detection\n{result.stderr}")
        elif result.returncode != 1 or "Compute readback differs" not in result.stderr:
            raise AssertionError(f"{NAME}: mutation '{label}' escaped detection\n{result.stderr}")
    print(f"{NAME}: {2 + len(INVALID)} dispatches, {len(WRONG) + len(INVALID) + 1} wrong references and "
          f"{len(MUTATIONS)} mutated sources checked")
    return 0


if __name__ == "__main__":
    mode, *args = sys.argv[1:] or ["controls"]
    controls()
    if mode == "controls":
        sys.exit(0)
    with tempfile.TemporaryDirectory() as tmp:
        if mode == "binary" and len(args) == 3:
            sys.exit(binary(Path(tmp), *args))
        if mode == "draw" and len(args) == 3:
            sys.exit(draw(Path(tmp), *args))
    sys.exit("usage: compute_load_format_check.py controls | binary <llvm-mc> <llvm-objcopy> <libLLVM> "
             "| draw <dxc> <renderer> <adapter>")
