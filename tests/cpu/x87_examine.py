"""Compare native and translated FTST, FXAM, FICOM, FICOMP, FNOP, FNCLEX, FCLEX and the 8087/287 controls
FENI, FNENI, FDISI, FNDISI and FSETPM, FFREEP ST(0), FFREE of an empty register, FSTSW AX and the pair FFREE ST(0),
FINCSTP, in the x87 profile.

Each export runs through the native harness (control word 0x027F, empty stack) and writes C0, C2 and C3
after each status-word writer, and any stored m80 value, to its output block. An independent model gives
the expected block: comparisons use exact rationals, and FXAM classifies the 80-bit encoding (unsupported,
NaN, normal, infinity, zero, empty, denormal); FFREEP ST(0) gives the physical TOP after each pop and leaves the
values below it unchanged, FFREE of an empty register keeps TOP and every value, FSTSW AX (WAIT and
FNSTSW AX) gives TOP from the depth, and FFREE ST(0), FINCSTP pops as FFREEP ST(0) does. Eleven mutated
translations must each mismatch, and translation must reject the sequences in check_rejections.
"""
from fractions import Fraction
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import x87_add as base  # noqa: E402  (tests/cpu is the script directory)

EXPORTS = ("ftst_fxam", "ficom_m16", "ficom_m32", "fnop_fclex", "legacy_controls", "ffreep_top", "ffree_empty",
           "fstsw_top", "ffree_fincstp")
INPUT, OUTPUT = base.INPUT, base.OUTPUT
J = 1 << 63
# The TOP literal of the FNSTSW AX right after the FINCSTP of an FFREE ST(0), FINCSTP pair.
PAIR_TOP = re.compile(r"(;  // fincstp\n\s+block_\w+: rax = \(rax & ~0xffffull\) \| static_cast<uint16_t>\(x87_sw \| )"
                      r"(0x[0-9a-f]{4})\)")
# Mutations of the lifted source that the check requires to mismatch: (pattern, replacement).
CONTROLS = {
    "normal": (re.compile(re.escape("return x87_exp(v) == 0 ? 0x4400 : 0x0400;")), "return 0x0400;"),
    "unsigned": (re.compile(r"static_cast<int(16|32)_t>\(load_always<uint\1_t>"), r"static_cast<uint\1_t>(load_always<uint\1_t>"),
    "positive": (re.compile(re.escape("x87_compare(x87_slot[0], X87Value{0, 0})")),
                 "x87_compare(x87_slot[0], X87Value{0x8000000000000000ull, 0x3fff})"),
    # A legacy control that changed ST(0) instead of acting as FNOP.
    "legacy": (re.compile(re.escape(";  // fnsetpm")), "x87_slot[0].se ^= 0x8000;  // fnsetpm"),
    # FFREEP that also overwrote the deepest register (slot 0, the input at depth 3 and 2) with the next one.
    "ffreep": (re.compile(re.escape(";  // ffreep st(0)")), "x87_slot[0] = x87_slot[1];  // ffreep st(0)"),
    # A TOP that FFREEP did not move: the status word after the first FFREEP keeps TOP 5. The literal also
    # matches the depth-2 status read of fstsw_top.
    "fftop": (re.compile(re.escape("x87_sw | 0x3000);  // fnstsw ax, top 6")), "x87_sw | 0x2800);  // fnstsw ax, top 6"),
    # FFREE ST(7) that freed ST(0) instead: at depth 7 the 1.0 in ST(0) (slot 6) becomes zero.
    "ffree": (re.compile(re.escape(";  // ffree st(7)")), "x87_slot[6] = X87Value{0, 0};  // ffree st(7)"),
    # FFREE that moved TOP as a pop does: at depth 1 the status word reads TOP 0 instead of 7. The literal also
    # matches the depth-1 status reads of ffreep_top and fstsw_top, so all three exports mismatch.
    "freetop": (re.compile(re.escape("x87_sw | 0x3800);  // fnstsw ax, top 7")), "x87_sw | 0x0000);  // fnstsw ax, top 7"),
    # FSTSW AX (the FNSTSW AX right after a WAIT) that reads TOP 0 at every depth.
    "fstsw": (re.compile(r"(;  // wait\n\s+block_\w+: rax = \(rax & ~0xffffull\) \| static_cast<uint16_t>\(x87_sw \| )"
                         r"0x[0-9a-f]{4}\)"), r"\g<1>0x0000)"),
    # FFREE ST(0), FINCSTP that did not pop (TOP as before the pair) or popped twice, seen by the status read right
    # after each pair.
    "nopop": (PAIR_TOP, lambda m: f"{m[1]}{(int(m[2], 16) - 0x0800) & 0x3800:#06x})"),
    "twopop": (PAIR_TOP, lambda m: f"{m[1]}{(int(m[2], 16) + 0x0800) & 0x3800:#06x})"),
}


def emit(image, output, control="lifted"):
    import pefile
    from cpu_leaf_aot import export_rva
    from cpu_ranges import function_ranges
    from ffx_version_query import INTEGER_HEADER, lower, region
    from x87_lower import X87_HEADER

    parts = [INTEGER_HEADER, X87_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            code = lower(region(pe, *bounds), ["rcx = a; rdx = b;"], {}, {}, {}, x87=True)
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b) {{\n{code}\n}}\n')
    text = "".join(parts)
    if control != "lifted":
        pattern, replacement = CONTROLS[control]
        text, count = pattern.subn(replacement, text)
        if not count:
            raise AssertionError(f"control {control} changed nothing")
    output.write_text(text, encoding="ascii")


# ---- Independent model -----------------------------------------------------------------------------

def kind(sig, se):
    e = se & 0x7FFF
    if e and not sig & J:
        return "unsupported"  # unnormal, pseudo-infinity, pseudo-NaN
    if e == 0x7FFF:
        return "infinity" if sig == J else "nan"
    if e == 0:
        return "zero" if sig == 0 else "denormal"  # a pseudo-denormal is a denormal
    return "normal"


def exact(sig, se):
    """The value of a finite encoding as a rational; a pseudo-denormal uses exponent 1 like a denormal."""
    e = se & 0x7FFF
    v = Fraction(sig) * Fraction(2) ** ((e or 1) - 16383 - 63)
    return -v if se >> 15 else v


def compare(sig, se, other):
    """C0, C2 and C3 of comparing the encoding with a rational."""
    k = kind(sig, se)
    if k in ("unsupported", "nan"):
        return (1, 1, 1)
    if k == "infinity":
        return (1, 0, 0) if se >> 15 else (0, 0, 0)
    v = exact(sig, se)
    return (1, 0, 0) if v < other else (0, 0, 1) if v == other else (0, 0, 0)


EXAMINE = {"unsupported": (0, 0, 0), "nan": (1, 0, 0), "normal": (0, 1, 0), "infinity": (1, 1, 0),
           "zero": (0, 0, 1), "denormal": (0, 1, 1)}
EMPTY = (1, 0, 1)


def reference(name, data):
    sig, se = struct.unpack_from("<QH", data, 0)
    m16, = struct.unpack_from("<h", data, 16)
    m32, = struct.unpack_from("<i", data, 24)
    out = bytearray(b"\xcc" * OUTPUT)
    if name == "ftst_fxam":
        out[0:3] = bytes(compare(sig, se, Fraction(0)))
        out[3:6] = bytes(EXAMINE[kind(sig, se)])
        out[6:9] = bytes(EMPTY)
        out[16:26] = data[0:10]
    elif name in ("ficom_m16", "ficom_m32"):
        bits = compare(sig, se, Fraction(m16 if name == "ficom_m16" else m32))
        out[0:3] = out[3:6] = bytes(bits)
    elif name == "ffreep_top":  # TOP 5 at depth 3, then 6, 7 and 0 after each FFREEP ST(0)
        out[0:4] = bytes((5 << 3, 6 << 3, 7 << 3, 0))
        out[16:26] = struct.pack("<QH", J, 0x3FFF)
        out[32:42] = data[0:10]
    elif name == "ffree_empty":  # TOP 0, 7 and 0; 0.0, 1.0 and the input stored from depth 8
        out[0:3] = bytes((0, 7 << 3, 0))
        out[16:26] = struct.pack("<QH", 0, 0)
        out[26:36] = struct.pack("<QH", J, 0x3FFF)
        out[40:50] = data[0:10]
    elif name == "ffree_fincstp":  # TOP 0, 7, 1 and 0 at depths 0, 1, 7 and 8; 1.0, 0.0, 1.0 and the input stored
        out[0:4] = bytes((0, 7 << 3, 1 << 3, 0))
        out[16:26] = struct.pack("<QH", J, 0x3FFF)
        out[26:36] = struct.pack("<QH", 0, 0)
        out[36:46] = struct.pack("<QH", J, 0x3FFF)
        out[46:56] = data[0:10]
    elif name == "fstsw_top":  # TOP 0, 7 and 6 at depths 0, 1 and 2; the input stored from depth 1
        out[0:3] = bytes((0, 7 << 3, 6 << 3))
        out[16:26] = data[0:10]
    else:
        out[16:26] = data[0:10]
    return bytes(out)


def values():
    """m80 encodings of every FXAM class and of comparisons with zero and with the integers below."""
    ones = (1 << 63) - 1
    v = [(0, 0), (0, 0x8000), (1, 0), (1, 0x8000), (ones, 0), (J, 0), (J | 5, 0x8000),  # zeros, denormals
         (J, 0x3FFF), (J, 0xBFFF), (J | 1 << 62, 0x4000), (J | 1 << 61, 0xC000), (J, 1), (J | ones, 0x7FFE),
         (J, 0x7FFF), (J, 0xFFFF), (J | 1 << 62, 0x7FFF), (J | 1, 0x7FFF), (J | 1 << 62 | 7, 0xFFFF),  # inf, NaNs
         (0, 0x3FFF), (1 << 62, 0x4000), (0, 0x7FFF), (1 << 62, 0x7FFF), (5, 0xFFFF)]  # unsupported
    for n in (3, -3, 32767, -32768, 2147483647, -2147483648, 1 << 31, 12345):
        m = abs(n)
        shift = m.bit_length() - 1
        v.append(((m << (63 - shift)) & ((1 << 64) - 1), (0x8000 if n < 0 else 0) | (0x3FFF + shift)))
    v += [(J | 1 << 62, 0x4000 + 0x8000), (J | 1 << 61, 0x4000), (J | 1, 0x401D)]  # -3, 2.5, just above 2^30
    return v


INTEGERS = (0, 1, -1, 3, -3, 2, 32767, -32768, 12345)
INTEGERS32 = (*INTEGERS, 2147483647, -2147483648, 1 << 30)


def cases():
    for sig, se in values():
        m80 = struct.pack("<QH", sig, se)
        for name in ("ftst_fxam", "fnop_fclex", "legacy_controls", "ffreep_top", "ffree_empty", "fstsw_top",
                     "ffree_fincstp"):
            yield name, (m80 + b"\0" * 6 + struct.pack("<h", 7) + b"\0" * 6 + struct.pack("<i", 9)).ljust(INPUT, b"\0")
        for n in INTEGERS:
            yield "ficom_m16", (m80 + b"\0" * 6 + struct.pack("<h", n)).ljust(INPUT, b"\0")
        for n in INTEGERS32:
            yield "ficom_m32", (m80 + b"\0" * 14 + struct.pack("<i", n)).ljust(INPUT, b"\0")


def check_rejections():
    """Translation must reject an underflowing FTST, FICOMP or FFREEP, a reader of C0 after FNOP, FNCLEX, FNENI,
    FNSETPM, FFREEP or FFREE, or after FTST and FSTSW AX, and FFREE of an occupied register, FFREEP ST(i) for
    i > 0, FINCSTP and FDECSTP. Of the pair FFREE ST(0), FINCSTP it rejects a reader of C0 or C1 after it, the
    pair on an empty stack, FFREE ST(1) or FNOP in place of FFREE ST(0) or between the two, the reverse order,
    and a branch to the FINCSTP."""
    from iced_x86 import Decoder
    from ffx_version_query import lower

    # fld1; <op>; fnstsw ax; test ah, 1; jne +0; fstp st(0); mov eax, 0; ret
    def sequence(op):
        return f"d9e8{op}dfe0f6c4017500ddd8b800000000c3"

    accepted = {"ftst then a reader of C0": sequence("d9e4"),
                # fld1; fld1; test eax, eax; je to the ffree; ffree st(0); fincstp; fstp st(0); ret
                "a branch to ffree st(0), fincstp": "d9e8d9e885c07400ddc0d9f7ddd8c3"}
    rejected = {
        "ftst on an empty stack": ("d9e4c3", "underflow"),
        "ficomp on an empty stack": ("de19c3", "underflow"),
        "fnop then a reader of C0": (sequence("d9d0"), "not known"),
        "fnclex then a reader of C0": (sequence("dbe2"), "not known"),
        "fneni then a reader of C0": (sequence("dbe0"), "not known"),
        "fnsetpm then a reader of C0": (sequence("dbe4"), "not known"),
        "ffreep on an empty stack": ("dfc0c3", "underflow"),
        # fld1; fld1; ffreep st(0); then the reader of C0 of sequence()
        "ffreep then a reader of C0": ("d9e8" + sequence("dfc0"), "not known"),
        # fld1; fld1; ffreep st(0); fnstsw ax; test ah, 2 (C1); jne +0; fstp st(0); mov eax, 0; ret
        "ffreep then a reader of C1": ("d9e8d9e8dfc0dfe0f6c4027500ddd8b800000000c3", "not known"),
        "ffreep st(2)": ("d9e8d9e8d9e8dfc2ddd8ddd8c3", "unsupported x87 operand"),
        "ffree st(0)": ("d9e8ddc0ddd8c3", "frees an occupied register"),
        # fld1; ftst (C0 known, see the accepted case); ffree st(1) (empty); then the reader of C0 of sequence()
        "ffree of an empty register then a reader of C0": (sequence("d9e4ddc1"), "not known"),
        "ffree st(1)": ("d9e8d9e8ddc1ddd8ddd8c3", "frees an occupied register"),
        "fincstp": ("d9e8d9f7ddd8c3", "unsupported x87 operation"),
        "fdecstp": ("d9e8d9f6ddd8c3", "unsupported x87 operation"),
        # fld1; fld1; ftst; ffree st(0); fincstp; then the reader of C0 of sequence()
        "ffree st(0), fincstp after ftst then a reader of C0": (sequence("d9e8d9e4ddc0d9f7"), "not known"),
        # fld1; fld1; ffree st(0); fincstp; fnstsw ax; test ah, 2 (C1); jne +0; fstp st(0); mov eax, 0; ret
        "ffree st(0), fincstp then a reader of C1": ("d9e8d9e8ddc0d9f7dfe0f6c4027500ddd8b800000000c3", "not known"),
        "ffree st(0), fincstp on an empty stack": ("ddc0d9f7c3", "underflow"),
        "ffree st(1), fincstp": ("d9e8d9e8ddc1d9f7ddd8c3", "frees an occupied register"),
        "ffree st(0), fnop, fincstp": ("d9e8ddc0d9d0d9f7c3", "frees an occupied register"),
        "fincstp, ffree st(0)": ("d9e8d9f7ddc0c3", "unsupported x87 operation"),
        # fld1; fld1; test eax, eax; je to the fincstp; ffree st(0); fincstp; fstp st(0); ret
        "a branch into ffree st(0), fincstp": ("d9e8d9e885c07402ddc0d9f7ddd8c3", "branch into the FFREE ST(0), FINCSTP"),
        # The accepted ftst case with fstsw ax (wait; fnstsw ax): the wait leaves C0 undefined.
        "fstsw ax after ftst then a reader of C0": (sequence("d9e49b"), "not known"),
    }
    for label, code in accepted.items():
        lower(list(Decoder(64, bytes.fromhex(code))), [], {}, {}, {}, x87=True)
    for label, (code, needle) in rejected.items():
        try:
            lower(list(Decoder(64, bytes.fromhex(code))), [], {}, {}, {}, x87=True)
        except ValueError as error:
            if needle not in str(error):
                raise AssertionError(f"{label}: expected a rejection containing {needle!r}, got {error}")
            continue
        raise AssertionError(f"translation accepted {label}")
    print(f"accepted {len(accepted)} and rejected {len(rejected)} sequences: " + "; ".join(rejected))


def check(harness_path, native_path, lifted_path, *control_paths):
    expected = [(name, data, reference(name, data)) for name, data in cases()]
    for label, path in (("native", native_path), ("lifted", lifted_path)):
        run = base.Runner(harness_path, path)
        bad = 0
        for name, data, want in expected:
            got, state = run(name, data)
            if not base.entry_state_ok(state):
                raise AssertionError(f"{label} {name} left state {state:#014x}; the profile requires control "
                                     "0x027f and an empty stack at TOP 0")
            if got != want:
                if bad < 3:
                    print(f"{label} mismatch: {name}(input {data[:32].hex()})\n  got      {got[:26].hex()}\n"
                          f"  expected {want[:26].hex()}")
                bad += 1
        if bad:
            raise AssertionError(f"{label}: {bad}/{len(expected)} mismatches")
    classes = sorted({kind(*struct.unpack_from('<QH', data, 0)) for _, data, _ in expected})
    print(f"{len(expected)} cases over {len(EXPORTS)} exports matched the model in native and translated DLLs; "
          f"FXAM classes {classes} and empty; each export returned with control 0x027f and an empty stack")
    for path in control_paths:
        control = Path(path).stem.rsplit("_", 1)[-1]
        run = base.Runner(harness_path, path)
        bad = sum(1 for name, data, want in expected if run(name, data)[0] != want)
        print(f"control {control}: {bad}/{len(expected)} mismatches")
        if not bad:
            raise AssertionError(f"control {control} matched every case")
    check_rejections()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "emit":
        emit(Path(sys.argv[2]), Path(sys.argv[3]), *sys.argv[4:])
    elif mode == "check":
        check(*sys.argv[2:])
    else:
        raise ValueError(mode)
