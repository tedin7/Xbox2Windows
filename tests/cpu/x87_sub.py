"""Compare native and translated FSUB and FSUBR forms, and the FADD forms besides FADDP, of the x87
leaf-function profile: m32 and m64 operands, ST(0), ST(i), ST(i), ST(0), FSUBP and FSUBRP ST(i), ST(0).

Usage:
    python tests/cpu/x87_sub.py asm <asm> <def>             regenerate the MASM source and exports
    python tests/cpu/x87_sub.py emit <dll> <cpp> [control]   translate (and optionally mutate) the fixture
    python tests/cpu/x87_sub.py check <harness> <native> <lifted> [controls...]

The exports, harness and guards follow tests/cpu/x87_add.py; the step format, the assembler and the
decode check of every x87 form come from tests/cpu/x87_div.py. The difference model below negates the
subtrahend in exact rational arithmetic and is independent of tools/x87_soft.h.
"""
from pathlib import Path
import platform
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import x87_add as base  # noqa: E402  (tests/cpu is the script directory)
import x87_div as forms  # noqa: E402
from x87_add import INDEFINITE, J, classify, from_binary, m80, m80_bytes, round_format, to_binary  # noqa: E402
from x87_mul import nan_result, operand  # noqa: E402

INPUT, OUTPUT, SIZE = base.INPUT, base.OUTPUT, forms.SIZE

P80 = [("fld", "m80", 0), ("fld", "m80", 16)]
SEQUENCES = {
    "sub_m80": ("pair80", P80 + [("fsubp", "pop", 1), ("fstp", "m80", 0)]),
    "sub_to_m64": ("pair80", P80 + [("fsubp", "pop", 1), ("fstp", "m64", 0)]),
    "sub_to_m32": ("pair80", P80 + [("fsubp", "pop", 1), ("fstp", "m32", 0)]),
    "subr_m80": ("pair80", P80 + [("fsubrp", "pop", 1), ("fstp", "m80", 0)]),
    "sub_st0_sti": ("pair80", P80 + [("fsub", "st0", 1), ("fstp", "m80", 0), ("fstp", "m80", 16)]),
    "sub_sti_st0": ("pair80", P80 + [("fsub", "sti", 1), ("fstp", "m80", 0), ("fstp", "m80", 16)]),
    "subr_st0_sti": ("pair80", P80 + [("fsubr", "st0", 1), ("fstp", "m80", 0), ("fstp", "m80", 16)]),
    "subr_sti_st0": ("pair80", P80 + [("fsubr", "sti", 1), ("fstp", "m80", 0), ("fstp", "m80", 16)]),
    "add_st0_sti": ("pair80", P80 + [("fadd", "st0", 1), ("fstp", "m80", 0), ("fstp", "m80", 16)]),
    "add_sti_st0": ("pair80", P80 + [("fadd", "sti", 1), ("fstp", "m80", 0), ("fstp", "m80", 16)]),
    "sub_self": ("m80", [("fld", "m80", 0), ("fsub", "st0", 0), ("fstp", "m80", 0)]),
    "add_self": ("m80", [("fld", "m80", 0), ("fadd", "st0", 0), ("fstp", "m80", 0)]),
    "sub_mem32": ("m80_f32", [("fld", "m80", 0), ("fsub", "mem", "m32", 16), ("fstp", "m80", 0)]),
    "sub_mem64": ("m80_f64", [("fld", "m80", 0), ("fsub", "mem", "m64", 16), ("fstp", "m80", 0)]),
    "subr_mem32": ("m80_f32", [("fld", "m80", 0), ("fsubr", "mem", "m32", 16), ("fstp", "m80", 0)]),
    "subr_mem64": ("m80_f64", [("fld", "m80", 0), ("fsubr", "mem", "m64", 16), ("fstp", "m80", 0)]),
    "add_mem32": ("m80_f32", [("fld", "m80", 0), ("fadd", "mem", "m32", 16), ("fstp", "m80", 0)]),
    "add_mem64": ("m80_f64", [("fld", "m80", 0), ("fadd", "mem", "m64", 16), ("fstp", "m80", 0)]),
    "sub_f32": ("pair32", [("fld", "m32", 0), ("fsub", "mem", "m32", 4), ("fstp", "m32", 0)]),
    "sub_f64": ("pair64", [("fld", "m64", 0), ("fsub", "mem", "m64", 8), ("fstp", "m64", 0)]),
    "add_f64": ("pair64", [("fld", "m64", 0), ("fadd", "mem", "m64", 8), ("fstp", "m64", 0)]),
    # b - (a - c) through FSUBP ST(2), ST(0) and then FSUBRP ST(1), ST(0).
    "sub_three": ("triple80", [("fld", "m80", 0), ("fld", "m80", 16), ("fld", "m80", 32), ("fsubp", "pop", 2),
                               ("fsubrp", "pop", 1), ("fstp", "m80", 0)]),
}

# Source mutations of the generated C++ (the helper header is part of it). Each control must mismatch.
CONTROLS = {
    # The difference rounds to 64 significand bits, as with precision control 64.
    "pc64": (base.ARITH, "x87_arith{64, 16383, 0x7ffe}"),
    # The difference rounds in the binary64 exponent range, as host double arithmetic would.
    "double": (base.ARITH, "x87_arith{53, 1023, 0x7fe}"),
    # The subtrahend is negated before the NaN rules, so a NaN subtrahend comes back with its sign flipped.
    "nansign": (re.compile(r"(x87_sub\(X87Value a, X87Value b, uint16_t cw\) \{\s+X87Value out;\s+)"
                           r"if \(x87_special_operands\(a, b, &out\)\) return out;"), r"\1"),
    # FSUBR subtracts in the FSUB order.
    "reverse": (re.compile(r"x87_subr\(X87Value a, X87Value b, uint16_t cw\) \{ return x87_sub\(b, a, cw\); \}"),
                "x87_subr(X87Value a, X87Value b, uint16_t cw) { return x87_sub(a, b, cw); }"),
    # An exact zero difference of nonzero values is -0.
    "cancel": (re.compile(r"return \{0, zero_sign\};  // x \+ \(-x\)"), "return {0, 0x8000};  // x + (-x)"),
    # Bits shifted out when aligning the smaller operand are dropped instead of kept as a sticky bit.
    "nosticky": (re.compile(r"lo \|= sticky;|lo = \(hi \| lo\) != 0;"),
                 lambda m: ";" if m.group(0).startswith("lo |=") else "lo = 0;"),
    # m32 and m64 operands are quieted before the NaN rules see them, as a plain load would.
    "memquiet": (re.compile(r"(x87_operand_f(?:32|64)\(uint(?:32|64)_t bits\) \{ return x87_from_binary\([^)]*), false\)"),
                 r"\1, true)"),
}


def write_asm(asm_path, def_path):
    lines = ["; Generated by tests/cpu/x87_sub.py asm; edit SEQUENCES there and regenerate.",
             "; Source-owned differential fixture for the FSUB, FSUBR and FADD forms of the x87 leaf-function",
             "; profile. Each export reads an input block at RCX, writes an output block at RDX and returns with",
             "; the x87 stack empty.", ".code", ""]
    for name, (_, steps) in SEQUENCES.items():
        lines += [f"{name} PROC FRAME", "    .endprolog", *(forms.asm_line(s) for s in steps), "    ret",
                  f"{name} ENDP", ""]
    lines.append("END")
    Path(asm_path).write_text("\n".join(lines) + "\n", encoding="ascii")
    Path(def_path).write_text("EXPORTS\n" + "".join(f"    {name}\n" for name in SEQUENCES), encoding="ascii")


def emit(image, output, control="lifted"):
    import pefile
    from cpu_leaf_aot import export_rva
    from cpu_ranges import function_ranges
    from ffx_version_query import INTEGER_HEADER, lower, region
    from x87_lower import X87_HEADER, is_x87

    parts = [INTEGER_HEADER, X87_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name, (_, steps) in SEQUENCES.items():
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            decoded = [forms.decoded_form(i) for i in body if is_x87(i)]
            wanted = [forms.expected_form(s) for s in steps]
            if decoded != wanted:
                raise AssertionError(f"{name} assembled as {decoded}, expected {wanted}")
            code = lower(body, ["rcx = a; rdx = b;"], {}, {}, {}, x87=True)
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b) {{\n{code}\n}}\n')
    text = "".join(parts)
    if control != "lifted":
        expression, replacement = CONTROLS[control]
        text, count = expression.subn(replacement, text)
        if not count:
            raise AssertionError(f"control {control} changed nothing")
        print(f"control {control}: {count} replacements")
    output.write_text(text, encoding="ascii")


# ---- Independent model -----------------------------------------------------------------------------

def sub(a, b):
    """FSUB a - b with control word 0x027F. A NaN operand is selected as it is (its sign kept); otherwise
    the exact difference rounds to 53 bits in the extended exponent range, and an exact zero is +0 unless
    a is -0 and b is +0."""
    ca, cb = classify(a), classify(b)
    if "unsupported" in (ca[0], cb[0]):
        return INDEFINITE
    if "nan" in (ca[0], cb[0]):
        return nan_result(a, b)
    if ca[0] == "inf" or cb[0] == "inf":
        if ca[0] == cb[0] == "inf" and ca[1] == cb[1]:
            return INDEFINITE
        return a if ca[0] == "inf" else (J, (cb[1] ^ 1) << 15 | 0x7FFF)
    if ca[0] == cb[0] == "zero":
        return 0, 0x8000 if ca[1] and not cb[1] else 0
    value = sum((-c[2] if c[1] else c[2]) * k for c, k in ((ca, 1), (cb, -1)) if c[0] == "finite")
    if value == 0:
        return 0, 0
    sign = int(value < 0)
    rounded = round_format(abs(value), 53, 16383, 0x7FFE)
    if rounded is None:
        return J, sign << 15 | 0x7FFF
    m, biased = rounded
    return m << 11, sign << 15 | biased


FUNCTIONS = {"fsub": sub, "fsubp": sub, "fsubr": lambda a, b: sub(b, a), "fsubrp": lambda a, b: sub(b, a),
             "fadd": base.add}


def reference(name, data):
    slots, out = [], bytearray(b"\xcc" * OUTPUT)

    def read(kind, offset):
        return int.from_bytes(data[offset:offset + SIZE[kind]], "little")

    for op, *args in SEQUENCES[name][1]:
        if op == "fld":
            raw = read(*args)
            slots.append((raw & (2 ** 64 - 1), raw >> 64) if args[0] == "m80" else from_binary(args[0], raw))
            continue
        if op == "fstp":
            kind, offset = args
            v = slots.pop()
            encoded = v[0] | v[1] << 64 if kind == "m80" else to_binary(kind, v)
            out[offset:offset + SIZE[kind]] = encoded.to_bytes(SIZE[kind], "little")
            continue
        f, form = FUNCTIONS[op], args[0]
        if form == "mem":
            slots[-1] = f(slots[-1], operand(args[1], read(args[1], args[2])))
        elif form == "st0":
            slots[-1] = f(slots[-1], slots[-1 - args[1]])
        else:
            slots[-1 - args[1]] = f(slots[-1 - args[1]], slots[-1])
            if form == "pop":
                slots.pop()
    assert not slots
    return bytes(out)


# ---- Inputs ----------------------------------------------------------------------------------------

def pairs80():
    """The addition pairs of tests/cpu/x87_add.py (specials, halfway sums, sticky tails, range edges,
    random) plus the same pairs with the second operand negated, so each case is also a subtraction."""
    rows = base.pairs80()
    rows += [(a, (b[0], b[1] ^ 0x8000)) for a, b in rows[len(base.special_m80()) ** 2:]]
    rows += [(m80(1), m80(1)), (m80(-3), m80(-3)), ((J - 1, 0), (J - 1, 0)), ((0, 0x8000), (0, 0)),
             ((0, 0), (0, 0x8000)), ((0, 0x8000), (0, 0x8000)), ((J, 0x7FFF), (J, 0x7FFF)), ((J, 0xFFFF), (J, 0x7FFF)),
             (m80((1 << 64) - 1, -63), m80(1)), (m80(1), m80((1 << 64) - 1, -64))]
    return rows


def cases():
    f64, f32 = base.f64_bits(), base.f32_bits()
    specials = base.special_m80()
    rng = random.Random(11)

    def random_m80():
        return J | rng.getrandbits(63), rng.randint(0x3FFF - 70, 0x3FFF + 70) | rng.choice((0, 0x8000))

    def m80_block(*values):
        return b"".join(m80_bytes(v) for v in values)

    blocks = {
        "pair80": [m80_block(a, b) for a, b in pairs80()],
        "m80": [m80_bytes(v) for v in base.m80_singles()] + [m80_bytes(random_m80()) for _ in range(200)],
        "m80_f32": [m80_bytes(v) + b.to_bytes(4, "little") for v in specials for b in f32[:17]]
                   + [m80_bytes(random_m80()) + rng.getrandbits(32).to_bytes(4, "little") for _ in range(300)],
        "m80_f64": [m80_bytes(v) + b.to_bytes(8, "little") for v in specials for b in f64[:18]]
                   + [m80_bytes(random_m80()) + rng.getrandbits(64).to_bytes(8, "little") for _ in range(300)],
        "pair32": [a.to_bytes(4, "little") + b.to_bytes(4, "little") for a in f32[:17] for b in f32[:17]]
                  + [rng.getrandbits(32).to_bytes(4, "little") + rng.getrandbits(32).to_bytes(4, "little")
                     for _ in range(300)],
        "pair64": [a.to_bytes(8, "little") + b.to_bytes(8, "little") for a in f64[:18] for b in f64[:18]]
                  + [rng.getrandbits(64).to_bytes(8, "little") + rng.getrandbits(64).to_bytes(8, "little")
                     for _ in range(300)],
        "triple80": [m80_block(*rng.sample(specials, 3)) for _ in range(200)]
                    + [m80_block(random_m80(), random_m80(), random_m80()) for _ in range(300)],
    }
    for name, (kind, _) in SEQUENCES.items():
        for block in blocks[kind]:
            yield name, block.ljust(INPUT, b"\0")


# ---- Check -----------------------------------------------------------------------------------------

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
                if not bad:
                    print(f"{label} first mismatch: {name}(input {data[:48].hex()})\n  got      {got[:16].hex()}\n"
                          f"  expected {want[:16].hex()}")
                bad += 1
        if bad:
            raise AssertionError(f"{label}: {bad}/{len(expected)} mismatches")
    print(f"{len(expected)} cases over {len(SEQUENCES)} exports (output block and guards) matched the reference "
          "in native and translated DLLs; each export returned with control 0x027f and an empty stack")
    print(f"processor: {platform.processor()}")
    first = {}
    for name, data, want in expected:
        first.setdefault(name, (data, want))
    for label, path in (("native", native_path), ("lifted", lifted_path)):
        run = base.Runner(harness_path, path)
        failed = [name for name, (data, want) in first.items() if not base.preserved(run, "x87_call", name, data, want)]
        if failed:
            raise AssertionError(f"x87_call lost a live caller value or control word around {label} {failed}")
    print(f"x87_call kept a live caller ST(0) value and control word around all {len(first)} native and "
          "translated exports")
    for path in control_paths:
        control = Path(path).stem.rsplit("_", 1)[-1]
        run = base.Runner(harness_path, path)
        bad = sum(1 for name, data, want in expected if run(name, data)[0] != want)
        print(f"control {control}: {bad}/{len(expected)} mismatches")
        if not bad:
            raise AssertionError(f"control {control} matched every case")
    check_rejections()


def check_rejections():
    from iced_x86 import Decoder
    from ffx_version_query import lower

    def attempt(hex_bytes):
        body = list(Decoder(64, bytes.fromhex(hex_bytes), ip=0x1000))
        try:
            lower(body, [], {}, {}, {}, x87=True)
        except ValueError as exc:
            return str(exc)
        return None

    fld, fstp = "db 29", "db 3a"  # fld tbyte ptr [rcx]; fstp tbyte ptr [rdx]
    rejected = {
        "fsubp st(0), st": (attempt(f"{fld} {fld} de e8 {fstp} {fstp} c3"), "unsupported x87 operation"),
        "fsubrp st(0), st": (attempt(f"{fld} {fld} de e0 {fstp} {fstp} c3"), "unsupported x87 operation"),
        "fsub m32 on an empty stack": (attempt(f"d8 21 {fld} {fstp} c3"), "x87 stack underflow"),
        "fadd st, st(1) with one value": (attempt(f"{fld} d8 c1 {fstp} c3"), "x87 stack underflow"),
        "fsubr st(2), st with two values": (attempt(f"{fld} {fld} dc e2 {fstp} {fstp} c3"), "x87 stack underflow"),
        "fldenv": (attempt(f"{fld} d9 21 {fstp} c3"), "unsupported x87 operation"),
        "fdecstp": (attempt(f"{fld} d9 f6 {fstp} c3"), "unsupported x87 operation"),
        "fxsave": (attempt(f"{fld} 0f ae 01 {fstp} c3"), "unsupported x87 operation"),
        "fnstenv": (attempt(f"{fld} {fld} d9 31 {fstp} {fstp} c3"), "stores the tag word of a non-empty x87 stack"),
        "fcos": (attempt(f"{fld} d9 ff {fstp} c3"), "unsupported x87 operation"),
    }
    for case, (message, needle) in rejected.items():
        if message is None or needle not in message:
            raise AssertionError(f"{case}: expected a rejection containing {needle!r}, got {message!r}")
    print(f"rejected {len(rejected)} sequences outside the profile: " + "; ".join(rejected))


if __name__ == "__main__":
    mode, first, second, *rest = sys.argv[1:]
    if mode == "asm":
        write_asm(first, second)
    elif mode == "emit":
        emit(Path(first), Path(second), *rest)
    elif mode == "check":
        check(first, second, *rest)
    else:
        raise ValueError(mode)
