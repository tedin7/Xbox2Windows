"""Compare native and translated constant loads of the x87 leaf-function profile: FLD1, FLDZ, FLDPI,
FLDL2T, FLDL2E, FLDLG2 and FLDLN2.

Usage:
    python tests/cpu/x87_constants.py asm <asm> <def>             regenerate the MASM source and exports
    python tests/cpu/x87_constants.py emit <dll> <cpp> [control]   translate (and optionally mutate) the fixture
    python tests/cpu/x87_constants.py check <harness> <native> <lifted> [controls...]

The exports, harness and guards follow tests/cpu/x87_add.py, and the other steps use the format of
tests/cpu/x87_moves.py. emit checks that every decoded x87 instruction is the form its sequence names.
The model computes each constant from an integer series and rounds it to 64 significand bits itself;
it does not read the table in tools/x87_lower.py.
"""
from pathlib import Path
import platform
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import x87_add as base  # noqa: E402  (tests/cpu is the script directory)
import x87_moves as moves  # noqa: E402
from x87_add import from_binary, m80_bytes, to_binary  # noqa: E402
from x87_div import SIZE, div  # noqa: E402
from x87_mul import mul, operand  # noqa: E402
from x87_sub import sub  # noqa: E402

INPUT, OUTPUT = base.INPUT, base.OUTPUT
NAMES = ("fld1", "fldz", "fldpi", "fldl2t", "fldl2e", "fldlg2", "fldln2")

# Constant steps are (name,); loads, stores and arithmetic use the x87_moves format.
SEQUENCES = {f"{name}_{kind}": ("none", [(name,), ("fstp", kind, 0)]) for name in NAMES for kind in ("m80", "m64", "m32")}
SEQUENCES.update({
    "fldpi_fst_m64": ("none", [("fldpi",), ("fst", "m64", 0), ("fstp", "m80", 16)]),
    # Constants as operands: x + 1, 1 / x, 0 - x, x * ln 2, pi * m64 and log2(e) * m32.
    "fld1_add": ("m80", [("fld", "m80", 0), ("fld1",), ("faddp", "pop", 1), ("fstp", "m80", 0)]),
    "fld1_divr": ("m80", [("fld", "m80", 0), ("fld1",), ("fdivrp", "pop", 1), ("fstp", "m80", 0)]),
    "fldz_subr": ("m80", [("fld", "m80", 0), ("fldz",), ("fsubrp", "pop", 1), ("fstp", "m80", 0)]),
    "fldln2_mul": ("m80", [("fld", "m80", 0), ("fldln2",), ("fmulp", "pop", 1), ("fstp", "m80", 0)]),
    "fldpi_mul_m64": ("f64", [("fldpi",), ("fmul", "mem", "m64", 0), ("fstp", "m80", 0)]),
    "fldl2e_mul_m32": ("f32", [("fldl2e",), ("fmul", "mem", "m32", 0), ("fstp", "m80", 0)]),
    # The eighth register, then every value back out.
    "fldpi_depth8": ("m64x7", [("fld", "m64", 8 * k) for k in range(7)] + [("fldpi",)]
                     + [("fstp", "m64", 8 * k) for k in range(8)]),
    "fldlg2_depth8": ("m64x7", [("fld", "m64", 8 * k) for k in range(7)] + [("fldlg2",)]
                      + [("fstp", "m64", 8 * k) for k in range(8)]),
})


# ---- Independent model -----------------------------------------------------------------------------

P = 320  # fixed-point fraction bits of the series; far beyond the 66 bits any rounding decision needs


def atan_inverse(n):
    """atan(1/n) * 2^P."""
    total, power, k = 0, (1 << P) // n, 0
    while power:
        term = power // (2 * k + 1)
        total += -term if k & 1 else term
        power //= n * n
        k += 1
    return total


def ln2():
    """ln 2 * 2^P = sum 1 / (k 2^k)."""
    return sum((1 << P) // (k << k) for k in range(1, P + 8))


def ln10():
    """ln 10 * 2^P = 3 ln 2 + ln(5/4), with ln(5/4) = 2 atanh(1/9)."""
    total, power, k = 0, (1 << P) // 9, 0
    while power:
        total += power // (2 * k + 1)
        power //= 81
        k += 1
    return 3 * ln2() + 2 * total


def exact_constants():
    """Each constant as a fixed-point integer value * 2^P."""
    pi = 16 * atan_inverse(5) - 4 * atan_inverse(239)
    l2, l10 = ln2(), ln10()
    one = 1 << P
    return {"fld1": one, "fldz": 0, "fldpi": pi, "fldl2t": (l10 << P) // l2, "fldl2e": (one << P) // l2,
            "fldlg2": (l2 << P) // l10, "fldln2": l2}


def round64(x, mode="nearest"):
    """The m80 encoding of the positive fixed-point value x * 2^-P rounded to 64 significand bits."""
    if x == 0:
        return 0, 0
    shift = x.bit_length() - 64
    sig, rest = x >> shift, x & ((1 << shift) - 1)
    if mode == "nearest" and (rest > 1 << (shift - 1) or (rest == 1 << (shift - 1) and sig & 1)):
        sig += 1
    exponent = x.bit_length() - 1 - P
    if sig >> 64:
        sig, exponent = sig >> 1, exponent + 1
    return sig, 0x3FFF + exponent


EXACT = exact_constants()
VALUES = {name: round64(x) for name, x in EXACT.items()}
FUNCTIONS = {"faddp": base.add, "fmul": mul, "fmulp": mul, "fdivrp": lambda a, b: div(b, a),
             "fsubrp": lambda a, b: sub(b, a)}


def reference(name, data):
    slots, out = [], bytearray(b"\xcc" * OUTPUT)

    def read(kind, offset):
        return int.from_bytes(data[offset:offset + SIZE[kind]], "little")

    def store(kind, offset, v):
        encoded = v[0] | v[1] << 64 if kind == "m80" else to_binary(kind, v)
        out[offset:offset + SIZE[kind]] = encoded.to_bytes(SIZE[kind], "little")

    for op, *args in SEQUENCES[name][1]:
        if op in VALUES:
            slots.append(VALUES[op])
        elif op == "fld":
            raw = read(*args)
            slots.append((raw & (2 ** 64 - 1), raw >> 64) if args[0] == "m80" else from_binary(args[0], raw))
        elif op == "fstp":
            store(*args, slots.pop())
        elif op == "fst":
            store(*args, slots[-1])
        elif args[0] == "mem":
            slots[-1] = FUNCTIONS[op](slots[-1], operand(args[1], read(args[1], args[2])))
        else:  # ("faddp" | "fmulp" | "fdivrp" | "fsubrp", "pop", i)
            k = len(slots) - 1 - args[1]
            slots[k] = FUNCTIONS[op](slots[k], slots[-1])
            slots.pop()
    assert not slots
    return bytes(out)


# Source mutations of the generated C++. Each control must mismatch.
LITERAL = re.compile(r"X87Value\{0x([0-9a-f]{16})ull, 0x([0-9a-f]{4})\};  // (fld\w+)")


def _round53(m):
    sig, se = int(m.group(1), 16), int(m.group(2), 16)
    kept, rest = sig >> 11, sig & 0x7FF
    if rest > 0x400 or (rest == 0x400 and kept & 1):
        kept += 1
    if kept >> 53:
        kept, se = kept >> 1, se + 1
    return f"X87Value{{{kept << 11:#018x}ull, {se:#06x}}};  // {m.group(3)}"


def _chop(m):
    sig, se = round64(EXACT[m.group(3)], "chop")
    return f"X87Value{{{sig:#018x}ull, {se:#06x}}};  // {m.group(3)}"


def _swap_logs(m):
    other = {"fldlg2": "fldln2", "fldln2": "fldlg2"}.get(m.group(3))
    if other is None:
        return m.group(0)
    sig, se = VALUES[other]
    return f"X87Value{{{sig:#018x}ull, {se:#06x}}};  // {m.group(3)}"


CONTROLS = {
    # The precision control applied to the constants: 53 significand bits.
    "pc53": (LITERAL, _round53),
    # Round toward zero instead of to nearest.
    "chop": (LITERAL, _chop),
    # FLDZ loads -0.
    "fldzneg": (re.compile(r"X87Value\{0x0{16}ull, 0x0000\};  // fldz"), "X87Value{0ull, 0x8000};  // fldz"),
    # The log10(2) and ln(2) entries exchanged.
    "swaplogs": (LITERAL, _swap_logs),
}


# ---- MASM generation and decoding ------------------------------------------------------------------

def asm_line(step):
    return f"    {step[0]}" if step[0] in NAMES else moves.asm_line(step)


def expected_form(step):
    return (step[0], "") if step[0] in NAMES else moves.expected_form(step)


def write_asm(asm_path, def_path):
    lines = ["; coverage: source cpu_aot_x87_constants",
             "; Generated by tests/cpu/x87_constants.py asm; edit SEQUENCES there and regenerate.",
             "; Source-owned differential fixture for the constant loads of the x87 leaf-function profile.",
             "; Each export reads an input block at RCX, writes an output block at RDX and returns with the x87",
             "; stack empty.", ".code", ""]
    for name, (_, steps) in SEQUENCES.items():
        lines += [f"{name} PROC FRAME", "    .endprolog", *(asm_line(s) for s in steps), "    ret", f"{name} ENDP", ""]
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
            decoded = [moves.decoded_form(i) for i in body if is_x87(i)]
            wanted = [expected_form(s) for s in steps]
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


# ---- Inputs ----------------------------------------------------------------------------------------

def cases():
    rng = random.Random(17)

    def random_m80():
        return 1 << 63 | rng.getrandbits(63), rng.randint(0x3FFF - 300, 0x3FFF + 300) | rng.choice((0, 0x8000))

    f64 = base.f64_bits()
    blocks = {
        # The constant-only exports read nothing; random blocks keep the input guard check meaningful.
        "none": [rng.randbytes(INPUT) for _ in range(8)],
        "m80": [m80_bytes(v) for v in base.m80_singles()] + [m80_bytes(random_m80()) for _ in range(200)],
        "f64": [b.to_bytes(8, "little") for b in f64],
        "f32": [b.to_bytes(4, "little") for b in base.f32_bits()],
        "m64x7": [b"".join(rng.choice(f64).to_bytes(8, "little") for _ in range(7)) for _ in range(300)],
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
                    print(f"{label} first mismatch: {name}(input {data[:48].hex()})\n  got      {got[:48].hex()}\n"
                          f"  expected {want[:48].hex()}")
                bad += 1
        if bad:
            raise AssertionError(f"{label}: {bad}/{len(expected)} mismatches")
    print(f"{len(expected)} cases over {len(SEQUENCES)} exports (output block and guards) matched the reference "
          "in native and translated DLLs; each export returned with control 0x027f and an empty stack")
    print(f"processor: {platform.processor()}")
    print("constants: " + ", ".join(f"{name} {sig:016x}/{se:04x}" for name, (sig, se) in VALUES.items()))
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
    rejected = {f"{name} at depth 8": (attempt(f"{fld} " * 8 + f"d9 {code:02x} " + f"{fstp} " * 9 + "c3"),
                                       "x87 stack overflow")
                for name, code in zip(("fld1", "fldl2t", "fldl2e", "fldpi", "fldlg2", "fldln2", "fldz"),
                                      range(0xE8, 0xEF))}
    rejected.update({
        "fldpi without a store": (attempt("d9 eb c3"), "x87 stack not empty at return"),
        "f2xm1": (attempt(f"{fld} d9 f0 {fstp} c3"), "unsupported x87 operation"),
        "fyl2x": (attempt(f"{fld} {fld} d9 f1 {fstp} c3"), "unsupported x87 operation"),
        "fnstenv": (attempt(f"{fld} d9 31 {fstp} c3"), "stores the tag word of a non-empty x87 stack"),
    })
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
