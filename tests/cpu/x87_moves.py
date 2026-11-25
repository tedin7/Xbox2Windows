"""Compare native and translated register data movement of the x87 leaf-function profile: FLD, FST and
FSTP with ST(i), FST with m32 and m64 operands, FXCH ST(i), FCHS and FABS.

Usage:
    python tests/cpu/x87_moves.py asm <asm> <def>             regenerate the MASM source and exports
    python tests/cpu/x87_moves.py emit <dll> <cpp> [control]   translate (and optionally mutate) the fixture
    python tests/cpu/x87_moves.py check <harness> <native> <lifted> [controls...]

The exports, harness and guards follow tests/cpu/x87_add.py, and the arithmetic steps use the format of
tests/cpu/x87_div.py. emit checks that every decoded x87 instruction is the form its sequence names.
The model below moves raw 80-bit encodings and is independent of tools/x87_soft.h.
"""
from pathlib import Path
import platform
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import x87_add as base  # noqa: E402  (tests/cpu is the script directory)
import x87_div as forms  # noqa: E402
from x87_add import J, from_binary, m80_bytes, to_binary  # noqa: E402
from x87_mul import mul  # noqa: E402
from x87_sub import sub  # noqa: E402

INPUT, OUTPUT, SIZE = base.INPUT, base.OUTPUT, forms.SIZE

# Data-movement steps: ("fld_st", i), ("fst_st", i), ("fstp_st", i), ("fxch", i), ("fchs",), ("fabs",)
# and ("fst", kind, offset) for FST m32/m64. Loads, FSTP to memory and arithmetic use the x87_div format.
P80 = [("fld", "m80", 0), ("fld", "m80", 16)]
T80 = P80 + [("fld", "m80", 32)]
OUT2 = [("fstp", "m80", 0), ("fstp", "m80", 16)]
OUT3 = OUT2 + [("fstp", "m80", 32)]
LOAD8 = [("fld", "m64", 8 * k) for k in range(8)]
STORE8 = [("fstp", "m64", 8 * k) for k in range(8)]
SEQUENCES = {
    "fld_st0": ("m80", [("fld", "m80", 0), ("fld_st", 0), ("faddp", "pop", 1), ("fstp", "m80", 0)]),
    "fld_st1": ("pair80", P80 + [("fld_st", 1)] + OUT3),
    "fld_st2": ("triple80", T80 + [("fld_st", 2)] + OUT3 + [("fstp", "m80", 48)]),
    "fld_st6_depth8": ("m64x8", LOAD8[:7] + [("fld_st", 6)] + STORE8),
    "fst_st1": ("pair80", P80 + [("fst_st", 1)] + OUT2),
    "fst_st2": ("triple80", T80 + [("fst_st", 2)] + OUT3),
    "fstp_st1": ("pair80", P80 + [("fstp_st", 1), ("fstp", "m80", 0)]),
    "fstp_st0": ("pair80", P80 + [("fstp_st", 0), ("fstp", "m80", 0)]),
    "fstp_st2": ("triple80", T80 + [("fstp_st", 2)] + OUT2),
    "fxch_st0": ("m80", [("fld", "m80", 0), ("fxch", 0), ("fstp", "m80", 0)]),
    "fxch_st1": ("pair80", P80 + [("fxch", 1)] + OUT2),
    "fxch_st2": ("triple80", T80 + [("fxch", 2)] + OUT3),
    "fxch_st7_depth8": ("m64x8", LOAD8 + [("fxch", 7)] + STORE8),
    "fchs_m80": ("m80", [("fld", "m80", 0), ("fchs",), ("fstp", "m80", 0)]),
    "fchs_m64": ("m80", [("fld", "m80", 0), ("fchs",), ("fstp", "m64", 0)]),
    "fchs_m32": ("m80", [("fld", "m80", 0), ("fchs",), ("fstp", "m32", 0)]),
    "fabs_m80": ("m80", [("fld", "m80", 0), ("fabs",), ("fstp", "m80", 0)]),
    "fabs_m64": ("m80", [("fld", "m80", 0), ("fabs",), ("fstp", "m64", 0)]),
    "fabs_m32": ("m80", [("fld", "m80", 0), ("fabs",), ("fstp", "m32", 0)]),
    "chs_abs_chs": ("m80", [("fld", "m80", 0), ("fchs",), ("fabs",), ("fchs",), ("fstp", "m80", 0)]),
    # FST stores ST(0) rounded to the destination and leaves the register unchanged.
    "fst_m32": ("m80", [("fld", "m80", 0), ("fst", "m32", 0), ("fstp", "m80", 16)]),
    "fst_m64": ("m80", [("fld", "m80", 0), ("fst", "m64", 0), ("fstp", "m80", 16)]),
    "fst_m64_m32": ("m80", [("fld", "m80", 0), ("fst", "m64", 0), ("fst", "m32", 8), ("fstp", "m80", 16)]),
    # Movement combined with arithmetic: a + (-b), |a| * |b| and b - a through FXCH.
    "chs_add": ("pair80", P80 + [("fchs",), ("faddp", "pop", 1), ("fstp", "m80", 0)]),
    "abs_mul": ("pair80", [("fld", "m80", 0), ("fabs",), ("fld", "m80", 16), ("fabs",), ("fmulp", "pop", 1),
                           ("fstp", "m80", 0)]),
    "fxch_sub": ("pair80", P80 + [("fxch", 1), ("fsubp", "pop", 1), ("fstp", "m80", 0)]),
}

# Source mutations of the generated C++ (the helper header is part of it). Each control must mismatch.
CONTROLS = {
    # FCHS as 0 - x, as a compiler negation through arithmetic would be: +0 stays +0, NaNs are quieted and
    # a 64-bit significand rounds to 53 bits.
    "chssub": (re.compile(r"x87_chs\(X87Value v\) \{ v\.se \^= 0x8000; return v; \}"),
               "x87_chs(X87Value v) { return x87_sub({0, 0}, v, 0x027f); }"),
    # FABS as x < 0 ? -x : x: -0 and negative NaNs keep their sign.
    "absnan": (re.compile(r"x87_abs\(X87Value v\) \{ v\.se &= 0x7fff; return v; \}"),
               "x87_abs(X87Value v) { if (x87_sign(v) && !x87_is_zero(v) && !x87_is_nan(v)) v.se &= 0x7fff; return v; }"),
    # FXCH does nothing.
    "fxchnone": (re.compile(r"\{ const X87Value t = [^}]*\}  // fxch"), ";"),
    # FLD ST(i) always copies ST(0).
    "fldtop": (re.compile(r"x87_slot\[(\d)\] = x87_slot\[\d\];  // fld st"),
               lambda m: f"x87_slot[{m.group(1)}] = x87_slot[{int(m.group(1)) - 1}];  // fld st"),
    # FSTP ST(i) only pops: the copy into ST(i) is dropped.
    "fstpdrop": (re.compile(r"x87_slot\[\d\] = x87_slot\[\d\];  // fstp st"), ";  // fstp st"),
    # FST m32/m64 also writes the rounded value back into ST(0).
    "fstwrite": (re.compile(r"store<uint(32|64)_t>\(a, x87_to_f\1\(x87_slot\[(\d)\], x87_cw\)\); \}  // fst m"),
                 r"store<uint\1_t>(a, x87_to_f\1(x87_slot[\2], x87_cw)); "
                 r"x87_slot[\2] = x87_from_f\1(x87_to_f\1(x87_slot[\2], x87_cw)); }  // fst m"),
    # Register copies quiet a signaling NaN, as a conversion would.
    "copyquiet": (re.compile(r"x87_slot\[(\d)\] = x87_slot\[(\d)\];  // (fld|fst|fstp) st"),
                  lambda m: f"x87_slot[{m.group(1)}] = x87_is_nan(x87_slot[{m.group(2)}]) ? "
                            f"x87_quiet(x87_slot[{m.group(2)}]) : x87_slot[{m.group(2)}];  // {m.group(3)} st"),
}


# ---- MASM generation and decoding ------------------------------------------------------------------

def asm_line(step):
    op, *args = step
    if op in ("fld_st", "fst_st", "fstp_st"):
        return f"    {op[:-3]} st({args[0]})"
    if op == "fxch":
        return f"    fxch st({args[0]})"
    if op in ("fchs", "fabs"):
        return f"    {op}"
    if op == "fst":
        return f"    fst {base.PTR[args[0]]} ptr [rdx+{args[1]}]"
    return forms.asm_line(step)


def expected_form(step):
    op, *args = step
    if op in ("fld_st", "fst_st", "fstp_st"):
        return op[:-3], f"st{args[0]}"
    if op == "fxch":
        return "fxch", f"st0,st{args[0]}"
    if op in ("fchs", "fabs"):
        return op, ""
    if op == "fst":
        return "fst", args[0]
    return forms.expected_form(step)


def decoded_form(i):
    from iced_x86 import Mnemonic as M, OpKind as K, Register as R

    if i.op_count == 0:
        return next(name for name, value in vars(M).items() if value == i.mnemonic).lower(), ""
    if i.op_count == 1 and i.op0_kind == K.REGISTER:
        return next(name for name, value in vars(M).items() if value == i.mnemonic).lower(), f"st{i.op0_register - R.ST0}"
    return forms.decoded_form(i)


def write_asm(asm_path, def_path):
    lines = ["; Generated by tests/cpu/x87_moves.py asm; edit SEQUENCES there and regenerate.",
             "; Source-owned differential fixture for the register data movement of the x87 leaf-function",
             "; profile. Each export reads an input block at RCX, writes an output block at RDX and returns with",
             "; the x87 stack empty.", ".code", ""]
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
            decoded = [decoded_form(i) for i in body if is_x87(i)]
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


# ---- Independent model -----------------------------------------------------------------------------

FUNCTIONS = {"faddp": base.add, "fmulp": mul, "fsubp": sub}


def reference(name, data):
    slots, out = [], bytearray(b"\xcc" * OUTPUT)

    def read(kind, offset):
        return int.from_bytes(data[offset:offset + SIZE[kind]], "little")

    def store(kind, offset, v):
        encoded = v[0] | v[1] << 64 if kind == "m80" else to_binary(kind, v)
        out[offset:offset + SIZE[kind]] = encoded.to_bytes(SIZE[kind], "little")

    for op, *args in SEQUENCES[name][1]:
        if op == "fld":
            raw = read(*args)
            slots.append((raw & (2 ** 64 - 1), raw >> 64) if args[0] == "m80" else from_binary(args[0], raw))
        elif op == "fstp":
            store(*args, slots.pop())
        elif op == "fst":
            store(*args, slots[-1])
        elif op == "fld_st":
            slots.append(slots[-1 - args[0]])
        elif op in ("fst_st", "fstp_st"):
            slots[-1 - args[0]] = slots[-1]
            if op == "fstp_st":
                slots.pop()
        elif op == "fxch":
            slots[-1], slots[-1 - args[0]] = slots[-1 - args[0]], slots[-1]
        elif op == "fchs":
            slots[-1] = (slots[-1][0], slots[-1][1] ^ 0x8000)
        elif op == "fabs":
            slots[-1] = (slots[-1][0], slots[-1][1] & 0x7FFF)
        else:  # ("faddp" | "fmulp" | "fsubp", "pop", i)
            k = len(slots) - 1 - args[1]
            slots[k] = FUNCTIONS[op](slots[k], slots[-1])
            slots.pop()
    assert not slots
    return bytes(out)


# ---- Inputs ----------------------------------------------------------------------------------------

def cases():
    f64 = base.f64_bits()
    specials = base.special_m80()
    rng = random.Random(13)

    def random_m80():
        return J | rng.getrandbits(63), rng.randint(0x3FFF - 300, 0x3FFF + 300) | rng.choice((0, 0x8000))

    def m80_block(*values):
        return b"".join(m80_bytes(v) for v in values)

    blocks = {
        "m80": [m80_bytes(v) for v in base.m80_singles()] + [m80_bytes(random_m80()) for _ in range(200)],
        "pair80": [m80_block(a, b) for a in specials for b in specials]
                  + [m80_block(random_m80(), random_m80()) for _ in range(300)],
        "triple80": [m80_block(*rng.sample(specials, 3)) for _ in range(300)]
                    + [m80_block(random_m80(), random_m80(), random_m80()) for _ in range(200)],
        "m64x8": [b"".join(rng.choice(f64).to_bytes(8, "little") for _ in range(8)) for _ in range(300)],
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
        "fld st(1) with one value": (attempt(f"{fld} d9 c1 {fstp} {fstp} c3"), "x87 stack underflow"),
        "fld st(0) at depth 8": (attempt(f"{fld} " * 8 + "d9 c0 " + f"{fstp} " * 9 + "c3"), "x87 stack overflow"),
        "fst st(2) with two values": (attempt(f"{fld} {fld} dd d2 {fstp} {fstp} c3"), "x87 stack underflow"),
        "fstp st(0) on an empty stack": (attempt("dd d8 c3"), "x87 stack underflow"),
        "fxch st(1) with one value": (attempt(f"{fld} d9 c9 {fstp} c3"), "x87 stack underflow"),
        "fchs on an empty stack": (attempt("d9 e0 c3"), "x87 stack underflow"),
        "fabs on an empty stack": (attempt("d9 e1 c3"), "x87 stack underflow"),
        "fld st(7) with seven values": (attempt(f"{fld} " * 7 + "d9 c7 " + f"{fstp} " * 8 + "c3"),
                                        "x87 stack underflow"),
        "fstp st(1) as DF D1": (attempt(f"{fld} {fld} df d1 {fstp} c3"), "unsupported x87 encoding"),
        "fstp st(1) as DF D9": (attempt(f"{fld} {fld} df d9 {fstp} c3"), "unsupported x87 encoding"),
        "fxch st(1) as DD C9": (attempt(f"{fld} {fld} dd c9 {fstp} {fstp} c3"), "unsupported x87 encoding"),
        "fxch st(1) as DF C9": (attempt(f"{fld} {fld} df c9 {fstp} {fstp} c3"), "unsupported x87 encoding"),
        "fnstenv m28": (attempt(f"{fld} d9 32 {fstp} c3"), "stores the tag word of a non-empty x87 stack"),
        "fsin": (attempt(f"{fld} d9 fe {fstp} c3"), "unsupported x87 operation"),
        "ffree st(0)": (attempt(f"{fld} dd c0 {fstp} c3"), "frees an occupied register"),
        "fdecstp": (attempt(f"{fld} d9 f6 {fstp} c3"), "unsupported x87 operation"),
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
