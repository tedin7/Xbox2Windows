"""Differential checks for recursion through direct calls in the jump-table translator.

Each export of tests/cpu/recursion.asm runs through the register probe. The source-built original
must match the Python reference, and the translation must match the original on the return value,
the guarded trace and the probe's RBX and RSI sentinels. The structure check reads the summaries
that the call cycles settle on. The refused exports must be refused with their stated reason, and
two of them must be accepted when the translator keeps the first summary of a cycle instead of
iterating, so the iteration is what refuses them.
"""

import ctypes
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direct_check import GUARDS, MASK, SENTINELS, observe, register_probe

EXPORTS = ("sum_export", "parity_export", "walk_export", "fold_export")
# (export, expected refusal), translated alone.
REFUSED = (
    ("tail_ping", "callee at RVA {tail_pong:#x} refused: callee at RVA {tail_ping:#x} refused: "
                  "tail jump within a call cycle"),
    ("wide_odd", "undefined register rax"),
    ("lost_k", "callee argument r8 is undefined"),
    ("bad_a", "callee at RVA {bad_b:#x} refused: unsupported instruction"),
    ("stale_e", "callee at RVA {stale_a:#x} refused: undefined register rax"),
)
# Refused only because a cycle's first summary is translated again.
ITERATED = ("wide_odd", "lost_k")


def trace(first=MASK, second=MASK, third=MASK):
    return (GUARDS[0], first & MASK, second & MASK, third & MASK, GUARDS[1])


def sum_down(n, k):
    return sum(i ^ k for i in range(1, n + 1)) & MASK


def count_down(n):
    return sum(i & 3 for i in range(1, n + 1))


def walk_a(n, v, k):
    return v ^ k if n == 0 else (walk_b(n - 1, (v + n) & MASK, k) + 1) & MASK


def walk_b(n, v, k):
    return v if n == 0 else walk_c(n - 1, 3 * v & MASK, k) ^ count_down(n)


def walk_c(n, v, k):
    return v if n == 0 else walk_a(n - 1, (v + 5) & MASK, k)


def sum_export(x, y):
    value = sum_down(x & 0x1ff, y)
    return (value + x) & MASK, trace(value)


def parity_export(x, y):
    even, odd = int((x & 0xff) % 2 == 0), int((y & 0xff) % 2 == 1)
    return even + odd, trace(second=even, third=odd)


def walk_export(x, y):
    value = walk_a(x & 0x3f, y, x ^ y)
    return value, trace(third=value)


def fold(x, y):
    if x == 0:
        return y, y
    value, base = fold(x >> 1, (2 * y + (x & 1)) & MASK)
    return (value + (x & 1)) & MASK, base


def fold_export(x, y):
    value, base = fold(x, y)
    return value, trace(third=base)


REFERENCES = {name: globals()[name] for name in EXPORTS}


def inputs():
    """Boundary words for both arguments, plus seeded random pairs."""
    edges = (0, 1, 2, 3, 0x3f, 0x40, 0xff, 0x100, 0x1ff, 0x200, 0x7fffffff, 0x80000000, 0xffffffff,
             0x100000000, 0x8000000000000000, MASK - 1, MASK, 0x0123456789abcdef)
    rng = random.Random(1221)
    pairs = [(x, y) for x in edges for y in edges]
    pairs += [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(160)]
    return pairs


def run(probe, library, prefix=""):
    observed = {}
    for name in EXPORTS:
        function = ctypes.cast(getattr(library, prefix + name), ctypes.c_void_p).value
        observed[name] = [observe(probe, function, x, y) for x, y in inputs()]
    return observed


def verify_reference(observed, label):
    for name in EXPORTS:
        for (x, y), (returned, words, registers) in zip(inputs(), observed[name]):
            want = REFERENCES[name](x, y)
            if (returned, words) != want or registers != SENTINELS:
                raise AssertionError(f"{label} {name}({x:#x}, {y:#x}): {returned:#x} {words} {registers} "
                                     f"!= {want[0]:#x} {want[1]} {SENTINELS}")


def tools():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import pefile
    from cpu_leaf_aot import export_rva
    import cpu_switch_aot
    return pefile, export_rva, cpu_switch_aot


def layout(pe, export_rva, aot):
    """RVAs of the exports and of the functions their direct calls and tail jumps reach."""
    from iced_x86 import Decoder, FlowControl
    ends = dict(aot.runtime_functions(pe))
    rvas = {name: export_rva(pe, name) for name in EXPORTS + tuple(name for name, _ in REFUSED)}

    def targets(rva):
        found = []
        for ins in Decoder(64, pe.get_data(rva, ends[rva] - rva), ip=rva):
            if ins.flow_control == FlowControl.CALL or (ins.flow_control == FlowControl.UNCONDITIONAL_BRANCH
                                                        and ins.near_branch_target in ends):
                found.append(ins.near_branch_target)
        return found

    rvas["sum_down"], = targets(rvas["sum_export"])
    rvas["is_even"], rvas["is_odd"] = targets(rvas["parity_export"])
    rvas["walk_a"], = targets(rvas["walk_export"])
    rvas["walk_b"], = targets(rvas["walk_a"])
    rvas["walk_c"], rvas["count_down"] = targets(rvas["walk_b"])
    rvas["tail_pong"], = targets(rvas["tail_ping"])
    rvas["bad_b"], = targets(rvas["bad_a"])
    rvas["stale_a"], = targets(rvas["stale_e"])
    return rvas


def check(probe_path, original_path, lifted_path):
    probe = register_probe(probe_path)
    original = run(probe, ctypes.CDLL(str(original_path)))
    verify_reference(original, "source fixture")
    translated = run(probe, ctypes.CDLL(str(lifted_path)), "lifted_")
    for name in EXPORTS:
        for (x, y), want, got in zip(inputs(), original[name], translated[name]):
            if got != want:
                raise AssertionError(f"translation {name}({x:#x}, {y:#x}): {got} != {want}")
    print(f"{len(EXPORTS)} exports x {len(inputs())} inputs: the source fixture matches the reference and the "
          "translation matches it on the return value, guarded trace, RBX and RSI")


def check_structure(original_path):
    pefile, export_rva, aot = tools()
    with pefile.PE(str(original_path), fast_load=True) as pe:
        pe.parse_data_directories()
        rvas = layout(pe, export_rva, aot)
        program = aot.Program(pe)
        functions = [aot.lift(pe, name, program=program) for name in EXPORTS]
    source = "\n".join(program.sources + [text for text, _ in functions])
    internal = ("sum_down", "is_even", "is_odd", "count_down", "walk_a", "walk_b", "walk_c", "fold_export")
    declaration = "static word fn_{:x}(word rcx, word rdx, word r8, word r9)"
    for name in internal:
        text = declaration.format(rvas[name])
        if source.count(text + " {") != 1 or source.count(text + ";") != 1:
            raise AssertionError(f"{name} at RVA {rvas[name]:#x} is not declared once and defined once")
    if source.count("static word fn_") != 2 * len(internal):
        raise AssertionError(f"expected {len(internal)} declared internal functions")
    widths = {name: program.callees[rvas[name]].return_width for name in internal}
    want = dict.fromkeys(internal, 8) | {"is_even": 1, "is_odd": 1}
    if widths != want:
        raise AssertionError(f"return widths {widths} != {want}")
    r8 = aot.R.R8
    demands = {name: program.callees[rvas[name]].demands.get(r8, 0) for name in internal}
    want = dict.fromkeys(internal, 0) | {"walk_a": 8, "walk_b": 8, "walk_c": 8, "fold_export": 8}
    if demands != want:
        raise AssertionError(f"R8 demands {demands} != {want}")
    print(f"{len(internal)} recursive functions declared and defined once; return widths {widths}; R8 demands {demands}")


def check_refusals(original_path):
    pefile, export_rva, aot = tools()
    with pefile.PE(str(original_path), fast_load=True) as pe:
        pe.parse_data_directories()
        rvas = layout(pe, export_rva, aot)
    for name, reason in REFUSED:
        reason = reason.format(**rvas)
        try:
            aot.translate(original_path, [name])
        except ValueError as error:
            if reason not in str(error):
                raise AssertionError(f"{name} refused for another reason: {error}") from error
        else:
            raise AssertionError(f"{name} accepted: {reason}")
        print(f"control {name}: {reason}")
    # The planted fault: keep a cycle's first summary, however it compares with the assumption.
    iterate = aot.at_least_as_good
    aot.at_least_as_good = lambda computed, assumed: True
    try:
        for name in ITERATED:
            try:
                aot.translate(original_path, [name])
            except ValueError as error:
                raise AssertionError(f"{name} is refused without the iteration: {error}") from error
            print(f"control {name}: accepted when the first summary of a cycle is kept")
    finally:
        aot.at_least_as_good = iterate
    print(f"{len(REFUSED)} recursion rejection controls passed")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: recursion_check.py PROBE ORIGINAL LIFTED")
    check(*sys.argv[1:])
    check_structure(sys.argv[2])
    check_refusals(sys.argv[2])
