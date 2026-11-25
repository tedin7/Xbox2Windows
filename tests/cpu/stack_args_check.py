"""Differential checks for stack arguments of internal callees in the jump-table translator.

Each export of tests/cpu/stack_args.asm runs through the register probe. The source-built original
must match the Python reference, and the translation must match the original on the return value,
the guarded trace and the probe's RBX and RSI sentinels. The structure check reads the stack
argument bytes each callee demands. The refused exports must be refused with their stated reason.
"""

import ctypes
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direct_check import GUARDS, MASK, SENTINELS, observe, register_probe

MASK32 = 0xffffffff
EXPORTS = ("call_six", "call_seven", "call_scratch", "call_rec", "call_eight", "call_fill", "call_edge", "call_twelfth")
# (export, expected refusal), translated alone.
REFUSED = (
    ("unset_slot", "stack argument 6 of the call at RVA {unset_slot_call:#x} is undefined at entry RSP-0x10"),
    ("narrow_slot", "stack argument 5 of the call at RVA {narrow_slot_call:#x} is undefined at entry RSP-0x14"),
    ("reread_slot", "undefined stack byte at entry RSP-0x18"),
    ("tail_six", "tail jump to a function with stack arguments"),
    ("export_fifth", "unsupported stack argument at RVA {export_fifth:#x}: entry RSP+0x28"),
    ("far_slot", "callee at RVA {far_ninth:#x} refused: unsupported stack argument at RVA {far_ninth:#x}: "
                 "entry RSP+0x68"),
    ("straddle_slot", "callee at RVA {straddle:#x} refused: unsupported stack argument at RVA {straddle:#x}: "
                      "entry RSP+0x24"),
    ("high_slots", "stack arguments of the call at RVA {high_slots_call:#x} reach the return address"),
    ("lost_slot", "callee at RVA {rec_lost:#x} refused: stack argument 5 of the call at RVA {rec_lost_call:#x} "
                  "is undefined at entry RSP-0x18"),
)
# Refused only because a cycle's first summary is translated again.
ITERATED = ("lost_slot",)

# Refused only because the frame proof follows the stack arguments of each call.
UNPROVEN = ("unset_slot", "narrow_slot", "reread_slot")


def trace(first=MASK, second=MASK, third=MASK):
    return (GUARDS[0], first & MASK, second & MASK, third & MASK, GUARDS[1])


def six_mix(a, b, c, d, e, f):
    return (a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * (f & MASK32)) & MASK


def seven_fwd(a, b, c, d, e, f, g):
    return (six_mix(a, b, c, d, f, e) + 7 * (g & 0xff)) & MASK


def scratch_five(a, b, e):
    return ((e + a) & MASK) ^ b


def rec_five(n, acc, step):
    return acc if n == 0 else rec_five(n - 1, (acc + step) & MASK, 3 * step & MASK)


def eight_sum(a, b, c, d, e, f, g, h):
    return (a + 3 * b + 5 * c + 7 * d + 9 * e + 11 * f + 13 * g + 15 * (h & 0xffff)) & MASK


def call_six(x, y):
    value = six_mix(x, y, x ^ y, (y + 1) & MASK, x, y)
    return value, trace(value)


def call_seven(x, y):
    value = seven_fwd(y, x, 5, (x + y) & MASK, y, x, x >> 8)
    return (value + y) & MASK, trace(second=value)


def call_scratch(x, y):
    first = scratch_five(x, y, 2 * y & MASK)
    return scratch_five(x, first, x) ^ x, trace(third=first)


def call_rec(x, y):
    value = rec_five(x & 0x1f, y, x)
    return value, trace(value)


def call_eight(x, y):
    value = eight_sum(x, y, (x + y) & MASK, (x - y) & MASK, y, x, x ^ y, y)
    return value, trace(second=value)


def call_fill(x, y):
    return 3 * x & MASK, trace()


def call_edge(x, y):
    return scratch_five(y, x, x), trace()


def call_twelfth(x, y):
    return x ^ y, trace()


REFERENCES = {name: globals()[name] for name in EXPORTS}


def inputs():
    """Boundary words for both arguments, plus seeded random pairs."""
    edges = (0, 1, 2, 0x1f, 0x20, 0xff, 0x100, 0xffff, 0x10000, 0x7fffffff, 0x80000000, MASK32,
             0x100000000, 0x8000000000000000, MASK - 1, MASK, 0x0123456789abcdef)
    rng = random.Random(1205)
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
    """RVAs of the exports, the callees their calls reach and each refused export's call."""
    from iced_x86 import Decoder, FlowControl
    ends = dict(aot.runtime_functions(pe))
    rvas = {name: export_rva(pe, name) for name in EXPORTS + tuple(name for name, _ in REFUSED)}

    def calls(rva):
        return [(ins.ip, ins.near_branch_target) for ins in Decoder(64, pe.get_data(rva, ends[rva] - rva), ip=rva)
                if ins.flow_control in (FlowControl.CALL, FlowControl.UNCONDITIONAL_BRANCH)]

    (_, rvas["six_mix"]), = calls(rvas["call_six"])
    (_, rvas["seven_fwd"]), = calls(rvas["call_seven"])
    (_, rvas["scratch_five"]), _ = calls(rvas["call_scratch"])
    (_, rvas["rec_five"]), = calls(rvas["call_rec"])
    (_, rvas["eight_sum"]), = calls(rvas["call_eight"])
    (_, rvas["far_ninth"]), = calls(rvas["far_slot"])
    (_, rvas["straddle"]), = calls(rvas["straddle_slot"])
    (_, rvas["fill_five"]), = calls(rvas["call_fill"])
    (_, rvas["twelfth"]), = calls(rvas["call_twelfth"])
    (_, rvas["rec_lost"]), = calls(rvas["lost_slot"])
    (rvas["rec_lost_call"], _), = calls(rvas["rec_lost"])
    for name in ("unset_slot", "narrow_slot", "high_slots"):
        (rvas[f"{name}_call"], _), = calls(rvas[name])
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
        for name in EXPORTS:
            aot.lift(pe, name, program=program)
    # Low bytes of each stack argument that some read of the callee reaches.
    want = {"six_mix": (8, 4), "seven_fwd": (8, 8, 1), "scratch_five": (8,), "rec_five": (8,),
            "eight_sum": (8, 8, 8, 2), "fill_five": (0,), "twelfth": (0,) * 7 + (8,)}
    stacks = {name: program.callees[rvas[name]].stack for name in want}
    if stacks != want:
        raise AssertionError(f"stack argument demands {stacks} != {want}")
    source = "\n".join(program.sources)
    for name, stack in want.items():
        parameters = ", ".join(["word rcx", "word rdx", "word r8", "word r9"]
                               + [f"word arg{5 + slot}" for slot in range(len(stack))])
        if f"static word fn_{rvas[name]:x}({parameters}) {{" not in source:
            raise AssertionError(f"{name} is not defined with {len(stack)} stack argument parameter(s)")
    print(f"{len(want)} callees take their stack arguments as parameters; demands {stacks}")


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
    # The planted fault: the frame proof ignores the stack arguments of calls.
    prove = aot.prove_frame
    aot.prove_frame = lambda function, order, effects, frame, stacks=None: prove(function, order, effects, frame)
    try:
        for name in UNPROVEN:
            try:
                aot.translate(original_path, [name])
            except ValueError as error:
                raise AssertionError(f"{name} is refused without the stack argument proof: {error}") from error
            print(f"control {name}: accepted when the frame proof ignores stack arguments")
    finally:
        aot.prove_frame = prove
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
    print(f"{len(REFUSED)} stack argument rejection controls passed")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: stack_args_check.py PROBE ORIGINAL LIFTED")
    check(*sys.argv[1:])
    check_structure(sys.argv[2])
    check_refusals(sys.argv[2])
