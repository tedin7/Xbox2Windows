"""Differential checks for direct calls and tail jumps in the jump-table translator.

Each export of tests/cpu/calls.asm runs through the register probe. The source-built original
must match the Python reference, and the translation must match the original on the return value
(its defined low bytes), the guarded trace and the probe's RBX and RSI sentinels. Each callee must
be emitted once, and patched copies must be refused with their stated reason.
"""

import ctypes
from pathlib import Path
import random
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direct_check import GUARDS, MASK, SENTINELS, observe, register_probe

MASK32 = 0xffffffff
EXPORTS = ("call_leaf", "chain3", "narrow_call", "narrow_tail", "tail_call", "tail_after_frame", "home_save_call")
# Low RAX bytes that each export defines at every return; the rest are not compared.
WIDTHS = {name: 8 for name in EXPORTS} | {"narrow_tail": 1}


def trace(first=MASK, second=MASK, third=MASK):
    return (GUARDS[0], first & MASK, second & MASK, third & MASK, GUARDS[1])


def leaf_mix(c, d):
    return (3 * c + d) & MASK


def mid_step(c, d):
    return (leaf_mix(c ^ d, d) + d) & MASK


def call_leaf(x, y):
    value = leaf_mix(x, y)
    return (value + 1) & MASK, trace(value)


def chain3(x, y):
    value = mid_step(x, y)
    return value, trace(second=value)


def narrow_call(x, y):
    value = (x + y) & 0xff
    return value, trace(third=value)


def narrow_tail(x, y):
    return (x + y) & 0xff, trace()


def tail_call(x, y):
    return leaf_mix(x + 5, y), trace()


def tail_after_frame(x, y):
    return mid_step(x + 1, y), trace(x)


def home_save_call(x, y):
    value = leaf_mix(x, y)
    return value ^ x, trace(second=value)


REFERENCES = {name: globals()[name] for name in EXPORTS}


def inputs():
    """Boundary words for both arguments, plus seeded random pairs."""
    edges = (0, 1, 2, 5, 0x7f, 0x80, 0xff, 0x7fff, 0xffff, 0x7fffffff, 0x80000000, MASK32,
             0x100000000, 0x8000000000000000, MASK - 4, MASK, 0x0123456789abcdef)
    rng = random.Random(3172)
    pairs = [(x, y) for x in edges for y in edges]
    pairs += [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(256)]
    return pairs


def narrowed(name, observation):
    returned, words, registers = observation
    return returned & ((1 << 8 * WIDTHS[name]) - 1), words, registers


def run(probe, library, prefix=""):
    observed = {}
    for name in EXPORTS:
        function = ctypes.cast(getattr(library, prefix + name), ctypes.c_void_p).value
        observed[name] = [narrowed(name, observe(probe, function, x, y)) for x, y in inputs()]
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


def layout(pe, export_rva):
    """Export RVAs, and the three internal callees found from the direct calls that reach them."""
    from iced_x86 import Decoder, FlowControl
    rvas = {name: export_rva(pe, name) for name in EXPORTS}

    def first_call(name):
        data = pe.get_data(rvas[name], 64)
        for ins in Decoder(64, data, ip=rvas[name]):
            if ins.flow_control == FlowControl.CALL:
                return ins.near_branch_target
        raise AssertionError(f"no call in {name}")

    rvas.update(leaf=first_call("call_leaf"), mid=first_call("chain3"), low=first_call("narrow_call"))
    data = pe.get_data(rvas["chain3"], 64)
    rvas["chain3_call"] = next(ins.ip for ins in Decoder(64, data, ip=rvas["chain3"])
                               if ins.flow_control == FlowControl.CALL)
    return rvas


def check_below_call(probe_path, original_path):
    """Natively, below_call's reload returns the CALL's return address, not the stored argument."""
    probe = register_probe(probe_path)
    function = ctypes.cast(ctypes.CDLL(str(original_path)).below_call, ctypes.c_void_p).value
    values = {observe(probe, function, x, y)[0] for x, y in inputs()[:64]}
    if len(values) != 1 or values & {x for x, _ in inputs()[:64]}:
        raise AssertionError(f"below_call returned {sorted(map(hex, values))[:4]}, expected one return address")
    print(f"below_call natively returns {values.pop():#x} for every input: the CALL overwrote the stored slot")


def check_structure(original_path):
    pefile, export_rva, aot = tools()
    source, functions = aot.translate(original_path, list(EXPORTS))
    with pefile.PE(str(original_path), fast_load=True) as pe:
        rvas = layout(pe, export_rva)
    for callee in ("leaf", "mid", "low"):
        definitions = source.count(f"static word fn_{rvas[callee]:x}(")
        if definitions != 1:
            raise AssertionError(f"callee {callee} at RVA {rvas[callee]:#x} emitted {definitions} times")
        if not rvas[callee] < min(rvas[name] for name in EXPORTS):
            raise AssertionError(f"callee {callee} is not below its callers")
    if source.count("static word fn_") != 3:
        raise AssertionError(f"expected 3 internal functions, got {source.count('static word fn_')}")
    widths = {name: function.return_width for name, function in zip(EXPORTS, functions)}
    if widths != WIDTHS:
        raise AssertionError(f"return widths {widths} != {WIDTHS}")
    print(f"3 internal callees below their callers, each emitted once; return widths {widths}")


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
          "translation matches it on the defined return bytes, guarded trace, RBX and RSI")


# (export to translate, [(function, offset, original bytes, patch)], expected refusal). A function is an
# export name or one of the callees "leaf", "mid" and "low". A patch is hex bytes or ("call", target),
# a CALL rel32 to the target RVA, given as a function, (function, delta) or ("rva", value). Each patch
# keeps the instruction length, so every other instruction stays where it was. The reason may name
# the callees' RVAs as {leaf}, {mid} and {low}.
CALL = "e8 ?? ?? ?? ??"
CONTROLS = [
    # Recursion through CALL is translated (tests/cpu/recursion_check.py); a cycle that a tail jump
    # closes is not. A reason of None means the patched copy must translate.
    ("call_leaf", [("call_leaf", 0x08, CALL, ("call", "call_leaf"))], None),
    ("chain3", [("mid", 0x0c, CALL, ("call", "chain3"))], None),
    ("tail_after_frame", [("mid", 0x0c, CALL, ("call", "tail_after_frame"))],
     "callee at RVA {mid:#x} refused: callee at RVA {tail_after_frame:#x} refused: tail jump within a call cycle"),
    ("call_leaf", [("call_leaf", 0x08, CALL, ("call", ("leaf", 1)))], "is not a function entry"),
    ("call_leaf", [("call_leaf", 0x08, CALL, ("call", ("leaf", 4)))], "is not a function entry"),
    ("call_leaf", [("call_leaf", 0x08, CALL, ("call", ("rva", "end")))], "call target outside the image"),
    ("call_leaf", [("call_leaf", 0x08, CALL, ("call", ("rva", ".rdata")))], "is not in an executable section"),
    # Reads of volatile registers after the call: the callee may have changed them.
    ("call_leaf", [("call_leaf", 0x0d, "48 89 03", "48 89 01")], "undefined register rcx"),
    ("chain3", [("chain3", 0x0d, "48 89 46 08", "49 89 40 08")], "undefined register r8"),
    # low_byte defines only AL, so a full read of RAX after the call is refused.
    ("narrow_call", [("narrow_call", 0x0d, "0f b6 c0", "48 8b c0")], "undefined register rax"),
    # A second call to the leaf right after the first: the leaf reads RCX and RDX, which the
    # first call left undefined.
    ("call_leaf", [("call_leaf", 0x0d, "48 89 03 48 83 c0 01", ("call", "leaf"))], "callee argument rcx is undefined"),
    ("narrow_call", [("narrow_call", 0x0d, "0f b6 c0", "0f 94 c0")], "flags clobbered by the call"),
    # mid_step reads the fifth-argument slot above its caller's home area, where chain3 pushed RSI.
    ("chain3", [("mid", 0x11, "48 03 44 24 20", "48 03 44 24 50")],
     "stack argument 5 of the call at RVA {chain3_call:#x} holds a saved register"),
    # An unsupported instruction in leaf_mix refuses mid_step and with it chain3.
    ("chain3", [("leaf", 0x04, "48 03 c2", "0f a2 90")],
     "callee at RVA {mid:#x} refused: callee at RVA {leaf:#x} refused: unsupported instruction"),
    # A spill into the callee's home area does not survive the call.
    ("chain3", [("mid", 0x04, "48 89 54 24 20", "48 89 54 24 18"), ("mid", 0x11, "48 03 44 24 20", "48 03 44 24 18")],
     "callee at RVA {mid:#x} refused: undefined stack byte at entry RSP-0x10"),
    # A save into the callee's home space does not survive the call either; one into the
    # allocation above it does (home_save_call as built).
    ("home_save_call", [("home_save_call", 0x09, "48 89 74 24 20", "48 89 74 24 18"),
                        ("home_save_call", 0x20, "48 8b 74 24 20", "48 8b 74 24 18")],
     "undefined stack byte at entry RSP-0x10"),
    ("call_leaf", [("call_leaf", 0x01, "48 83 ec 20", "48 83 ec 28"), ("call_leaf", 0x14, "48 83 c4 20", "48 83 c4 28")],
     "call without an aligned home space at entry RSP-0x30"),
    ("chain3", [("chain3", 0x01, "48 83 ec 20", "90 90 90 90"), ("chain3", 0x11, "48 83 c4 20", "90 90 90 90")],
     "call without an aligned home space at entry RSP-0x8"),
    ("tail_after_frame", [("tail_after_frame", 0x0b, "5b", "90")], "stack not balanced at tail jump"),
    # As built: a slot below the call-time RSP, which the CALL's return address overwrites, reloaded
    # after RSP falls back over it.
    ("below_call", [], "undefined stack byte at entry RSP-0x40"),
    ("tail_call", [("tail_call", 0x00, "48 83 c1 05", "48 8b d9 90")],
     "nonvolatile register rbx is not restored on every path to the tail jump"),
]


def check_rejections(original_path):
    pefile, export_rva, aot = tools()
    from tempfile import TemporaryDirectory

    data = Path(original_path).read_bytes()
    with pefile.PE(data=data, fast_load=True) as pe:
        rvas = layout(pe, export_rva)
        rvas.update({"end": pe.OPTIONAL_HEADER.SizeOfImage + 0x100})
        rvas.update({s.Name.rstrip(b"\0").decode(): s.VirtualAddress for s in pe.sections})
        offset = pe.get_offset_from_rva

        def resolve(target):
            if isinstance(target, str):
                return rvas[target]
            base, delta = target
            return rvas[delta] if base == "rva" else rvas[base] + delta

        with TemporaryDirectory() as folder:
            path = Path(folder) / "control.dll"
            for name, patches, reason in CONTROLS:
                reason = None if reason is None else reason.format(**rvas)
                damaged = bytearray(data)
                sites = []
                for function, delta, before, after in patches:
                    sites.append(f"{function}+{delta:#x}")
                    rva = rvas[function] + delta
                    at = offset(rva)
                    if before == CALL:
                        if data[at] != 0xe8:
                            raise AssertionError(f"no call at {function}+{delta:#x} for control {reason!r}")
                        target = resolve(after[1])
                        after = b"\xe8" + struct.pack("<i", target - (rva + 5))
                        sites[-1] += f" call {target:#x}"
                    else:
                        before = bytes.fromhex(before)
                        if isinstance(after, tuple):
                            # A CALL rel32 placed over other instructions, padded with NOPs.
                            target = resolve(after[1])
                            after = b"\xe8" + struct.pack("<i", target - (rva + 5))
                            after += b"\x90" * (len(before) - len(after))
                            sites[-1] += f" call {target:#x}"
                        else:
                            after = bytes.fromhex(after)
                        if data[at:at + len(before)] != before or len(after) != len(before):
                            raise AssertionError(f"unexpected encoding for control {reason!r} in {function}")
                    damaged[at:at + len(after)] = after
                path.write_bytes(damaged)
                if reason is None:
                    source, _ = aot.translate(path, [name])
                    if not re.search(r"^static word fn_[0-9a-f]+\(word rcx, word rdx, word r8, word r9\);$", source, re.M):
                        raise AssertionError(f"{name} control ({', '.join(sites)}) has no declared call cycle")
                    print(f"control {name} ({', '.join(sites)}): translated with a declared call cycle")
                    continue
                try:
                    aot.translate(path, [name])
                except ValueError as error:
                    if reason not in str(error):
                        raise AssertionError(f"{name} control {reason!r} refused for another reason: {error}") from error
                else:
                    raise AssertionError(f"{name} control accepted: {reason}")
                print(f"control {name} ({', '.join(sites)}): {reason}")
    print(f"{len(CONTROLS)} call rejection controls passed")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "check":
        check(*sys.argv[2:])
        check_below_call(sys.argv[2], sys.argv[3])
        check_structure(sys.argv[3])
        check_rejections(sys.argv[3])
    else:
        raise SystemExit("usage: calls_check.py check PROBE ORIGINAL LIFTED")
