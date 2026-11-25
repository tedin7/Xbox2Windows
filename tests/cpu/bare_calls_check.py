"""Opt-in range bindings for direct calls and tail jumps to functions without unwind data.

tests/cpu/bare_calls.asm has two leaves with no .pdata entry and no export. Without a binding, a
call or tail jump to them is refused with the target RVA and the binding it needs. With the range
manifest that the build writes from the linker map's source labels (bound to the image's SHA-256),
they translate, and the translation must match the original and a Python reference through the
register probe. Patched images and edited manifests must be refused with their stated reason.
"""

import copy
import ctypes
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direct_check import GUARDS, MASK, SENTINELS, observe, register_probe

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import pefile  # noqa: E402
from cpu_leaf_aot import export_rva  # noqa: E402
from cpu_switch_aot import translate  # noqa: E402
from range_manifest import map_symbols  # noqa: E402

MASK32 = 0xffffffff
EXPORTS = ("call_bare", "twice_bare", "tail_bare")
WIDTHS = {"call_bare": 8, "twice_bare": 8, "tail_bare": 1}
NEEDED = "no runtime-function entry or export; a range binding [start, end) for this image (--bind-ranges) is needed"


def trace(first=MASK, second=MASK, third=MASK):
    return (GUARDS[0], first & MASK, second & MASK, third & MASK, GUARDS[1])


def bare_mix(c, d):
    return (5 * c - d) & MASK


REFERENCES = {
    "call_bare": lambda x, y: (bare_mix(x, y), trace(bare_mix(x, y))),
    "twice_bare": lambda x, y: (bare_mix(x + 1, y), trace(second=bare_mix(x + 1, y))),
    "tail_bare": lambda x, y: ((x ^ y) & 0xff, trace()),
}


def inputs():
    edges = (0, 1, 2, 5, 0x7f, 0x80, 0xff, 0x7fff, 0xffff, 0x7fffffff, 0x80000000, MASK32,
             0x100000000, 0x8000000000000000, MASK - 4, MASK, 0x0123456789abcdef)
    rng = random.Random(3173)
    return [(x, y) for x in edges for y in edges] + [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(256)]


def run(probe, library, prefix=""):
    observed = {}
    for name in EXPORTS:
        function = ctypes.cast(getattr(library, prefix + name), ctypes.c_void_p).value
        rows = []
        for x, y in inputs():
            returned, words, registers = observe(probe, function, x, y)
            rows.append((returned & ((1 << 8 * WIDTHS[name]) - 1), words, registers))
        observed[name] = rows
    return observed


def check(probe_path, original_path, lifted_path):
    probe = register_probe(probe_path)
    original = run(probe, ctypes.CDLL(str(original_path)))
    for name in EXPORTS:
        for (x, y), (returned, words, registers) in zip(inputs(), original[name]):
            want = REFERENCES[name](x, y)
            if (returned, words) != want or registers != SENTINELS:
                raise AssertionError(f"source fixture {name}({x:#x}, {y:#x}): {returned:#x} {words} != {want}")
    translated = run(probe, ctypes.CDLL(str(lifted_path)), "lifted_")
    for name in EXPORTS:
        for (x, y), want, got in zip(inputs(), original[name], translated[name]):
            if got != want:
                raise AssertionError(f"translation {name}({x:#x}, {y:#x}): {got} != {want}")
    print(f"{len(EXPORTS)} exports x {len(inputs())} inputs: with the binding, the translation matches the original "
          "and the reference on the defined return bytes, guarded trace, RBX and RSI")


def layout(original_path, map_path):
    symbols = map_symbols(Path(map_path).read_text(encoding="utf-8", errors="replace"))
    with pefile.PE(str(original_path), fast_load=True) as pe:
        rvas = {name: export_rva(pe, name) for name in EXPORTS}
        rdata = next(s for s in pe.sections if s.Name.rstrip(b"\0") == b".rdata")
        text = next(s for s in pe.sections if s.Name.rstrip(b"\0") == b".text")
        directory = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = list(struct.iter_unpack("<III", pe.get_data(directory.VirtualAddress, directory.Size)))
        stamp = pe.FILE_HEADER.get_file_offset() + 4
    for name in ("bare_mix", "bare_mix_inner", "bare_mix_end", "bare_low", "bare_low_end", "fake_thunk",
                 "fake_thunk_end"):
        rvas[name] = symbols[name]
    for name in ("bare_mix", "bare_low", "fake_thunk"):
        if any(begin <= rvas[name] < end for begin, end, _ in records):
            raise AssertionError(f"{name} has unwind data")
    rvas.update(rdata=rdata.VirtualAddress, text_end=text.VirtualAddress + min(text.Misc_VirtualSize, text.SizeOfRawData),
                stamp=stamp)
    return rvas


def check_default_and_structure(original_path, manifest_path, rvas):
    for name, site, target in (("call_bare", "call target", "bare_mix"), ("tail_bare", "jump target", "bare_low")):
        reason = f"{site} {rvas[target]:#x} is not a function entry ({NEEDED})"
        try:
            translate(original_path, [name])
        except ValueError as error:
            if reason not in str(error):
                raise AssertionError(f"{name} without a binding refused for another reason: {error}") from error
        else:
            raise AssertionError(f"{name} accepted without a binding")
        print(f"default, no binding: {name}: {reason}")
    source, functions = translate(original_path, list(EXPORTS), manifest_path)
    for callee in ("bare_mix", "bare_low"):
        if source.count(f"static word fn_{rvas[callee]:x}(") != 1:
            raise AssertionError(f"bound callee {callee} is not emitted exactly once")
    if source.count("static word fn_") != 2:
        raise AssertionError("expected 2 internal functions")
    widths = {name: function.return_width for name, function in zip(EXPORTS, functions)}
    if widths != WIDTHS:
        raise AssertionError(f"return widths {widths} != {WIDTHS}")
    print(f"with the binding: 2 bound callees, each emitted once; return widths {widths}")
    # The interior label is a valid path to RET: bound as a range of its own, a call to it translates.
    # So the interior controls below are refused by the binding rule alone.
    from tempfile import TemporaryDirectory
    data = bytearray(Path(original_path).read_bytes())
    with pefile.PE(data=bytes(data), fast_load=True) as pe:
        at = pe.get_offset_from_rva(rvas["call_bare"] + 0x08)
    if data[at] != 0xe8:
        raise AssertionError("no call in call_bare")
    data[at + 1:at + 5] = struct.pack("<i", rvas["bare_mix_inner"] - (rvas["call_bare"] + 0x08 + 5))
    manifest = {"image_sha256": hashlib.sha256(data).hexdigest(), "functions": [
        {"name": "bare_mix_inner", "start": f"{rvas['bare_mix_inner']:#x}", "end": f"{rvas['bare_mix_end']:#x}"}]}
    with TemporaryDirectory() as folder:
        image, ranges = Path(folder) / "inner.dll", Path(folder) / "inner.json"
        image.write_bytes(data)
        ranges.write_text(json.dumps(manifest), encoding="utf-8")
        translate(image, ["call_bare"], ranges)
    print(f"the interior label {rvas['bare_mix_inner']:#x}, bound as its own range, translates as a callee")


def edit(name, start=None, end=None):
    """A manifest edit that moves the start or end of one binding, as a function of the RVAs."""
    def apply(manifest, rvas):
        for item in manifest["functions"]:
            if item["name"] == name:
                if start is not None:
                    item["start"] = f"{start(rvas):#x}"
                if end is not None:
                    item["end"] = f"{end(rvas):#x}"
    return apply


def add(name, start, end):
    def apply(manifest, rvas):
        manifest["functions"].append({"name": name, "start": f"{rvas[start]:#x}", "end": f"{rvas[end]:#x}"})
    return apply


# (export, image patches, manifest edit, expected refusal). A patch is (function, offset, original hex
# or None for the CALL or JMP already there, replacement), where the replacement is hex or (kind, target) with kind "call" (CALL rel32) or "jmp8"
# (JMP rel8), padded with NOPs; a target is a symbol or (symbol, delta). The patch "stamp" changes
# the image's TimeDateStamp and keeps the manifest's hash, as for another build of the same source;
# every other control rehashes the manifest for the patched image. The reason may use {symbol}.
CALL, JMP8 = None, None  # the existing CALL rel32 or JMP rel8 at that offset, checked by opcode
CONTROLS = [
    ("call_bare", ["stamp"], None, "range binding is for another image"),
    ("call_bare", [("call_bare", 0x08, CALL, ("call", "bare_mix_inner"))], None,
     "call target {bare_mix_inner:#x} is not a function entry (inside the bound range bare_mix [{bare_mix:#x}, {bare_mix_end:#x}), not its start)"),
    ("call_bare", [("call_bare", 0x08, CALL, ("call", ("bare_mix", 1)))], None,
     "inside the bound range bare_mix"),
    ("tail_bare", [("tail_bare", 0x00, JMP8, ("jmp8", ("bare_low", 2)))], None,
     "jump target {bare_low2:#x} is not a function entry (inside the bound range bare_low"),
    ("call_bare", [], edit("bare_mix", start=lambda r: r["rdata"], end=lambda r: r["rdata"] + 8),
     "range binding bare_mix [{rdata:#x}, {rdata8:#x}) is not inside executable code"),
    ("call_bare", [], edit("bare_low", end=lambda r: r["text_end"] + 0x10),
     "range binding bare_low [{bare_low:#x}, {text_end16:#x}) is not inside executable code"),
    ("call_bare", [], edit("bare_low", end=lambda r: r["call_bare"] + 4),
     "overlaps the runtime function at {call_bare:#x}"),
    ("tail_bare", [], edit("bare_mix", end=lambda r: r["bare_low_end"]),
     "range binding bare_low [{bare_low:#x}, {bare_low_end:#x}) overlaps range binding bare_mix"),
    ("call_bare", [], edit("bare_mix", end=lambda r: r["bare_mix_end"] - 1),
     "edge outside owned ranges at RVA {bare_mix_ret:#x}: outside the bound range bare_mix"),
    ("tail_bare", [("bare_low", 0x02, "32 c2", ("jmp8", ("call_bare", 1)))], None,
     "another function's unwind interval"),
    ("tail_bare", [("bare_low", 0x02, "32 c2", ("jmp8", "bare_mix_inner"))], None,
     "jump target {bare_mix_inner:#x} is not a function entry (inside the bound range bare_mix"),
    ("tail_bare", [("bare_low", 0x02, "32 c2", ("jmp8", "fake_thunk"))], None,
     "jump target {fake_thunk:#x} is not a function entry (" + NEEDED),
    ("call_bare", [("call_bare", 0x08, CALL, ("call", "fake_thunk"))], add("fake_thunk", "fake_thunk", "fake_thunk_end"),
     "bound function fake_thunk at RVA {fake_thunk:#x} is an import thunk"),
    ("call_bare", [("bare_mix", 0x00, "48 6b c9 05 48 8b c1", ("call", "bare_low"))], None,
     "call in a function without unwind data"),
    ("call_bare", [("bare_mix", 0x00, "48 6b c9 05", "53 5b 90 90")], None,
     "stack change in a function without unwind data"),
    ("call_bare", [("bare_mix", 0x00, "48 6b c9 05 48 8b c1", ("call", "bare_mix"))], None,
     "call in a function without unwind data"),
]


def check_rejections(original_path, manifest_path, rvas):
    from tempfile import TemporaryDirectory

    data = Path(original_path).read_bytes()
    built = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if built["image_sha256"] != hashlib.sha256(data).hexdigest():
        raise AssertionError("the built manifest is not bound to the built image")
    names = dict(rvas, bare_low2=rvas["bare_low"] + 2, rdata8=rvas["rdata"] + 8, text_end16=rvas["text_end"] + 0x10, bare_mix_ret=rvas["bare_mix_end"] - 1)
    with pefile.PE(data=data, fast_load=True) as pe:
        offset = pe.get_offset_from_rva

    def resolve(target):
        return rvas[target] if isinstance(target, str) else rvas[target[0]] + target[1]

    with TemporaryDirectory() as folder:
        image, manifest_file = Path(folder) / "control.dll", Path(folder) / "ranges.json"
        for name, patches, change, reason in CONTROLS:
            reason = reason.format(**names)
            damaged, sites = bytearray(data), []
            rehash = True
            for patch in patches:
                if patch == "stamp":
                    damaged[rvas["stamp"]] ^= 1
                    rehash = False
                    sites.append("TimeDateStamp")
                    continue
                function, delta, before, after = patch
                rva = rvas[function] + delta
                at = offset(rva)
                if before is None:
                    opcode, length = (0xe8, 5) if after[0] == "call" else (0xeb, 2)
                    if data[at] != opcode:
                        raise AssertionError(f"no {after[0]} at {function}+{delta:#x} for control {reason!r}")
                    before = data[at:at + length]
                else:
                    before = bytes.fromhex(before)
                if data[at:at + len(before)] != before:
                    raise AssertionError(f"unexpected encoding for control {reason!r} in {function}")
                if isinstance(after, tuple):
                    target = resolve(after[1])
                    after = (b"\xe8" + struct.pack("<i", target - (rva + 5)) if after[0] == "call"
                             else b"\xeb" + struct.pack("<b", target - (rva + 2)))
                    after += b"\x90" * (len(before) - len(after))
                    sites.append(f"{function}+{delta:#x} to {target:#x}")
                else:
                    after = bytes.fromhex(after)
                    sites.append(f"{function}+{delta:#x}")
                damaged[at:at + len(after)] = after
            manifest = copy.deepcopy(built)
            if change is not None:
                change(manifest, rvas)
                sites.append("manifest edited")
            if rehash:
                manifest["image_sha256"] = hashlib.sha256(damaged).hexdigest()
            image.write_bytes(damaged)
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            try:
                translate(image, [name], manifest_file)
            except ValueError as error:
                if reason not in str(error):
                    raise AssertionError(f"{name} control {reason!r} refused for another reason: {error}") from error
            else:
                raise AssertionError(f"{name} control accepted: {reason}")
            print(f"control {name} ({', '.join(sites) or 'as built'}): {reason}")
    print(f"{len(CONTROLS)} binding rejection controls passed")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit("usage: bare_calls_check.py PROBE ORIGINAL LIFTED MANIFEST MAP")
    probe, original, lifted, manifest, map_file = sys.argv[1:]
    check(probe, original, lifted)
    rvas = layout(original, map_file)
    check_default_and_structure(original, manifest, rvas)
    check_rejections(original, manifest, rvas)
