"""Check gfx1013 image_atomic_add, umax and cmpswap on a 2D R32_UINT UAV.

gfx1013 takes the gfx1030 image atomic lowering for these three operations. The cases are the
gfx1030 ones (tests/compute_store_check.py, mimg_atomic_cases) with the gfx1013 profile: the
controls pin identical HLSL, the gfx1013 refusals and, with --binary, the gfx1013 machine words;
--draw dispatches them on an adapter and compares every texel of a three-level texture with an
independent reference, including mutants that must mismatch.
"""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from compute_store_check import (MIMG_ATOMIC_COMPARE, MIMG_ATOMIC_EXCHANGE_FLAG, MIMG_ATOMIC_FORBIDDEN_FLAGS,
                                 MIMG_ATOMIC_MINMAX_FLAGS, MIMG_ATOMIC_REPLACEMENT, MIMG_ATOMIC_UINT_FLAG,
                                 MIMG_ATOMIC_VALUE, MIMG_STORE_GROUP, TEXTURE_WIDTH, aot, assemble, disassemble,
                                 mimg_atomic_cases, mimg_atomic_reference, rejects, store_container, variant)

HERE = Path(__file__).resolve().parent
ACCEPTED = ("add", "umax", "cmpswap")
SUITES = {"add": "integer", "umax": "minmax", "cmpswap": "exchange"}
FLAGS = {"add": MIMG_ATOMIC_UINT_FLAG, "umax": MIMG_ATOMIC_MINMAX_FLAGS["u"], "cmpswap": MIMG_ATOMIC_EXCHANGE_FLAG}
# Every other integer image atomic the shared lowering knows, refused on gfx1013.
REFUSED_UINT = ("sub", "and", "or", "xor", "umin", "swap", "inc", "dec")


def cases():
    """(name, op, contended, gfx1030 lines, gfx1013 binding) for the accepted forms."""
    found = []
    for op in ACCEPTED:
        for name, contended, lines, binding in mimg_atomic_cases(suite=SUITES[op]):
            if name.removesuffix("_contended") == op:
                found.append((name, op, contended, lines, binding | {"cpu": "gfx1013"}))
    assert [name for name, *_ in found] == ["add", "add_contended", "umax", "cmpswap"], found
    return found


def lowered(lines, binding):
    return aot.lower(lines, MIMG_STORE_GROUP, **binding)


def controls():
    sources = {}
    for name, op, _, lines, binding in cases():
        source = lowered(lines, binding)
        # The gfx1013 profile emits exactly the gfx1030 lowering.
        assert source == lowered(lines, binding | {"cpu": "gfx1030"}), name
        assert "RWTexture2D<uint> outputData : register(u0);" in source, source
        sources[name] = source
    value = f"{MIMG_ATOMIC_VALUE:#010x}u"
    assert f"InterlockedAdd(outputData[uint2(" in sources["add"] and f", {value});" in sources["add"]
    assert "InterlockedMax(outputData[uint2(" in sources["umax"] and f", {value});" in sources["umax"]
    assert "asint(" not in sources["umax"], "umax must stay unsigned"
    assert (f", {MIMG_ATOMIC_COMPARE:#010x}u, {MIMG_ATOMIC_REPLACEMENT:#010x}u);" in sources["cmpswap"]
            and sources["cmpswap"].count("InterlockedCompareStore(") == 1), sources["cmpswap"]

    name, op, _, lines, binding = cases()[0]
    instruction = next(line for line in lines if line.startswith("image_atomic_"))
    for other in REFUSED_UINT:
        rejects("gfx1013 image atomics require add, umax or cmpswap", lowered,
                variant(lines, ("image_atomic_add", f"image_atomic_{other}")), binding)
    # Signed and float formats are refused on gfx1013 before any instruction is read.
    umax_lines, umax_binding = cases()[2][3], cases()[2][4]
    for change in ({"image_format": "R32_SINT"}, {"image_format": "R32_FLOAT", "image_float_domain": "finite-normal"},
                   {"image_uav_mip": 1}):
        rejects("Invalid image UAV binding", lowered, umax_lines, umax_binding | change)
    smin = variant(umax_lines, ("image_atomic_umax", "image_atomic_smin"))
    rejects("Invalid image UAV binding", lowered, smin, umax_binding | {"image_format": "R32_SINT"})
    for changed, diagnostic in (
        (instruction + "\n" + instruction, "Unsupported image atomic target or binding"),
        (instruction.replace("IMG_2D", "IMG_2D_ARRAY"), "Unsupported compute instruction"),
        (instruction.replace("IMG_2D", "IMG_3D"), "Unsupported compute instruction"),
        (instruction.replace("dmask:0x1", "dmask:0x3"), "Unverified image atomic data range or mask"),
        (instruction.replace("s[24:31]", "s[28:35]"), "Unverified image atomic descriptor"),
        *((f"{instruction} {flag}", "Unsupported compute instruction") for flag in MIMG_ATOMIC_FORBIDDEN_FLAGS),
    ):
        rejects(diagnostic, lowered, variant(lines, (instruction, changed)), binding)
    rejects("Unverified image atomic descriptor", lowered,
            [line for line in lines if line != "s_waitcnt lgkmcnt(0)"], binding)
    # A store to the R32_UINT image is still refused on gfx1013.
    store = instruction.replace("image_atomic_add v14", "image_store v14")
    rejects("Unsupported image store target", lowered, variant(lines, (instruction, store)), binding)
    print("gfx1013 image atomic translation controls passed", flush=True)
    return sources


# The two MIMG dwords of each atomic as LLVM 15 encodes them for gfx1013: opcode 17 (add), 23 (umax)
# or 16 (cmpswap) in bits 24:18, dmask in 11:8, unorm in bit 12, dim 1 (2D) in bits 5:3.
ATOMIC_WORDS = {"add": "080144f0080e0600", "add_contended": "081144f0080e0600",
                "umax": "08015cf0080e0600", "cmpswap": "081340f0080e0600"}


def binary(directory, assembler, objcopy, library, sources):
    for name, _, _, lines, binding in cases():
        code = assemble(directory, assembler, objcopy, lines)
        assert bytes.fromhex(ATOMIC_WORDS[name]) in code, (name, code.hex())
        # The MIMG encoding is the same as on gfx1030.
        assert code == assemble(directory, assembler, objcopy, lines, cpu="gfx1030"), name
        print(f"{name}: gfx1013 machine words {code.hex()}", flush=True)
        if not library:
            continue
        decoded = aot.labelled(disassemble(code, library, offsets=True, cpu=b"gfx1013"))
        atomics = [line for line in decoded if line.startswith("image_atomic_")]
        assert atomics == [line for line in lines if line.startswith("image_atomic_")], (name, decoded)
        assert lowered(decoded, binding) == sources[name], name
        container, output = directory / f"{name}.dxbc", directory / f"{name}-cli.hlsl"
        container.write_bytes(store_container(code, MIMG_STORE_GROUP))
        options = [item for key, value in binding.items() for item in ("--" + key.replace("_", "-"), str(value))]
        subprocess.run([sys.executable, "-B", aot.__file__, container, "--output", output, "--llvm", library,
                        *options], check=True)
        assert output.read_text() == sources[name], name


def mutants(op, source):
    """(control, mutated source, reference op or compare_store) for one accepted source."""
    value = f"{MIMG_ATOMIC_VALUE:#010x}u"
    if op == "add":
        assert source.count(f", {value});") == 1
        return "subtracts instead of adds", source.replace(f", {value});", f", (0u - {value}));"), ("sub", None)
    if op == "umax":
        assert source.count("InterlockedMax(") == 1
        return "minimum instead of maximum", source.replace("InterlockedMax(", "InterlockedMin("), ("umin", None)
    compare, replacement = f"{MIMG_ATOMIC_COMPARE:#010x}u", f"{MIMG_ATOMIC_REPLACEMENT:#010x}u"
    assert source.count(f", {compare}, {replacement});") == 1
    return ("reversed compare and replacement", source.replace(f", {compare}, {replacement});",
                                                               f", {replacement}, {compare});"),
            ("cmpswap", (MIMG_ATOMIC_REPLACEMENT, MIMG_ATOMIC_COMPARE)))


def run_case(renderer, backend, cso, case, op, groups, initial, expected):
    case.write_bytes(struct.pack("<674I", groups | FLAGS[op], 0, *initial, *expected))
    result = subprocess.run([renderer, backend, "--compute-texture-store-case", cso, case],
                            capture_output=True, text=True, timeout=30)
    print(result.stdout, end="", flush=True)
    print(result.stderr, end="", file=sys.stderr, flush=True)
    if "debug_layer=enabled" in result.stdout:
        assert "debug_errors=0 debug_warnings=0" in result.stdout, "D3D12 debug layer reported a message"
    return result


def draw(directory, compiler, renderer, backend, sources):
    record_path = os.environ.get("FORM_COVERAGE_RECORD")
    completed = set()
    for name, op, contended, _, _ in cases():
        src, cso, case = (directory / f"{name}.{ext}" for ext in ("hlsl", "cso", "case"))
        compile_shader = [compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src]
        for groups in (1, 2):
            initial, expected = mimg_atomic_reference(op, contended, groups)
            assert len(initial) == len(expected) == 336 and initial != expected
            src.write_text(sources[name], encoding="ascii")
            subprocess.run(compile_shader, check=True)
            print(f"{name} groups={groups}", flush=True)
            result = run_case(renderer, backend, cso, case, op, groups, initial, expected)
            if result.returncode:
                return result.returncode
            control, mutant, (wrong_op, compare_store) = mutants(op, sources[name])
            assert mutant != sources[name]
            extra = {"compare_store": compare_store} if compare_store else {}
            _, wrong = mimg_atomic_reference(wrong_op, contended, groups, **extra)
            first = next(i for i, (a, b) in enumerate(zip(expected, wrong, strict=True)) if a != b)
            mismatch = f"mip=0 word={first} expected={expected[first]:08x} actual={wrong[first]:08x}"
            src.write_text(mutant, encoding="ascii")
            subprocess.run(compile_shader, check=True)
            bad = run_case(renderer, backend, cso, case, op, groups, initial, expected)
            if bad.returncode != 1 or mismatch not in bad.stderr:
                raise AssertionError(f"{name} groups={groups}: {control} escaped detection ({mismatch})")
            print(f"{name} groups={groups} {control}: mismatched as expected", flush=True)
        completed.add(f"image_atomic_{op}")
    if record_path:
        record = Path(record_path)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("\n".join(sorted(completed)) + "\n", encoding="utf-8")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--binary", nargs=2, metavar=("LLVM_MC", "LLVM_OBJCOPY"))
    modes.add_argument("--draw", nargs=3, metavar=("DXC", "RENDERER", "ADAPTER"))
    parser.add_argument("--llvm", metavar="LIBLLVM", help="Also check disassembly and the container CLI")
    args = parser.parse_args()
    sources = controls()
    if not (args.binary or args.draw):
        return 0
    out = HERE.parent / "out"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compute_image_atomics_gfx1013_", dir=out) as tmp:
        directory = Path(tmp)
        if args.binary:
            binary(directory, *args.binary, args.llvm, sources)
            return 0
        return draw(directory, *args.draw, sources)


if __name__ == "__main__":
    raise SystemExit(main())
