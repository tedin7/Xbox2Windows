"""Check the compute store profile's v_max3_u32 and v_min3_u32 lowering with an original synthetic kernel."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base  # noqa: E402

aot = base.aot
# A hand-written variant of the LLVM 15 read-compute-write fixture (base.READ_COMPUTE), with the same guarded read of
# word gid from t0 and the same bindings, and cap = params[0].y:
#   v = src[gid]; d = v * 0x9e3779b9;
#   out[gid] = umax3(v, cap, d) ^ umin3(d, 0x8c800000, v);
# Every source word is at least 0x8b02a7f1 and the cap and the literal are above 0x80000000, while the product d
# falls on either side of 0x80000000, so a signed comparison picks a different source for many lanes.
MINMAX3_CAP = 0x8c000000
MINMAX3_LITERAL = 0x8c800000
MINMAX3_FACTOR = 0x9e3779b9
MAX3 = "v_max3_u32 v3, v1, s0, v2"
MIN3 = "v_min3_u32 v1, v2, 0x8c800000, v1"
READ_MINMAX3 = base.variant(
    base.READ_COMPUTE,
    ("v_and_b32_e32 v2, 0xffff, v1", "v_mul_lo_u32 v2, 0x9e3779b9, v1"),
    ("v_lshrrev_b32_e32 v1, 16, v1", ""),
    ("v_mad_u32_u24 v1, v2, v1, v0", f"{MAX3}\n{MIN3}"),
    ("v_min_u32_e32 v1, s0, v1", "v_xor_b32_e32 v1, v3, v1"),
)
BINDING = base.READ_BINDING
LOAD = base.READ_LOADS["read_copy"]
PRODUCT = f"(0x9e3779b9u * {LOAD})"
MAX3_WORD = f"max(max({LOAD}, params[0].y), {PRODUCT})"
MIN3_WORD = f"min(min({PRODUCT}, 0x8c800000u), {LOAD})"
STORE = f"        outputData.Store(tid.x * 4u + 0u, ({MAX3_WORD} ^ {MIN3_WORD}));"


def signed(word):
    return word - (1 << 32) if word >> 31 else word


def operands(gid):
    """The two source lists of one lane, from the instruction definitions: (v, cap, d) and (d, literal, v)."""
    word = base.source_word(gid)
    product = (word * MINMAX3_FACTOR) % (1 << 32)
    return [word, MINMAX3_CAP, product], [product, MINMAX3_LITERAL, word]


def minmax3_word(gid, *, high="umax3", low="umin3"):
    """Independent CPU reference of one lane. The ordering helpers sort the three words instead of nesting two-way
    max and min, and a wrong form signs the comparison, swaps the operation or drops the third source."""
    upper, lower = operands(gid)
    forms = {
        "umax3": lambda ops: sorted(ops)[-1], "umin3": lambda ops: sorted(ops)[0],
        "smax3": lambda ops: sorted(ops, key=signed)[-1], "smin3": lambda ops: sorted(ops, key=signed)[0],
        "umax2": lambda ops: sorted(ops[:2])[-1], "umin2": lambda ops: sorted(ops[:2])[0],
    }
    return forms[high](upper) ^ forms[low](lower)


def minmax3_words(groups, bound, **reference):
    words = [base.SENTINEL] * 320
    for gid in range(min(32 * groups, bound)):
        words[gid] = minmax3_word(gid, **reference)
    return words


# Each wrong reference must differ from the right one; the draw requires it to fail at the first such word.
WRONG = (("signed max3", dict(high="smax3")), ("signed min3", dict(low="smin3")),
         ("min3 for max3", dict(high="umin3")), ("max3 for min3", dict(low="umax3")),
         ("max3 without its third source", dict(high="umax2")), ("min3 without its third source", dict(low="umin2")))


def coverage():
    """The inputs of the largest dispatch must pick every source of each operation, straddle 0x80000000 and
    make the signed forms differ."""
    groups, bound = base.READ_RUNS[-1]
    for which, index, pick in (("max3", 0, max), ("min3", 1, min)):
        chosen = {(ops := operands(gid)[index]).index(pick(ops)) for gid in range(bound)}
        assert chosen == {0, 1, 2}, f"{which} does not pick each source: {chosen}"
    products = [operands(gid)[0][2] for gid in range(bound)]
    assert any(p >> 31 for p in products) and any(not p >> 31 for p in products), "the product must straddle 2^31"
    assert all(operands(gid)[0][0] >> 31 for gid in range(bound)), "every source word has its sign bit set"
    expected = minmax3_words(groups, bound)
    for label, reference in WRONG:
        assert expected != minmax3_words(groups, bound, **reference), f"{label} would go unseen"


def lower(*changes, **binding):
    return aot.lower(base.variant(READ_MINMAX3, *changes), 32, **(BINDING | binding)).splitlines()


def controls():
    """Translator only: exact output, operand forms, then one rejection per form outside the family."""
    coverage()
    lines = lower()
    assert lines[:3] == ["cbuffer Constants : register(b0) { uint4 params[4]; };",
                         "ByteAddressBuffer inputData : register(t0);",
                         "RWByteAddressBuffer outputData : register(u0);"], lines
    assert lines[-4:] == [base.GUARD_IF, STORE, "    }", "}"], lines
    # Sources keep their order: a negative inline constant, a dispatch index, an SGPR and a literal.
    assert lower((MAX3, "v_max3_u32 v3, -16, v0, s0"))[-3] == STORE.replace(
        MAX3_WORD, "max(max(0xfffffff0u, tid.x), params[0].y)")
    assert lower((MIN3, "v_min3_u32 v1, s0, 0x7fffffff, 64"))[-3] == STORE.replace(
        MIN3_WORD, "min(min(params[0].y, 0x7fffffffu), 0x00000040u)")
    # A copy of a result is the same program.
    assert lower((MIN3, "v_min3_u32 v4, v2, 0x8c800000, v1\nv_mov_b32_e32 v1, v4")) == lines
    unsupported, operand = "Unsupported compute instruction: ", "Unverified ALU operand"
    for message, changes in (
        # Signed and 16-bit forms are other instructions.
        (unsupported + "v_max3_i32", ((MAX3, MAX3.replace("u32", "i32")),)),
        (unsupported + "v_min3_i32", ((MIN3, MIN3.replace("u32", "i32")),)),
        (unsupported + "v_max3_u16", ((MAX3, MAX3.replace("u32", "u16")),)),
        (unsupported + "v_min3_u16", ((MIN3, MIN3.replace("u32", "u16")),)),
        # Output and source modifiers, a register range and a missing source.
        (unsupported + "v_max3_u32", ((MAX3, MAX3 + " clamp"),)),
        (unsupported + "v_min3_u32", ((MIN3, MIN3 + " clamp"),)),
        (unsupported + "v_max3_u32", ((MAX3, "v_max3_u32 v3, |v1|, s0, v2"),)),
        (unsupported + "v_max3_u32", ((MAX3, "v_max3_u32 v3, v[1:2], s0, v2"),)),
        (unsupported + "v_min3_u32", ((MIN3, "v_min3_u32 v1, v2, 0x8c800000"),)),
        # Sources that are not known words fail the three-source match: an unknown SGPR, an unknown VGPR, the
        # descriptor SGPR, a literal wider than 32 bits and a byte offset.
        (unsupported + "v_max3_u32 v3, v1, s20, v2", ((MAX3, "v_max3_u32 v3, v1, s20, v2"),)),
        (unsupported + "v_max3_u32 v3, v5, s0, v2", ((MAX3, "v_max3_u32 v3, v5, s0, v2"),)),
        (unsupported + "v_max3_u32 v3, v1, s8, v2", ((MAX3, "v_max3_u32 v3, v1, s8, v2"),)),
        (unsupported + "v_min3_u32", ((MIN3, "v_min3_u32 v1, v2, 0x18c800000, v1"),)),
        (unsupported + "v_min3_u32", ((MIN3, "v_lshlrev_b32_e32 v4, 2, v0\nv_min3_u32 v1, v2, v4, v1"),)),
        # A destination past v255.
        (operand, ((MAX3, "v_max3_u32 v256, v1, s0, v2"),)),
        # A result used as an address or as a guard operand.
        ("Unsupported store address", (("v1, v0, s[8:11]", "v0, v1, s[8:11]"),)),
        ("Unverified lane guard compare", (("v_xor_b32_e32 v1, v3, v1",
                                            "v_xor_b32_e32 v1, v3, v1\nv_cmp_gt_u32_e32 vcc_lo, s3, v3"),)),
        # Expressions are inline, so a result that keeps tripling is rejected.
        ("ALU expression too large", ((MAX3, MAX3 + "\nv_max3_u32 v3, v3, v3, v3" * 3),)),
    ):
        base.rejects(message, lower, *changes)


def mutations(source):
    """Translations that a wrong lowering would produce; each must fail on the GPU."""
    signed_max3 = f"asuint(max(max(asint({LOAD}), asint(params[0].y)), asint({PRODUCT})))"
    signed_min3 = f"asuint(min(min(asint({PRODUCT}), asint(0x8c800000u)), asint({LOAD})))"
    yield "signed max3", source.replace(MAX3_WORD, signed_max3)
    yield "signed min3", source.replace(MIN3_WORD, signed_min3)
    yield "min3 for max3", source.replace(MAX3_WORD, MAX3_WORD.replace("max(max(", "min(min("))
    yield "max3 for min3", source.replace(MIN3_WORD, MIN3_WORD.replace("min(min(", "max(max("))
    yield "max3 without its third source", source.replace(MAX3_WORD, f"max({LOAD}, params[0].y)")
    yield "min3 without its third source", source.replace(MIN3_WORD, f"min({PRODUCT}, 0x8c800000u)")
    yield "cap from another word", source.replace("params[0].y", "params[0].z")


def draw(directory, compiler, renderer, backend):
    src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
    src.write_text(aot.lower(READ_MINMAX3, 32, **BINDING), encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    cmd = [renderer, backend, "--compute-store-case", cso, case]
    constants = [0x5eed0000 + i for i in range(16)]
    constants[1] = MINMAX3_CAP

    def write(groups, bound, words):
        case.write_bytes(struct.pack("<337I", groups, *words, bound, *constants[1:]))

    # No lane passes, a split wave, and five groups with the fifth split, as the read harness runs them.
    for groups, bound in base.READ_RUNS:
        write(groups, bound, minmax3_words(groups, bound))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    groups, bound = base.READ_RUNS[-1]
    expected = minmax3_words(groups, bound)
    for label, reference in WRONG + (("lane past the bound", None),):
        wrong = minmax3_words(groups, bound + 1) if reference is None else minmax3_words(groups, bound, **reference)
        word = next(i for i, (a, b) in enumerate(zip(expected, wrong)) if a != b)
        write(groups, bound, wrong)
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected=" not in bad.stderr:
            raise AssertionError(f"read_minmax3: reference '{label}' escaped detection\n{bad.stderr}")
        print(f"read_minmax3: reference '{label}' fails at word {word}")
    write(groups, bound, expected)
    source = src.read_text(encoding="ascii")
    for label, mutated in mutations(source):
        if mutated == source:
            raise AssertionError(f"read_minmax3: mutation '{label}' did not change the source")
        bad_src, bad_cso = directory / "mutated.hlsl", directory / "mutated.cso"
        bad_src.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", bad_cso, bad_src], check=True)
        bad = subprocess.run([*cmd[:3], bad_cso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"read_minmax3: mutation '{label}' escaped detection\n{bad.stderr}")
        print(f"read_minmax3: mutation '{label}' fails")
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        opcodes = sorted({line.split()[0] for line in READ_MINMAX3 if line.strip()})
        path.write_text("".join(f"{op}\n" for op in opcodes), encoding="utf-8")
    return 0


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    controls()
    if mode == "controls" and not args:
        sys.exit(0)
    if mode != "draw" or len(args) != 3:
        sys.exit("usage: compute_alu_minmax3_check.py controls | draw <dxc> <renderer> <adapter>")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
