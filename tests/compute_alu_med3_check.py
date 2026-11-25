"""Check v_med3_u32 and v_med3_i32 between a compute read and its store (#1546).

Modes: "controls" runs the translator checks only; "draw <dxc> <renderer> <adapter>" also dispatches the fixture
and compares the UAV readback with words from an independent CPU reference.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

import compute_store_check as base
from compute_store_check import aot, rejects, variant

MASK = 0xffffffff
INT_MIN = 0x80000000
# cap = params[0].y. It lies between 0 and 0x80000000, so a shifted word above it, below it or past 0x80000000
# (negative as i32) makes a different operand the median, and the unsigned and signed medians differ.
CAP = 0x3a5c0f1d
# Hand-written VOP3 forms on the read_compute fixture of compute_store_check.py (same guard, read, store and
# bindings), in LLVM 15 gfx1013 spelling. w is the source word of the lane and a = w << 9, whose bit 31 differs
# between lanes:
#   u = umed3(a, cap, 0x80000000)    i = smed3(a, cap, 0x80000000)
#   p = smed3(a, w, a)               q = umed3(-1, w, w)          (equal operands: p = a, q = w)
#   out = (u - i) + (p ^ q)
# Every source form appears: VGPR, SGPR, inline constant and literal; the equal operands sit in src0/src2 and
# src1/src2.
READ_MED3 = variant(
    base.READ_COMPUTE,
    ("v_and_b32_e32 v2, 0xffff, v1", "v_lshlrev_b32_e32 v2, 9, v1"),
    ("v_lshrrev_b32_e32 v1, 16, v1", "v_med3_u32 v3, v2, s0, 0x80000000\nv_med3_i32 v4, v2, s0, 0x80000000"),
    ("v_mad_u32_u24 v1, v2, v1, v0", "v_med3_i32 v5, v2, v1, v2\nv_med3_u32 v6, -1, v1, v1\n"
                                     "v_sub_nc_u32_e32 v3, v3, v4\nv_xor_b32_e32 v5, v5, v6"),
    ("v_min_u32_e32 v1, s0, v1", "v_add_nc_u32_e32 v1, v3, v5"),
)
CASE = ("read_med3", READ_MED3, 32, base.READ_BINDING)


def signed(word):
    return word - (1 << 32) if word >> 31 else word


def med3(values, signed_order, rank=1):
    """The rank-th smallest of three words (1 is the median), in unsigned or two's-complement order."""
    return sorted(values, key=signed if signed_order else None)[rank]


def med3_word(gid, *, order=(False, True, True, False), rank=1):
    """Independent CPU reference of the read_med3 result for one lane; order[k] is True where the k-th v_med3 of the
    program compares signed, and rank 0 or 2 turns every median into a minimum or a maximum."""
    word = base.source_word(gid)
    shifted = (word << 9) & MASK
    u = med3([shifted, CAP, INT_MIN], order[0], rank)
    i = med3([shifted, CAP, INT_MIN], order[1], rank)
    p = med3([shifted, word, shifted], order[2], rank)
    q = med3([MASK, word, word], order[3], rank)
    return ((u - i) + (p ^ q)) & MASK


def med3_words(groups, bound, **reference):
    words = [base.SENTINEL] * 320
    for gid in range(min(32 * groups, bound)):
        words[gid] = med3_word(gid, **reference)
    return words


# Worked by hand for lane 0: w = 0x8b02a7f1, a = 0x054fe200 (below cap). Unsigned, 0x80000000 is the largest, so
# u = cap; signed it is the smallest, so i = a. p ^ q = a ^ w = 0x8e4d45f1, and
# out = (0x3a5c0f1d - 0x054fe200) + 0x8e4d45f1 = 0x350c2d1d + 0x8e4d45f1 = 0xc359730e.
LANE0 = 0xc359730e
WRONG_REFERENCES = (
    ("v_med3_u32 compared signed", dict(order=(True, True, True, False))),
    ("v_med3_i32 compared unsigned", dict(order=(False, False, True, False))),
    ("minimum for median", dict(rank=0)),
    ("maximum for median", dict(rank=2)),
)


def lower(lines=READ_MED3, **binding):
    return aot.lower(lines, CASE[2], **(CASE[3] | binding)).splitlines()


def controls():
    """Exact output, the reference's coverage, then one rejection per operand form or modifier outside the subset."""
    assert med3_word(0) == LANE0, hex(med3_word(0))
    assert base.source_word(0) == 0x8b02a7f1
    # Over the largest dispatch, a lies on each side of cap and of 0x80000000, so each of a, cap and 0x80000000
    # is the unsigned median in some lane, each of a and cap the signed one, and the two medians differ.
    shifted = [(base.source_word(g) << 9) & MASK for g in range(150)]
    u = [med3([a, CAP, INT_MIN], False) for a in shifted]
    i = [med3([a, CAP, INT_MIN], True) for a in shifted]
    assert {INT_MIN, CAP} < set(u) and any(x == a != CAP for x, a in zip(u, shifted)), "an unsigned median is unseen"
    assert CAP in i and any(x == a != CAP for x, a in zip(i, shifted)), "a signed median is unseen"
    assert all(x != y for x, y in zip(u, i)), "u and i must differ in every lane"
    expected = med3_words(5, 150)
    for label, reference in WRONG_REFERENCES:
        assert expected != med3_words(5, 150, **reference), f"{label} would go unseen"

    lines = lower()
    load = base.READ_LOADS["read_copy"]
    a = f"({load} << (0x00000009u & 31u))"
    lit, cap, ones = "0x80000000u", "params[0].y", "0xffffffffu"
    umed = f"max(min({a}, {cap}), min(max({a}, {cap}), {lit}))"
    smed = (f"asuint(max(min(asint({a}), asint({cap})), min(max(asint({a}), asint({cap})), asint({lit}))))")
    tie_s = f"asuint(max(min(asint({a}), asint({load})), min(max(asint({a}), asint({load})), asint({a}))))"
    tie_u = f"max(min({ones}, {load}), min(max({ones}, {load}), {load}))"
    store = f"(({umed} - {smed}) + ({tie_s} ^ {tie_u}))"
    assert lines[-4:] == [base.GUARD_IF, f"        outputData.Store(tid.x * 4u + 0u, {store});", "    }", "}"], lines
    assert len(store) <= aot.ALU_EXPRESSION_LIMIT

    med_u, med_i = "v_med3_u32 v3, v2, s0, 0x80000000", "v_med3_i32 v4, v2, s0, 0x80000000"
    unsupported_u, unsupported_i = (f"Unsupported compute instruction: v_med3_{kind}" for kind in ("u32", "i32"))
    for message, changes in (
        # Modifiers. LLVM has no clamp, neg or abs for the integer v_med3 forms, so these are not verified encodings.
        (unsupported_u, ((med_u, med_u + " clamp"),)),
        (unsupported_i, ((med_i, med_i.replace("v2,", "-v2,")),)),
        (unsupported_u, ((med_u, med_u.replace("v2,", "|v2|,")),)),
        (unsupported_i, ((med_i, med_i + " op_sel:[1,0,0,0]"),)),
        # Other widths and types of the median, and a register range.
        ("Unsupported compute instruction: v_med3_u16", ((med_u, med_u.replace("_u32", "_u16")),)),
        ("Unsupported compute instruction: v_med3_i16", ((med_i, med_i.replace("_i32", "_i16")),)),
        ("Unsupported compute instruction: v_med3_f32", ((med_u, med_u.replace("_u32", "_f32")),)),
        (unsupported_u, ((med_u, med_u.replace("v2,", "v[2:3],")),)),
        # Sources that are not known words: an unknown, pending or descriptor SGPR, an unknown VGPR and a literal
        # wider than 32 bits. They fail the known-word guard of the three-source matcher, so no other form applies.
        (unsupported_u, ((med_u, med_u.replace("s0", "s20")),)),
        (unsupported_i, ((med_i, med_i.replace("s0", "s8")),)),
        (unsupported_u, (("s_buffer_load_dword s0, s[4:7], 0x4", ""),
                         ("v_lshlrev_b32_e32 v2", "s_buffer_load_dword s0, s[4:7], 0x4\nv_lshlrev_b32_e32 v2"))),
        (unsupported_i, (("v_med3_i32 v5, v2, v1, v2", "v_med3_i32 v5, v2, v1, v7"),)),
        (unsupported_u, ((med_u, med_u.replace("0x80000000", "0x180000000")),)),
        # A destination past v255.
        ("Unverified ALU operand", ((med_u, med_u.replace("v3,", "v256,")),)),
        # A result used as an address or as a guard operand.
        ("Unsupported store address", (("v1, v0, s[8:11]", "v0, v1, s[8:11]"),)),
        ("Unverified lane guard compare", (("v_add_nc_u32_e32 v1, v3, v5", "v_add_nc_u32_e32 v1, v3, v5\n"
                                            "v_cmp_gt_u32_e32 vcc_lo, s3, v1"),)),
        # Each median repeats two operands, so a chain of medians on medians soon exceeds the inline limit.
        ("ALU expression too large", (("v_add_nc_u32_e32 v1, v3, v5", "v_add_nc_u32_e32 v1, v3, v5"
                                       + "\nv_med3_u32 v1, v1, v1, s0" * 2),)),
    ):
        rejects(message, lower, variant(READ_MED3, *changes))


def mutations():
    """Translations that a wrong median would produce; each must fail on the GPU."""
    alu = aot.ALU
    minimum = "min(min({0}, {1}), {2})"
    for label, templates in (("v_med3_i32 lowered unsigned", {"v_med3_i32": alu["v_med3_u32"]}),
                             ("v_med3_u32 lowered signed", {"v_med3_u32": alu["v_med3_i32"]}),
                             ("minimum for median", {"v_med3_u32": minimum, "v_med3_i32": minimum}),
                             ("third operand dropped", {"v_med3_u32": "max(min({0}, {1}), max({0}, {1}))"})):
        with patch.dict(aot.ALU, templates):
            yield label, "\n".join(lower()) + "\n"


def draw(directory, compiler, renderer, backend):
    src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
    name, lines, width, binding = CASE
    source = aot.lower(lines, width, **binding)
    src.write_text(source, encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    cmd = [renderer, backend, "--compute-store-case", cso, case]
    constants = [0x5eed0000 + i for i in range(16)]
    constants[1] = CAP

    def write(groups, bound, words):
        case.write_bytes(struct.pack("<337I", groups, *words, bound, *constants[1:]))

    for groups, bound in base.READ_RUNS:
        write(groups, bound, med3_words(groups, bound))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    groups, bound = base.READ_RUNS[-1]
    expected = med3_words(groups, bound)
    # Each wrong reference, and the first lane past the bound, must fail the run at its first differing word.
    for label, reference in (*((label, med3_words(groups, bound, **r)) for label, r in WRONG_REFERENCES),
                             ("lane past the bound", med3_words(groups, bound + 1))):
        word = next(i for i, (x, y) in enumerate(zip(expected, reference)) if x != y)
        write(groups, bound, reference)
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected=" not in bad.stderr:
            raise AssertionError(f"{name}: reference '{label}' escaped detection\n{bad.stderr}")
    write(groups, bound, expected)
    for label, mutated in mutations():
        if mutated.strip() == source.strip():
            raise AssertionError(f"{name}: mutation '{label}' did not change the source")
        mutated_src, mutated_cso = directory / "mutated.hlsl", directory / "mutated.cso"
        mutated_src.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", mutated_cso, mutated_src], check=True)
        bad = subprocess.run([*cmd[:3], mutated_cso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"{name}: mutation '{label}' escaped detection")
    print(f"{name}: {bound} lanes match; {len(WRONG_REFERENCES) + 1} reference and 4 source mutations rejected")
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        opcodes = sorted({line.split()[0] for line in lines if line.strip()})
        path.write_text("".join(f"{op}\n" for op in opcodes), encoding="utf-8")
    return 0


if __name__ == "__main__":
    controls()
    mode, *args = sys.argv[1:]
    if mode == "controls" and not args:
        sys.exit(0)
    if mode != "draw" or len(args) != 3:
        sys.exit("usage: compute_alu_med3_check.py controls | draw <dxc> <renderer> <adapter>")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
