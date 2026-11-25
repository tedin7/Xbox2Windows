"""Check s_absdiff_i32 in the compute store profile's scalar ALU subset, translated and dispatched."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base  # noqa: E402  (also puts tools/ on the path)

aot = base.aot

# The guarded store of the read fixtures, without the read: each lane below the bound stores four scalar results,
# a 16-byte element per dispatch index. s[16:19] and s20 are constant words 1-5. In order: two registers, a
# register and a 32-bit literal (INT_MIN as src1), and an inline constant as src0 with the first result as src1.
# Text in LLVM 15 spelling.
ABSDIFF = [
    "s_mov_b32 s2, s1",
    "s_mov_b32 s1, 0",
    "v_lshl_add_u32 v0, s2, 5, v0",
    "s_load_dwordx4 s[4:7], s[0:1], 0x20",
    "s_waitcnt lgkmcnt(0)",
    "s_buffer_load_dword s3, s[4:7], 0x0",
    "s_waitcnt lgkmcnt(0)",
    "v_cmp_gt_u32_e32 vcc_lo, s3, v0",
    "s_and_saveexec_b32 s2, vcc_lo",
    "s_cbranch_execz .LBB0_2",
    "s_load_dwordx4 s[8:11], s[0:1], 0x0",
    "s_buffer_load_dwordx4 s[16:19], s[4:7], 0x4",
    "s_buffer_load_dword s20, s[4:7], 0x14",
    "s_waitcnt lgkmcnt(0)",
    "s_absdiff_i32 s22, s16, s17",
    "s_absdiff_i32 s23, s18, s19",
    "s_absdiff_i32 s24, s20, 0x80000000",
    "s_absdiff_i32 s25, -7, s22",
    "v_mov_b32_e32 v1, s22",
    "v_mov_b32_e32 v2, s23",
    "v_mov_b32_e32 v3, s24",
    "v_mov_b32_e32 v4, s25",
    "buffer_store_dwordx4 v[1:4], v0, s[8:11], 0 idxen",
    ".LBB0_2:",
    "s_endpgm",
]
BINDING = base.GUARD_BINDING | dict(stride=16)
FIRST = "asuint(abs(asint(params[0].y - params[0].z)))"
RESULTS = (FIRST, "asuint(abs(asint(params[0].w - params[1].x)))",
           "asuint(abs(asint(params[1].y - 0x80000000u)))", f"asuint(abs(asint(0xfffffff9u - {FIRST})))")


def absdiff_i32(a, b):
    """The RDNA ISA definition (opcode 44, SOP2): D.i = S0.i - S1.i; if (D.i < 0) D.i = -D.i, in 32-bit words."""
    d = (a - b) & 0xffffffff
    return -d & 0xffffffff if d >> 31 else d


# The functional examples printed with the definition in the RDNA 1 (25 September 2020) and RDNA 2 (30 November 2020)
# ISA references.
ISA_EXAMPLES = (((0x00000002, 0x00000005), 0x00000003), ((0xffffffff, 0x00000000), 0x00000001),
                ((0x80000000, 0x00000000), 0x80000000), ((0x80000000, 0x00000001), 0x7fffffff),
                ((0x80000000, 0xffffffff), 0x7fffffff), ((0x80000000, 0xfffffffe), 0x7ffffffe))

# Constant words 1-5 of each dispatch and the four words worked out by hand from the definition.
# A: a negative difference, INT_MIN - 0 (the negation wraps back to 0x80000000), INT_MAX - INT_MIN (the 32-bit
#    difference is -1, so 1, not 2^32 - 1), and -7 - 3.
# B: equal registers, INT_MIN - (-1), equal INT_MIN operands, and -7 - 0.
# C: -10 - 5 and 5 - (-10), 0 - INT_MIN (wraps to INT_MIN again), and -7 - 15.
SETS = {
    "A": ((0x00000002, 0x00000005, 0x80000000, 0x00000000, 0x7fffffff), (0x00000003, 0x80000000, 0x00000001, 0x0000000a)),
    "B": ((0x12345678, 0x12345678, 0x80000000, 0xffffffff, 0x80000000), (0x00000000, 0x7fffffff, 0x00000000, 0x00000007)),
    "C": ((0xfffffff6, 0x00000005, 0x00000005, 0xfffffff6, 0x00000000), (0x0000000f, 0x0000000f, 0x80000000, 0x00000016)),
}
# (groups, bound, constant set); 80 lanes of four words fill the 320-word UAV.
RUNS = ((1, 17, "A"), (2, 50, "B"), (3, 80, "C"))


def lane_words(constants, op=absdiff_i32):
    c1, c2, c3, c4, c5 = constants
    first = op(c1, c2)
    return [first, op(c3, c4), op(c5, 0x80000000), op(0xfffffff9, first)]


def to_signed(word):
    return word - (1 << 32) if word >> 31 else word


# Wrong definitions; each must fail the run at the first word where it differs from the right one.
WRONG = (
    ("difference without abs", lambda a, b: (a - b) & 0xffffffff),
    ("unsigned comparison", lambda a, b: (a - b if a > b else b - a) & 0xffffffff),
    ("difference not wrapped to 32 bits", lambda a, b: abs(to_signed(a) - to_signed(b)) & 0xffffffff),
    ("INT_MIN saturated", lambda a, b: min(absdiff_i32(a, b), 0x7fffffff)),
)


def words(groups, bound, constants, op=absdiff_i32):
    result = [base.SENTINEL] * 320
    lane = lane_words(constants, op)
    for gid in range(min(32 * groups, bound)):
        result[4 * gid:4 * gid + 4] = lane
    return result


def lower(*changes, **binding):
    return aot.lower(base.variant(ABSDIFF, *changes), 32, **(BINDING | binding)).splitlines()


def controls():
    """Reference, exact output, then one rejection per operand or SCC use outside the subset."""
    for (a, b), expected in ISA_EXAMPLES:
        assert absdiff_i32(a, b) == expected, (hex(a), hex(b))
    for name, (constants, expected) in SETS.items():
        assert lane_words(constants) == list(expected), name
        for label, op in WRONG:
            assert any(lane_words(constants, op) != list(expected) for constants, expected in SETS.values()), label

    lines = lower()
    assert lines[-4:] == [base.GUARD_IF, f"        outputData.Store4(tid.x * 16u + 0u, uint4({', '.join(RESULTS)}));",
                          "    }", "}"], lines
    # A copy of a result is the same program, and the inline constant -7 is the word 0xfffffff9.
    assert lower(("v_mov_b32_e32 v1, s22", "s_mov_b32 s26, s22\nv_mov_b32_e32 v1, s26")) == lines
    assert lower(("s_absdiff_i32 s25, -7, s22", "s_absdiff_i32 s25, 0xfffffff9, s22")) == lines
    # Source order is kept: the result is |src0 - src1| with src0 first.
    assert "asuint(abs(asint(params[0].z - params[0].y)))" in lower(
        ("s_absdiff_i32 s22, s16, s17", "s_absdiff_i32 s22, s17, s16"))[-3]

    unsupported, operand = "Unsupported compute instruction: ", "Unverified scalar ALU operand"
    for message, changes in (
        # Sources that are not known words: an unwritten SGPR, a descriptor SGPR, the saved exec mask, a constant
        # word still waiting for its s_waitcnt, and a literal wider than 32 bits.
        (operand, (("s_absdiff_i32 s22, s16, s17", "s_absdiff_i32 s22, s30, s17"),)),
        (operand, (("s_absdiff_i32 s22, s16, s17", "s_absdiff_i32 s22, s16, s8"),)),
        (operand, (("s_absdiff_i32 s22, s16, s17", "s_absdiff_i32 s22, s2, s17"),)),
        (operand, (("s_absdiff_i32 s22, s16, s17", "s_buffer_load_dword s21, s[4:7], 0x18\n"
                                                   "s_absdiff_i32 s22, s16, s21"),)),
        (operand, (("s_absdiff_i32 s24, s20, 0x80000000", "s_absdiff_i32 s24, s20, 0x180000000"),)),
        # A destination past s105, and a destination still waiting for a load.
        (operand, (("s_absdiff_i32 s22, s16, s17", "s_absdiff_i32 s106, s16, s17"),)),
        (operand, (("s_absdiff_i32 s22, s16, s17", "s_buffer_load_dword s22, s[4:7], 0x18\n"
                                                   "s_absdiff_i32 s22, s16, s17"),)),
        # A vector source, a float inline constant, a 64-bit destination and a spelling outside the subset.
        (unsupported + "s_absdiff_i32", (("s_absdiff_i32 s22, s16, s17", "s_absdiff_i32 s22, v0, s17"),)),
        (unsupported + "s_absdiff_i32", (("s_absdiff_i32 s24, s20, 0x80000000", "s_absdiff_i32 s24, s20, 0.5"),)),
        (unsupported + "s_absdiff_i32", (("s_absdiff_i32 s22, s16, s17", "s_absdiff_i32 s[22:23], s16, s17"),)),
        (unsupported + "s_absdiff_u32", (("s_absdiff_i32 s22, s16, s17", "s_absdiff_u32 s22, s16, s17"),)),
        # SCC is written but not modelled, so nothing may read it: a select and a branch on it are refused.
        ("Unverified SCC for scalar select", (("v_mov_b32_e32 v4, s25", "s_cselect_b32 s26, s25, 0\nv_mov_b32_e32 v4, s26"),)),
        (unsupported + "s_cbranch_scc0", (("v_mov_b32_e32 v1, s22", "s_cbranch_scc0 .LBB0_2\nv_mov_b32_e32 v1, s22"),)),
        # A scalar result is data only: not a store address and not a lane-guard compare operand.
        ("Unsupported store address", (("v_mov_b32_e32 v1, s22", "v_mov_b32_e32 v1, s22\nv_mov_b32_e32 v5, s22"),
                                       ("v[1:4], v0, s[8:11]", "v[1:4], v5, s[8:11]"))),
        ("Unverified lane guard compare", (("v_mov_b32_e32 v1, s22",
                                            "v_mov_b32_e32 v1, s22\nv_cmp_gt_u32_e32 vcc_lo, s22, v0"),)),
        # Expressions are inline, so a result that keeps doubling is rejected.
        ("ALU expression too large", (("s_absdiff_i32 s25, -7, s22",
                                       "s_absdiff_i32 s25, -7, s22" + "\ns_absdiff_i32 s25, s25, s25" * 6),)),
    ):
        base.rejects(message, lower, *changes)


def draw(directory, compiler, renderer, backend):
    src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
    src.write_text(aot.lower(ABSDIFF, 32, **BINDING), encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    cmd = [renderer, backend, "--compute-store-case", cso, case]

    def write(groups, bound, constants, expected):
        # Constant word 0 is the bound; words 6-15 are filler the program does not read.
        filler = [0x5eed0000 + i for i in range(6, 16)]
        case.write_bytes(struct.pack("<337I", groups, *expected, bound, *constants, *filler))

    for groups, bound, name in RUNS:
        constants = SETS[name][0]
        write(groups, bound, constants, words(groups, bound, constants))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    for label, op in WRONG:
        groups, bound, name = next(run for run in RUNS
                                   if lane_words(SETS[run[2]][0], op) != lane_words(SETS[run[2]][0]))
        constants = SETS[name][0]
        expected, wrong = words(groups, bound, constants), words(groups, bound, constants, op)
        word = next(i for i, (a, b) in enumerate(zip(expected, wrong)) if a != b)
        write(groups, bound, constants, wrong)
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected=" not in bad.stderr:
            raise AssertionError(f"reference '{label}' escaped detection\n{bad.stderr}")
        print(f"{label}: set {name}, word={word} expected={wrong[word]:08x} actual={expected[word]:08x}")
    # The case file holds the last run (set C) with the right words: a mutated translation must fail on it.
    groups, bound, name = RUNS[-1]
    write(groups, bound, SETS[name][0], words(groups, bound, SETS[name][0]))
    source = src.read_text(encoding="ascii")
    for label, old, new in (("abs dropped", "asuint(abs(asint(", "asuint((asint("),
                            ("INT_MIN literal changed", "- 0x80000000u", "- 0x7fffffffu"),
                            ("inline constant changed", "(0xfffffff9u - ", "(0x00000000u - ")):
        mutated = source.replace(old, new)
        if mutated == source:
            raise AssertionError(f"mutation '{label}' did not change the source")
        msrc, mcso = directory / "mutated.hlsl", directory / "mutated.cso"
        msrc.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", mcso, msrc], check=True)
        bad = subprocess.run([*cmd[:3], mcso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"mutation '{label}' escaped detection\n{bad.stderr}")
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        opcodes = sorted({line.split()[0] for line in ABSDIFF if line.strip()})
        path.write_text("".join(f"{op}\n" for op in opcodes), encoding="utf-8")
    return 0


if __name__ == "__main__":
    controls()
    mode, *args = sys.argv[1:]
    if mode == "controls":
        assert not args, args
        sys.exit(0)
    if mode != "draw" or len(args) != 3:
        sys.exit("usage: compute_salu_absdiff_check.py controls | draw <dxc> <renderer> <adapter>")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
