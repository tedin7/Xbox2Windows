"""Check v_bcnt_u32_b32 in the compute store profile: exact lowering, refusals and a WARP or GPU readback."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base  # noqa: E402

aot = base.aot
NAME = "read_bcnt"
MASK = 0xffffffff
# The key is params[0].y, a whole-word addend; the literal and the inline -16 are counted sources.
KEY = 0xa5c30f96
LITERAL = 0x2c5a96e1
INLINE = -16
# Read-compute-write with four v_bcnt_u32_b32 in place of the umin of the read_compute fixture (LLVM 15 output, see
# compute_store_check.py), hand-edited in LLVM 15 spelling (gfx10 has only the VOP3 encoding, printed unsuffixed):
#   p = (v & 0xffff) * (v >> 16) + gid;            // v = source word gid, as before
#   out[gid] = bcnt(-16) + bcnt(0x2c5a96e1) + bcnt(gid) + bcnt(p) + key;
# Each instruction is D = CountOneBits(S0) + S1 (RDNA2 ISA; LLVM AMDGPU V_BCNT_U32_B32_e64, ctpop(src0) + src1;
# Mesa ACO bit_count). The counted source is a result, the dispatch index, a 32-bit literal and a negative inline
# constant; the addend is an SGPR constant word and then the running result. The key keeps the sum a whole word,
# so swapping the operands or dropping the count changes every lane.
FIRST = "v_bcnt_u32_b32 v2, v1, s0"
LAST = f"v_bcnt_u32_b32 v1, {INLINE}, v2"
LINES = base.variant(
    base.READ_COMPUTE,
    ("v_min_u32_e32 v1, s0, v1",
     f"{FIRST}\nv_bcnt_u32_b32 v2, v0, v2\nv_bcnt_u32_b32 v2, 0x{LITERAL:08x}, v2\n{LAST}"),
)
BINDING = base.READ_BINDING
LOAD = base.READ_LOADS["read_copy"]
PRODUCT = f"((((0x0000ffffu & {LOAD}) & 0xffffffu) * (({LOAD} >> (0x00000010u & 31u)) & 0xffffffu)) + tid.x)"
STORE = (f"        outputData.Store(tid.x * 4u + 0u, (countbits(0xfffffff0u) + (countbits(0x{LITERAL:08x}u)"
         f" + (countbits(tid.x) + (countbits({PRODUCT}) + params[0].y)))));")
# Lane 0, worked by hand: v = 0x8b02a7f1, p = 0xa7f1 * 0x8b02 = 0x5b312ae2 (5 + 3 + 3 + 4 = 15 ones), gid = 0
# (none), 0x2c5a96e1 has 15 ones and 0xfffffff0 has 28, so the word is 0xa5c30f96 + 58 = 0xa5c30fd0.
LANE0 = 0xa5c30fd0


def ones(word):
    return bin(word & MASK).count("1")


def bcnt_word(gid, *, count=ones, swap=False, drop=(), inline=INLINE):
    """Independent CPU reference for one lane, from the instruction definition D = CountOneBits(S0) + S1.

    count replaces the bit count, swap counts the addend instead, drop removes named sources (a dropped counted source
    counts zero bits, a dropped addend is 0), inline replaces the inline constant.
    """
    word = base.source_word(gid)
    product = ((word & 0xffff) * (word >> 16) + gid) & MASK

    def bcnt(counted, addend):
        if swap:
            counted, addend = addend, counted
        return (count(counted) + addend) & MASK

    result = bcnt(0 if "product" in drop else product, 0 if "key" in drop else KEY)
    result = bcnt(0 if "gid" in drop else gid, result)
    result = bcnt(0 if "literal" in drop else LITERAL, result)
    return bcnt(0 if "inline" in drop else inline & MASK, result)


def bcnt_words(groups, bound, **reference):
    words = [base.SENTINEL] * 320
    for gid in range(min(32 * groups, bound)):
        words[gid] = bcnt_word(gid, **reference)
    return words


WRONG = (("plain add for the count", dict(count=lambda w: w & MASK)),
         ("count of zero bits", dict(count=lambda w: 32 - ones(w))),
         ("operands swapped", dict(swap=True)),
         ("product dropped", dict(drop=("product",))), ("key dropped", dict(drop=("key",))),
         ("dispatch index dropped", dict(drop=("gid",))), ("literal dropped", dict(drop=("literal",))),
         ("inline dropped", dict(drop=("inline",))), ("inline as +16", dict(inline=16)))
# Translations that a wrong template or operand would produce; each must fail on the GPU.
MUTATIONS = (("first bit for the count", f"(countbits({PRODUCT})", f"(firstbitlow({PRODUCT})"),
             ("count dropped", f"(countbits({PRODUCT})", f"(({PRODUCT})"),
             ("key from params[0].z", " + params[0].y)", " + params[0].z)"),
             ("dispatch index dropped", "(countbits(tid.x) + ", "(countbits(0u) + "),
             ("literal one bit off", f"countbits(0x{LITERAL:08x}u)", f"countbits(0x{LITERAL ^ 1:08x}u)"),
             ("inline as +16", "countbits(0xfffffff0u)", "countbits(0x00000010u)"))


def lower(*changes, **binding):
    return aot.lower(base.variant(LINES, *changes), 32, **(BINDING | binding)).splitlines()


def controls():
    """Exact output, then one rejection per opcode, modifier or operand outside the form."""
    assert bcnt_word(0) == LANE0, hex(bcnt_word(0))
    lines = lower()
    assert lines[-4:] == [base.GUARD_IF, STORE, "    }", "}"], lines
    # A copy of a result is the same program.
    assert lower((FIRST, FIRST + "\nv_mov_b32_e32 v3, v2"), ("v_bcnt_u32_b32 v2, v0, v2", "v_bcnt_u32_b32 v2, v0, v3")) == lines
    # An SGPR may be counted, a VGPR or an inline constant may be the addend, and a literal may be the addend.
    assert lower((FIRST, "v_bcnt_u32_b32 v2, s0, v1"))[-3] == STORE.replace(
        f"(countbits({PRODUCT}) + params[0].y)", f"(countbits(params[0].y) + {PRODUCT})")
    assert lower((LAST, "v_bcnt_u32_b32 v1, v2, 7"))[-3].startswith(
        "        outputData.Store(tid.x * 4u + 0u, (countbits((countbits(0x2c5a96e1u) + ")
    assert lower((LAST, "v_bcnt_u32_b32 v1, v2, 0x12345678"))[-3].endswith(" + 0x12345678u));")

    unsupported, operand = "Unsupported compute instruction: ", "Unverified ALU operand"
    for message, changes in (
        # Other bit-count opcodes and explicit encoding suffixes (LLVM refuses _e32; _e64 is printed unsuffixed).
        (unsupported + "v_bcnt_u32_b32_e32", ((FIRST, "v_bcnt_u32_b32_e32 v2, v1, v3"),)),
        (unsupported + "v_bcnt_u32_b32_e64", ((FIRST, "v_bcnt_u32_b32_e64 v2, v1, s0"),)),
        (unsupported + "v_mbcnt_lo_u32_b32", ((FIRST, "v_mbcnt_lo_u32_b32 v2, v1, s0"),)),
        # The scalar count takes exec_lo or a known scalar word, not the high half or an unknown register.
        ("Unsupported active lane count source: exec_hi", ((FIRST, "s_bcnt1_i32_b32 s9, exec_hi"),)),
        # Source modifiers and an output modifier, which LLVM does not accept for this opcode.
        (unsupported + "v_bcnt_u32_b32", ((FIRST, "v_bcnt_u32_b32 v2, -v1, s0"),)),
        (unsupported + "v_bcnt_u32_b32", ((FIRST, "v_bcnt_u32_b32 v2, |v1|, s0"),)),
        (unsupported + "v_bcnt_u32_b32", ((FIRST, "v_bcnt_u32_b32 v2, v1, neg(s0)"),)),
        (unsupported + "v_bcnt_u32_b32", ((FIRST, FIRST + " clamp"),)),
        # A register range, and a third source.
        (unsupported + "v_bcnt_u32_b32", ((FIRST, "v_bcnt_u32_b32 v[2:3], v1, s0"),)),
        (unsupported + "v_bcnt_u32_b32", ((FIRST, FIRST + ", v0"),)),
        # Sources that are not known words: an unknown, descriptor or exec-mask SGPR, an unknown VGPR and a literal
        # wider than 32 bits. The two-source VOP3 group stops these at the operand check.
        (operand, ((FIRST, "v_bcnt_u32_b32 v2, v1, s20"),)),
        (operand, ((FIRST, "v_bcnt_u32_b32 v2, s8, v1"),)),
        (operand, ((FIRST, "v_bcnt_u32_b32 v2, v1, s2"),)),
        (operand, ((FIRST, "v_bcnt_u32_b32 v2, v5, s0"),)),
        (operand, ((LAST, "v_bcnt_u32_b32 v1, 0x1ffffffff, v2"),)),
        # A destination past v255.
        (operand, ((FIRST, "v_bcnt_u32_b32 v256, v1, s0"),)),
        # A result is data only: not a store or read address and not a lane-guard compare operand.
        ("Unsupported store address", (("v1, v0, s[8:11]", "v0, v1, s[8:11]"),)),
        ("Unsupported read address", ((LAST, LAST + "\nbuffer_load_dword v3, v1, s[12:15], 0 offen"),)),
        ("Unverified lane guard compare", ((LAST, LAST + "\nv_cmp_gt_u32_e32 vcc_lo, s3, v1"),)),
        # Expressions are inline, so a result that keeps doubling is rejected.
        ("ALU expression too large", ((LAST, LAST + "\nv_bcnt_u32_b32 v1, v1, v1" * 8),)),
    ):
        base.rejects(message, lower, *changes)


def draw(directory, compiler, renderer, backend):
    """Translate, compile and dispatch the fixture, then require wrong references and mutated sources to fail."""
    src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
    source = aot.lower(LINES, 32, **BINDING)
    src.write_text(source, encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    cmd = [renderer, backend, "--compute-store-case", cso, case]
    constants = [0x5eed0000 + i for i in range(16)]
    constants[1] = KEY

    def write(groups, bound, words):
        case.write_bytes(struct.pack("<337I", groups, *words, bound, *constants[1:]))

    for groups, bound in base.READ_RUNS:
        write(groups, bound, bcnt_words(groups, bound))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    groups, bound = base.READ_RUNS[-1]
    expected = bcnt_words(groups, bound)
    assert expected[0] == LANE0
    # Each wrong reference, one word off and the first lane past the bound must fail at the first differing word.
    off = list(expected)
    off[bound - 1] ^= 1
    for label, reference in (("one word off", off), ("lane past the bound", bcnt_words(groups, bound + 1)),
                             *((label, bcnt_words(groups, bound, **wrong)) for label, wrong in WRONG)):
        word = next(i for i, (a, b) in enumerate(zip(expected, reference)) if a != b)
        write(groups, bound, reference)
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected=" not in bad.stderr:
            raise AssertionError(f"{NAME}: reference '{label}' escaped detection\n{bad.stderr}")
    write(groups, bound, expected)
    for label, old, new in MUTATIONS:
        mutated = source.replace(old, new, 1)
        if mutated == source:
            raise AssertionError(f"{NAME}: mutation '{label}' did not change the source")
        bad_src, bad_cso = directory / "mutated.hlsl", directory / "mutated.cso"
        bad_src.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", bad_cso, bad_src], check=True)
        bad = subprocess.run([*cmd[:3], bad_cso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"{NAME}: mutation '{label}' escaped detection\n{bad.stderr}")
    print(f"{NAME}: {len(base.READ_RUNS)} dispatches, {len(WRONG) + 2} wrong references and "
          f"{len(MUTATIONS)} mutated sources checked")
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        opcodes = sorted({line.split()[0] for line in LINES if line.strip()})
        path.write_text("".join(f"{op}\n" for op in opcodes), encoding="utf-8")
    return 0


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    controls()
    if mode == "controls":
        sys.exit(0)
    if mode != "draw" or len(args) != 3:
        sys.exit("usage: compute_alu_bcnt_check.py controls | draw <dxc> <renderer> <adapter>")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
