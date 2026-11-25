"""Check v_xor3_b32 in the compute store profile: exact lowering, refusals and a WARP or GPU readback."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base  # noqa: E402

aot = base.aot
NAME = "read_xor3"
MASK = 0xffffffff
# The key is params[0].y. It, the literal and the inline -16 each change every lane's word, and their bits overlap
# the product's, so an OR, an AND or an add in place of the XOR, or a dropped source, changes the readback.
KEY = 0xa5c30f96
LITERAL = 0x2c5a96e1
INLINE = -16
# Read-compute-write with two v_xor3_b32 in place of the umin of the read_compute fixture (LLVM 15 output, see
# compute_store_check.py), hand-edited in LLVM 15 spelling:
#   p = (v & 0xffff) * (v >> 16) + gid;            // v = source word gid, as before
#   out[gid] = (p ^ key ^ gid) ^ 0x2c5a96e1 ^ -16;
# The sources cover VGPRs (a result and the dispatch index), an SGPR constant word, a 32-bit literal and a
# negative inline constant. A gfx10 VOP3 takes one literal; llvm-mc -mcpu=gfx1013 encodes both lines.
LINES = base.variant(
    base.READ_COMPUTE,
    ("v_min_u32_e32 v1, s0, v1", f"v_xor3_b32 v2, v1, s0, v0\nv_xor3_b32 v1, 0x{LITERAL:08x}, v2, {INLINE}"),
)
BINDING = base.READ_BINDING
LOAD = base.READ_LOADS["read_copy"]
PRODUCT = f"((((0x0000ffffu & {LOAD}) & 0xffffffu) * (({LOAD} >> (0x00000010u & 31u)) & 0xffffffu)) + tid.x)"
STORE = f"        outputData.Store(tid.x * 4u + 0u, (0x{LITERAL:08x}u ^ ({PRODUCT} ^ params[0].y ^ tid.x) ^ 0xfffffff0u));"
# Lane 0, worked by hand: v = 0x8b02a7f1, p = 0xa7f1 * 0x8b02 = 0x5b312ae2, and
# 0x5b312ae2 ^ 0xa5c30f96 ^ 0 ^ 0x2c5a96e1 ^ 0xfffffff0 = 0x2d574c65.
LANE0 = 0x2d574c65


def xor3_word(gid, *, op="xor", drop=(), inline=INLINE):
    """Independent CPU reference for one lane, from the instruction definition D = S0 ^ S1 ^ S2.

    op replaces both three-input operations, drop removes named sources, inline replaces the inline constant.
    """
    word = base.source_word(gid)
    product = ((word & 0xffff) * (word >> 16) + gid) & MASK

    def three(a, b, c):
        if op == "or":
            return a | b | c
        if op == "and":
            return a & b & c
        if op == "add":
            return (a + b + c) & MASK
        return a ^ b ^ c

    # A dropped source is the identity of the operation: 0 for xor, or and add, all ones for and.
    zero = MASK if op == "and" else 0
    first = three(zero if "product" in drop else product, zero if "key" in drop else KEY,
                  zero if "gid" in drop else gid)
    return three(zero if "literal" in drop else LITERAL, first, zero if "inline" in drop else inline & MASK)


def xor3_words(groups, bound, **reference):
    words = [base.SENTINEL] * 320
    for gid in range(min(32 * groups, bound)):
        words[gid] = xor3_word(gid, **reference)
    return words


WRONG = (("or for xor", dict(op="or")), ("and for xor", dict(op="and")), ("add for xor", dict(op="add")),
         ("product dropped", dict(drop=("product",))), ("key dropped", dict(drop=("key",))),
         ("dispatch index dropped", dict(drop=("gid",))), ("literal dropped", dict(drop=("literal",))),
         ("inline dropped", dict(drop=("inline",))), ("inline as +16", dict(inline=16)))
# Translations that a wrong template or operand would produce; each must fail on the GPU.
MUTATIONS = (("or for the first xor", f"0x{LITERAL:08x}u ^ (", f"0x{LITERAL:08x}u | ("),
             ("or inside the inner xor3", " ^ params[0].y ^ ", " | params[0].y | "),
             ("key from params[0].z", " ^ params[0].y ^ ", " ^ params[0].z ^ "),
             ("dispatch index dropped", " ^ tid.x) ^ ", " ^ 0u) ^ "),
             ("literal one bit off", f"0x{LITERAL:08x}u ^", f"0x{LITERAL ^ 1:08x}u ^"),
             ("inline dropped", " ^ 0xfffffff0u)", " ^ 0u)"))


def lower(*changes, **binding):
    return aot.lower(base.variant(LINES, *changes), 32, **(BINDING | binding)).splitlines()


def controls():
    """Exact output, then one rejection per opcode, modifier or operand outside the form."""
    assert xor3_word(0) == LANE0, hex(xor3_word(0))
    lines = lower()
    assert lines[-4:] == [base.GUARD_IF, STORE, "    }", "}"], lines
    # A copy of a result is the same program, and the sources print in operand order.
    first = "v_xor3_b32 v2, v1, s0, v0"
    assert lower((first, first + "\nv_mov_b32_e32 v3, v2"), (f"0x{LITERAL:08x}, v2,", f"0x{LITERAL:08x}, v3,")) == lines
    assert lower((first, "v_xor3_b32 v2, v0, v1, s0"))[-3] == STORE.replace(
        f"({PRODUCT} ^ params[0].y ^ tid.x)", f"(tid.x ^ {PRODUCT} ^ params[0].y)")
    # Positive inline constants and an SGPR in any slot are known words too.
    assert lower((f"v_xor3_b32 v1, 0x{LITERAL:08x}, v2, {INLINE}", "v_xor3_b32 v1, 64, s0, v2"))[-3] == (
        f"        outputData.Store(tid.x * 4u + 0u, (0x00000040u ^ params[0].y ^ ({PRODUCT} ^ params[0].y ^ tid.x)));")

    unsupported, operand = "Unsupported compute instruction: ", "Unverified ALU operand"
    second = f"v_xor3_b32 v1, 0x{LITERAL:08x}, v2, {INLINE}"
    for message, changes in (
        # Other three-input opcodes and an explicit encoding suffix.
        (unsupported + "v_or3_b32", ((first, "v_or3_b32 v2, v1, s0, v0"),)),
        (unsupported + "v_xad_u32", ((first, "v_xad_u32 v2, v1, s0, v0"),)),
        (unsupported + "v_xor3_b32_e64", ((first, "v_xor3_b32_e64 v2, v1, s0, v0"),)),
        # Source modifiers and an output modifier, which LLVM does not accept for this opcode.
        (unsupported + "v_xor3_b32", ((first, "v_xor3_b32 v2, -v1, s0, v0"),)),
        (unsupported + "v_xor3_b32", ((first, "v_xor3_b32 v2, |v1|, s0, v0"),)),
        (unsupported + "v_xor3_b32", ((first, "v_xor3_b32 v2, v1, neg(s0), v0"),)),
        (unsupported + "v_xor3_b32", ((first, first + " clamp"),)),
        # A register range, and a fourth source.
        (unsupported + "v_xor3_b32", ((first, "v_xor3_b32 v[2:3], v1, s0, v0"),)),
        (unsupported + "v_xor3_b32", ((first, first + ", v1"),)),
        # Sources that are not known words: an unknown, descriptor, exec-mask or pending SGPR, an unknown VGPR and a
        # literal wider than 32 bits. The three-source group matches only known words (a v_mad_u32_u24 with a byte
        # offset falls through to the row pitch), so these stop at the instruction.
        (unsupported + "v_xor3_b32", ((first, "v_xor3_b32 v2, v1, s20, v0"),)),
        (unsupported + "v_xor3_b32", ((first, "v_xor3_b32 v2, v1, s8, v0"),)),
        (unsupported + "v_xor3_b32", ((first, "v_xor3_b32 v2, v1, s2, v0"),)),
        (unsupported + "v_xor3_b32", (("s_buffer_load_dword s0, s[4:7], 0x4", ""),
                                      ("v_and_b32_e32", "s_buffer_load_dword s0, s[4:7], 0x4\nv_and_b32_e32"))),
        (unsupported + "v_xor3_b32", ((first, "v_xor3_b32 v2, v1, s0, v5"),)),
        (unsupported + "v_xor3_b32", ((second, "v_xor3_b32 v1, 0x1ffffffff, v2, -16"),)),
        # A destination past v255.
        (operand, ((first, "v_xor3_b32 v256, v1, s0, v0"),)),
        # A result is data only: not a store or read address and not a lane-guard compare operand.
        ("Unsupported store address", (("v1, v0, s[8:11]", "v0, v1, s[8:11]"),)),
        ("Unsupported read address", ((second, second + "\nbuffer_load_dword v3, v1, s[12:15], 0 offen"),)),
        ("Unverified lane guard compare", ((second, second + "\nv_cmp_gt_u32_e32 vcc_lo, s3, v1"),)),
        # Expressions are inline, so a result that keeps tripling is rejected.
        ("ALU expression too large", ((second, second + "\nv_xor3_b32 v1, v1, v1, v1" * 3),)),
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
        write(groups, bound, xor3_words(groups, bound))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    groups, bound = base.READ_RUNS[-1]
    expected = xor3_words(groups, bound)
    assert expected[0] == LANE0
    # Each wrong reference, one word off and the first lane past the bound must fail at the first differing word.
    off = list(expected)
    off[bound - 1] ^= 1
    for label, reference in (("one word off", off), ("lane past the bound", xor3_words(groups, bound + 1)),
                             *((label, xor3_words(groups, bound, **wrong)) for label, wrong in WRONG)):
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
        sys.exit("usage: compute_alu_xor3_check.py controls | draw <dxc> <renderer> <adapter>")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
