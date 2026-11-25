"""Check v_mad_i32_i24 between a compute read and its store (#1546).

v_mad_i32_i24 vD, a, b, c is sext(a[23:0]) * sext(b[23:0]) + c, modulo 2^32: each multiplicand is the signed
24-bit integer in its low 24 bits, and c is a full 32-bit word. The fixture is the read-compute-write program of
tests/compute_store_check.py with its ALU lines replaced by three v_mad_i32_i24, in LLVM 15 gfx1013 spelling.

    controls                              translator only
    draw <dxc> <renderer> <adapter>       translate, dispatch and compare the UAV readback
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base  # noqa: E402

aot = base.aot

# params[0].y: garbage 0x5a above bit 23, and bit 23 set, so the 24-bit multiplicand is negative (-0x789abd).
MAD_I24_CAP = 0x5a876543
# The source word w of a lane keeps its top byte (0x8b to 0x8d), which is garbage to a 24-bit multiply; bit 23
# is set in 63 of the first 150 words. The chain, with gid the dispatch index and cap the word above:
#   a = sext24(w) * sext24(cap) + gid        VGPR, SGPR and VGPR sources
#   b = sext24(0x12fffffd) * sext24(w) + a   a literal with garbage above bit 23: the multiplicand is -3
#   c = sext24(a) * -16 + b                  a computed multiplicand with garbage, a negative inline constant
# The products are wider than 32 bits. Each result is the addend of the next, so a wrong extension of any
# multiplicand reaches bits [31:24] of c instead of being truncated away by a later multiply.
MAD_I24 = ("v_mad_i32_i24 v2, v1, s0, v0", "v_mad_i32_i24 v3, 0x12fffffd, v1, v2", "v_mad_i32_i24 v1, v2, -16, v3")
READ_MAD_I24 = base.variant(
    base.READ_COMPUTE,
    ("v_and_b32_e32 v2, 0xffff, v1", MAD_I24[0]),
    ("v_lshrrev_b32_e32 v1, 16, v1", MAD_I24[1]),
    ("v_mad_u32_u24 v1, v2, v1, v0", MAD_I24[2]),
    ("v_min_u32_e32 v1, s0, v1", ""),
)
CASES = [("read_mad_i24", READ_MAD_I24, 32, base.READ_BINDING)]
LOAD = base.READ_LOADS["read_copy"]


def sext24(word):
    """The signed integer in bits [23:0] of a word."""
    low = word & 0xffffff
    return low - (1 << 24) if low >= 1 << 23 else low


def zext24(word):
    return word & 0xffffff


def mad(a, b, c, *, extend=sext24, addend=lambda word: word):
    return (extend(a) * extend(b) + addend(c)) % (1 << 32)


def lane_word(gid, **reference):
    """Independent CPU reference of the chain for one lane, on Python integers."""
    word = base.source_word(gid)
    a = mad(word, MAD_I24_CAP, gid, **reference)
    b = mad(0x12fffffd, word, a, **reference)
    return mad(a, 0xfffffff0, b, **reference)


def lane_words(groups, bound, **reference):
    words = [base.SENTINEL] * 320
    for gid in range(min(32 * groups, bound)):
        words[gid] = lane_word(gid, **reference)
    return words


# Three lanes worked out by hand, with sext24(cap) = 0x876543 - 2^24 = -7903933.
# Lane 0: w = 0x8b02a7f1, sext24(w) = 0x02a7f1 = 174065; a = 0xac0b0913, b = a - 3 * 174065 = 0xac031140,
# sext24(a) = 0x0b0913 = 723219, c = b - 16 * 723219 = b - 0xb09130.
# Lane 3: w = 0x8b0ee304, sext24(w) = 975620; a = 0x9691fe0f, b = a - 0x2ca90c = 0x96655503,
# sext24(a) = 0x91fe0f - 2^24 = -7209457, c = b + 0x6e01f10.
# Lane 31: w = 0x8b810a60 has bit 23 set, sext24(w) = 0x810a60 - 2^24 = -8320416; a = 0xe07d973f,
# b = a + 3 * 8320416 = 0xe1fa781f, sext24(a) = 0x7d973f, c = b - 0x7d973f0.
HAND = {0: 0xab528010, 3: 0x9d457413, 31: 0xda21042f}


def sign_extend(word):
    return f"asuint(asint({word} << 8u) >> 8u)"


def expected_store():
    a = f"(({sign_extend(LOAD)} * {sign_extend('params[0].y')}) + tid.x)"
    b = f"(({sign_extend('0x12fffffdu')} * {sign_extend(LOAD)}) + {a})"
    c = f"(({sign_extend(a)} * {sign_extend('0xfffffff0u')}) + {b})"
    return f"        outputData.Store(tid.x * 4u + 0u, {c});"


def controls():
    """Exact output, then one rejection per operand form or modifier outside the lowering."""
    # The reference against the hand values, and the sign and garbage bits the fixture must exercise.
    assert {gid: lane_word(gid) for gid in HAND} == HAND
    assert base.source_word(0) & 0x800000 == 0 and base.source_word(31) & 0x800000
    assert all(base.source_word(gid) >> 24 for gid in range(150))
    assert 0 < sum(bool(base.source_word(gid) & 0x800000) for gid in range(150)) < 150
    assert 0 < sum(bool(mad(base.source_word(gid), MAD_I24_CAP, gid) & 0x800000) for gid in range(150)) < 150

    def lower(*changes, **binding):
        _, lines, width, bound = CASES[0]
        return aot.lower(base.variant(lines, *changes), width, **(bound | binding)).splitlines()

    lines = lower()
    assert lines[:3] == ["cbuffer Constants : register(b0) { uint4 params[4]; };",
                         "ByteAddressBuffer inputData : register(t0);",
                         "RWByteAddressBuffer outputData : register(u0);"], lines
    assert lines[-4:] == [base.GUARD_IF, expected_store(), "    }", "}"], lines
    # A copy of a result is the same program.
    assert lower((MAD_I24[2], MAD_I24[2].replace("v1,", "v4,", 1) + "\nv_mov_b32_e32 v1, v4")) == lines
    # The unsigned v_mad_u32_u24 is unchanged: the data form (its result a feeds both later steps), and the
    # row pitch of a byte offset.
    assert lower((MAD_I24[0], MAD_I24[0].replace("i32_i24", "u32_u24")))[-3].count(
        f"((({LOAD} & 0xffffffu) * (params[0].y & 0xffffffu)) + tid.x)") == 2
    offset = "v_lshlrev_b32_e32 v4, 2, v0\n"
    base.rejects("Unverified row pitch byte offset", lower, (MAD_I24[0], offset + "v_mad_u32_u24 v2, v1, v1, v4"))

    unsupported = "Unsupported compute instruction: v_mad_i32_i24"
    for message, changes in (
        # Modifiers: clamp saturates instead of wrapping, and LLVM refuses neg and abs on this integer
        # opcode, so the text forms are refused too. The VOP3 spelling with a suffix is not the one LLVM prints.
        (unsupported, ((MAD_I24[2], MAD_I24[2] + " clamp"),)),
        (unsupported, ((MAD_I24[0], "v_mad_i32_i24 v2, -v1, s0, v0"),)),
        (unsupported, ((MAD_I24[0], "v_mad_i32_i24 v2, |v1|, s0, v0"),)),
        (unsupported, ((MAD_I24[0], "v_mad_i32_i24 v2, v1, -|s0|, v0"),)),
        (unsupported + "_e64", ((MAD_I24[0], "v_mad_i32_i24_e64 v2, v1, s0, v0"),)),
        # Sources that are not known words fall through every branch: an unknown VGPR or SGPR, a pending
        # SGPR, a literal wider than 32 bits, a register range, and a byte offset, which is not a row pitch
        # for the signed form.
        (unsupported, ((MAD_I24[0], "v_mad_i32_i24 v2, v1, s0, v9"),)),
        (unsupported, ((MAD_I24[0], "v_mad_i32_i24 v2, v1, s20, v0"),)),
        (unsupported, (("s_buffer_load_dword s0, s[4:7], 0x4", ""),
                       ("v_mad_i32_i24 v2", "s_buffer_load_dword s0, s[4:7], 0x4\nv_mad_i32_i24 v2"))),
        (unsupported, ((MAD_I24[1], "v_mad_i32_i24 v3, 0x1ffffffff, v1, v2"),)),
        (unsupported, ((MAD_I24[0], "v_mad_i32_i24 v2, v[1:2], s0, v0"),)),
        (unsupported, ((MAD_I24[0], offset + "v_mad_i32_i24 v2, v1, v1, v4"),)),
        # A destination past v255.
        ("Unverified ALU operand", ((MAD_I24[2], "v_mad_i32_i24 v256, v2, -16, v3"),)),
        # Neighbouring signed opcodes stay outside the subset.
        ("Unsupported compute instruction: v_mul_i32_i24_e32", ((MAD_I24[1], "v_mul_i32_i24_e32 v3, v1, v2"),)),
        ("Unsupported compute instruction: v_mad_i64_i32",
         ((MAD_I24[1], "v_mad_i64_i32 v[3:4], s4, v1, v2, v[0:1]"),)),
        # The word read before its wait, a result as an address or a guard operand, and an expression
        # that keeps tripling.
        ("Vector load result used or overwritten before s_waitcnt vmcnt(0)", (("s_waitcnt vmcnt(0)", ""),)),
        ("Unsupported store address", (("v1, v0, s[8:11]", "v0, v1, s[8:11]"),)),
        ("Unverified lane guard compare", ((MAD_I24[2], MAD_I24[2] + "\nv_cmp_gt_u32_e32 vcc_lo, s3, v1"),)),
        ("ALU expression too large", ((MAD_I24[2], MAD_I24[2] + "\nv_mad_i32_i24 v1, v1, v1, v1" * 2),)),
    ):
        base.rejects(message, lower, *changes)


def run_rejected(cmd, label, word):
    bad = subprocess.run(cmd, capture_output=True, text=True)
    if bad.returncode != 1 or word not in bad.stderr:
        raise AssertionError(f"read_mad_i24: {label} escaped detection\n{bad.stderr}")
    print(f"read_mad_i24: {label} rejected")


def draw(directory, compiler, renderer, backend):
    """Dispatch the fixture and compare all 320 UAV words with lane_words, then the reference and HLSL controls."""
    _, lines, width, binding = CASES[0]
    src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
    src.write_text(aot.lower(lines, width, **binding), encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    cmd = [renderer, backend, "--compute-store-case", cso, case]
    constants = [0x5eed0000 + i for i in range(16)]
    constants[1] = MAD_I24_CAP

    def write(groups, bound, words):
        case.write_bytes(struct.pack("<337I", groups, *words, bound, *constants[1:]))

    for groups, bound in base.READ_RUNS:
        write(groups, bound, lane_words(groups, bound))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    groups, bound = base.READ_RUNS[-1]
    expected = lane_words(groups, bound)
    off = list(expected)
    off[bound - 1] ^= 1
    # A CPU reference that is wrong in one word, zero-extends, multiplies whole words, truncates the addend
    # to 24 bits, or runs one lane past the bound must fail at its first differing word.
    for label, reference in (("one word off", off),
                             ("zero-extended multiplicands", lane_words(groups, bound, extend=zext24)),
                             ("untruncated multiplicands", lane_words(groups, bound, extend=lambda word: word)),
                             ("24-bit addend", lane_words(groups, bound, addend=zext24)),
                             ("lane past the bound", lane_words(groups, bound + 1))):
        word = next(i for i, (a, b) in enumerate(zip(expected, reference)) if a != b)
        write(groups, bound, reference)
        run_rejected(cmd, f"reference '{label}'", f"word={word} expected=")
    write(groups, bound, expected)
    source = src.read_text(encoding="ascii")
    # Translations that a wrong extension of one operand would produce.
    for label, old, new in (
            ("cap zero-extended", sign_extend("params[0].y"), "(params[0].y & 0xffffffu)"),
            ("literal untruncated", sign_extend("0x12fffffdu"), "0x12fffffdu"),
            ("inline constant zero-extended", sign_extend("0xfffffff0u"), "(0xfffffff0u & 0xffffffu)"),
            ("source word shifted logically", f"asint({LOAD} << 8u) >> 8u", f"({LOAD} << 8u) >> 8u")):
        mutated = source.replace(old, new)
        if mutated == source:
            raise AssertionError(f"read_mad_i24: mutation '{label}' did not change the source")
        mutant, mutant_cso = directory / "mutated.hlsl", directory / "mutated.cso"
        mutant.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", mutant_cso, mutant], check=True)
        run_rejected([*cmd[:3], mutant_cso, case], f"mutation '{label}'", "Compute readback differs")
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        opcodes = sorted({line.split()[0] for line in lines if line.strip()})
        path.write_text("".join(f"{op}\n" for op in opcodes), encoding="utf-8")
    return 0


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    controls()
    if mode == "controls" and not args:
        sys.exit(0)
    if mode != "draw" or len(args) != 3:
        sys.exit(__doc__)
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
