"""Check v_cvt_pk_u8_f32 between a guarded compute read and its store (#1546).

The RDNA 1.0 and RDNA 2 ISA guides give D = (S2 & ~(0xff << 8 * S1[1:0])) | ((flt32_to_uint8(S0) & 0xff) << 8 * S1[1:0])
and do not define flt32_to_uint8. Every binary32 S0 that reaches the conversion here is +0.0 or an integer in [1, 255],
where truncation, any rounding mode and any clamp give the same byte. Fractional, negative, out-of-range and NaN inputs
are not exercised, and the translator refuses them as constants.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import compute_store_check as base

aot = base.aot
BINDING = base.READ_BINDING
# The cap word params[0].y is the S2 of the first insert. The other constant words differ from it.
CAP = 0xa5c3e17b
# The guarded read of READ_COMPUTE gives the source word w in v1. Two bytes of w become exact binary32 integers through
# 2^23: the bits 0x4b000000 | k are the float 2^23 + k, and subtracting 2^23 leaves k exactly (k = 0 gives +0.0).
#   x = float(w & 0xff); y = float((w >> 16) & 0xff); sel = w >> 8   (sel keeps bits above [1:0])
#   a = cvt_pk(x, sel, cap); a = cvt_pk(y, 3, a)
#   b = cvt_pk(255.0, 2, w); b = cvt_pk(1.0, 0, b); b = cvt_pk(0, 1, b)
#   out = a ^ b
# The sources cover VGPR, SGPR, literal and inline operands in each position that LLVM 15 prints for VOP3.
PROGRAM = base.variant(
    base.READ_COMPUTE,
    ("v_and_b32_e32 v2, 0xffff, v1", "v_and_b32_e32 v2, 0xff, v1\nv_or_b32_e32 v2, 0x4b000000, v2\n"
                                     "v_subrev_f32_e32 v2, 0x4b000000, v2"),
    ("v_lshrrev_b32_e32 v1, 16, v1", "v_lshrrev_b32_e32 v3, 16, v1\nv_and_b32_e32 v3, 0xff, v3\n"
                                     "v_or_b32_e32 v3, 0x4b000000, v3\nv_subrev_f32_e32 v3, 0x4b000000, v3\n"
                                     "v_lshrrev_b32_e32 v4, 8, v1"),
    ("v_mad_u32_u24 v1, v2, v1, v0", "v_cvt_pk_u8_f32 v5, v2, v4, s0\nv_cvt_pk_u8_f32 v5, v3, 3, v5"),
    ("v_min_u32_e32 v1, s0, v1", "v_cvt_pk_u8_f32 v6, 0x437f0000, 2, v1\nv_cvt_pk_u8_f32 v6, 1.0, 0, v6\n"
                                 "v_cvt_pk_u8_f32 v6, 0, 1, v6\nv_xor_b32_e32 v1, v6, v5"),
)
CASES = [("read_cvt_pk_u8", PROGRAM, 32, BINDING)]
# Worked by hand from source_word(gid) = 0x8b02a7f1 + gid * 0x413b1 and the pseudocode above.
#   gid 0: w = 0x8b02a7f1, x = 0xf1, y = 0x02, sel[1:0] = 3. a = 0x02c3e17b (y replaces x in byte 3).
#          b = 0x8bff0001. out = 0x893ce17a.
#   gid 4: w = 0x8b12f6b5, x = 0xb5, y = 0x12, sel[1:0] = 2. a = 0x12b5e17b. b = 0x8bff0001. out = 0x994ae17a.
#   gid 7: w = 0x8b1f31c8, x = 0xc8, y = 0x1f, sel[1:0] = 1. a = 0x1fc3c87b. b = 0x8bff0001. out = 0x943cc87a.
#   gid 10: w = 0x8b2b6cdb, x = 0xdb, y = 0x2b, sel[1:0] = 0. a = 0x2bc3e1db. b = 0x8bff0001. out = 0xa03ce1da.
HAND = {0: 0x893ce17a, 4: 0x994ae17a, 7: 0x943cc87a, 10: 0xa03ce1da}


def exact_byte(bits):
    """S0 as the conversion receives it, which must be +0.0 or an integer in [1, 255]."""
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    assert bits == 0 or (bits >> 31 == 0 and value.is_integer() and 1 <= value <= 255), hex(bits)
    return int(value)


def through_two23(k):
    """float(k) as the fixture forms it: the bits of 2^23 + k minus 2^23, both exact in binary64."""
    return struct.unpack("<I", struct.pack("<f", struct.unpack("<f", struct.pack("<I", 0x4b000000 | k))[0] - 2.0 ** 23))[0]


def insert(s0, s1, s2, *, wrong=()):
    """The ISA pseudocode on words; each name in wrong replaces one part with a plausible mistake."""
    u8 = s0 & 0xff if "bits" in wrong else exact_byte(s0)
    sel = s1 & 2 if "select" in wrong else s1 & 3
    sel = 3 - sel if "order" in wrong else sel
    base_word = s2 if "clear" in wrong else s2 & ~(0xff << 8 * sel) & 0xffffffff
    return base_word | (u8 << 8 * sel)


def lane(gid, **reference):
    w = base.source_word(gid)
    x, y = through_two23(w & 0xff), through_two23((w >> 16) & 0xff)
    a = insert(y, 3, insert(x, w >> 8, CAP, **reference), **reference)
    b = insert(0, 1, insert(0x3f800000, 0, insert(0x437f0000, 2, w, **reference), **reference), **reference)
    return a ^ b


def words(groups, bound, **reference):
    out = [base.SENTINEL] * 320
    for gid in range(min(32 * groups, bound)):
        out[gid] = lane(gid, **reference)
    return out


def lower(*changes):
    return aot.lower(base.variant(PROGRAM, *changes), 32, **BINDING).splitlines()


def controls():
    """Exact output, accepted operand spellings, then one rejection per operand or modifier outside the subset."""
    assert all(lane(gid) == word for gid, word in HAND.items()), [hex(lane(gid)) for gid in HAND]
    lines = lower()
    # The float subtraction that forms S0 brings its one helper; the conversion itself needs none.
    assert lines[:4] == ["cbuffer Constants : register(b0) { uint4 params[4]; };",
                         "ByteAddressBuffer inputData : register(t0);", base.aot.F32_HELPERS["sub"].rstrip("\n"),
                         "RWByteAddressBuffer outputData : register(u0);"], lines
    load = base.READ_LOADS["read_copy"]

    def pk(s0, s1, s2):
        return f"(({s2} & ~(0xffu << (({s1} & 3u) << 3u))) | ((uint(asfloat({s0})) & 0xffu) << (({s1} & 3u) << 3u)))"

    x = f"f32_sub((0x4b000000u | (0x000000ffu & {load})), 0x4b000000u)"
    y = f"f32_sub((0x4b000000u | (0x000000ffu & ({load} >> (0x00000010u & 31u)))), 0x4b000000u)"
    a = pk(y, "0x00000003u", pk(x, f"({load} >> (0x00000008u & 31u))", "params[0].y"))
    b = pk("0x00000000u", "0x00000001u", pk("0x3f800000u", "0x00000000u", pk("0x437f0000u", "0x00000002u", load)))
    assert lines[-4:] == [base.GUARD_IF, f"        outputData.Store(tid.x * 4u + 0u, ({b} ^ {a}));", "    }", "}"], lines
    # A copy of a result is the same program. Every constant S0 that is +0.0 or an integer in [1, 255] is accepted.
    assert lower(("v_cvt_pk_u8_f32 v6, 0, 1, v6", "v_cvt_pk_u8_f32 v7, 0, 1, v6\nv_mov_b32_e32 v6, v7")) == lines
    first = "v_cvt_pk_u8_f32 v6, 0x437f0000, 2, v1"
    for constant, bits in (("2.0", 0x40000000), ("4.0", 0x40800000), ("0x42f60000", 0x42f60000),
                           ("0x3f800000", 0x3f800000), ("0", 0)):
        assert f"(uint(asfloat(0x{bits:08x}u))" in lower((first, first.replace("0x437f0000", constant)))[-3]
    # An SGPR S0 and S1, a literal S2 and a constant word read from memory are known words too.
    assert "(params[0].y & 3u)" in lower(("v_cvt_pk_u8_f32 v5, v3, 3, v5", "v_cvt_pk_u8_f32 v5, v3, s0, v5"))[-3]
    assert "(0x12345678u & ~(0xffu <<" in lower((first, "v_cvt_pk_u8_f32 v6, v2, 2, 0x12345678"))[-3]
    unsupported, operand = "Unsupported compute instruction: ", "Unverified ALU operand"
    second = "v_cvt_pk_u8_f32 v5, v3, 3, v5"
    for message, changes in (
        # Input and output modifiers, the VOP3 suffix LLVM 15 does not print, and a register range.
        (unsupported + "v_cvt_pk_u8_f32", ((second, "v_cvt_pk_u8_f32 v5, -v3, 3, v5"),)),
        (unsupported + "v_cvt_pk_u8_f32", ((second, "v_cvt_pk_u8_f32 v5, |v3|, 3, v5"),)),
        (unsupported + "v_cvt_pk_u8_f32", ((second, "v_cvt_pk_u8_f32 v5, -|v3|, 3, v5"),)),
        (unsupported + "v_cvt_pk_u8_f32", ((second, second + " clamp"),)),
        (unsupported + "v_cvt_pk_u8_f32", ((second, second + " mul:2"),)),
        (unsupported + "v_cvt_pk_u8_f32_e64", ((second, "v_cvt_pk_u8_f32_e64 v5, v3, 3, v5"),)),
        (unsupported + "v_cvt_pk_u8_f32", ((second, "v_cvt_pk_u8_f32 v5, v3, 3, v[5:6]"),)),
        # Other float-to-integer packs whose conversion the guides leave open stay refused.
        (unsupported + "v_cvt_pknorm_u16_f32", ((second, "v_cvt_pknorm_u16_f32 v5, v3, v5"),)),
        # A constant S0 where readings of flt32_to_uint8 can differ: fractional, negative, -0.0, above 255, infinite,
        # NaN, denormal (an inline integer is a bit pattern), and a word wider than 32 bits.
        *((operand, ((first, first.replace("0x437f0000", constant)),))
          for constant in ("0.5", "0.15915494", "0x3fc00000", "-1.0", "-4.0", "0x80000000", "0x43800000", "0x7f800000",
                           "0x7fc00000", "-1", "5", "0x00000001", "0x1ffffffff")),
        # Sources that are not known words in each position, and a destination past v255.
        (operand, ((second, "v_cvt_pk_u8_f32 v5, v7, 3, v5"),)),
        (operand, ((second, "v_cvt_pk_u8_f32 v5, v3, s20, v5"),)),
        (operand, ((second, "v_cvt_pk_u8_f32 v5, v3, 3, s8"),)),
        (operand, ((second, "v_cvt_pk_u8_f32 v5, v3, 3, s2"),)),
        (operand, ((second, "v_cvt_pk_u8_f32 v256, v3, 3, v5"),)),
        # The cap word S2 read before its s_waitcnt.
        (operand, (("s_buffer_load_dword s0, s[4:7], 0x4", ""),
                   ("v_and_b32_e32 v2, 0xff, v1", "s_buffer_load_dword s0, s[4:7], 0x4\nv_and_b32_e32 v2, 0xff, v1"))),
        ("Vector load result used or overwritten before s_waitcnt vmcnt(0)", (("s_waitcnt vmcnt(0)", ""),)),
        # A result is data only: not an address and not a lane-guard operand, and a long chain is refused.
        ("Unsupported store address", (("v1, v0, s[8:11]", "v0, v1, s[8:11]"),)),
        ("Unverified lane guard compare", (("v_xor_b32_e32 v1, v6, v5",
                                            "v_xor_b32_e32 v1, v6, v5\nv_cmp_gt_u32_e32 vcc_lo, s3, v5"),)),
        ("ALU expression too large", ((second, second + "\nv_cvt_pk_u8_f32 v5, v3, v5, v5" * 2),)),
    ):
        base.rejects(message, lower, *changes)


def draw(directory, compiler, renderer, backend):
    """Dispatch the fixture, compare the UAV with words(), then require wrong references and translations to fail."""
    src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
    source = aot.lower(PROGRAM, 32, **BINDING)
    src.write_text(source, encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    cmd = [renderer, backend, "--compute-store-case", cso, case]
    constants = [0x5eed0000 + i for i in range(16)]
    constants[1] = CAP
    groups, bound = base.READ_RUNS[-1]

    def write(run_groups, run_bound, expected):
        case.write_bytes(struct.pack("<337I", run_groups, *expected, run_bound, *constants[1:]))

    # Every lane exercises each byte select, and both ends of the byte range reach the conversion from memory.
    w = [base.source_word(gid) for gid in range(bound)]
    assert {(word >> 8) & 3 for word in w} == {0, 1, 2, 3} and any(word >> 10 for word in w)
    assert {0, 255} <= {(word >> 16) & 0xff for word in w} and 255 in {word & 0xff for word in w}
    for run_groups, run_bound in base.READ_RUNS:
        write(run_groups, run_bound, words(run_groups, run_bound))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    expected = words(groups, bound)
    # Each reference with one plausible mistake must fail the run at the first word where it differs.
    for wrong in (("order",), ("clear",), ("bits",), ("select",)):
        reference = words(groups, bound, wrong=wrong)
        word = next(i for i, (a, b) in enumerate(zip(expected, reference)) if a != b)
        write(groups, bound, reference)
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected={reference[word]:08x}" not in bad.stderr:
            raise AssertionError(f"reference {wrong} escaped detection\n{bad.stderr}")
    # A lane past the bound that writes, and a word past every lane that changes, both fail at their own word.
    for word in (bound, 319):
        stale = expected.copy()
        stale[word] ^= 1
        write(groups, bound, stale)
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected={stale[word]:08x} actual={base.SENTINEL:08x}" not in bad.stderr:
            raise AssertionError(f"sentinel word {word} escaped detection\n{bad.stderr}")
    write(groups, bound, expected)
    for label, old, new in (("byte select bit 0 dropped", "& 3u) << 3u)", "& 2u) << 3u)"),
                            ("S2 byte not cleared", " & ~(0xffu << ", " & ~(0x00u << "),
                            ("float bits for its value", "(uint(asfloat(", "(asuint(asfloat("),
                            ("byte mask 0x7f", "& 0xffu) << ", "& 0x7fu) << "),
                            ("literal 254.0 for 255.0", "0x437f0000u", "0x437e0000u"),
                            ("cap from the next constant", "params[0].y", "params[0].z"),
                            ("lane at the bound writes", "if (params[0].x > tid.x)", "if (params[0].x >= tid.x)")):
        mutated = source.replace(old, new)
        if mutated == source:
            raise AssertionError(f"mutation '{label}' did not change the source")
        src, cso = directory / "mutated.hlsl", directory / "mutated.cso"
        src.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
        bad = subprocess.run([*cmd[:3], cso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"mutation '{label}' escaped detection")
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        opcodes = sorted({line.split()[0] for line in PROGRAM if line.strip()})
        path.write_text("".join(f"{op}\n" for op in opcodes), encoding="utf-8")
    return 0


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    controls()
    if mode == "controls" and not args:
        sys.exit(0)
    if mode != "draw" or len(args) != 3:
        sys.exit("usage: compute_alu_cvt_pk_u8_check.py controls | draw <dxc> <renderer> <adapter>")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
