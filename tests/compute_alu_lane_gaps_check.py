"""Check the scalar bit count of a general scalar, v_mbcnt over exec_lo and wave32 vcc_hi in the compute store profile.

controls: exact translations and one rejection per form outside the slice.
draw <dxc> <renderer> <adapter>: dispatch both programs and compare with an independent CPU reference.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base  # noqa: E402

aot = base.aot
MASK32 = 0xffffffff
NAME = "lane_gaps"
KEY = 0xa5c30f96     # params[0].y, a run-time scalar word
HI_MASK = 0x6d3a95c7  # written to vcc_hi by s_mov_b32
EXEC_WORD = 0x9d2c5a73

# Program A: guarded read-compute (the read_compute fixture) with the cap clamp replaced, wave32 statement.
#   p = (v & 0xffff) * (v >> 16) + gid;  lane = gid & 31
#   out[gid] = pop(pop(HI_MASK & below(lane)) + pop(KEY))  [s_bcnt1_i32_b32 of s0, a run-time word]
#            + pop(KEY & below(lane)) + p + pop(pop(HI_MASK))
# Each source is a tracked word: s0 is params[0].y, vcc_hi holds the s_mov_b32 literal, s21 its bit count.
# RDNA2 ISA: S_BCNT1_I32_B32 D = CountOneBits(S0); V_MBCNT_LO_U32_B32 D = CountOneBits(S0 & ((1 << lane) - 1)) + S1;
# V_MBCNT_HI_U32_B32 does the same over mask bits 63:32 (none in wave32). LLVM AMDGPU SOP1/VOP3 definitions,
# Mesa ACO bit_count / mbcnt.
PROGRAM_A = base.variant(
    base.READ_COMPUTE,
    ("v_min_u32_e32 v1, s0, v1",
     "s_bcnt1_i32_b32 s20, s0\n"
     f"s_mov_b32 vcc_hi, 0x{HI_MASK:08x}\n"
     "s_bcnt1_i32_b32 s21, vcc_hi\n"
     "v_mbcnt_lo_u32_b32 v2, vcc_hi, s20\n"
     "v_mbcnt_hi_u32_b32 v2, vcc_hi, v2\n"
     "v_mbcnt_lo_u32_b32 v4, s0, v1\n"
     "v_bcnt_u32_b32 v1, v2, v4\n"
     "v_bcnt_u32_b32 v1, s21, v1"),
)
BINDING_A = base.READ_BINDING | dict(wave_size=32)

# Program B: no guard, 64 threads (two wave32 waves), exec_lo as the v_mbcnt mask.
#   out[flat] = flat * 64 + (flat & 31) + pop(EXEC_WORD)
PROGRAM_B = [
    "s_mov_b32 s9, 0",
    "s_load_dwordx4 s[12:15], s[8:9], null",
    f"s_mov_b32 s5, 0x{EXEC_WORD:08x}",
    "s_bcnt1_i32_b32 s6, s5",
    "v_lshl_add_u32 v3, s10, 6, v0",
    "v_mbcnt_lo_u32_b32 v1, exec_lo, s6",
    "v_mbcnt_hi_u32_b32 v1, exec_lo, v1",
    "v_mad_u32_u24 v1, v3, 64, v1",
    "s_waitcnt lgkmcnt(0)",
    "buffer_store_dword v1, v3, s[12:15], 0 idxen",
    "s_endpgm",
]
BINDING_B = base.BCNT_EXEC_BINDING


def ones(word):
    return bin(word & MASK32).count("1")


def word_a(gid, *, hi_mask=HI_MASK, lane_bits=None, drop_hi=False, count=ones):
    """CPU reference for program A from the instruction definitions."""
    lane = gid & 31
    below = (1 << lane) - 1 if lane_bits is None else lane_bits(lane)
    source = base.source_word(gid)
    product = ((source & 0xffff) * (source >> 16) + gid) & MASK32
    v2 = count(hi_mask & below) + count(KEY)
    if drop_hi:
        v2 = 0
    v4 = count(KEY & below) + product
    return (count(v2) + v4 + count(count(hi_mask))) & MASK32


def words_a(groups, bound, **reference):
    words = [base.SENTINEL] * 320
    for gid in range(min(32 * groups, bound)):
        words[gid] = word_a(gid, **reference)
    return words


def words_b(*, base_count=ones(EXEC_WORD), lane_mod=32):
    words = [base.SENTINEL] * 320
    for flat in range(64):
        words[flat] = flat * 64 + (flat % lane_mod) + base_count
    return words


def lower_a(lines=PROGRAM_A, **binding):
    return aot.lower(lines, 32, **(BINDING_A | binding))


def lower_b(lines=PROGRAM_B, threads=64, **binding):
    return aot.lower(lines, threads, **(BINDING_B | binding))


def controls():
    reference = base.source_word
    assert reference(0) == 0x8b02a7f1
    text = lower_a()
    # s_bcnt1_i32_b32 of a run-time word is countbits of that word; of vcc_hi it folds to the literal's count.
    assert "countbits(params[0].y)" in text, text
    assert "countbits(0x6d3a95c7u & ((1u << (gindex & 31u)) - 1u))" in text, text
    assert f"countbits(0x{ones(HI_MASK):08x}u)" in text, text
    text_b = lower_b()
    assert "countbits(0xffffffffu & ((1u << (gindex & 31u)) - 1u))" in text_b, text_b
    assert f"0x{ones(EXEC_WORD):08x}u" in text_b, text_b

    # s_bcnt1_i32_b32 of a literal and of an inline constant folds.
    assert f"0x{ones(0x12345678):08x}u" in lower_b(base.variant(PROGRAM_B, ("s5, 0x9d2c5a73", "s5, 0x12345678")))
    assert "0x00000020u" in lower_b(base.variant(PROGRAM_B, ("s6, s5", "s6, -1")))

    rejects = base.rejects
    # s_bcnt1_i32_b32 sources that are not a known word.
    for source in ("s30", "exec_hi", "vcc_lo", "exec", "s[4:5]", "s4:", "m0"):
        rejects("Unsupported active lane count source: " + source, lower_b,
                base.variant(PROGRAM_B, ("s6, s5", "s6, " + source)))
    rejects("Unsupported active lane count source: s8", lower_b, base.variant(PROGRAM_B, ("s6, s5", "s6, s8")))
    rejects("Invalid scalar destination or pending load", lower_b,
            base.variant(PROGRAM_B, ("s6, s5", "s106, s5")))
    # exec_lo as an s_bcnt1 source keeps its group and wave rules.
    rejects("Unverified active lane count wave size or group size", lower_b,
            base.variant(PROGRAM_B, ("s6, s5", "s6, exec_lo")), 48)
    # exec_lo as the v_mbcnt mask: not inside a guarded region, not hi-only forms of other halves.
    guarded = base.variant(PROGRAM_A, ("v_mbcnt_lo_u32_b32 v2, vcc_hi, s20", "v_mbcnt_lo_u32_b32 v2, exec_lo, s20"))
    rejects("Active lane mask inside guarded region", lower_a, guarded)
    for source in ("exec_hi", "vcc_lo", "exec"):
        rejects("Unsupported compute instruction", lower_b,
                base.variant(PROGRAM_B, ("v1, exec_lo, s6", f"v1, {source}, s6")))
    rejects("Unverified lane count operand or wave size", lower_b, PROGRAM_B, 64, wave_size=None)
    # vcc_hi needs the wave32 statement, is written only by s_mov_b32 and read only as a known word.
    rejects("Unsupported compute instruction: s_mov_b32 vcc_hi", lower_a, PROGRAM_A, wave_size=None)
    rejects("Unsupported active lane count source: s107", lower_a,
            base.variant(PROGRAM_A, ("s_mov_b32 vcc_hi, 0x6d3a95c7", "s_nop 0")))
    rejects("Unverified lane count operand or wave size", lower_a,
            base.variant(PROGRAM_A, ("s_mov_b32 vcc_hi, 0x6d3a95c7", "s_nop 0"), ("s_bcnt1_i32_b32 s21, vcc_hi", "s_nop 0")))
    rejects("Invalid scalar destination or pending load", lower_a,
            base.variant(PROGRAM_A, ("s_mov_b32 vcc_hi, 0x6d3a95c7", "s_mov_b32 s107, 0x6d3a95c7")))
    rejects("Invalid scalar destination or pending load", lower_a,
            base.variant(PROGRAM_A, ("s_bcnt1_i32_b32 s21, vcc_hi", "s_bcnt1_i32_b32 s21, s107")))
    rejects("Unverified scalar ALU operand", lower_a,
            base.variant(PROGRAM_A, ("s_bcnt1_i32_b32 s21, vcc_hi", "s_and_b32 vcc_hi, vcc_hi, 1")))
    # A reference with a wrong rule must differ from the right one on at least one lane.
    assert words_a(1, 17) != words_a(1, 17, drop_hi=True)


def draw(directory, compiler, renderer, backend):
    """Translate, compile and dispatch both programs, then require wrong references and mutated sources to fail."""
    cmd = lambda cso, case: [renderer, backend, "--compute-store-case", cso, case]  # noqa: E731
    checked = 0
    # Program B: 64 threads, 321 words.
    src, cso, case = [directory / x for x in ("b.hlsl", "b.cso", "b.case")]
    source_b = lower_b()
    src.write_text(source_b, encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    expected = words_b()
    case.write_bytes(struct.pack("<321I", 1, *expected))
    result = subprocess.run(cmd(cso, case))
    if result.returncode:
        return result.returncode
    for label, reference in (("rank not masked to the wave", words_b(lane_mod=64)),
                             ("count of the word dropped", words_b(base_count=0)),
                             ("count off by one", words_b(base_count=ones(EXEC_WORD) + 1))):
        word = next(i for i, (a, b) in enumerate(zip(expected, reference)) if a != b)
        case.write_bytes(struct.pack("<321I", 1, *reference))
        bad = subprocess.run(cmd(cso, case), capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected=" not in bad.stderr:
            raise AssertionError(f"{NAME}: B reference '{label}' escaped detection\n{bad.stderr}")
        checked += 1
    case.write_bytes(struct.pack("<321I", 1, *expected))
    checked += mutate(directory, compiler, source_b, case, cmd, cso, (
        ("exec mask as zero", "countbits(0xffffffffu &", "countbits(0x00000000u &"),
        ("rank over half the wave", "(gindex & 31u)", "(gindex & 15u)"),
        ("count dropped", f"0x{ones(EXEC_WORD):08x}u", "0x00000000u")))

    # Program A: groups of 32, read-compute binding with the params constants.
    src, cso, case = [directory / x for x in ("a.hlsl", "a.cso", "a.case")]
    source_a = lower_a()
    src.write_text(source_a, encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    constants = [0x5eed0000 + i for i in range(16)]
    constants[1] = KEY

    def write(groups, bound, words):
        case.write_bytes(struct.pack("<337I", groups, *words, bound, *constants[1:]))

    for groups, bound in base.READ_RUNS:
        write(groups, bound, words_a(groups, bound))
        result = subprocess.run(cmd(cso, case))
        if result.returncode:
            return result.returncode
    groups, bound = base.READ_RUNS[-1]
    expected = words_a(groups, bound)
    for label, wrong in (("mask bits above the lane", dict(lane_bits=lambda lane: (1 << (lane + 1)) - 1)),
                         ("mask ignores the lane", dict(lane_bits=lambda lane: MASK32)),
                         ("vcc_hi mask dropped", dict(drop_hi=True)),
                         ("zero bit count", dict(count=lambda w: 32 - ones(w)))):
        reference = words_a(groups, bound, **wrong)
        word = next(i for i, (a, b) in enumerate(zip(expected, reference)) if a != b)
        write(groups, bound, reference)
        bad = subprocess.run(cmd(cso, case), capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected=" not in bad.stderr:
            raise AssertionError(f"{NAME}: A reference '{label}' escaped detection\n{bad.stderr}")
        checked += 1
    write(groups, bound, expected)
    checked += mutate(directory, compiler, source_a, case, cmd, cso, (
        ("vcc_hi mask zero", "0x6d3a95c7u &", "0x00000000u &"),
        ("run-time word count dropped", "countbits(params[0].y)", "countbits(0u)"),
        ("lane not masked", "(1u << (gindex & 31u)) - 1u)", "0xffffffffu)")))
    print(f"{NAME}: {checked} wrong references and mutated sources checked")
    return 0


def mutate(directory, compiler, source, case, cmd, cso, mutations):
    count = 0
    for label, old, new in mutations:
        mutated = source.replace(old, new)
        if mutated == source:
            raise AssertionError(f"{NAME}: mutation '{label}' did not change the source")
        msrc, mcso = directory / "mutated.hlsl", directory / "mutated.cso"
        msrc.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", mcso, msrc], check=True)
        bad = subprocess.run(cmd(mcso, case), capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"{NAME}: mutation '{label}' escaped detection\n{bad.stderr}")
        count += 1
    return count


if __name__ == "__main__":
    mode, *args = sys.argv[1:] or ["controls"]
    controls()
    if mode == "controls":
        sys.exit(0)
    if mode != "draw" or len(args) != 3:
        sys.exit("usage: compute_alu_lane_gaps_check.py controls | draw <dxc> <renderer> <adapter>")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
