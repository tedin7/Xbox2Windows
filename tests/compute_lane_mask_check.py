"""Check s_bitreplicate_b64_b32 lane masks consumed by v_cndmask_b32_e64 in the compute store profile."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base  # noqa: E402

aot = base.aot
SENTINEL = base.SENTINEL
# Source-built gfx1013 text in LLVM 15 spelling, for an amdgpu_cs entry with a 32-bit table pointer in s0 and the
# group X ID in s1, a one-dimensional group of 32 and u0 at table byte 0. Each thread writes one 16-byte element:
#   gid = group.x * 32 + local.x;  lane = local.x;
#   m = bitreplicate(0x3f3f3f3f);  n = bitreplicate(0x9c3a5e17);    // 64-bit masks
#   out[gid] = { m.lo[lane] ? 0x600d : gid, n.lo[lane] ? gid : 0xbeef, n.hi[lane] ? gid : 0xffffffff, n.hi };
# 0x3f3f3f3f is the pattern of the measured sites; 0x9c3a5e17 has different, irregular halves.
LANE_MASK = [
    "s_mov_b32 s2, s1",
    "s_mov_b32 s1, 0",
    "v_lshl_add_u32 v0, s2, 5, v0",
    "s_load_dwordx4 s[4:7], s[0:1], 0x0",
    "s_bitreplicate_b64_b32 s[8:9], 0x3f3f3f3f",
    "s_bitreplicate_b64_b32 s[10:11], 0x9c3a5e17",
    "s_mov_b32 s14, 0xbeef",
    "v_cndmask_b32_e64 v1, v0, 0x600d, s8",
    "v_cndmask_b32_e64 v2, s14, v0, s10",
    "v_cndmask_b32_e64 v3, -1, v0, s11",
    "v_mov_b32_e32 v4, s11",
    "s_waitcnt lgkmcnt(0)",
    "buffer_store_dwordx4 v[1:4], v0, s[4:7], 0 idxen",
    "s_endpgm",
]
BINDING = dict(table_reg=0, descriptor_offset=0, group_reg=1, stride=16)
# The same program for a group of 16: one wave of 16 lanes at either wave size, so the lane is local X, not gid.
LANE_MASK_16 = base.variant(LANE_MASK, ("v_lshl_add_u32 v0, s2, 5, v0", "v_lshl_add_u32 v0, s2, 4, v0"))
CASES = [("lane_mask", LANE_MASK, 32), ("lane_mask_16", LANE_MASK_16, 16)]
# Dispatches per case: every one fits the 320 readback words at four words per thread.
RUNS = {"lane_mask": (1, 2), "lane_mask_16": (1, 3, 5)}
SOURCES = (0x3f3f3f3f, 0x9c3a5e17)


def replicate(word):
    """Independent reference from the ISA text: for i in 0..31, D.u64[2i] = D.u64[2i + 1] = S0.u32[i]."""
    pair = 0
    for i in range(32):
        bit = (word >> i) & 1
        pair |= bit << (2 * i) | bit << (2 * i + 1)
    return pair & 0xffffffff, pair >> 32


def expected_words(width, groups, *, raw=False, halves=(0, 0, 1), swap=False, lane=None):
    """Expected readback; the keyword arguments build the wrong references of the negative controls."""
    m, n = (((w & 0xffffffff, w) if raw else replicate(w)) for w in SOURCES)
    masks = (m[halves[0]], n[halves[1]], n[halves[2]])
    words = [SENTINEL] * 320
    for gid in range(width * groups):
        index = (lane or (lambda g: g % width))(gid) % 32
        bits = [(mask >> index) & 1 for mask in masks]
        if swap:
            bits = [1 - bit for bit in bits]
        words[4 * gid:4 * gid + 4] = [0x600d if bits[0] else gid, gid if bits[1] else 0xbeef,
                                      gid if bits[2] else 0xffffffff, n[1]]
    return words


def lower(lines, width, **change):
    return aot.lower(lines, width, **(BINDING | change)).splitlines()


def select(mask, width, on, off):
    return f"(((0x{mask:08x}u >> (tid.x % {width}u)) & 1u) != 0u ? {on} : {off})"


def controls():
    """Exact output, the replicated words, then one rejection per source, destination, mask or group outside the subset."""
    assert replicate(0x0f0f) == (0x00ff00ff, 0) and replicate(0x80000001) == (0x3, 0xc0000000)
    assert replicate(0x3f3f3f3f) == (0x0fff0fff, 0x0fff0fff) and replicate(0x9c3a5e17) == (0x33fc033f, 0xc3f00fcc)
    for _, lines, width in CASES:
        m, n = replicate(SOURCES[0]), replicate(SOURCES[1])
        store = ("    outputData.Store4(tid.x * 16u + 0u, uint4("
                 f"{select(m[0], width, '0x0000600du', 'tid.x')}, {select(n[0], width, 'tid.x', '0x0000beefu')}, "
                 f"{select(n[1], width, 'tid.x', '0xffffffffu')}, 0x{n[1]:08x}u));")
        assert lower(lines, width)[-2:] == [store, "}"], lower(lines, width)
    lines = lower(LANE_MASK, 32)

    def mask(*changes, width=32, lines=LANE_MASK, **binding):
        return lower(base.variant(lines, *changes), width, **binding)

    # A copy of a mask word and a mask moved in as an immediate are the same program.
    assert mask(("v1, v0, 0x600d, s8", "v1, v0, 0x600d, s20"),
                ("s_mov_b32 s14, 0xbeef", "s_mov_b32 s14, 0xbeef\ns_mov_b32 s20, s8")) == lines
    assert mask(("v1, v0, 0x600d, s8", "v1, v0, 0x600d, s20"),
                ("s_mov_b32 s14, 0xbeef", "s_mov_b32 s14, 0xbeef\ns_mov_b32 s20, 0xfff0fff")) == lines
    # An inline source: -1 fills the pair and 0 clears it. Bits 0-15 fill the low word, bits 16-31 the high word.
    assert select(0xffffffff, 32, "0x0000600du", "tid.x") in mask(("0x3f3f3f3f", "-1"))[-2]
    assert select(0, 32, "0x0000600du", "tid.x") in mask(("0x3f3f3f3f", "0"))[-2]
    assert select(0xffffffff, 32, "0x0000600du", "tid.x") in mask(("0x3f3f3f3f", "0xffff"))[-2]
    assert select(0, 32, "0x0000600du", "tid.x") in mask(("0x3f3f3f3f", "0xffff0000"))[-2]
    # src0 is chosen where the mask bit is clear and src1 where it is set.
    assert select(0x0fff0fff, 32, "tid.x", "0x0000600du") in mask(("v1, v0, 0x600d, s8", "v1, 0x600d, v0, s8"))[-2]
    unsupported = "Unsupported compute instruction: "
    source, destination, lane_mask = ("Unverified lane mask source", "Invalid lane mask destination",
                                      "Unverified lane mask")
    group = "Lane mask select requires a one-dimensional group of at most 32 threads"
    for message, changes, extra in (
        # Non-immediate sources: an SGPR (known or not), a VGPR, a literal wider than 32 bits, a float constant.
        (source, (("s[8:9], 0x3f3f3f3f", "s[8:9], s14"),), {}),
        (source, (("s[8:9], 0x3f3f3f3f", "s[8:9], s2"),), {}),
        (source, (("s[8:9], 0x3f3f3f3f", "s[8:9], v0"),), {}),
        (source, (("s[8:9], 0x3f3f3f3f", "s[8:9], 0x13f3f3f3f"),), {}),
        (source, (("s[8:9], 0x3f3f3f3f", "s[8:9], 1.0"),), {}),
        # An odd, non-consecutive, out-of-range or pending destination pair, and a single SGPR.
        (destination, (("s[8:9], 0x3f3f3f3f", "s[9:10], 0x3f3f3f3f"),), {}),
        (destination, (("s[8:9], 0x3f3f3f3f", "s[8:10], 0x3f3f3f3f"),), {}),
        (destination, (("s[8:9], 0x3f3f3f3f", "s[106:107], 0x3f3f3f3f"),), {}),
        (destination, (("s[8:9], 0x3f3f3f3f", "s[6:7], 0x3f3f3f3f"),), {}),
        (unsupported + "s_bitreplicate_b64_b32 s8", (("s[8:9], 0x3f3f3f3f", "s8, 0x3f3f3f3f"),), {}),
        # A mask that is not a statically known word: unwritten, the group ID, a pending descriptor, a constant word.
        (lane_mask, (("0x600d, s8", "0x600d, s20"),), {}),
        (lane_mask, (("0x600d, s8", "0x600d, s2"),), {}),
        (lane_mask, (("0x600d, s8", "0x600d, s4"),), {}),
        (lane_mask, (("s_bitreplicate_b64_b32 s[8:9], 0x3f3f3f3f", "s_mov_b32 s8, 0x3f3f3f3f\ns_add_u32 s8, s8, s14"),
                     ("s_mov_b32 s14, 0xbeef", ""), ("s_mov_b32 s1, 0", "s_mov_b32 s1, 0\ns_mov_b32 s14, 0xbeef")), {}),
        # Mask uses the profile does not lower: a wave64 pair, exec_lo as the mask, an exec write from the mask,
        # and a saveexec from it instead of a compare; vcc_lo without a prior v_cmp_* cond is an unverified lane mask.
        (unsupported + "v_cndmask_b32_e64 v1, v0, 0x600d, s[8:9]", (("0x600d, s8", "0x600d, s[8:9]"),), {}),
        (lane_mask, (("0x600d, s8", "0x600d, vcc_lo"),), {}),
        (unsupported + "v_cndmask_b32_e64 v1, v0, 0x600d, exec_lo", (("0x600d, s8", "0x600d, exec_lo"),), {}),
        (lane_mask, (("v_cndmask_b32_e64 v2, s14, v0, s10",
                      "v_cndmask_b32_e32 v2, s14, v0, vcc_lo"),), {}),
        (unsupported + "s_mov_b32 exec_lo, s8", (("s_mov_b32 s14, 0xbeef", "s_mov_b32 s14, 0xbeef\n"
                                                                            "s_mov_b32 exec_lo, s8"),), {}),
        (unsupported + "s_and_saveexec_b32 s20, s8", (("s_mov_b32 s14, 0xbeef", "s_mov_b32 s14, 0xbeef\n"
                                                                                 "s_and_saveexec_b32 s20, s8"),), {}),
        # Source modifiers and clamp on the select.
        (unsupported + "v_cndmask_b32_e64", (("v1, v0, 0x600d, s8", "v1, -v0, 0x600d, s8"),), {}),
        (unsupported + "v_cndmask_b32_e64", (("v1, v0, 0x600d, s8", "v1, |v0|, 0x600d, s8"),), {}),
        (unsupported + "v_cndmask_b32_e64", (("v1, v0, 0x600d, s8", "v1, v0, 0x600d, s8 clamp"),), {}),
        # Select sources that are not known words, and a destination past v255.
        ("Unverified ALU operand", (("v2, s14, v0, s10", "v2, s20, v0, s10"),), {}),
        ("Unverified ALU operand", (("v2, s14, v0, s10", "v2, s14, v7, s10"),), {}),
        ("Invalid vector destination", (("v1, v0, 0x600d, s8", "v256, v0, 0x600d, s8"),), {}),
        # A select result is data only: not a store address.
        ("Unsupported store address", (("v[1:4], v0, s[4:7]", "v[1:4], v1, s[4:7]"),), {}),
        # Groups whose lanes depend on the unknown wave size, or on more than local X.
        (group, (("v_lshl_add_u32 v0, s2, 5, v0", "v_lshl_add_u32 v0, s2, 6, v0"),), dict(width=64)),
        (group, (), dict(width=(32, 2, 1))),
        (group, (), dict(width=(16, 1, 2), lines=LANE_MASK_16)),
    ):
        refuses(message, mask, *changes, **extra)


def refuses(message, function, *args, **kwargs):
    """base.rejects, which also requires that the call is refused at all."""
    try:
        function(*args, **kwargs)
    except ValueError:
        return base.rejects(message, function, *args, **kwargs)
    raise AssertionError(f"Expected a rejection: {message}")


def write(case, groups, words):
    case.write_bytes(struct.pack("<321I", groups, *words))


def draw(directory, compiler, renderer, backend):
    accepted = set()
    for name, lines, width in CASES:
        src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
        source = aot.lower(lines, width, **BINDING)
        accepted.update(line.split()[0] for line in lines if line.strip() and not line.strip().endswith(":"))
        src.write_text(source, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
        cmd = [renderer, backend, "--compute-store-case", cso, case]
        for groups in RUNS[name]:
            write(case, groups, expected_words(width, groups))
            result = subprocess.run(cmd)
            if result.returncode:
                return result.returncode
        groups = RUNS[name][-1]
        expected = expected_words(width, groups)
        # Wrong references, each built from the same model: each must fail at its first differing word.
        references = [("source not replicated", dict(raw=True)), ("high half for low", dict(halves=(1, 1, 1))),
                      ("low half for high", dict(halves=(0, 0, 0))), ("arms swapped", dict(swap=True)),
                      ("lane plus one", dict(lane=lambda g: g % width + 1))]
        if width < 32:
            references.append(("dispatch index as lane", dict(lane=lambda g: g)))
        for label, reference in references:
            wrong = expected_words(width, groups, **reference)
            word = next(i for i, (a, b) in enumerate(zip(expected, wrong)) if a != b)
            write(case, groups, wrong)
            bad = subprocess.run(cmd, capture_output=True, text=True)
            if bad.returncode != 1 or f"word={word} expected=" not in bad.stderr:
                raise AssertionError(f"{name}: reference '{label}' escaped detection\n{bad.stdout}\n{bad.stderr}")
        # A changed sentinel just past the last thread, when the dispatch leaves one.
        untouched = 4 * width * groups
        if untouched < 320:
            wrong = list(expected)
            wrong[untouched] ^= 1
            write(case, groups, wrong)
            bad = subprocess.run(cmd, capture_output=True, text=True)
            if bad.returncode != 1 or f"word={untouched} expected=" not in bad.stderr:
                raise AssertionError(f"{name}: a wrong untouched word escaped detection\n{bad.stderr}")
        write(case, groups, expected)
        m, n = replicate(SOURCES[0]), replicate(SOURCES[1])
        mutations = [("arms swapped", "? 0x0000600du : tid.x)", "? tid.x : 0x0000600du)"),
                     ("source not replicated", f"(0x{n[0]:08x}u >>", f"(0x{SOURCES[1]:08x}u >>"),
                     ("mask bit inverted", f"(0x{m[0]:08x}u >> (tid.x % {width}u)) & 1u) != 0u",
                      f"(0x{m[0]:08x}u >> (tid.x % {width}u)) & 1u) == 0u"),
                     ("lane plus one", f"(0x{n[1]:08x}u >> (tid.x % {width}u))",
                      f"(0x{n[1]:08x}u >> (tid.x % {width}u + 1u))")]
        if width < 32:
            mutations.append(("dispatch index as lane", f"(tid.x % {width}u)", "(tid.x)"))
        for label, old, new in mutations:
            mutated = source.replace(old, new)
            if mutated == source:
                raise AssertionError(f"{name}: mutation '{label}' did not change the source")
            msrc, mcso = directory / "mutated.hlsl", directory / "mutated.cso"
            msrc.write_text(mutated, encoding="ascii")
            subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", mcso, msrc], check=True)
            bad = subprocess.run([*cmd[:3], mcso, case], capture_output=True, text=True)
            if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
                raise AssertionError(f"{name}: mutation '{label}' escaped detection\n{bad.stderr}")
        print(f"{name}: {len(RUNS[name])} dispatches, {len(references)} wrong references and "
              f"{len(mutations)} mutated sources detected", flush=True)
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{op}\n" for op in sorted(accepted)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    controls()
    mode, *args = sys.argv[1:]
    if mode == "controls":
        sys.exit(0)
    if mode != "draw" or len(args) != 3:
        sys.exit("usage: compute_lane_mask_check.py controls | draw <dxc> <renderer> <adapter>")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
