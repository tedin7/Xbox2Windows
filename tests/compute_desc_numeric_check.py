"""Check descriptor word 2 read as a number in the compute store profile, translated and dispatched.

A raw buffer descriptor's num_records word equals the bound view's byte size (the caller states this). A scalar
compare, select or lane write may then use it; every other descriptor word stays opaque.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base  # noqa: E402  (also puts tools/ on the path)

aot = base.aot

# One thread. s[12:15] is the raw output descriptor and s[16:19] the raw source descriptor. The harness binds
# both as 1280-byte raw views, so word 2 of each is 0x500. Results, in order:
#   eq on the output size -> select 7 (SCC set), lg between both sizes -> select 13 (SCC clear),
#   a select that keeps the source size, and a lane write of the output size.
DESC_NUMERIC = [
    "s_mov_b32 s9, 0",
    "s_load_dwordx4 s[12:15], s[8:9], null",
    "s_load_dwordx4 s[16:19], s[8:9], 0x10",
    "s_waitcnt lgkmcnt(0)",
    "s_cmp_eq_i32 s14, 0x500",
    "s_cselect_b32 s20, 7, 9",
    "s_cmp_lg_u32 s14, s18",
    "s_cselect_b32 s21, 11, 13",
    "s_cmp_eq_u32 s18, 0x500",
    "s_cselect_b32 s22, s18, 0",
    "v_mov_b32_e32 v4, s20",
    "v_mov_b32_e32 v5, s21",
    "v_mov_b32_e32 v6, s22",
    "v_mov_b32_e32 v7, 0",
    "v_writelane_b32 v7, s14, 0",
    "buffer_store_dwordx4 v[4:7], off, s[12:15], 0",
    "s_endpgm",
]
BINDING = dict(table_reg=8, descriptor_offset=0, source_descriptor=16)
RESULTS = (7, 13, 0x500, 0x500)
UAV_BYTES = "uav_bytes()"
SRV_BYTES = "srv_bytes()"


def lower(lines=DESC_NUMERIC, **binding):
    return aot.lower(lines, 1, **(BINDING | binding))


def expected_words(results=RESULTS):
    words = [base.SENTINEL] * 320
    words[:4] = results
    return words


def controls():
    source = lower()
    assert f"uint uav_bytes() {{ uint n; outputData.GetDimensions(n); return n; }}" in source, source
    assert f"uint srv_bytes() {{ uint n; inputData.GetDimensions(n); return n; }}" in source, source
    assert f"(({UAV_BYTES} == 0x00000500u) ? 0x00000007u : 0x00000009u)" in source, source
    assert f"(({UAV_BYTES} != {SRV_BYTES}) ? 0x0000000bu : 0x0000000du)" in source, source
    assert f"(((gindex & 31u) == 0x00000000u) ? {UAV_BYTES} : 0x00000000u)" in source, source
    # The helpers are only declared when used.
    assert "uav_bytes" not in aot.lower(base.FIXED, 1, **base.FIXED_BINDING)

    # Every descriptor word other than word 2 stays opaque.
    for reg in ("s12", "s13", "s15", "s16", "s17", "s19"):
        base.rejects("Unverified scalar compare operand", lower,
                     base.variant(DESC_NUMERIC, ("s_cmp_eq_i32 s14, 0x500", f"s_cmp_eq_i32 {reg}, 0x500")))
    base.rejects("Unverified writelane scalar source", lower,
                 base.variant(DESC_NUMERIC, ("v_writelane_b32 v7, s14, 0", "v_writelane_b32 v7, s12, 0")))
    base.rejects("Unverified scalar ALU operand", lower,
                 base.variant(DESC_NUMERIC, ("s_cmp_eq_i32 s14, 0x500", "s_add_u32 s23, s13, 1")))
    # A descriptor with a stride, or an image output, has no byte size in word 2.
    base.rejects("Unverified scalar compare operand", aot.lower, DESC_NUMERIC, 1, **(BINDING | dict(stride=4,
                 group_reg=10)))
    base.rejects("Unverified scalar compare operand", lower, source_stride=8)
    # The constant buffer descriptor is not modelled as a size.
    cbv = base.variant(DESC_NUMERIC, ("s_cmp_eq_i32 s14, 0x500", "s_cmp_eq_i32 s30, 0x500"),
                       ("s_waitcnt lgkmcnt(0)", "s_load_dwordx4 s[28:31], s[8:9], 0x20\ns_waitcnt lgkmcnt(0)"))
    base.rejects("Unverified scalar compare operand", lower, cbv, constant_descriptor=32)
    # A pending load is no operand.
    base.rejects("Unverified scalar compare operand", lower,
                 base.variant(DESC_NUMERIC, ("s_waitcnt lgkmcnt(0)", "s_nop 0")))

    # SCC comes only from the last compare: nothing that writes SCC may sit between it and the select.
    for between in ("s_add_u32 s24, 1, 2", "s_and_b32 s24, 1, 3", "s_bcnt1_i32_b32 s24, exec_lo"):
        base.rejects("Unverified SCC for scalar select", lower,
                     base.variant(DESC_NUMERIC, ("s_cselect_b32 s20, 7, 9", f"{between}\ns_cselect_b32 s20, 7, 9")))
    base.rejects("Unverified SCC for scalar select", lower,
                 base.variant(DESC_NUMERIC, ("s_cmp_eq_i32 s14, 0x500", "s_nop 0")))
    # A select keeps SCC for the next one; a move does not touch it.
    kept = lower(base.variant(DESC_NUMERIC, ("s_cselect_b32 s20, 7, 9", "s_cselect_b32 s20, 7, 9\ns_mov_b32 s23, 5\n"
                                             "s_cselect_b32 s20, 1, 2")))
    assert "? 0x00000001u : 0x00000002u" in kept, kept
    # Other compares, 64-bit selects and SCC branches stay refused.
    for old, new, message in (
            ("s_cmp_eq_i32 s14, 0x500", "s_cmp_lt_i32 s14, 0x500", "Unsupported compute instruction: s_cmp_lt_i32"),
            ("s_cmp_eq_i32 s14, 0x500", "s_cmp_eq_u64 s[14:15], 0x500", "Unsupported compute instruction: s_cmp_eq_u64"),
            ("s_cselect_b32 s20, 7, 9", "s_cselect_b64 s[20:21], 7, 9",
             "Unsupported compute instruction: s_cselect_b64"),
            ("s_cselect_b32 s20, 7, 9", "s_cselect_b32 s20, v0, 9", "Unsupported compute instruction: s_cselect_b32"),
            ("s_cselect_b32 s20, 7, 9", "s_cbranch_scc1 .LBB0_1", "Unsupported compute instruction: s_cbranch_scc1"),
            ("s_cselect_b32 s20, 7, 9", "s_cselect_b32 s106, 7, 9", "Unverified scalar select operand"),
            ("s_cmp_eq_i32 s14, 0x500", "s_cmp_eq_i32 s30, 0x500", "Unverified scalar compare operand")):
        base.rejects(message, lower, base.variant(DESC_NUMERIC, (old, new)))


def binary(directory, assembler, objcopy, library):
    """The assembled program decodes to its text and lowers to the same translation."""
    decoded = aot.labelled(base.disassemble(base.assemble(directory, assembler, objcopy, DESC_NUMERIC), library,
                                            offsets=True))
    assert aot.lower(decoded, 1, **BINDING) == lower(), decoded


def run(cmd, case, expected):
    case.write_bytes(struct.pack("<321I", 1, *expected))
    return subprocess.run(cmd, capture_output=True, text=True)


def draw(directory, compiler, renderer, backend):
    src, cso, case = [directory / x for x in ("desc.hlsl", "desc.cso", "desc.case")]
    source = lower()
    src.write_text(source, encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    cmd = [renderer, backend, "--compute-store-sized-case", cso, case]
    good = run(cmd, case, expected_words())
    sys.stderr.write(good.stderr)
    if good.returncode:
        return good.returncode
    # References that read the size, the compare or the select wrongly must fail on the device.
    for label, results, word in (("size read as 0", (7, 13, 0, 0), 2),
                                 ("compare result inverted", (9, 13, 0x500, 0x500), 0),
                                 ("select arms swapped", (7, 11, 0x500, 0x500), 1),
                                 ("lane write dropped", (7, 13, 0x500, 0), 3)):
        wrong = expected_words(results)
        bad = run(cmd, case, wrong)
        if bad.returncode != 1 or f"word={word} expected=" not in bad.stderr:
            raise AssertionError(f"reference '{label}' escaped detection\n{bad.stderr}")
    # A translation that does not read the real size (the lowering disabled) must fail on the device.
    for label, old, new in (("uav size fixed at 0", "outputData.GetDimensions(n)", "n = 0"),
                            ("srv size fixed at 4", "inputData.GetDimensions(n)", "n = 4"),
                            ("equality inverted", "== 0x00000500u", "!= 0x00000500u"),
                            ("lg as eq", f"({UAV_BYTES} != {SRV_BYTES})", f"({UAV_BYTES} == {SRV_BYTES})")):
        mutated = source.replace(old, new)
        if mutated == source:
            raise AssertionError(f"mutation '{label}' did not change the source")
        msrc, mcso = directory / "mutated.hlsl", directory / "mutated.cso"
        msrc.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", mcso, msrc], check=True)
        case.write_bytes(struct.pack("<321I", 1, *expected_words()))
        bad = subprocess.run([*cmd[:3], mcso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"mutation '{label}' escaped detection\n{bad.stderr}")
    return 0


if __name__ == "__main__":
    controls()
    mode, *args = sys.argv[1:] or ["controls"]
    if mode == "controls" and not args:
        sys.exit(0)
    with tempfile.TemporaryDirectory() as tmp:
        if mode == "binary" and len(args) == 3:
            sys.exit(binary(Path(tmp), *args) or 0)
        if mode == "draw" and len(args) == 3:
            sys.exit(draw(Path(tmp), *args))
    sys.exit("usage: compute_desc_numeric_check.py controls | binary <mc> <objcopy> <llvm> | draw <dxc> <renderer> <adapter>")
