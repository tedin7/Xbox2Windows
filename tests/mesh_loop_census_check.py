"""Check the mesh loop census on source-built loops, one per shape it reports.

Each fixture is a few hand-written gfx1013 instructions assembled with LLVM 15.
The bytes are kept here so the check needs no LLVM; `binary` mode reassembles
every source and compares the llvm-mc listing with the Python decoder. None of
the fixtures is a complete mesh program: they carry the loop and nothing else.

Usage: mesh_loop_census_check.py [binary llvm-mc llvm-objcopy]
"""

from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import mesh_loop_census as census
from mesh_dynamic_geometry_check import WAVE64, decode as decode_geometry_program
from mesh_two_exits_check import mesh_container

LIMITS = (4, 2)
BACK_EDGE = "Expected a forward mesh branch"
SETREG = "Unsupported scalar instruction s_setreg_imm32_b32"
# Shape columns: barrier in body, LDS access in body, body writes EXEC, latch tests EXEC, loop-carried scalar, trip.
COUNTED = (False, False, False, False, True, "constant pattern")
SCALAR_OTHER = (False, False, False, False, True, "other")

# name: (source, LLVM 15 .text, shape of each back-edge in program order)
FIXTURES = {
    "loop free": (
        "s_mov_b32 s10, 0\ns_endpgm",
        "80038abe000081bf", []),
    "counted": (
        "s_mov_b32 s10, 0\nloop:\ns_add_u32 s10, s10, 1\ns_cmp_lt_u32 s10, 4\ns_cbranch_scc1 loop\ns_endpgm",
        "80038abe0a810a800a840abffdff85bf000081bf", [COUNTED]),
    "scalar, unknown bound": (
        "s_mov_b32 s10, 0\nloop:\ns_add_u32 s10, s10, 1\ns_cmp_lt_u32 s10, s11\ns_cbranch_scc1 loop\ns_endpgm",
        "80038abe0a810a800a0b0abffdff85bf000081bf", [SCALAR_OTHER]),
    "counted with a second exit": (
        "s_mov_b32 s10, 0\nloop:\ns_add_u32 s10, s10, 1\ns_cmp_eq_u32 s11, 7\ns_cbranch_scc1 done\n"
        "s_cmp_lt_u32 s10, 4\ns_cbranch_scc1 loop\ndone:\ns_endpgm",
        "80038abe0a810a800b8706bf020085bf0a840abffbff85bf000081bf", [SCALAR_OTHER]),
    "EXEC and scalar, unknown bound": (
        "s_mov_b32 s12, 0\nloop:\nv_cmpx_lt_u32_e32 0, v0\ns_add_u32 s12, s12, 1\nv_add_nc_u32_e32 v0, -1, v0\n"
        "s_cmp_lt_u32 s12, s11\ns_cbranch_scc1 loop\ns_endpgm",
        "80038cbe8000a27d0c810c80c100004a0c0b0abffbff85bf000081bf", [(False, False, True, False, True, "other")]),
    "EXEC latch": (
        "v_mov_b32_e32 v0, 1\nloop:\nv_add_nc_u32_e32 v0, -1, v0\nv_cmpx_lt_u32_e32 0, v0\ns_cbranch_execnz loop\n"
        "s_endpgm",
        "8102007ec100004a8000a27dfdff89bf000081bf", [(False, False, True, True, False, "other")]),
    "EXEC latch and scalar": (
        "s_mov_b32 s12, 0\nloop:\ns_add_u32 s12, s12, 1\nv_add_nc_u32_e32 v0, -1, v0\nv_cmpx_lt_u32_e32 0, v0\n"
        "s_cbranch_execnz loop\ns_endpgm",
        "80038cbe0c810c80c100004a8000a27dfcff89bf000081bf", [(False, False, True, True, True, "other")]),
    "VCC latch": (
        "loop:\nv_add_nc_u32_e32 v0, -1, v0\nv_cmp_lt_u32_e32 vcc, 0, v0\ns_cbranch_vccz loop\ns_endpgm",
        "c100004a8000827dfdff86bf000081bf", [(False, False, False, False, False, "other")]),
    "EXEC exit, unconditional latch": (
        "loop:\nv_cmpx_lt_u32_e32 0, v0\ns_cbranch_execz done\nv_add_nc_u32_e32 v0, -1, v0\ns_branch loop\n"
        "done:\ns_endpgm",
        "8000a27d020088bfc100004afcff82bf000081bf", [(False, False, True, False, False, "unconditional latch")]),
    "barrier in the body": (
        "s_mov_b32 s10, 0\nloop:\ns_barrier\ns_add_u32 s10, s10, 1\ns_cmp_lt_u32 s10, 4\ns_cbranch_scc1 loop\n"
        "s_endpgm",
        "80038abe00008abf0a810a800a840abffcff85bf000081bf", [(True, False, False, False, True, "constant pattern")]),
    "barrier before the header": (
        "s_mov_b32 s10, 0\ns_barrier\nloop:\ns_add_u32 s10, s10, 1\ns_cmp_lt_u32 s10, 4\ns_cbranch_scc1 loop\n"
        "s_endpgm",
        "80038abe00008abf0a810a800a840abffdff85bf000081bf", [COUNTED]),
    "LDS in the body": (
        "s_mov_b32 s10, 0\nloop:\nds_read_b32 v1, v0\ns_add_u32 s10, s10, 1\ns_cmp_lt_u32 s10, 4\n"
        "s_cbranch_scc1 loop\ns_endpgm",
        "80038abe0000d8d8000000010a810a800a840abffbff85bf000081bf",
        [(False, True, False, False, True, "constant pattern")]),
    "nested": (
        "s_mov_b32 s10, 0\nouter:\ns_mov_b32 s11, 0\ninner:\ns_add_u32 s11, s11, 1\ns_cmp_lt_u32 s11, 2\n"
        "s_cbranch_scc1 inner\ns_add_u32 s10, s10, 1\ns_cmp_lt_u32 s10, 4\ns_cbranch_scc1 outer\ns_endpgm",
        "80038abe80038bbe0b810b800b820abffdff85bf0a810a800a840abff9ff85bf000081bf",
        [COUNTED, COUNTED]),
    "flat-scratch setup": (
        "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0\ns_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_HI), 0\n"
        "s_mov_b32 s10, 0\nloop:\ns_add_u32 s10, s10, 1\ns_cmp_lt_u32 s10, 4\ns_cbranch_scc1 loop\ns_endpgm",
        "14f880ba0000000015f880ba0000000080038abe0a810a800a840abffdff85bf000081bf", [COUNTED]),
}

# Each mutant edits the decoded text of one fixture across one category boundary:
# (fixture, original line, replacement, shape of each back-edge afterwards).
MUTANTS = [
    ("counted", "s_cbranch_scc1 -3", "s_cbranch_scc1 0", []),  # no back-edge, no loop
    ("counted", "s_cmp_lt_u32 s10, 4", "s_cmp_lt_u32 s10, s11", [SCALAR_OTHER]),  # bound is not a constant
    ("counted", "s_mov_b32 s10, 0", "s_mov_b32 s10, s9", [SCALAR_OTHER]),  # start is not a constant
    ("counted", "s_add_u32 s10, s10, 1", "s_add_u32 s10, s10, s9", [SCALAR_OTHER]),  # step is not a constant
    ("counted", "s_add_u32 s10, s10, 1", "s_lshl_b32 s10, s10, 1", [SCALAR_OTHER]),  # counter not stepped by add/sub
    ("counted", "s_cbranch_scc1 -3", "s_cbranch_vccz -3", [SCALAR_OTHER]),  # latch does not test the compare
    ("counted", "s_add_u32 s10, s10, 1", "s_mov_b32 s10, 1", [(False, False, False, False, False, "other")]),
    ("counted with a second exit", "s_cbranch_scc1 2", "s_nop 0", [COUNTED]),  # only the second exit differed
    ("EXEC and scalar, unknown bound", "v_cmpx_lt_u32_e32 0, v0", "v_cmp_lt_u32_e32 vcc, 0, v0", [SCALAR_OTHER]),
    ("EXEC latch", "s_cbranch_execnz -3", "s_cbranch_vccz -3", [(False, False, True, False, False, "other")]),
    ("EXEC exit, unconditional latch", "s_branch -4", "s_cbranch_execnz -4",
     [(False, False, True, True, False, "other")]),
    ("barrier in the body", "s_barrier", "s_nop 0", [COUNTED]),
    ("LDS in the body", "ds_read_b32 v1, v0", "v_mov_b32_e32 v1, v0", [COUNTED]),
    ("nested", "s_mov_b32 s11, 0", "s_nop 0", [SCALAR_OTHER, COUNTED]),  # inner start unknown
]


def decode(code):
    """Adapt the Python disassembler spellings to LLVM's wave64 spellings."""
    return [(pc, text.replace("vcc_lo", "vcc")) for pc, text in decode_geometry_program(code)]


def program(name):
    return decode(bytes.fromhex(FIXTURES[name][1]))


def shapes(decoded):
    return [census.shape(decoded, header, latch) for header, latch in census.loops(decoded)]


def check_shapes():
    for name, (_, _, expected) in FIXTURES.items():
        assert shapes(program(name)) == expected, (name, shapes(program(name)))
    for name, original, replacement, expected in MUTANTS:
        decoded = program(name)
        assert [text for _, text in decoded].count(original) == 1, (name, original)
        mutant = [(pc, replacement if text == original else text) for pc, text in decoded]
        assert shapes(mutant) != shapes(decoded) and shapes(mutant) == expected, (name, replacement, shapes(mutant))


def check_branch_spelling():
    """LLVM 15 prints a negative SIMM16 unsigned; the census reads both spellings as one back-edge."""
    decoded = program("counted")
    unsigned = [(pc, text.replace("s_cbranch_scc1 -3", "s_cbranch_scc1 65533")) for pc, text in decoded]
    assert unsigned != decoded and census.signed_branches(unsigned) == decoded
    assert census.signed_branches(decoded) == decoded
    assert census.signed_branches([(0, "s_branch 32768")]) == [(0, "s_branch -32768")]
    assert census.signed_branches([(0, "s_branch 32767")]) == [(0, "s_branch 32767")]
    try:
        census.loops(unsigned)
    except ValueError as error:
        assert "instruction boundary" in str(error), error
    else:
        raise AssertionError("An unsigned back-edge offset was read as a forward branch")


def check_location():
    found = [(8, 20), (32, 40)]
    for offset, expected in ((0, "before the first loop"), (8, "in a loop body"), (20, "at a latch"),
                             (24, "after a loop"), (36, "in a loop body"), (40, "at a latch"), (44, "after a loop"),
                             (None, "after execution")):
        assert census.location("refused", offset, found) == expected, offset
    assert census.location(None, None, found) == "lowered"
    assert census.location("refused", 4, []) == "before the first loop"


def refusal(counters, lowerer, tier):
    return dict(counters["refusal"][(lowerer, tier)]), dict(counters["message"][(lowerer, tier)])


def check_lowerers():
    """Where the lowerers of this revision stop on a back-edge and on the flat-scratch setup."""
    as_decoded, skipped = census.TIERS
    for lowerer in census.LOWERERS:
        # A back-edge is refused by name at the latch, taken or not.
        counters = census.census([(program("counted"), LIMITS)])
        for tier in census.TIERS:
            assert refusal(counters, lowerer, tier) == (
                {("at a latch", "with loops"): 1}, {(BACK_EDGE, "with loops"): 1}), (lowerer, tier)
        # The scratch setup stops the program before its loop; with the two writes skipped the latch is reached.
        counters = census.census([(program("flat-scratch setup"), LIMITS)])
        assert refusal(counters, lowerer, as_decoded) == (
            {("before the first loop", "with loops"): 1}, {(SETREG, "with loops"): 1}), lowerer
        assert refusal(counters, lowerer, skipped) == (
            {("at a latch", "with loops"): 1}, {(BACK_EDGE, "with loops"): 1}), lowerer
        # An unsupported latch is counted by name at its own instruction.
        latch = [(pc, text.replace("s_cbranch_scc1", "s_cbranch_execnz")) for pc, text in program("counted")]
        counters = census.census([(latch, LIMITS)])
        assert refusal(counters, lowerer, as_decoded)[0] == {("at a latch", "with loops"): 1}, lowerer
        assert refusal(counters, lowerer, as_decoded)[1] == {
            ("Unsupported mesh branch s_cbranch_execnz", "with loops"): 1}, lowerer
        # A program without a back-edge is never placed relative to a loop.
        counters = census.census([(program("loop free"), LIMITS)])
        assert set(refusal(counters, lowerer, as_decoded)[0]) <= {
            ("after execution", "loop-free"), ("before the first loop", "loop-free")}, lowerer

    def exec_range():
        offset = 12
        return {}[offset]

    message, offset = census.first_refusal(exec_range)
    assert (message, offset) == ("exception KeyError", 12)
    assert census.location(message, offset, [(4, 12)]) == "at a latch"
    assert census.normalise(message, "s_cbranch_execnz -3") == "exception KeyError at s_cbranch_execnz"


def check_totals_and_output():
    counters = census.census([(program(name), LIMITS) for name in FIXTURES])
    loops = sum(len(expected) for _, _, expected in FIXTURES.values())
    assert counters["population"] == Counter({"with loops": len(FIXTURES) - 1, "loop-free": 1})
    assert counters["loops"]["back-edges"] == loops == sum(counters["shape"].values())
    assert sum(counters["latch"].values()) == loops
    assert counters["loops"]["nested back-edges"] == 1 and counters["loops"]["programs with 2 loops"] == 1
    assert counters["shape"] == Counter(shape for _, _, expected in FIXTURES.values() for shape in expected)
    for key in counters["refusal"]:
        assert sum(counters["refusal"][key].values()) == len(FIXTURES), key
    text = census.render(counters, Counter())
    assert "| no | no | no | no | yes | constant pattern | 5 | 3 / 3 / 7 |" in text, text
    assert "| yes | no | no | no | yes | constant pattern | 1 | 4 / 4 / 4 |" in text, text
    # Only counts and lowerer messages: no operand or encoding of an input.
    for leak in ("s10", "s11", "s12", "v0", "hwreg", "0x", "85bf"):
        assert leak not in text, leak
    assert census.normalise("Unproven runtime geometry word 3 in s14") == "Unproven runtime geometry word N in sN"
    assert census.normalise("exception KeyError", "s_cbranch_vccz -3") == "exception KeyError at s_cbranch_vccz"


def check_directory():
    """A directory holds two copies of one mesh container, a different one, another stage and a non-container."""
    def python_disassemble(code, library, *, cpu=b"gfx1013", offsets=False, features=b""):
        return decode(code) if offsets else [text for _, text in decode(code)]
    saved = census.pixel_aot.disassemble, census.mesh_entry_aot.disassemble
    census.pixel_aot.disassemble = census.mesh_entry_aot.disassemble = python_disassemble
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counted, free = (bytes.fromhex(FIXTURES[name][1]) for name in ("counted", "loop free"))
            (root / "nested").mkdir()
            (root / "a.bin").write_bytes(mesh_container(counted))
            (root / "nested" / "b.bin").write_bytes(mesh_container(counted))
            (root / "c.bin").write_bytes(mesh_container(free))
            (root / "d.bin").write_bytes(mesh_container(counted, stage=5))
            (root / "e.txt").write_bytes(b"not a container")
            population = Counter()
            programs = list(census.read_programs(root, population, None))
    finally:
        census.pixel_aot.disassemble, census.mesh_entry_aot.disassemble = saved
    assert sorted(shapes(decoded) for decoded, _ in programs) == [[], [COUNTED]], programs
    assert population["duplicate mesh programs"] == 1 and population["files without a DXBC container"] == 1
    assert sum(population.values()) == 3 and any(key.startswith("containers not decoded") for key in population)


def reassemble(assembler, objcopy):
    census.use_llvm_mc(assembler)
    for name, (source, code, _) in FIXTURES.items():
        with tempfile.TemporaryDirectory() as directory:
            asm, obj, raw = (Path(directory) / file for file in ("fixture.s", "fixture.o", "fixture.bin"))
            asm.write_text(source + "\n", encoding="ascii", newline="\n")
            subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-mattr=" + WAVE64,
                            "-filetype=obj", asm, "-o", obj], check=True)
            subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
            assert raw.read_bytes() == bytes.fromhex(code), name
        native = census.mesh_entry_aot.disassemble(bytes.fromhex(code), assembler, offsets=True,
                                                    features=WAVE64.encode("ascii"))
        if any(text.startswith(census.BRANCHES) and int(text.split()[1]) < 0 for _, text in program(name)):
            assert native != program(name), name  # llvm-mc spells the back-edge unsigned
        assert census.signed_branches(native) == program(name), (name, native)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "binary":
        reassemble(*sys.argv[2:])
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: mesh_loop_census_check.py [binary llvm-mc llvm-objcopy]")
    check_shapes()
    check_branch_spelling()
    check_location()
    check_lowerers()
    check_totals_and_output()
    check_directory()
    print(f"Mesh loop census classified {len(FIXTURES)} source-built programs; {len(MUTANTS)} mutants changed category")
