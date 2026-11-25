"""Check the group-uniform "any lane active" branch of a source-built mesh program.

Each wave stores whether it has a live lane in LDS, the group barrier publishes the
stores, and both waves branch on the combined result with SCC. A wave that looked
only at its own lanes would pick a different allocation than the group.

Usage: mesh_scc_any_lane_check.py [binary llvm-mc llvm-objcopy libLLVM]
"""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
from mesh_entry_aot import Wave, lower, run_group
from mesh_dynamic_geometry_check import SECOND, WAVE64, decode as decode_geometry_program
from pixel_aot import disassemble

SOURCE = HERE / "mesh_scc_any_lane.s"
LIMITS = (4, 2)
GROUP_LANES = 128

# LLVM 15 .text for SOURCE. Keep these bytes in source so CPU checks need no LLVM;
# `binary` mode compares them with a fresh assembly of SOURCE.
CODE = bytes.fromhex(
    """
    11038abe 80038bbe 050608f4 200000fa
    7fc08cbf cc0320f4 000000fa 8c0320f4
    040000fa 7fc08cbf 000065d7 c1000100
    000066d7 c1000200 03ff8893 18000400
    000046d7 080c0104 0f00867d 6a8013bf
    81800685 0602187e 0882098f 09021a7e
    000034d8 0d0c0000 7fc08cbf 00008abf
    80021c7e 0000d8d8 0e00000f 0400d8d8
    0e000010 7fc08cbf 0f050c7e 10050e7e
    06070688 ff038fbe 03100000 068007bf
    020084bf ff038fbe 04200000 0fff9093
    00000c00 0f8c1390 000065d7 c1000100
    000066d7 c1000200 03ff8893 18000400
    000046d7 080c0104 7e0496be 8000a47d
    020088bf 0f03fcbe 090090bf 1604febe
    00008abf 1000a87d 81001436 82001636
    1400c2d4 81140200 0c00c2d4 82160200
    010001d5 f3e45100 020001d5 f3e43100
    050001d5 80e45100 060001d5 80e43100
    8002067e f202087e f0020e7e cf0800f8
    01020304 0f0200f8 05060704 1604febe
    1300a87d 1400c2d4 81000200 ff02127e
    00042000 090001d5 091d5000 410900f8
    09000000 000081bf
    """
)
NOP = 0xBF800000


def decode(code):
    """Adapt the Python disassembler spellings to LLVM's wave64 spellings."""
    return [(pc, text.replace("vcc_lo", "vcc")) for pc, text in decode_geometry_program(code)]


def patched(code, instruction, replacement):
    """Replace the first instruction starting with `instruction` by `replacement` words."""
    offsets = [pc for pc, text in decode(code) if text.startswith(instruction)]
    assert offsets, instruction
    start = offsets[0]
    following = [pc for pc, _ in decode(code) if pc > start]
    end = following[0] if following else len(code)
    words = struct.pack(f"<{len(replacement)}I", *replacement)
    assert len(words) == end - start, (instruction, len(words), end - start)
    return code[:start] + words + code[end:]


def group_reference(threshold):
    """The allocation an independent model of the group expects for threshold T."""
    any_live = any(lane >= threshold for lane in range(GROUP_LANES))
    return (4, 2) if any_live else (3, 1)


def run(code, threshold):
    waves = [Wave(index, [threshold, SECOND]) for index in range(2)]
    run_group(waves, decode(code), {})
    return waves


def check_allocation(code):
    for threshold in (0, 1, 63, 64, 100, 127, 128, 200, 0xFFFFFFFF):
        first, second = run(code, threshold)
        expected = group_reference(threshold)
        assert first.counts == [expected] and not second.counts, (threshold, first.counts)
        assert first.barriers == second.barriers and len(first.barriers) == 2, threshold
        flags = [any(lane >= threshold for lane in range(64 * w, 64 * w + 64)) for w in range(2)]
        assert [bool(first.lds[4 * w]) for w in range(2)] == flags, (threshold, first.lds)
        assert len(first.positions) == expected[0] and len(first.indices) == expected[1], threshold
        # The shader that lowering emits declares the same allocation.
        text = lower(decode(code), [threshold, SECOND], LIMITS)
        assert f"SetMeshOutputCounts({expected[0]},{expected[1]});" in text, threshold


def refuses(code, text, threshold=100):
    try:
        run(code, threshold)
    except ValueError as error:
        assert text in str(error), error
    else:
        raise AssertionError(f"Expected refusal: {text}")


def check_refusals(code):
    # A wave alone cannot see the other wave's flag.
    try:
        Wave(0, [100, SECOND]).run(decode(code), {})
    except ValueError as error:
        assert "no wave published" in str(error), error
    else:
        raise AssertionError("A single wave read a flag nobody stored")
    refuses(patched(code, "s_barrier", [NOP]), "before a barrier")
    refuses(patched(code, "ds_write_b32", [NOP, NOP]), "no wave published")
    # Waits 0 and 1 follow the scalar loads; 2 follows the store and 3 the loads.
    refuses(patched_nth(code, "s_waitcnt lgkmcnt(0)", 2, [NOP]), "not waited before the barrier")
    refuses(patched_nth(code, "s_waitcnt lgkmcnt(0)", 3, [NOP]), "before wait")


def patched_nth(code, instruction, index, replacement):
    offsets = [pc for pc, text in decode(code) if text.startswith(instruction)]
    start = offsets[index]
    size = 4 * len(replacement)
    return code[:start] + struct.pack(f"<{len(replacement)}I", *replacement) + code[start + size:]


def check_mutants_change_the_result(code):
    """Each mutant must produce a different allocation than the reference for some T."""
    own_flag_only = patched(code, "ds_read_b32 v16", [0xD8D80000, 0x1000000E])
    flipped_branch = patched(code, "s_cbranch_scc0", [0xBF850002])
    narrow_compare = patched(code, "s_cmp_lg_u64", [0xBF07806A])
    for name, mutant in (("own flag only", own_flag_only), ("flipped branch", flipped_branch),
                         ("32-bit mask compare", narrow_compare)):
        wrong = [t for t in (0, 64, 100, 127, 128, 200)
                 if run(mutant, t)[0].counts != [group_reference(t)]]
        assert wrong, name


class StubWave:
    """A wave that only reports the LDS traffic of its barrier intervals."""

    def __init__(self, intervals):
        self.intervals = intervals
        self.staged, self.reads = {}, set()

    def steps(self, decoded, expressions):
        for staged, reads in self.intervals:
            self.staged, self.reads = dict(staged), set(reads)
            yield


def check_interval_hazards():
    """A store and a read of one dword in one interval are refused for either wave order."""
    store, read = ({0: 7}, ()), ({}, (0,))
    for first, second in ((store, read), (read, store)):
        for barrier_ends_interval in (True, False):
            tail = [({}, ())] if barrier_ends_interval else []
            waves = [StubWave([first] + tail), StubWave([second] + tail)]
            try:
                run_group(waves, [], {})
            except ValueError as error:
                assert "without a barrier" in str(error), error
            else:
                raise AssertionError((first, second, barrier_ends_interval))
    # Disjoint dwords stay accepted.
    run_group([StubWave([({0: 7}, ())]), StubWave([({}, (4,))])], [], {})


def check_pending_lds_destination():
    """A vector load or ALU write to a register with an unwaited LDS read is refused."""
    buffer_load = "buffer_load_dword v1, v5, s[32:35], 0 offen"
    for line in (buffer_load, "v_mov_b32_e32 v1, 0"):
        wave = Wave(0, [0] * 4)
        wave.lds = {0: 9}
        wave.exec = 1
        for j in range(4):
            wave.s[32 + j] = (32, j)
        wave.lds_op("ds_read_b32", "v1, v5")
        try:
            list(wave.steps([(0, line)], {}))
        except ValueError as error:
            assert "ending" in str(error) or "overwrites" in str(error), error
        else:
            raise AssertionError(line)


def reassemble(assembler, objcopy, library):
    with tempfile.TemporaryDirectory() as directory:
        obj, raw = (Path(directory) / name for name in ("fixture.o", "fixture.bin"))
        subprocess.run(
            [assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-mattr=" + WAVE64,
             "-filetype=obj", SOURCE, "-o", obj],
            check=True,
        )
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
        code = raw.read_bytes()
    assert code == CODE, "Embedded bytes differ from the assembled mesh"
    native = disassemble(code, library, offsets=True, features=WAVE64.encode("ascii"))
    for llvm_instruction, python_instruction in zip(native, decode(code), strict=True):
        assert llvm_instruction == python_instruction, (llvm_instruction, python_instruction)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "binary":
        reassemble(*sys.argv[2:])
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: mesh_scc_any_lane_check.py [binary llvm-mc llvm-objcopy libLLVM]")
    check_allocation(CODE)
    check_refusals(CODE)
    check_mutants_change_the_result(CODE)
    check_interval_hazards()
    check_pending_lds_destination()
    print("Group-wide any-lane branch matched the CPU model; mutants and refusals held")
