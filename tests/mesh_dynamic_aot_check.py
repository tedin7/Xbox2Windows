"""Check bounded runtime mesh specialization against the existing CPU oracle."""

from pathlib import Path
import struct
import sys
from unittest.mock import patch

from mesh_dynamic_geometry_check import CASES, CODE, MESH_LIMITS, RETIRED_PREFIX, SECOND, Wave, decode, words
from mesh_dynamic_geometry_check import WIDE_COUNTS, WIDE_SECOND, surviving_wave_code

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mesh_dynamic_aot import enumerate_variants, lower
from mesh_entry_aot import lower as lower_static


def select(variants, geometry):
    matches = [variant for variant in variants
               if all((geometry[index] == value) == equal for index, value, equal in variant.conditions)]
    assert len(matches) == 1, (geometry, matches)
    return matches[0]


def expect_refusal(decoded, geometry_word_count, text, limits=MESH_LIMITS):
    try:
        lower(decoded, geometry_word_count, limits)
    except ValueError as error:
        assert text in str(error), error
    else:
        raise AssertionError(f"Expected refusal: {text}")


def check_retired_wave():
    body = decode(CODE)
    decoded = decode(RETIRED_PREFIX + CODE)
    geometry = [0x1003, SECOND]
    body_barrier, = [pc + len(RETIRED_PREFIX) for pc, op in body if op == "s_barrier"]
    # Source schedule: both arrive at 0; wave 1 branches to S_ENDPGM at 28; wave 0
    # alone arrives at 20 and the body barrier, then exports the triangle.
    def arrivals(program):
        waves = [Wave(i, geometry) for i in range(2)]
        for wave in waves:
            wave.run(program, {})
        return [wave.barriers for wave in waves]

    assert arrivals(decoded) == [[0, 20, body_barrier], [0]]
    assert lower_static(decoded, geometry, MESH_LIMITS) == lower_static(body, geometry, MESH_LIMITS)
    assert lower(decoded, 2, MESH_LIMITS) == lower(body, 2, MESH_LIMITS)

    # Replace wave 1's branch to the exit with a barrier. Both traces now have the same
    # length, but the second rendezvous has two live waves at different PCs.
    assert dict(decoded)[28].startswith("s_branch ")
    divergent = [(pc, "s_barrier" if pc == 28 else op) for pc, op in decoded]
    assert arrivals(divergent) == [[0, 20, body_barrier], [0, 28, body_barrier]]
    # Wave 1 can also retire at the allocation skip, after the conflicting
    # rendezvous. The shorter trace must still agree on the common phases.
    exit_pc = decoded[-1][0]
    short = [(pc, f"s_cbranch_execz {(exit_pc - pc - 4) // 4}" if op.startswith("s_cbranch_execz ") else op)
             for pc, op in divergent]
    assert arrivals(short) == [[0, 20, body_barrier], [0, 28]]
    for program in (divergent, short):
        for lowering, binding in ((lower_static, geometry), (lower, 2)):
            try:
                lowering(program, binding, MESH_LIMITS)
            except ValueError as error:
                assert "Inconsistent mesh synchronization" in str(error), error
            else:
                raise AssertionError("Live waves at different barrier PCs were accepted")

    wide = decode(surviving_wave_code())
    for order in ((0, 1), (1, 0)):
        waves = [Wave(i, [WIDE_COUNTS, WIDE_SECOND]) for i in range(2)]
        for i in order:
            waves[i].run(wide, {})
        first, survivor = waves
        assert first.barriers == [172] and survivor.barriers == [172, 184]
        assert first.counts == [(68, 2)] and first.indices[1] == [2, 1, 67]
        assert set(first.positions) == set(range(64))
        assert set(survivor.positions) == {64, 65, 66, 67}
        assert tuple(survivor.positions[67]) == words((1, 1, 0, 1))


def insert_scalar(decoded, after_pc, lines, skip_pc=None):
    """Insert four-byte scalar lines after a pc; widen the skip branch that spans them."""
    out = []
    shift = 4 * len(lines)
    for pc, text in decoded:
        if pc == skip_pc:
            text = f"s_cbranch_execz {int(text.split()[1]) + len(lines)}"
        out.append((pc + (shift if pc > after_pc else 0), text))
        if pc == after_pc:
            out.extend((pc + 4 * (i + 1), line) for i, line in enumerate(lines))
    return out


def check_skipped_scalars(decoded):
    # Wave 0 only: the write sits in the range wave 1 skips with its empty EXEC, so wave 1 never defines it.
    skipped = insert_scalar(decoded, 160, ["s_mov_b32 s32, 1", "s_mov_b32 s33, 2"], skip_pc=156)
    # A dead equality compare on a skipped-write scalar leaves SCC undefined instead of refusing the shader.
    dead_compare = insert_scalar(skipped, 172, ["s_add_u32 s34, s32, 1", "s_cmp_eq_u32 s34, 0"])
    lower(dead_compare, 2, MESH_LIMITS)
    # A conditional mask that no live lane reads may be undefined.
    last = skipped[-2][0]
    masked_off = insert_scalar(skipped, last, ["s_mov_b64 exec, 0", "v_cndmask_b32_e64 v9, v9, v9, s[32:33]"])
    lower(masked_off, 2, MESH_LIMITS)


RESTORE = "s_mov_b64 exec, s[22:23]"
# A restore through a moved copy, the 32-bit halves, an OR saveexec and a complemented OR (#1998 item C).
RESTORE_FORMS = (
    ["s_mov_b64 s[42:43], s[22:23]", "s_mov_b64 exec, s[42:43]"],
    ["s_mov_b32 exec_lo, s22", "s_mov_b32 exec_hi, s23"],
    ["s_or_saveexec_b64 s[48:49], s[22:23]"],
    ["s_not_b64 s[44:45], s[22:23]", "s_orn2_b64 exec, exec, s[44:45]"],
)


def with_restore(decoded, lines, only=None):
    # Replace the restores (all, or the one at pc `only`) with `lines`. Later pcs move; no branch crosses a restore.
    result, shift = [], 0
    for pc, text in decoded:
        if text == RESTORE and only in (None, pc):
            result += [(pc + shift + 4 * i, line) for i, line in enumerate(lines)]
            shift += 4 * (len(lines) - 1)
        else:
            result.append((pc + shift, text))
    return result


def check_exec_restore(decoded):
    reference = lower(decoded, 2, MESH_LIMITS)
    for lines in RESTORE_FORMS:
        assert lower(with_restore(decoded, lines), 2, MESH_LIMITS) == reference, lines
    # The mask saved by a saveexec is the one before it, so restoring from it loses lanes, and a half that is not restored leaves lanes off: both refused.
    lost = ["s_or_saveexec_b64 s[48:49], s[22:23]", "s_mov_b64 exec, s[48:49]"]
    for lines in (lost, RESTORE_FORMS[1][1:], ["s_mov_b64 exec, s[52:53]"]):
        expect_refusal(with_restore(decoded, lines), 2, "")


def check():
    check_retired_wave()
    decoded = decode(CODE)
    variants = enumerate_variants(decoded, 2, MESH_LIMITS)
    for name, packed_counts, second_index, vertex_count, triangles in CASES:
        variant = select(variants, (packed_counts, second_index))
        assert variant.counts == (vertex_count, len(triangles)), (name, variant.counts)
        assert variant.indices == {i: list(indices) for i, indices in enumerate(triangles)}, name
        assert {k: tuple(v) for k, v in variant.positions.items()} == {
            i: words((2 * (i % 2) - 1, 2 * (i // 2) - 1, 0, 1)) for i in range(vertex_count)
        }, name
        assert {k: tuple(v) for k, v in variant.attributes.items()} == {
            i: words((i % 2, i // 2, 0.5, 1)) for i in range(vertex_count)
        }, name

    # A different immediate in the source changes the accepted inputs and output.
    original = struct.pack("<I", 0x300402)
    changed = struct.pack("<I", 0x300401)
    assert CODE.count(original) == 1
    varied = decode(CODE.replace(original, changed))
    varied_variants = enumerate_variants(varied, 2, MESH_LIMITS)
    assert select(varied_variants, (0x2004, 0x300402)).counts == (0, 0)
    assert select(varied_variants, (0x2004, 0x300401)).indices[1] == [1, 1, 3]
    assert lower(decoded, 2, MESH_LIMITS) != lower(varied, 2, MESH_LIMITS)

    # An input reaching a non-equality predicate or unguarded export is unsupported.
    expect_refusal([(pc, text.replace("s_cmp_eq_u32 s15", "s_cmp_lt_u32 s15"))
                    for pc, text in decoded], 2, "Unproven runtime geometry word")
    expect_refusal([(pc, text.replace("v_mov_b32_e32 v3, 0", "v_mov_b32_e32 v3, s14"))
                    for pc, text in decoded], 2, "Unproven runtime geometry word")
    expect_refusal([(pc, text.replace("v_cndmask_b32_e64 v9, v9, s14", "v_cndmask_b32_e64 v9, s14, v9"))
                    for pc, text in decoded], 2, "Unproven runtime geometry word")
    expect_refusal([(pc, "s_branch -1" if text == "s_cbranch_scc0 1" else text)
                    for pc, text in decoded], 2, "Expected a forward mesh branch")
    # LLVM 15 prints a negative branch SIMM16 unsigned: s_branch 65535 is the same back-edge.
    expect_refusal([(pc, "s_branch 65535" if text == "s_cbranch_scc0 1" else text)
                    for pc, text in decoded], 2, "Expected a forward mesh branch")
    expect_refusal(decoded, 1, "Geometry read outside binding")
    expect_refusal(decoded, 2, "Conflicting mesh counts", limits=(3, 1))
    with patch("mesh_dynamic_aot.MAX_VARIANTS", 1):
        expect_refusal(decoded, 2, "exceeds 1 input cases")

    check_skipped_scalars(decoded)
    check_exec_restore(decoded)

    hlsl = lower(decoded, 2, MESH_LIMITS)
    assert hlsl.count("SetMeshOutputCounts(") == 1
    assert "ByteAddressBuffer geometryData : register(t0);" in hlsl
    assert "geometryData.Load(0)" in hlsl and "geometryData.Load(4)" in hlsl
    assert "out vertices Vertex vertices[4], out indices uint3 primitives[2]" in hlsl
    print("Runtime mesh specializations matched the CPU cases and rejected unproven uses")


if __name__ == "__main__":
    check()
