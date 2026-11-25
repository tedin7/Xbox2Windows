"""A mesh descriptor chosen by a constant-buffer predicate and joined after two arms.

The source-built program loads a selector word, branches on it and loads a
different descriptor table in each arm (a per-material view, or a null
descriptor). After the join it reads the depth through the joined descriptor.
Without arguments the Python lowering and its refusals are checked; with
`<dxc> <renderer> <adapter>` the lowered mesh shader is drawn on D3D12.
"""

from pathlib import Path
import struct
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mesh_entry_aot import lower
from mesh_entry_check import PROGRAM, TRIANGLE, draw, enumerate_pcs

GEOMETRY = [0x1003]
POINT, MATRIX = "0x60", "0x80"


def program(arm_true, arm_false, selector="s_cmp_eq_u32 s14, 0"):
    """Splice a uniform if/else that defines the descriptor s[36:39]."""
    first = arm_true + ["s_branch %d" % len(arm_false)]
    return (
        PROGRAM[:7]  # descriptor tables, wait, geometry word s15
        + [
            "s_buffer_load_dword s14, s[16:19], null",
            "s_waitcnt lgkmcnt(0)",
            selector,
            "s_cbranch_scc0 %d" % len(first),
        ]
        + first
        + arm_false
        + [
            "s_buffer_load_dword s30, s[36:39], 0x4",
            "s_buffer_load_dword s31, s[20:23], 0xc",
            "s_waitcnt lgkmcnt(0)",
        ]
        + PROGRAM[10:]
    )


def table(offset):
    return [f"s_load_dwordx4 s[36:39], s[10:11], {offset}", "s_waitcnt lgkmcnt(0)"]


NULL = [f"s_mov_b32 s{i}, 0" for i in range(36, 40)]
PER_MATERIAL = program(table(POINT), table(MATRIX))
NULL_ARM = program(table(POINT), NULL)


def build(lines, **kw):
    return lower(enumerate_pcs(lines), GEOMETRY, (3, 1), **kw)


def refused(lines, text):
    try:
        build(lines)
    except ValueError as error:
        if text not in str(error):
            raise AssertionError(f"Wrong refusal for {text!r}: {error}")
        return
    raise AssertionError(f"Expected refusal: {text}")


def checks():
    source = build(PER_MATERIAL)
    expect = "(asuint(pointData[0].x) == 0u) ? pointData[0].y : matrixData[0].y"
    assert expect in source, source
    assert "(asuint(pointData[0].x) == 0u) ? pointData[0].y : 0.0f" in build(NULL_ARM)
    # LLVM 15 prints a branch to itself (SIMM16 = -1) as 65535.
    split = PER_MATERIAL.copy()
    split[split.index("s_cbranch_scc0 3")] = "s_cbranch_scc0 65535"
    refused(split, "Expected a forward mesh branch")

    arm = PER_MATERIAL.copy()
    latch = arm.index("s_branch 2")
    arm[latch] = "s_branch 65535"
    # Give the *unsigned* destination an instruction. Otherwise a missing
    # destination and a backward jump both produce the expected refusal.
    unsigned_target = latch + 1 + 65535
    arm.extend(["s_nop 0"] * (unsigned_target - len(arm)))
    arm.append("s_endpgm")
    refused(arm, "Expected a forward mesh branch")
    # Negative controls: precisely what the join does not cover.
    refused(program(table(POINT), table("0x20")), "Unsupported descriptor in a join")
    refused(
        program(table(POINT) + ["v_mov_b32_e32 v7, 0"], table(MATRIX)),
        "Unsupported instruction inside a descriptor join arm",
    )
    refused(
        program(table(POINT) + ["s_mov_b64 exec, 1"], table(MATRIX)),
        "changed EXEC, VCC or M0",
    )
    # Changes in the two arms must not cancel each other out.
    refused(
        program(table(POINT) + ["s_mov_b64 exec, 1"], table(MATRIX) + ["s_mov_b64 exec, -1"]),
        "changed EXEC, VCC or M0",
    )
    # A vector load waited for in only one arm is still pending after the join.
    lines = program(table(POINT) + ["s_waitcnt vmcnt(0)"], table(MATRIX))
    start = lines.index("s_buffer_load_dword s14, s[16:19], null")
    lines[start:start] = [
        "v_mov_b32_e32 v10, 0",
        "buffer_load_dword v10, v10, s[24:27], 0 offen",
    ]
    use = lines.index("s_buffer_load_dword s30, s[36:39], 0x4")
    lines[use:use] = ["v_readfirstlane_b32 s9, v10"]
    refused(lines, "Vector load used before wait")
    refused(
        program(table(POINT), ["s_branch 0"] + table(MATRIX)),
        "Unsupported instruction inside a descriptor join arm",
    )
    # A descriptor joined from two arms must not be consumed as a geometry table.
    refused(
        [
            line.replace("s_buffer_load_dword s30, s[36:39], 0x4", "s_mov_b32 s30, s36")
            for line in PER_MATERIAL
        ],
        "Scalar ALU requires concrete operands",
    )
    # The same program without a join (one concrete descriptor) still lowers.
    assert build(program(table(POINT), table(POINT)))
    print("mesh join descriptor lowering and refusals: ok")


def constants(selector, depth_point, depth_matrix, matrix_w):
    data = bytearray(1024)
    struct.pack_into("<f", data, 0, selector)
    struct.pack_into("<f", data, 4, depth_point)
    struct.pack_into("<f", data, 512 + 4, depth_matrix)
    struct.pack_into("<f", data, 512 + 12, matrix_w)
    return data


def cases(null_arm):
    def triangle(z):
        return [(x, y, z, w) for x, y, _, w in TRIANGLE]

    # selector 0.0 takes the fall-through arm (point table); 1.0 the other arm.
    other = 0.0 if null_arm else 0.25
    return [
        (constants(0.0, 0.125, 0.25, 0.375), triangle(0.5), True),
        (constants(1.0, 0.125, 0.25, 0.375), triangle(other + 0.375), True),
    ]


def check(dxc, renderer, adapter):
    for lines, null_arm in ((PER_MATERIAL, False), (NULL_ARM, True)):
        with tempfile.TemporaryDirectory() as directory:
            result = draw(dxc, renderer, adapter, Path(directory), build(lines), cases(null_arm))
            if result == 77:
                return 77
    print("mesh join descriptor draw: ok")
    return 0


if __name__ == "__main__":
    checks()
    sys.exit(check(*sys.argv[1:]) if len(sys.argv) == 4 else 0)
