"""A gfx1013 formatted store selects the byte range of a fixed u0 through a run-time view record."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import compute_store_check as base


GROUP = (4, 1, 1)
WORDS = 320
TABLE_BYTE = 256
FLOAT1 = 0x01016FAC  # gfx10 format 22 (BUF_FMT_32_FLOAT), identity selectors and RESOURCE_LEVEL set
UINT1 = 0x01014FAC   # gfx10 format 20 (BUF_FMT_32_UINT)
FLOAT2 = 0x01040FAC  # gfx10 format 64 (BUF_FMT_32_32_FLOAT)
FLOAT3 = 0x0104AFAC  # gfx10 format 74 (BUF_FMT_32_32_32_FLOAT)
UINT3 = 0x01048FAC   # gfx10 format 72 (BUF_FMT_32_32_32_UINT)
UINT4 = 0x0104BFAC   # gfx10 format 75, identity XYZW selectors and RESOURCE_LEVEL set
FLOAT4 = 0x0104DFAC
STORE = "buffer_store_format_xyzw v[10:13], v8, s[20:23], 0 idxen offset:4"
D16_STORE = "buffer_store_format_d16_xyzw v[10:11], v8, s[20:23], 0 idxen offset:4"
D16_X_STORE = "buffer_store_format_d16_x v10, v8, s[20:23], 0 idxen offset:4"
SETUP = [
    "s_mov_b32 s12, s7", "s_mov_b32 s13, 0", "s_load_dwordx4 s[20:23], s[12:13], 0x30",
    "s_load_dwordx4 s[16:19], s[12:13], 0x40", "v_lshl_add_u32 v8, s11, 2, v0",
    "s_waitcnt lgkmcnt(0)", "s_buffer_load_dwordx4 s[24:27], s[16:19], null", "s_waitcnt lgkmcnt(0)",
]
PROGRAM = SETUP + [
    "v_mov_b32_e32 v10, v8", "v_mov_b32_e32 v11, 0x12345678", "v_mov_b32_e32 v12, 0x80000001",
    "v_mov_b32_e32 v13, 0x3f800001", STORE, "s_endpgm",
]
D16_PROGRAM = SETUP + [
    "v_xor_b32_e32 v10, 0xc0003c00, v8", "v_mov_b32_e32 v11, 0x80000001", D16_STORE, "s_endpgm",
]
D16_X_PROGRAM = SETUP + [
    "v_xor_b32_e32 v10, s25, v8", D16_X_STORE, "s_endpgm",
]
D16_XY_STORE = D16_X_STORE.replace("d16_x ", "d16_xy ")
D16_XY_PROGRAM = base.variant(D16_X_PROGRAM, (D16_X_STORE, D16_XY_STORE))
D16_XYZ_STORE = D16_STORE.replace("d16_xyzw", "d16_xyz")
D16_XYZ_PROGRAM = base.variant(D16_PROGRAM, (D16_STORE, D16_XYZ_STORE),
                               ("v_mov_b32_e32 v11, 0x80000001", "v_mov_b32_e32 v11, s25"))
# 0x3c00..0x3c03 are consecutive binary16 values starting at 1.0.
D16_X = (0x3f800000, 0x3f802000, 0x3f804000, 0x3f806000)
# High half 0x4200 poisons each word so leaking high bits or missing & 0xffffu shifts fails readback.
D16_X_PAYLOADS = (
    ("positive 1.0..", 0x42003c00, D16_X),
    ("negative -2.0..", 0x4200c000, (0xc0000000, 0xc0002000, 0xc0004000, 0xc0006000)),
    ("min positive subnormal and +0", 0x42000001, (0x33800000, 0x00000000, 0x34400000, 0x34000000)),
    ("-0 and negative subnormals", 0x42008000, (0x80000000, 0xb3800000, 0xb4000000, 0xb4400000)),
)
# V# words 0..3 and live state, then the expected *word addresses* of the four component stores.
# The addresses are literal to keep the reference independent of the generated address calculation.
CASES = (
    ("four records", (12, 24 << 16, 4, UINT4, 1), (4, 10, 16, 22)),
    ("second base and short record count", (116, 32 << 16, 2, UINT4, 1), (30, 38)),
    ("released view", (12, 24 << 16, 4, UINT4, 2), ()),
    ("last complete texel in u0", (492, 20 << 16, 4, UINT4, 1), (124,)),
    ("partial texel at end of u0", (496, 20 << 16, 4, UINT4, 1), ()),
    ("wrong four-channel format", (12, 24 << 16, 4, FLOAT4, 1), ()),
    ("offset plus texel wider than stride", (12, 16 << 16, 4, UINT4, 1), ()),
    ("misaligned base", (14, 24 << 16, 4, UINT4, 1), ()),
    ("misaligned stride", (12, 22 << 16, 4, UINT4, 1), ()),
    ("unsupported high address", (12, (24 << 16) | 1, 4, UINT4, 1), ()),
    ("base past the declared u0 extent", (640, 24 << 16, 4, UINT4, 1), ()),
    ("live slot beyond declared count", (12, 24 << 16, 4, UINT4, 1), ()),
)
# 4-byte FLOAT1 records leaving adjacent dwords at SENTINEL so wider writes fail readback.
D16_X_CASES = (
    ("four float1 records", (12, 8 << 16, 4, FLOAT1, 1), (4, 6, 8, 10)),
    ("second base and short record count", (116, 12 << 16, 2, FLOAT1, 1), (30, 33)),
    ("released view", (12, 8 << 16, 4, FLOAT1, 2), ()),
    ("last complete float1 texel in u0", (504, 8 << 16, 4, FLOAT1, 1), (127,)),
    ("partial float1 texel at end of u0", (508, 8 << 16, 4, FLOAT1, 1), ()),
    ("wrong single-channel UINT1 format", (12, 8 << 16, 4, UINT1, 1), ()),
    ("offset plus 4-byte float1 wider than stride", (12, 4 << 16, 4, FLOAT1, 1), ()),
    ("misaligned base", (14, 8 << 16, 4, FLOAT1, 1), ()),
    ("misaligned stride", (12, 6 << 16, 4, FLOAT1, 1), ()),
    ("unsupported high address", (12, (8 << 16) | 1, 4, FLOAT1, 1), ()),
    ("base past the declared u0 extent", (640, 8 << 16, 4, FLOAT1, 1), ()),
    ("live slot beyond declared count", (12, 8 << 16, 4, FLOAT1, 1), ()),
)
VIEW = (0, TABLE_BYTE, len(CASES) - 1, WORDS * 4, 512)
# Expected starting word of each two-component write. An empty tuple means all 320 guards survive.
D16_XY_CASES = (
    ("four float2 records", (12, 12 << 16, 4, FLOAT2, 1), (4, 7, 10, 13)),
    ("second base and short count", (116, 16 << 16, 2, FLOAT2, 1), (30, 34)),
    ("released view", (12, 12 << 16, 4, FLOAT2, 2), ()),
    ("eight bytes ending at the output limit", (500, 12 << 16, 4, FLOAT2, 1), (126,)),
    ("only four bytes left", (504, 12 << 16, 4, FLOAT2, 1), ()),
    ("wrong FLOAT1 format", (12, 12 << 16, 4, FLOAT1, 1), ()),
    ("short stride", (12, 8 << 16, 4, FLOAT2, 1), ()),
    ("misaligned base", (14, 12 << 16, 4, FLOAT2, 1), ()),
    ("misaligned stride", (12, 14 << 16, 4, FLOAT2, 1), ()),
    ("unsupported high address", (12, (12 << 16) | 1, 4, FLOAT2, 1), ()),
    ("base past the declared extent", (640, 12 << 16, 4, FLOAT2, 1), ()),
    ("live slot beyond the declared count", (12, 12 << 16, 4, FLOAT2, 1), ()),
)
# Each accepted three-dword texel has an adjacent guard, including the one at the declared u0 limit.
D16_XYZ_CASES = (
    ("four float3 records", (12, 16 << 16, 4, FLOAT3, 1), (4, 8, 12, 16)),
    ("second base and short count", (116, 20 << 16, 2, FLOAT3, 1), (30, 35)),
    ("released view", (12, 16 << 16, 4, FLOAT3, 2), ()),
    ("last complete float3", (496, 16 << 16, 4, FLOAT3, 1), (125,)),
    ("only eight bytes left", (500, 16 << 16, 4, FLOAT3, 1), ()),
    ("wrong UINT3 format", (12, 16 << 16, 4, UINT3, 1), ()),
    ("short stride", (12, 12 << 16, 4, FLOAT3, 1), ()),
    ("misaligned base", (14, 16 << 16, 4, FLOAT3, 1), ()),
    ("misaligned stride", (12, 18 << 16, 4, FLOAT3, 1), ()),
    ("unsupported high address", (12, (16 << 16) | 1, 4, FLOAT3, 1), ()),
    ("base past the declared extent", (640, 16 << 16, 4, FLOAT3, 1), ()),
    ("live slot beyond the declared count", (12, 16 << 16, 4, FLOAT3, 1), ()),
)
BINDING = dict(table_reg=7, descriptor_offset=0x30, constant_descriptor=0x40, source_descriptor=0x50, group_reg=11)


def lower(program=PROGRAM, **options):
    return base.aot.lower(program, GROUP, **(BINDING | dict(heap_store_view=VIEW) | options))


def controls():
    assert base.aot.descriptor_format_word(75) == UINT4
    assert base.aot.descriptor_format_word(22) == FLOAT1
    assert base.aot.descriptor_format_word(20) == UINT1
    assert base.aot.descriptor_format_word(64) == FLOAT2
    assert base.aot.descriptor_format_word(74) == FLOAT3
    assert base.aot.descriptor_format_word(72) == UINT3
    # Pin the translator call here. draw() exercises the generated guards against actual UAV contents.
    assert "store_heap_words4(params[0].x, tid.x, 4u, false, uint4(tid.x," in lower()
    assert "store_heap_words4(params[0].x, tid.x, 4u, true," in lower(D16_PROGRAM)
    d16x_hlsl = lower(D16_X_PROGRAM)
    assert "store_heap_word_d16(params[0].x, tid.x, 4u, (params[0].y ^ tid.x));" in d16x_hlsl
    assert f"{FLOAT1:#x}u" in d16x_hlsl
    assert "store_heap_words4(" not in d16x_hlsl and "outputData.Load(" not in d16x_hlsl
    refuses = base.rejects
    refuses("Unsupported formatted buffer store target or address mode", lower, heap_store_view=None)
    refuses("Unsupported formatted buffer store target or address mode", lower, D16_X_PROGRAM, heap_store_view=None)
    refuses("Invalid heap store view binding", lower, D16_X_PROGRAM, stride=8)
    refuses("Invalid heap store view binding", lower, heap_store_view=VIEW, stride=24)
    refuses("Invalid heap store view binding", lower, heap_store_view=VIEW, heap_view=VIEW)
    refuses("Invalid heap view binding", lower, source_descriptor=None)
    refuses("Invalid heap view binding", lower, heap_store_view=(0, TABLE_BYTE, 11, TABLE_BYTE + 32, 512))
    refuses("Invalid heap store view output size", lower, heap_store_view=(*VIEW[:4], 511))
    refuses("Heap store view requires exactly one formatted UAV store", lower,
            PROGRAM[:-1] + [STORE.replace("offset:4", "offset:24"), "s_endpgm"])
    refuses("Heap store view slot constant is not ready", lower, PROGRAM[:7] + PROGRAM[8:])
    refuses("Heap store view slot constant is not ready", lower,
            base.variant(D16_X_PROGRAM, ("s_buffer_load_dwordx4 s[24:27], s[16:19], null",
                                         "s_buffer_load_dword s25, s[16:19], 0x4")))
    refuses("Unverified ALU operand", lower, D16_X_PROGRAM[:7] + D16_X_PROGRAM[8:])
    refuses("Heap store view requires buffer_store_format_xyzw", lower,
            base.variant(PROGRAM, (STORE, STORE.replace("xyzw v[10:13]", "xy v[10:11]"))))
    xyz = lower(D16_XYZ_PROGRAM)
    assert f"{FLOAT3:#x}u" in xyz and "outputData.Store3(" in xyz and "outputData.Load(" not in xyz
    for registers in ("v10", "v[10:12]"):
        refuses("Unverified raw UAV store", lower,
                base.variant(D16_XYZ_PROGRAM, (D16_XYZ_STORE, D16_XYZ_STORE.replace("v[10:11]", registers))))
    refuses("Heap store view requires buffer_store_format_xyzw", lower,
            base.variant(PROGRAM, (STORE, STORE.replace("xyzw v[10:13]", "xyz v[10:12]"))))
    xy = lower(D16_XY_PROGRAM)
    assert f"{FLOAT2:#x}u" in xy and "outputData.Store2(" in xy and "outputData.Load(" not in xy
    refuses("Unverified raw UAV store", lower,
            base.variant(D16_XY_PROGRAM, (D16_XY_STORE, D16_XY_STORE.replace("v10", "v[10:11]"))))
    refuses("Heap store view requires buffer_store_format_xyzw", lower,
            base.variant(D16_X_PROGRAM, (D16_X_STORE, D16_X_STORE.replace("d16_x", "x"))))
    refuses("Unverified raw UAV store", lower,
            base.variant(D16_PROGRAM, (D16_STORE, D16_STORE.replace("v[10:11]", "v[10:13]"))))
    refuses("Unverified raw UAV store", lower,
            base.variant(D16_X_PROGRAM, (D16_X_STORE, D16_X_STORE.replace("v10", "v[10:11]"))))
    refuses("Unverified raw UAV store", lower,
            base.variant(D16_X_PROGRAM, (D16_X_STORE, D16_X_STORE.replace("offset:4", "offset:2"))))
    refuses("Unsupported formatted buffer store target or address mode", lower,
            base.variant(PROGRAM, (STORE, STORE.replace(" idxen", " offen"))))
    refuses("Unsupported formatted buffer store target or address mode", lower,
            base.variant(D16_X_PROGRAM, (D16_X_STORE, D16_X_STORE.replace(" idxen", " offen"))))
    refuses("Unverified raw UAV store", lower, base.variant(PROGRAM, (STORE, STORE.replace("s[20:23]", "s[16:19]"))))
    refuses("Heap store view needs a formatted store indexed by dispatch thread ID", lower,
            base.variant(PROGRAM, (STORE, STORE.replace(", v8,", ", v12,"))))
    refuses("Heap store view needs a formatted store indexed by dispatch thread ID", lower,
            base.variant(D16_X_PROGRAM, (D16_X_STORE, D16_X_STORE.replace(", v8,", ", v12,"))))


def verify_binary(directory, code, library, program=PROGRAM):
    decoded = base.aot.labelled(base.disassemble(code, library, offsets=True, cpu=b"gfx1013"))
    expected = lower(program)
    assert lower(decoded) == expected
    container, output = directory / "store.dxbc", directory / "store.hlsl"
    container.write_bytes(base.store_container(code, GROUP))
    subprocess.run([sys.executable, "-B", base.aot.__file__, container, "--output", output, "--llvm", library,
                    *base.binding_cli(BINDING), "--heap-store-view", *map(str, VIEW)], check=True)
    assert output.read_text(encoding="ascii") == expected


def draw(directory, compiler, renderer, backend, mode):
    program = {"uint4": PROGRAM, "d16_xyzw": D16_PROGRAM, "d16_x": D16_X_PROGRAM,
               "d16_xy": D16_XY_PROGRAM, "d16_xyz": D16_XYZ_PROGRAM}[mode]
    cases = {"d16_x": D16_X_CASES, "d16_xy": D16_XY_CASES, "d16_xyz": D16_XYZ_CASES}.get(mode, CASES)
    source = lower(program)
    table = [0] * WORDS
    for slot, (_, record, _) in enumerate(cases):
        start = TABLE_BYTE // 4 + slot * 8
        table[start:start + 5] = record
        if mode == "d16_xyzw":
            table[start + 3] = FLOAT4 if record[3] == UINT4 else UINT4
    hlsl, cso = directory / "store.hlsl", directory / "store.cso"

    def compile_shader(text):
        hlsl.write_text(text, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)

    default_payload = 0x42000001 if mode == "d16_xyz" else D16_X_PAYLOADS[0][1]

    def run(slot, payload=default_payload, expected_x=D16_X, table_words=table, expected_y=0x40400000,
            expected_z=0x33800000):
        result = [base.SENTINEL] * WORDS
        for thread, start in enumerate(cases[slot][2]):
            if mode == "d16_x":
                result[start] = expected_x[thread]
            elif mode == "d16_xy":
                result[start:start + 2] = [expected_x[thread], expected_y]
            elif mode == "d16_xyz":
                result[start:start + 3] = [D16_X[thread], 0xc0000000, expected_z]
            elif mode == "d16_xyzw":
                # D16 y, z, w are -2, the smallest positive binary16 subnormal, and negative zero.
                result[start:start + 4] = [D16_X[thread], 0xc0000000, 0x33800000, 0x80000000]
            else:
                result[start:start + 4] = [thread, 0x12345678, 0x80000001, 0x3F800001]
        case = directory / "store.case"
        case.write_bytes(struct.pack("<321I", 1, *result) + struct.pack("<16I", slot, payload, *([0] * 14))
                         + struct.pack("<320I", *table_words))
        return subprocess.run([renderer, backend, "--compute-store-case", cso, case], capture_output=True, text=True)

    compile_shader(source)
    for slot, (label, _, _) in enumerate(cases):
        result = run(slot)
        if result.returncode == 77 and backend == "hardware":
            return 77
        assert result.returncode == 0, (mode, label, result.stderr)
    extra = 0
    if mode == "d16_x":
        for label, payload, want in D16_X_PAYLOADS[1:]:
            result = run(0, payload=payload, expected_x=want)
            assert result.returncode == 0, (label, result.stderr)
            extra += 1
        # A FLOAT4 format word must be refused just like UINT1 by store_heap_word_d16.
        wrong_float4 = list(table)
        wrong_float4[TABLE_BYTE // 4 + 3] = FLOAT4
        case = directory / "store.case"
        case.write_bytes(struct.pack("<321I", 1, *([base.SENTINEL] * WORDS)) + struct.pack("<16I", 0, D16_X_PAYLOADS[0][1], *([0] * 14))
                         + struct.pack("<320I", *wrong_float4))
        result = subprocess.run([renderer, backend, "--compute-store-case", cso, case], capture_output=True, text=True)
        assert result.returncode == 0, ("wrong FLOAT4 format", result.stderr)
        extra += 1
    if mode == "d16_xy":
        for payload, expected_y in ((0x80003c00, 0x80000000), (0x00013c00, 0x33800000), (0xc0003c00, 0xc0000000)):
            result = run(0, payload=payload, expected_y=expected_y)
            assert result.returncode == 0, (hex(payload), result.stderr)
            extra += 1
    elif mode == "d16_xyz":
        for payload, z in ((0x7e010001, 0x33800000), (0x3c008000, 0x80000000), (0x4200c000, 0xc0000000)):
            result = run(0, payload=payload, expected_z=z)
            assert result.returncode == 0, (hex(payload), result.stderr)
            extra += 1
    mismatched_formats = {"d16_xy": (FLOAT4, base.aot.descriptor_format_word(62)), "d16_xyz": (FLOAT2, FLOAT4)}
    for wrong_format in mismatched_formats.get(mode, ()):
        wrong_view = table.copy()
        wrong_view[TABLE_BYTE // 4 + 5 * 8 + 3] = wrong_format
        result = run(5, table_words=wrong_view)
        assert result.returncode == 0, (hex(wrong_format), result.stderr)
        extra += 1
    width_str = {"d16_x": "4u", "d16_xy": "8u", "d16_xyz": "12u"}.get(mode, "16u")
    format_guard = {"d16_x": f"{FLOAT1:#x}u", "d16_xy": f"{FLOAT2:#x}u",
                    "d16_xyz": f"{FLOAT3:#x}u"}.get(mode, "formatWord")
    wrong_stride = {"d16_x": 8, "d16_xy": 12, "d16_xyz": 16}.get(mode, 24)
    mutations = (
        ("view.x + index * stride + offset", "index * stride + offset", 0),
        (" || index >= view.z", "", 1),
        ("uint stride = view.y >> 16u;", f"uint stride = {wrong_stride}u;", 1),
        ("const uint bytes = 512u;", "const uint bytes = 1280u;", 4),
        (f"if (index > (bytes - view.x - offset - {width_str}) / stride) return;", "", 3),
        (f"view.w != {format_guard} || ", "", 5),
        ("!= 1u) return;", "== 0u) return;", 2),
        (f"slot >= {VIEW[2]}u", f"slot > {VIEW[2]}u", VIEW[2]),
    )
    if mode == "d16_xyzw":
        mutations += (
            ("value.x >> 16u", "value.x", 0),
            ("value.y >> 16u", "value.x >> 16u", 0),
            ("if (d16) value =", "if (false) value =", 0),
        )
    elif mode == "d16_x":
        mutations += (
            ("value & 0xffffu", "value >> 16u", 0),
            ("asuint(f16tof32(value & 0xffffu))", "value", 0),
            (f"{FLOAT1:#x}u", f"{FLOAT4:#x}u", 0),
        )
    elif mode == "d16_xy":
        mutations += (
            ("uint2(value & 0xffffu, value >> 16u)", "uint2(value >> 16u, value & 0xffffu)", 0,
             "word=4 expected=3f800000 actual=40400000"),
            ("value >> 16u", "value & 0xffffu", 0, "word=5 expected=40400000 actual=3f800000"),
            ("offset + 8u > bytes - view.x", "offset + 4u > bytes - view.x", 4,
             "word=127 expected=c5c5c5c5 actual=3f800000"),
        )
    elif mode == "d16_xyz":
        mutations += (
            ("value.y & 0xffffu", "value.y >> 16u", 0, "word=6 expected=33800000 actual=40400000"),
            ("Store3(view.x + index * stride + offset, asuint(f16tof32(halfWords)))",
             "Store4(view.x + index * stride + offset, uint4(asuint(f16tof32(halfWords)), 0u))", 0,
             "word=7 expected=c5c5c5c5 actual=00000000"),
            ("offset + 12u > bytes - view.x", "offset + 8u > bytes - view.x", 4,
             "word=126 expected=c5c5c5c5 actual=3f800000"),
        )
    for old, new, slot, *first_mismatch in mutations:
        assert source.count(old) == 1, old
        compile_shader(source.replace(old, new))
        result = run(slot)
        assert result.returncode == 1 and "Compute readback differs" in result.stderr, (old, result.stderr)
        if first_mismatch:
            words = [line for line in result.stderr.splitlines() if line.startswith("word=")]
            assert words[:1] == first_mismatch, (old, result.stderr)
    kind = {"uint4": "UINT4", "d16_xyzw": "D16-to-FLOAT4", "d16_x": "D16_X-to-FLOAT1",
            "d16_xy": "D16_XY-to-FLOAT2", "d16_xyz": "D16_XYZ-to-FLOAT3"}[mode]
    print(f"{backend} {kind}: {len(cases) + extra} heap-store cases and {len(mutations)} shader mutations checked")
    return 0


if __name__ == "__main__":
    controls()
    mode, *args = sys.argv[1:] or ["controls"]
    if mode == "controls" and not args:
        print("heap-store controls passed")
    elif mode in ("binary", "draw") and len(args) == 3:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            if mode == "draw":
                for kind in ("uint4", "d16_xyzw", "d16_x", "d16_xy", "d16_xyz"):
                    result = draw(directory, *args, kind)
                    if result:
                        sys.exit(result)
                sys.exit(0)
            assembler, objcopy, library = args
            for program in (PROGRAM, D16_PROGRAM, D16_X_PROGRAM, D16_XY_PROGRAM, D16_XYZ_PROGRAM):
                code = base.assemble(directory, assembler, objcopy, program, cpu="gfx1013")
                verify_binary(directory, code, library, program)
    else:
        sys.exit("usage: compute_heap_store_check.py controls | binary <llvm-mc> <llvm-objcopy> <libLLVM> "
                 "| draw <dxc> <offscreen_check> <backend>")
