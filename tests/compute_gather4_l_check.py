"""Check gfx1013 2D image_gather4_l on an RGBA32 FLOAT SRV with a constant sampler and a known integer LOD."""
import argparse
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from compute_store_check import aot, assemble, disassemble, rejects, store_container, variant

HERE = Path(__file__).resolve().parent
GROUP = (8, 8, 1)
GROUPS = 1
BINDING = dict(table_reg=7, descriptor_offset=0x30, group_reg=10, group_reg_y=11, stride=128,
               image_descriptor=0x40, image_format="R32G32B32A32_FLOAT", cpu="gfx1013", gather_mips=3)
# Sampler words of the pixel_aot gather profile: normalized clamp-to-edge, MIN_LOD 0, MAX_LOD 15.996,
# bilinear XY, point mip, no LOD bias.
SAMPLER = (0x12, 0xFFF000, 0x4500000, 0)
COORDINATE_SCALE, COORDINATE_BIAS = 0x3E400000, 0x3E19999A  # 0.1875 and 0.15 as binary32
LODS = {0: "0", 1: "1.0", 2: "2.0"}


def program(lod="1.0", dmask=0x2, u_bits=None):
    """U and V are 0.1875 * x - 0.15 and 0.1875 * y - 0.15, built from the dispatch-thread ID."""
    lines = [
        "s_mov_b32 s12, s7", "s_mov_b32 s13, 0",
        "s_load_dwordx4 s[20:23], s[12:13], 0x30",
        "s_load_dwordx8 s[24:31], s[12:13], 0x40",
        f"s_mov_b32 s32, {SAMPLER[0]}", f"s_mov_b32 s33, {SAMPLER[1]:#x}",
        f"s_mov_b32 s34, {SAMPLER[2]:#x}", f"s_mov_b32 s35, {SAMPLER[3]}",
        "v_lshl_add_u32 v8, s10, 3, v0",
        "v_lshl_add_u32 v9, s11, 3, v1",
        "s_mov_b32 s36, 0x41000000",
    ]
    for reg, tid in ((3, 8), (4, 9)):
        # tid * 2^20 + 8.0 is the binary32 8 + tid for tid < 8; subtracting 8.0 leaves tid exactly.
        lines += [f"v_mad_u32_u24 v{reg}, v{tid}, 0x100000, s36",
                  f"v_subrev_f32_e32 v{reg}, s36, v{reg}",
                  f"v_mul_f32_e32 v{reg}, {COORDINATE_SCALE:#x}, v{reg}",
                  f"v_subrev_f32_e32 v{reg}, {COORDINATE_BIAS:#x}, v{reg}"]
    if u_bits is not None:
        lines.append(f"v_or_b32_e32 v3, {u_bits:#x}, v3")
    lines += [
        f"v_mov_b32_e32 v5, {lod}",
        "s_waitcnt lgkmcnt(0)",
        f"image_gather4_l v[14:17], v[3:5], s[24:31], s[32:35] dmask:{dmask:#x} dim:SQ_RSRC_IMG_2D",
        "v_mov_b32_e32 v12, v9",
        "v_lshlrev_b32_e32 v13, 4, v8",
        "s_waitcnt vmcnt(0)",
        "buffer_store_dwordx4 v[14:17], v[12:13], s[20:23], 0 idxen offen",
        "s_endpgm",
    ]
    return lines


LINES = program()
GATHER = next(line for line in LINES if line.startswith("image_gather4_l"))
NAN_BITS = 0x7FC00000
# (name, LOD, DMASK, first resource mip of the view, visible mips, U override bits)
CASES = (
    ("lod0_red", 0, 0x1, 0, 3, None),
    ("lod1_green", 1, 0x2, 0, 3, None),
    ("lod2_blue", 2, 0x4, 0, 3, None),
    ("view1_lod0_alpha", 0, 0x8, 1, 2, None),
    ("view1_lod1_red", 1, 0x1, 1, 2, None),
    ("nan_u", 1, 0x2, 0, 3, NAN_BITS),
)
SENTINEL = 0xC5C5C5C5
FLOAT_TEXTURE_FLAG = 1 << 30  # tests/compute_check.cpp texture_rgba32_float_flag
VIEW_MIP1_FLAG = 1 << 31  # tests/compute_check.cpp texture_view_mip1_flag
WIDTH, HEIGHT = 32, 8


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def word_float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def texel(mip, x, y, channel):
    # Distinct normal binary32 values for every mip, coordinate and channel.
    return 0x3F000000 + mip * 0x100000 + y * 0x1000 + x * 0x20 + channel * 2


def texels():
    return [texel(mip, x, y, channel) for mip in range(3) for y in range(HEIGHT >> mip)
            for x in range(WIDTH >> mip) for channel in range(4)]


def coordinate(tid):
    return f32(f32(tid * word_float(COORDINATE_SCALE)) - word_float(COORDINATE_BIAS))


# Tap offsets in gather result order (x, y, z, w), RDNA 2 and Direct3D alike.
ORDER = ((0, 1), (1, 1), (1, 0), (0, 0))


def reference(lod, dmask, first, u_bits=None, *, order=ORDER, mip_shift=0):
    """Independent CPU model of the readback; order and mip_shift build the mutated expectations."""
    channel = (1, 2, 4, 8).index(dmask)
    level = first + lod + mip_shift
    width, height = WIDTH >> level, HEIGHT >> level
    words = [SENTINEL] * 320
    for y in range(8):
        for x in range(8 * GROUPS):
            u, v = coordinate(x), coordinate(y)
            if u_bits is not None:
                u = word_float(bits(u) | u_bits)
            if not (math.isfinite(u) and math.isfinite(v)):
                continue
            origin = []
            for value, size in ((u, width), (v, height)):
                scaled = f32(f32(value * size) - 0.5)
                # Keep every coordinate well away from a tap boundary, where texture-unit precision could differ.
                assert abs(scaled - round(scaled)) > 1 / 64, (x, y, level, scaled)
                origin.append(math.floor(scaled))
            for index, (dx, dy) in enumerate(order):
                tx = min(max(origin[0] + dx, 0), width - 1)
                ty = min(max(origin[1] + dy, 0), height - 1)
                words[y * 32 + x * 4 + index] = texel(level, tx, ty, channel)
    return words


def case_binding(mips):
    return BINDING | {"gather_mips": mips}


def case_lines(lod, dmask, u_bits):
    return program(LODS[lod], dmask, u_bits)


def controls():
    source = aot.lower(LINES, GROUP, **BINDING)
    assert "Texture2D<float4> imageData : register(t0);" in source, source
    assert "RWByteAddressBuffer outputData : register(u0);" in source, source
    assert "uint4 image_gather_0 = image_gather_2d(image_gather_0_uv, 1u, 1u);" in source, source
    assert "    if (!(all(isfinite(image_gather_0_uv)))) return;" in source, source
    assert source.count("imageData.Load(") == 4 and source.count("image_gather_2d(") == 2, source
    store = "outputData.Store4(tid.y * 128u + (tid.x << 4u) + 0u, uint4(image_gather_0.x, " \
            "image_gather_0.y, image_gather_0.z, image_gather_0.w));"
    assert store in source, source
    for name, lod, dmask, first, mips, u_bits in CASES:
        text = aot.lower(case_lines(lod, dmask, u_bits), GROUP, **case_binding(mips))
        component = (1, 2, 4, 8).index(dmask)
        assert f"image_gather_2d(image_gather_0_uv, {lod}u, {component}u);" in text, name
    # Only the explicit-LOD, non-comparison, offset-free 2D form is lowered.
    for form in ("image_gather4", "image_gather4_lz", "image_gather4_b", "image_gather4_cl", "image_gather4_c_l",
                 "image_gather4_l_o", "image_gather4_c_lz", "image_gather4_b_cl"):
        rejects(f"Unsupported gather form {form}", aot.lower,
                variant(LINES, ("image_gather4_l ", form + " ")), GROUP, **BINDING)
    for old, new in (
        ("dim:SQ_RSRC_IMG_2D", "dim:SQ_RSRC_IMG_CUBE"), ("dim:SQ_RSRC_IMG_2D", "dim:SQ_RSRC_IMG_2D_ARRAY"),
        ("dim:SQ_RSRC_IMG_2D", "dim:SQ_RSRC_IMG_1D"), ("dim:SQ_RSRC_IMG_2D", "dim:SQ_RSRC_IMG_3D"),
        ("dim:SQ_RSRC_IMG_2D", "dim:SQ_RSRC_IMG_2D unorm"), ("dim:SQ_RSRC_IMG_2D", "dim:SQ_RSRC_IMG_2D a16"),
        ("dim:SQ_RSRC_IMG_2D", "dim:SQ_RSRC_IMG_2D d16"), ("dim:SQ_RSRC_IMG_2D", "dim:SQ_RSRC_IMG_2D glc"),
        ("v[3:5]", "[v3, v4, v5]"),
    ):
        rejects("Unsupported image_gather4_l form", aot.lower, variant(LINES, (old, new)), GROUP, **BINDING)
    for old, new, message in (
        ("dmask:0x2", "dmask:0x3", "one-hot DMASK"),
        ("dmask:0x2", "dmask:0xf", "one-hot DMASK"),
        ("image_gather4_l v[14:17]", "image_gather4_l v[14:16]", "Invalid image_gather4_l register range"),
        ("v[3:5]", "v[3:4]", "Invalid image_gather4_l register range"),
        ("v[3:5]", "v[3:6]", "Invalid image_gather4_l register range"),
        ("v_mov_b32_e32 v5, 1.0", "v_mov_b32_e32 v5, 0.5", "known integer LOD inside the view"),
        ("v_mov_b32_e32 v5, 1.0", "v_mov_b32_e32 v5, 0x40400000", "known integer LOD inside the view"),
        ("v_mov_b32_e32 v5, 1.0", "v_mov_b32_e32 v5, -1.0", "known integer LOD inside the view"),
        ("v_mov_b32_e32 v5, 1.0", "v_mov_b32_e32 v5, 0x7fc00000", "known integer LOD inside the view"),
        ("v_mov_b32_e32 v5, 1.0", "v_mov_b32_e32 v5, v3", "known integer LOD inside the view"),
        ("v_mov_b32_e32 v5, 1.0", "v_mov_b32_e32 v5, 3.0", "Invalid literal move"),
        ("v_mov_b32_e32 v5, 1.0", "", "known integer LOD inside the view"),
        ("v[3:5]", "v[6:8]", "Unverified image coordinates"),
        ("s_mov_b32 s32, 18", "s_mov_b32 s32, 0", "Unsupported gather sampler"),
        ("s_mov_b32 s33, 0xfff000", "s_mov_b32 s33, 0xfff001", "Unsupported gather sampler"),
        ("s_mov_b32 s33, 0xfff000", "s_mov_b32 s33, 0x100000", "Unsupported gather sampler"),
        ("s_mov_b32 s34, 0x4500000", "s_mov_b32 s34, 0x5500000", "Unsupported gather sampler"),
        ("s_mov_b32 s34, 0x4500000", "s_mov_b32 s34, 0x4500001", "Unsupported gather sampler"),
        ("s_mov_b32 s35, 0", "s_mov_b32 s35, 1", "Unsupported gather sampler"),
        ("s_mov_b32 s35, 0", "", "constant sampler"),
        ("s[32:35] dmask", "s[33:36] dmask", "constant sampler"),
        ("s[32:35] dmask", "s[32:34] dmask", "constant sampler"),
        ("s[32:35] dmask", "s[24:27] dmask", "constant sampler"),
        ("s[24:31], s[32:35]", "s[26:33], s[32:35]", "Unresolved image descriptor"),
        ("s[24:31], s[32:35]", "s[20:27], s[32:35]", "Unverified image descriptor"),
        ("s[24:31], s[32:35]", "s[28:35], s[32:35]", "Unverified image descriptor"),
        ("s_waitcnt lgkmcnt(0)", "", "Unresolved image descriptor"),
        ("s_waitcnt vmcnt(0)", "", "before s_waitcnt vmcnt(0)"),
        ("s_waitcnt lgkmcnt(0)", "s_waitcnt lgkmcnt(0)\nv_cmp_gt_u32_e32 vcc_lo, 4, v8\ns_and_saveexec_b32 s40, vcc_lo",
         "requires an unguarded lane"),
        ("v_mov_b32_e32 v12, v9", "v_mov_b32_e32 v12, v9\n" + GATHER.replace("v[14:17]", "v[20:23]"),
         "Expected one image_gather4_l"),
    ):
        rejects(message, aot.lower, variant(LINES, (old, new)), GROUP, **BINDING)
    # The declared view needs the gfx1013 RGBA32 FLOAT SRV at t0 and 1 to 16 visible mips.
    message = "Gather requires a gfx1013 RGBA32 FLOAT image SRV at t0 with 1 to 16 visible mips"
    for change in ({"image_format": "R32G32B32A32_UINT"}, {"gather_mips": 0},
                   {"gather_mips": 17}, {"gather_mips": True}, {"gather_mips": 2.0}):
        rejects(message, aot.lower, LINES, GROUP, **(BINDING | change))
    # gfx1030 has no RGBA32 FLOAT SRV profile: the image binding itself is refused first.
    rejects("Invalid or overlapping image descriptor binding", aot.lower, LINES, GROUP, **(BINDING | {"cpu": "gfx1030"}))
    rejects(message, aot.lower, LINES, GROUP,
            **(BINDING | {"cpu": "gfx1030", "image_format": "R32G32B32A32_UINT"}))
    output = {key: value for key, value in BINDING.items() if key not in ("descriptor_offset", "stride")}
    rejects(message, aot.lower, LINES, GROUP, **(output | {"image_uav_mip": 0}))
    rejects("image_gather4_l requires an unguarded lane and a declared gather view", aot.lower, LINES, GROUP,
            **{key: value for key, value in BINDING.items() if key != "gather_mips"})
    # LOD 2 is outside a two-mip view.
    rejects("known integer LOD inside the view", aot.lower, case_lines(2, 0x2, None), GROUP,
            **case_binding(2))
    without = variant(LINES, (GATHER, "\n".join(f"v_mov_b32_e32 v{reg}, v8" for reg in range(14, 18))),
                      ("s_waitcnt vmcnt(0)", ""))
    rejects("Expected one image_gather4_l", aot.lower, without, GROUP, **BINDING)
    print("translation controls passed", flush=True)
    return source


def binary(directory, assembler, objcopy, library, source):
    for name, lod, dmask, first, mips, u_bits in CASES:
        lines = case_lines(lod, dmask, u_bits)
        code = assemble(directory, assembler, objcopy, lines)
        expected = aot.lower(lines, GROUP, **case_binding(mips))
        if not library:
            continue
        decoded = aot.labelled(disassemble(code, library, offsets=True, cpu=b"gfx1013"))
        assert decoded == lines, (name, decoded)
        assert aot.lower(decoded, GROUP, **case_binding(mips)) == expected, name
    print(f"{len(CASES)} gfx1013 programs assemble with llvm-mc", flush=True)
    if not library:
        return
    print("LLVM 15 disassembly matches the fixture text", flush=True)
    code = assemble(directory, assembler, objcopy, LINES)
    container, output = directory / "gather.dxbc", directory / "cli.hlsl"
    container.write_bytes(store_container(code, GROUP))
    binding = [arg for key, value in BINDING.items() for arg in ("--" + key.replace("_", "-"), str(value))]
    subprocess.run([sys.executable, "-B", aot.__file__, container, "--output", output, "--llvm", library,
                    *binding], check=True)
    assert output.read_text() == source
    print("container CLI matches the text lowering", flush=True)


def first_mismatch(expected, actual):
    word = next(index for index, (want, got) in enumerate(zip(expected, actual)) if want != got)
    return f"word={word} expected={expected[word]:08x} actual={actual[word]:08x}"


def run_case(directory, compiler, renderer, backend, name, hlsl_source, expected, first):
    hlsl, cso, case = (directory / f"{name}.{ext}" for ext in ("hlsl", "cso", "case"))
    payload = expected + texels()
    flags = FLOAT_TEXTURE_FLAG | (VIEW_MIP1_FLAG if first else 0)
    case.write_bytes(struct.pack(f"<I{len(payload)}I", GROUPS | flags, *payload))
    hlsl.write_text(hlsl_source, encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)
    result = subprocess.run([renderer, backend, "--compute-texture-case", cso, case],
                            capture_output=True, text=True, timeout=30)
    print(f"{name}:\n{result.stdout}{result.stderr}", end="", flush=True)
    if "debug_layer=enabled" in result.stdout:
        assert "debug_errors=0 debug_warnings=0" in result.stdout, name
    return result


def draw(directory, compiler, renderer, backend, source):
    for name, lod, dmask, first, mips, u_bits in CASES:
        text = aot.lower(case_lines(lod, dmask, u_bits), GROUP, **case_binding(mips))
        expected = reference(lod, dmask, first, u_bits)
        if u_bits is None:
            assert expected.count(SENTINEL) == 320 - 256, name
        else:
            assert expected == [SENTINEL] * 320, name
        result = run_case(directory, compiler, renderer, backend, name, text, expected, first)
        if result.returncode == 77:
            return 77
        assert result.returncode == 0 and "compared_words=320 mismatches=0" in result.stdout, name
    # Negative controls: each mutation must fail at the word its own CPU model predicts.
    helper = aot.GATHER_HLSL
    # Exchange the first (x) and last (w) taps.
    swapped = (helper.replace("clamp(origin + int2(0, 1), 0, edge)", "@")
               .replace("clamp(origin, 0, edge)", "clamp(origin + int2(0, 1), 0, edge)")
               .replace("@", "clamp(origin, 0, edge)"))
    assert swapped != helper
    good = aot.lower(case_lines(1, 0x2, None), GROUP, **BINDING)
    nan_source = aot.lower(case_lines(1, 0x2, NAN_BITS), GROUP, **BINDING)
    expected = reference(1, 0x2, 0)
    mutants = (
        ("wrong mip", good.replace("image_gather_0_uv, 1u, 1u", "image_gather_0_uv, 0u, 1u"), expected,
         reference(1, 0x2, 0, mip_shift=-1), 0),
        ("swapped gather order", good.replace(helper, swapped), expected,
         reference(1, 0x2, 0, order=(ORDER[3], ORDER[1], ORDER[2], ORDER[0])), 0),
        ("absolute mip on a view", aot.lower(case_lines(1, 0x1, None), GROUP, **case_binding(2)),
         reference(1, 0x1, 1), reference(1, 0x1, 0), 0),
        ("finite guard removed", nan_source.replace("    if (!(all(isfinite(image_gather_0_uv)))) return;\n", ""),
         [SENTINEL] * 320, None, 0),
    )
    for name, mutation, want, model, first in mutants:
        result = run_case(directory, compiler, renderer, backend, name.replace(" ", "_"), mutation, want, first)
        assert result.returncode == 1, name
        if model is not None:
            assert first_mismatch(want, model) in result.stderr, (name, first_mismatch(want, model))
        else:
            assert "word=0 expected=c5c5c5c5" in result.stderr, name
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        opcodes = sorted({line.split()[0] for line in LINES if line.strip()})
        path.write_text("".join(f"{op}\n" for op in opcodes), encoding="utf-8")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--binary", nargs=2, metavar=("LLVM_MC", "LLVM_OBJCOPY"))
    modes.add_argument("--draw", nargs=3, metavar=("DXC", "RENDERER", "ADAPTER"))
    parser.add_argument("--llvm", metavar="LIBLLVM", help="Also check disassembly and the container CLI")
    args = parser.parse_args()
    source = controls()
    if not (args.binary or args.draw):
        return 0
    out = HERE.parent / "out"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compute_gather4_l_", dir=out) as tmp:
        directory = Path(tmp)
        if args.binary:
            binary(directory, *args.binary, args.llvm, source)
            return 0
        return draw(directory, *args.draw, source)


if __name__ == "__main__":
    raise SystemExit(main())
