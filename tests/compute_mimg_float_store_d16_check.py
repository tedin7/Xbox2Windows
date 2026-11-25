"""Check gfx1013 2D D16 image_store into an RGBA16 FLOAT UAV."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from compute_store_check import aot, assemble, disassemble, rejects, store_container, variant

HERE = Path(__file__).resolve().parent
LINES = (HERE / "compute_mimg_float_store_d16.s").read_text(encoding="ascii").splitlines()
GROUP = (4, 2, 1)
GROUPS = 2
BINDING = dict(table_reg=7, group_reg=10, group_reg_y=11, image_descriptor=0x40,
               image_format="R16G16B16A16_FLOAT", image_uav_mip=0)
STORE = next(line for line in LINES if line.startswith("image_store"))
WORDS = bytes.fromhex(  # Little-endian instruction words from LLVM 15.0.7.
    "07038cbe 80038dbe 06060cf4 400000fa 080046d7 0a040104 090046d7 0b020504"
    " 0a0043d5 88122204 90141634 0b151838 ff181c38 003c00c0 ff181e38 00000080"
    " 7fc08cbf 080f20f0 080e0680 000081bf"
)
FLOAT16_TEXTURE_FLAG = 1 << 22  # tests/compute_check.cpp texture_rgba16_float_flag
COMPONENTS = (
    "f16tof32(image_store_data.x & 0xffffu)",
    "f16tof32(image_store_data.x >> 16u)",
    "f16tof32(image_store_data.y & 0xffffu)",
    "f16tof32(image_store_data.y >> 16u)",
)


def controls():
    source = aot.lower(LINES, GROUP, **BINDING)
    assert "RWTexture2D<float4> outputData : register(u0);" in source, source
    assert "imageData : register(t0)" not in source, source
    assert f"outputData[uint2(tid.x, tid.y)] = float4({', '.join(COMPONENTS)});" in source, source
    for changed, diagnostic in (
        (STORE.replace("v[14:15]", "v14").replace("dmask:0xf", "dmask:0x5"),
         "D16 image_store requires full DMASK 0xf"),
        (STORE.replace("v[14:15]", "v14"), "D16 image_store requires exactly two consecutive data VGPRs"),
        (STORE.replace("v[14:15]", "v[14:16]"), "D16 image_store requires exactly two consecutive data VGPRs"),
        (STORE.replace("v[14:15]", "v[255:256]"), "D16 image_store requires exactly two consecutive data VGPRs"),
        (STORE.replace("image_store", "image_load_mip").replace("v[8:9]", "v[8:10]"),
         "D16 requires a 2D image_store with full-width coordinates and implicit mip zero"),
        (STORE.replace(" d16", " a16 d16").replace("v[8:9]", "v8"),
         "D16 requires a 2D image_store with full-width coordinates and implicit mip zero"),
        (STORE.replace("image_store", "image_store_mip").replace("v[8:9]", "v[8:10]"),
         "D16 requires a 2D image_store with full-width coordinates and implicit mip zero"),
        (STORE.removesuffix(" d16").replace("v[14:15]", "v[14:17]"),
         "RGBA16 FLOAT image UAV requires a D16 image_store"),
    ):
        rejects(diagnostic, aot.lower, variant(LINES, (STORE, changed)), GROUP, **BINDING)
    for dimension, address in (("1D", "v8"), ("2D_ARRAY", "v[8:10]"), ("3D", "v[8:10]")):
        changed = STORE.replace("IMG_2D", f"IMG_{dimension}").replace("v[8:9]", address)
        rejects(f"Unsupported compute instruction: {changed}", aot.lower,
                variant(LINES, (STORE, changed)), GROUP, **BINDING)
    for binding, diagnostic in (
        ({"image_format": "R32G32B32A32_FLOAT"}, "D16 image_store requires a gfx1013 RGBA16 FLOAT UAV"),
        ({"cpu": "gfx1030"}, "RGBA16 FLOAT requires a gfx1013 D16 image store UAV"),
        ({"cpu": "gfx1030", "image_format": "R32G32B32A32_UINT"},
         "D16 image_store requires a gfx1013 RGBA16 FLOAT UAV"),
        ({"image_uav_mip": 1}, "Image store mip does not match the UAV view"),
    ):
        rejects(diagnostic, aot.lower, LINES, GROUP, **(BINDING | binding))
    atom = STORE.replace("image_store v[14:15]", "image_atomic_add v14").replace("dmask:0xf", "dmask:0x1")
    rejects(f"Unsupported compute instruction: {atom}", aot.lower, variant(LINES, (STORE, atom)), GROUP,
            **(BINDING | {"image_format": "R32_UINT"}))
    print("D16 translation controls passed", flush=True)
    return source


def binary(directory, assembler, objcopy, library, source):
    code = assemble(directory, assembler, objcopy, LINES)
    assert code == WORDS, f"assembled {code.hex()} != recorded {WORDS.hex()}"
    print("D16 gfx1013 machine words match llvm-mc", flush=True)
    if not library:
        return
    decoded = aot.labelled(disassemble(code, library, offsets=True, cpu=b"gfx1013"))
    assert [line for line in decoded if line.startswith("image_")] == [STORE]
    assert aot.lower(decoded, GROUP, **BINDING) == source
    container, output = directory / "image-d16.dxbc", directory / "cli.hlsl"
    container.write_bytes(store_container(code, GROUP))
    options = [arg for key, value in BINDING.items() for arg in ("--" + key.replace("_", "-"), str(value))]
    subprocess.run([sys.executable, "-B", aot.__file__, container, "--output", output, "--llvm", library,
                    *options], check=True)
    assert output.read_text() == source


def reference():
    initial, expected = [], []
    for mip in range(3):
        width = 32 >> mip
        for y in range(8 >> mip):
            for x in range(width):
                sentinels = [0x5000 + 0x400 * mip + 4 * (y * width + x) + c for c in range(4)]
                initial.extend(sentinels)
                if mip == 0 and x < GROUP[0] * GROUPS and y < GROUP[1]:
                    # The reference lists individual half encodings directly. At tag
                    # zero the last two channels are +0 and -0, then subnormals of both signs.
                    tag = 8 * y + x
                    expected.extend((0x3c00 + tag, 0xc000 + tag, tag, 0x8000 + tag))
                else:
                    expected.extend(sentinels)
    assert len(initial) == len(expected) == 1344
    assert sum(a != b for a, b in zip(initial, expected, strict=True)) == 64
    return initial + expected


def draw(directory, compiler, renderer, adapter, source):
    original = ", ".join(COMPONENTS)
    cases = (
        ("translated", source, 0, None),
        ("swap low and high halves of VDATA",
         source.replace(original, ", ".join((COMPONENTS[1], COMPONENTS[0], *COMPONENTS[2:]))),
         32, "mip=0 half=0 expected=3c00 actual=c000"),
        ("swap VDATA and VDATA+1",
         source.replace(original, ", ".join((*COMPONENTS[2:], *COMPONENTS[:2]))),
         64, "mip=0 half=0 expected=3c00 actual=0000"),
    )
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("image_store\n", encoding="ascii")
    for name, code, mismatches, _ in cases:
        assert (code == source) == (mismatches == 0), name
    return draw_cases(directory, compiler, renderer, adapter, reference(), cases)


def draw_cases(directory, compiler, renderer, adapter, payload, cases):
    """Compile each shader and compare the raw halves of all three mips."""
    hlsl, cso, case = (directory / f"image-d16.{ext}" for ext in ("hlsl", "cso", "case"))
    case.write_bytes(struct.pack(f"<2I{len(payload)}H", GROUPS | FLOAT16_TEXTURE_FLAG, 0, *payload))
    for name, code, mismatches, first_mismatch in cases:
        hlsl.write_text(code, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)
        result = subprocess.run([renderer, adapter, "--compute-texture-store-case", cso, case],
                                capture_output=True, text=True, timeout=15)
        print(f"{name}:\n{result.stdout}{result.stderr}", end="", flush=True)
        if result.returncode == 77:
            return 77
        if "debug_layer=enabled" in result.stdout:
            assert "debug_errors=0 debug_warnings=0" in result.stdout, name
        assert f"compared_halves=1344 mismatches={mismatches}" in result.stdout, name
        assert result.returncode == int(mismatches != 0), name
        if first_mismatch:
            assert first_mismatch in result.stderr, name
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
    with tempfile.TemporaryDirectory(prefix="compute_mimg_d16_", dir=out) as tmp:
        directory = Path(tmp)
        if args.binary:
            binary(directory, *args.binary, args.llvm, source)
            return 0
        return draw(directory, *args.draw, source)


if __name__ == "__main__":
    raise SystemExit(main())
