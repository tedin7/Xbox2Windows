"""Check gfx1013 2D RGBA32 FLOAT image_store and explicit-mip SRV reads."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from compute_store_check import aot, assemble, disassemble, rejects, store_container, variant

HERE = Path(__file__).resolve().parent
LINES = (HERE / "compute_mimg_float_store.s").read_text(encoding="ascii").splitlines()
GROUP = (4, 2, 1)
BINDING = dict(table_reg=7, group_reg=10, group_reg_y=11, image_descriptor=0x40,
               image_srv_descriptor=0x60, image_format="R32G32B32A32_FLOAT", image_uav_mip=0)
LOAD = next(line for line in LINES if line.startswith("image_load_mip"))
STORE = next(line for line in LINES if line.startswith("image_store"))
WORDS = bytes.fromhex(
    "07038cbe 80038dbe 06060cf4 400000fa 06080cf4 600000fa 080046d7 0a040104"
    " 090046d7 0b020504 8102147e 7fc08cbf 080504f0 080e0800 703f8cbf ff02207e 00008140"
    " ff02227e 010081bf 080f20f0 080e0600 000081bf"
)
GROUPS = 2
FLOAT_TEXTURE_FLAG = 1 << 30  # tests/compute_check.cpp texture_rgba32_float_flag
STORE_TAIL = (0x40810000, 0xBF810001)


def controls():
    source = aot.lower(LINES, GROUP, **BINDING)
    assert "Texture2D<float4> imageData : register(t0);" in source, source
    assert "RWTexture2D<float4> outputData : register(u0);" in source, source
    assert "asuint(imageData.Load(int3(asint(tid.x), asint(tid.y), asint(0x00000001u))))" in source, source
    assert ("outputData[uint2(tid.x, tid.y)] = "
            "asfloat(uint4(image_load_0.x, image_load_0.z, 0x40810000u, 0xbf810001u));") in source, source
    sparse_store = STORE.replace("v[14:17]", "v[14:15]").replace("dmask:0xf", "dmask:0x5")
    rejects("image_store requires full DMASK 0xf", aot.lower,
            variant(LINES, (STORE, sparse_store)), GROUP, **BINDING)
    for old, new, message in (
        (STORE, STORE + " d16", "Unsupported compute instruction"),
        (LOAD, LOAD + " d16", "Unsupported compute instruction"),
        (STORE, STORE.replace("image_store", "image_atomic_add"), "Unsupported image atomic target"),
        (LOAD, LOAD.replace("image_load_mip", "image_sample_l"), "Unsupported compute instruction"),
        (LOAD, LOAD.replace("s[32:39]", "s[24:31]"), "Unverified image descriptor"),
        (STORE, STORE.replace("s[24:31]", "s[32:39]"), "Unverified image descriptor"),
        ("s_waitcnt vmcnt(0)", "", "before s_waitcnt vmcnt(0)"),
        ("s_waitcnt lgkmcnt(0)", "", "Unresolved image descriptor"),
    ):
        rejects(message, aot.lower, variant(LINES, (old, new)), GROUP, **BINDING)
    for dimension in ("1D", "2D_ARRAY", "3D"):
        for instruction in (LOAD, STORE):
            changed = instruction.replace("IMG_2D", f"IMG_{dimension}")
            rejects("Unsupported compute instruction", aot.lower,
                    variant(LINES, (instruction, changed)), GROUP, **BINDING)
    # Only image_store (no mip) and image_load_mip are lowered on gfx1013.
    plain_load = LOAD.replace("image_load_mip v[14:15], v[8:10]", "image_load v[14:15], v[8:9]")
    store_mip = STORE.replace("image_store v[14:17], v[8:9]", "image_store_mip v[14:17], v[8:10]")
    rejects("Unsupported image load target", aot.lower, variant(LINES, (LOAD, plain_load)), GROUP, **BINDING)
    rejects("Unsupported image store target", aot.lower, variant(LINES, (STORE, store_mip)), GROUP, **BINDING)
    # The SRV descriptor must be aligned and clear of both 16-byte halves of the UAV descriptor at 0x40.
    for offset in (0x30, 0x34, 0x40, 0x41, 0x50, 0x62):
        rejects("overlapping image SRV descriptor", aot.lower, LINES, GROUP,
                **(BINDING | {"image_srv_descriptor": offset}))
    # The separate SRV needs a gfx1013 RGBA32 FLOAT UAV.
    for change in ({"image_format": "R32G32B32A32_UINT"}, {"cpu": "gfx1030"}):
        rejects("Invalid image UAV binding", aot.lower, LINES, GROUP, **(BINDING | change))
    raw_output = {key: value for key, value in BINDING.items() if key != "image_uav_mip"} | {"descriptor_offset": 0}
    rejects("A separate image SRV requires a gfx1013 RGBA32 FLOAT image UAV", aot.lower, LINES, GROUP, **raw_output)
    print("translation controls passed", flush=True)
    return source


def binary(directory, assembler, objcopy, library, source):
    code = assemble(directory, assembler, objcopy, LINES)
    assert code == WORDS, f"assembled {code.hex()} != recorded {WORDS.hex()}"
    print("gfx1013 machine words match llvm-mc", flush=True)
    if not library:
        return
    decoded = aot.labelled(disassemble(code, library, offsets=True, cpu=b"gfx1013"))
    assert [line for line in decoded if line.startswith("image_")] == [LOAD, STORE]
    assert aot.lower(decoded, GROUP, **BINDING) == source
    container, output = directory / "image.dxbc", directory / "cli.hlsl"
    container.write_bytes(store_container(code, GROUP))
    binding = [arg for key, value in BINDING.items() for arg in ("--" + key.replace("_", "-"), str(value))]
    subprocess.run([sys.executable, "-B", aot.__file__, container, "--output", output, "--llvm", library,
                    *binding], check=True)
    assert output.read_text() == source


def input_word(mip, x, y, channel):
    # Distinct normal binary32 values for all mips, coordinates and channels.
    return 0x3F000000 + mip * 0x100000 + y * 0x1000 + x * 0x20 + channel * 2


def reference():
    initial, expected, texels = [], [], []
    for mip in range(3):
        for y in range(8 >> mip):
            for x in range(32 >> mip):
                words = [input_word(mip, x, y, channel) for channel in range(4)]
                texels.extend(words)
                sentinel = [word | 0x80000000 for word in words]
                initial.extend(sentinel)
                if mip == 0 and x < 4 * GROUPS and y < 2:
                    # The sparse mip 1 load must pack R and B into v14 and v15.
                    # The full store writes those words and the constants in v16/v17.
                    expected.extend((input_word(1, x, y, 0), input_word(1, x, y, 2), *STORE_TAIL))
                else:
                    expected.extend(sentinel)
    return initial + expected + texels


def draw(directory, compiler, renderer, backend, source):
    hlsl, cso, case = (directory / f"image.{ext}" for ext in ("hlsl", "cso", "case"))
    payload = reference()
    case.write_bytes(struct.pack(f"<2I{len(payload)}I", GROUPS | FLOAT_TEXTURE_FLAG, 0, *payload))
    controls = (
        ("translated", source, None),
        ("swap sparse-load channels",
         source.replace("image_load_0.x, image_load_0.z", "image_load_0.z, image_load_0.x"),
         "mip=0 word=0 expected=3f100000 actual=3f100004"),
        ("force mip zero", aot.lower(variant(LINES, ("v_mov_b32_e32 v10, 1", "v_mov_b32_e32 v10, 0")),
                                    GROUP, **BINDING),
         "mip=0 word=0 expected=3f100000 actual=3f000000"),
    )
    for name, hlsl_source, mismatch in controls:
        assert (hlsl_source == source) == (mismatch is None), name
        hlsl.write_text(hlsl_source, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)
        result = subprocess.run([renderer, backend, "--compute-texture-store-case", cso, case],
                                capture_output=True, text=True, timeout=15)
        print(f"{name}:\n{result.stdout}{result.stderr}", end="", flush=True)
        if result.returncode == 77:
            return 77
        if "debug_layer=enabled" in result.stdout:
            assert "debug_errors=0 debug_warnings=0" in result.stdout, name
        if mismatch is None:
            assert result.returncode == 0 and "compared_words=1344 mismatches=0" in result.stdout, name
        else:
            assert result.returncode == 1 and mismatch in result.stderr, name
            assert "compared_words=1344 mismatches=32" in result.stdout, name
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
    with tempfile.TemporaryDirectory(prefix="compute_mimg_float_store_", dir=out) as tmp:
        directory = Path(tmp)
        if args.binary:
            binary(directory, *args.binary, args.llvm, source)
            return 0
        return draw(directory, *args.draw, source)


if __name__ == "__main__":
    raise SystemExit(main())
