"""Check the gfx1013 2D RGBA32 FLOAT same-texel UAV load and store."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from compute_store_check import aot, assemble, disassemble, rejects, store_container, variant

HERE = Path(__file__).resolve().parent
LINES = (HERE / "compute_mimg_image_uav_load.s").read_text(encoding="ascii").splitlines()
GROUP = (4, 2, 1)
GROUPS = 2
BINDING = dict(table_reg=7, group_reg=10, group_reg_y=11, image_descriptor=0x40,
               image_format="R32G32B32A32_FLOAT", image_uav_mip=0)
LOAD = next(line for line in LINES if line.startswith("image_load"))
STORE = next(line for line in LINES if line.startswith("image_store"))
WAIT = "s_waitcnt vmcnt(0)"
ADDRESS_X, ADDRESS_Y = (line for line in LINES if line.startswith("v_lshl_add_u32"))
READ = "asuint(outputData[uint2(tid.x, tid.y)])"
WORDS = bytes.fromhex(  # Little-endian instruction words from LLVM 15.0.7.
    "07038cbe 80038dbe 06060cf4 400000fa 080046d7 0a040104 090046d7 0b020504"
    " 7fc08cbf 080f00f0 080e0600 703f8cbf 0f03247e 1003267e 1103287e 0e032a7e"
    " 080f20f0 08120600 000081bf"
)
UAV_LOAD_FLAG = 1 << 21  # tests/compute_check.cpp texture_uav_load_flag


def controls():
    source = aot.lower(LINES, GROUP, **BINDING)
    assert "RWTexture2D<float4> outputData : register(u0);" in source, source
    assert "register(t0)" not in source, source
    read = f"uint4 image_load_0 = {READ};"
    write = ("outputData[uint2(tid.x, tid.y)] = "
             "asfloat(uint4(image_load_0.y, image_load_0.z, image_load_0.w, image_load_0.x));")
    assert source.count(read) == source.count(write) == 1, source
    assert source.index(read) < source.index(write), source

    load_index = LINES.index(LOAD)
    prologue, read_and_store = LINES[:load_index], LINES[load_index:]
    # Same address VGPRs, but X is one texel higher at the read than at the write.
    neighbour = prologue + ["v_add_nc_u32_e32 v8, 1, v8", LOAD, WAIT, ADDRESS_X] + read_and_store[2:]
    same_address = "Image UAV load and store must use the same coordinate VGPRs and values"
    rejects(same_address, aot.lower, neighbour, GROUP, **BINDING)
    # Identical values in different registers still do not meet this profile's coordinate rule.
    copies = ["v_mov_b32_e32 v10, v8", "v_mov_b32_e32 v11, v9"]
    different_regs = prologue + copies + variant(read_and_store, (LOAD, LOAD.replace("v[8:9]", "v[10:11]")))
    rejects(same_address, aot.lower, different_regs, GROUP, **BINDING)
    unique = "Image UAV load/store requires unique dispatch X/Y coordinates and local Z size 1"
    shared_x = variant(LINES, (ADDRESS_X, "v_mov_b32_e32 v8, 0"))
    rejects(unique, aot.lower, shared_x, GROUP, **BINDING)
    rejects(unique, aot.lower, LINES, (4, 2, 2), **BINDING)
    store_data = [f"v_mov_b32_e32 v{reg}, 0x3f800000" for reg in range(18, 22)]
    rejects("Image UAV load must precede the image store", aot.lower,
            prologue + store_data + [STORE, LOAD, WAIT, "s_endpgm"], GROUP, **BINDING)
    rejects("Only one image UAV load is supported", aot.lower,
            prologue + [LOAD, WAIT] + read_and_store, GROUP, **BINDING)
    rejects("Expected at least one observable store", aot.lower,
            variant(LINES, (STORE, "")), GROUP, **BINDING)
    rejects("before s_waitcnt vmcnt(0)", aot.lower, variant(LINES, (WAIT, "")), GROUP, **BINDING)
    rejects("image UAV load requires full-width mip-zero access", aot.lower,
            LINES, GROUP, **(BINDING | {"image_uav_mip": 1}))

    explicit_mip = LOAD.replace("image_load", "image_load_mip").replace("v[8:9]", "v[8:10]")
    for changed, diagnostic in (
        (LOAD.replace("dmask:0xf", "dmask:0x5").replace("v[14:17]", "v[14:15]"),
         "image UAV load requires full DMASK 0xf"),
        (LOAD.replace("v[14:17]", "v[14:15]") + " d16", "Unsupported compute instruction"),
        (LOAD.replace("v[8:9]", "v8") + " a16", "image UAV load requires full-width mip-zero access"),
        (explicit_mip, "image_load_mip requires a separate SRV"),
        (LOAD.replace("IMG_2D", "IMG_2D_ARRAY").replace("v[8:9]", "v[8:10]"), "Unsupported compute instruction"),
        (LOAD.replace("IMG_2D", "IMG_3D").replace("v[8:9]", "v[8:10]"), "Unsupported compute instruction"),
    ):
        rejects(diagnostic, aot.lower, variant(LINES, (LOAD, changed)), GROUP, **BINDING)
    rejects("overlapping image SRV descriptor", aot.lower,
            LINES, GROUP, **(BINDING | {"image_srv_descriptor": 0x40}))
    # A declared SRV at a distinct offset cannot make an explicit-mip load
    # from the UAV's SGPRs legal, even after a valid same-texel read.
    mixed_load = explicit_mip.replace("v[14:17]", "v[22:25]")
    mixed = prologue + [LOAD, WAIT, "v_mov_b32_e32 v10, 0", mixed_load, WAIT] + read_and_store[2:]
    rejects("Unverified image descriptor", aot.lower, mixed, GROUP, **(BINDING | {"image_srv_descriptor": 0x60}))
    print("Same-texel UAV load translation controls passed", flush=True)
    return source


def binary(directory, assembler, objcopy, library, source):
    code = assemble(directory, assembler, objcopy, LINES)
    assert code == WORDS, f"assembled {code.hex()} != recorded {WORDS.hex()}"
    print("Same-texel gfx1013 machine words match llvm-mc", flush=True)
    if not library:
        return
    decoded = aot.labelled(disassemble(code, library, offsets=True, cpu=b"gfx1013"))
    assert [line for line in decoded if line.startswith("image_")] == [LOAD, STORE]
    assert aot.lower(decoded, GROUP, **BINDING) == source
    container, output = directory / "image-uav-load.dxbc", directory / "cli.hlsl"
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
                texel = [0x3f800000 + (mip << 16) + 4 * (y * width + x) + c for c in range(4)]
                if mip == 0 and x < GROUP[0] * GROUPS and y < GROUP[1]:
                    initial.extend(texel)
                    expected.extend((*texel[1:], texel[0]))
                else:
                    sentinels = [word | 0x80000000 for word in texel]
                    initial.extend(sentinels)
                    expected.extend(sentinels)
    assert len(initial) == len(expected) == 1344
    assert sum(a != b for a, b in zip(initial, expected, strict=True)) == 64
    return initial + expected


def draw(directory, compiler, renderer, adapter, source):
    hlsl, cso, case = (directory / f"image-uav-load.{ext}" for ext in ("hlsl", "cso", "case"))
    payload = reference()
    case.write_bytes(struct.pack(f"<2I{len(payload)}I", GROUPS | UAV_LOAD_FLAG, 0, *payload))
    without_read = source.replace(READ, "uint4(0u, 0u, 0u, 0u)")
    assert without_read != source
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("image_load\nimage_store\n", encoding="ascii")
    for label, code, mismatches in (("translated", source, 0), ("drop UAV read", without_read, 64)):
        hlsl.write_text(code, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)
        result = subprocess.run([renderer, adapter, "--compute-texture-store-case", cso, case],
                                capture_output=True, text=True, timeout=15)
        print(f"{label}:\n{result.stdout}{result.stderr}", end="", flush=True)
        if result.returncode == 77:
            return 77
        if "debug_layer=enabled" in result.stdout:
            assert "debug_errors=0 debug_warnings=0" in result.stdout, label
        assert f"compared_words=1344 mismatches={mismatches}" in result.stdout, label
        assert result.returncode == int(mismatches != 0), label
        if mismatches:
            assert "mip=0 word=0 expected=3f800001 actual=00000000" in result.stderr, label
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
    with tempfile.TemporaryDirectory(prefix="compute_image_uav_load_", dir=out) as tmp:
        directory = Path(tmp)
        if args.binary:
            binary(directory, *args.binary, args.llvm, source)
            return 0
        return draw(directory, *args.draw, source)


if __name__ == "__main__":
    raise SystemExit(main())
