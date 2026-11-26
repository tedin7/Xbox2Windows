"""Publish placed Texture3D SRVs by address, sample slices and mips in translated draws, and check refusals."""
from pathlib import Path
from math import floor
import re
import struct
import subprocess
import sys
import tempfile
from itertools import product

from startup_address_srv_check import draw_texture, prepare_draw_cache

HEX = r"0x[0-9a-f]+"


def check_refusal_lines(log):
    prefix = r"x2w-d3d12 Device::slot70 "
    res_base = r"res\{dim=4 fmt=28 size=4x4x2 mips=2 layout=0x0 flags=0x0 align=0 samples=1/0\}"
    res_swizzle = r"res\{dim=4 fmt=28 size=4x4x2 mips=2 layout=0x105 flags=0x0 align=0 samples=1/0\}"
    expected = [
        # 1. 2D view on 3D resource
        prefix
        + r"unsupported memory SRV arguments: reason=unsupported_description "
        + res_base
        + r" view\{dim=4 fmt=28 map=0x1688 mostmip=1 mips=1 plane=0 clamp=0\} "
        + rf"dest={HEX} dest_in_view_heap=1 address={HEX} address_readable=1",
        # 2. Short backing / unregistered address
        prefix
        + r"unavailable texture at the supplied base and layout: reason=texture_(?:no_resource|unreadable) "
        + res_base
        + r" view\{dim=8 fmt=28 map=0x1688 mostmip=1 mips=1 clamp=0\} "
        + rf"dest={HEX} dest_in_view_heap=1 address={HEX} address_readable=1",
        # 3. Mip out of bounds
        prefix
        + r"unsupported memory SRV arguments: reason=unsupported_description "
        + res_base
        + r" view\{dim=8 fmt=28 map=0x1688 mostmip=2 mips=1 clamp=0\} "
        + rf"dest={HEX} dest_in_view_heap=1 address={HEX} address_readable=1",
        # 4. Explicit non-zero LOD clamp
        prefix
        + r"unsupported memory SRV arguments: reason=unsupported_description "
        + res_base
        + r" view\{dim=8 fmt=28 map=0x1688 mostmip=1 mips=1 clamp=1\} "
        + rf"dest={HEX} dest_in_view_heap=1 address={HEX} address_readable=1",
        # 5. Wrong layout (0x105)
        prefix
        + r"unsupported memory SRV arguments: reason=volume_unsupported_swizzle "
        + res_swizzle
        + r" view\{dim=8 fmt=28 map=0x1688 mostmip=1 mips=1 clamp=0\} "
        + rf"dest={HEX} dest_in_view_heap=1 address={HEX} address_readable=1",
    ]
    lines = [line.rstrip() for line in log.splitlines() if "Device::slot70" in line]
    if len(lines) != len(expected):
        raise AssertionError(f"{len(lines)} slot 70 lines, expected {len(expected)}:\n" + "\n".join(lines))
    for line, pattern in zip(lines, expected):
        if not re.fullmatch(pattern, line):
            raise AssertionError(f"slot 70 line does not match\n  line     {line}\n  expected {pattern}")


def check(directory, consumer, backend):
    hlsl = (
        "Texture3D<float4> t : register(t0);\n"
        "float4 main(float4 p : SV_Position) : SV_Target {\n"
        "  uint x = uint(p.x);\n"
        "  int slice = (x == 1 || x == 5) ? 1 : 0;\n"
        "  return t.Load(int4(0, 0, slice, 0));\n"
        "}\n"
    )
    trap_log = directory / "trap.log"
    actual = draw_texture(
        directory,
        consumer,
        backend,
        hlsl,
        {"X2W_TEST_ADDRESS_VOLUME_SRV": "1", "X2W_TRAP_LOG": str(trap_log), "X2W_VOLUME_TEXTURE": "0"},
        "tex3d",
    )
    if actual is None:
        return 77
    c0 = (200, 50, 10, 255)
    c1 = (50, 200, 10, 255)
    c2 = (10, 50, 200, 255)
    columns = [c0, c1, c2, c2, c0, c1, c0, c0]
    expected = bytes(c for _ in range(4) for column in columns for c in column)
    if actual != expected:
        for i in range(0, len(expected), 4):
            if actual[i:i + 4] != expected[i:i + 4]:
                raise AssertionError(f"column {(i // 4) % 8} row {i // 32}: actual {actual[i:i + 4].hex()} "
                                     f"expected {expected[i:i + 4].hex()}")
        raise AssertionError(f"actual length {len(actual)}, expected {len(expected)}")
    check_refusal_lines(trap_log.read_text(errors="replace"))
    print(f"{backend}: volume memory SRV (slices 0/1, mips 0/1, refused 2D view/short backing/mip overflow/clamp/layout) exact")
    return 0


SIDE = 17
POINTS = (
    (0, 0, 0), (16, 4, 0), (2, 16, 3), (0, 0, 16),
    (16, 16, 16), (5.25, 9.5, 2.75), (3.5, 4.25, 15.5), (16, 12.5, 16),
)


def voxel(x, y, z):
    return 4 + 2 * x + 3 * y + 4 * z + 3 * ((x ^ y ^ z) & 3)


def tiled_volume(reverse_slices=False):
    # Write each little-endian float by reading the published 32-bpp bit
    # diagram from low to high, consuming one coordinate bit per letter.
    backing = bytearray(2 * 65536)
    offsets = set()
    for x, y, z in product(range(SIDE), repeat=3):
        coordinates = {"X": x, "Y": y, "Z": z}
        offset = (z // 16) * 65536
        for bit, axis in enumerate(reversed("XYZXYZXYZZYYXX--")):
            if axis != "-":
                offset |= (coordinates[axis] & 1) << bit
                coordinates[axis] >>= 1
        assert offset not in offsets, (x, y, z, offset)
        offsets.add(offset)
        struct.pack_into("<f", backing, offset, voxel(x, y, SIDE - 1 - z if reverse_slices else z) / 255)
    return backing


def sample(point, reverse_slices=False):
    base = tuple(floor(value) for value in point)
    fraction = tuple(value - low for value, low in zip(point, base))
    value = 0.0
    for dx, dy, dz in product(range(2), repeat=3):
        x, y, z = (min(SIDE - 1, low + delta) for low, delta in zip(base, (dx, dy, dz)))
        fx, fy, fz = fraction
        weight = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
        value += weight * voxel(x, y, SIDE - 1 - z if reverse_slices else z)
    return round(value)


def pixel_mismatches(actual, reds):
    if len(actual) != 8 * 4 * 4:
        raise AssertionError(f"readback length {len(actual)}, expected 128 bytes")
    return [
        (x, y, actual[i:i+4].hex(), reds[x])
        for y in range(4) for x in range(8)
        for i in (4 * (y * 8 + x),)
        if abs(actual[i] - reds[x]) > 1 or actual[i+1:i+4] != bytes((0, 0, 255))
    ]


def check_tiled(directory, consumer, backend):
    points = ", ".join(f"float3({x}, {y}, {z})" for x, y, z in POINTS)
    shader = f"""
Texture3D<float> volume : register(t0);
SamplerState linearSampler : register(s0);
static const float3 points[8] = {{ {points} }};
float4 main(float4 p : SV_Position) : SV_Target {{
    float red = volume.SampleLevel(linearSampler, (points[uint(p.x)] + 0.5) / {SIDE}.0, 0);
    return float4(red, 0, 0, 1);
}}
"""
    backing = directory / "volume.bin"
    backing.write_bytes(tiled_volume())
    env, args, output = prepare_draw_cache(
        directory, consumer, shader, {"X2W_TEST_TILED_VOLUME": str(backing)},
        "srv 0 tex3d table 0 0\nsrvuse 0 sample\nsmp 0 0x92 0 0x500000 0",
    )
    expected = [sample(point) for point in POINTS]
    # Run the default setting, the opt-in, and an opt-in with reversed source Z.
    for enabled, mutant in ((False, False), (True, False), (True, True)):
        env.pop("X2W_VOLUME_TEXTURE", None)
        if enabled:
            env["X2W_VOLUME_TEXTURE"] = "1"
        if mutant:
            backing.write_bytes(tiled_volume(reverse_slices=True))
        output.unlink(missing_ok=True)
        result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode == 77:
            return 77
        if result.returncode:
            raise RuntimeError(f"volume consumer failed ({result.returncode})\n{result.stderr[-5000:]}")
        assert f"tiled volume descriptor_changed={int(enabled)}" in result.stdout.splitlines()
        refused = [line for line in result.stderr.splitlines() if "Device::slot70" in line]
        if enabled:
            assert not refused, refused
        else:
            assert len(refused) == 1 and "reason=volume_texture_disabled" in refused[0], refused
        actual = output.read_bytes()
        differences = pixel_mismatches(actual, expected if enabled else [0] * 8)
        if mutant:
            assert differences, "reversed Z slices passed the original CPU oracle"
            mutated = pixel_mismatches(actual, [sample(point, reverse_slices=True) for point in POINTS])
            assert not mutated, mutated[:1]
            print(f"reversed Z differs at {len(differences)} pixels, first {differences[0]}")
        else:
            assert not differences, differences[:1]
    print(f"{backend}: 17x17x17 R32_FLOAT, 32 pixels per run; trilinear, switch-off and reversed Z checked")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        consumer, backend, *mode = sys.argv[1:]
        run = check_tiled if mode == ["tiled"] else check
        sys.exit(run(Path(temporary), consumer, backend))
