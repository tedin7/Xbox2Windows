"""Publish placed Texture3D SRVs by address, sample slices and mips in translated draws, and check refusals."""
from pathlib import Path
import re
import sys
import tempfile

from startup_address_srv_check import draw_texture

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
        + r"unavailable texture at the supplied base and layout: reason=texture_unavailable "
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
        + r"unsupported memory SRV arguments: reason=unsupported_description "
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
        {"X2W_TEST_ADDRESS_VOLUME_SRV": "1", "X2W_TRAP_LOG": str(trap_log)},
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


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
