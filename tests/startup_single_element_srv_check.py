"""Draw one-texel 2D and 3D textures published by address (device slot 70) and check refused descriptions.

Each run draws eight 1x4 columns from a translated pixel shader that loads texel (0, 0, 0):
  0  RGBA8, layout UNKNOWN            1  RGBA8, layout 0x105 (another address)
  2  the column-0 texel rewritten and published again
  3  the same texel rewritten again without a new publication (it must read the new value, in the same frame),
     after sixteen refused calls that each name memory holding another texel: 2x1, two mips, two slices, four
     samples, XOR layout 0x115, reserved layout 0x10b, 4 MiB alignment, BC1, R8, a UAV flag, the other view
     dimension, a two-mip view, a zero address, and flagless views at the base of a placed UAV texture and
     256 bytes into a placed UAV buffer, plus a texel straddling that buffer's base; the consumer also checks that each refused call left the destination
     descriptor's 32 bytes unchanged and exits with 60 + the call's position otherwise
  4  B8G8R8A8, layout 0x100           5  R16G16B16A16_FLOAT, layout 0x105
  6  column 5's view drawn again      7  the column-0 address with 64 KiB alignment
Usage: startup_single_element_srv_check.py <startup_stream_consumer> <hardware|warp>
Without the single-element path every column except 3 and 6 reads a null view (zero).
"""
from pathlib import Path
import sys
import tempfile

from startup_address_srv_check import draw_texture

A = (0x11, 0x22, 0x33, 0xff)
B = (0xc0, 0x40, 0x80, 0xff)
C = (0x5a, 0xa5, 0x3c, 0xff)
D = (0x81, 0x18, 0xe7, 0xff)
BGRA = (0xf0, 0x20, 0x10, 0xff)
HALF = (64, 191, 255, 255)  # 0.25, 0.75, 1, 1 as UNORM8


def expected():
    columns = [A, B, C, D, BGRA, HALF, HALF, D]
    return bytes(c for _ in range(4) for column in columns for c in column)


def check(directory, consumer, backend, dimension):
    texture, load = ("Texture3D", "int4(0, 0, 0, 0)") if dimension == "3d" else ("Texture2D", "int3(0, 0, 0)")
    hlsl = (f"{texture}<float4> t : register(t0);\n"
            "float4 main(float4 p : SV_Position) : SV_Target {\n"
            f"  return t.Load({load});\n}}\n")
    kind = "tex3d" if dimension == "3d" else "tex2d"
    actual = draw_texture(directory, consumer, backend, hlsl, {"X2W_TEST_SINGLE_ELEMENT": dimension}, kind)
    if actual is None:
        return 77
    want = expected()
    if actual != want:
        for i in range(0, len(want), 4):
            if actual[i:i + 4] != want[i:i + 4]:
                raise AssertionError(f"{dimension} column {(i // 4) % 8} row {i // 32}: actual {actual[i:i + 4].hex()} "
                                     f"expected {want[i:i + 4].hex()}")
        raise AssertionError(f"{dimension}: readback length {len(actual)}, expected {len(want)}")
    print(f"{backend} {dimension}: 8 columns of one-texel views exact, sixteen refused calls kept the previous view")
    return 0


if __name__ == "__main__":
    for dimension in ("2d", "3d"):
        with tempfile.TemporaryDirectory() as temporary:
            result = check(Path(temporary), *sys.argv[1:], dimension)
            if result:
                sys.exit(result)
