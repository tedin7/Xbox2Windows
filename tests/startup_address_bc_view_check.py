"""Draw 2×2 4 KiB tiled BC3/BC7 textures from guest addresses, including swapped tiles."""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from bc7_draw_check import make_block as bc7_block
from bc7_tiles_check import block_index
from startup_address_srv_check import prepare_draw_cache

WIDTH, HEIGHT = 128, 128
HLSL = ("Texture2D<float4> t : register(t0);\n"
        "float4 main(float4 p : SV_Position) : SV_Target {\n"
        f"  return t.Load(int3(uint(p.x) % {WIDTH}, uint(p.y), 0));\n}}\n")


def bc3_block(x, y):
    # Literal DXT5 endpoint selection, avoiding interpolation tolerances.
    red, green, blue = bool(x & 16), bool(y & 16), bool((x ^ y) & 1)
    alpha = (7 * x + 13 * y) % 256
    color = (0xf800 if red else 0) | (0x07e0 if green else 0) | (0x001f if blue else 0)
    block = struct.pack("<BB6xHHI", alpha, 0, color, 0, 0)
    return block, bytes((255 * red, 255 * green, 255 * blue, alpha)) * 16


def make_surface(encode):
    tiled = bytearray(4 * 4096)
    rgba = bytearray(128 * 128 * 4)
    for y in range(32):
        for x in range(32):
            block, colors = encode(x, y)
            tile = (y // 16) * 2 + x // 16
            start = tile * 4096 + 16 * block_index(x % 16, y % 16)
            tiled[start:start + 16] = block
            for row in range(4):
                start = 4 * ((y * 4 + row) * 128 + x * 4)
                rgba[start:start + 16] = colors[row * 16:row * 16 + 16]
    return tiled, rgba


def source_block(x, y, tile_pitch, swapped):
    """The block of the 2x2 tile grid that a detile with this tile pitch shows at block (x, y)."""
    tile = (y // 16) * tile_pitch + x // 16
    tile ^= swapped
    return (tile % 2) * 16 + x % 16, (tile // 2) * 16 + y % 16


def expected_panels(rgba, width, height, tile_pitch):
    """Four panels of 128x128 texels. A partial view with a later row write reads as null."""
    image = bytearray(4 * WIDTH * 4 * HEIGHT)
    for panel, swapped in enumerate((0, 1, 1, 0)):
        if panel == 3 and (width % 64 or height % 64):
            continue
        for y in range(height):
            for x in range(width):
                bx, by = source_block(x // 4, y // 4, tile_pitch, swapped)
                start = 4 * ((by * 4 + y % 4) * 128 + bx * 4 + x % 4)
                pos = 4 * (y * 4 * WIDTH + panel * WIDTH + x)
                image[pos:pos + 4] = rgba[start:start + 4]
    return bytes(image)  # texels outside the texture load as zero


class Drawer:
    """Translates the pipeline once, then draws each case with the same shader cache."""

    def __init__(self, directory, consumer, backend):
        self.directory, self.consumer, self.backend, self.prepared = directory, consumer, backend, None

    def __call__(self, modes):
        if self.prepared is None:
            self.prepared = prepare_draw_cache(self.directory, self.consumer, HLSL, modes, "srv 0 tex2d table 0 0")
        env, args, output = self.prepared
        output.unlink(missing_ok=True)  # no case reads the previous case's pixels
        env = {key: value for key, value in env.items() if key != "X2W_TEST_BC3"}  # the consumer tests presence
        result = subprocess.run([*args, self.backend, output], env={**env, **modes}, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode == 77:
            return None
        if result.returncode:
            raise RuntimeError(f"consumer failed ({result.returncode})\n{result.stderr[-5000:]}")
        return output.read_bytes()


def check(directory, draw, backend, fmt, width, height):
    tiled, rgba = make_surface(bc3_block if fmt == "BC3" else bc7_block)
    source = directory / "tiled.bin"
    source.write_bytes(tiled)
    trap_log = directory / "trap.log"
    modes = {"X2W_TEST_BC_VIEW": str(source), "X2W_TEST_BC_SIZE": f"{width}x{height}", "X2W_TRAP_LOG": str(trap_log)}
    if fmt == "BC3":
        modes["X2W_TEST_BC3"] = "1"
    actual = draw(modes)
    if actual is None:
        return 77
    if len(actual) != WIDTH * 4 * HEIGHT * 4:
        raise AssertionError(f"{fmt} readback length {len(actual)}")
    expected = expected_panels(rgba, width, height, 2)
    for pos in range(0, len(expected), 4):
        if actual[pos:pos + 4] != expected[pos:pos + 4]:
            y, x = divmod(pos // 4, 4 * WIDTH)
            raise AssertionError(f"{fmt} {width}x{height} panel {x // WIDTH} ({x % WIDTH},{y}) "
                                 f"actual {actual[pos:pos + 4].hex()} expected {expected[pos:pos + 4].hex()}")
    # Controls: exchanged tiles, and for a partial last tile column the tile pitch of the unpadded width.
    swapped = sum(expected[p:p + 4] != expected[p + 4 * WIDTH:p + 4 * WIDTH + 4]
                  for p in range(0, len(expected), 4) if (p // 4) % (4 * WIDTH) < WIDTH)
    if swapped == 0:
        raise AssertionError("Swapping the tiles did not change the reference image")
    columns = width // 4
    unpadded = 0
    if columns % 16:
        wrong = expected_panels(rgba, width, height, columns // 16)
        unpadded = sum(actual[p:p + 4] != wrong[p:p + 4] for p in range(0, len(actual), 4))
        if unpadded == 0:
            raise AssertionError("The unpadded tile pitch did not change the image")
    log = trap_log.read_text(errors="replace")
    code = 77 if fmt == "BC3" else 98
    partial = width % 64 or height % 64
    tiled_bytes = 16384  # 2 x 2 tiles of 4 KiB, partial or not
    refusals = [("unsupported_description", width - 2, height), ("unsupported_description", width, height - 2),
                ("partial_tile_beyond_xg_size", 1028, 1028)]
    if partial:
        refusals.append(("partial_tile_xg_rows", width, height))
    for reason, w, h in refusals:
        if not re.search(rf"Device::slot70 .*: reason={reason} res\{{dim=3 fmt={code} size={w}x{h}x1 ", log):
            raise AssertionError(f"{fmt}: no {reason} refusal for {w}x{h}")
    lines = re.findall(r"partial-tile BC view (\d+x\d+) format \d+ layout 0x105: (\w+), (\d+) tiled bytes", log)
    # One line per first publication at an address: the first backing and the exchanged one, the 256x196
    # boundary (padded size equal to the reported size), the backing announced as XG rows (refused), the
    # one announced as opt-in tiled BC7 blocks, and the one published before an XG row copy over it.
    view = f"{width}x{height}"
    boundary = [("256x196", "published", "65536")]
    published = (view, "published", str(tiled_bytes))
    expected = ([published] * 2 + boundary + [(view, "partial_tile_xg_rows", str(tiled_bytes)), published, published]
                if partial else boundary)
    # Rows announced before the first publication, and rows copied over a published view: both refused.
    held = re.findall(rf"reason=partial_tile_xg_rows res\{{dim=3 fmt={code} size={view}x1 ", log)
    if len(held) != (2 if partial else 0):
        raise AssertionError(f"{fmt}: {len(held)} partial_tile_xg_rows refusals for {view}")
    if lines != expected:
        raise AssertionError(f"{fmt}: partial-tile lines {lines}, expected {expected}")
    print(f"{backend} {fmt} {width}x{height}: {4 * WIDTH * HEIGHT} RGBA pixels exact; exchanged tiles differ in "
          f"{swapped} texels; unpadded tile pitch differs in {unpadded}; {len(refusals)} refusals as expected")
    return 0


if __name__ == "__main__":
    consumer, backend = sys.argv[1:3]
    with tempfile.TemporaryDirectory() as shared:
        draw = Drawer(Path(shared), consumer, backend)
        for fmt, width, height in (("BC3", 128, 128), ("BC7", 128, 128), ("BC3", 100, 68), ("BC7", 100, 68)):
            with tempfile.TemporaryDirectory() as temporary:
                result = check(Path(temporary), draw, backend, fmt, width, height)
                if result:
                    sys.exit(result)
