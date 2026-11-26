"""WARP slot70 readback of single-row tiled BC blocks ending at an uncommitted page."""
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_address_bc_view_check import bc3_block, bc7_block
from startup_address_srv_check import prepare_draw_cache

HEIGHT = 4
TARGET_WIDTH, TARGET_HEIGHT = 512, 128
REFUSAL = "Device::slot70 unavailable texture at the supplied base and layout: reason=texture_unreadable"


def build(width, encode, mutant=False):
    """Return short backing and the RGBA image in logical texel order."""
    tiled = bytearray()
    image = bytearray(width * HEIGHT * 4)
    for x in range(width // 4):
        block, colors = encode(x, 0)
        tile, local = divmod(x, 16)
        micro, column = divmod(local, 4)
        if mutant:
            # Exchange the middle two 256-byte micro-tiles in the occupied tile row.
            micro = (0, 2, 1, 3)[micro]
        offset = tile * 4096 + (0, 512, 2048, 2560)[micro] + column * 64
        if len(tiled) < offset + 16:
            tiled.extend(bytes(offset + 16 - len(tiled)))
        tiled[offset:offset + 16] = block
        for y in range(HEIGHT):
            start = (y * width + x * 4) * 4
            image[start:start + 16] = colors[y * 16:(y + 1) * 16]
    return tiled, bytes(image)


def check(directory, consumer, width, fmt, encode):
    hlsl = ("Texture2D<float4> t : register(t0);\n"
            "float4 main(float4 p : SV_Position) : SV_Target {\n"
            f"  return t.Load(int3(uint(p.x) % {width}, uint(p.y) % {HEIGHT}, 0));\n}}\n")
    env, args, output = prepare_draw_cache(directory, consumer, hlsl, {}, "srv 0 tex2d table 0 0")
    source, trap = directory / "tiled.bin", directory / "trap.log"
    backing, image = build(width, encode)
    wrong, _ = build(width, encode, mutant=True)
    stride = width * 4
    expected = b"".join(image[y * stride:(y + 1) * stride] * (TARGET_WIDTH // width)
                        for y in (row % HEIGHT for row in range(TARGET_HEIGHT)))
    for mode, enabled, data, refused in (
        ("switch off", "0", backing, True),
        ("switch on", "1", backing, False),
        ("missing last block", "1", backing[:-16], True),
        ("wrong micro-tile order", "1", wrong, False),
    ):
        source.write_bytes(data)
        output.unlink(missing_ok=True)
        trap.unlink(missing_ok=True)
        modes = {"X2W_TEST_CONSOLE_SHORT_BACKING": "1", "X2W_TEST_CONSOLE_VIEW": str(source),
                 "X2W_TEST_CONSOLE_DESC": f"{width},{HEIGHT},1,{fmt},261,0", "X2W_TRAP_LOG": str(trap),
                 "X2W_SLOT70_DETILE_MORE": enabled}
        result = subprocess.run([*args, "warp", output], env={**env, **modes}, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode == 77:
            return 77
        if result.returncode:
            raise RuntimeError(f"consumer failed ({result.returncode})\n{result.stderr[-5000:]}")
        pixels = output.read_bytes()
        log = trap.read_text(errors="replace")
        label = f"format {fmt} {width}x{HEIGHT}, {mode}"
        if len(pixels) != len(expected) or (REFUSAL in log) != refused:
            raise AssertionError(f"{label}: unexpected readback length or slot70 refusal")
        if refused:
            if any(pixels):
                raise AssertionError(f"{label}: rejected SRV drew texels")
        else:
            if "published BC shape" not in log:
                raise AssertionError(f"{label}: missing publication")
            mismatch = next((i for i in range(0, len(pixels), 4)
                             if pixels[i:i + 4] != expected[i:i + 4]), None)
            if mode == "wrong micro-tile order":
                # The first micro-tile is intact. A null view cannot pass this control.
                if mismatch != 16 * 4 or not any(pixels):
                    raise AssertionError(f"{label}: expected first mismatch at pixel (16, 0), byte offset {mismatch}")
            elif mismatch is not None:
                y, x = divmod(mismatch // 4, TARGET_WIDTH)
                raise AssertionError(f"{label}: ({x},{y}) GPU {pixels[mismatch:mismatch + 4].hex()} "
                                     f"CPU {expected[mismatch:mismatch + 4].hex()}")
        print(f"WARP {label}: {len(data)} source bytes, expected outcome")
    return 0


if __name__ == "__main__":
    for width, fmt, encode in ((128, 77, bc3_block), (64, 98, bc7_block)):
        with tempfile.TemporaryDirectory() as temporary:
            result = check(Path(temporary), sys.argv[1], width, fmt, encode)
            if result:
                sys.exit(result)
