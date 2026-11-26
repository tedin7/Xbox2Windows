"""A constant buffer the title rewrites between two draws of one frame reaches the second draw with its new contents."""
from pathlib import Path
import sys
import tempfile

from startup_address_srv_check import draw_texture

HLSL = (
    "ByteAddressBuffer constants : register(t0);\n"
    "float4 main(float4 p : SV_Position) : SV_Target {\n"
    "  return asfloat(constants.Load4(0));\n"
    "}\n"
)
RED = bytes((255, 0, 0, 255))
BLUE = bytes((0, 0, 255, 255))


def run(directory, consumer, backend, mode):
    print(f"mode {mode}", flush=True)
    folder = directory / mode
    folder.mkdir()
    log = folder / "trap.log"
    pixels = draw_texture(folder, consumer, backend, HLSL, {"X2W_TEST_FRAME_COPY": mode, "X2W_TRAP_LOG": str(log)},
                          binding="srv 0 buf table 0 0")
    refreshed = [line for line in log.read_text(errors="replace").splitlines() if "frame copy of" in line] if log.exists() else []
    return pixels, refreshed


def check(directory, consumer, backend):
    # The left half of the 8x4 target is drawn first, the right half second.
    cases = (
        ("changed", (RED * 4 + BLUE * 4) * 4, 1),  # rewritten in place: the second draw must see blue
        ("same", RED * 8 * 4, 0),                  # untouched: both draws red, no new copy
        # A 64 KiB buffer, the second draw in the NEXT frame: a copy kept from the first frame must show the new bytes.
        ("big-changed", (RED * 4 + BLUE * 4) * 4, None),
        ("big-same", RED * 8 * 4, None),
    )
    for mode, expected, refreshed_lines in cases:
        pixels, refreshed = run(directory, consumer, backend, mode)
        if pixels is None:
            return 77
        if pixels != expected:
            raise AssertionError(f"{mode}: actual {pixels.hex()} expected {expected.hex()}")
        if refreshed_lines is not None and len(refreshed) != refreshed_lines:
            raise AssertionError(f"{mode}: {len(refreshed)} refresh log lines, expected {refreshed_lines}:\n" + "\n".join(refreshed))
        print(f"{backend} {mode}: 32 pixels exact, {refreshed_lines} refresh log lines")
    # The controls of the cross-frame reuse: reuse off still shows the new bytes; a reuse that never compares (the mutant) must show
    # the OLD bytes, which proves the big-changed case above would catch it.
    for name, extra, expected in (
        ("big-changed with reuse off", {"X2W_BIG_COPY_REUSE": "0"}, (RED * 4 + BLUE * 4) * 4),
        ("big-changed, never-refill mutant", {"X2W_TEST_FRAME_COPY_STALE": "1"}, RED * 8 * 4),
    ):
        folder = directory / name.replace(" ", "_").replace(",", "")
        folder.mkdir()
        pixels = draw_texture(folder, consumer, backend, HLSL, {"X2W_TEST_FRAME_COPY": "big-changed", **extra},
                              binding="srv 0 buf table 0 0")
        if pixels is None:
            return 77
        if pixels != expected:
            raise AssertionError(f"{name}: actual {pixels.hex()} expected {expected.hex()}")
        print(f"{backend} {name}: 32 pixels exact")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
