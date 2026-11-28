"""A graphics root SRV (list slot 44) at a title address without an upload mapping, read by a translated draw.

tests/startup_root_srv_draw.cpp (X2W_TEST_ROOT_SRV_DRAW=write-after): slot 44 sees the buffer holding red, the title
writes green, then draws, and the stand-in's copy counts are printed. Run with X2W_D3D12_ROOT_SRV_AT_DRAW unset and set:
  - both: the translated draw reads the SRV from its title address at the draw and draws green;
  - unset: slot 44 also copies a window into the upload ring for the desktop binding (copied 1);
  - set: no draw reads that binding, so slot 44 skips the copy (skipped 1, copied 0).
The left half of the 8x4 target is drawn, the right half keeps the clear colour.
"""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from startup_address_srv_check import prepare_draw_cache

HLSL = (
    "ByteAddressBuffer colour : register(t0);\n"
    "float4 main(float4 p : SV_Position) : SV_Target {\n"
    "  return asfloat(colour.Load4(0));\n"
    "}\n"
)
CLEAR = bytes((0, 0, 255, 255))
GREEN = bytes((0, 255, 0, 255))
SWITCH = "X2W_D3D12_ROOT_SRV_AT_DRAW"
COUNTS = re.compile(r"root SRV copies: skipped (-?\d+), copied (-?\d+)")


def run(directory, consumer, backend, at_draw):
    folder = directory / f"write-after-{int(at_draw)}"
    folder.mkdir()
    os.environ.pop(SWITCH, None)  # an inherited setting must not decide the unset case
    modes = {"X2W_TEST_ROOT_SRV_DRAW": "write-after", "X2W_TRAP_LOG": str(folder / "trap.log")}
    if at_draw:
        modes[SWITCH] = "1"
    env, args, output = prepare_draw_cache(folder, consumer, HLSL, modes, "srv 0 buf root 0")
    result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode == 77:
        return None, None
    if result.returncode:
        raise RuntimeError(f"consumer failed ({result.returncode})\n{result.stderr[-5000:]}")
    found = COUNTS.search(result.stdout)
    if not found:
        raise AssertionError(f"{SWITCH}={int(at_draw)}: no copy counts printed")
    return output.read_bytes(), tuple(int(value) for value in found.groups())


def check(directory, consumer, backend):
    expected_pixels = (GREEN * 4 + CLEAR * 4) * 4
    # (skipped, copied)
    expected_counts = {False: (0, 1), True: (1, 0)}
    for at_draw in (False, True):
        pixels, counts = run(directory, consumer, backend, at_draw)
        if pixels is None:
            return 77
        label = f"{SWITCH}={int(at_draw)}"
        if pixels != expected_pixels:
            raise AssertionError(f"{label}: actual {pixels.hex()} expected {expected_pixels.hex()}")
        if counts != expected_counts[at_draw]:
            raise AssertionError(f"{label}: copies (skipped, copied) {counts}, "
                                 f"expected {expected_counts[at_draw]}")
        print(f"{backend} {label}: 32 pixels exact (green, read at the draw); "
              f"skipped {counts[0]}, copied {counts[1]}")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
