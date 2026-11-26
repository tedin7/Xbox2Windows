"""A root constant buffer view (list slot 42) reaches a translated pixel shader that reads it through its root pointer."""
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
CLEAR = bytes((0, 0, 255, 255))


def run(directory, consumer, backend, mode):
    folder = directory / mode
    folder.mkdir()
    log = folder / "trap.log"
    pixels = draw_texture(folder, consumer, backend, HLSL, {"X2W_TEST_ROOT_CBV": mode, "X2W_TRAP_LOG": str(log)},
                          binding="srv 0 buf root 0")
    ignored = [line for line in log.read_text(errors="replace").splitlines() if "root CBV" in line] if log.exists() else []
    return pixels, ignored


def check(directory, consumer, backend):
    # The left half of the 8x4 target is drawn, the right half keeps the clear colour.
    cases = (
        ("bound", bytes((255, 255, 0, 255)), 0),   # the colour the title bound
        ("unbound", bytes(4), 0),                  # no call: the shader reads zeros
        ("null", bytes(4), 0),                     # binding 0 after a valid address clears it
        ("same_root", bytes((255, 255, 0, 255)), 0),
        ("switch_root", bytes(4), 0),
        ("wrong_type", bytes(4), 1),               # an SRV parameter ignores the call, with one log line for two calls
    )
    for mode, drawn, lines in cases:
        pixels, ignored = run(directory, consumer, backend, mode)
        if pixels is None:
            return 77
        expected = (drawn * 4 + CLEAR * 4) * 4
        if pixels != expected:
            raise AssertionError(f"{mode}: actual {pixels.hex()} expected {expected.hex()}")
        if len(ignored) != lines:
            raise AssertionError(f"{mode}: {len(ignored)} root CBV log lines, expected {lines}:\n" + "\n".join(ignored))
        print(f"{backend} {mode}: 32 pixels exact, {lines} ignored-call log lines")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
