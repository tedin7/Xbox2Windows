"""A table buffer view over a placed buffer: per-draw copies by default, a live alias with X2W_D3D12_TABLE_ALIAS=1.

Cases (X2W_TEST_TABLE_ALIAS_WRITE in tests/startup_address_srv.cpp):
  writes        records rewritten between two draws and after recording: copies show 40 | 80, the alias 120.
  shared-index  the alias is also the draw's index buffer. Run with the debug layer and GPU-based validation; no
                error may be reported, so the alias must hold both the index-buffer and shader-resource states.
  odd-size      a 254-byte buffer under a 127 x 2-byte structured view: the 256-byte raw view does not fit the
                alias, so even with the switch the draw reads a copy (40), not the later write (120).
"""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from startup_address_srv_check import draw_texture, prepare_draw_cache

VERTEX = """
ByteAddressBuffer records : register(t0);
struct Vertex {
    float4 position : SV_Position;
    nointerpolation uint2 values : TEXCOORD0;
};
Vertex main(uint id : SV_VertexID) {
    const float2 corners[3] = {float2(-1, 1), float2(3, 1), float2(-1, -3)};
    Vertex output;
    output.position = float4(corners[id], 0.5, 1);
    output.values = uint2(records.Load(16), records.Load(20));
    return output;
}
"""

PIXEL = """
float4 main(float4 position : SV_Position, nointerpolation uint2 values : TEXCOORD0) : SV_Target {
    return float4(float2(values) / 255.0, 0, 1);
}
"""

BINDING = "srv 0 buf table 0 0"
VALIDATION = {"X2W_TEST_DEBUG_LAYER": "1", "X2W_TEST_GPU_VALIDATION": "1"}


def texel(value):
    return bytes((value, value + 1, 0, 255))


def run(folder, consumer, backend, modes):
    """Pixels and stdout of one consumer run, or (None, "") for a capability skip."""
    try:
        env, args, output = prepare_draw_cache(folder, consumer, PIXEL, modes, BINDING, vertex_hlsl=VERTEX)
    except subprocess.CalledProcessError as error:
        if error.returncode == 77:  # no existing heaps, debug layer or GPU-based validation
            return None, ""
        raise
    result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode == 77:
        return None, ""
    if result.returncode:
        raise RuntimeError(f"consumer failed ({result.returncode})\n{result.stderr[-5000:]}")
    return output.read_bytes(), result.stdout


def debug_errors(stdout, name):
    found = re.search(r"^table alias debug layer errors: (\d+)$", stdout, re.M)
    if not found:
        raise AssertionError(f"{name}: no debug layer message count")
    return int(found.group(1))


def check(directory, consumer, backend):
    # The fixture writes 40, draws the left half, writes 80, draws the right half, then writes 120 before the list runs.
    cases = (
        # The copy case clears the switch, so it keeps testing the default when the switch is set in the environment.
        ("copy", {"X2W_D3D12_TABLE_ALIAS": "0"}, texel(40) * 4 + texel(80) * 4),
        ("alias", {"X2W_D3D12_TABLE_ALIAS": "1"}, texel(120) * 8),
    )
    for name, switches, row in cases:
        folder = directory / name
        folder.mkdir()
        pixels = draw_texture(folder, consumer, backend, PIXEL, {"X2W_TEST_TABLE_ALIAS_WRITE": "1", **switches},
                              binding=BINDING, vertex_hlsl=VERTEX)
        if pixels is None:
            return 77
        expected = row * 4
        if pixels != expected:
            raise AssertionError(f"{backend}: {name} table buffer got {pixels.hex()}, expected {expected.hex()}")
        print(f"{backend}: 32 pixels matched the {name} reading of a rewritten table buffer")

    for name, extra, value in (("shared-index", VALIDATION, 40), ("odd-size", VALIDATION, 40)):
        folder = directory / name
        folder.mkdir()
        modes = {"X2W_TEST_TABLE_ALIAS_WRITE": name, "X2W_D3D12_TABLE_ALIAS": "1", **extra}
        pixels, stdout = run(folder, consumer, backend, modes)
        if pixels is None:
            return 77
        expected = texel(value) * 32
        if pixels != expected:
            raise AssertionError(f"{backend}: {name} got {pixels.hex()}, expected {expected.hex()}")
        errors = debug_errors(stdout, name)
        if errors:
            raise AssertionError(f"{backend}: {name}: {errors} debug layer errors")
        print(f"{backend}: {name}: 32 pixels exact, no debug layer errors")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
