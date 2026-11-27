"""Sample every face of placed slot-70 cube SRVs on WARP and compare the readback."""
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_address_srv_check import prepare_draw_cache


REFUSAL = "Device::slot70 unsupported memory SRV arguments: reason=unsupported_description "


def reference(array, swapped=False):
    pixels = bytearray()
    for y in range(4):
        first_face = 6 + 6 * (y % 2) if array else 0
        for x in range(8):
            face = x % 6
            if swapped and face < 2:
                face = 1 - face
            pixels.extend((16 + 11 * (first_face + face), 0, 0, 255))
    return pixels


def check_pixels(actual, expected, case):
    if len(actual) != len(expected):
        raise AssertionError(f"{case}: got {len(actual)} bytes, expected {len(expected)}")
    for offset in range(0, len(expected), 4):
        if actual[offset:offset + 4] != expected[offset:offset + 4]:
            y, x = divmod(offset // 4, 8)
            raise AssertionError(f"{case}: pixel ({x},{y}) got {actual[offset:offset + 4].hex()}, "
                                 f"expected {expected[offset:offset + 4].hex()}")


def check(directory, consumer, array, enabled, tiled=False):
    modes = {"X2W_TEST_ADDRESS_CUBE_SRV": "1"}
    if array:
        modes["X2W_TEST_ADDRESS_CUBE_ARRAY"] = "1"
    if tiled:
        modes["X2W_TEST_ADDRESS_CUBE_TILED"] = "1"

    if enabled:
        texture = "TextureCubeArray" if array else "TextureCube"
        coordinate = "float4(direction, uint(p.y) % 2)" if array else "direction"
        hlsl = f"""
{texture}<float4> t : register(t0);
SamplerState point_sampler : register(s0);
float4 main(float4 p : SV_Position) : SV_Target {{
    const float3 faces[6] = {{
        float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0),
        float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1)
    }};
    float3 direction = faces[uint(p.x) % 6];
    return t.SampleLevel(point_sampler, {coordinate}, 0);
}}
"""
        kind = "texcubea" if array else "texcube"
        binding = f"srv 0 {kind} table 0 0\nsrvuse 0 sample\nsmp 0 0 0 0 0"
    else:
        hlsl = "float4 main() : SV_Target { return float4(0, 0, 0, 1); }\n"
        binding = ""

    directory.mkdir()
    env, args, output = prepare_draw_cache(directory, consumer, hlsl, modes, binding)
    env.pop("X2W_TEST_ADDRESS_CUBE_SWAP", None)
    if not array:
        env.pop("X2W_TEST_ADDRESS_CUBE_ARRAY", None)
    if not tiled:
        env.pop("X2W_TEST_ADDRESS_CUBE_TILED", None)
    if enabled:
        env["X2W_SLOT70_MEMORY_CUBE"] = "1"
    else:
        env.pop("X2W_SLOT70_MEMORY_CUBE", None)

    def run(expected_refusals):
        output.unlink(missing_ok=True)
        result = subprocess.run([*args, "warp", output], env=env, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode == 77:
            return None
        if result.returncode:
            raise AssertionError(f"consumer returned {result.returncode}\n{result.stderr[-5000:]}")
        refusals = [line for line in result.stderr.splitlines() if "Device::slot70" in line]
        if len(refusals) != expected_refusals or any(REFUSAL not in line for line in refusals):
            raise AssertionError(f"expected {expected_refusals} unsupported descriptions, got {refusals}")
        if f"slot 70 cube descriptor_changed={int(enabled)}" not in result.stdout.splitlines():
            raise AssertionError("unexpected descriptor update")
        return output.read_bytes()

    actual = run(int(array) + int(not enabled))
    if actual is None:
        return 77
    expected = reference(array) if enabled else bytes((0, 0, 0, 255)) * 32
    check_pixels(actual, expected, directory.name)
    if enabled:
        env["X2W_TEST_ADDRESS_CUBE_SWAP"] = "1"
        mutant = run(int(array))
        if mutant is None:
            return 77
        try:
            check_pixels(mutant, expected, f"{kind} swapped faces")
        except AssertionError as mismatch:
            print(mismatch)
        else:
            raise AssertionError("swapped +X/-X face mutant matched the CPU reference")
        check_pixels(mutant, reference(array, swapped=True), f"{kind} mutant readback")

    print(f"warp: {directory.name}, all 32 RGBA pixels exact")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        for shape, array in (("cube", False), ("cube_array", True)):
            for backing in ("placed", "tiled"):
                for enabled in (False, True):
                    name = f"{shape}_{backing}_{'on' if enabled else 'off'}"
                    result = check(Path(temporary) / name, sys.argv[1], array, enabled, backing == "tiled")
                    if result:
                        sys.exit(result)
