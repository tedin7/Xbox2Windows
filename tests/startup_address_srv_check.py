"""Publish placed R8 textures by address, render their texels, and test rejected updates."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from startup_stream_draw_check import gfx2hlsl, pixel, vertex


def prepare_draw_cache(directory, consumer, hlsl, modes, binding, *, vertex_hlsl=None):
    cache = directory / "cache"
    cache.mkdir()
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex())
    ps.write_bytes(pixel(False))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_D3D12_DEBUG="0", X2W_D3D12_DEVICE_NOOP="1", X2W_SHADER_CACHE=str(cache), **modes)
    args = [consumer, vs, ps, root]
    subprocess.run(args, env=env, check=True)
    subprocess.run([sys.executable, "-B", gfx2hlsl.__file__, "--pending", cache], env=env, check=True)
    manifest, = (path for path in cache.glob("*.txt") if path.name != "revision.txt")
    with manifest.open("a", encoding="ascii") as file:
        file.write(f"\n{binding}\n")
    source = manifest.with_suffix(".ps.hlsl")
    source.write_text(hlsl)
    subprocess.run([env["X2W_FXC"], "/nologo", "/T", "ps_5_0", "/E", "main", "/Fo",
                    manifest.with_suffix(".ps.cso"), source], check=True)
    if vertex_hlsl is not None:
        source = manifest.with_suffix(".vs.hlsl")
        source.write_text(vertex_hlsl, encoding="ascii")
        subprocess.run([env["X2W_FXC"], "/nologo", "/T", "vs_5_0", "/E", "main", "/Fo",
                        manifest.with_suffix(".vs.cso"), source], check=True)
    return env, args, output


def draw_texture(directory, consumer, backend, hlsl, modes, view_kind="tex2d", *, binding=None, vertex_hlsl=None):
    binding = binding or f"srv 0 {view_kind} table 0 0"
    env, args, output = prepare_draw_cache(directory, consumer, hlsl, modes, binding, vertex_hlsl=vertex_hlsl)
    result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode == 77:
        return None
    if result.returncode:
        raise RuntimeError(f"consumer failed ({result.returncode})\n{result.stderr[-5000:]}")
    return output.read_bytes()


R8_RESOURCE = r"res\{dim=3 fmt=61 size=%sx2x1 mips=1 layout=0x0 flags=0x0 align=0 samples=1/0\}"
R8_VIEW = r"view\{dim=4 fmt=61 map=0x1688 mostmip=0 mips=1 plane=0 clamp=0\}"
HEX = r"0x[0-9a-f]+"


def check_refusal_lines(log):
    """Each refused slot 70 call logs one line with the cause and the fields the checks read."""
    prefix = r"x2w-d3d12 Device::slot70 "
    expected = [
        # zero address, wrong width, interior address, buffer view, destination off the heap
        prefix + r"unsupported memory SRV arguments: reason=null_address " + R8_RESOURCE % 2 + " " + R8_VIEW
        + rf" dest={HEX} dest_in_view_heap=1 address=0x0 address_readable=0",
        prefix + r"unavailable texture at the supplied base and layout: reason=texture_no_resource " + R8_RESOURCE % 1
        + " " + R8_VIEW + rf" dest={HEX} dest_in_view_heap=1 address={HEX} address_readable=1",
        prefix + r"unavailable texture at the supplied base and layout: reason=texture_no_resource " + R8_RESOURCE % 2
        + " " + R8_VIEW + rf" dest={HEX} dest_in_view_heap=1 address={HEX} address_readable=1",
        prefix + r"unavailable buffer at the supplied address: reason=buffer_unavailable "
        + r"res\{dim=1 fmt=0 size=256x1x1 mips=1 layout=0x1 flags=0x0 align=0 samples=1/0\} "
        + r"view\{dim=1 fmt=0 map=0x1688 first=0 count=64 stride=4 flags=0x0 \(buffer views are not supported\)\} "
        + rf"dest={HEX} dest_in_view_heap=1 address={HEX} address_readable=1",
        prefix + r"unsupported memory SRV arguments: reason=destination_outside_view_heaps " + R8_RESOURCE % 2 + " "
        + R8_VIEW + rf" dest={HEX} dest_in_view_heap=0 address={HEX} address_readable=1",
    ]
    lines = [line.rstrip() for line in log.splitlines() if "Device::slot70" in line]
    if len(lines) != len(expected):
        raise AssertionError(f"{len(lines)} slot 70 lines, expected {len(expected)}:\n" + "\n".join(lines))
    for line, pattern in zip(lines, expected):
        if not re.fullmatch(pattern, line):
            raise AssertionError(f"slot 70 line does not match\n  line     {line}\n  expected {pattern}")


def check(directory, consumer, backend):
    hlsl = ("Texture2D<float> t : register(t0);\n"
            "float4 main(float4 p : SV_Position) : SV_Target {\n"
            "  return float4(t.Load(int3(uint2(p.xy) % 2, 0)), 0, 0, 1);\n}\n")
    trap_log = directory / "trap.log"
    actual = draw_texture(directory, consumer, backend, hlsl, {"X2W_TEST_ADDRESS_SRV": "1", "X2W_TRAP_LOG": str(trap_log)})
    if actual is None:
        return 77
    plane_at_x = (0, 1, 0, 0, 1, 1, 1, 0)
    expected = bytes(c for y in range(4) for x, plane in enumerate(plane_at_x)
                     for c in (16 + plane * 128 + (y % 2) * 32 + (x % 2) * 7, 0, 0, 255))
    if actual != expected:
        raise AssertionError(f"actual {actual.hex()} expected {expected.hex()}")
    check_refusal_lines(trap_log.read_text(errors="replace"))
    print(f"{backend}: ordinary view, alternating addresses, zero/interior/mismatched updates; all 32 RGBA pixels exact")
    if backend == "warp":
        one_d = directory / "texture1d"
        one_d.mkdir()
        check_1d(one_d, consumer, backend)
        placed = directory / "unregistered_1d"
        placed.mkdir()
        check_unregistered_1d(placed, consumer, backend)
    return 0


RAMP_SHADER = """Texture1D<float> ramp : register(t0);
SamplerState point_sampler : register(s0);
float4 main(float4 p : SV_Position) : SV_Target {
    return float4(ramp.SampleLevel(point_sampler, p.x / 8.0, 0), 0, 0, 1);
}
"""


def check_1d(directory, consumer, backend):
    env, args, output = prepare_draw_cache(
        directory, consumer, RAMP_SHADER, {"X2W_TEST_ADDRESS_SRV_1D": "1"},
        "srv 0 tex1d table 0 0\nsrvuse 0 sample\nsmp 0 0 0 0 0",
    )
    env.pop("X2W_PLACED_SRV_1D", None)
    env.pop("X2W_PLACED_1D_TEXTURE", None)
    for enabled in (False, True):
        if enabled:
            env["X2W_PLACED_SRV_1D"] = "1"
        output.unlink(missing_ok=True)
        result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode:
            raise RuntimeError(f"1D consumer failed ({result.returncode})\n{result.stderr[-5000:]}")
        expected = bytes(channel for _ in range(4) for x in range(8)
                         for channel in (16 + 24 * x if enabled else 0, 0, 0, 255))
        actual = output.read_bytes()
        if actual != expected:
            raise AssertionError(f"1D switch={enabled}: actual {actual.hex()} expected {expected.hex()}")
        refused = sum("Device::slot70" in line and "reason=unsupported_description" in line
                      for line in result.stderr.splitlines())
        if refused != (1 if enabled else 3):
            raise AssertionError(f"1D switch={enabled}: unexpected refusal count {refused}\n{result.stderr[-5000:]}")
        for layers in (1, 2):
            marker = f"slot 70 texture1d layers={layers} descriptor_changed={int(enabled)}"
            if marker not in result.stdout.splitlines():
                raise AssertionError(f"missing descriptor check {marker!r}")
    print(f"{backend}: 1D ramp and unset switch control; all 32 RGBA pixels exact per case")
    return 0


def check_unregistered_1d(directory, consumer, backend):
    env, args, output = prepare_draw_cache(
        directory, consumer, RAMP_SHADER, {"X2W_TEST_UNREGISTERED_1D_SRV": "1"},
        "srv 0 tex1d table 0 0\nsrvuse 0 sample\nsmp 0 0 0 0 0",
    )
    env.pop("X2W_PLACED_SRV_1D", None)
    env.pop("X2W_PLACED_1D_TEXTURE", None)
    env.pop("X2W_TEST_PLACED_1D_WRONG_WIDTH", None)
    env.pop("X2W_TEST_PLACED_1D_CONSOLE_LAYOUT", None)
    env.pop("X2W_TEST_PLACED_1D_SRGB", None)
    env.pop("X2W_SLOT70_DIAGNOSTICS", None)

    def reference(width, srgb=False):
        def red(x, y):
            texel = (2 * x + 1) * width // 16
            if not width:
                return 0
            if srgb:
                return 255 if (texel + y // 2) % 3 == 1 else 0
            return 11 + 19 * texel + 73 * (y // 2)

        return bytes(c for y in range(4) for x in range(8) for c in (red(x, y), 0, 0, 255))

    cases = (
        ("switch off", 0, {"X2W_PLACED_SRV_1D": "1"}, "texture_no_resource"),
        ("enabled", 7, {"X2W_PLACED_1D_TEXTURE": "1"}, None),
        ("wrong width", 4, {"X2W_PLACED_1D_TEXTURE": "1", "X2W_TEST_PLACED_1D_WRONG_WIDTH": "1"}, None),
        ("odd sRGB desktop rows", 7, {"X2W_PLACED_1D_TEXTURE": "1", "X2W_TEST_PLACED_1D_SRGB": "1",
                                      "X2W_SLOT70_DIAGNOSTICS": "1"}, None),
        ("unmapped layout", 0, {"X2W_PLACED_1D_TEXTURE": "1", "X2W_TEST_PLACED_1D_CONSOLE_LAYOUT": "1",
                                "X2W_SLOT70_DIAGNOSTICS": "1"}, "unsupported_desktop_format"),
    )
    for case, width, switches, refusal in cases:
        output.unlink(missing_ok=True)
        result = subprocess.run([*args, backend, output], env={**env, **switches}, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode:
            raise RuntimeError(f"{case}: consumer failed ({result.returncode})\n{result.stderr[-5000:]}")
        marker = f"unregistered 1D descriptor_changed={int(bool(width))}"
        refusals = [line for line in result.stderr.splitlines() if "Device::slot70" in line and "reason=" in line]
        if marker not in result.stdout.splitlines() or len(refusals) != (0 if width else 2) or any(
                f"reason={refusal}" not in line for line in refusals):
            raise AssertionError(f"{case}: descriptor or refusal count mismatch\n{result.stderr[-5000:]}")
        if case == "unmapped layout" and any("no console footprint mapping" not in line or "fmt=29" not in line
                                              for line in refusals):
            raise AssertionError(f"{case}: missing format and layout explanation\n{result.stderr[-5000:]}")
        if "X2W_PLACED_1D_TEXTURE" in switches:
            counts = ("1D published 1 bound 2 refused 0" if width else
                      "1D published 0 bound 0 refused 2 unsupported_desktop_format=2")
            if counts not in result.stdout:
                raise AssertionError(f"{case}: missing {counts}\n{result.stdout}")
        elif "memory SRV switches:" in result.stdout:
            raise AssertionError(f"{case}: unexpected switch census\n{result.stdout}")
        if case == "unmapped layout":
            shape = "reason=texture_1d_layout_unmapped dim=2 fmt=29 size=7x1x1 mips=1 layout=0x100 flags=0x0 calls=2"
            if shape not in result.stdout:
                raise AssertionError(f"{case}: missing layout census\n{result.stdout}")
        elif "memory SRV diagnostics" in result.stdout:
            raise AssertionError(f"{case}: unexpected diagnostics\n{result.stdout}")
        actual = output.read_bytes()
        srgb = case in {"odd sRGB desktop rows", "unmapped layout"}
        expected = reference(width, srgb)
        if actual != expected:
            raise AssertionError(f"{case}: actual {actual.hex()} expected {expected.hex()}")
        if width != 7 and actual == reference(7, srgb):
            raise AssertionError(f"{case}: unexpectedly passed the width-7 oracle")
        print(f"{backend}: placed 1D {case}; all 32 RGBA pixels exact for its CPU reference")


CUBE_SHADER = """TextureCube<float4> cube : register(t0);
SamplerState point_sampler : register(s0);
static const float3 directions[6] = {
    float3(1, 0.25, -0.125), float3(-1, 0.25, 0.125),
    float3(0.25, 1, -0.125), float3(0.25, -1, 0.125),
    float3(0.25, 0.125, 1), float3(-0.25, 0.125, -1)
};
float4 main(float4 p : SV_Position) : SV_Target {
    return cube.SampleLevel(point_sampler, directions[(uint)p.x % 6u], 0);
}
"""


def check_cube(directory, consumer, backend):
    env, args, output = prepare_draw_cache(
        directory, consumer, CUBE_SHADER, {"X2W_TEST_CUBE_SRV": "1", "X2W_TEST_FULL_TARGET": "1"},
        "srv 0 texcube table 0 0\nsrvuse 0 sample\nsmp 0 0 0 0 0",
    )
    result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode == 77:
        return 77
    if result.returncode:
        raise RuntimeError(f"consumer failed ({result.returncode})\n{result.stderr[-5000:]}")

    actual = output.read_bytes()
    assert len(actual) == 8 * 4 * 4, f"expected 32 RGBA8 pixels, got {len(actual)} bytes"
    # Mapping kind 5 to Texture2DArray breaks the first six columns.
    # Removing wrong_cube_view makes the last column read face 1.
    cases = (
        *((f"cube face {face}", bytes((31 + 29 * face, 73, 181, 255))) for face in range(6)),
        ("2D array view of six slices", bytes(4)),
        ("cube view of twelve slices", bytes(4)),
    )
    for x, (case, expected) in enumerate(cases):
        for y in range(4):
            at = 4 * (y * 8 + x)
            rgba = actual[at:at + 4]
            assert rgba == expected, f"{case} pixel ({x}, {y}): got {rgba.hex()}, expected {expected.hex()}"
    # Slot 18 records the title view. These diagnostics must come from the
    # real draw binder, including for the cube view whose dimension matches.
    mismatches = re.findall(r"srv 0: view dimension (\d+) does not match the shader; binding null", result.stderr)
    assert mismatches == ["5", "9"], f"expected array then cube mismatch, got {mismatches}\n{result.stderr[-5000:]}"
    print(f"{backend}: six cube faces, array-view null and twelve-slice null match the CPU reference")
    return 0


DESKTOP_ARRAY_SHADER = """Texture2DArray<float4> layers : register(t0);
float4 main(float4 p : SV_Position) : SV_Target {
    uint2 pixel = (uint2)p.xy;
    return layers.Load(int4(pixel.x & 1, pixel.y & 1, pixel.x % 6, 0));
}
"""


def desktop_array_reference(x, y, array_view, enabled):
    if not enabled:
        return bytes(4)
    if array_view:
        return bytes((31 + 13 * (6 + x % 6), 53 + 61 * (y % 2), 197 - 73 * (x % 2), 255))
    return bytes(((19 if y < 2 else 31) + 13 * (x % 6), 53, 197, 255))


def check_desktop_arrays(directory, consumer, backend):
    for array_view in (False, True):
        case = "array" if array_view else "cube"
        work = directory / case
        work.mkdir()
        modes = {"X2W_TEST_DESKTOP_ARRAY_SRVS": "1", "X2W_TEST_FULL_TARGET": "1"}
        if array_view:
            modes["X2W_TEST_DESKTOP_ARRAY_SLICES"] = "1"
            shader, binding = DESKTOP_ARRAY_SHADER, "srv 0 tex2da table 0 0\nsrvuse 0 texel"
        else:
            shader, binding = CUBE_SHADER, "srv 0 texcube table 0 0\nsrvuse 0 sample\nsmp 0 0 0 0 0"
        env, args, output = prepare_draw_cache(work, consumer, shader, modes, binding)
        if not array_view:
            env.pop("X2W_TEST_DESKTOP_ARRAY_SLICES", None)
        for enabled in ((True,) if array_view else (False, True)):
            env.pop("X2W_SLOT70_DESKTOP_ARRAYS", None)
            if enabled:
                env["X2W_SLOT70_DESKTOP_ARRAYS"] = "1"
            output.unlink(missing_ok=True)
            result = subprocess.run([*args, backend, output], env=env, capture_output=True, text=True)
            print(result.stdout, end="")
            if result.returncode == 77:
                return 77
            if result.returncode:
                raise RuntimeError(f"{case}: consumer failed ({result.returncode})\n{result.stderr[-5000:]}")
            reasons = re.findall(r"Device::slot70 [^\n]*reason=(\w+)", result.stderr)
            count = 0 if array_view else 2 if enabled else 4
            assert reasons == ["unsupported_description"] * count, (case, enabled, reasons, result.stderr[-5000:])
            actual = output.read_bytes()
            assert len(actual) == 8 * 4 * 4, f"expected 32 RGBA8 pixels, got {len(actual)} bytes"
            for y in range(4):
                for x in range(8):
                    expected = desktop_array_reference(x, y, array_view, enabled)
                    at = 4 * (y * 8 + x)
                    rgba = actual[at:at + 4]
                    assert rgba == expected, (
                        f"{case} switch={enabled} pixel ({x}, {y}): got {rgba.hex()}, expected {expected.hex()}"
                    )
    print(f"{backend}: desktop array cube faces and slices 6–11 match the CPU reference; refusal controls passed")
    print("Mutant: unrounded footprint stride makes cube faces 1 through 5 read zero padding.")
    return 0


if __name__ == "__main__":
    consumer, backend, *mode = sys.argv[1:]
    checks = {(): check, ("cube",): check_cube, ("desktop-arrays",): check_desktop_arrays}
    if tuple(mode) not in checks:
        raise SystemExit("expected consumer backend [cube|desktop-arrays]")
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(checks[tuple(mode)](Path(temporary), consumer, backend))
