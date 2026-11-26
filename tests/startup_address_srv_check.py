"""Publish two placed R8 textures by address, render their texels, and test rejected updates."""
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
    for replace in (False, True):
        run_directory = directory / ("null" if replace else "preserve")
        run_directory.mkdir()
        trap_log = run_directory / "trap.log"
        actual = draw_texture(run_directory, consumer, backend, hlsl, {
            "X2W_TEST_ADDRESS_SRV": "1", "X2W_TRAP_LOG": str(trap_log),
            "X2W_SLOT70_REFUSED_NULL": str(int(replace)),
        })
        if actual is None:
            return 77
        plane_at_x = (0, 1, 0, 0, 1, 1, 1, 0)
        expected = bytes(c for y in range(4) for x, plane in enumerate(plane_at_x)
                         for c in (0 if replace and x in (3, 5, 6)
                                   else 16 + plane * 128 + (y % 2) * 32 + (x % 2) * 7, 0, 0, 255))
        if actual != expected:
            raise AssertionError(f"X2W_SLOT70_REFUSED_NULL={int(replace)} actual {actual.hex()} expected {expected.hex()}")
        check_refusal_lines(trap_log.read_text(errors="replace"))
        print(f"{backend}: X2W_SLOT70_REFUSED_NULL={int(replace)}, all 32 RGBA pixels exact")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
