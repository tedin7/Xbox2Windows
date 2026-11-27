"""Create 16-bit packed render targets through the startup device wrapper, render into them and read them back.

usage: startup_packed_target_draw_check.py <startup_stream_consumer> <dxc> <warp|hardware>

Each case creates a 64x64 texture through device slot 133, clears it to (0.45, 0, 0.8, 1) through a
render-target view, samples it through a shader-resource view in a full-target draw and reads every
pixel of that draw back. The consumer reports the adapter's support bits and the texture format
src/startup_stubs/packed_target_format.h selects for them, so the expected pixel follows the path
this adapter takes:

  R8G8B8A8_UNORM stand-in  (115, 51, 204, 255)   8 bits a channel: not quantized like the console's
  B5G5R5A1_UNORM           (115, 49, 206, 255)
  B4G4R4A4_UNORM           (119, 51, 204, 255)
  B5G6R5_UNORM             (115, 53, 206, 255)

Control: R8G8B8A8_UNORM with its own view format must draw its own values. A packed view of a plain
R8G8B8A8_UNORM texture, and a B5G5R5A1 view of a B4G4R4A4 texture, must be refused by name and leave the draw without the clear
colour.

Not covered: unordered-access writes, and texel data written or read by the CPU in the 16-bit layout.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_stream_check import container, gfx2hlsl

B5G6R5, B5G5R5A1, RGBA8, B4G4R4A4 = 85, 86, 28, 115
RT, UAV = 0x1, 0x4
# The six-bit green channel in the B5G6R5 control uses an exact endpoint.
CLEAR = (0.45, 0.0, 0.8, 1.0)
BITS = {RGBA8: (8, 8, 8, 8), B5G5R5A1: (5, 5, 5, 1), B4G4R4A4: (4, 4, 4, 4), B5G6R5: (5, 6, 5, 0)}


def stored(texture_format):
    """The RGBA8 value read from a texture of that format after the clear."""
    out = []
    for value, bits in zip(CLEAR, BITS[texture_format]):
        steps = (1 << bits) - 1
        out.append(255 if bits == 0 else round(round(value * steps) / steps * 255))
    return tuple(out)


def check_placeholder(work, consumer, dxc, vs, ps, root, image, env, key):
    cache = Path(env["X2W_SHADER_CACHE"])
    sources = {
        "vs": """float4 main(uint id : SV_VertexID) : SV_Position {
            float2 corner = float2(id == 1 ? 4 : -2, id == 2 ? -4 : 2);
            return float4(corner, 0.5, 1);
        }""",
        "ps": """float4 main() : SV_Target { return float4(1, 1, 0, 1); }""",
    }
    for stage, source in sources.items():
        shader = work / f"placeholder.{stage}.hlsl"
        shader.write_text(source, encoding="ascii")
        subprocess.run([dxc, "-T", f"{stage}_6_0", "-E", "main", "-Fo",
                        cache / f"{key}.{stage}.cso", shader], check=True)

    reference = bytes((255, 255, 0, 255)) * (8 * 8)
    clear = bytes((0, 0, 255, 255)) * (8 * 8)
    expected_report = ["x2w-render placeholder total: draws 1",
                       f"x2w-render placeholder key {key}: draws 1, targets 8x8 format 28"]
    manifest = cache / f"{key}.txt"
    for marked, mode in ((False, "visible"), (True, "visible"), (True, "clipped")):
        marker = "placeholder synthetic pixel refusal\n" if marked else ""
        manifest.write_text("udsrc 0 rect\n" + marker, encoding="ascii")
        image.unlink(missing_ok=True)
        result = subprocess.run([consumer, vs, ps, root, "warp", image],
                                env=dict(env, X2W_TEST_PLACEHOLDER_DRAW=mode), capture_output=True, text=True)
        assert result.returncode == 0, f"{mode}: {result.returncode}\n{result.stderr[-2000:]}"
        lines = [line for line in result.stdout.splitlines() if line.startswith("x2w-render placeholder ")]
        assert lines == (expected_report if marked else []), lines
        pixels = image.read_bytes()
        if mode == "clipped":
            assert pixels == clear and pixels != reference, "clipped draw must fail the CPU colour comparison"
        else:
            assert pixels == reference, f"{mode}, marked={marked}: unexpected RGBA8 readback"
    print("startup placeholder WARP: marked and unmarked 8x8 readbacks matched; clipped draw failed the comparison")


def check(work, consumer, dxc, backend):
    cache = work / "cache"
    cache.mkdir()
    vs, ps, root, image, unused = (work / name for name in ("vs", "ps", "root", "pixels", "unused"))
    vs.write_bytes(container(1))
    ps.write_bytes(container(0))
    unused.write_bytes(b"")
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache))
    env.setdefault("X2W_D3D12_DEBUG", "0")
    base = [consumer, vs, ps, root, backend, image, unused, "64", "64"]
    bootstrap = subprocess.run(base + ["linear"], env=dict(env, X2W_TEST_TRUNCATED_ONLY="1"))
    if bootstrap.returncode == 77:
        print(f"{backend}: D3D12 feature level 12 unavailable")
        return 77
    bootstrap.check_returncode()
    key = f"{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes()):016x}"
    (cache / f"{key}.txt").write_text("udsrc 0 rect\nsrv 0 tex2d table 0 0\n", encoding="ascii")
    sources = {
        "vs": """float4 main(uint id : SV_VertexID) : SV_Position {
            return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
        }""",
        "ps": """Texture2D<float4> tex : register(t0);
        float4 main(float4 position : SV_Position) : SV_Target {
            return tex.Load(int3(int2(position.xy), 0));
        }""",
    }
    for stage, source in sources.items():
        shader = work / f"{stage}.hlsl"
        shader.write_text(source, encoding="ascii")
        subprocess.run([dxc, "-T", f"{stage}_6_0", "-E", "main", "-Fo", cache / f"{key}.{stage}.cso", shader], check=True)

    def draw(name, resource_format, flags, view_format):
        if image.exists():
            image.unlink()
        result = subprocess.run(base + [f"packed:{resource_format}:{flags:x}:{view_format}"], env=env, capture_output=True, text=True)
        report = [line for line in result.stdout.splitlines() if line.startswith("packed: ")]
        assert len(report) == 1, f"{name}: {result.returncode} {result.stdout!r} {result.stderr[-800:]!r}"
        expected_format = int(report[0].rsplit(" ", 1)[1])
        pixels = image.read_bytes() if image.exists() else b""
        colours = {tuple(pixels[i:i + 4]) for i in range(0, len(pixels), 4)}
        return result, expected_format, len(pixels), colours

    def rendered(name, resource_format, flags, allowed):
        """A case whose views must work: every pixel is the clear colour as the created format stores it."""
        result, created, size, colours = draw(name, resource_format, flags, resource_format)
        assert created in allowed, f"{name}: the helper selected format {created}"
        if result.returncode or "-> drawn;" not in result.stderr:
            raise RuntimeError(f"{name}: draw failed on {backend} ({result.returncode})\n{result.stderr[-2000:]}")
        expected = stored(created)
        assert size == 64 * 64 * 4 and len(colours) == 1, f"{name}: {size} bytes, colours {sorted(colours)[:4]}"
        got = next(iter(colours))
        assert all(abs(a - e) <= 1 for a, e in zip(got, expected)), f"{name}: read {got}, expected {expected} for format {created}"
        stand_in = created != resource_format
        logged = "created as format 28" in result.stderr
        assert logged == stand_in, f"{name}: stand-in {stand_in}, logged {logged}"
        assert "view skipped: packed format" not in result.stderr, f"{name}: a view was refused"
        print(f"{backend} {name}: texture format {created} ({'stand-in' if stand_in else 'as described'}), read {got}")
        return created, got

    def refused(name, resource_format, flags, view_format):
        """A packed view of a texture that is not that format and does not stand in for it: both views are skipped."""
        result, created, size, colours = draw(name, resource_format, flags, view_format)
        skips = [line for line in result.stderr.splitlines() if "view skipped: packed format %d" % view_format in line]
        assert len(skips) == 2, f"{name}: expected the render-target view and the shader-resource view refused, got {skips}\n{result.stderr[-1500:]}"
        assert "binding null" in result.stderr, f"{name}: the draw did not bind a null texture for the refused view\n{result.stderr[-1500:]}"
        would_be = stored(created)
        assert not any(all(abs(a - e) <= 1 for a, e in zip(colour, would_be)) for colour in colours), (
            f"{name}: the clear colour was drawn through a refused view: {sorted(colours)[:4]}")
        print(f"{backend} {name}: both views refused ({skips[0].split('view skipped: ')[1]})")

    paths = set()
    for name, resource_format, flags in (("B5G5R5A1 render target with unordered access", B5G5R5A1, RT | UAV),
                                         ("B4G4R4A4 render target", B4G4R4A4, RT),
                                         ("B5G6R5 render target", B5G6R5, RT)):
        created, got = rendered(name, resource_format, flags, {resource_format, RGBA8})
        paths.add("stand-in" if created == RGBA8 else "as described")
    # Control that draws: never substituted.
    plain, _ = rendered("control, R8G8B8A8 render target", RGBA8, RT, {RGBA8})
    assert stored(B5G5R5A1) != stored(RGBA8) != stored(B4G4R4A4) and stored(B5G6R5) != stored(RGBA8)
    # Controls that must be refused.
    refused("B5G5R5A1 views of a plain R8G8B8A8 texture", RGBA8, RT, B5G5R5A1)
    refused("B5G5R5A1 views of a B4G4R4A4 render target", B4G4R4A4, RT, B5G5R5A1)
    print(f"startup_packed_target_draw: PASS on {backend} (packed targets drawn {' and '.join(sorted(paths))} on this adapter; "
          f"1 drawn control, 2 refused view pairs; unordered-access writes and CPU-written texels not covered)")
    if backend == "warp":
        check_placeholder(work, consumer, dxc, vs, ps, root, image, env, key)
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
