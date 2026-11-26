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

Uploaded texels: the "fill" cases put a known 16-bit texel pattern into the title's memory of a 64x64 texture (no clear)
and read the sampled texels back. Without the render-target flag the texture keeps its own format on every adapter, so
every texel must decode exactly. With the render-target flag a texture the adapter cannot render to is created as
R8G8B8A8_UNORM (stand-in) and the title's 16-bit texels are NOT converted (packed_target_format.h says so): that case
reports how many texels still read back right instead of asserting, because it measures the shortcut, it does not
bless it. Not covered: unordered-access writes, and a title copy from an upload buffer into a stand-in.
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


def texel(x, y):
    """The 16-bit texel startup_stream_draw.cpp (packed_fill_texel) writes at (x, y)."""
    return (x * 257 + y * 131 + 0x1234) & 0xffff


def decoded(texture_format, t):
    """The RGBA8 value a sampler returns for that texel in the 16-bit layout (DXGI bit order, low bits first: B, G, R, A)."""
    def to8(v, bits):
        return round(v * 255 / ((1 << bits) - 1))
    if texture_format == B5G5R5A1:
        return (to8((t >> 10) & 31, 5), to8((t >> 5) & 31, 5), to8(t & 31, 5), 255 if t >> 15 else 0)
    return (to8((t >> 8) & 15, 4), to8((t >> 4) & 15, 4), to8(t & 15, 4), to8(t >> 12, 4))


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

    def uploaded(name, texture_format, flags):
        """Known 16-bit texels in title memory, sampled back. Returns (created format, texels that read back right)."""
        if image.exists():
            image.unlink()
        run = subprocess.run(base + [f"packed:{texture_format}:{flags:x}:{texture_format}:1"], env=env, capture_output=True, text=True)
        report = [line for line in run.stdout.splitlines() if line.startswith("packed: ")]
        assert len(report) == 1, f"{name}: {run.returncode} {run.stdout!r} {run.stderr[-800:]!r}"
        created = int(report[0].rsplit(" ", 1)[1])
        if run.returncode or "-> drawn;" not in run.stderr:
            raise RuntimeError(f"{name}: draw failed on {backend} ({run.returncode})\n{run.stderr[-2000:]}")
        pixels = image.read_bytes()
        assert len(pixels) == 64 * 64 * 4, f"{name}: {len(pixels)} bytes"
        distinct = len({tuple(pixels[i:i + 4]) for i in range(0, len(pixels), 4)})
        uploads = [line for line in run.stderr.splitlines() if "texture upload" in line or "texture not uploaded" in line]
        print(f"{backend} {name}: {distinct} distinct colours; upload log lines: {uploads[:3] or 'NONE (no upload was recorded)'}")
        good = 0
        for y in range(64):
            for x in range(64):
                got = tuple(pixels[(y * 64 + x) * 4:(y * 64 + x) * 4 + 4])
                good += all(abs(a - e) <= 1 for a, e in zip(got, decoded(texture_format, texel(x, y))))
        return created, good

    paths = set()
    for name, resource_format, flags in (("B5G5R5A1 render target with unordered access", B5G5R5A1, RT | UAV),
                                         ("B4G4R4A4 render target", B4G4R4A4, RT),
                                         ("B5G6R5 render target", B5G6R5, RT)):
        created, got = rendered(name, resource_format, flags, {resource_format, RGBA8})
        paths.add("stand-in" if created == RGBA8 else "as described")
    # Uploaded 16-bit texels. No render-target flag: never substituted, so every texel must decode exactly.
    for name, texture_format in (("B5G5R5A1 uploaded texels", B5G5R5A1), ("B4G4R4A4 uploaded texels", B4G4R4A4)):
        created, good = uploaded(name, texture_format, 0)
        assert created == texture_format, f"{name}: created as {created}"
        assert good == 64 * 64, f"{name}: only {good} of 4096 texels read back as the 16-bit layout decodes them"
        print(f"{backend} {name}: all 4096 texels exact (format {created})")
    # With the render-target flag: native on adapters that can render to it (must be exact), a stand-in elsewhere
    # (title texels are not converted by design; reported as a finding, not asserted).
    for name, texture_format in (("B5G5R5A1 render-target uploaded texels", B5G5R5A1), ("B4G4R4A4 render-target uploaded texels", B4G4R4A4)):
        created, good = uploaded(name, texture_format, RT)
        if created == texture_format:
            assert good == 64 * 64, f"{name}: native format but only {good} of 4096 texels exact"
            print(f"{backend} {name}: native format, all 4096 texels exact")
        else:
            print(f"{backend} {name}: FINDING stand-in format {created}: {good} of 4096 uploaded texels read back as the 16-bit layout decodes them")
    # Control that draws: never substituted.
    plain, _ = rendered("control, R8G8B8A8 render target", RGBA8, RT, {RGBA8})
    assert stored(B5G5R5A1) != stored(RGBA8) != stored(B4G4R4A4) and stored(B5G6R5) != stored(RGBA8)
    # Controls that must be refused.
    refused("B5G5R5A1 views of a plain R8G8B8A8 texture", RGBA8, RT, B5G5R5A1)
    refused("B5G5R5A1 views of a B4G4R4A4 render target", B4G4R4A4, RT, B5G5R5A1)
    print(f"startup_packed_target_draw: PASS on {backend} (packed targets drawn {' and '.join(sorted(paths))} on this adapter; "
          f"1 drawn control, 2 refused view pairs; unordered-access writes and CPU-written texels not covered)")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
