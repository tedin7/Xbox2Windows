# Slot-70 one-dimensional textures in the console linear layout (0x100)

[startup-gfx10-linear.md](startup-gfx10-linear.md) reads two-dimensional `0x100` textures through
device slot 70. A one-dimensional `0x100` description was still refused with
`reason=unsupported_description`, because `gfx10::supports` took only 2D textures and
`address_view_fits` had no 1D view.

## Layout

`Gfx10Lib::HwlComputeSurfaceInfoLinear` (Mesa 25.1.0, `src/amd/addrlib`, MIT) lays out a 1D surface
with the same rule as a 2D one, one row per mip: each mip is padded to 256 bytes and the mips are
stored from the smallest up. It returns `ADDR_INVALIDPARAMS` for a 1D surface taller than one row.
A 300-texel RGBA8 texture with 4 mips (300, 150, 75 and 38 texels, rounded up at each level) has
mip 3 at 0, mip 2 at 256, mip 1 at 768 and mip 0 at 1536.

The change:

- `gfx10_layout.h`: `supports` takes a 1D texture in the linear mode with a height of one and an
  element of one texel. Block-compressed formats, 1D arrays and the tiled modes stay refused for 1D.
- `address_texture.h`: a 1D array is refused, since the draw uploads a 1D texture only when it is not
  an array (`d3d12_draw.inl`). `address_view_fits` takes a `TEXTURE1D` view of a 1D texture that fits
  the mip range with no LOD clamp.
- `d3d12_memory_srv.inl`: the desktop format check asks for `D3D12_FORMAT_SUPPORT1_TEXTURE1D` for a
  1D texture.

## Results

Build: `slot70/linear-1d`, stacked on `slot70/linear-layout` (ce2d6f50), dev preset (debug layer on).
Windows 11 Pro 10.0.26200, NVIDIA GeForce RTX 5090 (driver 32.0.16.1088) and Microsoft WARP.

| Test | Result |
| --- | --- |
| `startup_gfx10_layout` | PASS: 15 round trips (2 new 1D linear shapes: RGBA8 sRGB 300 with 9 mips, R8 4096 with 13 mips), 15 refusals (new: 1D in `0x105`, 1D two rows high, 1D BC1, a 1D array), 22 address descriptions (new: a 1D texture is accepted, a 1D array and a desktop-layout 1D texture are refused), 33 views (new: the whole 1D chain, its last mip, past the chain, an LOD clamp, a 1D view of a 2D texture and a 2D view of a 1D texture) |
| `startup_gfx10_layout_linear_addrlib` | PASS: 349,547 sizes and element addresses match addrlib, now with three 1D textures (`ADDR_RSRC_TEX_1D`: RGBA8 sRGB 300 with 9 mips, R8 4096 with 13 mips, RGBA16F 1000 with 10 mips) |
| `startup_address_console_{warp,hardware}` | PASS: 3 new 1D draws through a `Texture1D` view are exact on all 65,536 pixels: RGBA8 300 with one mip, and its 4-mip chain viewed at mip 0 and at mip 2 |
| `startup_address_console_linear_unpadded_{warp,hardware}` | PASS (control): the 4-mip 1D chain with its mips packed without padding (mip 2 at 152 instead of 256) is published and all 65,536 pixels of the mip-2 draw differ; the 2D case is unchanged (63,488 differ) |

Without the source change (ce2d6f50), the first 1D draw reads the null view: pixel (0,0) is
`00000000`, but `010300ff` was expected.

Commands (PowerShell, at the repository root, after `. .\tools\dev.ps1`):

```powershell
cmake --preset dev -DXBOX2WINDOWS_ADDRLIB_DIR=<mesa>/src/amd/addrlib
cmake --build --preset dev
ctest --preset dev --no-tests=error -R '^startup_gfx10_layout'
ctest --preset dev --no-tests=error -R '^startup_address_console_(warp|hardware|linear_unpadded_(warp|hardware))$' -V
```

## Open

- As for 2D, the rule is addrlib's desktop GFX10 rule; that console memory uses it for `0x100` is
  inferred from the layout word, not measured on console hardware.
- 1D arrays and 1D textures in the tiled modes stay refused.
