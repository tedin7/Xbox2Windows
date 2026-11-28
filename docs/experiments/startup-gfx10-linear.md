# Slot-70 textures in the console linear layout (0x100)

Device slot 70 publishes a texture by address. A description whose `Layout` is `0x100` (the swizzle
mode word for `ADDR_SW_LINEAR`) was refused with `reason=unsupported_description` unless it was a
single texel, because the extent of its rows was not established. This note establishes it from AMD
addrlib and reads such textures like the tiled console layouts.

## Layout

`Gfx10Lib::HwlComputeSurfaceInfoLinear` (Mesa 25.1.0, commit
[5c142e46](https://gitlab.freedesktop.org/mesa/mesa/-/blob/5c142e46f3f6e752ed745fd48912ebb8fad67145/src/amd/addrlib/src/gfx10/gfx10addrlib.cpp),
`src/amd/addrlib`, MIT) gives the linear mode:

- The pitch of a mip is its width in elements, rounded up at each level (`GetMipSize`), then aligned
  to `256 / element bytes` elements: 256 bytes.
- The mip rows are packed with no tail block. The mips run from the smallest to the largest, so the
  largest mip comes last.
- A slice is the sum of its mips, and the array slices follow one another.

`src/startup_stubs/gfx10_layout.h` restates this rule for two-dimensional textures. The rest of the
slot-70 console path is unchanged: view checks, the XG holds and the upload as desktop footprint rows.
One-dimensional and three-dimensional linear textures are still refused.

## Results

Build: `slot70/linear-layout` on main 6ee878d8, dev preset (debug layer on). Windows 11 Pro 10.0.26200,
NVIDIA GeForce RTX 5090 (driver 32.0.16.1088) and Microsoft WARP.

| Test | Result |
| --- | --- |
| `startup_gfx10_layout` | PASS: 13 round trips (4 of them linear, including a 2-slice R8 chain and a 3-slice BC1 chain) and one hand-worked linear chain (a 100x10 RGBA16F texture with 3 mips: 32x3 elements at offset 0, 64x5 at 768, 128x10 at 3328, 13568 bytes a slice) |
| `startup_gfx10_layout_linear_order_control` | expected failure: with the largest mip first, all 6 offset and address checks fail |
| `startup_gfx10_layout_linear_addrlib` | PASS: 338,763 sizes and element addresses of 11 linear textures (R8 to RGBA32F, BC1, BC7, odd sizes, up to 9 mips and 3 slices) match `Addr2ComputeSurfaceInfo` and `Addr2ComputeSurfaceAddrFromCoord` with `ADDR_SW_LINEAR`; skips (77) without addrlib |
| `startup_address_console_{warp,hardware}` | PASS: 4 new linear draws are exact on all 65,536 pixels. They are RGBA8 100x40 (400-byte rows padded to 512); its 4-mip chain viewed at mip 0 and at mip 2; and BC1 200x96 (50 blocks padded to 64). The 7 earlier draws and the 3 held/late cases are unchanged |
| `startup_address_console_linear_unpadded_{warp,hardware}` | PASS (control): the same RGBA8 100x40 texture with rows written at 400 bytes is published, and 63,488 of 65,536 pixels differ |

Without the `gfx10_layout.h` and `address_texture.h` change (main 6ee878d8), the first linear draw
reads the null view: pixel (0,0) is `00000000`, but `010300ff` was expected. There is also no
`published console layout` line in the log.

Commands (PowerShell, at the repository root, after `. .\tools\dev.ps1`; addrlib fetched with
`tools/fetch_addrlib.py`):

```powershell
cmake --preset dev -DXBOX2WINDOWS_ADDRLIB_DIR=<mesa>/src/amd/addrlib
cmake --build --preset dev
ctest --preset dev --no-tests=error -R '^startup_gfx10_layout'
ctest --preset dev --no-tests=error -R '^startup_address_console_(warp|hardware|linear_unpadded_(warp|hardware))$' -V
```

## Open

- The rule is addrlib's desktop GFX10 rule. That the console stores `0x100` textures with the same
  256-byte pitch is inferred from the layout word naming `ADDR_SW_LINEAR`; no console hardware
  measurement backs it.
- One-dimensional linear textures are covered in [startup-gfx10-linear-1d.md](startup-gfx10-linear-1d.md).
