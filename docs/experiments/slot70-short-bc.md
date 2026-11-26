# Slot 70 BC surfaces with short tiled backing

`X2W_SLOT70_DETILE_MORE=1` lets slot 70 publish single-mip 2D BC3_UNORM
and BC7_UNORM textures in layout `0x105` when the readable range ends
after the final addressed block. The existing 4 KiB detiler copies those
blocks into the desktop texture's upload footprint. The padded size still
applies to the allocation checks. An unreadable addressed block still
refuses the view. The switch defaults off.

The synthetic WARP test places 6,864 bytes for a 128×4 BC3 surface and
2,768 bytes for a 64×4 BC7 surface immediately before a reserved page.
Their padded sizes are 8,192 and 4,096 bytes. A pixel shader loads each
texel and the test compares the 512×128 RGBA readback to the CPU image.
The switch-off and missing-last-block controls require
`reason=texture_unreadable` and zero readback. Exchanging two occupied
micro-tiles must produce the first pixel mismatch at (16, 0).

Run on a configured Windows test machine.

```powershell
cmake --build --preset dev --target startup_stream_consumer
ctest --preset dev --no-tests=error -V -R '^startup_address_short_bc_warp$'
```
