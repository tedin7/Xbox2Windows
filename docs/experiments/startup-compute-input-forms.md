# Translated compute input forms

Set `X2W_COMPUTE_INPUT_FORMS=1` to enable converted 2D target views,
CPU-readable heap index words outside list upload memory, and raw input
ranges padded through the next dword inside their original allocation.
The switch defaults to off. Compute totals report each input form used by
recorded dispatches.

Converted UAV and RTV views bind one mip and retain their array slice
range. Converted depth views bind the readable depth plane of a matching
typeless format family. A converted input may share a texture with an
output when their subresources are disjoint. The dispatch refuses overlapping
views and heap words written by a recorded GPU operation.

Run the WARP checks on Windows after building the development preset.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_dispatch_(texture|heap_word|input)_warp$'
```

The fixtures compare D3D12 readback with CPU references. Their controls cover
the disabled switch, overlapping views, GPU-written heap words, and padding
past a buffer end. The wrong mip, zero heap index, and shortened raw view each
produce a different readback from the required reference.
