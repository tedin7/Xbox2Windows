# Explicit-LOD 2D gather in the direct DXIL frontend

The direct C++ frontend lowers gfx1013 `image_gather4_l` for a fixed
table-bound 2D SRV. The instruction reads U, V and explicit LOD from three
contiguous VGPRs and returns the four taps of one component. The supported
constant sampler has normalized clamp addressing, zero minimum LOD, point
mip filtering and no mip bias. Unsupported image and sampler forms refuse.
The direct path (`image_io`) and the HLSL compute emitter (`image`) treat MIMG word 1 bits 29:26 as reserved; bit 25 is the top bit of the sampler index.

The lowering follows the existing startup HLSL implementation. Clamp LOD
to the sampler's maximum, round with `floor(lod + 0.5)`, and cap the index
at the last visible mip. Both dimension queries and texel loads use
view-relative mip indices. The four loads use the lower-left, lower-right,
upper-right, upper-left order of the gather footprint. Nonfinite coordinates
or LOD produce zeroes in the active destination lanes.
[AMD RDNA 2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
[Microsoft Texture2D GetDimensions](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-texture2d-getdimensions)

```powershell
cmake --build --preset dev --target compile_dxil offscreen_check
ctest --preset dev --no-tests=error -j 1 -R '^native_gather_lod_' -V
```

The compile controls check for four texture loads and two dimension queries.
Source-built pixel and compute programs cover all four channels, both sides
of the half-mip threshold, sampler and view limits, clamped coordinates and
nonfinite operands. The readback covers a two-mip SRV starting at resource
mip 1 and a full view. Four compute lanes select different LODs in a single
dispatch. A shader forced to mip zero fails the higher-mip comparison.
Compute replay requires shader model 6.6 and wave32. Guest implicit mip
selection is tracked separately in #1802.
