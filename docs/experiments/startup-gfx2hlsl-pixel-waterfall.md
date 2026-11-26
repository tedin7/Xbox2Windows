# Pixel descriptor waterfall

The pixel translator can collapse a 64-bit EXEC waterfall around a 2D
`image_sample` when the lane key is masked immediately before the loop and
indexes a bounded SRV range at the start of a pixel-visible root table. The
proof checks that the equality compare selects one key, the closing XOR or WREXEC
removes those lanes, and the restored EXEC and subsequent instructions cannot
observe the loop's scratch state. The sample runs in straight-line code with
explicit screen-space gradients and a nonuniform texture index. The manifest
uses consecutive SRV slots and the existing draw binding path.

Previously translated shaders take the existing path. Unbounded heaps and
other loop shapes retain their refusals.

```text
python -B tests/startup_gfx2hlsl_pixel_waterfall_check.py
ctest --preset dev --no-tests=error -V -R '^startup_gfx2hlsl_pixel_waterfall_(controls|warp)$'
```

The Python control translates two adjacent waterfalls and checks their
bindings, sample calls, and refusal cases. The WARP fixture draws a 2×2 quad
against two 4×4 textures and compares all four pixels with a CPU reference.
Each loop selects a different descriptor at adjacent pixels and samples mip 1.
The fixed-index and forced-mip-zero mutants each must match their own CPU
reference and differ from the correct reference at every pixel.

On Linux at base revision `34da218c7` with the waterfall change applied,
the Python command above passed. Windows WARP readback is pending.
