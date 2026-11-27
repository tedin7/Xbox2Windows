# Constant-lane writes to vector registers

Pixel, vertex and small compute shaders can write a uniform numeric word to a
constant lane of a VGPR used by ordinary vector instructions. The selected
invocation takes the new word even when its EXEC bit is clear. Other invocations
keep the old word. Graphics uses the desktop wave lane index. Compute groups
of up to 32 threads use the x-major logical lane index, as in the compute
`v_mbcnt` lowering. Lane selectors above 31 require a known guest wave size.
Mixed-use constant-lane reloads and writes in divergent control flow retain
their refusals.

Run the pixel draw and compute dispatch on D3D12 WARP:

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_writelane_vector_(controls|warp)$' -V
```

The pixel test reads the lane index alongside the translated color so the CPU
reference can use the lane assignment observed in the draw. The compute test
checks all output words, including the untouched guard words, for a 4×2 group.
It clears EXEC for the selected lane before the write and restores EXEC before
the multiply and store. Both tests repeat readback with the assignment moved
one lane higher and require a mismatch.

The source-built fixture encodings can be checked independently:

```bash
python3 -B tests/startup_gfx2hlsl_writelane_vector_check.py binary llvm-mc-15
```
