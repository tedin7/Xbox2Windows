# Read-only depth and stencil in translated draws

Set `X2W_READONLY_DEPTH_DRAWS=1` to enable this path. The desktop pipeline
keeps the title's depth and stencil tests and masks writes to each read-only
plane. The two view bits distinguish pipeline cache entries. The sampled plane
keeps `DEPTH_READ` combined with shader read states. Render totals report the
number of recorded translated draws that needed a write mask.

The WARP fixture uses an 8×4 D32 depth texture and a source-built quad. It
clears each depth column through a writable view, then binds a read-only view
and reads the same texture in the pixel shader. The two colour channels read
paired columns so the covered pixels expose all eight depth values. A second
draw uses the same pipeline and shader with a writable view of a separate
depth texture initialized to the same gradient. This makes the two draws use
the same pipeline cache while isolating the tested write. The CPU reference
compares every colour byte and every depth bit. The indirect switch-off case
requires the legacy refusal, clear colour, and intact depth. Each run requires
zero debug-layer errors and reports a capability skip if the layer is absent.

Run on Windows with FXC and the configured Python environment.

```powershell
cmake --build --preset dev --target startup_draw_depth_target startup_indirect_draw_indexed
ctest --preset dev --no-tests=error -V -R '^startup_draw_readonly_depth_(direct|indirect|indirect-off)_warp$'
```

For the cache-key mutant, use a disposable checkout of the same revision.
Delete just the line containing `if (readonly_planes) key = fnv_bytes` in
`src/startup_stubs/d3d12_draw.inl`. Rebuild the same targets and rerun the
same filter. The direct and indirect switch-on tests must fail with
`FAIL writable: correct colour, depth remained at the pre-pass gradient`.
The indirect switch-off case must still pass. A compile error, debug-layer
error, or skipped test does not satisfy the mutant check.
