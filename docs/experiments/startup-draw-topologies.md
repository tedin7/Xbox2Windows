# Startup draw topologies

Set `X2W_DRAW_TOPOLOGIES=1` before the process starts. The switch enables point
lists and line lists and strips on the translated draw paths. It also enables
triangle strips on the translated indirect path and writable depth-only
triangle draws for a native stream pipeline with no color outputs. With the
switch unset, draw acceptance is unchanged. Refusal diagnostics run in both
modes.

The WARP fixtures compare every pixel of a 16×16 target against hand-written
coverage masks derived from the [Direct3D rasterization rules](https://learn.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-rasterizer-stage-rules). The indirect fixture also compares D16 readback against the
encoding of 0.25 and checks that the separate color target stays at its clear
value. The switch-off case checks that indirect topology 5 leaves the target
clear and emits a topology refusal. The switch-on cases check patch refusal and
deduplication of the empty-target diagnostic.

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_(stream_topologies|indirect_topologies(_off)?)_warp$'
```

Apply the cache-key mutant with
`git apply tests/startup_draw_topology_cache_mutant.patch`. It removes the
`topology_type` hash from `pipeline_for`. Build the shim on the Windows test
machine and rerun the same command. Both switch-on tests must fail.
Each draws the line list immediately after the triangle strip using the same
guest PSO and attachment format. The switch-off control must still pass.
Reverse the patch with `git apply -R tests/startup_draw_topology_cache_mutant.patch`
and rebuild before recording the final result.

The WARP results and mutant failure are pending Windows execution. The author
checked the coverage masks against the fixture geometry on Linux.
