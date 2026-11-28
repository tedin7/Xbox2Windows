# Slot 57 translated mesh runtime fixture

`startup_mesh57_runtime_warp` and `startup_mesh57_runtime_hardware` publish a
source-built request, translate it with DXC, and submit the resulting mesh and
pixel stages through the guest indirect dispatch path. The readback is compared
with the Python rasterizer. The group shapes `(2,1,1)` and `(1,2,1)` must both
match the two-group image. A `(1,1,1)` command must match the group-zero image
and differ from the two-group image; omitted root constants must leave the
target clear. Missing and mismatched manifests are refusal controls.

Run on a configured x64 Windows release build with a physical adapter:

```powershell
ctest --preset release -j1 --no-tests=error -R '^startup_mesh57_runtime_hardware$' -V
```

Use `^startup_mesh57_runtime_warp$` for WARP. A return of 77 is a capability
skip, not a pass. Record the commit, adapter, driver, Windows build and debug
layer status with the result. These fixtures do not establish application startup.

For a manual negative control, change the mesh argument guard to write `g.x`
instead of `n` into the first dispatch argument and rerun the fixture: the
`(1,2,1)` image must differ from the two-group reference. Restore the guard
afterward. This control requires Windows and has not been run here.

The path accepts one rootless mesh command and no external count buffer. It
uses the translated graphics state policy: slot 57 has no decoded rasterizer
state and therefore uses solid fill, no culling and depth clipping. The shader
guard suppresses group products above 65535, the desktop per-axis limit. The
runtime does not report that GPU-side suppression as a CPU-side refusal.
