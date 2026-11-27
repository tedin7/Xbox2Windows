# Placeholder graphics draw diagnostics

A placeholder graphics translation uses the regular `<key>.vs.cso` and
`<key>.ps.cso` files. Its `<key>.txt` manifest adds a line of the form
`placeholder <original pixel-stage refusal>`. The translator option
`X2W_PLACEHOLDER_PS=1` controls emission. The runtime reads the marker with
the other manifest entries.

The periodic per-pipeline report shows the cumulative number of recorded
placeholder draws and the 20 most used keys. Each key lists distinct
attachment dimensions and numeric DXGI formats observed on its draws. The
tester archive adds the latest cumulative total to the render summary in
`null-inputs.txt` when the runtime emitted one.

On Windows, run:

```powershell
ctest --preset dev --no-tests=error -R '^(startup_packed_target_draw_warp|startup_tester_pack)$' -V
```

The WARP fixture checks all 64 RGBA8 pixels against a CPU reference for a
marked translation and its unmarked control. A scissor outside the target
keeps the marked command counted while leaving the blue clear colour. The
reference comparison must reject that readback.
