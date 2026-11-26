# Sampled texture states in translated draws

Set `X2W_SAMPLED_STATE_STAGES=1` before starting the shim to select the
shader-resource state from the draw's manifest. The translator writes a
`srvstage` entry for each sampled texture. The binder combines the stages
across views of the same desktop resource. Older manifests without stage
entries use both shader-resource bits when the switch is on. Set the same
switch during offline translation to allow explicit-level cube samples in
vertex or pixel shaders. The cube binder requires an explicit cube view.

```powershell
ctest --preset dev --no-tests=error -V -R '^(startup_sampled_state_stages_warp|startup_compute_cube_warp)$'
```

The draw fixture compares 32 RGBA pixels per case against a CPU reference
after a texture copy and a translated draw on the same direct command list.
It exercises vertex-only, pixel-only, shared and aliased bindings, a legacy
manifest, and a cube view. The switch-off control must still match the pixels
on WARP and must produce a resource-state error if debug validation is
available. Without the debug layer, every readback comparison still runs.
The compute cube fixture compares 24 samples from six UV gradients and
requires each face-order or orientation control to produce mismatches.

The Python translation checks run on Linux. WARP and debug validation require
the Windows test machine.
