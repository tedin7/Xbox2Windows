# Startup pixel image atomics

The gfx1013 pixel translator lowers 2D no-return integer image atomics
with `dmask:0x1`. The runtime binds an `R32_UINT` UAV for each `tex2du`
manifest entry. See [the manifest and binding rules](../startup-runtime.md).

Run the translation controls with Python on synthetic instruction words.

```sh
python3 -B tests/startup_pixel_image_atomic_check.py
python3 -B tests/startup_pixel_uav_check.py
```

On Windows, run the offscreen WARP and runtime readbacks, the translation
controls, and the pure format and capability checks.

```powershell
ctest --preset dev --no-tests=error -R '^startup_pixel_(image_atomic_(controls|warp)|uav_(atomic_warp|support(_accept_sint_control|_no_load_check_control)?))$' -V
```

The offscreen test draws 16 pixels into two 4×4 UAVs. The CPU folds each
active pixel's contribution into shared texels and checks all remaining
texels for unchanged sentinels. One fixture narrows EXEC with `v_cmpx`.
Another fixture restores EXEC after whole-quad mode, narrows it with a
scalar mask, and uses the same data register in both atomic blocks.
Controls substitute wrong operations, use signed min, split OR into a
load and a store, and remove EXEC guards. Each control requires a texel
mismatch.

The runtime test draws through the guest descriptor table with typed
loads masked off. It accepts an `R32_UINT` resource and an explicit
`R32_UINT` view of a typeless resource. The format refusals require a
matching log message and unchanged textures.

This change's local checks use Python and synthetic inputs. WARP
readback and the C++ format check require a Windows run.
