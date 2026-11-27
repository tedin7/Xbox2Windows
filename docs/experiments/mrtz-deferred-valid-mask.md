# Depth export before the terminal valid-mask export

The startup pixel translator accepts a single uncompressed depth export
followed in its basic block by a terminating valid-mask export, with no
intervening EXEC write. The depth assignment is predicated when EXEC can
exclude a lane. A path that bypasses depth must reach a terminal null export
with EXEC zero. The translator content fingerprint refreshes the existing
`mrtz` refusal family for the pre-run retry.

With Linux and Python 3.14.8, the fixture passes with this change and reaches
the MRTZ refusal with the export and path logic from base revision
`e8015da7a`. Three previously accepted synthetic MRTZ variants emit identical
HLSL. The fixture also checks container intake, coverage of both exits, and
rejection of invalid export forms or control flow.

```sh
python3 -B tests/startup_gfx2hlsl_mrtz_check.py
python3 -B tests/startup_mesh57_depth_only_check.py
python3 -B tests/startup_gfx2hlsl_retry_lifted_check.py
python3 -B tests/startup_image_load_mip_check.py
```

The Windows fixture uses a 32×32 synthetic R32_FLOAT source and compares every
D32_FLOAT and RGBA8 pixel to a CPU reference. The expected mismatch counts
(depth, colour) are (0, 0) for the translated shader, (256, 0) for a depth leak,
and (768, 768) when the null path skips the discard. The Windows results are
pending.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_mrtz(_warp)?$'
```
