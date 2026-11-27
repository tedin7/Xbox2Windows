# Pixel lane-0 reads across forward branches

When a graphics pixel stage refuses a lane-0 read, the translator retries with
the forward uniformity proof. The proof checks entry EXEC on the incoming paths.
A formatted buffer load qualifies when its table descriptor is proven, the
offset is constant and the address uses no vector register. A scalar stays
uniform across a join when every incoming path supplies a uniform definition.
Scalar Condition Code (SCC) keeps its uniform mark when the same proven write
reaches the join on every edge. The retry requires forward control flow and
uses no compute wave lowering.

The synthetic shader selects either a formatted buffer word or a constant
buffer word using a runtime flag. The saved flag drives a second branch and
the SCC at its join drives a scalar select. The 8 × 8 WARP draw compares every
channel with the CPU model for each flag value. Three mutations change the
joined SCC polarity, the joined scalar value and the buffer offset. Each must
produce 64 differing pixels. The Python controls require the translator to
refuse a per-pixel index, a per-pixel branch condition, an SCC write on only
one arm and a saved flag written on only one arm.

On Linux x86-64 with Python 3.14.8, the Python checks passed. The new fixture
refused on base revision `741f301fa` and translated with the retry. Ten
previously accepted synthetic graphics fixtures emitted identical HLSL on
both revisions. Windows WARP readback is pending.

```bash
python -B tests/startup_gfx2hlsl_readfirstlane_uniform_check.py
python -B tests/startup_gfx2hlsl_src_scc_join_check.py
```

```powershell
ctest --preset dev --no-tests=error -R '^(startup_gfx2hlsl_readfirstlane_uniform|startup_gfx2hlsl_src_scc_join(_draw_warp)?)$' -V
```
