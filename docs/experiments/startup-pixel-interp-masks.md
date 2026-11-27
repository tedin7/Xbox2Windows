# Startup pixel EXEC masks and binary16 compares

The startup translator omits the partial value from `v_interp_p1ll_f16`.
Its liveness scan allows a read after a covering write when the read stays
within the covered lanes and the selected HLSL operation reads only that
lane's registers. For `image_sample` and `image_sample_c`,
the lowering must select an explicit level at the recorded instruction.
A sample whose LOD still depends on derivatives fails the proof.
The translator refuses `image_sample_o` under whole and narrowed EXEC.

An EXEC mask from `s_wqm_b64` contains its input lane mask. The scan tracks
this relation through mask copies, restores, intersection and CFG joins.
When a write covers a saved quad mask, a subsequent read under a proven
subset can use the replacement value. A write under the live-pixel mask
alone leaves the saved quad's helper lanes uncovered.

Binary16 VOPC SDWA supports the existing `lt`, `eq`, `le`, `gt`, `ge`
and `neq` relations with `WORD_0` and `WORD_1`. The synthetic fixtures
exercise VCC, scalar and EXEC destinations, with register pairs that cover
signed zero, subnormals, finite values, infinities and NaNs.

See sections 6.8, 8.2.4, 12.3 and 13.3.8 of the
[AMD RDNA 2 ISA reference guide](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture).

Run the synthetic controls on a Python host.

```bash
nice -n 15 python3 -B tests/startup_gfx2hlsl_interp_f16_check.py
nice -n 15 python3 -B tests/startup_gfx2hlsl_vopc_sdwa_check.py
```

Both commands passed on Linux with Python 3.14.8. The HLSL from 26 legacy
fixtures matched revision `ab02c55` byte for byte. The five retained fixtures
matched revision `cf0fe87f7` byte for byte. The controls detected a bypassed
sample obligation, two reversed subset facts and a binary16 compare that
read the wrong half.

The Windows tests compare WARP readback with CPU references. They check a
three-level 4×4 texture, covering writes across mask restores, and 256 pixel
positions per binary16 compare program. Each draw has a wrong-input or
wrong-result control.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_(interp_f16_(controls|exec_halves_warp)|vopc_sdwa(_warp)?)$' -V
```
