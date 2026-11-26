# Lane-mask NAND and literal row selectors

The graphics fixture writes an SGPR lane mask using `s_nand_b32` with
one mask and one uniform-bit immediate. It selects four render-target
values from the result bits. A CPU reference predicts the four values
and the mismatch when the generated shader drops the complement.
A later read of the whole-word SCC still requires a modeled SCC write.

The 64-thread compute fixture gathers within four rows of 16 lanes.
It exercises register selectors and an appended literal in either
selector field of `v_permlane16_b32`. The groupshared exchange uses
guest flat thread indices behind group barriers with entry EXEC.
The CPU reference checks the entire output buffer. The wrong-row and
swapped-selector shaders must each report the predicted first mismatch.

```powershell
ctest --preset dev --no-tests=error -R '^(startup_gfx2hlsl_sop2_complement(_nand_warp)?|startup_compute_permlane16_(controls|warp|wrong-row_warp|swapped-selects_warp))$' -V
```

The fixtures use synthetic instruction words. Windows WARP readback
results are pending.
