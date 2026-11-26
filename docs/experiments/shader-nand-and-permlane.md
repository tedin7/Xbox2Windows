# Lane-mask NAND with a uniform-bit immediate

The graphics fixture writes an SGPR lane mask using `s_nand_b32` with
one mask and one uniform-bit immediate (0 or 0xffffffff). It selects four
render-target values from the result bits. A CPU reference predicts the four
values and the mismatch when the generated shader drops the complement.
A later read of the whole-word SCC still requires a modeled SCC write.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_sop2_complement(_nand_warp)?$' -V
```

The fixture uses synthetic instruction words. Windows WARP readback
results are pending.

A literal selector for `v_permlane16_b32` stays refused: the LLVM 15
assembler is the oracle for that encoding and it does not accept one.
