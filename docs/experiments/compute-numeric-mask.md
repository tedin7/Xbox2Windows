# Compute lane masks as numeric operands

The startup translator reconstructs the low mask word for `s_mul_i32`,
`v_sub_f32` and `v_mul_f32` in straight-line compute code at entry EXEC. The
thread group must fit in one guest wave of either supported size (at most 32
threads). Each thread contributes its mask bit at its flattened guest lane
index. The group synchronizes before collecting the bits and again before
reusing the shared storage. The float operations interpret the assembled word
as binary32. The gate accepts entry EXEC, proven compare results and copies
of those masks.

The synthetic fixtures pass a VCC copy to each opcode and an EXEC copy to
integer multiply. The Windows test compares all 320 output words, including
the unwritten sentinel words, to a Python reference. A variant for each
fixture supplies just the current lane's bit and must produce a readback
mismatch at word zero. The Python controls also cover the one-thread path,
unproven mask writers, larger groups, branches and a changed EXEC.

```bash
python -B tests/startup_compute_numeric_mask_check.py
```

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_numeric_mask_(controls|warp)$'
```
