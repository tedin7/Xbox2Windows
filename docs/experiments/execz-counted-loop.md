# EXEC-guarded scalar counted loop

The compute translator recognizes a loop whose EXEC-zero guard jumps to an
unconditional latch and whose SCC-zero exit jumps past that latch. The guard
follows a literal counter initialization. The loop compares the counter
against an unsigned literal and advances it by one after lane-local vector
arithmetic. The counter and SCC must be dead at the exit. With EXEC empty,
the guest loop would run only the finite scalar progression. A lane that
skips it in HLSL retains its vector values and has no observable scalar
difference at the join.

The synthetic four-lane fixture accumulates the counter into a VGPR and
restores EXEC before storing, so the readback covers active and skipped lanes.
The CPU reference expects 23 for active lanes and 13 for skipped lanes. The
mutant changes the SCC compare bound from five to four and must report 19
instead of 23 for lane 0 in each nonempty case. The cases exercise empty,
partial and full EXEC, and compare untouched output words as well.

```text
python -B tests/startup_gfx2hlsl_execz_counted_loop_check.py --llvm-mc llvm-mc-19
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_execz_counted_loop_(controls|warp)$' -V
```

On Linux with Python and LLVM 19, the assembly and translation check passed.
Baseline `e8015da7a` refused the fixture at `s_cbranch_scc0` with
`branch out of the block`. The 33 existing synthetic compute cases produced
byte-identical HLSL and manifests against that baseline. WARP readback is
pending on Windows.
