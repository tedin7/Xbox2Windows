# Scalar selection of lane-masked compute arms

The compute translator can select between two forward arms with a uniform scalar
comparison. Each arm initializes EXEC from the same VCC comparison and uses
`s_cbranch_execz` to skip its lane-masked body. The translator follows the first
body's successors up to the join. This distinguishes the jump over the second
arm from a second entry into the first body. The proof requires entry EXEC at
the comparison that writes VCC and at the scalar split on every incoming path.
The HLSL uses scalar `if / else` for the split and lane-local guards for the
EXEC skips.

The synthetic fixture reads the selector from a read-only buffer and compares
the thread index with a dispatch constant. The arms store distinct values to
disjoint output regions. Host controls cover extra entries into each arm and
a path that narrows EXEC before the split. The WARP test compares all 320 output
words with the CPU reference for selectors 0 and 1 and bounds 3 and 5. It checks
the first mismatching word for each mutant that runs the second arm unmasked,
runs both arms, or chooses the wrong arm.

```text
python -B tests/startup_gfx2hlsl_execz_arms_check.py
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_execz_arms_(controls|warp)$' -V
```
