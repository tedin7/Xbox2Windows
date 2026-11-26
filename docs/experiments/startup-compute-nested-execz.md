# Nested EXECZ in compute shaders

The structured lowering handles forward nested EXECZ regions for compute groups
of up to 32 threads. The group represents one guest wave. Each EXECZ decision
reduces the per-thread EXEC bits over the group, so it works across host wave
sizes. Each thread keeps a running flag for an early exit to an enclosing join.
The flags agree across the group because the exit uses a group EXEC vote or a
proven uniform SCC comparison. The structured path requires an adjacent
comparison for SCC branches and rejects edges outside the enclosing regions.

For a live `s_mov_b32`, the translator checks the source for uniformity. All
threads in the guest wave execute the move when the region runs. If no thread
has EXEC set at the branch, the shader keeps the previous scalar value. The
legacy lowering remains the entry point for previously accepted shaders.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_nested_execz_(controls|warp)$'
```

The WARP test checks three four-thread shaders against a CPU reference. Six
inputs per shader exercise empty, partial and full EXEC masks. Independent
mutants suppress each early exit or limit the scalar move to EXEC-active
threads. Each mutant must produce the expected first word mismatch.

The Python test checks translation and refusal paths, including a crossing exit,
a non-uniform source and a group beyond the supported size.

```bash
python3 -B tests/startup_compute_nested_execz_check.py --llvm-mc llvm-mc-15
```
