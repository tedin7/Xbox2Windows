# Compute EXECZ exits

A forward compute `s_branch` can leave an EXECZ-skipped block when the two paths
reconverge in a forward, single-entry region. The translator checks EXEC and
live scalar state through the common join. The per-invocation program counter
keeps each lane on its selected path.

The synthetic fixture exercises a jump to the original join, a jump to a shared
tail, and a jump to a separate arm before the common tail. It checks four
active-lane counts and both outcomes of the uniform branch for each program.
The WARP test compares the UAV readback against a CPU reference and separately
dispatches each program with the early jump replaced by a no-op. It requires an
exact word mismatch for the cases in which that mutation changes the result.

Run the Python controls from the repository root:

```sh
python -B tests/startup_compute_execz_exit_check.py
```

After configuring and building the Windows dev preset:

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_compute_execz_exit_(controls|warp)$'
```
