# Compute EXEC scopes and embedded constants

The synthetic fixtures cover a nested EXEC scope with two incoming paths, a
32-bit EXEC count on guest wave32 and wave64, and a PC-relative load from bytes
after the shader terminator. The last fixture also has an unused GETPC pair.
The count uses a masked ballot when the workgroup occupies one host wave. A
group-shared reduction uses guest thread order when the host splits the group.
The translator requires a workgroup convergence point for the count.

```text
python -B tests/startup_gfx2hlsl_compute_exec_pc_check.py
python -B tests/startup_gfx2hlsl_getpc_refusal_check.py
ctest --preset dev --no-tests=error -R '^(startup_gfx2hlsl_compute_exec_pc_(controls|warp)|startup_gfx2hlsl_getpc_refusal)$'
```

The WARP test compares the full output buffer against a CPU reference for six
dispatch configurations. Each compiles a second shader whose changed restore,
count, or constant must produce a readback mismatch.

Local evidence on Linux with Python 3.14.8 used baseline `e8015da7a`.
All five new binary fixture variants hit their target refusals on that revision.
The 33 existing cases in `startup_gfx2hlsl_compute_check.py` that translated
on the baseline produced identical HLSL and manifests with the patch.
The two Python commands above passed locally. WARP readback is pending on Windows.
