# Compute DPP after EXEC restoration

The startup compute translator supports DPP16 `v_min_f32`, `v_max_f32` and
`v_add_f32` with full row and bank masks after a forward branch. Every path
must reach the exchange block, and every incoming path must prove the guest
entry EXEC at the instruction. The translator emits the converged block outside
the per-thread program-counter guard so the whole group executes each barrier.

The synthetic WARP fixture dispatches a wave32 and a 16-thread partial wave.
It checks inclusive min prefixes within the active rows, an in-row broadcast
(`row_share` in the RDNA2 decoder), arbitrary quad routing, and zero-fill at
a row boundary. The partial wave has one active row, so its prefix spans all
active lanes. The full wave also checks that a shift cannot read from the
previous row. Move and OR forms restore the saved EXEC in separate runs. A third run
checks a branch with unchanged entry EXEC.
Each run compares the output buffer against a CPU reference. The fixture also
requires mismatches for wrong quad lanes, a cross-row shift, a wrong broadcast
lane, incorrect zero-fill, and an omitted restore.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_dpp_flow_(controls|warp)$' -V
```

The Python controls check the emitted barrier placement and refuse narrowed
EXEC, restoration on only one edge, a clobbered saved mask, and an early exit
that bypasses DPP.
They can run independently with
`python3 -B tests/startup_compute_dpp_flow_check.py`.

The controls passed on Linux x86_64 with Python 3.14.8. Starting from
`e8015da7a`, the parent translator refused all three fixture variants and
the revised translator emitted HLSL for each. The existing straight-line
compute quad-swap and row-rotate fixtures emitted byte-identical HLSL in a
direct comparison. WARP readback is pending on Windows.
