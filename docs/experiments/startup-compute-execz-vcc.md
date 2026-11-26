# VCC joins after compute EXECZ branches

The translator retries a compute shader after a live-VCC EXECZ refusal. The
retry uses the guest wave32 ballot even when a desktop wave contains a
different number of threads. When any guest lane is active, the block runs
for the wave, and the existing compare lowering writes the predicate AND
EXEC into each lane's VCC bit. The retry also clears the high word on a
compare to VCC. An empty EXEC skips both writes. A scalar move to VCC
assigns the source's bit for each lane, including inactive lanes.

The retry requires a known wave32 size, complete waves in the thread group,
and forward branches with wave-wide conditions. Multi-wave groups also
require that the translated body use no other thread-group or host-wave
exchange. The regular translation runs first and returns its original HLSL
and manifest on success.

```sh
python -B tests/startup_gfx2hlsl_execz_vcc_check.py
```

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_compute_execz_vcc_(controls|warp)$'
```

The WARP check compares the full output buffer against a CPU reference for
signed integer, floating-point and floating-point class compares and for
scalar moves from an immediate and from EXEC. It covers empty EXEC, active
and inactive lanes in the same wave, and two waves at different points in
the shader. For each instruction case it also dispatches an HLSL mutant
that makes the EXECZ decision per lane. The test requires the readback to
report the exact first mismatching word for each mutant.
