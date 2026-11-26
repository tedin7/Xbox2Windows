# Compute loops with retiring EXEC lanes

A wave32 compute loop can clear lanes permanently with a vector compare that
writes EXEC. If the exit tests whether the remaining EXEC mask is empty and
the latch counts up to a literal limit, the HLSL thread can leave the loop as
soon as its bit clears. The guest wave may continue to the latch for other
lanes. A cleared lane cannot do further vector work, and the latch counter
and SCC must be dead at the common exit because their values can differ there.

The proof requires a single entry, one latch, one exit, and only the supported
lane-local vector operations in the guarded body. A cross-lane operation or
group-memory exchange keeps the refusal.

`startup_lane_exec_warp` dispatches one group of 64 threads with distinct input
counts. The readback compares all 64 accumulated results and the untouched
buffer words. The wave-wide break mutation must produce a readback mismatch.
Python controls cover a wave reduction in the loop and a live latch counter.
On Linux with LLVM 19, the fixture bytes matched the assembly and both refusal
controls passed. The original translator refused the positive fixture. The 33
existing synthetic compute fixtures emitted byte-identical HLSL. WARP readback
and the GPU mutant are pending on Windows.

```sh
python -B tests/startup_lane_exec_check.py --llvm-mc llvm-mc-19
```

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_lane_exec(_warp)?$'
```
