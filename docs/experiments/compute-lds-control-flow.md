# Bounded compute LDS exchanges with branches

The compute translator checks both arms of forward branches on verified uniform
SCC. Each path must complete its fill, reach the same barrier, and restrict
reads to initialized words. The fill layout must agree across paths so the
same local X ID owns each word even if guest waves take different arms.

An EXEC block can narrow participation with an unsigned comparison of entry
local X against a literal upper bound. Its LDS stores and loads are
lane-predicated. A forward EXECZ skip may cross lane operations and waits, but
the fill wait and barrier must be reached by the entire group. The translator
emits the group sync outside the HLSL program-counter dispatch.
[HLSL group sync rules](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/groupmemorybarrierwithgroupsync)

The synthetic lane fixture fills 64 dwords with the first 64 threads of a
128-thread group. Every thread reads a neighbour from that prefix. The first
64 also execute an LDS broadcast read. The SCC fixture dispatches two groups
to exercise both fill and read arms. The Windows test compares all 320 output
words against a CPU reference, including the untouched sentinel words. Four
HLSL controls must produce readback mismatches.

```powershell
ctest --preset dev --no-tests=error -j 1 -R '^startup_compute_lds_b128_(controls|warp|(lane-wrong-peer|lane-unmasked-read|scc-invert-fill|scc-invert-read)_warp)$' -V
```

On Linux with Python 3.14.8 and LLVM 19, the following synthetic checks passed.
Both new shaders hit the straight-line refusal on base revision `e8015da7a`.
All 42 previously accepted shaders in the four fixture tables generated
byte-identical HLSL against that revision. WARP readback is pending on Windows.

```sh
python -B tests/startup_gfx2hlsl_compute_check.py
python -B tests/startup_compute_lds_b64_check.py
python -B tests/startup_compute_lds_b128_check.py
python -B tests/startup_compute_lds_b128_check.py binary llvm-mc-19
python -B tests/startup_compute_lds_dual_atomics_check.py
```
