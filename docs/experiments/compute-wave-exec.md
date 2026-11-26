# Compute off stores and live scalar joins

For a raw compute buffer store in the `off` form, active guest lanes share a
byte address. The fallback finds the highest active guest lane by linear
thread-group index. It uses a ballot to locate that thread's host wave lane,
reads each data word from it with `WaveReadLaneAt` and issues one store.
An empty ballot produces no write.

For a straight `s_cbranch_execz` block with a live SGPR defined by
`s_buffer_load_dword` or `s_lshl_b32`, the fallback votes with
`WaveActiveAnyTrue`. The whole host wave executes the scalar assignment if
any guest EXEC bit is set. The empty wave takes the skip edge and retains the
incoming SGPR. The scalar operands must be uniform and the load must use a
read-only descriptor.

The fallback requires forward wave-uniform control flow and a group that fits
one guest wave. Groups of at most four threads use the ordinary `cs_6_0`
target. Larger covered groups request a fixed host wave size with `cs_6_6`.
See the [D3D12 wave size specification](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html).
The original lowering runs first.

```bash
python -B tests/startup_compute_wave_exec_check.py --llvm-mc llvm-mc-19
```

On Linux this matched the three synthetic instruction streams against the
assembler. Each input reproduced its target refusal on parent revision `e8015da7a`.
A separate comparison against the parent found byte-identical HLSL and
manifests across the existing 33 compute fixtures.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_wave_exec_(controls|warp)$' -V
```

On Windows the WARP test is expected to compare all 320 readback words for
each of four EXEC masks per shader. The lowest-lane store mutant and the two
per-lane branch mutants must produce the specified mismatches for sparse EXEC.
The WARP test is pending on Windows.
