# Compute reads and writes through one buffer descriptor

The translator retries a compute program when a fixed table buffer descriptor
needs both read and write access. The retry emits one `RWByteAddressBuffer`
and one `uav N table P O` manifest entry. Indexed loads and formatted operations
read descriptor metadata from that UAV slot. The runtime uses the same table
record to populate the entry constants and to bind a raw UAV in unordered
access state. The shader preserves same-lane access order. The fixtures use
disjoint records across lanes.

The WARP check runs seven source-built programs, including two existing
byte-addressed programs. The five added cases cover indexed loads with and
without a vector offset, atomic exchange, a formatted 16-bit store, and a
formatted four-component load and store. The runner compares all 320 output
words against the CPU reference, including padding and words the lanes leave
untouched. Each dropped-load mutant must produce the predicted mismatch count.

Run the translation and manifest checks on any supported Python host:

```sh
python3 -B tests/startup_compute_rw_buffer_check.py
python3 -B tests/startup_compute_rw_buffer_check.py --llvm-mc llvm-mc-19
```

Run the readback check on Windows with the `dev` preset and DXC configured:

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_rw_buffer_warp$' -V
```

The local Linux checks passed. Against the baseline `e8015da7a`, all five new
programs reproduce a refusal and 35 existing compute fixtures retain identical
HLSL and manifests. WARP execution and dispatch of the reported keys remain
to be verified on the Windows test machine.
