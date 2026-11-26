# Compute input snapshot reuse

Set `X2W_COMPUTE_SNAPSHOT_REUSE=1` to reuse placed-input snapshots during a
frame. The switch defaults off and applies when
`X2W_TRANSLATED_TITLE_INPUT_SNAPSHOT=1` takes the title-memory snapshot path.
Ranges over 1 MiB require `X2W_COMPUTE_SNAPSHOT_VERIFY=1` for reuse. Each hit
compares the current byte-range hash and takes a hold on the original ring
chunk for the consuming recording. Present advances the cache epoch. Reset or
release invalidates entries from that recording. The totals report snapshots
made, reused and replaced after a rewrite, and upload-ring MiB saved.

Run the synthetic WARP readback on Windows.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_dispatch_input_warp$' -V
```

The fixture copies each 40 KiB input into its own output range and compares the
entire readback against a CPU reference. The 200-dispatch case expects 200
snapshots with reuse off and one with reuse on. It requires the ring to reserve
less than 5% of the off value. Other cases cover a four-byte rewrite after
dispatch 100, a Present on the same recording, two recording threads, and an
exhausted ring that recycles and overwrites the old slice after reset. The
range above 1 MiB runs with both verification settings. The no-hash mutant
must mismatch on the rewritten word in dispatch 101.

On Linux, the following command verifies the fixture encoding and translates
it to HLSL and a binding manifest. It passed with LLVM 19 on x86-64 Linux.
DXIL compilation, WARP readback and independent Windows verification are
pending.

```bash
python3 -B tests/startup_compute_dispatch_input_check.py --offline llvm-mc-19
```
