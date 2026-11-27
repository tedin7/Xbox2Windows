# Compute LDS: ordered b64 pair

The general retry currently handles only one `ds_write_b64` followed by one
`ds_read_b64`, with entry EXEC, straight-line control flow, no guest barrier,
`lgkmcnt(0)` immediately after each DS operation, and proven dword-aligned
addresses and offsets. Each dword is bounded against the work-group allocation.
The retry inserts a group barrier before the read. It runs only after the
legacy exchange proof refuses; it does not cover general LDS instruction
sequences, sub-dword operations, branches or unaligned addresses.

On Linux, base revision `a08c038e2` rejected the fixture with `LDS read needs a
bounded address after the fill barrier`. With the change, the following command
passes the Python controls, including the original-output digest, the ordered
pair, out-of-bounds reads, and alignment and branch refusals:

```sh
nice -n 15 python3 -B tests/startup_compute_lds_general_check.py
```

The registered `startup_compute_lds_general_warp` and `_hardware` tests use
D3D12 readback and a Python LDS model for cross-wave and same-wave exchange and
out-of-bounds reads. They require Windows, DXC and the existing offscreen
runner; they were not run in the Linux Python-only environment. The CPU mutant
that selects the current lane instead of the peer differs from the expected
result. Removing the barrier is not a deterministic negative control: a host
scheduler may happen to execute the accesses in the desired order.
