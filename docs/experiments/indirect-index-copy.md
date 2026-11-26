# Plain-memory index views for translated indirect draws

Set `X2W_INDIRECT_INDEX_COPY=1` with `X2W_TRANSLATED_INDIRECT_DRAW=1`
to let translated indexed commands stage plain title memory in the upload ring.
The copy path is off by default. It runs after the placed-buffer, recording-upload
and address-alias lookups. It requires a fully readable view with a dword-aligned
address and length of at most 16 MiB. It reports `index_unreadable` or
`index_over_copy_limit` for those two rejection cases.

The fallback caches up to 256 snapshots by address and length until Present.
The frame and every recording that uses a snapshot hold its ring chunk.
A snapshot is not refreshed inside the frame: if the title rewrites the same
address and length between two draws of one frame, the later draw reuses the
older bytes. Only a new frame takes a fresh copy.
A GPU-written command selects the index subrange through the existing GPU
bounds check. Direct indexed draws can calculate their subrange on the CPU.

Run the synthetic fixture on Windows WARP after building the normal dev preset.

```powershell
ctest --preset dev --no-tests=error -R '^startup_indirect_index_copy(_off|_late|_wide)?_warp$' -V
```

The fixture compares every pixel against a CPU triangle reference for R16 and
R32 views in virtual memory. A repeat recording checks reuse within the frame.
A second frame checks that writes to the same address replace the previous
snapshot. The switch-off case expects a clear target and
`index_no_desktop_range`. The copy-start and format mutants must produce
clear pixels where the reference expects coverage. The fixture also checks the
census for an oversized view and a view crossing into a protected page.

The WARP result and the runtime census for a title run remain unmeasured.
