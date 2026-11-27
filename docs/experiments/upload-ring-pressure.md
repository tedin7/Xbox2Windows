# Upload ring pressure during a frame

Set `X2W_UPLOAD_RING_OVERFLOW=1` to let the allocator grow after the regular
upload ring runs out of usable space. The default is off. The additional
budget is at most 256 MiB, allocated on demand. The allocator reuses a full
chunk only after its recording holds are released and every queue that used
it has completed. A request can still fail when both budgets are exhausted
and no chunk is reusable, or when allocation fails.

The render totals report the cumulative number of frames that reached the
overflow path, once per frame even if several reservations needed it.

Run the synthetic WARP fixture on x64 Windows:

```powershell
ctest --preset dev --no-tests=error -R '^startup_frame_copy_ring_pressure_warp$' -V
```

Each of 40 separately submitted frames records eight 512-byte reservations
against a 2 KiB regular ring. The test expects all 1,280 pixels to match the
CPU reference, 40 pressure frames and zero copy failures. It also checks that
the same two chunks recycle after submission. The switch-off control expects
the last four columns of every frame to retain their clear color, with 160
refused reservations. WARP execution is pending on Windows.
