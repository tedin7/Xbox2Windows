# Frame-scoped readability for small draw bindings

`X2W_READABLE_FRAME_CACHE=1` enables a per-thread, 1,024-entry readability
table for translated buffer bindings below 64 KiB. The switch defaults off.
Each entry uses a 64 KiB region number and the Present counter, with a bit for
each 4 KiB page proven readable by a successful probe. A collision evicts the
previous entry. Each draw still copies its own bytes into the upload ring.

A title can decommit or protect a page after binding it within a frame. The
guarded ring copy catches the access failure, discards the verdict and supplies
the existing unresolved-binding behavior. The next frame probes again.
The heartbeat's `buffer binding steps (under 64 KiB)` line reports cumulative
readability hits and probes. Compare trim-time deltas over the same frame
window with and without the switch.

On the Windows test machine, run the WARP readback fixture with:

```powershell
ctest --preset dev --no-tests=error -R '^startup_readable_frame_cache_warp$' -V
```

The fixture compares 2,000 draw-specific pixels against a CPU reference and
requires two probes in that frame. It decommits one source before a draw in
the next frame, which must produce a null-buffer pixel with one probe and zero
copy faults. The frozen-stamp mutant still produces the null pixel through the
copy guard, but fails the counter assertion with one hit and one copy fault.
