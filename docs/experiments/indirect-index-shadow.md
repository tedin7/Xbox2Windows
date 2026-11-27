# Indexed indirect draws from title memory

`X2W_INDIRECT_INDEX_SHADOW=1` lets slot 61 stage a readable index view when
there is no placed desktop buffer for it. The runtime copies the declared view
because the GPU can write the draw's start index and count. The frame cache
keys its slices by title address and byte length. Present releases the frame's
holds, and command recordings and submitted queues retain their own holds.
The title must populate the index data before its first use each frame.

`X2W_INDIRECT_INDEX_COPY=1` uses the same cache for translated indexed draws
and keeps its dword alignment requirement when the shadow switch is off.
Existing placed, recording-upload and alias routes retain priority where
supported.

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_indirect_index_.*_warp$'
```

The fixture writes the indirect record on the GPU and checks every render
target pixel against a CPU reference. It uses one unregistered R16 view, changes
its bytes, then draws with both the original and a shorter view before Present.
After Present, the original view must reflect the new bytes. With shadow off,
the target stays clear and the census must record `index_no_desktop_range`.
The stale-cache mutant must produce the exact previous-frame image and fail
the refreshed-image comparison.

The padding fixture submits a six-byte R16 view with three complete indices.
The test hook `X2W_TEST_INDIRECT_INDEX_COPY=seed-tail` fills the eight-byte
upload slice with `0x01` before staging and exposes the padded halfword as the
last index of the indirect draw. The WARP readback must match the CPU image
for indices `(1, 2, 0)`. The `skip-zero` control retains the seed in the tail.
Index `0x0101` then selects the same corner as index `2`. The control must
fail the same image comparison with an entirely clear target.
