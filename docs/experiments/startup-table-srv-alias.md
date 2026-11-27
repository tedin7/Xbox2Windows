# Table buffer views over aliased placed buffers (opt-in)

A translated draw copies every buffer it reads from the title's descriptor table into the upload ring
(`frame_copy`, see [Copies of title memory within a frame](../startup-runtime.md#copies-of-title-memory-within-a-frame)).
Within a frame the copy is reused only after a full comparison with the bytes it was made from, and the copy is
cut at 16 MiB. Vertex streams and live indices in a slot 133 placed buffer that aliases title memory bind that
alias instead (#2724, #2834). `X2W_D3D12_TABLE_ALIAS=1` does the same for table buffer views.

## What the switch changes

- In the `draw_translated` view loop, a table buffer view (`source 0`) that is not a formatted load, with a
  nonzero range of at most 512 MiB, looks its whole range up with `placed_buffer_range`. When that finds one live
  placement of kind `aliased_buffer` and the view starts 16-byte aligned in it, the draw keeps the alias alive
  with the recording, requires the shader-resource states and creates a raw view over the full range.
- The raw view covers whole DWORDs, so the lookup uses the range rounded up to 4 bytes. A view whose rounded
  range would run past the placement keeps the copy.
- When the same alias is also the draw's live index buffer, it is put in the index-buffer and both
  shader-resource states at once, so neither binding leaves the other in a wrong state.
- The GPU then reads the title's bytes at execution. A title write between recording and submission is visible
  to the draw, as on the console, where the GPU reads the same memory when the work runs. With the copy, each
  draw sees the bytes as they were when it was recorded.
- Ranges past 16 MiB are bound in full; the copy reads zeros there.
- Everything else keeps the copy: formatted loads, views outside a placement or across its end, placements
  that are copied or shared (`X2W_D3D12_LINEAR_BUFFERS`), released resources, unaligned offsets.
- Unset, nothing changes.

## Fixture

```powershell
cmake --build --preset release --target startup_stream_consumer
ctest --preset release --no-tests=error -j 1 -R '^startup_table_alias_(warp|hardware)$' -V
```

`X2W_TEST_TABLE_ALIAS_WRITE`: a 256-byte slot 133 buffer placed on `VirtualAlloc` memory, a raw view the title
creates over it (device slot 18) at table entry 0, and a vertex shader that reads one 8-byte record. The title
writes 40, draws the left half, writes 80, draws the right half, and writes 120 after recording, before the
list runs.

- Unset: left 40, right 80 (each draw's copy).
- Set: both halves 120 (the alias, read at execution).
- Negative control: the same fixture with the buffer in image memory (a static array) is placed "not aliased",
  and the set arm then reads 40 | 80 and fails.

Two more cases run with the switch set. The fixture skips (77) when the adapter has no existing heaps.

- `shared-index`: the buffer also holds the indices of one indexed draw (indices 0–2 at offset 0, index view
  over the same placement). It runs under the debug layer with GPU-based validation
  (`X2W_TEST_DEBUG_LAYER=1 X2W_TEST_GPU_VALIDATION=1`) and must report no error and draw 40.
- `odd-size`: a 254-byte buffer under a structured view of 127 two-byte records. The 256-byte raw view would
  run past the alias, so the draw takes the copy and shows 40, not the value 120 written after recording.

## Startup run of the main executable

Local diagnostic runs, Windows 10 22H2 (19045), RTX 3080 Ti, NVIDIA driver 32.0.16.1047, i7-4790K, release
build of this change on top of #2949 with `X2W_D3D12_RING_RELEASE_AT_SUBMIT=1`, warm shader cache,
`run.ps1 -Render` without `-Translate`, scripted input, 250 s. One build; only `X2W_D3D12_TABLE_ALIAS` differs.
Two pairs, the second in reverse order. Aggregates only; logs stay local.

| frames per second, 30 s windows | 150–180 s | 180–210 s | 210–240 s |
|---|---|---|---|
| unset, pair 1 / pair 2 | 8.9 / 8.6 | 5.1 / 7.8 | 3.7 / 4.2 |
| set, pair 1 / pair 2 | 11.5 / 9.8 | 8.0 / 11.2 | 6.6 / 6.1 |

| per run, pair 1 / pair 2 | unset | set |
|---|---|---|
| translated draws | 64,296 / 88,027 | 97,419 / 100,254 |
| list executes | 14,774 / 18,151 | 22,979 / 21,373 |
| null bindings with cause `buffer` | 0 / 0 | 0 / 0 |
| null bindings per 1,000 load-phase draws (all cause `token`) | 53.8 / 46.5 | 54.0 / 50.8 |
| process commit peak | 18.1 / 17.1 GiB | 17.9 / 18.1 GiB |
| device removal | none | none |

- Before 150 s (intro and menus) both settings run at about 30 fps.
- Where the time went, from local timers that are not part of this change (150–240 s, threads busy):
  - unset: about 1.9 threads waited for `draw_lock`; the lock was held about 58 % of the time, mostly by
    `draw_translated`, and 86 % of that hold was `frame_copy`;
  - 97 % of the `frame_copy` time was one view: 16 MiB (cut from a 90 MiB slot 133 placed buffer that aliases
    title memory) compared in full on about 140 draws per second, almost always unchanged;
  - set: the lock waits fell to about 0.1–0.3 threads and the copy of that view was gone.
- A user watching the set run saw no visual difference or glitches.

Retest after the review fixes (index state, rounded range), same settings, two more pairs (the second again in
reverse order), build = this branch after merging `main` `fb955a51`, on top of #2949:

| frames per second, 30 s windows | 150–180 s | 180–210 s | 210–240 s | mean |
|---|---|---|---|---|
| unset, pair 3 / pair 4 | 6.3 / 7.9 | 9.3 / 4.5 | 3.8 / 4.0 | 6.5 / 5.5 |
| set, pair 3 / pair 4 | 11.9 / 11.7 | 7.9 / 10.8 | 6.5 / 8.0 | 8.8 / 10.2 |

| per run, pair 3 / pair 4 | unset | set |
|---|---|---|
| translated draws | 73,904 / 89,833 | 120,081 / 122,176 |
| list executes | 16,022 / 21,080 | 28,516 / 25,474 |
| not drawn in the load (missing translations) | 47 % / 50 % | 50 % / 49 % |
| null bindings with cause `buffer` | 0 / 0 | 0 / 0 |
| process commit peak | 17.5 / 17.3 GiB | 17.6 / 17.5 GiB |
| device removal | none | none |

- `draw_lock` waits 150–240 s: unset 1.0–1.5 threads, set 0.08–0.14; the draw hold 0.4–0.56 vs 0.07–0.09.
- Unset, the 16 MiB view is again 83–86 % of `frame_copy` time; set, no copy above 64 KiB remains.
- The pair 4 set arm reached the load later (27.8 fps at 120–150 s vs 19.2), which favours its later windows.

Not established: other titles, AMD adapters, and whether any draw in this title depends on a value written
between recording and submission (none was seen).
