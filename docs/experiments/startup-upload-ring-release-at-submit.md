# Upload ring: release a closed list's chunks at its fenced submit (opt-in)

Refs [#2837](https://github.com/R0Y00/Xbox2Windows/issues/2837),
[#2849](https://github.com/R0Y00/Xbox2Windows/pull/2849).

Since #2823 a reservation in the shared upload ring keeps its bytes until its chunk is recycled, and #2849
recycles a chunk only when no command-list recording holds it and every queue that used it has passed its
fence. A closed list keeps its hold until Reset, Release or a failed Close, because D3D12 lets it be submitted
again. `X2W_D3D12_RING_RELEASE_AT_SUBMIT=1` moves that release to the list's fenced submit and sizes the ring
in 16 MiB chunks up to 512 MiB.

## What the switch changes

- After a submit whose queue fence was signalled, each submitted list stops holding its chunks
  (`release_upload_chunks_at_submit`). The queue fence still keeps those chunks from reuse until the GPU has
  passed it. Without a signalled fence nothing changes: the chunk stays busy, as without the switch.
- The list records the generation of each chunk it released. If the same list is submitted again before its
  Reset, chunks with an unchanged generation are fenced again by that submit
  (`resubmitted_upload_chunks`). If a chunk was recycled in between, the list would read replaced bytes: the
  queue leaves it out of `ExecuteCommandLists` (`released_chunks_recycled`, checked with `draw_lock` and
  `allocator_lock` held, the same locks the submit bookkeeping holds), counts it and logs the first eight
  (`... were reused; refused, it is not executed`). The other lists of that call still run, as a list that is not
  closed is left out today. It stays refused until its Reset, which records it afresh.
- The ring grows in 16 MiB chunks up to 512 MiB, so one list that is still recording or waiting to be submitted
  pins 16 MiB instead of a chunk of up to 128 MiB.
- Unset, the ring behaves exactly as before: doubling chunks up to 256 MiB, holds until Reset.

## Fixtures

```powershell
cmake --build --preset release --target startup_stream_consumer upload_reservation_check d3d12_xs_shim
ctest --preset release --no-tests=error -j 1 -R '^(upload_reservation|startup_address_console_gate_ring_keep(_release)?_(warp|hardware))$' -V
```

- `upload_reservation`: `check_release_at_submit` releases two chunks after a fenced submit, keeps them from
  reuse until the fence passes, reuses one, then finds the other intact and the reused one recycled on a
  resubmit, and fences the intact one again. Mutants: keeping the holds fails "release: holds dropped";
  ignoring generations fails "a resubmit finds chunk 1 intact and chunk 0 reused".
- `startup_address_console_gate_ring_keep_*` (`X2W_TEST_CONSOLE_KEEP`): a second list A uploads 32 by-address
  payloads of 16 MiB each from the ring (512 MiB), is submitted, completes and is never Reset. The fixture's own
  list then draws payload 32 over the right half of the target, A is submitted again, and the target is read back
  after that submit.
  - Without the switch, A keeps every chunk: the ring refuses past 256 MiB and the right half reads the null
    view.
  - `..._keep_release_*` sets the switch: the right half holds payload 32 from a chunk A released. A's second
    submit would read a reused chunk, so it is refused and not executed: the right half still holds payload 32,
    not A's payload 31.
  - Mutants: disabling the release leaves the right half on the null view; executing the stale resubmit (the
    check before the execute removed) puts A's draws back over the right half and fails on wrong pixels.
- Fixtures that describe the default ring clear the switch for their own process, so they keep testing the
  default mode when it is set in the environment: the #2849 `startup_address_console_gate_ring{,_reuse}_*`
  (a refusal at the 256 MiB cap) and `startup_slot133_heap_cap_capped_*` (doubling growth to 48 MiB).

## Startup run of the main executable

Local diagnostic runs, Windows 10 22H2 (19045), RTX 3080 Ti, NVIDIA driver 32.0.16.1047, i7-4790K, 16 GB,
release build of this change, warm shader cache, `run.ps1 -Render` without `-Translate`, scripted input, 250 s.
One build; only the switch differs. Aggregates only; logs stay local.

| | switch unset | switch set |
|---|---|---|
| null bindings with cause `buffer` | 33,632 | 0 |
| `upload ring reservation refused` logged | yes | no |
| resubmits after a reuse (now refused) | n/a | 0 |
| process commit peak | 16.7 GiB | 18.1 GiB |
| visible | single full black frames in the post-load animation | none seen |

- Without the switch, a census at each refusal showed every chunk held by closed lists not yet Reset, while
  every queue had passed the chunk's fence. Most refused requests were 64 KiB: the window
  `upload_from_title` copies for a graphics root SRV without an existing upload mapping.
- The title never submitted a closed list again without a Reset: 0 of more than 500,000 submits over five runs.
  A title that does resubmit after a reuse gets the refusal described above: that list does not run.
- The larger ring cost about 1 GiB of process commit at peak in this run.
- Repeated after the stale-resubmit refusal, on a later `main`, same settings: buffer null bindings 50,665 →
  0, reservation refusals logged → none, stale resubmits refused 0 → 0, commit peak 17.9 → 18.0 GiB, black
  frames → none seen.

Not established: other titles' resubmit behaviour, and whether 512 MiB is enough on other workloads.
