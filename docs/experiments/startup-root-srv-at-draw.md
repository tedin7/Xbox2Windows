# Graphics root SRV: no desktop copy that no draw reads (opt-in)

Refs [#2948](https://github.com/R0Y00/Xbox2Windows/issues/2948),
[#2949](https://github.com/R0Y00/Xbox2Windows/pull/2949).

List slot 44 (`SetGraphicsRootShaderResourceView`) at a title address without an upload mapping copies a 64 KiB
window of title memory into the shared upload ring and binds it on the desktop list (`list_root_srv`). In the current
code no draw reads that desktop binding:

- Translated draws set their own root signature and read the SRV from its title address (`pass.root_address`,
  `source == 2` in `draw_translated`) at the draw.
- `list_draw` forwards a draw only with `pass.pipeline_real`, which `list_pipeline` sets for a desktop pipeline
  without an indirect graphics root. `build_stream_pipeline` accepts a graphics stream only with VS/PS or MS/PS
  ("desktop indirect graphics stream needs ...") and gives every accepted one that root, so `pipeline_real` holds only
  for compute pipelines. A vertex-only desktop stream is refused by the same rule.
- An indirect graphics draw (list slot 61) requires `pass.indirect_graphics.root == pass.graphics_root`: an empty
  root signature (`root_without_bindings`), or for a cached mesh pipeline the title's root, whose only root SRV
  (`cached_srv_root`) the draw binds from its validated lease.

## What the switch changes

- `X2W_D3D12_ROOT_SRV_AT_DRAW=1`: slot 44 skips that copy. The title address is kept for translated draws as before,
  and the parameter stays unbound on the desktop list (`srv_bound` clear), so a forwarded `list_draw` that needs bit 0
  is skipped rather than drawn with an older binding. A desktop graphics pipeline that reads a root SRV, if one is
  added, would need the copy at its draw.
- With the switch set, the render totals are followed every 300 frames by
  `x2w-render root SRV copies (X2W_D3D12_ROOT_SRV_AT_DRAW): skipped N, copied M`.
- Unset, slot 44 copies at every call, as before.

## Fixture

```powershell
ctest --preset release --no-tests=error -j 1 -R '^startup_root_srv_draw_(warp|hardware)$' -V
```

`startup_root_srv_draw_*` (`X2W_TEST_ROOT_SRV_DRAW=write-after`, `tests/startup_root_srv_draw.cpp`): the stream
root's parameter 0 is a root SRV. Slot 44 binds a `VirtualAlloc` buffer holding red, the title writes green, then a
translated draw reads the SRV through its root pointer. Both modes draw green over the left half. After the draw is
recorded the fixture reads the stand-in's counts (`x2w_d3d12_root_srv_copies`): unset, `skipped 0, copied 1`; set,
`skipped 1, copied 0`. Mutant: ignoring the switch fails
`X2W_D3D12_ROOT_SRV_AT_DRAW=1: copies (skipped, copied) (0, 1), expected (1, 0)`.

Not covered: a desktop draw that reads the binding, because none can be built (see above).

## Startup run of the main executable

Local diagnostic runs, Windows 10 22H2 (19045), RTX 3080 Ti, NVIDIA driver 32.0.16.1047, i7-4790K, 16 GB, release
build on `main` `98e4a2ee`, warm shader cache (revision 29), scripted input, 250 s. One build; only the switches
differ. Aggregates only; logs stay local. The first two tables were measured with the first revision of this change
(`bedfcafc`), which kept the address pending for a desktop draw instead of skipping the copy; no run made a copy at
a draw, so slot 44 did the same work as now.

Each switch alone, against no switch:

| | none | `ROOT_SRV_AT_DRAW` | `RING_RELEASE_AT_SUBMIT` |
|---|---|---|---|
| null bindings with cause `buffer` | 164,727 | 88 | 0 |
| `upload ring reservation refused` logged | yes | yes | no |
| visible | single black frames in the post-load animation | none seen | not watched |

With `X2W_D3D12_RING_RELEASE_AT_SUBMIT=1` and `X2W_D3D12_TABLE_ALIAS=1` set in every run, two pairs, the second in
reverse order:

| | switch unset | switch set |
|---|---|---|
| mean fps 150–240 s, pair 1 / pair 2 | 6.3 / 7.1 | 14.8 / 14.0 |
| frames presented, pair 1 / pair 2 | 3,663 / 3,541 | 4,689 / 4,804 |
| executes, pair 1 / pair 2 | 33,408 / 36,029 | 72,177 / 70,916 |
| slot 44 time 150–240 s (threads; µs per call) | 1.14–1.19; 43–49 | 0.28–0.29; 5 |
| copies not made at slot 44 / made at a draw | n/a | about 7 million / 0 |
| null bindings `token` per 1,000 translated draws | 33.8 / 32.3 | 36.7 / 26.9 |
| null bindings `buffer`, device removal | 0, none | 0, none |
| process commit peak | 19.1 / 19.0 GiB | 19.8 / 19.4 GiB |

- The slot 44 copies were most of the upload ring traffic that #2948 describes: the switch alone removes all but
  88 of the buffer null bindings, but some reservations are still refused, so it complements #2949 rather than
  replacing it.
- Without the table alias the frame rate barely changes (5.9 → 6.1 fps): the draw path is then bound by `frame_copy`
  under `draw_lock`.
- Every run read `deferred N, made at a draw 0`: no draw in these runs read the desktop binding.

Retest of this revision (merged with `main` `3b6c38a8`), same machine, cache and input, ring and alias set in every
run, two pairs, the second in reverse order:

| | switch unset | switch set |
|---|---|---|
| mean fps 150–240 s, pair 1 / pair 2 | 6.3 / 6.5 | 13.9 / 13.4 |
| frames presented, pair 1 / pair 2 | 3,839 / 3,706 | 4,798 / 4,960 |
| executes, pair 1 / pair 2 | 24,065 / 33,220 | 69,730 / 54,789 |
| slot 44 time 150–240 s (threads; µs per call) | 1.17 / 1.11; 52 / 48 | 0.29 / 0.29; 5 / 6 |
| copies skipped / copied (render totals) | n/a | 6.9 / 4.9 million; 0 |
| null bindings `token` per 1,000 translated draws | 23.4 / 29.8 | 24.2 / 33.0 |
| null bindings `wrapper`, pair 1 / pair 2 | 0 / 40 | 107 / 101 |
| null bindings `buffer`, ring refusals, device removal | 0, none, none | 0, none, none |
| process commit peak | 18.7 / 18.5 GiB | 18.7 / 18.7 GiB |

- `wrapper` null bindings also appear in an unset run (40). Per 1,000 translated draws they are 0 / 0.26 unset and
  0.37 / 0.40 set: few, but not explained here.

Not established: desktop graphics pipelines that read a root SRV (none can be built today), and other adapters.
