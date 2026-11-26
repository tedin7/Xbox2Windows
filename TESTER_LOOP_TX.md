# Tester loop branch with the closed translator PRs (`tester-loop-tx`)

This branch is `tester-loop` plus the translator pull requests that were closed on 2026-11-26 without merging, so a
tester run can compare the two branches on the same title. It is not a release and nothing here goes upstream from
this branch. It follows `tester-loop`: each `tester-loop` push is merged in with a merge commit.

The cache revision is `x2w-startup-shader-29` / `29cs` here (28 on `tester-loop`): the translations differ under the
same request keys, so a cache warmed on `tester-loop` is not reused (`run.ps1` / `run.sh` warn about it; start with
`-FreshShaderCache` or a separate `X2W_SHADER_CACHE`).

## Translator PRs

Gate = the independent check recorded before the PR was closed.

| PR | Head | Result | Notes | Gate before closing |
|---|---|---|---|---|
| #2030 s_nor / s_nand on lane masks | `ef54c20a6` | merged | conflict in `gfx2hlsl.py` resolved: SCC deferred after the lane-complement write | none recorded |
| #2079 v_frexp, v_cvt_flr / rpi, v_ldexp | `53b4f4a5f` | merged | `CMakeLists.txt` lists kept both | PASS |
| #2096 NGG pass-through without the LDS round trip | `27f400935` | merged | both sides kept in `gfx2hlsl.py`; no method defined twice | none recorded |
| #2741 binary16 VOPC compares | `c9439d044` | merged | clean | PASS |
| #2745 first-lane read on a 32-thread groupshared array | `92ceff230` | skipped | superseded by #2791 (its description says so) | none recorded |
| #2747 `s_wqm exec, exec` on a narrowed EXEC as a quad vote | `aa473ac85` | merged | stacked on #2736 (already in main): only its own commits applied; main's later refusal of a restored pre-`s_wqm` mask is kept | PASS |
| #2755 pixel `image_store` with d16 | `eb177fdde` | merged | docstring conflict only | PASS |
| #2758 per-lane `sample_b` bias past a divergent EXEC | `26e4d012c` | merged | stacked on #2747: only its own commits applied; the old `bias_per_lane` refusal removed (now drawn as `bias_lane`) | PASS (Linux; Windows not recorded) |
| #2760 `image_gather4_lz` with repeat addressing | `596bd3d00` | merged | gate found the draw harness builds a clamp sampler (`pixel_aot_render.cpp check_pixel_gather`), translator judged right; the WARP draw test of the repeat edge is expected to fail until the harness reads the manifest sampler | FAIL (harness) |
| #2761 pixel resource-heap image record | `9b7f6f976` | merged | gate FAIL: the pixel runtime dropped the `srv ... heap` line; #2763 (below) adds the runtime parser and refuses unsupported heap lines, so take both together | FAIL (closed by #2763) |
| #2763 bind the resource-heap image record of a pixel draw | `80a17a3d2` | merged | clean; head is 2 commits after the gated `5989abe79` (heap-mark fix, main merge) | PASS at `5989abe79` |
| #2772 refuse compute branches that decide for the whole wave | `157d06f64` | skipped | breaks two suites: refuses correct programs (`startup_gfx2hlsl_compute_mask_spill_check` fixture V from main's #2759: a uniform mask through a lane spill; #2030's loop in `startup_gfx2hlsl_sop2_complement_check`). Merged, then reverted (`fe3ea8795`). A new version must carry uniformity through lane spills and loops on uniform SGPRs | FAIL (vccz/vccnz inside an execz block, also on main) |
| #2784 32-bit integer scalar result ends the lane-mask tag | `473429b53` | merged | clean | PASS |
| #2789 `v_fma_mix` reading the low half of a plain `v_interp_p2_f16` | `959b38b64` | merged | its test's evaluator read the old `a * b + c` text; main's #2818 writes `mix32(...)`: evaluator updated (`795b7b9ec`) | PASS |
| #2791 WaveSync and the guest wave size for compute | `2c7d1bb78` | merged | gate finding G1 fixed here (`7e305ba9d`): a first-lane read of a per-lane VGPR is not group-uniform; new case in `startup_gfx2hlsl_wavesync_check` (fails without the fix) | FAIL (G1, fixed on this branch) |
| #2792 undefined upper half through `v_mov` / `v_cndmask` copies | `92f8ee9fd` | merged | clean | none (abandoned) |
| #2793 buffer_store_format index bound without a division | `3f1f05bd6` | merged | clean | none (stopped) |
| #2796 mesh parameter exports and pixel reads up to param31 | `ca2881a4e` | merged | clean | none (stopped) |

## Stage branches (other PRs closed on 2026-11-24 to 26)

| Branch | Head | What it brings here |
|---|---|---|
| `tx-stage-26` | `96e51beaf` | #2786 (slot 57 mesh buffer reads). Its #1951 duplicates #1408, already on `tester-loop`: `debug_launch.py` keeps #1408's hold-and-step; #1951's `step_gate.py`, its check and CI line are dropped |
| `tx-stage-25` | `50eace9c4` | #2067, #2048. #1726 dropped: its `startup_mrt_check` change needs a `startup_stream_draw` part that was not carried, and `startup_mrt` failed on Windows (reverted here too) |
| `tx-stage-24` | `0435d6c8b` | nothing new: #1801 is the older copy of #1939 (kept as on `tester-loop`); #1170's concurrency block is already in main's `windows.yml` |
| `tx-stage-shader` | `186904f55` | #1946, #2005, #2029 (translator PRs closed before 2026-11-26 that merged cleanly; the others need a hand merge) |
| `tx-stage-hand` | `6ceaafb85` | hand merges, batch 1: #1724 (1D `image_load`/`load_mip` on the existing 1D path; `startup_gfx2hlsl_image_1d_check` updated) and #1952 (code already here, its check added). Not merged: #1812 (`class_f16` already here, its test expects other helper text), #1973 (its test refuses formatted buffer loads that this branch lowers). #1796 skipped: duplicate of #2012. Win2025 gate PASS on `5ad36b656` (image_1d, mul_hi_u32, fract, dev and release); the `v_mul_hi_u32`/`v_mul_hi_i32` WARP and hardware readbacks (`startup_gfx2hlsl_mul_hi_*`) pass 7/7 on `3dbbbe68a` and 6 of 7 fail with the carry term dropped |
| `w7-mesh-rfl-marker` | `90b0b747b` | the opt-in engine (`tools/startup/gfx2hlsl_engine.py`, `X2W_ENGINE=1`, default off; see `docs/startup-engine.md`) and its single-block mesh marker fix. The engine's copies of the WaveSync and slot 57 buffer-read checks are `tests/startup_engine_wavesync_check.py` and `tests/startup_engine_mesh57_buffer_read_check.py` (CTest `startup_engine_*`); the default checks keep their names. Engine keys use revision 29 like the runtime here |

## Other fixes on this branch

- Since #2815 the runtime writes `revision.txt` into the shader cache: the draw checks that expect one `.txt` there
  (logical view, address SRV, joined table pointer, d16 SRV, MUBUF descriptor, pixel UAV, pixel heap SRV) skip it.

## Checks

- Linux: every `tests/startup_gfx2hlsl_*_check.py` (136) on the branch head; the two that need Windows arguments
  (`export_null_draw`, `mul24_align_draw`) exit 1 the same way on `tester-loop`.
- Each merged PR's own tests after its merge; the engine and default checks touched by the `w7` merge.
- Windows Server 2025, D3D12 WARP, dev, on `a27cddc31` (the translator PRs only), CTest regex
  `^(startup_gfx2hlsl_|startup_image_(sample_b|get_lod|gather_lz)|startup_pixel_(image_store|heap_srv)|startup_compute_(readfirstlane|request|store_d16_x|store_format_xyzw)|mesh_param_link|startup_shader_cache_|startup_indirect_draw_indexed|startup_resource_retire_title)`:
  410 of 424 passed. Failed: `startup_pixel_heap_srv_*` (8, the `revision.txt` count, fixed above),
  `startup_image_gather_lz_{warp,hardware}` (the draw helper has no `--check-pixel-gather`, see #2760 above),
  `startup_gfx2hlsl_f16_alu_draw_warp` and `startup_gfx2hlsl_cvt_pk_int16_hardware` (`PermissionError` replacing an
  entry of the shared DXC cache that another test held open), `startup_resource_retire_title_{warp,hardware}_debug0`
  (the warp one passed when run again alone). Release was not reached in that run.
- Known issue: `startup_pixel_heap_srv{,_word}_{fxc,dxc}_{warp,hardware}` (8 draws of #2763) pass when run alone
  (CTest regex `^startup_pixel_heap_srv`, `148786973`, dev and release 9/9) but fail inside the large
  `^startup_gfx2hlsl|^startup_pixel_heap|^startup_compute` run: there the request the consumer publishes reaches the
  translator with a root signature without the heap-indexing flag (0x400), so it is refused (`s_load through s10`).
  The cause under parallel load is not found yet.
- A later Windows run reported 470/470 for `^startup_gfx2hlsl|^startup_pixel_heap|^startup_compute`, but it built
  another head, not this branch, so it does not count; a run of that regex on `4d8b44580` is pending.
