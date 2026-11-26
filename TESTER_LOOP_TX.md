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

## Checks

- Linux: every `tests/startup_gfx2hlsl_*_check.py` (136) on the branch head; the two that need Windows arguments
  (`export_null_draw`, `mul24_align_draw`) exit 1 the same way on `tester-loop`.
- Each merged PR's own tests after its merge.
