# Tester loop branch

This branch is `origin/main` plus the tooling a tester needs for one fast round trip: run one command, send back
the zip it prints. It is not a release; upstream gets each piece through its own pull request.

## On top of main

| Piece | Source | Head taken |
|---|---|---|
| `raw=` and `completed=` in the DirectStorage status lines | PR #2817 | `0355a7745` |
| revision warning for the shader cache and the translator (`revision.txt`, `-FreshShaderCache`) | PR #2820 (issue #2815), with the fix `d39f65cfc` | `d39f65cfc` |
| periodic window capture (`-CaptureEvery`, `--capture-every`) | PR #2822 (issue #2819) | `0d4a8afe4` |
| opt-in lazy slot 133 textures (`X2W_D3D12_LAZY_TEXTURES=1`): external PR #2773, opt-in; its conflict with main is resolved on this branch only | PR #2773 (triggered02) | `92db6d4eb` |
| C++ exception record decoded into a sanitized summary | PR #1939 | `49ca19e34` |
| name the reason a BCPack request fails | PR #1940 | `7de1131b9` |
| causes of a refused placed build, PERFORM_UPDATE with a source structure | PR #1943 | `5a79addf7` |
| hold other threads while a hook steps over its byte | PR #1408 | `08f0f4e9f` |
| `startup_runtime` target and `runtime` build preset (`cmake --build --preset runtime`) | PR #2827 (branch `x2w-startup-runtime-target`) | `eb82f1d39` |
| sampler_x linear tolerance (one 8-bit weight step) | PR #2794 | `6ceb980ba` |
| refusal cache for SRV buffer requests | PR #2782 | `a4eeb366d` |
| fast draw path | PR #2766 | `619ae7009` |
| `-Input menus2` (waits for a new picture after each confirm) | fork branch `c24-menus2`, gate PASS | `43a382c7b` |
| fix for #2773 (lazy textures), gate PASS | fork branch `fix-2773` | `5ad90a328` |
| slot 70 texture_unavailable causes (row 3), gate PASS | fork branch `c23-slot70-texture-causes` | `d814a2e9e` |
| window capture of flip-model windows through PrintWindow (black BitBlt) | fork branch `c22-capture-flip` | `1dd66de57` |
| refused draws counted as "not drawn" with reasons | fork branch `c22-notdrawn` | `a3d15d7bc` |
| commit counters in the heartbeat | fork branch `x2w-commit-counters` | `3c7b2a56f` |
| watch translator serves the newest requests first | fork branch `x2w-watch-newest` | `e7b617bcb` |
| residency slice 1 | fork branch `f4-residency-s1` | `df147080e` |
| tester_run turns on `X2W_XMEM_STARTUP_BELOW_40` (`-NoStartupBelow40` off) | fork branch `c15-bvh40` | `2b9643b13` |
| tester_run: finish translation after the run, PowerShell 5.1 native stderr fix | fork branch `c18-tester-run` | `8bab8da2d` |
| 1D and 3D image_store | fork branch `c22-img-1d3d` | `d17d221a9` |
| residency slice 1 (update) | fork branch `f4-residency-s1` | `5cc2a859c` |
| per-reason SRV shape table | fork branch `c22-srv-shapes` | `3a826e171` |
| tester pack: reasons in the report | fork branch `tester-pack-reasons` | `10a88f822` |
| image_gather4_lz_o with a constant packed offset | fork branch `x2w-w5-gather-lz-o` | `0c5b82846` |
| DPP in compute (w6) | fork branch `w6-dpp-cs` | `49bc35454` |
| execz refusal names its cause (text only) | fork branch `c7-execz-cause` | `dd0d392e3` |
| opt-in drop of the top mip of big sampled textures (`X2W_D3D12_TEXTURE_DROP_MIPS=<levels>`), with the manifest hint `srvuse` | this branch (branch `c7-s4-dropmips`) | its own tests: `startup_texture_drop_mips`, `startup_srvuse_manifest`, `startup_texture_drop_mips_draw_*` |
| `tools/startup/tester_run.ps1`, `tools/startup/tester_pack.py` | this branch | its own tests: `startup_tester_pack`, `startup_tester_run_ps1` |

Each merge into this branch is a merge commit that names the heads it took. Both PRs touch `tools/startup/run.sh`; the merge keeps both flags. Nothing is rebased or force-pushed.

## Run

From a PowerShell in the checkout, with the title's executable folder prepared as for `run.ps1`:

```powershell
.\tools\startup\tester_run.ps1 -Exe <path to the exe> -DataDir <package root> -Seconds 600 -Input menus -CaptureEvery 15
```

What it does:

1. Checks that the build in `out/build/release` was made from this checkout's current commit and builds when it was
   not (`-NoBuild` stops with a message instead). A build from another commit is used as it is when the runtime sources
   (`src`, `cmake`, `CMakeLists.txt`, `CMakePresets.json`) are the same, so pulling a tools-only fix needs no rebuild; the
   summary prints why and the manifest records it as `build_check`.
2. Uses one shader cache folder per shader revision, `out/startup/shader_cache_<rev>` (for example `shader_cache_28-28-3`,
   from the `x2w-startup-shader-N` strings of `gfx2hlsl.py`): kept across runs and across branch updates that leave
   the revision alone, a new empty one when a revision is bumped. The `--watch` translator needs minutes to catch up,
   so emptying the cache on every run would leave pipelines untranslated and the screen black.
3. Runs `run.ps1 -Render -Translate -FakeUser -GpuMirror -ContinueAsserts 1000` with `X2W_D3D12_BUDGET=1`.
4. Writes `out/tester/report_<commit>_<yyyymmdd-hhmm>.zip` and prints its path, its size, how many translations were
   already in the shader cache at start, and when the title first polled the pad.

`-Input menus` presses D-pad down, D-pad down, A, one press every second held 0.8 s (the 0.3 s default fell between two polls of a title at 2 to 3 fps), starting 5 s after the first pad poll and for the first 256 presses (about 260 s). The input Default (-PrivacyBoxes 2): the presses are clock-free passes for the privacy screen (up x3, toggle at most one box, down to Conferma, A; a different single box each pass, Gray-code order), repeated until the screen changes; -PrivacyBoxes 0 gives the plain pattern.
script clock starts at that poll, which the title reaches late, so give the run enough seconds (the default is 600).
`-Input menus2` presses the same buttons, but after every A it waits until the presented picture has changed and stayed still for 2 s (at most 90 s) before the next press, so nothing is pressed blind while a screen loads. It needs render mode, which the script always uses.
Any other `-Input` value is a raw `X2W_INPUT_SCRIPT`.

The launcher's job memory cap is set to 75% of physical RAM (at least 14 GiB; `-MemoryGiB` overrides): the default 14 GiB cap
ends a long run with error 1455. The summary prints it and the manifest records it.

Lazy slot 133 textures are on for now: the script sets `X2W_D3D12_LAZY_TEXTURES=1` so read-only slot 133 textures are created
on first use and video memory does not run out. `-NoLazyTextures` turns it off; the manifest records the choice.

The system-memory fallback is on too (`X2W_D3D12_SYSMEM_FALLBACK=1`): when a slot 133 texture runs out of video memory it is created
once more in system memory instead of returning an error, and the `x2w-budget` line ends with `sysmem-fallback=N bytes=M`.
`-NoSysmemFallback` turns it off; the manifest records the choice.

Lazy memory commit is on as well (`run.ps1 -LazyMemory`: `X2W_XMEM_LAZY_COMMIT`, `_LAZY_STARTUP`, `_SHARED_VIEWS`), which cuts the section commit; `-NoLazyMemory` turns it off and the manifest records it.

`-Deep` also turns on `-TraceCalls` and `-TraceApi`: slower, more detail in the logs.

## Drop the top mip (opt-in, off by default)

`X2W_D3D12_TEXTURE_DROP_MIPS=1` (1 to 4 levels) and `X2W_D3D12_TEXTURE_DROP_MIPS_MIN_BYTES` (default 4 MiB; needs `X2W_D3D12_LAZY_TEXTURES=1`):
a read-only slot 133 texture above the threshold is created without its top mip level(s) the first time a translated draw binds it for
plain sampling. The picture gets softer; nothing is read at a wrong place. A texture keeps every level when the shader loads texels,
asks for the size or the level of detail, when a copy, a compute view or a map comes first, when a view starts below the top level, or
when the translated pipeline carries no hint. The heartbeat line `x2w-drop-mips` counts the dropped textures, the estimated MiB saved and
every reason a texture kept its levels.

The hint is a manifest line `srvuse N sample|texel|lod` that `tools/startup/gfx2hlsl.py` writes next to `srv N ...` (metadata only,
the emitted HLSL is the same). No line means unknown, and unknown never drops. **Before this can go upstream**, R0Y00's C++ frontend
would need to write the same hint for its pipelines, and the translator change would go through the translator review; on this
branch it is a fork-only addition.

## What to send

Send the zip it printed. If the zip is over 20 MB it is split into `report_..._part01of03.zip`, `..._part02of03.zip`, ...:
send every part (mail limit). The zip holds every file the run wrote in its log folder (stdout, stderr, events,
finished, threads, files, trap.log, translate, translate_err, the window captures, the debugger summary) and
`manifest.json` (commit, build commit, dirty flag, seconds, input script, `X2W_*` variables, graphics card and driver,
Windows build, wall time). It holds no game files, no shader cache and no memory dumps.

## Waiting for a fix on their own branch (conflict with this branch)

| Branch | Head | Conflict |
|---|---|---|
| `c15-sysmem-fallback` | `f45c6849d` | `d3d12_xs_device.cpp`, `triage_run.py`, `tester_run.ps1` and their tests |
| `c21-g3-bufunavail` | `b9a06c4b5` | `d3d12_memory_srv.inl` (`unplaced_memory_buffer` signature with #2782) |
| `c7-s4-dropmips` | `8a95e07f9` | `d3d12_xs_device.cpp`: the lazy SRV code moved into `create_lazy_srv` (fix-2773); the drop-levels view handling has to move with it |
