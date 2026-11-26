# Tester loop branch

This branch is `origin/main` plus the tooling a tester needs for one fast round trip: run one command, send back
the zip it prints. It is not a release; upstream gets each piece through its own pull request.

## On top of main

| Piece | Source | Head taken |
|---|---|---|
| `raw=` and `completed=` in the DirectStorage status lines | PR #2817 | `0355a7745` |
| revision warning for the shader cache and the translator (`revision.txt`, `-FreshShaderCache`) | PR #2820 (issue #2815) | `1bc4a534a` |
| periodic window capture (`-CaptureEvery`, `--capture-every`) | PR #2822 (issue #2819) | `0d4a8afe4` |
| opt-in lazy slot 133 textures (`X2W_D3D12_LAZY_TEXTURES=1`): external PR #2773, opt-in; its conflict with main is resolved on this branch only | PR #2773 (triggered02) | `92db6d4eb` |
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
   not (`-NoBuild` stops with a message instead).
2. Uses one shader cache folder per shader revision, `out/startup/shader_cache_<rev>` (for example `shader_cache_28-28-3`,
   from the `x2w-startup-shader-N` strings of `gfx2hlsl.py`): kept across runs and across branch updates that leave
   the revision alone, a new empty one when a revision is bumped. The `--watch` translator needs minutes to catch up,
   so emptying the cache on every run would leave pipelines untranslated and the screen black.
3. Runs `run.ps1 -Render -Translate -FakeUser -GpuMirror -ContinueAsserts 1000` with `X2W_D3D12_BUDGET=1`.
4. Writes `out/tester/report_<commit>_<yyyymmdd-hhmm>.zip` and prints its path, its size, how many translations were
   already in the shader cache at start, and when the title first polled the pad.

`-Input menus` presses D-pad down, D-pad down, A, one press every 6 s, starting 5 s after the first pad poll and up to 500 s. The input
script clock starts at that poll, which the title reaches late, so give the run enough seconds (the default is 600).
Any other `-Input` value is a raw `X2W_INPUT_SCRIPT`.

Lazy slot 133 textures are on for now: the script sets `X2W_D3D12_LAZY_TEXTURES=1` so read-only slot 133 textures are created
on first use and video memory does not run out. `-NoLazyTextures` turns it off; the manifest records the choice.

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
