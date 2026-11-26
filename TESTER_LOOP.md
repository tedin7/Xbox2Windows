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
`-Input menus2` presses the same buttons, but after every A it waits until the presented picture has changed and stayed still for 2 s (at most 90 s) before the next press, so nothing is pressed blind while a screen loads. It needs render mode, which the script always uses.
Any other `-Input` value is a raw `X2W_INPUT_SCRIPT`.

Lazy slot 133 textures are on for now: the script sets `X2W_D3D12_LAZY_TEXTURES=1` so read-only slot 133 textures are created
on first use and video memory does not run out. `-NoLazyTextures` turns it off; the manifest records the choice.

`-Deep` also turns on `-TraceCalls` and `-TraceApi`: slower, more detail in the logs.

## What to send

Send the zip it printed. If the zip is over 20 MB it is split into `report_..._part01of03.zip`, `..._part02of03.zip`, ...:
send every part (mail limit). The zip holds every file the run wrote in its log folder (stdout, stderr, events,
finished, threads, files, trap.log, translate, translate_err, the window captures, the debugger summary) and
`manifest.json` (commit, build commit, dirty flag, seconds, input script, `X2W_*` variables, graphics card and driver,
Windows build, wall time). It holds no game files, no shader cache and no memory dumps.
