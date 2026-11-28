# Startup UI regression matrix

This note describes a report tool for the startup UI sampler, BC7 and indexed-rectangle work. It runs
existing CTest cases and does not change the runtime, the shaders or the fixtures. The matrix is a
way to collect comparable pass, skip and fail rows across builds; it is not a new rendering
measurement.

The default rows are:

| CTest family | Related change | What the fixture measures |
|---|---|---|
| startup_sampler_heap_wrapped_{warp,hardware} | #3153 | The wrapped D3D12 path creates a three-record sampler heap at the title's 16-byte stride, returns the title CPU handle, creates two distinct samplers, copies a record and a two-record range, and checks the destination words. The test enables X2W_D3D12_PERMISSIVE=1 and X2W_D3D12_DEBUG=1. |
| startup_xg_bc7_draw_{warp,hardware} | #3153 | A source-built BC7 surface is uploaded through the wrapped XG path with X2W_XG_BC7_4K_S_LAYOUT=1. The draw checks a 64x64 surface (4,096 pixels) and the visible part of an 81x69 surface (5,589 pixels), for both linear and 4 KiB tiled backing. |
| startup_rect_gap_{warp,hardware} | #3142 | A synthetic eight-record, 12-byte-stride stream is drawn as an R16 indexed triangle list and as six- and eight-vertex rectangle lists. A mask-controlled pixel shader and a constant-colour control distinguish coverage from attribute interpolation. The check compares all 32 target pixels with a point-in-triangle model, including a 15-degree rotation and the back-face-culling control. |
| startup_ui_texture_bands_{warp,hardware} | #3141 | A source-built 32x32, three-mip RGBA8 texture follows an indexed gradient draw and an indexed textured UI draw. The stage-by-stage check covers a linear CopyTextureRegion reference, a slot-133 copy, a compact small-texture path, and slot-70 0x105/0x10a layouts, with wrong-row-pitch negative controls. |

The first two families are registered in the base tree used for this note. startup_rect_gap is
registered by PR #3142 and startup_ui_texture_bands by PR #3141. A build that contains only one of
those changes reports only the families it registers; the matrix does not invent missing tests.

## Run it

From the repository root on Windows, enter the developer environment and build the selected
configuration:

~~~powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev
~~~

List the fixture names that this build actually registers:

~~~powershell
ctest --preset dev -N -R '^(startup_sampler_heap_wrapped|startup_xg_bc7_draw|startup_rect_gap|startup_ui_texture_bands)_(warp|hardware)$'
~~~

Run the default matrix:

~~~powershell
python -B tools/startup/ui_regression_matrix.py `
  --build-dir out/build/dev `
  --json out/ui-regression-matrix.json `
  --output out/ui-regression-matrix.md
~~~

The script accepts --ctest for a CTest executable, --adapter warp or --adapter hardware to restrict
the rows, and --filter (or one positional filter) for a CTest name regex. For example, this run
reports only the rotated-rectangle regression introduced by PR #3142:

~~~powershell
python -B tools/startup/ui_regression_matrix.py `
  --build-dir out/build/dev `
  --filter '^startup_rect_geometry_(warp|hardware)$' `
  --json out/rect-geometry-matrix.json `
  --output out/rect-geometry-matrix.md
~~~

The tool first asks CTest for the matching names with --show-only=json-v1, applies the adapter
filter to that list, and then runs exactly those names with -R. It reads CTest's JUnit report, so
a CTest testcase with no failure or skip marker is the only thing counted as a pass. A notrun
testcase is a skip only when its skipped message starts with SKIP_RETURN_CODE or
SKIP_REGULAR_EXPRESSION; a missing executable, missing configuration or disabled test is a fail.
A testcase carrying a skipped element after a runnable status is a skip; a failure or error is a
fail. The Markdown
table keeps the three states visible and the JSON object has the same rows plus the summary counts
and the CTest command.

Exit status is 0 when every selected row passed, 1 when at least one row failed or CTest returned
nonzero without a failed row, 3 when there were no failures but at least one skip, and 2 for a
malformed or empty selection, a missing CTest executable or a build that produced no usable JUnit
report. A skip therefore never turns a matrix green. A selected build with no matching tests is
rejected before any output is written.

## What remains a limit

- The matrix observes the fixtures registered by the current build. It does not merge PR #3141 or
  PR #3142, and a family absent from that build is not a failure row.
- startup_rect_gap does not reproduce the exact indexed UI draw from issue #3131. Its synthetic
  input passes without showing the reported diagonal gap. The fixed rotated-corner case is in
  startup_rect_geometry; the #3142 note records the remaining unverified candidates.
- startup_ui_texture_bands demonstrates the tested upload and sampling routes. It does not prove
  that the original UI draw in #3132 has the same input, texture layout or draw path.
- The adapter suffix in a CTest name is the requested backend. It is not proof that the runtime
  selected that device: the rectangle fixture, for example, can report the Microsoft Basic Render
  Driver for a hardware invocation. Record the observed adapter, driver, Windows build and
  debug-layer status separately when a row is used as hardware evidence.
- A skip is a capability or environment result, not a pass. CTest's notrun status also covers a
  missing executable, a missing multi-config configuration and a disabled test; those rows are
  failures unless the skipped message explicitly carries SKIP_RETURN_CODE or
  SKIP_REGULAR_EXPRESSION. A nonzero CTest exit with no failed row is also a failure.
- The matrix does not run the full startup, does not load game files and does not establish a
  playable port or a frame rate.

## References

- [PR #3153: runtime: restore menu sampler bindings and BC7 display uploads](https://github.com/R0Y00/Xbox2Windows/pull/3153)
- [PR #3141: runtime: trace UI texture row-pitch bands and fix XG mip-0 and CopyTextureRegion tracking](https://github.com/R0Y00/Xbox2Windows/pull/3141)
- [PR #3142: startup: identify a rotated rectangle corner by its diagonal](https://github.com/R0Y00/Xbox2Windows/pull/3142)
- [Issue #3131: native UI gradient draw leaves a diagonal gap](https://github.com/R0Y00/Xbox2Windows/issues/3131)
- [Issue #3132: horizontal color bands from a textured UI draw](https://github.com/R0Y00/Xbox2Windows/issues/3132)

The PRs provide the regression matrix and its fixtures. This note makes no closing claim for
#3131 or #3132; those conversations remain with the authors and reviewers of the respective PRs.
