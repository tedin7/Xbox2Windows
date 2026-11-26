# Opt-in translation engine (fork tester-loop-tx only)

`X2W_ENGINE=1` makes `tools/startup/gfx2hlsl.py` hand over to `tools/startup/gfx2hlsl_engine.py`: a complete copy of the translator
with the wave-wide engine (WaveSync gathers, mesh wave-wide execz skip, tick mode for loops, barriers and lane reads, lane-class
definedness with an opt-in zero-init tier, compute heap rows). Default off: with the variable unset nothing changes, the default
translator never imports the engine (`tests/startup_engine_hook_check.py`).

Keys: the engine's pipeline and compute keys hash `-engine1` after the revision string, and `src/startup_stubs/d3d12_draw.inl` does the
same when `X2W_ENGINE=1`, so an engine run and a default run never share a cached translation or refusal. Set the variable for the
game process and for the translator (`tester_run.ps1` starts both from one environment). Bump the suffix with every behaviour change.

Tests (same code as the default tests, run against the engine): `tests/x2w_engine_shim.py` makes `import gfx2hlsl` resolve to the
engine module. `startup_engine_compute_check.py`, `startup_engine_readfirstlane_check.py` and `startup_engine_waterfall.py` are the
engine versions of the compute checks; the wavesync, tick, tick-loops, execz-wave, mesh57 and heap-row checks drive the engine
directly and read results back on WARP / hardware (CTest names in CMakeLists.txt).

How the copy is made (regenerate, do not hand-edit): merge the engine branches (bench/engine, bench/c4-loops, bench/c2-r11, the
tick readback tests) into a branch from tester-loop, resolve conflicts on the engine side, copy `tools/startup/gfx2hlsl.py` to
`gfx2hlsl_engine.py`, add `ENGINE_KEY_SUFFIX` to the two key seeds. Pixel-stage lowerings added to `gfx2hlsl.py` later (list A) are
NOT in the copy until it is regenerated.

Known gaps (listed because tests were changed or skipped, not hidden):
- Heap-row rule meets tick mode: branch targets between the mask, the stride and the load are accepted by tick mode; the three
  controls that refused them are disabled in `startup_gfx2hlsl_heap_row_check.py` until the rule knows joins.
- A guest barrier inside an execz-skipped block is translated by tick mode (park site), not refused; `startup_compute_execz_wave_check.py`
  accepts that.
- Engine merge broke c3's #2702 rule (`s_wqm` keeps the lane-mask tag): fixed in the copy, found by `check_lane_mask_wqm_keeps_tag`.
- bench/c25-tick e85cf7d76 (newer tick restructure) is not merged: conflicts with the tick sites already in; bench/c6-r5 (pixel
  fma_mix) is covered by list A.
