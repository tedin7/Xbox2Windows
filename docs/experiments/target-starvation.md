# Render target starvation diagnostics

The render totals report a target as starved when a preceding window recorded a draw to it,
more than 30 frames have elapsed since the last draw to it, and a translated draw has read
it within the last 30 frames. Counts describe recorded draws. A refusal is associated with
its first colour attachment, or its depth attachment for a depth-only pass. Up to three
pipeline keys are shown per target, ranked by calls, with the most recent cause for each
pair. The census reports omissions at its limits of 512 targets and 1,024 key-target pairs.
Writer captures include refused attachment rows with the cause. The tester pack places
starved targets below the render summary in `null-inputs.txt`.

## Verification

The change is based on `d8f225d51`. On Linux, `python -B tests/startup_tester_pack_check.py`
passed with synthetic log rows for a zero-draw window, both starvation messages and
pipeline targets. The Windows checks below are pending.

```powershell
ctest --preset dev --no-tests=error -R '^(startup_target_starvation(_control|_mutant)?_warp|startup_null_cause_labels(_buffer_token_control)?|startup_tester_pack)$'
```

The WARP fixture reads back all 40 frames and compares the output pixels against a CPU
reference. It expects 39 refusals across direct and indirect draws. The continuing-writer
control expects no starvation row. The mutant books refusals without their target and
must fail the diagnostic assertion after the pixel checks succeed.
