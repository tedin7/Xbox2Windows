# Nested compute EXEC inside a per-lane skip

The compute translator retains the existing lowering for blocks whose EXEC
writes already passed the clear-mask proof. The additional proof covers a
straight fallthrough region under `s_cbranch_execz`. It tracks which saved
masks are still available and requires a restore inside the block or an
unmodified pre-block copy restored at the join. A restore inside the block
can only use the mask at the skip or a copy made inside it.

A lane outside the skip cannot run the block in HLSL. Consequently an OR
source must be clear for every excluded lane, even if a later instruction
restores EXEC. A predicated vector compare inside the block can provide that
proof. The accepted lowering stores each local copy in a Boolean and guards
vector writes using the current EXEC bit. Saved values cannot escape as
numeric words. The translator reports a missing restore or a wave-wide
read of the narrowed mask when it encounters either in the additional proof.
The instruction semantics are in the [RDNA 2 ISA reference](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture).

## Reproduction

Run the synthetic translation and encoding checks on Linux:

```sh
python3 -B tests/startup_lane_exec_check.py --llvm-mc llvm-mc-19
python3 -B tests/startup_gfx2hlsl_compute_check.py
python3 -B tests/startup_gfx2hlsl_saveexec_same_reg_check.py
python3 -B tests/startup_compute_readfirstlane_exec_check.py
```

The controls passed on Linux. The seven positive programs exercise AND and
OR with both mask widths, explicit compare into EXEC, stacked saves, and a
restore at the join. Each produced the old EXEC-write refusal on the base
revision. The three negative programs cover a count of EXEC, a missing
restore and an OR that would turn on lanes outside the skip. A comparison
against the base revision produced byte-identical HLSL for 33 existing
compute fixtures.

On Windows, after configuring and building the dev preset:

```powershell
ctest --preset dev --no-tests=error -R '^startup_lane_exec(_warp)?$'
```

The WARP test dispatches 64 threads per fixture and compares all 320 output
words against a CPU reference, including lanes absent from the outer mask
and unwritten padding. Each fixture has mixed, empty-outer and empty-inner
inputs. The two HLSL mutants per fixture must produce a readback mismatch
at the CPU-predicted first word. One drops the guard on the first store. The
other drops the restore before the second store. WARP and these GPU mutants
await execution on Windows.
