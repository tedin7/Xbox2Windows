# Compute wave32 EXEC early-out and VCC scalar pair

The synthetic early-out fixture covers a single guest wave with one
`s_cbranch_execz` and one forward `s_branch`. The taken block clears the
running mask and jumps to the EXEC restore and mask intersection. Every
thread writes its EXEC bit to group-shared memory before the branch. The
group synchronizes once and all threads take the same decision. The fixture
runs with zero and eight active lanes. The per-lane branch mutant must
first differ at word 32.

The VCC pair fixture loads an ordinary scalar pair into `vcc` while EXEC
is full. A compare overwrites the low word before any read can observe it.
The high word survives another compare inside a skipped block and feeds a
float add at the join. The mutant zeroes the copy of the high word and
must first differ at word 1. The translator keeps the existing refusal
when it cannot prove wave32 or the pair's low word can reach a read.

At baseline `234997f8c`, the new fixtures were refused for the live
definitions of `s20` and `vcc_hi` respectively. The baseline census also
reports `branch out of the block` for the early-out. With the change, both
translate and the original 34 compute fixture HLSL outputs compared
byte-identically against that baseline. Local checks on x86-64 Linux
with Python 3.14.8 passed:

```bash
python3 -B tests/startup_gfx2hlsl_compute_vcc_hi_check.py
python3 -B tests/startup_gfx2hlsl_compute_check.py
python3 -B tests/startup_gfx2hlsl_desc_vcc_check.py
python3 -B tests/startup_gfx2hlsl_compare_dst_check.py
python3 -B tests/startup_gfx2hlsl_vcc_hi_check.py
```

Run the GPU readbacks and their mutants on Windows:

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_vcc_hi_(early-out|vcc-pair)_warp$'
```

WARP execution and tester shader coverage remain unverified. The
early-out proof requires the converged single-wave diamond. The
general live-scalar and 64-bit mask-phi cases remain open.
