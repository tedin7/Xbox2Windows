# Compute lane masks as scalar words

The compute lowering reconstructs low mask words for mixed-input
`s_nor_b32`, `s_lshl{1,2,3,4}_add_u32` and `s_mul_i32`. It accepts
straight-line, one-dimensional groups of two to four threads and requires
entry EXEC or an unmodified compare mask. It uses ballots keyed by logical
thread index, independent of the host's physical lane numbering. The result
of NOR supplies a symbolic lane bit and sets SCC from the whole word.
Shift-add records unsigned overflow, including bits shifted out of the first
operand. Multiply keeps the existing SCC value.

The four-thread fixture compares all 320 UAV words against a CPU reference.
It exercises sparse, empty and full compare masks, both operand positions
for shift-by-four and multiply, and SCC readback. For every variant, a shader
mutant that votes only for logical lane zero must produce the predicted
readback mismatch. Python controls check the original refusals with the
lowering disabled, a five-thread group, and a mask overwritten by another
scalar operation.

```sh
python3 -B tests/startup_compute_mask_word_check.py
```

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_compute_mask_word_(controls|warp)$'
```

The Python controls passed on Linux against the change based on `e8015da7a`.
All 33 existing compute fixture HLSL and manifest pairs matched the base
revision byte for byte. WARP readback awaits the Windows test run.
