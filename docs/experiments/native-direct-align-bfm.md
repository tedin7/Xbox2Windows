# v_alignbit_b32, v_alignbyte_b32 and v_bfm_b32 in the direct C++ compute path (#2985)

The direct compute frontend (`src/gfx1013_direct_compute.inl`) now lowers three VOP3 bit operations that it
refused with `direct DXIL lowering for <name>`. The semantics are the expressions of the `gfx2hlsl.py` VOP3 table:

- `v_alignbit_b32`: bits [k + 31 : k] of the 64-bit value {S0, S1}, with k = S2[4:0]. A zero k returns S1.
- `v_alignbyte_b32`: the same with k = 8 * S2[1:0].
- `v_bfm_b32`: ((1 << S0[4:0]) - 1) << S1[4:0], the rule `s_bfm_b32` already uses in this path.

Shift amounts are masked to five bits. The zero case is a select, so the masked shift by 32 - k in the other arm
stays in range.

These three were found with a local probe that is not part of this change. It compiles one instruction per VOP3
name of `src/gfx1013_isa_tables.inl` (VOP3 encoding, VGPR operands) through `compile_dxil ... gfx1013-direct-cs@0`.
On `cpp-migration` `05438d65` it saw 79 names refused, and these three had a `gfx2hlsl.py` lowering and no issue or
PR at the time.

## Fixture

`tests/native_align_bfm_check.py controls|hardware`
(`ctest --preset dev --no-tests=error -R '^native_align_bfm_' -V`). The words were assembled with LLVM 15.0.7 (`clang
-cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm -show-encoding`).

- The dispatch has two groups of 32 lanes. Lane L of group g uses S0 = 0x89abcdef, S1 = 0x01234567 and
  S2 = L + 32g, so every shift 0 to 31 and every byte count appears. Group 1 repeats them with bit 5 set, which the
  field must ignore. `v_bfm_b32` takes S0 = S2 and S1 = S2 ^ 0x15. The 192 results and the guard words are compared
  with a CPU reference.
- The controls compile the program and check that the references with the zero shift not special-cased, with S0
  and S1 exchanged, and with the field not masked each differ from the expected words.

Measured on 27.11.2026, `cpp-migration` `05438d65` plus this change, Windows 10.0.26100.9711. The controls printed
`compiled; the zero-shift, exchanged-source and unmasked references each differ from the expected words`. The
readback printed `requires shader model 6.6 with wave32` and reported a skip (77) on the NVIDIA Quadro M1000M (driver
32.0.15.7371, debug layer unavailable). The readback is not measured.
