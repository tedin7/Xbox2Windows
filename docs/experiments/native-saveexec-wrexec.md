# Remaining SOP1 `*_saveexec_*` and `*_wrexec_*` instructions in direct Compute DXIL

Issue #3296 (Refs #2935) lowers the remaining 12 SOP1 EXEC-mask instructions in
the direct Compute DXIL path (`src/gfx1013_direct_compute.inl`), updates
`uses_scc_after()` so `SCC` liveness treats all scalar mask and SAVEEXEC/WREXEC
definitions consistently, and updates `transfer()` in
`src/gfx1013_compute_native.cpp` so all wave32 `*_saveexec_b32` and
`*_wrexec_b32` instructions clear symbolic provenance on both the destination
register and `EXEC_LO`.

## Lowered Instructions

- **Wave32 (`SOP1`, 32-bit):**
  - `s_xor_saveexec_b32` (`0x3e`): `D = EXEC_LO; EXEC_LO = S0 ^ EXEC_LO`
  - `s_andn2_saveexec_b32` (`0x3f`): `D = EXEC_LO; EXEC_LO = S0 & ~EXEC_LO`
  - `s_orn2_saveexec_b32` (`0x40`): `D = EXEC_LO; EXEC_LO = S0 | ~EXEC_LO`
  - `s_nand_saveexec_b32` (`0x41`): `D = EXEC_LO; EXEC_LO = ~(S0 & EXEC_LO)`
  - `s_nor_saveexec_b32` (`0x42`): `D = EXEC_LO; EXEC_LO = ~(S0 | EXEC_LO)`
  - `s_xnor_saveexec_b32` (`0x43`): `D = EXEC_LO; EXEC_LO = ~(S0 ^ EXEC_LO)`
  - `s_andn1_saveexec_b32` (`0x44`): `D = EXEC_LO; EXEC_LO = ~S0 & EXEC_LO`
  - `s_orn1_saveexec_b32` (`0x45`): `D = EXEC_LO; EXEC_LO = ~S0 | EXEC_LO`
  - `s_andn1_wrexec_b32` (`0x46`): `EXEC_LO = ~S0 & EXEC_LO; D = EXEC_LO`
- **Wave64 (`SOP1`, 64-bit):**
  - `s_andn1_saveexec_b64` (`0x37`): `D = EXEC; EXEC = ~S0 & EXEC`
  - `s_orn1_saveexec_b64` (`0x38`): `D = EXEC; EXEC = ~S0 | EXEC`
  - `s_andn1_wrexec_b64` (`0x39`): `EXEC = ~S0 & EXEC; D = EXEC`

All 12 instructions update `SCC` from `(new_exec != 0)` across all participating
bits (`EXEC_LO` in wave32; both `EXEC_LO` and `EXEC_HI` in wave64).

## Destination & Mode Rules

- Newly lowered wave32 ops require `dst <= 107` (`SGPR` / `VCC`), refusing
  `M0`/`EXEC_LO` destinations with `unsupported wave32 SAVEEXEC destination`, and
  are restricted to wave32 compute shaders.
- `s_andn1_wrexec_b64` matches `s_andn2_wrexec_b64` rules (`compute && wave64`,
  even SGPR pair `<= 104` for both `dst` and `src`).

## Verification

- [native_saveexec_wrexec_check.py](../../tests/native_saveexec_wrexec_check.py)
  covers all 12 instructions across non-zero (`SCC = 1`), zero-result
  (`SCC = 0`), high-word-only (`SCC = 1` from `EXEC_HI`), `dst == src` aliasing,
  `vcc_lo`/`exec_lo`/inline/literal sources, partial-wave dispatches, and 8
  negative operand/mode refusal cases.
- CMake/CTest targets (`tests/cmake/native_saveexec_wrexec.cmake`):
  `native_saveexec_wrexec_ir`, `native_saveexec_wrexec_controls`, and
  `native_saveexec_wrexec_hardware`.
