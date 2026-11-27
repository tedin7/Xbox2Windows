# Binary16 <-> 16-bit integer conversions in the direct C++ compute path (#3148)

`src/gfx1013_direct_compute.inl` now lowers `v_cvt_f16_u16`, `v_cvt_f16_i16`, `v_cvt_u16_f16`, and `v_cvt_i16_f16`
(VOP1 and VOP3A) in the direct C++ compute path (`DirectCompute::float_alu()`):

- **High-half preservation:** Every operation reads `S0[15:0]`, writes the converted 16-bit result into `D[15:0]`, and
  preserves `D[31:16]`, matching LLVM `GCNSubtarget::zeroesHigh16BitsOfDest` (`"On gfx10, all 16-bit instructions preserve
  the high bits"`).
- **`v_cvt_f16_u16` / `v_cvt_f16_i16`:** Sign-extends (for `i16`) or zero-extends (for `u16`) `S0[15:0]`, converts to
  binary32 (`sitofp`/`uitofp`), rounds to nearest-even binary16 with `packed_half_rne()` (e.g. `2049 -> 2048`, `2051 ->
  2052`, `65519 -> 65504`, `65520 -> +inf`, `32767 -> 32768.0h`), and optionally applies VOP3 `clamp` (`[0.0, 1.0]`).
- **`v_cvt_u16_f16` / `v_cvt_i16_f16`:** Applies optional VOP3 `abs`/`neg` on `S0[15:0]`, widens with
  `@dx.op.legacyF16ToF32`, truncates toward zero, saturates to `[0, 65535]` / `[-32768, 32767]`, and maps NaN to `0`.
- **Mode & modifier contract:** Included in `half_binary` in `DirectCompute::lower()` so the default FP16 rounding
  (`round16 = 0` RNE) and denormal (`denorm16 = 3` keep, `FP16_OVFL = 0`) mode contract is enforced. Refuses DPP, SDWA,
  output modifiers (`omod`), `clamp` on `*_f16` integer targets, `op_sel` high-half bits, `abs`/`neg` on integer sources,
  unused VOP3 `src1`/`src2`, and non-default FP16 modes. Inline float constants use the same binary16 bit table as
  `v_add_f16`.

## Fixture

[`tests/native_cvt_f16_i16_check.py`](../../tests/native_cvt_f16_i16_check.py) via
[`tests/cmake/native_half_int_cvt.cmake`](../../tests/cmake/native_half_int_cvt.cmake) (`native_cvt_f16_i16_ir`,
`native_cvt_f16_i16_controls`, `native_cvt_f16_i16_hardware`):

- **IR (`native_cvt_f16_i16_ir`):** Evaluates lane 0 across 82 conversion cases (VGPR, SGPR, literal, inline integer,
  inline float, VOP3 `clamp`, `abs`, `neg`, `-|x|`, rounding ties, saturation, subnormals, infinities, NaNs, and
  non-zero upper halves) plus 38 refusal cases (DPP, SDWA, `omod`, `op_sel`, illegal `abs`/`neg`/`clamp`, `src1`, and
  MODE writes).
- **Controls (`native_cvt_f16_i16_controls`):** Compiles the full 82-case shader to DXIL via `compile_dxil` and checks
  `dxc.exe -dumpbin` plus all 38 refusals.
- **Hardware (`native_cvt_f16_i16_hardware`):** Replays all 82 conversions on hardware via `native_store_replay --direct`.

Measured on 27.11.2026, `main` `429f11d5` plus this change, Windows 10.0.26100 (`dev` preset): `native_cvt_f16_i16_ir`
(`1.09s`: 82 conversions + 38 refusals), `native_cvt_f16_i16_controls` (`3.84s`), all 47 `native_*_(controls|ir)` tests
(`54.43s`), and `tests/docs_links_check.py` (`101` Markdown files, `589` links) passed; `native_cvt_f16_i16_hardware`
skipped (`77`, Quadro M1000M does not support shader model 6.6 with wave32, so GPU readback is not measured). Five
broken-lowering sensitivity checks (`zero_high_half`, `drop_ordered_nan`, `swap_signed`, `drop_clamp`,
`wrong_inline_float`) each failed `native_cvt_f16_i16_ir`.
