# Scalar binary16 FMA in the direct C++ compute path (#3186)

`src/gfx1013_direct_compute.inl` now lowers `v_fma_f16` (VOP3), `v_fmac_f16` (VOP2/VOP3A), `v_fmamk_f16` (VOP2), and
`v_fmaak_f16` (VOP2) in the direct C++ compute path (`DirectCompute::float_alu()`):

- **Exact single-rounded FMA (`fused_half_rne()`):** Reuses the shared 32-bit integer `fused_half_rne(a, b, c)` helper
  (introduced in #3151 for `v_pk_fma_f16`) so `a * b + c` computes an exact product, aligns the addend with a sticky
  bit, and rounds once to nearest-even binary16 rather than double-rounding through binary32.
- **High-half preservation by default:** The default forms read `S0..S2[15:0]` / `K[15:0]` / `D[15:0]`,
  writes the result into `D[15:0]`, and keeps `D[31:16]` unchanged.
- **Modifiers & MODE contract:** Supports VOP3 `abs`, `neg`, and `clamp` (`[0.0, 1.0]` saturation); refuses DPP, SDWA,
  `omod`, high-half `op_sel` and third-source modifiers/`src2` on `v_fmac_f16_e64`, and non-default FP16 MODE writes
  (`half_binary` in `DirectCompute::lower()`).
- **Test adjustment:** Updated the unlowered-instruction refusal placeholder in
  [`tests/native_f16_alu_check.py`](../../tests/native_f16_alu_check.py) from `v_fmac_f16` to `v_pk_fmac_f16`, and added
  `@dx.op.unaryBits.i32(i32 33, ...)` (`FirstbitHi`) evaluation to its IR interpreter so `native_fma_f16_ir` evaluates
  `fused_half_rne()` directly.

Direct compute also accepts VOP3 `v_fma_f16` with `op_sel`. Bits 0 to 2 select the high half of the respective VGPR source before abs and neg. Bit 3 selects the high destination half and preserves its low half. A clear destination selector writes the low half and preserves the high half. Selected non-VGPR sources and clamp combined with `op_sel` are refused. The [half-selection fixture](../../tests/native_fma_f16_opsel_check.py) checks six captured forms in IR and direct GPU replay.

## Fixture

[`tests/native_fma_f16_check.py`](../../tests/native_fma_f16_check.py) via
[`tests/cmake/native_fma_f16.cmake`](../../tests/cmake/native_fma_f16.cmake) (`native_fma_f16_ir`,
`native_fma_f16_controls`, `native_fma_f16_hardware`):

- **IR (`native_fma_f16_ir`):** Evaluates lane 0 across 41 cases (all 4 `#2073` single-rounding vs double-rounding tie
  witnesses across all 4 instructions, SGPR/literal/inline-float/inline-int sources, `dst = src0` overlap, VOP3
  `abs`/`neg`/`clamp`, signed zeros, exact cancellation, subnormals, `diff_e >= 30` sticky-bit midpoint, underflow
  ties, overflow, `Inf`/`NaN`) against an exact `fractions.Fraction` reference, plus 12 refusals.
- **Controls (`native_fma_f16_controls`):** Compiles the full 41-case shader to DXIL via `compile_dxil` and verifies all
  12 refusals.
- **Hardware (`native_fma_f16_hardware`):** Replays all 41 cases via `native_store_replay --direct`.

Measured on 27.11.2026, `main` `c45bf9ab` plus this change, Windows 10.0.26100 (`dev` preset): `native_fma_f16_ir`
(41 cases + 12 refusals), `native_fma_f16_controls`, `native_f16_alu_ir`, `native_packed_f16_controls`,
`native_packed_f16_pixel_warp`, `native_packed_f16_pixel_hardware`, all 57 `^native_.*_(controls|ir)$` tests, and
`tests/docs_links_check.py` passed; `native_fma_f16_hardware` skipped (`77`,
`Intel(R) HD Graphics 530` `31.0.101.2111`: requires shader model 6.6 with wave32, so GPU compute readback is
not measured). Five broken-lowering sensitivity checks (`zero_high_half`, `drop_sticky`, `swap_fmamk_fmaak`,
`drop_clamp`, `drop_inf_cancel_nan`) each failed `native_fma_f16_ir` (exit `1`).
