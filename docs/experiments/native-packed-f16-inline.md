# Inline constants in direct packed binary16 arithmetic and min/max (#3226)

`src/gfx1013_direct_compute.inl` now accepts inline constant sources in both `DirectCompute::packed_f16_alu()`
(`v_pk_add_f16`, `v_pk_mul_f16`, `v_pk_fma_f16`) and `DirectCompute::packed_minmax()` (`v_pk_min_f16`, `v_pk_max_f16`,
`v_pk_min_{i,u}16`, `v_pk_max_{i,u}16`):

- **Shared `packed_f16_inline_constant()` contract:**
  - Inline binary16 float constants (`240..248`: `0.5`, `-0.5`, `1.0`, `-1.0`, `2.0`, `-2.0`, `4.0`, `-4.0`, `1/(2*pi)`)
    decode to the low 16 bits (`0x3800`, `0xb800`, `0x3c00`, `0xbc00`, `0x4000`, `0xc000`, `0x4400`, `0xc400`, `0x3118`)
    with bits `[31:16]` zeroed, matching `gfx1013` hardware (`op_sel_hi = 1` reads `+0.0`).
  - Inline integer constants (`128..208`: `0..64` and `-1..-16`) decode to full 32-bit sign-extended values (`0..64`,
    `0xFFFFFFFF..0xFFFFFFF0`).
  - Integer `v_pk_min_{i,u}16` / `v_pk_max_{i,u}16` accept `128..208` and continue refusing float `240..248`.
- **Kept refusals:** `vcc_lo`/`vcc_hi`, `m0`, `exec_lo`/`exec_hi`, `scc`, `clamp` on packed f16 arithmetic, reserved
  VOP3P bits, and non-default FP16 MODE writes keep their existing named diagnostics.

## Fixture

[`tests/native_packed_f16_inline_check.py`](../../tests/native_packed_f16_inline_check.py) via
[`tests/cmake/native_packed_f16_inline.cmake`](../../tests/cmake/native_packed_f16_inline.cmake)
(`native_packed_f16_inline_ir`, `native_packed_f16_inline_controls`, `native_packed_f16_inline_compute_hardware`,
`native_packed_f16_inline_pixel_warp`, `native_packed_f16_inline_pixel_hardware`):

- Covers 24 cases: all 9 inline f16 constants `240..248`, `op_sel_hi` high-half zeroing (`+0.0`), `neg_lo`/`neg_hi`
  (including `-0.0`), single-rounding FMA tie witness with inline `0.5` (`240`) + `2^-24` (`129`), inline integer
  constants (`128..208`), `v_pk_min_f16`/`v_pk_max_f16` (including `193` NaN selection), and `v_pk_min_u16`/`v_pk_max_i16`,
  plus 10 refusals.

Measured on 28.11.2026, `main` `c7215add` plus this change, Windows 10.0.26100 (`dev` preset):
- `native_packed_f16_inline_ir` (24 cases + 10 refusals, `0.68s`), `native_packed_f16_inline_controls` (24 compiled + 10
  refusals, `1.47s`), `native_packed_f16_inline_pixel_warp` (12 paired draws / 24 cases, `2.18s`), and
  `native_packed_f16_inline_pixel_hardware` (`Intel(R) HD Graphics 530`, 12 paired draws / 24 cases, `5.20s`) passed.
- `native_packed_f16_inline_compute_hardware` skipped (`77`, `Intel(R) HD Graphics 530` `31.0.101.2111`: requires shader
  model 6.6 with wave32, so GPU compute readback is not measured).
- `^native_.*_(controls|ir)$`: 65/65 passed (`80.10s`).
- Four broken-lowering sensitivity checks (`replicate_high_half`, `use_f32_inline`, `zero_extend_neg_int`,
  `drop_minmax_float_inline`) each failed `native_packed_f16_inline_ir` (exit `1`).
