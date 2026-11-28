# Normalized integer conversion from binary16 (`v_cvt_norm_i16_f16` / `v_cvt_norm_u16_f16`) in direct DXIL

The native frontend lowers gfx1013 VOP1 (`0x63` / `0x64`) and VOP3 (`0x1e3` /
`0x1e4`) `v_cvt_norm_i16_f16` and `v_cvt_norm_u16_f16` in
`DirectCompute::float_alu()` for both Compute and Pixel shaders.

## Exact integer lowering and semantics

The lowering matches [pixel_aot.py](../../tools/pixel_aot.py)'s
`f16_cvt_norm_result` contract in exact 32-bit integer arithmetic instead of
widening to binary32:

- Decodes the binary16 exponent (`mag >> 10`) and significand (normal `0x400 |
  (mag & 0x3ff)` or subnormal `mag`), multiplies by `32767` (`i16`) or `65535`
  (`u16`), and rounds once to nearest even with exact remainder/midpoint checks.
  This avoids binary32 double-rounding mismatches (such as `0x3801` and
  `0xb802`) and preserves binary16 subnormal inputs.
- `+0.0` and `-0.0` map to `0`.
- Inputs with magnitude `>= 1.0` (including `±Inf`) saturate to `+32767` /
  `-32767` (`i16`) or `65535` / `0` (`u16`).
- Any negative value (`sign != 0`) maps to `0` on `v_cvt_norm_u16_f16`.
- Non-finite NaN operands (`mag > 0x7c00`) map to `0`.
- Supported VOP3 modifiers are `abs[0]`, `neg[0]`, `op_sel[0]` (VGPR high-half
  source read), and `op_sel[3]` (high-half destination write), preserving the
  unwritten destination half and respecting `EXEC` masking.
- Supported sources are VGPRs (`256..511`), numeric SGPRs (`0..105`), 32-bit
  literals (`255`), inline binary16 float constants (`240..248`), and inline
  integer words (`128..208`).

## Refused encodings and mode gate

The translator refuses `v_cvt_norm_i16_f16` and `v_cvt_norm_u16_f16` when any
of the following hold:

- VOP3 `clamp != 0` or `omod != 0`.
- Unused VOP3 operand modifiers (`abs[2:1]`, `neg[2:1]`, `op_sel[2:1]`,
  non-zero `src1` / `src2`).
- High-half `op_sel[0]` on a non-VGPR source (`src0 < 256`).
- DPP16, SDWA, permuted/subword encodings, or special scalar sources such as
  `vcc_lo` (`106`).
- Non-default binary16 rounding/denormal mode
  (`((mode_bits ^ 0xf0u) & 0x8000ccu) != 0`) or dynamic/unknown `MODE` writes.

## Fixture coverage and reproduction

[native_cvt_norm_f16_check.py](../../tests/native_cvt_norm_f16_check.py) checks
30 Compute cases (`<= 32` words/lane in `native_store_replay`), 11 refusal
controls, and lossless 16-bit Pixel readback across 15 draws (29 full-`EXEC`
cases padded to 30) on WARP and hardware:

```powershell
cmake --build --preset dev --target compile_dxil native_compute_ir_dump native_store_replay offscreen_check
ctest --test-dir out/build/dev --no-tests=error --output-on-failure -j 1 -R '^native_cvt_norm_f16_' -V
```

Compute replay requires shader model 6.6 with wave32; adapters without wave32
report a skip (`77`) on `native_cvt_norm_f16_compute_hardware`.
