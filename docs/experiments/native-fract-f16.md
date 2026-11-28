# Binary16 fractional part (`v_fract_f16`) in direct DXIL

The native frontend lowers gfx1013 VOP1 (`0x5f`) and VOP3 (`0x1df`)
`v_fract_f16` in `DirectCompute::float_alu()` for both Compute and Pixel
shaders.

## Semantics and supported forms

For each active lane:

- Finite binary16 inputs (decoded with `legacyF16ToF32`) compute
  `x - floor(x)` in binary32, round to binary16 round-to-nearest-even via
  `packed_half_rne()`, and clamp a result of `1.0` (`0x3c00`, produced by tiny
  negative fractions such as `-2^-12` or `-2^-24`) strictly below `1.0` to
  `0x3bff`, matching `min(x - floor(x), 0x3bff)`. Exact integers and `-0.0`
  produce `+0.0` (`0x0000`), and positive binary16 subnormals are preserved.
- Non-finite inputs (`+Inf`, `-Inf`, and NaN) produce canonical quiet NaN
  (`0x7e00`), or `+0.0` (`0x0000`) when `clamp=1`. Because finite outputs already
  lie in `[+0.0, 1.0)`, `clamp=1` preserves finite results unchanged.
- Supported VOP3 modifiers are `abs[0]`, `neg[0]`, `clamp`, `op_sel[0]`
  (VGPR high-half source read), and `op_sel[3]` (high-half destination write),
  preserving the unwritten half of the destination VGPR under `EXEC` masking.
- Supported sources are VGPRs (`256..511`), numeric SGPRs (`0..105`), 32-bit
  literals (`255`), inline binary16 float constants (`240..248`), and inline
  integer words (`128..208`).

## Refused encodings and mode gate

The translator refuses `v_fract_f16` when any of the following hold:

- Non-zero VOP3 output modifier (`omod != 0`).
- Unused VOP3 operand modifiers (`abs[2:1]`, `neg[2:1]`), unused `op_sel[2:1]`,
  or non-zero `src1` / `src2` fields.
- High-half `op_sel[0]` on a non-VGPR source (`src0 < 256`).
- DPP16, SDWA, permuted/subword encodings, or special scalar sources such as
  `vcc_lo` (`106`).
- Non-default binary16 rounding/denormal mode (`((mode_bits ^ 0xf0u) & 0x8000ccu) != 0`),
  or an unknown/dynamic `MODE` state (for example a conditional `s_denorm_mode`
  write).

## Fixture coverage and reproduction

[native_fract_f16_check.py](../../tests/native_fract_f16_check.py) checks
32 Compute cases (`<= 32` words/lane in the `native_store_replay` buffer),
10 refusal controls, and lossless 16-bit Pixel readback across 16 draws (31
full-`EXEC` cases, padded to 32) on both WARP and D3D12 hardware:

```powershell
cmake --build --preset dev --target compile_dxil native_compute_ir_dump native_store_replay offscreen_check
ctest --test-dir out/build/dev --no-tests=error --output-on-failure -j 1 -R '^native_fract_f16_' -V
```

Compute replay requires shader model 6.6 with wave32; adapters without wave32
report a skip (`77`) on `native_fract_f16_compute_hardware`.
