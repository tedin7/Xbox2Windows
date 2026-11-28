# Direct DXIL `v_dot2_f32_f16` Translation

Experiment for direct Compute and Pixel DXIL lowering of RDNA 2 VOP3P `v_dot2_f32_f16` (`0x13`), addressing `#3238` and `#2935`.

## What Changed

- Lowered `v_dot2_f32_f16` in `DirectCompute::float_alu()` inside [src/gfx1013_direct_compute.inl](../../src/gfx1013_direct_compute.inl):
  - Enforces the full RDNA 2 `MODE` contract (`((mode_bits ^ 0xf0u) & 0x8000ffu)` and `half_mode_known`, requiring RNE rounding and denormal preservation for both binary16 and binary32, plus `half_binary = true` in `DirectCompute::lower()`).
  - Accepts `op_sel[1:0]` (`i.a >> 11`), `op_sel_hi[1:0]` (`i.b >> 27`), `neg_lo[2:0]` (`i.neg`: pair 0 binary16 signs + `src2` binary32 sign), `neg_hi[1:0]` (`i.abs`: pair 1 binary16 signs), and `clamp` (`dx.op.unary.f32` opcode `7`, saturate). Ignores unused `op_sel_hi[2]` (bit 14).
  - Decodes `src0` and `src1` through `packed_f16_inline_constant()` (binary16 inline float constants `240..248` with zero high half, plus inline integers `128..208`) or `operand32()`, widens each selected half via `dx.op.legacyF16ToF32`, computes `((a0 * b0) + (a1 * b1)) + c`, and preserves binary32 subnormal accumulators `c` (`0 < |c| < 2^-126`) across desktop DXIL `fadd` / `saturate` when `(a0 * b0) + (a1 * b1) == +/-0.0` (every finite nonzero binary16 product sum has magnitude `>= 2^-48 >> 2^-126`).
  - Refuses reserved `op_sel[2]` / `neg_hi[2]` bits (`i.a & 0x2400`), DPP/permuted encodings, special scalar registers on `src0`/`src1`/`src2`, and unsupported or dynamic rounding/denormal modes.
- Extended the Python DXIL IR evaluator in [tests/native_f16_alu_check.py](../../tests/native_f16_alu_check.py) to evaluate `dx.op.unary.f32(i32 7, ...)` (`Saturate`).
- Added compute/pixel test coverage in [tests/native_dot2_f32_f16_check.py](../../tests/native_dot2_f32_f16_check.py) (including `inline_int_src2_129` `0x00000001` plus VGPR-sourced binary32 subnormals `0x00000001` and `0x80400000` with and without `clamp=1`, and an `s_denorm_mode 0xC` binary32 flush refusal) and registered CTest targets in [tests/cmake/native_packed_dot2_f16.cmake](../../tests/cmake/native_packed_dot2_f16.cmake).

## Verification

Host: Windows 10.0.26100, `dev` and `release` (`/O2`) presets, DXC `10.0.26100.0`.

- `cmake --build --preset dev` & `--preset release` (`compile_dxil`, `native_compute_ir_dump`, `native_store_replay`, `offscreen_check`): passed.
- `ctest --preset dev --no-tests=error --output-on-failure -j 1 -V -R '^native_(packed_dot2(_f16)?_.*|packed_f16_inline_(ir|controls|compute_hardware)|f16_alu_(ir|hardware))$'`:
  - `native_packed_dot2_f16_ir`: **passed** (`30` cases matched exact binary32 model; `10` refusals matched).
  - `native_packed_dot2_f16_controls`: **passed** (`30` cases compiled to DXIL; verified `10` refusals).
  - `native_packed_dot2_f16_pixel_warp`: **passed** (`Microsoft Basic Render Driver` `10.0.26100.9711`, `7` quad draws matched, including `inline_int_src2_129` and all 4 VGPR subnormal cases).
  - `native_packed_dot2_f16_pixel_hardware`: **passed** (`NVIDIA Quadro M1000M` `32.0.15.7371`, `7` quad draws matched; debug layer unavailable).
  - `native_packed_dot2_f16_compute_hardware`: **skipped** (`77`, `Intel(R) HD Graphics 530` `31.0.101.2111` requires SM 6.6 with wave32).
  - Neighbor `native_packed_dot2_controls`, `native_packed_f16_inline_ir/controls`, and `native_f16_alu_ir` passed (`*_hardware` compute tests skipped `77` on HD 530).
- `ctest --preset release --no-tests=error --output-on-failure -j 1 -V -R '^native_(packed_dot2(_f16)?_.*|packed_f16_inline_(ir|controls|compute_hardware)|f16_alu_(ir|hardware))$'`:
  - Same `8` IR/controls/pixel tests **passed** and `4` SM 6.6 wave32 compute tests skipped (`77`) on HD 530.
- `ctest --preset dev --no-tests=error -R '^native_.*_(controls|ir)$' --output-on-failure`: **69/69 passed** (`0` failures, `0` skips).
- Sensitivity mutants in `src/gfx1013_direct_compute.inl`:
  - `drop_subnormal_keep` (`keep_subnormal = "0"`): failed `native_packed_dot2_f16_pixel_warp` (`rc=8`, `00000001,00000001,80400000,00000001` vs `00000000,00000000,00000000,00000000`).
  - `drop_hi_term`, `drop_neg_c`, `drop_clamp`, and `swap_opsel_hi`: failed `native_packed_dot2_f16_ir` (`rc=8`).
- `python -B tests/docs_links_check.py`: passed (`131` Markdown files, `631` links).

## Limits

- Direct Compute GPU execution (`native_packed_dot2_f16_compute_hardware`) could not run locally because `Intel(R) HD Graphics 530` lacks SM 6.6 wave32 (`77` skip). Compute IR evaluation, DXIL `controls`, and Pixel WARP + NVIDIA hardware draws (including binary32 subnormal retention on inline and VGPR `src2`) are verified.
- `signed_zero_keep`, `signed_zero_cancel_pos`, `inf_propagation`, `nan_propagation`, and `clamp_nan_to_zero` (`pixel_safe=False`) are verified in IR/controls/compute fixtures because `--startup-pixel-words` checks bitwise equality where drivers can vary on qNaN payload bits or pixel export `-0.0`.
