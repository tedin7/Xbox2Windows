# Direct DXIL `v_madmk_f32` and `v_madak_f32` Translation

Experiment for direct Compute and Pixel DXIL lowering of RDNA 2 unfused literal multiply-adds `v_madmk_f32` (`VOP2 0x20`: `S0 * K + S1`) and `v_madak_f32` (`VOP2 0x21`: `S0 * S1 + K`), addressing `#3249`, `#2935`, and `#3005`.

## What Changed

- Extended `unfused_mad` inside `DirectCompute::float_alu()` in [src/gfx1013_direct_compute.inl](../../src/gfx1013_direct_compute.inl) to cover `v_madmk_f32` and `v_madak_f32`:
  - Uses the existing `#3005` unfused sequence: flush denormal inputs to signed zero (`ftz`), round `fmul` separately without fast-math/FMA contraction, flush the product, add the flushed addend (`S1` for `v_madmk_f32`, trailing 32-bit literal `K` for `v_madak_f32`), and flush the sum.
  - Refuses invalid non-VOP2 forms (`i.wide || i.subword || i.dpp_control || permuted`).
- Added [tests/native_mad32_k_check.py](../../tests/native_mad32_k_check.py) and registered CTest entries in [tests/cmake/native_mad32_k.cmake](../../tests/cmake/native_mad32_k.cmake).

## Verification

Host: Windows 10.0.26100, `dev` preset, DXC `10.0.26100.0`.

- `cmake --build --preset dev --target compile_dxil native_compute_ir_dump native_store_replay offscreen_check`: passed.
- `ctest --preset dev --no-tests=error -R '^(native_mad32_k_|native_mad32_)' -V`:
  - `native_mad32_k_ir`: **passed** (`14` cases matched unfused FTZ model; `4` refusals matched).
  - `native_mad32_k_controls`: **passed** (`14` cases compiled with separate `fmul`/`fadd` and no `tertiary.f32(i32 46,)` or fast-math flags; `4` refusals matched).
  - `native_mad32_k_pixel_warp`: **passed** (`Microsoft Basic Render Driver`, `3` quad draws matched).
  - `native_mad32_k_pixel_hardware`: **passed** (`NVIDIA Quadro M1000M`, `3` quad draws matched; debug layer unavailable).
  - `native_mad32_k_compute_hardware`: **skipped** (`77`, `Intel(R) HD Graphics 530` `31.0.101.2111` requires SM 6.6 with wave32).
  - Neighbor `native_mad32_controls` passed (`native_mad32_hardware` skipped `77` on HD 530).
- `ctest --preset dev --no-tests=error -R '^native_.*_(controls|ir)$' --output-on-failure`: **68/68 passed** (`0` failures, `0` skips).
- Sensitivity mutants in `src/gfx1013_direct_compute.inl` tested against `native_mad32_k_ir`:
  - `swap_madmk_madak` (`v_madak_f32` vs `v_madmk_f32` literal role): failed (`rc=8`).
  - `drop_ftz_input` (`auto x = a;`): failed (`rc=8`, `madmk_ftz_src0`).
  - `drop_vop3_refusal` (bypass `literal_k` non-VOP2 check): failed (`rc=8`, `unsupported v_madmk_f32 encoding`).
  - Clean tree restored, rebuilt `native_compute_ir_dump`, and `native_mad32_k_ir` passed again.
- `python -B tests/docs_links_check.py`: passed (`131` Markdown files, `631` links).

## Limits

- Direct Compute GPU execution (`native_mad32_k_compute_hardware`) could not run locally because `Intel(R) HD Graphics 530` lacks SM 6.6 wave32 (`77` skip). Compute IR, DXIL `controls`, and Pixel WARP + NVIDIA hardware draws are verified.
- `madak_ftz_sum`, `madmk_zero_inf_nan`, and `madak_neg_zero` (`pixel_safe=False`) are verified in IR/controls/compute fixtures rather than `--startup-pixel-words` because WARP/GPU drivers can differ on raw `-0.0`/qNaN payload patterns on pixel export.
