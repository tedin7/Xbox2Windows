# Direct DXIL `v_rcp_iflag_f32` Translation

Experiment for direct Compute and Pixel DXIL lowering of `v_rcp_iflag_f32` (`VOP1 0x2b`, `VOP3 0x1ab`, SDWA, DPP16), addressing `#3252` and `#2935`.

## What Changed

- [src/gfx1013_compute_alu.inl](../../src/gfx1013_compute_alu.inl): added `v_rcp_iflag_f32` alongside `v_rcp_f32` (`1.0 / {0}`, also enabling `alu_rules` lookup in `DirectCompute::dpp()`).
- [src/gfx1013_compute_opcodes.inl](../../src/gfx1013_compute_opcodes.inl): added `case 0x2b: return "v_rcp_iflag_f32";`.
- [src/gfx1013_compute_native.cpp](../../src/gfx1013_compute_native.cpp): added `"v_rcp_iflag_f32"` to the SDWA `floats` set next to `"v_rcp_f32"`.
- [src/gfx1013_direct_compute.inl](../../src/gfx1013_direct_compute.inl): lowered `v_rcp_iflag_f32` identically to `v_rcp_f32` (`fdiv float 1.000000e+00, a`) in `DirectCompute::float_alu()`.
- Added [tests/native_rcp_iflag_check.py](../../tests/native_rcp_iflag_check.py) and registered CTest entries in [tests/cmake/native_rcp_iflag.cmake](../../tests/cmake/native_rcp_iflag.cmake).

## Verification

Host: Windows 10.0.26100, `dev` preset, DXC `10.0.26100.0`.

- `cmake --build --preset dev --target compile_dxil native_store_replay offscreen_check native_compute_ir_dump`: passed.
- `ctest --preset dev --no-tests=error -R '^native_rcp_iflag_' -V`:
  - `native_rcp_iflag_controls`: **passed** (`16` Compute + `4` Pixel cases compiled identically to `v_rcp_f32` DXIL; `2` refusals matched).
  - `native_rcp_iflag_pixel_warp`: **passed** (`Microsoft Basic Render Driver`, `4` quad draws matched).
  - `native_rcp_iflag_pixel_hardware`: **passed** (`NVIDIA Quadro M1000M`, `4` quad draws matched; debug layer unavailable).
  - `native_rcp_iflag_compute_hardware`: **skipped** (`77`, `Intel(R) HD Graphics 530` `31.0.101.2111` requires SM 6.6 with wave32).
- `ctest --preset dev --no-tests=error -R '^native_.*_(controls|ir)$' --output-on-failure`: **68/68 passed** (`0` failures, `0` skips).
- Sensitivity mutants verified against `native_rcp_iflag_controls`:
  - `drop_direct_iflag` (`src/gfx1013_direct_compute.inl`): failed (`rc=8`, `direct DXIL lowering for v_rcp_iflag_f32`).
  - `drop_sdwa_iflag` (`src/gfx1013_compute_native.cpp`): failed (`rc=8`, `unimplemented direct SDWA operand profile`).
  - Clean tree restored, rebuilt `compile_dxil`, and `native_rcp_iflag_controls` passed again.
- `python -B tests/docs_links_check.py`: passed (`131` Markdown files, `631` links).

## Limits

- Direct Compute GPU execution (`native_rcp_iflag_compute_hardware`) could not run locally because `Intel(R) HD Graphics 530` lacks SM 6.6 wave32 (`77` skip). Compute DXIL (`controls`), Pixel WARP, and Pixel hardware (`NVIDIA Quadro M1000M`) are verified on power-of-two operands (`0 ULP`); non-power-of-two reciprocal ULP tolerance matches DXIL `fdiv` hardware behavior.
