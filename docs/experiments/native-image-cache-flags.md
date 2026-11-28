# Direct DXIL Read-Only Image Cache Flags (`GLC` / `DLC`)

Experiment for accepting MIMG cache-policy flags `GLC` (word 0 bit 13, `0x2000`) and `DLC` (word 0 bit 7, `0x0080`) on read-only image operations in `DirectCompute::image_io()`, addressing `#3242`, `#3224` (Rank 8), and `#2935`.

## What Changed

- Updated `DirectCompute::image_io()` in [src/gfx1013_direct_compute.inl](../../src/gfx1013_direct_compute.inl) so `(i.a & (store ? 0x3e0c0u : 0x3c040u))` accepts `GLC` (`0x2000`) and `DLC` (`0x0080`) on read-only image instructions (`image_load`, `image_load_mip`, `image_sample*`, `image_gather4*`, `image_getlod`, `image_getresinfo`) while keeping refusals for:
  - `image_store` with `GLC` or `DLC` (`0x3e0c0u`)
  - `LWE` (`0x20000`), `TFE` (`0x10000`), `R128` (`0x8000`), and reserved word 0 bits (`0x4040`)
  - word 1 reserved bits (`0x3c000000u`)
- Added [tests/native_image_cache_flags_check.py](../../tests/native_image_cache_flags_check.py) and registered CTest entries in [tests/cmake/native_image_cache_flags.cmake](../../tests/cmake/native_image_cache_flags.cmake).

## Verification

Host: Windows 10.0.26100, `dev` preset, DXC `10.0.26100.0`.

- `cmake --build --preset dev --target compile_dxil offscreen_check native_compute_ir_dump`: passed.
- `ctest --preset dev --no-tests=error -R '^(native_image_cache_flags_|native_gather_|native_image_atomic_|native_sample_ssamp_)' -V`:
  - `native_image_cache_flags_controls`: **passed** (`48` flagged DXIL modules matched plain `flags=0` baselines across Compute and Pixel stages and `GLC`/`DLC`/`SLC`/`GLC|DLC|SLC`; `16` refusals matched across Compute and Pixel).
  - `native_image_cache_flags_ps_warp`: **passed** (`12` cached-flag `image_gather4_l` cases matched on Microsoft Basic Render Driver `10.0.26100.9711`).
  - `native_image_cache_flags_ps_hardware`: **passed** (`12` cached-flag `image_gather4_l` cases matched on `NVIDIA Quadro M1000M` `32.0.15.7371`; debug layer unavailable).
  - `native_image_cache_flags_cs_hardware`: **passed** (`12` cached-flag `image_gather4_l` cases matched on `NVIDIA Quadro M1000M` `32.0.15.7371`; debug layer unavailable).
  - `native_image_cache_flags_cs_warp`: **skipped** (`77`, Microsoft Basic Render Driver has 4-lane waves and lacks SM 6.6 wave32).
  - Neighbor suites (`native_gather_compare_warp`, `native_gather_lod_controls/ps_warp/ps_hardware/cs_hardware`, `native_image_atomic_controls/hardware`, `native_sample_ssamp_hardware`): passed (with SM 6.6 wave32 WARP tests skipping `77`; `native_image_atomic_runtime_hardware` failed on this dual-GPU host because `startup_stream_consumer` runs on adapter 0 `Intel(R) HD Graphics 530` which lacks SM 6.6 wave32).
- `ctest --preset dev --no-tests=error -R '^native_.*_(controls|ir)$' --output-on-failure`: **67/67 passed** (`0` failures, `0` skips).
- Sensitivity mutants in `src/gfx1013_direct_compute.inl` tested against `native_image_cache_flags_controls`:
  - `revert_ro_cache_flags` (`(i.a & 0x3e0c0u)`): failed (`rc=8`, rejected `image_load` with `GLC`).
  - `drop_r128_refusal` (`(i.a & (store ? 0x3e0c0u : 0x34040u))`): failed (`rc=8`, accepted `sample_r128`).
  - Clean tree restored, rebuilt, and `native_image_cache_flags_controls` passed (`48` modules, `16` refusals).
- `python -B tests/docs_links_check.py`: passed (`131` Markdown files, `631` links).

## Limits

- `native_image_cache_flags_cs_warp` skips (`77`) on WARP because WARP uses 4-lane waves.
- `GLC` and `DLC` remain refused on `image_store*` and explicit/policy-checked on `image_atomic_*` (`image_atomic()`).
- Local `offscreen_check` hardware runs reported `debug_layer=unavailable` on `NVIDIA Quadro M1000M`.
