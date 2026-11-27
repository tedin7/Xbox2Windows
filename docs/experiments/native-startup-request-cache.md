# Startup request consumption and runtime cache publication in `compile_dxil` (#2933)

[The fixture](../../tests/native_startup_request_cache_check.py) verifies that `compile_dxil --requests` consumes
revision-29 startup graphics (`x2w-startup-shader-29`, including ASCII `req_<key>.psin` pixel-input descriptions)
and compute (`x2w-startup-shader-29cs`, `req_<key>.c67`) requests, publishes runtime-named stage binaries and
manifests (`<key>.vs.cso` / `<key>.ps.cso` or `<key>.cs.cso`, followed by `<key>.txt`), and loads directly through
the startup runtime cache loaders on WARP and hardware without manual renaming or conversion.

## Alignment between `compile_dxil --requests` and the startup cache

1. **Request keys and `.psin` stream descriptions (`src/shader_compute_request.cpp`):**
   - Graphics requests validate `x2w-startup-shader-29` or the archived revision 28 over `.a`, `.b`, `.root`, and optional `.ia`. When
     `req_<key>.psin` is present, the reader parses `psin1 <key> ena=0x<mask> addr=0x<mask>`, checks that the
     embedded key matches `<key>` and `(ena & ~addr) == 0`, and folds `b"psin1" + struct.pack("<II", ena, addr)`
     into the FNV-1a key identically to `tools/startup/gfx2hlsl.py:pipeline_key`.
   - Compute requests validate `x2w-startup-shader-29cs` or the archived revision 28 over length-framed `.c67`, `.data`, `.root`, `.desc`,
     `.sa`, and `.sb` identically to `d3d12_draw.inl:publish_compute_request` and `gfx2hlsl.py:compute_key`.
   - In raw graphics translation (`src/gfx1013_compute_native.cpp`, `src/gfx1013_direct_graphics.inl`), pixel
     entry VGPR offsets follow `SPI_PS_INPUT_ADDR` while only inputs enabled in `SPI_PS_INPUT_ENA` are readable.
2. **Stage and manifest publication (`src/direct_compute_cache.cpp`, `tools/startup/compile_dxil.cpp`):**
   - In `--requests` batch mode, `compile_dxil` strips the `req_` prefix from output stems (`<key>.vs.cso` /
     `<key>.ps.cso`, `<key>.gs.cso` for `gfx1013-container-rect`, and `<key>.cs.cso` for compute), withdraws any
     prior `<key>.txt` before rewriting stages or on refusal, records `<key>.psdep` when `.psin` was used, and
     atomically publishes `<key>.txt` after all stage `.cso` files are in place.

## Verification and negative controls

1. **`controls` mode (`native_startup_request_cache_controls`):**
   - Compiles two `.psin`-described VS/PS requests (`(0x8002, 0x8003)` placing `POS_FIXED_PT` in `v4` after an
     allocated-only `PERSP_SAMPLE` pair, and `(0x8002, 0x8002)` placing `POS_FIXED_PT` in `v2`) and a `.c67`
     compute request in place, checks that `<key>.txt` is written after stage `.cso` files and recognised as
     complete by `gfx2hlsl.pending_keys`, and verifies that mutating a stage blob behind an existing `.psin`
     withdraws `<key>.txt`.
   - Exercises eight fresh-directory refusal controls (graphics key mismatch, compute key mismatch, `.psin`
     naming another pipeline, missing `addr`, missing `ena`, contradictory `ena`/`addr`, malformed `.psin`, and
     reading an allocated-but-disabled pixel input VGPR), requiring a clear diagnostic on `stderr` and no
     `<key>.txt` or stage `.cso` output.
2. **`e2e` mode (`native_startup_request_cache_warp`, `native_startup_request_cache_hardware`):**
   - Draws both `.psin` pairs through `offscreen_check --startup-packed-position` (all 91 pixels match, while a
     mismatched-mask control fails readback).
   - Drives a runtime-published graphics pair and a runtime-published slot-67 `.c67` compute request through
     `startup_stream_consumer` -> `compile_dxil --requests` -> `d3d12_xs` (`load_translation` / `draw_translated`
     and `load_compute_translation` / `record_translated_dispatch`) with the D3D12 debug layer enabled (`0`
     errors), checking all 32 RGBA pixels and all 320 compute output dwords.

```powershell
cmake --preset release
cmake --build --preset release --target compile_dxil startup_stream_consumer offscreen_check
ctest --preset release --no-tests=error -V -R "^native_startup_request_cache_(controls|warp|hardware)$"
```

## Limits

- Startup runtime manifests are emitted only for stage bindings that the existing `d3d12_xs` manifest loaders
  bind directly (for example, pairs without IA `InputLayout` vertex elements and compute shaders using raw table
  buffers, 2D textures, 2D/2D-array float image UAVs, static non-comparison samplers, and root constants).
- Direct compute shaders carry `[WaveSize(32)]`; on Wave-4 WARP (`10.0.26100`), `load_compute_translation`
  verifies the manifest, loads `<key>.cs.cso`, and creates the desktop root signature before `CreateComputePipelineState`
  declines `[WaveSize(32)]`, while full GPU dispatch and 320-word readback run on Wave-32 hardware (`AMD Radeon RX 9070 XT`).
