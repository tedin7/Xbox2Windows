# Comparison samplers in a published compute manifest (#3075)

The stand-in reads the published translation of a slot 67 compute request from `<key>.txt`
(`load_compute_translation`, `src/startup_stubs/d3d12_compute_dispatch.inl`). It refused every `smpc` line there
with `a comparison sampler`. It now reads `smpc <i> <w0> <w1> <w2> <w3>` under the same index and whole-line
checks as `smp` and `smpl`, through `TranslatedSamplers::read_manifest_line`. `decode_statics` already turns the
line into a comparison static sampler for the draw path, and the compute root signature takes it the same way.
An `smpheap` line is still refused with `a sampler from the sampler heap`.

## What this covers

- Only `tools/startup/gfx2hlsl.py` writes `smpc` for a compute program today (`image_sample_c_lz` and
  `image_sample_c` with a constant comparison sampler). Before this change the runtime refused that translation
  when it loaded the manifest, so such a request could not be dispatched.
- The C++ compiler's publisher is unchanged. `format_compute_runtime_manifest` (`src/direct_compute_cache.cpp`)
  returns no manifest for a comparison sampler, so `compile_dxil` publishes no compute `smpc` line.
- The in-process route (`X2W_NATIVE_SHADERS`, `translate_compute_natively`) is unchanged. It reads no manifest
  lines and refuses every sampler, and every view that is not a raw buffer, before it reaches one.

## Fixture

[`tests/startup_compute_dispatch_texture_check.py`](../../tests/startup_compute_dispatch_texture_check.py) gains
a compared case. It is case A of
[`startup_gfx2hlsl_compute_check.py`](../../tests/startup_gfx2hlsl_compute_check.py) (group size 2x3x4, 3x2x2
groups, one word stored per thread) with these instructions before the store:

```
s_mov_b64 s[20:21], 0
s_mov_b64 s[22:23], 0
s_mov_b32 s20, 0x8001012
s_load_dwordx8 s[24:31], s[10:11], 0x40
v_cvt_f32_u32_e32 v7, v2
v_cvt_f32_u32_e32 v8, v0
v_mov_b32_e32 v9, 0
image_sample_c_lz v12, v[7:9], s[24:31], s[20:23] dmask:0x1 dim:SQ_RSRC_IMG_2D
v_xor_b32_e32 v5, v5, v12
```

The five instructions that are new here were not assembled by llvm-mc. The check asserts that the Python decoder
reads them as written, that the HLSL holds
`v12 = U(r1.SampleCmpLevelZero(q0, float2(F(v8), F(v9)), F(v7)));` and that the manifest ends with
`smpc 0 0x8001012 0x0 0x0 0x0` (clamp in U and V, TRUNC_COORD, point filters, LESS).

The consumer's new mode `dispatch-sampled-depth` binds a 2x1 `R32_FLOAT` texture with depths 1.0 (left) and 2.5
(right). The reference is the thread's z id (0.0 to 3.0). u is its x id: 0 reads the left texel and 1 clamps to
the right one. The CPU model XORs the bits of 1.0f into a word where `reference < texel` and leaves the word as
case A wrote it otherwise. That is 144 of the 288 written words. The 32 words past them keep the sentinel.

1. Before the translation is published, the Dispatch is refused and every word keeps the sentinel.
2. After `gfx2hlsl.py --pending`, all 320 words match the model and the debug layer reports no error.
3. Control: the same program with the manifest's word 0 changed to GREATER (`0x8004012`) matches the model of
   `reference > texel` (108 words pass). The two models differ at 252 words, so the function bound is the
   manifest's.

The case replaces an older control in the same file, which rewrote an `smpl` line as `smpc` and expected the
refusal.

## Command and result

```powershell
.\tools\dev.ps1
cmake --build --preset dev --target startup_stream_consumer
ctest --preset dev --no-tests=error -R '^startup_compute_dispatch_texture_(warp|hardware)$' -V
```

Measured on 27.11.2026: `main` at `e1ea515d` plus this change, Windows 11 build 26300.9974, Windows SDK
10.0.26100, DXC 1.9.2602.17, Debug build, D3D12 debug layer enabled.

| Test | Adapter, driver | Result |
|---|---|---|
| `startup_compute_dispatch_texture_hardware` | NVIDIA GeForce RTX 4090, 32.0.16.1088 | PASS (18.81 s). The compared case matched all 320 words under LESS and under GREATER, with no debug layer error. |
| `startup_compute_dispatch_texture_warp` | Microsoft Basic Render Driver, 10.0.26100.9818 | PASS (7.72 s), with the same lines. |

There was no capability skip.

Negative control: with `src/startup_stubs/d3d12_compute_dispatch.inl` put back to `e1ea515d` and the same target
rebuilt, both tests fail in the compared case, after the cases before it pass:

```
AssertionError: smpc LESS: x2w-draw compute e9f1d94c2215f572 Dispatch(3, 2, 2) refused: a comparison sampler
```

## Not measured

- A bilinear comparison filter, a border address mode, a 2D array and the other six functions. The fixture runs
  point, clamp, LESS and GREATER on a 2D texture with one mip.
- A comparison sampler on the in-process route or from the C++ compiler's publisher. Both still stop before the
  manifest reader, as listed above.
- Console behaviour. The comparison is the desktop sampler's. The function mapping is the draw path's
  (`decode_sampler` in `src/startup_stubs/translated_sampler_bindings.h`).
