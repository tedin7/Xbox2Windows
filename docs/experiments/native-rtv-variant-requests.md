# RTV variant requests in `compile_dxil --requests` (#2980)

[The fixture](../../tests/native_rtv_variant_check.py) gives `compile_dxil gfx1013-container-pair --requests` a
directory with one source-built VS/PS base request and RTV variant requests that share its base key. It checks the
published entries, their format metadata and, through the stand-in, a fenced readback of a target of each format.

## What changed

1. **Reader (`src/shader_compute_request.cpp`).** `read_graphics_request` accepts `req_<key>.rtv`. The text is
   checked as `tools/startup/gfx2hlsl.py::request_source` checks it: at most 256 bytes, `rtv1 <base key>
   <format> ...` in decimal with one to eight formats and nothing else, and the FNV-1a hash of the text equal to
   the key in the file name (`invalid RTV variant request`); the base key at most 64 bits and each format at most
   32 bits (`invalid RTV variant values`). The base parts `req_<base>.{a,b,root}` and the optional `.ia` and
   `.psin` are then read and checked against the base key by the existing base path. Its refusals are prefixed
   with `RTV variant source: `.
2. **Translation (`src/direct_compute_cache.cpp`, `src/gfx1013_direct_graphics.inl`).** The ordered formats type
   the pixel stage's color targets as `mrt_kind` does in `gfx2hlsl.py`: `u32` or `i32` for the integer formats,
   `f32` for every other format. `R32G32B32_UINT` (7) and `R32G32B32_SINT` (8) have no verified readback there
   and are refused as `unverified UINT RTV format 7` and `unverified SINT RTV format 8`, as there only when the
   pixel stage exports to that target. A float32 target needs no extra mechanism: the pixel stage stores the
   register bits as a float.
3. **Cache and manifest.** The target types are part of the `.pairir` key, so formats of one type share a
   frontend entry. A base request adds nothing to its key, and its translation takes the same path as before.
   (The frontend sources feed `X2W_NATIVE_SHADER_REVISION`, so existing `.pairir` entries are rekeyed once, as
   with any frontend change.) The runtime manifest accepts an integer target only for the variant whose format
   asked for it, and ends with the request text, which the runtime compares with the request it made. The entry
   is published as `<variant key>.vs.cso`, `<variant key>.ps.cso` and then `<variant key>.txt`.
4. **Tool (`tools/startup/compile_dxil.cpp`).** The pair profile scans `req_<key>.rtv` next to `req_<key>.a`. The
   report line is `rtv_variant <request> types=<type>,...`. A variant whose pair has no runtime manifest (an
   input layout or vertex fetch, for example) is refused as `RTV variant has no runtime manifest`, where a base
   pair still publishes its stages. A key with both markers is one request, the variant, as in `request_source`.
   The rectangle profile does not take variants.

Two differences from `gfx2hlsl.py`, both refusals where it translates: it types a target by the format alone,
while a container whose own output signature gives another type is refused here (`RTV variant format disagrees
with the pixel signature of target N`); and it has no runtime-manifest condition.

The stand-in is not changed. It already publishes `req_<key>.rtv` and loads `<variant key>.txt`.

## Pair, formats and expected values

The vertex container covers the viewport from the vertex index. The pixel container moves the words `ffff7ffe`,
`15`, `eb` and `68` into `v0..v3` and exports them to MRT0.

| Request formats | Target types | `SV_Target0` in the compiled stage | Expected MRT0 bytes |
|---|---|---|---|
| `R16G16B16A16_UINT` (12), `R8G8B8A8_UNORM` (28) | `u32,f32` | `uint` | `ffff 1500 eb00 6800`: the first word clamps to 65535 |
| `R16G16B16A16_SINT` (14), `R8G8B8A8_UNORM` (28) | `i32,f32` | `int` | `0080 1500 eb00 6800`: the first word, -32770, clamps to -32768 |
| `R32G32B32A32_FLOAT` (2) | `f32` | `float` | `fe7fffff 15000000 eb000000 68000000`: all four words bit for bit, the first a NaN with a payload |

For the two integer formats the consumer binds a second RGBA8 target, which the pixel stage does not write. It
keeps the clear value `0000ffff`. The base pair drawn to an RGBA8 target stores `00000000` over the blue clear
(a NaN and three denormals as UNORM).

## Controls

`native_rtv_variant_controls` needs no device:

1. The base, the three variants of the table and a fourth, `R16G16_UINT` (36) with RGBA8, compile from an empty
   compiler cache with `--jobs 4` (`requests=5 compiled=5 refused=0`). The two UINT variants have the same target
   types and share one frontend entry. Each variant manifest is the base manifest plus the request line, the
   report names the types, and `dxc -dumpbin` shows `SV_Target0` as `uint`, `int` and `float`. The base manifest
   has no `rtv1` line and its pixel stage is `float`. A second run reports a frontend cache hit and two bytecode
   cache hits for every request and publishes the same bytes. `gfx2hlsl.pending_keys` then lists nothing.
2. Refusals, each leaving only `<key>.fail` with the reason under the variant key (the `.fail` lifecycle of
   [#2984](native-compile-dxil-refusals.md)): a forged variant key, base content that no longer
   matches the base key, no format, nine formats, a format above 32 bits, a trailing newline, a missing base,
   format 7 and format 8 in the exported slot, and a forged variant file under the base's own key. Where the
   base request is intact, it still compiles (`compiled=1 refused=1`). Format 7 in the slot the pixel stage does
   not export to compiles (`types=u32,u32`); moved to the exported slot it is refused although the frontend entry
   of those types is cached by then. A base with an input layout publishes its stages without a manifest, as
   before; its variant is refused for having none, and again when the layout no longer matches the base key.
3. A published variant whose base then changes is refused on the next run (`requests=5 compiled=0 refused=5`)
   and no `<key>.txt` is left in the directory.
4. With a pixel container whose output signature says `uint`, the UINT variant compiles, the FLOAT variant is
   refused by name, and the base pair publishes no manifest, as before.

`native_rtv_variant_{warp,hardware}`: the consumer creates the pair through device slot 86 and draws one pixel
into a target of each format, so the stand-in publishes the base request and the three variant requests. The
tool compiles the directory cold and warm with the checks of step 1. The consumer then draws each format in a
new process. The log names the desktop pipeline for the variant key with the bound formats (`0x00000000`) and the
drawn call, and the readback after the fence equals the table above. The base pair is drawn last.

## Command and result

```powershell
cmake --preset dev   # run here with -DINTAKE_PYTHON=<an existing out/pe-env python.exe>, the worktree having none
cmake --build --preset dev
ctest --preset dev --no-tests=error -V -R '^native_rtv_variant_'
```

Commit `be9a0706` on `cpp-migration` at `4ff5e36d`, Windows 11 build 26300.9974, Debug build, Windows SDK
10.0.26100 DXC, D3D12 debug layer enabled.

| Test | Adapter, driver | Result |
|---|---|---|
| `native_rtv_variant_controls` | none (no device) | PASS. |
| `native_rtv_variant_hardware` | NVIDIA GeForce RTX 4090, 32.0.16.1088 | PASS. The three readbacks match the table. No debug-layer error message. |
| `native_rtv_variant_warp` | Microsoft Basic Render Driver, 10.0.26100.9818 | PASS. The three readbacks match the table. No debug-layer error message. |

Release: the same commit built with the `release` preset (`cmake --preset release`, `cmake --build --preset
release`, `ctest --preset release` with the same filters; MSVC 19.44.35230) on the same machine. The three tests
pass on both adapters with the same readbacks and no debug-layer error message, the controls pass 12 runs in a
row, and the first regression set below gives the same 45 passes and 3 capability skips. The pull request
preset is a Debug preset and was not run in Release.

Baseline: at `05438d65` the tool does not scan `.rtv` files. A directory with a base request and one variant
request gives `requests=1 compiled=1 refused=0` and nothing under the variant key (run with a format 2 request
published by the stand-in on each of the two adapters).

Mutation: with the format ignored where the target is typed, `native_rtv_variant_controls` and
`native_rtv_variant_warp` fail. This was run on an earlier state of the branch, on `05438d65`; the mutated line
is the same in `be9a0706`.

Regression runs at the same commit, same machine:

```powershell
ctest --preset dev --no-tests=error -R '^(native_rtv_variant_|native_compile_dxil_refusals|native_startup_request_cache|startup_mrt_|native_direct_container|native_sampled_replay|native_global_srv_heap_replay|native_recording|native_graphics|native_dxil_llvm_ident|native_.*_pso_|startup_stream_|startup_compute_dispatch_native)'
ctest --preset pr --no-tests=error
```

- The first set: 45 of 48 pass. Capability skips, not passes: `native_comparison_pso_warp`,
  `native_global_srv_heap_replay_warp` and `startup_compute_dispatch_native_warp`.
- The pull request preset: 36 of 40 pass. Capability skips, not passes: `spirv_minlod_warp`,
  `spirv_minlod_exact_warp`, `spirv_minlod_levels_warp` and `spirv_minlod_fractional_warp`.

These two sets were not run at `4ff5e36d` without the change, so whether the seven skips predate it was not
measured here.

## Limits

- Three formats were drawn: 12, 14 and 2. The other integer formats take the same two types and were compiled
  by no fixture here. The readback sets of `gfx2hlsl.py` were verified with that translator's output, not with
  this one.
- One pair: no resources, no samplers, no root constants, a pixel stage that writes MRT0 only. A variant whose
  pixel stage writes two typed targets was not run.
- The readbacks use containers without an output signature. The signature branch is compiled by the controls
  and was not drawn.
- The controls ran the directory with four workers 12 times in a row without a failure. That is the only
  evidence for two requests publishing one frontend entry side by side, beyond reading the cache code.
- The variants came from slot 86 containers. A raw slot 57 pair with a `.psin` goes through the same reader
  path, and no fixture ran one as a variant.
- A format that is not an integer format is typed `f32` without a check that it is a renderable format, as in
  `gfx2hlsl.py`.
- The runtime's `mrtbits` entry value is not used: the native pixel stage keeps float32 bits without it. This
  was measured for the four words above on the two adapters only.
- Only the two adapters above were run, in Debug and Release builds on one machine. No start-up run used a
  variant compiled this way.
