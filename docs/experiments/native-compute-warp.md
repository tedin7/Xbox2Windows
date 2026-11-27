# Native wave32 compute on app-local WARP 1.0.22

Refs [#3160](https://github.com/R0Y00/Xbox2Windows/issues/3160) (stage 1),
[#145](https://github.com/R0Y00/Xbox2Windows/issues/145), [#147](https://github.com/R0Y00/Xbox2Windows/pull/147).

Every direct compute module declares wave size 32 (`src/dxil_module.h`). The in-box WARP has 4-lane waves, so the
native compute `_warp` tests skip with 77. This note measures them with the Microsoft WARP 1.0.22 package placed
beside the test executables. No translator or test code changes.

## Result

**The app-local `d3d10warp.dll` is what provides 32-lane waves. The Agility SDK runtime alone does not:** loaded
through `CreateDeviceFactory` (see below), Agility 1.620.2 with the in-box WARP reports 4..4 lanes. On this machine,
WARP 1.0.22 loaded by the in-box D3D12 runtime reports `WaveOps=1`, lanes 4 to 128 and SM 6.8. A configure with
`OFFSCREEN_WARP_DIR` already copies the DLL into the build root, next to every test executable, so the tests below
use it without any change to their CMake targets.

| Test | Executable | In-box runtime, in-box WARP (`10.0.26100.9818`) | In-box runtime, WARP 1.0.22 beside the EXE |
|---|---|---|---|
| `native_comparison_pso_warp` | `native_recording_check` | skip 77, `range=4..4` | pass: cold and cached comparison-sampling compute PSOs (no dispatch) |
| `native_heap_handle_probe_warp` | `native_heap_handle_probe` | skip 77, `(4..4)` | pass: reference and native modules, selectors 0 and 1 match the CPU reference, guard intact |
| `native_global_srv_heap_replay_warp` | `native_global_srv_heap_replay` | skip 77, `(4..4)` | pass: record and offline replay, 64 words per selector, refusals |
| `startup_compute_dispatch_native_warp` | `startup_stream_consumer` | skip 77, refusal checks pass | **fail**: `Dispatch(3, 2, 2) translated`, then `word=0 got=c5c5c5c5 expected=ea74e376 (288 words differ)` |
| `native_startup_request_cache_warp` | `startup_stream_consumer`, `offscreen_check` | pass (4-lane fallback branch) | **fail**: `compute dispatch output mismatch` (translated branch) |
| `shader_wave32_warp` (gate) | `offscreen_check` | not run | pass: `forced32 width=32 waves=2 ... mismatches=0`, debug layer enabled, 0 errors |
| `bvh_wave64_warp` (#145) | `offscreen_check` | not run | **fail**: `record=0 field=25 expected=f0094d90 actual=f03aeb40` |

Control: with `d3d10warp.dll` renamed away, the same `warp122` binaries of `native_heap_handle_probe_warp` and
`native_global_srv_heap_replay_warp` report `(4..4)` and skip with 77. The `native_comparison_pso_warp` row in the
in-box column comes from the options-unset build. The three executables have no export table, so the in-box `D3D12Core.dll` loads. A scratch probe started from the same
directory printed `C:\WINDOWS\SYSTEM32\D3D12Core.dll` and the `d3d10warp.dll` from the build root.

Negative control: changing one expected word in `reference_output()` of `tests/native_global_srv_heap_replay.cpp`
(a temporary local edit, reverted) makes `native_global_srv_heap_replay_warp` fail with
`mismatch word 5: got 0xa0000005 expected 0xa0000004`. The pass is a real comparison, not a skip.

With the options unset (an options-unset build of this commit), all five native compute tests give the same results
as before: four skip with 77 and `native_startup_request_cache_warp` passes through its 4-lane branch.

## The two stand-in failures

Both run the native module through the stand-in (`d3d12_xs_shim`). The dispatch is recorded and the fence signals,
but the output keeps its `0xc5c5c5c5` fill. This is not the #145 shape, where values arrive from the wrong lanes:
here no word is written. `startup_compute_dispatch_warp` uses the same stand-in, executable and WARP, with an HLSL
module compiled by DXC, and it passes. A rerun with `X2W_TEST_DEBUG_LAYER=1` set printed no `x2w-d3d12-debug` line.
I did not confirm that the debug layer was active on that path.

On the RTX 4090 (driver 32.0.16.1088), `startup_compute_dispatch_native_hardware` (all 320 words match, from an
empty cache and from a published translation) and `native_startup_request_cache_hardware` pass in the same build.
The failure is therefore specific to WARP. It happens with a native module dispatched through the stand-in, and not
with the standalone native fixtures or with an HLSL module dispatched through the stand-in.

**Still open:** the cause. A WARP 1.0.22 defect and a translator or stand-in construct that hardware tolerates are
both possible.

## Agility SDK 1.620.2 on this machine

- **The export-based load fails.** Exporting `D3D12SDKVersion = 620` and `D3D12SDKPath = ".\\D3D12\\"` (the
  `offscreen_check` pattern) makes every D3D12 call return `0x887E0003` (`D3D12_ERROR_INVALID_REDIST`). The loader's
  debug output reads `Unable to load D3D12Core.dll. Verify D3D12SDKPath is correct.` The same happens from a second
  directory on another drive. With these exports added as a local edit, the three standalone tests above skip or fail.
- **The DLL itself is fine.** `LoadLibrary` of the same `D3D12Core.dll` succeeds.
  `ID3D12SDKConfiguration1::CreateDeviceFactory(620, <absolute path>)` loads it, and WARP 1.0.22 again reports lanes
  4 to 128.
- **Unknown:** the cause of the export-path failure on this build.
- **The opt-in build does not compile on current `main`.** `src/startup_stubs/d3d12_compute_dispatch.inl` asserts
  `sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC) == 40` and the same for the UAV description. With the Agility 1.620 headers
  the UAV description is 48 bytes (the SRV description stays 40), so `d3d12_xs_shim` and `offscreen_check` fail to build when `OFFSCREEN_AGILITY_620_DIR` is set.
  That is why the last four rows were measured in an options-unset build with the DLL copied in by hand.

## #145 and the native lowerings

`bvh_wave64_warp` still fails at field 25, the non-uniform `WaveReadLaneAt` shuffle that #145 describes. On this
machine the value is different. With the #145 encoding (`0xf0000001 ^ (lane * 0x319db)`):
- expected `f0094d90` is lane 3;
- this run's `f03aeb40` decodes to lane 19;
- the `f053ba18` in #145 decodes to lane 27, the inverse permutation.

This run used the in-box runtime, not Agility. I have not investigated it further.

None of the three passing native tests lowers `v_readlane`, DPP or `v_permlane*`. Those fixtures run on
`native_store_replay`, which creates its device on the default adapter and has only `_hardware` entries. So whether
#145 affects native compute on WARP is still unmeasured.

## Reproduce

Microsoft NuGet packages, not committed: [Agility SDK 1.620.2](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.620.2)
(35,006,223 bytes, kept as `sdk.nupkg` next to its x64 `D3D12Core.dll` and `d3d12SDKLayers.dll`) and
[WARP 1.0.22](https://www.nuget.org/packages/Microsoft.Direct3D.WARP/1.0.22) (15,564,228 bytes, extracted).

```powershell
cmake --preset dev -B out/build/warp122 "-DOFFSCREEN_AGILITY_620_DIR=<agility dir>" "-DOFFSCREEN_WARP_DIR=<warp>/build/native/bin/x64"
cmake --build out/build/warp122 --target compile_dxil native_recording_check native_heap_handle_probe native_global_srv_heap_replay
ctest --test-dir out/build/warp122 --no-tests=error -j1 -V -R '^native_(comparison_pso|heap_handle_probe|global_srv_heap_replay)_warp$'
```

The stand-in rows and the gate rows need `offscreen_check` and `startup_stream_consumer`, which do not build with the
Agility option (see above). Build them with the options unset and copy the DLL in for the run:

```powershell
cmake --build --preset dev --target offscreen_check compile_dxil startup_stream_consumer
Copy-Item <warp>/build/native/bin/x64/d3d10warp.dll out/build/dev/
ctest --test-dir out/build/dev --no-tests=error -j1 -V -R '^(startup_compute_dispatch_native|native_startup_request_cache|shader_wave32|bvh_wave64)_warp$'
Remove-Item out/build/dev/d3d10warp.dll
```

## Environment

Commit `88cddb09`. Windows 11 build 26300, in-box `D3D12Core.dll` 10.0.26100.9818 (SDK version 616). MSVC
14.44.35207, Windows SDK 10.0.26100 (DXC from that SDK), Debug build. NVIDIA GeForce RTX 4090 (driver 32.0.16.1088),
used only for the two stand-in `_hardware` runs. WARP `d3d10warp.dll` 1.0.22.0.20261027.1 and Agility `D3D12Core.dll`
1.620.2.0.20261106.5 were Microsoft-signed. `offscreen_check` reported `debug_layer=enabled`. The native fixtures
do not enable the debug layer.

## Hypotheses

- The in-box result depends on this Windows build. On Windows 10, #147 found that WARP 1.0.22 on the in-box runtime
  misses every DXR ray. That is why `OFFSCREEN_WARP_DIR` requires `OFFSCREEN_AGILITY_620_DIR`. The in-box runtimes of
  CI runner images have not been measured.
- The cause of the two WARP-only stand-in failures is unknown (see above).
