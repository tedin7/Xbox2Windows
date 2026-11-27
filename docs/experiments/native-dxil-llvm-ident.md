# Direct DXIL `!llvm.ident` metadata for AMD RDNA 4 pipeline creation (#2915)

[The fixture](../../tests/native_dxil_llvm_ident_check.py) verifies that every direct DXIL emitter in
`src/dxil_module.h` (`DxilModule::finish_compute`, covering `cs`, `vs`, `gs`, and `ps`) and
`src/gfx1013_pixel.cpp` (`x2w::direct_gfx1013_constant_pixel`) emits `!llvm.ident = !{!<id>}` (`!{!"x2w"}`),
and draws both direct pixel paths on WARP and hardware with negative controls.

## Cause of the AMD `0xC0000005` PSO crash

On `cpp-migration` (`40365770`..`ee76dfa8`), all direct-DXIL `_hardware` tests crashed with `0xC0000005`
inside `amdxc64.dll` during `CreateComputePipelineState` and `CreateGraphicsPipelineState` on an
AMD Radeon RX 9070 XT (`1002:7550`, driver `32.0.32029.2010`), even though `IDxcValidator::Validate`
accepted every container and both WARP and NVIDIA passed:

- `native_scalar_lds_hardware`
- `native_adjacent_srv_hardware`
- `native_mtbuf_hardware`
- `native_indexed_atomic_hardware`
- `native_dpp_mirror_hardware`
- `native_graphics_wqm_hardware`
- `native_graphics_vertex_vcc_join_hardware`

Comparing disassembled DXIL modules (`dxc.exe -dumpbin`) re-assembled through `IDxcAssembler::AssembleToContainer`
and `IDxcValidator::Validate` isolated the crash to a single missing named metadata entry:

- Re-assembling DXC's `ps_6_6` output passed (`rc = 0`, `debug_errors=0 debug_warnings=0`).
- Re-assembling `DxilModule` output as-is crashed (`0xC0000005`), and bumping `!dx.valver` to `1.8` or `1.9`
  alone still crashed (`0xC0000005`).
- Adding `!llvm.ident = !{!<id>}` (`!{!"x2w"}`) alone made the identical `DxilModule` IR pass (`rc = 0`,
  `debug_errors=0 debug_warnings=0`).

`dxc.exe` always emits `!llvm.ident`; `IDxcValidator` treats it as optional, whereas `amdxc64.dll` queries
`llvm::Module::getNamedMetadata("llvm.ident")` and dereferences the returned `NamedMDNode*` (`[rcx + 0x40]`)
without a null check.

## Verification and negative controls

1. **`container` mode (`native_dxil_llvm_ident_controls`):** compiles source-built shaders across
   `gfx1013-container-cs`, `gfx1013-container-vs`, `gfx1013-container-ps`, `gfx1013-container-rect`
   (`vs`, `gs`, `ps`), and `gfx1013-direct-constant-ps`, checks `dxc.exe -dumpbin` for `!llvm.ident`
   resolving to `!"x2w"`, and runs a negative control that strips `!llvm.ident`, re-assembles + validates
   through `dxcompiler.dll` / `dxil.dll`, and verifies the `!llvm.ident` check rejects the stripped module.
2. **`draw` mode (`native_dxil_llvm_ident_warp`, `native_dxil_llvm_ident_hardware`):** pairs a direct-DXIL
   fullscreen-triangle vertex shader (`gfx1013-container-vs`) with `gfx1013-direct-constant-ps` and
   `gfx1013-container-ps` on an `8x2` float target, checks bit-exact RGBA readback and zero D3D12 debug-layer
   errors/warnings, and requires mutated-channel negative controls on each path to fail readback.

```powershell
cmake --build --preset release --target compile_dxil native_store_replay native_recording_check offscreen_check
ctest --preset release --no-tests=error -V -R "^native_dxil_llvm_ident_(controls|warp|hardware)$"
```

Verified on Windows 11 (`10.0.26100`), Release build, with the D3D12 debug layer enabled (`0` errors, `0` warnings)
across Microsoft Basic Render Driver (`1414:008c`, driver `10.0.26100.9818`) and AMD Radeon RX 9070 XT (`1002:7550`,
driver `32.0.32029.2010`), alongside all seven direct-DXIL `_hardware` tests listed above.

## Limits

- The committed `draw` check does not execute the stripped negative-control container on a GPU adapter, because
  triggering an access violation inside a user-mode driver is driver-version-dependent rather than a D3D12
  contract.
- The separate adapter-independent decoder refusal message in `native_wave_request_hardware` is tracked independently
  on #2915.
