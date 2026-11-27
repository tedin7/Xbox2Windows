# Direct DXIL `v_xnor_b32`

Lowers `v_xnor_b32` (VOP2 `0x1e`, VOP3A `0x11e`, and VOP2 SDWA) in the native
direct-DXIL compute path (`src/gfx1013_direct_compute.inl` and
`src/gfx1013_compute_native.cpp`) as `~(S0 ^ S1)`.

## Scope

- Covers VOP2 VGPR+VGPR, VOP2 literal+VGPR, VOP3A VGPR+VGPR, and VOP2 SDWA
  (`SEXT(BYTE_0)` with `WORD_0`) in a synthetic direct-compute fixture.
- Negative control: omitting the final bitwise invert (`S0 ^ S1`) differs from
  the reference readback.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_v_xnor_b32_' -V
```
