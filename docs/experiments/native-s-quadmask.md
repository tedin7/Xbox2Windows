# Direct DXIL `s_quadmask_b32` and `s_quadmask_b64`

Lowers `s_quadmask_b32` (SOP1 `0x2c`) and `s_quadmask_b64` (SOP1 `0x2d`) in the
native direct-DXIL compute path (`src/gfx1013_direct_compute.inl`), packing one
bit per non-zero 4-bit quad (`bits[7:0]` for `b32`, `bits[15:0]` with zeroed
high dword for `b64`) and setting `SCC = (D != 0)`.

## Scope

- Covers 8 32-bit cases for `s_quadmask_b32` and 4 64-bit cases (including
  hi-only non-zero and in-place destination aliasing) for `s_quadmask_b64`.
- Refuses odd destination or odd source SGPR pairs on `s_quadmask_b64`.
- Negative controls: requiring all 4 quad bits set and ignoring the high dword
  of `s_quadmask_b64` each differ from the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_s_quadmask_' -V
```
