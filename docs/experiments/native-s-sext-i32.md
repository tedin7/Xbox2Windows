# Direct DXIL `s_sext_i32_i8` and `s_sext_i32_i16`

Lowers `s_sext_i32_i8` (SOP1 `0x19`) and `s_sext_i32_i16` (SOP1 `0x1a`) in the
native direct-DXIL compute path (`src/gfx1013_direct_compute.inl`) as 32-bit
arithmetic sign extensions (`ashr(shl(a, 24), 24)` and `ashr(shl(a, 16), 16)`)
that preserve `SCC`.

## Scope

- Covers 8 representative 32-bit inputs across positive, negative, boundary,
  and upper-bit-junk cases for both `s_sext_i32_i8` and `s_sext_i32_i16`,
  verifying both the sign-extended result and `SCC` preservation.
- Negative controls: zero-extending instead of sign-extending and overwriting
  `SCC` with `result != 0` each differ from the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_s_sext_i32_' -V
```
