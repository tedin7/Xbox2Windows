# Direct DXIL `v_frexp_exp_i32_f32` and `v_frexp_mant_f32`

Lowers `v_frexp_exp_i32_f32` (VOP1 `0x3f` / VOP3A `0x1bf`) and
`v_frexp_mant_f32` (VOP1 `0x40` / VOP3A `0x1c0`) in the native direct-DXIL
compute path (`src/gfx1013_direct_compute.inl`) via exact 32-bit integer
IEEE-754 binary32 manipulation.

## Scope

- Covers ±0.0, normals, smallest normal, largest and smallest subnormals, and
  +Inf across VOP1 and VOP3A (`neg`, `abs`, and `clamp` on `v_frexp_mant_f32`).
- Refuses unsupported modifiers (`clamp` on `v_frexp_exp_i32_f32`, `omod` on
  `v_frexp_mant_f32`).
- Negative controls: flushing denormals and omitting `clamp` each differ from
  the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_frexp_f32_' -V
```
