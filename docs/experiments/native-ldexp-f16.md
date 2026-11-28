# Direct DXIL `v_ldexp_f16`

Lowers `v_ldexp_f16` (VOP2 `0x3b` / VOP3A `0x13b`) in the native direct-DXIL
compute path (`src/gfx1013_direct_compute.inl`) via exact 32-bit integer
IEEE-754 binary16 scaling while preserving `D[31:16]`.

## Scope

- Covers normal scaling, negative 16-bit signed exponents, denormal-to-normal
  promotion, normal-to-subnormal underflow with round-to-nearest-even ties,
  overflow to `±Inf`, VOP3A `abs`/`neg`/`clamp`, and `D[31:16]` preservation.
- Refuses unsupported modifiers (`omod`, `neg`/`abs` on the integer exponent
  `src1`, `op_sel`, unused `src2`).
- Negative controls: flushing denormals, truncating subnormals instead of
  round-to-nearest-even, omitting `clamp`, and zeroing `D[31:16]` each differ
  from the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_ldexp_f16_' -V
```
