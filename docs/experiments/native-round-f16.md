# Direct DXIL `v_floor_f16`, `v_ceil_f16`, `v_trunc_f16`, and `v_rndne_f16`

Lowers `v_floor_f16` (VOP1 `0x5b` / VOP3A `0x1db`), `v_ceil_f16` (VOP1 `0x5c` /
VOP3A `0x1dc`), `v_trunc_f16` (VOP1 `0x5d` / VOP3A `0x1dd`), and `v_rndne_f16`
(VOP1 `0x5e` / VOP3A `0x1de`) in the native direct-DXIL compute path
(`src/gfx1013_direct_compute.inl`) using exact 32-bit integer IEEE-754 binary16
rounding while preserving `D[31:16]`.

## Scope

- Covers `-0.0h`, `±min_sub`, `0.5h`, `1.5h`, `2.5h`, `-1.5h`, and `1023.5h`
  (mantissa-to-exponent carry) across VOP1 and VOP3A with pre-seeded `D[31:16]`.
- Refuses unsupported modifiers (`omod`, `op_sel`, unused VOP3 sources).
- Negative controls: round-half-away-from-zero, losing `-0.0h`, and zeroing
  `D[31:16]` each differ from the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_round_f16_' -V
```
