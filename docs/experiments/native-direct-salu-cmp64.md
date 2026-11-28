# 64-bit scalar SOPC comparisons in the direct C++ compute path

Issue #3078 (part of #2935) lowers the remaining 64-bit SOPC scalar comparison
and bit-test instructions in the direct C++ compute DXIL path
(`src/gfx1013_direct_compute.inl`): `s_cmp_eq_u64`, `s_cmp_lg_u64`,
`s_bitcmp0_b64`, and `s_bitcmp1_b64`.

All four instructions write `SCC` only and leave scalar registers untouched.
`s_cmp_eq_u64` and `s_cmp_lg_u64` accept aligned even SGPR pairs, `null` (`0`),
sign-extended negative inline integers (`-1..-16`), and zero-extended positive
inline integers (`0..64`) or 32-bit literals (`0x80000001` -> `(0x80000001, 0)`),
while refusing `vcc`, `exec`, `m0`, `scc`, floats, and odd SGPRs.
`s_bitcmp0_b64` and `s_bitcmp1_b64` test bit `S1 & 63` of the 64-bit `S0` operand
using pure 32-bit DXIL operations (no `Int64Ops` shader feature bit required).

## Verification

`native_salu_cmp64_ir` dumps DXIL IR via `native_compute_ir_dump` for all 22
test cases, asserts pure 32-bit IR (no 64-bit integer IR ops), evaluates lane 0
to verify both the preserved SGPR pair (`s[22:23]`) and exact `SCC` output
(pinning compare operands, sign/zero-extension rules, and bit-5 high-word
selection), and verifies all 7 negative operand refusal controls.
`native_salu_cmp64_controls` compiles all 22 test cases together with
`compile_dxil`, checks `dxc.exe -dumpbin` for pure 32-bit IR (`icmp eq i32`,
`icmp ne i32`, `lshr i32`) with no 64-bit integer IR ops, checks 7 negative
operand refusal controls, and checks 6 oracle mutants (`low_word_only_cmp`,
`high_word_only_cmp`, `zero_ext_neg_inline`, `sign_ext_lit`,
`ignore_bit5_bitcmp`, `inverted_bitcmp`). `native_salu_cmp64_hardware` runs
`native_store_replay --direct` and returns skip code 77 when shader model 6.6
wave32 hardware is unavailable.

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay native_compute_ir_dump
ctest --preset dev --no-tests=error -R '^native_salu_cmp64_' -V
ctest --preset dev --no-tests=error -R '^native_.*_(controls|ir)$' --output-on-failure
```

Results (Windows `10.0.26100`, `dev` preset, base `88cddb09`):
- `native_salu_cmp64_ir`: Passed (`0.98s`) — evaluated DXIL IR for all 22 cases in pure 32-bit IR (`s[22:23]` preserved + exact `SCC`), verified 7 operand refusals, and caught 6 oracle mutants. Five deliberate broken lowerings (`low_word_only_cmp`, `high_word_only_cmp`, `swap_and_or`, `ignore_bit5`, `invert_bitcmp`) each failed `native_salu_cmp64_ir` (`exit 1`).
- `native_salu_cmp64_controls`: Passed (`1.14s`) — compiled all 22 cases together in pure 32-bit DXIL IR (`6.77 ms` cold DXIL compile, `6948` bytes), passed 7 negative operand refusal checks, and caught all 6 SCC oracle mutants.
- `native_salu_cmp64_hardware`: Skipped (exit code `77`, `requires shader model 6.6 with wave32`) on local NVIDIA Quadro M1000M (SM 6.1); GPU readback is unmeasured locally.
- `ctest --preset dev --no-tests=error -R '^native_.*_(controls|ir)$' --output-on-failure`: `47/47` passed (`51.67s`).
- `tests/docs_links_check.py`: `102` Markdown files, `589` links passed.
