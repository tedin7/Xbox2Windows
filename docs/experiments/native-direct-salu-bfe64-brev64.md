# 64-bit scalar bitfield extract and bit-reverse in the direct C++ compute path

Issue #3086 (part of #2935) lowers `s_brev_b64`, `s_bfe_u64`, and `s_bfe_i64`
in `src/gfx1013_direct_compute.inl` and updates the symbolic pair-clear rule
in `src/gfx1013_compute_native.cpp` (`transfer()`) to include `_u64`.

- `s_brev_b64`: lowers via pure 32-bit DXIL `@dx.op.unary.i32` opcode `30`
  (`reversebits`) on both halves before writing the aligned even SGPR pair
  (swapping low and high words), leaving `SCC` unchanged and requiring no
  `Int64Ops` feature bit.
- `s_bfe_u64` / `s_bfe_i64`: reads `S0` (aligned even SGPR pair, `null`,
  inline integer `-16..64`, or 32-bit literal sign-extended only for
  `s_bfe_i64`) and `S1` (`S1[5:0]` offset, `S1[22:16]` width) before writing
  the aligned even SGPR destination pair and setting `SCC = (D != 0)`.
- Symbolic liveness: `transfer()` in `src/gfx1013_compute_native.cpp` now
  clears `s[dst + 1]` for `_u64` suffix instructions (`s_bfe_u64`), and
  `uses_scc_after()` in `src/gfx1013_direct_compute.inl` marks
  `s_bfe_u64`/`s_bfe_i64` as overwriting `SCC` and `s_brev_b64` as preserving `SCC`.

## Verification

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay native_compute_ir_dump
ctest --preset dev --no-tests=error -R '^(native_salu_bfe64_brev64_|native_salu_shift64_|native_salu64_)' -V
ctest --preset dev --no-tests=error -R '^native_.*_(controls|ir)$' -j 4 --output-on-failure
```

Results (base `ca659396`):
- `native_salu_bfe64_brev64_controls`: Passed (`1.30s`) — verified pure 32-bit DXIL lowering of `s_brev_b64` (`@dx.op.unary.i32(i32 30, ...)` without `i64` ops), combined compilation of all 21 `s_brev_b64`/`s_bfe_u64`/`s_bfe_i64` cases, symbolic `transfer()` `_u64` descriptor-clobber refusal, 9 negative operand/destination refusal controls, and all 6 oracle mutants.
- `native_salu_bfe64_brev64_hardware`: Skipped (exit code `77`, `requires shader model 6.6 with wave32`) on local NVIDIA Quadro M1000M (SM 6.1); GPU readback is unmeasured locally.
- `ctest --preset dev --no-tests=error -R '^native_.*_(controls|ir)$' -j 4`: `27/27` passed (`18.72s`).
- `tests/docs_links_check.py`: `79` Markdown files, `586` links passed.
