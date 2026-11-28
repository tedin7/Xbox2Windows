# 64-bit scalar shifts in the direct C++ compute path (#3062)

`src/gfx1013_direct_compute.inl` now lowers the SOP2 64-bit scalar shifts `s_lshl_b64`, `s_lshr_b64` and `s_ashr_i64`
matching `tools/startup/gfx2hlsl.py`'s `scalar_wide_op`:

- **Destination:** aligned SGPR pair (`s0..s104` even) or `vcc` (`106`).
- **Source 0 (`S0`):** aligned SGPR pair (`s0..s104` even), inline integer (`-16..64`), or 32-bit literal (zero-extended
  for `s_lshl_b64` and `s_lshr_b64`, sign-extended for `s_ashr_i64`).
- **Source 1 (`S1`):** general SGPR (`s0..s105`), inline integer (`-16..64`), or 32-bit literal, masked with `& 63`.
- **Execution:** all sources are read before writing `D[0:1]` so overlapping destinations (`s[22:23]` as both source and
  destination) keep their original input values. Sets `ir.features |= 1048576` (`Int64Ops`) and `SCC = (D != 0)`.
- **Analysis & liveness:** `src/gfx1013_compute_native.cpp` (`transfer()`) now clears `s[dst + 1]` for SOP1/SOP2
  `_i64` instructions as well as `_b64`, preventing a stale symbolic descriptor tag on `s_ashr_i64`. `uses_scc_after()`
  in `src/gfx1013_direct_compute.inl` marks all three shifts as overwriting `SCC`.

## Fixture

`tests/native_salu_shift64_check.py controls|hardware` (`ctest --preset dev --no-tests=error -R '^native_salu_shift64_' -V`).
Encodings were verified with LLVM 15.0.7 (`clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm -show-encoding`).

- **Controls:** compiles a combined shader exercising `s_lshl_b64`, `s_lshr_b64` and `s_ashr_i64` and checks `dxc.exe -dumpbin`
  for `shl i64`, `lshr i64` and `ashr i64`; compiles an `s_load_dwordx2` -> `s_ashr_i64` -> `v_mov_b32` sequence confirming
  both symbolic pair tags are cleared; checks 7 invalid destination/source refusals (`exec`, odd destination, odd `S0`,
  `vcc` as `S0`, inline float as `S0`, inline float as `S1`, `vcc_lo` as `S1`); checks 6 mutant references differ.
- **Hardware cases:** SGPR pair `(0x89ABCDEF, 0x81234567)` across amounts `0, 1, 31, 32, 33, 63` and wrapped `65` via odd
  SGPR `s27`; inline integers `-1, -16, 1, 64`; 32-bit literal `0x80000001` for `s_lshr_b64` and `s_ashr_i64`; overlapping
  in-place `s[22:23]` shifts; high-word-only `vcc` (`SCC = 1`); zero result clearing primed `SCC = 1`.

Measured on 27.11.2026, `main` `db28a796` plus this change, Windows 10.0.26100 (`dev` preset): `native_salu_shift64_controls`
passed; `native_salu_shift64_hardware` skipped (`77`, Quadro M1000M does not support shader model 6.6 with wave32, so
readback is not measured).
