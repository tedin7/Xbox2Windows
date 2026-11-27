# Scalar conditional moves in the direct C++ compute path (#3070)

`src/gfx1013_direct_compute.inl` now lowers `s_cmov_b32`, `s_cmov_b64` (SOP1) and `s_cmovk_i32` (SOPK) in the direct
compute path:

- **Operation:** `D = SCC ? S0 : D`, leaving `SCC` unchanged.
- **32-bit forms (`s_cmov_b32`, `s_cmovk_i32`):** destination is a general SGPR (`s0..s105`), `vcc_lo` (`106`), or wave32
  `vcc_hi` (`107`). `s_cmovk_i32` sign-extends its 16-bit immediate; `s_cmov_b32` accepts a general SGPR, `vcc_lo`, wave32
  `vcc_hi`, inline constant, or 32-bit literal.
- **64-bit form (`s_cmov_b64`):** destination is an aligned SGPR pair (`s0..s104` even) or `vcc` (`106`). `S0` accepts an
  aligned SGPR pair, inline integer (`-16..64`, sign-extended when negative), or 32-bit literal (`pair_high()` zero-extends
  32-bit literals consistently with `s_mov_b64`, `s_cselect_b64`, `s_not_b64`, and `s_lshr_b64`; unmeasured on wave32
  hardware here).
- **Differences from [`tools/startup/gfx2hlsl.py`](../../tools/startup/gfx2hlsl.py) `scalar_cmov`:** the HLSL startup path
  refuses `vcc` and lane-mask operands and takes only an SGPR pair as the 64-bit `s_cmov_b64` source, whereas the direct
  compute path accepts `vcc` as the `s_cmov_b64` destination, wave32 `vcc_lo`/`vcc_hi` on the 32-bit forms, and inline
  integer or 32-bit literal sources on `s_cmov_b64`.
- **Analysis & liveness:** reading the previous destination through `operand32()` refuses symbolic-descriptor destinations.
  `transfer()` in `src/gfx1013_compute_native.cpp` clears constant/symbolic state on `s_cmovk_i32` (observed by refusing a
  table descriptor load whose high word had been constant zero prior to `s_cmovk_i32`). `uses_scc_after()` in
  `src/gfx1013_direct_compute.inl` marks all three instructions in `reads_scc` (matching the existing `uncertain_scalar`
  coverage).

## Fixture

`tests/native_salu_cmov_check.py ir|controls|hardware` (`ctest --preset dev --no-tests=error -R '^native_salu_cmov_' -V`).
Encodings were verified with LLVM 15.0.7 (`clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm -show-encoding`).

- **IR (`native_salu_cmov_ir`):** dumps DXIL IR via `native_compute_ir_dump` for all 24 cases (`SCC = 0` and `SCC = 1` across
  all 12 accepted forms: `s_cmov_b32` SGPR, literal, `1.0`, wave32 `vcc_hi` dst and src; `s_cmovk_i32` `-1234`, `0x7FFF`,
  wave32 `vcc_hi` dst; `s_cmov_b64` pair, `-1`, literal `0x80000001`, `vcc` dst), evaluates lane 0 to verify exact
  `D = SCC ? S0 : D`, preserved `SCC`, untouched neighbor words, and sign-extended `-1234` (`4294966062` rather than
  `64302`), verifies the 8 invalid operand refusals, verifies 3 no-epilogue symbolic-descriptor destination refusals at exact
  pc `+0x20` (`unbound numeric SGPR s24`), and verifies the `transfer()` `s_cmovk_i32` positive baseline + refusal at `+0x10`
  (`direct descriptor address outside bound table`).
- **Controls (`native_salu_cmov_controls`):** compiles all 12 accepted forms into DXIL via `compile_dxil` and checks
  `dxc.exe -dumpbin` for `select i1`, runs the same `transfer()` baseline and 12 exact diagnostic/pc refusals, and checks
  the 5 mutant references differ.
- **Hardware (`native_salu_cmov_hardware`):** dispatches all 24 cases via `native_store_replay --direct` and checks output
  words, preserved `SCC`, and guard words.

Measured on 27.11.2026, `main` `429f11d5` plus this change, Windows 10.0.26100 (`dev` preset): `native_salu_cmov_ir`,
`native_salu_cmov_controls`, and all 47 `native_*_(controls|ir)` tests passed; `native_salu_cmov_hardware` skipped (`77`,
Quadro M1000M does not support shader model 6.6 with wave32, so GPU readback is not measured). Six deliberate broken
builds (`ignores_scc`, `inverted_select`, `writes_nothing`, `zero_ext_cmovk`, `keeps_old_high_word`, and dropping
`s_cmovk_i32` from `transfer()`) each failed `native_salu_cmov_ir`.
