# 64-bit scalar bit counts and bit scans in the direct C++ compute path (#3072)

`src/gfx1013_direct_compute.inl` now lowers the six 64-bit-source SOP1 bit-count and bit-scan instructions
`s_bcnt0_i32_b64`, `s_bcnt1_i32_b64`, `s_ff0_i32_b64`, `s_ff1_i32_b64`, `s_flbit_i32_b64` and `s_flbit_i32_i64`
using 32-bit DXIL `unaryBits.i32` (`31` `Countbits`, `32` `FirstbitLo`, `33` `FirstbitHi`), matching
`tools/startup/gfx2hlsl.py`:

- **Destination:** general 32-bit SGPR (`s0..s105`), leaving `s[dst + 1]` untouched.
- **Source `S0`:** aligned SGPR pair (`s0..s104` even), EXEC (`exec_lo`/`exec_hi`, both words, in wave32 and wave64), inline integer (`-16..64`, sign-extended when negative), or
  32-bit literal (sign-extended for `s_flbit_i32_i64`, zero-extended for `_b64`).
- **Semantics & `SCC`:** `s_bcnt0_i32_b64` and `s_bcnt1_i32_b64` sum 32-bit `Countbits` over both halves and write
  `SCC = (D != 0)`. `s_ff0_i32_b64`, `s_ff1_i32_b64`, `s_flbit_i32_b64` and `s_flbit_i32_i64` preserve `SCC` and return
  `0xffffffff` when no qualifying bit exists.
- **Analysis & liveness:** `transfer()` in `src/gfx1013_compute_native.cpp` checks `i.name.find("_i32_") == Name::npos`
  before clearing `s[dst + 1]` on `_b64` SOP1/SOP2 instructions so 32-bit-destination `_i32_b64` ops do not clobber
  adjacent descriptors. `uses_scc_after()` in `src/gfx1013_direct_compute.inl` marks `s_bcnt0_i32_b64` and
  `s_bcnt1_i32_b64` as overwriting `SCC`.

## Fixture

`tests/native_salu_bitscan64_check.py controls|hardware` (`ctest --preset dev --no-tests=error -R '^native_salu_bitscan64_' -V`).
Encodings were verified with LLVM 15.0.7 (`clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm -show-encoding`).

- **Controls:** compiles all 6 ops and checks `dxc.exe -dumpbin` for `@dx.op.unaryBits.i32` opcodes `31`, `32`, `33`
  with no `i64` shifts; tests `s_ff1_i32_b64 s31, s[22:23]` right below `s[32:35]` to verify `s32` stays intact;
  checks 6 invalid destination/source refusals (`vcc`/`exec` dst, odd/`vcc`/`exec_hi`/float `S0`; since EXEC became a valid `S0`, the source control uses the odd `exec_hi`); checks 6 mutant
  references differ.
- **EXEC as `S0`:** read back with `s_bcnt1_i32_b64` by `tests/native_wave64_exec_salu_check.py`, including masks that
  span both host halves of a wave64 (`ctest --preset dev --no-tests=error -R '^native_wave64_exec_salu_hardware$' -V`).
  The other five ops accept EXEC through the same source check but are not run with it.
- **Hardware cases:** all 6 ops across `(0, 0)`, `(0xffffffff, 0xffffffff)`, `(0, 1)`, `(0x80000000, 0)`,
  `(0xfffffffe, 0xffffffff)`, `(0, 0x80000000)`, `(1, 0)`; inline `-1`; 32-bit literal `0x80000001`; explicit `SCC`
  preservation (`0` and `1`) across count-then-scan sequences; overlapping `s22 = ff1(s[22:23])`.

Measured on 27.11.2026, `main` `428c08c1` plus this change, Windows 10.0.26100 (`dev` preset): `native_salu_bitscan64_controls`
passed; `native_salu_bitscan64_hardware` skipped (`77`, Quadro M1000M does not support shader model 6.6 with wave32, so
readback is not measured).
