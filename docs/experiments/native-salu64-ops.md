# 64-bit scalar operations in the direct C++ compute path

Issue #2957 adds `s_bfm_b64`, `s_cselect_b64` and `s_not_b64`.
`s_bfm_b64` uses the low six bits of each 32-bit source as its width and
offset, writes both words of an aligned SGPR pair, and preserves SCC.
`s_cselect_b64` selects both words using SCC and preserves SCC.
`s_not_b64` complements both words and sets SCC when either result word
is nonzero. Both take aligned SGPR pairs or constants, with sign extension
for inline negatives and zero extension for other integers. They can
write VCC as a pair. `s_bfm_b64` and `s_not_b64` may also write EXEC, as both
words, in wave32 and wave64. `s_cselect_b64` still refuses EXEC destinations.
`s_not_b64` and `s_cselect_b64` also take VCC and EXEC as 64-bit sources
(`tests/native_s_not_b64_exec_check.py`, `ctest --preset dev --no-tests=error -R '^native_s_not_b64_exec_'`).

Direct compute also accepts `s_bitreplicate_b64_b32 exec, imm`, writing both EXEC words and preserving SCC. Wave32 lane activity uses only `exec_lo`.

## Verification

`native_salu64_controls` checks that compiling `s_cselect_b64 exec, -1, 0`
fails. `native_salu64_hardware` checks both result words, SCC, and the
buffer guards across two 32-thread groups. It covers mask widths 0, 1, 31,
32, 33, 63 and 64, including the captured width and offset of 32, with
the raw width 64 wrapping to 0. The select cases cover both SCC values,
an SGPR pair, inline -1 and the literal `0x80000001`. The complement
cases include zero output and output with only the high word nonzero.
Hardware without shader model 6.6 and wave32 support returns skip code 77.
`native_wave64_exec_salu_hardware` covers `s_bfm_b64 exec, 16, 0` and `exec, 32, 32` in wave64, 64- and
128-thread groups (the wave32 EXEC destination is accepted but not run), checking which lanes stay active through a per-lane store.

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_salu64_(controls|hardware)$'
ctest --preset dev --no-tests=error -R '^native_wave64_exec_salu_hardware$'
```

Results: on an NVIDIA GeForce RTX 5080 (Windows 11, DXC from the Windows SDK 10.0.26100) all fifteen cases
matched their 64-bit words and SCC in both workgroups, with the guards intact, and the control refused an EXEC
destination. A 32-bit `s_bfm_b64`, a zero-extended `-1` in `s_cselect_b64` and an `s_not_b64` SCC taken
from the low word only each fail the hardware check.
