# Logical 64-bit shift sources in the direct C++ compute path

Issue #2951 covers the shifted value for `v_lshrrev_b64` and `v_lshlrev_b64`.
Inline integers supply their 32-bit value as the low word and all ones as the
high word for negative values. Nonnegative integers and 32-bit literals
supply a zero high word. An even SGPR pair from `s[0:1]` through `s[104:105]`
supplies the low and high words in register order. Both SGPRs must be numeric.
VGPR pairs and inline zero remain accepted. The shift amount uses its low six
bits.

Inline floats, odd or out-of-range SGPR pairs and special registers are
refused with `direct 64-bit shift operand`.

## Verification

`native_shift64_controls` checks that an inline float is refused during
compilation. `native_shift64_hardware` uses source-built programs and checks
both words and the buffer guards across two 32-thread groups against an
independent masked Python integer reference. The fixture shifts inline
`-1`, `-16`, `64` and `1`, literal `0x80000001`, and SGPR pair
`0x01234567_89abcdef` in both directions by 0, 1, 31, 32, 33 and 63.
Hardware without shader model 6.6 and wave32 support returns skip code 77.

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_shift64_(controls|hardware)$'
```

Results: on an NVIDIA GeForce RTX 5080 (Windows 11, DXC from the Windows SDK 10.0.26100) the controls refused the
inline float, and every source matched its 64-bit reference for both directions and all six shift amounts, in both
workgroups, with the guards intact. Zero-extending the inline values, sign-extending the literal and swapping the
SGPR pair's words each fail the hardware check.
