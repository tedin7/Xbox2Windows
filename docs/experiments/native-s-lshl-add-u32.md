# Scalar shift-and-add (`s_lshl{1,2,3,4}_add_u32`) in direct compute

The native compute frontend translates the gfx1013 SOP2 instructions
`s_lshl1_add_u32` (`0x2e`), `s_lshl2_add_u32` (`0x2f`), `s_lshl3_add_u32`
(`0x30`), and `s_lshl4_add_u32` (`0x31`). Each computes
`temp = ((uint64)S0 << k) + (uint64)S1`, writes `temp[31:0]` to `D`, and sets
`SCC = (temp >= 2^32)` (true when either `S0 >> (32 - k) != 0` or the 32-bit
addition carries).

Section 12.1 of the [AMD RDNA 2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
defines the guest semantics.

## Reproduce

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_s_lshl_add_u32_' -V
```

The compile controls validate the generated DXIL and verify that omitting
shift-out overflow diverges from the reference. Two source-built dispatches
exercise eight input pairs across all four shift amounts (`k = 1..4`),
verifying both `D` and `SCC` (including shift-out-only overflow, addition
carry, and exact `0xFFFFFFFF` / `0x100000000` boundaries; 2,048 result/SCC
words and 16 guard words per dispatch). The native replay requires shader
model 6.6 and wave32.
