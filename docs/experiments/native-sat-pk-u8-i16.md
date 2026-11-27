# Packed signed-16 to saturated-u8 conversion (`v_sat_pk_u8_i16`) in direct compute

The native compute frontend translates both the VOP1 (`0x62`) and VOP3A
(`0x1e2`) encodings of gfx1013 `v_sat_pk_u8_i16`. It sign-extends both 16-bit
halves of `src0`, clamps each to `[0, 255]`, packs the low and high results
into `D[7:0]` and `D[15:8]`, and zeroes `D[31:16]`.

`clamp`, `omod`, `abs`, `neg`, `op_sel`, SDWA, DPP, and nonzero unused
`src1`/`src2` fields on the VOP3A encoding are refused.

Section 12.6 of the [AMD RDNA 2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
defines the guest semantics.

## Reproduce

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_sat_pk_u8_i16_' -V
```

The compile controls validate the generated DXIL and refuse six VOP3A
modifier/unused-source encodings (`clamp`, `op_sel`, `abs`, `neg`, nonzero
`src1`, nonzero `src2`). Two source-built dispatches exercise 16 input patterns
across both VOP1 and VOP3A encodings, verifying upper-half zeroing (`D[31:16] = 0`)
and a sparse `EXEC` mask (2,048 result words and 16 guard words per dispatch).
The native replay requires shader model 6.6 and wave32.
