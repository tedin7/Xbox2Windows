# Signed 4-bit offset to binary32 (`v_cvt_off_f32_i4`) in direct compute

The native compute frontend translates both the VOP1 (`0x0e`) and VOP3A
(`0x18e`) encodings of gfx1013 `v_cvt_off_f32_i4`. It sign-extends `src0[3:0]`
to a signed integer in `[-8, +7]` and multiplies by `0.0625` (`1/16`),
producing exact binary32 values in `[-0.5, +0.4375]`. VOP3A output modifiers
(`omod` and `clamp`) apply to the float result.

Integer source modifiers (`abs`, `neg`), `op_sel`, and nonzero unused
`src1`/`src2` fields on the VOP3A encoding are refused.

Section 12.6 of the [AMD RDNA 2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
defines the guest semantics.

## Reproduce

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_cvt_off_f32_i4_' -V
```

The compile controls validate the generated DXIL and refuse five VOP3A
modifier/unused-source encodings (`abs`, `neg`, `op_sel`, nonzero `src1`,
nonzero `src2`). Three source-built dispatches exercise all 16 nibble values
across VOP1, VOP3A, `mul:2`, `clamp`, a sparse `EXEC` mask, and a no-clamp
negative control (2,048 result words and 16 guard words per dispatch). The
native replay requires shader model 6.6 and wave32.
