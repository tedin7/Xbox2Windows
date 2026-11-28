# 24-bit integer multiplies (`v_mul_i32_i24`, `v_mul_hi_i32_i24`, `v_mul_hi_u32_u24`) in direct compute

The native compute frontend translates both the VOP2 (`0x09`, `0x0a`, `0x0c`)
and VOP3A (`0x109`, `0x10a`, `0x10c`) encodings of `v_mul_i32_i24`,
`v_mul_hi_i32_i24`, and `v_mul_hi_u32_u24`. Each instruction reads only the
low 24 bits of `src0` and `src1` (sign-extended for `i24`, zero-extended for
`u24`). `v_mul_i32_i24` returns bits `[31:0]` of the product; the two `hi`
forms return bits `[63:32]` of the 48-bit product via an exact 12+12 split in
32-bit integer arithmetic.

`clamp`, `omod`, `abs`, `neg`, `op_sel`, and nonzero unused `src2` on the
VOP3A encoding are refused.

Section 12.7 of the [AMD RDNA 2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
defines the guest semantics.

## Reproduce

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_mul24_' -V
```

The compile controls validate the generated DXIL and refuse five VOP3A
modifier/unused-source encodings (`clamp`, `op_sel`, `abs`, `neg`, nonzero
`src2`). Two source-built dispatches exercise eight input patterns across all
three opcodes, both VOP2 and VOP3A encodings, upper-byte noise, sign
combinations, and a sparse `EXEC` mask (2,048 result words and 16 guard words
per dispatch). The native replay requires shader model 6.6 and wave32.
