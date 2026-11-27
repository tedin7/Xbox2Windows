# Packed 16-bit integer add/sub in direct compute

The native compute frontend translates the gfx1013 VOP3P instructions
`v_pk_add_i16` (`0x02`), `v_pk_sub_i16` (`0x03`), `v_pk_add_u16` (`0x0a`),
and `v_pk_sub_u16` (`0x0b`). Each instruction adds or subtracts two
independent 16-bit lanes selected by `op_sel[1:0]` (low output half) and
`op_sel_hi[1:0]` (high output half).

Without `clamp`, each lane wraps modulo 2^16. With `clamp`, signed forms
saturate each lane to `[-32768, 32767]` and unsigned forms saturate each lane
to `[0, 65535]`. `neg_lo`, `neg_hi`, `op_sel[2]` and nonzero `src2` are
refused; `op_sel_hi[2]` (bit 14, LLVM's default on 2-source VOP3P) is ignored.

Section 12.10 of the [AMD RDNA 2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
defines the guest semantics.

## Reproduce

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_pk_add_sub_16_' -V
```

The compile controls validate the generated DXIL and refuse four noncanonical
modifier and reserved-bit encodings (`op_sel[2]`, `neg_hi`, `neg_lo`, nonzero
`src2`). Three source-built dispatches exercise eight input patterns across all
four opcodes, default and swizzled/broadcast `op_sel`/`op_sel_hi` modes, signed
and unsigned saturation in both directions, a sparse `EXEC` mask, and a
no-clamp negative control (2,048 result words and 16 guard words per dispatch).
The native replay requires shader model 6.6 and wave32.
