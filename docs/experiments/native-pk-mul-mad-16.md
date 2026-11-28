# Packed 16-bit integer multiply and multiply-add in direct compute

The native compute frontend translates the gfx1013 VOP3P instructions
`v_pk_mul_lo_u16` (`0x01`), `v_pk_mad_u16` (`0x09`), and `v_pk_mad_i16`
(`0x00`). Each instruction operates on two independent 16-bit lanes selected
by `op_sel` (low output half) and `op_sel_hi` (high output half).

`v_pk_mul_lo_u16` wraps each 16-bit product modulo 2^16 and refuses `clamp`,
`op_sel[2]`, and nonzero `src2` (ignoring LLVM's default 2-source
`op_sel_hi[2]` bit 14). `v_pk_mad_u16` and `v_pk_mad_i16` wrap modulo 2^16
when unclamped, or saturate each lane to `[0, 65535]` / `[-32768, 32767]` when
`clamp` is set (32-bit intermediates never overflow). `neg_lo` and `neg_hi`
are refused on all three opcodes.

Section 12.10 of the [AMD RDNA 2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
defines the guest semantics.

## Reproduce

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_pk_mul_mad_16_' -V
```

The compile controls validate the generated DXIL and refuse five noncanonical
modifier and reserved-bit encodings (`clamp`, `op_sel[2]`, and nonzero `src2`
on `v_pk_mul_lo_u16`; `neg_hi` and `neg_lo` on `v_pk_mad_i16`). Three
source-built dispatches exercise eight input patterns across all three opcodes,
2-source and 3-source `op_sel`/`op_sel_hi` swizzles, signed and unsigned
saturation, a sparse `EXEC` mask, and a no-clamp negative control (2,048 result
words and 16 guard words per dispatch). The native replay requires shader model
6.6 and wave32.
