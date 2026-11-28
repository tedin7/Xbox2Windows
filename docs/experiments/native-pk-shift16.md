# Packed 16-bit integer shifts in direct compute

The native compute frontend translates the gfx1013 VOP3P instructions
`v_pk_lshlrev_b16` (`0x04`), `v_pk_lshrrev_b16` (`0x05`), and
`v_pk_ashrrev_i16` (`0x06`). Each instruction shifts two independent 16-bit
lanes of `src1` by the low 4 bits (`& 15`) of the corresponding 16-bit lanes
of `src0`, selected by `op_sel[1:0]` (low output half) and `op_sel_hi[1:0]`
(high output half).

`v_pk_lshlrev_b16` masks each shifted lane to 16 bits so bits shifted above
bit 15 do not bleed into the high half. `v_pk_ashrrev_i16` places each
selected 16-bit half in bits `[31:16]` and performs a signed arithmetic right
shift by `16 + (amt & 15)` (always in `[16, 31]`). `clamp`, `neg_lo`,
`neg_hi`, `op_sel[2]` and nonzero `src2` are refused; `op_sel_hi[2]` (bit 14,
LLVM's default on 2-source VOP3P) is ignored.

Section 12.10 of the [AMD RDNA 2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
defines the guest semantics.

## Reproduce

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_pk_shift16_' -V
```

The compile controls validate the generated DXIL and refuse five noncanonical
modifier and reserved-bit encodings (`clamp`, `op_sel[2]`, `neg_hi`, `neg_lo`,
nonzero `src2`). Two source-built dispatches exercise eight input patterns across
all three opcodes, default and swizzled/broadcast `op_sel`/`op_sel_hi` modes,
shift counts above 15, sign extension, and a sparse `EXEC` mask (2,048 result
words and 16 guard words per dispatch). The native replay requires shader model
6.6 and wave32.
