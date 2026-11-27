# Packed 4-bit nibble dot products in direct compute

The native compute frontend translates the gfx1013 VOP3P instructions
`v_dot8_i32_i4` (`0x18`) and `v_dot8_u32_u4` (`0x19`). Each instruction
multiplies corresponding 4-bit nibbles in two packed 32-bit words and adds
their eight products to a 32-bit accumulator. The nibble and accumulator
signedness follows the opcode. Without `clamp`, the result wraps modulo 2^32.
With `clamp`, the final sum saturates to the signed or unsigned 32-bit range.

Because eight 4-bit products sum to `[-448, 512]` (signed) or `[0, 1800]`
(unsigned), the dot product itself always fits in 32 bits; only the final
addition with the guest accumulator can overflow and saturate. The supported
encoding requires canonical VOP3P operand selectors (`OP_SEL = 0`,
`OP_SEL_HI = [1, 1, 1]`) and zero negation modifiers (`NEG_LO = 0`,
`NEG_HI = 0`).

Section 12.10 of the [AMD RDNA 2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
defines the guest operands and accumulator semantics.

## Reproduce

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_packed_dot8_' -V
```

The compile controls validate the generated DXIL and refuse five noncanonical
selector and negation encodings. Two source-built dispatches exercise eight
input patterns, both opcodes, both clamp modes, signed overflow in both
directions, unsigned carry, exact saturation boundaries and a sparse `EXEC`
mask. A third dispatch drops all `clamp` bits and confirms that the affected
words change to the wrapping results. Each dispatch compares 2,048 words and
16 guard words. The native replay requires shader model 6.6 and wave32.
