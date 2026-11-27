# Packed byte dot products in direct compute

The native compute frontend translates the gfx1013 VOP3P instructions
`v_dot4_i32_i8` and `v_dot4_u32_u8`. Each instruction multiplies corresponding
bytes in two packed words and adds their four products to a 32-bit accumulator.
The byte and accumulator signedness follows the opcode. Without `clamp`, the
result wraps modulo 2³². With `clamp`, the final sum saturates to the signed or
unsigned 32-bit range.

The DXIL operation is `Dot4AddI8Packed` or `Dot4AddU8Packed`. The clamped path
uses a zero accumulator for the packed dot because the four products alone fit
in 32 bits. It then detects overflow on the addition of the guest accumulator.
The supported encoding uses the canonical VOP3P operand selectors and no
negation modifiers.

Section 12.10 of the AMD guide defines the guest operands and accumulator.
The DXIL opcode definitions specify the corresponding packed operations.
[AMD RDNA 2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
[DXIL opcode definitions](https://github.com/microsoft/DirectXShaderCompiler/blob/main/include/dxc/DXIL/DxilConstants.h)

## Reproduce

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_packed_dot4_' -V
```

The compile controls check the signed and unsigned DXIL instructions and
refuse noncanonical selectors and negation. Two source-built dispatches use
eight input patterns, both opcodes and both clamp modes. They exercise signed
overflow in both directions, unsigned carry, the exact saturation boundaries
and a sparse EXEC mask. A third dispatch drops all clamp bits and verifies
that the affected words change to the wrapping results. Each dispatch
compares 2,048 words and 16 guards. The native replay requires shader model
6.6 and wave32.
