# Packed halfword dot products in direct compute

The gfx1013 VOP3P instructions `v_dot2_i32_i16` and `v_dot2_u32_u16`
multiply corresponding halfwords and add both products to the 32-bit
accumulator. The opcode determines the signedness of the halfwords and
accumulator. The ordinary result wraps at 32 bits. The clamped form saturates
the exact final sum to the signed or unsigned destination range.
[AMD RDNA 2 ISA, section 12.10](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)

Each product fits in one word. The clamped lowering tracks the carries from
both additions and, for the signed instruction, the sign extensions of the
products and accumulator. This uses only 32-bit DXIL arithmetic. The upper
word determines whether the exact sum fits the destination range. Waiting
until the last addition matters when both signed products are `2^30` and
the accumulator is `-2^31`. The exact result is zero, but clamping the product
sum first would give -1.

## Reproduce

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_packed_dot2_' -V
```

The source-built fixture compares two dispatches with full and sparse EXEC
against a Python integer reference. It covers the signed cancellation,
signed overflow in both directions, unsigned double carry, exact bounds and
guard words. A third dispatch removes the clamp bits and checks the wrapping
readback. The compile controls reject five noncanonical selector or
negation encodings. The native replay requires shader model 6.6 and wave32.
