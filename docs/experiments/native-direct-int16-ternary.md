# Three-source 16-bit integer VOP3 operations in the direct C++ compute path (#669)

The direct compute frontend (`src/gfx1013_direct_compute.inl`) now lowers the eight three-source 16-bit integer
VOP3 operations that it refused with `direct DXIL lowering for <name>`:
- `v_mad_u16` and `v_mad_i16`
- `v_min3_u16` and `v_min3_i16`
- `v_max3_u16` and `v_max3_i16`
- `v_med3_u16` and `v_med3_i16`

## Semantics

Each operation computes `D[15:0]` from `S0[15:0]`, `S1[15:0]` and `S2[15:0]`, and **`D[31:16]` preserves its
previous value** (measured on RDNA 2 silicon; LLVM `GCNSubtarget::zeroesHigh16BitsOfDest`: "On gfx10, all 16-bit
instructions preserve the high bits").

- `v_mad_u16` and `v_mad_i16`: `(S0[15:0] * S1[15:0] + S2[15:0]) mod 2^16`. As LLVM 15.0.7 notes
  (`VOP3_Profile<VOP_I16_I16_I16_I16, VOP3_CLAMP>` with `Ternary_i16_Pats<mul, add>`), the signed and unsigned
  forms have identical low 16 bits when clamp is not modelled.
- `v_min3_u16`, `v_max3_u16`, `v_med3_u16`: unsigned minimum, maximum and median of the three 16-bit halfwords.
- `v_min3_i16`, `v_max3_i16`, `v_med3_i16`: sign-extended minimum, maximum and median of the three 16-bit halfwords.

## Refusals

`op_sel` (bits 11..14 of VOP3 word 0) would select high halves and stays refused (`unsupported <name> op_sel`).
Modifiers such as `clamp`, `scale`, `neg` and `abs` are refused with `direct ALU operand modifiers` by the path's
existing common rule.

## Fixture

`tests/native_int16_ternary_check.py controls|hardware`
(`ctest --preset dev --no-tests=error -R '^native_int16_ternary_' -V`). The words were assembled with LLVM 15.0.7
(`clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm -show-encoding`).

- **Dispatch:** two groups of 32 lanes (64 lanes total). With G = lane + 32 * group:
  - S0 = mul_u32_u24(G, 0x05f3d1) ^ 0xa5800000
  - S1 = mul_u32_u24(G, 0x000371) ^ 0x7f9abcde
  - S2 = mul_u32_u24(G, 0x001001) ^ 0x80007fff
  Both halves are nonzero, values cross the 16-bit signed boundary (0x8000), 63 of 64 lanes wrap mad modulo $2^{16}$,
  and all permutations of signed and unsigned orders are exercised.
- **Destinations:** destination registers `v10` through `v17` are preloaded with `0xcafebabe`. Each lane writes all
  8 results into the UAV buffer at offset $G \times 32$.
- **Controls:** verify CSO compilation, spot-check definitions, require `op_sel` and `clamp` to be refused, and
  verify that mutated references differ from the expected words:
  - zeroed high half (tests high-half preservation);
  - signed comparisons as unsigned;
  - median computed as minimum;
  - mad computed without addend;
  - mad operand read from high half instead of low half.
- **Hardware:** dispatches the shader through `native_store_replay --direct` and compares all 512 words and the
  sentinel guards against the independent CPU reference.

## Measured

Measured on 27.11.2026, Windows 11 build 26300, AMD Radeon RX 6600M, driver 32.0.21045.11001.

| Check | Result |
|---|---|
| `native_int16_ternary_controls` | pass: compiled; op_sel and clamp refused; definitions spot-checked; all mutants differ |
| `native_int16_ternary_hardware` | pass: eight 16-bit ternary operations: 512 words of two groups and the guards matched, high halves kept |
