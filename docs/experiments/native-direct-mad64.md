# v_mad_u64_u32 and v_mad_i64_i32 in the direct C++ compute path (#3000)

The direct compute frontend now lowers the two 64-bit multiply-adds. Before, it refused them with
`direct DXIL lowering for <name>`. The contract is `gfx2hlsl.py` `mad64`: D = S0 * S1 + S2 modulo 2^64, with
32-bit factors and a 64-bit S2. The factors are zero-extended for `v_mad_u64_u32` and sign-extended for
`v_mad_i64_i32`.

- `v_mad_u64_u32` writes the carry-out of the 64-bit add (sum < S2, unsigned) to its VOP3B SDST, which is VCC or
  an SGPR. This uses the lane-mask path of the other carry instructions (`write_ballot`). `carry_destination` in
  `src/gfx1013_compute_native.cpp` names its SDST so that the symbolic state forgets the old SGPR value.
- `v_mad_i64_i32` is lowered only with the null SDST. As in `gfx2hlsl.py`, the signed carry-out is not
  established. Any other SDST is refused with `v_mad_i64_i32 carry-out destination`.
- S2 must be a VGPR pair or an aligned SGPR pair. Inline and literal 64-bit addends are refused, a narrower
  contract than `gfx2hlsl.py`'s `i64_words`.
- In VOP3B, the ABS field holds SDST, so only NEG, clamp and the output modifier are refused among the modifiers.
  SDWA, DPP, pixel stages and graphics inputs are refused as for the carry adds. An inline float factor is refused,
  as in `gfx2hlsl.py`.
- The unsigned form with the null SDST is refused by the symbolic transfer (`unsupported carry mask destination`),
  also as in `gfx2hlsl.py`, which takes only VCC or an SGPR there.

## Fixture

`tests/native_mad64_check.py controls|hardware` (`ctest --preset dev --no-tests=error -R '^native_mad64_' -V`).
The words were assembled with LLVM 15.0.7 (`clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm
-show-encoding`).

- The dispatch has two groups of 32 lanes. With n = lane + 32 * group:
  - S0 = mul_u32_u24(n, 0x0f0f0f) ^ 0xffff0000
  - S1 = mul_u32_u24(n, 0x123457) ^ 0xfff00000
  - S2 = {((n & 1) << 31) | 0x10, mul_u32_u24(n, 0x777777) ^ 0x89abcdef}

  The products lie just below 2^64, so the unsigned sum carries exactly for the odd lanes (mask 0xaaaaaaaa). Each
  lane stores the unsigned and signed results and the carry mask copied from s30.
- The controls:
  - check the module for two `mul i64`, a sign extension and the unsigned carry compare;
  - refuse `v_mad_i64_i32` with an SGPR carry-out, `v_mad_u64_u32` with the null SDST, and an inline float factor;
  - check that both groups have mixed carries;
  - check that three mutated references each differ from the expected words: high word dropped, signed factors
    zero-extended, and no carry.

Measured on 27.11.2026, `cpp-migration` `e0d83759` plus this change, Windows 10.0.26100.9711. The controls passed.
The readback reported a skip (77) on the NVIDIA Quadro M1000M (driver 32.0.15.7371, debug layer unavailable): it
needs shader model 6.6 with wave32. The readback is not measured.
