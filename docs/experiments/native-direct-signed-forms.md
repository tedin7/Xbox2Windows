# v_ashrrev_i64 and v_mad_i32_i24 in the direct C++ compute path (#2991)

The direct compute frontend (`src/gfx1013_direct_compute.inl`) lowered `v_lshrrev_b64`, `v_lshlrev_b64` and
`v_mad_u32_u24`, but refused their signed counterparts with `direct DXIL lowering for <name>`. It now lowers them:

- `v_ashrrev_i64`: an arithmetic `ashr i64` of S1 by S0[5:0]. The shifted value takes the operands of the
  logical shifts: a VGPR or SGPR pair, an inline integer sign-extended to 64 bits, or a 32-bit literal. Following
  `gfx2hlsl.py` `i64_words(signed=True)`, the literal is sign-extended for this form, where the logical shifts
  zero-extend it.
- `v_mad_i32_i24`: sext(S0[23:0]) * sext(S1[23:0]) + S2 modulo 2^32, the `gfx2hlsl.py` VOP3-table expression.

## Fixture

`tests/native_signed_forms_check.py controls|hardware`
(`ctest --preset dev --no-tests=error -R '^native_signed_forms_' -V`). The words were assembled with LLVM 15.0.7
(`clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm -show-encoding`).

- The controls check the modules with `dxc -dumpbin`: exactly one `ashr i64` per shift program, and an arithmetic
  shift right with `mul i32` for the multiply-add.
- The shift dispatches reuse `native_shift64_check.compute_program` with the `v_ashrrev_i64` opcode (VOP3 0x301).
  They cover shift amounts 0, 1, 31, 32, 33 and 63 and six sources: inline -1, -16, 64 and 1, the literal 0x80000001
  and the negative SGPR pair 0x89abcdef:01234567. References with a zero-extended literal or a logical shift must
  differ from the expected words.
- The multiply-add dispatch uses, with n = lane + 32 * group:
  - S0 = mul_u32_u24(n, 0x05f3d1) ^ 0xa5800000
  - S1 = mul_u32_u24(n, 0x0371) ^ 0x7f9abcde
  - S2 = mul_u32_u24(n, 0x1001)

  Bit 23 is set in many factors and the bits above it are not zero. References with unsigned or unmasked factors
  must differ.

Measured on 27.11.2026, `cpp-migration` `4ff5e36d` plus this change, Windows 10.0.26100.9711. The controls compiled all
programs and printed `compiled six v_ashrrev_i64 sources and v_mad_i32_i24; the zero-extended-literal, logical,
unsigned and unmasked references each differ`. `native_shift64_controls` still passes. The readback printed `requires
shader model 6.6 with wave32` and reported a skip (77) on the NVIDIA Quadro M1000M (driver 32.0.15.7371, debug
layer unavailable). The readback is not measured.
