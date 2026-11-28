# Sum-of-absolute-differences operations in the direct C++ compute path (#2987)

The direct compute frontend (`src/gfx1013_direct_compute.inl`) now lowers `v_sad_u8`, `v_sad_hi_u8`, `v_sad_u16`,
`v_sad_u32` and `v_msad_u8`, which it refused with `direct DXIL lowering for <name>`. The results are those of
`gfx2hlsl.py`'s `sad_u8`, `sad4_u8`, `sad_u16` and `sad_u32` helpers and of `tools/pixel_aot.py`. Each adds S2 and
wraps modulo 2^32:

- `v_sad_u8`: the sum of the four unsigned byte differences |S0 byte - S1 byte|.
- `v_sad_hi_u8`: that sum shifted left by 16.
- `v_sad_u16`: the two unsigned halfword differences.
- `v_sad_u32`: the unsigned difference of the dwords.
- `v_msad_u8`: like `v_sad_u8`, but a byte whose S1 byte is zero contributes nothing.

Source abs/neg, clamp and output modifiers stay refused, with `direct ALU operand modifiers` as for the other
integer VOP3 forms. The fixture checks clamp.

## Fixture

`tests/native_sad_check.py controls|hardware` (`ctest --preset dev --no-tests=error -R '^native_sad_' -V`). The
words were assembled with LLVM 15.0.7 (`clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm
-show-encoding`).

- The dispatch has two groups of 32 lanes. With n = lane + 32 * group:
  - S0 = mul_u32_u24(n, 0x050311) ^ 0x80f00a33
  - S1 = mul_u32_u24(n, 0x010203) & 0xffff00ff. Byte 1 is always zero, so `v_msad_u8` skips it.
  - S2 = mul_u32_u24(n, 0x1001)

  S0 bytes both above and below the S1 bytes occur. The 320 results and the guard words are compared with a CPU
  reference.
- That reference agrees, for all 64 lanes, with the existing references in `tests/startup_gfx2hlsl_sad_check.py`
  (`ref_sad_u8`, `ref_sad_u16`, `ref_sad_u32`) and `tests/startup_gfx2hlsl_sad_hi_msad_check.py` (`sad4_u8`).
  This was checked once by hand while writing the fixture; the fixture does not import those files.
- The controls compile the program and refuse the `v_sad_u8` form with the clamp bit. They also check that four
  mutated references each differ from the expected words: signed byte differences, `v_msad_u8` without its mask,
  `v_sad_hi_u8` without its shift, and S2 dropped.

Measured on 27.11.2026, `cpp-migration` `4ff5e36d` plus this change, Windows 10.0.26100.9711. The controls printed
`compiled; clamp refused; the signed, unmasked, unshifted and S2-dropped references each differ`. The readback
printed `requires shader model 6.6 with wave32` and reported a skip (77) on the NVIDIA Quadro M1000M (driver
32.0.15.7371, debug layer unavailable). The readback is not measured.
