# v_ldexp_f32 in the direct C++ compute path (#3013)

The direct compute frontend refused `v_ldexp_f32` with `direct DXIL lowering for v_ldexp_f32`. It now lowers it with
the semantics of `tools/pixel_aot.py`'s `ldexp32`:

- the result is S0 scaled exactly by 2^S1;
- zero, Inf and NaN pass unchanged, including the NaN payload;
- a denormal S0 is normalized rather than flushed;
- S1 is a signed integer clamped to [-512, 512];
- a result above the range becomes a signed Inf; a result below the normal range is a denormal rounded to nearest
  even, or a signed zero at or below 2^-150 (2^-150 itself is a tie that rounds to even zero);
- `clamp` saturates the result with integer tests: NaN and negative values (also -0) give +0, values from 1.0 give
  1.0, and a denormal stays. `pixel_aot` uses HLSL `saturate`, which agrees except that it may flush a denormal or
  keep -0 depending on the float mode; the integer form leaves nothing to that mode.

Every step is an integer bit operation, so no float operation can flush or round. A denormal mantissa is normalized
by five conditional left shifts (16, 8, 4, 2, 1), not by a bit-scan intrinsic. Every shift amount stays in 1..24, also
in lanes whose result a later select discards. Abs and neg on S0 are bit operations.
Abs or neg on S1 or S2, a nonzero src2 field, omod, op_sel, SDWA and DPP forms are refused.

## Fixture

`tests/native_ldexp_check.py controls|hardware` (`ctest --preset dev --no-tests=error -R '^native_ldexp_' -V`). The
VOP3 words were assembled with LLVM 15.0.7 (`clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm
-show-encoding`). Eight cases set S0 and S1 by `v_mov` literals. Every lane of two groups stores `v_ldexp_f32` of S0,
of -|S0| and the clamp form. The cases cover:

- an ordinary scale;
- a denormal input;
- the two round-to-even ties (1.5 * 2^-149 and 2^-150);
- a NaN payload;
- -0 with a huge exponent;
- a saturated result;
- an overflow to -Inf.

The controls compile the program. They check the module for integer shifts and for no `fmul` and no DXIL unary float
operation. They check that neg on S1, abs on S1, omod, an op_sel bit and a nonzero src2 field are refused (the SDWA
and DPP refusals are not exercised: LLVM 15 has no such encodings of `v_ldexp_f32`). They also check that five
references each differ from the expected words: flushed denormals, truncation, rounding half away from zero, an
unclamped S1 that wraps, and a float clamp that flushes denormals and keeps -0.

Measured on 27.11.2026, `main` `36d2cbeb` plus this change, Windows 10.0.26100. The controls passed. The
readback reported a skip (77) because the default adapter does not report shader model 6.6 with wave32. The readback
is not measured. The log of this run is `out/logs/ldexp_ctest.log` (not committed).
