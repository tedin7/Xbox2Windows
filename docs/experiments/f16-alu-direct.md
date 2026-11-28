# Binary16 arithmetic in the direct compute path

The direct C++ compute frontend supports VOP2 and VOP3 `v_add_f16`, `v_sub_f16`, `v_subrev_f16` and `v_mul_f16`, plus VOP1 and VOP3 `v_rcp_f16`, `v_sqrt_f16`, `v_rsq_f16` and `v_exp_f16`. All eight operations support SDWA. Graphics programs keep the generic direct-lowering refusal.

## Semantics

- Each source is its low 16 bits by default. An inline float uses the binary16 rounding of that f32 value. An integer inline or a literal uses its low 16 bits. SDWA accepts WORD_0, WORD_1 or DWORD (low half).
- abs clears bit 15 of the half, then neg toggles it.
- The operation is an f32 add, sub or mul. An f32 value has a 24-bit significand, which covers a correctly rounded binary16 add, sub or mul.
- Reciprocal computes `1.0 / x` in f32. The other unary operations use DXIL Sqrt (24), Rsqrt (25) and Exp (21, base two).
- Reciprocal exchanges signed zeros and signed infinities. Square root preserves signed zero and produces NaN for negative nonzero inputs. Exp returns 1 for either signed zero, infinity at and above 16, positive zero at and below -25, and the canonical quiet NaN for NaN inputs. The emitter handles the exact underflow tie at -25 explicitly.
- The result is rounded with the same integer round-to-nearest-even conversion `v_cvt_f16_f32` uses. The low 16 bits are written and the high half is preserved. SDWA writes the selected half and pads or preserves the other half.
- Rsq maps +0 to +infinity, -0 to -infinity and +infinity to +0. It returns a quiet NaN for NaN and negative nonzero inputs, including -infinity.
- The emitter selects these results using the binary16 source bits after abs and neg. D3D Rsqrt does not guarantee -infinity for -0, so the rsq zero result takes its sign directly from the source.
- `v_subrev_f16` subtracts the first source from the second.
- DPP16 VOP2 permutes the full src0 VGPR word before selecting its low half and honors FI, BOUND_CTRL, and row and bank write masks.

## Exp and rsq accuracy

The Shader Model 4 assembly references specify a maximum relative error of 2^-21 for both host operations. For a normal f32 result this is less than eight f32 ULP. Every binary16 exp rounding midpoint is in the normal f32 range. [Microsoft Learn exp](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/exp--sm4---asm-) and [Microsoft Learn rsq](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/rsq--sm4---asm-).

Exp targets RDNA 2's 0.5 ULP by selecting the correctly rounded half for the 70 non-tie inputs within that window and resolving the exact tie separately. The [90-digit generator](../../tests/f16_exp_midpoints.py) enumerates the binary16 inputs and the model check compares its entries with the C++ array. The IR and GPU comparisons require exact exp bits, including the canonical NaN, the tie at -25, and all 70 inputs in both VOP1 and VOP3. The table covers 0x11c4, 0x11c5 and 0x11c6. Modifier cases exercise correction after source selection and modification.

For positive finite rsq results, a relative error of 2^-21 contributes at most 2^-10 of a binary16 ULP. Integer RNE adds at most 0.5 ULP in the rounding interval, for a total of at most 0.5 + 2^-10 < 0.51 ULP. The [reference model](../../tests/f16_packed_model.py) compares squared rational endpoints against the exact input to locate the result. It accepts either adjoining half only when the exact result is closer than eight f32 ULP to their midpoint. Both plain encodings and SDWA cover 0x3bfe, which exercises the alternate result.

The [plain-form fixture](../../tests/native_f16_alu_check.py) covers signed zeros and infinities, quiet and signaling NaNs, negative finite values, subnormals, 1, powers of two and general values. Its IR interpreter models Rsqrt returning +infinity for -0 to exercise the sign fix. The [SDWA fixture](../../tests/native_sdwa_f16_check.py) exercises source selection and modifiers, destination padding and preservation, and negation of +0 before rsq.

## Refused forms

The SDWA operand profile is covered by the [SDWA experiment](native-sdwa-f16.md).

Each of these is a named diagnostic:

- DPP8
- clamp or omod
- a high-half `op_sel` (instruction bits 11 through 14)
- a binary16 rounding mode other than round-to-nearest-even, an f16 denormal mode other than keep on input and output, or `FP16_OVFL`
- `s_setreg_b32` of those MODE bits, because the source is a register

`s_round_mode` and `s_denorm_mode` that leave the f16 fields at that contract, and that leave the f32 fields unchanged, are accepted. A write that changes an f32 rounding or denormal field keeps the previous control-instruction refusal. A guest branch together with a MODE write makes the contract unprovable and is refused with the rounding diagnostic.

For the plain scalar-half forms, fma, fmac, fmamk, fmaak, min, max, ldexp, the min3/max3/med3 forms, div_fixup and the VOP1 ops other than rcp, sqrt, exp and rsq stay on the generic direct-lowering diagnostic. The fixture checks `v_fmac_f16`, `v_min_f16` and `v_log_f16`. The host Log absolute error allowance of 2^-21 near an input of 1 can exceed 0.51 ULP of the small binary16 result, so log needs a tighter bound or a bounded correction before lowering. [Microsoft Learn log](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/log--sm4---asm-).

## Linux

Python 3.12.3. `compile_dxil` is created only in the `WIN32` section of `CMakeLists.txt`, so this host did not build that target. The IR dumper was compiled from `src/gfx1013_compute_native.cpp`, `src/gfx1013_isa.cpp`, `src/guest_shader_container.cpp`, `src/gfx1013_compute_ir.cpp` and `tests/native_compute_ir_dump.cpp` with g++ 13.3.0 and a local stub of `d3d12.h` and `wrl/client.h`. That stub is not in the tree.

```
python3 -B tests/native_f16_alu_check.py model
python3 -B tests/native_f16_alu_check.py ir <native_compute_ir_dump>
```

- Before unary coverage was added, `model` checked 82 results, including controls for a sub/subrev swap and a zeroed upper half.
- Before unary coverage was added, `ir` reported 82 results and 20 refusals. The module used `legacyF16ToF32`, `fadd`, `fsub` and `fmul`, and the integer round contained the constant `939524096`.

`native_f16_alu_hardware` was not run. It is registered with skip code 77, the same code `native_store_replay` returns when the device cannot run a wave32 shader. WARP is expected to take that skip.

## Windows

Pending. On the Windows test machine:

```
cmake --preset dev
cmake --build --preset dev --target compile_dxil offscreen_check
ctest --preset dev --no-tests=error -R "^native_f16_alu_(ir|hardware)$" -V -j 1
```
