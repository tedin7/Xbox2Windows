# Binary16 add, sub, subrev and mul in the direct compute path

Lowers VOP2 and VOP3 `v_add_f16`, `v_sub_f16`, `v_subrev_f16` and `v_mul_f16` in the direct C++ compute frontend ([`src/gfx1013_direct_compute.inl`](../../src/gfx1013_direct_compute.inl)). The handler returns without lowering when the stage is graphics, so a graphics program keeps the generic direct-lowering refusal. A graphics container was not dumped on this host.

## Semantics

- Each source is its low 16 bits. An inline float uses the binary16 rounding of that f32 value. An integer inline or a literal uses its low 16 bits.
- abs clears bit 15 of the half, then neg toggles it.
- The operation is an f32 add, sub or mul. An f32 value has a 24-bit significand, which covers a correctly rounded binary16 add, sub or mul.
- The result is rounded with the same integer round-to-nearest-even conversion `v_cvt_f16_f32` uses. The low 16 bits are written and the high half is zero.
- `v_subrev_f16` subtracts the first source from the second.

The bit-exact reference is `half_op` in [`tests/f16_packed_model.py`](../../tests/f16_packed_model.py). NaN results match by class. The check is [`tests/native_f16_alu_check.py`](../../tests/native_f16_alu_check.py).

## Refused forms

The SDWA operand profile is covered by the [SDWA experiment](native-sdwa-f16.md).

Each of these is a named diagnostic:

- DPP
- clamp or omod
- a high-half `op_sel` (instruction bits 11 through 14)
- a binary16 rounding mode other than round-to-nearest-even, an f16 denormal mode other than keep on input and output, or `FP16_OVFL`
- `s_setreg_b32` of those MODE bits, because the source is a register

`s_round_mode` and `s_denorm_mode` that leave the f16 fields at that contract, and that leave the f32 fields unchanged, are accepted. A write that changes an f32 rounding or denormal field keeps the previous control-instruction refusal. A guest branch together with a MODE write makes the contract unprovable and is refused with the rounding diagnostic.

For the plain scalar-half forms, fma, fmac, fmamk, fmaak, min, max, ldexp, the min3/max3/med3 forms, div_fixup and VOP1 stay on the generic direct-lowering diagnostic. The fixture checks `v_fmac_f16` and `v_min_f16`. VOP3P arithmetic has a [separate experiment](native-packed-f16.md).

## Linux

Python 3.12.3. `compile_dxil` is created only in the `WIN32` section of `CMakeLists.txt`, so this host did not build that target. The IR dumper was compiled from `src/gfx1013_compute_native.cpp`, `src/gfx1013_isa.cpp`, `src/guest_shader_container.cpp`, `src/gfx1013_compute_ir.cpp` and `tests/native_compute_ir_dump.cpp` with g++ 13.3.0 and a local stub of `d3d12.h` and `wrl/client.h`. That stub is not in the tree.

```
python3 -B tests/native_f16_alu_check.py model
python3 -B tests/native_f16_alu_check.py ir <native_compute_ir_dump>
```

- `model` printed `native_f16_alu model: 82 results match f32-then-round; a sub/subrev swap and a non-zero upper half are rejected`.
- `ir` printed `native_f16_alu IR: 82 results match the model; 20 refusals carry the named diagnostic`. The module calls `legacyF16ToF32`, `fadd`, `fsub` and `fmul`, and the integer round contains the constant `939524096`.

`native_f16_alu_hardware` was not run. It is registered with skip code 77, the same code `native_store_replay` returns when the device cannot run a wave32 shader. WARP is expected to take that skip.

## Windows

Pending. On the Windows test machine:

```
cmake --preset dev
cmake --build --preset dev --target compile_dxil offscreen_check
ctest --preset dev --no-tests=error -R "^native_f16_alu_(ir|hardware)$" -V -j 1
```
