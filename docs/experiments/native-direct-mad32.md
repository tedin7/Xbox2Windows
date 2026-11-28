# Unfused f32 multiply-adds in the direct C++ compute path (#3005)

The direct compute frontend lowered the fused `v_fma_f32` family (DXIL `FMad`). It refused `v_mad_f32` and
`v_mad_legacy_f32` with `direct DXIL lowering for <name>`, and the VOP2 accumulators `v_mac_f32` and
`v_mac_legacy_f32` had no lowering. It now lowers all four with the semantics of `gfx2hlsl.py`'s `mad32`, which matches
`pixel_aot`. `gfx2hlsl.py` has no `v_mac_legacy_f32` entry; its semantics follow the derivation in
`tests/pixel_aot_check.py` (the `v_mac_legacy_f32` comment):

- denormal inputs are flushed to signed zero;
- the product is rounded on its own; the legacy forms make 0 * anything +0;
- the flushed addend is added (`v_mac_*` add the destination);
- the sum is flushed.

The product and the sum are plain `fmul` and `fadd`. They carry no fast-math flags, so DXIL does not allow them to be
contracted into `FMad`. The flush is an integer mask that keeps the sign bit of a denormal. Source and output
modifiers follow the other f32 VOP3 forms of this path. A `v_mac_*` VOP3 form with abs or neg on the accumulator
(src2) is refused, as in `pixel_aot`.

## Fixture

`tests/native_mad32_check.py controls|hardware` (`ctest --preset dev --no-tests=error -R '^native_mad32_' -V`). The
VOP3 and VOP2 words were assembled with LLVM 15.0.7 (`clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm
-show-encoding`). Six cases set S0, S1 and S2 by `v_mov` literals. Each lane of two groups stores the four results of
every case. The cases cover:

- (1 + 2^-12)^2 - 1, whose separately rounded product gives 2^-11, while fused rounding would give 2^-11 + 2^-24;
- a denormal input;
- 0 * inf, which is NaN for the plain forms and gives 2.0 for the legacy forms;
- -0 * 1 + -0, which gives -0 for the plain forms and +0 for the legacy forms;
- a sum of 2^-127, which is flushed to +0;
- an ordinary value.

The controls compile the program. They check that the module has separate `fmul` and `fadd`, no `FMad` (DXIL tertiary
opcode 46) and no fast-math flags. They also check that three references each differ from the expected words: fused,
unflushed, and legacy treated like plain. In the readback, NaN results compare by class.

Measured on 27.11.2026, `cpp-migration` `e0d83759` plus this change, Windows 10.0.26100.9711. The controls passed. The
readback reported a skip (77) on the NVIDIA Quadro M1000M (driver 32.0.15.7371, debug layer unavailable): it needs
shader model 6.6 with wave32. The readback is not measured, and in particular whether the desktop driver honours the
missing contract flag is not measured.
