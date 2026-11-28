# Signed and saturating 32-bit add/sub in the direct C++ compute path (#3009)

The direct compute frontend refused the VOP3-only `v_add_nc_i32` and `v_sub_nc_i32` with `direct DXIL lowering for
<name>`, and the `clamp` forms of `v_add_nc_u32`, `v_sub_nc_u32` and `v_subrev_nc_u32` with `direct ALU operand
modifiers`. It now lowers them:

- without `clamp`, the signed forms wrap modulo 2^32 like the unsigned ones;
- with `clamp`, `v_*_nc_i32` saturates to the i32 range (an overflow saturates toward the sign of S0) and
  `v_*_nc_u32` saturates to 0 or 0xffffffff.

LLVM 15.0.7 selects exactly these clamp forms for `llvm.sadd.sat.i32`, `llvm.ssub.sat.i32`, `llvm.uadd.sat.i32` and
`llvm.usub.sat.i32` (`clang -nogpulib -target amdgcn-- -mcpu=gfx1013 -O2 -S`). `tools/startup/gfx2hlsl.py` has no
lowering for these forms. `tools/compute_store_aot.py` lowers the unclamped signed forms the same way (wrap) and
refuses clamp (`tests/compute_store_check.py`, `vadd_nc_i32_controls`). Overflow is detected with 32-bit sign tests, so the module needs no 64-bit integers.
Source negation or abs, omod, op_sel, SDWA and DPP forms are refused.

## Fixture

`tests/native_add_sub_sat_check.py controls|hardware` (`ctest --preset dev --no-tests=error -R '^native_add_sub_sat_'
-V`). The VOP3 words were assembled with LLVM 15.0.7 (`clang -cc1as -triple amdgcn -target-cpu gfx1013 -filetype asm
-show-encoding`). Four cases set S0 and S1 by `v_mov` literals at the i32 and u32 edges: 0x7fffffff + 1,
0x80000000 - 1, -1 + 0x80000000 and 0x7fffffff - (-1). Every lane of two groups stores the seven results of each case.

The controls compile the program and check the module for the add, sub and select. They check that a neg modifier on
S0 is refused. They also check that three references each differ from the expected words: unclamped, signedness
swapped, saturation toward the sign of S1, and `v_subrev_nc_u32` without the operand swap. The log of this run is
`out/logs/addsub_ctest.log` (not committed).

Measured on 27.11.2026, `cpp-migration` `e0d83759` plus this change, Windows 10.0.26100, with an NVIDIA Quadro M1000M
(driver 32.0.15.7371, debug layer unavailable). The controls passed. The readback reported a skip (77) because it needs
shader model 6.6 with wave32. The readback is not measured.
