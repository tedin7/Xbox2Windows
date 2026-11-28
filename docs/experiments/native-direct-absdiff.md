# s_absdiff_i32 in the direct C++ compute path

The direct compute frontend (`src/gfx1013_direct_compute.inl`) used to refuse `s_absdiff_i32` with
`direct DXIL lowering for s_absdiff_i32`. It now lowers it as the RDNA 2 ISA describes it:

- D = S0 - S1, as a 32-bit difference that wraps.
- If D is negative, D = -D. So a difference of 0x80000000 stays 0x80000000.
- SCC = D != 0, written by the shared scalar tail as for the other SOP2 integer operations.

The rules are the same as those of `gfx2hlsl.py` (`asuint(abs(asint({0} - {1})))`, SCC `$ != 0`), `tools/pixel_aot.py` and the
HLSL path in `src/gfx1013_compute_native.cpp`.

## Fixture

`tests/native_absdiff_check.py controls|hardware` (`ctest --preset dev --no-tests=error -R '^native_absdiff_' -V`).

- **Encoding:** the program is built with the encoding helpers of the other native fixtures (`scalar`, `sop1`, `vop2`, `vmove`).
  Each word was checked with `tools/disasm_rdna2.py --hex`.
- **Inputs:** two groups of 32 lanes. `s24 = group * 0x01000193`. Six register pairs are XORed with `s24` at run time, so group 1
  sees different values, and equal pairs stay equal:
  - (5, 3), (3, 5);
  - (0x80000000, 0) and (0x7fffffff, 0xffffffff), whose differences wrap to 0x80000000;
  - (0x80000000, 1), whose difference wraps to a positive value;
  - (7, 7).
- **Constant operands:** two more cases take S1 as the literal 0xfffffff0 and as the inline constant -1.
- **What is read back:** each lane stores the result and SCC (through `s_cselect_b32`) of each of the eight cases.
  - Before the equal pair, SCC is 1, so a lowering that left SCC unchanged would show there.
- **Controls:**
  - The program must compile.
  - Both wrap results and the zero result must be present in the expected words.
  - The reference must differ from each of these mutated references:
    - a difference that does not wrap;
    - the unsigned distance;
    - 0x80000000 saturated to 0x7fffffff;
    - the difference without the absolute value;
    - SCC left unchanged.

## Measured

Measured on 27.11.2026: `main` `1ec2c938` plus this change, Windows 11 Pro 10.0.26200, NVIDIA GeForce RTX 5090 (driver
32.0.16.1088), dev preset with the debug layer on.

- **Controls:** printed `compiled; the unwrapped, unsigned, saturated, signed-difference and SCC-unchanged references each differ`.
- **Readback:** printed `s_absdiff_i32: 1024 words of two groups and the guards matched`.
- **Negative controls:** each mutation below was built and run.

  | Mutation | Result |
  |---|---|
  | Lowering removed | Both tests fail (`direct DXIL lowering for s_absdiff_i32`) |
  | No absolute value | The readback fails (`0xfffffffe` for (3, 5), expected 2) |
  | SCC not written | The readback fails at the SCC words |

- **Not run:** the readback has not been run on WARP or on an AMD GPU.
- **Local start-up capture:** `compile_dxil --refusals` was run on one capture of 2,689 compute requests (counts only).
  - At `main` `1ec2c938`, 399 compile and 7 stop first at `s_absdiff_i32`.
  - With this change, 1 of the 7 compiles, so the total that compile goes from 399 to 400.
  - The other 6 stop at a later refusal: 3 at `direct descriptor address outside bound table`, 2 at
    `direct image opcode or descriptor modifiers` and 1 at `unbound numeric SGPR sN`.
