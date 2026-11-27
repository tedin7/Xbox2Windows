# 32-bit bit counts and bit scans in the direct C++ compute path

The direct compute frontend (`src/gfx1013_direct_compute.inl`) used to refuse these operations with
`direct DXIL lowering for <name>`. It now lowers them through DXIL `Countbits` (31), `FirstbitLo` (32), `FirstbitHi` (33) and `FirstbitSHi` (34).

| Operation | Result | SCC |
|---|---|---|
| `s_bcnt0_i32_b32`, `s_bcnt1_i32_b32` | number of zero or one bits of S0 | D != 0 |
| `s_ff0_i32_b32`, `s_ff1_i32_b32`, `v_ffbl_b32` | index of the first zero or one bit, counted from bit 0 | unchanged |
| `s_flbit_i32_b32`, `v_ffbh_u32` | position of the first one bit, counted from bit 31 | unchanged |
| `s_flbit_i32`, `v_ffbh_i32` | position of the first bit that differs from the sign bit, counted from bit 31 | unchanged |

- **No qualifying bit:** every scan returns `0xffffffff`. That covers 0 for all of them, and `0xffffffff` for the signed forms and `s_ff0_i32_b32`.
- **Counting direction:** DXIL `FirstbitHi` and `FirstbitSHi` already count from bit 31, as the gfx10 scans do. That is why HLSL's `firstbithigh` compiles to `31 - FirstbitHi`, and why no subtraction is emitted here.
- **Not covered:** the 64-bit scalar forms (`*_b64`). Like the other integer VOP3 forms, source abs/neg, clamp and output modifiers stay refused with `direct ALU operand modifiers`.

## Fixture

`tests/native_bitscan_check.py controls|hardware` (`ctest --preset dev --no-tests=error -R '^native_bitscan_' -V`).

- **Encoding:** the program is built with the encoding helpers of the other native fixtures, `sop1`, `scalar`, `vop2` and `vmove`. Each word was checked with `tools/disasm_rdna2.py --hex`.
- **Inputs:** two groups of 32 lanes. Lane L of group g holds:
  - x = (0x80000000 >> g) >> L, shifted arithmetically. Group 0 holds negative values down to `0xffffffff`; group 1 holds single bits down to 0.
  - z = x ^ mul_u32_u24(L + 32g, 0x9e3779).
  - The scalar inputs are s9 = g * 0xff0f0010, which is 0 in group 0, and s10 = ~s9.
- **What is read back:** each lane stores 16 words: the three vector scans of x and of z, the six scalar results for s9, two SCC words, and `s_flbit_i32` and `s_ff0_i32_b32` of s10.
  - One SCC word is read after an `s_bcnt1_i32_b32` followed by an `s_ff1_i32_b32`, so a scan that wrote SCC would show.
  - The other is read straight after `s_bcnt0_i32_b32`.
- **Controls:**
  - The program must compile.
  - The VOP3 `v_ffbl_b32` with the clamp bit must be refused.
  - The reference must differ from each of these mutated references: scans counted from the wrong end, the unsigned scan in place of the signed one, a scan of 0 that returns 32, a bit count that leaves SCC alone, and a scan that writes SCC.

## Measured

Measured on 27.11.2026: `main` `98007de5` plus this change, Windows 11 Pro 10.0.26200, NVIDIA GeForce RTX 5090 (driver 32.0.16.1088), dev preset with the debug layer on.

- **Controls:** printed `compiled; clamp refused; the wrong-end, unsigned, zero-is-32, SCC-dropped and SCC-written references each differ`.
- **Readback:** printed `bit counts and scans: 1024 words of two groups and the guards matched`.
- **Negative controls:** each mutation below was built and run.

  | Mutation | Result |
  |---|---|
  | Lowering removed | Both tests fail (`direct DXIL lowering for v_ffbh_u32`) |
  | `FirstbitLo` and `FirstbitHi` exchanged | The readback fails |
  | Scans that also write SCC | The readback fails at the SCC word after `s_ff1_i32_b32` |

- **Not run:** the readback has not been run on WARP or on an AMD GPU.
- **Local start-up capture:** `compile_dxil --refusals` was run on one capture of 2,689 compute requests (counts only).
  - 12 requests stopped first at one of these operations: `s_bcnt1_i32_b32` (5), `s_ff1_i32_b32` (3), `v_ffbl_b32` (3) and `v_ffbh_i32` (1).
  - With this change, 6 of the 12 compile, so the total that compile goes from 392 to 398.
  - The other 6 stop at a later refusal: 2 at `s_absdiff_i32`, 2 at `unbound numeric SGPR`, 1 at `direct descriptor address outside bound table` and 1 at `non-contiguous resource descriptor`.
