# v_perm_b32 and v_lerp_u8 in the direct C++ compute path

The direct compute frontend (`src/gfx1013_direct_compute.inl`) now lowers two VOP3 byte operations that it refused
with `direct DXIL lowering for <name>`. The semantics are the RDNA 2 ISA definitions, as the `gfx2hlsl.py`
lowerings in #2875 and #2873 state them:

- **`v_perm_b32`:** each result byte i takes the selector s = S2[8i+7:8i].
  - s 0..7 picks byte s of the 64-bit value {S0, S1}; S1 holds bytes 0..3.
  - s 8..11 gives 0xff or 0x00 from bit 15 of S1, bit 31 of S1, bit 15 of S0 or bit 31 of S0.
  - s 12 gives 0x00, and s 13..255 give 0xff.
- **`v_lerp_u8`:** each result byte is (S0 + S1 + S2[8i]) >> 1. Only the low bit of each S2 byte rounds.

Source modifiers and `clamp` stay refused (`direct ALU operand modifiers`). The path's existing rule for 32-bit
operations ignores `op_sel` bits, and this change does not alter that.

A local probe found both names. It compiles one instruction per VOP3 name of `src/gfx1013_isa_tables.inl` through
`compile_dxil ... gfx1013-direct-cs@0`, and is not part of this change. On `cpp-migration` `999e17cd` it saw 68 of
120 names refused. These two have `gfx2hlsl.py` pull requests from the start-up census, and neither had an issue
or pull request for the direct path.

## Fixture

`tests/native_perm_lerp_check.py controls|hardware`
(`ctest --preset dev --no-tests=error -R '^native_perm_lerp_' -V`). The words were assembled with LLVM 15
(`llvm-mc -arch=amdgcn -mcpu=gfx1013 -show-encoding`).

- **Dispatch:** two groups of 32 lanes. G = L + 32g is the lane's global index.
- **`v_perm_b32`:** S0 = 0x1b6c9e37, S1 = 0x80f07ad5, and the S2 bytes are 4G .. 4G+3. So the 64 lanes cover all 256 selectors once. A second `v_perm_b32` exchanges S0 and S1. The two words differ at bits 15 and 31, and their bits 7 and 23 differ from bits 15 and 31.
- **`v_lerp_u8`:** S1 = G * 0x9e3779b9, S0 = S1 ^ 0xff80017f, and S2 = G * 0x01030507.
- **Readback:** the 192 results and the guard words are compared with a CPU reference.
- **Controls:** they compile the program and check spot values of the definitions. These references must each differ from the expected words:
  - S0 and S1 exchanged;
  - the sign read from bits 7 and 23;
  - selector 12 as 0xff;
  - the lerp rounding bit dropped, or taken from the whole S2 byte.

## Measured

On `cpp-migration` `999e17cd` plus this change, Windows 10 build 19045, NVIDIA GeForce RTX 3080, driver
32.0.15.9174:
- **Controls:** pass.
- **Readback:** matched all 192 words and the guards.
- **Mutated lowerings:** each fails the readback at its first affected words. The mutations are the sign always taken from S0, the lerp rounding bit dropped, and selector 12 as 0xff.
- **Neighbour:** the existing `native_align_bfm` readback still passes.

The registered `native_store_replay` uses the inbox D3D12 runtime. On Windows 10 that runtime does not report
shader model 6.6, so the readback reports a skip (77) there. These readbacks used a local copy of the replay
linked with the app-local Agility SDK exports (`D3D12SDKVersion` 620), as `offscreen_check` has them. That local
copy is not part of this change.
