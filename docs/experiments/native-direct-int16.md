# Non-packed 16-bit integer VOP3 operations in the direct C++ compute path

The direct compute frontend (`src/gfx1013_direct_compute.inl`) now lowers twelve VOP3-only 16-bit integer operations
that it refused with `direct DXIL lowering for <name>`:
- `v_add_nc_u16`, `v_sub_nc_u16`, `v_add_nc_i16` and `v_sub_nc_i16`;
- `v_mul_lo_u16`;
- `v_lshlrev_b16`, `v_lshrrev_b16` and `v_ashrrev_i16`;
- `v_min_u16`, `v_max_u16`, `v_min_i16` and `v_max_i16`.

Each writes D[15:0] from S0[15:0] and S1[15:0], and D[31:16] keeps its value. #669 records two sources for keeping
the high half on RDNA 2:
- an inline-assembly measurement on RDNA 2 silicon, which kept D[31:16] for all of these without `op_sel`;
- LLVM's `GCNSubtarget::zeroesHigh16BitsOfDest`: "On gfx10, all 16-bit instructions preserve the high bits".

The shifts take S1 shifted by S0[3:0]. Additions, subtractions and the multiply wrap modulo 2^16. The `_i16`
minimum and maximum compare signed halves, and `v_ashrrev_i16` sign-extends S1[15:0].

`op_sel` selects other halves, so it stays refused (`unsupported <name> op_sel`). Source modifiers and `clamp`
stay refused by the path's existing rule.

The three-source 16-bit forms are not covered here: `v_mad_u16`, `v_mad_i16`, `v_min3`, `v_max3` and
`v_med3` `_i16`/`_u16`.

## Fixture

`tests/native_int16_check.py controls|hardware`
(`ctest --preset dev --no-tests=error -R '^native_int16_' -V`). The words were assembled with LLVM 15
(`llvm-mc -arch=amdgcn -mcpu=gfx1013 -show-encoding`).

- **Dispatch:** two groups of 32 lanes. G = L + 32g is the lane's global index.
- **Sources:** S0 = G * 0x9e3779b9 and S1 = (G * 0x7f4a7c15) ^ 0x5bd1e995. Both high halves are nonzero, the shift counts cover 0..15, and bit 4 of S0 varies.
- **Destinations:** each is set to 0xcafebabe first, then the twelve results are stored per lane.
- **Readback:** the 768 results and the guard words are compared with a CPU reference.
- **Controls:** they compile the program, check spot values, and require `op_sel` on `v_add_nc_u16` to be refused. These references must each differ from the expected words:
  - a zeroed high half;
  - a five-bit shift count;
  - signed and unsigned order exchanged;
  - a logical shift for `v_ashrrev_i16`.

## Measured

On `cpp-migration` `999e17cd` plus this change, Windows 10 build 19045, NVIDIA GeForce RTX 3080, driver
32.0.15.9174:
- **Controls:** pass.
- **Readback:** matched all 768 words and the guards.
- **Mutated lowerings:** each fails the readback. The mutations are the high half zeroed, which reads back `0x0000e995` against the expected `0xcafee995`, a five-bit shift count, and signed minimum and maximum compared unsigned.
- **Neighbour:** the existing `native_align_bfm` readback still passes.

The registered `native_store_replay` reports a skip (77) on Windows 10, whose inbox D3D12 runtime does not report
shader model 6.6. These readbacks used a local copy of the replay linked with the app-local Agility SDK exports
(`D3D12SDKVersion` 620, as `offscreen_check` has them). That local copy is not part of this change.
