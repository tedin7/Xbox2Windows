# Slot 57 mesh: a pre-exchange wave block with several scalar loads

Part of #2413, follow-up to #2660.

## Shape

Between the group-index exchange and the count exchange of a slot 57 mesh program:

```
s_mov_b64 exec, -1
...                                ; no EXEC or VCC operand, no branch entering from outside
v_cmpx_gt_u32_e32 s<bound>, v<t>   ; v<t> the proven v_mbcnt thread index
...                                ; no EXEC write
s_cbranch_execz SKIP
  ...  straight-line: s_buffer_load_dword*, s_load_dword*, scalar ALU,
  ...  lane-only vector code, a per-lane v_cmp into VCC
SKIP:
```

#2660 accepted exactly one `s_buffer_load_dword` whose result is read after the join, with the reset
directly before the compare.
This change accepts any number of the listed scalar loads and any scalar ALU in the block. The reset
may be further up when nothing between it and the compare names EXEC or VCC and no branch from
outside lands in that range.

## Semantics

The guest runs the block once per wave when any lane of the wave is active. The active lanes are a
prefix (`thread < bound`).
- The lowering computes `wave_any = (tid & ~63u) < bound` at the compare, as in #2660.
- It branches around the whole block only when `wave_any` is false.
- Every scalar instruction of the block then runs for every lane of a wave with an active lane, so
  each lane holds the values the guest wave holds.
- Vector writes stay predicated by each lane's EXEC bit.
- An empty wave issues no load, so an out-of-range address on it is never read.

Requirements, refused with `slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes`
unless noted:
- **Scalar instructions** may only read and write SGPRs, SCC and VCC. EXEC, M0, `src_*` operands,
  `s_*_saveexec`, MODE writes, branches, barriers, messages, stores and atomics are refused.
- **VCC written in the block, and SCC an ALU op writes there,** must not be read after the join.
  A scalar load does not write SCC, so a #2660-shaped block is unaffected.
- **Scalars read after the join** must be wave-uniform and defined before the block. An empty wave
  keeps their old values.
- **Another compare form**, an EXEC write between the reset and the compare, a branch into that
  range from outside, and any branch inside the block are refused.
- **Branches past the compare:** a branch landing between the compare and the execz branch is
  refused, because that path reaches the block without the compare.
- **Implicit EXEC writers** (`v_cmpx`, `s_*_saveexec`) between the reset and the compare end the
  reset search. The EXEC tracker of `mesh_skeleton` then refuses them with its own messages.
- **A block ending at the count exchange** keeps the exchange refusal of #2643.

## Tests

`tests/startup_mesh57_skeleton_check.py`, source-built fixture `WAVE_BLOCK_MESH` (assembled words checked
by the `binary` mode). The block holds:
- two `s_buffer_load_dword`, the second at an offset computed by `s_add_u32` in the block;
- a dead `s_load_dwordx4` table read;
- a dead `v_cmp_class_f32` into VCC;
- a scalar write between the reset and the compare.

The draw (`startup_mesh57_skeleton_draw_warp` / `_hardware`, same geometry as #2660) covers:
- a mixed wave whose provoking thread is outside EXEC and reads the loaded word;
- an empty wave that keeps the old scalar while its addresses lie past the buffer;
- the swapped old/new selection, which must fail on 259 pixels;
- a mutant that replaces the wave predicate by the lane's own EXEC bit. It must differ from the wave
  image on 66 pixels and match its own per-lane image.

Controls:
- New refusals:
  - VCC live at the join, `src_scc` in the block and EXEC copied in the block;
  - a branch inside the block;
  - a branch from outside between the reset and the compare, and a branch from that range past
    the compare to the execz branch;
  - an EXEC write between the reset and the compare.
- Without the compare-bypass rule, the bypass control translates. That is how the control was
  checked.
- Changed #2660 controls:
  - A wider `s_buffer_load` and a dead loaded scalar are now accepted.
  - The `s_nop` that replaced the adjacent reset now reaches the reset that closes the group-index
    exchange, so it is accepted.
  - The missing-reset control now writes EXEC from an SGPR pair.
  - The second live scalar write stays refused, because its register has no value before the
    block.
  - The register-offset `s_load` now stops at the existing register-offset rule.

## Local capture census (counts only)

On a local capture, 407 slot 57 requests stop at the refusal above on main `96a7b4fb`.
- 91 of them now pass the pre-exchange block.
- They stop at the same message one block later: an `s_add_u32` that is dead after the join, in the
  post-count primitive-export block, which jumps to `s_endpgm`. A separate pull request covers that
  block.
- With both changes applied, they stop at `v_readfirstlane_b32 outside the group-index read is not
  covered`.
- No new complete translation results from either change.
- The other 316 have a branch inside the block, `src_scc`, or the `v_cmpx_lt_u32 v, v` compare form.

## Wave collective readback controls

The slot 57 collective lowering extends the original wave-block proof above.
The synthetic WARP cases in `startup_mesh57_skeleton_wave_collectives_warp`
use the same indexed mesh draw and 32×32 CPU rasterizer. Each shader mutant
must fail against the correct reference and pass against its own reference.

- Force the inner VCC-zero branch to fall through on a zero load. The mixed
  wave writes zero red instead of retaining the initial red.
- Drop the EXEC predicate from the green write at the wave-load join. Only
  threads 0–2 should write green there, before the count exchange.
- Replace the first-active-lane broadcast with each thread's copy. The
  uniformly defined active value is 1.0, while inactive provoking threads
  still hold zero in the source VGPR.

Host verification on Linux against base `e8015da7a` with the fixture changes
applied: `python -B tests/startup_mesh57_skeleton_check.py` and
`python -B tests/startup_mesh57_skeleton_check.py binary /usr/bin/llvm-mc-19`
both reported `ok`. The latter checks the encoded synthetic programs against
assembly. GPU verification is pending on Windows. Run
`ctest --preset release --no-tests=error -R '^startup_mesh57_skeleton_wave_collectives_warp$' -V`.
