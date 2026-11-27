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

## Voted mask if/else

The new proof runs after the legacy wave-block proof declines. A forward SCC
switch can define the same mask pair in multiple arms. `s_and_saveexec_b64`
consumes the mask before writing its save destination, including when the
operands alias. The inner EXEC-zero branch votes across 64 guest lanes so each
lane of a wave sees the same scalar state after the arm. A wave that skips the
scalar loads must already hold each live scalar. The complement uses the saved
EXEC and the current EXEC. After the optional explicit OR restore, the enclosing
EXEC reset ends the block. A VCC-zero or VCC-nonzero branch in the mask setup
votes on the full guest wave's VCC bits.

The synthetic tests cover the scalar descriptor load followed by a buffer load,
constant and mask selects, EXEC copies, narrowing, and a VCC vote. Separate
draws cover predicated typed buffer and image loads. A group has two guest
waves. The first draw contains a mixed wave and an empty wave. A second draw
makes the second wave full. The pixel reference exports each wave's scalar
result from lanes outside the then arm as well as active lanes. Mutants include
a lane-local branch, a wrong complement, a wrong select, and an unpredicated
load. The offline purity control includes a whole-word numeric use of the pair.

```text
python3 -B tests/startup_mesh57_skeleton_check.py
python3 -B tests/startup_mesh57_skeleton_check.py binary /usr/bin/llvm-mc-19
ctest --preset dev --no-tests=error -R '^startup_mesh57_skeleton(_lanemask_warp)?$'
```

The Python and assembler checks run on a non-Windows host. The readback and
mutant draws require the Windows runner and report capability skip 77 when
the mesh shader tier is unavailable.

On Linux with Python 3.14.8 and LLVM 19, the first two commands passed. The S1
CPU model predicted 66 changed pixels for a lane-local EXEC branch and 259
for the wrong complement in the mixed/empty draw. A before/after comparison
against `d4a1fa9b7` covered 1,671 emitted HLSL and manifest outputs from the
existing mesh57 and gfx2hlsl fixture scripts, including the graphics shaders
from two draw-only scripts. All outputs matched byte for byte. The standalone
watch, retry and draw-skip checks also passed. The Windows draws remain to be
run.


## Shared EXEC-zero joins

The shared proof also accepts post-count mesh skips with a dead pure mask or
a zero-initialized compare pair consumed by a lane select after the join. The
existing mesh lane-mask readback mode covers both forms. Compute has four
wave64 readback cases for VCC, an SGPR compare pair, all-ones initialization
and a lane-preserving write. Changing the initialization in each case changes
16 CPU reference words. The pixel derivative suite tests a compare pair that
survives a quad-voted EXEC-zero join.

```text
ctest --preset dev --no-tests=error -R '^startup_(mesh57_skeleton(_lanemask_(warp|hardware))?|gfx2hlsl_compute_(controls|phi64_(warp|hardware))|image_sample_b_flow_(controls|warp|hardware))$'
```

On Linux with Python 3.14.8 and LLVM 19, the offline controls and assembler
checks passed in all three suites. The before/after comparison against
`871025f1c` matched all 484 emitted HLSL and manifest outputs from the seven
existing fixture scripts exercised in the comparison. Windows readback
remains pending.


Round 2 checks numeric reuse before and after a mask lifetime, a mixed
numeric/mask join, and high-word data reads. The same lane-mask draw mode
now covers a standalone uniform SCC select, sequential regions and a region
nested in the then arm. Reversing SCC in the standalone draw changes the
CPU reference. The offline wave64 high-VCC numeric-read control still
reports the same refusal.


On Linux with Python 3.14.8 and LLVM 19, the mesh offline and assembly
checks passed. Round 2 adds seven accepting fixtures and four refusal controls.
A before/after comparison against `fd5f476b7` covered seven existing scripts
and matched 496 emitted HLSL and manifest outputs byte for byte. Windows
readback remains pending.


## Defined positions under the published count

Round 3 tracks the sources of mask operations even when no mask instruction
writes those pairs. The loaded-pair and entry-only-pair controls changed
from pure to impure. The compare-and-NOT control stayed pure.

Both S1 arms now supply position VGPRs in a synthetic guarded draw. The
offline tests accept the same bound, a zero-or-bound select, and an unsigned
export compare. Five controls retain the legacy undefined-VGPR refusal.
The widened-count mutant adds one allocated vertex and changes 80 CPU
reference pixels. The mesh, compute and pixel controls, the mesh and
compute LLVM 19 assembly checks, and docs links passed on Linux. A
before/after check against `cf0e0c388` matched 517 legacy HLSL and manifest
outputs across seven scripts. Windows readback remains pending.


## Group-uniform values and guarded exports

Forward value numbering tracks reaching SGPR words, SCC, VGPRs and known
full EXEC between the exchanges. The GS ES+GS entry layout places per-draw
user data in s8–s39. The proof uses these values only before any program
write to their word. Uniform SALU operands yield uniform results. Uniform
addresses, descriptors and offsets preserve uniformity through read-only
scalar loads. A full-EXEC buffer load with an all-lane uniform address can
establish an all-lane uniform VGPR, which a first-lane read can turn into a
group-uniform scalar. The exchanged group index is also group-uniform when
the original v5 reaches the read. The numeric entry value of v5 still
requires a runtime source. The synthetic draw supplies the scalar arm's
index as draw data. [RDNA 2 Instruction Set Architecture, §3.12.2 p. 24,
ch. 7 p. 51 and V_READFIRSTLANE_B32 p. 137](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)

A join can merge distinct uniform values under group-uniform SCC. A join
at or inside the post-dominator interval of a divergent branch provides no
such proof. EXEC-zero and VCC branches are conservative divergent edges.
A backward edge disables value numbering for the entire pre-count region.
A memory write in either shader disables the load-based proof. [Convergence
And Uniformity, Uniformity](https://llvm.org/docs/ConvergenceAndUniformity.html)
[RDNA 2 Instruction Set Architecture, S_CBRANCH_EXECZ p. 125](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)

At a wave block's full EXEC reset, the walk tags the partially defined
VGPRs that all lanes of the guard wrote along every path. Joins retain
only agreeing tags on incomplete definitions. A later VGPR write or
exchange clobber drops the tag. The export can use a tagged VGPR when the
published count is the same bound or zero, including an SCC select between
them. With a signed compare, a bound with the high bit set admits no
non-negative thread index before the host clamp, and its unsigned
defining guard covers every thread. Different bounds, incomplete
definitions and wave-derived values keep the existing refusals. [RDNA 2
Instruction Set Architecture, S_CSELECT_B32 and S_MIN_U32 pp. 101–102 and
V_CMPX_GT_U32 / V_CMPX_GT_I32 p. 160](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)

Reusing a published count's SGPR preserves the active EXEC mask. A later
compare of that register loses its count provenance. Allocation and
primitive clamping use the published snapshot for each reused count. The
emitter keeps the original register expressions when the program leaves
the counts intact.

### Synthetic controls and draw

`python3 -B tests/startup_mesh57_skeleton_check.py` exercises the scalar
diamond, both wave-load blocks, signed and unsigned export guards, and
count-register reuse. The controls include a wave-dependent bound or load
address, a system entry word, a store in either stage, a first-lane source
written under narrowed EXEC, a divergent VCC diamond, a divergent bound or
count at a wave-load join, a different or widened count, a missing position
write, a later position overwrite, and a compare of an overwritten count.

`ctest --preset dev --no-tests=error -R '^startup_mesh57_skeleton(_lanemask_warp)?$' -V`
also draws both prefix arms through the mesh path. The CPU reference
models partial wave 0, empty wave 1, and an index that the guest clamps to
the last published vertex. The offline mutations mark the wave-derived
bound uniform (m1) and retain a tag across the later masked write (m2).
The draw compiles a mutant with the exporter using the decremented count
(m3) and tests it against both the guest and mutant CPU images. The
existing widened-export draw runs m4. Windows readback is pending.


On Linux, the no-argument check passed. Against `4955e24fb`, the later
masked write changed from accepted to the existing undefined-VGPR refusal.
The former dead overwrite of the bound register now passes because the
count already holds the old value. The bound-rewrite control now feeds the
replacement value into the count and retains the refusal. The rewritten
clamp changed 72 CPU pixels in the scalar arm and 38 in the formatted-load
arm. The widened export changed 80 CPU pixels.
