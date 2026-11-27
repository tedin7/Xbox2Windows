# EXEC removal and structurization for the shader frontend

The startup translator refuses a large fraction of the title's shaders on control-flow and lane-mask
shapes. The same shapes recur across compute, graphics and mesh programs:

| Refusal family | Example measured on 26 November 2026 | Count |
|---|---|---:|
| `s_cbranch_execz` / `s_cbranch_scc0` inside a skipped block | `s_branch inside an s_cbranch_execz block` | 331 (RTX 3090 run) |
| Slot 57 mesh skipped-block effects | `slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes` | 241 |
| `v_readfirstlane_b32` without a proven uniform source | `v_readfirstlane_b32 requires a proven uniform source` | 526 |
| `src_scc` with unverified SCC | `src_scc with unverified SCC` | 762 (rev28 census) |
| Wave reductions in branched programs | `wave reduction in a program with branches` | present in the census |

The 26 November census held 3,007 `.fail` records (2,564 compute, 275 graphics, 162 mesh). Adding
one accepted shape at a time moves a request to the next refusal more often than it produces a
complete translation. The mesh notes record exactly that: a change accepts 91 of 407 requests at one
pre-check, and all 91 stop at the next refusal, with no new complete translation.

This note proposes a structural fix: lower the program **as one lane runs it**, without EXEC, then
emit HLSL from the structured control flow. It is a design for the C++ frontend; the current Python
translator remains the reference for instructions that already translate.

## Why the local pattern approach does not converge

EXEC, VCC, SCC and the saved-exec registers are one dependency graph. A rule that proves a scalar
dead at one join, or a mask equal to a lane's own bit, depends on the paths that reach the join and
on every writer of the mask. A pattern that accepts one shape without a general representation of
that graph leaves the next shape refused. The refused shapes also interact:

- a branch on a mask test is also a join of the mask;
- a scalar load inside an EXEC-zero block is uniform only because the guest runs the block once per
  wave, while the per-lane model may run it per lane;
- a `v_readfirstlane_b32` waterfall has one descriptor per lane and needs the per-lane source that
  produced it;
- a wave reduction is a loop over lanes that changes EXEC on purpose.

## Proposed pipeline

```
decode -> CFG/SSA -> normalise special registers -> per-lane mask booleans
       -> structurize without EXEC -> path-sensitive guard proof
       -> resource/address tracing -> HLSL -> DXIL
```

### 1. Decode and CFG/SSA

Keep the existing gfx10 decoder. Build basic blocks and SSA values for VGPRs, SGPRs, SCC, VCC and
EXEC. The CFG must be reducible for DXIL; refuse irreducible graphs with a named reason rather than
guessing. Record, for each instruction, the block and the word offset for diagnostics.

### 2. Normalise special registers

Turn writes to EXEC, VCC, VCCZ and the saved-exec registers into explicit values. A write from
`v_cmp*`, `s_and_b64`, `s_or_b64`, `s_cselect_b64`, `s_mov_b64` or `s_*_saveexec` becomes a mask
value, not an opaque register. A write the pass cannot follow becomes an `Opaque` mask atom and is
kept as a per-lane boolean, with a refusal only where a later use needs a stronger fact.

### 3. Per-lane mask booleans

Represent a lane mask as a Boolean function over these atoms:

- `Lane(value)`: the lane's bit of a VALU-written compare or carry result, before EXEC masking;
- `Uniform(value)`: an SCC value that is the same in every lane;
- `Phi(value)`: a mask joined at a block entry;
- `Quad(value)`: an `s_wqm` result (some lane of the quad is in the source mask);
- `Opaque(value)`: a mask the pass cannot see into;
- `Constant(mask)`: all lanes, no lanes, or a constant mask.

A reduced ordered binary decision diagram (BDD) gives canonical equality and cheap `and`/`or`/
`not`/`ite`. The alternative is a bitset per atom set; the BDD wins when the same mask is rebuilt
along many paths, and loses on pathological atom counts. Whichever representation is chosen, the
interface must expose: `lane_bit(mask, lane)`, `is_all`, `is_none`, `is_equal(a,b)`, and substitution
of a lane's bit for a uniform branch.

### 4. Remove EXEC

Translate the program as one lane. For each vector instruction, compute the guard
`lane_bit(EXEC_at_instruction, lane)`. If the guard is `true` for every lane that reaches the
instruction, emit it unguarded. Otherwise emit a per-lane `if` around it. This is valid because the
guest drops the instruction's side effects outside EXEC, and PC hardware only requires the same
result in the lanes that matter; helper-lane work is dead except where derivatives or quad
operations read it.

A branch on a wave-wide mask test (`s_cbranch_execz`, `s_cbranch_vccz`, an SCC computed from a mask)
becomes a branch on `lane_bit(mask, lane)` for the lane being modelled. A path-sensitive pass then
proves where the guard is constant and removes the `if`.

### 5. Structurize

Recover `if`/`else`, `break`, `continue` and loops from the reducible CFG. Keep the block shape so
diagnostics and the existing instruction offsets survive. A loop's EXEC changes must be modelled per
iteration; the pass must not hoist a mask out of a loop unless the loop is proven not to change it.

### 6. Path-sensitive guard proof

For each instruction, collect the mask facts that hold on every path to it. Prove a guard `true` when
all reaching paths imply it; prove `false` when none reaches it. Only emit a per-lane `if` when the
proof cannot decide. This is the step that turns the general model into the small HLSL the current
translator emits for its accepted shapes.

### 7. Resource and address tracing

Trace each descriptor and scalar address to a root parameter (table, heap, root descriptor, root
constant) or to a raw title address. A descriptor that a waterfall reads in the first lane is traced
to the per-lane load or choice that produced it. A raw address is a 40-bit title address, not a
desktop pointer; it needs the page directory and a bound resource, or a named refusal.

## Mapping to the C++ frontend

The Python translator's accepted instruction set is the semantic reference, not the architecture.
Keep its tests and its refusal strings. Move the following into the C++ frontend as separate passes
with their own fixtures:

| Current Python area | C++ pass | Keep | Replace |
|---|---|---|---|
| `Stage.execz_refusals`, per-shape mask proofs | CFG/SSA + mask BDD | refusal reasons and tested shapes | ad-hoc shape matching |
| `spill_*`, `uniform_branches` | normalise special registers + guard proof | spill-slot semantics | per-instruction exception tables |
| `uniform_image_load_regs` / `v_readfirstlane` proof | resource/address tracing | per-lane descriptor facts | local scan heuristics |
| wave-reduction recogniser | cross-lane operation pass | opcode set and wave32/wave64 rules | pattern matching on the chain |
| mesh skeleton | mesh pass on the same CFG | group-index/count exchange rules | duplicated EXEC rules |

The frontend should keep a `--explain` path that names the first unresolved fact for each refused
program, so a census remains comparable with the Python `.fail` reasons.

## Phases

### Phase 1: uniform skipped blocks

Accept an `s_cbranch_execz` block entered by a wave-uniform condition whose scalar writes are dead at
the join. Emit the block once per wave, predicate vector writes by the lane's EXEC bit, and keep an
empty wave from issuing loads. This is the shape already measured in the slot 57 notes; it is a
useful first test of the model but not the end state.

Acceptance: a source-built gfx1013 compute or mesh program with a uniform skip and a dead scalar
write translates; a control with a live scalar at the join is refused by name; the current
translator's accepted programs still produce the same pixels.

### Phase 2: mask joins and SCC branches

Add the BDD or bitset representation. Cover `s_and_b64`, `s_or_b64`, `s_xor_b64`, `s_not_b64`,
`s_cselect_b64`, `s_cmov_b32`, `s_cbranch_scc0/1`, and `v_cndmask_b32` mask selection. The first
target is the 762 `src_scc` census entries and the 95 `s_cbranch_scc0` inside skipped blocks.

Acceptance: a differential set of programs where the current translator accepts one form and the new
pass accepts the neighbours; a control that uses a varying SCC as a uniform branch must be refused,
not miscompiled.

#### Phase 2, first slice

The slot 57 wave-load block uses shared helpers to classify wave64 mask pairs.
The proof inspects decoded accesses, including overlapping SGPR words, and
checks that a mask read has a definition on every incoming path. The existing
one-lane representation carries the lane's EXEC bit through comparisons,
Boolean operations, copies, and selects on verified uniform SCC.

The first shape has forward uniform branches, one inner AND-saveexec if/else
and a complement. An EXEC vote gates its scalar then arm. VALU and read-only
vector loads obey the lane predicate. The outer full reset closes the region.
Numeric reads of masks, SCC produced by mask operations, non-uniform scalar
inputs, undefined skip values, and scalar else arms without a vote are refused.
Additional nested if/else regions and uses of the saved mask beyond the region
remain refused.

#### Shared EXEC-zero join proof

Compute, pixel and mesh use the same full-pair classifier and skipped-lane
write rules. A compare or an EXEC-masked write requires zero on the skip path.
A write that produces one for inactive lanes requires all ones. A preserving
write may retain any proven lane mask. Once the region joins, a live mask can
feed a lane select, a carry, or an operation that intersects it with EXEC.
Either half used as numeric data, a whole-wave observation, an SCC read of
a mask operation, or unexpected control flow keeps the prior refusal.
The new proof covers straight forward blocks with stable EXEC. Existing
compute and graphics programs get their original lowering first.

### Phase 3: `v_readfirstlane` and waterfalls

Trace the per-lane source of a first-lane read through `v_cndmask_b32`, an if/else whose arms set
the same descriptor, or a load from a per-lane descriptor. Cover the 526 first-lane refusals and the
remaining mesh group-index reads.

Acceptance: a source-built descriptor waterfall whose lanes select different buffers; a control with
an unproven lane source is refused; a draw reads the expected texels.

### Phase 4: wave reductions

Model the reduction as a lane loop or as the corresponding wave intrinsic when the shape is proven.
Keep the existing wave32/wave64 opcode set (`v_or_b32`, `v_min_u32`, `v_max_u32`, the saveexec form)
and extend only with a tested shape. The current straight-line-only restriction is the thing to
remove; the identity fill and the partial-mask rules must stay explicit.

Acceptance: a reduction inside a branch and a loop; a wave64 control; a partial-mask control that
must refuse.

### Phase 5: helper lanes, quads and mesh

Helper lanes count as live for derivatives and quad operations. Model `s_wqm`, quad reads and the
mesh group-index/count exchange on the same CFG. This is where the slot 57 mesh path and the pixel
helper-lane path converge; do not fork a second EXEC model for mesh.

Acceptance: the existing quad/helper-lane fixtures, the mesh wave-block fixtures, and a new mesh
program with a count exchange and a real export.

## Test and oracle strategy

- **Differential against the current translator:** for every shape the Python translator accepts,
  the new pass must produce the same pixels or the same raw words. This is the strongest regression
  oracle and costs no new golden data.
- **Source-built gfx1013 programs:** one program per phase, assembled with the existing fixtures,
  with expected values written out independently of the pass.
- **Negative controls:** a live scalar at a join, a varying SCC used as a uniform branch, an
  unproven first-lane source, a partial-mask reduction, and an irreducible CFG. Each must be refused
  by name, not translated approximately.
- **GPU coverage:** WARP for every shape; hardware on the available Radeon/NVIDIA adapters for the
  first program of each phase. Report capability skips separately.

## Risks and non-goals

- **BDD blow-up:** cap the atom count and fall back to a bitset or a named refusal. A cap that
  silently changes semantics is not acceptable.
- **Irreducible CFG:** refuse; DXIL requires reducible control flow.
- **Helper lanes:** a per-lane model can drop work that derivatives read. Keep helper lanes live in
  pixel stages and prove the quad operations.
- **Wave width:** wave64 semantics on a wave32 host need exact lane-mask handling; the existing
  wave64 tests remain the reference.
- **Resource tracing is not part of the control-flow pass.** Keep it separate so a mask bug does not
  hide a binding bug.
- **Non-goal:** translating every instruction at once. The pass order must land behind tests, one
  phase at a time.

## Recommended first patch

Start with Phase 1 in the C++ frontend or, if the frontend is not ready, add the same uniform-skip
rule to the Python translator behind a narrow fixture. The patch should:

1. accept one wave-uniform `s_cbranch_execz` block with scalar writes dead at the join;
2. predicate vector writes by the lane's EXEC bit;
3. refuse a live scalar at the join and an empty-wave load by name;
4. prove it on the existing slot 57 wave-block fixture and one compute program.

That is small enough for one reviewable PR and gives the C++ frontend its first end-to-end control-
flow test without pretending the general pass is finished.

## References

- `tools/startup/gfx2hlsl.py`: `Stage.execz_refusals`, `spill_*`, `op_v_readfirstlane_b32`, the
  wave-reduction recogniser, and the slot 57 skeleton.
- `docs/experiments/startup-mesh57-wave-block.md` and `startup-mesh57-dead-block-scalar.md`.
- Issues #2125, #2413, #2023, #1723 and #1143.
- The public design description of DexNom/VIctory's `structure.rs` (GPL-3.0; read-only, no code
  copied): per-lane booleans, EXEC removal, structured control flow, helper lanes.
