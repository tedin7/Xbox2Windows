# Slot 57 mesh: plan for a complete DXIL translation

Slot 57 is the fixed mesh pipeline path: the title creates a mesh pipeline whose programs are
console gfx1013, then issues `DispatchMesh` (directly or through an indirect signature). The
tracking issue #2413 recorded 298 distinct requests and **0 complete translations**. The partial
slices merged so far remove pre-checks; they do not produce a complete program, and the next
refusal is always waiting.

This note separates the work that must happen before a first complete translation from the runtime
work that consumes it. It is a plan, not a claim that a scene renders.

## Entry ABI: user data and group index

For this mesh entry, `s3` carries the wave index in bits 24–27. The platform user
words occupy `s8..s15`; root user data begins at `s16` and follows the root layout.
Entry `v5` is otherwise undefined in the translator. Only the recognized exchange
reads lane 0 of wave 0, shifts `v5` right by seven and broadcasts the result as a
wave-uniform linear mesh group index.

The [RDNA 3 ISA reference, February 2023, §3.5.4, p. 31](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
identifies `v5` as the ES base vertex index in GS(+ES) Fast Launch 1 combined mode,
following the five GS registers; in ordinary ES-is-VS mode it is the vertex index.
The [geometry-engine presentation](https://gpuopen.com/gdc-presentations/2024/GDC2024_Mesh_Shaders_in_AMD_RDNA_3_Architecture.pdf)
describes mesh thread launch. With 128 thread slots per group, this model infers
that the first wave's base vertex index is 128 times the linear group index; the
exchange's shift recovers that index. The [D3D12 mesh specification](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html)
also limits a mesh group to 128 threads.

The runtime contract for a slot 57 draw is `DispatchMesh(linear_group_count, 1, 1)`;
under this contract the translated word is `SV_GroupID.x`. The runtime does not yet
load slot 57 outputs. This contract does not define any other use of entry `v5`.

The synthetic group fixture was checked on Linux x86-64, Python 3.14.8, with
step-1 revision `90a5ef90a` plus this change. `python3 -B
tests/startup_mesh57_skeleton_check.py` and `python3 -B
tests/startup_mesh57_skeleton_check.py binary /usr/lib/llvm-19/bin/llvm-mc`
both passed. Restoring the previous group symbol makes the new offline acceptance
and group mode refuse numeric use of the index. The CPU references for the correct
and zero-index images differ at 134 pixels. WARP readback and its shader mutant
remain to be run on Windows; the offline result does not establish GPU behavior.
The additional no-argument Linux CI checks ran with the system Python: 92 passed
and 13 failed to start because `pefile` or `iced_x86` was absent. They require
the development dependencies before the full CI list can pass.

## Dependency map

```
device slot 57 request
  -> gfx2hlsl inspect_mesh57_request  (pre-checks, skeleton, pixel stage)
  -> translate_mesh57                 (mesh body, exports, container)
  -> <key>.ms.cso / <key>.ps57.cso
  -> runtime CreateMeshPipelineState + root signature
  -> DispatchMesh (direct or indirect signature)
  -> GPU mesh pipeline + pixel stage
```

The translator and the runtime have separate gates. A program can translate and still not run if
the runtime refuses `DispatchMesh` (`#1759`) or the pipeline root; a runtime path can be ready and
draw nothing without a translation. Keep them as separate PRs.

## Measured refusal shapes

From the local capture census in the #2413 notes and the 26 November runs:

| Shape | Requests | Where it stops |
|---|---:|---|
| `s_cbranch_execz` pre-exchange block with scalar loads | 407 | `s_cbranch_execz skips code with effects outside active lanes` |
| Same, after the wave-block and dead-scalar slices | 91 | `v_readfirstlane_b32 outside the group-index read is not covered` |
| Branch inside the skipped block, `src_scc`, or `v_cmpx_lt_u32 v, v` | 316 | one of those named refusals |
| Pixel stage mixes the null export with another export | 97 | `slot 57 pixel stage mixes the null export with another export` |
| Group-index/count exchange shape | 15 | exchange refusal |
| Loops | 11 | loop refusal |
| `v_fma_mix_f32` reading the upper half of `v_interp_p2_f16` | 39 | pixel-stage refusal |
| `s_load` with a register offset through the resource heap | 47 | resource-heap refusal |

The numbers are counts from one capture, not a promise that fixing a shape completes a program. The
91-request slice is the clearest evidence: two accepted blocks still produced zero complete
translations because the next refusal was waiting.

## The smallest end-to-end target

Do not start from the 298 real requests. Start from a source-built slot 57 program with the smallest
shape the runtime can create and dispatch:

- one mesh workgroup, one output lane per thread (wave64 or wave32 depending on the fixture);
- three vertices and one triangle, written through the mesh export path;
- a group-index read through `v_readfirstlane_b32` of a formatted buffer load;
- one `SetMeshOutputCounts` and no LDS exchange beyond the group index;
- a constant-colour pixel stage, then a textured pixel stage.

This program proves the chain: request -> `MeshPixelStage` -> DXIL mesh container -> desktop
`CreateMeshPipelineState` -> `DispatchMesh` -> readback. Only then take the first real request whose
shape is closest to this program.

## Pass plan

### 1. Mesh skeleton and wave model

The existing `mesh_skeleton` already handles the group-index/count exchange, the wave64 lane masks
and the fixed user-SGPR layout. Keep it. The mesh stage must model one thread per output lane and
must not reuse the pixel helper-lane rules blindly.

Requirements: GS user data starts at s8 and the root starts at s16; the root layout is tables, a root descriptor pair, root
constants, then heap bases; a program with one fewer table shifts by one. `s_bitcmp0_b32 s20, 28`
tests a packed root-constant field, not a hardware system value. Resource-heap records are indexed
as `(rootconst & 0x3ff) << 5` and loaded with `s_load_dwordx4` from the 32-bit heap base.

### 2. `v_readfirstlane_b32` of the group index

The 91 requests that pass the wave block stop on a first-lane read of a formatted buffer load in the
pre-exchange code. The pass must trace that read to the per-lane descriptor and the lane that
produced it. This is the same fact the compute waterfall path needs; do not add a mesh-only rule.

Acceptance: the source-built group-index program translates and reads the expected index; a control
whose first-lane source is not proven is refused by name.

### 3. Branches and SCC inside skipped blocks

The 316 remaining pre-check requests have a branch inside the block, `src_scc`, or the
`v_cmpx_lt_u32 v, v` compare form. Do not add one pattern per shape. Use the EXEC/SSA structurization
from `docs/experiments/exe-structurization.md`: represent the block as one lane, branch on the lane's
own mask bit, and prove the scalar values that survive the join. A mesh-specific exception table is
the wrong direction.

### 4. Group memory and output counts

The mesh stage writes group memory (LDS) and then exports vertices and primitives. The pass must:

- keep the group-index and count exchanges wave-uniform;
- keep `SetMeshOutputCounts` once per workgroup, not once per lane;
- respect the declared maximum vertices/primitives and the output-size limit;
- map `v_mbcnt`, `ds_*` and the export registers to the DXIL mesh output arrays;
- keep helper lanes out of the mesh output count path.

Acceptance: a source-built triangle with three vertices and one primitive; a control with a wrong
count, a wrong index, and a missing export each produces a named failure or a pixel mismatch.

### 5. Pixel stage linkage

The slot 57 pixel stage is a separate console program. Link its inputs to the mesh outputs by
semantic, not by register order. The 97 requests that stop on `slot 57 pixel stage mixes the null
export with another export` are the early-exit form: the null export after the main `s_endpgm` is a
kill, not a second colour target. Accept only that block; leave other mixed-export forms refused.

### 6. Runtime creation and dispatch

The translator can publish `<key>.ms.cso` and `<key>.ps57.cso` before the runtime consumes them. The
runtime needs:

- `CreateMeshPipelineState` with the translated root signature and the MS/PS bytecode;
- the mesh container fields (max vertices, max primitives, topology, output size);
- `DispatchMesh` direct and indirect, with the console command signature;
- the pipeline root and the group counts on the same command list.

Keep the runtime slice behind a source-built `startup_mesh57_*` fixture; a translation without a
runtime fixture is not an end-to-end result.

## Phased schedule

| Phase | Scope | Observable acceptance | Owner |
|---|---|---|---|
| A | source-built minimal mesh + group-index read | translate, create PSO, dispatch, read back 3 vertices/1 triangle on WARP and hardware | translator + runtime |
| B | `v_readfirstlane` group index on a real request shape | the 91-request class moves past the refusal; one request translates completely if the rest is present | translator |
| C | EXEC/SSA branches inside skipped blocks | the 316-request class moves; no mesh-only exception table | frontend |
| D | group memory, counts, export limits | source-built triangle with LDS and count controls | translator |
| E | pixel-stage linkage and null-export kill | 97-request class pixel programs translate and draw the expected pixels | translator + GPU |
| F | one real slot 57 request end-to-end | a real request creates a desktop mesh PSO and `DispatchMesh` draws the expected geometry in a capture | runtime + GPU |

Phase A is the only one that can claim a complete translation. B-E are enabling slices; each should
say how many requests moved and whether any complete translation resulted.

## Risks

- **Wave64 on wave32:** the title's mesh programs use 64-lane waves and 64-bit EXEC operations. A
  wave32 host needs exact lane-mask handling; a silent 32-lane assumption changes the group index
  and the export count.
- **Workgroup size and LDS:** the console mesh group is not the desktop mesh group. The pass must
  preserve the group-index/count exchange and the LDS size the program declares.
- **Output limits:** DXIL mesh outputs have a size limit; a program that fits the console may need a
  refusal or a split. Do not let the validator fail at runtime without a named translation refusal.
- **Slot 57 vs slot 86:** slot 57 is the fixed mesh pipeline; slot 86 is the stream pipeline path.
  Do not merge their request formats or their root rules.
- **Helper lanes:** a pixel stage that uses derivatives needs helper lanes; a mesh-only per-lane
  model that drops them will produce wrong edges.

## First patch recommendation

Add the `v_readfirstlane_b32` group-index read to the mesh skeleton, with the source-built minimal
mesh from Phase A. The 91 requests already pass the wave block and stop there; the patch tests the
same fact on a program that can be drawn end-to-end. If the minimal program cannot yet create a
desktop mesh PSO, land the translator test and the runtime fixture separately and do not claim a
complete translation.

## References

- Issue #2413 and its comments, especially the T6/T7 wave and dead-scalar slices.
- `docs/experiments/startup-mesh57-wave-block.md`, `startup-mesh57-dead-block-scalar.md`.
- `tools/startup/gfx2hlsl.py`: `inspect_mesh57_request`, `translate_mesh57`, `MeshPixelStage`.
- The public design description of DexNom/VIctory's mesh emitter and diff tests (GPL-3.0; read-only,
  no code copied).
