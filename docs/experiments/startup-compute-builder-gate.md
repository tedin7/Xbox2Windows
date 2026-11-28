# All-or-none gate for the BVH builder's compute stages

The console ray-tracing structure builder is a set of compute programs from `XboxBvhBuilder.dll`.
Its stages feed each other through title memory; a stage that runs on data a missing stage never
wrote computes store offsets from garbage. In the observed failure, 14 of 35 translated stages ran
and the title died within a second on pointers that had become bounding-box bit patterns.

`X2W_COMPUTE_BUILDER_GATE=1` makes the set run all or none. It is off by default and changes no
existing path when unset.

## Identification

At device slot 62, the gate resolves the caller module of the return address with
`GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS)` and compares the basename with
`XboxBvhBuilder.dll`, case-insensitively. A compute pipeline is tracked only when that resolution
succeeds; an unresolved caller is left untracked rather than guessed. The header's pure classifier
and state machine are tested without a device or a title module.

## Decision

A builder Dispatch is allowed when every tracked builder pipeline has a real desktop PSO. On the
first builder Dispatch the gate makes the missing pipelines on the spot. If one cannot be made, the
builder is turned off for the rest of the run and the Dispatch is refused with a named reason. A
non-builder Dispatch is unchanged. The gate is keyed on the slot 62 wrapper's builder tag, not on
`compute_request_key`: slot 67 never receives that tag, so its request publication path is
unchanged. The gate is counted in the compute totals line as `builder gate refused N`.

When the gate completes the set it also refreshes the recording's cached compute state
(`pipeline_real`, `compute_pipeline`, `compute_pipeline_root`) and binds the new desktop PSO, and
the matching desktop root when the title already bound it. Without that refresh the caller's own
`Dispatch` decision still sees the placeholder that was bound before the gate ran and drops the
first Dispatch.

## Scope and limits

This is a safety gate, not a translation. The slot 62 programs are console GPU code; the gate's
on-the-spot creation first requires a structurally valid desktop compute container and a desktop
root signature. A console DXIL program fails by name and latches the builder off, which is the
intended conservative result until the translation path publishes real compute PSOs.

The runtime test loads a source-built `XboxBvhBuilder.dll` helper and calls the public slot 62
path through it. In the incomplete-set mode two fake console pipelines are tracked, a direct
`Dispatch` and an indirect `ExecuteIndirect` are both refused with the gate enabled, and the same
direct call through a non-builder slot 62 wrapper is left untouched. In the completed-set mode two
desktop DXIL compute shaders are tracked, the gate reports `all have a desktop PSO`, and the first
direct `Dispatch` plus an indirect `ExecuteIndirect` with a placed argument buffer reach the list.
A separate run of the existing slot 67 request check with `X2W_COMPUTE_BUILDER_GATE=1` passes, so
request publication keeps its normal path.

The helper test proves the caller-module tag, the wiring into both public Dispatch paths, and the
completed-set refresh on source-built desktop DXIL. It does not prove a translated builder set from
the title; the fake console container still fails desktop PSO creation by name and latches the
builder off, which is the intended conservative result.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_builder_gate$' -V
ctest --preset dev --no-tests=error -R '^startup_compute_builder_gate_runtime$' -V
ctest --preset dev --no-tests=error -R '^startup_compute_builder_gate_request_warp$' -V
```
