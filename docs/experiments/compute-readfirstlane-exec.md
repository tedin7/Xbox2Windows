# Compute first-lane reads after EXEC recovery

The compute translator retries a uniform-source refusal with a control-flow
proof. Each basic block inherits only the entry-mask facts common to all
incoming edges, including loop back edges. A direct move restores the fact when
all paths still hold the saved entry mask. A uniform VGPR written with entry
EXEC may also survive mask changes within a block if no instruction overwrites
that VGPR before the read.

The compute waterfall proof accepts unsigned bitfield extract, unsigned maximum,
32-bit multiply, eight-word fixed table descriptor loads, and two-word
read-only scalar buffer loads. A lane's key may set the byte offset in that
buffer. The translated scalar load clears the final offset's low two bits
before reading the pair. The descriptor and descriptor-table address must be
loop-invariant. The proof still rejects loop-carried scalars and escaping
key-dependent scalars. Pixel shaders with key-selected textures remain open.
See section 7.2.1 of the [RDNA 2 ISA reference](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf).

The [synthetic programs](../../tests/startup_compute_readfirstlane_exec_check.py)
cover an indexed formatted load at a branch target, 32-bit and 64-bit restores,
and a waterfall with repeated keys, inactive lanes and unaligned byte offsets.
Each program has a machine-code mutant that must disagree with the CPU
reference in D3D12 WARP readback. Refusal controls cover narrowing on a
predecessor or back edge, an overwritten mask or VGPR, varying load addresses,
key-dependent descriptor selection, a writable buffer and a live-out scalar.

```bash
python -B tests/startup_compute_readfirstlane_exec_check.py --llvm-mc llvm-mc-19
```

The synthetic programs passed local translation, refusal and encoding checks on
Linux. Against revision `234997f8c`, 37 existing synthetic programs produced
byte-identical HLSL, and all four new programs went from refusal to translation.
Windows readback is pending.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_readfirstlane_exec_(controls|warp)$' -V
```

## Uniform joins and first-active-lane reads

A further compute retry intersects uniform register facts across forward
wave-uniform branches. An entry-EXEC definition can copy a constant, a scalar
load or a group ID into a VGPR. Lane-local VALU operations and read-only buffer
loads preserve the fact when their inputs or addresses are uniform. The proof
allows different uniform values on different incoming edges.

For straight-line groups up to four threads, wave32 and wave64 guests each
fit in one Shader Model 6.0 host wave. The lowering elects the smallest
guest thread index enabled by EXEC, reads from that invocation with
`WaveReadLaneFirst` and broadcasts its word to the whole wave. Empty EXEC
elects guest lane zero. Before the read, the fallback supports entry EXEC and
narrowing by `v_cmpx`. Larger groups and other unproven EXEC or control-flow
cases retain the uniform-source refusal.
[RDNA 2 ISA reference](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
· [Shader Model 6 wave intrinsics](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/shader-model-6-0)
· [D3D12 wave widths](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html)

The synthetic kernels cover SGPR and VGPR joins, two workgroup IDs, a uniform
buffer address, a numeric scalar load and a per-lane source on each guest
wave width. The readback checks 320 words per dispatch against a CPU reference
that also checks inactive lanes after EXEC is restored. Each kernel has a
machine-code mutant whose first differing word must match the predicted
diagnostic. Static controls reject a varying phi input and eight-thread
groups with varying first-lane sources.

```bash
python -B tests/startup_compute_readfirstlane_wave_check.py --llvm-mc llvm-mc-19
```

Linux translation and instruction encoding checks passed. Against
`e8015da7a`, 41 existing compute fixtures emitted byte-identical HLSL.
The four new kernels and their mutations previously reported
`v_readfirstlane_b32 requires a proven uniform source`. WARP readback is pending.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_readfirstlane_wave_(controls|warp)$' -V
```
