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

## Wide vector raw loads

The same waterfall proof accepts one to four dwords per raw vector load. With
`offen`, the lane's VGPR byte offset can be combined with the MUBUF immediate.
All consecutive destination VGPRs belong to the executing lane. The descriptor
must stay fixed throughout the loop. The proof also accepts 32-bit signed
scalar field extract, signed scalar minimum and maximum, unsigned scalar
minimum, and vector AND, OR, NOT and XNOR. The scalar variants have the same
SCC effects as their admitted counterparts. The vector bitwise operations
write one VGPR in the executing lane.

The [readback fixtures](../../tests/startup_compute_waterfall.py) use two
32-thread dispatches and five unevenly distributed keys. The keys select
regions of one bound raw input buffer. Each lane writes all six raw load
results or all eight integer ALU results. The four-word load is independently
mutated to emit only its first destination and to omit its immediate offset.
Indexed and cache-flagged loads and a lane-count body remain refusal controls.

On Linux, the translation, controls and assembly comparison passed with
`python -B tests/startup_compute_readfirstlane_check.py --llvm-mc llvm-mc-19`.
The existing accepted waterfall controls produced byte-identical HLSL against
`e8015da7a`. Windows GPU readback remains pending.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_readfirstlane(_warp)?$' -V
```
