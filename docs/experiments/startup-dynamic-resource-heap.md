# Dynamic raw buffer records

The graphics translator can use a uniform scalar byte offset into a heap of
32-byte records. The compute translator also handles an index returned by a
uniform scalar load from a fixed heap buffer or an upload-backed root SRV or CBV.
The compute forms that already name a root constant or a dispatch-constant
table buffer word keep the original manifest and HLSL.

The new shaders use Shader Model 6.6. The runtime creates a shader-visible
snapshot with the raw buffer records at their original indices and the fixed
translated bindings after them. The command recording retains the snapshot
until its submissions finish. The shader returns zero for a record index
greater than or equal to the logical record count. Empty and unbindable raw
views in the snapshot use null descriptors. This lowering covers direct
draws and dispatches, four-word descriptors, raw reads, and straight-line
shaders.

## Texture records

Graphics draws whose manifest carries `dynamic_heap_images` (the translator
writes it when a stage reads an image record from the heap) mirror texture SRV
tokens from 32-byte image records at their original heap indices; other heap
draws and compute dispatches keep the buffer-only snapshot, so a bindless heap's
textures are uploaded only for draws that can sample them. The snapshot retains the sampled resources until the
recording retires. A texture that cannot be bound, including a render target
or writable depth attachment of the same draw, gets a null SRV of the token's
dimension. The shader and the indexed view must agree on the resource type.
Empty slots have no texture dimension, so the image lowering's slot-zero
fallback requires an explicitly populated, matching view at that slot.

Run `python3 -B tests/startup_heap_texture_draw_check.py` for the translation
and CPU reference checks. On Windows, use `ctest --preset dev --no-tests=error
-R '^startup_heap_texture_draw_(warp|hardware)$'` for readback at byte offsets
32 and 96 and a raw buffer control at 64. The WARP run also disables texture
mirroring and requires a pixel mismatch at both image indices.
The stream fixture requires Shader Model 6.6, resource binding tier 3 and
32-byte descriptor slots.

## Reproduce

On Linux, run the translation fixtures and the compatibility checks:

```sh
python3 -B tools/run_checks_parallel.py -j 2 tests/startup_gfx2hlsl_dynamic_heap_check.py tests/startup_compute_dynamic_heap_check.py tests/startup_gfx2hlsl_heap_record_check.py tests/startup_gfx2hlsl_heap_word_check.py tests/startup_translator_check.py tests/startup_compute_translate_check.py
```

All six Python checks passed in the Linux worktree. Baseline `e8015da7a` refuses the graphics register offset and the compute index
from a fixed heap buffer. The legacy heap fixtures compare their HLSL and
manifest against the established output.

On Windows with DXC, run the D3D12 WARP readback tests. These runs are pending:

```powershell
ctest --preset dev --no-tests=error -R '^startup_(gfx2hlsl|compute)_dynamic_heap_warp$'
```

The draw checks two records and an index equal to the logical heap size.
The dispatch checks all three index sources and confirms that the other 319
output dwords retain the sentinel. Changing the record size or index-word
address must change readback. A separate bounds mutant must read the poison
descriptor immediately past the logical end. The WARP tests require Shader
Model 6.6 and resource binding tier 3 and skip when either is unavailable.

## Graphics image records and scalar tables

Vertex stages can read heap image views using texel loads or explicit-level
samples. The loop fixture loads two view indices from a root-table buffer and
adds their red texels under narrowed EXEC. The loop proof runs after a heap
load refusal and checks all back edges. Separate fixtures exercise numeric
reads of fixed root-table descriptor words and low-half mixed-FMA constants.
The WARP fixture binds texture SRVs directly. The runtime's dynamic-heap
snapshot still creates raw buffer SRVs.

Image dimensions use the complete shift-and-clamp calculation for the view's
top mip. The fixtures use 37 × 11 at mip 0 and 43 × 15 at mip 1. A separate
width select compares image record word 1 with zero and draws with a bound 37
× 11 view and a typed null view. The CPU reference expects widths 37 and 0.
Inverting the comparison changes both readbacks. Raw
mip-zero size fields and absolute mip indices remain unavailable because the
desktop view supplies its relative mip range. [AMD RDNA 2 ISA, §8.2.6](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)

The vertex scalar table fixture reads a program-image table through two raw
descriptors with different bounds. It checks a misaligned byte offset, a
two-dword read that overwrites its offset register and reads past the bounds.
The WARP variant also writes a compare mask into a copied address word after
its last table access. The Python controls cover an ordinary rewrite on both
branch arms and reject escapes through a still-tainted address word.
The shape without a high-word add uses the same 4 GiB address-window assumption
as the vector table lowering. [AMD RDNA 2 ISA, §7.2.1 and §7.5](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)

Run the source-built controls on Linux. The loop and vector table regressions
compare HLSL and manifest hashes captured before the respective changes.

```sh
nice -n 15 python3 -B tests/startup_gfx2hlsl_resheap_branch_check.py
nice -n 15 python3 -B tests/startup_gfx2hlsl_getpc_refusal_check.py
nice -n 15 python3 -B tests/startup_gfx2hlsl_fma_mix_check.py
```

Run the six new WARP draw modes and the existing mixed-FMA WARP dispatch on
Windows. The draws require a readback mismatch from each modified shader.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_(resheap_(vertex_image|vertex_loop|descriptor_word|image_dimensions|fma_mix|embedded_scalar)_warp|fma_mix_dispatch_warp)$'
```
