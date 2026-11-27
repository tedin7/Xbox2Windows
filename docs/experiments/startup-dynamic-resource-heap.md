# Dynamic resource-heap records

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

## Graphics 2D textures

The graphics translator also lowers an eight-word texture descriptor loaded
through a 32-byte heap offset derived from a uniform index. The descriptor
and its uses must stay in one block. A closed readfirstlane loop with one to
four texture operations lowers directly to each lane's index when its scalar
temporaries are dead at the exit. Implicit samples still require the existing
EXEC and wave-uniform control-flow proof.

The manifest line `dynamic_heap texture2d` selects the 2D snapshot layout.
Each readable compatible record becomes a float-compatible 2D texture SRV.
Unreadable or unsupported records use null SRVs. A dedicated null at the
logical record count receives every out-of-range lookup. Fixed SRV and UAV
bindings follow that slot. Indirect draws still report
`dynamic_resource_heap_unavailable`.

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

### Graphics texture fixture

The Python translation and refusal checks passed on Linux for the change
based on `741f301fa`. The raw graphics and four compute heap fixtures matched
that revision's HLSL and manifests byte for byte.

```sh
python -B tests/startup_gfx2hlsl_dynamic_heap_check.py
python -B tests/startup_gfx2hlsl_dynamic_heap_check.py binary llvm-mc-19 llvm-objcopy-19
python -B tests/startup_gfx2hlsl_heap_record_check.py --llvm-mc llvm-mc-19
```

`startup_gfx2hlsl_dynamic_heap_warp` also draws four distinct 2×2 textures
using a wave-uniform branch and then a flattened per-pixel loop. It compares
all 64 pixels per draw with the CPU reference, including null lookups,
implicit samples and texel loads. The index and bounds mutants must mismatch.
Windows readback and runtime execution are pending.
