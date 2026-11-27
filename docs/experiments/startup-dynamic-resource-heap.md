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

## Formatted loads

Formatted dynamic buffer loads use the same raw views as ordinary heap loads.
The snapshot places the record views first, a raw format-information view at
index `heap_count`, then the translated descriptor tables. Each information
entry contains the record stride and descriptor word 3. Missing or unsupported
records have a zero word 3, so every selected component reads as zero.
The translation reads FORMAT and DST_SEL from that word; see the AMD RDNA 2
Instruction Set Architecture, sections 8.1.2, 8.1.5 and 8.1.8.

Run the Python translation controls with
`python3 -B tests/startup_gfx2hlsl_heap_format_check.py`. These controls do not
exercise the Windows runtime or a GPU.
