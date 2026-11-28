# Buffer-stride condition code in direct compute

The direct DXIL frontend can carry a guest buffer descriptor symbolically
while calculating SCC from its bound stride. It recognizes a 14-bit stride
extraction from descriptor word 1 and the subsequent unsigned max with 1. The
first sets SCC when the extracted stride is nonzero. The max sets SCC when its
first operand wins strictly, so both operand orders have meaningful tests.

This applies to fixed, unambiguous buffer SRV slots covered by a declared
descriptor-table range. The compiler adds one stride word per descriptor,
after the regular root constants. Native dispatch and the published manifest
both use the stride from the input view the runtime has already resolved. The
symbolic descriptor fields still carry their slot identity for later buffer
addressing. A dynamic heap or CBV/UAV slot, unproven stride source, and numeric
reads of other symbolic fields retain their refusals.

The [source-built fixture](../../tests/native_descriptor_scc_check.py) checks
the layout and cache replay, then dispatches with strides of 0, 1, 4 and 16.
The runtime fixture changes guest descriptor fields between runs of the same
shader on the in-process and published paths. It compares the three status
words, the neighboring root constant, and every untouched output word.
The isolated replay also verifies 14-bit extraction and refuses to dispatch
if the stride input is missing. The fixtures use the repository's MIT license.

```powershell
cmake --build --preset release --target compile_dxil native_store_replay startup_stream_consumer
ctest --preset release --no-tests=error -R '^(native_descriptor_scc|startup_compute_descriptor_scc)_' -V
```

The direct shaders require Shader Model 6.6 and 32-lane waves. An available
adapter without wave32 reports a skip. The instruction behavior follows the
[AMD RDNA 2 ISA, revision 1.0](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture).
