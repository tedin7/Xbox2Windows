# Dynamic compute heap records

Set `X2W_DYNAMIC_RESOURCE_HEAP=1` for the translator and runtime. The translator tries its existing lowering first. For an otherwise refused four-dword resource-heap load, it can capture a proven uniform SGPR byte offset and use `ResourceDescriptorHeap[offset / 32]` for raw buffer reads with explicit byte addresses. The dynamic shader uses Shader Model 6.6. The runtime requires resource binding tier 3.

The runtime snapshots the bound logical heap as raw buffer SRVs in a private heap for the dispatch. It appends fixed table bindings after the title's records and passes the original record count as a root constant. The shader returns zero for a misaligned offset or an index at or beyond that count. All-zero records bind null SRVs. The runtime refuses unreadable entries, entries it cannot bind as buffer ranges, heaps beyond the translated capacity, and dynamic SRVs that alias an output. Descriptor offsets captured inside loops, image records, indexed buffer addresses that need a descriptor stride, and indirect dispatch remain outside this path.

From the patch based on `34da218c7`, the synthetic lowering check passed on Linux x86-64 with Python 3.14.8:

```text
python -B tests/startup_compute_dispatch_heap_dynamic_check.py
```

On Windows, the WARP test publishes a synthetic slot-67 request, compiles it and compares all 320 readback words to a CPU reference. Group X selects record 0, record 2, or an index equal to the count. A second program changes the offset shift, selects a null record, and must fail the original reference at word 24. The test also checks untouched output before translation and when the runtime switch is off. WARP execution is pending.

```powershell
ctest --preset dev --no-tests=error -R '^(startup_gfx2hlsl_heap_dynamic|startup_compute_dispatch_heap_dynamic_warp)$' -V
```
