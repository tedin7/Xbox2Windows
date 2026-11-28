# Sampler descriptors read from the directly indexed sampler heap at a fixed index (translator, stage 1)

Issue #3076, part of #2935 (related #1143). The native compute translator now accepts a sampler that the program loads from
the sampler heap with `s_load_dwordx4` / `s_load_dwordx8` at a constant 16-byte aligned offset and uses with
`image_sample_l` or `image_sample_lz`. It lowers the sampler to `SamplerDescriptorHeap[index]` in the DXIL module. This is
stage 1: the translator, the binding record, the cache, the root signature and `bind()`. Binding a real, mirrored sampler heap
at dispatch time in the runtime is stage 2 and is not part of this change.

## Why

A local start-up capture (counts only; `compile_dxil` at the base commit, 2,689 slot 67 compute requests): 400 compile,
1,830 contain `image_sample*` or `image_gather*`, and in 993 of those at least one sampler is loaded from memory. The
pattern has root flags `0xc00`; the entry SGPR after the resource heap holds the sampler heap base; the program copies it
into an even SGPR pair, sets the high word to 0 and loads one sampler with `s_load_dwordx4` (or two with `s_load_dwordx8`) at
a 16-byte aligned immediate offset; `image_sample_lz` or `image_sample_l` then uses it (95,839 and 20,556 of 127,822
sample and gather instructions). Before this change the native translator left that SGPR `unbound`, so the load was refused
("direct descriptor address outside bound table") and `sampler_index` refused non-literal words ("nonconstant sampler").
There is no compiled-count gain on its own: with the mixed-contract chain of #3023 applied locally (uncommitted probe), 130
requests stop first at such a load (117 `dwordx4`, 13 `dwordx8`). The change removes the next blocker of that chain.

## What changed

- `compute_sampler_heap` (`~0u - 1`) is a second sentinel next to `compute_resource_heap`. Under the existing condition
  (root flags `0xc00`, exactly two user SGPRs left after the root parameters) the SGPR after the resource heap is now the table
  `{table, compute_sampler_heap}` instead of `unbound`. Entry `n` of the sampler heap is title byte offset `16 * n`.
- Direct path, load from the sampler heap (`gfx1013_direct_compute.inl`), refused in this order: a scalar buffer load ("scalar
  buffer load from the sampler heap is not translated"); an SGPR offset ("sampler heap load with an SGPR offset (a
  GPU-selected sampler) is not translated"); any width but `dwordx4` / `dwordx8` ("sampler heap load must read one or two
  samplers"); an immediate offset that is not a multiple of 16 ("sampler heap offset is not a multiple of 16"); an entry
  index above 2046 ("sampler heap index beyond 2046 (2047 is the runtime fallback entry)"). The existing rule that the table
  address has a constant zero high word then applies.
- HLSL path: a load through that base is refused ("sampler heap reads need the direct DXIL path").
- `Emit::binding`: a descriptor from the sampler heap used as an image or buffer is refused ("a sampler heap descriptor is
  read as a resource").
- `sampler_index` returns `SamplerRef {heap, index}`. Four words that are all literals take the static-sampler path as before;
  four contiguous words of one sampler-heap entry make a heap sampler; anything else is "nonconstant sampler". For a heap
  sampler, in this order: gathers ("gather through the sampler heap is not translated"), comparison samplers ("comparison
  sampler from the sampler heap is not translated"), gradient samplers ("sampler from the sampler heap needs image_sample_l or
  image_sample_lz"). Static and heap samplers may appear in one program (the mixed fixture).
- DXIL: `DxilModule::sampler_heap_handle` emits `createHandleFromHeap(i32 218, i32 <index>, i1 true, i1 false)` and
  `annotateHandle` with resource properties `{14, 0}`; the emitter sets module flag `0x80000000` (sampler heap indexing), as DXC
  does for `SamplerDescriptorHeap[i]` (DXC reference: flags `0x80000010` with raw buffers).
- Binding record: `ComputeViewKind::heap_sampler`, one per program, `{heap_sampler, compute_sampler_heap, byte_offset 0,
  register 0, required_bytes 16 * (highest index + 1)}`. The frontend cache accepts the kind with exactly that root value; an
  older decoder rejects it and recomputes. `format_compute_runtime_manifest` returns no manifest for it (root parameter above
  15), so `compile_dxil --requests` publishes `<key>.cs.cso` without `<key>.txt`; the in-process dispatch route refuses the
  kind. `compile_dxil` reports `sampler_heap span=<bytes>`.
- `create_native_compute_pipeline`: `sampler_heap_indexed` adds root flag `0x800` (`SAMPLER_HEAP_DIRECTLY_INDEXED`) and does not
  set `heap_indexed` or `0x400`. `NativeComputePipeline::bind` has a last parameter `ID3D12DescriptorHeap *sampler_heap`: with
  `sampler_heap_indexed` a null heap throws ("a sampler heap is needed"); with views too, the heaps are set as views first, then
  samplers; without it nothing changes. A root without `0x800` and a shader that reads `SamplerDescriptorHeap` fails
  `CreateComputePipelineState` with `E_INVALIDARG` (debug layer: "Shader uses sampler descriptor heap indexing, but root
  signature is missing the D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED flag"), measured on the RTX 3080 with a
  throw-away probe; a root with static samplers and `0x800` with a shader that reads one static and one heap sampler creates
  without a debug-layer message.

## Assumptions for stage 2

The runtime must fill the sampler heap bound for a natively translated compute dispatch so that entry `n` is title byte offset
`16 * n` (index parity), with `dynamic_sampler()` decoding (`src/startup_stubs/translated_sampler_bindings.h`: MIN_LOD and MAX_LOD
applied, which is why only the explicit-LOD forms are translated), and entry 2047 (`translated_sampler_fallback`) holding the
fallback sampler. The translator refuses index 2047, so the fallback is never read. None of this is implemented here; the fixture
builds its heap with `dynamic_sampler()` and the index parity itself.

## Fixture

`tests/native_sampler_heap.cpp`, `tests/native_sampler_heap_check.py`, `tests/cmake/native_sampler_heap.cmake`; tests
`native_sampler_heap_hardware` and `native_sampler_heap_warp` (`SKIP_RETURN_CODE 77`: WARP has no 32-lane waves).

The script writes source-built compute requests (`.c67`, as `tests/native_compute_raw_root_check.py`) with the root
`RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED), DescriptorTable(SRV(t0), UAV(u0))`, group 32x1x1 and
seven user SGPRs (s4 table, s5 resource heap, s6 sampler heap). The instruction words were assembled once with LLVM 15
(`clang -cc1as -triple amdgcn-amd-amdhsa -target-cpu gfx1013 -filetype asm -show-encoding`) and are embedded with one source
comment per instruction; the test needs no LLVM. One SRV+UAV table is accepted by the direct path (no split into two tables was
needed). The programs copy s6 into an even SGPR pair with a constant-zero high word and sample a 4x4 two-mip texture at (1.375,
-0.375), outside [0, 1]:

| Program | Shape | Expected |
|---|---|---|
| SAMPLE | `s_load_dwordx4` at 0x430 (entry 67), `image_sample_lz` | wrap texel (1, 2) of mip 0 |
| TWO | `s_load_dwordx8` at 0x430, two samples | entry 67 wrap, entry 68 clamp (a swapped second group fails) |
| LOD | `image_sample_l`, LOD 1 | MaxLOD 1 picks mip 1; decoys with MaxLOD 0 (bilinear and point) pick mip 0 |
| MIXED | one literal static sampler and entry 67 | static clamp texel and heap wrap texel; one static sampler in the program |

The fixture dispatches on the RTX 3080 with a 128-entry sampler heap made with `dynamic_sampler()`. Every entry but the entries
under test is a decoy (clamp, mirror, border, a bilinear border with a white colour). It compares every lane with a CPU model,
then dispatches once per decoy with the decoy in place of the entry under test and requires the result to change. It also calls
`prepare_direct_compute` twice on the same fresh cache directory (a miss, then a hit) and requires exactly one binding
`{heap_sampler, compute_sampler_heap, 0, 0, 16 * (highest index + 1)}` in the computed and in the cached program, and that
`bind()` without a sampler heap throws.

The script also compiles the refusal shapes and requires each exact text and no `.cso`: offset 0x438, entry 2047, `dwordx8`
at entries 2046 and 2047, `dwordx2` and `dwordx16`, an SGPR offset, a scalar buffer load, a sampler heap entry used as an image
descriptor, mixed literal and heap words in one sampler, a join that replaces the copied base on one path, `image_gather4_lz`,
`image_gather4_c_lz`, `image_sample_c_lz` and `image_sample_d` with a heap sampler, the HLSL profile, and a root with only
`0x400` (today's refusal, unchanged). Each has a control that compiles (entry 2046, `dwordx8` at 2045 and 2046, the branch
without the overwrite, the four operations with a literal sampler).

## Result

Environment: NVIDIA GeForce RTX 3080, Windows SDK 10.0.26100.0 DXC, D3D12 debug layer not enabled in the fixture.

```
ctest --preset dev --no-tests=error -R "^native_sampler_heap_(hardware|warp)$" -V
```

`native_sampler_heap_hardware` passes with 69 `ok:` assertions in the check script (the fixture's own per-mode checks print
`pass:` lines and are counted inside them);
`native_sampler_heap_warp` skips (77: no 32-lane waves). The regression set
`ctest --preset dev --no-tests=error -R "native_|startup_compute_dispatch|compute_heap" -j 6` passes (163 tests, 0 failed, 5
skips: the four WARP tests that already skipped and the new one).

## Limits

- Stage 1 only: no runtime binding of a mirrored sampler heap for natively translated compute dispatch.
- GPU-selected sampler indices (SGPR offset), comparison, gather and gradient sampling through the heap, and containers are
  refused. Roots with only the `0x800` flag (10 requests in the capture) are not covered.
- The fixture samples with point filters and compares exact texels; decoys are checked to change the result (and to run:
  not the output sentinel), bilinear decoys only that.
- Not exercised by a test: the highest accepted entry 2046 in a sample (the edge controls only load it), and `bind()`'s
  "a view heap is needed" throw (no test program reads the sampler heap without a view table).
- A passing fixture supports the behaviour under test; it does not establish that any title shader runs.
