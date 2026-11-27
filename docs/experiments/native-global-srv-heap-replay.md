# GPU-selected global SRV heap slots: capture and replay of view types

Issue #2908, branch `cpp-migration`. Builds on the heap read in the direct C++ DXIL path from #2912
(`ResourceDescriptorHeap[slot]` for a raw buffer read, `src/gfx1013_direct_compute.inl` `heap_resource`).

## What changed

- The direct compute translation takes a view contract for a GPU-selected heap slot (`x2w::HeapViewContract`:
  `none`, `raw_buffer`, `other`). An SMEM descriptor load from the resource heap is refused unless the contract is
  `raw_buffer`; without a contract the existing text stays: "direct resource-heap load requires the selected GPU slot and
  its view type". A scalar buffer load from the heap base, a heap descriptor load with an immediate offset or a constant
  index (both could address another slot than the captured selection), and a numeric use of a GPU-selected slot's
  stride in the direct path are refused with their own texts. The contract is part of the frontend cache key when it is
  not `none`.
- Byte-address rewrite (#3023): under `raw_buffer` the direct path tracks the rewrite of the loaded heap descriptor that
  `byte_descriptor_field` already models for table descriptors (`s_bfe_u32 ... 0xe0010` stride, `s_max_u32 ... 1`,
  `s_mul_i32` byte count, `s_and_b32 ... 0xc000ffff`, `s_and_b32 ... 0xcfffffff`, `s_or_b32 ... 0x30000000`, word 0
  copied). A raw load through the rewritten descriptor reads the same `ResourceDescriptorHeap[slot]` raw view. The words
  must all come from one heap load (same pc). As a precaution, a backward branch whose loop contains that heap load
  refuses the window, because the loop would run the load again for another selection. Loops elsewhere in the program
  do not refuse it. The caller's contract: every slot the
  program can select is a raw view of exactly `num_records * max(stride, 1)` bytes; the translator does not check that,
  nor the swizzle bits or the other word 3 fields. Without the contract the first refusal stays as before.
- Modules that read `ResourceDescriptorHeap` now also set shader flag `0x40000000`, as DXC does for the HLSL reference.
  Without it the validator refuses the module ("Flags must match usage. Validation failed."), measured by removing the
  flag and running the hardware test.
- `src/startup_stubs/native_global_srv.h` writes and reads version 2 of the view metadata, which adds the buffer size:

  ```
  x2w-global-srv-views 2
  select <slot>
  need <type>
  slot <n> <type> <stride> <bytes>
  dword <n> <channels>   (after the slot lines; typed buffers of 32-bit channels only, #3023)
  ```

- `compile_dxil ... gfx1013-container-cs` reads an optional `req_<key>.views` next to the `.cdx` (single file and
  `--requests`). It decodes the file, refuses a selection whose slot holds another view type than `need`, refuses a `need`
  other than `raw_buffer` or a one-channel 32-bit `typed_buffer` (#3023), translates with the matching contract and
  prints `global_srv select=<n> need=<type>`. Every refusal happens before the `.cso` is published. Without the file a heap-reading shader is refused as before. The
  `.views` file is not part of the `.cdx`/`.root` content key: it describes one captured dispatch.

## Meaning of "native recording" and "runtime path" here

The issue's reproduction names `native_recording_check` and `native_dispatch_states_warp`, which test
`x2w_startup::NativeRecording` (descriptor-table pages) and `x2w_startup::NativeDispatchStates` (dispatch barriers). On
`cpp-migration` only tests use these classes. The D3D12 stand-in translates slot 67 requests natively since #2958
(opt-in `X2W_NATIVE_SHADERS`) through `translate_compute` without a heap view contract, so a GPU-selected heap read is
still refused there; passing captured views into that path is follow-up work. The fixture's `record`
mode records the dispatch with both classes and applies the view guard before `Dispatch`; that is the native recording
path checked here, and its unchanged UAV sentinels are the "runtime path" sentinels of the issue. The startup runtime
itself is not covered.

The slot is selected by GPU data. The `record` mode reads the selector back from its GPU buffer before recording the
dispatch (one copy to a readback buffer and one fence per recorded dispatch) and uses that value as the captured
selection, so the guard checks the slot the GPU input actually selects, before the shader runs.

## Fixture

`tests/native_global_srv_heap_replay.cpp` with `tests/native_global_srv_heap_replay_check.py`. A source-built compute
container (two root inputs: a descriptor table in s[0:1], the descriptor heap in s[2:3]; instruction words assembled with
LLVM 15 `clang -cc1as -triple amdgcn-amd-amdhsa -target-cpu gfx1013 -show-encoding`) loads the selector from t0, loads
the descriptor of `ResourceDescriptorHeap[selector]`, reads 64 words through it and stores them to u0. Heap slots: 0 and
1 raw buffers with different data, 2 a Texture2D, 3 a typed buffer. u0 holds 72 words; the last 8 and, on a refused
selection, all 72 must keep the sentinel `0x69696969`.

The check runs, in order:

1. `compile_dxil --requests` on the request without `.views`: refused with the heap text, no `.cso`.
2. `record` for selections 0 and 1: fenced output equals the selected slot's data, sentinels intact, `.views` captured.
   Selections 2 and 3: "global SRV slot 2 has view type texture, the shader needs raw_buffer" (and `typed_buffer` for 3)
   before the dispatch, nothing dispatched, sentinels intact.
3. Discrimination control: `record --no-guard` on slot 0 expecting a refusal sees the changed sentinels (exit 3). A
   texture slot is not dispatched without the guard: reading a texture descriptor as a raw buffer is undefined in D3D12.
4. `compile_dxil --requests` on each captured selection: 0 and 1 compile, 2 and 3 are refused with the slot/type text
   and publish no `.cso`.
5. `replay` of the compiled `.cso` for 0 and 1 matches the CPU reference; the `.views` of 2 and 3 are refused before the
   dispatch with sentinels intact.
6. Mutation control: replaying selection 0 against the slot 1 reference reports a mismatch.
7. Negative controls on both `compile_dxil` and `replay`: slot 0 rewritten to `texture`, no `need` line, no `select`
   line, no line for the selected slot, `select 7`, and (replay) a missing `.views` file. Each gives an explicit
   diagnostic, no `.cso`, and intact sentinels.
8. Shader shapes refused by `compile_dxil` even with valid metadata: a constant heap index, a heap descriptor load with
   an immediate offset, a stride read used as a number (`unbound numeric SGPR s24`), and four byte-address rewrites the
   raw slot view cannot stand for: a wrong stride mask (`unbound numeric SGPR s21`), word 0 from a second heap load
   (`non-contiguous resource descriptor`), a store through the rewritten descriptor (`a GPU-selected heap slot is read
   as a raw buffer SRV only`) and a never-taken backward branch to (or before) the heap load (`heap byte window loaded
   inside a loop`).
9. The byte-address rewrite: a stride read that nothing uses compiles, a never-taken backward branch after the heap
   load (the loop does not contain the load) compiles, and the program with the full rewrite is recorded
   for selections 0 and 1 (output equals the slot's data), compiled by `compile_dxil --requests`, replayed with the
   same result, and replaying selection 0 against the slot 1 reference reports a mismatch.

## Result

```
cmake --preset dev
cmake --build --preset dev
ctest --preset dev --no-tests=error -R "^native_global_srv_heap_replay(_warp|_hardware)?$" -V
```

- `native_global_srv_heap_replay_hardware`: passed, 70 assertions (NVIDIA GeForce RTX 3080, driver 32.0.16.2124,
  Windows 11 build 26200, MSVC toolset 14.51.36231, Windows SDK 10.0.26100 DXC, D3D12 debug layer not enabled).
- `native_global_srv_heap_replay_warp`: skipped (77), "adapter has no 32-lane waves (4..4)". The native module declares
  `WaveSize=32`; WARP offers 4-lane waves only. The skip is not evidence for the fixture.
- `ctest --preset dev --no-tests=error -R ^native_ -j 1`: same result as before the change apart from the two new tests
  (one known failure, `native_wave_request_hardware`, #2915, addressed by #2928).
- Revert control: with the heap refusal restored for every contract, `record` for selection 0 exits 4 with "direct
  resource-heap load requires the selected GPU slot and its view type".
- Byte-address rewrite (#3023): `native_global_srv_heap_replay_hardware` passed, 83 assertions, on the same machine and
  driver. The check run against `compile_dxil` and the fixture built at `999e17cd` (before the change) fails at
  `stride_read` (old text "a GPU-selected heap slot stride read is not translated with a view contract") and, with the
  shapes skipped, at "an unused stride read compiles" (the run stops there, so the full-rewrite record step was not
  reached on that build). `ctest --preset dev --no-tests=error -R
  "native_|startup_compute_dispatch|compute_heap" -j 6`: 147 tests, 142 passed, 4 skipped (77, WARP), 1 failed
  (`native_cvt_pk_u8_hardware`, failing before the change as well). `compile_dxil --refusals` on a local start-up
  capture gives the same output as before (it translates without a contract).
- Loop guard narrowed: the first version of the guard refused the window for any backward branch in the program.
  With the contract forced locally (an uncommitted switch), it refused 921 of the 2,689 requests of the same capture.
  With the guard limited to loops around the heap load, none of these is refused for a loop; 906 then stop at a typed
  load (`buffer_load_format_*`) through the heap descriptor. After the change, the test passes with 84 assertions. The
  run against the earlier commit `7582c01d` fails at the `window_loop` shape (old text "heap byte window in a program
  with a backward branch"). The run stops there.
- Formatted loads through the byte window (#3023): `tbuffer_load_format_*` through the rewritten descriptor now reads
  the same raw view as `buffer_load_dword` (`src/gfx1013_compute_mtbuf.inl` picks `byte_window_srv` when word 1 is the
  stride-cleared word). Before, it was refused with "non-contiguous resource descriptor". On the local capture with the
  raw contract forced and a typed heap load prototyped (both uncommitted), 861 of the 2,689 requests stopped first at
  this refusal; with the change they move on to other, already-known refusals. `native_global_srv_heap_replay_hardware`
  passes with 94 assertions (record, compile and replay of a 32_UINT `tbuffer_load_format_x` through the heap window
  for selections 0 and 1, the cross-slot mismatch control, the loop guard for the formatted load, and a compile-only
  8_8_8_8_UNORM case). `native_mtbuf_hardware` repeats its 11-format value check through a byte window of its table SRV
  and matches, including the element at the end of the view that returns 0 (`tests/native_store_replay.cpp` accepts
  the window binding as the same raw view). `native_direct_container` compiles both formats through a compute table
  window and refuses the window without the byte-count product. Against `837728dd` the heap check fails at the
  `twindow_loop` shape (text "non-contiguous resource descriptor").
- Typed heap loads (#3023): under the new `HeapViewContract::typed_dword` a one-channel typed load
  (`buffer_load_format_x`, an `idxen` element index) through the GPU-selected heap descriptor reads
  `ResourceDescriptorHeap[slot]` as a typed buffer of one U32 component (binding kind `heap_typed_srv`, cache identity
  `/heap-view/typed_dword`). `.views` gains a `dword <slot> <channels>` line for each typed buffer slot of 32-bit
  channels; `compile_dxil` and the fixture derive the contract from `need typed_buffer` when every captured typed
  buffer slot is a one-channel 32-bit buffer (`x2w_startup::global_srv_contract`). `record` takes
  `--need typed_buffer`. `native_global_srv_heap_replay_hardware` passes with 122 assertions: the typed load is
  recorded for slot 3 (`R32_UINT`), compiled, compiled again from the frontend cache, and replayed (output equals the
  slot 3 data); replaying it against slot 0 reports a mismatch; selections 0 and 2 are refused before the dispatch;
  ten compile refusals cover a two-channel load, a raw load and a `tbuffer` load under the typed contract, the typed
  load under the raw contract, and metadata without the `dword` line, with a channel count of 5, with a `dword` line for
  a raw slot, a duplicate or an early `dword` line, or with another selection. On the local capture (counts only) all 906 typed heap loads behind the raw contract have this
  one-channel `idxen` shape; with `typed_dword` forced by an uncommitted switch, 895 of those programs then stop at a
  raw heap read of another slot: they read the heap both ways, which one captured selection does not cover.

## Limits

- Only raw buffer heap reads (`raw_buffer`) and one-channel 32-bit typed buffer heap reads (`typed_dword`) are
  translated, one class per program; other view classes are refused with a named diagnostic.
- Under `typed_dword` the translator sees no descriptor values. The caller guarantees that every selectable slot is an
  `R32_UINT`, `R32_SINT` or `R32_FLOAT` buffer view bound as `R32_UINT` over the same element range, that the guest
  descriptor has stride 4, a 32-bit data format, identity `dst_sel_x`, swizzle and `add_tid` disabled, and that its
  `num_records` is the view's element count (reads past it return 0). `.views` records the resource size, not the
  view's first element or element count, so `global_srv_contract` checks necessary conditions only. Only `R32_UINT`
  slots are exercised; `R32_SINT` and `R32_FLOAT` are accepted but untested. No runtime binds a typed heap slot yet.
- The selected slot's index is per-lane VGPR data in general; reads stay inside the raw buffer view, which returns 0
  out of bounds. No static index bound is checked.
- Replay regenerates the slot contents deterministically from the slot index; `.views` captures type, stride and size,
  not contents.
- In the direct DXIL path a GPU-selected slot's stride is never a number: it is tracked symbolically and only the
  byte-address rewrite consumes it. The fixture's descriptor words are never evaluated, so a stride above 1 and the
  size rule of the contract are not exercised. The HLSL path's `heap_stride_srv` (`src/gfx1013_compute_native.cpp`,
  used by `tests/native_compute_grid.cpp`) is unchanged.
- A formatted load (`tbuffer_load_format_*`) bounds its element with the size of the desktop raw view
  (`getDimensions`), not with the guest window's byte count, and returns 0 for the whole element when it does not fit.
  A runtime that binds a byte window, as a table view or as a GPU-selected heap slot, must create the raw view with
  exactly the guest window's byte count. The descriptor's DST_SEL (word 3) is not applied: only the identity channel
  order is translated, and it is not checked. In the startup dispatch, the in-process route
  (`translate_compute_natively`, `src/startup_stubs/d3d12_compute_dispatch.inl`) binds only `raw_vector_srv` inputs
  and falls through to the manifest route for a byte window. A published manifest names a table byte window as
  `srv <i> buf table <param> <byte>`, as for the raw byte window; the dispatch then binds the table descriptor as a raw
  view of
  `max(stride, 1) * num_records` bytes (`indirect_table_buffer`, `src/startup_stubs/d3d12_draw.inl`), the guest
  window's byte count, or of the title's own view size for a view token. No fixture runs this manifest path for a
  window.
- Multi-channel typed heap loads, typed loads under the raw contract and scalar loads through the rewritten descriptor
  stay refused. The startup runtime
  does not pass a contract yet, so none of this reaches a title run (#3023).
- The captured `select` must be the slot index the dispatch addresses; producing it from a title capture is outside this
  fixture.
