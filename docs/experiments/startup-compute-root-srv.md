# Startup compute root SRV (`srv <i> buf root <param>`) manifest consumer

Refs [#2618](https://github.com/R0Y00/Xbox2Windows/issues/2618),
[#2274](https://github.com/R0Y00/Xbox2Windows/issues/2274).
Follows the compute `global_load_dword` lowering in
[#2715](https://github.com/R0Y00/Xbox2Windows/pull/2715) (ticket `C500-01/031`).

When a translated compute shader loads through a 64-bit root SRV pointer via
`global_load_dword`, `tools/startup/gfx2hlsl.py` emits `srv <i> buf root <param>`
in the compute manifest (`<key>.txt`) and lowers the read to
`ByteAddressBuffer r<i> : register(t<i>)` in the translated descriptor table.

The production startup compute manifest consumer in
`src/startup_stubs/d3d12_compute_dispatch.inl` and command-list tracking in
`src/startup_stubs/d3d12_xs_device.cpp` now consume `srv <i> buf root <param>`:

- Command-list slot 43 (`SetComputeRootShaderResourceView`) records the guest
  address and a bound bit per compute root parameter on the open recording while
  a compute root signature is bound. Binding a different compute root signature
  in slot 33 (`SetComputeRootSignature`), closing the list (slot 9), or
  resetting the list (slot 139) clears the recorded compute root SRV bindings;
  re-binding the same compute root signature preserves them.
- `load_compute_translation` parses `srv %u buf root %d%n` (requiring an exact
  whole-line match, `param >= 0`, and sequential `i == tr.srvs.size()`), and
  rejects descriptor-word entry values (`stride`, `descword3`) that reference a
  root SRV slot.
- At translated compute `Dispatch` (slot 14), `compute_root_srv_buffer` resolves
  each `buf root <param>` input by verifying that `<param>` in the bound compute
  root signature has `D3D12_ROOT_PARAMETER_TYPE_SRV`, was set by slot 43 since
  the compute root signature was bound or reset, has a non-null 4-byte-aligned
  guest address at a 16-byte-aligned offset (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`)
  inside either a recording upload allocation (slot 148) or a live aliased
  placed buffer in `COMMON` state (slot 133) with at least one whole dword
  remaining, and does not alias an output UAV buffer. The resolved range is
  bound as a raw `DXGI_FORMAT_R32_TYPELESS` `ByteAddressBuffer` view from that
  offset to the end of its backing allocation or placed buffer (up to 16 MiB).

## Verification

`tests/startup_compute_root_srv_dispatch_check.py` (registered in
`tests/cmake/startup_compute_root_srv.cmake`) drives source-built gfx1013
compute `global_load_dword` programs (1-root-SRV and 2-root-SRV variants)
through slot 67, slot 33, slot 43, slot 37/39, slot 35, and slot 14 in
`startup_stream_consumer` with `X2W_D3D12_DEBUG=1` on both `warp` and `hardware`
(`AMD Radeon RX 9070 XT`):

- **Single and two-root-SRV dispatches:** 3x2x2 groups of 2x3x4 threads (288
  active lanes, 320 total output words) load at offset `0` and `+128` bytes from
  one or two distinct placed input buffers (`srv 0 buf root 2` and
  `srv 1 buf root 3`), including re-binding the same compute root signature,
  non-zero byte offsets into placed buffers (`+16` bytes on root SRV 0, `+32`
  bytes on root SRV 1) and recording upload memory (`+16` bytes via slot 148),
  and out-of-bounds loads (`v4 + 128u` past the bounded 1,280-byte backing)
  returning `0` rather than reading adjacent `0xdeadbeef` / `0xfeedface`
  neighbor words in the same 64 KiB page.
- **Refusal controls:** malformed manifest lines (`root -1`, trailing tokens,
  non-sequential SRV index, `udsrc ... stride` on a root SRV), unset root SRV
  slot (first or second), root SRV cleared by `Reset` or by binding a different
  compute root signature, wrong root parameter kind (`D3D12_ROOT_PARAMETER_TYPE_CBV`
  at `<param>`), null guest address, unaligned guest address (`addr + 2`),
  non-16-byte-aligned interior resource offset (`addr + 4`), unmapped guest
  address between placed buffers, non-aliased copied placed buffer, and
  input-aliases-output-UAV hazard all refuse cleanly with their expected log
  reasons and leave all 320 output sentinel words (`0xc5c5c5c5`) untouched.
- **Negative controls:** skipping `Dispatch`, swapping `srv 0 buf root 2` and
  `srv 1 buf root 3` in the manifest, or dropping the `+ 128u` offset in the
  translated HLSL changes all 288 active output words as predicted.

Reproduce on Windows:

```powershell
cmake --build --preset release --target d3d12_xs_shim startup_stream_consumer
ctest --preset release -j1 --no-tests=error -R '^startup_compute_root_srv_dispatch_(warp|hardware)$' -V
```

What remains a hypothesis: retail title compute dispatches that combine root SRV
pointers with unplaced host memory outside slot-133 placed buffers and slot-148
recording upload allocations are not yet modelled here and remain refused until
a title call site establishes their backing contract.
