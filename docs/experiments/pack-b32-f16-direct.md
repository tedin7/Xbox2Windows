# v_pack_b32_f16 in the direct C++ DXIL path

Covers the lowering of VOP3 `v_pack_b32_f16` (0x311) in `src/gfx1013_direct_compute.inl` (Fixes #2909) and
what was and was not verified.

## Semantics

- For each of the two sources, `op_sel` bit 11 (source 0) or 12 (source 1) picks the low or high half.
- abs clears bit 15 of the half, then neg toggles it, so neg after abs.
- D[15:0] is source 0 and D[31:16] is source 1.
- Only `lshr`/`and`/`xor`/`shl`/`or` on i32 are emitted. Signed zero and NaN payloads keep their exact bits.
- Both sources are read before the destination is written, so the destination may overlap a source.
- VGPR and SGPR sources are supported.

## Refused forms

Each is refused with a named diagnostic, with no approximation: clamp, omod, a destination `op_sel` bit (14),
`op_sel` bit 13 (third source), abs/neg on the third source, inline constants, a literal source.

## Linux IR verification

`python -B tests/native_pack_b32_f16_check.py ir <native_pixel_ir_dump>` evaluates the emitted IR and compares it
with `ref()` of `tests/startup_gfx2hlsl_pack_b32_f16_check.py`: 261 results match (issue words, control, 64
`op_sel`/abs/neg forms x 4 value pairs, overlap, SGPR sources, refusals). The swapped-half and neg-before-abs
variants differ from the emitted bits. The dumper was built on Linux from the C++ sources with the Wine d3d12
headers and a stub `wrl/client.h`.

## Windows result (review of PR #2917)

Windows 11 build 26300.9974, AMD Radeon RX 9070 XT, driver 32.0.32029.2010, MSVC 19.44.35221, `release`
preset, Windows SDK 10.0.26100 DXC, D3D12 debug layer not enabled.

- `native_pack_b32_f16_ir`: passed.
- `compile_dxil ... gfx1013-container-ps` on the issue words: rc 0, and `dxc -dumpbin` reads the `.cso`.
- `native_pack_b32_f16_hardware`: not verified. `native_store_replay --direct` exits `0xC0000005` with no output,
  the crash tracked in #2915.

## GPU readback: no run yet

`native_pack_b32_f16_check.py gpu` has not produced a result on any machine. Workgroups are 128 words apart so
that the 66 results of one group (64 forms, the SGPR case, the overlap case) never share a word with another
group; `group_offsets` asserts this. There is no WARP variant because `native_store_replay` takes the default
adapter. A hardware result is still needed once #2915 is fixed or on an adapter that does not crash.
