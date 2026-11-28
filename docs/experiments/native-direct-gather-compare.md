# Comparison gather in the direct C++ image path (#2969)

`image_io` in `src/gfx1013_direct_compute.inl` now lowers `image_gather4_c_lz` (MIMG opcode 79), which
`tools/startup/gfx2hlsl.py` lowers as `op_image_gather4_c_lz`. Before, the direct frontend refused it with
`direct image opcode or descriptor modifiers`, while the same program with `image_sample_c_lz` compiled (the fixture
keeps that control).

## The form

- 2D and 2D-array images with `dmask` 0x1. The address is the reference, then u and v (and the slice), as for
  the comparison samples of this path.
- The lowering is DXIL `textureGatherCmp` (opcode 74). It uses channel 0, because gather4_c compares the resource's
  red component, and takes the reference as its compare value. Like `image_gather4_lz` here, the gather reads the
  view's first mip: DXIL gathers have no LOD operand, so the static sampler is fixed to mip zero.
- The sampler is a constant comparison sampler under the existing field checks of `sampler_index` (from
  `pixel_aot.comparison_sampler_state`).
- The four results come back in the order DXIL returns them, the same assumption `image_gather4` and
  `image_gather4_lz` already make in this path. A console readback is not part of this change.
- This is wider than the `gfx2hlsl.py` contract, which takes the form only in pixel stages, on 2D images and with
  a bilinear comparison sampler (`pixel_aot.check_gather_sampler`). Here compute stages and 2D arrays are accepted,
  as for the comparison samples of this path, and so is the point filter of the existing comparison field checks.
  A gather reads the same four texels whatever the filter (D3D11.3 gather rule; not measured on the console).

Refused by name (`direct image opcode or descriptor modifiers`): other dmasks, cube and 3D images,
`image_gather4_c` with implicit LOD, `_c_l` and `_c_b`. D16 is refused at decode
(`unsupported image dimension or D16`). Offset forms keep their existing refusal (not covered by the fixture).

## Fixture

`tests/native_gather_compare_check.py <compile_dxil> <sdk bin dir> <native_recording_check>`
(`ctest --preset dev --no-tests=error -R '^native_gather_compare_warp$' -V`). The programs reuse
`comparison_program()` of `tests/native_direct_container_check.py`: clamp X/Y, point with TRUNC_COORD, LESS.

- A compute container compiles. `dxc -dumpbin` shows one `textureGatherCmp` with zero offsets, channel 0 and a float
  compare value, and no `textureGather` or `sampleCmp`.
- dmask 0x2, a cube image, a 3D image, D16, `image_gather4_c`, `_c_l` and `_c_b` are refused by message; a 2D
  array compiles. The module check traces each float operand to its register: u from v1, v from v2 and the
  compare value from v0, the reference.
- `native_recording_check --graphics-sampler` creates, on WARP, the graphics pipeline of
  `native_graphics_comparison_pso_warp`, with its comparison sample replaced by this gather. It runs cold and from
  the cache, and checks the comparison and regular static samplers.

Measured on 27.11.2026, `cpp-migration` `b260984f` plus this change, Windows 10.0.26100.9711: all three checks passed
(the pipeline on WARP, Microsoft Basic Render Driver 10.0.26100.9711). The gather results themselves are not read
back.
