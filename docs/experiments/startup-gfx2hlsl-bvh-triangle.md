# `image_bvh_intersect_ray` triangle nodes

`tools/startup/gfx2hlsl.py` now lowers the 64-byte triangle node forms (kinds 0..3) of the 32-bit
`image_bvh_intersect_ray` (RDNA 2 MIMG opcode 230), in addition to the FP32 and binary16 box nodes.

## Node and test

A triangle node is 64 bytes: five `float3` vertices at bytes 0..59 and one ID dword at byte 60. The
kind selects one of the four 8-bit maps in the ID dword. Each map holds `i` in two bits and `j` in
two bits; the third index is `(3 - i - j) & 3`. The kind also selects the base vertex triple, which
is the table in Mesa 25.1.0 RADV's `radv_bvh_triangle_node` (MIT):

```text
kind 0: 0 1 2
kind 1: 1 3 2
kind 2: 2 3 4
kind 3: 2 4 0
```

The test is the watertight ray/triangle formulation from Woop, Benthin and Wald, as RADV uses it.
The helper projects the vertices onto the dominant ray axis, shears them, computes the three
barycentric numerators `u`, `v` and `w`, the determinant `u + (v + w)` and
`t = u*az + v*bz + w*cz` in binary32. A hit requires all three barycentrics to share a sign and
`t` to have the matching sign. The instruction does not test `ray_extent` for triangles.

## Return and descriptor

Triangle return mode 1 (bit 120 of the BVH T#, bit 24 of descriptor DWORD 3) returns the four words
`{t, determinant, v, w}` as float bit patterns. A miss is `{+inf, 1.0, 0, 0}`. For triangle nodes,
return mode 0 and every other reserved word-3 bit are refused by name before any HLSL is emitted.
Box nodes (kinds 4 and 5) ignore bit 24 because one T# can cover both node types; every other
word-3 bit is still refused for them.

The accepted slice uses:

- compute stage only;
- `image_bvh_intersect_ray v[4], v[11], s[4]` with the packed address form;
- a constant inline T# built by four SGPR constants, type 8, nonzero encoded base, box growing zero,
  size field (a count minus one in 64-byte units) whose decoded extent covers the node; `size=0` is
  a one-unit span, and the field stays below 4 GiB;
- a constant node pointer whose low three bits are 0..3 and whose address is inside the descriptor
  size and the 48-bit absolute address space;
- either value of the sort bit; it is ignored for triangles.

The manifest still names the allocation with `srv <i> bvh base 0x<base> size 0x<size>`, using the
decoded byte address for `base`. The host binds `(size + 1) * 64` bytes. That binding remains
outside this change.

## Refused by name

Return mode 0, every node kind other than 0..3, `image_bvh64_intersect_ray`, A16, non-sequential
addresses, a pixel or vertex stage, a node pointer that is not a straight-line constant, a T# loaded
at run time, a type other than 8, nonzero box growing, an absolute (zero) base, a size that does not
cover the node, size bits above 4 GiB, a 48-bit node wrap, and every reserved T# field. The checks
assert that a refusal emits no helper call and no destination VGPR assignment.

## Evidence

```sh
python3 -B tests/startup_gfx2hlsl_bvh_triangle_check.py
python3 -B tests/startup_gfx2hlsl_bvh_triangle_check.py dispatch <dxc> <offscreen_check> warp
```

The check reuses the 119 triangle witnesses in `tests/gfx1030_bvh_box/check.py` and the independent
model in `tests/gfx1030_bvh_box/bvh_model.py`. These witnesses come from the CPU model (#2921), not a
hardware capture; the dispatch shows that the HLSL matches the model, not RDNA 2 hardware. It covers
all four kinds, each of the four ID map bytes, edge and vertex hits, outside and behind rays, zero
direction and negative zero, `t = 0`, `extent = 0`, 1024x scaling in both directions, degenerate and
malformed nodes, and the unknown kinds 6 and 7 (instance and procedural in the model).

The dispatch writes 71 unique triangle images at 64-byte offsets, runs the emitted helper for all
119 witnesses, and compares all 476 result words. It passes on WARP (`1414:008c`, driver
`10.0.26100.10068`) and on an AMD Radeon RX 9060 XT (`1002:7590`, driver `32.0.32029.2010`), in
both the dev and release presets, with `dxc -Gis` matching the runtime's strict-IEEE compile. Four
mutants change the determinant sign, the ID shift, the `u` term and the miss words; each must change
at least one witness. The existing box32/box16 tests still pass 6/6 in both presets.

This is a translator slice. It does not bind the node buffer on a D3D12 device, does not execute the
native gfx1030 probe, and does not establish title behavior. The runtime manifest loader does not yet
consume the `bvhbase` line, so a real dispatch is refused rather than bound at a wrong address.

