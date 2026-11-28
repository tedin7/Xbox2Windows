# The indexed rectangle rule and a 12-byte vertex stream (#3131)

Issue #3131 reports that an indexed UI gradient draw leaves a sharply bounded region of its colour target
uncovered, with a diagonal boundary, and that the draw's recording-time vertex read uses a 12-byte stride. The
reported rule was checked against source-built input: an eight-vertex quad and edge ring, 12-byte vertex
records with XY at +4, an indexed triangle list, a mask-controlled pixel shader and an RGBA8 target, through
the runtime entry (`startup_stream_consumer`), plus a fixture for the shared rectangle-expansion stage alone.

## What the runtime draw measured

`tests/startup_rect_gap.cpp` (draw) and `tests/startup_rect_gap_check.py` (input, model and comparison) bind a
placed 12-byte-record stream twice, slot 0 as R32_FLOAT (the mask at +0) and slot 1 as R32G32_FLOAT at +4, and
draw four triangles: records 0-1-2 and 0-2-6 (the edge ring), then 3-4-5 and 3-5-7 (the inner quad), so the
same eight records serve an R16 indexed triangle list and a six-vertex rectangle list. A mask-controlled pixel
shader paints the quad red and the ring green; a constant-red pixel shader is the coverage control. The target
is cleared blue, and the check compares all 32 pixels with a point-in-triangle model of the two triangles.

Every mode covered exactly the pixels the model expects, on WARP and through the `hardware` backend:

| mode | draw | result |
|---|---|---|
| `triangle` / `triangle-copy` | R16 indexed list, placed stream / unregistered stream (recording-time copy and index scan) | 12 red, 20 green |
| `rect` / `rect8` / `rect8-copy` | six- and eight-vertex rectangle lists, expanded by the vertex stage | 12 red, 20 green |
| `rect8`, `triangle` with the constant shader | coverage only | 32 red |
| `triangle-rotated` / `rect-rotated` | the same rectangles turned by 15 degrees | 12 red, 18 green, 2 clear |
| `none` | no draw | 32 clear |

The check also asserts the bound bytes (8 records of 12 bytes, then the 12 indices), the input-layout request
(`TEXCOORD 0 41 0 0 0 0` and `POSITION 0 16 1 4 0 0`), and the generated vertex stage's two loads,
`a = (vid + ud1) * ud3 + 4;` for POSITION and `a = (vid + ud1) * ud4 + 0;` for the mask. With
`X2W_TEST_RASTERIZER=back` (back-face culling, clockwise front) the picture is unchanged, and with `back-ccw`
(counter-clockwise front) every triangle is culled and the target stays clear, so the winding survives the
expansion. The described diagonal gap did not appear in any of these draws.

## The one stage that did leave a region uncovered

`tests/startup_rect_geometry.cpp` compiles the shared rule (`src/startup_stubs/indexed_rect_geometry.h`) the way
the runtime does, binds it with back-face culling on, and draws one rectangle named by three corners. Its
`axis` order was covered exactly; its `rotated` order, the same square turned by 15 degrees, was dropped whole:
32 of 32 pixels kept the clear colour. The rule located the right-angle corner by looking for a corner that
shares an x or a y with each of its neighbours, so a rectangle with no axis-aligned edge matched no corner and
the stage returned without emitting either triangle. The vertex-stage rectangle expansion never had that
restriction: it uses `v0 + v2 - v1` directly, so the two implementations of the same rule disagreed, and a
rotated UI rectangle lost both of its triangles.

The fix identifies the same corner from the shape: the right-angle corner is the vertex opposite the diagonal,
which is the longest of the three edges, and the two sides must also meet at a right angle. Both
`indexed_rect_geometry.h` and the native rule in `src/gfx1013_rect_geometry.cpp` now use it, with the axis test
keeping precedence, so every input the old rule accepted is unchanged. After the fix `axis`, `swung` (the same
square in another corner order, which back-face culling rejects by its winding), `rotated` and `skewed` all
match the model: 0 of 32 pixels differ. `skewed` moves the top-left corner a quarter of the way toward the
top-right; it has no axis corner and no right angle, so the stage must leave the target clear and pins the
right-angle guard in both copies.

The native IR copy had the same corner rule, but its LLVM IR wrote the tolerance as the decimal
`1.000000e-06`; LLVM rejects that non-representable float constant, so every native rectangle pair was
refused before a PSO existed. It now uses the binary32 value's exact hexadecimal double,
`0x3EB0C6F7A0000000`. `native_rectangle_geometry_warp` compiles that geometry CSO and runs the same
`axis`, `swung`, `rotated` and `skewed` scenes through the native VS/PS pair with back-face culling on; all
four cover exactly the literal model's 32 pixels. The HLSL and native implementations are therefore pinned
to the same rule by readback, not only by compiling the IR.

## Commands and environment

```powershell
ctest --preset dev --no-tests=error -R '^(startup_rect_geometry|startup_rect_gap)_(warp|hardware)$' -V
ctest --preset dev --no-tests=error -R '^(native_rectangle_pso|native_rectangle_geometry)_warp$' -V
ctest --preset dev --no-tests=error -R '^(startup_indirect_translated_indexed_rect|startup_indirect_rectlist|native_direct_container|rectlist_topology|rectlist_draw)_(warp|hardware)?$'
```

Measured on Windows 11 build 26300, MSVC 19.44.35228, Windows SDK 10.0.26100.0 (DXC 10.0.26100.0), at
`63a30dc1` with this change on top. `startup_rect_geometry` ran on WARP (Microsoft Basic Render Driver
10.0.26100.10068) and on the AMD Radeon RX 9060 XT (driver 32.0.32029.2010). The runtime draw reported the
Microsoft Basic Render Driver in both of its backends, so its picture is a WARP result. The D3D12 debug layer is
not installed on this machine (`D3D12GetDebugInterface` fails), so the fixtures report `debug_layer=unavailable`
and no debug-layer messages were collected.

## What remains open

Which draw produced the reported gap is not established. A stream renamed by a view that declares fewer bytes
than the indices name draws nothing at all here (both the placed view and the recording-time copy stop at the
declared size), so that is a separate, unverified candidate. The indexed rectangle stage is only reached through
an indirect draw, and two of its paths are not covered by these fixtures: a translated non-indexed indirect
rectangle list keeps the title's three-vertices-per-rectangle count (the vertex stage then draws half of the
rectangles), and the stage's flat (`FPARAM`) outputs take the value of the first vertex for both triangles
instead of each triangle's provoking vertex. Neither was exercised here.
