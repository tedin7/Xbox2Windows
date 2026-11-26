# Slot 70 memory shader-resource views

## Refusals with the switch unset

Slot 70 first checks the arguments and the destination heap. A null destination, unreadable resource or view description, null address, or destination outside a registered view heap causes a refusal. An acceleration-structure view uses its own address when present and skips the resource description. It requires UNKNOWN format, default component mapping, 256-byte alignment, a live placed buffer and acceleration-structure state.

The following table enumerates the existing description gates. Any description outside these cases returns `unsupported_description`. Passing a gate can still require a matching live resource or suitable readable memory.

| View dimension | Existing accepted descriptions and resulting refusals |
|---|---|
| BUFFER | Requires a row-major BUFFER resource with UNKNOWN format, positive width, height/depth/mips/samples all one, quality zero, alignment zero or 64 KiB, and shader-resource access. The view requires default component mapping, a nonempty element range inside the declared width, and valid buffer flags. RAW requires R32_TYPELESS and zero stride. Structured requires UNKNOWN and a stride from 4 to 2048 bytes in multiples of four. Typed requires zero stride and an entry in [the exact typed-format list](../../src/startup_stubs/memory_srv_buffer.h). All other combinations, including typed sRGB, BGRA, depth, BC, typeless and planar formats, are refused. Live backing must contain the requested byte interval. The address must fall on an element boundary relative to its backing, and the resolved RAW first byte must be 16-byte aligned. The unplaced fallback requires the entire rounded 64 KiB placement extent to be committed in the allocation, a clear alias check and a span no larger than 256 MiB. |
| TEXTURE2D | The legacy placed gate accepts one slice and one mip of R8_UNORM, and the special one-mip BC3_UNORM/BC7_UNORM layout 0x105 path. Those views must match the resource format, use default mapping and select mip 0, plane 0, zero LOD clamp. The BC path also requires whole 4×4 blocks, alignment zero, desktop-sized dimensions and flags zero or 0x300000. The special BC chains, single-texel cases and general console-layout cases below also apply. Other ordinary placed 2D color textures and mip chains are refused by the description gate. |
| TEXTURE2DARRAY | Accepted only on the general supported console-layout path or on a two-slice special BC chain. An ordinary placed array with layout UNKNOWN is refused. An empty or out-of-bounds array slice range is refused. |
| TEXTURE3D | An ordinary volume requires layout UNKNOWN, a same-format view, nonzero extents and mips, one sample with zero quality, shader-resource access, a known non-depth format and an in-range mip selection at zero LOD clamp. Block formats must have whole top-level blocks. A same-family reinterpretation of typeless storage is refused. A general console-tiled volume is refused, with the single-texel exception below. A 3D SRV has a mip range and no depth-slice range field. |
| TEXTURECUBE, TEXTURECUBEARRAY | Both are refused for every resource and format. |
| UNKNOWN, TEXTURE1D, TEXTURE1DARRAY, TEXTURE2DMS, TEXTURE2DMSARRAY, unrecognized view dimensions | All are refused. TEXTURE1D resources have no accepted view. A dimension mismatch is refused. |

The special BC chains are BC5_UNORM or BC7_UNORM on layout 0x109, width 1024, one sample with zero quality, zero flags and alignment. They cover 1024-high six-mip single slices and 512- or 1024-high nine-mip textures with one or two slices. Views require the same format, default component mapping, explicit in-range mips, plane zero and zero LOD clamp.

The single-texel path covers 1×1 2D and 1×1×1 3D, one mip, one sample with zero quality, alignment zero or 64 KiB, the [listed color formats](../../src/startup_stubs/single_element_srv.h) and layouts 0, 0x100, 0x101, 0x102, 0x105, 0x106, 0x109 or 0x10A. It refuses render-target, depth, UAV and deny-SRV flags, mismatched view format or dimension, nondefault mapping and any view other than the whole texel at zero LOD clamp.

The general console-layout path covers 2D and 2D arrays only, with non-XOR modes 0x101, 0x102, 0x105, 0x106, 0x109 and 0x10A and the [listed color element formats](../../src/startup_stubs/gfx10_layout.h). It refuses depth formats, render-target/depth/UAV/cross-adapter/deny-SRV flags, nonstandard alignments, zero or oversized extents, multisampling, 256-byte block layouts with mip chains, excess mip counts and BC dimensions without whole top-level blocks. Its views require a valid mapping, a typed format equal to the storage format or from its typeless family, plane zero, zero LOD clamp and in-bounds mip and slice ranges. Only this gate and the ordinary volume gate accept all-remaining mips via `UINT_MAX`.

The backing and placement gates can also refuse GPU-written overlap for single-texel publication, unavailable or ambiguous live textures, unreadable publication memory, publications beyond the 64 MiB limit, desktop-unsupported formats, overlaps with live resources, partial BC tiles whose last write used desktop rows or whose size exceeds the reported allocation, and console layout data with desktop rows, missing footprints or an extent beyond the reported allocation. Publication failures are also counted. The detailed reason names are in [the census](../../src/startup_stubs/memory_srv_census.h).

## Opt-in path

Set `X2W_PLACED_SRV_SHAPES=1` to publish 2D mip ranges, 2D array slices, cube and cube-array views and typed views over compatible typeless volumes. New texture views require a matching live texture with layout UNKNOWN, one sample, an adapter-supported single-plane color format and valid subresource ranges. Cube views require square faces. Cube-array views take whole groups of six starting at a six-face boundary. The existing slot 18 path creates the descriptor.

For an unplaced typed, structured or raw buffer whose placement extent crosses uncommitted pages, only the readable requested window is copied to an upload buffer. Existing bounds, raw alignment and overlap rules still apply. Repeated publications of identical window bytes reuse the upload. Changed bytes get a new upload so earlier descriptors retain their data. These backings live until device teardown. A refusal leaves the descriptor unchanged.

The opt-in census emits cumulative counts per reason, resource dimension and view dimension at 30-second intervals on presents or slot-70 refusals. Dimension `-1` covers unreadable or unrecognized values. Counts include refusals beyond the detailed shape table.

## Windows verification

```powershell
ctest --preset dev --no-tests=error -R '^startup_(placed_srv_shapes_.*|memory_srv_census)$' -V
```

The WARP and hardware fixtures publish each shape through slot 70 and dispatch a native shader against the resulting descriptor. The CPU oracle predicts the texel or buffer element selected by the view. With the variable absent, the same call must leave a preinstalled null descriptor and return zero. The census fixture checks the 30-second cadence and totals beyond the detailed table limit. The fixtures require Windows and are pending execution. The shader readback exercises descriptor publication. Translation metadata for cube arrays is outside that check.
