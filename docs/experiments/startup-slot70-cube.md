# Slot-70 cube views of console-layout textures

Device slot 70 publishes a texture by address. A description that is a two-dimensional array of six slices
in a console layout, with a `TEXTURECUBE` view, was refused with `reason=unsupported_description`, because
`address_view_fits` had no cube view. This note takes such views and binds them in a translated draw that
declares a cube texture.

The translator does not declare one yet: `tools/startup/gfx2hlsl.py` refuses an image instruction with
`dim:SQ_RSRC_IMG_CUBE` (`image dimension SQ_RSRC_IMG_CUBE`). Until it lowers cube sampling to a
`TextureCube` and writes the `texcube` manifest kind, no translated shader reads these views. The draws here
use a hand-written pixel shader and manifest.

## Change

- `src/startup_stubs/address_texture.h`: `address_view_fits` takes a `TEXTURECUBE` view of a 2D texture of
  exactly six slices that fits the mip range with no LOD clamp. The faces are the slices in D3D order
  (+X, -X, +Y, -Y, +Z, -Z). The slices are already addressed and uploaded like any 2D array.
- `src/startup_stubs/d3d12_memory_srv.inl`: the desktop format check also asks for
  `D3D12_FORMAT_SUPPORT1_TEXTURECUBE` for a cube view.
- `src/startup_stubs/d3d12_draw.inl`: a manifest `srv` line of kind `texcube` (kind 5) binds a
  `TEXTURECUBE` view. The view dimension of each kind is now in one function.
  A cube view where the shader declares `tex2da`, or the reverse, still binds null (view dimension mismatch).
  The d16 exact path still takes only kinds 1 to 4.

## Results

Build: `slot70/cube-views`, stacked on `slot70/linear-1d` (42c3bff5), dev preset (debug layer on).
Windows 11 Pro 10.0.26200, NVIDIA GeForce RTX 5090 (driver 32.0.16.1088) and Microsoft WARP.

| Test | Result |
| --- | --- |
| `startup_gfx10_layout` | PASS: 39 views (new: a whole BC1 cube chain, its last mip, past the chain, an LOD clamp, a cube view of one slice and of an array of twelve) |
| `startup_address_console_{warp,hardware}` | PASS: 3 new cube draws are exact on all 65,536 pixels, covering every texel of all six faces: RGBA8 64x64 in `0x105`, its 4-mip chain viewed at mip 1, and BC1 64x64 in `0x105`. The shader point-samples each texel centre along the face's direction |
| `startup_address_console_cube_order_{warp,hardware}` | PASS (control): the first cube with face f written into slice 5 - f is published and all 65,536 pixels differ |

Without the source change (42c3bff5), the first cube draw reads the null view: pixel (0,0) is `00000000`,
but `010300ff` was expected.

Commands (PowerShell, at the repository root, after `. .\tools\dev.ps1`):

```powershell
cmake --preset dev
cmake --build --preset dev
ctest --preset dev --no-tests=error -R '^startup_gfx10_layout$'
ctest --preset dev --no-tests=error -R '^startup_address_console_(warp|hardware|cube_order_(warp|hardware))$' -V
```

## Open

- The translator side: `image_sample*` and `image_load*` with `dim:SQ_RSRC_IMG_CUBE`. A GFX10 cube sample
  takes face-relative coordinates and a face index that the `v_cube*` instructions compute (which
  `tools/startup/gfx2hlsl.py` already lowers). A lowering to `TextureCube` would rebuild a direction from
  them. That is not attempted here.
- A cube view of a larger array, and cube arrays (`TEXTURECUBEARRAY`), stay refused.
