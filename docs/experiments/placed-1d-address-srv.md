# Placed 1D textures by address

`X2W_PLACED_1D_TEXTURE=1` allows slot 70 to publish a read-only 1D texture from
committed CPU memory when the existing placement lookup misses. The source uses
the same desktop copyable footprints and array slice stride as slot 133
textures. The draw uploader stages the rows, and republishing the same address
and description increments the write generation. The switch also enables 1D
view validation. `X2W_PLACED_SRV_1D=1` by itself still requires an existing resource.

The adapter's format support selects native 1D storage first. A format that
supports 2D storage only can use a height-one texture if D3D12 accepts the
dimensions. That view requires a shader with a matching 2D declaration.
The address path accepts layout `UNKNOWN` with desktop-footprint rows. It also
accepts console linear layout `0x100` with one mip and one slice. The single row
starts at the base address. Linear mip and slice alignment remain unmapped.

```powershell
ctest --preset dev --no-tests=error -R '^startup_address_srv_warp$'
```

The WARP case compares 32 RGBA pixels against a CPU reference. It samples an
unregistered R8 ramp before and after republishing changed bytes in the same
frame. With only the older switch enabled, both calls must report
`texture_no_resource` and leave the null view intact. A width-four mutant must
produce the width-four reference and fail the width-eight comparison.
