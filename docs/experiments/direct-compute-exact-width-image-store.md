# Exact-width image stores in direct compute

The direct gfx1013 compute translator lowers non-D16 `image_store` with
`dmask:0x1`, `0x3`, or `0x7` on 2D, 2D array, and 3D images. It consumes
one, two, or three consecutive VDATA VGPRs and declares a typed float,
float2, or float3 UAV. The DXIL store uses the full typed-store mask.
The narrower resource type determines which components reach the texture.
The `unorm` instruction modifier continues to leave the integer texel
coordinates unchanged.

Manifest version 2 describes these outputs with
`uav <index> <tex2d|tex2da|tex3d> float table <parameter> <byte> channels <1|2|3>`.
The in-process path gets the same width from the translated binding.
The runtime requires the view and resource formats to have the specified
component count and checks adapter support for typed stores on the resource
format. It creates the UAV with that format and retains the bound subresource
range. This makes each accepted store a whole texel write. Sparse masks still
receive the translator's descriptor modifier refusal.

The readback fixture dispatches two X groups of 4x2x1 threads into a 32x8 mip.
It explicitly sets EXEC to `0x55`, writing columns 0, 2, 4, and 6 in rows
0 and 1. The array and volume cases use relative slice 1 of a view that
starts at physical slice 1. Every other texel retains its sentinel.
The format cases use R32_FLOAT, R32G32_FLOAT, and R11G11B10_FLOAT. The
RGBA8 control leaves the complete image untouched and checks the runtime
channel-count refusal.
