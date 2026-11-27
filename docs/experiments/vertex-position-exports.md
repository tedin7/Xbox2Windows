# Vertex position exports

The graphics translator preserves POS1 through POS3 as local float4 values.
An epilogue reads the enabled clip and cull distances in position export order.
It reads the layer and viewport indices from the integer bits of the misc
vector. D3D12 point size is ignored. The manifest lists the exported slots in
`vsposx` and marks shaders that require vertex layer or viewport output. The
runtime checks the D3D12 feature option before loading those shaders.

The optional `req_<key>.vsout` file holds one little-endian 32-bit vertex output
control word. The word contributes to the graphics cache key. RTV variants use
their base request's word. The runtime publishes the word before the ready
marker when a slot-57 state block supplies it in a complete SET_CONTEXT_REG
packet sequence. A fixed offset in an opaque state block and a source for the
word in a slot-86 stream are still unverified. A request without the word keeps
the explicit refusal. Indexed rectangle draws with these system values require
geometry-stage forwarding and have a separate refusal.

References

- [RDNA 2 export instruction](https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content)
- [Vertex output control register fields](https://docs.amd.com/api/khub/documents/9fuBVmqajj07G~5~aeTUig/content)
- [D3D12 feature options](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_d3d12_options)

On Linux, run `python3 -B tests/startup_packed_position_check.py controls`.
It checks the unchanged fixture HLSL, the unknown-word refusal, and the
withdrawal of translations when the word is replaced or removed.
This check passed on Linux with Python 3.14.8 against the parent fixture output
from revision `e8015da7a`.

On Windows, run the WARP check with FXC available.

```powershell
ctest --preset dev --no-tests=error -R '^(startup_packed_screen_position_warp|startup_stream_position_exports_warp)$'
```

The stream fixture reads all 64 pixels from two render-target layers. Its
reference covers clipping by pixel centre, culling, layer and viewport
selection, ignored point size, and disabled exports. The mutants change the
distance source, swap index semantics, or apply clipping with the output
disabled. The WARP result is pending.
