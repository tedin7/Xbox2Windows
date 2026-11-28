# Startup compute stores to 3D and 1D UAVs

The startup translator lowers full-channel compute `image_store` to
`RWTexture3D<float4>` with an integer XYZ address or `RWTexture1D<float4>`
with a scalar integer address. A 3D address can be a VGPR tuple or an NSA list.
Both forms use the fixed table descriptor and emit a `tex3d` or `tex1d` UAV
entry. Requests refused before this change are retried by `--pending`, so
the compute cache revision is unchanged.

Dispatch uses the title's view at its selected mip. For 3D the shader's Z
coordinate is relative to `FirstWSlice`, and a W size of all ones selects
the remaining slices. The Dispatch fixture compares all mip-one texels
against a CPU reference and checks that a 2D view or an out-of-range W
selection leaves the texels unchanged.
