# Static sampler LOD bias

The startup translator emits `smplb` for a constant sampler with a nonzero
signed 14-bit `LOD_BIAS`. The draw and compute readers accept manifest version
3 and `smplb` with `X2W_SAMPLER_LOD_BIAS=1`. Version 3 makes older readers
reject the manifest. The static sampler carries the bias and both LOD limits.
Heap entries retain their existing LOD handling. The translator rejects a
secondary bias, unsupported bias range, or a biased sampler with mipmapping
disabled. In units of 1/256 mip, the supported interval is −4096 through 4093.
[D3D12 constants](https://learn.microsoft.com/en-us/windows/win32/direct3d12/constants)

The Windows fixture translates synthetic `image_sample` and `image_sample_lz`
programs against a three-level texture with one flat color per level. The CPU
reference selects the nearest mip after adding −0.5, +1.0, or 0 and clamping.
The implicit LOD fixture uses a 7/4-texel footprint. Expected mip indices for
the three biases are 0, 2, 1 for implicit LOD and 0, 1, 0 for explicit zero.
Static sampler bias applies to both implicit and explicit LOD.
[Direct3D sampling rules](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/sample-l--sm4---asm-)

The sign-flipped and dropped-bias controls must fail the color comparison.
A 13-bit decoder would accept the invalid +16 word as −16, submit the draw,
and fail the clear-color comparison. The valid inputs fit both signed widths.
With the switch unset, biased draws must leave the clear color intact.

```powershell
ctest --preset dev --no-tests=error -R '^startup_(sampler_lod(_bias)?_warp|image_sample_lod_controls)$'
```

The synthetic controls passed on Linux with
`python3 -B tests/startup_image_sample_lod_check.py`. A differential against
`e8015da7a` found 30 previously supported fixture outputs byte-identical
in generated HLSL and manifests. The new graphics bias control hit the
expected bias refusal on that revision. Windows WARP readback remains pending.
