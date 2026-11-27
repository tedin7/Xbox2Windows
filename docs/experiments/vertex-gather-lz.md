# Vertex mip-zero 2D gather

The startup translator accepts vertex `image_gather4_lz` with a 2D texture
and one-hot component selection. It uses the compute path's sampler profile
and coordinate normalization. Direct3D `Gather*` reads the most detailed
mip of the bound view independently of derivatives. It returns lower left,
lower right, upper right and upper left in that order.
[Direct3D gather rules](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/gather4--sm5---asm-)

The synthetic fixture emits a triangle that covers the target and exports
four gathered words as a flat parameter. The pixel stage copies the words
to the render target. Six draws cover a texel centre, a texel edge, clamp
and wrap at the texture border, instruction `unorm` and sampler
`FORCE_UNNORMALIZED`. The draws select red or blue. The resource view starts
at physical mip 1 and the fixture gives each mip distinct data. A CPU
reference computes all four taps from the sample coordinates and address modes.
The negative control swaps the first two exported taps and requires the
renderer to report a pixel mismatch.

On Linux x86-64, the following encoding and translation checks passed. The
base revision `70d8e0742` refused all six vertex programs. Across the
mip-zero checks, the new translator matched the base HLSL and manifests
byte for byte for 17 previously accepted graphics fixtures and 25 compute
fixtures.

```sh
python3 -B tests/startup_image_gather_lz_check.py --llvm-mc llvm-mc-19
python3 -B tests/startup_compute_gather_lz_check.py --llvm-mc llvm-mc-19
python3 -B tests/startup_compute_gather_l_check.py --llvm-mc llvm-mc-19
python3 -B tests/startup_image_gather_c_lz_check.py --llvm-mc llvm-mc-19
```

The WARP readback and its negative control await a Windows run.

```powershell
ctest --preset dev --no-tests=error -R '^startup_(image|compute)_gather_lz(_warp)?$' -V
```
