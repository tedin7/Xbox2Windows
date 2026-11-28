# Derivative-selected gather with a nonzero SRV base mip

This differential for #1802 keeps UV at `(9/16, 9/16)` at the checked pixel.
The source-built gfx1013 shader routes the result of `image_get_lod` to
`image_gather4_l` using the existing startup translator. The four taps are
read back as RGBA32F. An 8×8 texture has three patterned levels, and the main
SRV exposes resource mips 1 and 2.

| UV step per pixel | Relative mip | Resource mip |
| --- | ---: | ---: |
| 1/4 | 0 | 1 |
| 1/2 | 1 | 2 |

The expected taps come from the texel pattern and are checked independently
against `image_gather4_lz` on one-level views of the two resource mips. Both
levels contain four distinct, interior taps at the chosen UV. A separate
`image_gather4_lz` draw on the original two-level view continues to read its
first visible mip. That control differs at all four taps when the derivatives
select resource mip 2. Desktop 2D Gather uses the top level of the view.
[Microsoft gather4 instruction reference](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/gather4--sm4-1---asm-)

The fixture covers two exact integer mip selections with a constant point-mip
sampler and an ordinary 2D SRV. The guest implicit opcode still requires proof
of the guest mip selection. Fractional thresholds and RDNA hardware readback
are open.

```powershell
cmake --build --preset dev --target offscreen_check
ctest --preset dev --no-tests=error -R '^startup_gather_implicit_mip_(warp|hardware)$' -V
```
