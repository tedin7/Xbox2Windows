# Odd D16 image-load tails

The [RDNA 2 ISA guide, section 8.2.5](https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content)
specifies low-half-first packing for D16 image loads. It gives no value for
the unused upper half of the final VGPR when the DMask population is odd.
The translator continues to emit a zero-filled upper half for these loads
and checks that no supported consumer can observe it.

Plain VGPR moves carry the restriction forward. Low-half consumers remove
the restriction from a fully written result, including a 32-bit left shift
by 16 or more. Across joins and back edges the existing block-boundary guard
conservatively follows copies. For a compressed RG export to an explicitly
declared R32_FLOAT primary render target, only R contributes to the stored
result. G is independent of R for blending. The ISA's paired export masks
still apply.

```powershell
ctest --preset dev --no-tests=error -R '^startup_image_load_d16_(controls|warp|odd_tail_warp)$' -V
```

The WARP fixture draws synthetic shaders against an 8×2 RGBA16F texture.
It compares a three-component load followed by a copy, unpack and repack, a
one-component load followed by two copies and unpack, and the compressed RG
export to R32_FLOAT. The reference comes from the source texels' binary16
words. In each draw, inserting the neighbouring component in bits 31:16
must keep every result unchanged. Substituting it for the valid low half
must change all 16 pixels and match the mutated CPU reference. The controls
also require refusal of a 32-bit add after two copies and of the compressed
export when G is present on the target.

On Linux, `python -B tests/startup_image_load_d16_check.py` passed. Compared
with base `e8015da7a`, fixtures A–D emit identical vertex and pixel HLSL and
manifests. The base refuses each of the three newly accepted fixtures.
WARP execution requires the Windows test machine.
