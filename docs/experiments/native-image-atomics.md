# Direct Compute atomics on unsigned 2D textures

The native Compute compiler lowers `image_atomic_add`, `image_atomic_umax`,
and `image_atomic_cmpswap` on a fixed table descriptor for a 2D `R32_UINT`
UAV. DXIL uses `RWTexture2D<uint>` with two integer coordinates and honours
the guest EXEC mask. GLC writes the pre-operation value to the data VGPR.
Compare-and-swap takes replacement from the first data VGPR and comparison
from the next one. Atomic writes are included when translating memory waits
and barriers.

Other dimensions, packed addresses, non-sequential coordinate lists, cache
modifiers, status returns, and noncanonical operand masks are refused.
Published compute manifests describe this binding as
`uav <i> tex2d uint table <parameter> <byte> channels 1`. The stand-in accepts
the manifest and checks the actual UAV format before recording a dispatch.
In-process binding for this type can follow the changes in #3049 and #3154.

## Reproduction

From an x64 Visual Studio Developer PowerShell with Python and DXC configured:

```powershell
$env:XBOX2WINDOWS_ADAPTER = 'RTX 4070'
cmake --preset dev
cmake --build --preset dev --target compile_dxil offscreen_check startup_stream_consumer d3d12_xs_shim
ctest --preset dev --no-tests=error -j 1 -R '^native_image_atomic_' -V
```

The [synthetic fixture](../../tests/native_image_atomic_check.py) compares
336 words per case, including unchanged texels and mips. It exercises unsigned
max across the sign boundary, compare success and failure, and a contended
add with only 16 active lanes. Each returning operation also feeds its result
to a second atomic. Clearing only the first operation's GLC bit changes the
second readback and fails the comparison.

The runtime fixture publishes the native compiler's DXIL and manifest from a
source-built device slot 67 request. A dispatch through the stand-in matches
all 256 words, with the one target texel incremented by 32 across two groups.
The same program with a float UAV is refused and leaves all 1,024 halfwords
unchanged. On Windows 11 build `26300.10075` both ran on the RTX 4070 with
driver `32.0.16.2124`. Direct DXIL requires wave32. Local WARP exposes four
lanes, so the native execution test reports a capability skip there.

## References

- [AMD RDNA 2 ISA, §8.2](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
  and [LLVM 21.1.0 GFX1030 MIMG](https://releases.llvm.org/21.1.0/docs/AMDGPU/AMDGPUAsmGFX1030.html#mimg)
  document the encoding and operands. LLVM release commit
  `3623fe661ae35c6c80ac221f14d85be76aa870f1` is Apache-2.0 WITH LLVM-exception.
- [DXIL AtomicBinOp and AtomicCompareExchange](https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#atomicbinop)
  define the destination and coordinate contract.
- The public [shadPS4 compare-and-swap report](https://github.com/shadps4-emu/shadPS4/pull/4109)
  was observed 2026-11-27 at merge `85476e55ea70201c470eb618193864844bead976`
  under GPL-2.0. It documents capability. The implementation and fixtures
  in this repository were developed independently.
