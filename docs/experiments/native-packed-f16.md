# Packed binary16 arithmetic in direct Pixel and Compute DXIL

The direct compiler lowers the VOP3P `v_pk_add_f16`, `v_pk_mul_f16`, and
`v_pk_fma_f16` instructions. Each result half has independent source-half
selectors and input negation. The packed word uses the normal EXEC-masked
destination write. The lowering accepts numeric registers, literal words, and
inline zero.

ADD and MUL widen the inputs, use binary32 arithmetic, then round to binary16
with the existing integer nearest-even conversion. FMA forms the exact 22-bit
product and aligns the addend in integer arithmetic. Alignment keeps a sticky
bit for distant operands, where cancellation cannot reach the discarded bits.
The intermediate keeps 24 significant bits, with its last bit set to one when
discarded bits are nonzero. This round-to-odd step preserves which side of a
binary16 midpoint the exact sum fell on. The final nearest-even conversion
therefore agrees with a single binary16 rounding.

The supported MODE preserves subnormals, rounds to nearest even, and leaves
`FP16_OVFL` clear. Clamp, unsupported source encodings, and undetermined or
unsupported MODE keep explicit refusals. NaN results use the canonical
`0x7e00` half from the project's arithmetic reference. NaN payload behavior
has not been measured on an RDNA2 device.

## Reproduction

Run from an x64 Visual Studio Developer PowerShell with DXC and Python configured.

```powershell
$env:XBOX2WINDOWS_ADAPTER = 'RTX 4070'
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay offscreen_check
ctest --preset dev --no-tests=error -j 1 -R '^native_packed_f16_' -V
```

The [source-built fixture](../../tests/native_packed_f16_check.py) compares
both halves against the [exact rational model](../../tests/f16_packed_model.py).
Compute checks full and sparse EXEC readbacks, including inactive lanes and
buffer guards. Pixel checks the same FMA cases and alternates ADD and MUL in
the other two output channels. Two FMA cases distinguish one binary16 rounding
from a nearest-even binary32 intermediate. The first stores `0x3e053e01`,
where the double-rounded version would store `0x3e043e02`. The second stores
`0x6201be01` instead of `0x6202be02`. Compute replay requires wave32.

On Windows 11 with SDK 10.0.26100 and DXC 1.9.2602.17, the RTX 4070
(driver `32.0.16.2124`) matched all 2,064 Compute readback words in each of
the full and sparse dispatches. Pixel matched all ten cases on both the RTX
4070 and WARP (`10.0.26100.10068`). The local WARP adapter exposes four
lanes, so the wave32 Compute replay used the RTX 4070.

## References

- [AMD RDNA 2 ISA, section 12.10](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
  and [LLVM 21.1.0 GFX1030 assembly syntax](https://releases.llvm.org/21.1.0/docs/AMDGPU/AMDGPUAsmGFX1030.html#vop3p)
  document the operand fields. The LLVM release is `3623fe661ae35c6c80ac221f14d85be76aa870f1`, Apache-2.0 WITH LLVM-exception.
- [Boldo and Melquiond, *Emulation of a FMA and Correctly Rounded Sums*](https://guillaume.melquiond.fr/doc/08-tc.pdf)
  explains rounding to odd and the subsequent nearest-even conversion.
- [shadPS4 #4354](https://github.com/shadps4-emu/shadPS4/pull/4354) demonstrates
  the corresponding translation scope. Observed 2026-11-27 at
  `34b35b526e3bfbb8a6f22c11035cb051174ac6f1`, GPL-2.0. The code and
  fixtures in this repository were developed independently.
