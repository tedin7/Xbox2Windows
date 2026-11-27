# Packed 16-bit min/max in direct DXIL

The native frontend lowers the six gfx1013 VOP3P forms
`v_pk_min/max_{i16,u16,f16}`. Each source supplies one halfword to each
result lane through `op_sel` and `op_sel_hi`. The instruction writes one
packed word and uses the existing vector write predicate. It accepts
numeric VGPRs and SGPRs, literal words and inline zero. It refuses
noncanonical unused source fields, negation and clamp.

Signed and unsigned integer lanes compare the selected 16-bit values.
Binary16 lanes compare the bit patterns using the startup HLSL
`minmax16_bits` contract. This preserves signed-zero ordering and
subnormal encodings and prefers the numeric operand over a NaN. Two NaNs
produce the repository's canonical quiet NaN, a payload convention that
still needs RDNA hardware evidence. A non-preserving or unproven binary16
denormal mode is refused.

The public shadPS4 packed-FP16 work helped identify this family as a
recompiler capability to check. The observation was made on 2026-11-27
at its merged commit `34b35b526e3bfbb8a6f22c11035cb051174ac6f1` under
GPL-2.0. The DXIL lowering and fixtures were written in Xbox2Windows.
[shadPS4 PR #4354](https://github.com/shadps4-emu/shadPS4/pull/4354)
[Pinned shadPS4 commit](https://github.com/shadps4-emu/shadPS4/commit/34b35b526e3bfbb8a6f22c11035cb051174ac6f1)

Instruction forms and half selectors follow AMD's public ISA and LLVM
21.1.0 syntax documentation. The LLVM revision observed on 2026-11-27
was `3623fe661ae35c6c80ac221f14d85be76aa870f1` (Apache-2.0 with the
LLVM exception). The existing HLSL path provides the Xbox2Windows
reference for the selected halves and floating-point edge behavior.
[AMD RDNA 2 ISA, §12.10](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
[LLVM 21.1.0 GFX1030 syntax](https://releases.llvm.org/21.1.0/docs/AMDGPU/AMDGPUAsmGFX1030.html#vop3p)
[LLVM licence](https://github.com/llvm/llvm-project/blob/3623fe661ae35c6c80ac221f14d85be76aa870f1/llvm/LICENSE.TXT)

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay offscreen_check
ctest --preset dev --no-tests=error -j 1 -R '^native_packed_minmax_' -V
```

The synthetic Compute replays compare 2,064 words per type, including the
untouched words and guards. A sparse EXEC replay checks that inactive lanes
keep their previous word. Swapping the low and high selector groups changes
the binary16 readback. The Pixel fixture exports each resulting
halfword as an exactly representable binary32 integer and checks both
results. Compute replay requires shader model 6.6 and wave32.
