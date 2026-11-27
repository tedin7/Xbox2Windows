# Pixel DPP16 derivative readback

The pixel translator preserves the subtraction's operand order and the DPP16
source-lane selection. The fine exchanges use `QuadReadAcrossX` and
`QuadReadAcrossY`, so neighbour-minus-self changes sign across each pair.
For coarse differences, `[1,0,1,0]` and `[3,2,3,2]` select the upper or lower
row. `[2,2,0,0]` and `[3,3,1,1]` select the left or right column. These
projected exchanges use constant-index `QuadReadLaneAt` and support FP32
subtract, reverse subtract, and a move followed by a separate subtraction.
See the [DPP16 ISA description](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
and the [Shader Model 6 quad layout](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12).

A scalar-only prefix may branch before `s_wqm exec, exec` when WQM dominates
the derivative read and the control flow preserves the quad. A path that
bypasses WQM, a divergent branch, or a vector write in the late-WQM prefix
still fails the proof.

On Linux x86-64 with Python 3.14.8, the translation controls passed:

```sh
python -B tests/startup_gfx2hlsl_dpp_derivative_check.py
```

The Windows test compares all four FP32 words of each pixel in a 16 × 16
target with a CPU reference for the gradient `(x + 2) * (y + 5)`. It runs
fused and split forms of fine and coarse differences in straight-line and uniform
branch flow. Each draw has four selector mutants, and each branched draw
also has a wrong-branch mutant. Each mutant must reach readback and report
a pixel mismatch. WARP results are pending.

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_gfx2hlsl_dpp_derivative_(controls|warp)$'
```
