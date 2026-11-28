# Direct binary32 fract upper bound

The direct DXIL frontend accepted `v_fract_f32` but emitted only DXIL `Frc`.
For the synthetic inputs `0xb0800000` (-2^-30) and `0xb1800000` (-2^-28),
that returned `0x3f800000` (1.0) on the tested desktop adapter. The gfx1013
instruction keeps its result strictly below one, returning `0x3f7fffff` for
these inputs. The Python translator already applies this bound; see the
[earlier fract experiments](../experiments.md),
[#267](https://github.com/R0Y00/Xbox2Windows/pull/267) and
[#2010](https://github.com/R0Y00/Xbox2Windows/pull/2010). This is the same
boundary in the newer direct DXIL path, rather than a change to those Python
lowerings.

The direct frontend now caps a result at or above 1.0 with an ordered comparison.
A NaN fails that comparison and passes through. The cap precedes the output
scale and clamp: `mul:2` of these tiny negative inputs must yield `0x3fffffff`,
not 2.0. Changing the direct frontend also changes its generated cache revision.

The fixture constructs eight inputs and four instruction forms per input:
VOP1, VOP3 with input negation, VOP3 with clamp, and VOP3 with `mul:2`.
Each lane of two wave32 groups writes all 32 results. The readback compares
2,048 output words and 16 guard words, accepting NaNs by class. The controls
compile and validate DXIL and reject reference mutations that omit the cap or
replace NaN with the finite cap.

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -V -R '^native_fract_f32_(controls|hardware)$'
```

At parent revision `240e0fb9c460f24d51683abe2d4cce2800b136c1`, the hardware
fixture failed on 384 of 2,064 words. With the bound, all words matched.
The adapter was an NVIDIA GeForce RTX 4070 Ti SUPER, driver 32.0.15.7270,
on Windows 10 build 19045, with DXC 1.9.2609 and Agility SDK 1.620.2.
The ordinary Windows 10 fixture executable reported the shader-model capability
skip; the measured runs relinked the same fixture objects locally with the
Agility SDK version/path exports and placed its matching runtime beside them.
The D3D12 debug layer was not enabled for those compute readbacks.
The compile controls, `native_f16_alu_ir`, `native_frexp_f32_controls` and
`native_mad32_controls` all passed. No console hardware or full program run
is claimed by these fixture results.

After rebasing onto `aeef65b2da356b15cf1a81c5d6a73bd6cee77bcd`, the two
targets rebuilt, the compile controls passed, and the Agility-linked hardware
fixture again matched all 2,064 words. The ordinary executable still reported
the capability skip.
