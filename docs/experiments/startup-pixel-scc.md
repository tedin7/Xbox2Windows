# Scalar condition data in pixel shaders

The RDNA 2 [ISA reference](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)
describes the scalar condition code (SCC) in sections 5.2–5.7 and 12.1–12.5.

| SCC writers | Value assigned to SCC |
| --- | --- |
| `s_cmp_*`, `s_cmpk_*` | Whether the signed or unsigned comparison is true. |
| `s_bitcmp0_*`, `s_bitcmp1_*` | Whether the indexed bit has the requested value. |
| `s_add_u32`, `s_addc_u32` | Unsigned carry out. The carry-in form adds the incoming bit. |
| `s_sub_u32`, `s_subb_u32` | Unsigned borrow out. The borrow-in form subtracts the incoming bit as well. |
| `s_add_i32`, `s_sub_i32`, `s_addk_i32` | Signed overflow. |
| `s_lshl[1–4]_add_u32` | Unsigned overflow of the combined shift and addition. |
| `s_min_*`, `s_max_*` | Whether the first operand is strictly smaller or larger, respectively. |
| `s_and_*`, `s_or_*`, `s_xor_*`, their inverted forms, and `s_not_*` | Whether the result is nonzero. |
| `s_lshl_*`, `s_lshr_*`, `s_ashr_*`, `s_bfe_*`, `s_bcnt[01]_*`, `s_abs_i32`, `s_absdiff_i32`, `s_wqm_*`, `s_quadmask_*` | Whether the result is nonzero. |
| `s_*_saveexec_*`, `s_*_wrexec_*` | Whether the resulting EXEC mask is nonzero. |

Operand 253 reads SCC as an integer zero or one. Conditional selects and
moves read SCC implicitly and leave it intact. The translator already stores
modeled SCC results in the HLSL boolean `scc`. At a pixel join, this variable
has the value assigned on the executed predecessor, including the loop latch
on a subsequent iteration. The additional proof checks all reaching paths
when an explicit read fails the existing single-writer check. It also checks
conditional selects and moves in shaders that use the new rule.

The fallback requires modeled, wave-uniform SCC producers and branches. It
covers forward control flow and the existing bounded scalar loop shape
with a constant initial counter, step and bound. It reports the producer or
incoming edge when the proof fails. Existing descriptor and mask checks
still apply. The fallback excludes legacy min/max expressions because of
their tie behavior. Numeric `s_addc_u32` and `s_subb_u32` retain their prior
support boundaries. The fixture exercises carry using modeled
`s_add_u32` with explicit `src_scc`.

From the repository root, the following commands exercise the synthetic
translation and encoding on Linux.

```sh
python3 -B tests/startup_gfx2hlsl_src_scc_check.py
python3 -B tests/startup_gfx2hlsl_src_scc_join_check.py
python3 -B tests/startup_gfx2hlsl_src_scc_join_check.py binary /usr/bin/llvm-mc-19
```

The three commands passed on Linux with Python 3.14.8 and LLVM 19.1.7. Comparing
the fixture output against baseline `e8015da7a` found four byte-identical
pixel shaders and five shaders that previously refused the SCC operand
and now translate. The missing-writer diagnostic names edge
`+0x4 -> +0xc`. This checkout has no request-cache inputs for confirming
the original gameplay shaders.

On Windows, the draw test compiles the translated pixel shaders, draws 4×4
targets on D3D12 WARP and compares every channel with the CPU model. Separate
draws exercise the scalar branch arms. Pixel position supplies a distinct
colour input to each invocation. Inverting the logic writer must produce
16 mismatches. The missing-writer fixture must stop before compilation.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_src_scc_join_draw_warp$' -V
```
