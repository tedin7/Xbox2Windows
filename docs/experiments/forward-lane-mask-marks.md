# Lane-mask marks at forward shader targets

An SCC branch selects one arm for the wave. A VALU compare can write a lane
mask into an SGPR pair in one arm, then an unconditional branch can skip the
other arm. The skipped arm still sees the numeric value from before the branch.
See the [RDNA 2 ISA reference](https://gpuopen.com/download/rdna2-shader-instruction-set-architecture.pdf),
sections 4.2 and 12.9.

The pixel heap fixture exercises an `s_load_dwordx8` through `s[10:11]` after
such a skip. The prior walk refused it with
`s_load_dwordx8 reads lane mask s[10:11] as numeric data`. The scalar-add fixture
also tests the high word of the pair and requires the old refusal when the mask
reaches the add via fall-through or when the add starts a loop header.
Other controls carry a mask through an earlier join and a scalar copy, and
through a back edge to a forward loop exit.

On Linux, these commands run the Python translation controls.

```bash
nice -n 15 python3 -B tests/startup_gfx2hlsl_dynamic_heap_check.py
nice -n 15 python3 -B tests/startup_gfx2hlsl_src_scc_join_check.py
```

The WARP tests draw both flag values and compare the render target with CPU
references. In the heap fixture, advancing the heap index by one slot must
change the returned texel. In the scalar-add fixture, omitting the addend
must change all 64 pixels on the numeric arm.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_(dynamic_heap(_warp)?|src_scc_join(_draw_warp)?)$' -V
```
