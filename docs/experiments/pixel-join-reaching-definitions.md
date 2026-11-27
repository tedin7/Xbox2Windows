# Reaching scalar definitions at pixel joins

On a forward pixel path, a scalar buffer load can overwrite a register that an
earlier arm also wrote. The join proof keeps that load when it is both the only
definition reaching the join and the last write in instruction order. A path
that never wrote the register contributes its entry value. A conditional write
can leave the previous definition in place. Back edges keep the existing proof.
The translator tries this proof only after the pixel stage refuses a wave-wide
read at an unproven `s_cbranch_scc0`.
[RDNA 2 ISA reference, chapters 2 and 12](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)

The synthetic program writes the scalar pair on an early arm, overwrites it
with a two-word load, then branches on two other buffer words before testing bit
16 of the overwritten register. The Python controls insert a write on one arm
of each diamond in turn and require the join to keep both reaching definitions.
The WARP case binds the six loaded words through the pixel AOT harness. It
compares all four channels with the CPU reference for each bit value. Reversing
the bit test must cause a readback mismatch in both draws.

```bash
nice -n 15 python3 -B tests/startup_gfx2hlsl_reaching_defs_check.py
```

```powershell
ctest --preset dev --no-tests=error -R '^(startup_gfx2hlsl_reaching_defs_controls|pixel_aot_reaching_defs_warp)$' -V
```

The Python control passed on Linux x86-64 with Python 3.14.8. Base revision
`ab02c55aef` refused the dominating-load case at `s_cbranch_scc0`. Disabling
the refinement reproduced that refusal. Dropping `s36` despite two reaching
definitions failed the last-diamond control. Windows WARP readback is pending.
