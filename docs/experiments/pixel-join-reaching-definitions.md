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

The retry also restores scalar symbols and uniformity from a sole incoming
forward branch proven wave-uniform when the first refinement still refuses.
The branch carries the buffer word and a constant past a sibling arm that
rewrites both. Targets reached by fall-through
or by several branches keep the existing join rules. The per-lane branch
control uses a position-dependent vector compare and keeps its lane-mask
refusal.

The WARP range cases load 15, 16 and 31 to exercise both outcomes of the first
branch and both outcomes of the range check in the other arm. The green channel
uses the constant that reaches the selected arm. Reversing the inner range
check must preserve the draw at 15 and produce mismatches at 16 and 31.

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

The single-edge control passes on the same Linux setup. Revision `1c8dd2b85`
refuses the new case at `s_cbranch_scc0`. Removing the wave-uniform branch
guard makes the per-lane control fail by losing its expected refusal.
