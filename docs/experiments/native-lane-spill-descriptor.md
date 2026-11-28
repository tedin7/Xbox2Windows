# Descriptor lane spills in direct compute DXIL

The direct C++ path can carry an immediate table descriptor, a table
pointer word, a root address word or an unbound value through a constant
guest VGPR lane. An unbound value refilled is still unbound and is refused
where it is used.
The CFG tracks the identity per VGPR and guest lane. A symbolic spill or
refill emits no DXIL. A refilled descriptor still has to pass the usual
binding checks. A numeric write clears only its lane's identity. Any
other write to a VGPR holding symbolic lanes is refused.
Joins require the same identity on each path. If any path has a symbol
but the identity differs, the refill is refused.

Numeric lane operations use the existing DXIL lowering. In a split
guest wave64, that lowering selects the physical lane and exchanges
values between the wave32 halves as needed. Symbolic slots use the guest
lane before the physical mapping. Indexed descriptors that carry a
runtime index remain outside this experiment.

The source-built fixture spills four SRV descriptor words into lanes
0 through 3 and two numeric words into lanes 4 and 5, then clobbers the
descriptor SGPRs. It refills the six words and stores the loaded SRV
value and both numeric values for every lane. The CPU reference covers
192 written words across two wave32 workgroups and 1,872 guard and
unwritten words. Compile controls cover VGPR and SGPR lane selectors,
a VALU overwrite followed by descriptor refill, a conflicting CFG
join, and a direct VALU read of a VGPR with symbolic lanes.

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_lane_spill_descriptor_(warp|hardware)$'
```

The fixture is registered for Microsoft WARP and the default hardware
adapter. It requires shader model 6.6 and wave32 support.
