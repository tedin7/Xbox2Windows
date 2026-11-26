# Compute vector lane reads

Slot 67 compute shaders can read an ordinary VGPR with
`v_readlane_b32 sD, vN, sK` when scalar dataflow proves that the selector and
every reachable branch are group-uniform. The proof follows both arms of SCC
branches and intersects uniform scalar state at joins, including loop back
edges. Read-only scalar loads can supply a loop bound. Vector-derived data
cannot establish uniform control flow.

Each invocation publishes its VGPR at its flattened local thread index. A
group barrier makes the values visible, then each invocation reads the selected
index in its guest wave. A second barrier protects those reads before the next
exchange reuses the same storage. Groups larger than 32 threads need a proven
guest wave size. When the size is unknown for a smaller group, the populated
lanes use the same numbering in either mode.

The synthetic fixture walks two fields of 36-byte records with two different
scalar selectors. A scalar load supplies the loop count and the first field
changes each iteration. The WARP test compares every output word with a CPU
reference for zero iterations, partial and full single-wave groups, two guest
wave32 waves in one group, and a wave64 group. Two mutants must produce GPU readback
mismatches, one with the second lane selector shifted and one with the guest
wave offset cleared. Python controls reject an unproven selector, a loop bound
that loses uniformity through a back edge, and a multi-wave group with unknown
guest wave size.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_vector_readlane_(controls|warp)$' -V
```
