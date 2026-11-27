# DPP16 quad_perm in the direct C++ compute path

Issue #2936 covers DPP16 control values 0–255 with full row and bank masks.
The decoder retains the reserved-bit check and decodes FI, BOUND_CTRL, abs and
neg. The compute emitter uses the destination lane's low two bits to select a
two-bit field in the control byte. It adds that field to the base of the
destination lane's quad and reads source 0 from that physical wave32 lane.
Quads fit within each physical half of a split guest wave64.

Without FI, the guest EXEC bit of the selected source decides whether it can
be read. BOUND_CTRL supplies a zero source for a disabled lane. With
BOUND_CTRL clear, the instruction preserves the old destination. FI reads the
last VGPR value from a disabled source. Destination writes still obey EXEC.
The arithmetic and modifier checks are shared with the row mirror lowering.

## Refused forms

Compute requires a group size divisible by four for quad permutations.
Controls above 255 require a supported row mirror, shift or rotate.
[Row shift and rotate coverage](native-dpp-row.md)

## Verification

`native_dpp_quad_controls` checks compilation of a valid swap and named
refusals for row_share:0 and a 30-thread group.
`native_dpp_quad_hardware` compares both workgroups and buffer guards against
an independent CPU model. The fixtures cover integer OR and float add with
pairwise swaps, reverse subtraction with quad-local broadcast, identity,
rotation and the three disabled-source behaviors. The rotation exposes
reversed field order and the sparse EXEC add exposes ignored BOUND_CTRL.
Hardware without shader model 6.6 and wave32 support returns skip code 77.

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_dpp_quad_(controls|hardware)$'
```

On an NVIDIA GeForce RTX 5080 (Windows 11, DXC from the Windows SDK 10.0.26100) both tests passed: all eight
hardware programs matched the CPU reference for both groups and the guard words. At the time of that run,
the controls refused `row_shl:1` and a 30-thread group. Reading the permutation in reversed lane order fails the first case, and
ignoring BOUND_CTRL fails the disabled-source zero case.
