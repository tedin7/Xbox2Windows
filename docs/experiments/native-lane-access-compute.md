# Lane access in direct compute DXIL

This experiment covers numeric SGPR spill and refill through VGPR lanes
(#2940). A lane selector is masked to the guest wave width. Split guest
wave64 reads exchange the values from the two physical wave32 halves.
Lane writes select a guest lane independently of EXEC. Scalar lane reads
can target `vcc_lo` and `vcc_hi` in direct compute. In wave32, `vcc_hi`
(SGPR 107) holds ordinary scalar data and VCC uses only `vcc_lo`. A
wave32 scalar pair read of VCC reads s106 and s107, as the hardware does. In
wave64, `vcc_hi` holds the high half of the VCC mask. Refills carry
numeric values and clear the destination's symbolic provenance.

Direct compute `s_mov_b32 exec_hi` stores a readable high word in wave32 and updates the upper-half EXEC mask in split wave64.

The source-built hardware fixture compares two 32-thread workgroups with
a CPU reference and checks the untouched buffer words. It covers constant
and SGPR selectors, writes to an EXEC-disabled lane, and lane reads feeding
VCC. It also reads back wave32 `vcc_hi` after both lane-read instructions
and a scalar move, and checks the high word of a pair move from VCC.
Compile controls exercise partial waves, out-of-range direct lane-read
destinations, and HLSL-native lane reads targeting either VCC register.
Graphics lane reads also reject both VCC destinations.

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_lane_access_hardware$'
```

On an NVIDIA GeForce RTX 5080 (Windows 11, DXC from the Windows SDK 10.0.26100) the test
matched all 768 lane-access words and the guards in both workgroups. Reading lane `lane ^ 1`,
letting `v_writelane_b32` honour EXEC, reading `vcc_hi` as zero in wave32, and reading the high word of a
wave32 VCC pair as zero each fail it.
The fixture is one 32-thread wave32 group per workgroup; the wave64 half selection and the
accepted partial-wave constant lane are not exercised.
