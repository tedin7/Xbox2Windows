# Compute scalar results across EXECZ

For the covered numeric SGPR operations, a forward EXECZ branch can use
`WaveActiveAnyTrue` to keep the host wave on one control-flow edge. On the
fall-through edge each thread updates its scalar state. On the taken edge it
retains its previous state. The runtime register join selects the value on the
executed edge for every destination dword. Vector writes keep their per-lane
EXEC guard. The fallback requires a group of at most 32 threads, a host wave
wide enough for the group, and wave-uniform control flow. It uses the existing
scalar ALU and SMEM emitters.

The synthetic fixtures test AND, bitfield mask, left shift, and two- and
four-dword reads from both a buffer descriptor and a root SRV. Two of four lanes
enter the block in one dispatch. Zero lanes enter in the other. All four lanes
store their results after restoring EXEC. The CPU reference compares all 320
words, including untouched sentinel words. Each fixture compiles two mutations.
A lane-local branch must mismatch at the first inactive lane in the partial
dispatch. An always-enter branch must mismatch at the first word in the empty
dispatch.

From the repository root on Linux:

```sh
python3 -B tests/startup_compute_execz_scalar_check.py --llvm-mc llvm-mc-19
python3 -B tests/startup_gfx2hlsl_compute_check.py
python3 -B tests/startup_compute_readfirstlane_exec_check.py
```

The scalar checks passed on Linux with LLVM 19. All seven fixture inputs had
the live-definition refusal at baseline `e8015da7a`. The 33 existing compute
fixtures produced byte-identical HLSL when compared with that baseline.

On the Windows test machine, run the registered controls and WARP readback:

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_execz_scalar_(controls|warp)$'
```

WARP readback and mutation outcomes are pending Windows verification.
