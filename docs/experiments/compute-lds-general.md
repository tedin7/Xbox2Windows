# General compute LDS lowering

The compute translator keeps the legacy exchange proof first. When that proof
refuses, the general retry accepts supported indexed LDS reads, writes and
atomics in straight-line code with entry EXEC and an allocation of at most
32 KiB. Addresses and single offsets are byte-addressed. Dual offsets use the
instruction's element size and its ST64 multiplier. Unaligned accesses use
byte operations; sub-dword writes use interlocked masks. The AMD RDNA 2 ISA
guide, sections 9.6, 10.4.3 and 12.13, describes allocation bounds and indexed
DS accesses. The Microsoft HLSL documentation for
`GroupMemoryBarrierWithGroupSync` specifies group synchronization.

The retry inserts a group barrier between conflicting plain and atomic LDS
operations and translates each guest barrier. It refuses branches and EXEC
changes because the group-uniform execution of inserted barriers has not been
proved. Existing dual/atomic-only translations retain their prior output.
An unaligned atomic address is refused unless its alignment is proven.

On Linux, starting from `bf72f11d9`, run:

```sh
nice -n 15 python3 -B tests/startup_compute_lds_general_check.py
```

The controls translate every supported DS opcode and compare a CPU LDS model
under both wave orders for multidimensional transpose, same-wave exchange,
byte/halfword updates, dual and wide operations, unaligned accesses and bounds.
The same modeled results are used by the registered D3D12 readback tests:

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_lds_general_(controls|warp|hardware)$'
```

WARP and hardware readback require Windows, DXC and the offscreen runner; they
were not run in the Linux Python-only environment. A same-lane reference differs
from the peer-exchange reference; lost byte updates differ from the byte case.
Removing a barrier or an out-of-bounds guard is not guaranteed to fail on every
adapter: scheduling and out-of-bounds behavior can mask those mutations.
