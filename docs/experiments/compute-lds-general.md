# General compute LDS lowering

The compute translator keeps the legacy exchange proof first. When that proof
refuses, the general retry accepts supported indexed LDS reads, writes and
atomics in straight-line code or forward-only control flow with an allocation of at
most 32 KiB. Addresses and single offsets are byte-addressed. Dual offsets use the
instruction's element size and its ST64 multiplier. Unaligned accesses use
byte operations; sub-dword writes use interlocked masks. The AMD RDNA 2 ISA
guide, sections 9.6, 10.4.3 and 12.13, describes allocation bounds and indexed
DS accesses. The Microsoft HLSL documentation for
`GroupMemoryBarrierWithGroupSync` specifies group synchronization.

The retry inserts a group barrier between conflicting plain and atomic LDS
operations and translates each guest barrier. A forward branch may not cross a
guest or inserted barrier unless `compute_dpp_sites` proves the point. This
rule also applies to scalar branches: wave-uniform SCC does not prove that every
wave in a group takes the same edge. Backward branches remain refused.
Lane-local compare masks and tracked lane-mask bitwise operations can update
EXEC. The all-zero and all-one constants select the same bit in each lane;
other constants use the flattened group-thread index modulo the proven guest
wave size. This assumes waves contain consecutive flat thread IDs. A 32-bit
EXEC write needs wave32 proof, because on wave64 it only changes the low half.
Unknown wave sizes, scalar-data masks and 64-bit literals with ambiguous
extension remain refused. Saved EXEC masks use the lane-mask flow proof. The
general LDS path uses the live EXEC bit after mask writes. Unaligned atomic
addresses still require proven alignment.

On Linux, run against the commit containing this note:

```sh
nice -n 15 python3 -B tests/startup_compute_lds_general_check.py
```

The controls translate every supported DS opcode and compare a CPU LDS model
under both wave orders for multidimensional transpose, same-wave exchange,
byte/halfword updates, dual and wide operations, unaligned accesses, bounds,
forward EXEC and scalar branches, and a narrowed-EXEC neighbour read. The
masked fixture initializes every slot before narrowing; deleting the LDS write
guard changes its CPU reference from 7 to the inactive neighbour lane number.
The same modeled results are used by the registered D3D12 readback tests:

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_lds_general_(controls|warp|hardware)$'
```

WARP and hardware readback require Windows, DXC and the offscreen runner; they
were not run in the Linux Python-only environment. A same-lane reference differs
from the peer-exchange reference; lost byte updates differ from the byte case.
Removing a barrier or an out-of-bounds guard is not guaranteed to fail on every
adapter: scheduling and out-of-bounds behavior can mask those mutations.

On Linux with Python 3.14.8, the working tree based on `c5b1fa87e` passed
`nice -n 15 python3 -B tests/startup_compute_lds_general_check.py` and, one at
a time, the same command for `startup_compute_lds_b64_check.py`,
`startup_compute_lds_b128_check.py`, `startup_compute_lds_dual_atomics_check.py`
and `startup_gfx2hlsl_compute_check.py`. The write-guard mutant failed the
general control: the CPU model reads 32 instead of 7 for lane 0. The WARP
dispatch fixtures were not run on this Python-only host.

On Linux with Python 3.14.8, the working tree based on `6a37665bb` passed
the five targeted Python controls above using `nice -n 15`, one at a time. The
wave32 lane-zero fixture writes slots from lanes 0 and 32 and reads them from
all 64 lanes; its CPU results agree in both wave orders. It is registered in
the WARP dispatch path, which was not run on this Python-only host.
