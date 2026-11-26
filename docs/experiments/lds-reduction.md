# Compute LDS reduction

The synthetic gfx1013 fixture launches one 4×64 group with two dwords of LDS
per local thread. Six stages read peers at byte strides 1024, 512, 256, 128, 64
and 32. Each stage reads an initialized upper span and writes a disjoint lower
span. The translator masks these accesses with the per-thread EXEC bit and
places all seven group barriers outside the masks. The proof requires LDS
completion waits, a saved entry EXEC and a forward skip that rejoins at its
restore or the immediately preceding LDS wait.

The readback checks 256 output words plus 64 guard words against a CPU
reference. Mutants select the same thread instead of its peer, select the
wrong FMA or literal half, select the wrong half of an LDS dword, or discard the
destination half that the 16-bit read should preserve. Each mutant must report
a readback mismatch.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_lds_reduction_(proof|warp|(exchange|fma-select|literal-select|half-address|half-preserve)_warp)$'
```

The offline proof checks the six-stage shape and rejects inputs with a wrong
local index, an incorrect peer address, a skip past the EXEC restore, an
incorrect saved EXEC pair, an unaligned halfword read or insufficient LDS.
The fixture covers the reported reduction idiom. The other shader keys require
separate verification.
