# Compute loads from the program image

The slot 67 compute translator can resolve a dword load from a table stored
after the last reachable instruction in the request's `.c67` image. It tracks
both words of an address from GETPC through moves and paired additions. The
high addition must consume the carry from the matching low addition. The same
address pair can still supply a calculated jump.

The proof follows forward control flow with unchanged EXEC. It checks the
range and alignment of each load and emits the accessed bytes as HLSL
constants. The compute manifest has the same bindings as an ordinary compute
shader. See the [RDNA 2 instruction reference](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)
for the GETPC, SETPC, carry and memory instruction semantics.

On Linux, the synthetic request, calculated jumps, encoding checks and
refusal controls can be reproduced with the following commands.

```sh
python3 -B tests/startup_gfx2hlsl_calculated_target_check.py
python3 -B tests/startup_compute_global_load_check.py
python3 -B tests/startup_compute_global_load_check.py binary llvm-mc-19
```

On Windows, the WARP test dispatches 32 threads for each of the GLOBAL vector
address, FLAT vector address and GLOBAL scalar base forms. The CPU reference
expects the same 16 known dwords twice. The address model forces a carry
across bit 31. Removing that carry must cause translation to refuse, and
reading the table from byte zero must cause a GPU readback mismatch.

```powershell
ctest --preset dev --no-tests=error -V -R '^(startup_gfx2hlsl_calculated_target|startup_compute_global_load_(controls|warp|image_warp))$'
```
