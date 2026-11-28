# Slot 67 wave size in direct compute

Issue #2999. Slot 67 compute requests carry no documented wave size, and direct compute used to compile every
`.c67` request as wave32. Bit 20 of `.sa` dword 4 (little-endian; bits 15:0 are Z) now sets
`ComputeRequest::wave32`: set means wave32, clear means wave64. A `.sa` that is not 28 bytes is still refused.

## Why this bit

This is an inference from a local capture of 2,689 slot 67 requests, not from documentation. The capture was
decoded up to the first `s_endpgm`, and only counts and instruction shape were kept.

- Every program with a `*_saveexec_b32` and no `*_saveexec_b64` has the bit set (1,385 of 1,385). Every program
  with a `*_saveexec_b64` and no `*_saveexec_b32` has it clear (565 of 565).
- The next best bit of `.sa` or `.sb` agrees on 87%: bit 5 of dword 3, the X = 32 bit.
- `v_mbcnt_hi_u32_b32` appears only with the bit clear (197 programs). Every bit-clear program that uses
  `v_mbcnt_lo_u32_b32` also uses `v_mbcnt_hi_u32_b32`. The 84 bit-set programs that use `v_mbcnt_lo_u32_b32`
  never use `v_mbcnt_hi_u32_b32`.
- Group sizes mostly follow the bit. Of the 784 requests with a 64-thread group, 739 have it clear. Of the 1,433
  with a 32-thread group, 1,405 have it set.
- Bits 31:16 of dword 4 take only two values in the capture, 0x0000 and 0x0010.

## Translation

The direct DXIL route of `translate_compute` compiles a wave64 request through the wave64 path that
`translate_compute_container` already uses for a stage without `wave32` (two 32-lane host waves per guest wave).
A wave64 request whose group has at most 32 threads stays on the wave32 model, as before the wave size was read.
Its one guest wave only has lanes 0..31, so EXEC's high word is zero and `v_mbcnt_hi` adds nothing for its lanes.
One known difference remains: a wave64 compare or carry-out also writes zero to the high word of an SGPR pair
destination, and the wave32 model leaves that word as it was. The legacy HLSL profile and the container route are
unchanged.

Compute requests keep their `29cs` keys, and revision-28 archives are still read. Published native compute manifests now include `native 1`.
`compile_dxil --watch` rebuilds an older native output under the same key. The stand-in refuses it until
the rebuilt manifest appears, looking again on subsequent dispatches on the ManifestRetry schedule (1 s, doubling up to 32 s). An unmarked
manifest with a `<key>.cs.hlsl` sidecar is a Python translation and stays usable. `--requests` rebuilds
every output, the frontend cache is keyed on a hash of the compiler sources, and the in-process route
translates again in each process. A native publish now removes a `<key>.cs.hlsl` left by the Python translator,
so the pair cannot pass as a Python output. One case is not detected: a directory where an earlier
`compile_dxil` already republished a Python-translated key keeps that key's `.cs.hlsl` next to an unmarked
native manifest. Rerun `--requests` into such a directory once. The reload of a refused
manifest is not covered by a test; each check runs a new process.

Measured with `compile_dxil --jobs 8 gfx1013-direct-cs@36 --requests` on `cpp-migration` `e0d83759`, before and
after:

| Requests | Compiled before | Compiled after | Newly compiled | No longer compiled |
|---|---|---|---|---|
| Capture, 2,689 | 389 | 393 | 46 | 42 |
| One start-up run, 2,718 | 402 | 406 | 46 | 42 |

- The 46 newly compiled requests are wave64 programs with 64-thread groups.
- The 42 that no longer compile are wave64 programs that the wave32 model used to accept. They now stop at
  "multi-wave64 exchange requires converged group control flow", a limit of the existing wave64 path.

## Fixture

`tests/native_slot67_wave_size_check.py` builds the same program as three slot 67 requests that differ in this
bit and in the group size. The program stores `v_mbcnt_lo` followed by `v_mbcnt_hi` at each thread's index. The
CPU reference expects:

- with the bit clear and 64-thread groups, `tid % 64`;
- with the bit set and 64-thread groups, `tid % 32`;
- with the bit clear and 32-thread groups, `tid % 32`.

The fixture also checks every untouched buffer word.

```powershell
ctest --preset dev --no-tests=error -R '^native_slot67_wave_size_hardware$' -V
```

The fixture returns 77 when replay reports that the required adapter capabilities are unavailable. Source-built
fixtures that relied on the wave32 default now set the bit.
