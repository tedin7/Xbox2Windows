# Byte-window stores in the direct C++ compute path

Issue #2965 covers raw stores through a rewritten table descriptor. The
guest loads the descriptor with `s_load_dwordx4`, extracts the stride
from word 1 with `s_bfe_u32` and clamps it to at least 1 with
`s_max_u32`. It clears the stride in word 1 with `0xc000ffff`, scales
word 2 by the clamped stride with `s_mul_i32`, and sets the raw OOB
selector in word 3 with `s_or_b32` and `0x30000000`.

The compiler proves that all three edits originate from the descriptor
selected by word 0. The OR works on a loaded word 3 or one whose selector
has already been cleared. The original table slot supplies the raw UAV.
The host UAV's extent determines which writes are in range.
Direct compute raw stores accept and ignore the GLC, DLC and SLC cache-policy bits.
Non-indexed dword atomics reuse the raw store UAV window, and the wave32 fixture checks add with and without GLC, unsigned min, compare-and-swap, and indexed and 64-bit refusals.

## Verification

`native_bytewin_store_controls` expects refusal for a plain SGPR
multiplier, a byte count from another table slot, and the OOB mask
`0x20000000`. It also refuses an indexed atomic and a 64-bit atomic through the byte
window. Dropping the stride proof or accepting a foreign
descriptor's word should make the corresponding control fail.

`native_bytewin_store_hardware` dispatches two 32-thread groups through
`native_store_replay`. Each store width has a 1024-byte plane with a
16-byte slot per lane. Widths 1 to 4 write the first 1 to 4 words of each
slot with `16 * lane + component + 1`. Unwritten words and both
eight-word guards retain `0x69696969`. All writes fit in the 8192-byte
raw UAV. Hardware without shader model 6.6 and wave32 support returns
skip code 77.

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_bytewin_store_(controls|hardware)$'
```

Results: on an NVIDIA GeForce RTX 5080 (Windows 11, DXC from the Windows SDK 10.0.26100) all four store widths
matched in both workgroups, with the unwritten words and guards intact, and the controls were refused. A later change
accepts non-indexed dword atomics through the byte window: add with and without GLC, unsigned min and compare-and-swap
read back against a CPU reference, while an indexed atomic and a 64-bit atomic through the window stay refused.
Dropping the clamped-stride proof lets the constant-multiplier control compile, and ignoring descriptor identity
lets the foreign word 2 control compile; each fails the controls.
