# Startup compute LDS exchange across local axes

The bounded compute LDS exchange accepts a fill address equal to the element
width times `x + X * (y + Y * z)`, where `(x, y, z)` comes from the entry
`SV_GroupThreadID` and `(X, Y, Z)` is the group size. The proof follows VGPR
copies, unsigned adds, literal shifts and multiplies, and fused shift-adds.
The fill must fit the allocation for every thread. The exchange requires a
uniform barrier after the completed fill and aligned reads from the initialized
span.

The [compute fixtures](../../tests/startup_gfx2hlsl_compute_check.py) fill LDS
with packed thread coordinates, synchronize, and read from a peer across a row
or plane. They cover groups of 8×8×1, 16×16×1, and 4×4×2. The 8×1×1 fixture
copies X and overwrites its original VGPR before the fill. The D3D12 runner
compares all 320 output words against the CPU reference, including the untouched
tail. Each same-lane control keeps the LDS address in bounds and must differ
from the reference at word 0.

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_compute_(controls|warp|(lds-copy|lds-8x8|lds-16x16|lds-4x4x2)-same-lane_warp)$'
```

On x86-64 Linux with Python 3.14.8, the compute and 64-/128-bit LDS Python
controls passed. The compute fixture words matched `llvm-mc-19`. Compared with
baseline `e8015da7a`, all 39 existing HLSL fixtures across the three suites
were byte-identical. The four new fixtures each reached the expected LDS
refusal on the baseline and translated with the new proof. WARP readback
remains to be measured on Windows.

```sh
python -B tests/startup_gfx2hlsl_compute_check.py
python -B tests/startup_gfx2hlsl_compute_check.py binary llvm-mc-19
python -B tests/startup_compute_lds_b64_check.py
python -B tests/startup_compute_lds_b128_check.py
```
