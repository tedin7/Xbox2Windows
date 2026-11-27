# Slot 57 mesh scalar mask select

The mesh stage lowers `s_cselect_b32` with a proven wave-uniform SCC and
known EXEC, VCC, or saved EXEC words and the zero and all-ones literals.
A group-shared ballot reconstructs the selected 32-bit word per guest wave64,
independently of the desktop wave size. Writes to EXEC or VCC preserve the
other half of the mask. SGPR writes retain the full word for numeric use.

The synthetic fixture drives SCC from the primitive count. The CPU rasterizer
predicts the 32×32 RGBA8 target for both counts. Three mutants exchange select
operands, one per colour channel. Each must produce the CPU-predicted number
of different pixels on the same input. The draw uses D3D12 DispatchMesh and
reports capability skips separately.

From the repository root:

```bash
python3 -B tests/startup_gfx2hlsl_mesh_cselect_check.py
```

On Windows with the development preset built:

```powershell
ctest --preset dev --no-tests=error -R '^startup_mesh57_cselect_mask_(cpu|warp)$' -V
```
