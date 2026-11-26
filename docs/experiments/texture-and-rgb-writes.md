# Texture and RGB buffer writes

Compute cube stores use a 2D-array UAV. Their third address component selects
the face slice, including the cube offset in a cube array. Packed 16-bit
addresses unpack the low half first for 1D, 2D, 2D-array and 3D stores.
Pixel stores can unpack four half-float data components. A plain pixel load
and a store through one table entry can share a typed UAV. Red-only pairs use
an R32_UINT view. Floating-point vector pairs use a compatible float UAV.
The runtime checks typed-load and typed-store support before binding the view.
RGB formatted buffer stores accept three 32-bit components and store 12 bytes
through a raw UAV. They leave the following word intact.

The source-built WARP cases compare readback with CPU references for a cube,
a cube array, a packed-address volume, a pixel increment across two draws,
packed pixel data, and RGB buffer records. Each case includes a shader
mutation that must produce a readback mismatch.

```powershell
ctest --preset dev --no-tests=error -R '^(startup_compute_image_store(_warp)?|startup_(pixel_image_store|compute_store_format_xyzw)_(warp|controls)|startup_pixel_uav_support(_no_load_check_control)?)$'
```

The Linux translation and assembly checks use synthetic data.

```bash
python -B tests/startup_compute_image_store_check.py --llvm-mc /usr/bin/llvm-mc-19
python -B tests/startup_pixel_image_store_check.py binary /usr/bin/llvm-mc-19
python -B tests/startup_compute_store_format_xyzw_check.py
```

These checks passed, and 13 existing fixture outputs matched their base
revision byte for byte. Windows GPU results are pending.
