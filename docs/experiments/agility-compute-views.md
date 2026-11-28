# Compute texture views with Agility SDK headers

The translated Compute dispatch keeps the SRV or UAV description until it
recreates the typed view in its desktop descriptor heap. The dispatch now
stores the corresponding D3D12 type, so the temporary description has the
size and alignment specified by the build's headers. The active member
follows the input/output branch that creates the view.

Guest capture records supply 40 bytes for each view. Dispatch copies that
prefix into a zero-initialized desktop descriptor before recreation. Checking
the record against the desktop size had treated valid 40-byte UAV records as
absent with Agility headers, leading 2D UAVs to use default mip 0 and causing
explicit 3D views to be refused.

On x64, the Windows SDK 10.0.26100 header reports 40 bytes for both view
types. The Agility SDK 1.620.2 header reports 40 for an SRV and 48 for a
UAV. At `88cddb09`, configuring with `sdk.nupkg` in the Agility directory
therefore stops the `offscreen_check` build at the 40-byte assertion in
`record_translated_dispatch`.

## Reproduction

Use x64 Visual Studio Developer PowerShell with DXC and Python configured.
Set `$agility` to the absolute path of the directory with the Agility x64
DLLs and `sdk.nupkg`, and `$warp` to the absolute path of the WARP x64 DLL
directory. The versions used
were [Agility SDK 1.620.2](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.620.2)
and [WARP 1.0.22](https://www.nuget.org/packages/Microsoft.Direct3D.WARP/1.0.22).

```powershell
cmake --preset release
cmake --build --preset release --target startup_stream_consumer d3d12_xs_shim
cmake --preset release -B out/build/agility "-DOFFSCREEN_AGILITY_620_DIR=$agility" "-DOFFSCREEN_WARP_DIR=$warp"
cmake --build out/build/agility --target offscreen_check startup_stream_consumer d3d12_xs_shim compile_dxil
$tests = '^startup_compute_dispatch_(texture|two_mips)_(warp|hardware)$'
ctest --test-dir out/build/release --no-tests=error -j 1 -R $tests -V
ctest --test-dir out/build/agility --no-tests=error -j 1 -R $tests -V
ctest --test-dir out/build/agility --no-tests=error -j 1 -R '^(shader_wave32_warp|native_image_atomic_warp)$' -V
```

These cases exercise translated SRV and UAV binding, texture views with
different dimensions, and two UAVs at different mips of one resource.
On Windows 11 build `26300.10075`, all four texture cases passed with
both header configurations. The two-mip test matched 1,280 halfwords and
detected 64 changed halfwords when the views were swapped. The wave32
probe and all seven native image-atomic readbacks also passed on app-local
WARP 1.0.22 with the Agility headers.

The [WARP setup report](https://github.com/R0Y00/Xbox2Windows/issues/3160#issuecomment-7226553102)
records the two header sizes.
