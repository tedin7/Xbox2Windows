# Identify the D3D12 adapter used by native compute replay

`native_store_replay` honors `XBOX2WINDOWS_ADAPTER` as a case-insensitive
substring of a D3D12 hardware adapter's description. With the variable unset,
it uses the default D3D12 device. With an unmatched or overlong filter, it
exits with code 2 before constructing a shader pipeline or writing a readback.

The runner writes the actual adapter name, vendor and device IDs, DXGI driver
version when available, and the LUID from the D3D12 device to stderr. The LUID
distinguishes adapters with the same name within a run. Binding rows still
go to stdout. [Microsoft's DXGI adapter lookup documentation](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgifactory4-enumadapterbyluid)
describes how to look up the adapter using the device's LUID.

From x64 Visual Studio Developer PowerShell with Python and DXC configured:

```powershell
cmake --preset dev
cmake --build --preset dev --target native_store_replay
$env:XBOX2WINDOWS_ADAPTER = 'RTX 4070'
ctest --preset dev --no-tests=error -j 1 -R '^native_(store_replay_adapter_controls|cvt_pk_u8_hardware)$' -V
```

Choose a filter for your hardware. The controls set their own filters and
require the expected errors without a shader file or output. The packed-U8
test runs synthetic literal and lane-derived shaders. On Windows 11 build
`26300.10075`, both matched 2,064 words on RTX 4070 with driver
`32.0.16.2124`. The report showed the same LUID with the explicit filter and
with the variable unset. Selection between multiple physical GPUs was not
measured on this host.
