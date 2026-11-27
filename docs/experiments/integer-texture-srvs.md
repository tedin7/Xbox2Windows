# Integer texture inputs

Set `X2W_INTEGER_TEXTURE_SRVS=1` for the offline startup translator to enable
typed integer texture loads. Use a fresh shader cache for this mode so an
existing compiled shader and its legacy manifest do not satisfy the request.
With the variable unset, previously supported programs emit their original
HLSL and manifests. Legacy manifests keep their original binding behavior.

The typed mode supports 1D, 2D, 2D-array and 3D loads. Each full-width load
resource keeps its original float declaration. A manifest line
`srvint N R E` adds UINT at `tR`, SINT at `t(R+1)` and a selector in entry
word `E`. The runtime binds the real view at the register with its component
type and supplies typed null views at the other registers. The signed
declaration preserves the signed return type for narrow formats as well as
32-bit formats. The shader selects the loaded bits as unsigned words before
putting them in the guest registers.

The `srvtypes 1` line also identifies typed manifests with filtered-only
inputs. The runtime binds a float null for an integer view on those inputs
and counts and logs `integer_view_filtered` once per pipeline key. D16 loads
retain their existing format validation.

## Reproduction

Linux x86-64, Python 3.14.8, baseline `e8015da7a`. The following commands
passed with synthetic inputs.

```sh
python3 -B tests/startup_image_load_mip_check.py
python3 -B tests/startup_image_load_d16_check.py
python3 -B tests/startup_image_3d_check.py
python3 -B tests/startup_gfx2hlsl_image_1d_check.py
python3 -B tests/startup_gfx2hlsl_compute_check.py
python3 -B tests/startup_compute_translate_check.py
```

An additional Python comparison loaded the baseline translator from Git and
compared six legacy draw and three legacy compute fixtures. Both HLSL and
manifests matched byte for byte. Ten typed draw and compute fixtures lowered
successfully, including explicit mip 0 loads in 1D.

Run the GPU tests on Windows with the D3D12 debug layer and DXC available.

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_(image_load_mip_(controls|warp)|compute_dispatch_texture_warp|null_cause_labels(_buffer_token_control)?)$'
```

The wrapped draw reads five texels per format into an R32_UINT target. The
compute dispatch writes the same values to a readback buffer and includes
out-of-bounds zero reads. The expected result is bit-exact UINT, SINT and
FLOAT readback with zero debug errors. Each typed load case also runs with
swapped float and UINT registers and with the integer selector inverted.
Each mutant must produce a mismatch or a debug-layer error. Controls cover
filtered integer inputs and the legacy manifest ABI. Windows execution,
adapter and driver results remain pending.
