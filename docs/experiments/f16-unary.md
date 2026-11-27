# Binary16 exponential and reciprocal

The startup compute translator lowers `v_exp_f16` to `exp2` and `v_rcp_f16`
to `1.0 / x`. Both widen the selected half to float and round
the result to binary16 with `f16rn`. SDWA uses the existing half-word
selection, padding and preservation rules.

From the repository root, run the portable translation and retry checks:

```bash
python -B tests/startup_gfx2hlsl_f16_alu_check.py
python -B tests/startup_gfx2hlsl_compute_check.py
python -B tests/startup_gfx2hlsl_compute_check.py binary /usr/lib/llvm-19/bin/llvm-mc
python -B tests/startup_gfx2hlsl_retry_lifted_check.py
```

On Windows, build the runner and dispatch the fixture on WARP:

```powershell
cmake --build --preset dev --target offscreen_check
ctest --preset dev --no-tests=error -V -R '^startup_gfx2hlsl_compute_warp$'
```

The fixture reads 16 packed input words and compares all four result words
per lane against a CPU binary16 reference. Inputs include signed zeros,
subnormals, finite values, infinities and a positive canonical quiet NaN.
Three mutated shaders must report a readback mismatch for the wrong source
half, dropped preservation and natural exponential. The output words outside
the records must retain the sentinel.

The portable checks passed on Linux x86-64 against this change on base
`e8015da7a`. DXC 1.8.2407 compiled the fixture and all three mutants to
distinct DXIL binaries. Windows WARP readback and the three GPU mismatch
checks remain pending.
