# Startup packed binary16 min/max

The startup translator lowers `v_pk_min_f16` and `v_pk_max_f16` on gfx1013.
Each source is a single VGPR, numeric SGPR, 32-bit hex literal, or inline
integer zero. The inputs are packed words. `op_sel` selects each source's
half for the result's low half and `op_sel_hi` selects each source's half for
the result's high half. The defaults select the low and high halves
respectively. Other modifiers and source forms are refused. Packed add, mul
and fma still require VGPRs.

The helper strips the binary16 sign and compares the remaining encoding
against infinity to identify NaNs. One NaN yields the other operand's
encoding. Two NaNs yield the chosen canonical quiet NaN `0x7e00`. The
two-NaN payload choice has no hardware confirmation. Numeric comparison uses
ordered integer keys and returns the original bits. In either operand order,
min chooses negative zero and max chooses positive zero when the signs
differ. Equal zeros retain their sign.

The Python controls check emitted HLSL and operand refusals. The WARP and
hardware draw modes use a source-built gfx1013 fixture and compare both halves
with an independent CPU reference, including signed zeros, NaNs, infinity,
subnormals and literal and SGPR selector flips.

```sh
python -B tests/startup_gfx2hlsl_pk_minmax_check.py
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_pk_minmax'
```
