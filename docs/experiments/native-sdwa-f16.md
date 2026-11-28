# SDWA binary16 arithmetic in direct compute

Issue #2963 adds SDWA to `v_add_f16`, `v_sub_f16`, `v_subrev_f16` and `v_mul_f16`.
Each source can be a VGPR, SGPR, or inline constant. WORD_0 and DWORD read the low
half, and WORD_1 reads the high half. Inline floats use a low-half binary16 encoding
with a zero high half. Source abs clears the binary16 sign bit before neg toggles it. The compiler
widens the halves to binary32, performs the operation, and rounds once to binary16
using the integer nearest-even path from the [plain forms](f16-alu-direct.md).
The binary16 rounding and denormal mode requirements apply to both forms.

WORD_0 and WORD_1 destinations accept UNUSED_PAD, which zeroes the other half,
and UNUSED_PRESERVE, which copies it from the old destination. DWORD and byte
destinations, UNUSED_SEXT, byte sources, source sext, literals,
clamp, and output modifiers are refused. SDWA min and max remain refused.

Direct wave32 compute SDWA VOPC compares can write a 32-bit mask to `vcc_hi` and read it as a scalar source.

Direct compute `v_cvt_f32_f16_sdwa` supports clamp with WORD_0 or WORD_1 and DWORD/UNUSED_PAD (NaN and -0 become +0), and still refuses omod.

## Fixture

The wave32 GPU fixture also covers SDWA packed RTZ f32-to-f16 placement and add-with-carry, including per-lane VCC input and output before destination placement.

The fixture compares the IR and hardware results against the same exact binary16
model as the plain-form fixture. The hardware run dispatches two wave32 groups,
with one lane per group writing the case words. Every untouched word must keep
the guard value `69696969`. An adapter without wave32 support returns skip code 77.

| Destination and sources | Add | Sub | Subrev | Mul |
|---|---|---|---|---|
| WORD_0 preserve, WORD_1 and DWORD | a55a4200 | a55abc00 | a55a3c00 | a55a4000 |
| WORD_0 pad, DWORD and WORD_0 | 00004400 | 00004000 | 0000c000 | 00004200 |
| WORD_1 preserve, neg(abs(WORD_0)) and WORD_1 | 3c001357 | c2001357 | 42001357 | c0001357 |

The add of `3c01` and `1000` rounds its tie to `a55a3c02` with low-half preserve.
The product of `0003` and `3800` rounds to the subnormal word `00000002`.
The product of `7bff` and `4000` overflows to `7c000000` with high-half pad.
The add of `7e01` and `3c00` expects a NaN in the low half and exactly `a55a`
in the preserved high half. The NaN payload and sign may vary.

Compile-only controls cover DWORD destination, BYTE_1 source, source sext,
UNUSED_SEXT, clamp, omod, min and max. The scalar-source fixture separately
covers SGPR and inline source selection, literal refusal, and scalar source sext.
Swapping WORD_0 and WORD_1 on either source and forcing UNUSED_PRESERVE to pad
are manual lowering mutations that must fail the readback comparison.

```powershell
cmake --preset dev
cmake --build --preset dev --target native_compute_ir_dump native_store_replay
ctest --preset dev --no-tests=error -R "^native_sdwa_f16_(ir|hardware)$" -V -j 1
```

Results: on an NVIDIA GeForce RTX 5080 (Windows 11, DXC from the Windows SDK 10.0.26100) the IR check matched all
17 results and the 10 refusals, and the GPU check matched 34 results in two workgroups with 2,030 guard words
intact. Reading the other half of a WORD_0/WORD_1 source, and writing the result without its SDWA placement,
each fail both checks, and so does dropping abs or dropping neg.
