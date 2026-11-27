# Vertex loads from an embedded table

A vertex shader can build a raw buffer descriptor from GETPC and two immediate
descriptor words. The proof resolves the byte displacement relative to the
instruction after GETPC and checks that the descriptor's entire range fits
after the executable prefix. The setup must precede any branch. Forward
branches may reach loads through the same descriptor. The proof accepts an
unchanged high word or a zero high addition consuming the low addition's carry.

The vertex HLSL contains the table as constant words. Each raw dword load
checks the descriptor's byte bound, returns zero outside it and uses the usual
two-word reconstruction for an unaligned load. The descriptor adds no binding
to the manifest. See the [RDNA 2 instruction reference](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)
for the buffer resource and range-checking rules.

The Linux controls reproduce the translation and refusals using a source-built
synthetic request.

```sh
python3 -B tests/startup_gfx2hlsl_getpc_refusal_check.py
```

The Windows fixture draws eight one-pixel instances into an 8×1 floating-point
target on WARP. It compares the four read-back words per pixel with a CPU
reference for aligned, unaligned, and out-of-range loads. The address mutant
moves the table by one row and must mismatch at all eight pixels against the
original reference. The seven-row descriptor must mismatch at one pixel
against that reference. The test uses DXC and also FXC when configured.

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_gfx2hlsl_getpc_(refusal|table_warp)$'
```

## Pixel scalar loads from an embedded table

The pixel path accepts an entry GETPC, an immediate `s_load_dwordx4` through
that pair and a full wait. The program runs straight through its terminator.
The four words must be aligned to a dword and fit wholly after the terminator.
The proof also checks that the address pair has no other use. The HLSL assigns
the four constant words to the destination SGPRs without a resource binding.
See [RDNA 2 scalar memory addressing, §7.2.1](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf).

On Linux, `python3 -B tools/startup/gfx2hlsl.py --refusals out/getpc-scalar-diagnostic`
examined one synthetic request using the pixel fixture and the existing vertex
table fixture. Revision `82f5cf5c0` reported the GETPC refusal first, followed
by a scalar load through an unknown base. The revised translator reported zero
refusal sites. The Python command above tests the table extraction and the
refusals for invalid bounds, alignment, control flow, address use and encoding.

The existing WARP test also draws each of two embedded RGBA rows across a 3×2
floating-point target. All 24 readback words per draw must match the CPU
reference. Shifting the load to the second row must fail at all six pixels
against the first row's reference. The Windows command above runs this case
with DXC and with FXC when available. WARP verification is pending.
