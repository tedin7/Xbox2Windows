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
