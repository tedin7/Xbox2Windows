# Formatted compute loads from descriptor heaps

A compute shader can read a formatted buffer through a heap record whose index
and four descriptor words satisfy the raw-load proofs. Dispatch supplies the
record's stride and descriptor word 3. A typed view supplies its element size,
format and component mapping. A typed view with no format code refuses dispatch
with `a formatted input view has no buffer format code`. An unknown format code
in a native descriptor returns zero in every destination. Root views still
refuse translation.

The WARP regression uses synthetic half-float, packed normalized and unsigned
byte views of the same bytes. It also checks the table-word index path, two
native records and the unsupported-view refusal. The CPU reference must match
GPU readback. Compiled mutants that freeze the format or force a four-byte
stride must dispatch and give a readback mismatch.

```powershell
ctest --preset dev --no-tests=error -R '^startup_(gfx2hlsl_tbuffer_16f|compute_dispatch_heap_warp)$' -V
```

On Linux x86-64 with Python 3.14.8, the following checks passed. The first
reassembled both synthetic shaders and compared the existing table fixture's
HLSL byte-for-byte via its SHA-256 hash, recorded at baseline `e8015da7a`.
The baseline refused the synthetic formatted heap load. WARP is pending.

```bash
python3 -B tests/startup_gfx2hlsl_tbuffer_16f_check.py --llvm-mc llvm-mc-19
python3 -B tests/startup_gfx2hlsl_heap_record_check.py --llvm-mc llvm-mc-19
python3 -B tests/startup_gfx2hlsl_heap_word_check.py --llvm-mc llvm-mc-19
python3 -B tests/startup_gfx2hlsl_compute_check.py
python3 -B tests/startup_buffer_format_word3_table_check.py
python3 -B tests/startup_buffer_address_check.py
```
