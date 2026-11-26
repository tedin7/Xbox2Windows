# Compute packed buffer formats

`buffer_store_format_d16_xyzw` reads four binary16 values from two VGPRs.
It writes an eight-byte element using the descriptor's FLOAT, UNORM or SNORM
conversion. The store uses the scalar D16 path's address and bounds rules.
A dword-aligned element uses a raw UAV Store2. At a halfword-aligned address,
atomic updates preserve the neighboring halves of the first and last dwords.

Compute formatted loads use the same FORMAT switch and DST_SEL mapping for
hardware table descriptors, resource-heap descriptors and descriptors whose
format word the shader supplies in the entry block. Resource-heap bindings
use the dispatch snapshot of the buffer and its format word. A replacement
format word keeps the original descriptor's base, stride and record count.

From the repository root on Linux, run
`python -B tests/startup_compute_packed_formats_check.py` to check synthetic
instruction decoding, translation and manifest bindings. On the Windows test
machine with DXC and the existing offscreen runner, run:

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_compute_packed_formats_(controls|warp|(store|register|memory|heap)_mutant_warp)$'
```

The WARP test compares the entire raw UAV with CPU-generated reference words.
It covers both format encodings for each packed store conversion, dword- and
halfword-aligned writes, and preservation of the out-of-bounds record.
All three load variants exercise unsigned, signed normalized and binary16 sources
with a component reorder and a forced one. The four mutant tests require a
readback mismatch from duplicating the first packed pair or hardcoding the
load format. The offscreen harness binds the source and destination buffers
directly and supplies the heap descriptor's format word through the translated
entry constants.
