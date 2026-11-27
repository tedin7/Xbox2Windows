# Descriptor table joins

Scalar loads can give the same descriptor word different symbols when one path
loads it as part of a wider record. At a forward join, the translator can use
the binding when every incoming path proves the same table, byte address and
pointer provenance for each word in the operand. This proof is available in
the join block and is invalidated when a scalar write replaces a word.

The resource binding comes from the root signature's table. Scalar constants
and root constant data do not establish such a binding by themselves. A fixed
scalar buffer read proves where the bytes came from, but the runtime manifest
has no mapping from arbitrary buffer contents to a desktop resource view.

The synthetic fixture exercises a scalar buffer read in compute and a 2D image
load in a pixel shader. The compute control reads a different dword. The pixel
control selects a different mip. Both controls must produce a readback mismatch.
The Python control also refuses a join whose incoming table addresses differ.

```bash
python -B tests/startup_gfx2hlsl_descriptor_join_check.py
```

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_descriptor_join_(controls|warp)$' -V
```
