# Scalar descriptor words

The startup translator recognizes all-zero resource operands for its supported
image and buffer forms. Reads produce zero and stores emit no write. A
`s_mov_b64` into VCC can track a descriptor word paired with another descriptor
word or a known constant, including when the pair starts at an odd SGPR.
Overlapping resource accesses still check the VCC words that reach the use.
An ordinary move of two constants also retains its numeric mask assignment.

A numeric consumer can materialize arithmetic over descriptor words that have
a runtime binding. Graphics manifests use version 3 when they need
`udsrc N descword TABLE BYTE`. The draw supplies the dword from the bound table
as an entry constant. An unavailable word causes the draw to refuse.
Translations that never consume a descriptor word numerically still use their
previous manifest version.

The WARP fixtures read back an RGBA32F pixel. The null sample adds 0.25 to red,
the odd pair selects a known texel of a 2×2 texture, and the descriptor count
fixture masks and scales the record count from two synthetic descriptors.
The swapped pair is refused by the translator. Its GPU control maps that invalid
descriptor to zero in the harness and checks for a mismatch. The other mutants
return one for a null read and bind an adjacent word as the record count.

On Windows, after building the regular test preset, run:

```powershell
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_(desc_vcc|desc_ptr_prop|scalar_null)_warp$'
```

Linux with Python 3.14.8 passed these synthetic checks:

```bash
out/pe-env/bin/python -B tools/run_checks_parallel.py -j 2 \
  tests/startup_gfx2hlsl_desc_vcc_check.py \
  tests/startup_gfx2hlsl_desc_ptr_prop_check.py \
  tests/startup_gfx2hlsl_scalar_null_check.py \
  tests/startup_gfx2hlsl_compute_check.py tests/docs_links_check.py
```

The three new probes reproduce the respective refusals on the parent
revision. A comparison of the vertex HLSL, pixel HLSL and manifest for 19
existing synthetic pipelines gives 57 byte-identical outputs. The Windows
readbacks and their mutants are pending.

## Descriptor-derived texel limits

The pixel fixture loads a fixed eight-word descriptor, computes two limits
with scalar field extraction and shift-add, selects on the zero-word compare,
and keeps the selected scalars across a forward SCC branch. A 4×2 texture
provides distinct texels for the CPU comparison of all 64 output pixels.
The second draw binds a null SRV and passes eight zero descriptor words.
The adjacent-size-word and swapped-select controls must first differ at
`(1,0)` and `(2,0)` respectively.

On Linux with Python 3.14.8, the synthetic fixture translates with manifest
version 3. The merged prerequisite `af9f12757` refuses its shift-add.
Four existing synthetic pipelines produced byte-identical vertex HLSL,
pixel HLSL and manifests against that revision. The descriptor controls
also exercise unavailable, advanced and modified words, writes to lane masks,
and ambiguous SCC definitions.

```bash
python -B tests/startup_gfx2hlsl_desc_ptr_prop_check.py
python -B tests/startup_gfx2hlsl_desc_vcc_check.py
```

Windows WARP readback remains pending. Run the existing registered test with
`ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_desc_ptr_prop_warp$'`.
