# One-word pixel mask from a buffer

A scalar buffer load into `vcc_lo` can feed a pixel select when the program
establishes wave32 through an EXEC instruction. The translator loads the
buffer word and extracts the bit of the current lane inside its quad (desktop wave index
modulo 4: a host wave may have only four lanes, and the quad slot in the guest
wave is not known). The lowering is exact for masks that repeat every four bits,
such as left and right columns. The lift covers straight-line pixel programs whose remaining
instructions are waits, VCC selects, exports and termination.

`python -B tests/startup_gfx2hlsl_desc_mask_word_check.py` runs the read-only
request diagnostic on original synthetic shader words. The wave32 program
previously reported `s_buffer_load destination vcc_lo for 1 dwords`. The
diagnostic is now empty for that program and still reports the same refusal
when the wave size cannot be established or the shader consumes the whole word
as scalar data. The Python check passed on Linux x86-64 with Python 3.14.8.
The baseline diagnostic at `82f5cf5c0` reported the refusal for the same
wave32 request.

On Windows, the WARP fixture binds the mask as the second word of a raw buffer
and compares all four pixels of a 2×2 render target with the CPU reference. Two
complementary words select opposite halves. The bit-zero mutant must disagree
at two pixels for each word. This checks the quad's horizontal lane mapping
with the same mask repeated across the wave. Ordering between distinct quads
remains unverified.

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_gfx2hlsl_desc_mask_word_(controls|warp)$'
```
