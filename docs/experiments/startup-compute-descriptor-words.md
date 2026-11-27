# Compute descriptor words and wide heap reads

Dispatch supplies raw non-address descriptor words and the result of an
encoded address-pair null comparison as entry values. The runtime resolves
table locations and indexed heap locations even when the shader never binds
the descriptor as an SRV or UAV. For desktop view tokens it handles buffer
record counts and the dimensions word of a 2D SRV. A dispatch reports an
explicit refusal for token words whose full encoding it cannot derive.

A wide heap load can supply separate buffers, images and samplers. Each
binding uses its own byte offset within the load. A compute sampler resolved
this way is recorded into a shader-visible heap with the same submission
lifetime as a translated draw's sampler heap. For groups of at most 32 threads,
a conditional select can use each lane's bit from a raw descriptor word.
The special VCC low-word move currently requires a straight-line program.

On x64 Windows with the development preset, run the following WARP readback
tests after building the usual test targets.

```powershell
ctest --preset dev --no-tests=error -V -R '^startup_compute_dispatch_desc_(word|mask|pair|mixed)_warp$'
```

The checks compare all output words with a CPU reference after the submission
fence. The mutants change the descriptor word, invert the null condition,
shift the mask into its upper half, reuse the first image offset, or select
the other sampler. Each mutant must dispatch and produce a mismatch in the
written range. Linux translator checks exercise the address-word refusal and
the two-dword heap-load refusal. They also check the new manifests and retain
the legacy fixture's exact HLSL and manifest assertions.
