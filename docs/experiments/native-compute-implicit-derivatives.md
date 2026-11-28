# Implicit derivatives in direct compute

Groups with a multiple of four threads can use implicit derivatives when
they pass the existing wave guards: a group of more than 32 threads must be
a multiple of 32, and a wave64 group a multiple of 64. The DXIL module uses
`numthreads(X*Y*Z,1,1)` and validator version 1.8 only when it emits
compute derivatives. Guest local X, Y and Z are reconstructed from the
linear guest index.

A guest quad is four consecutive guest lanes. The emitter reconstructs
the linear guest index as host wave rank * 32 + host lane index. The
SM 6.6 derivatives spec section "1D quads" guarantees the matching position
within each quad. The full reconstruction also assumes that host lane
order follows thread order within a wave, including across quads.

A 6-thread wave32 group has an incomplete quad and refuses with
"implicit derivatives require a pixel stage". A 36-thread wave32 group
has complete quads and fails the earlier wave guard with "partial final
wave in multi-wave group". Implicit Gather with a constant sampler requires
that sampler to be fixed to mip zero (LOD range word zero).

```powershell
cmake --build --preset dev --target compile_dxil offscreen_check
ctest --preset dev --no-tests=error -j 1 -R '^native_compute_implicit_lod_' -V
```

On an NVIDIA GeForce RTX 5080 (Windows 11, Release), the hardware fixture
matches its CPU reference for 4-, 12-, 32- and 96-thread wave32 groups,
64- and 128-thread wave64 groups, and the multidimensional (8,4,1) and
(8,4,2) groups. The multidimensional fixture varies the texture coordinate
only in the quad's vertical direction. Its derivative spans 0.75 texels
for consecutive quads and three texels for spatial 2D quads. This keeps
the expected sample below the first mip transition and leaves the wrong
quad layout observable with the fixture's trilinear sampler.
The compile controls check that implicit shaders emit flattened DXIL
group dimensions and explicit-LOD shaders keep the requested dimensions.
Its other readbacks cover bias selecting mip 2, point depth
comparison at mip 1, both LOD query outputs, and the four taps of mip-zero
Gather. Bias and Gather each use the same CPU reference for their implicit
and explicit-LOD shaders. Inbox WARP reports the capability skip
"requires SM 6.6, WaveOps, and a 32-lane wave".
