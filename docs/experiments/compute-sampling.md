# Compute sampling

Plain compute `image_sample` defaults to `SampleLevel` with a requested LOD of
zero. The console derives implicit LOD from quad derivatives, which this
lowering does not reconstruct. The approximation can change mip selection and
filtering. Sampler LOD limits still apply to the requested level. Set
`X2W_COMPUTE_IMPLICIT_LOD=0` when translating a request to retain the refusal.
Direct callers can pass `compute_implicit_lod=False` to `ComputePipeline`.

An intact four-dword sampler loaded from a fixed compute table offset uses
`SamplerDescriptorHeap`. A manifest line `smptable <index> table <parameter>
<byte-offset>` locates its source. Each dispatch snapshots the bound record,
decodes its address modes, filters, anisotropy, fixed border color and LOD
limits with the shared sampler decoder, and binds the resulting heap before
setting the compute root. A disabled mip filter pins both limits to zero.
Nonzero LOD bias, an inverted range, a reserved mip filter and a programmable
border color still cause a binding refusal. The direct heap path requires
Shader Model 6.6 and resource binding tier 3. Other successful sampler
lowerings keep their bindings and compiler profiles.

## Reproduction

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_sample_(controls|table_warp|implicit_warp)$' -V
```

The WARP fixtures read back the translated dispatch against a CPU reference.
The table fixture samples red 0.5 from a two-texel mip with linear wrap and
requires the same shader with a clamp record to read red 0. The implicit LOD
fixture uses a static sampler and adds a second mip with red 1. It requires
mip zero to read red 0.5 and a shader mutated to request mip one to read red 1.
Both controls must fail the baseline comparison. The Python controls exercise
the combined lowering, the opt-out and a modified descriptor.

On Linux x86-64, the local synthetic check passed with
`python3 -B tests/startup_compute_dispatch_texture_check.py sampler-controls`.
Against base revision `e8015da7a`, the fixed table and implicit programs
reproduced their original refusals, and 35 successful compute fixtures kept
byte-identical HLSL and manifests. WARP readback is pending on Windows.
