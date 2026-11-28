# GDK shader corpus

tools/gdk_shader_corpus.py compiles original HLSL with the GXDK Scarlett dxc,
retargets the Xbox container with tools/dxil_retarget.py, and records the
source and container SHA-256 values. The HLSL sources are versioned; the .cso
files and the corpus manifest stay outside the repository.

    python -B tests/gdk_shader_corpus_check.py
    python -B tools/gdk_shader_corpus.py build --gdk-root D:\gdk-ref\261000\gxdk\Microsoft GDK\261000 --corpus tests\gdk\shaders --out D:\gdk-oracle\knowledge\shaders\corpus
    python -B tools/gdk_shader_corpus.py verify --manifest D:\gdk-oracle\knowledge\shaders\corpus\corpus.json

Measured on 28 November 2026 with dxcompiler_xs 1.8, build
2610.2608.12522.14643:

| Case | Profile | Xbox bytes | Desktop bytes | Triple |
|---|---|---:|---:|---|
| wave64_ballot_ps | ps_6_0 | 4,316 | 4,152 | dxil-ms-xb2 -> dxil-ms-dx |
| root_constants_vs | vs_6_0 | 3,972 | 3,804 | dxil-ms-xb2 -> dxil-ms-dx |
| image_sample_c_lz_ps | ps_6_0 | 3,912 | 3,756 | dxil-ms-xb2 -> dxil-ms-dx |
| packed_types_cs | cs_6_0 | 3,224 | 3,068 | dxil-ms-xb2 -> dxil-ms-dx |
| derivatives_ps | ps_6_0 | 3,204 | 3,048 | dxil-ms-xb2 -> dxil-ms-dx |
| mesh_ngg_ms | ms_6_5 | 3,104 | 2,944 | dxil-ms-xb2 -> dxil-ms-dx |

All six cases compiled, retargeted and verified. The retarget removed the
xbox.compiler.options named metadata in every case. The sources cover wave
ballot and lane access, a root signature with one constant buffer, a
comparison sample at level zero, packed 16-bit fields in compute, implicit
derivatives and fwidth, and a fixed three-vertex mesh shader.

The check runs on Linux without dxc and covers case loading, hashing, manifest
verification, triple checks and the missing-dxc skip. build returns 77 when
dxc is absent.

Limits: a retargeted container is not a GPU execution result. The corpus does
not prove that a desktop driver accepts the mesh or pixel pipeline, and it does
not cover game shaders. It records the compiler version, the source SHA-256,
the container sizes and the metadata the retarget removed. The mesh case is a
fixed topology and does not exercise NGG culling or amplification.
