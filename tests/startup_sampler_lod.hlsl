// Sampler heap LOD limits (#2221): one derivative quad per draw. The root signature is the startup stand-in's
// (src/startup_stubs/translated_sampler_bindings.h): root constants at b0, one view table, the sampler
// table at space 1 and the static samplers at space 0. Every mip of the texture holds one value.
Texture2D<float> texels : register(t0);
SamplerState heap[2048] : register(s0, space1);
SamplerComparisonState shadow : register(s0);

cbuffer Probe : register(b0)
{
    uint mode;      // 0: explicit LOD, 1: static comparison, 2: implicit LOD, 3: centre reduction
    uint entry;
    float lod;
    float reference;
};

struct Vertex
{
    float4 pos : SV_Position;
    float4 uv : PARAM0;
};

Vertex vs_main(uint id : SV_VertexID)
{
    Vertex output;
    output.pos = float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
    output.uv = float4(id == 1 ? 2.0 : 0.0, id == 2 ? 2.0 : 0.0, 0.0, 0.0);
    return output;
}

float4 ps_main(Vertex input) : SV_Target
{
    if (mode == 3) return texels.SampleLevel(heap[entry], float2(0.5, 0.5), 0).xxxx;
    const float2 uv = float2(0.5625, 0.5625);
    if (mode == 0) return texels.SampleLevel(heap[entry], uv, lod).xxxx;
    // UV spans 0..1 over two pixels. On the 8x8 base mip, 1/4 of that UV gives one texel per pixel.
    if (mode == 2) return texels.Sample(heap[entry], input.uv.xy * (0.25 * exp2(lod))).xxxx;
    return texels.SampleCmpLevelZero(shadow, uv, reference).xxxx;
}
