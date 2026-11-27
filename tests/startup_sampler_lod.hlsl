// Sampler heap LOD limits (#2221): one pixel per draw. The root signature is the startup stand-in's
// (src/startup_stubs/translated_sampler_bindings.h): root constants at b0, one view table, the sampler
// table at space 1 and the static samplers at space 0. Every mip of the texture holds one value.
Texture2D<float> texels : register(t0);
SamplerState heap[2048] : register(s0, space1);
SamplerComparisonState shadow : register(s0);

cbuffer Probe : register(b0)
{
    uint mode;      // 0: SampleLevel through heap[entry]; 1: SampleCmpLevelZero through the static sampler
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

float4 ps_main() : SV_Target
{
    const float2 uv = float2(0.5625, 0.5625);
    if (mode == 0) return texels.SampleLevel(heap[entry], uv, lod).xxxx;
    return texels.SampleCmpLevelZero(shadow, uv, reference).xxxx;
}
