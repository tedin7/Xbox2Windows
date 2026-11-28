// Original corpus case: comparison sample at an explicit level.
Texture2D<float4> Depth : register(t0);
SamplerComparisonState Shadow : register(s0);

float4 main(float2 uv : TEXCOORD0) : SV_Target
{
    return Depth.SampleCmpLevelZero(Shadow, uv, 0.5);
}
