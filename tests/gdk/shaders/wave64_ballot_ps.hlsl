// Original corpus case: wave ballot and lane index in a pixel stage.
Texture2D<float4> Color : register(t0);
SamplerState Linear : register(s0);

float4 main(float2 uv : TEXCOORD0) : SV_Target
{
    uint4 ballot = WaveActiveBallot(uv.x > 0.5);
    uint count = WaveActiveCountBits(uv.y > 0.5);
    float4 color = Color.Sample(Linear, uv);
    return color + float4(ballot.x & 1u, count & 1u, WaveGetLaneIndex() & 1u, 0.0);
}
