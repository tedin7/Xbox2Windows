// Original corpus case: a root signature with one constant buffer.
cbuffer Constants : register(b0)
{
    float4x4 Transform;
    float4 Tint;
};

[RootSignature("RootFlags(0), CBV(b0)")]
float4 main(uint id : SV_VertexID) : SV_Position
{
    float4 position = float4(id & 1u, (id >> 1) & 1u, 0.0, 1.0);
    return mul(Transform, position) + Tint;
}
