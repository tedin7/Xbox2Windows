// Each group covers one half of a 4 x 2 target with two triangles.
struct Vertex {
    float4 position : SV_Position;
    nointerpolation uint section : TEXCOORD0;
};

[outputtopology("triangle")]
[numthreads(4, 1, 1)]
void mesh(uint lane : SV_GroupIndex, uint3 group : SV_GroupID,
          out vertices Vertex corners[4], out indices uint3 faces[2])
{
    SetMeshOutputCounts(4, 2);
    Vertex corner;
    corner.position = float4(float(group.x + (lane & 1)) - 1.0,
                             (lane & 2) ? -1.0 : 1.0, 0.5, 1.0);
    corner.section = group.x;
    corners[lane] = corner;
    if (lane < 2)
        faces[lane] = lane == 0 ? uint3(0, 1, 2) : uint3(2, 1, 3);
}

float4 pixel(Vertex input) : SV_Target0
{
    return input.section ? float4(0, 1, 0, 1) : float4(1, 0, 0, 1);
}
