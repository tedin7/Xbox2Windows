// Original corpus case: a fixed three-vertex mesh shader.
struct Vertex
{
    float4 position : SV_Position;
};

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(out vertices Vertex verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    verts[0].position = float4(-1.0, -1.0, 0.0, 1.0);
    verts[1].position = float4(1.0, -1.0, 0.0, 1.0);
    verts[2].position = float4(0.0, 1.0, 0.0, 1.0);
    tris[0] = uint3(0, 1, 2);
}
