ByteAddressBuffer source : register(t0);

struct Vertex {
    float4 position : SV_Position;
    nointerpolation uint word : WORD;
};

#ifndef X2W_MESH_DUAL
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
    SetMeshOutputCounts(3, 1);
    uint word = source.Load(0);
    verts[0].position = float4(-1, -1, 0, 1);
    verts[1].position = float4(-1,  3, 0, 1);
    verts[2].position = float4( 3, -1, 0, 1);
    verts[0].word = word;
    verts[1].word = word;
    verts[2].word = word;
    triangles[0] = uint3(0, 1, 2);
}
#else
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh(uint3 group : SV_GroupID, out vertices Vertex verts[4], out indices uint3 triangles[2]) {
    SetMeshOutputCounts(4, 2);
    uint word = source.Load(0);
    float left = float(group.x) - 1.0;
    float right = left + 1.0;
    verts[0].position = float4(left, -1, 0, 1);
    verts[1].position = float4(left, 1, 0, 1);
    verts[2].position = float4(right, -1, 0, 1);
    verts[3].position = float4(right, 1, 0, 1);
    verts[0].word = word;
    verts[1].word = word;
    verts[2].word = word;
    verts[3].word = word;
    triangles[0] = uint3(0, 1, 2);
    triangles[1] = uint3(2, 1, 3);
}
#endif

Vertex vertex(uint index : SV_VertexID) {
    Vertex result;
    result.position = float4(index == 2 ? 3 : -1, index == 1 ? 3 : -1, 0, 1);
    result.word = source.Load(0);
    return result;
}

#ifdef X2W_MESH_LEASE_COLOR
float4 pixel(Vertex vertex) : SV_Target {
    uint word = vertex.word;
    return float4(word & 255, (word >> 8) & 255, (word >> 16) & 255, word >> 24) / 255.0;
}
#else
uint pixel(Vertex vertex) : SV_Target {
    return vertex.word;
}
#endif
