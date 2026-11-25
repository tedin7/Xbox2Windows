struct Vertex {
    float4 position : SV_Position;
    nointerpolation uint side : SIDE;
};

void emit_triangle(uint side, out Vertex verts[3], out uint3 triangles[1]) {
    SetMeshOutputCounts(3, 1);
    verts[0].position = float4(-1, -1, 0, 1);
    verts[1].position = float4(-1, 3, 0, 1);
    verts[2].position = float4(3, -1, 0, 1);
    verts[0].side = side;
    verts[1].side = side;
    verts[2].side = side;
    triangles[0] = uint3(0, 1, 2);
}

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void mesh(uint3 group : SV_GroupID, out vertices Vertex verts[3], out indices uint3 triangles[1]) {
    emit_triangle(group.x, verts, triangles);
}

struct Payload { uint side; };

[numthreads(1, 1, 1)]
void amplify(uint3 group : SV_GroupID) {
    Payload payload;
    payload.side = group.x;
    DispatchMesh(1, 1, 1, payload);
}

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void amplified_mesh(in payload Payload payload, out vertices Vertex verts[3], out indices uint3 triangles[1]) {
    emit_triangle(payload.side, verts, triangles);
}

float4 pixel(Vertex input) : SV_Target {
    if (input.side == 0 && input.position.x < 2) return float4(1, 0, 0, 1);
    if (input.side == 1 && input.position.x >= 2) return float4(0, 1, 0, 1);
    discard;
    return float4(0, 0, 0, 1);
}
