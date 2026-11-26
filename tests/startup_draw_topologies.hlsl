cbuffer Entry : register(b0) { uint unused; uint start_vertex; }

// On a 16x16 viewport the vertices land on pixel centers at 4.5 and 12.5.
// Order: upper left, upper right, lower left, lower right, upper right.
float4 corner(uint id) {
    bool right = (id & 1) || id == 4;
    bool bottom = (id & 2) != 0;
    return float4(right ? 0.5625 : -0.4375, bottom ? -0.5625 : 0.4375, 0, 1);
}

float4 direct_vertex(uint id : SV_VertexID) : SV_Position { return corner(id + start_vertex); }
float4 indirect_vertex(uint id : SV_VertexID) : SV_Position { return corner(id); }
float4 indirect_line_strip_vertex(uint id : SV_VertexID) : SV_Position { return corner(id + 2); }
float4 depth_vertex(uint id : SV_VertexID) : SV_Position {
    return float4(id == 2 ? 3 : -1, id == 1 ? 3 : -1, 0, 1);
}
float4 solid_red() : SV_Target { return float4(1, 0, 0, 1); }
float solid_depth() : SV_Depth { return 0.25; }
