#pragma once
#include <string>
#include <string_view>

namespace x2w {
// Raw straight-line vertex entry with SV_VertexID in v0, POS0 and PARAM0.
// Root constant b0.x selects the runtime's rectangle-list expansion.
std::string translate_gfx1013_vertex_id(std::string_view code);

struct BufferedVertex {
    std::string hlsl;
    std::string bindings;
};
// Vertex-buffer table in s[0:1], base vertex in s2, root tables in s8/s9.
// One indexed raw vertex stream, float PARAM0/1, no input-layout fetch routine.
BufferedVertex translate_gfx1013_buffered_vertex(std::string_view code);
}
