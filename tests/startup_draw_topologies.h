#pragma once

#include <cstdint>

namespace draw_topology_fixture {
struct Shape {
    D3D12_PRIMITIVE_TOPOLOGY topology;
    UINT count;
    UINT first;
    const char* label;
    std::uint16_t rows[16]; // bit x in rows[y] denotes a covered pixel
};

// Triangle top-left fill and aliased diamond-exit line coverage. The last
// endpoint of each segment is excluded. Points cover their center pixels.
constexpr Shape strip{D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP, 4, 0, "triangle strip",
    {0, 0, 0, 0, 0x0ff0, 0x0ff0, 0x0ff0, 0x0ff0, 0x0ff0, 0x0ff0, 0x0ff0, 0x0ff0}};
constexpr Shape lines{D3D_PRIMITIVE_TOPOLOGY_LINELIST, 4, 0, "line list",
    {0, 0, 0, 0, 0x0ff0, 0, 0, 0, 0, 0, 0, 0, 0x0ff0}};
constexpr Shape line_strip{D3D_PRIMITIVE_TOPOLOGY_LINESTRIP, 3, 2, "line strip",
    {0, 0, 0, 0, 0, 0x1000, 0x1000, 0x1000, 0x1000, 0x1000, 0x1000, 0x1000, 0x1ff0}};
constexpr Shape points{D3D_PRIMITIVE_TOPOLOGY_POINTLIST, 4, 0, "point list",
    {0, 0, 0, 0, 0x1010, 0, 0, 0, 0, 0, 0, 0, 0x1010}};
constexpr Shape patch{D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST, 4, 0, "patch refusal", {}};
}
