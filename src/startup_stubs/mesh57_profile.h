#pragma once
#include "mesh_pixel_heap.h"
#include <d3d12.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace x2w_startup
{
inline std::vector<unsigned char> normalize_mesh57_descriptor(std::vector<unsigned char> desc)
{
    if (desc.size() < 15 * sizeof(std::uint64_t))
        return {};
    for (const auto index : {0, 1, 3, 5, 13})
        std::memset(desc.data() + index * sizeof(std::uint64_t), 0, sizeof(std::uint64_t));
    return desc;
}
// A reviewed fixed-state/entry record. The header is followed by the exact normalized fixed
// descriptor, the two register blocks, creation root blob, guest MS and guest PS, in this order.
// Only the five explicitly captured pointer fields are normalized in the descriptor.
struct Mesh57Profile {
    char magic[8];
    UINT lengths[6];
    UINT threads, vertices, primitives, topology, group_stride, wave_size;
    D3D12_BLEND_DESC blend;
    D3D12_RASTERIZER_DESC rasterizer;
    D3D12_DEPTH_STENCIL_DESC depth;
    D3D12_RT_FORMAT_ARRAY targets;
    DXGI_FORMAT depth_format;
    DXGI_SAMPLE_DESC samples;
    UINT sample_mask;
};
static_assert(sizeof(Mesh57Profile) == 532, "update the offline mesh57 profile reader when the ABI changes");
// Version 2 stores this immediately after Mesh57Profile, before the six byte spans.
struct Mesh57PixelEntry {
    UINT enable, address, wave_size;
};
// Version 3 adds one reviewed MS entry mapping after the version 2 pixel entry.
struct Mesh57ConstantEntry {
    UINT sgpr, root_parameter, count;
};
// Version 4 adds a count followed by this many MS table/root-address SGPR mappings.
// Version 5 then stores the physical resource-heap SGPR; selector and view class come from the MS.
struct Mesh57ViewEntry {
    UINT sgpr, root_parameter, root_address;
};
struct Mesh57BufferRead {
    UINT root_parameter, offset, minimum_bytes;
    bool root_address;
};
} // namespace x2w_startup
