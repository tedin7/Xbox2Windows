// A depth stencil view with one read-only plane (D3D12_DSV_FLAG_READ_ONLY_DEPTH or _STENCIL, not both): the read-only
// plane rests in DEPTH_READ (alone, or with the shader-resource bits when the pass samples it), the other plane in
// DEPTH_WRITE. A whole-resource barrier cannot express that, so each plane is moved on its own, one barrier per subresource
// of the plane (subresource = mip + slice * mips + plane * mips * slices). Planes rest in COMMON between lists.
#pragma once

#include <d3d12.h>

#include <vector>

namespace x2w_dsv {

struct Planes {
    ID3D12Resource* resource = nullptr;
    D3D12_RESOURCE_STATES state[2] = {D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COMMON};  // plane 0 depth, plane 1 stencil
    UINT mips = 1, slices = 1;
};

inline Planes planes_of(ID3D12Resource* resource)
{
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    Planes planes;
    planes.resource = resource;
    planes.mips = desc.MipLevels ? desc.MipLevels : 1;
    planes.slices = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D || !desc.DepthOrArraySize ? 1 : desc.DepthOrArraySize;
    return planes;
}

inline UINT subresource(const Planes& planes, UINT plane, UINT mip, UINT slice) { return mip + slice * planes.mips + plane * planes.mips * planes.slices; }

// Moves one plane to `to` with a barrier per subresource; nothing when it is there already.
inline void move_plane(ID3D12GraphicsCommandList* list, Planes& planes, UINT plane, D3D12_RESOURCE_STATES to)
{
    if (planes.state[plane] == to) return;
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    barriers.reserve(static_cast<std::size_t>(planes.mips) * planes.slices);
    for (UINT slice = 0; slice < planes.slices; ++slice)
        for (UINT mip = 0; mip < planes.mips; ++mip) {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = planes.resource;
            barrier.Transition.Subresource = subresource(planes, plane, mip, slice);
            barrier.Transition.StateBefore = planes.state[plane];
            barrier.Transition.StateAfter = to;
            barriers.push_back(barrier);
        }
    if (!barriers.empty()) list->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
    planes.state[plane] = to;
}

// From COMMON into the two plane states of a view.
inline void bind(ID3D12GraphicsCommandList* list, Planes& planes, D3D12_RESOURCE_STATES depth, D3D12_RESOURCE_STATES stencil)
{
    move_plane(list, planes, 0, depth);
    move_plane(list, planes, 1, stencil);
}

// Back to COMMON, plane by plane.
inline void release(ID3D12GraphicsCommandList* list, Planes& planes)
{
    move_plane(list, planes, 0, D3D12_RESOURCE_STATE_COMMON);
    move_plane(list, planes, 1, D3D12_RESOURCE_STATE_COMMON);
}

}  // namespace x2w_dsv
