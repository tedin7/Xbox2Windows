#pragma once

#include <d3d12.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace x2w_substates {

struct Range {
    UINT first_mip, mips, first_slice, slices, first_plane, planes;
};

inline Range whole(const D3D12_RESOURCE_DESC& resource)
{
    UINT mips = resource.MipLevels;
    if (!mips) {
        UINT64 extent = (std::max)(resource.Width, UINT64{resource.Height});
        if (resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
            extent = (std::max)(extent, UINT64{resource.DepthOrArraySize});
        for (mips = 1; extent > 1; extent >>= 1) ++mips;
    }
    // Resource Heaps, D3D12CalcSubresource: a 3D texture has one array slice.
    const UINT slices = resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
        ? 1u : (std::max)(1u, UINT{resource.DepthOrArraySize});
    return {0, mips, 0, slices, 0, 1};
}

inline Range clamp(Range range, const Range& full)
{
    if (!range.mips || !range.slices || !range.planes ||
        range.first_mip >= full.mips || range.first_slice >= full.slices ||
        range.first_plane >= full.planes)
        return full;
    range.mips = (std::min)(range.mips, full.mips - range.first_mip);
    range.slices = (std::min)(range.slices, full.slices - range.first_slice);
    range.planes = (std::min)(range.planes, full.planes - range.first_plane);
    return range;
}

inline Range srv(const D3D12_RESOURCE_DESC& resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* view)
{
    const Range full = whole(resource);
    // CreateShaderResourceView: a default SRV covers the resource.
    if (!view) return full;
    Range range{0, 1, 0, 1, 0, 1};
    switch (view->ViewDimension) {
    case D3D12_SRV_DIMENSION_TEXTURE1D:
        range.first_mip = view->Texture1D.MostDetailedMip;
        range.mips = view->Texture1D.MipLevels;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
        range = {view->Texture1DArray.MostDetailedMip, view->Texture1DArray.MipLevels,
                 view->Texture1DArray.FirstArraySlice, view->Texture1DArray.ArraySize, 0, 1};
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2D:
        range.first_mip = view->Texture2D.MostDetailedMip;
        range.mips = view->Texture2D.MipLevels;
        range.first_plane = view->Texture2D.PlaneSlice;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
        range = {view->Texture2DArray.MostDetailedMip, view->Texture2DArray.MipLevels,
                 view->Texture2DArray.FirstArraySlice, view->Texture2DArray.ArraySize,
                 view->Texture2DArray.PlaneSlice, 1};
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMS:
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY:
        range.first_slice = view->Texture2DMSArray.FirstArraySlice;
        range.slices = view->Texture2DMSArray.ArraySize;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE3D:
        range.first_mip = view->Texture3D.MostDetailedMip;
        range.mips = view->Texture3D.MipLevels;
        break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE:
        range = {view->TextureCube.MostDetailedMip, view->TextureCube.MipLevels, 0, 6, 0, 1};
        break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
        range = {view->TextureCubeArray.MostDetailedMip, view->TextureCubeArray.MipLevels,
                 view->TextureCubeArray.First2DArrayFace,
                 static_cast<UINT>((std::min)(UINT64{view->TextureCubeArray.NumCubes} * 6, UINT64{full.slices})), 0, 1};
        break;
    default:  // Buffer, raytracing acceleration structure or unknown.
        return full;
    }
    return clamp(range, full);
}

inline Range rtv(const D3D12_RESOURCE_DESC& resource, const D3D12_RENDER_TARGET_VIEW_DESC* view)
{
    const Range full = whole(resource);
    // CreateRenderTargetView: a default RTV covers the first mip and all array slices.
    if (!view) return {0, 1, 0, full.slices, 0, full.planes};
    Range range{0, 1, 0, 1, 0, 1};
    switch (view->ViewDimension) {
    case D3D12_RTV_DIMENSION_TEXTURE1D:
        range.first_mip = view->Texture1D.MipSlice;
        break;
    case D3D12_RTV_DIMENSION_TEXTURE1DARRAY:
        range = {view->Texture1DArray.MipSlice, 1, view->Texture1DArray.FirstArraySlice,
                 view->Texture1DArray.ArraySize, 0, 1};
        break;
    case D3D12_RTV_DIMENSION_TEXTURE2D:
        range.first_mip = view->Texture2D.MipSlice;
        range.first_plane = view->Texture2D.PlaneSlice;
        break;
    case D3D12_RTV_DIMENSION_TEXTURE2DARRAY:
        range = {view->Texture2DArray.MipSlice, 1, view->Texture2DArray.FirstArraySlice,
                 view->Texture2DArray.ArraySize, view->Texture2DArray.PlaneSlice, 1};
        break;
    case D3D12_RTV_DIMENSION_TEXTURE2DMS:
        break;
    case D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY:
        range.first_slice = view->Texture2DMSArray.FirstArraySlice;
        range.slices = view->Texture2DMSArray.ArraySize;
        break;
    case D3D12_RTV_DIMENSION_TEXTURE3D:
        range.first_mip = view->Texture3D.MipSlice;
        break;
    default:  // Buffer or unknown.
        return full;
    }
    return clamp(range, full);
}

inline bool overlaps(const Range& a, const Range& b)
{
    const auto axis = [](UINT x, UINT n, UINT y, UINT m) {
        return n && m && x < UINT64{y} + m && y < UINT64{x} + n;
    };
    return axis(a.first_mip, a.mips, b.first_mip, b.mips) &&
           axis(a.first_slice, a.slices, b.first_slice, b.slices) &&
           axis(a.first_plane, a.planes, b.first_plane, b.planes);
}

struct Split {
    ID3D12Resource* resource;
    UINT mips, slices, planes;
    std::vector<D3D12_RESOURCE_STATES> state;
};

inline Split make_split(ID3D12Resource* resource, const Range& full, D3D12_RESOURCE_STATES state)
{
    return {resource, full.mips, full.slices, full.planes,
            std::vector<D3D12_RESOURCE_STATES>(full.mips * full.slices * full.planes, state)};
}

// Resource Heaps, D3D12CalcSubresource.
template <class Visitor>
void visit(const Split& split, const Range& range, Visitor visitor)
{
    for (UINT p = range.first_plane; p < range.first_plane + range.planes; ++p)
        for (UINT s = range.first_slice; s < range.first_slice + range.slices; ++s)
            for (UINT m = range.first_mip; m < range.first_mip + range.mips; ++m)
                visitor(m + s * split.mips + p * split.mips * split.slices);
}

// Enhanced Barriers, D3D12 Legacy Resource Barrier Shortcomings: legacy transitions
// name one subresource or all subresources.
template <class List>
bool move(List* list, Split& split, const Range& range, D3D12_RESOURCE_STATES to)
{
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    visit(split, range, [&](UINT index) {
        auto& from = split.state[index];
        if (from == to) return;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {split.resource, index, from, to};
        barriers.push_back(barrier);
        from = to;
    });
    if (barriers.empty()) return false;
    list->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
    return true;
}

template <class List>
bool move_all(List* list, Split& split, D3D12_RESOURCE_STATES to)
{
    return move(list, split, {0, split.mips, 0, split.slices, 0, split.planes}, to);
}

inline bool uniform(const Split& split, D3D12_RESOURCE_STATES* state)
{
    if (split.state.empty()) return false;
    const auto first = split.state.front();
    if (!std::all_of(split.state.begin(), split.state.end(), [first](auto value) { return value == first; }))
        return false;
    *state = first;
    return true;
}

}  // namespace x2w_substates
