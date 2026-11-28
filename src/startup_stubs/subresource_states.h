#pragma once

#include <d3d12.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace x2w_substates {

struct Range {
    UINT first_mip, mips, first_slice, slices, first_plane, planes;
};

// Planar Depth Stencil: these formats store depth in plane 0 and stencil in plane 1.
inline UINT plane_count(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return 2;
    default:
        return 1;
    }
}

inline Range whole(const D3D12_RESOURCE_DESC& resource)
{
    if (resource.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return {0, 1, 0, 1, 0, 1};
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
    return {0, mips, 0, slices, 0, plane_count(resource.Format)};
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
    // Planar Depth Stencil: the typed SRV format selects the aspect.
    switch (view->Format) {
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        range.first_plane = 0;
        break;
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
        range.first_plane = 1;
        break;
    default:
        break;
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

inline Range dsv(const D3D12_RESOURCE_DESC& resource, const D3D12_DEPTH_STENCIL_VIEW_DESC* view)
{
    const Range full = whole(resource);
    // CreateDepthStencilView: a default DSV covers the first mip and all array slices.
    if (!view) return {0, 1, 0, full.slices, 0, full.planes};
    Range range{0, 1, 0, 1, 0, full.planes};
    switch (view->ViewDimension) {
    case D3D12_DSV_DIMENSION_TEXTURE1D:
        range.first_mip = view->Texture1D.MipSlice;
        break;
    case D3D12_DSV_DIMENSION_TEXTURE1DARRAY:
        range = {view->Texture1DArray.MipSlice, 1, view->Texture1DArray.FirstArraySlice,
                 view->Texture1DArray.ArraySize, 0, full.planes};
        break;
    case D3D12_DSV_DIMENSION_TEXTURE2D:
        range.first_mip = view->Texture2D.MipSlice;
        break;
    case D3D12_DSV_DIMENSION_TEXTURE2DARRAY:
        range = {view->Texture2DArray.MipSlice, 1, view->Texture2DArray.FirstArraySlice,
                 view->Texture2DArray.ArraySize, 0, full.planes};
        break;
    case D3D12_DSV_DIMENSION_TEXTURE2DMS:
        break;
    case D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY:
        range.first_slice = view->Texture2DMSArray.FirstArraySlice;
        range.slices = view->Texture2DMSArray.ArraySize;
        break;
    default:
        return full;
    }
    return clamp(range, full);
}

inline Range plane(Range range, UINT index)
{
    range.first_plane = index;
    range.planes = 1;
    return range;
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

struct DepthBinding {
    Range range{};
    D3D12_RESOURCE_STATES states[2]{D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_DEPTH_WRITE};
};

inline DepthBinding bind_depth(Range range, UINT flags)
{
    DepthBinding binding{range};
    if (flags & D3D12_DSV_FLAG_READ_ONLY_DEPTH) binding.states[0] = D3D12_RESOURCE_STATE_DEPTH_READ;
    if (flags & D3D12_DSV_FLAG_READ_ONLY_STENCIL) binding.states[1] = D3D12_RESOURCE_STATE_DEPTH_READ;
    return binding;
}

inline bool writable_overlap(const DepthBinding& binding, const Range& read)
{
    for (UINT p = 0; p < binding.range.planes; ++p)
        if (binding.states[p] == D3D12_RESOURCE_STATE_DEPTH_WRITE && overlaps(read, plane(binding.range, p)))
            return true;
    return false;
}

inline D3D12_RESOURCE_STATES attachment_state(D3D12_RESOURCE_STATES bind_state, D3D12_RESOURCE_STATES current)
{
    if (bind_state != D3D12_RESOURCE_STATE_DEPTH_READ) return bind_state;
    // Planar Depth Stencil, Resource Barrier Transitions: a read-only plane can also be a shader resource.
    return bind_state | (current & (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
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

inline D3D12_RESOURCE_STATES sampled_state(const Split& split, UINT index, const DepthBinding& binding,
                                          D3D12_RESOURCE_STATES read_state)
{
    const UINT mip = index % split.mips;
    const UINT slice = index / split.mips % split.slices;
    const UINT p = index / (split.mips * split.slices);
    if (p < binding.range.planes && binding.states[p] == D3D12_RESOURCE_STATE_DEPTH_READ &&
        overlaps(binding.range, {mip, 1, slice, 1, p, 1}))
        return read_state | D3D12_RESOURCE_STATE_DEPTH_READ;
    return read_state;
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
template <class List, class StateFor>
bool move_each(List* list, Split& split, const Range& range, StateFor state_for)
{
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    visit(split, range, [&](UINT index) {
        auto& from = split.state[index];
        const auto to = state_for(index, from);
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
bool move(List* list, Split& split, const Range& range, D3D12_RESOURCE_STATES to)
{
    return move_each(list, split, range, [to](UINT, D3D12_RESOURCE_STATES) { return to; });
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
