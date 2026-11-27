#pragma once

#include "packed_target_format.h"
#include <cmath>

namespace x2w_startup {
// Console-only formats use the nearest desktop format for creation. Their
// texel layouts differ, so this mapping does not convert resource contents.
inline constexpr DXGI_FORMAT desktop_format(DXGI_FORMAT format)
{
    switch (static_cast<UINT>(format)) {
    case 116: case 117: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case 118: return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case 119: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case 120: return DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    default: return format;
    }
}

// Preserve the clear payload and use the stand-in format when the clear
// describes the packed target that was substituted.
inline const D3D12_CLEAR_VALUE* desktop_clear_value(const D3D12_CLEAR_VALUE* clear, DXGI_FORMAT title_desc_format,
                                                   DXGI_FORMAT created_format, D3D12_CLEAR_VALUE& copy)
{
    if (!clear) return nullptr;
    DXGI_FORMAT format = desktop_format(clear->Format);
    const auto* mapping = packed_target_mapping(desktop_format(title_desc_format));
    if (mapping && created_format == mapping->replacement && format == mapping->requested)
        format = created_format;
    if (format == clear->Format) return clear;
    copy = *clear;
    copy.Format = format;
    return &copy;
}
enum class ClearDrop { none, buffer, flags, format, kind };
inline bool clear_depth_format(DXGI_FORMAT f) {
    return f == DXGI_FORMAT_D32_FLOAT || f == DXGI_FORMAT_D16_UNORM ||
           f == DXGI_FORMAT_D24_UNORM_S8_UINT || f == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
}
inline bool clear_typeless_format(DXGI_FORMAT f) {
    switch (f) {
    case DXGI_FORMAT_UNKNOWN: case DXGI_FORMAT_R32G32B32A32_TYPELESS: case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R32G32_TYPELESS: case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: case DXGI_FORMAT_X24_TYPELESS_G8_UINT: case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC2_TYPELESS: case DXGI_FORMAT_BC3_TYPELESS: case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC5_TYPELESS: case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_BC6H_TYPELESS: case DXGI_FORMAT_BC7_TYPELESS: return true;
    default: return false;
    }
}
inline const D3D12_CLEAR_VALUE* optimized_clear_value(const D3D12_CLEAR_VALUE* mapped,
        const D3D12_RESOURCE_DESC& created, D3D12_CLEAR_VALUE& copy, ClearDrop* reason = nullptr) {
    ClearDrop why = ClearDrop::none;
    if (!mapped) return nullptr;
    const bool ds = (created.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
    const bool rt = (created.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0;
    if (created.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) why = ClearDrop::buffer;
    else if (!rt && !ds) why = ClearDrop::flags;
    else if (clear_typeless_format(mapped->Format)) why = ClearDrop::format;
    else if (clear_depth_format(mapped->Format) != ds) why = ClearDrop::kind;
    if (reason) *reason = why;
    if (why != ClearDrop::none) return nullptr;
    if (ds && (!(mapped->DepthStencil.Depth >= 0.0f) || mapped->DepthStencil.Depth > 1.0f)) {
        copy = *mapped;
        copy.DepthStencil.Depth = !(mapped->DepthStencil.Depth >= 0.0f) ? 0.0f : 1.0f;
        return &copy;
    }
    return mapped;
}
}  // namespace x2w_startup
