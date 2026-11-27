#pragma once

#include "packed_target_format.h"

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
}  // namespace x2w_startup
