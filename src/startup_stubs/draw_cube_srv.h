#pragma once

#include <d3d12.h>

namespace x2w_startup {

inline D3D12_SHADER_RESOURCE_VIEW_DESC null_draw_cube_srv()
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.TextureCube.MipLevels = 1;
    return desc;
}

inline bool matches_draw_cube_srv(const D3D12_RESOURCE_DESC& resource,
                                  const D3D12_SHADER_RESOURCE_VIEW_DESC* view, bool address_array = false)
{
    // The instruction uses the same dimension code for cube arrays. An opted-in address
    // texture can expose its first six faces through an explicit single-cube view.
    return view && view->ViewDimension == D3D12_SRV_DIMENSION_TEXTURECUBE &&
           resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
           (resource.DepthOrArraySize == 6 || (address_array && resource.DepthOrArraySize > 6));
}

}  // namespace x2w_startup
