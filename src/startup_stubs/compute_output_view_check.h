// Pure validation for translated compute output views.
#pragma once
#include <d3d12.h>
#include "pixel_uav_support.h"

namespace x2w_startup {
inline const char* compute_output_shape_reason(const D3D12_RESOURCE_DESC& resource,
        const D3D12_UNORDERED_ACCESS_VIEW_DESC& desc, bool described, bool array, int dimension)
{
    const char* reason = nullptr;
    if (array) {
        // The layer in HLSL is relative to FirstArraySlice: retain the caller's complete view.
        const auto& a = desc.Texture2DArray;
        if (!described || desc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE2DARRAY)
            reason = "an output texture view is not a 2D array";
        else if (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.SampleDesc.Count != 1 ||
                 a.PlaneSlice || a.MipSlice >= resource.MipLevels || a.FirstArraySlice >= resource.DepthOrArraySize ||
                 !a.ArraySize || a.ArraySize > resource.DepthOrArraySize - a.FirstArraySlice)
            reason = "an output texture array view is outside the resource";
    } else if (dimension == 1) {
        if (!described || desc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE1D)
            reason = "an output texture view is not 1D";
        else if (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D || desc.Texture1D.MipSlice >= resource.MipLevels)
            reason = "an output texture view is outside the resource";
    } else if (dimension == 3) {
        // The W coordinate in HLSL is relative to FirstWSlice: retain the caller's complete view.
        const auto& v = desc.Texture3D;
        if (!described || desc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE3D)
            reason = "an output texture view is not 3D";
        else if (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D || v.MipSlice >= resource.MipLevels ||
                 !v.WSize || v.FirstWSlice >= (resource.DepthOrArraySize >> v.MipSlice ? resource.DepthOrArraySize >> v.MipSlice : 1u) ||
                 (v.WSize != 0xffffffffu && v.WSize > (resource.DepthOrArraySize >> v.MipSlice ? resource.DepthOrArraySize >> v.MipSlice : 1u) - v.FirstWSlice))
            reason = "an output texture view is outside the resource";
    } else if (described ? desc.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE2D || desc.Texture2D.PlaneSlice
                       : resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.DepthOrArraySize != 1)
        reason = "an output texture view is not 2D";
    else if (resource.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || resource.SampleDesc.Count != 1 ||
             (described && desc.Texture2D.MipSlice >= resource.MipLevels))
        reason = "an output texture view is outside the resource";
    return reason;
}
inline const char* compute_output_support_reason(unsigned bits, bool load)
{
    return pixel_uav_support_reason(bits, load);
}
}
