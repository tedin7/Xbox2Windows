#pragma once

#include <d3d12.h>

namespace x2w_startup {

enum class TargetInteger { none, unsigned_integer, signed_integer };

inline TargetInteger target_integer(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32_UINT: case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R10G10B10A2_UINT: case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R16G16_UINT: case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R8_UINT:
        return TargetInteger::unsigned_integer;
    case DXGI_FORMAT_R32G32B32A32_SINT: case DXGI_FORMAT_R32G32B32_SINT: case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_SINT: case DXGI_FORMAT_R8G8B8A8_SINT: case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_SINT: case DXGI_FORMAT_R8G8_SINT: case DXGI_FORMAT_R16_SINT: case DXGI_FORMAT_R8_SINT:
        return TargetInteger::signed_integer;
    default:
        return TargetInteger::none;
    }
}

inline bool integer_color_format(DXGI_FORMAT format)
{
    return target_integer(format) != TargetInteger::none;
}

inline bool adjust_target_blend(D3D12_BLEND_DESC& blend, const DXGI_FORMAT* formats, UINT count)
{
    bool adjusted = false;
    if (count && blend.AlphaToCoverageEnable && integer_color_format(formats[0])) {
        blend.AlphaToCoverageEnable = FALSE;
        adjusted = true;
    }

    auto& first = blend.RenderTarget[0];
    if (!blend.IndependentBlendEnable && first.LogicOpEnable) {
        for (UINT i = 0; i < count; ++i) {
            if (target_integer(formats[i]) == TargetInteger::unsigned_integer) continue;
            // D3D12 applies a logic operation to every bound target and requires shared state.
            first.LogicOpEnable = FALSE;
            adjusted = true;
            break;
        }
    }

    for (UINT i = 0; i < count; ++i) {
        const auto kind = target_integer(formats[i]);
        const auto& effective = blend.RenderTarget[blend.IndependentBlendEnable ? i : 0];
        const bool disable_blend = effective.BlendEnable && kind != TargetInteger::none;
        const bool disable_logic = effective.LogicOpEnable && (disable_blend || kind != TargetInteger::unsigned_integer);
        if (!disable_blend && !disable_logic) continue;
        if (!blend.IndependentBlendEnable) {
            for (UINT j = 1; j < count; ++j) blend.RenderTarget[j] = first;
            blend.IndependentBlendEnable = TRUE;
        }
        if (disable_blend) blend.RenderTarget[i].BlendEnable = FALSE;
        if (disable_logic) blend.RenderTarget[i].LogicOpEnable = FALSE;
        adjusted = true;
    }
    return adjusted;
}

}  // namespace x2w_startup
