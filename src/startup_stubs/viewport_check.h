#pragma once

#include <d3d12.h>
#include <cmath>

namespace x2w_startup {
inline bool sanitize_viewport(D3D12_VIEWPORT& v)
{
    bool changed = false;
    const auto clamp = [&](float& value, float low, float high, float nan_value) {
        if (std::isnan(value)) { value = nan_value; changed = true; }
        else if (value < low) { value = low; changed = true; }
        else if (value > high) { value = high; changed = true; }
    };
    clamp(v.MinDepth, 0.0f, 1.0f, 0.0f);
    clamp(v.MaxDepth, 0.0f, 1.0f, 1.0f);
    constexpr float low = static_cast<float>(D3D12_VIEWPORT_BOUNDS_MIN);
    constexpr float high = static_cast<float>(D3D12_VIEWPORT_BOUNDS_MAX);
    clamp(v.TopLeftX, low, high, 0.0f);
    clamp(v.TopLeftY, low, high, 0.0f);
    const auto extent = [&](float origin, float& size) {
        if (std::isnan(size) || size < 0.0f) { size = 0.0f; changed = true; }
        if (static_cast<double>(origin) + static_cast<double>(size) > high) {
            size = static_cast<float>(static_cast<double>(high) - origin);
            if (static_cast<double>(origin) + size > high) size = std::nextafter(size, 0.0f);
            changed = true;
        }
    };
    extent(v.TopLeftX, v.Width);
    extent(v.TopLeftY, v.Height);
    return changed;
}
inline bool sanitize_scissor(D3D12_RECT& r)
{
    bool changed = false;
    if (r.right < r.left) { r.right = r.left; changed = true; }
    if (r.bottom < r.top) { r.bottom = r.top; changed = true; }
    return changed;
}
// The stream format describes a pipeline, not a view bound to this pass.
inline DXGI_FORMAT draw_dsv_format(bool bound, DXGI_FORMAT pass_format, DXGI_FORMAT /*stream_format*/)
{
    return bound ? pass_format : DXGI_FORMAT_UNKNOWN;
}
inline bool draw_depth_state_enabled(bool has_state, DXGI_FORMAT format)
{
    return has_state && format != DXGI_FORMAT_UNKNOWN;
}
}
