#pragma once
#include "gfx1013_compute_native.h"
#include <dxgiformat.h>

namespace x2w {
struct NativeHalfStoreFormat {
    DXGI_FORMAT typed{}, typeless{};
    bool accepts(DXGI_FORMAT view, DXGI_FORMAT resource) const {
        return view == typed && (resource == typed || resource == typeless);
    }
};
inline NativeHalfStoreFormat native_half_store_format(ComputeViewKind kind) {
    switch (kind) {
    case ComputeViewKind::half_scalar_uav: return {DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_TYPELESS};
    case ComputeViewKind::half2_texture_uav: return {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_TYPELESS};
    case ComputeViewKind::half4_texture_uav: return {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_TYPELESS};
    default: return {};
    }
}
}
