#pragma once
#include <d3d12.h>

// A typed UAV load of `format` is available when Support2 has UAV_TYPED_LOAD.
// R32_FLOAT, R32_UINT and R32_SINT do not need TypedUAVLoadAdditionalFormats.
// Every other format does. Returns the refusal reason, or nullptr when a load may proceed.
inline const char* typed_uav_load_refusal(DXGI_FORMAT format, unsigned support2, bool additional_formats)
{
    const bool core = format == DXGI_FORMAT_R32_FLOAT || format == DXGI_FORMAT_R32_UINT ||
                      format == DXGI_FORMAT_R32_SINT;
    if ((support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD) != 0 && (core || additional_formats))
        return nullptr;
    return "format has no typed UAV load on this device";
}
