#pragma once

#include "memory_srv_buffer.h"
#include <cstddef>
#include <cstring>

namespace x2w_startup {
inline UINT typed_dword_channels(const D3D12_SHADER_RESOURCE_VIEW_DESC &view) {
    if (view.ViewDimension != D3D12_SRV_DIMENSION_BUFFER || view.Buffer.StructureByteStride ||
        view.Buffer.Flags || view.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING)
        return 0;
    switch (view.Format) {
    case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT: case DXGI_FORMAT_R32_FLOAT: return 1;
    case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_SINT: case DXGI_FORMAT_R32G32_FLOAT: return 2;
    case DXGI_FORMAT_R32G32B32_UINT: case DXGI_FORMAT_R32G32B32_SINT: case DXGI_FORMAT_R32G32B32_FLOAT: return 3;
    case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT: case DXGI_FORMAT_R32G32B32A32_FLOAT: return 4;
    default: return 0;
    }
}

inline DXGI_FORMAT typed_dword_uint_format(UINT channels) {
    constexpr DXGI_FORMAT formats[]{DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_UINT,
                                    DXGI_FORMAT_R32G32B32_UINT, DXGI_FORMAT_R32G32B32A32_UINT};
    return channels <= 4 ? formats[channels] : DXGI_FORMAT_UNKNOWN;
}

// Only a verified materialized token supplies the slot and address. Unregistered
// entries keep their empty record (slot zero), even if their heap bytes are zero.
inline bool native_null_typed_array_srv(UINT slot, UINT64 address, const void *bytes, size_t size,
                                        UINT required_bytes, D3D12_SHADER_RESOURCE_VIEW_DESC &view) {
    if (slot != 18 || address || !bytes || size < sizeof(view)) return false;
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    std::memcpy(&desc, bytes, sizeof(desc));
    const UINT channels = typed_dword_channels(desc);
    if (!channels || channels * 4 < required_bytes) return false;
    desc.Format = typed_dword_uint_format(channels);
    view = desc;
    return true;
}

// The shader indexes raw views in bytes and structured views in elements.
inline bool native_view_stride(UINT slot, const void *bytes, std::size_t size, UINT &stride) {
    bool raw = false;
    DXGI_FORMAT format{};
    stride = 0;
    if (slot == 18) {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
        if (size < sizeof(desc)) return false;
        std::memcpy(&desc, bytes, sizeof(desc));
        if (desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER) return true;
        stride = desc.Buffer.StructureByteStride;
        raw = (desc.Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW) != 0;
        format = desc.Format;
    } else if (slot == 19) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        if (size < sizeof(desc)) return false;
        std::memcpy(&desc, bytes, sizeof(desc));
        if (desc.ViewDimension != D3D12_UAV_DIMENSION_BUFFER) return true;
        stride = desc.Buffer.StructureByteStride;
        raw = (desc.Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW) != 0;
        format = desc.Format;
    } else {
        return false;
    }
    if (!stride)
        stride = raw ? 1u : x2w_startup::typed_buffer_format_bytes(format);
    return stride > 0 && stride <= 0x3fff;
}

} // namespace x2w_startup
