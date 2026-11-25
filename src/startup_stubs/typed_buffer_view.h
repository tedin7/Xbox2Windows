#pragma once

#include <d3d12.h>
#include <cstdint>
#include <optional>

namespace x2w_startup {

// gfx10 V# word 3 contains other fields besides FORMAT and DST_SEL. The
// caller supplies backing and range separately. This function encodes only
// the fields consumed by a shader's formatted load.
inline std::optional<std::uint32_t> typed_buffer_word3(const D3D12_SHADER_RESOURCE_VIEW_DESC& view)
{
    if (view.ViewDimension != D3D12_SRV_DIMENSION_BUFFER ||
        view.Buffer.Flags != D3D12_BUFFER_SRV_FLAG_NONE ||
        view.Buffer.StructureByteStride || !view.Buffer.NumElements)
        return std::nullopt;

    unsigned format = 0;
    switch (view.Format) {
    case DXGI_FORMAT_R8_UNORM: format = 1; break;
    case DXGI_FORMAT_R8_UINT: format = 5; break;
    case DXGI_FORMAT_R8G8_UNORM: format = 14; break;
    case DXGI_FORMAT_R32_UINT: format = 20; break;
    case DXGI_FORMAT_R32_FLOAT: format = 22; break;
    case DXGI_FORMAT_R16G16_UNORM: format = 23; break;
    case DXGI_FORMAT_R16G16_SNORM: format = 24; break;
    case DXGI_FORMAT_R16G16_FLOAT: format = 29; break;
    case DXGI_FORMAT_R8G8B8A8_UNORM: format = 56; break;
    case DXGI_FORMAT_R32G32_FLOAT: format = 64; break;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: format = 71; break;
    case DXGI_FORMAT_R32G32B32_FLOAT: format = 74; break;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: format = 77; break;
    default: return std::nullopt;
    }

    constexpr UINT present = D3D12_SHADER_COMPONENT_MAPPING_ALWAYS_SET_BIT_AVOIDING_ZEROMEM_MISTAKES;
    const UINT mapping = view.Shader4ComponentMapping;
    if (!(mapping & present) || (mapping & ~(present | 0xfffu))) return std::nullopt;

    std::uint32_t word = format << 12;
    for (UINT component = 0; component < 4; ++component) {
        const auto source = D3D12_DECODE_SHADER_4_COMPONENT_MAPPING(component, mapping);
        UINT selected = 0;
        switch (source) {
        case D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0:
        case D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1:
        case D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2:
        case D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3:
            selected = static_cast<UINT>(source) + 4;
            break;
        case D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0:
            break;
        case D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1:
            selected = 1;
            break;
        default:
            return std::nullopt;
        }
        word |= selected << (3 * component);
    }
    return word;
}

}  // namespace x2w_startup
