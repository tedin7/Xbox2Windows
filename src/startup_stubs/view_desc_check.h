// Device-independent validation of desktop SRVs.
#pragma once

#include "startup_stubs/list_copy.h"
#include "startup_stubs/memory_srv_buffer.h"
#include "startup_stubs/texture_drop_mips.h"

#include <cmath>

namespace x2w_startup::view_desc {

enum class Refusal {
    none, dimension, mip_range, array_range, format, plane, lod, denied, dropped_mips, buffer_range, count
};

enum class Field { mip_levels, array_size, num_cubes, mapping, format, count };

inline bool valid_mapping(UINT mapping)
{
    constexpr UINT required = D3D12_SHADER_COMPONENT_MAPPING_ALWAYS_SET_BIT_AVOIDING_ZEROMEM_MISTAKES;
    if ((mapping & required) == 0 || (mapping & ~(required | 0xfffu)) != 0) return false;
    for (unsigned component = 0; component < 4; ++component)
        if (((mapping >> (3 * component)) & 7u) > 5) return false;
    return true;
}

inline bool depth_format(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_D16_UNORM || format == DXGI_FORMAT_D32_FLOAT ||
        format == DXGI_FORMAT_D24_UNORM_S8_UINT || format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
}

enum class NumericFormat { other, floating, snorm, unorm };

inline NumericFormat numeric_format(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_BC6H_UF16: case DXGI_FORMAT_BC6H_SF16:
        return NumericFormat::floating;
    case DXGI_FORMAT_R16G16B16A16_SNORM: case DXGI_FORMAT_R16G16_SNORM: case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SNORM: case DXGI_FORMAT_R8G8_SNORM: case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_BC4_SNORM: case DXGI_FORMAT_BC5_SNORM:
        return NumericFormat::snorm;
    case DXGI_FORMAT_R16G16B16A16_UNORM: case DXGI_FORMAT_R16G16_UNORM: case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_BC4_UNORM: case DXGI_FORMAT_BC5_UNORM:
        return NumericFormat::unorm;
    default:
        return NumericFormat::other;
    }
}

inline bool format_cast_fits(DXGI_FORMAT resource, DXGI_FORMAT view, bool fully_typed_casting)
{
    if (resource == view) return true;
    if (!compatible_formats(resource, view)) return false;
    if (is_typeless(resource)) return true;
    if (!fully_typed_casting || depth_format(resource)) return false;
    const auto from = numeric_format(resource);
    const auto to = numeric_format(view);
    if ((from == NumericFormat::floating) != (to == NumericFormat::floating)) return false;
    return !((from == NumericFormat::snorm && to == NumericFormat::unorm) ||
             (from == NumericFormat::unorm && to == NumericFormat::snorm));
}

inline D3D12_SRV_DIMENSION default_dimension(const D3D12_RESOURCE_DESC& resource)
{
    const bool array = resource.DepthOrArraySize > 1;
    switch (resource.Dimension) {
    case D3D12_RESOURCE_DIMENSION_BUFFER: return D3D12_SRV_DIMENSION_BUFFER;
    case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
        return array ? D3D12_SRV_DIMENSION_TEXTURE1DARRAY : D3D12_SRV_DIMENSION_TEXTURE1D;
    case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
        if (resource.SampleDesc.Count > 1)
            return array ? D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY : D3D12_SRV_DIMENSION_TEXTURE2DMS;
        return array ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY : D3D12_SRV_DIMENSION_TEXTURE2D;
    case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
        return D3D12_SRV_DIMENSION_TEXTURE3D;
    default:
        return D3D12_SRV_DIMENSION_UNKNOWN;
    }
}

inline Refusal check_default(const D3D12_RESOURCE_DESC& resource)
{
    if (resource.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) return Refusal::denied;
    if (default_dimension(resource) == D3D12_SRV_DIMENSION_UNKNOWN) return Refusal::dimension;
    const auto format = resource.Format;
    const auto family = format_family(format);
    if (format == DXGI_FORMAT_UNKNOWN || is_typeless(format) || depth_format(format) ||
        family == DXGI_FORMAT_R24G8_TYPELESS || family == DXGI_FORMAT_R32G8X24_TYPELESS ||
        format == DXGI_FORMAT_NV12 || format == DXGI_FORMAT_P010 || format == DXGI_FORMAT_P016 ||
        format == DXGI_FORMAT_420_OPAQUE)
        return Refusal::format;
    return Refusal::none;
}

// Validate first, then update individual fields so valid descriptors retain their original bytes.
// The callback receives each changed field and its original and replacement values.
template<class Repair>
Refusal check(D3D12_SHADER_RESOURCE_VIEW_DESC& view, const D3D12_RESOURCE_DESC& resource,
              unsigned dropped_levels, Repair repair, bool fully_typed_casting = false)
{
    if (resource.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) return Refusal::denied;

    if (view.ViewDimension == D3D12_SRV_DIMENSION_BUFFER ||
        resource.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        if (view.ViewDimension != D3D12_SRV_DIMENSION_BUFFER ||
            resource.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
            return Refusal::dimension;
        const UINT stride = buffer_srv_element_bytes(view, false);
        if (!stride) return view.Buffer.NumElements ? Refusal::format : Refusal::buffer_range;
        const UINT64 total = resource.Width / stride;
        if (view.Buffer.FirstElement >= total || view.Buffer.NumElements > total - view.Buffer.FirstElement ||
            (view.Buffer.Flags == D3D12_BUFFER_SRV_FLAG_RAW &&
             view.Buffer.FirstElement % (D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT / 4) != 0))
            return Refusal::buffer_range;
        if (!valid_mapping(view.Shader4ComponentMapping)) {
            repair(Field::mapping, view.Shader4ComponentMapping, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
            view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        }
        return Refusal::none;
    }

    const bool tex1d = resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;
    const bool tex2d = resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    const bool tex3d = resource.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const bool single_sample = resource.SampleDesc.Count == 1;
    const bool multi_sample = resource.SampleDesc.Count > 1;
    const UINT slices = resource.DepthOrArraySize;
    bool fits = false;
    bool cube = false;
    UINT first_slice = 0;
    UINT first_mip = 0;
    UINT plane = 0;
    float min_lod = 0.0f;
    UINT* mip_count = nullptr;
    UINT* array_count = nullptr;

    const auto mips = [&](auto& desc) {
        first_mip = desc.MostDetailedMip;
        mip_count = &desc.MipLevels;
        min_lod = desc.ResourceMinLODClamp;
    };
    switch (view.ViewDimension) {
    case D3D12_SRV_DIMENSION_TEXTURE1D:
        fits = tex1d && single_sample && slices == 1;
        mips(view.Texture1D);
        break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
        fits = tex1d && single_sample;
        mips(view.Texture1DArray);
        first_slice = view.Texture1DArray.FirstArraySlice;
        array_count = &view.Texture1DArray.ArraySize;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2D:
        fits = tex2d && single_sample && slices == 1;
        mips(view.Texture2D);
        plane = view.Texture2D.PlaneSlice;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
        fits = tex2d && single_sample;
        mips(view.Texture2DArray);
        first_slice = view.Texture2DArray.FirstArraySlice;
        array_count = &view.Texture2DArray.ArraySize;
        plane = view.Texture2DArray.PlaneSlice;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMS:
        fits = tex2d && multi_sample && slices == 1;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY:
        fits = tex2d && multi_sample;
        first_slice = view.Texture2DMSArray.FirstArraySlice;
        array_count = &view.Texture2DMSArray.ArraySize;
        break;
    case D3D12_SRV_DIMENSION_TEXTURE3D:
        fits = tex3d && single_sample;
        mips(view.Texture3D);
        break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE:
        fits = tex2d && single_sample && slices == 6 && resource.Width == resource.Height;
        mips(view.TextureCube);
        break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
        fits = tex2d && single_sample && resource.Width == resource.Height;
        mips(view.TextureCubeArray);
        cube = true;
        first_slice = view.TextureCubeArray.First2DArrayFace;
        array_count = &view.TextureCubeArray.NumCubes;
        break;
    default:
        break;
    }
    if (!fits) return Refusal::dimension;
    if (!mip_count && !resource.MipLevels) return Refusal::mip_range;

    if (mip_count && (first_mip >= resource.MipLevels || *mip_count == 0))
        return Refusal::mip_range;
    if (!std::isfinite(min_lod) || min_lod < 0.0f) return Refusal::lod;
    if (dropped_levels && ((view.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D &&
                           view.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2DARRAY) ||
                          first_mip != 0 || min_lod != 0.0f))
        return Refusal::dropped_mips;

    UINT remaining_slices = 0;
    if (array_count) {
        if (first_slice >= slices || *array_count == 0) return Refusal::array_range;
        remaining_slices = (slices - first_slice) / (cube ? 6u : 1u);
        if (!remaining_slices) return Refusal::array_range;
    }

    DXGI_FORMAT format = view.Format;
    const auto family = format_family(resource.Format);
    // Planar video resources use a different format family for each plane.
    if (resource.Format == DXGI_FORMAT_NV12 || resource.Format == DXGI_FORMAT_P010 ||
        resource.Format == DXGI_FORMAT_P016) {
        if (plane > 1) return Refusal::plane;
        if (resource.Format == DXGI_FORMAT_NV12)
            format = plane ? DXGI_FORMAT_R8G8_UNORM : DXGI_FORMAT_R8_UNORM;
        else
            format = plane ? DXGI_FORMAT_R16G16_UNORM : DXGI_FORMAT_R16_UNORM;
        if (view.Format != format) return Refusal::format;
    } else {
        if (format == DXGI_FORMAT_UNKNOWN || is_typeless(format) || depth_format(format) ||
            !format_cast_fits(resource.Format, format, fully_typed_casting))
            format = resource.Format;
        if (format == DXGI_FORMAT_UNKNOWN || is_typeless(format) || depth_format(format) ||
            format == DXGI_FORMAT_420_OPAQUE)
            return Refusal::format;
        if (family == DXGI_FORMAT_R24G8_TYPELESS || family == DXGI_FORMAT_R32G8X24_TYPELESS) {
            const bool stencil = format == DXGI_FORMAT_X24_TYPELESS_G8_UINT ||
                                 format == DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
            if (plane != static_cast<UINT>(stencil)) return Refusal::plane;
        } else if (plane != 0) {
            return Refusal::plane;
        }
    }

    const auto update = [&](Field field, auto& value, auto replacement) {
        if (value == replacement) return;
        repair(field, static_cast<UINT>(value), static_cast<UINT>(replacement));
        value = replacement;
    };
    if (mip_count && *mip_count != ~0u) {
        // Match the lazy texture policy: subtract dropped top levels, keeping one level.
        const UINT count = drop_mips::view_mip_levels(*mip_count, dropped_levels);
        update(Field::mip_levels, *mip_count, (std::min)(count, UINT{resource.MipLevels} - first_mip));
    }
    if (array_count)
        update(cube ? Field::num_cubes : Field::array_size, *array_count,
               (std::min)(*array_count, remaining_slices));
    if (!valid_mapping(view.Shader4ComponentMapping))
        update(Field::mapping, view.Shader4ComponentMapping, UINT{D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING});
    update(Field::format, view.Format, format);
    return Refusal::none;
}

} // namespace x2w_startup::view_desc
