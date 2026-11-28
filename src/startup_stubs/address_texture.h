#pragma once

// Textures a title publishes by address (device slot 70): a resource description, a view and the
// address of the texels, with no resource object behind them. This header decides, without a
// device, which of those descriptions the startup renderer reads from the address: a sampled-only,
// two-dimensional, single-sample texture whose Layout field holds a console swizzle mode that
// gfx10_layout.h addresses. Its texels are in that console layout.
// The console's linear mode (0x100) is one of those layouts: rows with addrlib's 256-byte pitch. In
// that mode a one-dimensional texture that is not an array is taken too.
// Desktop-footprint arrays (device-owned layout, two or more slices) are read as well when the caller opts in.
// Everything else is refused here: the desktop's unknown layout, other dimensions, multisampled
// textures, textures that are also targets, formats without a known element size and swizzle modes
// without an equation.

#include "startup_stubs/gfx10_layout.h"

namespace x2w_startup {
inline bool address_depth_format(DXGI_FORMAT format)
{
    return (format >= DXGI_FORMAT_R32G8X24_TYPELESS && format <= DXGI_FORMAT_X32_TYPELESS_G8X24_UINT) ||
           (format >= DXGI_FORMAT_R24G8_TYPELESS && format <= DXGI_FORMAT_X24_TYPELESS_G8_UINT) ||
           format == DXGI_FORMAT_D32_FLOAT || format == DXGI_FORMAT_D16_UNORM;
}

inline bool address_texture_flags_supported(const D3D12_RESOURCE_DESC& r)
{
    // Each flag is cast on its own: with MinGW-w64 headers the flag enum's operator| is not constexpr.
    constexpr UINT incompatible_flags = static_cast<UINT>(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) |
                                       static_cast<UINT>(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) |
                                       static_cast<UINT>(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) |
                                       static_cast<UINT>(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) |
                                       static_cast<UINT>(D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER);
    return !(static_cast<UINT>(r.Flags) & incompatible_flags);
}

inline bool console_address_texture(const D3D12_RESOURCE_DESC& r)
{
    if (!gfx10::supports(r) || r.SampleDesc.Quality || !address_texture_flags_supported(r) ||
        r.DepthOrArraySize > D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION ||
        // The draw uploads a one-dimensional texture only when it is not an array (d3d12_draw.inl).
        (r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && r.DepthOrArraySize != 1) ||
        // The largest side, depth included for a volume, must allow the mip count.
        !((std::max)({r.Width, UINT64{r.Height},
                      UINT64{r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? r.DepthOrArraySize : 1u}}) >> (r.MipLevels - 1)) ||
        (r.Alignment && r.Alignment != D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT &&
         r.Alignment != D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) ||
        address_depth_format(r.Format))
        return false;

    // A desktop block-compressed texture needs whole 4x4 blocks in its largest mip.
    const UINT texels = gfx10::element_of(r.Format).texels;
    return r.Width % texels == 0 && r.Height % texels == 0;
}

// A colour format's fully typeless parent, when the format has one.
inline DXGI_FORMAT address_format_family(DXGI_FORMAT format)
{
    const UINT f = static_cast<UINT>(format);
    constexpr UINT first[] = {1, 5, 9, 15, 23, 27, 33, 39, 48, 53, 60, 70, 73, 76, 79, 82, 94, 97};
    constexpr UINT last[] = {4, 8, 14, 18, 25, 32, 38, 43, 52, 59, 64, 72, 75, 78, 81, 84, 96, 99};
    for (UINT i = 0; i < sizeof(first) / sizeof(first[0]); ++i)
        if (f >= first[i] && f <= last[i]) return static_cast<DXGI_FORMAT>(first[i]);
    if (f == 87 || f == 90 || f == 91) return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    if (f == 88 || f == 92 || f == 93) return DXGI_FORMAT_B8G8R8X8_TYPELESS;
    return DXGI_FORMAT_UNKNOWN;
}

inline bool address_view_format(DXGI_FORMAT storage, DXGI_FORMAT view)
{
    if (!gfx10::element_of(view).bytes || address_depth_format(view)) return false;
    const auto parent = address_format_family(view);
    return view != parent && (view == storage || (parent != DXGI_FORMAT_UNKNOWN && storage == parent));
}

inline bool address_component_mapping(UINT mapping)
{
    constexpr UINT required = D3D12_SHADER_COMPONENT_MAPPING_ALWAYS_SET_BIT_AVOIDING_ZEROMEM_MISTAKES;
    if (mapping < required || mapping >= required * 2) return false;
    for (UINT channel = 0; channel < 4; ++channel)
        if (D3D12_DECODE_SHADER_4_COMPONENT_MAPPING(channel, mapping) > D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1)
            return false;
    return true;
}

// Convert a mip window relative to an alias into one relative to its desktop backing.
template<class MipView>
inline bool remap_alias_mips(MipView& view, UINT alias_mips, UINT offset, UINT backing_mips)
{
    if (view.MostDetailedMip >= alias_mips || offset >= backing_mips ||
        view.MostDetailedMip >= backing_mips - offset)
        return false;
    const UINT first = view.MostDetailedMip + offset;
    const UINT count = view.MipLevels == ~0u ? alias_mips - view.MostDetailedMip : view.MipLevels;
    if (!count || count > alias_mips - view.MostDetailedMip || count > backing_mips - first)
        return false;
    view.MostDetailedMip = first;
    view.MipLevels = count;
    return true;
}

inline bool desktop_address_array(const D3D12_RESOURCE_DESC& r)
{
    const auto element = gfx10::element_of(r.Format);
    return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && r.Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN &&
           r.DepthOrArraySize >= 2 && r.DepthOrArraySize <= D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION &&
           r.Width && r.Width <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
           r.Height && r.Height <= D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION &&
           r.MipLevels && r.MipLevels <= gfx10::max_mips &&
           ((std::max)(r.Width, UINT64{r.Height}) >> (r.MipLevels - 1)) &&
           r.SampleDesc.Count == 1 && !r.SampleDesc.Quality &&
           (!r.Alignment || r.Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) &&
           address_texture_flags_supported(r) && element.bytes && element.texels == 1 &&
           address_view_format(r.Format, r.Format);
}

// A one-dimensional, two-dimensional, array or cube view that stays inside the texture and reads its texels
// as stored.
inline bool address_view_fits(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v,
                              bool allow_cubes = false)
{
    if (!address_view_format(r.Format, v.Format) || !address_component_mapping(v.Shader4ComponentMapping))
        return false;
    const auto inside = [](UINT first, UINT count, UINT total) {
        return first < total && count && count <= total - first;
    };
    const auto mips_fit = [&](UINT first, UINT count) {
        return count == ~0u ? first < r.MipLevels : inside(first, count, r.MipLevels);
    };
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE1D)
        return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && r.DepthOrArraySize == 1 &&
               mips_fit(v.Texture1D.MostDetailedMip, v.Texture1D.MipLevels) && v.Texture1D.ResourceMinLODClamp == 0.0f;
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D)
        return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && r.DepthOrArraySize == 1 &&
               mips_fit(v.Texture2D.MostDetailedMip, v.Texture2D.MipLevels) &&
               !v.Texture2D.PlaneSlice && v.Texture2D.ResourceMinLODClamp == 0.0f;
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY)
        return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
               mips_fit(v.Texture2DArray.MostDetailedMip, v.Texture2DArray.MipLevels) &&
               inside(v.Texture2DArray.FirstArraySlice, v.Texture2DArray.ArraySize, r.DepthOrArraySize) &&
               !v.Texture2DArray.PlaneSlice && v.Texture2DArray.ResourceMinLODClamp == 0.0f;
    // A cube view reads the six slices of a two-dimensional array of six as its faces (+X, -X, +Y, -Y, +Z, -Z).
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURECUBE && r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
        r.DepthOrArraySize == 6)
        return mips_fit(v.TextureCube.MostDetailedMip, v.TextureCube.MipLevels) &&
               v.TextureCube.ResourceMinLODClamp == 0.0f;
    if (allow_cubes && r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
        v.Format == r.Format && v.Shader4ComponentMapping == D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING) {
        if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURECUBE) {
            const auto& cube = v.TextureCube;
            return r.DepthOrArraySize >= 6 && mips_fit(cube.MostDetailedMip, cube.MipLevels) &&
                   cube.ResourceMinLODClamp == 0.0f;
        }
        if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURECUBEARRAY) {
            const auto& array = v.TextureCubeArray;
            return array.First2DArrayFace < r.DepthOrArraySize && array.NumCubes &&
                   array.NumCubes <= (r.DepthOrArraySize - array.First2DArrayFace) / 6u &&
                   mips_fit(array.MostDetailedMip, array.MipLevels) && array.ResourceMinLODClamp == 0.0f;
        }
    }
    if (v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE3D)
        return r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D &&
               mips_fit(v.Texture3D.MostDetailedMip, v.Texture3D.MipLevels) &&
               v.Texture3D.ResourceMinLODClamp == 0.0f;
    return false;
}

inline bool supports_volume_memory_srv(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v)
{
    if (r.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D) return false;
    if (!r.Width || !r.Height || !r.DepthOrArraySize || !r.MipLevels) return false;
    if (r.SampleDesc.Count != 1 || r.SampleDesc.Quality != 0) return false;
    if (r.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN) return false;
    if (r.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) return false;
    if (v.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE3D) return false;
    if (v.Format != r.Format) return false;
    if (address_depth_format(r.Format)) return false;
    const auto elem = gfx10::element_of(r.Format);
    if (!elem.bytes || (elem.texels > 1 && ((r.Width % elem.texels) != 0 || (r.Height % elem.texels) != 0)))
        return false;
    return address_view_fits(r, v);
}

// Packed texel payload bytes of a 3D volume across all mip levels (excluding D3D12 256-byte row pitch padding).
// In device contexts, address_texture_bytes evaluates SliceFootprints::size to account for D3D12 copyable footprint
// row pitch (256-byte alignment) and subresource placement alignment.
inline std::uint64_t volume_texture_bytes(const D3D12_RESOURCE_DESC& desc)
{
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D) return 0;
    const auto elem = gfx10::element_of(desc.Format);
    if (!elem.bytes || !elem.texels || !desc.Width || !desc.Height || !desc.DepthOrArraySize) return 0;
    std::uint64_t total = 0;
    std::uint64_t w = desc.Width, h = desc.Height, d = desc.DepthOrArraySize;
    const UINT mips = desc.MipLevels ? desc.MipLevels : 1;
    for (UINT i = 0; i < mips; ++i) {
        const std::uint64_t blocks_w = (w + elem.texels - 1) / elem.texels;
        const std::uint64_t blocks_h = (h + elem.texels - 1) / elem.texels;
        total += blocks_w * blocks_h * d * elem.bytes;
        w = (w > 1) ? (w >> 1) : 1;
        h = (h > 1) ? (h >> 1) : 1;
        d = (d > 1) ? (d >> 1) : 1;
    }
    return total;
}
} // namespace x2w_startup
