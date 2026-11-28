// Pure validation for the whole-resource command-list copies: CopyBufferRegion (list slot 15),
// CopyResource (17) and ResolveSubresource (19). It reads only resource descriptions, so it runs on any host.
//
// Desktop D3D12 removes the device on a copy it rejects, so a copy whose arguments the desktop would
// refuse is named and skipped instead of forwarded. Each check returns the first refusal, in the
// order listed in CopyRefusal.
#pragma once

#include <d3d12.h>

#include <algorithm>
#include <cstdint>

namespace x2w_startup {

enum class CopyRefusal {
    none,
    unresolved,                // a null argument, or one that is not a resource
    same_resource,             // source and destination are one resource
    not_buffer,                // CopyBufferRegion on a texture
    not_texture,               // ResolveSubresource on something other than a 2D texture
    dimension_mismatch,        // CopyResource between different resource dimensions
    size_mismatch,             // different width, height, depth, mips or buffer size
    format_mismatch,           // formats that are neither equal nor members of one typeless family
    sample_mismatch,           // different sample count or quality
    bad_subresource,           // a subresource index outside the texture, or of another size
    source_not_multisampled,   // the resolve source has one sample
    destination_multisampled,  // the resolve destination has more than one sample
    bad_format,                // a resolve format that is unknown, or typeless with no typed choice
    zero_size,                 // CopyBufferRegion of no bytes
    range_overflow,            // an offset and size that wrap, or run past a buffer
    heap_direction,            // an upload buffer used as a destination, or a readback buffer as a source
    initial_state,             // a placed buffer whose first state is not COMMON, which the state tracker starts from
    title_backed_source,       // a source whose title memory has no desktop copy the list could refresh
};

inline const char* copy_refusal_name(CopyRefusal refusal)
{
    switch (refusal) {
    case CopyRefusal::unresolved: return "unresolved";
    case CopyRefusal::same_resource: return "same_resource";
    case CopyRefusal::not_buffer: return "not_buffer";
    case CopyRefusal::not_texture: return "not_texture";
    case CopyRefusal::dimension_mismatch: return "dimension_mismatch";
    case CopyRefusal::size_mismatch: return "size_mismatch";
    case CopyRefusal::format_mismatch: return "format_mismatch";
    case CopyRefusal::sample_mismatch: return "sample_mismatch";
    case CopyRefusal::bad_subresource: return "bad_subresource";
    case CopyRefusal::source_not_multisampled: return "source_not_multisampled";
    case CopyRefusal::destination_multisampled: return "destination_multisampled";
    case CopyRefusal::bad_format: return "bad_format";
    case CopyRefusal::zero_size: return "zero_size";
    case CopyRefusal::range_overflow: return "range_overflow";
    case CopyRefusal::heap_direction: return "heap_direction";
    case CopyRefusal::initial_state: return "initial_state";
    case CopyRefusal::title_backed_source: return "title_backed_source";
    default: return "none";
    }
}

// The typeless member of the DXGI format family `format` belongs to; `format` itself when it has none.
// Formats of one family may be copied to each other (CopyResource) and cast by a view.
inline DXGI_FORMAT format_family(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT:
        return DXGI_FORMAT_R32G32B32A32_TYPELESS;
    case DXGI_FORMAT_R32G32B32_TYPELESS: case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32_UINT: case DXGI_FORMAT_R32G32B32_SINT:
        return DXGI_FORMAT_R32G32B32_TYPELESS;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM: case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SNORM: case DXGI_FORMAT_R16G16B16A16_SINT:
        return DXGI_FORMAT_R16G16B16A16_TYPELESS;
    case DXGI_FORMAT_R32G32_TYPELESS: case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_SINT:
        return DXGI_FORMAT_R32G32_TYPELESS;
    case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
        return DXGI_FORMAT_R32G8X24_TYPELESS;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_UINT:
        return DXGI_FORMAT_R10G10B10A2_TYPELESS;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UINT: case DXGI_FORMAT_R8G8B8A8_SNORM: case DXGI_FORMAT_R8G8B8A8_SINT:
        return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    case DXGI_FORMAT_B8G8R8X8_TYPELESS: case DXGI_FORMAT_B8G8R8X8_UNORM: case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8X8_TYPELESS;
    case DXGI_FORMAT_R16G16_TYPELESS: case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_UINT: case DXGI_FORMAT_R16G16_SNORM: case DXGI_FORMAT_R16G16_SINT:
        return DXGI_FORMAT_R16G16_TYPELESS;
    case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT:
        return DXGI_FORMAT_R32_TYPELESS;
    case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
        return DXGI_FORMAT_R24G8_TYPELESS;
    case DXGI_FORMAT_R8G8_TYPELESS: case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SNORM: case DXGI_FORMAT_R8G8_SINT:
        return DXGI_FORMAT_R8G8_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R16_SNORM: case DXGI_FORMAT_R16_SINT:
        return DXGI_FORMAT_R16_TYPELESS;
    case DXGI_FORMAT_R8_TYPELESS: case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SNORM: case DXGI_FORMAT_R8_SINT:
        return DXGI_FORMAT_R8_TYPELESS;
    case DXGI_FORMAT_BC1_TYPELESS: case DXGI_FORMAT_BC1_UNORM: case DXGI_FORMAT_BC1_UNORM_SRGB:
        return DXGI_FORMAT_BC1_TYPELESS;
    case DXGI_FORMAT_BC2_TYPELESS: case DXGI_FORMAT_BC2_UNORM: case DXGI_FORMAT_BC2_UNORM_SRGB:
        return DXGI_FORMAT_BC2_TYPELESS;
    case DXGI_FORMAT_BC3_TYPELESS: case DXGI_FORMAT_BC3_UNORM: case DXGI_FORMAT_BC3_UNORM_SRGB:
        return DXGI_FORMAT_BC3_TYPELESS;
    case DXGI_FORMAT_BC4_TYPELESS: case DXGI_FORMAT_BC4_UNORM: case DXGI_FORMAT_BC4_SNORM:
        return DXGI_FORMAT_BC4_TYPELESS;
    case DXGI_FORMAT_BC5_TYPELESS: case DXGI_FORMAT_BC5_UNORM: case DXGI_FORMAT_BC5_SNORM:
        return DXGI_FORMAT_BC5_TYPELESS;
    case DXGI_FORMAT_BC6H_TYPELESS: case DXGI_FORMAT_BC6H_UF16: case DXGI_FORMAT_BC6H_SF16:
        return DXGI_FORMAT_BC6H_TYPELESS;
    case DXGI_FORMAT_BC7_TYPELESS: case DXGI_FORMAT_BC7_UNORM: case DXGI_FORMAT_BC7_UNORM_SRGB:
        return DXGI_FORMAT_BC7_TYPELESS;
    default:
        return format;
    }
}

// The typed format a typeless one resolves as; DXGI_FORMAT_UNKNOWN for a format that is not typeless or has no
// single natural choice. Floating point for the HDR families, normalised for the 8 and 10 bit ones.
inline DXGI_FORMAT typed_choice(DXGI_FORMAT typeless)
{
    switch (typeless) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

inline bool is_planar_depth_stencil(DXGI_FORMAT format)
{
    const auto family = format_family(format);
    return family == DXGI_FORMAT_R24G8_TYPELESS || family == DXGI_FORMAT_R32G8X24_TYPELESS;
}

inline bool is_typeless(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: case DXGI_FORMAT_R32G32B32_TYPELESS: case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R32G32_TYPELESS: case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R16G16_TYPELESS: case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_R8G8_TYPELESS: case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_BC1_TYPELESS: case DXGI_FORMAT_BC2_TYPELESS: case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC4_TYPELESS: case DXGI_FORMAT_BC5_TYPELESS: case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC7_TYPELESS:
        return true;
    default:
        return false;
    }
}

inline bool is_typeless_choice_format(DXGI_FORMAT format) { return typed_choice(format) != DXGI_FORMAT_UNKNOWN; }

inline bool compatible_formats(DXGI_FORMAT a, DXGI_FORMAT b)
{
    return a == b || (a != DXGI_FORMAT_UNKNOWN && format_family(a) == format_family(b));
}

// CopyBufferRegion(destination, destination_offset, source, source_offset, bytes). `same` is true when both
// arguments are one resource (it would need two states at once).
inline CopyRefusal check_copy_buffer_region(const D3D12_RESOURCE_DESC* destination, std::uint64_t destination_offset,
                                            const D3D12_RESOURCE_DESC* source, std::uint64_t source_offset,
                                            std::uint64_t bytes, bool same)
{
    if (!destination || !source) return CopyRefusal::unresolved;
    if (same) return CopyRefusal::same_resource;
    if (destination->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || source->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
        return CopyRefusal::not_buffer;
    if (!bytes) return CopyRefusal::zero_size;
    if (destination_offset > destination->Width || bytes > destination->Width - destination_offset ||
        source_offset > source->Width || bytes > source->Width - source_offset)
        return CopyRefusal::range_overflow;
    return CopyRefusal::none;
}

// CopyResource(destination, source): the two descriptions must have one shape.
inline CopyRefusal check_copy_resource(const D3D12_RESOURCE_DESC* destination, const D3D12_RESOURCE_DESC* source, bool same)
{
    if (!destination || !source) return CopyRefusal::unresolved;
    if (same) return CopyRefusal::same_resource;
    if (destination->Dimension == D3D12_RESOURCE_DIMENSION_UNKNOWN || destination->Dimension != source->Dimension)
        return CopyRefusal::dimension_mismatch;
    if (destination->Width != source->Width) return CopyRefusal::size_mismatch;
    if (destination->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return CopyRefusal::none;
    if (destination->Height != source->Height || destination->DepthOrArraySize != source->DepthOrArraySize ||
        destination->MipLevels != source->MipLevels)
        return CopyRefusal::size_mismatch;
    if (!compatible_formats(destination->Format, source->Format)) return CopyRefusal::format_mismatch;
    if (destination->SampleDesc.Count != source->SampleDesc.Count || destination->SampleDesc.Quality != source->SampleDesc.Quality)
        return CopyRefusal::sample_mismatch;
    return CopyRefusal::none;
}

struct ResolvePlan {
    CopyRefusal refusal{};
    DXGI_FORMAT format{};  // the typed format to resolve as, when not refused
};

// One mip of a 2D texture: its size in texels.
inline std::uint64_t mip_extent(std::uint64_t extent, std::uint32_t mip) { return (std::max<std::uint64_t>)(extent >> mip, 1); }

// ResolveSubresource(destination, destination_subresource, source, source_subresource, format). A typeless
// `format` (or one the title left unknown over a typeless surface) resolves as the family's typed choice, so a
// half-float surface stored as typeless resolves as half float.
inline ResolvePlan plan_resolve(const D3D12_RESOURCE_DESC* destination, std::uint32_t destination_subresource,
                                const D3D12_RESOURCE_DESC* source, std::uint32_t source_subresource, DXGI_FORMAT format,
                                bool same)
{
    if (!destination || !source) return {CopyRefusal::unresolved, {}};
    if (same) return {CopyRefusal::same_resource, {}};
    if (destination->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || source->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        return {CopyRefusal::not_texture, {}};
    // The legacy resolve path has no depth/stencil resolve mode.
    if (is_planar_depth_stencil(destination->Format) || is_planar_depth_stencil(source->Format))
        return {CopyRefusal::bad_format, {}};
    if (source->SampleDesc.Count <= 1) return {CopyRefusal::source_not_multisampled, {}};
    if (destination->SampleDesc.Count != 1) return {CopyRefusal::destination_multisampled, {}};
    const std::uint32_t destination_mips = destination->MipLevels ? destination->MipLevels : 1;
    const std::uint32_t source_mips = source->MipLevels ? source->MipLevels : 1;
    if (destination_subresource >= std::uint64_t{destination_mips} * destination->DepthOrArraySize ||
        source_subresource >= std::uint64_t{source_mips} * source->DepthOrArraySize)
        return {CopyRefusal::bad_subresource, {}};
    const std::uint32_t destination_mip = destination_subresource % destination_mips;
    const std::uint32_t source_mip = source_subresource % source_mips;
    if (mip_extent(destination->Width, destination_mip) != mip_extent(source->Width, source_mip) ||
        mip_extent(destination->Height, destination_mip) != mip_extent(source->Height, source_mip))
        return {CopyRefusal::bad_subresource, {}};
    // An endpoint of a fully typed format fixes the resolve format exactly; family compatibility only
    // satisfies a typeless endpoint. Pick the format from a typed endpoint before any typeless fallback.
    const bool source_typed = source->Format != DXGI_FORMAT_UNKNOWN && !is_typeless(source->Format);
    const bool destination_typed = destination->Format != DXGI_FORMAT_UNKNOWN && !is_typeless(destination->Format);
    DXGI_FORMAT typed = format;
    if (typed == DXGI_FORMAT_UNKNOWN || is_typeless(typed)) {
        if (source_typed) typed = source->Format;
        else if (destination_typed) typed = destination->Format;
        else if (typed == DXGI_FORMAT_UNKNOWN) typed = source->Format;
        if (is_typeless_choice_format(typed)) typed = typed_choice(typed);
    }
    if (typed == DXGI_FORMAT_UNKNOWN || is_typeless(typed)) return {CopyRefusal::bad_format, {}};
    const auto fits = [typed](const D3D12_RESOURCE_DESC* endpoint, bool is_typed) {
        return is_typed ? endpoint->Format == typed : compatible_formats(typed, endpoint->Format);
    };
    if (!fits(source, source_typed) || !fits(destination, destination_typed)) return {CopyRefusal::format_mismatch, {}};
    return {CopyRefusal::none, typed};
}

}  // namespace x2w_startup
