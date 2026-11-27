// Descriptor word 3 (FORMAT bits 12..18, DST_SEL bits 0..11) for translated buffer_load_format_* loads.
// A hardware buffer descriptor carries the word itself. For a typed buffer view the title created through the
// runtime, the console driver builds it from the view's DXGI format and component mapping; this header does the
// same for the formats the translation decodes. Pure functions, checked by tests/startup_buffer_format_word3_check.cpp.
#pragma once

#include <d3d12.h>

#include "startup_stubs/formatted_buffer.h"

#include <cstdint>

namespace x2w_startup {

// FORMAT codes of tools/startup/gfx2hlsl.py DESC_FORMAT_CODES (gfx10 unified formats, as LLVM 15 encodes the
// tbuffer format: operand) for the DXGI format whose typed read gives the same components; 0 for any other
// format. tests/startup_buffer_format_word3_table_check.py keeps both tables equal.
inline std::uint32_t buffer_format_code(DXGI_FORMAT format, bool extended = false)
{
    switch (format) {
    case DXGI_FORMAT_R8_UNORM: return 1;
    case DXGI_FORMAT_R8_UINT: return 5;
    case DXGI_FORMAT_R8G8_UNORM: return 14;
    case DXGI_FORMAT_R32_FLOAT: return 22;
    case DXGI_FORMAT_R16G16_UNORM: return 23;
    case DXGI_FORMAT_R16G16_SNORM: return 24;
    case DXGI_FORMAT_R16G16_UINT: return 27;
    case DXGI_FORMAT_R16G16_FLOAT: return 29;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return 50;
    case DXGI_FORMAT_R8G8B8A8_UNORM: return 56;
    case DXGI_FORMAT_R8G8B8A8_UINT: return 60;
    case DXGI_FORMAT_R32G32_FLOAT: return 64;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return 71;
    case DXGI_FORMAT_R32G32B32_FLOAT: return 74;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return 77;
    default: break;
    }
    if (!extended) return 0;
    switch (format) {
    case DXGI_FORMAT_R8_SNORM: return 2;
    case DXGI_FORMAT_R8_SINT: return 6;
    case DXGI_FORMAT_R16_UNORM: return 7;
    case DXGI_FORMAT_R16_SNORM: return 8;
    case DXGI_FORMAT_R16_UINT: return 11;
    case DXGI_FORMAT_R16_SINT: return 12;
    case DXGI_FORMAT_R16_FLOAT: return 13;
    case DXGI_FORMAT_R8G8_SNORM: return 15;
    case DXGI_FORMAT_R8G8_UINT: return 18;
    case DXGI_FORMAT_R8G8_SINT: return 19;
    case DXGI_FORMAT_R32_UINT: return 20;
    case DXGI_FORMAT_R32_SINT: return 21;
    case DXGI_FORMAT_R16G16_SINT: return 28;
    case DXGI_FORMAT_R11G11B10_FLOAT: return 36;
    case DXGI_FORMAT_R10G10B10A2_UINT: return 54;
    case DXGI_FORMAT_R8G8B8A8_SNORM: return 57;
    case DXGI_FORMAT_R8G8B8A8_SINT: return 61;
    case DXGI_FORMAT_R32G32_UINT: return 62;
    case DXGI_FORMAT_R32G32_SINT: return 63;
    case DXGI_FORMAT_R16G16B16A16_UNORM: return 65;
    case DXGI_FORMAT_R16G16B16A16_SNORM: return 66;
    case DXGI_FORMAT_R16G16B16A16_UINT: return 69;
    case DXGI_FORMAT_R16G16B16A16_SINT: return 70;
    case DXGI_FORMAT_R32G32B32A32_UINT: return 75;
    case DXGI_FORMAT_R32G32B32A32_SINT: return 76;
    default: return 0;
    }
}

inline constexpr char undecoded_view_format[] = "view format has no decoded buffer FORMAT";

// Word 3 for a typed buffer view: DST_SEL 4..7 for channels x..w, 0 and 1 for the forced constants. Returns
// nullptr and sets `word`, or the reason no word can be given (the draw is then refused, not read as zeros).
inline const char* view_word3(const D3D12_SHADER_RESOURCE_VIEW_DESC& view, std::uint32_t& word, bool extended = false)
{
    word = 0;
    if (view.ViewDimension != D3D12_SRV_DIMENSION_BUFFER || view.Buffer.Flags != D3D12_BUFFER_SRV_FLAG_NONE ||
        view.Buffer.StructureByteStride != 0)
        return "not a typed buffer view";
    const std::uint32_t code = buffer_format_code(view.Format, extended);
    if (!code) return undecoded_view_format;
    const UINT mapping = view.Shader4ComponentMapping;
    if ((mapping >> 12) != 1) return "invalid component mapping";
    std::uint32_t selects = 0;
    for (int c = 0; c < 4; ++c) {
        const UINT source = (mapping >> (3 * c)) & 7;
        if (source > 5) return "invalid component mapping";
        selects |= (source < 4 ? 4 + source : source - 4) << (3 * c);  // FORCE_VALUE_0 is 4, FORCE_VALUE_1 is 5
    }
    word = code << 12 | selects;
    return nullptr;
}

// A DST_SEL the title wrote in a hardware descriptor: nullptr unless one selector is reserved (2 or 3). The FORMAT is
// checked by formatted_buffer_supported() in formatted_buffer.h.
inline const char* descriptor_word3_refusal(std::uint32_t word)
{
    for (int c = 0; c < 4; ++c) {
        const std::uint32_t select = (word >> (3 * c)) & 7;
        if (select == 2 || select == 3) return "descriptor DST_SEL reserved value";
    }
    return nullptr;
}

}  // namespace x2w_startup
