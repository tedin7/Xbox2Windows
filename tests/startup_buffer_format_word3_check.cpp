// Checks src/startup_stubs/buffer_format_word3.h without a device: the descriptor word 3 built from a title's typed
// buffer view, and the refusals for views and hardware words the translation cannot decode.
// --mutate mapping: expect the identity DST_SEL whatever the view's mapping (a negative control: must fail).
#include "startup_stubs/buffer_format_word3.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace {
int failures;

void check(const char* name, bool condition)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL %s\n", name);
}

D3D12_SHADER_RESOURCE_VIEW_DESC typed(DXGI_FORMAT format, UINT mapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = format;
    v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    v.Shader4ComponentMapping = mapping;
    v.Buffer.NumElements = 4;
    return v;
}

void refused(const char* name, const char* got, const char* expected)
{
    if (got && std::strcmp(got, expected) == 0) return;
    ++failures;
    std::printf("FAIL %s: \"%s\", expected \"%s\"\n", name, got ? got : "(accepted)", expected);
}
}  // namespace

int main(int argc, char** argv)
{
    const bool ignore_mapping = argc == 3 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "mapping") == 0;
    if (argc != 1 && !ignore_mapping) {
        std::printf("usage: startup_buffer_format_word3_check [--mutate mapping]\n");
        return 2;
    }
    constexpr std::uint32_t identity = 4 | 5 << 3 | 6 << 6 | 7 << 9;
    std::uint32_t word = 0;
    check("float4 identity", !x2w_startup::view_word3(typed(DXGI_FORMAT_R32G32B32A32_FLOAT), word) &&
                                 word == (77u << 12 | identity));
    check("R10G10B10A2 unorm identity", !x2w_startup::view_word3(typed(DXGI_FORMAT_R10G10B10A2_UNORM), word) &&
                                            word == (50u << 12 | identity));
    check("half2 identity", !x2w_startup::view_word3(typed(DXGI_FORMAT_R16G16_FLOAT), word) && word == (29u << 12 | identity));
    check("snorm2 identity", !x2w_startup::view_word3(typed(DXGI_FORMAT_R16G16_SNORM), word) && word == (24u << 12 | identity));
    check("uint8 identity", !x2w_startup::view_word3(typed(DXGI_FORMAT_R8_UINT), word) && word == (5u << 12 | identity));
    constexpr struct {
        DXGI_FORMAT format;
        std::uint32_t code;
    } identity_formats[] = {
        {DXGI_FORMAT_R8_UNORM, 1},
        {DXGI_FORMAT_R8G8_UNORM, 14},
        {DXGI_FORMAT_R32_FLOAT, 22},
        {DXGI_FORMAT_R16G16_UNORM, 23},
        {DXGI_FORMAT_R8G8B8A8_UNORM, 56},
        {DXGI_FORMAT_R32G32_FLOAT, 64},
        {DXGI_FORMAT_R16G16B16A16_FLOAT, 71},
        {DXGI_FORMAT_R32G32B32_FLOAT, 74},
    };
    for (const auto& entry : identity_formats) {
        auto view = typed(entry.format);
        view.Buffer.FirstElement = 5;
        if (x2w_startup::view_word3(view, word) || word != (entry.code << 12 | identity)) {
            ++failures;
            std::printf("FAIL identity word for DXGI %u\n", static_cast<UINT>(entry.format));
        }
    }
    // Word 3 describes the format and selectors independently of the view range.
    auto empty = typed(DXGI_FORMAT_R16G16_FLOAT);
    empty.Buffer.NumElements = 0;
    check("empty range still encodes word 3", !x2w_startup::view_word3(empty, word) &&
                                             word == (29u << 12 | identity));
    // Mapping (w, 0, x, 1): DST_SEL 7, 0, 4, 1.
    const UINT swizzle = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(3, D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0, 0,
                                                                D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1);
    const std::uint32_t swizzled = ignore_mapping ? identity : (7u | 0u << 3 | 4u << 6 | 1u << 9);
    check("float4 swizzled", !x2w_startup::view_word3(typed(DXGI_FORMAT_R32G32B32A32_FLOAT, swizzle), word) &&
                                 word == (77u << 12 | swizzled));

    // Every decoded DXGI format gives a code the translation decodes; a few others give none.
    const DXGI_FORMAT decoded[] = {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R16G16_UNORM, DXGI_FORMAT_R16G16_SNORM, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R10G10B10A2_UNORM};
    for (DXGI_FORMAT f : decoded) {
        const std::uint32_t code = x2w_startup::buffer_format_code(f);
        if (!code || !x2w_startup::formatted_buffer_supported(code)) {
            ++failures;
            std::printf("FAIL DXGI format %d: code %u\n", static_cast<int>(f), code);
        }
    }

    refused("integer 32-bit view", x2w_startup::view_word3(typed(DXGI_FORMAT_R32_UINT), word), "view format has no decoded buffer FORMAT");
    check("refused view leaves word 0", word == 0);
    refused("16-bit UNORM 4-channel view", x2w_startup::view_word3(typed(DXGI_FORMAT_R16G16B16A16_UNORM), word),
            "view format has no decoded buffer FORMAT");
    refused("sRGB view", x2w_startup::view_word3(typed(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB), word), "view format has no decoded buffer FORMAT");
    auto raw = typed(DXGI_FORMAT_R32_TYPELESS);
    raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    refused("raw view", x2w_startup::view_word3(raw, word), "not a typed buffer view");
    auto structured = typed(DXGI_FORMAT_UNKNOWN);
    structured.Buffer.StructureByteStride = 16;
    refused("structured view", x2w_startup::view_word3(structured, word), "not a typed buffer view");
    refused("texture view", x2w_startup::view_word3(D3D12_SHADER_RESOURCE_VIEW_DESC{}, word), "not a typed buffer view");
    refused("selector 6", x2w_startup::view_word3(typed(DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(6, 1, 2, 3)), word),
            "invalid component mapping");
    const UINT invalid_last = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(1, 4, 0, 7);
    refused("selector 7 in fourth component",
            x2w_startup::view_word3(typed(DXGI_FORMAT_R16G16_FLOAT, invalid_last), word),
            "invalid component mapping");
    refused("bit 12 missing", x2w_startup::view_word3(typed(DXGI_FORMAT_R32G32B32A32_FLOAT, 0x688), word), "invalid component mapping");

    check("hardware word decoded", !x2w_startup::descriptor_word3_refusal(77u << 12 | identity));
    check("hardware constants decoded", !x2w_startup::descriptor_word3_refusal(56u << 12 | 1u | 0u << 3 | 4u << 6 | 7u << 9));
    check("hardware format codes", !x2w_startup::formatted_buffer_supported(0) && !x2w_startup::formatted_buffer_supported(2) &&
                                       x2w_startup::formatted_buffer_supported(77));
    refused("hardware DST_SEL 3", x2w_startup::descriptor_word3_refusal(77u << 12 | 3u | 5u << 3 | 6u << 6 | 7u << 9),
            "descriptor DST_SEL reserved value");

    static constexpr struct {
        DXGI_FORMAT format;
        std::uint32_t code;
    } added[] = {
        {DXGI_FORMAT_R8_SNORM, 2},
        {DXGI_FORMAT_R8_SINT, 6},
        {DXGI_FORMAT_R16_UNORM, 7},
        {DXGI_FORMAT_R16_SNORM, 8},
        {DXGI_FORMAT_R16_UINT, 11},
        {DXGI_FORMAT_R16_SINT, 12},
        {DXGI_FORMAT_R16_FLOAT, 13},
        {DXGI_FORMAT_R8G8_SNORM, 15},
        {DXGI_FORMAT_R8G8_UINT, 18},
        {DXGI_FORMAT_R8G8_SINT, 19},
        {DXGI_FORMAT_R32_UINT, 20},
        {DXGI_FORMAT_R32_SINT, 21},
        {DXGI_FORMAT_R16G16_SINT, 28},
        {DXGI_FORMAT_R11G11B10_FLOAT, 36},
        {DXGI_FORMAT_R10G10B10A2_UINT, 54},
        {DXGI_FORMAT_R8G8B8A8_SNORM, 57},
        {DXGI_FORMAT_R8G8B8A8_SINT, 61},
        {DXGI_FORMAT_R32G32_UINT, 62},
        {DXGI_FORMAT_R32G32_SINT, 63},
        {DXGI_FORMAT_R16G16B16A16_UNORM, 65},
        {DXGI_FORMAT_R16G16B16A16_SNORM, 66},
        {DXGI_FORMAT_R16G16B16A16_UINT, 69},
        {DXGI_FORMAT_R16G16B16A16_SINT, 70},
        {DXGI_FORMAT_R32G32B32A32_UINT, 75},
        {DXGI_FORMAT_R32G32B32A32_SINT, 76},
    };
    for (const auto& entry : added) {
        check("extended FORMAT disabled by default",
              x2w_startup::buffer_format_code(entry.format) == 0 &&
              !x2w_startup::formatted_buffer_supported(entry.code));
        check("extended hardware FORMAT", x2w_startup::formatted_buffer_supported(entry.code, true));
        check("extended view identity",
              !x2w_startup::view_word3(typed(entry.format), word, true) &&
              word == (entry.code << 12 | identity));
        static constexpr UINT mapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(1, 4, 0, 5);
        const std::uint32_t selects = ignore_mapping ? identity : (5 | 4 << 6 | 1 << 9);
        if (x2w_startup::view_word3(typed(entry.format, mapping), word, true) ||
            word != (entry.code << 12 | selects)) {
            std::printf("FAIL extended view mapping for DXGI %u\n", static_cast<UINT>(entry.format));
            ++failures;
        }
    }
    for (const auto format : {DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                              DXGI_FORMAT_R16_TYPELESS}) {
        refused("unsupported extended view", x2w_startup::view_word3(typed(format), word, true),
                x2w_startup::undecoded_view_format);
        check("unsupported extended view leaves word 0", word == 0);
    }
    refused("extended view invalid mapping", x2w_startup::view_word3(typed(DXGI_FORMAT_R8_SINT, 0), word, true),
            "invalid component mapping");

    if (failures) return 1;
    std::printf("startup_buffer_format_word3: all checks passed\n");
    return 0;
}
