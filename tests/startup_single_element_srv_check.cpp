// x2w_startup::single_element_bytes and supports_single_element_srv (src/startup_stubs/single_element_srv.h)
// without a device. Accepted: 1x1 2D and 1x1x1 3D textures with one mip and one sample, layout UNKNOWN, linear
// or a defined GFX10 swizzle mode without XOR, default or 64 KiB alignment, a plain colour format and a view naming
// the whole texture. Each refused case
// differs from an accepted one in a single field.
// --mutate xor: expect the XOR layout 0x115 to be accepted (a negative control).
#include "startup_stubs/single_element_srv.h"

#include <cstdio>
#include <cstring>

namespace {
D3D12_RESOURCE_DESC texel(D3D12_RESOURCE_DIMENSION dimension, DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM, UINT layout = 0)
{
    return {dimension, 0, 1, 1, 1, 1, format, {1, 0}, static_cast<D3D12_TEXTURE_LAYOUT>(layout),
            static_cast<D3D12_RESOURCE_FLAGS>(0x300000)};
}

D3D12_SHADER_RESOURCE_VIEW_DESC view_of(const D3D12_RESOURCE_DESC& r)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = r.Format;
    v.ViewDimension = r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? D3D12_SRV_DIMENSION_TEXTURE3D : D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Texture2D.MipLevels = 1;
    return v;
}

struct Case {
    const char* name;
    D3D12_RESOURCE_DESC desc;
    D3D12_SHADER_RESOURCE_VIEW_DESC view;
    UINT bytes;  // expected single_element_bytes
    bool accepted;
};

template<typename Edit> Case edited(const char* name, D3D12_RESOURCE_DIMENSION dimension, UINT bytes, bool accepted, Edit edit)
{
    Case c{name, texel(dimension), {}, bytes, accepted};
    c.view = view_of(c.desc);
    edit(c.desc, c.view);
    return c;
}
}  // namespace

int main(int argc, char** argv)
{
    const bool xor_mutation = argc == 3 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "xor");
    if (argc != 1 && !xor_mutation) {
        std::printf("usage: startup_single_element_srv_check [--mutate xor]\n");
        return 2;
    }
    constexpr auto d2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D, d3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const auto none = [](auto&, auto&) {};
    const Case cases[] = {
        edited("2D RGBA8 layout UNKNOWN", d2, 4, true, none),
        edited("3D RGBA8 layout UNKNOWN", d3, 4, true, none),
        edited("3D RGBA8 linear 0x100", d3, 4, true, [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x100); }),
        edited("2D RGBA8 4 KiB S 0x105", d2, 4, true, [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x105); }),
        edited("3D RGBA8 64 KiB D 0x10a", d3, 4, true, [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x10a); }),
        edited("2D RGBA8 256 B S 0x101", d2, 4, true, [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101); }),
        edited("3D 64 KiB alignment", d3, 4, true, [](auto& d, auto&) { d.Alignment = 0x10000; }),
        edited("3D R16G16B16A16_FLOAT", d3, 8, true, [](auto& d, auto& v) { d.Format = v.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; }),
        edited("2D R32G32B32A32_FLOAT", d2, 16, true, [](auto& d, auto& v) { d.Format = v.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; }),
        edited("2D R16_FLOAT", d2, 2, true, [](auto& d, auto& v) { d.Format = v.Format = DXGI_FORMAT_R16_FLOAT; }),
        edited("2D A8_UNORM", d2, 1, true, [](auto& d, auto& v) { d.Format = v.Format = DXGI_FORMAT_A8_UNORM; }),
        edited("2D zero flags", d2, 4, true, [](auto& d, auto&) { d.Flags = D3D12_RESOURCE_FLAG_NONE; }),
        // Shape.
        edited("2x1", d2, 0, false, [](auto& d, auto&) { d.Width = 2; }),
        edited("1x2", d3, 0, false, [](auto& d, auto&) { d.Height = 2; }),
        edited("1x1x2 volume", d3, 0, false, [](auto& d, auto&) { d.DepthOrArraySize = 2; }),
        edited("two array slices", d2, 0, false, [](auto& d, auto&) { d.DepthOrArraySize = 2; }),
        edited("two mips", d2, 0, false, [](auto& d, auto&) { d.MipLevels = 2; }),
        edited("zero mips", d3, 0, false, [](auto& d, auto&) { d.MipLevels = 0; }),
        edited("four samples", d2, 0, false, [](auto& d, auto&) { d.SampleDesc.Count = 4; }),
        edited("sample quality 1", d2, 0, false, [](auto& d, auto&) { d.SampleDesc.Quality = 1; }),
        edited("1D", d2, 0, false, [](auto& d, auto&) { d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D; }),
        // Layout: XOR modes and desktop layouts.
        edited("4 KiB S with XOR 0x115", d2, xor_mutation ? 4 : 0, xor_mutation,
               [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x115); }),
        edited("64 KiB Z T 0x110", d3, 0, false, [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x110); }),
        edited("reserved mode 0x10c", d3, 0, false, [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x10c); }),
        edited("reserved mode 0x103", d2, 0, false, [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x103); }),
        edited("reserved mode 0x104", d3, 0, false, [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x104); }),
        edited("reserved mode 0x107", d2, 0, false, [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x107); }),
        edited("reserved mode 0x108", d3, 0, false, [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x108); }),
        edited("reserved mode 0x10b", d3, 0, false, [](auto& d, auto&) { d.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x10b); }),
        edited("4 KiB alignment", d2, 0, false, [](auto& d, auto&) { d.Alignment = 0x1000; }),
        edited("4 MiB alignment", d3, 0, false, [](auto& d, auto&) { d.Alignment = 0x400000; }),
        edited("ROW_MAJOR", d2, 0, false, [](auto& d, auto&) { d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; }),
        // Flags and formats.
        edited("render-target flag", d2, 0, false, [](auto& d, auto&) { d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET; }),
        edited("UAV flag", d3, 0, false, [](auto& d, auto&) { d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; }),
        edited("deny shader resource", d2, 0, false, [](auto& d, auto&) {
            d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE; }),
        edited("R8_UNORM (placed path)", d2, 0, false, [](auto& d, auto& v) { d.Format = v.Format = DXGI_FORMAT_R8_UNORM; }),
        edited("BC1_UNORM", d2, 0, false, [](auto& d, auto& v) { d.Format = v.Format = DXGI_FORMAT_BC1_UNORM; }),
        edited("R8G8B8A8_TYPELESS", d2, 0, false, [](auto& d, auto& v) { d.Format = v.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS; }),
        edited("D32_FLOAT", d2, 0, false, [](auto& d, auto& v) { d.Format = v.Format = DXGI_FORMAT_D32_FLOAT; }),
        edited("NV12", d2, 0, false, [](auto& d, auto& v) { d.Format = v.Format = DXGI_FORMAT_NV12; }),
        edited("console format 116", d2, 0, false,
               [](auto& d, auto& v) { d.Format = v.Format = static_cast<DXGI_FORMAT>(116); }),
        // The view (single_element_bytes accepts the description; the view check refuses).
        edited("view format SRGB", d2, 4, false, [](auto&, auto& v) { v.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; }),
        edited("2D view of a volume", d3, 4, false, [](auto&, auto& v) { v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; }),
        edited("3D view of a 2D texture", d2, 4, false, [](auto&, auto& v) { v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D; }),
        edited("array view", d2, 4, false, [](auto&, auto& v) { v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY; }),
        edited("two-mip view", d3, 4, false, [](auto&, auto& v) { v.Texture3D.MipLevels = 2; }),
        edited("most detailed mip 1", d2, 4, false, [](auto&, auto& v) { v.Texture2D.MostDetailedMip = 1; }),
        edited("plane slice 1", d2, 4, false, [](auto&, auto& v) { v.Texture2D.PlaneSlice = 1; }),
        edited("LOD clamp 1", d3, 4, false, [](auto&, auto& v) { v.Texture3D.ResourceMinLODClamp = 1.0f; }),
        edited("swizzled mapping", d2, 4, false, [](auto&, auto& v) {
            v.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(2, 1, 0, 3); }),
    };
    int failures = 0, accepted = 0;
    for (const Case& c : cases) {
        const UINT bytes = x2w_startup::single_element_bytes(c.desc);
        const bool actual = x2w_startup::supports_single_element_srv(c.desc, c.view);
        accepted += actual;
        if (bytes != c.bytes || actual != c.accepted) {
            std::printf("FAIL %s: bytes %u accepted %d, expected bytes %u accepted %d\n", c.name, bytes, actual, c.bytes,
                        c.accepted);
            ++failures;
        }
    }
    const int total = static_cast<int>(sizeof(cases) / sizeof(cases[0]));
    std::printf("%d cases, %d accepted, %d refused, %d mismatches\n", total, accepted, total - accepted, failures);
    return failures ? 1 : 0;
}
