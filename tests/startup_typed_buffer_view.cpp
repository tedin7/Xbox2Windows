#include "../src/startup_stubs/typed_buffer_view.h"

#include <cstdio>

int main()
{
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Buffer.FirstElement = 5;
    view.Buffer.NumElements = 2;

    const auto verify = [](const D3D12_SHADER_RESOURCE_VIEW_DESC& desc,
                           std::optional<std::uint32_t> expected, const char* name) {
        const auto actual = x2w_startup::typed_buffer_word3(desc);
        if (actual == expected) return true;
        std::fprintf(stderr, "%s: expected %08x, got %08x\n", name,
            expected.value_or(0xffffffffu), actual.value_or(0xffffffffu));
        return false;
    };

    constexpr struct { DXGI_FORMAT dxgi; std::uint32_t encoded; } formats[] = {
        {DXGI_FORMAT_R8_UNORM, 0x01fac}, {DXGI_FORMAT_R8_UINT, 0x05fac},
        {DXGI_FORMAT_R8G8_UNORM, 0x0efac}, {DXGI_FORMAT_R32_UINT, 0x14fac},
        {DXGI_FORMAT_R32_FLOAT, 0x16fac}, {DXGI_FORMAT_R16G16_UNORM, 0x17fac},
        {DXGI_FORMAT_R16G16_SNORM, 0x18fac}, {DXGI_FORMAT_R16G16_FLOAT, 0x1dfac},
        {DXGI_FORMAT_R8G8B8A8_UNORM, 0x38fac}, {DXGI_FORMAT_R32G32_FLOAT, 0x40fac},
        {DXGI_FORMAT_R16G16B16A16_FLOAT, 0x47fac}, {DXGI_FORMAT_R32G32B32_FLOAT, 0x4afac},
        {DXGI_FORMAT_R32G32B32A32_FLOAT, 0x4dfac},
    };
    bool ok = true;
    for (const auto& format : formats) {
        view.Format = format.dxgi;
        ok = verify(view, format.encoded, "identity FORMAT mapping") && ok;
    }
    view.Format = DXGI_FORMAT_R16G16_FLOAT;
    view.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(1, 4, 0, 5);
    ok = verify(view, 0x1d305, "R16G16_FLOAT swizzle and constants") && ok;

    auto bad = view;
    bad.Format = DXGI_FORMAT_UNKNOWN;
    bad.Buffer.StructureByteStride = 8;
    ok = verify(bad, std::nullopt, "structured eight-byte record") && ok;
    bad = view;
    bad.Format = DXGI_FORMAT_R32_TYPELESS;
    bad.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    ok = verify(bad, std::nullopt, "raw view") && ok;
    bad = view;
    bad.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    ok = verify(bad, std::nullopt, "texture view") && ok;
    bad = view;
    bad.Format = DXGI_FORMAT_R16G16_UINT;
    ok = verify(bad, std::nullopt, "unsupported typed view") && ok;
    bad = view;
    bad.Buffer.NumElements = 0;
    ok = verify(bad, std::nullopt, "empty view") && ok;
    bad = view;
    bad.Shader4ComponentMapping = 0;
    ok = verify(bad, std::nullopt, "mapping sentinel missing") && ok;
    bad = view;
    bad.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(1, 4, 0, 7);
    ok = verify(bad, std::nullopt, "invalid last selector") && ok;
    if (ok) std::puts("typed buffer FORMAT and DST_SEL verified (13 formats, 1 swizzle, 7 refusals)");
    return ok ? 0 : 1;
}
