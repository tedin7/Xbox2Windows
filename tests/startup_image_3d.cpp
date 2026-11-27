// Desktop readback for tools/startup/gfx2hlsl.py's 3D image sampling (dim:SQ_RSRC_IMG_3D, #1149).
//
// The translated pixel shader samples r0 : register(t0), a Texture3D<float4>, through q0 :
// register(s0) at (0.75, 0.25, z), with z taken from an entry constant (cbuffer entry : register(b0),
// ud1). This draws one pixel with that shader and a source-built full-screen vertex shader, then
// reads the colour back.
//
// The texture is 2x2x2 R32G32B32A32_FLOAT. Texel (x, y, z) holds ((1 + x + 2y + 4z) / 8, x, y, z),
// so the readback names the texel the coordinate selected. By default it uses the all-zero inline
// sampler (point wrap). The inline sampler fixture passes its own manifest line for the same draw.
#include "desktop_gpu.h"

#include "startup_stubs/translated_sampler_bindings.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT entry_constants = 4;  // ud0..ud3 of the fixture's manifest
constexpr UINT side = 2;
// The mip fixture: a 4x4 array of two slices with three mip levels (4x4, 2x2, 1x1).
constexpr UINT mip_base = 4;
constexpr UINT max_levels = 3;

// The value at texel (x, y) of slice s and mip level m. One level keeps the 2x2x2 fixture's values. With
// several levels every (slice, mip, x, y) cell is distinct: the red channel is a dyadic fraction and
// alpha is s + 2m, so exchanging slice and mip, or reading another level, changes the colour.
void texel_value(UINT x, UINT y, UINT s, UINT m, UINT levels, float out[4])
{
    if (levels == 1) {
        const float one[4] = {(1.0f + x + 2.0f * y + 4.0f * s) / 8.0f, float(x), float(y), float(s)};
        std::memcpy(out, one, sizeof(one));
        return;
    }
    const float many[4] = {(1.0f + x + 4.0f * y + 16.0f * s + 32.0f * m) / 128.0f, float(x), float(y), float(s + 2 * m)};
    std::memcpy(out, many, sizeof(many));
}

std::vector<char> read_blob(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (!file.good() && !file.eof()) throw std::runtime_error(std::string("Cannot read ") + path);
    if (bytes.empty()) throw std::runtime_error(std::string("Empty shader ") + path);
    return bytes;
}

ComPtr<ID3D12RootSignature> create_root(ID3D12Device* device, const x2w_startup::TranslatedSamplers& samplers)
{
    ComPtr<ID3D12RootSignature> root;
    check(x2w_startup::create_translated_root(device, entry_constants, 1, samplers, &root),
          "Create translated image 3D root signature");
    return root;
}
}  // namespace

// array: the same eight texels as two slices of a 2D array (dim:SQ_RSRC_IMG_2D_ARRAY); the default
// view of a two-slice texture is a Texture2DArray over both. Multi-mip fixtures use a 4x4 base.
// A nonzero view_levels selects an explicit 2D SRV starting at first_mip. When sampler_line is a
// comparison sampler (smpc), the texture uses DXGI_FORMAT_R32_FLOAT holding the red channel as depth.
void check_startup_image_3d(ID3D12Device* device, const char* vertex_path, const char* pixel_path, float z,
                            float color[4], const char* sampler_line, bool array, UINT levels, bool one_dimensional,
                            bool two_dimensional, UINT first_mip, UINT view_levels)
{
    if (levels < 1 || levels > max_levels || (levels > 1 && !array && !two_dimensional && !one_dimensional))
        throw std::runtime_error("Unsupported image mip level count");
    if (view_levels && (!two_dimensional || first_mip >= levels || view_levels > levels - first_mip))
        throw std::runtime_error("2D mip window exceeds the resource");
    const UINT base = levels > 1 ? mip_base : side;
    const auto vs = read_blob(vertex_path);
    const auto ps = read_blob(pixel_path);
    x2w_startup::TranslatedSamplers samplers;
    if (!samplers.read_manifest_line(sampler_line) || samplers.statics.size() != 1)
        throw std::runtime_error("Expected one inline sampler in the image 3D draw");
    const bool comparison = !samplers.comparison.empty() && samplers.comparison[0];
    const auto root = create_root(device, samplers);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.VS = {vs.data(), vs.size()};
    pd.PS = {ps.data(), ps.size()};
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    pd.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Create image 3D PSO");

    D3D12_RESOURCE_DESC td{};
    td.Dimension = one_dimensional ? D3D12_RESOURCE_DIMENSION_TEXTURE1D
                 : (array || two_dimensional) ? D3D12_RESOURCE_DIMENSION_TEXTURE2D
                 : D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    td.Width = base;
    td.Height = one_dimensional ? 1 : base;
    td.DepthOrArraySize = (one_dimensional || two_dimensional) ? 1 : side;
    td.MipLevels = levels;
    td.Format = comparison ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.SampleDesc.Count = 1;
    const auto texture = resource(device, td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    // A volume is one subresource of two slices; an array is two subresources of one slice each.
    // The subresource of slice s and mip m is m + s * levels.
    const UINT subresources = (array ? side : 1) * levels;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[side * max_levels]{};
    UINT64 upload_size = 0;
    device->GetCopyableFootprints(&td, 0, subresources, 0, footprints, nullptr, nullptr, &upload_size);
    const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint = footprints[0];
    const auto upload = make_buffer(device, upload_size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    const D3D12_RANGE none{0, 0};
    check(upload->Map(0, &none, &mapped), "Map image 3D texels");
    auto* const start = static_cast<std::uint8_t*>(mapped);
    const UINT slice_pitch = footprint.Footprint.RowPitch * footprint.Footprint.Height;
    for (UINT tz = 0; tz < td.DepthOrArraySize; ++tz)
        for (UINT level = 0; level < levels; ++level) {
            const UINT extent = base >> level;
            const auto& place = footprints[level + (array ? tz * levels : 0)];
            for (UINT ty = 0; ty < (one_dimensional ? 1u : extent); ++ty)
                for (UINT tx = 0; tx < extent; ++tx) {
                    float texel[4];
                    texel_value(tx, ty, tz, level, levels, texel);
                    if (one_dimensional) texel[3] = tx ? 0.75f : 0.5f;
                    auto* const slice = array ? start + place.Offset : start + place.Offset + tz * slice_pitch;
                    const UINT texel_bytes = comparison ? sizeof(float) : sizeof(texel);
                    std::memcpy(slice + ty * place.Footprint.RowPitch + tx * texel_bytes, texel, texel_bytes);
                }
        }
    upload->Unmap(0, nullptr);

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = 1;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R32G32B32A32_FLOAT, {-1.0f, -1.0f, -1.0f, -1.0f}};
    const auto target = resource(device, rd, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT target_footprint{};
    UINT64 readback_size = 0;
    device->GetCopyableFootprints(&rd, 0, 1, 0, &target_footprint, nullptr, nullptr, &readback_size);
    const auto readback = make_buffer(device, readback_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> views, rtvs;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)), "Create image 3D view heap");
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    if (view_levels) {
        view.Format = td.Format;
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Texture2D.MostDetailedMip = first_mip;
        view.Texture2D.MipLevels = view_levels;
    }
    device->CreateShaderResourceView(texture.Get(), view_levels ? &view : nullptr, views->GetCPUDescriptorHandleForHeapStart());
    hd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvs)), "Create image 3D RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    for (UINT i = 0; i < subresources; ++i) {
        D3D12_TEXTURE_COPY_LOCATION to{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        to.SubresourceIndex = i;
        D3D12_TEXTURE_COPY_LOCATION from{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        from.PlacedFootprint = footprints[i];
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    transition(list, texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    list->SetGraphicsRootSignature(root.Get());
    ID3D12DescriptorHeap* heaps[]{views.Get()};
    list->SetDescriptorHeaps(1, heaps);
    // ud0 is the rectangle-list switch (0: plain triangles), ud1 the z coordinate's bits.
    std::uint32_t z_bits;
    std::memcpy(&z_bits, &z, sizeof(z_bits));
    const std::uint32_t constants[entry_constants] = {0, z_bits, 0, 0};
    list->SetGraphicsRoot32BitConstants(0, entry_constants, constants, 0);
    list->SetGraphicsRootDescriptorTable(1, views->GetGPUDescriptorHandleForHeapStart());
    const D3D12_VIEWPORT viewport{0, 0, 1, 1, 0, 1};
    const D3D12_RECT scissor{0, 0, 1, 1};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION out{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    out.PlacedFootprint = target_footprint;
    D3D12_TEXTURE_COPY_LOCATION in{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&out, 0, 0, 0, &in, nullptr);
    commands.submit_and_wait(device);

    const D3D12_RANGE read{0, 4 * sizeof(float)};
    check(readback->Map(0, &read, &mapped), "Map image 3D readback");
    std::memcpy(color, static_cast<std::uint8_t*>(mapped) + target_footprint.Offset, 4 * sizeof(float));
    readback->Unmap(0, &none);
    std::printf("startup_image_%s z=%g color=(%g, %g, %g, %g)\n",
                one_dimensional ? "1d" : two_dimensional ? "2d" : array ? "2d_array" : "3d",
                z, color[0], color[1], color[2], color[3]);
}

// Draws an 8x2 R32G32B32A32_UINT target against a two-mip (8x2 mip 0, 4x1 mip 1) 2D or two-slice 2D-array
// source texture with no static samplers:
//   depth16 == true:  DXGI_FORMAT_R16_UNORM (16-bit depth SRV format) covering zero, binary16 subnormals,
//                     exact fractions, double-rounding discriminators (0xffaf, 0xffef) and RNE-vs-RTZ codes;
//   depth16 == false: DXGI_FORMAT_R32G32B32A32_FLOAT covering four distinct channels per texel, halfway ties
//                     to even, 1-ULP midpoint neighbours, subnormals, signed zero, overflow and quiet NaNs.
//   half_float:      DXGI_FORMAT_R16G16B16A16_FLOAT with exactly representable finite values.
//   red_only:        DXGI_FORMAT_R32_FLOAT target. Read R into each pixel's first output word.
void check_startup_image_load_d16(ID3D12Device* device, const char* vertex_path, const char* pixel_path,
                                  bool depth16, std::uint32_t words[64], bool array, bool sampled, bool half_float, bool red_only)
{
    constexpr UINT width = 8, height = 2, d16_mips = 2;
    static constexpr std::uint16_t kDepthTexels[width * height] = {
        0x0000, 0x0001, 0x0003, 0x1234, 0x2aad, 0x4008, 0x5555, 0x8000,
        0x8010, 0xa018, 0xc018, 0xe00c, 0xffaf, 0xffef, 0xfffe, 0xffff,
    };
    static constexpr std::uint16_t kDepthMip1[2][width / 2] = {
        {0xffaf, 0xffef, 0x0003, 0x8010},
        {0xffef, 0xffaf, 0x5555, 0xa018},
    };
    static constexpr std::uint32_t kFloatTexels[width * height][4] = {
        {0x3f801000u, 0x3f803000u, 0x3f801001u, 0x3f7fffffu},
        {0x387fe000u, 0x387fdfffu, 0x33800000u, 0x33000000u},
        {0x33000001u, 0xb5c00000u, 0x80000000u, 0x00000000u},
        {0xbf803000u, 0xbf801000u, 0x477fefffu, 0x477ff000u},
        {0xc77ff000u, 0x7f800000u, 0xff800000u, 0x7fc00001u},
        {0x40490fdbu, 0x35555555u, 0xbeaaaaabu, 0xffc00123u},
        {0x3c000000u, 0xbc800000u, 0x3d000000u, 0xbd800000u},
        {0x42000000u, 0xc2200000u, 0x42400000u, 0xc2600000u},
        {0x3e200000u, 0x3ea00000u, 0x3ef00000u, 0x3f100000u},
        {0xbe200000u, 0xbea00000u, 0xbef00000u, 0xbf100000u},
        {0x39001000u, 0x39003000u, 0xb9003000u, 0x39001001u},
        {0x44002000u, 0x44006000u, 0xc4006000u, 0x44002001u},
        {0x34000000u, 0x34800000u, 0x35000000u, 0x35800000u},
        {0xb4000000u, 0xb4800000u, 0xb5000000u, 0xb5800000u},
        {0x3fc00000u, 0xc0000000u, 0x40200000u, 0xc0400000u},
        {0x477fe000u, 0xc77fe000u, 0x38800000u, 0xb8800000u},
    };
    static constexpr std::uint32_t kFloatMip1[width / 2][4] = {
        {0x3f803000u, 0x387fe000u, 0xbf803000u, 0x477fefffu},
        {0x33800000u, 0x33000001u, 0x80000000u, 0x7fc00001u},
        {0x44006000u, 0xc4006000u, 0x39003000u, 0xb9003000u},
        {0x477ff000u, 0xc77ff000u, 0x3e200000u, 0xbe200000u},
    };

    const auto vs = read_blob(vertex_path);
    const auto ps = read_blob(pixel_path);
    constexpr UINT d16_entry_constants = 5;  // ud0..ud3 from FULLSCREEN_VS plus ud4 ('d16unorm16 0')
    x2w_startup::TranslatedSamplers samplers;
    if (sampled && !samplers.read_manifest_line("smpl 0 0x0 0x100000 0x4000000 0x0"))
        throw std::runtime_error("invalid image_sample_lz d16 fixture sampler");
    ComPtr<ID3D12RootSignature> root;
    check(x2w_startup::create_translated_root(device, d16_entry_constants, 1, samplers, &root),
          "Create translated image_load d16 root signature");

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.VS = {vs.data(), vs.size()};
    pd.PS = {ps.data(), ps.size()};
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = red_only ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R32G32B32A32_UINT;
    pd.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Create image_load d16 PSO");

    const UINT slices = array ? 2u : 1u;
    const UINT subresources = slices * d16_mips;
    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = width;
    td.Height = height;
    td.DepthOrArraySize = static_cast<UINT16>(slices);
    td.MipLevels = d16_mips;
    td.Format = depth16 ? DXGI_FORMAT_R16_UNORM
                        : half_float ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.SampleDesc.Count = 1;
    const auto texture = resource(device, td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT src_footprints[4]{};
    UINT64 upload_size = 0;
    device->GetCopyableFootprints(&td, 0, subresources, 0, src_footprints, nullptr, nullptr, &upload_size);
    const auto upload = make_buffer(device, upload_size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    const D3D12_RANGE none{0, 0};
    check(upload->Map(0, &none, &mapped), "Map image_load d16 texels");
    const UINT texel_bytes = depth16 ? sizeof(std::uint16_t)
                                    : half_float ? 4 * sizeof(std::uint16_t) : 4 * sizeof(std::uint32_t);
    for (UINT s = 0; s < slices; ++s) {
        for (UINT m = 0; m < d16_mips; ++m) {
            const UINT sub = m + s * d16_mips;
            const UINT mw = width >> m, mh = height >> m;
            auto* const sub_bytes = static_cast<std::uint8_t*>(mapped) + src_footprints[sub].Offset;
            for (UINT y = 0; y < mh; ++y) {
                for (UINT x = 0; x < mw; ++x) {
                    auto* const cell = sub_bytes + y * src_footprints[sub].Footprint.RowPitch + x * texel_bytes;
                    if (half_float) {
                        // Per-pixel finite binary16 values, with distinct signs and magnitudes across channels.
                        const UINT pixel = y * mw + x;
                        const std::uint16_t texel[4] = {
                            static_cast<std::uint16_t>(0x3800 + pixel),
                            static_cast<std::uint16_t>(0xc000 + pixel),
                            static_cast<std::uint16_t>(0x4400 + pixel),
                            static_cast<std::uint16_t>(0x3400 + pixel),
                        };
                        std::memcpy(cell, texel, sizeof(texel));
                    } else if (m == 0) {
                        const UINT idx = s == 0 ? (y * width + x) : (width * height - 1u - (y * width + x));
                        if (depth16) std::memcpy(cell, &kDepthTexels[idx], texel_bytes);
                        else std::memcpy(cell, &kFloatTexels[idx][0], texel_bytes);
                    } else {
                        const UINT idx = s == 0 ? x : ((width / 2 - 1u) - x);
                        if (depth16) std::memcpy(cell, &kDepthMip1[s][x], texel_bytes);
                        else std::memcpy(cell, &kFloatMip1[idx][0], texel_bytes);
                    }
                }
            }
        }
    }
    upload->Unmap(0, nullptr);

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = width;
    rd.Height = height;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = pd.RTVFormats[0];
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const auto target = resource(device, rd, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT target_footprint{};
    UINT64 readback_size = 0;
    device->GetCopyableFootprints(&rd, 0, 1, 0, &target_footprint, nullptr, nullptr, &readback_size);
    const auto readback = make_buffer(device, readback_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> views, rtvs;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)), "Create image_load d16 SRV heap");
    device->CreateShaderResourceView(texture.Get(), nullptr, views->GetCPUDescriptorHandleForHeapStart());
    hd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvs)), "Create image_load d16 RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    for (UINT sub = 0; sub < subresources; ++sub) {
        D3D12_TEXTURE_COPY_LOCATION to{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        to.SubresourceIndex = sub;
        D3D12_TEXTURE_COPY_LOCATION from{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        from.PlacedFootprint = src_footprints[sub];
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    transition(list, texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    list->SetGraphicsRootSignature(root.Get());
    ID3D12DescriptorHeap* heaps[]{views.Get()};
    list->SetDescriptorHeaps(1, heaps);
    const std::uint32_t constants[d16_entry_constants] = {0, 0, 0, 0, depth16 ? 1u : 0u};
    list->SetGraphicsRoot32BitConstants(0, d16_entry_constants, constants, 0);
    list->SetGraphicsRootDescriptorTable(1, views->GetGPUDescriptorHandleForHeapStart());
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION out{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    out.PlacedFootprint = target_footprint;
    D3D12_TEXTURE_COPY_LOCATION in{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    in.SubresourceIndex = 0;
    list->CopyTextureRegion(&out, 0, 0, 0, &in, nullptr);
    commands.submit_and_wait(device);

    const D3D12_RANGE read{0, static_cast<SIZE_T>(readback_size)};
    check(readback->Map(0, &read, &mapped), "Map image_load d16 readback");
    const auto* const rb_bytes = static_cast<const std::uint8_t*>(mapped) + target_footprint.Offset;
    for (UINT y = 0; y < height; ++y) {
        const auto* row = rb_bytes + y * target_footprint.Footprint.RowPitch;
        if (red_only) {
            for (UINT x = 0; x < width; ++x)
                std::memcpy(&words[(y * width + x) * 4], row + x * sizeof(std::uint32_t), sizeof(std::uint32_t));
        } else {
            std::memcpy(&words[y * width * 4], row, width * 4 * sizeof(std::uint32_t));
        }
    }
    readback->Unmap(0, &none);
}
