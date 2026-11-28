// Desktop readback for tools/startup/gfx2hlsl.py's pixel-stage image_store (#1996 item F).
//
// The translated pixel shader stores to w0 : register(u0), a RWTexture2D<float4>, and reads its entry
// constants from cbuffer entry : register(b0). This draws one or two pixels (coverage) with that shader
// and a source-built full-screen vertex shader into a 4x4 R32G32B32A32_FLOAT texture whose texels all
// start as -1.0, then reads every texel back. The caller names the one texel it expects the store to
// change and its value; every other texel must still hold -1.0. Coverage 2 uses a 2x1 viewport so two
// invocations run in one draw.
#include "desktop_gpu.h"
#include "startup_stubs/typed_uav_load.h"
#include "startup_stubs/translated_sampler_bindings.h"

#include <array>
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
constexpr UINT entry_constants = 4;
constexpr UINT side = 4;
constexpr std::uint32_t untouched = 0xbf800000;  // -1.0f

std::vector<char> read_shader(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (bytes.empty()) throw std::runtime_error(std::string("Cannot read shader ") + path);
    return bytes;
}

ComPtr<ID3D12RootSignature> store_root(ID3D12Device* device, UINT count)
{
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, count, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, entry_constants};
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = {1, &range};
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    const D3D12_ROOT_SIGNATURE_DESC desc{2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob, errors;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors), "Serialize image store root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)),
          "Create image store root");
    return root;
}
struct Image {
    DXGI_FORMAT format;
    UINT channels;
    const std::uint32_t* initial;
    const std::uint32_t* expected;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    ComPtr<ID3D12Resource> texture, upload, readback;
};

int draw_and_check(ID3D12Device* device, const char* vertex_path, const char* pixel_path,
                   Image* images, UINT count, UINT draw_width, UINT draw_height)
{
    const auto vs = read_shader(vertex_path);
    const auto ps = read_shader(pixel_path);
    const auto root = store_root(device, count);

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
    check(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Create image store PSO");

    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = side;
    td.Height = side;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.SampleDesc.Count = 1;
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const D3D12_RANGE none{0, 0};
    for (UINT i = 0; i < count; ++i) {
        Image& image = images[i];
        td.Format = image.format;
        image.texture = resource(device, td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        device->GetCopyableFootprints(&td, 0, 1, 0, &image.footprint, nullptr, nullptr, &image.bytes);
        image.upload = make_buffer(device, image.bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        image.readback = make_buffer(device, image.bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        void* mapped = nullptr;
        check(image.upload->Map(0, &none, &mapped), "Map image write upload");
        const UINT row_words = side * image.channels;
        for (UINT row = 0; row < side; ++row)
            std::memcpy(static_cast<std::uint8_t*>(mapped) + image.footprint.Offset + row * image.footprint.Footprint.RowPitch,
                        image.initial + row * row_words, sizeof(std::uint32_t) * row_words);
        image.upload->Unmap(0, nullptr);
    }

    D3D12_RESOURCE_DESC rd = td;
    rd.Width = draw_width;
    rd.Height = draw_height;
    rd.Format = pd.RTVFormats[0];
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_CLEAR_VALUE clear{rd.Format, {0.0f, 0.0f, 0.0f, 0.0f}};
    const auto target = resource(device, rd, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, count, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> views, rtvs;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)), "Create image store view heap");
    auto dest = views->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < count; ++i) {
        device->CreateUnorderedAccessView(images[i].texture.Get(), nullptr, nullptr, dest);
        dest.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    hd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvs)), "Create image store RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    for (UINT i = 0; i < count; ++i) {
        const Image& image = images[i];
        const D3D12_TEXTURE_COPY_LOCATION texture{image.texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION upload{image.upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        upload.PlacedFootprint = image.footprint;
        list->CopyTextureRegion(&texture, 0, 0, 0, &upload, nullptr);
        transition(list, image.texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    list->SetGraphicsRootSignature(root.Get());
    ID3D12DescriptorHeap* heaps[]{views.Get()};
    list->SetDescriptorHeaps(1, heaps);
    const std::uint32_t constants[entry_constants] = {0, 0, 0, 0};  // ud0 = 0: plain triangles
    list->SetGraphicsRoot32BitConstants(0, entry_constants, constants, 0);
    list->SetGraphicsRootDescriptorTable(1, views->GetGPUDescriptorHandleForHeapStart());
    const D3D12_VIEWPORT viewport{0, 0, float(draw_width), float(draw_height), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(draw_width), LONG(draw_height)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    for (UINT i = 0; i < count; ++i) {
        const Image& image = images[i];
        transition(list, image.texture.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        const D3D12_TEXTURE_COPY_LOCATION texture{image.texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION readback{image.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        readback.PlacedFootprint = image.footprint;
        list->CopyTextureRegion(&readback, 0, 0, 0, &texture, nullptr);
    }
    commands.submit_and_wait(device);

    int result = 0;
    for (UINT i = 0; i < count; ++i) {
        const Image& image = images[i];
        const D3D12_RANGE read{0, static_cast<SIZE_T>(image.bytes)};
        void* mapped = nullptr;
        check(image.readback->Map(0, &read, &mapped), "Map image write readback");
        for (UINT y = 0; y < side; ++y)
            for (UINT x = 0; x < side; ++x) {
                std::uint32_t actual[4]{};
                const auto* want = image.expected + (y * side + x) * image.channels;
                const auto* address = static_cast<const std::uint8_t*>(mapped) + image.footprint.Offset
                                    + y * image.footprint.Footprint.RowPitch + 4 * x * image.channels;
                const auto size = sizeof(std::uint32_t) * image.channels;
                std::memcpy(actual, address, size);
                if (std::memcmp(actual, want, size) == 0) continue;
                if (image.channels == 1)
                    std::fprintf(stderr, "texel_mismatch uav=%u x=%u y=%u expected=%08x actual=%08x\n",
                                 i, x, y, want[0], actual[0]);
                else
                    std::fprintf(stderr, "texel_mismatch x=%u y=%u expected=%08x,%08x,%08x,%08x actual=%08x,%08x,%08x,%08x\n",
                                 x, y, want[0], want[1], want[2], want[3], actual[0], actual[1], actual[2], actual[3]);
                result = 1;
            }
        image.readback->Unmap(0, &none);
    }
    return result;
}
}  // namespace

int check_startup_image_store(ID3D12Device* device, const char* vertex_path, const char* pixel_path, UINT x, UINT y,
                              const std::uint32_t expected[4], unsigned coverage)
{
    if (coverage != 1 && coverage != 2) throw std::runtime_error("Image store coverage is not 1 or 2");
    if (x >= side || y >= side) throw std::runtime_error("Expected a texel inside the 4x4 image");
    std::array<std::uint32_t, side * side * 4> initial, want;
    initial.fill(untouched);
    want = initial;
    std::memcpy(want.data() + (y * side + x) * 4, expected, 4 * sizeof(std::uint32_t));
    Image image{DXGI_FORMAT_R32G32B32A32_FLOAT, 4, initial.data(), want.data()};
    const int result = draw_and_check(device, vertex_path, pixel_path, &image, 1, coverage, 1);
    if (!result) std::printf("image store: texel (%u, %u) written, %u others untouched\n", x, y, side * side - 1);
    return result;
}

int check_startup_image_atomic(ID3D12Device* device, const char* vertex_path, const char* pixel_path,
                               const char* fixture_path, bool signed_view)
{
    // Initial texels of u0 and u1, followed by both CPU references, in row-major order.
    std::uint32_t fixture[4][side * side]{};
    std::ifstream file(fixture_path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(fixture), sizeof(fixture)) || file.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("Expected four 4x4 images in the atomic fixture");
    Image images[] = {
        {signed_view ? DXGI_FORMAT_R32_SINT : DXGI_FORMAT_R32_UINT, 1, fixture[0], fixture[2]},
        {DXGI_FORMAT_R32_UINT, 1, fixture[1], fixture[3]},
    };
    return draw_and_check(device, vertex_path, pixel_path, images, 2, side, side);
}

// Colour formats a pixel UAV may store. None of these is an R32 typed-load format, so a missing
// UAV_TYPED_LOAD bit, or a device without TypedUAVLoadAdditionalFormats, is a real refusal.
// The function does not create a command list and does not draw.
const char* check_startup_image_store_typed_load(ID3D12Device* device)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    const bool options_ok = SUCCEEDED(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)));
    const bool additional = options_ok && options.TypedUAVLoadAdditionalFormats;
    const DXGI_FORMAT candidates[] = {
        DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R16G16B16A16_UNORM,
        DXGI_FORMAT_R16G16B16A16_SNORM, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R32G32_FLOAT,
        DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R32G32B32A32_FLOAT,
    };
    for (DXGI_FORMAT format : candidates) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
        if (FAILED(check_feature(device, D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)))) continue;
        if (typed_uav_load_refusal(format, support.Support2, additional)) {
            std::fprintf(stderr, "typed_uav_load_refused format=%d\n", static_cast<int>(format));
            return "typed_uav_load_refused";
        }
    }
    // Every queried format can load on this device. The same helper must still refuse a cleared
    // UAV_TYPED_LOAD bit, and this path still returns before any draw.
    if (typed_uav_load_refusal(DXGI_FORMAT_R11G11B10_FLOAT, 0, additional)) {
        std::fprintf(stderr, "typed_uav_load_refused format=%d support2=0\n",
                     static_cast<int>(DXGI_FORMAT_R11G11B10_FLOAT));
        return "typed_uav_load_refused";
    }
    return nullptr;
}

// The fixture holds the manifest's entry words, R32_FLOAT texels, initial and expected
// raw buffer words, and the expected uniform RGBA32_FLOAT target word.
int check_startup_pixel_formatted_store(ID3D12Device* device, const char* vertex_path, const char* pixel_path,
                                        const char* sampler_line, const char* fixture_path)
{
    struct Fixture {
        std::uint32_t entry_count, entries[32];
        std::uint32_t image[4], ramp[4], initial[12], expected[12], color;
    } input{};
    static_assert(sizeof(Fixture) == 264, "Pixel formatted store fixture layout");
    std::ifstream fixture(fixture_path, std::ios::binary);
    if (!fixture.read(reinterpret_cast<char*>(&input), sizeof(input)) || fixture.peek() != EOF ||
        !input.entry_count || input.entry_count > 32)
        throw std::runtime_error("Invalid pixel formatted store fixture");

    const auto vs = read_shader(vertex_path);
    const auto ps = read_shader(pixel_path);
    x2w_startup::TranslatedSamplers samplers;
    if (!samplers.read_manifest_line(sampler_line) || samplers.statics.size() != 1)
        throw std::runtime_error("Expected one static sampler for the pixel formatted store");
    ComPtr<ID3D12RootSignature> root;
    check(x2w_startup::create_translated_root(device, input.entry_count, 2, samplers, &root, 1),
          "Create pixel formatted store root");

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
    check(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Create pixel formatted store PSO");

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> views, rtvs;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)), "Create formatted store view heap");
    hd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvs)), "Create formatted store RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    auto view = views->GetCPUDescriptorHandleForHeapStart();
    const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    ComPtr<ID3D12Resource> textures[2], uploads[2];
    const D3D12_RANGE none{0, 0};
    for (UINT i = 0; i < 2; ++i) {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = i ? D3D12_RESOURCE_DIMENSION_TEXTURE1D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = i ? 4 : 2;
        desc.Height = i ? 1 : 2;
        desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R32_FLOAT;
        desc.SampleDesc.Count = 1;
        textures[i] = resource(device, desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64 bytes = 0;
        device->GetCopyableFootprints(&desc, 0, 1, 0, &fp, nullptr, nullptr, &bytes);
        uploads[i] = make_buffer(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        check(uploads[i]->Map(0, &none, &mapped), "Map formatted store texture upload");
        const auto* texels = i ? input.ramp : input.image;
        const auto width = static_cast<std::size_t>(desc.Width);
        for (UINT y = 0; y < desc.Height; ++y)
            std::memcpy(static_cast<char*>(mapped) + fp.Offset + y * fp.Footprint.RowPitch, texels + y * width,
                        width * sizeof(*texels));
        uploads[i]->Unmap(0, nullptr);
        D3D12_TEXTURE_COPY_LOCATION src{uploads[i].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        src.PlacedFootprint = fp;
        const D3D12_TEXTURE_COPY_LOCATION dst{textures[i].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(list, textures[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        device->CreateShaderResourceView(textures[i].Get(), nullptr, view);
        view.ptr += increment;
    }

    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = sizeof(input.initial);
    bd.Height = bd.DepthOrArraySize = bd.MipLevels = bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto buffer = resource(device, bd, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    const auto upload = make_buffer(device, bd.Width, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    const auto buffer_readback = make_buffer(device, bd.Width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    void* mapped = nullptr;
    check(upload->Map(0, &none, &mapped), "Map formatted store buffer upload");
    std::memcpy(mapped, input.initial, sizeof(input.initial));
    upload->Unmap(0, nullptr);
    list->CopyBufferRegion(buffer.Get(), 0, upload.Get(), 0, bd.Width);
    transition(list, buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
    raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    raw.Buffer.NumElements = sizeof(input.initial) / sizeof(input.initial[0]);
    raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    device->CreateUnorderedAccessView(buffer.Get(), nullptr, &raw, view);

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = rd.Height = side;
    rd.DepthOrArraySize = rd.MipLevels = rd.SampleDesc.Count = 1;
    rd.Format = pd.RTVFormats[0];
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const auto target = resource(device, rd, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);
    const float clear[4]{-1, -1, -1, -1};
    list->ClearRenderTargetView(rtv, clear, 0, nullptr);
    list->SetGraphicsRootSignature(root.Get());
    ID3D12DescriptorHeap* heaps[]{views.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetGraphicsRoot32BitConstants(0, input.entry_count, input.entries, 0);
    list->SetGraphicsRootDescriptorTable(1, views->GetGPUDescriptorHandleForHeapStart());
    const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
    const D3D12_RECT scissor{0, 0, side, side};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);

    transition(list, buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(buffer_readback.Get(), 0, buffer.Get(), 0, bd.Width);
    transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 target_bytes = 0;
    device->GetCopyableFootprints(&rd, 0, 1, 0, &fp, nullptr, nullptr, &target_bytes);
    const auto target_readback = make_buffer(device, target_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const D3D12_TEXTURE_COPY_LOCATION src{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION dst{target_readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint = fp;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    commands.submit_and_wait(device);

    const D3D12_RANGE buffer_range{0, sizeof(input.expected)};
    check(buffer_readback->Map(0, &buffer_range, &mapped), "Map formatted store buffer readback");
    bool mismatch = false;
    const auto* buffer_words = static_cast<const std::uint32_t*>(mapped);
    for (UINT i = 0; i < sizeof(input.expected) / sizeof(input.expected[0]); ++i) {
        if (buffer_words[i] == input.expected[i]) continue;
        std::fprintf(stderr, "record_mismatch record=%u word=%u expected=%08x actual=%08x\n",
                     i / 2, i % 2, input.expected[i], buffer_words[i]);
        mismatch = true;
    }
    buffer_readback->Unmap(0, &none);
    const D3D12_RANGE target_range{0, static_cast<SIZE_T>(target_bytes)};
    check(target_readback->Map(0, &target_range, &mapped), "Map formatted store target readback");
    for (UINT y = 0; y < side; ++y) {
        const auto* row = reinterpret_cast<const std::uint32_t*>(
            static_cast<const char*>(mapped) + fp.Offset + y * fp.Footprint.RowPitch);
        for (UINT x = 0; x < side; ++x) {
            for (UINT c = 0; c < 4; ++c) {
                if (row[4 * x + c] == input.color) continue;
                std::fprintf(stderr, "pixel_mismatch x=%u y=%u channel=%u expected=%08x actual=%08x\n",
                             x, y, c, input.color, row[4 * x + c]);
                mismatch = true;
            }
        }
    }
    target_readback->Unmap(0, &none);
    return mismatch ? 1 : 0;
}
