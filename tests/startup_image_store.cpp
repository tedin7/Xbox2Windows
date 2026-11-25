// Desktop readback for tools/startup/gfx2hlsl.py's pixel-stage image_store (#1996 item F).
//
// The translated pixel shader stores to w0 : register(u0), a RWTexture2D<float4>, and reads its entry
// constants from cbuffer entry : register(b0). This draws one pixel with that shader and a source-built
// full-screen vertex shader into a 4x4 R32G32B32A32_FLOAT texture whose texels all start as -1.0, then
// reads every texel back. The caller names the one texel it expects the store to change and its value;
// every other texel must still hold -1.0.
#include "desktop_gpu.h"

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

ComPtr<ID3D12RootSignature> store_root(ID3D12Device* device)
{
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
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
}  // namespace

// Returns 0 when texel (x, y) holds expected and every other texel holds -1.0; otherwise prints the first
// differing texel as "texel_mismatch x=.. y=.. expected=.. actual=.." and returns 1.
int check_startup_image_store(ID3D12Device* device, const char* vertex_path, const char* pixel_path, UINT x, UINT y,
                              const std::uint32_t expected[4])
{
    if (x >= side || y >= side) throw std::runtime_error("Expected a texel inside the 4x4 image");
    const auto vs = read_shader(vertex_path);
    const auto ps = read_shader(pixel_path);
    const auto root = store_root(device);

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
    td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.SampleDesc.Count = 1;
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto image = resource(device, td, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&td, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    const auto upload = make_buffer(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    const auto readback = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    void* mapped = nullptr;
    const D3D12_RANGE none{0, 0};
    check(upload->Map(0, &none, &mapped), "Map image store upload");
    for (UINT row = 0; row < side; ++row)
        for (UINT word = 0; word < side * 4; ++word)
            std::memcpy(static_cast<std::uint8_t*>(mapped) + footprint.Offset + row * footprint.Footprint.RowPitch + 4 * word,
                        &untouched, 4);
    upload->Unmap(0, nullptr);

    D3D12_RESOURCE_DESC rd = td;
    rd.Width = rd.Height = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R32G32B32A32_FLOAT, {0.0f, 0.0f, 0.0f, 0.0f}};
    const auto target = resource(device, rd, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> views, rtvs;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)), "Create image store view heap");
    device->CreateUnorderedAccessView(image.Get(), nullptr, nullptr, views->GetCPUDescriptorHandleForHeapStart());
    hd = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvs)), "Create image store RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    D3D12_TEXTURE_COPY_LOCATION texture_location{image.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION upload_location{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    upload_location.PlacedFootprint = footprint;
    list->CopyTextureRegion(&texture_location, 0, 0, 0, &upload_location, nullptr);
    transition(list, image.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    list->SetGraphicsRootSignature(root.Get());
    ID3D12DescriptorHeap* heaps[]{views.Get()};
    list->SetDescriptorHeaps(1, heaps);
    const std::uint32_t constants[entry_constants] = {0, 0, 0, 0};  // ud0 = 0: plain triangles
    list->SetGraphicsRoot32BitConstants(0, entry_constants, constants, 0);
    list->SetGraphicsRootDescriptorTable(1, views->GetGPUDescriptorHandleForHeapStart());
    const D3D12_VIEWPORT viewport{0, 0, 1, 1, 0, 1};
    const D3D12_RECT scissor{0, 0, 1, 1};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    transition(list, image.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION readback_location{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    readback_location.PlacedFootprint = footprint;
    list->CopyTextureRegion(&readback_location, 0, 0, 0, &texture_location, nullptr);
    commands.submit_and_wait(device);

    const D3D12_RANGE read{0, static_cast<SIZE_T>(bytes)};
    check(readback->Map(0, &read, &mapped), "Map image store readback");
    int result = 0;
    for (UINT ty = 0; ty < side && !result; ++ty)
        for (UINT tx = 0; tx < side && !result; ++tx) {
            std::uint32_t texel[4];
            std::memcpy(texel, static_cast<const std::uint8_t*>(mapped) + footprint.Offset + ty * footprint.Footprint.RowPitch +
                                   16 * tx, sizeof(texel));
            const bool stored = tx == x && ty == y;
            const std::uint32_t want[4] = {stored ? expected[0] : untouched, stored ? expected[1] : untouched,
                                           stored ? expected[2] : untouched, stored ? expected[3] : untouched};
            if (std::memcmp(texel, want, sizeof(texel)) != 0) {
                std::fprintf(stderr, "texel_mismatch x=%u y=%u expected=%08x,%08x,%08x,%08x actual=%08x,%08x,%08x,%08x\n",
                             tx, ty, want[0], want[1], want[2], want[3], texel[0], texel[1], texel[2], texel[3]);
                result = 1;
            }
        }
    readback->Unmap(0, &none);
    if (!result) std::printf("image store: texel (%u, %u) written, %u others untouched\n", x, y, side * side - 1);
    return result;
}
