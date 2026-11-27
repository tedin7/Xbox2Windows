// Desktop readback for flat parameters in tools/startup/gfx2hlsl.py (v_interp_mov_f32, #1259).
//
// Draws with a translated vertex and pixel stage and returns four words per pixel.
// The entry constants at b0 are visible to both stages. ud0 is the rectangle-list switch,
// which defaults to zero for plain triangles. R16_UNORM readback zero-extends each texel
// into the first word and returns zeros in the other three.
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
constexpr UINT entry_constants = 4;  // ud0..ud3 of the fixture's manifest

std::vector<char> read_blob(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (!file.good() && !file.eof()) throw std::runtime_error(std::string("Cannot read ") + path);
    if (bytes.empty()) throw std::runtime_error(std::string("Empty shader ") + path);
    return bytes;
}
}  // namespace

void check_startup_flat_param(ID3D12Device* device, const char* vertex_path, const char* pixel_path,
                              std::uint32_t* words, UINT width, UINT height,
                              D3D12_CULL_MODE cull, BOOL front_ccw, UINT vertices,
                              const std::uint32_t* constants, DXGI_FORMAT format)
{
    const auto vs = read_blob(vertex_path);
    const auto ps = read_blob(pixel_path);

    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants = {0, 0, entry_constants};
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rs{};
    rs.NumParameters = 1;
    rs.pParameters = &param;
    ComPtr<ID3DBlob> signature, errors;
    if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &errors)))
        throw std::runtime_error(errors ? static_cast<const char*>(errors->GetBufferPointer())
                                        : "Serialize flat parameter root signature");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&root)),
          "Create flat parameter root signature");

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.VS = {vs.data(), vs.size()};
    pd.PS = {ps.data(), ps.size()};
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = cull;
    pd.RasterizerState.FrontCounterClockwise = front_ccw;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = format;
    pd.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Create flat parameter PSO");

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = width;
    rd.Height = height;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = format;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear{format, {-1.0f, -1.0f, -1.0f, -1.0f}};
    if (format == DXGI_FORMAT_R16_UNORM) clear.Color[0] = 1.0f;
    const auto target = resource(device, rd, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 readback_size = 0;
    device->GetCopyableFootprints(&rd, 0, 1, 0, &footprint, nullptr, nullptr, &readback_size);
    const auto readback = make_buffer(device, readback_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    ComPtr<ID3D12DescriptorHeap> rtvs;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvs)), "Create flat parameter RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    list->SetGraphicsRootSignature(root.Get());
    const std::uint32_t zeros[entry_constants] = {0, 0, 0, 0};
    list->SetGraphicsRoot32BitConstants(0, entry_constants, constants ? constants : zeros, 0);
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(vertices, 1, 0, 0);
    transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION out{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    out.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION in{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&out, 0, 0, 0, &in, nullptr);
    commands.submit_and_wait(device);

    void* mapped = nullptr;
    const D3D12_RANGE read{0, static_cast<SIZE_T>(readback_size)};
    const D3D12_RANGE none{0, 0};
    check(readback->Map(0, &read, &mapped), "Map flat parameter readback");
    const auto* first = static_cast<const std::uint8_t*>(mapped) + footprint.Offset;
    for (UINT y = 0; y < height; ++y) {
        const auto* row = first + size_t(y) * footprint.Footprint.RowPitch;
        if (format == DXGI_FORMAT_R16_UNORM) {
            for (UINT x = 0; x < width; ++x) {
                std::uint16_t packed;
                std::memcpy(&packed, row + size_t(x) * sizeof(packed), sizeof(packed));
                auto* pixel = words + (size_t(y) * width + x) * 4;
                pixel[0] = packed;
                pixel[1] = pixel[2] = pixel[3] = 0;
            }
        } else {
            std::memcpy(words + size_t(y) * width * 4, row, size_t(width) * 4 * sizeof(*words));
        }
    }
    readback->Unmap(0, &none);
    std::printf("startup_flat_param words=%08x,%08x,%08x,%08x\n", words[0], words[1], words[2], words[3]);
}
