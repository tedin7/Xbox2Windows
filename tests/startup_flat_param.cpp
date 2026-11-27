// Desktop readback for flat parameters in tools/startup/gfx2hlsl.py (v_interp_mov_f32, #1259).
//
// Draws a full-screen triangle and returns four words per R32G32B32A32_FLOAT pixel.
// The caller can supply b0 entry constants and bind raw input words at t0.
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
                              const std::uint32_t* constants, UINT constant_count,
                              const std::vector<std::uint32_t>& raw_srv, UINT instances,
                              bool rgba8, std::uint32_t* uint_words)
{
    const auto vs = read_blob(vertex_path);
    const auto ps = pixel_path ? read_blob(pixel_path) : std::vector<char>{};

    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, constant_count};
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor = {0, 0};
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC rs{};
    rs.NumParameters = raw_srv.empty() ? 1 : 2;
    rs.pParameters = params;
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
    pd.BlendState.IndependentBlendEnable = uint_words != nullptr;
    pd.BlendState.RenderTarget[1].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = cull;
    pd.RasterizerState.FrontCounterClockwise = front_ccw;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = uint_words ? 2 : 1;
    pd.RTVFormats[0] = rgba8 ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R32G32B32A32_FLOAT;
    if (uint_words) pd.RTVFormats[1] = DXGI_FORMAT_R8G8B8A8_UINT;
    pd.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    if (pixel_path)
        check(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Create flat parameter PSO");

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = width;
    rd.Height = height;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, pd.NumRenderTargets, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
    ComPtr<ID3D12DescriptorHeap> rtvs;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvs)), "Create flat parameter RTV heap");
    const auto first_rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    const UINT stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    ComPtr<ID3D12Resource> targets[2], readbacks[2];
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[2]{};
    UINT64 sizes[2]{};

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    for (UINT t = 0; t < pd.NumRenderTargets; ++t) {
        rd.Format = pd.RTVFormats[t];
        D3D12_CLEAR_VALUE clear{rd.Format, {-1.0f, -1.0f, -1.0f, -1.0f}};
        if (rgba8 || t != 0) {
            clear.Color[0] = clear.Color[1] = clear.Color[2] = 0.0f;
            clear.Color[3] = 1.0f;
        }
        targets[t] = resource(device, rd, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
        device->GetCopyableFootprints(&rd, 0, 1, 0, &footprints[t], nullptr, nullptr, &sizes[t]);
        readbacks[t] = make_buffer(device, sizes[t], D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{first_rtv.ptr + size_t(t) * stride};
        device->CreateRenderTargetView(targets[t].Get(), nullptr, rtv);
        list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    }
    list->SetGraphicsRootSignature(root.Get());
    const std::vector<std::uint32_t> zeros(constants ? 0 : constant_count);
    list->SetGraphicsRoot32BitConstants(0, constant_count, constants ? constants : zeros.data(), 0);
    ComPtr<ID3D12Resource> source;
    if (!raw_srv.empty()) {
        const auto size = raw_srv.size() * sizeof(raw_srv[0]);
        source = make_buffer(device, size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        const D3D12_RANGE no_read{0, 0};
        void* mapped = nullptr;
        check(source->Map(0, &no_read, &mapped), "Map pixel buffer input");
        std::memcpy(mapped, raw_srv.data(), size);
        source->Unmap(0, nullptr);
        list->SetGraphicsRootShaderResourceView(1, source->GetGPUVirtualAddress());
    }
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(pd.NumRenderTargets, &first_rtv, TRUE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    if (pixel_path) list->DrawInstanced(vertices, instances, 0, 0);
    for (UINT t = 0; t < pd.NumRenderTargets; ++t) {
        transition(list, targets[t].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION out{readbacks[t].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        out.PlacedFootprint = footprints[t];
        D3D12_TEXTURE_COPY_LOCATION in{targets[t].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        list->CopyTextureRegion(&out, 0, 0, 0, &in, nullptr);
    }
    commands.submit_and_wait(device);

    for (UINT t = 0; t < pd.NumRenderTargets; ++t) {
        void* mapped = nullptr;
        const D3D12_RANGE read{0, static_cast<SIZE_T>(sizes[t])};
        const D3D12_RANGE none{0, 0};
        check(readbacks[t]->Map(0, &read, &mapped), "Map flat parameter readback");
        const auto& footprint = footprints[t];
        const auto* first = static_cast<const std::uint8_t*>(mapped) + footprint.Offset;
        auto* output = t ? uint_words : words;
        const size_t row_words = size_t(width) * ((rgba8 || t != 0) ? 1 : 4);
        for (UINT y = 0; y < height; ++y) {
            std::memcpy(output + size_t(y) * row_words, first + size_t(y) * footprint.Footprint.RowPitch,
                        row_words * sizeof(*words));
        }
        readbacks[t]->Unmap(0, &none);
    }
    if (rgba8) return;
    std::printf("startup_flat_param words=%08x,%08x,%08x,%08x\n", words[0], words[1], words[2], words[3]);
}
