#include "desktop_gpu.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>
using Microsoft::WRL::ComPtr;

namespace {
std::vector<char> read_bytes(const char* path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open draw input");
    return {std::istreambuf_iterator<char>(stream), {}};
}

}

void check_depth_mask_draw(ID3D12Device* device, const char* first, const char* second, const char* fixture,
                           bool swap, bool color)
{
    constexpr UINT side = 32;
    constexpr UINT cb_bytes = 2048;
    constexpr UINT image_bytes = side * side * 4;
    const auto input = read_bytes(fixture);
    const UINT upload_bytes = color ? image_bytes : 2 * cb_bytes;
    if (input.size() != upload_bytes + (color ? 2 : 1) * image_bytes)
        throw std::runtime_error("Invalid draw fixture size");
    auto shader = read_bytes(first);
    auto pixel = read_bytes(second);
    D3D12_ROOT_PARAMETER params[2]{};
    for (UINT i = 0; i < 2; ++i) {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[i].Descriptor.ShaderRegister = i;
    }
    D3D12_ROOT_SIGNATURE_DESC layout{};
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    if (color) {
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable = {1, &range};
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    }
    layout.NumParameters = color ? 1 : 2;
    layout.pParameters = params;
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&layout, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "Serialize draw root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "Draw root");
    ComPtr<ID3D12PipelineState> pipeline;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.Get();
    desc.VS = {shader.data(), shader.size()};
    desc.PS = {pixel.data(), pixel.size()};
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.DepthStencilState.DepthEnable = TRUE;
    desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    if (color) {
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        auto& blend = desc.BlendState.RenderTarget[0];
        blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
        blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
        blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        blend.LogicOp = D3D12_LOGIC_OP_NOOP;
        blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    }
    desc.SampleMask = UINT_MAX;
    desc.SampleDesc.Count = 1;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    check(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline)), "Create depth-only pipeline");
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = texture.Height = side;
    texture.DepthOrArraySize = texture.MipLevels = texture.SampleDesc.Count = 1;
    texture.Format = DXGI_FORMAT_D32_FLOAT;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    constexpr auto draw_state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
    D3D12_CLEAR_VALUE clear_value{};
    clear_value.Format = texture.Format;
    clear_value.DepthStencil.Depth = 0.9375f;
    auto target = resource(device, texture, D3D12_HEAP_TYPE_DEFAULT, draw_state, &clear_value);
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    heap_desc.NumDescriptors = 1;
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "Create target heap");
    auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(target.Get(), nullptr, handle);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 size = 0;
    device->GetCopyableFootprints(&texture, 0, 1, 0, &footprint, nullptr, nullptr, &size);
    auto readback = make_buffer(device, size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT source_footprint{}, color_footprint{};
    UINT64 source_size = upload_bytes, color_size = 0;
    ComPtr<ID3D12Resource> source, color_target, color_readback;
    ComPtr<ID3D12DescriptorHeap> srv_heap, rtv_heap;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
    if (color) {
        auto source_desc = texture;
        source_desc.Format = DXGI_FORMAT_R32_FLOAT;
        source_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        device->GetCopyableFootprints(&source_desc, 0, 1, 0, &source_footprint, nullptr, nullptr, &source_size);
        source = resource(device, source_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&srv_heap)), "Create source heap");
        device->CreateShaderResourceView(source.Get(), nullptr, srv_heap->GetCPUDescriptorHandleForHeapStart());

        auto color_desc = texture;
        color_desc.Format = desc.RTVFormats[0];
        color_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        device->GetCopyableFootprints(&color_desc, 0, 1, 0, &color_footprint, nullptr, nullptr, &color_size);
        D3D12_CLEAR_VALUE blue{};
        blue.Format = color_desc.Format;
        blue.Color[2] = blue.Color[3] = 1;
        color_target = resource(device, color_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &blue);
        color_readback = make_buffer(device, color_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap)), "Create color heap");
        rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(color_target.Get(), nullptr, rtv);
    }
    auto upload = make_buffer(device, source_size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    D3D12_RANGE empty{};
    check(upload->Map(0, &empty, &mapped), "Map draw input");
    if (color) {
        for (UINT y = 0; y < side; ++y)
            std::memcpy(static_cast<char*>(mapped) + source_footprint.Offset + y * source_footprint.Footprint.RowPitch,
                        input.data() + y * side * 4, side * 4);
    } else {
        std::memcpy(mapped, input.data(), upload_bytes);
    }
    upload->Unmap(0, nullptr);
    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    list->SetGraphicsRootSignature(root.Get());
    if (color) {
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = upload.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = source_footprint;
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = source.Get();
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(list, source.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        ID3D12DescriptorHeap* heaps[]{srv_heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetGraphicsRootDescriptorTable(0, srv_heap->GetGPUDescriptorHandleForHeapStart());
    } else {
        const auto base = upload->GetGPUVirtualAddress();
        list->SetGraphicsRootConstantBufferView(0, base + (swap ? cb_bytes : 0));
        list->SetGraphicsRootConstantBufferView(1, base + (swap ? 0 : cb_bytes));
    }
    const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
    const D3D12_RECT scissor{0, 0, side, side};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(color ? 1 : 0, color ? &rtv : nullptr, FALSE, &handle);
    if (color) {
        const float blue[]{0, 0, 1, 1};
        list->ClearRenderTargetView(rtv, blue, 0, nullptr);
    }
    list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, 0.9375f, 0, 0, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    transition(list, target.Get(), draw_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = target.Get();
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (color) {
        transition(list, color_target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        src.pResource = color_target.Get();
        dst.pResource = color_readback.Get();
        dst.PlacedFootprint = color_footprint;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    commands.submit_and_wait(device);
    auto compare = [&](ID3D12Resource* buffer, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& layout,
                       const char* reference) {
        check(buffer->Map(0, nullptr, &mapped), "Map draw result");
        const char* rows = static_cast<const char*>(mapped) + layout.Offset;
        unsigned mismatches = 0;
        for (UINT y = 0; y < side; ++y)
            for (UINT x = 0; x < side; ++x)
                mismatches += std::memcmp(rows + y * layout.Footprint.RowPitch + 4 * x,
                                           reference + (y * side + x) * 4, 4) != 0;
        buffer->Unmap(0, &empty);
        return mismatches;
    };
    const char* reference = input.data() + upload_bytes;
    const auto depth_errors = compare(readback.Get(), footprint, reference);
    unsigned color_errors = 0;
    if (color) {
        color_errors = compare(color_readback.Get(), color_footprint, reference + image_bytes);
        std::printf("depth mismatches=%u color mismatches=%u\n", depth_errors, color_errors);
    } else {
        std::printf("pixels=1024 mismatches=%u\n", depth_errors);
    }
    if (depth_errors || color_errors) throw ReadbackMismatch("Draw reference mismatch");
}
