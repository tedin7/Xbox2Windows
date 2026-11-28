// Desktop readback with caller-supplied entry constants for tools/startup/gfx2hlsl.py (v_med3_f32, #2044).
//
// Draws one pixel with a translated vertex and pixel stage and returns the four 32-bit words of
// the colour. The entry constants (cbuffer entry : register(b0)) are visible to both stages, because
// the translated vertex stage reads ud0, the rectangle-list switch; every word comes from the caller,
// so a value under test is never a compile-time literal the shader compiler could fold. The target is
// R32G32B32A32_FLOAT, so the readback is the raw words the pixel stage exported.
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

void check_startup_entry_words(ID3D12Device* device, const char* vertex_path, const char* pixel_path,
                               const std::uint32_t* entry, UINT entry_count, std::uint32_t* words,
                               UINT width, UINT height, bool dynamic_heap,
                               const std::vector<std::vector<std::uint32_t>>& records, bool integer_images)
{
    const auto vs = read_blob(vertex_path);
    const auto ps = read_blob(pixel_path);

    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, entry_count};
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor = {0, 0};
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rs{};
    if (dynamic_heap) rs.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(0x400);
    rs.NumParameters = dynamic_heap ? 2 : 1;
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
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    pd.SampleDesc.Count = 1;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Create flat parameter PSO");

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = width;
    rd.Height = height;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R32G32B32A32_FLOAT, {-1.0f, -1.0f, -1.0f, -1.0f}};
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
    ComPtr<ID3D12Resource> source;
    ComPtr<ID3D12DescriptorHeap> views, image_rtvs;
    ComPtr<ID3D12Resource> images[2], integer_textures[3], integer_uploads[3];
    constexpr UINT buffer_stride = D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT;
    constexpr UINT index_table_offset = 3 * buffer_stride;
    if (dynamic_heap && !records.empty()) {
        constexpr UINT region = 256;
        source = make_buffer(device, records.size() * region, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        const D3D12_RANGE none{0, 0};
        check(source->Map(0, &none, &mapped), "Map heap records");
        std::memset(mapped, 0, records.size() * region);
        for (size_t i = 0; i < records.size(); ++i)
            std::memcpy(static_cast<unsigned char*>(mapped) + i * region, records[i].data(), records[i].size() * 4);
        source->Unmap(0, nullptr);
        const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, static_cast<UINT>(records.size()), D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&views)), "Create heap records");
        auto cpu = views->GetCPUDescriptorHandleForHeapStart();
        const UINT increment = device->GetDescriptorHandleIncrementSize(desc.Type);
        for (UINT i = 0; i < records.size(); ++i, cpu.ptr += increment) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{DXGI_FORMAT_R32_TYPELESS, D3D12_SRV_DIMENSION_BUFFER, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
            srv.Buffer = {i * (region / 4), static_cast<UINT>(records[i].size()), 0, D3D12_BUFFER_SRV_FLAG_RAW};
            device->CreateShaderResourceView(source.Get(), &srv, cpu);
        }
        ID3D12DescriptorHeap* binding[]{views.Get()};
        list->SetDescriptorHeaps(1, binding);
    } else if (integer_images) {
        // Slots 0..2 are Texture1D views of R32_UINT, R32_SINT and R32_FLOAT, texel 0 of each holding one value.
        // Slot 3 is the heap info table the translated shader reads (ByteAddressBuffer at heap_count): per
        // record the stride word, then the view class the runtime writes for a mirrored texture (1 UINT, 2 SINT, 0 other).
        constexpr std::uint32_t values[3]{0x3fa00000u, 0x40200000u, 0x40800000u};
        constexpr DXGI_FORMAT formats[3]{DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32_FLOAT};
        constexpr std::uint32_t info[6]{0, 1, 0, 2, 0, 0};
        source = make_buffer(device, index_table_offset + 16, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        const D3D12_RANGE none{0, 0};
        check(source->Map(0, &none, &mapped), "Map integer heap info");
        std::memset(mapped, 0, index_table_offset + 16);
        std::memcpy(mapped, info, sizeof(info));
        source->Unmap(0, nullptr);
        const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&views)), "Create integer fixture heap");
        auto cpu = views->GetCPUDescriptorHandleForHeapStart();
        const UINT increment = device->GetDescriptorHandleIncrementSize(desc.Type);
        for (UINT i = 0; i < 3; ++i, cpu.ptr += increment) {
            D3D12_RESOURCE_DESC texture_desc{};
            texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
            texture_desc.Width = 4;
            texture_desc.Height = 1;
            texture_desc.DepthOrArraySize = 1;
            texture_desc.MipLevels = 1;
            texture_desc.Format = formats[i];
            texture_desc.SampleDesc.Count = 1;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT texture_footprint{};
            UINT64 upload_size = 0;
            device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &texture_footprint, nullptr, nullptr, &upload_size);
            integer_uploads[i] = make_buffer(device, upload_size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            void* texels = nullptr;
            check(integer_uploads[i]->Map(0, &none, &texels), "Map integer texture upload");
            std::memset(texels, 0, static_cast<size_t>(upload_size));
            std::memcpy(static_cast<unsigned char*>(texels) + texture_footprint.Offset, &values[i], sizeof(values[i]));
            integer_uploads[i]->Unmap(0, nullptr);
            integer_textures[i] = resource(device, texture_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, nullptr);
            D3D12_TEXTURE_COPY_LOCATION to{integer_textures[i].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            D3D12_TEXTURE_COPY_LOCATION from{integer_uploads[i].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            from.PlacedFootprint = texture_footprint;
            list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            transition(list, integer_textures[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            D3D12_SHADER_RESOURCE_VIEW_DESC view{formats[i], D3D12_SRV_DIMENSION_TEXTURE1D,
                                                 D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
            view.Texture1D = {0, 1, 0.0f};
            device->CreateShaderResourceView(integer_textures[i].Get(), &view, cpu);
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC table{DXGI_FORMAT_R32_TYPELESS, D3D12_SRV_DIMENSION_BUFFER,
                                              D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        table.Buffer = {0, 6, 0, D3D12_BUFFER_SRV_FLAG_RAW};
        device->CreateShaderResourceView(source.Get(), &table, cpu);
        ID3D12DescriptorHeap* binding[]{views.Get()};
        list->SetDescriptorHeaps(1, binding);
    } else if (dynamic_heap) {
        constexpr std::uint32_t values[]{0x3fa00000u, 0x40200000u, 0x40800000u};
        constexpr std::uint32_t image_indices[]{6, 7};
        source = make_buffer(device, index_table_offset + sizeof(image_indices), D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        const D3D12_RANGE none{0, 0};
        check(source->Map(0, &none, &mapped), "Map heap fixture input");
        for (UINT i = 0; i < 3; ++i)
            std::memcpy(static_cast<unsigned char*>(mapped) + i * buffer_stride, values + i, sizeof(values[i]));
        std::memcpy(static_cast<unsigned char*>(mapped) + index_table_offset, image_indices, sizeof(image_indices));
        source->Unmap(0, nullptr);
        // Slots 0..2 are raw buffers. Slots 3 and 4 are textures, repeated at 6 and 7 for the shift control.
        // Slot 5 is a typed null texture for the image dimensions control.
        const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8,
                                            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&views)), "Create raw fixture heap");
        auto cpu = views->GetCPUDescriptorHandleForHeapStart();
        const UINT increment = device->GetDescriptorHandleIncrementSize(desc.Type);
        for (UINT i = 0; i < 3; ++i, cpu.ptr += increment) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{DXGI_FORMAT_R32_TYPELESS, D3D12_SRV_DIMENSION_BUFFER,
                                               D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
            srv.Buffer = {i * (buffer_stride / 4), 1, 0, D3D12_BUFFER_SRV_FLAG_RAW};
            device->CreateShaderResourceView(source.Get(), &srv, cpu);
        }
        D3D12_RESOURCE_DESC image_desc{};
        image_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        image_desc.DepthOrArraySize = 1;
        image_desc.MipLevels = 3;
        image_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        image_desc.SampleDesc.Count = 1;
        image_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        static constexpr float texels[2][4]{{1.25f, 2.5f, 4.0f, 8.0f}, {16.0f, 32.0f, 64.0f, 128.0f}};
        static constexpr UINT widths[]{37, 43}, heights[]{11, 15};
        const D3D12_DESCRIPTOR_HEAP_DESC image_rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE};
        check(device->CreateDescriptorHeap(&image_rtv_desc, IID_PPV_ARGS(&image_rtvs)), "Create heap fixture RTV heap");
        auto image_rtv = image_rtvs->GetCPUDescriptorHandleForHeapStart();
        const UINT rtv_increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        for (UINT i = 0; i < 2; ++i, cpu.ptr += increment, image_rtv.ptr += rtv_increment) {
            image_desc.Width = widths[i];
            image_desc.Height = heights[i];
            D3D12_CLEAR_VALUE value{DXGI_FORMAT_R32G32B32A32_FLOAT, {}};
            std::memcpy(value.Color, texels[i], sizeof(value.Color));
            images[i] = resource(device, image_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &value);
            D3D12_RENDER_TARGET_VIEW_DESC image_target{DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RTV_DIMENSION_TEXTURE2D};
            image_target.Texture2D = {i, 0};
            device->CreateRenderTargetView(images[i].Get(), &image_target, image_rtv);
            list->ClearRenderTargetView(image_rtv, value.Color, 0, nullptr);
            transition(list, images[i].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            D3D12_SHADER_RESOURCE_VIEW_DESC view{DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_SRV_DIMENSION_TEXTURE2D,
                                                 D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
            view.Texture2D = {i, static_cast<UINT>(image_desc.MipLevels) - i, 0, 0.0f};
            device->CreateShaderResourceView(images[i].Get(), &view, cpu);
            auto alias = cpu;
            alias.ptr += 3 * increment;
            device->CreateShaderResourceView(images[i].Get(), &view, alias);
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC null_view{DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_SRV_DIMENSION_TEXTURE2D,
                                                 D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING};
        null_view.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(nullptr, &null_view, cpu);
        ID3D12DescriptorHeap* binding[]{views.Get()};
        list->SetDescriptorHeaps(1, binding);
    }
    list->SetGraphicsRootSignature(root.Get());
    list->SetGraphicsRoot32BitConstants(0, entry_count, entry, 0);
    if (dynamic_heap) list->SetGraphicsRootShaderResourceView(1, source->GetGPUVirtualAddress() + index_table_offset);
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
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
        std::memcpy(words + size_t(y) * width * 4, first + size_t(y) * footprint.Footprint.RowPitch,
                    size_t(width) * 4 * sizeof(*words));
    }
    readback->Unmap(0, &none);
    std::printf("startup_entry_words words=%08x,%08x,%08x,%08x\n", words[0], words[1], words[2], words[3]);
}
