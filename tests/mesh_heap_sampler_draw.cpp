// Desktop readback for tools/mesh_dynamic_aot.py's sampler heap path (#1998 item E).
//
// The translated mesh shader reads one geometry word from t0, samples meshTexture : register(t1)
// through qh[n] : register(s0, space1) and exports the sample as TEXCOORD0. The pixel shader writes
// that attribute. The fixture file holds the geometry word, the title's sampler heap entries
// (16-byte words, decoded below), the texture's two mip levels and the expected RGBA8 color.
//
// The sampler table is built by the stand-in's helpers from translated_sampler_bindings.h: entries
// past the fixture's count use a transparent-black border, and entry 2047 is the reserved fallback.
#include "desktop_gpu.h"
#include "startup_stubs/translated_sampler_bindings.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT side = 32;
constexpr UINT mip_widths[2] = {4, 2};

std::vector<char> read_file(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open mesh heap sampler input");
    return {std::istreambuf_iterator<char>(file), {}};
}

template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind, typename T>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE kind = Kind;
    T value{};
};

struct Stream {
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> formats;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
    Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
};

// The stand-in's decoder, LOD clamp included: dynamic_sampler() applies MIN_LOD (SQ_IMG_SAMP_WORD1 bits 11:0)
// and MAX_LOD (bits 23:12), both unsigned 4.8 fixed point (RDNA2 ISA, image sampler descriptor).
bool fixture_sampler(const x2w_startup::SamplerWords& words, D3D12_SAMPLER_DESC& out)
{
    out = x2w_startup::dynamic_sampler(words);
    return true;
}
}  // namespace

bool check_mesh_heap_sampler_draw(ID3D12Device* device, const char* mesh_path, const char* pixel_path,
                                  const char* fixture_path)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
    check(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options)), "Query mesh tier");
    if (options.MeshShaderTier < D3D12_MESH_SHADER_TIER_1) {
        std::puts("mesh_shader_tier=unsupported");
        return false;
    }
    const auto mesh = read_file(mesh_path);
    const auto pixel = read_file(pixel_path);
    const auto input = read_file(fixture_path);
    std::uint32_t geometry_word = 0, entry_count = 0;
    if (input.size() < 8) throw std::runtime_error("Invalid mesh heap sampler fixture");
    std::memcpy(&geometry_word, input.data(), 4);
    std::memcpy(&entry_count, input.data() + 4, 4);
    const std::size_t texel_bytes = 4 * (mip_widths[0] + mip_widths[1]);
    if (entry_count == 0 || entry_count > x2w_startup::translated_sampler_fallback ||
        input.size() != 8 + 16 * std::size_t{entry_count} + texel_bytes + 4)
        throw std::runtime_error("Invalid mesh heap sampler fixture size");
    std::vector<x2w_startup::SamplerWords> entries(entry_count);
    std::memcpy(entries.data(), input.data() + 8, 16 * std::size_t{entry_count});
    const char* texels = input.data() + 8 + 16 * std::size_t{entry_count};
    std::uint32_t expected = 0;
    std::memcpy(&expected, texels + texel_bytes, 4);

    // t0: geometry words (root SRV), t1: texture (table), s0-s2047 space1: sampler table.
    const D3D12_DESCRIPTOR_RANGE texture_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 0, 0};
    const D3D12_DESCRIPTOR_RANGE sampler_range{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
                                               x2w_startup::translated_sampler_count, 0,
                                               x2w_startup::translated_sampler_space, 0};
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_MESH;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = {1, &texture_range};
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_MESH;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable = {1, &sampler_range};
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_MESH;
    const D3D12_ROOT_SIGNATURE_DESC layout{3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&layout, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr),
          "Serialize mesh heap sampler root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)),
          "Create mesh heap sampler root");

    Stream stream;
    stream.root.value = root.Get();
    stream.ms.value = {mesh.data(), mesh.size()};
    stream.ps.value = {pixel.data(), pixel.size()};
    stream.formats.value.NumRenderTargets = 1;
    stream.formats.value.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    stream.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
    stream.raster.value.CullMode = D3D12_CULL_MODE_NONE;
    stream.raster.value.DepthClipEnable = TRUE;
    stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    ComPtr<ID3D12Device2> device2;
    check(device->QueryInterface(IID_PPV_ARGS(&device2)), "Query mesh pipeline device");
    const D3D12_PIPELINE_STATE_STREAM_DESC stream_desc{sizeof(stream), &stream};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device2->CreatePipelineState(&stream_desc, IID_PPV_ARGS(&pipeline)), "Create mesh heap sampler pipeline");

    // A 4x1 RGBA8 texture with a 2x1 second level.
    D3D12_RESOURCE_DESC texture_desc{};
    texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture_desc.Width = mip_widths[0];
    texture_desc.Height = 1;
    texture_desc.DepthOrArraySize = 1;
    texture_desc.MipLevels = 2;
    texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture_desc.SampleDesc.Count = 1;
    const auto texture = resource(device, texture_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT mips[2]{};
    UINT64 upload_bytes = 0;
    device->GetCopyableFootprints(&texture_desc, 0, 2, 0, mips, nullptr, nullptr, &upload_bytes);
    const auto upload = make_buffer(device, upload_bytes + 256, D3D12_HEAP_TYPE_UPLOAD,
                                    D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    const D3D12_RANGE none{0, 0};
    check(upload->Map(0, &none, &mapped), "Map mesh heap sampler upload");
    auto* bytes = static_cast<char*>(mapped);
    std::memcpy(bytes + mips[0].Offset, texels, 4 * mip_widths[0]);
    std::memcpy(bytes + mips[1].Offset, texels + 4 * mip_widths[0], 4 * mip_widths[1]);
    std::memcpy(bytes + upload_bytes, &geometry_word, 4);  // t0 starts on a 256-byte boundary
    upload->Unmap(0, nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC view_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                         D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> views;
    check(device->CreateDescriptorHeap(&view_desc, IID_PPV_ARGS(&views)), "Create mesh texture view heap");
    device->CreateShaderResourceView(texture.Get(), nullptr, views->GetCPUDescriptorHandleForHeapStart());

    const auto snapshot = x2w_startup::snapshot_samplers(
        {reinterpret_cast<std::uintptr_t>(entries.data()), entry_count}, true);
    ComPtr<ID3D12DescriptorHeap> samplers;
    x2w_startup::NullSamplerCounts counts;
    check(x2w_startup::create_sampler_snapshot_heap(device, snapshot, fixture_sampler, samplers.GetAddressOf(),
                                                    counts),
          "Create mesh sampler table");
    if (counts.unreadable || counts.invalid_token ||
        counts.out_of_range != x2w_startup::translated_sampler_fallback - entry_count)
        throw std::runtime_error("Unexpected mesh sampler table counts");

    D3D12_RESOURCE_DESC target_desc = texture_desc;
    target_desc.Width = target_desc.Height = side;
    target_desc.MipLevels = 1;
    target_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 1}};
    const auto target = resource(device, target_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                 &clear);
    D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvs;
    check(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtvs)), "Create mesh heap sampler RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT target_footprint{};
    UINT64 readback_bytes = 0;
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, &target_footprint, nullptr, nullptr, &readback_bytes);
    const auto readback = make_buffer(device, readback_bytes, D3D12_HEAP_TYPE_READBACK,
                                      D3D12_RESOURCE_STATE_COPY_DEST);

    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    for (UINT level = 0; level < 2; ++level) {
        D3D12_TEXTURE_COPY_LOCATION to{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        to.SubresourceIndex = level;
        D3D12_TEXTURE_COPY_LOCATION from{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        from.PlacedFootprint = mips[level];
        list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
    transition(list, texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
    list->SetGraphicsRootSignature(root.Get());
    x2w_startup::bind_translated_heaps(list, views.Get(), samplers.Get());
    list->SetGraphicsRootShaderResourceView(0, upload->GetGPUVirtualAddress() + upload_bytes);
    list->SetGraphicsRootDescriptorTable(1, views->GetGPUDescriptorHandleForHeapStart());
    list->SetGraphicsRootDescriptorTable(2, samplers->GetGPUDescriptorHandleForHeapStart());
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
    const D3D12_RECT scissor{0, 0, side, side};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    ComPtr<ID3D12GraphicsCommandList6> mesh_list;
    check(commands.list.As(&mesh_list), "Query mesh dispatch list");
    mesh_list->DispatchMesh(1, 1, 1);
    transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION out{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    out.PlacedFootprint = target_footprint;
    const D3D12_TEXTURE_COPY_LOCATION in{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&out, 0, 0, 0, &in, nullptr);
    commands.submit_and_wait(device);

    check(readback->Map(0, nullptr, &mapped), "Map mesh heap sampler readback");
    const auto* rows = static_cast<const char*>(mapped);
    unsigned mismatches = 0;
    std::uint32_t first = 0;
    std::memcpy(&first, rows + target_footprint.Offset, 4);
    for (UINT y = 0; y < side; ++y)
        for (UINT x = 0; x < side; ++x) {
            std::uint32_t color = 0;
            std::memcpy(&color, rows + target_footprint.Offset + y * target_footprint.Footprint.RowPitch + 4 * x, 4);
            mismatches += color != expected;
        }
    readback->Unmap(0, &none);
    std::printf("mesh_heap_sampler word=%08x entries=%u expected=%08x first=%08x pixels=%u mismatches=%u\n",
                geometry_word, entry_count, expected, first, side * side, mismatches);
    if (mismatches) throw std::runtime_error("Mesh heap sampler readback mismatch");
    return true;
}
