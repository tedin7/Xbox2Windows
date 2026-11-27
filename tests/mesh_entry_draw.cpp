#include "desktop_gpu.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>
using Microsoft::WRL::ComPtr;

namespace {
std::vector<char> read_bytes(const char* path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open draw input");
    return {std::istreambuf_iterator<char>(stream), {}};
}

void copy_mesh_texture(ID3D12GraphicsCommandList* list, ID3D12Resource* buffer, ID3D12Resource* texture,
                       const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint,
                       D3D12_RESOURCE_STATES read_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
{
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = buffer;
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = footprint;
    to.pResource = texture;
    list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    transition(list, texture, D3D12_RESOURCE_STATE_COPY_DEST, read_state);
}

struct ProducedTexture {
    ComPtr<ID3D12Resource> buffer, sampled;
    ComPtr<ID3D12PipelineState> pipeline;
};

template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind, typename T>
struct alignas(void*) Field {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE kind = Kind;
    T value{};
};

struct MeshPipeline {
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> formats;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT> depth_format;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
};

// An empty AS subobject is invalid, so an amplified pipeline keeps its own stream.
struct MeshAmplifiedPipeline {
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE> as;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> formats;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT> depth_format;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
};

ComPtr<ID3D12PipelineState> fallback_draw_pipeline(ID3D12Device* device, ID3D12RootSignature* root,
    D3D12_SHADER_BYTECODE vs, D3D12_SHADER_BYTECODE ps, const D3D12_DEPTH_STENCIL_DESC& depth,
    DXGI_FORMAT depth_format)
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.VS = vs;
    desc.PS = ps;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    auto& blend = desc.BlendState.RenderTarget[0];
    blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
    blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = UINT_MAX;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = desc.SampleDesc.Count = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.DepthStencilState = depth;
    desc.DSVFormat = depth_format;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline)), "Create fallback vertex pipeline");
    return pipeline;
}

ComPtr<ID3D12RootSignature> create_draw_root(ID3D12Device* device, const D3D12_ROOT_SIGNATURE_DESC& layout)
{
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&layout, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "Serialize draw root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)), "Draw root");
    return root;
}

bool check_mesh_draw(ID3D12Device* device, const char* first, const char* second, const char* fixture,
                     bool vertices, MeshConstantBinding binding, const char* fallback_vertex, MeshDrawMode mode,
                     const MeshTextureProducers* producers, const char* amplification)
{
    const bool fallback = fallback_vertex != nullptr;
    if (!fallback) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
        check(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options)), "Query mesh tier");
        if (options.MeshShaderTier < D3D12_MESH_SHADER_TIER_1) {
            std::puts("mesh_shader_tier=unsupported");
            return false;
        }
    }
    constexpr UINT side = 32;
    constexpr UINT cb_bytes = 512;
    constexpr UINT image_bytes = side * side * 4;
    constexpr UINT texel_bytes = 2 * 2 * 4;
    const auto input = read_bytes(fixture);
    const UINT draw_bytes = 2 * cb_bytes + (vertices ? 512 : 0);
    UINT trace_stride = fallback ? 128 : 80;
    if (mode == MeshDrawMode::multi_primitive) {
        // Counts, 128 float4 positions and 128 uint3 indices fit in 3,600 bytes.
        if (fallback || vertices || input.size() < draw_bytes + image_bytes + 80 ||
            input.size() > draw_bytes + image_bytes + 3600 || (input.size() - draw_bytes - image_bytes) % 16)
            throw std::runtime_error("Invalid multi-primitive draw fixture");
        trace_stride = static_cast<UINT>(input.size() - draw_bytes - image_bytes);
    }
    const bool mesh_texture = mode == MeshDrawMode::mesh_texture_load;
    const bool textured = mode == MeshDrawMode::textured || mesh_texture;
    const bool depth_pair = mode == MeshDrawMode::depth_pair;
    const auto depth_format = depth_pair ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_UNKNOWN;
    const UINT draw_count = depth_pair ? 2 : 1;
    const UINT input_bytes = draw_count * draw_bytes;
    const UINT trace_bytes = draw_count * trace_stride;
    const UINT reference_end = input_bytes + trace_bytes + draw_count * image_bytes;
    if (input.size() != reference_end + (textured && !producers ? texel_bytes : 0))
        throw std::runtime_error("Invalid draw fixture size");
    auto shader = read_bytes(first);
    auto pixel = read_bytes(second);
    D3D12_ROOT_PARAMETER params[5]{};
    D3D12_DESCRIPTOR_RANGE cbv_range{D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 2, 0, 0, 0};
    const bool uses_table = binding == MeshConstantBinding::table || binding == MeshConstantBinding::swapped_table;
    if (uses_table) {
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable = {1, &cbv_range};
    } else {
        for (UINT i = 0; i < 2; ++i) {
            params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            params[i].Descriptor.ShaderRegister = i;
        }
    }
    const UINT trace_slot = uses_table ? 1 : 2;
    const UINT vertex_slot = trace_slot + 1;
    params[trace_slot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC layout{};
    params[vertex_slot].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    const bool separate_roots = fallback && textured;
    const UINT input_slots = vertex_slot + vertices;
    const UINT geometry_slot = separate_roots ? trace_slot : input_slots;
    const UINT texture_slot = separate_roots ? vertex_slot : vertex_slot + 1;
    layout.NumParameters = input_slots + fallback;
    layout.pParameters = params;
    ComPtr<ID3D12RootSignature> export_root;
    if (separate_roots) {
        // Compute reads a buffer at t0. Graphics keeps the CBVs and uses t1 for
        // the exported geometry and t0 for the sampled texture.
        layout.NumParameters = input_slots;
        export_root = create_draw_root(device, layout);
    }
    if (fallback) {
        auto& geometry_param = params[geometry_slot];
        geometry_param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        geometry_param.Descriptor.ShaderRegister = 1;
        geometry_param.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    }
    D3D12_DESCRIPTOR_RANGE texture_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_STATIC_SAMPLER_DESC sampler{};
    if (textured) {
        // On the native path, mesh and pixel each use t0 with stage-specific visibility.
        if (!fallback) params[vertex_slot].ShaderVisibility = D3D12_SHADER_VISIBILITY_MESH;
        auto& texture_param = params[texture_slot];
        texture_param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        texture_param.DescriptorTable = {1, &texture_range};
        if (mesh_texture) texture_range.BaseShaderRegister = 1;
        texture_param.ShaderVisibility = mesh_texture ? D3D12_SHADER_VISIBILITY_MESH : D3D12_SHADER_VISIBILITY_PIXEL;
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MaxAnisotropy = 1;
        sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        layout.NumParameters = texture_slot + 1;
        layout.NumStaticSamplers = 1;
        layout.pStaticSamplers = &sampler;
    }
    auto root = create_draw_root(device, layout);
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12PipelineState> compute_pipeline;
    const D3D12_SHADER_BYTECODE ps{pixel.data(), pixel.size()};
    D3D12_DEPTH_STENCIL_DESC depth_state{};
    if (depth_pair) {
        depth_state.DepthEnable = TRUE;
        depth_state.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        depth_state.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    }
    if (fallback) {
        if (!export_root) export_root = root;
        const auto vertex = read_bytes(fallback_vertex);
        pipeline = fallback_draw_pipeline(device, root.Get(), {vertex.data(), vertex.size()}, ps,
                                          depth_state, depth_format);
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = export_root.Get();
        desc.CS = {shader.data(), shader.size()};
        check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&compute_pipeline)), "Create mesh export compute pipeline");
    } else {
        ComPtr<ID3D12Device2> device2;
        check(device->QueryInterface(IID_PPV_ARGS(&device2)), "Query mesh pipeline device");
        // A source-built AS dispatches the mesh groups and passes the payload this
        // fixture reads. Mesh-only pipelines keep the AS subobject out of their stream.
        const auto amplification_shader = amplification ? read_bytes(amplification) : std::vector<char>{};
        // tools/mesh_od_pipeline.py reads this line to separate pipeline creation from the process time.
        const auto pipeline_start = std::chrono::steady_clock::now();
        HRESULT created;
        if (amplification) {
            MeshAmplifiedPipeline stream;
            stream.root.value = root.Get();
            stream.as.value = {amplification_shader.data(), amplification_shader.size()};
            stream.ms.value = {shader.data(), shader.size()};
            stream.ps.value = ps;
            stream.formats.value.NumRenderTargets = 1;
            stream.formats.value.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            stream.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
            stream.raster.value.CullMode = D3D12_CULL_MODE_NONE;
            stream.raster.value.DepthClipEnable = TRUE;
            stream.depth.value = depth_state;
            stream.depth_format.value = depth_format;
            stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
            created = device2->CreatePipelineState(&desc, IID_PPV_ARGS(&pipeline));
        } else {
            MeshPipeline stream;
            stream.root.value = root.Get();
            stream.ms.value = {shader.data(), shader.size()};
            stream.ps.value = ps;
            stream.formats.value.NumRenderTargets = 1;
            stream.formats.value.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            stream.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
            stream.raster.value.CullMode = D3D12_CULL_MODE_NONE;
            stream.raster.value.DepthClipEnable = TRUE;
            stream.depth.value = depth_state;
            stream.depth_format.value = depth_format;
            stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
            created = device2->CreatePipelineState(&desc, IID_PPV_ARGS(&pipeline));
        }
        const std::chrono::duration<double, std::milli> pipeline_time = std::chrono::steady_clock::now() - pipeline_start;
        std::printf("mesh_pipeline_ms=%.3f\n", pipeline_time.count());
        std::fflush(stdout);  // Keep the line if the process dies after pipeline creation.
        check(created, "Create mesh pipeline");
    }
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = texture.Height = side;
    texture.DepthOrArraySize = texture.MipLevels = texture.SampleDesc.Count = 1;
    texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    constexpr auto draw_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear_value{};
    clear_value.Format = texture.Format;
    clear_value.Color[3] = 1;
    auto target = resource(device, texture, D3D12_HEAP_TYPE_DEFAULT, draw_state, &clear_value);
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heap_desc.NumDescriptors = 1;
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "Create target heap");
    auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, handle);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 size = 0;
    device->GetCopyableFootprints(&texture, 0, 1, 0, &footprint, nullptr, nullptr, &size);
    auto readback = make_buffer(device, size + trace_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    auto upload = make_buffer(device, input_bytes + trace_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    D3D12_RANGE empty{};
    check(upload->Map(0, &empty, &mapped), "Map draw input");
    std::memcpy(mapped, input.data(), input_bytes);
    std::memset(static_cast<char*>(mapped) + input_bytes, 0xcd, trace_bytes);
    upload->Unmap(0, nullptr);
    ComputeCommands commands(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();
    list->SetGraphicsRootSignature(root.Get());
    const auto base = upload->GetGPUVirtualAddress();
    std::array<ComPtr<ID3D12Resource>, 2> constants;
    ComPtr<ID3D12DescriptorHeap> cbv_heap;
    ComPtr<ID3D12Resource> buffer_b;
    if (uses_table) {
        D3D12_DESCRIPTOR_HEAP_DESC cbv_heap_desc{};
        cbv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        cbv_heap_desc.NumDescriptors = static_cast<UINT>(constants.size());
        cbv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        check(device->CreateDescriptorHeap(&cbv_heap_desc, IID_PPV_ARGS(&cbv_heap)), "Create mesh CBV heap");
        const UINT increment = device->GetDescriptorHandleIncrementSize(cbv_heap_desc.Type);
        const auto first_slot = cbv_heap->GetCPUDescriptorHandleForHeapStart();
        for (UINT i = 0; i < constants.size(); ++i) {
            constants[i] = make_buffer(device, cb_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            check(constants[i]->Map(0, &empty, &mapped), "Map mesh CBV");
            std::memcpy(mapped, input.data() + i * cb_bytes, cb_bytes);
            constants[i]->Unmap(0, nullptr);
            const UINT slot = binding == MeshConstantBinding::swapped_table ? 1 - i : i;
            const D3D12_CONSTANT_BUFFER_VIEW_DESC view{constants[i]->GetGPUVirtualAddress(), cb_bytes};
            device->CreateConstantBufferView(&view, {first_slot.ptr + slot * increment});
        }
        ID3D12DescriptorHeap* heaps[]{cbv_heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetGraphicsRootDescriptorTable(0, cbv_heap->GetGPUDescriptorHandleForHeapStart());
    } else {
        D3D12_GPU_VIRTUAL_ADDRESS cbv[2]{base, base + cb_bytes};
        if (binding != MeshConstantBinding::root) {
            // Keep A and B in distinct resources with identical contents across binding orders.
            buffer_b = make_buffer(device, cb_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            check(buffer_b->Map(0, &empty, &mapped), "Map constant buffer B");
            std::memcpy(mapped, input.data() + cb_bytes, cb_bytes);
            buffer_b->Unmap(0, nullptr);
            cbv[1] = buffer_b->GetGPUVirtualAddress();
            if (binding == MeshConstantBinding::swapped_root_pair) std::swap(cbv[0], cbv[1]);
        }
        list->SetGraphicsRootConstantBufferView(0, cbv[0]);
        list->SetGraphicsRootConstantBufferView(1, cbv[1]);
    }
    if (vertices && !fallback) list->SetGraphicsRootShaderResourceView(vertex_slot, base + 2 * cb_bytes);
    ComPtr<ID3D12Resource> sampled, texture_upload;
    ComPtr<ID3D12DescriptorHeap> srv_heap;
    ComPtr<ID3D12RootSignature> compute_root;
    std::array<ProducedTexture, 2> produced;
    if (textured) {
        auto sampled_desc = texture;
        sampled_desc.Width = sampled_desc.Height = 2;
        sampled_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT sampled_layout{};
        UINT64 upload_bytes = 0;
        device->GetCopyableFootprints(&sampled_desc, 0, 1, 0, &sampled_layout, nullptr, nullptr, &upload_bytes);
        if (producers) {
            if (sampled_layout.Offset != 0 || sampled_layout.Footprint.RowPitch != D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
                throw std::runtime_error("Unexpected footprint for the compute texture fixture");
            D3D12_ROOT_PARAMETER output{};
            output.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            D3D12_ROOT_SIGNATURE_DESC compute_layout{};
            compute_layout.NumParameters = 1;
            compute_layout.pParameters = &output;
            ComPtr<ID3DBlob> compute_blob;
            check(D3D12SerializeRootSignature(&compute_layout, D3D_ROOT_SIGNATURE_VERSION_1,
                                             &compute_blob, nullptr), "Serialize texture producer root");
            check(device->CreateRootSignature(0, compute_blob->GetBufferPointer(), compute_blob->GetBufferSize(),
                                              IID_PPV_ARGS(&compute_root)), "Create texture producer root");
            list->SetComputeRootSignature(compute_root.Get());
            const char* paths[]{producers->first, producers->second};
            const unsigned count = producers->second ? 2 : 1;
            for (unsigned i = 0; i < count; ++i) {
                auto& producer = produced[i];
                const auto bytes = read_bytes(paths[i]);
                D3D12_COMPUTE_PIPELINE_STATE_DESC compute{};
                compute.pRootSignature = compute_root.Get();
                compute.CS = {bytes.data(), bytes.size()};
                check(device->CreateComputePipelineState(&compute, IID_PPV_ARGS(&producer.pipeline)),
                      "Create texture producer pipeline");
                producer.buffer = make_buffer(device, upload_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                              D3D12_RESOURCE_STATE_COMMON,
                                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
                producer.sampled = resource(device, sampled_desc, D3D12_HEAP_TYPE_DEFAULT,
                                            D3D12_RESOURCE_STATE_COPY_DEST);
                transition(list, producer.buffer.Get(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                list->SetPipelineState(producer.pipeline.Get());
                list->SetComputeRootUnorderedAccessView(0, producer.buffer->GetGPUVirtualAddress());
                list->Dispatch(1, 1, 1);
                transition(list, producer.buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COPY_SOURCE);
                copy_mesh_texture(list, producer.buffer.Get(), producer.sampled.Get(), sampled_layout);
            }
            const unsigned selected = producers->bind_first ? 0 : count - 1;
            sampled = produced[selected].sampled;
            list->SetPipelineState(pipeline.Get());
            std::printf("texture_producers=%u bound_texture=%u\n", count, selected);
        } else {
            sampled = resource(device, sampled_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
            texture_upload = make_buffer(device, upload_bytes, D3D12_HEAP_TYPE_UPLOAD,
                                         D3D12_RESOURCE_STATE_GENERIC_READ);
            check(texture_upload->Map(0, &empty, &mapped), "Map mesh texture upload");
            for (UINT y = 0; y < 2; ++y)
                std::memcpy(static_cast<char*>(mapped) + sampled_layout.Offset + y * sampled_layout.Footprint.RowPitch,
                            input.data() + reference_end + y * 8, 8);
            texture_upload->Unmap(0, nullptr);
            copy_mesh_texture(list, texture_upload.Get(), sampled.Get(), sampled_layout,
                              mesh_texture ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                           : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }
        D3D12_DESCRIPTOR_HEAP_DESC srv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        check(device->CreateDescriptorHeap(&srv_desc, IID_PPV_ARGS(&srv_heap)), "Create mesh texture SRV heap");
        device->CreateShaderResourceView(sampled.Get(), nullptr, srv_heap->GetCPUDescriptorHandleForHeapStart());
        ID3D12DescriptorHeap* heaps[]{srv_heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetGraphicsRootDescriptorTable(texture_slot, srv_heap->GetGPUDescriptorHandleForHeapStart());
    }
    const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
    const D3D12_RECT scissor{0, 0, side, side};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    auto trace = make_buffer(device, trace_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    list->CopyBufferRegion(trace.Get(), 0, upload.Get(), input_bytes, trace_bytes);
    transition(list, trace.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto trace_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if (fallback) {
        list->SetPipelineState(compute_pipeline.Get());
        list->SetComputeRootSignature(export_root.Get());
        // The dispatches write disjoint trace ranges. Transition once before either draw reads them.
        for (UINT i = 0; i < draw_count; ++i) {
            const auto draw_base = base + i * draw_bytes;
            list->SetComputeRootConstantBufferView(0, draw_base);
            list->SetComputeRootConstantBufferView(1, draw_base + cb_bytes);
            if (vertices) list->SetComputeRootShaderResourceView(vertex_slot, draw_base + 2 * cb_bytes);
            list->SetComputeRootUnorderedAccessView(trace_slot, trace->GetGPUVirtualAddress() + i * trace_stride);
            list->Dispatch(1, 1, 1);
        }
        trace_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        transition(list, trace.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, trace_state);
        list->SetPipelineState(pipeline.Get());
    } else {
        list->SetGraphicsRootUnorderedAccessView(trace_slot, trace->GetGPUVirtualAddress());
    }
    ComPtr<ID3D12Resource> depth, depth_readback;
    ComPtr<ID3D12DescriptorHeap> depth_heap;
    D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT depth_footprint{};
    if (depth_pair) {
        auto depth_desc = texture;
        depth_desc.Format = depth_format;
        depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        D3D12_CLEAR_VALUE clear_depth{};
        clear_depth.Format = depth_format;
        clear_depth.DepthStencil.Depth = 1.0f;
        depth = resource(device, depth_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear_depth);
        D3D12_DESCRIPTOR_HEAP_DESC depth_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
        check(device->CreateDescriptorHeap(&depth_heap_desc, IID_PPV_ARGS(&depth_heap)), "Create mesh depth heap");
        dsv = depth_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateDepthStencilView(depth.Get(), nullptr, dsv);
        list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        UINT64 depth_bytes = 0;
        device->GetCopyableFootprints(&depth_desc, 0, 1, 0, &depth_footprint, nullptr, nullptr, &depth_bytes);
        depth_readback = make_buffer(device, depth_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    list->OMSetRenderTargets(1, &handle, FALSE, depth_pair ? &dsv : nullptr);
    const float clear[4]{0, 0, 0, 1};
    list->ClearRenderTargetView(handle, clear, 0, nullptr);
    if (fallback) {
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        for (UINT i = 0; i < draw_count; ++i) {
            list->SetGraphicsRootShaderResourceView(geometry_slot, trace->GetGPUVirtualAddress() + i * trace_stride);
            list->DrawInstanced(3, 1, 0, 0);
        }
    } else {
        ComPtr<ID3D12GraphicsCommandList6> mesh_list;
        check(commands.list.As(&mesh_list), "Query mesh dispatch list");
        mesh_list->DispatchMesh(1, 1, 1);
        if (depth_pair) {
            const auto second_base = base + draw_bytes;
            list->SetGraphicsRootConstantBufferView(0, second_base);
            list->SetGraphicsRootConstantBufferView(1, second_base + cb_bytes);
            list->SetGraphicsRootShaderResourceView(vertex_slot, second_base + 2 * cb_bytes);
            list->SetGraphicsRootUnorderedAccessView(trace_slot, trace->GetGPUVirtualAddress() + trace_stride);
            mesh_list->DispatchMesh(1, 1, 1);
        }
    }
    transition(list, trace.Get(), trace_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), size, trace.Get(), 0, trace_bytes);
    transition(list, target.Get(), draw_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = target.Get();
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (depth_pair) {
        transition(list, depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        src.pResource = depth.Get();
        dst.pResource = depth_readback.Get();
        dst.PlacedFootprint = depth_footprint;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    commands.submit_and_wait(device);
    check(readback->Map(0, nullptr, &mapped), "Map draw result");
    const char* rows = static_cast<const char*>(mapped);
    const char* reference = input.data() + input_bytes;
    const auto* trace_end = reference + trace_bytes;
    const auto* first_bad = std::mismatch(reference, trace_end, rows + size).first;
    const bool trace_ok = first_bad == trace_end;
    if (fallback || depth_pair || !trace_ok) {
        std::printf("mesh_exports bytes=%u matched=%d\n", trace_bytes, trace_ok ? 1 : 0);
        if (!trace_ok) std::printf("first_mesh_export_mismatch_byte=%td\n", first_bad - reference);
    }
    reference = trace_end;
    unsigned mismatches = 0;
    for (UINT y = 0; y < side; ++y)
        for (UINT x = 0; x < side; ++x)
            mismatches += std::memcmp(rows + footprint.Offset + y * footprint.Footprint.RowPitch + 4*x,
                                      reference + (y * side + x) * 4, 4) != 0;
    readback->Unmap(0, &empty);
    std::printf("pixels=1024 mismatches=%u\n", mismatches);
    unsigned depth_mismatches = 0;
    if (depth_pair) {
        check(depth_readback->Map(0, nullptr, &mapped), "Map mesh depth result");
        rows = static_cast<const char*>(mapped);
        const char* expected_depth = reference + image_bytes;
        for (UINT y = 0; y < side; ++y)
            for (UINT x = 0; x < side; ++x)
                depth_mismatches += std::memcmp(
                    rows + depth_footprint.Offset + y * depth_footprint.Footprint.RowPitch + 4 * x,
                    expected_depth + (y * side + x) * 4, 4) != 0;
        depth_readback->Unmap(0, &empty);
        std::printf("depth_pixels=1024 depth_mismatches=%u\n", depth_mismatches);
    }
    if (!trace_ok) throw std::runtime_error("Mesh export trace mismatch");
    if (mismatches) throw std::runtime_error("Draw reference mismatch");
    if (depth_mismatches) throw std::runtime_error("Depth reference mismatch");
    return true;
}
}

bool check_mesh_entry_draw(ID3D12Device* device, const char* mesh, const char* pixel, const char* fixture,
                           bool vertices, MeshConstantBinding binding, MeshDrawMode mode,
                           const MeshTextureProducers* producers, const char* amplification)
{
    return check_mesh_draw(device, mesh, pixel, fixture, vertices, binding, nullptr, mode, producers,
                           amplification);
}

bool check_mesh_amplified_draw(ID3D12Device* device, const char* amplification, const char* mesh,
                               const char* pixel, const char* fixture)
{
    return check_mesh_entry_draw(device, mesh, pixel, fixture, false, MeshConstantBinding::root,
                                 MeshDrawMode::color, nullptr, amplification);
}

void check_mesh_fallback_draw(ID3D12Device* device, const char* compute, const char* vertex,
                              const char* pixel, const char* fixture, MeshDrawMode mode,
                              const MeshTextureProducers* producers)
{
    check_mesh_draw(device, compute, pixel, fixture, true, MeshConstantBinding::root, vertex, mode, producers,
                    nullptr);
}
