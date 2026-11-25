#pragma once

#include "xmem_mesh_root_srv_mesh.h"
#include "xmem_mesh_root_srv_vertex.h"
#include "xmem_mesh_root_srv_pixel.h"

namespace {

template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind, class T>
struct alignas(void*) MeshSubobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE kind = Kind;
    T value{};
};

struct LeaseGraphicsStream {
    MeshSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
    MeshSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    MeshSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> formats;
    MeshSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
    MeshSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
};

template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind>
struct LeaseGraphicsPipeline {
    LeaseGraphicsStream graphics;
    MeshSubobject<Kind, D3D12_SHADER_BYTECODE> shader;
};

void check_leased_root_srv(ID3D12Device3* device, ID3D12Resource* alias, unsigned char* cpu,
                         const X2WXMemSectionLease& lease, bool mesh)
{
    constexpr UINT64 page = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    constexpr UINT width = 4, height = 2;
    constexpr UINT before = 0x12345678, written = 0x9abcdef0, wrong = 0x1372ba5d;
    const UINT64 offset = lease.first_byte;
    const UINT64 wrong_offset = offset + sizeof(UINT) <= page ? page : 0;
    const auto base = alias->GetGPUVirtualAddress();
    require(base && !(offset & 3) && offset + sizeof(UINT) <= lease.window_bytes &&
            offset / page != wrong_offset / page && wrong_offset + sizeof(UINT) <= lease.window_bytes,
            "two distinct aligned root SRV ranges in the leased window");

    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameter.ShaderVisibility = mesh ? D3D12_SHADER_VISIBILITY_MESH : D3D12_SHADER_VISIBILITY_VERTEX;
    const D3D12_ROOT_SIGNATURE_DESC layout{1, &parameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> root_blob;
    check(D3D12SerializeRootSignature(&layout, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, nullptr),
          "serialize mesh SRV root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(),
                                      IID_PPV_ARGS(&root)), "create mesh SRV root");
    LeaseGraphicsStream graphics;
    graphics.root.value = root.Get();
    graphics.ps.value = {xmem_mesh_root_srv_pixel, sizeof(xmem_mesh_root_srv_pixel)};
    graphics.formats.value.NumRenderTargets = 1;
    graphics.formats.value.RTFormats[0] = DXGI_FORMAT_R32_UINT;
    graphics.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
    graphics.raster.value.CullMode = D3D12_CULL_MODE_NONE;
    graphics.raster.value.DepthClipEnable = TRUE;
    graphics.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    LeaseGraphicsPipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS> mesh_stream{
        graphics, {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, {xmem_mesh_root_srv_mesh, sizeof(xmem_mesh_root_srv_mesh)}}};
    LeaseGraphicsPipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS> vertex_stream{
        graphics, {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, {xmem_mesh_root_srv_vertex, sizeof(xmem_mesh_root_srv_vertex)}}};
    const D3D12_PIPELINE_STATE_STREAM_DESC pipeline_desc = mesh
        ? D3D12_PIPELINE_STATE_STREAM_DESC{sizeof(mesh_stream), &mesh_stream}
        : D3D12_PIPELINE_STATE_STREAM_DESC{sizeof(vertex_stream), &vertex_stream};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreatePipelineState(&pipeline_desc, IID_PPV_ARGS(&pipeline)), "create leased SRV pipeline");

    D3D12_RESOURCE_DESC target_desc{};
    target_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    target_desc.Width = width;
    target_desc.Height = height;
    target_desc.DepthOrArraySize = target_desc.MipLevels = target_desc.SampleDesc.Count = 1;
    target_desc.Format = DXGI_FORMAT_R32_UINT;
    target_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear_value{};
    clear_value.Format = target_desc.Format;
    const auto target = resource(device, target_desc, D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_STATE_RENDER_TARGET, &clear_value);
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    ComPtr<ID3D12DescriptorHeap> rtvs;
    check(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtvs)), "create mesh SRV RTV heap");
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 size = 0;
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, &footprint, nullptr, nullptr, &size);
    const auto readback = make_buffer(device, size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const auto upload = make_buffer(device, sizeof(UINT), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* data = nullptr;
    const D3D12_RANGE empty{};
    check(upload->Map(0, &empty, &data), "map mesh producer upload");
    std::memcpy(data, &written, sizeof(written));
    upload->Unmap(0, nullptr);

    const auto fence = [device] {
        ComPtr<ID3D12Fence> result;
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&result)), "create mesh handoff fence");
        return result;
    };
    enum class Scenario { ordered, missing_wait, wrong_page };
    for (const auto scenario : {Scenario::ordered, Scenario::missing_wait, Scenario::wrong_page}) {
        const bool wait_for_writer = scenario != Scenario::missing_wait;
        const bool wrong_page = scenario == Scenario::wrong_page;
        const UINT64 bound_offset = wrong_page ? wrong_offset : offset;
        const UINT expected = wrong_page ? wrong : wait_for_writer ? written : before;
        const char* label = wrong_page ? "wrong page" : wait_for_writer ? "ordered" : "missing wait";
        // Separate the three runs. Both queues from the preceding run have completed.
        std::memcpy(cpu + offset, &before, sizeof(before));
        std::memcpy(cpu + wrong_offset, &wrong, sizeof(wrong));
        ComputeCommands writer(device, nullptr, D3D12_COMMAND_LIST_TYPE_COPY);
        writer.list->CopyBufferRegion(alias, offset, upload.Get(), 0, sizeof(written));
        check(writer.list->Close(), "close GPU producer");

        ComputeCommands reader(device, pipeline.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
        auto* draw = reader.list.Get();
        draw->SetGraphicsRootSignature(root.Get());
        draw->SetGraphicsRootShaderResourceView(0, base + bound_offset);
        const D3D12_VIEWPORT viewport{0, 0, width, height, 0, 1};
        const D3D12_RECT scissor{0, 0, width, height};
        const FLOAT clear[4]{};
        draw->RSSetViewports(1, &viewport);
        draw->RSSetScissorRects(1, &scissor);
        draw->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        draw->ClearRenderTargetView(rtv, clear, 0, nullptr);
        if (mesh) {
            ComPtr<ID3D12GraphicsCommandList6> mesh_list;
            check(reader.list.As(&mesh_list), "query mesh command list");
            mesh_list->DispatchMesh(1, 1, 1);
        } else {
            draw->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            draw->DrawInstanced(3, 1, 0, 0);
        }
        transition(draw, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION dst{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = footprint;
        draw->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        transition(draw, target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        check(draw->Close(), "close mesh consumer");

        const auto gate = fence(), produced = fence(), consumed = fence();
        // An exception must also release the producer gate before COM teardown.
        struct OpenGate {
            ID3D12Fence* fence;
            ~OpenGate() { if (fence) fence->Signal(1); }
        } open_gate{gate.Get()};
        check(writer.queue->Wait(gate.Get(), 1), "hold producer behind gate");
        ID3D12CommandList* producer_list = writer.list.Get();
        writer.queue->ExecuteCommandLists(1, &producer_list);
        check(writer.queue->Signal(produced.Get(), 1), "queue writer completion");
        if (wait_for_writer) check(reader.queue->Wait(produced.Get(), 1), "wait for writer on mesh queue");
        ID3D12CommandList* consumer_list = draw;
        reader.queue->ExecuteCommandLists(1, &consumer_list);
        check(reader.queue->Signal(consumed.Get(), 1), "queue mesh completion");

        // Without the dependency, wait for the reader while the writer is still gated.
        // This makes the stale-data control independent of GPU scheduling speed.
        if (!wait_for_writer)
            check(consumed->SetEventOnCompletion(1, nullptr), "finish mesh read before releasing writer");
        check(gate->Signal(1), "release producer");
        open_gate.fence = nullptr;
        check(produced->SetEventOnCompletion(1, nullptr), "wait for producer");
        check(consumed->SetEventOnCompletion(1, nullptr), "wait for mesh readback");
        const D3D12_RANGE pixels{0, static_cast<SIZE_T>(size)};
        check(readback->Map(0, &pixels, &data), "map mesh pixel readback");
        for (UINT y = 0; y < height; ++y) {
            const auto* row = static_cast<const unsigned char*>(data) + y * footprint.Footprint.RowPitch;
            for (UINT x = 0; x < width; ++x) {
                UINT actual;
                std::memcpy(&actual, row + x * sizeof(actual), sizeof(actual));
                if (actual != expected) {
                    std::fprintf(stderr, "%s pixel (%u,%u): %08x != %08x\n", label, x, y, actual, expected);
                    readback->Unmap(0, &empty);
                    throw std::runtime_error("leased root SRV readback mismatch");
                }
            }
        }
        readback->Unmap(0, &empty);
        std::printf("%s: lease %llu, producer section page +0x%llx byte +0x%llx, "
                    "root section page +0x%llx byte +0x%llx, writer wait %u, 8 pixels %08x\n",
                    label, lease.id, lease.section_offset + offset / page * page, offset % page,
                    lease.section_offset + bound_offset / page * page, bound_offset % page,
                    static_cast<unsigned>(wait_for_writer), expected);
    }
}

}  // namespace
