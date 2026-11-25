// Argument conversion for an indirect Draw with the console-only rectangle-list topology (0x11).
//
// A rectangle list names three vertices per rectangle, so the title's GPU-written VertexCountPerInstance is
// three times the rectangle count. The desktop draw covers each rectangle with two triangles (six vertices),
// so the count has to become rectangles * 6 on the GPU, after the title's producer has written it. A
// one-thread compute pass reads the title's four argument words and writes the converted four words into a
// scratch buffer that the desktop ExecuteIndirect then reads. The title's own argument buffer is left as it is.
//
// The conversion mirrors the direct draw (`count / 3 * 6`, a trailing partial rectangle is dropped). A count
// whose converted value would not fit 32 bits becomes 0 vertices (nothing is drawn) instead of wrapping.
// Python model: tools/rectlist_topology.py convert_indirect_draw_arguments.
#pragma once

#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <memory>
#include <mutex>
#include <vector>

namespace x2w_startup {

// The scratch buffer holds the copied title words at byte 16 and receives the converted words at byte 0. The pass
// never reads the title's buffer: that buffer is a small placed resource read at an arbitrary 4-byte offset, and a
// root SRV there drew nothing on an Intel Gen9 (cause not isolated; a copy has no alignment rule and the UAV is on a buffer this
// pass owns, so both avoid the title buffer binding).
inline constexpr char rect_indirect_args_hlsl[] = R"(
RWByteAddressBuffer scratch : register(u0);

[numthreads(1, 1, 1)]
void convert() {
    uint4 args = uint4(scratch.Load(16), scratch.Load(20), scratch.Load(24), scratch.Load(28));
    uint rectangles = args.x / 3;
    // 0x2AAAAAAA * 6 is the largest product below 2^32.
    uint vertices = rectangles > 0x2AAAAAAAu ? 0 : rectangles * 6;
    scratch.Store(0, vertices);
    scratch.Store(4, args.y);
    scratch.Store(8, args.z);
    scratch.Store(12, args.w);
}
)";

class RectIndirectArguments {
public:
    // A fresh scratch buffer (rests in COMMON). The caller owns one per recording and keeps it alive until every
    // queue that runs that recording has passed its fence, so concurrent recordings never share arguments.
    static Microsoft::WRL::ComPtr<ID3D12Resource> make_scratch(ID3D12Device* device)
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> scratch;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = 256;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        buffer.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (!device || FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                              D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&scratch))))
            scratch.Reset();
        return scratch;
    }
    // Records the conversion of the four draw words at `source_offset` of `source` into `scratch` (from make_scratch,
    // in COMMON) and leaves it in INDIRECT_ARGUMENT state. `source` must already be in a state that includes
    // COPY_SOURCE. Returns false if the pass could not be set up (nothing was recorded then). The compute root
    // signature and pipeline of the list are replaced: the caller refuses first when the title has live compute
    // root arguments. `finish` puts the scratch back in COMMON.
    bool convert(ID3D12GraphicsCommandList* list, ID3D12Resource* source, UINT64 source_offset, ID3D12Resource* scratch)
    {
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        if (!list || !scratch || !source || FAILED(list->GetDevice(IID_PPV_ARGS(&device)))) return false;
        Entry* entry = entry_for(device.Get());
        if (!entry) return false;
        transition(list, scratch, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(scratch, 16, source, source_offset, 16);
        transition(list, scratch, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetComputeRootSignature(entry->root.Get());
        list->SetPipelineState(entry->pipeline.Get());
        list->SetComputeRootUnorderedAccessView(0, scratch->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
        // Make the shader's stores visible before the same buffer is read as indirect arguments. The transition below
        // implies this on a conforming driver; the explicit barrier removes the dependence on that for the command
        // processor's argument fetch.
        D3D12_RESOURCE_BARRIER written{};
        written.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        written.UAV.pResource = scratch;
        list->ResourceBarrier(1, &written);
        transition(list, scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        return true;
    }
    static void finish(ID3D12GraphicsCommandList* list, ID3D12Resource* scratch)
    {
        transition(list, scratch, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
    }

private:
    struct Entry {
        ID3D12Device* device = nullptr;
        Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline;
    };
    static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                           D3D12_RESOURCE_STATES after)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        list->ResourceBarrier(1, &barrier);
    }
    // Root signature and pipeline per device. Entries are never freed: a conversion may still be queued when its list is released.
    Entry* entry_for(ID3D12Device* device)
    {
        std::lock_guard<std::mutex> guard(lock_);
        for (auto& known : entries_)
            if (known->device == device) return known->pipeline ? known.get() : nullptr;
        auto entry = std::make_unique<Entry>();
        entry->device = device;
        build(device, *entry);
        entries_.push_back(std::move(entry));
        return entries_.back()->pipeline ? entries_.back().get() : nullptr;
    }
    static void build(ID3D12Device* device, Entry& entry)
    {
        Microsoft::WRL::ComPtr<ID3DBlob> code, error;
        if (FAILED(D3DCompile(rect_indirect_args_hlsl, sizeof(rect_indirect_args_hlsl) - 1, "rect_indirect_args", nullptr,
                              nullptr, "convert", "cs_5_0", 0, 0, &code, &error)))
            return;
        D3D12_ROOT_PARAMETER parameter{};
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameter.Descriptor.ShaderRegister = 0;
        parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        const D3D12_ROOT_SIGNATURE_DESC desc{1, &parameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        Microsoft::WRL::ComPtr<ID3DBlob> serialized;
        if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &serialized, &error)) ||
            FAILED(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                               IID_PPV_ARGS(&entry.root))))
            return;
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
        pipeline.pRootSignature = entry.root.Get();
        pipeline.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        if (FAILED(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&entry.pipeline)))) entry.pipeline.Reset();
    }
    std::mutex lock_;
    std::vector<std::unique_ptr<Entry>> entries_;
};

inline RectIndirectArguments& rect_indirect_arguments()
{
    static RectIndirectArguments converter;
    return converter;
}

}  // namespace x2w_startup
