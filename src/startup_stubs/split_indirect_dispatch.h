// Split of a batched indirect Dispatch into single-command calls, for adapters whose driver runs only the first
// command of an ExecuteIndirect with a Dispatch signature. Opt-in only: slot 61 applies it with
// X2W_SPLIT_INDIRECT_DISPATCH=1, to a rootless Dispatch signature with a desktop compute pipeline and root.
//
// On an AMD Radeon RX 5700 XT (RDNA 1, device 731f), drivers 31.0.23013.1023 to 32.0.21045.29003 run exactly one
// command of ExecuteIndirect(Dispatch signature, MaxCommandCount > 1), with or without a count buffer, and ignore
// the count buffer of a single-command call. Single-command calls without a count buffer are exact. Measurements:
// docs/experiments.md#batched-indirect-dispatch-on-rdna-1.
//
// The split copies the count word and the records into a scratch buffer, and a one-group compute pass rewrites
// record i as itself when i < min(count, MaxCommandCount) and as (0,0,0) otherwise. A zero-group Dispatch does
// nothing. The caller then issues MaxCommandCount single-command calls without a count buffer over the rewritten
// records. The title's own buffers are only read by copies, as in rect_indirect_args.h.
#pragma once

#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <mutex>
#include <memory>
#include <vector>

namespace x2w_startup {

// Scratch layout: the count word at byte 0, the title's records copied at byte `split_records_copy`, the rewritten
// 12-byte records at `rewritten_offset(...)`.
inline constexpr UINT64 split_records_copy = 16;

inline constexpr char split_indirect_dispatch_hlsl[] = R"(
RWByteAddressBuffer scratch : register(u0);
cbuffer Split : register(b0) { uint max_count; uint stride; uint counted; uint rewritten; };

[numthreads(64, 1, 1)]
void split(uint i : SV_DispatchThreadID) {
    if (i >= max_count) return;
    uint count = counted != 0 ? scratch.Load(0) : max_count;
    uint3 groups = i < count ? scratch.Load3(16 + i * stride) : uint3(0, 0, 0);
    scratch.Store3(rewritten + 12 * i, groups);
}
)";

class SplitIndirectDispatch {
public:
    // Bytes the title's records occupy: every record up to the maximum, the last one only its 12 Dispatch bytes.
    static UINT64 record_span(UINT max_count, UINT stride)
    {
        return UINT64{max_count - 1} * stride + sizeof(D3D12_DISPATCH_ARGUMENTS);
    }
    static UINT64 rewritten_offset(UINT max_count, UINT stride)
    {
        return (split_records_copy + record_span(max_count, stride) + 15) & ~UINT64{15};
    }
    // A fresh scratch buffer for one call (rests in COMMON). The caller keeps it alive with the recording.
    static Microsoft::WRL::ComPtr<ID3D12Resource> make_scratch(ID3D12Device* device, UINT max_count, UINT stride)
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> scratch;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = rewritten_offset(max_count, stride) + UINT64{max_count} * sizeof(D3D12_DISPATCH_ARGUMENTS);
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        buffer.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (!device || FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                              D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&scratch))))
            scratch.Reset();
        return scratch;
    }
    // Records the copies and the rewrite into `scratch` (from make_scratch, in COMMON) and leaves it in
    // INDIRECT_ARGUMENT state. `records` (and `counts`, if given) must already be in a state that includes
    // COPY_SOURCE. Returns false if the pass could not be set up (nothing was recorded then). The compute root
    // signature and pipeline of the list are replaced: the caller restores them. `finish` puts the scratch back in COMMON.
    bool rewrite(ID3D12GraphicsCommandList* list, ID3D12Resource* records, UINT64 records_offset, UINT max_count, UINT stride,
                 ID3D12Resource* counts, UINT64 count_offset, ID3D12Resource* scratch)
    {
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        if (!list || !records || !scratch || !max_count || stride < sizeof(D3D12_DISPATCH_ARGUMENTS) ||
            FAILED(list->GetDevice(IID_PPV_ARGS(&device))))
            return false;
        Entry* entry = entry_for(device.Get());
        if (!entry) return false;
        transition(list, scratch, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        if (counts) list->CopyBufferRegion(scratch, 0, counts, count_offset, sizeof(UINT));
        list->CopyBufferRegion(scratch, split_records_copy, records, records_offset, record_span(max_count, stride));
        transition(list, scratch, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetComputeRootSignature(entry->root.Get());
        list->SetPipelineState(entry->pipeline.Get());
        list->SetComputeRootUnorderedAccessView(0, scratch->GetGPUVirtualAddress());
        const UINT constants[4]{max_count, stride, counts ? 1u : 0u, static_cast<UINT>(rewritten_offset(max_count, stride))};
        list->SetComputeRoot32BitConstants(1, 4, constants, 0);
        list->Dispatch((max_count + 63) / 64, 1, 1);
        // As in rect_indirect_args.h: make the stores visible to the argument fetch without relying on the transition.
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
    // Root signature and pipeline per device. Entries are never freed: a rewrite may still be queued when its list is released.
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
        if (FAILED(D3DCompile(split_indirect_dispatch_hlsl, sizeof(split_indirect_dispatch_hlsl) - 1, "split_indirect_dispatch",
                              nullptr, nullptr, "split", "cs_5_0", 0, 0, &code, &error)))
            return;
        D3D12_ROOT_PARAMETER parameters[2]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[1].Constants = {0, 0, 4};
        parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        const D3D12_ROOT_SIGNATURE_DESC desc{2, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
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

// Never destroyed: a function-local static would release its root signatures and pipelines in the destructor during
// DLL detach, and on an RX 9070 XT (driver 32.0.32029.2010) the driver fail-fasts there (0xC0000409 in amdxc64,
// review of #2683). The entries are already kept for the life of the process (see entry_for).
// Lifetime boundary: this cache is process-owned and released only by process exit. Explicitly unloading and
// reloading d3d12_xs.dll in a live host is not a supported path: each load allocates a new cache and nothing tears
// the previous one down outside the loader lock, so its entries stay allocated. Supporting it needs that teardown
// and its own coverage. The kept root signature and pipeline each hold a reference to their device, so a cached
// device is never destroyed and its address cannot be reused by another device while the entry lives.
inline SplitIndirectDispatch& split_indirect_dispatch()
{
    static SplitIndirectDispatch* const splitter = new SplitIndirectDispatch;
    return *splitter;
}

}  // namespace x2w_startup
