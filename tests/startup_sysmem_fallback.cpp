// System-memory fallback for a texture whose default-heap creation ran out of video memory
// (src/startup_stubs/sysmem_create.h, X2W_D3D12_SYSMEM_FALLBACK=1).
//
// The first default-heap creation is forced to fail with E_OUTOFMEMORY by the creator callable this test passes in; every
// other creation goes to the real device. The fallback texture must live in system memory (pool L0), take uploaded texels,
// be sampled by a compute shader, and read back equal to the CPU reference. Controls: the switch off keeps the failure,
// another failure is not retried, and releasing a fallback texture takes it out of the budget's system-memory count.
//
// Usage: startup_sysmem_fallback <warp|hardware>. Exit 77 when the backend is missing.
#include "startup_stubs/resource_budget.h"
#include "startup_stubs/sysmem_create.h"

#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT size = 4;
constexpr UINT pitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
int failures;

void expect(bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

bool require(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) std::printf("FAIL: %s hr=0x%08lx\n", stage, static_cast<unsigned long>(hr));
    if (FAILED(hr)) ++failures;
    return SUCCEEDED(hr);
}

ComPtr<IDXGIAdapter1> pick_adapter(bool warp)
{
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return adapter;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        return adapter;
    }
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)))
            return adapter;
    }
    adapter.Reset();
    return adapter;
}

D3D12_RESOURCE_DESC texture_desc()
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = size;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    return desc;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    return {D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags};
}

ComPtr<ID3D12Resource> committed_buffer(ID3D12Device* device, D3D12_HEAP_TYPE type, UINT64 bytes, D3D12_RESOURCE_STATES state,
                                        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    const auto desc = buffer_desc(bytes, flags);
    ComPtr<ID3D12Resource> result;
    require(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&result)),
            "Create buffer");
    return result;
}

// The texel at (x, y): bytes chosen so that no two texels or channels agree.
std::array<UINT8, 4> texel(UINT x, UINT y)
{
    return {static_cast<UINT8>(17 + 11 * x + 3 * y), static_cast<UINT8>(200 - 9 * x - 7 * y),
            static_cast<UINT8>(5 + 13 * y + x), static_cast<UINT8>(255 - 4 * (x + 4 * y))};
}

const char* shader_source = R"hlsl(

Texture2D<float4> source : register(t0);
RWBuffer<uint> result : register(u0);
SamplerState point_sampler : register(s0);
[RootSignature("DescriptorTable(SRV(t0), UAV(u0)), StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_POINT)")]
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float2 uv = (float2(id.x % 4, id.x / 4) + 0.5) / 4.0;
    uint4 c = (uint4)round(saturate(source.SampleLevel(point_sampler, uv, 0)) * 255.0);
    result[id.x] = c.x | (c.y << 8) | (c.z << 16) | (c.w << 24);
}
)hlsl";

// Uploads the known texels, samples every one with a compute shader and compares with the CPU reference.
void sample_and_compare(ID3D12Device* device, ID3D12Resource* texture)
{
    ComPtr<ID3DBlob> code, errors;
    if (FAILED(D3DCompile(shader_source, std::strlen(shader_source), "sysmem_fallback", nullptr, nullptr, "main", "cs_5_1", 0,
                          0, &code, &errors))) {
        std::printf("FAIL: compile: %s\n", errors ? static_cast<const char*>(errors->GetBufferPointer()) : "?");
        ++failures;
        return;
    }
    ComPtr<ID3D12RootSignature> signature;
    if (!require(device->CreateRootSignature(0, code->GetBufferPointer(), code->GetBufferSize(), IID_PPV_ARGS(&signature)),
                 "Create root signature from the shader"))
        return;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = signature.Get();
    pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    ComPtr<ID3D12PipelineState> pipeline;
    if (!require(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline)), "Create compute pipeline")) return;

    auto upload = committed_buffer(device, D3D12_HEAP_TYPE_UPLOAD, pitch * size, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto output = committed_buffer(device, D3D12_HEAP_TYPE_DEFAULT, 16 * 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback = committed_buffer(device, D3D12_HEAP_TYPE_READBACK, 16 * 4, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!upload || !output || !readback) return;
    void* mapped = nullptr;
    if (!require(upload->Map(0, nullptr, &mapped), "Map upload")) return;
    for (UINT y = 0; y < size; ++y)
        for (UINT x = 0; x < size; ++x) std::memcpy(static_cast<UINT8*>(mapped) + y * pitch + 4 * x, texel(x, y).data(), 4);
    upload->Unmap(0, nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
    ComPtr<ID3D12DescriptorHeap> heap;
    if (!require(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "Create descriptor heap")) return;
    const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(texture, &srv, cpu);
    cpu.ptr += step;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_UINT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = 16;
    device->CreateUnorderedAccessView(output.Get(), nullptr, &uav, cpu);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    if (!require(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "Create queue") ||
        !require(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "Create allocator") ||
        !require(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pipeline.Get(),
                                           IID_PPV_ARGS(&list)), "Create list"))
        return;

    D3D12_TEXTURE_COPY_LOCATION destination{texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION source{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    source.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, size, size, 1, pitch};
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Transition = {texture, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    list->ResourceBarrier(1, &barrier);
    ID3D12DescriptorHeap* heaps[]{heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(signature.Get());
    list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    list->Dispatch(1, 1, 1);
    barrier.Transition = {output.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, 16 * 4);
    if (!require(list->Close(), "Close list")) return;
    ID3D12CommandList* lists[]{list.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    if (!require(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create fence") ||
        !require(queue->Signal(fence.Get(), 1), "Signal") || !require(fence->SetEventOnCompletion(1, nullptr), "Wait"))
        return;
    if (!require(device->GetDeviceRemovedReason(), "Device state after the submission")) return;

    const D3D12_RANGE range{0, 16 * 4};
    if (!require(readback->Map(0, &range, &mapped), "Map readback")) return;
    const auto* words = static_cast<const UINT32*>(mapped);
    UINT wrong = 0;
    for (UINT i = 0; i < 16; ++i) {
        const auto expected = texel(i % 4, i / 4);
        const UINT32 reference = expected[0] | (expected[1] << 8) | (expected[2] << 16) | (UINT32(expected[3]) << 24);
        if (words[i] != reference) {
            if (wrong++ < 3) std::printf("texel %u: got 0x%08x expected 0x%08x\n", i, words[i], reference);
        }
    }
    readback->Unmap(0, nullptr);
    expect(!wrong, "the sampled texels of the fallback texture equal the CPU reference");
}
}  // namespace

int main(int argc, char** argv)
{
    const bool warp = argc < 2 || std::string(argv[1]) != "hardware";
    auto adapter = pick_adapter(warp);
    ComPtr<ID3D12Device> device;
    if (!adapter || FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
        std::puts("SKIP: backend unavailable");
        return 77;
    }
    D3D12_FEATURE_DATA_ARCHITECTURE1 architecture{};
    device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &architecture, sizeof(architecture));
    std::printf("adapter UMA=%d cache-coherent=%d\n", architecture.UMA, architecture.CacheCoherentUMA);

    const auto desc = texture_desc();
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    int creations = 0, default_creations = 0;
    HRESULT other_failure = S_OK;
    // The first default-heap creation runs out of video memory; every other creation reaches the device.
    ComPtr<ID3D12Resource> created;
    const auto create = [&](const D3D12_HEAP_PROPERTIES& heap) -> HRESULT {
        ++creations;
        if (other_failure != S_OK) return other_failure;
        if (heap.Type == D3D12_HEAP_TYPE_DEFAULT && ++default_creations == 1) return x2w_sysmem::out_of_memory;
        created.Reset();
        return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&created));
    };

    // Switch on: the texture is created in system memory and holds, samples and returns the uploaded texels.
    bool used = false;
    const HRESULT hr = x2w_sysmem::create_with_fallback(device.Get(), default_heap, create, used, true);
    if (!require(hr, "Create the texture with the fallback on") || !created) return 1;
    expect(used, "the retry produced the texture");
    expect(creations == 2, "one failed default-heap creation and one retry");
    D3D12_HEAP_PROPERTIES where{};
    D3D12_HEAP_FLAGS flags{};
    created->GetHeapProperties(&where, &flags);
    std::printf("fallback heap: type=%d cpu-page=%d pool=%d\n", where.Type, where.CPUPageProperty, where.MemoryPoolPreference);
    expect(where.MemoryPoolPreference == D3D12_MEMORY_POOL_L0, "the fallback texture lives in the system pool (L0)");
    expect(where.Type == D3D12_HEAP_TYPE_CUSTOM, "the fallback heap is a custom heap");
    sample_and_compare(device.Get(), created.Get());

    // Control: with the switch off the failure stays as it is today, and nothing is retried.
    creations = default_creations = 0;
    created.Reset();
    used = true;
    const HRESULT off = x2w_sysmem::create_with_fallback(device.Get(), default_heap, create, used, false);
    expect(off == x2w_sysmem::out_of_memory && !used && !created && creations == 1,
           "with the switch off the creation fails with out of memory and is not retried");

    // Control: a failure other than out of memory is returned as it is, without a retry.
    creations = default_creations = 0;
    other_failure = E_INVALIDARG;
    const HRESULT invalid = x2w_sysmem::create_with_fallback(device.Get(), default_heap, create, used, true);
    expect(invalid == E_INVALIDARG && !used && creations == 1, "another failure is returned without a retry");
    other_failure = S_OK;

    // Control: a heap that is not the default one is never retried (an upload-heap texture failing with out of memory).
    creations = 0;
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const HRESULT upload = x2w_sysmem::create_with_fallback(device.Get(), upload_heap, [&](const D3D12_HEAP_PROPERTIES&) {
        ++creations;
        return x2w_sysmem::out_of_memory;
    }, used, true);
    expect(upload == x2w_sysmem::out_of_memory && !used && creations == 1, "an upload heap is not retried");

    // Control: a fallback texture is counted as system memory, and releasing it takes the count back.
    if (x2w_budget::enabled()) {
        creations = default_creations = 0;
        auto before = x2w_budget::snapshot();
        {
            auto fallback = x2w_sysmem::create_with_fallback(device.Get(), default_heap, create, used, true);
            require(fallback, "Create a second fallback texture");
            auto* ticket = x2w_budget::committed(device.Get(), created.Get(), D3D12_HEAP_TYPE_DEFAULT, used);
            const auto held = x2w_budget::snapshot();
            const auto& live = held.live[x2w_budget::sysmem];
            expect(live.objects == before.live[x2w_budget::sysmem].objects + 1 && live.bytes > before.live[x2w_budget::sysmem].bytes,
                   "a fallback texture is counted in the system-memory category");
            expect(held.live[x2w_budget::texture].objects == before.live[x2w_budget::texture].objects,
                   "and not in the video-memory texture category");
            expect(held.sysmem_fallbacks == before.sysmem_fallbacks + 1 && held.sysmem_fallback_bytes > before.sysmem_fallback_bytes,
                   "and in the fallback totals");
            (void)ticket;
            created.Reset();
        }
        const auto after = x2w_budget::snapshot();
        expect(after.live[x2w_budget::sysmem].objects == before.live[x2w_budget::sysmem].objects &&
                   after.live[x2w_budget::sysmem].bytes == before.live[x2w_budget::sysmem].bytes &&
                   after.retained[x2w_budget::sysmem].objects == before.retained[x2w_budget::sysmem].objects,
               "releasing the fallback texture takes the system-memory count back");
        expect(after.sysmem_fallbacks == before.sysmem_fallbacks + 1, "the fallback total is not reduced by the release");
    } else {
        std::puts("FAIL: run with X2W_D3D12_BUDGET=1");
        ++failures;
    }

    if (failures) return 1;
    std::puts("sysmem fallback checks passed");
    return 0;
}
