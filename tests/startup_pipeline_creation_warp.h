#pragma once

#include "startup_stubs/pipeline_creation.h"
#include "startup_stubs/local_log.h"

#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <thread>

namespace pipeline_creation_warp {

using Microsoft::WRL::ComPtr;
using x2w_startup::PipelineCreation;
using x2w_startup::PipelineCreationTotals;

constexpr unsigned key_count = 16;
using Pixel = std::array<unsigned char, 4>;
using Picture = std::array<Pixel, 2 * key_count>;

struct Lock {
    SRWLOCK lock = SRWLOCK_INIT;
    LONG64 begin = 0, longest = 0;

    static LONG64 now()
    {
        LARGE_INTEGER ticks;
        QueryPerformanceCounter(&ticks);
        return ticks.QuadPart;
    }

    void acquire()
    {
        AcquireSRWLockExclusive(&lock);
        begin = now();
    }

    void end_hold()
    {
        const auto elapsed = now() - begin;
        if (elapsed > longest) longest = elapsed;
    }

    void release()
    {
        end_hold();
        ReleaseSRWLockExclusive(&lock);
    }

    void wait(CONDITION_VARIABLE& ready)
    {
        end_hold();
        SleepConditionVariableSRW(&ready, &lock, INFINITE, 0);
        begin = now();
    }
};

struct Entry {
    PipelineCreation creation;
    ComPtr<ID3D12PipelineState> result;
    unsigned arrivals = 0;
};
struct Cache {
    Lock lock;
    std::array<Entry, key_count + 1> entries;
    PipelineCreationTotals totals;

    template<class Create>
    ID3D12PipelineState* get(unsigned key, bool unlocked, bool race, bool mutant, Create&& create)
    {
        auto& entry = entries[key];
        lock.acquire();
        const bool first = ++entry.arrivals == 1;
        if (entry.arrivals == 2) WakeAllConditionVariable(&entry.creation.ready);
        entry.creation.wait([&](CONDITION_VARIABLE& ready) { lock.wait(ready); });
        if (!entry.creation.finished) {
            ComPtr<ID3D12PipelineState> result;
            const auto release = [&] {
                // Emulate a miss that was never reserved, while preserving the real driver call.
                if (mutant) entry.creation.pending = false;
                lock.release();
            };
            const auto invoke = [&] {
                if (race && unlocked && first) {
                    // Hold the first creation open until the peer has looked up this key.
                    // Use the raw lock here since the wait below intentionally consumes no hold time.
                    AcquireSRWLockExclusive(&lock.lock);
                    while (entry.arrivals < 2)
                        SleepConditionVariableSRW(&entry.creation.ready, &lock.lock, INFINITE, 0);
                    ReleaseSRWLockExclusive(&lock.lock);
                    Sleep(30);
                }
                return create(result);
            };
            x2w_startup::create_pipeline(entry.creation, unlocked, totals, release, [&] { lock.acquire(); }, invoke);
            // A command list may still reference the peer's result in the mutant run.
            if (!entry.creation.finished) entry.result = result;
            entry.creation.finish();
        }
        auto* result = entry.result.Get();
        lock.release();
        return result;
    }
};

struct Target {
    ComPtr<ID3D12Resource> color, depth, readback;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{}, dsv{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
};

inline DXGI_FORMAT format(unsigned key)
{
    return key < 8 ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_B8G8R8A8_UNORM;
}

inline Pixel reference(unsigned key)
{
    // The incoming depth is .25, and the depth clear is .5.
    const bool passes[] = {false, true, false, true, false, true, false, true};
    if (!passes[key % 8]) return {0, 255, 0, 255};
    return key < 8 ? Pixel{255, 0, 0, 255} : Pixel{0, 0, 255, 255};
}

inline HRESULT buffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE heap_type,
                      D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags, ID3D12Resource** out)
{
    const D3D12_HEAP_PROPERTIES heap{heap_type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                                   {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, flags};
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(out));
}

inline bool target(ID3D12Device* device, unsigned key, Target& t)
{
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 1, 1, 1, format(key),
                             {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    UINT64 size = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &t.footprint, nullptr, nullptr, &size);
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                               nullptr, IID_PPV_ARGS(&t.color)))) return false;
    desc.Format = DXGI_FORMAT_D32_FLOAT;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                               nullptr, IID_PPV_ARGS(&t.depth)))) return false;
    device->CreateRenderTargetView(t.color.Get(), nullptr, t.rtv);
    device->CreateDepthStencilView(t.depth.Get(), nullptr, t.dsv);
    return SUCCEEDED(buffer(device, size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_FLAG_NONE, t.readback.GetAddressOf()));
}

inline void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                       D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &barrier);
}

struct Commands {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;

    bool open(ID3D12Device* device)
    {
        return SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) &&
               SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                                   nullptr, IID_PPV_ARGS(&list)));
    }
};

inline bool execute(ID3D12Device* device, ID3D12CommandList* const* lists, UINT count)
{
    const D3D12_COMMAND_QUEUE_DESC desc{};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return false;
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    queue->ExecuteCommandLists(count, lists);
    const bool complete = SUCCEEDED(queue->Signal(fence.Get(), 1)) &&
                          SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                          WaitForSingleObject(event, 30000) == WAIT_OBJECT_0;
    CloseHandle(event);
    return complete && SUCCEEDED(device->GetDeviceRemovedReason());
}

inline bool compile(const char* source, const char* profile, ComPtr<ID3DBlob>& code)
{
    ComPtr<ID3DBlob> errors;
    const HRESULT hr = D3DCompile(source, std::strlen(source), nullptr, nullptr, nullptr, "main", profile,
                                  0, 0, &code, &errors);
    if (errors) std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
    return SUCCEEDED(hr);
}

inline bool root_signature(ID3D12Device* device, bool compute, ComPtr<ID3D12RootSignature>& root)
{
    D3D12_ROOT_PARAMETER uav{};
    uav.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    if (compute) {
        desc.NumParameters = 1;
        desc.pParameters = &uav;
    }
    ComPtr<ID3DBlob> bytes;
    return SUCCEEDED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &bytes, nullptr)) &&
           SUCCEEDED(device->CreateRootSignature(0, bytes->GetBufferPointer(), bytes->GetBufferSize(),
                                                  IID_PPV_ARGS(&root)));
}

inline void draw(ID3D12GraphicsCommandList* list, Target& t, ID3D12RootSignature* root, ID3D12PipelineState* pso)
{
    const float clear[] = {0, 1, 0, 1};
    list->ClearRenderTargetView(t.rtv, clear, 0, nullptr);
    list->ClearDepthStencilView(t.dsv, D3D12_CLEAR_FLAG_DEPTH, .5f, 0, 0, nullptr);
    const D3D12_VIEWPORT viewport{0, 0, 1, 1, 0, 1};
    const D3D12_RECT scissor{0, 0, 1, 1};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &t.rtv, FALSE, &t.dsv);
    list->SetGraphicsRootSignature(root);
    list->SetPipelineState(pso);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3, 1, 0, 0);
    transition(list, t.color.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dest{}, source{};
    dest.pResource = t.readback.Get();
    dest.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dest.PlacedFootprint = t.footprint;
    source.pResource = t.color.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list->CopyTextureRegion(&dest, 0, 0, 0, &source, nullptr);
}

inline bool read(ID3D12Resource* resource, void* out, SIZE_T bytes)
{
    const D3D12_RANGE range{0, bytes}, written{0, 0};
    void* data = nullptr;
    if (FAILED(resource->Map(0, &range, &data))) return false;
    std::memcpy(out, data, bytes);
    resource->Unmap(0, &written);
    return true;
}

inline bool graphics(ID3D12Device* device, bool unlocked, bool threaded, bool mutant,
                     Picture& picture, bool& duplicate)
{
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3DBlob> vs, ps;
    if (!root_signature(device, false, root) ||
        !compile("float4 main(uint i : SV_VertexID) : SV_Position {"
                 "return float4(i == 1 ? 3 : -1, i == 2 ? -3 : 1, .25, 1); }", "vs_5_0", vs) ||
        !compile("float4 main() : SV_Target { return float4(1, 0, 0, 1); }", "ps_5_0", ps)) return false;

    Cache cache;
    std::array<Target, 2 * key_count> targets;
    ComPtr<ID3D12DescriptorHeap> rtvs, dsvs;
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, static_cast<UINT>(targets.size())};
    const D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, static_cast<UINT>(targets.size())};
    if (FAILED(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtvs))) ||
        FAILED(device->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsvs)))) return false;
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    const auto dsv = dsvs->GetCPUDescriptorHandleForHeapStart();
    for (unsigned i = 0; i < targets.size(); ++i) {
        targets[i].rtv.ptr = rtv.ptr + i * device->GetDescriptorHandleIncrementSize(rtv_desc.Type);
        targets[i].dsv.ptr = dsv.ptr + i * device->GetDescriptorHandleIncrementSize(dsv_desc.Type);
        if (!target(device, i / 2, targets[i])) return false;
    }

    auto get = [&](unsigned key, bool paired) {
        return cache.get(key, unlocked, paired, mutant, [&](ComPtr<ID3D12PipelineState>& result) {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = root.Get();
            desc.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
            desc.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
            auto& blend = desc.BlendState.RenderTarget[0];
            blend = {FALSE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD,
                     D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL};
            desc.SampleMask = ~0u;
            desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            desc.RasterizerState.DepthClipEnable = TRUE;
            desc.DepthStencilState.DepthEnable = TRUE;
            desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
            desc.DepthStencilState.DepthFunc = static_cast<D3D12_COMPARISON_FUNC>(1 + key % 8);
            auto& front = desc.DepthStencilState.FrontFace;
            front = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_COMPARISON_FUNC_ALWAYS};
            desc.DepthStencilState.BackFace = front;
            desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
            desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            desc.NumRenderTargets = 1;
            desc.RTVFormats[0] = format(key);
            desc.SampleDesc.Count = key == key_count ? 0 : 1;
            return device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&result));
        });
    };

    std::array<Commands, 4> commands;
    const unsigned count = threaded ? 4 : 1;
    for (unsigned i = 0; i < count; ++i)
        if (!commands[i].open(device)) return false;
    std::array<bool, 4> ok{};
    auto record = [&](unsigned worker) {
        auto* list = commands[worker].list.Get();
        const unsigned begin = threaded ? (worker / 2) * key_count / 2 : 0;
        const unsigned end = threaded ? begin + key_count / 2 : key_count;
        bool drew_all = true;
        for (unsigned key = begin; key < end; ++key) {
            auto* pso = get(key, threaded);
            if (!pso) {
                drew_all = false;
                continue;
            }
            if (!threaded || worker % 2 == 0) draw(list, targets[key * 2], root.Get(), pso);
            if (!threaded || worker % 2 == 1) draw(list, targets[key * 2 + 1], root.Get(), pso);
        }
        ok[worker] = SUCCEEDED(list->Close()) && drew_all;
    };
    if (threaded) {
        std::array<std::thread, 4> workers;
        for (unsigned i = 0; i < count; ++i) workers[i] = std::thread(record, i);
        for (auto& worker : workers) worker.join();
    } else {
        record(0);
    }

    const auto valid = cache.totals;
    const auto longest = cache.lock.longest;
    std::printf("graphics created %lld outside %lld, maximum hold %lld ticks, maximum creation %lld ticks\n",
                valid.created, valid.outside, longest, valid.max_ticks);
    duplicate = valid.created > key_count;
    if ((!mutant && valid.created != key_count) || valid.failed ||
        valid.outside != (unlocked ? valid.created : 0) ||
        (threaded && unlocked && longest >= valid.max_ticks)) return false;

    for (unsigned retry = 0; retry < 32; ++retry)
        if (get(key_count, false)) return false;
    if (cache.totals.failed != 1 || cache.totals.created != valid.created) return false;
    std::printf("invalid graphics key failed %lld time in 32 lookups\n", cache.totals.failed);

    std::array<ID3D12CommandList*, 4> lists{};
    for (unsigned i = 0; i < count; ++i) {
        if (!ok[i]) return false;
        lists[i] = commands[i].list.Get();
    }
    if (!execute(device, lists.data(), count)) return false;
    for (unsigned i = 0; i < picture.size(); ++i) {
        if (!read(targets[i].readback.Get(), picture[i].data(), picture[i].size()) ||
            picture[i] != reference(i / 2)) {
            std::fprintf(stderr, "pixel mismatch at key %u draw %u\n", i / 2, i % 2);
            return false;
        }
    }
    return true;
}

inline bool compute(ID3D12Device* device, bool unlocked)
{
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3DBlob> cs;
    ComPtr<ID3D12Resource> output, readback;
    if (!root_signature(device, true, root) ||
        !compile("RWByteAddressBuffer dst : register(u0);"
                 "[numthreads(4,1,1)] void main(uint3 idx : SV_DispatchThreadID) {"
                 "dst.Store(4 * idx.x, 3 * idx.x + 7); }", "cs_5_0", cs) ||
        FAILED(buffer(device, 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, output.GetAddressOf())) ||
        FAILED(buffer(device, 16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                      D3D12_RESOURCE_FLAG_NONE, readback.GetAddressOf()))) return false;

    Cache cache;
    std::array<ID3D12PipelineState*, 2> results{};
    auto get = [&](unsigned worker) {
        results[worker] = cache.get(0, unlocked, true, false, [&](ComPtr<ID3D12PipelineState>& result) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = root.Get();
            desc.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
            return device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&result));
        });
    };
    std::thread peer(get, 1);
    get(0);
    peer.join();
    if (!results[0] || results[0] != results[1] || cache.totals.created != 1 ||
        cache.totals.failed || cache.totals.outside != (unlocked ? 1 : 0)) return false;

    Commands commands;
    if (!commands.open(device)) return false;
    auto* list = commands.list.Get();
    list->SetComputeRootSignature(root.Get());
    list->SetPipelineState(results[0]);
    list->SetComputeRootUnorderedAccessView(0, output->GetGPUVirtualAddress());
    list->Dispatch(1, 1, 1);
    transition(list, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, 16);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list};
    std::array<UINT, 4> actual{}, expected{};
    for (UINT i = 0; i < static_cast<UINT>(expected.size()); ++i) expected[i] = 3 * i + 7;
    return execute(device, lists, 1) && read(readback.Get(), actual.data(), sizeof(actual)) && actual == expected;
}

inline int run(bool mutant)
{
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        std::fprintf(stderr, "WARP device creation failed\n");
        return 2;
    }
    const bool unlocked = x2w_local::env_on("X2W_PIPELINE_CREATE_UNLOCKED");
    Picture baseline{}, threaded{};
    bool duplicate = false;
    if (!graphics(device.Get(), false, false, false, baseline, duplicate) ||
        !graphics(device.Get(), unlocked, true, mutant, threaded, duplicate) ||
        baseline != threaded || !compute(device.Get(), unlocked)) {
        std::fprintf(stderr, "pipeline creation or readback check failed\n");
        return 2;
    }
    if (duplicate) {
        std::fprintf(stderr, "duplicate graphics pipeline creation\n");
        return 1;
    }
    return 0;
}

}  // namespace pipeline_creation_warp
