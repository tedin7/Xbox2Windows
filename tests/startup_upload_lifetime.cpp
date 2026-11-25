// Lifetime control for the command-list upload arena (Refs #1758, #1774).
// A is recorded and submitted while the queue is blocked on a fence. The arena is then
// reset, a distinct B is written and submitted, and only after that is the gate released.
// The readbacks must hold the exact A and B payloads. The overwrite control writes B into
// A's live range before the release and must read B back from A's copy, which proves that
// the harness sees late GPU reads.
#include "startup_stubs/list_upload_pages.h"

#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <windows.h>

#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT64 kPayloadBytes = 640;
constexpr unsigned char kPayloadA = 0xA1;
constexpr unsigned char kPayloadB = 0xB2;
constexpr DWORD kWaitMs = 10000;

int select_adapter(bool hardware, IDXGIFactory6* factory, ComPtr<IDXGIAdapter>& adapter)
{
    if (!hardware) return SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ? 0 : 3;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> candidate;
        const HRESULT hr = factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                                IID_PPV_ARGS(&candidate));
        if (hr == DXGI_ERROR_NOT_FOUND) return 77;
        if (FAILED(hr)) return 3;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(candidate->GetDesc1(&desc))) return 3;
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
            adapter = candidate;
            return 0;
        }
    }
}

HRESULT make_readback(ID3D12Device* device, UINT64 bytes, ComPtr<ID3D12Resource>& out)
{
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                           nullptr, IID_PPV_ARGS(&out));
}

bool all_bytes(ID3D12Resource* readback, unsigned char value)
{
    D3D12_RANGE read{0, static_cast<SIZE_T>(kPayloadBytes)};
    void* mapped = nullptr;
    if (FAILED(readback->Map(0, &read, &mapped)) || !mapped) return false;
    const auto* bytes = static_cast<const unsigned char*>(mapped);
    bool match = true;
    for (UINT64 i = 0; i < kPayloadBytes; ++i) match = match && bytes[i] == value;
    D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return match;
}

// Records one copy of an allocation into a readback buffer and submits it behind the gate.
struct Submission {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Resource> readback;
};

HRESULT submit_copy(ID3D12Device* device, ID3D12CommandQueue* queue, const x2w_upload::Allocation& source,
                    Submission& out)
{
    HRESULT hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&out.allocator));
    if (FAILED(hr)) return hr;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, out.allocator.Get(), nullptr,
                                   IID_PPV_ARGS(&out.list));
    if (FAILED(hr)) return hr;
    hr = make_readback(device, kPayloadBytes, out.readback);
    if (FAILED(hr)) return hr;
    out.list->CopyBufferRegion(out.readback.Get(), 0, source.resource, source.offset, kPayloadBytes);
    hr = out.list->Close();
    if (FAILED(hr)) return hr;
    ID3D12CommandList* lists[] = {out.list.Get()};
    queue->ExecuteCommandLists(1, lists);
    return S_OK;
}

// Returns nullptr on success, otherwise the name of the first failed check.
const char* run_lifetime(ID3D12Device* device, bool overwrite_live_range)
{
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> gate, done;
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done)))) return "queue or fence creation";
    HANDLE done_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!done_event) return "event creation";

    x2w_upload::Arena arena;
    const auto a = arena.allocate(device, kPayloadBytes, 16);
    if (!a.cpu || a.cpu % 16 || a.gpu % 16 || a.gpu != a.resource->GetGPUVirtualAddress() + a.offset) {
        CloseHandle(done_event);
        return "A allocation";
    }
    // Stands in for the retirement that a production caller owes: the page outlives the arena reset.
    ComPtr<ID3D12Resource> a_page = a.resource;
    std::memset(reinterpret_cast<void*>(a.cpu), kPayloadA, kPayloadBytes);

    const char* failure = nullptr;
    Submission sub_a, sub_b;
    if (FAILED(queue->Wait(gate.Get(), 1))) {
        CloseHandle(done_event);
        return "queue wait";
    }
    if (FAILED(submit_copy(device, queue.Get(), a, sub_a))) failure = "A submission";
    if (!failure && FAILED(queue->Signal(done.Get(), 1))) failure = "A signal";

    // Reset while the GPU has not started A, then write a distinct B.
    arena.reset();
    if (!failure && arena.translate(a.cpu, 1)) failure = "retired A translated";
    x2w_upload::Allocation b{};
    if (!failure) {
        b = arena.allocate(device, kPayloadBytes, 16);
        if (!b.cpu || b.cpu % 16 || b.gpu % 16 || b.resource == a.resource ||
            b.gpu != b.resource->GetGPUVirtualAddress() + b.offset ||
            arena.translate(b.cpu, kPayloadBytes) != b.gpu) failure = "B allocation";
    }
    if (!failure) {
        std::memset(reinterpret_cast<void*>(b.cpu), kPayloadB, kPayloadBytes);
        if (overwrite_live_range) std::memset(reinterpret_cast<void*>(a.cpu), kPayloadB, kPayloadBytes);
        if (FAILED(submit_copy(device, queue.Get(), b, sub_b))) failure = "B submission";
    }
    // Queue a drain point even if an earlier check failed before submitting B.
    const HRESULT drain_signal = queue->Signal(done.Get(), 2);
    if (!failure && FAILED(drain_signal)) failure = "queue drain signal";

    // The gate is still closed, so neither copy may have run.
    Sleep(50);
    if (!failure && done->GetCompletedValue() != 0) failure = "copy ran before the gate opened";

    // Release the queue even after a failed check. Wait only for a successfully queued signal.
    const HRESULT opened = gate->Signal(1);
    if (!failure && FAILED(opened)) failure = "gate signal";
    if (SUCCEEDED(opened) && SUCCEEDED(drain_signal) && SUCCEEDED(done->SetEventOnCompletion(2, done_event))) {
        if (WaitForSingleObject(done_event, kWaitMs) != WAIT_OBJECT_0 && !failure) failure = "GPU completion timeout";
    } else if (!failure) {
        failure = "completion event";
    }
    CloseHandle(done_event);
    if (failure) return failure;

    if (!all_bytes(sub_b.readback.Get(), kPayloadB)) return "B payload";
    if (overwrite_live_range) return all_bytes(sub_a.readback.Get(), kPayloadB) ? nullptr : "overwrite control did not read B";
    return all_bytes(sub_a.readback.Get(), kPayloadA) ? nullptr : "A payload";
}
}

int main(int argc, char** argv)
{
    if (argc != 2 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware"))) return 2;
    ComPtr<ID3D12Debug> debug;
    const bool debugging = SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
    if (debugging) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    ComPtr<IDXGIAdapter> adapter;
    if (const int result = select_adapter(!std::strcmp(argv[1], "hardware"), factory.Get(), adapter)) return result;
    DXGI_ADAPTER_DESC adapter_desc{};
    if (FAILED(adapter->GetDesc(&adapter_desc))) return 3;
    std::printf("adapter=%ls debug_layer=%s\n", adapter_desc.Description, debugging ? "enabled" : "unavailable");
    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) return 4;
    if (const char* mismatch = run_lifetime(device.Get(), false)) {
        std::fprintf(stderr, "Upload lifetime contract failed: %s\n", mismatch);
        return 5;
    }
    if (const char* mismatch = run_lifetime(device.Get(), true)) {
        std::fprintf(stderr, "Upload lifetime overwrite control failed: %s\n", mismatch);
        return 7;
    }
    if (const HRESULT hr = device->GetDeviceRemovedReason(); FAILED(hr)) {
        std::fprintf(stderr, "Device removed: 0x%08lx\n", static_cast<unsigned long>(hr));
        return 6;
    }
    std::puts("A and B payloads survive an arena reset behind a closed gate; overwrite control reads B");
    return 0;
}
