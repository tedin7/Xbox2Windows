#include "startup_stubs/presenter_fence_win.h"
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

using Microsoft::WRL::ComPtr;
using namespace x2w_startup;

struct Fake {
    std::uint64_t time = 0, done = 0;
    bool dead = false, arm_ok = true, fail_wait = false;
    int arms = 0, waits = 0, warnings = 0, complete_after = 3, remove_after = 0;
    bool signal_slice = false;
    std::uint64_t now() { return time; }
    bool removed() { return dead; }
    std::uint64_t completed() { return done; }
    bool arm(std::uint64_t) { ++arms; return arm_ok; }
    FenceSlice wait() {
        ++waits; time += 1001;
        if (fail_wait) return FenceSlice::failed;
        if (waits == complete_after) done = 1;
        if (waits == remove_after) dead = true;
        return signal_slice ? FenceSlice::signaled : FenceSlice::timeout;
    }
    void slow() { ++warnings; }
};
static bool pure_checks()
{
    Fake ready; ready.done = 1;
    Fake before = ready;
    if (wait_for_submission(ready, 1) != FenceWait::reached || std::memcmp(&ready, &before, sizeof ready)) return false;
    Fake slow;
    if (wait_for_submission(slow, 1) != FenceWait::reached || slow.waits != 3 || slow.arms != 1 || slow.warnings != 1) return false;
    Fake removed; removed.dead = true; removed.done = ~std::uint64_t{0};
    if (wait_for_submission(removed, 1) != FenceWait::removed || removed.arms) return false;
    Fake later_removed; later_removed.remove_after = 1;
    if (wait_for_submission(later_removed, 1) != FenceWait::removed || later_removed.waits != 1) return false;
    Fake signaled; signaled.signal_slice = true; signaled.complete_after = 2;
    if (wait_for_submission(signaled, 1) != FenceWait::reached || signaled.arms != 2) return false;
    Fake arm; arm.arm_ok = false;
    if (wait_for_submission(arm, 1) != FenceWait::arm_failed || arm.waits) return false;
    Fake failed; failed.fail_wait = true;
    return wait_for_submission(failed, 1) == FenceWait::wait_failed;
}
static std::atomic<unsigned> errors{0};
static void WINAPI message(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID, LPCSTR text, void*)
{
    if (severity <= D3D12_MESSAGE_SEVERITY_ERROR && ++errors <= 12) std::printf("debug layer: %.300s\n", text);
}
int main(int argc, char** argv)
{
    if (!pure_checks()) return 1;
    if (argc == 1) return 0;
    if (argc != 2) return 2;
    ComPtr<ID3D12Debug> debug;
    if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) { std::puts("no debug layer"); return 3; }
    debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 4;
    ComPtr<IDXGIAdapter> adapter;
    if (!std::strcmp(argv[1], "warp")) {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 4;
    } else if (!std::strcmp(argv[1], "hardware")) {
        for (UINT i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> candidate;
            HRESULT hr = factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr)) return 4;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return 4;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) { adapter = candidate; break; }
        }
    } else return 2;
    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) return 5;
    ComPtr<ID3D12InfoQueue1> info;
    DWORD cookie = 0;
    if (FAILED(device.As(&info)) || FAILED(info->RegisterMessageCallback(message, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie))) {
        std::puts("no debug layer"); return 3;
    }
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> gate, completed;
    D3D12_COMMAND_QUEUE_DESC desc{};
    if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completed))) || FAILED(list->Close())) return 6;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event || FAILED(queue->Wait(gate.Get(), 1))) return 7;
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(completed.Get(), 1))) return 7;
    std::thread release([&] { std::this_thread::sleep_for(std::chrono::milliseconds(2600)); gate->Signal(1); });
    const auto start = std::chrono::steady_clock::now();
    const auto result = wait_for_fence(completed.Get(), 1, event, device.Get(), [] {});
    const auto elapsed = std::chrono::steady_clock::now() - start;
    bool passed = result == FenceWait::reached && elapsed >= std::chrono::milliseconds(2200) &&
                  completed->GetCompletedValue() >= 1 && SUCCEEDED(allocator->Reset());
    release.join();
    passed = passed && device->GetDeviceRemovedReason() == S_OK && errors == 0;
    CloseHandle(event);
    info->UnregisterMessageCallback(cookie);
    return passed ? 0 : 8;
}
