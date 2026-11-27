// Synthetic viewport and scissor validation through the shim.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0, Word f = 0, Word g = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f, g);
}
bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }

std::atomic<unsigned long long> debug_error_count{0};
void WINAPI on_message(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID, LPCSTR text, void*)
{
    if (severity > D3D12_MESSAGE_SEVERITY_ERROR) return;
    if (++debug_error_count <= 12) std::printf("debug layer: %.300s\n", text);
}

struct Probe {
    ComPtr<ID3D12Device> desktop;
    DWORD cookie = 0;
    ComPtr<ID3D12InfoQueue1> queue1;
    bool watch()
    {
        return SUCCEEDED(desktop.As(&queue1)) &&
               SUCCEEDED(queue1->RegisterMessageCallback(on_message, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie));
    }
};
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    ComPtr<IDXGIAdapter> adapter;
    if (std::strcmp(argv[1], "hardware") == 0) {
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT hr = factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr)) return 3;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    } else if (std::strcmp(argv[1], "warp") != 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 3;
    }
    DXGI_ADAPTER_DESC adapter_desc{};
    if (FAILED(adapter->GetDesc(&adapter_desc))) return 3;
    std::printf("adapter=%ls\n", adapter_desc.Description);

    Probe probe;
    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 4;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&probe.desktop)))) return 4;

    if (!probe.watch()) { std::printf("no debug layer\n"); return 4; }

    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 4, 4, 1, 1,
        DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 0}};
    ComPtr<ID3D12Resource> target;
    if (FAILED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COMMON, &clear, IID_PPV_ARGS(&target)))) return 5;
    struct HeapDesc { D3D12_DESCRIPTOR_HEAP_DESC desc; Word backing, bytes; } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1}, 0, 0};
    void* rtv_heap = nullptr;
    if (!ok(invoke(device, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&rtv_heap)))) return 6;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{static_cast<SIZE_T>(invoke(rtv_heap, 9))};
    D3D12_RENDER_TARGET_VIEW_DESC view{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    invoke(device, 20, arg(target.Get()), arg(&view), rtv.ptr);
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void *queue = nullptr, *fence = nullptr, *allocator = nullptr, *list = nullptr;
    if (!ok(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&queue))) ||
        !ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&fence))) ||
        !ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))) ||
        !ok(invoke(list, 139, arg(allocator), 0))) return 7;
    D3D12_VIEWPORT views[] = {{0, 0, 4, 4, -0.5f, 1.5f}, {0, 0, -4, 4, 0, 1}};
    D3D12_RECT rect{10, 0, 2, 4};
    invoke(list, 21, 2, arg(views));
    invoke(list, 22, 1, arg(&rect));
    invoke(list, 50, 1, arg(&rtv), FALSE, 0);
    const float color[] = {1, 0, 0, 1};
    invoke(list, 52, rtv.ptr, arg(color));
    if (!ok(invoke(list, 9))) return 8;
    void* lists[] = {list};
    invoke(queue, 10, 1, arg(lists));
    if (!ok(invoke(queue, 14, arg(fence), 1))) return 9;
    const auto deadline = GetTickCount64() + 10000;
    while (invoke(fence, 8) < 1) { if (GetTickCount64() >= deadline) return 10; Sleep(1); }
    const auto errors = debug_error_count.load();
    const HRESULT removed = probe.desktop->GetDeviceRemovedReason();
    std::printf("debug errors=%llu removed=0x%08lx\n", errors, static_cast<unsigned long>(removed));
    // Informational: D3D12 documents the individual depth bounds, not their ordering.
    if (!ok(invoke(list, 139, arg(allocator), 0))) return 11;
    D3D12_VIEWPORT inverted{0, 0, 4, 4, 1, 0};
    const auto before = debug_error_count.load();
    invoke(list, 21, 1, arg(&inverted));
    std::printf("inverted depth informational errors=%llu\n", debug_error_count.load() - before);
    invoke(list, 9);
    probe.queue1->UnregisterMessageCallback(probe.cookie);
    return errors == 0 && removed == S_OK && probe.desktop->GetDeviceRemovedReason() == S_OK ? 0 : 70;
}
