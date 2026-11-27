// Synthetic debug-layer regression through the startup device wrapper.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) std::size_t x2w_d3d12_resource_map_count();

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
    bool watching = false;
    bool watch()
    {
        watching = SUCCEEDED(desktop.As(&queue1)) &&
            SUCCEEDED(queue1->RegisterMessageCallback(on_message, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie));
        return watching;
    }
    ~Probe() { if (watching) queue1->UnregisterMessageCallback(cookie); }
};

bool check_resource_maps(void* device)
{
    constexpr unsigned buffer_count = 64;
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    const D3D12_RANGE empty{0, 0};
    const auto create_buffer = [&](ComPtr<ID3D12Resource>& buffer) {
        return ok(invoke(device, 27, arg(&heap), D3D12_HEAP_FLAG_NONE, arg(&desc),
                         D3D12_RESOURCE_STATE_GENERIC_READ, 0, arg(&__uuidof(ID3D12Resource)),
                         arg(buffer.GetAddressOf()))) && buffer.Get();
    };
    const auto map = [&](ID3D12Resource* buffer) {
        void* data = nullptr;
        return SUCCEEDED(buffer->Map(0, &empty, &data)) && data;
    };

    const auto before = x2w_d3d12_resource_map_count();
    for (unsigned i = 0; i < buffer_count; ++i) {
        ComPtr<ID3D12Resource> buffer;
        if (!create_buffer(buffer) || !map(buffer.Get())) {
            std::printf("persistent upload buffer %u: create or Map failed\n", i);
            return false;
        }
        if (i == 0) {
            auto extra_ref = buffer;
            extra_ref.Reset();
            if (x2w_d3d12_resource_map_count() != before + 1) {
                std::printf("live upload buffer missing from map census after non-final Release\n");
                return false;
            }
        }
        // Release while still mapped.
    }
    const auto after = x2w_d3d12_resource_map_count();
    std::printf("map census after %u persistent releases: %zu -> %zu\n", buffer_count, before, after);
    if (after != before) return false;

    ComPtr<ID3D12Resource> buffer;
    if (!create_buffer(buffer)) return false;
    buffer->Unmap(0, &empty);
    for (unsigned i = 0; i < 2; ++i) {
        if (!map(buffer.Get())) return false;
        buffer->Unmap(0, &empty);
        if (x2w_d3d12_resource_map_count() != before) {
            std::printf("paired upload Map/Unmap %u left a census entry\n", i);
            return false;
        }
    }
    return true;
}
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

    if (!probe.watch()) { std::printf("no debug layer\n"); return 5; }
    const bool resource_maps_ok = check_resource_maps(device);

    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC texture{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 32, 32, 1, 1,
        DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    D3D12_CLEAR_VALUE clear{}; clear.Format = texture.Format;
    void* tex = nullptr;
    if (!ok(invoke(device, 27, arg(&heap), 0, arg(&texture), D3D12_RESOURCE_STATE_COMMON,
                   arg(&clear), arg(&__uuidof(ID3D12Resource)), arg(&tex)))) return 6;
    auto depth = texture; depth.Format = DXGI_FORMAT_D32_FLOAT; depth.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    clear.Format = DXGI_FORMAT_D32_FLOAT; clear.DepthStencil.Depth = 2.0f;
    void* ds = nullptr;
    if (!ok(invoke(device, 27, arg(&heap), 0, arg(&depth), D3D12_RESOURCE_STATE_DEPTH_WRITE,
                   arg(&clear), arg(&__uuidof(ID3D12Resource)), arg(&ds)))) return 7;
    D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 65536, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    void* buf = nullptr;
    if (!ok(invoke(device, 27, arg(&heap), 0, arg(&buffer), D3D12_RESOURCE_STATE_COMMON,
                   arg(&clear), arg(&__uuidof(ID3D12Resource)), arg(&buf)))) return 8;
    void* data = nullptr;
    if (SUCCEEDED(static_cast<HRESULT>(invoke(tex, 8, 0, 0, arg(&data))))) return 9;
    invoke(tex, 9, 0, 0);
    struct HeapRequest { D3D12_DESCRIPTOR_HEAP_DESC desc; std::uint64_t address, bytes; } request{};
    request.desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    request.desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE; request.desc.NumDescriptors = 1048577;
    void* view = nullptr;
    if (!ok(invoke(device, 14, arg(&request.desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&view)))) return 10;
    request.desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER; request.desc.NumDescriptors = 4096;
    void* sampler = nullptr;
    if (!ok(invoke(device, 14, arg(&request.desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&sampler)))) return 11;
    request.desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV; request.desc.NumDescriptors = 1;
    void* rtv = nullptr;
    if (!ok(invoke(device, 14, arg(&request.desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&rtv)))) return 12;
    const HRESULT removed = probe.desktop->GetDeviceRemovedReason();
    std::printf("device removed: 0x%08lx errors: %llu\n", static_cast<unsigned long>(removed), debug_error_count.load());
    return resource_maps_ok && removed == S_OK && debug_error_count.load() == 0 ? 0 : 70;
}
