#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
template<class Result = HRESULT, class... Args> Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}
static std::atomic<unsigned> errors{0};
static void WINAPI on_message(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID, LPCSTR text, void*)
{
    if (severity <= D3D12_MESSAGE_SEVERITY_ERROR && ++errors <= 12) std::printf("debug layer: %.300s\n", text);
}
int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    ComPtr<IDXGIAdapter1> adapter;
    if (!std::strcmp(argv[1], "warp")) {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 3;
    } else if (!std::strcmp(argv[1], "hardware")) {
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i, adapter.Reset()) {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) break;
        }
        if (!adapter) return 77;
    } else return 2;
    DXGI_ADAPTER_DESC1 adapter_desc{}; adapter->GetDesc1(&adapter_desc);
    std::printf("adapter=%ls\n", adapter_desc.Description);
    std::array<unsigned char, 64> parameters{};
    ComPtr<IUnknown> guest;
    ComPtr<ID3D12Device> desktop;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), reinterpret_cast<void**>(guest.GetAddressOf()))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 4;
    ComPtr<ID3D12InfoQueue1> queue;
    DWORD cookie = 0;
    if (FAILED(desktop.As(&queue)) || FAILED(queue->RegisterMessageCallback(on_message, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie))) {
        std::puts("FAIL: no debug layer"); return 1;
    }
    const struct { D3D12_DESCRIPTOR_HEAP_DESC desc; UINT64 backing, bytes; } request{
        {D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 7, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0}, 0, 0};
    ComPtr<IUnknown> heap;
    if (FAILED(invoke(guest.Get(), 14, &request, &__uuidof(ID3D12DescriptorHeap), heap.GetAddressOf()))) return 4;
    const UINT64 dest = invoke<UINT64>(heap.Get(), 9);
    D3D12_SAMPLER_DESC valid{};
    valid.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    valid.AddressU = valid.AddressV = valid.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    valid.MaxLOD = D3D12_FLOAT32_MAX;
    D3D12_SAMPLER_DESC cases[7] = {valid, valid, valid, valid, valid, valid, valid};
    cases[0].Filter = cases[1].Filter = D3D12_FILTER_ANISOTROPIC;
    cases[0].MaxAnisotropy = 0; cases[1].MaxAnisotropy = 64;
    cases[2].MinLOD = 4; cases[2].MaxLOD = 2;
    cases[3].MipLODBias = 20;
    cases[4].Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR;
    cases[5].AddressU = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(0);
    for (auto& sampler : cases) invoke<void>(guest.Get(), 22, &sampler, dest);
    const HRESULT removed = desktop->GetDeviceRemovedReason();
    queue->UnregisterMessageCallback(cookie);
    std::printf("debug errors=%u removed=0x%08lx\n", errors.load(), static_cast<unsigned long>(removed));
    return errors || removed != S_OK ? 1 : 0;
}
