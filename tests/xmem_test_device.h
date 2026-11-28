// D3D12 device for the xmem working-set tests (tests/xmem_shim_check.cpp,
// tests/cpu/working_set_xmem_check.cpp) and the range translation check (tests/vm_translation_check.cpp).
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cwchar>

// As in offscreen_check: the first D3D12 hardware adapter in high-performance order, so a hybrid
// laptop measures its discrete GPU. With XBOX2WINDOWS_ADAPTER, the first whose description
// contains the value, ignoring case; a value that matches no adapter sets `unmatched`.
// `create(adapter, iid, out)` makes the device; the default is desktop D3D12CreateDevice.
using CreateDeviceFn = HRESULT (*)(IUnknown* adapter, REFIID iid, void** device);

inline HRESULT create_desktop_device(IUnknown* adapter, REFIID iid, void** device)
{
    return D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, iid, device);
}

inline Microsoft::WRL::ComPtr<ID3D12Device> create_device(bool warp, bool& unmatched,
                                                          CreateDeviceFn create = create_desktop_device,
                                                          std::FILE* diagnostics = stdout)
{
    unmatched = false;
    Microsoft::WRL::ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return nullptr;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    if (warp) {
        if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
            create(adapter.Get(), IID_PPV_ARGS(&device));
        }
        return device;
    }
    wchar_t wanted[128]{};
    const DWORD filter = GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", wanted, 128);
    if (filter >= 128) {
        std::fputs("XBOX2WINDOWS_ADAPTER is longer than 127 characters\n", diagnostics);
        unmatched = true;
        return nullptr;
    }
    for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                          IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        if (filter) {
            bool match = false;
            for (const wchar_t* text = desc.Description; *text && !match; ++text)
                match = _wcsnicmp(text, wanted, filter) == 0;
            if (!match) continue;
        }
        if (SUCCEEDED(create(adapter.Get(), IID_PPV_ARGS(&device)))) {
            std::fprintf(diagnostics, "device adapter %04x:%04x %ls\n", desc.VendorId, desc.DeviceId, desc.Description);
            return device;
        }
    }
    if (filter) {
        std::fprintf(diagnostics, "No D3D12 hardware adapter matches XBOX2WINDOWS_ADAPTER=%ls\n", wanted);
        unmatched = true;
    }
    return nullptr;
}
