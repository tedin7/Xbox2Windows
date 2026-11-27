#include "startup_stubs/helper_lane_guard.h"

#include <dxgi1_6.h>

#include <cstdio>

using Microsoft::WRL::ComPtr;
using x2w_startup::helper_lane_adapter_reliable;

static_assert(!helper_lane_adapter_reliable(0x1414, 0x008c, false));
static_assert(helper_lane_adapter_reliable(0x10de, 0x2482, false));
static_assert(helper_lane_adapter_reliable(0x1002, 0x73bf, false));
static_assert(!helper_lane_adapter_reliable(0x10de, 0x2482, true));

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    if (!warp && std::strcmp(argv[1], "hardware") != 0) return 2;

    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;

    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    DXGI_ADAPTER_DESC1 desc{};
    if (warp) {
        const HRESULT hr = factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        if (hr == DXGI_ERROR_NOT_FOUND) return 77;
        if (FAILED(hr) || FAILED(adapter->GetDesc1(&desc))) return 3;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))))
            return 4;
    } else {
        for (UINT index = 0;; ++index) {
            const HRESULT hr = factory->EnumAdapterByGpuPreference(
                index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf()));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr) || FAILED(adapter->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))))
                break;
        }
    }

    const bool expected = !warp && desc.VendorId != 0x1414;
    const bool actual = x2w_startup::device_helper_lane_reliable(device.Get());
    std::printf("backend=%s adapter=%ls vendor=0x%04x device=0x%04x IsHelperLane %s (expected %s)\n",
                argv[1], desc.Description, desc.VendorId, desc.DeviceId,
                actual ? "ok" : "unreliable", expected ? "ok" : "unreliable");
    return actual == expected ? 0 : 1;
}
