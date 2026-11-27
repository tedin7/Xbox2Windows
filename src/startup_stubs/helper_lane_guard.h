#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstring>
#include <map>
#include <mutex>
#include <utility>

namespace x2w_startup {

constexpr bool helper_lane_adapter_reliable(unsigned vendor, unsigned /*device*/, bool software_flag)
{
    return vendor != 0x1414 && !software_flag;
}

inline bool device_helper_lane_reliable(ID3D12Device* device, void (*log)(const char*, ...) = nullptr)
{
    char override_value[7]{};
    const auto length = GetEnvironmentVariableA("X2W_HELPER_LANE", override_value, sizeof(override_value));
    if (length == 5 && std::strcmp(override_value, "trust") == 0) return true;
    if (length == 6 && std::strcmp(override_value, "refuse") == 0) return false;

    using Microsoft::WRL::ComPtr;
    static std::mutex mutex;
    static std::map<std::pair<LONG, DWORD>, bool> verdicts;
    const LUID luid = device->GetAdapterLuid();
    const auto key = std::make_pair(luid.HighPart, luid.LowPart);
    const std::lock_guard<std::mutex> lock(mutex);
    const auto cached = verdicts.find(key);
    if (cached != verdicts.end()) return cached->second;

    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 desc{};
    bool reliable = true;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) &&
        (SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) ||
         SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf())))) &&
        SUCCEEDED(adapter->GetDesc1(&desc)) &&
        desc.AdapterLuid.LowPart == luid.LowPart && desc.AdapterLuid.HighPart == luid.HighPart) {
        reliable = helper_lane_adapter_reliable(desc.VendorId, desc.DeviceId,
                                                (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0);
        if (log)
            log("x2w-draw adapter vendor=0x%04x device=0x%04x IsHelperLane %s",
                desc.VendorId, desc.DeviceId, reliable ? "ok" : "unreliable");
    }
    verdicts.emplace(key, reliable);
    return reliable;
}

}  // namespace x2w_startup
