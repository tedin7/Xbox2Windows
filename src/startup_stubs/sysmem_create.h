#pragma once

#include "sysmem_policy.h"

#include <windows.h>
#include <d3d12.h>

#include <cstdint>

namespace x2w_sysmem {

// X2W_D3D12_SYSMEM_FALLBACK=1 (opt in): a default-heap texture that fails with E_OUTOFMEMORY is created once more in
// system memory, so the title gets a resource instead of a null one.
inline bool switch_on()
{
    static const bool on = [] {
        char value[2]{};
        return GetEnvironmentVariableA("X2W_D3D12_SYSMEM_FALLBACK", value, sizeof(value)) == 1 && value[0] == '1';
    }();
    return on;
}

// Creates with `create(heap_properties)` (a callable returning HRESULT that fills the caller's resource pointer on
// success). On E_OUTOFMEMORY of a default heap, with the switch on, retries in a custom system-memory heap whose page
// property fits the adapter. A failed retry returns the first failure, so the caller sees what it saw before.
// `used` reports that the retry produced the resource. `enabled` is the switch; the tests pass their own.
template<class Create>
HRESULT create_with_fallback(ID3D12Device* device, const D3D12_HEAP_PROPERTIES& props, Create&& create, bool& used,
                             bool enabled = switch_on())
{
    used = false;
    const HRESULT first = create(props);
    if (!wants_fallback(first, enabled, props.Type == D3D12_HEAP_TYPE_DEFAULT) || !device) return first;
    bool uma = false;
    D3D12_FEATURE_DATA_ARCHITECTURE1 architecture{};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &architecture, sizeof(architecture)))) {
        uma = architecture.UMA != FALSE;
    } else {
        D3D12_FEATURE_DATA_ARCHITECTURE legacy{};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &legacy, sizeof(legacy)))) return first;
        uma = legacy.UMA != FALSE;
    }
    const Candidates candidates = candidate_pages(uma);
    for (int i = 0; i < candidates.count; ++i) {
        D3D12_HEAP_PROPERTIES system{};
        system.Type = D3D12_HEAP_TYPE_CUSTOM;
        system.CPUPageProperty = static_cast<D3D12_CPU_PAGE_PROPERTY>(candidates.pages[i]);
        system.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        system.CreationNodeMask = props.CreationNodeMask;
        system.VisibleNodeMask = props.VisibleNodeMask;
        if (SUCCEEDED(create(system))) {
            used = true;
            return S_OK;
        }
    }
    return first;
}

}  // namespace x2w_sysmem
