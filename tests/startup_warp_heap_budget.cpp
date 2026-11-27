// WARP headroom probe: which kind of memory counts against WARP's LOCAL video-memory budget, and can any of it count as NON-LOCAL?
//
// The replay player opens two title heaps with OpenExistingHeapFromAddress. On WARP they count as LOCAL usage although they are
// system memory, which leaves less budget for real textures (E_OUTOFMEMORY once usage passes the budget). This probe creates each
// kind of 256 MiB allocation one at a time and prints the change of QueryVideoMemoryInfo LOCAL and NON_LOCAL CurrentUsage.
// It only measures: every line is informational, the test fails only when the adapter cannot be queried at all.
//
// Usage: startup_warp_heap_budget. Exit 77 when WARP is missing.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT64 probe_bytes = 256ull << 20;

struct Usage {
    UINT64 local, local_budget, nonlocal, nonlocal_budget;
};

Usage usage(IDXGIAdapter3* adapter)
{
    Usage result{};
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        result.local = info.CurrentUsage;
        result.local_budget = info.Budget;
    }
    info = {};
    if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
        result.nonlocal = info.CurrentUsage;
        result.nonlocal_budget = info.Budget;
    }
    return result;
}

double mib(UINT64 after, UINT64 before)
{
    return (static_cast<double>(after) - static_cast<double>(before)) / (1024.0 * 1024.0);
}

void report(const char* what, HRESULT hr, const Usage& before, const Usage& after)
{
    if (FAILED(hr)) {
        std::printf("%-40s FAILED hr=0x%08lx\n", what, static_cast<unsigned long>(hr));
        return;
    }
    std::printf("%-40s local %+9.1f MiB  nonlocal %+9.1f MiB  (budget local %.0f MiB, nonlocal %.0f MiB)\n", what,
                mib(after.local, before.local), mib(after.nonlocal, before.nonlocal),
                static_cast<double>(after.local_budget) / (1024.0 * 1024.0),
                static_cast<double>(after.nonlocal_budget) / (1024.0 * 1024.0));
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

void committed(ID3D12Device* device, IDXGIAdapter3* adapter, const char* what, D3D12_HEAP_TYPE type)
{
    const D3D12_HEAP_PROPERTIES props{type};
    const D3D12_RESOURCE_DESC desc = buffer_desc(probe_bytes);
    const D3D12_RESOURCE_STATES state = type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COMMON;
    ComPtr<ID3D12Resource> resource;
    const Usage before = usage(adapter);
    const HRESULT hr = device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&resource));
    report(what, hr, before, usage(adapter));
}

void heap(ID3D12Device* device, IDXGIAdapter3* adapter, const char* what, const D3D12_HEAP_PROPERTIES& props)
{
    D3D12_HEAP_DESC desc{};
    desc.SizeInBytes = probe_bytes;
    desc.Properties = props;
    desc.Flags = D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES;
    ComPtr<ID3D12Heap> object;
    const Usage before = usage(adapter);
    const HRESULT hr = device->CreateHeap(&desc, IID_PPV_ARGS(&object));
    report(what, hr, before, usage(adapter));
}

D3D12_HEAP_PROPERTIES custom(D3D12_CPU_PAGE_PROPERTY page)
{
    D3D12_HEAP_PROPERTIES props{};
    props.Type = D3D12_HEAP_TYPE_CUSTOM;
    props.CPUPageProperty = page;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    return props;
}
}  // namespace

int main()
{
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> warp;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)))) {
        std::puts("SKIP: no WARP adapter");
        return 77;
    }
    ComPtr<IDXGIAdapter3> adapter;
    ComPtr<ID3D12Device> device;
    if (FAILED(warp.As(&adapter)) || FAILED(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
        std::puts("SKIP: WARP device unavailable");
        return 77;
    }
    D3D12_FEATURE_DATA_ARCHITECTURE1 architecture{};
    device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &architecture, sizeof(architecture));
    const Usage start = usage(adapter.Get());
    std::printf("adapter UMA=%d cache-coherent=%d; start: local %.0f of %.0f MiB, nonlocal %.0f of %.0f MiB\n", architecture.UMA,
                architecture.CacheCoherentUMA, static_cast<double>(start.local) / 1048576.0,
                static_cast<double>(start.local_budget) / 1048576.0, static_cast<double>(start.nonlocal) / 1048576.0,
                static_cast<double>(start.nonlocal_budget) / 1048576.0);

    committed(device.Get(), adapter.Get(), "committed buffer, default heap", D3D12_HEAP_TYPE_DEFAULT);
    committed(device.Get(), adapter.Get(), "committed buffer, upload heap", D3D12_HEAP_TYPE_UPLOAD);
    heap(device.Get(), adapter.Get(), "CreateHeap default", D3D12_HEAP_PROPERTIES{D3D12_HEAP_TYPE_DEFAULT});
    heap(device.Get(), adapter.Get(), "CreateHeap custom L0 write-back", custom(D3D12_CPU_PAGE_PROPERTY_WRITE_BACK));
    heap(device.Get(), adapter.Get(), "CreateHeap custom L0 write-combine", custom(D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE));
    heap(device.Get(), adapter.Get(), "CreateHeap custom L0 not-available", custom(D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE));

    ComPtr<ID3D12Device3> device3;
    void* memory = VirtualAlloc(nullptr, probe_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (memory && SUCCEEDED(device.As(&device3))) {
        for (UINT64 at = 0; at < probe_bytes; at += 4096) static_cast<volatile char*>(memory)[at] = 1;
        const Usage before = usage(adapter.Get());
        ComPtr<ID3D12Heap> opened;
        const HRESULT hr = device3->OpenExistingHeapFromAddress(memory, IID_PPV_ARGS(&opened));
        report("OpenExistingHeapFromAddress (player)", hr, before, usage(adapter.Get()));
        if (SUCCEEDED(hr)) {
            D3D12_HEAP_DESC desc = opened->GetDesc();
            std::printf("  opened heap: size %llu MiB, type %d, cpu-page %d, pool %d\n", desc.SizeInBytes >> 20, desc.Properties.Type,
                        desc.Properties.CPUPageProperty, desc.Properties.MemoryPoolPreference);
        }
    } else {
        std::puts("OpenExistingHeapFromAddress (player)     SKIPPED: no allocation or no ID3D12Device3");
    }
    if (memory) VirtualFree(memory, 0, MEM_RELEASE);
    std::puts("warp heap budget probe done");
    return 0;
}
