#include "startup_stubs/resource_budget.h"

#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <cwchar>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_budget_snapshot(x2w_budget::Snapshot*, SIZE_T);

namespace {
using namespace x2w_budget;
constexpr UINT64 page = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

template<class Result = HRESULT, class... Args>
Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

template<class T>
ComPtr<T> desktop_object(IUnknown* guest, ID3D12Device* device, IUnknown* object)
{
    // Same temporary private-data bridge as the slot 148 fixture. Clear it before releasing the wrapper.
    constexpr GUID key{0x0e34d52f, 0xd5a5, 0x4790, {0x91, 0xd4, 0x11, 0xa6, 0x7c, 0xbc, 0x2b, 0xc1}};
    ComPtr<T> output;
    if (FAILED(invoke(guest, 5, &key, object))) return output;
    ComPtr<IUnknown> reference;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT hr = device->GetPrivateData(key, &bytes, reference.GetAddressOf());
    invoke(guest, 5, &key, static_cast<IUnknown*>(nullptr));
    if (SUCCEEDED(hr)) reference.As(&output);
    return output;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes)
{
    return {D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
}

bool require(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) std::printf("FAIL: %s hr=0x%08lx\n", stage, static_cast<unsigned long>(hr));
    return SUCCEEDED(hr);
}

bool snap(Snapshot& counts)
{
    if (x2w_d3d12_budget_snapshot(&counts, sizeof(counts))) return true;
    std::puts("FAIL: budget snapshot unavailable");
    return false;
}

bool same(Counts a, Counts b) { return a.bytes == b.bytes && a.objects == b.objects; }

bool matches(const Snapshot& expected, const Snapshot& actual, const char* stage)
{
    bool pass = expected.unmeasured == actual.unmeasured;
    if (!pass) std::printf("FAIL: %s unmeasured=%llu expected=%llu\n", stage, actual.unmeasured, expected.unmeasured);
    for (unsigned i = 0; i < category_count; ++i) {
        if (!same(expected.live[i], actual.live[i]) || !same(expected.retained[i], actual.retained[i])) {
            const auto a = actual.live[i], b = expected.live[i], c = actual.retained[i], d = expected.retained[i];
            std::printf("FAIL: %s %s live=%llu/%llu expected=%llu/%llu retained=%llu/%llu expected=%llu/%llu\n",
                        stage, names[i], a.bytes, a.objects, b.bytes, b.objects, c.bytes, c.objects, d.bytes, d.objects);
            pass = false;
        }
    }
    return pass;
}

ComPtr<IDXGIAdapter1> pick_adapter(IDXGIFactory4* factory, bool warp)
{
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        return adapter;
    }
    wchar_t name[128]{};
    GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", name, static_cast<DWORD>(std::size(name)));
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            (!name[0] || std::wcsstr(desc.Description, name)) &&
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)))
            return adapter;
    }
    return nullptr;
}

int check(IUnknown* guest, ID3D12Device* device, ID3D12InfoQueue* debug, bool control)
{
    constexpr auto allocation_size = [](ID3D12Device* d, const D3D12_RESOURCE_DESC& desc) {
        return d->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes;
    };
    Snapshot baseline{};
    if (!snap(baseline)) return 4;
    Snapshot active = baseline, retired = baseline;
    const auto measure = [&](Category category, UINT64 bytes, bool guest_owned = true) {
        active.live[category].bytes += bytes;
        ++active.live[category].objects;
        if (guest_owned) {
            retired.retained[category].bytes += bytes;
            ++retired.retained[category].objects;
        }
    };
    const auto resource_bytes = [&](IUnknown* child) {
        const auto native = desktop_object<ID3D12Resource>(guest, device, child);
        return native ? allocation_size(device, native->GetDesc()) : UINT64{0};
    };

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    const auto b = buffer_desc(8192);
    const D3D12_RESOURCE_DESC t{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 32, 16, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
        {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    const GUID& resource_iid = __uuidof(ID3D12Resource);
    ComPtr<IUnknown> committed_texture, committed_buffer, readback_buffer;
    if (!require(invoke(guest, 127, &default_heap, D3D12_HEAP_FLAG_NONE, &t, D3D12_RESOURCE_STATE_COMMON,
                        nullptr, nullptr, 0u, nullptr, &resource_iid, committed_texture.GetAddressOf()), "slot 127 texture")
        || !require(invoke(guest, 27, &default_heap, D3D12_HEAP_FLAG_NONE, &b, D3D12_RESOURCE_STATE_COMMON,
                           nullptr, &resource_iid, committed_buffer.GetAddressOf()), "slot 27 buffer")
        || !require(invoke(guest, 27, &readback_heap, D3D12_HEAP_FLAG_NONE, &b, D3D12_RESOURCE_STATE_COPY_DEST,
                           nullptr, &resource_iid, readback_buffer.GetAddressOf()), "slot 27 readback")) return 4;
    measure(texture, resource_bytes(committed_texture.Get()));
    const auto buffer_bytes = resource_bytes(committed_buffer.Get());
    measure(buffer, buffer_bytes);
    measure(readback, resource_bytes(readback_buffer.Get()));

    // Two buffers share one real heap. A second real heap supports the placed-texture case on tier 1 too.
    const auto make_heap = [&](UINT64 bytes, D3D12_HEAP_FLAGS flags, ComPtr<IUnknown>& out) {
        const D3D12_HEAP_DESC desc{bytes, default_heap, 0, flags};
        if (!require(invoke(guest, 28, &desc, &__uuidof(ID3D12Heap), out.GetAddressOf()), "slot 28 heap")) return false;
        const auto native = desktop_object<ID3D12Heap>(guest, device, out.Get());
        if (!native) return false;
        measure(heap, native->GetDesc().SizeInBytes);
        return true;
    };
    const auto place = [&](IUnknown* owner, UINT64 offset, const D3D12_RESOURCE_DESC& desc, ComPtr<IUnknown>& out) {
        if (!require(invoke(guest, 29, owner, offset, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                            &resource_iid, out.GetAddressOf()), "slot 29 placement")) return false;
        const auto bytes = resource_bytes(out.Get());
        if (!bytes) return false;
        measure(placed, bytes);
        return true;
    };
    ComPtr<IUnknown> buffer_heap, texture_heap, first, second, placed_texture;
    if (!make_heap(2 * page, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, buffer_heap)
        || !make_heap(page, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES, texture_heap)
        || !place(buffer_heap.Get(), 0, b, first) || !place(buffer_heap.Get(), page, b, second)
        || !place(texture_heap.Get(), 0, t, placed_texture)) return 4;

    // The unaligned offset forces the documented slot 133 committed backing on both adapters.
    auto* memory = static_cast<char*>(VirtualAlloc(nullptr, static_cast<SIZE_T>(page), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!memory) return 4;
    ComPtr<IUnknown> copied_buffer, copied_texture;
    if (!require(invoke(guest, 133, memory + 16, &b, D3D12_RESOURCE_STATE_COMMON, nullptr, &resource_iid,
                        copied_buffer.GetAddressOf()), "slot 133 buffer")
        || !require(invoke(guest, 133, memory, &t, D3D12_RESOURCE_STATE_COMMON, nullptr, &resource_iid,
                           copied_texture.GetAddressOf()), "slot 133 texture")) return 4;
    measure(buffer, resource_bytes(copied_buffer.Get()));
    measure(texture, resource_bytes(copied_texture.Get()));

    ComPtr<IUnknown> mapped_first, mapped_second;
    D3D12_FEATURE_DATA_EXISTING_HEAPS existing{};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &existing, sizeof(existing))) && existing.Supported) {
        // Keep this range until exit because the shim caches its opened heap.
        auto* mapped = static_cast<char*>(VirtualAlloc(nullptr, static_cast<SIZE_T>(2 * page),
                                                      MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!mapped) return 4;
        ComPtr<ID3D12Device3> device3;
        ComPtr<ID3D12Heap> probe;
        if (!require(device->QueryInterface(IID_PPV_ARGS(&device3)), "existing heap device")
            || !require(device3->OpenExistingHeapFromAddress(mapped, IID_PPV_ARGS(&probe)), "existing heap probe")) return 4;
        const auto bytes = probe->GetDesc().SizeInBytes;
        probe.Reset();
        const auto place_mapped = [&](char* address, ComPtr<IUnknown>& out) {
            if (!require(invoke(guest, 133, address, &b, D3D12_RESOURCE_STATE_COMMON, nullptr, &resource_iid,
                                out.GetAddressOf()), "slot 133 mapped placement")) return false;
            const auto native = desktop_object<ID3D12Resource>(guest, device, out.Get());
            if (!native || !(native->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER)) {
                std::puts("FAIL: slot 133 fell back on an existing heap");
                return false;
            }
            measure(placed, allocation_size(device, native->GetDesc()));
            return true;
        };
        if (!place_mapped(mapped, mapped_first) || !place_mapped(mapped + page, mapped_second)) return 4;
        measure(heap, bytes);
        std::printf("slot 133 shared mapped heap bytes=%llu\n", bytes);
    } else {
        std::puts("slot 133 mapped-heap subcase unavailable: D3D12_FEATURE_EXISTING_HEAPS");
    }

    const D3D12_COMMAND_QUEUE_DESC q{};
    const auto type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<IUnknown> queue, allocator, next_allocator, list, fence;
    if (!require(invoke(guest, 8, &q, &__uuidof(ID3D12CommandQueue), queue.GetAddressOf()), "queue")
        || !require(invoke(guest, 9, type, &__uuidof(ID3D12CommandAllocator), allocator.GetAddressOf()), "allocator")
        || !require(invoke(guest, 9, type, &__uuidof(ID3D12CommandAllocator), next_allocator.GetAddressOf()),
                    "next allocator")
        || !require(invoke(guest, 73, &type, allocator.Get(), nullptr, &__uuidof(ID3D12GraphicsCommandList),
                           list.GetAddressOf()), "list")
        || !require(invoke(guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), fence.GetAddressOf()), "fence")
        || !require(invoke(list.Get(), 139, allocator.Get(), nullptr), "list reset")) return 4;
    auto* a = invoke<char*>(list.Get(), 148, UINT64{0x280}, UINT64{0x10});
    auto* c = invoke<char*>(list.Get(), 148, UINT64{0x40}, UINT64{0x10});
    if (!a || !c || a == c) return 4;
    a[0] = 1;
    c[0] = 2;
    measure(upload, allocation_size(device, buffer_desc(page)), false);

    Snapshot actual{};
    if (!snap(actual) || !matches(active, actual, "allocated")) return 5;
    for (unsigned i = 0; i < category_count; ++i) {
        const auto bytes = active.live[i].bytes - baseline.live[i].bytes;
        const auto objects = active.live[i].objects - baseline.live[i].objects;
        if (!bytes || !objects) return 5;
        std::printf("allocated %s bytes=%llu objects=%llu\n", names[i], bytes, objects);
    }

    if (!require(invoke(list.Get(), 9), "list close")) return 4;
    IUnknown* lists[] = {list.Get()};
    invoke<void>(queue.Get(), 10, 1u, lists);
    if (!require(invoke(queue.Get(), 14, fence.Get(), UINT64{1}), "queue signal")) return 4;
    const auto real_fence = desktop_object<ID3D12Fence>(guest, device, fence.Get());
    if (!real_fence) return 4;
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!done) return 4;
    const bool flushed = SUCCEEDED(real_fence->SetEventOnCompletion(1, done)) &&
        WaitForSingleObject(done, 5000) == WAIT_OBJECT_0 && real_fence->GetCompletedValue() == 1;
    CloseHandle(done);
    // A closed list can be submitted again. Move it to another allocator before retiring its old pages.
    if (!flushed || !require(invoke(list.Get(), 139, next_allocator.Get(), nullptr), "list reset on next allocator")
        || !require(invoke(list.Get(), 9), "replacement list close")
        || !require(invoke(allocator.Get(), 8), "allocator reset after completion")) return 4;

    // Let two child placements keep the heap live after the final guest heap Release.
    buffer_heap.Reset();
    if (!snap(actual)) return 5;
    active.live[upload] = baseline.live[upload];
    if (!matches(active, actual, "allocator reset and shared heap released")) return 5;
    const auto first_bytes = resource_bytes(first.Get());
    first.Reset();
    active.live[placed].bytes -= first_bytes;
    --active.live[placed].objects;
    active.retained[placed].bytes += first_bytes;
    ++active.retained[placed].objects;
    if (!snap(actual) || !matches(active, actual, "second placement keeps shared heap live")) return 5;
    second.Reset();
    placed_texture.Reset();
    texture_heap.Reset();
    copied_buffer.Reset();
    copied_texture.Reset();
    mapped_first.Reset();
    mapped_second.Reset();
    VirtualFree(memory, 0, MEM_RELEASE);
    committed_texture.Reset();
    readback_buffer.Reset();
    if (!control) committed_buffer.Reset();
    if (!snap(actual)) return 5;

    const UINT64 errors = debug ? debug->GetNumStoredMessages() : 0;
    std::printf("debug_errors_or_warnings=%llu\n", errors);
    if (errors || FAILED(device->GetDeviceRemovedReason())) return 4;
    if (control) {
        Snapshot skipped = retired;
        skipped.live[buffer].bytes += buffer_bytes;
        ++skipped.live[buffer].objects;
        skipped.retained[buffer].bytes -= buffer_bytes;
        --skipped.retained[buffer].objects;
        if (!matches(skipped, actual, "control isolation") || same(retired.live[buffer], actual.live[buffer])) return 5;
        matches(retired, actual, "released");
        std::puts("FAIL: inline-control missed Release left buffer above baseline");
        return 6;
    }
    if (!matches(retired, actual, "released")) return 5;
    std::puts("all live categories returned to baseline");
    return 0;
}
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware"))
        || (argc == 3 && std::strcmp(argv[2], "inline-control"))) return 2;
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), std::strcmp(argv[1], "warp") == 0);
    if (!adapter) return 77;
    char debug_env[2]{};
    const bool debug_on = GetEnvironmentVariableA("X2W_D3D12_DEBUG", debug_env, sizeof(debug_env)) && debug_env[0] == '1';
    if (debug_on) {
        ComPtr<ID3D12Debug> debug;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) return 77;
    }
    ComPtr<IUnknown> guest;
    ComPtr<ID3D12Device> device;
    std::array<unsigned char, 64> parameters{};
    if (!require(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                      reinterpret_cast<void**>(guest.GetAddressOf())), "Xbox device")
        || !require(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "desktop device")) return 3;
    ComPtr<ID3D12InfoQueue> debug;
    if (debug_on && FAILED(device.As(&debug))) return 3;
    if (debug) {
        D3D12_MESSAGE_SEVERITY severities[] = {
            D3D12_MESSAGE_SEVERITY_CORRUPTION, D3D12_MESSAGE_SEVERITY_ERROR, D3D12_MESSAGE_SEVERITY_WARNING};
        D3D12_INFO_QUEUE_FILTER filter{};
        filter.AllowList.NumSeverities = static_cast<UINT>(std::size(severities));
        filter.AllowList.pSeverityList = severities;
        if (FAILED(debug->AddStorageFilterEntries(&filter))) return 3;
        debug->ClearStoredMessages();
    }
    DXGI_ADAPTER_DESC1 desc{};
    LARGE_INTEGER version{};
    adapter->GetDesc1(&desc);
    adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &version);
    const auto driver = static_cast<UINT64>(version.QuadPart);
    std::printf("adapter=%ls driver=%llu.%llu.%llu.%llu debug_layer=%s\n", desc.Description,
                driver >> 48, (driver >> 32) & 0xffff, (driver >> 16) & 0xffff, driver & 0xffff,
                debug ? "enabled" : "off");
    return check(guest.Get(), device.Get(), debug.Get(), argc == 3);
}
