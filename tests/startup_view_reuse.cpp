// View records are reused (docs/startup-runtime.md#d3d12-view-reuse). A title that keeps creating new views into the
// same descriptor slots must not grow the view table without bound: an id no tracked slot names any more is freed, and
// after the age window a new view takes its record with the next generation. A token of the old view that the title
// copied elsewhere in its own memory then resolves to no view, never to the new one.
//
//   startup_view_reuse <warp|hardware> [control]
//     (X2W_D3D12_BUDGET=1 X2W_D3D12_PERMISSIVE=1 X2W_D3D12_RENDER=1 X2W_D3D12_VIEW_REUSE_SECONDS=0
//      X2W_D3D12_VIEW_REUSE_NOTES=0; control adds X2W_D3D12_VIEW_REUSE=0, so the reuse expectations must fail and the
//      fixture must print FAIL)
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;

struct ViewReuseProbe {
    int token;
    unsigned id, generation;
    unsigned long long records, ids, reused, free_list;
};
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void x2w_d3d12_view_reuse_probe(unsigned long long, ViewReuseProbe*);

namespace {

constexpr UINT slots = 4096;              // the descriptor slots the title keeps rewriting
constexpr UINT views = 2000000;           // distinct views created into them
constexpr unsigned long long slack = 64;  // records beyond one per slot: the stale-token heap's few views

template<class Result = HRESULT, class... Args>
Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

int failures;

void expect(bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
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

ViewReuseProbe probe(UINT64 handle)
{
    ViewReuseProbe p{};
    x2w_d3d12_view_reuse_probe(handle, &p);
    return p;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) ||
        (argc == 3 && std::strcmp(argv[2], "control")))
        return 2;
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // the lines before a crash must reach the test log
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), std::strcmp(argv[1], "warp") == 0);
    if (!adapter) return 77;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    if (SUCCEEDED(adapter->GetDesc1(&adapter_desc))) std::printf("adapter=%ls\n", adapter_desc.Description);
    ComPtr<IUnknown> guest, small_heap, big_heap;
    std::array<unsigned char, 64> parameters{};
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                     reinterpret_cast<void**>(guest.GetAddressOf()))))
        return 3;
    const auto make_heap = [&](UINT count, ComPtr<IUnknown>& heap) {
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            UINT64 backing, bytes;
        } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, count}, 0, 0};
        return SUCCEEDED(invoke(guest.Get(), 14, &heap_desc, &__uuidof(ID3D12DescriptorHeap), heap.GetAddressOf()));
    };
    if (!make_heap(4, small_heap) || !make_heap(slots, big_heap)) return 3;
    const UINT stride = invoke<UINT>(guest.Get(), 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    // A constant buffer view per index: each a distinct view (the stand-in tells views apart by every description byte,
    // so the description is zeroed with memset before its fields are set).
    const auto constant_view = [&](UINT index, UINT64 handle) {
        D3D12_CONSTANT_BUFFER_VIEW_DESC desc;
        std::memset(&desc, 0, sizeof(desc));
        desc.BufferLocation = 0x100000ull + 256ull * index;
        desc.SizeInBytes = 256;
        invoke<void>(guest.Get(), 17, &desc, handle);
    };

    // Stale token: view X in slot 0, its token copied by the title into slot 1 with its own memory copy (nothing tracks
    // that copy). Y and then Z replace X in slot 0; Z takes X's freed record.
    const UINT64 small = invoke<UINT64>(small_heap.Get(), 9);
    const UINT64 slot0 = small, slot1 = small + stride, slot2 = small + 2 * UINT64{stride};
    constant_view(0, slot0);
    const auto x = probe(slot0);
    std::memcpy(reinterpret_cast<void*>(slot1), reinterpret_cast<const void*>(slot0), 32);
    if (x.token != 1 || probe(slot1).token != 1) {
        std::puts("FAIL: the first view or the title's copy of its token does not resolve");
        return 3;
    }
    constant_view(1, slot0);
    constant_view(2, slot0);
    const auto z = probe(slot0), copy = probe(slot1);
    expect(z.token == 1, "the newest view in the slot does not resolve");
    expect(z.id == x.id && z.generation != x.generation, "the third view did not take the first view's freed record");
    expect(copy.token == 0, "the title's copy of the first view's token still resolves after its record was reused");
    // X's content again: a record of its own (its old content no longer maps to the reused id).
    constant_view(0, slot2);
    const auto again = probe(slot2);
    expect(again.token == 1 && again.id != z.id, "the first view created again shares the reused record");

    // Two million distinct views into the same 4096 slots: the table stays at about one record per slot.
    const UINT64 big = invoke<UINT64>(big_heap.Get(), 9);
    const auto before = probe(big);
    for (UINT n = 0; n < views; ++n) constant_view(16 + n, big + UINT64{n % slots} * stride);
    const auto after = probe(big);
    std::printf("view records %llu -> %llu after %u views into %u slots (reused %llu, free list %llu, ids %llu)\n",
                before.records, after.records, views, slots, after.reused, after.free_list, after.ids);
    expect(after.records <= before.records + slots + slack, "the view records grew with views rewritten into the same slots");
    expect(after.ids <= before.ids + slots + slack, "the view id map grew with views rewritten into the same slots");
    expect(probe(big).token == 1 && probe(big + UINT64{slots - 1} * stride).token == 1,
           "the newest views in the rewritten slots do not resolve");

    if (failures) return 5;
    std::puts("view reuse checks passed");
    return 0;
}
