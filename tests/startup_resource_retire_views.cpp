// Live SRVs keep retired textures available for later bindings (#2379).
//   startup_resource_retire_views <warp|hardware> [keep]
// `keep` runs without base retirement, as a control for the budget drop after explicit overwrite.
// The legacy views switch may be on or off. Both settings preserve the original and copied slots.
// The fixture uses the shim's descriptor creation and copying paths and reads the desktop descriptors
// with a compute probe. The gated queue and the two recorded copies cover pending GPU work.
#include "startup_stubs/resource_budget.h"

#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <deque>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_budget_snapshot(x2w_budget::Snapshot*, SIZE_T);

namespace {
using namespace x2w_budget;

constexpr UINT side = 64, pitch = 256, image_bytes = side * pitch;
constexpr DWORD wait_ms = 10000;
constexpr int settle_limit = 200;
constexpr std::uint32_t view_token_magic = 0x56573258;  // "X2WV", d3d12_capture.inl

template<class Result = HRESULT, class... Args>
Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

template<class T>
ComPtr<T> desktop_object(IUnknown* guest, ID3D12Device* device, IUnknown* object)
{
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

bool require(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) std::printf("FAIL: %s hr=0x%08lx\n", stage, static_cast<unsigned long>(hr));
    return SUCCEEDED(hr);
}

std::uint8_t pattern_byte(int seed, UINT i) { return static_cast<std::uint8_t>(i * 7 + seed * 13 + 1); }

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

struct Res {
    ComPtr<IUnknown> guest;
    const char* name = "";
    UINT64 bytes = 0;
    bool released = false;
    bool freed = false;
};

struct GuestList {
    ComPtr<IUnknown> allocator, list;
};

struct GuestQueue {
    ComPtr<IUnknown> queue, done;
    ComPtr<ID3D12Fence> done_real;
    UINT64 value = 0;
};

struct GuestHeap {
    ComPtr<IUnknown> guest;
    ComPtr<ID3D12DescriptorHeap> real;
    UINT64 cpu = 0, gpu = 0;
    bool visible = false;
};

constexpr char probe_shader[] = R"(
Texture2D<float4> source : register(t0);
RWByteAddressBuffer result : register(u0);
[numthreads(1, 1, 1)] void main()
{
    result.Store4(0, asuint(source.Load(int3(0, 0, 0))));
}
)";

struct Context {
    IUnknown* guest;
    ID3D12Device* device;
    ComPtr<ID3D12CommandQueue> own_queue;
    ComPtr<ID3D12CommandAllocator> own_allocator;
    ComPtr<ID3D12GraphicsCommandList> own_list;
    ComPtr<ID3D12Fence> own_fence;
    UINT64 own_value = 0;
    HANDLE event = nullptr;
    ComPtr<ID3D12Resource> upload, readback, probe_out;
    ComPtr<ID3D12RootSignature> probe_root;
    ComPtr<ID3D12PipelineState> probe_pso;
    ComPtr<ID3D12DescriptorHeap> probe_heap;
    UINT increment = 0;
    Snapshot base{};
    std::deque<Res> pool;
    GuestQueue direct, gated;
    GuestList drain_list;
    D3D12_RESOURCE_DESC desc{};
    UINT64 texture_bytes = 0;
    std::array<GUID, 1> iid_resource{__uuidof(ID3D12Resource)};

    bool wait(ID3D12Fence* fence, UINT64 value)
    {
        if (fence->GetCompletedValue() >= value) return true;
        if (FAILED(fence->SetEventOnCompletion(value, event))) return false;
        return WaitForSingleObject(event, wait_ms) == WAIT_OBJECT_0;
    }
    bool own_run()
    {
        if (!require(own_list->Close(), "own list close")) return false;
        ID3D12CommandList* lists[] = {own_list.Get()};
        own_queue->ExecuteCommandLists(1, lists);
        if (!require(own_queue->Signal(own_fence.Get(), ++own_value), "own signal")) return false;
        if (wait(own_fence.Get(), own_value)) return true;
        std::puts("FAIL: own queue timed out");
        return false;
    }
    D3D12_TEXTURE_COPY_LOCATION footprint(ID3D12Resource* buffer) const
    {
        D3D12_TEXTURE_COPY_LOCATION location{buffer, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        location.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, side, side, 1, pitch};
        return location;
    }
    static D3D12_TEXTURE_COPY_LOCATION footprint_tex(ID3D12Resource* tex)
    {
        return {tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    }
    bool make_queue(GuestQueue& q)
    {
        const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
        if (!require(invoke(guest, 8, &qd, &__uuidof(ID3D12CommandQueue), q.queue.GetAddressOf()), "queue") ||
            !require(invoke(guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), q.done.GetAddressOf()), "fence"))
            return false;
        q.done_real = desktop_object<ID3D12Fence>(guest, device, q.done.Get());
        return q.done_real != nullptr;
    }
    bool sync(GuestQueue& q)
    {
        if (!require(invoke(q.queue.Get(), 14, q.done.Get(), ++q.value), "queue signal")) return false;
        if (wait(q.done_real.Get(), q.value)) return true;
        std::puts("FAIL: guest queue timed out");
        return false;
    }
    bool make_list(GuestList& l)
    {
        const unsigned list_desc[] = {static_cast<unsigned>(D3D12_COMMAND_LIST_TYPE_DIRECT), 2, 1, 0};
        return require(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator),
                              l.allocator.GetAddressOf()), "allocator") &&
               require(invoke(guest, 73, list_desc, l.allocator.Get(), nullptr,
                              &__uuidof(ID3D12GraphicsCommandList), l.list.GetAddressOf()), "command list") &&
               require(invoke(l.list.Get(), 139, l.allocator.Get(), nullptr), "list reset");
    }
    bool close(GuestList& l) { return require(invoke(l.list.Get(), 9), "list close"); }
    HRESULT reset_list(GuestList& l)
    {
        ComPtr<IUnknown> next;
        if (!require(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator),
                            next.GetAddressOf()), "allocator"))
            return E_FAIL;
        const HRESULT hr = invoke(l.list.Get(), 139, next.Get(), nullptr);
        if (SUCCEEDED(hr)) l.allocator = next;
        return hr;
    }
    void execute(GuestQueue& q, GuestList& l)
    {
        IUnknown* lists[] = {l.list.Get()};
        invoke<void>(q.queue.Get(), 10, 1u, lists);
    }
    Res& make_committed(const char* name)
    {
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC d = desc;
        Res& r = pool.emplace_back();
        r.name = name;
        r.bytes = texture_bytes;
        if (!require(invoke(guest, 127, &heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, 0u,
                            nullptr, &iid_resource[0], r.guest.GetAddressOf()), name))
            throw 4;
        return r;
    }
    bool fill_desktop(IUnknown* resource, int seed)
    {
        const auto real = desktop_object<ID3D12Resource>(guest, device, resource);
        if (!real) return false;
        void* mapped = nullptr;
        const D3D12_RANGE none{};
        if (!require(upload->Map(0, &none, &mapped), "upload map")) return false;
        auto* dst = static_cast<std::uint8_t*>(mapped);
        for (UINT i = 0; i < image_bytes; ++i) dst[i] = pattern_byte(seed, i);
        upload->Unmap(0, nullptr);
        if (!require(own_allocator->Reset(), "own allocator") || !require(own_list->Reset(own_allocator.Get(), nullptr), "own list"))
            return false;
        D3D12_TEXTURE_COPY_LOCATION tex_dst = footprint_tex(real.Get());
        D3D12_TEXTURE_COPY_LOCATION tex_src = footprint(upload.Get());
        own_list->CopyTextureRegion(&tex_dst, 0, 0, 0, &tex_src, nullptr);
        D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE, {}};
        b.Transition = {real.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COMMON};
        own_list->ResourceBarrier(1, &b);
        return own_run();
    }
    bool holds(IUnknown* resource, int seed, const char* what)
    {
        const auto real = desktop_object<ID3D12Resource>(guest, device, resource);
        if (!real) return false;
        if (!require(own_allocator->Reset(), "own allocator") || !require(own_list->Reset(own_allocator.Get(), nullptr), "own list"))
            return false;
        D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE, {}};
        b.Transition = {real.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_SOURCE};
        own_list->ResourceBarrier(1, &b);
        D3D12_TEXTURE_COPY_LOCATION tex_dst = footprint(readback.Get());
        D3D12_TEXTURE_COPY_LOCATION tex_src = footprint_tex(real.Get());
        own_list->CopyTextureRegion(&tex_dst, 0, 0, 0, &tex_src, nullptr);
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        own_list->ResourceBarrier(1, &b);
        if (!own_run()) return false;
        void* mapped = nullptr;
        const D3D12_RANGE all{0, image_bytes};
        if (!require(readback->Map(0, &all, &mapped), "readback map")) return false;
        const auto* src = static_cast<const std::uint8_t*>(mapped);
        UINT mismatches = 0;
        for (UINT y = 0; y < side; ++y)
            for (UINT x = 0; x < side * 4; ++x)
                if (src[y * pitch + x] != pattern_byte(seed, y * pitch + x)) ++mismatches;
        readback->Unmap(0, nullptr);
        if (mismatches) {
            std::printf("FAIL: %s readback had %u mismatched bytes\n", what, mismatches);
            return false;
        }
        return true;
    }
    void release(Res& r)
    {
        r.guest.Reset();
        r.released = true;
    }
    void drain_once()
    {
        IUnknown* lists[] = {drain_list.list.Get()};
        invoke<void>(direct.queue.Get(), 10, 1u, lists);
    }
    bool expect(const char* stage, const char* fail) const
    {
        Snapshot now{};
        if (!x2w_d3d12_budget_snapshot(&now, sizeof(now))) return false;
        UINT64 live_n = base.live[texture].objects, ret_n = base.retained[texture].objects;
        UINT64 live_b = base.live[texture].bytes, ret_b = base.retained[texture].bytes;
        for (const Res& r : pool) {
            if (!r.released) {
                live_b += r.bytes;
                ++live_n;
            } else if (!r.freed) {
                ret_b += r.bytes;
                ++ret_n;
            }
        }
        const auto& got_live = now.live[texture];
        const auto& got_ret = now.retained[texture];
        if (got_live.bytes == live_b && got_live.objects == live_n && got_ret.bytes == ret_b && got_ret.objects == ret_n)
            return true;
        std::printf("FAIL: %s\n  stage '%s' live=%llu/%llu expected=%llu/%llu retained=%llu/%llu expected=%llu/%llu\n",
                    fail, stage, got_live.bytes, got_live.objects, live_b, live_n,
                    got_ret.bytes, got_ret.objects, ret_b, ret_n);
        return false;
    }
    bool stays(const char* stage, const char* fail)
    {
        drain_once();
        return sync(direct) && expect(stage, fail);
    }
    bool settle(const char* stage, const char* fail)
    {
        for (int i = 0; i < settle_limit; ++i) {
            drain_once();
            Snapshot now{};
            if (!x2w_d3d12_budget_snapshot(&now, sizeof(now))) return false;
            UINT64 ret_n = base.retained[texture].objects;
            for (const Res& r : pool) if (r.released && !r.freed) ++ret_n;
            if (now.retained[texture].objects == ret_n) return expect(stage, fail);
        }
        return expect(stage, fail);
    }
    bool make_heap(GuestHeap& h, UINT count, bool visible)
    {
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            UINT64 backing, bytes;
        } request{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, count,
                   visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE}, 0, 0};
        if (!require(invoke(guest, 14, &request, &__uuidof(ID3D12DescriptorHeap), h.guest.GetAddressOf()), "guest heap"))
            return false;
        h.real = desktop_object<ID3D12DescriptorHeap>(guest, device, h.guest.Get());
        if (!h.real) {
            std::puts("FAIL: guest heap has no desktop object");
            return false;
        }
        h.cpu = invoke<UINT64>(h.guest.Get(), 9);
        h.gpu = visible ? invoke<UINT64>(h.guest.Get(), 10) : 0;
        h.visible = visible;
        return true;
    }
    // Loads texel (0,0) through the desktop descriptor behind slot `index` of the guest heap.
    bool read_slot(const GuestHeap& h, UINT index, float (&texel)[4])
    {
        if (!require(own_allocator->Reset(), "probe allocator") || !require(own_list->Reset(own_allocator.Get(), nullptr), "probe list"))
            return false;
        ID3D12DescriptorHeap* bound = probe_heap.Get();
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = probe_heap->GetGPUDescriptorHandleForHeapStart();
        if (h.visible) {
            bound = h.real.Get();
            gpu = h.real->GetGPUDescriptorHandleForHeapStart();
            gpu.ptr += std::uint64_t{index} * increment;
        } else {
            D3D12_CPU_DESCRIPTOR_HANDLE src = h.real->GetCPUDescriptorHandleForHeapStart();
            src.ptr += std::uint64_t{index} * increment;
            device->CopyDescriptorsSimple(1, probe_heap->GetCPUDescriptorHandleForHeapStart(), src,
                                          D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
        own_list->SetDescriptorHeaps(1, &bound);
        own_list->SetComputeRootSignature(probe_root.Get());
        own_list->SetPipelineState(probe_pso.Get());
        own_list->SetComputeRootDescriptorTable(0, gpu);
        own_list->SetComputeRootUnorderedAccessView(1, probe_out->GetGPUVirtualAddress());
        own_list->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE, {}};
        b.Transition = {probe_out.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE};
        own_list->ResourceBarrier(1, &b);
        own_list->CopyBufferRegion(readback.Get(), 0, probe_out.Get(), 0, 16);
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        own_list->ResourceBarrier(1, &b);
        if (!own_run()) return false;
        void* mapped = nullptr;
        const D3D12_RANGE all{0, 16};
        if (!require(readback->Map(0, &all, &mapped), "probe readback map")) return false;
        std::memcpy(texel, mapped, sizeof(texel));
        readback->Unmap(0, nullptr);
        return true;
    }
    // want_null: the slot must read as null (all zero); otherwise it must read the texture's texels (not all zero).
    bool slot_is(const GuestHeap& h, UINT index, bool want_null, const char* what)
    {
        float texel[4]{};
        if (!read_slot(h, index, texel)) return false;
        const bool zero = texel[0] == 0.0f && texel[1] == 0.0f && texel[2] == 0.0f && texel[3] == 0.0f;
        if (zero == want_null) return true;
        std::printf("FAIL: %s: descriptor reads %.3f %.3f %.3f %.3f, expected %s\n", what, texel[0], texel[1], texel[2],
                    texel[3], want_null ? "null (all zero)" : "texels of the texture");
        return false;
    }
    // The view token the shim wrote into title descriptor memory for a slot (first word).
    static std::uint32_t token_word(const GuestHeap& h, UINT index, UINT stride)
    {
        std::uint32_t word = 0;
        std::memcpy(&word, reinterpret_cast<const void*>(h.cpu + std::uint64_t{index} * stride), sizeof(word));
        return word;
    }
};

D3D12_SHADER_RESOURCE_VIEW_DESC texture_view()
{
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    return srv;
}

void overwrite_srv(Context& c, const GuestHeap& heap, UINT index)
{
    const auto srv = texture_view();
    invoke<void>(c.guest, 18, static_cast<IUnknown*>(nullptr), &srv, heap.cpu + c.increment * static_cast<UINT64>(index));
}

// 1: a later binding of the same slot after the last guest Release.
bool case1_single_view(Context& c, bool retire_on, GuestHeap& visible)
{
    Res& texture = c.make_committed("T1");
    if (!c.fill_desktop(texture.guest.Get(), 1)) return false;
    const auto srv = texture_view();
    invoke<void>(c.guest, 18, texture.guest.Get(), &srv, visible.cpu);
    if (!c.slot_is(visible, 0, false, "case 1: live descriptor") ||
        Context::token_word(visible, 0, c.increment) != view_token_magic) return false;

    c.release(texture);
    for (int i = 0; i < 8; ++i)
        if (!c.stays("T1 with a live slot", "T1 retired before its slot was overwritten")) return false;
    if (!c.slot_is(visible, 0, false, "case 1: rebound descriptor after Release") ||
        Context::token_word(visible, 0, c.increment) != view_token_magic) return false;

    overwrite_srv(c, visible, 0);
    if (!c.slot_is(visible, 0, true, "case 1: explicitly overwritten slot")) return false;
    texture.freed = retire_on;
    return c.settle("T1 after explicit overwrite", "unexpected backing lifetime after overwrite");
}

// 2: a copy in a second heap, release behind a gated queue.
bool case2_copy_and_gate(Context& c, bool retire_on, GuestHeap& visible, GuestHeap& staging)
{
    Res& texture = c.make_committed("T2");
    if (!c.fill_desktop(texture.guest.Get(), 2)) return false;
    const auto srv = texture_view();
    invoke<void>(c.guest, 18, texture.guest.Get(), &srv, visible.cpu + c.increment);
    invoke<void>(c.guest, 24, 1u, staging.cpu, visible.cpu + c.increment, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    ComPtr<IUnknown> gate;
    if (!require(invoke(c.guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), gate.GetAddressOf()), "gate") ||
        !require(invoke(c.gated.queue.Get(), 17, gate.Get(), UINT64{1}), "wait")) return false;

    c.release(texture);
    for (int i = 0; i < 8; ++i)
        if (!c.stays("T2 behind a closed gate", "T2 retired before the release fences passed")) return false;
    if (!c.slot_is(visible, 1, false, "case 2: original before the gate opens") ||
        !c.slot_is(staging, 0, false, "case 2: copy before the gate opens")) return false;
    if (!require(invoke(gate.Get(), 10, UINT64{1}), "signal gate") || !c.sync(c.gated)) return false;
    if (!c.stays("T2 after its release fences", "T2 retired while both descriptors were live") ||
        !c.slot_is(visible, 1, false, "case 2: original after the release fences") ||
        !c.slot_is(staging, 0, false, "case 2: copy after the release fences")) return false;

    overwrite_srv(c, visible, 1);
    if (!c.stays("T2 with only its copy live", "T2 retired while its copy was live") ||
        !c.slot_is(staging, 0, false, "case 2: copy after overwriting the original")) return false;
    overwrite_srv(c, staging, 0);
    if (!c.slot_is(staging, 0, true, "case 2: explicitly overwritten copy")) return false;
    texture.freed = retire_on;
    return c.settle("T2 after both slots were overwritten", "unexpected backing lifetime after overwriting both slots");
}

// 3 and 4: lists that name the texture. Overwriting the slots cannot drop list or submission holds.
bool case3_list_holds(Context& c, bool retire_on, GuestHeap& visible)
{
    Res& T3 = c.make_committed("T3");
    Res& D3 = c.make_committed("D3");
    Res& T4 = c.make_committed("T4");
    Res& D4 = c.make_committed("D4");
    if (!c.fill_desktop(T3.guest.Get(), 3) || !c.fill_desktop(D3.guest.Get(), 99) || !c.fill_desktop(T4.guest.Get(), 4) ||
        !c.fill_desktop(D4.guest.Get(), 99)) return false;
    const auto srv = texture_view();
    invoke<void>(c.guest, 18, T3.guest.Get(), &srv, visible.cpu + 2 * c.increment);
    invoke<void>(c.guest, 18, T4.guest.Get(), &srv, visible.cpu + 3 * c.increment);
    GuestList submitted, discarded;
    if (!c.make_list(submitted) || !c.make_list(discarded)) return false;
    invoke<void>(submitted.list.Get(), 17, D3.guest.Get(), T3.guest.Get());
    invoke<void>(discarded.list.Get(), 17, D4.guest.Get(), T4.guest.Get());
    if (!c.close(submitted) || !c.close(discarded)) return false;

    c.release(T3);
    c.release(T4);
    for (int i = 0; i < 8; ++i)
        if (!c.stays("T3/T4 behind closed lists", "a texture retired while its list was closed")) return false;
    if (!c.slot_is(visible, 2, false, "case 3: slot with a closed list") ||
        !c.slot_is(visible, 3, false, "case 4: slot with a closed list")) return false;

    ComPtr<IUnknown> gate;
    if (!require(invoke(c.guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), gate.GetAddressOf()), "gate3") ||
        !require(invoke(c.gated.queue.Get(), 17, gate.Get(), UINT64{1}), "wait3")) return false;
    c.execute(c.gated, submitted);
    if (!SUCCEEDED(c.reset_list(submitted)) ||
        !c.slot_is(visible, 2, false, "case 3: live slot during an in-flight recording")) return false;
    overwrite_srv(c, visible, 2);
    overwrite_srv(c, visible, 3);
    for (int i = 0; i < 8; ++i)
        if (!c.stays("T3/T4 after overwrite", "a texture retired while its list or submission still held it")) return false;
    if (!SUCCEEDED(c.reset_list(discarded))) return false;
    T4.freed = retire_on;
    if (!c.settle("T4 after its recording was discarded", "unexpected T4 lifetime after discarding the list")) return false;
    if (!require(invoke(gate.Get(), 10, UINT64{1}), "signal gate3") || !c.sync(c.gated)) return false;
    T3.freed = retire_on;
    if (!c.settle("T3 after its submission's fence", "unexpected T3 lifetime after the fence")) return false;
    return c.holds(D3.guest.Get(), 3, "D3 copy of T3 recorded before Release and overwrite");
}

// 5: a live texture is never touched.
bool case5_live_control(Context& c, GuestHeap& visible)
{
    Res& T = c.make_committed("T5");
    if (!c.fill_desktop(T.guest.Get(), 5)) return false;
    const auto srv = texture_view();
    invoke<void>(c.guest, 18, T.guest.Get(), &srv, visible.cpu + 4 * c.increment);
    for (int i = 0; i < 8; ++i)
        if (!c.stays("T5 live", "a live texture left the budget")) return false;
    return c.slot_is(visible, 4, false, "case 5: descriptor of a live texture");
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) ||
        (argc == 3 && std::strcmp(argv[2], "keep")))
        return 2;
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), std::strcmp(argv[1], "warp") == 0);
    if (!adapter) return 77;
    char debug_env[2]{};
    const bool debug_on = GetEnvironmentVariableA("X2W_D3D12_DEBUG", debug_env, sizeof(debug_env)) && debug_env[0] == '1';
    if (debug_on) {
        ComPtr<ID3D12Debug> debug;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) return 77;
        debug->EnableDebugLayer();
    }
    ComPtr<IUnknown> guest;
    ComPtr<ID3D12Device> device;
    std::array<unsigned char, 64> parameters{};
    if (!require(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                      reinterpret_cast<void**>(guest.GetAddressOf())), "Xbox device") ||
        !require(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "desktop device"))
        return 3;
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
    char value[2]{};
    const bool retire_on = GetEnvironmentVariableA("X2W_D3D12_RELEASE_RETIRED", value, sizeof(value)) == 1 && value[0] == '1';
    value[0] = 0;
    const bool views_on = GetEnvironmentVariableA("X2W_D3D12_RELEASE_RETIRED_VIEWS", value, sizeof(value)) == 1 && value[0] == '1';
    const bool keep_mode = argc == 3;
    if (retire_on == keep_mode) return 2;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    adapter->GetDesc1(&adapter_desc);
    std::printf("adapter=%ls debug_layer=%s retire=%s legacy_views=%s\n", adapter_desc.Description,
                debug ? "enabled" : "off", retire_on ? "on" : "off", views_on ? "on" : "off");

    Context c{guest.Get(), device.Get()};
    try {
        c.desc = {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, side, side, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0},
                  D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
        c.texture_bytes = device->GetResourceAllocationInfo(0, 1, &c.desc).SizeInBytes;
        c.increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        c.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const D3D12_COMMAND_QUEUE_DESC direct{};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD}, readback_heap{D3D12_HEAP_TYPE_READBACK},
            default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, image_bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                         D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        D3D12_RESOURCE_DESC out_desc = buffer;
        out_desc.Width = 256;
        out_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_DESCRIPTOR_HEAP_DESC probe_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                                         D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        if (!c.event || !require(device->CreateCommandQueue(&direct, IID_PPV_ARGS(&c.own_queue)), "own queue") ||
            !require(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&c.own_allocator)), "own allocator") ||
            !require(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, c.own_allocator.Get(), nullptr, IID_PPV_ARGS(&c.own_list)), "own list") ||
            !require(c.own_list->Close(), "own list close") ||
            !require(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c.own_fence)), "own fence") ||
            !require(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                     nullptr, IID_PPV_ARGS(&c.upload)), "upload") ||
            !require(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                                     nullptr, IID_PPV_ARGS(&c.readback)), "readback") ||
            !require(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &out_desc,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&c.probe_out)), "probe output") ||
            !require(device->CreateDescriptorHeap(&probe_heap_desc, IID_PPV_ARGS(&c.probe_heap)), "probe heap"))
            return 4;

        // The probe: a compute shader that loads texel (0,0) through descriptor table slot 0.
        const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER parameters_root[2]{};
        parameters_root[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters_root[0].DescriptorTable = {1, &range};
        parameters_root[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        parameters_root[1].Descriptor = {0, 0};
        const D3D12_ROOT_SIGNATURE_DESC root_desc{2, parameters_root, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> root_blob, errors, code;
        if (!require(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, root_blob.GetAddressOf(),
                                                errors.GetAddressOf()), "probe root serialize") ||
            !require(device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(),
                                                 IID_PPV_ARGS(&c.probe_root)), "probe root signature") ||
            !require(D3DCompile(probe_shader, std::strlen(probe_shader), nullptr, nullptr, nullptr, "main", "cs_5_1", 0, 0,
                                code.GetAddressOf(), errors.ReleaseAndGetAddressOf()), "probe shader"))
            return 4;
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = c.probe_root.Get();
        pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        if (!require(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&c.probe_pso)), "probe pipeline")) return 4;

        if (!c.make_queue(c.direct) || !c.make_queue(c.gated) || !c.make_list(c.drain_list) || !c.close(c.drain_list)) return 4;
        if (!x2w_d3d12_budget_snapshot(&c.base, sizeof(c.base))) return 4;

        GuestHeap visible, staging;
        if (!c.make_heap(visible, 8, true) || !c.make_heap(staging, 1, false)) return 4;

        if (!case1_single_view(c, retire_on, visible)) return 5;
        std::puts("1 same slot rebind reads its texture after Release and permits retirement after overwrite");
        if (!case2_copy_and_gate(c, retire_on, visible, staging)) return 5;
        std::puts("2 original and copied slots read their texture after the release fences, retaining it through the final overwrite");
        if (!case3_list_holds(c, retire_on, visible)) return 5;
        std::puts("3/4 overwritten slots preserve list and submission holds, recorded copy reads the original texels");
        if (!case5_live_control(c, visible)) return 5;
        std::puts("5 live texture: descriptor untouched");
    } catch (int code) {
        return code;
    }
    const UINT64 errors = debug ? debug->GetNumStoredMessages() : 0;
    std::printf("debug_errors_or_warnings=%llu\n", errors);
    if (errors || FAILED(device->GetDeviceRemovedReason())) return 4;
    std::puts(retire_on ? "live descriptor retirement checks passed" : "base retirement off: all retention controls passed");
    return 0;
}
