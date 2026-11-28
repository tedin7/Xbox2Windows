// A replay declares the descriptor increment the recorded title was given (x2w_d3d12_replay_title_increment). When it
// differs from this device's, title CPU and GPU handles must map to desktop descriptors by index, and a CPU handle that
// is in title memory no desktop heap backs must never reach D3D12 (the debug layer reports #646
// INVALID_DESCRIPTOR_HANDLE and raises 0x87D; without the layer the descriptor is written into that title memory).
//
//   startup_title_stride_heap <warp|hardware>   (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1 X2W_D3D12_DEBUG=1)
//
// Title stride T = 32, or 64 where the device's own increment is 32. Two heaps of 8: a shader-visible one over title
// memory of 8 * T bytes, and a CPU-side one. UAVs of A at index 1 and B at index 2 (odd index on purpose), an SRV at
// index 3, then two SRVs that must be refused: one into title memory of stride 2 * T that a third heap offered and the
// layer could not back (that memory must stay zero), and one in the middle of a title descriptor.
// ClearUnorderedAccessViewUint through index 1 and 2, read back: A a1a1a1a1, B b2b2b2b2. The device must not be
// removed and the debug layer must report no error.
// Exit codes: 0 pass, 77 no physical adapter, other values name the failing step.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void x2w_d3d12_replay_title_increment(UINT, UINT);
extern "C" __declspec(dllimport) void x2w_d3d12_render_target_stats(unsigned* binds, unsigned* unresolved);

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
    ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    void* queue = nullptr;
    void* fence = nullptr;
    Word fence_value = 0;

    bool submit(void* list)
    {
        void* lists[] = {list};
        invoke(queue, 10, 1, arg(lists));
        if (!ok(invoke(queue, 14, arg(fence), ++fence_value))) return false;
        const auto deadline = GetTickCount64() + 10000;
        while (invoke(fence, 8) < fence_value) {
            if (GetTickCount64() >= deadline) return false;
            Sleep(1);
        }
        return true;
    }
    // Cumulative error count from a message callback registered before any recording; the shim drains the
    // stored queue at every submit, so the stored messages alone would miss the recording errors.
    DWORD cookie = 0;
    ComPtr<ID3D12InfoQueue1> queue1;
    bool watch()
    {
        return SUCCEEDED(desktop.As(&queue1)) &&
               SUCCEEDED(queue1->RegisterMessageCallback(on_message, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie));
    }
    // The one texel of a target (4 bytes), copied on a desktop queue.
    std::uint32_t texel(ID3D12Resource* target)
    {
        D3D12_COMMAND_QUEUE_DESC desc{};
        ComPtr<ID3D12CommandQueue> observer;
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> copy;
        ComPtr<ID3D12Fence> done;
        if (FAILED(desktop->CreateCommandQueue(&desc, IID_PPV_ARGS(&observer))) ||
            FAILED(desktop->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
            FAILED(desktop->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&copy))) ||
            FAILED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done))))
            return 0;
        D3D12_TEXTURE_COPY_LOCATION to{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        to.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION from{target, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        copy->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        if (FAILED(copy->Close())) return 0;
        ID3D12CommandList* lists[] = {copy.Get()};
        observer->ExecuteCommandLists(1, lists);
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool finished = event && SUCCEEDED(observer->Signal(done.Get(), 1)) &&
                              SUCCEEDED(done->SetEventOnCompletion(1, event)) && WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
        if (event) CloseHandle(event);
        void* memory = nullptr;
        if (!finished || FAILED(readback->Map(0, nullptr, &memory))) return 0;
        std::uint32_t bits = 0;
        std::memcpy(&bits, memory, sizeof(bits));
        readback->Unmap(0, nullptr);
        return bits;
    }
};

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
    } else if (std::strcmp(argv[1], "warp") != 0) {
        std::printf("unknown adapter\n"); return 2;
    } else if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        std::printf("WARP unavailable\n"); return 77;
    }
    DXGI_ADAPTER_DESC adapter_desc{};
    if (FAILED(adapter->GetDesc(&adapter_desc))) return 3;
    std::printf("adapter=%ls\n", adapter_desc.Description);

    Probe probe;
    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 4;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&probe.desktop)))) return 4;

    // Without the debug layer the error count stays 0; the readback and untouched-memory checks still run.
    if (!probe.watch()) std::printf("no debug layer message callback: validation not counted\n");
    const UINT desktop_stride = probe.desktop->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT stride = desktop_stride == 32 ? 64 : 32;
    x2w_d3d12_replay_title_increment(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, stride);

    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC uint_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 1, 1, 1, DXGI_FORMAT_R32_UINT,
                                        {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    ComPtr<ID3D12Resource> a, b;
    const auto make = [&](ComPtr<ID3D12Resource>& out) {
        return SUCCEEDED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &uint_desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&out)));
    };
    if (!make(a) || !make(b)) { std::printf("texture creation failed\n"); return 5; }
    UINT64 bytes = 0;
    probe.desktop->GetCopyableFootprints(&uint_desc, 0, 1, 0, &probe.footprint, nullptr, nullptr, &bytes);
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                     D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
    if (FAILED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                                      nullptr, IID_PPV_ARGS(&probe.readback)))) return 5;

    // `backing` is title memory of the declared stride. `offered` is title memory of another stride (2 * T per
    // descriptor): the layer cannot back it, as for a recording whose increments were not declared.
    const UINT offered_stride = 2 * stride;
    void* backing = VirtualAlloc(nullptr, 8 * stride, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    void* offered = VirtualAlloc(nullptr, 8 * offered_stride, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!backing || !offered) { std::printf("allocation failed\n"); return 6; }
    const auto make_heap = [&](D3D12_DESCRIPTOR_HEAP_FLAGS flags, void* memory, Word bytes, void** out) {
        const struct { D3D12_DESCRIPTOR_HEAP_DESC desc; Word backing, bytes; }
            desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8, flags}, arg(memory), bytes};
        return ok(invoke(device, 14, arg(&desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(out)));
    };
    void *visible_heap = nullptr, *plain_heap = nullptr, *offered_heap = nullptr;
    if (!make_heap(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, backing, 8ull * stride, &visible_heap) ||
        !make_heap(D3D12_DESCRIPTOR_HEAP_FLAG_NONE, nullptr, 0, &plain_heap) ||
        !make_heap(D3D12_DESCRIPTOR_HEAP_FLAG_NONE, offered, 8ull * offered_stride, &offered_heap)) {
        std::printf("heap creation failed\n"); return 7;
    }
    const Word visible_cpu = invoke(visible_heap, 9), plain_cpu = invoke(plain_heap, 9), gpu_start = invoke(visible_heap, 10);
    if (visible_cpu != arg(backing)) { std::printf("title heap start differs\n"); return 8; }
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{DXGI_FORMAT_R32_UINT, D3D12_UAV_DIMENSION_TEXTURE2D, {}};
    ID3D12Resource* views[] = {a.Get(), b.Get()};
    for (unsigned i = 1; i <= 2; ++i) {
        invoke(device, 19, arg(views[i - 1]), 0, arg(&view), visible_cpu + i * stride);
        invoke(device, 19, arg(views[i - 1]), 0, arg(&view), plain_cpu + i * stride);
    }
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_UINT;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    invoke(device, 18, arg(a.Get()), arg(&srv), visible_cpu + 3 * stride);
    invoke(device, 18, arg(a.Get()), arg(&srv), arg(offered) + offered_stride);
    invoke(device, 18, arg(a.Get()), arg(&srv), visible_cpu + stride / 2);
    if (debug_error_count.load()) { std::printf("invalid SRV reached D3D12\n"); return 10; }
    const auto* page = static_cast<const unsigned char*>(offered);
    for (unsigned i = 0; i < 8 * offered_stride; ++i)
        if (page[i]) { std::printf("a descriptor was written into title memory no heap backs (byte %u)\n", i); return 17; }

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void *allocator = nullptr, *list = nullptr;
    if (!ok(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&probe.queue))) ||
        !ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&probe.fence))) ||
        !ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))))
        { std::printf("command objects failed\n"); return 15; }
    if (!ok(invoke(list, 139, arg(allocator), 0))) { std::printf("list reset failed\n"); return 16; }
    void* heaps[] = {visible_heap};
    invoke(list, 32, 1, arg(heaps));

    const std::uint32_t values_a[4] = {0xa1a1a1a1, 0xa1a1a1a1, 0xa1a1a1a1, 0xa1a1a1a1};
    const std::uint32_t values_b[4] = {0xb2b2b2b2, 0xb2b2b2b2, 0xb2b2b2b2, 0xb2b2b2b2};
    invoke(list, 53, gpu_start + stride, plain_cpu + stride, arg(a.Get()), arg(values_a), 0, 0);
    invoke(list, 53, gpu_start + 2 * stride, plain_cpu + 2 * stride, arg(b.Get()), arg(values_b), 0, 0);
    invoke(list, 9);
    if (!probe.submit(list)) { std::printf("submit failed\n"); return 11; }
    const std::uint32_t bits_a = probe.texel(a.Get()), bits_b = probe.texel(b.Get());
    if (bits_a != 0xa1a1a1a1) { std::printf("A readback: %08x\n", bits_a); return 12; }
    if (bits_b != 0xb2b2b2b2) { std::printf("B readback: %08x\n", bits_b); return 13; }
    // Depth stencil heap. The console's DSV descriptors are 8 bytes; a desktop device's are larger (32 on WARP and most
    // hardware). A replay declares 8, and title handle k * 8 must reach descriptor k of the desktop heap. Without the
    // declaration (its minimum was 32) handles k * 8 land inside one another's descriptors and the bind or clear reads garbage.
    {
        constexpr UINT dsv_title = 8;
        x2w_d3d12_replay_title_increment(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, dsv_title);
        void* dsv_memory = VirtualAlloc(nullptr, 8 * dsv_title, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        const struct { D3D12_DESCRIPTOR_HEAP_DESC desc; Word backing, bytes; }
            dsv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 8, D3D12_DESCRIPTOR_HEAP_FLAG_NONE}, arg(dsv_memory), 8ull * dsv_title};
        void* dsv_heap = nullptr;
        if (!dsv_memory || !ok(invoke(device, 14, arg(&dsv_heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&dsv_heap)))) {
            std::printf("DSV heap creation failed\n"); return 18;
        }
        const Word dsv_start = invoke(dsv_heap, 9);
        if (dsv_start != arg(dsv_memory)) { std::printf("DSV title heap start differs\n"); return 19; }
        D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_RESOURCE_DESC depth_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 4, 4, 1, 1, DXGI_FORMAT_D32_FLOAT,
                                             {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
        ComPtr<ID3D12Resource> depth[2];
        for (auto& resource : depth)
            if (FAILED(probe.desktop->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &depth_desc,
                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource)))) { std::printf("depth creation failed\n"); return 20; }
        // Descriptor 1 and 5 (title offsets 8 and 40): 5 is not a multiple of a 32-byte desktop descriptor's 8-byte step.
        const unsigned slots[2] = {1, 5};
        const float clears[2] = {0.75f, 0.25f};
        D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
        dsv.Format = DXGI_FORMAT_D32_FLOAT;
        dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        for (unsigned i = 0; i < 2; ++i) invoke(device, 21, arg(depth[i].Get()), arg(&dsv), dsv_start + slots[i] * dsv_title);
        unsigned binds_before = 0, unresolved_before = 0, binds_after = 0, unresolved_after = 0;
        x2w_d3d12_render_target_stats(&binds_before, &unresolved_before);
        if (!ok(invoke(list, 139, arg(allocator), 0))) { std::printf("list reset failed (DSV)\n"); return 21; }
        using ClearDepth = void (WINAPI*)(void*, Word, UINT, float, UINT8, UINT, const void*);
        const auto clear_depth = reinterpret_cast<ClearDepth>((*static_cast<void***>(list))[51]);
        for (unsigned i = 0; i < 2; ++i) {
            const Word handle = dsv_start + slots[i] * dsv_title;
            invoke(list, 50, 0, 0, 0, arg(&handle));
            clear_depth(list, handle, D3D12_CLEAR_FLAG_DEPTH, clears[i], 0, 0, nullptr);
        }
        invoke(list, 9);
        if (!probe.submit(list)) { std::printf("DSV submit failed\n"); return 22; }
        x2w_d3d12_render_target_stats(&binds_after, &unresolved_after);
        if (binds_after < binds_before + 2 || unresolved_after != unresolved_before) {
            std::printf("DSV binds %u -> %u, unresolved %u -> %u\n", binds_before, binds_after, unresolved_before, unresolved_after);
            return 23;
        }
        // Read each depth texel back on a desktop queue: COMMON to COPY_SOURCE, one copy of the first texel.
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT depth_footprint{};
        UINT64 depth_bytes = 0;
        probe.desktop->GetCopyableFootprints(&depth_desc, 0, 1, 0, &depth_footprint, nullptr, nullptr, &depth_bytes);
        D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        const D3D12_RESOURCE_DESC depth_buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, depth_bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        for (unsigned i = 0; i < 2; ++i) {
            ComPtr<ID3D12Resource> depth_readback;
            D3D12_COMMAND_QUEUE_DESC desc{};
            ComPtr<ID3D12CommandQueue> observer;
            ComPtr<ID3D12CommandAllocator> observer_allocator;
            ComPtr<ID3D12GraphicsCommandList> copy;
            ComPtr<ID3D12Fence> done;
            if (FAILED(probe.desktop->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &depth_buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&depth_readback))) ||
                FAILED(probe.desktop->CreateCommandQueue(&desc, IID_PPV_ARGS(&observer))) ||
                FAILED(probe.desktop->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&observer_allocator))) ||
                FAILED(probe.desktop->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, observer_allocator.Get(), nullptr,
                                                        IID_PPV_ARGS(&copy))) ||
                FAILED(probe.desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done)))) { std::printf("observer failed\n"); return 24; }
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = depth[i].Get();
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            copy->ResourceBarrier(1, &barrier);
            D3D12_TEXTURE_COPY_LOCATION to{depth_readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            to.PlacedFootprint = depth_footprint;
            D3D12_TEXTURE_COPY_LOCATION from{depth[i].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            copy->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            if (FAILED(copy->Close())) return 24;
            ID3D12CommandList* lists[] = {copy.Get()};
            observer->ExecuteCommandLists(1, lists);
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            const bool finished = event && SUCCEEDED(observer->Signal(done.Get(), 1)) &&
                                  SUCCEEDED(done->SetEventOnCompletion(1, event)) && WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
            if (event) CloseHandle(event);
            void* memory = nullptr;
            if (!finished || FAILED(depth_readback->Map(0, nullptr, &memory))) { std::printf("DSV readback failed\n"); return 25; }
            float value = 0;
            std::memcpy(&value, memory, sizeof(value));
            depth_readback->Unmap(0, nullptr);
            std::printf("DSV descriptor %u cleared to %.3f, read %.3f\n", slots[i], clears[i], value);
            if (value != clears[i]) return 26;
        }
    }
    const HRESULT removed = probe.desktop->GetDeviceRemovedReason();
    const auto errors = debug_error_count.load();
    std::printf("desktop stride=%u title stride=%u A=%08x B=%08x removed=0x%08lx errors=%llu\n",
                desktop_stride, stride, bits_a, bits_b, static_cast<unsigned long>(removed), errors);
    return removed == S_OK && errors == 0 ? 0 : 14;
}
