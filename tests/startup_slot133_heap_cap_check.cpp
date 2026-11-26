// Slot 133 existing-heap cap (opt-in X2W_D3D12_CAP_EXISTING_HEAP=1, <= 256 MiB), deferred
// commit_lazy_block, and incremental reserve_upload_ring growth (Refs #1199, Fixes #2765).
//
//   startup_slot133_heap_cap_check <warp|hardware> [cap]
//
// 1. With X2W_XMEM_LAZY_COMMIT=1 and X2W_XMEM_LAZY_STARTUP=1:
//    - Allocate a 260 MiB (> 256 MiB) MEM_PRIVATE lazy startup block via XMemVirtualAlloc.
//    - Default (no `cap`): the aligned 64 KiB placement at +0 opens the 260 MiB existing heap
//      (aliased, ALLOW_CROSS_ADAPTER) and commits the block. A guest CopyBufferRegion (list slot 15)
//      recorded from it into a readback buffer must observe a late CPU write made after recording
//      and before submit (fenced readback, every byte 0x59): a capped copied placement would be
//      refused as title_backed_source and lose that write, so the cap stays opt-in.
//    - With `cap` (X2W_D3D12_CAP_EXISTING_HEAP=1): place a 64 KiB aligned buffer at +0 and a
//      64 KiB unaligned buffer at +2 MiB + 16 via slot 133: neither opens an existing_heap
//      (x2w-budget heap stays 0/0, desktop resources lack ALLOW_CROSS_ADAPTER) and untouched
//      pages of the 260 MiB block remain MEM_RESERVE.
//    - Meanwhile, <= 256 MiB blocks (4 MiB VirtualAlloc and 64 MiB XMemVirtualAlloc lazy block at
//      +2 MiB with an untouched [0, 2 MiB) MEM_RESERVE prefix) open existing_heap (aliased) cleanly,
//      verified with a late A-to-B CPU write and GPU readback; unaligned and out-of-bounds nonzero
//      offsets in a 64 MiB lazy block leave untouched pages MEM_RESERVE.
// 2. reserve_upload_ring incremental growth (cap variant):
//    - First upload (64x64 RGBA8 CopyResource) commits 16 MiB (not 256 MiB) in x2w-budget upload.
//    - A second small upload (32x32 RGBA8) reuses the same 16 MiB chunk.
//    - Uploading a 2048x2048 RGBA8 (16 MiB) texture exceeds chunk 0 and appends a 32 MiB chunk
//      (48 MiB / 2 objects), and another small upload lands in chunk 1; GPU readbacks verify the
//      uploaded texels from both chunks with zero debug-layer errors.
#include "startup_stubs/resource_budget.h"

#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
using namespace x2w_budget;

using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using CreateXboxDeviceFn = HRESULT(WINAPI*)(IUnknown*, const void*, REFIID, void**);
using BudgetSnapshotFn = BOOL (*)(Snapshot*, SIZE_T);

XMemVirtualAllocFn g_xmem_virtual_alloc = nullptr;
CreateXboxDeviceFn g_create_xbox_device = nullptr;
BudgetSnapshotFn g_budget_snapshot = nullptr;

constexpr UINT64 page = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
constexpr UINT64 mib = UINT64{1} << 20;
constexpr DWORD wait_ms = 10000;

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

bool check(bool ok, const char* what)
{
    if (!ok) std::printf("FAIL: %s\n", what);
    return ok;
}

bool snap(Snapshot& counts)
{
    if (g_budget_snapshot && g_budget_snapshot(&counts, sizeof(counts))) return true;
    std::puts("FAIL: budget snapshot unavailable");
    return false;
}

DWORD mem_state(const void* address)
{
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) ? info.State : 0;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes)
{
    return {D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
}

D3D12_RESOURCE_DESC tex2d_desc(UINT width, UINT height)
{
    return {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0},
            D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
}

std::uint8_t pattern_byte(int seed, UINT64 index)
{
    return static_cast<std::uint8_t>((index * 131 + seed * 37 + 19) & 0xff);
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

bool load_shims(bool cap, bool video)
{
    SetEnvironmentVariableA("X2W_XMEM_LAZY_COMMIT", "1");
    SetEnvironmentVariableA("X2W_XMEM_LAZY_STARTUP", "1");
    SetEnvironmentVariableA("X2W_D3D12_BUDGET", "1");
    SetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", "1");
    SetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", "1");
    SetEnvironmentVariableA("X2W_D3D12_RENDER", "1");
    SetEnvironmentVariableA("X2W_D3D12_CAP_EXISTING_HEAP", cap ? "1" : nullptr);
    // A CPU-written texture of 1 MiB or more is filled from its own upload buffer. The ring-growth check below needs a
    // 16 MiB upload to go through the shared ring, so it runs with the fixture switch that restores the ring; the "video"
    // variant runs without it.
    SetEnvironmentVariableA("X2W_TEST_ADDRESS_UPLOAD_RING", video ? nullptr : "1");

    wchar_t exe_dir[MAX_PATH]{};
    const DWORD n = GetModuleFileNameW(nullptr, exe_dir, MAX_PATH);
    if (!n || n >= MAX_PATH) return false;
    std::wstring dir(exe_dir, n);
    const std::size_t slash = dir.find_last_of(L"\\/");
    dir.resize(slash == std::wstring::npos ? 0 : slash + 1);

    const HMODULE xmem = LoadLibraryW((dir + L"xmem.dll").c_str());
    const HMODULE d3d = LoadLibraryW((dir + L"d3d12_xs.dll").c_str());
    if (!xmem || !d3d) {
        std::printf("FAIL: could not load xmem.dll (%p) or d3d12_xs.dll (%p), err=%lu\n",
                    static_cast<void*>(xmem), static_cast<void*>(d3d), GetLastError());
        return false;
    }
    g_xmem_virtual_alloc = reinterpret_cast<XMemVirtualAllocFn>(GetProcAddress(xmem, "XMemVirtualAlloc"));
    g_create_xbox_device = reinterpret_cast<CreateXboxDeviceFn>(GetProcAddress(d3d, "D3D12XboxCreateDevice"));
    g_budget_snapshot = reinterpret_cast<BudgetSnapshotFn>(GetProcAddress(d3d, "x2w_d3d12_budget_snapshot"));
    return g_xmem_virtual_alloc && g_create_xbox_device && g_budget_snapshot;
}

bool verify_texture_box(ID3D12Device* device, ID3D12CommandQueue* own_queue, ID3D12CommandAllocator* own_alloc,
                        ID3D12GraphicsCommandList* own_list, ID3D12Fence* own_fence, UINT64& own_val,
                        ID3D12Resource* real_tex, UINT box_x, UINT box_y, UINT box_w, UINT box_h,
                        UINT full_w, int seed, const char* label)
{
    const UINT pitch = (box_w * 4 + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);
    const UINT64 rb_bytes = UINT64{pitch} * box_h;
    const D3D12_HEAP_PROPERTIES rb_heap{D3D12_HEAP_TYPE_READBACK};
    const auto rb_desc = buffer_desc(rb_bytes);
    ComPtr<ID3D12Resource> rb;
    if (!require(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb)),
                 "readback buffer"))
        return false;

    if (!require(own_alloc->Reset(), "own alloc reset") || !require(own_list->Reset(own_alloc, nullptr), "own list reset"))
        return false;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {real_tex, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE};
    own_list->ResourceBarrier(1, &b);

    D3D12_TEXTURE_COPY_LOCATION dst{rb.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, box_w, box_h, 1, pitch};
    D3D12_TEXTURE_COPY_LOCATION src{real_tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    D3D12_BOX box{box_x, box_y, 0, box_x + box_w, box_y + box_h, 1};
    own_list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);

    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    own_list->ResourceBarrier(1, &b);
    if (!require(own_list->Close(), "own list close")) return false;
    ID3D12CommandList* lists[] = {own_list};
    own_queue->ExecuteCommandLists(1, lists);
    if (!require(own_queue->Signal(own_fence, ++own_val), "own signal")) return false;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return false;
    const bool waited = SUCCEEDED(own_fence->SetEventOnCompletion(own_val, ev)) &&
                        WaitForSingleObject(ev, wait_ms) == WAIT_OBJECT_0;
    CloseHandle(ev);
    if (!check(waited, "own fence wait")) return false;

    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(rb_bytes)};
    if (!require(rb->Map(0, &range, &mapped), "readback map")) return false;
    const auto* got = static_cast<const std::uint8_t*>(mapped);
    UINT mismatches = 0;
    for (UINT y = 0; y < box_h; ++y) {
        for (UINT x = 0; x < box_w * 4; ++x) {
            const UINT64 src_idx = (UINT64{box_y + y} * full_w * 4) + (UINT64{box_x} * 4) + x;
            const std::uint8_t expected = pattern_byte(seed, src_idx);
            if (got[UINT64{y} * pitch + x] != expected) ++mismatches;
        }
    }
    rb->Unmap(0, nullptr);
    if (mismatches) {
        std::printf("FAIL: %s readback mismatches=%u\n", label, mismatches);
        return false;
    }
    return true;
}

bool verify_buffer_pattern(ID3D12Device* device, ID3D12CommandQueue* own_queue, ID3D12CommandAllocator* own_alloc,
                           ID3D12GraphicsCommandList* own_list, ID3D12Fence* own_fence, UINT64& own_val,
                           ID3D12Resource* real_buf, UINT64 bytes, int seed, const char* label)
{
    const D3D12_HEAP_PROPERTIES rb_heap{D3D12_HEAP_TYPE_READBACK};
    const auto rb_desc = buffer_desc(bytes);
    ComPtr<ID3D12Resource> rb;
    if (!require(device->CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb)),
                 "buffer readback"))
        return false;

    if (!require(own_alloc->Reset(), "own alloc reset") || !require(own_list->Reset(own_alloc, nullptr), "own list reset"))
        return false;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {real_buf, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE};
    own_list->ResourceBarrier(1, &b);
    own_list->CopyBufferRegion(rb.Get(), 0, real_buf, 0, bytes);
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    own_list->ResourceBarrier(1, &b);
    if (!require(own_list->Close(), "own list close")) return false;
    ID3D12CommandList* lists[] = {own_list};
    own_queue->ExecuteCommandLists(1, lists);
    if (!require(own_queue->Signal(own_fence, ++own_val), "own signal")) return false;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return false;
    const bool waited = SUCCEEDED(own_fence->SetEventOnCompletion(own_val, ev)) &&
                        WaitForSingleObject(ev, wait_ms) == WAIT_OBJECT_0;
    CloseHandle(ev);
    if (!check(waited, "own fence wait")) return false;

    void* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)};
    if (!require(rb->Map(0, &range, &mapped), "buffer readback map")) return false;
    const auto* got = static_cast<const std::uint8_t*>(mapped);
    UINT mismatches = 0;
    for (UINT64 i = 0; i < bytes; ++i) {
        if (got[i] != pattern_byte(seed, i)) ++mismatches;
    }
    rb->Unmap(0, nullptr);
    if (mismatches) {
        std::printf("FAIL: %s readback mismatches=%u\n", label, mismatches);
        return false;
    }
    return true;
}

int run_check(IUnknown* guest, ID3D12Device* device, ID3D12InfoQueue* debug, bool cap, bool video)
{
    const GUID& resource_iid = __uuidof(ID3D12Resource);
    D3D12_FEATURE_DATA_EXISTING_HEAPS existing{};
    const bool existing_supported =
        SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &existing, sizeof(existing))) &&
        existing.Supported;

    Snapshot base_snap{};
    if (!snap(base_snap)) return 4;
    if (!check(base_snap.live[heap].objects == 0 && base_snap.retained[heap].objects == 0,
               "baseline heap budget starts at 0/0"))
        return 5;
    if (!check(base_snap.live[upload].objects == 0 && base_snap.retained[upload].objects == 0,
               "baseline upload budget starts at 0/0"))
        return 5;

    // -------------------------------------------------------------------------
    // Part 1: slot 133 multi-GiB / >256 MiB existing-heap cap & lazy block commit
    // -------------------------------------------------------------------------
    constexpr SIZE_T large_bytes = 260 * mib;
    auto* const large_block = static_cast<char*>(
        g_xmem_virtual_alloc(nullptr, large_bytes, 0x20403000, 0x181, PAGE_READWRITE));
    if (!check(large_block != nullptr, "XMemVirtualAlloc 260 MiB block")) return 4;
    if (!check(mem_state(large_block) == MEM_RESERVE &&
                   mem_state(large_block + 1 * mib) == MEM_RESERVE &&
                   mem_state(large_block + 128 * mib) == MEM_RESERVE &&
                   mem_state(large_block + 259 * mib) == MEM_RESERVE,
               "260 MiB startup block starts MEM_RESERVE under X2W_XMEM_LAZY_STARTUP")) {
        VirtualFree(large_block, 0, MEM_RELEASE);
        return 4;
    }

    const auto buf_64k = buffer_desc(page);
    ComPtr<IUnknown> large_aligned;
    if (!require(invoke(guest, 133, large_block, &buf_64k, D3D12_RESOURCE_STATE_COMMON, nullptr,
                        &resource_iid, large_aligned.GetAddressOf()),
                 "slot 133 aligned buffer in 260 MiB block")) {
        VirtualFree(large_block, 0, MEM_RELEASE);
        return 4;
    }
    const auto large_aligned_real = desktop_object<ID3D12Resource>(guest, device, large_aligned.Get());
    if (!check(large_aligned_real != nullptr, "desktop resource for aligned 260 MiB buffer")) {
        VirtualFree(large_block, 0, MEM_RELEASE);
        return 4;
    }

    if (!cap) {
        if (!existing_supported) {
            std::puts("skipping default aliased existing-heap check: D3D12_FEATURE_EXISTING_HEAPS=0");
            large_aligned.Reset();
            VirtualFree(large_block, 0, MEM_RELEASE);
            return 77;
        }
        Snapshot s{};
        if (!snap(s)) return 4;
        // A failed placement check does not stop the late-write case below: with the cap forced on,
        // the copied fallback is refused by slot 15 and the readback sentinel survives, which is the
        // loss this fixture exists to catch. Any failure still fails the run.
        bool aliased_ok = check((large_aligned_real->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) != 0,
                                "default keeps aliased placement on 260 MiB block");
        aliased_ok &= check(s.live[heap].bytes == large_bytes && s.live[heap].objects == 1,
                            "default accounts 260 MiB heap");
        aliased_ok &= check(s.live[placed].objects == 1, "default accounts placed buffer");
        aliased_ok &= check(mem_state(large_block + 128 * mib) == MEM_COMMIT,
                            "default commits lazy block right before OpenExistingHeapFromAddress");

        // R0Y00's late-write case: record a guest CopyBufferRegion (list slot 15) from the aliased
        // buffer into a readback buffer, change the title memory after recording and before submit,
        // and require the fenced readback to hold the late bytes. The default alias keeps the
        // contents authoritative; a capped copied placement is refused as title_backed_source
        // (list_copy.inl sync_title_source) and the readback keeps its 0x6d sentinel.
        std::memset(large_block, 0x2a, static_cast<std::size_t>(page));
        const D3D12_COMMAND_QUEUE_DESC late_qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
        const unsigned late_list_desc[] = {static_cast<unsigned>(D3D12_COMMAND_LIST_TYPE_DIRECT), 2, 1, 0};
        ComPtr<IUnknown> late_queue, late_alloc, late_list, late_fence;
        if (!require(invoke(guest, 8, &late_qd, &__uuidof(ID3D12CommandQueue), late_queue.GetAddressOf()),
                     "late-write guest queue") ||
            !require(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator),
                            late_alloc.GetAddressOf()),
                     "late-write guest allocator") ||
            !require(invoke(guest, 73, late_list_desc, late_alloc.Get(), nullptr, &__uuidof(ID3D12GraphicsCommandList),
                            late_list.GetAddressOf()),
                     "late-write guest list") ||
            !require(invoke(guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence),
                            late_fence.GetAddressOf()),
                     "late-write guest fence") ||
            !require(invoke(late_list.Get(), 139, late_alloc.Get(), nullptr), "late-write guest list reset"))
            return 4;
        const auto late_real_fence = desktop_object<ID3D12Fence>(guest, device, late_fence.Get());
        if (!late_real_fence) return 4;

        const D3D12_HEAP_PROPERTIES late_rb_heap{D3D12_HEAP_TYPE_READBACK};
        const auto late_rb_desc = buffer_desc(page);
        ComPtr<ID3D12Resource> late_readback;
        if (!require(device->CreateCommittedResource(&late_rb_heap, D3D12_HEAP_FLAG_NONE, &late_rb_desc,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&late_readback)),
                     "late-write readback buffer"))
            return 4;
        void* primed = nullptr;
        const D3D12_RANGE prime_range{0, static_cast<SIZE_T>(page)};
        if (!require(late_readback->Map(0, &prime_range, &primed), "late-write readback prime map")) return 4;
        std::memset(primed, 0x6d, static_cast<std::size_t>(page));
        late_readback->Unmap(0, nullptr);

        // Slot 15 takes the raw desktop readback pointer as its destination (copy_endpoint).
        invoke<void>(late_list.Get(), 15, reinterpret_cast<UINT64>(late_readback.Get()), UINT64{0},
                     reinterpret_cast<UINT64>(large_aligned.Get()), UINT64{0}, UINT64{page});
        if (!require(invoke(late_list.Get(), 9), "late-write list close")) return 4;

        // The late guest write: after recording, before submit.
        std::memset(large_block, 0x59, static_cast<std::size_t>(page));

        IUnknown* late_lists[] = {late_list.Get()};
        invoke<void>(late_queue.Get(), 10, 1u, late_lists);
        UINT64 late_val = 1;
        if (!require(invoke(late_queue.Get(), 14, late_fence.Get(), late_val), "late-write queue signal")) return 4;
        HANDLE late_ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!late_ev) return 4;
        const bool late_waited = SUCCEEDED(late_real_fence->SetEventOnCompletion(late_val, late_ev)) &&
                                 WaitForSingleObject(late_ev, wait_ms) == WAIT_OBJECT_0;
        CloseHandle(late_ev);
        if (!check(late_waited, "late-write fence wait")) return 4;

        void* late_mapped = nullptr;
        const D3D12_RANGE late_read_range{0, static_cast<SIZE_T>(page)};
        if (!require(late_readback->Map(0, &late_read_range, &late_mapped), "late-write readback map")) return 4;
        const auto* late_got = static_cast<const std::uint8_t*>(late_mapped);
        UINT64 late_stale = 0, late_first = 0;
        std::uint8_t late_first_byte = 0;
        for (UINT64 i = 0; i < page; ++i) {
            if (late_got[i] != 0x59) {
                if (!late_stale) { late_first = i; late_first_byte = late_got[i]; }
                ++late_stale;
            }
        }
        late_readback->Unmap(0, nullptr);
        if (late_stale) {
            std::printf("FAIL: late write lost on 260 MiB buffer: %llu of %llu readback bytes are not 0x59 "
                        "(first at offset %llu = 0x%02x; 0x6d is the readback sentinel, 0x2a the pre-record "
                        "contents; a capped copied placement is refused as title_backed_source)\n",
                        static_cast<unsigned long long>(late_stale), static_cast<unsigned long long>(page),
                        static_cast<unsigned long long>(late_first), late_first_byte);
            return 5;
        }
        if (!aliased_ok) return 5;
        large_aligned.Reset();
        std::puts("PASS (default aliased + late write)");
        return 0;
    }

    // Cap mode (X2W_D3D12_CAP_EXISTING_HEAP=1): > 256 MiB block must NOT open an existing_heap and
    // untouched pages remain MEM_RESERVE.
    if (!check((large_aligned_real->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) == 0,
               "260 MiB aligned buffer must fall back (not aliased)")) {
        VirtualFree(large_block, 0, MEM_RELEASE);
        return 5;
    }
    Snapshot after_large_aligned{};
    if (!snap(after_large_aligned)) return 4;
    if (!check(after_large_aligned.live[heap].bytes == 0 && after_large_aligned.live[heap].objects == 0 &&
                   after_large_aligned.retained[heap].bytes == 0 && after_large_aligned.retained[heap].objects == 0,
               "260 MiB aligned buffer keeps x2w-budget heap at 0/0") ||
        !check(after_large_aligned.live[buffer].objects == 1,
               "260 MiB aligned buffer uses committed buffer fallback") ||
        !check(mem_state(large_block + 1 * mib) == MEM_RESERVE &&
                   mem_state(large_block + 128 * mib) == MEM_RESERVE &&
                   mem_state(large_block + 259 * mib) == MEM_RESERVE,
               "untouched pages of 260 MiB block stay MEM_RESERVE after aligned slot 133")) {
        VirtualFree(large_block, 0, MEM_RELEASE);
        return 5;
    }

    // Unaligned 64 KiB buffer inside the 260 MiB lazy startup block must also leave untouched pages MEM_RESERVE.
    ComPtr<IUnknown> large_unaligned;
    if (!require(invoke(guest, 133, large_block + 2 * mib + 16, &buf_64k, D3D12_RESOURCE_STATE_COMMON, nullptr,
                        &resource_iid, large_unaligned.GetAddressOf()),
                 "slot 133 unaligned buffer in 260 MiB block")) {
        VirtualFree(large_block, 0, MEM_RELEASE);
        return 4;
    }
    const auto large_unaligned_real = desktop_object<ID3D12Resource>(guest, device, large_unaligned.Get());
    Snapshot after_large_unaligned{};
    if (!snap(after_large_unaligned)) return 4;
    if (!check(large_unaligned_real &&
                   (large_unaligned_real->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) == 0,
               "260 MiB unaligned buffer is not aliased") ||
        !check(after_large_unaligned.live[heap].bytes == 0 && after_large_unaligned.live[heap].objects == 0 &&
                   after_large_unaligned.retained[heap].bytes == 0 && after_large_unaligned.retained[heap].objects == 0,
               "260 MiB unaligned buffer keeps x2w-budget heap at 0/0") ||
        !check(after_large_unaligned.live[buffer].objects == 2,
               "260 MiB unaligned buffer uses committed buffer fallback") ||
        !check(mem_state(large_block + 1 * mib) == MEM_RESERVE &&
                   mem_state(large_block + 128 * mib) == MEM_RESERVE &&
                   mem_state(large_block + 259 * mib) == MEM_RESERVE,
               "untouched pages of 260 MiB block stay MEM_RESERVE after unaligned slot 133")) {
        VirtualFree(large_block, 0, MEM_RELEASE);
        return 5;
    }
    large_aligned.Reset();
    large_unaligned.Reset();
    if (!check(VirtualFree(large_block, 0, MEM_RELEASE) != 0, "release 260 MiB lazy block")) return 4;

    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    ComPtr<ID3D12CommandQueue> own_queue;
    ComPtr<ID3D12CommandAllocator> own_alloc;
    ComPtr<ID3D12GraphicsCommandList> own_list;
    ComPtr<ID3D12Fence> own_fence;
    if (!require(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&own_queue)), "own queue") ||
        !require(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&own_alloc)), "own alloc") ||
        !require(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, own_alloc.Get(), nullptr,
                                           IID_PPV_ARGS(&own_list)),
                 "own list") ||
        !require(own_list->Close(), "own list close") ||
        !require(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&own_fence)), "own fence"))
        return 4;
    UINT64 own_val = 0, guest_val = 0;

    if (existing_supported) {
        // <= 256 MiB (4 MiB VirtualAlloc) MUST open an existing_heap (aliased) and reuse it.
        constexpr SIZE_T small_block_bytes = 4 * mib;
        auto* const small_block = static_cast<char*>(
            VirtualAlloc(nullptr, small_block_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!check(small_block != nullptr, "VirtualAlloc 4 MiB block")) return 4;
        ComPtr<IUnknown> small_a, small_b;
        if (!require(invoke(guest, 133, small_block + page, &buf_64k, D3D12_RESOURCE_STATE_COMMON, nullptr,
                            &resource_iid, small_a.GetAddressOf()),
                     "slot 133 aligned buffer in 4 MiB block") ||
            !require(invoke(guest, 133, small_block + 2 * page, &buf_64k, D3D12_RESOURCE_STATE_COMMON, nullptr,
                            &resource_iid, small_b.GetAddressOf()),
                     "slot 133 second aligned buffer in 4 MiB block"))
            return 4;
        const auto small_a_real = desktop_object<ID3D12Resource>(guest, device, small_a.Get());
        const auto small_b_real = desktop_object<ID3D12Resource>(guest, device, small_b.Get());
        Snapshot after_4m{};
        if (!snap(after_4m)) return 4;
        if (!check(small_a_real && (small_a_real->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) != 0 &&
                       small_b_real && (small_b_real->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) != 0,
                   "4 MiB <= 256 MiB buffers are aliased") ||
            !check(after_4m.live[heap].bytes == small_block_bytes && after_4m.live[heap].objects == 1,
                   "4 MiB block opens one existing_heap") ||
            !check(after_4m.live[placed].objects == 2, "4 MiB block places 2 aliased buffers")) {
            return 5;
        }
        small_a.Reset();
        small_b.Reset();

        // Negative controls on a <= 256 MiB lazy startup block (64 MiB): unaligned and out-of-bounds
        // nonzero offsets must NOT commit the whole lazy block or open an existing_heap.
        constexpr SIZE_T mid_bytes = 64 * mib;
        auto* const mid_ctrl = static_cast<char*>(
            g_xmem_virtual_alloc(nullptr, mid_bytes, 0x20403000, 0x181, PAGE_READWRITE));
        if (!check(mid_ctrl != nullptr, "XMemVirtualAlloc 64 MiB control block") ||
            !check(mem_state(mid_ctrl) == MEM_RESERVE && mem_state(mid_ctrl + 1 * mib) == MEM_RESERVE &&
                       mem_state(mid_ctrl + 4 * mib) == MEM_RESERVE,
                   "64 MiB control block starts lazy")) {
            if (mid_ctrl) VirtualFree(mid_ctrl, 0, MEM_RELEASE);
            return 4;
        }
        ComPtr<IUnknown> mid_unaligned;
        if (!require(invoke(guest, 133, mid_ctrl + 2 * mib + 0x100, &buf_64k, D3D12_RESOURCE_STATE_COMMON, nullptr,
                            &resource_iid, mid_unaligned.GetAddressOf()),
                     "slot 133 unaligned nonzero offset in 64 MiB lazy block")) {
            VirtualFree(mid_ctrl, 0, MEM_RELEASE);
            return 4;
        }
        const auto mid_unaligned_real = desktop_object<ID3D12Resource>(guest, device, mid_unaligned.Get());
        Snapshot after_ctrl_unaligned{};
        if (!snap(after_ctrl_unaligned)) return 4;
        if (!check(mid_unaligned_real &&
                       (mid_unaligned_real->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) == 0,
                   "64 MiB unaligned nonzero offset is not aliased") ||
            !check(after_ctrl_unaligned.live[heap].objects == 0,
                   "64 MiB unaligned nonzero offset does not open existing_heap") ||
            !check(mem_state(mid_ctrl) == MEM_RESERVE && mem_state(mid_ctrl + 1 * mib) == MEM_RESERVE &&
                       mem_state(mid_ctrl + 4 * mib) == MEM_RESERVE,
                   "64 MiB unaligned nonzero offset leaves untouched pages MEM_RESERVE")) {
            VirtualFree(mid_ctrl, 0, MEM_RELEASE);
            return 5;
        }
        const auto buf_128k = buffer_desc(2 * page);
        ComPtr<IUnknown> mid_oob;
        if (!require(invoke(guest, 133, mid_ctrl + mid_bytes - page, &buf_128k, D3D12_RESOURCE_STATE_COMMON, nullptr,
                            &resource_iid, mid_oob.GetAddressOf()),
                     "slot 133 out-of-bounds aligned offset in 64 MiB lazy block")) {
            VirtualFree(mid_ctrl, 0, MEM_RELEASE);
            return 4;
        }
        const auto mid_oob_real = desktop_object<ID3D12Resource>(guest, device, mid_oob.Get());
        Snapshot after_ctrl_oob{};
        if (!snap(after_ctrl_oob)) return 4;
        if (!check(mid_oob_real && (mid_oob_real->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) == 0,
                   "64 MiB out-of-bounds aligned offset is not aliased") ||
            !check(after_ctrl_oob.live[heap].objects == 0,
                   "64 MiB out-of-bounds aligned offset does not open existing_heap") ||
            !check(mem_state(mid_ctrl) == MEM_RESERVE && mem_state(mid_ctrl + 1 * mib) == MEM_RESERVE &&
                       mem_state(mid_ctrl + 4 * mib) == MEM_RESERVE,
                   "64 MiB out-of-bounds aligned offset leaves untouched pages MEM_RESERVE")) {
            VirtualFree(mid_ctrl, 0, MEM_RELEASE);
            return 5;
        }
        mid_unaligned.Reset();
        mid_oob.Reset();
        if (!check(VirtualFree(mid_ctrl, 0, MEM_RELEASE) != 0, "release 64 MiB control block")) return 4;

        // <= 256 MiB lazy startup block (64 MiB) at nonzero aligned offset (+2 MiB): [0, 2 MiB) starts
        // MEM_RESERVE before slot 133, commits whole when eligible, aliases, and sees late A-to-B CPU writes.
        auto* const mid_block = static_cast<char*>(
            g_xmem_virtual_alloc(nullptr, mid_bytes, 0x20403000, 0x181, PAGE_READWRITE));
        if (!check(mid_block != nullptr, "XMemVirtualAlloc 64 MiB block") ||
            !check(mem_state(mid_block) == MEM_RESERVE && mem_state(mid_block + 1 * mib) == MEM_RESERVE &&
                       mem_state(mid_block + 2 * mib) == MEM_RESERVE,
                   "64 MiB startup block starts lazy"))
            return 4;
        ComPtr<IUnknown> mid_buf;
        if (!require(invoke(guest, 133, mid_block + 2 * mib, &buf_64k, D3D12_RESOURCE_STATE_COMMON, nullptr,
                            &resource_iid, mid_buf.GetAddressOf()),
                     "slot 133 aligned +2 MiB buffer in 64 MiB lazy block"))
            return 4;
        const auto mid_real = desktop_object<ID3D12Resource>(guest, device, mid_buf.Get());
        Snapshot after_64m{};
        if (!snap(after_64m)) return 4;
        if (!check(mid_real && (mid_real->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) != 0,
                   "64 MiB <= 256 MiB lazy block buffer is aliased") ||
            !check(after_64m.live[heap].bytes == mid_bytes && after_64m.live[heap].objects == 1,
                   "64 MiB block opens existing_heap") ||
            !check(mem_state(mid_block) == MEM_COMMIT && mem_state(mid_block + 1 * mib) == MEM_COMMIT &&
                       mem_state(mid_block + 2 * mib) == MEM_COMMIT,
                   "64 MiB lazy block committed right before OpenExistingHeapFromAddress")) {
            return 5;
        }
        auto* const mid_cpu = reinterpret_cast<std::uint8_t*>(mid_block + 2 * mib);
        for (UINT64 i = 0; i < page; ++i) mid_cpu[i] = pattern_byte(10, i);
        if (!verify_buffer_pattern(device, own_queue.Get(), own_alloc.Get(), own_list.Get(), own_fence.Get(), own_val,
                                   mid_real.Get(), page, 10, "64 MiB +2 MiB initial A"))
            return 5;
        for (UINT64 i = 0; i < page; ++i) mid_cpu[i] = pattern_byte(11, i);
        if (!verify_buffer_pattern(device, own_queue.Get(), own_alloc.Get(), own_list.Get(), own_fence.Get(), own_val,
                                   mid_real.Get(), page, 11, "64 MiB +2 MiB late write B"))
            return 5;
        mid_buf.Reset();
        std::printf("existing_heaps <= 256 MiB aliased cleanly (4 MiB + 64 MiB lazy +2 MiB A-to-B)\n");
    } else {
        std::puts("skipping <= 256 MiB existing_heap check: D3D12_FEATURE_EXISTING_HEAPS=0");
    }

    // -------------------------------------------------------------------------
    // Part 2: reserve_upload_ring incremental growth (16 MiB start, grows > 16 MiB)
    // -------------------------------------------------------------------------
    Snapshot pre_upload{};
    if (!snap(pre_upload)) return 4;
    if (!check(pre_upload.live[upload].bytes == 0 && pre_upload.live[upload].objects == 0,
               "upload budget starts at 0/0 before any upload"))
        return 5;

    const unsigned list_desc[] = {static_cast<unsigned>(D3D12_COMMAND_LIST_TYPE_DIRECT), 2, 1, 0};
    ComPtr<IUnknown> queue, allocator, allocator2, list, fence;
    if (!require(invoke(guest, 8, &qd, &__uuidof(ID3D12CommandQueue), queue.GetAddressOf()), "guest queue") ||
        !require(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator), allocator.GetAddressOf()),
                 "guest allocator") ||
        !require(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator), allocator2.GetAddressOf()),
                 "guest allocator2") ||
        !require(invoke(guest, 73, list_desc, allocator.Get(), nullptr, &__uuidof(ID3D12GraphicsCommandList),
                        list.GetAddressOf()),
                 "guest list") ||
        !require(invoke(guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), fence.GetAddressOf()),
                 "guest fence") ||
        !require(invoke(list.Get(), 139, allocator.Get(), nullptr), "guest list reset"))
        return 4;
    const auto real_fence = desktop_object<ID3D12Fence>(guest, device, fence.Get());
    if (!real_fence) return 4;

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto make_slot133_tex = [&](UINT w, UINT h, int seed, std::vector<std::uint8_t>& backing,
                                      ComPtr<IUnknown>& src, ComPtr<IUnknown>& dst) {
        backing.resize(UINT64{w} * h * 4);
        for (UINT64 i = 0; i < backing.size(); ++i) backing[i] = pattern_byte(seed, i);
        const auto td = tex2d_desc(w, h);
        return require(invoke(guest, 133, backing.data(), &td, D3D12_RESOURCE_STATE_COMMON, nullptr,
                              &resource_iid, src.GetAddressOf()),
                       "slot 133 tex") &&
               require(invoke(guest, 127, &default_heap, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COMMON,
                              nullptr, nullptr, 0u, nullptr, &resource_iid, dst.GetAddressOf()),
                       "slot 127 dst tex");
    };

    if (video) {
        // Part 3 (video variant): forty video-sized CPU-written textures (1920x1080 RGBA8 = 8.3 MB, the bytes of a 3840x2160
        // R8 plane), each copied to the desktop, submitted and read back. Every upload gets its own upload buffer: the first
        // one shows one buffer of about the texture's size (the shared ring would show a 16 MiB chunk), and every texture
        // reads back its own pattern at both corners.
        constexpr UINT video_w = 1920, video_h = 1080;
        constexpr int video_count = 40;
        for (int i = 0; i < video_count; ++i) {
            std::vector<std::uint8_t> backing;
            ComPtr<IUnknown> src, dst, video_allocator;
            if (!make_slot133_tex(video_w, video_h, 20 + i, backing, src, dst)) return 4;
            // The first texture records into the list as created; later ones start a new recording with a fresh allocator.
            if (i > 0 &&
                (!require(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator),
                                 video_allocator.GetAddressOf()),
                          "video allocator") ||
                 !require(invoke(list.Get(), 139, video_allocator.Get(), nullptr), "video list reset")))
                return 4;
            invoke<void>(list.Get(), 17, dst.Get(), src.Get());
            if (i == 0) {
                Snapshot after_first{};
                if (!snap(after_first)) return 4;
                if (!check(after_first.live[upload].objects == 1 && after_first.live[upload].bytes >= UINT64{video_w} * video_h * 4 &&
                               after_first.live[upload].bytes < 16 * mib,
                           "a video-sized upload owns one buffer of about its size (not a 16 MiB ring chunk)")) {
                    std::printf("  got upload=%llu/%llu\n", after_first.live[upload].bytes, after_first.live[upload].objects);
                    return 5;
                }
            }
            if (!require(invoke(list.Get(), 9), "video list close")) return 4;
            IUnknown* video_lists[] = {list.Get()};
            invoke<void>(queue.Get(), 10, 1u, video_lists);
            if (!require(invoke(queue.Get(), 14, fence.Get(), ++guest_val), "video queue signal")) return 4;
            HANDLE video_ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!video_ev) return 4;
            const bool video_done = SUCCEEDED(real_fence->SetEventOnCompletion(guest_val, video_ev)) &&
                                    WaitForSingleObject(video_ev, wait_ms) == WAIT_OBJECT_0;
            CloseHandle(video_ev);
            if (!check(video_done, "video queue wait")) return 4;
            const auto dst_real = desktop_object<ID3D12Resource>(guest, device, dst.Get());
            char label[64];
            std::snprintf(label, sizeof(label), "video texture %d top-left", i);
            if (!verify_texture_box(device, own_queue.Get(), own_alloc.Get(), own_list.Get(), own_fence.Get(), own_val,
                                    dst_real.Get(), 0, 0, 64, 64, video_w, 20 + i, label))
                return 5;
            std::snprintf(label, sizeof(label), "video texture %d bottom-right", i);
            if (!verify_texture_box(device, own_queue.Get(), own_alloc.Get(), own_list.Get(), own_fence.Get(), own_val,
                                    dst_real.Get(), video_w - 64, video_h - 64, 64, 64, video_w, 20 + i, label))
                return 5;
        }
        const UINT64 video_errors = debug ? debug->GetNumStoredMessages() : 0;
        std::printf("video uploads: %d textures of %ux%u read back exact; debug_errors_or_warnings=%llu\n", video_count, video_w,
                    video_h, video_errors);
        if (video_errors || FAILED(device->GetDeviceRemovedReason())) return 4;
        std::puts("PASS");
        return 0;
    }

    std::vector<std::uint8_t> small_mem, small2_mem, big_mem, small3_mem;
    ComPtr<IUnknown> small_src, small_dst, small2_src, small2_dst, big_src, big_dst, small3_src, small3_dst;
    if (!make_slot133_tex(64, 64, 1, small_mem, small_src, small_dst) ||
        !make_slot133_tex(64, 16, 2, small2_mem, small2_src, small2_dst) ||
        !make_slot133_tex(2048, 2048, 3, big_mem, big_src, big_dst) ||
        !make_slot133_tex(64, 64, 4, small3_mem, small3_src, small3_dst))
        return 4;

    // First small upload (16 KiB): upload ring starts at 16 MiB (1 object), NOT 256 MiB.
    invoke<void>(list.Get(), 17, small_dst.Get(), small_src.Get());
    Snapshot after_small1{};
    if (!snap(after_small1)) return 4;
    if (!check(after_small1.live[upload].bytes == 16 * mib && after_small1.live[upload].objects == 1,
               "first upload reserves 16 MiB / 1 chunk (not 256 MiB)")) {
        std::printf("  got upload=%llu/%llu\n", after_small1.live[upload].bytes, after_small1.live[upload].objects);
        return 5;
    }

    // Second small upload (4 KiB): fits in chunk 0 (remains 16 MiB / 1 object).
    invoke<void>(list.Get(), 17, small2_dst.Get(), small2_src.Get());
    Snapshot after_small2{};
    if (!snap(after_small2)) return 4;
    if (!check(after_small2.live[upload].bytes == 16 * mib && after_small2.live[upload].objects == 1,
               "second small upload reuses 16 MiB chunk"))
        return 5;

    // Submit & verify small1 and small2 before growing past 16 MiB.
    if (!require(invoke(list.Get(), 9), "list close 1")) return 4;
    IUnknown* lists1[] = {list.Get()};
    invoke<void>(queue.Get(), 10, 1u, lists1);
    if (!require(invoke(queue.Get(), 14, fence.Get(), ++guest_val), "queue signal 1")) return 4;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return 4;
    const bool done1 = SUCCEEDED(real_fence->SetEventOnCompletion(guest_val, ev)) &&
                       WaitForSingleObject(ev, wait_ms) == WAIT_OBJECT_0;
    CloseHandle(ev);
    if (!check(done1, "guest queue wait 1")) return 4;

    const auto small_dst_real = desktop_object<ID3D12Resource>(guest, device, small_dst.Get());
    const auto small2_dst_real = desktop_object<ID3D12Resource>(guest, device, small2_dst.Get());
    if (!verify_texture_box(device, own_queue.Get(), own_alloc.Get(), own_list.Get(), own_fence.Get(), own_val,
                            small_dst_real.Get(), 0, 0, 64, 64, 64, 1, "small1 (chunk 0)") ||
        !verify_texture_box(device, own_queue.Get(), own_alloc.Get(), own_list.Get(), own_fence.Get(), own_val,
                            small2_dst_real.Get(), 0, 0, 64, 16, 64, 2, "small2 (chunk 0)"))
        return 5;

    // Next upload: 2048x2048 RGBA8 (16 MiB exact) > remaining space in chunk 0 -> appends 32 MiB chunk (48 MiB / 2).
    if (!require(invoke(list.Get(), 139, allocator2.Get(), nullptr), "guest list reset 2")) return 4;
    invoke<void>(list.Get(), 17, big_dst.Get(), big_src.Get());
    Snapshot after_big{};
    if (!snap(after_big)) return 4;
    if (!check(after_big.live[upload].objects == 2 && after_big.live[upload].bytes == 48 * mib,
               "upload > 16 MiB appends 32 MiB chunk (48 MiB / 2 objects)")) {
        std::printf("  got upload=%llu/%llu\n", after_big.live[upload].bytes, after_big.live[upload].objects);
        return 5;
    }

    // Subsequent small upload lands in chunk 1 without adding another chunk.
    invoke<void>(list.Get(), 17, small3_dst.Get(), small3_src.Get());
    Snapshot after_small3{};
    if (!snap(after_small3)) return 4;
    if (!check(after_small3.live[upload].objects == 2 && after_small3.live[upload].bytes == 48 * mib,
               "third small upload reuses chunk 1 (48 MiB / 2 objects)"))
        return 5;

    if (!require(invoke(list.Get(), 9), "list close 2")) return 4;
    IUnknown* lists2[] = {list.Get()};
    invoke<void>(queue.Get(), 10, 1u, lists2);
    if (!require(invoke(queue.Get(), 14, fence.Get(), ++guest_val), "queue signal 2")) return 4;
    ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return 4;
    const bool done2 = SUCCEEDED(real_fence->SetEventOnCompletion(guest_val, ev)) &&
                       WaitForSingleObject(ev, wait_ms) == WAIT_OBJECT_0;
    CloseHandle(ev);
    if (!check(done2, "guest queue wait 2")) return 4;

    const auto big_dst_real = desktop_object<ID3D12Resource>(guest, device, big_dst.Get());
    const auto small3_dst_real = desktop_object<ID3D12Resource>(guest, device, small3_dst.Get());
    if (!verify_texture_box(device, own_queue.Get(), own_alloc.Get(), own_list.Get(), own_fence.Get(), own_val,
                            big_dst_real.Get(), 0, 0, 64, 64, 2048, 3, "big top-left (chunk 1)") ||
        !verify_texture_box(device, own_queue.Get(), own_alloc.Get(), own_list.Get(), own_fence.Get(), own_val,
                            big_dst_real.Get(), 1984, 1984, 64, 64, 2048, 3, "big bottom-right (chunk 1)") ||
        !verify_texture_box(device, own_queue.Get(), own_alloc.Get(), own_list.Get(), own_fence.Get(), own_val,
                            small3_dst_real.Get(), 0, 0, 64, 64, 64, 4, "small3 (chunk 1)"))
        return 5;

    const UINT64 errors = debug ? debug->GetNumStoredMessages() : 0;
    std::printf("upload_ring grew 16 MiB -> 48 MiB (2 chunks); debug_errors_or_warnings=%llu\n", errors);
    if (errors || FAILED(device->GetDeviceRemovedReason())) return 4;

    std::puts("PASS");
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) ||
        (argc == 3 && std::strcmp(argv[2], "cap") && std::strcmp(argv[2], "video"))) {
        std::puts("usage: startup_slot133_heap_cap_check <warp|hardware> [cap|video]");
        return 2;
    }
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    const bool video = argc == 3 && std::strcmp(argv[2], "video") == 0;
    const bool cap = argc == 3;
    if (!load_shims(cap, video)) return 3;

    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), warp);
    if (!adapter) return 77;

    char debug_env[2]{};
    const bool debug_on = GetEnvironmentVariableA("X2W_D3D12_DEBUG", debug_env, sizeof(debug_env)) && debug_env[0] == '1';
    if (debug_on) {
        ComPtr<ID3D12Debug> debug_iface;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_iface)))) return 77;
    }

    ComPtr<IUnknown> guest;
    ComPtr<ID3D12Device> device;
    std::array<unsigned char, 64> parameters{};
    if (!require(g_create_xbox_device(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                      reinterpret_cast<void**>(guest.GetAddressOf())),
                 "Xbox device") ||
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

    DXGI_ADAPTER_DESC1 desc{};
    LARGE_INTEGER version{};
    adapter->GetDesc1(&desc);
    adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &version);
    const auto driver = static_cast<UINT64>(version.QuadPart);
    std::printf("adapter=%ls driver=%llu.%llu.%llu.%llu debug_layer=%s\n", desc.Description,
                driver >> 48, (driver >> 32) & 0xffff, (driver >> 16) & 0xffff, driver & 0xffff,
                debug ? "enabled" : "off");
    return run_check(guest.Get(), device.Get(), debug.Get(), cap, video);
}
