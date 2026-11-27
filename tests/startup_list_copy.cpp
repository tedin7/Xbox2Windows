// CopyResource, ResolveSubresource and CopyBufferRegion on a recording list (slots 17, 19 and 15).
//
//   startup_list_copy <warp|hardware>     (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1 X2W_D3D12_DEBUG=1)
//
// Through D3D12XboxCreateDevice the probe records one list over desktop resources and reads them back on
// a desktop queue:
//
//   1. a 4x4 target cleared red, CopyResource into a second texture             second texture is red
//   2. the same copy with source and destination swapped (an empty source into a blue target)
//                                                                               the empty texture stays empty
//   3. a 4x4 four-sample target cleared green, ResolveSubresource into a one-sample texture   green
//   4. the same with a typeless half-float pair and a typeless format argument   half float 1.0 in red
//   5. an upload buffer copied to a buffer, then 64 bytes at another offset into a third   exact bytes
//   6. refused and counted, forwarded to nothing: a CopyResource between different sizes, a null source,
//      a resolve from a one-sample source, a buffer range that runs past its end, a copy into an upload
//      buffer and a copy out of a readback buffer
//   7. a title-backed texture copied to a texture: the destination holds its texels; one whose memory cannot
//      be read is refused and the destination stays empty
//   8. after Close, the same calls change no counter
//   9. a copy into a title-backed texture, discarded by Close and Reset unsubmitted: a later copy out of that
//      texture still reads its own memory
//
// The desktop device must not be removed and the debug layer must store no errors.
//
// Exit codes: 0 pass, 77 no physical adapter, other values name the failing step.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void x2w_d3d12_list_copy_stats(unsigned* buffer_copies, unsigned* resource_copies,
                                                                unsigned* resolves, unsigned* refused);

extern "C" __declspec(dllimport) void x2w_d3d12_fixture_copy15(BOOL skip_probes);
extern "C" __declspec(dllimport) void x2w_d3d12_fixture_hold_draw_lock(HANDLE acquired, DWORD milliseconds);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0, Word f = 0, Word g = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f, g);
}
bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }

constexpr float red[] = {1, 0, 0, 1};
constexpr float green[] = {0, 1, 0, 1};
constexpr float blue[] = {0, 0, 1, 1};
constexpr std::uint32_t red_bits = 0xff0000ff;
constexpr std::uint32_t green_bits = 0xff00ff00;
constexpr std::uint32_t half_one = 0x3c00;

struct Counts {
    unsigned buffers = 0, resources = 0, resolves = 0, refused = 0;
    static Counts read()
    {
        Counts c;
        x2w_d3d12_list_copy_stats(&c.buffers, &c.resources, &c.resolves, &c.refused);
        return c;
    }
};

bool step(const char* name, Counts expected_delta, Counts& before)
{
    const Counts now = Counts::read();
    const bool good = now.buffers - before.buffers == expected_delta.buffers && now.resources - before.resources == expected_delta.resources &&
                      now.resolves - before.resolves == expected_delta.resolves && now.refused - before.refused == expected_delta.refused;
    std::printf("%s: buffers +%u (%u), resources +%u (%u), resolves +%u (%u), refused +%u (%u)\n", name, now.buffers - before.buffers,
                expected_delta.buffers, now.resources - before.resources, expected_delta.resources, now.resolves - before.resolves,
                expected_delta.resolves, now.refused - before.refused, expected_delta.refused);
    before = now;
    return good;
}

D3D12_RESOURCE_DESC texture(UINT size, DXGI_FORMAT format, UINT samples, D3D12_RESOURCE_FLAGS flags)
{
    return {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, size, size, 1, 1, format, {samples, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, flags};
}
D3D12_RESOURCE_DESC buffer(UINT64 width)
{
    return {D3D12_RESOURCE_DIMENSION_BUFFER, 0, width, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            D3D12_RESOURCE_FLAG_NONE};
}

struct Probe {
    ComPtr<ID3D12Device> desktop;
    ComPtr<ID3D12Resource> readback;
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
    UINT64 debug_errors()
    {
        ComPtr<ID3D12InfoQueue> info;
        if (FAILED(desktop.As(&info))) return 0;
        UINT64 errors = 0;
        for (UINT64 i = 0, n = info->GetNumStoredMessages(); i < n; ++i) {
            SIZE_T bytes = 0;
            if (FAILED(info->GetMessage(i, nullptr, &bytes))) continue;
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if (FAILED(info->GetMessage(i, message, &bytes))) continue;
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                ++errors;
                std::printf("debug layer: %.*s\n", static_cast<int>(message->DescriptionByteLength), message->pDescription);
            }
        }
        return errors;
    }
    // The first bytes of a resource (a buffer's first `count`, a texture's first texels), copied on a desktop queue.
    bool bytes(ID3D12Resource* target, std::size_t count, std::vector<std::uint8_t>& out)
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
            return false;
        const D3D12_RESOURCE_DESC described = target->GetDesc();
        if (described.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            copy->CopyBufferRegion(readback.Get(), 0, target, 0, count);
        } else {
            D3D12_TEXTURE_COPY_LOCATION to{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            desktop->GetCopyableFootprints(&described, 0, 1, 0, &to.PlacedFootprint, nullptr, nullptr, nullptr);
            D3D12_TEXTURE_COPY_LOCATION from{target, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            copy->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        }
        if (FAILED(copy->Close())) return false;
        ID3D12CommandList* lists[] = {copy.Get()};
        observer->ExecuteCommandLists(1, lists);
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool finished = event && SUCCEEDED(observer->Signal(done.Get(), 1)) &&
                              SUCCEEDED(done->SetEventOnCompletion(1, event)) && WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
        if (event) CloseHandle(event);
        void* memory = nullptr;
        if (!finished || FAILED(readback->Map(0, nullptr, &memory))) return false;
        out.assign(static_cast<std::uint8_t*>(memory), static_cast<std::uint8_t*>(memory) + count);
        readback->Unmap(0, nullptr);
        return true;
    }
    std::uint32_t texel(ID3D12Resource* target)
    {
        std::vector<std::uint8_t> data;
        std::uint32_t bits = 0xdeadbeef;
        if (bytes(target, 4, data)) std::memcpy(&bits, data.data(), sizeof(bits));
        return bits;
    }
};
// Catch only the intended mutation fault. An unrelated exception still fails the child.
int copy15_source_fault(EXCEPTION_POINTERS* exception, const void* source)
{
    const auto& record = *exception->ExceptionRecord;
    return record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record.NumberParameters >= 2 &&
        record.ExceptionInformation[0] == 0 && record.ExceptionInformation[1] >= arg(source) &&
        record.ExceptionInformation[1] - arg(source) < 64
        ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH;
}

bool copy15_protected_source(void* list, void* to, void* from, const void* source)
{
    __try {
        invoke(list, 15, arg(to), 0, arg(from), 0, 64);
    } __except (copy15_source_fault(GetExceptionInformation(), source)) {
        return false;
    }
    return true;
}

int check_copy15(void* device, Probe& probe, void* list, ID3D12Resource* upload, bool mutate)
{
    // The 4 KiB offset cannot be aliased by slot 133. Give each placement its
    // own allocation and leave the first page outside the buffer.
    constexpr std::size_t page = 4096, payload = 256, chunk = 64;
    auto* source = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 2 * page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    auto* destination = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 2 * page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!source || !destination) return 90;
    source += page;
    destination += page;
    void *from = nullptr, *to = nullptr;
    const auto place = [&](void* memory, const D3D12_RESOURCE_DESC& desc, void** out) {
        return ok(invoke(device, 133, arg(memory), arg(&desc), D3D12_RESOURCE_STATE_COMMON, 0,
                         arg(&__uuidof(ID3D12Resource)), arg(out))) && *out;
    };
    if (!place(source, buffer(payload), &from) || !place(destination, buffer(payload), &to))
        return 90;

    std::array<std::uint8_t, payload> expected{};
    for (std::size_t i = 0; i < payload; ++i)
        expected[i] = source[i] = static_cast<std::uint8_t>(i * 13 + (i >> 3) + 1);

    HANDLE acquired = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!acquired) return 90;
    Counts seen = Counts::read();
    x2w_d3d12_fixture_copy15(FALSE);
    double elapsed_us = 0;
    for (unsigned batch = 0; batch < 20; ++batch) {
        std::thread holder;
        if (batch % 2) {
            holder = std::thread(x2w_d3d12_fixture_hold_draw_lock, acquired, 50);
            if (WaitForSingleObject(acquired, 5000) != WAIT_OBJECT_0) {
                holder.join();
                CloseHandle(acquired);
                return 90;
            }
        }
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < 100; ++i) {
            const Word offset = (i % 4) * chunk;
            invoke(list, 15, arg(to), offset, arg(from), offset, chunk);
        }
        elapsed_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        if (holder.joinable()) holder.join();
    }
    CloseHandle(acquired);
    x2w_d3d12_fixture_copy15(mutate);
    std::printf("copy15 record us %.3f\n", elapsed_us);
    if (!step("copy15 CPU copies", {}, seen)) return 91;

    // Change protection after successful copies warmed the same page.
    DWORD previous = 0;
    if (!VirtualProtect(source, page, PAGE_NOACCESS, &previous)) return 90;
    if (!copy15_protected_source(list, to, from, source)) {
        std::puts("copy15 unreadable source fault");
        return 92;
    }
    if (!step("copy15 unreadable source", {0, 0, 0, 1}, seen)) {
        std::puts("copy15 missing unreadable source refusal");
        return 93;
    }
    if (!VirtualProtect(source, page, previous, &previous)) return 90;

    invoke(list, 15, arg(from), 0, arg(upload), 0, chunk);
    if (!step("copy15 GPU producer", {1, 0, 0, 0}, seen)) return 94;
    invoke(list, 15, arg(to), 0, arg(from), 0, chunk);
    if (!step("copy15 GPU-written source", {0, 0, 0, 1}, seen)) return 95;

    if (std::memcmp(destination, expected.data(), expected.size())) return 97;
    if (!ok(invoke(list, 9)) || !probe.submit(list)) return 96;
    return probe.desktop->GetDeviceRemovedReason() == S_OK && !probe.debug_errors() ? 0 : 98;
}

}  // namespace

int main(int argc, char** argv)
{
    const bool copy15 = argc == 3 && (!std::strcmp(argv[2], "copy15") || !std::strcmp(argv[2], "copy15-mutant"));
    if (argc != 2 && !copy15) return 2;
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
    } else if (std::strcmp(argv[1], "warp") != 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 3;
    }
    DXGI_ADAPTER_DESC adapter_desc{};
    if (FAILED(adapter->GetDesc(&adapter_desc))) return 3;
    std::printf("adapter=%ls\n", adapter_desc.Description);

    Probe probe;
    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 4;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&probe.desktop)))) return 4;

    // Resources. Targets that are cleared carry the render target flag and a typed clear value.
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto create = [&](const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear, ComPtr<ID3D12Resource>& out) {
        return SUCCEEDED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, clear, IID_PPV_ARGS(&out)));
    };
    const D3D12_CLEAR_VALUE clear_rgba{DXGI_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 0}};
    const D3D12_CLEAR_VALUE clear_half{DXGI_FORMAT_R16G16B16A16_FLOAT, {0, 0, 0, 0}};
    const auto rt = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, none = D3D12_RESOURCE_FLAG_NONE;
    ComPtr<ID3D12Resource> source_a, copy_b, source_c, empty_d, large_e, msaa_m, resolved_r, msaa_h, resolved_h;
    ComPtr<ID3D12Resource> upload_u, buffer_x, buffer_y;
    if (!create(texture(4, DXGI_FORMAT_R8G8B8A8_UNORM, 1, rt), D3D12_RESOURCE_STATE_COMMON, &clear_rgba, source_a) ||
        !create(texture(4, DXGI_FORMAT_R8G8B8A8_UNORM, 1, none), D3D12_RESOURCE_STATE_COMMON, nullptr, copy_b) ||
        !create(texture(4, DXGI_FORMAT_R8G8B8A8_UNORM, 1, rt), D3D12_RESOURCE_STATE_COMMON, &clear_rgba, source_c) ||
        !create(texture(4, DXGI_FORMAT_R8G8B8A8_UNORM, 1, none), D3D12_RESOURCE_STATE_COMMON, nullptr, empty_d) ||
        !create(texture(8, DXGI_FORMAT_R8G8B8A8_UNORM, 1, none), D3D12_RESOURCE_STATE_COMMON, nullptr, large_e) ||
        !create(texture(4, DXGI_FORMAT_R8G8B8A8_UNORM, 4, rt), D3D12_RESOURCE_STATE_COMMON, &clear_rgba, msaa_m) ||
        !create(texture(4, DXGI_FORMAT_R8G8B8A8_UNORM, 1, none), D3D12_RESOURCE_STATE_COMMON, nullptr, resolved_r) ||
        !create(texture(4, DXGI_FORMAT_R16G16B16A16_TYPELESS, 4, rt), D3D12_RESOURCE_STATE_COMMON, &clear_half, msaa_h) ||
        !create(texture(4, DXGI_FORMAT_R16G16B16A16_TYPELESS, 1, none), D3D12_RESOURCE_STATE_COMMON, nullptr, resolved_h) ||
        !create(buffer(256), D3D12_RESOURCE_STATE_COMMON, nullptr, buffer_x) ||
        !create(buffer(256), D3D12_RESOURCE_STATE_COMMON, nullptr, buffer_y))
        return 5;
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    const D3D12_RESOURCE_DESC upload_desc = buffer(256);
    if (FAILED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &upload_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&upload_u))))
        return 5;
    std::array<std::uint8_t, 256> pattern{};
    for (std::size_t i = 0; i < pattern.size(); ++i) pattern[i] = static_cast<std::uint8_t>(i * 3 + 1);
    void* mapped = nullptr;
    if (FAILED(upload_u->Map(0, nullptr, &mapped))) return 5;
    std::memcpy(mapped, pattern.data(), pattern.size());
    upload_u->Unmap(0, nullptr);
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    const D3D12_RESOURCE_DESC readback_desc = buffer(4096);
    if (FAILED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                      nullptr, IID_PPV_ARGS(&probe.readback))))
        return 5;

    // Render target views of the four targets that are cleared.
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4}, 0, 0};
    void* rtv_heap = nullptr;
    if (!ok(invoke(device, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&rtv_heap)))) return 6;
    const UINT stride = probe.desktop->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const SIZE_T first = static_cast<SIZE_T>(invoke(rtv_heap, 9));
    const D3D12_CPU_DESCRIPTOR_HANDLE view_a{first}, view_c{first + stride}, view_m{first + 2 * stride}, view_h{first + 3 * stride};
    D3D12_RENDER_TARGET_VIEW_DESC flat{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2D, {}};
    D3D12_RENDER_TARGET_VIEW_DESC multisampled{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2DMS, {}};
    D3D12_RENDER_TARGET_VIEW_DESC multisampled_half{DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RTV_DIMENSION_TEXTURE2DMS, {}};
    invoke(device, 20, arg(source_a.Get()), arg(&flat), view_a.ptr);
    invoke(device, 20, arg(source_c.Get()), arg(&flat), view_c.ptr);
    invoke(device, 20, arg(msaa_m.Get()), arg(&multisampled), view_m.ptr);
    invoke(device, 20, arg(msaa_h.Get()), arg(&multisampled_half), view_h.ptr);

    // Title-backed textures, placed at addresses of the title's own memory (device slot 133). One holds a known
    // pattern; the other has its pages decommitted, so no upload can stage it.
    constexpr std::uint32_t title_texel = 0x44332211;
    constexpr std::size_t title_bytes = 0x10000;
    auto* const readable_memory = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, title_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    auto* const gone_memory = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, title_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!readable_memory || !gone_memory) return 30;
    for (std::size_t i = 0; i < title_bytes / sizeof(std::uint32_t); ++i) readable_memory[i] = title_texel;
    VirtualFree(gone_memory, title_bytes, MEM_DECOMMIT);
    // A third placed texture whose only copy is recorded and then discarded without a submission.
    auto* const discard_memory = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, title_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!discard_memory) return 30;
    for (std::size_t i = 0; i < title_bytes / sizeof(std::uint32_t); ++i) discard_memory[i] = title_texel;
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    const D3D12_RESOURCE_DESC title_desc = texture(4, DXGI_FORMAT_R8G8B8A8_UNORM, 1, none);
    void *title_readable = nullptr, *title_gone = nullptr;
    void* title_discarded = nullptr;
    ComPtr<ID3D12Resource> dest_readable, dest_gone, discard_out;
    if (!create(title_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, dest_readable) ||
        !create(title_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, discard_out) ||
        !create(title_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, dest_gone))
        return 31;
    if (!ok(invoke(device, 133, arg(readable_memory), arg(&title_desc), D3D12_RESOURCE_STATE_COMMON, 0, arg(&__uuidof(ID3D12Resource)),
                   arg(&title_readable))) || !title_readable ||
        !ok(invoke(device, 133, arg(discard_memory), arg(&title_desc), D3D12_RESOURCE_STATE_COMMON, 0, arg(&__uuidof(ID3D12Resource)),
                   arg(&title_discarded))) || !title_discarded ||
        !ok(invoke(device, 133, arg(gone_memory), arg(&title_desc), D3D12_RESOURCE_STATE_COMMON, 0, arg(&__uuidof(ID3D12Resource)),
                   arg(&title_gone))) || !title_gone)
        return 32;

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void *allocator = nullptr, *list = nullptr;
    if (!ok(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&probe.queue))) ||
        !ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&probe.fence))) ||
        !ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))))
        return 7;
    if (!ok(invoke(list, 139, arg(allocator), 0))) return 8;
    if (copy15) return check_copy15(device, probe, list, upload_u.Get(), !std::strcmp(argv[2], "copy15-mutant"));

    invoke(list, 52, view_a.ptr, arg(red));
    invoke(list, 52, view_c.ptr, arg(blue));
    invoke(list, 52, view_m.ptr, arg(green));
    invoke(list, 52, view_h.ptr, arg(red));

    Counts seen = Counts::read();
    invoke(list, 17, arg(copy_b.Get()), arg(source_a.Get()));
    if (!step("1 CopyResource", {0, 1, 0, 0}, seen)) return 10;
    invoke(list, 17, arg(source_c.Get()), arg(empty_d.Get()));  // swapped: the empty texture is the source
    if (!step("2 CopyResource, source and destination swapped", {0, 1, 0, 0}, seen)) return 11;
    invoke(list, 19, arg(resolved_r.Get()), 0, arg(msaa_m.Get()), 0, DXGI_FORMAT_R8G8B8A8_UNORM);
    if (!step("3 ResolveSubresource", {0, 0, 1, 0}, seen)) return 12;
    invoke(list, 19, arg(resolved_h.Get()), 0, arg(msaa_h.Get()), 0, DXGI_FORMAT_R16G16B16A16_TYPELESS);
    if (!step("4 ResolveSubresource, typeless half float", {0, 0, 1, 0}, seen)) return 13;
    invoke(list, 15, arg(buffer_x.Get()), 0, arg(upload_u.Get()), 0, 256);
    invoke(list, 15, arg(buffer_y.Get()), 16, arg(buffer_x.Get()), 32, 64);
    if (!step("5 CopyBufferRegion twice", {2, 0, 0, 0}, seen)) return 14;

    invoke(list, 17, arg(large_e.Get()), arg(source_a.Get()));
    if (!step("6a CopyResource between different sizes", {0, 0, 0, 1}, seen)) return 15;
    invoke(list, 17, arg(copy_b.Get()), 0);
    if (!step("6b CopyResource with a null source", {0, 0, 0, 1}, seen)) return 16;
    invoke(list, 19, arg(resolved_r.Get()), 0, arg(source_a.Get()), 0, DXGI_FORMAT_R8G8B8A8_UNORM);
    if (!step("6c ResolveSubresource from one sample", {0, 0, 0, 1}, seen)) return 17;
    invoke(list, 15, arg(buffer_y.Get()), 0, arg(buffer_x.Get()), 200, 100);
    if (!step("6d CopyBufferRegion past the end", {0, 0, 0, 1}, seen)) return 18;
    invoke(list, 15, arg(upload_u.Get()), 0, arg(buffer_x.Get()), 0, 16);
    if (!step("6e CopyBufferRegion into an upload buffer", {0, 0, 0, 1}, seen)) return 28;
    invoke(list, 15, arg(buffer_y.Get()), 0, arg(probe.readback.Get()), 0, 16);
    if (!step("6f CopyBufferRegion out of a readback buffer", {0, 0, 0, 1}, seen)) return 29;

    // A title-backed texture source is uploaded first, and the destination holds its texels. One whose memory
    // cannot be staged is refused, and the destination keeps what it had.
    invoke(list, 17, arg(dest_readable.Get()), arg(title_readable));
    if (!step("8a CopyResource from a staged title-backed texture", {0, 1, 0, 0}, seen)) return 33;
    invoke(list, 17, arg(dest_gone.Get()), arg(title_gone));
    if (!step("8b CopyResource from an unreadable title-backed texture", {0, 0, 0, 1}, seen)) return 34;

    invoke(list, 9);
    if (!probe.submit(list)) return 19;
    invoke(list, 17, arg(copy_b.Get()), arg(empty_d.Get()));
    invoke(list, 19, arg(resolved_r.Get()), 0, arg(msaa_m.Get()), 0, DXGI_FORMAT_R8G8B8A8_UNORM);
    invoke(list, 15, arg(buffer_y.Get()), 0, arg(buffer_x.Get()), 0, 16);
    if (!step("9 calls on a closed list", {0, 0, 0, 0}, seen)) return 20;

    const std::uint32_t copied = probe.texel(copy_b.Get());
    const std::uint32_t swapped_source = probe.texel(empty_d.Get());
    const std::uint32_t swapped_target = probe.texel(source_c.Get());
    const std::uint32_t resolved = probe.texel(resolved_r.Get());
    const std::uint32_t resolved_half = probe.texel(resolved_h.Get());
    std::printf("copy: %08x (expected %08x)\n", copied, red_bits);
    std::printf("swapped copy, source stays empty: %08x (expected 00000000), target now empty: %08x (expected 00000000)\n",
                swapped_source, swapped_target);
    std::printf("resolve: %08x (expected %08x), half float resolve: %08x (expected %08x)\n", resolved, green_bits, resolved_half, half_one);
    if (copied != red_bits) return 21;
    if (swapped_source != 0 || swapped_target != 0) return 22;
    if (resolved != green_bits) return 23;
    if (resolved_half != half_one) return 24;

    const std::uint32_t staged_copy = probe.texel(dest_readable.Get());
    const std::uint32_t refused_copy = probe.texel(dest_gone.Get());
    std::printf("title-backed source copied: %08x (expected %08x), refused source copy leaves destination: %08x (expected 00000000)\n",
                staged_copy, title_texel, refused_copy);
    if (staged_copy != title_texel) return 35;
    if (refused_copy != 0) return 36;

    // A copy into a title-backed texture that is recorded, then discarded by Close and Reset without ever being
    // submitted, did not happen: the texture still takes its first upload from its own memory.
    if (!ok(invoke(list, 139, arg(allocator), 0))) return 38;
    invoke(list, 17, arg(title_discarded), arg(empty_d.Get()));
    if (!step("10a CopyResource into a title-backed texture, then discarded", {0, 1, 0, 0}, seen)) return 37;
    invoke(list, 9);
    if (!ok(invoke(list, 139, arg(allocator), 0))) return 38;
    invoke(list, 17, arg(discard_out.Get()), arg(title_discarded));
    if (!step("10b CopyResource out of the texture after the discard", {0, 1, 0, 0}, seen)) return 39;
    invoke(list, 9);
    if (!probe.submit(list)) return 40;
    const std::uint32_t after_discard = probe.texel(discard_out.Get());
    std::printf("copy out after a discarded copy: %08x (expected %08x)\n", after_discard, title_texel);
    if (after_discard != title_texel) return 41;

    std::vector<std::uint8_t> whole, moved;
    if (!probe.bytes(buffer_x.Get(), 256, whole) || !probe.bytes(buffer_y.Get(), 80, moved)) return 25;
    if (std::memcmp(whole.data(), pattern.data(), pattern.size()) != 0) return 26;
    if (std::memcmp(moved.data() + 16, pattern.data() + 32, 64) != 0) return 27;
    std::printf("buffers: whole copy and offset copy exact\n");

    const HRESULT removed = probe.desktop->GetDeviceRemovedReason();
    const UINT64 errors = probe.debug_errors();
    std::printf("device removed reason: 0x%08lx, debug layer errors: %llu\n", static_cast<unsigned long>(removed),
                static_cast<unsigned long long>(errors));
    return removed == S_OK && errors == 0 ? 0 : 70;
}
