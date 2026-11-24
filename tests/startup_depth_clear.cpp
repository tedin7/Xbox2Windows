// ClearDepthStencilView (command-list slot 51) on a depth view, checked by reading the depth and stencil planes back.
//
//   startup_depth_clear <warp|hardware>     (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1 X2W_D3D12_DEBUG=1)
//
// Through D3D12XboxCreateDevice the probe makes one 4x4 D32_FLOAT_S8X24_UINT texture and a DSV heap with four views:
// 0 writable, 1 depth plane read-only, 2 never created, 3 stencil plane read-only. Calls are recorded on one list, each
// followed by a submit and a readback of texel (0,0) and texel (3,3) of both planes:
//
//   0  nothing cleared yet                                      control: the readback is not the value of step 1
//   1  flags 3, depth 0.75, stencil 0x11                        both planes written
//   2  flags 1, depth 0.25, stencil argument 0x77               depth written; the stencil argument is ignored
//   3  flags 2, depth argument 7.0, stencil 0x5A                stencil written; the depth argument is ignored
//   4  flags 1, depth 0.5 with the rectangle (0,0)-(2,2)        only the rectangle is cleared
//   5  refused: depth 1.5, NaN, depth plane read-only, stencil plane read-only, unreadable rectangles; the empty
//      view; flags 0 does nothing                               nothing changes, refusals are counted
//   6  a call on the closed list                                ignored
//
// Counters come from x2w_d3d12_depth_clear_stats (forwarded, refused by rule) and x2w_d3d12_render_target_stats
// (handles that held no view). The desktop device must not be removed and the debug layer must store no errors.
//
// Exit codes: 0 pass, 77 no physical adapter, other values name the failing step.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void x2w_d3d12_render_target_stats(unsigned* binds, unsigned* unresolved);
extern "C" __declspec(dllimport) void x2w_d3d12_depth_clear_stats(unsigned* forwarded, unsigned* refused);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0, Word f = 0, Word g = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f, g);
}
bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }

// ClearDepthStencilView(view, flags, depth, stencil, rect count, rects): the depth is a float, so the call must be
// typed (the integer-only invoke above would put it in the wrong register).
void clear_depth_stencil(void* list, D3D12_CPU_DESCRIPTOR_HANDLE view, UINT flags, float depth, UINT stencil, UINT count = 0,
                         const D3D12_RECT* rects = nullptr)
{
    using Slot = void (WINAPI*)(void*, Word, UINT, float, UINT, UINT, const D3D12_RECT*);
    reinterpret_cast<Slot>((*static_cast<void***>(list))[51])(list, view.ptr, flags, depth, stencil, count, rects);
}

struct Counts {
    unsigned forwarded = 0, refused = 0, unresolved = 0;
    static Counts read()
    {
        Counts counts;
        unsigned binds = 0;
        x2w_d3d12_depth_clear_stats(&counts.forwarded, &counts.refused);
        x2w_d3d12_render_target_stats(&binds, &counts.unresolved);
        return counts;
    }
};
bool counted(const char* name, unsigned forwarded, unsigned refused, unsigned unresolved, Counts& before)
{
    const Counts now = Counts::read();
    const bool good = now.forwarded - before.forwarded == forwarded && now.refused - before.refused == refused &&
                      now.unresolved - before.unresolved == unresolved;
    std::printf("%s: forwarded +%u (expected +%u), refused +%u (expected +%u), unresolved +%u (expected +%u)\n", name,
                now.forwarded - before.forwarded, forwarded, now.refused - before.refused, refused,
                now.unresolved - before.unresolved, unresolved);
    before = now;
    return good;
}

struct Texels {
    float depth00 = 0, depth33 = 0;
    std::uint8_t stencil00 = 0, stencil33 = 0;
};

struct Probe {
    ComPtr<ID3D12Device> desktop;
    ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint[2]{};
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
    // Both planes of the texture, copied on a desktop queue. The resource rests in COMMON between title lists; the
    // barrier is explicit because a depth texture is not implicitly promoted.
    bool read(ID3D12Resource* texture, Texels& out)
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
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = texture;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        copy->ResourceBarrier(1, &barrier);
        for (UINT plane = 0; plane < 2; ++plane) {
            D3D12_TEXTURE_COPY_LOCATION to{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            to.PlacedFootprint = footprint[plane];
            D3D12_TEXTURE_COPY_LOCATION from{texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            from.SubresourceIndex = plane;
            copy->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        }
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        copy->ResourceBarrier(1, &barrier);
        if (FAILED(copy->Close())) return false;
        ID3D12CommandList* lists[] = {copy.Get()};
        observer->ExecuteCommandLists(1, lists);
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool finished = event && SUCCEEDED(observer->Signal(done.Get(), 1)) &&
                              SUCCEEDED(done->SetEventOnCompletion(1, event)) && WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
        if (event) CloseHandle(event);
        unsigned char* memory = nullptr;
        if (!finished || FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&memory)))) return false;
        const auto depth_at = [&](UINT x, UINT y) {
            float value;
            std::memcpy(&value, memory + footprint[0].Offset + y * footprint[0].Footprint.RowPitch + x * 4, sizeof(value));
            return value;
        };
        out.depth00 = depth_at(0, 0);
        out.depth33 = depth_at(3, 3);
        out.stencil00 = memory[footprint[1].Offset];
        out.stencil33 = memory[footprint[1].Offset + 3 * footprint[1].Footprint.RowPitch + 3];
        readback->Unmap(0, nullptr);
        return true;
    }
};

bool same_float(float a, float b) { return std::fabs(a - b) < 1e-6f; }
bool expect(const char* name, const Texels& got, float depth00, float depth33, int stencil00, int stencil33)
{
    const bool good = same_float(got.depth00, depth00) && same_float(got.depth33, depth33) && got.stencil00 == stencil00 && got.stencil33 == stencil33;
    std::printf("%s: depth (0,0) %.3f (3,3) %.3f stencil %02x %02x (expected %.3f %.3f %02x %02x)\n", name, got.depth00,
                got.depth33, got.stencil00, got.stencil33, depth00, depth33, stencil00, stencil33);
    return good;
}
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

    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC texture_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 4, 4, 1, 1, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, {1, 0},
                                           D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
    D3D12_CLEAR_VALUE clear{};
    clear.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    clear.DepthStencil = {1.0f, 0};
    ComPtr<ID3D12Resource> texture;
    if (FAILED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &texture_desc, D3D12_RESOURCE_STATE_COMMON, &clear,
                                                      IID_PPV_ARGS(&texture))))
        return 5;
    UINT64 bytes = 0;
    probe.desktop->GetCopyableFootprints(&texture_desc, 0, 2, 0, probe.footprint, nullptr, nullptr, &bytes);
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                     D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
    if (FAILED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                      IID_PPV_ARGS(&probe.readback))))
        return 5;

    // The DSV heap: view 0 writable, 1 depth plane read-only, 2 never created, 3 stencil plane read-only.
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 4}, 0, 0};
    void* dsv_heap = nullptr;
    if (!ok(invoke(device, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&dsv_heap)))) return 6;
    const UINT stride = probe.desktop->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    const D3D12_CPU_DESCRIPTOR_HANDLE first{static_cast<SIZE_T>(invoke(dsv_heap, 9))};
    const std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 4> dsv{{first, {first.ptr + stride}, {first.ptr + 2 * stride}, {first.ptr + 3 * stride}}};
    D3D12_DEPTH_STENCIL_VIEW_DESC writable{DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_FLAG_NONE, {}};
    D3D12_DEPTH_STENCIL_VIEW_DESC depth_read_only = writable, stencil_read_only = writable;
    depth_read_only.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH;
    stencil_read_only.Flags = D3D12_DSV_FLAG_READ_ONLY_STENCIL;
    invoke(device, 21, arg(texture.Get()), arg(&writable), dsv[0].ptr);
    invoke(device, 21, arg(texture.Get()), arg(&depth_read_only), dsv[1].ptr);
    invoke(device, 21, arg(texture.Get()), arg(&stencil_read_only), dsv[3].ptr);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void *allocator = nullptr, *list = nullptr;
    if (!ok(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&probe.queue))) ||
        !ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&probe.fence))) ||
        !ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))))
        return 7;

    // Records one case: reset, the calls, close, submit, read both planes back.
    Texels seen;
    const auto run = [&](auto&& calls) {
        if (!ok(invoke(list, 139, arg(allocator), 0))) return false;
        calls();
        invoke(list, 9);
        return probe.submit(list) && probe.read(texture.Get(), seen);
    };
    Counts counts = Counts::read();
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();

    if (!run([&] {})) return 8;
    if (!counted("0 nothing recorded", 0, 0, 0, counts)) return 9;
    const Texels before = seen;
    std::printf("0 before any clear: depth (0,0) %.3f, stencil %02x\n", before.depth00, before.stencil00);
    if (same_float(before.depth00, 0.75f) || before.stencil00 == 0x11) return 10;  // control: step 1 must change what is read

    if (!run([&] { clear_depth_stencil(list, dsv[0], 3, 0.75f, 0x11); })) return 11;
    if (!counted("1 both planes", 1, 0, 0, counts) || !expect("1", seen, 0.75f, 0.75f, 0x11, 0x11)) return 12;

    if (!run([&] { clear_depth_stencil(list, dsv[0], 1, 0.25f, 0x77); })) return 13;
    if (!counted("2 depth only", 1, 0, 0, counts) || !expect("2", seen, 0.25f, 0.25f, 0x11, 0x11)) return 14;

    if (!run([&] { clear_depth_stencil(list, dsv[0], 2, 7.0f, 0x5A); })) return 15;
    if (!counted("3 stencil only", 1, 0, 0, counts) || !expect("3", seen, 0.25f, 0.25f, 0x5A, 0x5A)) return 16;

    const D3D12_RECT corner{0, 0, 2, 2};
    if (!run([&] { clear_depth_stencil(list, dsv[0], 1, 0.5f, 0, 1, &corner); })) return 17;
    if (!counted("4 rectangle", 1, 0, 0, counts) || !expect("4", seen, 0.5f, 0.25f, 0x5A, 0x5A)) return 18;

    const auto* unreadable = reinterpret_cast<const D3D12_RECT*>(static_cast<std::uintptr_t>(0x10));
    if (!run([&] {
            clear_depth_stencil(list, dsv[0], 1, 1.5f, 0);          // depth above 1
            clear_depth_stencil(list, dsv[0], 1, nan, 0);           // not a number
            clear_depth_stencil(list, dsv[1], 1, 0.9f, 0);          // depth plane read-only
            clear_depth_stencil(list, dsv[3], 3, 0.9f, 0x99);       // stencil plane read-only
            clear_depth_stencil(list, dsv[0], 1, 0.9f, 0, 2, unreadable);
            clear_depth_stencil(list, dsv[2], 1, 0.9f, 0);          // a view that holds nothing
            clear_depth_stencil(list, dsv[0], 0, 0.9f, 0x99);       // no plane: nothing to do
        }))
        return 19;
    if (!counted("5 refusals", 0, 5, 1, counts) || !expect("5", seen, 0.5f, 0.25f, 0x5A, 0x5A)) return 20;

    // The list is closed now: a call must be ignored.
    clear_depth_stencil(list, dsv[0], 3, 0.0f, 0xff);
    if (!counted("6 closed list", 0, 0, 0, counts)) return 21;

    const HRESULT removed = probe.desktop->GetDeviceRemovedReason();
    const UINT64 errors = probe.debug_errors();
    std::printf("device removed reason: 0x%08lx, debug layer errors: %llu\n", static_cast<unsigned long>(removed),
                static_cast<unsigned long long>(errors));
    return removed == S_OK && errors == 0 ? 0 : 70;
}
