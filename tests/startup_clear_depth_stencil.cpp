// ClearDepthStencilView on a console command list (list slot 51, device slot 21).
//
//   startup_clear_depth_stencil <warp|hardware>     (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1 X2W_D3D12_DEBUG=1)
//
// Through D3D12XboxCreateDevice the probe makes D32_FLOAT depth targets and records, on one list:
//
//   A  1x1 D32_FLOAT   ClearDepthStencilView(0.25f, DEPTH)                read back: 0.25f (0x3e800000)
//   B  2x2 D32_FLOAT   cleared 0.75f with rect covering surface,
//                      then a clear with a read-only DSV (refused, no overwrite)
//                                                                         read back: 0.75f (0x3f400000)
//   D  1x1 console format 118   slot 76 creation and matching depth/stencil clear
//   C  1x1 RGBA8 RTV   a clear through an RTV handle (wrong kind, refused, counted)
//   -  Empty handle    a clear through a CPU handle that holds no view (refused, counted)
//   -  Zero flags      a clear with flags == 0 (refused, no-op)
//
// The desktop device must not be removed and the debug layer must report no errors.
//
// Exit codes: 0 pass, 77 no physical adapter or debug callback, other values name the failing step.
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

using ClearDepthStencilFn = void (WINAPI*)(void* list, D3D12_CPU_DESCRIPTOR_HANDLE handle, D3D12_CLEAR_FLAGS flags,
                                           FLOAT depth, UINT8 stencil, UINT rect_count, const D3D12_RECT* rects);
void clear_dsv(void* list, D3D12_CPU_DESCRIPTOR_HANDLE handle, D3D12_CLEAR_FLAGS flags,
               FLOAT depth, UINT8 stencil, UINT rect_count, const D3D12_RECT* rects)
{
    auto fn = reinterpret_cast<ClearDepthStencilFn>((*static_cast<void***>(list))[51]);
    fn(list, handle, flags, depth, stencil, rect_count, rects);
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* res, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = res;
    barrier.Transition.StateBefore = from;
    barrier.Transition.StateAfter = to;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &barrier);
}

unsigned unresolved_now()
{
    unsigned binds = 0, unresolved = 0;
    x2w_d3d12_render_target_stats(&binds, &unresolved);
    return unresolved;
}

std::atomic<unsigned long long> debug_error_count{0};
std::atomic<unsigned> clear_mismatches{0};
void WINAPI on_message(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id, LPCSTR text, void*)
{
    if (id == D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE) {
        ++clear_mismatches;
        std::printf("debug layer: %.300s\n", text);
    }
    if (severity > D3D12_MESSAGE_SEVERITY_ERROR) return;
    if (++debug_error_count <= 12) std::printf("debug layer: %.300s\n", text);
}

struct Probe {
    ComPtr<ID3D12Device> desktop;
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

    DWORD cookie = 0;
    ComPtr<ID3D12InfoQueue1> queue1;
    bool watch()
    {
        return SUCCEEDED(desktop.As(&queue1)) &&
               SUCCEEDED(queue1->RegisterMessageCallback(on_message, D3D12_MESSAGE_CALLBACK_IGNORE_FILTERS, nullptr, &cookie));
    }

    bool readback_depth(ID3D12Resource* target, UINT width, UINT height, std::vector<float>& out)
    {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 total = 0;
        const auto desc = target->GetDesc();
        desktop->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);

        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, total, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                         D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        ComPtr<ID3D12Resource> readback;
        if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                                    nullptr, IID_PPV_ARGS(&readback))))
            return false;

        D3D12_COMMAND_QUEUE_DESC qdesc{};
        ComPtr<ID3D12CommandQueue> observer;
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> copy;
        ComPtr<ID3D12Fence> done;
        if (FAILED(desktop->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&observer))) ||
            FAILED(desktop->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
            FAILED(desktop->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&copy))) ||
            FAILED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done))))
            return false;

        transition(copy.Get(), target, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION to{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        to.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION from{target, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        copy->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        transition(copy.Get(), target, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);

        if (FAILED(copy->Close())) return false;
        ID3D12CommandList* lists[] = {copy.Get()};
        observer->ExecuteCommandLists(1, lists);

        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool finished = event && SUCCEEDED(observer->Signal(done.Get(), 1)) &&
                              SUCCEEDED(done->SetEventOnCompletion(1, event)) &&
                              WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
        if (event) CloseHandle(event);
        if (!finished) return false;

        void* memory = nullptr;
        if (FAILED(readback->Map(0, nullptr, &memory))) return false;

        out.resize(width * height);
        for (UINT y = 0; y < height; ++y) {
            const auto* row = reinterpret_cast<const float*>(static_cast<const unsigned char*>(memory) +
                                                             footprint.Offset + footprint.Footprint.RowPitch * y);
            std::memcpy(&out[y * width], row, width * sizeof(float));
        }
        readback->Unmap(0, nullptr);
        return true;
    }
};

bool check_count(const char* name, unsigned& before, unsigned expected_added)
{
    const unsigned now = unresolved_now();
    const bool good = now - before == expected_added;
    std::printf("%s: refused +%u (expected +%u)\n", name, now - before, expected_added);
    before = now;
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

    if (!probe.watch()) {
        std::puts("SKIP: no debug layer message callback");
        return 77;
    }

    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto depth_desc = [](UINT w, UINT h) {
        return D3D12_RESOURCE_DESC{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, w, h, 1, 1, DXGI_FORMAT_D32_FLOAT, {1, 0},
                                   D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
    };
    D3D12_CLEAR_VALUE clear_a{}, clear_b{};
    clear_a.Format = clear_b.Format = DXGI_FORMAT_D32_FLOAT;
    clear_a.DepthStencil = {0.25f, 0};
    clear_b.DepthStencil = {0.75f, 0};

    ComPtr<ID3D12Resource> a, b, c;
    const D3D12_RESOURCE_DESC desc_a = depth_desc(1, 1);
    const D3D12_RESOURCE_DESC desc_b = depth_desc(2, 2);
    const D3D12_RESOURCE_DESC desc_c{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 1, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0},
                                     D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const D3D12_CLEAR_VALUE rtv_clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 0}};

    if (FAILED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc_a, D3D12_RESOURCE_STATE_COMMON,
                                                      &clear_a, IID_PPV_ARGS(&a))) ||
        FAILED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc_b, D3D12_RESOURCE_STATE_COMMON,
                                                      &clear_b, IID_PPV_ARGS(&b))) ||
        FAILED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc_c, D3D12_RESOURCE_STATE_COMMON,
                                                      &rtv_clear, IID_PPV_ARGS(&c))))
        return 5;

    constexpr auto console_depth = static_cast<DXGI_FORMAT>(118);
    auto desc_d = depth_desc(1, 1);
    desc_d.Format = console_depth;
    D3D12_CLEAR_VALUE clear_d{};
    clear_d.Format = console_depth;
    clear_d.DepthStencil = {0.5f, 0x5a};
    ComPtr<ID3D12Resource> d;
    const auto errors_before_create = debug_error_count.load();
    const auto hr = invoke(device, 76, arg(&heap), D3D12_HEAP_FLAG_NONE, arg(&desc_d), D3D12_RESOURCE_STATE_DEPTH_WRITE,
                           arg(&clear_d), arg(&__uuidof(ID3D12Resource)), arg(d.GetAddressOf()));
    D3D12_RESOURCE_DESC created_d{};
    if (d) invoke(d.Get(), 10, arg(&created_d));  // Resource::GetDesc with the return buffer explicit.
    if (!ok(hr) || created_d.Format != DXGI_FORMAT_D24_UNORM_S8_UINT ||
        debug_error_count.load() != errors_before_create) {
        std::printf("FAIL: slot 76 console depth creation or debug validation, hr=0x%08lx\n",
                    static_cast<unsigned long>(static_cast<HRESULT>(hr)));
        return 21;
    }

    // DSV heap with 5 descriptors:
    // 0: DSV for A (writable)
    // 1: DSV for B (writable)
    // 2: DSV for B (read-only)
    // 3: Empty (unmapped)
    // 4: DSV for D (writable)
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    } dsv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 5, D3D12_DESCRIPTOR_HEAP_FLAG_NONE}, 0, 0},
      rtv_heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE}, 0, 0};

    void* dsv_heap = nullptr;
    void* rtv_heap = nullptr;
    if (!ok(invoke(device, 14, arg(&dsv_heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&dsv_heap))) ||
        !ok(invoke(device, 14, arg(&rtv_heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&rtv_heap))))
        return 6;

    const UINT dsv_stride = probe.desktop->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    const Word dsv_cpu = invoke(dsv_heap, 9);
    const Word rtv_cpu = invoke(rtv_heap, 9);

    const D3D12_DEPTH_STENCIL_VIEW_DESC dsv_writable{DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_FLAG_NONE, {}};
    const D3D12_DEPTH_STENCIL_VIEW_DESC dsv_readonly{DXGI_FORMAT_D32_FLOAT, D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_FLAG_READ_ONLY_DEPTH, {}};
    const D3D12_RENDER_TARGET_VIEW_DESC rtv_desc{DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RTV_DIMENSION_TEXTURE2D, {}};

    // Create DSVs:
    // Slot 21: CreateDepthStencilView(resource, desc, dest)
    invoke(device, 21, arg(a.Get()), arg(&dsv_writable), dsv_cpu);
    invoke(device, 21, arg(b.Get()), arg(&dsv_writable), dsv_cpu + dsv_stride);
    invoke(device, 21, arg(b.Get()), arg(&dsv_readonly), dsv_cpu + 2 * dsv_stride);

    auto dsv_mapped = dsv_writable;
    dsv_mapped.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    invoke(device, 21, arg(d.Get()), arg(&dsv_mapped), dsv_cpu + 4 * dsv_stride);

    // Create RTV:
    // Slot 20: CreateRenderTargetView(resource, desc, dest)
    invoke(device, 20, arg(c.Get()), arg(&rtv_desc), rtv_cpu);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void *allocator = nullptr, *list = nullptr;
    if (!ok(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&probe.queue))) ||
        !ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&probe.fence))) ||
        !ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))))
        return 7;
    if (!ok(invoke(list, 139, arg(allocator), 0))) return 8;

    unsigned seen = unresolved_now();

    // 1. Valid clear of A to depth 0.25f
    const D3D12_CPU_DESCRIPTOR_HANDLE handle_a{static_cast<SIZE_T>(dsv_cpu)};
    clear_dsv(list, handle_a, D3D12_CLEAR_FLAG_DEPTH, 0.25f, 0, 0, nullptr);
    if (!check_count("1 valid clear of A to 0.25", seen, 0)) return 10;

    // 2. Valid clear of B to depth 0.75f with rect covering surface
    const D3D12_CPU_DESCRIPTOR_HANDLE handle_b{static_cast<SIZE_T>(dsv_cpu + dsv_stride)};
    const D3D12_RECT whole_b{0, 0, 2, 2};
    clear_dsv(list, handle_b, D3D12_CLEAR_FLAG_DEPTH, 0.75f, 0, 1, &whole_b);
    if (!check_count("2 valid clear of B with rect to 0.75", seen, 0)) return 11;

    // 3. Clear with read-only DSV (attempting to clear to 0.125f) -> refused without overwriting B
    const D3D12_CPU_DESCRIPTOR_HANDLE handle_b_ro{static_cast<SIZE_T>(dsv_cpu + 2 * dsv_stride)};
    clear_dsv(list, handle_b_ro, D3D12_CLEAR_FLAG_DEPTH, 0.125f, 0, 0, nullptr);
    if (!check_count("3 read-only DSV clear refusal", seen, 0)) return 12;

    // 4. Negative test: RTV handle passed -> wrong kind, refused and counted
    const D3D12_CPU_DESCRIPTOR_HANDLE handle_c_rtv{static_cast<SIZE_T>(rtv_cpu)};
    clear_dsv(list, handle_c_rtv, D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0, 0, nullptr);
    if (!check_count("4 RTV handle passed as DSV (wrong kind)", seen, 1)) return 13;

    // 5. Negative test: CPU handle holding no view -> refused and counted
    const D3D12_CPU_DESCRIPTOR_HANDLE handle_empty{static_cast<SIZE_T>(dsv_cpu + 3 * dsv_stride)};
    clear_dsv(list, handle_empty, D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0, 0, nullptr);
    if (!check_count("5 empty descriptor handle", seen, 1)) return 14;

    // 6. Negative test: clear flags == 0 -> refused
    clear_dsv(list, handle_a, static_cast<D3D12_CLEAR_FLAGS>(0), 0.5f, 0, 0, nullptr);
    if (!check_count("6 zero clear flags", seen, 0)) return 15;

    // Mutant: forwarding raw format 118 fails on debug validation or the slot 76
    // retry (CTest also rejects the "(dropped)" log marker).
    const D3D12_CPU_DESCRIPTOR_HANDLE handle_d{static_cast<SIZE_T>(dsv_cpu + 4 * dsv_stride)};
    clear_dsv(list, handle_d, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.5f, 0x5a, 0, nullptr);
    if (!check_count("7 mapped depth/stencil clear", seen, 0)) return 22;

    // Close and submit
    invoke(list, 9);
    if (!probe.submit(list)) return 16;

    if (clear_mismatches.load() != 0) {
        std::puts("FAIL: depth/stencil clear did not match the optimized clear value");
        return 23;
    }

    // Read back depth values
    std::vector<float> values_a, values_b;
    if (!probe.readback_depth(a.Get(), 1, 1, values_a)) return 17;
    if (!probe.readback_depth(b.Get(), 2, 2, values_b)) return 18;

    std::printf("8 A depth: %.4f (expected 0.2500)\n", values_a[0]);
    if (values_a[0] != 0.25f) return 19;

    for (int i = 0; i < 4; ++i) {
        std::printf("9 B[%d] depth: %.4f (expected 0.7500)\n", i, values_b[i]);
        if (values_b[i] != 0.75f) return 20;
    }

    const HRESULT removed = probe.desktop->GetDeviceRemovedReason();
    const unsigned long long errors = debug_error_count.load();
    std::printf("device removed reason: 0x%08lx, debug layer errors: %llu\n", static_cast<unsigned long>(removed),
                static_cast<unsigned long long>(errors));
    return removed == S_OK && errors == 0 ? 0 : 70;
}
