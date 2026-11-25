// ClearUnorderedAccessViewUint / ClearUnorderedAccessViewFloat / DiscardResource on a console command list
// (list slots 53, 54 and 55, device slot 19).
//
//   startup_clear_uav <warp|hardware>     (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1 X2W_D3D12_DEBUG=1)
//
// Through D3D12XboxCreateDevice the probe makes 1x1 targets with unordered access and records, on one list:
//
//   A  R32_UINT   ClearUnorderedAccessViewUint(0x12345678)            read back: 12345678
//   B  R32_UINT   cleared 0x11111111, then a clear with a GPU handle on no heap (refused, counted)
//                                                                     read back: 11111111
//   C  R32_FLOAT  ClearUnorderedAccessViewFloat(1.5)                  read back: 3fc00000
//   D  R32_UINT   a clear through a CPU handle that holds no view (refused, counted)
//   E  RGBA8 RTV  DiscardResource, with and without a region          list closes and executes
//
// The shader-visible heap holds the GPU-side descriptors, a second heap holds the CPU-side copies the
// desktop API requires. The desktop device must not be removed and the debug layer must store no errors.
//
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

unsigned unresolved_now()
{
    unsigned binds = 0, unresolved = 0;
    x2w_d3d12_render_target_stats(&binds, &unresolved);
    return unresolved;
}

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

    // Without the debug layer (no Graphics Tools) the error count stays 0; the clear and readback checks still run.
    if (!probe.watch()) std::printf("no debug layer message callback: validation not counted\n");

    // Five 1x1 targets: four with unordered access (A, B, C, D) and one render target (E).
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const auto texture = [](DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
        return D3D12_RESOURCE_DESC{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 1, 1, 1, format, {1, 0},
                                   D3D12_TEXTURE_LAYOUT_UNKNOWN, flags};
    };
    const D3D12_RESOURCE_DESC uint_desc = texture(DXGI_FORMAT_R32_UINT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const D3D12_RESOURCE_DESC float_desc = texture(DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const D3D12_RESOURCE_DESC rt_desc = texture(DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    const D3D12_CLEAR_VALUE rt_clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 0}};
    ComPtr<ID3D12Resource> a, b, c, d, e;
    const auto make = [&](const D3D12_RESOURCE_DESC& desc, ComPtr<ID3D12Resource>& out, const D3D12_CLEAR_VALUE* clear) {
        return SUCCEEDED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                                clear, IID_PPV_ARGS(&out)));
    };
    if (!make(uint_desc, a, nullptr) || !make(uint_desc, b, nullptr) || !make(float_desc, c, nullptr) ||
        !make(uint_desc, d, nullptr) || !make(rt_desc, e, &rt_clear))
        return 5;
    UINT64 bytes = 0;
    probe.desktop->GetCopyableFootprints(&uint_desc, 0, 1, 0, &probe.footprint, nullptr, nullptr, &bytes);
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                     D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
    if (FAILED(probe.desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                                      nullptr, IID_PPV_ARGS(&probe.readback)))) return 5;

    // Heaps: the shader-visible one is read by GPU handle, the plain one by CPU handle. Index 3 of the plain
    // heap is left empty. Both hold a UAV for A, B and C at the same indexes.
    const auto make_heap = [&](D3D12_DESCRIPTOR_HEAP_FLAGS flags, void** out) {
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            Word backing, bytes;
        } desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, flags}, 0, 0};
        return ok(invoke(device, 14, arg(&desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(out)));
    };
    void *visible_heap = nullptr, *plain_heap = nullptr;
    if (!make_heap(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, &visible_heap) || !make_heap(D3D12_DESCRIPTOR_HEAP_FLAG_NONE, &plain_heap))
        return 6;
    const UINT stride = probe.desktop->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const Word visible_cpu = invoke(visible_heap, 9), plain_cpu = invoke(plain_heap, 9), gpu_start = invoke(visible_heap, 10);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uint_view{DXGI_FORMAT_R32_UINT, D3D12_UAV_DIMENSION_TEXTURE2D, {}};
    D3D12_UNORDERED_ACCESS_VIEW_DESC float_view{DXGI_FORMAT_R32_FLOAT, D3D12_UAV_DIMENSION_TEXTURE2D, {}};
    ID3D12Resource* views[] = {a.Get(), b.Get(), c.Get()};
    D3D12_UNORDERED_ACCESS_VIEW_DESC* view_descs[] = {&uint_view, &uint_view, &float_view};
    for (unsigned i = 0; i < 3; ++i) {
        invoke(device, 19, arg(views[i]), 0, arg(view_descs[i]), visible_cpu + i * stride);
        invoke(device, 19, arg(views[i]), 0, arg(view_descs[i]), plain_cpu + i * stride);
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    void *allocator = nullptr, *list = nullptr;
    if (!ok(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&probe.queue))) ||
        !ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&probe.fence))) ||
        !ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))))
        return 7;
    if (!ok(invoke(list, 139, arg(allocator), 0))) return 8;
    void* heaps[] = {visible_heap};
    invoke(list, 32, 1, arg(heaps));

    const std::uint32_t uint_a[4] = {0x12345678, 0x12345678, 0x12345678, 0x12345678};
    const std::uint32_t uint_b1[4] = {0x11111111, 0x11111111, 0x11111111, 0x11111111};
    const std::uint32_t uint_b2[4] = {0x0badf00d, 0x0badf00d, 0x0badf00d, 0x0badf00d};
    const float float_c[4] = {1.5f, 1.5f, 1.5f, 1.5f};
    const D3D12_RECT whole{0, 0, 1, 1};

    unsigned seen = unresolved_now();
    invoke(list, 53, gpu_start, plain_cpu, arg(a.Get()), arg(uint_a), 0, 0);
    if (!check_count("1 uint clear of A", seen, 0)) return 10;
    invoke(list, 53, gpu_start + stride, plain_cpu + stride, arg(b.Get()), arg(uint_b1), 1, arg(&whole));
    if (!check_count("2 uint clear of B with a rect", seen, 0)) return 11;
    invoke(list, 54, gpu_start + 2 * stride, plain_cpu + 2 * stride, arg(c.Get()), arg(float_c), 0, 0);
    if (!check_count("3 float clear of C", seen, 0)) return 12;
    // The GPU handle is far past the four-descriptor heap: no known heap holds it.
    invoke(list, 53, gpu_start + 64 * stride, plain_cpu + stride, arg(b.Get()), arg(uint_b2), 0, 0);
    if (!check_count("4 clear of B through a GPU handle on no heap", seen, 1)) return 13;
    invoke(list, 53, gpu_start + 3 * stride, plain_cpu + 3 * stride, arg(d.Get()), arg(uint_b2), 0, 0);
    if (!check_count("5 clear through a CPU handle holding no view", seen, 1)) return 14;
    invoke(list, 53, gpu_start, plain_cpu, arg(a.Get()), 0, 0, 0);
    if (!check_count("6 clear with no values", seen, 1)) return 15;

    const D3D12_DISCARD_REGION region{1, &whole, 0, 1};
    invoke(list, 55, arg(e.Get()), 0);
    invoke(list, 55, arg(e.Get()), arg(&region));
    invoke(list, 55, 0, 0);
    if (!check_count("7 discards of the render target", seen, 0)) return 16;

    invoke(list, 9);
    if (!probe.submit(list)) return 17;
    const std::uint32_t bits_a = probe.texel(a.Get());
    const std::uint32_t bits_b = probe.texel(b.Get());
    const std::uint32_t bits_c = probe.texel(c.Get());
    std::printf("8 A after the uint clear: %08x (expected 12345678)\n", bits_a);
    std::printf("9 B after the refused clear: %08x (expected 11111111)\n", bits_b);
    std::printf("10 C after the float clear: %08x (expected 3fc00000)\n", bits_c);
    if (bits_a != 0x12345678) return 18;
    if (bits_b != 0x11111111) return 19;
    if (bits_c != 0x3fc00000) return 20;

    const HRESULT removed = probe.desktop->GetDeviceRemovedReason();
    const unsigned long long errors = debug_error_count.load();
    std::printf("device removed reason: 0x%08lx, debug layer errors: %llu\n", static_cast<unsigned long>(removed),
                static_cast<unsigned long long>(errors));
    return removed == S_OK && errors == 0 ? 0 : 70;
}
