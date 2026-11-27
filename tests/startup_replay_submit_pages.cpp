// Two fenced submissions to a compute queue. The second copy reads a late CPU write to the same placed buffer.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void x2w_d3d12_replay_record_stop(int);

namespace {

constexpr UINT64 page_bytes = 4096;
constexpr UINT64 allocation_bytes = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
constexpr UINT64 hole_allocation_bytes = UINT64{4} << 20;
constexpr UINT64 hole_prefix_bytes = 2 * allocation_bytes;
constexpr auto queue_type = D3D12_COMMAND_LIST_TYPE_COMPUTE;

template<class Result = HRESULT, class... Args>
Result invoke(void* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(void*, Args...);
    return reinterpret_cast<Call>((*static_cast<void***>(object))[slot])(object, args...);
}

bool check(HRESULT hr, const char* operation)
{
    if (SUCCEEDED(hr)) return true;
    std::fprintf(stderr, "%s failed (0x%08lx)\n", operation, static_cast<unsigned long>(hr));
    return false;
}

ComPtr<IDXGIAdapter> pick_adapter(bool hardware)
{
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return adapter;
    if (!hardware) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        return adapter;
    }
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> candidate;
        if (FAILED(factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate))))
            return adapter;
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(candidate->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
            candidate.As(&adapter);
            return adapter;
        }
    }
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes)
{
    return {D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
            {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
}

int scene(IDXGIAdapter* adapter, void* memory, const char* readback_path, void* hole_base)
{
    unsigned char parameters[64]{};
    void *device = nullptr, *placed = nullptr, *readback = nullptr;
    void *queue = nullptr, *allocator = nullptr, *list = nullptr, *fence = nullptr;
    const GUID& resource = __uuidof(ID3D12Resource);
    const auto placed_desc = buffer_desc(hole_base ? hole_allocation_bytes - hole_prefix_bytes : allocation_bytes);
    const auto readback_desc = buffer_desc(2 * page_bytes);
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    const D3D12_COMMAND_QUEUE_DESC queue_desc{queue_type};
    const unsigned list_desc[] = {queue_type, 2, 1, 0};
    if (!check(D3D12XboxCreateDevice(adapter, parameters, __uuidof(ID3D12Device), &device), "device") ||
        !check(invoke(device, 133, memory, &placed_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, &resource, &placed), "placed buffer") ||
        !check(invoke(device, 76, &readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                      nullptr, &resource, &readback), "readback buffer") ||
        !check(invoke(device, 8, &queue_desc, &__uuidof(ID3D12CommandQueue), &queue), "compute queue") ||
        !check(invoke(device, 9, queue_type, &__uuidof(ID3D12CommandAllocator), &allocator), "compute allocator") ||
        !check(invoke(device, 73, list_desc, allocator, nullptr, &__uuidof(ID3D12GraphicsCommandList), &list), "compute list") ||
        !check(invoke(device, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), &fence), "fence"))
        return 1;
    if (hole_base) {
        // Slot 133 records the texture's 128 KiB title footprint without using it as live GPU backing.
        // Its range joins the buffer's at +128 KiB, putting the hole and copy source in the same scan slab.
        const D3D12_RESOURCE_DESC texture_desc{
            D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 256, 128, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM,
            {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
        void* texture = nullptr;
        if (!check(invoke(device, 133, hole_base, &texture_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, &resource,
                          &texture), "placed texture spanning hole"))
            return 1;
    }
    // X2W_TEST_SUBMIT_PAGES_BIG=1: a second placed buffer of 4 MiB whose pages all differ, bigger than a small page budget.
    if (GetEnvironmentVariableA("X2W_TEST_SUBMIT_PAGES_BIG", nullptr, 0)) {
        constexpr UINT64 big_bytes = UINT64{4} << 20;
        auto* big = static_cast<std::uint32_t*>(VirtualAlloc(reinterpret_cast<void*>(0x300000100000ull), big_bytes,
                                                              MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!big) {
            std::fprintf(stderr, "Could not allocate the big buffer's title memory (error %lu)\n", GetLastError());
            return 1;
        }
        for (std::uint32_t i = 0; i < big_bytes / sizeof(*big); ++i) big[i] = i + 1;
        const auto big_desc = buffer_desc(big_bytes);
        void* big_placed = nullptr;
        if (!check(invoke(device, 133, static_cast<void*>(big), &big_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, &resource,
                          &big_placed), "big placed buffer"))
            return 1;
    }

    const unsigned char values[] = {0x2a, 0x59};
    for (UINT64 i = 0; i < 2; ++i) {
        if ((i && !check(invoke(allocator, 8), "reset allocator")) ||
            !check(invoke(list, 139, allocator), "reset list")) return 1;
        invoke<void>(list, 15, readback, i * page_bytes, placed, UINT64{0}, page_bytes);
        if (!check(invoke(list, 9), "close list")) return 1;
        std::memset(memory, values[i], page_bytes);
        void* lists[] = {list};
        invoke<void>(queue, 10, 1u, lists);
        if (!check(invoke(queue, 16, fence, i + 1), "signal")) return 1;
        const auto deadline = GetTickCount64() + 10000;
        while (invoke<UINT64>(fence, 8) < i + 1) {
            if (GetTickCount64() >= deadline) {
                std::fprintf(stderr, "fence timed out on submission %llu\n", i + 1);
                return 1;
            }
            Sleep(1);
        }
    }

    const D3D12_RANGE range{0, 2 * page_bytes}, no_writes{0, 0};
    const unsigned char* mapped = nullptr;
    if (!check(invoke(readback, 8, 0u, &range, &mapped), "map readback") || !mapped) return 1;
    for (UINT64 i = 0; i < range.End; ++i) {
        if (mapped[i] != values[i / page_bytes]) {
            std::fprintf(stderr, "live readback differs at byte %llu (0x%02x)\n", i, static_cast<unsigned>(mapped[i]));
            return 1;
        }
    }
    std::FILE* file = std::fopen(readback_path, "wb");
    if (!file) return 1;
    const bool written = std::fwrite(mapped, 1, range.End, file) == range.End;
    const int closed = std::fclose(file);
    invoke<void>(readback, 9, 0u, &no_writes);
    x2w_d3d12_replay_record_stop(0);
    return written && !closed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    const auto adapter = pick_adapter(std::strcmp(argv[1], "hardware") == 0);
    ComPtr<ID3D12Device> desktop;
    if (!adapter || FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 77;
    D3D12_FEATURE_DATA_EXISTING_HEAPS existing{};
    if (FAILED(desktop->CheckFeatureSupport(D3D12_FEATURE_EXISTING_HEAPS, &existing, sizeof(existing))) || !existing.Supported) {
        std::puts("Adapter lacks heaps backed by existing CPU memory");
        return 77;
    }
    // A fixed, allocation-aligned address keeps the independently launched replay reproducible.
    void* hole_base = nullptr;
    void* memory = nullptr;
    if (GetEnvironmentVariableA("X2W_TEST_SUBMIT_PAGES_HOLE", nullptr, 0)) {
        // Use adjacent allocations so the buffer can alias a fully committed allocation of its own.
        auto* base = reinterpret_cast<unsigned char*>(0x300000400000ull);
        hole_base = VirtualAlloc(base, hole_prefix_bytes, MEM_RESERVE, PAGE_READWRITE);
        if (hole_base && VirtualAlloc(hole_base, allocation_bytes, MEM_COMMIT, PAGE_READWRITE))
            memory = VirtualAlloc(base + hole_prefix_bytes, hole_allocation_bytes - hole_prefix_bytes,
                                  MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    } else {
        memory = VirtualAlloc(reinterpret_cast<void*>(0x300000000000ull), allocation_bytes,
                              MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    }
    if (!memory) {
        std::fprintf(stderr, "Could not allocate the fixture's title memory (error %lu)\n", GetLastError());
        return 1;
    }
    return scene(adapter.Get(), memory, argv[2], hole_base);
}
