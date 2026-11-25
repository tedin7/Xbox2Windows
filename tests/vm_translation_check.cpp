// Source-built consumer for the main executable's A/B range translation (#874).
// The title keeps a CPU range A and a range B of the same size and translates by offset:
// gpu = cpu - A + B, and a GPU address inside [B, B + size) back to cpu = gpu - B + A
// (anything else passes through unchanged). These checks ask whether a desktop D3D12
// device can give A a GPU view that keeps that offset.
//   shared_range        no device: A and B are one reservation, so both directions are the identity
//   device_<adapter>    A is one committed allocation wrapped as a D3D12 heap with one buffer over it
//   sequence_<adapter>  the title's order through d3d12_xs.dll: configure, create the device, create
//                       the first DIRECT queue, then the device_<adapter> checks on that queue
//   range_<adapter>     a 32 GiB range on its owning handle (src/d3d12_xs_range.h): mapped and copied
//                       tiles read by dispatch at the range's GPU address, then Close() after idle
//   range_lifetime_<adapter>
//                       an open range's windows hold the device; releasing or closing the range lets
//                       the device's last release destroy it; Close() overlapping reads on a thread
//   range_partial_<adapter>
//                       window creation failing at window 1 or 3
// Returns 0 on pass, 1 on failure, 77 (skip) without the adapter or ID3D12Device3, or when a dispatch
// cannot read a reserved buffer's copied tile.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "d3d12_xs_range.h"
#include "vm_translation_cs.h"
#include "xmem_test_device.h"

extern "C" HRESULT WINAPI D3D12ConfigureVirtualMemoryX(const void* config);
extern "C" HRESULT WINAPI D3D12XboxCreateDevice(IUnknown* adapter, const void* parameters, REFIID iid,
                                                void** device);

using Microsoft::WRL::ComPtr;
using namespace std::literals;

namespace {
constexpr std::uint64_t range_size = 64ull << 20;
constexpr UINT words = 1024;
constexpr std::uint32_t key = 0x5a5a5a5au;

// Mirrors the title's two helpers.
struct RangePair {
    std::uintptr_t a;
    std::uint64_t b;
    std::uint64_t size;
    std::uint64_t to_gpu(const void* p) const { return reinterpret_cast<std::uintptr_t>(p) - a + b; }
    std::uint64_t to_cpu(std::uint64_t va) const { return va >= b && va - b < size ? va - b + a : va; }
};

// Owns a VirtualAlloc range. Objects that use the range are declared after it, so they are released first.
struct VirtualRange {
    std::uint8_t* base = nullptr;
    explicit VirtualRange(void* p) : base(static_cast<std::uint8_t*>(p)) {}
    VirtualRange(const VirtualRange&) = delete;
    VirtualRange& operator=(const VirtualRange&) = delete;
    ~VirtualRange()
    {
        if (base) VirtualFree(base, 0, MEM_RELEASE);
    }
};

unsigned failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
    failures += !ok;
}

bool shared_range_checks()
{
    // The title's no-device path releases A and B and reserves one range used as both.
    void* a = VirtualAlloc(nullptr, range_size, MEM_RESERVE, PAGE_READWRITE);
    void* b = VirtualAlloc(nullptr, range_size, MEM_RESERVE, PAGE_READWRITE);
    expect(a && b && a != b, "two separate reservations");
    VirtualFree(a, 0, MEM_RELEASE);
    VirtualFree(b, 0, MEM_RELEASE);
    auto* shared = static_cast<std::uint8_t*>(
        VirtualAlloc(nullptr, range_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!shared) return false;
    const RangePair pair{reinterpret_cast<std::uintptr_t>(shared), reinterpret_cast<std::uintptr_t>(shared), range_size};

    bool identity = true;
    for (std::uint64_t offset : {0ull, 0x10000ull, range_size / 2, range_size - 4}) {
        const void* p = shared + offset;
        identity &= pair.to_gpu(p) == reinterpret_cast<std::uintptr_t>(p);
        identity &= pair.to_cpu(pair.to_gpu(p)) == reinterpret_cast<std::uintptr_t>(p);
    }
    expect(identity, "A == B: both directions are the identity inside the range");
    const std::uint64_t outside = pair.b + range_size;
    expect(pair.to_cpu(outside) == outside && pair.to_cpu(pair.b - 1) == pair.b - 1,
           "addresses just outside the range pass through");
    VirtualFree(shared, 0, MEM_RELEASE);
    std::printf("shared_range failures=%u\n", failures);
    return failures == 0;
}

struct Gpu {
    ComPtr<ID3D12Device3> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    UINT64 value = 0;

    bool init()
    {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        D3D12_ROOT_PARAMETER params[2]{};
        for (UINT i = 0; i < 2; ++i) {
            params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            params[i].Descriptor.ShaderRegister = i;
        }
        const D3D12_ROOT_SIGNATURE_DESC root_desc{2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob;
        if ((!queue && FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)))) ||
            FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                             IID_PPV_ARGS(&list))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) ||
            FAILED(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr)) ||
            FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                               IID_PPV_ARGS(&root)))) {
            return false;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get();
        pso.CS = {vm_translation_cs, sizeof(vm_translation_cs)};
        return SUCCEEDED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline)));
    }

    // Waits for all submitted work, whether or not the last dispatch completed its own wait.
    void wait_idle()
    {
        if (!queue || !fence) return;
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (event && SUCCEEDED(queue->Signal(fence.Get(), ++value)) &&
            SUCCEEDED(fence->SetEventOnCompletion(value, event))) {
            WaitForSingleObject(event, INFINITE);
        }
        if (event) CloseHandle(event);
    }

    // Moves each non-null resource between COMMON and `state`.
    void transition(ID3D12Resource* const* resources, UINT count, D3D12_RESOURCE_STATES state, bool back)
    {
        D3D12_RESOURCE_BARRIER barriers[2]{};
        UINT used = 0;
        for (UINT i = 0; i < count; ++i) {
            if (!resources[i]) continue;
            auto& barrier = barriers[used++];
            barrier.Transition.pResource = resources[i];
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = back ? state : D3D12_RESOURCE_STATE_COMMON;
            barrier.Transition.StateAfter = back ? D3D12_RESOURCE_STATE_COMMON : state;
        }
        if (used) list->ResourceBarrier(used, barriers);
    }

    // One dispatch with both root UAVs given as raw GPU addresses. `resource` (and `second`) are the
    // buffers those addresses fall in.
    bool dispatch(ID3D12Resource* resource, std::uint64_t src, std::uint64_t dst, ID3D12Resource* second = nullptr)
    {
        ID3D12Resource* const used[] = {resource, second};
        transition(used, 2, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false);
        list->SetComputeRootSignature(root.Get());
        list->SetPipelineState(pipeline.Get());
        list->SetComputeRootUnorderedAccessView(0, src);
        list->SetComputeRootUnorderedAccessView(1, dst);
        list->Dispatch(words / 64, 1, 1);
        transition(used, 2, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
        return submit();
    }

    // Copies `bytes` from src to dst at the given offsets.
    bool copy(ID3D12Resource* dst, std::uint64_t dst_offset, ID3D12Resource* src, std::uint64_t src_offset,
              std::uint64_t bytes)
    {
        transition(&dst, 1, D3D12_RESOURCE_STATE_COPY_DEST, false);
        transition(&src, 1, D3D12_RESOURCE_STATE_COPY_SOURCE, false);
        list->CopyBufferRegion(dst, dst_offset, src, src_offset, bytes);
        transition(&dst, 1, D3D12_RESOURCE_STATE_COPY_DEST, true);
        transition(&src, 1, D3D12_RESOURCE_STATE_COPY_SOURCE, true);
        return submit();
    }

    // Closes, executes and waits for the recorded list, then reopens it.
    bool submit()
    {
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool done = event && SUCCEEDED(queue->Signal(fence.Get(), ++value)) &&
                          SUCCEEDED(fence->SetEventOnCompletion(value, event)) &&
                          WaitForSingleObject(event, 20000) == WAIT_OBJECT_0;
        if (event) CloseHandle(event);
        return done && SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.Get(), nullptr));
    }
};

D3D12_RESOURCE_DESC buffer_desc(std::uint64_t size, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

void fill(std::uint32_t* words_out, std::uint32_t seed)
{
    for (UINT i = 0; i < words; ++i) words_out[i] = seed + i * 2654435761u;
}

unsigned mismatches(const std::uint32_t* src, const std::uint32_t* dst)
{
    unsigned bad = 0;
    for (UINT i = 0; i < words; ++i) bad += dst[i] != (src[i] ^ key);
    return bad;
}

// The #912 checks on an initialized device and queue.
int translation_checks(Gpu& gpu)
{
    // Refusals: a reserve-only range, and an address inside an allocation.
    {
        const VirtualRange reserved(VirtualAlloc(nullptr, range_size, MEM_RESERVE, PAGE_READWRITE));
        if (!reserved.base) {
            std::puts("setup error: reserve-only VirtualAlloc failed");
            return 1;
        }
        ComPtr<ID3D12Heap> refused;
        const HRESULT hr = gpu.device->OpenExistingHeapFromAddress(reserved.base, IID_PPV_ARGS(&refused));
        std::printf("reserve-only range hr=0x%08lx\n", static_cast<unsigned long>(hr));
        expect(FAILED(hr), "a reserve-only range cannot be opened as a heap");
    }
    const VirtualRange range(VirtualAlloc(nullptr, range_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!range.base) {
        std::puts("setup error: committed VirtualAlloc failed");
        return 1;
    }
    std::uint8_t* const a = range.base;
    {
        ComPtr<ID3D12Heap> refused;
        const HRESULT hr = gpu.device->OpenExistingHeapFromAddress(a + range_size / 2, IID_PPV_ARGS(&refused));
        std::printf("interior address hr=0x%08lx\n", static_cast<unsigned long>(hr));
        expect(FAILED(hr), "an address inside the allocation cannot be opened as a heap");
    }

    // Everything below that uses A is declared after `range`. On every return, `idle` waits for
    // submitted work first, then these are released, and only then does `range` free A.
    ComPtr<ID3D12Heap> heap;
    ComPtr<ID3D12Resource> whole, part, other;
    struct Idle {
        Gpu& gpu;
        ~Idle() { gpu.wait_idle(); }
    } idle{gpu};

    // Device path: wrap all of A once and place one buffer over it. Its GPU address is B.
    HRESULT hr = gpu.device->OpenExistingHeapFromAddress(a, IID_PPV_ARGS(&heap));
    std::printf("committed range hr=0x%08lx\n", static_cast<unsigned long>(hr));
    if (hr == E_NOTIMPL || hr == DXGI_ERROR_UNSUPPORTED) {
        std::puts("OpenExistingHeapFromAddress unsupported on this adapter");
        return 77;
    }
    const auto whole_desc = buffer_desc(range_size, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER |
                                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (SUCCEEDED(hr)) {
        hr = gpu.device->CreatePlacedResource(heap.Get(), 0, &whole_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                              IID_PPV_ARGS(&whole));
    }
    expect(SUCCEEDED(hr), "wrap the committed range and place one buffer over it");
    if (FAILED(hr)) return 1;
    const RangePair pair{reinterpret_cast<std::uintptr_t>(a), whole->GetGPUVirtualAddress(), range_size};
    std::printf("B != A: %d\n", pair.b != pair.a);

    // CPU -> GPU: the shader sees only addresses computed as p - A + B.
    auto* src = reinterpret_cast<std::uint32_t*>(a + range_size / 2);
    auto* dst = reinterpret_cast<std::uint32_t*>(a + range_size - 0x8000);
    fill(src, 0x10000000u);
    std::memset(dst, 0xcd, words * 4);
    expect(gpu.dispatch(whole.Get(), pair.to_gpu(src), pair.to_gpu(dst)) && mismatches(src, dst) == 0,
           "GPU reads CPU-written words and writes CPU-visible words through p - A + B");

    // GPU -> CPU: a live resource's own GPU address, translated back with va - B + A.
    {
        constexpr std::uint64_t offset = 0x200000;
        const auto part_desc = buffer_desc(0x10000, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER |
                                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        hr = gpu.device->CreatePlacedResource(heap.Get(), offset, &part_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                              IID_PPV_ARGS(&part));
        expect(SUCCEEDED(hr), "place a second live buffer inside the wrapped range");
        if (SUCCEEDED(hr)) {
            const std::uint64_t va = part->GetGPUVirtualAddress();
            std::printf("placed buffer at +0x%llx: gpu - B = 0x%llx\n", static_cast<unsigned long long>(offset),
                        static_cast<unsigned long long>(va - pair.b));
            // A wrong translation is not dereferenced: outside A it would be an arbitrary pointer.
            const std::uint64_t translated = pair.to_cpu(va);
            const bool same_offset = translated == pair.a + offset;
            expect(same_offset, "its GPU address translates to the CPU address at the same offset");
            if (same_offset) {
                auto* cpu = reinterpret_cast<std::uint32_t*>(translated);
                fill(src, 0x30000000u);
                std::memset(cpu, 0xcd, words * 4);
                expect(gpu.dispatch(whole.Get(), pair.to_gpu(src), va) && mismatches(src, cpu) == 0,
                       "GPU writes through the live resource's address are visible at the translated CPU address");
            }
        }
    }

    // Mismatched ranges: an address from another resource passes through, and a wrong B misplaces data.
    {
        D3D12_HEAP_PROPERTIES props{};
        props.Type = D3D12_HEAP_TYPE_DEFAULT;
        const auto other_desc = buffer_desc(0x10000, D3D12_RESOURCE_FLAG_NONE);
        hr = gpu.device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &other_desc,
                                                 D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&other));
        if (SUCCEEDED(hr)) {
            const std::uint64_t va = other->GetGPUVirtualAddress();
            const bool outside = va < pair.b || va - pair.b >= range_size;
            expect(outside && pair.to_cpu(va) == va,
                   "an unrelated buffer's GPU address lies outside [B, B + size) and passes through untranslated");
        } else {
            expect(false, "create an unrelated committed buffer");
        }

        // Both shifted addresses stay inside the wrapped buffer.
        auto* mid_src = reinterpret_cast<std::uint32_t*>(a + range_size / 4);
        auto* mid_dst = reinterpret_cast<std::uint32_t*>(a + range_size / 2);
        const RangePair shifted{pair.a, pair.b + 0x10000, range_size};
        fill(mid_src, 0x50000000u);
        std::memset(mid_dst, 0xcd, words * 4);
        const bool ran = gpu.dispatch(whole.Get(), shifted.to_gpu(mid_src), shifted.to_gpu(mid_dst));
        const unsigned bad = mismatches(mid_src, mid_dst);
        const bool untouched = mid_dst[0] == 0xcdcdcdcdu && mid_dst[words - 1] == 0xcdcdcdcdu;
        std::printf("B shifted by 64 KiB: %u/%u words wrong at the intended destination\n", bad, words);
        expect(ran && bad == words && untouched, "a B off by one 64 KiB page misses the intended words");
    }

    std::printf("device failures=%u\n", failures);
    return failures ? 1 : 0;
}

int device_checks(bool warp)
{
    bool unmatched = false;
    const auto base = create_device(warp, unmatched);
    if (unmatched) return 1;
    Gpu gpu;
    if (!base || FAILED(base.As(&gpu.device))) {
        std::puts(warp ? "No ID3D12Device3 on WARP" : "No hardware adapter with ID3D12Device3");
        return 77;
    }
    if (!gpu.init()) {
        std::puts("compute setup failed");
        return 1;
    }
    return translation_checks(gpu);
}

HRESULT create_xbox_device(IUnknown* adapter, REFIID iid, void** device)
{
    return D3D12XboxCreateDevice(adapter, nullptr, iid, device);
}

// The title's 32-byte configuration block between guard bytes, so a write next to it is caught too.
struct ConfigBlock {
    static constexpr std::size_t guard = 16;
    std::array<unsigned char, guard + 32 + guard> bytes{};
    unsigned char* block() { return bytes.data() + guard; }
    void reset(std::uint32_t value)
    {
        bytes.fill(0xa5);
        std::memset(block(), 0, 32);
        std::memcpy(block() + 4, &value, sizeof(value));
    }
};

void expect_configure(ConfigBlock& config, HRESULT expected, const char* what)
{
    const auto before = config.bytes;
    const HRESULT hr = D3D12ConfigureVirtualMemoryX(config.block());
    std::printf("configure hr=0x%08lx: ", static_cast<unsigned long>(hr));
    expect(hr == expected && before == config.bytes, what);
}

// A refused queue request returns no queue.
void expect_queue_refused(ID3D12Device* device, const D3D12_COMMAND_QUEUE_DESC& desc, const char* what)
{
    ID3D12CommandQueue* queue = reinterpret_cast<ID3D12CommandQueue*>(std::uintptr_t{1});
    const HRESULT hr = device->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue));
    std::printf("queue hr=0x%08lx: ", static_cast<unsigned long>(hr));
    const bool refused = FAILED(hr) && (!queue || queue == reinterpret_cast<ID3D12CommandQueue*>(std::uintptr_t{1}));
    if (SUCCEEDED(hr) && queue) queue->Release();
    expect(refused, what);
}

int sequence_checks(bool warp)
{
    // 1. Configuration before any device, as the title calls it.
    ConfigBlock config;
    {
        const HRESULT hr = D3D12ConfigureVirtualMemoryX(nullptr);
        std::printf("configure hr=0x%08lx: ", static_cast<unsigned long>(hr));
        expect(hr == E_POINTER, "a null block is refused");
    }
    for (std::uint32_t value : {0u, 1u, 2u, 5u, 0x80000003u, 0xffffffffu}) {
        config.reset(value);
        std::printf("value 0x%08x ", value);
        expect_configure(config, E_INVALIDARG, "a value other than 3 or 4 is refused and the block is unchanged");
    }
    for (std::size_t offset : {0u, 3u, 8u, 31u}) {
        config.reset(4);
        config.block()[offset] = 1;
        std::printf("byte +%zu ", offset);
        expect_configure(config, E_INVALIDARG, "a non-zero byte outside +4 is refused and the block is unchanged");
    }
    config.reset(4);
    expect_configure(config, S_OK, "the observed block (4 at +4) is accepted and left unchanged");
    expect_configure(config, DXGI_ERROR_INVALID_CALL, "a second configuration is refused");
    config.reset(3);
    expect_configure(config, DXGI_ERROR_INVALID_CALL, "a second configuration with the other sample is refused");

    // 2. The device through d3d12_xs.dll.
    bool unmatched = false;
    const auto base = create_device(warp, unmatched, create_xbox_device);
    if (unmatched) return 1;
    Gpu gpu;
    if (!base || FAILED(base.As(&gpu.device))) {
        std::puts(warp ? "No ID3D12Device3 on WARP" : "No hardware adapter with ID3D12Device3");
        return 77;
    }
    {
        ComPtr<IDXGIFactory4> factory;
        ComPtr<IDXGIAdapter1> adapter;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
            FAILED(factory->EnumAdapterByLuid(gpu.device->GetAdapterLuid(), IID_PPV_ARGS(&adapter)))) {
            std::puts("setup error: adapter of the created device not found");
            return 1;
        }
        const std::uint32_t parameters[16]{};
        void* device = reinterpret_cast<void*>(std::uintptr_t{1});
        HRESULT hr = D3D12XboxCreateDevice(adapter.Get(), parameters, __uuidof(ID3D12Device), &device);
        std::printf("device hr=0x%08lx: ", static_cast<unsigned long>(hr));
        expect(hr == E_INVALIDARG && !device, "a device request with a parameter block is refused");
        device = reinterpret_cast<void*>(std::uintptr_t{1});
        hr = D3D12XboxCreateDevice(adapter.Get(), nullptr, __uuidof(ID3D12Device3), &device);
        std::printf("device hr=0x%08lx: ", static_cast<unsigned long>(hr));
        expect(hr == E_NOINTERFACE && !device, "a device request for another interface is refused");
    }
    config.reset(4);
    expect_configure(config, DXGI_ERROR_INVALID_CALL, "configuration after device creation is refused");

    D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT va{};
    if (SUCCEEDED(gpu.device->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &va, sizeof(va)))) {
        std::printf("adapter VA bits: per resource %u, per process %u\n", va.MaxGPUVirtualAddressBitsPerResource,
                    va.MaxGPUVirtualAddressBitsPerProcess);
    }

    // 3. The first DIRECT queue with the default description, then refused requests.
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = gpu.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&gpu.queue));
    std::printf("queue hr=0x%08lx: ", static_cast<unsigned long>(hr));
    expect(SUCCEEDED(hr), "the first DIRECT queue with the default description is created");
    if (FAILED(hr)) return 1;
    {
        D3D12_COMMAND_QUEUE_DESC bad = queue_desc;
        bad.Type = static_cast<D3D12_COMMAND_LIST_TYPE>(0x7fffffff);
        expect_queue_refused(gpu.device.Get(), bad, "a queue of an unknown type is refused");
        bad = queue_desc;
        bad.Priority = 12345;
        expect_queue_refused(gpu.device.Get(), bad, "a queue with an unknown priority is refused");
        bad = queue_desc;
        bad.NodeMask = 1u << 31;
        expect_queue_refused(gpu.device.Get(), bad, "a queue on a node the adapter lacks is refused");
    }
    hr = gpu.device->GetDeviceRemovedReason();
    std::printf("device removed reason hr=0x%08lx: ", static_cast<unsigned long>(hr));
    expect(hr == S_OK, "the refusals leave the device in place");

    // 4. The first queue and the configured device still run the range translation.
    if (!gpu.init()) {
        std::puts("compute setup failed");
        return 1;
    }
    return translation_checks(gpu);
}

// ---- The configured range (d3d12_xs_range.h) ----

constexpr std::uint64_t tile = 64 * 1024;

// A 64 KiB pagefile section with a CPU view and a buffer placed over it, as one #934 physical page.
struct SectionBuffer {
    HANDLE section = nullptr;
    std::uint32_t* view = nullptr;
    ComPtr<ID3D12Heap> heap;
    ComPtr<ID3D12Resource> buffer;
    bool open(ID3D12Device3* device)
    {
        section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast<DWORD>(tile),
                                     nullptr);
        if (section) view = static_cast<std::uint32_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, tile));
        // OpenExistingHeapFromFileMapping can leave the section committed after exit (#2234), so the heap
        // is opened on the mapped view. The destructor releases the heap before it unmaps the view.
        if (!view || FAILED(device->OpenExistingHeapFromAddress(view, IID_PPV_ARGS(&heap)))) return false;
        const auto desc = buffer_desc(tile, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER |
                                                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        return SUCCEEDED(device->CreatePlacedResource(heap.Get(), 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                      IID_PPV_ARGS(&buffer)));
    }
    SectionBuffer() = default;
    SectionBuffer(const SectionBuffer&) = delete;
    SectionBuffer& operator=(const SectionBuffer&) = delete;
    ~SectionBuffer()
    {
        buffer.Reset();
        heap.Reset();
        if (view) UnmapViewOfFile(view);
        if (section) CloseHandle(section);
    }
};

// Maps the tile at range offset `offset` to pool tile `pool_tile` on the queue.
bool map_tile(Gpu& gpu, IX2WGpuRange* range, std::uint64_t offset, ID3D12Heap* pool, UINT pool_tile)
{
    const UINT index = static_cast<UINT>(offset / range->GetWindowSize());
    ComPtr<ID3D12Resource> window;
    if (FAILED(range->GetWindow(index, &window))) return false;
    D3D12_TILED_RESOURCE_COORDINATE coordinate{};
    coordinate.X = static_cast<UINT>((offset - index * range->GetWindowSize()) / tile);
    D3D12_TILE_REGION_SIZE region{};
    region.NumTiles = 1;
    const D3D12_TILE_RANGE_FLAGS flags = D3D12_TILE_RANGE_FLAG_NONE;
    const UINT count = 1;
    gpu.queue->UpdateTileMappings(window.Get(), 1, &coordinate, &region, pool, 1, &flags, &pool_tile, &count,
                                  D3D12_TILE_MAPPING_FLAG_NONE);
    return true;
}

// Reads the first `words` of the range at `offset` with one dispatch into `out` (dst = src ^ key).
bool read_range(Gpu& gpu, IX2WGpuRange* range, std::uint64_t offset, SectionBuffer& out)
{
    const D3D12_GPU_VIRTUAL_ADDRESS va = range->GetGPUVirtualAddress(offset, words * 4);
    ComPtr<ID3D12Resource> window;
    if (!va || FAILED(range->GetWindow(static_cast<UINT>(offset / range->GetWindowSize()), &window))) return false;
    std::memset(out.view, 0xcd, words * 4);
    return gpu.dispatch(window.Get(), va, out.buffer->GetGPUVirtualAddress(), out.buffer.Get());
}

unsigned zero_reads(const std::uint32_t* dst)
{
    unsigned zero = 0;
    for (UINT i = 0; i < words; ++i) zero += dst[i] == key;
    return zero;
}

constexpr std::uint64_t range_bytes = 32ull << 30;  // the 4 sample: (4 - 2) * 16 GiB

void CALLBACK count_destruction(void* count)
{
    ++*static_cast<unsigned*>(count);
}

bool watch_destruction(ID3D12Device* device, unsigned& count)
{
    ComPtr<ID3DDestructionNotifier> notifier;
    UINT id = 0;
    return SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&notifier))) &&
           SUCCEEDED(notifier->RegisterDestructionCallback(count_destruction, &count, &id));
}

// Releases the last caller-owned device reference and returns what Release reported.
ULONG release_last(ComPtr<ID3D12Device>& device)
{
    return device.Detach()->Release();
}

// A 32 GiB range on its owning handle: windows, mapped tiles read by dispatch, then Close().
int range_checks(bool warp)
{
    bool unmatched = false;
    const auto base = create_device(warp, unmatched);
    if (unmatched) return 1;
    Gpu gpu;
    if (!base || FAILED(base.As(&gpu.device))) {
        std::puts(warp ? "No ID3D12Device3 on WARP" : "No hardware adapter with ID3D12Device3");
        return 77;
    }
    D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT va{};
    if (SUCCEEDED(gpu.device->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &va, sizeof(va)))) {
        std::printf("adapter VA bits: per resource %u, per process %u\n", va.MaxGPUVirtualAddressBitsPerResource,
                    va.MaxGPUVirtualAddressBitsPerProcess);
    }
    if (!gpu.init()) {
        std::puts("compute setup failed");
        return 1;
    }
    ComPtr<IX2WGpuRange> range;
    HRESULT hr = d3d12_xs_range::create(gpu.device.Get(), range_bytes, &range);
    std::printf("create hr=0x%08lx\n", static_cast<unsigned long>(hr));
    expect(SUCCEEDED(hr) && range, "a 32 GiB range is created");
    if (!range) return 1;
    const std::uint64_t size = range->GetSize(), window_size = range->GetWindowSize();
    const UINT count = range->GetWindowCount();
    std::printf("range %llu GiB in %u windows of %llu MiB\n", static_cast<unsigned long long>(size >> 30), count,
                static_cast<unsigned long long>(window_size >> 20));
    expect(size == range_bytes && window_size <= (4ull << 30) && window_size % tile == 0 &&
               count == (size + window_size - 1) / window_size,
           "the range covers 32 GiB in windows of at most 4 GiB");
    expect(range->GetGPUVirtualAddress(window_size - 4, 8) == 0 && range->GetGPUVirtualAddress(size, 1) == 0 &&
               range->GetGPUVirtualAddress(size - 4, 4) != 0,
           "spans across a window edge or past the range have no GPU address");

    D3D12_HEAP_DESC pool_desc{};
    pool_desc.SizeInBytes = 3 * tile;
    pool_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    pool_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    ComPtr<ID3D12Heap> pool;
    SectionBuffer staging, out;
    if (FAILED(gpu.device->CreateHeap(&pool_desc, IID_PPV_ARGS(&pool))) || !staging.open(gpu.device.Get()) ||
        !out.open(gpu.device.Get())) {
        std::puts("setup error: tile pool or staging sections");
        return 1;
    }
    struct Idle {
        Gpu& gpu;
        ~Idle() { gpu.wait_idle(); }
    } idle{gpu};

    // Adapter capability, independent of the range: a test-owned reserved buffer with its fourth tile
    // mapped to pool tile 3, filled by copy and read by dispatch.
    {
        D3D12_HEAP_DESC probe_desc = pool_desc;
        probe_desc.SizeInBytes = tile;
        ComPtr<ID3D12Heap> probe_pool;
        ComPtr<ID3D12Resource> probe;
        const auto desc = buffer_desc(16 * tile, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (FAILED(gpu.device->CreateHeap(&probe_desc, IID_PPV_ARGS(&probe_pool))) ||
            FAILED(gpu.device->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                      IID_PPV_ARGS(&probe)))) {
            std::puts("setup error: capability probe resources");
            return 1;
        }
        D3D12_TILED_RESOURCE_COORDINATE coordinate{};
        coordinate.X = 3;
        D3D12_TILE_REGION_SIZE region{};
        region.NumTiles = 1;
        const D3D12_TILE_RANGE_FLAGS flags = D3D12_TILE_RANGE_FLAG_NONE;
        const UINT start = 0, tiles = 1;
        gpu.queue->UpdateTileMappings(probe.Get(), 1, &coordinate, &region, probe_pool.Get(), 1, &flags, &start,
                                      &tiles, D3D12_TILE_MAPPING_FLAG_NONE);
        fill(staging.view, 0x0f000000u);
        std::uint32_t expected_words[words];
        std::memcpy(expected_words, staging.view, sizeof(expected_words));
        std::memset(out.view, 0xcd, words * 4);
        const bool read = gpu.copy(probe.Get(), 3 * tile, staging.buffer.Get(), 0, tile) &&
                          gpu.dispatch(probe.Get(), probe->GetGPUVirtualAddress() + 3 * tile,
                                       out.buffer->GetGPUVirtualAddress(), out.buffer.Get()) &&
                          mismatches(expected_words, out.view) == 0;
        if (!read || FAILED(gpu.device->GetDeviceRemovedReason())) {
            std::puts("capability: a dispatch does not read a copied tile of a test-owned reserved buffer on this "
                      "adapter");
            return 77;
        }
    }

    // Range start, first tile of the second window, last tile of the range.
    const std::uint64_t offsets[] = {0, window_size, size - tile};
    expect(read_range(gpu, range.Get(), offsets[0], out) && zero_reads(out.view) == words,
           "an unmapped tile reads zero through the range");
    std::uint32_t staged[3][words];
    for (UINT i = 0; i < 3; ++i) {
        fill(staging.view, 0x10000000u * (i + 1));
        std::memcpy(staged[i], staging.view, sizeof(staged[i]));
        const UINT index = static_cast<UINT>(offsets[i] / window_size);
        ComPtr<ID3D12Resource> window;
        const bool mapped = SUCCEEDED(range->GetWindow(index, &window)) &&
                            map_tile(gpu, range.Get(), offsets[i], pool.Get(), i) &&
                            gpu.copy(window.Get(), offsets[i] - index * window_size, staging.buffer.Get(), 0, tile);
        const bool read = mapped && read_range(gpu, range.Get(), offsets[i], out);
        const unsigned bad = read ? mismatches(staged[i], out.view) : words;
        std::printf("tile at range offset 0x%llx (window %u): %u/%u words wrong\n",
                    static_cast<unsigned long long>(offsets[i]), index, bad, words);
        expect(bad == 0, "a mapped and copied tile is read back through the range's GPU address");
    }
    expect(read_range(gpu, range.Get(), offsets[0] + tile, out) && zero_reads(out.view) == words,
           "the range address shifted by one 64 KiB tile reads the unmapped neighbour");
    fill(staging.view, 0x70000000u);
    expect(read_range(gpu, range.Get(), offsets[0], out) && mismatches(staged[0], out.view) == 0,
           "CPU writes to the staging page after the copy are not seen through the range");

    // The owner closes after the queue is idle.
    gpu.wait_idle();
    range->Close();
    ComPtr<ID3D12Resource> none;
    expect(range->GetSize() == 0 && range->GetWindowCount() == 0 && range->GetGPUVirtualAddress(0, 4) == 0 &&
               range->GetWindow(0, &none) == E_INVALIDARG && !none,
           "after Close the range is empty");
    std::printf("range failures=%u\n", failures);
    return failures ? 1 : 0;
}

// Ownership: the windows hold the device only while the range holds them. Two rounds on a fresh device:
// round 0 keeps the range past the device's last caller release, then releases the range; round 1
// closes the range while another thread reads it, then releases the device.
int range_lifetime_checks(bool warp)
{
    // The exported path: a configured device from D3D12XboxCreateDevice carries no range, so its last
    // ordinary release destroys it.
    {
        ConfigBlock config;
        config.reset(4);
        expect_configure(config, S_OK, "the observed block (4 at +4) is accepted");
        bool unmatched = false;
        auto device = create_device(warp, unmatched, create_xbox_device);
        if (unmatched) return 1;
        if (!device) {
            std::puts(warp ? "No WARP device" : "No hardware device");
            return 77;
        }
        unsigned destroyed = 0;
        if (!watch_destruction(device.Get(), destroyed)) {
            std::puts("setup error: ID3DDestructionNotifier");
            return 1;
        }
        IUnknown* attached = nullptr;
        UINT bytes = sizeof(attached);
        const HRESULT hr = device->GetPrivateData(__uuidof(IX2WGpuRange), &bytes, &attached);
        if (attached) attached->Release();
        expect(hr == DXGI_ERROR_NOT_FOUND, "the configured device from d3d12_xs.dll carries no range");
        const ULONG left = release_last(device);
        std::printf("configured exported device released: %lu references left, %u callbacks\n", left, destroyed);
        expect(left == 0 && destroyed == 1, "its last ordinary release destroys it");
    }
    for (int round = 0; round < 2; ++round) {
        bool unmatched = false;
        auto device = create_device(warp, unmatched);
        if (unmatched) return 1;
        if (!device) {
            std::puts(warp ? "No WARP device" : "No hardware device");
            return 77;
        }
        unsigned destroyed = 0;
        if (!watch_destruction(device.Get(), destroyed)) {
            std::puts("setup error: ID3DDestructionNotifier");
            return 1;
        }
        IX2WGpuRange* range = nullptr;
        const HRESULT hr = d3d12_xs_range::create(device.Get(), range_bytes, &range);
        expect(SUCCEEDED(hr) && range, "a 32 GiB range is created");
        if (!range) return 1;
        const UINT count = range->GetWindowCount();
        if (round == 0) {
            const ULONG left = release_last(device);
            std::printf("round 0: device released with the range open: %lu references left, %u callbacks\n", left,
                        destroyed);
            expect(left == count && destroyed == 0, "an open range's windows hold the device, one reference each");
            const ULONG range_left = range->Release();
            std::printf("round 0: range released: %lu range references left, %u callbacks\n", range_left, destroyed);
            expect(range_left == 0 && destroyed == 1, "releasing the range destroys the device");
            continue;
        }
        // Readers see either the open range or an empty one while Close runs.
        const D3D12_GPU_VIRTUAL_ADDRESS first = range->GetGPUVirtualAddress(0, 4);
        std::atomic<bool> stop{false};
        std::atomic<unsigned> inconsistent{0}, reads{0};
        std::thread reader([&] {
            while (!stop.load()) {
                const D3D12_GPU_VIRTUAL_ADDRESS va = range->GetGPUVirtualAddress(0, 4);
                const UINT windows = range->GetWindowCount();
                ComPtr<ID3D12Resource> window;
                const HRESULT got = range->GetWindow(0, &window);
                inconsistent += (va != 0 && va != first) || (windows != 0 && windows != count) ||
                                (SUCCEEDED(got) != (window != nullptr));
                ++reads;
            }
        });
        while (reads.load() < 1000) SwitchToThread();
        range->Close();
        Sleep(10);
        stop = true;
        reader.join();
        std::printf("round 1: %u reads by the reader thread around Close, %u inconsistent\n", reads.load(),
                    inconsistent.load());
        expect(inconsistent == 0, "reads that overlap Close see the open range or an empty one");
        range->Close();
        expect(range->GetWindowCount() == 0, "a second Close does nothing");
        const ULONG left = release_last(device);
        std::printf("round 1: device released after Close: %lu references left, %u callbacks\n", left, destroyed);
        expect(left == 0 && destroyed == 1, "after Close the device's last release destroys it");
        range->Release();
    }
    std::printf("lifetime failures=%u\n", failures);
    return failures ? 1 : 0;
}

// ---- Window creation failing part way ----

unsigned window_calls = 0;
unsigned fail_window = 0;

HRESULT failing_window(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, ID3D12Resource** window)
{
    if (++window_calls == fail_window) return E_OUTOFMEMORY;
    return device->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(window));
}

// Window creation failing at window 1 or 3 returns its HRESULT and no range, and leaves no window
// holding the device.
int range_partial_checks(bool warp)
{
    for (const unsigned fail_at : {1u, 3u}) {
        bool unmatched = false;
        auto device = create_device(warp, unmatched);
        if (unmatched) return 1;
        if (!device) {
            std::puts(warp ? "No WARP device" : "No hardware device");
            return 77;
        }
        unsigned destroyed = 0;
        if (!watch_destruction(device.Get(), destroyed)) {
            std::puts("setup error: ID3DDestructionNotifier");
            return 1;
        }
        window_calls = 0;
        fail_window = fail_at;
        if (fail_at == 1) {
            for (const UINT64 invalid_size : {UINT64{0}, UINT64{65537}, ~UINT64{0} - 65535}) {
                auto* invalid_range = reinterpret_cast<IX2WGpuRange*>(std::uintptr_t{1});
                const HRESULT invalid = d3d12_xs_range::create(device.Get(), invalid_size, &invalid_range,
                                                              failing_window);
                expect(invalid == E_INVALIDARG && !invalid_range && window_calls == 0,
                       "invalid range size is rejected before creating any window");
                if (SUCCEEDED(invalid) && invalid_range) invalid_range->Release();
            }
        }
        IX2WGpuRange* range = reinterpret_cast<IX2WGpuRange*>(std::uintptr_t{1});
        const HRESULT hr = d3d12_xs_range::create(device.Get(), range_bytes, &range, failing_window);
        std::printf("window %u fails: hr=0x%08lx after %u calls\n", fail_at, static_cast<unsigned long>(hr),
                    window_calls);
        expect(hr == E_OUTOFMEMORY && window_calls == fail_at && !range,
               "the failing window's HRESULT is returned with no range");
        const ULONG left = release_last(device);
        expect(left == 0 && destroyed == 1, "the windows created before the failure no longer hold the device");
    }
    std::printf("partial failures=%u\n", failures);
    return failures ? 1 : 0;
}
}

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    const std::string_view scenario = argv[1];
    int result = 2;
    if (scenario == "shared_range"sv) {
        result = shared_range_checks() ? 0 : 1;
    } else if (scenario == "device_warp"sv) {
        result = device_checks(true);
    } else if (scenario == "device_hardware"sv) {
        result = device_checks(false);
    } else if (scenario == "sequence_warp"sv) {
        result = sequence_checks(true);
    } else if (scenario == "sequence_hardware"sv) {
        result = sequence_checks(false);
    } else {
        for (const bool warp : {true, false}) {
            const std::string suffix = warp ? "_warp" : "_hardware";
            if (scenario == "range" + suffix) result = range_checks(warp);
            if (scenario == "range_lifetime" + suffix) result = range_lifetime_checks(warp);
            if (scenario == "range_partial" + suffix) result = range_partial_checks(warp);
        }
    }
    std::printf("vm_translation %s: %s\n", argv[1], result == 0 ? "PASS" : result == 77 ? "SKIP" : "FAIL");
    return result;
}
