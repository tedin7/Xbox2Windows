// Exercise the console COPY queue's slot 31 with source-built XMem page grants.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "xmem_page_batch.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <iterator>
#include <stdexcept>
#include <vector>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
constexpr SIZE_T page_bytes = 65536;
constexpr UINT pool_pages = 1024;
constexpr UINT window_pages = 128;
constexpr UINT consecutive = 0, repeat = 1, null_range = 2, skip = 3;

using QueueMap = HRESULT (WINAPI*)(IUnknown*, UINT, const X2WPageBatch*, UINT64, UINT, UINT);
using Reserve = void* (WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using AllocPages = BOOL (WINAPI*)(ULONG_PTR, ULONG_PTR*, ULONG_PTR*);
using FreePages = BOOL (WINAPI*)(ULONG_PTR, ULONG_PTR*);
using Scatter = BOOL (WINAPI*)(void* const*, ULONG_PTR, const ULONG_PTR*);

template<class T> T load(HMODULE module, const char* name)
{
    const auto function = reinterpret_cast<T>(GetProcAddress(module, name));
    if (!function) throw std::runtime_error(name);
    return function;
}

template<class T> T slot(IUnknown* object, unsigned index)
{
    return reinterpret_cast<T>((*reinterpret_cast<void***>(object))[index]);
}

UINT64 address(const void* p) { return reinterpret_cast<UINT64>(p); }

void require(bool ok, const char* step)
{
    if (!ok) throw std::runtime_error(step);
}

void check(HRESULT hr, const char* step)
{
    if (SUCCEEDED(hr)) return;
    std::fprintf(stderr, "%s returned 0x%08lx\n", step, static_cast<unsigned long>(hr));
    throw std::runtime_error(step);
}

void expect_page(const BYTE* window, UINT page, BYTE value)
{
    const auto* first = window + page * page_bytes;
    const auto* end = first + page_bytes;
    const auto* mismatch = std::find_if(first, end, [value](BYTE byte) { return byte != value; });
    if (mismatch == end) return;
    std::fprintf(stderr, "page %u, byte %zu: expected 0x%02x, got 0x%02x\n", page,
                 static_cast<SIZE_T>(mismatch - first), static_cast<unsigned>(value),
                 static_cast<unsigned>(*mismatch));
    throw std::runtime_error("page contents");
}

void expect_placeholder(const BYTE* pointer)
{
    MEMORY_BASIC_INFORMATION info{};
    require(VirtualQuery(pointer, &info, sizeof(info)) && info.State == MEM_RESERVE,
            "null range leaves a reservation");
}

template<class Call> double elapsed_us(Call call)
{
    using Clock = std::chrono::steady_clock;
    const auto before = Clock::now();
    check(call(), "timed slot 31 batch");
    return std::chrono::duration<double, std::micro>(Clock::now() - before).count();
}

template<size_t N> void print_times(const char* name, std::array<double, N>& results)
{
    std::sort(results.begin(), results.end());
    std::printf("slot31 %s: median %.2f us, p95 %.2f us (%zu samples)\n",
                name, results[N / 2], results[(95 * N + 99) / 100 - 1], N);
}
}  // namespace

int wmain(int argc, wchar_t** argv) try
{
    const bool runs_requested = argc == 3 && std::wcscmp(argv[2], L"runs") == 0;
    if (argc != 2 && !runs_requested) return 2;
    ComPtr<IDXGIFactory4> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory");
    ComPtr<IDXGIAdapter> adapter;
    check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "select WARP");
    DXGI_ADAPTER_DESC adapter_desc{};
    check(adapter->GetDesc(&adapter_desc), "describe WARP adapter");
    std::printf("adapter=%ls debug=off\n", adapter_desc.Description);

    std::array<BYTE, 64> parameters{};
    ComPtr<IUnknown> device, queue;
    check(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                reinterpret_cast<void**>(device.GetAddressOf())), "create shim device");
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    using CreateQueue = HRESULT (WINAPI*)(IUnknown*, const D3D12_COMMAND_QUEUE_DESC*, const GUID*, void**);
    check(slot<CreateQueue>(device.Get(), 8)(device.Get(), &desc, &__uuidof(ID3D12CommandQueue),
                                              reinterpret_cast<void**>(queue.GetAddressOf())), "create shim COPY queue");
    const auto queue_map = slot<QueueMap>(queue.Get(), 31);

    // Load the provider after queue creation, as it may first appear at runtime.
    const HMODULE xmem = LoadLibraryW(argv[1]);
    require(xmem != nullptr, "load xmem.dll");
    const auto reserve = load<Reserve>(xmem, "XMemVirtualAlloc");
    const auto alloc = load<AllocPages>(xmem, "XMemAllocatePhysicalPages");
    const auto free_pages = load<FreePages>(xmem, "XMemFreePhysicalPages");
    const auto scatter = load<Scatter>(xmem, "XMemMapPhysicalPagesScatter");
    const auto stats_map = load<decltype(&X2WMapPageBatch)>(xmem, "X2WMapPageBatch");

    constexpr SIZE_T startup_reservation = SIZE_T{16} << 30;
    auto* const pool = static_cast<BYTE*>(reserve(nullptr, startup_reservation, 0x20402000, 0x21, PAGE_READWRITE));
    require(pool != nullptr, "reserve XMem addresses");
    auto* const window = pool + pool_pages * page_bytes;
    auto* const timing_window = window + window_pages * page_bytes;
    std::vector<ULONG_PTR> grants(pool_pages);
    std::vector<void*> pool_addresses(pool_pages);
    ULONG_PTR count = pool_pages;
    require(alloc(0x280, &count, grants.data()) && count == pool_pages, "allocate the 64 MiB source pool");
    for (UINT i = 0; i < pool_pages; ++i) pool_addresses[i] = pool + i * page_bytes;
    require(scatter(pool_addresses.data(), pool_pages, grants.data()), "map the source pool");
    MEMORY_BASIC_INFORMATION pool_info{};
    require(VirtualQuery(pool, &pool_info, sizeof(pool_info)) && pool_info.State == MEM_COMMIT,
            "query the backing of the source pool");
    const bool shared_views = pool_info.RegionSize > page_bytes;
    require(runs_requested || !shared_views, "default source pool uses individual views");
    for (UINT i = 0; i < pool_pages; ++i)
        std::memset(pool_addresses[i], i < pool_pages / 2 ? 0x2f : 0x90, page_bytes);
    std::memset(pool_addresses[1], 0x41, page_bytes);
    std::memset(pool_addresses[2], 0x42, page_bytes);
    std::memset(pool_addresses[3], 0x43, page_bytes);

    const auto submit = [&](UINT batch_count, const X2WPageBatch* batches, UINT flags = 0) {
        return queue_map(queue.Get(), batch_count, batches, address(pool), pool_pages, flags);
    };
    const X2WPageRange initial{consecutive, 0, 4};
    const X2WPageBatch first{address(window), 1, &initial};
    check(submit(1, &first), "map consecutive pool pages");
    expect_page(window, 0, 0x2f);
    expect_page(window, 1, 0x41);
    expect_page(window, 2, 0x42);
    expect_page(window, 3, 0x43);
    std::memset(pool, 0x71, page_bytes);
    expect_page(window, 0, 0x71);
    std::memset(window + page_bytes, 0x72, page_bytes);
    expect_page(pool, 1, 0x72);

    const X2WPageRange remap[]{{consecutive, 1, 1}, {consecutive, 0, 1}, {skip, 0, 1}, {null_range, 0, 1},
                              {repeat, 3, 2}};
    const X2WPageBatch update{address(window), static_cast<UINT>(std::size(remap)), remap};
    check(submit(1, &update), "swap, preserve, unmap and repeat");
    expect_page(window, 0, 0x72);
    expect_page(window, 1, 0x71);
    expect_page(window, 2, 0x42);
    expect_placeholder(window + 3 * page_bytes);
    expect_page(window, 4, 0x43);
    expect_page(window, 5, 0x43);
    std::memset(window + 4 * page_bytes, 0xb3, page_bytes);
    expect_page(window, 5, 0xb3);
    expect_page(pool, 3, 0xb3);

    const X2WPageRange outside{consecutive, pool_pages - 1, 2};
    const X2WPageBatch invalid[]{first, {address(window + 4 * page_bytes), 1, &outside}};
    require(FAILED(submit(2, invalid)), "reject a later range outside the declared pool");
    require(FAILED(submit(1, &first, 0x80000000)), "reject an unsupported batch flag");
    require(FAILED(submit(1, nullptr)), "reject a missing batch array");
    expect_page(window, 0, 0x72);
    expect_page(window, 1, 0x71);
    expect_placeholder(window + 3 * page_bytes);
    expect_page(window, 4, 0xb3);

    const X2WPageRange overlap_initial{consecutive, 0, 3};
    const X2WPageRange overlap_final[]{{repeat, 3, 1}, {skip, 0, 1}};
    const X2WPageBatch overlap[]{{address(window + 6 * page_bytes), 1, &overlap_initial},
                                  {address(window + 7 * page_bytes), 2, overlap_final}};
    check(submit(2, overlap), "apply overlapping batches in submission order");
    expect_page(window, 6, 0x71);
    expect_page(window, 7, 0xb3);
    expect_page(window, 8, 0x42);

    // The guard is inside an owned reservation but its private backing prevents MapViewOfFile3
    // from replacing the address. The valid first edit must be rolled back after that OS failure.
    auto* const occupied = timing_window + window_pages * page_bytes;
    require(VirtualFree(occupied, page_bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER),
            "split the rollback control placeholder after both destination windows");
    require(VirtualAlloc2(GetCurrentProcess(), occupied, page_bytes, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                          PAGE_READWRITE, nullptr, 0) == occupied, "occupy the rollback control page");
    std::memset(occupied, 0xcc, page_bytes);
    const X2WPageRange move{consecutive, 2, 1};
    const X2WPageBatch blocked[]{{address(window + 6 * page_bytes), 1, &move}, {address(occupied), 1, &move}};
    require(FAILED(submit(2, blocked)), "refuse an OS placement over private committed memory");
    expect_page(window, 6, 0x71);
    expect_page(window, 7, 0xb3);
    expect_page(window, 8, 0x42);
    expect_page(occupied, 0, 0xcc);
    std::memset(pool, 0xe0, page_bytes);
    expect_page(window, 6, 0xe0);
    std::puts("slot31: consecutive mapping, two-way late writes, swap, skip, null and repeated aliases matched");
    std::puts("slot31: invalid range, flags and missing batch were rejected; existing mappings persisted");
    std::puts("slot31: later overlapping update took effect and skip preserved the earlier update");
    std::puts("slot31: OS placement failure restored the first alias and preserved the private guard");

    // Four regions in one call, each pointing into a different part of the source pool.
    // The second set shifts every source region by 32 MiB.
    constexpr UINT regions = 4;
    constexpr UINT region_pages = window_pages / regions;
    X2WPageRange ranges[2][regions]{};
    X2WPageBatch batches[2][regions]{};
    for (UINT set = 0; set < 2; ++set) {
        for (UINT i = 0; i < regions; ++i) {
            ranges[set][i] = {consecutive, set * (pool_pages / 2) + 64 * (i + 1), region_pages};
            batches[set][i] = {address(timing_window + i * region_pages * page_bytes), 1, &ranges[set][i]};
        }
    }
    check(submit(regions, batches[0]), "populate the benchmark window");
    X2WPageBatchStats unchanged{}, changed{};
    check(stats_map(regions, batches[0], address(pool), pool_pages, 0, &unchanged), "query repeat stats");
    require(unchanged.pagesSkipped == window_pages && unchanged.pagesChanged == 0 &&
            unchanged.mapCalls == 0 && unchanged.unmapCalls == 0, "repeat avoids every OS remap");
    check(stats_map(regions, batches[1], address(pool), pool_pages, 0, &changed), "query changing stats");
    std::printf("provider stats (changed/skipped/map/unmap): repeat %llu/%llu/%llu/%llu, changed %llu/%llu/%llu/%llu\n",
                unchanged.pagesChanged, unchanged.pagesSkipped, unchanged.mapCalls, unchanged.unmapCalls,
                changed.pagesChanged, changed.pagesSkipped, changed.mapCalls, changed.unmapCalls);
    const UINT expected_os_calls = shared_views ? regions : window_pages;
    require(changed.pagesChanged == window_pages && changed.pagesSkipped == 0 &&
            changed.mapCalls == expected_os_calls && changed.unmapCalls == expected_os_calls,
            "changing ranges remaps exactly one view per region in shared mode or per page in default mode");

    std::array<double, 51> same{}, rotating{};
    UINT active = 1;
    for (size_t sample = 0; sample < same.size(); ++sample) {
        same[sample] = elapsed_us([&] { return submit(regions, batches[active]); });
        active ^= 1;
        rotating[sample] = elapsed_us([&] { return submit(regions, batches[active]); });
    }
    for (UINT i = 0; i < window_pages; ++i) expect_page(timing_window, i, active ? 0x90 : 0x2f);
    std::printf("pool=%u pages (%u MiB), batch=%u regions x %u pages\n", pool_pages,
                static_cast<UINT>(pool_pages * page_bytes >> 20), regions, region_pages);
    print_times("unchanged", same);
    print_times("changing", rotating);

    const X2WPageRange clear{null_range, 0, window_pages};
    const X2WPageBatch cleanup[]{{address(window), 1, &clear}, {address(timing_window), 1, &clear}};
    check(submit(2, cleanup), "clear the destination views");
    const std::vector<ULONG_PTR> holes(pool_pages);
    require(scatter(pool_addresses.data(), pool_pages, holes.data()), "unmap the source pool");
    require(free_pages(pool_pages, grants.data()), "release source grants");
    require(VirtualFree(occupied, 0, MEM_RELEASE), "release the private rollback control page");
    require(FreeLibrary(xmem), "release XMem provider");
    if (runs_requested && !shared_views) {
        std::puts("shared XMem views unavailable; the one-view-per-page fallback passed");
        return 77;
    }
    return 0;
}
catch (const std::exception& error)
{
    std::fprintf(stderr, "startup_queue_page_batch: %s\n", error.what());
    return 1;
}
