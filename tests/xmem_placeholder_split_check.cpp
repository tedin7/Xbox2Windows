// Queue slot 31 (X2WMapPageBatch) into the middle of an XMem reservation needs a placeholder split. Where the region
// query does not report placeholders (Wine), xmem probes whether a placeholder release refuses a plain reservation
// and then offers reserved regions to the split (xmem placeholder_fallback).
//   xmem_placeholder_split_check <xmem.dll>           split succeeds; a committed target still fails and stays intact
//   xmem_placeholder_split_check <xmem.dll> control   the same checks in the test-controls build (X2W_XMEM_TEST_CONTROLS)
//                                                     with X2W_XMEM_TEST_NO_PLACEHOLDER_FALLBACK=1; CTest expects the
//                                                     split step to fail
//   xmem_placeholder_split_check <xmem.dll> deployed  the same checks in the deployed build with the same switch, which
//                                                     that build ignores, so they pass
//   xmem_placeholder_split_check <xmem.dll> partial   test-controls build with shared views: a two-page view over a
//                                                     private reservation whose second page is committed is refused
//                                                     by the region walk (x2w_xmem_alias_counts), and the committed
//                                                     page keeps its bytes; prints where its allocation starts after
//                                                     the batch (the limit in alias_placeholders)
// control, deployed and partial exit 77 where the OS reports the bit, because there the fallback is never used. Each
// mode sets or clears X2W_XMEM_TEST_NO_PLACEHOLDER_FALLBACK itself, so an inherited value cannot change the result.
#include <windows.h>
#include "xmem_page_batch.h"
#include <cstdio>
#include <cstring>
#include <cwchar>

namespace {
constexpr SIZE_T page_bytes = 65536;
constexpr UINT pool_pages = 4;
using Reserve = void* (WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using AllocPages = BOOL (WINAPI*)(ULONG_PTR, ULONG_PTR*, ULONG_PTR*);
using Scatter = BOOL (WINAPI*)(void* const*, ULONG_PTR, const ULONG_PTR*);
using AliasCounts = BOOL (WINAPI*)(ULONGLONG*, ULONGLONG*);

bool require(bool ok, const char* step)
{
    if (!ok) std::printf("FAIL: %s (last error %lu)\n", step, GetLastError());
    return ok;
}

bool os_reports_placeholder_bit()
{
    void* probe = VirtualAlloc2(GetCurrentProcess(), nullptr, 2 * page_bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                PAGE_NOACCESS, nullptr, 0);
    if (!probe) return false;
    WIN32_MEMORY_REGION_INFORMATION info{};
    const bool bit = QueryVirtualMemoryInformation(GetCurrentProcess(), probe, MemoryRegionInfo, &info, sizeof(info),
                                                   nullptr) && (info.Flags & 0x100);
    VirtualFree(probe, 0, MEM_RELEASE);
    return bit;
}

MEMORY_BASIC_INFORMATION query(const BYTE* pointer)
{
    MEMORY_BASIC_INFORMATION info{};
    VirtualQuery(pointer, &info, sizeof(info));
    return info;
}

// Where the region's allocation starts, in pages from `at` (negative: before it).
long long base_pages(const MEMORY_BASIC_INFORMATION& info, const BYTE* at)
{
    return (static_cast<const BYTE*>(info.AllocationBase) - at) / static_cast<long long>(page_bytes);
}

// partial: two pages of the reservation become a private reservation whose second page is committed and tagged. Its
// first region is reserved, so a check of that region alone would accept the whole two-page view.
int partial_case(HMODULE xmem, BYTE* pool, decltype(&X2WMapPageBatch) batch_map)
{
    const auto counts = reinterpret_cast<AliasCounts>(GetProcAddress(xmem, "x2w_xmem_alias_counts"));
    ULONGLONG refusals0 = 0, splits0 = 0, refusals1 = 0, splits1 = 0;
    if (!require(counts && counts(&refusals0, &splits0), "read the alias counts (test-controls build)")) return 1;
    BYTE* const piece = pool + 192 * page_bytes;
    if (!require(VirtualFree(piece, 2 * page_bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), "split out two pages") ||
        !require(VirtualAlloc2(GetCurrentProcess(), piece, 2 * page_bytes, MEM_RESERVE | MEM_REPLACE_PLACEHOLDER,
                               PAGE_READWRITE, nullptr, 0) == piece, "reserve the two pages privately") ||
        !require(VirtualAlloc(piece + page_bytes, page_bytes, MEM_COMMIT, PAGE_READWRITE) == piece + page_bytes,
                 "commit the second page"))
        return 1;
    std::memset(piece + page_bytes, 0x7a, page_bytes);
    const MEMORY_BASIC_INFORMATION before = query(piece + page_bytes);
    if (!require(before.AllocationBase == piece, "the two pages are one allocation")) return 1;
    const X2WPageRange two{0, 1, 2};
    const X2WPageBatch onto_partial{reinterpret_cast<UINT64>(piece), 1, &two};
    const HRESULT hr = batch_map(1, &onto_partial, reinterpret_cast<UINT64>(pool), pool_pages, 0, nullptr);
    if (!require(counts(&refusals1, &splits1), "read the alias counts")) return 1;
    const MEMORY_BASIC_INFORMATION first = query(piece), second = query(piece + page_bytes);
    std::printf("batch onto a reserved page and a committed one -> 0x%08lx; alias refusals %llu, placeholder "
                "releases tried %llu; after: state 0x%lx type 0x%lx, then state 0x%lx type 0x%lx byte 0x%02x\n",
                static_cast<unsigned long>(hr), refusals1 - refusals0, splits1 - splits0, first.State, first.Type,
                second.State, second.Type, second.State == MEM_COMMIT ? piece[page_bytes] : 0u);
    std::printf("allocation base in pages from the first page: first page %+lld (region %llu pages); committed page "
                "%+lld before the batch, %+lld after\n",
                base_pages(first, piece), static_cast<unsigned long long>(first.RegionSize / page_bytes),
                base_pages(before, piece), base_pages(second, piece));
    // The walk refuses the two-page view. The batch then retries each page on its own (install_alias_plan): the
    // reserved page alone is accepted, released and mapped, and the rollback unmaps it again; the committed page alone
    // is refused. A walk that checks only the first region of each allocation accepts the view, and under Wine the
    // release then succeeds and the view replaces the committed page (measured 2026-11-27 before the fix).
    // Limit: the fallback cannot tell a reserved page of a private allocation made in place of a placeholder
    // (MEM_REPLACE_PLACEHOLDER) from a placeholder. Under Wine the per-page release cuts that page out of the caller's
    // allocation: afterwards the committed page's allocation starts one page later (measured 2026-11-27). The row
    // asserts only that the first page is still one reserved page with its own address as its allocation base, and
    // prints where the committed page's allocation starts.
    if (!require(refusals1 - refusals0 >= 1, "the region walk refuses the two-page view") ||
        !require(FAILED(hr), "the partly committed target is refused") ||
        !require(first.State == MEM_RESERVE && first.AllocationBase == piece && first.RegionSize == page_bytes &&
                     second.State == MEM_COMMIT && second.Type == MEM_PRIVATE && piece[page_bytes] == 0x7a &&
                     piece[2 * page_bytes - 1] == 0x7a,
                 "the committed page keeps its bytes and the first page stays one reserved page at its address"))
        return 1;
    if (second.AllocationBase != piece)
        std::puts("limit: the per-page retry cut the reserved page out of the caller's allocation "
                  "(docs/experiments/xmem-page-runs.md)");
    std::puts("partial: a view over a reserved and a committed page was refused; the committed page kept its bytes");
    return 0;
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    const bool control = argc == 3 && std::wcscmp(argv[2], L"control") == 0;
    const bool deployed = argc == 3 && std::wcscmp(argv[2], L"deployed") == 0;
    const bool partial = argc == 3 && std::wcscmp(argv[2], L"partial") == 0;
    if (argc != 2 && !control && !deployed && !partial) return 2;
    // xmem reads the switch once, in its first placeholder probe; set it before loading xmem.dll.
    SetEnvironmentVariableW(L"X2W_XMEM_TEST_NO_PLACEHOLDER_FALLBACK", control || deployed ? L"1" : nullptr);
    if (partial) {
        // A two-page view needs shared views (one view per run of pool pages); joins across calls stay off.
        SetEnvironmentVariableW(L"X2W_XMEM_SHARED_VIEWS", L"1");
        SetEnvironmentVariableW(L"X2W_XMEM_JOIN_CALLS", nullptr);
    }
    const bool bit = os_reports_placeholder_bit();
    std::printf("placeholder bit reported=%d\n", bit);
    if ((control || deployed || partial) && bit) {
        std::puts("SKIP: the OS reports the placeholder bit, so the fallback is never used");
        return 77;
    }
    const HMODULE xmem = LoadLibraryW(argv[1]);
    if (!require(xmem != nullptr, "load xmem.dll")) return 1;
    const auto reserve = reinterpret_cast<Reserve>(GetProcAddress(xmem, "XMemVirtualAlloc"));
    const auto alloc = reinterpret_cast<AllocPages>(GetProcAddress(xmem, "XMemAllocatePhysicalPages"));
    const auto scatter = reinterpret_cast<Scatter>(GetProcAddress(xmem, "XMemMapPhysicalPagesScatter"));
    const auto batch_map = reinterpret_cast<decltype(&X2WMapPageBatch)>(GetProcAddress(xmem, "X2WMapPageBatch"));
    if (!require(reserve && alloc && scatter && batch_map, "find the xmem exports")) return 1;

    auto* const pool = static_cast<BYTE*>(reserve(nullptr, SIZE_T{16} << 30, 0x20402000, 0x21, PAGE_READWRITE));
    if (!require(pool != nullptr, "reserve XMem addresses")) return 1;
    ULONG_PTR grants[pool_pages]{}, count = pool_pages;
    void* addresses[pool_pages]{};
    for (UINT i = 0; i < pool_pages; ++i) addresses[i] = pool + i * page_bytes;
    if (!require(alloc(0x280, &count, grants) && count == pool_pages, "allocate the pool") ||
        !require(scatter(addresses, pool_pages, grants), "map the pool"))
        return 1;
    for (UINT i = 0; i < pool_pages; ++i) std::memset(addresses[i], 0x51 + i, page_bytes);
    if (partial) return partial_case(xmem, pool, batch_map);

    // Negative control: a committed page inside the reservation is not a placeholder. The batch must fail and leave
    // the page committed with its bytes.
    BYTE* const committed = pool + 64 * page_bytes;
    if (!require(VirtualFree(committed, page_bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), "split out a page") ||
        !require(VirtualAlloc2(GetCurrentProcess(), committed, page_bytes, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                               PAGE_READWRITE, nullptr, 0) == committed, "commit that page"))
        return 1;
    std::memset(committed, 0x77, page_bytes);
    const X2WPageRange one{0, 0, 1};
    const X2WPageBatch onto_committed{reinterpret_cast<UINT64>(committed), 1, &one};
    const HRESULT refused = batch_map(1, &onto_committed, reinterpret_cast<UINT64>(pool), pool_pages, 0, nullptr);
    std::printf("batch onto a committed page -> 0x%08lx\n", static_cast<unsigned long>(refused));
    // The code is the OS's map failure: Windows 487, Wine 87 (measured 2026-11-26).
    if (!require(FAILED(refused), "a committed target is refused") ||
        !require(query(committed).State == MEM_COMMIT && committed[0] == 0x77 && committed[page_bytes - 1] == 0x77,
                 "the committed target keeps its bytes"))
        return 1;

    // The split: pool pages 1 and 2 into the middle of an untouched part of the reservation.
    BYTE* const window = pool + 128 * page_bytes;
    const X2WPageRange two{0, 1, 2};
    const X2WPageBatch split{reinterpret_cast<UINT64>(window + page_bytes), 1, &two};
    const HRESULT hr = batch_map(1, &split, reinterpret_cast<UINT64>(pool), pool_pages, 0, nullptr);
    std::printf("batch into the middle of the reservation -> 0x%08lx\n", static_cast<unsigned long>(hr));
    if (!require(SUCCEEDED(hr), "map two pool pages into the middle of the reservation") ||
        !require(window[page_bytes] == 0x52 && window[2 * page_bytes] == 0x53 && window[3 * page_bytes - 1] == 0x53,
                 "the destination reads the pool pages") ||
        !require(query(window).State == MEM_RESERVE && query(window + 3 * page_bytes).State == MEM_RESERVE,
                 "the neighbours stay reserved"))
        return 1;
    window[page_bytes] = 0x60;
    if (!require(pool[page_bytes] == 0x60, "a write through the destination reaches the pool page")) return 1;
    std::puts("slot31 split inside an XMem reservation matched; a committed target was refused and kept");
    if (deployed) std::puts("deployed build: X2W_XMEM_TEST_NO_PLACEHOLDER_FALLBACK=1 was ignored");
    return 0;
}
