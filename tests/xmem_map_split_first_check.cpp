// XMemMapPhysicalPages into an owned XMem reservation: when the target is in the middle of a larger placeholder, xmem
// splits first and maps once (one MapViewOfFile3 per page). Under Wine each failed map is a server round trip whose
// cost grows with the number of views, and the title maps one 64 KiB page per call at startup.
//   xmem_map_split_first_check <xmem.dll>           N single-page maps take N map attempts; an exact placeholder
//                                                   (left by an unmap) is mapped without a split
//   xmem_map_split_first_check <xmem.dll> control   test-controls build (X2W_XMEM_TEST_CONTROLS) with
//                                                   X2W_XMEM_TEST_MAP_FIRST=1: the old order takes 2N map attempts and
//                                                   prints "map-first control: maps=2N"
//   xmem_map_split_first_check <xmem.dll> deployed  deployed build with X2W_XMEM_TEST_MAP_FIRST=1: the build ignores
//                                                   the switch, so the checks of the first mode pass
// Each mode sets or clears X2W_XMEM_TEST_MAP_FIRST itself, so an inherited value cannot change the result.
#include <windows.h>
#include <cstdio>
#include <cwchar>

namespace {
constexpr SIZE_T page_bytes = 65536;
constexpr ULONG_PTR pages = 8;
using Reserve = void* (WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using AllocPages = BOOL (WINAPI*)(ULONG_PTR, ULONG_PTR*, ULONG_PTR*);
using MapPages = void* (WINAPI*)(void*, ULONG_PTR, const ULONG_PTR*);
using MapCounts = BOOL (WINAPI*)(ULONGLONG*, ULONGLONG*);

bool require(bool ok, const char* step)
{
    if (!ok) std::printf("FAIL: %s (last error %lu)\n", step, GetLastError());
    return ok;
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    const bool control = argc == 3 && std::wcscmp(argv[2], L"control") == 0;
    const bool deployed = argc == 3 && std::wcscmp(argv[2], L"deployed") == 0;
    if (argc != 2 && !control && !deployed) return 2;
    // xmem reads the switch on first use and keeps the value; set it before any map.
    SetEnvironmentVariableW(L"X2W_XMEM_TEST_MAP_FIRST", control || deployed ? L"1" : nullptr);
    const HMODULE xmem = LoadLibraryW(argv[1]);
    if (!require(xmem != nullptr, "load xmem.dll")) return 1;
    const auto reserve = reinterpret_cast<Reserve>(GetProcAddress(xmem, "XMemVirtualAlloc"));
    const auto alloc = reinterpret_cast<AllocPages>(GetProcAddress(xmem, "XMemAllocatePhysicalPages"));
    const auto map = reinterpret_cast<MapPages>(GetProcAddress(xmem, "XMemMapPhysicalPages"));
    const auto counts = reinterpret_cast<MapCounts>(GetProcAddress(xmem, "x2w_xmem_map_counts"));
    if (!require(reserve && alloc && map && counts, "find the xmem exports")) return 1;
    auto* const pool = static_cast<BYTE*>(reserve(nullptr, SIZE_T{16} << 30, 0x20402000, 0x21, PAGE_READWRITE));
    if (!require(pool != nullptr, "reserve XMem addresses")) return 1;
    ULONG_PTR grants[pages]{}, count = pages;
    if (!require(alloc(0x280, &count, grants) && count == pages, "allocate the pages")) return 1;

    // N single-page maps, each into the middle of the reservation's placeholder (every other page, so no two
    // targets are neighbours).
    ULONGLONG maps0 = 0, splits0 = 0, maps1 = 0, splits1 = 0;
    if (!require(counts(&maps0, &splits0), "read the map counts")) return 1;
    for (ULONG_PTR i = 0; i < pages; ++i) {
        BYTE* const target = pool + (16 + 2 * i) * page_bytes;
        if (!require(map(target, 1, &grants[i]) == target, "map one page into the reservation")) return 1;
        target[0] = static_cast<BYTE>(0x40 + i);
        target[page_bytes - 1] = static_cast<BYTE>(0x40 + i);
    }
    if (!require(counts(&maps1, &splits1), "read the map counts")) return 1;
    const ULONGLONG maps = maps1 - maps0, splits = splits1 - splits0;
    std::printf("%llu single-page maps: map attempts %llu, split attempts %llu\n", static_cast<unsigned long long>(pages),
                maps, splits);
    for (ULONG_PTR i = 0; i < pages; ++i) {
        const BYTE* const target = pool + (16 + 2 * i) * page_bytes;
        if (!require(target[0] == 0x40 + i && target[page_bytes - 1] == 0x40 + i, "each mapped page keeps its bytes"))
            return 1;
    }
    MEMORY_BASIC_INFORMATION gap{};
    VirtualQuery(pool + 17 * page_bytes, &gap, sizeof(gap));
    if (!require(gap.State == MEM_RESERVE, "the page between two targets stays reserved")) return 1;
    if (control) {
        if (maps == 2 * pages && splits == pages) {
            std::puts("map-first control: maps=2N");
            return 1;  // CTest passes the control on the line above, not on the exit code
        }
        std::printf("FAIL: map-first control measured %llu maps for %llu pages\n", maps,
                    static_cast<unsigned long long>(pages));
        return 1;
    }
    if (!require(maps == pages && splits == pages, "split-first: one map attempt and one split per page")) return 1;

    // An exact placeholder: unmap the first page (the placeholder is kept), then map a page there again. No split.
    BYTE* const first = pool + 16 * page_bytes;
    if (!require(map(first, 1, nullptr) == first, "unmap the first page")) return 1;
    MEMORY_BASIC_INFORMATION hole{};
    VirtualQuery(first, &hole, sizeof(hole));
    std::printf("after unmap: state 0x%lx base %p allocation %p size 0x%llx\n", hole.State, hole.BaseAddress,
                hole.AllocationBase, static_cast<unsigned long long>(hole.RegionSize));
    if (!require(counts(&maps0, &splits0), "read the map counts") ||
        !require(map(first, 1, &grants[0]) == first, "map a page into the exact placeholder") ||
        !require(counts(&maps1, &splits1), "read the map counts"))
        return 1;
    std::printf("exact placeholder: map attempts %llu, split attempts %llu\n", maps1 - maps0, splits1 - splits0);
    if (!require(maps1 - maps0 == 1 && splits1 - splits0 == 0, "an exact placeholder is mapped without a split") ||
        !require(first[0] == 0x40, "the remapped page reads its bytes"))
        return 1;
    std::puts("split-first: maps=N; exact placeholder mapped without a split");
    if (deployed) std::puts("deployed build: X2W_XMEM_TEST_MAP_FIRST=1 was ignored");
    return 0;
}
