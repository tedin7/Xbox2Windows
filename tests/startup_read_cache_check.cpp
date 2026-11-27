#include "startup_stubs/local_log.h"

#include <cstdio>
#include <cwchar>
#include <thread>

int wmain(int argc, wchar_t** argv)
{
    if (argc != 3 || !LoadLibraryW(argv[1])) return 2;
    const bool enabled = std::wcscmp(argv[2], L"off") != 0;
    const auto* cache = x2w_local::read_cache();
    if (enabled != (cache != nullptr)) return 3;
    constexpr SIZE_T page = 4096, slots = 96;
    auto* base = static_cast<unsigned char*>(VirtualAlloc(nullptr, slots * 2 * page, MEM_RESERVE, PAGE_NOACCESS));
    if (!base) return 4;
    for (SIZE_T i = 0; i < slots; ++i) {
        if (!VirtualAlloc(base + 2 * i * page, page, MEM_COMMIT, PAGE_READWRITE)) return 5;
    }

    const auto probe = [](const void* at, SIZE_T bytes) {
        return x2w_local::probe_readable(reinterpret_cast<std::uintptr_t>(at), bytes);
    };
    const auto before = cache ? cache->stats().queries : 0;
    for (SIZE_T i = 0; i < 10000; ++i) {
        const SIZE_T bytes = 1 + i % 239;
        const auto* at = base + (i % slots) * 2 * page + page - bytes;
        // Both switch settings run the same requests and CPU expectation.
        if (probe(at, bytes) != bytes) return 6;
    }
    if (cache && cache->stats().queries - before >= 200) return 7;

    const auto warm = cache ? cache->stats() : x2w_local::ReadCacheStats{};
    std::size_t observed = 0;
    std::thread reader([&] { observed = probe(base + 8, 16); });
    reader.join();
    if (observed != 16 || (cache && cache->stats().queries != warm.queries)) return 14;
    if (cache) std::printf("checkerboard: %llu system queries for 10000 probes\n", warm.queries - before);

    auto* changed = base + 34 * page;
    if (!VirtualFree(changed, page, MEM_DECOMMIT)) return 8;
    const auto queries = cache ? cache->stats().queries : 0;
    if (probe(changed, page) != 0) return 9;
    // The guarded read catches even the mutant. Its missing miss proves it had
    // incorrectly entered the touch with a stale positive cache fact.
    if (cache && cache->stats().queries != queries + 1) {
        std::puts("FAIL: decommitted page was still a cache hit");
        return 10;
    }
    const auto neighbor_queries = cache ? cache->stats().queries : 0;
    if (probe(changed + 2 * page, 1) != 1) return 11;
    if (cache && cache->stats().queries != neighbor_queries) return 12;
    if (!VirtualFree(base, 0, MEM_RELEASE)) return 13;
    return 0;
}
