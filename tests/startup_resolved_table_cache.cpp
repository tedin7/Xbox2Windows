// Root descriptor table cache (src/startup_stubs/resolved_table_cache.h): what it keeps and what it must not.
#include "../src/startup_stubs/resolved_table_cache.h"

#include <cstdio>

using x2w_startup::ResolvedTableCache;
using x2w_startup::ViewTable;

static int failures;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

int main()
{
    ResolvedTableCache cache;
    const ViewTable found = x2w_startup::resolve_view_table(0x1000, 0x500000, 64, 32, 0x1000 + 32 * 5);
    CHECK(found.address == 0x500000 + 32 * 5 && found.bytes == 32 * 59);

    CHECK(cache.find(7, 0x1000 + 32 * 5).address == 0);           // nothing kept yet
    cache.keep(7, 0x1000 + 32 * 5, found);
    CHECK(cache.find(7, 0x1000 + 32 * 5).address == found.address);
    CHECK(cache.find(7, 0x1000 + 32 * 5).bytes == found.bytes);
    CHECK(cache.find(8, 0x1000 + 32 * 5).address == 0);           // another view heap: not the same answer
    CHECK(cache.find(7, 0x1000 + 32 * 6).address == 0);           // another handle

    cache.keep(7, 0x2000, ViewTable{});                          // a miss is never kept
    CHECK(cache.find(7, 0x2000).address == 0);

    // two handles that land on one slot: the newer wins and the older is a miss, never a wrong answer
    std::uint64_t clash = 0x1000 + 32 * 5 + 16;
    for (;; clash += 16) {
        ResolvedTableCache probe;
        probe.keep(7, 0x1000 + 32 * 5, found);
        probe.keep(7, clash, ViewTable{1, 1});
        if (!probe.find(7, 0x1000 + 32 * 5).address) break;     // same slot
    }
    CHECK(cache.find(7, clash).address == 0);
    cache.keep(7, clash, ViewTable{0x900000, 64});
    CHECK(cache.find(7, clash).address == 0x900000);
    CHECK(cache.find(7, 0x1000 + 32 * 5).address == 0);

    cache.clear();
    CHECK(cache.find(7, clash).address == 0);
    if (!failures) std::puts("resolved table cache: all checks pass");
    return failures ? 1 : 0;
}
