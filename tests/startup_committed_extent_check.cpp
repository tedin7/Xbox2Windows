// src/startup_stubs/committed_range.h committed_extent_with: where the data of a bound range can end, judged from VirtualQuery
// answers only (a fake memory map here). Reserved-only pages are untouched title memory (zero); a write commits them and the
// next call sees the new end, so the copy of a 16 MiB binding follows what the title has really written.
// --mutate prefix: use the committed PREFIX instead (stops at the first reserved gap): data committed after a gap is lost.
#include "startup_stubs/committed_range.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {
int failures;
void check(const char* name, bool ok)
{
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}

struct Region {
    std::uintptr_t begin, size;
    DWORD state, protect;
};
std::vector<Region> map;

bool query(const void* address, MEMORY_BASIC_INFORMATION* info)
{
    const auto at = reinterpret_cast<std::uintptr_t>(address);
    for (const Region& r : map) {
        if (at >= r.begin && at < r.begin + r.size) {
            *info = MEMORY_BASIC_INFORMATION{};
            info->BaseAddress = reinterpret_cast<void*>(r.begin);
            info->RegionSize = r.size;
            info->State = r.state;
            info->Protect = r.protect;
            return true;
        }
    }
    return false;
}
}  // namespace

int main(int argc, char** argv)
{
    const bool mutate = argc == 3 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "prefix");
    constexpr std::uintptr_t base = 0x100000000ull, mib = 1u << 20, total = 16 * mib;
    const auto extent = [&](std::uintptr_t bytes, bool* ok) {
        if (mutate) {
            *ok = true;
            return x2w_local::committed_prefix_with(base, bytes, query);
        }
        return x2w_local::committed_extent_with(base, bytes, query, ok);
    };
    bool ok = false;

    map = {{base, total, MEM_RESERVE, 0}};
    check("an untouched reserved buffer has no data", extent(total, &ok) == 0 && ok);

    map = {{base, 64 * 1024, MEM_COMMIT, PAGE_READWRITE}, {base + 64 * 1024, total - 64 * 1024, MEM_RESERVE, 0}};
    check("the first slot written: 64 KiB can hold data", extent(total, &ok) == 64 * 1024 && ok);

    // The title writes into a slot at 9 MiB: the data after the reserved gap must be covered.
    map = {{base, 64 * 1024, MEM_COMMIT, PAGE_READWRITE},
           {base + 64 * 1024, 9 * mib - 64 * 1024, MEM_RESERVE, 0},
           {base + 9 * mib, 64 * 1024, MEM_COMMIT, PAGE_READWRITE},
           {base + 9 * mib + 64 * 1024, total - 9 * mib - 64 * 1024, MEM_RESERVE, 0}};
    check("data committed after a reserved gap is covered", extent(total, &ok) == 9 * mib + 64 * 1024 && ok);

    map = {{base, total, MEM_COMMIT, PAGE_READWRITE}};
    check("a fully committed buffer is covered whole", extent(total, &ok) == total && ok);

    map = {{base, 4 * mib, MEM_COMMIT, PAGE_READWRITE}, {base + 4 * mib, total - 4 * mib, MEM_FREE, PAGE_NOACCESS}};
    extent(total, &ok);
    check("a free part is not ok", !ok);

    map = {{base, 4 * mib, MEM_COMMIT, PAGE_NOACCESS}, {base + 4 * mib, total - 4 * mib, MEM_RESERVE, 0}};
    extent(total, &ok);
    check("a no-access committed part is not ok", !ok);

    if (failures) return 1;
    std::printf("startup_committed_extent: PASS\n");
    return 0;
}
