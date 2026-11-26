// Checks src/startup_stubs/committed_range.h: a range is judged from VirtualQuery alone and never touched. Real address
// space is used (reserved, committed and no-access pages); a vectored exception handler counts every access violation, so a
// version that touches the memory it asks about is caught.
// --mutate touch: judge a range by reading its first byte, as probe_readable does (a reserved page faults: the control must fail).
#include "startup_stubs/committed_range.h"

#include <cstdio>
#include <cstring>

namespace {
volatile LONG faults;
int failures;

LONG CALLBACK count_fault(EXCEPTION_POINTERS* e)
{
    if (e->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        InterlockedIncrement(&faults);  // only the mutant gets here
        // End with a plain exit code: CTest reports an unhandled access violation as an exception, which WILL_FAIL does not turn into a pass.
        std::printf("startup_committed_range: FAILED, memory was touched (access violation at %p)\n", e->ExceptionRecord->ExceptionAddress);
        std::fflush(stdout);
        ExitProcess(1);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

void check(const char* name, bool ok)
{
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}

// The mutant's judge: read the first byte. A reserved page raises an access violation (counted by the handler, then the process
// ends with the exception), so the control fails.
bool touch_readable(const void* address, std::size_t)
{
    volatile unsigned char sink = *static_cast<const volatile unsigned char*>(address);
    (void)sink;
    return true;
}
}  // namespace

int main(int argc, char** argv)
{
    const bool mutate = argc == 3 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "touch") == 0;
    if (argc != 1 && !mutate) {
        std::printf("usage: startup_committed_range_check [--mutate touch]\n");
        return 2;
    }
    AddVectoredExceptionHandler(1, count_fault);
    const auto judge = [&](const void* start, std::size_t bytes) {
        return mutate ? touch_readable(start, bytes) : x2w_local::committed_range(start, bytes);
    };

    constexpr std::size_t page = 0x1000;
    // 8 pages: [0,2) committed read-write, [2,4) reserved only, [4,6) committed read-only, [6,7) committed no-access, [7,8) reserved.
    auto* base = static_cast<unsigned char*>(VirtualAlloc(nullptr, 8 * page, MEM_RESERVE, PAGE_NOACCESS));
    check("address space reserved", base != nullptr);
    if (!base) return 1;
    check("commit 0-2", VirtualAlloc(base, 2 * page, MEM_COMMIT, PAGE_READWRITE) != nullptr);
    check("commit 4-6", VirtualAlloc(base + 4 * page, 2 * page, MEM_COMMIT, PAGE_READONLY) != nullptr);
    check("commit 6", VirtualAlloc(base + 6 * page, page, MEM_COMMIT, PAGE_NOACCESS) != nullptr);

    check("a committed range is readable", judge(base, 2 * page));
    check("a read-only committed range is readable", judge(base + 4 * page, 2 * page));
    const LONG before = faults;
    check("a reserved page is not there yet", !judge(base + 2 * page, page));
    check("a range running into a reserved page is refused", !judge(base, 3 * page));
    check("a no-access page is refused", !judge(base + 6 * page, page));
    check("free address space is refused", !judge(reinterpret_cast<void*>(0x10000), page));
    check("an empty range is refused", !judge(base, 0));
    check("none of the refusals touched memory", faults == before);
    if (!mutate) {
        check("prefix of a range that runs into a reserved page", x2w_local::committed_prefix(base, 4 * page) == 2 * page);
        // Fake queries: the walk follows region sizes, refuses a query that fails and a region that does not advance.
        std::size_t calls = 0;
        const std::size_t whole = x2w_local::committed_prefix_with(0x100000, 0x30000, [&](const void* at, MEMORY_BASIC_INFORMATION* info) {
            ++calls;
            info->BaseAddress = const_cast<void*>(at);
            info->RegionSize = 0x10000;
            info->State = MEM_COMMIT;
            info->Protect = PAGE_READONLY;
            return true;
        });
        check("a committed range of three regions is walked in three queries", whole == 0x30000 && calls == 3);
        const std::size_t stuck = x2w_local::committed_prefix_with(0x100000, 0x30000, [&](const void* at, MEMORY_BASIC_INFORMATION* info) {
            info->BaseAddress = const_cast<void*>(at);
            info->RegionSize = 0;
            info->State = MEM_COMMIT;
            info->Protect = PAGE_READONLY;
            return true;
        });
        check("a region that does not advance ends the walk", stuck == 0);
    }

    if (failures) {
        std::printf("startup_committed_range: %d FAILED (%ld access violations)\n", failures, static_cast<long>(faults));
        return 1;
    }
    std::printf("startup_committed_range: PASS\n");
    return 0;
}
