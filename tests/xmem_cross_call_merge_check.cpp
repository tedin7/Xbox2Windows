// Cross-call merge of shared views (X2W_XMEM_SHARED_VIEWS=1 and X2W_XMEM_JOIN_CALLS=1, src/xmem_mapped_pages.cpp,
// merge_with_previous). A run mapped by a call of its own joins the recorded run that ends at its address when its
// first page is the next slot of the section behind that run's last page and both lie in one reservation of the shim.
// What a view is, is measured with VirtualQuery: inside a view AllocationBase is the view's base and RegionSize the
// bytes to the view's end.
//
// Modes (each sets X2W_XMEM_SHARED_VIEWS and X2W_XMEM_JOIN_CALLS itself; JOIN_CALLS=1 in every mode but no-join):
//   check     (test-controls xmem.dll) every case below.
//   control   (test-controls xmem.dll, X2W_XMEM_TEST_NO_MERGE=1) single-page calls keep a view each: the layout
//             before the merge. Prints "no-merge control: 32 views".
//   deployed  (deployed xmem.dll, X2W_XMEM_TEST_NO_MERGE=1) the deployed shim ignores the test switch and merges.
//   default   (deployed xmem.dll, X2W_XMEM_SHARED_VIEWS unset) every page keeps a view of its own; the join
//             switch alone does nothing.
//   no-join   (deployed xmem.dll, X2W_XMEM_JOIN_CALLS unset) single-page calls keep a view each: joins are opt-in.
//             Prints "no-join: 32 views".
//   timing    (test-controls xmem.dll; not a ctest case) ms of a join, of a break and of an own-view map against the
//             run length R (X2W_XMEM_TEST_MERGE_CAP): a table of "timing:" lines, one row per R from 16 to 4096.
// The pass line of control, deployed, default and no-join is printed at the end of main, only when no check failed.
//
// Cases of `check`:
//   joined      32 single-page calls on consecutive slots: one view of 32 pages; each page keeps the mark written
//               right after its own call.
//   middle      unmapping page 10 of that view leaves views [0, 10) and [11, 32) with their marks.
//   reversed    32 single-page calls on slots in reverse order (adjacent addresses, not adjacent slots): 32 views,
//               page i shows slot 31 - i.
//   window      X2W_XMEM_TEST_HOLD_WINDOW holds the merge while the earlier run is unmapped: a raw query of one
//               of its pages is not committed; a thread that reads the page and an x2w_xmem_stable_call query both
//               wait, then read the mark and MEM_COMMIT; the result is one view of 9 pages.
//   refused-4/5 X2W_XMEM_TEST_FAULT=4:0 (the joined map refused) and 5:0 (joining the placeholders refused): the call
//               succeeds, the earlier run is a view of 8 pages again, the new page a view of its own, marks kept,
//               the pages unmap and release, and the live page count returns to where it was.
//   two-reservations  a caller placeholder split in two pieces, each recorded with XMemVirtualAlloc (permissive
//               replace-placeholder call): a page at the start of the second piece does not join the first's run.
//   capped      192 single-page calls make three runs of 64 pages (max_merged_run_pages); an unmap held in its window
//               inside the middle run takes apart at most 64 pages, and the outer runs stay mapped.
//
// Usage: xmem_cross_call_merge_check <xmem.dll> check|control|deployed|default|no-join|timing
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {
using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI*)(SIZE_T, SIZE_T*, SIZE_T*);
using XMemMapPhysicalPagesFn = void*(WINAPI*)(void*, SIZE_T, SIZE_T*);
using XMemFreePhysicalPagesFn = BOOL(WINAPI*)(SIZE_T, SIZE_T*);
using StableCallFn = BOOL(WINAPI*)(void (*)(void*), void*);
using PageStatsFn = BOOL(WINAPI*)(ULONGLONG*);
using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);

constexpr SIZE_T page = SIZE_T{64} << 10;
constexpr SIZE_T count = 32;

int failures = 0;
const char* token = nullptr;  // the mode's pass line; main prints it at its end, and only with no failure

bool check(bool ok, const char* name, const char* detail = "")
{
    std::printf("%s: %s%s%s\n", ok ? "ok" : "check_failed", name, *detail ? " " : "", detail);
    if (!ok) ++failures;
    return ok;
}

struct Region {
    ULONG_PTR allocation_base = 0;
    SIZE_T size = 0;
    DWORD state = 0;
};

Region region(const unsigned char* address)
{
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(address, &info, sizeof(info)) != sizeof(info)) return {};
    return {reinterpret_cast<ULONG_PTR>(info.AllocationBase), info.RegionSize, info.State};
}

// The view that holds address starts at first and ends at end (exclusive).
bool view_is(const unsigned char* address, const unsigned char* first, const unsigned char* end)
{
    const Region r = region(address);
    return r.state == MEM_COMMIT && r.allocation_base == reinterpret_cast<ULONG_PTR>(first) &&
           r.size == static_cast<SIZE_T>(end - address);
}

std::uint32_t& word(unsigned char* address) { return *reinterpret_cast<std::uint32_t*>(address); }

struct Shim {
    XMemVirtualAllocFn virtual_alloc;
    XMemAllocatePhysicalPagesFn allocate;
    XMemMapPhysicalPagesFn map;
    XMemFreePhysicalPagesFn release;
    StableCallFn stable_call;
    PageStatsFn page_stats;

    std::vector<SIZE_T> pages(SIZE_T n) const
    {
        std::vector<SIZE_T> tokens(n);
        SIZE_T got = n;
        if (!allocate(0x20400000, &got, tokens.data()) || got != n) tokens.clear();
        return tokens;
    }
    ULONGLONG live() const
    {
        ULONGLONG value = ~0ull;
        page_stats(&value);
        return value;
    }
};

// Maps tokens[order(i)] at base + i pages, one call each, and marks each page right after its call, before the next
// call merges it. Returns false when a call fails.
template <class Order>
bool map_one_by_one(const Shim& shim, unsigned char* base, std::vector<SIZE_T>& tokens, std::uint32_t tag, Order order)
{
    for (SIZE_T i = 0; i < tokens.size(); ++i) {
        unsigned char* const at = base + i * page;
        if (shim.map(at, 1, &tokens[order(i)]) != at) return false;
        word(at) = tag + static_cast<std::uint32_t>(i);
        word(at + page - 4) = static_cast<std::uint32_t>(order(i));
    }
    return true;
}

bool marked(unsigned char* base, SIZE_T n, std::uint32_t tag, SIZE_T skip = SIZE_T(-1))
{
    for (SIZE_T i = 0; i < n; ++i) {
        if (i == skip) continue;
        if (region(base + i * page).state != MEM_COMMIT || word(base + i * page) != tag + i) return false;
    }
    return true;
}

// joined (merging modes) or the per-page layout (control, default); then middle in `check`.
void single_page_calls(const Shim& shim, unsigned char* base, bool merges, bool middle)
{
    auto tokens = shim.pages(count);
    if (!check(tokens.size() == count, "allocate 32 pages")) return;
    if (!check(map_one_by_one(shim, base, tokens, 0x61000000u, [](SIZE_T i) { return i; }), "32 single-page calls")) return;
    if (merges) {
        if (check(view_is(base, base, base + count * page) && view_is(base + (count - 1) * page, base, base + count * page),
                  "joined: one view of 32 pages"))
            token = "merge: 1 view of 32 pages";
    } else {
        bool each = true;
        for (SIZE_T i = 0; i < count; ++i) each &= view_is(base + i * page, base + i * page, base + (i + 1) * page);
        if (check(each, "one view per call")) token = "no-merge control: 32 views";
    }
    check(marked(base, count, 0x61000000u), "joined: marks kept");
    if (middle) {
        check(shim.map(base + 10 * page, 1, nullptr) == base + 10 * page, "middle: unmap page 10");
        check(view_is(base, base, base + 10 * page) && view_is(base + 9 * page, base, base + 10 * page), "middle: view [0, 10)");
        check(view_is(base + 11 * page, base + 11 * page, base + count * page), "middle: view [11, 32)");
        check(region(base + 10 * page).state == MEM_RESERVE, "middle: page 10 reserved");
        check(marked(base, count, 0x61000000u, 10), "middle: neighbours keep their marks");
        check(shim.map(base, 10, nullptr) == base && shim.map(base + 11 * page, count - 11, nullptr) == base + 11 * page,
              "middle: unmap the rest");
    } else {
        check(shim.map(base, count, nullptr) == base, "unmap 32 pages");
    }
    check(shim.release(count, tokens.data()) != FALSE, "release 32 pages");
}

void reversed(const Shim& shim, unsigned char* base, unsigned char* scratch)
{
    auto tokens = shim.pages(count);
    if (!check(tokens.size() == count, "reversed: allocate")) return;
    // Mark every slot through one view first: slot k holds 0x62000000 + k.
    if (!check(shim.map(scratch, count, tokens.data()) == scratch, "reversed: map the slots in order")) return;
    for (SIZE_T k = 0; k < count; ++k) word(scratch + k * page) = 0x62000000u + static_cast<std::uint32_t>(k);
    check(shim.map(scratch, count, nullptr) == scratch, "reversed: unmap the slots");
    bool mapped = true;
    for (SIZE_T i = 0; i < count && mapped; ++i)
        mapped = shim.map(base + i * page, 1, &tokens[count - 1 - i]) == base + i * page;
    if (!check(mapped, "reversed: 32 single-page calls")) return;
    bool each = true, data = true;
    for (SIZE_T i = 0; i < count; ++i) {
        each &= view_is(base + i * page, base + i * page, base + (i + 1) * page);
        data &= region(base + i * page).state == MEM_COMMIT && word(base + i * page) == 0x62000000u + (count - 1 - i);
    }
    check(each, "reversed: 32 views");
    check(data, "reversed: page i shows slot 31 - i");
    check(shim.map(base, count, nullptr) == base && shim.release(count, tokens.data()), "reversed: unmap and release");
}

void window(const Shim& shim, unsigned char* base)
{
    auto tokens = shim.pages(9);
    if (!check(tokens.size() == 9 && shim.map(base, 8, tokens.data()) == base, "window: map a run of 8 pages")) return;
    for (SIZE_T i = 0; i < 8; ++i) word(base + i * page) = 0x63000000u + static_cast<std::uint32_t>(i);
    char hold[64], event_name[96];
    std::snprintf(hold, sizeof(hold), "x2w-cross-call-merge-%lu", GetCurrentProcessId());
    std::snprintf(event_name, sizeof(event_name), "%s.unmapped", hold);
    HANDLE unmapped = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::snprintf(event_name, sizeof(event_name), "%s.release", hold);
    HANDLE release = CreateEventA(nullptr, TRUE, FALSE, event_name);
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", hold);
    bool joined_call = false;
    std::thread mapper([&] { joined_call = shim.map(base + 8 * page, 1, &tokens[8]) == base + 8 * page; });
    const bool opened = WaitForSingleObject(unmapped, 10000) == WAIT_OBJECT_0;
    check(opened, "window: the merge held the window");
    unsigned char* const target = base + 2 * page;
    if (opened) check(region(target).state != MEM_COMMIT, "window: raw query inside the window not committed");
    std::atomic<bool> read_done{false}, query_done{false};
    std::uint32_t value = 0;
    DWORD state = 0;
    std::thread reader([&] {
        value = word(target);  // faults inside the window; retried by the shim after it
        read_done = true;
    });
    std::thread querier([&] {
        struct Query { unsigned char* target; DWORD state; } query{target, 0};
        shim.stable_call([](void* context) {
            auto& q = *static_cast<Query*>(context);
            q.state = region(q.target).state;
        }, &query);
        state = query.state;
        query_done = true;
    });
    Sleep(300);
    check(!read_done.load(), "window: the reader waits for the merge");
    check(!query_done.load(), "window: x2w_xmem_stable_call waits for the merge");
    SetEvent(release);
    reader.join();
    querier.join();
    mapper.join();
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", nullptr);
    check(joined_call, "window: the mapping call succeeded");
    check(value == 0x63000002u, "window: the reader read the mark");
    check(state == MEM_COMMIT, "window: x2w_xmem_stable_call answered committed");
    check(view_is(base, base, base + 9 * page), "window: one view of 9 pages");
    check(marked(base, 8, 0x63000000u), "window: marks kept");
    check(shim.map(base, 9, nullptr) == base && shim.release(9, tokens.data()), "window: unmap and release");
    CloseHandle(unmapped);
    CloseHandle(release);
}

void refused(const Shim& shim, unsigned char* base, const char* stage, const char* name)
{
    const ULONGLONG live_before = shim.live();
    auto tokens = shim.pages(9);
    if (!check(tokens.size() == 9 && shim.map(base, 8, tokens.data()) == base, name, "map a run of 8 pages")) return;
    for (SIZE_T i = 0; i < 8; ++i) word(base + i * page) = 0x64000000u + static_cast<std::uint32_t>(i);
    SetEnvironmentVariableA("X2W_XMEM_TEST_FAULT", stage);
    const bool mapped = shim.map(base + 8 * page, 1, &tokens[8]) == base + 8 * page;
    SetEnvironmentVariableA("X2W_XMEM_TEST_FAULT", nullptr);
    check(mapped, name, "the mapping call succeeded");
    // State first: a page without a view is not touched.
    bool committed = true;
    for (SIZE_T i = 0; i < 9; ++i) committed &= region(base + i * page).state == MEM_COMMIT;
    if (!check(committed, name, "all 9 pages committed")) return;
    check(view_is(base, base, base + 8 * page) && view_is(base + 7 * page, base, base + 8 * page), name,
          "the run is a view of 8 pages again");
    check(view_is(base + 8 * page, base + 8 * page, base + 9 * page), name, "the new page has a view of its own");
    check(marked(base, 8, 0x64000000u), name, "marks kept");
    word(base + 8 * page) = 0x64000008u;
    check(marked(base, 9, 0x64000000u), name, "the new page holds its write");
    check(shim.map(base, 9, nullptr) == base, name, "unmap 9 pages");
    check(shim.release(9, tokens.data()) != FALSE, name, "release 9 pages");
    check(shim.live() == live_before, name, "live page count back where it was");
}

// Named events for X2W_XMEM_TEST_HOLD_WINDOW: "<name>.unmapped" (set by the held call) and "<name>.release".
struct Hold {
    char name[64];
    HANDLE unmapped, release;
    explicit Hold(const char* stem)
    {
        std::snprintf(name, sizeof(name), "x2w-cross-call-merge-%s-%lu", stem, GetCurrentProcessId());
        char event_name[96];
        std::snprintf(event_name, sizeof(event_name), "%s.unmapped", name);
        unmapped = CreateEventA(nullptr, TRUE, FALSE, event_name);
        std::snprintf(event_name, sizeof(event_name), "%s.release", name);
        release = CreateEventA(nullptr, TRUE, FALSE, event_name);
    }
    ~Hold()
    {
        CloseHandle(unmapped);
        CloseHandle(release);
    }
};

// capped: 3 * cap single-page calls make three runs of cap pages (the cap of the shim, 64), not one. The unmap of a
// page in the middle run is held in its window: the pages that read as not committed there are the pages the break
// took apart, at most one run of cap pages; the runs before and after it stay mapped.
void capped(const Shim& shim, unsigned char* base)
{
    constexpr SIZE_T cap = 64, n = 3 * cap, hole = cap + cap / 2;
    auto tokens = shim.pages(n);
    if (!check(tokens.size() == n, "capped: allocate 192 pages")) return;
    if (!check(map_one_by_one(shim, base, tokens, 0x65000000u, [](SIZE_T i) { return i; }), "capped: 192 single-page calls"))
        return;
    bool runs = true;
    for (SIZE_T r = 0; r < 3; ++r)
        runs &= view_is(base + r * cap * page, base + r * cap * page, base + (r + 1) * cap * page) &&
                view_is(base + ((r + 1) * cap - 1) * page, base + r * cap * page, base + (r + 1) * cap * page);
    check(runs, "capped: three views of 64 pages");
    Hold hold("capped");
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", hold.name);
    bool unmapped_call = false;
    std::thread unmapper([&] { unmapped_call = shim.map(base + hole * page, 1, nullptr) == base + hole * page; });
    const bool opened = WaitForSingleObject(hold.unmapped, 10000) == WAIT_OBJECT_0;
    SIZE_T taken_apart = 0;
    bool outer_mapped = true;
    for (SIZE_T i = 0; opened && i < n; ++i) {
        const bool committed = region(base + i * page).state == MEM_COMMIT;
        taken_apart += !committed;
        if (i < cap || i >= 2 * cap) outer_mapped &= committed;
    }
    SetEvent(hold.release);
    unmapper.join();
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", nullptr);
    char detail[64];
    std::snprintf(detail, sizeof(detail), "(%zu pages)", static_cast<std::size_t>(taken_apart));
    check(opened, "capped: the unmap held the window");
    check(opened && taken_apart <= cap, "capped: the break took apart at most 64 pages", detail);
    check(opened && outer_mapped, "capped: the runs before and after stay mapped in the window");
    check(unmapped_call, "capped: the unmap succeeded");
    check(view_is(base + cap * page, base + cap * page, base + hole * page) &&
          view_is(base + (hole + 1) * page, base + (hole + 1) * page, base + 2 * cap * page), "capped: views [64, 96) and [97, 128)");
    check(marked(base, n, 0x65000000u, hole), "capped: marks kept");
    check(shim.map(base, hole, nullptr) == base && shim.map(base + (hole + 1) * page, n - hole - 1, nullptr) == base + (hole + 1) * page &&
          shim.release(n, tokens.data()), "capped: unmap and release");
}

// timing (not a ctest case; one row per R): for a run of R pages, the ms of the call that joins page R to a run of
// R - 1 (X2W_XMEM_TEST_MERGE_CAP=R), of the unmap of page R/2 (one break of the whole run) and, for comparison, of a
// single-page call that maps a view of its own (X2W_XMEM_TEST_NO_MERGE=1). Every byte of the run is written before
// the join and before the break, as a title's use of the pages would (a run that is mostly untouched unmaps much
// faster). Median of 5 (3 from 1024 pages). This process has a few
// views; a title process has tens of thousands, and under Wine each map costs more the more views there are.
double ms_since(const LARGE_INTEGER& from)
{
    LARGE_INTEGER now, frequency;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&frequency);
    return 1000.0 * static_cast<double>(now.QuadPart - from.QuadPart) / static_cast<double>(frequency.QuadPart);
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

void timing(const Shim& shim, unsigned char* base)
{
    std::printf("timing: R pages | join ms | break ms | own-view map ms (median of 5)\n");
    for (const SIZE_T r : {SIZE_T{16}, SIZE_T{32}, SIZE_T{64}, SIZE_T{128}, SIZE_T{256}, SIZE_T{512}, SIZE_T{1024},
                           SIZE_T{2048}, SIZE_T{4096}}) {
        std::vector<double> join, unmap, own;
        char cap[24];
        std::snprintf(cap, sizeof(cap), "%zu", static_cast<std::size_t>(r));
        for (int rep = 0; rep < (r >= 1024 ? 3 : 5); ++rep) {
            auto tokens = shim.pages(r + 1);
            if (!check(tokens.size() == r + 1 && shim.map(base, r - 1, tokens.data()) == base, "timing: map R - 1 pages")) return;
            std::memset(base, 0x5a, (r - 1) * page);
            SetEnvironmentVariableA("X2W_XMEM_TEST_MERGE_CAP", cap);
            LARGE_INTEGER t;
            QueryPerformanceCounter(&t);
            const bool joined = shim.map(base + (r - 1) * page, 1, &tokens[r - 1]) == base + (r - 1) * page;
            join.push_back(ms_since(t));
            SetEnvironmentVariableA("X2W_XMEM_TEST_MERGE_CAP", nullptr);
            check(joined && view_is(base, base, base + r * page), "timing: joined run of R pages");
            std::memset(base, 0x5b, r * page);
            QueryPerformanceCounter(&t);
            check(shim.map(base + (r / 2) * page, 1, nullptr) == base + (r / 2) * page, "timing: unmap page R/2");
            unmap.push_back(ms_since(t));
            SetEnvironmentVariableA("X2W_XMEM_TEST_NO_MERGE", "1");
            QueryPerformanceCounter(&t);
            const bool alone = shim.map(base + r * page, 1, &tokens[r]) == base + r * page;
            own.push_back(ms_since(t));
            SetEnvironmentVariableA("X2W_XMEM_TEST_NO_MERGE", nullptr);
            check(alone && view_is(base + r * page, base + r * page, base + (r + 1) * page), "timing: own view");
            check(shim.map(base, r / 2, nullptr) == base && shim.map(base + (r / 2 + 1) * page, r / 2, nullptr) &&
                  shim.release(r + 1, tokens.data()), "timing: unmap and release");
        }
        std::printf("timing: %4zu | %7.3f | %7.3f | %7.3f\n", static_cast<std::size_t>(r), median(join), median(unmap), median(own));
    }
}

void two_reservations(const Shim& shim)
{
    const auto virtual_alloc2 = reinterpret_cast<VirtualAlloc2Fn>(
        GetProcAddress(GetModuleHandleA("kernelbase.dll"), "VirtualAlloc2"));
    if (!check(virtual_alloc2 != nullptr, "two-reservations: VirtualAlloc2")) return;
    auto* const piece = static_cast<unsigned char*>(virtual_alloc2(GetCurrentProcess(), nullptr, 8 * page,
                                                                   MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0));
    if (!check(piece && VirtualFree(piece, 4 * page, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), "two-reservations: split a placeholder"))
        return;
    SetEnvironmentVariableA("X2W_XMEM_PERMISSIVE", "1");
    const bool recorded = shim.virtual_alloc(piece, 4 * page, 0x20406000, 0x21, PAGE_READWRITE) == piece &&
                          shim.virtual_alloc(piece + 4 * page, 4 * page, 0x20406000, 0x21, PAGE_READWRITE) == piece + 4 * page;
    SetEnvironmentVariableA("X2W_XMEM_PERMISSIVE", nullptr);
    if (!check(recorded, "two-reservations: record both pieces")) return;
    auto tokens = shim.pages(5);
    if (!check(tokens.size() == 5 && shim.map(piece, 4, tokens.data()) == piece &&
               shim.map(piece + 4 * page, 1, &tokens[4]) == piece + 4 * page, "two-reservations: map 4 + 1 pages"))
        return;
    check(view_is(piece, piece, piece + 4 * page), "two-reservations: the first piece's run stays 4 pages");
    check(view_is(piece + 4 * page, piece + 4 * page, piece + 5 * page), "two-reservations: the page has a view of its own");
    check(shim.map(piece, 4, nullptr) == piece && shim.map(piece + 4 * page, 1, nullptr) == piece + 4 * page &&
          shim.release(5, tokens.data()), "two-reservations: unmap and release");
    VirtualFree(piece, 0, MEM_RELEASE);
    VirtualFree(piece + 4 * page, 0, MEM_RELEASE);
}
}  // namespace

int main(int argc, char** argv)
{
    const char* const mode = argc == 3 ? argv[2] : "";
    const bool all = std::strcmp(mode, "check") == 0, control = std::strcmp(mode, "control") == 0,
               deployed = std::strcmp(mode, "deployed") == 0, plain = std::strcmp(mode, "default") == 0,
               timed = std::strcmp(mode, "timing") == 0, no_join = std::strcmp(mode, "no-join") == 0;
    if (!all && !control && !deployed && !plain && !timed && !no_join) {
        std::printf("usage: xmem_cross_call_merge_check <xmem.dll> check|control|deployed|default|no-join|timing\n");
        return 2;
    }
    // The mode is read by the first mapping call.
    SetEnvironmentVariableA("X2W_XMEM_SHARED_VIEWS", plain ? nullptr : "1");
    SetEnvironmentVariableA("X2W_XMEM_JOIN_CALLS", no_join ? nullptr : "1");
    const HMODULE xmem = LoadLibraryA(argv[1]);
    if (!xmem) {
        std::printf("check_failed: load (error %lu)\n", GetLastError());
        return 1;
    }
    const Shim shim{reinterpret_cast<XMemVirtualAllocFn>(GetProcAddress(xmem, "XMemVirtualAlloc")),
                    reinterpret_cast<XMemAllocatePhysicalPagesFn>(GetProcAddress(xmem, "XMemAllocatePhysicalPages")),
                    reinterpret_cast<XMemMapPhysicalPagesFn>(GetProcAddress(xmem, "XMemMapPhysicalPages")),
                    reinterpret_cast<XMemFreePhysicalPagesFn>(GetProcAddress(xmem, "XMemFreePhysicalPages")),
                    reinterpret_cast<StableCallFn>(GetProcAddress(xmem, "x2w_xmem_stable_call")),
                    reinterpret_cast<PageStatsFn>(GetProcAddress(xmem, "x2w_xmem_page_stats"))};
    if (!shim.virtual_alloc || !shim.allocate || !shim.map || !shim.release || !shim.stable_call || !shim.page_stats) {
        std::printf("check_failed: exports\n");
        return 1;
    }
    auto* const whole = static_cast<unsigned char*>(shim.virtual_alloc(nullptr, SIZE_T{16} << 30, 0x20402000, 0x21, PAGE_READWRITE));
    if (!whole) {
        std::printf("check_failed: startup reservation (error %lu)\n", GetLastError());
        return 1;
    }
    unsigned char* const base = whole + page;
    if (timed) {
        timing(shim, base);
        VirtualFree(whole, 0, MEM_RELEASE);
        std::printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
        return failures ? 1 : 0;
    }
    single_page_calls(shim, base, all || deployed, all);
    if (no_join && token) token = "no-join: 32 views";
    if (all) {
        reversed(shim, base + 64 * page, base + 128 * page);
        window(shim, base + 192 * page);
        refused(shim, base + 256 * page, "4:0", "refused-4");
        refused(shim, base + 320 * page, "5:0", "refused-5");
        two_reservations(shim);
        capped(shim, base + 512 * page);
    }
    VirtualFree(whole, 0, MEM_RELEASE);
    if (!failures && token) std::printf("%s\n", token);
    std::printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
