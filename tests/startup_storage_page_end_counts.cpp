// The DirectStorage stand-in counts page-destination requests (option bit 21) by where page_offset +
// destination_size ends against the first 64 KiB page, each split into completed and failed, and the
// heartbeat's third line reports those counts (#1198). Sibling of startup_storage_failure_counts.
//
// The test compiles the stand-in's source into itself, so it reads the counters and formats the heartbeat
// line through the stand-in's own code. A page destination is written through x2w_xmem_write_physical, so
// the test loads the xmem shim given on the command line (as xmem.dll, which the stand-in looks up) and
// takes its page tokens from XMemAllocatePhysicalPages under X2W_XMEM_PERMISSIVE. No device is used.
//
// One Submit carries, each request followed by its status entry and the Submit closed by a fence, all from
// memory sources (no chunked-stream cache applies to them):
//   inside the first page (end < 0x10000):
//     completed: raw 0x1000 at offset 0x100; a zlib stream of 0x800 bytes into 0xFFFF bytes at offset 0;
//     failed:    destination size 0 at offset 0x20 (capacity); a page array made unreadable before Submit
//                (fails in run_entry, before execute_request); an unknown option bit;
//   at the first page's end (end == 0x10000):
//     completed: raw 0x8000 at offset 0x8000 (+0x10 carries high bits, which are ignored); raw 1 byte at
//                offset 0xFFFF; raw 0x10000 at offset 0;
//     failed:    a zlib stream of 0x8010 bytes into 0x8000 bytes at offset 0x8000 (capacity); a raw read
//                whose page token has a low bit set (the page write fails);
//   past it (end > 0x10000):
//     completed: raw 0x2000 at offset 0xF000 over two pages;
//     failed:    the same with the second token 0 (missing grant); a zlib stream with a wrong Adler-32;
//                a destination of 2^32 - 1 bytes at offset 1 (its page array cannot be read in full at
//                Enqueue, so it runs without tokens; a 32-bit end would wrap to 0, inside);
//   not counted: a page request with source size 0 ("empty", neither completed nor failed); a raw
//                linear-destination request that completes and one that fails.
// Checked: every status, the exact count of each end and outcome, that the page-request failures are the
// failed total less the one linear failure, that completed page requests are done + inflated less the one
// linear completion, the exact heartbeat line, the logged outcome of every failure (so each failed where
// the plan says), and the end of edge ranges near 2^32 (no wrap).
// --mutate one-bucket: the expectations of a counter that books every page request at the page end (a
// negative control; it must fail).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "startup_stubs/xcrdapi.cpp"

namespace {

constexpr DWORD completion_timeout_ms = 15000;
constexpr unsigned end_count = static_cast<unsigned>(PageEnd::count);
constexpr u64 page_destination = 1ull << 21;
constexpr u64 memory_source = 0x400000;

using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using EnqueueSignalFn = void(__stdcall*)(void*, void*, std::uint64_t);
using SubmitFn = void(__stdcall*)(void*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
using GetHResultFn = long(__stdcall*)(void*, unsigned);

// The status of a refused request: a too-small destination reports its own code, the others E_FAIL.
long failed_hresult(const char* why)
{
    return why && std::strcmp(why, "destination capacity exceeded") == 0 ? HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)
                                                                          : E_FAIL;
}
using AllocatePhysicalFn = BOOL(WINAPI*)(ULONG_PTR, PULONG_PTR, PULONG_PTR);
using FreePhysicalFn = BOOL(WINAPI*)(ULONG_PTR, PULONG_PTR);

struct Fence {
    void** vtable;
    volatile LONG signaled;
    static long __stdcall signal(void* self, std::uint64_t)
    {
        InterlockedExchange(&static_cast<Fence*>(self)->signaled, 1);
        return 0;
    }
};

std::vector<unsigned char> stored_zlib(const std::vector<unsigned char>& data, bool damage_checksum)
{
    std::vector<unsigned char> out{0x78, 0x01, 0x01};  // one final stored block
    const auto size = static_cast<std::uint16_t>(data.size());
    out.push_back(static_cast<unsigned char>(size));
    out.push_back(static_cast<unsigned char>(size >> 8));
    out.push_back(static_cast<unsigned char>(~size));
    out.push_back(static_cast<unsigned char>((~size) >> 8));
    out.insert(out.end(), data.begin(), data.end());
    std::uint32_t a = 1, b = 0;
    for (unsigned char byte : data) {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    std::uint32_t sum = (b << 16) | a;
    if (damage_checksum) sum ^= 1;
    for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<unsigned char>(sum >> shift));
    return out;
}

std::vector<unsigned char> pattern(std::size_t size, unsigned seed)
{
    std::vector<unsigned char> bytes(size);
    for (std::size_t i = 0; i < size; ++i) bytes[i] = static_cast<unsigned char>((i * 7 + (i >> 8) + seed) ^ 0x3c);
    return bytes;
}

std::string read_text(const wchar_t* path)
{
    std::string text;
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return text;
    char buffer[4096];
    DWORD got = 0;
    while (ReadFile(file, buffer, sizeof buffer, &got, nullptr) && got) text.append(buffer, got);
    CloseHandle(file);
    return text;
}

std::size_t occurrences(const std::string& text, const std::string& part)
{
    std::size_t count = 0;
    for (std::size_t at = 0; (at = text.find(part, at)) != std::string::npos; at += part.size()) ++count;
    return count;
}

bool expect(bool condition, const char* what)
{
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", what);
    return condition;
}

const char* end_name(unsigned end)
{
    static const char* const names[end_count] = {"inside", "at-page-end", "past"};
    return end < end_count ? names[end] : "?";
}

// Ends a 32-bit sum would get wrong: page_offset + destination_size wraps past 2^32.
bool check_edge_ends()
{
    struct Case { unsigned offset, size; PageEnd end; const char* what; };
    const Case cases[] = {
        {0, 0, PageEnd::inside, "offset 0, size 0"},
        {0xFFFF, 0, PageEnd::inside, "offset 0xFFFF, size 0"},
        {0, 0xFFFF, PageEnd::inside, "one byte short of the page end"},
        {0xFFFF, 1, PageEnd::at_page_end, "last byte of the page"},
        {0, 0x10000, PageEnd::at_page_end, "one whole page"},
        {0x8000, 0x8001, PageEnd::past, "one byte past the page end"},
        {0, 0xFFFFFFFF, PageEnd::past, "size 2^32 - 1"},
        {1, 0xFFFFFFFF, PageEnd::past, "offset 1, size 2^32 - 1 (a 32-bit sum wraps to 0)"},
        {0xFFFF, 0xFFFFFFFF, PageEnd::past, "offset 0xFFFF, size 2^32 - 1 (a 32-bit sum wraps to 0xFFFE)"},
        {0x10, 0xFFFFFFF0, PageEnd::past, "a 32-bit sum wraps to exactly 0"},
    };
    bool ok = true;
    for (const Case& c : cases) {
        char what[192];
        const PageEnd got = page_end(c.offset, c.size);
        std::snprintf(what, sizeof what, "edge end (%s) is %s, expected %s", c.what, end_name(static_cast<unsigned>(got)),
                      end_name(static_cast<unsigned>(c.end)));
        ok = expect(got == c.end, what) && ok;
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv)
{
    const bool one_bucket = argc == 4 && std::strcmp(argv[2], "--mutate") == 0 && std::strcmp(argv[3], "one-bucket") == 0;
    if (argc != 2 && !one_bucket) {
        std::fprintf(stderr, "usage: startup_storage_page_end_counts <xmem.dll> [--mutate one-bucket]\n");
        return 2;
    }
    wchar_t temp_dir[MAX_PATH] = {}, log_path[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, temp_dir);
    std::swprintf(log_path, MAX_PATH, L"%lsx2w_page_end_counts_%lu.log", temp_dir, GetCurrentProcessId());
    DeleteFileW(log_path);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path);
    SetEnvironmentVariableW(L"X2W_XMEM_PERMISSIVE", L"1");

    // The stand-in finds x2w_xmem_write_physical through GetModuleHandleW(L"xmem.dll") on its first page
    // write, so the shim is loaded before any request runs.
    const HMODULE xmem = LoadLibraryA(argv[1]);
    if (!xmem || GetModuleHandleW(L"xmem.dll") != xmem) {
        std::fprintf(stderr, "cannot load %s as xmem.dll\n", argv[1]);
        return 1;
    }
    const auto allocate = reinterpret_cast<AllocatePhysicalFn>(GetProcAddress(xmem, "XMemAllocatePhysicalPages"));
    const auto release = reinterpret_cast<FreePhysicalFn>(GetProcAddress(xmem, "XMemFreePhysicalPages"));
    ULONG_PTR page_count = 2, pages[2] = {};
    if (!allocate || !release || !allocate(0x20400000ull, &page_count, pages) || page_count != 2 || !pages[0] ||
        !pages[1] || (pages[0] & 0xF) || (pages[1] & 0xF)) {
        std::fprintf(stderr, "cannot allocate two physical pages\n");
        return 1;
    }

    void* object = nullptr;
    if (DStorageGetFactory(nullptr, &object) != 0 || !object) {
        std::fprintf(stderr, "cannot get the factory\n");
        return 1;
    }
    auto* factory = static_cast<Object*>(object);

    // Token arrays. The unreadable one sits in a page of its own, made no-access after Enqueue.
    const ULONG_PTR one_page[1] = {pages[0]};
    const ULONG_PTR two_pages[2] = {pages[0], pages[1]};
    const ULONG_PTR missing_second[2] = {pages[0], 0};
    const ULONG_PTR low_bit[1] = {pages[0] | 1};
    auto* unreadable = static_cast<ULONG_PTR*>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!unreadable) {
        std::fprintf(stderr, "cannot allocate the page array\n");
        return 1;
    }
    unreadable[0] = pages[0];
    // A token at the end of a committed page followed by a reserved one: an array of more than one entry
    // cannot be read, so a destination of 2^32 - 1 bytes captures no tokens at Enqueue.
    auto* edge = static_cast<unsigned char*>(VirtualAlloc(nullptr, 2 * 4096, MEM_RESERVE, PAGE_NOACCESS));
    if (!edge || !VirtualAlloc(edge, 4096, MEM_COMMIT, PAGE_READWRITE)) {
        std::fprintf(stderr, "cannot allocate the edge page array\n");
        return 1;
    }
    auto* edge_token = reinterpret_cast<ULONG_PTR*>(edge + 4096 - sizeof(ULONG_PTR));
    *edge_token = pages[0];

    const auto raw_1000 = pattern(0x1000, 1);
    const auto raw_8000 = pattern(0x8000, 2);
    const auto raw_1 = pattern(1, 3);
    const auto raw_10000 = pattern(0x10000, 4);
    const auto raw_2000 = pattern(0x2000, 5);
    const auto raw_10 = pattern(0x10, 6);
    const auto zlib_800 = stored_zlib(pattern(0x800, 7), false);
    const auto zlib_8010 = stored_zlib(pattern(0x8010, 8), false);
    const auto zlib_damaged = stored_zlib(pattern(0x100, 9), true);
    std::vector<unsigned char> linear(0x1000, 0xA5);

    enum Outcome { completed, failed, not_counted };
    struct Plan {
        u64 options;
        const ULONG_PTR* tokens;  // nullptr: a linear destination (the linear buffer)
        u64 offset_field;         // the u64 at +0x10; its low 16 bits are the page offset
        unsigned destination_size;
        const std::vector<unsigned char>* source;
        unsigned source_size;
        unsigned uncompressed_size;
        PageEnd end;
        Outcome outcome;
        const char* why = nullptr;  // the logged outcome of a failure
    };
    const u64 raw = page_destination | memory_source;
    const u64 zlib = raw | 1;
    using E = PageEnd;
    const Plan plan[] = {
        {raw, one_page, 0x100, 0x1000, &raw_1000, 0x1000, 0, E::inside, completed},
        {zlib, one_page, 0, 0xFFFF, &zlib_800, static_cast<unsigned>(zlib_800.size()), 0x800, E::inside, completed},
        {raw, one_page, 0x20, 0, &raw_10, 0x10, 0, E::inside, failed, "destination capacity exceeded"},
        {raw, unreadable, 0x100, 0x1000, &raw_1000, 0x1000, 0, E::inside, failed, "failed"},
        {raw | (1ull << 40), one_page, 0x10, 0x10, &raw_10, 0x10, 0, E::inside, failed, "unknown option bits, not done"},
        {raw, one_page, 0x1234567800008000ull, 0x8000, &raw_8000, 0x8000, 0, E::at_page_end, completed},
        {raw, one_page, 0xFFFF, 1, &raw_1, 1, 0, E::at_page_end, completed},
        {raw, one_page, 0, 0x10000, &raw_10000, 0x10000, 0, E::at_page_end, completed},
        {zlib, one_page, 0x8000, 0x8000, &zlib_8010, static_cast<unsigned>(zlib_8010.size()), 0x8010, E::at_page_end, failed, "destination capacity exceeded"},
        {raw, low_bit, 0x4000, 0xC000, &raw_8000, 0x8000, 0, E::at_page_end, failed, "page destination write failed"},
        {raw, two_pages, 0xF000, 0x2000, &raw_2000, 0x2000, 0, E::past, completed},
        {raw, missing_second, 0xF000, 0x2000, &raw_2000, 0x2000, 0, E::past, failed, "missing page grant token"},
        {zlib, two_pages, 0xFFF0, 0x100, &zlib_damaged, static_cast<unsigned>(zlib_damaged.size()), 0x100, E::past, failed, "zlib failed"},
        {raw, edge_token, 1, 0xFFFFFFFF, &raw_10, 0x10, 0, E::past, failed, "missing page tokens"},
        {raw, one_page, 0, 0x1000, &raw_1000, 0, 0, E::inside, not_counted},
        {memory_source, nullptr, 0, 0x1000, &raw_1000, 0x1000, 0, E::inside, not_counted},
        {memory_source, nullptr, 0, 0x800, &raw_1000, 0x1000, 0, E::inside, not_counted, "destination capacity exceeded"},
    };
    constexpr unsigned request_count = sizeof(plan) / sizeof(plan[0]);
    // The status each request reports: the linear capacity failure is the last plan entry.
    auto valid = [&](unsigned i) { return plan[i].outcome == completed || (plan[i].outcome == not_counted && i + 1 < request_count); };

    void *status_ptr = nullptr, *queue_ptr = nullptr;
    unsigned char description[0x28] = {};
    // Room for every entry, so the queue does not submit itself before the page array is made unreadable.
    *reinterpret_cast<std::uint16_t*>(description + 8) = 4 * request_count;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, request_count, "PageEnds", nullptr, &status_ptr) != 0 ||
        reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue_ptr) != 0) {
        std::fprintf(stderr, "cannot create the queue\n");
        return 1;
    }
    auto* status = static_cast<Object*>(status_ptr);
    auto* queue = static_cast<Object*>(queue_ptr);
    for (unsigned i = 0; i < request_count; ++i) {
        alignas(8) unsigned char request[0x48] = {};
        *reinterpret_cast<u64*>(request + 0x00) = plan[i].options;
        *reinterpret_cast<const void**>(request + 0x08) =
            plan[i].tokens ? static_cast<const void*>(plan[i].tokens) : static_cast<const void*>(linear.data());
        *reinterpret_cast<u64*>(request + 0x10) = plan[i].offset_field;
        *reinterpret_cast<unsigned*>(request + 0x18) = plan[i].destination_size;
        *reinterpret_cast<const unsigned char**>(request + 0x20) = plan[i].source->data();
        *reinterpret_cast<unsigned*>(request + 0x30) = plan[i].source_size;
        *reinterpret_cast<unsigned*>(request + 0x38) = plan[i].uncompressed_size;
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, i);
    }
    DWORD old_protection = 0;
    if (!VirtualProtect(unreadable, 4096, PAGE_NOACCESS, &old_protection)) {
        std::fprintf(stderr, "cannot make the page array unreadable\n");
        return 1;
    }
    void* fence_vtable[11] = {};
    fence_vtable[10] = reinterpret_cast<void*>(&Fence::signal);
    Fence fence{fence_vtable, 0};
    reinterpret_cast<EnqueueSignalFn>(queue->vtable[5])(queue, &fence, 1);
    reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
    const DWORD start = GetTickCount();
    while (!fence.signaled && GetTickCount() - start < completion_timeout_ms) Sleep(2);
    bool ok = expect(fence.signaled, "the fence after the requests was never signaled");
    if (!ok) return 1;

    PageEndCounts expected = {};
    LONG page_failed = 0, page_completed = 0;
    for (unsigned i = 0; i < request_count; ++i) {
        char what[128];
        const bool done = reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, i);
        const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, i);
        std::snprintf(what, sizeof what, "request %u status is %s (complete %d), got 0x%08lx", i, valid(i) ? "S_OK" : "the failure's HRESULT",
                      done ? 1 : 0, static_cast<unsigned long>(hr));
        ok = expect(done && hr == (valid(i) ? S_OK : failed_hresult(plan[i].why)), what) && ok;
        if (plan[i].outcome == not_counted) continue;
        ++expected[static_cast<unsigned>(plan[i].end)][plan[i].outcome == failed ? 1 : 0];
        ++(plan[i].outcome == failed ? page_failed : page_completed);
    }
    if (one_bucket) {
        for (auto& end : expected) end[0] = end[1] = 0;
        expected[static_cast<unsigned>(E::at_page_end)][0] = page_completed;
        expected[static_cast<unsigned>(E::at_page_end)][1] = page_failed;
    }

    PageEndCounts counts;
    sample_page_end_counts(counts);
    for (unsigned end = 0; end < end_count; ++end) {
        for (unsigned f = 0; f < 2; ++f) {
            char what[128];
            std::snprintf(what, sizeof what, "%s-%s counted %ld, expected %ld", end_name(end), f ? "failed" : "completed",
                          counts[end][f], expected[end][f]);
            ok = expect(counts[end][f] == expected[end][f], what) && ok;
        }
    }
    // Cross-checks against the totals: one linear request completed and one failed.
    LONG counted_failed = 0, counted_completed = 0;
    for (unsigned end = 0; end < end_count; ++end) {
        counted_completed += counts[end][0];
        counted_failed += counts[end][1];
    }
    char what[192];
    std::snprintf(what, sizeof what, "page failures %ld, failed total %ld less 1 linear failure, expected %ld", counted_failed,
                  requests_failed, page_failed);
    ok = expect(counted_failed == requests_failed - 1 && counted_failed == page_failed, what) && ok;
    std::snprintf(what, sizeof what, "completed page requests %ld, done %ld + inflated %ld less 1 linear, expected %ld",
                  counted_completed, requests_done, requests_compressed, page_completed);
    ok = expect(counted_completed == requests_done + requests_compressed - 1 && counted_completed == page_completed, what) && ok;
    std::snprintf(what, sizeof what, "done=%ld inflated=%ld failed=%ld, expected 6, 1 and 9", requests_done, requests_compressed,
                  requests_failed);
    ok = expect(requests_done == 6 && requests_compressed == 1 && requests_failed == 9, what) && ok;

    // The heartbeat's line, through the stand-in's formatter, against text written out here. A request
    // counted after the sample does not change the line formatted from it.
    char line[384], want[384];
    alignas(8) unsigned char late[0x48] = {};
    *reinterpret_cast<u64*>(late) = raw;
    count_page_request(late, true);
    format_page_ends(line, sizeof line, 30, counts);
    std::snprintf(want, sizeof want,
                  "x2w-dstorage heartbeat 30 s: page requests by end against the first 64 KiB page inside-completed=%ld "
                  "inside-failed=%ld at-page-end-completed=%ld at-page-end-failed=%ld past-completed=%ld past-failed=%ld",
                  expected[0][0], expected[0][1], expected[1][0], expected[1][1], expected[2][0], expected[2][1]);
    if (!expect(std::strcmp(line, want) == 0, "heartbeat line differs")) {
        std::fprintf(stderr, "  got:  %s\n  want: %s\n", line, want);
        ok = false;
    }
    ok = expect(std::strstr(line, "inflated=") == nullptr && std::strstr(line, "options=") == nullptr &&
                    std::strstr(line, "commit peak=") == nullptr,
                "the line carries text the startup tools match") && ok;

    // Each failure took the path the plan names: its outcome is logged once per request (every failure is
    // logged, the first twelve of them at least), and the per-request text is unchanged.
    const std::string log_text = read_text(log_path);
    for (unsigned i = 0; i < request_count; ++i) {
        if (!plan[i].why) continue;
        bool first = true;
        std::size_t want_count = 0;
        for (unsigned k = 0; k < request_count; ++k) {
            if (!plan[k].why || std::strcmp(plan[k].why, plan[i].why) != 0) continue;
            first = first && k >= i;
            ++want_count;
        }
        if (!first) continue;
        const std::size_t got_count = occurrences(log_text, std::string("-> ") + plan[i].why + "\r\n");
        std::snprintf(what, sizeof what, "outcome \"%s\" logged %zu times, expected %zu", plan[i].why, got_count, want_count);
        ok = expect(got_count == want_count, what) && ok;
    }

    ok = check_edge_ends() && ok;

    VirtualFree(unreadable, 0, MEM_RELEASE);
    VirtualFree(edge, 0, MEM_RELEASE);
    release(2, pages);
    DeleteFileW(log_path);
    if (!ok) return 1;
    std::printf("%s\n", line);
    std::puts("each page-destination request is counted once by where it ends, as completed or failed");
    return 0;
}
