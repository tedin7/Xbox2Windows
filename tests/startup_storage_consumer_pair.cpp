// One refused page-destination request tracked end to end, next to a successful one of the same shape
// (#1198): enqueue, completion status, fence, the first consumer read of its destination through the
// title-visible view of the pages, and their release.
//
// The stand-in DLL (argv[1]) runs the requests. Page destinations are written through
// x2w_xmem_write_physical, so the xmem shim (argv[2]) is loaded first, as xmem.dll, under
// X2W_XMEM_PERMISSIVE. The test is the consumer: it takes page tokens from XMemAllocatePhysicalPages,
// reserves a view with XMemVirtualAlloc and maps the pages into it with XMemMapPhysicalPages, which is
// how a title reads its physical pages. Every read below goes through that view.
//
// One Submit carries two zlib requests from memory sources with the same shape: page_offset 0xF000 and
// destination_size 0x1000, so each ends exactly at the end of its single 64 KiB page. Each is followed by
// its status entry and an EnqueueSignal fence.
//   S: a stream that inflates to exactly 0x1000 bytes (completes);
//   R: a stream that inflates to 0x1800 bytes (refused as "destination capacity exceeded").
// Before Submit each page holds a sentinel pattern of its own; the sentinels differ from each other and
// from each stream's bytes at every offset of the destination, so a sentinel byte is never a data byte.
// Checked:
//   - status: S is S_OK, R is ERROR_INSUFFICIENT_BUFFER;
//   - fences: each runs after its own status entry and before the next request's, in queue order;
//   - consumer read at each fence (in the fence callback) and again on the test thread after both:
//     S's page shows its sentinel below 0xF000 and the exact inflated bytes from 0xF000; every byte of
//     R's page is still R's sentinel, and no byte of R's destination equals R's stream prefix;
//   - the stand-in's log names "inflated" for S and "destination capacity exceeded" for R;
//   - release: the pages are unmapped from the view and freed (both calls succeed and the shim's count of
//     live page backings drops by two), then one page is allocated again and mapped at the same address;
//     it reads zero-filled (nothing of S or R carries over), is given a new sentinel, and a follow-up
//     request of S's shape on the same queue completes with the exact bytes at its fence.
// --mutate accept-refused: a consumer that takes R's destination as data, i.e. expects R's stream prefix
// there instead of R's sentinel (a negative control; it must fail).
// Usage: startup_storage_consumer_pair <XCRDAPI.dll> <xmem.dll> [--mutate accept-refused]
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
#include <cwchar>
#include <string>
#include <vector>

namespace {

using u64 = std::uint64_t;
constexpr DWORD completion_timeout_ms = 15000;
constexpr std::size_t page_bytes = 0x10000;
constexpr unsigned page_offset = 0xF000;
constexpr unsigned destination_size = 0x1000;  // page_offset + destination_size == 0x10000
constexpr unsigned refused_inflated_size = 0x1800;
constexpr unsigned slot_count = 3;  // S, R, follow-up
constexpr u64 page_destination = 1ull << 21;
constexpr u64 memory_source = 0x400000;
constexpr u64 zlib_page_request = 1 | page_destination | memory_source;

struct Object { void** vtable; };
using GetFactoryFn = long(__stdcall*)(const void*, void**);
using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using EnqueueSignalFn = void(__stdcall*)(void*, void*, u64);
using SubmitFn = void(__stdcall*)(void*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
using GetHResultFn = long(__stdcall*)(void*, unsigned);
using AllocatePhysicalFn = BOOL(WINAPI*)(ULONG_PTR, PULONG_PTR, PULONG_PTR);
using FreePhysicalFn = BOOL(WINAPI*)(ULONG_PTR, PULONG_PTR);
using VirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using MapPhysicalFn = void*(WINAPI*)(void*, ULONG_PTR, const ULONG_PTR*);
using PageStatsFn = BOOL(WINAPI*)(ULONGLONG*);

// Notes, when signaled, what a consumer sees: every status entry and the bytes of the mapped view.
struct Fence {
    void** vtable;
    Object* status;
    const unsigned char* view;
    std::size_t view_bytes;
    std::vector<unsigned char> seen;  // the view at signal time; sized before Submit
    bool complete_at_signal[slot_count];
    long hr_at_signal[slot_count];
    u64 value;
    LONG position;
    volatile LONG signaled;
    static volatile LONG order;
    static long __stdcall signal(void* self, u64 value)
    {
        auto* fence = static_cast<Fence*>(self);
        for (unsigned i = 0; i < slot_count; ++i) {
            fence->complete_at_signal[i] = reinterpret_cast<IsCompleteFn>(fence->status->vtable[3])(fence->status, i);
            fence->hr_at_signal[i] = reinterpret_cast<GetHResultFn>(fence->status->vtable[4])(fence->status, i);
        }
        std::memcpy(fence->seen.data(), fence->view, fence->view_bytes);
        fence->value = value;
        fence->position = InterlockedIncrement(&order);
        InterlockedExchange(&fence->signaled, 1);
        return 0;
    }
};
volatile LONG Fence::order = 0;

std::vector<unsigned char> stored_zlib(const std::vector<unsigned char>& data)
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
    const std::uint32_t adler = (b << 16) | a;
    for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<unsigned char>(adler >> shift));
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

bool expect(bool condition, const char* what)
{
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", what);
    return condition;
}

// Offsets in [from, from + size) where page[from + k] != want[k]; the first one in *first.
std::size_t differing(const unsigned char* page, std::size_t from, const unsigned char* want, std::size_t size,
                      std::size_t* first)
{
    std::size_t count = 0;
    *first = SIZE_MAX;
    for (std::size_t k = 0; k < size; ++k) {
        if (page[from + k] == want[k]) continue;
        if (!count) *first = from + k;
        ++count;
    }
    return count;
}

// A successful request's page: its sentinel below page_offset, the exact inflated bytes from there.
bool check_published(const char* when, const char* name, const unsigned char* page, const std::vector<unsigned char>& sentinel,
                     const std::vector<unsigned char>& data)
{
    char what[224];
    std::size_t first = 0;
    std::size_t bad = differing(page, 0, sentinel.data(), page_offset, &first);
    std::snprintf(what, sizeof what, "%s: %s page below 0x%x changed in %zu byte(s), first at 0x%zx", when, name, page_offset,
                  bad, first);
    bool ok = expect(bad == 0, what);
    bad = differing(page, page_offset, data.data(), destination_size, &first);
    std::snprintf(what, sizeof what, "%s: %s destination differs from the inflated bytes in %zu byte(s), first at 0x%zx", when,
                  name, bad, first);
    return expect(bad == 0, what) && ok;
}

// The refused request's page: every byte is still its sentinel, and no byte of the destination is the
// stream's byte at that position. With accept_refused the consumer takes the destination as data instead
// and expects the stream prefix there.
bool check_refused(const char* when, const unsigned char* page, const std::vector<unsigned char>& sentinel,
                   const std::vector<unsigned char>& stream_bytes, bool accept_refused)
{
    char what[224];
    std::size_t first = 0;
    if (accept_refused) {
        const std::size_t bad = differing(page, page_offset, stream_bytes.data(), destination_size, &first);
        std::snprintf(what, sizeof what,
                      "%s: R destination taken as data differs from the stream prefix in %zu byte(s), first at 0x%zx", when,
                      bad, first);
        return expect(bad == 0, what);
    }
    std::size_t bad = differing(page, 0, sentinel.data(), page_bytes, &first);
    std::snprintf(what, sizeof what, "%s: R page is not its sentinel in %zu byte(s), first at 0x%zx", when, bad, first);
    bool ok = expect(bad == 0, what);
    std::size_t prefix = 0;
    for (std::size_t k = 0; k < destination_size; ++k) prefix += page[page_offset + k] == stream_bytes[k];
    std::snprintf(what, sizeof what, "%s: R destination holds %zu byte(s) of its stream prefix", when, prefix);
    return expect(prefix == 0, what) && ok;
}

// The text the stand-in logs after "-> " for the request with this memory source.
std::string logged_outcome(const std::string& log_text, const void* source)
{
    char key[64];
    std::snprintf(key, sizeof key, "source=0x%llx ", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(source)));
    const std::size_t at = log_text.find(key);
    if (at == std::string::npos) return "(not logged)";
    const std::size_t arrow = log_text.find("-> ", at);
    const std::size_t end = log_text.find_first_of("\r\n", at);
    if (arrow == std::string::npos || (end != std::string::npos && arrow > end)) return "(not logged)";
    return log_text.substr(arrow + 3, end == std::string::npos ? std::string::npos : end - arrow - 3);
}

ULONGLONG live_sections(PageStatsFn stats)
{
    ULONGLONG live = 0;
    return stats(&live) ? live : ~0ull;
}

}  // namespace

int main(int argc, char** argv)
{
    const bool accept_refused = argc == 5 && std::strcmp(argv[3], "--mutate") == 0 && std::strcmp(argv[4], "accept-refused") == 0;
    if (argc != 3 && !accept_refused) {
        std::fprintf(stderr, "usage: startup_storage_consumer_pair <XCRDAPI.dll> <xmem.dll> [--mutate accept-refused]\n");
        return 2;
    }
    wchar_t temp_dir[MAX_PATH] = {}, log_path[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, temp_dir);
    std::swprintf(log_path, MAX_PATH, L"%lsx2w_consumer_pair_%lu.log", temp_dir, GetCurrentProcessId());
    DeleteFileW(log_path);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path);
    SetEnvironmentVariableW(L"X2W_XMEM_PERMISSIVE", L"1");

    // The stand-in finds x2w_xmem_write_physical through GetModuleHandleW(L"xmem.dll") on its first page
    // write, so the shim is loaded before any request runs.
    const HMODULE xmem = LoadLibraryA(argv[2]);
    if (!xmem || GetModuleHandleW(L"xmem.dll") != xmem) {
        std::fprintf(stderr, "cannot load %s as xmem.dll\n", argv[2]);
        return 1;
    }
    const auto allocate = reinterpret_cast<AllocatePhysicalFn>(GetProcAddress(xmem, "XMemAllocatePhysicalPages"));
    const auto release = reinterpret_cast<FreePhysicalFn>(GetProcAddress(xmem, "XMemFreePhysicalPages"));
    const auto reserve = reinterpret_cast<VirtualAllocFn>(GetProcAddress(xmem, "XMemVirtualAlloc"));
    const auto map = reinterpret_cast<MapPhysicalFn>(GetProcAddress(xmem, "XMemMapPhysicalPages"));
    const auto stats = reinterpret_cast<PageStatsFn>(GetProcAddress(xmem, "x2w_xmem_page_stats"));
    if (!allocate || !release || !reserve || !map || !stats) {
        std::fprintf(stderr, "the xmem shim lacks an export this test uses\n");
        return 1;
    }
    const HMODULE library = LoadLibraryA(argv[1]);
    auto* get_factory = library ? reinterpret_cast<GetFactoryFn>(GetProcAddress(library, "DStorageGetFactory")) : nullptr;
    void* object = nullptr;
    if (!get_factory || get_factory(nullptr, &object) != 0 || !object) {
        std::fprintf(stderr, "cannot load the DirectStorage stand-in %s\n", argv[1]);
        return 1;
    }
    auto* factory = static_cast<Object*>(object);

    // Inputs. The sentinels differ from each other and from the data at every byte of the destination.
    const auto sentinel_s = pattern(page_bytes, 0x15);
    const auto sentinel_r = pattern(page_bytes, 0x8B);
    const auto sentinel_f = pattern(page_bytes, 0xC7);
    const auto plain_s = pattern(destination_size, 0x41);
    const auto plain_r = pattern(refused_inflated_size, 0x92);
    const auto plain_f = pattern(destination_size, 0x27);
    const auto stream_s = stored_zlib(plain_s);
    const auto stream_r = stored_zlib(plain_r);
    const auto stream_f = stored_zlib(plain_f);
    for (std::size_t j = 0; j < page_bytes; ++j) {
        bool distinct = sentinel_s[j] != sentinel_r[j];
        if (j >= page_offset) {
            const std::size_t k = j - page_offset;
            distinct = distinct && sentinel_s[j] != plain_s[k] && sentinel_r[j] != plain_r[k] && sentinel_r[j] != plain_s[k] &&
                       sentinel_s[j] != plain_r[k] && sentinel_f[j] != plain_f[k] && plain_s[k] != plain_r[k];
        }
        if (!distinct) {
            std::fprintf(stderr, "the fixture's sentinels and data share a byte at page offset 0x%zx\n", j);
            return 1;
        }
    }

    // Two pages, mapped at view (S) and view + 64 KiB (R).
    ULONG_PTR page_count = 2, pages[2] = {};
    if (!allocate(0x20400000ull, &page_count, pages) || page_count != 2 || !pages[0] || !pages[1] || (pages[0] & 0xF) ||
        (pages[1] & 0xF)) {
        std::fprintf(stderr, "cannot allocate two physical pages\n");
        return 1;
    }
    auto* view = static_cast<unsigned char*>(reserve(nullptr, 2 * page_bytes, 0x20402000, 0x21, PAGE_READWRITE));
    if (!view || map(view, 2, pages) != view) {
        std::fprintf(stderr, "cannot map the two pages into a view\n");
        return 1;
    }
    unsigned char* const page_s = view;
    unsigned char* const page_r = view + page_bytes;
    std::memcpy(page_s, sentinel_s.data(), page_bytes);
    std::memcpy(page_r, sentinel_r.data(), page_bytes);
    const ULONGLONG live_before = live_sections(stats);

    void *status_ptr = nullptr, *queue_ptr = nullptr;
    unsigned char description[0x28] = {};
    *reinterpret_cast<std::uint16_t*>(description + 8) = 16;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, slot_count, "ConsumerPair", nullptr, &status_ptr) != 0 ||
        reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue_ptr) != 0) {
        std::fprintf(stderr, "cannot create the queue\n");
        return 1;
    }
    auto* status = static_cast<Object*>(status_ptr);
    auto* queue = static_cast<Object*>(queue_ptr);
    void* fence_vtable[11] = {};
    fence_vtable[10] = reinterpret_cast<void*>(&Fence::signal);
    Fence fences[slot_count] = {};
    for (Fence& fence : fences) {
        fence.vtable = fence_vtable;
        fence.status = status;
        fence.view = view;
        fence.view_bytes = 2 * page_bytes;
        fence.seen.resize(2 * page_bytes);
    }
    fences[2].view_bytes = page_bytes;  // the follow-up maps one page

    // The token arrays stay readable until the requests run: the stand-in reads them again then.
    const ULONG_PTR token_s[1] = {pages[0]};
    const ULONG_PTR token_r[1] = {pages[1]};
    auto enqueue = [&](unsigned slot, const ULONG_PTR* tokens, const std::vector<unsigned char>& stream, unsigned inflated) {
        alignas(8) unsigned char request[0x48] = {};
        *reinterpret_cast<u64*>(request + 0x00) = zlib_page_request;
        *reinterpret_cast<const ULONG_PTR**>(request + 0x08) = tokens;
        *reinterpret_cast<u64*>(request + 0x10) = page_offset;
        *reinterpret_cast<unsigned*>(request + 0x18) = destination_size;
        *reinterpret_cast<const unsigned char**>(request + 0x20) = stream.data();
        *reinterpret_cast<unsigned*>(request + 0x30) = static_cast<unsigned>(stream.size());
        *reinterpret_cast<unsigned*>(request + 0x38) = inflated;
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, slot);
        reinterpret_cast<EnqueueSignalFn>(queue->vtable[5])(queue, &fences[slot], 0x200 + slot);
    };
    auto wait = [&](unsigned slot) {
        const DWORD start = GetTickCount();
        while (!fences[slot].signaled && GetTickCount() - start < completion_timeout_ms) Sleep(2);
        return fences[slot].signaled != 0;
    };
    // At fence `slot`, the entries up to it are complete with their result and the later ones are not.
    const long result[slot_count] = {S_OK, HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER), S_OK};
    auto check_fence = [&](unsigned slot, unsigned upto) {
        char what[160];
        const Fence& fence = fences[slot];
        bool order = fence.value == 0x200u + slot && fence.position == static_cast<LONG>(slot + 1);
        for (unsigned k = 0; k < upto; ++k) {
            order = order && fence.complete_at_signal[k] == (k <= slot);
            if (k <= slot) order = order && fence.hr_at_signal[k] == result[k];
        }
        std::snprintf(what, sizeof what, "fence %u (value 0x%llx, position %ld) ran out of order with its status entries", slot,
                      static_cast<unsigned long long>(fence.value), fence.position);
        return expect(order, what);
    };

    enqueue(0, token_s, stream_s, destination_size);
    enqueue(1, token_r, stream_r, refused_inflated_size);
    reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
    bool ok = expect(wait(0) && wait(1), "a fence of the pair was never signaled");
    if (!ok) return 1;

    // Status of each request.
    for (unsigned i = 0; i < 2; ++i) {
        char what[96];
        const bool done = reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, i);
        const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, i);
        std::snprintf(what, sizeof what, "%s status is %s (complete %d), got 0x%08lx", i ? "R" : "S", i ? "ERROR_INSUFFICIENT_BUFFER" : "S_OK",
                      done ? 1 : 0, static_cast<unsigned long>(hr));
        ok = expect(done && hr == result[i], what) && ok;
    }
    // Fence order, then the consumer's read at each fence and on this thread after both.
    ok = check_fence(0, 2) && ok;
    ok = check_fence(1, 2) && ok;
    ok = check_published("at S's fence", "S", fences[0].seen.data(), sentinel_s, plain_s) && ok;
    ok = check_refused("at S's fence (before R runs)", fences[0].seen.data() + page_bytes, sentinel_r, plain_r, false) && ok;
    ok = check_published("at R's fence", "S", fences[1].seen.data(), sentinel_s, plain_s) && ok;
    ok = check_refused("at R's fence", fences[1].seen.data() + page_bytes, sentinel_r, plain_r, accept_refused) && ok;
    ok = check_published("after both fences", "S", page_s, sentinel_s, plain_s) && ok;
    ok = check_refused("after both fences", page_r, sentinel_r, plain_r, accept_refused) && ok;

    // Release: unmap both pages from the view, then free them.
    ok = expect(map(view, 2, nullptr) == view, "unmapping the pair's pages failed") && ok;
    ok = expect(release(2, pages) != FALSE, "freeing the pair's pages failed") && ok;
    const ULONGLONG live_released = live_sections(stats);
    char what[192];
    std::snprintf(what, sizeof what, "live page backings %llu after release, expected %llu - 2", live_released, live_before);
    ok = expect(live_before >= 2 && live_released == live_before - 2, what) && ok;

    // Follow-up of S's shape on the same queue, into a page allocated after the release.
    ULONG_PTR follow_count = 1, follow_page[1] = {};
    if (!allocate(0x20400000ull, &follow_count, follow_page) || follow_count != 1 || !follow_page[0] || (follow_page[0] & 0xF) ||
        map(view, 1, follow_page) != view) {
        std::fprintf(stderr, "cannot allocate and map the follow-up page\n");
        return 1;
    }
    std::snprintf(what, sizeof what, "live page backings %llu after the follow-up page is mapped, expected %llu",
                  live_sections(stats), live_released + 1);
    ok = expect(live_sections(stats) == live_released + 1, what) && ok;
    // Nothing of the released pages carries over: neither S's bytes nor R's sentinel.
    const std::vector<unsigned char> zero(page_bytes, 0);
    std::size_t first = 0;
    const std::size_t carried = differing(view, 0, zero.data(), page_bytes, &first);
    std::snprintf(what, sizeof what, "the follow-up page is not zero-filled in %zu byte(s), first at 0x%zx", carried, first);
    ok = expect(carried == 0, what) && ok;
    std::memcpy(view, sentinel_f.data(), page_bytes);
    const ULONG_PTR token_f[1] = {follow_page[0]};
    enqueue(2, token_f, stream_f, destination_size);
    reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
    ok = expect(wait(2), "the follow-up fence was never signaled") && ok;
    if (fences[2].signaled) {
        const bool done = reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 2);
        const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 2);
        std::snprintf(what, sizeof what, "follow-up status is S_OK (complete %d), got 0x%08lx", done ? 1 : 0,
                      static_cast<unsigned long>(hr));
        ok = expect(done && hr == S_OK, what) && ok;
        ok = check_fence(2, 3) && ok;
        ok = check_published("at the follow-up fence", "follow-up", fences[2].seen.data(), sentinel_f, plain_f) && ok;
        ok = check_published("after the follow-up fence", "follow-up", view, sentinel_f, plain_f) && ok;
    }

    // The stand-in's own account of each request.
    const std::string log_text = read_text(log_path);
    const struct { const char* name; const void* source; const char* outcome; } logged[] = {
        {"S", stream_s.data(), "inflated"},
        {"R", stream_r.data(), "destination capacity exceeded"},
        {"follow-up", stream_f.data(), "inflated"},
    };
    for (const auto& entry : logged) {
        const std::string outcome = logged_outcome(log_text, entry.source);
        std::snprintf(what, sizeof what, "%s logged outcome \"%s\", expected \"%s\"", entry.name, outcome.c_str(), entry.outcome);
        ok = expect(outcome == entry.outcome, what) && ok;
    }

    map(view, 1, nullptr);
    release(1, follow_page);
    VirtualFree(view, 0, MEM_RELEASE);
    DeleteFileW(log_path);
    if (!ok) return 1;
    std::puts("S published its exact bytes at its fence; R reported ERROR_INSUFFICIENT_BUFFER and every byte of its page stayed its sentinel "
              "(no stream prefix) at its fence and after; the pages were unmapped and freed, and a follow-up of S's shape "
              "into a page allocated afterwards completed with exact bytes");
    return 0;
}
