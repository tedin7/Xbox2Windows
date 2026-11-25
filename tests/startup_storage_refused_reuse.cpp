// A refused page request's destination followed through a later valid write and a later refused write
// (#1198). Sibling of startup_storage_consumer_pair: that test releases the pages before reusing them;
// this one reuses the same mapped page without any release, as a title that polls status and then
// re-issues a read into the same buffer would.
//
// The stand-in DLL (argv[1]) runs the requests; page destinations are written through
// x2w_xmem_write_physical, so the xmem shim (argv[2]) is loaded first, as xmem.dll, under
// X2W_XMEM_PERMISSIVE. The test is the consumer: it maps one page with XMemMapPhysicalPages and reads
// the title-visible view at each fence.
//
// One Submit carries three zlib page requests for the same page, all with page_offset 0xF000 and
// destination_size 0x1000, each with its own status entry and an EnqueueSignal fence:
//   0 R1: a stream that inflates to 0x1800 bytes (refused as "destination capacity exceeded");
//   1 V:  a stream that inflates to exactly 0x1000 bytes (completes, replaces the destination);
//   2 R2: a second oversized stream with other bytes (refused, over bytes V already published).
// The page starts as a sentinel pattern. Sentinel, R1 and R2 prefixes and V differ at every byte of the
// destination, so one byte tells which of them a consumer is looking at.
// Checked:
//   - at R1's fence: entry 0 is ERROR_INSUFFICIENT_BUFFER and the page is still every byte its sentinel;
//   - at V's fence: entry 1 is S_OK, entry 0 is still ERROR_INSUFFICIENT_BUFFER (a later success does not rewrite an earlier
//     failure's status), and the destination holds exactly V's bytes, the sentinel below 0xF000;
//   - at R2's fence: entry 2 is ERROR_INSUFFICIENT_BUFFER, entries 0 and 1 are unchanged, and the page is byte-identical to
//     what it was at V's fence (the bounded refusal wrote nothing, not even a prefix);
//   - fences run in queue order, each after its own status entry;
//   - the stand-in's log names "destination capacity exceeded", "inflated", "destination capacity exceeded".
// --mutate accept-refused: a consumer that takes R1's destination as data (expects R1's stream prefix
// there at R1's fence), a negative control that must fail.
// Usage: startup_storage_refused_reuse <XCRDAPI.dll> <xmem.dll> [--mutate accept-refused]
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
constexpr unsigned oversized_inflated_size = 0x1800;
constexpr unsigned slot_count = 3;  // R1, V, R2
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

// Notes, when signaled, what a consumer sees: every status entry and the bytes of the mapped page.
struct Fence {
    void** vtable;
    Object* status;
    const unsigned char* view;
    std::vector<unsigned char> seen;  // the page at signal time; sized before Submit
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
        std::memcpy(fence->seen.data(), fence->view, page_bytes);
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

// Count of k in [0, size) where page[from + k] != want[k]; the first such offset in *first.
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

// How many bytes of the destination equal `want` (a stream's bytes at that position).
std::size_t matching(const unsigned char* page, const std::vector<unsigned char>& want)
{
    std::size_t same = 0;
    for (std::size_t k = 0; k < destination_size; ++k) same += page[page_offset + k] == want[k];
    return same;
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

// A page that is every byte `sentinel`, optionally with `data` from page_offset on.
bool page_is(const char* when, const unsigned char* page, const std::vector<unsigned char>& sentinel,
             const std::vector<unsigned char>* data)
{
    char what[192];
    std::size_t first = 0;
    const std::size_t below = differing(page, 0, sentinel.data(), page_offset, &first);
    std::snprintf(what, sizeof what, "%s: page below 0x%x is not the sentinel in %zu byte(s), first at 0x%zx", when, page_offset,
                  below, first);
    bool ok = expect(below == 0, what);
    const std::vector<unsigned char>& want_top = data ? *data : sentinel;
    const std::size_t top_from = data ? 0 : page_offset;
    const std::size_t bad = differing(page, page_offset, want_top.data() + top_from, destination_size, &first);
    std::snprintf(what, sizeof what, "%s: destination is not the expected %s in %zu byte(s), first at 0x%zx", when,
                  data ? "inflated bytes" : "sentinel", bad, first);
    return expect(bad == 0, what) && ok;
}

}  // namespace

int main(int argc, char** argv)
{
    const bool accept_refused = argc == 5 && std::strcmp(argv[3], "--mutate") == 0 && std::strcmp(argv[4], "accept-refused") == 0;
    if (argc != 3 && !accept_refused) {
        std::fprintf(stderr, "usage: startup_storage_refused_reuse <XCRDAPI.dll> <xmem.dll> [--mutate accept-refused]\n");
        return 2;
    }
    wchar_t temp_dir[MAX_PATH] = {}, log_path[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, temp_dir);
    std::swprintf(log_path, MAX_PATH, L"%lsx2w_refused_reuse_%lu.log", temp_dir, GetCurrentProcessId());
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
    if (!allocate || !release || !reserve || !map) {
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

    // Inputs. Sentinel, both oversized prefixes and V's bytes differ at every byte of the destination.
    const auto sentinel = pattern(page_bytes, 0x15);
    const auto plain_r1 = pattern(oversized_inflated_size, 0x8B);
    const auto plain_v = pattern(destination_size, 0x41);
    const auto plain_r2 = pattern(oversized_inflated_size, 0xD3);
    const auto stream_r1 = stored_zlib(plain_r1);
    const auto stream_v = stored_zlib(plain_v);
    const auto stream_r2 = stored_zlib(plain_r2);
    for (std::size_t k = 0; k < destination_size; ++k) {
        const unsigned char s = sentinel[page_offset + k];
        if (s == plain_r1[k] || s == plain_v[k] || s == plain_r2[k] || plain_r1[k] == plain_v[k] || plain_r2[k] == plain_v[k] ||
            plain_r1[k] == plain_r2[k]) {
            std::fprintf(stderr, "the fixture's sentinel and data share a byte at destination offset 0x%zx\n", k);
            return 1;
        }
    }

    ULONG_PTR page_count = 1, pages[1] = {};
    if (!allocate(0x20400000ull, &page_count, pages) || page_count != 1 || !pages[0] || (pages[0] & 0xF)) {
        std::fprintf(stderr, "cannot allocate a physical page\n");
        return 1;
    }
    auto* view = static_cast<unsigned char*>(reserve(nullptr, page_bytes, 0x20402000, 0x21, PAGE_READWRITE));
    if (!view || map(view, 1, pages) != view) {
        std::fprintf(stderr, "cannot map the page into a view\n");
        return 1;
    }
    std::memcpy(view, sentinel.data(), page_bytes);

    void *status_ptr = nullptr, *queue_ptr = nullptr;
    // The stand-in logs the first 40 bytes of the descriptor.
    alignas(8) unsigned char description[0x28] = {};
    *reinterpret_cast<std::uint16_t*>(description + 8) = 16;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, slot_count, "RefusedReuse", nullptr, &status_ptr) != 0 ||
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
        fence.seen.resize(page_bytes);
    }

    // The token array stays readable until the requests run: the stand-in reads it again then.
    const ULONG_PTR tokens[1] = {pages[0]};
    auto enqueue = [&](unsigned slot, const std::vector<unsigned char>& stream) {
        alignas(8) unsigned char request[0x48] = {};
        *reinterpret_cast<u64*>(request + 0x00) = zlib_page_request;
        *reinterpret_cast<const ULONG_PTR**>(request + 0x08) = tokens;
        *reinterpret_cast<u64*>(request + 0x10) = page_offset;
        *reinterpret_cast<unsigned*>(request + 0x18) = destination_size;
        *reinterpret_cast<const unsigned char**>(request + 0x20) = stream.data();
        *reinterpret_cast<unsigned*>(request + 0x30) = static_cast<unsigned>(stream.size());
        // CancellationTag is opaque and independent of the decoded length.
        *reinterpret_cast<u64*>(request + 0x38) = 0x100000000ull + slot;
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, slot);
        reinterpret_cast<EnqueueSignalFn>(queue->vtable[5])(queue, &fences[slot], 0x300 + slot);
    };
    enqueue(0, stream_r1);
    enqueue(1, stream_v);
    enqueue(2, stream_r2);
    reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

    bool all_signaled = true;
    for (unsigned slot = 0; slot < slot_count; ++slot) {
        const DWORD start = GetTickCount();
        while (!fences[slot].signaled && GetTickCount() - start < completion_timeout_ms) Sleep(2);
        all_signaled = all_signaled && fences[slot].signaled != 0;
    }
    if (!expect(all_signaled, "a fence of the sequence was never signaled")) return 1;

    // Status at each fence: entries up to it are complete with their final result, later ones are not.
    const long result[slot_count] = {HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER), S_OK, HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)};
    const char* const name[slot_count] = {"R1", "V", "R2"};
    bool ok = true;
    for (unsigned slot = 0; slot < slot_count; ++slot) {
        char what[192];
        const Fence& fence = fences[slot];
        bool sequence = fence.value == 0x300u + slot && fence.position == static_cast<LONG>(slot + 1);
        for (unsigned k = 0; k < slot_count; ++k) {
            sequence = sequence && fence.complete_at_signal[k] == (k <= slot);
            if (k <= slot) sequence = sequence && fence.hr_at_signal[k] == result[k];
        }
        std::snprintf(what, sizeof what, "fence of %s (value 0x%llx, position %ld) ran out of order or saw a changed status", name[slot],
                      static_cast<unsigned long long>(fence.value), fence.position);
        ok = expect(sequence, what) && ok;
        const bool done = reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, slot);
        const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, slot);
        std::snprintf(what, sizeof what, "%s status finally is 0x%08lx (complete %d), expected 0x%08lx", name[slot],
                      static_cast<unsigned long>(hr), done ? 1 : 0, static_cast<unsigned long>(result[slot]));
        ok = expect(done && hr == result[slot], what) && ok;
    }

    // The consumer's view of the page at each fence.
    if (accept_refused) {
        char what[160];
        const std::size_t bad = matching(fences[0].seen.data(), plain_r1);
        std::snprintf(what, sizeof what, "R1 destination taken as data matches the stream prefix in %zu of %u byte(s)", bad,
                      destination_size);
        ok = expect(bad == destination_size, what) && ok;
    } else {
        ok = page_is("at R1's fence", fences[0].seen.data(), sentinel, nullptr) && ok;
        char what[128];
        const std::size_t prefix = matching(fences[0].seen.data(), plain_r1);
        std::snprintf(what, sizeof what, "at R1's fence: destination holds %zu byte(s) of R1's stream prefix", prefix);
        ok = expect(prefix == 0, what) && ok;
    }
    ok = page_is("at V's fence", fences[1].seen.data(), sentinel, &plain_v) && ok;
    ok = page_is("at R2's fence", fences[2].seen.data(), sentinel, &plain_v) && ok;
    ok = page_is("after all fences", view, sentinel, &plain_v) && ok;
    {
        char what[128];
        const std::size_t prefix = matching(fences[2].seen.data(), plain_r2);
        std::snprintf(what, sizeof what, "at R2's fence: destination holds %zu byte(s) of R2's stream prefix", prefix);
        ok = expect(prefix == 0, what) && ok;
    }

    // The stand-in's own account of each request.
    const std::string log_text = read_text(log_path);
    const struct { const char* name; const void* source; const char* outcome; } logged[] = {
        {"R1", stream_r1.data(), "destination capacity exceeded"},
        {"V", stream_v.data(), "inflated"},
        {"R2", stream_r2.data(), "destination capacity exceeded"},
    };
    for (const auto& entry : logged) {
        char what[192];
        const std::string outcome = logged_outcome(log_text, entry.source);
        std::snprintf(what, sizeof what, "%s logged outcome \"%s\", expected \"%s\"", entry.name, outcome.c_str(), entry.outcome);
        ok = expect(outcome == entry.outcome, what) && ok;
    }

    map(view, 1, nullptr);
    release(1, pages);
    VirtualFree(view, 0, MEM_RELEASE);
    DeleteFileW(log_path);
    if (!ok) return 1;
    std::puts("R1 reported ERROR_INSUFFICIENT_BUFFER and left its page the sentinel; V then replaced the destination with its exact bytes "
              "while R1's status stayed ERROR_INSUFFICIENT_BUFFER; R2 reported ERROR_INSUFFICIENT_BUFFER and left V's bytes untouched, with no stream prefix");
    return 0;
}
