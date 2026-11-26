// The DirectStorage stand-in counts each failed request under one cause, and the heartbeat's second line
// reports those counts (#1198). Sibling of startup_storage_refusal_order: the same kind of generated file
// and one Submit, here with a request for every cause the heartbeat names.
//
// The test compiles the stand-in's source into itself (like startup_storage_worker_start), so it reads the
// counters and formats the heartbeat line through the stand-in's own code, without waiting for the 10 s
// heartbeat and without an export or environment variable added for the test.
//
// One Submit carries, each request followed by its status entry and the Submit closed by a fence:
//   valid raw read; valid zlib stream;
//   at or past EOF: a raw read at offset == file size, a zlib read at offset 2^64 - 8 (offset + size wraps);
//   crossing EOF: raw and zlib reads of size 0x98cabbec at offset 0, a raw read of 16 bytes 8 bytes before EOF;
//   other read failure: a read through a File object that was closed;
//   destination capacity: a valid zlib stream of 4096 bytes into a 2048-byte destination;
//   destination not writable: a valid raw read into a read-only page;
//   zlib: a stream whose Adler-32 is wrong;
//   BCPack: a BC7 BCPack request whose destination is not a whole number of blocks;
//   other: a request with an unknown option bit.
// Checked: every status, the exact count of every cause, that the causes add up to the failed total, the
// done and inflated totals, the exact heartbeat line, the per-request text "read failed" of the six refused
// reads, and the cause of edge ranges near 2^64 (no wrap).
// --mutate one-bucket: the expectations of a counter that books every failure under one cause (a negative
// control; it must fail).
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

#include "startup_stubs/xcrdapi.cpp"

namespace {

constexpr DWORD completion_timeout_ms = 15000;
constexpr unsigned implausible_size = 0x98cabbec;
constexpr unsigned piece = 4096;
constexpr unsigned cause_count = static_cast<unsigned>(StorageFailure::count);

using OpenFileFn = long(__stdcall*)(void*, const wchar_t*, const void*, void**);
using CloseFileFn = void(__stdcall*)(void*);
using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using EnqueueSignalFn = void(__stdcall*)(void*, void*, std::uint64_t);
using SubmitFn = void(__stdcall*)(void*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
// The status a refused request reports, written out: EOF causes and a too-small destination are
// distinct codes; every other cause stays E_FAIL.
long failure_hresult(StorageFailure cause)
{
    if (cause == StorageFailure::read_at_or_past_eof || cause == StorageFailure::read_crossing_eof)
        return HRESULT_FROM_WIN32(ERROR_HANDLE_EOF);
    if (cause == StorageFailure::destination_capacity) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    return E_FAIL;
}
using GetHResultFn = long(__stdcall*)(void*, unsigned);

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

const char* cause_name(unsigned cause)
{
    static const char* const names[cause_count] = {"read-at-or-past-eof", "read-crossing-eof", "read-other",
                                                   "destination-capacity", "destination-not-writable", "zlib",
                                                   "bcpack", "other"};
    return cause < cause_count ? names[cause] : "?";
}

// Ranges a naive end-offset comparison would get wrong: offset + count wraps past 2^64.
bool check_edge_ranges()
{
    constexpr u64 max = UINT64_MAX;
    struct Case { bool known; u64 size, offset, count; StorageFailure cause; const char* what; };
    const Case cases[] = {
        {true, 100, 100, 1, StorageFailure::read_at_or_past_eof, "offset == size"},
        {true, 100, max, 1, StorageFailure::read_at_or_past_eof, "offset 2^64 - 1"},
        {true, 100, max - 7, 16, StorageFailure::read_at_or_past_eof, "offset + count wraps, offset past EOF"},
        {true, 100, 99, 2, StorageFailure::read_crossing_eof, "one byte past EOF"},
        {true, 100, 1, max, StorageFailure::read_crossing_eof, "count 2^64 - 1, offset + count wraps to 0"},
        {true, max, max - 1, 2, StorageFailure::read_crossing_eof, "size 2^64 - 1, offset + count wraps to 0"},
        {true, 100, 99, 1, StorageFailure::read_other, "range ends at EOF"},
        {true, 0, 0, 1, StorageFailure::read_at_or_past_eof, "empty file"},
        {false, 0, 0, 1, StorageFailure::read_other, "file size unknown"},
    };
    bool ok = true;
    for (const Case& c : cases) {
        char what[160];
        const StorageFailure got = read_failure_cause(c.known, c.size, c.offset, c.count);
        std::snprintf(what, sizeof what, "edge range (%s) is %s, expected %s", c.what,
                      cause_name(static_cast<unsigned>(got)), cause_name(static_cast<unsigned>(c.cause)));
        ok = expect(got == c.cause, what) && ok;
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv)
{
    const bool one_bucket = argc == 3 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "one-bucket") == 0;
    if (argc != 1 && !one_bucket) {
        std::fprintf(stderr, "usage: startup_storage_failure_counts [--mutate one-bucket]\n");
        return 2;
    }
    wchar_t temp_dir[MAX_PATH] = {}, log_path[MAX_PATH] = {}, data_path[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, temp_dir);
    std::swprintf(log_path, MAX_PATH, L"%lsx2w_failure_counts_%lu.log", temp_dir, GetCurrentProcessId());
    std::swprintf(data_path, MAX_PATH, L"%lsx2w_failure_counts_%lu.bin", temp_dir, GetCurrentProcessId());
    DeleteFileW(log_path);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path);

    // File layout: raw[0, 2 * piece) patterned, a valid zlib stream of piece bytes, the same stream with
    // a wrong Adler-32.
    std::vector<unsigned char> raw(2 * piece), plain(piece);
    for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<unsigned char>((i * 7 + (i >> 8)) ^ 0x3c);
    for (std::size_t i = 0; i < plain.size(); ++i) plain[i] = static_cast<unsigned char>((i % 241) ^ 0x6b);
    const auto stream = stored_zlib(plain, false);
    const auto damaged = stored_zlib(plain, true);
    std::vector<unsigned char> file_bytes = raw;
    const u64 zlib_offset = file_bytes.size();
    file_bytes.insert(file_bytes.end(), stream.begin(), stream.end());
    const u64 damaged_offset = file_bytes.size();
    file_bytes.insert(file_bytes.end(), damaged.begin(), damaged.end());
    const u64 file_size = file_bytes.size();
    {
        HANDLE out = CreateFileW(data_path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        DWORD written = 0;
        if (out == INVALID_HANDLE_VALUE ||
            !WriteFile(out, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr) ||
            written != file_bytes.size()) {
            std::fprintf(stderr, "cannot write the test file\n");
            return 1;
        }
        CloseHandle(out);
    }

    void* object = nullptr;
    if (DStorageGetFactory(nullptr, &object) != 0 || !object) {
        std::fprintf(stderr, "cannot get the factory\n");
        return 1;
    }
    auto* factory = static_cast<Object*>(object);
    void *file_ptr = nullptr, *closed_ptr = nullptr;
    const auto open_file = reinterpret_cast<OpenFileFn>(factory->vtable[4]);
    if (open_file(factory, data_path, nullptr, &file_ptr) != 0 || !file_ptr ||
        open_file(factory, data_path, nullptr, &closed_ptr) != 0 || !closed_ptr) {
        std::fprintf(stderr, "cannot open the test file\n");
        return 1;
    }
    reinterpret_cast<CloseFileFn>(static_cast<Object*>(closed_ptr)->vtable[3])(closed_ptr);

    unsigned char* read_only = static_cast<unsigned char*>(VirtualAlloc(nullptr, piece, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY));
    if (!read_only) {
        std::fprintf(stderr, "cannot allocate the read-only destination\n");
        return 1;
    }

    using C = StorageFailure;
    struct Plan {
        u64 options;
        void* source;
        u64 offset;
        unsigned size;
        unsigned destination_size;
        bool read_only_destination;
        bool valid;
        C cause;
    };
    const unsigned stream_size = static_cast<unsigned>(stream.size());
    // The capacity request is followed by a raw one, so it cannot be taken for the first piece of a
    // chunked stream (a zlib request at offset + 2048 in the same Submit would be).
    const Plan plan[] = {
        {0, file_ptr, 0, piece, piece, false, true, C::other},
        {1, file_ptr, zlib_offset, stream_size, piece, false, true, C::other},
        {0, file_ptr, file_size, 16, piece, false, false, C::read_at_or_past_eof},
        {1, file_ptr, UINT64_MAX - 7, 16, piece, false, false, C::read_at_or_past_eof},
        {0, file_ptr, 0, implausible_size, piece, false, false, C::read_crossing_eof},
        {1, file_ptr, 0, implausible_size, piece, false, false, C::read_crossing_eof},
        {0, file_ptr, file_size - 8, 16, piece, false, false, C::read_crossing_eof},
        {0, closed_ptr, 16, 32, piece, false, false, C::read_other},
        {1, file_ptr, zlib_offset, stream_size, piece / 2, false, false, C::destination_capacity},
        {0, file_ptr, piece, piece, piece, true, false, C::destination_not_writable},
        {1, file_ptr, damaged_offset, static_cast<unsigned>(damaged.size()), piece, false, false, C::zlib},
        {7u << 1, file_ptr, 0, 64, piece + 8, false, false, C::bcpack},  // BC7 needs whole 16-byte blocks
        {1ull << 40, file_ptr, 32, 64, piece, false, false, C::other},
    };
    constexpr unsigned request_count = sizeof(plan) / sizeof(plan[0]);

    void *status_ptr = nullptr, *queue_ptr = nullptr;
    unsigned char description[0x28] = {};
    *reinterpret_cast<std::uint16_t*>(description + 8) = 16;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, request_count, "Counts", nullptr, &status_ptr) != 0 ||
        reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue_ptr) != 0) {
        std::fprintf(stderr, "cannot create the queue\n");
        return 1;
    }
    auto* status = static_cast<Object*>(status_ptr);
    auto* queue = static_cast<Object*>(queue_ptr);
    std::vector<unsigned char> destination[request_count];
    for (unsigned i = 0; i < request_count; ++i) {
        destination[i].assign(plan[i].destination_size, 0xA5);
        alignas(8) unsigned char request[0x48] = {};
        *reinterpret_cast<u64*>(request + 0x00) = plan[i].options;
        *reinterpret_cast<unsigned char**>(request + 0x08) = plan[i].read_only_destination ? read_only : destination[i].data();
        *reinterpret_cast<unsigned*>(request + 0x18) = plan[i].destination_size;
        *reinterpret_cast<void**>(request + 0x20) = plan[i].source;
        *reinterpret_cast<u64*>(request + 0x28) = plan[i].offset;
        *reinterpret_cast<unsigned*>(request + 0x30) = plan[i].size;
        *reinterpret_cast<unsigned*>(request + 0x38) = piece;
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, i);
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

    LONG expected[cause_count] = {};
    LONG expected_failed = 0;
    for (unsigned i = 0; i < request_count; ++i) {
        char what[128];
        const bool done = reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, i);
        const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, i);
        std::snprintf(what, sizeof what, "request %u status is %s (complete %d), got 0x%08lx", i,
                      plan[i].valid ? "S_OK" : "its cause's failure", done ? 1 : 0, static_cast<unsigned long>(hr));
        ok = expect(done && hr == (plan[i].valid ? S_OK : failure_hresult(plan[i].cause)), what) && ok;
        if (!plan[i].valid) {
            ++expected[static_cast<unsigned>(plan[i].cause)];
            ++expected_failed;
        }
    }
    if (one_bucket) {
        for (LONG& count : expected) count = 0;
        expected[static_cast<unsigned>(C::other)] = expected_failed;
    }

    StorageFailureCounts failure_counts;
    const LONG sum = sample_failure_counts(failure_counts);
    for (unsigned cause = 0; cause < cause_count; ++cause) {
        const LONG got = failure_counts[cause];
        char what[128];
        std::snprintf(what, sizeof what, "cause %s counted %ld, expected %ld", cause_name(cause), got, expected[cause]);
        ok = expect(got == expected[cause], what) && ok;
    }
    char what[160];
    std::snprintf(what, sizeof what, "the causes add up to %ld, failed total %ld, expected %ld", sum, requests_failed,
                  expected_failed);
    ok = expect(sum == requests_failed && requests_failed == expected_failed, what) && ok;
    std::snprintf(what, sizeof what, "raw=%ld inflated=%ld, expected 1 and 1", requests_done, requests_compressed);
    ok = expect(requests_done == 1 && requests_compressed == 1, what) && ok;

    // The heartbeat's line, through the stand-in's formatter, against text written out here.
    char line[384], want[384];
    // A worker can publish a new failure after the totals line is emitted.
    // The breakdown for that same heartbeat must still use the earlier sample.
    count_failure(C::other);
    format_failure_causes(line, sizeof line, 20, failure_counts);
    std::snprintf(want, sizeof want,
                  "x2w-dstorage heartbeat 20 s: failed requests by cause read-at-or-past-eof=%ld read-crossing-eof=%ld "
                  "read-other=%ld destination-capacity=%ld destination-not-writable=%ld zlib=%ld bcpack=%ld other=%ld",
                  expected[0], expected[1], expected[2], expected[3], expected[4], expected[5], expected[6], expected[7]);
    if (!expect(std::strcmp(line, want) == 0, "heartbeat line differs")) {
        std::fprintf(stderr, "  got:  %s\n  want: %s\n", line, want);
        ok = false;
    }

    // The per-request text of every refused read is still exactly "read failed".
    const std::string log_text = read_text(log_path);
    auto outcome_of = [&](u64 options, u64 offset, unsigned size) {
        char key[64], tail[96];
        std::snprintf(key, sizeof key, "options=0x%llx ", static_cast<unsigned long long>(options));
        std::snprintf(tail, sizeof tail, "offset=0x%llx size=0x%x -> ", static_cast<unsigned long long>(offset), size);
        for (std::size_t at = 0; (at = log_text.find(tail, at)) != std::string::npos; ++at) {
            const std::size_t line_start = log_text.rfind('\n', at);
            const std::size_t from = line_start == std::string::npos ? 0 : line_start + 1;
            if (log_text.substr(from, at - from).find(key) == std::string::npos) continue;
            const std::size_t end = log_text.find('\r', at);
            const std::size_t begin = at + std::strlen(tail);
            return log_text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        }
        return std::string("(not logged)");
    };
    for (unsigned i = 0; i < request_count; ++i) {
        const auto cause = plan[i].cause;
        if (plan[i].valid ||
            (cause != C::read_at_or_past_eof && cause != C::read_crossing_eof && cause != C::read_other))
            continue;
        const std::string outcome = outcome_of(plan[i].options, plan[i].offset, plan[i].size);
        std::snprintf(what, sizeof what, "request %u logged outcome \"%s\", expected \"read failed\"", i, outcome.c_str());
        ok = expect(outcome == "read failed", what) && ok;
    }

    // The status-line counters (#2816). This mixed run finished one raw copy and one zlib request and
    // failed the rest: completed is raw + inflated + bcpack-blank, a failure is not in it, and the
    // "inflated=%ld failed=%ld" text that tools/startup/live_view.py parses is unchanged.
    // The heartbeat check above added one more failure (count_failure(C::other)) after the totals were sampled.
    const LONG failed_now = expected_failed + 1;
        char counts[160], want_counts[160];
    format_request_counts(counts, sizeof counts, requests_done, requests_compressed, requests_failed,
                          requests_bcpack_blank);
    std::snprintf(want_counts, sizeof want_counts, "raw=1 inflated=1 failed=%ld bcpack-blank=0 completed=2", failed_now);
    std::snprintf(what, sizeof what, "mixed run counts \"%s\", expected \"%s\"", counts, want_counts);
    ok = expect(std::strcmp(counts, want_counts) == 0, what) && ok;
    char parsed_tail[64];
    std::snprintf(parsed_tail, sizeof parsed_tail, "inflated=1 failed=%ld", failed_now);
    ok = expect(std::strstr(counts, parsed_tail) != nullptr, "the inflated=N failed=N text live_view.py parses changed") && ok;

    // A run whose requests are all zlib: nothing is counted raw, yet completed equals the finished requests,
    // so the status line no longer reads as if storage were stuck.
    {
        requests_done = requests_compressed = requests_failed = requests_bcpack_blank = 0;
        constexpr unsigned zlib_requests = 3;
        void *zlib_status_ptr = nullptr, *zlib_queue_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, zlib_requests, "Zlib", nullptr, &zlib_status_ptr) != 0 ||
            reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &zlib_queue_ptr) != 0) {
            std::fprintf(stderr, "cannot create the zlib-only queue\n");
            return 1;
        }
        auto* zlib_status = static_cast<Object*>(zlib_status_ptr);
        auto* zlib_queue = static_cast<Object*>(zlib_queue_ptr);
        std::vector<unsigned char> zlib_destination[zlib_requests];
        const LONG enqueued_before = requests_enqueued;
        const LONG disk_reads_before = disk_snapshot().reads;
        const LONG64 disk_bytes_before = disk_snapshot().bytes, disk_ticks_before = disk_snapshot().ticks;
        const LONG64 read_before = bytes_read_total, inflated_before = bytes_inflated_total;
        for (unsigned i = 0; i < zlib_requests; ++i) {
            zlib_destination[i].assign(piece, 0xA5);
            alignas(8) unsigned char request[0x48] = {};
            *reinterpret_cast<u64*>(request + 0x00) = 1;
            *reinterpret_cast<unsigned char**>(request + 0x08) = zlib_destination[i].data();
            *reinterpret_cast<unsigned*>(request + 0x18) = piece;
            *reinterpret_cast<void**>(request + 0x20) = file_ptr;
            *reinterpret_cast<u64*>(request + 0x28) = zlib_offset;
            *reinterpret_cast<unsigned*>(request + 0x30) = stream_size;
            *reinterpret_cast<unsigned*>(request + 0x38) = piece;
            reinterpret_cast<EnqueueRequestFn>(zlib_queue->vtable[3])(zlib_queue, request);
            reinterpret_cast<EnqueueStatusFn>(zlib_queue->vtable[4])(zlib_queue, zlib_status, i);
        }
        Fence zlib_fence{fence_vtable, 0};
        reinterpret_cast<EnqueueSignalFn>(zlib_queue->vtable[5])(zlib_queue, &zlib_fence, 1);
        reinterpret_cast<SubmitFn>(zlib_queue->vtable[6])(zlib_queue);
        const DWORD zlib_start = GetTickCount();
        while (!zlib_fence.signaled && GetTickCount() - zlib_start < completion_timeout_ms) Sleep(2);
        ok = expect(zlib_fence.signaled, "the fence after the zlib-only requests was never signaled") && ok;
        unsigned finished = 0;
        for (unsigned i = 0; i < zlib_requests; ++i)
            finished += reinterpret_cast<IsCompleteFn>(zlib_status->vtable[3])(zlib_status, i) ? 1 : 0;
        format_request_counts(counts, sizeof counts, requests_done, requests_compressed, requests_failed,
                              requests_bcpack_blank);
        std::snprintf(want_counts, sizeof want_counts, "raw=0 inflated=%u failed=0 bcpack-blank=0 completed=%u", finished, finished);
        std::snprintf(what, sizeof what, "zlib-only run: %u finished, counts \"%s\", expected \"%s\"", finished, counts, want_counts);
        ok = expect(finished == zlib_requests && std::strcmp(counts, want_counts) == 0, what) && ok;
        // Loading progress: every request is counted at its enqueue, its compressed bytes as read and what it wrote as inflated.
        std::snprintf(what, sizeof what, "progress counters: enqueued +%ld (want %u), read +%lld (want %llu), inflated +%lld (want %llu)",
                      requests_enqueued - enqueued_before, zlib_requests, static_cast<long long>(bytes_read_total - read_before),
                      static_cast<unsigned long long>(zlib_requests) * stream_size,
                      static_cast<long long>(bytes_inflated_total - inflated_before), static_cast<unsigned long long>(zlib_requests) * piece);
        ok = expect(requests_enqueued - enqueued_before == static_cast<LONG>(zlib_requests) &&
                        bytes_read_total - read_before == static_cast<LONG64>(zlib_requests) * stream_size &&
                        bytes_inflated_total - inflated_before == static_cast<LONG64>(zlib_requests) * piece, what) && ok;
        {
            // The disk statistics: the real reads were counted, and a scripted sequence of reads on a fake handle is
            // classified first / sequential / forward / backward with its size buckets.
            const DiskStats real = disk_snapshot();
            std::snprintf(what, sizeof what, "disk stats: %ld reads (+%ld, want at least %u), %lld bytes", real.reads, real.reads - disk_reads_before,
                          zlib_requests, static_cast<long long>(real.bytes));
            ok = expect(real.reads - disk_reads_before == static_cast<LONG>(zlib_requests) &&
                            real.bytes - disk_bytes_before == static_cast<LONG64>(zlib_requests) * stream_size &&
                            real.ticks >= disk_ticks_before, what) && ok;
            const DiskStats start = disk_snapshot();
            HANDLE fake = reinterpret_cast<HANDLE>(static_cast<ULONG_PTR>(0xF4C0));
            note_disk_read(fake, 0, 4096, 10, true);        // first of its file, <=4K
            note_disk_read(fake, 4096, 70000, 10, true);    // sequential, <=256K
            note_disk_read(fake, 900000, 2000000, 10, true);  // forward, >1M
            note_disk_read(fake, 0, 100, 10, true);         // backward, <=4K
            note_disk_read(fake, 5, 5, 10, false);          // a failed read is counted apart
            const DiskStats now = disk_snapshot();
            std::snprintf(what, sizeof what, "disk order: first %ld sequential %ld forward %ld backward %ld failed %ld; sizes %ld %ld %ld %ld %ld %ld",
                          now.first - start.first, now.sequential - start.sequential, now.forward - start.forward,
                          now.backward - start.backward, now.failed - start.failed, now.sizes[0] - start.sizes[0],
                          now.sizes[1] - start.sizes[1], now.sizes[2] - start.sizes[2], now.sizes[3] - start.sizes[3],
                          now.sizes[4] - start.sizes[4], now.sizes[5] - start.sizes[5]);
            ok = expect(now.first - start.first == 1 && now.sequential - start.sequential == 1 && now.forward - start.forward == 1 &&
                            now.backward - start.backward == 1 && now.failed - start.failed == 1 && now.sizes[0] - start.sizes[0] == 2 &&
                            now.sizes[3] - start.sizes[3] == 1 && now.sizes[5] - start.sizes[5] == 1, what) && ok;
            char disk_text[512];
            DiskStats before{};
            DiskStats after{};
            after.ticks = 5000000;  // 0.5 s of reads at 10 MHz
            after.bytes = 50 << 20;   // 50 MiB
            after.reads = 100;
            after.sizes[4] = 100;
            after.sequential = 99;
            after.first = 1;
            format_disk(disk_text, sizeof disk_text, 20, after, before, 10.0, 10000000.0);
            const char* const want_disk =
                "x2w-dstorage disk 20 s: 100 reads (+100), 100.0 MB/s while reading, reading 5% of the window; "
                "read sizes <=4K 0 <=16K 0 <=64K 0 <=256K 0 <=1M 100 >1M 0; offsets sequential 99 forward 0 backward 0 first 1; read failures 0";
            if (!expect(std::strcmp(disk_text, want_disk) == 0, "disk line differs")) {
                std::fprintf(stderr, "  got:  %s\n  want: %s\n", disk_text, want_disk);
                ok = false;
            }
        }
        {
            // The progress line: 10 enqueued, 7 completed, the request numbered 7 waited 1234 ms, 2 s window.
            enqueue_ring_tick[7 & (enqueue_ring_size - 1)] = static_cast<LONG>(100000 - 1234);
            char progress[256];
            format_progress(progress, sizeof progress, 20, 10, 7, 2 << 20, 4 << 20, 5 << 20, 9 << 20, 2.0, 100000);
            const char* const want_progress =
                "x2w-dstorage progress 20 s: pending 3 (oldest 1234 ms), read 1.0 MB/s, inflated 2.0 MB/s, "
                "enqueued 10 completed 7, total read 5120 KiB inflated 9216 KiB";
            if (!expect(std::strcmp(progress, want_progress) == 0, "progress line differs")) {
                std::fprintf(stderr, "  got:  %s\n  want: %s\n", progress, want_progress);
                ok = false;
            }
            format_progress(progress, sizeof progress, 30, 10, 10, 0, 0, 5 << 20, 9 << 20, 10.0, 100000);
            ok = expect(std::strstr(progress, "pending 0 (oldest none), read 0.0 MB/s, inflated 0.0 MB/s") != nullptr,
                        "an idle queue must read pending 0, oldest none") && ok;
        }
        const std::string status_text = read_text(log_path);
        ok = expect(status_text.find("(requests raw=") != std::string::npos &&
                        status_text.find(" completed=") != std::string::npos &&
                        status_text.find("requests done=") == std::string::npos,
                    "the EnqueueStatus line does not carry raw= and completed= (or still says done=)") && ok;
    }

    ok = check_edge_ranges() && ok;

    reinterpret_cast<CloseFileFn>(static_cast<Object*>(file_ptr)->vtable[3])(file_ptr);
    VirtualFree(read_only, 0, MEM_RELEASE);
    DeleteFileW(data_path);
    DeleteFileW(log_path);
    if (!ok) return 1;
    std::printf("%s\n", line);
    std::puts("each failed request is counted under its cause; the causes add up to the failed total");
    return 0;
}
