// HDD mode of the DirectStorage stand-in through its real queue (src/startup_stubs/xcrdapi.cpp, dstorage_read_cache.h).
//
//   startup_dstorage_hdd_queue on    X2W_STORAGE_HDD=1: the prefetch thread reads the planned ranges sorted and merged
//   startup_dstorage_hdd_queue off   mode off: the same requests, read one by one, as before
//
// One Submit carries 48 raw requests of 4 KiB at shuffled offsets of a generated 192 KiB file (neighbouring ranges), then a
// fence. Checked in both modes: every status is S_OK and every destination holds exactly the file's bytes at that offset.
// In mode "on": every request but the first (taken before the window is planned) is served from the cache (47 hits, 1 miss),
// the 47 planned reads merge into a few disk reads (47 minus the groups saved); the disk statistics show 1 + groups file reads, not 48.
// In mode "off": 48 file reads, no cache activity.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// Intercept only the file reader used by the queue and prefetch cache.
static bool slow_reader = false;
static BOOL read_with_delay(HANDLE file, void* buffer, DWORD count, DWORD* read, OVERLAPPED* position)
{
    if (slow_reader) std::this_thread::sleep_for(std::chrono::milliseconds(8));
    return ReadFile(file, buffer, count, read, position);
}
#define ReadFile read_with_delay
#include "startup_stubs/dstorage_file_read.h"
#undef ReadFile
#include "startup_stubs/xcrdapi.cpp"

namespace {

constexpr DWORD completion_timeout_ms = 15000;
constexpr unsigned piece = 4096;
constexpr unsigned base_count = 48;

using OpenFileFn = long(__stdcall*)(void*, const wchar_t*, const void*, void**);
using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using EnqueueSignalFn = void(__stdcall*)(void*, void*, std::uint64_t);
using SubmitFn = void(__stdcall*)(void*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
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

bool expect(bool condition, const char* what)
{
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", what);
    return condition;
}

}  // namespace

// A closed file's bytes are never served again, even when the next open gets the same handle value (the cache is keyed by a
// per-open number, and forget() drops the slots and queued reads and waits for a read in progress before the handle closes).
int check_forget()
{
    wchar_t temp_dir[MAX_PATH] = {}, path_a[MAX_PATH] = {}, path_b[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, temp_dir);
    std::swprintf(path_a, MAX_PATH, L"%lsx2w_hdd_forget_a_%lu.bin", temp_dir, GetCurrentProcessId());
    std::swprintf(path_b, MAX_PATH, L"%lsx2w_hdd_forget_b_%lu.bin", temp_dir, GetCurrentProcessId());
    std::vector<unsigned char> a(64 << 10), b(64 << 10);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<unsigned char>((i * 7 + 1) ^ 0x11);
        b[i] = static_cast<unsigned char>((i * 13 + 5) ^ 0xee);
    }
    const auto write_file = [](const wchar_t* path, const std::vector<unsigned char>& bytes) {
        HANDLE out = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        DWORD written = 0;
        const bool ok = out != INVALID_HANDLE_VALUE && WriteFile(out, bytes.data(), static_cast<DWORD>(bytes.size()), &written, nullptr) &&
                        written == bytes.size();
        if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
        return ok;
    };
    if (!write_file(path_a, a) || !write_file(path_b, b)) {
        std::fprintf(stderr, "cannot write the test files\n");
        return 1;
    }
    const auto open_read = [](const wchar_t* path) {
        return CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    };
    const auto planned = [](std::uintptr_t generation, HANDLE handle, std::uint64_t offset) {
        x2w_startup::PlannedRead r;
        r.file = generation;
        r.handle = reinterpret_cast<std::uintptr_t>(handle);
        r.offset = offset;
        r.size = 4096;
        return r;
    };
    const auto wait_prefetched = [](x2w_startup::ReadCache& cache, LONG64 bytes) {
        for (int i = 0; i < 500 && cache.counters().prefetched_bytes < bytes; ++i) Sleep(10);
        return cache.counters().prefetched_bytes >= bytes;
    };
    bool ok = true;
    x2w_startup::ReadCache cache(nullptr, 64 << 10, 4 << 20, 64 << 20, 20000);
    HANDLE handle_a = open_read(path_a);
    ok = expect(handle_a != INVALID_HANDLE_VALUE, "open file A") && ok;
    cache.submit({planned(1, handle_a, 0), planned(1, handle_a, 8192)});
    ok = expect(wait_prefetched(cache, 12288), "file A's two reads were prefetched") && ok;
    cache.forget(1);  // the file is about to close
    CloseHandle(handle_a);
    std::vector<unsigned char> got;
    ok = expect(!cache.take(1, 0, 4096, got, 0) && !cache.take(1, 8192, 4096, got, 0),
                "a closed file's prefetched bytes are still served") && ok;
    // A new open: possibly the very same handle value. Its bytes, not A's.
    HANDLE handle_b = open_read(path_b);
    std::printf("handle value reused by the next open: %s\n", handle_b == handle_a ? "yes" : "no");
    cache.submit({planned(2, handle_b, 0), planned(2, handle_b, 8192)});
    ok = expect(wait_prefetched(cache, 24576), "file B's two reads were prefetched") && ok;
    bool b_exact = cache.take(2, 0, 4096, got, 3000) && std::memcmp(got.data(), b.data(), 4096) == 0;
    b_exact = cache.take(2, 8192, 4096, got, 3000) && std::memcmp(got.data(), b.data() + 8192, 4096) == 0 && b_exact;
    ok = expect(b_exact, "the reads after the reopen return file B's bytes") && ok;
    // The closed generation takes no new reads either.
    const LONG64 before = cache.counters().prefetched_bytes;
    cache.submit({planned(1, handle_b, 0)});
    Sleep(100);
    ok = expect(cache.counters().prefetched_bytes == before && !cache.take(1, 0, 4096, got, 0), "a closed file's generation takes no new reads") && ok;
    CloseHandle(handle_b);
    DeleteFileW(path_a);
    DeleteFileW(path_b);
    if (!ok) return 1;
    std::puts("hdd forget: a closed file's bytes are never served; the reopen reads its own bytes");
    return 0;
}

int check_queue(int argc, char** argv, bool cold_control)
{
    if (argc == 2 && std::strcmp(argv[1], "forget") == 0) return check_forget();
    const bool deep = argc == 2 && std::strcmp(argv[1], "deep") == 0;  // 1500 scattered requests: the look-ahead window is deeper than 256 entries
    const bool cold = cold_control || (argc == 2 && std::strcmp(argv[1], "cold") == 0);
    slow_reader = cold;
    SetEnvironmentVariableA("X2W_STORAGE_HDD_ADAPTIVE", cold && !cold_control ? "1" : nullptr);
    const bool on = cold || deep || (argc == 2 && std::strcmp(argv[1], "on") == 0);
    if (!on && !(argc == 2 && std::strcmp(argv[1], "off") == 0)) {
        std::fprintf(stderr, "usage: startup_dstorage_hdd_queue on|off|deep|forget|cold|cold-control\n");
        return 2;
    }
    SetEnvironmentVariableA("X2W_STORAGE_HDD", on ? "1" : nullptr);  // read once, at the first use

    wchar_t temp_dir[MAX_PATH] = {}, data_path[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, temp_dir);
    std::swprintf(data_path, MAX_PATH, L"%lsx2w_hdd_queue_%lu.bin", temp_dir, GetCurrentProcessId());
    const unsigned request_count = deep ? 1500 : base_count;
    const unsigned stride = cold ? 192 * 1024 : deep ? 16384 : piece;  // deep: pieces 16 KiB apart, so only a deep window merges them
    std::vector<unsigned char> file_bytes(static_cast<std::size_t>(request_count) * stride);
    for (std::size_t i = 0; i < file_bytes.size(); ++i) file_bytes[i] = static_cast<unsigned char>((i * 131 + (i >> 8) + (i >> 16)) ^ 0x5a);
    {
        HANDLE out = CreateFileW(data_path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        DWORD written = 0;
        if (out == INVALID_HANDLE_VALUE || !WriteFile(out, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr) ||
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
    void* file_ptr = nullptr;
    if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, data_path, nullptr, &file_ptr) != 0 || !file_ptr) {
        std::fprintf(stderr, "cannot open the test file\n");
        return 1;
    }
    void *status_ptr = nullptr, *queue_ptr = nullptr;
    unsigned char description[0x28] = {};
    *reinterpret_cast<std::uint16_t*>(description + 8) = deep ? 4096 : 16;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, request_count, "Hdd", nullptr, &status_ptr) != 0 ||
        reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue_ptr) != 0) {
        std::fprintf(stderr, "cannot create the queue\n");
        return 1;
    }
    auto* status = static_cast<Object*>(status_ptr);
    auto* queue = static_cast<Object*>(queue_ptr);

    // Shuffled piece numbers (a fixed permutation: 5 is coprime with 48).
    std::vector<unsigned> slot_of(request_count);
    for (unsigned i = 0; i < request_count; ++i) slot_of[i] = (i * (deep ? 37 : 5) + 7) % request_count;
    std::vector<std::vector<unsigned char>> destination(request_count);
    if (cold) {
        // Prime the previous window with an actual delayed read of the same package.
        const HANDLE handle = static_cast<Object*>(file_ptr)->handle;
        std::vector<unsigned char> seed;
        LARGE_INTEGER begin{}, end{};
        QueryPerformanceCounter(&begin);
        const bool read = x2w_startup::read_storage_file(handle, 0, piece, seed);
        QueryPerformanceCounter(&end);
        if (!expect(read, "slow-reader seed failed")) return 1;
        note_disk_read(handle, 0, piece, end.QuadPart - begin.QuadPart, read);
    }
    const DiskStats disk_before = disk_snapshot();
    for (unsigned i = 0; i < request_count; ++i) {
        destination[i].assign(piece, 0xA5);
        alignas(8) unsigned char request[0x48] = {};
        *reinterpret_cast<u64*>(request + 0x00) = 0;  // raw read into memory
        *reinterpret_cast<unsigned char**>(request + 0x08) = destination[i].data();
        *reinterpret_cast<unsigned*>(request + 0x18) = piece;
        *reinterpret_cast<void**>(request + 0x20) = file_ptr;
        *reinterpret_cast<u64*>(request + 0x28) = static_cast<u64>(slot_of[i]) * stride;
        *reinterpret_cast<unsigned*>(request + 0x30) = piece;
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

    unsigned wrong = 0, bad_status = 0;
    for (unsigned i = 0; i < request_count; ++i) {
        const bool done = reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, i);
        const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, i);
        if (!done || hr != S_OK) ++bad_status;
        if (std::memcmp(destination[i].data(), file_bytes.data() + static_cast<std::size_t>(slot_of[i]) * stride, piece) != 0) ++wrong;
    }
    char what[200];
    std::snprintf(what, sizeof what, "%u of %u destinations differ from the file, %u bad statuses", wrong, request_count, bad_status);
    ok = expect(wrong == 0 && bad_status == 0, what) && ok;

    const DiskStats disk = disk_snapshot();
    const LONG reads = disk.reads - disk_before.reads;
    bool reduction_failed = false;
    if (cold) {
        const auto c = hdd_cache().counters();
        ok = expect(c.failed == 0 && disk.failed == disk_before.failed, "cold read failed") && ok;
        if (cold_control)
            ok = expect(reads == static_cast<LONG>(request_count), "legacy planner should issue one read per request") && ok;
        // The prefetch thread and the worker race, so how many small reads slip in unmerged varies between builds (14 of 48 in a
        // release run). The legacy planner (cold-control) always issues one read per request, 48, so half still separates them.
        std::snprintf(what, sizeof what, "cold disk: %ld calls for %u requests (want at most %u)",
                      reads, request_count, request_count / 2);
        reduction_failed = !expect(reads <= static_cast<LONG>(request_count / 2), what);
    } else if (deep) {
        // A window of 1024 entries (about 512 requests) closes the 16 KiB gaps between scattered pieces, so the file is read in
        // far fewer reads than requests; a window of 256 entries (about 128 requests) leaves most of them unmerged.
        const x2w_startup::ReadCache::Counters c = hdd_cache().counters();
        std::snprintf(what, sizeof what, "deep window: %ld file reads for %u requests (want under %u), %ld groups, failed %ld", reads, request_count, request_count / 3, c.groups, c.failed);
        ok = expect(reads < static_cast<LONG>(request_count / 3) && c.failed == 0, what) && ok;
    } else if (on) {
        const x2w_startup::ReadCache::Counters c = hdd_cache().counters();
        // Submit hands the worker its entries in steps, so the plan may be made in a few windows: what is fixed is that every
        // request but the first is served by the cache, the merged reads are far fewer than the requests, and every merged
        // read is one file read (plus the worker's own read of the first request).
        std::snprintf(what, sizeof what, "cache: %ld hits (want %u), %ld misses (want 1), %ld groups (want 1..8), reads saved %ld (want %u minus groups), failed %ld",
                      c.hits, request_count - 1, c.misses, c.groups, c.reads_saved, request_count - 1, c.failed);
        ok = expect(c.hits == static_cast<LONG>(request_count - 1) && c.misses == 1 && c.groups >= 1 && c.groups <= 8 &&
                        c.reads_saved == static_cast<LONG>(request_count - 1) - c.groups && c.failed == 0, what) && ok;
        std::snprintf(what, sizeof what, "file reads: %ld (want 1 + %ld groups: the first request and one read per merged group)", reads, c.groups);
        ok = expect(reads == 1 + c.groups, what) && ok;
    } else {
        std::snprintf(what, sizeof what, "mode off: %ld file reads (want %u, one per request)", reads, request_count);
        ok = expect(reads == static_cast<LONG>(request_count), what) && ok;
    }
    DeleteFileW(data_path);
    if (!ok) return 1;
    if (reduction_failed) return 3;
    std::printf("hdd mode %s: all %u requests returned the file's bytes; %ld file reads\n", on ? "on" : "off", request_count, reads);
    return 0;
}

int main(int argc, char** argv)
{
    const bool cold_control = argc == 2 && std::strcmp(argv[1], "cold-control") == 0;
    const int result = check_queue(argc, argv, cold_control);
    // WILL_FAIL accepts only a verified miss of the reduction target.
    return cold_control ? (result == 3 ? 1 : 0) : result;
}
