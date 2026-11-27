// Desktop XCRDAPI.dll stand-in for startup experiments (docs/startup-runtime.md).
//
// DStorageGetFactory returns a recording COM stand-in. Methods whose PC DirectStorage meaning matches
// the observed call shape are implemented on Win32 and logged as "x2w-dstorage"; every other vtable
// slot of every object logs "x2w-trap XCRDAPI.dll!<Object>::slot<N>" with its argument registers and
// terminates the process, so each run names the next method that needs an implementation.
// Exit code for an object slot: 0xE2570400 | (object base + slot), see Kind.
#define X2W_TRAP_MODULE "XCRDAPI.dll"
#define X2W_TRAP_MODULE_ID 4
#include "startup_stubs/trap.h"
#include "startup_stubs/profile_zone.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/inflate.h"
#include "startup_stubs/dstorage_error_record.h"
#include "startup_stubs/dstorage_file_read.h"
#include "startup_stubs/dstorage_read_cache.h"
#include "startup_stubs/bcpack_decoder.h"

#include <psapi.h>

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace {

using u64 = std::uint64_t;
using x2w_startup::StorageErrorRecord;
using x2w_startup::StorageRequest;
constexpr int slots = 32;
constexpr u64 memory_source_option = 1ull << 22;

enum Kind : unsigned { factory = 0x20, file = 0x40, status = 0x60, queue = 0x80 };
const char* kind_name(unsigned kind)
{
    switch (kind) {
    case factory: return "Factory";
    case file: return "File";
    case status: return "StatusArray";
    case queue: return "Queue";
    default: return "Object";
    }
}

struct Object {
    void** vtable;
    unsigned kind;
    LONG refs;
    HANDLE handle;      // File
    unsigned long long generation = 0;  // File: a number unique to this open (handle values are reused); the HDD cache's key
    unsigned capacity;  // StatusArray, Queue
    LONG* results;      // StatusArray: one HRESULT per index, E_PENDING until a status is enqueued
    LONG pending;       // Queue: worst HRESULT of the requests since the last status
    struct QueueWork* work = nullptr;  // Queue: entries waiting for Submit and the worker running them
    struct Unrecorded* unrecorded_head = nullptr;  // Queue: entries no queue storage could hold, see push()
    struct Unrecorded* unrecorded_tail = nullptr;
};

struct FileObject : Object {
    wchar_t filename[MAX_PATH]{};
};

struct QueueObject : Object {
    // Independent of QueueWork so a queue can report failures even when it has no worker.
    std::mutex error_mutex;
    StorageErrorRecord error_record = x2w_startup::empty_error_record();
    HANDLE error_event = nullptr;
};

// Live stand-in objects by kind, and running queue workers. The lifetime fixture reads them through
// x2w_xcrdapi_live_objects and x2w_xcrdapi_live_workers to prove that a final Release frees the
// object and that the queue's worker has left (#2788).
LONG live_by_kind[4];
LONG live_workers;

// Test-only mutant (#2788): X2W_TEST_XCRDAPI_LIFETIME_MUTANT=naive deletes at final Release without
// stopping the worker or retaining the objects a queue entry uses. The fixture must reject it.
bool naive_lifetime_mutant() noexcept
{
    static const bool naive = [] {
        char text[16];
        const DWORD length = GetEnvironmentVariableA("X2W_TEST_XCRDAPI_LIFETIME_MUTANT", text, sizeof(text));
        return length == 5 && std::memcmp(text, "naive", 5) == 0;
    }();
    return naive;
}

// A queue entry as enqueued: a request (its 0x48 bytes copied, and a snapshot of its destination page
// tokens), a status entry or a fence signal. Entries run in order on the queue's worker after Submit,
// like DirectStorage; a destination page token that was 0 at enqueue is read again at the entry's turn
// (titles fill those in after EnqueueRequest).
//
// An Entry is default-constructed, moved and move-assigned on the Enqueue/Submit path, where an
// allocation may fail and nothing may escape. Those operations are implicitly noexcept, so no member
// may allocate in them: std::vector does in MSVC's debug iterator mode (its default and move
// constructors allocate a container proxy), and a bad_alloc there is std::terminate. The page tokens
// are therefore a plain array, allocated once with new (std::nothrow) at EnqueueRequest.
struct Entry {
    enum Type { request, status_entry, signal } type;
    unsigned char bytes[sizeof(StorageRequest)];
    std::unique_ptr<ULONG_PTR[]> tokens;
    std::size_t token_count = 0;
    Object* array;
    Object* source;  // request: the file the entry reads, retained while the entry is pending
    unsigned index;
    void* fence;
    u64 value;
    u64 id = 0;
    u64 submission_id = 0;
    u64 after = 0;  // failing: entries the worker must have finished before this one completes
};
struct CachedStream {
    u64 source = 0;
    u64 base_offset = 0;
    unsigned compressed_size = 0;
    unsigned chunk_size = 0;
    std::vector<u64> continuation_ids;
    std::vector<unsigned char> data;
};

struct QueueWork {
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Entry> waiting, submitted;
    std::deque<Entry> failing;   // snapshot of waiting entries a Submit could not move, completed in order
    struct Unrecorded* deferred_head = nullptr;  // snapshot of unrecorded entries, in order; guarded by mutex
    struct Unrecorded* deferred_tail = nullptr;
    bool completing = false;     // a thread owns completion of snapshot entries, so they stay in order; guarded by mutex
    bool stopping = false;       // final Release asked the worker to leave; guarded by mutex
    bool destroy_on_worker = false;  // the final Release ran on the worker itself; guarded by mutex
    HANDLE thread = nullptr;     // kept for the join; closed by the releasing thread, guarded by the setup mutex
    DWORD thread_id = 0;         // guarded by the setup mutex
    std::condition_variable drained;  // signaled when the worker finishes an entry
    u64 submitted_total = 0;          // entries ever moved to `submitted`, guarded by mutex
    u64 finished_total = 0;           // entries the worker has finished running, guarded by mutex
    u64 prefetch_next = 0;            // HDD mode: the first entry (by finished_total + 1 + position) not yet planned, guarded by mutex
    bool worker_started = false;  // guarded by the setup mutex
    u64 next_entry_id = 0;
    u64 next_submission_id = 0;
    static constexpr std::size_t max_cached_streams = 16;
    std::deque<CachedStream> cached_streams;

    void put_stream(u64 source, u64 base_offset, unsigned compressed_size, unsigned chunk_size,
                    std::vector<u64> continuation_ids, std::vector<unsigned char> data)
    {
        for (auto it = cached_streams.begin(); it != cached_streams.end(); ++it) {
            if (it->source == source && it->base_offset == base_offset && it->compressed_size == compressed_size) {
                cached_streams.erase(it);
                break;
            }
        }
        if (cached_streams.size() >= max_cached_streams) {
            cached_streams.pop_front();
        }
        cached_streams.push_back({source, base_offset, compressed_size, chunk_size,
                                  std::move(continuation_ids), std::move(data)});
    }
};

static bool has_zlib_header(const unsigned char* bytes, std::size_t size)
{
    if (!bytes || size < 2) return false;
    const unsigned char cmf = bytes[0];
    const unsigned char flg = bytes[1];
    if ((cmf & 0x0F) != 8) return false;
    if ((cmf >> 4) > 7) return false;
    if ((static_cast<unsigned>(cmf) * 256 + flg) % 31 != 0) return false;
    return true;
}

// True when every page of [start, end) is resident and plain read-write: valid in the working set, PAGE_READWRITE
// or PAGE_EXECUTE_READWRITE, no guard bit. One QueryWorkingSetEx per 512 pages, so the cost follows the
// destination's size, not the region's. Reads attributes only; no page is touched.
static bool resident_read_write(std::uintptr_t start, std::uintptr_t end)
{
    PSAPI_WORKING_SET_EX_INFORMATION pages[512];
    std::uintptr_t page = start & ~std::uintptr_t{0xFFF};
    while (page < end) {
        DWORD count = 0;
        for (; count < 512 && page < end; ++count, page += 0x1000) pages[count].VirtualAddress = reinterpret_cast<void*>(page);
        if (!QueryWorkingSetEx(GetCurrentProcess(), pages, count * sizeof(pages[0]))) return false;
        for (DWORD i = 0; i < count; ++i) {
            const auto& page_info = pages[i].VirtualAttributes;
            const DWORD protect = static_cast<DWORD>(page_info.Win32Protection);
            if (!page_info.Valid || page_info.Bad || (protect & PAGE_GUARD) ||
                ((protect & 0xFF) != PAGE_READWRITE && (protect & 0xFF) != PAGE_EXECUTE_READWRITE))
                return false;
        }
    }
    return true;
}

// True when every byte of [destination, destination + bytes) is committed and writable, with no guard or
// no-access page in between. Nothing is written or touched. Resident read-write pages are accepted from
// their working-set attributes; any other page (not resident, copy-on-write, guard, no access) sends the
// destination to the VirtualQuery walk, which judges it region by region. That walk is slow on the title's
// large regions: each query scans to the end of the region (#2684).
static bool destination_writable(const void* destination, std::size_t bytes)
{
    if (bytes == 0) return true;
    std::uintptr_t cursor = reinterpret_cast<std::uintptr_t>(destination);
    if (!destination || bytes > UINTPTR_MAX - cursor) return false;
    x2w_local::commit_lazy(destination, bytes);  // lazily mapped physical pages (X2W_XMEM_LAZY_COMMIT)
    const std::uintptr_t end = cursor + bytes;
    if (resident_read_write(cursor, end)) return true;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) || info.State != MEM_COMMIT)
            return false;
        if (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) return false;
        const DWORD protect = info.Protect & 0xFF;
        if (protect != PAGE_READWRITE && protect != PAGE_WRITECOPY && protect != PAGE_EXECUTE_READWRITE &&
            protect != PAGE_EXECUTE_WRITECOPY)
            return false;
        const std::uintptr_t region_end = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
        if (region_end <= cursor) return false;
        cursor = region_end;
    }
    return true;
}

// Return the exact queued requests that continue this stream in this submission, in queue order.
// Status entries and requests for unrelated files are skipped. In the same file, any non-matching
// size, non-consecutive offset, or submission boundary ends the continuation chain (Refs #1136, #1144, #1198, #1214).
static std::vector<u64> following_chunk_ids(Object* self, u64 current_submission_id, u64 source, u64 offset,
                                            unsigned source_size, unsigned destination_size, std::size_t total_size)
{
    std::vector<u64> ids;
    if (!self || !self->work) return ids;
    QueueWork& work = *self->work;
    const std::lock_guard guard(work.mutex);
    const unsigned step = destination_size ? destination_size : 0x10000;
    u64 expected = offset + step;
    if (expected < offset) return ids;

    for (const auto& e : work.submitted) {
        if (e.submission_id != current_submission_id) {
            break;
        }
        if (e.type != Entry::request) continue;
        const u64 e_options = *reinterpret_cast<const u64*>(e.bytes);
        const u64 e_source = *reinterpret_cast<const u64*>(e.bytes + 0x20);
        // Skip requests for other files (interleaving across unrelated files within one submission).
        if (e_source != source) {
            continue;
        }
        // For the same file, requests must strictly be compressed file requests continuing this sequence.
        if (!(e_options & 1) || (e_options & 0x400000)) {
            break;
        }
        const u64 e_offset = *reinterpret_cast<const u64*>(e.bytes + 0x28);
        const unsigned e_source_size = *reinterpret_cast<const unsigned*>(e.bytes + 0x30);
        if (e_source_size != source_size || e_offset != expected) {
            break;
        }
        const u64 delta = expected - offset;
        if (delta >= total_size || delta > UINT32_MAX) {
            break;
        }
        ids.push_back(e.id);
        if (expected > UINT64_MAX - step) {
            break;
        }
        expected += step;
    }
    return ids;
}

void log(const char* format, ...)
{
    char line[1024];
    va_list args;
    va_start(args, format);
    int n = std::vsnprintf(line, sizeof(line) - 2, format, args);
    va_end(args);
    if (n < 0) return;
    if (n > static_cast<int>(sizeof(line)) - 3) n = sizeof(line) - 3;
    line[n++] = '\r';
    line[n++] = '\n';
    x2w_trap::write_all(GetStdHandle(STD_ERROR_HANDLE), line, static_cast<DWORD>(n));
    x2w_trap::append_log(line, static_cast<DWORD>(n), false);
}

void guid_text(const void* guid, char* out, std::size_t size)
{
    if (!guid) {
        std::snprintf(out, size, "null");
        return;
    }
    const auto* g = static_cast<const GUID*>(guid);
    std::snprintf(out, size, "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g->Data1, g->Data2, g->Data3,
                  g->Data4[0], g->Data4[1], g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5], g->Data4[6],
                  g->Data4[7]);
}

template <int N>
__declspec(noinline) u64 trap_slot(Object* self, u64 rdx, u64 r8, u64 r9)
{
    char name[64];
    std::snprintf(name, sizeof(name), "%s::slot%d", kind_name(self->kind), N);
    x2w_trap::fire(X2W_TRAP_MODULE, X2W_TRAP_MODULE_ID, name, (self->kind + N) & 0xFF,
                   reinterpret_cast<u64>(self), rdx, r8, r9, _ReturnAddress());
}

template <std::size_t... I>
void fill_traps(void** table, std::index_sequence<I...>)
{
    ((table[I] = reinterpret_cast<void*>(&trap_slot<static_cast<int>(I)>)), ...);
}

// IUnknown, shared by every object.
long __stdcall query_interface(Object* self, const GUID* riid, void** out)
{
    char text[48];
    guid_text(riid, text, sizeof(text));
    log("x2w-dstorage %s::QueryInterface riid=%s -> self", kind_name(self->kind), text);
    if (!out) return E_POINTER;
    InterlockedIncrement(&self->refs);
    *out = self;
    return S_OK;
}
unsigned long __stdcall add_ref(Object* self) { return static_cast<unsigned long>(InterlockedIncrement(&self->refs)); }

void destroy_object(Object* self) noexcept;

// HDD mode: drop the cache's slots and queued reads of this open, and wait for a read in progress on it, before its handle is
// closed (defined with the HDD cache below).
void hdd_forget_file(unsigned long long generation) noexcept;

// Final Release of a queue: stop the worker, wait for it to leave, then drop the entries that never
// ran. Returns true when the worker owns the destruction (Release ran on the worker itself).
// Defined after the entry helpers below.
bool shutdown_queue(Object* self);

unsigned long __stdcall release(Object* self)
{
    const auto refs = static_cast<unsigned long>(InterlockedDecrement(&self->refs));
    if (refs != 0) return refs;
    // The factory is a process singleton: it stays alive for the life of the process (#2788).
    if (self->kind == factory) return 0;
    if (self->kind == queue && !naive_lifetime_mutant() && shutdown_queue(self)) return 0;
    destroy_object(self);
    return 0;
}

void** make_table(unsigned kind);

Object* make(unsigned kind)
{
    Object* object;
    if (kind == queue) object = new QueueObject{};
    else if (kind == file) object = new FileObject{};
    else object = new Object{};
    object->vtable = make_table(kind);
    object->kind = kind;
    object->refs = 1;
    object->handle = INVALID_HANDLE_VALUE;
    InterlockedIncrement(&live_by_kind[kind / 0x20 - 1]);
    return object;
}

void destroy_object(Object* self) noexcept
{
    const unsigned index = self->kind / 0x20 - 1;
    if (self->kind == queue) {
        auto* object = static_cast<QueueObject*>(self);
        {
            const std::lock_guard guard(object->error_mutex);
            if (object->error_event) CloseHandle(std::exchange(object->error_event, nullptr));
        }
        delete object;
    } else if (self->kind == file) {
        auto* object = static_cast<FileObject*>(self);
        if (object->handle != INVALID_HANDLE_VALUE) {
            hdd_forget_file(object->generation);
            CloseHandle(std::exchange(object->handle, INVALID_HANDLE_VALUE));
        }
        delete object;
    } else if (self->kind == status) {
        delete[] self->results;
        delete self;
    } else {
        return;  // the factory singleton is never freed
    }
    InterlockedDecrement(&live_by_kind[index]);
}

// The status array an entry completes and the file a request reads must outlive the guest's Release
// while the entry is pending: the entry owns one reference on each until it is consumed (#2788).
void retain_entry_object(Object* object) noexcept
{
    if (object && !naive_lifetime_mutant()) InterlockedIncrement(&object->refs);
}

void release_entry_object(Object* object) noexcept
{
    if (object && !naive_lifetime_mutant()) release(object);
}

void release_entry(Entry& entry) noexcept
{
    release_entry_object(entry.source);
    entry.source = nullptr;
    release_entry_object(entry.array);
    entry.array = nullptr;
}

// Factory slot 4, PC IDStorageFactory::OpenFile(PCWSTR path, REFIID riid, void** ppv).
long __stdcall factory_open_file(Object*, const wchar_t* path, const GUID* riid, void** out)
{
    char id[48];
    guid_text(riid, id, sizeof(id));
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!path) {
        log("x2w-dstorage Factory::OpenFile path=(null) riid=%s -> E_INVALIDARG", id);
        return E_INVALIDARG;
    }
    HANDLE handle = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
    const DWORD error = handle == INVALID_HANDLE_VALUE ? GetLastError() : 0;
    log("x2w-dstorage Factory::OpenFile path=\"%ls\" riid=%s -> %s (err %lu)", path ? path : L"(null)", id,
        handle == INVALID_HANDLE_VALUE ? "fail" : "ok", error);
    if (handle == INVALID_HANDLE_VALUE) return HRESULT_FROM_WIN32(error);
    auto* object = static_cast<FileObject*>(make(file));
    object->handle = handle;
    static LONG64 opens;
    object->generation = static_cast<unsigned long long>(InterlockedIncrement64(&opens));
    wcsncpy_s(object->filename, path, _TRUNCATE);
    *out = object;
    return S_OK;
}

// Factory slot 5, PC IDStorageFactory::CreateStatusArray(UINT32 capacity, PCSTR name, REFIID, void**).
long __stdcall factory_create_status_array(Object*, unsigned capacity, const char* name, const GUID* riid, void** out)
{
    char id[48];
    guid_text(riid, id, sizeof(id));
    log("x2w-dstorage Factory::CreateStatusArray capacity=%u name=\"%s\" riid=%s", capacity, name ? name : "", id);
    if (!out) return E_POINTER;
    Object* object = make(status);
    object->capacity = capacity;
    object->results = new LONG[capacity ? capacity : 1];
    // A new slot is complete with S_OK until EnqueueStatus marks it pending (#1843), as in Microsoft's
    // Win32 implementation of this API (Xbox-GDK-Samples, MIT: SimpleDirectStorageCombo/DirectStorageWin32/
    // DirectStorageWin32WrapperStatusArray.cpp). A caller may poll IsComplete before a slot's first use.
    for (unsigned i = 0; i < capacity; ++i) object->results[i] = S_OK;
    *out = object;
    return S_OK;
}

// Factory slot 8: called once at startup with 32 MiB (staging buffer size in PC terms).
long __stdcall factory_slot8(Object*, unsigned value)
{
    log("x2w-dstorage Factory::slot8 value=0x%x -> S_OK", value);
    return S_OK;
}

// Factory slot 10: (descriptor*, riid, ppv); creates a queue (the PC CreateQueue shape, wider descriptor).
long __stdcall factory_create_queue(Object*, const unsigned char* desc, const GUID* riid, void** out)
{
    char id[48];
    guid_text(riid, id, sizeof(id));
    char bytes[3 * 40 + 1] = "";
    if (desc)
        for (int i = 0; i < 40; ++i) std::snprintf(bytes + 3 * i, 4, "%02x ", desc[i]);
    const char* name = desc ? *reinterpret_cast<const char* const*>(desc + 0x10) : nullptr;
    log("x2w-dstorage Factory::CreateQueue desc=[%s] name=\"%s\" riid=%s", bytes,
        name && x2w_local::readable(name, 1) ? name : "?", id);
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!desc) return E_INVALIDARG;
    Object* object = make(queue);
    object->capacity = *reinterpret_cast<const std::uint16_t*>(desc + 8);
    *out = object;
    return S_OK;
}

// File slot 3, PC IDStorageFile::Close().
void __stdcall file_close(Object* self)
{
    log("x2w-dstorage File::Close handle=%p", self->handle);
    if (self->handle != INVALID_HANDLE_VALUE) {
        hdd_forget_file(self->generation);
        CloseHandle(self->handle);
    }
    self->handle = INVALID_HANDLE_VALUE;
}

// File slot 4, PC IDStorageFile::GetFileInformation(BY_HANDLE_FILE_INFORMATION*).
long __stdcall file_information(Object* self, BY_HANDLE_FILE_INFORMATION* info)
{
    const BOOL ok = info && GetFileInformationByHandle(self->handle, info);
    const DWORD error = ok ? 0 : GetLastError();  // before log(), whose Win32 calls overwrite it
    log("x2w-dstorage File::GetFileInformation -> %s size=0x%llx", ok ? "ok" : "fail",
        ok ? (static_cast<unsigned long long>(info->nFileSizeHigh) << 32) | info->nFileSizeLow : 0ull);
    return ok ? S_OK : HRESULT_FROM_WIN32(error);
}

// Queue slot 3, EnqueueRequest(request*) by the PC numbering. Request layout from the title's builder
// (a hypothesis from its two code paths): +0x00 options (0x400000: memory source; low byte: compression
// format, 1 seen), +0x08 destination pointer, +0x18 destination size (u32), +0x20 source (File object
// or memory pointer), +0x28 source offset, +0x30 source size (u32). Uncompressed requests are executed
// synchronously; compressed ones are logged and left undone. The first requests are logged in full.
// requests_done counts the uncompressed copies only (the "raw" count of the status lines); a compressed request
// is counted in requests_compressed ("inflated"), or in requests_bcpack_blank when its BCPack data was not decoded.
LONG requests_done, requests_compressed, requests_failed, requests_bcpack_blank;

// Loading progress (the "x2w-dstorage progress" line every heartbeat): requests enqueued against requests completed
// (done, inflated, blank or failed) give the queue depth; a ring of enqueue ticks, indexed by sequence number, names the
// oldest request still pending (queues run in order, so the request numbered "completed" is the oldest; with several
// queues the age is approximate). Bytes: read = file bytes consumed (the compressed bytes of a compressed request, the
// copied bytes of an uncompressed one); inflated = the bytes a compressed request wrote. Fixed counters, no allocation.
LONG requests_enqueued;
LONG64 bytes_read_total, bytes_inflated_total;
constexpr unsigned enqueue_ring_size = 16384;  // a power of two
volatile LONG enqueue_ring_tick[enqueue_ring_size];  // GetTickCount() at the enqueue of request number i % size
void note_enqueued() noexcept
{
    const LONG number = InterlockedIncrement(&requests_enqueued) - 1;
    enqueue_ring_tick[static_cast<unsigned>(number) & (enqueue_ring_size - 1)] = static_cast<LONG>(GetTickCount());
}
// The progress line: queue depth, the oldest pending request's age, and MB/s read and inflated over the last
// `seconds`. `retired` is the sum of every request outcome (written + failed): the queue depth is enqueued - retired.
// `written` is the requests that really wrote their destination (raw + inflated + bcpack-blank) and `failed` the ones
// that did not, so the line tells a run that retires fast by failing from one that loads. `now` is GetTickCount().
int format_progress(char* out, std::size_t size, int second, LONG enqueued, LONG written, LONG failed, LONG64 read_delta,
                    LONG64 inflated_delta, LONG64 read_total, LONG64 inflated_total, double seconds, DWORD now) noexcept
{
    const LONG completed = written + failed;   // retired
    const LONG pending = enqueued > completed ? enqueued - completed : 0;
    char oldest[48] = "none";
    if (pending > 0 && pending <= static_cast<LONG>(enqueue_ring_size))
        std::snprintf(oldest, sizeof(oldest), "%lu ms",
                      static_cast<unsigned long>(now - static_cast<DWORD>(
                          enqueue_ring_tick[static_cast<unsigned>(completed) & (enqueue_ring_size - 1)])));
    else if (pending > 0)
        std::snprintf(oldest, sizeof(oldest), "unknown, more than %u pending", enqueue_ring_size);
    return std::snprintf(out, size,
                         "x2w-dstorage progress %d s: pending %ld (oldest %s), read %.1f MB/s, inflated %.1f MB/s, "
                         "enqueued %ld retired %ld (written %ld, failed %ld), total read %lld KiB inflated %lld KiB", second, pending, oldest,
                         static_cast<double>(read_delta) / 1048576.0 / seconds,
                         static_cast<double>(inflated_delta) / 1048576.0 / seconds, enqueued, completed, written, failed,
                         static_cast<long long>(read_total >> 10), static_cast<long long>(inflated_total >> 10));
}
// The disk side of loading (the title package can sit on a hard disk, where seeks, not bandwidth, set the speed):
// time spent inside the file reads (QueryPerformanceCounter ticks), their sizes in six buckets, and whether each read
// continues the previous one of the same file (sequential), jumps forward (still in offset order), jumps back, or is
// the first of its file. One short lock per read; the map of last end offsets is cleared past 256 files.
constexpr unsigned disk_size_buckets = 6;
const unsigned long long disk_bucket_limit[disk_size_buckets - 1] = {4096, 16384, 65536, 262144, 1048576};
struct DiskStats {
    LONG64 ticks = 0, bytes = 0;
    LONG reads = 0, failed = 0;
    LONG sizes[disk_size_buckets] = {};
    LONG sequential = 0, forward = 0, backward = 0, first = 0;
};
DiskStats disk_stats;
std::mutex disk_mutex;
std::unordered_map<HANDLE, unsigned long long> disk_last_end;
x2w_startup::ReadLatencyWindow disk_latency;

bool adaptive_hdd_reads() noexcept
{
    static const bool on = [] {
        char text[8];
        const DWORD length = GetEnvironmentVariableA("X2W_STORAGE_HDD_ADAPTIVE", text, sizeof(text));
        return length == 1 && text[0] == '1';
    }();
    return on;
}

LONG64 disk_frequency() noexcept
{
    static const LONG64 frequency = [] {
        LARGE_INTEGER value{};
        QueryPerformanceFrequency(&value);
        return value.QuadPart;
    }();
    return frequency;
}

bool slow_disk_window()
{
    if (!adaptive_hdd_reads()) return false;
    const std::lock_guard guard(disk_mutex);
    return disk_latency.slow(disk_frequency());
}

void note_disk_read(HANDLE file, unsigned long long offset, unsigned long long count, LONG64 ticks, bool ok) noexcept
{
    try {
        const std::lock_guard guard(disk_mutex);
        DiskStats& d = disk_stats;
        d.ticks += ticks;
        if (!ok) {
            ++d.failed;
            return;
        }
        if (adaptive_hdd_reads()) disk_latency.record(ticks);
        d.bytes += static_cast<LONG64>(count);
        ++d.reads;
        unsigned bucket = 0;
        while (bucket < disk_size_buckets - 1 && count > disk_bucket_limit[bucket]) ++bucket;
        ++d.sizes[bucket];
        if (disk_last_end.size() > 256) disk_last_end.clear();
        const auto found = disk_last_end.find(file);
        if (found == disk_last_end.end()) ++d.first;
        else if (offset == found->second) ++d.sequential;
        else if (offset > found->second) ++d.forward;
        else ++d.backward;
        disk_last_end[file] = offset + count;
    } catch (...) {
    }
}
DiskStats disk_snapshot()
{
    const std::lock_guard guard(disk_mutex);
    return disk_stats;
}

// HDD mode (X2W_STORAGE_HDD=1, off by default): look ahead 1024 queue entries and
// prefetch sorted merged ranges. The worker processes entries in submission order and
// reads misses directly. The usual limits are a 64 KiB gap and a 4 MiB merged span.
// The cache holds at most 128 MiB and drops unused entries after 20 s.
bool hdd_mode() noexcept
{
    static const bool on = [] {
        char text[8];
        const DWORD length = GetEnvironmentVariableA("X2W_STORAGE_HDD", text, sizeof(text));
        return length == 1 && text[0] == '1';
    }();
    return on;
}
constexpr std::size_t hdd_look_ahead_entries = 1024;  // queue entries (requests, status entries and fences) planned ahead
x2w_startup::ReadCache& hdd_cache()
{
    // Never destroyed: the prefetch thread can outlive the static destructors at process exit.
    static x2w_startup::ReadCache* const cache = new x2w_startup::ReadCache(&note_disk_read, 64 << 10, 4 << 20, 128 << 20, 20000);
    return *cache;
}
void hdd_forget_file(unsigned long long generation) noexcept
{
    if (hdd_mode()) hdd_cache().forget(static_cast<std::uintptr_t>(generation));
}
// Called with the queue mutex held, right after the worker took an entry: plan the file reads of the entries still waiting.
void hdd_look_ahead(QueueWork& work)
{
    // The entry the worker just took has sequence number finished_total; the entry at position i of `submitted` has
    // finished_total + 1 + i. Plan the window's entries that earlier calls did not plan.
    const u64 first = work.finished_total + 1;
    const std::size_t window = work.submitted.size() < hdd_look_ahead_entries ? work.submitted.size() : hdd_look_ahead_entries;
    const std::size_t begin = work.prefetch_next > first ? static_cast<std::size_t>(work.prefetch_next - first) : 0;
    // Plan again only when less than half a window is planned ahead (or nothing is left to plan), so that each plan
    // covers many new entries and merges well, instead of one new request per worker step.
    if (begin >= window || begin >= hdd_look_ahead_entries / 2) return;
    work.prefetch_next = first + window;
    std::vector<x2w_startup::PlannedRead> reads;
    for (std::size_t i = begin; i < window; ++i) {
        const Entry& e = work.submitted[i];
        if (e.type != Entry::request || !e.source || e.source->kind != file || e.source->handle == INVALID_HANDLE_VALUE) continue;
        const u64 options = *reinterpret_cast<const u64*>(e.bytes);
        if (options & memory_source_option) continue;
        x2w_startup::PlannedRead r;
        r.file = static_cast<std::uintptr_t>(e.source->generation);
        r.handle = reinterpret_cast<std::uintptr_t>(e.source->handle);
        r.offset = *reinterpret_cast<const u64*>(e.bytes + 0x28);
        r.size = *reinterpret_cast<const unsigned*>(e.bytes + 0x30);
        r.index = i;
        reads.push_back(r);
    }
    if (!reads.empty()) hdd_cache().submit(reads, slow_disk_window());
}
// Include failed attempts in both the elapsed time and the denominator.
double disk_mean_ms(const DiskStats& stats, double frequency) noexcept
{
    const double attempts = static_cast<double>(stats.reads) + stats.failed;
    return attempts > 0 && frequency > 0 ? 1000.0 * stats.ticks / frequency / attempts : 0.0;
}

void log_disk_summary()
{
    DiskStats stats;
    {
        // DLL shutdown may run after another thread was stopped inside note_disk_read.
        const std::unique_lock guard(disk_mutex, std::try_to_lock);
        if (!guard.owns_lock()) return;
        stats = disk_stats;
    }
    log("x2w-dstorage disk summary: mean %.6f ms/read since start", disk_mean_ms(stats, static_cast<double>(disk_frequency())));
}

// The disk line: MB/s while reading and the share of the window spent in reads (high share and low MB/s = seek-bound),
// read sizes (<=4K <=16K <=64K <=256K <=1M >1M) and the offset order of the reads, all since the start.
int format_disk(char* out, std::size_t size, int second, const DiskStats& now, const DiskStats& before, double seconds,
                double ticks_per_second) noexcept
{
    const double busy = static_cast<double>(now.ticks - before.ticks) / ticks_per_second;
    const double megabytes = static_cast<double>(now.bytes - before.bytes) / 1048576.0;
    return std::snprintf(out, size,
                         "x2w-dstorage disk %d s: %ld reads (+%ld), %.1f MB/s while reading, reading %.0f%% of the window; "
                         "read sizes <=4K %ld <=16K %ld <=64K %ld <=256K %ld <=1M %ld >1M %ld; offsets sequential %ld "
                         "forward %ld backward %ld first %ld; read failures %ld; mean %.6f ms/read since start",
                         second, now.reads, now.reads - before.reads, busy > 0 ? megabytes / busy : 0.0,
                         seconds > 0 ? 100.0 * busy / seconds : 0.0, now.sizes[0], now.sizes[1], now.sizes[2], now.sizes[3],
                         now.sizes[4], now.sizes[5], now.sequential, now.forward, now.backward, now.first, now.failed,
                         disk_mean_ms(now, ticks_per_second));
}
void note_written(bool compressed, std::size_t source_bytes, std::size_t written_bytes) noexcept
{
    InterlockedExchangeAdd64(&bytes_read_total, static_cast<LONG64>(compressed ? source_bytes : written_bytes));
    if (compressed) InterlockedExchangeAdd64(&bytes_inflated_total, static_cast<LONG64>(written_bytes));
}

// Failed requests by cause, for the heartbeat (#1198). Each failure site names its cause and
// count_failure adds it to requests_failed as well, so the causes add up to that total. "other" holds
// the failures without a cause of their own: unknown option bits, page grants and page writes, a page
// array unreadable at run, an entry that threw, an entry never started. Fixed counters, no allocation.
enum class StorageFailure : unsigned {
    read_at_or_past_eof, read_crossing_eof, read_other, destination_capacity, destination_not_writable,
    zlib, bcpack, other, count
};
using StorageFailureCounts = LONG[static_cast<unsigned>(StorageFailure::count)];
StorageFailureCounts storage_failures;

void count_failure(StorageFailure cause, Object* self = nullptr, const unsigned char* bytes = nullptr,
                   HRESULT result = E_FAIL) noexcept
{
    InterlockedIncrement(&storage_failures[static_cast<unsigned>(cause)]);
    InterlockedIncrement(&requests_failed);
    if (!self || !bytes) return;

    auto& object = *static_cast<QueueObject*>(self);
    const std::lock_guard guard(object.error_mutex);
    auto& record = object.error_record;
    const bool first = record.failure_count == 0;
    if (record.failure_count != UINT32_MAX) ++record.failure_count;
    if (!first) return;

    StorageRequest request;
    std::memcpy(&request, bytes, sizeof request);
    auto& failure = record.first_failure;
    failure.result = result;
    failure.command_type = x2w_startup::command_type_request;
    failure.options = request.options;
    failure.destination = request.destination;
    failure.destination_size = request.destination_size;
    if (request.options & memory_source_option) {
        failure.source = request.source;
    } else {
        const auto* source = static_cast<const FileObject*>(request.source);
        if (x2w_local::readable(source, sizeof(*source)) && source->kind == file)
            std::memcpy(failure.filename, source->filename, sizeof failure.filename);
        failure.file_offset = request.file_offset;
    }
    failure.source_size = request.source_size;
    failure.cancellation_tag = request.cancellation_tag;
    const std::size_t name_bytes = x2w_local::probe_readable(reinterpret_cast<std::uintptr_t>(request.name),
                                                           sizeof failure.name - 1);
    for (std::size_t i = 0; i < name_bytes && request.name[i]; ++i) failure.name[i] = request.name[i];
    if (object.error_event) SetEvent(object.error_event);
}

// The cause of a refused file read, from its range against the file size. No end offset is formed, so
// neither an offset nor a count near 2^64 can wrap: at or past the end is offset >= size, crossing it
// is count > size - offset (evaluated only when offset < size).
StorageFailure read_failure_cause(bool size_known, u64 file_size, u64 offset, u64 count) noexcept
{
    if (!size_known) return StorageFailure::read_other;
    if (offset >= file_size) return StorageFailure::read_at_or_past_eof;
    if (count > file_size - offset) return StorageFailure::read_crossing_eof;
    return StorageFailure::read_other;
}

StorageFailure read_failure_cause(const Object* file_object, u64 offset, u64 count) noexcept
{
    LARGE_INTEGER length{};
    const bool known = file_object && file_object->kind == file && file_object->handle != INVALID_HANDLE_VALUE &&
                       GetFileSizeEx(file_object->handle, &length) && length.QuadPart >= 0;
    return read_failure_cause(known, known ? static_cast<u64>(length.QuadPart) : 0, offset, count);
}

// The status HRESULT a failed request reports, by cause. A destination too small for the data and a
// read at or past the end of the file are told apart; every other cause stays E_FAIL.
HRESULT failure_result(StorageFailure cause) noexcept
{
    switch (cause) {
    case StorageFailure::destination_capacity: return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    case StorageFailure::read_at_or_past_eof:
    case StorageFailure::read_crossing_eof: return HRESULT_FROM_WIN32(ERROR_HANDLE_EOF);
    default: return E_FAIL;
    }
}

// The heartbeat's line of failure causes. It is a line of its own after the totals line, whose text
// stays as it was; it carries no "inflated=", "options=" or "commit peak=" text, which
// tools/startup/live_view.py, tools/startup/triage_run.py and tests/startup_heartbeat_check.py match.
LONG sample_failure_counts(StorageFailureCounts& counts) noexcept
{
    ULONG total = 0;
    for (unsigned i = 0; i < static_cast<unsigned>(StorageFailure::count); ++i) {
        counts[i] = InterlockedCompareExchange(&storage_failures[i], 0, 0);
        total += static_cast<ULONG>(counts[i]);
    }
    return static_cast<LONG>(total);
}

// The counters of the status lines: "raw=R inflated=I failed=F bcpack-blank=B completed=C". "inflated=%ld
// failed=%ld" stays byte-for-byte what it was, because tools/startup/live_view.py parses it. completed is every
// request that wrote its destination and finished: raw + inflated + bcpack-blank (a failed request is not in it), so
// a run whose requests are all compressed still shows storage progressing.
int format_request_counts(char* out, std::size_t size, LONG raw, LONG inflated, LONG failed, LONG blank) noexcept
{
    return std::snprintf(out, size, "raw=%ld inflated=%ld failed=%ld bcpack-blank=%ld completed=%ld", raw, inflated,
                         failed, blank, raw + inflated + blank);
}

int format_failure_causes(char* out, std::size_t size, int second, const StorageFailureCounts& n) noexcept
{
    return std::snprintf(out, size,
                         "x2w-dstorage heartbeat %d s: failed requests by cause read-at-or-past-eof=%ld "
                         "read-crossing-eof=%ld read-other=%ld destination-capacity=%ld destination-not-writable=%ld "
                         "zlib=%ld bcpack=%ld other=%ld",
                         second, n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]);
}

// Page-destination requests (option bit 21) by where their destination ends against the first 64 KiB
// page, each split into completed and failed, for the heartbeat (#1198). The end is page_offset (+0x10,
// low 16 bits) plus destination_size (+0x18, u32), formed in 64 bits so it cannot wrap. A request is
// counted once, when it runs: by execute_request for every outcome but "empty" (which is neither
// completed nor failed), and as failed when its page array is unreadable at run or its run threw. An
// entry that never started is not counted (it failed in Submit, without running). Fixed counters, no
// allocation; nothing here changes an outcome.
enum class PageEnd : unsigned { inside, at_page_end, past, count };
using PageEndCounts = LONG[static_cast<unsigned>(PageEnd::count)][2];  // [end][0 completed, 1 failed]
PageEndCounts page_end_counts;

PageEnd page_end(unsigned page_offset, unsigned destination_size) noexcept
{
    const u64 end = u64{page_offset} + destination_size;
    if (end < 0x10000) return PageEnd::inside;
    if (end == 0x10000) return PageEnd::at_page_end;
    return PageEnd::past;
}

void count_page_request(const unsigned char* request, bool failed) noexcept
{
    const u64 options = *reinterpret_cast<const u64*>(request);
    if (!(options & (1ull << 21))) return;
    const unsigned page_offset = static_cast<unsigned>(*reinterpret_cast<const u64*>(request + 0x10) & 0xFFFF);
    const unsigned destination_size = *reinterpret_cast<const unsigned*>(request + 0x18);
    InterlockedIncrement(&page_end_counts[static_cast<unsigned>(page_end(page_offset, destination_size))][failed ? 1 : 0]);
}

void sample_page_end_counts(PageEndCounts& counts) noexcept
{
    for (unsigned end = 0; end < static_cast<unsigned>(PageEnd::count); ++end)
        for (unsigned failed = 0; failed < 2; ++failed)
            counts[end][failed] = InterlockedCompareExchange(&page_end_counts[end][failed], 0, 0);
}

// The heartbeat's line of page-destination requests by end, after the failure causes. Like that line it
// carries no "inflated=", "options=" or "commit peak=" text.
int format_page_ends(char* out, std::size_t size, int second, const PageEndCounts& n) noexcept
{
    return std::snprintf(out, size,
                         "x2w-dstorage heartbeat %d s: page requests by end against the first 64 KiB page "
                         "inside-completed=%ld inside-failed=%ld at-page-end-completed=%ld at-page-end-failed=%ld "
                         "past-completed=%ld past-failed=%ld",
                         second, n[0][0], n[0][1], n[1][0], n[1][1], n[2][0], n[2][1]);
}

LONG bcpack_requests[16], bcpack_blank[16];
LONG zero_tokens_at_enqueue, zero_tokens_at_run, tokens_replaced_after_enqueue;

const char* format_counts(char* out, std::size_t size, LONG* counts)
{
    out[0] = 0;
    std::size_t used = 0;
    for (unsigned format = 1; format < 16; ++format) {
        const LONG count = InterlockedCompareExchange(&counts[format], 0, 0);
        if (!count) continue;
        if (used >= size) break;
        const int n = std::snprintf(out + used, size - used, "%s%u:%ld", used ? " " : "", format, count);
        if (n < 0 || static_cast<std::size_t>(n) >= size - used) {
            out[size - 1] = 0;
            break;
        }
        used += static_cast<std::size_t>(n);
    }
    return used ? out : "none";
}

void log_bcpack_summary()
{
    LONG total = 0;
    for (unsigned format = 1; format < 16; ++format)
        total += InterlockedCompareExchange(&bcpack_requests[format], 0, 0);
    if (!total) return;

    char requests[256], blank[256];
    log("x2w-dstorage BCPack summary: requests formats=%s; zero-filled formats=%s",
        format_counts(requests, sizeof(requests), bcpack_requests),
        format_counts(blank, sizeof(blank), bcpack_blank));
}

// Try to serve an identified continuation slice of an already-inflated chunked stream from the in-memory cache
// (Refs #1136, #1144, #1198, #1214). Returns true if this request was matched as a continuation slice.
static bool try_serve_cached_continuation(Object* self, u64 source, u64 offset, unsigned source_size,
                                          unsigned destination_size, u64 entry_id,
                                          std::vector<unsigned char>& output, std::size_t& decode_length,
                                          const char*& outcome, StorageFailure& failure, const unsigned char*& data)
{
    if (!self || !self->work) return false;
    QueueWork& work = *self->work;
    const std::lock_guard guard(work.mutex);
    for (auto it = work.cached_streams.begin(); it != work.cached_streams.end(); ++it) {
        if (it->source == source && it->compressed_size == source_size && offset > it->base_offset) {
            auto pos = std::find(it->continuation_ids.begin(), it->continuation_ids.end(), entry_id);
            if (pos != it->continuation_ids.end()) {
                const u64 delta = offset - it->base_offset;
                const unsigned step = it->chunk_size ? it->chunk_size : 0x10000;
                if ((delta % step) == 0 && delta < it->data.size() && delta <= UINT32_MAX) {
                    const std::size_t available = it->data.size() - static_cast<std::size_t>(delta);
                    if (destination_size > available) {
                        outcome = "destination capacity exceeded";
                        failure = StorageFailure::destination_capacity;
                        data = nullptr;
                        it->continuation_ids.erase(pos);
                        if (it->continuation_ids.empty()) {
                            work.cached_streams.erase(it);
                        }
                        return true;
                    }
                    const std::size_t slice_size = destination_size < available ? destination_size : available;
                    if (slice_size > 0) {
                        output.assign(it->data.begin() + delta, it->data.begin() + delta + slice_size);
                        decode_length = slice_size;
                        data = output.data();
                        outcome = "inflated (slice from stream cache)";
                        it->continuation_ids.erase(pos);
                        if (it->continuation_ids.empty()) {
                            work.cached_streams.erase(it);
                        }
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

void execute_request(Object* self, const unsigned char* request, const ULONG_PTR* tokens, u64 entry_id, u64 submission_id)
{
    X2W_ZONE("DirectStorage request");
    if (!request) return;
    const u64 options = *reinterpret_cast<const u64*>(request);
    auto* const destination = *reinterpret_cast<unsigned char* const*>(request + 0x08);
    const unsigned destination_size = *reinterpret_cast<const unsigned*>(request + 0x18);
    const u64 source = *reinterpret_cast<const u64*>(request + 0x20);
    const u64 offset = *reinterpret_cast<const u64*>(request + 0x28);
    const unsigned source_size = *reinterpret_cast<const unsigned*>(request + 0x30);
    const bool memory_source = (options & memory_source_option) != 0;
    const unsigned compression = static_cast<unsigned>(options & 0xFF);
    static LONG logged;
    if (x2w_local::first_calls(logged, 12)) {
        char bytes[3 * 0x40 + 1] = "";
        for (int i = 0; i < 0x40; ++i) std::snprintf(bytes + 3 * i, 4, "%02x ", request[i]);
        log("x2w-dstorage Queue::EnqueueRequest [%s]", bytes);
    }
    // Option bits (from the reference projects' notes, matched against this title's requests): bit 0
    // zlib, bits 1-4 a BCPack BC format number, bit 21 a destination given as 64 KiB physical page
    // tokens (+0x10 low 16 bits: offset into the first page), bit 22 a memory source.
    const bool zlib = (options & 1) != 0;
    const unsigned bcpack = static_cast<unsigned>((options >> 1) & 0xF);
    if (bcpack) InterlockedIncrement(&bcpack_requests[bcpack]);
    const bool page_destination = (options & (1ull << 21)) != 0;
    const unsigned page_offset = static_cast<unsigned>(*reinterpret_cast<const u64*>(request + 0x10) & 0xFFFF);
    const u64 known = 1 | (0xFull << 1) | (1ull << 21) | 0x400000;
    const char* outcome = "done";
    StorageFailure failure = StorageFailure::other;  // the heartbeat's cause, if the request fails
    std::vector<unsigned char> packed, output;
    const unsigned char* input = nullptr;
    const unsigned char* data = nullptr;
    std::size_t decode_length = source_size;
    const std::size_t destination_capacity = destination_size;
    bool written = false;

    if (options & ~known) {
        outcome = "unknown option bits, not done";
    } else if (!destination || !source_size) {
        outcome = "empty";
    } else if (memory_source) {
        input = reinterpret_cast<const unsigned char*>(source);
    } else {
        auto* file_object = reinterpret_cast<Object*>(source);
        bool file_read = false;
        if (file_object && file_object->kind == file) {
            // HDD mode: the prefetch may already hold these bytes (counted as read by the prefetch thread's own read).
            if (hdd_mode() && file_object->handle != INVALID_HANDLE_VALUE)
                file_read = hdd_cache().take(static_cast<std::uintptr_t>(file_object->generation), offset, source_size, packed, 3000);
            if (!file_read) {
                LARGE_INTEGER began, ended;
                QueryPerformanceCounter(&began);
                file_read = x2w_startup::read_storage_file(file_object->handle, offset, source_size, packed);
                QueryPerformanceCounter(&ended);
                note_disk_read(file_object->handle, offset, source_size, ended.QuadPart - began.QuadPart, file_read);
            }
        }
        if (file_read)
            input = packed.data();
        else {
            outcome = "read failed";
            failure = read_failure_cause(file_object, offset, source_size);
        }
    }

    if (input) {
        data = input;
        if (zlib) {
            bool direct_decoded = false;
            bool decode_error = false;
            if (has_zlib_header(input, source_size)) {
                // The public DSTORAGE_REQUEST layout has SourceSize at +0x30, IntermediateSize at +0x34 (the
                // number of BCPack bytes when zlib and BCPack are both requested) and CancellationTag at
                // +0x38. The tag is an arbitrary value, not a size, so no buffer is sized from it. A zlib-only
                // request inflates to DestinationSize; with BCPack after it the inflated size is
                // IntermediateSize, and the earlier destination-based bound is kept as the minimum.
                const unsigned intermediate_size = *reinterpret_cast<const unsigned*>(request + 0x34);
                std::size_t target_size = destination_size;
                if (bcpack) {
                    target_size = target_size * 2 + 0x10000;
                    if (intermediate_size > target_size) target_size = intermediate_size;
                }
                output.resize(target_size);
                std::size_t produced = 0;
                bool ok = x2w_inflate::inflate_zlib(input, source_size, output.data(), output.size(), &produced);
                if (!ok && output.size() < 0x80000) {
                    // If destination_size was smaller than the uncompressed stream, expand up to 512 KiB to
                    // accommodate larger chunks.
                    std::size_t big_size = std::size_t{destination_size} * 8 + 0x40000;
                    if (big_size < 0x80000) big_size = 0x80000;
                    std::vector<unsigned char> big(big_size);
                    std::size_t big_produced = 0;
                    if (x2w_inflate::inflate_zlib(input, source_size, big.data(), big.size(), &big_produced)) {
                        output = std::move(big);
                        produced = big_produced;
                        ok = true;
                    }
                }
                if (ok) {
                    output.resize(produced);
                    bool cached_chunked = false;
                    // If the stream expanded beyond this request's destination and continuation chunks are queued
                    // in this submission, scope the cache to those identified continuation requests and serve slice 0 (Refs #1136, #1144, #1198, #1214).
                    // BCPack streams are atomic multi-lane rANS payloads; they are not chunk-split continuation streams (Refs #1139).
                    if (!memory_source && !bcpack && produced > destination_size && self && self->work) {
                        auto continuation_ids = following_chunk_ids(self, submission_id, source, offset, source_size,
                                                                    destination_size, produced);
                        if (!continuation_ids.empty()) {
                            QueueWork& work = *self->work;
                            const std::lock_guard guard(work.mutex);
                            work.put_stream(source, offset, source_size, destination_size, std::move(continuation_ids), output);
                            decode_length = destination_size;
                            data = output.data();
                            outcome = "inflated (chunked stream piece 0)";
                            cached_chunked = true;
                        }
                    }
                    if (!cached_chunked) {
                        decode_length = produced;
                        data = output.data();
                        outcome = produced == destination_size || bcpack ? "inflated" : "inflated (larger/short)";
                    }
                    direct_decoded = true;
                } else {
                    decode_error = true;
                }
            }

            // If not directly decoded from a standalone stream, check if this request is a continuation
            // slice of an already-inflated chunked stream on this queue (Refs #1136, #1144, #1214).
            // A request presenting a zlib header that failed decode (e.g. damaged checksum) is an actual
            // decode error and must stay distinct from a proven continuation. Memory requests stay strictly on direct decode.
            if (!direct_decoded && !decode_error && !memory_source && !bcpack && destination && source_size && self && self->work) {
                auto* file_object = reinterpret_cast<Object*>(source);
                if (file_object && file_object->kind == file && file_object->handle != INVALID_HANDLE_VALUE &&
                    try_serve_cached_continuation(self, source, offset, source_size, destination_size, entry_id,
                                                  output, decode_length, outcome, failure, data)) {
                    direct_decoded = true;
                }
            }

            if (!direct_decoded) {
                outcome = "zlib failed";
                failure = StorageFailure::zlib;
                data = nullptr;
                static LONG diagnosed;
                if (x2w_local::first_calls(diagnosed, 12)) {
                    unsigned h[4] = {};
                    for (unsigned i = 0; i < 4 && i < source_size; ++i) h[i] = input[i];
                    log("x2w-dstorage zlib failure: header %02x %02x %02x %02x (dest 0x%x, source 0x%x)",
                        h[0], h[1], h[2], h[3], destination_size, source_size);
                }
            }
        }
    }
    if (input) {
        bool bcpack_decoded = false;
        if (data && bcpack) {
            if (bcpack == 7 || bcpack == 1 || bcpack == 4 || bcpack == 5 || bcpack == 3) {
                std::vector<unsigned char> decoded(destination_size);
                std::size_t produced = 0;
                if (x2w_bcpack::decode_bcpack(data, decode_length, bcpack, decoded.data(), decoded.size(), &produced)) {
                    output = std::move(decoded);
                    data = output.data();
                    decode_length = produced;
                    outcome = zlib ? "inflated + bcpack decoded" : "bcpack decoded";
                    bcpack_decoded = true;
                } else {
                    outcome = "bcpack failed";
                    failure = StorageFailure::bcpack;
                    data = nullptr;
                }
            } else {
                // shortcut for formats not decoded yet: texels are left zero (a blank texture) so the
                // title's loading continues. Counted as bcpack_blank.
                output.assign(destination_size, 0);
                data = output.data();
                decode_length = destination_size;
                outcome = "BCPack not decoded: zero-filled";
            }
        }
        if (data) {
            // Track decode length and destination capacity as separate values (Refs #1144).
            // A short destination or missing/insufficient page grant must fail the request
            // and leave the destination memory completely unchanged.
            if (decode_length > destination_capacity) {
                outcome = "destination capacity exceeded";
                failure = StorageFailure::destination_capacity;
                data = nullptr;
            } else if (page_destination) {
                const ULONG_PTR needed_pages = (page_offset + decode_length + 0xFFFF) / 0x10000;
                const ULONG_PTR granted_pages = (page_offset + destination_capacity + 0xFFFF) / 0x10000;
                if (needed_pages > granted_pages) {
                    outcome = "insufficient page grants";
                    data = nullptr;
                } else if (!tokens) {
                    outcome = "missing page tokens";
                    data = nullptr;
                } else {
                    for (ULONG_PTR i = 0; i < needed_pages; ++i) {
                        if (!tokens[i]) {
                            outcome = "missing page grant token";
                            data = nullptr;
                            break;
                        }
                    }
                }
            }
        }
        if (data) {
            written = true;
            if (page_destination) {
                using WritePhysical = BOOL(WINAPI*)(const ULONG_PTR*, ULONG_PTR, SIZE_T, const void*, SIZE_T);
                static const WritePhysical write_physical = [] {
                    const HMODULE xmem = GetModuleHandleW(L"xmem.dll");
                    return xmem ? reinterpret_cast<WritePhysical>(GetProcAddress(xmem, "x2w_xmem_write_physical")) : nullptr;
                }();
                const ULONG_PTR pages = (page_offset + decode_length + 0xFFFF) / 0x10000;
                written = write_physical && write_physical(tokens, pages, page_offset, data, decode_length);
                if (!written) {
                    outcome = "page destination write failed";
                    data = nullptr;
                }
            } else if (data != destination) {
                // The title's destination can be smaller in memory than its declared capacity (#870:
                // a write fault inside this copy). Judge the whole declared capacity before the first
                // byte is written, so a refusal leaves every page untouched.
                if (destination_writable(destination, destination_capacity)) {
                    std::memcpy(destination, data, decode_length);
                } else {
                    outcome = "destination not writable";
                    failure = StorageFailure::destination_not_writable;
                    data = nullptr;
                    written = false;
                }
            }
            if (written) {
                if (bcpack && !bcpack_decoded) {
                    InterlockedIncrement(&requests_bcpack_blank);
                    InterlockedIncrement(&bcpack_blank[bcpack]);
                } else if (zlib || bcpack) {
                    InterlockedIncrement(&requests_compressed);
                } else {
                    InterlockedIncrement(&requests_done);
                }
                note_written(zlib || bcpack, source_size, decode_length);
            }
        }
    }
    const bool empty = std::strcmp(outcome, "empty") == 0;
    const bool failed = (data == nullptr || !written) && !empty;
    const HRESULT result = failure_result(failure);
    if (failed) count_failure(failure, self, request, result);
    if (!empty) count_page_request(request, failed);
    if (failed && self) InterlockedExchange(&self->pending, static_cast<LONG>(result));
    static LONG summaries, failures;
    if (x2w_local::first_calls(summaries, 12) || (failed && x2w_local::first_calls(failures, 12)))
        log("x2w-dstorage   options=0x%llx dest=%p dest_size=0x%x source=0x%llx offset=0x%llx size=0x%x -> %s", options,
            static_cast<void*>(destination), destination_size, source, offset, source_size, outcome);
}

void signal_fence(void* fence, u64 value);

void run_entry(Object* self, Entry& entry)
{
    if (entry.type == Entry::request) {
        const u64 options = *reinterpret_cast<const u64*>(entry.bytes);
        if ((options & (1ull << 21)) && entry.token_count) {
            const auto* title = *reinterpret_cast<const ULONG_PTR* const*>(entry.bytes + 0x08);
            // The title's page array is read again in full when the request runs: entries are filled,
            // and also replaced, after enqueue. Writing through a token captured at enqueue put data
            // into a page the title had already reassigned. The caller has to keep the array readable
            // until the request runs; if it is not, the request fails before any page is written, so
            // no token captured at enqueue is ever used.
            const std::size_t bytes = entry.token_count * sizeof(ULONG_PTR);
            if (!title || !x2w_local::readable(title, bytes)) {
                count_failure(StorageFailure::other, self, entry.bytes);
                count_page_request(entry.bytes, true);
                if (self) InterlockedExchange(&self->pending, static_cast<LONG>(E_FAIL));
                static LONG unreadable;
                if (x2w_local::first_calls(unreadable, 12))
                    log("x2w-dstorage   options=0x%llx page array %p (%zu entries) unreadable when the request runs -> failed", options,
                        static_cast<const void*>(title), entry.token_count);
                return;
            }
            // Overwrite in place: the queue entry is kept free of allocations on this path.
            for (std::size_t i = 0; i < entry.token_count; ++i) {
                const ULONG_PTR now = title[i];
                if (entry.tokens[i] && now != entry.tokens[i]) InterlockedIncrement(&tokens_replaced_after_enqueue);
                if (!now) InterlockedIncrement(&zero_tokens_at_run);
                entry.tokens[i] = now;
            }
        }
        execute_request(self, entry.bytes, entry.token_count ? entry.tokens.get() : nullptr, entry.id, entry.submission_id);
    } else if (entry.type == Entry::status_entry) {
        const LONG result = InterlockedExchange(&self->pending, S_OK);
        InterlockedExchange(&entry.array->results[entry.index], result);
    } else {
        signal_fence(entry.fence, entry.value);
    }
}

// Stack committed up front for a queue worker. The thread decodes request buffers sized from request
// fields, and when one allocation fails it also runs the exception unwinder and the failure log. Once
// the system is out of commit a stack page that was never touched cannot be committed either, so the
// stack is committed at creation (CreateThread without STACK_SIZE_PARAMS_ARE_A_RESERVATION treats the
// size as the committed size; the reserved size stays the image default or larger).
constexpr SIZE_T worker_stack_commit = 1u << 20;

// Failure text is held in caller-provided storage: nothing here allocates.
void copy_reason(char (&reason)[96], const char* text) noexcept
{
    std::size_t i = 0;
    for (; text && text[i] && i + 1 < sizeof reason; ++i) reason[i] = text[i];
    reason[i] = 0;
}

// Fails the entry that threw: the next status entry reports E_FAIL, the worker goes on with the queue.
void fail_entry(Object* self, const Entry& entry, const char* reason) noexcept
{
    count_failure(StorageFailure::other, self, entry.type == Entry::request ? entry.bytes : nullptr);
    if (entry.type == Entry::request) count_page_request(entry.bytes, true);
    InterlockedExchange(&self->pending, static_cast<LONG>(E_FAIL));
    static LONG reported;
    if (x2w_local::first_calls(reported, 12))
        log("x2w-dstorage entry type %d failed: %s (options=0x%llx)", static_cast<int>(entry.type), reason,
            entry.type == Entry::request ? *reinterpret_cast<const unsigned long long*>(entry.bytes) : 0ull);
}

thread_local bool on_queue_worker = false;  // Submit from a worker must not wait for itself

void complete_deferred(Object* self, QueueWork* work);

bool deferred_pending(const QueueWork& work);

void discard_entries(QueueWork* work) noexcept;

DWORD WINAPI queue_worker(void* parameter)
{
    on_queue_worker = true;
    auto* const self = static_cast<Object*>(parameter);
    QueueWork& work = *self->work;
    for (;;) {
        Entry entry;
        {
            std::unique_lock lock(work.mutex);
            work.wake.wait(lock, [&] {
                return work.stopping || (!work.submitted.empty() && !deferred_pending(work));
            });
            if (work.stopping) break;
            entry = std::move(work.submitted.front());
            work.submitted.pop_front();
            if (hdd_mode()) hdd_look_ahead(work);
        }
        // Buffers are sized from request fields; a failed allocation fails that request instead of
        // ending the process (an exception leaving a thread calls terminate). The handlers only note
        // the reason in a fixed buffer: the request's buffers are already freed when a handler runs,
        // and the log is written after the handler has left, so no formatting runs while the
        // exception is in flight on this stack.
        char reason[96] = "";
        bool failed = false;
        try {
            run_entry(self, entry);
        } catch (const std::bad_alloc&) {
            failed = true;
            copy_reason(reason, "out of memory");
        } catch (const std::exception& error) {
            failed = true;
            copy_reason(reason, error.what());
        } catch (...) {
            failed = true;
            copy_reason(reason, "unknown exception");
        }
        if (failed) fail_entry(self, entry, reason);
        release_entry(entry);
        {
            const std::lock_guard guard(work.mutex);
            if (work.submitted.empty()) {
                work.cached_streams.clear();
            }
            ++work.finished_total;
        }
        work.drained.notify_all();
        complete_deferred(self, &work);
    }
    bool destroy = false;
    {
        const std::lock_guard guard(work.mutex);
        destroy = work.destroy_on_worker;
    }
    if (destroy) {
        // The final Release ran on this worker (a fence callback released the queue). The worker cannot
        // join itself: it drops the entries that never ran and frees the queue as it exits.
        discard_entries(&work);
        self->work = nullptr;
        delete &work;
        destroy_object(self);
        InterlockedDecrement(&live_workers);
        return 0;
    }
    InterlockedDecrement(&live_workers);
    return 0;
}

// Guards queue setup: building a queue's work, starting its worker and the unrecorded list.
std::mutex& setup_mutex()
{
    static std::mutex setup;
    return setup;
}

// Builds the queue's work once. The three deques allocate in their constructors, so even
// new (std::nothrow) can throw std::bad_alloc from here: that is treated like a null result, and
// nothing escapes to the title. The failure is logged after the handler has left.
QueueWork* ensure_work(Object* self)
{
    const std::lock_guard guard(setup_mutex());
    if (self->work) return self->work;
    QueueWork* work = nullptr;
    try {
        work = new (std::nothrow) QueueWork;
    } catch (const std::bad_alloc&) {
        work = nullptr;
    }
    if (!work) {
        log("x2w-dstorage queue work allocation failed");
        return nullptr;
    }
    self->work = work;
    return work;
}

// Starts the queue's worker unless it runs. The worker is created suspended with a committed stack,
// then once more with the default stack; false when neither thread could be created, which leaves
// the queue as it was so a later call tries again.
bool start_worker(Object* self, QueueWork* work)
{
    const std::lock_guard guard(setup_mutex());
    if (work->worker_started) return true;
    HANDLE worker = CreateThread(nullptr, worker_stack_commit, queue_worker, self, CREATE_SUSPENDED, nullptr);
    if (!worker) worker = CreateThread(nullptr, 0, queue_worker, self, CREATE_SUSPENDED, nullptr);
    if (!worker) {
        log("x2w-dstorage queue worker not started: error %lu", GetLastError());
        return false;
    }
    work->worker_started = true;
    work->thread = worker;
    work->thread_id = GetThreadId(worker);
    InterlockedIncrement(&live_workers);
    ResumeThread(worker);
    return true;
}

// What an entry that could not be stored needs in order to fail at Submit, including the request
// bytes for the error record. No page tokens or separately allocated request storage are needed.
struct Unrecorded {
    Unrecorded* next;
    u64 submission_id;
    u64 after;
    Entry::Type type;
    Object* array;
    Object* source;  // request: the file the entry would read, retained until the node is dropped
    unsigned index;
    void* fence;
    u64 value;
    unsigned char bytes[sizeof(StorageRequest)];
};

void release_unrecorded(Unrecorded& node) noexcept
{
    release_entry_object(node.source);
    node.source = nullptr;
    release_entry_object(node.array);
    node.array = nullptr;
}

// True while snapshot entries of an earlier Submit are ready or being completed by some thread. The
// worker starts no newer entry then, so a deferred completion and a normal one keep the same order.
// The caller holds work.mutex.
bool deferred_pending(const QueueWork& work)
{
    if (work.completing) return true;
    if (!work.failing.empty() && work.failing.front().after <= work.finished_total) return true;
    return work.deferred_head && work.deferred_head->after <= work.finished_total;
}

// Fails an entry that will not run: a request counts as failed, a status entry completes with the
// pending failure (E_OUTOFMEMORY when nothing worse is pending), a fence is signaled.
void fail_unstarted(Object* self, Entry::Type type, Object* array, unsigned index, void* fence, u64 value,
                    const unsigned char* request)
{
    if (type == Entry::request) {
        count_failure(StorageFailure::other, self, request, E_OUTOFMEMORY);
        InterlockedCompareExchange(&self->pending, static_cast<LONG>(E_OUTOFMEMORY), S_OK);
    } else if (type == Entry::status_entry) {
        const LONG pending = InterlockedExchange(&self->pending, S_OK);
        if (array && array->results && index < array->capacity) {
            InterlockedExchange(&array->results[index],
                                pending < 0 ? pending : static_cast<LONG>(E_OUTOFMEMORY));
        }
    } else {
        signal_fence(fence, value);
    }
}

// An entry the queue has no room for (its work could not be built or grown): remembered in a small
// record that needs no queue storage, failed by the next Submit. Only when even that record cannot be
// allocated is the entry failed here, because nothing is left to remember it with (a documented last
// resort: that entry may then be failed ahead of the ones stored before it).
//
// Order. One accumulator (Object::pending) holds the worst failure since the previous status entry,
// and a status entry consumes it, so an entry that lost its place in the queue order would leak a
// failure past the status entry it belongs to. Therefore, once an entry of this queue has been
// deferred, every later entry until the next Submit is deferred too, even if it could be stored:
// the stored entries (`waiting`) are an in-order prefix and the deferred ones an in-order suffix.
// Submit handles the prefix first (moves it to the worker, or fails it), then fails the suffix.
// Submit snapshots the failing entries at its cutoff and the worker completes that snapshot right after
// it finishes the prefix, so suffix completions are visible in queue order and see the prefix's failures,
// a later batch cannot be completed by this Submit, and Submit never waits for the prefix. After the first allocation failure the rest of the batch fails with E_OUTOFMEMORY even if
// memory has recovered, which is acceptable under out-of-memory and preferable to reordering.
bool has_unrecorded(Object* self)
{
    const std::lock_guard guard(setup_mutex());
    return self->unrecorded_head != nullptr;
}

void defer_unrecorded(Object* self, Entry& entry)
{
    auto* node = new (std::nothrow) Unrecorded{};
    if (!node) {
        fail_unstarted(self, entry.type, entry.array, entry.index, entry.fence, entry.value, entry.bytes);
        release_entry(entry);
        return;
    }
    node->next = nullptr;
    node->submission_id = 0;
    node->after = 0;
    node->type = entry.type;
    node->array = entry.array;
    node->source = entry.source;
    node->index = entry.index;
    node->fence = entry.fence;
    node->value = entry.value;
    entry.array = nullptr;
    entry.source = nullptr;  // the node now owns the entry's references
    if (entry.type == Entry::request) std::memcpy(node->bytes, entry.bytes, sizeof node->bytes);
    const std::lock_guard guard(setup_mutex());
    (self->unrecorded_tail ? self->unrecorded_tail->next : self->unrecorded_head) = node;
    self->unrecorded_tail = node;
}

void fail_unrecorded(Object* self)
{
    Unrecorded* node;
    {
        const std::lock_guard guard(setup_mutex());
        node = self->unrecorded_head;
        self->unrecorded_head = self->unrecorded_tail = nullptr;
    }
    while (node) {
        Unrecorded* const next = node->next;
        fail_unstarted(self, node->type, node->array, node->index, node->fence, node->value, node->bytes);
        release_unrecorded(*node);
        delete node;
        node = next;
    }
}

// Fails every entry still waiting, in order; none of them will run.
void fail_waiting(Object* self, QueueWork* work)
{
    for (;;) {
        Entry entry{};
        {
            const std::lock_guard guard(work->mutex);
            if (work->waiting.empty()) break;
            entry = std::move(work->waiting.front());
            work->waiting.pop_front();
        }
        fail_unstarted(self, entry.type, entry.array, entry.index, entry.fence, entry.value, entry.bytes);
        release_entry(entry);
    }
}

// Submit: waiting entries (the in-order prefix) go to the worker, which is started if it is not
// running yet (resources may have recovered since Enqueue). If it still cannot be started, or the
// worker's list cannot grow, the waiting entries fail. Only then do the entries that could not be
// stored (the in-order suffix, see defer_unrecorded) fail, so every failure reaches the status entry
// after it, in order. Their status entries and fences complete here, not earlier in Enqueue. No
// exception leaves Submit.
void submit(Object* self)
{
    QueueWork* work = ensure_work(self);
    if (!work) {
        fail_unrecorded(self);
        return;
    }
    if (!start_worker(self, work)) {
        fail_waiting(self, work);
        fail_unrecorded(self);
        return;
    }
    bool all_moved = true;
    bool snapshot_ok = true;
    bool snapshotted = false;
    u64 prefix_end = 0;
    {
        const std::lock_guard guard(work->mutex);
        const u64 submission_id = ++work->next_submission_id;
        try {
            // An entry leaves `waiting` only after it is in `submitted`, so a failed growth loses none.
            while (!work->waiting.empty()) {
                Entry& entry = work->waiting.front();
                entry.submission_id = submission_id;
                work->submitted.push_back(std::move(entry));
                ++work->submitted_total;
                work->waiting.pop_front();
            }
        } catch (const std::bad_alloc&) {
            all_moved = false;
        }
        prefix_end = work->submitted_total;
        // The cutoff: the entries that fail are snapshotted here, under the same lock, so a batch enqueued
        // later cannot be completed by this Submit. The snapshot completes on the worker right after it
        // finishes the prefix (see complete_deferred), so this call never waits for the prefix.
        try {
            while (!work->waiting.empty()) {
                Entry& entry = work->waiting.front();
                entry.submission_id = submission_id;
                entry.after = prefix_end;
                work->failing.push_back(std::move(entry));
                work->waiting.pop_front();
                snapshotted = true;
            }
        } catch (const std::bad_alloc&) {
            snapshot_ok = false;
        }
        if (snapshot_ok) {
            Unrecorded* node;
            {
                const std::lock_guard setup(setup_mutex());
                node = self->unrecorded_head;
                self->unrecorded_head = self->unrecorded_tail = nullptr;
            }
            for (Unrecorded* n = node; n; n = n->next) {
                n->submission_id = submission_id;
                n->after = prefix_end;
                if (!n->next) {
                    (work->deferred_tail ? work->deferred_tail->next : work->deferred_head) = node;
                    work->deferred_tail = n;
                }
                snapshotted = true;
            }
        }
    }
    work->wake.notify_one();
    if (!snapshot_ok) {
        // Last resort when even the snapshot could not grow: wait for the prefix as before, then fail in order.
        if (!on_queue_worker) {
            std::unique_lock lock(work->mutex);
            work->drained.wait(lock, [&] { return work->finished_total >= prefix_end; });
        }
        complete_deferred(self, work);
        fail_waiting(self, work);
        fail_unrecorded(self);
        return;
    }
    if (snapshotted) complete_deferred(self, work);
}

// Completes the snapshot entries (failed entries of earlier Submits) whose prefix the worker has finished,
// in Submit order and, within one Submit, stored entries before unrecorded ones. Called by the worker after
// every entry and by Submit, so nothing waits on a calling thread. Nothing allocates.
void complete_deferred(Object* self, QueueWork* work)
{
    // One owner completes at a time, but no lock is held while it does: a fence is a caller-supplied
    // operation. A thread that finds an owner returns at once; the owner takes what it queued before
    // it gives up ownership, so nothing is left behind and no Submit waits behind a fence.
    {
        const std::lock_guard guard(work->mutex);
        if (work->completing) return;
        work->completing = true;
    }
    for (;;) {
        Entry entry{};
        Unrecorded* node = nullptr;
        {
            const std::lock_guard guard(work->mutex);
            bool take_entry = !work->failing.empty() && work->failing.front().after <= work->finished_total;
            bool take_node = work->deferred_head && work->deferred_head->after <= work->finished_total;
            if (take_entry && take_node) {
                if (work->failing.front().submission_id <= work->deferred_head->submission_id) take_node = false;
                else take_entry = false;
            }
            if (take_entry) {
                entry = std::move(work->failing.front());
                work->failing.pop_front();
            } else if (take_node) {
                node = work->deferred_head;
                work->deferred_head = node->next;
                if (!work->deferred_head) work->deferred_tail = nullptr;
            } else {
                work->completing = false;
                work->wake.notify_all();  // the worker may be waiting for the owner to finish
                return;
            }
        }
        if (node) {
            fail_unstarted(self, node->type, node->array, node->index, node->fence, node->value, node->bytes);
            release_unrecorded(*node);
            delete node;
        } else {
            fail_unstarted(self, entry.type, entry.array, entry.index, entry.fence, entry.value, entry.bytes);
            release_entry(entry);
        }
    }
}

// Drops every entry that will never run and releases the objects those entries held. Called after the
// worker has left, so no other thread touches the queue.
void discard_entries(QueueWork* work) noexcept
{
    for (Entry& entry : work->waiting) release_entry(entry);
    work->waiting.clear();
    for (Entry& entry : work->submitted) release_entry(entry);
    work->submitted.clear();
    for (Entry& entry : work->failing) release_entry(entry);
    work->failing.clear();
    Unrecorded* node = work->deferred_head;
    work->deferred_head = work->deferred_tail = nullptr;
    while (node) {
        Unrecorded* const next = node->next;
        release_unrecorded(*node);
        delete node;
        node = next;
    }
    work->cached_streams.clear();
}

bool shutdown_queue(Object* self)
{
    QueueWork* work = self->work;
    if (!work) return false;
    bool on_worker;
    {
        const std::lock_guard guard(work->mutex);
        work->stopping = true;
        on_worker = work->worker_started && GetCurrentThreadId() == work->thread_id;
        if (on_worker) work->destroy_on_worker = true;
    }
    work->wake.notify_all();
    if (on_worker) return true;  // the worker frees the queue as it exits
    if (work->thread) {
        WaitForSingleObject(work->thread, INFINITE);
        CloseHandle(work->thread);
        work->thread = nullptr;
    }
    discard_entries(work);
    self->work = nullptr;
    delete work;
    return false;
}

// Enqueue only records the entry, whether or not a worker runs; nothing completes before Submit.
void push(Object* self, Entry&& entry)
{
    QueueWork* work = ensure_work(self);
    if (!work || has_unrecorded(self)) {  // after a deferred entry every later one is deferred: order
        defer_unrecorded(self, entry);
        return;
    }
    start_worker(self, work);  // a failure shows at Submit
    bool recorded = false;
    bool full = false;
    {
        const std::lock_guard guard(work->mutex);
        // The handler sits inside the lock's scope, like every other growth of these lists: a
        // std::bad_alloc must not unwind through the guard (the MSVC build under test then leaves
        // the mutex locked, and the next Submit throws std::system_error on its own lock).
        try {
            entry.id = ++work->next_entry_id;
            work->waiting.push_back(std::move(entry));
            recorded = true;
            full = self->capacity && work->waiting.size() >= self->capacity;
        } catch (const std::bad_alloc&) {
        }
    }
    if (!recorded) {
        defer_unrecorded(self, entry);
        return;
    }
    if (full) submit(self);  // a full queue submits itself, as on PC
}

// Queue slot 3, EnqueueRequest(request*): copied now, run after Submit.
void __stdcall queue_enqueue_request(Object* self, const unsigned char* request)
{
    if (!request) return;
    note_enqueued();
    Entry entry{};
    entry.type = Entry::request;
    std::memcpy(entry.bytes, request, sizeof entry.bytes);
    const u64 options = *reinterpret_cast<const u64*>(request);
    if (!(options & memory_source_option)) {
        // The request keeps the file it reads alive until its turn: the guest may Release it after
        // EnqueueRequest (#2788).
        auto* source = *reinterpret_cast<Object* const*>(request + 0x20);
        if (source && x2w_local::readable(source, sizeof(Object)) && source->kind == file) {
            retain_entry_object(source);
            entry.source = source;
        }
    }
    if (options & (1ull << 21)) {
        const auto* title = *reinterpret_cast<const ULONG_PTR* const*>(request + 0x08);
        const unsigned destination_size = *reinterpret_cast<const unsigned*>(request + 0x18);
        const unsigned page_offset = static_cast<unsigned>(*reinterpret_cast<const u64*>(request + 0x10) & 0xFFFF);
        const std::size_t pages = (std::size_t{page_offset} + destination_size + 0xFFFF) / 0x10000;
        if (title && pages && x2w_local::readable(title, pages * sizeof(ULONG_PTR))) {
            entry.tokens.reset(new (std::nothrow) ULONG_PTR[pages]);
            if (!entry.tokens) {
                defer_unrecorded(self, entry);  // fails at Submit; nothing leaves Enqueue
                return;
            }
            entry.token_count = pages;
            for (std::size_t i = 0; i < pages; ++i) {
                entry.tokens[i] = title[i];
                if (!title[i]) InterlockedIncrement(&zero_tokens_at_enqueue);
            }
        }
    }
    push(self, std::move(entry));
}

// Queue slot 4, EnqueueStatus(statusArray, index): completes, in order, with the worst result of the
// requests since the previous status.
void __stdcall queue_enqueue_status(Object* self, Object* array, unsigned index)
{
    const bool valid = array && array->kind == status && index < array->capacity;
    static LONG calls;
    if (!valid || x2w_local::first_calls(calls, 8)) {
        char counts[128];
        format_request_counts(counts, sizeof(counts), requests_done, requests_compressed, requests_failed,
                              requests_bcpack_blank);
        log("x2w-dstorage Queue::EnqueueStatus array=%p index=%u%s (requests %s)", static_cast<void*>(array), index,
            valid ? "" : " INVALID ARRAY", counts);
    }
    if (!valid) return;
    InterlockedExchange(&array->results[index], static_cast<LONG>(0x8000000A));  // E_PENDING until its turn
    Entry entry{};
    entry.type = Entry::status_entry;
    retain_entry_object(array);
    entry.array = array;
    entry.index = index;
    push(self, std::move(entry));
}

// The remaining queue and status-array slots follow the public PC IDStorageQueue / IDStorageStatusArray
// order, which slots 3, 4 and 6 matched; each logs its first calls with its arguments.
// Queue slot 5, EnqueueSignal(fence, value): requests are already complete, so the fence is signaled
// now through its own vtable (ID3D12Fence::Signal, slot 10).
void __stdcall queue_enqueue_signal(Object* self, void* fence, u64 value)
{
    Entry entry{};
    entry.type = Entry::signal;
    entry.fence = fence;
    entry.value = value;
    push(self, std::move(entry));
}
void signal_fence(void* fence, u64 value)
{
    long hr = E_POINTER;
    if (fence) {
        using Signal = long(__stdcall*)(void*, u64);
        hr = reinterpret_cast<Signal>((*static_cast<void***>(fence))[10])(fence, value);
    }
    static LONG calls;
    if (x2w_local::first_calls(calls, 4))
        log("x2w-dstorage Queue::EnqueueSignal fence=%p value=0x%llx -> 0x%08lx", fence, value, static_cast<unsigned long>(hr));
}
void __stdcall queue_submit(Object* self, u64 rdx, u64 r8)
{
    static LONG calls;
    if (x2w_local::first_calls(calls, 4)) log("x2w-dstorage Queue::Submit rdx=0x%llx r8=0x%llx", rdx, r8);
    submit(self);
}
void __stdcall queue_cancel(Object*, u64 mask, u64 value)
{
    static LONG calls;
    if (x2w_local::first_calls(calls, 4)) log("x2w-dstorage Queue::CancelRequestsWithTag mask=0x%llx value=0x%llx (no-op)", mask, value);
}
void __stdcall queue_close(Object*) { log("x2w-dstorage Queue::Close"); }
// Slots 9 and 10 follow Microsoft's Xbox-GDK-Samples dstorage_win32.h. GetErrorEvent's
// caller owns its handle (Microsoft Learn IDStorageQueueX::GetErrorEvent).
HANDLE __stdcall queue_error_event(Object* self)
{
    auto& object = *static_cast<QueueObject*>(self);
    HANDLE event = nullptr;
    {
        const std::lock_guard guard(object.error_mutex);
        if (!object.error_event)
            object.error_event = CreateEventW(nullptr, TRUE, object.error_record.failure_count != 0, nullptr);
        if (object.error_event)
            DuplicateHandle(GetCurrentProcess(), object.error_event, GetCurrentProcess(), &event,
                            0, FALSE, DUPLICATE_SAME_ACCESS);
    }
    static LONG calls;
    if (x2w_local::first_calls(calls, 4)) log("x2w-dstorage Queue::GetErrorEvent -> %p", event);
    return event;
}
void __stdcall queue_retrieve_error_record(Object* self, StorageErrorRecord* out)
{
    if (!out) return;
    auto& object = *static_cast<QueueObject*>(self);
    const std::lock_guard guard(object.error_mutex);
    std::memcpy(out, &object.error_record, sizeof *out);
    object.error_record = x2w_startup::empty_error_record();
    // The sample header and Microsoft Learn describe a new notification after each retrieve.
    // Keep the manual-reset event signaled until retrieval consumes the record. Reset under
    // the record's lock so a concurrent failure cannot lose its notification.
    if (object.error_event) ResetEvent(object.error_event);
}
// Status array slot 3 IsComplete(index), slot 4 GetHResult(index).
bool __stdcall status_is_complete(Object* self, unsigned index)
{
    const bool complete = index < self->capacity && self->results[index] != static_cast<LONG>(0x8000000A);
    static LONG calls;
    if (x2w_local::first_calls(calls, 4)) log("x2w-dstorage StatusArray::IsComplete index=%u -> %d", index, complete);
    return complete;
}
long __stdcall status_hresult(Object* self, unsigned index)
{
    const long hr = index < self->capacity ? self->results[index] : E_INVALIDARG;
    static LONG calls;
    if (x2w_local::first_calls(calls, 4)) log("x2w-dstorage StatusArray::GetHResult index=%u -> 0x%08lx", index, static_cast<unsigned long>(hr));
    return hr;
}

void** make_table(unsigned kind)
{
    static void* tables[4][slots];
    static bool ready[4];
    const unsigned index = kind / 0x20 - 1;
    void** table = tables[index];
    if (!ready[index]) {
        fill_traps(table, std::make_index_sequence<slots>{});
        table[0] = reinterpret_cast<void*>(&query_interface);
        table[1] = reinterpret_cast<void*>(&add_ref);
        table[2] = reinterpret_cast<void*>(&release);
        if (kind == factory) {
            table[4] = reinterpret_cast<void*>(&factory_open_file);
            table[5] = reinterpret_cast<void*>(&factory_create_status_array);
            table[8] = reinterpret_cast<void*>(&factory_slot8);
            table[10] = reinterpret_cast<void*>(&factory_create_queue);
        } else if (kind == queue) {
            table[3] = reinterpret_cast<void*>(&queue_enqueue_request);
            table[4] = reinterpret_cast<void*>(&queue_enqueue_status);
            table[5] = reinterpret_cast<void*>(&queue_enqueue_signal);
            table[6] = reinterpret_cast<void*>(&queue_submit);
            table[7] = reinterpret_cast<void*>(&queue_cancel);
            table[8] = reinterpret_cast<void*>(&queue_close);
            table[9] = reinterpret_cast<void*>(&queue_error_event);
            table[10] = reinterpret_cast<void*>(&queue_retrieve_error_record);
        } else if (kind == status) {
            table[3] = reinterpret_cast<void*>(&status_is_complete);
            table[4] = reinterpret_cast<void*>(&status_hresult);
        } else if (kind == file) {
            table[3] = reinterpret_cast<void*>(&file_close);
            table[4] = reinterpret_cast<void*>(&file_information);
        }
        ready[index] = true;
    }
    return table;
}

// What the heartbeat reports about memory (#1199): this process's commit charge and the live
// physical-page sections of the xmem stand-in, read through its probe export. The flags keep the
// line's shape when a reading is unavailable.
struct MemorySample {
    ULONGLONG peak_commit = 0, commit = 0, sections = 0;
    bool commit_known = false, sections_known = false;
};

MemorySample memory_sample() noexcept
{
    MemorySample sample;
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters))) {
        sample.commit = counters.PagefileUsage;
        sample.peak_commit = counters.PeakPagefileUsage;
        sample.commit_known = true;
    }
    using PageStats = BOOL(WINAPI*)(ULONGLONG*);
    HMODULE xmem = nullptr;
    if (GetModuleHandleExW(0, L"xmem.dll", &xmem)) {
        const auto page_stats = reinterpret_cast<PageStats>(GetProcAddress(xmem, "x2w_xmem_page_stats"));
        sample.sections_known = page_stats && page_stats(&sample.sections);
        FreeLibrary(xmem);
    }
    return sample;
}

// The heartbeat interval in ms: 10000 unless X2W_HEARTBEAT_INTERVAL_MS holds a whole number from 1000
// to 10000. Only tests set it, to see a sample without waiting 10 s; anything else keeps the default.
DWORD heartbeat_interval_ms() noexcept
{
    char text[16];
    const DWORD length = GetEnvironmentVariableA("X2W_HEARTBEAT_INTERVAL_MS", text, sizeof(text));
    if (length == 0 || length >= sizeof(text)) return 10000;
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return (*end == '\0' && value >= 1000 && value <= 10000) ? static_cast<DWORD>(value) : 10000;
}

}  // namespace

// Fixture-only probes (#2788): objects alive by kind (0 = every kind) and running queue workers.
extern "C" __declspec(dllexport) unsigned long x2w_xcrdapi_live_objects(unsigned kind)
{
    if (kind == 0) {
        unsigned long total = 0;
        for (LONG count : live_by_kind) total += static_cast<unsigned long>(count);
        return total;
    }
    if (kind != factory && kind != file && kind != status && kind != queue) return 0;
    return static_cast<unsigned long>(live_by_kind[kind / 0x20 - 1]);
}

extern "C" __declspec(dllexport) unsigned long x2w_xcrdapi_live_workers()
{
    return static_cast<unsigned long>(live_workers);
}

extern "C" long __stdcall DStorageGetFactory(const GUID* riid, void** out)
{
    char id[48];
    guid_text(riid, id, sizeof(id));
    log("x2w-dstorage DStorageGetFactory riid=%s", id);
    if (!out) return E_POINTER;
    static const bool summaries = [] {
        std::atexit(log_bcpack_summary);
        std::atexit(log_disk_summary);
        return true;
    }();
    (void)summaries;
    // Heartbeat: the request counters every 10 s, so a run shows whether loading still progresses
    // after the per-call logs have gone quiet. Peak commit and the live physical-page sections show
    // what the run spends (#1199), including while requests are stalled. A second line splits the
    // failed requests by cause, a third the page-destination requests by end (#1198).
    static HANDLE heartbeat = CreateThread(nullptr, 0, [](void*) -> DWORD {
        LONG last = -1;
        LONG64 last_read = 0, last_inflated = 0;
        DiskStats last_disk;
        MemorySample last_memory;
        const DWORD interval_ms = heartbeat_interval_ms();
        for (DWORD elapsed_ms = interval_ms;; elapsed_ms += interval_ms) {
            const int second = static_cast<int>(elapsed_ms / 1000);
            Sleep(interval_ms);
            // Sample the counters once for every line of this report, even while the worker is reporting
            // new failures.
            StorageFailureCounts failure_counts;
            const LONG failed = sample_failure_counts(failure_counts);
            PageEndCounts page_ends;
            sample_page_end_counts(page_ends);
            const LONG total = requests_done + requests_compressed + failed;
            const LONG total2 = total + requests_bcpack_blank;
            {
                // Every report, changed or not: "loading slow" (pending shrinks, MB/s above 0) against "loading stuck"
                // (pending flat, oldest age growing, 0 MB/s).
                const LONG64 read_now = bytes_read_total, inflated_now = bytes_inflated_total;
                char progress[256];
                format_progress(progress, sizeof progress, second, requests_enqueued, total2 - failed, failed, read_now - last_read,
                                inflated_now - last_inflated, read_now, inflated_now, interval_ms / 1000.0, GetTickCount());
                log("%s", progress);
                const DiskStats disk_now = disk_snapshot();
                const LONG64 frequency = disk_frequency();
                char disk_line[512];
                format_disk(disk_line, sizeof disk_line, second, disk_now, last_disk, interval_ms / 1000.0,
                            frequency > 0 ? static_cast<double>(frequency) : 1.0);
                log("%s", disk_line);
                if (hdd_mode()) {
                    const x2w_startup::ReadCache::Counters c = hdd_cache().counters();
                    log("x2w-dstorage hdd %d s: cache hits %ld misses %ld (waited for a read in progress %ld); prefetch groups %ld, disk reads saved by "
                        "merging %ld, prefetched %lld KiB, unused entries dropped %ld, failed group reads %ld", second, c.hits, c.misses,
                        c.waited, c.groups, c.reads_saved, static_cast<long long>(c.prefetched_bytes >> 10), c.evicted_unused, c.failed);
                }
                last_disk = disk_now;
                last_read = read_now;
                last_inflated = inflated_now;
            }
            const MemorySample sample = memory_sample();
            const ULONGLONG peak_mib = sample.peak_commit >> 20;
            const ULONGLONG current_mib = sample.commit >> 20;
            if (total2 != last || peak_mib != (last_memory.peak_commit >> 20) ||
                current_mib != (last_memory.commit >> 20) || sample.sections != last_memory.sections ||
                sample.commit_known != last_memory.commit_known || sample.sections_known != last_memory.sections_known) {
                char memory[96];
                if (sample.commit_known)
                    std::snprintf(memory, sizeof(memory), "commit peak=%llu MiB current=%llu MiB",
                                  static_cast<unsigned long long>(peak_mib),
                                  static_cast<unsigned long long>(current_mib));
                else
                    std::snprintf(memory, sizeof(memory), "commit unknown");
                char section_text[24];
                if (sample.sections_known)
                    std::snprintf(section_text, sizeof(section_text), "%llu",
                                  static_cast<unsigned long long>(sample.sections));
                else
                    std::snprintf(section_text, sizeof(section_text), "unknown");
                // The counts keep "inflated=%ld failed=%ld" for tools/startup/live_view.py (format_request_counts);
                // requests_compressed counts all compressed streams (zlib and standalone/embedded BCPack).
                char counts[128];
                format_request_counts(counts, sizeof(counts), requests_done, requests_compressed, failed,
                                      requests_bcpack_blank);
                log("x2w-dstorage heartbeat %d s: requests %s; page tokens 0 at enqueue=%ld, "
                    "still 0 at run=%ld, replaced after enqueue=%ld; %s, live physical-page sections=%s", second, counts,
                    zero_tokens_at_enqueue, zero_tokens_at_run, tokens_replaced_after_enqueue, memory, section_text);
                char causes[384];
                if (format_failure_causes(causes, sizeof(causes), second, failure_counts) > 0) log("%s", causes);
                char ends[384];
                if (format_page_ends(ends, sizeof(ends), second, page_ends) > 0) log("%s", ends);
                last = total2;
                last_memory = sample;
            }
        }
    }, nullptr, 0, nullptr);
    (void)heartbeat;
    static Object* instance = make(factory);
    InterlockedIncrement(&instance->refs);
    *out = instance;
    return S_OK;
}
