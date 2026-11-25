// Source-built consumer for the startup trap stubs. Imports by name from chad.dll, xg_xs.dll and
// xmem.dll and calls the export the first argument selects. (XMemTransferMemory and
// D3D12ConfigureVirtualMemoryX have real implementations in the shims and their own tests.)
//   none     -> returns 0 without calling a stub (control)
//   chad     -> XDspActivate(0x11, 0x22, 0x33, 0x44)
//   xg_xs    -> XGSuggestSwizzleMode(0x55, 0, 0, 0)
//   physical -> XMemAllocatePhysicalPages/XMemFreePhysicalPages round trip (no trap expected)
//   dstorage -> DirectStorage multi-page RFC 1950 zlib request, BCPack decode and failure controls (Refs #1139, #1144)
//   dstorage-pages -> page-array entries replaced, or made unreadable, between EnqueueRequest and Submit
//   dstorage-error-record -> queue-local error events and the first failed request record
//   store    -> XStoreCreateContext for the local user, a mutated provider, and a non-local user (Refs #1147)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "startup_stubs/dstorage_error_record.h"
#include "bcpack_encoder.h"
#include <cstdint>
#include <cstdio>
#include <array>
#include <fstream>
#include <cstring>
#include <vector>

using u64 = std::uint64_t;

extern "C" __declspec(dllimport) u64 XDspActivate(u64, u64, u64, u64);
extern "C" __declspec(dllimport) u64 XGSuggestSwizzleMode(u64, u64, u64, u64);
extern "C" __declspec(dllimport) int XMemAllocatePhysicalPages(u64 flags, std::size_t* count, u64* pages);
extern "C" __declspec(dllimport) int XMemFreePhysicalPages(std::size_t count, u64* pages);
extern "C" __declspec(dllimport) void* WINAPI XMemVirtualAlloc(void* address, SIZE_T bytes, DWORD allocation_type,
                                                              ULONGLONG attributes, DWORD protection);
extern "C" __declspec(dllimport) void* WINAPI XMemMapPhysicalPages(void* base, ULONG_PTR count,
                                                                   const ULONG_PTR* tokens);
int x2w_store_method3_check();

namespace {

uint32_t adler32_calc(const unsigned char* data, size_t size) {
    uint32_t a = 1, b = 0;
    while (size) {
        size_t chunk = size < 5552 ? size : 5552;
        size -= chunk;
        while (chunk--) {
            a += *data++;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    return (b << 16) | a;
}

std::vector<unsigned char> make_stored_zlib(const unsigned char* data, size_t size) {
    std::vector<unsigned char> out;
    out.push_back(0x78);
    out.push_back(0x01);
    size_t pos = 0;
    while (pos < size) {
        size_t chunk = (size - pos > 32768) ? 32768 : (size - pos);
        bool last = (pos + chunk == size);
        out.push_back(last ? 0x01 : 0x00);
        out.push_back(static_cast<unsigned char>(chunk & 0xFF));
        out.push_back(static_cast<unsigned char>((chunk >> 8) & 0xFF));
        unsigned short check = static_cast<unsigned short>(chunk ^ 0xFFFF);
        out.push_back(static_cast<unsigned char>(check & 0xFF));
        out.push_back(static_cast<unsigned char>((check >> 8) & 0xFF));
        out.insert(out.end(), data + pos, data + pos + chunk);
        pos += chunk;
    }
    uint32_t adler = adler32_calc(data, size);
    out.push_back(static_cast<unsigned char>((adler >> 24) & 0xFF));
    out.push_back(static_cast<unsigned char>((adler >> 16) & 0xFF));
    out.push_back(static_cast<unsigned char>((adler >> 8) & 0xFF));
    out.push_back(static_cast<unsigned char>(adler & 0xFF));
    return out;
}

struct BitWriter {
    std::vector<unsigned char> bytes;
    unsigned bit_buf = 0;
    int bit_cnt = 0;

    void put_bit(int b) {
        bit_buf |= (b & 1) << bit_cnt++;
        if (bit_cnt == 8) {
            bytes.push_back(static_cast<unsigned char>(bit_buf));
            bit_buf = 0;
            bit_cnt = 0;
        }
    }

    void put_code_msb(unsigned code, int bits) {
        for (int i = bits - 1; i >= 0; --i) {
            put_bit((code >> i) & 1);
        }
    }

    void flush() {
        if (bit_cnt > 0) {
            bytes.push_back(static_cast<unsigned char>(bit_buf));
            bit_buf = 0;
            bit_cnt = 0;
        }
    }
};

std::vector<unsigned char> make_fixed_deflate_zlib(unsigned char val, size_t count) {
    BitWriter bw;
    bw.bytes.push_back(0x78);
    bw.bytes.push_back(0x01);
    bw.put_bit(1);
    bw.put_bit(1);
    bw.put_bit(0);
    bw.put_code_msb(48 + val, 8);

    size_t remaining = count - 1;
    while (remaining >= 258) {
        bw.put_code_msb(197, 8); // symbol 285 = length 258
        bw.put_code_msb(0, 5);   // distance 1
        remaining -= 258;
    }
    while (remaining >= 3) {
        size_t len = remaining > 10 ? 10 : remaining;
        unsigned sym = static_cast<unsigned>(257 + (len - 3));
        bw.put_code_msb(sym - 256, 7);
        bw.put_code_msb(0, 5);
        remaining -= len;
    }
    while (remaining > 0) {
        bw.put_code_msb(48 + val, 8);
        --remaining;
    }
    bw.put_code_msb(0, 7); // end of block
    bw.flush();

    std::vector<unsigned char> plain(count, val);
    uint32_t a = adler32_calc(plain.data(), count);
    bw.bytes.push_back(static_cast<unsigned char>((a >> 24) & 0xFF));
    bw.bytes.push_back(static_cast<unsigned char>((a >> 16) & 0xFF));
    bw.bytes.push_back(static_cast<unsigned char>((a >> 8) & 0xFF));
    bw.bytes.push_back(static_cast<unsigned char>(a & 0xFF));
    return bw.bytes;
}

struct Object {
    void** vtable;
};

using DStorageGetFactoryFn = long(__stdcall*)(const void*, void**);
using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using EnqueueSignalFn = void(__stdcall*)(void*, void*, u64);
using SubmitFn = void(__stdcall*)(void*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
using GetHResultFn = long(__stdcall*)(void*, unsigned);
using OpenFileFn = long(__stdcall*)(void*, const wchar_t*, const void*, void**);
using CloseFileFn = void(__stdcall*)(void*);
using GetErrorEventFn = HANDLE(__stdcall*)(void*);
using RetrieveErrorRecordFn = void(__stdcall*)(void*, void*);
using ReleaseFn = unsigned long(__stdcall*)(void*);

struct SubmitGate {
    void** vtable = nullptr;
    HANDLE entered = nullptr;
    HANDLE resume = nullptr;
    volatile LONG resumed = 0;
};

long __stdcall hold_worker(void* fence, u64) {
    auto& gate = *static_cast<SubmitGate*>(fence);
    SetEvent(gate.entered);
    if (WaitForSingleObject(gate.resume, 5000) != WAIT_OBJECT_0) return E_FAIL;
    InterlockedExchange(&gate.resumed, 1);
    return S_OK;
}

// DSTORAGE_ERROR_RECORD offsets as the Xbox GDK reference (dstorage_xs.h) publishes the structure on x64,
// declared here independently of src/startup_stubs/dstorage_error_record.h so a wrong stand-in layout
// cannot agree with itself. FirstFailure is pointer-aligned, so HResult and CommandType share its
// first 8 bytes and the Request arm of the union starts at 0x10.
namespace xbox_error_record {
constexpr std::size_t failure_count = 0x000, hresult = 0x008, command_type = 0x00c, options = 0x010,
                      destination = 0x018, destination_size = 0x020, filename = 0x024, file_offset = 0x230,
                      source = 0x238, source_size = 0x240, cancellation_tag = 0x248, name = 0x250, size = 0x290;
constexpr std::int32_t command_none = -1, command_request = 0;  // DSTORAGE_COMMAND_TYPE
template <class T> T read(const unsigned char* record, std::size_t offset)
{
    T value;
    std::memcpy(&value, record + offset, sizeof value);
    return value;
}
}  // namespace xbox_error_record

bool dstorage_error_record_checks()
{
    using x2w_startup::StorageRequest;
    namespace rec = xbox_error_record;
    auto check = [](bool condition, const char* label) {
        if (!condition) std::fprintf(stderr, "dstorage error record: %s\n", label);
        return condition;
    };
    HMODULE dll = LoadLibraryW(L"XCRDAPI.dll");
    if (!check(dll != nullptr, "load XCRDAPI")) return false;
    auto factory_export = reinterpret_cast<DStorageGetFactoryFn>(GetProcAddress(dll, "DStorageGetFactory"));
    void* factory_ptr = nullptr;
    if (!check(factory_export && factory_export(nullptr, &factory_ptr) == S_OK && factory_ptr, "get factory"))
        return false;
    auto* factory = static_cast<Object*>(factory_ptr);

    wchar_t workdir[MAX_PATH]{}, filename[MAX_PATH]{};
    const DWORD dir_chars = GetCurrentDirectoryW(MAX_PATH, workdir);
    if (!check(dir_chars && dir_chars < MAX_PATH, "get fixture directory")) return false;
    if (!GetTempFileNameW(workdir, L"x2w", 0, filename)) {
        std::fprintf(stderr, "dstorage error record: create source path (Win32 %lu)\n", GetLastError());
        return false;
    }
    const std::array<unsigned char, 4> input{0x12, 0x34, 0x56, 0x78};
    HANDLE writer = CreateFileW(filename, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    DWORD written = 0;
    const bool wrote = writer != INVALID_HANDLE_VALUE &&
        WriteFile(writer, input.data(), static_cast<DWORD>(input.size()), &written, nullptr) && written == input.size();
    if (writer != INVALID_HANDLE_VALUE) CloseHandle(writer);
    void* file_ptr = nullptr;
    const bool opened = wrote &&
        reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, filename, nullptr, &file_ptr) == S_OK && file_ptr;
    DeleteFileW(filename);  // The stand-in's open handle shares delete access.
    if (!check(opened, "open source-built file")) return false;
    auto* file = static_cast<Object*>(file_ptr);

    struct Queue {
        Object* object = nullptr;
        Object* status = nullptr;
        HANDLE event = nullptr;
        ~Queue() {
            if (event) CloseHandle(event);
            if (object) reinterpret_cast<ReleaseFn>(object->vtable[2])(object);
        }
        void enqueue(const StorageRequest& request) const {
            reinterpret_cast<EnqueueRequestFn>(object->vtable[3])(object, &request);
        }
        HRESULT submit() const {
            reinterpret_cast<EnqueueStatusFn>(object->vtable[4])(object, status, 0);
            reinterpret_cast<SubmitFn>(object->vtable[6])(object);
            auto complete = reinterpret_cast<IsCompleteFn>(status->vtable[3]);
            const auto start = GetTickCount64();
            while (!complete(status, 0) && GetTickCount64() - start < 5000) Sleep(1);
            return reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0);
        }
        void retrieve(unsigned char* record) const {
            reinterpret_cast<RetrieveErrorRecordFn>(object->vtable[10])(object, record);
        }
    } failing, successful;

    auto create_queue = [&](Queue& q) {
        alignas(u64) unsigned char desc[40]{};
        const std::uint16_t capacity = 16;
        std::memcpy(desc + 8, &capacity, sizeof capacity);
        void* ptr = nullptr;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &ptr) != S_OK || !ptr)
            return false;
        q.object = static_cast<Object*>(ptr);
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "ErrorStatus", nullptr, &ptr) != S_OK || !ptr)
            return false;
        q.status = static_cast<Object*>(ptr);
        q.event = reinterpret_cast<GetErrorEventFn>(q.object->vtable[9])(q.object);
        return q.event && WaitForSingleObject(q.event, 0) == WAIT_TIMEOUT;
    };
    if (!check(create_queue(failing) && create_queue(successful), "new queues have unsignaled events")) return false;

    const std::array<unsigned char, 4> sentinel{0xaa, 0xaa, 0xaa, 0xaa};
    auto first_dest = sentinel, next_dest = sentinel, good_dest = sentinel;
    const StorageRequest first{1, first_dest.data(), 0, 4, file, 8, 3, 0, 0x1020304050607080ull, "first read"};
    const StorageRequest next{0, next_dest.data(), 0, 4, file, 3, 4, 0, 0x2233445566778899ull, "crossing read"};
    const StorageRequest good{0, good_dest.data(), 0, 4, file, 0, 4, 0, 0, "good read"};
    failing.enqueue(first);
    failing.enqueue(next);
    if (!check(WaitForSingleObject(failing.event, 0) == WAIT_TIMEOUT, "event signaled before Submit")) return false;
    if (!check(failing.submit() == HRESULT_FROM_WIN32(ERROR_HANDLE_EOF) && first_dest == sentinel && next_dest == sentinel,
               "out-of-range requests failed and preserved their destinations")) return false;
    if (!check(WaitForSingleObject(failing.event, 0) == WAIT_OBJECT_0 &&
               WaitForSingleObject(failing.event, 0) == WAIT_OBJECT_0, "failed request did not leave the error event signaled"))
        return false;
    successful.enqueue(good);
    if (!check(successful.submit() == S_OK && good_dest == input, "success-only queue returned the file bytes") ||
        !check(WaitForSingleObject(successful.event, 0) == WAIT_TIMEOUT, "success-only queue's event was signaled"))
        return false;
    reinterpret_cast<CloseFileFn>(file->vtable[3])(file);

    struct {
        alignas(8) unsigned char record[rec::size];
        u64 after;
    } buffer{{}, 0xabcdef0123456789ull};
    auto retrieve = [&](const Queue& q) -> const unsigned char* {
        std::memset(buffer.record, 0xaa, sizeof buffer.record);
        q.retrieve(buffer.record);
        return buffer.record;
    };
    auto text_equal = [](const unsigned char* record, std::size_t offset, const char* text, std::size_t bytes) {
        return strncmp(reinterpret_cast<const char*>(record + offset), text, bytes) == 0;
    };
    // An empty record: no failures and CommandType NONE (the docs leave the rest undefined), guard intact.
    auto empty_record = [&] {
        return rec::read<std::uint32_t>(buffer.record, rec::failure_count) == 0 &&
               rec::read<std::int32_t>(buffer.record, rec::command_type) == rec::command_none &&
               buffer.after == 0xabcdef0123456789ull;
    };

    const unsigned char* record = retrieve(failing);
    wchar_t recorded_name[MAX_PATH];
    std::memcpy(recorded_name, record + rec::filename, sizeof recorded_name);
    if (!check(buffer.after == 0xabcdef0123456789ull && rec::read<std::uint32_t>(record, rec::failure_count) == 2,
               "expected two failures and an intact guard") ||
        !check(rec::read<std::int32_t>(record, rec::command_type) == rec::command_request,
               "first failure's CommandType is not DSTORAGE_COMMAND_TYPE_REQUEST") ||
        !check(rec::read<HRESULT>(record, rec::hresult) == HRESULT_FROM_WIN32(ERROR_HANDLE_EOF) && rec::read<u64>(record, rec::options) == first.options &&
               rec::read<void*>(record, rec::destination) == first.destination &&
               rec::read<std::uint32_t>(record, rec::destination_size) == first.destination_size &&
               rec::read<u64>(record, rec::file_offset) == first.file_offset &&
               rec::read<void*>(record, rec::source) == nullptr &&
               rec::read<std::uint32_t>(record, rec::source_size) == first.source_size &&
               rec::read<u64>(record, rec::cancellation_tag) == first.cancellation_tag &&
               wcsncmp(recorded_name, filename, MAX_PATH) == 0 && text_equal(record, rec::name, first.name, 64),
               "record did not preserve the first file request"))
        return false;
    if (!check(WaitForSingleObject(failing.event, 0) == WAIT_TIMEOUT, "retrieve did not reset the event")) return false;
    retrieve(failing);
    if (!check(empty_record(), "second retrieve did not clear the failure count and first record")) return false;
    retrieve(successful);
    if (!check(empty_record(), "success-only queue returned a failure record")) return false;

    auto memory = sentinel;
    auto source = input;
    // A 70-character name: the record keeps its first 63 characters and a NUL (CHAR Name[64]).
    static const char long_name[] = "memory read with a name longer than the record's sixty-four bytes 0123";
    static_assert(sizeof long_name - 1 == 70);
    const StorageRequest too_small{1ull << 22, memory.data(), 0, 2, source.data(), 0, 4, 0, 0xaabbccddull, long_name};
    failing.enqueue(too_small);
    if (!check(failing.submit() == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) && memory == sentinel, "short memory destination was not refused") ||
        !check(WaitForSingleObject(failing.event, 0) == WAIT_OBJECT_0, "next failure did not re-signal the event")) return false;
    const unsigned char* memory_failure = retrieve(failing);
    return check(rec::read<std::uint32_t>(memory_failure, rec::failure_count) == 1 &&
                 rec::read<std::int32_t>(memory_failure, rec::command_type) == rec::command_request &&
                 rec::read<HRESULT>(memory_failure, rec::hresult) == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) &&
                 rec::read<u64>(memory_failure, rec::options) == too_small.options &&
                 rec::read<void*>(memory_failure, rec::destination) == too_small.destination &&
                 rec::read<std::uint32_t>(memory_failure, rec::destination_size) == too_small.destination_size &&
                 rec::read<wchar_t>(memory_failure, rec::filename) == 0 &&
                 rec::read<u64>(memory_failure, rec::file_offset) == 0 &&
                 rec::read<void*>(memory_failure, rec::source) == too_small.source &&
                 rec::read<std::uint32_t>(memory_failure, rec::source_size) == too_small.source_size &&
                 rec::read<u64>(memory_failure, rec::cancellation_tag) == too_small.cancellation_tag &&
                 text_equal(memory_failure, rec::name, long_name, 63) && memory_failure[rec::name + 63] == 0 &&
                 buffer.after == 0xabcdef0123456789ull && WaitForSingleObject(failing.event, 0) == WAIT_TIMEOUT,
                 "record did not restart with the next memory request");
}

// Page-destination requests whose page array changes between EnqueueRequest and Submit:
//   replaced:   a non-zero entry is replaced; the new page receives the data and the old page keeps
//               its sentinel.
//   unreadable: the array becomes unreadable; that request fails without writing any page, and a
//               valid request queued after it on the same queue still completes.
// Prints one line per case; returns false on the first mismatch.
bool dstorage_page_array_checks() {
    _putenv("X2W_XMEM_PERMISSIVE=1");
    HMODULE xcrdapi = LoadLibraryA("XCRDAPI.dll");
    if (!xcrdapi) return false;
    auto get_factory = reinterpret_cast<DStorageGetFactoryFn>(GetProcAddress(xcrdapi, "DStorageGetFactory"));
    void* factory_ptr = nullptr;
    if (!get_factory || get_factory(nullptr, &factory_ptr) != 0 || !factory_ptr) return false;
    auto* factory = static_cast<Object*>(factory_ptr);

    const size_t page = 64 * 1024;
    std::vector<unsigned char> plain(2 * page);
    for (size_t i = 0; i < plain.size(); ++i) plain[i] = static_cast<unsigned char>((i % 251) ^ 0x5a);
    const auto two_pages = make_stored_zlib(plain.data(), plain.size());
    const auto one_page = make_stored_zlib(plain.data(), page);

    size_t page_count = 3;
    u64 pages[3] = {};
    if (!XMemAllocatePhysicalPages(0x20400000ull, &page_count, pages) || page_count != 3) return false;
    // Each physical page gets its own view, so every page can be checked after a request.
    void* view[3] = {};
    for (int i = 0; i < 3; ++i) {
        const ULONG_PTR token = pages[i];
        view[i] = XMemVirtualAlloc(nullptr, page, 0x20402000, 0x21, PAGE_READWRITE);
        if (!view[i] || XMemMapPhysicalPages(view[i], 1, &token) != view[i]) return false;
    }
    auto fill = [&](unsigned char value) {
        for (void* v : view) std::memset(v, value, page);
    };
    auto untouched = [&](int i, unsigned char value) {
        const auto* bytes = static_cast<const unsigned char*>(view[i]);
        for (size_t k = 0; k < page; ++k)
            if (bytes[k] != value) return false;
        return true;
    };
    auto request_for = [](unsigned char* request, const ULONG_PTR* tokens, unsigned size, const std::vector<unsigned char>& data,
                          unsigned decoded) {
        std::memset(request, 0, 0x48);
        *reinterpret_cast<u64*>(request + 0x00) = 1 | (1ull << 21) | 0x400000;  // zlib, page destination, memory source
        *reinterpret_cast<const ULONG_PTR**>(request + 0x08) = tokens;
        *reinterpret_cast<unsigned*>(request + 0x18) = size;
        *reinterpret_cast<const unsigned char**>(request + 0x20) = data.data();
        *reinterpret_cast<unsigned*>(request + 0x30) = static_cast<unsigned>(data.size());
        *reinterpret_cast<unsigned*>(request + 0x38) = decoded;
    };
    auto make_queue = [&](unsigned statuses, Object** queue, Object** status) {
        void* status_ptr = nullptr;
        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, statuses, "Status", nullptr, &status_ptr) != 0 ||
            !status_ptr || reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr)
            return false;
        *queue = static_cast<Object*>(queue_ptr);
        *status = static_cast<Object*>(status_ptr);
        return true;
    };
    auto result_of = [](Object* status, unsigned index, long* hr) {
        for (int i = 0; i < 200 && !reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, index); ++i) Sleep(10);
        if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, index)) return false;
        *hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, index);
        return true;
    };

    // replaced: enqueue with {page 0, page 1}, then replace entry 1 with page 2 before Submit.
    {
        Object *queue = nullptr, *status = nullptr;
        if (!make_queue(1, &queue, &status)) return false;
        fill(0xAA);
        ULONG_PTR tokens[2] = {pages[0], pages[1]};
        unsigned char request[0x48];
        request_for(request, tokens, static_cast<unsigned>(2 * page), two_pages, static_cast<unsigned>(2 * page));
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
        tokens[1] = pages[2];
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
        long hr = -1;
        const bool done = result_of(status, 0, &hr);
        const bool first = done && std::memcmp(view[0], plain.data(), page) == 0;
        const bool moved = done && std::memcmp(view[2], plain.data() + page, page) == 0;
        const bool old_kept = done && untouched(1, 0xAA);
        std::printf("replaced: hr=0x%08lx first page %s, new page %s, old page %s\n", static_cast<unsigned long>(hr),
                    first ? "ok" : "WRONG", moved ? "ok" : "WRONG", old_kept ? "kept" : "WRITTEN");
        if (!done || hr != 0 || !first || !moved || !old_kept) return false;
    }

    // unreadable: the first request's array is made no-access before Submit; a valid one follows.
    {
        Object *queue = nullptr, *status = nullptr;
        if (!make_queue(2, &queue, &status)) return false;
        fill(0xAA);
        auto* tokens = static_cast<ULONG_PTR*>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!tokens) return false;
        tokens[0] = pages[0];
        tokens[1] = pages[1];
        unsigned char bad[0x48];
        request_for(bad, tokens, static_cast<unsigned>(2 * page), two_pages, static_cast<unsigned>(2 * page));
        const ULONG_PTR good_tokens[1] = {pages[2]};
        unsigned char good[0x48];
        request_for(good, good_tokens, static_cast<unsigned>(page), one_page, static_cast<unsigned>(page));
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, bad);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, good);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 1);
        DWORD old_protection = 0;
        if (!VirtualProtect(tokens, 4096, PAGE_NOACCESS, &old_protection)) return false;
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
        long bad_hr = 0, good_hr = -1;
        const bool done = result_of(status, 0, &bad_hr) && result_of(status, 1, &good_hr);
        const bool kept = done && untouched(0, 0xAA) && untouched(1, 0xAA);
        const bool good_written = done && std::memcmp(view[2], plain.data(), page) == 0;
        std::printf("unreadable: failed request hr=0x%08lx, pages %s; following request hr=0x%08lx, page %s\n",
                    static_cast<unsigned long>(bad_hr), kept ? "kept" : "WRITTEN", static_cast<unsigned long>(good_hr),
                    good_written ? "ok" : "WRONG");
        VirtualFree(tokens, 0, MEM_RELEASE);
        if (!done || bad_hr == 0 || !kept || good_hr != 0 || !good_written) return false;
    }

    for (void* v : view) {
        XMemMapPhysicalPages(v, 1, nullptr);
        VirtualFree(v, 0, MEM_RELEASE);
    }
    XMemFreePhysicalPages(3, pages);
    return true;
}

// #1843: a new status array's slots start complete with S_OK; EnqueueStatus makes one pending until
// its turn, and the others are untouched.
bool dstorage_status_initial_checks() {
    _putenv("X2W_XMEM_PERMISSIVE=1");
    HMODULE xcrdapi = LoadLibraryA("XCRDAPI.dll");
    if (!xcrdapi) return false;
    auto get_factory = reinterpret_cast<DStorageGetFactoryFn>(GetProcAddress(xcrdapi, "DStorageGetFactory"));
    void* factory_ptr = nullptr;
    if (!get_factory || get_factory(nullptr, &factory_ptr) != 0 || !factory_ptr) return false;
    auto* factory = static_cast<Object*>(factory_ptr);

    const unsigned capacity = 4;
    void* status_ptr = nullptr;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, capacity, "InitialStatus", nullptr, &status_ptr) != 0 ||
        !status_ptr)
        return false;
    auto* status = static_cast<Object*>(status_ptr);
    auto complete = [&](unsigned i) { return reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, i); };
    auto result = [&](unsigned i) { return reinterpret_cast<GetHResultFn>(status->vtable[4])(status, i); };
    const long e_pending = static_cast<long>(0x8000000A), e_invalidarg = static_cast<long>(0x80070057);

    // A fresh array: every slot complete with S_OK. An index past the end stays refused.
    for (unsigned i = 0; i < capacity; ++i) {
        if (!complete(i) || result(i) != S_OK) {
            std::printf("dstorage-status-initial: fresh slot %u complete=%d hr=0x%08lx\n", i, complete(i),
                        static_cast<unsigned long>(result(i)));
            return false;
        }
    }
    if (complete(capacity) || result(capacity) != e_invalidarg) return false;

    // Hold the queue's worker in a fence signal, then enqueue a status into slot 2 behind it.
    void* queue_ptr = nullptr;
    alignas(void*) unsigned char desc[40] = {};
    *reinterpret_cast<uint16_t*>(desc + 8) = 16;
    if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr)
        return false;
    auto* queue = static_cast<Object*>(queue_ptr);
    static SubmitGate gate;
    static void* fence_vtable[11] = {};
    fence_vtable[10] = reinterpret_cast<void*>(&hold_worker);
    gate.vtable = fence_vtable;
    gate.entered = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    gate.resume = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!gate.entered || !gate.resume) return false;
    reinterpret_cast<EnqueueSignalFn>(queue->vtable[5])(queue, &gate, 1);
    reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 2);
    reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
    if (WaitForSingleObject(gate.entered, 3000) != WAIT_OBJECT_0) {
        SetEvent(gate.resume);
        return false;
    }
    // Before its turn, slot 2 is pending and the other slots keep their S_OK.
    bool held = !complete(2) && result(2) == e_pending;
    for (unsigned i : {0u, 1u, 3u}) held = held && complete(i) && result(i) == S_OK;
    SetEvent(gate.resume);
    for (int i = 0; i < 200 && !complete(2); ++i) Sleep(10);
    const bool done = complete(2) && result(2) == S_OK && InterlockedCompareExchange(&gate.resumed, 0, 0);
    if (!held || !done) {
        std::printf("dstorage-status-initial: held=%d done=%d slot2 hr=0x%08lx\n", held, done,
                    static_cast<unsigned long>(result(2)));
        return false;
    }
    return true;
}

bool dstorage_checks() {
    _putenv("X2W_XMEM_PERMISSIVE=1");

    HMODULE xcrdapi = LoadLibraryA("XCRDAPI.dll");
    if (!xcrdapi) return false;
    auto get_factory = reinterpret_cast<DStorageGetFactoryFn>(GetProcAddress(xcrdapi, "DStorageGetFactory"));
    if (!get_factory) return false;

    void* factory_ptr = nullptr;
    if (get_factory(nullptr, &factory_ptr) != 0 || !factory_ptr) return false;
    auto* factory = static_cast<Object*>(factory_ptr);

    const size_t plain_size = 128 * 1024;
    std::vector<unsigned char> plain(plain_size);
    for (size_t i = 0; i < plain_size; ++i) plain[i] = static_cast<unsigned char>((i % 251) ^ 0x5a);
    const auto comp = make_stored_zlib(plain.data(), plain_size);

    const char* cases[] = { "multipage", "short", "missing_grant", "linear_short" };
    for (const char* mode : cases) {
        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "Status", nullptr, &status_ptr) != 0 || !status_ptr)
            return false;
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr)
            return false;
        auto* queue = static_cast<Object*>(queue_ptr);

        if (std::strcmp(mode, "linear_short") == 0) {
            std::vector<unsigned char> dest(64 * 1024, 0xAA);
            unsigned char request[0x48] = {};
            *reinterpret_cast<u64*>(request + 0x00) = 1 | 0x400000;
            *reinterpret_cast<void**>(request + 0x08) = dest.data();
            *reinterpret_cast<u64*>(request + 0x10) = 0;
            *reinterpret_cast<unsigned*>(request + 0x18) = static_cast<unsigned>(dest.size());
            *reinterpret_cast<const unsigned char**>(request + 0x20) = comp.data();
            *reinterpret_cast<u64*>(request + 0x28) = 0;
            *reinterpret_cast<unsigned*>(request + 0x30) = static_cast<unsigned>(comp.size());
            *reinterpret_cast<unsigned*>(request + 0x38) = static_cast<unsigned>(plain_size);

            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
            reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

            for (int i = 0; i < 200; ++i) {
                if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) break;
                Sleep(10);
            }
            if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) return false;
            long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0);
            if (hr == 0) return false;
            for (unsigned char b : dest) {
                if (b != 0xAA) return false;
            }
        } else {
            size_t page_count = 2;
            u64 pages[2] = {};
            if (!XMemAllocatePhysicalPages(0x20400000ull, &page_count, pages) || page_count != 2) return false;

            ULONG_PTR tokens[2] = { pages[0], pages[1] };
            unsigned destination_size = 128 * 1024;

            void* va = XMemVirtualAlloc(nullptr, 2 * 65536, 0x20402000, 0x21, PAGE_READWRITE);
            if (!va) return false;
            if (XMemMapPhysicalPages(va, 2, tokens) != va) return false;
            std::memset(va, 0xAA, 128 * 1024);

            if (std::strcmp(mode, "short") == 0) {
                destination_size = 64 * 1024;
            } else if (std::strcmp(mode, "missing_grant") == 0) {
                tokens[1] = 0;
            }

            unsigned char request[0x48] = {};
            *reinterpret_cast<u64*>(request + 0x00) = 1 | (1ull << 21) | 0x400000;
            *reinterpret_cast<ULONG_PTR**>(request + 0x08) = tokens;
            *reinterpret_cast<u64*>(request + 0x10) = 0;
            *reinterpret_cast<unsigned*>(request + 0x18) = destination_size;
            *reinterpret_cast<const unsigned char**>(request + 0x20) = comp.data();
            *reinterpret_cast<u64*>(request + 0x28) = 0;
            *reinterpret_cast<unsigned*>(request + 0x30) = static_cast<unsigned>(comp.size());
            *reinterpret_cast<unsigned*>(request + 0x38) = static_cast<unsigned>(plain_size);

            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
            reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

            for (int i = 0; i < 200; ++i) {
                if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) break;
                Sleep(10);
            }
            if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0)) return false;
            long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0);

            if (std::strcmp(mode, "multipage") == 0) {
                if (hr != 0) return false;
                if (std::memcmp(va, plain.data(), plain_size) != 0) return false;
            } else {
                if (hr == 0) return false;
                const auto* bytes = static_cast<const unsigned char*>(va);
                for (size_t i = 0; i < 128 * 1024; ++i) {
                    if (bytes[i] != 0xAA) return false;
                }
            }

            XMemMapPhysicalPages(va, 2, nullptr);
            VirtualFree(va, 0, MEM_RELEASE);
            XMemFreePhysicalPages(2, pages);
        }
    }

    // Test chunked streaming reads across 4 requests on a file source (Refs #1136)
    {
        const size_t total_chunks = 4;
        const size_t chunk_dest_size = 64 * 1024;
        const size_t total_plain_size = total_chunks * chunk_dest_size; // 256 KiB
        std::vector<unsigned char> full_plain(total_plain_size);
        for (size_t i = 0; i < total_plain_size; ++i) full_plain[i] = static_cast<unsigned char>((i % 239) ^ 0xa5);
        const auto full_comp = make_stored_zlib(full_plain.data(), total_plain_size);

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        // Padded to 600 KiB so offset 500 KiB can read zeroes from file and test zlib continuation bounds
        std::vector<unsigned char> file_bytes(600 * 1024, 0);
        std::memcpy(file_bytes.data(), full_comp.data(), full_comp.size());

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, static_cast<unsigned>(total_chunks), "ChunkedStatus", nullptr, &status_ptr) != 0 || !status_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        // The last four bytes are zero padding. Only the first request writes;
        // the short read fails, and the existing empty-request path succeeds.
        const u64 offsets[]{file_bytes.size() - 4, file_bytes.size() - 3, UINT64_MAX};
        const unsigned counts[]{4, 4, 0};
        unsigned char destinations[3][6];
        std::memset(destinations, 0xa5, sizeof(destinations));
        for (unsigned i = 0; i < 3; ++i) {
            alignas(8) unsigned char req[0x48]{};
            *reinterpret_cast<void**>(req + 0x08) = destinations[i] + 1;
            *reinterpret_cast<unsigned*>(req + 0x18) = 4;
            *reinterpret_cast<void**>(req + 0x20) = file_obj;
            *reinterpret_cast<u64*>(req + 0x28) = offsets[i];
            *reinterpret_cast<unsigned*>(req + 0x30) = counts[i];
            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, i);
        }
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
        for (int i = 0; i < 200 && !reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 2); ++i) Sleep(10);
        bool bounds_ok = true;
        for (unsigned i = 0; i < 3; ++i) {
            const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, i);
            bounds_ok &= hr == (i == 1 ? HRESULT_FROM_WIN32(ERROR_HANDLE_EOF) : S_OK);
            for (unsigned j = 0; j < 6; ++j)
                bounds_ok &= destinations[i][j] == (i == 0 && j >= 1 && j <= 4 ? 0 : 0xa5);
        }
        if (!bounds_ok) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }

        std::vector<std::vector<unsigned char>> chunks(total_chunks, std::vector<unsigned char>(chunk_dest_size, 0));
        for (size_t k = 0; k < total_chunks; ++k) {
            unsigned char req[0x48] = {};
            *reinterpret_cast<u64*>(req + 0x00) = 1; // zlib file request
            *reinterpret_cast<void**>(req + 0x08) = chunks[k].data();
            *reinterpret_cast<u64*>(req + 0x10) = 0;
            *reinterpret_cast<unsigned*>(req + 0x18) = static_cast<unsigned>(chunk_dest_size);
            *reinterpret_cast<void**>(req + 0x20) = file_obj;
            *reinterpret_cast<u64*>(req + 0x28) = k * chunk_dest_size; // offset stepped by destination size
            *reinterpret_cast<unsigned*>(req + 0x30) = static_cast<unsigned>(full_comp.size());
            *reinterpret_cast<unsigned*>(req + 0x38) = static_cast<unsigned>(total_plain_size);

            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, static_cast<unsigned>(k));
        }
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, static_cast<unsigned>(total_chunks - 1))) break;
            Sleep(10);
        }
        bool ok = true;
        for (size_t k = 0; k < total_chunks; ++k) {
            if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, static_cast<unsigned>(k))) ok = false;
            long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, static_cast<unsigned>(k));
            if (hr != 0) ok = false;
            if (std::memcmp(chunks[k].data(), full_plain.data() + k * chunk_dest_size, chunk_dest_size) != 0) ok = false;
        }
        if (!ok) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }

        // Negative control: chunk request with out-of-bounds or wrapping offset must fail and leave destination untouched
        void* neg_status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 2, "ChunkNegStatus", nullptr, &neg_status_ptr) != 0 || !neg_status_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* neg_status = static_cast<Object*>(neg_status_ptr);

        std::vector<unsigned char> dest1(64 * 1024, 0xEE); // sentinel
        std::vector<unsigned char> dest2(64 * 1024, 0xDD); // sentinel for wrap test

        // Piece 1 with invalid offset (500 KiB > 256 KiB)
        unsigned char req1[0x48] = {};
        *reinterpret_cast<u64*>(req1 + 0x00) = 1;
        *reinterpret_cast<void**>(req1 + 0x08) = dest1.data();
        *reinterpret_cast<unsigned*>(req1 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req1 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1 + 0x28) = 500 * 1024;
        *reinterpret_cast<unsigned*>(req1 + 0x30) = static_cast<unsigned>(full_comp.size());
        *reinterpret_cast<unsigned*>(req1 + 0x38) = static_cast<unsigned>(total_plain_size);

        // Piece 2 with near-wrapping offset (UINT64_MAX - 0xFFFF)
        unsigned char req2[0x48] = {};
        *reinterpret_cast<u64*>(req2 + 0x00) = 1;
        *reinterpret_cast<void**>(req2 + 0x08) = dest2.data();
        *reinterpret_cast<unsigned*>(req2 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req2 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req2 + 0x28) = 0xFFFFFFFFFFFF0000ull;
        *reinterpret_cast<unsigned*>(req2 + 0x30) = static_cast<unsigned>(full_comp.size());
        *reinterpret_cast<unsigned*>(req2 + 0x38) = static_cast<unsigned>(total_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, neg_status, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req2);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, neg_status, 1);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(neg_status->vtable[3])(neg_status, 1)) break;
            Sleep(10);
        }
        if (!reinterpret_cast<IsCompleteFn>(neg_status->vtable[3])(neg_status, 0) ||
            !reinterpret_cast<IsCompleteFn>(neg_status->vtable[3])(neg_status, 1)) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        long hr1 = reinterpret_cast<GetHResultFn>(neg_status->vtable[4])(neg_status, 0);
        long hr2 = reinterpret_cast<GetHResultFn>(neg_status->vtable[4])(neg_status, 1);
        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        if (hr1 == 0 || hr2 == 0) return false;
        for (unsigned char b : dest1) {
            if (b != 0xEE) return false;
        }
        for (unsigned char b : dest2) {
            if (b != 0xDD) return false;
        }
    }

    // Control: standalone valid stream sharing cache-key fields decodes its own bytes (Refs #1214)
    {
        const size_t stream1_plain_size = 64 * 1024;
        const size_t stream2_plain_size = 64 * 1024;
        std::vector<unsigned char> plain1(stream1_plain_size, 0x33);
        std::vector<unsigned char> plain2(stream2_plain_size, 0x77);
        const auto comp1 = make_stored_zlib(plain1.data(), stream1_plain_size);
        const auto comp2 = make_stored_zlib(plain2.data(), stream2_plain_size);
        const unsigned shared_compressed_size = static_cast<unsigned>(comp1.size());

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        const size_t offset2 = 128 * 1024;
        std::vector<unsigned char> file_bytes(offset2 + comp2.size(), 0);
        std::memcpy(file_bytes.data(), comp1.data(), comp1.size());
        std::memcpy(file_bytes.data() + offset2, comp2.data(), comp2.size());

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 3, "SharedStatus", nullptr, &status_ptr) != 0 || !status_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        std::vector<unsigned char> dest1_0(32 * 1024, 0);
        std::vector<unsigned char> dest1_1(32 * 1024, 0);
        std::vector<unsigned char> dest2(64 * 1024, 0);

        // Stream 1 piece 0: offset 0, compressed size S, destination 32 KiB
        unsigned char req1_0[0x48] = {};
        *reinterpret_cast<u64*>(req1_0 + 0x00) = 1;
        *reinterpret_cast<void**>(req1_0 + 0x08) = dest1_0.data();
        *reinterpret_cast<unsigned*>(req1_0 + 0x18) = 32 * 1024;
        *reinterpret_cast<void**>(req1_0 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1_0 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req1_0 + 0x30) = shared_compressed_size;
        *reinterpret_cast<unsigned*>(req1_0 + 0x38) = static_cast<unsigned>(stream1_plain_size);

        // Stream 1 piece 1: offset 32 KiB, compressed size S, destination 32 KiB
        unsigned char req1_1[0x48] = {};
        *reinterpret_cast<u64*>(req1_1 + 0x00) = 1;
        *reinterpret_cast<void**>(req1_1 + 0x08) = dest1_1.data();
        *reinterpret_cast<unsigned*>(req1_1 + 0x18) = 32 * 1024;
        *reinterpret_cast<void**>(req1_1 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1_1 + 0x28) = 32 * 1024;
        *reinterpret_cast<unsigned*>(req1_1 + 0x30) = shared_compressed_size;
        *reinterpret_cast<unsigned*>(req1_1 + 0x38) = static_cast<unsigned>(stream1_plain_size);

        // Stream 2: offset 128 KiB, exact same compressed size S, destination 64 KiB
        unsigned char req2[0x48] = {};
        *reinterpret_cast<u64*>(req2 + 0x00) = 1;
        *reinterpret_cast<void**>(req2 + 0x08) = dest2.data();
        *reinterpret_cast<unsigned*>(req2 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req2 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req2 + 0x28) = offset2;
        *reinterpret_cast<unsigned*>(req2 + 0x30) = shared_compressed_size;
        *reinterpret_cast<unsigned*>(req2 + 0x38) = static_cast<unsigned>(stream2_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1_0);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1_1);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 1);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req2);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 2);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 2)) break;
            Sleep(10);
        }
        if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0) ||
            !reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1) ||
            !reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 2)) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        long hr0 = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0);
        long hr1 = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 1);
        long hr2 = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 2);
        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        if (hr0 != 0 || hr1 != 0 || hr2 != 0) return false;
        for (unsigned char b : dest1_0) {
            if (b != 0x33) return false;
        }
        for (unsigned char b : dest1_1) {
            if (b != 0x33) return false;
        }
        for (unsigned char b : dest2) {
            if (b != 0x77) return false;
        }
    }

    // Control: standalone stream with damaged checksum at matching offset fails and leaves destination unchanged (Refs #1214)
    {
        const size_t stream_plain_size = 128 * 1024;
        const auto comp1 = make_fixed_deflate_zlib(0x33, stream_plain_size);
        auto comp_bad = make_fixed_deflate_zlib(0x77, stream_plain_size);
        // Corrupt the Adler-32 checksum in comp_bad:
        comp_bad.back() ^= 0xFF;
        const unsigned shared_compressed_size = static_cast<unsigned>(comp1.size());

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        const size_t offset_bad = 64 * 1024; // Matching continuation offset (0 + 64 KiB)
        std::vector<unsigned char> file_bytes(offset_bad + comp_bad.size(), 0);
        std::memcpy(file_bytes.data(), comp1.data(), comp1.size());
        std::memcpy(file_bytes.data() + offset_bad, comp_bad.data(), comp_bad.size());

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 2, "DamagedStatus", nullptr, &status_ptr) != 0 || !status_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        std::vector<unsigned char> dest1(64 * 1024, 0);
        std::vector<unsigned char> dest_bad(64 * 1024, 0xBB); // Sentinel

        // Stream 1 piece 0: offset 0, compressed size S, destination 64 KiB (inflates 128 KiB of 0x33, caches stream)
        unsigned char req1[0x48] = {};
        *reinterpret_cast<u64*>(req1 + 0x00) = 1;
        *reinterpret_cast<void**>(req1 + 0x08) = dest1.data();
        *reinterpret_cast<unsigned*>(req1 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req1 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req1 + 0x30) = shared_compressed_size;
        *reinterpret_cast<unsigned*>(req1 + 0x38) = static_cast<unsigned>(stream_plain_size);

        // Standalone stream at matching continuation offset 64 KiB with damaged checksum
        unsigned char req_bad[0x48] = {};
        *reinterpret_cast<u64*>(req_bad + 0x00) = 1;
        *reinterpret_cast<void**>(req_bad + 0x08) = dest_bad.data();
        *reinterpret_cast<unsigned*>(req_bad + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req_bad + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req_bad + 0x28) = offset_bad;
        *reinterpret_cast<unsigned*>(req_bad + 0x30) = shared_compressed_size;
        *reinterpret_cast<unsigned*>(req_bad + 0x38) = static_cast<unsigned>(stream_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req_bad);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 1);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1)) break;
            Sleep(10);
        }
        if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0) ||
            !reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1)) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        long hr0 = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0);
        long hr_bad = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 1);
        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        // Stream 1 piece 0 must succeed; the damaged standalone stream must fail with non-zero HRESULT
        if (hr0 != 0 || hr_bad == 0) return false;
        for (unsigned char b : dest1) {
            if (b != 0x33) return false;
        }
        // Destination buffer of the damaged stream must remain completely unchanged (sentinel 0xBB)
        for (unsigned char b : dest_bad) {
            if (b != 0xBB) return false;
        }
    }

    // Test short final destination in a multi-piece stream vs invalid oversized decode targeting physical pages (Refs #1144)
    {
        // Multi-piece stream: total plain size = 100 KiB (piece 0 = 64 KiB, piece 1 = 36 KiB short final piece)
        const size_t total_plain_size = 100 * 1024;
        std::vector<unsigned char> full_plain(total_plain_size);
        for (size_t i = 0; i < total_plain_size; ++i) full_plain[i] = static_cast<unsigned char>((i % 251) ^ 0x3c);
        const auto full_comp = make_stored_zlib(full_plain.data(), total_plain_size);

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        std::vector<unsigned char> file_bytes(256 * 1024, 0);
        std::memcpy(file_bytes.data(), full_comp.data(), full_comp.size());

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        // Allocate 2 physical pages for the valid multi-piece stream: page 0 (piece 0) and page 1 (piece 1)
        size_t page_count = 2;
        u64 pages[2] = {};
        if (!XMemAllocatePhysicalPages(0x20400000ull, &page_count, pages) || page_count != 2) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }

        ULONG_PTR tokens_all[2] = { pages[0], pages[1] };
        void* va = XMemVirtualAlloc(nullptr, 2 * 65536, 0x20402000, 0x21, PAGE_READWRITE);
        if (!va || XMemMapPhysicalPages(va, 2, tokens_all) != va) {
            XMemFreePhysicalPages(2, pages);
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        // Prefill both physical pages with guard byte 0xAA
        std::memset(va, 0xAA, 2 * 65536);

        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 2, "ShortFinalStatus", nullptr, &status_ptr) != 0 || !status_ptr) {
            XMemMapPhysicalPages(va, 2, nullptr);
            VirtualFree(va, 0, MEM_RELEASE);
            XMemFreePhysicalPages(2, pages);
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            XMemMapPhysicalPages(va, 2, nullptr);
            VirtualFree(va, 0, MEM_RELEASE);
            XMemFreePhysicalPages(2, pages);
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        // Piece 0: full 64 KiB physical page destination
        ULONG_PTR tokens0[1] = { pages[0] };
        unsigned char req0[0x48] = {};
        *reinterpret_cast<u64*>(req0 + 0x00) = 1 | (1ull << 21); // zlib | page_destination
        *reinterpret_cast<ULONG_PTR**>(req0 + 0x08) = tokens0;
        *reinterpret_cast<u64*>(req0 + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req0 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req0 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req0 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req0 + 0x30) = static_cast<unsigned>(full_comp.size());
        *reinterpret_cast<unsigned*>(req0 + 0x38) = static_cast<unsigned>(total_plain_size);

        // Piece 1: short final destination below 64 KiB (36 KiB) with 1 page grant; neighboring token left null
        ULONG_PTR tokens1[2] = { pages[1], 0 };
        unsigned char req1[0x48] = {};
        *reinterpret_cast<u64*>(req1 + 0x00) = 1 | (1ull << 21); // zlib | page_destination
        *reinterpret_cast<ULONG_PTR**>(req1 + 0x08) = tokens1;
        *reinterpret_cast<u64*>(req1 + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req1 + 0x18) = 36 * 1024; // Short destination capacity (< 64 KiB)
        *reinterpret_cast<void**>(req1 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1 + 0x28) = 64 * 1024; // Stepped by previous chunk size
        *reinterpret_cast<unsigned*>(req1 + 0x30) = static_cast<unsigned>(full_comp.size());
        *reinterpret_cast<unsigned*>(req1 + 0x38) = static_cast<unsigned>(total_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req0);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 1);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1)) break;
            Sleep(10);
        }
        bool valid_complete = reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0) &&
                              reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 1);
        long hr0 = valid_complete ? reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0) : -1;
        long hr1 = valid_complete ? reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 1) : -1;

        bool valid_ok = valid_complete && hr0 == 0 && hr1 == 0;
        const auto* page0_bytes = static_cast<const unsigned char*>(va);
        const auto* page1_bytes = static_cast<const unsigned char*>(va) + 65536;

        // Check piece 0 exact match
        if (valid_ok && std::memcmp(page0_bytes, full_plain.data(), 64 * 1024) != 0) valid_ok = false;
        // Check piece 1 exact 36 KiB match
        if (valid_ok && std::memcmp(page1_bytes, full_plain.data() + 64 * 1024, 36 * 1024) != 0) valid_ok = false;
        // Check guard bytes in page 1 beyond 36 KiB remain intact (0xAA)
        if (valid_ok) {
            for (size_t i = 36 * 1024; i < 65536; ++i) {
                if (page1_bytes[i] != 0xAA) {
                    valid_ok = false;
                    break;
                }
            }
        }
        // Neighboring token in tokens1 must have remained untouched (0)
        if (valid_ok && tokens1[1] != 0) valid_ok = false;

        // Negative control: invalid oversized decode targeting short physical page destination
        // (100 KiB standalone stream into 36 KiB destination with only 1 page grant)
        void* neg_status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "OversizedNegStatus", nullptr, &neg_status_ptr) != 0 || !neg_status_ptr) {
            valid_ok = false;
        }
        auto* neg_status = static_cast<Object*>(neg_status_ptr);

        // Reset page 1 to guard byte 0x55
        std::memset(static_cast<unsigned char*>(va) + 65536, 0x55, 65536);
        ULONG_PTR tokens_neg[2] = { pages[1], 0 }; // Neighboring grant is 0

        unsigned char req_oversized[0x48] = {};
        *reinterpret_cast<u64*>(req_oversized + 0x00) = 1 | (1ull << 21) | 0x400000; // zlib | page_destination | memory_source
        *reinterpret_cast<ULONG_PTR**>(req_oversized + 0x08) = tokens_neg;
        *reinterpret_cast<u64*>(req_oversized + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req_oversized + 0x18) = 36 * 1024; // Capacity 36 KiB < 100 KiB decode
        *reinterpret_cast<const unsigned char**>(req_oversized + 0x20) = full_comp.data();
        *reinterpret_cast<u64*>(req_oversized + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req_oversized + 0x30) = static_cast<unsigned>(full_comp.size());
        *reinterpret_cast<unsigned*>(req_oversized + 0x38) = static_cast<unsigned>(total_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req_oversized);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, neg_status, 0);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(neg_status->vtable[3])(neg_status, 0)) break;
            Sleep(10);
        }
        bool neg_complete = reinterpret_cast<IsCompleteFn>(neg_status->vtable[3])(neg_status, 0);
        long hr_neg = neg_complete ? reinterpret_cast<GetHResultFn>(neg_status->vtable[4])(neg_status, 0) : 0;

        // Oversized decode must fail with non-zero HRESULT
        bool neg_ok = neg_complete && hr_neg != 0;
        // All 64 KiB of destination page must remain completely untouched (guard byte 0x55)
        for (size_t i = 0; i < 65536; ++i) {
            if (page1_bytes[i] != 0x55) {
                neg_ok = false;
                break;
            }
        }
        // Neighboring token must remain untouched (0)
        if (tokens_neg[1] != 0) neg_ok = false;

        XMemMapPhysicalPages(va, 2, nullptr);
        VirtualFree(va, 0, MEM_RELEASE);
        XMemFreePhysicalPages(2, pages);
        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        if (!valid_ok || !neg_ok) return false;
    }

    // Control: valid chunked submission followed by a separate Submit with same-size non-zlib read at matching offset fails cleanly (Refs #1214)
    {
        const size_t stream_plain_size = 128 * 1024;
        const auto comp = make_fixed_deflate_zlib(0x44, stream_plain_size);
        const unsigned comp_size = static_cast<unsigned>(comp.size());
        const size_t offset_cont = 64 * 1024; // Matching continuation offset (0 + 64 KiB)

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        // File layout:
        // Offset 0: valid compressed stream (size comp_size)
        // Offset 64 KiB: non-zlib bytes (0xEE), size comp_size
        std::vector<unsigned char> file_bytes(offset_cont + comp_size, 0xEE);
        std::memcpy(file_bytes.data(), comp.data(), comp_size);

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        void* status_ptr1 = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 2, "ChunkedStatus1", nullptr, &status_ptr1) != 0 || !status_ptr1) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status1 = static_cast<Object*>(status_ptr1);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        std::vector<unsigned char> dest0(64 * 1024, 0);
        std::vector<unsigned char> dest1(64 * 1024, 0);

        // Piece 0: offset 0, compressed size S, destination 64 KiB
        unsigned char req0[0x48] = {};
        *reinterpret_cast<u64*>(req0 + 0x00) = 1;
        *reinterpret_cast<void**>(req0 + 0x08) = dest0.data();
        *reinterpret_cast<unsigned*>(req0 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req0 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req0 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req0 + 0x30) = comp_size;
        *reinterpret_cast<unsigned*>(req0 + 0x38) = static_cast<unsigned>(stream_plain_size);

        // Piece 1: offset 64 KiB, compressed size S, destination 64 KiB
        unsigned char req1[0x48] = {};
        *reinterpret_cast<u64*>(req1 + 0x00) = 1;
        *reinterpret_cast<void**>(req1 + 0x08) = dest1.data();
        *reinterpret_cast<unsigned*>(req1 + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req1 + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req1 + 0x28) = offset_cont;
        *reinterpret_cast<unsigned*>(req1 + 0x30) = comp_size;
        *reinterpret_cast<unsigned*>(req1 + 0x38) = static_cast<unsigned>(stream_plain_size);

        // First Submit: valid 2-slice chunked submission
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req0);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status1, 0);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req1);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status1, 1);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status1->vtable[3])(status1, 1)) break;
            Sleep(10);
        }
        bool comp1_ok = reinterpret_cast<IsCompleteFn>(status1->vtable[3])(status1, 0) &&
                        reinterpret_cast<IsCompleteFn>(status1->vtable[3])(status1, 1);
        long hr0 = comp1_ok ? reinterpret_cast<GetHResultFn>(status1->vtable[4])(status1, 0) : -1;
        long hr1 = comp1_ok ? reinterpret_cast<GetHResultFn>(status1->vtable[4])(status1, 1) : -1;

        if (hr0 != 0 || hr1 != 0) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        for (unsigned char b : dest0) if (b != 0x44) return false;
        for (unsigned char b : dest1) if (b != 0x44) return false;

        // Separate Submit: same queue and same live file, same-size non-zlib read at matching offset 64 KiB
        void* status_ptr2 = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "SeparateStatus", nullptr, &status_ptr2) != 0 || !status_ptr2) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status2 = static_cast<Object*>(status_ptr2);

        std::vector<unsigned char> dest_unrelated(64 * 1024, 0x99); // Sentinel

        unsigned char req_unrelated[0x48] = {};
        *reinterpret_cast<u64*>(req_unrelated + 0x00) = 1; // zlib decompression requested
        *reinterpret_cast<void**>(req_unrelated + 0x08) = dest_unrelated.data();
        *reinterpret_cast<unsigned*>(req_unrelated + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req_unrelated + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req_unrelated + 0x28) = offset_cont; // Matching continuation offset
        *reinterpret_cast<unsigned*>(req_unrelated + 0x30) = comp_size; // Same size
        *reinterpret_cast<unsigned*>(req_unrelated + 0x38) = static_cast<unsigned>(stream_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req_unrelated);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status2, 0);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status2->vtable[3])(status2, 0)) break;
            Sleep(10);
        }
        bool comp2_ok = reinterpret_cast<IsCompleteFn>(status2->vtable[3])(status2, 0);
        long hr_unrelated = comp2_ok ? reinterpret_cast<GetHResultFn>(status2->vtable[4])(status2, 0) : 0;

        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        // The separate read must fail with non-zero HRESULT and leave its sentinel untouched
        if (!comp2_ok || hr_unrelated == 0) return false;
        for (unsigned char b : dest_unrelated) {
            if (b != 0x99) return false;
        }
    }

    // Control: deterministic boundary test across two Submit() calls (Refs #1214, #1235)
    // First Submit(): compressed stream into undersized destination buffer (no continuation in this batch).
    // Second Submit(): matching non-zlib request at the continuation offset, queued while the first batch is gated.
    // Both requests must fail cleanly, and both destination sentinel buffers must remain untouched.
    {
        const size_t stream_plain_size = 128 * 1024;
        const auto comp = make_fixed_deflate_zlib(0x5A, stream_plain_size);
        const unsigned comp_size = static_cast<unsigned>(comp.size());
        const size_t offset_cont = 64 * 1024; // Matching continuation offset (0 + 64 KiB)

        wchar_t temp_dir[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_dir)) return false;
        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_dir, L"x2w", 0, temp_file)) return false;

        // File layout:
        // Offset 0: valid compressed stream (size comp_size)
        // Offset 64 KiB: non-zlib bytes (0xEE), size comp_size
        std::vector<unsigned char> file_bytes(offset_cont + comp_size, 0xEE);
        std::memcpy(file_bytes.data(), comp.data(), comp_size);

        HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD written = 0;
        WriteFile(h, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr);
        CloseHandle(h);

        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            DeleteFileW(temp_file);
            return false;
        }
        auto* file_obj = static_cast<Object*>(file_ptr);

        void* status_ptr1 = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "UndersizedStatus", nullptr, &status_ptr1) != 0 || !status_ptr1) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status1 = static_cast<Object*>(status_ptr1);

        void* status_ptr2 = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 1, "MatchingStatus", nullptr, &status_ptr2) != 0 || !status_ptr2) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* status2 = static_cast<Object*>(status_ptr2);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 16;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
            DeleteFileW(temp_file);
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        std::vector<unsigned char> dest_undersized(64 * 1024, 0xA5); // Sentinel 0xA5
        std::vector<unsigned char> dest_matching(64 * 1024, 0xB6);   // Sentinel 0xB6

        // First Submit: compressed stream into undersized destination buffer (64 KiB < 128 KiB uncompressed)
        // with NO continuation requests queued in this batch.
        unsigned char req_undersized[0x48] = {};
        *reinterpret_cast<u64*>(req_undersized + 0x00) = 1;
        *reinterpret_cast<void**>(req_undersized + 0x08) = dest_undersized.data();
        *reinterpret_cast<unsigned*>(req_undersized + 0x18) = 64 * 1024; // Undersized capacity
        *reinterpret_cast<void**>(req_undersized + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req_undersized + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req_undersized + 0x30) = comp_size;
        *reinterpret_cast<unsigned*>(req_undersized + 0x38) = static_cast<unsigned>(stream_plain_size);

        // The signal callback holds the worker at the head of the first batch. It uses static
        // storage so even a failed test can exit without abandoning a pointer to the stack.
        static SubmitGate gate;
        static void* fence_vtable[11] = {};
        fence_vtable[10] = reinterpret_cast<void*>(&hold_worker);
        gate.vtable = fence_vtable;
        gate.entered = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        gate.resume = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!gate.entered || !gate.resume) return false;
        reinterpret_cast<EnqueueSignalFn>(queue->vtable[5])(queue, &gate, 1);
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req_undersized);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status1, 0);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
        if (WaitForSingleObject(gate.entered, 3000) != WAIT_OBJECT_0) {
            SetEvent(gate.resume);
            return false;
        }

        // Immediately in second Submit: matching non-zlib request at continuation offset
        unsigned char req_matching[0x48] = {};
        *reinterpret_cast<u64*>(req_matching + 0x00) = 1; // zlib decompression requested
        *reinterpret_cast<void**>(req_matching + 0x08) = dest_matching.data();
        *reinterpret_cast<unsigned*>(req_matching + 0x18) = 64 * 1024;
        *reinterpret_cast<void**>(req_matching + 0x20) = file_obj;
        *reinterpret_cast<u64*>(req_matching + 0x28) = offset_cont; // Matching continuation offset
        *reinterpret_cast<unsigned*>(req_matching + 0x30) = comp_size;
        *reinterpret_cast<unsigned*>(req_matching + 0x38) = static_cast<unsigned>(stream_plain_size);

        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, req_matching);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status2, 0);
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
        SetEvent(gate.resume);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status1->vtable[3])(status1, 0) &&
                reinterpret_cast<IsCompleteFn>(status2->vtable[3])(status2, 0)) break;
            Sleep(10);
        }
        bool complete1 = reinterpret_cast<IsCompleteFn>(status1->vtable[3])(status1, 0);
        bool complete2 = reinterpret_cast<IsCompleteFn>(status2->vtable[3])(status2, 0);
        long hr1 = complete1 ? reinterpret_cast<GetHResultFn>(status1->vtable[4])(status1, 0) : 0;
        long hr2 = complete2 ? reinterpret_cast<GetHResultFn>(status2->vtable[4])(status2, 0) : 0;
        if (complete1 && complete2) {
            CloseHandle(gate.resume);
            CloseHandle(gate.entered);
        }

        reinterpret_cast<CloseFileFn>(file_obj->vtable[3])(file_obj);
        DeleteFileW(temp_file);

        // Both requests must fail cleanly
        if (!complete1 || hr1 == 0 || !InterlockedCompareExchange(&gate.resumed, 0, 0)) return false;
        if (!complete2 || hr2 == 0) return false;

        // Both sentinels must remain completely untouched
        for (unsigned char b : dest_undersized) {
            if (b != 0xA5) return false;
        }
        for (unsigned char b : dest_matching) {
            if (b != 0xB6) return false;
        }
    }

    // Test: BCPack DirectStorage requests decode into destination memory and malformed streams fail (Refs #1139)
    {
        // 2-block BC7 Mode 6 stream from tests/bc7_mode6_multiblock_check.py (104 bytes)
        const unsigned char bc7_mode6_packed[] = {
            0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x05, 0x05, 0x05, 0x05,
            0x05, 0x05, 0x06, 0x06, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x10,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x20, 0x02, 0x02, 0x75, 0x46,
            0x28, 0x18, 0x00, 0x00, 0x03, 0x03, 0xdf, 0x5f, 0x4b, 0x7c, 0x05, 0x21,
            0x03, 0x03, 0x50, 0x78, 0x23, 0x37, 0x04, 0x31, 0x10, 0x10, 0x10, 0x10,
            0x10, 0x10, 0x10, 0x10, 0x06, 0x05, 0x06, 0x05, 0x05, 0x05, 0x05, 0x05,
            0x10, 0x10, 0x01, 0x01, 0x01, 0x01, 0x01, 0x10, 0x20, 0x08, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0xe4, 0xdb, 0x27, 0xa4,
            0x21, 0x09, 0x02, 0x94, 0xfe, 0x6b, 0x02, 0x04, 0x31, 0x33, 0x04, 0x04,
            0xc4, 0xfb, 0x37, 0x25, 0x02, 0x02, 0xbd, 0x19, 0x50, 0x0f, 0x03, 0x41,
            0x41, 0x05, 0x09, 0x34, 0xf4, 0xcb, 0x07, 0x05, 0x02, 0x41, 0x41, 0x11
        };
        const unsigned char bc7_mode6_expected[32] = {
            0x40, 0x31, 0x72, 0x04, 0xa5, 0x68, 0x19, 0xbe,
            0x02, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
            0x40, 0x0a, 0x94, 0x37, 0x77, 0xa3, 0xc8, 0x78,
            0x11, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe
        };

        // 1-block BC7 Mode 1 stream (172 bytes) exercising 2 subsets, shared p-bit,
        // dictionary lookup and non-zero endpoint residual delta
        const unsigned char bc7_mode1_packed[172] = {
            0x15, 0x13, 0x1a, 0x18, 0x18, 0x1e, 0x10, 0x12, 0x04, 0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x03,
            0x02, 0x09, 0x20, 0x40, 0x10, 0x06, 0x01, 0x73, 0x70, 0x97, 0x1c, 0x30, 0x18, 0xe7, 0x00, 0xc9,
            0xfe, 0x4f, 0xa1, 0x88, 0x90, 0x6a, 0xeb, 0x89, 0x66, 0xd3, 0x19, 0x74, 0xb2, 0x82, 0x00, 0xc4,
            0x18, 0x71, 0x19, 0x11, 0x1a, 0x18, 0x13, 0x14, 0x13, 0x1a, 0x18, 0x10, 0x12, 0x03, 0x03, 0x03,
            0x03, 0x03, 0x03, 0x03, 0x03, 0x0a, 0x27, 0x13, 0x06, 0x14, 0x40, 0x01, 0x57, 0xbd, 0x2c, 0xd5,
            0xb4, 0x2e, 0x30, 0x00, 0x9d, 0x30, 0xac, 0xc3, 0x21, 0x6b, 0x94, 0xf8, 0x45, 0x56, 0x7f, 0x38,
            0x11, 0xa8, 0x6b, 0x00, 0x34, 0x03, 0x05, 0x03, 0x19, 0x11, 0x1a, 0x1c, 0x01, 0x8a, 0x18, 0x15,
            0x03, 0x03, 0x03, 0x7e, 0xe7, 0x8d, 0x03, 0x03, 0x11, 0x27, 0x08, 0x17, 0xee, 0x64, 0x28, 0x99,
            0x77, 0x2c, 0x07, 0x6c, 0xc5, 0xf1, 0x81, 0xd7, 0x71, 0xcd, 0x99, 0xba, 0x65, 0xfe, 0x78, 0x2c,
            0x20, 0x18, 0x70, 0x14, 0xa8, 0x90, 0x03, 0x03, 0x40, 0x03, 0x18, 0x15, 0x30, 0x03, 0x03, 0x03,
            0x90, 0xfd, 0x40, 0x08, 0x6e, 0x59, 0x30, 0xf5, 0x94, 0x38, 0x6b, 0xf4
        };
        const unsigned char bc7_mode1_expected[16] = {
            0xc2, 0x5b, 0x48, 0x7a, 0xea, 0x84, 0xa4, 0x4b, 0xfd, 0xf7, 0x15, 0x87, 0x90, 0xc4, 0x49, 0xac
        };

        // 3-block BC4, BC5, and BC3 streams generated with tests/bcpack_encoder.h (q=0, narrow=0, trim=false).
        // Each stream exercises non-zero delta (er != xr), dictionary lookup (r > i), and history lookup (r <= i),
        // with expected output bytes independently computed from bcpack_tables_bcn.inl.
        auto encode_bcn = [](const uint8_t ab[12], size_t target, const std::vector<uint16_t> syms[12],
                             std::vector<uint8_t>& out) -> bool {
            std::vector<std::vector<uint8_t>> lane_bytes(12);
            for (size_t l = 0; l < 12; ++l) {
                if (!bcpack_test::encode_fragment(ab[l], 0, 0, syms[l].data(), target, false, lane_bytes[l]))
                    return false;
            }
            out.clear();
            return bcpack_test::interleave(lane_bytes, ab, 12, target, out);
        };

        const uint8_t ab_bc4[12] = {7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5};
        const std::vector<uint16_t> bc4_syms[12] = {
            {2, 1, 4},     // delta (+1, -1, +2) -> er != xr in all 3 blocks
            {0, 1, 1},     // xr (0=zero, 1=hist[0], 1=hist[1]; er=1=dict[0], 0=zero, 3=dict[0])
            {1, 3, 5},     {2, 4, 6},     {1, 2, 3},     {4, 5, 6},
            {7, 8, 9},     {10, 11, 12},  {13, 14, 15},  {16, 17, 18},
            {19, 20, 21},  {22, 23, 24},
        };
        std::vector<uint8_t> bc4_packed;
        if (!encode_bcn(ab_bc4, 3, bc4_syms, bc4_packed)) return false;
        const unsigned char bc4_expected[24] = {
            0x01, 0x02, 0x00, 0x89, 0xfd, 0x33, 0x93, 0x51,
            0x7e, 0x82, 0x5b, 0x62, 0x25, 0xf2, 0xd6, 0xc9,
            0xff, 0x04, 0x89, 0xd2, 0x57, 0x6a, 0xd5, 0x98
        };

        const uint8_t ab_bc5[12] = {3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5};
        const std::vector<uint16_t> bc5_syms[12] = {
            {2, 0, 1, 0, 0, 1},     // delta (+2, -1, +1) -> er != xr in all 3 blocks
            {0, 0, 1, 0, 0, 1},     // xr (0=zero, 1=hist[0], 2=hist[0]; er=2=dict[1], 0=zero, 3=dict[0])
            {3, 5, 7, 9, 11, 13},   {2, 4, 6, 8, 10, 12},
            {1, 2, 3, 4, 5, 6},     {7, 8, 9, 10, 11, 12},
            {13, 14, 15, 16, 17, 18}, {19, 20, 21, 22, 23, 24},
            {25, 26, 27, 28, 29, 30}, {31, 32, 33, 34, 35, 36},
            {37, 38, 39, 40, 41, 42}, {43, 44, 45, 46, 47, 48},
        };
        std::vector<uint8_t> bc5_packed;
        if (!encode_bcn(ab_bc5, 6, bc5_syms, bc5_packed)) return false;
        const unsigned char bc5_expected[48] = {
            0x7d, 0x7b, 0x09, 0x09, 0x22, 0xad, 0x90, 0x8f,
            0x7f, 0x7f, 0x9e, 0x99, 0x69, 0x37, 0xeb, 0x8b,
            0x7c, 0x7b, 0x12, 0xa0, 0x26, 0x35, 0x7c, 0x61,
            0x83, 0x84, 0x49, 0x4a, 0xbb, 0xba, 0xc9, 0xe5,
            0x7b, 0x78, 0xf6, 0x39, 0x42, 0x62, 0xd7, 0x0f,
            0x85, 0x84, 0x48, 0xc3, 0x55, 0x50, 0xe8, 0x3d
        };

        const uint8_t ab_bc3[12] = {3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7};
        const std::vector<uint16_t> bc3_syms[12] = {
            {2, 0, 1, 0, 0, 1},     // delta (+2, -1, +1) -> er != xr in all 3 blocks
            {0, 0, 1, 0, 0, 1},     // xr (0=zero, 1=hist[0], 2=hist[0]; er=2=dict[1], 0=zero, 3=dict[0])
            {3, 5, 7, 9, 11, 13},
            {1, 2, 3, 4, 5, 6},     {7, 8, 9, 10, 11, 12},
            {13, 14, 15, 16, 17, 18}, {19, 20, 21, 22, 23, 24},
            {3, 5, 7, 9, 11, 13},   {40, 45, 50, 55, 60, 62},  // green residuals > 31 exercise 6-bit mask
            {4, 6, 8, 10, 12, 14},  {10, 20, 30, 40, 50, 60},  {70, 80, 90, 100, 110, 120},
        };
        std::vector<uint8_t> bc3_packed;
        if (!encode_bcn(ab_bc3, 6, bc3_syms, bc3_packed)) return false;
        const unsigned char bc3_expected[48] = {
            0x7e, 0x7d, 0x89, 0x1d, 0x3b, 0xd8, 0x30, 0x72,
            0xb2, 0x76, 0x13, 0x61, 0x0b, 0xfa, 0x2a, 0x06,
            0x7c, 0x7b, 0x5b, 0x92, 0x36, 0x20, 0xe1, 0xec,
            0x24, 0xe3, 0x85, 0xdc, 0x8f, 0x7f, 0x94, 0x37,
            0x79, 0x7a, 0x24, 0xe9, 0xd6, 0xe2, 0xf9, 0xd1,
            0xf6, 0x57, 0xd6, 0x47, 0xd2, 0x54, 0x6e, 0x51
        };

        // 260-block wrap streams for BC4, BC5, and BC3: block 59 sets ep[0]=10, block 187 sets ep[0]=20,
        // and block 259 looks back r=200 (259-200=59) with residual +1 -> ep[0]=11 (encoded byte0 = 11^0x80 = 0x8b).
        // A 128-entry history ring would overwrite slot 59 at block 187 and yield 0x95 instead.
        auto make_wrap_stream = [&](unsigned fmt, std::vector<uint8_t>& out) -> bool {
            const uint8_t* ab = (fmt == 4) ? ab_bc4 : (fmt == 5) ? ab_bc5 : ab_bc3;
            const size_t blocks = 260;
            const size_t target = (fmt == 4) ? blocks : 2 * blocks;
            std::vector<uint16_t> syms[12];
            for (size_t l = 0; l < 12; ++l) syms[l].assign(target, 0);
            if (fmt == 4) {
                syms[2][59] = 20;   // unzigzag(20) = +10
                syms[2][187] = 40;  // unzigzag(40) = +20
                syms[1][259] = 200; // xr = 200, er = 200 (delta = 0) -> hist[59]
                syms[2][259] = 2;   // unzigzag(2) = +1 -> ep[0] = 11
            } else {
                syms[2][2 * 59] = 20;
                syms[2][2 * 187] = 40;
                syms[1][2 * 259] = 8;      // interleave(8, 10) = 200
                syms[1][2 * 259 + 1] = 10;
                syms[2][2 * 259] = 2;
            }
            return encode_bcn(ab, target, syms, out);
        };
        std::vector<uint8_t> bc4_wrap_packed, bc5_wrap_packed, bc3_wrap_packed;
        if (!make_wrap_stream(4, bc4_wrap_packed) || !make_wrap_stream(5, bc5_wrap_packed) ||
            !make_wrap_stream(3, bc3_wrap_packed))
            return false;

        // 64-block BC1 stream from tools/_bcpack_rans/vectors/synthetic_11.json (515 bytes)
        const unsigned char bc1_packed[515] = {
            0x15, 0x14, 0x1b, 0x1b, 0xdf, 0x10, 0x13, 0x14, 0x3d, 0x3e, 0x09, 0x11, 0x03, 0x28, 0x14, 0x03,
            0x01, 0x01, 0xf5, 0x1c, 0x01, 0x10, 0x05, 0x11, 0x71, 0x17, 0xb7, 0xaf, 0x00, 0x0f, 0xfd, 0x54,
            0x6c, 0xd2, 0xea, 0xa4, 0x74, 0x68, 0xdc, 0x76, 0x6c, 0x10, 0x2b, 0xed, 0x1d, 0x65, 0x01, 0xf4,
            0x1b, 0x18, 0x91, 0xe3, 0xc1, 0x74, 0x1b, 0x18, 0x1c, 0x27, 0x03, 0x1c, 0xd5, 0x05, 0x49, 0x28,
            0x21, 0x23, 0xf3, 0xa6, 0x81, 0x06, 0x1d, 0x6b, 0xab, 0x65, 0xdb, 0x8a, 0x93, 0xdc, 0xa8, 0x3d,
            0xa8, 0x04, 0x66, 0xf5, 0x1a, 0x54, 0x3d, 0xdf, 0x1a, 0x93, 0x4a, 0x77, 0x1b, 0x36, 0x3c, 0x93,
            0x60, 0x22, 0x31, 0x2a, 0xc4, 0xa3, 0x25, 0x9f, 0x97, 0xe1, 0xfc, 0xa1, 0x79, 0x66, 0xd0, 0xbb,
            0xa1, 0x18, 0x28, 0xc4, 0xc7, 0x27, 0xae, 0x63, 0xd6, 0xdc, 0x02, 0xed, 0xbb, 0x72, 0x0a, 0x0a,
            0x2f, 0xae, 0xf4, 0x2a, 0xe4, 0xec, 0x11, 0xa7, 0x91, 0xd6, 0xc2, 0x3d, 0x1f, 0x5f, 0xe9, 0xe9,
            0xf5, 0x71, 0x34, 0xc7, 0x52, 0x1a, 0xcb, 0x80, 0xd4, 0x0a, 0x22, 0xfd, 0x09, 0x5b, 0xf9, 0x64,
            0xef, 0x54, 0x3a, 0xaa, 0x20, 0x93, 0x06, 0x39, 0x84, 0x15, 0x01, 0xff, 0x6d, 0xc7, 0x4c, 0xba,
            0x23, 0x5d, 0x2b, 0x5d, 0x25, 0x5e, 0x3d, 0xb2, 0x45, 0x19, 0x15, 0x49, 0x8b, 0xe6, 0x5e, 0xb9,
            0x08, 0x2a, 0x45, 0x0c, 0x74, 0xda, 0x1a, 0xf0, 0xd7, 0x16, 0x01, 0xbc, 0xce, 0x47, 0xf0, 0x79,
            0xf3, 0x1a, 0x58, 0xe9, 0x6a, 0xdf, 0x03, 0x77, 0x27, 0x11, 0x73, 0x85, 0x0d, 0xe4, 0x83, 0x01,
            0x00, 0x63, 0x6b, 0xbf, 0x67, 0x43, 0x85, 0xcd, 0x75, 0x22, 0x3f, 0x3b, 0x0c, 0x4d, 0x0b, 0xea,
            0xdc, 0xf7, 0x09, 0x03, 0x01, 0x0c, 0xf5, 0x0b, 0xa3, 0x16, 0x88, 0x3d, 0x93, 0x20, 0x25, 0x16,
            0xb7, 0xd8, 0x2c, 0x59, 0xe8, 0x9f, 0x04, 0x87, 0x2e, 0x7e, 0x04, 0xc1, 0x6b, 0x42, 0xfc, 0x15,
            0x97, 0xb5, 0x15, 0x13, 0x2b, 0x18, 0x60, 0x01, 0x72, 0x07, 0xc4, 0x56, 0xe0, 0x12, 0xa7, 0xf4,
            0xc7, 0x9c, 0x8e, 0x7b, 0xfb, 0xfb, 0xa0, 0x11, 0x96, 0x07, 0x7c, 0x34, 0x7d, 0x05, 0xdb, 0x97,
            0x0b, 0x32, 0x99, 0xdf, 0xf6, 0x2c, 0xde, 0xf0, 0x5d, 0x21, 0x49, 0x64, 0x3d, 0x6a, 0x7e, 0x20,
            0x6b, 0xf9, 0xb1, 0x79, 0x22, 0x20, 0xb6, 0x94, 0x64, 0x4f, 0xa3, 0xd2, 0x0d, 0x84, 0x1e, 0x44,
            0x70, 0x09, 0x51, 0x8b, 0x3e, 0x67, 0x36, 0x07, 0x7a, 0x9a, 0xbb, 0x35, 0x79, 0x94, 0x03, 0x0f,
            0x24, 0x8f, 0x77, 0x11, 0xaf, 0xcd, 0x1d, 0x05, 0x34, 0xdf, 0x39, 0x85, 0x57, 0x89, 0x86, 0xc5,
            0xa2, 0x7a, 0xd4, 0x13, 0x9a, 0xed, 0x76, 0x52, 0x6f, 0x2c, 0x25, 0x3d, 0xe0, 0x23, 0x1d, 0xf3,
            0xa2, 0xee, 0x66, 0xb8, 0x4e, 0x33, 0x3a, 0xf3, 0xca, 0xa4, 0x42, 0x2c, 0x88, 0x2b, 0xb4, 0x3f,
            0xd1, 0x91, 0x6a, 0xf1, 0x27, 0x50, 0x04, 0x5d, 0x96, 0xb6, 0xc4, 0x1f, 0xeb, 0x95, 0xca, 0xf6,
            0x03, 0x01, 0x04, 0x48, 0x11, 0x9f, 0x6d, 0x23, 0xf6, 0x1e, 0x50, 0x81, 0x7e, 0xfb, 0x91, 0x18,
            0x0e, 0xb7, 0xa8, 0x4e, 0x56, 0x09, 0xf6, 0xe8, 0x35, 0xe1, 0xf3, 0x52, 0x17, 0x0a, 0x01, 0xae,
            0x4b, 0x3d, 0xef, 0xff, 0x85, 0xa5, 0x59, 0xda, 0x2e, 0x66, 0x06, 0x6f, 0xf6, 0x30, 0x36, 0x2d,
            0x99, 0xb0, 0x5f, 0x86, 0x25, 0xd5, 0xb3, 0x46, 0x27, 0x62, 0xf2, 0xd0, 0xd2, 0x87, 0xf7, 0x96,
            0x00, 0x9c, 0x35, 0xd2, 0x2b, 0x88, 0x14, 0x85, 0x6b, 0x20, 0x88, 0x87, 0x60, 0xf0, 0xc1, 0x0c,
            0x77, 0x72, 0xd3, 0xd3, 0x20, 0x03, 0xe5, 0x3d, 0x97, 0x20, 0x46, 0x31, 0x94, 0xe0, 0xa4, 0xc0,
            0x80, 0xf6, 0x4c
        };

        // Negative control: valid BC1 stream with non-zero trailing byte
        unsigned char bc1_packed_trailing[516];
        std::memcpy(bc1_packed_trailing, bc1_packed, sizeof(bc1_packed));
        bc1_packed_trailing[515] = 0x01;

        const unsigned char bc1_expected[512] = {
            0x85, 0x31, 0x64, 0xd5, 0xaf, 0xbe, 0xff, 0xbf, 0x50, 0x84, 0xc7, 0x6b, 0xba, 0xa5, 0xbf, 0xbf,
            0x50, 0x84, 0x27, 0xc9, 0x56, 0x06, 0x5e, 0x5f, 0x25, 0x6a, 0x43, 0xaa, 0x5e, 0xee, 0xf5, 0xf5,
            0xe8, 0x72, 0xa0, 0x91, 0x5e, 0x5e, 0xfb, 0xfb, 0x2a, 0x3b, 0x83, 0xf1, 0x9a, 0xa5, 0x7a, 0xfa,
            0x81, 0x08, 0xcc, 0xa2, 0x9a, 0x9d, 0xbb, 0xff, 0x28, 0x73, 0x20, 0xbe, 0x8a, 0x8f, 0x05, 0x05,
            0x41, 0x00, 0x8c, 0xda, 0xfa, 0x5f, 0xfa, 0xdb, 0xfe, 0x8a, 0x7a, 0x5a, 0x72, 0x3a, 0xfa, 0xba,
            0x41, 0xf8, 0x8c, 0x32, 0x0f, 0xff, 0xfa, 0xba, 0xec, 0x5a, 0xa6, 0xa9, 0x6b, 0x5f, 0xaf, 0xaf,
            0x50, 0x84, 0x26, 0x89, 0x5a, 0x5f, 0x5f, 0x5f, 0x44, 0x21, 0x23, 0xb1, 0xaf, 0x5d, 0xaf, 0xab,
            0x0d, 0x6b, 0x24, 0x52, 0xaf, 0xf7, 0xbf, 0xfd, 0x0c, 0x5b, 0x26, 0x7c, 0xbe, 0x9e, 0x05, 0x05,
            0x3e, 0x83, 0x7a, 0x39, 0xd5, 0x95, 0xaa, 0x5a, 0x61, 0x10, 0x6e, 0x68, 0x5b, 0x0f, 0xff, 0xbd,
            0x8a, 0x9b, 0x65, 0x51, 0x2f, 0xaa, 0xaa, 0x5a, 0xb8, 0xc6, 0xd3, 0x4a, 0x4f, 0x45, 0xf0, 0xb0,
            0x1e, 0xb6, 0x1e, 0x6d, 0x5b, 0x5f, 0xba, 0xba, 0x50, 0x84, 0x27, 0xe4, 0xfa, 0x5a, 0xaa, 0xaf,
            0x65, 0x52, 0x45, 0xd2, 0x5e, 0x0f, 0xaf, 0xef, 0xaf, 0x73, 0x06, 0x55, 0xae, 0x8f, 0xae, 0x5f,
            0x89, 0x52, 0xa1, 0x80, 0x0f, 0x0b, 0xa5, 0xa5, 0x03, 0x21, 0xe2, 0xe8, 0x7b, 0x5f, 0x0a, 0x0a,
            0xa5, 0x39, 0x23, 0x99, 0xd0, 0xd0, 0xaa, 0xff, 0x2c, 0x8b, 0xca, 0x0a, 0x20, 0xa0, 0xaf, 0xfb,
            0x28, 0x6b, 0xc1, 0x08, 0x0a, 0x5a, 0xff, 0xaa, 0x4a, 0x4a, 0x45, 0x09, 0xba, 0xab, 0x5e, 0x5e,
            0x0c, 0x7a, 0x40, 0xaa, 0x7f, 0x7e, 0x0e, 0x0f, 0x20, 0x48, 0x9d, 0x3b, 0x5f, 0x5f, 0x00, 0x00,
            0x4b, 0xd6, 0xc4, 0xd4, 0x60, 0x60, 0xfb, 0xef, 0x45, 0x3a, 0x41, 0xa9, 0xba, 0x9f, 0xa7, 0xf7,
            0x0d, 0x5b, 0x37, 0x4a, 0xdb, 0x2b, 0xef, 0xaa, 0x20, 0x48, 0x8e, 0x0b, 0x8e, 0xaa, 0xaf, 0xaf,
            0xd8, 0xcd, 0x75, 0x6d, 0x0e, 0x0a, 0xba, 0xff, 0xbc, 0xa3, 0xf8, 0x3e, 0xbf, 0xdf, 0x5a, 0xfa,
            0x9f, 0x08, 0xd2, 0x9a, 0x25, 0x5f, 0xff, 0xba, 0x09, 0x94, 0xc3, 0x82, 0xff, 0xfe, 0xba, 0xba,
            0x21, 0x79, 0x3c, 0x31, 0xaf, 0xfe, 0xaf, 0x5a, 0xbf, 0x53, 0xe1, 0xd1, 0xbf, 0xab, 0x5f, 0x5b,
            0x26, 0x6b, 0xc3, 0xd8, 0x2a, 0xb5, 0xfb, 0xbf, 0x89, 0x42, 0x5f, 0xe1, 0xaf, 0xff, 0xb5, 0xb5,
            0x0a, 0x94, 0x8b, 0xfb, 0xa5, 0xd5, 0x0f, 0x0f, 0xa8, 0x52, 0xc7, 0x01, 0x9f, 0xab, 0x55, 0x55,
            0x29, 0xf8, 0x86, 0x7a, 0xfa, 0xfa, 0xff, 0xaa, 0x40, 0x00, 0x9a, 0x7a, 0xbe, 0xfe, 0xfa, 0xfa,
            0x54, 0x8c, 0x29, 0xc4, 0x0a, 0x0a, 0xaf, 0xab, 0x6c, 0x6b, 0x08, 0x0a, 0xa0, 0xd0, 0xf0, 0xf0,
            0x6b, 0x7c, 0xd9, 0x51, 0xa5, 0xd5, 0xf5, 0xe5, 0x82, 0x21, 0x00, 0x32, 0x00, 0x50, 0xa0, 0xa0,
            0x42, 0x19, 0x35, 0x99, 0xbf, 0xe7, 0xaf, 0xff, 0x46, 0x7b, 0x60, 0x42, 0x60, 0x45, 0x0f, 0x0f,
            0xd4, 0x31, 0x33, 0x99, 0xda, 0xaa, 0xa5, 0xaf, 0x1a, 0xd7, 0xe9, 0x80, 0xba, 0xba, 0x0a, 0x0f,
            0xc6, 0x08, 0x97, 0xe0, 0xff, 0xfa, 0xba, 0xba, 0x51, 0xd6, 0xdf, 0xec, 0x65, 0x65, 0x5f, 0x5a,
            0xbb, 0xbe, 0xda, 0x22, 0x2e, 0xaa, 0xf0, 0xf0, 0x6c, 0x63, 0x1d, 0x72, 0xeb, 0xeb, 0x5f, 0x5f,
            0xce, 0x31, 0x36, 0x91, 0x70, 0x70, 0xaa, 0xf2, 0x57, 0x8c, 0x27, 0xb4, 0x8f, 0xab, 0xf0, 0xb0,
            0x49, 0x7b, 0x6a, 0x42, 0xef, 0xfb, 0xa5, 0xb5, 0x7f, 0x6b, 0x2a, 0x3a, 0x75, 0x75, 0x5a, 0x5f
        };

        void* status_ptr = nullptr;
        if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 17, "BcpackStatus", nullptr, &status_ptr) != 0 || !status_ptr) {
            return false;
        }
        auto* status = static_cast<Object*>(status_ptr);

        void* queue_ptr = nullptr;
        unsigned char desc[0x20] = {};
        *reinterpret_cast<uint16_t*>(desc + 8) = 32;
        if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, desc, nullptr, &queue_ptr) != 0 || !queue_ptr) {
            return false;
        }
        auto* queue = static_cast<Object*>(queue_ptr);

        std::vector<unsigned char> dest_bc7_mode6(32, 0xAA);
        std::vector<unsigned char> dest_wrong_size(16, 0xBB);
        std::vector<unsigned char> dest_bc7_mode1(16, 0xAA);
        std::vector<unsigned char> dest_truncated(32, 0xCC);
        std::vector<unsigned char> dest_bc1_valid(512, 0xEE);
        std::vector<unsigned char> dest_bc1_trailing(512, 0xDD);
        std::vector<unsigned char> dest_bc6h_blank(64, 0xAA);
        std::vector<unsigned char> dest_bc4_wrong_size(8, 0x5A);
        std::vector<unsigned char> dest_bc4_non_mult(7, 0x5B);
        std::vector<unsigned char> dest_bc5_wrong_size(16, 0x5C);
        std::vector<unsigned char> dest_bc3_wrong_size(16, 0x5D);
        std::vector<unsigned char> dest_bc4(24, 0x44);
        std::vector<unsigned char> dest_bc5(48, 0x55);
        std::vector<unsigned char> dest_bc3(48, 0x33);
        std::vector<unsigned char> dest_bc4_wrap(260 * 8, 0x64);
        std::vector<unsigned char> dest_bc5_wrap(260 * 16, 0x65);
        std::vector<unsigned char> dest_bc3_wrap(260 * 16, 0x63);

        // Request 0: valid BC7 Mode 6 memory request (32 bytes destination, 2 blocks)
        unsigned char req0[0x48] = {};
        *reinterpret_cast<u64*>(req0 + 0x00) = (7ull << 1) | 0x400000;
        *reinterpret_cast<void**>(req0 + 0x08) = dest_bc7_mode6.data();
        *reinterpret_cast<u64*>(req0 + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req0 + 0x18) = 32;
        *reinterpret_cast<const unsigned char**>(req0 + 0x20) = bc7_mode6_packed;
        *reinterpret_cast<u64*>(req0 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req0 + 0x30) = sizeof(bc7_mode6_packed);
        *reinterpret_cast<unsigned*>(req0 + 0x38) = 32;

        // Request 1: wrong destination size (negative control: 16 bytes destination for 32-byte stream)
        unsigned char req1[0x48] = {};
        *reinterpret_cast<u64*>(req1 + 0x00) = (7ull << 1) | 0x400000;
        *reinterpret_cast<void**>(req1 + 0x08) = dest_wrong_size.data();
        *reinterpret_cast<u64*>(req1 + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req1 + 0x18) = 16;
        *reinterpret_cast<const unsigned char**>(req1 + 0x20) = bc7_mode6_packed;
        *reinterpret_cast<u64*>(req1 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req1 + 0x30) = sizeof(bc7_mode6_packed);
        *reinterpret_cast<unsigned*>(req1 + 0x38) = 16;

        // Request 2: valid BC7 Mode 1 memory request (subsequent valid request after rejection)
        unsigned char req2[0x48] = {};
        *reinterpret_cast<u64*>(req2 + 0x00) = (7ull << 1) | 0x400000;
        *reinterpret_cast<void**>(req2 + 0x08) = dest_bc7_mode1.data();
        *reinterpret_cast<u64*>(req2 + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req2 + 0x18) = 16;
        *reinterpret_cast<const unsigned char**>(req2 + 0x20) = bc7_mode1_packed;
        *reinterpret_cast<u64*>(req2 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req2 + 0x30) = sizeof(bc7_mode1_packed);
        *reinterpret_cast<unsigned*>(req2 + 0x38) = 16;

        // Request 3: truncated BC7 memory request (negative control: 32 bytes instead of 104)
        unsigned char req3[0x48] = {};
        *reinterpret_cast<u64*>(req3 + 0x00) = (7ull << 1) | 0x400000;
        *reinterpret_cast<void**>(req3 + 0x08) = dest_truncated.data();
        *reinterpret_cast<u64*>(req3 + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req3 + 0x18) = 32;
        *reinterpret_cast<const unsigned char**>(req3 + 0x20) = bc7_mode6_packed;
        *reinterpret_cast<u64*>(req3 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req3 + 0x30) = 32;
        *reinterpret_cast<unsigned*>(req3 + 0x38) = 32;

        // Request 4: valid BC1 memory request (64 blocks / 512 bytes, subsequent valid request)
        unsigned char req4[0x48] = {};
        *reinterpret_cast<u64*>(req4 + 0x00) = (1ull << 1) | 0x400000;
        *reinterpret_cast<void**>(req4 + 0x08) = dest_bc1_valid.data();
        *reinterpret_cast<u64*>(req4 + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req4 + 0x18) = 512;
        *reinterpret_cast<const unsigned char**>(req4 + 0x20) = bc1_packed;
        *reinterpret_cast<u64*>(req4 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req4 + 0x30) = sizeof(bc1_packed);
        *reinterpret_cast<unsigned*>(req4 + 0x38) = 512;

        // Request 5: malformed BC1 memory request with trailing non-zero byte (negative control)
        unsigned char req5[0x48] = {};
        *reinterpret_cast<u64*>(req5 + 0x00) = (1ull << 1) | 0x400000;
        *reinterpret_cast<void**>(req5 + 0x08) = dest_bc1_trailing.data();
        *reinterpret_cast<u64*>(req5 + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req5 + 0x18) = 512;
        *reinterpret_cast<const unsigned char**>(req5 + 0x20) = bc1_packed_trailing;
        *reinterpret_cast<u64*>(req5 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req5 + 0x30) = sizeof(bc1_packed_trailing);
        *reinterpret_cast<unsigned*>(req5 + 0x38) = 512;

        // Request 6: BCPack format 6 (BC6H) is not decoded yet; it must be zero-filled and counted.
        static const unsigned char bc6h_packed[] = {0x00, 0x01, 0x02, 0x03};
        unsigned char req6[0x48] = {};
        *reinterpret_cast<u64*>(req6 + 0x00) = (6ull << 1) | 0x400000;
        *reinterpret_cast<void**>(req6 + 0x08) = dest_bc6h_blank.data();
        *reinterpret_cast<u64*>(req6 + 0x10) = 0;
        *reinterpret_cast<unsigned*>(req6 + 0x18) = 64;
        *reinterpret_cast<const unsigned char**>(req6 + 0x20) = bc6h_packed;
        *reinterpret_cast<u64*>(req6 + 0x28) = 0;
        *reinterpret_cast<unsigned*>(req6 + 0x30) = sizeof(bc6h_packed);
        *reinterpret_cast<unsigned*>(req6 + 0x38) = 64;

        auto fill_bc_req = [](unsigned char* req, u64 fmt, void* dst, unsigned dst_size,
                              const unsigned char* src, unsigned src_size) {
            *reinterpret_cast<u64*>(req + 0x00) = (fmt << 1) | 0x400000;
            *reinterpret_cast<void**>(req + 0x08) = dst;
            *reinterpret_cast<u64*>(req + 0x10) = 0;
            *reinterpret_cast<unsigned*>(req + 0x18) = dst_size;
            *reinterpret_cast<const unsigned char**>(req + 0x20) = src;
            *reinterpret_cast<u64*>(req + 0x28) = 0;
            *reinterpret_cast<unsigned*>(req + 0x30) = src_size;
            *reinterpret_cast<unsigned*>(req + 0x38) = dst_size;
        };

        // Requests 7-10: wrong/unaligned destination size negative controls (BC4, BC5, BC3).
        unsigned char req7[0x48] = {};
        fill_bc_req(req7, 4, dest_bc4_wrong_size.data(), 8, bc4_packed.data(), static_cast<unsigned>(bc4_packed.size()));
        unsigned char req8[0x48] = {};
        fill_bc_req(req8, 4, dest_bc4_non_mult.data(), 7, bc4_packed.data(), static_cast<unsigned>(bc4_packed.size()));
        unsigned char req9[0x48] = {};
        fill_bc_req(req9, 5, dest_bc5_wrong_size.data(), 16, bc5_packed.data(), static_cast<unsigned>(bc5_packed.size()));
        unsigned char req10[0x48] = {};
        fill_bc_req(req10, 3, dest_bc3_wrong_size.data(), 16, bc3_packed.data(), static_cast<unsigned>(bc3_packed.size()));

        // Requests 11-13: valid 3-block BC4, BC5, and BC3 streams (non-zero delta, dict ref, history ref).
        unsigned char req11[0x48] = {};
        fill_bc_req(req11, 4, dest_bc4.data(), 24, bc4_packed.data(), static_cast<unsigned>(bc4_packed.size()));
        unsigned char req12[0x48] = {};
        fill_bc_req(req12, 5, dest_bc5.data(), 48, bc5_packed.data(), static_cast<unsigned>(bc5_packed.size()));
        unsigned char req13[0x48] = {};
        fill_bc_req(req13, 3, dest_bc3.data(), 48, bc3_packed.data(), static_cast<unsigned>(bc3_packed.size()));

        // Requests 14-16: 260-block BC4, BC5, and BC3 streams exercising 256-entry history wrap.
        unsigned char req14[0x48] = {};
        fill_bc_req(req14, 4, dest_bc4_wrap.data(), 260 * 8, bc4_wrap_packed.data(), static_cast<unsigned>(bc4_wrap_packed.size()));
        unsigned char req15[0x48] = {};
        fill_bc_req(req15, 5, dest_bc5_wrap.data(), 260 * 16, bc5_wrap_packed.data(), static_cast<unsigned>(bc5_wrap_packed.size()));
        unsigned char req16[0x48] = {};
        fill_bc_req(req16, 3, dest_bc3_wrap.data(), 260 * 16, bc3_wrap_packed.data(), static_cast<unsigned>(bc3_wrap_packed.size()));

        unsigned char* reqs[] = {
            req0, req1, req2, req3, req4, req5, req6, req7, req8, req9,
            req10, req11, req12, req13, req14, req15, req16,
        };
        for (unsigned idx = 0; idx < 17; ++idx) {
            reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, reqs[idx]);
            reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, idx);
        }
        reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

        for (int i = 0; i < 200; ++i) {
            if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 16)) break;
            Sleep(10);
        }
        for (unsigned idx = 0; idx < 17; ++idx) {
            if (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, idx)) {
                return false;
            }
        }

        long hrs[17] = {};
        for (unsigned idx = 0; idx < 17; ++idx) {
            hrs[idx] = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, idx);
        }

        if (hrs[0] != 0 || std::memcmp(dest_bc7_mode6.data(), bc7_mode6_expected, 32) != 0) return false;
        if (hrs[1] == 0) return false;
        for (unsigned char b : dest_wrong_size) {
            if (b != 0xBB) return false;
        }
        if (hrs[2] != 0 || std::memcmp(dest_bc7_mode1.data(), bc7_mode1_expected, 16) != 0) return false;
        if (hrs[3] == 0) return false;
        for (unsigned char b : dest_truncated) {
            if (b != 0xCC) return false;
        }
        if (hrs[4] != 0 || std::memcmp(dest_bc1_valid.data(), bc1_expected, 512) != 0) return false;
        if (hrs[5] == 0) return false;
        for (unsigned char b : dest_bc1_trailing) {
            if (b != 0xDD) return false;
        }
        if (hrs[6] != 0) return false;
        for (unsigned char b : dest_bc6h_blank) {
            if (b != 0) return false;
        }
        if (hrs[7] == 0) return false;
        for (unsigned char b : dest_bc4_wrong_size) {
            if (b != 0x5A) return false;
        }
        if (hrs[8] == 0) return false;
        for (unsigned char b : dest_bc4_non_mult) {
            if (b != 0x5B) return false;
        }
        if (hrs[9] == 0) return false;
        for (unsigned char b : dest_bc5_wrong_size) {
            if (b != 0x5C) return false;
        }
        if (hrs[10] == 0) return false;
        for (unsigned char b : dest_bc3_wrong_size) {
            if (b != 0x5D) return false;
        }
        if (hrs[11] != 0 || std::memcmp(dest_bc4.data(), bc4_expected, 24) != 0) return false;
        if (hrs[12] != 0 || std::memcmp(dest_bc5.data(), bc5_expected, 48) != 0) return false;
        if (hrs[13] != 0 || std::memcmp(dest_bc3.data(), bc3_expected, 48) != 0) return false;
        if (hrs[14] != 0 || dest_bc4_wrap[259 * 8] != 0x8b) return false;
        if (hrs[15] != 0 || dest_bc5_wrap[259 * 16] != 0x8b) return false;
        if (hrs[16] != 0 || dest_bc3_wrap[259 * 16] != 0x8b) return false;
    }

    return true;
}

// The Python fixture supplies absolute ranges from each synthetic file's RDS5
// directory. Keep one real queue and one open file object per input for all batches.
bool dstorage_directory_batches()
{
    struct Request { unsigned batch, file; unsigned long long offset; unsigned length; };
    std::vector<Request> requests;
    Request r{};
    while (std::scanf("%u%u%llu%u", &r.batch, &r.file, &r.offset, &r.length) == 4) {
        if (r.file >= 2 || !r.length || requests.size() >= 32) return false;
        requests.push_back(r);
    }
    if (requests.empty() || !std::feof(stdin)) return false;
    HMODULE library = LoadLibraryW(L"XCRDAPI.dll");
    if (!library) return false;
    auto factory_export = reinterpret_cast<DStorageGetFactoryFn>(GetProcAddress(library, "DStorageGetFactory"));
    void* object = nullptr;
    if (!factory_export || factory_export(nullptr, &object) || !object) return false;
    auto* factory = static_cast<Object*>(object);
    void* files[2]{};
    const wchar_t* names[]{L"small.ttpf", L"large.ttpf"};
    for (unsigned i = 0; i < 2; ++i)
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, names[i], nullptr, &files[i])) return false;
    const unsigned char queue_description[0x28] = {};
    if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, queue_description, nullptr, &object)) return false;
    auto* queue = static_cast<Object*>(object);
    const unsigned count = static_cast<unsigned>(requests.size());
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, count, "directory", nullptr, &object)) return false;
    auto* status = static_cast<Object*>(object);

    constexpr unsigned page = 4096, guard = 16;
    std::vector<std::array<unsigned char, page + 2 * guard>> output(count);
    for (unsigned i = 0; i < count; ++i) {
        const auto& request = requests[i];
        output[i].fill(0xa5);
        alignas(8) unsigned char record[0x48]{};
        const u64 fields[]{1, reinterpret_cast<u64>(output[i].data() + guard), 0, page,
                           reinterpret_cast<u64>(files[request.file]), request.offset, request.length};
        std::memcpy(record, fields, sizeof(fields));
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, record);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, i);
        if (i + 1 == count || requests[i + 1].batch != request.batch)
            reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
    }
    const auto deadline = GetTickCount64() + 5000;
    while (!reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, count - 1)) {
        if (GetTickCount64() >= deadline) return false;
        Sleep(1);
    }
    std::ofstream readback("readback.bin", std::ios::binary);
    for (unsigned i = 0; i < count; ++i) {
        const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, i);
        std::printf("%08lx\n", static_cast<unsigned long>(hr));
        readback.write(reinterpret_cast<const char*>(output[i].data()), output[i].size());
    }
    for (void* file : files)
        reinterpret_cast<CloseFileFn>(static_cast<Object*>(file)->vtable[3])(file);
    readback.close();
    return readback.good();
}

using FileInformationFn = long(__stdcall*)(void*, BY_HANDLE_FILE_INFORMATION*);

// File::GetFileInformation must return the Win32 error of its own failed call, not the error left by the
// logging that follows. Runs the same sequence with the trap-log variable unset and set, because the stand-in's
// logger calls GetEnvironmentVariableW (and CreateFileW when the variable is set), and both overwrite the
// thread's last error. A closed file object keeps its vtable and holds INVALID_HANDLE_VALUE, so the call is
// well defined and must fail with ERROR_INVALID_HANDLE.
bool dstorage_file_information_checks()
{
    HMODULE library = LoadLibraryW(L"XCRDAPI.dll");
    if (!library) return false;
    auto get_factory = reinterpret_cast<DStorageGetFactoryFn>(GetProcAddress(library, "DStorageGetFactory"));
    void* factory_ptr = nullptr;
    if (!get_factory || get_factory(nullptr, &factory_ptr) != 0 || !factory_ptr) return false;
    auto* factory = static_cast<Object*>(factory_ptr);

    wchar_t temp_dir[MAX_PATH], temp_file[MAX_PATH], log_file[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, temp_dir) || !GetTempFileNameW(temp_dir, L"x2w", 0, temp_file) ||
        !GetTempFileNameW(temp_dir, L"x2w", 0, log_file))
        return false;
    constexpr DWORD size = 1234;
    HANDLE h = CreateFileW(temp_file, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    std::vector<unsigned char> bytes(size, 0x5a);
    DWORD written = 0;
    const bool wrote = WriteFile(h, bytes.data(), size, &written, nullptr) && written == size;
    CloseHandle(h);
    if (!wrote) {
        DeleteFileW(temp_file);
        return false;
    }

    bool ok = true;
    for (const bool log_set : {false, true}) {
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_set ? log_file : nullptr);
        void* file_ptr = nullptr;
        if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, temp_file, nullptr, &file_ptr) != 0 || !file_ptr) {
            ok = false;
            break;
        }
        auto* file = static_cast<Object*>(file_ptr);
        auto information = reinterpret_cast<FileInformationFn>(file->vtable[4]);
        BY_HANDLE_FILE_INFORMATION info{};
        const long open_hr = information(file, &info);
        const bool open_ok = open_hr == S_OK && info.nFileSizeHigh == 0 && info.nFileSizeLow == size;
        reinterpret_cast<CloseFileFn>(file->vtable[3])(file);
        SetLastError(0);
        const long closed_hr = information(file, &info);
        const long expected = HRESULT_FROM_WIN32(ERROR_INVALID_HANDLE);
        const bool closed_ok = closed_hr == expected;
        std::printf("file-information log=%s open: hr=0x%08lx size=%lu %s; closed: hr=0x%08lx expected=0x%08lx %s\n",
                    log_set ? "set" : "unset", static_cast<unsigned long>(open_hr), info.nFileSizeLow,
                    open_ok ? "ok" : "MISMATCH", static_cast<unsigned long>(closed_hr),
                    static_cast<unsigned long>(expected), closed_ok ? "ok" : "MISMATCH");
        ok = ok && open_ok && closed_ok;
    }
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", nullptr);
    DeleteFileW(temp_file);
    DeleteFileW(log_file);
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    const char* which = argv[1];
    if (!std::strcmp(which, "none")) return 0;
    if (!std::strcmp(which, "chad")) return static_cast<int>(XDspActivate(0x11, 0x22, 0x33, 0x44));
    if (!std::strcmp(which, "xg_xs")) return static_cast<int>(XGSuggestSwizzleMode(0x55, 0, 0, 0));
    if (!std::strcmp(which, "physical")) {
        std::size_t count = 4;
        u64 pages[4] = {};
        if (!XMemAllocatePhysicalPages(0x20400000ull /* 64 KiB pages */, &count, pages) || count != 4) return 3;
        for (u64 page : pages)
            if (!page) return 4;
        if (!XMemFreePhysicalPages(count, pages)) return 5;
        if (XMemFreePhysicalPages(count, pages)) return 6;  // double free must be refused
        std::puts("physical ok");
        return 0;
    }
    if (!std::strcmp(which, "dstorage-directory")) return dstorage_directory_batches() ? 0 : 8;
    if (!std::strcmp(which, "dstorage-error-record")) {
        if (!dstorage_error_record_checks()) return 10;
        std::puts("dstorage-error-record ok");
        return 0;
    }
    if (!std::strcmp(which, "dstorage-pages")) {
        if (!dstorage_page_array_checks()) return 9;
        std::puts("dstorage-pages ok");
        return 0;
    }
    if (!std::strcmp(which, "dstorage-file-information")) {
        if (!dstorage_file_information_checks()) return 10;
        std::puts("dstorage-file-information ok");
        return 0;
    }
    if (!std::strcmp(which, "dstorage-status-initial")) {
        if (!dstorage_status_initial_checks()) return 10;
        std::puts("dstorage-status-initial ok");
        return 0;
    }
    if (!std::strcmp(which, "dstorage")) {
        if (!dstorage_checks()) return 7;
        std::puts("dstorage ok");
        return 0;
    }
    if (!std::strcmp(which, "store")) return x2w_store_method3_check();
    return 2;
}
