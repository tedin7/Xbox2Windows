// Experimental (docs/startup-runtime.md): opt-in switches and logging shared by the
// permissive startup paths. Lines go to stderr and are appended to %X2W_TRAP_LOG% when set.
#pragma once

#include "startup_stubs/trap.h"
#include "startup_stubs/read_cache.h"

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <cstdio>

namespace x2w_local {

// Reads a switch without disturbing the caller's last-error value (GetEnvironmentVariable sets
// ERROR_ENVVAR_NOT_FOUND for an absent variable, and shim callers observe GetLastError).
inline bool env_on(const char* name) noexcept
{
    const DWORD saved = GetLastError();
    char value[4]{};
    const bool on = GetEnvironmentVariableA(name, value, sizeof(value)) && value[0] == '1';
    SetLastError(saved);
    return on;
}

inline bool read_cache_on() noexcept
{
    static const bool enabled = env_on("X2W_READ_CACHE");
    return enabled;
}

inline const ReadCacheApi* read_cache() noexcept
{
    if (!read_cache_on()) return nullptr;
    // xmem owns the one process-wide table. Keep its API and table alive for all shims.
    static const ReadCacheApi* api = [] {
        const DWORD error = GetLastError();
        HMODULE module = nullptr;
        if (!GetModuleHandleExW(0, L"xmem.dll", &module)) module = LoadLibraryExW(L"xmem.dll", nullptr, LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        using GetApi = const ReadCacheApi* (WINAPI*)() noexcept;
        const auto get = module ? reinterpret_cast<GetApi>(GetProcAddress(module, "x2w_xmem_read_cache")) : nullptr;
        SetLastError(error);
        return get ? get() : nullptr;
    }();
    return api;
}

inline void record_read_span(const void* address, SIZE_T bytes, DWORD protect = PAGE_READWRITE) noexcept
{
    if (const auto* cache = read_cache()) cache->record(address, bytes, protect);
}

inline void erase_read_span(const void* address, SIZE_T bytes) noexcept
{
    if (const auto* cache = read_cache()) cache->erase(address, bytes);
}

// MEM_RELEASE takes a size of zero. Determine the allocation extent before releasing
// it, on the allocator path, so adjacent allocations in a merged span survive.
inline SIZE_T read_span_release_size(const void* address) noexcept
{
    if (!read_cache_on()) return 0;
    const DWORD error = GetLastError();
    const auto begin = reinterpret_cast<std::uintptr_t>(address);
    auto at = begin;
    MEMORY_BASIC_INFORMATION info{};
    while (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) &&
           info.AllocationBase == address) {
        const auto end = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
        if (end <= at) break;
        at = end;
    }
    SetLastError(error);
    return at - begin;
}

inline BOOL release_read_span(void* address) noexcept
{
    const auto bytes = read_span_release_size(address);
    const BOOL released = VirtualFree(address, 0, MEM_RELEASE);
    if (released && bytes) erase_read_span(address, bytes);
    return released;
}

// Scratch facts describe readable commits only. They never grant permission to dereference
// title memory: keep the actual read under SEH, since an unobserved unmap can race a cache hit.
inline bool query_read_span(const void* address, MEMORY_BASIC_INFORMATION* info) noexcept
{
    if (const auto* cache = read_cache()) return cache->query(address, info);
    return VirtualQuery(address, info, sizeof(*info)) == sizeof(*info);
}

inline void logf(const char* format, ...) noexcept
{
    char line[1536];
    va_list args;
    va_start(args, format);
    int n = std::vsnprintf(line, sizeof(line) - 2, format, args);
    va_end(args);
    if (n < 0) return;
    if (n > static_cast<int>(sizeof(line)) - 3) n = sizeof(line) - 3;
    line[n++] = '\r';
    line[n++] = '\n';
    const DWORD error = GetLastError();
    x2w_trap::write_all(GetStdHandle(STD_ERROR_HANDLE), line, static_cast<DWORD>(n));
    x2w_trap::append_log(line, static_cast<DWORD>(n), false);
    SetLastError(error);
}

// True for the first `limit` calls on `counter`. The counter stops at the limit instead of counting
// every call: a gate on a hot path would otherwise pass 2^31 calls, turn negative and pass again.
inline bool first_calls(volatile LONG& counter, LONG limit) noexcept
{
    LONG seen = counter;
    while (seen < limit) {
        const LONG previous = InterlockedCompareExchange(&counter, seen + 1, seen);
        if (previous == seen) return true;
        seen = previous;
    }
    return false;
}

// First-chance exceptions stop the whole process while a debugger is attached, so a probe or a copy that is
// refused before it touches an unmapped address is worth a query. faults_avoided counts those refusals; the
// totals line of the render heartbeat prints it.
inline volatile LONG faults_avoided = 0;

// Number of leading bytes of [start, start + bytes) that are not in free address space (MEM_FREE), found with
// VirtualQuery. Only MEM_FREE is refused: a reserved or committed page may be a lazily committed xmem page that
// the fault handler commits on first touch, a guard page or no-access page keeps its handling in the callers,
// so those still go to the touch/__try path. One query walks a whole region, which on the title's
// multi-gigabyte blocks costs about a millisecond, so the spans found are cached per thread; a span freed
// after it was cached is caught by the touch/__try path as before (it only costs the fault it always cost).
struct MappedSpan {
    std::uintptr_t begin, end;
};
struct MappedCache {
    MappedSpan spans[16];
    int next;
};
inline thread_local MappedCache mapped_cache;
// Forgets the spans this thread has found (a span can be freed later; a failed touch or copy calls this).
inline void forget_mapped_spans(std::uintptr_t start, std::size_t bytes) noexcept
{
    if (const auto* cache = read_cache()) cache->erase(reinterpret_cast<const void*>(start), bytes);
    else mapped_cache = MappedCache{};
}

// Forgets every span this thread cached (the shared read cache keeps its own spans; erase them by range above).
inline void forget_mapped_spans() noexcept { mapped_cache = MappedCache{}; }

inline std::size_t mapped_prefix(std::uintptr_t start, std::size_t bytes, bool* committed = nullptr) noexcept
{
    if (bytes == 0 || start + bytes < start) return bytes;
    const std::uintptr_t end = start + bytes;
    std::uintptr_t at = start;
    if (const auto* cache = read_cache()) {
        bool all_committed = true;
        while (at < end) {
            MEMORY_BASIC_INFORMATION info{};
            if (!cache->query(reinterpret_cast<const void*>(at), &info)) return bytes;
            if (info.State == MEM_FREE) return static_cast<std::size_t>(at - start);
            all_committed = all_committed && info.State == MEM_COMMIT && !(info.Protect & PAGE_GUARD) &&
                            (info.Protect & 0xff) != PAGE_NOACCESS && (info.Protect & 0xff) != 0;
            const auto next = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
            if (next <= at) return bytes;
            at = next;
        }
        if (committed) *committed = all_committed;
        return bytes;
    }
    while (at < end) {
        bool hit = false;
        for (const MappedSpan& span : mapped_cache.spans) {
            if (span.begin <= at && at < span.end) {
                at = span.end < end ? span.end : end;
                hit = true;
                break;
            }
        }
        if (hit) continue;
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info))) return bytes;  // unknown: let the touch decide
        if (info.State == MEM_FREE) return static_cast<std::size_t>(at - start);
        const auto begin = reinterpret_cast<std::uintptr_t>(info.BaseAddress);
        mapped_cache.spans[mapped_cache.next] = {begin, begin + info.RegionSize};
        mapped_cache.next = (mapped_cache.next + 1) % 16;
        at = begin + info.RegionSize < end ? begin + info.RegionSize : end;
    }
    return bytes;
}

// With X2W_XMEM_LAZY_COMMIT, physical pages are committed on first touch (xmem_mapped_pages.cpp). A check of the
// commit state, or a write that cannot fault (kernel I/O, GPU), commits the range first. A no-op without xmem.dll
// or without the switch.
inline void commit_lazy(const void* pointer, std::size_t bytes) noexcept
{
    using CommitRange = BOOL(WINAPI*)(const void*, SIZE_T);
    static const CommitRange commit_range = [] {
        const HMODULE xmem = GetModuleHandleW(L"xmem.dll");
        return xmem ? reinterpret_cast<CommitRange>(GetProcAddress(xmem, "x2w_xmem_commit_range")) : nullptr;
    }();
    if (commit_range) commit_range(pointer, bytes);
}

// Number of contiguous readable bytes at start, up to bytes, found by touching one byte per page.
// VirtualQuery is not used for this: on the title's multi-gigabyte regions one query takes about a
// millisecond (it walks the region to size it). A bad pointer costs a first-chance exception, so
// callers on hot paths should pass values that normally are pointers. A guard page that the probe
// trips is re-armed.
// Lazily committed memory (X2W_XMEM_LAZY_COMMIT) is committed through commit_lazy one window ahead of the touches,
// so the probe does not fault once per 64 KiB slot (each fault is a stop under a debugger). The last window a
// thread committed is remembered; a slot released and mapped again simply faults once more, as before.
inline constexpr std::size_t probe_commit_window = std::size_t{1} << 20;
inline std::size_t probe_readable(std::uintptr_t start, std::size_t bytes) noexcept
{
    if (start < 0x10000 || start > 0x00007FFFFFFEFFFFull || bytes == 0) return 0;
    std::size_t done = 0;
    DWORD code = 0;
    std::uintptr_t at = start;
    // Free address space is refused without touching it; the touch below only sees mapped pages.
    bool committed = false;
    const std::size_t mapped = mapped_prefix(start, bytes, &committed);
    const bool refused_tail = mapped < bytes;
    if (mapped == 0) {
        if (refused_tail) InterlockedIncrement(&faults_avoided);
        return 0;
    }
    const std::uintptr_t end = start + mapped;
    static thread_local std::uintptr_t committed_from = 0, committed_to = 0;
    std::uintptr_t window_end = at;
    __try {
        volatile unsigned char sink = 0;
        while (at < end) {
            if (!committed && at >= window_end) {
                window_end = end - at > probe_commit_window ? at + probe_commit_window : end;
                if (at < committed_from || window_end > committed_to) {
                    commit_lazy(reinterpret_cast<const void*>(at), static_cast<std::size_t>(window_end - at));
                    committed_from = at;
                    committed_to = window_end;
                }
            }
            sink = *reinterpret_cast<const volatile unsigned char*>(at);
            const std::uintptr_t next = (at | 0xFFF) + 1;
            done = static_cast<std::size_t>((next < end ? next : end) - start);
            at = next;
        }
        (void)sink;
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        if (code == 0x80000001u) {  // STATUS_GUARD_PAGE_VIOLATION: the system dropped the guard bit
            MEMORY_BASIC_INFORMATION info{};
            DWORD old = 0;
            if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) && info.State == MEM_COMMIT)
                VirtualProtect(reinterpret_cast<void*>(at & ~std::uintptr_t{0xFFF}), 0x1000, info.Protect | PAGE_GUARD, &old);
        }
    }
    if (done < mapped) forget_mapped_spans(at, 1);  // the span was cached, then freed or protected
    // A free tail only counts as an avoided fault when the mapped prefix was walked without one: a protected page
    // inside the prefix faults exactly as it did before the free tail was refused.
    else if (refused_tail) InterlockedIncrement(&faults_avoided);
    return done;
}

inline LONG title_read_exception(const EXCEPTION_POINTERS* exception, const void* address,
                                 std::size_t bytes) noexcept
{
    const auto& record = *exception->ExceptionRecord;
    if (record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION && record.ExceptionCode != EXCEPTION_IN_PAGE_ERROR &&
        record.ExceptionCode != 0x80000001u) return EXCEPTION_CONTINUE_SEARCH;
    if (record.NumberParameters >= 2) {
        const auto fault = record.ExceptionInformation[1];
        if (fault - reinterpret_cast<std::uintptr_t>(address) < bytes) {
            erase_read_span(reinterpret_cast<const void*>(fault), 1);
            return EXCEPTION_EXECUTE_HANDLER;
        }
    }
    erase_read_span(address, bytes);
    return EXCEPTION_EXECUTE_HANDLER;
}

// For callers that already probed the range. The off path retains the original
// read. Under the cache, SEH also covers the interval between probe and actual use.
template<class Read>
inline bool guarded_title_read(const void* address, std::size_t bytes, const Read& read)
{
    if (!read_cache_on()) {
        read();
        return true;
    }
    __try {
        read();
        return true;
    } __except (title_read_exception(GetExceptionInformation(), address, bytes)) {
        return false;
    }
}

inline bool copy_probed_title(void* to, const void* from, std::size_t bytes)
{
    return guarded_title_read(from, bytes, [&] { std::memcpy(to, from, bytes); });
}

inline LONG guarded_copy_exception(const EXCEPTION_POINTERS* exception, const void* from,
                                   std::size_t bytes, bool any_exception) noexcept
{
    const DWORD code = exception->ExceptionRecord->ExceptionCode;
    if (read_cache() && title_read_exception(exception, from, bytes) == EXCEPTION_EXECUTE_HANDLER)
        return EXCEPTION_EXECUTE_HANDLER;
    if (!any_exception && code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_IN_PAGE_ERROR)
        return EXCEPTION_CONTINUE_SEARCH;
    forget_mapped_spans(reinterpret_cast<std::uintptr_t>(from), bytes);
    return EXCEPTION_EXECUTE_HANDLER;
}

// memcpy from title memory that may have been freed or unmapped since it was probed. A source that is in free
// address space is refused up front with no exception (counted in faults_avoided); everything else is copied under
// __try as the last guard. any_exception false: only an access violation or an in-page error is caught.
inline bool guarded_memcpy(void* to, const void* from, std::size_t bytes, bool any_exception) noexcept
{
    if (bytes && mapped_prefix(reinterpret_cast<std::uintptr_t>(from), bytes) < bytes) {
        InterlockedIncrement(&faults_avoided);
        return false;
    }
    __try {
        std::memcpy(to, from, bytes);
    } __except (guarded_copy_exception(GetExceptionInformation(), from, bytes, any_exception)) {
        return false;
    }
    return true;
}

// Commits the whole lazy startup block holding pointer (xmem's x2w_xmem_commit_block, X2W_XMEM_LAZY_STARTUP)
// before the GPU uses memory in it; does nothing when xmem.dll lacks the export.
inline void commit_lazy_block(const void* pointer) noexcept
{
    using CommitBlock = BOOL(WINAPI*)(const void*);
    static const CommitBlock commit_block = [] {
        const HMODULE xmem = GetModuleHandleW(L"xmem.dll");
        return xmem ? reinterpret_cast<CommitBlock>(GetProcAddress(xmem, "x2w_xmem_commit_block")) : nullptr;
    }();
    if (commit_block) commit_block(pointer);
}

// Readability by probing (see probe_readable); writability by one VirtualQuery (rare callers).
inline bool accessible(const void* pointer, std::size_t bytes, bool write) noexcept
{
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    if (start < 0x10000 || start > 0x00007FFFFFFEFFFFull || bytes == 0) return false;
    if (!write) return probe_readable(start, bytes) == bytes;
    commit_lazy(pointer, bytes);
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(pointer, &info, sizeof(info)) || info.State != MEM_COMMIT) return false;
    const DWORD protect = info.Protect & 0xFF;
    if ((info.Protect & PAGE_GUARD) || protect == PAGE_NOACCESS || protect == 0) return false;
    const bool writable = protect == PAGE_READWRITE || protect == PAGE_EXECUTE_READWRITE || protect == PAGE_WRITECOPY ||
                          protect == PAGE_EXECUTE_WRITECOPY;
    const auto begin = reinterpret_cast<std::uintptr_t>(info.BaseAddress);
    const auto end = begin + info.RegionSize;
    // start lies inside the queried region, so end - start cannot wrap; start + bytes could.
    if (bytes > end - start || (write && !writable)) return false;
    return true;
}
inline bool readable(const void* pointer, std::size_t bytes) noexcept { return accessible(pointer, bytes, false); }
inline bool writable(const void* pointer, std::size_t bytes) noexcept { return accessible(pointer, bytes, true); }

}  // namespace x2w_local
