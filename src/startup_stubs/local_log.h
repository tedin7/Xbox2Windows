// Experimental (docs/startup-runtime.md): opt-in switches and logging shared by the
// permissive startup paths. Lines go to stderr and are appended to %X2W_TRAP_LOG% when set.
#pragma once

#include "startup_stubs/trap.h"

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <memory>
#include <new>
#include <type_traits>

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

// Seconds since this process was created: one time base for lines from different stand-in DLLs.
inline double process_seconds() noexcept
{
    FILETIME created{}, exited{}, kernel{}, user{}, now{};
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0;
    GetSystemTimeAsFileTime(&now);
    const auto ticks = [](const FILETIME& t) { return (std::uint64_t{t.dwHighDateTime} << 32) | t.dwLowDateTime; };
    return ticks(now) > ticks(created) ? static_cast<double>(ticks(now) - ticks(created)) / 1e7 : 0.0;
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
struct LegacyMappedCache {
    MappedSpan spans[16];
    int next;
};
inline thread_local LegacyMappedCache legacy_mapped_cache;

inline bool mapped_cache_enabled() noexcept
{
    static const bool enabled = env_on("X2W_MAPPED_SPAN_TABLE");
    return enabled;
}

inline constexpr std::size_t mapped_span_capacity = 4096;

struct MappedCache {
    MappedSpan spans[mapped_span_capacity];
    std::size_t size;

    // First span whose end is greater than address (or equal when merging).
    std::size_t lower(std::uintptr_t address, bool merge = false) const noexcept
    {
        std::size_t lo = 0, hi = size;
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (spans[mid].end < address || (!merge && spans[mid].end == address)) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    const MappedSpan* find(std::uintptr_t address) const noexcept
    {
        const auto i = lower(address);
        return i < size && spans[i].begin <= address ? &spans[i] : nullptr;
    }

    void remember(MappedSpan span, std::size_t capacity) noexcept
    {
        auto first = lower(span.begin, true);
        auto last = first;
        while (last < size && spans[last].begin <= span.end) {
            if (spans[last].begin < span.begin) span.begin = spans[last].begin;
            if (spans[last].end > span.end) span.end = spans[last].end;
            ++last;
        }
        if (first == last && size == capacity) {
            // Start a new working set when the table fills.
            size = first = last = 0;
        }
        if (last != first + 1 && last < size)
            std::memmove(spans + first + 1, spans + last, (size - last) * sizeof(MappedSpan));
        spans[first] = span;
        size = first + 1 + size - last;
    }
};

struct MappedTables {
    MappedCache mapped;
    // Trim can skip a query only for known committed, readable pages.
    // Reserved extents must be checked again in case the title committed them.
    MappedCache readable;
};

// Logs once the largest free block below 2 GB (the hole X2W_LOW_ADDRESSES needs for the title's arena).
// Walks the address space with VirtualQuery and allocates nothing.
inline void log_low_address_margin() noexcept
{
    static volatile LONG done = 0;
    if (InterlockedExchange(&done, 1)) return;
    constexpr std::uintptr_t limit = 0x80000000u;
    std::uintptr_t at = 0x10000, best = 0, best_at = 0, free_total = 0;
    unsigned regions = 0;
    while (at < limit && regions < 100000) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info)) break;
        const auto begin = reinterpret_cast<std::uintptr_t>(info.BaseAddress);
        auto end = begin + info.RegionSize;
        if (end > limit) end = limit;
        if (info.State == MEM_FREE && end > at) {
            const auto size = end - at;
            free_total += size;
            if (size > best) { best = size; best_at = at; }
        }
        if (end <= at) break;
        at = end;
        ++regions;
    }
    logf("x2w-runtime low address space: largest free block below 2 GiB %llu MiB at 0x%llx (%llu MiB free in total, %u regions)",
         static_cast<unsigned long long>(best >> 20), static_cast<unsigned long long>(best_at),
         static_cast<unsigned long long>(free_total >> 20), regions);
}

// The tables are 128 KB per thread. Under X2W_LOW_ADDRESSES that is a new low block, and a startup
// that makes one before the title's 2 GB arena can leave the arena without its hole (the title then
// writes through the NULL it got). So startup keeps the allocation-free legacy cache; the first
// translated draw arms the tables.
inline volatile LONG mapped_table_armed = 0;
inline void arm_mapped_table() noexcept
{
    if (mapped_cache_enabled()) InterlockedExchange(&mapped_table_armed, 1);
}

inline MappedTables* mapped_span_tables() noexcept
{
    if (!mapped_cache_enabled() || !InterlockedCompareExchange(&mapped_table_armed, 0, 0)) return nullptr;
    static thread_local const std::unique_ptr<MappedTables> tables(new (std::nothrow) MappedTables{});
    return tables.get();
}

inline LONG64 mapped_queries, mapped_hits, mapped_forgets;

inline bool query_mapped_span(MappedTables& tables, const void* address, MEMORY_BASIC_INFORMATION* info) noexcept
{
    static const std::size_t capacity = env_on("X2W_TEST_MAPPED_SPAN_ONE") ? 1 : mapped_span_capacity;
    InterlockedIncrement64(&mapped_queries);
    if (VirtualQuery(address, info, sizeof(*info)) != sizeof(*info)) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(info->BaseAddress);
    const auto end = begin + info->RegionSize;
    if (info->State != MEM_FREE && end > reinterpret_cast<std::uintptr_t>(address)) {
        tables.mapped.remember({begin, end}, capacity);
        if (info->State == MEM_COMMIT && !(info->Protect & (PAGE_GUARD | PAGE_NOACCESS)) && (info->Protect & 0xFF))
            tables.readable.remember({begin, end}, capacity);
    }
    return true;
}

// Query adapter for committed_extent_with. The caller must still touch the returned range.
inline bool query_readable_mapped_span(const void* address, MEMORY_BASIC_INFORMATION* info) noexcept
{
    auto* tables = mapped_span_tables();
    if (!tables) return VirtualQuery(address, info, sizeof(*info)) == sizeof(*info);
    if (const auto* span = tables->readable.find(reinterpret_cast<std::uintptr_t>(address))) {
        InterlockedIncrement64(&mapped_hits);
        info->BaseAddress = reinterpret_cast<void*>(span->begin);
        info->RegionSize = span->end - span->begin;
        info->State = MEM_COMMIT;
        info->Protect = PAGE_READONLY;
        return true;
    }
    return query_mapped_span(*tables, address, info);
}
// Forgets the spans this thread has found (a span can be freed later; a failed touch or copy calls this).
inline void forget_mapped_spans() noexcept
{
    if (auto* tables = mapped_span_tables()) {
        tables->mapped.size = tables->readable.size = 0;
        InterlockedIncrement64(&mapped_forgets);
    } else {
        legacy_mapped_cache = LegacyMappedCache{};
    }
}

// The pre-check is a debugging aid: it only pays off while a debugger would stop on every first-chance fault.
// X2W_PROBE_PRECHECK decides when it is set (0 off, anything else on); unset, the pre-check runs when a debugger is
// attached. Read once, cached. Title runs always start under debug_launch.py, where a VirtualQuery per probe can
// cost far more than the faults it avoids, so X2W_PROBE_PRECHECK=0 turns it off there. Off, probes and copies keep
// the plain touch / __try path at no extra cost.
inline bool precheck_enabled() noexcept
{
    static const bool enabled = [] {
        char value[8] = {};
        const DWORD n = GetEnvironmentVariableA("X2W_PROBE_PRECHECK", value, sizeof(value));
        if (n != 0 && n < sizeof(value)) return value[0] != '0';
        return IsDebuggerPresent() != FALSE;
    }();
    return enabled;
}

inline std::size_t mapped_prefix(std::uintptr_t start, std::size_t bytes) noexcept
{
    if (bytes == 0 || start + bytes < start || !precheck_enabled()) return bytes;
    const std::uintptr_t end = start + bytes;
    std::uintptr_t at = start;
    if (auto* tables = mapped_span_tables()) {
        while (at < end) {
            if (const auto* span = tables->mapped.find(at)) {
                InterlockedIncrement64(&mapped_hits);
                at = span->end < end ? span->end : end;
                continue;
            }
            MEMORY_BASIC_INFORMATION info{};
            if (!query_mapped_span(*tables, reinterpret_cast<const void*>(at), &info)) return bytes;
            if (info.State == MEM_FREE) return static_cast<std::size_t>(at - start);
            const auto region_end = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
            if (region_end <= at) return bytes;
            at = region_end < end ? region_end : end;
        }
        return bytes;
    }
    while (at < end) {
        bool hit = false;
        for (const MappedSpan& span : legacy_mapped_cache.spans) {
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
        legacy_mapped_cache.spans[legacy_mapped_cache.next] = {begin, begin + info.RegionSize};
        legacy_mapped_cache.next = (legacy_mapped_cache.next + 1) % 16;
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
    const std::size_t mapped = mapped_prefix(start, bytes);
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
            if (at >= window_end) {
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
    if (done < mapped) forget_mapped_spans();  // the span was cached, then freed or protected
    // A free tail only counts as an avoided fault when the mapped prefix was walked without one: a protected page
    // inside the prefix faults exactly as it did before the free tail was refused.
    else if (refused_tail) InterlockedIncrement(&faults_avoided);
    return done;
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
    } __except (any_exception || GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ||
                        GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR
                    ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        forget_mapped_spans();
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

// Runs f() outside xmem's run breaks (x2w_xmem_stable_call, X2W_XMEM_SHARED_VIEWS): inside a break the pages of
// the run being taken apart read as reserved to a VirtualQuery and make ReadProcessMemory fail, although a touch of
// them would be retried. f must be a query, or a copy into the caller's memory, that can run again; it may run
// twice. Without xmem.dll, or with one that lacks the export, f runs once, directly.
using StableCall = BOOL(WINAPI*)(void (*)(void*), void*);
inline StableCall stable_call_export() noexcept
{
    static const StableCall stable_call = [] {
        const HMODULE xmem = GetModuleHandleW(L"xmem.dll");
        return xmem ? reinterpret_cast<StableCall>(GetProcAddress(xmem, "x2w_xmem_stable_call")) : nullptr;
    }();
    return stable_call;
}
template <typename F>
inline void outside_run_breaks(F&& f) noexcept
{
    const StableCall stable_call = stable_call_export();
    using Call = std::remove_reference_t<F>;
    if (!stable_call || !stable_call([](void* context) { (*static_cast<Call*>(context))(); }, &f)) f();
}

// Readability by probing (see probe_readable); writability by one VirtualQuery (rare callers).
inline bool accessible(const void* pointer, std::size_t bytes, bool write) noexcept
{
    const auto start = reinterpret_cast<std::uintptr_t>(pointer);
    if (start < 0x10000 || start > 0x00007FFFFFFEFFFFull || bytes == 0) return false;
    if (!write) return probe_readable(start, bytes) == bytes;
    commit_lazy(pointer, bytes);
    MEMORY_BASIC_INFORMATION info{};
    SIZE_T queried = 0;
    outside_run_breaks([&] { queried = VirtualQuery(pointer, &info, sizeof(info)); });
    if (!queried || info.State != MEM_COMMIT) return false;
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

// True when every byte of [pointer, pointer + bytes) is committed and writable, with no guard or no-access page
// in between. Walks the regions one VirtualQuery at a time, so a range that spans regions of different
// protection is judged page by page. Nothing is written, touched or committed (see commit_lazy), so it is safe
// under xmem's mapping lock. An empty range is writable; a null or wrapping one is not.
inline bool committed_writable(const void* pointer, std::size_t bytes) noexcept
{
    if (bytes == 0) return true;
    std::uintptr_t cursor = reinterpret_cast<std::uintptr_t>(pointer);
    if (!pointer || bytes > ~std::uintptr_t{0} - cursor) return false;
    const std::uintptr_t end = cursor + bytes;
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

}  // namespace x2w_local
